"""Submit from the GUI (PLAN_submit T1-T4): the CLI's own ``mace submit`` /
``mace run``, previewed exactly, then run once on a click.

* **The GUI never submits on its own.** The run route needs a fresh, unused
  preview id whose inputs have not changed since (HMAC over the argv, the
  stdin, the folder, its generation, the decks and job scripts, the state of
  each deck's .out/.f9, the queued rows, the site settings and the CRYSTAL
  module they resolve to). Conflicts (a finished .out or a .f9 the job would
  replace, a deck already queued, an account that is not the user's) each
  need their own tick.
* **Preview == submit.** The preview runs the same argv in a staged copy of
  the folder with ``slurm_preview_shim`` first on PATH: read-only SLURM
  queries reach the real cluster, every state change is recorded instead.
  What the preview shows as written is the copy's before/after listing, never
  fixed text. After the real run the job scripts are compared with the
  preview's (normalized); a difference is reported.
* **GUI route == manual CLI route.** The real run is the argv a user types
  (``cd <folder> && mace submit ...``), in that folder, with the environment
  ``mace gui`` was started with (no nice, no thread caps, no GUI variables),
  so ``sbatch --export=ALL`` gives the job what a shell gives it.
* A submit cannot be stopped halfway (no cancel, no time limit); a local
  ``mace run`` stops with one SIGTERM and its own grace period.

Optional pieces of other areas are used when present (they land at the
merge): ``history`` (File History before any write), ``dbsafety`` (the
database backup), ``dbwriters`` (one GUI writer per database) and
``livefile`` (an .out still being written counts as "ongoing").
"""
from __future__ import annotations

import getpass
import hashlib
import hmac
import json
import os
import re
import secrets
import shlex
import shutil
import socket
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import clirun
from . import slurm_preview_shim as shim
from .childenv import login_env

PREVIEW_TTL_S = 900.0
AUTO_PREVIEW_MAX = 20                  # above this many decks the preview is explicit (Update Preview)
STAGE_CAP = 2 << 30
PREVIEW_TIMEOUT_S = 300.0
# sbatch --time's forms: M, M:S, H:M:S, D-H, D-H:M, D-H:M:S (a typo such as
# 0:10:000:10:00 is refused here, before mace submit records the decks)
WALLTIME_RE = re.compile(r"^(?=.{1,20}$)(?:\d+-\d+(?::\d{1,2}){0,2}|\d+(?::\d{1,2}){0,2})$")
TIMEOUT_RE = re.compile(r"^[0-9hms:]{1,12}$")
EXCLUDE_NODES_RE = re.compile(r"^[A-Za-z0-9,\-\[\]]{1,512}$")
EXCLUDE_TYPE_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
DECK_SUFFIXES = (".d12", ".d3")
SLURM_TOOLS = ("sbatch", "squeue", "scancel", "sacct", "scontrol", "sinfo", "sacctmgr")
QUEUED_STATES = ("submitted", "pending", "running", "queued")
NO_PREVIEW_TEMPLATES = ("custom", "interactive")
JOB_ID_RE = re.compile(r"Submitted batch job (\d+)")
IDS_LINE_RE = re.compile(r"(?:Job ID\(s\)|Submitted \d+ job\(s\)):\s*(.+)")
TASK_ID_RE = re.compile(r"\b(\d+(?:_\d+)?)\b")


class SubmitError(ValueError):
    """A request the dialog must correct (HTTP 400)."""


class SubmitConflict(Exception):
    """The preview no longer holds, or a tick is missing (HTTP 409)."""

    def __init__(self, message: str, **details):
        super().__init__(message)
        self.details = details


# ------------------------------------------------------------------ environment
def submit_env(base: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """The environment a shell user who typed ``mace gui`` had: the login
    environment minus the GUI's own variables (MACE_GUI_*), minus fake-SLURM
    test variables unless the test hooks are on, and, inside a SLURM
    allocation (``mace gui connect --slurm``), without the allocation's
    SLURM_* variables (SLURM_CONF stays). Nothing is added."""
    env = dict(login_env() if base is None else base)
    hooks = env.get("MACE_GUI_TEST_HOOKS") == "1" or os.environ.get("MACE_GUI_TEST_HOOKS") == "1"
    for k in list(env):
        if k.startswith("MACE_GUI_"):
            env.pop(k)
        elif k.startswith("FAKE_SLURM_") and not hooks:
            env.pop(k)
    from mace.queue.slurm_ids import outside_allocation_env
    return outside_allocation_env(env)


def local_env(base: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Run on This Node's environment: the login environment minus the GUI's
    own variables, with the allocation's SLURM_* kept (``mace run`` refuses
    to run inside a SLURM job, by its own words)."""
    env = dict(login_env() if base is None else base)
    hooks = env.get("MACE_GUI_TEST_HOOKS") == "1" or os.environ.get("MACE_GUI_TEST_HOOKS") == "1"
    return {k: v for k, v in env.items()
            if not k.startswith("MACE_GUI_") and (hooks or not k.startswith("FAKE_SLURM_"))}


def which(tool: str, env: Optional[Dict[str, str]] = None) -> Optional[str]:
    env = submit_env() if env is None else env
    return shutil.which(tool, path=env.get("PATH", os.defpath))


def mace_argv(*args: str) -> List[str]:
    """``mace_cli args`` as the CLI child; no --no-banner (it would export
    MACE_NO_BANNER into the job, which a shell submit does not)."""
    return [sys.executable, str(clirun.MACE_CLI), *args]


# ------------------------------------------------------------------ selection
@dataclass
class Selection:
    cwd: Path                          # the real folder the CLI runs in
    targets: List[str]                 # one CLI target per command, relative to cwd
    decks: List[Path]                  # every deck the commands may submit
    folder: bool                       # the target is a folder (job arrays need one)
    mixed: bool = False                # a folder with .d12 and .d3 decks


def _rel(root: Path, p: Path) -> str:
    r = p.relative_to(root).as_posix()
    return "" if r == "." else r


def path_ok(p: Path) -> bool:
    """The job scripts use $DIR/$JOB unquoted: only plain names can run."""
    from mace.submission.local_run import PATH_OK
    return bool(PATH_OK.match(str(p)))


def selection(root: Path, paths: Sequence[Path]) -> Selection:
    """The CLI's target(s) for these files/folders (resolved, inside the root)."""
    if not paths:
        raise SubmitError("Select a deck (.d12 or .d3) or a folder of decks")
    dirs = [p for p in paths if p.is_dir()]
    files = [p for p in paths if p.is_file()]
    for f in files:
        if f.suffix not in DECK_SUFFIXES:
            raise SubmitError(f"{f.name} is not a deck (.d12 or .d3)")
    if dirs and (len(dirs) > 1 or files):
        raise SubmitError("Submit one folder at a time, or decks of one folder")
    if dirs:
        d = dirs[0]
        decks = sorted(x for x in d.iterdir() if x.is_file() and x.suffix in DECK_SUFFIXES)
        if not decks:
            raise SubmitError(f"No D12 or D3 files found in {_rel(root, d) or '.'}")
        cwd = d.parent if d != root else root
        target = d.name if d != root else "."
        # a Batch Setup folder (batch_<calc>_<date>/decks) is submitted from the
        # folder that holds the batch, as commands.sh and the manual route do,
        # so tracked rows join the parents' materials.db there
        if d.name == "decks" and d.parent != root and (d.parent.name.startswith("batch_")
                                                       or (d.parent / "batch.json").is_file()):
            cwd, target = d.parent.parent, f"{d.parent.name}/decks"
        kinds = {x.suffix for x in decks}
        sel = Selection(cwd, [target], decks, True, mixed=len(kinds) > 1)
    else:
        parents = {f.parent for f in files}
        if len(parents) > 1:
            raise SubmitError("Select decks of one folder (mace submit takes one folder or one file at a time)")
        cwd = files[0].parent
        files = sorted(files)
        sel = Selection(cwd, [f.name for f in files], files, False)
    for d in sel.decks:
        if not path_ok(d):
            raise SubmitError(f"File names with spaces or shell characters cannot be submitted; rename {d.name} "
                              f"(or its folder)")
    return sel


def deck_calc_type(p: Path) -> str:
    from mace.submission.local_run import deck_calc_type as dct
    return dct(p, "prop" if p.suffix == ".d3" else "scf")


def templates_for(calc_types: Sequence[str]) -> List[Dict[str, Any]]:
    """The --progress templates the dialog offers: never custom/interactive
    (they need stdin); a template is usable when every selected deck's
    calculation type has a step in it."""
    from mace.workflow.common.constants import WORKFLOW_TEMPLATES
    from mace.workflow.manual_progress import position_in_sequence
    out = []
    for name in sorted(WORKFLOW_TEMPLATES):
        if name in NO_PREVIEW_TEMPLATES:
            continue
        seq = list(WORKFLOW_TEMPLATES[name])
        missing = sorted({c for c in calc_types if position_in_sequence(c, seq) is None})
        out.append({"name": name, "sequence": seq, "ok": not missing,
                    "why": f"does not contain {', '.join(missing)}" if missing else ""})
    return out


# ------------------------------------------------------------------ argv
@dataclass
class Command:
    argv: List[str]                    # after mace_cli
    stdin: Optional[str]
    cwd: Path
    display: str                       # what a user types


def _opt_int(v, lo, hi, name) -> Optional[int]:
    if v in (None, ""):
        return None
    try:
        n = int(v)
    except (TypeError, ValueError):
        raise SubmitError(f"{name} is a whole number from {lo} to {hi}")
    if not lo <= n <= hi:
        raise SubmitError(f"{name} is a whole number from {lo} to {hi}")
    return n


def build_commands(root: Path, where: str, sel: Selection, options: Dict[str, Any],
                   mixed_choice: Optional[str] = None, dry_run: bool = False) -> List[Command]:
    """The whitelisted argv (and stdin) for this selection; never --nosubmit."""
    options = dict(options or {})
    cwd_rel = _rel(root, sel.cwd) or "."
    show_cd = f"cd {shlex.quote(cwd_rel)} && " if cwd_rel != "." else ""
    if where == "local":
        from mace.submission import local_run
        cores = _opt_int(options.get("cores"), 1, local_run.MAX_CORES, "Cores") or local_run.DEFAULT_CORES
        args = ["run", "-n", str(cores)]
        t = str(options.get("timeout") or "").strip()
        if t:
            if not TIMEOUT_RE.match(t):
                raise SubmitError("Timeout: 90s, 30m, 1h or H:MM:SS")
            args += ["--timeout", t]
        if options.get("track"):
            args.append("--track")
        if dry_run:
            args.append("--dry-run")
        names = [d.name for d in sel.decks] if not sel.folder else [sel.targets[0]]
        args += names
        shown = [a for a in args if a != "--dry-run"]
        return [Command(args, None, sel.cwd, show_cd + "mace " + " ".join(shlex.quote(a) for a in shown))]
    if where != "slurm":
        raise SubmitError("where is slurm or local")
    flags: List[str] = []
    array = bool(options.get("array"))
    if array:
        if not sel.folder:
            raise SubmitError("Job arrays submit a whole folder. Use Batch Setup to collect these decks in their own folder.")
        flags.append("--array")
        mr = _opt_int(options.get("max_running"), 1, 10000, "Max running at once")
        if mr:
            flags += ["--max-running", str(mr)]
    elif options.get("max_running") not in (None, ""):
        raise SubmitError("Max running at once applies to job arrays")
    progress = str(options.get("progress") or "").strip()
    if progress:
        ok = {t["name"]: t for t in templates_for([deck_calc_type(d) for d in sel.decks])}
        if progress not in ok:
            raise SubmitError(f"{progress} is not a workflow template the GUI can start")
        if not ok[progress]["ok"]:
            raise SubmitError(f"{progress} {ok[progress]['why']}")
        flags += ["--progress", progress]
    elif options.get("track"):
        flags.append("--track")
    w = str(options.get("walltime") or "").strip()
    if w:
        if not WALLTIME_RE.match(w):
            raise SubmitError("Walltime: a SLURM time such as 30, 2:00:00, 1-00 or 1-00:00:00")
        flags += ["--walltime", w]
    ex = options.get("exclude") or {}
    kind = (ex.get("kind") if isinstance(ex, dict) else None) or "none"
    value = str((ex.get("value") if isinstance(ex, dict) else "") or "").strip()
    if kind == "nodes":
        if not EXCLUDE_NODES_RE.match(value):
            raise SubmitError("Exclude these nodes: node names, commas and [ranges] only")
        flags += ["--exclude-nodes", value]
    elif kind == "type":
        if not EXCLUDE_TYPE_RE.match(value):
            raise SubmitError("Exclude a node type: letters, digits, _ and - only")
        flags += ["--exclude-type", value]
    elif kind != "none":
        raise SubmitError("Exclude is none, nodes or type")
    if options.get("overwrite_sh"):
        flags.append("--overwrite-sh")
    stdin = None
    if sel.mixed:
        if str(mixed_choice) not in ("1", "2", "3"):
            raise SubmitError("This folder has D12 and D3 decks: choose which to submit")
        stdin = f"{mixed_choice}\n"
    cmds = []
    for target in sel.targets:
        args = ["submit", *flags, target]
        text = "mace " + " ".join(shlex.quote(a) for a in args)
        if stdin:
            text = f"echo {mixed_choice} | {text}"
        cmds.append(Command(args, stdin, sel.cwd, show_cd + text))
    return cmds


def submitted_decks(sel: Selection, mixed_choice: Optional[str]) -> List[Path]:
    if sel.mixed:
        c = str(mixed_choice)
        return [d for d in sel.decks if (c == "3" or (c == "1" and d.suffix == ".d12") or (c == "2" and d.suffix == ".d3"))]
    return list(sel.decks)


# ------------------------------------------------------------------ staging
STATE_DIRS = (".mace_arrays", "workflow_configs", "workflow_scripts")
MARKER_DIRS = ("workflow_outputs", "workflow_inputs", "workflow_temp")
DB_NAMES = ("materials.db", "structures.db")


def _copy_db(src: Path, dst: Path) -> None:
    """A consistent copy of a database another process may be writing:
    the SQLite backup API from a read-only connection (dbsafety's when present)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        from .dbsafety_loader import dbsafety as _dbsafety
        dbsafety = _dbsafety()
        if hasattr(dbsafety, "snapshot_backup"):
            made = dbsafety.snapshot_backup(src, dst.parent, stem=".preview_copy")
            os.replace(made, dst)
            return
    except ImportError:
        pass
    s = sqlite3.connect(f"file:{src}?mode=ro", uri=True, timeout=30)
    d = sqlite3.connect(str(dst))
    try:
        s.backup(d)
        try:
            d.execute("PRAGMA journal_mode=DELETE")
        except sqlite3.Error:
            pass
    finally:
        d.close()
        s.close()


def _inside(p: Path, root: Path) -> bool:
    try:
        p.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def stage_plan(sel: Selection, root: Path) -> Tuple[List[Tuple[Path, str]], int]:
    """[(real path, how)], bytes: what the mirror copies (how: file, stub
    (a one-byte stand-in for a wavefunction, empty when the real one is
    empty), outhead (the start of the CRYSTAL output a deck is sized from,
    when [resources] auto is on), db, tree, marker)."""
    items: List[Tuple[Path, str]] = []
    seen = set()

    def add(p: Path, how: str):
        if p not in seen and (p.exists() or how == "marker"):
            seen.add(p)
            items.append((p, how))
    folders = {sel.cwd} | {d.parent for d in sel.decks}
    for d in sel.decks:
        add(d, "file")
        add(d.with_suffix(".sh"), "file")
        add(d.with_suffix(".f9"), "stub")
        # an EXTERNAL deck's fort.34 source: the CLI warns when it is missing
        if d.suffix.lower() == ".d12":
            from mace.submission.external_geometry import deck_keyword, fort34_source
            if deck_keyword(d):
                src = fort34_source(d)
                if src is not None and _inside(src, root):
                    add(src, "stub")
    if _auto_on():
        from mace.submission.resources_suggest import find_size_out
        for d in sel.decks:
            out = find_size_out(d)
            if out is not None:
                add(out, "outhead")
    for f in folders:
        for name in STATE_DIRS:
            add(f / name, "tree")
        for name in MARKER_DIRS:
            if (f / name).is_dir():
                add(f / name, "marker")
        for name in DB_NAMES:
            add(f / name, "db")
        add(f / "crystal_job_status.json", "file")
        try:
            for ctx in sorted(f.glob(".mace_context_*")):
                if ctx.is_dir():
                    add(ctx, "tree")
        except OSError:
            pass
    size = 0
    for p, how in items:
        try:
            if how in ("file", "db"):
                size += p.stat().st_size
            elif how == "outhead":
                size += min(p.stat().st_size, OUT_HEAD_BYTES)
            elif how == "tree":
                size += sum(x.stat().st_size for x in p.rglob("*") if x.is_file() and not x.is_symlink())
        except OSError:
            pass
    return items, size


OUT_HEAD_BYTES = 4 << 20


def _auto_on() -> bool:
    """[resources] auto is on for some calculation type (submit_env's site)."""
    from mace.submission import site
    try:
        s = site.load(submit_env())
    except Exception:
        return False
    return any(s.auto_raw.values())


def _out_head(p: Path) -> bytes:
    """The start of a CRYSTAL output, to its first SCF cycle: all the sizing
    reads (resources_suggest.read_out_size)."""
    out = []
    size = 0
    with open(p, "rb") as fh:
        for line in fh:
            if line.startswith(b" CYC ") or size > OUT_HEAD_BYTES:
                break
            out.append(line)
            size += len(line)
    return b"".join(out)


def stage(sel: Selection, root: Path, mirror_root: Path, budget_free: Optional[int] = None) -> int:
    items, size = stage_plan(sel, root)
    free = budget_free if budget_free is not None else shutil.disk_usage(mirror_root.parent).free
    cap = min(STAGE_CAP, free // 2)
    if size > cap:
        raise SubmitError(f"The preview would copy {size / 1e9:.1f} GB; free space in the GUI scratch is "
                          f"{free / 1e9:.1f} GB")
    for p, how in items:
        dst = mirror_root / _rel(root, p)
        if how == "file":
            if p.is_symlink():              # never followed (it may lead out of the folder)
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
        elif how == "stub":
            dst.parent.mkdir(parents=True, exist_ok=True)
            # keep empty vs non-empty: the CLI refuses a .d3 whose .f9 is empty
            try:
                filled = p.stat().st_size > 0
            except OSError:
                filled = False
            dst.write_bytes(b"\0" if filled else b"")
        elif how == "outhead":
            if p.is_symlink():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(_out_head(p))
        elif how == "marker":
            dst.mkdir(parents=True, exist_ok=True)
        elif how == "db":
            _copy_db(p, dst)
        elif how == "tree":
            def ignore(d, names):
                return [n for n in names if n.endswith((".db", ".db-wal", ".db-shm", ".db-journal"))
                        or os.path.islink(os.path.join(d, n))]
            shutil.copytree(p, dst, symlinks=False, ignore=ignore, dirs_exist_ok=True)
            for db in p.rglob("*.db"):
                if db.is_file() and not db.is_symlink():
                    _copy_db(db, dst / db.relative_to(p))
    mirror_root.mkdir(parents=True, exist_ok=True)
    (mirror_root / _rel(root, sel.cwd)).mkdir(parents=True, exist_ok=True)
    return size


def listing(top: Path) -> Dict[str, Tuple[int, int, str]]:
    out = {}
    if not top.exists():
        return out
    for p in top.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                st = p.stat()
                h = hashlib.sha256(p.read_bytes()).hexdigest() if st.st_size <= (8 << 20) else ""
                out[p.relative_to(top).as_posix()] = (st.st_size, st.st_mtime_ns, h)
        except OSError:
            pass
    return out


# ------------------------------------------------------------------ normalization (S1)
NORM_RULES = (
    # the job script's trailing checksum (job_script_age.py) covers its lines, so it moves with the
    # install path and the ids; first, so a sum that looks like a Unix time is not half-normalized
    (re.compile(r"(?<=# MACE-generated; cksum of the lines above: )\d+ \d+"), "<CKSUM>"),
    (re.compile(r"\b9\d{8}\b"), "<JOB>"),
    (re.compile(r"(wf_[0-9a-f]{6,}|workflow_\d{8}_\d{6}(?:_[0-9a-f]{4})?|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"), "<WF>"),
    (re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?"), "<TS>"),
    (re.compile(r"(?<!\d)\d{8}_\d{6}\d*(?:_[0-9a-f]{4}(?![0-9A-Za-z]))?(?!\d)"), "<TS>"),
    (re.compile(r"\b1[6-9]\d{8}(\.\d+)?\b"), "<TS>"),
)


def normalize(text: str, mirror: Optional[Path] = None, real: Optional[Path] = None) -> str:
    if mirror is not None and real is not None:
        text = text.replace(str(mirror), str(real))
    for rx, tag in NORM_RULES:
        text = rx.sub(tag, text)
    return text


# ------------------------------------------------------------------ conflicts
_BUCKETS: Dict[Tuple[str, int, int], str] = {}


def out_bucket(p: Path) -> str:
    """The CLI's own classification of an output (completion_checker),
    cached by size and mtime; an .out still being written is "ongoing"."""
    try:
        st = p.stat()
    except OSError:
        return "missing"
    key = (str(p), st.st_size, st.st_mtime_ns)
    if key in _BUCKETS:
        return _BUCKETS[key]
    from mace.completion_checker import categorize_output_file
    bucket = categorize_output_file(p)[0]
    try:
        from mace.utils import livefile
        if bucket not in ("completed",) and livefile.is_live(p):
            bucket = "ongoing"
    except Exception:
        pass
    _BUCKETS[key] = bucket
    return bucket


def queued_rows(dbs: Sequence[Path], decks: Sequence[Path]) -> List[Dict[str, Any]]:
    """Rows of these databases (read-only) whose input file is one of the
    decks and whose status is submitted/pending/running (a pending row only
    with a job id: without one it was never queued)."""
    want = {str(d.resolve()) for d in decks} | {str(d) for d in decks}
    names = {d.name for d in decks}
    out = []
    for db in dbs:
        if not db.is_file():
            continue
        try:
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5)
        except sqlite3.Error:
            continue
        try:
            rows = con.execute("SELECT calc_id, input_file, status, slurm_job_id FROM calculations "
                               "WHERE lower(status) IN (?, ?, ?, ?)", QUEUED_STATES).fetchall()
        except sqlite3.Error:
            rows = []
        finally:
            con.close()
        for calc_id, inp, status, job in rows:
            if not inp:
                continue
            if not job and str(status or "").lower() == "pending":
                # recorded, never queued: a submission sbatch refused leaves its
                # rows like this ("recorded without a job id")
                continue
            if inp in want or (os.path.basename(inp) in names and str(Path(inp).resolve()) in want):
                out.append({"calc_id": calc_id, "input": inp, "status": status, "job_id": job, "db": str(db)})
    return out


def context_dbs(folders: Sequence[Path]) -> List[Path]:
    out = []
    for f in folders:
        p = f / "materials.db"
        if p.is_file():
            out.append(p)
        try:
            for ctx in f.glob(".mace_context_*"):
                out += [x for x in ctx.rglob("*.db") if x.is_file()]
        except OSError:
            pass
    return sorted(set(out))


def conflicts(root: Path, sel: Selection, decks: Sequence[Path], own_jobs: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    finished, wave, queued = [], [], []
    for d in decks:
        out = d.with_suffix(".out")
        if out.is_file():
            finished.append({"deck": d.name, "out_rel": _rel(root, out), "bucket": out_bucket(out),
                             "bytes": out.stat().st_size})
        f9 = d.with_suffix(".f9")
        if f9.is_file():
            wave.append({"rel": _rel(root, f9), "bytes": f9.stat().st_size})
    for j in own_jobs or []:
        wd = j.get("workdir") or ""
        for d in decks:
            if j.get("name") == d.stem and wd and os.path.realpath(wd) == os.path.realpath(str(d.parent)):
                queued.append({"deck": d.name, "job_id": j.get("id"), "source": "squeue"})
    for r in queued_rows(context_dbs({sel.cwd} | {d.parent for d in decks}), decks):
        queued.append({"deck": os.path.basename(r["input"]), "calc_id": r["calc_id"], "job_id": r.get("job_id"),
                       "source": "database"})
    return {"finished": finished, "wavefunction": wave, "queued": queued}


# ------------------------------------------------------------------ fingerprint
def _sha(p: Path) -> str:
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()
    except OSError:
        return ""


def _stamp(p: Path) -> List[Any]:
    try:
        st = p.stat()
        return [str(p), st.st_size, st.st_mtime_ns, _sha(p)]
    except OSError:
        return [str(p), None]


def _exists(p: Path) -> bool:
    try:
        return p.exists()
    except OSError:              # a folder this user cannot read (/etc/xdg/... on some clusters)
        return False


def site_inputs(env: Dict[str, str]) -> Dict[str, Any]:
    from mace.submission import site
    path = site.config_path(env)
    # every settings layer (group files too) and every variable that sets a value
    groups = [str(p) for tag, p, _ in site.config_layers(env) if tag == "group" and _exists(p)]
    names = sorted({"MACE_SITE_CONFIG", "MACE_SITE_GROUP_CONFIG", "XDG_CONFIG_DIRS", "MACE_SLURM_ACCOUNT",
                    "MACE_SLURM_PARTITION", "MACE_CRYSTAL_MODULE"}
                   | {k for k in env if k.startswith(("MACE_SLURM_", "MACE_MPI_", "MACE_CRYSTAL_", "MACE_SCRATCH_",
                                                      "MACE_MODULE_", "MACE_RUN_"))})
    return {"path": str(path) if path else None, "file": _stamp(path) if path and path.exists() else None,
            "group": [_stamp(Path(g)) for g in groups], "env": {k: env.get(k) for k in names}}


def site_resolution(env: Dict[str, str]) -> Dict[str, Any]:
    from mace.submission import site
    try:
        res = site.crystal_module(env)
        return {"modules": res.modules, "source": res.source, "root_from_path": res.root_from_path}
    except site.SiteConfigError as exc:
        return {"error": str(exc)}


def fingerprint(root_gen: Any, cmds: Sequence[Command], decks: Sequence[Path], conf: Dict[str, Any],
                env: Dict[str, str]) -> Dict[str, Any]:
    files = []
    for d in decks:
        files.append(_stamp(d))
        sh = d.with_suffix(".sh")
        files.append(_stamp(sh) if sh.exists() else [str(sh), "absent"])
    outs = sorted([c["out_rel"], c["bucket"]] for c in conf["finished"])
    waves = sorted(c["rel"] for c in conf["wavefunction"])
    queued = sorted(json.dumps(q, sort_keys=True) for q in conf["queued"])
    fp = {"root_gen": root_gen, "argv": [c.argv for c in cmds], "stdin": [c.stdin for c in cmds],
          "cwd": [str(c.cwd) for c in cmds], "files": files, "outs": outs, "waves": waves, "queued": queued,
          "site": site_inputs(env), "module": site_resolution(env)}
    # a project folder's mace-folder.toml ([submit] job size, walltime): only when there is one
    folder = {str(f) for f in (folder_file(c.cwd, env) for c in cmds) if f is not None}
    if folder:
        fp["folder"] = [_stamp(Path(f)) for f in sorted(folder)]
    return fp


def folder_file(cwd: Path, env: Dict[str, str]) -> Optional[Path]:
    """The mace-folder.toml `mace submit` reads when it runs in ``cwd``."""
    from mace.utils import folder_settings
    return folder_settings.find(cwd, env)


# ------------------------------------------------------------------ previews
@dataclass
class Preview:
    id: str
    created: float
    where: str
    cmds: List[Command]
    decks: List[Path]
    fp: Dict[str, Any]
    result: Dict[str, Any]
    needs: Dict[str, bool]
    scripts: Dict[str, str] = field(default_factory=dict)       # real <deck>.sh -> normalized preview text
    modified: List[str] = field(default_factory=list)            # real rel paths the CLI changes
    tracked: bool = False
    used: bool = False


class Store:
    """The server's previews (15 minutes, single use) and its secret."""

    def __init__(self):
        self.secret = secrets.token_bytes(32)
        self.lock = threading.Lock()
        self.items: Dict[str, Preview] = {}
        self.session_jobs: Dict[str, Dict[str, Any]] = {}     # job id -> {name, workdir, t}
        self.running = 0                                     # submit tasks in flight (shutdown waits)

    def sign(self, fp: Dict[str, Any]) -> str:
        body = json.dumps(fp, sort_keys=True, default=str).encode()
        return hmac.new(self.secret, body + secrets.token_bytes(8), hashlib.sha256).hexdigest()[:40]

    def put(self, p: Preview) -> None:
        with self.lock:
            now = time.time()
            for k in [k for k, v in self.items.items() if now - v.created > PREVIEW_TTL_S or v.used]:
                del self.items[k]
            self.items[p.id] = p

    def take(self, pid: str) -> Preview:
        with self.lock:
            p = self.items.get(str(pid or ""))
            if p is None:
                raise SubmitConflict("No such preview; preview again")
            if p.used:
                raise SubmitConflict("This preview was already used; preview again")
            if time.time() - p.created > PREVIEW_TTL_S:
                raise SubmitConflict("This preview has expired; preview again")
            p.used = True
            return p


def _resources(script_text: str) -> Dict[str, Any]:
    from mace.submission.job_array import request_of
    req = request_of(script_text)
    return {"ntasks": req.get("--ntasks"), "mem_per_cpu": req.get("--mem-per-cpu"), "mem": req.get("--mem"),
            "walltime": req.get("--time"), "account": req.get("--account"), "partition": req.get("--partition"),
            "nodes": req.get("--nodes"), "exclude": req.get("--exclude"), "array": req.get("--array"),
            "qos": req.get("--qos"), "constraint": req.get("--constraint")}


def _module_lines(script_text: str) -> List[str]:
    return [l.strip() for l in script_text.splitlines()
            if l.strip().startswith("module load ") and "crystal" in l.lower()]


def _cli_warnings(log: str) -> List[str]:
    out = []
    for line in log.splitlines():
        s = line.strip()
        if s.lower().startswith(("warning", "⚠")) or "warning:" in s.lower():
            if s not in out:
                out.append(s)
    return out[:20]


def preview(root: Path, root_gen: Any, where: str, sel: Selection, options: Dict[str, Any],
            mixed_choice: Optional[str], work: Path, ctx=None, own_jobs=None, network: bool = False,
            store: Optional[Store] = None) -> Tuple[Preview, Dict[str, Any]]:
    """Run the commands in a staged mirror (``work``) and describe them."""
    env = submit_env()
    real_cmds = build_commands(root, where, sel, options, mixed_choice)
    decks = submitted_decks(sel, mixed_choice) if where == "slurm" else list(sel.decks)
    warnings: List[str] = []
    if network:
        warnings.append("Submitting is turned off while MACE GUI is open to the network")
    conf = conflicts(root, sel, decks, own_jobs) if where == "slurm" else {"finished": [], "wavefunction": [], "queued": []}
    fp = fingerprint(root_gen, real_cmds, decks, conf, env)
    if "error" in fp["module"]:
        raise SubmitError(fp["module"]["error"])
    mirror = work / "mirror"
    mirror.mkdir(parents=True, exist_ok=True)
    result: Dict[str, Any] = {"where": where, "commands": [c.display for c in real_cmds],
                              "command": " ; ".join(c.display for c in real_cmds),
                              "argv": [["mace", *c.argv] for c in real_cmds],
                              "stdin": real_cmds[0].stdin if real_cmds else None,
                              "cwd_rel": _rel(root, sel.cwd), "decks": [_rel(root, d) for d in decks],
                              "conflicts": conf, "warnings": warnings, "mixed": sel.mixed,
                              "templates": templates_for(sorted({deck_calc_type(d) for d in decks})) if where == "slurm" else [],
                              "existing_scripts": [_rel(root, d.with_suffix(".sh")) for d in decks if d.with_suffix(".sh").exists()],
                              "n_decks": len(decks)}
    if where == "local":
        # mace run --dry-run on the selected decks, in the real folder: it writes nothing (spike S3)
        cmds = build_commands(root, where, sel, options, mixed_choice, dry_run=True)
        r = clirun.run(mace_argv(*cmds[0].argv), sel.cwd, ctx, timeout=PREVIEW_TIMEOUT_S, env=local_env())
        log = r.log.replace(str(root) + os.sep, "")
        result.update(code=r.returncode, cli_log=log[-8000:], writes=[], job_writes=[], resources=[],
                      arrays=[], refused=(r.stderr.strip() if r.returncode else ""),
                      run_writes=sorted({_rel(root, x.parent / ".mace_local") + "/" for x in decks})
                      + [_rel(root, x.with_suffix(".out")) for x in decks])
        p = Preview(id="", created=time.time(), where=where, cmds=real_cmds, decks=decks, fp=fp, result=result,
                    needs={})
        return p, result
    # --- SLURM: the staged mirror + the recording shim
    staged = stage(sel, root, mirror)
    mcwd = mirror / _rel(root, sel.cwd)
    shim_dir = work / "shim"
    log_file = work / "shim.log"
    real = {t: which(t, env) for t in shim.TOOLS}
    shim.install(shim_dir, real, log_file)
    if not real.get("sbatch"):
        warnings.append("No SLURM here: the preview uses default cluster limits")
    penv = dict(env)
    penv["PATH"] = f"{shim_dir}{os.pathsep}{env.get('PATH', '')}"
    # the staged copy reads the real folder's mace-folder.toml, as the real run will (or none)
    from mace.utils import folder_settings
    ff = folder_file(sel.cwd, env)
    penv[folder_settings.ENV] = str(ff) if ff is not None else "none"
    before = listing(mirror)
    logs, codes = [], []
    for c in real_cmds:
        r = clirun.run(mace_argv(*c.argv), mcwd, ctx, timeout=PREVIEW_TIMEOUT_S, stdin_text=c.stdin, env=penv)
        logs.append(r.log)
        codes.append(r.returncode)
    after = listing(mirror)
    calls = [json.loads(l) for l in log_file.read_text().splitlines() if l.strip()] if log_file.exists() else []
    if any(c.get("no_slurm") for c in calls) and "No SLURM here: the preview uses default cluster limits" not in warnings:
        warnings.append("No SLURM here: the preview uses default cluster limits")

    def unmirror(argv):
        return [a.replace(str(mirror), str(root)) for a in argv]
    for c in calls:
        c["argv"] = unmirror(c["argv"])
        c["cwd"] = c.get("cwd", "").replace(str(mirror), str(root))
    writes = sorted(k for k in after if k not in before or after[k][2] != before[k][2] or after[k][0] != before[k][0])
    modified = sorted(k for k in writes if k in before)
    log = "\n".join(logs)
    log = log.replace(str(mirror) + os.sep, "").replace(str(mirror), ".")
    sbatches = [c for c in calls if c["tool"] == "sbatch" and c.get("intercepted")]
    resources, arrays, singles, scripts = [], [], [], {}
    from mace.submission.job_array import read_manifest
    for c in sbatches:
        text = c.get("script_text") or ""
        spath = Path(c.get("script") or "")
        try:
            srel = spath.relative_to(mirror).as_posix()
        except ValueError:
            srel = spath.name
        res = _resources(text)
        if res.get("array"):
            members = []
            for m in (spath.parent.glob("*manifest*") if spath.parent.exists() else []):
                try:
                    members = read_manifest(m)
                    break
                except OSError:
                    pass
            names = [m["job"] for m in members]
            for m in members:
                mp = Path(m["script"])
                try:
                    scripts[Path(m["dir"]).relative_to(mirror).as_posix() + f"/{m['job']}.sh"] = normalize(
                        mp.read_text(), mirror, root)
                except (OSError, ValueError):
                    pass
            arrays.append({"wrapper": srel, "decks": names, "request": res, "throttle": res.get("array"),
                           "tasks": len(names), "argv": c["argv"]})
            for n in names:
                resources.append({"deck": n, "script": srel, **res, "array": True,
                                  "reused_script": False, "modules": _module_lines(text)})
        else:
            deck = spath.stem
            singles.append({"deck": deck, "script": srel, "argv": c["argv"]})
            existed = (root / srel).exists()
            resources.append({"deck": deck, "script": srel, **res,
                              "reused_script": bool(existed and not options.get("overwrite_sh")),
                              "modules": _module_lines(text)})
            scripts[srel] = normalize(text, mirror, root)
    job_writes = []
    for d in decks:
        job_writes += [_rel(root, d.with_suffix(".out")), _rel(root, d.with_suffix(".f9"))]
    intercepted = [c["argv"] for c in calls if c.get("intercepted")]
    passed = [c["argv"] for c in calls if c.get("passed")]
    warnings += [w for w in _cli_warnings(log) if w not in warnings]
    tracked = bool(options.get("track") or options.get("progress"))
    result.update(code=max(codes) if codes else 0, cli_log=log[-8000:], writes=writes, modified=modified,
                  job_writes=job_writes, resources=resources, arrays=arrays, singles=singles,
                  staged_bytes=staged, intercepted=intercepted, passed=passed, tracked=tracked,
                  n_jobs=len(singles) + sum(a["tasks"] for a in arrays),
                  # the size-aware suggestion per deck, and whether [resources] auto applied it
                  suggestions=suggestions(root, decks, env))
    if tracked and (sel.cwd / "materials.db").is_file():
        result["writes"] = sorted(set(result["writes"]) | {_rel(root, sel.cwd / "backups") + "/materials_backup_<TS>.db"})
    needs = {"overwrite": bool(conf["finished"] or conf["wavefunction"]), "duplicate": bool(conf["queued"])}
    p = Preview(id="", created=time.time(), where=where, cmds=real_cmds, decks=decks, fp=fp, result=result,
                needs=needs, scripts=scripts, modified=modified, tracked=tracked)
    return p, result


def suggestions(root: Path, decks: Sequence[Path], env: Optional[Dict[str, str]] = None,
                calc_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """The size-aware job sizes of ``decks`` (mace site suggest: the same
    function, site.suggest_decks), deck paths relative to the root."""
    from mace.submission import site
    try:
        out = site.suggest_decks(list(decks), env=submit_env() if env is None else env, calc_type=calc_type)
    except site.SiteConfigError as exc:
        return [{"deck": _rel(root, Path(d)), "error": str(exc), "suggestion": None, "applies": None,
                 "auto": False, "calc_type": None} for d in decks]
    for e in out:
        try:
            e["deck"] = _rel(root, Path(e["deck"]))
        except ValueError:
            pass
    return out


def finalize(store: Store, p: Preview, account_check: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Sign and keep the preview; the answer for the dialog."""
    if account_check and account_check.get("status") == "not_member" and p.where == "slurm":
        p.needs["account"] = True
        p.result["account_check"] = account_check
    p.id = store.sign(p.fp)
    store.put(p)
    p.result["preview_id"] = p.id
    p.result["needs"] = dict(p.needs)
    return p.result


# ------------------------------------------------------------------ run
def check_run(store: Store, pid: str, root: Path, root_gen: Any, ticks: Dict[str, bool], own_jobs=None) -> Preview:
    """The preview, if it still holds and every needed tick was given (409 otherwise)."""
    p = store.take(pid)
    if p.fp.get("root_gen") != root_gen:
        raise SubmitConflict("The folder changed; reload")
    env = submit_env()
    sel_cwd = p.cmds[0].cwd if p.cmds else root
    if p.where == "slurm":
        conf = conflicts(root, Selection(sel_cwd, [], p.decks, False), p.decks, own_jobs)
    else:
        conf = {"finished": [], "wavefunction": [], "queued": []}
    now = fingerprint(root_gen, p.cmds, p.decks, conf, env)
    if now["files"] != p.fp["files"] or now["argv"] != p.fp["argv"]:
        raise SubmitConflict("The files changed since the preview; preview again")
    if now["site"] != p.fp["site"]:
        raise SubmitConflict("Site settings changed since the preview; preview again")
    if now.get("folder") != p.fp.get("folder"):
        raise SubmitConflict("The folder settings (mace-folder.toml) changed since the preview; preview again")
    if now["module"] != p.fp["module"]:
        old = " ".join(p.fp["module"].get("modules") or []) or "?"
        new = " ".join(now["module"].get("modules") or []) or "?"
        raise SubmitConflict(f"The CRYSTAL module MACE would load changed since the preview ({old} → {new}); preview again")
    if now["outs"] != p.fp["outs"] or now["waves"] != p.fp["waves"] or now["queued"] != p.fp["queued"]:
        raise SubmitConflict("The outputs or the queue changed since the preview; preview again")
    if p.needs.get("overwrite") and not ticks.get("confirm_overwrite"):
        raise SubmitConflict("Tick “Overwrite finished results” to submit these decks", need="overwrite")
    if p.needs.get("duplicate") and not ticks.get("confirm_duplicate"):
        raise SubmitConflict("Tick “Submit again although … already queued” to submit these decks", need="duplicate")
    if p.needs.get("account") and not ticks.get("confirm_account"):
        raise SubmitConflict("Tick “Submit with this account anyway”, or set your account first", need="account")
    if p.needs.get("limits") and not ticks.get("confirm_limits"):
        raise SubmitConflict("Tick “Submit anyway” (SLURM will refuse some jobs)", need="limits")
    if p.result.get("code"):
        raise SubmitConflict("The preview failed; nothing can be submitted (see its log)")
    return p


def _keep_history(root: Path, rels: Sequence[str], reason: str, kind: str = "input") -> List[str]:
    """File History copies before the CLI changes these files (C20); raises
    SubmitConflict when the store cannot be written. Without the history
    module (before the merge) nothing is kept here."""
    try:
        from . import history
    except ImportError:
        return []
    kept = []
    for rel in rels:
        p = root / rel
        if not p.is_file():
            continue
        try:
            e = history.keep(p, reason, route="submit", kind=kind)
        except getattr(history, "HistoryUnavailable", OSError) as exc:
            raise SubmitConflict(f"MACE could not keep the previous version of {rel} ({exc}); nothing was submitted.")
        if e is not None:
            kept.append(rel)
    return kept


def _backup_db(db: Path) -> Optional[Path]:
    """One backup of a tracked submit's database before the CLI writes it (C20.6)."""
    if not db.is_file():
        return None
    try:
        from .dbsafety_loader import dbsafety as _dbsafety
        dbsafety = _dbsafety()
    except (ImportError, OSError):
        dbsafety = None
    if dbsafety is not None and hasattr(dbsafety, "snapshot_backup"):
        try:
            made = dbsafety.snapshot_backup(db, db.parent / "backups", "materials_backup")
        except getattr(dbsafety, "BackupTimeout", TimeoutError) as exc:
            raise SubmitConflict(str(exc))
        if hasattr(dbsafety, "prune_backups"):
            dbsafety.prune_backups(db.parent / "backups", keep=3, out=None)
        return made
    from datetime import datetime
    d = db.parent / "backups"
    d.mkdir(exist_ok=True)
    dst = d / f"materials_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
    n = 2
    while dst.exists():
        dst = d / f"materials_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{n}.db"
        n += 1
    _copy_db(db, dst)
    olds = sorted(x for x in d.glob("materials_backup_*.db"))
    for x in olds[:-3]:
        try:
            x.unlink()
        except OSError:
            pass
    return dst


class _NoLock:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def _writer_lock(db: Path, tracked: bool):
    if not tracked:
        return _NoLock()
    try:
        from . import dbwriters
    except ImportError:
        return _NoLock()
    try:
        return dbwriters.holding(db, "Submit (mace submit --track)")
    except Exception as exc:          # WriterBusy
        raise SubmitConflict(str(exc))


def _file_names(top: Path) -> set:
    """The relative names of the regular files below ``top`` (no symlink
    followed, nothing read or hashed: a submit may run in a large folder)."""
    out = set()
    for d, dirs, files in os.walk(top):
        dirs[:] = [x for x in dirs if not os.path.islink(os.path.join(d, x))]
        for f in files:
            full = os.path.join(d, f)
            if not os.path.islink(full):
                out.add(os.path.relpath(full, top).replace(os.sep, "/"))
    return out


def array_dirs(cmds: Sequence[Command]) -> List[Path]:
    """The folders whose .mace_arrays/ a submit of ``cmds`` can write (each
    command's cwd and its folder target): never a walk of the whole tree."""
    out: List[Path] = []
    for c in cmds:
        for d in (c.cwd, c.cwd / c.argv[-1] if c.argv else c.cwd):
            if d.is_dir() and d not in out:
                out.append(d)
    return out


def job_ids_of(log: str, dirs, since: float) -> List[str]:
    ids: List[str] = []
    for m in JOB_ID_RE.finditer(log):
        if m.group(1) not in ids:
            ids.append(m.group(1))
    for m in IDS_LINE_RE.finditer(log):
        for t in TASK_ID_RE.findall(m.group(1)):
            if t not in ids:
                ids.append(t)
    # job arrays: every task of each array this run submitted
    dirs = [dirs] if isinstance(dirs, Path) else list(dirs)
    found = [f for d in dirs for f in d.glob(".mace_arrays/*/submitted.json")]
    for f in found[:200]:
        try:
            if f.stat().st_mtime < since - 1:
                continue
            data = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        a = data.get("array_job_id")
        if a:
            ids = [i for i in ids if i != str(a)]
            for m in data.get("members") or []:
                t = f"{a}_{m.get('task')}"
                if t not in ids:
                    ids.append(t)
    return ids


def run(store: Store, p: Preview, root: Path, ctx=None, problems=None, on_output=None) -> Dict[str, Any]:
    """The real run of a checked preview (submit: no cancel, no time limit)."""
    env = submit_env()
    t0 = time.time()
    if p.where == "local":
        from mace.submission import local_run
        grace = local_run.KILL_GRACE + 10
        killed = []

        def stopper(proc):
            return clirun.stop_gracefully(proc, grace, on_kill=lambda s: killed.append(s))
        n = max(1, len(p.decks))
        try:
            tmo = local_run.parse_duration(str(_opt_of(p.cmds[0].argv, "--timeout") or "30m"))
        except Exception:
            tmo = local_run.DEFAULT_TIMEOUT
        try:
            r = clirun.run(mace_argv(*p.cmds[0].argv), p.cmds[0].cwd, ctx, timeout=n * tmo + 300, env=local_env(),
                           nice=False, stopper=stopper, on_output=on_output)
        finally:
            if killed and problems:
                problems("mace run did not stop in %d s; check for leftover CRYSTAL processes with: "
                         "pgrep -u $USER -f Pcrystal" % grace)
        log = r.log.replace(str(root) + os.sep, "")
        return {"code": r.returncode, "log": log[-8000:], "job_ids": [], "script_drift": []}
    # SLURM
    store.running += 1
    try:
        kept = _keep_history(root, p.modified, f"Submit: {p.result.get('command', '')}")
        if p.needs.get("overwrite"):
            outs = [c["out_rel"] for c in p.result["conflicts"]["finished"]] + \
                   [c["rel"] for c in p.result["conflicts"]["wavefunction"]]
            kept += _keep_history(root, outs, "Submit overwrote this finished result", kind="output")
        cwd = p.cmds[0].cwd
        db = cwd / "materials.db"
        backups = []
        logs, codes = [], []
        with _writer_lock(db, p.tracked):
            if p.tracked:
                b = _backup_db(db)
                if b:
                    backups.append(str(Path(b).relative_to(root)))
            try:
                from . import history as _history          # File History (area C), when present
            except ImportError:
                _history = None
            before = _file_names(cwd) if _history is not None else set()
            for c in p.cmds:
                r = clirun.run(mace_argv(*c.argv), c.cwd, ctx, timeout=None, stdin_text=c.stdin, env=env,
                               nice=False, cancelable=False, on_output=on_output)
                logs.append(r.log)
                codes.append(r.returncode)
        log = "\n".join(logs)
        ids = job_ids_of(log, array_dirs(p.cmds), t0)
        for j in ids:
            store.session_jobs[j] = {"t": time.time(), "workdir": str(cwd)}
        # File History lists the files the CLI created (restore = delete them while unchanged)
        if _history is not None:
            for rel in sorted(_file_names(cwd) - before)[:500]:
                _history.record_created(cwd / rel, f"Submit: {p.result.get('command', '')}", route="submit")
        # the job scripts against the preview's (cluster state may have changed in between)
        drift = []
        for rel, text in p.scripts.items():
            real = root / rel
            try:
                now = normalize(real.read_text())
            except OSError:
                continue
            if now != text:
                import difflib
                diff = "".join(difflib.unified_diff(text.splitlines(True), now.splitlines(True),
                                                    "preview/" + rel, rel, n=1))
                drift.append({"script": rel, "diff": diff[-4000:]})
        code = max(codes) if codes else 0
        if code and problems:
            problems("mace submit exited with %d:\n%s" % (code, "\n".join(log.splitlines()[-40:])))
        return {"code": code, "log": log.replace(str(root) + os.sep, "")[-8000:], "job_ids": ids,
                "script_drift": drift, "backups": backups, "history": kept}
    finally:
        store.running -= 1


def _opt_of(argv: Sequence[str], name: str) -> Optional[str]:
    if name in argv:
        i = list(argv).index(name)
        if i + 1 < len(argv):
            return argv[i + 1]
    return None


# ------------------------------------------------------------------ capabilities
def capabilities(network: bool) -> Dict[str, Any]:
    env = submit_env()
    from mace.queue.slurm_ids import own_job_ids
    from mace.submission.local_run import gateway_reason
    try:
        user = getpass.getuser()
    except Exception:
        user = ""
    tools = {t: bool(which(t, env)) for t in ("sbatch", "squeue", "scancel", "sacct")}
    le = login_env()
    return {**tools, "in_slurm_job": bool(le.get("SLURM_JOB_ID")), "slurm_job_id": le.get("SLURM_JOB_ID") or None,
            "own_job_ids": sorted(own_job_ids(le)), "user": user,
            "local_run_hint": gateway_reason(le, socket.gethostname()), "network": network,
            "auto_preview_max": AUTO_PREVIEW_MAX}
