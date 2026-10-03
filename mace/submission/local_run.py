"""Run CRYSTAL decks on the node you are on, without SLURM (`mace run`).

For the calculations that take seconds or minutes - a MATDUMP, BAND/DOSS/POTC
properties, a small SP, a quick rerun - on an HPCC development node.

Each deck runs through the job script `mace submit` would submit: the shipped
generator (submitcrystal23.sh for a .d12, submit_prop.sh for a .d3) writes it
into <deck dir>/.mace_local/<job>.sh, and it runs under `bash --login` the way
SLURM runs it, with SLURM_SUBMIT_DIR=<deck dir> and SLURM_NTASKS=<ranks>. So
the staging (INPUT, fort.9/fort.20/fort.34), the per-job scratch directory
($SCRATCH/crys23/<job>, crys23/prop/<job>) and the copy-back names (.out, .f9,
.BAND.DAT, .DOSS.DAT, .f25, ...) are the ones a SLURM job has; there is no
second copy of them here. MACE_LOCAL_RUN=1 makes the script skip its completion
callback (it would submit follow-up jobs from the dev node).

What this module adds around the script:

* guards before anything is written: not on the login gateway, not inside a
  SLURM job, CRYSTAL loadable, cores and timeout below the dev-node limits
  (every process there is stopped after 2 h of CPU time), deck names the
  script can use;
* the order: every .d12 first, then each .d3 after the SCF that feeds it, with
  the SCF's fresh <scf>.f9 copied to <d3>.f9 (the copy opt2d3 makes);
* supervision: the script runs in a session of its own. Ctrl-C, a hang-up or
  the timeout stops the whole session (SIGTERM to the process group, then
  SIGKILL to anything left in the session), so no CRYSTAL rank is left behind
  and no half-written .f9 is copied back;
* the verdict, read from this run's <job>.out with MACE's completion check.

The job's own log (what SLURM would write to <job>-<jobid>.o) is
<job>-local.o; "local" is no number, so nothing reads it as a SLURM job id.
"""

import argparse
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

SUBMISSION_DIR = Path(__file__).resolve().parent
GENERATORS = {"scf": SUBMISSION_DIR / "submitcrystal23.sh",
              "prop": SUBMISSION_DIR / "submit_prop.sh"}
SCRATCH_SUBDIR = {"scf": "crys23", "prop": "crys23/prop"}
CRYSTAL_MODULE = "CRYSTAL/23-intel-2023a"
BINARIES = {"scf": "Pcrystal", "prop": "Pproperties"}
LOCAL_MARKER = "# LOCAL RUN."

DEFAULT_CORES = 4
MAX_CORES = 16
POLITE_CORES = 8
DEFAULT_TIMEOUT = 30 * 60
MAX_TIMEOUT = 115 * 60          # HPCC dev nodes stop a process at 2 h of CPU time
CPU_LIMIT_MARGIN = 5 * 60
KILL_GRACE = 15                 # seconds between SIGTERM and SIGKILL

# What the job script can take: it uses $DIR/$JOB unquoted, as under SLURM.
# '+' is in MACE's own names (<base>_charge+potential.d3).
PATH_OK = re.compile(r"^[A-Za-z0-9._+,=^/-]+$")
PROP_TYPE_RE = re.compile(r"(charge\+potential|transport|potential|charge|band|doss)\d*",
                          re.IGNORECASE)
PARENT_SUFFIXES = ("_opt", "_sp", "_OPT", "_SP")      # what opt2d3 strips
# Point the queue manager at a workflow context database or plan; never
# carried into a local run.
WORKFLOW_ENV = ("MACE_WORKFLOW_ID", "MACE_CONTEXT_DIR", "MACE_ISOLATION_MODE", "MACE_DB_PATH")

EXIT_OK, EXIT_FAILED, EXIT_REFUSED, EXIT_TIMEOUT = 0, 1, 2, 124


class Refusal(Exception):
    """Nothing was run: a guard or the command line said no."""


def say(*parts, err: bool = False) -> None:
    """Print, and survive a terminal that has gone away (after a hang-up the
    cleanup must still reach the SIGKILL sweep)."""
    try:
        print(*parts, file=sys.stderr if err else sys.stdout, flush=True)
    except (OSError, ValueError):
        pass


def fmt_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"


def parse_duration(text: str) -> int:
    """'90s', '30m', '1h', '1:30:00' or a bare number of seconds."""
    t = text.strip().lower()
    m = re.fullmatch(r"(\d+)\s*([smh]?)", t)
    if m:
        value = int(m.group(1)) * {"": 1, "s": 1, "m": 60, "h": 3600}[m.group(2)]
    else:
        m = re.fullmatch(r"(\d+):([0-5]?\d):([0-5]?\d)", t)
        if not m:
            raise ValueError(f"cannot read the time {text!r} (use 90s, 30m, 1h or H:MM:SS)")
        value = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
    if value <= 0:
        raise ValueError(f"the timeout must be positive, not {text!r}")
    return value


def timeout_cap():
    """(longest timeout allowed, the CPU-time limit behind it or None)."""
    try:
        import resource
        soft = resource.getrlimit(resource.RLIMIT_CPU)[0]
        unlimited = resource.RLIM_INFINITY
    except (ImportError, OSError, ValueError):
        return MAX_TIMEOUT, None
    if soft == unlimited or soft < 0:
        return MAX_TIMEOUT, None
    return min(MAX_TIMEOUT, soft - CPU_LIMIT_MARGIN), soft


def available_cores() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def gateway_reason(env, hostname: str) -> Optional[str]:
    if "/gateway/" in env.get("MODULEPATH", ""):
        return "its MODULEPATH is the gateway's"
    host = hostname.lower()
    if host.startswith("gateway") or host.endswith(".dmz"):
        return f"the host is {hostname}"
    return None


def crystal_root():
    """Where the job script's CRYSTAL will come from: (source, root).

    The script runs `module purge; module load CRYSTAL/...` and uses the
    $EBROOTCRYSTAL that sets, so where `module` exists the same load is
    probed here. Without `module` (a workstation, the test stubs) the script
    keeps the $EBROOTCRYSTAL it inherits.
    """
    script = ("type module >/dev/null 2>&1 || exit 3\n"
              "module purge >/dev/null 2>&1\n"
              f"module load {CRYSTAL_MODULE} >/dev/null 2>&1 || exit 4\n"
              'printf "\\nMACE_EBROOTCRYSTAL=%s\\n" "$EBROOTCRYSTAL"\n')
    try:
        proc = subprocess.run(["bash", "--login", "-c", script], stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return "environment", os.environ.get("EBROOTCRYSTAL") or None
    if proc.returncode == 3:
        return "environment", os.environ.get("EBROOTCRYSTAL") or None
    for line in reversed((proc.stdout or "").splitlines()):
        if line.startswith("MACE_EBROOTCRYSTAL="):
            return "module", line.split("=", 1)[1].strip() or None
    return "module", None


# --------------------------------------------------------------------------
# Decks and the order they run in
# --------------------------------------------------------------------------

class Deck:
    def __init__(self, path: Path):
        self.path = path
        self.dir = path.parent
        self.job = path.stem
        self.kind = "scf" if path.suffix == ".d12" else "prop"
        self.parent: Optional["Deck"] = None
        self.ambiguous: List["Deck"] = []
        self.calc_type = deck_calc_type(path, self.kind)
        self.status = "not run"
        self.note = ""
        self.f9_fresh = False
        self.out_fresh = False

    @property
    def out(self) -> Path:
        return self.dir / f"{self.job}.out"

    @property
    def f9(self) -> Path:
        return self.dir / f"{self.job}.f9"

    @property
    def log(self) -> Path:
        return self.dir / f"{self.job}-local.o"


def deck_calc_type(path: Path, kind: str) -> str:
    try:
        from mace.utils.calc_detection import deck_records, d3_calc_type, d12_calc_type
        records = deck_records(path.read_text(errors="ignore"), is_d3=(kind == "prop"))
        if kind == "prop":
            return d3_calc_type(records) or "properties"
        return d12_calc_type(records)
    except Exception:
        return "properties" if kind == "prop" else "SCF"


def collect_decks(targets) -> List[Deck]:
    paths = []
    for target in targets or ["."]:
        p = Path(os.path.abspath(target))
        if p.is_dir():
            found = sorted(p.glob("*.d12")) + sorted(p.glob("*.d3"))
            if not found:
                say(f"mace run: no .d12 or .d3 decks in {p}")
            paths.extend(found)
        elif p.is_file():
            if p.suffix not in (".d12", ".d3"):
                raise Refusal(f"{target} is not a .d12 or .d3 deck")
            paths.append(p)
        else:
            raise Refusal(f"{target}: no such file or directory")
    seen, unique = set(), []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    if not unique:
        raise Refusal("no decks to run")
    for p in unique:
        if not PATH_OK.match(str(p)):
            raise Refusal(f"{p}: the job script cannot use this path (it takes letters, digits "
                          "and . _ + , = ^ - only, as under SLURM); rename or move the deck")
    return [Deck(p) for p in unique]


def parent_matches(scf_job: str, prop_job: str) -> bool:
    """opt2d3 names the deck of <scf> <base>_<type>[N], where <base> is the SCF
    name with one trailing _opt/_sp taken off - or the whole name, when the
    _opt and _sp outputs sit side by side."""
    bases = {scf_job}
    for suffix in PARENT_SUFFIXES:
        if scf_job.endswith(suffix):
            bases.add(scf_job[:-len(suffix)])
            break
    for base in bases:
        if prop_job.startswith(base + "_") and PROP_TYPE_RE.fullmatch(prop_job[len(base) + 1:]):
            return True
    return False


def plan(decks: List[Deck]) -> List[Deck]:
    """Every .d12 first, then the .d3 decks, each knowing its parent SCF."""
    by_name = {}
    for d in decks:
        clash = by_name.get((d.dir, d.job))
        if clash:
            raise Refusal(f"{clash.path.name} and {d.path.name} would both write {d.job}.out "
                          f"in {d.dir}; run them separately")
        by_name[(d.dir, d.job)] = d
    by_scratch = {}
    for d in decks:
        clash = by_scratch.get((d.kind, d.job))
        if clash:
            raise Refusal(f"{clash.path} and {d.path} would share the scratch directory "
                          f"$SCRATCH/{SCRATCH_SUBDIR[d.kind]}/{d.job}; run them separately")
        by_scratch[(d.kind, d.job)] = d
    scf = sorted((d for d in decks if d.kind == "scf"), key=lambda d: str(d.path))
    prop = sorted((d for d in decks if d.kind == "prop"), key=lambda d: str(d.path))
    for d in prop:
        parents = [s for s in scf if s.dir == d.dir and parent_matches(s.job, d.job)]
        if len(parents) == 1:
            d.parent = parents[0]
        elif len(parents) > 1:
            d.ambiguous = parents
    return scf + prop


# --------------------------------------------------------------------------
# Running one deck
# --------------------------------------------------------------------------

def file_sig(path: Path):
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_ino, st.st_size, st.st_mtime_ns)


def rewritten(path: Path, before) -> bool:
    """Written by this run: not there before, or changed since."""
    now = file_sig(path)
    return now is not None and now != before


class StopFlag:
    def __init__(self):
        self.signum = None
        self.count = 0

    def __call__(self, signum, frame):
        if self.signum is None:
            self.signum = signum
        self.count += 1


def session_members(sid: int) -> List[int]:
    """Live processes of session ``sid`` (Linux /proc; [] elsewhere)."""
    if not os.path.isdir("/proc"):
        return []
    found = []
    for name in os.listdir("/proc"):
        if not name.isdigit() or int(name) == os.getpid():
            continue
        try:
            with open(f"/proc/{name}/stat", "rb") as fh:
                data = fh.read().decode("ascii", "replace")
        except OSError:
            continue
        fields = data[data.rfind(")") + 2:].split()
        if len(fields) >= 4 and fields[0] != "Z" and fields[3] == str(sid):
            found.append(int(name))
    return found


def _signal_group(pgid: int, sig) -> None:
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def stop_session(proc, sid: int, stop: StopFlag, grace: float = KILL_GRACE) -> List[int]:
    """SIGTERM the job's process group (bash, mpirun, hydra - which hands it to
    the ranks), then SIGKILL whatever is left in its session. A second Ctrl-C
    skips the wait. Returns the pids still alive afterwards."""
    asked = stop.count
    _signal_group(sid, signal.SIGTERM)
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        if proc.poll() is not None and not session_members(sid):
            break
        if stop.count > asked:
            break
        time.sleep(0.2)
    for pid in session_members(sid):
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    _signal_group(sid, signal.SIGKILL)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    deadline = time.monotonic() + 5
    while session_members(sid) and time.monotonic() < deadline:
        time.sleep(0.1)
    return session_members(sid)


def append_log(log: Path, text: str) -> None:
    try:
        with open(log, "a") as fh:
            fh.write(text.rstrip("\n") + "\n")
    except OSError:
        pass


def child_env(deck: Deck, ranks: int) -> dict:
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("SLURM_") and k not in WORKFLOW_ENV}
    env["SLURM_SUBMIT_DIR"] = str(deck.dir)
    env["SLURM_NTASKS"] = str(ranks)
    env["MACE_LOCAL_RUN"] = "1"
    return env


def generate_script(deck: Deck) -> Path:
    """The job script `mace submit` would submit, written to .mace_local/ so a
    <job>.sh of the user's own is never touched, and always fresh (an old one
    could lack the LOCAL RUN block and would run the completion callback)."""
    local = deck.dir / ".mace_local"
    local.mkdir(exist_ok=True)
    script = local / f"{deck.job}.sh"
    if script.exists():
        script.unlink()
    r = subprocess.run(["bash", str(GENERATORS[deck.kind]), deck.job], cwd=str(local),
                       stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120)
    if r.returncode != 0 or not script.is_file():
        raise RuntimeError(f"the job script generator failed: {(r.stderr or r.stdout).strip()}")
    if LOCAL_MARKER not in script.read_text(errors="ignore"):
        raise RuntimeError(f"{script} has no LOCAL RUN block (generator "
                           f"{GENERATORS[deck.kind]} is older than this mace run)")
    return script


def verdict(deck: Deck, out_before):
    """(completed?, reason) from the .out this run wrote."""
    if not rewritten(deck.out, out_before):
        return False, "no .out written"
    from mace.completion_checker import categorize_output_file
    category, _ = categorize_output_file(deck.out)
    if category.startswith("complete"):
        return True, ""
    from mace.submission.preflight import first_error_line
    try:
        text = deck.out.read_text(errors="ignore")
    except OSError:
        text = ""
    return False, first_error_line(text) or f"CRYSTAL did not finish ({category})"


def scratch_used(deck: Deck, record_before) -> str:
    record = deck.dir / f".{deck.job}.scratch"
    if rewritten(record, record_before):
        try:
            return record.read_text().strip()
        except OSError:
            pass
    return f"$SCRATCH/{SCRATCH_SUBDIR[deck.kind]}/{deck.job}"


def run_deck(deck: Deck, opts, stop: StopFlag, host: str) -> None:
    """Run one deck's job script and set deck.status / deck.note."""
    try:
        script = generate_script(deck)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        deck.status, deck.note = "failed", str(exc)
        return
    if stop.signum:
        deck.status = "not run"
        return
    out_before = file_sig(deck.out)
    f9_before = file_sig(deck.f9)
    record_before = file_sig(deck.dir / f".{deck.job}.scratch")
    how = f"{opts.cores} ranks"
    start = time.monotonic()
    with open(deck.log, "w") as fh:
        fh.write(f"mace run: {deck.job} on {host} ({how}, timeout "
                 f"{fmt_duration(opts.timeout)}), started {datetime.now().isoformat(timespec='seconds')}\n"
                 f"mace run: job script {script}\n")
    with open(deck.log, "ab") as logf:
        proc = subprocess.Popen(["bash", "--login", str(script)], cwd=str(deck.dir),
                                env=child_env(deck, opts.cores),
                                stdin=subprocess.DEVNULL, stdout=logf, stderr=subprocess.STDOUT,
                                start_new_session=True)
    sid = proc.pid
    append_log(deck.log, f"mace run: session {sid}; if mace run itself is killed, stop the job "
                         f"with: pkill -s {sid}")
    deadline = start + opts.timeout
    reason = None
    while True:
        try:
            proc.wait(timeout=0.5)
            break
        except subprocess.TimeoutExpired:
            pass
        if stop.signum:
            reason = "interrupted"
            break
        if time.monotonic() >= deadline:
            reason = "timeout"
            break
    elapsed = time.monotonic() - start
    if reason:
        left = stop_session(proc, sid, stop)
        why = (f"timeout {fmt_duration(opts.timeout)}" if reason == "timeout"
               else f"interrupted ({signal.Signals(stop.signum).name})")
        note = (f"mace run: stopped ({why}) after {fmt_duration(elapsed)} on {host}; "
                f"scratch kept at {scratch_used(deck, record_before)}")
        tmpdir = os.environ.get("TMPDIR")
        if tmpdir and scratch_used(deck, record_before).startswith(tmpdir.rstrip("/") + "/"):
            note += " (in $TMPDIR, which nothing removes after a local run)"
        if left:
            note += f"; still running: {' '.join(map(str, left))}"
        append_log(deck.log, note)
        deck.status = reason
        deck.note = why + (f"; still running: {' '.join(map(str, left))}" if left else "")
        deck.elapsed = elapsed
        return
    deck.elapsed = elapsed
    deck.out_fresh = rewritten(deck.out, out_before)
    ok, why = verdict(deck, out_before)
    deck.f9_fresh = rewritten(deck.f9, f9_before) and deck.f9.stat().st_size > 0
    deck.status = "completed" if ok else "failed"
    deck.note = why
    append_log(deck.log, f"mace run: {deck.status} after {fmt_duration(elapsed)}"
                         + (f" ({why})" if why else ""))


def hand_off(src: Path, dst: Path, stop: StopFlag) -> Optional[bool]:
    """Copy the SCF's .f9 to the properties deck's .f9 by an atomic replace.

    Returns whether an existing file was replaced, or None when interrupted (the
    old file is then left as it was). A symlinked destination is replaced, not
    written through.
    """
    if stop.signum:
        return None
    tmp = dst.with_name(dst.name + ".mace_tmp")
    if os.path.lexists(tmp):
        os.unlink(tmp)
    replaced = os.path.lexists(dst)
    shutil.copyfile(src, tmp)
    shutil.copystat(src, tmp)
    if stop.signum:
        os.unlink(tmp)
        return None
    os.replace(tmp, dst)
    return replaced


def prop_ready(deck: Deck, stop: StopFlag) -> Optional[str]:
    """None when the properties deck can run (after any .f9 hand-off), else why not."""
    if deck.ambiguous:
        names = ", ".join(p.job for p in deck.ambiguous)
        return f"ambiguous parent ({names}): run the one you mean with it"
    parent = deck.parent
    if parent is not None:
        if parent.status != "completed":
            return f"parent {parent.job} did not complete"
        if not parent.f9_fresh:
            return f"parent {parent.job} wrote no new {parent.job}.f9"
        replaced = hand_off(parent.f9, deck.f9, stop)
        if replaced is None:
            return "interrupted"
        say(f"      {parent.job}.f9 -> {deck.job}.f9"
            + (" (replaced the existing one)" if replaced else ""))
        return None
    try:
        size = deck.f9.stat().st_size
    except OSError:
        size = 0
    if size == 0:
        stem = deck.job
        return (f"no {stem}.f9 - run the SCF first (mace run <scf>.d12 {stem}.d3) "
                "or regenerate the deck with opt2d3")
    return None


# --------------------------------------------------------------------------
# --track: the run recorded in ./materials.db
# --------------------------------------------------------------------------

WORKFLOW_MARKERS = ("workflow_scripts", "workflow_configs", "workflow_outputs", "workflow_inputs")


def workflow_tree(dirs) -> Optional[Path]:
    """The workflow directory above any of ``dirs`` (as the queue manager
    detects one), or None."""
    for d in dirs:
        d = Path(d)
        for candidate in [d, *list(d.parents)[:7]]:
            if any((candidate / m).exists() for m in WORKFLOW_MARKERS):
                return candidate
    return None


class Tracker:
    """Records each local run in ./materials.db, as `mace submit --track` does,
    without the SLURM parts: no job id (none is faked), no job script, no
    error recovery (it would resubmit through sbatch), no workflow progression.
    The verdict is this run's own, from this deck's own .out."""

    def __init__(self, host: str):
        import contextlib
        import io
        from mace.queue.manager import EnhancedCrystalQueueManager
        with contextlib.redirect_stdout(io.StringIO()):
            self.manager = EnhancedCrystalQueueManager(
                d12_dir=os.getcwd(), enable_tracking=True, organize_outputs=False,
                enable_error_recovery=False, db_path=str(Path.cwd() / "materials.db"))
        self.manager.auto_submit_followups = False
        self.manager.workflow_enabled = False
        self.db = self.manager.db
        self.host = host
        self.mark_lost()

    def mark_lost(self) -> None:
        """A run whose mace run was killed outright is left 'running', and the
        queue manager, which goes by SLURM job ids, can never settle it."""
        import json
        for calc in self.db.get_calculations_by_status("running"):
            try:
                settings = json.loads(calc.get("settings_json") or "{}")
                pid = int(settings.get("pid"))
            except (TypeError, ValueError):
                continue
            if settings.get("run_mode") != "local" or settings.get("host") != self.host:
                continue
            try:
                os.kill(pid, 0)
                continue                                # still running
            except ProcessLookupError:
                pass
            except PermissionError:
                continue
            self.db.update_calculation_status(
                calc["calc_id"], "failed", error_type="local_lost",
                error_message=f"mace run (pid {pid} on {self.host}) ended without recording a result")
            say(f"mace run: marked {calc['calc_id']} failed (local_lost): its mace run is gone")

    def start(self, deck: Deck, opts) -> str:
        m = self.manager
        material_id, formula, metadata = m.extract_material_info_from_d12(deck.path)
        calc_type = m.determine_calc_type_from_file(deck.path)
        if not self.db.get_material(material_id):
            self.db.create_material(material_id=material_id, formula=formula, source_type="d12",
                                    source_file=str(deck.path), metadata=metadata)
        calc_id = self.db.create_calculation(
            material_id=material_id, calc_type=calc_type, input_file=str(deck.path),
            work_dir=str(deck.dir),
            settings={"run_mode": "local", "host": self.host, "pid": os.getpid(),
                      "ranks": opts.cores})
        self.db.update_calculation_status(calc_id, "running")
        return calc_id

    def finish(self, calc_id: str, deck: Deck) -> None:
        if deck.status == "completed":
            self.db.update_calculation_status(calc_id, "completed", output_file=str(deck.out))
            self.manager.handle_completed_calculation(calc_id)
            return
        out = str(deck.out) if deck.out_fresh else None
        if deck.status in ("timeout", "interrupted", "not run"):
            error_type = "local_timeout" if deck.status == "timeout" else "local_interrupted"
            message = deck.note or "interrupted before CRYSTAL started"
        elif out:
            self.db.update_calculation_status(calc_id, "failed", output_file=out)
            analysed, message = self.manager.analyze_calculation_error(self.db.get_calculation(calc_id))
            # Prefixed, so no recovery sweep ever takes it for a SLURM job to resubmit.
            error_type = f"local_{analysed}"
        else:
            error_type, message = "local_no_output", deck.note or "no .out written"
        self.db.update_calculation_status(calc_id, "failed", output_file=out,
                                          error_type=error_type, error_message=message)


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mace run",
        description="Run CRYSTAL decks now on the node you are on (an HPCC dev node), "
                    "without SLURM, through the same job script mace submit uses.")
    p.add_argument("targets", nargs="*", metavar="FILE|DIR",
                   help=".d12/.d3 decks or directories of them (default: current directory)")
    p.add_argument("-n", "--cores", type=int, default=None, metavar="N",
                   help=f"MPI ranks (default: {DEFAULT_CORES}, or fewer on a smaller machine; "
                        f"at most {MAX_CORES})")
    p.add_argument("--timeout", default=None, metavar="T",
                   help="wall time per deck: 90s, 30m, 1h or H:MM:SS (default: 30m)")
    p.add_argument("--track", action="store_true",
                   help="record the runs in ./materials.db (as mace submit --track)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the order, scratch directories and .f9 hand-offs; run nothing")
    return p


def check_options(opts) -> List[str]:
    """Validate cores and timeout in place; returns warnings. Raises Refusal."""
    warnings = []
    cpus = available_cores()
    if opts.cores is None:
        opts.cores = min(DEFAULT_CORES, cpus)
    else:
        limit = min(MAX_CORES, cpus)
        if not 1 <= opts.cores <= limit:
            raise Refusal(f"-n {opts.cores}: use 1 to {limit} ranks here "
                          f"({cpus} cores available to this process; mace run stops at "
                          f"{MAX_CORES}) - for more, use mace submit")
        if opts.cores > POLITE_CORES:
            warnings.append(f"-n {opts.cores}: dev nodes are shared; {DEFAULT_CORES}-"
                            f"{POLITE_CORES} ranks is polite")
    cap, cpu_limit = timeout_cap()
    limit_text = (f"this node stops any process after {fmt_duration(cpu_limit)} of CPU time "
                  "(ulimit -t)" if cpu_limit else
                  f"mace run allows at most {fmt_duration(MAX_TIMEOUT)}")
    if cap <= 0:
        raise Refusal(f"{limit_text}: too little for a run; use mace submit")
    if opts.timeout is None:
        opts.timeout = DEFAULT_TIMEOUT
        if opts.timeout > cap:
            opts.timeout = cap
            warnings.append(f"timeout {fmt_duration(cap)} per deck: {limit_text}")
    else:
        try:
            opts.timeout = parse_duration(opts.timeout)
        except ValueError as exc:
            raise Refusal(str(exc))
        if opts.timeout > cap:
            raise Refusal(f"--timeout {fmt_duration(opts.timeout)}: {limit_text}, so at most "
                          f"{fmt_duration(cap)}; use mace submit for longer runs")
    return warnings


def check_node(decks: List[Deck]) -> str:
    """Refuse where CRYSTAL cannot run as the job script would run it."""
    if os.environ.get("SLURM_JOB_ID"):
        raise Refusal(f"inside a SLURM job ({os.environ['SLURM_JOB_ID']}): use sbatch, or run "
                      "the job script directly - mace run is for a dev node outside SLURM")
    host = socket.gethostname()
    gateway = gateway_reason(os.environ, host)
    if gateway:
        raise Refusal(f"This is the HPCC gateway ({gateway}): CRYSTAL cannot run here. Log in "
                      "to a dev node first (ssh dev-amd20, dev-amd24 or dev-intel18) and rerun "
                      "mace run.")
    source, root = crystal_root()
    if not root:
        if source == "module":
            raise Refusal(f"the {CRYSTAL_MODULE} module cannot be loaded on {host}; run "
                          "mace run on a dev node (ssh dev-amd20)")
        raise Refusal(f"no CRYSTAL here: $EBROOTCRYSTAL is not set and there is no module "
                      f"command to load {CRYSTAL_MODULE}")
    needed = sorted({BINARIES[d.kind] for d in decks})
    missing = [b for b in needed if not os.access(os.path.join(root, "bin", b), os.X_OK)]
    if missing:
        raise Refusal(f"{', '.join(missing)} not found in {root}/bin")
    if source == "environment" and not shutil.which("mpirun"):
        raise Refusal("no mpirun on PATH")
    return host


def check_slurm_jobs(decks: List[Deck]) -> None:
    """Refuse a deck whose name a SLURM job of this user has: they would share
    the scratch directory and write the same output names. (Conservative - a
    workflow step of that name uses another scratch directory - and one way
    only: mace submit does not look for mace run.)"""
    squeue = shutil.which("squeue")
    if not squeue:
        return
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or str(os.getuid())
    try:
        r = subprocess.run([squeue, "-h", "-u", user, "-o", "%j %i %T"],
                           stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        say(f"mace run: warning: could not ask squeue for running jobs ({exc})", err=True)
        return
    if r.returncode != 0:
        say(f"mace run: warning: squeue failed ({r.stderr.strip()}); not checked for "
            "SLURM jobs of the same name", err=True)
        return
    jobs = {}
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 3:
            jobs.setdefault(parts[0], (parts[1], parts[2]))
    for d in decks:
        if d.job in jobs:
            job_id, state = jobs[d.job]
            raise Refusal(f"{d.job} is also SLURM job {job_id} ({state}): it uses the same "
                          f"scratch directory and output names - wait for it or scancel it")


def lock_dir() -> Path:
    """$XDG_RUNTIME_DIR/mace-run-<uid> (/tmp/mace-run-<uid> without it), and
    only when it is this user's alone: in /tmp anyone could create it first."""
    import stat
    base = os.environ.get("XDG_RUNTIME_DIR") or "/tmp"
    d = Path(base) / f"mace-run-{os.getuid()}"
    d.mkdir(mode=0o700, exist_ok=True)
    st = os.lstat(d)
    if (not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid()
            or st.st_mode & (stat.S_IWGRP | stat.S_IWOTH)):
        raise Refusal(f"the lock directory {d} is not private to you (owner and mode); "
                      "remove it, or set XDG_RUNTIME_DIR to a directory of your own")
    return d


def acquire_locks(decks: List[Deck], host: str) -> List[int]:
    """One lock per scratch directory ($SCRATCH/crys23[/prop]/<job>), held for
    the whole run, so two runs on this node never share one. The lock files
    are never removed: unlinking one while another run waits on it would let
    both hold a lock, each on its own inode."""
    import fcntl
    held = []
    try:
        locks = lock_dir()
        for d in decks:
            path = locks / f"{d.kind}-{d.job}.lock"
            fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                try:
                    owner = os.pread(fd, 400, 0).decode(errors="replace").strip()
                except OSError:
                    owner = ""
                os.close(fd)
                raise Refusal(f"another mace run has {d.job} ($SCRATCH/{SCRATCH_SUBDIR[d.kind]}/"
                              f"{d.job}){' - ' + owner if owner else ''}; wait for it to finish")
            os.ftruncate(fd, 0)
            os.pwrite(fd, f"pid {os.getpid()} on {host}: {d.path}\n".encode(), 0)
            held.append(fd)
    except BaseException:
        for fd in held:
            os.close(fd)
        raise
    return held


def print_plan(decks: List[Deck], opts) -> None:
    how = f"{opts.cores} ranks"
    say(f"mace run: {len(decks)} deck(s), {how}, timeout {fmt_duration(opts.timeout)} each, "
        "in this order:")
    for i, d in enumerate(decks, 1):
        say(f"  {i}. {d.path}  ({d.calc_type})")
        say(f"     scratch $SCRATCH/{SCRATCH_SUBDIR[d.kind]}/{d.job} (resolved by the job script)")
        if d.ambiguous:
            say(f"     skipped: ambiguous parent ({', '.join(p.job for p in d.ambiguous)})")
        elif d.parent:
            say(f"     after {d.parent.job}: {d.parent.job}.f9 -> {d.job}.f9")
        elif d.kind == "prop":
            say(f"     uses the {d.job}.f9 beside it")


def report(deck: Deck, i: int, n: int) -> None:
    elapsed = getattr(deck, "elapsed", None)
    took = f" in {fmt_duration(elapsed)}" if elapsed is not None else ""
    if deck.status == "completed":
        say(f"      completed{took}")
        return
    label = {"failed": "FAILED", "timeout": "TIMED OUT", "interrupted": "STOPPED",
             "skipped": "skipped"}.get(deck.status, deck.status)
    say(f"      {label}{took}" + (f": {deck.note}" if deck.note else ""))
    if deck.status in ("failed", "timeout", "interrupted") and deck.log.exists():
        say(f"      job log: {deck.log}")
        if deck.note == "no .out written":
            try:
                tail = [l for l in deck.log.read_text(errors="ignore").splitlines() if l.strip()]
            except OSError:
                tail = []
            for line in tail[-10:]:
                say(f"        {line}")


def main(argv=None) -> int:
    parser = build_parser()
    opts = parser.parse_args(argv)
    try:
        decks = plan(collect_decks(opts.targets))
        warnings = check_options(opts)
        host = check_node(decks)
        check_slurm_jobs(decks)
        if opts.track:
            tree = workflow_tree([Path(os.getcwd())] + [d.dir for d in decks])
            if tree:
                raise Refusal(f"--track inside a workflow tree ({tree}): its calculations are "
                              "recorded by the workflow - use mace workflow")
    except Refusal as exc:
        say(f"mace run: {exc}", err=True)
        return EXIT_REFUSED
    for w in warnings:
        say(f"mace run: warning: {w}", err=True)
    for d in decks:
        if d.calc_type in ("OPT", "FREQ"):
            say(f"mace run: warning: {d.path.name} is a {d.calc_type}, which rarely fits on a dev "
                f"node; it is stopped at the timeout ({fmt_duration(opts.timeout)})", err=True)
    if opts.dry_run:
        print_plan(decks, opts)
        say("mace run: dry run - nothing was written")
        return EXIT_OK

    try:
        locks = acquire_locks(decks, host)
    except Refusal as exc:
        say(f"mace run: {exc}", err=True)
        return EXIT_REFUSED
    except OSError as exc:
        say(f"mace run: cannot take the run lock: {exc}", err=True)
        return EXIT_REFUSED

    tracker = None
    if opts.track:
        # A workflow context in the environment would send the records to its
        # database and let a completion progress the workflow.
        for key in WORKFLOW_ENV:
            os.environ.pop(key, None)
        try:
            tracker = Tracker(host)
        except Exception as exc:
            for fd in locks:
                os.close(fd)
            say(f"mace run: --track: cannot open materials.db: {exc}", err=True)
            return EXIT_REFUSED

    stop = StopFlag()
    previous = {signal.SIGINT: signal.signal(signal.SIGINT, stop),
                signal.SIGTERM: signal.signal(signal.SIGTERM, stop)}
    if signal.getsignal(signal.SIGHUP) is not signal.SIG_IGN:      # nohup keeps it ignored
        previous[signal.SIGHUP] = signal.signal(signal.SIGHUP, stop)
    if hasattr(signal, "SIGTSTP"):
        # Suspended, the runner would stop supervising the job and its timeout.
        previous[signal.SIGTSTP] = signal.signal(signal.SIGTSTP, signal.SIG_IGN)
    try:
        run_all(decks, opts, stop, host, tracker)
    finally:
        for sig, handler in previous.items():
            try:
                signal.signal(sig, handler)
            except (OSError, ValueError, TypeError):
                pass
        for fd in locks:
            os.close(fd)
    return summarize(decks, stop)


def run_all(decks: List[Deck], opts, stop: StopFlag, host: str, tracker=None) -> None:
    n = len(decks)
    how = f"{opts.cores} ranks"
    for i, d in enumerate(decks, 1):
        if stop.signum:
            break
        where = "" if d.dir == Path(os.getcwd()) else f" in {d.dir}"
        say(f"[{i}/{n}] {d.path.name} ({d.calc_type}, {how}){where}")
        if d.kind == "prop":
            why = prop_ready(d, stop)
            if why == "interrupted":
                break
            if why:
                d.status, d.note = "skipped", why
                report(d, i, n)
                continue
        calc_id = None
        if tracker:
            try:
                calc_id = tracker.start(d, opts)
            except Exception as exc:
                say(f"mace run: warning: --track could not record {d.path.name}: {exc}", err=True)
        run_deck(d, opts, stop, host)
        report(d, i, n)
        if tracker and calc_id:
            try:
                tracker.finish(calc_id, d)
            except Exception as exc:
                say(f"mace run: warning: --track could not record the result of "
                    f"{d.path.name}: {exc}", err=True)


def summarize(decks: List[Deck], stop: StopFlag) -> int:
    counts = {}
    for d in decks:
        counts[d.status] = counts.get(d.status, 0) + 1
    say("mace run: " + ", ".join(f"{v} {k}" for k, v in counts.items()))
    not_run = [d for d in decks if d.status == "not run"]
    if not_run:
        say("mace run: not run: " + " ".join(d.path.name for d in not_run))
    if stop.signum:
        return 128 + int(stop.signum)
    if any(d.status == "timeout" for d in decks):
        return EXIT_TIMEOUT
    if any(d.status != "completed" for d in decks):
        return EXIT_FAILED
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
