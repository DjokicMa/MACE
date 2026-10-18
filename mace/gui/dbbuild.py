"""Create or Update Database from a Folder (PLAN_analysis C10).

The GUI runs the same command a terminal would, ``mace analyze
--extract-properties FOLDER --db-path DB --register|--no-register [--skip-unchanged]
[--backup --keep-backups N]``, as a cancelable task through clirun (nice 10,
SIGTERM then SIGKILL): the preview adds ``--dry-run --json``, the build adds
``--progress-json`` and reads one JSON line per file for its progress. The
command shown in the dialog is built from the same argument list.

The server never opens the database read-write (C17.5): the dialog's facts
(property rows, backups) come from read-only connections and directory
listings.
"""
from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from . import clirun

# The exact sentence of the dialog and of docs/gui.md (a test compares them)
GUARANTEE = ("Nothing is deleted from the database. Re-extracting a calculation replaces its own earlier property "
             "values; rows filed under an old material id for that calculation are moved, not removed. "
             "A backup is made first.")
OWNER = "A database build"
OUTSIDE = ("This database is outside the served folder. Run the command below in a terminal instead.")
CPU_LIMIT = ("The database update reached this node's CPU-time limit after {n} file{s}; what it wrote is kept. "
             "Run it again with 'Only new and changed files' to continue where it stopped.")
LARGE_FOLDER = 500            # outputs: above this the dialog suggests a compute node
OUTPUT_SCAN_CAP = 20000


def _backups_dir(db: Path) -> Path:
    return db.parent / "backups"


def backups(db: Path, keep: int) -> Dict[str, Any]:
    """The existing MACE backups of this database (newest first, with
    sizes) and the names the next --backup run would prune."""
    from .dbsafety_loader import dbsafety
    pat = dbsafety().BACKUP_NAME
    d = _backups_dir(db)
    items = []
    if d.is_dir():
        for p in d.iterdir():
            m = pat.match(p.name)
            if m and m.group(1) == "materials" and p.is_file():
                try:
                    items.append((m.group(2), int(m.group(3) or 1), p.name, p.stat().st_size))
                except OSError:
                    continue
    items.sort(reverse=True)
    rows = [{"name": n, "size": s} for _, _, n, s in items]
    # the run adds one, then keeps the newest `keep`: the oldest beyond keep-1 go
    prune = [r["name"] for r in rows[max(0, keep - 1):]] if db.is_file() else []
    return {"dir": "backups", "files": rows, "prune": prune}


def property_rows(db: Path) -> Optional[int]:
    """How many property rows the database has (read-only), None when it
    cannot be read now."""
    if not db.is_file():
        return None
    from . import db as gui_db
    try:
        conn = gui_db.connect(db)
    except Exception:
        return None
    try:
        return int(conn.execute("SELECT count(*) FROM properties").fetchone()[0])
    except Exception:
        return None
    finally:
        conn.close()


def count_outputs(folder: Path) -> int:
    n = 0
    for _ in folder.rglob("*.out"):
        n += 1
        if n >= OUTPUT_SCAN_CAP:
            break
    return n


def context_db(folder: Path) -> Optional[Path]:
    """The workflow-context database the CLI would use for this folder:
    exactly one ``.mace_context_*`` with a materials.db (read-only check; the
    context is never activated in the server)."""
    dirs = sorted(folder.glob(".mace_context_*"))
    if len(dirs) == 1 and (dirs[0] / "materials.db").is_file():
        return dirs[0] / "materials.db"
    return None


def choose_db(root: Path, folder: Path, open_db: Optional[Path]) -> Dict[str, Any]:
    """The database the dialog proposes: the open one, then
    ``<folder>/materials.db``, then the workflow-context database, then a new
    ``<folder>/materials.db``."""
    if open_db is not None:
        return {"path": open_db, "source": "open"}
    here = folder / "materials.db"
    if here.is_file():
        return {"path": here, "source": "folder"}
    ctx = context_db(folder)
    if ctx is not None:
        return {"path": ctx, "source": "context"}
    return {"path": here, "source": "new"}


def inside(root: Path, p: Path) -> bool:
    try:
        r = p.resolve()
    except (OSError, RuntimeError):
        return False
    return r == root or root in r.parents


def rel_or_abs(root: Path, p: Path) -> str:
    try:
        r = p.resolve()
        if r == root or root in r.parents:
            return r.relative_to(root).as_posix() or "."
    except (OSError, RuntimeError, ValueError):
        pass
    return str(p)


def cli_args(folder: str, db: str, register: bool, skip_unchanged: bool, backup: bool, keep: int,
             dry_run: bool = False, progress: bool = False) -> List[str]:
    """The arguments after ``mace``: one list for the command shown, the
    preview and the build."""
    a = ["analyze", "--extract-properties", folder, "--db-path", db]
    # registering is the CLI's default for a folder: Register off says so
    a.append("--register" if register else "--no-register")
    if skip_unchanged:
        a.append("--skip-unchanged")
    if backup:
        a += ["--backup", "--keep-backups", str(int(keep))]
    if dry_run:
        a += ["--dry-run", "--json"]
    if progress:
        a.append("--progress-json")
    return a


def command_text(args: List[str]) -> str:
    return "mace " + " ".join(shlex.quote(x) for x in args)


def options(body: Dict[str, Any], db_exists: bool) -> Dict[str, Any]:
    """The dialog's options, validated: a backup is always made when the
    database exists (locked on), keep is 1-100."""
    register = bool(body.get("register", True))
    skip = bool(body.get("skip_unchanged", True))
    try:
        keep = int(body.get("keep", 3))
    except (TypeError, ValueError):
        raise ValueError("Keep the last N backups: a number from 1 to 100")
    if not 1 <= keep <= 100:
        raise ValueError("Keep the last N backups: a number from 1 to 100")
    backup = True if db_exists else bool(body.get("backup", False))
    return {"register": register, "skip_unchanged": skip, "backup": backup, "keep": keep}


def preview(root: Path, folder_arg: str, db_arg: str, opts: Dict[str, Any], ctx: Any) -> Dict[str, Any]:
    """``--dry-run --json`` in a CLI child: the classification of every
    output (new, changed, unchanged, not tracked, unreadable, incomplete)."""
    args = cli_args(folder_arg, db_arg, opts["register"], False, False, opts["keep"], dry_run=True)
    r = clirun.run(clirun.mace_argv("--no-banner", *args), root, ctx, timeout=600)
    doc = None
    for line in r.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                doc = json.loads(line)
            except ValueError:
                continue
    if r.returncode != 0 or doc is None:
        raise RuntimeError("The preview failed: " + (r.stderr.strip().splitlines() or ["no output"])[-1][-500:])
    for row in doc.get("files", []):
        row["rel"] = rel_or_abs(root, Path(row["file"]) if Path(row["file"]).is_absolute() else root / row["file"])
        row.pop("file", None)
    return {"counts": doc.get("counts", {}), "files": doc.get("files", []), "db_exists": doc.get("db_exists"),
            "command": command_text(args[:-2])}


def build(root: Path, folder_arg: str, db_arg: str, opts: Dict[str, Any], ctx: Any,
          progress: Callable[[float, str], None]) -> Dict[str, Any]:
    """The build: ``--progress-json``; progress i/n with the current file.
    A cancel stops the CLI (SIGTERM then SIGKILL); extraction commits per
    file, so what was written stays and the summary names it."""
    args = cli_args(folder_arg, db_arg, opts["register"], opts["skip_unchanged"], opts["backup"], opts["keep"],
                    progress=True)
    state: Dict[str, Any] = {"files": [], "done": None, "backup": None, "pruned": [], "resumed": None}

    def on_line(line: str) -> None:
        line = line.strip()
        if not line.startswith("{"):
            return
        try:
            rec = json.loads(line)
        except ValueError:
            return
        if rec.get("done"):
            state["done"] = rec
            return
        if isinstance(rec.get("resumed"), dict):
            # an earlier build of this database stopped before its end: this one continues it
            from mace.utils import db_build
            state["resumed"] = rec["resumed"]
            progress(0.0, db_build.resumed_message(rec["resumed"].get("already", 0), rec["resumed"].get("total", 0)))
            return
        if "backup" in rec and "i" not in rec:
            state["backup"] = rel_or_abs(root, Path(rec["backup"]))
            state["pruned"] = rec.get("pruned", [])
            return
        f = rec.get("file")
        if f:
            rec["rel"] = rel_or_abs(root, Path(f) if Path(f).is_absolute() else root / f)
        state["files"].append(rec)
        n = max(1, int(rec.get("n") or 1))
        progress(min(1.0, int(rec.get("i") or 0) / n), f"{rec.get('i')}/{n} {Path(f or '').name}")

    try:
        r = clirun.run(clirun.mace_argv("--no-banner", *args), root, ctx, timeout=24 * 3600, on_line=on_line)
    except clirun.CliCanceled:
        written = [x["rel"] for x in state["files"] if x.get("status") == "extracted"]
        raise BuildCanceled(written, state["backup"])
    if getattr(r, "cpu_limit", False) and state["done"] is None:
        # the kernel stopped the CLI at the per-process CPU-time limit (cpulimit.py): every file it
        # finished is committed, and the next build with --skip-unchanged continues from there
        n = sum(1 for x in state["files"] if x.get("status") == "extracted")
        raise RuntimeError(CPU_LIMIT.format(n=n, s="" if n == 1 else "s"))
    if r.returncode != 0 and state["done"] is None:
        tail = (r.stderr.strip().splitlines() or ["no output"])[-1][-500:]
        raise RuntimeError(f"The database build failed: {tail}")
    return summary(state, args)


def summary(state: Dict[str, Any], args: List[str]) -> Dict[str, Any]:
    """The build's result: counted and worded by the CLI's own functions
    (db_build.summarize / summary_message), so the notification says what the
    terminal says."""
    from mace.utils import db_build
    files = state["files"]
    counted = db_build.summarize(files)
    done = dict(state["done"] or {})
    for k, v in (("new", counted["new"]), ("changed", counted["changed"]), ("reextracted", counted["reextracted"]),
                 ("failed_outputs", len(counted["failed"])), ("skipped_outputs", counted["skipped"])):
        done.setdefault(k, v)
    return {"done": state["done"] or {}, "files": files, "backup": state["backup"], "pruned": state["pruned"],
            "resumed": state.get("resumed"),
            "new": counted["new"], "changed": counted["changed"], "failed": counted["failed"],
            "skipped": counted["skipped"], "command": command_text(args[:-1]),
            "message": done.get("message") or db_build.summary_message(done)}


class BuildCanceled(Exception):
    def __init__(self, written: List[str], backup: Optional[str]):
        self.written, self.backup = written, backup
        msg = f"Canceled after {len(written)} file{'' if len(written) == 1 else 's'} written"
        if backup:
            msg += f"; the backup is {backup}"
        msg += ". With 'Only new and changed files' the next update continues where this one stopped"
        super().__init__(msg)
