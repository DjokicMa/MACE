"""A SLURM job log's calculation (plan E14): which CRYSTAL output a log
belongs to, read-only.

The log -> output direction asks the CLI's own helper the other way round:
every ``.out`` in the folder whose stem starts the log's name is a
candidate, and ``completion_checker.find_slurm_job_id`` (the rule the queue
manager uses to find an output's log) must name this log's job for that
output, or the stem alone decides when the log carries no id. Array ids go
through ``mace.queue.slurm_ids``. A ``slurm-<id>.out`` (no stem) is looked up
in ``materials.db`` by ``slurm_job_id``, read only, retrying while a
completion callback holds the write lock and never falling back to an
immutable read (a database a callback is writing could be read half-written).
"""
from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

DB_RETRY_S = 5.0
TEXT_SNIFF = 64 * 1024
# the job id a log name carries: <stem>-<id>.o, <stem>-<A>_<T>.o (array task), <stem>.o<id>,
# slurm-<id>.out, <name>_<id>.out; the .e/.err partners alike
_ID_IN_NAME = (
    re.compile(r"^(?P<stem>.+)-(?P<id>\d+(?:_\d+)?)\.(?:o|e|out|err|log)$", re.I),
    re.compile(r"^(?P<stem>.+)\.[oe](?P<id>\d+)$", re.I),
    re.compile(r"^slurm-(?P<id>\d+(?:_\d+)?)\.(?:out|err)$", re.I),
    re.compile(r"^(?P<stem>.+)_(?P<id>\d{5,})\.(?:o|e|out|err|log)$", re.I),
)
# DB_BUSY_HOOK: tests count connection modes (never immutable)
OPEN_LOG: List[str] = []


class DbBusy(RuntimeError):
    pass


def id_from_name(name: str) -> Tuple[Optional[str], Optional[str]]:
    """(stem, job id) a log's name carries; (None, None) when it carries none."""
    for rx in _ID_IN_NAME:
        m = rx.match(name)
        if m:
            d = m.groupdict()
            return d.get("stem"), d["id"]
    return None, None


def _out_candidates(log: Path) -> List[Path]:
    """The .out files beside the log (and one folder up for a logs/ folder)
    whose stem starts the log's name, longest stem first."""
    dirs = [log.parent]
    if log.parent.name.lower() == "logs":
        dirs.append(log.parent.parent)
    found = []
    for d in dirs:
        try:
            entries = list(d.iterdir())
        except OSError:
            continue
        for f in entries:
            if f == log or f.suffix.lower() != ".out" or not f.is_file():
                continue
            stem = f.stem
            if log.name.startswith(stem) and len(log.name) > len(stem) and log.name[len(stem)] in "-._":
                found.append(f)
    return sorted(found, key=lambda f: -len(f.stem))


def _db_lookup(db: Path, job_id: str) -> Optional[Dict[str, Any]]:
    """The calculation row with this slurm_job_id, read only; retries while
    the database is locked, then DbBusy. Never an immutable open."""
    uri = f"file:{quote(str(db.resolve()))}?mode=ro"
    deadline = time.monotonic() + DB_RETRY_S
    delay = 0.05
    while True:
        try:
            OPEN_LOG.append("ro")
            conn = sqlite3.connect(uri, uri=True, timeout=0.5)
            try:
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA busy_timeout = 500")
                row = conn.execute("SELECT calc_id, calc_type, output_file, work_dir, input_file FROM calculations "
                                   "WHERE slurm_job_id = ? LIMIT 1", (job_id,)).fetchone()
                return dict(row) if row is not None else None
            finally:
                conn.close()
        except sqlite3.OperationalError as exc:
            msg = str(exc).lower()
            if "no such table" in msg or "unable to open" in msg or "not a database" in msg:
                return None
            if time.monotonic() >= deadline:
                raise DbBusy("Database busy") from exc
            time.sleep(delay)
            delay = min(delay * 2, 0.5)


def _db_files(root: Path, log: Path) -> List[Path]:
    """The materials.db files beside the log, one folder up and at the root:
    only those inside the served folder (a log at the root never reads a
    database in the folder above it)."""
    out = []
    for d in (log.parent, log.parent.parent, root):
        p = _confined(root, d / "materials.db")
        if p is not None and p not in out:
            out.append(p)
    return out


def _confined(root: Path, p: Path) -> Optional[Path]:
    try:
        r = p.resolve()
    except OSError:
        return None
    if r != root and root not in r.parents:
        return None
    return r if r.is_file() else None


def link(root: Path, log: Path) -> Dict[str, Any]:
    """{job_id, array, calc (path relative to root or None), how, busy}."""
    from mace.completion_checker import find_slurm_job_id
    from mace.queue.slurm_ids import array_label
    root = Path(root).resolve()
    stem, job_id = id_from_name(log.name)
    out: Dict[str, Any] = {"job_id": job_id, "array": array_label(job_id).strip() if job_id else "",
                           "calc": None, "how": None, "busy": False}
    # 1. by name: the output whose stem starts the log's name (the longest such stem); the CLI's own
    # find_slurm_job_id on that output says whether this log is its newest one
    # only outputs inside the served folder (a logs/ folder at the root, or a symlink, could point outside)
    cands = [c for c in _out_candidates(log) if _confined(root, c) is not None]
    if cands:
        cand = cands[0]
        try:
            newest = find_slurm_job_id(cand)
        except Exception:
            newest = None
        out.update(calc=cand.resolve().relative_to(root).as_posix(), how="name",
                   newest=newest is None or job_id is None or newest == job_id, newest_id=newest)
        return out
    # 2. by the job id in materials.db
    if job_id:
        for db in _db_files(root, log):
            try:
                row = _db_lookup(db, job_id)
            except DbBusy:
                out["busy"] = True
                return out
            if not row:
                continue
            for key in ("output_file", "input_file"):
                v = row.get(key)
                if not v:
                    continue
                p = Path(v)
                if not p.is_absolute():
                    p = Path(row.get("work_dir") or db.parent) / p
                if key == "input_file":
                    p = p.with_suffix(".out")
                r = _confined(root, p)
                if r is not None:
                    out.update(calc=r.relative_to(root).as_posix(), how="database")
                    return out
    # 3. a CRYSTAL output the log text names, in the same folder
    try:
        with open(log, "rb") as fh:
            text = fh.read(TEXT_SNIFF).decode("utf-8", "replace")
    except OSError:
        text = ""
    for m in re.finditer(r"([A-Za-z0-9][\w.+-]*)\.(?:out|d12)\b", text):
        p = _confined(root, log.parent / (m.group(1) + ".out"))
        if p is not None and p != log.resolve():
            out.update(calc=p.relative_to(root).as_posix(), how="log text")
            return out
    return out


def first_line(log: Path) -> str:
    """The first non-empty line of a .e/.err file (for its Problems row)."""
    try:
        with open(log, "rb") as fh:
            for raw in fh.read(TEXT_SNIFF).decode("utf-8", "replace").splitlines():
                if raw.strip():
                    return raw.strip()[:200]
    except OSError:
        pass
    return ""
