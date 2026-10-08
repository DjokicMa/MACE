"""Files that are still being written (PLAN_analysis C18): the one place
MACE reads a growing .out, scratch file or properties file.

A CRYSTAL job writes its .out as it runs (``mpirun ... >& $JOB.out``),
truncates it when a resubmission starts, and copies the wavefunction at the
very end. So every reader:

* takes a ``snapshot`` first (size, mtime, inode, the hash of the first
  64 KB) and reads only ``[0, snapshot.size)``;
* uses only complete lines (``read_complete``): a line without its newline
  may be half a number;
* asks ``same_file`` before continuing from an offset: a smaller file, a new
  inode or a different head means the file was replaced (a new run);
* decides "live" with ``is_live``: no terminal marker and (the job is known
  to be queued or running, or the file changed within the live window).

Standard library only (the CLI side, PLAN_F's progress.py, imports it too).
"""
from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

HEAD = 64 * 1024
LIVE_MINUTES = 10
# CRYSTAL's normal and error ends, and the line `mace run` appends to a run it stopped
TERMINAL_MARKERS = ("TOTAL CPU TIME =", "EEEEEEEEEE TERMINATION", "OPT END - FAILED")


def _markers():
    """The terminal markers: mace.completion_checker's own (OPT END,
    TOTAL CPU TIME, MACE_RUN_STOP_MARK, the error patterns), imported, not
    copied, plus CRYSTAL's termination banner."""
    out = list(TERMINAL_MARKERS)
    try:
        from mace.completion_checker import ERROR_PATTERNS, MACE_RUN_STOP_MARK
        out.append(MACE_RUN_STOP_MARK)
        for keys in ERROR_PATTERNS.values():
            out.extend(keys)
    except Exception:          # pragma: no cover - the checker is part of MACE
        out.append("mace run: STOPPED")
    return tuple(dict.fromkeys(out))


@dataclass(frozen=True)
class Snap:
    size: int
    mtime_ns: int
    ino: int
    dev: int
    head_sha1: str
    head_len: int

    def token(self) -> str:
        return f"{self.ino}:{self.size}:{self.head_sha1[:16]}"


def _head_hash(path: Path, n: int) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha1(fh.read(n)).hexdigest()


def snapshot(path) -> Snap:
    path = Path(path)
    st = os.stat(path)
    n = min(HEAD, st.st_size)
    return Snap(int(st.st_size), int(st.st_mtime_ns), int(st.st_ino), int(st.st_dev), _head_hash(path, n), n)


def same_file(old: Snap, new: Snap, path=None) -> bool:
    """The same file, grown or unchanged: same inode/device, not smaller,
    and the same bytes over the old snapshot's head (a file shorter than
    64 KB that grew keeps its old prefix)."""
    if old.ino != new.ino or old.dev != new.dev or new.size < old.size:
        return False
    if new.head_len == old.head_len:
        return new.head_sha1 == old.head_sha1
    if path is None:
        return False
    try:
        return _head_hash(Path(path), old.head_len) == old.head_sha1
    except OSError:
        return False


def read_complete(path, snap: Snap, start: int = 0) -> Tuple[bytes, int]:
    """Bytes ``[start, snap.size)`` cut after the last newline, and the
    offset where the next read starts."""
    if start >= snap.size:
        return b"", start
    with open(path, "rb") as fh:
        fh.seek(start)
        data = fh.read(snap.size - start)
    k = data.rfind(b"\n")
    if k < 0:
        return b"", start
    return data[:k + 1], start + k + 1


def tail_text(path, snap: Snap, n: int = 65536) -> str:
    """The last ``n`` bytes of the snapshot, as text (where the markers are)."""
    with open(path, "rb") as fh:
        fh.seek(max(0, snap.size - n))
        return fh.read(min(n, snap.size)).decode("utf-8", "replace")


def has_terminal_marker(text: str) -> bool:
    return any(m in text for m in _markers())


def is_live(path, snap: Optional[Snap] = None, text_tail: Optional[str] = None, job_state: Optional[str] = None,
            live_minutes: float = LIVE_MINUTES, now: Optional[float] = None) -> bool:
    """No terminal marker, and the job is known to be queued or running, or
    (job state unknown) the file changed within ``live_minutes``."""
    snap = snap or snapshot(path)
    if text_tail is None:
        text_tail = tail_text(path, snap)
    if has_terminal_marker(text_tail):
        return False
    if job_state is not None:
        return job_state.upper() in ("RUNNING", "PENDING", "CONFIGURING", "COMPLETING", "SUBMITTED", "QUEUED")
    now = time.time() if now is None else now
    return now - snap.mtime_ns / 1e9 <= live_minutes * 60.0


def state(path, live_minutes: float = LIVE_MINUTES, job_state: Optional[str] = None,
          now: Optional[float] = None) -> dict:
    """{live, finished, age_s, snap}: what a live document shows in its chip."""
    snap = snapshot(path)
    tail = tail_text(path, snap)
    finished = has_terminal_marker(tail)
    now = time.time() if now is None else now
    return {"live": (not finished) and is_live(path, snap, tail, job_state, live_minutes, now),
            "finished": finished, "age_s": max(0.0, now - snap.mtime_ns / 1e9), "snap": snap.token(),
            "size": snap.size, "job_state": job_state}
