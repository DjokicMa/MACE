"""Small JSON files that several `mace gui` servers of one account may write
at once (gui.json, gui-docprefs.json in $XDG_CONFIG_HOME/mace).

Two servers (two folders, two ports, or two cluster nodes sharing $HOME on
NFS) can save close together. Every write is therefore a read-modify-write
under an advisory POSIX lock (``fcntl.lockf`` on ``<file>.lock``, which NFS
carries through its lock protocol; ``flock`` can stay host-local on NFS
without any error), then a temp file and ``os.replace`` (atomic), mode 0600.
A POSIX lock belongs to the process, not the thread, so the request threads
of one server also take a lock of this module per file first (without it
two saves at the same moment, e.g. a setting and a remembered choice, both
"held" the lock and the later write dropped the other's change).

Where locks do not work at all (``lockf`` raises, or a second process can
take a lock this one holds), writes go on without the lock and one line
says so. A lock that "works" but only on this host is caught afterwards:
the file is read back, and when another server's write replaced this one
(the revision or a patched value differs), the change is merged onto what
is on disk and written once more; if that loses too, ``Conflict`` is
raised and the page keeps the value in the browser ("saved in this browser
only"). The worst case is a change re-applied or reported, never a broken
file.
"""
from __future__ import annotations

import errno
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

LOCK_TIMEOUT = 5.0


class Busy(RuntimeError):
    """The lock was held longer than LOCK_TIMEOUT (another server is saving)."""


class Conflict(RuntimeError):
    """Another server's write replaced this one twice in a row."""


_probe_cache: Dict[str, bool] = {}
_probe_lock = threading.Lock()
# one writer per file within this process (the POSIX lock only orders processes)
_thread_locks: Dict[str, "threading.RLock"] = {}
_thread_locks_guard = threading.Lock()


def _thread_lock(path: Path) -> "threading.RLock":
    key = os.path.abspath(str(path))
    with _thread_locks_guard:
        lk = _thread_locks.get(key)
        if lk is None:
            lk = _thread_locks[key] = threading.RLock()
        return lk
_warned = set()
# the verify-after-write retry, counted (tests and the HPCC check read it)
stats = {"retries": 0, "conflicts": 0, "unlocked_writes": 0}
# a test hook: called between os.replace and the read-back (simulates a host-local lock)
_after_replace: Optional[Callable[[Path], None]] = None

_PROBE = r"""
import fcntl, os, sys
fd = os.open(sys.argv[1], os.O_RDWR)
try:
    fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
except OSError:
    sys.exit(1)        # blocked: locks work
sys.exit(0)            # got a lock another process holds: they do not
"""


def locks_work(directory: Path) -> bool:
    """Whether POSIX locks in `directory` keep a second process out. Checked
    once per directory per process (about 30 ms: one short child process)."""
    key = str(directory)
    with _probe_lock:
        if key in _probe_cache:
            return _probe_cache[key]
        ok = False
        probe = directory / ".mace-lock-probe"
        fd = None
        try:
            import fcntl
            fd = os.open(str(probe), os.O_RDWR | os.O_CREAT, 0o600)
            fcntl.lockf(fd, fcntl.LOCK_EX)
            r = subprocess.run([sys.executable, "-I", "-c", _PROBE, str(probe)], capture_output=True, timeout=20)
            ok = r.returncode == 1
        except (OSError, ImportError, subprocess.SubprocessError):
            ok = False
        finally:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
            try:
                probe.unlink()
            except OSError:
                pass
        _probe_cache[key] = ok
        if not ok and key not in _warned:
            _warned.add(key)
            print(f"mace gui: file locks do not work in {directory}; settings are saved without them "
                  "(two servers saving at the same moment may need a second try)", flush=True)
        return ok


class _Lock:
    def __init__(self, path: Path, use: bool):
        self.path, self.use, self.fd = path, use, None

    def __enter__(self):
        if not self.use:
            return self
        import fcntl
        self.fd = os.open(str(self.path), os.O_RDWR | os.O_CREAT, 0o600)
        end = time.monotonic() + LOCK_TIMEOUT
        while True:
            try:
                fcntl.lockf(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except OSError as e:
                if e.errno not in (errno.EAGAIN, errno.EACCES):
                    # the lock itself is broken here: carry on without it, once said
                    os.close(self.fd)
                    self.fd, self.use = None, False
                    _probe_cache[str(self.path.parent)] = False
                    return self
                if time.monotonic() > end:
                    os.close(self.fd)
                    self.fd = None
                    raise Busy("profile busy, try again")
                time.sleep(0.02)

    def __exit__(self, *exc):
        if self.fd is not None:
            try:
                import fcntl
                fcntl.lockf(self.fd, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(self.fd)
            self.fd = None
        return False


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    """The file's object; None when absent; ValueError when unreadable JSON."""
    try:
        text = path.read_text()
    except FileNotFoundError:
        return None
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("not a JSON object")
    return data


def write_json(path: Path, data: Dict[str, Any], max_bytes: int) -> None:
    text = json.dumps(data, indent=2, sort_keys=True)
    if len(text) > max_bytes:
        raise ValueError("too large")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text + "\n")
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def update(path: Path, change: Callable[[Optional[Dict[str, Any]]], Dict[str, Any]],
           holds: Callable[[Dict[str, Any]], bool], max_bytes: int) -> Dict[str, Any]:
    """Read, change, write `path` under the lock, then check it holds.

    `change(current_or_None)` returns the new object (it must bump "rev");
    `holds(on_disk)` says whether this change survived on disk. One re-merge
    on a lost write, then Conflict."""
    path.parent.mkdir(parents=True, exist_ok=True)
    use = locks_work(path.parent)
    if not use:
        stats["unlocked_writes"] += 1
    with _thread_lock(path):
        return _update_locked(path, change, holds, max_bytes, use)


def _update_locked(path: Path, change: Callable[[Optional[Dict[str, Any]]], Dict[str, Any]],
                   holds: Callable[[Dict[str, Any]], bool], max_bytes: int, use: bool) -> Dict[str, Any]:
    for attempt in range(2):
        with _Lock(path.with_name(path.name + ".lock"), use):
            try:
                cur = read_json(path)
            except ValueError:
                cur = None
            new = change(cur)
            write_json(path, new, max_bytes)
            if _after_replace is not None:
                _after_replace(path)
            try:
                back = read_json(path)
            except ValueError:
                back = None
        if back is not None and back.get("rev") == new.get("rev") and holds(back):
            return new
        stats["retries"] += 1
        if attempt == 0:
            print("mace gui: the profile changed by another server while saving; your change was re-applied", flush=True)
    stats["conflicts"] += 1
    raise Conflict("another server saved at the same moment; the change is kept in this browser")
