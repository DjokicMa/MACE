"""One place for MACE's SQLite connection policy (PLAN_analysis C14).

Why this exists: a materials.db is written by the SLURM completion callback
of every job (on compute nodes), by ``mace`` on the login node and by the GUI
on a dev node. SQLite's WAL mode keeps its index in a shared-memory file
(``-shm``) and is documented as unsafe when the processes are on different
hosts or the database is on a network file system. So:

* **Journal mode follows the file system** (``MACE_DB_JOURNAL=auto``, the
  default): WAL on a local disk (unchanged), the rollback journal
  (``DELETE`` + ``synchronous=FULL``) on a network or cluster file system
  (NFS, GPFS, BeeGFS, CephFS, CIFS/SMB, ... - see NETWORK_FS). ``MACE_DB_JOURNAL=wal``
  or ``delete`` forces either mode. Set it in ``~/.bashrc`` or not at all:
  every process that opens the database must agree (``sbatch`` passes the
  submitting shell's environment on to the job and its callback).
* **Rollback mode needs POSIX locks that work across hosts.** That depends
  on the mount options, so they are read too (``lock_support``). On a mount
  without cluster-wide locks (``localflock``/``noflock`` mounts, NFS
  ``nolock`` or ``local_lock=posix|all``, CIFS ``nobrl``, BeeGFS, FUSE),
  MACE's writers also serialize on an O_EXCL lock file
  ``<db>.mace-writer.lock`` (degraded mode), and say so once.
  ``MACE_DB_LOCKING=trust`` marks such a mount reliable (a site that knows
  its BeeGFS has global locks); ``MACE_DB_LOCKING=degraded`` forces the
  lock file anywhere.
* **Backups** use the SQLite backup API from a read-only connection, into a
  new file created with O_EXCL, checked with ``PRAGMA quick_check``;
  large databases are copied in steps so a writer never waits long.
* **Retries** around whole idempotent transactions only (a commit, one
  file's extraction), decided by the SQLite error code.

Standard library only; nothing here changes a database on a local disk with
no environment variable set (golden test: still WAL).
"""
from __future__ import annotations

import contextlib
import os
import random
import re
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import quote
from typing import Callable, Iterator, List, Optional, Tuple

# file systems whose locks and shared memory are not one host's: SQLite's WAL
# mode is not safe on them
NETWORK_FS = {"nfs", "nfs4", "gpfs", "lustre", "beegfs",  # british-ok: "lustre" is a file system type
              "cephfs", "ceph", "panfs", "wekafs", "cifs", "smb3",
              "smbfs", "9p", "fuse", "fuseblk", "afs", "glusterfs", "orangefs", "pvfs2"}
BUSY_CODES = (5, 6)                       # SQLITE_BUSY, SQLITE_LOCKED (and their extended forms)
SQLITE_READONLY_ROLLBACK = 776
SQLITE_READONLY_DIRECTORY = 1544
SQLITE_READONLY_CANTINIT = 1288
SQLITE_READONLY_RECOVERY = 264
SQLITE_CANTOPEN = 14
WRITER_LOCK_STALE_S = 120.0
ONE_STEP_MAX = 64 * 1024 * 1024           # a backup of a DB up to this size is one step
STEP_BYTES = 16 * 1024 * 1024             # larger ones: this much per step
BACKUP_NAME = re.compile(r"^(materials|structures)_backup_(\d{8}_\d{6})(?:_(\d+))?\.db$")

_said: set = set()
_said_lock = threading.Lock()


def _once(key: str, msg: str):
    with _said_lock:
        if key in _said:
            return
        _said.add(key)
    print(msg, file=sys.stderr)


# ------------------------------------------------------------------ mounts
@dataclass(frozen=True)
class Mount:
    mount_point: str
    fstype: str
    mount_opts: Tuple[str, ...]
    super_opts: Tuple[str, ...]

    def has(self, opt: str) -> bool:
        return opt in self.mount_opts or opt in self.super_opts

    def value(self, key: str) -> Optional[str]:
        for o in self.mount_opts + self.super_opts:
            if o.startswith(key + "="):
                return o.split("=", 1)[1]
        return None


def _unescape(s: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), s)


def parse_mountinfo(text: str) -> List[Mount]:
    out = []
    for line in text.splitlines():
        parts = line.split()
        if "-" not in parts or len(parts) < 7:
            continue
        sep = parts.index("-")
        if sep + 1 >= len(parts):
            continue
        mp = _unescape(parts[4])
        mopts = tuple(parts[5].split(","))
        fstype = parts[sep + 1]
        sopts = tuple(parts[sep + 3].split(",")) if sep + 3 < len(parts) else ()
        out.append(Mount(mp, fstype, mopts, sopts))
    return out


_mounts_cache: List = [0.0, None]


def _system_mounts() -> List[Mount]:
    """/proc/self/mountinfo, parsed, re-read at most every 30 s (a
    MaterialDatabase opens a connection per call)."""
    now = time.monotonic()
    if _mounts_cache[1] is None or now - _mounts_cache[0] > 30.0:
        with open("/proc/self/mountinfo") as fh:
            _mounts_cache[1] = parse_mountinfo(fh.read())
        _mounts_cache[0] = now
    return _mounts_cache[1]


def mount_of(path, mountinfo: Optional[str] = None) -> Optional[Mount]:
    """The mount holding ``path`` (longest mount-point prefix of its realpath),
    from /proc/self/mountinfo. None off Linux or on any read error (treated
    as a local disk, so macOS and Windows are unchanged)."""
    try:
        if mountinfo is None:
            mounts = _system_mounts()
        else:
            mounts = parse_mountinfo(mountinfo)
        rp = os.path.realpath(os.path.abspath(str(path)))     # resolves existing symlinks; fine for new files
    except Exception:
        return None
    best = None
    for m in mounts:
        mp = m.mount_point.rstrip("/") or "/"
        if rp == mp or rp.startswith(mp + "/") or mp == "/":
            if best is None or len(mp) >= len(best.mount_point.rstrip("/") or "/"):
                best = m
    return best


def fs_type(path, mountinfo: Optional[str] = None) -> Optional[str]:
    m = mount_of(path, mountinfo)
    return m.fstype if m else None


def is_network(m: Optional[Mount]) -> bool:
    return m is not None and (m.fstype in NETWORK_FS or m.fstype.startswith("fuse."))


def lock_support(m: Optional[Mount]) -> str:
    """"local", "reliable" or "unreliable" (cross-host POSIX locks), then the
    ``MACE_DB_LOCKING`` override (trust | degraded)."""
    env = (os.environ.get("MACE_DB_LOCKING") or "").strip().lower()
    if env == "degraded":
        return "unreliable"
    if not is_network(m):
        return "local"
    t = m.fstype
    if t in ("nfs", "nfs4"):
        ll = m.value("local_lock")
        res = "unreliable" if m.has("nolock") or ll in ("posix", "all") else "reliable"
    elif t == "lustre":           # british-ok: a file system type
        res = "reliable" if m.has("flock") and not (m.has("localflock") or m.has("noflock")) else "unreliable"
    elif t in ("gpfs", "cephfs", "ceph"):
        res = "reliable"
    elif t in ("cifs", "smb3", "smbfs"):
        res = "unreliable" if m.has("nobrl") else "reliable"
    else:                    # beegfs (global locks are a client setting), panfs, wekafs, 9p, fuse, fuse.*
        res = "unreliable"
    if res == "unreliable" and env == "trust":
        return "reliable"
    return res


def journal_policy(path, mountinfo: Optional[str] = None) -> Tuple[str, bool]:
    """("wal" | "delete", degraded) for a database at ``path``."""
    env = (os.environ.get("MACE_DB_JOURNAL") or "auto").strip().lower()
    m = mount_of(path, mountinfo)
    support = lock_support(m)
    degraded = support == "unreliable"
    if env == "wal":
        if is_network(m):
            _once("forced-wal", f"MACE_DB_JOURNAL=wal: {Path(path).name} is on {m.fstype}, where SQLite's WAL "
                                "mode is not safe across nodes.")
        return "wal", False
    if env == "delete":
        return "delete", degraded
    if is_network(m) or degraded:
        if degraded and (os.environ.get("MACE_DB_LOCKING") or "").strip().lower() == "degraded":
            _once("degraded-forced", f"MACE_DB_LOCKING=degraded: MACE serializes the writers of {Path(path).name} "
                                     "with a lock file.")
        elif degraded:
            opt = next((o for o in ("localflock", "noflock", "nolock", "local_lock=posix", "local_lock=all", "nobrl")
                        if m is not None and (o in m.mount_opts or o in m.super_opts)), "no cluster-wide lock option")
            _once("degraded:" + (m.mount_point if m else ""),
                  f"{Path(path).name} is on {m.fstype if m else 'this file system'} mounted without cluster-wide "
                  f"locks ({opt}); MACE serializes its writers with a lock file. Set MACE_DB_LOCKING=trust if "
                  "your site enables global locks.")
        return "delete", degraded
    return "wal", False


# ------------------------------------------------------------------ connections
def error_code(exc: BaseException) -> Optional[int]:
    """SQLite's extended result code of an sqlite3 error (Python >= 3.11)."""
    return getattr(exc, "sqlite_errorcode", None)


def is_busy(exc: BaseException) -> bool:
    code = error_code(exc)
    if code is not None:
        return (code & 0xFF) in BUSY_CODES
    # Python < 3.11 has no error codes: SQLite's own two busy messages
    msg = str(exc).lower()
    return isinstance(exc, sqlite3.OperationalError) and ("database is locked" in msg or "database table is locked" in msg)


def configure(conn: sqlite3.Connection, path) -> str:
    """Apply the journal policy to a fresh connection; returns the mode in
    use ("wal" or "delete"). Never raises because of the policy."""
    mode, _ = journal_policy(path)
    try:
        if mode == "wal":
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=30000")
            return "wal"
        conn.execute("PRAGMA busy_timeout=30000")
        cur = (conn.execute("PRAGMA journal_mode").fetchone() or ["delete"])[0].lower()
        got = cur
        if cur == "wal":
            # the switch needs the database to itself: try briefly, never wait behind another process
            conn.execute("PRAGMA busy_timeout=200")
            try:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                got = (conn.execute("PRAGMA journal_mode=DELETE").fetchone() or ["wal"])[0].lower()
            except sqlite3.Error:
                got = "wal"
            conn.execute("PRAGMA busy_timeout=30000")
            if got == "wal":
                fs = fs_type(path) or "this file system"
                _once("held-wal:" + str(path),
                      f"{Path(path).name} is on a network file system ({fs}) but another process holds it in WAL "
                      "mode; MACE switches it to rollback mode when it is free.")
                conn.execute("PRAGMA synchronous=NORMAL")
                return "wal"
        elif cur != "delete":
            got = (conn.execute("PRAGMA journal_mode=DELETE").fetchone() or [cur])[0].lower()
        conn.execute("PRAGMA synchronous=FULL")
        return "delete" if got == "delete" else got
    except sqlite3.Error:
        return mode


def connect_rw(path, timeout: float = 30.0) -> sqlite3.Connection:
    """A read-write connection under the policy (for raw writers that do not
    go through MaterialDatabase)."""
    conn = sqlite3.connect(str(path), timeout=timeout, check_same_thread=False)
    configure(conn, path)
    return conn


@contextlib.contextmanager
def immediate(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """BEGIN IMMEDIATE ... COMMIT (ROLLBACK on error): the write lock is taken
    up front, so the busy handler waits instead of a deferred transaction
    failing when it upgrades."""
    if conn.in_transaction:
        conn.commit()
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
        conn.execute("COMMIT")
    except BaseException:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        raise


def retry_locked(fn: Callable, budget_s: float = 120.0, base: float = 0.5, cap: float = 10.0,
                 on_retry: Optional[Callable[[int, BaseException], None]] = None,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic):
    """Run ``fn()``; on a busy/locked SQLite error wait (jittered exponential
    backoff) and run it again, for at most ``budget_s``. Only for whole
    idempotent transactions. Other errors pass through at once."""
    t0 = clock()
    attempt = 0
    while True:
        attempt += 1
        try:
            return fn()
        except sqlite3.Error as exc:
            if not is_busy(exc):
                raise
            delay = min(cap, base * (2 ** (attempt - 1))) * (0.5 + random.random() / 2)
            if clock() - t0 + delay > budget_s:
                raise sqlite3.OperationalError(f"{exc} (gave up after {attempt} attempts in "
                                               f"{clock() - t0:.0f} s)") from exc
            if on_retry is not None:
                on_retry(attempt, exc)
            sleep(delay)


_held: dict = {}
_held_lock = threading.RLock()


@contextlib.contextmanager
def writer_lock(path, stale_s: float = WRITER_LOCK_STALE_S, wait_s: float = 300.0,
                poll: float = 0.05) -> Iterator[Path]:
    """The degraded-mode writer lock: O_EXCL ``<db>.mace-writer.lock`` (atomic
    on NFSv3+ and on the metadata servers of parallel file systems), taken over when older
    than ``stale_s`` by mtime, removed on exit."""
    lock = Path(str(path) + ".mace-writer.lock")
    key = str(lock)
    with _held_lock:
        if _held.get(key, 0) > 0:          # this process already holds it (nested connection)
            _held[key] += 1
            nested = True
        else:
            nested = False
    if nested:
        try:
            yield lock
        finally:
            with _held_lock:
                _held[key] -= 1
        return
    t0 = time.monotonic()
    while True:
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            os.write(fd, f"{os.getpid()} {os.uname().nodename if hasattr(os, 'uname') else ''}\n".encode())
            os.close(fd)
            break
        except FileExistsError:
            try:
                age = time.time() - lock.stat().st_mtime
            except FileNotFoundError:
                continue
            if age > stale_s:
                try:                       # a crashed writer's lock: take it over (rename is atomic)
                    tmp = lock.with_name(lock.name + f".stale.{os.getpid()}")
                    os.rename(lock, tmp)
                    os.unlink(tmp)
                except OSError:
                    pass
                continue
            if time.monotonic() - t0 > wait_s:
                raise TimeoutError(f"{lock.name} is held by another MACE writer (waited {wait_s:.0f} s)")
            time.sleep(poll)
    with _held_lock:
        _held[key] = 1
    try:
        yield lock
    finally:
        with _held_lock:
            _held[key] = 0
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


@contextlib.contextmanager
def write_guard(path) -> Iterator[None]:
    """The writer lock in degraded mode, nothing otherwise."""
    _, degraded = journal_policy(path)
    if degraded:
        with writer_lock(path):
            yield
    else:
        yield


# ------------------------------------------------------------------ backups
class BackupTimeout(RuntimeError):
    pass


def _new_backup_file(dst_dir: Path, stem: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    n = 1
    while True:
        name = f"{stem}_{ts}.db" if n == 1 else f"{stem}_{ts}_{n}.db"
        p = dst_dir / name
        try:
            fd = os.open(str(p), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            os.close(fd)
            return p
        except FileExistsError:
            n += 1


def snapshot_backup(src, dst_dir, stem: str = "materials_backup", deadline_s: float = 120.0,
                    one_step_max: int = ONE_STEP_MAX, step_bytes: int = STEP_BYTES,
                    sleep_s: float = 0.05) -> Path:
    """A consistent copy of the SQLite database ``src`` in ``dst_dir`` as
    ``<stem>_YYYYMMDD_HHMMSS[_n].db``. The source is opened read-only (never
    through MaterialDatabase: no migrations, no side files). Up to
    ``one_step_max`` bytes it is copied in one step; larger ones in steps
    of ``step_bytes`` so a writer waits well under a second per step. A copy
    that cannot finish within ``deadline_s`` (the source keeps changing) is
    deleted and BackupTimeout raised; a copy that fails ``quick_check`` is
    deleted and the error raised."""
    src = Path(src)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    if not src.is_file():
        raise FileNotFoundError(str(src))
    dst = _new_backup_file(dst_dir, stem)
    t0 = time.monotonic()
    restarts = [0]
    last_remaining = [None]
    try:
        # quoted: a "#" or "?" in the path would otherwise end it, and SQLite would
        # open (and create) a different, empty file read-write and back that up
        s = sqlite3.connect(f"file:{quote(str(src.resolve()))}?mode=ro", uri=True, timeout=30)
        d = sqlite3.connect(str(dst))
        try:
            size = src.stat().st_size
            if size <= one_step_max:
                s.backup(d)
            else:
                page = s.execute("PRAGMA page_size").fetchone()[0] or 4096

                def progress(status, remaining, total):
                    if last_remaining[0] is not None and remaining > last_remaining[0]:
                        restarts[0] += 1
                    last_remaining[0] = remaining
                    if time.monotonic() - t0 > deadline_s:
                        raise BackupTimeout(
                            f"{src.name} was being written continuously and the backup did not finish in "
                            f"{deadline_s:.0f} s; nothing was written. Try again when fewer jobs are finishing.")
                s.backup(d, pages=max(1, step_bytes // page), progress=progress, sleep=sleep_s)
            ok = d.execute("PRAGMA quick_check").fetchone()[0]
            if ok != "ok":
                raise sqlite3.DatabaseError(f"the backup copy of {src.name} failed quick_check: {ok}")
            try:
                d.execute("PRAGMA journal_mode=DELETE")     # a standalone file, no -wal beside it
            except sqlite3.Error:
                pass
        finally:
            d.close()
            s.close()
    except BaseException:
        for q in (dst, Path(str(dst) + "-journal"), Path(str(dst) + "-wal"), Path(str(dst) + "-shm")):
            try:
                q.unlink()
            except FileNotFoundError:
                pass
        raise
    snapshot_backup.last_restarts = restarts[0]
    return dst


snapshot_backup.last_restarts = 0


def prune_backups(dst_dir, keep: int = 3, out: Optional[Callable[[str], None]] = print) -> List[str]:
    """Keep the newest ``keep`` files of each MACE backup kind in ``dst_dir``
    (exactly ``materials_backup_YYYYMMDD_HHMMSS[_n].db`` and the same for
    ``structures_backup``); other files are never touched. Returns the
    pruned names, each printed through ``out``."""
    dst_dir = Path(dst_dir)
    if keep < 1 or not dst_dir.is_dir():
        return []
    groups = {}
    for p in dst_dir.iterdir():
        m = BACKUP_NAME.match(p.name)
        if m and p.is_file():
            groups.setdefault(m.group(1), []).append((m.group(2), int(m.group(3) or 1), p))
    pruned = []
    for kind, items in groups.items():
        items.sort(key=lambda t: (t[0], t[1]), reverse=True)
        for _, _, p in items[keep:]:
            try:
                p.unlink()
                pruned.append(p.name)
                if out:
                    out(f"Pruned old backup {p.name}")
            except OSError:
                pass
    return sorted(pruned)
