"""The one cached, rate-limited runner for every cluster command.

Every SLURM, quota, df and ssh probe MACE's cluster views run goes through
``run(key, argv, ...)``:

* **Cache per key** in ``$XDG_CACHE_HOME/mace/cluster/<cluster>/<key>.json``
  (dir 0700, files 0600, written atomically, one file per key overwritten in
  place). Within ``ttl`` the cached value is returned and nothing runs.
* **Refresh floor.** ``refresh=True`` (the CLI's ``--refresh``, the GUI's
  Refresh) skips the TTL but never the floor: within ``refresh_floor``
  seconds of the last fetch the cached value comes back with
  ``next_allowed_in``.
* **Single flight.** One fetch per key at a time across threads (a lock per
  key) and processes (``fcntl.flock`` on ``<key>.lock``, taken with
  ``LOCK_NB`` in a retry loop of at most ``timeout + 5`` seconds; then the
  stale value comes back with "Another MACE process is reading <key>").
  After the lock is taken the cache is read again, so 50 threads and 3
  processes asking at once make one call. Without ``fcntl`` (Windows) only
  the thread lock applies.
* **Backoff.** A failing command is retried after 30, 60, 120 ... 300 s; in
  between, the last good value comes back marked stale with the error.
* **Stale-while-revalidate** (``background=True``, the GUI server): past the
  TTL the cached value comes back at once (``stale``) and one background
  thread refreshes it.
* **Fresh calls** (``fresh=True``) always run and are never cached: only for
  a cancel guard or a preview, one per user action.
* **Fetch log**: the last 50 fetches as ``[timestamp, tag]``, tag ``poll`` |
  ``refresh`` | ``cli`` | ``fresh`` | ``revalidate`` (what the rate tests and
  the HPCC check count).

The child's environment is ``slurm_env(outside_allocation_env(...))``:
inside an allocation every ``SLURM_*`` but ``SLURM_CONF`` is dropped. It
runs with stdin from /dev/null, a 30 s timeout, niceness +10 and a neutral
working directory.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from mace.queue.slurm_ids import outside_allocation_env, slurm_env

LOG_KEEP = 50
BACKOFF_MIN = 30.0
BACKOFF_MAX = 300.0
DEFAULT_TIMEOUT = 30.0
LOCK_EXTRA = 5.0
TAGS = ("poll", "refresh", "cli", "fresh", "revalidate")

_thread_locks: Dict[str, threading.Lock] = {}
_thread_locks_guard = threading.Lock()
_revalidating: Dict[str, bool] = {}
_which_cache: Dict[str, Optional[str]] = {}
_name_cache: Dict[str, Any] = {}


@dataclass
class Result:
    value: Any = None
    fetched_at: Optional[float] = None
    stale: bool = False
    error: Optional[str] = None
    source: str = "cache"            # "cache" | "fetch" | "fresh" | "none"
    next_allowed_in: float = 0.0
    note: Optional[str] = None
    key: str = ""

    @property
    def age(self) -> Optional[float]:
        return None if self.fetched_at is None else max(0.0, now() - self.fetched_at)

    def as_dict(self) -> Dict[str, Any]:
        return {"fetched_at": self.fetched_at, "age": None if self.age is None else round(self.age, 1),
                "stale": self.stale, "error": self.error, "source": self.source,
                "next_allowed_in": round(self.next_allowed_in, 1), "note": self.note}


# ------------------------------------------------------------------ clock
def now() -> float:
    """time.time(), or the test clock (only with MACE_GUI_TEST_HOOKS=1):
    ``MACE_CLUSTER_TEST_CLOCK`` is a number (a fixed time) or a file holding
    one (a clock a test moves)."""
    if os.environ.get("MACE_GUI_TEST_HOOKS") == "1":
        spec = os.environ.get("MACE_CLUSTER_TEST_CLOCK")
        if spec:
            try:
                return float(spec)
            except ValueError:
                try:
                    return float(Path(spec).read_text().strip())
                except (OSError, ValueError):
                    pass
    return time.time()


# ------------------------------------------------------------------ places
def cache_root(env: Optional[Dict[str, str]] = None) -> Path:
    env = os.environ if env is None else env
    base = env.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(base) / "mace" / "cluster"


def _private_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(p, 0o700)
    except OSError:
        pass
    return p


def available(tool: str) -> Optional[str]:
    """Absolute path of a tool on PATH (cached per process and PATH)."""
    key = f"{tool}\0{os.environ.get('PATH', '')}"
    if key not in _which_cache:
        _which_cache[key] = shutil.which(tool)
    return _which_cache[key]


def _safe(name: str) -> str:
    keep = "".join(c if (c.isalnum() or c in "._-") else "_" for c in name)[:80]
    if keep != name:
        keep = f"{keep}-{hashlib.sha1(name.encode()).hexdigest()[:8]}"
    return keep or "default"


def cluster_name() -> str:
    """``ClusterName`` from ``scontrol show config`` (cached one day), else
    the domain of this host, else "local"."""
    root = cache_root()
    ck = str(root)
    if ck in _name_cache:
        return _name_cache[ck]
    name = None
    f = root / "cluster-name.json"
    try:
        data = json.loads(f.read_text())
        if now() - float(data.get("at", 0)) < 86400 and data.get("name"):
            name = str(data["name"])
    except (OSError, ValueError):
        pass
    if name is None and available("scontrol"):
        try:
            res = subprocess.run([available("scontrol"), "show", "config"], capture_output=True, text=True,
                                 timeout=DEFAULT_TIMEOUT, stdin=subprocess.DEVNULL,
                                 env=slurm_env(outside_allocation_env()), cwd=_neutral_cwd())
            for line in (res.stdout or "").splitlines():
                if line.strip().lower().startswith("clustername"):
                    name = line.split("=", 1)[1].strip() or None
                    break
        except (OSError, subprocess.SubprocessError):
            name = None
        if name:
            try:
                _private_dir(root)
                _atomic_write(f, {"name": name, "at": now()})
            except OSError:
                pass
    if not name:
        fq = socket.getfqdn()
        name = fq.split(".", 1)[1] if "." in fq else "local"
    _name_cache[ck] = _safe(name)
    return _name_cache[ck]


def cache_dir() -> Path:
    return _private_dir(cache_root() / cluster_name())


def _neutral_cwd() -> str:
    # never the served folder or the shell's folder (a mace/queue there would shadow modules)
    return tempfile.gettempdir()


# ------------------------------------------------------------------ files
def _atomic_write(path: Path, data: Dict[str, Any]) -> None:
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        os.fchmod(fd, 0o600) if hasattr(os, "fchmod") else None
        with os.fdopen(fd, "w") as fh:
            json.dump(data, fh)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _load(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def read_cache(key: str) -> Dict[str, Any]:
    """The raw cache record of a key ({} when none): value, fetched_at,
    failures, next_try, error, log."""
    return _load(cache_dir() / f"{_safe(key)}.json")


def fetch_log(key: str) -> List[List[Any]]:
    return list(read_cache(key).get("log") or [])


def _log_append(rec: Dict[str, Any], tag: str) -> None:
    log = list(rec.get("log") or [])
    log.append([round(now(), 3), tag])
    rec["log"] = log[-LOG_KEEP:]


# ------------------------------------------------------------------ locks
def _thread_lock(key: str) -> threading.Lock:
    with _thread_locks_guard:
        lk = _thread_locks.get(key)
        if lk is None:
            lk = _thread_locks[key] = threading.Lock()
        return lk


class _FileLock:
    """flock on <key>.lock with a bounded wait; a no-op without fcntl."""

    def __init__(self, path: Path, wait: float):
        self.path, self.wait, self.fh, self.ok = path, wait, None, False

    def __enter__(self):
        try:
            import fcntl
        except ImportError:
            self.ok = True
            return self
        self.fh = open(self.path, "a")
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass
        deadline = time.monotonic() + self.wait
        while True:
            try:
                fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.ok = True
                return self
            except OSError:
                if time.monotonic() >= deadline:
                    return self
                time.sleep(0.05)

    def __exit__(self, *exc):
        if self.fh is not None:
            try:
                if self.ok:
                    import fcntl
                    fcntl.flock(self.fh, fcntl.LOCK_UN)
            finally:
                self.fh.close()
        return False


# ------------------------------------------------------------------ running
@dataclass
class Proc:
    returncode: int
    stdout: str
    stderr: str


def _nice():
    try:
        os.nice(10)
    except OSError:
        pass


def execute(argv: Sequence[str], env: Optional[Dict[str, str]] = None, timeout: float = DEFAULT_TIMEOUT,
            input_text: Optional[str] = None, cwd: Optional[str] = None) -> Proc:
    """Run one command the way every cluster query runs (no cache)."""
    argv = [str(a) for a in argv]
    exe = argv[0] if os.path.isabs(argv[0]) else available(argv[0])
    if not exe:
        return Proc(127, "", f"{argv[0]} not found")
    full_env = slurm_env(outside_allocation_env(env))
    kw: Dict[str, Any] = {}
    if os.name == "posix":
        kw["preexec_fn"] = _nice
    try:
        res = subprocess.run([exe] + argv[1:], capture_output=True, text=True, timeout=timeout,
                             input=input_text, stdin=None if input_text is not None else subprocess.DEVNULL,
                             env=full_env, cwd=cwd or _neutral_cwd(), **kw)
    except subprocess.TimeoutExpired:
        return Proc(124, "", f"{os.path.basename(argv[0])} did not answer within {int(timeout)} s")
    except (OSError, subprocess.SubprocessError) as exc:
        return Proc(126, "", f"{os.path.basename(argv[0])}: {exc}")
    return Proc(res.returncode, res.stdout or "", res.stderr or "")


def _first_line(text: str) -> str:
    for line in (text or "").splitlines():
        if line.strip():
            return line.strip()[:300]
    return "no output"


def _result_from(rec: Dict[str, Any], key: str, stale: bool, source: str = "cache",
                 note: Optional[str] = None, next_allowed: float = 0.0) -> Result:
    return Result(value=rec.get("value"), fetched_at=rec.get("fetched_at"), stale=stale,
                  error=rec.get("error"), source=source if "fetched_at" in rec else "none",
                  next_allowed_in=next_allowed, note=note, key=key)


def _fetch(key: str, argv, env, timeout, parse, accept, input_text, cwd, tag) -> Result:
    """Run and store (the caller holds the locks). ``argv`` may be a callable
    returning a Proc (a chain of commands cached as one key)."""
    path = cache_dir() / f"{_safe(key)}.json"
    rec = _load(path)
    if callable(argv):
        try:
            proc = argv()
        except Exception as exc:                       # pragma: no cover - a fetcher bug
            proc = Proc(1, "", f"{type(exc).__name__}: {exc}")
    else:
        proc = execute(argv, env=env, timeout=timeout, input_text=input_text, cwd=cwd)
    _log_append(rec, tag)
    ok = proc.returncode == 0 or (accept is not None and accept(proc))
    if ok:
        try:
            value = parse(proc) if parse else proc.stdout
            rec.update(value=value, fetched_at=now(), failures=0, next_try=0, error=None)
        except Exception as exc:                       # a parser bug never breaks the view
            ok = False
            proc = Proc(proc.returncode, proc.stdout, f"could not read the output: {type(exc).__name__}: {exc}")
    if not ok:
        failures = int(rec.get("failures") or 0) + 1
        wait = min(BACKOFF_MAX, BACKOFF_MIN * 2 ** (failures - 1))
        rec.update(failures=failures, next_try=now() + wait,
                   error=_first_line(proc.stderr or proc.stdout))
    try:
        _atomic_write(path, rec)
    except OSError:
        pass
    if ok:
        return _result_from(rec, key, stale=False, source="fetch")
    res = _result_from(rec, key, stale=True)
    res.error = rec["error"]
    return res


def run(key: str, argv: Sequence[str], *, ttl: float, refresh_floor: Optional[float] = None,
        min_interval: Optional[float] = None, fresh: bool = False, refresh: bool = False,
        tag: Optional[str] = None, env: Optional[Dict[str, str]] = None,
        parse: Optional[Callable[[Proc], Any]] = None, accept: Optional[Callable[[Proc], bool]] = None,
        timeout: float = DEFAULT_TIMEOUT, background: bool = False, input_text: Optional[str] = None,
        cwd: Optional[str] = None) -> Result:
    """The cached value of ``key`` (see the module doc).

    ``argv`` is a command, or a callable returning a ``Proc`` (several
    commands cached under one key). ``parse(proc)`` turns a successful run
    into a JSON-able value (default: stdout). ``accept(proc)`` says a non-zero exit is still an answer (an
    ``sbatch --test-only`` refusal). ``min_interval`` is another name for
    ``refresh_floor``."""
    floor = refresh_floor if refresh_floor is not None else (min_interval if min_interval is not None else 15.0)
    tag = tag or ("fresh" if fresh else "refresh" if refresh else "cli")
    if fresh:
        proc = argv() if callable(argv) else execute(argv, env=env, timeout=timeout, input_text=input_text, cwd=cwd)
        path = cache_dir() / f"{_safe(key)}.json"
        with _thread_lock(key):
            rec = _load(path)
            _log_append(rec, "fresh")
            try:
                _atomic_write(path, rec)
            except OSError:
                pass
        ok = proc.returncode == 0 or (accept is not None and accept(proc))
        if not ok:
            return Result(error=_first_line(proc.stderr or proc.stdout), source="fresh", key=key, stale=True)
        return Result(value=parse(proc) if parse else proc.stdout, fetched_at=now(), source="fresh", key=key)

    path = cache_dir() / f"{_safe(key)}.json"
    started = now()

    def decide(rec) -> Optional[Result]:
        """A cached answer, or None when a fetch is due."""
        has = "fetched_at" in rec
        age = started - float(rec["fetched_at"]) if has else None
        if has and rec.get("fetched_at", 0) >= started and not rec.get("error"):
            return _result_from(rec, key, stale=False)        # someone fetched while we waited
        if refresh and has and age is not None and age < floor:
            return _result_from(rec, key, stale=age >= ttl, next_allowed=floor - age)
        if not refresh and has and age is not None and age < ttl:
            return _result_from(rec, key, stale=False)
        if rec.get("next_try") and started < float(rec["next_try"]):
            res = _result_from(rec, key, stale=True)
            res.error = rec.get("error")
            res.next_allowed_in = float(rec["next_try"]) - started
            return res
        return None

    rec = _load(path)
    ans = decide(rec)
    if ans is not None:
        return ans
    if background and "fetched_at" in rec:
        _revalidate(key, argv, env, timeout, parse, accept, input_text, cwd, floor)
        return _result_from(rec, key, stale=True)
    with _thread_lock(key):
        with _FileLock(cache_dir() / f"{_safe(key)}.lock", timeout + LOCK_EXTRA) as lk:
            if not lk.ok:
                rec = _load(path)
                res = _result_from(rec, key, stale=True)
                age = res.age
                res.note = (f"Another MACE process is reading {key}; showing data from "
                            f"{int(age)} s ago" if age is not None else
                            f"Another MACE process is reading {key}; no data yet")
                return res
            rec = _load(path)
            ans = decide(rec)
            if ans is not None:
                return ans
            return _fetch(key, argv, env, timeout, parse, accept, input_text, cwd, tag)


def _revalidate(key, argv, env, timeout, parse, accept, input_text, cwd, floor) -> None:
    with _thread_locks_guard:
        if _revalidating.get(key):
            return
        _revalidating[key] = True

    def work():
        try:
            with _thread_lock(key):
                with _FileLock(cache_dir() / f"{_safe(key)}.lock", timeout + LOCK_EXTRA) as lk:
                    if not lk.ok:
                        return
                    rec = _load(cache_dir() / f"{_safe(key)}.json")
                    if "fetched_at" in rec and now() - float(rec["fetched_at"]) < floor:
                        return
                    if rec.get("next_try") and now() < float(rec["next_try"]):
                        return
                    _fetch(key, argv, env, timeout, parse, accept, input_text, cwd, "revalidate")
        finally:
            with _thread_locks_guard:
                _revalidating.pop(key, None)

    threading.Thread(target=work, name=f"mace-cluster-{key}", daemon=True).start()


def wait_revalidations(timeout: float = 10.0) -> None:
    """Tests: wait until no background refresh runs."""
    end = time.monotonic() + timeout
    while _revalidating and time.monotonic() < end:
        time.sleep(0.02)
