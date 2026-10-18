"""MACE's user-wide SLURM job budget: per-job admission, held jobs, drains.

Every sbatch MACE runs for a calculation goes through ``sbatch()`` here. It
takes a lock shared by all of this user's MACE processes (every workflow,
database and node: a POSIX lock on a file in the queue state folder), counts
the user's queued jobs (``squeue``: running + pending, each array task, plus
a short ledger of MACE's own submissions squeue does not list yet) and
submits only while the count stays within the budget:

    budget = the user's SLURM submit limit - reserve   ([queue] in site.toml)

A job that does not fit is not lost: its database row becomes ``held`` with
the sbatch command that submits it (``held_submit``), a reason
(``held_reason``: budget, workflow_cap or slurm_limit) and ``held_since``.
``drain()`` releases held rows of every active workflow (a small registry of
databases) into free slots, in the [queue] order. It runs after every
completion callback, from ``mace workflow --process``, ``mace queue drain``
and the GUI's watcher. When work is held and no MACE job is queued, the
drain submits one small sentinel job (``mace-drain``) that runs ``mace queue
drain`` again 15 minutes later.

A SLURM per-user limit refusal (QOSMaxSubmitJobPerUserLimit,
AssocMaxSubmitJobLimit, "job submit limit") holds the job (slurm_limit),
stops submitting for the rest of that pass and remembers the observed limit
for an hour. Any other sbatch error stays a failure, with SLURM's message.
"""
from __future__ import annotations

import contextlib
import contextvars
import errno
import fcntl
import json
import os
import re
import shlex
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

HELD = "held"
REASONS = ("budget", "workflow_cap", "slurm_limit")
REASON_WORDS = {"budget": "queue budget full", "workflow_cap": "the workflow's max_jobs reached",
                "slurm_limit": "SLURM refused it (submit limit)"}
LIVE_ROW_STATUSES = ("submitted", "pending", "queued", "running")
# squeue states SLURM counts against MaxSubmitJobs
LIVE_SLURM_STATES = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED", "REQUEUED",
                     "REQUEUE_HOLD", "REQUEUE_FED", "RESIZING", "SIGNALING", "STAGE_OUT", "STOPPED"}
# SLURM's reason codes for a per-user (or per-account) submit limit
LIMIT_TOKEN_RE = re.compile(r"QOSMaxSubmitJob|AssocMaxSubmitJob|QOSMaxSubmitJobsPerAccount|MaxSubmitJobs", re.I)
# the generic accounting-policy text; SLURM prints it for a submit limit and
# also for other policy limits (size, time), so it alone is weaker evidence
LIMIT_TEXT_RE = re.compile(r"job submit limit", re.I)
SUBMITTED_RE = re.compile(r"Submitted batch job (\d+)")
SENTINEL_NAME = "mace-drain"
SENTINEL_BEGIN_MIN = 15
# an unreadable queue trusts a sentinel marker this long: begin + run + slack
SENTINEL_MARKER_S = (SENTINEL_BEGIN_MIN + 10) * 60
LEDGER_WINDOW_S = 120          # a submission squeue may not list yet
UNKNOWN_RETRY_S = 3            # squeue failed: read the queue once more after this
OBSERVED_TTL_S = 3600          # a limit learned from a refusal
PHANTOM_AGE_S = 600            # an old 'pending' row without a job id
LOCK_TIMEOUT_S = 120
FALLBACK = (950, 50)           # the built-in callback numbers: 900 queued
SHIM_MESSAGE = "MACE_QUEUE_SBATCH_SHIM"


# ------------------------------------------------------------------ state folder
def state_dir(env: Optional[Dict[str, str]] = None) -> Path:
    """MACE_QUEUE_STATE_DIR, else $XDG_CACHE_HOME/mace/queue, else
    ~/.cache/mace/queue (one per user, on the shared home of a cluster)."""
    env = os.environ if env is None else env
    if env.get("MACE_QUEUE_STATE_DIR"):
        d = Path(env["MACE_QUEUE_STATE_DIR"])
    else:
        base = env.get("XDG_CACHE_HOME") or os.path.join(env.get("HOME") or str(Path.home()), ".cache")
        d = Path(base) / "mace" / "queue"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _read_json(path: Path, default):
    try:
        return json.loads(Path(path).read_text() or "null") or default
    except (OSError, ValueError):
        return default


def _write_json(path: Path, data) -> None:
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True))
    os.replace(tmp, path)


@contextlib.contextmanager
def _file_lock(path: Path):
    """A short POSIX lock of its own for a small JSON file (registry)."""
    fd = os.open(str(path) + ".lock", os.O_RDWR | os.O_CREAT, 0o666)
    try:
        try:
            fcntl.lockf(fd, fcntl.LOCK_EX)
        except OSError:
            pass
        yield
    finally:
        try:
            fcntl.lockf(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(fd)


def _update_json(path: Path, fn: Callable[[Any], Any], default):
    with _file_lock(path):
        data = _read_json(path, default)
        out = fn(data)
        _write_json(path, data)
        return out


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _say_once(key: str, text: str, out=print) -> None:
    """A note said once per process."""
    if key not in _SAID:
        _SAID.add(key)
        out(text)


_SAID: set = set()


# ------------------------------------------------------------------ configuration
@dataclass
class Budget:
    budget: int
    limit: Optional[int]
    reserve: int
    source: str
    order: str = "depth"
    followup_share: float = 0.75
    sentinel: bool = True
    account: Optional[str] = None
    note: str = ""

    def describe(self) -> str:
        if self.limit:
            return f"queue budget {self.budget} of {self.limit} (reserve {self.reserve}, {self.source})"
        return f"queue budget {self.budget} ({self.source})"


def _site(env):
    try:
        from mace.submission import site
    except ImportError:          # pragma: no cover
        return None, None
    try:
        return site, site.load(env)
    except Exception:
        return site, site.Site()


def detect_limit(account: Optional[str], env: Optional[Dict[str, str]] = None
                 ) -> Tuple[Optional[int], str]:
    """(the user's SLURM submit limit, where it was read): the smallest of
    ``scontrol show assoc_mgr`` MaxSubmitJobs and ``sacctmgr`` association /
    QOS MaxSubmit (each cached a day); (None, "") when none is readable."""
    env = dict(os.environ if env is None else env)
    site, _ = _site(env)
    found: List[Tuple[int, str]] = []
    if site is not None:
        try:
            v = site.submit_limit(account, env)
            if v:
                found.append((v, "MaxSubmitJobs from assoc_mgr"))
        except Exception:
            pass
        v = _sacctmgr_limit(site, account, env)
        if v:
            found.append((v, "MaxSubmit from sacctmgr"))
    if not found:
        return None, ""
    return min(found)


_MEMO: Dict[str, Any] = {}


def _sacctmgr_limit(site, account: Optional[str], env: Dict[str, str]) -> Optional[int]:
    exe = shutil.which("sacctmgr", path=env.get("PATH"))
    user = site._user(env) if hasattr(site, "_user") else env.get("USER", "")
    if not exe or not user:
        return None
    key = json.dumps(["SLURM-maxsubmit-sacctmgr", user, account or "", exe])
    if key in _MEMO:
        return _MEMO[key]
    _MEMO[key] = v = _sacctmgr_limit_read(site, account, env, exe, user, key)
    return v


def _sacctmgr_limit_read(site, account, env, exe, user, key) -> Optional[int]:
    try:
        hit = site._slurm_cached(key, env)
    except Exception:
        hit = None
    if hit and "limit" in hit:
        return hit["limit"]
    try:
        from mace.cluster import limits as L
    except ImportError:          # pragma: no cover
        return None
    a = site._slurm_run([exe, "-nP", "show", "assoc", f"user={user}", f"format={L.ASSOC_FORMAT}"], env)
    q = site._slurm_run([exe, "-nP", "show", "qos", f"format={L.QOS_FORMAT}"], env)
    if a is None or a.returncode != 0:
        return None
    assoc = L.parse_assoc(a.stdout)
    if not assoc:
        return None
    qos = L.parse_qos(q.stdout) if q is not None and q.returncode == 0 else {}
    acct = account
    if acct is None:
        try:
            acct = (site.slurm_associations(env) or {}).get("default")
        except Exception:
            acct = None
    if acct and not any(r["account"] == acct for r in assoc):
        acct = None
    limit = L.effective_limits(assoc, qos, acct).get("max_submit")
    try:
        from mace.utils import envmodules
        envmodules.cache_put(key, {"limit": limit, "name": "SLURM"}, env)
    except Exception:
        pass
    return limit


def _observed(env) -> Optional[int]:
    data = _read_json(state_dir(env) / "observed_limit.json", {})
    try:
        if time.time() - float(data.get("t", 0)) <= OBSERVED_TTL_S:
            return int(data["limit"])
    except (KeyError, TypeError, ValueError):
        pass
    return None


def remember_refusal(used: int, env=None) -> None:
    """SLURM refused at ``used`` queued jobs: the real limit, for an hour."""
    if used and used > 0:
        _write_json(state_dir(env) / "observed_limit.json", {"limit": int(used), "t": time.time(),
                                                              "at": _now()})
        forget_resolved()


_RESOLVED: Dict[str, Tuple[float, "Budget"]] = {}
RESOLVE_TTL_S = 30.0


def resolve(env: Optional[Dict[str, str]] = None, legacy_cap: Optional[int] = None,
            out=print, quiet: bool = False) -> Budget:
    """See _resolve; remembered for RESOLVE_TTL_S in a process (a loop of
    submissions asks once)."""
    e = dict(os.environ if env is None else env)
    key = json.dumps([legacy_cap] + [e.get(k, "") for k in ("MACE_SITE_CONFIG", "XDG_CONFIG_HOME", "HOME",
                                                           "MACE_QUEUE_STATE_DIR", "XDG_CACHE_HOME", "PATH")])
    hit = _RESOLVED.get(key)
    if hit and time.time() - hit[0] < RESOLVE_TTL_S:
        cfg = hit[1]
    else:
        cfg = _resolve(e, legacy_cap)
        _RESOLVED[key] = (time.time(), cfg)
    if cfg.note and not quiet:
        _say_once("fallback", cfg.note, out)
    return cfg


def forget_resolved() -> None:
    _RESOLVED.clear()


def _resolve(env: Dict[str, str], legacy_cap: Optional[int] = None) -> Budget:
    """The user-wide budget: [queue] budget, else (legacy [queue] max_jobs:
    max_jobs - reserve), else the detected limit - reserve, else 900 (said
    once). ``legacy_cap``: a callback's --max-jobs minus --reserve (a job
    script of 1.1.5/1.1.6); the smaller one wins. A limit learned from a
    refusal in the last hour lowers it too."""
    env = dict(os.environ if env is None else env)
    site, s = _site(env)
    account = None
    if site is not None:
        try:
            account = site.resolve_account(s, env)[0]
        except Exception:
            account = getattr(s, "account", None)
    reserve_set = bool(s is not None and s.is_set("reserve"))
    limit, lsrc = (s.queue_limit, "[queue] limit") if (s is not None and s.queue_limit) else (None, "")
    if limit is None:
        limit, lsrc = detect_limit(account, env)

    def reserve_for(lim):
        if reserve_set:
            return int(s.reserve)
        if site is not None and lim:
            return site.default_reserve(lim)
        return 100

    note = ""
    if s is not None and s.queue_budget:
        rs = reserve_for(limit)
        b, src = int(s.queue_budget), "[queue] budget"
        if limit:
            b = min(b, limit)
    elif s is not None and s.is_set("max_jobs"):
        rs = int(s.reserve)
        b, src = max(1, int(s.max_jobs) - rs), "[queue] max_jobs - reserve"
        if limit:
            b = min(b, max(1, limit - rs))
    elif limit:
        rs = reserve_for(limit)
        b, src = max(1, limit - rs), lsrc
    else:
        rs = int(s.reserve) if reserve_set else FALLBACK[1]
        mj = FALLBACK[0]
        b, src = max(1, mj - rs), "built-in (the limit could not be read)"
        note = (f"MACE: your SLURM submit limit could not be read; MACE keeps at most {b} jobs queued "
                f"({mj} - {rs}). Set [queue] limit or budget in site.toml to change it.")
    obs = _observed(env)
    if obs:
        ob = max(1, obs - min(rs, max(1, obs // 10)))
        if ob < b:
            b, src = ob, f"SLURM refused at {obs} jobs (learned in the last hour)"
            limit = min(limit or obs, obs)
            rs = min(rs, max(0, limit - b))      # the reserve that is left below that limit
    if legacy_cap is not None and 0 < legacy_cap < b:
        b, src = legacy_cap, "the job script's --max-jobs minus --reserve"
    return Budget(budget=b, limit=limit, reserve=rs, source=src,
                  order=(s.queue_order if s is not None else "depth"),
                  followup_share=(s.queue_followup_share if s is not None else 0.75),
                  sentinel=(s.queue_sentinel if s is not None else True), account=account, note=note)


# legacy flags of the running callback (--max-jobs/--reserve): set by manager.main
_LEGACY_CAP: List[Optional[int]] = [None]


def set_legacy_cap(max_jobs: Optional[int], reserve: Optional[int]) -> None:
    if max_jobs is None:
        _LEGACY_CAP[0] = None
        return
    _LEGACY_CAP[0] = max(1, int(max_jobs) - int(reserve or 0))


# ------------------------------------------------------------------ the user-wide lock
class _UserLock:
    """One POSIX record lock per process on <state>/budget.lock, re-entrant
    (a nested acquire only counts). Record locks are coherent across the
    nodes of a cluster on NFS and GPFS; without working locks
    (ENOLCK/ENOSYS) the queue manager's owner-token lock is used."""

    def __init__(self):
        self.depth = 0
        self.fd: Optional[int] = None
        self.token_mgr = None

    def acquire(self, timeout: float, env=None) -> bool:
        if self.depth:
            self.depth += 1
            return True
        path = state_dir(env) / "budget.lock"
        deadline = time.time() + max(0.0, timeout)
        use_token = (os.environ.get("MACE_QUEUE_LOCK") or "").strip().lower() == "token"
        while not use_token:
            fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o666)
            try:
                fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as e:
                os.close(fd)
                if e.errno in (errno.EACCES, errno.EAGAIN):
                    if time.time() >= deadline:
                        return False
                    time.sleep(0.05 + 0.1 * (os.getpid() % 7) / 7)
                    continue
                if e.errno in (errno.ENOLCK, errno.ENOSYS, errno.EOPNOTSUPP):
                    use_token = True
                    break
                raise
            self.fd = fd
            self.depth = 1
            return True
        from mace.queue.queue_lock_manager import QueueLockManager
        if self.token_mgr is None:
            saved = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
            self.token_mgr = QueueLockManager(lock_dir=path.parent, lock_timeout=300)
            self.token_mgr.use_token = True
            for sig, h in saved.items():
                try:
                    signal.signal(sig, h)
                except (ValueError, TypeError):
                    pass
        if self.token_mgr.acquire_lock("budget_token", timeout=int(max(1, timeout))):
            self.depth = 1
            return True
        return False

    def release(self) -> None:
        if not self.depth:
            return
        self.depth -= 1
        if self.depth:
            return
        if self.fd is not None:
            try:
                fcntl.lockf(self.fd, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(self.fd)
            self.fd = None
        elif self.token_mgr is not None:
            self.token_mgr.release_lock("budget_token")


_LOCK = _UserLock()


# ------------------------------------------------------------------ counting
@dataclass
class Snapshot:
    """The user's queue: live (id, state, name) rows, one per array task."""
    rows: List[Tuple[str, str, str]]

    @property
    def live(self) -> List[Tuple[str, str, str]]:
        return [r for r in self.rows if r[1] in LIVE_SLURM_STATES]

    def bases(self) -> set:
        return {r[0].split("_", 1)[0] for r in self.live}

    def listed_bases(self) -> set:
        """Every job squeue lists, in any state (SLURM knows it)."""
        return {r[0].split("_", 1)[0] for r in self.rows}

    def ids(self) -> set:
        return {r[0] for r in self.live} | self.bases()

    def count(self) -> int:
        from mace.queue.slurm_ids import Truncated, expand_queue_id
        n = 0
        for jid, _, _ in self.live:
            ids = expand_queue_id(jid)
            n += 1 if isinstance(ids, Truncated) else max(1, len(ids))
        return n


def _user(env) -> str:
    return env.get("USER") or env.get("LOGNAME") or ""


def squeue_snapshot(env: Optional[Dict[str, str]] = None) -> Optional[Snapshot]:
    env = dict(os.environ if env is None else env)
    user = _user(env)
    if not user:
        return None
    env["SLURM_BITSTR_LEN"] = "0"
    try:
        # -a: jobs in hidden partitions count against the limit too (MSU
        # HPCC's login shells set SQUEUE_ALL; a job's or cron's may not)
        r = subprocess.run(["squeue", "-a", "-u", user, "-h", "-r", "-o", "%i|%T|%j"], env=env,
                           stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    rows = []
    for line in r.stdout.splitlines():
        parts = line.strip().split("|", 2)
        if len(parts) >= 2 and parts[0]:
            rows.append((parts[0].strip(), parts[1].strip().upper(), parts[2].strip() if len(parts) > 2 else ""))
    return Snapshot(rows)


def _ledger_path(env=None) -> Path:
    return state_dir(env) / "ledger.json"


def _ledger_extra(snapshot: Snapshot, env=None, save: bool = True) -> Tuple[int, Dict[str, Any]]:
    """(slots of recent MACE submissions squeue does not list yet, ledger),
    dropping entries that left the queue (called under the budget lock)."""
    path = _ledger_path(env)
    led = _read_json(path, {})
    jobs = led.get("jobs") or {}
    bases = snapshot.listed_bases()
    now = time.time()
    extra, keep = 0, {}
    for jid, e in jobs.items():
        if jid in bases:
            e["seen"] = True              # squeue lists it: counted there
            keep[jid] = e
            continue
        if e.get("seen") or jid in _ENDING:
            continue                      # listed before and gone now, or ending: it ended
        age = now - float(e.get("t", 0))
        if age < LEDGER_WINDOW_S:
            keep[jid] = e
            if e.get("name") != SENTINEL_NAME:     # the sentinel is in the reserve
                extra += int(e.get("n", 1))
    led["jobs"] = keep
    if save:
        try:
            _write_json(path, led)
        except OSError:
            pass
    return extra, led


def _ledger_add(job_id: str, n: int, name: str, env=None) -> None:
    path = _ledger_path(env)
    led = _read_json(path, {})
    led.setdefault("jobs", {})[str(job_id).split("_", 1)[0]] = {"t": time.time(), "n": int(n), "name": name}
    try:
        _write_json(path, led)
    except OSError:
        pass


# ------------------------------------------------------------------ outcomes
@dataclass
class Outcome:
    status: str                       # submitted | held | failed | skipped
    job_id: Optional[str] = None
    reason: Optional[str] = None      # held: budget | workflow_cap | slurm_limit
    message: str = ""
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0

    @property
    def submitted(self) -> bool:
        return self.status == "submitted"

    @property
    def held(self) -> bool:
        return self.status == HELD


_LAST: List[Optional[Outcome]] = [None]


def last_outcome() -> Optional[Outcome]:
    """The outcome of this process's last gated submission (callers that get
    only a job id from deeper code tell held from failed with it)."""
    return _LAST[0]


def clear_last() -> None:
    _LAST[0] = None


def _set_last(o: Outcome) -> Outcome:
    _LAST[0] = o
    return o


# ------------------------------------------------------------------ the row being submitted
@dataclass
class RowRef:
    db_path: str
    calc_id: str
    on_release: Dict[str, Any] = field(default_factory=dict)


_ROW: contextvars.ContextVar = contextvars.ContextVar("mace_budget_row", default=None)


@contextlib.contextmanager
def row(db_path, calc_id, **on_release):
    """Submissions inside belong to this calculation row: a job that does
    not fit holds the row. ``on_release``: columns set when a drain submits
    it (output_file)."""
    if not db_path or not calc_id:
        yield
        return
    tok = _ROW.set(RowRef(str(Path(db_path).resolve()), str(calc_id), dict(on_release)))
    try:
        yield
    finally:
        _ROW.reset(tok)


_DEFER = [0]


@contextlib.contextmanager
def deferred():
    """Inside, a submission of a tracked row only holds it (reason budget);
    the caller drains afterwards, so every free slot goes out in the
    [queue] order, not in the order the rows were made."""
    _DEFER[0] += 1
    try:
        yield
    finally:
        _DEFER[0] -= 1


def is_deferred() -> bool:
    return _DEFER[0] > 0


# ------------------------------------------------------------------ sessions
class Session:
    """The budget lock held, the queue counted once; ``take(n)`` admits up to
    n more jobs, ``note()`` records a submission. Nested sessions in one
    process share the outer one."""

    def __init__(self, cfg: Budget, env=None, timeout: float = LOCK_TIMEOUT_S):
        self.cfg = cfg
        self.env = dict(os.environ if env is None else env)
        self.timeout = timeout
        self.locked = False
        self.snapshot: Optional[Snapshot] = None
        self.used = 0
        self.unknown = False
        self.sentinels = 0
        self.submitted = 0
        self.t0 = 0.0

    @property
    def stopped(self) -> bool:
        """SLURM refused a submission in this process (this pass) lately: no
        more sbatch until the pass ends (a long-running process: 5 minutes)."""
        return bool(_REFUSED[0]) and time.time() - _REFUSED[0] < REFUSAL_PAUSE_S

    @stopped.setter
    def stopped(self, value: bool) -> None:
        _REFUSED[0] = time.time() if value else 0.0

    def __enter__(self):
        self.locked = _LOCK.acquire(self.timeout, self.env)
        if self.locked:
            self.count()
        return self

    def __exit__(self, *exc):
        if self.locked:
            _LOCK.release()
            self.locked = False
        return False

    def count(self) -> None:
        self.t0 = time.time()
        if not shutil.which("squeue", path=self.env.get("PATH")):
            # no SLURM client here at all (a test's lone sbatch, a machine
            # without squeue): nothing to count, nothing is held
            self.snapshot = None
            self.unknown = False
            self.used = self.submitted
            return
        snap = squeue_snapshot(self.env)
        if snap is not None and _ENDING:
            # the job this process runs in is ending (a completion callback is
            # its script's last line; the sentinel): its slot is free already
            snap = Snapshot([r for r in snap.rows if r[0] not in _ENDING])
        if snap is None:
            # squeue's job list failed: the queue is unreadable. assoc_mgr's
            # used count says how many, not which jobs, and old-row conversion
            # and every release need to know which (a 1.1.5 row whose job is
            # queued would be submitted twice)
            self.snapshot = Snapshot([])
            self.unknown = True
            self.used = self.cfg.budget              # admit nothing
            return
        self.snapshot = snap
        extra, _ = _ledger_extra(snap, self.env)
        # the mace-drain sentinel lives in the reserve, not in the budget
        self.sentinels = sum(1 for r in snap.live if r[2] == SENTINEL_NAME)
        self.used = snap.count() - self.sentinels + extra
        self.unknown = False

    def free(self, reserve: bool = False) -> int:
        if not self.locked or self.unknown:
            return 0
        if time.time() - self.t0 > 60:
            self.count()
        if reserve:
            return max(0, (self.cfg.limit or (self.cfg.budget + self.cfg.reserve)) - self.used - self.sentinels)
        return max(0, self.cfg.budget - self.used)

    def take(self, n: int, reserve: bool = False) -> int:
        if self.stopped:
            return 0
        return max(0, min(int(n), self.free(reserve)))

    def note(self, job_id: str, n: int = 1, name: str = "") -> None:
        self.used += int(n)
        self.submitted += int(n)
        _ledger_add(job_id, n, name, self.env)


_SESSIONS: List[Session] = []
_ENDING: set = set()           # squeue ids of the job this process is the end of


def job_is_ending(env=None) -> None:
    """Called by a completion callback (and the sentinel): the job it runs
    in is about to leave the queue, so it is not counted."""
    env = os.environ if env is None else env
    for k in ("SLURM_JOB_ID", "MACE_ARRAY_TASK"):
        if env.get(k):
            _ENDING.add(str(env[k]))
    if env.get("SLURM_ARRAY_JOB_ID") and env.get("SLURM_ARRAY_TASK_ID"):
        _ENDING.add(f"{env['SLURM_ARRAY_JOB_ID']}_{env['SLURM_ARRAY_TASK_ID']}")
_REFUSED = [0.0]               # when SLURM last refused a submission of this process
REFUSAL_PAUSE_S = 300


@contextlib.contextmanager
def session(cfg: Optional[Budget] = None, env=None, timeout: float = LOCK_TIMEOUT_S):
    if _SESSIONS:
        yield _SESSIONS[-1]
        return
    cfg = cfg or resolve(env, _LEGACY_CAP[0])
    s = Session(cfg, env, timeout)
    with s:
        _SESSIONS.append(s)
        try:
            yield s
        finally:
            _SESSIONS.pop()


# ------------------------------------------------------------------ the gate
def classify(stderr: str, used: Optional[int] = None, limit: Optional[int] = None) -> str:
    """'limit' for SLURM's per-user submit limit refusal, else 'error'.
    A reason code (QOSMaxSubmitJobPerUserLimit, AssocMaxSubmitJobLimit, ...)
    is a limit refusal; the generic "job submit limit" text is one too,
    unless the user's queue is known to be well below a known limit (then
    another policy, size or time, refused the job)."""
    text = stderr or ""
    if LIMIT_TOKEN_RE.search(text):
        return "limit"
    if LIMIT_TEXT_RE.search(text):
        if used is not None and limit and used < 0.9 * limit:
            return "error"
        return "limit"
    return "error"


def _tasks_of(argv: Sequence[str], cwd) -> int:
    """How many submit slots an sbatch takes: the tasks of an --array (in
    the argv or the script's #SBATCH lines), else 1."""
    spec = None
    for i, a in enumerate(argv):
        if a.startswith("--array="):
            spec = a.split("=", 1)[1]
        elif a in ("-a", "--array") and i + 1 < len(argv):
            spec = argv[i + 1]
    if spec is None and argv:
        script = Path(cwd or ".") / argv[-1]
        try:
            for line in script.read_text(errors="replace").splitlines()[:200]:
                m = re.match(r"\s*#SBATCH\s+(?:--array[= ]|-a\s*)(\S+)", line)
                if m:
                    spec = m.group(1)
        except OSError:
            pass
    if not spec:
        return 1
    spec = spec.split("%", 1)[0]
    n = 0
    for part in spec.split(","):
        part = part.split(":", 1)[0]
        if "-" in part:
            lo, hi = part.split("-", 1)
            try:
                n += int(hi) - int(lo) + 1
            except ValueError:
                n += 1
        elif part:
            n += 1
    return max(1, n)


def sbatch(argv: Sequence[str], cwd=None, *, n: Optional[int] = None, env=None, name: str = "",
           reserve: bool = False, quiet: bool = False, out=print) -> Outcome:
    """Submit ``argv`` (sbatch and its arguments) from ``cwd`` if the budget has
    room, else hold the current row (``row()``) and return a held Outcome.
    Never raises for a refusal; SLURM's message is printed and returned."""
    argv = list(argv)
    cwd = str(Path(cwd).resolve()) if cwd else os.getcwd()
    if n is None:
        n = _tasks_of(argv[1:], cwd)
    ref: Optional[RowRef] = _ROW.get()
    if ref is not None and is_deferred():
        hold_row(ref, "budget", argv, cwd, n)
        return _set_last(Outcome(HELD, reason="budget", message="held until the queue drain"))
    with session(env=env) as s:
        if not s.locked:
            msg = "the queue budget lock is busy"
            if ref is not None:
                hold_row(ref, "budget", argv, cwd, n, msg)
            return _set_last(Outcome(HELD, reason="budget", message=msg))
        if ref is not None:
            cap = _workflow_cap_left(ref)
            if cap is not None and cap < n:
                hold_row(ref, "workflow_cap", argv, cwd, n)
                return _set_last(Outcome(HELD, reason="workflow_cap",
                                         message="the workflow has its max_jobs queued"))
        if s.take(n, reserve=reserve) < n:
            why = "SLURM refused a submission in this pass" if s.stopped else s.cfg.describe()
            if s.unknown:
                why = "your queue could not be read (squeue failed)"
            if ref is not None:
                hold_row(ref, "slurm_limit" if s.stopped else "budget", argv, cwd, n, why)
            return _set_last(Outcome(HELD, reason="slurm_limit" if s.stopped else "budget", message=why))
        return _set_last(_run_sbatch(argv, cwd, n, s, ref, name, quiet, out))


def _run_sbatch(argv, cwd, n, s: Session, ref: Optional[RowRef], name: str, quiet: bool, out) -> Outcome:
    try:
        res = subprocess.run(argv, cwd=cwd, capture_output=True, text=True)
    except OSError as e:
        return Outcome("failed", message=f"sbatch could not be run: {e}", returncode=127)
    if res.returncode == 0:
        m = SUBMITTED_RE.search(res.stdout or "") or re.match(r"^\s*(\d+)", res.stdout or "")
        job_id = m.group(1) if m else None
        from mace.queue.slurm_ids import note_submission
        if job_id:
            s.note(job_id, n, name or Path(argv[-1]).stem)
        note_submission()
        return Outcome("submitted", job_id=job_id, stdout=res.stdout, stderr=res.stderr)
    err = (res.stderr or res.stdout or "").strip()
    if classify(err, None if s.unknown or s.snapshot is None else s.used, s.cfg.limit) == "limit":
        s.stopped = True
        if LIMIT_TOKEN_RE.search(err) and int(n) <= 1:
            # learned only from SLURM's own reason code, and only for one job:
            # an array of n is refused anywhere below used + n, so its refusal
            # says little about the limit (its rows are held and go in singly)
            remember_refusal(s.used, s.env)
        if ref is not None:
            hold_row(ref, "slurm_limit", argv, cwd, n, err)
        if not quiet:
            out(f"  SLURM refused the job at its submit limit ({s.used} queued); held, it starts as "
                f"slots free up:\n    {err.splitlines()[-1] if err else ''}")
        return Outcome(HELD, reason="slurm_limit", message=err, stdout=res.stdout, stderr=res.stderr,
                       returncode=res.returncode)
    if not quiet:
        out(f"  sbatch refused it: {err}")
    return Outcome("failed", message=err, stdout=res.stdout, stderr=res.stderr, returncode=res.returncode)


def run_generator(run: Callable[[Dict[str, str]], Any], cwd=None):
    """Run a job script generator (``run(env)`` returns its CompletedProcess)
    with an ``sbatch`` on PATH that only records what the generator asks
    for; each recorded sbatch then goes through the gate, and the result's
    stdout gets the ``Submitted batch job N`` line of each one submitted
    (so callers parse it as before). The shipped generators never call
    sbatch, so for them this changes nothing."""
    shim_dir = Path(tempfile.mkdtemp(prefix="mace_queue_shim_"))
    try:
        shim = shim_dir / "sbatch"
        shim.write_text("#!/bin/sh\n"
                        f"d={shlex.quote(str(shim_dir))}\n"
                        'f="$d/call.$(date +%s%N).$$"\n'   # time first: replayed in call order
                        'pwd > "$f.cwd"\n'
                        'for a in "$@"; do printf \'%s\\0\' "$a"; done > "$f.argv"\n'
                        f"echo {SHIM_MESSAGE} >&2\n"
                        "exit 1\n")
        shim.chmod(0o755)
        env = dict(os.environ)
        env["PATH"] = f"{shim_dir}:{env.get('PATH', '')}"
        result = run(env)
        calls = sorted(shim_dir.glob("call.*.argv"))
        if not calls:
            return result
        lines, ok = [], True
        for c in calls:
            args = [a for a in c.read_bytes().decode(errors="replace").split("\0") if a]
            ccwd = Path(str(c)[:-5] + ".cwd").read_text().strip() or cwd
            o = sbatch(["sbatch"] + args, ccwd)
            if o.submitted and o.job_id:
                lines.append(f"Submitted batch job {o.job_id}")
            elif not o.held:
                ok = False
        if result is not None:
            try:
                result.stdout = (result.stdout or "").replace(SHIM_MESSAGE, "") + \
                    ("\n" + "\n".join(lines) + "\n" if lines else "")
                result.stderr = (result.stderr or "").replace(SHIM_MESSAGE + "\n", "")
                result.returncode = 0 if ok else (result.returncode or 1)
            except AttributeError:
                pass
        return result
    finally:
        shutil.rmtree(shim_dir, ignore_errors=True)


# ------------------------------------------------------------------ held rows
def _connect(db_path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=60)
    conn.row_factory = sqlite3.Row
    return conn


def _settings(text) -> Dict[str, Any]:
    try:
        s = json.loads(text or "{}")
        return s if isinstance(s, dict) else {}
    except (TypeError, ValueError):
        return {}


# a row in one of these is never made held again (a release refused at the
# limit after mace workflow --cancel, a job that ended meanwhile)
_FINAL_STATUSES = ("cancelled", "canceled", "completed")  # british-ok: stored database value


def hold_row(ref: RowRef, reason: str, argv: Sequence[str], cwd, n: int = 1, message: str = "") -> None:
    """Make the row held with what submits it."""
    try:
        with contextlib.closing(_connect(ref.db_path)) as conn, conn:
            r = conn.execute("SELECT settings_json, status FROM calculations WHERE calc_id=?",
                             (ref.calc_id,)).fetchone()
            if r is None or r["status"] in _FINAL_STATUSES:
                return                # canceled (mace workflow --cancel) or done meanwhile
            st = _settings(r["settings_json"])
            if r["status"] != HELD or not st.get("held_since"):
                st["held_since"] = _now()
            st["held_reason"] = reason
            if message:
                st["held_message"] = message[-500:]
            else:
                st.pop("held_message", None)
            st["held_submit"] = {"argv": list(argv), "cwd": str(cwd), "n": int(n),
                                 "on_release": dict(ref.on_release or {})}
            cur = conn.execute("UPDATE calculations SET status=?, settings_json=? WHERE calc_id=? "
                               f"AND status NOT IN ({','.join('?' * len(_FINAL_STATUSES))})",
                               (HELD, json.dumps(st), ref.calc_id, *_FINAL_STATUSES))
            if not cur.rowcount:
                return
    except sqlite3.Error as e:
        print(f"  Could not record {ref.calc_id} as held: {e}")
        return
    register(ref.db_path)


def _workflow_cap_left(ref: RowRef) -> Optional[int]:
    """The workflow's explicit max_jobs minus its queued rows, or None."""
    try:
        with contextlib.closing(_connect(ref.db_path)) as conn:
            r = conn.execute("SELECT settings_json FROM calculations WHERE calc_id=?", (ref.calc_id,)).fetchone()
            wid = _settings(r["settings_json"]).get("workflow_id") if r else None
            if not wid:
                return None
            cap = workflow_cap(ref.db_path, wid)
            if cap is None:
                return None
            return cap - _live_rows(conn, wid)
    except sqlite3.Error:
        return None


def _live_rows(conn, workflow_id: Optional[str]) -> int:
    n = 0
    q = f"SELECT settings_json FROM calculations WHERE status IN ({','.join('?' * len(LIVE_ROW_STATUSES))})"
    for r in conn.execute(q, LIVE_ROW_STATUSES):
        wid = _settings(r["settings_json"]).get("workflow_id")
        if workflow_id is None or wid == workflow_id or not wid:
            n += 1
    return n


def normalize_cap(cap) -> Optional[int]:
    """An explicit max_jobs, or None: unset, invalid, or the 200 every plan
    up to 1.1.5 stored whether the user chose it or not."""
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1 or cap == 200:
        return None
    return cap


def workflow_cap(db_path, workflow_id: str) -> Optional[int]:
    entry = (_registry().get("dbs") or {}).get(str(Path(db_path).resolve())) or {}
    caps = entry.get("caps") or {}
    if workflow_id in caps:
        return normalize_cap(caps[workflow_id])
    plan = find_plan(db_path, workflow_id)
    if plan is None:
        return None
    return normalize_cap((plan.get("execution_settings") or {}).get("max_concurrent_jobs"))


def workflow_folder(db_path, workflow_id: str) -> Optional[Path]:
    """The folder whose workflow_configs/ holds the workflow's plan."""
    tail = str(workflow_id).replace("workflow_", "")
    d = Path(db_path).resolve().parent
    for p in [d] + list(d.parents)[:5]:
        if (p / "workflow_configs" / f"workflow_plan_{tail}.json").is_file():
            return p
    return None


def workflow_on_hold(db_path, workflow_id: Optional[str]) -> Optional[str]:
    """'paused' or 'canceled' when mace workflow --pause/--cancel holds the
    workflow (its held jobs are not released), else None."""
    if not workflow_id:
        return None
    folder = workflow_folder(db_path, workflow_id)
    if folder is None:
        return None
    try:
        from mace.workflow import control
        st = control.state_of(control.read_control(folder, workflow_id))
        return st if st in control.HOLD_STATES else None
    except Exception:
        return None


def cancel_held(db_path, workflow_id: str) -> int:
    """mace workflow --cancel: the workflow's held rows are canceled too."""
    n = 0
    # under the budget lock: a drain re-reads each row there just before its
    # sbatch, so a row canceled here is never submitted
    locked = _LOCK.acquire(LOCK_TIMEOUT_S)
    try:
        with contextlib.closing(_connect(db_path)) as conn, conn:
            own = Path(db_path).resolve().parent.name.startswith(".mace_context_")
            for r in conn.execute("SELECT calc_id, settings_json FROM calculations WHERE status=?", (HELD,)).fetchall():
                wid = _settings(r["settings_json"]).get("workflow_id")
                if wid == workflow_id or (own and not wid):
                    conn.execute("UPDATE calculations SET status='cancelled', "  # british-ok: stored database value
                                 "error_message='canceled with the workflow while held' WHERE calc_id=?",
                                 (r["calc_id"],))
                    n += 1
    except sqlite3.Error:
        pass
    finally:
        if locked:
            _LOCK.release()
    return n


def find_plan(db_path, workflow_id: str) -> Optional[Dict[str, Any]]:
    """workflow_configs/workflow_plan_<id>.json near the database."""
    tail = str(workflow_id).replace("workflow_", "")
    d = Path(db_path).resolve().parent
    for p in [d] + list(d.parents)[:5]:
        cand = p / "workflow_configs" / f"workflow_plan_{tail}.json"
        if cand.is_file():
            try:
                return json.loads(cand.read_text())
            except (OSError, ValueError):
                return None
    return None


# ------------------------------------------------------------------ registry
def _registry_path(env=None) -> Path:
    return state_dir(env) / "active_workflows.json"


def _registry(env=None) -> Dict[str, Any]:
    return _read_json(_registry_path(env), {})


def register(db_path, workflow_id: Optional[str] = None, cap=None, env=None) -> None:
    """Remember a database that has (or will have) held work, so a drain
    started anywhere sees it. Never fails."""
    try:
        key = str(Path(db_path).resolve())

        def upd(data):
            e = data.setdefault("dbs", {}).setdefault(key, {"registered": _now()})
            if workflow_id:
                e.setdefault("workflows", [])
                if workflow_id not in e["workflows"]:
                    e["workflows"].append(workflow_id)
                if cap is not None:
                    e.setdefault("caps", {})[workflow_id] = cap
            e["seen"] = _now()
        _update_json(_registry_path(env), upd, {})
    except OSError:
        pass


def unregister(db_path, env=None) -> None:
    try:
        key = str(Path(db_path).resolve())
        _update_json(_registry_path(env), lambda d: (d.get("dbs") or {}).pop(key, None), {})
    except OSError:
        pass


def registered_dbs(env=None) -> List[str]:
    return list((_registry(env).get("dbs") or {}).keys())


# ------------------------------------------------------------------ migration of old rows
def migrate_db(db_path, snapshot: Optional[Snapshot]) -> int:
    """Rows MACE 1.1.5/1.1.6 left behind after SLURM refused them become
    held/slurm_limit: 'pending' without a job id (older than 10 minutes, no
    job of its name queued) and 'failed' rows without a job id whose error
    is empty, 'SLURM submission failed' or a limit refusal and which never
    ran. A row needs its job script to be released. Returns the count."""
    names = {r[2] for r in (snapshot.live if snapshot else [])}
    cutoff = (datetime.now() - timedelta(seconds=PHANTOM_AGE_S)).isoformat()
    n = 0
    try:
        with contextlib.closing(_connect(db_path)) as conn, conn:
            rows = conn.execute(
                "SELECT * FROM calculations WHERE (slurm_job_id IS NULL OR slurm_job_id='') AND "
                "((status='pending' AND created_at < ?) OR status='failed')", (cutoff,)).fetchall()
            for r in rows:
                if r["status"] == "failed":
                    err = (r["error_message"] or "").strip()
                    if err and err != "SLURM submission failed" and classify(err) != "limit":
                        continue
                inp = Path(r["input_file"] or "")
                work = Path(r["work_dir"] or inp.parent or ".")
                script = Path(r["job_script"]) if r["job_script"] else work / f"{inp.stem}.sh"
                if not inp.stem or not script.is_file():
                    continue
                if (work / f"{inp.stem}.out").exists():
                    continue
                if inp.stem in names:
                    continue
                if not _is_batch_script(script):
                    continue
                st = _settings(r["settings_json"])
                st.update({"held_since": r["created_at"] or _now(), "held_reason": "slurm_limit",
                           "held_message": "SLURM refused this submission (an earlier MACE recorded it "
                                           f"as {r['status']}); MACE submits it as slots free up",
                           "held_submit": {"argv": ["sbatch"] + [script.name], "cwd": str(script.parent), "n": 1,
                                           "on_release": {}}})
                conn.execute("UPDATE calculations SET status=?, settings_json=?, error_message=NULL "
                             "WHERE calc_id=?", (HELD, json.dumps(st), r["calc_id"]))
                n += 1
    except sqlite3.Error:
        return n
    if n:
        register(db_path)
    return n


def _is_batch_script(path: Path) -> bool:
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return False
    return bool(re.search(r"(?m)^\s*#SBATCH\b", text)) and "echo '#SBATCH" not in text \
        and 'echo "#SBATCH' not in text


# ------------------------------------------------------------------ drain
@dataclass
class Candidate:
    db_path: str
    calc_id: str
    material_id: str
    calc_type: str
    step: int
    held_since: str
    workflow_id: Optional[str]
    reason: str
    submit: Dict[str, Any]

    @property
    def followup(self) -> bool:
        return self.step > 1


def _step_of(r, st) -> int:
    for k in ("workflow_step", "step_number", "step_num"):
        v = st.get(k)
        if isinstance(v, int) and not isinstance(v, bool):
            return v
        if isinstance(v, str) and v.isdigit():
            return int(v)
    return 2 if r["prerequisite_calc_id"] else 1


def held_rows(db_path) -> List[Candidate]:
    out = []
    try:
        with contextlib.closing(_connect(db_path)) as conn:
            for r in conn.execute("SELECT * FROM calculations WHERE status=?", (HELD,)):
                st = _settings(r["settings_json"])
                out.append(Candidate(str(Path(db_path).resolve()), r["calc_id"], r["material_id"],
                                     r["calc_type"], _step_of(r, st), st.get("held_since") or r["created_at"] or "",
                                     st.get("workflow_id"), st.get("held_reason") or "budget",
                                     st.get("held_submit") or {}))
    except sqlite3.Error:
        pass
    return out


def order_candidates(cands: List[Candidate], order: str, free: int, share: float = 0.75) -> List[Candidate]:
    """depth: next steps first (deepest step first), then first steps, each
    oldest first; breadth: first steps first; mixed: ``share`` of the free
    slots to next steps, the rest (at least one) to first steps."""
    fol = sorted([c for c in cands if c.followup], key=lambda c: (-c.step, c.held_since, c.calc_id))
    first = sorted([c for c in cands if not c.followup], key=lambda c: (c.held_since, c.calc_id))
    if order == "breadth":
        return sorted(cands, key=lambda c: (c.step, c.held_since, c.calc_id))
    if order == "mixed" and fol and first and free > 0:
        q = int(round(free * share))
        q = max(0, min(q, free - 1, len(fol)))
        k = free - q
        if k > len(first):              # first steps cannot use their share
            q = min(len(fol), free - len(first))
            k = free - q
        return fol[:q] + first[:k] + fol[q:] + first[k:]
    return fol + first


def release(c: Candidate, s: Session, out=print, quiet: bool = False) -> Outcome:
    """Submit a held row with its recorded sbatch (the lock held)."""
    sub = c.submit or {}
    argv = list(sub.get("argv") or [])
    cwd = sub.get("cwd")
    n = int(sub.get("n") or 1)
    if not argv or not cwd or not Path(cwd).is_dir():
        return Outcome("failed", message="nothing recorded to submit it with (its folder is gone)")
    ref = RowRef(c.db_path, c.calc_id, dict(sub.get("on_release") or {}))
    # the drain read its candidates a while ago: mace workflow --cancel (or
    # --pause) may have run since. Read the row again (under the lock, which
    # cancel_held takes too) and submit only a row that is still held.
    try:
        with contextlib.closing(_connect(c.db_path)) as conn:
            r = conn.execute("SELECT status, settings_json FROM calculations WHERE calc_id=?",
                             (c.calc_id,)).fetchone()
    except sqlite3.Error:
        r = None
    if r is None or r["status"] != HELD:
        return Outcome("skipped", message="no longer held")
    if c.workflow_id and workflow_on_hold(c.db_path, c.workflow_id):
        return Outcome("skipped", message="its workflow is paused or canceled")
    # an earlier drain was stopped (killed at a callback's walltime) between
    # sbatch accepting this row's job and recording it: the row is still held
    # but its job is queued. Record that job instead of submitting a second.
    name = Path(argv[-1]).stem
    if _settings(r["settings_json"]).get("held_releasing") and s.snapshot is not None:
        jobs = [jid for jid, _, nm in s.snapshot.live if nm == name]
        if len(jobs) == 1:
            o = Outcome("submitted", job_id=jobs[0], message="already queued (an interrupted drain submitted it)")
            _record_release(c, ref, argv, cwd, o, out)
            return o
    _mark_releasing(c)
    o = _run_sbatch(argv, cwd, n, s, ref, "", True, out)
    _record_release(c, ref, argv, cwd, o, out)
    return o


def _mark_releasing(c: Candidate) -> None:
    """Note in the held row that its sbatch is about to run (see release)."""
    try:
        with contextlib.closing(_connect(c.db_path)) as conn, conn:
            r = conn.execute("SELECT settings_json FROM calculations WHERE calc_id=? AND status=?",
                             (c.calc_id, HELD)).fetchone()
            if r is not None:
                st = _settings(r["settings_json"])
                st["held_releasing"] = _now()
                conn.execute("UPDATE calculations SET settings_json=? WHERE calc_id=? AND status=?",
                             (json.dumps(st), c.calc_id, HELD))
    except sqlite3.Error:
        pass


def _record_release(c: Candidate, ref: "RowRef", argv, cwd, o: Outcome, out) -> None:
    """The row after its release's sbatch: submitted, failed, or held again."""
    try:
        with contextlib.closing(_connect(c.db_path)) as conn, conn:
            r = conn.execute("SELECT settings_json, status, job_script FROM calculations WHERE calc_id=?",
                             (c.calc_id,)).fetchone()
            if r is None:
                return
            st = _settings(r["settings_json"])
            st.pop("held_releasing", None)
            if o.submitted:
                for k in ("held_reason", "held_since", "held_message", "held_submit"):
                    st.pop(k, None)
                st["released_from_hold"] = _now()
                script = str(Path(cwd) / argv[-1])
                cols = {"status": "submitted", "slurm_job_id": o.job_id, "submitted_at": _now(),
                        "settings_json": json.dumps(st)}
                if not r["job_script"] and Path(script).is_file():
                    cols["job_script"] = script
                for k, v in (ref.on_release or {}).items():
                    if k in ("output_file",):
                        cols[k] = v
                conn.execute(f"UPDATE calculations SET {', '.join(f'{k}=?' for k in cols)} "
                             f"WHERE calc_id=? AND status=?", list(cols.values()) + [c.calc_id, HELD])
            elif not o.held:
                for k in ("held_reason", "held_since", "held_message", "held_submit"):
                    st.pop(k, None)
                conn.execute("UPDATE calculations SET status='failed', error_message=?, settings_json=? "
                             "WHERE calc_id=? AND status=?",
                             (f"sbatch refused it: {o.message}"[:1000], json.dumps(st), c.calc_id, HELD))
            else:
                conn.execute("UPDATE calculations SET settings_json=? WHERE calc_id=? AND status=?",
                             (json.dumps(st), c.calc_id, HELD))
    except sqlite3.Error as e:
        out(f"  Could not record the release of {c.calc_id}: {e}")


def drain(env=None, dry_run: bool = False, dbs: Iterable = (), out=print, quiet: bool = False,
          sentinel: bool = True, from_sentinel: bool = False, migrate: bool = True,
          timeout: float = LOCK_TIMEOUT_S) -> Dict[str, Any]:
    """Release held rows of every registered database (plus ``dbs``) into
    the budget's free slots in the [queue] order; then, when work is still
    held and no MACE job is queued, make sure one sentinel job exists.
    Returns a report (``--json``)."""
    env = dict(os.environ if env is None else env)
    cfg = resolve(env, _LEGACY_CAP[0], out=out, quiet=quiet)
    paths = []
    for p in list(dbs) + registered_dbs(env):
        try:
            rp = str(Path(p).resolve())
        except OSError:
            continue
        if rp not in paths:
            paths.append(rp)
    report: Dict[str, Any] = {"budget": cfg.budget, "limit": cfg.limit, "reserve": cfg.reserve,
                              "source": cfg.source, "order": cfg.order, "released": [], "failed": [],
                              "converted": 0, "held": {}, "used": None, "free": None, "dry_run": dry_run,
                              "sentinel": None, "lock": True}
    with session(cfg, env, timeout) as s:
        if not s.locked:
            report["lock"] = False
            if not quiet:
                out("Another MACE process holds the queue budget lock; nothing released now.")
            return report
        if s.unknown:
            # squeue failed (a slurmctld timeout): once more
            time.sleep(UNKNOWN_RETRY_S)
            s.count()
        alive = []
        for p in paths:
            if not Path(p).is_file():
                unregister(p, env)
                continue
            alive.append(p)
            # without a queue listing, "no job of its name queued" cannot be
            # checked: a pending row whose job is queued would be submitted twice
            if migrate and not dry_run and not s.unknown and s.snapshot is not None:
                report["converted"] += migrate_db(p, s.snapshot)
        cands = [c for p in alive for c in held_rows(p)]
        report["used"], report["free"] = s.used, s.free()
        ordered = order_candidates(cands, cfg.order, s.free(), cfg.followup_share)
        caps: Dict[Tuple[str, str], Optional[int]] = {}
        live: Dict[Tuple[str, str], int] = {}
        virtual = 0
        on_hold: Dict[Tuple[str, str], Optional[str]] = {}
        for c in ordered:
            if s.stopped:
                break
            n = int((c.submit or {}).get("n") or 1)
            if c.workflow_id:
                hk = (c.db_path, c.workflow_id)
                if hk not in on_hold:
                    on_hold[hk] = workflow_on_hold(c.db_path, c.workflow_id)
                if on_hold[hk]:
                    continue                  # paused or canceled: --resume releases it
            if c.workflow_id:
                key = (c.db_path, c.workflow_id)
                if key not in caps:
                    caps[key] = workflow_cap(c.db_path, c.workflow_id)
                    if caps[key] is not None:
                        with contextlib.closing(_connect(c.db_path)) as conn:
                            live[key] = _live_rows(conn, c.workflow_id)
                if caps[key] is not None and live[key] + n > caps[key]:
                    if c.reason != "workflow_cap" and not dry_run:
                        _set_reason(c, "workflow_cap")
                    continue
            if s.free() - virtual < n:
                if dry_run:
                    continue
                break
            if dry_run:
                virtual += n
                report["released"].append(_cand_dict(c))
                if c.workflow_id and caps.get((c.db_path, c.workflow_id)) is not None:
                    live[(c.db_path, c.workflow_id)] += n
                continue
            o = release(c, s, out, quiet)
            if o.submitted:
                d = _cand_dict(c)
                d["job_id"] = o.job_id
                report["released"].append(d)
                if c.workflow_id and caps.get((c.db_path, c.workflow_id)) is not None:
                    live[(c.db_path, c.workflow_id)] += n
                if not quiet:
                    out(f"  Released {c.calc_type} {c.material_id}: job {o.job_id}")
            elif o.held:
                if not quiet:
                    out(f"  SLURM refused {c.calc_type} {c.material_id} at its submit limit; it stays held")
            elif o.status == "skipped":
                continue
            else:
                report["failed"].append(dict(_cand_dict(c), message=o.message))
                if not quiet:
                    out(f"  {c.calc_type} {c.material_id} could not be submitted: {o.message}")
        remaining = [c for p in alive for c in held_rows(p)]
        held: Dict[str, int] = {}
        for c in remaining:
            held[c.reason] = held.get(c.reason, 0) + 1
        report["held"] = held
        report["used"] = s.used
        report["free"] = s.free()
        for p in alive:
            if not _db_active(p):
                unregister(p, env)
        if sentinel and not dry_run:
            report["sentinel"] = ensure_sentinel(s, cfg, remaining, alive, env, out=out, quiet=quiet,
                                                 from_sentinel=from_sentinel)
        elif dry_run and remaining:
            report["sentinel"] = {"would_check": True}
    return report


def _set_reason(c: Candidate, reason: str) -> None:
    try:
        with contextlib.closing(_connect(c.db_path)) as conn, conn:
            r = conn.execute("SELECT settings_json FROM calculations WHERE calc_id=?", (c.calc_id,)).fetchone()
            if r:
                st = _settings(r["settings_json"])
                st["held_reason"] = reason
                conn.execute("UPDATE calculations SET settings_json=? WHERE calc_id=?",
                             (json.dumps(st), c.calc_id))
    except sqlite3.Error:
        pass


def _cand_dict(c: Candidate) -> Dict[str, Any]:
    return {"calc_id": c.calc_id, "material_id": c.material_id, "calc_type": c.calc_type, "step": c.step,
            "held_since": c.held_since, "reason": c.reason, "workflow_id": c.workflow_id, "db": c.db_path}


def _db_active(db_path) -> bool:
    """Held or queued rows left (else the registry forgets the database)."""
    try:
        with contextlib.closing(_connect(db_path)) as conn:
            r = conn.execute(f"SELECT COUNT(*) FROM calculations WHERE status IN "
                             f"({','.join('?' * (len(LIVE_ROW_STATUSES) + 1))})",
                             (HELD,) + LIVE_ROW_STATUSES).fetchone()
            return bool(r and r[0])
    except sqlite3.Error:
        return False


def held_summary(db_path, workflow_id: Optional[str] = None) -> Dict[str, int]:
    """{reason: count} of the held rows (of one workflow)."""
    out: Dict[str, int] = {}
    for c in held_rows(db_path):
        if workflow_id and c.workflow_id and c.workflow_id != workflow_id:
            continue
        out[c.reason] = out.get(c.reason, 0) + 1
    return out


def describe_held(summary: Dict[str, int]) -> str:
    parts = [f"{n} {REASON_WORDS.get(k, k)}" for k, n in sorted(summary.items())]
    return ", ".join(parts)


# ------------------------------------------------------------------ sentinel
def _sentinel_marker(env=None) -> Path:
    return state_dir(env) / "sentinel.json"


def mace_job_live(snapshot: Optional[Snapshot], dbs: Sequence[str], env=None,
                  exclude: Iterable[str] = ()) -> bool:
    """A MACE calculation job of this user is in the queue: one MACE
    submitted (the ledger) or one a registered database lists as queued."""
    if snapshot is None:
        return True                      # cannot tell: assume yes (no sentinel)
    excl = {str(x) for x in exclude}
    live = {r[0].split("_", 1)[0] for r in snapshot.live if r[2] != SENTINEL_NAME} - excl
    live_full = {r[0] for r in snapshot.live if r[2] != SENTINEL_NAME}
    led = _read_json(_ledger_path(env), {}).get("jobs") or {}
    listed = snapshot.listed_bases()
    now = time.time()
    for j, e in led.items():
        if e.get("name") == SENTINEL_NAME or j in excl:
            continue
        if j in live:
            return True
        if j not in listed and not e.get("seen") and now - float(e.get("t", 0)) < LEDGER_WINDOW_S:
            return True                  # just submitted; squeue does not list it yet
    if not live:
        return False
    for p in dbs:
        try:
            with contextlib.closing(_connect(p)) as conn:
                for (jid,) in conn.execute("SELECT slurm_job_id FROM calculations WHERE slurm_job_id IS NOT NULL "
                                           "AND status IN ('submitted','running','queued','pending')"):
                    if jid and (str(jid) in live_full or str(jid).split("_", 1)[0] in live):
                        return True
        except sqlite3.Error:
            continue
    return False


def sentinel_job(snapshot: Optional[Snapshot], exclude: Iterable[str] = ()) -> Optional[str]:
    excl = {str(x) for x in exclude}
    for jid, _, name in (snapshot.live if snapshot else []):
        if name == SENTINEL_NAME and jid not in excl:
            return jid
    return None


def sentinel_script(cfg: Budget, env=None) -> Path:
    """The sentinel's batch script: one core, five minutes, starting
    SENTINEL_BEGIN_MIN minutes later, running ``mace queue drain``."""
    env = dict(os.environ if env is None else env)
    d = state_dir(env) / "sentinel"
    d.mkdir(parents=True, exist_ok=True)
    site, s = _site(env)
    lines = ["#!/bin/bash",
             f"#SBATCH --job-name={SENTINEL_NAME}",
             f"#SBATCH --begin=now+{SENTINEL_BEGIN_MIN}minutes",
             "#SBATCH --time=0:05:00",
             "#SBATCH --ntasks=1",
             "#SBATCH --cpus-per-task=1",
             "#SBATCH --mem=500M",
             f"#SBATCH --output={d}/{SENTINEL_NAME}-%j.out"]
    if cfg.account:
        lines.append(f"#SBATCH --account={cfg.account}")
    for attr, flag in (("partition", "--partition"), ("qos", "--qos"), ("constraint", "--constraint")):
        v = getattr(s, attr, None) if s is not None else None
        if v:
            lines.append(f"#SBATCH {flag}={v}")
    home = Path(__file__).resolve().parents[2]
    lines += ["# MACE: work is held under your queue budget and no MACE job is queued; this job",
              "# releases it (mace queue drain) and comes back only while that is still so.",
              "# [queue] sentinel = false in site.toml turns it off.",
              f"export MACE_HOME={shlex.quote(str(home))}",
              f'export PYTHONPATH={shlex.quote(str(home))}${{PYTHONPATH:+:$PYTHONPATH}}']
    for k in ("MACE_SITE_CONFIG", "MACE_QUEUE_STATE_DIR", "XDG_CACHE_HOME", "XDG_CONFIG_HOME"):
        if env.get(k):
            lines.append(f"export {k}={shlex.quote(env[k])}")
    py = sys.executable or "python3"
    lines.append(f"exec {shlex.quote(py)} {shlex.quote(str(home / 'mace_cli'))} --no-banner queue drain --sentinel")
    path = d / f"{SENTINEL_NAME}.sh"
    path.write_text("\n".join(lines) + "\n")
    return path


def ensure_sentinel(s: Session, cfg: Budget, remaining: Sequence[Candidate], dbs: Sequence[str], env=None,
                    out=print, quiet: bool = False, from_sentinel: bool = False) -> Optional[Dict[str, Any]]:
    """One sentinel per user, only while work is held and no MACE job is
    queued (called under the budget lock)."""
    env = dict(os.environ if env is None else env)
    # the job this drain runs in (a completion callback, or the sentinel
    # itself) is ending: it is no MACE job that will release held work later
    own = env.get("SLURM_JOB_ID")
    me = own if from_sentinel else None
    exclude = [me] if me else []
    if not remaining:
        return {"state": "none", "reason": "nothing held"}
    if not cfg.sentinel:
        return {"state": "off", "reason": "[queue] sentinel = false"}
    if s.snapshot is None:
        return {"state": "none", "reason": "the queue could not be read"}
    marker = _read_json(_sentinel_marker(env), {})
    if s.unknown:
        # squeue failed twice: this may be the last MACE job, and
        # then nothing would ever release the held work. Submit the sentinel
        # unless one was submitted recently enough to be still waiting (it
        # starts SENTINEL_BEGIN_MIN minutes on and runs 5 minutes at most; it
        # lives in the reserve: at most one job over the budget). An older
        # marker names a sentinel that has run, so it is not trusted.
        mj = marker.get("job_id")
        if mj and mj != me and time.time() - float(marker.get("t", 0)) < SENTINEL_MARKER_S:
            return {"state": "queued", "job_id": mj, "note": "the queue could not be read"}
        if s.stopped:
            return {"state": "none", "reason": "the queue could not be read"}
        return _submit_sentinel(s, cfg, env, out, quiet)
    if mace_job_live(s.snapshot, dbs, env, [own] if own else []):
        return {"state": "none", "reason": "MACE jobs are queued; their completion releases held work"}
    existing = sentinel_job(s.snapshot, exclude)
    if existing:
        return {"state": "queued", "job_id": existing}
    mj = marker.get("job_id")
    if mj and mj != me and mj in s.snapshot.ids() and time.time() - float(marker.get("t", 0)) < 3600:
        return {"state": "queued", "job_id": mj}
    if mj and mj != me and time.time() - float(marker.get("t", 0)) < LEDGER_WINDOW_S:
        return {"state": "queued", "job_id": mj, "note": "submitted moments ago"}
    if s.stopped or s.free(reserve=True) < 1:
        return {"state": "none", "reason": "no free slot below the submit limit"}
    return _submit_sentinel(s, cfg, env, out, quiet)


def _submit_sentinel(s: Session, cfg: Budget, env, out, quiet: bool) -> Dict[str, Any]:
    script = sentinel_script(cfg, env)
    try:
        from mace.queue.slurm_ids import outside_allocation_env
        senv = outside_allocation_env(env)
    except Exception:
        senv = env
    try:
        res = subprocess.run(["sbatch", script.name], cwd=str(script.parent), env=senv,
                             capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as e:
        return {"state": "failed", "message": str(e)}
    m = SUBMITTED_RE.search(res.stdout or "")
    if res.returncode != 0 or not m:
        msg = (res.stderr or res.stdout or "").strip()
        if not quiet:
            out(f"  The {SENTINEL_NAME} job could not be submitted: {msg}")
        return {"state": "failed", "message": msg}
    from mace.queue.slurm_ids import note_submission
    note_submission()
    jid = m.group(1)
    s.note(jid, 1, SENTINEL_NAME)
    s.used -= 1
    s.sentinels += 1
    _write_json(_sentinel_marker(env), {"job_id": jid, "t": time.time(), "at": _now()})
    if not quiet:
        out(f"  Work is held and no MACE job is queued: submitted the {SENTINEL_NAME} job {jid}; it runs "
            f"mace queue drain in {SENTINEL_BEGIN_MIN} minutes ([queue] sentinel = false turns it off).")
    return {"state": "submitted", "job_id": jid}


# ------------------------------------------------------------------ untracked submits
_UNSUBMITTED: List[Tuple[str, str, List[str]]] = []


def note_unsubmitted(name: str, cwd: str, argv: Sequence[str]) -> None:
    _UNSUBMITTED.append((name, str(cwd), list(argv)))


def offer_reserve(out=print, assume_yes: Optional[bool] = None, ask=input) -> Tuple[List[str], List[str]]:
    """Untracked mace submit: list what did not fit and, after a yes (or
    MACE_SUBMIT_USE_RESERVE=1 / --yes), submit it from the reserve.
    Returns (submitted job ids, names still not submitted)."""
    if not _UNSUBMITTED:
        return [], []
    items = list(_UNSUBMITTED)
    _UNSUBMITTED.clear()
    cfg = resolve(quiet=True)
    out(f"\nNot submitted ({cfg.describe()}): {len(items)} job(s)")
    for name, _, _ in items:
        out(f"  {name}")
    if assume_yes is None:
        assume_yes = os.environ.get("MACE_SUBMIT_USE_RESERVE") == "1"
    yes = assume_yes
    if not yes and sys.stdin is not None and sys.stdin.isatty():
        try:
            yes = ask(f"Use {len(items)} of the {cfg.reserve} reserved slots for them? [y/N] ").strip().lower() \
                in ("y", "yes")
        except EOFError:
            yes = False
    if not yes:
        out("Submit them later (mace submit again), with --yes to use the reserve, or with --track so MACE "
            "holds them and submits them as slots free up.")
        return [], [n for n, _, _ in items]
    ids, left = [], []
    with session(cfg) as s:
        for name, cwd, argv in items:
            if not s.locked or s.take(1, reserve=True) < 1:
                left.append(name)
                continue
            o = _run_sbatch(argv, cwd, 1, s, None, name, False, out)
            if o.submitted:
                ids.append(o.job_id or "")
                out(f"  Submitted {name} from the reserve (job {o.job_id})")
            else:
                left.append(name)
    if left:
        out(f"Still not submitted (the reserve is full too): {', '.join(left)}")
    return ids, left


# ------------------------------------------------------------------ CLI
def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        prog="mace queue",
        description="MACE's queue budget: release held jobs into free slots (drain), or show the budget.")
    sub = p.add_subparsers(dest="action")
    d = sub.add_parser("drain", help="submit held jobs of every workflow while the budget has room")
    d.add_argument("--dry-run", action="store_true", help="show what would be submitted; submit nothing")
    d.add_argument("--json", action="store_true", help="print the report as JSON")
    d.add_argument("--db", action="append", default=[], help="also drain this database (repeatable)")
    d.add_argument("--quiet", action="store_true", help="print nothing but errors")
    d.add_argument("--sentinel", action="store_true", help=argparse.SUPPRESS)
    st = sub.add_parser("status", help="the budget, your queued jobs and held work")
    st.add_argument("--json", action="store_true", help="print it as JSON")
    args = p.parse_args(list(argv) if argv is not None else None)
    if args.action == "drain":
        if args.sentinel:
            job_is_ending()
        say = (lambda *a, **k: None) if (args.json or args.quiet) else print
        rep = drain(dry_run=args.dry_run, dbs=args.db, out=say, quiet=args.json or args.quiet,
                    from_sentinel=args.sentinel)
        if args.json:
            print(json.dumps(rep, indent=1))
            return 0 if rep.get("lock", True) and not rep.get("failed") else 1
        if not rep.get("lock", True):
            return 1
        verb = "Would submit" if args.dry_run else "Submitted"
        if not args.quiet:
            left = sum(rep["held"].values())
            print(f"{verb} {len(rep['released'])} held job(s)"
                  + (f"; {left} still held ({describe_held(rep['held'])})" if left and not args.dry_run else ""))
            print(f"Queue: {rep['used']} of {rep['budget']} in use"
                  + (f" (limit {rep['limit']}, reserve {rep['reserve']})" if rep.get("limit") else "")
                  + f"; order {rep['order']}")
            sen = rep.get("sentinel") or {}
            if sen.get("state") in ("queued", "submitted"):
                print(f"Sentinel job {SENTINEL_NAME}: {sen.get('job_id')}")
        if rep.get("failed"):
            print(f"{len(rep['failed'])} held job(s) could not be submitted (see above)", file=sys.stderr)
            return 1
        return 0
    if args.action == "status":
        cfg = resolve(quiet=True)
        snap = squeue_snapshot()
        dbs = registered_dbs()
        held = {}
        for db in dbs:
            for k, v in held_summary(db).items():
                held[k] = held.get(k, 0) + v
        used = snap.count() if snap else None
        sen = sentinel_job(snap)
        data = {"budget": cfg.budget, "limit": cfg.limit, "reserve": cfg.reserve, "source": cfg.source,
                "order": cfg.order, "sentinel_enabled": cfg.sentinel, "used": used, "held": held,
                "databases": dbs, "sentinel_job": sen}
        if args.json:
            print(json.dumps(data, indent=1))
            return 0
        print(f"Queue budget: {cfg.budget}" + (f" of {cfg.limit} (reserve {cfg.reserve})" if cfg.limit else "")
              + f" - {cfg.source}")
        print(f"Your queued jobs: {used if used is not None else 'unknown (squeue failed)'}")
        print(f"Held: {sum(held.values())}" + (f" ({describe_held(held)})" if held else ""))
        print(f"Order: {cfg.order}; sentinel: {'on' if cfg.sentinel else 'off'}"
              + (f" (job {sen} queued)" if sen else ""))
        return 0
    p.print_help()
    return 2


if __name__ == "__main__":     # pragma: no cover
    sys.exit(main())
