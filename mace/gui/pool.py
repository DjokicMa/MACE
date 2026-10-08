"""The GUI's process pool for CPU-heavy work (spec sections 8.4.1 and 9.1).

Builders over the inline size, Clean geometry, symmetrize sweeps, defect
searches and batch parsing run here, never in a request thread:

* created lazily (the first job), shut down after IDLE_S seconds without a
  job, which gives the workers' memory back;
* ``resources.effective_workers()`` processes (a quarter of the cores, 1 to
  8, within SLURM, ``max_workers`` / ``MACE_GUI_MAX_WORKERS``), each at nice
  10 with one BLAS/OpenMP thread;
* one CPU budget with the request threads: a job holds a ``compute.GATE``
  slot while it runs (the dispatcher thread waits for it, never a request);
* workers never fork the threaded server and never re-run the main script:
  a spawn context whose preparation data drops the main module and the
  ``mace/`` folder (and anything below it) from the child's sys.path, and
  starts it in a neutral working directory (childenv.py; spike S1: plain spawn and
  forkserver both re-ran ``mace_cli`` as ``__mp_main__``).

This relies on two CPython internals, ``popen_spawn_posix.Popen._launch``
and ``spawn.get_preparation_data``; tested on Python 3.11 and 3.12.
``tests/gui/test_pool_internals.py`` fails loudly when either changes.

Progress and cancel: every worker gets the same progress queue and array of
cancel flags at start; a task holds one slot of the array while it runs.
"""
from __future__ import annotations

import contextlib
import os
import signal
import sys
import threading
import time
from concurrent.futures import Future, ProcessPoolExecutor
from multiprocessing import context, popen_spawn_posix, spawn
from typing import Any, Callable, Dict, List, Optional

from . import childenv, pool_worker, resources

IDLE_S = 60.0
SLOTS = 64
MAX_TASKS_PER_CHILD = 50
_LAUNCH = threading.Lock()


@contextlib.contextmanager
def no_hangup():
    """SIGHUP blocked in this thread while children start: a worker or the
    resource tracker started here inherits the mask, so a hangup sent to the
    whole process group (the terminal or the ssh session closed) reaches only
    the server, which then stops the pool itself, as on Ctrl+C."""
    sig = getattr(signal, "SIGHUP", None)
    if sig is None or not hasattr(signal, "pthread_sigmask"):
        yield
        return
    old = signal.pthread_sigmask(signal.SIG_BLOCK, {sig})
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, old)


def _clean_prep(orig: Callable[[str], Dict[str, Any]]) -> Callable[[str], Dict[str, Any]]:
    def prep(name: str) -> Dict[str, Any]:
        d = orig(name)
        d.pop("init_main_from_path", None)      # the child does not run mace_cli again
        d.pop("init_main_from_name", None)
        # no mace/ (its queue package would shadow the standard library's) and
        # no relative entry; the checkout root and Crystal_d12/d3 stay
        d["sys_path"] = childenv.clean_sys_path(d.get("sys_path", []))
        # the worker starts in a neutral folder, not wherever `mace gui` was run
        d["dir"] = childenv.neutral_cwd()
        return d
    return prep


class _CleanPopen(popen_spawn_posix.Popen):
    def _launch(self, process_obj):
        # popen_spawn_posix reads spawn.get_preparation_data at launch: swapped
        # for this launch only, under a lock (the pool's processes start one by one)
        with _LAUNCH, no_hangup():
            orig = spawn.get_preparation_data
            spawn.get_preparation_data = _clean_prep(orig)
            try:
                return super()._launch(process_obj)
            finally:
                spawn.get_preparation_data = orig


class CleanSpawnProcess(context.SpawnProcess):
    _start_method = "spawn"

    @staticmethod
    def _Popen(process_obj):
        return _CleanPopen(process_obj)


class CleanSpawnContext(context.SpawnContext):
    Process = CleanSpawnProcess


def internals_ok() -> bool:
    """The two CPython internals this module relies on are where it expects them."""
    import inspect
    try:
        launch = inspect.signature(popen_spawn_posix.Popen._launch)
        prep = inspect.signature(spawn.get_preparation_data)
    except (AttributeError, TypeError, ValueError):
        return False
    return list(launch.parameters) == ["self", "process_obj"] and list(prep.parameters) == ["name"]


_WARNED = False


def make_context():
    """The clean spawn context, or the stock one (with one warning on stderr)
    when a Python version changed the internals: slower to start (each worker
    imports mace_cli again), but never a failed request."""
    global _WARNED
    if internals_ok():
        return CleanSpawnContext()
    if not _WARNED:
        _WARNED = True
        print("mace gui: this Python's multiprocessing internals differ from 3.11/3.12; "
              "the process pool uses the standard spawn start method", file=sys.stderr)
    import multiprocessing
    return multiprocessing.get_context("spawn")


def _status_kb(pid: int, key: str) -> Optional[int]:
    try:
        with open(f"/proc/{pid}/status") as fh:
            for line in fh:
                if line.startswith(key + ":"):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


class Pool:
    def __init__(self, workers: Callable[[], int], on_progress: Callable[[int, float, str], None]):
        self._workers = workers
        self._on_progress = on_progress
        self.ctx = make_context()
        self.lock = threading.Lock()
        self.ex: Optional[ProcessPoolExecutor] = None
        self.n = 0
        self.busy = 0
        self.idle_s = IDLE_S
        self._idle_timer: Optional[threading.Timer] = None
        self._free = list(range(SLOTS))
        self.cancel = None
        self.progress = None
        self._drain: Optional[threading.Thread] = None
        self.peak_rss_mb = 0.0
        self.pids_seen: set = set()
        self.started = 0          # executors created (tests: lazily, and again after an idle shutdown)
        self.slot_pids = None     # Array('i'): the worker pid of each slot (CPU-time limits)
        self.cpu: Dict[int, float] = {}        # pid -> CPU seconds last sampled
        self.procs: Dict[int, Any] = {}        # pid -> multiprocessing.Process (exit codes after a break)
        self.recycled = 0
        self.broken = 0
        self.limits: Dict[int, Optional[float]] = {}  # pid -> that worker's own CPU-time limit
        self._sampler: Optional[threading.Thread] = None

    # -- the executor -------------------------------------------------------
    def _executor(self) -> ProcessPoolExecutor:
        with self.lock, no_hangup():
            if self._idle_timer is not None:
                self._idle_timer.cancel()
                self._idle_timer = None
            if self.ex is not None and self.busy == 0 and self._near_limit():
                # idle and a worker past 60 % of the per-process CPU-time limit: a fresh executor
                old, self.ex = self.ex, None
                self.recycled += 1
                old.shutdown(wait=False, cancel_futures=True)
            if self.ex is None:
                if self.cancel is None:
                    self.cancel = self.ctx.Array("b", SLOTS, lock=False)
                    self.slot_pids = self.ctx.Array("i", SLOTS, lock=False)
                    self.progress = self.ctx.Queue()
                    self._drain = threading.Thread(target=self._drain_progress, name="mace-pool-progress", daemon=True)
                    self._drain.start()
                self.n = max(1, int(self._workers()))
                kw: Dict[str, Any] = {"max_workers": self.n, "mp_context": self.ctx, "initializer": pool_worker.init,
                                      "initargs": (self.progress, self.cancel, resources.WORKER_NICENESS, self.slot_pids)}
                if sys.version_info >= (3, 11):
                    kw["max_tasks_per_child"] = MAX_TASKS_PER_CHILD
                self.ex = ProcessPoolExecutor(**kw)
                self.started += 1
                if self._sampler is None or not self._sampler.is_alive():
                    self._sampler = threading.Thread(target=self._sample_loop, name="mace-pool-cpu", daemon=True)
                    self._sampler.start()
            return self.ex

    CPU_SAMPLE_S = 1.0

    def _sample_loop(self) -> None:
        """While jobs run: each worker's CPU time and own limit, every second
        (to tell a CPU-time-limit kill from another kill when the pool breaks)."""
        while self.ex is not None:
            if self.busy:
                try:
                    self.sample()
                except Exception:
                    pass
            time.sleep(self.CPU_SAMPLE_S)

    def _drain_progress(self):
        while True:
            try:
                slot, frac, msg = self.progress.get()
            except (EOFError, OSError, ValueError):
                return
            except Exception:
                continue
            try:
                self._on_progress(slot, frac, msg)
            except Exception:
                pass

    def acquire_slot(self) -> int:
        with self.lock:
            if not self._free:
                raise RuntimeError("too many tasks at once")
            s = self._free.pop(0)
        if self.cancel is not None:
            self.cancel[s] = 0
        return s

    def release_slot(self, s: int) -> None:
        if s is None or s < 0:
            return
        if self.cancel is not None:
            self.cancel[s] = 0
        with self.lock:
            if s not in self._free:
                self._free.append(s)

    def set_cancel(self, s: int) -> None:
        if self.cancel is not None and s is not None and 0 <= s < SLOTS:
            self.cancel[s] = 1

    def submit(self, kind: str, payload: Dict[str, Any], slot: int) -> Future:
        ex = self._executor()
        with self.lock:
            self.busy += 1
        if self.slot_pids is not None and 0 <= slot < SLOTS:
            self.slot_pids[slot] = 0
        with no_hangup():       # 3.12 starts workers (and the manager thread) here
            try:
                fut = ex.submit(pool_worker.run, kind, payload, slot)
            except BaseException:
                with self.lock:
                    self.busy -= 1
                raise
        fut.mace_executor = ex
        fut.add_done_callback(self._job_done)
        self.sample()
        return fut

    # -- CPU-time limits and a broken pool (plan D18) ----------------------------------
    def _near_limit(self, share: Optional[float] = None) -> bool:
        from . import cpulimit
        lim = cpulimit.limit()
        if lim is None:
            return False
        share = cpulimit.RESTART_AT if share is None else share
        for pid in self.pids():
            u = cpulimit.used(pid)
            if u is not None:
                self.cpu[pid] = u
            if self.limits.get(pid) is None:      # the worker's initializer may set it after its start
                self.limits[pid] = cpulimit.proc_limit(pid)
                if u >= share * lim:
                    return True
        return False

    def recover(self, old_ex) -> None:
        """The executor broke (a worker was killed): let it go; the next submit starts a new one."""
        with self.lock:
            if self.ex is old_ex:
                self.ex = None
                self.broken += 1

    def slot_death(self, slot: int) -> Optional[Dict[str, Any]]:
        """{pid, exitcode, cpu, limit} when the worker that ran `slot` died by a signal, else None."""
        from . import cpulimit
        if self.slot_pids is None or not 0 <= slot < SLOTS:
            return None
        pid = int(self.slot_pids[slot] or 0)
        if not pid:
            return None
        proc = self.procs.get(pid)
        code = getattr(proc, "exitcode", None) if proc is not None else None
        if code is None and proc is not None:
            try:
                proc.join(2)
                code = proc.exitcode
            except (AssertionError, ValueError, OSError):
                pass
        if code is None or code >= 0:
            return None
        return {"pid": pid, "exitcode": code, "cpu": self.cpu.get(pid),
                "limit": self.limits.get(pid) or cpulimit.limit(), "sample_s": self.CPU_SAMPLE_S}

    def _job_done(self, _fut):
        try:
            self.sample()
        except Exception:
            pass
        with self.lock:
            self.busy -= 1
            if self.busy == 0 and self.ex is not None:
                if self._idle_timer is not None:
                    self._idle_timer.cancel()
                self._idle_timer = threading.Timer(self.idle_s, self._idle_shutdown)
                self._idle_timer.daemon = True
                self._idle_timer.start()

    def _idle_shutdown(self):
        with self.lock:
            if self.busy or self.ex is None:
                return
            ex, self.ex = self.ex, None
            self._idle_timer = None
        ex.shutdown(wait=False, cancel_futures=True)

    def shutdown(self, kill: bool = False):
        """Stop the executor. ``kill`` (the server is stopping): also end the
        workers at once, so a running job does not hold the server's exit."""
        with self.lock:
            ex, self.ex = self.ex, None
            if self._idle_timer is not None:
                self._idle_timer.cancel()
                self._idle_timer = None
        if ex is None:
            return
        procs = list((getattr(ex, "_processes", None) or {}).values())
        if kill and self.cancel is not None:
            for k in range(len(self.cancel)):
                self.cancel[k] = 1
        ex.shutdown(wait=False, cancel_futures=True)
        if kill:
            for proc in procs:
                try:
                    proc.terminate()
                except (OSError, AttributeError, ValueError):
                    pass
            deadline = time.monotonic() + 3.0
            for proc in procs:
                try:
                    proc.join(max(0.0, deadline - time.monotonic()))
                    if proc.is_alive():
                        proc.kill()
                except (OSError, AttributeError, ValueError, AssertionError):
                    pass

    # -- reports ------------------------------------------------------------
    def pids(self) -> List[int]:
        ex = self.ex
        procs = getattr(ex, "_processes", None) or {}
        return [p for p in list(procs) if isinstance(p, int)]

    def sample(self) -> None:
        from . import cpulimit
        ex = self.ex
        procs = getattr(ex, "_processes", None) or {}
        for pid, proc in list(procs.items()):
            if isinstance(pid, int):
                self.procs[pid] = proc
        if len(self.procs) > 256:
            for pid in list(self.procs)[:-128]:
                self.procs.pop(pid, None)
        for pid in self.pids():
            self.pids_seen.add(pid)
            hwm = _status_kb(pid, "VmHWM")
            if hwm:
                self.peak_rss_mb = max(self.peak_rss_mb, hwm / 1024)
            u = cpulimit.used(pid)
            if u is not None:
                self.cpu[pid] = u
            if self.limits.get(pid) is None:      # the worker's initializer may set it after its start
                self.limits[pid] = cpulimit.proc_limit(pid)

    def report(self) -> Dict[str, Any]:
        self.sample()
        alive = [p for p in self.pids() if _status_kb(p, "VmRSS") is not None]
        return {"workers": self.n if self.ex is not None else 0, "cap": max(1, int(self._workers())),
                "alive": len(alive), "pids": alive, "busy": self.busy, "started": self.ex is not None,
                "executors_started": self.started, "idle_s": self.idle_s, "recycled": self.recycled,
                "broken": self.broken,
                "peak_rss_mb": round(self.peak_rss_mb, 1), "workers_used": len(self.pids_seen)}
