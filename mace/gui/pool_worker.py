"""What a ``mace gui`` pool worker runs (spec section 8.4.1, plan B3).

This module is the worker's whole entry point: it imports the standard
library only, and each job imports what it needs (numpy, ase, spglib) when
it runs. The worker is started by pool.py without re-running the main
script (``mace_cli``), so the CLI's banner, its ui facade and the ``mace/``
directory on sys.path (where ``mace/queue`` would shadow the stdlib
``queue``) never reach it.

Jobs receive plain data (numpy arrays, numbers, strings), never ase objects,
and report progress and check for cancellation through two objects the pool
hands every worker at start: a progress queue and a shared array of cancel
flags, one per task slot.
"""
from __future__ import annotations

import os
import signal
import sys
import threading
import time
import traceback
from typing import Any, Callable, Dict

_PROGRESS = None        # multiprocessing.Queue: (slot, fraction, message)
_CANCEL = None          # multiprocessing.Array('b'): 1 = cancel the task in that slot
_PIDS = None            # multiprocessing.Array('i'): the worker pid running each slot (CPU-time limits)
_LAST: Dict[int, float] = {}
REPORT_EVERY_S = 0.1    # the parent throttles further, to 4 events per second per task
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
# what init() (or a math library it loads) may set: left out of the environment fingerprint
ENV_SET_BY_INIT = THREAD_VARS + ("MKL_THREADING_LAYER",)


class JobCanceled(Exception):
    """The task's cancel flag was set; the job stopped at its next check."""


def init(progress, cancel, niceness: int = 10, pids=None) -> None:
    """The pool's initializer: lower priority, one BLAS/OpenMP thread, few
    malloc arenas (spec section 9), and the progress/cancel channels."""
    global _PROGRESS, _CANCEL, _PIDS
    _PROGRESS, _CANCEL, _PIDS = progress, cancel, pids
    # test hook: a CPU-time limit for this worker only (never the server's own)
    if os.environ.get("MACE_GUI_TEST_HOOKS") == "1" and os.environ.get("MACE_GUI_TEST_WORKER_RLIMIT_CPU"):
        try:
            import resource
            n = int(os.environ["MACE_GUI_TEST_WORKER_RLIMIT_CPU"])
            resource.setrlimit(resource.RLIMIT_CPU, (n, n))
        except (ImportError, ValueError, OSError):
            pass
    # Ctrl+C reaches the whole foreground process group: the server stops the
    # pool itself, so a worker ignores it (no KeyboardInterrupt traceback each)
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (ValueError, OSError):
        pass
    _exit_with_parent(os.getppid())
    for k in THREAD_VARS:
        os.environ[k] = "1"
    try:
        cur = os.nice(0)
        if cur < niceness:
            os.nice(niceness - cur)
    except (AttributeError, OSError):
        pass
    try:
        from . import memory
        memory.limit_arenas(2)
    except Exception:
        pass


PARENT_POLL_S = 1.0


def _exit_with_parent(ppid: int) -> None:
    """End this worker when the server is gone, even after a SIGKILL that left
    it no time to stop the pool: a daemon thread polls the parent pid (a
    spawn worker holds both ends of its call queue's pipe, so it would never
    see end-of-file). prctl(PR_SET_PDEATHSIG) is not used: it follows the
    thread that started the worker, and the server's request threads end."""
    if ppid <= 1:
        return

    def watch():
        while True:
            time.sleep(PARENT_POLL_S)
            if os.getppid() != ppid:
                os._exit(0)
    threading.Thread(target=watch, name="mace-pool-parent", daemon=True).start()


def report(slot: int, fraction: float, message: str = "", force: bool = False) -> None:
    """Progress 0..1 for the task in ``slot`` (at most every 0.1 s here)."""
    if _PROGRESS is None or slot is None or slot < 0:
        return
    now = time.monotonic()
    if not force and now - _LAST.get(slot, 0.0) < REPORT_EVERY_S:
        return
    _LAST[slot] = now
    try:
        _PROGRESS.put_nowait((int(slot), float(max(0.0, min(1.0, fraction))), str(message)[:200]))
    except Exception:
        pass


def canceled(slot: int) -> bool:
    return _CANCEL is not None and slot is not None and 0 <= slot < len(_CANCEL) and bool(_CANCEL[slot])


def check(slot: int) -> None:
    """Raise JobCanceled when the task was canceled (jobs call this at least every 0.5 s)."""
    if canceled(slot):
        raise JobCanceled("canceled")


class Ctx:
    """What a job function gets: its slot's progress and cancel check."""

    def __init__(self, slot: int):
        self.slot = slot

    def progress(self, fraction: float, message: str = "", force: bool = False) -> None:
        report(self.slot, fraction, message, force)

    def check(self) -> None:
        check(self.slot)

    @property
    def canceled(self) -> bool:
        return canceled(self.slot)


# ---------------------------------------------------------------- jobs
def _sleep(ctx: Ctx, seconds: float = 1.0, steps: int = 10, fail: bool = False, **_):
    """Test hook: a slow job with progress (and an optional failure)."""
    t0 = time.time()
    steps = max(1, int(steps))
    for k in range(steps):
        end = t0 + seconds * (k + 1) / steps
        while time.time() < end:
            ctx.check()
            time.sleep(min(0.05, max(0.0, end - time.time())))
        ctx.progress((k + 1) / steps, f"step {k + 1} of {steps}")
        if fail and k == steps // 2:
            raise RuntimeError("the test job failed on purpose")
    return {"pid": os.getpid(), "start": t0, "end": time.time(), "nice": os.nice(0)}


def _probe(ctx: Ctx, **_):
    """Test hook: what this worker process holds (the clean-imports test)."""
    mm = sys.modules.get("__mp_main__")
    q = sys.modules.get("queue")
    mace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repo = os.path.dirname(mace_dir)
    return {"pid": os.getpid(), "nice": os.nice(0),
            "mp_main": getattr(mm, "__file__", None) if mm is not None else None,
            "main": getattr(sys.modules.get("__main__"), "__file__", None),
            "queue_file": getattr(q, "__file__", None),
            "loaded": sorted(m for m in ("pandas", "pyarrow", "plotly", "mace.ui", "mace.utils.ui", "mace.utils.banner",
                                         "mace.gui.server") if m in sys.modules),
            # never the environment or sys.path themselves (they can hold a token or
            # a user's paths): booleans, and a one-way fingerprint of the environment
            "mace_dir_on_sys_path": any(os.path.abspath(p or os.curdir) == mace_dir for p in sys.path),
            "mace_subdir_on_sys_path": [os.path.relpath(os.path.abspath(p or os.curdir), repo)
                                        for p in sys.path if _inside(os.path.abspath(p or os.curdir), mace_dir)],
            "cwd_inside_repo": _inside(os.getcwd(), repo),
            "env_fingerprint": _fingerprint(), "pythonpath_set": "PYTHONPATH" in os.environ,
            "threads": {k: os.environ.get(k) for k in THREAD_VARS}}


def _inside(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def _fingerprint() -> str:
    from .childenv import env_fingerprint
    return env_fingerprint(skip=ENV_SET_BY_INIT)


def _builders(name: str) -> Callable:
    def call(ctx: Ctx, **kw):
        from . import builders
        return getattr(builders, name)(ctx=ctx, **kw)
    return call


def _volume(name: str) -> Callable:
    """The Volume document's jobs (volume.py): load, slice, combine, mep, save."""
    def call(ctx: Ctx, **kw):
        from . import volume
        if name == "mep":
            return volume.mep_range(**kw)
        if name == "check":
            return {"problem": volume.same_cell(**kw)}
        return getattr(volume, f"{name}_job")(ctx=ctx, **kw)
    return call


def _orbitals(name: str) -> Callable:
    """The orbital picker's jobs (orbitals.py): index, grid, save."""
    def call(ctx: Ctx, **kw):
        from . import orbitals
        return getattr(orbitals, f"{name}_job")(ctx=ctx, **kw)
    return call


def _freq(ctx: Ctx, **kw):
    """The FREQ section of an output, parsed by mace plotting's parser (freq.py)."""
    from . import freq
    return freq.parse_job(ctx=ctx, **kw)


def _out_meta(ctx: Ctx, **kw):
    """A run's metadata card (runinfo.py), its prints kept off the worker's stdout."""
    from . import runinfo
    return runinfo.meta_job(ctx=ctx, **kw)


def _out_track(ctx: Ctx, path: str = "", measurement=None, **_):
    """A measurement along every frame of a run (runinfo.track), frame by frame."""
    from pathlib import Path
    from . import outindex, runinfo
    idx = outindex.build(Path(path))
    frames = []
    for k in range(idx.n_steps):
        if k % 8 == 0:
            ctx.check()
            ctx.progress(0.9 * k / max(1, idx.n_steps), f"Frame {k + 1} of {idx.n_steps}")
        frames.append(idx.block_structure(k))
    return {"values": runinfo.track(frames, measurement or {}), "n": idx.n_steps}


def _burn(ctx: Ctx, seconds: float = 1.0, **_):
    """Test hook: busy CPU for `seconds` of this process's CPU time."""
    end = time.process_time() + float(seconds)
    x = 0
    while time.process_time() < end:
        for _ in range(20000):
            x += 1
        ctx.check()
    return {"pid": os.getpid(), "cpu": time.process_time()}


JOBS: Dict[str, Callable] = {"sleep": _sleep, "probe": _probe, "freq.parse": _freq, "out.meta": _out_meta,
                             "out.track": _out_track}
if os.environ.get("MACE_GUI_TEST_HOOKS") == "1":
    JOBS["test.burn"] = _burn
# jobs that must never run twice: never resubmitted after a broken pool (all pure today)
NOT_IDEMPOTENT: set = set()


def register(kind: str, fn: Callable) -> None:
    JOBS[kind] = fn


def run(kind: str, payload: Dict[str, Any], slot: int):
    """The function the pool executes: one job, its result or its exception."""
    fn = JOBS.get(kind)
    if fn is None:
        if kind.startswith("builders."):
            fn = _builders(kind.split(".", 1)[1])
        elif kind.startswith("volume."):
            fn = _volume(kind.split(".", 1)[1])
        elif kind.startswith("orbitals.") and kind.split(".", 1)[1] in ("index", "grid", "save"):
            fn = _orbitals(kind.split(".", 1)[1])
        else:
            raise ValueError(f"unknown pool job {kind!r}")
    _LAST.pop(slot, None)
    if _PIDS is not None and 0 <= slot < len(_PIDS):
        _PIDS[slot] = os.getpid()
    ctx = Ctx(slot)
    try:
        return fn(ctx, **(payload or {}))
    except JobCanceled:
        raise
    except Exception as exc:
        # the traceback text travels with it (the Tasks panel's Details)
        exc.mace_traceback = traceback.format_exc()
        raise
