"""Background tasks: records, progress over Server-Sent Events, cancel
(spec section 8.4.8, plan B4).

A task is one job in the process pool (pool.py), run by a dispatcher thread
that first waits for a slot of the shared CPU budget (compute.GATE), never
by a request thread. A request that starts a task waits up to WAIT_S for it:
done by then, the result comes back in that same response; otherwise the
request answers 202 with the task, the page follows it on /api/events and
fetches the result once at the end.

Events: ``event: task`` with the task's record, at most 4 per second per
task for progress, and at once for every state change. Every SSE connection
has its own bounded queue (the oldest event goes when it is full).
"""
from __future__ import annotations

import itertools
import queue as _stdlib_queue
import secrets
import threading
import time
import traceback
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional

from . import compute

WAIT_S = 0.3
PROGRESS_EVERY_S = 0.25         # 4 events per second per task at most
KEEP_DONE = 64                  # finished tasks kept for the Tasks panel
RESULT_TTL_S = 300.0            # a result nobody fetched goes after this
EVENT_QUEUE = 64
STATES = ("queued", "running", "done", "failed", "canceled")


class Task:
    _ids = itertools.count(1)

    def __init__(self, kind: str, title: str, job: str, payload: Dict[str, Any], doc: Optional[str] = None,
                 client: Optional[str] = None, finish: Optional[Callable[[Any], Any]] = None, inline: Optional[Callable] = None):
        self.id = f"t{next(Task._ids)}-{secrets.token_hex(3)}"
        self.kind, self.title, self.job, self.payload, self.doc, self.client = kind, title, job, payload, doc, client
        self.finish = finish          # turns the job's result into the route's answer (in the dispatcher thread)
        self.inline = inline          # a callable(ctx) run in the dispatcher thread instead of the pool
        self.state = "queued"
        self.progress = 0.0
        self.message = ""
        self.log = ""
        self.created = time.time()
        self.started: Optional[float] = None
        self.ended: Optional[float] = None
        self.result: Any = None
        self.error: Optional[str] = None
        self.traceback: Optional[str] = None
        self.slot: Optional[int] = None
        self.future = None
        self.cancel_requested = False
        self.done = threading.Event()
        self.last_sent = 0.0
        self.pid: Optional[int] = None
        self.retry_of: Optional[str] = None
        self.quiet = False            # pool work nobody needs to see (the explorer's small batches): never published
        self.supersede_key = None     # (kind, doc): a newer task with the same key cancels this one
        self.cancel_s: Optional[float] = None    # how long a CLI child took to stop after the cancel
        self.canceled_at: Optional[float] = None
        self.weight = 1               # CPU slots the job holds (an ML clean: its worker's threads)
        self.acquire: Optional[Callable[["Task"], Optional[Callable[[], None]]]] = None
        # ^ runs before the slots are taken (an ML clean waits for the worker
        #   lease there, holding no slot); returns the release callback

    @property
    def elapsed(self) -> float:
        start = self.started or self.created
        return (self.ended or time.time()) - start

    def record(self) -> Dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "title": self.title, "doc": self.doc, "state": self.state,
                "progress": round(self.progress, 4), "message": self.message, "log": self.log,
                "started": self.started, "ended": self.ended, "elapsed": round(self.elapsed, 3),
                "error": self.error, "traceback": self.traceback if self.state == "failed" else None,
                "has_result": self.result is not None and self.state == "done",
                "retry_of": self.retry_of, "client": self.client, "pid": self.pid,
                "cancel_s": None if self.cancel_s is None else round(self.cancel_s, 3),
                "superseded": bool(self.supersede_key) and self.state == "canceled" and self.message == "superseded"}


class InlineCtx:
    """The pool_worker.Ctx interface for a job run in the dispatcher thread."""

    def __init__(self, task: "Task", manager: "TaskManager"):
        self.task, self.manager = task, manager

    def progress(self, fraction: float, message: str = "", force: bool = False) -> None:
        self.manager.progress(self.task, fraction, message)

    def check(self) -> None:
        if self.task.cancel_requested:
            from .pool_worker import JobCanceled
            raise JobCanceled("canceled")

    @property
    def canceled(self) -> bool:
        return self.task.cancel_requested

    def set_pid(self, pid: Optional[int]) -> None:
        """The CLI child running for this task (clirun.run), or None when it ended."""
        self.task.pid = pid

    def note_cancel(self, seconds: float) -> None:
        self.task.cancel_s = seconds


class TaskManager:
    def __init__(self, pool_factory: Callable[["TaskManager"], Any]):
        self.lock = threading.Lock()
        self.tasks: "OrderedDict[str, Task]" = OrderedDict()
        self.subscribers: List[_stdlib_queue.Queue] = []
        self._pool = None
        self._pool_factory = pool_factory
        self.by_slot: Dict[int, Task] = {}
        self.on_cancel: Optional[Callable[[Task], None]] = None   # e.g. the ML host's grace-then-kill

    @property
    def pool(self):
        if self._pool is None:
            self._pool = self._pool_factory(self)
        return self._pool

    def pool_if_started(self):
        return self._pool

    # -- events -------------------------------------------------------------
    def subscribe(self) -> _stdlib_queue.Queue:
        q: _stdlib_queue.Queue = _stdlib_queue.Queue(maxsize=EVENT_QUEUE)
        with self.lock:
            self.subscribers.append(q)
        return q

    def unsubscribe(self, q) -> None:
        with self.lock:
            if q in self.subscribers:
                self.subscribers.remove(q)

    def broadcast(self, event: str, data: Dict[str, Any]) -> None:
        with self.lock:
            subs = list(self.subscribers)
        for q in subs:
            while True:
                try:
                    q.put_nowait((event, data))
                    break
                except _stdlib_queue.Full:
                    try:
                        q.get_nowait()          # the oldest event goes
                        q.task_done()
                    except _stdlib_queue.Empty:
                        break

    def delivered(self, timeout: float) -> bool:
        """Wait up to ``timeout`` s until every event broadcast so far has been
        written to each open stream (the stream calls task_done() after its
        flush). False when the time ran out first."""
        end = time.monotonic() + timeout
        while True:
            with self.lock:
                subs = list(self.subscribers)
            if all(q.unfinished_tasks == 0 for q in subs):
                return True
            if time.monotonic() >= end:
                return False
            time.sleep(0.02)

    def _publish(self, task: Task, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - task.last_sent < PROGRESS_EVERY_S:
            return
        task.last_sent = now
        if task.quiet:
            return
        self.broadcast("task", task.record())

    # -- tasks --------------------------------------------------------------
    def submit(self, kind: str, title: str, job: str, payload: Dict[str, Any], doc: Optional[str] = None,
               client: Optional[str] = None, finish: Optional[Callable[[Any], Any]] = None,
               inline: Optional[Callable] = None, quiet: bool = False, supersede: Optional[Any] = None,
               weight: int = 1, acquire: Optional[Callable] = None) -> Task:
        """``supersede``: a key such as (kind, doc); a queued or running task
        with the same key is canceled at once (typing in Setup: the newer
        preview wins, plan B2)."""
        t = Task(kind, title, job, payload, doc, client, finish, inline)
        t.quiet = bool(quiet)
        t.supersede_key = supersede
        t.weight = max(1, int(weight))
        t.acquire = acquire
        older = []
        with self.lock:
            if supersede is not None:
                older = [o for o in self.tasks.values()
                         if o.supersede_key == supersede and o.state in ("queued", "running")]
            self.tasks[t.id] = t
            self._trim()
        for o in older:
            o.message = "superseded"
            self.cancel(o.id)
        self._publish(t, force=True)
        threading.Thread(target=self._dispatch, args=(t,), name=f"mace-task-{t.id}", daemon=True).start()
        return t

    def retry(self, tid: str) -> Task:
        old = self.get(tid)
        t = self.submit(old.kind, old.title, old.job, old.payload, old.doc, old.client, old.finish, old.inline)
        t.retry_of = old.id
        return t

    def _trim(self) -> None:
        done = [k for k, v in self.tasks.items() if v.state in ("done", "failed", "canceled")]
        while len(done) > KEEP_DONE:
            self.tasks.pop(done.pop(0), None)
        now = time.time()
        for v in self.tasks.values():   # results nobody fetched are let go of
            if v.result is not None and v.ended and now - v.ended > RESULT_TTL_S:
                v.result = None

    def get(self, tid: str) -> Task:
        with self.lock:
            t = self.tasks.get(tid or "")
        if t is None:
            raise KeyError(f"task {tid}")
        return t

    def list(self) -> List[Dict[str, Any]]:
        with self.lock:
            return [t.record() for t in self.tasks.values() if not t.quiet]

    def remove(self, tid: str) -> bool:
        with self.lock:
            t = self.tasks.get(tid)
            if t is None or t.state in ("queued", "running"):
                return False
            del self.tasks[tid]
        if not t.quiet:
            self.broadcast("task-removed", {"id": tid})
        return True

    def progress(self, task: Task, fraction: float, message: str = "") -> None:
        if task.state != "running":
            return
        task.progress = max(task.progress, float(fraction))
        if message:
            task.message = message
            task.log = message
        self._publish(task)

    def on_pool_progress(self, slot: int, fraction: float, message: str) -> None:
        t = self.by_slot.get(slot)
        if t is not None:
            self.progress(t, fraction, message)

    def cancel(self, tid: str) -> Dict[str, Any]:
        t = self.get(tid)
        if t.state not in ("queued", "running"):
            return t.record()
        t.cancel_requested = True
        t.canceled_at = time.time()
        if t.future is not None and t.future.cancel():      # still queued in the pool
            pass
        if t.slot is not None and self._pool is not None:
            self._pool.set_cancel(t.slot)
        if self.on_cancel is not None:
            try:
                self.on_cancel(t)
            except Exception:
                pass
        return t.record()

    def _set_state(self, t: Task, state: str, **kw) -> None:
        t.state = state
        for k, v in kw.items():
            setattr(t, k, v)
        if state in ("done", "failed", "canceled"):
            t.ended = time.time()
            if state == "canceled":
                t.log = ("Superseded by a newer run" if t.message == "superseded"
                         else getattr(t, "cancel_note", None) or f"Canceled after {t.elapsed:.1f} s")
        try:
            self._publish(t, force=True)
        finally:
            if state in ("done", "failed", "canceled"):
                t.done.set()            # after the event is queued: a stop that waits on it sends it too

    def _dispatch(self, t: Task) -> None:
        from .pool_worker import JobCanceled
        release = None
        try:
            if t.acquire is not None:          # e.g. the ML worker's lease, before any CPU slot is held
                release = t.acquire(t)
            with compute.GATE.slot(canceled=lambda: t.cancel_requested, pool=t.inline is None, weight=t.weight):
                if t.cancel_requested:
                    raise JobCanceled("canceled")
                t.started = time.time()
                self._set_state(t, "running")
                if t.inline is not None:
                    raw = t.inline(InlineCtx(t, self))
                else:
                    pool = self.pool
                    t.slot = pool.acquire_slot()
                    self.by_slot[t.slot] = t
                    try:
                        if t.cancel_requested:
                            raise JobCanceled("canceled")
                        raw = self._run_in_pool(t, pool)
                    finally:
                        self.by_slot.pop(t.slot, None)
                        pool.release_slot(t.slot)
            if t.cancel_requested:
                raise JobCanceled("canceled")
            result = t.finish(raw) if t.finish is not None else raw
            t.progress = 1.0
            t.result = result
            self._set_state(t, "done", message=t.message or "done")
        except (JobCanceled, compute.Canceled):
            self._set_state(t, "canceled", message="superseded" if t.message == "superseded" else "canceled")
        except BaseException as exc:                 # the task fails; the server keeps serving
            if t.cancel_requested:          # a cancel cut it short (the pool's own cancel error too, a stopped CLI)
                self._set_state(t, "canceled", message="superseded" if t.message == "superseded" else "canceled")
                return
            tb = getattr(exc, "mace_traceback", None)
            cause = getattr(exc, "__cause__", None)
            tb = tb or getattr(cause, "tb", None) or "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            line = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
            self._set_state(t, "failed", error=line, traceback=tb, log=line, message="failed")
        finally:
            if release is not None:
                try:
                    release()
                except Exception:
                    pass

    def _run_in_pool(self, t: Task, pool) -> Any:
        """The job in the pool. When the pool breaks (a worker was killed, e.g.
        by a per-process CPU-time limit), only the task whose worker died
        fails; any other task caught in the break runs again once, on a new
        executor (pool jobs are pure functions of their payload)."""
        from concurrent.futures.process import BrokenProcessPool
        from .pool_worker import NOT_IDEMPOTENT
        t.future = pool.submit(t.job, t.payload, t.slot)
        if t.cancel_requested:          # canceled while the pool started: its cancel flag did not exist yet
            pool.set_cancel(t.slot)
        try:
            return t.future.result()
        except BrokenProcessPool:
            if t.cancel_requested:
                raise
            pool.recover(getattr(t.future, "mace_executor", None))
            death = pool.slot_death(t.slot) if hasattr(pool, "slot_death") else None
            # SIGTERM is how the executor ends the other workers of a broken pool: those tasks run again
            if death is not None and death["exitcode"] != -15:
                from . import cpulimit
                if cpulimit.killed_by_limit(death["exitcode"], death["cpu"], death["limit"], death.get("sample_s", 0)):
                    raise RuntimeError(cpulimit.MESSAGE.format(m=cpulimit.minutes(death["limit"]), what="task"))
                raise RuntimeError(f"The worker process was killed (signal {-death['exitcode']}).")
            if t.job in NOT_IDEMPOTENT or getattr(t, "resubmitted", False):
                raise
            t.resubmitted = True
            t.message = t.log = "restarted after another task's worker stopped"
            t.future = pool.submit(t.job, t.payload, t.slot)
            if t.cancel_requested:
                pool.set_cancel(t.slot)
            return t.future.result()

    def wait(self, t: Task, seconds: float = WAIT_S) -> bool:
        return t.done.wait(seconds)

    def running_count(self) -> int:
        """Tasks the user started that are queued or running (quiet previews
        not counted): what keeps a managed server alive (contract X1)."""
        with self.lock:
            return sum(1 for t in self.tasks.values()
                       if t.state in ("queued", "running") and not getattr(t, "quiet", False))

    def running(self) -> List[Dict[str, Any]]:
        with self.lock:
            return [t.record() for t in self.tasks.values()
                    if t.state in ("queued", "running") and not getattr(t, "quiet", False)]

    def report(self) -> Dict[str, Any]:
        with self.lock:
            states = [t.state for t in self.tasks.values()]
        return {s: states.count(s) for s in STATES}


def answer(handler, manager: TaskManager, t: Task, wait_s: float = WAIT_S):
    """The common route ending: the result now when the task was quick, else 202 {task}."""
    if manager.wait(t, wait_s):
        if t.state == "done":
            res = t.result
            t.result = None                  # handed over: not kept twice
            if isinstance(res, dict):
                res = {**res, "task": t.record()}
            return handler._json(res)
        if t.state == "canceled":
            return handler._json({"canceled": True, "task": t.record()}, 409)
        return handler._json({"error": t.error or "the task failed", "task": t.record(),
                              "traceback": t.traceback}, 500)
    return handler._json({"detached": True, "task": t.record()}, 202)
