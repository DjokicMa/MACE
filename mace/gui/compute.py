"""Bounded CPU work in the request threads (spec section 9.1).

``ThreadingHTTPServer`` gives every request its own thread, so a long drag
in the editor (one symmetry preview per 150 ms) could otherwise keep several
cores busy with spglib. Every spglib call made for the GUI goes through one
``ComputeGate``: at most ``cap`` run at once (``resources.effective_workers``,
a quarter of the cores by default), and the rest wait without using CPU.

Calls that carry a ``key`` (the document a symmetry preview is for) are
coalesced: one call per key runs at a time, and while another waits, a newer
call with the same key replaces it (the older one returns ``SUPERSEDED``
without running). So at most one call per key runs and one waits, whatever
the browser sends.

The process pool (pool.py) draws on the same cap (spec section 9.1, plan
B3): a pool job holds a ``slot()`` while it runs, so inline calls and pool
jobs together never run more than ``cap`` at once, not cap + cap.

A thread that already holds a slot (a task running a CLI child, an inline
build) and makes a gated call itself (reading a CIF computes its symmetry)
runs that call in the slot it holds: it is the same thread, so no more CPU,
and waiting for a second slot would deadlock once every slot is held that
way (one worker on a 4-core allocation).

A call on a large structure (``heavy``: more than ``HEAVY_ATOMS`` atoms)
also waits for any other heavy call to finish: spglib's working memory on a
5000-atom supercell is some 35 MB while it runs, so heavy calls one at a time
keep that spike single (spec section 9.2), while small structures in other
tabs are not held up behind them.
"""
from __future__ import annotations

import contextlib
import threading
import time
from typing import Any, Callable, Dict, Optional

SUPERSEDED = object()
HEAVY_ATOMS = 2000
RESERVE_AFTER_S = 2.0


class Canceled(Exception):
    """A task was canceled (while waiting for a slot, or by its own check)."""


class ComputeGate:
    def __init__(self, cap: int = 1):
        self.cap = max(1, int(cap))
        self.cond = threading.Condition()
        self.running = 0
        self.waiting: Dict[Any, object] = {}
        self.running_keys = set()
        self.heavy_running = 0
        self.pool_running = 0      # slots held by pool jobs (the rest are inline calls)
        self.reserved: Dict[object, int] = {}   # weighted waiters past RESERVE_AFTER_S: what they need
        # counters, for the benchmark rows and tests
        self.max_running = 0
        self.ran = 0
        self.superseded = 0
        self._local = threading.local()      # slots this thread holds

    def _held(self) -> int:
        return getattr(self._local, "n", 0)

    def resize(self, cap: int) -> None:
        with self.cond:
            self.cap = max(1, int(cap))
            self.cond.notify_all()

    def run(self, key: Optional[Any], fn: Callable[[], Any], hold_s: float = 0.0, heavy: bool = False):
        """fn() once a slot is free; SUPERSEDED if a newer call for the same
        key arrived while this one waited. ``hold_s`` (test hooks only) keeps
        the slot that much longer, as a slow spglib would. ``heavy``: also
        wait until no other heavy call runs."""
        if self._held():
            return fn()     # inside this thread's own slot (see the module docstring)
        token = object()
        with self.cond:
            if key is not None:
                if key in self.waiting:
                    self.cond.notify_all()      # the older waiter sees it was replaced
                self.waiting[key] = token
            while True:
                if key is not None and self.waiting.get(key) is not token:
                    self.superseded += 1
                    return SUPERSEDED
                # one call per key at a time: the next waits for it, whatever the cap
                if self.running < self.cap and (key is None or key not in self.running_keys) \
                        and not (heavy and self.heavy_running):
                    break
                self.cond.wait()
            if key is not None:
                del self.waiting[key]
                self.running_keys.add(key)
            self.running += 1
            self.heavy_running += bool(heavy)
            self.ran += 1
            self.max_running = max(self.max_running, self.running)
        self._local.n = 1
        try:
            if hold_s > 0:
                time.sleep(hold_s)
            return fn()
        finally:
            self._local.n = 0
            with self.cond:
                self.running -= 1
                self.heavy_running -= bool(heavy)
                self.running_keys.discard(key)
                self.cond.notify_all()

    @contextlib.contextmanager
    def slot(self, canceled: Optional[Callable[[], bool]] = None, pool: bool = True, weight: int = 1):
        """Hold one slot of the cap for the duration (a pool job, an inline
        build). Not ``heavy``: pool jobs would otherwise run one at a time.
        ``canceled()`` is checked while waiting; when it turns true the wait
        ends with ``Canceled`` and no slot is taken.

        ``weight`` (an ML job with k threads holds k slots): the amount is
        ``min(weight, cap)``, recomputed on every wake-up so a smaller cap
        never strands the waiter, and exactly what was taken is released. A
        weighted waiter that has waited ``RESERVE_AFTER_S`` reserves its
        amount, and new single-slot requests then leave room for it, so a
        stream of small jobs cannot starve it. With weight 1 and no weighted
        waiter this is the single-slot logic as before."""
        if self._held():
            yield           # this thread's slot already covers it
            return
        weight = max(1, int(weight))
        token = object()
        t0 = time.monotonic()
        with self.cond:
            while True:
                need = min(weight, self.cap)
                if weight > 1:
                    if time.monotonic() - t0 > RESERVE_AFTER_S:
                        self.reserved[token] = need
                    # only reservations made before this one count (first come, first
                    # served): two weighted waiters must not each wait for the other
                    order = list(self.reserved)
                    earlier = order[:order.index(token)] if token in self.reserved else order
                    free = self.cap - self.running - sum(self.reserved[k] for k in earlier)
                    if need <= free:
                        break
                else:
                    reserved = sum(self.reserved.values())
                    if self.running + 1 <= self.cap - reserved or (not reserved and self.running < self.cap):
                        break
                if canceled is not None and canceled():
                    self.reserved.pop(token, None)
                    self.cond.notify_all()
                    raise Canceled("canceled while waiting for a free core")
                self.cond.wait(0.1)
            self.reserved.pop(token, None)
            self.running += need
            self.pool_running += need if pool else 0
            self.ran += 1
            self.max_running = max(self.max_running, self.running)
        self._local.n = 1
        try:
            yield
        finally:
            self._local.n = 0
            with self.cond:
                self.running -= need
                self.pool_running -= need if pool else 0
                self.cond.notify_all()

    def budget(self) -> Dict[str, int]:
        """What /api/resources reports: the one cap and what holds it now."""
        with self.cond:
            return {"cap": self.cap, "inline_running": self.running - self.pool_running,
                    "pool_running": self.pool_running, "max_running": self.max_running}

    def stats(self) -> Dict[str, int]:
        with self.cond:
            return {"cap": self.cap, "running": self.running, "waiting": len(self.waiting),
                    "max_running": self.max_running, "ran": self.ran, "superseded": self.superseded,
                    "pool_running": self.pool_running}

    def reset_stats(self) -> None:
        with self.cond:
            self.max_running = self.running
            self.ran = self.superseded = 0


GATE = ComputeGate(1)
