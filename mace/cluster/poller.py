"""The one server-side poller of the user's own jobs.

The GUI server starts one ``Poller`` when SLURM is on PATH; nothing else in
the server polls SLURM. Each tick (every ``cluster.pollSeconds``, 30..300,
default 60) it reads ``squeue_own`` once, finds the jobs seen PENDING or
RUNNING on the last tick and gone now, makes **one** batched ``sacct`` call
for those plus the ids of open Job documents, and emits one
``jobs.finished {id, name, state, exit, workdir}`` event per finished job
(deduplicated by id for the life of the server, so a page reload never
repeats one). The Jobs tab's toasts and the browser notifications both
listen to that one event.

It ticks only while ``active()`` says so: a GUI page is connected and the
Jobs tab, the Cluster view or a Job document is visible, or an own job of
the served folder is pending or running.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set

from mace.cluster import reader

POLL_MIN, POLL_MAX, POLL_DEFAULT = 30, 300, 60
ACTIVE_STATES = ("PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "REQUEUED", "SUSPENDED")
WATCH_GRACE_TICKS = 2


def poll_seconds(stored: Any = None) -> float:
    """``cluster.pollSeconds`` clamped to 30..300; the test hook
    ``MACE_CLUSTER_TEST_POLL_SECONDS`` (only with MACE_GUI_TEST_HOOKS=1) may
    go lower."""
    if os.environ.get("MACE_GUI_TEST_HOOKS") == "1":
        hook = os.environ.get("MACE_CLUSTER_TEST_POLL_SECONDS")
        if hook:
            try:
                return max(0.2, float(hook))
            except ValueError:
                pass
    try:
        v = int(stored)
    except (TypeError, ValueError):
        return float(POLL_DEFAULT)
    return float(min(POLL_MAX, max(POLL_MIN, v)))


class Poller:
    def __init__(self, emit: Callable[[str, Dict[str, Any]], None], interval: Callable[[], float],
                 active: Callable[[], bool], workdir_filter: Optional[Callable[[str], bool]] = None):
        self.emit = emit
        self.interval = interval
        self.active = active
        self.workdir_filter = workdir_filter
        self.prev: Dict[str, Dict[str, Any]] = {}       # id -> row seen active on the last tick
        self.emitted: Set[str] = set()
        self.watched: Dict[str, int] = {}                # id -> ticks left after its document closed (-1: open)
        self.rows: List[Dict[str, Any]] = []
        self.ticks = 0
        self.last_tick: Optional[float] = None
        self.lock = threading.Lock()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.events: List[Dict[str, Any]] = []           # the finished events of this server (newest last)

    # -- watched ids (open Job documents) ----------------------------------
    def watch(self, job_id: str) -> None:
        with self.lock:
            self.watched[str(job_id)] = -1

    def unwatch(self, job_id: str) -> None:
        with self.lock:
            if str(job_id) in self.watched:
                self.watched[str(job_id)] = WATCH_GRACE_TICKS

    def watched_ids(self) -> List[str]:
        with self.lock:
            return sorted(self.watched)

    def own_active_in_folder(self) -> bool:
        """An own job of the served folder pending or running (from the last rows)."""
        for r in self.rows:
            if (r.get("state") or "").upper() in ACTIVE_STATES and (
                    self.workdir_filter is None or self.workdir_filter(r.get("workdir") or "")):
                return True
        return False

    # -- one tick -----------------------------------------------------------
    def tick(self) -> List[Dict[str, Any]]:
        """Poll once; returns the events emitted."""
        max_age = min(60.0, self.interval())
        res = reader.squeue_own(max_age=max_age, tag="poll")
        if res.value is None:
            return []
        rows = res.value
        current = {r["id"]: r for r in rows}
        gone = [i for i in self.prev if i not in current]
        with self.lock:
            for i, left in list(self.watched.items()):
                if left == 0:
                    self.watched.pop(i)
                elif left > 0:
                    self.watched[i] = left - 1
            watched = set(self.watched)
        ask = sorted(set(gone) | watched)
        acct: Dict[str, Dict[str, Any]] = {}
        if ask and reader.query.available("sacct"):
            r = reader.sacct_batch(ask, refresh=bool(gone), tag="poll", max_age=max_age)
            acct = (r.value or {}).get("rows") or {}
        out = []
        for i in gone:
            if i in self.emitted:
                continue
            self.emitted.add(i)
            row = self.prev[i]
            a = acct.get(i) or {}
            state = (a.get("State") or "").split()[0] if a.get("State") else None
            ev = {"id": i, "name": row.get("name"), "state": state, "exit": a.get("ExitCode") or None,
                  "workdir": row.get("workdir"), "left_queue": state is None, "at": time.time()}
            out.append(ev)
            self.events.append(ev)
            self.events = self.events[-200:]
            self.emit("jobs.finished", ev)
        self.prev = {i: r for i, r in current.items() if (r.get("state") or "").upper() in ACTIVE_STATES}
        self.rows = rows
        self.ticks += 1
        self.last_tick = time.time()
        return out

    # -- thread ---------------------------------------------------------------
    def start(self) -> None:
        if self.thread is not None:
            return
        self.thread = threading.Thread(target=self._loop, name="mace-cluster-poller", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def nudge(self) -> None:
        """Run the next tick now (a view became visible)."""
        self._wake.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                if self.active():
                    self.tick()
            except Exception:                    # the poller never dies on a bad answer
                pass
            self._wake.wait(self.interval())
            self._wake.clear()
