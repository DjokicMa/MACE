"""The GUI server's one ML worker and its lease (plan: Clean tiers, §8).

An ML clean runs as a pool job like every other clean; the model, though,
is loaded once, in one worker process owned by the server (eight pool
workers each loading MACE-MPA-0 would take about 11 GB). The worker serves a
private Unix socket (``mlip_client.WorkerProcess(use_socket=True)``, nice
10); the pool job connects to it.

* ``acquire(task, model_id, threads)`` runs in the task's dispatcher thread
  *before* any CPU slot is taken: it waits (cancelably) while another job
  holds the worker, pings it, restarts it when it died or another model is
  asked for, and hands back the lease ``{"model", "socket", "hello"}``. A
  worker that cannot start gives a lease with an ``error`` instead: the
  clean then keeps the spring result and shows why.
* ``cancel_lease(task)``: the canceled job closes its socket; an in-flight
  call gets ``GRACE_S`` to finish (a ping then answers and the loaded model
  stays); past that the worker's process group is killed and the next job
  reloads.
* The worker stops after ``IDLE_S`` without a job, at server shutdown, and
  by itself when the server dies (its stdin closes).

The server's threads only start, ping and stop the worker; they relay no
data and run no optimizer.
"""
from __future__ import annotations

import atexit
import sys
import threading
import time
from typing import Any, Callable, Dict, Optional

GRACE_S = 5.0
IDLE_S = 300.0
NICE = 10


class MlipHost:
    def __init__(self, nice: int = NICE):
        self.lock = threading.Condition()
        self.worker = None
        self.model_id: Optional[str] = None
        self.owner: Optional[str] = None           # the task id holding the lease
        self.nice = nice
        self.loads = 0                             # hellos seen (one per model load)
        self.kills = 0
        self.idle_s = IDLE_S
        self._idle: Optional[threading.Timer] = None
        self.last_error: Optional[str] = None
        self.cpu_restarts = 0                      # restarts before a per-process CPU-time limit
        atexit.register(self.shutdown)

    # -- the lease ---------------------------------------------------------------
    def acquire(self, task, model_id: str, threads: int) -> Callable[[], None]:
        """Wait for the worker, make sure it runs ``model_id``, put the lease
        in ``task.payload["mlip_lease"]``; returns the release callback."""
        from mace.gui import compute
        from mace.preopt import detect, mlip_client, proc, registry
        canceled = lambda: bool(getattr(task, "cancel_requested", False))  # noqa: E731
        with self.lock:
            told = False
            while self.owner is not None:
                if canceled():
                    raise compute.Canceled("canceled while waiting for the ML worker")
                if not told:
                    task.message = task.log = "queued: ML worker busy"
                    told = True
                self.lock.wait(0.1)
            self.owner = task.id
            if self._idle is not None:
                self._idle.cancel()
                self._idle = None
        lease: Dict[str, Any] = {"model": model_id}
        try:
            w = self.worker
            if w is not None and self._near_cpu_limit(w, task, threads):
                # reloading costs seconds; a kill in the middle of a run costs the run
                self._stop_worker()
                w = None
                self.cpu_restarts += 1
                task.message = task.log = "Reloading the ML model (CPU-time limit)…"
                print("mace gui: restarted the ML worker before this node's CPU-time limit", file=sys.stderr, flush=True)
            if w is not None and (self.model_id != model_id or not w.alive() or not self._ping(w)):
                self._stop_worker()
                w = None
            if w is None:
                model = registry.get(model_id)
                py = detect.configured_python()
                ws = detect.weights_status(model) if model else {"status": "unknown"}
                mem = proc.mem_available_gb()
                if model is None or not py:
                    raise mlip_client.MlipError("setup", "no ML environment is set up (mace mlip setup)")
                if ws["status"] not in ("ready", "bundled"):
                    raise mlip_client.MlipError("weights", ws.get("reason") or ws["status"])
                if mem is not None and mem < 2.0:
                    raise mlip_client.MlipError("memory", f"only {mem:.1f} GB of RAM available (an ML model "
                                                          "needs about 2 GB)")
                w = mlip_client.WorkerProcess(py, model, weights=ws.get("path"), threads=threads, use_socket=True,
                                              nice=self.nice, canceled=canceled)
                with self.lock:
                    self.worker, self.model_id = w, model_id
                    self.loads += 1
            lease.update(socket=w.socket_path, hello=w.hello, pid=w.pid)
        except proc.Canceled:
            self._release(task.id)
            raise compute.Canceled("canceled while the ML worker was loading")
        except mlip_client.MlipError as exc:
            self.last_error = str(exc)
            lease["error"] = f"{exc.kind}: {exc}"
        except Exception as exc:              # anything else (a temp dir, the registry): the clean keeps the springs
            self.last_error = str(exc)
            lease["error"] = f"exception: {(str(exc).strip().splitlines() or [type(exc).__name__])[0]}"
        except BaseException:
            self._release(task.id)            # never leave the lease held by a task that is gone
            raise
        task.payload["mlip_lease"] = lease
        return lambda: self._release(task.id)

    @staticmethod
    def _near_cpu_limit(w, task, threads: int) -> bool:
        """Restart before reuse at 60 % of the limit, or when one run of this
        Clean (its time limit x threads) could cross 90 % of it."""
        from mace.gui import cpulimit
        lim = cpulimit.limit()
        pid = getattr(w, "pid", None)
        if lim is None or not pid:
            return False
        used = cpulimit.used(pid)
        if used is None:
            return False
        try:
            run = float((getattr(task, "payload", None) or {}).get("time_limit") or 0) * max(1, int(threads))
        except (TypeError, ValueError):
            run = 0.0
        return used >= cpulimit.RESTART_AT * lim or used + run >= 0.9 * lim

    def _release(self, tid: str) -> None:
        with self.lock:
            if self.owner == tid:
                self.owner = None
                self.lock.notify_all()
                if self.worker is not None:
                    self._idle = threading.Timer(self.idle_s, self._idle_shutdown)
                    self._idle.daemon = True
                    self._idle.start()

    def cancel_lease(self, task) -> None:
        """The job holding the lease was canceled: grace, then kill."""
        if self.owner != getattr(task, "id", None) or self.worker is None:
            return
        w = self.worker

        def watch():
            if not self._ping(w, GRACE_S):
                with self.lock:
                    if self.worker is w:
                        self._stop_worker()
                        self.kills += 1
        threading.Thread(target=watch, name="mace-mlip-grace", daemon=True).start()

    # -- worker lifecycle ----------------------------------------------------------
    @staticmethod
    def _ping(w, timeout: float = 5.0) -> bool:
        from mace.preopt import mlip_client
        try:
            c = mlip_client.SocketClient(w.socket_path, connect_timeout=timeout)
        except Exception:
            return False
        try:
            return c.request({"op": "ping"}, timeout).get("op") == "pong"
        except Exception:
            return False
        finally:
            c.close()

    def _stop_worker(self) -> None:
        w, self.worker, self.model_id = self.worker, None, None
        if w is not None:
            w.kill()
            w.close()

    def _idle_shutdown(self) -> None:
        with self.lock:
            if self.owner is None:
                self._stop_worker()
            self._idle = None

    def shutdown(self) -> None:
        with self.lock:
            if self._idle is not None:
                self._idle.cancel()
                self._idle = None
            self._stop_worker()

    def report(self) -> Dict[str, Any]:
        w = self.worker
        return {"running": bool(w and w.alive()), "pid": w.pid if w else None, "model": self.model_id,
                "owner": self.owner, "loads": self.loads, "kills": self.kills, "last_error": self.last_error,
                "cpu_restarts": self.cpu_restarts}
