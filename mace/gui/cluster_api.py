"""The GUI's Cluster view and Job documents: ``/api/cluster/*``.

Everything here reads through ``mace.cluster`` (cached, rate-limited,
read-only). The server answers from the cache at once and refreshes past the
TTL in the background (stale-while-revalidate). One poller per server
(``mace.cluster.poller``) watches the user's own jobs and emits the single
``jobs.finished`` server event (Jobs tab toasts and browser notifications
both listen to it); it ticks only while a page is connected and a cluster
view is visible or an own job of this folder is queued or running.

Job files are judged by their real path: inside the served folder they open
here; inside another allowed root (your home or scratch) the progress is
computed but the text is not served in this window; anywhere else nothing
is read. On a network-bound server (``--insecure-network``) the routes that
touch your files or the controller (job, scan, estimate, predict) are off.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from mace.cluster import poller as poller_mod
from mace.cluster import query, reader, views

NETWORK_OFF = "Turned off on a network-bound server"
CREDENTIAL_PARTS = (".ssh", ".gnupg", ".aws", ".config/gcloud", ".kube", ".netrc", ".pgpass", ".docker")


def _loopback(handler) -> bool:
    from .server import is_loopback
    try:
        return is_loopback(handler.server.server_address[0])
    except Exception:
        return True


class ClusterService:
    """Per-server state: the poller, what is visible, the watched job ids."""

    def __init__(self, st):
        self.st = st
        self.visible_until = 0.0
        self.lock = threading.Lock()
        self.poller: Optional[poller_mod.Poller] = None
        self.software: Dict[str, Any] = {}             # the Software tab's answer: {state, result, error}
        self.slurm = reader.slurm_present()
        def in_folder(workdir: str) -> bool:
            if not workdir:
                return False
            root = str(st.root)                        # the served folder now (Open Folder can change it)
            rp = os.path.realpath(workdir)
            return rp == root or rp.startswith(root.rstrip("/") + "/")
        self.in_folder = in_folder
        if self.slurm:
            self.poller = poller_mod.Poller(self._emit, self.interval, self.active, workdir_filter=in_folder)

    def start(self):
        if self.poller is not None:
            self.poller.start()

    def stop(self):
        if self.poller is not None:
            self.poller.stop()

    def interval(self) -> float:
        try:
            from . import settings as gui_settings
            stored = gui_settings.load().get("cluster.pollSeconds")
        except Exception:
            stored = None
        return poller_mod.poll_seconds(stored)

    def mark_visible(self):
        first = self.visible_until < time.time()
        self.visible_until = time.time() + 3 * self.interval()
        if first and self.poller is not None:
            self.poller.nudge()

    def active(self) -> bool:
        if not self.st.clients:
            return False
        if time.time() < self.visible_until:
            return True
        return self.poller is not None and self.poller.own_active_in_folder()

    def _emit(self, name: str, data: Dict[str, Any]) -> None:
        # jobs.finished goes through the server's once-per-job record (shared with the Jobs panel)
        announce = getattr(self.st, "_job_event", None)
        if name == "jobs.finished" and announce is not None:
            announce(name, data)
            return
        self.st.tasks.broadcast(name, data)

    # -- paths ---------------------------------------------------------------
    def allowed_roots(self):
        try:
            from . import roots                       # PLAN_connect D12, when merged
            return [str(p) for p in roots.allowed_roots(None, os.environ)]
        except Exception:
            from mace.cluster.quotas import scratch_roots
            return [os.path.expanduser("~")] + scratch_roots()

    def research_real(self):
        try:
            from . import roots
            return {os.path.realpath(str(p)) for p in roots.research_roots()}
        except Exception:
            return set()

    def path_policy(self, real: str) -> str:
        root = str(self.st.root)
        if real == root or real.startswith(root.rstrip("/") + "/"):
            return "in_root"
        home = os.path.realpath(os.path.expanduser("~"))
        rel = os.path.relpath(real, home) if real.startswith(home + "/") else ""
        if rel and any(rel == c or rel.startswith(c + "/") for c in CREDENTIAL_PARTS):
            return "outside"
        for r in self.allowed_roots():
            rr = os.path.realpath(r)
            if real == rr or real.startswith(rr.rstrip("/") + "/"):
                try:
                    # a research space is shared with your group: its files belong to the group's members
                    if os.stat(real).st_uid != os.getuid() and rr not in self.research_real():
                        return "outside"
                except OSError:
                    pass
                return "other_root"
        return "outside"

    def software_answer(self, refresh: bool = False) -> Dict[str, Any]:
        """The Software tab's rows; the module probes (a subshell each, up to
        a minute on a busy login node) run once in a thread, and the tab polls."""
        from mace.cluster import software
        with self.lock:
            cur = dict(self.software)
            if cur.get("state") == "checking" or (cur.get("state") == "done" and not refresh):
                return cur
            self.software = {"state": "checking"}

        def work():
            try:
                ans = {"state": "done", "result": software.rows(refresh=refresh)}
            except Exception as exc:                  # pragma: no cover - rows() catches its own
                ans = {"state": "done", "error": f"{type(exc).__name__}: {exc}"}
            with self.lock:
                self.software = ans
        threading.Thread(target=work, name="mace-cluster-software", daemon=True).start()
        return {"state": "checking"}

    def scratch_ok(self, path: str) -> bool:
        from mace.cluster.quotas import scratch_roots
        real = os.path.realpath(path)
        for r in scratch_roots():
            rr = os.path.realpath(r)
            if real == rr or real.startswith(rr.rstrip("/") + "/"):
                return True
        return "/.mace_scratch/" in real + "/" and self.path_policy(real) in ("in_root", "other_root")


def service(st) -> ClusterService:
    svc = getattr(st, "cluster", None)
    if svc is None:
        svc = st.cluster = ClusterService(st)
    return svc


def start(st) -> None:
    """Called once by make_server."""
    service(st).start()


def _bool(v) -> bool:
    return str(v or "").lower() in ("1", "true", "yes", "on")


def _abbrev(path: Optional[str]) -> Optional[str]:
    if not path:
        return path
    home = os.path.expanduser("~")
    return "~" + path[len(home):] if path.startswith(home + "/") else path


def _shell_path(p: str) -> str:
    """A path for a command to copy: quoted, with a leading ``~/`` left
    outside the quotes so the shell still expands it."""
    import shlex
    if p.startswith("~/"):
        return "~/" + shlex.quote(p[2:])
    return shlex.quote(p)


def _job_payload(st, svc, data: Dict[str, Any]) -> Dict[str, Any]:
    files = data.get("files") or {}
    access = data.get("access") or {}
    entries = []
    for k, label in (("workdir", "Work dir"), ("out", "Output (live)"), ("stdout", "Job log"),
                     ("stderr", "Job error log"), ("deck", "Input deck"), ("scratch", "Scratch dir")):
        p = files.get(k)
        if not p or (k == "stderr" and p == files.get("stdout")):
            continue
        real = os.path.realpath(p)
        a = access.get(k) or svc.path_policy(real)
        e = {"kind": k, "label": label, "path": _abbrev(p), "access": a}
        if a == "in_root":
            e["rel"] = os.path.relpath(real, str(st.root))
            e["exists"] = os.path.exists(real)
        elif a == "other_root":
            e["open_dir"] = _abbrev(os.path.dirname(real) if k != "workdir" else real)
            e["command"] = f"mace gui --dir {_shell_path(e['open_dir'])}"
        entries.append(e)
    out = {k: v for k, v in data.items() if k not in ("files", "access")}
    out["file_entries"] = entries
    out["in_root"] = any(e["access"] == "in_root" and e["kind"] == "out" for e in entries)
    return out


def _scan_job(root: str, sizes: Optional[str], max_seconds: Optional[float]):
    def job(ctx):
        fd, out = tempfile.mkstemp(prefix="mace-scan-", suffix=".json")
        os.close(fd)
        pkg_root = str(Path(__file__).resolve().parents[2])
        env = dict(os.environ)
        env["PYTHONPATH"] = pkg_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        cmd = [sys.executable, "-m", "mace.cluster.filescan", root, "--out", out]
        if sizes:
            cmd += ["--sizes", sizes]
        if max_seconds:
            cmd += ["--max-seconds", str(max_seconds)]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env,
                                cwd=pkg_root, start_new_session=True)
        ctx.set_pid(proc.pid)
        last = {}
        busy = []

        def read():
            for line in proc.stdout:
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                if msg.get("error"):
                    busy.append(msg["error"])
                last.update(msg)
        t = threading.Thread(target=read, daemon=True)
        t.start()
        try:
            while proc.poll() is None:
                if ctx.canceled:
                    proc.terminate()
                    try:
                        proc.wait(5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    ctx.check()
                if last.get("entries") is not None:
                    ctx.progress(min(0.95, (last.get("elapsed") or 0) / float(max_seconds or 600)),
                                 f"{last['entries']:,} entries in {last.get('dirs', 0):,} folders")
                time.sleep(0.2)
            t.join(2)
        finally:
            ctx.set_pid(None)
        if busy:
            raise RuntimeError(busy[0])
        try:
            res = json.loads(Path(out).read_text())
        finally:
            try:
                os.unlink(out)
            except OSError:
                pass
        return res
    return job


# ------------------------------------------------------------------ routes
def get(handler, route: str, query_args) -> None:
    st = handler.state
    svc = service(st)
    q = lambda k, d=None: (query_args.get(k) or [d])[0]
    refresh = _bool(q("refresh"))
    tag = "refresh" if refresh else "revalidate"
    if route == "capabilities":
        caps = views.capabilities(str(st.root))
        caps["network_bound"] = not _loopback(handler)
        caps["poll_seconds"] = svc.interval()
        caps["events"] = list(svc.poller.events[-20:]) if svc.poller else []
        from mace.cluster import software
        caps["module_system"] = software.module_system()
        caps["software"] = software.available() and _loopback(handler)
        return handler._json(caps)
    if route in ("overview", "nodes", "reservations", "limits", "jobs") and not svc.slurm:
        return handler._json({"error": views.NO_SLURM, "no_slurm": True}, 503)
    svc.mark_visible()
    if route == "overview":
        return handler._json(views.overview(refresh=refresh, background=not refresh, tag=tag,
                                            served_root=str(st.root)))
    if route == "nodes":
        return handler._json(views.nodes_view(group_by=q("group", "feature"), all_partitions=_bool(q("all")),
                                              refresh=refresh, background=not refresh, tag=tag,
                                              page=int(q("page", "0") or 0), feature=q("feature") or None))
    if route == "reservations":
        return handler._json(views.reservations_view(refresh=refresh, background=not refresh, tag=tag))
    if route == "limits":
        n = q("n")
        return handler._json(views.limits_view(check_submit=int(n) if n else None, cpus=int(q("cpus", "1") or 1),
                                               refresh=refresh, background=not refresh, tag=tag))
    if route == "jobs":
        r = reader.squeue_own(refresh=refresh, background=not refresh, tag=tag)
        rows = [{**x, "in_folder": svc.in_folder(x.get("workdir") or "")} for x in (r.value or [])]
        return handler._json({"jobs": rows, **views._meta(r)})
    if route == "quotas":
        return handler._json(views.quotas_view(refresh=refresh, tag=tag, served_root=str(st.root),
                                               background=not refresh))
    if route == "job":
        if not _loopback(handler):
            return handler._error(403, NETWORK_OFF)
        if not svc.slurm:
            return handler._json({"error": views.NO_SLURM, "no_slurm": True}, 503)
        jid = q("id") or ""
        from mace.cluster import jobinfo
        if not reader.valid_job_id(jid):
            return handler._error(400, f"not a job id: {jid[:40]!r}")
        try:
            data = jobinfo.job(jid, refresh=refresh, tag=tag, use_engine=True, path_policy=svc.path_policy,
                               scratch_ok=svc.scratch_ok, batch_sacct_now=svc.poller is None)
        except jobinfo.NotYours:
            return handler._error(404, "Not one of your jobs")
        if svc.poller is not None:
            svc.poller.watch(jid)
        return handler._json(_job_payload(st, svc, data))
    if route == "software":
        if not _loopback(handler):                   # it runs the login profile's module command
            return handler._error(403, NETWORK_OFF)
        return handler._json(svc.software_answer(refresh=refresh))
    if route == "scan":
        if not _loopback(handler):
            return handler._error(403, NETWORK_OFF)
        t = st.tasks.get(q("task") or "")
        rec = t.record()
        if t.state == "done" and t.result is not None:
            rec["result"] = t.result
        return handler._json(rec)
    return handler._error(404, f"unknown API route cluster/{route}")


def post(handler, route: str, body: Dict[str, Any]) -> None:
    st = handler.state
    svc = service(st)
    if route in ("predict", "scan", "estimate", "job/live", "job/close") and not _loopback(handler):
        return handler._error(403, NETWORK_OFF)
    if route == "fit":
        if not svc.slurm:
            return handler._json({"error": views.NO_SLURM, "no_slurm": True}, 503)
        from mace.cluster import fit as fit_mod
        req = fit_mod.request_from(int(body.get("cpus") or 0), int(body.get("nodes") or 1), body.get("mem_per_cpu"),
                                   body.get("mem"), body.get("gpus"), body.get("constraint"), body.get("partition"),
                                   body.get("time"))
        svc.mark_visible()
        return handler._json(views.fit_view(req, background=True, tag="revalidate"))
    if route == "predict":
        from mace.cluster import fit as fit_mod
        script = None
        req = None
        if body.get("script_rel"):
            p = st.resolve(str(body["script_rel"]))         # the path is checked first, SLURM or not
            if p.suffix != ".sh" or not p.is_file():
                raise ValueError("Predict Start needs a .sh job script in this folder")
            script = p
        if not svc.slurm:
            return handler._json({"error": views.NO_SLURM, "no_slurm": True}, 503)
        if script is None:
            req = fit_mod.request_from(int(body.get("cpus") or 0), int(body.get("nodes") or 1),
                                       body.get("mem_per_cpu"), body.get("mem"), body.get("gpus"),
                                       body.get("constraint"), body.get("partition"), body.get("time"))
        return handler._json(views.fit_view(req or {}, predict=True, script_path=script, server_gap=True,
                                            tag="refresh", background=True))
    if route == "scan":
        root = body.get("root") or "."
        p = st.resolve(str(root)) if not os.path.isabs(str(root)) else Path(root)
        real = os.path.realpath(str(p))
        if svc.path_policy(real) == "outside" or not os.path.isdir(real):
            raise PermissionError("Scans run only in this folder, your home or your scratch folders")
        sizes = body.get("sizes") if body.get("sizes") in ("top", "all", "off") else None
        maxs = float(body["max_seconds"]) if body.get("max_seconds") else None
        from .server import cli_task
        t = cli_task(st, "cluster.scan", f"Find what uses files in {_abbrev(real)}", _scan_job(real, sizes, maxs),
                     client=handler.headers.get("X-Mace-Client"), supersede=("cluster.scan",))
        return handler._json({"task": t.record()}, 202)
    if route == "job/live":
        jid = str(body.get("id") or "")
        if not reader.valid_job_id(jid):
            raise ValueError("not a job id")
        from mace.cluster import jobinfo
        if jobinfo.own_row(jid) is None and jid not in reader.sacct_for([jid]):
            return handler._error(404, "Not one of your jobs")
        r = reader.sstat(jid, tag="refresh")
        return handler._json({"rows": r.value, "efficiency": jobinfo.live_efficiency(jid, r.value or []),
                              **r.as_dict()})
    if route == "job/close":
        jid = str(body.get("id") or "")
        if svc.poller is not None and reader.valid_job_id(jid):
            svc.poller.unwatch(jid)
        return handler._json({"ok": True})
    if route == "estimate":
        from mace.cluster import history
        p = st.resolve(str(body.get("deck_rel") or ""))
        if p.suffix != ".d12" or not p.is_file():
            raise ValueError("Estimate needs a .d12 deck in this folder")
        cores = int(body["cores"]) if body.get("cores") else None
        v = history.estimate(p, cores=cores, history=p.parent, use_sacct=svc.slurm)
        # the Submit dialog's runtime estimate (1.1.6): what `mace cluster estimate DECK` prints, and that command
        import shlex
        from mace.cluster.cli import estimate_text
        rel = st.rel(p)
        v["text"] = estimate_text(v)
        v["command"] = "mace cluster estimate " + shlex.quote(rel) + (f" --cores {cores}" if cores else "")
        return handler._json(v)
    return handler._error(404, f"unknown API route cluster/{route}")
