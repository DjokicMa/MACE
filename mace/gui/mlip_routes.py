"""Settings › ML potentials: the ML environment and the weights, from the GUI.

GET  mlip/status    the environment (where it came from), each model's status,
                    license and weights, the packages the installer offers, a
                    running install, and whether this host can run one
POST mlip/plan      {dir, backends, gpu, python} -> mlip_install.make_plan (the
                    folder, Python, licenses, size, free space, quota)
POST mlip/install   {dir, backends, gpu, python, fetch, accept_license} -> a task
                    running mlip_install.install (``mace mlip install``'s code):
                    its log streams in the task's ``log``; Cancel stops pip and
                    removes a folder the run created
POST mlip/record    {python} -> a task: probe that interpreter and record it
                    (``mace mlip setup --python``)
POST mlip/fetch     {id, accept_license: true} -> a task: the weights, after the
                    license was shown in the dialog and accepted there

Every POST already needs the session token, ``X-Requested-With`` and a
loopback Host (server.py). A GUI bound to the network installs only into
the default place and records no interpreter the browser names. After a
task succeeds, Clean's and the Calculators' tiers are detected again and an
idle ML worker is stopped, so the new environment is used without a
restart.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

LOG_LINES = 400
_LOCK = threading.Lock()


def _loopback(st) -> bool:
    from .server import is_loopback
    return is_loopback(getattr(st, "bind_host", "127.0.0.1"))


def _models() -> List[Dict[str, Any]]:
    from mace.preopt import detect, registry
    rows = []
    for m in registry.load(scope=None, include_academic=True):
        try:
            ms = detect.mlip_status(m, timeout=detect.PROBE_TIMEOUT_GUI)
        except Exception as exc:              # never take the server down
            ms = {"status": "import_failed", "reason": str(exc), "hint": ""}
        w = m["weights"]
        rows.append({"id": m["id"], "label": m["label"], "package": m["code"]["package"],
                     "code_license": m["code"]["license"], "weights_license": w["license"],
                     "weights_license_url": w.get("license_text_url") or m["code"].get("license_text_url"),
                     "size_mb": round(w["size_bytes"] / 1e6, 1) if "size_bytes" in w else 0.0,
                     "fetchable": "url" in w or bool(m.get("gated")), "status": ms["status"], "reason": ms.get("reason", ""),
                     "hint": ms.get("hint", ""), "default": bool(m.get("default")),
                     "scopes": list(m.get("scopes", registry.SCOPES)), "env": detect.env_name(m),
                     "unrelated_note": m.get("unrelated_note"), "training_data": m.get("training_data") or [],
                     "citation": [c.get("text") for c in m.get("citation") or []],
                     "academic": bool(m.get("academic")), "role": m.get("role"), "limits": m.get("limits"),
                     "academic_ack": (detect.academic_ack(m) or {}).get("acknowledged_utc") if m.get("academic") else None,
                     "gated": ({"page": m["gated"]["page"], "terms": m["gated"].get("terms"),
                                "aup_url": m["gated"].get("aup_url")} if m.get("gated") else None),
                     "size_mb_gated": round(int(w["size_bytes"]) / 1e6) if m.get("gated") else None,
                     "fetchable_any": "url" in w or bool(m.get("gated"))})
    return rows


def status(st) -> Dict[str, Any]:
    from mace.preopt import detect, mlip_install as MI
    py, source = detect.python_source()
    models = _models()
    host_ok, host_note = True, ""
    # an environment that is set up but does not start on this host (a gateway without the Python module)
    broken = [m for m in models if m["status"] == "import_failed" and "shared librar" in (m["reason"] or "")]
    if broken:
        host_ok = False
        host_note = (f"The ML environment ({py}) does not start on this host: {broken[0]['reason']}. Load the "
                     "Python module it was made with (on a dev node) and use it there once (Use an Existing "
                     "Python…, or mace mlip setup --python): that records its library folder, and MACE then "
                     "starts it on this host too.")
    pk = MI.packages()
    envs = sorted({v["env"] for v in pk.values()})
    return {"python": py, "source": source, "default_dir": str(detect.default_env_dir()),
            "default_dirs": {e: str(detect.default_env_dir(e)) for e in envs},
            "env_pythons": {e: detect.python_source(e)[0] for e in envs},
            "loopback": _loopback(st), "host_ok": host_ok, "host_note": host_note, "models": models,
            "packages": [{"package": k, "pin": v["pin"], "license": v["license"], "license_url": v["license_url"],
                          "env": v["env"], "models": v["models"],
                          "min_python": ".".join(map(str, v["min_python"]))} for k, v in pk.items()],
            "default_backends": list(MI.DEFAULT_BACKENDS), "install_task": getattr(st, "mlip_install_task", None)}


def _body_plan(st, body: Dict[str, Any]):
    from mace.preopt import mlip_install as MI
    backends = body.get("backends") or list(MI.DEFAULT_BACKENDS)
    if not isinstance(backends, list) or not all(isinstance(b, str) for b in backends):
        raise ValueError("backends is a list of package names")
    directory = body.get("dir") or None
    python = body.get("python") or None
    if not _loopback(st):
        if directory or python:
            raise PermissionError("MACE GUI is on the network: it installs only into the default place, from the "
                                  "Python it finds (run mace mlip install in a shell for another folder)")
    if directory is not None and not isinstance(directory, str) or python is not None and not isinstance(python, str):
        raise ValueError("dir and python are paths")
    return MI.make_plan(directory, backends, gpu=bool(body.get("gpu")), python=python)


def _refresh(st) -> None:
    """The new environment, seen at once: Clean's and the Calculators' tiers
    detected again, an idle ML worker stopped (the next job starts one from
    the new interpreter)."""
    def run():
        try:
            from . import calc_routes
            from .server import _detect_clean_methods
            host = getattr(st, "mlip_host", None)
            if host is not None:
                with host.lock:
                    if host.owner is None:
                        host._stop_worker()
            _detect_clean_methods(st, recheck=True)
            calc_routes._start_detect(st, recheck=True)
        except Exception:
            pass
    threading.Thread(target=run, name="mace-mlip-refresh", daemon=True).start()


def _task(handler, kind: str, title: str, fn):
    """``fn(log, canceled, progress)`` as an inline task whose ``log`` is the
    tail of what it said."""
    from . import tasks as gui_tasks
    st = handler.state
    lines: List[str] = []

    def job(ctx):
        def log(s: str) -> None:
            lines.append(str(s))
            del lines[:-LOG_LINES]
            ctx.task.log = "\n".join(lines)
            ctx.progress(ctx.task.progress, str(s)[:120])

        def progress(frac: float, text: str) -> None:
            ctx.progress(frac, text)
        try:
            return fn(log, lambda: ctx.canceled, progress)
        finally:
            ctx.task.log = "\n".join(lines)
    t = st.tasks.submit(kind, title, "", {}, client=handler.headers.get("X-Mace-Client"), inline=job)
    return t, gui_tasks.answer(handler, st.tasks, t, 0.05)


def _install_route(handler, body: Dict[str, Any]):
    from mace.preopt import mlip_install as MI, registry
    from . import tasks as gui_tasks
    st = handler.state
    plan = _body_plan(st, body)
    if plan["problems"]:
        return handler._json({"error": "; ".join(plan["problems"]), "plan": plan}, 400)
    fetch_id = body.get("fetch") or None
    if fetch_id:
        m = registry.get(str(fetch_id), scope=None)
        if m is None:
            return handler._json({"error": registry.unknown(str(fetch_id), scope=None)}, 400)
        if body.get("accept_license") is not True:
            return handler._json({"error": f"{m['label']}: accept its weights license ({m['weights']['license']}) "
                                           "to download them, or untick the download"}, 400)
    with _LOCK:
        cur = getattr(st, "mlip_install_task", None)
        if cur:
            try:
                if st.tasks.get(cur).state in ("queued", "running"):
                    return handler._json({"error": "an ML environment is being installed already", "task": cur}, 409)
            except KeyError:
                pass

        def fn(log, canceled, progress):
            try:
                res = MI.install(plan, log=log, progress=progress, canceled=canceled, fetch_id=fetch_id,
                                 accept_license=True, isatty=False)
            except MI.InstallCanceled:
                from .pool_worker import JobCanceled
                raise JobCanceled("canceled")
            except MI.InstallError as exc:
                raise gui_tasks.TaskFailed(str(exc), hint="The log shows each step; running it again repairs "
                                                          "an environment that was there before.")
            _refresh(st)
            return {"ok": True, "python": res["python"], "fetch": res["fetch"], "dir": plan["dir"]}
        t, answer = _task(handler, "mlip-install", f"Set up the ML environment in {Path(plan['dir']).name}", fn)
        st.mlip_install_task = t.id
    return answer


def _record_route(handler, body: Dict[str, Any]):
    from mace.preopt import mlip_install as MI
    from . import tasks as gui_tasks
    st = handler.state
    py = body.get("python")
    if not isinstance(py, str) or not py.strip():
        return handler._json({"error": "name the interpreter (DIR/bin/python)"}, 400)
    if not _loopback(st):
        return handler._json({"error": "MACE GUI is on the network: record an interpreter in a shell "
                                       "(mace mlip setup --python PATH)"}, 403)

    def fn(log, canceled, progress):
        log(f"Probing {py} (up to a minute on a cold start)…")
        r = MI.record_python(py.strip())
        for name, ver, stt in r["rows"]:
            log(f"  {name:11s} {ver:12s} {stt}")
        if not r["ok"]:
            raise gui_tasks.TaskFailed(f"Not recorded: {r['reason']}")
        log(f"Recorded in {r['path']}.")
        if r["override"]:
            log(f"Note: {r['override']} names another interpreter and takes precedence over this record.")
        _refresh(st)
        return {"ok": True, "python": r["python"], "override": r["override"]}
    return _task(handler, "mlip-record", "Use an existing Python for ML potentials", fn)[1]


def _fetch_route(handler, body: Dict[str, Any]):
    from mace.preopt import mlip_install as MI, registry
    from . import tasks as gui_tasks
    st = handler.state
    m = registry.get(str(body.get("id") or ""), scope=None)
    if m is None:
        return handler._json({"error": registry.unknown(str(body.get("id") or ""), scope=None)}, 400)
    if m.get("academic") and body.get("accept_academic") is not True:
        return handler._json({"error": f"{m['label']} is for academic, non-commercial use only: acknowledge that "
                                       "under its license text first"}, 400)
    if not m.get("academic") and body.get("accept_license") is not True:
        return handler._json({"error": f"{m['label']}: accept its weights license ({m['weights']['license']}) first"}, 400)

    def fn(log, canceled, progress):
        r = MI.fetch_weights(m["id"], log=log, yes=True, isatty=False, accept_academic=bool(m.get("academic")),
                             academic_how="gui" if m.get("academic") else None, consent_how="gui")
        if r["rc"] != 0:
            raise gui_tasks.TaskFailed(f"{m['label']}: the weights were not downloaded (the log says why)")
        _refresh(st)
        return {"ok": True, "status": r["status"]}
    return _task(handler, "mlip-fetch", f"Download the weights of {m['label']}", fn)[1]


def handle(handler, method: str, route: str, payload):
    st = handler.state
    body = payload if isinstance(payload, dict) else {}
    try:
        if method == "GET" and route == "mlip/status":
            return handler._json(status(st))
        if method == "POST" and route == "mlip/plan":
            return handler._json(_body_plan(st, body))
        if method == "POST" and route == "mlip/install":
            return _install_route(handler, body)
        if method == "POST" and route == "mlip/record":
            return _record_route(handler, body)
        if method == "POST" and route == "mlip/fetch":
            return _fetch_route(handler, body)
    except PermissionError as exc:
        return handler._json({"error": str(exc)}, 403)
    except ValueError as exc:                 # InstallError is a RuntimeError: a bad package list says so here
        return handler._json({"error": str(exc)}, 400)
    except RuntimeError as exc:
        return handler._json({"error": str(exc)}, 400)
    if method == "GET" and route.startswith("mlip/"):
        return handler._json({"error": f"no route {route} (POST?)"}, 405 if route in POST_ROUTES else 404)
    return handler._json({"error": f"no route {route}"}, 405 if route == "mlip/status" else 404)


POST_ROUTES = ("mlip/plan", "mlip/install", "mlip/record", "mlip/fetch")
