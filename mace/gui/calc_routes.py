"""The Calculators panel's routes (``mace calc`` in the GUI).

GET  calc/backends            what each backend offers and whether it is ready here
                              ({checking: true} until the first detection finished)
POST calc/backends/recheck    detect again (skips the xtb, DFTB+ and ML-environment caches)
POST calc/run                 {source, backend, method, task, params..., save, save_dir, open_result}
                              -> a pool task; its answer is the result (energies, charges,
                              frequencies, MD summary), the files written when ``save``, and
                              for opt/md the final structure as a new document

The calculation runs in the process pool (``calc.run``, ``run_job`` below,
through ``mace.calculators.tasks.run``: the CLI's code). An ML potential is
loaded once in the server's one ML worker (``mlip_host``, the Clean tier's)
on the device asked for: GPU 0 by default when there is one. Files are
written by the server thread after the job, each through ``check_write``,
into the folder of the document's file (or ``save_dir``); nothing is written
for a document that has no folder unless ``save_dir`` names one.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Dict

MAX_TIME = 24 * 3600.0
DEFAULT_TIME = 3600.0
_LOCK = threading.Lock()


# ---------------------------------------------------------------- detection

def _detect(st, recheck: bool = False) -> None:
    from mace.calculators import device, registry
    rows = []
    for be in registry.all_backends():
        try:
            rows.append(be.describe(recheck=recheck))
        except Exception as exc:                       # never take the server down
            rows.append({"name": be.name, "label": be.label, "error": str(exc), "methods": [], "available": False})
    try:
        gpus, dev = device.gpus(refresh=recheck), device.default_text()
    except Exception:
        gpus, dev = [], "auto"
    with _LOCK:
        st.calc_backends = {"checking": False, "backends": rows, "gpus": gpus, "device_default": dev,
                            "checked": time.time()}


def _start_detect(st, recheck: bool = False) -> None:
    with _LOCK:
        cur = getattr(st, "calc_backends", None)
        if cur is not None and cur.get("checking") and not recheck:
            return
        st.calc_backends = {"checking": True, "backends": (cur or {}).get("backends") or _skeleton(),
                            "gpus": (cur or {}).get("gpus") or [], "device_default": (cur or {}).get("device_default")}
    threading.Thread(target=_detect, args=(st, recheck), name="mace-calc-detect", daemon=True).start()


def _skeleton():
    from mace.calculators import registry
    out = []
    for name in registry.names():
        be = registry.get(name)
        out.append({"name": be.name, "label": be.label, "family": be.family, "license": be.license,
                    "tasks": sorted(be.capabilities.tasks), "methods": [
                        {**m.as_dict(), "available": name in ("uff", "eqeq"), "status": "checking"}
                        for m in be.methods()], "available": name in ("uff", "eqeq")})
    return out


def backends(st) -> Dict[str, Any]:
    with _LOCK:
        cur = getattr(st, "calc_backends", None)
    if cur is None:
        _start_detect(st)
        with _LOCK:
            cur = st.calc_backends
    return dict(cur)


def _method_status(st, backend: str, method: str) -> Dict[str, Any]:
    cur = backends(st)
    for b in cur.get("backends") or []:
        if b.get("name") == backend:
            for m in b.get("methods") or []:
                if m.get("id") == method:
                    return {**m, "checking": bool(cur.get("checking"))}
    return {"checking": bool(cur.get("checking"))}


# ---------------------------------------------------------------- the pool job

def run_job(ctx=None, structure=None, spec=None, mlip_lease=None, **_):
    """In a pool worker: one calculation through mace.calculators.tasks.run."""
    from mace.calculators import base, structio, tasks
    atoms = structio.atoms_from_payload(structure)
    if structure.get("source_path"):          # a model that takes a charge and spin reads them from the file
        atoms.info["source_path"] = structure["source_path"]
    s = base.CalcSpec(**spec)
    if mlip_lease is not None:
        s.options = dict(s.options or {}, _lease=mlip_lease)
    try:
        r = tasks.run(atoms, s, progress=(lambda f, m: ctx.progress(f, m)) if ctx else None,
                      canceled=(lambda: ctx.canceled) if ctx else None)
    except base.CalcError as exc:
        return {"error": str(exc), "kind": exc.kind, "hint": exc.hint}
    if ctx is not None:
        ctx.check()
    r.provenance.get("settings", {}).get("options", {}).pop("_lease", None)
    return {"result": r}


# ---------------------------------------------------------------- routes

_NUM = {"fmax": float, "steps": int, "displacement": float, "temperature": float, "timestep": float,
        "md_steps": int, "friction": float, "pressure": float, "interval": int, "time_limit": float, "threads": int}
_BOOL = ("relax_cell", "keep_symmetry")
_STR = {"optimizer": ("fire", "bfgs", "lbfgs"), "ensemble": ("nve", "nvt", "npt")}
_OPTIONS = {"kpts", "kpoint_density", "skf_dir", "engine", "charges", "cutoff", "charge_centers", "eqeq_method",
            "gpu_mem_fraction", "dtype", "task_head"}


def _spec(body: Dict[str, Any]):
    from mace.calculators import base, registry
    name, method = registry.resolve(str(body.get("backend") or ""), body.get("method") or None)
    be = registry.get(name)
    m = be.method(method)
    task = str(body.get("task") or "sp")
    if task not in base.TASKS or task not in be.capabilities.tasks:
        raise ValueError(f"{be.label} does not do {base.TASK_LABELS.get(task, task)}")
    spec = base.CalcSpec(backend=name, method=m.id, task=task)
    spec.device = str(body.get("device") or "auto")
    from mace.preopt import mlip_client
    mlip_client.device_mapping(spec.device)            # ValueError on a bad device
    for k, conv in _NUM.items():
        v = body.get(k)
        if v not in (None, ""):
            v = conv(v)
            if k != "temperature" and k != "pressure" and not v > 0:
                raise ValueError(f"{k} must be positive")
            setattr(spec, k, v)
    for k in _BOOL:
        if k in body:
            setattr(spec, k, bool(body[k]))
    for k, allowed in _STR.items():
        if body.get(k):
            if body[k] not in allowed:
                raise ValueError(f"{k} is one of {', '.join(allowed)}")
            setattr(spec, k, body[k])
    if body.get("seed") not in (None, ""):
        spec.seed = int(body["seed"])
    for k in ("charge", "uhf", "multiplicity"):
        v = body.get(k)
        if v not in (None, ""):
            if isinstance(v, bool) or (isinstance(v, float) and not v.is_integer()):
                raise ValueError(f"{k} is a whole number")
            if k == "multiplicity":
                if int(v) < 1:
                    raise ValueError("multiplicity is 1 or more")
                spec.uhf, spec.uhf_set = int(v) - 1, True
            else:
                setattr(spec, k, int(v))
                setattr(spec, f"{k}_set", True)
    opts = body.get("options") or {}
    if not isinstance(opts, dict):
        raise ValueError("options is an object")
    spec.options = {k: v for k, v in opts.items() if k in _OPTIONS and v not in (None, "")}
    if spec.time_limit is None:
        spec.time_limit = DEFAULT_TIME
    spec.time_limit = min(float(spec.time_limit), MAX_TIME)
    return spec, be, m


def _save_dir(st, body) -> Path:
    rel = body.get("save_dir")
    if rel in (None, ""):
        path = (body.get("source") or {}).get("path")
        if path:
            return st.resolve(str(path)).parent
        raise ValueError("this document has no folder: choose where to save, or turn saving off")
    p = st.root if rel in (".", "/") else st.resolve(str(rel))
    if not p.is_dir():
        raise ValueError(f"{rel} is not a folder")
    return p


def _run_route(handler, body: Dict[str, Any]):
    st = handler.state
    from . import server as S
    from . import tasks as gui_tasks
    from mace.calculators import base, tasks
    try:
        spec, be, m = _spec(body)
    except (ValueError, TypeError, base.CalcError) as exc:
        return handler._json({"error": str(exc)}, 400)
    ms = _method_status(st, spec.backend, spec.method)
    if not ms.get("available") and spec.backend not in ("uff", "eqeq"):
        why = ms.get("reason") or ("still checking what is installed" if ms.get("checking") else "not available")
        if spec.backend == "dftb" and spec.options.get("skf_dir") and ms.get("status") == "needs_skf":
            why = None                                   # a folder given in the panel: checked by the run itself
        if why:
            return handler._json({"error": f"{m.label}: {why}", "hint": ms.get("hint", ""),
                                  "commands": ms.get("commands") or [], "unavailable": True}, 400)
    if spec.options.get("skf_dir"):
        try:
            spec.options["skf_dir"] = str(st.resolve(str(spec.options["skf_dir"])))
        except (PermissionError, FileNotFoundError):
            from .server import is_loopback
            if not is_loopback(getattr(st, "bind_host", "127.0.0.1")):
                # a network-exposed GUI reads no folder outside the one it serves
                return handler._json({"error": "the Slater-Koster folder must be inside the served folder when "
                                               "MACE GUI is on the network (or set MACE_DFTB_SKF where it starts)"}, 400)
            p = Path(str(spec.options["skf_dir"])).expanduser()
            if not p.is_absolute() or not p.is_dir():
                return handler._json({"error": "the Slater-Koster folder is not a folder"}, 400)
    save = bool(body.get("save"))
    out_dir = None
    if save:
        # the save folder is checked before the structure loads (which may need ase)
        try:
            out_dir = _save_dir(st, body)
        except (ValueError, PermissionError, FileNotFoundError) as exc:
            return handler._json({"error": f"Save: {exc}"}, 400)
    src = S._source(st, body.get("source") or {})
    atoms = src.atoms
    if len(atoms) == 0:
        return handler._json({"error": "the structure has no atoms"}, 400)
    if len(atoms) > S.MAX_BUILD_ATOMS:
        return handler._json({"error": f"{len(atoms):,} atoms is more than the GUI runs"}, 400)
    if save:
        try:
            st.check_write(out_dir / f"{_stem(src.name)}_calc.json")
        except (ValueError, PermissionError, FileNotFoundError) as exc:
            return handler._json({"error": f"Save: {exc}"}, 400)
    stem = _stem(src.name)
    structure = {"cell": atoms.cell.array.tolist(), "frac": atoms.get_scaled_positions(wrap=False).tolist(),
                 "cart": atoms.positions.tolist(), "numbers": [int(z) for z in atoms.numbers],
                 "pbc": [bool(p) for p in atoms.pbc]}
    if src.kind == "path":                     # the file as it is on disk: its deck or output states charge and spin
        try:
            structure["source_path"] = str(st.resolve(str((body.get("source") or {}).get("path"))))
        except Exception:
            pass
    weight, acquire = 1, None
    if spec.backend == "mlip":
        threads = S._clean_threads()
        spec.threads = spec.threads or threads
        weight = spec.threads
        frac = spec.options.get("gpu_mem_fraction")
        acquire = (lambda t, _m=spec.method, _k=spec.threads, _d=spec.device, _f=frac:
                   st.mlip_host.acquire(t, _m, _k, device=_d, gpu_mem_fraction=_f, scope=None))
    elif spec.backend in ("xtb", "dftb"):
        spec.threads = spec.threads or S._clean_threads()
        weight = spec.threads
    client = handler.headers.get("X-Mace-Client")
    open_result = body.get("open_result", True) is not False

    def finish(raw):
        if raw.get("error"):
            raise gui_tasks.TaskFailed(raw["error"], hint=raw.get("hint", ""), kind=raw.get("kind", "failed"))
        r = raw["result"]
        out = r.as_dict(full=False)
        out["calc_task"] = out.pop("task")              # "task" is the task record of every answer
        if r.forces is not None and len(r.forces) <= 2000:
            import numpy as np
            out["forces_eV_A"] = np.round(np.asarray(r.forces), 5).tolist()
        if save and out_dir is not None:
            try:
                files = tasks.write_outputs(r, out_dir, stem, spec, force=bool(body.get("overwrite")),
                                            check=lambda p: st.check_write(p))
                out["files"] = {k: st.rel(out_dir / v) for k, v in files.items()}
                out["saved_to"] = st.rel_new(out_dir) or "."
            except FileExistsError as exc:
                out["save_error"] = f"{exc}; nothing was written (tick Overwrite to replace them)"
            except PermissionError as exc:
                out["save_error"] = f"not written: {exc}"
        if open_result and r.task in ("opt", "md") and r.atoms is not None:
            a = r.atoms
            raw_s = {"numbers": [int(z) for z in a.numbers], "cell": a.cell.array.tolist(),
                     "frac": a.get_scaled_positions(wrap=False).tolist(), "pbc": [bool(p) for p in a.pbc],
                     "summary": r.summary()}
            name = f"{stem}_{tasks.tag(spec)}_{r.task}"[:60]
            params = {"backend": spec.backend, "method": spec.method, "task": spec.task,
                      "energy_eV": r.energy, "converged": r.converged}
            try:
                if all(a.pbc):
                    out["open"] = S._finish_build(st, raw_s, src, f"calc_{r.task}", params,
                                                  {"name": name, "label": f"{m.label} {r.task} of {src.label}"[:120]},
                                                  client)
            except Exception as exc:
                out["open_error"] = str(exc)
        return out

    t = st.tasks.submit("calc", f"{base.TASK_LABELS[spec.task]} ({m.label}) of {src.label}"[:120], "calc.run",
                        {"structure": structure, "spec": spec.as_dict()}, doc=getattr(src, "name", None),
                        client=client, finish=finish, weight=weight, acquire=acquire)
    return gui_tasks.answer(handler, st.tasks, t, 0.05)


def _stem(name: str) -> str:
    import re
    s = re.sub(r"\.[A-Za-z0-9]+$", "", str(name or "structure"))
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("._-") or "structure"
    return s[:40]


def handle(handler, method: str, route: str, payload):
    st = handler.state
    try:
        if method == "GET" and route == "calc/backends":
            return handler._json(backends(st))
        if method == "POST" and route == "calc/backends/recheck":
            _start_detect(st, recheck=True)
            return handler._json(backends(st))
        if method == "POST" and route == "calc/run":
            return _run_route(handler, payload or {})
    except PermissionError as exc:
        return handler._json({"error": str(exc)}, 403)
    except (ValueError, FileNotFoundError) as exc:
        return handler._json({"error": str(exc)}, 400)
    return handler._json({"error": f"no route {route}"}, 404)
