"""Analysis › Structure Analysis (1.1.6): the mace.analysis tools for the
open document, run as background tasks (one CPU slot each).

GET  analysis/info           probes, X-ray sources, tools, Zeo++ and RCSR status
POST analysis/run            {tool, source, params, pattern_path | pattern_text, pattern_name}
                             -> the tool's result (202 + task when it takes longer)
POST analysis/rcsr/import    {path}: a newer RCSR 3dall.txt or a .cgd in the served folder (used before the bundled list)
POST analysis/remesh         {source, params, mesh, doc, key}: a porosity result's surfaces at another
                             mesh quality (mesh_quality.pores), without the analysis when this server made it

``source`` is what Export Structure sends ({path, frame} | {session, rev} |
{held}). Measured patterns come from a file in the served folder or as
text the page read from this computer (at most 20 MB).
"""
from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any, Dict

import numpy as np

GET_ROUTES = {"analysis/info"}
POST_ROUTES = {"analysis/run", "analysis/rcsr/import", "analysis/remesh"}
MAX_PATTERN = 20 * 1024 * 1024
TOOLS = ("porosity", "pxrd", "peaks", "index", "refine", "topology", "polyhedra", "bonds", "morphology",
         "planes", "cleavage")


def _clean(o):
    """JSON-safe: numpy to lists, NaN/inf to None."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def info(st) -> Dict[str, Any]:
    from mace.analysis import porosity, pxrd, rcsr, zeopp
    try:
        rs = rcsr.status()
    except Exception as exc:               # pragma: no cover - a broken import file is reported
        rs = {"error": str(exc)}
    return {"tools": list(TOOLS),
            "probes": {k: {"radius": v["kinetic_diameter"] / 2, "label": v["label"]} for k, v in porosity.PROBES.items()},
            "sources": {k: {"wavelength": v[0], "ka2": v[1]} for k, v in pxrd.SOURCES.items()},
            "zeopp": zeopp.find(), "zeopp_hint": zeopp.INSTALL_HINT, "rcsr": rs}


def _num(p, k, default, lo=None, hi=None, cast=float):
    v = p.get(k, default)
    if v is None or v == "":
        return default
    try:
        v = cast(v)
    except (TypeError, ValueError):
        raise ValueError(f"{k}: not a number") from None
    if lo is not None and v < lo or hi is not None and v > hi:
        raise ValueError(f"{k} must be between {lo} and {hi}")
    return v


def _pattern(handler, body, path=None):
    from mace.analysis import pxrd
    if path is not None:
        return pxrd.read_pattern(path)
    text = body.get("pattern_text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("choose a measured pattern (a file in the folder, or one from this computer)")
    if len(text) > MAX_PATTERN:
        raise ValueError("the pattern is over 20 MB")
    return pxrd.read_pattern_text(text, str(body.get("pattern_name") or "pattern"))


def _structure(handler, body, need=True):
    from mace.analysis.structure import from_atoms
    from . import server
    if not body.get("source"):
        if need:
            raise ValueError("this tool needs a structure: open one first")
        return None
    src = server._source(handler.state, body.get("source"))
    s = from_atoms(src.atoms, src.name)
    return s


# 1.1.6: the last porosity analyses' surface makers (porosity.analyze keep_remesh), so a new mesh
# quality draws the surfaces again without the analysis; keyed by the structure and the numbers'
# parameters (never the mesh's), the oldest let go of first
_REMESH: "OrderedDict[str, Any]" = OrderedDict()
_REMESH_MAX = 3


def _remesh_key(st, params) -> str:
    import hashlib
    import numpy as np
    p = params or {}
    h = hashlib.sha1()
    for a in (st.cell, st.frac, st.numbers):
        h.update(np.ascontiguousarray(np.asarray(a, float)).tobytes())
    h.update(repr((p.get("probe") or "N2", p.get("probe_radius"), p.get("radii") or "ccdc", p.get("spacing"),
                   list(getattr(st, "pbc", ())))).encode())
    return h.hexdigest()


def _porosity(st, p, prog, chk, keep=False):
    from mace.analysis import porosity
    probe = p.get("probe") or "N2"
    if p.get("probe_radius") not in (None, ""):
        probe = _num(p, "probe_radius", 1.82, 0.0, 10.0)
    out = porosity.analyze(st, probe=probe, radii=p.get("radii") or "ccdc",
                           spacing=_num(p, "spacing", porosity.DEFAULT_SPACING, 0.05, 1.0),
                           render=bool(p.get("render", True)), progress=prog, check=chk,
                           mesh=p.get("mesh") if isinstance(p.get("mesh"), dict) else None, keep_remesh=keep)
    remesh = out.pop("_remesh", None)
    if remesh is not None:
        _REMESH[_remesh_key(st, p)] = remesh
        _REMESH.move_to_end(_remesh_key(st, p))
        while len(_REMESH) > _REMESH_MAX:
            _REMESH.popitem(last=False)
    return out


def remesh(st, params: Dict[str, Any], mesh: Dict[str, Any], ctx=None) -> Dict[str, Any]:
    """A porosity result's surfaces at another quality (``mesh``: mesh_quality.pores' spec): the kept
    maker when this server made the analysis, else the analysis again (its numbers are the same)."""
    prog = (lambda f, m="": ctx.progress(f, m)) if ctx else None
    fn = _REMESH.get(_remesh_key(st, params))
    if fn is None:
        out = _porosity(st, {**(params or {}), "render": True, "mesh": mesh}, prog, (lambda: ctx.check()) if ctx else None, keep=True)
        rd = out.get("render") or {}
    else:
        rd = fn(mesh, prog)
    return {"meshes": rd.get("meshes"), "shape": rd.get("shape"), "origin": rd.get("origin"), "voxels": rd.get("voxels"),
            "has_inaccessible": rd.get("has_inaccessible"), "has_accessible": rd.get("has_accessible"), "level": rd.get("level")}


def run_tool(tool: str, st, params: Dict[str, Any], pattern=None, ctx=None) -> Dict[str, Any]:
    """The work of one analysis (no server state: also used by tests)."""
    from mace.analysis import geometry, indexing, morphology, planes, polyhedra, porosity, pxrd, refine, topology, zeopp
    p = params or {}
    prog = (lambda f, m="": ctx.progress(f, m)) if ctx else None
    chk = (lambda: ctx.check()) if ctx else None
    if tool == "porosity":
        probe = p.get("probe") or "N2"
        if p.get("probe_radius") not in (None, ""):
            probe = _num(p, "probe_radius", 1.82, 0.0, 10.0)
        out = _porosity(st, p, prog, chk, keep=ctx is not None and bool(p.get("render", True)))
        rd = out.get("render")
        if rd and rd.get("meshes"):
            # the 3D view draws the meshes; the two fields (mace analyze porosity --cube) stay here
            rd.pop("accessible", None)
            rd.pop("inaccessible", None)
        if p.get("zeopp") and zeopp.find():
            try:
                out["zeopp"] = zeopp.analyze(st, probe=probe, radii=p.get("radii") or "ccdc")
            except Exception as exc:
                out["zeopp_error"] = str(exc)
        return out
    if tool == "pxrd":
        rng = (_num(p, "min", 5.0, 0.5, 170), _num(p, "max", 90.0, 1.0, 179))
        out = pxrd.simulate(st, p.get("source") or "CuKa", rng, _num(p, "step", 0.02, 0.001, 1.0),
                            _num(p, "b_iso", 0.0, -2, 20), _num(p, "U", 0.0), _num(p, "V", 0.0),
                            _num(p, "W", 0.01, 0.0, 5.0), _num(p, "eta", 0.5, 0.0, 1.0),
                            _num(p, "size_nm", None, 0.5, 1e5) if p.get("size_nm") not in (None, "") else None)
        if pattern is not None:
            out["measured"] = {"two_theta": pattern["two_theta"], "intensity": pattern["intensity"],
                               "name": pattern.get("name")}
            out["similarity"] = pxrd.compare_patterns(out["two_theta"], out["intensity"],
                                                      pattern["two_theta"], pattern["intensity"])
        return out
    if tool == "peaks":
        if pattern is None:
            raise ValueError("choose a measured pattern")
        lam = pxrd.source(p.get("source") or "CuKa")[0]
        r = pxrd.find_peaks(pattern["two_theta"], pattern["intensity"], prominence=_num(p, "prominence", 0.02, 0.001, 0.9))
        peaks = [{**x, "d": float(pxrd.d_spacing(x["two_theta"], lam))} for x in r["peaks"]]
        return {"peaks": sorted(peaks, key=lambda x: x["two_theta"]), "two_theta": pattern["two_theta"],
                "intensity": pattern["intensity"], "background": r["background"], "name": pattern.get("name")}
    if tool == "index":
        if pattern is None:
            raise ValueError("choose a measured pattern")
        pk = pxrd.find_peaks(pattern["two_theta"], pattern["intensity"], prominence=_num(p, "prominence", 0.02, 0.001, 0.9))
        tt = [x["two_theta"] for x in pxrd.strip_ka2(pk["peaks"], p.get("source") or "CuKa")]
        if p.get("mode") == "structure" and st is not None:
            return {"mode": "structure", **indexing.index_with_cell(tt, None, p.get("source") or "CuKa", structure=st,
                                                                     tol_deg=_num(p, "tol", 0.05, 0.001, 1.0))}
        return {"mode": "ab_initio", **indexing.index_ab_initio(tt, p.get("source") or "CuKa",
                                                                 tol_deg=_num(p, "tol", 0.03, 0.001, 1.0))}
    if tool == "refine":
        if pattern is None:
            raise ValueError("choose a measured pattern")
        method = p.get("method") or "pawley"
        rng = None
        if p.get("min") not in (None, "") and p.get("max") not in (None, ""):
            rng = (_num(p, "min", 5.0), _num(p, "max", 90.0))
        po = p.get("po")
        if isinstance(po, str) and po.strip():
            po = [int(x) for x in po.replace(",", " ").split()]
            if len(po) != 3:
                raise ValueError("preferred orientation: three integers h k l")
        else:
            po = None
        return refine.fit(pattern, method, structure=st, wavelength=p.get("source") or "CuKa", two_theta_range=rng,
                          n_background=int(_num(p, "background", 6, 1, 20)), refine_positions=bool(p.get("positions")),
                          preferred_orientation=po, progress=prog, check=chk)
    if tool == "topology":
        return topology.analyze(st, p.get("mode") or "auto", _num(p, "tolerance", 1.15, 0.8, 1.6))
    if tool == "polyhedra":
        return polyhedra.analyze(st, _num(p, "tolerance", 1.15, 0.8, 1.6))
    if tool == "bonds":
        return geometry.analyze(st, _num(p, "tolerance", 1.15, 0.8, 1.6), int(_num(p, "bins", 40, 5, 400)))
    if tool == "morphology":
        en = morphology.parse_energies(p["energies"]) if p.get("energies") else None
        return morphology.wulff(st, en, int(_num(p, "max_index", 2, 1, 4)))
    if tool == "planes":
        hkl = [int(x) for x in str(p.get("hkl") or "1 0 0").replace(",", " ").split()]
        if len(hkl) != 3 or not any(hkl):
            raise ValueError("hkl: three integers, not all zero")
        off = _num(p, "offset", 0.0, -100, 100)
        bb = planes.broken_bonds(st, hkl, tolerance=_num(p, "tolerance", 1.15, 0.8, 1.6))
        on = planes.atoms_on_plane(st, hkl, off, _num(p, "tol", 0.3, 0.0, 5.0))
        return {"hkl": hkl, "d": planes.d_spacing(st.cell, hkl), "offset": off,
                "planes": planes.planes_in_box(st.cell, hkl, off), "atoms_on_plane": on,
                "broken": bb}
    if tool == "cleavage":
        return planes.cleavage(st, int(_num(p, "max_index", 2, 1, 3)), _num(p, "tolerance", 1.15, 0.8, 1.6))
    raise ValueError(f"unknown tool {tool!r}")


def _summary(tool: str, result, label: str, params):
    """The result's row of `mace analyze <tool> FILE... --csv` (the Analysis tab's batch summary), or None."""
    try:
        from mace.analysis.cli import summary_row
        return summary_row(tool, result, label, params)
    except Exception:                    # a summary is a convenience: the result stands without it
        return None


NEEDS_STRUCTURE = {"porosity", "pxrd", "topology", "polyhedra", "bonds", "morphology", "planes", "cleavage", "refine"}
NEEDS_PATTERN = {"peaks", "index", "refine"}


def handle(handler, method: str, route: str, payload):
    from . import server, tasks as gui_tasks
    from .submit_routes import _network
    st = handler.state
    if method == "GET" and route == "analysis/info":
        return handler._json(_clean(info(st)))
    if method == "POST" and route == "analysis/rcsr/import":
        if _network(st):
            return handler._json({"error": "Importing reference nets is turned off while MACE GUI is open to the network"}, 409)
        from mace.analysis import rcsr
        p = st.resolve(str(payload.get("path") or ""))
        return handler._json(_clean(rcsr.import_file(p)))
    if method == "POST" and route == "analysis/remesh":
        # 1.1.6: a porosity result's surfaces at another mesh quality (the result's own request + mesh)
        body = payload or {}
        stc = _structure(handler, body, need=True)
        params = body.get("params") or {}
        mesh = body.get("mesh") if isinstance(body.get("mesh"), dict) else {}
        from mace.analysis import mesh_quality
        mesh_quality.pores(mesh)                    # a wrong knob is the request's error (400)
        label = mesh_quality.preset(mesh.get("quality")).capitalize()

        def rjob(ctx):
            return _clean({"tool": "porosity", "render": remesh(stc, params, mesh, ctx)})

        t = server.cli_task(st, "analysis", f"Pore surfaces ({label}): {stc.name}", rjob, doc=body.get("doc"),
                            client=handler.headers.get("X-Mace-Client"),
                            supersede=("remesh", body.get("doc"), body.get("key"), handler.headers.get("X-Mace-Client")))
        return gui_tasks.answer(handler, st.tasks, t, gui_tasks.clamp_wait(body.get("wait", 2.0)))
    if method == "POST" and route == "analysis/run":
        body = payload or {}
        tool = str(body.get("tool") or "")
        if tool not in TOOLS:
            raise ValueError(f"tool must be one of {', '.join(TOOLS)}")
        method_ = (body.get("params") or {}).get("method")
        need_st = tool in NEEDS_STRUCTURE and not (tool == "refine" and method_ == "pawley" and not body.get("source"))
        need_pat = tool in NEEDS_PATTERN or (tool == "pxrd" and (body.get("pattern_path") or body.get("pattern_text")))
        # a pattern path outside the folder is refused before the structure loads (which may need ase)
        pat_file = st.resolve(str(body["pattern_path"])) if need_pat and body.get("pattern_path") else None
        stc = _structure(handler, body, need=need_st) if (need_st or body.get("source")) else None
        pattern = _pattern(handler, body, pat_file) if need_pat else None
        params = body.get("params") or {}
        if tool == "refine" and stc is None:
            raise ValueError("Pawley in the GUI starts from the open structure's cell: open the structure first")
        title = {"porosity": "Porosity", "pxrd": "Powder pattern", "peaks": "Peak search", "index": "Indexing",
                 "refine": "Pattern fit", "topology": "Topology", "polyhedra": "Polyhedra", "bonds": "Bond distributions",
                 "morphology": "Morphology", "planes": "Miller plane", "cleavage": "Cleavage"}[tool]

        src = body.get("source") if isinstance(body.get("source"), dict) else {}
        label = str(src.get("path") or (stc.name if stc is not None else "") or "")

        def job(ctx):
            res = run_tool(tool, stc, params, pattern, ctx)
            return _clean({"tool": tool, "result": res, "summary": _summary(tool, res, label, params),
                           "structure": stc.name if stc is not None else None,
                           "cell": stc.cell.tolist() if stc is not None else None,
                           "origin": (stc.meta or {}).get("origin") if stc is not None else None})

        if stc is not None and stc.name:
            title = f"{title}: {stc.name}"
        client = handler.headers.get("X-Mace-Client")
        # a newer run of the same tool on the same document (in the same window) replaces this one;
        # document ids are per window, so another window's runs are never touched
        t = server.cli_task(st, "analysis", title, job, doc=body.get("doc"), client=client,
                            supersede=("analysis", tool, client, body.get("doc")))
        return gui_tasks.answer(handler, st.tasks, t, gui_tasks.clamp_wait(body.get("wait", 2.0)))
    return handler._json({"error": "not found"}, 404)
