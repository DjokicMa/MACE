"""CRYSTAL23 ORBITALS in the GUI (spec 4.4): the orbital picker's index and
the orbital grids the Volume document draws.

The reader (``mace.orbitals``) does all the science: it finds the run's
Molden files and .out, checks each k point (orthonormality on load), and
evaluates an orbital on a grid. This module only turns its answers into what
the page needs:

- ``index_job``: ``api.describe`` plus the picker's extras (energy reference
  per spin, occupation wording, default supercell);
- ``grid_job``: one orbital (or a degenerate set's summed |psi|^2) on a grid,
  returned in the Volume document's ``{meta, grid}`` shape, so the existing
  view draws it (``signed`` for Re/Im, not for |psi|^2), with the reader's
  ``default_isovalue`` (about 90 % of |psi|^2 inside) as the level;
- ``save_job``: the cube of one request in ``<job>_orbitals/`` through
  ``api.make_cube`` (an existing cube there is reused; one MACE wrote for another
  MO set or an older basis convention is recomputed).

Every job runs in a pool worker. A worker keeps the last few runs it opened
(their k points stay loaded), keyed by the files' sizes and times. The
.molden files are only read.
"""
from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from mace.constants import BOHR_TO_ANGSTROM as BOHR, HARTREE_TO_EV as HARTREE_EV  # noqa: E402
# the grid's point budget per quality: the reader widens the spacing until the grid fits,
# which also caps a supercell (preview/normal/fine are the reader's spacings, 0.40/0.25/0.15 bohr)
QUALITY_POINTS = {"preview": 300_000, "normal": 1_000_000, "fine": 2_000_000}
QUALITIES = tuple(QUALITY_POINTS)
MAX_CELLS = 4                      # per periodic axis
MAX_MOS = 64                       # orbitals in one request (a summed set)
PARTS = ("re", "im", "abs2")
PART_LABEL = {"re": "Re ψ", "im": "Im ψ", "abs2": "|ψ|²"}
SPIN_LABEL = {"ALPHA+BETA": "closed shell", "ALPHA": "α", "BETA": "β"}

_RUNS: "OrderedDict[Tuple, Any]" = OrderedDict()
_RUNS_MAX = 4


# ---------------------------------------------------------------- the run (worker side)
def _stamp(files: Sequence[Path]) -> Tuple:
    out = []
    for f in files:
        try:
            s = os.stat(f)
            out.append((str(f), int(s.st_mtime_ns), int(s.st_size)))
        except OSError:
            out.append((str(f), 0, 0))
    return tuple(out)


def _inside(path: Path, root: Optional[str]) -> bool:
    if not root:
        return True
    r = Path(root)
    p = Path(path).resolve()
    return p == r or r in p.parents


def open_run(path: str, root: Optional[str] = None):
    """The OrbitalRun of a .molden file, reused while its files are unchanged.

    ``root``: the served folder; a run whose files or .out lead out of it (a
    symlink) is refused, as every other route refuses such a path."""
    from mace.orbitals import api
    run = api.open_run(path)
    files = list(run.files) + ([Path(run.out.path)] if run.out is not None and getattr(run.out, "path", None) else [])
    for f in files:
        if not _inside(f, root):
            raise PermissionError(f"{Path(f).name} leads outside the served directory")
    key = (str(Path(path).resolve()),) + _stamp(files)
    hit = _RUNS.get(key)
    if hit is not None:
        _RUNS.move_to_end(key)
        return hit
    _RUNS[key] = run
    while len(_RUNS) > _RUNS_MAX:
        _RUNS.popitem(last=False)
    return run


def default_cells(dim: int, periodic: Sequence[int]) -> List[int]:
    """A supercell that keeps a Bloch orbital's lobes whole at the cell edge:
    2 cells along each periodic axis (4 along a polymer's short repeat); a molecule 1x1x1."""
    if dim == 0:
        return [1, 1, 1]
    n = 4 if dim == 1 else 2
    return [n if i in periodic else 1 for i in range(3)]


def _occ_text(occ: float, max_occ: float, closed: bool, metallic: bool = False) -> str:
    if metallic and occ > 0.02 * max_occ:
        return f"occupation {occ:.3f}"           # a metal's smeared occupations: the number itself
    if occ >= 0.98 * max_occ:
        return "2 e (closed shell)" if closed else "1 e"
    if occ <= 0.02 * max_occ:
        return "empty"
    return f"partly filled ({occ:.3f})"


def _occ_short(occ: float, max_occ: float, closed: bool, metallic: bool = False) -> str:
    """The occupation as the level list's narrow column shows it (the words are its tooltip)."""
    if metallic and occ > 0.02 * max_occ:
        return f"{occ:.3f} e"
    if occ >= 0.98 * max_occ:
        return "2 e" if closed else "1 e"
    if occ <= 0.02 * max_occ:
        return "empty"
    return f"{occ:.3f} e"


def _spin_extras(info: Dict[str, Any], kd) -> None:
    """Per spin: the energy reference (HOMO, or E_F for a metal), eV relative to it, and the occupation's words."""
    efermi = info.get("efermi")
    closed = kd.spins == ["ALPHA+BETA"]
    for s in info["k"]["spins"]:
        lv = s["levels"]
        if s["metallic"] and efermi is not None:
            ref, ref_kind = float(efermi) * HARTREE_EV, "E_F"
        elif s["homo"] is not None:
            ref, ref_kind = lv[s["homo"] - 1]["e_ev"], "HOMO"
        else:
            ref, ref_kind = 0.0, "0"
        s["ref_ev"], s["ref_kind"] = ref, ref_kind
        s["label"] = SPIN_LABEL.get(s["spin"], s["spin"])
        sizes: Dict[int, int] = {}
        for l in lv:
            sizes[l["group"]] = sizes.get(l["group"], 0) + 1
        for l in lv:
            l["rel_ev"] = l["e_ev"] - ref
            l["occupied"] = l["occ"] > 0.5 * kd.max_occ
            l["occ_text"] = _occ_text(l["occ"], kd.max_occ, closed, s["metallic"])
            l["occ_short"] = _occ_short(l["occ"], kd.max_occ, closed, s["metallic"])
            l["degenerate"] = sizes[l["group"]]
        if s["metallic"] and ref_kind == "E_F":
            below = [l for l in lv if l["e_ev"] <= ref]
            s["below_ef"] = below[-1]["n"] if below else None
        s["default_mo"] = s["homo"] if s["homo"] is not None else (lv[0]["n"] if lv else None)


def index_job(ctx=None, path: str = "", k: Optional[int] = None, root: Optional[str] = None, **_):
    """``api.describe`` of the run, for the k point asked (default: the first usable one)."""
    from mace.orbitals import api
    if ctx is not None:
        ctx.progress(0.1, "Reading the orbitals")
    run = open_run(path, root)
    asked = None
    if k is not None:
        e = next((x for x in run.kpoints() if x.index == k), None)
        if e is None or not e.usable:
            # the file opened is of a k point the picker cannot draw: the first usable one, and why
            asked = {"k": k, "reason": e.reason if e is not None else f"no k point {k} in this run"}
            k = None
    info = api.describe(run, k)
    info["asked"] = asked
    if ctx is not None:
        ctx.check()
    info["max_occ"] = None
    sel = info.get("selected")
    if sel is not None and "k" in info:
        kd = run.load(sel)
        info["max_occ"] = kd.max_occ
        info["closed_shell"] = kd.spins == ["ALPHA+BETA"]
        _spin_extras(info, kd)
        per = list(kd.basis.periodic)
        info["periodic_axes"] = per
        info["default_cells"] = default_cells(info["dimensionality"], per)
    info["max_cells"] = MAX_CELLS
    info["qualities"] = list(QUALITIES)
    info["path_name"] = Path(path).name
    return info


# ---------------------------------------------------------------- the grid
def check_request(k, spin, mos, part, cells, quality, degenerate_sum) -> Dict[str, Any]:
    """The request's values, checked (a wrong value is the request's 400, not a failed task)."""
    try:
        k = int(k)
    except (TypeError, ValueError):
        raise ValueError("k: the k point's index (a whole number)")
    if part not in PARTS:
        raise ValueError("part is re, im or abs2")
    if quality not in QUALITY_POINTS:
        raise ValueError(f"quality is {', '.join(QUALITIES)}")
    if isinstance(mos, (str, int)):
        mos = [mos]
    out_mos: List[Any] = []
    for m in mos or []:
        s = str(m).strip()
        if not s:
            continue
        out_mos.append(int(s) if s.isdigit() else s.upper())
    if not out_mos or len(out_mos) > MAX_MOS:
        raise ValueError(f"mo: 1 to {MAX_MOS} orbitals (numbers, or HOMO / LUMO±n)")
    try:
        cells = [int(c) for c in (cells or [1, 1, 1])]
    except (TypeError, ValueError):
        raise ValueError("cells: three whole numbers, such as 2x2x1")
    if len(cells) != 3 or not all(1 <= c <= MAX_CELLS for c in cells):
        raise ValueError(f"cells: three whole numbers from 1 to {MAX_CELLS}")
    sp = str(spin or "").upper() or None
    return {"k": k, "spin": sp, "mos": out_mos, "part": part, "cells": cells, "quality": quality,
            "degenerate_sum": bool(degenerate_sum)}


def _symbol(z: int) -> str:
    from .volume import _symbol as sym
    return sym(z)


def _spin_of(kd, spin: Optional[str]) -> str:
    if spin in kd.spins:
        return spin
    a = {"A": "ALPHA", "B": "BETA", "ALPHA": "ALPHA", "BETA": "BETA"}.get(str(spin or "").upper())
    if a in kd.spins:
        return a
    if len(kd.spins) == 1 or spin is None:
        return kd.spins[0]
    raise ValueError(f"spin {spin!r} is not in this file ({', '.join(kd.spins)})")


def grid_payload(run, req: Dict[str, Any], name: str = "") -> Tuple[Dict[str, Any], Any]:
    """(meta, float32 grid) in the Volume document's shape, from ``run.grid``."""
    import numpy as np
    from mace.orbitals import api
    from mace.orbitals.dataset import default_isovalue
    kd = run.load(req["k"])
    spin = _spin_of(kd, req["spin"])
    cells = tuple(req["cells"]) if kd.basis.periodic else (1, 1, 1)
    # along a non-periodic axis there is one "cell" (the vacuum box)
    cells = tuple(c if i in kd.basis.periodic else 1 for i, c in enumerate(cells))
    grid, data, meta = run.grid(req["k"], spin, req["mos"], part=req["part"], cells=cells, quality=req["quality"],
                                degenerate_sum=req["degenerate_sum"], max_points=QUALITY_POINTS[req["quality"]])
    signed = bool(meta["signed"])
    level = default_isovalue(data, signed)
    g = np.ascontiguousarray(data, dtype="<f4")
    Z, pos = run.atoms_bohr(req["k"])
    pos, Z = api._atoms_in_box(run, kd, grid, Z, pos)
    per = list(kd.basis.periodic)
    shape = np.array(grid.shape)
    vecs = np.asarray(grid.vecs, float)
    # the box drawn as the cell: n cells along a periodic axis, the grid's extent along a vacuum one
    edges = np.array([vecs[i] * (shape[i] if i in per else max(1, shape[i] - 1)) for i in range(3)])
    origin = np.asarray(grid.origin, float)
    d = np.asarray(data, float)
    ns = meta["mos"]
    lv = kd.levels(spin)
    homo = kd.homo(spin)
    part = meta["part"]
    if len(ns) > 1:
        what = f"MOs {ns[0]}–{ns[-1]} summed"
    else:
        tag = ""
        if homo is not None and ns[0] == homo:
            tag = " (HOMO)"
        elif homo is not None and ns[0] == homo + 1:
            tag = " (LUMO)"
        what = f"MO {ns[0]}{tag}"
    klabel = "" if run.molecule else f" · k {kd.entry.label}"
    spin_txt = "" if kd.spins == ["ALPHA+BETA"] else f" · {SPIN_LABEL.get(spin, spin)}"
    title = f"{name or run.root}{klabel}{spin_txt} · {what} · {PART_LABEL[part]}"
    out = {
        "name": title, "type": "orbital", "label": "Orbital" if signed else "Orbital density", "signed": signed,
        "quality": req["quality"], "shape": [int(x) for x in g.shape], "full_shape": [int(x) for x in grid.shape],
        "stride": 1, "origin": (origin * BOHR).round(8).tolist(), "voxels": (vecs * BOHR).round(10).tolist(),
        "crop": [[0, int(n)] for n in grid.shape], "cell_origin": (origin * BOHR).round(8).tolist(),
        "full_voxels": (vecs * BOHR).round(10).tolist(), "lattice": (edges * BOHR).round(8).tolist(),
        "periodic": bool(per), "pbc": [i in per for i in range(3)],
        "atoms": {"numbers": [int(z) for z in Z], "symbols": [_symbol(int(z)) for z in Z],
                  "cart": (np.asarray(pos, float).reshape(-1, 3) * BOHR).round(6).tolist()},
        "stats": {"min": float(d.min()), "max": float(d.max()), "mean": float(d.mean()),
                  "absmax": float(np.abs(d).max()), "n": int(d.size)},
        "level": float(level) if level > 0 else float(np.abs(d).max() / 10 or 1e-6),
        "nbytes": int(g.nbytes), "comment": [title[:200], "MACE ORBITALS (GUI preview)"],
        "orbital": {**{k: v for k, v in meta.items() if k != "k"}, "kpoint": meta["k"], "k_index": kd.entry.index,
                    "isovalue": float(level), "cells": list(cells), "spacing_bohr": float(np.linalg.norm(vecs, axis=1).max()),
                    "levels": [{"n": n, "e_ev": lv[n - 1].energy * HARTREE_EV, "occ": lv[n - 1].occupation} for n in ns],
                    "gauge_note": ("A member of a degenerate set: its shape depends on how CRYSTAL mixed the set; "
                                   "sum the set for a gauge-free picture.")
                    if len(ns) == 1 and len(kd.degenerate_set(spin, ns[0])) > 1 else ""},
    }
    return out, g


def grid_job(ctx=None, path: str = "", root: Optional[str] = None, **req):
    if ctx is not None:
        ctx.progress(0.1, "Reading the orbitals")
    run = open_run(path, root)
    if ctx is not None:
        ctx.check()
        ctx.progress(0.3, "Evaluating the orbital on the grid")
    r = check_request(req.get("k"), req.get("spin"), req.get("mos"), req.get("part", "re"), req.get("cells"),
                      req.get("quality", "normal"), req.get("degenerate_sum"))
    meta, g = grid_payload(run, r)
    meta["request"] = {**r, "spin": meta["orbital"]["spin"], "mos": meta["orbital"]["mos"]}
    return {"meta": meta, "grid": g.tobytes()}


def save_job(ctx=None, path: str = "", root: Optional[str] = None, **req):
    """The request's cube in ``<job>_orbitals/`` (api.make_cube); an existing up-to-date one is reused."""
    from mace.orbitals import api
    run = open_run(path, root)
    r = check_request(req.get("k"), req.get("spin"), req.get("mos"), req.get("part", "re"), req.get("cells"),
                      req.get("quality", "normal"), req.get("degenerate_sum"))
    kd = run.load(r["k"])
    spin = _spin_of(kd, r["spin"])
    quality = "fine" if r["quality"] == "fine" else "normal"       # a saved cube is never the preview grid
    folder = api.cache_dir_for(run.files[0].parent, run.root)
    if root and not _inside(folder.parent, root):
        raise PermissionError("the cube folder would be outside the served directory")
    if folder.is_symlink():
        raise PermissionError(f"{folder.name} is a symbolic link; MACE does not write through it")
    cube, meta = api.make_cube(run, r["k"], spin, r["mos"], part=r["part"], cells=r["cells"], quality=quality,
                               degenerate_sum=r["degenerate_sum"], out_dir=folder)
    files = [f for f in folder.glob("*.cube") if f.is_file()]
    n_bytes = sum(f.stat().st_size for f in files)
    warning = api.quota_warning(len(files), n_bytes)
    return {"cube": str(cube), "cached": bool(meta.get("cached")), "bytes": cube.stat().st_size,
            "folder_files": len(files), "folder_bytes": n_bytes, "warning": warning}


# ---------------------------------------------------------------- routes (server side)
def _molden_path(st, rel) -> Path:
    from . import filekind
    p = st.resolve(rel)
    if not p.is_file():
        raise FileNotFoundError(st.rel(p))
    if filekind.kind_by_name(p.name) != "orbitals":
        raise ValueError(f"{p.name} is not a CRYSTAL ORBITALS Molden file (.molden)")
    return p


def _run_stamp(p: Path) -> Tuple:
    """The sizes and times of the run's files (its .molden set and the .out beside them)."""
    from mace.orbitals.outfile import split_name
    root = split_name(p.name)["root"] or p.stem
    sib = [f for f in p.parent.iterdir() if f.is_file() and (f.name.startswith(root) or f.suffix.lower() == ".out")
           and f.suffix.lower() in (".molden", ".out")] if p.parent.is_dir() else []
    return _stamp(sorted(sib))


def _req_of(get) -> Dict[str, Any]:
    cells = str(get("cells", "1x1x1") or "1x1x1").lower().replace(",", "x").split("x")
    mos = [m for m in str(get("mo", "") or "").split(",") if m.strip()]
    return check_request(get("k"), get("spin"), mos, get("part", "re"), cells, get("quality", "normal"),
                         str(get("sum", "0")).lower() in ("1", "true", "yes"))


def get_route(handler, route, query):
    from . import cache as gui_cache
    from . import tasks as gui_tasks
    from .server import _volume_finish
    st = handler.state
    q = lambda k, d=None: handler._q(query, k, d)          # noqa: E731
    p = _molden_path(st, q("path"))
    client = handler.headers.get("X-Mace-Client")
    stamp = _run_stamp(p)
    if route == "orbitals":
        k = q("k")
        if k not in (None, ""):
            try:
                k = int(k)
            except ValueError:
                raise ValueError("k: the k point's index (a whole number)")
        else:
            k = None
        key = gui_cache.file_key(p, ("orbitals-index", k, stamp))
        hit = st.parse_cache.get(key)
        if hit is not None:
            return handler._json({**hit, "path": st.rel(p)})

        def finish(res):
            st.parse_cache.put(key, res, 65536)
            st.ledger.enforce()
            return {**res, "path": st.rel(p)}
        t = st.tasks.submit("orbitals", f"Orbitals of {p.name}", "orbitals.index",
                            {"path": str(p), "k": k, "root": str(st.root)}, client=client, finish=finish, quiet=True)
        return gui_tasks.answer(handler, st.tasks, t, float(q("wait", "0.3")))
    # orbitals/grid
    r = _req_of(q)
    key = gui_cache.file_key(p, ("orbitals-grid", stamp, r["k"], r["spin"], tuple(map(str, r["mos"])), r["part"],
                                 tuple(r["cells"]), r["quality"], r["degenerate_sum"]))
    hit = st.parse_cache.get(key)
    if hit is not None and st.parse_cache.get(("volgrid", hit["token"])) is not None:
        return handler._json({**hit["meta"], "path": st.rel(p)})
    mo_txt = ",".join(map(str, r["mos"]))
    t = st.tasks.submit("orbitals", f"Orbital {mo_txt} of {p.name}", "orbitals.grid",
                        {"path": str(p), "root": str(st.root), **r}, client=client,
                        finish=_volume_finish(st, key), supersede=("orbitals.grid", client, str(p)))
    return gui_tasks.answer(handler, st.tasks, t, float(q("wait", "0.3")))


def post_route(handler, route, body):
    """orbitals/save: the shown orbital's cube in ``<job>_orbitals/``."""
    from mace.orbitals.cube import cache_dir_for
    from mace.orbitals.outfile import split_name
    st = handler.state
    p = _molden_path(st, body.get("path"))
    cells = body.get("cells") or [1, 1, 1]
    if isinstance(cells, str):
        cells = cells.lower().split("x")
    r = check_request(body.get("k"), body.get("spin"), body.get("mos") or body.get("mo"), body.get("part", "re"),
                      cells, body.get("quality", "normal"), body.get("degenerate_sum"))
    root = split_name(p.name)["root"] or p.stem
    folder = cache_dir_for(p.parent, root)
    # the folder's checks (inside the root, no hidden part, no symlink out); the cube itself is a new .cube
    # there (make_cube reuses an up-to-date one; it rewrites only its own stale cubes)
    st.check_write(folder / "orbital.cube", allow=("grid",))
    t = st.tasks.submit("orbitals", f"Saving the cube of {p.name}", "orbitals.save",
                        {"path": str(p), "root": str(st.root), **r}, client=handler.headers.get("X-Mace-Client"))
    if not st.tasks.wait(t, 600) or t.state != "done":
        return handler._json({"error": t.error or "the save did not finish", "task": t.record()}, 500)
    res = t.result
    st.tasks.remove(t.id)
    cube = Path(res["cube"])
    return handler._json({"path": st.rel(cube), "cached": res["cached"], "bytes": res["bytes"],
                          "folder": st.rel(cube.parent), "folder_files": res["folder_files"],
                          "folder_bytes": res["folder_bytes"], "warning": res["warning"]})
