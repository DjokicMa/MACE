"""Volumetric grids for the GUI's Volume document (plan E5): densities,
potentials, spin densities, differences and CRYSTAL's signed Wannier
orbital cubes.

Nothing re-reads a cube in a new way: ``CubeFile`` of mace plotting's cube
engine reads the file (its type rule, its slab handling, its vacuum crop);
``perform_cube_arithmetic`` with ``align_grids`` combines grids;
``interpolate_cube_at_vertices`` samples a potential on a surface for the
MEP colors, with the CLI's own range rule; ``write_cube_file`` saves a
result. This module only decimates the grid for the browser (3Dmol builds
the isosurface there), computes the statistics on the full grid, and cuts
slices. Everything runs in a pool worker; the arrays travel as float32.
"""
from __future__ import annotations

import contextlib
import io
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from mace.constants import BOHR_TO_ANGSTROM as BOHR  # noqa: E402
QUALITY = {"low": 250_000, "medium": 1_000_000, "high": 2_000_000,
           # 1.1.6: the Publication mesh preset (mace/analysis/mesh_quality.py VOLUME)
           "publication": 8_000_000}
MAX_OPERANDS = 8
MAX_VERTICES = 50_000
SIGNED = ("spin", "potential", "density_difference", "potential_difference", "spin_difference", "difference", "orbital")


def _engine():
    from mace.plotting.engines import crystal_cubeviz_plotly as E
    return E


def read(path: str, skip_vacuum: bool = False):
    """CubeFile, its progress prints kept out of the server's log."""
    E = _engine()
    with contextlib.redirect_stdout(io.StringIO()):
        c = E.CubeFile(str(path), skip_vacuum_detection=skip_vacuum)
    from .filekind import cube_type
    token = cube_type(Path(path).name)
    if token == "orbital":
        c.data_type = "orbital"          # CRYSTAL's LOCALWF/PRINTPLO Wannier functions: a signed volume
    elif token and "difference" not in str(c.data_type) and c.data_type != token:
        # the CLI's rule reads the whole path, so a folder named ECH3POT3 makes every cube a
        # density; the file's own name token (DENS / POT / SPIN) decides here
        c.data_type = token
    return c


def _symbol(z: int) -> str:
    """An element's symbol: ase's table, else MACE's own H-Lr table (no ase in a lean install).

    The fallback needs neither ase nor plotly, so the orbitals viewer, which calls this
    without reading a cube, works without the cube engine."""
    try:
        from ase.data import chemical_symbols
        return chemical_symbols[z] if 0 < z < len(chemical_symbols) else "X"
    except ImportError:
        from mace.utils.formula import ELEMENTS
        return ELEMENTS[z - 1] if 0 < z <= len(ELEMENTS) else "X"


def _label(t: str) -> str:
    return {"density": "Density", "spin": "Spin density", "potential": "Potential", "orbital": "Orbital (Wannier function)",
            "density_difference": "Density difference", "potential_difference": "Potential difference",
            "spin_difference": "Spin difference", "difference": "Difference"}.get(t, t.capitalize())


def _ecp(cube_path: Path) -> Optional[bool]:
    """Did the run use effective core potentials? From the sibling SCF .out's
    basis section ("PSEUDOPOTENTIAL" lines), when one is beside the cube."""
    stem = cube_path.name
    for tok in ("_DENS.CUBE", "_SPIN.CUBE", "_POT.CUBE", "DENS_CUBE.DAT", "SPIN_CUBE.DAT", "POT_CUBE.DAT"):
        if stem.upper().endswith(tok):
            stem = stem[: -len(tok)]
            break
    stem = stem.rstrip("_.")
    cands = [cube_path.parent / f"{stem}.out"] if stem else []
    for c in cands:
        if c.is_file():
            try:
                with open(c, "rb") as fh:
                    head = fh.read(8 << 20).decode("utf-8", "replace")
            except OSError:
                return None
            return "PSEUDOPOTENTIAL" in head.upper()
    return None


def _periodic_axes(c) -> List[bool]:
    """Which lattice directions are periodic: CRYSTAL writes the cell (Bohr) on the
    second comment line; a molecule's is all zeros, a slab's c (a polymer's b, c)
    the formal 500 A."""
    cp = getattr(c, "cell_parameters", None) or {}
    return [0 < float(cp.get(k, 0)) < 400 / BOHR * 0.5 for k in "abc"]


def stats(c, path: Optional[Path] = None) -> Dict[str, Any]:
    """Min, max, percentiles and (for a density) the integral, all on the FULL grid."""
    import numpy as np
    d = np.asarray(c.data, dtype=np.float64)
    out: Dict[str, Any] = {"min": float(d.min()), "max": float(d.max()), "mean": float(d.mean()),
                           "p01": float(np.percentile(d, 1)), "p99": float(np.percentile(d, 99)),
                           "absmax": float(np.abs(d).max()), "n": int(d.size)}
    # values this small are numerical noise (a closed-shell run's SPIN cube): mace plotting warns too
    if 0 < out["absmax"] < 1e-6:
        out["noise_note"] = ("The values are at most {:.1e}: numerical noise".format(out["absmax"])
                             + (" (no spin density: a closed-shell run)" if c.data_type == "spin" else "")
                             + ", so the surface shows nothing physical.")
    if c.data_type != "density":
        return out
    dv = abs(float(np.linalg.det(np.asarray(c.voxel_vectors, float))))
    # which axes are periodic: CRYSTAL writes the cell (Bohr) on the second comment line; a molecule's
    # is all zeros, a slab's c (a polymer's b, c) the formal 500 A
    cp = getattr(c, "cell_parameters", None) or {}
    nvox = np.asarray(c.nvoxels, int)
    step = np.linalg.norm(np.asarray(c.voxel_vectors, float), axis=1)
    periodic = _periodic_axes(c)
    # a periodic grid that repeats its first plane as its last: that plane is counted once
    sl = [slice(None)] * 3
    dup = []
    for ax in range(3):
        if not periodic[ax]:
            continue
        a = np.take(d, 0, axis=ax)
        b = np.take(d, d.shape[ax] - 1, axis=ax)
        if d.shape[ax] > 2 and np.allclose(a, b, rtol=0, atol=1e-8 * max(1e-30, out["absmax"])):
            sl[ax] = slice(0, d.shape[ax] - 1)
            dup.append("abc"[ax])
    core = d[tuple(sl)]
    short = False
    for ax in range(3):
        if periodic[ax]:
            covered = min(abs(nvox[ax] * step[ax] - cp["abc"[ax]]), abs((nvox[ax] - 1) * step[ax] - cp["abc"[ax]]))
            short = short or covered > 0.02 * cp["abc"[ax]]
    # cut off: along a non-periodic axis the density at the grid's faces is not small
    faces = [float(np.abs(np.take(core, i, axis=ax)).max()) for ax in range(3) if not periodic[ax]
             for i in (0, core.shape[ax] - 1)]
    cut = bool(faces) and max(faces) > 1e-3 * out["absmax"]
    if cut or short:
        out["integral"] = None
        out["integral_note"] = ("The grid does not cover the cell: the density is cut off at its faces"
                                if cut else "The grid spans less than the cell")
    else:
        out["integral"] = float(core.sum() * dv)
        ecp = _ecp(path) if path is not None else None
        out["ecp"] = ecp
        out["integral_note"] = ("Counts the electrons the grid covers. With effective core potentials (ECP) the core "
                                "electrons are not in the density, so this is the valence count. On a coarse grid the "
                                "density peaks at the nuclei make it a rough estimate.")
        if ecp is None:
            out["integral_note"] += " (Valence electrons if ECPs were used: no output beside the cube to tell.)"
        elif ecp:
            out["integral_note"] += " This run used ECPs."
    if dup:
        out["duplicate_planes"] = dup
    return out


def default_level(c) -> float:
    """The level the isosurface view starts at, the same rule as mace plotting's
    (``default_iso_level``): the middle of generate_smart_isovalues; an orbital at
    +-0.2 (CRYSTAL's PRINTPLO example) or half its largest |value|."""
    return _engine().default_iso_level(c)


def _crop(c):
    b = getattr(c, "data_bounds", None)
    if not b:
        return (0, c.data.shape[0]), (0, c.data.shape[1]), (0, c.data.shape[2])
    return tuple(b[k] for k in ("x", "y", "z"))


def mesh_knobs(quality: str = "medium", points=None, upsample=None) -> Dict[str, Any]:
    """{points, upsample} of a grid request: the quality's budget (low/medium/high, or a mesh
    preset's: publication), with the Advanced knobs given by hand inside their limits."""
    from mace.analysis import mesh_quality as MQ
    if quality not in QUALITY:
        raise ValueError(f"quality is {', '.join(QUALITY)}")
    spec = {"quality": "publication" if quality == "publication" else "standard", "points": points, "upsample": upsample}
    out = MQ.volume(spec, legacy=quality)
    return {"points": int(out["points"]), "upsample": int(out["upsample"])}


def wrap_axes(c) -> List[Optional[tuple]]:
    """Per axis, ``(planes, period)`` of a periodic axis the page repeats the surface along, else None:
    the grid's distinct planes in one cell (a closing plane that repeats the first, as CRYSTAL writes
    it, is not counted) and the cell's length in voxels. A grid kept at every 2nd point of CRYSTAL's
    spans a period that is not whole voxels (50 planes, 49.5 voxels); a grid shorter or longer than
    its cell by more than a plane does not wrap. The cell's length is CRYSTAL's (the second comment
    line): CubeFile's lattice_vectors are the points times the voxel, a voxel more than a closed
    grid's cell."""
    import numpy as np
    d = np.asarray(c.data)
    periodic = _periodic_axes(c)
    cp = getattr(c, "cell_parameters", None) or {}
    vv = np.asarray(c.voxel_vectors, float)
    top = max(1e-30, float(np.abs(d).max())) if d.size else 1.0
    out: List[Optional[tuple]] = []
    for ax in range(3):
        n, s = int(d.shape[ax]), float(np.linalg.norm(vv[ax]))
        if not periodic[ax] or n < 3 or s <= 0:
            out.append(None)
            continue
        planes = n - 1 if np.allclose(np.take(d, 0, axis=ax), np.take(d, n - 1, axis=ax), rtol=0, atol=1e-8 * top) else n
        period = float(cp["abc"[ax]]) / s
        if abs(period - round(period)) < 1e-3:   # the header's 6 decimals
            period = float(round(period))
        gap = period - (planes - 1)              # from the last plane to the next cell's first
        out.append((planes, period) if 0.05 <= gap <= 2.0 + 1e-6 else None)
    return out


def _wrap_resample(g, ax: int, planes: int, period: float, m: int, cubic: bool):
    """``m`` planes of one period along ``ax`` at x_j = j * period / m (voxels of ``g``), plus the
    closing plane (the first again). Whole-voxel positions are taken as they are; others are
    interpolated across the cell's face too: linearly, or (``cubic``, a refined grid) by the
    periodic cubic spline."""
    import numpy as np
    x = np.arange(m) * (period / m)
    src = np.take(g, np.arange(planes), axis=ax)
    if abs(period - planes) < 1e-6 and m == planes:
        out = src
    elif abs(period - planes) < 1e-6 and np.allclose(x, np.round(x), rtol=0, atol=1e-9):
        out = np.take(src, np.round(x).astype(int) % planes, axis=ax)
    elif cubic:
        from scipy.interpolate import CubicSpline
        xs = np.r_[np.arange(planes, dtype=float), period]
        ys = np.concatenate([src, np.take(src, [0], axis=ax)], axis=ax)
        out = CubicSpline(xs, ys, axis=ax, bc_type="periodic")(x)
    else:
        i0 = np.minimum(np.floor(x + 1e-9).astype(int), planes - 1)
        last = i0 == planes - 1
        h = np.where(last, period - (planes - 1), 1.0)
        t = np.clip((x - i0) / h, 0.0, 1.0)
        shape = [1, 1, 1]
        shape[ax] = m
        t = t.reshape(shape)
        out = np.take(src, i0, axis=ax) * (1 - t) + np.take(src, np.where(last, 0, i0 + 1), axis=ax) * t
    return np.concatenate([out, np.take(out, [0], axis=ax)], axis=ax)


def decimate(c, quality: str = "medium", points=None, upsample=None):
    """(grid float32 C-order, meta) of the vacuum-cropped grid, strided to the quality budget
    (``points``/``upsample``: mesh_knobs); a grid under the budget is refined ``upsample`` times
    by cubic interpolation while it stays under it (corners kept: the same box, finer voxels).

    A periodic axis (``wrap_axes``) is never cropped and is sent as whole voxels of one period
    plus the closing plane, its voxel the lattice vector over its count (``meta['wrap']``): the
    page pads it with the next cell's planes, so each cell's surface reaches the face and the
    surfaces of a supercell join."""
    import numpy as np
    kn = mesh_knobs(quality, points, upsample)
    budget = kn["points"]
    wrap = wrap_axes(c)
    bounds = [(0, int(c.data.shape[ax])) if wrap[ax] else b for ax, b in enumerate(_crop(c))]
    (x0, x1), (y0, y1), (z0, z1) = bounds
    sub = c.data[x0:x1, y0:y1, z0:z1]
    n = sub.size
    stride = max(1, int(math.ceil((n / budget) ** (1.0 / 3.0)))) if n > budget else 1
    g = sub[tuple(slice(None) if wrap[ax] else slice(None, None, stride) for ax in range(3))]
    vv = np.asarray(c.voxel_vectors, float)

    def count(ax, m, u):
        # the planes sent along an axis: a period's (strided, or refined u times) plus the closing plane
        if wrap[ax]:
            return (int(math.ceil(wrap[ax][0] / stride)) if stride > 1 else wrap[ax][0] * u) + 1
        return (m - 1) * u + 1

    up = kn["upsample"]
    while up > 1 and stride == 1 and math.prod(count(ax, m, up) for ax, m in enumerate(g.shape)) > budget:
        up -= 1
    if not (up > 1 and stride == 1 and min(g.shape) > 1):
        up = 1
    vox = vv * stride / up
    if up > 1 and not all(wrap):
        from scipy import ndimage
        out = tuple(m if wrap[ax] else count(ax, m, up) for ax, m in enumerate(g.shape))
        g = ndimage.zoom(np.asarray(g, float), [o / m for o, m in zip(out, g.shape)], order=3, mode="nearest", grid_mode=False)
    for ax in range(3):
        if wrap[ax]:
            planes, period = wrap[ax]
            m = count(ax, 0, up) - 1
            g = _wrap_resample(np.asarray(g, float), ax, planes, period, m, cubic=up > 1)
            vox[ax] = vv[ax] * period / m
    g = np.ascontiguousarray(g, dtype="<f4")
    origin = np.asarray(c.origin, float) + x0 * vv[0] + y0 * vv[1] + z0 * vv[2]
    meta = {"shape": list(g.shape), "full_shape": [int(v) for v in c.nvoxels], "stride": stride, "upsample": up,
            "points_budget": budget,
            "origin": (origin * BOHR).round(8).tolist(), "voxels": (vox * BOHR).round(10).tolist(),
            "crop": [[int(x0), int(x1)], [int(y0), int(y1)], [int(z0), int(z1)]],
            "wrap": [bool(w) for w in wrap],
            "cell_origin": (np.asarray(c.origin, float) * BOHR).round(8).tolist(), "full_voxels": (vv * BOHR).round(10).tolist()}
    return g, meta


def cell_lattice(c, meta: Dict[str, Any]):
    """The cell's vectors (Å, rows) the page repeats the atoms and the surfaces by: along a wrapping
    axis its period (CubeFile's points times the voxel overshot a closed grid's cell by one voxel, and
    the cells of a supercell were drawn that far apart), else the grid's points times its voxel."""
    import numpy as np
    lat = np.asarray(c.lattice_vectors, float) * BOHR
    for ax in range(3):
        if meta.get("wrap", [False] * 3)[ax]:
            lat[ax] = np.asarray(meta["voxels"][ax], float) * (meta["shape"][ax] - 1)
    return lat


def payload(c, path: Path, quality: str = "medium", name: Optional[str] = None, points=None, upsample=None) -> Dict[str, Any]:
    """Everything the page needs except the grid's bytes."""
    import numpy as np
    g, meta = decimate(c, quality, points, upsample)
    lat = cell_lattice(c, meta)
    atoms = {"numbers": [int(z) % 100 for z in c.atomic_numbers], "symbols": [_symbol(int(z) % 100) for z in c.atomic_numbers],
             "cart": (np.asarray(c.atomic_positions, float).reshape(-1, 3) * BOHR).round(6).tolist()}
    t = c.data_type
    lo, hi = float(c.data.min()), float(c.data.max())
    # signed: a signed kind, or values of both signs beyond round-off (a density's -1e-15 is not a lobe)
    signed = t in SIGNED or (lo < 0 < hi and -lo > 1e-6 * max(abs(lo), abs(hi)))
    # unit: the values' unit in plot markup (the slice's colorbar), as mace plotting's slices
    return {"name": name or path.name, "type": t, "label": _label(t), "unit": _engine().cube_value_unit(c), "signed": signed,
            "quality": quality, **meta, "lattice": lat.round(8).tolist(), "periodic": any(_periodic_axes(c)),
            "atoms": atoms, "stats": stats(c, path), "level": default_level(c), "nbytes": int(g.nbytes),
            "pbc": _periodic_axes(c), "box": grid_box(c),
            "comment": [c.comment1[:200], c.comment2[:200]]}, g


# ---------------------------------------------------------------- pool jobs
def load_job(ctx=None, path: str = "", quality: str = "medium", points=None, upsample=None, **_):
    if ctx is not None:
        ctx.progress(0.1, "Reading the cube")
    c = read(path)
    if ctx is not None:
        ctx.check()
        ctx.progress(0.7, "Statistics on the full grid")
    meta, g = payload(c, Path(path), quality, points=points, upsample=upsample)
    return {"meta": meta, "grid": g.tobytes()}


def slice_job(ctx=None, path: str = "", axis: str = "c", index: int = 0, paths: Sequence[str] = (),
              coefficients: Sequence[float] = (), **_):
    """A lattice plane at full resolution: the raw values, and the same plane
    resampled on Cartesian in-plane axes (a skewed cell's plane is a parallelogram).
    With ``paths`` and ``coefficients``: a plane of that combination (a difference
    made in the GUI), on the grid ``combine_job`` draws."""
    import numpy as np
    if paths:
        data, c = combination(paths, coefficients)
        c.data = data
    else:
        c = read(path, skip_vacuum=True)
    ax = "abc".index(axis)
    n = c.data.shape[ax]
    index = int(index)
    if not 0 <= index < n:
        raise ValueError(f"index runs from 0 to {n - 1}")
    vals = np.take(c.data, index, axis=ax)
    others = [i for i in range(3) if i != ax]
    vv = np.asarray(c.voxel_vectors, float) * BOHR
    u, v = vv[others[0]] * c.data.shape[others[0]], vv[others[1]] * c.data.shape[others[1]]
    e1 = u / np.linalg.norm(u)
    v_perp = v - np.dot(v, e1) * e1
    e2 = v_perp / (np.linalg.norm(v_perp) or 1)
    # the plane's corners in (X, Y): 0, u, v, u + v
    P = np.array([[0, 0], [np.dot(u, e1), 0], [np.dot(v, e1), np.dot(v, e2)], [np.dot(u + v, e1), np.dot(u + v, e2)]])
    nx, ny = vals.shape
    X = np.linspace(P[:, 0].min(), P[:, 0].max(), nx)
    Y = np.linspace(P[:, 1].min(), P[:, 1].max(), ny)
    M = np.array([[np.dot(u, e1), np.dot(v, e1)], [0.0, np.dot(v, e2)]])
    XX, YY = np.meshgrid(X, Y, indexing="ij")
    st = np.linalg.solve(M, np.vstack([XX.ravel(), YY.ravel()]))
    s, t = st[0] * nx, st[1] * ny               # fractional -> grid index
    inside = (s >= 0) & (s <= nx - 1) & (t >= 0) & (t <= ny - 1)
    i0 = np.clip(np.floor(s).astype(int), 0, nx - 2 if nx > 1 else 0)
    j0 = np.clip(np.floor(t).astype(int), 0, ny - 2 if ny > 1 else 0)
    fs, ft = s - i0, t - j0
    i1, j1 = np.minimum(i0 + 1, nx - 1), np.minimum(j0 + 1, ny - 1)
    cart = (vals[i0, j0] * (1 - fs) * (1 - ft) + vals[i1, j0] * fs * (1 - ft) + vals[i0, j1] * (1 - fs) * ft + vals[i1, j1] * fs * ft)
    cart = np.where(inside, cart, np.nan).reshape(nx, ny)
    return {"axis": axis, "index": index, "n": n, "frac": index / n, "values": vals.astype(float).round(10).tolist(),
            "x": X.round(5).tolist(), "y": Y.round(5).tolist(),
            "cart": [[None if not np.isfinite(z) else round(float(z), 10) for z in row] for row in cart],
            "corners": P.round(5).tolist(), "position_A": float(np.linalg.norm(vv[ax]) * index),
            "type": c.data_type}


# ---------------------------------------------------------------- arbitrary planes (1.1.6)
# A slice through the grid along a Miller plane (hkl) or normal to a lattice direction [uvw], at a
# position across the box, optionally tilted and twisted about its own in-plane axes. The box is the
# grid's own: a periodic axis spans its period (a grid that repeats its first plane as its last counts
# it once), a non-periodic one (a slab's c, a molecule's x, y, z) spans the grid points. Miller indices
# refer to the box's edges, which are the lattice vectors of a 3D-periodic cube. The page draws the same
# geometry (js/docs/slice_widget.js planeGeometry); `mace slice` writes it as a picture or a table.
PLANE_MODES = ("hkl", "uvw")
PLANE_MAX_SIDE = 400          # sample points along each in-plane axis
PLANE_MAX_INDEX = 12          # |h|, |k|, |l| (or |u|, |v|, |w|) at most
PLANE_END_EPS = 1e-3          # the plane stays this fraction of the box's depth inside its ends


def grid_period(c) -> List[int]:
    """Grid points per period along each axis: a periodic grid that repeats its first plane as its
    last counts it once; a non-periodic axis has no period (its point count is returned)."""
    import numpy as np
    d = np.asarray(c.data)
    periodic = _periodic_axes(c)
    top = max(1e-30, float(np.abs(d).max())) if d.size else 1.0
    out = []
    for ax in range(3):
        n = int(d.shape[ax])
        if periodic[ax] and n > 2:
            a, b = np.take(d, 0, axis=ax), np.take(d, n - 1, axis=ax)
            if np.allclose(a, b, rtol=0, atol=1e-8 * top):
                n -= 1
        out.append(n)
    return out


def grid_box(c) -> Dict[str, Any]:
    """The box the plane slices cut (Å): origin and edge vectors, which axes wrap, and their periods."""
    import numpy as np
    periodic = _periodic_axes(c)
    period = grid_period(c)
    vv = np.asarray(c.voxel_vectors, float) * BOHR
    edges = [(vv[k] * (period[k] if periodic[k] else max(1, period[k] - 1))).round(8).tolist() for k in range(3)]
    return {"origin": (np.asarray(c.origin, float) * BOHR).round(8).tolist(), "edges": edges,
            "pbc": periodic, "period": period}


def _plane_ints(ints) -> List[int]:
    try:
        vals = [int(round(float(x))) for x in ints]
    except (TypeError, ValueError, OverflowError):
        raise ValueError("the plane's indices are three whole numbers")
    if len(vals) != 3:
        raise ValueError("the plane's indices are three whole numbers")
    if not any(vals):
        raise ValueError("(0 0 0) is not a plane: give at least one nonzero index")
    if max(abs(v) for v in vals) > PLANE_MAX_INDEX:
        raise ValueError(f"indices run from -{PLANE_MAX_INDEX} to {PLANE_MAX_INDEX}")
    return vals


def _rotate(v, axis, deg):
    """v turned by deg about the unit vector axis (Rodrigues)."""
    import numpy as np
    v, k = np.asarray(v, float), np.asarray(axis, float)
    t = math.radians(float(deg))
    if not t:
        return v
    return v * math.cos(t) + np.cross(k, v) * math.sin(t) + k * np.dot(k, v) * (1 - math.cos(t))


def box_polygon(origin, edges, normal, point, e1, e2) -> List[List[float]]:
    """Where the plane (normal, point) cuts the box: its corners in order around the plane (Å)."""
    import numpy as np
    o, E = np.asarray(origin, float), np.asarray(edges, float)
    n, p = np.asarray(normal, float), np.asarray(point, float)
    corners = {(i, j, k): o + i * E[0] + j * E[1] + k * E[2] for i in (0, 1) for j in (0, 1) for k in (0, 1)}
    scale = max(1e-9, float(np.abs(E).max()))
    pts: List[Any] = []
    for key, c0 in corners.items():
        for ax in range(3):
            if key[ax]:
                continue
            c1 = corners[tuple(1 if a == ax else key[a] for a in range(3))]
            s0, s1 = float(np.dot(c0 - p, n)), float(np.dot(c1 - p, n))
            if abs(s0) < 1e-9 * scale:
                cand = c0
            elif abs(s1) < 1e-9 * scale:
                cand = c1
            elif s0 * s1 < 0:
                cand = c0 + (c1 - c0) * (s0 / (s0 - s1))
            else:
                continue
            if not any(np.linalg.norm(cand - q) < 1e-6 * scale for q in pts):
                pts.append(cand)
    if len(pts) < 3:
        return []
    P = np.array(pts)
    m = P.mean(axis=0)
    ang = np.arctan2((P - m) @ np.asarray(e2, float), (P - m) @ np.asarray(e1, float))
    return P[np.argsort(ang)].round(8).tolist()


def plane_geometry(box: Dict[str, Any], mode: str, ints, pos: float = 0.5, tilt: float = 0.0,
                   twist: float = 0.0) -> Dict[str, Any]:
    """The plane: (hkl) (normal h a* + k b* + l c*) or normal to [uvw] (u a + v b + w c), at ``pos``
    (0 to 1) across the box along that normal, turned by ``tilt`` about its first in-plane axis and
    ``twist`` about its second (degrees) around the point of the plane nearest the box's center."""
    import numpy as np
    if mode not in PLANE_MODES:
        raise ValueError("the plane is hkl (Miller indices) or uvw (normal to a direction)")
    vals = _plane_ints(ints)
    pos, tilt, twist = float(pos), float(tilt), float(twist)
    if not (math.isfinite(pos) and 0.0 <= pos <= 1.0):
        raise ValueError("the position runs from 0 to 1 across the box")
    if not (math.isfinite(tilt) and math.isfinite(twist)) or abs(tilt) > 90 or abs(twist) > 90:
        raise ValueError("tilt and twist run from -90 to 90 degrees")
    o, E = np.asarray(box["origin"], float), np.asarray(box["edges"], float)
    if abs(np.linalg.det(E)) < 1e-12:
        raise ValueError("the grid's box is flat: no plane to cut")
    nvec = np.asarray(vals, float) @ (np.linalg.inv(E).T if mode == "hkl" else E)
    nlen = float(np.linalg.norm(nvec))
    nhat = nvec / nlen
    corners = np.array([o + i * E[0] + j * E[1] + k * E[2] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
    proj = (corners - o) @ nhat
    dmin, dmax = float(proj.min()), float(proj.max())
    # a hair inside the box: at 0 or 1 an oblique plane only touches a corner or an edge
    eps = PLANE_END_EPS * (dmax - dmin)
    d = min(max(dmin + pos * (dmax - dmin), dmin + eps), dmax - eps)
    center = o + 0.5 * E.sum(axis=0)
    point = center + (d - float(np.dot(center - o, nhat))) * nhat
    # the in-plane axes: the box edge least along the normal, projected into the plane, and normal x it
    k = int(np.argmin([abs(np.dot(E[i], nhat)) / max(1e-12, float(np.linalg.norm(E[i]))) for i in range(3)]))
    e1 = E[k] - np.dot(E[k], nhat) * nhat
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(nhat, e1)
    n2, e2b = _rotate(nhat, e1, tilt), _rotate(e2, e1, tilt)
    n3, e1c = _rotate(n2, e2b, twist), _rotate(e1, e2b, twist)
    poly = box_polygon(o, E, n3, point, e1c, e2b)
    out = {"mode": mode, "ints": vals, "pos": pos, "tilt": tilt, "twist": twist,
           "normal": n3.round(10).tolist(), "point": point.round(8).tolist(), "e1": e1c.round(10).tolist(),
           "e2": e2b.round(10).tolist(), "polygon": poly, "d": d, "dmin": dmin, "dmax": dmax,
           "position_A": d - dmin}
    if mode == "hkl":
        out["spacing_A"] = 1.0 / nlen                    # d_hkl of the box's lattice
        out["miller_t"] = d * nlen                       # h x + k y + l z = t (fractional x, y, z)
    return out


def plane_sample(data, grid_origin, voxels, period, pbc, geom: Dict[str, Any], step: Optional[float] = None,
                 max_side: int = PLANE_MAX_SIDE) -> Dict[str, Any]:
    """The grid's values on the plane's polygon, linearly interpolated (periodic axes wrap), on a
    square in-plane mesh (Å along e1 and e2 from the plane's pivot point). Outside the polygon: NaN."""
    import numpy as np
    D = np.asarray(data)
    V = np.asarray(voxels, float)
    go = np.asarray(grid_origin, float)
    poly = np.asarray(geom["polygon"], float)
    if len(poly) < 3:
        raise ValueError("the plane does not cut the grid's box (lower the tilt or twist)")
    p, e1, e2 = (np.asarray(geom[k], float) for k in ("point", "e1", "e2"))
    U, W = (poly - p) @ e1, (poly - p) @ e2
    span = float(max(U.max() - U.min(), W.max() - W.min()))
    step = float(step) if step else float(np.linalg.norm(V, axis=1).min())
    step = max(step, span / max(2, max_side - 1))
    xs = np.arange(U.min(), U.max() + 0.5 * step, step)
    ys = np.arange(W.min(), W.max() + 0.5 * step, step)
    # a sliver (a plane through a corner or an edge of the box): two points across it at least
    if len(xs) < 2:
        xs = np.array([U.min(), U.max()])
    if len(ys) < 2:
        ys = np.array([W.min(), W.max()])
    XX, YY = np.meshgrid(xs, ys)                          # rows: y; columns: x
    pts = p + XX[..., None] * e1 + YY[..., None] * e2
    g = (pts - go) @ np.linalg.inv(V)                     # fractional grid index along each axis
    idx0, idx1, frac = [], [], []
    valid = np.ones(XX.shape, bool)
    for ax in range(3):
        x = g[..., ax]
        if pbc[ax]:
            P = max(1, int(period[ax]))
            x = np.mod(x, P)
            i0 = np.floor(x).astype(int)
            f = x - i0
            i0 = np.mod(i0, P)
            i1 = np.mod(i0 + 1, P)
        else:
            n = D.shape[ax]
            valid &= (x >= -1e-6) & (x <= n - 1 + 1e-6)
            x = np.clip(x, 0, n - 1)
            i0 = np.clip(np.floor(x).astype(int), 0, max(0, n - 2))
            f = x - i0
            i1 = np.minimum(i0 + 1, n - 1)
        idx0.append(i0)
        idx1.append(i1)
        frac.append(f)
    val = np.zeros(XX.shape, float)
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                w = (frac[0] if a else 1 - frac[0]) * (frac[1] if b else 1 - frac[1]) * (frac[2] if c else 1 - frac[2])
                val += w * D[idx1[0] if a else idx0[0], idx1[1] if b else idx0[1], idx1[2] if c else idx0[2]]
    # inside the polygon (it is convex: the cut of a box): on the inner side of every edge
    Q = np.stack([U, W], axis=1)
    m = len(Q)
    area = sum(Q[i, 0] * Q[(i + 1) % m, 1] - Q[(i + 1) % m, 0] * Q[i, 1] for i in range(m))
    sgn = 1.0 if area >= 0 else -1.0
    inside = np.ones(XX.shape, bool)
    tol = 1e-9 * max(1.0, span) ** 2
    for i in range(m):
        a, b = Q[i], Q[(i + 1) % m]
        inside &= sgn * ((b[0] - a[0]) * (YY - a[1]) - (b[1] - a[1]) * (XX - a[0])) >= -tol
    val = np.where(valid & inside, val, np.nan)
    return {"x": xs.round(5).tolist(), "y": ys.round(5).tolist(), "values": val,
            "outline": [U.round(5).tolist(), W.round(5).tolist()], "step_A": step}


def plane_of_cube(c, mode: str, ints, pos: float = 0.5, tilt: float = 0.0, twist: float = 0.0,
                  max_side: int = PLANE_MAX_SIDE) -> Dict[str, Any]:
    """The geometry and the sampled values (an array, NaN outside) of a plane of a read cube (Å)."""
    import numpy as np
    box = grid_box(c)
    geom = plane_geometry(box, mode, ints, pos, tilt, twist)
    vv = np.asarray(c.voxel_vectors, float) * BOHR
    s = plane_sample(c.data, box["origin"], vv, box["period"], box["pbc"], geom, max_side=max_side)
    return {**geom, **s, "box": box, "type": c.data_type}


def plane_job(ctx=None, path: str = "", mode: str = "hkl", ints: Sequence[int] = (0, 0, 1), pos: float = 0.5,
              tilt: float = 0.0, twist: float = 0.0, paths: Sequence[str] = (), coefficients: Sequence[float] = (), **_):
    """A (hkl) or [uvw] plane at full resolution (the Volume document's Slice section)."""
    import numpy as np
    if paths:
        data, c = combination(paths, coefficients)
        c.data = data
    else:
        c = read(path, skip_vacuum=True)
    if ctx is not None:
        ctx.check()
        ctx.progress(0.6, "Sampling the plane")
    r = plane_of_cube(c, mode, ints, pos, tilt, twist)
    vals = r.pop("values")
    fin = vals[np.isfinite(vals)]
    r["z"] = [[None if not math.isfinite(z) else round(float(z), 10) for z in row] for row in vals]
    r["min"] = float(fin.min()) if fin.size else None
    r["max"] = float(fin.max()) if fin.size else None
    return r


def plane_route(handler, query):
    """GET /api/volume/plane?path=&mode=hkl|uvw&i=&j=&k=&pos=&tilt=&twist= (or the operands of a
    difference made in the GUI: path=...&coef=... pairs): a pool task, answered as volume/slice is."""
    from . import server as S
    from . import tasks as gui_tasks
    st = handler.state
    q = lambda k, d=None: handler._q(query, k, d)          # noqa: E731
    mode = q("mode", "hkl")
    if mode not in PLANE_MODES:
        raise ValueError("mode is hkl or uvw")
    ints = _plane_ints([q("i", "0"), q("j", "0"), q("k", "1")])

    def num(key, default, lo, hi, what):
        try:
            x = float(q(key, default))
        except (TypeError, ValueError):
            raise ValueError(f"{what} is a number")
        if not (math.isfinite(x) and lo <= x <= hi):
            raise ValueError(f"{what} runs from {lo:g} to {hi:g}")
        return x
    job: Dict[str, Any] = {"mode": mode, "ints": ints, "pos": num("pos", "0.5", 0, 1, "the position"),
                           "tilt": num("tilt", "0", -90, 90, "the tilt"), "twist": num("twist", "0", -90, 90, "the twist")}
    if query.get("coef"):
        coefs = []
        for x in query.get("coef") or []:
            try:
                coefs.append(float(x))
            except ValueError:
                raise ValueError("coefficients must be finite numbers")
        paths, coefs = S._volume_operands(st, {"paths": list(query.get("path") or []), "coefficients": coefs})
        job.update(paths=[str(x) for x in paths], coefficients=coefs)
        title = "Plane of the difference"
    else:
        p = S._cube_path(st, q("path"))
        job["path"] = str(p)
        title = f"Plane of {p.name}"
    t = st.tasks.submit("volume", title, "volume.plane", job, client=handler.headers.get("X-Mace-Client"), quiet=True)
    return gui_tasks.answer(handler, st.tasks, t, 120)


def align_affine(cubes):
    """The grids of ``cubes`` on one grid that holds them all, by the exact affine
    map from a grid point to each cube's own (fractional) voxel index, sampled
    linearly (0 outside a cube). Grids that differ by whole voxels are copied, not
    interpolated. Returns (list of arrays, origin (Bohr), voxel vectors (Bohr)).

    mace plotting's ``align_cube_grids`` interpolates one grid point at a time in
    Python (hours for two 100 x 100 x 792 slab grids) and finds a point's index by
    projecting onto each voxel direction, which shears a skewed (hexagonal) cell;
    this is the same alignment with the full inverse of the voxel matrix."""
    import numpy as np
    from scipy.ndimage import map_coordinates
    ref = cubes[0]
    V0 = np.asarray(ref.voxel_vectors, float)
    o0 = np.asarray(ref.origin, float)
    inv0 = np.linalg.inv(V0.T)
    starts, ends = [], []
    for c in cubes:
        V = np.asarray(c.voxel_vectors, float)
        if np.abs(V - V0).max() > 1e-6 * np.abs(V0).max() and not E_spacing_ok(V0, V):
            raise ValueError("the grids' voxel vectors differ in direction or by more than 5 %: they cannot be aligned")
        corners = [np.asarray(c.origin, float) + V.T @ np.array(n) for n in
                   [(0, 0, 0)] + [tuple((np.array(c.data.shape) - 1) * np.array(m)) for m in
                                  [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]]]
        f = np.array([inv0 @ (x - o0) for x in corners])
        starts.append(f.min(axis=0))
        ends.append(f.max(axis=0))
    start = np.floor(np.min(starts, axis=0) + 1e-6)
    end = np.ceil(np.max(ends, axis=0) - 1e-6)
    shape = tuple(int(x) for x in (end - start + 1))
    origin = o0 + V0.T @ start
    out = []
    jj, kk = np.meshgrid(np.arange(shape[1]), np.arange(shape[2]), indexing="ij")
    for c in cubes:
        V = np.asarray(c.voxel_vectors, float)
        inv = np.linalg.inv(V.T)
        shift = inv @ (origin - np.asarray(c.origin, float))
        A = inv @ V0.T                       # unified index -> this cube's index
        res = np.zeros(shape, dtype=float)
        exact = np.allclose(A, np.eye(3), atol=1e-9) and np.allclose(shift, np.round(shift), atol=1e-6)
        if exact:
            s = np.round(shift).astype(int)
            src = [slice(max(0, s[d]), min(c.data.shape[d], s[d] + shape[d])) for d in range(3)]
            dst = [slice(src[d].start - s[d], src[d].stop - s[d]) for d in range(3)]
            res[tuple(dst)] = c.data[tuple(src)]
        else:
            for i in range(shape[0]):
                pts = np.stack([np.full(jj.shape, i, float), jj.astype(float), kk.astype(float)])
                idx = np.tensordot(A, pts, axes=1) + shift[:, None, None]
                res[i] = map_coordinates(c.data, idx, order=1, mode="constant", cval=0.0)
        out.append(res)
    return out, origin, V0


def E_spacing_ok(V0, V) -> bool:
    """Same directions, spacing within 5 % (the CLI's check_grid_alignment_compatibility rule)."""
    import numpy as np
    for i in range(3):
        a, b = V0[i] / np.linalg.norm(V0[i]), V[i] / np.linalg.norm(V[i])
        if not np.allclose(a, b, atol=1e-6):
            return False
        if abs(np.linalg.norm(V[i]) - np.linalg.norm(V0[i])) / np.linalg.norm(V0[i]) > 0.05:
            return False
    return True


def combination(paths: Sequence[str], coefficients: Sequence[float]):
    """(result array, template CubeFile on the result's grid): the CLI's linear
    combination; same-shaped grids go through perform_cube_arithmetic itself."""
    import numpy as np
    E = _engine()
    cubes = [read(p, skip_vacuum=True) for p in paths]
    shapes = {tuple(c.data.shape) for c in cubes}
    same = len(shapes) == 1 and all(E.validate_cube_compatibility(cubes[0], c)[0] for c in cubes[1:])
    if same:
        with contextlib.redirect_stdout(io.StringIO()):
            data, template = E.perform_cube_arithmetic(cubes, list(map(float, coefficients)), "linear")
        return data, template
    grids, origin, V = align_affine(cubes)
    data = np.zeros_like(grids[0])
    for g, c in zip(grids, coefficients):
        data += float(c) * g
    t = cubes[0]
    t.origin, t.voxel_vectors, t.nvoxels = origin, V, np.array(data.shape)
    t.lattice_vectors = V * np.array(data.shape)[:, None]
    return data, t


def combine_job(ctx=None, paths: Sequence[str] = (), coefficients: Sequence[float] = (), quality: str = "medium",
                names: Sequence[str] = (), points=None, upsample=None, **_):
    """A linear combination (A - B - C ...), grids aligned when they differ."""
    if ctx is not None:
        ctx.progress(0.2, "Reading and aligning the grids")
    data, template = combination(paths, coefficients)
    template.data = data
    first = read(paths[0], skip_vacuum=True).data_type
    template.data_type = {"density": "density_difference", "potential": "potential_difference",
                          "spin": "spin_difference"}.get(first, "difference")
    with contextlib.redirect_stdout(io.StringIO()):
        template._detect_and_crop_vacuum()
    title = " ".join(("" if k == 0 and c > 0 else ("+ " if c > 0 else "− ")) + (f"{abs(c):g}×" if abs(c) != 1 else "") + n
                     for k, (c, n) in enumerate(zip(coefficients, names or [Path(p).name for p in paths])))
    meta, g = payload(template, Path(paths[0]), quality, name=title, points=points, upsample=upsample)
    meta["combined"] = {"paths": list(paths), "coefficients": [float(c) for c in coefficients]}
    return {"meta": meta, "grid": g.tobytes()}


def save_job(ctx=None, paths: Sequence[str] = (), coefficients: Sequence[float] = (), target: str = "", title: str = "", **_):
    """Write a combination as a cube with the CLI's write_cube_file: into a
    temporary file beside the target, then linked to the target's name, which
    fails if a file of that name appeared meanwhile (never written over, and
    never a half-written cube under the target's name)."""
    import os
    import tempfile
    E = _engine()
    data, template = combination(paths, coefficients)
    tgt = Path(target)
    fd, tmp = tempfile.mkstemp(prefix=".mace-cube-", suffix=".tmp", dir=str(tgt.parent))
    os.close(fd)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            E.write_cube_file(tmp, template, data, comment=f"MACE GUI difference: {title}"[:200])
        try:
            os.link(tmp, tgt)                   # atomic, and refuses an existing name
        except FileExistsError:
            raise ValueError(f"{tgt.name} exists; Save Result As never writes over a file")
        except OSError:
            # a filesystem without hard links: an exclusive create, then the bytes
            with open(tgt, "xb") as out, open(tmp, "rb") as src:
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
    return {"target": target, "bytes": tgt.stat().st_size}


def _range_rule(vals, signed: bool):
    """plot_dual_cube_isosurface's auto range for values sampled on a surface."""
    import numpy as np
    vals = np.asarray(vals, float)
    if signed:
        lo, hi = float(vals.min()), float(vals.max())
        if lo < 0 < hi:
            neg, pos = vals[vals < 0], vals[vals > 0]
            m = max(abs(np.percentile(neg, 5)) if len(neg) else 0.0, np.percentile(pos, 95) if len(pos) else 0.0)
            return -float(m), float(m)
        return lo, hi
    return float(np.percentile(vals, 5)), float(np.percentile(vals, 95))


def mep_range(color_path: str, vertices: Optional[List[float]] = None, shape_path: Optional[str] = None,
              level: Optional[float] = None) -> Dict[str, Any]:
    """The CLI's color range for a surface colored by a potential.

    With the density's file and level (the usual case) the CLI's own
    ``plot_dual_cube_isosurface`` runs on the full grids (its marching-cubes
    surface, its sampling, its rule) when scikit-image is installed; without
    it, the potential is sampled at the density's grid points within 2 % of
    the level (the same surface, no mesh). Without a file (a difference made
    here) the page's own surface vertices are sampled with
    ``interpolate_cube_at_vertices``. The rule: a signed potential gets the
    symmetric +-max(|p5 of the negative values|, p95 of the positive ones),
    anything else p5..p95."""
    import numpy as np
    E = _engine()
    c = read(color_path, skip_vacuum=True)
    signed = bool(c.is_diverging_data())
    if shape_path and level:
        s = read(shape_path)
        cc = read(color_path)
        if getattr(E, "HAS_SKIMAGE", False):
            with contextlib.redirect_stdout(io.StringIO()):
                fig = E.plot_dual_cube_isosurface(s, cc, float(level))
            if fig is not None:
                mesh = next(t for t in fig.data if t.type == "mesh3d")
                return {"vmin": float(mesh.cmin), "vmax": float(mesh.cmax), "n": len(mesh.x), "signed": signed,
                        "type": c.data_type, "method": "cli-mesh"}
        D = np.asarray(s.data)
        lev = float(level)
        idx = np.argwhere(np.abs(D - lev) <= 0.02 * abs(lev))
        if not len(idx):
            raise ValueError("no surface at this level")
        verts = (np.asarray(s.origin) + idx @ np.asarray(s.voxel_vectors)) * BOHR
        method = "near-iso"
    else:
        verts = np.asarray(vertices or [], float).reshape(-1, 3)
        if not len(verts):
            raise ValueError("no surface vertices")
        if len(verts) > MAX_VERTICES:
            verts = verts[:: int(math.ceil(len(verts) / MAX_VERTICES))]
        method = "vertices"
    with contextlib.redirect_stdout(io.StringIO()):
        vals = E.interpolate_cube_at_vertices(c.data, c.origin, c.nvoxels, c.voxel_vectors, verts, BOHR)
    if vals is None:
        raise ValueError("scipy is needed to sample the potential on the surface")
    vmin, vmax = _range_rule(vals, signed)
    return {"vmin": vmin, "vmax": vmax, "n": int(len(verts)), "signed": signed, "type": c.data_type, "method": method}


def same_cell(a_path: str, b_path: str) -> Optional[str]:
    """None when two cubes share a grid (the MEP needs one), else why not."""
    E = _engine()
    a, b = read(a_path, skip_vacuum=True), read(b_path, skip_vacuum=True)
    ok, msg = E.validate_cube_compatibility(a, b)
    return None if ok else f"the two cubes are of different grids or cells ({msg})"
