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

BOHR = 0.529177210903
QUALITY = {"low": 250_000, "medium": 1_000_000, "high": 2_000_000}
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


def decimate(c, quality: str = "medium"):
    """(grid float32 C-order, meta) of the vacuum-cropped grid, strided to the quality budget."""
    import numpy as np
    budget = QUALITY.get(quality, QUALITY["medium"])
    (x0, x1), (y0, y1), (z0, z1) = _crop(c)
    sub = c.data[x0:x1, y0:y1, z0:z1]
    n = sub.size
    stride = max(1, int(math.ceil((n / budget) ** (1.0 / 3.0)))) if n > budget else 1
    g = np.ascontiguousarray(sub[::stride, ::stride, ::stride], dtype="<f4")
    vv = np.asarray(c.voxel_vectors, float)
    origin = np.asarray(c.origin, float) + x0 * vv[0] + y0 * vv[1] + z0 * vv[2]
    meta = {"shape": list(g.shape), "full_shape": [int(v) for v in c.nvoxels], "stride": stride,
            "origin": (origin * BOHR).round(8).tolist(), "voxels": (vv * stride * BOHR).round(10).tolist(),
            "crop": [[int(x0), int(x1)], [int(y0), int(y1)], [int(z0), int(z1)]],
            "cell_origin": (np.asarray(c.origin, float) * BOHR).round(8).tolist(), "full_voxels": (vv * BOHR).round(10).tolist()}
    return g, meta


def payload(c, path: Path, quality: str = "medium", name: Optional[str] = None) -> Dict[str, Any]:
    """Everything the page needs except the grid's bytes."""
    import numpy as np
    g, meta = decimate(c, quality)
    lat = np.asarray(c.lattice_vectors, float) * BOHR
    atoms = {"numbers": [int(z) % 100 for z in c.atomic_numbers], "symbols": [_symbol(int(z) % 100) for z in c.atomic_numbers],
             "cart": (np.asarray(c.atomic_positions, float).reshape(-1, 3) * BOHR).round(6).tolist()}
    t = c.data_type
    lo, hi = float(c.data.min()), float(c.data.max())
    # signed: a signed kind, or values of both signs beyond round-off (a density's -1e-15 is not a lobe)
    signed = t in SIGNED or (lo < 0 < hi and -lo > 1e-6 * max(abs(lo), abs(hi)))
    return {"name": name or path.name, "type": t, "label": _label(t), "signed": signed,
            "quality": quality, **meta, "lattice": lat.round(8).tolist(), "periodic": any(_periodic_axes(c)),
            "atoms": atoms, "stats": stats(c, path), "level": default_level(c), "nbytes": int(g.nbytes),
            "pbc": _periodic_axes(c),
            "comment": [c.comment1[:200], c.comment2[:200]]}, g


# ---------------------------------------------------------------- pool jobs
def load_job(ctx=None, path: str = "", quality: str = "medium", **_):
    if ctx is not None:
        ctx.progress(0.1, "Reading the cube")
    c = read(path)
    if ctx is not None:
        ctx.check()
        ctx.progress(0.7, "Statistics on the full grid")
    meta, g = payload(c, Path(path), quality)
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
                names: Sequence[str] = (), **_):
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
    meta, g = payload(template, Path(paths[0]), quality, name=title)
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
