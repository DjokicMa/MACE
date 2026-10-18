"""Stateless structure builders for the GUI (spec section 2.5, plan B1).

Each builder takes plain arrays (the cell's rows in Angstrom, fractional
coordinates, atomic numbers) and returns a new structure as plain arrays,
so it runs the same in a request thread and in a pool worker (pool.py),
where it reports progress and checks for cancellation through ``ctx``.
Nothing here writes a file or touches a session: the server turns a result
into a new document (a session, or a held view-only structure) or the
editor applies the matching in-place operation (editor.py).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


class _NoCtx:
    """The ctx of a call that nobody follows (tests, inline previews)."""

    def progress(self, fraction: float, message: str = "", force: bool = False) -> None:
        pass

    def check(self) -> None:
        pass

    canceled = False


NOCTX = _NoCtx()


def result(cell, frac, numbers, pbc=(True, True, True), **info) -> Dict[str, Any]:
    out = {"cell": np.asarray(cell, float).reshape(3, 3), "frac": np.asarray(frac, float).reshape(-1, 3),
           "numbers": np.asarray(numbers, dtype=np.int64).ravel(), "pbc": [bool(p) for p in pbc]}
    out.update(info)
    return out


def identity(ctx=NOCTX, cell=None, frac=None, numbers=None, pbc=(True, True, True), delay_s: float = 0.0, **_):
    """Test hook (MACE_GUI_TEST_HOOKS): the source as it is, after a delay
    with progress, so the builder frame can be tested before any builder."""
    t0 = time.time()
    while time.time() - t0 < delay_s:
        ctx.check()
        ctx.progress((time.time() - t0) / max(delay_s, 1e-9), "copying")
        time.sleep(min(0.05, delay_s))
    return result(cell, frac, numbers, pbc, summary=f"{len(numbers)} atoms, unchanged")


# ---------------------------------------------------------------- supercell (T4)
MAX_ENTRY = 30


def check_matrix(matrix) -> np.ndarray:
    """A 3x3 integer matrix with a positive determinant (|m_ij| <= 30)."""
    try:
        m = np.asarray(matrix, dtype=float).reshape(3, 3)
    except (TypeError, ValueError):
        raise ValueError("the matrix is 3 rows of 3 integers")
    if not np.all(np.isfinite(m)) or np.any(np.abs(m - np.round(m)) > 1e-9):
        raise ValueError("the matrix holds integers only")
    m = np.round(m).astype(np.int64)
    if np.abs(m).max() > MAX_ENTRY:
        raise ValueError(f"Matrix entries are limited to ±{MAX_ENTRY}")
    det = int(round(np.linalg.det(m)))
    if det <= 0:
        raise ValueError("The matrix must have a positive determinant")
    return m


def lattice_points(m: np.ndarray, ctx=NOCTX) -> np.ndarray:
    """The integer vectors n (in the old cell) inside the supercell M (rows =
    the new vectors in old units): det(M) of them, lexicographic, z fastest,
    as ase's repeat orders a diagonal supercell's images."""
    det = int(round(np.linalg.det(m)))
    if np.count_nonzero(m - np.diag(np.diag(m))) == 0:
        d = np.diag(m)
        g = np.stack(np.meshgrid(np.arange(d[0]), np.arange(d[1]), np.arange(d[2]), indexing="ij"), -1).reshape(-1, 3)
        return g.astype(np.int64)
    corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)]) @ m
    lo, hi = corners.min(0), corners.max(0)
    inv = np.linalg.inv(m.astype(float))
    out = []
    xs = np.arange(lo[0], hi[0] + 1)
    for k, x in enumerate(xs):                 # one plane at a time: bounded memory, a cancel check per plane
        if k % 16 == 0:
            ctx.check()
            ctx.progress(0.6 * k / max(1, len(xs)), "enumerating lattice points")
        yz = np.stack(np.meshgrid(np.arange(lo[1], hi[1] + 1), np.arange(lo[2], hi[2] + 1), indexing="ij"), -1).reshape(-1, 2)
        pts = np.concatenate([np.full((len(yz), 1), x), yz], 1)
        f = pts @ inv
        keep = np.all((f > -1e-9) & (f < 1 - 1e-9), axis=1)
        out.append(pts[keep])
    pts = np.concatenate(out, 0) if out else np.zeros((0, 3), np.int64)
    if len(pts) != det:
        raise ValueError(f"found {len(pts)} lattice points for a determinant of {det}")
    return pts.astype(np.int64)


def check_periodic(m, pbc) -> None:
    """A slab or polymer repeats along its periodic directions only: M leaves
    each non-periodic vector as it is (1 on the diagonal, 0 beside it)."""
    pbc = [bool(p) for p in pbc]
    if not any(pbc):
        raise ValueError("A molecule has no periodic direction to repeat")
    m = np.asarray(m)
    names = "abc"
    along = " and ".join(names[i] for i in range(3) if pbc[i])
    for i in range(3):
        if pbc[i]:
            continue
        if m[i, i] != 1:
            raise ValueError(f"This structure is periodic along {along} only: n{i + 1} must be 1")
        if any(m[i, j] or m[j, i] for j in range(3) if j != i):
            raise ValueError(f"This structure is periodic along {along} only: the matrix must leave {names[i]} as it is "
                             f"(row and column {i + 1}: 1 on the diagonal, 0 elsewhere)")


def supercell(ctx=NOCTX, cell=None, frac=None, numbers=None, pbc=(True, True, True), matrix=None, **_):
    """The supercell with new vectors M @ cell: every atom at every lattice
    point, image by image (atoms inside each image in their old order), new
    fractional coordinates wrapped into [0, 1) along the periodic directions
    (a slab or polymer repeats along those only; check_periodic)."""
    m = check_matrix(matrix)
    if not all(pbc):
        check_periodic(m, pbc)
    cell = np.asarray(cell, float).reshape(3, 3)
    frac = np.asarray(frac, float).reshape(-1, 3)
    numbers = np.asarray(numbers, dtype=np.int64)
    pts = lattice_points(m, ctx)
    ctx.progress(0.6, "building coordinates")
    inv = np.linalg.inv(m.astype(float))
    n_img, n_at = len(pts), len(frac)
    new = np.empty((n_img * n_at, 3))
    step = max(1, 200000 // max(1, n_at))
    for i0 in range(0, n_img, step):
        ctx.check()
        blk = pts[i0:i0 + step]
        new[i0 * n_at:(i0 + len(blk)) * n_at] = ((blk[:, None, :] + frac[None, :, :]) @ inv).reshape(-1, 3)
        ctx.progress(0.6 + 0.3 * min(1.0, (i0 + len(blk)) / n_img), "building coordinates")
    per = np.array([bool(p) for p in pbc])
    w = new[:, per]
    w -= np.floor(w + 1e-12)
    w[np.abs(w - 1.0) < 1e-12] = 0.0
    new[:, per] = w
    out = result(m @ cell, new, np.tile(numbers, n_img), pbc, summary=f"{n_img * n_at} atoms ({n_img} × {n_at})")
    out["info"] = {"det": n_img, "matrix": m.tolist()}
    ctx.progress(0.9, "packing")
    return out


def symmetry_of(ctx=NOCTX, cell=None, frac=None, numbers=None, pbc=(True, True, True), symprec: float = 1e-3, **_):
    """spglib on a big built structure ("Compute now", a pool task)."""
    from . import structures
    ctx.progress(0.1, "spglib")
    sym = structures.symmetry(np.asarray(cell, float), np.asarray(frac, float), [int(z) for z in numbers], pbc, symprec=symprec)
    ctx.progress(1.0, "done")
    return {"symmetry": sym}


# ---------------------------------------------------------------- slab (T5)
PLANE_TOL = 0.15          # atoms within this height (Å) on the surface normal form one atomic plane


def check_hkl(hkl):
    try:
        h = [float(x) for x in hkl]
    except (TypeError, ValueError):
        raise ValueError("h k l are three whole numbers")
    if len(h) != 3 or any(abs(x - round(x)) > 1e-9 for x in h):
        raise ValueError("h k l are three whole numbers")
    h = tuple(int(round(x)) for x in h)
    if h == (0, 0, 0):
        raise ValueError("h k l can't all be zero")
    if max(abs(x) for x in h) > 12:
        raise ValueError("h k l are limited to ±12")
    return h


def _bulk(cell, frac, numbers):
    from ase import Atoms
    return Atoms(numbers=[int(z) for z in numbers], cell=np.asarray(cell, float).reshape(3, 3),
                 scaled_positions=np.asarray(frac, float).reshape(-1, 3), pbc=True)


def _planes(heights, tol=PLANE_TOL):
    order = np.argsort(heights, kind="stable")
    groups = []
    for i in order:
        if groups and heights[i] - heights[groups[-1][-1]] < tol:
            groups[-1].append(int(i))
        else:
            groups.append([int(i)])
    return groups


def _formula(numbers) -> str:
    """A cell's full formula in ase's "metal" order (mace.utils.formula)."""
    from ase.data import chemical_symbols
    from mace.utils.formula import count_symbols, format_formula
    return format_formula(count_symbols(chemical_symbols[int(z)] for z in numbers), "metal")


def _plane_name(formula: str) -> str:
    """A plane's composition as a reader names its termination: "Mg2O2" -> "MgO", "O4" -> "O"."""
    from mace.utils.formula import parse_formula, reduced_formula
    counts = parse_formula(formula)
    return reduced_formula(counts, "metal")[0] if counts else formula


def _surface_cells(bulk, hkl, n):
    import warnings
    from ase.build import surface
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return surface(bulk, hkl, n, vacuum=None)


def _plane_info(s):
    """The atomic planes of a surface cell stack, bottom to top: (normal,
    heights, groups, signature per plane: composition and the gap below)."""
    a, b = s.cell[0], s.cell[1]
    nrm = np.cross(a, b)
    nrm /= np.linalg.norm(nrm)
    h = s.positions @ nrm
    groups = _planes(h)
    mean = [float(np.mean(h[g])) for g in groups]
    sigs = []
    for k, g in enumerate(groups):
        gap = round(mean[k] - mean[k - 1], 2) if k else None
        sigs.append((_formula(s.numbers[g]), gap))
    return nrm, h, groups, mean, sigs


def slab_terminations(cell, frac, numbers, hkl):
    """The distinct terminations of the (hkl) surface: one per kind of top
    plane over one period of the plane stack (its composition, and the gap to
    the plane under it, which tells a Si(111) bilayer's two cuts apart)."""
    bulk = _bulk(cell, frac, numbers)
    one = _surface_cells(bulk, hkl, 1)
    period = len(_plane_info(one)[2])
    s = _surface_cells(bulk, hkl, 3)
    nrm, h, groups, mean, sigs = _plane_info(s)
    mid = list(range(period, 2 * period))            # the middle cell: each plane has its real neighbors
    seen, out = {}, []
    for k in reversed(mid):
        if sigs[k] in seen:
            continue
        seen[sigs[k]] = len(out)
        g = groups[k]
        inv = np.linalg.inv(np.array([s.cell[0], s.cell[1], nrm]).T)     # ase leaves c at zero without vacuum
        f = (inv @ s.positions[g].T).T
        out.append({"index": len(out), "formula": sigs[k][0], "name": _plane_name(sigs[k][0]), "gap_below": sigs[k][1],
                    "atoms_per_plane": len(g), "thumb": {"frac": np.round(f[:, :2] % 1.0, 4).tolist(),
                                                         "numbers": [int(z) for z in s.numbers[g]],
                                                         "ab": [s.cell[0][:2].tolist(), s.cell[1][:2].tolist()]}})
    return {"terminations": out, "period": period, "atoms_per_cell": len(one),
            "spacing": float(np.mean(np.diff(mean[period:2 * period + 1]))) if period else 0.0}


def slab(ctx=NOCTX, cell=None, frac=None, numbers=None, pbc=(True, True, True), hkl=(0, 0, 1), layers=5, vacuum=15.0,
         termination=0, symmetric=False, center=True, **_):
    """A slab of ``layers`` atomic planes cut from the bulk along (hkl), its
    top plane the chosen termination, with ``vacuum`` Å between periodic
    images along c (c normal to the slab). Symmetric: the fewest planes >=
    layers whose bottom plane matches the top one."""
    hkl = check_hkl(hkl)
    layers = int(layers)
    if not 1 <= layers <= 200:
        raise ValueError("layers: 1 to 200")
    vacuum = float(vacuum)
    if not 0 <= vacuum <= 200:
        raise ValueError("vacuum: 0 to 200 Å")
    term = slab_terminations(cell, frac, numbers, hkl)
    if not term["terminations"]:
        raise ValueError("no atomic planes along this direction")
    termination = int(termination) % len(term["terminations"])
    want = term["terminations"][termination]
    ctx.progress(0.2, "cutting")
    bulk = _bulk(cell, frac, numbers)
    period = term["period"]
    n_cells = int(np.ceil((layers + 2 * period + 2) / period)) + 1
    s = _surface_cells(bulk, hkl, n_cells)
    ctx.check()
    nrm, h, groups, mean, sigs = _plane_info(s)
    sig_want = (want["formula"], want["gap_below"])
    # the topmost plane of that kind with real planes under it (the stack's own top may be cut short)
    tops = [k for k in range(len(groups) - 1, -1, -1) if sigs[k] == sig_want and k >= layers + period]
    if not tops:
        raise ValueError("this termination does not fit; try more layers")
    t = tops[0]
    L = layers
    if symmetric:
        for L in range(layers, layers + period + 1):
            if sigs[t - L + 1][0] == sigs[t][0]:
                break
        else:
            L = layers
    keep = [i for g in groups[t - L + 1:t + 1] for i in g]
    pos = s.positions[keep]
    hk = pos @ nrm
    bottom, top = float(hk.min()), float(hk.max())
    span = top - bottom
    new_cell = np.array([s.cell[0], s.cell[1], nrm * (span + vacuum)])
    shift = (bottom - (vacuum / 2 if center else 0.0))
    cart = pos - np.outer(np.full(len(pos), shift), nrm)
    f = np.linalg.solve(new_cell.T, cart.T).T
    f[:, :2] -= np.floor(f[:, :2] + 1e-12)
    ctx.progress(0.9, "done")
    nums = s.numbers[keep]
    order = np.lexsort((f[:, 1], f[:, 0], np.round(f[:, 2], 6)))        # bottom plane first
    out = result(new_cell, f[order], nums[order], (True, True, True),
                 summary=f"{L} layers, {len(keep)} atoms, {span:.3f} Å thick")
    out["info"] = {"hkl": list(hkl), "layers": L, "asked_layers": layers, "thickness": span, "c": span + vacuum,
                   "vacuum": vacuum, "termination": termination, "terminations": term["terminations"],
                   "spacing": term["spacing"], "top": want["formula"], "top_name": want["name"]}
    return out


# ---------------------------------------------------------------- point defects (T6)
INTERSTITIAL_MAX = 2000      # atoms: the interstitial search stops here
GRID_STEP = 0.25             # Å between grid points of the empty-space search
MIN_EMPTY = 1.0              # Å: a candidate is at least this far from every atom
MERGE = 0.5                  # Å: maxima closer than this are one candidate


def _dataset(cell, frac, numbers, symprec=1e-3):
    import spglib
    from spglib_compat import attribute_dataset   # Crystal_d12 is on sys.path through structures.py
    return attribute_dataset(spglib.get_symmetry_dataset((np.asarray(cell, float), np.asarray(frac, float) % 1.0,
                                                          [int(z) for z in numbers]), symprec=symprec))


def defect_sites(ctx=NOCTX, cell=None, frac=None, numbers=None, symprec=1e-3, **_):
    """The symmetry-inequivalent atoms: {index (the representative), element,
    wyckoff, multiplicity (conventional cell), orbit (indices)}."""
    from ase.data import chemical_symbols
    from . import structures
    numbers = [int(z) for z in numbers]
    sym = structures.symmetry(cell, frac, numbers, symprec=symprec)
    out = []
    if not sym.get("available"):
        for i, z in enumerate(numbers):
            out.append({"index": i, "element": chemical_symbols[z % 100 or z], "wyckoff": "a", "multiplicity": 1, "orbit": [i]})
        return {"sites": out, "group": None}
    eq = sym["equivalent_atoms"]
    for rep in sorted(set(eq)):
        orbit = [i for i, e in enumerate(eq) if e == rep]
        out.append({"index": rep, "element": chemical_symbols[numbers[rep] % 100 or numbers[rep]],
                    "wyckoff": sym["wyckoffs"][rep], "multiplicity": int(sym["multiplicity"][rep]), "orbit": orbit})
    return {"sites": out, "group": {"number": sym["number"], "international": sym["international"]}}


def _nearest_distance(cell, frac, points_frac, chunk=20000, ctx=NOCTX):
    """For each point (fractional), the minimum-image distance to the nearest atom."""
    cell = np.asarray(cell, float)
    atoms = (np.asarray(frac, float) % 1.0) @ cell
    shifts = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], float) @ cell
    images = (atoms[None, :, :] + shifts[:, None, :]).reshape(-1, 3)
    pts = (np.asarray(points_frac, float) % 1.0) @ cell
    try:
        from scipy.spatial import cKDTree
        return cKDTree(images).query(pts)[0]
    except ImportError:
        out = np.empty(len(pts))
        for i0 in range(0, len(pts), chunk):
            ctx.check()
            p = pts[i0:i0 + chunk]
            d2 = ((p[:, None, :] - images[None, :, :]) ** 2).sum(-1)
            out[i0:i0 + chunk] = np.sqrt(d2.min(1))
        return out


def interstitial_candidates(ctx=NOCTX, cell=None, frac=None, numbers=None, symprec=1e-3, step=GRID_STEP,
                            min_dist=MIN_EMPTY, merge=MERGE, **_):
    """Empty sites: the local maxima of the distance to the nearest atom on a
    periodic grid (spacing <= ``step`` Å), at least ``min_dist`` from every
    atom, merged within ``merge`` Å, one per symmetry orbit, each with its
    Wyckoff letter and multiplicity."""
    cell = np.asarray(cell, float).reshape(3, 3)
    frac = np.asarray(frac, float).reshape(-1, 3)
    if len(frac) > INTERSTITIAL_MAX:
        raise ValueError(f"Interstitial search is limited to {INTERSTITIAL_MAX} atoms; pick a position instead")
    lengths = np.linalg.norm(cell, axis=1)
    # multiples of 12: the grid holds 1/2, 1/3, 1/4 and 1/6, where special positions sit
    n = np.maximum(12, (np.ceil(lengths / step / 12) * 12).astype(int))
    g = np.stack(np.meshgrid(*[np.arange(k) / k for k in n], indexing="ij"), -1).reshape(-1, 3)
    ctx.progress(0.1, "distances on the grid")
    d = _nearest_distance(cell, frac, g, ctx=ctx).reshape(*n)
    ctx.check()
    ctx.progress(0.6, "local maxima")
    is_max = d >= min_dist
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            for dk in (-1, 0, 1):
                if di or dj or dk:
                    is_max &= d >= np.roll(d, (di, dj, dk), axis=(0, 1, 2)) - 1e-9
    idx = np.argwhere(is_max)
    cand = [(float(d[tuple(i)]), (i / n)) for i in idx]
    cand.sort(key=lambda x: -x[0])
    merged = []
    for dist, f in cand:                          # strongest first; neighbors within `merge` join it
        if any(np.linalg.norm(((f - m[1] + 0.5) % 1.0 - 0.5) @ cell) < merge for m in merged):
            continue
        merged.append((dist, f))
        if len(merged) > 200:
            break
    ctx.progress(0.8, "symmetry")
    ds = _dataset(cell, frac, numbers, symprec)
    if ds is not None:
        # each maximum onto its site's symmetry (the mean of its images under
        # the operations that keep it within 0.3 Å): 8c at 1/4 1/4 1/4, not a grid point near it
        snapped = []
        for dist, f in merged:
            g = np.asarray(f, float)
            for _ in range(3):
                imgs = np.einsum("kij,j->ki", np.asarray(ds.rotations), g) + np.asarray(ds.translations)
                diff = (imgs - g + 0.5) % 1.0 - 0.5
                near = np.linalg.norm(diff @ cell, axis=1) < 0.3
                g = (g + diff[near].mean(0)) % 1.0
            snapped.append((float(_nearest_distance(cell, frac, g[None, :])[0]), g))
        merged = snapped
    out, seen = [], []
    for dist, f in merged:
        if ds is not None:
            orbit = (np.einsum("kij,j->ki", np.asarray(ds.rotations), f) + np.asarray(ds.translations)) % 1.0
        else:
            orbit = f[None, :] % 1.0
        if any(min(np.linalg.norm(((o - s + 0.5) % 1.0 - 0.5) @ cell) for o in orbit) < merge for s in seen):
            continue
        uniq = []
        for o in orbit:
            if not any(np.linalg.norm(((o - u + 0.5) % 1.0 - 0.5) @ cell) < 1e-3 for u in uniq):
                uniq.append(o)
        seen.append(f)
        wy, mult = _wyckoff_of(cell, frac, numbers, np.array(uniq), symprec)
        out.append({"frac": [float(x) for x in f], "distance": dist, "wyckoff": wy, "multiplicity": mult or len(uniq),
                    "orbit_size": len(uniq)})
    ctx.progress(1.0, "done")
    return {"candidates": out, "grid": [int(x) for x in n]}


def _wyckoff_of(cell, frac, numbers, orbit, symprec):
    """The Wyckoff letter of an empty site: spglib on the structure plus
    dummy atoms on the site's whole orbit (so the group is the parent's)."""
    try:
        numbers = [int(z) for z in numbers]
        dummy = max(numbers) + 1
        orbit = np.asarray(orbit, float).reshape(-1, 3)
        from . import structures
        sym = structures.symmetry(cell, np.vstack([frac, orbit]), numbers + [dummy] * len(orbit), symprec=symprec)
        if not sym.get("available"):
            return "?", None
        return sym["wyckoffs"][-1], int(sym["multiplicity"][-1])
    except Exception:
        return "?", None


def _shortest(basis) -> float:
    """The shortest nonzero lattice vector of a basis (Niggli-reduced first;
    integer combinations within ±3 then hold it)."""
    import spglib
    b = np.asarray(basis, float)
    red = spglib.niggli_reduce(b)
    r = np.asarray(red if red is not None else b, float)
    k = np.arange(-3, 4)
    g = np.stack(np.meshgrid(k, k, k, indexing="ij"), -1).reshape(-1, 3)
    g = g[np.any(g != 0, axis=1)]
    return float(np.linalg.norm(g @ r, axis=1).min())


def defect_supercell(cell=None, n_atoms=1, min_dist=10.0, search_matrices=False, ctx=NOCTX, **_):
    """The smallest supercell whose defect images are at least ``min_dist``
    apart (its shortest lattice vector): diagonal n1 n2 n3 from 1 to
    ceil(d / h_i) (h_i: the cell's height across face i), the fewest atoms
    first, then the most isotropic; with ``search_matrices``, integer
    matrices with |m_ij| <= 3 too, taken only when they need fewer atoms."""
    cell = np.asarray(cell, float).reshape(3, 3)
    vol = abs(np.linalg.det(cell))
    heights = [vol / np.linalg.norm(np.cross(cell[(i + 1) % 3], cell[(i + 2) % 3])) for i in range(3)]
    top = [max(1, int(np.ceil(min_dist / h))) for h in heights]
    best = None
    for n1 in range(1, top[0] + 1):
        for n2 in range(1, top[1] + 1):
            for n3 in range(1, top[2] + 1):
                m = np.diag([n1, n2, n3])
                det = n1 * n2 * n3
                if best is not None and det > best[0]:
                    continue
                s = _shortest(m @ cell)
                if s < min_dist - 1e-9:
                    continue
                lens = np.linalg.norm(m @ cell, axis=1)
                iso = float(lens.max() / lens.min())
                if best is None or (det, iso) < (best[0], best[1]):
                    best = (det, iso, m, s)
    if best is None:                               # the ceil bound always suffices; a guard
        m = np.diag(top)
        best = (int(np.prod(top)), 0.0, m, _shortest(m @ cell))
    if search_matrices and best[0] > 1:
        # integer matrices with |m_ij| <= 2 (5^9 of them, vectorized in chunks):
        # those with fewer atoms whose three rows are already long enough, then
        # the exact shortest vector of the fewest-atom ones first
        import itertools
        vals = np.arange(-2, 3)
        rows = np.array(list(itertools.product(vals, repeat=3)))                       # 125 candidate rows
        rl = np.linalg.norm(rows @ cell, axis=1)
        rows = rows[rl >= min_dist - 1e-9]
        combos = np.array([c for c in itertools.product((-1, 0, 1), repeat=3) if any(c)], float)
        found = []
        bc = np.array(list(itertools.product(range(len(rows)), repeat=2)))
        for i0 in range(len(rows)):
            ctx.check()
            if i0 % 8 == 0:
                ctx.progress(0.9 * i0 / max(1, len(rows)), "searching matrices")
            m = np.stack([np.broadcast_to(rows[i0], (len(bc), 3)), rows[bc[:, 0]], rows[bc[:, 1]]], 1)
            det = np.rint(np.linalg.det(m.astype(float))).astype(int)
            ok = (det > 0) & (det < best[0])
            if not ok.any():
                continue
            m, det = m[ok], det[ok]
            # every short combination of the three rows is a lattice vector too: all must reach d
            vec = np.einsum("cr,mrk->mck", combos, (m @ cell))
            ok = np.linalg.norm(vec, axis=2).min(1) >= min_dist - 1e-9
            for mm, dd in zip(m[ok], det[ok]):
                found.append((int(dd), mm))
        for dd, mm in sorted(found, key=lambda x: x[0]):
            if dd >= best[0]:
                break
            s = _shortest(mm @ cell)
            if s >= min_dist - 1e-9:
                lens = np.linalg.norm(mm @ cell, axis=1)
                best = (dd, float(lens.max() / lens.min()), mm, s)
                break
    det, iso, m, s = best
    return {"matrix": np.asarray(m, int).tolist(), "det": int(det), "atoms": int(det * n_atoms), "nearest": float(s),
            "diagonal": bool(np.count_nonzero(np.asarray(m) - np.diag(np.diag(m))) == 0)}


def make_defect(ctx=NOCTX, cell=None, frac=None, numbers=None, pbc=(True, True, True), kind="vacancy", site=0,
                element=None, position=None, matrix=None, **_):
    """One point defect in a supercell (matrix, the identity for none):
    vacancy (the site's atom removed), substitution or antisite (its element
    changed to ``element``), interstitial (an ``element`` atom added at
    ``position``, fractional in the source cell)."""
    from ase.data import atomic_numbers
    cell = np.asarray(cell, float).reshape(3, 3)
    frac = np.asarray(frac, float).reshape(-1, 3)
    numbers = np.asarray(numbers, dtype=np.int64)
    m = check_matrix(matrix if matrix is not None else np.eye(3, dtype=int))
    sc = supercell(ctx, cell, frac, numbers, pbc, m.tolist())
    new_cell, f, z = sc["cell"], sc["frac"], sc["numbers"].copy()
    inv = np.linalg.inv(m.astype(float))
    pts = lattice_points(m)
    image0 = int(np.argmin(np.abs(pts).sum(1)))              # the image nearest the origin
    n_at = len(frac)
    nearest = _shortest(new_cell)
    if kind in ("vacancy", "substitution", "antisite"):
        site = int(site)
        if not 0 <= site < n_at:
            raise ValueError(f"no atom {site + 1}")
        k = image0 * n_at + site
        if kind == "vacancy":
            keep = np.ones(len(z), bool)
            keep[k] = False
            f, z = f[keep], z[keep]
            where = (frac[site] + pts[image0]) @ inv @ new_cell
        else:
            zz = atomic_numbers.get(str(element), 0) if not str(element).isdigit() else int(element)
            if not 1 <= zz <= 103:
                raise ValueError(f"{element!r} is not an element")
            if zz == (z[k] % 100 or z[k]):
                raise ValueError("Pick a different element")
            z[k] = zz
            where = f[k] @ new_cell
    elif kind == "interstitial":
        zz = atomic_numbers.get(str(element), 0)
        if not 1 <= zz <= 103:
            raise ValueError(f"{element!r} is not an element")
        try:
            p = np.asarray(position, float).reshape(3)
        except (TypeError, ValueError):
            raise ValueError("an interstitial needs a position (fractional x y z)")
        q = ((p + pts[image0]) @ inv) % 1.0
        d = _nearest_distance(new_cell, f, q[None, :])[0]
        if d < 0.5:
            raise ValueError(f"an atom already sits there ({d:.2f} Å away)")
        f = np.vstack([f, q])
        z = np.append(z, zz)
        where = q @ new_cell
    else:
        raise ValueError("kind is vacancy, substitution, interstitial or antisite")
    out = result(new_cell, f, z, pbc, summary=f"{kind} in a {len(z)}-atom cell")
    out["info"] = {"kind": kind, "matrix": m.tolist(), "det": int(len(pts)), "nearest_image": nearest,
                   "where": [float(x) for x in where]}
    return out


def defect_info(ctx=NOCTX, cell=None, frac=None, numbers=None, symprec=1e-3, min_dist=10.0, search_matrices=False,
                interstitial=False, **_):
    """The pool's half of /api/defect/info: the supercell search (with
    matrices) and the empty-site search on a big cell."""
    out = {"supercell": defect_supercell(cell, len(numbers), min_dist, search_matrices, ctx=ctx)}
    if interstitial:
        out.update(interstitial_candidates(ctx, cell, frac, numbers, symprec))
    return out


# ---------------------------------------------------------------- transform cell, symmetrize (T7)
SWEEP = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1]      # 13 stops, Å


def standardize(ctx=NOCTX, cell=None, frac=None, numbers=None, pbc=(True, True, True), kind="primitive", symprec=1e-3, **_):
    """Transform Cell: spglib's primitive or conventional standard cell, or
    the Niggli-reduced cell with the atoms mapped into it."""
    import spglib
    cell = np.asarray(cell, float).reshape(3, 3)
    frac = np.asarray(frac, float).reshape(-1, 3)
    numbers = [int(z) for z in numbers]
    if kind in ("primitive", "conventional"):
        res = spglib.standardize_cell((cell, frac % 1.0, numbers), to_primitive=kind == "primitive", no_idealize=False,
                                      symprec=symprec)
        if res is None:
            raise ValueError(f"Couldn't standardize this cell at {symprec:g} Å")
        lat, pos, nums = res
        out = result(lat, pos, nums, pbc, summary=f"{len(numbers)} → {len(nums)} atoms")
    elif kind == "niggli":
        red = spglib.niggli_reduce(cell)
        if red is None:
            raise ValueError("spglib could not Niggli-reduce this cell")
        red = np.asarray(red, float)
        f = (frac @ cell) @ np.linalg.inv(red)
        f -= np.floor(f + 1e-12)
        out = result(red, f, numbers, pbc, summary=f"{len(numbers)} atoms, Niggli-reduced cell")
    else:
        raise ValueError("kind is primitive, conventional or niggli")
    out["info"] = {"kind": kind, "before": len(numbers), "after": len(out["numbers"]),
                   "a_before": float(np.linalg.norm(cell[0])), "a_after": float(np.linalg.norm(out["cell"][0]))}
    return out


def _group(cell, frac, numbers, symprec):
    ds = _dataset(cell, frac, numbers, symprec)
    return (None, None) if ds is None else (int(ds.number), str(ds.international))


def symmetrize_sweep(ctx=NOCTX, cell=None, frac=None, numbers=None, stops=None, **_):
    """The space group spglib finds at each tolerance (1e-5 to 0.1 Å)."""
    stops = list(stops or SWEEP)
    out = []
    for k, tol in enumerate(stops):
        ctx.check()
        ctx.progress(k / len(stops), f"{tol:g} Å")
        num, sym = _group(cell, frac, numbers, tol)
        out.append({"symprec": tol, "number": num, "international": sym,
                    "change": bool(out) and out[-1]["number"] != num})
    ctx.progress(1.0, "done")
    return {"sweep": out}


def symmetrize(ctx=NOCTX, cell=None, frac=None, numbers=None, pbc=(True, True, True), symprec=1e-3, **_):
    """Symmetrize: spglib's refined (conventional, idealized) cell at
    ``symprec``, as the editor's op_symmetrize makes it, and how far each atom
    moved: every original atom matched, by spglib's own transformation, to
    the nearest refined atom of its element (minimum image)."""
    import spglib
    cell = np.asarray(cell, float).reshape(3, 3)
    frac = np.asarray(frac, float).reshape(-1, 3)
    numbers = [int(z) for z in numbers]
    ds = _dataset(cell, frac, numbers, symprec)
    if ds is None:
        raise ValueError(f"spglib found no symmetry at {symprec:g} Å; try a larger tolerance")
    res = spglib.standardize_cell((cell, frac % 1.0, numbers), to_primitive=False, no_idealize=False, symprec=symprec)
    if res is None:
        raise ValueError(f"Couldn't standardize this cell at {symprec:g} Å")
    lat, pos, nums = (np.asarray(x) for x in res)
    P, p = np.asarray(ds.transformation_matrix, float), np.asarray(ds.origin_shift, float)
    A_s = np.linalg.inv(P).T @ cell                      # the standard cell before idealizing
    xs = (frac @ P.T + p) % 1.0                          # each original atom in its fractional coordinates
    disp = np.empty(len(numbers))
    vec = np.zeros((len(numbers), 3))
    for i, (x, z) in enumerate(zip(xs, numbers)):
        same = pos[nums == z]
        if not len(same):
            disp[i] = 0.0
            continue
        d = ((same - x + 0.5) % 1.0 - 0.5) @ A_s           # Cartesian, in the original frame
        j = int(np.argmin(np.linalg.norm(d, axis=1)))
        vec[i] = d[j]
        disp[i] = float(np.linalg.norm(d[j]))
    out = result(lat, pos, nums, pbc, summary=f"{int(ds.number)} {ds.international}")
    out["info"] = {"number": int(ds.number), "international": str(ds.international), "symprec": symprec,
                   "max_shift": float(disp.max()) if len(disp) else 0.0,
                   "rms_shift": float(np.sqrt((disp ** 2).mean())) if len(disp) else 0.0,
                   "before": len(numbers), "after": int(len(nums)), "vectors": np.round(vec, 6).tolist()}
    return out


# ---------------------------------------------------------------- Clean geometry in the pool (T8)
def clean_job(ctx=NOCTX, cell=None, frac=None, numbers=None, pbc=(True, True, True), keep_symmetry=True,
              relax_cell=False, symprec=1e-3, time_limit=10.0, max_steps=2000, charge=0, uhf=0, method="springs",
              symmetrize_tol=None, fmax=None, threads=1, mlip_lease=None, **_):
    """Clean geometry in a worker, through ``mace.preopt.core.preoptimize``
    (the CLI's ``mace preopt`` runs the same pipeline). For the springs the
    one hook (once per FIRE step) is, as before, the cancel check and the
    progress (steps or elapsed time, whichever is further along); a GFN-FF or
    ML run polls the pure cancel check and reports progress separately. The
    document's charge (and spin) travel with the job and are recorded in the
    report: the spring model ignores them, the other tiers need them."""
    from ase import Atoms
    from mace.preopt import core
    atoms = Atoms(numbers=[int(z) for z in numbers], cell=np.asarray(cell, float), scaled_positions=np.asarray(frac, float), pbc=True)
    springs = method == "springs"
    r = core.preoptimize(atoms, method, charge=int(charge), uhf=int(uhf), keep_symmetry=bool(keep_symmetry),
                         symprec=max(float(symprec), 1e-5), symmetrize_tol=symmetrize_tol, relax_cell=bool(relax_cell),
                         fmax=fmax, steps=int(max_steps) if (springs or max_steps) else None,
                         time_limit=float(time_limit), threads=int(threads), canceled=lambda: ctx.canceled,
                         progress=lambda f, m: ctx.progress(f, m), mlip_lease=mlip_lease, route="gui")
    ctx.check()
    a = r.atoms
    out = result(a.cell.array, a.get_scaled_positions(wrap=False), a.numbers, pbc, summary="clean")
    out["report"] = {k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in (r.report or {}).items()}
    out["report"]["charge"], out["report"]["uhf"] = int(charge), int(uhf)
    import os
    out["report"]["worker_pid"], out["report"]["worker_nice"] = os.getpid(), os.nice(0)
    out["preopt"] = {"method_requested": r.method_requested, "method_ran": r.method_ran, "label": r.label_ran,
                     "attempts": r.attempts, "fell_back": r.fell_back, "summary": r.summary,
                     "warnings": r.warnings, "notes": r.notes, "provenance": core.provenance(r),
                     "citation": core.citation_text(r), "tier": {k: v for k, v in (r.tier_info or {}).items()
                                                                 if isinstance(v, (str, int, float, list, dict))},
                     "symmetrize": core.provenance(r)["symmetrize"], "nothing_to_clean": r.nothing_to_clean,
                     "params": {k: v for k, v in r.params.items() if isinstance(v, (str, int, float, bool, type(None)))}}
    return out


# ---------------------------------------------------------------- file details (plan T11)
def file_details(ctx=None, paths=(), **_):
    """The explorer's secondary text for structure files, read in a pool
    worker: {path: {formula, n, sg, kind}} or {path: {error}}. A CRYSTAL
    output gives its last frame, through the byte-offset index."""
    from . import filekind, outindex, structures as S
    out = {}
    paths = list(paths)
    for k, p in enumerate(paths):
        if ctx is not None:
            ctx.check()
            ctx.progress((k + 1) / max(1, len(paths)), f"{k + 1} of {len(paths)} files")
        try:
            path = Path(p)
            if filekind.is_refinable(path.name):
                # what the file is, by content (plan E1): a job log or a properties output has no geometry
                info = filekind.refine(path)
                if info["role"] != "crystal_out":
                    out[p] = {**info, "kind": info["role"]}
                    continue
                idx = outindex.build(path)
                if not idx.n_steps:
                    out[p] = {**info, "kind": "freq" if idx.freq else "out", "nogeom": True}
                    continue
                if idx.dim == 0:
                    info["dim"] = 0
                data = idx.payload(-1, {"kind": "out", "path": str(path)})
                sym = data.get("symmetry") or {}
                out[p] = {"formula": data.get("reduced_formula") or data.get("formula"), "n": len(data["numbers"]),
                          "sg": sym.get("international") if sym.get("available") else None,
                          "sg_number": sym.get("number") if sym.get("available") else None,
                          "kind": "freq" if idx.freq else "out", **info}
                continue
            else:
                data = S.read_structure(path)
                kind = path.suffix.lower().lstrip(".")
            sym = data.get("symmetry") or {}
            out[p] = {"formula": data.get("reduced_formula") or data.get("formula"), "n": len(data["numbers"]),
                      "sg": sym.get("international") if sym.get("available") else None,
                      "sg_number": sym.get("number") if sym.get("available") else None, "kind": kind}
        except Exception as exc:          # one bad file never spoils the batch
            out[p] = {"error": (str(exc).splitlines() or [type(exc).__name__])[0][:200]}
    return out
