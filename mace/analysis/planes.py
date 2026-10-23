"""Miller planes and cleavage (VESTA's lattice planes, Materials Studio's
Cleave): the polygon where the plane (hkl) at a given position cuts the
unit cell (to draw it), its spacing, the atoms lying on it, and a
broken-bond estimate of where a crystal cleaves.

Cleavage model: for a plane family (hkl) at offset c (planes h.x = c + n),
count the bonds that cross a plane, per unit plane area (bonds/nm^2), over
one period of c; the minimum is the best cut for that family, and families
with fewer broken bonds cleave more easily. This is the bond-breaking
picture only: no relaxation, bond strengths or polarity.
"""
from __future__ import annotations

import itertools
import math
from typing import Dict, List, Sequence

import numpy as np

from .structure import Structure

_EDGES = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]


def d_spacing(cell: np.ndarray, hkl: Sequence[int]) -> float:
    g = np.asarray(hkl, float) @ np.linalg.inv(np.asarray(cell, float)).T
    n = np.linalg.norm(g)
    if n == 0:
        raise ValueError("(000) is not a plane")
    return float(1.0 / n)


def polygon(cell: np.ndarray, hkl: Sequence[int], c: float, lo=(0, 0, 0), hi=(1, 1, 1)) -> List[List[float]]:
    """Corners (Cartesian) of the plane h.x = c inside the box lo..hi (fractional)."""
    h = np.asarray(hkl, float)
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    corners = np.array([[lo[0] if i & 1 == 0 else hi[0], lo[1] if i & 2 == 0 else hi[1], lo[2] if i & 4 == 0 else hi[2]]
                        for i in range(8)])
    val = corners @ h - c
    pts = []
    for a, b in _EDGES:
        va, vb = val[a], val[b]
        if abs(va) < 1e-12:
            pts.append(corners[a])
        if (va < 0 < vb) or (vb < 0 < va):
            t = va / (va - vb)
            pts.append(corners[a] + t * (corners[b] - corners[a]))
    if len(pts) < 3:
        return []
    P = np.unique(np.round(np.array(pts), 10), axis=0) @ np.asarray(cell, float)
    if len(P) < 3:
        return []
    cen = P.mean(0)
    n = h @ np.linalg.inv(np.asarray(cell, float)).T
    n /= np.linalg.norm(n)
    e1 = P[0] - cen
    e1 -= (e1 @ n) * n
    e1 /= max(np.linalg.norm(e1), 1e-12)
    e2 = np.cross(n, e1)
    ang = np.arctan2((P - cen) @ e2, (P - cen) @ e1)
    return P[np.argsort(ang)].tolist()


def planes_in_box(cell, hkl, offset: float = 0.0, lo=(0, 0, 0), hi=(1, 1, 1)) -> List[Dict]:
    """Every plane h.x = offset + n crossing the box, as polygons."""
    h = np.asarray(hkl, float)
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    corners = np.array(list(itertools.product(*zip(lo, hi))))
    vals = corners @ h
    out = []
    for n in range(int(math.floor(vals.min() - offset)), int(math.ceil(vals.max() - offset)) + 1):
        poly = polygon(cell, hkl, offset + n, lo, hi)
        if poly:
            out.append({"n": n, "polygon": poly})
    return out


def atoms_on_plane(st: Structure, hkl, c: float, tol: float = 0.3) -> List[int]:
    """Atoms within tol (Angstrom) of any plane h.x = c + n."""
    d = d_spacing(st.cell, hkl)
    v = st.wrapped().frac @ np.asarray(hkl, float) - c
    dist = np.abs(v - np.round(v)) * d
    return [int(i) for i in np.flatnonzero(dist <= tol)]


def broken_bonds(st: Structure, hkl, steps: int = 200, tolerance: float = 1.15) -> Dict:
    """Bonds cut per nm^2 as the plane moves through one period."""
    from .topology import bonds
    w = st.wrapped()
    h = np.asarray(hkl, float)
    d = d_spacing(w.cell, hkl)
    B = bonds(w, tolerance)
    if not B:
        return {"hkl": list(hkl), "d": d, "min_per_nm2": 0.0, "best_offset": 0.0, "profile": []}
    a = np.array([w.frac[i] @ h for i, _, _ in B])
    b = np.array([(w.frac[j] + np.array(t)) @ h for _, j, t in B])
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    area_per_cell = w.volume / d                 # plane area per unit cell volume, summed over the family
    cs = (np.arange(steps) + 0.5) / steps
    prof = []
    for c in cs:
        # integers m with lo < c + m <= hi
        n = np.floor(hi - c) - np.ceil(lo - c + 1e-12) + 1
        prof.append(float(np.clip(n, 0, None).sum()) / area_per_cell * 100.0)
    prof = np.array(prof)
    k = int(np.argmin(prof))
    return {"hkl": [int(x) for x in hkl], "d": d, "min_per_nm2": float(prof[k]), "best_offset": float(cs[k]),
            "max_per_nm2": float(prof.max()), "profile": prof.tolist(), "offsets": cs.tolist()}


def cleavage(st: Structure, max_index: int = 2, tolerance: float = 1.15, steps: int = 120) -> Dict:
    """Low-index families ranked by the fewest bonds broken per area."""
    st.require_3d("Cleavage ranking")
    from .morphology import family_members
    from .refine import symmetry_ops
    from .pxrd import conventional
    st = conventional(st)
    rot, _, _ = symmetry_ops(st)
    seen = set()
    rows = []
    r = range(-max_index, max_index + 1)
    for h in itertools.product(r, r, r):
        if not any(h) or math.gcd(math.gcd(abs(h[0]), abs(h[1])), abs(h[2])) != 1:
            continue
        key = max(family_members(h, rot))
        if key in seen:
            continue
        seen.add(key)
        bb = broken_bonds(st, key, steps, tolerance)
        bb.pop("profile")
        bb.pop("offsets")
        rows.append(bb)
    rows.sort(key=lambda x: (x["min_per_nm2"], -x["d"]))
    return {"families": rows, "model": "bonds broken per area (no relaxation)", "cell": st.cell.tolist()}
