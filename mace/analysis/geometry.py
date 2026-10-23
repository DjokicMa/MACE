"""Bond-length and bond-angle distributions (Materials Studio / VESTA
style): every bond by element pair and every angle X-A-Y around each atom,
with statistics and histograms. Bonds are covalent-radius bonds
(topology.bonds) and follow the structure's periodicity.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Optional

import numpy as np

from .structure import Structure


def _hist(values, bins, lo=None, hi=None):
    v = np.asarray(values, float)
    if not len(v):
        return {"edges": [], "counts": []}
    lo = float(v.min()) if lo is None else lo
    hi = float(v.max()) if hi is None else hi
    if hi - lo < 1e-6:
        lo, hi = lo - 0.05, hi + 0.05
    c, e = np.histogram(v, bins=bins, range=(lo, hi))
    return {"edges": e.tolist(), "counts": c.tolist()}


def _stats(v):
    v = np.asarray(v, float)
    return {"n": int(len(v)), "min": float(v.min()), "max": float(v.max()), "mean": float(v.mean()),
            "std": float(v.std())}


def analyze(st: Structure, tolerance: float = 1.15, bins: int = 40, metal_metal: Optional[bool] = None,
            max_angles: int = 200000) -> Dict:
    from .topology import bonds
    w = st.wrapped()
    sym = w.symbols
    nb = defaultdict(list)
    pair_len = defaultdict(list)
    allb = []
    for a, b, t in bonds(w, tolerance, metal_metal):
        v = (w.frac[b] + np.array(t) - w.frac[a]) @ w.cell
        d = float(np.linalg.norm(v))
        key = "-".join(sorted((sym[a], sym[b])))
        pair_len[key].append(d)
        allb.append(d)
        nb[a].append(v)
        nb[b].append(-v)
    ang = defaultdict(list)
    alla = []
    nb_sym = defaultdict(list)
    for a, b, t in bonds(w, tolerance, metal_metal):
        nb_sym[a].append(sym[b])
        nb_sym[b].append(sym[a])
    count = 0
    for i, vecs in nb.items():
        if len(vecs) < 2:
            continue
        V = np.array(vecs)
        U = V / np.linalg.norm(V, axis=1)[:, None]
        names = nb_sym[i]
        for p in range(len(V)):
            for q in range(p + 1, len(V)):
                x = float(np.degrees(np.arccos(np.clip(U[p] @ U[q], -1, 1))))
                ends = sorted((names[p], names[q]))
                ang[f"{ends[0]}-{sym[i]}-{ends[1]}"].append(x)
                alla.append(x)
                count += 1
                if count >= max_angles:
                    break
    out = {"bonds": {}, "angles": {}, "tolerance": tolerance, "n_bonds": len(allb), "n_angles": len(alla)}
    if allb:
        out["bond_histogram"] = _hist(allb, bins)
        out["bond_stats"] = _stats(allb)
    if alla:
        out["angle_histogram"] = _hist(alla, bins * 2, 0.0, 180.0)
        out["angle_stats"] = _stats(alla)
    for k, v in sorted(pair_len.items()):
        out["bonds"][k] = {**_stats(v), "histogram": _hist(v, bins)}
    for k, v in sorted(ang.items()):
        out["angles"][k] = {**_stats(v), "histogram": _hist(v, bins * 2, 0.0, 180.0)}
    return out


def rdf(st: Structure, r_max: float = 10.0, dr: float = 0.05, pairs: Optional[str] = None) -> Dict:
    """Radial distribution g(r) of a 3D crystal (all atoms, or one A-B pair)."""
    st.require_3d("g(r)")
    w = st.wrapped()
    ii, jj, _, d = w.neighbor_pairs(r_max)
    sym = w.symbols
    if pairs:
        a, b = pairs.split("-")
        m = np.array([{sym[x], sym[y]} == {a, b} for x, y in zip(ii, jj)], bool)
        d = d[m]
        na = sum(1 for s in sym if s == a)
        nbb = sum(1 for s in sym if s == b)
        norm_pairs = na * nbb * (1 if a != b else 1)
    else:
        norm_pairs = w.n * w.n
    edges = np.arange(0, r_max + dr, dr)
    c, _ = np.histogram(d, bins=edges)
    r = 0.5 * (edges[1:] + edges[:-1])
    shell = 4 * np.pi * r * r * dr
    g = c * w.volume / (norm_pairs * shell)
    return {"r": r.tolist(), "g": g.tolist(), "pair": pairs or "all"}
