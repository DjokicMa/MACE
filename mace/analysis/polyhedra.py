"""Coordination polyhedra as VESTA reports them: for each polyhedron center,
the coordination number, bond lengths, polyhedral volume, Baur's
distortion index, the quadratic elongation and bond-angle variance
(Robinson et al., Science 172, 567 (1971)), Hoppe's effective coordination
number (ECoN), and the closest ideal shape by continuous shape measure
(CShM, 0 = ideal; Pinsky and Avnir, Inorg. Chem. 37, 5575 (1998)).

Which atoms are centers follows the viewer's Polyhedral style rule
(mace.gui.structures.polyhedron_centers: cations with 4-12 anion ligands,
not a perovskite's A site), so the table and the drawing list the same
polyhedra; their vertices are those ligands.
Bonds and neighbors follow the structure's periodicity.
"""
from __future__ import annotations

import itertools
import math
from typing import Dict, List, Optional, Sequence

import numpy as np

from .structure import Structure


def _ideal_shapes() -> Dict[int, Dict[str, np.ndarray]]:
    s3 = math.sqrt(3)
    phi = (1 + 5 ** 0.5) / 2
    tet = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
    octa = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
    tp = lambda h: np.array([[1, 0, h], [-0.5, s3 / 2, h], [-0.5, -s3 / 2, h], [1, 0, -h], [-0.5, s3 / 2, -h],
                             [-0.5, -s3 / 2, -h]], float)
    sq = np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]], float)
    pent = np.array([[math.cos(2 * math.pi * k / 5), math.sin(2 * math.pi * k / 5), 0] for k in range(5)])
    cube = np.array(list(itertools.product((-1, 1), repeat=3)), float)
    sapr = np.array([[math.cos(math.pi * k / 2), math.sin(math.pi * k / 2), 0.6] for k in range(4)] +
                    [[math.cos(math.pi * k / 2 + math.pi / 4), math.sin(math.pi * k / 2 + math.pi / 4), -0.6] for k in range(4)])
    cuboct = np.array([p for p in itertools.product((-1, 0, 1), repeat=3) if sorted(map(abs, p)) == [0, 1, 1]], float)
    ico = np.array([[0, a, b * phi] for a in (-1, 1) for b in (-1, 1)] + [[a, b * phi, 0] for a in (-1, 1) for b in (-1, 1)] +
                   [[b * phi, 0, a] for a in (-1, 1) for b in (-1, 1)], float)
    hcp = np.array([[math.cos(math.pi * k / 3), math.sin(math.pi * k / 3), 0] for k in range(6)] +
                   [[math.cos(math.pi * k / 3 + math.pi / 6) / s3 * 1.0, math.sin(math.pi * k / 3 + math.pi / 6) / s3, z]
                    for k in (0, 2, 4) for z in (0.8165,)] +
                   [[math.cos(math.pi * k / 3 + math.pi / 6) / s3, math.sin(math.pi * k / 3 + math.pi / 6) / s3, -0.8165]
                    for k in (0, 2, 4)], float)
    tricap = np.vstack([tp(0.7), [[-1.2, 0, 0], [0.6, 1.2 * s3 / 2, 0], [0.6, -1.2 * s3 / 2, 0]]])
    return {
        2: {"linear": np.array([[0, 0, 1], [0, 0, -1]], float)},
        3: {"trigonal planar": np.array([[1, 0, 0], [-0.5, s3 / 2, 0], [-0.5, -s3 / 2, 0]]),
            "trigonal pyramid": np.array([[1, 0, -0.35], [-0.5, s3 / 2, -0.35], [-0.5, -s3 / 2, -0.35]]),
            "T-shape": np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0]], float)},
        4: {"tetrahedron": tet, "square planar": sq,
            "seesaw": np.array([[0, 0, 1], [0, 0, -1], [1, 0, 0], [-0.5, s3 / 2, 0]], float)},
        5: {"trigonal bipyramid": np.vstack([[[0, 0, 1], [0, 0, -1]], [[1, 0, 0], [-0.5, s3 / 2, 0], [-0.5, -s3 / 2, 0]]]),
            "square pyramid": np.vstack([sq + [0, 0, -0.2], [[0, 0, 1]]])},
        6: {"octahedron": octa, "trigonal prism": tp(0.8165)},
        7: {"pentagonal bipyramid": np.vstack([pent, [[0, 0, 1], [0, 0, -1]]]),
            "capped octahedron": np.vstack([octa, [[0.577, 0.577, 0.577]]]),
            "capped trigonal prism": np.vstack([tp(0.8165), [[-1.0, 0, 0]]])},
        8: {"cube": cube / s3, "square antiprism": sapr},
        9: {"tricapped trigonal prism": tricap},
        12: {"cuboctahedron": cuboct, "icosahedron": ico, "anticuboctahedron": hcp},
    }


_SHAPES = None


def _norm(P):
    P = np.asarray(P, float)
    P = P - P.mean(0)
    return P / math.sqrt((P * P).sum() / len(P))


def _rand_rot(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    a, b, c, d = q
    return np.array([[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                     [2 * (b * c + a * d), a * a - b * b + c * c - d * d, 2 * (c * d - a * b)],
                     [2 * (b * d - a * c), 2 * (c * d + a * b), a * a - b * b - c * c + d * d]])


def shape_measure(Q: np.ndarray, P: np.ndarray, tries: int = 24, seed: int = 0) -> float:
    """Continuous shape measure of points Q against ideal P (same count)."""
    from scipy.optimize import linear_sum_assignment
    Qn, Pn = _norm(Q), _norm(P)
    rng = np.random.default_rng(seed)
    best = np.inf
    for k in range(tries):
        R = np.eye(3) if k == 0 else _rand_rot(rng)
        Pr = Pn @ R.T
        for _ in range(8):
            C = ((Qn[:, None, :] - Pr[None, :, :]) ** 2).sum(-1)
            r, c = linear_sum_assignment(C)
            A = Pn[c]
            H = A.T @ Qn
            U, S, Vt = np.linalg.svd(H)
            dd = np.sign(np.linalg.det(Vt.T @ U.T))
            Rk = Vt.T @ np.diag([1, 1, dd]) @ U.T
            Pr_new = Pn @ Rk.T
            if np.allclose(Pr_new, Pr):
                break
            Pr = Pr_new
        C = ((Qn[:, None, :] - Pr[None, :, :]) ** 2).sum(-1)
        r, c = linear_sum_assignment(C)
        A = Pr[c]
        s = (Qn * A).sum() / max((A * A).sum(), 1e-12)
        val = 100 * ((Qn - s * A) ** 2).sum() / max((Qn * Qn).sum(), 1e-12)
        best = min(best, val)
    return float(best)


def econ(lengths: np.ndarray) -> float:
    """Hoppe's effective coordination number (iterated weighted mean)."""
    l = np.asarray(lengths, float)
    lav = l.min()
    for _ in range(50):
        w = np.exp(1 - (l / lav) ** 6)
        new = (l * w).sum() / w.sum()
        if abs(new - lav) < 1e-9:
            break
        lav = new
    return float(np.exp(1 - (l / lav) ** 6).sum())


def _centers_and_neighbors(st: Structure, scale: float, centers: Optional[Sequence[int]]):
    """(centers, {center: [(j, shift, vector)]}): the centers by the viewer's rule
    (structures.polyhedron_centers) unless given."""
    if centers is None:
        from mace.gui.structures import polyhedron_centers
        centers = polyhedron_centers(st.to_atoms(), scale)
    from .topology import bonds
    w = st.wrapped()
    nb = {i: [] for i in range(st.n)}
    for a, b, t in bonds(w, scale):
        va = (w.frac[b] + np.array(t) - w.frac[a]) @ w.cell
        nb[a].append((b, t, va))
        nb[b].append((a, tuple(-x for x in t), -va))
    # the vertices are the ligands (structures.polyhedron_ligands): anions more electronegative than the
    # center, never a cation-cation contact (the Si 2.7 Å from an olivine's Mg)
    from mace.gui.structures import ANION_Z, _en
    z = st.to_atoms().numbers
    anion = [int(z[j]) in ANION_Z and all(_en(int(z[k])) <= _en(int(z[j])) for k, _, _ in nb[j]) for j in range(st.n)]
    for i in centers:
        nb[i] = [x for x in nb[i] if anion[x[0]] and _en(int(z[x[0]])) > _en(int(z[i]))]
    return list(centers), nb


def analyze(st: Structure, scale: float = 1.15, centers: Optional[Sequence[int]] = None, shapes: bool = True) -> Dict:
    from scipy.spatial import ConvexHull
    global _SHAPES
    if _SHAPES is None:
        _SHAPES = _ideal_shapes()
    cen, nb = _centers_and_neighbors(st, scale, centers)
    sym = st.symbols
    rows = []
    for i in cen:
        vecs = np.array([v for _, _, v in nb[i]])
        if len(vecs) < 2:
            continue
        L = np.linalg.norm(vecs, axis=1)
        lav = float(L.mean())
        row = {"index": int(i), "element": sym[i], "cn": int(len(L)),
               "ligands": sorted({sym[j] for j, _, _ in nb[i]}),
               "bond_min": float(L.min()), "bond_max": float(L.max()), "bond_mean": lav,
               "distortion_index": float(np.abs(L - lav).sum() / len(L) / lav), "econ": econ(L)}
        vol = None
        if len(L) >= 4:
            try:
                vol = float(ConvexHull(vecs).volume)
            except Exception:
                vol = None
        row["volume"] = vol
        u = vecs / L[:, None]
        cosang = np.clip(u @ u.T, -1, 1)
        ang = np.degrees(np.arccos(cosang[np.triu_indices(len(L), 1)]))
        if len(L) == 4 and vol:
            l0 = (vol * 9 * math.sqrt(3) / 8) ** (1 / 3)             # tetrahedron of the same volume: V = 8 l0^3 / (9 sqrt 3)
            row["quadratic_elongation"] = float(((L / l0) ** 2).mean())
            row["angle_variance"] = float(((ang - 109.4712) ** 2).sum() / 5)
        elif len(L) == 6 and vol:
            l0 = (3 * vol / 4) ** (1 / 3)                            # octahedron: V = 4 l0^3 / 3
            row["quadratic_elongation"] = float(((L / l0) ** 2).mean())
            cis = np.sort(ang)[:12]
            row["angle_variance"] = float(((cis - 90.0) ** 2).sum() / 11)
        if shapes and len(L) in _SHAPES:
            scores = {name: shape_measure(vecs, P) for name, P in _SHAPES[len(L)].items()}
            best = min(scores, key=scores.get)
            row["shape"] = best
            row["cshm"] = scores[best]
            row["shape_scores"] = scores
        rows.append(row)
    # one line per distinct environment, too
    summary = {}
    for r in rows:
        k = (r["element"], r["cn"], r.get("shape"))
        s = summary.setdefault(k, {"element": r["element"], "cn": r["cn"], "shape": r.get("shape"), "count": 0,
                                    "bond_mean": 0.0, "volume": 0.0, "distortion_index": 0.0})
        s["count"] += 1
        s["bond_mean"] += r["bond_mean"]
        s["volume"] += r["volume"] or 0.0
        s["distortion_index"] += r["distortion_index"]
    for s in summary.values():
        for key in ("bond_mean", "volume", "distortion_index"):
            s[key] /= s["count"]
    return {"polyhedra": rows, "summary": list(summary.values()), "centers_rule": "cations with 4-12 anion ligands (A sites and carbon left out)",
            "scale": scale}
