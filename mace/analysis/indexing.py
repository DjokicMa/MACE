"""Indexing powder peaks: Miller indices for measured lines, against a known
cell or from scratch.

All work is in Q = 1/d^2, which is linear in the six components of the
reciprocal metric tensor: Q(hkl) = h^T G* h.

- ``index_with_cell``: each line gets the nearest calculated reflection of a
  starting cell (optionally without the space group's absences), then G* is
  refined by linear least squares within the crystal system of that cell.
- ``index_ab_initio``: trial cells for the cubic, tetragonal, hexagonal and
  orthorhombic systems from the first lines (each hypothesis assigns small
  Miller indices to one, two or three lines and solves for the lattice
  constants), each refined on every line it indexes and ranked by de Wolff's
  figure of merit M20 (Smith-Snyder F_N too). Monoclinic and triclinic cells
  are not searched; for those, index against a cell from a structure model.
"""
from __future__ import annotations

import itertools
import math
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import pxrd


def q_of_two_theta(tt, lam):
    s = np.sin(np.radians(np.asarray(tt, float)) / 2)
    return (2 * s / lam) ** 2


def two_theta_of_q(q, lam):
    s = np.sqrt(np.asarray(q, float)) * lam / 2
    return np.degrees(2 * np.arcsin(np.clip(s, 0, 1)))


def _metric_from_cell(cell):
    rec = np.linalg.inv(np.asarray(cell, float)).T
    return rec @ rec.T


def cell_from_metric(Gs: np.ndarray) -> np.ndarray:
    """A cell (rows) whose reciprocal metric is Gs (upper-triangular setting)."""
    G = np.linalg.inv(Gs)
    a, b, c = np.sqrt(np.diag(G))
    al = math.degrees(math.acos(np.clip(G[1, 2] / (b * c), -1, 1)))
    be = math.degrees(math.acos(np.clip(G[0, 2] / (a * c), -1, 1)))
    ga = math.degrees(math.acos(np.clip(G[0, 1] / (a * b), -1, 1)))
    from ase.geometry import cellpar_to_cell
    return cellpar_to_cell([a, b, c, al, be, ga])


def _lattice(cell):
    from .structure import Structure
    return Structure(cell, np.zeros((1, 3)), [1]).lattice()


def figures_of_merit(q_obs: np.ndarray, q_calc_all: np.ndarray, dq: np.ndarray, n: int = 20) -> Dict[str, float]:
    """de Wolff M_N and Smith-Snyder F_N for the first n indexed lines."""
    n = min(n, len(q_obs))
    if n == 0:
        return {"M20": 0.0, "F20": 0.0, "N": 0}
    qn = q_obs[n - 1]
    eps = float(np.mean(np.abs(dq[:n]))) or 1e-12
    uniq = np.unique(np.round(q_calc_all[q_calc_all <= qn * (1 + 1e-9)], 9))
    N = max(len(uniq), 1)
    m = qn / (2 * eps * N)
    return {"M20": float(m), "N": n, "Ncalc": int(N), "eps": eps}


def _assign(q_obs, q_calc, hkl, tol_q):
    """Nearest calculated line for each observed one."""
    keep = np.flatnonzero(q_calc <= q_obs.max() * 1.05 + 1e-12)
    if len(keep) < 2:
        keep = np.argsort(q_calc)[:2]
    order = keep[np.argsort(q_calc[keep])]
    qc = q_calc[order]
    j = np.clip(np.searchsorted(qc, q_obs), 1, len(qc) - 1)
    left = qc[j - 1]
    right = qc[j]
    pick = np.where(np.abs(q_obs - left) <= np.abs(q_obs - right), j - 1, j)
    dq = q_obs - qc[pick]
    ok = np.abs(dq) <= tol_q
    return hkl[order][pick], dq, ok


def _refine_metric(q_obs, hkl, basis):
    """Least squares on G* = sum_k c_k E_k: Q = h^T G* h."""
    X = np.stack([np.einsum("ni,ij,nj->n", hkl, E, hkl) for E in basis], 1)
    c, *_ = np.linalg.lstsq(X, q_obs, rcond=None)
    return np.tensordot(c, basis, axes=1)


_SYSTEM_BASIS = {
    "cubic": [np.eye(3)],
    "tetragonal": [np.diag([1.0, 1.0, 0]), np.diag([0, 0, 1.0])],
    "hexagonal": [np.array([[1.0, 0.5, 0], [0.5, 1.0, 0], [0, 0, 0]]), np.diag([0, 0, 1.0])],
    "orthorhombic": [np.diag([1.0, 0, 0]), np.diag([0, 1.0, 0]), np.diag([0, 0, 1.0])],
}


def _metric_basis(Gs0: np.ndarray, rotations=None) -> List[np.ndarray]:
    """Basis of reciprocal metrics with the symmetry of rotations (frac)."""
    if rotations is None:
        rotations = [np.eye(3, dtype=int)]
    out = []
    for i in range(3):
        for j in range(i, 3):
            E = np.zeros((3, 3))
            E[i, j] = E[j, i] = 1.0
            # G* transforms as R^T... average the action h -> h R
            P = sum(np.asarray(R, float) @ E @ np.asarray(R, float).T for R in rotations) / len(rotations)
            out.append(P[np.triu_indices(3)])
    M = np.array(out).T
    u, s, _ = np.linalg.svd(M)
    k = int((s > 1e-8 * max(s.max(), 1)).sum())
    res = []
    for c in range(k):
        E = np.zeros((3, 3))
        E[np.triu_indices(3)] = u[:, c]
        E = E + E.T - np.diag(np.diag(E))
        res.append(E)
    return res


def index_with_cell(two_theta: Sequence[float], cell, wavelength="CuKa", structure=None, tol_deg: float = 0.1,
                    refine: bool = True) -> Dict:
    """Index measured lines with a starting cell (or the cell of a structure)."""
    from .refine import absent, lattice_ops, symmetry_ops
    lam = pxrd.source(wavelength)[0]
    tt = np.sort(np.asarray(two_theta, float))
    qo = q_of_two_theta(tt, lam)
    if structure is not None:
        st = pxrd.conventional(structure)
        cell = st.cell
        rot, tr, _ = symmetry_ops(st)
    else:
        cell = np.asarray(cell, float)
        rot, tr = lattice_ops(cell), np.zeros((1, 3))
    d_min = lam / (2 * math.sin(math.radians(min(tt.max() + 5, 179)) / 2))
    H, d = pxrd.hkl_list(cell, d_min)
    if structure is not None:
        keep = ~absent(H, rot, tr)
        H, d = H[keep], d[keep]
    basis = _metric_basis(_metric_from_cell(cell), [np.asarray(R) for R in rot])
    Gs = _metric_from_cell(cell)
    tol_q = np.abs(q_of_two_theta(tt + tol_deg, lam) - qo)
    for _ in range(3 if refine else 1):
        qc = np.einsum("ni,ij,nj->n", H, Gs, H)
        hk, dq, ok = _assign(qo, qc, H, tol_q)
        if refine and ok.sum() >= len(basis):
            Gs = _refine_metric(qo[ok], hk[ok], np.array(basis))
    qc = np.einsum("ni,ij,nj->n", H, Gs, H)
    hk, dq, ok = _assign(qo, qc, H, tol_q)
    newcell = cell_from_metric(Gs)
    fom = figures_of_merit(qo[ok], qc, dq[ok])
    lines = [{"two_theta": float(t), "hkl": h.tolist() if o else None, "two_theta_calc": float(two_theta_of_q(q - e, lam)) if o else None,
              "delta": float(t - two_theta_of_q(q - e, lam)) if o else None} for t, q, h, e, o in zip(tt, qo, hk, dq, ok)]
    return {"lines": lines, "indexed": int(ok.sum()), "total": len(tt), "lattice": _lattice(newcell),
            "cell": newcell.tolist(), "figures_of_merit": fom, "wavelength": lam}


_CENTERINGS = {"cubic": ("P", "I", "F"), "tetragonal": ("P", "I"), "hexagonal": ("P", "R"),
               "orthorhombic": ("P", "C", "A", "B", "I", "F")}


def _centering_mask(H: np.ndarray, cen: str) -> np.ndarray:
    """Reflections a lattice centering allows (hexagonal axes for R, obverse)."""
    h, k, l = H[:, 0], H[:, 1], H[:, 2]
    if cen == "I":
        return (h + k + l) % 2 == 0
    if cen == "F":
        return ((h % 2) == (k % 2)) & ((k % 2) == (l % 2))
    if cen == "C":
        return (h + k) % 2 == 0
    if cen == "A":
        return (k + l) % 2 == 0
    if cen == "B":
        return (h + l) % 2 == 0
    if cen == "R":
        return (-h + k + l) % 3 == 0
    return np.ones(len(H), bool)


def _hkl_small(maxi=4):
    r = range(-maxi, maxi + 1)
    H = np.array([h for h in itertools.product(r, r, r) if any(h)])
    return H


_QUAD_CACHE = {}


def _quad(H):
    """Rows (h^2, k^2, l^2, 2hk, 2hl, 2kl), so Q = rows . (G11, G22, G33, G12, G13, G23)."""
    key = (id(H), len(H))
    if key not in _QUAD_CACHE:
        if len(_QUAD_CACHE) > 16:
            _QUAD_CACHE.clear()
        h, k, l = H[:, 0].astype(float), H[:, 1].astype(float), H[:, 2].astype(float)
        _QUAD_CACHE[key] = (H, np.stack([h * h, k * k, l * l, 2 * h * k, 2 * h * l, 2 * k * l], 1))
    return _QUAD_CACHE[key][1]


def _score(qo, Gs, H, tol_q):
    g = np.array([Gs[0, 0], Gs[1, 1], Gs[2, 2], Gs[0, 1], Gs[0, 2], Gs[1, 2]])
    qc = _quad(H) @ g
    hk, dq, ok = _assign(qo, qc, H, tol_q)
    return qc, hk, dq, ok


def index_ab_initio(two_theta: Sequence[float], wavelength="CuKa", systems=("cubic", "tetragonal", "hexagonal", "orthorhombic"),
                    tol_deg: float = 0.03, n_lines: int = 20, max_volume: float = 4000.0, min_volume: float = 10.0,
                    top: int = 5) -> Dict:
    """Trial cells for measured lines (see the module docstring)."""
    lam = pxrd.source(wavelength)[0]
    tt = np.sort(np.asarray(two_theta, float))[:n_lines]
    if len(tt) < 4:
        raise ValueError("indexing needs at least 4 lines")
    qo = q_of_two_theta(tt, lam)
    tol_q = np.abs(q_of_two_theta(tt + tol_deg, lam) - qo)
    H = _hkl_small(6)
    cands = []

    tried = set()
    masks = {cen: H[_centering_mask(H, cen)] for cen in ("P", "I", "F", "C", "A", "B", "R")}

    def consider(system, Gs):
        if not np.all(np.isfinite(Gs)) or np.any(np.linalg.eigvalsh(Gs) <= 0):
            return
        key = (system, tuple(np.round(np.diag(Gs) / max(qo[0], 1e-12), 4)), round(float(Gs[0, 1] / max(qo[0], 1e-12)), 4))
        if key in tried:
            return
        tried.add(key)
        vol = 1 / math.sqrt(max(np.linalg.det(Gs), 1e-30))
        if vol > max_volume or vol < min_volume:
            return
        basis = np.array(_SYSTEM_BASIS[system])
        need = max(3, len(basis) + 1, int(0.5 * len(qo)))      # a trial cell must already index half the lines
        for _ in range(2):
            qc, hk, dq, ok = _score(qo, Gs, H, 3 * tol_q)
            if ok.sum() < need:
                return
            Gs = _refine_metric(qo[ok], hk[ok], basis)
            if np.any(np.linalg.eigvalsh(Gs) <= 0):
                return
        # the lattice centering that keeps every indexed line and counts the
        # fewest calculated ones (the best M20)
        best = None
        for cen in _CENTERINGS[system]:
            Hc = masks[cen]
            qc, hk, dq, ok = _score(qo, Gs, Hc, tol_q)
            fom = figures_of_merit(qo[ok], qc, dq[ok], n=min(20, len(qo)))
            key = (int(ok.sum()), fom["M20"])
            if best is None or key > best[0]:
                best = (key, cen)
        (n_ok, M), cen = best
        cands.append({"system": system, "Gs": Gs, "indexed": n_ok, "M": M, "centering": cen,
                      "volume": 1 / math.sqrt(np.linalg.det(Gs))})

    q1 = qo[: min(6, len(qo))]
    if "cubic" in systems:
        for i in range(min(3, len(q1))):
            for N in (1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12):
                consider("cubic", np.eye(3) * q1[i] / N)
    m_t = (1, 2, 4, 5, 8, 9, 10)
    m_h = (1, 3, 4, 7, 9, 12)
    l2 = (0, 1, 4, 9)
    for system, ms in (("tetragonal", m_t), ("hexagonal", m_h)):
        if system not in systems:
            continue
        for i, j in itertools.combinations(range(len(q1)), 2):
            for (mi, li), (mj, lj) in itertools.product(itertools.product(ms, l2), repeat=2):
                Mx = np.array([[mi, li], [mj, lj]], float)
                if abs(np.linalg.det(Mx)) < 1e-9:
                    continue
                A, C = np.linalg.solve(Mx, [q1[i], q1[j]])
                if A <= 0 or C <= 0:
                    continue
                Gs = A * _SYSTEM_BASIS[system][0] + C * _SYSTEM_BASIS[system][1]
                consider(system, Gs)
    if "orthorhombic" in systems:
        small = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1), (2, 0, 0), (0, 2, 0),
                 (0, 0, 2), (2, 1, 0), (1, 2, 0), (0, 2, 1), (0, 1, 2), (2, 0, 1), (1, 0, 2)]
        sq = np.array(small, float) ** 2
        for i, j, k in itertools.combinations(range(len(q1)), 3):
            for a, b, c in itertools.permutations(range(len(sq)), 3):
                Mx = np.array([sq[a], sq[b], sq[c]])
                if abs(np.linalg.det(Mx)) < 1e-9:
                    continue
                ABC = np.linalg.solve(Mx, [q1[i], q1[j], q1[k]])
                if np.any(ABC <= 0):
                    continue
                # a <= b <= c convention removes the permutation copies
                if not (ABC[0] >= ABC[1] >= ABC[2]):
                    continue
                consider("orthorhombic", np.diag(ABC))
    # rank: all lines indexed first, then M20 favoring the higher symmetry
    # (a lower-symmetry cell can always index what a higher one does); drop
    # near-duplicate cells
    bonus = {"cubic": 2.0, "hexagonal": 1.5, "tetragonal": 1.5, "orthorhombic": 1.0}
    cands.sort(key=lambda c: (-c["indexed"], -c["M"] * bonus[c["system"]], c["volume"]))
    out, seen = [], []
    for c in cands:
        key = (c["system"], round(c["volume"], 1))
        if key in seen:
            continue
        seen.append(key)
        cell = cell_from_metric(c["Gs"])
        qc, hk, dq, ok = _score(qo, c["Gs"], masks[c["centering"]], tol_q)
        out.append({"system": c["system"], "centering": c["centering"], "lattice": _lattice(cell), "cell": cell.tolist(),
                    "volume": c["volume"], "indexed": c["indexed"], "total": len(tt), "M20": c["M"],
                    "lines": [{"two_theta": float(t), "hkl": h.tolist() if o else None} for t, h, o in zip(tt, hk, ok)]})
        if len(out) >= top:
            break
    return {"solutions": out, "wavelength": lam, "lines_used": len(tt),
            "note": "Monoclinic and triclinic cells are not searched."}
