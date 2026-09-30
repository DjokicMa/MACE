# Copyright (c) 2025 William Comaskey. MIT License, see LICENSE in this directory.
# Vendored into MACE from lcao2wannier 1.0.0; local changes are listed in VENDORED.md.
"""Exact reimplementation of wannier90's k-mesh neighbor construction.

Faithful transcription of ``kmesh.F90`` (wannier90; validated against the
3.1 binary's ``-pp`` output for every material in ``calculations/``), so the
pipeline can emit the ``.mmn`` in the exact (k, b) order the wannierise run
will independently derive — eliminating the ``wannier90.x -pp`` round trip
and the two-stage split it forces.

Faithfulness notes (each mirrors a specific construct in kmesh.F90):
  * supercell search order = ``kmesh_supercell_sort``: cells of the
    (2*5+1)^3 supercell ordered by |G|, ties broken by the "reproducible
    maxloc" (extract max, degenerates within 1e-8 resolved to the LOWEST
    enumeration index, filling from the END) — i.e. ties end up in
    DESCENDING enumeration order. Enumeration is (0,0,0) first, then the
    l,m,n triple loop, l slowest.
  * shell distances measured from k-point 1 against ALL k-points + images,
    with the ABSOLUTE tolerance ``tol`` (kmesh_tol, default 1e-6); shell
    membership in the neighbor/b-vector collection uses the RELATIVE
    window dnn*(1 +- tol).
  * automatic shell selection = ``kmesh_shell_automatic``: reject shells
    parallel to an accepted one (|cos| within 1e-6 of 1), append shell,
    solve the B1 system by SVD of the 6 x n_shell moment matrix
    (rows xx,xy,yy,xz,yz,zz; target [1,0,1,0,0,1]); reject the shell if
    any singular value < 1e-5 (fatal if it is the first shell); accept
    when the B1 residuals are within ``tol``.
  * neighbor list order = the ``kmesh_get`` standard path: per k-point,
    shells in shell_list order, supercell images in sorted order,
    k-points in index order, early exit at the shell multiplicity.

Only the first-order finite-difference path (higher_order_n = 1) is
implemented — the default for every standard wannier90 run. Explicit
``shell_list`` / ``nnkpts`` / ``search_shells`` overrides in the .win are
NOT honored here; runs needing them must fall back to ``wannier90.x -pp``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

__all__ = ["kmesh_get", "KmeshInfo", "write_nnkp", "reciprocal_lattice"]

_ETA = 99999999.0
_EPS5 = 1.0e-5
_EPS6 = 1.0e-6
_EPS8 = 1.0e-8


@dataclass
class KmeshInfo:
    nntot: int
    nnlist: np.ndarray        # (nk, nntot) int, 0-based neighbour k index
    nncell: np.ndarray        # (nk, nntot, 3) int, G shift (l, m, n)
    bk: np.ndarray            # (nk, nntot, 3) float, b-vectors (Ang^-1)
    wb: np.ndarray            # (nntot,) float, b-weights (Ang^2)
    shell_list: List[int]     # accepted shells (1-based, as in W90 output)
    dnn: np.ndarray           # shell distances
    multi: np.ndarray         # shell multiplicities


def reciprocal_lattice(real_lattice: np.ndarray) -> np.ndarray:
    """Rows b1,b2,b3 with a_i . b_j = 2 pi delta_ij (W90 convention)."""
    A = np.asarray(real_lattice, float)
    return 2.0 * np.pi * np.linalg.inv(A).T


def _supercell_sort(recip: np.ndarray, nsupcell: int = 5) -> np.ndarray:
    """kmesh_supercell_sort: images ordered by |G| with the reproducible
    tie-break (extract-max filling from the end; degenerate distances
    within 1e-8 of the current max resolve to the lowest enumeration
    index)."""
    ncell = (2 * nsupcell + 1) ** 3
    lmn = np.zeros((ncell, 3), int)
    dist = np.zeros(ncell)
    counter = 0                        # (0,0,0) at position 0, dist 0
    for l in range(-nsupcell, nsupcell + 1):
        for m in range(-nsupcell, nsupcell + 1):
            for n in range(-nsupcell, nsupcell + 1):
                if l == 0 and m == 0 and n == 0:
                    continue
                counter += 1
                lmn[counter] = (l, m, n)
                dist[counter] = np.linalg.norm(lmn[counter] @ recip)
    order = np.empty(ncell, int)
    work = dist.copy()
    for pos in range(ncell - 1, -1, -1):
        guess = int(np.argmax(work))                     # first max
        ties = np.where(np.abs(work - work[guess]) < _EPS8)[0]
        pick = int(ties.min())                           # lowest index
        order[pos] = pick
        work[pick] = -1.0
    return lmn[order]


def _shell_distances(kpt_cart: np.ndarray, images: np.ndarray,
                     search_shells: int, tol: float
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Distances (and multiplicities) of the nearest-neighbour shells of
    k-point 1, scanning all k-points + supercell images in kmesh_get's
    exact loop order (nkp outer, image inner)."""
    # all candidate distances (the sequential scan in kmesh_get reduces
    # exactly to: eligible minimum above the previous shell, multiplicity
    # counted in the strict (dnn1 - tol, dnn1 + tol) window)
    cand = (kpt_cart[:, None, :] + images[None, :, :]) - kpt_cart[0]
    dists = np.sqrt((cand ** 2).sum(axis=2)).ravel()
    dnn = np.zeros(search_shells)
    multi = np.zeros(search_shells, int)
    dnn0 = 0.0
    for nlist in range(search_shells):
        eligible = dists[(dists > tol) & (dists > dnn0 + tol)]
        if eligible.size:
            dnn1 = float(eligible.min())
            multi[nlist] = int(np.count_nonzero(
                (eligible > dnn1 - tol) & (eligible < dnn1 + tol)))
        else:
            dnn1 = _ETA
            multi[nlist] = 0
        dnn[nlist] = dnn1
        dnn0 = dnn1
    return dnn, multi


def _shell_bvectors(kpt_cart: np.ndarray, images: np.ndarray,
                    shell_dist: float, multi: int, tol: float) -> np.ndarray:
    """kmesh_get_bvectors from k-point 1: image loop outer, k loop inner,
    relative tolerance, early exit at the shell multiplicity."""
    out = np.zeros((multi, 3))
    n = 0
    for G in images:
        for kc in kpt_cart:
            v = G + kc
            dist = np.linalg.norm(kpt_cart[0] - v)
            if shell_dist * (1 - tol) <= dist <= shell_dist * (1 + tol):
                out[n] = v - kpt_cart[0]
                n += 1
            if n == multi:
                return out
    raise RuntimeError("kmesh: not enough b-vectors found for shell")


def _shells_automatic(kpt_cart, images, dnn, multi, search_shells, tol):
    """kmesh_shell_automatic: returns (shell_list [0-based], bweight)."""
    target = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    # (num_x, num_y, num_z) for the 6 second-moment equations, in
    # kmesh_get_amat's enumeration: xx, xy, yy, xz, yz, zz
    powers = [(2, 0, 0), (1, 1, 0), (0, 2, 0), (1, 0, 1), (0, 1, 1),
              (0, 0, 2)]
    shell_list: List[int] = []
    shell_bvecs: List[np.ndarray] = []
    bweight = None
    for shell in range(search_shells):
        bv = _shell_bvectors(kpt_cart, images, dnn[shell],
                             int(multi[shell]), tol)
        # parallel-shell rejection against ALL accepted shells
        lpar = False
        for prev in shell_bvecs:
            cos = (bv @ prev.T) / np.outer(
                np.linalg.norm(bv, axis=1), np.linalg.norm(prev, axis=1))
            if np.any(np.abs(np.abs(cos) - 1.0) < _EPS6):
                lpar = True
                break
        if lpar:
            continue
        shell_list.append(shell)
        shell_bvecs.append(bv)
        amat = np.zeros((6, len(shell_list)))
        for s, bvs in enumerate(shell_bvecs):
            for i, (px, py, pz) in enumerate(powers):
                amat[i, s] = np.sum(bvs[:, 0] ** px * bvs[:, 1] ** py
                                    * bvs[:, 2] ** pz)
        U, singv, Vt = np.linalg.svd(amat, full_matrices=True)
        if np.any(np.abs(singv) < _EPS5):
            if len(shell_list) == 1:
                raise RuntimeError(
                    "kmesh: SVD found a very small singular value on the "
                    "first shell")
            shell_list.pop()
            shell_bvecs.pop()
            continue
        # bweight = V^T Sigma^-1 U^T target (pseudo-inverse onto target)
        bweight = Vt.T @ ((U.T @ target)[: len(singv)] / singv)
        # B1 check within kmesh tol
        bsat = True
        for i, (px, py, pz) in enumerate(powers):
            delta = sum(w * np.sum(bvs[:, 0] ** px * bvs[:, 1] ** py
                                   * bvs[:, 2] ** pz)
                        for w, bvs in zip(bweight, shell_bvecs))
            if abs(delta - target[i]) > tol:
                bsat = False
        if bsat:
            return shell_list, np.asarray(bweight)
    raise RuntimeError(
        f"kmesh: unable to satisfy B1 with the first {search_shells} "
        f"shells (long cell or irregular MP grid?); fall back to "
        f"wannier90.x -pp")


def kmesh_get(real_lattice: np.ndarray, kpt_latt: np.ndarray,
              search_shells: int = 36, tol: float = 1.0e-6,
              nsupcell: int = 5) -> KmeshInfo:
    """W90-exact neighbor construction for the MP k-point list ``kpt_latt``
    (fractional, in the same order they will appear in the .win)."""
    recip = reciprocal_lattice(real_lattice)
    kpt_cart = np.asarray(kpt_latt, float) @ recip
    nk = kpt_cart.shape[0]
    lmn = _supercell_sort(recip, nsupcell)
    images = lmn @ recip                                # (ncell, 3) cart

    dnn, multi = _shell_distances(kpt_cart, images, search_shells, tol)
    shell_list, bweight = _shells_automatic(kpt_cart, images, dnn, multi,
                                            search_shells, tol)

    nntot = int(sum(multi[s] for s in shell_list))
    nnlist = np.zeros((nk, nntot), int)
    nncell = np.zeros((nk, nntot, 3), int)
    bk = np.zeros((nk, nntot, 3))
    wb = np.concatenate([np.full(multi[s], w)
                         for s, w in zip(shell_list, bweight)])

    for nkp in range(nk):
        nnx = 0
        for s in shell_list:
            lo, hi = dnn[s] * (1 - tol), dnn[s] * (1 + tol)
            found = 0
            for img_idx in range(images.shape[0]):
                # vectorized inner k loop; np.where returns ascending nkp2,
                # matching kmesh_get's acceptance order exactly
                v = images[img_idx] + kpt_cart           # (nk, 3)
                dist = np.linalg.norm(kpt_cart[nkp] - v, axis=1)
                for nkp2 in np.where((dist >= lo) & (dist <= hi))[0]:
                    nnlist[nkp, nnx] = nkp2
                    nncell[nkp, nnx] = lmn[img_idx]
                    bk[nkp, nnx] = v[nkp2] - kpt_cart[nkp]
                    nnx += 1
                    found += 1
                    if found == multi[s]:
                        break
                if found == multi[s]:
                    break
    return KmeshInfo(nntot=nntot, nnlist=nnlist, nncell=nncell, bk=bk,
                     wb=np.asarray(wb),
                     shell_list=[s + 1 for s in shell_list],
                     dnn=dnn, multi=multi)


def write_nnkp(path: str, real_lattice: np.ndarray, kpt_latt: np.ndarray,
               info: KmeshInfo) -> None:
    """Write a wannier90-compatible .nnkp (same block formats as -pp).

    The projections block is written empty: nothing in this pipeline reads
    it (the .amn is built internally), and wannier90 itself never reads the
    .nnkp back — it regenerates the k-mesh at wannierise time, which is
    exactly why ``kmesh_get`` must (and does) reproduce it.

    PRECISION MATTERS: pass the same full-precision lattice/kpoints that go
    into the .win. Pseudo-symmetric cells (e.g. a hexagonal lattice printed
    to 7 decimals) carry ~1e-8 shell-distance splittings, right at the
    sort's degeneracy tolerance — rounded inputs reorder the b-vectors.
    """
    recip = reciprocal_lattice(real_lattice)
    nk = np.asarray(kpt_latt).shape[0]
    with open(path, "w") as f:
        f.write("# File written by lcao2wannier.kmesh "
                "(wannier90-exact internal k-mesh)\n\n")
        f.write("calc_only_A  :  F\n\n")
        for name, mat in (("real_lattice", np.asarray(real_lattice, float)),
                          ("recip_lattice", recip)):
            f.write(f"begin {name}\n")
            for row in mat:
                f.write(f"{row[0]:12.7f}{row[1]:12.7f}{row[2]:12.7f}\n")
            f.write(f"end {name}\n\n")
        f.write("begin kpoints\n")
        f.write(f"{nk:6d}\n")
        for kp in np.asarray(kpt_latt, float):
            f.write(f"{kp[0]:14.8f}{kp[1]:14.8f}{kp[2]:14.8f}\n")
        f.write("end kpoints\n\n")
        f.write("begin projections\n     0\nend projections\n\n")
        f.write("begin nnkpts\n")
        f.write(f"{info.nntot:4d}\n")
        for k in range(nk):
            for i in range(info.nntot):
                g = info.nncell[k, i]
                f.write(f"{k + 1:8d}{info.nnlist[k, i] + 1:8d}"
                        f"{g[0]:7d}{g[1]:4d}{g[2]:4d}\n")
        f.write("end nnkpts\n\n")
        f.write("begin exclude_bands\n   0\nend exclude_bands\n")
