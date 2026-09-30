# Copyright (c) 2025 Computational Materials Science Team (lcao2wannier, author William Comaskey).
# MIT License, see LICENSE in this directory.
# Vendored into MACE from lcao2wannier 1.0.0; local changes are listed in VENDORED.md.
"""Gauge-invariant spread (Omega_I) and Omega_I-optimal disentanglement windows.

Ported from wien2wannier's ``mod_omega.f`` / ``pdwf_optwin.f`` (Comaskey, 2026).

The gauge-invariant spread is

    Omega_I = (1/Nk) sum_{k,b} w_b ( Nw - || V_k^H M^{(k,b)} V_{k+b} ||_F^2 )

where M^{(k,b)}_{mn} = <u_mk|u_n,k+b> are the .mmn overlaps, V_k is an orthonormal
Nb x Nw basis of the selected subspace, and w_b are the Marzari-Vanderbilt
finite-difference weights.  Omega_I depends only on the *subspace* (it is
gauge-invariant), so it can be evaluated directly from the .mmn overlaps and
.amn projections -- without running the full maximal-localization minimisation --
and re-evaluated cheaply for many candidate disentanglement windows.

This enables automatic, spread-minimising selection of the disentanglement
windows, as an alternative to the energy-extrema + projectability heuristics in
``projectability.py`` / ``lcao_pdwf.py``.

Array conventions (0-based indices throughout):
    mmn   : (nk, nntot, nb, nb) complex   M^{(k,b)}
    amn   : (nk, nb, nw)        complex   A^{(k)} projections
    eig   : (nk, nb)            float      band energies (eV)
    neigh : (nk, nntot)         int        k+b index for each (k, b)
    w     : (nntot,)            float      MV b-weights
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "mv_weights",
    "omega_invariant",
    "disentangle_omega",
    "optimize_window",
    "auto_window_from_seedname",
    "read_amn",
    "read_eig",
    "read_mmn",
    "read_nnkp_bvectors",
    "have_fortran",
]

# Optional compiled kernel (wien2wannier mod_omega.f via f2py); ~5x faster.
# Build with scripts/build_spread_fortran.sh.
try:
    from . import _spread_fortran as _spread_fortran_mod
    _FORTRAN = _spread_fortran_mod.mod_omega
except Exception:
    _FORTRAN = None


def have_fortran() -> bool:
    """True if the compiled Fortran disentanglement kernel is available."""
    return _FORTRAN is not None


def mv_weights(bcart: np.ndarray, tol: float = 1e-6):
    """Marzari--Vanderbilt finite-difference shell weights.

    Parameters
    ----------
    bcart : (nntot, 3) Cartesian b-vectors (1/Ang) for a single k-point's shell
            of neighbours (a regular MP mesh shares the same b-set at every k).

    Returns
    -------
    w : (nntot,) per-b weights.
    max_resid : completeness residual  max|sum_b w_b b_a b_b - delta_ab|.
                For a 2D mesh the out-of-plane (zz) component is unsatisfiable
                and contributes ~1 to the residual; this does not affect
                Omega_I, which only sums over the in-plane b-vectors actually
                present.
    """
    bcart = np.asarray(bcart, float)
    nbv = bcart.shape[0]
    bn = np.linalg.norm(bcart, axis=1)

    # group neighbours into shells of equal |b|
    shell = np.full(nbv, -1, int)
    shval: list[float] = []
    for i in range(nbv):
        s = next((j for j, v in enumerate(shval) if abs(bn[i] - v) < tol), -1)
        if s < 0:
            shval.append(bn[i]); s = len(shval) - 1
        shell[i] = s
    ns = len(shval)

    # 6 x ns matrix of shell second moments, components [xx, yy, zz, xy, yz, zx]
    Mat = np.zeros((6, ns))
    bx, by, bz = bcart[:, 0], bcart[:, 1], bcart[:, 2]
    for comp, vals in enumerate((bx * bx, by * by, bz * bz,
                                 bx * by, by * bz, bz * bx)):
        np.add.at(Mat[comp], shell, vals)
    rhs = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

    wshell, *_ = np.linalg.lstsq(Mat, rhs, rcond=None)
    w = wshell[shell]

    comp_mat = np.einsum("b,bi,bj->ij", w, bcart, bcart)
    max_resid = float(np.max(np.abs(comp_mat - np.eye(3))))
    return w, max_resid


def omega_invariant(V: np.ndarray, mmn: np.ndarray,
                    neigh: np.ndarray, w: np.ndarray) -> float:
    """Gauge-invariant spread Omega_I for orthonormal subspaces ``V``.

    V : (nk, nb, nw) with V_k^H V_k = I.
    """
    nw = V.shape[2]
    # batched matmul (BLAS) — the einsum form of the same contractions is
    # >10x slower at production scale (Sc: nb=441, nw=294)
    MV = mmn @ V[neigh]                              # (nk, nntot, nb, nw)
    Vh = V.conj().transpose(0, 2, 1)[:, None]        # (nk, 1, nw, nb)
    Mt = Vh @ MV                                     # V_k^H M V_{k+b}
    proj = (Mt.real**2 + Mt.imag**2).sum(axis=(2, 3))       # ||.||_F^2
    omega = float((w * (nw - proj)).sum())
    return omega / V.shape[0]


def _svd_left(A: np.ndarray, na: int) -> np.ndarray:
    """Top-``na`` left singular vectors of ``A`` (m x n)."""
    U, _s, _Vh = np.linalg.svd(A, full_matrices=False)
    return U[:, :na]


def disentangle_omega(mmn, amn, eig, neigh, w, num_wann,
                      froz_min, froz_max, dis_min, dis_max,
                      niter: int = 4000, tol: float = 1e-7, backend: str = "auto"):
    """Minimised Omega_I for a disentanglement window (Souza--Marzari--Vanderbilt).

    Dispatches to the compiled Fortran kernel (built from wien2wannier's
    ``mod_omega.f`` via f2py, ~5x faster) when available, otherwise the pure-Python
    reference.  ``backend`` is 'auto' (prefer Fortran), 'fortran', or 'python'.
    Returns ``(omega_I, info)``; info=1 for an invalid window.
    """
    if backend == "fortran" or (backend == "auto" and _FORTRAN is not None):
        if _FORTRAN is None:
            raise RuntimeError("Fortran spread kernel not built; "
                               "run scripts/build_spread_fortran.sh")
        return _disentangle_omega_fortran(mmn, amn, eig, neigh, w,
                                          froz_min, froz_max, dis_min, dis_max,
                                          niter, tol)
    return _disentangle_omega_python(mmn, amn, eig, neigh, w, num_wann,
                                     froz_min, froz_max, dis_min, dis_max, niter, tol)


def _disentangle_omega_fortran(mmn, amn, eig, neigh, w,
                               froz_min, froz_max, dis_min, dis_max, niter, tol):
    """f2py bridge: translate C-order arrays to the Fortran layout (column-major,
    axes (nb,nb,nntot,nk) etc., 1-based neighbours) and call mod_omega."""
    mmn_f = np.asfortranarray(np.asarray(mmn, complex).transpose(2, 3, 1, 0))
    amn_f = np.asfortranarray(np.asarray(amn, complex).transpose(1, 2, 0))
    eig_f = np.asfortranarray(np.asarray(eig, float).T)
    neigh_f = np.asfortranarray((np.asarray(neigh, np.int64) + 1).T.astype(np.int32))
    w_f = np.asfortranarray(np.asarray(w, float))
    omega, info = _FORTRAN.disentangle_omega(
        mmn_f, amn_f, eig_f, neigh_f, w_f,
        float(froz_min), float(froz_max), float(dis_min), float(dis_max),
        int(niter), float(tol))
    info = int(info)
    # Fortran leaves omega unset on the invalid-window early return; normalise to
    # +inf so both backends agree (and optimize_window skips it on info != 0).
    return (np.inf if info != 0 else float(omega)), info


def _disentangle_omega_python(mmn, amn, eig, neigh, w, num_wann,
                              froz_min, froz_max, dis_min, dis_max,
                              niter: int = 4000, tol: float = 1e-7):
    """Souza--Marzari--Vanderbilt disentanglement: minimised Omega_I for a window.

    Frozen bands (eig in [froz_min, froz_max]) are forced into the subspace; the
    remaining ``num_wann - n_froz`` dimensions are taken from the
    projection-optimal complement among the disentanglement bands (eig in
    [dis_min, dis_max]) and iterated to minimise Omega_I.

    Returns ``(omega_I, info)`` with info=0 on success, info=1 if the window is
    invalid (n_froz > num_wann or n_froz + n_dis < num_wann at some k).
    """
    nk, nntot = neigh.shape
    nb = eig.shape[1]
    nw = num_wann

    froz, dis = [], []
    for k in range(nk):
        is_froz = (eig[k] >= froz_min) & (eig[k] <= froz_max)
        is_dis = (eig[k] >= dis_min) & (eig[k] <= dis_max) & ~is_froz
        fk = np.where(is_froz)[0]
        dk = np.where(is_dis)[0]
        if len(fk) > nw or len(fk) + len(dk) < nw:
            return np.inf, 1
        froz.append(fk); dis.append(dk)

    # initial subspace: frozen unit columns + SVD of A on the disentangle bands
    U = np.zeros((nk, nb, nw), complex)
    for k in range(nk):
        c = len(froz[k])
        U[k, froz[k], np.arange(c)] = 1.0
        nadd = nw - c
        if nadd > 0:
            Vt = _svd_left(amn[k][dis[k], :], nadd)        # (ndis, nadd)
            U[k][np.ix_(dis[k], np.arange(c, nw))] = Vt

    nfroz = np.array([len(f) for f in froz])
    cols = [np.arange(nfroz[k], nw) for k in range(nk)]
    omp = np.inf
    om = np.inf
    for _it in range(niter):
        # Gauss--Seidel sweep (in-place: each k sees neighbours already updated this
        # sweep).  Pure Jacobi diverges for this fixed-point iteration, so the
        # update must be sequential.  The per-k work is vectorised over neighbours.
        for k in range(nk):
            c = nfroz[k]; nadd = nw - c
            if nadd == 0:
                continue
            MU = np.einsum("iab,ibw->iaw", mmn[k], U[neigh[k]])   # (nntot, nb, nw)
            Z = np.einsum("i,iaw,icw->ac", w, MU, MU.conj())      # (nb, nb)
            _ev, evec = np.linalg.eigh(Z[np.ix_(dis[k], dis[k])])  # ascending
            U[k] = 0.0
            U[k, froz[k], np.arange(c)] = 1.0
            U[k][np.ix_(dis[k], cols[k])] = evec[:, : -nadd - 1 : -1]
        om = omega_invariant(U, mmn, neigh, w)
        if abs(omp - om) < tol:
            break
        omp = om
    return float(om), 0


def optimize_window(mmn, amn, eig, neigh, bcart, num_wann,
                    froz_min, dis_min, dis_max,
                    cover_override: float | None = None,
                    niter: int = 4000, tol: float = 1e-6,
                    froz_max_fixed: float | None = None):
    """Choose the Omega_I-minimising disentanglement window (port of pdwf_optwin).

    The frozen ceiling is set from a coverage constraint so the window never
    under-freezes:  E_cov = max_k eig_sorted(num_wann)  (or ``cover_override``),
    capped below the first invalid edge  min_k eig_sorted(num_wann+1).  Candidate
    outer edges ``dis_max`` are then scanned and the one minimising Omega_I kept.

    Returns a dict with the chosen froz/outer edges and the minimised Omega_I.
    """
    nk = eig.shape[0]
    w, _resid = mv_weights(bcart)

    # coverage floor: count only bands above froz_min (exclude semicore below it)
    ecov, evalid = -np.inf, np.inf
    for k in range(nk):
        e = np.sort(eig[k][eig[k] >= froz_min])
        if num_wann <= len(e):
            ecov = max(ecov, e[num_wann - 1])
        if num_wann + 1 <= len(e):
            evalid = min(evalid, e[num_wann])
    if cover_override is not None:
        ecov = cover_override
    froz_max = min(ecov, evalid - 1e-4)
    if froz_max_fixed is not None:
        # the automatic window route: the frozen ceiling is derived upstream
        # (resolve_frozen_ceiling); only the capacity guard of this band set
        # may lower it, and only the outer window is optimized
        froz_max = min(float(froz_max_fixed), evalid - 1e-4)

    dmax_try = [dis_max, eig.max() + 1.0, eig.max() - 2.0,
                froz_max + 8.0, froz_max + 5.0, froz_max + 3.0]
    best, best_dmax = np.inf, dis_max
    for dmax in dmax_try:
        om, info = disentangle_omega(mmn, amn, eig, neigh, w, num_wann,
                                     froz_min, froz_max, dis_min, dmax,
                                     niter=niter, tol=tol)
        if info == 0 and om < best:
            best, best_dmax = om, dmax

    return {
        "froz_min": froz_min, "froz_max": froz_max,
        "dis_min": dis_min, "dis_max": best_dmax,
        "omega_I": best, "e_cov": ecov, "e_valid": evalid,
    }


# ---------------------------------------------------------------------------
# Wannier90 input-file readers + seedname driver
# ---------------------------------------------------------------------------

def read_nnkp_bvectors(path):
    """Parse a .nnkp: returns (recip 3x3, kpts (nk,3), neigh0, gshift0, nntot).

    ``neigh0``/``gshift0`` describe the first k-point's neighbour block (a regular
    MP mesh shares the same b-set at every k), used to build Cartesian b-vectors.
    """
    with open(path) as f:
        lines = f.read().splitlines()

    def block(tag):
        i = next(j for j, l in enumerate(lines) if l.strip().startswith("begin " + tag))
        k = next(j for j, l in enumerate(lines) if l.strip().startswith("end " + tag))
        return lines[i + 1:k]

    recip = np.array([[float(x) for x in r.split()] for r in block("recip_lattice")])
    kb = block("kpoints"); nk = int(kb[0].split()[0])
    kpts = np.array([[float(x) for x in kb[1 + i].split()[:3]] for i in range(nk)])
    nnb = block("nnkpts"); nntot = int(nnb[0].split()[0])
    neigh0, g0 = [], []
    for i in range(nntot):
        p = nnb[1 + i].split()
        neigh0.append(int(p[1]) - 1)
        g0.append([int(p[2]), int(p[3]), int(p[4])])
    return recip, kpts, np.array(neigh0), np.array(g0, float), nntot


def read_eig(path, nb, nk):
    """Parse a .eig -> (nk, nb) band energies."""
    d = np.loadtxt(path)
    eig = np.zeros((nk, nb))
    eig[d[:, 1].astype(int) - 1, d[:, 0].astype(int) - 1] = d[:, 2]
    return eig


def read_amn(path):
    """Parse a .amn -> (amn (nk,nb,nw), nb, nk, nw)."""
    with open(path) as f:
        f.readline()
        nb, nk, nw = map(int, f.readline().split())
    d = np.loadtxt(path, skiprows=2)
    amn = np.zeros((nk, nb, nw), complex)
    m = d[:, 0].astype(int) - 1; n = d[:, 1].astype(int) - 1; k = d[:, 2].astype(int) - 1
    amn[k, m, n] = d[:, 3] + 1j * d[:, 4]
    return amn, nb, nk, nw


def read_mmn(path):
    """Parse a .mmn -> (mmn (nk,nntot,nb,nb), neigh (nk,nntot))."""
    with open(path) as f:
        lines = f.read().splitlines()
    nb, nk, nntot = map(int, lines[1].split())
    nbb = nb * nb
    mmn = np.empty((nk * nntot, nb, nb), complex)
    neigh = np.empty(nk * nntot, int)
    base = 2
    for blk in range(nk * nntot):
        h = lines[base].split()
        neigh[blk] = int(h[1]) - 1
        arr = np.fromstring(" ".join(lines[base + 1:base + 1 + nbb]), sep=" ").reshape(nbb, 2)
        mmn[blk] = (arr[:, 0] + 1j * arr[:, 1]).reshape(nb, nb, order="F")
        base += nbb + 1
    return mmn.reshape(nk, nntot, nb, nb), neigh.reshape(nk, nntot)


def auto_window_from_seedname(seedname, num_wann, froz_min, froz_max,
                              dis_min, dis_max, cover_override=None,
                              niter=1000, tol=1e-4, verbose=True,
                              froz_max_fixed=None):
    """Choose the Omega_I-optimal disentanglement window for an existing run.

    Reads ``seedname.{nnkp,amn,eig,mmn}``, evaluates Omega_I at the incoming
    (Stage-1) window, then returns the optimizer result with the incoming value
    added as ``omega_I_before`` for an improvement report.  ``froz_max`` is the
    original frozen ceiling; the optimizer recomputes it from the coverage floor.
    """
    recip, kpts, neigh0, g0, nntot = read_nnkp_bvectors(seedname + ".nnkp")
    amn, nb, nk, nw = read_amn(seedname + ".amn")
    eig = read_eig(seedname + ".eig", nb, nk)
    mmn, neigh = read_mmn(seedname + ".mmn")
    bcart = np.array([(kpts[neigh0[i]] + g0[i] - kpts[0]) @ recip for i in range(nntot)])
    w, resid = mv_weights(bcart)
    if verbose:
        print(f"  [auto-window] nb={nb} nw={nw} nk={nk} nntot={nntot}  "
              f"MV-weight residual={resid:.3g}")
        backend = ("Fortran kernel" if _FORTRAN is not None
                   else "pure Python -- build scripts/build_spread_fortran.sh for ~5x")
        print(f"  [auto-window] evaluating Omega_I over candidate windows "
              f"(<={niter} disentangle iters each, {backend})...")

    om_before, info_b = disentangle_omega(mmn, amn, eig, neigh, w, num_wann,
                                          froz_min, froz_max, dis_min, dis_max,
                                          niter=niter, tol=tol)
    res = optimize_window(mmn, amn, eig, neigh, bcart, num_wann,
                          froz_min, dis_min, dis_max,
                          cover_override=cover_override, niter=niter, tol=tol,
                          froz_max_fixed=froz_max_fixed)
    res["omega_I_before"] = om_before
    res["info_before"] = info_b
    return res
