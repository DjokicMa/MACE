"""Hybrid PDWF-subspace + SCDM-gauge Wannierization (in-pipeline disentanglement).

Scalar-energy-window disentanglement cannot express per-state decisions
("freeze band 12 at the k-points where it is AO-like, release it where it is
free-electron-like"), and wannier90's Omega_I minimization has no
projectability floor, so it admits diffuse NFE components that lower Omega_I
while poisoning the gauge-optimization landscape (h-BN/MgB2 A/B evidence,
2026-07). This module performs the subspace selection in-pipeline instead:

  1. Per-state masks from Lowdin projectability P_nk:
       freeze   P_nk >= p_froz   (PDWF semantics; Qiao et al., npj Comput.
                                  Mater. 9, 208 (2023))
       admit    P_nk >= p_floor  (the projectability floor wannier90 lacks)
  2. Frozen-constrained Souza-Marzari-Vanderbilt Omega_I minimization
     (PRB 65, 035109 (2001)) over the admitted pool -> semi-unitary V(k).
     Mask-generalization of ``spread.disentangle_omega`` (energy windows).
  3. SCDM gauge in the AO-column discretization (Damle & Lin, Multiscale
     Model. Simul. 16, 1392 (2018), with projectability masks replacing the
     erfc energy weighting): QRCP anchors are k-independent AO indices, so
     the induced gauge is smooth in k.
  4. Rotation into the eigenbasis of the projected Hamiltonian, ready to hand
     wannier90 an isolated num_bands == num_wann problem (no dis_* keywords;
     works with stock wannier90 3.1).

Array conventions follow ``spread.py`` (0-based):
    mmn   : (nk, nntot, nb, nb) complex   M^{(k,b)} in the pool band basis
    amn   : (nk, nb, nproj)     complex   projections used to seed the free part
    neigh : (nk, nntot)         int       k+b index for each (k, b)
    w     : (nntot,)            float     MV b-weights (from spread.mv_weights)
    V     : (nk, nb, nw)        complex   semi-unitary subspace frames
"""
from __future__ import annotations

import time
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .spread import mv_weights, omega_invariant, _svd_left

# Optional compiled twin of disentangle_frozen (scripts/build_disentangle_fortran.sh).
# NOT the default: measured single-threaded it does NOT beat the NumPy path at
# production band counts, because NumPy dispatches to Accelerate/AMX and the
# interpreter overhead the kernel removes is smaller than that difference. It
# wins only where dispatch dominates (small nb). It exists as the foundation for
# colored-k threading, which Python cannot do (Gauss-Seidel is sequential in k
# and the GIL blocks it) — note that colouring changes the reached fixed point,
# so it is a research change, not a drop-in.
try:                                    # pragma: no cover - optional build
    from . import _disentangle_fortran as _dis_fortran_mod
    _DIS_FORTRAN = _dis_fortran_mod.mod_disentangle
except Exception:                       # pragma: no cover
    _DIS_FORTRAN = None


# Below this the geometric taper has spent the acceleration (alpha-1 < 1%),
# which is where the restart ("kick") re-raises it. Upstream's value.
_KICK_FLOOR = 1.0e-2


def have_disentangle_fortran() -> bool:
    """True if the compiled disentanglement kernel is importable."""
    return _DIS_FORTRAN is not None


__all__ = [
    "have_disentangle_fortran",
    "select_pool_and_frozen",
    "audit_pool_coverage",
    "degeneracy_regime",
    "symmetrize_degenerate",
    "auto_trust_thresholds",
    "disentangle_frozen",
    "scdm_anchors",
    "rotate_to_subspace",
    "mv_weights",           # re-export: callers need weights for the same b-set
    "omega_invariant",
]


def _shell_ceiling(
    ee: np.ndarray,
    fermi_shell: Tuple[float, float],
) -> Tuple[bool, np.ndarray, float]:
    """Classify metal/insulator and return the Fermi-shell ceiling.

    ``ee`` is (nk, nbands) energies relative to E_F. Returns
    ``(insulator, fermi_band, ceiling)``; states at or below ``ceiling``
    are the unconditional shell (for insulators the ceiling is mid-gap and
    ``fermi_band`` is all-False).

    Gap-aware: in an insulator, E_F + shell can reach past the gap into
    conduction the target AOs cannot represent — SnTe (0.3 eV gap,
    gap-edge conduction with min_k P = 0) froze P=0 states and the
    wannierise stalled (grad 0.48, Omega_OD 18). Capping the unconditional
    shell at MID-GAP pins exactly the occupied manifold and leaves
    conduction freezing to the (representability-aware) projectability
    masks. Metals unchanged.

    Factored out of select_pool_and_frozen so the pool-slice ceiling that
    builds the masks and the full-band-set ceiling that audit_pool_coverage
    reports cannot drift apart (T1.8: the two must be the SAME rule applied
    to different band sets, or the audit proves nothing).
    """
    occ = ee[ee <= 0.0]
    emp = ee[ee > 0.0]
    vbm = float(occ.max()) if occ.size else 0.0
    cbm = float(emp.min()) if emp.size else 0.0
    # Metal test: with energy-sorted bands a global gap forces every band
    # entirely below or above E_F, so ANY band crossing (min_k < 0 < max_k)
    # means metal. The old sampled-gap criterion (cbm - vbm > 0.05 eV)
    # misclassifies metals on coarse meshes: MgB2 on 8^3 has a sampled
    # "gap" of 0.055 eV because no k-point sits on the Fermi surface,
    # which silently disabled the metal Fermi shell.
    crossing = (ee.min(axis=0) < 0.0) & (ee.max(axis=0) > 0.0)
    insulator = (occ.size > 0 and emp.size > 0
                 and not bool(crossing.any()) and (cbm - vbm) > 0.0)
    if insulator:
        return True, np.zeros(ee.shape[1], dtype=bool), 0.5 * (vbm + cbm)
    fermi_band = ee.min(axis=0) <= abs(fermi_shell[1])
    ceiling = (float(ee[:, fermi_band].max()) if fermi_band.any()
               else -np.inf)
    return False, fermi_band, ceiling


def degeneracy_regime(
    eigenvalues: np.ndarray,
    dtol: float = 0.02,
    degen_frac_thr: float = 0.7,
) -> dict:
    """Measure how degenerate the band structure is, and pick the mask regime.

    ``degen_frac`` = 2 * count(consecutive gaps < dtol) / (nk * nb), the
    upstream statistic (wien2wannier ``auto_regime``, same dtol = 0.02 eV and
    threshold 0.7). Above the threshold the bands come in multiplets --
    Kramers pairs under SOC, cubic t2g/eg at high-symmetry k -- and a purely
    per-state projectability threshold can cut through one, freezing part of
    a degenerate group and releasing the rest.

    Measured on this repo: SnTe 1.063, 2D COF (2-component) 1.052, MgB2
    (no SOC) 0.013. So the SOC systems sit far above the threshold and the
    non-SOC metal far below -- the statistic separates them cleanly.

    Returns ``{'degen_frac', 'band_level', 'dtol'}``.
    """
    e = np.sort(np.asarray(eigenvalues, float), axis=1)
    frac = float(2 * np.sum(np.diff(e, axis=1) < dtol) / e.size)
    return {'degen_frac': frac, 'band_level': frac > degen_frac_thr,
            'dtol': float(dtol)}


def symmetrize_degenerate(
    proj: np.ndarray,
    eigenvalues: np.ndarray,
    dtol: float = 0.02,
) -> np.ndarray:
    """Average the projectability over each degenerate group, per k.

    States that are exactly degenerate are only defined up to a rotation
    within their multiplet, so their INDIVIDUAL projectabilities are not
    physically meaningful -- only the group mean is. Averaging first makes
    every threshold decision act on the whole multiplet at once, which is
    what makes the masks rotation-invariant.

    Cheap (one argsort per k) and a no-op on non-degenerate systems.
    """
    proj = np.asarray(proj, float).copy()
    eig = np.asarray(eigenvalues, float)
    nk, nb = proj.shape
    for k in range(nk):
        o = np.argsort(eig[k])
        e = eig[k][o]
        i = 0
        while i < nb:
            j = i
            while j + 1 < nb and e[j + 1] - e[i] < dtol:
                j += 1
            if j > i:
                grp = o[i:j + 1]
                proj[k, grp] = proj[k, grp].mean()
            i = j + 1
    return proj


def audit_pool_coverage(
    eigenvalues: np.ndarray,
    pool: np.ndarray,
    e_fermi: float = 0.0,
    fermi_shell: Optional[Tuple[float, float]] = (2.0, 3.0),
    seed_window: Tuple[float, float] = (-30.0, 20.0),
) -> dict:
    """Count sub-ceiling states the band pool cannot see (T1.8 detection).

    The pool is capped at ``max(nw+1, ceil(pool_factor*nw))`` and then every
    mask, every energy AND the ceiling itself are computed on pool-restricted
    slices — so cap and ceiling are mutually self-consistent by construction
    and a state below the true physical ceiling but outside the pool is
    invisible to every mask, count and diagnostic. This recomputes the
    ceiling on the FULL solved band set and counts what the pool missed, at
    BOTH ends: upstream's equivalent guard recovered 16 sub-ceiling bands at
    the pool BOTTOM (pool [17..34] -> [1..34]), and _select_pool's block
    shift can place the pool bottom above deep bands the frozen manifold
    needs.

    DETECTION ONLY — deliberately does not widen the pool. Oversized pools
    trap the Gauss-Seidel iteration (h-BN: 48-band pool -> Omega_I 19 vs
    8.58 on 24 bands with identical masks), so any repair must admit exactly
    the missing states, not raise pool_factor.

    Returns a dict with ``n_below``/``n_above`` (sub-ceiling STATES outside
    the pool, per end), ``bands_below``/``bands_above`` (the band indices
    involved), ``ceiling``, ``insulator``, and ``ceiling_at_solve_top``
    (True when the full-set Fermi band set reaches the highest solved band,
    i.e. the ceiling is only a lower bound because the solve itself was
    truncated).
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    pool = np.asarray(pool, dtype=int)
    nb = eigenvalues.shape[1]
    empty = {'n_below': 0, 'n_above': 0, 'bands_below': np.array([], int),
             'bands_above': np.array([], int), 'ceiling': None,
             'insulator': None, 'ceiling_at_solve_top': False,
             'missed_e_range': None}
    if fermi_shell is None:
        return empty

    ee_full = eigenvalues - e_fermi
    insulator, fermi_band, ceiling = _shell_ceiling(ee_full, fermi_shell)
    if not np.isfinite(ceiling):
        return empty

    # Only bands the model could actually have used: _select_pool rejects
    # anything outside seed_window (deep all-electron core states, diffuse
    # ghosts), so counting those would fire the warning on every
    # all-electron run — noise, not signal. h-BN: the four 1s core bands
    # sit ~180 eV below E_F, are trivially "sub-ceiling", and are excluded
    # by exactly this window.
    avg_e = ee_full.mean(axis=0)
    candidate = (avg_e >= seed_window[0]) & (avg_e <= seed_window[1])
    outside = np.setdiff1d(np.where(candidate)[0], pool)
    if outside.size == 0:
        return {**empty, 'ceiling': float(ceiling),
                'insulator': bool(insulator),
                'ceiling_at_solve_top': bool(fermi_band.any()
                                             and fermi_band[-1])}
    sub = ee_full[:, outside] <= ceiling + 1e-6          # (nk, n_outside)
    lo, hi = int(pool.min()), int(pool.max())
    below = outside < lo
    above = outside > hi
    bands_below = outside[below & sub.any(axis=0)]
    bands_above = outside[above & sub.any(axis=0)]
    missed = np.concatenate([bands_below, bands_above])
    return {
        'n_below': int(sub[:, below].sum()),
        'n_above': int(sub[:, above].sum()),
        'bands_below': bands_below,
        'bands_above': bands_above,
        # energy span of the missed bands, so the caller can tell a real
        # omission from something the target was never meant to include
        'missed_e_range': ((float(ee_full[:, missed].min()),
                            float(ee_full[:, missed].max()))
                           if missed.size else None),
        'ceiling': float(ceiling),
        'insulator': bool(insulator),
        # with --solve-nbands the full solved set can itself stop below the
        # true physical ceiling; then `ceiling` is only a lower bound
        'ceiling_at_solve_top': bool(fermi_band.any() and fermi_band[-1]),
    }


def _select_pool(
    proj: np.ndarray,
    eigenvalues: np.ndarray,
    num_wann: int,
    pool_factor: float = 1.5,
    e_fermi: float = 0.0,
    seed_window: Tuple[float, float] = (-30.0, 20.0),
) -> np.ndarray:
    """Contiguous band-index pool covering the seed manifold (see
    select_pool_and_frozen for the validated rationale)."""
    nk, nb = proj.shape
    avg_p = proj.mean(axis=0)
    avg_e = eigenvalues.mean(axis=0) - e_fermi
    n_pool_max = max(num_wann + 1, int(np.ceil(pool_factor * num_wann)))

    # manifold seed: the num_wann most-projectable PHYSICAL bands (the energy
    # window rejects ghost seeds, mirroring gap_aware_windows step 1)
    in_win = np.where((avg_e >= seed_window[0]) & (avg_e <= seed_window[1]))[0]
    if in_win.size < num_wann:
        in_win = np.arange(nb)
    seed = np.sort(in_win[np.argsort(avg_p[in_win])[::-1][:num_wann]])

    # contiguous index block covering the seed, sized to the pool budget
    s0, s1 = int(seed[0]), int(seed[-1])
    npool = max(n_pool_max, s1 - s0 + 1)
    s0 = max(0, min(s0, nb - npool))
    return np.arange(s0, min(nb, s0 + npool))


# ---------------------------------------------------------------------------
# Pool / frozen selection from projectability
# ---------------------------------------------------------------------------

def select_pool_and_frozen(
    proj: np.ndarray,
    eigenvalues: np.ndarray,
    num_wann: int,
    p_froz: float = 0.95,
    p_floor: float = 0.10,
    pool_factor: float = 1.5,
    p_froz_band: float = 0.85,
    e_fermi: float = 0.0,
    seed_window: Tuple[float, float] = (-30.0, 20.0),
    fermi_shell: Tuple[float, float] = (2.0, 3.0),
    shell_p_floor: Optional[object] = 'auto',
    shell_floor_max_frac: float = 0.01,
    capacity: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Global band pool + per-(k, band) admit/frozen masks from P_nk.

    The pool is one band-index list shared by all k (keeps every M(k,b) block
    square); the per-state decisions live in the boolean masks.

    Pool (validated on h-BN, 2026-07): a CONTIGUOUS band-index block covering
    the seed manifold (top-num_wann by mean P within ``seed_window`` rel E_F).
    Bands are energy-ordered at each k, so contiguity keeps the low conduction
    a smooth manifold needs and excludes high-energy diffuse-GTO ghost states
    by construction — ghosts project >= 0.85 onto the target AOs, so ANY
    pure-projectability ranking pulls them in (h-BN: pool [4..67], frozen
    ghosts, Omega_I 22-38 instead of 8.58 on the contiguous 24-band block).
    Keep pool_factor ~1.5: the Omega_I optimum saturates there, and oversized
    pools trap the Gauss-Seidel iteration in local minima (h-BN: 48-band pool
    -> Omega_I 19 vs 8.58 on the 24-band pool with identical masks).

    Frozen: P_nk >= p_froz AND the band's mean P >= p_froz_band. The band
    filter stops gray-zone bands (avg P ~0.8) from being frozen at the
    isolated k where P_nk spikes >= p_froz — such k-discontinuous pinning is
    unoptimizable and inflates Omega_I. Per-state release of good bands at
    their bad k (the PDWF advantage) is unaffected.

    Fermi shell: states with eig in [E_F - fermi_shell[0], E_F + fermi_shell[1]]
    are frozen UNCONDITIONALLY (and admitted), regardless of projectability.
    In a metal nothing else pins the Fermi-crossing states, and the Omega_I
    optimum will happily deform the Fermi surface for smoothness (MgB2:
    converged to Omega_Total 11.98 but band RMS 820 meV without the shell).
    Mirrors the std path's "energy-range freeze to E_F + 3 eV". For an
    insulator the shell adds nothing in the gap and merely reinforces the
    (already well-projected) top valence states.

    Admit: P_nk >= p_floor, topped up with the highest-P remaining states so
    at least num_wann are admitted at every k.

    ``capacity``: optional dict, filled in place with the frozen-set demand
    the mask rules actually produced (``need`` = max over k of the pre-cap
    frozen count, ``headroom`` = num_wann - need, ``n_capped`` /
    ``n_dropped`` = how many k-points and states the per-k cap silently
    discarded, ``n_zero_freedom`` = k-points left with no variational
    direction). An out-param rather than a 4th return value: six call sites
    unpack this function's tuple, including two test modules and the
    w2w bridge.

    Returns
    -------
    pool : (npool,) int — band indices (ascending) entering the problem.
    admit, frozen : (nk, npool) bool, frozen implies admit.
    """
    proj = np.asarray(proj, dtype=float)
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    nk, nb = proj.shape
    if not (0.0 <= p_floor < p_froz <= 1.0):
        raise ValueError(f"need 0 <= p_floor < p_froz <= 1, got "
                         f"p_floor={p_floor}, p_froz={p_froz}")

    avg_p = proj.mean(axis=0)
    pool = _select_pool(proj, eigenvalues, num_wann, pool_factor=pool_factor,
                        e_fermi=e_fermi, seed_window=seed_window)
    pp = proj[:, pool]                                    # (nk, npool)

    admit = pp >= p_floor
    frozen = (pp >= p_froz) & (avg_p[pool] >= p_froz_band)[None, :]
    if fermi_shell is not None:
        # BAND-level and ONE-SIDED: freeze every pool band whose bottom lies
        # below E_F + fermi_shell[1], at ALL k — the whole occupied+crossing
        # manifold, exactly the std path's E_F+3 energy-range rule. Weaker
        # variants measured on MgB2 (RMS in the std window, std ~20 meV):
        # no shell 820 meV; state-level shell 311 meV; intersecting-bands
        # 150 meV (crossing bands frozen but occupied 7/8 only patchily).
        # For insulators this freezes the full valence (h-BN: Omega_I 8.61
        # vs 8.58 per-state — negligible cost, and interpolation is exact
        # below E_F by construction).
        ee = eigenvalues[:, pool] - e_fermi                # (nk, npool)
        insulator, fermi_band, ceiling = _shell_ceiling(ee, fermi_shell)
        if insulator:
            # STATE-level shell: freeze exactly the occupied manifold (states
            # below mid-gap). Band-level rules necessarily over-freeze in
            # band-INVERTED narrow-gap materials (SnTe): the inverted bands
            # dip below E_F at some k (bot -4.9 eV) yet are P=0 NFE-like at
            # others, so "band starting below the ceiling" drags their whole
            # range into the frozen set and the wannierise stalls.
            shell = ee <= ceiling
        else:
            # Fermi ceiling = top of the highest band starting below
            # E_F+shell. Freeze ALL states below it (state-level, like a W90
            # energy window): higher bands dip under the ceiling (MgB2 band
            # 12 at +4.6..+7.7) and leaving those states free cost 150 meV
            # RMS vs the std window.
            shell = ((ee <= ceiling + 1e-6)
                     | np.broadcast_to(fermi_band, admit.shape))
        # Representability guard (generalizes the insulator mid-gap cap to
        # metals/semimetals): an energy-window shell that freezes ALL states
        # below the ceiling over-freezes band-inverted SEMIMETALS -- the
        # inverted states sit near E_F but are NFE/interstitial (P ~ 0), and
        # pinning them forces the WF subspace to contain a direction the
        # atomic anchors cannot represent -> singular gauge, dead
        # disentanglement (alpha-Sn: 14/14 frozen at every k, Omega_I stuck).
        # When shell_p_floor is set, freeze a near-E_F state only if it is
        # actually atomic-like (pp >= floor); genuinely-NFE states in the
        # window are released back to the variational subspace. Metals whose
        # Fermi states ARE projectable (MgB2 sigma/pi) keep the full shell;
        # insulators with all-projectable valence (h-BN) are untouched.
        # Default None preserves the original unconditional behavior.
        # SELF-CONSISTENT FLOOR ('auto', the default).
        #
        # The shell freezes near-E_F states UNCONDITIONALLY, so it can admit
        # states whose projectability is below p_froz -- states the mask rule
        # itself would reject. That is the whole defect: on SnTe the shell
        # pinned 6 states with P down to 0.4998 against p_froz = 0.550, and
        # removing exactly those restores the pre-regression Omega_I
        # (68.5446) to the digit.
        #
        # But the same rule must NOT fire on a metal whose Fermi states are
        # legitimately below p_froz: on MgB2 (p_froz 0.766) it would strip
        # 17.4% of the shell -- the sigma/pi Fermi surface -- and costs
        # eta 18.6 -> 281.8 meV, the exact trade the shell exists to prevent.
        #
        # The two are separated by HOW MUCH the floor removes, measured per
        # run (fraction of shell states below p_froz):
        #     h-BN  0.00%   SnTe  0.04%   MgB2 17.44%
        # so a 1% cut sits 25x above SnTe and 17x below MgB2. Note the
        # discriminator is NOT bimodality: measured in-shell distributions
        # are unimodal-high on both (SnTe min P 0.500, MgB2 min P 0.439,
        # neither has any P < 0.30 state).
        floor_applied, floor_frac = None, 0.0
        if shell_p_floor == 'auto':
            if shell.any():
                floor_frac = float((pp[shell] < p_froz).mean())
            if floor_frac <= shell_floor_max_frac:
                shell = shell & (pp >= p_froz)
                floor_applied = float(p_froz)
        elif shell_p_floor is not None:
            shell = shell & (pp >= float(shell_p_floor))
            floor_applied = float(shell_p_floor)
        frozen |= shell
        admit = admit | shell
    if fermi_shell is None:
        floor_applied, floor_frac = None, 0.0
    shell_m = shell if fermi_shell is not None else np.zeros_like(admit)
    # T1.7 capacity: record what the mask rules ACTUALLY demanded at each k,
    # measured here at the cap itself rather than predicted by a parallel
    # formula. A separate "count states below the ceiling" estimate diverges
    # from reality by construction, because it cannot see the insulator
    # mid-gap cap or the shell_p_floor release applied above.
    demand = np.count_nonzero(frozen, axis=1).astype(int)   # pre-cap, per k
    for ik in range(nk):
        fidx = np.where(frozen[ik])[0]
        if fidx.size > num_wann:
            # cap at num_wann, Fermi-shell states first (they are the physics
            # the subspace must reproduce), then highest projectability
            score = pp[ik, fidx] + 10.0 * shell_m[ik, fidx]
            keep = fidx[np.argsort(score)[::-1][:num_wann]]
            frozen[ik, :] = False
            frozen[ik, keep] = True
        short = num_wann - int(admit[ik].sum())
        if short > 0:
            rest = np.where(~admit[ik])[0]
            take = rest[np.argsort(pp[ik, rest])[::-1][:short]]
            admit[ik, take] = True
    frozen &= admit
    if capacity is not None:
        # frozen-set demand vs the num_wann budget. need > num_wann means the
        # per-k cap above SILENTLY discarded frozen states the mask rules
        # asked for; headroom == 0 at some k means the disentangler has no
        # free directions there (ntake == 0) and is inert, however well the
        # frozen set itself is chosen.
        capacity.update(
            shell_floor=floor_applied,
            shell_floor_frac=float(floor_frac),
            shell_floor_max_frac=float(shell_floor_max_frac),
            demand_per_k=demand,
            need=int(demand.max()),
            num_wann=int(num_wann),
            headroom=int(num_wann) - int(demand.max()),
            n_capped=int(np.count_nonzero(demand > num_wann)),
            n_dropped=int(np.clip(demand - num_wann, 0, None).sum()),
            n_zero_freedom=int(np.count_nonzero(
                np.count_nonzero(frozen, axis=1) >= num_wann)),
        )
    return pool, admit, frozen


# ---------------------------------------------------------------------------
# Automatic trust thresholds from the projectability distribution
# ---------------------------------------------------------------------------

def auto_trust_thresholds(
    proj: np.ndarray,
    eigenvalues: np.ndarray,
    num_wann: int,
    e_fermi: float = 0.0,
    fid_emax: float = 8.0,
    margin: float = 0.015,
    eps: float = 0.02,
    thr_floor: float = 0.55,
    mode: str = "fidelity",
    pool_factor: float = 1.5,
    seed_window: Tuple[float, float] = (-30.0, 20.0),
) -> Optional[dict]:
    """Derive (p_froz, p_froz_band) from the band-avg P distribution.

    No fixed threshold pair survived all validated materials (h-BN champion
    0.70/0.75; Bi needs p_froz <= ~0.87 or its conduction unfreezes, 0.32 ->
    152 meV; defaults 0.95/0.85 fine for MgB2 only because the metal Fermi
    shell dominates). The distribution itself, however, has the same
    structure everywhere: a well-projected trusted cluster, a gray-band
    plateau of diffuse/NFE bands below it, then junk. The 2x2 h-BN
    experiment showed the convergence cliff is ENTIRELY the band gate — it
    must sit above the gray plateau's avg P — while the state threshold
    only tunes how much of the trusted bands is pinned (full-band freeze =
    best fidelity, and it converges in the hybrid machinery).

    Rule (fidelity mode):
      relevant  = pool bands whose minimum energy dips below E_F + fid_emax
                  (bands that can contribute fidelity in the requested range)
      plateau   = highest band-avg P among IRRELEVANT pool bands (they can
                  only pollute the subspace, never help the fidelity window)
      gate      = plateau + margin          (p_froz_band)
      trusted   = relevant bands with avg P >= gate
      threshold = min_k P over trusted bands - eps   (p_froz; full-band
                  freeze of every trusted band, clipped to
                  [thr_floor, 0.95])
    Localization mode keeps the derived gate but pins p_froz = 0.95 (freeze
    only near-perfectly-AO states; smallest Omega, weakest conduction
    fidelity).

    Derived values on the 2026-07 validation set (12/12/1 Bi, 6^3 h-BN,
    8^3 MgB2, 8^3 SnTe-spd): h-BN 0.706/0.740 (= the hand-tuned champion
    0.70/0.75: conduction RMS 92.17 meV, Omega_T 13.127, converged);
    Bi 0.855/0.608 (full 16-band freeze, same physics as the manual 0.80
    fix; RMS 0.27 meV); MgB2 0.766/0.815 (trusted bands all below the
    Fermi ceiling -> exact no-op, Omega_T 14.752); SnTe 0.55(floored)/
    0.796 (trusts the gap-edge bands 50/51 the d-channel study proved
    essential; occ 10.4 / cond 46.6 meV = the study champion).

    ``thr_floor`` = 0.55 is calibrated, not cosmetic: SnTe's trusted set
    includes bands whose min_k P dips to 0.43, and chasing them
    (thr 0.408) froze 32/36 bands at EVERY k -> 4 free directions,
    disentangle stuck at Omega_I 69.8, gauge min-SV 0.007, occupied RMS
    41.9 meV. The A/B at gate 0.796: thr 0.55 -> occ 10.4/cond 46.6;
    thr 0.70 -> occ 9.8/cond 68.7. States below P ~ 0.55 are junk at
    that k no matter how trusted their band is elsewhere.

    Returns None when the rule cannot decide (no irrelevant band to place
    the plateau, or nothing trusted above the gate) — caller should fall
    back to fixed defaults and say so.
    """
    proj = np.asarray(proj, float)
    eigenvalues = np.asarray(eigenvalues, float)
    pool = _select_pool(proj, eigenvalues, num_wann, pool_factor=pool_factor,
                        e_fermi=e_fermi, seed_window=seed_window)
    avg_p = proj.mean(axis=0)
    min_p = proj.min(axis=0)
    bot_e = eigenvalues.min(axis=0) - e_fermi

    relevant = pool[bot_e[pool] <= fid_emax]
    irrelevant = pool[bot_e[pool] > fid_emax]
    if irrelevant.size == 0 or relevant.size == 0:
        return None
    plateau_top = float(avg_p[irrelevant].max())
    gate = plateau_top + margin

    trusted = relevant[avg_p[relevant] >= gate]
    if trusted.size == 0:
        return None
    if mode == "localization":
        thr = 0.95
    else:
        thr = float(min_p[trusted].min()) - eps
        thr = min(max(thr, thr_floor), 0.95)
    return {
        "p_froz": thr,
        "p_froz_band": gate,
        "plateau_top": plateau_top,
        "trusted": trusted,
        "relevant": relevant,
        "pool": pool,
        "trusted_min_p": float(min_p[trusted].min()),
    }


# ---------------------------------------------------------------------------
# Frozen-constrained Omega_I minimization with per-state masks
# ---------------------------------------------------------------------------

def disentangle_frozen(
    mmn: np.ndarray,
    amn: np.ndarray,
    neigh: np.ndarray,
    w: np.ndarray,
    num_wann: int,
    admit: np.ndarray,
    frozen: np.ndarray,
    niter: int = 2000,
    tol: float = 1e-9,
    rel_tol: float = 1e-10,
    verbose: bool = False,
    check_every: int = 1,
    mix: float = 1.0,
    sym=None,
    mix_schedule: Optional[float] = None,
    taper: float = 0.5,
    taper_on: str = "check",
    taper_reset_history: bool = False,
    relax_late: Optional[float] = None,
    log_path: Optional[str] = None,
    info: Optional[dict] = None,
    backend: str = "python",
) -> Tuple[np.ndarray, float, List[float]]:
    """Souza-Marzari-Vanderbilt Omega_I minimization with per-state masks.

    Mask-based generalization of ``spread._disentangle_omega_python``: the
    frozen/disentangle sets are given per (k, band) instead of via energy
    windows. Same Gauss-Seidel sweep (each k sees neighbours already updated
    this sweep; pure Jacobi diverges for this fixed-point iteration).

    Parameters
    ----------
    mmn : (nk, nntot, nb, nb) overlaps in the pool band basis.
    amn : (nk, nb, nproj) projections seeding the free directions (any
          reasonable AO projections; only the disentangle rows are used).
    admit, frozen : (nk, nb) bool masks (frozen implies admit).

    Returns
    -------
    V : (nk, nb, num_wann) semi-unitary; frozen states exactly contained
        (V V^H e_n = e_n for every frozen (k, n)).
    omega : final Omega_I (Ang^2)
    history : Omega_I at each CHECK (one entry per ``check_every`` sweeps,
        not per sweep — callers must not read len(history) as a sweep count;
        pass ``info`` to get the true ``n_sweeps``)

    ``mix`` applies fixed over-relaxation to the k-local operator before the
    eigendecomposition::

        Zd_eff(k) <- mix * Zd_new(k) + (1 - mix) * Zd_eff_prev(k)

    Note this mixes against the previous **effective** (already-mixed) Zd,
    which is the upstream recursion; mixing against the previous *raw* Zd is
    a different iteration with different convergence. Zd is Hermitian and a
    real combination of Hermitian matrices is Hermitian, so ``eigh`` is
    untouched for any real ``mix``. ``mix > 1`` is over-relaxation (the
    useful direction; upstream measured damping ``mix < 1`` as strictly
    worse). Default 1.0 reproduces the unmixed iteration exactly.

    ``mix_schedule`` (ported from upstream, where it beat every fixed mix:
    142 sweeps and the lowest Omega_I of any configuration on its scaling
    case) OVERRIDES ``mix``: start over-relaxed at the given PEAK (positive;
    upstream writes the same setting as a NEGATIVE ``relax`` because there a
    minus sign is the "schedule" sentinel — that convention is w2w-only, and
    a negative value here would compute Zd = peak*Zd_new + (1-peak)*Zd_prev
    with peak < 0, i.e. garbage) and taper geometrically toward 1.0:

        cur_mix <- 1 + taper * (cur_mix - 1)

    ``taper`` (default 0.5) is the geometric factor; ``taper_on`` selects
    when it fires:

      * ``"check"`` (default, upstream semantics): every exact check that
        did not converge. This is what every upstream measurement was made
        under. Because the taper is unconditional it spends itself fast —
        from 2.5 it is under 1.02 after ~10 checks — which is precisely why
        upstream added a restart ("kick"); without one, a schedule under
        this trigger is an OPENING accelerator, not a whole-run one.
      * ``"stall"`` (this code's original rule): only on a check that failed
        to improve (``om >= omp``), so the peak is held for as long as
        Omega_I keeps falling. NOT the upstream schedule; numbers measured
        under one trigger are not comparable to the other.

    ``taper_reset_history`` drops the per-k mixing history at each taper
    event. Upstream never does this (its Zhist is seeded only on the very
    first sweep of the run), so the default is False. The Sec. 2 fixed-point
    argument is untouched either way — at a fixed point the reset is a no-op
    — but the trajectory is not, and the trajectory is what selects the
    basin, so this is a measured choice rather than a safe one.

    Over-relaxation is a BASIN SELECTOR, not just an accelerator: the
    fixed-point set is independent of the mixing factor (at a fixed point
    the mixed and unmixed operators coincide for every non-zero factor), so
    the schedule cannot create or move a minimum — but on materials whose
    Omega_I landscape has several inequivalent minima it chooses which one
    attracts, silently and with ``converged=True``. Upstream measured a
    setting that converged in 60 sweeps to an Omega_I 7-9% worse than the
    best reachable one. Only a multi-alpha comparison detects this; do not
    read a low sweep count as a quality signal.

    Exclusive alternative to Anderson acceleration (NOT ported): upstream
    measured the two ANTI-composing — oscillation into the sweep cap.
    """
    nk, nntot = neigh.shape
    nb = mmn.shape[2]
    nw = num_wann
    admit = np.asarray(admit, bool)
    frozen = np.asarray(frozen, bool)
    if np.any(frozen & ~admit):
        raise ValueError("frozen mask must be a subset of the admit mask")

    if taper_on not in ("check", "stall"):
        raise ValueError(f"taper_on must be 'check' or 'stall', got "
                         f"{taper_on!r}")
    if not 0.0 <= taper <= 1.0:
        raise ValueError(f"taper must lie in [0, 1], got {taper}")
    # The opening factor and the taper are COUPLED, not independent knobs.
    # Kahan's bound says a stationary over-relaxation at alpha >= 2 cannot
    # converge, so a super-critical opening survives ONLY if the taper puts
    # it back under 2 within one step. Measured upstream from 2.5:
    # taper 0.5 spends one step above 2 (2.5 -> 1.75) and converges;
    # taper 0.8 spends two (2.5 -> 2.2 -> 1.96) and diverges outright
    # (Omega_I 181.6 after 200 sweeps). Refuse the combination at entry
    # instead of letting it blow up thousands of sweeps later.
    # Clearing the history on every check (which is when the upstream trigger
    # tapers) means zd_prev is always None at the top of a sweep, so the
    # mixing branch never fires: the schedule silently degenerates to plain
    # Gauss-Seidel. Measured on h-BN 6^3 — peaks 1.7/1.9/2.5 all returned the
    # mix=1.0 answer 10.530007503640226 in the same 278 sweeps.
    if (mix_schedule is not None and taper_on == "check"
            and taper_reset_history):
        print("    [hybrid] WARNING: taper_on='check' with "
              "taper_reset_history=True clears the mixing history before it "
              "can ever be used — the schedule is a NO-OP and this run is "
              "plain Gauss-Seidel. Use taper_reset_history=False (upstream), "
              "or taper_on='stall' to reproduce pre-v1.6 runs.")
    if relax_late is not None and not 0.0 < float(relax_late) < 2.0:
        # Kahan: a stationary over-relaxation at alpha >= 2 cannot converge,
        # and just BELOW 2 the failure is quality rather than stability --
        # upstream measured relax_late = 2.00 "converging" in 1907 sweeps to a
        # DEGRADED Omega_I (21.8280846 vs 21.82791941) and 2.05 diverging.
        # A clamp strictly below 2 is the minimum, not a conservative choice.
        raise ValueError(
            f"relax_late must lie in (0, 2), got {relax_late}; at or above "
            f"2 the iteration either degrades the minimum it reports or "
            f"diverges outright")
    if mix_schedule is not None and float(mix_schedule) > 2.0 and taper > 0.6:
        raise ValueError(
            f"unstable relaxation schedule: peak {float(mix_schedule):g} with "
            f"taper {taper:g} stays above the alpha = 2 stability bound for "
            f"more than one step and will diverge. Use taper <= 0.6 with a "
            f"super-critical peak (upstream default 0.5), or a peak <= 2.")

    if backend == "fortran":
        if mix_schedule is not None:
            raise ValueError("mix_schedule is not implemented in the "
                             "compiled kernel; use backend='python'")
        if _DIS_FORTRAN is None:
            raise RuntimeError(
                "compiled disentanglement kernel not built; run "
                "scripts/build_disentangle_fortran.sh (or use "
                "backend='python')")
        U, om, nsw, ierr = _DIS_FORTRAN.disentangle_frozen(
            np.asfortranarray(np.asarray(mmn, complex).transpose(2, 3, 1, 0)),
            np.asfortranarray(np.asarray(amn, complex).transpose(1, 2, 0)),
            np.asfortranarray((np.asarray(neigh, np.int64) + 1).T
                              .astype(np.int32)),
            np.asfortranarray(np.asarray(w, float)),
            np.asfortranarray(admit.T.astype(np.int32)),
            np.asfortranarray(frozen.T.astype(np.int32)),
            int(niter), float(tol), float(rel_tol), int(check_every),
            float(mix))
        if ierr == 1:
            raise ValueError("infeasible masks (see disentangle_frozen)")
        if ierr == 2:
            raise ValueError("frozen mask must be a subset of the admit mask")
        if ierr != 0:
            raise RuntimeError(f"compiled disentangle kernel failed: {ierr}")
        if info is not None:
            info.update(n_sweeps=int(nsw), n_checks=0,
                        check_every=int(check_every), converged=nsw < niter,
                        mix=float(mix), backend="fortran")
        V = np.ascontiguousarray(U.transpose(2, 0, 1))
        return V, float(om), [float(om)]
    if backend != "python":
        raise ValueError(f"backend must be 'python' or 'fortran', got {backend!r}")

    froz, dis = [], []
    for k in range(nk):
        fk = np.where(frozen[k])[0]
        dk = np.where(admit[k] & ~frozen[k])[0]
        if len(fk) > nw or len(fk) + len(dk) < nw:
            raise ValueError(
                f"infeasible masks at k={k}: {len(fk)} frozen, "
                f"{len(fk) + len(dk)} admitted, num_wann={nw}")
        froz.append(fk)
        dis.append(dk)

    # initial subspace: frozen unit columns + SVD of A on the disentangle rows
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
    # contiguous dis-row slices of M, once: the Gauss-Seidel update needs
    # only Z on the (dis, dis) subblock, and Z_ac = sum_i w_i (M_i U)_a .
    # (M_i U)_c^* with a,c in dis needs only the dis ROWS of M. Building
    # the full (nb, nb) Z per k (the old einsum) is what made Sc-scale
    # (nb=441, nw=294) cost ~100 s/sweep; dis-restricted batched matmul
    # (BLAS) brings a sweep to seconds. Same math, same fixed point.
    #
    # STRUCTURAL FORM (default). Three further exact reductions, each
    # measured on production shapes and leaving the SUBSPACE bit-identical
    # (max |P_old - P_new| ~ 1e-13 across h-BN/MgB2/SnTe/Sc):
    #
    #  (1) FROZEN-BLOCK SPLIT. The frozen columns of U are unit vectors that
    #      never change, so P(k) = P_froz(k) + V V^H with P_froz constant for
    #      the whole run. Hence
    #          Zd(k) = Zf(k) + sum_b w_b (M_b V_{k+b})(M_b V_{k+b})^H
    #      with Zf(k) = sum_b w_b M[dis(k), froz(k+b)] M[...]^H a per-k
    #      CONSTANT, precomputed once. The per-sweep product then carries
    #      only nadd = nw - nfroz columns instead of nw (Sc: 114 vs 294).
    #  (2) COLUMN RESTRICTION. V_{k+b} is identically zero outside
    #      dis(k+b), so M only needs those columns.
    #  (3) ZHERK. Zd is Hermitian PSD; folding sqrt(w_b) into M lets a
    #      single Hermitian rank-k update build one triangle (half the
    #      flops), fed straight to eigh(UPLO='L').
    #
    # Measured k-loop speedup: h-BN 1.60x, MgB2 1.50x, SnTe 2.65x, Sc 1.73x.
    #
    # NOT adopted: Omega_I from the retained eigenvalues (SMV Eq. 18). It is
    # nearly free, but under Gauss-Seidel it uses V(k+b) as of when k was
    # visited, so it LAGS the post-sweep value by 3-6% (h-BN 3.3%, MgB2
    # 5.8%, Sc 5.1%) and agrees only at the fixed point. Reporting it as
    # Omega_I would misstate the headline number, so the exact
    # omega_invariant is still used -- on the check_every stride.
    from scipy.linalg.blas import zherk as _zherk
    sw = np.sqrt(w)
    ndis_max = max(len(dis[k]) for k in range(nk))
    # M restricted to (dis rows of k, dis columns of the neighbour),
    # sqrt(w)-scaled. Neighbour widths differ (ndis varies with k), so pad
    # to ndis_max: the padded columns multiply zero rows of Vf and so
    # contribute nothing, which keeps the neighbour gather a single
    # batched matmul.
    # ONE stacked array, row- AND column-padded; the sweep consumes row-slice
    # VIEWS so it never sees the padded rows (zero rows would add spurious
    # zero eigenvalues to Zd). The stacked form is what lets the restricted
    # Omega_I monitor below be a single batched matmul.
    Mdd = np.zeros((nk, nntot, ndis_max, ndis_max), complex)
    for k in range(nk):
        for b in range(nntot):
            dnb = dis[neigh[k, b]]
            Mdd[k, b, :len(dis[k]), :len(dnb)] = (
                sw[b] * mmn[k, b][np.ix_(dis[k], dnb)])
    mmn_dis = [Mdd[k, :, :len(dis[k]), :] for k in range(nk)]
    # Sweep-invariant frozen contribution to Zd, PLUS the two constants the
    # restricted Omega_I monitor needs (see _omega_restricted below):
    #   aconst = sum_kb w_b ||M[froz(k), froz(k+b)]||_F^2      (a scalar)
    #   Q(k)   = Zf(k) + sum_b w_b M[froz,dis]^H M[froz,dis]   (per-k const)
    # Both are frozen-only or frozen-crossed blocks, and the frozen columns
    # of U are unit vectors that never move, so neither changes during the
    # iteration. Building them here costs one extra pass over the same
    # blocks the Zf loop already touches.
    Zf = [np.zeros((len(dis[k]), len(dis[k])), complex) for k in range(nk)]
    Q = np.zeros((nk, ndis_max, ndis_max), complex)
    aconst = 0.0
    for k in range(nk):
        for b in range(nntot):
            kp = neigh[k, b]
            Mf = sw[b] * mmn[k, b][np.ix_(dis[k], froz[kp])]
            Zf[k] += Mf @ Mf.conj().T
            X = sw[b] * mmn[k, b][np.ix_(froz[k], dis[kp])]
            Q[kp, :len(dis[kp]), :len(dis[kp])] += X.conj().T @ X
            Mff = mmn[k, b][np.ix_(froz[k], froz[kp])]
            aconst += w[b] * float((Mff.real**2 + Mff.imag**2).sum())
    for k in range(nk):
        Q[k, :len(dis[k]), :len(dis[k])] += Zf[k]
    nwW = nw * float(w.sum())
    # free columns per k, zero-padded to a common width so the neighbour
    # gather V[neigh[k]] stays a single batched op when nfroz varies with k
    # (it does: MgB2 5-7, SnTe 30-32). Padded columns are identically zero
    # and contribute nothing to Zd.
    nadd_max = max(nw - nfroz[k] for k in range(nk))
    Vf = np.zeros((nk, ndis_max, nadd_max), complex)
    for k in range(nk):
        na, nd_k = nw - nfroz[k], len(dis[k])
        Vf[k, :nd_k, :na] = U[k][np.ix_(dis[k], np.arange(nfroz[k], nw))]
    # Hoist all sweep-invariant per-k indexing out of the hot loop: the
    # masks (froz/dis/cols/nfroz) never change during disentanglement, so
    # np.ix_ and np.arange were being rebuilt nk*niter times for nothing
    # (MgB2 profile: np.ix_ alone 1.2 s / 9.2 s). Bit-identical result.
    ndis_k = [len(dis[k]) for k in range(nk)]
    frz_rows = [np.asarray(froz[k], dtype=np.intp) for k in range(nk)]
    frz_cols = [np.arange(nfroz[k], dtype=np.intp) for k in range(nk)]
    dis_ix = [np.ix_(dis[k], cols[k]) for k in range(nk)]
    ntake = [nw - nfroz[k] for k in range(nk)]
    wb = w[:, None, None]

    def _omega_restricted() -> float:
        """EXACT Omega_I, evaluated on the free block only.

        Identical value to ``omega_invariant(U, mmn, neigh, w)`` -- the same
        sum regrouped by P(k) = F(k) + V(k)V(k)^H, which holds exactly
        because the frozen columns are unit vectors. The frozen-frozen and
        frozen-free pieces are the run-constants ``aconst``/``Q`` built
        above, leaving only the free-free term to evaluate each check.

        This is NOT the rejected SMV Eq. 18: that substitutes the retained
        eigenvalues of the Zd built DURING the sweep from stale neighbours
        (hence its 3-6% lag). This runs AFTER the sweep against all-current
        Vf and contracts explicitly against Vf.

        Measured whole-run effect vs the old full-U monitor, threads pinned:
        SnTe 2.23x, h-BN 1.33x, MgB2 1.17x at check_every=1, with the
        converged projector BIT-IDENTICAL (max|dP| = 0.0) and Omega_I equal
        to ~1e-14 relative (FP reassociation only, ~2000x below rel_tol).
        Keep it VECTORIZED: a per-k python loop here would put nk numpy
        dispatches back into what used to be 50% of the runtime.
        """
        Y = Vf.conj().transpose(0, 2, 1)[:, None] @ Mdd
        Td = Y @ Vf[neigh]
        return nwW - (aconst
                      + float((Vf.conj() * (Q @ Vf)).real.sum())
                      + float((Td.real**2 + Td.imag**2).sum())) / nk
    history: List[float] = []
    omp = np.inf
    converged = False
    n_sweeps = 0
    # Live convergence trace. Written and FLUSHED at every check so a
    # multi-hour disentanglement can be followed while it runs instead of
    # inferred afterwards from a buffered stdout.
    _log = None
    _t0 = time.time()
    _d_prev = 0.0
    if log_path:
        _log = open(log_path, 'w')
        _log.write(f"# disentangle_frozen  nk={nk} nb={nb} num_wann={nw} "
                   f"ndis={min(ndis_k)}-{max(ndis_k)} "
                   f"nfroz={int(nfroz.min())}-{int(nfroz.max())}\n")
        _log.write(f"# niter={niter} tol={tol:g} rel_tol={rel_tol:g} "
                   f"check_every={check_every} mix={mix:g} "
                   f"mix_schedule={mix_schedule} taper={taper:g} "
                   f"taper_on={taper_on} relax_late={relax_late}\n")
        _log.write("#   sweep    alpha              Omega_I       dOmega "
                   "     rho   est_left    elapsed_s\n")
        _log.flush()
    # scheduled relaxation: start at the peak, taper toward 1.0 on checks
    # that fail to improve (see docstring). cur_mix is read by the sweep.
    cur_mix = float(mix_schedule) if mix_schedule is not None else float(mix)
    mix_trajectory: List[Tuple[int, float]] = []
    n_kicks = 0
    # per-k previous EFFECTIVE Zd for over-relaxation (see docstring)
    zd_prev: List[Optional[np.ndarray]] = [None] * nk
    for it in range(niter):
        n_sweeps = it + 1
        # SYMMETRY-CONSTRAINED SWEEP (sym is not None).
        #
        # Minimise Omega_I *within* symmetry-adapted subspaces instead of
        # minimising freely and repairing afterwards. Two changes, both tiny:
        #
        #   1. Symmetrise Zd over the little group of k before diagonalising,
        #      Zd <- (1/|G_k|) sum_{g in G_k} T(g,k) Zd T(g,k)^H. Zd then
        #      COMMUTES with the little group, so eigh's eigenvectors are
        #      automatically symmetry-adapted and its degenerate multiplets
        #      ARE the irrep components -- no character tables. (Same fact
        #      irrep_matched_trials uses.)
        #   2. Sweep only the IRREDUCIBLE k and propagate each star with the
        #      operator, so every star member is the exact image of its
        #      representative.
        #
        # This is wannier90's sitesym_symmetrize_zmatrix + dis_extract_symmetry
        # done in our own solver -- which matters because it COMPOSES WITH THE
        # FROZEN BLOCK, a combination wannier90 forbids outright.
        #
        # Post-hoc repair (minimise, then symmetrise, then project onto
        # span(W)) was measured on h-BN at Omega_I +17.8% AND eta 391.8 meV
        # against plain hybrid's 30.1 -- the projection picked its directions
        # by AO character, not by Omega_I. This does not have that failure
        # mode: nothing is projected, the minimiser simply never leaves the
        # symmetric manifold.
        _ks = range(nk) if sym is None else sym['kirr']
        for k in _ks:
            nadd = ntake[k]
            if nadd == 0:
                continue
            nd_k = ndis_k[k]
            # free-column product only: (nntot, ndis, nadd_max)
            MV = mmn_dis[k] @ Vf[neigh[k]]
            Bs = MV.transpose(1, 0, 2).reshape(nd_k, -1)
            Zd = _zherk(1.0, Bs, trans=0, lower=1)     # lower triangle
            Zd += Zf[k]                                # constant frozen part
            if cur_mix != 1.0:
                if zd_prev[k] is not None:
                    Zd = cur_mix * Zd + (1.0 - cur_mix) * zd_prev[k]
                zd_prev[k] = Zd
            if sym is None:
                _ev, evec = np.linalg.eigh(Zd, UPLO='L')  # ascending
                Vf[k, :nd_k, :nadd] = evec[:, : -nadd - 1 : -1]
                continue
            # zherk filled only the lower triangle; symmetrising needs the
            # full Hermitian matrix
            Zf_full = Zd + np.tril(Zd, -1).conj().T
            grp = sym['little'][k]
            if len(grp) > 1:
                acc = np.zeros_like(Zf_full)
                for g in grp:
                    T = sym['T'][k][g]
                    acc += T @ Zf_full @ T.conj().T
                Zf_full = acc / len(grp)
                Zf_full = 0.5 * (Zf_full + Zf_full.conj().T)
            _ev, evec = np.linalg.eigh(Zf_full)        # ascending
            # A degenerate eigenvalue straddling the cut would SPLIT an irrep
            # multiplet and silently break the invariance the symmetrisation
            # just bought. Count it -- it is the per-k irrep-budget mismatch,
            # measured rather than assumed.
            if nadd < nd_k:
                gap = float(_ev[-nadd] - _ev[-nadd - 1])
                scale = max(abs(float(_ev[-1])), 1e-30)
                if gap < 1e-6 * scale:
                    sym['split'][k] = sym['split'].get(k, 0) + 1
            Vf[k, :nd_k, :nadd] = evec[:, : -nadd - 1 : -1]
            # propagate the star: every member is the exact image of k
            for g, kg in sym['star'][k]:
                nd_g = ndis_k[kg]
                Vf[kg, :nd_g, :ntake[kg]] = (
                    sym['Tstar'][k][g] @ Vf[k, :nd_k, :nadd])
        # Omega_I is needed only for logging + the convergence test; at
        # small nb it costs as much as the sweep itself (MgB2 profile:
        # 7.1 of 16.7 s). Evaluate on a stride: comparing consecutive
        # CHECKS spans check_every sweeps, so the tolerance criterion is
        # strictly stronger than the per-sweep one, never weaker. The full
        # U is reassembled from the free block only here, for the same
        # reason -- the sweep itself never needs it.
        if it % check_every == 0 or it == niter - 1:
            # No U reassembly here any more: the restricted monitor reads the
            # free block directly, so the full (nk, nb, nw) array is rebuilt
            # exactly once, at the return.
            om = _omega_restricted()
            history.append(om)
            if verbose and it % 50 == 0:
                print(f"    [hybrid] disentangle sweep {it:4d}  "
                      f"Omega_I = {om:.8f}")
            if _log is not None:
                d_om = (omp - om) if np.isfinite(omp) else float('nan')
                rho = (d_om / _d_prev) if (_d_prev and np.isfinite(d_om)
                                           and _d_prev > 0) else float('nan')
                # geometric-tail extrapolation of what is LEFT to gain, so a
                # long run can be judged live instead of guessed at
                rem = (d_om * rho / (1.0 - rho)
                       if np.isfinite(rho) and 0.0 < rho < 1.0 else float('nan'))
                _log.write(f"{n_sweeps:8d} {cur_mix:8.4f} {om:20.10f} "
                           f"{d_om:12.3e} {rho:8.4f} {rem:12.3e} "
                           f"{time.time() - _t0:10.1f}\n")
                _log.flush()          # live: the file is the progress bar
                if np.isfinite(d_om) and d_om > 0:
                    _d_prev = d_om
            # absolute AND relative convergence: the absolute tol alone
            # never fires on production-scale Omega_I (Sc: ~300 Ang^2,
            # sweep deltas ~1e-3 at the cap -> silently rode the cap)
            if (abs(omp - om) < tol
                    or abs(omp - om) < rel_tol * max(abs(om), 1.0)):
                converged = True
                break
            if mix_schedule is not None and (taper_on == "check"
                                             or om >= omp):
                # This check did not converge, so taper toward plain
                # Gauss-Seidel. Under the upstream trigger ("check") that is
                # unconditional; under "stall" it fires only when Omega_I
                # also failed to improve, which holds the peak much longer.
                if cur_mix > 1.0 + _KICK_FLOOR or relax_late is None:
                    cur_mix = 1.0 + taper * (cur_mix - 1.0)
                else:
                    # RESTART ("kick"): the geometric taper halves alpha-1
                    # every check, so after ~10 checks it is spent and the
                    # rest of the run crawls UNRELAXED -- which is what the
                    # 2D COF did (8000 sweeps, still falling 1.3e-2/check).
                    # Re-raise alpha once the taper has decayed so the
                    # acceleration is available for the whole run, not just
                    # its opening. Self-pacing: driven by the decay, not by
                    # per-material tuning. The mixing history is deliberately
                    # NOT dropped here (upstream does not either) -- the
                    # basin was committed in the opening sweeps.
                    cur_mix = float(relax_late)
                    n_kicks += 1
                mix_trajectory.append((n_sweeps, cur_mix))
                if taper_reset_history:
                    zd_prev = [None] * nk  # drop the over-relaxed history
            omp = om
    if not converged and len(history) >= 2:
        # history entries are check_every sweeps apart, so the last delta
        # spans that many sweeps — report the span rather than implying the
        # improvement came from a single sweep.
        slope = history[-2] - history[-1]
        span = "last sweep" if check_every == 1 else f"last {check_every} sweeps"
        print(f"    [hybrid] WARNING: disentangle hit the sweep cap "
              f"(niter={niter}) without converging: Omega_I = "
              f"{history[-1]:.4f}, {span} improved {slope:.2e} Ang^2. "
              f"Raise --hybrid-dis-niter (or loosen --hybrid-dis-tol) if "
              f"the tail slope is still significant.")
    # Rebuild the full U ONCE, here: the sweep and the restricted monitor
    # both work on the free block, so this is the only place the caller's
    # (nk, nb, nw) array is needed.
    for k in range(nk):
        U[k] = 0.0
        U[k, frz_rows[k], frz_cols[k]] = 1.0
        if ntake[k]:
            U[k][dis_ix[k]] = Vf[k, :ndis_k[k], :ntake[k]]
    if info is not None:
        info.update(n_sweeps=n_sweeps, n_checks=len(history),
                    check_every=int(check_every), converged=bool(converged),
                    mix=float(mix), mix_schedule=mix_schedule,
                    final_mix=float(cur_mix), taper=float(taper),
                    taper_on=taper_on,
                    taper_reset_history=bool(taper_reset_history),
                    relax_late=relax_late, n_kicks=n_kicks,
                    mix_trajectory=mix_trajectory)
    if _log is not None:
        _log.write(f"# {'converged' if converged else 'CAP REACHED'} after "
                   f"{n_sweeps} sweeps, Omega_I = {history[-1]:.10f}, "
                   f"{time.time() - _t0:.1f} s\n")
        _log.close()
    return U, history[-1], history


# ---------------------------------------------------------------------------
# SCDM gauge in the AO-column discretization
# ---------------------------------------------------------------------------

def scdm_anchors(
    B: Sequence[np.ndarray],
    num_wann: int,
    gamma_index: int = 0,
    f_weights: Optional[Sequence[np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """k-independent AO anchor columns by QRCP on the subspace density matrix.

    LCAO-native SCDM: in plane-wave codes the density-matrix columns live on a
    real-space grid; here the natural localized discretization is the AO index
    itself. The pivoted column rho[:, mu] is a localized function lying exactly
    in the span of the subspace, so the trials are guaranteed well-conditioned;
    fixed (k-independent) anchors make the induced gauge smooth in k — the two
    SCDM properties that matter for the initial gauge.

    Parameters
    ----------
    B : per-k (nao, num_wann) AO coefficients of the (disentangled) subspace
        states in a fixed localized frame.
    f_weights : optional per-k (num_wann,) occupation weights f(eps) for the
        density accumulation (the Damle-Lin entangled-SCDM weighting, same
        role as the erfc weighting in scdm_select_projections). Biases the
        anchor CHOICE and the rho-column trials toward the physically
        important states; the gauge rows B(k)^H stay unweighted so the
        conditioning of A is not degraded.

    Returns
    -------
    anchors : (num_wann,) int — selected AO indices.
    A : (nk, num_wann, num_wann) initial gauge, A[k]_{mn} =
        conj(B[k][anchor_n, m]) — the "value" of state m at anchor n.
    info : dict with the winning candidate's conditioning, so callers can
        gate the emitted gauge without a second O(nk) SVD sweep:
        'candidate' (anchor-set/gauge name), 'sigma_min', 'sigma_max',
        'cond' (= sigma_max/sigma_min, scale-free), 'argmin_k' (the worst
        k-point). NOTE on scales: sigma_min is NOT comparable across
        systems — measured here it spans 0.39 (Bi, cond 4) to 0.036 (h-BN,
        cond 110), and production logs show champion runs at 0.001-0.002
        (h-BN, SnTe) and below 5e-4 (Sc, 294 WFs) that nevertheless
        interpolate to 0.25-92 meV. Gate on 'cond' or on rank deficiency,
        not on an absolute sigma_min ported from another code.
    """
    from scipy.linalg import qr

    nao = B[0].shape[0]
    rho = np.zeros((nao, nao), complex)
    for k, Bk in enumerate(B):
        if f_weights is not None:
            rho += (Bk * np.asarray(f_weights[k])[None, :]) @ Bk.conj().T
        else:
            rho += Bk @ Bk.conj().T
    rho /= len(B)
    if f_weights is not None:
        rho = 0.5 * (rho + rho.conj().T)   # weights are real; kill roundoff

    def _gauge(anchors):
        # Damle-Lin trials are density-matrix COLUMNS, not point evaluations:
        # A(k) = B(k)^H rho[:, anchors]. A point-row gauge B[anchors]^H goes
        # exactly singular when a subspace state has zero amplitude on every
        # anchor AO at some k (h-BN high-symmetry k); the rho-column trial is
        # a full function in the manifold's span, so it stays conditioned.
        return np.stack([Bk.conj().T @ rho[:, anchors] for Bk in B])

    def _sv_stats(A):
        # one SVD pass per k, reused for both the bake-off comparison and
        # the returned conditioning report (the pipeline used to repeat
        # this whole sweep just to print sigma_min).
        sv = np.array([np.linalg.svd(A[k], compute_uv=False)
                       for k in range(A.shape[0])])
        kmin = int(np.argmin(sv[:, -1]))
        return float(sv[:, -1].min()), float(sv[:, 0].max()), kmin

    def _gauge_row(anchors):
        # textbook Damle-Lin point-evaluation gauge: A[k]_{mn} =
        # conj(B[k][anchor_n, m]). Goes singular when a subspace state
        # vanishes on every anchor row — viable only when the
        # discretization rows genuinely cover the manifold (e.g. MT
        # channels augmented with interstitial points).
        return np.stack([Bk[anchors, :].T.conj() for Bk in B])

    # Pivot candidates: QRCP of Psi^H at Gamma (textbook SCDM-k — the
    # smooth-gauge anchor set) and QRCP of the k-averaged density matrix.
    # Gauge candidates: rho-column trials (robust when the k-averaged
    # density spans everything) and point-row trials (robust for
    # k-specific mixtures the averaged density misses). Keep whichever
    # combination has the best worst-k conditioning.
    cands = {}
    _q, _r, piv_g = qr(B[gamma_index].conj().T, pivoting=True)
    cands['gamma'] = np.sort(np.asarray(piv_g[:num_wann], dtype=int))
    _q, _r, piv_r = qr(rho, pivoting=True)
    cands['rho'] = np.sort(np.asarray(piv_r[:num_wann], dtype=int))

    best_name, best_anchors, best_A, best_sv = None, None, None, -1.0
    best_smax, best_kmin = 1.0, 0
    for name, anchors in cands.items():
        for gname, gfun in (('col', _gauge), ('row', _gauge_row)):
            A = gfun(anchors)
            sv, smax, kmin = _sv_stats(A)
            if sv > best_sv:
                best_name, best_anchors, best_A, best_sv = (
                    f'{name}/{gname}', anchors, A, sv)
                best_smax, best_kmin = smax, kmin
    info = {'candidate': best_name, 'sigma_min': best_sv,
            'sigma_max': best_smax, 'argmin_k': best_kmin,
            'cond': (best_smax / best_sv if best_sv > 0.0 else np.inf)}
    return best_anchors, best_A, info


# ---------------------------------------------------------------------------
# Rotation into the projected-Hamiltonian eigenbasis (the hand-off basis)
# ---------------------------------------------------------------------------

def rotate_to_subspace(
    eig_pool: np.ndarray,
    V: np.ndarray,
    mmn: np.ndarray,
    neigh: np.ndarray,
    amn: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Diagonalize V^H H V per k and rotate overlaps/projections along.

    wannier90 treats .eig entries as the energies of the states the .mmn
    refers to, so the hand-off basis must be the eigenbasis of the projected
    Hamiltonian. H is diagonal in the pool band basis: H(k) = diag(eig_pool).

    Parameters
    ----------
    eig_pool : (nk, nb) band energies of the pool bands (eV).
    V : (nk, nb, nw) semi-unitary subspace frames.
    mmn : (nk, nntot, nb, nb) pool-basis overlaps.
    amn : optional (nk, nw, nw) gauge matrices ALREADY in the subspace frame
          (e.g. from scdm_anchors on B = C_pool V); rotated as R^H A.

    Returns
    -------
    eig_tilde : (nk, nw) disentangled band energies (ascending per k).
    W : (nk, nb, nw) — V R, the full band-basis -> hand-off transforms.
    mmn_rot : (nk, nntot, nw, nw) — W(k)^H M W(k+b).
    amn_rot : (nk, nw, nw) or None — R(k)^H A(k).
    """
    nk, nb, nw = V.shape
    eig_tilde = np.empty((nk, nw))
    W = np.empty_like(V)
    R_all = np.empty((nk, nw, nw), complex)
    for k in range(nk):
        H_sub = V[k].conj().T @ (eig_pool[k][:, None] * V[k])
        H_sub = 0.5 * (H_sub + H_sub.conj().T)
        evals, R = np.linalg.eigh(H_sub)
        eig_tilde[k] = evals
        R_all[k] = R
        W[k] = V[k] @ R
    Wn = W[neigh]                                        # (nk, nntot, nb, nw)
    MW = np.einsum("kiab,kibw->kiaw", mmn, Wn)
    mmn_rot = np.einsum("kav,kiaw->kivw", W.conj(), MW)
    amn_rot = None
    if amn is not None:
        amn_rot = np.einsum("kvm,kmn->kvn", R_all.conj().transpose(0, 2, 1), amn)
    return eig_tilde, W, mmn_rot, amn_rot
