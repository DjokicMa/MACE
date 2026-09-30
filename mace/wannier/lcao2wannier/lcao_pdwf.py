# Copyright (c) 2025 William Comaskey. MIT License, see LICENSE in this directory.
# Vendored into MACE from lcao2wannier 1.0.0; local changes are listed in VENDORED.md.
"""
LCAO-PDWF: Projectability-Disentangled Wannier Functions for LCAO bases.

Adapts the PDWF method (Qiao, Pizzi, Marzari, npj Comput. Mater. 9, 208, 2023)
to the non-orthogonal LCAO setting by computing Lowdin projectability onto a
chemistry-defined target orbital subset.

Key differences from plane-wave PDWF:
  - Target orbitals are the first radial function of each valence l-channel
    (determined by valence_config.py), rather than PAOs from pseudopotentials.
  - Projectability is computed in the Lowdin-orthogonalized basis using S^{1/2},
    not via pseudopotential projectors.
  - The LCAO basis naturally provides a finite, localized expansion, so the
    target/complement decomposition is exact (not truncated).
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ClassificationParams:
    """Parameters for PDWF band classification."""
    p_high: float = 0.95       # Above this: FROZEN
    p_low: float = 0.10        # Below this: EXCLUDED
    e_fermi: float = 0.0       # Fermi energy (eV)
    e_above_max: float = 20.0  # Max energy above E_F to consider (eV)
    e_below_min: float = -30.0 # Min energy below E_F to consider (eV)


@dataclass
class BandClassification:
    """Results from PDWF band classification."""
    category: np.ndarray            # shape (num_bands,): 'frozen'/'disent'/'excluded'
    avg_projectability: np.ndarray  # shape (num_bands,)
    min_projectability: np.ndarray  # shape (num_bands,)
    band_energies: np.ndarray       # shape (num_bands,): k-averaged eigenvalues
    frozen_indices: np.ndarray
    disent_indices: np.ndarray
    excluded_indices: np.ndarray
    num_wann: int
    gap_quality: float


@dataclass
class WindowParameters:
    """Wannier90 disentanglement window parameters."""
    num_wann: int
    dis_froz_min: Optional[float] = None
    dis_froz_max: Optional[float] = None
    dis_win_min: Optional[float] = None
    dis_win_max: Optional[float] = None
    exclude_bands: Optional[List[int]] = None  # 1-based Wannier90 convention


def compute_matrix_sqrt(S: np.ndarray, threshold: float = 1e-8) -> np.ndarray:
    """
    Compute S^{1/2} via eigendecomposition with ill-conditioning protection.

    Parameters
    ----------
    S : ndarray, shape (n, n)
        Hermitian positive semi-definite matrix.
    threshold : float
        Eigenvalues below this are clamped to avoid numerical instability.

    Returns
    -------
    S_half : ndarray, shape (n, n)
    """
    eigvals, eigvecs = np.linalg.eigh(S)
    eigvals = np.maximum(eigvals, threshold)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.conj().T


def compute_lowdin_projectability(
    C_k_list: List[np.ndarray],
    S_k_list: List[np.ndarray],
    target_mask: np.ndarray,
    S_half_cache: Optional[List[np.ndarray]] = None,
) -> np.ndarray:
    """
    Compute Lowdin projectability onto target orbital subset.

    For band n at k-point k:
        p_nk = sum_{mu in T} |C_tilde_{mu,n}(k)|^2

    where C_tilde(k) = S^{1/2}(k) C(k) is the Lowdin-transformed
    eigenvector matrix and T is the target orbital index set.

    Properties:
      - C_tilde^dag C_tilde = C^dag S C = I, so sum_mu |C_tilde_{mu,n}|^2 = 1
      - Therefore p_nk in [0, 1] exactly
      - p_nk = 1 means band n lives entirely in the target manifold

    Parameters
    ----------
    C_k_list : list of ndarray, shape (num_ao, num_bands)
        Eigenvectors at each k-point.
    S_k_list : list of ndarray, shape (num_ao, num_ao)
        Overlap matrices at each k-point.
    target_mask : ndarray, shape (num_ao,), dtype=bool
        True for target orbitals.
    S_half_cache : optional list of ndarray
        Pre-computed S^{1/2}(k) matrices.

    Returns
    -------
    proj : ndarray, shape (num_kpoints, num_bands)
        Projectability in [0, 1] for each band at each k-point.
    """
    nk = len(C_k_list)
    nb = C_k_list[0].shape[1]
    proj = np.zeros((nk, nb))

    for ik in range(nk):
        if S_half_cache is not None:
            S_half = S_half_cache[ik]
        else:
            S_half = compute_matrix_sqrt(S_k_list[ik])

        # Lowdin-transformed eigenvectors
        C_tilde = S_half @ C_k_list[ik]  # (num_ao, num_bands)

        # Projectability = sum of |C_tilde|^2 over target rows
        proj[ik, :] = np.sum(np.abs(C_tilde[target_mask, :])**2, axis=0)

    return proj


def classify_bands(
    proj: np.ndarray,
    eigenvalues: np.ndarray,
    num_wann: int,
    params: Optional[ClassificationParams] = None,
) -> BandClassification:
    """
    Classify bands into frozen/disentangle/excluded.

    Stage 1 (projectability-based):
      - avg_p >= p_high  -->  FROZEN candidate
      - avg_p <  p_low   -->  EXCLUDED
      - otherwise         -->  DISENTANGLE candidate

    Stage 2 (energy-based cleanup):
      - Remove FROZEN/DISENT bands outside energy range
      - Gap-bridge: promote DISENT bands sandwiched between FROZEN bands
        if their projectability is within 0.05 of p_high.

    Parameters
    ----------
    proj : ndarray, shape (nk, nb)
        Per-band per-k projectability from compute_lowdin_projectability.
    eigenvalues : ndarray, shape (nk, nb)
        Eigenvalues at each k-point in eV.
    num_wann : int
        Target number of Wannier functions (from valence config).
    params : ClassificationParams, optional
        Thresholds. If None, defaults are used.

    Returns
    -------
    BandClassification
    """
    if params is None:
        params = ClassificationParams()

    avg_p = np.mean(proj, axis=0)
    min_p = np.min(proj, axis=0)
    avg_e = np.mean(eigenvalues, axis=0)
    e_rel = avg_e - params.e_fermi

    nb = len(avg_p)
    category = np.full(nb, 'excluded', dtype='U10')

    # Stage 1: projectability thresholds
    category[avg_p >= params.p_high] = 'frozen'
    category[(avg_p >= params.p_low) & (avg_p < params.p_high)] = 'disent'

    # Stage 2: energy window cleanup
    too_low = e_rel < params.e_below_min
    too_high = e_rel > params.e_above_max
    category[too_low | too_high] = 'excluded'

    # Gap-bridging: if DISENT bands sit between FROZEN bands and have
    # projectability close to p_high, promote them to FROZEN
    frozen_mask = category == 'frozen'
    if np.any(frozen_mask):
        f_indices = np.where(frozen_mask)[0]
        f_min, f_max = f_indices[0], f_indices[-1]
        for b in range(f_min, f_max + 1):
            if category[b] == 'disent' and avg_p[b] >= params.p_high - 0.05:
                category[b] = 'frozen'

    # Compute gap quality
    frozen_mask = category == 'frozen'
    excluded_mask = category == 'excluded'
    if np.any(frozen_mask) and np.any(excluded_mask):
        gap_quality = float(np.min(avg_p[frozen_mask]) /
                           (np.max(avg_p[excluded_mask]) + 1e-10))
    else:
        gap_quality = 0.0

    return BandClassification(
        category=category,
        avg_projectability=avg_p,
        min_projectability=min_p,
        band_energies=avg_e,
        frozen_indices=np.where(category == 'frozen')[0],
        disent_indices=np.where(category == 'disent')[0],
        excluded_indices=np.where(category == 'excluded')[0],
        num_wann=num_wann,
        gap_quality=gap_quality,
    )


def auto_classification_params(
    proj: np.ndarray,
    eigenvalues: np.ndarray,
    num_wann: int,
    e_fermi: float = 0.0,
    fid_emax: float = 8.0,
    margin: float = 0.015,
    thr_floor: float = 0.55,
):
    """Derive the PDWF window thresholds from the projectability distribution.

    The same rule the hybrid method uses for its per-state masks
    (``hybrid.auto_trust_thresholds``) fixes the two thresholds of the window
    route, so both automatic routes read the same quantities:

      p_high = the band gate (gray-plateau top + margin): bands whose
               band-averaged projectability clears it are FROZEN
      p_low  = the junk cliff: below the gate, the largest downward jump in
               the sorted band-averaged projectabilities separates the gray
               plateau (kept in the outer window, disentangled) from junk
               (EXCLUDED). If no jump of at least 0.05 exists, half the gate.

    Returns ``(ClassificationParams, info)``; ``(None, None)`` when the rule
    cannot decide (no irrelevant band to place the plateau, or nothing
    trusted above the gate), in which case the caller falls back to the
    fixed defaults and says so.
    """
    from .hybrid import auto_trust_thresholds
    proj = np.asarray(proj, float)
    eigenvalues = np.asarray(eigenvalues, float)
    auto = auto_trust_thresholds(proj, eigenvalues, num_wann, e_fermi=e_fermi,
                                 fid_emax=fid_emax, margin=margin,
                                 thr_floor=thr_floor)
    if auto is None:
        return None, None
    gate = float(auto['p_froz_band'])
    avg_p = proj.mean(axis=0)
    pool = np.asarray(auto['pool'], int)
    below = np.sort(avg_p[pool][avg_p[pool] < gate])[::-1]
    p_low, cliff = None, False
    if below.size >= 2:
        jumps = below[:-1] - below[1:]
        j = int(np.argmax(jumps))
        if jumps[j] >= 0.05:
            p_low, cliff = float(0.5 * (below[j] + below[j + 1])), True
    if p_low is None:
        p_low = max(0.05, 0.5 * gate)
    params = ClassificationParams(p_high=gate, p_low=p_low, e_fermi=e_fermi)
    info = dict(p_high=gate, p_low=p_low, plateau=float(auto['plateau_top']),
                junk_cliff=cliff, n_pool=int(pool.size),
                n_trusted=int(np.asarray(auto['trusted']).size))
    return params, info


def determine_windows(
    classification: BandClassification,
    eigenvalues: np.ndarray,
    e_fermi: float,
    padding_froz: float = 0.5,
    padding_win: float = 2.0,
) -> WindowParameters:
    """
    Determine Wannier90 disentanglement window parameters.

    The frozen window encompasses all eigenvalues (across ALL k-points)
    of frozen bands, plus padding. The outer window encompasses all
    eigenvalues of frozen + disentangle bands, plus padding.

    Parameters
    ----------
    classification : BandClassification
        Band classification results.
    eigenvalues : ndarray, shape (nk, nb)
        Eigenvalues at each k-point in eV.
    e_fermi : float
        Fermi energy in eV.
    padding_froz : float
        eV padding on frozen window edges.
    padding_win : float
        eV padding on outer window edges.

    Returns
    -------
    WindowParameters
    """
    frozen = classification.frozen_indices
    disent = classification.disent_indices
    all_active = np.union1d(frozen, disent)

    # Frozen window
    if len(frozen) > 0:
        froz_eigs = eigenvalues[:, frozen]
        dis_froz_min = float(np.min(froz_eigs) - padding_froz)
        dis_froz_max = float(np.max(froz_eigs) + padding_froz)
    else:
        dis_froz_min = None
        dis_froz_max = None

    # Outer window (only needed if there are disentanglement bands)
    if len(all_active) > 0 and len(disent) > 0:
        active_eigs = eigenvalues[:, all_active]
        dis_win_min = float(np.min(active_eigs) - padding_win)
        dis_win_max = float(np.max(active_eigs) + padding_win)
    else:
        dis_win_min = None
        dis_win_max = None

    # Exclude bands (1-based for Wannier90)
    excluded = classification.excluded_indices
    exclude_bands = (excluded + 1).tolist() if len(excluded) > 0 else None

    return WindowParameters(
        num_wann=classification.num_wann,
        dis_froz_min=dis_froz_min,
        dis_froz_max=dis_froz_max,
        dis_win_min=dis_win_min,
        dis_win_max=dis_win_max,
        exclude_bands=exclude_bands,
    )


def check_frozen_interlopers(
    eigenvalues: np.ndarray,
    frozen_indices: np.ndarray,
    excluded_indices: np.ndarray,
    dis_froz_min: float,
    dis_froz_max: float,
) -> List[str]:
    """
    Check for excluded bands with eigenvalues inside the frozen window.

    Returns list of warning strings (empty if clean).
    """
    warnings = []
    for band in excluded_indices:
        eigs = eigenvalues[:, band]
        inside = (eigs >= dis_froz_min) & (eigs <= dis_froz_max)
        if np.any(inside):
            n_kpts = int(np.sum(inside))
            e_range = (float(np.min(eigs[inside])), float(np.max(eigs[inside])))
            warnings.append(
                f"Band {band}: excluded but has eigenvalues inside frozen "
                f"window at {n_kpts} k-points (E = {e_range[0]:.4f} to "
                f"{e_range[1]:.4f} eV). Consider promoting to disentanglement."
            )
    return warnings


def check_band_count(
    eigenvalues: np.ndarray,
    dis_win_min: float,
    dis_win_max: float,
    num_wann: int,
) -> List[str]:
    """
    Verify that every k-point has at least num_wann bands in the outer window.

    Returns list of warning strings (empty if clean).
    """
    warnings = []
    nk = eigenvalues.shape[0]
    for ik in range(nk):
        n_inside = int(np.sum(
            (eigenvalues[ik, :] >= dis_win_min) &
            (eigenvalues[ik, :] <= dis_win_max)
        ))
        if n_inside < num_wann:
            warnings.append(
                f"k-point {ik}: only {n_inside} bands in outer window, "
                f"need >= {num_wann}. Outer window must be expanded."
            )
    return warnings


def print_pdwf_summary(
    classification: BandClassification,
    windows: WindowParameters,
    eigenvalues: np.ndarray,
    e_fermi: float,
    validation_warnings: List[str],
) -> str:
    """Print a diagnostic summary of the PDWF classification."""
    lines = []
    lines.append("")
    lines.append("=" * 70)
    lines.append("LCAO-PDWF BAND CLASSIFICATION")
    lines.append("=" * 70)
    lines.append(f"Fermi energy: {e_fermi:.4f} eV")
    lines.append(f"num_wann (from valence config): {classification.num_wann}")
    lines.append(f"Gap quality: {classification.gap_quality:.2f}"
                 f" ({'clean' if classification.gap_quality > 5 else 'moderate' if classification.gap_quality > 2 else 'POOR'})")
    lines.append("")

    # Per-band table
    lines.append(f"  {'Band':>6s}  {'Avg p':>8s}  {'Min p':>8s}  {'E_avg':>10s}  {'E_rel':>8s}  {'Category'}")
    lines.append(f"  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*15}")
    for i in range(len(classification.avg_projectability)):
        cat = classification.category[i].upper()
        e_rel = classification.band_energies[i] - e_fermi
        lines.append(
            f"  {i:6d}  {classification.avg_projectability[i]:8.4f}  "
            f"{classification.min_projectability[i]:8.4f}  "
            f"{classification.band_energies[i]:10.4f}  {e_rel:8.4f}  {cat}"
        )

    lines.append("")
    lines.append(f"Frozen bands: {len(classification.frozen_indices)}")
    lines.append(f"Disentangle bands: {len(classification.disent_indices)}")
    lines.append(f"Excluded bands: {len(classification.excluded_indices)}")

    if windows.dis_froz_min is not None:
        lines.append(f"\nFrozen window: [{windows.dis_froz_min:.4f}, {windows.dis_froz_max:.4f}] eV")
    if windows.dis_win_min is not None:
        lines.append(f"Outer window:  [{windows.dis_win_min:.4f}, {windows.dis_win_max:.4f}] eV")

    if validation_warnings:
        lines.append(f"\nWARNINGS ({len(validation_warnings)}):")
        for w in validation_warnings:
            lines.append(f"  {w}")

    lines.append("=" * 70)

    result = '\n'.join(lines)
    print(result)
    return result
