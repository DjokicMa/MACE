# Copyright (c) 2025 William Comaskey. MIT License, see LICENSE in this directory.
# Vendored into MACE from lcao2wannier 1.0.0; local changes are listed in VENDORED.md.
"""
Projectability-based band selection (PDWF-inspired).

Computes per-band projectability onto the LCAO basis to determine
which Bloch bands are well-described by the local orbital basis.
Bands with high projectability are kept; low-projectability bands
(typically numerically ill-conditioned high-energy solutions) are discarded.

Reference: Qiao, Pizzi, Marzari, npj Comput. Mater. 9, 206 (2023)
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ProjectabilityResult:
    """Results from projectability-based band selection."""
    band_projectabilities: np.ndarray     # shape (num_bands,): avg projectability per band
    selected_band_indices: np.ndarray     # indices of bands above threshold
    rejected_band_indices: np.ndarray     # indices of bands below threshold
    num_wann: int                         # = len(selected_band_indices)
    threshold: float                      # threshold used
    projectability_per_kpoint: np.ndarray # shape (num_kpoints, num_bands) for diagnostics


@dataclass
class SmartSelectionResult:
    """Results from smart projectability-based band selection."""
    selected_band_indices: np.ndarray     # final selected indices (contiguous, even)
    num_wann: int                         # = len(selected_band_indices)
    band_projectabilities: np.ndarray     # shape (num_bands,): avg projectability per band
    band_relevance: np.ndarray            # shape (num_bands,): combined relevance score
    band_energies: np.ndarray             # shape (num_bands,): avg band center energies
    quality_score: float                  # min relevance over selected set (0-1 scale)
    e_fermi: float                        # Fermi energy used
    has_soc: bool                         # whether Kramers pairing was enforced
    energy_range: Tuple[float, float]     # (E_min, E_max) of selected bands
    # Diagnostics
    proj_threshold_used: float
    bands_removed_core: int               # how many deep core bands trimmed
    bands_removed_high: int               # how many high-energy bands trimmed
    bands_removed_noncontiguous: int      # how many outlier bands trimmed
    # Frontier detection / disentanglement
    frontier_band_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    num_frontier: int = 0                 # = len(frontier_band_indices)
    recommended_num_wann: int = 0         # = num_frontier (Kramers-enforced)
    recommended_dis_win: Optional[Tuple[float, float]] = None   # outer window (rel E_F)
    recommended_dis_froz: Optional[Tuple[float, float]] = None  # frozen window (rel E_F)


def _clamp_frozen_window_kresolved(eig_sel_rel, num_wann, froz):
    """Shrink a frozen window so the per-k count rule holds across the BZ sample.

    Wannier90 requires that at EVERY k-point at most num_wann of the selected
    bands lie inside the frozen (inner) window. A window sized to the frontier
    bands' energy span still traps interloper bands that disperse into that range
    at some k-points (e.g. steep bands at the zone center), so the naive window
    can hold far more than num_wann bands at a single k.

    This trims whichever edge most reduces the worst-case per-k count (preserving
    the Fermi-level region) until the rule is satisfied.

    Parameters
    ----------
    eig_sel_rel : ndarray (nk, n_selected)
        Selected-band energies at each k-point, relative to E_F.
    num_wann : int
        Target number of Wannier functions (max bands allowed per k in window).
    froz : (float, float)
        Candidate (froz_min, froz_max), relative to E_F.

    Returns
    -------
    (float, float) satisfying the rule, the input unchanged if already valid,
    or None if no non-empty window can hold <= num_wann bands at every k.
    """
    lo, hi = froz
    n_sel = eig_sel_rel.shape[1]
    for _ in range(n_sel + 1):
        inside = (eig_sel_rel >= lo) & (eig_sel_rel <= hi)
        if int(inside.sum(axis=1).max()) <= num_wann:
            return (lo, hi)
        vals = eig_sel_rel[inside]
        if vals.size == 0:
            return (lo, hi)
        new_hi = float(vals.max()) - 1e-6
        new_lo = float(vals.min()) + 1e-6
        worst_hi = int(((eig_sel_rel >= lo) & (eig_sel_rel <= new_hi)).sum(axis=1).max())
        worst_lo = int(((eig_sel_rel >= new_lo) & (eig_sel_rel <= hi)).sum(axis=1).max())
        if worst_hi <= worst_lo:
            hi = new_hi
        else:
            lo = new_lo
        if hi <= lo:
            return None
    return (lo, hi)


def smart_select_bands(
    eigenvectors_list: List[np.ndarray],
    S_k_list: List[np.ndarray],
    eigenvalues_list: List[np.ndarray],
    e_fermi: float,
    has_soc: bool = False,
    proj_threshold: float = 0.9,
    energy_sigma: float = 3.0,
    core_cutoff: float = -15.0,
    frontier_threshold: float = 0.4,
    verbose: bool = True,
    target_mask: Optional[np.ndarray] = None,
) -> SmartSelectionResult:
    """
    Smart projectability-based band selection with Fermi-proximity weighting.

    Algorithm:
      1. Compute per-band projectability (averaged over k-points)
      2. Compute band center energies (averaged over k-points, relative to E_F)
      3. Compute relevance = projectability * exp(-|E_center - E_F| / sigma)
      4. Filter: keep bands with projectability >= threshold
      5. Remove deep core bands (E_center < core_cutoff below E_F)
      6. Find largest contiguous block of surviving bands around E_F
      7. Enforce Kramers pairing (even count) for SOC systems
      8. Frontier detection: split selected bands into frontier (high relevance)
         and support (low relevance) for disentanglement
      9. Report quality_score = min(relevance) over selected set

    Parameters
    ----------
    eigenvectors_list : list of ndarray, each (num_orbitals, num_bands)
        Eigenvectors at each k-point.
    S_k_list : list of ndarray, each (num_orbitals, num_orbitals)
        Overlap matrices at each k-point.
    eigenvalues_list : list of ndarray, each (num_bands,)
        Eigenvalues at each k-point in eV.
    e_fermi : float
        Fermi energy in eV.
    has_soc : bool
        If True, enforce even num_wann (Kramers pairs).
    proj_threshold : float
        Minimum average projectability to keep a band (default: 0.9).
    energy_sigma : float
        Energy scale (eV) for Fermi-proximity Gaussian weighting (default: 3.0).
    core_cutoff : float
        Bands with E_center more than this many eV below E_F are considered
        deep core and trimmed (default: -15.0 eV).
    frontier_threshold : float
        Minimum relevance score to classify a band as "frontier" (default: 0.4).
        Frontier bands become num_wann; remaining selected bands are support
        bands for disentanglement.
    verbose : bool
        Print diagnostic information.

    Returns
    -------
    SmartSelectionResult
    """
    num_kpoints = len(eigenvectors_list)
    num_bands = eigenvectors_list[0].shape[1]

    # Step 1: Compute projectability
    _check_target_mask(target_mask, eigenvectors_list[0].shape[0],
                       proj_threshold)
    proj_per_kpoint = compute_band_projectability(
        eigenvectors_list, S_k_list, target_mask=target_mask)
    avg_proj = np.mean(proj_per_kpoint, axis=0)  # shape (num_bands,)

    # Step 2: Compute band center energies relative to E_F
    all_eigs = np.array([eigs[:num_bands] for eigs in eigenvalues_list])  # (nk, nb)
    band_energies = np.mean(all_eigs, axis=0)  # (num_bands,)
    band_energies_rel = band_energies - e_fermi  # relative to Fermi

    # Step 3: Compute relevance = projectability * Gaussian(distance from E_F)
    fermi_weight = np.exp(-np.abs(band_energies_rel) / energy_sigma)
    band_relevance = avg_proj * fermi_weight

    # Step 4: Filter by projectability threshold
    passes_proj = avg_proj >= proj_threshold
    n_total_passing = np.sum(passes_proj)

    # Step 5: Remove deep core bands
    passes_core = band_energies_rel >= core_cutoff
    bands_removed_core = int(np.sum(passes_proj & ~passes_core))

    # Combined mask: passes projectability AND not deep core
    candidate_mask = passes_proj & passes_core
    candidate_indices = np.where(candidate_mask)[0]

    if len(candidate_indices) == 0:
        # Fallback: if nothing passes, just take all projectable bands
        candidate_indices = np.where(passes_proj)[0]
        bands_removed_core = 0

    # Step 6: Find largest contiguous block around E_F
    # Find the band closest to E_F among candidates
    if len(candidate_indices) > 0:
        fermi_band = candidate_indices[
            np.argmin(np.abs(band_energies_rel[candidate_indices]))
        ]

        # Find contiguous runs in candidate_indices
        contiguous_blocks = _find_contiguous_blocks(candidate_indices)

        # Pick the block that contains the Fermi-level band
        best_block = None
        for block in contiguous_blocks:
            if fermi_band in block:
                best_block = block
                break

        # Fallback: pick largest block
        if best_block is None:
            best_block = max(contiguous_blocks, key=len)

        bands_removed_noncontiguous = len(candidate_indices) - len(best_block)
        selected = np.array(best_block)
    else:
        selected = np.array([], dtype=int)
        bands_removed_noncontiguous = 0

    # Count high-energy bands removed (above the contiguous block)
    bands_removed_high = int(np.sum(passes_proj)) - bands_removed_core - len(selected) - bands_removed_noncontiguous
    bands_removed_high = max(0, bands_removed_high)  # safety

    # Step 7: Enforce Kramers pairing for SOC (even band count)
    if has_soc and len(selected) % 2 != 0:
        # Drop the band with lowest relevance at either edge
        if len(selected) > 1:
            rel_first = band_relevance[selected[0]]
            rel_last = band_relevance[selected[-1]]
            if rel_first < rel_last:
                selected = selected[1:]  # drop lowest-energy edge
            else:
                selected = selected[:-1]  # drop highest-energy edge
        else:
            selected = np.array([], dtype=int)

    # Step 8: Frontier detection for disentanglement
    if len(selected) > 0:
        frontier_mask = band_relevance[selected] >= frontier_threshold
        frontier_indices = selected[frontier_mask]

        # Enforce Kramers on frontier count too
        if has_soc and len(frontier_indices) % 2 != 0:
            if len(frontier_indices) > 1:
                # Drop the frontier band with lowest relevance at either edge
                rel_first = band_relevance[frontier_indices[0]]
                rel_last = band_relevance[frontier_indices[-1]]
                if rel_first < rel_last:
                    frontier_indices = frontier_indices[1:]
                else:
                    frontier_indices = frontier_indices[:-1]
            else:
                frontier_indices = np.array([], dtype=int)

        num_frontier = len(frontier_indices)
        recommended_num_wann = num_frontier if num_frontier > 0 else len(selected)

        # Auto-compute disentanglement windows (relative to E_F)
        if num_frontier < len(selected) and num_frontier > 0:
            # Use actual eigenvalue range across ALL k-points (not just band centers)
            # to ensure the window captures every selected eigenvalue at every k-point
            e_block_min_rel = float(np.min(all_eigs[:, selected]) - e_fermi)
            e_block_max_rel = float(np.max(all_eigs[:, selected]) - e_fermi)
            e_front_min_rel = float(np.min(all_eigs[:, frontier_indices]) - e_fermi)
            e_front_max_rel = float(np.max(all_eigs[:, frontier_indices]) - e_fermi)

            # Frozen window: start from the frontier energy span with adaptive padding.
            frontier_bandwidth = e_front_max_rel - e_front_min_rel
            froz_padding = max(0.5, 0.1 * frontier_bandwidth)
            froz_lo_rel = e_front_min_rel - froz_padding
            froz_hi_rel = e_front_max_rel + froz_padding

            if froz_hi_rel > froz_lo_rel:
                recommended_dis_froz = (froz_lo_rel, froz_hi_rel)
            else:
                # Bands too dense around E_F to freeze any -> pure disentanglement.
                recommended_dis_froz = None

            # Outer window: wide, covering all selected bands at all k (+ gap beyond
            # the frozen window when present), capped at the global eigenvalue range.
            e_global_min_rel = float(np.min(all_eigs) - e_fermi)
            e_global_max_rel = float(np.max(all_eigs) - e_fermi)
            min_gap = 1.0  # eV minimum gap between frozen and outer boundaries
            win_lo = e_block_min_rel - 0.5
            win_hi = e_block_max_rel + 0.5
            if recommended_dis_froz is not None:
                win_lo = min(win_lo, recommended_dis_froz[0] - min_gap)
                win_hi = max(win_hi, recommended_dis_froz[1] + min_gap)
            win_lo = max(win_lo, e_global_min_rel - 1.0)
            win_hi = min(win_hi, e_global_max_rel + 1.0)
            recommended_dis_win = (win_lo, win_hi)

            # Balanced frozen-window policy (k-resolved across the BZ sample).
            # The frozen energy window spans the frontier bands, but interloper
            # bands disperse into that range at some k-points (steep bands at the
            # zone center), so the per-k count can exceed num_frontier and
            # wannier90 aborts. Rather than aggressively trimming the window
            # (which would un-freeze good upper frontier bands), we GROW num_wann
            # to admit those interlopers as Wannier functions -- but only up to a
            # projectability cap (selected bands that pass proj_threshold), so we
            # never promote unprojectable junk. If the required count exceeds the
            # cap, we grow to the cap and then shrink the window to fit. The outer
            # window above is unaffected (the frozen window stays a subset).
            eig_sel_rel = all_eigs[:, selected] - e_fermi
            froz_counts = ((eig_sel_rel >= recommended_dis_froz[0]) &
                           (eig_sel_rel <= recommended_dis_froz[1])).sum(axis=1)
            max_in_froz = int(froz_counts.max())
            if max_in_froz > num_frontier:
                # Cap: only bands we trust as WFs (pass the projectability filter).
                projectable_cap = int(np.sum(avg_proj[selected] >= proj_threshold))
                cap = max(num_frontier, projectable_cap)
                target_nw = min(max_in_froz, cap)
                if target_nw >= max_in_froz:
                    # Keep the full frozen window; raise num_wann to admit interlopers.
                    if verbose:
                        print(f"  [frozen-window] keeping window "
                              f"[{recommended_dis_froz[0]:.3f}, {recommended_dis_froz[1]:.3f}] eV; "
                              f"raising num_wann {num_frontier} -> {max_in_froz} to admit "
                              f"{max_in_froz - num_frontier} interloper band(s) "
                              f"(all pass proj >= {proj_threshold:g})")
                    recommended_num_wann = max_in_froz
                else:
                    # Projectability-capped: grow to the cap, then shrink window.
                    recommended_num_wann = target_nw
                    _clamped = _clamp_frozen_window_kresolved(
                        eig_sel_rel, target_nw, recommended_dis_froz)
                    if _clamped is None:
                        recommended_dis_froz = None
                        if verbose:
                            print(f"  [frozen-window] num_wann capped at {target_nw} "
                                  f"(projectability); no valid frozen window -> disabled")
                    else:
                        if verbose:
                            print(f"  [frozen-window] num_wann {num_frontier} -> {target_nw} "
                                  f"(projectability cap), window clamped "
                                  f"[{recommended_dis_froz[0]:.3f}, {recommended_dis_froz[1]:.3f}] -> "
                                  f"[{_clamped[0]:.3f}, {_clamped[1]:.3f}] eV")
                        recommended_dis_froz = _clamped
        else:
            # All bands are frontier — no disentanglement needed
            recommended_dis_win = None
            recommended_dis_froz = None
    else:
        frontier_indices = np.array([], dtype=int)
        num_frontier = 0
        recommended_num_wann = 0
        recommended_dis_win = None
        recommended_dis_froz = None

    # Step 9: Compute quality score
    if len(selected) > 0:
        quality_score = float(np.min(band_relevance[selected]))
        e_min = float(np.min(band_energies[selected]))
        e_max = float(np.max(band_energies[selected]))
    else:
        quality_score = 0.0
        e_min = 0.0
        e_max = 0.0

    # Print diagnostics
    if verbose:
        print(f"\n{'='*70}")
        print("SMART PROJECTABILITY-BASED BAND SELECTION")
        print(f"{'='*70}")
        print(f"Total bands: {num_bands}")
        print(f"Fermi energy: {e_fermi:.4f} eV")
        print(f"SOC (Kramers enforced): {'Yes' if has_soc else 'No'}")
        print(f"Projectability threshold: {proj_threshold:.3f}")
        print(f"Frontier threshold: {frontier_threshold:.3f}")
        print(f"Energy sigma: {energy_sigma:.1f} eV")
        print(f"Core cutoff: {core_cutoff:.1f} eV below E_F")
        print(f"\nFiltering summary:")
        print(f"  Bands passing projectability: {n_total_passing}")
        print(f"  Deep core bands removed: {bands_removed_core}")
        print(f"  Non-contiguous bands removed: {bands_removed_noncontiguous}")
        print(f"  High-energy outliers removed: {bands_removed_high}")
        print(f"  Final selected bands (num_bands): {len(selected)}")
        if len(selected) > 0:
            print(f"  Band index range: {selected[0]}-{selected[-1]}")
            print(f"  Energy range: [{e_min:.4f}, {e_max:.4f}] eV")
            print(f"  Energy range (rel E_F): [{e_min - e_fermi:.4f}, {e_max - e_fermi:.4f}] eV")
            print(f"  Quality score: {quality_score:.4f}")

        # Frontier / disentanglement summary
        print(f"\nFrontier detection:")
        print(f"  Frontier bands (num_wann): {num_frontier}")
        print(f"  Support bands: {len(selected) - num_frontier}")
        if recommended_dis_win is not None:
            print(f"  Disentanglement: ENABLED")
            print(f"    dis_win  (rel E_F): [{recommended_dis_win[0]:.4f}, {recommended_dis_win[1]:.4f}] eV")
            if recommended_dis_froz is not None:
                print(f"    dis_froz (rel E_F): [{recommended_dis_froz[0]:.4f}, {recommended_dis_froz[1]:.4f}] eV")
        else:
            print(f"  Disentanglement: NOT NEEDED (all bands are frontier)")

        # Detailed per-band table
        print(f"\nPer-band details:")
        print(f"  {'Band':>6s}  {'Proj':>8s}  {'E_center':>10s}  {'E_rel':>8s}  {'Relevance':>10s}  {'Status'}")
        print(f"  {'-'*6}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*20}")
        for i in range(num_bands):
            status_parts = []
            if i in frontier_indices:
                status_parts.append("FRONTIER")
            elif i in selected:
                status_parts.append("SUPPORT")
            elif passes_proj[i] and not passes_core[i]:
                status_parts.append("core")
            elif passes_proj[i]:
                status_parts.append("non-contiguous")
            else:
                status_parts.append("low proj")
            status = ", ".join(status_parts)
            print(f"  {i:6d}  {avg_proj[i]:8.4f}  {band_energies[i]:10.4f}  "
                  f"{band_energies_rel[i]:8.4f}  {band_relevance[i]:10.4f}  {status}")
        print(f"{'='*70}")

    return SmartSelectionResult(
        selected_band_indices=selected,
        num_wann=len(selected),
        band_projectabilities=avg_proj,
        band_relevance=band_relevance,
        band_energies=band_energies,
        quality_score=quality_score,
        e_fermi=e_fermi,
        has_soc=has_soc,
        energy_range=(e_min, e_max),
        proj_threshold_used=proj_threshold,
        bands_removed_core=bands_removed_core,
        bands_removed_high=bands_removed_high,
        bands_removed_noncontiguous=bands_removed_noncontiguous,
        frontier_band_indices=frontier_indices,
        num_frontier=num_frontier,
        recommended_num_wann=recommended_num_wann,
        recommended_dis_win=recommended_dis_win,
        recommended_dis_froz=recommended_dis_froz,
    )


def _find_contiguous_blocks(indices: np.ndarray) -> List[np.ndarray]:
    """Split sorted indices into contiguous blocks."""
    if len(indices) == 0:
        return []
    blocks = []
    current_block = [indices[0]]
    for i in range(1, len(indices)):
        if indices[i] == indices[i - 1] + 1:
            current_block.append(indices[i])
        else:
            blocks.append(np.array(current_block))
            current_block = [indices[i]]
    blocks.append(np.array(current_block))
    return blocks


def compute_band_projectability(
    eigenvectors_list: List[np.ndarray],
    S_k_list: List[np.ndarray],
    target_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Per-band projectability. Two distinct quantities, by ``target_mask``.

    **With ``target_mask`` (preferred).** The bounded Löwdin projectability
    onto the target orbital set,
    ``p_nk = sum_{mu in T} |[S^{1/2} C]_{mu n}|^2``, delegated to
    :func:`lcao_pdwf.compute_lowdin_projectability` so there is exactly one
    implementation of the bounded form. Since ``C^dag S C = I`` the full-row
    sum is 1, so any subset gives ``p_nk`` in [0, 1] and the value means
    "fraction of this state's character carried by the target orbitals".

    **Without ``target_mask`` (legacy, the historical default).**
    ``p_nk = sum_mu |[S(k) C(k)]_{mu n}|^2 = (C^dag S^2 C)_nn``. This is
    **NOT a projectability and is NOT bounded by 1**: writing
    ``C = S^{-1/2}U`` with U unitary gives ``p_n = (U^dag S U)_nn``, a
    Rayleigh quotient of S, so p lies in [lambda_min(S), lambda_max(S)] and
    — because the AOs are normalized, ``diag(S) = 1`` — the mean of p over
    bands is exactly ``trace(S)/N = 1``. Values above 1 therefore occur at
    every k whenever ``S != I``, i.e. always. What it actually measures is
    basis *compactness*: states built from tightly-overlapping AOs score
    above 1, diffuse/ghost states below. It is retained because the legacy
    selection thresholds (0.9 / 0.4 / 0.29) were calibrated against it;
    read those as compactness cuts, never as "90% of the character".

    NOTE for migration: passing a mask that covers (nearly) the whole basis
    makes the bounded form identically 1 and any threshold vacuous — in
    pure LCAO every solved band lies exactly in the basis span. A target
    mask is therefore constitutive, not a detail; see the guard in
    :func:`select_bands_by_projectability`.

    Parameters
    ----------
    eigenvectors_list : list of ndarray, each shape (num_orbitals, num_bands)
        Eigenvectors at each k-point.
    S_k_list : list of ndarray, each shape (num_orbitals, num_orbitals)
        Overlap matrices at each k-point.
    target_mask : ndarray of bool, shape (num_orbitals,), optional
        Target orbital set. When given, the bounded Löwdin form is returned.

    Returns
    -------
    ndarray, shape (num_kpoints, num_bands)
        Projectability for each band at each k-point.
    """
    if target_mask is not None:
        from .lcao_pdwf import compute_lowdin_projectability
        return compute_lowdin_projectability(
            eigenvectors_list, S_k_list, np.asarray(target_mask, dtype=bool))
    num_kpoints = len(eigenvectors_list)
    num_bands = eigenvectors_list[0].shape[1]

    projectability = np.zeros((num_kpoints, num_bands))

    for k_idx in range(num_kpoints):
        C_k = eigenvectors_list[k_idx]
        S_k = S_k_list[k_idx]

        # A(k) = S(k) @ C(k), shape (num_orbitals, num_bands)
        A_k = S_k @ C_k

        # p_n(k) = sum_m |A_mn(k)|^2 for each band n
        projectability[k_idx, :] = np.sum(np.abs(A_k)**2, axis=0)

    return projectability


def _check_target_mask(target_mask, num_orbitals, threshold):
    """Reject a target mask that makes the bounded criterion vacuous.

    In pure LCAO every solved band lies exactly in the basis span, so a mask
    covering the whole basis gives p == 1 identically and ANY threshold
    below 1 selects ALL bands. That failure is silent — a full run that
    quietly kept every band — so refuse it up front rather than emit a model
    nobody can tell is wrong.
    """
    if target_mask is None:
        return
    mask = np.asarray(target_mask, dtype=bool)
    if mask.size != num_orbitals:
        raise ValueError(
            f"target_mask has {mask.size} entries but the eigenvectors carry "
            f"{num_orbitals} orbitals. Align it to the engine dimension "
            f"first (SOC doubling and basis-parser overcount are both real; "
            f"see the trimming in _apply_method_pdwf).")
    if not mask.any():
        raise ValueError("target_mask selects no orbitals: the bounded "
                         "projectability would be 0 for every band.")
    if mask.all() and threshold < 1.0:
        raise ValueError(
            "target_mask selects the ENTIRE basis, so the bounded Löwdin "
            "projectability is identically 1 and the threshold "
            f"{threshold:g} would select every band. Supply the real target "
            "orbital set (valence configuration), or drop target_mask to "
            "use the legacy unbounded criterion deliberately.")


def select_bands_by_projectability(
    eigenvectors_list: List[np.ndarray],
    S_k_list: List[np.ndarray],
    threshold: float = 0.9,
    verbose: bool = True,
    target_mask: Optional[np.ndarray] = None,
) -> ProjectabilityResult:
    """
    Select bands based on average projectability above threshold.

    Parameters
    ----------
    eigenvectors_list : list of ndarray
        Eigenvectors at each k-point.
    S_k_list : list of ndarray
        Overlap matrices at each k-point.
    threshold : float
        Minimum average projectability to include a band (default: 0.9).
        With ``target_mask`` this is a fraction of target character; without
        it, it is a cut on the unbounded legacy compactness measure — the
        two are NOT interchangeable, see compute_band_projectability.
    verbose : bool
        Print diagnostic information.
    target_mask : ndarray of bool, optional
        Target orbital set; switches to the bounded Löwdin projectability.

    Returns
    -------
    ProjectabilityResult
    """
    _check_target_mask(target_mask, eigenvectors_list[0].shape[0], threshold)
    proj_per_kpoint = compute_band_projectability(
        eigenvectors_list, S_k_list, target_mask=target_mask)
    avg_proj = np.mean(proj_per_kpoint, axis=0)  # shape (num_bands,)

    selected = np.where(avg_proj >= threshold)[0]
    rejected = np.where(avg_proj < threshold)[0]

    if verbose:
        print(f"\n{'='*70}")
        print("PROJECTABILITY-BASED BAND SELECTION")
        print(f"{'='*70}")
        kind = ("bounded Lowdin (target)" if target_mask is not None
                else "LEGACY unbounded compactness (C^dag S^2 C)")
        print(f"Criterion: {kind}")
        print(f"Threshold: {threshold:.3f}")
        if target_mask is None:
            # make the unboundedness impossible to misread: the band-mean of
            # this quantity is exactly 1 by construction, so p > 1 is normal
            print(f"  [note] observed range [{avg_proj.min():.3f}, "
                  f"{avg_proj.max():.3f}] — this measure is NOT bounded by 1 "
                  f"(band-mean is exactly 1); the threshold is a compactness "
                  f"cut, not a fraction of orbital character")
        print(f"Total bands: {len(avg_proj)}")
        print(f"Selected bands (p >= {threshold}): {len(selected)}")
        print(f"Rejected bands (p < {threshold}): {len(rejected)}")
        if len(selected) > 0:
            print(f"Selected band indices: {selected[0]}-{selected[-1]}")
            print(f"Min projectability in selected: {avg_proj[selected].min():.6f}")
        if len(rejected) > 0:
            print(f"Max projectability in rejected: {avg_proj[rejected].max():.6f}")
        print(f"\nPer-band average projectability:")
        for i, p in enumerate(avg_proj):
            marker = " <-- selected" if p >= threshold else ""
            print(f"  Band {i:4d}: p = {p:.6f}{marker}")
        print(f"{'='*70}")

    return ProjectabilityResult(
        band_projectabilities=avg_proj,
        selected_band_indices=selected,
        rejected_band_indices=rejected,
        num_wann=len(selected),
        threshold=threshold,
        projectability_per_kpoint=proj_per_kpoint,
    )
