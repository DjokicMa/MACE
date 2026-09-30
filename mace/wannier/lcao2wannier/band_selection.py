"""
Band Selection Module

This module provides intelligent band and orbital selection for lcao2wannier
conversion. It includes:
- Energy window analysis
- Automatic Fermi level detection
- Automatic frozen window determination
- Intelligent projection orbital selection

For LCAO bases, disentanglement is typically unnecessary since the basis
functions are already localized atomic orbitals.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
import warnings

from scipy.optimize import brentq
from scipy.special import erfc


@dataclass
class BandWindowResult:
    """Results from band window analysis."""
    frozen_indices: np.ndarray      # Band indices fully within window
    partial_indices: np.ndarray     # Band indices crossing window boundaries
    excluded_indices: np.ndarray    # Band indices outside window
    band_ranges: List[Tuple[float, float]]  # (E_min, E_max) for each band
    num_wann: int                   # Number of frozen bands
    frozen_energy_range: Tuple[float, float]  # Actual energy range of frozen bands
    e_fermi: float                  # Fermi energy used
    outer_window: Tuple[float, float]  # User-specified window (absolute)


@dataclass 
class OrbitalSelectionResult:
    """Results from automatic orbital selection."""
    selected_indices: np.ndarray    # Indices of selected projection orbitals
    orbital_weights: np.ndarray     # Weight of each orbital (contribution to bands)
    num_selected: int               # Number of selected orbitals


def compute_fermi_level(
    eigenvalues_list: List[np.ndarray],
    num_electrons: int,
    k_weights: Optional[np.ndarray] = None,
    smearing: str = 'gaussian',
    sigma: float = 1e-5,
    spin_degeneracy: int = 1,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> float:
    """
    Compute the Fermi energy by solving the electron-counting constraint with
    finite-temperature smearing.

    Solves self-consistently for ``E_F`` such that

        N_elec = Σ_k w_k Σ_i f((ε_ik - E_F) / σ)

    using Brent's method for robust, bracketed root-finding. Handles both
    insulators (where σ → 0 recovers the pure aufbau midgap) and metals
    (where partial occupation around E_F is integrated correctly).

    Parameters
    ----------
    eigenvalues_list : list of (num_bands,) ndarrays
        Eigenvalues at each k-point, in eV.
    num_electrons : int
        Total electron count. For SOC systems with spinor eigenvalues
        (each holding 1 electron), this is the total count, not divided by 2.
    k_weights : ndarray, optional
        Per-k-point weights; must sum to 1. Defaults to uniform 1/N_k
        (correct for Monkhorst-Pack grids without IBZ reduction).
    smearing : str
        Occupation function:
        - 'gaussian'         : f = ½ erfc(x/σ)  (DEFAULT, unbiased)
        - 'fermi_dirac'      : f = 1/(exp(x/σ)+1)  (σ acts as k_B T)
        - 'methfessel_paxton': 1st-order MP; reduces σ-dependence for metals
    sigma : float
        Smearing width in eV. Default 1e-5 eV (~0.1 K — effectively ground
        state). Validated on MgB2 (5832 k-point metal): result is identical
        across σ ∈ [1e-7, 1e-5] eV to 6 decimal places, with E_F matching
        CRYSTAL's SCF Fermi to within the k-grid discretization floor
        (~4 meV on 18³). Increase to 0.01–0.1 eV only if stability of
        downstream calculations benefits from finite-T broadening, or if
        the k-grid is pathologically coarse and you want to smooth over
        discretization jitter.
    spin_degeneracy : int
        Number of electrons per band. Default 1 for spinor (SOC) systems.
        Set to 2 for non-SOC paramagnetic systems where each spatial band
        holds 2 electrons. Set to 1 for spin-polarized calculations that
        already resolve α/β separately into distinct bands.
    tol : float
        Relative convergence tolerance on E_F.
    max_iter : int
        Maximum iterations for Brent's method.

    Returns
    -------
    E_F : float
        Fermi energy in eV.

    Notes
    -----
    - For insulators with gap ≫ σ: result equals the midgap within σ.
    - For metals: gives the true Fermi, not the discrete midpoint of two
      sorted eigenvalues. Converges as k-grid density increases.
    - Σ σ→0: reduces to the aufbau fill, with E_F at the last-filled energy
      (use method='zero_temp' in :func:`estimate_fermi_energy` for that).

    References
    ----------
    - Methfessel & Paxton, PRB 40, 3616 (1989) — high-order smearing
    - Marzari et al., PRL 82, 3296 (1999) — cold smearing
    - Dal Corso, PRB 82, 075116 (2010) — comparison of schemes
    """
    if smearing not in ('gaussian', 'fermi_dirac', 'methfessel_paxton'):
        raise ValueError(
            f"Unknown smearing '{smearing}'. "
            "Use 'gaussian', 'fermi_dirac', or 'methfessel_paxton'."
        )

    all_eigs = np.asarray([np.asarray(e) for e in eigenvalues_list])
    if all_eigs.ndim != 2:
        raise ValueError(
            f"eigenvalues_list must produce a 2D array, got shape {all_eigs.shape}"
        )
    nk, nb = all_eigs.shape

    if k_weights is None:
        k_weights = np.full(nk, 1.0 / nk)
    else:
        k_weights = np.asarray(k_weights, dtype=float)
        if k_weights.shape != (nk,):
            raise ValueError(
                f"k_weights shape {k_weights.shape} != (nk={nk},)"
            )
        total_w = float(k_weights.sum())
        if abs(total_w - 1.0) > 1e-9:
            k_weights = k_weights / total_w

    # --- Fast path: clean-gap insulator detection -------------------------
    # For an undoped insulator at T=0, the residual function is flat-zero
    # across the entire gap, so `brentq` would return an arbitrary point
    # inside the zero-interval (typically a gap edge) rather than midgap.
    # Detect this case by per-k-point band counting and return midgap
    # explicitly — matching the physically sensible convention and agreeing
    # with the legacy `zero_temp` aufbau behaviour.
    if num_electrons > 0 and (num_electrons % spin_degeneracy) == 0:
        n_fill_per_k = num_electrons // spin_degeneracy
        if 0 < n_fill_per_k < nb:
            sorted_per_k = np.sort(all_eigs, axis=1)        # (nk, nb)
            local_homo = sorted_per_k[:, n_fill_per_k - 1]   # (nk,)
            local_lumo = sorted_per_k[:, n_fill_per_k]       # (nk,)
            vbm = float(local_homo.max())
            cbm = float(local_lumo.min())
            # Require a clean gap wider than a few σ so smearing truly
            # doesn't overlap the HOMO/LUMO manifolds.
            if cbm - vbm > max(5 * sigma, 1e-4):
                return 0.5 * (vbm + cbm)

    # Occupation function f(x) where x = (eig - E_F)/sigma
    if smearing == 'gaussian':
        def _occ(E_F: float) -> np.ndarray:
            return 0.5 * erfc((all_eigs - E_F) / sigma)
    elif smearing == 'fermi_dirac':
        def _occ(E_F: float) -> np.ndarray:
            x = np.clip((all_eigs - E_F) / sigma, -500.0, 500.0)
            return 1.0 / (np.exp(x) + 1.0)
    elif smearing == 'methfessel_paxton':
        # First-order (N=1) MP smearing: f = ½erfc(x) + ½x exp(-x²)/√π
        # (with sign convention that gives N_elec → N_elec at E_F)
        sqrt_pi = np.sqrt(np.pi)
        def _occ(E_F: float) -> np.ndarray:
            x = (all_eigs - E_F) / sigma
            return 0.5 * erfc(x) - 0.5 * x * np.exp(-x ** 2) / sqrt_pi
    else:
        raise ValueError(
            f"Unknown smearing '{smearing}'. "
            "Use 'gaussian', 'fermi_dirac', or 'methfessel_paxton'."
        )

    def _residual(E_F: float) -> float:
        # N(E_F) - N_elec; want to find root.
        f = _occ(E_F)                              # shape (nk, nb)
        per_k = spin_degeneracy * np.sum(f, axis=1)  # shape (nk,)
        total = float(np.sum(k_weights * per_k))   # scalar
        return total - num_electrons

    # Bracket: well below lowest eigenvalue ⇒ N = 0 < N_elec, so r < 0.
    #          well above highest eigenvalue ⇒ N = nb > N_elec, so r > 0.
    E_lo = float(all_eigs.min()) - max(10 * sigma, 1.0)
    E_hi = float(all_eigs.max()) + max(10 * sigma, 1.0)

    r_lo = _residual(E_lo)
    r_hi = _residual(E_hi)

    # Guard pathological inputs (e.g. N_elec <= 0 or >= total states)
    max_possible = spin_degeneracy * nb  # electrons per k-point × 1 (w_k sums to 1)
    if num_electrons <= 0:
        warnings.warn(
            f"num_electrons={num_electrons} <= 0; returning min eigenvalue."
        )
        return float(all_eigs.min())
    if num_electrons >= max_possible:
        warnings.warn(
            f"num_electrons={num_electrons} >= max occupation "
            f"({spin_degeneracy}×{nb}={max_possible}); "
            f"returning max eigenvalue + σ."
        )
        return float(all_eigs.max()) + sigma

    # Widen bracket if necessary (can happen for pathological smearings)
    widen = 0
    while r_lo > 0 and widen < 10:
        E_lo -= 1.0
        r_lo = _residual(E_lo)
        widen += 1
    while r_hi < 0 and widen < 10:
        E_hi += 1.0
        r_hi = _residual(E_hi)
        widen += 1
    if r_lo > 0 or r_hi < 0:
        raise RuntimeError(
            f"Cannot bracket Fermi level: residual({E_lo:.3f})={r_lo:.3e}, "
            f"residual({E_hi:.3f})={r_hi:.3e}. Check electron count and "
            f"eigenvalues."
        )

    E_F = brentq(_residual, E_lo, E_hi, xtol=tol, maxiter=max_iter)
    return float(E_F)


def estimate_fermi_energy(
    eigenvalues_list: List[np.ndarray],
    num_electrons: Optional[int] = None,
    num_orbitals: Optional[int] = None,
    method: str = 'auto',
    spin_degeneracy: int = 1,
) -> float:
    """
    Estimate the Fermi energy from eigenvalues.
    
    Parameters
    ----------
    eigenvalues_list : list of ndarrays
        Eigenvalues for each k-point, shape (num_bands,) each
    num_electrons : int, optional
        Number of electrons in the system. If provided, E_F is set
        to fill exactly this many states.
    num_orbitals : int, optional
        Number of spatial orbitals (without spin). Used for half-filling
        estimate if num_electrons not provided.
    method : str
        Method for Fermi level estimation:
        - 'auto': Use num_electrons if provided, else 'midgap'
        - 'electron_count': Solve N_elec = Σ_ik w_k f(ε_ik - E_F; σ) with
          Gaussian smearing σ = 0.01 eV. Rigorous for insulators and metals.
          Delegates to :func:`compute_fermi_level`.
        - 'zero_temp': Legacy T=0 aufbau fill. Equivalent to the pre-2026
          behaviour. Use only for reproducing old results.
        - 'midgap': Find largest gap near center of spectrum (no electron
          count needed).
        - 'half_filling': Assume half the bands are occupied.
        
    Returns
    -------
    float
        Estimated Fermi energy in same units as eigenvalues
        
    Notes
    -----
    For spin-orbit coupled systems with 2N×2N matrices, all bands
    are already spin-resolved, so num_electrons should be the total
    electron count (not divided by 2).
    """
    num_kpoints = len(eigenvalues_list)
    num_bands = len(eigenvalues_list[0])
    
    # Collect all eigenvalues
    all_eigenvalues = np.array(eigenvalues_list)  # Shape: (num_kpoints, num_bands)
    
    if method == 'auto':
        if num_electrons is not None:
            method = 'electron_count'
        else:
            method = 'midgap'
    
    if method == 'electron_count':
        if num_electrons is None:
            raise ValueError("num_electrons required for 'electron_count' method")

        # Proper Fermi-level calculation: smeared occupation + root-find.
        # Rigorous for insulators (gap ≫ σ → midgap) and metals (partial
        # occupation integrated correctly). Replaces the old T=0 aufbau
        # patch. Default σ = 1e-5 eV (~0.1 K) is effectively ground state
        # and matches CRYSTAL SCF Fermi values to k-grid precision on
        # validation cases (see compute_fermi_level docstring).
        return compute_fermi_level(
            eigenvalues_list,
            num_electrons=num_electrons,
            smearing='gaussian',
            sigma=1e-5,
            spin_degeneracy=spin_degeneracy,
        )

    elif method == 'zero_temp':
        # Legacy T=0 aufbau fill, preserved for reproducibility of old runs.
        # New code should prefer 'electron_count' (which uses smeared
        # root-finding via compute_fermi_level).
        if num_electrons is None:
            raise ValueError("num_electrons required for 'zero_temp' method")

        sorted_eigenvalues = np.sort(all_eigenvalues.flatten())
        # spin_degeneracy divides electrons/band → number of states to fill
        states_to_fill = (num_electrons * num_kpoints) // spin_degeneracy

        if states_to_fill >= len(sorted_eigenvalues):
            warnings.warn("num_electrons exceeds available states, using highest energy")
            return sorted_eigenvalues[-1] + 0.1
        if states_to_fill <= 0:
            warnings.warn("num_electrons <= 0, using lowest energy")
            return sorted_eigenvalues[0] - 0.1

        e_homo = sorted_eigenvalues[states_to_fill - 1]
        e_lumo = sorted_eigenvalues[states_to_fill]
        return (e_homo + e_lumo) / 2
    
    elif method == 'half_filling':
        # Assume half the bands are occupied
        num_occupied = num_bands // 2
        
        # Average energy of HOMO and LUMO across k-points
        homo_energies = all_eigenvalues[:, num_occupied - 1]
        lumo_energies = all_eigenvalues[:, num_occupied]
        
        e_fermi = (homo_energies.mean() + lumo_energies.mean()) / 2
        return e_fermi
    
    elif method == 'midgap':
        # Find the largest gap in the band structure near the center
        # Average band energies across k-points
        avg_band_energies = np.mean(all_eigenvalues, axis=0)
        
        # Compute gaps between consecutive bands
        gaps = np.diff(avg_band_energies)
        
        # Weight gaps toward center of spectrum (prefer gaps near middle)
        center_idx = num_bands // 2
        weights = np.exp(-0.1 * (np.arange(len(gaps)) - center_idx)**2)
        weighted_gaps = gaps * weights
        
        # Find the largest weighted gap
        gap_idx = np.argmax(weighted_gaps)
        
        # Fermi level is in the middle of this gap
        e_fermi = (avg_band_energies[gap_idx] + avg_band_energies[gap_idx + 1]) / 2
        
        return e_fermi
    
    else:
        raise ValueError(f"Unknown method: {method}")


def analyze_band_window(
    eigenvalues_list: List[np.ndarray],
    outer_window: Tuple[float, float],
    e_fermi: Optional[float] = None,
    window_is_relative: bool = True,
    num_electrons: Optional[int] = None
) -> BandWindowResult:
    """
    Analyze bands relative to an energy window and identify frozen bands.
    
    Frozen bands are those whose ENTIRE dispersion (across all k-points)
    falls within the outer window. This guarantees a consistent band count
    at all k-points, which is required for Wannier90.
    
    Parameters
    ----------
    eigenvalues_list : list of ndarrays
        Eigenvalues for each k-point
    outer_window : tuple of float
        (E_min, E_max) energy window
    e_fermi : float, optional
        Fermi energy. If None and window_is_relative=True, will be auto-detected.
    window_is_relative : bool
        If True, outer_window is relative to E_F. If False, it's absolute.
    num_electrons : int, optional
        Number of electrons (for Fermi level auto-detection)
        
    Returns
    -------
    BandWindowResult
        Dataclass containing analysis results
    """
    num_kpoints = len(eigenvalues_list)
    num_bands = len(eigenvalues_list[0])
    
    # Auto-detect Fermi level if needed
    if e_fermi is None and window_is_relative:
        e_fermi = estimate_fermi_energy(
            eigenvalues_list, 
            num_electrons=num_electrons,
            method='auto'
        )
        print(f"  Auto-detected Fermi energy: {e_fermi:.4f} eV")
    elif e_fermi is None:
        e_fermi = 0.0
    
    # Convert window to absolute energies
    if window_is_relative:
        e_min = e_fermi + outer_window[0]
        e_max = e_fermi + outer_window[1]
    else:
        e_min, e_max = outer_window
    
    # Find energy range of each band across all k-points
    all_eigenvalues = np.array(eigenvalues_list)
    band_ranges = []
    for band_idx in range(num_bands):
        band_energies = all_eigenvalues[:, band_idx]
        band_ranges.append((band_energies.min(), band_energies.max()))
    
    # Classify bands
    frozen_indices = []
    partial_indices = []
    excluded_indices = []
    
    for band_idx, (b_min, b_max) in enumerate(band_ranges):
        if b_min >= e_min and b_max <= e_max:
            # Entire band within window - frozen
            frozen_indices.append(band_idx)
        elif b_max < e_min or b_min > e_max:
            # Entire band outside window - excluded
            excluded_indices.append(band_idx)
        else:
            # Band crosses window boundary - partial
            partial_indices.append(band_idx)
    
    frozen_indices = np.array(frozen_indices, dtype=int)
    partial_indices = np.array(partial_indices, dtype=int)
    excluded_indices = np.array(excluded_indices, dtype=int)
    
    # Determine actual energy range of frozen bands
    if len(frozen_indices) > 0:
        frozen_mins = [band_ranges[i][0] for i in frozen_indices]
        frozen_maxs = [band_ranges[i][1] for i in frozen_indices]
        frozen_energy_range = (min(frozen_mins), max(frozen_maxs))
    else:
        frozen_energy_range = (0.0, 0.0)
    
    return BandWindowResult(
        frozen_indices=frozen_indices,
        partial_indices=partial_indices,
        excluded_indices=excluded_indices,
        band_ranges=band_ranges,
        num_wann=len(frozen_indices),
        frozen_energy_range=frozen_energy_range,
        e_fermi=e_fermi,
        outer_window=(e_min, e_max)
    )


def select_bands_near_fermi(
    band_indices: np.ndarray,
    band_ranges: List[Tuple[float, float]],
    e_fermi: float,
    num_bands: int
) -> np.ndarray:
    """
    Select bands closest to Fermi level.

    This function selects the bands whose center energies are closest to
    the Fermi level, rather than simply taking the first N bands by index.
    This ensures physically meaningful band selection for Wannierization.

    Parameters
    ----------
    band_indices : ndarray
        Available band indices to select from
    band_ranges : list of tuples
        (E_min, E_max) for each band across all k-points
    e_fermi : float
        Fermi energy
    num_bands : int
        Number of bands to select

    Returns
    -------
    selected_bands : ndarray
        Selected band indices, sorted in ascending order

    Examples
    --------
    >>> band_indices = np.array([20, 21, 22, 23, 24, 25, 26, 27, 28, 29])
    >>> band_ranges = [(-5.0, -4.5), (-4.8, -4.2), ..., (2.0, 2.5)]
    >>> e_fermi = -3.728
    >>> selected = select_bands_near_fermi(band_indices, band_ranges, e_fermi, 5)
    >>> # Returns the 5 bands with centers closest to -3.728 eV
    """
    if num_bands >= len(band_indices):
        return np.sort(band_indices)

    # Compute center energy of each band
    band_centers = np.array([
        (band_ranges[i][0] + band_ranges[i][1]) / 2
        for i in band_indices
    ])

    # Sort by distance from Fermi level
    distances = np.abs(band_centers - e_fermi)
    sorted_positions = np.argsort(distances)
    selected = band_indices[sorted_positions[:num_bands]]

    return np.sort(selected)


def print_band_analysis(result: BandWindowResult, verbose: bool = True) -> str:
    """
    Generate a human-readable report of band window analysis.
    
    Parameters
    ----------
    result : BandWindowResult
        Results from analyze_band_window()
    verbose : bool
        If True, print detailed per-band information
        
    Returns
    -------
    str
        Formatted analysis report
    """
    lines = []
    lines.append("=" * 70)
    lines.append("BAND WINDOW ANALYSIS")
    lines.append("=" * 70)
    lines.append(f"Fermi energy: {result.e_fermi:.4f} eV")
    lines.append(f"Outer window: [{result.outer_window[0]:.2f}, {result.outer_window[1]:.2f}] eV (absolute)")
    lines.append(f"              [{result.outer_window[0] - result.e_fermi:.2f}, "
                 f"{result.outer_window[1] - result.e_fermi:.2f}] eV (relative to E_F)")
    lines.append("")
    
    num_bands = len(result.band_ranges)
    
    # Group consecutive bands for cleaner output
    def group_consecutive(indices):
        """Group consecutive indices into ranges."""
        if len(indices) == 0:
            return []
        groups = []
        start = indices[0]
        end = indices[0]
        for idx in indices[1:]:
            if idx == end + 1:
                end = idx
            else:
                groups.append((start, end))
                start = idx
                end = idx
        groups.append((start, end))
        return groups
    
    lines.append("Band classification:")
    
    # Excluded below
    excluded_below = [i for i in result.excluded_indices 
                      if result.band_ranges[i][1] < result.outer_window[0]]
    if excluded_below:
        groups = group_consecutive(excluded_below)
        for start, end in groups:
            e_min = min(result.band_ranges[i][0] for i in range(start, end + 1))
            e_max = max(result.band_ranges[i][1] for i in range(start, end + 1))
            if start == end:
                lines.append(f"  Band {start}:      E = [{e_min:.2f}, {e_max:.2f}] eV  → EXCLUDED (below window)")
            else:
                lines.append(f"  Bands {start}-{end}:   E = [{e_min:.2f}, {e_max:.2f}] eV  → EXCLUDED (below window)")
    
    # Partial (lower boundary)
    partial_lower = [i for i in result.partial_indices 
                     if result.band_ranges[i][0] < result.outer_window[0]]
    if partial_lower:
        groups = group_consecutive(partial_lower)
        for start, end in groups:
            e_min = min(result.band_ranges[i][0] for i in range(start, end + 1))
            e_max = max(result.band_ranges[i][1] for i in range(start, end + 1))
            if start == end:
                lines.append(f"  Band {start}:      E = [{e_min:.2f}, {e_max:.2f}] eV  → PARTIAL (crosses lower boundary)")
            else:
                lines.append(f"  Bands {start}-{end}:   E = [{e_min:.2f}, {e_max:.2f}] eV  → PARTIAL (crosses lower boundary)")
    
    # Frozen
    if len(result.frozen_indices) > 0:
        groups = group_consecutive(list(result.frozen_indices))
        for start, end in groups:
            e_min = min(result.band_ranges[i][0] for i in range(start, end + 1))
            e_max = max(result.band_ranges[i][1] for i in range(start, end + 1))
            if start == end:
                lines.append(f"  Band {start}:      E = [{e_min:.2f}, {e_max:.2f}] eV  → FROZEN ✓")
            else:
                lines.append(f"  Bands {start}-{end}:  E = [{e_min:.2f}, {e_max:.2f}] eV  → FROZEN ✓")
    
    # Partial (upper boundary)
    partial_upper = [i for i in result.partial_indices 
                     if result.band_ranges[i][1] > result.outer_window[1]]
    if partial_upper:
        groups = group_consecutive(partial_upper)
        for start, end in groups:
            e_min = min(result.band_ranges[i][0] for i in range(start, end + 1))
            e_max = max(result.band_ranges[i][1] for i in range(start, end + 1))
            if start == end:
                lines.append(f"  Band {start}:      E = [{e_min:.2f}, {e_max:.2f}] eV  → PARTIAL (crosses upper boundary)")
            else:
                lines.append(f"  Bands {start}-{end}:   E = [{e_min:.2f}, {e_max:.2f}] eV  → PARTIAL (crosses upper boundary)")
    
    # Excluded above
    excluded_above = [i for i in result.excluded_indices 
                      if result.band_ranges[i][0] > result.outer_window[1]]
    if excluded_above:
        groups = group_consecutive(excluded_above)
        for start, end in groups:
            e_min = min(result.band_ranges[i][0] for i in range(start, end + 1))
            e_max = max(result.band_ranges[i][1] for i in range(start, end + 1))
            if start == end:
                lines.append(f"  Band {start}:      E = [{e_min:.2f}, {e_max:.2f}] eV  → EXCLUDED (above window)")
            else:
                lines.append(f"  Bands {start}-{end}:  E = [{e_min:.2f}, {e_max:.2f}] eV  → EXCLUDED (above window)")
    
    lines.append("")
    lines.append("-" * 70)
    lines.append("RESULT:")
    lines.append(f"  Frozen bands: {result.num_wann} (indices {result.frozen_indices[0]}-{result.frozen_indices[-1]})" 
                 if result.num_wann > 0 else "  Frozen bands: 0 (WARNING: no bands fully within window!)")
    lines.append(f"  → Setting num_wann = {result.num_wann}")
    
    if result.num_wann > 0:
        lines.append("")
        lines.append(f"Exact frozen window: [{result.frozen_energy_range[0]:.4f}, "
                     f"{result.frozen_energy_range[1]:.4f}] eV")
    
    # Suggestions for partial bands
    if len(result.partial_indices) > 0:
        lines.append("")
        lines.append("SUGGESTIONS for partial bands:")
        for idx in result.partial_indices:
            b_min, b_max = result.band_ranges[idx]
            if b_min < result.outer_window[0]:
                lines.append(f"  Band {idx}: To include, expand lower bound to {b_min:.2f} eV")
            if b_max > result.outer_window[1]:
                lines.append(f"  Band {idx}: To include, expand upper bound to {b_max:.2f} eV")
    
    lines.append("=" * 70)
    
    report = "\n".join(lines)
    print(report)
    return report


def check_frozen_continuity(frozen_indices: np.ndarray) -> Tuple[bool, List[int]]:
    """
    Check if frozen bands form a contiguous set.
    
    Non-contiguous frozen bands may indicate unusual band structure
    or an inappropriately chosen energy window.
    
    Parameters
    ----------
    frozen_indices : ndarray
        Indices of frozen bands
        
    Returns
    -------
    is_contiguous : bool
        True if bands are contiguous
    gap_positions : list of int
        Indices where gaps occur in the frozen set
    """
    if len(frozen_indices) <= 1:
        return True, []
    
    sorted_indices = np.sort(frozen_indices)
    gaps = np.diff(sorted_indices)
    gap_positions = np.where(gaps > 1)[0].tolist()
    
    is_contiguous = len(gap_positions) == 0
    
    if not is_contiguous:
        warnings.warn(
            f"Frozen bands are non-contiguous! Gaps at indices: "
            f"{[sorted_indices[i] for i in gap_positions]}. "
            "This may indicate an unusual band structure."
        )
    
    return is_contiguous, gap_positions


def validate_fermi_coverage(
    frozen_indices: np.ndarray,
    eigenvalues_list: List[np.ndarray],
    e_fermi: float
) -> Dict[str, any]:
    """
    Validate that frozen window properly covers the Fermi level.
    
    For meaningful Wannier functions, the frozen window should typically
    include both occupied and unoccupied states near E_F.
    
    Parameters
    ----------
    frozen_indices : ndarray
        Indices of frozen bands
    eigenvalues_list : list of ndarrays
        Eigenvalues for each k-point
    e_fermi : float
        Fermi energy
        
    Returns
    -------
    dict
        Validation results with keys:
        - 'has_occupied': bool, whether frozen set includes occupied bands
        - 'has_unoccupied': bool, whether frozen set includes unoccupied bands
        - 'num_occupied': int, number of frozen bands below E_F
        - 'num_unoccupied': int, number of frozen bands above E_F
        - 'crosses_fermi': bool, whether any band crosses E_F
    """
    if len(frozen_indices) == 0:
        return {
            'has_occupied': False,
            'has_unoccupied': False,
            'num_occupied': 0,
            'num_unoccupied': 0,
            'crosses_fermi': False
        }
    
    all_eigenvalues = np.array(eigenvalues_list)
    
    num_occupied = 0
    num_unoccupied = 0
    crosses_fermi = False
    
    for band_idx in frozen_indices:
        band_energies = all_eigenvalues[:, band_idx]
        band_min = band_energies.min()
        band_max = band_energies.max()
        
        if band_max < e_fermi:
            num_occupied += 1
        elif band_min > e_fermi:
            num_unoccupied += 1
        else:
            # Band crosses Fermi level
            crosses_fermi = True
            # Count as both for simplicity
            num_occupied += 0.5
            num_unoccupied += 0.5
    
    return {
        'has_occupied': num_occupied > 0,
        'has_unoccupied': num_unoccupied > 0,
        'num_occupied': int(num_occupied),
        'num_unoccupied': int(num_unoccupied),
        'crosses_fermi': crosses_fermi
    }


def select_projection_orbitals(
    eigenvectors_list: List[np.ndarray],
    S_k_list: List[np.ndarray],
    num_wann: int,
    band_indices: Optional[np.ndarray] = None
) -> OrbitalSelectionResult:
    """
    Automatically select LCAO orbitals for projection based on their
    contribution to the selected bands.
    
    For LCAO bases, this provides an optimal set of projection orbitals
    without requiring user-specified projections (like Bi:s,p in wien2wannier).
    
    Parameters
    ----------
    eigenvectors_list : list of ndarrays
        Eigenvectors C(k) for each k-point, shape (num_orbitals, num_bands)
    S_k_list : list of ndarrays
        Overlap matrices S(k) for each k-point
    num_wann : int
        Number of Wannier functions (orbitals to select)
    band_indices : ndarray, optional
        If provided, only consider contributions from these bands.
        If None, use all bands in eigenvectors_list.
        
    Returns
    -------
    OrbitalSelectionResult
        Contains selected orbital indices, weights, and count
        
    Notes
    -----
    The algorithm computes the projection of bands onto each LCAO orbital:
        A_nk = S(k)† C(k)
    Then sums |A_nk|² over all selected bands and k-points for each orbital n.
    The orbitals with largest total weight are selected.
    """
    num_kpoints = len(eigenvectors_list)
    num_orbitals = eigenvectors_list[0].shape[0]
    
    if band_indices is None:
        num_bands = eigenvectors_list[0].shape[1]
        band_indices = np.arange(num_bands)
    
    orbital_weights = np.zeros(num_orbitals)
    
    for k_idx in range(num_kpoints):
        C_k = eigenvectors_list[k_idx][:, band_indices]  # Select only specified bands
        S_k = S_k_list[k_idx]
        
        # A_k = S† C gives projection of bands onto orbitals
        # Shape: (num_orbitals, num_selected_bands)
        A_k = S_k.conj().T @ C_k
        
        # Sum squared magnitude of projections for each orbital
        orbital_weights += np.sum(np.abs(A_k)**2, axis=1)
    
    # Normalize by number of k-points and bands
    orbital_weights /= (num_kpoints * len(band_indices))
    
    # Select orbitals with largest total contribution
    selected_indices = np.argsort(orbital_weights)[-num_wann:]
    selected_indices = np.sort(selected_indices)  # Sort in ascending order
    
    return OrbitalSelectionResult(
        selected_indices=selected_indices,
        orbital_weights=orbital_weights,
        num_selected=num_wann
    )


def compute_subspace_projections(
    eigenvectors_list: List[np.ndarray],
    S_k_list: List[np.ndarray],
    band_indices: np.ndarray,
    orbital_indices: np.ndarray
) -> List[np.ndarray]:
    """
    Compute projection matrices for a selected subspace.
    
    This creates the A(k) matrices for Wannier90 when using a subset
    of bands and/or orbitals.
    
    Parameters
    ----------
    eigenvectors_list : list of ndarrays
        Full eigenvectors C(k) for each k-point
    S_k_list : list of ndarrays
        Overlap matrices S(k) for each k-point
    band_indices : ndarray
        Indices of bands to include
    orbital_indices : ndarray
        Indices of LCAO orbitals to project onto
        
    Returns
    -------
    list of ndarrays
        Projection matrices A(k) for each k-point
        Shape: (num_selected_bands, num_selected_orbitals)
    """
    A_k_list = []
    
    for k_idx in range(len(eigenvectors_list)):
        C_k_full = eigenvectors_list[k_idx]
        S_k_full = S_k_list[k_idx]
        
        # Select bands
        C_k = C_k_full[:, band_indices]
        
        # Compute full projection
        A_k_full = S_k_full.conj().T @ C_k  # Shape: (num_orbitals, num_bands)
        
        # Select orbitals (rows)
        A_k = A_k_full[orbital_indices, :]  # Shape: (num_wann, num_bands)
        
        # Transpose to match Wannier90 convention: (num_bands, num_wann)
        A_k_list.append(A_k.T)
    
    return A_k_list


def scdm_select_projections(
    eigenvectors_list: List[np.ndarray],
    S_k_list: List[np.ndarray],
    eigenvalues_list: List[np.ndarray],
    num_wann: int,
    e_fermi: float = 0.0,
    mu: Optional[float] = None,
    sigma: Optional[float] = None,
    method: str = 'erfc',
    band_indices: Optional[np.ndarray] = None,
    orbital_mask: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> OrbitalSelectionResult:
    """
    Select projection orbitals using SCDM-L (Selected Columns of the
    Density Matrix with Lowdin orthogonalization).

    Unlike the weight-based method (select_projection_orbitals) which ranks
    orbitals independently by their projection weight, SCDM-L uses QR
    factorization with column pivoting (QRCP) on the k-averaged Lowdin
    density matrix to select a maximally independent set of orbitals.
    This accounts for correlations between orbitals and avoids selecting
    redundant projections.

    Parameters
    ----------
    eigenvectors_list : list of ndarrays
        Eigenvectors C(k) for each k-point, shape (num_orbitals, num_bands)
    S_k_list : list of ndarrays
        Overlap matrices S(k) for each k-point, shape (num_orbitals, num_orbitals)
    eigenvalues_list : list of ndarrays
        Eigenvalues for each k-point, shape (num_bands,)
    num_wann : int
        Number of Wannier functions (orbitals to select)
    e_fermi : float
        Fermi energy in eV
    mu : float, optional
        Energy centre for occupation function. Default: e_fermi
    sigma : float, optional
        Energy width for occupation function in eV. Default: auto from band gap
    method : str
        Occupation weighting function:
        - 'erfc': f(e) = 0.5 * erfc((e - mu) / sigma) — smooth Fermi-like
        - 'gaussian': f(e) = exp(-((e - mu) / sigma)^2) — symmetric around mu
        - 'isolated': f(e) = 1 — all bands weighted equally
    band_indices : ndarray, optional
        Restrict to these band indices. If None, use all bands.
    orbital_mask : ndarray of bool, optional
        Boolean mask of shape (num_orbitals,) indicating which AO indices
        are valid for selection. When provided, columns of the Löwdin
        density matrix corresponding to False entries are zeroed out before
        QR pivoting, forcing SCDM to only select from the desired orbital
        subspace. This is used by --symmetrize to constrain projections to
        complete orbital shells. If None, all orbitals are eligible.
    verbose : bool
        Print diagnostic information

    Returns
    -------
    OrbitalSelectionResult
        Contains selected orbital indices, weights, and count

    References
    ----------
    Damle, Lin, Ying, J. Comput. Phys. 334, 1 (2017) — SCDM-k method
    Qiu et al., J. Chem. Theory Comput. 17, 7373 (2021) — SCDM-L for AO basis
    Carlson and Keller, Phys. Rev. 105, 102 (1957) — Lowdin optimality
    """
    from scipy.linalg import qr
    from scipy.special import erfc as sp_erfc

    num_kpoints = len(eigenvectors_list)
    num_orbitals = eigenvectors_list[0].shape[0]

    if band_indices is None:
        num_bands = eigenvectors_list[0].shape[1]
        band_indices = np.arange(num_bands)

    # Set occupation function parameters
    if mu is None:
        mu = e_fermi
    if sigma is None:
        # Auto-determine sigma from band gap or default
        all_evals = np.concatenate([ev[band_indices] for ev in eigenvalues_list])
        # Find approximate gap near E_F
        below = all_evals[all_evals <= e_fermi]
        above = all_evals[all_evals > e_fermi]
        if len(below) > 0 and len(above) > 0:
            gap = np.min(above) - np.max(below)
            sigma = max(gap, 1.0)
        else:
            sigma = 1.0

    if verbose:
        print(f"\n  SCDM-L projection selection:")
        print(f"    Occupation function: {method}")
        print(f"    mu = {mu:.4f} eV, sigma = {sigma:.4f} eV")
        print(f"    Number of bands considered: {len(band_indices)}")
        print(f"    Target num_wann: {num_wann}")

    # Accumulate k-averaged Lowdin density matrix
    P_L = np.zeros((num_orbitals, num_orbitals), dtype=complex)

    for ik in range(num_kpoints):
        C_k = eigenvectors_list[ik][:, band_indices]
        S_k = S_k_list[ik]
        evals = eigenvalues_list[ik][band_indices]

        # Compute occupation weights
        if method == 'erfc':
            f_occ = 0.5 * sp_erfc((evals - mu) / sigma)
        elif method == 'gaussian':
            f_occ = np.exp(-((evals - mu) / sigma) ** 2)
        elif method == 'isolated':
            f_occ = np.ones_like(evals)
        else:
            raise ValueError(f"Unknown occupation method: {method}")

        # Compute S(k)^{1/2} via eigendecomposition
        eigvals_s, eigvecs_s = np.linalg.eigh(S_k)
        # Clamp small/negative eigenvalues for numerical stability
        eigvals_s = np.maximum(eigvals_s, 1e-12)
        S_sqrt = eigvecs_s @ np.diag(np.sqrt(eigvals_s)) @ eigvecs_s.conj().T

        # Weighted density matrix: P(k) = C(k) diag(f) C(k)^dag
        P_k = C_k @ np.diag(f_occ) @ C_k.conj().T

        # Lowdin transform: P_L(k) = S^{1/2}(k) P(k) S^{1/2}(k)
        P_L += S_sqrt @ P_k @ S_sqrt

    P_L /= num_kpoints

    # Apply orbital_mask: zero out columns/rows for excluded orbitals
    # This forces QR column pivoting to rank masked-out orbitals last
    if orbital_mask is not None:
        orbital_mask = np.asarray(orbital_mask, dtype=bool)
        if len(orbital_mask) != num_orbitals:
            raise ValueError(
                f"orbital_mask length ({len(orbital_mask)}) != "
                f"num_orbitals ({num_orbitals})"
            )
        excluded = ~orbital_mask
        P_L[excluded, :] = 0.0
        P_L[:, excluded] = 0.0
        n_allowed = np.sum(orbital_mask)
        if verbose:
            print(f"    Orbital mask: {n_allowed}/{num_orbitals} orbitals allowed")
        if n_allowed < num_wann:
            raise ValueError(
                f"orbital_mask allows only {n_allowed} orbitals but "
                f"num_wann={num_wann} requested"
            )

    # Apply QR with column pivoting (QRCP)
    # P_L is Hermitian, so columns = rows in terms of importance
    Q, R, piv = qr(P_L, pivoting=True)

    # The first num_wann pivots are the most important orbital indices
    selected_indices = np.sort(piv[:num_wann])

    # Compute diagnostic weights from R diagonal (measures column importance)
    R_diag = np.abs(np.diag(R))
    # Map pivot-ordered weights back to original orbital ordering
    orbital_weights = np.zeros(num_orbitals)
    for i, p in enumerate(piv):
        if i < len(R_diag):
            orbital_weights[p] = R_diag[i]
    # Normalize
    max_weight = np.max(orbital_weights) if np.max(orbital_weights) > 0 else 1.0
    orbital_weights /= max_weight

    if verbose:
        print(f"    Selected orbital indices: {selected_indices}")
        # Show top contributors
        top_indices = piv[:min(5, num_wann)]
        print(f"    Top {len(top_indices)} SCDM pivots (by importance):")
        for rank, idx in enumerate(top_indices):
            print(f"      Rank {rank}: orbital {idx}, |R_diag| = {R_diag[rank]:.6f}")

        # Show occupation weight statistics
        all_evals_sel = np.concatenate([ev[band_indices] for ev in eigenvalues_list])
        if method == 'erfc':
            all_f = 0.5 * sp_erfc((all_evals_sel - mu) / sigma)
        elif method == 'gaussian':
            all_f = np.exp(-((all_evals_sel - mu) / sigma) ** 2)
        else:
            all_f = np.ones_like(all_evals_sel)
        print(f"    Occupation weights: min={np.min(all_f):.4f}, max={np.max(all_f):.4f}, "
              f"mean={np.mean(all_f):.4f}")

    return OrbitalSelectionResult(
        selected_indices=selected_indices,
        orbital_weights=orbital_weights,
        num_selected=num_wann,
    )


def suggest_optimal_window(
    eigenvalues_list: List[np.ndarray],
    e_fermi: float,
    target_num_wann: Optional[int] = None,
    padding: float = 0.1
) -> Tuple[float, float]:
    """
    Suggest an optimal energy window based on band structure.
    
    Parameters
    ----------
    eigenvalues_list : list of ndarrays
        Eigenvalues for each k-point
    e_fermi : float
        Fermi energy
    target_num_wann : int, optional
        Target number of Wannier functions. If provided, tries to find
        a window containing approximately this many bands.
    padding : float
        Extra energy padding (eV) around the tight band edges
        
    Returns
    -------
    tuple of float
        (E_min, E_max) suggested window in absolute energies
    """
    all_eigenvalues = np.array(eigenvalues_list)
    num_bands = all_eigenvalues.shape[1]
    
    # Compute band ranges
    band_mins = np.min(all_eigenvalues, axis=0)
    band_maxs = np.max(all_eigenvalues, axis=0)
    
    # Find gaps between bands
    avg_energies = np.mean(all_eigenvalues, axis=0)
    gaps = np.diff(avg_energies)
    
    if target_num_wann is None:
        # Find bands around Fermi level with significant gaps above/below
        center_bands = np.where(
            (band_mins < e_fermi + 2.0) & (band_maxs > e_fermi - 2.0)
        )[0]
        
        if len(center_bands) > 0:
            first_band = center_bands[0]
            last_band = center_bands[-1]
            
            # Expand to nearest significant gaps
            gap_threshold = np.percentile(gaps, 75)  # Use 75th percentile as threshold
            
            # Search for gap below
            for i in range(first_band - 1, -1, -1):
                if gaps[i] > gap_threshold:
                    first_band = i + 1
                    break
            else:
                first_band = 0
            
            # Search for gap above
            for i in range(last_band, len(gaps)):
                if gaps[i] > gap_threshold:
                    last_band = i
                    break
            else:
                last_band = num_bands - 1
            
            e_min = band_mins[first_band] - padding
            e_max = band_maxs[last_band] + padding
        else:
            # Fallback: center ±5 eV around Fermi level
            e_min = e_fermi - 5.0
            e_max = e_fermi + 5.0
    else:
        # Find window containing target_num_wann bands centered on E_F
        # Start from Fermi level and expand symmetrically
        
        # Find band closest to Fermi level
        distances_to_fermi = np.abs(avg_energies - e_fermi)
        center_band = np.argmin(distances_to_fermi)
        
        # Expand symmetrically
        half_bands = target_num_wann // 2
        first_band = max(0, center_band - half_bands)
        last_band = min(num_bands - 1, center_band + half_bands)
        
        # Adjust if we hit boundaries
        actual_count = last_band - first_band + 1
        if actual_count < target_num_wann:
            if first_band == 0:
                last_band = min(num_bands - 1, first_band + target_num_wann - 1)
            elif last_band == num_bands - 1:
                first_band = max(0, last_band - target_num_wann + 1)
        
        e_min = band_mins[first_band] - padding
        e_max = band_maxs[last_band] + padding
    
    return (e_min, e_max)