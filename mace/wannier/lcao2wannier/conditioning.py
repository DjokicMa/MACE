# Copyright (c) 2025 Computational Materials Science Team (lcao2wannier, author William Comaskey).
# MIT License, see LICENSE in this directory.
# Vendored into MACE from lcao2wannier 1.0.0; local changes are listed in VENDORED.md.
"""
Overlap matrix conditioning validation.

Validates that the R-vector set used in the Fourier sum produces
well-conditioned overlap matrices S(k). If S(k) is ill-conditioned
(too few R-vectors), the eigenvalue problem HC=SCE becomes numerically
unstable and Wannier functions will be poorly localized.

This check runs BEFORE the expensive eigensolve, sampling S(k) at
a small set of k-points and computing condition numbers.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import warnings

import numpy as np

from .fourier import StackedMatrices, fourier_all_kpoints


class OverlapConditioningError(RuntimeError):
    """Raised when overlap matrices are too ill-conditioned to proceed."""

    def __init__(self, result: 'OverlapConditioningResult'):
        self.result = result
        super().__init__(result.message)


@dataclass
class OverlapConditioningResult:
    """Diagnostics from overlap matrix conditioning analysis."""

    num_R: int
    num_orbitals: int
    num_kpoints_sampled: int

    worst_condition_number: float
    worst_condition_kpoint: np.ndarray
    min_eigenvalue: float
    min_eigenvalue_kpoint: np.ndarray
    mean_condition_number: float
    median_condition_number: float

    num_not_positive_definite: int
    fraction_needing_regularization: float

    condition_numbers: np.ndarray
    min_eigenvalues: np.ndarray
    sampled_kpoints: np.ndarray

    status: str  # 'good', 'marginal', 'bad'
    message: str


def _build_sample_kpoints(
    k_grid: Optional[Tuple[int, int, int]],
    num_sample: int,
) -> np.ndarray:
    """Build a set of k-points to sample for conditioning analysis."""
    is_2d = k_grid is not None and k_grid[2] == 1
    k_sample = []

    # Phase 1: High-symmetry points
    high_sym = [
        [0.0, 0.0, 0.0],    # Gamma
        [0.5, 0.0, 0.0],    # X
        [0.0, 0.5, 0.0],    # Y
        [0.0, 0.0, 0.5],    # Z
        [0.5, 0.5, 0.0],    # M
        [0.5, 0.0, 0.5],    # L
        [0.0, 0.5, 0.5],    # N
        [0.5, 0.5, 0.5],    # R
        [1/3, 1/3, 0.0],    # K
        [1/3, 1/3, 0.5],    # H
    ]
    for kpt in high_sym:
        if is_2d and kpt[2] != 0.0:
            continue
        k_sample.append(kpt)

    # Phase 2: BZ edge midpoints
    edge_points = [
        [0.25, 0.0, 0.0],
        [0.0, 0.25, 0.0],
        [0.25, 0.25, 0.0],
        [0.25, 0.5, 0.0],
        [0.5, 0.25, 0.0],
        [0.25, 0.0, 0.25],
        [0.0, 0.25, 0.25],
        [0.25, 0.25, 0.25],
    ]
    for kpt in edge_points:
        if is_2d and kpt[2] != 0.0:
            continue
        k_sample.append(kpt)

    # Phase 3: Random k-points to fill up to num_sample
    num_random = max(0, num_sample - len(k_sample))
    if num_random > 0:
        rng = np.random.default_rng(seed=42)
        random_kpts = rng.random((num_random, 3))
        if is_2d:
            random_kpts[:, 2] = 0.0
        k_sample.extend(random_kpts.tolist())

    return np.array(k_sample)


def _print_conditioning_report(result: OverlapConditioningResult) -> None:
    """Print formatted conditioning diagnostic report."""
    print(f"\n{'=' * 70}")
    print("Overlap Matrix Conditioning Check")
    print(f"{'=' * 70}")
    print(f"  R-vectors in Fourier sum:     {result.num_R}")
    print(f"  Orbital space dimension:      {result.num_orbitals} × {result.num_orbitals}")
    print(f"  K-points sampled:             {result.num_kpoints_sampled}")
    print()

    k_worst = result.worst_condition_kpoint
    k_min = result.min_eigenvalue_kpoint
    print(f"  Worst condition number:       {result.worst_condition_number:.3e}"
          f"  at k = ({k_worst[0]:.3f}, {k_worst[1]:.3f}, {k_worst[2]:.3f})")
    print(f"  Median condition number:      {result.median_condition_number:.3e}")
    print(f"  Mean condition number:        {result.mean_condition_number:.3e}")
    print(f"  Min eigenvalue of S(k):       {result.min_eigenvalue:.3e}"
          f"  at k = ({k_min[0]:.3f}, {k_min[1]:.3f}, {k_min[2]:.3f})")
    print()

    K = result.num_kpoints_sampled
    npd = result.num_not_positive_definite
    frac_reg = result.fraction_needing_regularization
    print(f"  K-points not positive def.:   {npd} / {K} ({100*npd/K:.1f}%)")
    print(f"  K-points needing regulariz.:  {int(frac_reg*K)} / {K} ({100*frac_reg:.1f}%)")
    print()

    status_symbols = {'good': '✓ GOOD', 'marginal': '⚠ MARGINAL', 'bad': '✗ BAD'}
    print(f"  STATUS: {status_symbols.get(result.status, result.status)}")

    if result.status == 'bad':
        print()
        print(f"  The Fourier sum with {result.num_R} R-vectors does not produce")
        print(f"  well-conditioned overlap matrices. This will lead to:")
        print(f"    - Large Wannier function spreads")
        print(f"    - Poor band interpolation quality")
        print(f"    - Numerically unreliable results")
        print()
        print(f"  REMEDY: Re-run the DFT calculation with more R-vectors")
        print(f"  (increase TOLINTEG in CRYSTAL or the real-space grid cutoff).")
        print(f"  Typical good values: 80-150+ R-vectors for N={result.num_orbitals}.")
    elif result.status == 'marginal':
        print()
        print(f"  The overlap matrices show moderate ill-conditioning.")
        print(f"  Results may be acceptable but could benefit from more R-vectors.")

    print(f"{'=' * 70}")


def validate_overlap_conditioning(
    stacked: StackedMatrices,
    k_grid: Optional[Tuple[int, int, int]] = None,
    num_sample: int = 100,
    overlap_threshold: float = 1e-6,
    condition_threshold_bad: float = 1e8,
    condition_threshold_marginal: float = 1e4,
    min_eigenvalue_threshold: float = 1e-4,
    verbose: bool = True,
    strict: bool = True,
) -> OverlapConditioningResult:
    """
    Validate the conditioning of overlap matrices S(k) before eigensolving.

    Samples k-points from the Brillouin zone, Fourier-assembles S(k),
    and checks condition numbers and positive-definiteness.

    Parameters
    ----------
    stacked : StackedMatrices
        Pre-stacked R-space matrices.
    k_grid : tuple of 3 ints, optional
        K-point grid dimensions. Used to detect 2D systems (kz=1).
    num_sample : int
        Number of k-points to sample (default: 100).
    overlap_threshold : float
        Eigenvalues below this count as needing regularization (default: 1e-6).
    condition_threshold_bad : float
        Condition number above this → status 'bad' (default: 1e8).
    condition_threshold_marginal : float
        Condition number above this → status 'marginal' (default: 1e4).
    min_eigenvalue_threshold : float
        Min eigenvalue below this → status 'marginal' (default: 1e-4).
    verbose : bool
        Print diagnostic report (default: True).
    strict : bool
        Raise OverlapConditioningError for 'bad' status (default: True).

    Returns
    -------
    OverlapConditioningResult

    Raises
    ------
    OverlapConditioningError
        If strict=True and status is 'bad'.
    """
    # 1. Build sample k-points.
    # Cap the sample to the actual grid size: the calculation only has
    # prod(k_grid) k-points, so a hardcoded 100 samples wastes eigvalsh calls
    # (~13 s at 6x6, more at larger N). See PARALLELIZATION_PLAN.md (TW-4).
    if k_grid is not None:
        num_sample = min(num_sample, int(np.prod(k_grid)))
    k_sample = _build_sample_kpoints(k_grid, num_sample)
    K = len(k_sample)

    # 2. Batch Fourier transform S(k) only
    _, S_all = fourier_all_kpoints(k_sample, stacked, S_only=True)

    # 3. Analyze each S(k)
    condition_numbers = np.zeros(K)
    min_eigenvalues = np.zeros(K)

    for i in range(K):
        S_k = S_all[i]
        # Force Hermiticity (numerical noise)
        S_k = 0.5 * (S_k + S_k.conj().T)
        eigvals = np.linalg.eigvalsh(S_k)

        lambda_min = eigvals[0]
        lambda_max = eigvals[-1]
        min_eigenvalues[i] = lambda_min

        if lambda_min > 0:
            condition_numbers[i] = lambda_max / lambda_min
        else:
            condition_numbers[i] = np.inf

    # 4. Aggregate metrics
    worst_idx = np.argmax(condition_numbers)
    min_eig_idx = np.argmin(min_eigenvalues)

    num_not_pd = int(np.sum(min_eigenvalues <= 0))
    num_needing_reg = int(np.sum(min_eigenvalues < overlap_threshold))

    finite_mask = np.isfinite(condition_numbers)
    if np.any(finite_mask):
        mean_cond = float(np.exp(np.mean(np.log(condition_numbers[finite_mask]))))
        median_cond = float(np.median(condition_numbers[finite_mask]))
    else:
        mean_cond = np.inf
        median_cond = np.inf

    global_min_eig = float(min_eigenvalues[min_eig_idx])
    worst_cond = float(condition_numbers[worst_idx])

    # 5. Determine status
    if num_not_pd > 0 or worst_cond > condition_threshold_bad:
        status = 'bad'
    elif (worst_cond > condition_threshold_marginal
          or global_min_eig < min_eigenvalue_threshold):
        status = 'marginal'
    else:
        status = 'good'

    # 6. Build message
    if status == 'bad':
        message = (
            f"Overlap matrices severely ill-conditioned with {stacked.num_R} R-vectors. "
            f"Worst cond(S)={worst_cond:.2e}, min λ(S)={global_min_eig:.2e}, "
            f"{num_not_pd}/{K} k-points not positive definite. "
            f"Re-run DFT with more R-vectors (need 80-150+)."
        )
    elif status == 'marginal':
        message = (
            f"Overlap matrices moderately ill-conditioned with {stacked.num_R} R-vectors. "
            f"Worst cond(S)={worst_cond:.2e}, min λ(S)={global_min_eig:.2e}. "
            f"Results may be acceptable but could improve with more R-vectors."
        )
    else:
        message = (
            f"Overlap matrices well-conditioned with {stacked.num_R} R-vectors. "
            f"Worst cond(S)={worst_cond:.2e}, min λ(S)={global_min_eig:.2e}."
        )

    # 7. Build result
    result = OverlapConditioningResult(
        num_R=stacked.num_R,
        num_orbitals=stacked.num_orbitals,
        num_kpoints_sampled=K,
        worst_condition_number=worst_cond,
        worst_condition_kpoint=k_sample[worst_idx].copy(),
        min_eigenvalue=global_min_eig,
        min_eigenvalue_kpoint=k_sample[min_eig_idx].copy(),
        mean_condition_number=mean_cond,
        median_condition_number=median_cond,
        num_not_positive_definite=num_not_pd,
        fraction_needing_regularization=num_needing_reg / K,
        condition_numbers=condition_numbers,
        min_eigenvalues=min_eigenvalues,
        sampled_kpoints=k_sample,
        status=status,
        message=message,
    )

    # 8. Print and act
    if verbose:
        _print_conditioning_report(result)

    if strict and status == 'bad':
        raise OverlapConditioningError(result)
    elif status == 'marginal':
        warnings.warn(result.message, stacklevel=2)

    return result
