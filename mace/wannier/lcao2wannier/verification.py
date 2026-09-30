"""
Verification Module

This module contains functions for verifying the numerical accuracy
and physical correctness of computed quantities, including 
real-space matrix symmetries and k-space properties.
"""

import numpy as np
import warnings
from typing import List, Tuple, Dict

# ==============================
# Basic Utility
# ==============================

def is_hermitian(matrix: np.ndarray, tol: float = 1e-10) -> bool:
    """Check if a matrix is Hermitian."""
    return np.allclose(matrix, matrix.conj().T, atol=tol)


# ==============================
# Real-Space Verification
# ==============================

def verify_real_space_symmetry(
    real_space_matrices: Dict[Tuple[int, int, int], Dict[str, np.ndarray]],
    tolerance: float = 1e-10,
    verbose: bool = True
) -> bool:
    """
    Verify that the constructed real-space matrices satisfy fundamental physical symmetries:
    1. H(0) is Hermitian.
    2. H(R) = H(-R)†
    3. S(R) = S(-R)^T
    
    Parameters
    ----------
    real_space_matrices : dict
        Dictionary mapping (R_int) -> {'H': matrix, 'S': matrix}
    tolerance : float
        Numerical tolerance for checks
    verbose : bool
        Whether to print detailed output
    
    Returns
    -------
    bool
        True if all checks pass.
    """
    if verbose:
        print("\n" + "-" * 60)
        print("VERIFYING REAL-SPACE MATRIX SYMMETRIES")
        print("-" * 60)
    
    all_passed = True
    max_error_H = 0.0
    max_error_S = 0.0
    
    # 1. Check Origin
    origin = (0, 0, 0)
    if origin in real_space_matrices:
        mats = real_space_matrices[origin]
        if 'H' in mats:
            # Check H(0) == H(0)†
            diff = np.max(np.abs(mats['H'] - mats['H'].conj().T))
            if diff > tolerance:
                if verbose:
                    print(f"FAIL: Origin H(0) is not Hermitian. Max Diff: {diff:.2e}")
                all_passed = False
            max_error_H = max(max_error_H, diff)
            
    # 2. Check Pairs
    checked_R = set()
    for R in real_space_matrices:
        if R == (0, 0, 0) or R in checked_R:
            continue
            
        minus_R = tuple(-x for x in R)
        
        # Check if pair exists
        if minus_R not in real_space_matrices:
            if verbose:
                print(f"WARNING: pair {minus_R} missing for {R}")
            continue
            
        # Check Hamiltonian: H(R) - H(-R)† == 0
        if 'H' in real_space_matrices[R] and 'H' in real_space_matrices[minus_R]:
            H_R = real_space_matrices[R]['H']
            H_mR = real_space_matrices[minus_R]['H']
            
            diff = np.max(np.abs(H_R - H_mR.conj().T))
            max_error_H = max(max_error_H, diff)
            
            if diff > tolerance:
                if verbose:
                    print(f"FAIL H: Pair {R}/{minus_R} symmetry violation. Diff: {diff:.2e}")
                all_passed = False

        # Check Overlap: S(R) - S(-R)^T == 0 (Transpose only, S is real)
        if 'S' in real_space_matrices[R] and 'S' in real_space_matrices[minus_R]:
            S_R = real_space_matrices[R]['S']
            S_mR = real_space_matrices[minus_R]['S']
            
            diff = np.max(np.abs(S_R - S_mR.T))
            max_error_S = max(max_error_S, diff)
            
            if diff > tolerance:
                if verbose:
                    print(f"FAIL S: Pair {R}/{minus_R} symmetry violation. Diff: {diff:.2e}")
                all_passed = False

        checked_R.add(R)
        checked_R.add(minus_R)

    if verbose:
        print(f"  Max Real-Space H(R) Symmetry Error: {max_error_H:.2e}")
        print(f"  Max Real-Space S(R) Symmetry Error: {max_error_S:.2e}")
    
    if max_error_H > tolerance:
        warnings.warn(f"Real-space H(R) symmetry violated (Max Err: {max_error_H:.2e})")
    if max_error_S > tolerance:
        warnings.warn(f"Real-space S(R) symmetry violated (Max Err: {max_error_S:.2e})")
        
    return all_passed


# ==============================
# K-Space Verification
# ==============================

def verify_hermiticity(
    H_k_list: List[np.ndarray],
    S_k_list: List[np.ndarray],
    tol: float = 1e-10,
    verbose: bool = True
) -> Tuple[float, float]:
    """Verify that H(k) and S(k) are Hermitian for all k-points."""
    if verbose:
        print("\nVerifying Hermiticity of H(k) and S(k)...")
    
    max_H_deviation = 0.0
    max_S_deviation = 0.0
    
    for k_idx in range(len(H_k_list)):
        H_k = H_k_list[k_idx]
        S_k = S_k_list[k_idx]
        
        H_deviation = np.max(np.abs(H_k - H_k.conj().T))
        S_deviation = np.max(np.abs(S_k - S_k.conj().T))
        
        max_H_deviation = max(max_H_deviation, H_deviation)
        max_S_deviation = max(max_S_deviation, S_deviation)
    
    if verbose:
        print(f"  Max H(k) Hermiticity deviation: {max_H_deviation:.2e}")
        print(f"  Max S(k) Hermiticity deviation: {max_S_deviation:.2e}")
    
    if max_H_deviation > tol:
        warnings.warn(f"H(k) is not Hermitian within tolerance {tol}")
    if max_S_deviation > tol:
        warnings.warn(f"S(k) is not Hermitian within tolerance {tol}")
    
    return max_H_deviation, max_S_deviation


def verify_orthonormality(
    eigenvectors_list: List[np.ndarray],
    S_k_list: List[np.ndarray],
    num_wann: int,
    num_check: int = 5,
    tol: float = 1e-8,
    verbose: bool = True
) -> float:
    """Verify that eigenvectors satisfy C†(k) S(k) C(k) ≈ I."""
    if verbose:
        print(f"\nVerifying orthonormality at {num_check} k-points...")
    
    num_kpoints = len(eigenvectors_list)
    # Ensure we don't try to check more points than exist
    actual_checks = min(num_check, num_kpoints)
    check_indices = np.linspace(0, num_kpoints - 1, actual_checks, dtype=int)
    
    max_deviation = 0.0
    
    for k_idx in check_indices:
        C_k = eigenvectors_list[k_idx]
        S_k = S_k_list[k_idx]
        
        # Compute C†(k) S(k) C(k)
        overlap = C_k.conj().T @ S_k @ C_k
        
        # Should be identity matrix
        identity = np.eye(num_wann)
        deviation = np.max(np.abs(overlap - identity))
        
        max_deviation = max(max_deviation, deviation)
        
        if verbose:
            print(f"  k-point {k_idx}: max deviation from identity = {deviation:.2e}")
        
        if deviation > tol:
            warnings.warn(f"Orthonormality check failed at k-point {k_idx}")
    
    return max_deviation


def read_mmn_blocks(path: str) -> Tuple[Dict[Tuple[int, int, Tuple[int, int, int]], np.ndarray], int, int, int]:
    """
    Parse a .mmn file into its individual M(k,b) blocks, keyed by (k, k_next, G).

    Unlike ``spread.read_mmn`` (which drops the G-shift columns and so cannot
    identify which b-vector a block belongs to), this keeps the G-shift as part
    of the key. That matters: for small meshes the same (k, k_next) index pair
    can appear more than once with different G.

    Parameters
    ----------
    path : str
        Path to the .mmn file

    Returns
    -------
    blocks : dict
        Maps (k, k_next, (G1, G2, G3)) -> M, a (num_bands x num_bands) complex
        array. k and k_next are 0-based.
    num_bands, num_kpoints, nntot : int
        Header quantities from line 2 of the file.
    """
    with open(path) as f:
        lines = f.read().splitlines()

    num_bands, num_kpoints, nntot = map(int, lines[1].split())
    nbb = num_bands * num_bands

    blocks = {}
    base = 2
    for _ in range(num_kpoints * nntot):
        h = lines[base].split()
        k = int(h[0]) - 1
        k_next = int(h[1]) - 1
        G = (int(h[2]), int(h[3]), int(h[4]))

        arr = np.fromstring(" ".join(lines[base + 1:base + 1 + nbb]),
                            sep=" ").reshape(nbb, 2)
        # Column-major: the writer's element loop is `for n: for m: M[m, n]`
        # (see _format_complex_block in wannier90.py).
        blocks[(k, k_next, G)] = (arr[:, 0] + 1j * arr[:, 1]).reshape(
            num_bands, num_bands, order='F')
        base += nbb + 1

    return blocks, num_bands, num_kpoints, nntot


def verify_mmn_adjoint_pairs(
    blocks: Dict[Tuple[int, int, Tuple[int, int, int]], np.ndarray],
    tol: float = 1e-10,
    verbose: bool = True
) -> Tuple[float, dict]:
    """
    Verify the .mmn adjoint-pair relation M(k+b, -b) = M(k, b)†.

    Wannier90's .mmn stores M^(k,b)_mn = <u_mk | u_n,k+b>. Every entry keyed
    (k, k', G) -- meaning k + b = k_{k'} + G -- has a partner entry keyed
    (k', k, -G), and the two are related by

        blocks[(k', k, -G)] == blocks[(k, k', G)].conj().T

    This is an adjoint relation BETWEEN TWO DISTINCT ENTRIES, not
    self-adjointness of any single M: no entry is ever its own partner. It is
    insensitive to the unitarity of M, so it does NOT detect the midpoint
    method's non-unitarity. What it does catch is conjugation or operand-order
    slips, one-sided phases or G-shifts, an overlap evaluated at the wrong
    k-point, and an asymmetric neighbour list.

    Assumes the cell-gauge Bloch phase exp(2*pi*i k.R) used throughout this
    package (fourier.py, gto_mmn.py), for which C(k) is strictly G-periodic:
    no extra exp(-i G.tau) phase is applied or needed.

    Parameters
    ----------
    blocks : dict
        Output of :func:`read_mmn_blocks`.
    tol : float
        Numerical tolerance for the pair relation.
    verbose : bool
        Whether to print detailed output.

    Returns
    -------
    max_error : float
        Largest elementwise deviation over all pairs that exist.
    report : dict
        Keys 'n_pairs', 'max_error', 'argmax_key', 'partner_key',
        'missing_partners' (list of keys whose partner is absent).
    """
    if verbose:
        print("\nVerifying .mmn adjoint pairs M(k+b,-b) = M(k,b)†...")

    max_error = 0.0
    argmax_key = None
    partner_key = None
    missing_partners = []
    checked = set()
    n_pairs = 0

    for key in blocks:
        k, k_next, G = key
        partner = (k_next, k, tuple(-g for g in G))

        if partner not in blocks:
            missing_partners.append(key)
            if verbose:
                print(f"  MISSING: no partner {partner} for entry {key}")
            continue

        if key in checked:
            continue

        deviation = np.max(np.abs(blocks[partner] - blocks[key].conj().T))
        if deviation > max_error:
            max_error = deviation
            argmax_key = key
            partner_key = partner

        checked.add(key)
        checked.add(partner)
        n_pairs += 1

    if verbose:
        print(f"  Pairs checked: {n_pairs}")
        print(f"  Max adjoint-pair deviation: {max_error:.2e}")
        if argmax_key is not None:
            print(f"  Worst pair: {argmax_key} -> {partner_key}")

    if missing_partners and verbose:
        print(f"  {len(missing_partners)} entrie(s) without a partner")

    if max_error > tol:
        warnings.warn(
            f"MMN adjoint-pair relation violated (Max Err: {max_error:.2e}) "
            f"at {argmax_key} -> {partner_key}"
        )

    report = {
        'n_pairs': n_pairs,
        'max_error': max_error,
        'argmax_key': argmax_key,
        'partner_key': partner_key,
        'missing_partners': missing_partners,
    }
    return max_error, report


def verify_mmn_file_adjoint_pairs(
    path: str,
    tol: float = 1e-10,
    verbose: bool = True
) -> Tuple[float, dict]:
    """Read a .mmn file and check its adjoint-pair relation.

    Convenience wrapper: :func:`read_mmn_blocks` then
    :func:`verify_mmn_adjoint_pairs`. Returns (max_error, report).
    """
    blocks, _nb, _nk, _nntot = read_mmn_blocks(path)
    return verify_mmn_adjoint_pairs(blocks, tol=tol, verbose=verbose)


def verify_eigenvalue_sorting(
    eigenvalues_list: List[np.ndarray],
    verbose: bool = True
) -> bool:
    """Verify that eigenvalues are sorted in ascending order at each k-point."""
    if verbose:
        print("\nVerifying eigenvalue sorting...")
    
    all_sorted = True
    
    for k_idx, eigenvalues in enumerate(eigenvalues_list):
        if not np.all(eigenvalues[:-1] <= eigenvalues[1:]):
            all_sorted = False
            if verbose:
                print(f"  Warning: Eigenvalues at k-point {k_idx} are not sorted")
    
    if verbose and all_sorted:
        print("  ✓ All eigenvalues are properly sorted")
    
    return all_sorted


def compute_band_gaps(
    eigenvalues_list: List[np.ndarray],
    num_wann: int
) -> Tuple[float, float, int]:
    """Compute the minimum direct band gap across all k-points."""
    gaps = []
    
    for k_idx, eigenvalues in enumerate(eigenvalues_list):
        if len(eigenvalues) >= 2:
            gap = eigenvalues[-1] - eigenvalues[0]
            gaps.append((gap, k_idx))
    
    if gaps:
        min_gap, k_idx_min = min(gaps)
        max_gap, k_idx_max = max(gaps)
        return min_gap, max_gap, k_idx_min
    else:
        return 0.0, 0.0, 0


def verify_energy_range(
    eigenvalues_list: List[np.ndarray],
    verbose: bool = True
) -> Tuple[float, float]:
    """Check the energy range of computed eigenvalues."""
    if not eigenvalues_list:
        return 0.0, 0.0
        
    all_eigenvalues = np.concatenate(eigenvalues_list)
    E_min = np.min(all_eigenvalues.real)
    E_max = np.max(all_eigenvalues.real)
    
    if verbose:
        print("\nEnergy range:")
        print(f"  Minimum eigenvalue: {E_min:.6f}")
        print(f"  Maximum eigenvalue: {E_max:.6f}")
        print(f"  Energy span: {E_max - E_min:.6f}")
    
    return E_min, E_max


def run_all_verifications(
    eigenvalues_list: List[np.ndarray],
    eigenvectors_list: List[np.ndarray],
    H_k_list: List[np.ndarray],
    S_k_list: List[np.ndarray],
    num_wann: int,
    verbose: bool = True
) -> dict:
    """Run all k-space verification checks and return results."""
    results = {}
    
    # Hermiticity check
    max_H_dev, max_S_dev = verify_hermiticity(H_k_list, S_k_list, verbose=verbose)
    results['hermiticity'] = {'H_deviation': max_H_dev, 'S_deviation': max_S_dev}
    
    # Orthonormality check
    max_ortho_dev = verify_orthonormality(
        eigenvectors_list, S_k_list, num_wann, verbose=verbose
    )
    results['orthonormality'] = {'max_deviation': max_ortho_dev}
    
    # Eigenvalue sorting check
    sorting_ok = verify_eigenvalue_sorting(eigenvalues_list, verbose=verbose)
    results['eigenvalue_sorting'] = {'sorted': sorting_ok}
    
    # Energy range
    E_min, E_max = verify_energy_range(eigenvalues_list, verbose=verbose)
    results['energy_range'] = {'E_min': E_min, 'E_max': E_max}
    
    return results


class HandoffValidationError(RuntimeError):
    """A serialized hand-off failed final acceptance (see message for why)."""


def validate_handoff(eig_path: str, amn_path: str, mmn_path: str,
                     num_wann: int, tau: float = 1e-6,
                     verbose: bool = True) -> dict:
    """Full-coverage acceptance check of a serialized N_b = N_w hand-off.

    Validates what was actually WRITTEN (the formatted files), not the
    in-memory arrays, so serialization is inside the acceptance boundary:

      * every value in .eig/.amn/.mmn is finite,
      * the .mmn header matches num_wann (isolated manifold),
      * sigma_max(M^(k,b)) <= 1 + tau over EVERY (k,b) block -- no sampling
        (the hand-off is contractive by construction; a violation means a
        broken gauge/rotation, not noise),
      * the SVD itself succeeds on every block.

    tau covers SVD roundoff plus text-serialization rounding; 1e-6 is the
    same criterion the LAPW frontend and the paper use. Raises
    HandoffValidationError on any failure -- never repairs, clips, or
    unitarizes. Returns {'max_sv': float, 'n_blocks': int}.
    """
    import os
    for p in (eig_path, amn_path, mmn_path):
        if not os.path.exists(p):
            raise HandoffValidationError(f"hand-off file missing: {p}")

    eig = np.loadtxt(eig_path, ndmin=2)
    if not np.isfinite(eig).all():
        raise HandoffValidationError(
            f"{eig_path}: non-finite eigenvalue(s) "
            f"({np.size(eig) - np.isfinite(eig).sum()} bad entries)")

    amn = np.loadtxt(amn_path, skiprows=2, ndmin=2)
    if not np.isfinite(amn).all():
        raise HandoffValidationError(
            f"{amn_path}: non-finite projection entrie(s) "
            f"({np.size(amn) - np.isfinite(amn).sum()} bad entries)")

    blocks, nb, nk, nntot = read_mmn_blocks(mmn_path)
    if nb != num_wann:
        raise HandoffValidationError(
            f"{mmn_path}: header num_bands = {nb} but the hand-off contract "
            f"is num_bands = num_wann = {num_wann}")
    max_sv = 0.0
    for key, M in blocks.items():
        if not np.isfinite(M).all():
            raise HandoffValidationError(
                f"{mmn_path}: non-finite overlap block at (k, k', G) = {key}")
        try:
            sv = np.linalg.svd(M, compute_uv=False)[0]
        except np.linalg.LinAlgError as e:
            raise HandoffValidationError(
                f"{mmn_path}: SVD failed on block (k, k', G) = {key}: {e}")
        if sv > max_sv:
            max_sv = float(sv)
            worst = key
    if max_sv > 1.0 + tau:
        raise HandoffValidationError(
            f"{mmn_path}: hand-off is NOT contractive: "
            f"sigma_max = {max_sv:.8f} > 1 + {tau:g} at (k, k', G) = "
            f"{worst}. The overlaps are not repaired or clipped; fix the "
            f"gauge/rotation that produced them.")
    if verbose:
        print(f"  hand-off acceptance: max SV = {max_sv:.6f} over "
              f"{len(blocks)} (k,b) blocks (full coverage, serialized "
              f"values; tolerance 1 + {tau:g})")
    return {'max_sv': max_sv, 'n_blocks': len(blocks)}
