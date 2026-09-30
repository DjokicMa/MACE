"""
Utility Functions Module

This module contains helper functions for data format conversion,
matrix organization, and rigorous symmetry checking.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

# ==============================
# Data Preparation & Conversion
# ==============================


def prune_zero_rvectors(
    real_space_matrices: Dict[Tuple[int, int, int], Dict[str, np.ndarray]],
    threshold: float = 0.0,
    verbose: bool = True,
) -> Tuple[Dict[Tuple[int, int, int], Dict[str, np.ndarray]], int]:
    """Drop R-vectors whose H(R) and S(R) are both negligible.

    CRYSTAL emits real-space H(R)/S(R) for a generous number of cells (set by
    the requested radial shells), so the far cells come back all-zero. Since
    H(k) = sum_R e^{i k.R} H(R) (and likewise S(k)), an all-zero cell adds
    nothing at any k. Removing it is therefore exact for ``threshold == 0`` and
    shrinks every R-indexed structure downstream — most importantly the
    engine's stacked H/S arrays, which persist through Stage 2.

    The origin cell (0,0,0) is never pruned (it always carries on-site terms).
    After (R,-R) symmetrization zero cells occur as both-zero pairs, so pruning
    preserves Hermiticity of H(k).

    Parameters
    ----------
    real_space_matrices : dict
        Maps R-tuple -> {'H': matrix, 'S': matrix}.
    threshold : float
        A cell is pruned if max|H(R)| <= threshold and max|S(R)| <= threshold.
        0.0 (default) prunes only exactly-zero cells (bit-identical results).
        A small positive value (e.g. 1e-10) also drops numerically negligible
        cells — faster/smaller but no longer guaranteed bit-identical.
    verbose : bool
        Print a one-line summary.

    Returns
    -------
    (pruned_matrices, num_pruned)
    """
    kept = {}
    pruned = 0
    for R, mats in real_space_matrices.items():
        if R == (0, 0, 0):
            kept[R] = mats
            continue
        H = mats.get('H')
        S = mats.get('S')
        h_max = float(np.abs(H).max()) if H is not None and H.size else 0.0
        s_max = float(np.abs(S).max()) if S is not None and S.size else 0.0
        if h_max <= threshold and s_max <= threshold:
            pruned += 1
            continue
        kept[R] = mats
    if verbose and pruned:
        total = pruned + len(kept)
        mode = "exact-zero" if threshold == 0.0 else f"|.|<={threshold:g}"
        print(f"  Pruned {pruned} zero R-vectors ({mode}): "
              f"{total} -> {len(kept)}")
    return kept, pruned

def prepare_real_space_matrices(
    H_full_list: List[Tuple[np.ndarray, np.ndarray]],
    S_full_list: List[Tuple[np.ndarray, np.ndarray]],
    lattice_vectors: np.ndarray,
    prune_tol: float = 0.0,
    prune_verbose: bool = True,
) -> Dict[Tuple[int, int, int], Dict[str, np.ndarray]]:
    """
    Convert from parsed (Cartesian) format to the (Integer) lattice format
    expected by the main engine.
    
    Parameters
    ----------
    H_full_list : list
        List of (R_cartesian, H_matrix) from parser
    S_full_list : list
        List of (R_cartesian, S_matrix) from parser
    lattice_vectors : ndarray
        3x3 matrix of real-space lattice vectors
    
    Returns
    -------
    real_space_matrices : dict
        Maps (n1, n2, n3) -> {'H': H_matrix, 'S': S_matrix}
    """
    real_space_matrices = {}
    
    # Process Hamiltonian matrices
    for R_cartesian, H_matrix in H_full_list:
        # Solve R_cart = n * A to find integers n
        # Rounding is necessary due to floating point noise
        R_integer = np.round(np.linalg.solve(lattice_vectors.T, R_cartesian)).astype(int)
        R_tuple = tuple(R_integer)
        
        if R_tuple not in real_space_matrices:
            real_space_matrices[R_tuple] = {}
        real_space_matrices[R_tuple]['H'] = H_matrix
    
    # Process overlap matrices
    for R_cartesian, S_matrix in S_full_list:
        R_integer = np.round(np.linalg.solve(lattice_vectors.T, R_cartesian)).astype(int)
        R_tuple = tuple(R_integer)

        if R_tuple not in real_space_matrices:
            real_space_matrices[R_tuple] = {}
        real_space_matrices[R_tuple]['S'] = S_matrix

    # Prune negligible R-vectors before they reach the Fourier sum. tol=0 removes
    # only exact-zero matrices (exact: e^{ik.R}*0 contributes nothing), which alone
    # drops the overcomplete tail CRYSTAL often emits.
    return prune_real_space_matrices(real_space_matrices, tol=prune_tol,
                                     verbose=prune_verbose)


def prune_real_space_matrices(real_space_matrices, tol=0.0, warn_threshold=1e-3,
                              verbose=True):
    """Drop R-vectors whose H(R) and S(R) are both negligible.

    The Fourier assembly H(k)=sum_R e^{ik.R} H(R) (and the MMN's S(k+b/2)) costs
    work proportional to the number of R-vectors, so all-zero or vanishingly small
    R-vector matrices are pure overhead — especially for the memory-bound MMN,
    whose traffic scales with nR.

    Parameters
    ----------
    real_space_matrices : dict
        Maps (n1,n2,n3) -> {'H': ndarray, 'S': ndarray}.
    tol : float
        Prune an R-vector if max(|H(R)|, |S(R)|) <= tol. Default 0.0 removes only
        exact-zero matrices. A positive tol prunes a near-zero / overcomplete set.
    warn_threshold : float
        If the largest *discarded* magnitude exceeds this, emit a warning — the
        tolerance is high enough to be dropping real hopping/overlap.
    verbose : bool
        Print a one-line pruning summary.

    Returns
    -------
    dict
        A new dict with negligible R-vectors removed. The onsite R=(0,0,0) is
        never pruned.
    """
    import warnings
    onsite = (0, 0, 0)
    kept, pruned = {}, []
    for R, mats in real_space_matrices.items():
        if tuple(R) == onsite:
            kept[R] = mats
            continue
        mag = 0.0
        if mats.get('H') is not None:
            mag = max(mag, float(np.abs(mats['H']).max()))
        if mats.get('S') is not None:
            mag = max(mag, float(np.abs(mats['S']).max()))
        if mag <= tol:
            pruned.append(mag)
        else:
            kept[R] = mats
    if pruned:
        max_pruned = max(pruned)
        if verbose:
            print(f"  Pruned {len(pruned)}/{len(real_space_matrices)} R-vectors "
                  f"(max|H,S| <= {tol:g}); largest discarded magnitude {max_pruned:.2e}")
        if max_pruned > warn_threshold:
            warnings.warn(
                f"R-vector pruning (tol={tol:g}) discarded a contribution of magnitude "
                f"{max_pruned:.2e} > {warn_threshold:g} — this may remove real "
                f"hopping/overlap and degrade results. Lower the tolerance if accuracy drops.",
                stacklevel=2,
            )
    elif verbose:
        print(f"  R-vector pruning (tol={tol:g}): nothing to prune "
              f"({len(real_space_matrices)} kept).")
    return kept


def organize_matrices_by_lattice_vector(
    matrices: List[dict]
) -> Tuple[Dict[Tuple[int, int, int], Dict[str, np.ndarray]], 
           Dict[Tuple[int, int, int], np.ndarray]]:
    """
    Organize raw parsed matrix blocks by lattice vector and spin channel.
    
    This bridges the gap between 'parse_overlap_and_fock_matrices' and
    'create_spin_block_matrices'.
    """
    H_R_dict = {}
    S_R_dict = {}
    
    for matrix_info in matrices:
        matrix_type = matrix_info['type']
        lattice_vector = tuple(matrix_info['lattice_vector'])
        data = matrix_info['data']
        
        if data is None:
            continue
            
        if matrix_type == 'overlap':
            S_R_dict[lattice_vector] = data
        elif matrix_type == 'fock':
            spin_channel = matrix_info['spin_channel']
            if lattice_vector not in H_R_dict:
                H_R_dict[lattice_vector] = {}
            H_R_dict[lattice_vector][spin_channel] = data
    
    return H_R_dict, S_R_dict


def get_basis_size(matrices_dict: dict) -> int:
    """Get the basis size (N) from the matrices dictionary."""
    if not matrices_dict:
        return 0
    first_key = next(iter(matrices_dict))
    
    if 'H' in matrices_dict[first_key]:
        # H is 2N x 2N, we want N (spatial orbitals)
        return matrices_dict[first_key]['H'].shape[0] // 2
    elif 'ALPHA_ALPHA' in matrices_dict[first_key]:
        # Raw dict is N x N
        return matrices_dict[first_key]['ALPHA_ALPHA'].shape[0]
    return 0


# ==============================
# Verification & Diagnostics
# ==============================

def verify_matrix_symmetry(
    real_space_matrices: Dict[Tuple[int, int, int], Dict[str, np.ndarray]],
    tolerance: float = 1e-10
) -> bool:
    """
    Verify that the constructed matrices satisfy the fundamental physical symmetries:
    1. H(0) is Hermitian.
    2. H(R) = H(-R)†
    3. S(R) = S(-R)^T
    
    Returns True if all checks pass.
    """
    print("\n" + "-" * 60)
    print("VERIFYING MATRIX SYMMETRY AND HERMITICITY")
    print("-" * 60)
    
    all_passed = True
    max_error_H = 0.0
    max_error_S = 0.0
    
    # Check Origin
    origin = (0, 0, 0)
    if origin in real_space_matrices:
        mats = real_space_matrices[origin]
        if 'H' in mats:
            # Check H(0) == H(0)†
            diff = np.max(np.abs(mats['H'] - mats['H'].conj().T))
            if diff > tolerance:
                print(f"FAIL: Origin H(0) is not Hermitian. Max Diff: {diff:.2e}")
                all_passed = False
            max_error_H = max(max_error_H, diff)
            
    # Check Pairs
    checked_R = set()
    for R in real_space_matrices:
        if R == (0, 0, 0) or R in checked_R:
            continue
            
        minus_R = tuple(-x for x in R)
        if minus_R not in real_space_matrices:
            print(f"WARNING: pair {minus_R} missing for {R}")
            continue
            
        # Check Hamiltonian: H(R) - H(-R)† == 0
        if 'H' in real_space_matrices[R] and 'H' in real_space_matrices[minus_R]:
            H_R = real_space_matrices[R]['H']
            H_mR = real_space_matrices[minus_R]['H']
            
            diff = np.max(np.abs(H_R - H_mR.conj().T))
            max_error_H = max(max_error_H, diff)
            
            if diff > tolerance:
                print(f"FAIL H: Pair {R}/{minus_R} symmetry violation. Diff: {diff:.2e}")
                all_passed = False

        # Check Overlap: S(R) - S(-R)^T == 0 (Transpose only, S is real)
        if 'S' in real_space_matrices[R] and 'S' in real_space_matrices[minus_R]:
            S_R = real_space_matrices[R]['S']
            S_mR = real_space_matrices[minus_R]['S']
            
            diff = np.max(np.abs(S_R - S_mR.T))
            max_error_S = max(max_error_S, diff)
            
            if diff > tolerance:
                print(f"FAIL S: Pair {R}/{minus_R} symmetry violation. Diff: {diff:.2e}")
                all_passed = False

        checked_R.add(R)
        checked_R.add(minus_R)

    print(f"Max Symmetry Error H: {max_error_H:.2e}")
    print(f"Max Symmetry Error S: {max_error_S:.2e}")
    
    if all_passed:
        print("SUCCESS: All matrix symmetries are satisfied.")
    else:
        print("FAILURE: Matrix symmetries violated.")
        
    return all_passed


def check_matrix_consistency(
    real_space_matrices: Dict[Tuple[int, int, int], Dict[str, np.ndarray]]
) -> bool:
    """Check that all matrices have consistent dimensions (square and equal size)."""
    sizes = []
    
    for R_tuple, matrices in real_space_matrices.items():
        if 'H' in matrices:
            H_shape = matrices['H'].shape
            if H_shape[0] != H_shape[1]:
                print(f"Warning: H matrix at R={R_tuple} is not square")
                return False
            sizes.append(H_shape[0])
        
        if 'S' in matrices:
            S_shape = matrices['S'].shape
            if S_shape[0] != S_shape[1]:
                print(f"Warning: S matrix at R={R_tuple} is not square")
                return False
            sizes.append(S_shape[0])
    
    if len(set(sizes)) > 1:
        print(f"Warning: Inconsistent matrix sizes: {set(sizes)}")
        return False
    
    return True


# ==============================
# Reporting
# ==============================

def print_matrix_summary(
    real_space_matrices: Dict[Tuple[int, int, int], Dict[str, np.ndarray]]
) -> None:
    """Print a summary of the real-space matrices."""
    print("\n" + "=" * 70)
    print("Real-Space Matrices Summary")
    print("=" * 70)
    
    num_R_vectors = len(real_space_matrices)
    print(f"Number of R-vectors: {num_R_vectors}")
    
    if num_R_vectors > 0:
        # Get matrix size
        first_key = next(iter(real_space_matrices))
        if 'H' in real_space_matrices[first_key]:
            dim = real_space_matrices[first_key]['H'].shape[0]
            print(f"Hamiltonian Dimension: {dim} × {dim}")
    
    # List R-vectors
    print("\nR-vectors:")
    for R_tuple in sorted(real_space_matrices.keys()):
        has_H = 'H' in real_space_matrices[R_tuple]
        has_S = 'S' in real_space_matrices[R_tuple]
        print(f"  {R_tuple}: H={'✓' if has_H else '✗'}, S={'✓' if has_S else '✗'}")
    
    print("=" * 70)


def print_calculation_info(
    num_kpoints: int,
    k_grid: Tuple[int, int, int],
    num_orbitals: int,
    num_wann: int
) -> None:
    """Print information about the calculation parameters and memory."""
    print("\n" + "=" * 70)
    print("Calculation Information")
    print("=" * 70)
    print(f"K-point grid: {k_grid[0]} × {k_grid[1]} × {k_grid[2]}")
    print(f"Number of k-points: {num_kpoints}")
    print(f"Number of orbitals: {num_orbitals}")
    print(f"Number of Wannier functions: {num_wann}")
    
    # Memory estimate
    bytes_per_complex = 16
    matrices_mb = 2 * num_kpoints * num_orbitals**2 * bytes_per_complex / 1e6
    eigenvalues_mb = num_kpoints * num_wann * 8 / 1e6
    eigenvectors_mb = num_kpoints * num_orbitals * num_wann * bytes_per_complex / 1e6
    total_mb = matrices_mb + eigenvalues_mb + eigenvectors_mb
    
    print(f"\nEstimated memory usage:")
    print(f"  Matrices (H, S): {matrices_mb:.1f} MB")
    print(f"  Eigenvalues:     {eigenvalues_mb:.1f} MB")
    print(f"  Eigenvectors:    {eigenvectors_mb:.1f} MB")
    print(f"  Total:           {total_mb:.1f} MB")
    print("=" * 70)