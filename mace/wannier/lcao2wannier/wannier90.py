"""
Wannier90 File Writer Module

This module contains functions for writing Wannier90 input files
(.eig, .amn, .mmn) according to the Wannier90 file format specification.
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Optional

from .fourier import fourier_transform_to_kspace, fourier_transform_vectorized, StackedMatrices


def _format_complex_block(M: np.ndarray, order: str = 'F') -> str:
    """Format a complex matrix as one ``"<re> <im>"`` line per element.

    Byte-identical to a per-element
    ``f.write(f"{v.real:18.12f} {v.imag:18.12f}\n")`` loop, but builds the whole
    block with a single buffered string format. The per-element write loop was
    the dominant cost of serializing large ``.mmn`` files (millions of writes);
    this is ~2x faster and removes the per-element-write scaling cliff.

    ``order='F'`` (default) emits column-major order (first index fastest),
    matching a ``for n: for m: M[m, n]`` loop.
    """
    flat = np.asarray(M).ravel(order=order)
    data = np.empty(flat.size * 2, dtype=float)
    data[0::2] = flat.real
    data[1::2] = flat.imag
    return ("%18.12f %18.12f\n" * flat.size) % tuple(data)


def _format_amn_block(A_k: np.ndarray, num_bands: int, num_proj: int, k_idx: int) -> str:
    """Format one k-point's .amn block: ``"<band> <wann> <k> <re> <im>"`` lines.

    CONVENTION (both halves matter, and both were wrong before):

    * VALUE is the canonical ``A_mn(k) = <psi_m | g_n>``. The caller hands us
      ``A_k = (S C)[proj, band]``, i.e. ``A_k[n, m] = <chi_n | psi_m>``, so the
      emitted value is its CONJUGATE. pw2wannier90 computes the same quantity
      as ``zdotc(evc(:,ibnd), sgf(:,iw))`` -- band conjugated, projection not.
    * ORDER is PROJECTION-OUTER, band-inner, matching pw2wannier90
      (``DO iw ... DO ibnd ... WRITE``, pw2wannier90.f90:3598-3603).

    wannier90 itself reads by LABEL, so the order is cosmetic to it -- but
    positional readers (WannierBerri's AMN.from_w90_file slices the numeric
    columns and reshapes) silently misread a band-outer file. For a square
    .amn the old order and the old conjugation COMPOSED INTO A^H, which reads
    as a plausible, near-unitary, wrong gauge.

    Measured cost of the conjugation on h-BN 4x4x2 --method pdwf: Omega_I
    11.134242567 -> 11.134242566, eta 34.75 -> 35.00 meV. It is the
    time-reversed gauge, so wannier90 results are equivalent either way.
    """
    n_lines = num_bands * num_proj
    # proj outer, band inner; conjugate to the canonical <psi_m|g_n>
    vals = A_k[:num_proj, :num_bands].conj().ravel(order='C')
    data = np.empty(n_lines * 5, dtype=object)
    data[0::5] = np.tile(np.arange(1, num_bands + 1), num_proj)
    data[1::5] = np.repeat(np.arange(1, num_proj + 1), num_bands)
    data[2::5] = k_idx + 1
    data[3::5] = vals.real
    data[4::5] = vals.imag
    return ("%5d %5d %5d %18.12f %18.12f\n" * n_lines) % tuple(data)


def diagnose_mmn_matrix(
    M: np.ndarray,
    k_idx: int,
    next_k_idx: int,
    G_shift: Tuple[int, int, int],
    tolerance: float = 1e-6
) -> Dict[str, Any]:
    """
    Diagnose properties of an MMN matrix for debugging.

    Parameters
    ----------
    M : ndarray
        The MMN matrix, shape (num_bands, num_bands)
    k_idx : int
        Current k-point index
    next_k_idx : int
        Neighbor k-point index
    G_shift : tuple
        The G-vector shift (n1, n2, n3)
    tolerance : float
        Tolerance for numerical checks

    Returns
    -------
    diagnostics : dict
        Dictionary with diagnostic information:
        - 'trace': Trace of M (should be real and close to num_bands for identity)
        - 'det': Determinant of M (magnitude should be close to 1 for unitary)
        - 'is_identity': True if M is close to identity (for self-overlap)
        - 'singular_values': Singular values (should all be close to 1 for unitary)
        - 'max_off_diag': Maximum off-diagonal element magnitude
        - 'hermitian_error': ||M†M - I|| / num_bands
    """
    num_bands = M.shape[0]

    trace = np.trace(M)
    det = np.linalg.det(M)

    # Singular value decomposition
    U, s, Vh = np.linalg.svd(M)

    # Check if M is identity (for self-overlap with G=0)
    identity_error = np.linalg.norm(M - np.eye(num_bands)) / num_bands
    is_identity = identity_error < tolerance

    # Check off-diagonal elements
    diag_mask = np.eye(num_bands, dtype=bool)
    off_diag = np.abs(M[~diag_mask])
    max_off_diag = np.max(off_diag) if len(off_diag) > 0 else 0.0

    # Check unitarity: M†M should be identity
    MdagM = M.conj().T @ M
    hermitian_error = np.linalg.norm(MdagM - np.eye(num_bands)) / num_bands

    return {
        'k_idx': k_idx,
        'next_k_idx': next_k_idx,
        'G_shift': G_shift,
        'trace': trace,
        'det': det,
        'det_magnitude': np.abs(det),
        'is_identity': is_identity,
        'identity_error': identity_error,
        'singular_values': s,
        'sv_min': np.min(s),
        'sv_max': np.max(s),
        'max_off_diag': max_off_diag,
        'hermitian_error': hermitian_error,
        'is_unitary': hermitian_error < tolerance and np.abs(np.abs(det) - 1) < tolerance
    }


def compute_mmn_lowdin(
    k_idx: int,
    next_k_idx: int,
    C_tilde_list: List[np.ndarray],
    b_cart: np.ndarray = None,
    atom_positions: np.ndarray = None,
    basis_atom_map: np.ndarray = None,
    berry_phase: bool = True,
) -> np.ndarray:
    """
    Compute M_mn using the Lowdin orthogonalization approach.

    The Lowdin transformation orthogonalizes the eigenvectors for numerical
    stability. When berry_phase=True (default), the Berry phase
    exp(-i b_cart . tau_mu) encodes the position operator e^{-ib.r} that
    carries intra-cell atomic position information. Both are essential:
    without Lowdin, SVs diverge due to cond(S) ~ 10^4; without the Berry
    phase, Wannier centers are wrong because the method cannot distinguish
    atoms at different positions within the unit cell.

    Formula (with Berry phase):
        C_tilde(k) = S(k)^{1/2} C_sel(k)    (orthonormal columns)
        phase_mu = exp(-i b_cart . tau_mu)
        M_mn(k,b) = C_tilde^dag(k) . diag(phase) . C_tilde(k+b)

    Formula (without Berry phase):
        M_mn(k,b) = C_tilde^dag(k) . C_tilde(k+b)

    where b_cart is the full Cartesian b-vector connecting k to k+b
    (including any G_shift for BZ boundary wrapping). This unified
    formula handles both interior (G=0) and boundary (G!=0) neighbors.

    Parameters
    ----------
    k_idx : int
        Index of the current k-point
    next_k_idx : int
        Index of the neighbor k-point
    C_tilde_list : list of ndarray
        Precomputed Lowdin-transformed eigenvectors, shape (N, N_w) each
    b_cart : ndarray of shape (3,), optional
        Cartesian b-vector from k to k+b. Required when berry_phase=True.
    atom_positions : ndarray of shape (num_atoms, 3), optional
        Atomic positions in Cartesian coordinates. Required when berry_phase=True.
    basis_atom_map : ndarray of shape (num_orbitals,), optional
        Maps each orbital to its atom index. Required when berry_phase=True.
    berry_phase : bool, optional
        If True (default), apply Berry phase exp(-i b_cart . tau_mu).
        If False, compute plain inner product M = C_tilde^dag(k) . C_tilde(k+b).

    Returns
    -------
    M_mn : ndarray
        MMN overlap matrix, shape (num_bands, num_bands).
        Singular values are bounded by [0, 1].
    """
    C_k = C_tilde_list[k_idx]
    C_next = C_tilde_list[next_k_idx]

    if berry_phase:
        # Berry phase: exp(-i b_cart . tau_mu) per orbital
        # This encodes the position operator e^{-ib.r} evaluated at each
        # atomic center, carrying the intra-cell position information that
        # Wannier90 needs for computing Wannier centers and spreads.
        #
        # The phase is applied to the RIGHT (neighbor) vector so that:
        #   M_mn = sum_a C_tilde*_am(k) x exp(-i b.tau_a) x C_tilde_an(k+b)
        #        = C_tilde^dag(k) . diag(exp(-i b.tau)) . C_tilde(k+b)
        #
        # This matches the midpoint method's sign convention (exp(-ib.tau))
        # which gives correct Wannier centers via:
        #   <r>_n = -(1/N_k) sum_{k,b} w_b b Im(ln M_nn)
        taus = atom_positions[basis_atom_map]  # (num_orbitals, 3)
        phase = np.exp(-1j * (taus @ b_cart))  # (num_orbitals,)

        # M = C_tilde^dag(k) . diag(phase) . C_tilde(k+b)
        M_mn = C_k.conj().T @ (phase[:, None] * C_next)
    else:
        # Plain inner product without Berry phase
        M_mn = C_k.conj().T @ C_next

    return M_mn


def precompute_lowdin_eigenvectors(
    eigenvectors_list: List[np.ndarray],
    S_k_list: List[np.ndarray],
    band_indices: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    """
    Precompute Löwdin-transformed eigenvectors C̃(k) = S(k)^{1/2} C(k).

    The transformed vectors have orthonormal columns (C̃†C̃ = I),
    enabling Mmn computation with bounded singular values.

    Parameters
    ----------
    eigenvectors_list : list of ndarray
        Full eigenvector matrices C(k), shape (N, N_all)
    S_k_list : list of ndarray
        Full overlap matrices S(k), shape (N, N)
    band_indices : ndarray, optional
        If given, select these bands from C(k) before transforming.

    Returns
    -------
    C_tilde_list : list of ndarray
        Löwdin-transformed eigenvectors, shape (N, num_bands) per k-point
    """
    C_tilde_list = []
    for ik in range(len(eigenvectors_list)):
        C_k = eigenvectors_list[ik]
        S_k = S_k_list[ik]

        # Select bands
        if band_indices is not None:
            C_sel = C_k[:, band_indices]
        else:
            C_sel = C_k

        # Compute S^{1/2} via eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(S_k)
        eigvals = np.maximum(eigvals, 1e-14)  # Clip for stability
        S_half = eigvecs @ (np.sqrt(eigvals)[:, None] * eigvecs.conj().T)

        # Löwdin-transformed eigenvectors
        C_tilde = S_half @ C_sel
        C_tilde_list.append(C_tilde)

    return C_tilde_list


def unitarize_mmn(M: np.ndarray) -> np.ndarray:
    """
    Force an overlap matrix to be unitary via polar decomposition.

    Computes M_unitary = U @ V† where M = U @ Σ @ V† (SVD).
    This is the closest unitary matrix to M in Frobenius norm.

    Essential for LCAO methods with all-electron basis sets where the
    midpoint approximation produces non-unitary overlap matrices due to
    the large condition number of the overlap matrix S(k).

    Parameters
    ----------
    M : ndarray of shape (N, N)
        Input overlap matrix (possibly non-unitary)

    Returns
    -------
    M_unitary : ndarray of shape (N, N)
        Closest unitary matrix to M
    """
    U, _, Vh = np.linalg.svd(M)
    return U @ Vh


def soft_condition_mmn(M: np.ndarray, knee: float = 0.5) -> np.ndarray:
    """
    Soft-knee conditioning of an overlap matrix via SVD.

    Smoothly pushes singular values toward 1 using a tanh function.
    SVs near 1 are minimally perturbed; SVs far from 1 are pulled
    toward 1 with a soft transition at the knee width.

    The transformation is:
        sigma_new = 1 + width * tanh((sigma - 1) / width)
    where width = 1 - knee.

    Parameters
    ----------
    M : ndarray of shape (N, N)
        Input overlap matrix
    knee : float
        Controls the sharpness of the transition. SVs within
        (1 - width) to (1 + width) are mostly preserved.
        knee=0.5 gives width=0.5 (gentle); knee=0.99 gives
        width=0.01 (aggressive, nearly hard unitarization).

    Returns
    -------
    M_conditioned : ndarray of shape (N, N)
        Conditioned matrix with SVs smoothly pushed toward 1
    """
    U, s, Vh = np.linalg.svd(M)
    width = 1.0 - knee
    if width < 1e-12:
        # Degenerate case: knee=1.0 is equivalent to hard unitarization
        return U @ Vh
    s_new = 1.0 + width * np.tanh((s - 1.0) / width)
    return U @ (s_new[:, np.newaxis] * Vh)


def compute_mmn_direct(
    k_idx: int,
    next_k_idx: int,
    kpoints: np.ndarray,
    b_vector_cart: np.ndarray,
    real_space_matrices: Dict[Tuple[int, int, int], Dict[str, np.ndarray]],
    lattice_vectors: np.ndarray,
    atom_positions: np.ndarray,
    basis_atom_map: np.ndarray,
    eigenvectors: List[np.ndarray],
    symmetrize: bool = True,
    G_shift: np.ndarray = None,
    stacked: 'StackedMatrices' = None
) -> np.ndarray:
    """
    Compute M_mn using the Symmetric Midpoint Approximation.

    This implements the MMN matrix for LCAO methods by evaluating the
    overlap matrix S at the midpoint between k and k+b:

        M_mn(k,b) = C†_m(k) · S(k+b/2) · C_n(k+b)

    The b-vector in fractional coordinates is computed directly from the
    k-point indices and G_shift (from the .nnkp file), which is numerically
    exact and avoids Cartesian-to-fractional conversion errors.

    Parameters
    ----------
    k_idx : int
        Index of the current k-point
    next_k_idx : int
        Index of the neighbor k-point
    kpoints : ndarray of shape (num_kpoints, 3)
        K-points in fractional coordinates
    b_vector_cart : ndarray of shape (3,)
        Cartesian b-vector connecting k to k+b (unused, kept for API)
    real_space_matrices : dict
        Dict mapping (n1,n2,n3) -> {'H': H_matrix, 'S': S_matrix}
    lattice_vectors : ndarray of shape (3, 3)
        Real-space lattice vectors (rows are [a1, a2, a3])
    atom_positions : ndarray of shape (num_atoms, 3)
        Atomic positions (unused, kept for API compatibility)
    basis_atom_map : ndarray of shape (num_orbitals,)
        Maps each orbital to its atom index (unused, kept for API)
    eigenvectors : list of ndarray
        Eigenvector matrices C(k), shape (num_orbitals, num_bands)
    symmetrize : bool, optional
        Not used in this implementation.
    G_shift : ndarray of shape (3,), optional
        Integer G-vector shift from .nnkp file

    Returns
    -------
    M_mn : ndarray
        MMN overlap matrix, shape (num_bands, num_bands)
    """
    k_curr = kpoints[k_idx]
    k_next = kpoints[next_k_idx]

    # Use G_shift from nnkp file (passed as parameter)
    # This tells us the BZ wrapping: k' + G = k + b (physical path)
    if G_shift is None:
        G_shift = np.zeros(3, dtype=int)
    G_shift = np.asarray(G_shift)

    # Calculate PHYSICAL b-vector in fractional coordinates
    # b = k_next + G_shift - k_curr (this is the actual displacement)
    b_frac = k_next + G_shift - k_curr

    # Calculate PHYSICAL Midpoint
    # k_mid is the actual location in reciprocal space where overlap occurs
    k_mid = k_curr + 0.5 * b_frac

    # Compute S at the midpoint
    if stacked is not None:
        _, S_mid = fourier_transform_vectorized(k_mid, stacked)
    else:
        _, S_mid = fourier_transform_to_kspace(k_mid, real_space_matrices, lattice_vectors)

    # Project onto eigenvectors
    C_k = eigenvectors[k_idx]
    C_next = eigenvectors[next_k_idx]

    M_mn = C_k.conj().T @ S_mid @ C_next

    return M_mn


def compute_mmn_matrix(
    k_idx: int,
    next_k_idx: int,
    kpoints: np.ndarray,
    b_vector_cart: np.ndarray,
    real_space_matrices: Dict[Tuple[int, int, int], Dict[str, np.ndarray]],
    lattice_vectors: np.ndarray,
    atom_positions: np.ndarray,
    basis_atom_map: np.ndarray,
    eigenvectors: List[np.ndarray]
) -> np.ndarray:
    """
    Compute M_mn matrix using the Symmetric Midpoint Approximation.

    This uses the same formula as compute_mmn_direct() but converts b_frac
    from the Cartesian b-vector rather than k-point indices. Prefer
    compute_mmn_direct() when G_shift is available (e.g., from .nnkp file)
    since it avoids the Cartesian-to-fractional conversion.

    Formula: M_mn(k,b) = C†_m(k) · [Phase ⊙ S(k+b/2)] · C_n(k+b)

    where Phase_μν = exp(-i b_cart · (τ_μ + τ_ν) / 2)

    Parameters
    ----------
    k_idx : int
        Index of the current k-point
    next_k_idx : int
        Index of the neighbor k-point
    kpoints : ndarray of shape (num_kpoints, 3)
        K-points in fractional coordinates
    b_vector_cart : ndarray of shape (3,)
        Cartesian b-vector connecting k to k+b (in Angstrom^-1)
    real_space_matrices : dict
        Dict mapping (n1,n2,n3) -> {'H': H_matrix, 'S': S_matrix}
    lattice_vectors : ndarray of shape (3, 3)
        Real-space lattice vectors (rows are [a1, a2, a3])
    atom_positions : ndarray of shape (num_atoms, 3)
        Atomic positions in Cartesian coordinates (Angstrom)
    basis_atom_map : ndarray of shape (num_orbitals,)
        Maps each orbital to its atom index
    eigenvectors : list of ndarray
        Eigenvector matrices C(k), shape (num_orbitals, num_bands)

    Returns
    -------
    M_mn : ndarray
        MMN overlap matrix, shape (num_bands, num_bands)
    """
    # 1. Get current k-point in fractional coordinates
    k_curr = kpoints[k_idx]

    # 2. Convert Cartesian b-vector to fractional coordinates
    # Convention: k_cart = 2π * A^{-1} @ k_frac, so k_frac = (1/2π) * A @ k_cart
    # where A = lattice_vectors has rows as real-space lattice vectors
    b_frac = np.dot(lattice_vectors, b_vector_cart) / (2 * np.pi)

    # k_mid is exactly half a b-step away from k_curr
    k_mid = k_curr + 0.5 * b_frac

    # 3. Get S(k_mid) using Fourier transform
    _, S_mid = fourier_transform_to_kspace(k_mid, real_space_matrices, lattice_vectors)

    # 4. Symmetric Berry Phase Correction
    # Formula: exp[-i * b_cart · (τ_μ + τ_ν) / 2]
    taus = atom_positions[basis_atom_map]  # (num_orbitals, 3)
    tau_mid = (taus[:, None, :] + taus[None, :, :]) / 2.0  # (N, N, 3)
    phase_exponent = np.sum(tau_mid * b_vector_cart, axis=2)  # (N, N)
    phase_matrix = np.exp(-1j * phase_exponent)

    # 5. Apply phase correction element-wise to get cross-overlap
    S_cross = S_mid * phase_matrix

    # 6. Project onto eigenvectors
    C_k = eigenvectors[k_idx]
    C_next = eigenvectors[next_k_idx]

    M_mn = C_k.conj().T @ S_cross @ C_next

    return M_mn




def write_eig_file(
    filename: str,
    eigenvalues_list: List[np.ndarray],
    num_kpoints: int,
    num_wann: int
) -> None:
    """
    Write the .eig file containing eigenvalues.
    
    Format:
        band_index  k_index  eigenvalue(real)
    
    Parameters
    ----------
    filename : str
        Output filename (e.g., 'material.eig')
    eigenvalues_list : list of ndarrays
        Eigenvalues for each k-point
    num_kpoints : int
        Total number of k-points
    num_wann : int
        Number of Wannier functions (bands)
    
    Notes
    -----
    Wannier90 uses 1-based indexing for band and k-point indices.
    """
    with open(filename, 'w') as f:
        for k_idx in range(num_kpoints):
            eigenvalues = eigenvalues_list[k_idx]
            for band_idx in range(num_wann):
                # Wannier90 uses 1-based indexing
                f.write(
                    f"{band_idx + 1:5d} {k_idx + 1:5d} "
                    f"{eigenvalues[band_idx].real:18.12f}\n"
                )


def write_amn_file_pdwf(
    filename: str,
    eigenvectors_list: List[np.ndarray],
    S_k_list: List[np.ndarray],
    target_mask: np.ndarray,
    band_indices: np.ndarray,
    num_kpoints: int,
    num_wann: int,
) -> None:
    """
    Write the .amn file using PDWF target subspace projection via SVD.

    Projects each Bloch eigenstate onto the FULL target orbital subspace
    (all multi-zeta AOs), then uses SVD to find the optimal num_wann-dimensional
    trial functions. This avoids the bias of selecting specific AO indices.

    The projection is:
        A_full(k) = [S(k) C(k)]_{target, bands}^T  (num_bands × num_target)
        U, Σ, V† = SVD(A_full)
        A_optimal(k) = U[:, :num_wann] @ diag(Σ[:num_wann])  (num_bands × num_wann)

    Parameters
    ----------
    filename : str
        Output .amn filename
    eigenvectors_list : list of ndarray
        Full eigenvector matrices C(k), shape (num_orbitals, num_bands_all)
    S_k_list : list of ndarray
        Full overlap matrices S(k), shape (num_orbitals, num_orbitals)
    target_mask : ndarray of bool
        Boolean mask identifying ALL target orbitals (including multi-zeta)
    band_indices : ndarray
        Indices of selected bands
    num_kpoints : int
        Number of k-points
    num_wann : int
        Number of Wannier functions (must be <= num_target)
    """
    target_indices = np.where(target_mask)[0]
    num_target = len(target_indices)
    num_bands = len(band_indices)

    if num_wann > num_target:
        raise ValueError(f"num_wann ({num_wann}) > num_target ({num_target})")

    with open(filename, 'w') as f:
        f.write("Created by lcao2wannier (PDWF target subspace SVD)\n")
        f.write(f"{num_bands:5d} {num_kpoints:5d} {num_wann:5d}\n")

        for k_idx in range(num_kpoints):
            C_k = eigenvectors_list[k_idx]
            S_k = S_k_list[k_idx]

            # Full projection onto ALL target orbitals
            # P_k = S(k) @ C(k), then select target rows and band columns
            P_k = S_k @ C_k  # (num_orbitals, num_bands_all)
            A_full = P_k[np.ix_(target_indices, band_indices)].T  # (num_bands, num_target)

            # SVD to find optimal num_wann-dimensional projection
            U, sigma, Vh = np.linalg.svd(A_full, full_matrices=False)
            # A_optimal = U[:, :num_wann] @ diag(sigma[:num_wann])
            # This preserves the magnitude structure from the SVD
            A_opt = U[:, :num_wann] * sigma[:num_wann]  # (num_bands, num_wann)

            # Canonical A_mn = <psi_m|g_n>, PROJECTION-OUTER (pw2wannier90
            # order). A_full came from (S C)[target, band].T, i.e. the
            # conjugate of the canonical projection, and the SVD inherits
            # that -- so conjugate on the way out. See _format_amn_block.
            for n in range(num_wann):
                for m in range(num_bands):
                    a = A_opt[m, n].conjugate()
                    f.write(
                        f"{m + 1:5d} {n + 1:5d} {k_idx + 1:5d} "
                        f"{a.real:18.12f} {a.imag:18.12f}\n"
                    )


def write_amn_file_lcao(
    filename: str,
    eigenvectors_list: List[np.ndarray],
    S_k_list: List[np.ndarray],
    orbital_indices: np.ndarray,
    band_indices: np.ndarray,
    num_kpoints: int,
    num_wann: int,
) -> None:
    """
    Write the .amn file for LCAO methods with non-orthogonal basis correction.

    For LCAO methods with non-orthogonal basis, the projection is:
        A_mn(k) = <χ_m | ψ_n(k)> = [S(k) @ C(k)]_mn

    where:
    - χ_m are the selected LCAO trial orbitals (projection functions)
    - ψ_n(k) are the Bloch eigenstates
    - S(k) is the overlap matrix
    - C(k) are the eigenvector coefficients

    This accounts for the non-orthogonality of the LCAO basis functions.

    Parameters
    ----------
    filename : str
        Output .amn filename
    eigenvectors_list : list of ndarray
        Full eigenvector matrices C(k) for each k-point, shape (num_orbitals, num_bands_all)
    S_k_list : list of ndarray
        Full overlap matrices S(k) for each k-point, shape (num_orbitals, num_orbitals)
    orbital_indices : ndarray
        Indices of selected projection orbitals
    band_indices : ndarray
        Indices of selected bands
    num_kpoints : int
        Number of k-points
    num_wann : int
        Number of Wannier functions
    """
    num_bands = len(band_indices)  # Number of bands to use
    num_proj = len(orbital_indices)  # Number of projections

    with open(filename, 'w') as f:
        # Write header
        # Wannier90 format: num_bands num_kpoints num_wann
        f.write("Created by lcao2wannier (Overlap-Corrected)\n")
        f.write(f"{num_bands:5d} {num_kpoints:5d} {num_proj:5d}\n")

        # Write projection matrices
        for k_idx in range(num_kpoints):
            C_k = eigenvectors_list[k_idx]  # Shape: (num_orbitals, num_bands_all)
            S_k = S_k_list[k_idx]  # Shape: (num_orbitals, num_orbitals)

            # Compute projection: P = S @ C
            # This gives <χ_i | ψ_n> for all orbitals i and bands n
            P_k = S_k @ C_k  # Shape: (num_orbitals, num_bands_all)

            # Extract selected orbitals (rows) and selected bands (columns)
            A_k = P_k[np.ix_(orbital_indices, band_indices)]  # Shape: (num_proj, num_bands)

            # Write elements: loop over bands m, then projectors n (m outer).
            # Wannier90 format: band_idx  wannier_idx  kpoint_idx  Re(A)  Im(A).
            # Vectorised single buffered write (byte-identical to the per-element
            # loop); the per-element write was a scaling cost on large systems.
            f.write(_format_amn_block(A_k, num_bands, num_proj, k_idx))


def write_amn_file(
    filename: str,
    eigenvectors_list: List[np.ndarray],
    S_k_list: List[np.ndarray],
    num_kpoints: int,
    num_wann: int
) -> None:
    """
    Write the .amn file containing projection matrices A(k).
    
    A(k) = S(k)† C(k)
    
    Format:
        Header: num_bands  num_kpoints  num_wann
        For each k-point and band:
            band_m  projection_n  k_idx  Re(A_mn)  Im(A_mn)
    
    Parameters
    ----------
    filename : str
        Output filename (e.g., 'material.amn')
    eigenvectors_list : list of ndarrays
        Eigenvectors C(k) for each k-point
    S_k_list : list of ndarrays
        Overlap matrices S(k) for each k-point
    num_kpoints : int
        Total number of k-points
    num_wann : int
        Number of Wannier functions
    
    Notes
    -----
    The projection matrix is computed as A(k) = S(k)† C(k), where
    † denotes the conjugate transpose.
    """
    with open(filename, 'w') as f:
        # Write header
        f.write("Created by lcao2wannier Engine\n")
        f.write(f"{num_wann:5d} {num_kpoints:5d} {num_wann:5d}\n")
        
        # Write projection matrices
        for k_idx in range(num_kpoints):
            C_k = eigenvectors_list[k_idx]
            S_k = S_k_list[k_idx]

            # For LCAO methods: A(k) = projection of eigenvectors onto selected orbitals
            # If S_k is a projection subspace matrix (num_wann x num_orbitals),
            # it contains the selected orbital indices via row selection.
            # The projection is simply the selected rows of C_k.
            if S_k.shape[0] == num_wann and S_k.shape[0] < S_k.shape[1]:
                # S_k indicates which orbitals to use via its row structure
                # For LCAO: A_mn(k) = C_mn(k) where m are selected orbital indices
                # We need to extract which orbitals S_k represents
                # Actually, S_k rows ARE the projections: A(k) = S_k @ C_k normalizes them
                A_k = S_k @ C_k
            else:
                # S_k is the full overlap matrix (num_orbitals x num_orbitals)
                # Use traditional formula: A(k) = S(k)† C(k)
                A_k = S_k.conj().T @ C_k
            
            # Write elements: loop over bands m, then projectors n
            # Canonical A_mn = <psi_m|g_n>, PROJECTION-OUTER. A_k[n, m] is
            # <chi_n|psi_m>, hence the conjugate. See _format_amn_block.
            for n in range(num_wann):
                for m in range(num_wann):
                    a = A_k[n, m].conjugate()
                    f.write(
                        f"{m + 1:5d} {n + 1:5d} {k_idx + 1:5d} "
                        f"{a.real:18.12f} {a.imag:18.12f}\n"
                    )


# ---------------------------------------------------------------------------
# Parallel .mmn writer for the midpoint/direct path. Workers compute M(k,b) and
# format the block into ordered temp files; the main process writes the header
# and concatenates them. Byte-identical to the serial loop (same operations, same
# order). The .mmn write is ~98% CPU string-formatting, which is per-block
# independent and CPU-bound, so it parallelizes well. See OPTIMIZATION_RESULTS.md.
# ---------------------------------------------------------------------------

_MMNW: Dict[str, Any] = {}


def _mmn_choose_n_proc(num_kpoints: int, nntot: int, n_orbitals: int) -> int:
    """Worker count for the parallel .mmn write, or 1 (serial) below the work gate."""
    import multiprocessing as mp
    import os
    if os.environ.get('LCAO_SERIAL_MMN'):   # kill-switch / serial baseline
        return 1
    requested = os.environ.get('LCAO2WANNIER_WORKERS')
    if requested:
        try:
            budget = max(1, int(requested))
        except ValueError:
            budget = 1
        if budget == 1:
            return 1
    else:
        budget = max(1, mp.cpu_count() - 2)
    est = num_kpoints * nntot * 0.06 * (n_orbitals / 1424.0) ** 2   # ~ serial seconds
    if est < 2.0 or num_kpoints <= 1:
        return 1
    return min(6, num_kpoints, budget)


def _mmn_worker_init(meta: dict) -> None:
    from multiprocessing import shared_memory
    nk, N, nw, nR = meta['nk'], meta['N'], meta['nw'], meta['nR']
    _MMNW['shm'] = [shared_memory.SharedMemory(name=meta['C']),
                    shared_memory.SharedMemory(name=meta['S'])]
    _MMNW['C'] = np.ndarray((nk, N, nw), np.dtype(meta['C_dtype']), buffer=_MMNW['shm'][0].buf)
    _MMNW['S'] = np.ndarray((nR, N, N), np.dtype(meta['S_dtype']), buffer=_MMNW['shm'][1].buf)
    _MMNW['R'] = meta['R']
    _MMNW['kpts'] = meta['kpts']
    _MMNW['tasks'] = meta['tasks']


def _mmn_chunk_worker(args) -> str:
    start, end, tmppath = args
    C, S, R, kpts, tasks = (_MMNW['C'], _MMNW['S'], _MMNW['R'],
                            _MMNW['kpts'], _MMNW['tasks'])
    out = []
    for ti in range(start, end):
        k, k_next, g0, g1, g2 = tasks[ti]
        # Identical to compute_mmn_direct: k_mid = k + b/2, S(k_mid)=sum_R e^{ik.R}S(R)
        k_mid = kpts[k] + 0.5 * (kpts[k_next] + np.array([g0, g1, g2]) - kpts[k])
        phases = np.exp(2j * np.pi * (R @ k_mid))
        S_mid = np.tensordot(phases, S, axes=([0], [0]))
        M = C[k].conj().T @ S_mid @ C[k_next]
        out.append(f"{k+1:5d} {k_next+1:5d} {g0:5d} {g1:5d} {g2:5d}\n")
        out.append(_format_complex_block(M, order='F'))
    with open(tmppath, 'w') as f:
        f.write(''.join(out))
    return tmppath


def _write_mmn_parallel(filename, eigenvectors_list, kpoints, stacked, neighbor_list,
                        num_kpoints, num_wann, method_name, n_proc) -> None:
    """Parallel midpoint/direct .mmn write; byte-identical to the serial loop."""
    import multiprocessing as mp
    import os
    import shutil
    import tempfile
    from multiprocessing import shared_memory

    print(f"  Writing .mmn in parallel ({n_proc} workers)...")
    nk = num_kpoints
    N = eigenvectors_list[0].shape[0]
    # Ordered tasks (k-major, neighbor-minor) — exactly the serial loop order.
    tasks = []
    for k in range(nk):
        for nb in neighbor_list[k]:
            g = np.asarray(nb['G_shift'], dtype=int)
            tasks.append((k, int(nb['id']), int(g[0]), int(g[1]), int(g[2])))

    # M[:nw,:nw] only needs the first num_wann columns of C(k); share those + S(R).
    # Preserve native dtypes: S(R) is real (float64) for non-SOC, and the real-vs-
    # complex tensordot must match the serial path exactly for a byte-identical .mmn.
    C = np.empty((nk, N, num_wann), eigenvectors_list[0].dtype)
    for k in range(nk):
        C[k] = eigenvectors_list[k][:, :num_wann]
    S = np.ascontiguousarray(stacked.S_stack)
    nR = S.shape[0]
    shC = shared_memory.SharedMemory(create=True, size=C.nbytes)
    shS = shared_memory.SharedMemory(create=True, size=S.nbytes)
    tmpdir = tempfile.mkdtemp(prefix='mmn_')
    try:
        np.ndarray(C.shape, C.dtype, buffer=shC.buf)[:] = C
        np.ndarray(S.shape, S.dtype, buffer=shS.buf)[:] = S
        meta = dict(nk=nk, N=N, nw=num_wann, nR=nR, C=shC.name, S=shS.name,
                    C_dtype=str(C.dtype), S_dtype=str(S.dtype),
                    R=np.asarray(stacked.R_vectors), kpts=np.asarray(kpoints), tasks=tasks)
        nt = len(tasks)
        bounds = [round(i * nt / n_proc) for i in range(n_proc + 1)]
        chunk_args = [(bounds[i], bounds[i + 1], os.path.join(tmpdir, f'chunk_{i:03d}'))
                      for i in range(n_proc) if bounds[i] < bounds[i + 1]]
        ctx = mp.get_context('spawn')
        with ctx.Pool(len(chunk_args), initializer=_mmn_worker_init,
                      initargs=(meta,)) as pool:
            pool.map(_mmn_chunk_worker, chunk_args)
        # Header + ordered concatenation of the worker chunks.
        with open(filename, 'w') as out:
            out.write(f"Created by lcao2wannier ({method_name} Method)\n")
            out.write(f"{num_wann:5d} {num_kpoints:5d} {len(neighbor_list[0]):5d}\n")
            for _, _, p in chunk_args:
                with open(p) as tf:
                    shutil.copyfileobj(tf, out)
    finally:
        shC.close(); shC.unlink()
        shS.close(); shS.unlink()
        shutil.rmtree(tmpdir, ignore_errors=True)


# --- Parallel Lowdin .mmn write: parallel S^1/2 C precompute, then parallel M ---
_LWDW = {}


def _lowdin_pre_init(meta: dict) -> None:
    from multiprocessing import shared_memory
    nk, N, nsel, nR = meta['nk'], meta['N'], meta['nsel'], meta['nR']
    _LWDW['shm'] = [shared_memory.SharedMemory(name=meta['S']),
                    shared_memory.SharedMemory(name=meta['Csel']),
                    shared_memory.SharedMemory(name=meta['Ct'])]
    _LWDW['S'] = np.ndarray((nR, N, N), np.dtype(meta['S_dtype']), buffer=_LWDW['shm'][0].buf)
    _LWDW['Csel'] = np.ndarray((nk, N, nsel), np.dtype(meta['C_dtype']), buffer=_LWDW['shm'][1].buf)
    _LWDW['Ct'] = np.ndarray((nk, N, nsel), np.complex128, buffer=_LWDW['shm'][2].buf)
    _LWDW['R'] = meta['R']
    _LWDW['kpts'] = meta['kpts']


def _lowdin_pre_worker(args) -> None:
    kstart, kend = args
    S, Csel, Ct, R, kpts = (_LWDW['S'], _LWDW['Csel'], _LWDW['Ct'], _LWDW['R'], _LWDW['kpts'])
    for k in range(kstart, kend):
        # S(k) = sum_R e^{i2pi k.R} S(R) — identical source to the serial path's S_k.
        phases = np.exp(2j * np.pi * (R @ kpts[k]))
        S_k = np.tensordot(phases, S, axes=([0], [0]))
        evals, evecs = np.linalg.eigh(S_k)
        evals = np.maximum(evals, 1e-14)
        S_half = evecs @ (np.sqrt(evals)[:, None] * evecs.conj().T)
        Ct[k] = S_half @ Csel[k]


def _lowdin_write_init(meta: dict) -> None:
    from multiprocessing import shared_memory
    nk, N, nsel = meta['nk'], meta['N'], meta['nsel']
    _LWDW['shm_w'] = shared_memory.SharedMemory(name=meta['Ct'])
    _LWDW['Ctw'] = np.ndarray((nk, N, nsel), np.complex128, buffer=_LWDW['shm_w'].buf)
    _LWDW['tau'] = meta['tau']
    _LWDW['recip'] = meta['recip']
    _LWDW['kpts'] = meta['kpts']
    _LWDW['tasks'] = meta['tasks']
    _LWDW['berry'] = meta['berry']


def _lowdin_write_worker(args) -> str:
    start, end, tmppath = args
    Ct, tau, recip, kpts, tasks, berry = (_LWDW['Ctw'], _LWDW['tau'], _LWDW['recip'],
                                          _LWDW['kpts'], _LWDW['tasks'], _LWDW['berry'])
    out = []
    for ti in range(start, end):
        k, k_next, g0, g1, g2 = tasks[ti]
        if berry:
            b_frac = kpts[k_next] + np.array([g0, g1, g2], dtype=float) - kpts[k]
            b_cart = recip.T @ b_frac
            phase = np.exp(-1j * (tau @ b_cart))
            M = Ct[k].conj().T @ (phase[:, None] * Ct[k_next])
        else:
            M = Ct[k].conj().T @ Ct[k_next]
        out.append(f"{k+1:5d} {k_next+1:5d} {g0:5d} {g1:5d} {g2:5d}\n")
        out.append(_format_complex_block(M, order='F'))
    with open(tmppath, 'w') as f:
        f.write(''.join(out))
    return tmppath


def _write_mmn_parallel_lowdin(filename, eigenvectors_list, stacked, kpoints,
                               neighbor_list, num_kpoints, num_wann, band_indices,
                               atom_positions, basis_atom_map, recip_lattice,
                               use_berry_phase, method_name, n_proc) -> None:
    """Parallel Lowdin .mmn write: phase 1 computes C_tilde=S^1/2 C per k-chunk
    into shared memory; phase 2 assembles M=C_tilde^dag diag(phase) C_tilde and
    formats. Numerically identical to the serial Lowdin loop."""
    import multiprocessing as mp
    import os
    import shutil
    import tempfile
    from multiprocessing import shared_memory

    print(f"  Writing .mmn in parallel ({n_proc} workers, Lowdin precompute+write)...")
    nk = num_kpoints
    N = eigenvectors_list[0].shape[0]
    if band_indices is not None:
        Csel = np.ascontiguousarray(np.stack([C[:, band_indices] for C in eigenvectors_list]))
    else:
        Csel = np.ascontiguousarray(np.stack([C[:, :num_wann] for C in eigenvectors_list]))
    nsel = Csel.shape[2]
    S = np.ascontiguousarray(stacked.S_stack)
    nR = S.shape[0]
    R = np.asarray(stacked.R_vectors)
    kpts = np.asarray(kpoints)
    tau = atom_positions[basis_atom_map]
    tasks = []
    for k in range(nk):
        for nb in neighbor_list[k]:
            g = np.asarray(nb['G_shift'], dtype=int)
            tasks.append((k, int(nb['id']), int(g[0]), int(g[1]), int(g[2])))

    shS = shared_memory.SharedMemory(create=True, size=S.nbytes)
    shCsel = shared_memory.SharedMemory(create=True, size=Csel.nbytes)
    shCt = shared_memory.SharedMemory(
        create=True, size=nk * N * nsel * np.dtype(np.complex128).itemsize)
    tmpdir = tempfile.mkdtemp(prefix='mmn_lwd_')
    try:
        np.ndarray(S.shape, S.dtype, buffer=shS.buf)[:] = S
        np.ndarray(Csel.shape, Csel.dtype, buffer=shCsel.buf)[:] = Csel
        ctx = mp.get_context('spawn')
        pre_meta = dict(nk=nk, N=N, nsel=nsel, nR=nR, S=shS.name, Csel=shCsel.name,
                        Ct=shCt.name, S_dtype=str(S.dtype), C_dtype=str(Csel.dtype),
                        R=R, kpts=kpts)
        kb = [round(i * nk / n_proc) for i in range(n_proc + 1)]
        kchunks = [(kb[i], kb[i + 1]) for i in range(n_proc) if kb[i] < kb[i + 1]]
        with ctx.Pool(len(kchunks), initializer=_lowdin_pre_init,
                      initargs=(pre_meta,)) as pool:
            pool.map(_lowdin_pre_worker, kchunks)
        w_meta = dict(nk=nk, N=N, nsel=nsel, Ct=shCt.name, tau=tau,
                      recip=recip_lattice, kpts=kpts, tasks=tasks, berry=use_berry_phase)
        nt = len(tasks)
        tb = [round(i * nt / n_proc) for i in range(n_proc + 1)]
        chunk_args = [(tb[i], tb[i + 1], os.path.join(tmpdir, f'chunk_{i:03d}'))
                      for i in range(n_proc) if tb[i] < tb[i + 1]]
        with ctx.Pool(len(chunk_args), initializer=_lowdin_write_init,
                      initargs=(w_meta,)) as pool:
            pool.map(_lowdin_write_worker, chunk_args)
        with open(filename, 'w') as out:
            out.write(f"Created by lcao2wannier ({method_name} Method)\n")
            out.write(f"{num_wann:5d} {num_kpoints:5d} {len(neighbor_list[0]):5d}\n")
            for _, _, p in chunk_args:
                with open(p) as tf:
                    shutil.copyfileobj(tf, out)
    finally:
        shS.close(); shS.unlink()
        shCsel.close(); shCsel.unlink()
        shCt.close(); shCt.unlink()
        shutil.rmtree(tmpdir, ignore_errors=True)


# --- Parallel analytic-GTO M assembly: shared S^(b)(g) tensor + C_sel ----------
_ANW = {}


def _analytic_w_init(meta: dict) -> None:
    from multiprocessing import shared_memory
    nk, N, nsel, nb_, ng = meta['nk'], meta['N'], meta['nsel'], meta['num_b'], meta['num_g']
    _ANW['shm'] = [shared_memory.SharedMemory(name=meta['C']),
                   shared_memory.SharedMemory(name=meta['Sb'])]
    _ANW['C'] = np.ndarray((nk, N, nsel), np.dtype(meta['C_dtype']), buffer=_ANW['shm'][0].buf)
    _ANW['Sb'] = np.ndarray((nb_, ng, N, N), np.complex128, buffer=_ANW['shm'][1].buf)
    _ANW['gvecs'] = meta['gvecs']
    _ANW['kpts'] = meta['kpts']
    _ANW['tasks'] = meta['tasks']
    _ANW['bidx'] = meta['bidx']


def _analytic_w_worker(args) -> str:
    start, end, tmppath = args
    C, Sb, gvecs, kpts, tasks, bidx = (_ANW['C'], _ANW['Sb'], _ANW['gvecs'],
                                       _ANW['kpts'], _ANW['tasks'], _ANW['bidx'])
    out = []
    for ti in range(start, end):
        k, k_next, g0, g1, g2 = tasks[ti]
        kpb = kpts[k_next] + np.array([g0, g1, g2], dtype=float)   # (k+b)_frac
        ph = np.exp(2j * np.pi * (gvecs @ kpb))                    # sum_g phase
        Stilde = np.tensordot(ph, Sb[bidx[ti]], axes=([0], [0]))   # Sb_at_kpb
        M = C[k].conj().T @ Stilde @ C[k_next]
        out.append(f"{k+1:5d} {k_next+1:5d} {g0:5d} {g1:5d} {g2:5d}\n")
        out.append(_format_complex_block(M, order='F'))
    with open(tmppath, 'w') as f:
        f.write(''.join(out))
    return tmppath


def _write_mmn_parallel_analytic_write(filename, C_sel, Sb_by_b, bkey_of, cells_R,
                                       kpoints, neighbor_list, num_kpoints, num_wann,
                                       method_name, n_proc) -> None:
    """Parallel analytic-GTO M assembly: share the S^(b)(g) tensor and C_sel,
    workers Fourier-sum to Stilde(k+b) and form M = C_sel^dag Stilde C_sel and
    format into ordered temp files. Numerically identical to the serial loop."""
    import multiprocessing as mp
    import os
    import shutil
    import tempfile
    from multiprocessing import shared_memory

    print(f"  Writing .mmn in parallel ({n_proc} workers, analytic M assembly)...")
    nk = num_kpoints
    N = C_sel[0].shape[0]
    nsel = C_sel[0].shape[1]
    bkeys = list(Sb_by_b)
    key_to_bidx = {key: i for i, key in enumerate(bkeys)}
    gvecs = np.array([g for g, _R in cells_R], dtype=float)
    num_g, num_b = len(gvecs), len(bkeys)
    Sb_flat = np.empty((num_b, num_g, N, N), dtype=np.complex128)
    for bi, key in enumerate(bkeys):
        cells = Sb_by_b[key]
        for gi, (g, _R) in enumerate(cells_R):
            Sb_flat[bi, gi] = cells[g]
    C = np.ascontiguousarray(np.stack(C_sel))
    kpts = np.asarray(kpoints)
    tasks, bidx = [], []
    for k in range(nk):
        for ni, nb in enumerate(neighbor_list[k]):
            g = np.asarray(nb['G_shift'], dtype=int)
            tasks.append((k, int(nb['id']), int(g[0]), int(g[1]), int(g[2])))
            bidx.append(key_to_bidx[bkey_of[(k, ni)]])
    bidx = np.array(bidx, dtype=int)

    shC = shared_memory.SharedMemory(create=True, size=C.nbytes)
    shSb = shared_memory.SharedMemory(create=True, size=Sb_flat.nbytes)
    tmpdir = tempfile.mkdtemp(prefix='mmn_an_')
    try:
        np.ndarray(C.shape, C.dtype, buffer=shC.buf)[:] = C
        np.ndarray(Sb_flat.shape, Sb_flat.dtype, buffer=shSb.buf)[:] = Sb_flat
        meta = dict(nk=nk, N=N, nsel=nsel, num_b=num_b, num_g=num_g,
                    C=shC.name, Sb=shSb.name, C_dtype=str(C.dtype),
                    gvecs=gvecs, kpts=kpts, tasks=tasks, bidx=bidx)
        nt = len(tasks)
        tb = [round(i * nt / n_proc) for i in range(n_proc + 1)]
        chunk_args = [(tb[i], tb[i + 1], os.path.join(tmpdir, f'chunk_{i:03d}'))
                      for i in range(n_proc) if tb[i] < tb[i + 1]]
        ctx = mp.get_context('spawn')
        with ctx.Pool(len(chunk_args), initializer=_analytic_w_init,
                      initargs=(meta,)) as pool:
            pool.map(_analytic_w_worker, chunk_args)
        with open(filename, 'w') as out:
            out.write(f"Created by lcao2wannier ({method_name} Method)\n")
            out.write(f"{num_wann:5d} {num_kpoints:5d} {len(neighbor_list[0]):5d}\n")
            for _, _, p in chunk_args:
                with open(p) as tf:
                    shutil.copyfileobj(tf, out)
    finally:
        shC.close(); shC.unlink()
        shSb.close(); shSb.unlink()
        shutil.rmtree(tmpdir, ignore_errors=True)


def write_mmn_file_lcao(
    filename: str,
    eigenvectors_list: List[np.ndarray],
    kpoints: np.ndarray,
    real_space_matrices: Dict[Tuple[int, int, int], Dict[str, np.ndarray]],
    lattice_vectors: np.ndarray,
    neighbor_list: Dict[int, List[Dict[str, Any]]],
    atom_positions: np.ndarray,
    basis_atom_map: np.ndarray,
    num_kpoints: int,
    num_wann: int,
    convention: str = 'pi',
    use_direct_method: bool = True,
    verbose: bool = False,
    stacked: 'StackedMatrices' = None,
    unitarize: bool = False,
    method: str = 'lowdin',
    S_k_list: Optional[List[np.ndarray]] = None,
    band_indices: Optional[np.ndarray] = None,
    recip_lattice: Optional[np.ndarray] = None,
    conditioning: Optional[str] = None,
    conditioning_knee: float = 0.5,
    gto_aos: Optional[List[dict]] = None,
    gto_cutoff: float = 16.0,
) -> None:
    """
    Write the .mmn file for LCAO methods.

    This implements the MMN matrix calculation for LCAO methods with three
    available algorithms:

    1. Löwdin method (method='lowdin', default, RECOMMENDED):
       Transforms eigenvectors to Löwdin-orthogonalized basis where S=I,
       then computes M as inner products. SVs bounded by [0,1].
       Formula: M = [S^{1/2}(k)C(k)]† · D_G · [S^{1/2}(k+b)C(k+b)]

    2. Direct midpoint method (method='midpoint', use_direct_method=True):
       Uses symmetric midpoint approximation with Berry phase correction.
       Formula: M = C†(k) · [Phase ⊙ S(k+b/2)] · C(k+b)

    3. Symmetric midpoint method (method='midpoint', use_direct_method=False):
       Same formula via Cartesian b-vector conversion.

    Parameters
    ----------
    filename : str
        Output .mmn filename
    eigenvectors_list : list of ndarray
        Eigenvector matrices C(k), shape (num_orbitals, num_bands)
    kpoints : ndarray
        K-points in fractional coordinates, shape (num_kpoints, 3)
    real_space_matrices : dict
        Dict mapping (n1,n2,n3) -> {'H': H_matrix, 'S': S_matrix}
    lattice_vectors : ndarray
        Real-space lattice vectors, shape (3, 3)
    neighbor_list : dict
        Maps k_idx -> list of neighbor dicts with keys:
            'id': neighbor k-point index
            'G_shift': integer lattice vector (n1, n2, n3)
            'b_vec_cart': Cartesian b-vector (used by midpoint method only)
    atom_positions : ndarray
        Atomic positions in Cartesian coordinates, shape (num_atoms, 3)
    basis_atom_map : ndarray
        Maps each orbital to its atom index, shape (num_orbitals,)
    num_kpoints : int
        Number of k-points
    num_wann : int
        Number of Wannier functions (selected bands)
    convention : str, optional
        'pi' for π convention (Crystal23), '2pi' for 2π convention
    use_direct_method : bool, optional
        If True, use the direct method for midpoint approach (default).
    verbose : bool, optional
        If True, print diagnostic information (default: False)
    unitarize : bool, optional
        If True, force each M matrix to be unitary via SVD polar
        decomposition. Default: False.
    method : str, optional
        'lowdin' (default): Lowdin orthogonalization method with Berry phase.
            Requires S_k_list parameter.
        'lowdin_no_berry': Lowdin method WITHOUT Berry phase (for comparison).
            Requires S_k_list parameter. Produces plain inner products.
        'midpoint': Symmetric midpoint approximation with Berry phase.
        'analytic': Exact analytic Gaussian-overlap MMN. Requires gto_aos
            (parsed CRYSTAL GTO basis). No approximation — M is unitary on any
            k-mesh. See lcao2wannier.gto_mmn.
    gto_aos : list of dict, optional
        Parsed GTO basis (from gto_mmn.parse_gto_basis). Required for
        method='analytic'.
    gto_cutoff : float, optional
        AO-pair centre-distance cutoff in Bohr for the analytic method
        (default: 16.0). Pairs farther apart are skipped (negligible overlap).
    S_k_list : list of ndarray, optional
        Full overlap matrices S(k) for each k-point (required for Lowdin).
    band_indices : ndarray, optional
        Band indices for Lowdin method (if eigenvectors_list contains
        full eigenvectors rather than band-selected ones).
    recip_lattice : ndarray, optional
        Reciprocal lattice vectors (rows), shape (3, 3).
        Required for Lowdin with Berry phase.
    conditioning : str or None, optional
        Post-processing of M matrices. Overrides `unitarize` if set.
        None: use `unitarize` parameter for backward compatibility.
        'none': no conditioning.
        'svd': SVD polar decomposition (hard unitarization, M = U V†).
        'soft': soft-knee conditioning via tanh (smooth push toward 1).
    conditioning_knee : float, optional
        Knee parameter for soft conditioning (default: 0.5).
        Controls sharpness: 0.5 = gentle, 0.99 = aggressive.
    """
    # Resolve conditioning mode
    if conditioning is not None:
        do_svd = (conditioning == 'svd')
        do_soft = (conditioning == 'soft')
    else:
        do_svd = unitarize
        do_soft = False

    use_lowdin = (method in ('lowdin', 'lowdin_no_berry') and S_k_list is not None)
    use_berry_phase = (method == 'lowdin')  # no berry for 'lowdin_no_berry'
    use_analytic = (method == 'analytic')

    # Parallel Lowdin fast path: parallel S^1/2 C precompute + parallel M write.
    # Numerically identical to the serial Lowdin loop; serial below if small/forced.
    if (use_lowdin and stacked is not None and not do_svd and not do_soft
            and (not use_berry_phase or recip_lattice is not None)):
        _nproc_l = _mmn_choose_n_proc(num_kpoints, len(neighbor_list[0]),
                                      eigenvectors_list[0].shape[0])
        if _nproc_l > 1:
            _write_mmn_parallel_lowdin(
                filename, eigenvectors_list, stacked, kpoints, neighbor_list,
                num_kpoints, num_wann, band_indices, atom_positions, basis_atom_map,
                recip_lattice, use_berry_phase,
                "Lowdin" + ("" if use_berry_phase else " (no Berry phase)"), _nproc_l)
            return

    if use_analytic:
        if gto_aos is None:
            raise ValueError(
                "method='analytic' requires gto_aos (parsed GTO basis). "
                "Thread it through via engine.gto_aos / write_files."
            )
        from . import gto_mmn as _gto
        method_name = "Analytic GTO"
        n_ao = len(gto_aos)
        n_orb = eigenvectors_list[0].shape[0]
        _spinor = (n_orb == 2 * n_ao)     # 2-component SOC: eigenvectors stack alpha|beta
        _N = n_ao
        if n_orb != n_ao and not _spinor:
            raise ValueError(
                f"Analytic MMN: GTO basis has {n_ao} AOs but eigenvectors have "
                f"{n_orb} orbitals (expected {n_ao}, or {2 * n_ao} for 2-component SOC)."
            )

        def _gto_M(Ca, S, Cb):
            """M = Ca^dag . S . Cb. For a 2-component SOC basis the eigenvector rows
            stack the alpha block (0:N) above the beta block (N:2N), and the GTO
            overlap S (N x N, spatial) is spin-diagonal (S_spinor = I2 (x) S), so the
            spinor overlap is just the sum over the two spin blocks."""
            if _spinor:
                return (Ca[:_N].conj().T @ S @ Cb[:_N]
                        + Ca[_N:].conj().T @ S @ Cb[_N:])
            return Ca.conj().T @ S @ Cb
        # Lattice in Bohr. The CRYSTAL parser always returns lattice_vectors in
        # Angstrom, and the Gaussian exponents are in atomic units, so convert
        # unconditionally (the AO centres from parse_gto_basis are already Bohr).
        lattice_bohr = np.asarray(lattice_vectors, dtype=float) * _gto.ANG2BOHR
        recip_bohr = _gto.reciprocal_bohr(lattice_bohr)
        cells_R = [(g, np.asarray(g, dtype=float) @ lattice_bohr)
                   for g in real_space_matrices]
        # Precompute S^{(b)}(g) ONCE per UNIQUE b-vector across the whole mesh.
        # The unique b-vectors are independent, so build_Sb_cells (the n_ao-pair
        # analytic integral over cells — the dominant cost) is distributed across
        # worker processes. Identical results to the serial dict build.
        bkey_of, unique_b, b0_key = _gto.build_b_table(
            kpoints,
            ((k_idx, ni, neighbor['id'], neighbor['G_shift'])
             for k_idx in range(num_kpoints)
             for ni, neighbor in enumerate(neighbor_list[k_idx])),
            recip_bohr)
        # Precompute + orthonormalize via the SHARED analytic-MMN helpers
        # (gto_mmn.precompute_Sb_parallel / orthonormalize_gto_metric) — one
        # implementation for this path and hybrid_pipeline, no drift.
        Sb_by_b = _gto.precompute_Sb_parallel(
            gto_aos, unique_b, cells_R, cutoff=gto_cutoff,
            n_proc=_mmn_choose_n_proc(num_kpoints, len(neighbor_list[0]),
                                      n_orb),
            verbose=verbose)
        # Band-selected eigenvectors (full AO rows, selected band columns).
        C_sel = [C[:, band_indices] if band_indices is not None else C
                 for C in eigenvectors_list]
        C_sel = _gto.orthonormalize_gto_metric(C_sel, Sb_by_b[b0_key],
                                               kpoints, len(gto_aos))
        if verbose:
            print("  Analytic GTO: orthonormalized C against the b=0 GTO metric"
                  " (M(k,0)=I, SVs <= 1)")
    elif use_lowdin:
        method_name = "Lowdin" + ("" if use_berry_phase else " (no Berry phase)")
        # Precompute Lowdin-transformed eigenvectors
        if verbose:
            print(f"  Precomputing Lowdin-transformed eigenvectors...")
        C_tilde_list = precompute_lowdin_eigenvectors(
            eigenvectors_list, S_k_list, band_indices
        )
    else:
        if method == 'lowdin' and S_k_list is None:
            if verbose:
                print(f"  WARNING: S_k_list not provided, falling back to midpoint method")
        method_name = "Direct Real-Space" if use_direct_method else "Symmetric Midpoint"
        # Apply band selection for midpoint path (Lowdin does this in precompute_lowdin_eigenvectors)
        if band_indices is not None:
            eigenvectors_list = [C[:, band_indices] for C in eigenvectors_list]

    # Parallel fast path — byte-identical to the serial loop below for the common
    # midpoint/direct case (no Lowdin, no analytic GTO, no conditioning). The verbose
    # self-overlap diagnostics it skips are empty for a standard neighbor list (k->k,
    # G=0 never occurs), so correctness of the .mmn is unaffected. Analytic GTO must
    # NOT take this path: _write_mmn_parallel computes the midpoint overlap from the
    # stacked S(R), not the GTO overlap, so it would silently emit a midpoint .mmn.
    if (not use_lowdin and not use_analytic and use_direct_method and not do_svd
            and not do_soft and stacked is not None):
        _nproc = _mmn_choose_n_proc(num_kpoints, len(neighbor_list[0]),
                                    eigenvectors_list[0].shape[0])
        if _nproc > 1:
            _write_mmn_parallel(filename, eigenvectors_list, kpoints, stacked,
                                neighbor_list, num_kpoints, num_wann, method_name, _nproc)
            return

    # Parallel analytic-GTO M assembly: workers Fourier-sum the shared S^(b)(g)
    # to Stilde and form M = C_sel^dag Stilde C_sel. Memory-gated on the shared
    # S^(b) tensor; numerically identical to the serial analytic loop below.
    if use_analytic:
        _Na = C_sel[0].shape[0]
        _sb_bytes = len(Sb_by_b) * len(cells_R) * _Na * _Na * 16
        _na = _mmn_choose_n_proc(num_kpoints, len(neighbor_list[0]), _Na)
        if _na > 1 and _sb_bytes < 2.0e9 and not _spinor:
            _write_mmn_parallel_analytic_write(
                filename, C_sel, Sb_by_b, bkey_of, cells_R, kpoints,
                neighbor_list, num_kpoints, num_wann, method_name, _na)
            return

    with open(filename, 'w') as f:
        # Write header
        f.write(f"Created by lcao2wannier ({method_name} Method)\n")
        num_neighbors = len(neighbor_list[0])
        f.write(f"{num_wann:5d} {num_kpoints:5d} {num_neighbors:5d}\n")

        # Diagnostic tracking
        if verbose:
            diag_traces = []
            unitarity_errors = []

        # Process each k-point
        for k_idx in range(num_kpoints):
            for ni, neighbor in enumerate(neighbor_list[k_idx]):
                k_next_idx = neighbor['id']
                G_shift = neighbor['G_shift']

                if use_analytic:
                    # Exact analytic overlap:
                    #   M = C(k)^dag . Stilde^{(b)}(k+b) . C(k+b)
                    # (k+b)_frac = kpoints[k_next] + G_shift; integer G_shift
                    # contributes only trivial phases (e^{i2pi G.g}=1).
                    kpb_frac = kpoints[k_next_idx] + np.asarray(G_shift, dtype=float)
                    Stilde = _gto.Sb_at_kpb(Sb_by_b[bkey_of[(k_idx, ni)]], kpb_frac)
                    M_kb = _gto_M(C_sel[k_idx], Stilde, C_sel[k_next_idx])
                elif use_lowdin:
                    # Lowdin method:
                    # With Berry: M = C_tilde^dag(k) . diag(exp(-i b.tau)) . C_tilde(k+b)
                    # Without:    M = C_tilde^dag(k) . C_tilde(k+b)
                    if use_berry_phase:
                        G_shift_float = np.asarray(G_shift, dtype=float)
                        b_frac = kpoints[k_next_idx] + G_shift_float - kpoints[k_idx]
                        b_cart = recip_lattice.T @ b_frac
                        M_kb = compute_mmn_lowdin(
                            k_idx, k_next_idx, C_tilde_list,
                            b_cart=b_cart,
                            atom_positions=atom_positions,
                            basis_atom_map=basis_atom_map,
                            berry_phase=True,
                        )
                    else:
                        M_kb = compute_mmn_lowdin(
                            k_idx, k_next_idx, C_tilde_list,
                            berry_phase=False,
                        )
                else:
                    # Midpoint method
                    b_vec_cart = neighbor.get('b_vec_cart', np.zeros(3))

                    if use_direct_method:
                        M_kb = compute_mmn_direct(
                            k_idx, k_next_idx, kpoints, b_vec_cart,
                            real_space_matrices, lattice_vectors,
                            atom_positions, basis_atom_map, eigenvectors_list,
                            symmetrize=True,
                            G_shift=G_shift,
                            stacked=stacked
                        )
                    else:
                        M_kb = compute_mmn_matrix(
                            k_idx, k_next_idx, kpoints, b_vec_cart,
                            real_space_matrices, lattice_vectors,
                            atom_positions, basis_atom_map, eigenvectors_list
                        )

                # Apply conditioning if requested
                if do_svd or do_soft:
                    if verbose:
                        MhM = M_kb.conj().T @ M_kb
                        unitarity_errors.append(
                            np.linalg.norm(MhM - np.eye(M_kb.shape[0])) / M_kb.shape[0]
                        )
                    if do_svd:
                        M_kb = unitarize_mmn(M_kb)
                    elif do_soft:
                        M_kb = soft_condition_mmn(M_kb, knee=conditioning_knee)

                # Diagnostic: check self-overlap
                if verbose and k_idx == k_next_idx and np.all(np.asarray(G_shift) == 0):
                    trace = np.trace(M_kb)
                    diag_traces.append((k_idx, trace))

                # --- WRITE TO FILE ---
                f.write(f"{k_idx+1:5d} {k_next_idx+1:5d} "
                       f"{G_shift[0]:5d} {G_shift[1]:5d} {G_shift[2]:5d}\n")

                f.write(_format_complex_block(M_kb[:num_wann, :num_wann], order='F'))

        if verbose:
            print(f"\n  MMN Diagnostics ({method_name} method):")
            if unitarity_errors:
                mean_err = np.mean(unitarity_errors)
                max_err = np.max(unitarity_errors)
                cond_label = "svd" if do_svd else "soft"
                print(f"  Pre-conditioning ({cond_label}): ||M†M-I||/N mean={mean_err:.6f}, max={max_err:.6f}")
            if diag_traces:
                print(f"  Self-overlap traces (should be ≈ {num_wann}):")
                for k_idx_d, trace in diag_traces[:5]:
                    print(f"    k={k_idx_d}: Tr(M) = {trace.real:.6f} + {trace.imag:.6f}i")


def write_mmn_file(
    filename: str,
    eigenvectors_list: List[np.ndarray],
    S_k_list: List[np.ndarray],
    neighbor_list: Dict[int, List[Tuple[int, np.ndarray]]],
    num_kpoints: int,
    num_wann: int
) -> None:
    """
    Write the .mmn file containing overlap matrices M(k,b).

    M(k,b) = C†(k) S(k+b) C(k+b)

    Format:
        Header: num_bands  num_kpoints  num_neighbors
        For each k-point and neighbor:
            k_idx  neighbor_idx  b1  b2  b3
            M_mn matrix elements (num_wann x num_wann)
    
    Parameters
    ----------
    filename : str
        Output filename (e.g., 'material.mmn')
    eigenvectors_list : list of ndarrays
        Eigenvectors C(k) for each k-point
    S_k_list : list of ndarrays
        Overlap matrices S(k) for each k-point
    neighbor_list : dict
        Maps k_idx -> list of (neighbor_idx, b_vector) tuples
    num_kpoints : int
        Total number of k-points
    num_wann : int
        Number of Wannier functions
    
    Notes
    -----
    The overlap matrix is computed as M(k,b) = C†(k) S(k+b) C(k+b),
    where b is the lattice vector connecting k to its neighbor.
    """
    # Each k-point typically has 6 neighbors (±x, ±y, ±z)
    num_neighbors = len(neighbor_list[0])
    
    with open(filename, 'w') as f:
        # Write header
        f.write("Created by lcao2wannier Engine\n")
        f.write(f"{num_wann:5d} {num_kpoints:5d} {num_neighbors:5d}\n")
        
        # Loop over all k-points
        for k_idx in range(num_kpoints):
            C_k = eigenvectors_list[k_idx]
            
            # Get neighbors of this k-point
            neighbors = neighbor_list[k_idx]
            
            for neighbor_idx, b in neighbors:
                C_k_plus_b = eigenvectors_list[neighbor_idx]
                S_k_plus_b = S_k_list[neighbor_idx]
                
                # Compute M(k,b) = C†(k) S(k+b) C(k+b)
                M_kb = C_k.conj().T @ S_k_plus_b @ C_k_plus_b
                
                # Write k-point indices and lattice vector b
                # Wannier90 uses 1-based indexing
                f.write(
                    f"{k_idx + 1:5d} {neighbor_idx + 1:5d} "
                    f"{b[0]:5d} {b[1]:5d} {b[2]:5d}\n"
                )
                
                # Write M_mn matrix elements
                for m in range(num_wann):
                    for n in range(num_wann):
                        f.write(
                            f"{M_kb[m, n].real:18.12f} "
                            f"{M_kb[m, n].imag:18.12f}\n"
                        )


def write_wannier90_files(
    seedname: str,
    eigenvalues_list: List[np.ndarray],
    eigenvectors_list: List[np.ndarray],
    S_k_list: List[np.ndarray],
    neighbor_list: Dict[int, List[Tuple[int, np.ndarray]]],
    num_kpoints: int,
    num_wann: int,
    verbose: bool = True
) -> None:
    """
    Write all three Wannier90 input files (.eig, .amn, .mmn).
    
    Parameters
    ----------
    seedname : str
        Prefix for output files
    eigenvalues_list : list of ndarrays
        Eigenvalues for each k-point
    eigenvectors_list : list of ndarrays
        Eigenvectors for each k-point
    S_k_list : list of ndarrays
        Overlap matrices for each k-point
    neighbor_list : dict
        Neighbor connectivity information
    num_kpoints : int
        Total number of k-points
    num_wann : int
        Number of Wannier functions
    verbose : bool
        Whether to print progress messages
    """
    if verbose:
        print(f"\nWriting Wannier90 files with seedname '{seedname}'...")
    
    # Write .eig file
    eig_file = f"{seedname}.eig"
    write_eig_file(eig_file, eigenvalues_list, num_kpoints, num_wann)
    if verbose:
        num_entries = num_kpoints * num_wann
        print(f"  ✓ {eig_file}: {num_entries} eigenvalues")
    
    # Write .amn file
    amn_file = f"{seedname}.amn"
    write_amn_file(amn_file, eigenvectors_list, S_k_list, num_kpoints, num_wann)
    if verbose:
        num_entries = num_kpoints * num_wann * num_wann
        print(f"  ✓ {amn_file}: {num_entries} matrix elements")
    
    # Write .mmn file
    mmn_file = f"{seedname}.mmn"
    write_mmn_file(
        mmn_file, eigenvectors_list, S_k_list, neighbor_list, num_kpoints, num_wann
    )
    if verbose:
        num_neighbors = len(neighbor_list[0])
        num_entries = num_kpoints * num_neighbors * num_wann * num_wann
        print(f"  ✓ {mmn_file}: {num_entries} matrix elements")
    
    if verbose:
        print(f"\nWannier90 files generated successfully!")
        print(f"  Files: {seedname}.eig, {seedname}.amn, {seedname}.mmn")

