"""
Fourier Transform Module

This module contains functions for Fourier transforming real-space matrices
to k-space using the phase factor e^(i 2π k·R).

NOTE: This uses the standard 2π convention which ensures proper BZ periodicity:
      H(k+G) = H(k) and S(k+G) = S(k) for any reciprocal lattice vector G.
"""

import numpy as np
from typing import Dict, Tuple, Optional


class StackedMatrices:
    """Pre-stacked real-space matrices for vectorized Fourier assembly.

    Converts the dict-based R-space storage into contiguous numpy arrays,
    eliminating per-k-point dict iteration, tuple-to-array conversion, and
    scalar-matrix multiply overhead.

    Attributes
    ----------
    R_vectors : ndarray of shape (num_R, 3)
        R-vectors in integer lattice coordinates
    H_stack : ndarray of shape (num_R, N, N)
        Stacked Hamiltonian matrices H(R)
    S_stack : ndarray of shape (num_R, N, N)
        Stacked overlap matrices S(R)
    num_orbitals : int
        Number of orbitals (N)
    num_R : int
        Number of R-vectors
    """
    __slots__ = ('R_vectors', 'H_stack', 'S_stack', 'num_orbitals', 'num_R')

    def __init__(self, R_vectors, H_stack, S_stack, num_orbitals, num_R):
        self.R_vectors = R_vectors
        self.H_stack = H_stack
        self.S_stack = S_stack
        self.num_orbitals = num_orbitals
        self.num_R = num_R


def stack_real_space_matrices(
    real_space_matrices: Dict[Tuple[int, int, int], Dict[str, np.ndarray]]
) -> StackedMatrices:
    """Convert dict-based R-space matrices to contiguous stacked arrays.

    Parameters
    ----------
    real_space_matrices : dict
        Maps (R1, R2, R3) -> {'H': H_matrix, 'S': S_matrix}

    Returns
    -------
    stacked : StackedMatrices
        Pre-stacked arrays ready for vectorized Fourier assembly
    """
    R_keys = list(real_space_matrices.keys())
    R_vectors = np.array(R_keys, dtype=np.float64)
    H_stack = np.array([real_space_matrices[k]['H'] for k in R_keys])
    S_stack = np.array([real_space_matrices[k]['S'] for k in R_keys])
    N = H_stack.shape[1]
    return StackedMatrices(R_vectors, H_stack, S_stack, N, len(R_keys))


def fourier_transform_vectorized(
    k_point: np.ndarray,
    stacked: StackedMatrices
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized Fourier assembly for a single k-point.

    Replaces the Python for-loop over R-vectors with np.tensordot,
    dispatching to BLAS for the weighted sum.

    Parameters
    ----------
    k_point : ndarray of shape (3,)
        k-point in fractional coordinates
    stacked : StackedMatrices
        Pre-stacked R-space matrices

    Returns
    -------
    H_k, S_k : ndarray of shape (N, N)
        Hamiltonian and overlap in k-space
    """
    phases = np.exp(2j * np.pi * stacked.R_vectors @ k_point)
    H_k = np.tensordot(phases, stacked.H_stack, axes=([0], [0]))
    S_k = np.tensordot(phases, stacked.S_stack, axes=([0], [0]))
    return H_k, S_k


def fourier_all_kpoints(
    k_points: np.ndarray,
    stacked: StackedMatrices,
    S_only: bool = False
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """Batch Fourier assembly for ALL k-points in one call.

    Computes phases for all k-points at once via matrix multiply,
    then uses np.einsum to sum over R-vectors — a single BLAS call.

    Parameters
    ----------
    k_points : ndarray of shape (K, 3)
        All k-points in fractional coordinates
    stacked : StackedMatrices
        Pre-stacked R-space matrices
    S_only : bool
        If True, only compute S(k) (for MMN midpoint evaluation)

    Returns
    -------
    H_all : ndarray of shape (K, N, N) or None if S_only
        Hamiltonian at all k-points
    S_all : ndarray of shape (K, N, N)
        Overlap matrix at all k-points
    """
    # phases: (K, R) — one matrix multiply
    phases = np.exp(2j * np.pi * k_points @ stacked.R_vectors.T)
    S_all = np.einsum('kr,rij->kij', phases, stacked.S_stack, optimize=True)
    if S_only:
        return None, S_all
    H_all = np.einsum('kr,rij->kij', phases, stacked.H_stack, optimize=True)
    return H_all, S_all


def fourier_transform_to_kspace(
    k_point: np.ndarray,
    real_space_matrices: Dict[Tuple[int, int, int], Dict[str, np.ndarray]],
    lattice_vectors: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fourier transform real-space H(R) and S(R) matrices to k-space.

    Computes:
        H(k) = Σ_R e^(i 2π k·R) H(R)
        S(k) = Σ_R e^(i 2π k·R) S(R)

    Uses the standard 2π convention which ensures BZ periodicity:
    H(k+G) = H(k) for any reciprocal lattice vector G (integer).

    Parameters
    ----------
    k_point : ndarray of shape (3,)
        k-point in fractional coordinates
    real_space_matrices : dict
        Maps (R1, R2, R3) -> {'H': H_matrix, 'S': S_matrix}
        where H_matrix and S_matrix are complex ndarrays
    lattice_vectors : ndarray of shape (3, 3)
        Real-space lattice vectors (rows are vectors)

    Returns
    -------
    H_k : ndarray
        Hamiltonian in k-space, shape (num_orbitals, num_orbitals)
    S_k : ndarray
        Overlap matrix in k-space, shape (num_orbitals, num_orbitals)
    """
    # Get matrix dimensions from first entry
    first_key = next(iter(real_space_matrices))
    num_orbitals = real_space_matrices[first_key]['H'].shape[0]

    # Initialize k-space matrices
    H_k = np.zeros((num_orbitals, num_orbitals), dtype=np.complex128)
    S_k = np.zeros((num_orbitals, num_orbitals), dtype=np.complex128)

    # Loop over all R vectors and accumulate Fourier sum
    for R_tuple, matrices in real_space_matrices.items():
        R = np.array(R_tuple)

        # Calculate phase factor: e^(i 2π k·R)
        # Using 2π convention for proper BZ periodicity
        phase = np.exp(2j * np.pi * np.dot(k_point, R))

        # Accumulate the Fourier sum
        H_k += phase * matrices['H']
        S_k += phase * matrices['S']

    return H_k, S_k


def inverse_fourier_transform(
    k_points: np.ndarray,
    H_k_list: list,
    S_k_list: list,
    R_vectors: list,
    k_grid: Tuple[int, int, int]
) -> Dict[Tuple[int, int, int], Dict[str, np.ndarray]]:
    """
    Inverse Fourier transform from k-space to real-space (optional utility).

    Computes:
        H(R) = (1/N_k) Σ_k e^(-i 2π k·R) H(k)
        S(R) = (1/N_k) Σ_k e^(-i 2π k·R) S(k)

    Uses the standard 2π convention for proper BZ periodicity.

    Parameters
    ----------
    k_points : ndarray of shape (num_kpoints, 3)
        k-points in fractional coordinates
    H_k_list : list of ndarrays
        List of Hamiltonian matrices at each k-point
    S_k_list : list of ndarrays
        List of overlap matrices at each k-point
    R_vectors : list of tuples
        List of R vectors to compute
    k_grid : tuple of 3 ints
        Dimensions of the k-point grid

    Returns
    -------
    real_space_matrices : dict
        Maps (R1, R2, R3) -> {'H': H_matrix, 'S': S_matrix}
    """
    num_kpoints = len(k_points)
    num_orbitals = H_k_list[0].shape[0]
    real_space_matrices = {}

    for R_tuple in R_vectors:
        R = np.array(R_tuple)
        H_R = np.zeros((num_orbitals, num_orbitals), dtype=np.complex128)
        S_R = np.zeros((num_orbitals, num_orbitals), dtype=np.complex128)

        for k_idx, k_point in enumerate(k_points):
            # Phase factor: e^(-i 2π k·R) - using 2π convention
            phase = np.exp(-2j * np.pi * np.dot(k_point, R))

            H_R += phase * H_k_list[k_idx]
            S_R += phase * S_k_list[k_idx]

        # Normalize by number of k-points
        H_R /= num_kpoints
        S_R /= num_kpoints

        real_space_matrices[R_tuple] = {'H': H_R, 'S': S_R}

    return real_space_matrices


def compute_phase_factors(
    k_points: np.ndarray,
    R_vectors: list
) -> np.ndarray:
    """
    Pre-compute phase factors for all k-points and R vectors.

    Parameters
    ----------
    k_points : ndarray of shape (num_kpoints, 3)
        k-points in fractional coordinates
    R_vectors : list of tuples
        List of (R1, R2, R3) tuples

    Returns
    -------
    phase_factors : ndarray of shape (num_kpoints, num_R_vectors)
        Pre-computed phase factors using 2π convention
    """
    R_arr = np.array(R_vectors, dtype=np.float64)
    return np.exp(2j * np.pi * k_points @ R_arr.T)


class LazySkList:
    """Sequence view that reconstructs S(k) on demand from the stacked
    S(R) — a drop-in for the materialized S_k_list under ``--memory low``.

    Retaining every S(k) costs nk * N^2 * 16 bytes (Sc monolayer at
    6x6x1: 1.17 GB; a 12x12x1 grid would hold 4.7 GB); each on-demand
    reconstruction is one phase contraction (~10 ms at N=1424), and every
    consumer in the pipeline only indexes or iterates. Values match the
    batch-assembled S(k) to floating-point reordering (same S(R), same
    2*pi*k.R phases, tensordot instead of the batched einsum).
    """

    def __init__(self, stacked: StackedMatrices, kpoints):
        self._stacked = stacked
        self._kpoints = np.asarray(kpoints, float)

    def __len__(self):
        return self._kpoints.shape[0]

    def __getitem__(self, ik):
        phases = np.exp(2j * np.pi *
                        (self._stacked.R_vectors @ self._kpoints[ik]))
        return np.tensordot(phases, self._stacked.S_stack, axes=1)

    def __iter__(self):
        for ik in range(len(self)):
            yield self[ik]
