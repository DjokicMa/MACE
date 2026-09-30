"""
Eigenvalue Solver Module

This module contains functions for solving the generalized eigenvalue problem
H(k) C(k) = S(k) C(k) E(k) at each k-point.

Supports two backends:
- Sequential: Python loop over k-points (dict-based Fourier)
- Batched: Vectorized Fourier (einsum) + Python eigensolve loop
"""

import numpy as np
from scipy.linalg import eigh
from typing import Tuple, Dict, Optional
from .fourier import (
    fourier_transform_to_kspace,
    fourier_all_kpoints,
    StackedMatrices,
)


def solve_generalized_eigenvalue_problem(
    H_k: np.ndarray,
    S_k: np.ndarray,
    num_wann: int = None,
    overlap_threshold: float = 1e-6,
    subset_by_index=None,
    subset_by_value=None,
    eigvals_only: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve the generalized eigenvalue problem H(k) C = S(k) C E.

    Uses scipy.linalg.eigh for Hermitian matrices, which is more stable
    and efficient than the general eigenvalue solver. Falls back to
    S regularization when the overlap matrix is near-singular.

    Parameters
    ----------
    H_k : ndarray of shape (num_orbitals, num_orbitals)
        Hamiltonian matrix at k-point
    S_k : ndarray of shape (num_orbitals, num_orbitals)
        Overlap matrix at k-point
    num_wann : int, optional
        Number of lowest eigenvalues/eigenvectors to keep
        If None, keeps all
    overlap_threshold : float
        Eigenvalues of S below this threshold trigger regularization

    Returns
    -------
    eigenvalues : ndarray of shape (num_wann,)
        Eigenvalues sorted in ascending order
    eigenvectors : ndarray of shape (num_orbitals, num_wann)
        Eigenvectors (columns), normalized such that C† S C = I
    """
    # subset_by_index/value -> compute only the requested bands (scipy heevx/hegvx);
    # eigvals_only -> skip the eigenvector back-transform. Both reduce work when the
    # full spectrum/vectors aren't needed (see OPTIMIZATION_BACKLOG.md #1).
    eigh_kw = {}
    if subset_by_index is not None:
        eigh_kw['subset_by_index'] = subset_by_index
    if subset_by_value is not None:
        eigh_kw['subset_by_value'] = subset_by_value
    if eigvals_only:
        eigh_kw['eigvals_only'] = True

    try:
        result = eigh(H_k, S_k, **eigh_kw)
    except np.linalg.LinAlgError:
        # Overlap matrix is not positive definite — regularize it.
        # Adding a small shift eps*I to S makes it positive definite while
        # preserving the full basis dimension. High-energy eigenvalues from
        # regularized near-null-space directions will have degraded
        # projectability and get filtered downstream.
        s_evals, U = eigh(S_k)
        s_min = np.min(s_evals)

        # Shift S so its smallest eigenvalue equals overlap_threshold
        if s_min < overlap_threshold:
            shift = overlap_threshold - s_min
            S_reg = S_k + shift * np.eye(S_k.shape[0])
            result = eigh(H_k, S_reg, **eigh_kw)
        else:
            # S is actually positive definite; retry (shouldn't happen)
            result = eigh(H_k, S_k, **eigh_kw)

    # A subset_by_* request already restricts the output; only apply num_wann
    # truncation when no explicit subset was given (preserves prior behavior).
    _subset = subset_by_index is not None or subset_by_value is not None
    if eigvals_only:
        eigenvalues = result
        if num_wann is not None and not _subset:
            eigenvalues = eigenvalues[:num_wann]
        return eigenvalues, None

    eigenvalues, eigenvectors = result
    if num_wann is not None and not _subset:
        eigenvalues = eigenvalues[:num_wann]
        eigenvectors = eigenvectors[:, :num_wann]

    return eigenvalues, eigenvectors


def solve_kpoint(
    k_idx: int,
    k_point: np.ndarray,
    real_space_matrices: Dict[Tuple[int, int, int], Dict[str, np.ndarray]],
    lattice_vectors: np.ndarray,
    num_wann: int
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the generalized eigenvalue problem for a single k-point.

    This function is designed to be easily parallelizable.

    Parameters
    ----------
    k_idx : int
        Index of the k-point
    k_point : ndarray of shape (3,)
        k-point in fractional coordinates
    real_space_matrices : dict
        Real-space H(R) and S(R) matrices
    lattice_vectors : ndarray of shape (3, 3)
        Real-space lattice vectors
    num_wann : int
        Number of Wannier functions (bands to keep)

    Returns
    -------
    k_idx : int
        k-point index (for sorting in parallel execution)
    eigenvalues : ndarray
        Eigenvalues for this k-point
    eigenvectors : ndarray
        Eigenvectors for this k-point
    H_k : ndarray
        Hamiltonian at this k-point (for verification)
    S_k : ndarray
        Overlap at this k-point (for verification)
    """
    # Fourier transform to k-space
    H_k, S_k = fourier_transform_to_kspace(k_point, real_space_matrices, lattice_vectors)

    # Solve generalized eigenvalue problem
    eigenvalues, eigenvectors = solve_generalized_eigenvalue_problem(H_k, S_k, num_wann)

    return k_idx, eigenvalues, eigenvectors, H_k, S_k


def solve_all_kpoints_batched(
    k_points: np.ndarray,
    stacked: StackedMatrices,
    num_wann: int,
    subset: bool = False,
) -> Tuple[list, list, list]:
    """
    Solve eigenvalue problems using batch Fourier assembly.

    Assembles H(k) and S(k) for ALL k-points in one vectorized call,
    then loops only over eigensolves. Eliminates per-k-point Python
    overhead for the Fourier step.

    Parameters
    ----------
    k_points : ndarray of shape (num_kpoints, 3)
        All k-points in fractional coordinates
    stacked : StackedMatrices
        Pre-stacked R-space matrices
    num_wann : int
        Number of bands to solve for

    Returns
    -------
    eigenvalues_list : list of ndarrays
        Eigenvalues for each k-point
    eigenvectors_list : list of ndarrays
        Eigenvectors for each k-point
    S_k_list : list of ndarrays
        Overlap matrices for each k-point (needed for AMN/projectability)
    """
    K = len(k_points)

    # Batch Fourier assembly — single BLAS call for all k-points
    H_all, S_all = fourier_all_kpoints(k_points, stacked)

    eigenvalues_list = []
    eigenvectors_list = []
    S_k_list = []

    # subset=True -> LAPACK ?hegvx computes ONLY the lowest num_wann bands
    # (2.5x faster than full-solve-then-truncate at ~35% of the spectrum,
    # N=1424 benchmark) and the stored eigenvectors shrink accordingly.
    n_full = H_all.shape[1]
    use_subset = subset and num_wann is not None and num_wann < n_full
    for k_idx in range(K):
        if use_subset:
            eigenvalues, eigenvectors = solve_generalized_eigenvalue_problem(
                H_all[k_idx], S_all[k_idx],
                subset_by_index=[0, num_wann - 1]
            )
        else:
            eigenvalues, eigenvectors = solve_generalized_eigenvalue_problem(
                H_all[k_idx], S_all[k_idx], num_wann
            )
        eigenvalues_list.append(eigenvalues)
        eigenvectors_list.append(eigenvectors)
        S_k_list.append(S_all[k_idx])

    return eigenvalues_list, eigenvectors_list, S_k_list


def solve_all_kpoints_sequential(
    k_points: np.ndarray,
    real_space_matrices: Dict[Tuple[int, int, int], Dict[str, np.ndarray]],
    lattice_vectors: np.ndarray,
    num_wann: int
) -> Tuple[list, list, list, list]:
    """
    Solve eigenvalue problems at all k-points sequentially.

    Parameters
    ----------
    k_points : ndarray of shape (num_kpoints, 3)
        All k-points in fractional coordinates
    real_space_matrices : dict
        Real-space H(R) and S(R) matrices
    lattice_vectors : ndarray of shape (3, 3)
        Real-space lattice vectors
    num_wann : int
        Number of Wannier functions

    Returns
    -------
    eigenvalues_list : list of ndarrays
        Eigenvalues for each k-point
    eigenvectors_list : list of ndarrays
        Eigenvectors for each k-point
    H_k_list : list of ndarrays
        Hamiltonian matrices for each k-point
    S_k_list : list of ndarrays
        Overlap matrices for each k-point
    """
    eigenvalues_list = []
    eigenvectors_list = []
    H_k_list = []
    S_k_list = []

    for k_idx in range(len(k_points)):
        k_point = k_points[k_idx]
        _, eigenvalues, eigenvectors, H_k, S_k = solve_kpoint(
            k_idx, k_point, real_space_matrices, lattice_vectors, num_wann
        )
        eigenvalues_list.append(eigenvalues)
        eigenvectors_list.append(eigenvectors)
        H_k_list.append(H_k)
        S_k_list.append(S_k)

    return eigenvalues_list, eigenvectors_list, H_k_list, S_k_list


def solve_all_kpoints_parallel(
    k_points: np.ndarray,
    real_space_matrices: Dict[Tuple[int, int, int], Dict[str, np.ndarray]],
    lattice_vectors: np.ndarray,
    num_wann: int,
    num_processes: int = None
) -> Tuple[list, list, list, list]:
    """
    Solve eigenvalue problems at all k-points in parallel.

    Parameters
    ----------
    k_points : ndarray of shape (num_kpoints, 3)
        All k-points in fractional coordinates
    real_space_matrices : dict
        Real-space H(R) and S(R) matrices
    lattice_vectors : ndarray of shape (3, 3)
        Real-space lattice vectors
    num_wann : int
        Number of Wannier functions
    num_processes : int, optional
        Number of parallel processes (default: use all CPUs)

    Returns
    -------
    eigenvalues_list : list of ndarrays
        Eigenvalues for each k-point
    eigenvectors_list : list of ndarrays
        Eigenvectors for each k-point
    H_k_list : list of ndarrays
        Hamiltonian matrices for each k-point
    S_k_list : list of ndarrays
        Overlap matrices for each k-point
    """
    import multiprocessing as mp
    from functools import partial

    if num_processes is None:
        # Use at most half the CPUs to avoid saturating the system
        num_processes = min(max(1, mp.cpu_count() // 2), len(k_points))

    # Create partial function with fixed parameters
    solve_func = partial(
        solve_kpoint,
        real_space_matrices=real_space_matrices,
        lattice_vectors=lattice_vectors,
        num_wann=num_wann
    )

    # Prepare arguments
    args = [(k_idx, k_points[k_idx]) for k_idx in range(len(k_points))]

    # Run parallel computation
    with mp.Pool(processes=num_processes) as pool:
        results = pool.starmap(solve_func, args)

    # Sort results by k_idx
    results.sort(key=lambda x: x[0])

    # Extract results
    eigenvalues_list = [r[1] for r in results]
    eigenvectors_list = [r[2] for r in results]
    H_k_list = [r[3] for r in results]
    S_k_list = [r[4] for r in results]

    return eigenvalues_list, eigenvectors_list, H_k_list, S_k_list


# ---------------------------------------------------------------------------
# Shared-memory parallel eigensolve (spawn-safe; see PARALLELIZATION_PLAN.md)
#
# The per-k generalized eigensolve does not thread on Apple Accelerate (the
# tridiagonal reduction is memory-bound), so we parallelize across k-points with
# processes. Inputs (H(k), S(k)) and outputs (eigenvalues, eigenvectors) live in
# shared memory so nothing large is pickled across the process boundary — unlike
# the legacy solve_all_kpoints_parallel, which pickled H_k/S_k back per worker.
# ---------------------------------------------------------------------------

_SHARED: Dict[str, object] = {}


def _shared_worker_init(meta: dict) -> None:
    """Pool initializer: attach shared-memory views as worker-global arrays."""
    import os
    from multiprocessing import shared_memory
    # Pin BLAS to 1 thread/worker so P workers don't oversubscribe (harmless on
    # Accelerate where eigh is serial; required for a future OpenBLAS build).
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    nk, n, nw = meta['nk'], meta['n'], meta['nw']
    _SHARED['subset'] = meta.get('subset', False)
    _SHARED['sh'] = [shared_memory.SharedMemory(name=meta[x]) for x in ('H', 'S', 'V', 'W')]
    _SHARED['H'] = np.ndarray((nk, n, n), np.complex128, buffer=_SHARED['sh'][0].buf)
    _SHARED['S'] = np.ndarray((nk, n, n), np.complex128, buffer=_SHARED['sh'][1].buf)
    _SHARED['V'] = np.ndarray((nk, n, nw), np.complex128, buffer=_SHARED['sh'][2].buf)
    _SHARED['W'] = np.ndarray((nk, nw), np.float64, buffer=_SHARED['sh'][3].buf)
    _SHARED['nw'] = nw


def _shared_solve_k(k: int) -> int:
    """Worker task: solve k-point k in place into the shared output arrays."""
    if _SHARED.get('subset'):
        ev, evec = solve_generalized_eigenvalue_problem(
            _SHARED['H'][k], _SHARED['S'][k],
            subset_by_index=[0, _SHARED['nw'] - 1]
        )
    else:
        ev, evec = solve_generalized_eigenvalue_problem(
            _SHARED['H'][k], _SHARED['S'][k], _SHARED['nw']
        )
    _SHARED['V'][k] = evec
    _SHARED['W'][k] = ev
    return k


def choose_n_proc(num_kpoints: int, n_orbitals: int,
                  per_k_ref: float = 1.37, n_ref: int = 1424) -> int:
    """
    Resource trip-wires (measured): return the worker count, or 1 for serial.

    TW-1 (min-work): below ~0.6 s of serial eigensolve, the ~150 ms spawn cost
    isn't worth it -> serial. TW-2 (worker cap): speedup saturates at ~6 workers
    and regresses beyond (memory-bandwidth-bound), so never scale to core count.
    """
    import multiprocessing as mp
    per_k = per_k_ref * (n_orbitals / float(n_ref)) ** 3
    if per_k * num_kpoints < 0.6:                       # TW-1
        return 1
    return min(6, num_kpoints, max(1, mp.cpu_count() - 2))  # TW-2


def solve_all_kpoints_shared(
    k_points: np.ndarray,
    stacked: StackedMatrices,
    num_wann: int,
    n_proc: int = None,
    subset: bool = False,
) -> Tuple[list, list, list]:
    """
    Parallel eigensolve over k-points via spawn Pool + shared memory.

    Drop-in replacement for solve_all_kpoints_batched: same inputs, same
    (eigenvalues_list, eigenvectors_list, S_k_list) return. Eigenvalues are
    numerically identical to the serial path. Falls back to the batched solver
    when the trip-wires (choose_n_proc) select serial.
    """
    import multiprocessing as mp
    from multiprocessing import shared_memory

    nk = len(k_points)
    if n_proc is None:
        n_proc = choose_n_proc(nk, stacked.H.shape[-1] if hasattr(stacked, 'H') else 0)
    if n_proc <= 1:
        return solve_all_kpoints_batched(k_points, stacked, num_wann,
                                     subset=subset)

    # Batch-assemble H(k), S(k) once (vectorized), then fan out the eigensolves.
    H_all, S_all = fourier_all_kpoints(k_points, stacked)
    n = H_all.shape[1]
    nw = num_wann if num_wann is not None else n

    shH = shared_memory.SharedMemory(create=True, size=H_all.nbytes)
    shS = shared_memory.SharedMemory(create=True, size=S_all.nbytes)
    shV = shared_memory.SharedMemory(create=True, size=nk * n * nw * 16)
    shW = shared_memory.SharedMemory(create=True, size=nk * nw * 8)
    try:
        np.ndarray(H_all.shape, H_all.dtype, buffer=shH.buf)[:] = H_all
        np.ndarray(S_all.shape, S_all.dtype, buffer=shS.buf)[:] = S_all
        meta = dict(nk=nk, n=n, nw=nw, subset=bool(subset and nw < n),
                    H=shH.name, S=shS.name, V=shV.name, W=shW.name)
        ctx = mp.get_context('spawn')
        with ctx.Pool(n_proc, initializer=_shared_worker_init, initargs=(meta,)) as pool:
            pool.map(_shared_solve_k, range(nk))
        W = np.ndarray((nk, nw), np.float64, buffer=shW.buf).copy()
        V = np.ndarray((nk, n, nw), np.complex128, buffer=shV.buf).copy()
    finally:
        for sh in (shH, shS, shV, shW):
            sh.close()
            sh.unlink()

    eigenvalues_list = [W[k] for k in range(nk)]
    eigenvectors_list = [V[k] for k in range(nk)]
    S_k_list = [S_all[k] for k in range(nk)]
    return eigenvalues_list, eigenvectors_list, S_k_list


def solve_all_kpoints_auto(
    k_points: np.ndarray,
    stacked: StackedMatrices,
    num_wann: int,
    backend: str = 'auto',
    subset: bool = False,
) -> Tuple[list, list, list]:
    """
    Solve eigenvalue problems at all k-points using the batched vectorized backend.

    Parameters
    ----------
    k_points : ndarray of shape (num_kpoints, 3)
        All k-points in fractional coordinates
    stacked : StackedMatrices
        Pre-stacked R-space matrices
    num_wann : int
        Number of bands to solve for
    backend : str
        'auto' or 'python' - Use Python batched backend

    Returns
    -------
    eigenvalues_list : list of ndarrays
        Eigenvalues for each k-point
    eigenvectors_list : list of ndarrays
        Eigenvectors for each k-point
    S_k_list : list of ndarrays
        Overlap matrices for each k-point
    """
    if backend not in ('auto', 'python'):
        raise ValueError(
            f"Unknown backend: {backend!r}. Use 'auto' or 'python'."
        )
    return solve_all_kpoints_batched(k_points, stacked, num_wann,
                                     subset=subset)
