# Copyright (c) 2025 William Comaskey. MIT License, see LICENSE in this directory.
# Vendored into MACE from lcao2wannier 1.0.0; local changes are listed in VENDORED.md.
"""
Main Engine Module

This module contains the Wannier90Engine class that coordinates all
operations for converting LCAO calculations to Wannier90 format.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
import warnings

from .kpoints import generate_kpoint_grid, generate_neighbor_list
from .solver import (
    solve_all_kpoints_sequential,
    solve_all_kpoints_parallel,
    solve_all_kpoints_batched,
    solve_all_kpoints_auto,
    solve_all_kpoints_shared,
    choose_n_proc,
)
from .fourier import stack_real_space_matrices
from .wannier90 import write_wannier90_files
from .verification import run_all_verifications
from .utils import check_matrix_consistency, print_calculation_info
from .band_selection import (
    estimate_fermi_energy,
    analyze_band_window,
    print_band_analysis,
    check_frozen_continuity,
    validate_fermi_coverage,
    select_projection_orbitals,
    compute_subspace_projections,
    suggest_optimal_window,
    BandWindowResult,
    OrbitalSelectionResult
)
from .win_file import (
    write_win_file,
    create_win_config_from_engine,
    Wannier90WinConfig
)


class Wannier90Engine:
    """
    Main computational engine for lcao2wannier conversion.
    
    This class coordinates the entire workflow:
    1. K-point grid generation
    2. Fourier transforms and eigenvalue solving
    3. Band window analysis and selection
    4. Automatic projection orbital selection
    5. Verification of results
    6. Writing Wannier90 input files
    
    Attributes
    ----------
    real_space_matrices : dict
        Real-space H(R) and S(R) matrices
    k_grid : tuple of 3 ints
        Dimensions of k-point grid
    lattice_vectors : ndarray
        Real-space lattice vectors
    num_wann : int
        Number of Wannier functions (may be auto-determined from window)
    seedname : str
        Prefix for output files
    outer_window : tuple of 2 floats, optional
        Energy window (E_min, E_max) for band selection
    e_fermi : float, optional
        Fermi energy (auto-detected if None)
    kpoints : ndarray
        Generated k-point grid
    num_kpoints : int
        Total number of k-points
    num_orbitals : int
        Number of orbitals
    neighbor_list : dict
        K-point neighbor connectivity
    eigenvalues_list : list
        Computed eigenvalues for each k-point
    eigenvectors_list : list
        Computed eigenvectors for each k-point
    H_k_list : list
        Hamiltonian matrices at each k-point
    S_k_list : list
        Overlap matrices at each k-point
    band_analysis : BandWindowResult
        Results from band window analysis
    selected_band_indices : ndarray
        Indices of bands to include in output
    selected_orbital_indices : ndarray
        Indices of orbitals used for projections
    
    Examples
    --------
    Basic usage with all bands:
    
    >>> engine = Wannier90Engine(
    ...     real_space_matrices=matrices,
    ...     k_grid=(4, 4, 4),
    ...     lattice_vectors=lattice_vecs,
    ...     seedname='material'
    ... )
    >>> engine.run(parallel=True, verify=True)
    
    With automatic band selection using energy window:
    
    >>> engine = Wannier90Engine(
    ...     real_space_matrices=matrices,
    ...     k_grid=(15, 15, 1),
    ...     lattice_vectors=lattice_vecs,
    ...     seedname='bismuth',
    ...     outer_window=(-3.0, 5.0),  # eV relative to E_F
    ...     num_electrons=10
    ... )
    >>> engine.run(analyze_window=True)
    """
    
    def __init__(
        self,
        real_space_matrices: Dict[Tuple[int, int, int], Dict[str, np.ndarray]],
        k_grid: Tuple[int, int, int],
        lattice_vectors: np.ndarray,
        num_wann: Optional[int] = None,
        seedname: str = "wannier90",
        outer_window: Optional[Tuple[float, float]] = None,
        e_fermi: Optional[float] = None,
        num_electrons: Optional[int] = None,
        window_is_relative: bool = True
    ):
        """
        Initialize the Wannier90 engine.
        
        Parameters
        ----------
        real_space_matrices : dict
            Maps (R1, R2, R3) -> {'H': H_matrix, 'S': S_matrix}
        k_grid : tuple of 3 ints
            Dimensions of the k-point grid
        lattice_vectors : ndarray of shape (3, 3)
            Real-space lattice vectors (rows)
        num_wann : int, optional
            Number of Wannier functions. If None and outer_window is provided,
            will be determined automatically from the frozen bands.
            If None and no outer_window, uses all orbitals.
        seedname : str, optional
            Prefix for output files (default: 'wannier90')
        outer_window : tuple of 2 floats, optional
            Energy window (E_min, E_max) for band selection. Bands whose
            entire dispersion falls within this window become "frozen" bands.
            If None, all bands are used.
        e_fermi : float, optional
            Fermi energy in eV. If None and outer_window is provided,
            will be auto-detected from the band structure.
        num_electrons : int, optional
            Number of electrons in the system. Used for Fermi level
            auto-detection when e_fermi is None.
        window_is_relative : bool, optional
            If True (default), outer_window values are relative to E_F.
            If False, they are absolute energies.
        """
        self.real_space_matrices = real_space_matrices
        self.k_grid = k_grid
        self.lattice_vectors = lattice_vectors
        self.seedname = seedname

        # Compute reciprocal lattice (needed for neighbor list conversion)
        # b_i = 2π * (a_j × a_k) / (a_i · (a_j × a_k))
        self.recip_lattice = 2 * np.pi * np.linalg.inv(lattice_vectors).T

        # Band selection parameters
        self.outer_window = outer_window
        self.e_fermi = e_fermi
        self.num_electrons = num_electrons
        self.window_is_relative = window_is_relative
        self._user_num_wann = num_wann  # Store user-specified value
        
        # Check matrix consistency
        if not check_matrix_consistency(real_space_matrices):
            raise ValueError("Inconsistent matrix dimensions in real_space_matrices")
        
        # Get number of orbitals
        first_key = next(iter(real_space_matrices))
        self.num_orbitals = real_space_matrices[first_key]['H'].shape[0]
        
        # Set initial num_wann (may be updated after band analysis)
        if num_wann is not None:
            if num_wann > self.num_orbitals:
                raise ValueError(
                    f"num_wann ({num_wann}) cannot exceed num_orbitals ({self.num_orbitals})"
                )
            self.num_wann = num_wann
        else:
            # Will be determined later (from window analysis or full basis)
            self.num_wann = self.num_orbitals
        
        # Generate k-point grid and neighbor list
        self.kpoints = generate_kpoint_grid(k_grid)
        self.num_kpoints = len(self.kpoints)
        self.neighbor_list = generate_neighbor_list(k_grid)
        
        # Storage for results (initialized as empty)
        self.eigenvalues_list = []
        self.eigenvectors_list = []
        self.H_k_list = []
        self.S_k_list = []
        
        # Band selection results (initialized as None)
        self.band_analysis = None
        self.selected_band_indices = None
        self.selected_orbital_indices = None
        self.orbital_selection_result = None

        # Atomic basis information (for phase-corrected MMN, set later)
        self.atom_positions = None
        self.basis_atom_map = None

        # Override for num_iter (used by direct method to skip minimization)
        self._override_num_iter = None
        self._override_dis_num_iter = None

        # Disentanglement parameters (set by smart band selector)
        self._num_bands_for_win = None   # num_bands for .win (may differ from num_wann)
        self._dis_win = None             # (dis_win_min, dis_win_max) relative to E_F
        self._dis_froz = None            # (dis_froz_min, dis_froz_max) relative to E_F

        # Pre-stack R-space matrices for vectorized Fourier assembly
        self._stacked = stack_real_space_matrices(real_space_matrices)

        # PDWF target mask (set by _apply_method_pdwf for SVD-based Amn)
        self.pdwf_target_mask = None

        # Conditioning validation result (set by solve_all_kpoints)
        self._conditioning_result = None

        # Print initialization info
        self._print_init_info()
    
    def _print_init_info(self):
        """Print initialization information."""
        print("\n" + "=" * 70)
        print("lcao2wannier Engine Initialized")
        print("=" * 70)
        print(f"Seedname: {self.seedname}")
        print(f"K-grid: {self.k_grid}")
        print(f"Number of k-points: {self.num_kpoints}")
        print(f"Number of orbitals: {self.num_orbitals}")
        if self._user_num_wann is not None:
            print(f"Number of Wannier functions: {self._user_num_wann} (user-specified)")
        elif self.outer_window is not None:
            print(f"Number of Wannier functions: (to be determined from window)")
        else:
            print(f"Number of Wannier functions: {self.num_orbitals} (full basis)")
        print(f"Number of R-vectors: {len(self.real_space_matrices)}")
        
        if self.outer_window is not None:
            window_type = "relative to E_F" if self.window_is_relative else "absolute"
            print(f"Energy window: [{self.outer_window[0]:.2f}, {self.outer_window[1]:.2f}] eV ({window_type})")
            if self.e_fermi is not None:
                print(f"Fermi energy: {self.e_fermi:.4f} eV (user-specified)")
            else:
                print(f"Fermi energy: (to be auto-detected)")
        
        print("=" * 70)
    
    def solve_all_kpoints(
        self,
        parallel: bool = True,
        num_processes: Optional[int] = None,
        convert_to_eV: bool = True,
        backend: str = 'auto',
        validate_overlap: bool = True,
        overlap_strict: bool = True,
        num_bands: Optional[int] = None,
        store_S: bool = True
    ) -> None:
        """
        Solve the generalized eigenvalue problem for all k-points.

        This solves for ALL bands initially. Band selection is applied
        later in analyze_bands() or write_files().

        Parameters
        ----------
        parallel : bool, optional
            Use parallel processing (default: True)
        num_processes : int, optional
            Number of processes for parallel computation
            If None, uses all available CPUs
        convert_to_eV : bool, optional
            Convert eigenvalues from Hartree to eV (default: True)
            Set to False if your Fock matrix is already in eV
        backend : str, optional
            Eigensolve backend: 'auto' (Fortran if available, else Python),
            'fortran', or 'python' (default: 'auto')
        validate_overlap : bool, optional
            Run overlap matrix conditioning check before eigensolving
            (default: True)
        overlap_strict : bool, optional
            If True, abort on badly-conditioned overlap matrices
            (default: True)
        """
        print(f"\n{'=' * 70}")
        print(f"Solving Eigenvalue Problems at {self.num_kpoints} K-Points")
        print(f"{'=' * 70}")

        # Pre-solve overlap conditioning check
        if validate_overlap and self._stacked is not None:
            from .conditioning import (
                validate_overlap_conditioning,
                OverlapConditioningError,
            )
            try:
                self._conditioning_result = validate_overlap_conditioning(
                    self._stacked,
                    k_grid=self.k_grid,
                    verbose=True,
                    strict=overlap_strict,
                )
            except OverlapConditioningError:
                print("\nAborting: fix the R-vector count and re-run.")
                raise

        # Solve all orbitals by default; num_bands requests only the lowest
        # bands via LAPACK ?hegvx (2.5x faster at ~35% of the spectrum,
        # N=1424 benchmark) and shrinks the stored eigenvectors accordingly.
        _subset = num_bands is not None and num_bands < self.num_orbitals
        solve_num_bands = num_bands if _subset else self.num_orbitals
        if _subset:
            print(f"Band-subset solve: lowest {solve_num_bands} of "
                  f"{self.num_orbitals} bands (LAPACK subset driver)")

        # Determine which backend to use.
        # For the default ('auto'/'python') backend, choose_n_proc applies the
        # measured resource trip-wires (PARALLELIZATION_PLAN.md): it returns >1
        # only when the per-k eigensolves are large enough to amortize process
        # spawn, and caps workers at ~6 (speedup saturates there, regresses past).
        # Otherwise we fall back to the serial batched path, byte-for-byte unchanged.
        # num_processes=1 forces serial; num_processes=N overrides the trip-wire.
        use_python_backend = backend in ('auto', 'python')
        if parallel and use_python_backend and num_processes != 1:
            n_proc = num_processes or choose_n_proc(self.num_kpoints, self.num_orbitals)
        else:
            n_proc = 1

        if use_python_backend and n_proc > 1:
            print(f"Mode: Shared-memory parallel ({n_proc} workers)")
            eig_list, evec_list, sk_list = solve_all_kpoints_shared(
                self.kpoints,
                self._stacked,
                solve_num_bands,
                n_proc=n_proc,
                subset=_subset,
            )
        else:
            print(f"Mode: Batched vectorized")
            eig_list, evec_list, sk_list = solve_all_kpoints_auto(
                self.kpoints,
                self._stacked,
                solve_num_bands,
                backend=backend,
                subset=_subset,
            )
        self.eigenvalues_list = eig_list
        self.eigenvectors_list = evec_list
        self.H_k_list = []  # Not stored in batched/parallel mode
        if store_S:
            self.S_k_list = sk_list
        else:
            # --memory low: drop the materialized S(k) stack (nk*N^2*16 B;
            # 1.17 GB at Sc scale) and reconstruct per k on demand
            from .fourier import LazySkList
            del sk_list
            self.S_k_list = LazySkList(self._stacked, self.kpoints)
            print("Low-memory mode: S(k) reconstructed on demand "
                  "(not retained)")
        
        # Convert eigenvalues from Hartree to eV
        if convert_to_eV:
            HARTREE_TO_EV = 27.2114
            self.eigenvalues_list = [eigs * HARTREE_TO_EV for eigs in self.eigenvalues_list]
            print(f"✓ Eigenvalues converted from Hartree to eV")
        
        print(f"✓ Eigenvalue problems solved successfully")
        print(f"{'=' * 70}")
    
    def analyze_bands(self, verbose: bool = True) -> Optional[BandWindowResult]:
        """
        Analyze band structure and determine frozen bands from energy window.
        
        This method identifies which bands fall entirely within the specified
        outer_window, making them suitable as frozen bands for Wannier90.
        
        Must be called after solve_all_kpoints().
        
        Parameters
        ----------
        verbose : bool, optional
            Print detailed analysis (default: True)
            
        Returns
        -------
        BandWindowResult or None
            Analysis results, or None if no window was specified
            
        Raises
        ------
        RuntimeError
            If called before solve_all_kpoints()
        """
        if not self.eigenvalues_list:
            raise RuntimeError("Must call solve_all_kpoints() first")
        
        # If no window specified, use all bands
        if self.outer_window is None:
            self.selected_band_indices = np.arange(self.num_orbitals)
            self.num_wann = self._user_num_wann or self.num_orbitals
            if verbose:
                print(f"\nNo energy window specified - using all {self.num_wann} bands")
            return None
        
        print(f"\n{'=' * 70}")
        print("Band Window Analysis")
        print(f"{'=' * 70}")
        
        # Auto-detect Fermi level if needed
        if self.e_fermi is None:
            self.e_fermi = estimate_fermi_energy(
                self.eigenvalues_list,
                num_electrons=self.num_electrons,
                method='auto'
            )
            if verbose:
                print(f"Auto-detected Fermi energy: {self.e_fermi:.4f} eV")
        
        # Analyze window
        self.band_analysis = analyze_band_window(
            self.eigenvalues_list,
            outer_window=self.outer_window,
            e_fermi=self.e_fermi,
            window_is_relative=self.window_is_relative,
            num_electrons=self.num_electrons
        )
        
        if verbose:
            print_band_analysis(self.band_analysis)
        
        # Check for issues
        if self.band_analysis.num_wann == 0:
            warnings.warn(
                "No bands are fully contained within the energy window! "
                "Consider widening the window."
            )
            return self.band_analysis
        
        is_contiguous, gaps = check_frozen_continuity(
            self.band_analysis.frozen_indices
        )
        
        coverage = validate_fermi_coverage(
            self.band_analysis.frozen_indices,
            self.eigenvalues_list,
            self.band_analysis.e_fermi
        )
        
        if verbose:
            if not is_contiguous:
                print("\n⚠ Warning: Frozen bands are non-contiguous!")
            if not coverage['has_occupied']:
                print("⚠ Warning: No occupied bands in frozen window!")
            if not coverage['has_unoccupied']:
                print("⚠ Warning: No unoccupied bands in frozen window!")
            if coverage['crosses_fermi']:
                print("ℹ Note: Some bands cross the Fermi level (metallic)")
        
        # Set selected bands
        self.selected_band_indices = self.band_analysis.frozen_indices
        
        # Update num_wann (user-specified takes precedence)
        if self._user_num_wann is not None:
            if self._user_num_wann > self.band_analysis.num_wann:
                warnings.warn(
                    f"User-specified num_wann ({self._user_num_wann}) exceeds frozen bands "
                    f"({self.band_analysis.num_wann}). Using frozen band count."
                )
                self.num_wann = self.band_analysis.num_wann
            else:
                self.num_wann = self._user_num_wann
        else:
            self.num_wann = self.band_analysis.num_wann
        
        print(f"{'=' * 70}")
        
        return self.band_analysis
    
    def select_projections(self, verbose: bool = True, method: str = 'weight',
                           orbital_mask: Optional[np.ndarray] = None) -> OrbitalSelectionResult:
        """
        Automatically select optimal projection orbitals.

        For LCAO bases, this selects the orbitals with the largest
        contribution to the selected bands, providing optimal initial
        projections for Wannier90 without user specification.

        Must be called after analyze_bands() or after setting
        selected_band_indices manually.

        Parameters
        ----------
        verbose : bool, optional
            Print selection details (default: True)
        method : str, optional
            Projection selection method:
            - 'weight': Simple weight-based ranking (default, original method)
            - 'scdm': SCDM-L (QR column pivoting on Lowdin density matrix)
        orbital_mask : ndarray of bool, optional
            Boolean mask for constraining SCDM orbital selection.
            Only used when method='scdm'. See scdm_select_projections.

        Returns
        -------
        OrbitalSelectionResult
            Contains selected orbital indices and weights

        Raises
        ------
        RuntimeError
            If called before bands are selected
        """
        if self.selected_band_indices is None:
            raise RuntimeError("Must call analyze_bands() first or set selected_band_indices")

        if not self.eigenvectors_list:
            raise RuntimeError("Must call solve_all_kpoints() first")

        print(f"\n{'=' * 70}")
        print(f"Automatic Projection Orbital Selection (method={method})")
        print(f"{'=' * 70}")

        if method == 'scdm':
            from lcao2wannier.band_selection import scdm_select_projections
            self.orbital_selection_result = scdm_select_projections(
                self.eigenvectors_list,
                self.S_k_list,
                self.eigenvalues_list,
                num_wann=self.num_wann,
                e_fermi=self.e_fermi if self.e_fermi is not None else 0.0,
                band_indices=self.selected_band_indices,
                orbital_mask=orbital_mask,
                verbose=verbose,
            )
        else:
            self.orbital_selection_result = select_projection_orbitals(
                self.eigenvectors_list,
                self.S_k_list,
                num_wann=self.num_wann,
                band_indices=self.selected_band_indices
            )

        self.selected_orbital_indices = self.orbital_selection_result.selected_indices

        if verbose:
            print(f"Selected {self.orbital_selection_result.num_selected} projection orbitals")
            print(f"Orbital indices: {self.selected_orbital_indices}")

            if method != 'scdm':
                # Show top contributors (SCDM already prints its own diagnostics)
                weights = self.orbital_selection_result.orbital_weights
                top_indices = np.argsort(weights)[-5:][::-1]
                print("\nTop 5 contributing orbitals:")
                for idx in top_indices:
                    print(f"  Orbital {idx}: weight = {weights[idx]:.4f}")

        print(f"{'=' * 70}")

        return self.orbital_selection_result
    
    def suggest_window(
        self,
        target_num_wann: Optional[int] = None,
        padding: float = 0.1
    ) -> Tuple[float, float]:
        """
        Suggest an optimal energy window based on band structure.
        
        Must be called after solve_all_kpoints().
        
        Parameters
        ----------
        target_num_wann : int, optional
            Target number of Wannier functions. If provided, attempts
            to find a window containing approximately this many bands.
        padding : float, optional
            Extra energy padding (eV) around the tight band edges
            
        Returns
        -------
        tuple of 2 floats
            Suggested (E_min, E_max) window in absolute energies
        """
        if not self.eigenvalues_list:
            raise RuntimeError("Must call solve_all_kpoints() first")
        
        if self.e_fermi is None:
            self.e_fermi = estimate_fermi_energy(
                self.eigenvalues_list,
                num_electrons=self.num_electrons,
                method='auto'
            )
        
        suggested = suggest_optimal_window(
            self.eigenvalues_list,
            e_fermi=self.e_fermi,
            target_num_wann=target_num_wann,
            padding=padding
        )
        
        print(f"\nSuggested energy window: [{suggested[0]:.2f}, {suggested[1]:.2f}] eV (absolute)")
        print(f"                         [{suggested[0] - self.e_fermi:.2f}, "
              f"{suggested[1] - self.e_fermi:.2f}] eV (relative to E_F)")
        
        return suggested
    
    def verify_results(self) -> dict:
        """
        Run all verification checks on computed results.
        
        Returns
        -------
        dict
            Dictionary containing verification results
        """
        if not self.eigenvalues_list:
            raise RuntimeError("No results to verify. Run solve_all_kpoints() first.")
        
        print(f"\n{'=' * 70}")
        print("Verification Checks")
        print(f"{'=' * 70}")
        
        # If band selection is active, extract only selected bands for verification
        if self.selected_band_indices is not None:
            band_indices = self.selected_band_indices
            eigenvalues_to_verify = [eigs[band_indices] for eigs in self.eigenvalues_list]
            eigenvectors_to_verify = [vecs[:, band_indices] for vecs in self.eigenvectors_list]
            num_bands_to_verify = len(band_indices)
            print(f"Verifying {num_bands_to_verify} selected bands (indices {band_indices[0]}-{band_indices[-1]})")
        else:
            eigenvalues_to_verify = self.eigenvalues_list
            eigenvectors_to_verify = self.eigenvectors_list
            num_bands_to_verify = self.num_wann
        
        # If H_k_list is empty (batched mode), recompute for verification
        H_k_for_verify = self.H_k_list
        if not H_k_for_verify and self._stacked is not None:
            from .fourier import fourier_transform_vectorized
            H_k_for_verify = []
            for k_idx in range(len(self.kpoints)):
                H_k, _ = fourier_transform_vectorized(self.kpoints[k_idx], self._stacked)
                H_k_for_verify.append(H_k)

        results = run_all_verifications(
            eigenvalues_to_verify,
            eigenvectors_to_verify,
            H_k_for_verify,
            self.S_k_list,
            num_bands_to_verify,
            verbose=True
        )
        
        print(f"{'=' * 70}")
        
        return results
    
    def write_files(
        self,
        verbose: bool = True,
        write_win: bool = True,
        use_nnkp: bool = True,
        atoms: Optional[List[Tuple[str, np.ndarray]]] = None,
        projections: Optional[List[str]] = None,
        spinors: bool = True,
        bands_plot: bool = False,
        kpoint_path: Optional[List[Tuple[str, np.ndarray]]] = None,

        mmn_method: str = 'midpoint',

    ) -> None:
        """
        Write all Wannier90 input files (.win, .eig, .amn, .mmn).

        Uses only the selected bands if band analysis has been performed.

        IMPORTANT: For best results, run 'wannier90.x -pp seedname' first to
        generate the .nnkp file, then call this method with use_nnkp=True.
        This ensures the .mmn file uses exactly the neighbor structure that
        Wannier90 expects.

        Parameters
        ----------
        verbose : bool, optional
            Print progress messages (default: True)
        write_win : bool, optional
            Write the .win parameter file (default: True)
        use_nnkp : bool, optional
            Use neighbor list from .nnkp file if available (default: True)
            If True and .nnkp file exists, uses those neighbors
            If True and .nnkp missing, falls back to generated neighbors with warning
            If False, always uses internally generated neighbors
        atoms : list of tuples, optional
            Atomic positions as [(symbol, frac_coords), ...]
            If None, atoms section will be omitted from .win file
        projections : list of str, optional
            Wannier90 projection specifications
            If None, uses 'random' (relies on .amn file)
        spinors : bool, optional
            Whether calculation includes spin-orbit coupling (default: True)
        bands_plot : bool, optional
            Enable band structure plotting in Wannier90 (default: False)
        kpoint_path : list of tuples, optional
            High-symmetry path as [(label, coords), ...]
        """
        if not self.eigenvalues_list:
            raise RuntimeError("No results to write. Run solve_all_kpoints() first.")

        print(f"\n{'=' * 70}")
        print("Writing Wannier90 Files")
        print(f"{'=' * 70}")

        # Determine which bands to use
        if self.selected_band_indices is not None:
            band_indices = self.selected_band_indices
            if verbose:
                print(f"Using {len(band_indices)} selected bands: "
                      f"{band_indices[0]}-{band_indices[-1]}")
        else:
            band_indices = np.arange(self.num_wann)
            if verbose:
                print(f"Using first {self.num_wann} bands")

        # Extract selected bands
        eigenvalues_selected = [
            eig[band_indices] for eig in self.eigenvalues_list
        ]
        eigenvectors_selected = [
            C[:, band_indices] for C in self.eigenvectors_list
        ]

        # Extract selected orbital subspace for projections
        if self.selected_orbital_indices is not None:
            orbital_indices = self.selected_orbital_indices
            S_k_selected = [
                S_k[orbital_indices, :] for S_k in self.S_k_list
            ]
            if verbose:
                print(f"\n✓ Using {len(orbital_indices)} selected projection orbitals")
        else:
            S_k_selected = self.S_k_list
            if verbose:
                print(f"\n⚠ Warning: No orbital selection performed, using all orbitals")

        # Determine which neighbor list to use
        neighbor_list_to_use = self.neighbor_list
        if use_nnkp:
            from os.path import exists
            from .kpoints import read_nnkp_neighbors

            nnkp_file = f"{self.seedname}.nnkp"
            if exists(nnkp_file):
                if verbose:
                    print(f"\n✓ Reading neighbor list from {nnkp_file}")
                try:
                    neighbor_list_to_use = read_nnkp_neighbors(nnkp_file)
                    if verbose:
                        num_neighbors = len(neighbor_list_to_use[0])
                        print(f"  Using {num_neighbors} neighbors per k-point from Wannier90")
                except Exception as e:
                    print(f"\n⚠ Warning: Could not read {nnkp_file}: {e}")
                    print(f"  Falling back to internally generated neighbors")
            else:
                print(f"\n⚠ Warning: {nnkp_file} not found")
                print(f"  Using internally generated neighbors (may not match Wannier90)")
                print(f"  Recommendation: Run 'wannier90.x -pp {self.seedname}' first")

        # Write .eig file
        from .wannier90 import write_eig_file, write_amn_file, write_mmn_file

        # Shift eigenvalues to be relative to Fermi energy for Wannier90
        # (when fermi_energy is specified in .win, Wannier90 expects relative energies)
        if self.e_fermi is not None:
            eigenvalues_relative = []
            for eigs in eigenvalues_selected:
                eigenvalues_relative.append(eigs - self.e_fermi)
        else:
            # No Fermi energy set — use absolute eigenvalues
            eigenvalues_relative = eigenvalues_selected

        eig_file = f"{self.seedname}.eig"
        write_eig_file(eig_file, eigenvalues_relative, self.num_kpoints, len(band_indices))
        if verbose:
            num_entries = self.num_kpoints * len(band_indices)
            if self.e_fermi is not None:
                print(f"  ✓ {eig_file}: {num_entries} eigenvalues (relative to E_F = {self.e_fermi:.6f} eV)")
            else:
                print(f"  ✓ {eig_file}: {num_entries} eigenvalues (absolute energies, no E_F set)")

        # DISABLED: Auto-update feature commented out to allow num_bands > actual bands in window
        # This is needed for disentanglement where num_bands should be > num_wann
        #
        # # Validate/update .win file's num_bands to match .eig file
        # actual_bands_written = len(band_indices)
        # win_file = f"{self.seedname}.win"
        #
        # try:
        #     from .win_file import read_win_parameter, update_win_parameter
        #     current_num_bands = read_win_parameter(win_file, 'num_bands')
        #
        #     if current_num_bands is not None and current_num_bands != actual_bands_written:
        #         if verbose:
        #             print(f"\n⚠ IMPORTANT: Updating {win_file}")
        #             print(f"  num_bands: {current_num_bands} → {actual_bands_written}")
        #             print(f"  Reason: Energy window contains {actual_bands_written} bands, not {current_num_bands}")
        #
        #         update_win_parameter(win_file, 'num_bands', actual_bands_written)
        #
        #         if verbose:
        #             print(f"  ✓ Updated {win_file}")
        #             print()
        #             print(f"  {'='*76}")
        #             print(f"  NEXT STEPS:")
        #             print(f"  1. Re-run preprocessing: wannier90.x -pp {self.seedname}")
        #             print(f"  2. Re-run Stage 2:       python3 lcao_to_wannier90.py --stage 2 ...")
        #             print(f"  {'='*76}")
        # except Exception as e:
        #     if verbose:
        #         print(f"\n⚠ Warning: Could not validate/update .win file: {e}")

        # Write .amn file (uses overlap-corrected LCAO projections)
        amn_file = f"{self.seedname}.amn"

        if self.pdwf_target_mask is not None:
            # PDWF method: SVD-based projection onto full target subspace
            from .wannier90 import write_amn_file_pdwf
            write_amn_file_pdwf(
                amn_file,
                self.eigenvectors_list,
                self.S_k_list,
                self.pdwf_target_mask,
                band_indices,
                self.num_kpoints,
                self.num_wann,
            )
            if verbose:
                n_target = int(np.sum(self.pdwf_target_mask))
                print(f"  ℹ PDWF Amn: SVD projection onto {n_target} target AOs → {self.num_wann} WFs")
        elif self.selected_orbital_indices is not None:
            from .wannier90 import write_amn_file_lcao
            write_amn_file_lcao(
                amn_file,
                self.eigenvectors_list,
                self.S_k_list,
                self.selected_orbital_indices,
                band_indices,
                self.num_kpoints,
                len(band_indices),
            )
        else:
            write_amn_file(amn_file, eigenvectors_selected, S_k_selected, self.num_kpoints, len(band_indices))
        if verbose:
            num_entries = self.num_kpoints * len(band_indices) * len(band_indices)
            print(f"  ✓ {amn_file}: {num_entries} matrix elements")

        # Write .mmn file using Löwdin orthogonalization method
        mmn_file = f"{self.seedname}.mmn"
        if self.atom_positions is not None and self.basis_atom_map is not None:
            from .wannier90 import write_mmn_file_lcao

            # Convert neighbor list to dict format if needed
            # Old format: List[Tuple[int, np.ndarray]] per k-point
            # New format: List[Dict[str, Any]] per k-point with 'id', 'G_shift', 'b_vec_cart'
            if neighbor_list_to_use and isinstance(neighbor_list_to_use[0][0], tuple):
                # Old format - convert to new (pass kpoints for correct b_vec_cart)
                if verbose:
                    print("  ℹ Converting internally generated neighbors to dict format...")
                from .kpoints import convert_neighbor_list_to_dict_format
                neighbor_list_to_use = convert_neighbor_list_to_dict_format(
                    neighbor_list_to_use,
                    self.recip_lattice,
                    kpoints=self.kpoints,
                )

            write_mmn_file_lcao(
                mmn_file,
                self.eigenvectors_list,  # Pass FULL eigenvectors (Löwdin selects bands)
                self.kpoints,
                self.real_space_matrices,
                self.lattice_vectors,
                neighbor_list_to_use,
                self.atom_positions,
                self.basis_atom_map,
                self.num_kpoints,
                len(band_indices),
                convention='pi',
                use_direct_method=True,
                verbose=verbose,
                stacked=self._stacked,
                unitarize=False,
                method=mmn_method,
                S_k_list=self.S_k_list,
                band_indices=band_indices,
                recip_lattice=self.recip_lattice,
                gto_aos=getattr(self, 'gto_aos', None),
                gto_cutoff=getattr(self, 'gto_cutoff', 16.0),
            )
        else:
            # Fallback to standard MMN writer (no phase correction)
            write_mmn_file(
                mmn_file, eigenvectors_selected, self.S_k_list, neighbor_list_to_use,
                self.num_kpoints, len(band_indices)
            )
        if verbose:
            num_neighbors = len(neighbor_list_to_use[0])
            num_entries = self.num_kpoints * num_neighbors * len(band_indices) * len(band_indices)
            print(f"  ✓ {mmn_file}: {num_entries} matrix elements")

        # Write parameter file (.win)
        if write_win:
            extra_kwargs = {}
            if self._override_dis_num_iter is not None:
                extra_kwargs['dis_num_iter'] = self._override_dis_num_iter
            extra_kwargs.update(getattr(self, '_additional_keywords', {}))
            win_config = create_win_config_from_engine(
                self,
                atoms=atoms,
                projections=projections,
                spinors=spinors,
                bands_plot=bands_plot,
                kpoint_path=kpoint_path,
                write_hr=True,
                use_bloch_phases=False,
                num_iter=self._override_num_iter,
                guiding_centres=getattr(self, '_guiding_centres', False),
                **extra_kwargs,
            )
            write_win_file(self.seedname, win_config, verbose=verbose)

        print(f"{'=' * 70}")
    
    def run(
        self,
        parallel: bool = True,
        verify: bool = True,
        analyze_window: bool = True,
        select_orbitals: bool = True,
        num_processes: Optional[int] = None
    ) -> dict:
        """
        Run the complete workflow.
        
        This method executes the entire calculation pipeline:
        1. Solve eigenvalue problems at all k-points
        2. Analyze band window and select frozen bands (if outer_window specified)
        3. Automatically select projection orbitals (optional)
        4. Verify results (optional)
        5. Write Wannier90 input files
        
        Parameters
        ----------
        parallel : bool, optional
            Use parallel processing (default: True)
        verify : bool, optional
            Run verification checks (default: True)
        analyze_window : bool, optional
            Perform band window analysis if outer_window is set (default: True)
        select_orbitals : bool, optional
            Automatically select projection orbitals (default: True)
        num_processes : int, optional
            Number of processes for parallel computation
        
        Returns
        -------
        dict
            Results dictionary containing:
            - 'verification': Verification results (if verify=True)
            - 'band_analysis': Band window analysis (if performed)
            - 'orbital_selection': Orbital selection results (if performed)
            - 'eigenvalues': All computed eigenvalues
            - 'selected_bands': Indices of selected bands
            - 'num_wann': Final number of Wannier functions
        
        Examples
        --------
        Basic usage:
        
        >>> engine = Wannier90Engine(...)
        >>> results = engine.run()
        
        With energy window:
        
        >>> engine = Wannier90Engine(..., outer_window=(-3.0, 5.0))
        >>> results = engine.run(analyze_window=True)
        """
        print("\n" + "=" * 70)
        print("lcao2wannier Complete Workflow")
        print("=" * 70)
        
        # Print calculation info
        print_calculation_info(
            self.num_kpoints,
            self.k_grid,
            self.num_orbitals,
            self.num_wann
        )
        
        # Step 1: Solve eigenvalue problems (always for all bands)
        self.solve_all_kpoints(parallel=parallel, num_processes=num_processes)
        
        # Step 2: Band window analysis (if requested and window specified)
        if analyze_window and self.outer_window is not None:
            self.analyze_bands(verbose=True)
            
            # Step 3: Automatic orbital selection (if requested)
            if select_orbitals and self.band_analysis is not None:
                if self.band_analysis.num_wann > 0:
                    self.select_projections(verbose=True)
        else:
            # No window analysis - use all bands or user-specified num_wann
            self.selected_band_indices = np.arange(self.num_wann)
        
        # Step 4: Verification (optional)
        verification_results = None
        if verify:
            verification_results = self.verify_results()
        
        # Step 5: Write output files
        self.write_files(verbose=True)
        
        # Final summary
        print("\n" + "=" * 70)
        print("✓ Workflow Completed Successfully!")
        print("=" * 70)
        print(f"Output files created:")
        print(f"  • {self.seedname}.win - Wannier90 input parameters")
        print(f"  • {self.seedname}.eig - Band energies ({self.num_wann} bands)")
        print(f"  • {self.seedname}.amn - Projection matrices")
        print(f"  • {self.seedname}.mmn - Overlap matrices")
        if self.band_analysis is not None:
            print(f"\nBand selection:")
            print(f"  • Frozen bands: {self.band_analysis.num_wann}")
            print(f"  • Energy range: [{self.band_analysis.frozen_energy_range[0]:.2f}, "
                  f"{self.band_analysis.frozen_energy_range[1]:.2f}] eV")
        print("=" * 70)
        
        return {
            'verification': verification_results,
            'band_analysis': self.band_analysis,
            'orbital_selection': self.orbital_selection_result,
            'eigenvalues': self.eigenvalues_list,
            'selected_bands': self.selected_band_indices,
            'num_wann': self.num_wann
        }
    
    def get_band_structure(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the computed band structure.
        
        Returns
        -------
        kpoints : ndarray
            K-points in fractional coordinates
        eigenvalues : ndarray of shape (num_kpoints, num_bands)
            Eigenvalues at each k-point (all bands, not just selected)
        """
        if not self.eigenvalues_list:
            raise RuntimeError("No band structure available. Run solve_all_kpoints() first.")
        
        eigenvalues_array = np.array([eigs.real for eigs in self.eigenvalues_list])
        return self.kpoints, eigenvalues_array
    
    def get_selected_band_structure(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the band structure for selected bands only.
        
        Returns
        -------
        kpoints : ndarray
            K-points in fractional coordinates
        eigenvalues : ndarray of shape (num_kpoints, num_selected_bands)
            Eigenvalues at each k-point for selected bands
        """
        if not self.eigenvalues_list:
            raise RuntimeError("No band structure available. Run solve_all_kpoints() first.")
        
        if self.selected_band_indices is None:
            return self.get_band_structure()
        
        eigenvalues_array = np.array([
            eigs[self.selected_band_indices].real 
            for eigs in self.eigenvalues_list
        ])
        return self.kpoints, eigenvalues_array
    
    def get_density_of_states(
        self,
        energy_range: Optional[Tuple[float, float]] = None,
        num_bins: int = 100,
        selected_only: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute a simple density of states from the band structure.
        
        Parameters
        ----------
        energy_range : tuple of 2 floats, optional
            (E_min, E_max) for the DOS range
        num_bins : int, optional
            Number of energy bins (default: 100)
        selected_only : bool, optional
            If True, compute DOS only for selected bands (default: False)
        
        Returns
        -------
        energies : ndarray
            Energy values at bin centers
        dos : ndarray
            Density of states
        """
        if not self.eigenvalues_list:
            raise RuntimeError("No data available. Run solve_all_kpoints() first.")
        
        if selected_only and self.selected_band_indices is not None:
            all_eigenvalues = np.concatenate([
                eigs[self.selected_band_indices].real 
                for eigs in self.eigenvalues_list
            ])
        else:
            all_eigenvalues = np.concatenate([
                eigs.real for eigs in self.eigenvalues_list
            ])
        
        if energy_range is None:
            energy_range = (np.min(all_eigenvalues), np.max(all_eigenvalues))
        
        dos, bin_edges = np.histogram(all_eigenvalues, bins=num_bins, range=energy_range)
        energies = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Normalize
        dos = dos.astype(float) / (self.num_kpoints * (energies[1] - energies[0]))
        
        return energies, dos
    
    def get_energy_statistics(self) -> Dict[str, float]:
        """
        Get energy statistics for the computed band structure.
        
        Returns
        -------
        dict
            Dictionary with keys:
            - 'e_min': Minimum eigenvalue
            - 'e_max': Maximum eigenvalue
            - 'e_fermi': Fermi energy (if determined)
            - 'bandwidth': Total bandwidth
            - 'gap': Band gap (if semiconductor, else None)
        """
        if not self.eigenvalues_list:
            raise RuntimeError("No data available. Run solve_all_kpoints() first.")
        
        all_eigenvalues = np.array(self.eigenvalues_list)
        
        e_min = np.min(all_eigenvalues)
        e_max = np.max(all_eigenvalues)
        
        stats = {
            'e_min': float(e_min),
            'e_max': float(e_max),
            'e_fermi': float(self.e_fermi) if self.e_fermi is not None else None,
            'bandwidth': float(e_max - e_min),
            'gap': None
        }
        
        # Try to detect a gap
        num_bands = all_eigenvalues.shape[1]
        avg_energies = np.mean(all_eigenvalues, axis=0)
        gaps = np.diff(avg_energies)
        
        # Find the largest gap
        if len(gaps) > 0:
            max_gap_idx = np.argmax(gaps)
            max_gap = gaps[max_gap_idx]
            if max_gap > 0.1:  # Threshold for considering it a gap
                stats['gap'] = float(max_gap)
        
        return stats
    
    def __repr__(self):
        """String representation of the engine."""
        lines = [
            "Wannier90Engine(",
            f"  seedname='{self.seedname}',",
            f"  k_grid={self.k_grid},",
            f"  num_kpoints={self.num_kpoints},",
            f"  num_orbitals={self.num_orbitals},",
            f"  num_wann={self.num_wann},"
        ]
        
        if self.outer_window is not None:
            lines.append(f"  outer_window={self.outer_window},")
        if self.e_fermi is not None:
            lines.append(f"  e_fermi={self.e_fermi:.4f},")
        if self.band_analysis is not None:
            lines.append(f"  frozen_bands={self.band_analysis.num_wann},")
        
        lines.append(")")
        
        return "\n".join(lines)