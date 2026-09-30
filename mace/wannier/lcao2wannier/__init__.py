# Copyright (c) 2025 William Comaskey. MIT License, see LICENSE in this directory.
# Vendored into MACE from lcao2wannier 1.0.0; local changes are listed in VENDORED.md.
"""Canonical Python API for CRYSTAL/LCAO to Wannier90 conversion.

Public objects are loaded on first access. This keeps ``lcao2wannier --help``
and ``--version`` independent of NumPy/SciPy while preserving the established
top-level Python API.
"""

from __future__ import annotations

from importlib import import_module

__version__ = "1.0.0"
__author__ = "William Comaskey"

_EXPORTS = {
    "Wannier90Engine": "engine",
    "parse_overlap_and_fock_matrices": "parser",
    "create_spin_block_matrices": "parser",
    "create_nonsoc_full_matrices": "parser",
    "fill_raw_matrix": "parser",
    "is_hermitian": "parser",
    "parse_calculation_parameters": "parser",
    "parse_atomic_basis_info": "parser",
    "parse_orbital_types": "parser",
    "CalculationParameters": "parser",
    "AtomicBasisInfo": "parser",
    "prepare_real_space_matrices": "utils",
    "organize_matrices_by_lattice_vector": "utils",
    "get_basis_size": "utils",
    "verify_matrix_symmetry": "utils",
    "check_matrix_consistency": "utils",
    "print_matrix_summary": "utils",
    "print_calculation_info": "utils",
    "generate_kpoint_grid": "kpoints",
    "generate_neighbor_list": "kpoints",
    "kpoint_index_to_grid": "kpoints",
    "grid_to_kpoint_index": "kpoints",
    "fourier_transform_to_kspace": "fourier",
    "inverse_fourier_transform": "fourier",
    "compute_phase_factors": "fourier",
    "StackedMatrices": "fourier",
    "fourier_all_kpoints": "fourier",
    "fourier_transform_vectorized": "fourier",
    "stack_real_space_matrices": "fourier",
    "solve_generalized_eigenvalue_problem": "solver",
    "solve_kpoint": "solver",
    "solve_all_kpoints_sequential": "solver",
    "solve_all_kpoints_parallel": "solver",
    "solve_all_kpoints_batched": "solver",
    "verify_real_space_symmetry": "verification",
    "verify_hermiticity": "verification",
    "verify_orthonormality": "verification",
    "verify_eigenvalue_sorting": "verification",
    "verify_energy_range": "verification",
    "run_all_verifications": "verification",
    "write_eig_file": "wannier90",
    "write_amn_file": "wannier90",
    "write_amn_file_pdwf": "wannier90",
    "write_mmn_file": "wannier90",
    "write_wannier90_files": "wannier90",
    "compute_mmn_matrix": "wannier90",
    "compute_mmn_direct": "wannier90",
    "compute_mmn_lowdin": "wannier90",
    "precompute_lowdin_eigenvectors": "wannier90",
    "unitarize_mmn": "wannier90",
    "read_nnkp_neighbors": "kpoints",
    "write_win_file": "win_file",
    "create_win_config_from_engine": "win_file",
    "parse_atoms_from_crystal_output": "win_file",
    "Wannier90WinConfig": "win_file",
    "KPATH_HEXAGONAL_2D": "win_file",
    "KPATH_HEXAGONAL_3D": "win_file",
    "KPATH_SQUARE_2D": "win_file",
    "KPATH_FCC": "win_file",
    "KPATH_BCC": "win_file",
    "KPATH_SIMPLE_CUBIC": "win_file",
    "run_band_structure": "band_plot",
    "read_w90_band_outputs": "band_plot",
    "compute_band_structure": "band_plot",
    "plot_band_structure": "band_plot",
    "text_band_summary": "band_plot",
    "generate_kpath": "band_plot",
    "detect_lattice_type": "band_plot",
    "kpath_from_win_format": "band_plot",
    "get_kpath_for_lattice": "band_plot",
    "parse_custom_kpath": "band_plot",
    "compute_path_projectability": "band_plot",
    "KPathSpec": "band_plot",
    "KPathResult": "band_plot",
    "BandStructureData": "band_plot",
    "PlotConfig": "band_plot",
    "estimate_fermi_energy": "band_selection",
    "compute_fermi_level": "band_selection",
    "analyze_band_window": "band_selection",
    "print_band_analysis": "band_selection",
    "check_frozen_continuity": "band_selection",
    "validate_fermi_coverage": "band_selection",
    "select_projection_orbitals": "band_selection",
    "scdm_select_projections": "band_selection",
    "compute_subspace_projections": "band_selection",
    "suggest_optimal_window": "band_selection",
    "BandWindowResult": "band_selection",
    "OrbitalSelectionResult": "band_selection",
    "compute_band_projections": "orbital_analysis",
    "compute_band_character": "orbital_analysis",
    "identify_dominant_character": "orbital_analysis",
    "analyze_all_bands_character": "orbital_analysis",
    "format_band_character_table": "orbital_analysis",
    "BandCharacter": "orbital_analysis",
    "compute_band_projectability": "projectability",
    "select_bands_by_projectability": "projectability",
    "smart_select_bands": "projectability",
    "ProjectabilityResult": "projectability",
    "SmartSelectionResult": "projectability",
    "enforce_hermiticity": "postprocess",
    "enforce_time_reversal": "postprocess",
    "validate_overlap_conditioning": "conditioning",
    "OverlapConditioningResult": "conditioning",
    "OverlapConditioningError": "conditioning",
    "get_valence_l": "valence_config",
    "get_num_target_orbitals": "valence_config",
    "compute_num_wann": "valence_config",
    "build_target_mask": "valence_config",
    "summarize_config": "valence_config",
    "VALENCE_CONFIG": "valence_config",
    "ELEMENT_SYMBOLS": "valence_config",
    "ELEMENT_Z": "valence_config",
    "parse_basis_shells": "basis_parser",
    "get_atom_list": "basis_parser",
    "ShellInfo": "basis_parser",
    "compute_lowdin_projectability": "lcao_pdwf",
    "compute_matrix_sqrt": "lcao_pdwf",
    "classify_bands": "lcao_pdwf",
    "determine_windows": "lcao_pdwf",
    "check_frozen_interlopers": "lcao_pdwf",
    "check_band_count": "lcao_pdwf",
    "print_pdwf_summary": "lcao_pdwf",
    "ClassificationParams": "lcao_pdwf",
    "BandClassification": "lcao_pdwf",
    "WindowParameters": "lcao_pdwf",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module_name}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
