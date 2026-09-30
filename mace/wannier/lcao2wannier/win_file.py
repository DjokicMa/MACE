# Copyright (c) 2025 William Comaskey. MIT License, see LICENSE in this directory.
# Vendored into MACE from lcao2wannier 1.0.0; local changes are listed in VENDORED.md.
"""
Wannier90 .win File Writer

Generates the main Wannier90 input file with all necessary parameters
for running wannier90.x after lcao2wannier conversion.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field


@dataclass
class Wannier90WinConfig:
    """
    Configuration for Wannier90 .win file generation.

    Attributes
    ----------
    num_wann : int
        Number of Wannier functions
    num_bands : int
        Number of bands (can be > num_wann if using disentanglement)
    mp_grid : tuple of 3 ints
        Monkhorst-Pack grid dimensions
    lattice_vectors : ndarray
        3x3 array of lattice vectors in Angstrom (rows are vectors)
    kpoints : ndarray
        Array of k-points in fractional coordinates, shape (num_kpts, 3)

    # Optional: Energy windows (eV)
    dis_win_min : float, optional
        Bottom of outer energy window
    dis_win_max : float, optional
        Top of outer energy window
    dis_froz_min : float, optional
        Bottom of frozen (inner) energy window
    dis_froz_max : float, optional
        Top of frozen (inner) energy window

    # Optional: Atoms and projections
    atoms : list of tuples, optional
        List of (symbol, position) where position is fractional coords
    projections : list of str, optional
        Projection specifications (e.g., ["Bi:sp3", "Bi:p"])

    # Optional: Control parameters
    num_iter : int
        Maximum number of iterations for Wannierization
    dis_num_iter : int
        Maximum iterations for disentanglement
    num_print_cycles : int
        Print frequency
    write_hr : bool
        Write Hamiltonian in WF basis
    write_xyz : bool
        Write WF centers in xyz format
    write_r2mn : bool
        Write r^2_mn matrix elements
    bands_plot : bool
        Calculate band structure
    wannier_plot : bool
        Plot Wannier functions
    spinors : bool
        System has spinor wavefunctions (SOC)
    use_bloch_phases : bool
        Use Bloch-like phases (for LCAO inputs)

    Examples
    --------
    >>> import numpy as np
    >>> config = Wannier90WinConfig(
    ...     num_wann=10,
    ...     num_bands=10,
    ...     mp_grid=(8, 8, 8),
    ...     lattice_vectors=np.eye(3) * 5.0,
    ...     kpoints=np.random.rand(512, 3),
    ...     spinors=True
    ... )
    """
    # Required parameters
    num_wann: int
    num_bands: int
    mp_grid: Tuple[int, int, int]
    lattice_vectors: np.ndarray
    kpoints: np.ndarray

    # Energy windows (optional, only needed if num_bands > num_wann)
    dis_win_min: Optional[float] = None
    dis_win_max: Optional[float] = None
    dis_froz_min: Optional[float] = None
    dis_froz_max: Optional[float] = None

    # Projectability disentanglement (Wannier90 3.x, introduced with PDWF;
    # Qiao, Pizzi & Marzari, npj Comput. Mater. 9, 208 (2023)). When
    # dis_froz_proj is True, Wannier90 freezes states per (k, band) from the
    # .amn projectability p_nk = sum_m |A_mn|^2 (frozen if p >= dis_proj_max,
    # excluded if p < dis_proj_min) instead of using dis_froz_min/max, and
    # the energy windows become optional (Wannier90 defaults dis_win to the
    # full band range). Used by the `--baseline dis-froz-proj` emission mode.
    dis_froz_proj: bool = False
    dis_proj_min: Optional[float] = None
    dis_proj_max: Optional[float] = None

    # Atoms and projections (optional)
    atoms: Optional[List[Tuple[str, np.ndarray]]] = None
    projections: Optional[List[str]] = None

    # Control parameters with sensible defaults
    num_iter: int = 500
    dis_num_iter: int = 1000
    num_print_cycles: int = 10
    conv_tol: float = 1.0e-10
    conv_window: int = 3

    # Output options
    write_hr: bool = True
    write_tb: bool = False
    write_xyz: bool = True
    write_r2mn: bool = False
    bands_plot: bool = False
    wannier_plot: bool = False
    hr_cutoff: Optional[float] = None
    dist_cutoff: Optional[float] = None

    # System properties
    spinors: bool = False
    fermi_energy: Optional[float] = None
    use_bloch_phases: bool = False
    guiding_centres: bool = False
    site_symmetry: bool = False

    # Band structure path (optional)
    kpoint_path: Optional[List[Tuple[str, np.ndarray]]] = None

    # Additional parameters (for advanced users)
    additional_keywords: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.num_wann > self.num_bands:
            raise ValueError(
                f"num_wann ({self.num_wann}) cannot exceed num_bands ({self.num_bands})"
            )

        if self.lattice_vectors.shape != (3, 3):
            raise ValueError(
                f"lattice_vectors must be shape (3, 3), got {self.lattice_vectors.shape}"
            )

        if self.kpoints.ndim != 2 or self.kpoints.shape[1] != 3:
            raise ValueError(
                f"kpoints must be shape (num_kpts, 3), got {self.kpoints.shape}"
            )

        num_kpts_expected = np.prod(self.mp_grid)
        if len(self.kpoints) != num_kpts_expected:
            raise ValueError(
                f"Number of k-points ({len(self.kpoints)}) does not match "
                f"mp_grid product ({num_kpts_expected})"
            )

        # Validate energy windows
        if self.num_bands > self.num_wann:
            # Disentanglement required
            if self.dis_win_min is None or self.dis_win_max is None:
                # With projectability freezing the windows are optional
                # (Wannier90 defaults dis_win to the full band range).
                if not self.dis_froz_proj:
                    raise ValueError(
                        "dis_win_min and dis_win_max must be specified when "
                        "num_bands > num_wann (disentanglement mode)"
                    )
            elif self.dis_win_min >= self.dis_win_max:
                raise ValueError(
                    f"dis_win_min ({self.dis_win_min}) must be < "
                    f"dis_win_max ({self.dis_win_max})"
                )
            if self.dis_froz_min is not None and self.dis_froz_max is not None:
                if self.dis_froz_min >= self.dis_froz_max:
                    raise ValueError(
                        f"dis_froz_min ({self.dis_froz_min}) must be < "
                        f"dis_froz_max ({self.dis_froz_max})"
                    )
                if (self.dis_win_min is not None and
                    self.dis_win_max is not None and
                    (self.dis_froz_min < self.dis_win_min or
                     self.dis_froz_max > self.dis_win_max)):
                    raise ValueError(
                        "Frozen window must be within outer window"
                    )


def write_win_file(
    seedname: str,
    config: Wannier90WinConfig,
    comment: str = "Generated by lcao2wannier package",
    verbose: bool = True
) -> str:
    """
    Write a Wannier90 .win input file.

    Parameters
    ----------
    seedname : str
        Base name for output file (will write seedname.win)
    config : Wannier90WinConfig
        Configuration dataclass with all parameters
    comment : str, optional
        Comment to include at top of file
    verbose : bool, optional
        Print confirmation message

    Returns
    -------
    str
        Path to written file

    Examples
    --------
    >>> import numpy as np
    >>> config = Wannier90WinConfig(
    ...     num_wann=10,
    ...     num_bands=10,
    ...     mp_grid=(4, 4, 4),
    ...     lattice_vectors=np.eye(3) * 5.0,
    ...     kpoints=np.random.rand(64, 3)
    ... )
    >>> filepath = write_win_file('material', config)
    """
    lines = []

    # Header
    lines.append(f"! {comment}")
    lines.append(f"! Wannier90 input file for seedname: {seedname}")
    lines.append(f"!")
    lines.append("")

    # Basic parameters
    lines.append("!====================")
    lines.append("! SYSTEM PARAMETERS")
    lines.append("!====================")
    lines.append(f"num_wann = {config.num_wann}")
    lines.append(f"num_bands = {config.num_bands}")
    lines.append("")

    # System properties
    if config.spinors:
        lines.append("! Spin-orbit coupling included")
        lines.append("spinors = .true.")
        lines.append("")

    if config.fermi_energy is not None:
        lines.append(f"! Fermi energy (for relative energy windows)")
        lines.append(f"fermi_energy = {config.fermi_energy:.6f}")
        lines.append("")

    if config.use_bloch_phases:
        lines.append("! Use Bloch-like phases (for LCAO inputs)")
        lines.append("use_bloch_phases = .true.")
        lines.append("")

    # Disentanglement (only if num_bands > num_wann)
    if config.num_bands > config.num_wann:
        lines.append("!====================")
        lines.append("! DISENTANGLEMENT")
        lines.append("!====================")
        if config.dis_win_min is not None and config.dis_win_max is not None:
            lines.append(f"dis_win_min = {config.dis_win_min:.8f}")
            lines.append(f"dis_win_max = {config.dis_win_max:.8f}")
        if config.dis_froz_min is not None and config.dis_froz_max is not None:
            lines.append(f"dis_froz_min = {config.dis_froz_min:.8f}")
            lines.append(f"dis_froz_max = {config.dis_froz_max:.8f}")
        if config.dis_froz_proj:
            lines.append("! Per-state projectability freezing from the .amn")
            lines.append("! (PDWF; Qiao et al., npj Comput. Mater. 9, 208 (2023))")
            lines.append("dis_froz_proj = .true.")
            if config.dis_proj_min is not None:
                lines.append(f"dis_proj_min = {config.dis_proj_min:.4f}")
            if config.dis_proj_max is not None:
                lines.append(f"dis_proj_max = {config.dis_proj_max:.4f}")
        lines.append(f"dis_num_iter = {config.dis_num_iter}")
        lines.append(f"dis_conv_tol = {config.conv_tol:.2e}")
        lines.append(f"dis_conv_window = {config.conv_window}")
        lines.append("")

    # Wannierization
    lines.append("!====================")
    lines.append("! WANNIERIZATION")
    lines.append("!====================")
    lines.append(f"num_iter = {config.num_iter}")
    lines.append(f"conv_tol = {config.conv_tol:.2e}")
    lines.append(f"conv_window = {config.conv_window}")
    lines.append(f"num_print_cycles = {config.num_print_cycles}")
    if config.guiding_centres:
        lines.append("guiding_centres = .true.")
    if config.site_symmetry:
        lines.append("site_symmetry = .true.")
    lines.append("")

    # Output options
    lines.append("!====================")
    lines.append("! OUTPUT OPTIONS")
    lines.append("!====================")
    lines.append(f"write_hr = {'T' if config.write_hr else 'F'}")
    lines.append(f"write_xyz = {'T' if config.write_xyz else 'F'}")
    lines.append(f"write_r2mn = {'T' if config.write_r2mn else 'F'}")
    if config.write_tb:
        lines.append("write_tb = T")
    if config.hr_cutoff is not None:
        lines.append(f"hr_cutoff = {config.hr_cutoff:.6f}")
    if config.dist_cutoff is not None:
        lines.append(f"dist_cutoff = {config.dist_cutoff:.6f}")
    if config.bands_plot:
        lines.append("bands_plot = T")
    if config.wannier_plot:
        lines.append("wannier_plot = T")
        lines.append("wannier_plot_supercell = 3")
    lines.append("")

    # Unit cell
    lines.append("!====================")
    lines.append("! UNIT CELL")
    lines.append("!====================")
    lines.append("begin unit_cell_cart")
    lines.append("Ang")
    for i in range(3):
        v = config.lattice_vectors[i]
        lines.append(f"  {v[0]:18.12f}  {v[1]:18.12f}  {v[2]:18.12f}")
    lines.append("end unit_cell_cart")
    lines.append("")

    # Atoms (if provided)
    if config.atoms is not None and len(config.atoms) > 0:
        lines.append("!====================")
        lines.append("! ATOMIC POSITIONS")
        lines.append("!====================")
        lines.append("begin atoms_frac")
        for symbol, pos in config.atoms:
            lines.append(f"  {symbol:4s}  {pos[0]:18.12f}  {pos[1]:18.12f}  {pos[2]:18.12f}")
        lines.append("end atoms_frac")
        lines.append("")

    # Projections
    lines.append("!====================")
    lines.append("! PROJECTIONS")
    lines.append("!====================")
    lines.append("begin projections")
    if config.projections is not None and len(config.projections) > 0:
        for proj in config.projections:
            lines.append(f"  {proj}")
    else:
        # Use random projections - Wannier90 will rely on .amn file
        lines.append("  ! Using LCAO basis projections from .amn file")
        lines.append("  random")
    lines.append("end projections")
    lines.append("")

    # K-point grid
    lines.append("!====================")
    lines.append("! K-POINT MESH")
    lines.append("!====================")
    lines.append(f"mp_grid = {config.mp_grid[0]} {config.mp_grid[1]} {config.mp_grid[2]}")
    lines.append("")
    lines.append("begin kpoints")
    for kpt in config.kpoints:
        lines.append(f"  {kpt[0]:18.12f}  {kpt[1]:18.12f}  {kpt[2]:18.12f}")
    lines.append("end kpoints")
    lines.append("")

    # Band structure path (if provided)
    if config.kpoint_path is not None and len(config.kpoint_path) > 0:
        lines.append("!====================")
        lines.append("! BAND STRUCTURE PATH")
        lines.append("!====================")
        lines.append("! Note: add 'restart = plot' after initial wannierization")
        lines.append("! to generate band structure interpolation")
        lines.append("bands_num_points = 100")
        lines.append("")
        lines.append("begin kpoint_path")
        # Wannier90 format: label1 x1 y1 z1 label2 x2 y2 z2
        for i in range(0, len(config.kpoint_path) - 1, 2):
            if i + 1 < len(config.kpoint_path):
                label1, pos1 = config.kpoint_path[i]
                label2, pos2 = config.kpoint_path[i + 1]
                lines.append(
                    f"  {label1:3s} {pos1[0]:8.4f} {pos1[1]:8.4f} {pos1[2]:8.4f}  "
                    f"{label2:3s} {pos2[0]:8.4f} {pos2[1]:8.4f} {pos2[2]:8.4f}"
                )
        lines.append("end kpoint_path")
        lines.append("")

    # Additional keywords
    if config.additional_keywords:
        lines.append("!====================")
        lines.append("! ADDITIONAL PARAMETERS")
        lines.append("!====================")
        for key, value in config.additional_keywords.items():
            if isinstance(value, bool):
                lines.append(f"{key} = {'T' if value else 'F'}")
            else:
                lines.append(f"{key} = {value}")
        lines.append("")

    # Write to file
    filepath = f"{seedname}.win"
    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))

    if verbose:
        print(f"  ✓ {filepath}: Wannier90 input parameters")

    return filepath


def create_win_config_from_engine(
    engine,
    atoms: Optional[List[Tuple[str, np.ndarray]]] = None,
    projections: Optional[List[str]] = None,
    spinors: bool = True,
    write_hr: bool = True,
    bands_plot: bool = False,
    kpoint_path: Optional[List[Tuple[str, np.ndarray]]] = None,
    use_bloch_phases: bool = False,
    num_iter: Optional[int] = None,
    **kwargs
) -> Wannier90WinConfig:
    """
    Create a Wannier90WinConfig from an initialized Wannier90Engine.

    This function intelligently extracts all necessary parameters from
    a Wannier90Engine instance and creates a properly configured
    Wannier90WinConfig object.

    Parameters
    ----------
    engine : Wannier90Engine
        Initialized engine with solved eigenvalue problems
    atoms : list of tuples, optional
        Atom specifications as [(symbol, frac_coords), ...]
        If None, atoms section will be omitted from .win file
    projections : list of str, optional
        Projection specifications for Wannier90
        If None, will use 'random' (Wannier90 reads from .amn file)
    spinors : bool, optional
        Whether system has SOC (default: True)
    write_hr : bool, optional
        Write Hamiltonian in WF basis (default: True)
    bands_plot : bool, optional
        Enable band structure plotting (default: False)
    kpoint_path : list of tuples, optional
        High-symmetry path for band plotting as [(label, coords), ...]
    use_bloch_phases : bool, optional
        Use Bloch-like phases (recommended for LCAO, default: False)
    num_iter : int, optional
        Override num_iter in the .win file. If None, uses default (500).
        Set to 0 for direct LCAO mapping (skip spread minimization).
    **kwargs
        Additional keyword arguments passed to Wannier90WinConfig

    Returns
    -------
    Wannier90WinConfig
        Configuration ready for write_win_file()

    Raises
    ------
    RuntimeError
        If engine has not solved eigenvalue problems yet

    Examples
    --------
    >>> from lcao2wannier import Wannier90Engine
    >>> engine = Wannier90Engine(...)
    >>> engine.solve_all_kpoints()
    >>> config = create_win_config_from_engine(
    ...     engine,
    ...     spinors=True,
    ...     write_hr=True
    ... )
    >>> write_win_file('material', config)
    """
    # Check that engine has been solved
    if not engine.eigenvalues_list:
        raise RuntimeError(
            "Engine has not solved eigenvalue problems. "
            "Call engine.solve_all_kpoints() first."
        )

    # Determine num_wann and num_bands
    num_wann = engine.num_wann

    if hasattr(engine, 'selected_band_indices') and engine.selected_band_indices is not None:
        band_indices = engine.selected_band_indices
    else:
        band_indices = np.arange(num_wann)

    # Check if engine has disentanglement info from smart selector
    if hasattr(engine, '_num_bands_for_win') and engine._num_bands_for_win is not None:
        num_bands = engine._num_bands_for_win
    else:
        num_bands = num_wann

    # Determine energy windows (only if disentanglement is needed)
    dis_win_min = None
    dis_win_max = None
    dis_froz_min = None
    dis_froz_max = None

    # Use disentanglement windows from smart selector if available
    if hasattr(engine, '_dis_win') and engine._dis_win is not None:
        dis_win_min, dis_win_max = engine._dis_win
    if hasattr(engine, '_dis_froz') and engine._dis_froz is not None:
        dis_froz_min, dis_froz_max = engine._dis_froz

    # Override defaults with any provided kwargs
    config_params = {
        'num_wann': num_wann,
        'num_bands': num_bands,
        'mp_grid': engine.k_grid,
        'lattice_vectors': engine.lattice_vectors,
        'kpoints': engine.kpoints,
        'atoms': atoms,
        'projections': projections,
        'spinors': spinors,
        'fermi_energy': getattr(engine, 'e_fermi', None),
        'write_hr': write_hr,
        'bands_plot': bands_plot,
        'kpoint_path': kpoint_path,
        'use_bloch_phases': use_bloch_phases,
        'dis_win_min': dis_win_min,
        'dis_win_max': dis_win_max,
        'dis_froz_min': dis_froz_min,
        'dis_froz_max': dis_froz_max,
    }

    # Override num_iter if specified (e.g., 0 for direct method)
    if num_iter is not None:
        config_params['num_iter'] = num_iter

    # Update with user-provided kwargs
    config_params.update(kwargs)

    config = Wannier90WinConfig(**config_params)

    return config


def parse_atoms_from_crystal_output(lines: List[str]) -> Tuple[List[Tuple[str, np.ndarray]], np.ndarray]:
    """
    Parse atomic positions from CRYSTAL output file.

    Extracts both lattice vectors and atomic positions in fractional coordinates.

    Parameters
    ----------
    lines : list of str
        Lines from CRYSTAL output file

    Returns
    -------
    atoms : list of tuples
        [(symbol, fractional_coords), ...] for each atom
    lattice_vectors : ndarray
        3x3 array of lattice vectors in Angstrom

    Examples
    --------
    >>> with open('crystal.out') as f:
    ...     lines = f.readlines()
    >>> atoms, lattice = parse_atoms_from_crystal_output(lines)
    """
    atoms = []
    lattice_vectors = None

    # Find lattice vectors
    for i, line in enumerate(lines):
        if 'DIRECT LATTICE VECTOR COMPONENTS' in line and 'ANGSTROM' in line:
            vectors = []
            for j in range(1, 4):
                if i + j < len(lines):
                    parts = lines[i + j].split()
                    if len(parts) >= 3:
                        try:
                            # Handle Fortran D notation
                            vec = [float(parts[k].replace('D', 'E').replace('d', 'e'))
                                   for k in range(3)]
                            vectors.append(vec)
                        except (ValueError, IndexError):
                            pass
            if len(vectors) == 3:
                lattice_vectors = np.array(vectors)
                break

    # Element symbols mapping (atomic number -> symbol)
    element_symbols = {
        1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O',
        9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P',
        16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti',
        23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu',
        30: 'Zn', 31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr',
        37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr', 41: 'Nb', 42: 'Mo', 43: 'Tc',
        44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn',
        51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 55: 'Cs', 56: 'Ba', 57: 'La',
        58: 'Ce', 59: 'Pr', 60: 'Nd', 61: 'Pm', 62: 'Sm', 63: 'Eu', 64: 'Gd',
        65: 'Tb', 66: 'Dy', 67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb', 71: 'Lu',
        72: 'Hf', 73: 'Ta', 74: 'W', 75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt',
        79: 'Au', 80: 'Hg', 81: 'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At',
        86: 'Rn', 87: 'Fr', 88: 'Ra', 89: 'Ac', 90: 'Th', 91: 'Pa', 92: 'U',
    }

    # Find atomic positions
    # Look for the geometry table
    in_atom_section = False
    for i, line in enumerate(lines):
        if 'ATOM' in line and 'N.AT.' in line and 'X(A)' in line and 'Y(A)' in line:
            in_atom_section = True
            continue

        if in_atom_section:
            # Skip decorative asterisk lines (they appear before/after atom data)
            if '****' in line:
                # Don't break yet - check if we have atoms first
                if atoms:
                    # We already parsed atoms, asterisks mark end
                    break
                else:
                    # Asterisks before atom data, skip and continue
                    continue

            # End markers
            if 'INFORMATION' in line:
                if atoms:
                    break
                in_atom_section = False
                continue

            # Skip empty lines
            if not line.strip():
                continue

            # Parse atom line
            # Format: ATOM_INDEX  ATOMIC_NUM  SYMBOL  ...  X(A)  Y(A)  Z(A)  ...
            parts = line.split()
            if len(parts) >= 7:
                try:
                    # Normalize symbol: CRYSTAL uses ALL-CAPS (TE, SN, BI)
                    # Convert to standard form (Te, Sn, Bi)
                    symbol = parts[2].capitalize()
                    # Coordinates are typically in columns 4, 5, 6 (0-indexed)
                    x = float(parts[4])
                    y = float(parts[5])
                    z = float(parts[6])

                    cart_pos = np.array([x, y, z])

                    # Convert to fractional coordinates
                    if lattice_vectors is not None:
                        frac_pos = np.linalg.solve(lattice_vectors.T, cart_pos)
                    else:
                        frac_pos = cart_pos

                    atoms.append((symbol, frac_pos))
                except (ValueError, IndexError):
                    continue

    if lattice_vectors is None:
        lattice_vectors = np.eye(3) * 5.0  # Default if not found

    return atoms, lattice_vectors


# Standard high-symmetry k-point paths for common lattices
KPATH_HEXAGONAL_2D = [
    ('G', np.array([0.0, 0.0, 0.0])),
    ('M', np.array([0.5, 0.0, 0.0])),
    ('M', np.array([0.5, 0.0, 0.0])),
    ('K', np.array([1/3, 1/3, 0.0])),
    ('K', np.array([1/3, 1/3, 0.0])),
    ('G', np.array([0.0, 0.0, 0.0])),
]

KPATH_HEXAGONAL_3D = [
    ('G', np.array([0.0, 0.0, 0.0])),
    ('M', np.array([0.5, 0.0, 0.0])),
    ('M', np.array([0.5, 0.0, 0.0])),
    ('K', np.array([1/3, 1/3, 0.0])),
    ('K', np.array([1/3, 1/3, 0.0])),
    ('G', np.array([0.0, 0.0, 0.0])),
    ('G', np.array([0.0, 0.0, 0.0])),
    ('A', np.array([0.0, 0.0, 0.5])),
    ('A', np.array([0.0, 0.0, 0.5])),
    ('L', np.array([0.5, 0.0, 0.5])),
    ('L', np.array([0.5, 0.0, 0.5])),
    ('H', np.array([1/3, 1/3, 0.5])),
    ('H', np.array([1/3, 1/3, 0.5])),
    ('A', np.array([0.0, 0.0, 0.5])),
]

KPATH_SQUARE_2D = [
    ('G', np.array([0.0, 0.0, 0.0])),
    ('X', np.array([0.5, 0.0, 0.0])),
    ('X', np.array([0.5, 0.0, 0.0])),
    ('M', np.array([0.5, 0.5, 0.0])),
    ('M', np.array([0.5, 0.5, 0.0])),
    ('G', np.array([0.0, 0.0, 0.0])),
]

KPATH_FCC = [
    ('G', np.array([0.0, 0.0, 0.0])),
    ('X', np.array([0.5, 0.0, 0.5])),
    ('X', np.array([0.5, 0.0, 0.5])),
    ('W', np.array([0.5, 0.25, 0.75])),
    ('W', np.array([0.5, 0.25, 0.75])),
    ('K', np.array([0.375, 0.375, 0.75])),
    ('K', np.array([0.375, 0.375, 0.75])),
    ('G', np.array([0.0, 0.0, 0.0])),
    ('G', np.array([0.0, 0.0, 0.0])),
    ('L', np.array([0.5, 0.5, 0.5])),
    ('L', np.array([0.5, 0.5, 0.5])),
    ('U', np.array([0.625, 0.25, 0.625])),
    ('U', np.array([0.625, 0.25, 0.625])),
    ('W', np.array([0.5, 0.25, 0.75])),
    ('W', np.array([0.5, 0.25, 0.75])),
    ('L', np.array([0.5, 0.5, 0.5])),
    ('L', np.array([0.5, 0.5, 0.5])),
    ('K', np.array([0.375, 0.375, 0.75])),
]

KPATH_BCC = [
    ('G', np.array([0.0, 0.0, 0.0])),
    ('H', np.array([0.5, -0.5, 0.5])),
    ('H', np.array([0.5, -0.5, 0.5])),
    ('N', np.array([0.0, 0.0, 0.5])),
    ('N', np.array([0.0, 0.0, 0.5])),
    ('G', np.array([0.0, 0.0, 0.0])),
    ('G', np.array([0.0, 0.0, 0.0])),
    ('P', np.array([0.25, 0.25, 0.25])),
    ('P', np.array([0.25, 0.25, 0.25])),
    ('H', np.array([0.5, -0.5, 0.5])),
]

KPATH_SIMPLE_CUBIC = [
    ('G', np.array([0.0, 0.0, 0.0])),
    ('X', np.array([0.5, 0.0, 0.0])),
    ('X', np.array([0.5, 0.0, 0.0])),
    ('M', np.array([0.5, 0.5, 0.0])),
    ('M', np.array([0.5, 0.5, 0.0])),
    ('G', np.array([0.0, 0.0, 0.0])),
    ('G', np.array([0.0, 0.0, 0.0])),
    ('R', np.array([0.5, 0.5, 0.5])),
    ('R', np.array([0.5, 0.5, 0.5])),
    ('X', np.array([0.5, 0.0, 0.0])),
    ('X', np.array([0.5, 0.0, 0.0])),
    ('M', np.array([0.5, 0.5, 0.0])),
    ('M', np.array([0.5, 0.5, 0.0])),
    ('R', np.array([0.5, 0.5, 0.5])),
]


def read_win_parameter(filename: str, parameter: str) -> Optional[Any]:
    """
    Read a single parameter from a .win file.

    Parameters
    ----------
    filename : str
        Path to .win file
    parameter : str
        Parameter name (e.g., 'num_bands', 'num_wann')

    Returns
    -------
    value : int, float, bool, str, or None
        Parameter value, or None if not found

    Examples
    --------
    >>> num_bands = read_win_parameter('material.win', 'num_bands')
    >>> print(num_bands)
    16
    """
    import re

    with open(filename, 'r') as f:
        for line in f:
            # Skip comments and empty lines
            if line.strip().startswith('!') or not line.strip():
                continue

            # Match "parameter = value"
            match = re.match(rf'^\s*{parameter}\s*=\s*(.+)', line, re.IGNORECASE)
            if match:
                value_str = match.group(1).strip()

                # Try to parse as int
                try:
                    return int(value_str)
                except ValueError:
                    pass

                # Try to parse as float
                try:
                    return float(value_str)
                except ValueError:
                    pass

                # Check for boolean
                if value_str.upper() in ['.TRUE.', 'T', 'TRUE']:
                    return True
                if value_str.upper() in ['.FALSE.', 'F', 'FALSE']:
                    return False

                # Return as string
                return value_str

    return None


def update_win_parameter(filename: str, parameter: str, value: Any) -> None:
    """
    Update a single parameter in a .win file.

    Parameters
    ----------
    filename : str
        Path to .win file
    parameter : str
        Parameter name (e.g., 'num_bands')
    value : int, float, bool, or str
        New value

    Examples
    --------
    >>> update_win_parameter('material.win', 'num_bands', 14)
    """
    import re

    with open(filename, 'r') as f:
        lines = f.readlines()

    # Format value
    if isinstance(value, bool):
        value_str = '.true.' if value else '.false.'
    else:
        value_str = str(value)

    # Find and update parameter
    updated = False
    for i, line in enumerate(lines):
        # Skip comments
        if line.strip().startswith('!'):
            continue

        # Match "parameter = old_value"
        match = re.match(rf'^\s*{parameter}\s*=\s*(.+)', line, re.IGNORECASE)
        if match:
            # Preserve indentation
            indent = len(line) - len(line.lstrip())
            lines[i] = ' ' * indent + f'{parameter} = {value_str}\n'
            updated = True
            break

    if not updated:
        raise ValueError(f"Parameter '{parameter}' not found in {filename}")

    # Write back
    with open(filename, 'w') as f:
        f.writelines(lines)
