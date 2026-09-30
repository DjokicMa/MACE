# Copyright (c) 2025 William Comaskey. MIT License, see LICENSE in this directory.
# Vendored into MACE from lcao2wannier 1.0.0; local changes are listed in VENDORED.md.
"""
Parser Module for LCAO Output Files

This module contains functions to parse overlap and Fock matrices from
CRYSTAL/LCAO output files and create spin-block matrices using 
Robust Global Pair-Symmetry Construction.
"""

import numpy as np
import re
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

# ==============================
# Regular Expression Patterns
# ==============================

overlap_header_pattern = re.compile(
    r'^\s*OVERLAP MATRIX - CELL N\.\s+\d+\(\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*\)'
)
fock_header_pattern = re.compile(
    r'^\s*FOCK MATRIX \((REAL|IMAG) PART\) - CELL N\.\s+\d+\(\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*\)'
)
# Simple Fock matrix header (no REAL/IMAG split, used in some CRYSTAL formats)
fock_simple_header_pattern = re.compile(
    r'^\s*FOCK MATRIX - CELL N\.\s+\d+\(\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*\)'
)
spin_channel_pattern = re.compile(
    r'^\s*(ALPHA_ALPHA|ALPHA_BETA|BETA_ALPHA|BETA_BETA) ELECTRONS', re.IGNORECASE
)
# Simple spin channel header (ALPHA/BETA without compound labels)
spin_simple_pattern = re.compile(
    r'^\s*(ALPHA|BETA)\s+ELECTRONS', re.IGNORECASE
)
column_indices_pattern = re.compile(r'^\s*\d+(\s+\d+)*\s*$')
data_line_pattern = re.compile(r'^\s*(\d+)\s+(.+)$')
float_pattern = re.compile(
    r'[-+]?\d*\.\d+(?:[eEdD][-+]?\d+)?|[-+]?\d+(?:[eEdD][-+]?\d+)?'
)
direct_lattice_header_pattern = re.compile(
    r'^\s*DIRECT LATTICE VECTOR COMPONENTS \(ANGSTROM\)', re.IGNORECASE
)
vector_line_pattern = re.compile(
    r'^\s*' + r'\s+'.join([float_pattern.pattern] * 3) + r'\s*$'
)


# ==============================
# Calculation Parameters Dataclass
# ==============================

@dataclass
class CalculationParameters:
    """
    Parameters parsed from CRYSTAL/LCAO output file header.
    
    Attributes
    ----------
    fermi_energy : float or None
        Fermi energy in eV (converted from Hartree)
    fermi_energy_hartree : float or None
        Fermi energy in Hartree (raw value from file)
    num_electrons : int or None
        Number of electrons per cell
    k_grid : tuple of 3 ints or None
        Monkhorst-Pack k-point grid dimensions
    num_atoms : int or None
        Number of atoms per cell
    num_shells : int or None
        Number of shells
    num_ao : int or None
        Number of atomic orbitals
    total_energy : float or None
        Total energy in Hartree
    """
    fermi_energy: Optional[float] = None
    fermi_energy_hartree: Optional[float] = None
    num_electrons: Optional[int] = None
    k_grid: Optional[Tuple[int, int, int]] = None
    num_atoms: Optional[int] = None
    num_shells: Optional[int] = None
    num_ao: Optional[int] = None
    total_energy: Optional[float] = None
    has_soc: bool = False  # Spin-orbit coupling detected


def parse_calculation_parameters(lines: List[str]) -> CalculationParameters:
    """
    Parse calculation parameters from CRYSTAL/LCAO output file.
    
    Extracts key parameters from the file header including Fermi energy,
    electron count, k-grid, and other useful information.
    
    Parameters
    ----------
    lines : list of str
        Lines from the CRYSTAL output file
        
    Returns
    -------
    CalculationParameters
        Dataclass containing parsed parameters
        
    Examples
    --------
    >>> with open('crystal.out', 'r') as f:
    ...     lines = f.readlines()
    >>> params = parse_calculation_parameters(lines)
    >>> print(f"Fermi energy: {params.fermi_energy} eV")
    >>> print(f"K-grid: {params.k_grid}")
    """
    params = CalculationParameters()
    
    # Conversion factor
    HARTREE_TO_EV = 27.2114
    
    for line in lines:
        # Parse FERMI ENERGY (in Hartree)
        # Format: FERMI ENERGY              -0.137E+00
        if 'FERMI ENERGY' in line and params.fermi_energy is None:
            match = re.search(r'FERMI ENERGY\s+([-+]?\d*\.?\d+[EeDd]?[+-]?\d*)', line)
            if match:
                fermi_str = match.group(1).replace('D', 'E').replace('d', 'e')
                params.fermi_energy_hartree = float(fermi_str)
                params.fermi_energy = params.fermi_energy_hartree * HARTREE_TO_EV
        
        # Parse N. OF ELECTRONS PER CELL
        # Format: N. OF ELECTRONS PER CELL    46
        if 'N. OF ELECTRONS PER CELL' in line and params.num_electrons is None:
            match = re.search(r'N\. OF ELECTRONS PER CELL\s+(\d+)', line)
            if match:
                params.num_electrons = int(match.group(1))
        
        # Parse SHRINK. FACT.(MONKH.) k-grid
        # Format: SHRINK. FACT.(MONKH.)    15 15  1  SHRINKING FACTOR(GILAT NET)       15
        if 'SHRINK. FACT.(MONKH.)' in line and params.k_grid is None:
            match = re.search(r'SHRINK\. FACT\.\(MONKH\.\)\s+(\d+)\s+(\d+)\s+(\d+)', line)
            if match:
                params.k_grid = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
        
        # Parse N. OF ATOMS PER CELL
        # Format: N. OF ATOMS PER CELL         2
        if 'N. OF ATOMS PER CELL' in line and params.num_atoms is None:
            match = re.search(r'N\. OF ATOMS PER CELL\s+(\d+)', line)
            if match:
                params.num_atoms = int(match.group(1))
        
        # Parse NUMBER OF SHELLS
        # Format: NUMBER OF SHELLS            20
        if 'NUMBER OF SHELLS' in line and params.num_shells is None:
            match = re.search(r'NUMBER OF SHELLS\s+(\d+)', line)
            if match:
                params.num_shells = int(match.group(1))
        
        # Parse NUMBER OF AO
        # Format: NUMBER OF AO                56
        if 'NUMBER OF AO' in line and params.num_ao is None:
            match = re.search(r'NUMBER OF AO\s+(\d+)', line)
            if match:
                params.num_ao = int(match.group(1))
        
        # Parse TOTAL ENERGY
        # Format: TOTAL ENERGY -4.2943592973046E+02
        if 'TOTAL ENERGY' in line and params.total_energy is None:
            match = re.search(r'TOTAL ENERGY\s+([-+]?\d*\.?\d+[EeDd]?[+-]?\d*)', line)
            if match:
                energy_str = match.group(1).replace('D', 'E').replace('d', 'e')
                params.total_energy = float(energy_str)

        # Detect spin-orbit coupling from TWO-COMPONENT SCF
        # Format: "DENSITY MATRIX FROM A TWO-COMPONENT SCF"
        if 'TWO-COMPONENT' in line and 'SCF' in line:
            params.has_soc = True

    return params


@dataclass
class AtomicBasisInfo:
    """
    Information about atomic positions and basis function mapping.

    Attributes
    ----------
    atom_positions : ndarray
        Cartesian coordinates of atoms in Angstroms, shape (num_atoms, 3)
    atom_symbols : list of str
        Chemical symbols for each atom
    basis_atom_map : ndarray
        Maps each basis function index to its atom index, shape (num_basis,)
    orbital_types : dict, optional
        Maps orbital index (1-indexed) to orbital type ('s', 'p', 'd', 'f', 'g')
    num_atoms : int
        Total number of atoms
    num_basis : int
        Total number of basis functions
    """
    atom_positions: np.ndarray
    atom_symbols: List[str]
    basis_atom_map: np.ndarray
    orbital_types: Optional[Dict[int, str]] = None
    num_atoms: int = 0
    num_basis: int = 0


def parse_atomic_basis_info(lines: List[str]) -> AtomicBasisInfo:
    """
    Parse atomic positions and basis-to-atom mapping from CRYSTAL output.

    Extracts information from the "LOCAL ATOMIC FUNCTIONS BASIS SET" section,
    which contains atom positions in Bohr and the organization of basis functions.

    Parameters
    ----------
    lines : list of str
        Lines from the CRYSTAL output file

    Returns
    -------
    AtomicBasisInfo
        Dataclass containing atomic positions and basis mapping

    Notes
    -----
    The BASIS SET section has format:
    ```
    LOCAL ATOMIC FUNCTIONS BASIS SET
       ATOM   X(AU)   Y(AU)   Z(AU)  N. TYPE  EXPONENT  ...
       1 BI   2.342   0.000   1.698
                                    1 S
                                    ...
                              5-     7 P
                                    ...
       2 BI  -2.342  -0.000  -1.698
                                   29 S
                                    ...
    ```

    Examples
    --------
    >>> with open('crystal.out', 'r') as f:
    ...     lines = f.readlines()
    >>> info = parse_atomic_basis_info(lines)
    >>> print(f"Atom 0 position: {info.atom_positions[0]} Angstrom")
    >>> print(f"Basis function 10 belongs to atom {info.basis_atom_map[10]}")
    """
    # Conversion factor: Bohr to Angstrom
    BOHR_TO_ANGSTROM = 0.529177210903

    # Find the LOCAL ATOMIC FUNCTIONS BASIS SET section
    start_idx = None
    for i, line in enumerate(lines):
        if 'LOCAL ATOMIC FUNCTIONS BASIS SET' in line:
            start_idx = i + 1  # Start from next line (asterisks)
            break

    if start_idx is None:
        raise ValueError("Could not find 'LOCAL ATOMIC FUNCTIONS BASIS SET' section")

    atom_positions_bohr = []
    atom_symbols = []
    basis_ranges = []  # List of (atom_idx, first_basis, last_basis)
    current_atom = None

    # Pattern to match basis function ranges: "  1 S  " or "  5-     7 P  "
    basis_single_pattern = re.compile(r'^\s+(\d+)\s+([SPDFG])\s*$')
    basis_range_pattern = re.compile(r'^\s+(\d+)-\s+(\d+)\s+([SPDFG])\s*$')

    i = start_idx
    while i < len(lines):
        line = lines[i]

        # Check for end of section (usually OVERLAP MATRIX or empty line with asterisks)
        if 'OVERLAP MATRIX' in line or ('*****' in line and i > start_idx + 5):
            break

        # Try to match atom header by splitting
        # Format: "   1 BI   2.342   0.000   1.698"
        parts = line.split()
        if len(parts) >= 5:
            try:
                atom_idx_test = int(parts[0])
                # Check if this looks like an atom line (symbol is all letters)
                if parts[1].isalpha() and parts[1].isupper():
                    atom_idx = atom_idx_test - 1  # Convert to 0-based
                    # Normalize symbol: CRYSTAL uses ALL-CAPS (TE, SN, BI)
                    # Convert to standard form (Te, Sn, Bi)
                    symbol = parts[1].capitalize()
                    x, y, z = float(parts[2]), float(parts[3]), float(parts[4])

                    atom_positions_bohr.append([x, y, z])
                    atom_symbols.append(symbol)
                    current_atom = atom_idx
                    i += 1
                    continue
            except (ValueError, IndexError):
                pass

        # Try to match basis function single index
        match = basis_single_pattern.match(line)
        if match and current_atom is not None:
            basis_idx = int(match.group(1)) - 1  # Convert to 0-based
            basis_ranges.append((current_atom, basis_idx, basis_idx))
            i += 1
            continue

        # Try to match basis function range
        match = basis_range_pattern.match(line)
        if match and current_atom is not None:
            first_basis = int(match.group(1)) - 1  # Convert to 0-based
            last_basis = int(match.group(2)) - 1
            basis_ranges.append((current_atom, first_basis, last_basis))
            i += 1
            continue

        i += 1

    if not atom_positions_bohr:
        raise ValueError("No atomic positions found in BASIS SET section")

    # Convert positions to Angstrom
    atom_positions = np.array(atom_positions_bohr) * BOHR_TO_ANGSTROM

    # Determine total number of basis functions
    if not basis_ranges:
        raise ValueError("Could not determine number of basis functions from BASIS SET")

    # Check which atoms have explicit basis function listings
    atoms_with_basis = set(atom_idx for atom_idx, _, _ in basis_ranges)
    max_basis_index = max(end for _, _, end in basis_ranges)
    num_atoms = len(atom_symbols)

    # Identify atoms WITHOUT explicit basis function listings
    # Crystal23 omits symmetry-equivalent atoms from the basis section
    atoms_without_basis = [i for i in range(num_atoms) if i not in atoms_with_basis]

    # PER-SPECIES COUNT (preferred). CRYSTAL lists shells for the first atom of
    # each species with GLOBAL indices, so the span 1..max_basis_index ALREADY
    # covers every atom up to the last listed one -- the gaps between listed
    # atoms belong to the unlisted atoms of the same species. Adding a
    # replicated count for every missing atom therefore DOUBLE-COUNTS.
    #
    # Measured on Sc_all_center (89 atoms; C 36, H 24, B 12, O 12, Sc 5):
    # max_basis_index+1 = 1264 and extra_basis = 1324 gave num_basis = 2588,
    # against a true basis of 1424 -- which is exactly the per-species sum
    # 36*18 + 24*6 + 12*18 + 12*18 + 5*40, and exactly the S(k) dimension.
    species_count = {}
    for atom_idx in atoms_with_basis:
        rng = [(f, l) for a, f, l in basis_ranges if a == atom_idx]
        if rng:
            species_count[atom_symbols[atom_idx]] = (max(l for _, l in rng)
                                                     - min(f for f, _ in rng) + 1)
    if species_count and all(sym in species_count for sym in atom_symbols):
        num_basis = sum(species_count[sym] for sym in atom_symbols)
        basis_atom_map = np.zeros(num_basis, dtype=int)
        off = 0
        for atom_idx, sym in enumerate(atom_symbols):
            n = species_count[sym]
            basis_atom_map[off:off + n] = atom_idx
            off += n
        return AtomicBasisInfo(
            atom_positions=atom_positions,
            atom_symbols=atom_symbols,
            basis_atom_map=basis_atom_map,
            num_atoms=num_atoms,
            num_basis=num_basis,
        )

    if len(atoms_with_basis) == 1 and num_atoms > 1:
        # Only one atom has explicit orbitals (homoatomic, e.g. Bi2)
        # Assume symmetric: replicate for all atoms
        num_basis_per_atom = max_basis_index + 1
        num_basis = num_basis_per_atom * num_atoms
    elif atoms_without_basis:
        # Some atoms lack explicit basis listings (e.g., MgB2 where B2
        # is a symmetry copy of B1). Replicate basis from same-element atom.
        # First, compute per-atom basis counts from explicitly listed atoms
        atom_basis_counts = {}
        for atom_idx in atoms_with_basis:
            atom_ranges = [(f, l) for a, f, l in basis_ranges if a == atom_idx]
            if atom_ranges:
                first = min(f for f, _ in atom_ranges)
                last = max(l for _, l in atom_ranges)
                atom_basis_counts[atom_idx] = last - first + 1

        # For each missing atom, find a same-element atom with explicit basis
        extra_basis = 0
        missing_atom_basis_count = {}
        for missing_idx in atoms_without_basis:
            missing_sym = atom_symbols[missing_idx]
            # Find a reference atom of the same element
            ref_count = None
            for ref_idx in atoms_with_basis:
                if atom_symbols[ref_idx] == missing_sym and ref_idx in atom_basis_counts:
                    ref_count = atom_basis_counts[ref_idx]
                    break
            if ref_count is None:
                # No same-element reference found; use average
                ref_count = (max_basis_index + 1) // max(len(atoms_with_basis), 1)
            missing_atom_basis_count[missing_idx] = ref_count
            extra_basis += ref_count

        num_basis = max_basis_index + 1 + extra_basis
    else:
        # All atoms have explicit orbitals with global indexing (heteroatomic, e.g. SnTe)
        num_basis = max_basis_index + 1

    # Create basis-to-atom mapping
    basis_atom_map = np.zeros(num_basis, dtype=int)

    if len(atoms_with_basis) == 1 and num_atoms > 1:
        # Fill in symmetrically for all atoms
        num_basis_per_atom = max_basis_index + 1
        for atom_idx in range(num_atoms):
            start_basis = atom_idx * num_basis_per_atom
            end_basis = (atom_idx + 1) * num_basis_per_atom
            basis_atom_map[start_basis:end_basis] = atom_idx
    elif atoms_without_basis:
        # Fill explicit atoms first
        for atom_idx, first, last in basis_ranges:
            basis_atom_map[first:last+1] = atom_idx
        # Append replicated basis for missing atoms
        offset = max_basis_index + 1
        for missing_idx in atoms_without_basis:
            count = missing_atom_basis_count[missing_idx]
            basis_atom_map[offset:offset+count] = missing_idx
            offset += count
    else:
        # Use explicit ranges
        for atom_idx, first, last in basis_ranges:
            basis_atom_map[first:last+1] = atom_idx

    return AtomicBasisInfo(
        atom_positions=atom_positions,
        atom_symbols=atom_symbols,
        basis_atom_map=basis_atom_map,
        num_atoms=num_atoms,
        num_basis=num_basis
    )


def parse_orbital_types(lines: List[str], has_soc: bool = False, num_atoms: Optional[int] = None) -> Dict[int, str]:
    """
    Parse orbital type (S, P, D, F, G) for each basis function from CRYSTAL output.

    Extracts orbital angular momentum type from the "LOCAL ATOMIC FUNCTIONS BASIS SET"
    section. For systems with spin-orbit coupling, the mapping is doubled to account
    for spinor components.

    Parameters
    ----------
    lines : list of str
        Lines from the CRYSTAL output file
    has_soc : bool
        Whether spin-orbit coupling is enabled (doubles the orbital count)
    num_atoms : int, optional
        Number of atoms (for replicating orbital types across atoms)
        If not provided, auto-detected from atomic basis info

    Returns
    -------
    dict
        Dictionary mapping orbital_index (1-indexed) → orbital_type ('s', 'p', 'd', 'f', 'g')

    Examples
    --------
    >>> with open('crystal.out', 'r') as f:
    ...     lines = f.readlines()
    >>> orbital_types = parse_orbital_types(lines, has_soc=True)
    >>> orbital_types[1]
    's'
    >>> orbital_types[5]
    'p'

    Notes
    -----
    Orbital indices are 1-indexed to match CRYSTAL convention.
    The BASIS SET section format:
        1 S         ← orbital 1 is S type
        2 S         ← orbital 2 is S type
      5-     7 P    ← orbitals 5-7 are P type
     14-    18 D    ← orbitals 14-18 are D type

    For SOC systems, the orbital count is doubled (each spatial orbital has 2 spinor components),
    so we duplicate the mapping.
    """
    orbital_types = {}
    # species -> ordered shell types for ONE atom of that species. CRYSTAL
    # lists shells only for the FIRST atom of each species (with GLOBAL
    # orbital indices), so this is what has to be replicated -- see the
    # per-species reconstruction below.
    species_shells = {}
    cur_symbol = None
    atom_line = re.compile(r'^\s*(\d+)\s+([A-Z][A-Z]?)\s+-?\d')

    # Find the LOCAL ATOMIC FUNCTIONS BASIS SET section
    in_basis_section = False
    # ANCHORED, and SP-aware. The old pattern was r'...\s+([SPDFG])' with no
    # end anchor, so it matched an ATOM line whose element symbol merely STARTS
    # with a shell letter: "  85 SC -16.558   0.000   0.000" was read as orbital
    # 85 being an S shell. Measured on Sc_all_center: 47 matches of which 5 were
    # atom lines (85-89 SC), corrupting the map with the ATOM number used as an
    # orbital index. Requiring the shell letter to end the line removes exactly
    # those 5 and keeps all 42 real shells.
    #
    # (SP|[SPDFG]) not [SPDFG] because CRYSTAL shell code 1 is the SP hybrid,
    # four AOs ordered s,x,y,z. A single-letter class silently reads "SP" as S.
    orbital_type_pattern = re.compile(
        r'^\s*(\d+)(?:-\s*(\d+))?\s+(SP|[SPDFG])\s*$')

    for line in lines:
        if 'LOCAL ATOMIC FUNCTIONS BASIS SET' in line:
            in_basis_section = True
            continue

        if in_basis_section:
            # End of BASIS SET section
            if line.strip().startswith('*****') or line.strip() == '' or 'INFORMATION' in line:
                # Check if we've reached the end
                if orbital_types:  # Only break if we've found some orbitals
                    break

            a = atom_line.match(line)
            if a:
                cur_symbol = a.group(2).capitalize()
                species_shells.setdefault(cur_symbol, [])

            # Try to match orbital type lines
            match = orbital_type_pattern.match(line)
            if match:
                start_idx = int(match.group(1))
                end_idx_str = match.group(2)
                orb_type = match.group(3).lower()

                end_idx = int(end_idx_str) if end_idx_str else start_idx
                if orb_type == 'sp':
                    # CRYSTAL shell code 1: four AOs ordered s, x, y, z. The
                    # first index is the s, the remaining three are p.
                    types = ['s' if j == 0 else 'p'
                             for j in range(end_idx - start_idx + 1)]
                else:
                    types = [orb_type] * (end_idx - start_idx + 1)
                for idx, t in zip(range(start_idx, end_idx + 1), types):
                    orbital_types[idx] = t
                if cur_symbol is not None:
                    species_shells[cur_symbol].extend(types)

    if not orbital_types:
        return {}

    max_idx = max(orbital_types.keys())

    # Auto-detect number of atoms if not provided
    if num_atoms is None:
        try:
            atomic_info = parse_atomic_basis_info(lines)
            num_atoms = atomic_info.num_atoms
        except:
            num_atoms = 1

    # Determine if the file already lists orbitals for ALL atoms (heteroatomic)
    # or only the first atom (homoatomic, needs replication)
    try:
        atomic_info = parse_atomic_basis_info(lines)
        total_basis = atomic_info.num_basis
    except:
        total_basis = None

    # PER-SPECIES RECONSTRUCTION (preferred). CRYSTAL lists shells for the
    # first atom of EACH species, so the thing to replicate is a species'
    # shell list -- not the whole first atom's pattern across every atom.
    #
    # The old global replication assumed a homoatomic cell. On Sc_all_center
    # (89 atoms; C 36, H 24, B 12, O 12, Sc 5) it produced 8900 entries with
    # indices up to 112496 for a 1424-orbital basis. Per species it comes out
    # exactly right: 36x18 + 24x6 + 12x18 + 12x18 + 5x40 = 1424, matching the
    # S(k) dimension.
    try:
        symbols = list(parse_atomic_basis_info(lines).atom_symbols)
    except Exception:
        symbols = []
    if symbols and all(species_shells.get(sym) for sym in symbols):
        all_orbital_types = {}
        idx = 1
        for sym in symbols:
            for t in species_shells[sym]:
                all_orbital_types[idx] = t
                idx += 1
    elif total_basis is not None and max_idx == total_basis:
        # All atoms' orbitals already listed with global indices (e.g. SnTe: 1-71)
        all_orbital_types = dict(orbital_types)
    else:
        # Only first atom listed (e.g. Bi2: 1-28) → replicate for all atoms
        orbitals_per_atom = max_idx
        all_orbital_types = {}
        for atom_idx in range(num_atoms):
            offset = atom_idx * orbitals_per_atom
            for orb_idx, orb_type in orbital_types.items():
                all_orbital_types[orb_idx + offset] = orb_type

    # For SOC systems, double the mapping (each spatial orbital → 2 spinor components)
    if has_soc:
        num_spatial_orbitals = max(all_orbital_types.keys())
        soc_orbital_types = {}

        # Copy original mapping
        for idx, orb_type in all_orbital_types.items():
            soc_orbital_types[idx] = orb_type

        # Add spinor duplicates (shifted by num_spatial_orbitals)
        for idx, orb_type in all_orbital_types.items():
            soc_orbital_types[idx + num_spatial_orbitals] = orb_type

        return soc_orbital_types

    return all_orbital_types


# ==============================
# Matrix Utility Functions
# ==============================

def is_hermitian(matrix: np.ndarray, tol: float = 1e-34) -> bool:
    """Check if a matrix is Hermitian."""
    return np.allclose(matrix, matrix.conj().T, atol=tol)


def fill_raw_matrix(
    H_R_dict: Dict, 
    S_R_dict: Dict, 
    R: Tuple[int, int, int], 
    N_basis: int, 
    is_fock: bool = True
) -> np.ndarray:
    """
    Constructs a 2N x 2N matrix containing ONLY the raw lower-triangular 
    data found in the file for a specific R vector.
    """
    if is_fock:
        M_raw = np.zeros((2 * N_basis, 2 * N_basis), dtype=np.complex128)
        data_source = H_R_dict
    else:
        M_raw = np.zeros((2 * N_basis, 2 * N_basis), dtype=np.float64)
        data_source = S_R_dict

    def insert_block(spin_key, row_offset, col_offset):
        if is_fock:
            if R in data_source and spin_key in data_source[R]:
                block = data_source[R][spin_key]
                # Enforce lower triangular reading
                M_raw[row_offset:row_offset+N_basis, col_offset:col_offset+N_basis] = np.tril(block)
        else:
            if R in data_source:
                block = data_source[R]
                L = np.tril(block)
                # Apply to Top-Left (AA)
                M_raw[0:N_basis, 0:N_basis] = L
                # Apply to Bottom-Right (BB)
                M_raw[N_basis:2*N_basis, N_basis:2*N_basis] = L
    
    if is_fock:
        insert_block('ALPHA_ALPHA', 0, 0)
        insert_block('BETA_BETA', N_basis, N_basis)
        insert_block('ALPHA_BETA', 0, N_basis) 
        insert_block('BETA_ALPHA', N_basis, 0) 
    else:
        # Overlap handling (assumed symmetric for AA and BB)
        insert_block(None, 0, 0)

    return M_raw


# ==============================
# Matrix Parsing Functions
# ==============================

def parse_matrix_data(lines: List[str], start_index: int) -> Dict:
    """Parse matrix data from file lines starting at a given index."""
    data_lines = []
    i = start_index
    col_indices = np.array([], dtype=np.int64)
    
    while i < len(lines):
        line = lines[i].rstrip('\n')
        line_stripped = line.strip()
        
        if not line_stripped:
            i += 1
            continue

        # Fast path: data rows and column-index rows start with a digit; only a
        # non-digit-leading line can be a section header. Testing that first skips
        # the 6 header regexes on the millions of data lines (~27M -> ~3.4M matches,
        # ~2.4 s saved at N=1424).
        if line_stripped[0].isdigit():
            if column_indices_pattern.match(line):
                col_indices = np.asarray(line_stripped.split(), dtype=np.int64) - 1
            else:
                parts = line.split()
                if len(parts) >= 2:
                    row_index = int(parts[0]) - 1
                    tail = parts[1:]
                    # CRYSTAL emits 'E' exponents; only pay the D->E replace if present
                    if 'D' in line or 'd' in line:
                        tail = [v.replace('D', 'E').replace('d', 'e') for v in tail]
                    data_values = np.asarray(tail, dtype=np.float64)
                    data_lines.append((row_index, col_indices, data_values))
            i += 1
            continue

        # A non-digit-leading line that is a section header ends this matrix.
        if (overlap_header_pattern.match(line) or
            fock_header_pattern.match(line) or
            fock_simple_header_pattern.match(line) or
            spin_channel_pattern.match(line) or
            spin_simple_pattern.match(line) or
            direct_lattice_header_pattern.match(line)):
            break
        i += 1
    
    if data_lines:
        max_index = max(row for row, _, _ in data_lines)
        n = max_index + 1
        matrix = np.zeros((n, n), dtype=np.float64)
        for row, cols, values in data_lines:
            k = min(cols.size, values.size)
            matrix[row, cols[:k]] = values[:k]
        return {'matrix': matrix, 'next_index': i}
    else:
        print(f"Warning: No matrix data found starting at line {start_index}")
        return {'matrix': None, 'next_index': i}


def parse_overlap_and_fock_matrices_cached(filepath, lines, *, cache_dir=None,
                                           cache_path=None):
    """Cache wrapper for parse_overlap_and_fock_matrices.

    The 530 MB regex parse is the serial Amdahl floor and is repeated by every
    Stage 1 and Stage 2 invocation. This caches the parsed (raw_matrices, lattice)
    to <filepath>.parsecache.pkl keyed on the file's (mtime_ns, size); subsequent
    runs deserialize in ~1-2 s instead of re-parsing. ``cache_dir`` redirects the
    cache without changing the input file's directory. Its filename contains a
    stable hash of the resolved input path, so equal-named files cannot share a
    redirected cache. ``cache_path`` selects an exact cache location. The caller is
    responsible for creating a redirected directory. Best-effort: any cache miss,
    stale key, or error falls through to a normal parse.
    """
    import hashlib
    import os
    import pickle
    source_path = os.path.realpath(os.path.abspath(os.fspath(filepath)))
    redirected_cache = cache_dir is not None or cache_path is not None
    if cache_path is None:
        if cache_dir is None:
            cache_path = str(filepath) + '.parsecache.pkl'
        else:
            source_id = hashlib.sha256(source_path.encode('utf-8')).hexdigest()[:16]
            cache_path = os.path.join(
                os.fspath(cache_dir),
                os.path.basename(os.fspath(filepath))
                + '.' + source_id + '.parsecache.pkl',
            )
    else:
        cache_path = os.fspath(cache_path)
    try:
        st = os.stat(filepath)
        key = (st.st_mtime_ns, st.st_size)
        if redirected_cache:
            key = (source_path,) + key
    except OSError:
        return parse_overlap_and_fock_matrices(lines)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as fh:
                ckey, raw, latv = pickle.load(fh)
            if ckey == key:
                return raw, latv
        except Exception:
            pass  # corrupt / incompatible cache -> reparse
    raw, latv = parse_overlap_and_fock_matrices(lines)
    try:
        with open(cache_path, 'wb') as fh:
            pickle.dump((key, raw, latv), fh, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass  # caching is best-effort; never block the parse on a write failure
    return raw, latv


def parse_overlap_and_fock_matrices(lines: List[str]) -> Tuple[List[Dict], Optional[List[List[float]]]]:
    """
    Parse overlap and Fock matrices from CRYSTAL/LCAO output file lines.
    
    NOTE: This version returns RAW complex matrices. It does NOT enforce 
    Hermiticity at this stage, as that is handled during global construction.
    """
    matrices = []
    direct_lattice_vectors = None
    current_spin_channel = None
    i = 0
    fock_temp_storage = {}
    
    while i < len(lines):
        line = lines[i].rstrip('\n')
        
        # Parse direct lattice vectors
        if direct_lattice_header_pattern.match(line):
            vectors = []
            i += 1
            for _ in range(3):
                if i >= len(lines):
                    print("Warning: Unexpected end of file while reading direct lattice vectors.")
                    break
                vector_line = lines[i].strip()
                vector_match = vector_line_pattern.match(vector_line)
                if vector_match:
                    components = [
                        float(val.replace('D', 'E').replace('d', 'e'))
                        for val in float_pattern.findall(vector_line)
                    ]
                    vectors.append(components)
                i += 1
            if len(vectors) == 3:
                direct_lattice_vectors = vectors
        
        # Parse spin channel header (compound: ALPHA_ALPHA, ALPHA_BETA, etc.)
        elif spin_channel_pattern.match(line):
            spin_match = spin_channel_pattern.match(line)
            current_spin_channel = spin_match.group(1).upper()
            i += 1

        # Parse simple spin channel header (ALPHA/BETA ELECTRONS)
        elif spin_simple_pattern.match(line):
            spin_match = spin_simple_pattern.match(line)
            simple_label = spin_match.group(1).upper()
            # Map simple labels to compound labels for spin-block assembly
            if simple_label == 'ALPHA':
                current_spin_channel = 'ALPHA_ALPHA'
            elif simple_label == 'BETA':
                current_spin_channel = 'BETA_BETA'
            i += 1

        # Parse overlap matrix
        elif overlap_header_pattern.match(line):
            header_match = overlap_header_pattern.match(line)
            lattice_vector = [int(header_match.group(j)) for j in range(1, 4)]
            matrix_type = 'overlap'
            i += 1
            S_parsed = parse_matrix_data(lines, i)
            matrices.append({
                'type': matrix_type,
                'part': 'real',
                'spin_channel': None,
                'lattice_vector': lattice_vector,
                'data': S_parsed['matrix'],
            })
            i = S_parsed['next_index']

        # Parse Fock matrix with REAL/IMAG parts (complex SOC format)
        elif fock_header_pattern.match(line):
            header_match = fock_header_pattern.match(line)
            part = header_match.group(1).lower()
            lattice_vector = tuple(int(header_match.group(j)) for j in range(2, 5))
            matrix_type = 'fock'
            i += 1
            F_parsed = parse_matrix_data(lines, i)
            key = (current_spin_channel, lattice_vector)
            if key not in fock_temp_storage:
                fock_temp_storage[key] = {}
            fock_temp_storage[key][part] = F_parsed['matrix']
            i = F_parsed['next_index']

        # Parse simple Fock matrix (real-valued, no REAL/IMAG split)
        elif fock_simple_header_pattern.match(line):
            header_match = fock_simple_header_pattern.match(line)
            lattice_vector = tuple(int(header_match.group(j)) for j in range(1, 4))
            i += 1
            F_parsed = parse_matrix_data(lines, i)
            key = (current_spin_channel, lattice_vector)
            if key not in fock_temp_storage:
                fock_temp_storage[key] = {}
            # Store as real part only (no imaginary component)
            fock_temp_storage[key]['real'] = F_parsed['matrix']
            i = F_parsed['next_index']
        else:
            i += 1
    
    # Combine real and imaginary parts for Fock matrices
    for key, parts in fock_temp_storage.items():
        spin_channel, lattice_vector = key
        real_part = parts.get('real')
        imag_part = parts.get('imag')
        
        if real_part is not None and imag_part is not None:
            complex_matrix = real_part + 1j * imag_part
        elif real_part is not None:
            complex_matrix = real_part.astype(np.complex128)
        elif imag_part is not None:
            complex_matrix = 1j * imag_part
        else:
            continue
        
        # We append the complex matrix AS IS.
        # Hermiticity is enforced later in create_spin_block_matrices.
        matrices.append({
            'type': 'fock',
            'part': 'complex',
            'spin_channel': spin_channel,
            'lattice_vector': lattice_vector,
            'data': complex_matrix,
        })
    
    return matrices, direct_lattice_vectors


def parse_overlap_and_fock_matrices_streaming(
    filepath: str,
    promote_complex: bool = True,
) -> Tuple[Dict, Dict, Optional[List[List[float]]], List[str]]:
    """
    Streaming, low-memory equivalent of ``parse_overlap_and_fock_matrices``
    *plus* the H_R_dict/S_R_dict organizing step.

    Reads the CRYSTAL output line by line, building the per-R-vector matrices
    directly. It never holds the whole file (no ``f.readlines()``) and never
    builds the intermediate list of every parsed block, so peak memory is the
    R-space matrices alone — not file_buffer + list + dicts simultaneously.

    The control flow mirrors ``parse_overlap_and_fock_matrices`` exactly (same
    regexes, same lower-triangular reconstruction, same (spin, R) Fock
    combination order), so the returned dicts are identical to feeding that
    function's output through the main script's organizing loop.

    Parameters
    ----------
    filepath : str
        Path to the CRYSTAL/LCAO output file.
    promote_complex : bool or 'auto'
        If True (default, matches legacy behaviour), real-only Fock blocks are
        promoted to complex128. If False, real-only blocks are kept float64
        (Design C: halves R-space memory for non-SOC systems). If 'auto', the
        choice is made from SOC detection in the header (TWO-COMPONENT SCF):
        complex for SOC, float64 for non-SOC. Blocks with a genuine imaginary
        part are always complex regardless.

    Returns
    -------
    H_R_dict : dict
        Maps R-tuple -> {spin_channel: matrix} (complex128, or float64 when
        ``promote_complex`` is False and there is no imaginary part).
    S_R_dict : dict
        Maps R-tuple -> overlap matrix (float64).
    direct_lattice_vectors : list of list of float or None
    header_lines : list of str
        All lines preceding the first matrix block (with newlines), suitable
        for ``parse_calculation_parameters`` / ``parse_atomic_basis_info`` /
        ``parse_orbital_types`` without re-reading the file.
    """
    H_R_dict: Dict = {}
    S_R_dict: Dict = {}
    direct_lattice_vectors = None
    fock_temp_storage: Dict = {}
    header_lines: List[str] = []
    seen_first_matrix = False

    # In-progress matrix-collection state
    collecting = False
    col_indices: List[int] = []
    data_lines: List[Tuple[int, List[int], List[float]]] = []
    target = None  # ('overlap', R) or ('fock', spin, R, part)

    current_spin_channel = None
    capture_vectors = 0
    pending_vectors: List[List[float]] = []

    def finalize_matrix():
        """Build the accumulated lower-triangular block and store it."""
        nonlocal collecting, col_indices, data_lines, target
        if target is not None and data_lines:
            max_index = max(row for row, _, _ in data_lines)
            n = max_index + 1
            matrix = np.zeros((n, n), dtype=np.float64)
            for row, cols, values in data_lines:
                for col, val in zip(cols, values):
                    matrix[row, col] = val
            kind = target[0]
            if kind == 'overlap':
                S_R_dict[target[1]] = matrix
            else:  # ('fock', spin, R, part)
                _, spin, R, part = target
                fock_temp_storage.setdefault((spin, R), {})[part] = matrix
        collecting = False
        col_indices = []
        data_lines = []
        target = None

    with open(filepath, 'r') as f:
        for raw in f:
            line = raw.rstrip('\n')

            # Capture the 3 lattice-vector lines following a DIRECT LATTICE header
            if capture_vectors > 0:
                vm = vector_line_pattern.match(line.strip())
                if vm:
                    comps = [float(v.replace('D', 'E').replace('d', 'e'))
                             for v in float_pattern.findall(line.strip())]
                    pending_vectors.append(comps)
                capture_vectors -= 1
                if capture_vectors == 0 and len(pending_vectors) == 3:
                    direct_lattice_vectors = pending_vectors
                continue

            m_overlap = overlap_header_pattern.match(line)
            m_fock = fock_header_pattern.match(line)
            m_fock_simple = fock_simple_header_pattern.match(line)
            m_spin = spin_channel_pattern.match(line)
            m_spin_simple = spin_simple_pattern.match(line)
            m_direct = direct_lattice_header_pattern.match(line)

            is_header = (m_overlap or m_fock or m_fock_simple or m_spin
                         or m_spin_simple or m_direct)

            if is_header:
                # Any header terminates the matrix currently being read.
                finalize_matrix()

                if not seen_first_matrix and (m_overlap or m_fock or m_fock_simple):
                    seen_first_matrix = True

                if m_direct:
                    pending_vectors = []
                    capture_vectors = 3
                elif m_spin:
                    current_spin_channel = m_spin.group(1).upper()
                elif m_spin_simple:
                    label = m_spin_simple.group(1).upper()
                    current_spin_channel = ('ALPHA_ALPHA' if label == 'ALPHA'
                                            else 'BETA_BETA')
                elif m_overlap:
                    R = tuple(int(m_overlap.group(j)) for j in range(1, 4))
                    target = ('overlap', R)
                    collecting = True
                elif m_fock:
                    part = m_fock.group(1).lower()
                    R = tuple(int(m_fock.group(j)) for j in range(2, 5))
                    target = ('fock', current_spin_channel, R, part)
                    collecting = True
                elif m_fock_simple:
                    R = tuple(int(m_fock_simple.group(j)) for j in range(1, 4))
                    target = ('fock', current_spin_channel, R, 'real')
                    collecting = True
                continue

            # Non-header line
            if not seen_first_matrix:
                header_lines.append(raw)

            if collecting:
                stripped = line.strip()
                if stripped == '':
                    continue
                if column_indices_pattern.match(line):
                    col_indices = [int(num) - 1 for num in stripped.split()]
                else:
                    dm = data_line_pattern.match(line)
                    if dm:
                        row_index = int(dm.group(1)) - 1
                        values = [float(v.replace('D', 'E').replace('d', 'e'))
                                  for v in float_pattern.findall(dm.group(2))]
                        data_lines.append((row_index, col_indices.copy(), values))

        # Flush the final block at EOF
        finalize_matrix()

    # Resolve 'auto': float64 for non-SOC, complex128 for SOC.
    if promote_complex == 'auto':
        has_soc = any('TWO-COMPONENT' in l and 'SCF' in l for l in header_lines)
        promote_complex = bool(has_soc)

    # Combine Fock real/imag parts (same order/semantics as the legacy parser)
    for (spin_channel, R), parts in fock_temp_storage.items():
        real_part = parts.get('real')
        imag_part = parts.get('imag')
        if real_part is not None and imag_part is not None:
            combined = real_part + 1j * imag_part
        elif real_part is not None:
            combined = real_part if not promote_complex else real_part.astype(np.complex128)
        elif imag_part is not None:
            combined = 1j * imag_part
        else:
            continue
        H_R_dict.setdefault(R, {})[spin_channel] = combined

    return H_R_dict, S_R_dict, direct_lattice_vectors, header_lines


# ==============================
# Spin Block Matrix Creation
# ==============================

def create_spin_block_matrices(
    H_R_dict: Dict[Tuple[int, int, int], Dict[str, np.ndarray]],
    S_R_dict: Dict[Tuple[int, int, int], np.ndarray],
    N_basis: int,
    direct_lattice_vectors: List[List[float]],
    PRINTOUT: bool = False
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Construct full 2N x 2N spin block Hamiltonian and overlap matrices.
    
    This function uses Global Pair Symmetry Construction:
    1. It identifies pairs of vectors (R, -R).
    2. It combines the raw lower-triangular data of R with the raw lower-triangular
       data of -R to strictly enforce the relationship H(R) = H(-R)†.
    3. It handles the origin H(0) by ensuring exact Hermiticity and real diagonal.

    Returns
    -------
    H_full_list : list of tuples (R_cartesian, H_matrix)
    S_full_list : list of tuples (R_cartesian, S_matrix)

    """
    H_full_list = []
    S_full_list = []
    direct_lattice_vectors = np.array(direct_lattice_vectors)
    
    all_R_vectors = set(H_R_dict.keys())
    
    # Identify valid pairs (ensure we only process (R, -R) once)
    processed_R = set()
    valid_pairs = [] 
    
    # Sort for consistent output/processing order
    sorted_R = sorted(list(all_R_vectors))
    
    for R in sorted_R:
        if R in processed_R: continue
        
        minus_R = tuple(-x for x in R)
        
        if R == (0, 0, 0):
            valid_pairs.append((R, R))
            processed_R.add(R)
        elif minus_R in all_R_vectors:
            valid_pairs.append((R, minus_R))
            processed_R.add(R)
            processed_R.add(minus_R)
        else:
            if PRINTOUT:
                print(f"Warning: Vector {R} exists but {minus_R} is missing. Excluding.")

    # Iterate over pairs and construct matrices
    for R, minus_R in valid_pairs:
        is_origin = (R == minus_R)
        
        # Calculate Cartesian vectors
        R_cart = np.dot(np.array(R), direct_lattice_vectors)
        minus_R_cart = np.dot(np.array(minus_R), direct_lattice_vectors)
        
        # ============================================================
        # 1. FOCK CONSTRUCTION
        # ============================================================
        
        # Build Raw Lower Data (Global 2N x 2N container)
        M_raw_R = fill_raw_matrix(H_R_dict, S_R_dict, R, N_basis, is_fock=True)
        M_raw_minus_R = fill_raw_matrix(H_R_dict, S_R_dict, minus_R, N_basis, is_fock=True)
        
        # Prepare the Upper part from the -R partner
        # We take the STRICT lower triangle of -R (remove diagonal)
        # Its Conjugate Transpose becomes the Strict Upper Triangle of R
        M_minus_R_nodiag = M_raw_minus_R.copy()
        rows, cols = np.diag_indices_from(M_minus_R_nodiag)
        M_minus_R_nodiag[rows, cols] = 0 
        
        # Combine: H(R) = Lower(R) + [Lower_Strict(-R)]†
        H_R_full = M_raw_R + M_minus_R_nodiag.conj().T
        
        # Enforce Origin Hermiticity (Physical Requirement)
        if is_origin:
            diags = np.diag(H_R_full)
            np.fill_diagonal(H_R_full, np.real(diags))
        
        # Add H(R) to list
        H_full_list.append((R_cart, H_R_full))
        
        # Derive and Add H(-R)
        if not is_origin:
            # Enforce exact symmetry: H(-R) = H(R)†
            H_minus_R_full = H_R_full.conj().T
            H_full_list.append((minus_R_cart, H_minus_R_full))
        
        # ============================================================
        # 2. OVERLAP CONSTRUCTION
        # ============================================================
        if R in S_R_dict and (is_origin or minus_R in S_R_dict):
            # Note: For origin, S_raw_R and S_raw_minus_R are the same
            S_raw_R = fill_raw_matrix(H_R_dict, S_R_dict, R, N_basis, is_fock=False)
            S_raw_minus_R = fill_raw_matrix(H_R_dict, S_R_dict, minus_R, N_basis, is_fock=False)
            
            # Prepare Upper part from -R partner
            S_minus_R_nodiag = S_raw_minus_R.copy()
            rows, cols = np.diag_indices_from(S_minus_R_nodiag)
            S_minus_R_nodiag[rows, cols] = 0
            
            # Combine: S(R) = Lower(R) + [Lower_Strict(-R)]^T 
            # (Transpose only, because Overlap is Real)
            S_R_full = S_raw_R + S_minus_R_nodiag.T
            
            S_full_list.append((R_cart, S_R_full))

            if not is_origin:
                # Enforce exact symmetry: S(-R) = S(R)^T
                S_minus_R_full = S_R_full.T
                S_full_list.append((minus_R_cart, S_minus_R_full))

    return H_full_list, S_full_list


def _select_spin_matrix(H_mats, spin_channel):
    """Pick one spin channel's Fock matrix from H_R_dict[R].

    H_mats is either a bare matrix or a {spin_channel: matrix} dict. When
    spin_channel is given (e.g. 'ALPHA_ALPHA' / 'BETA_BETA') and present, that
    channel is returned; otherwise the first channel is used (legacy behaviour,
    correct for restricted single-channel systems).
    """
    if not isinstance(H_mats, dict):
        return H_mats
    if spin_channel is not None and spin_channel in H_mats:
        return H_mats[spin_channel]
    return list(H_mats.values())[0]


def create_nonsoc_full_matrices(
    H_R_dict: Dict[Tuple[int, int, int], Dict[str, np.ndarray]],
    S_R_dict: Dict[Tuple[int, int, int], np.ndarray],
    direct_lattice_vectors: List[List[float]],
    PRINTOUT: bool = False,
    spin_channel: Optional[str] = None,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Construct full N x N Hamiltonian and overlap matrices for non-SOC systems.

    Crystal23 prints matrices in lower-triangular format. This function
    reconstructs the full symmetric matrices using the (R, -R) pair relationship:
        H(R) = Lower(R) + StrictLower(-R)^T
        H(-R) = H(R)^T   (real symmetric for non-SOC)
        S(R) = Lower(R) + StrictLower(-R)^T
        S(-R) = S(R)^T

    For the origin R=(0,0,0), the matrix is self-symmetric:
        S(0) = Lower(S_raw) + StrictLower(S_raw)^T

    This is the non-SOC equivalent of create_spin_block_matrices.

    Parameters
    ----------
    H_R_dict : dict
        Maps R-vector tuple -> {spin_channel: matrix}
    S_R_dict : dict
        Maps R-vector tuple -> matrix
    direct_lattice_vectors : list of list of float
        3x3 lattice vectors
    PRINTOUT : bool
        Whether to print diagnostic information

    Returns
    -------
    H_full_list : list of tuples (R_cartesian, H_matrix)
    S_full_list : list of tuples (R_cartesian, S_matrix)
    """
    H_full_list = []
    S_full_list = []
    direct_lattice_vectors = np.array(direct_lattice_vectors)

    all_R_vectors = set(H_R_dict.keys()) | set(S_R_dict.keys())

    # Identify valid (R, -R) pairs
    processed_R = set()
    valid_pairs = []

    sorted_R = sorted(list(all_R_vectors))

    for R in sorted_R:
        if R in processed_R:
            continue

        minus_R = tuple(-x for x in R)

        if R == (0, 0, 0):
            valid_pairs.append((R, R))
            processed_R.add(R)
        elif minus_R in all_R_vectors:
            valid_pairs.append((R, minus_R))
            processed_R.add(R)
            processed_R.add(minus_R)
        else:
            if PRINTOUT:
                print(f"Warning: Vector {R} exists but {minus_R} is missing. Excluding.")

    for R, minus_R in valid_pairs:
        is_origin = (R == minus_R)

        # Calculate Cartesian vectors
        R_cart = np.dot(np.array(R), direct_lattice_vectors)
        minus_R_cart = np.dot(np.array(minus_R), direct_lattice_vectors)

        # ============================================================
        # 1. HAMILTONIAN CONSTRUCTION
        # ============================================================
        if R in H_R_dict:
            # Get the raw lower-triangular Fock matrix for R
            H_mats_R = H_R_dict[R]
            H_raw_R = _select_spin_matrix(H_mats_R, spin_channel)
            H_raw_R = np.tril(H_raw_R)  # Enforce lower triangle

            if is_origin:
                # Origin: H(0) = Lower + StrictLower^T (symmetric)
                H_R_full = H_raw_R + np.tril(H_raw_R, -1).T
                H_full_list.append((R_cart, H_R_full))
            else:
                if minus_R in H_R_dict:
                    H_mats_mR = H_R_dict[minus_R]
                    H_raw_mR = _select_spin_matrix(H_mats_mR, spin_channel)
                    H_raw_mR = np.tril(H_raw_mR)

                    # H(R) = Lower(R) + StrictLower(-R)^T
                    H_mR_nodiag = H_raw_mR.copy()
                    np.fill_diagonal(H_mR_nodiag, 0)
                    H_R_full = H_raw_R + H_mR_nodiag.T

                    H_full_list.append((R_cart, H_R_full))

                    # H(-R) = H(R)^T (real symmetric for non-SOC)
                    H_minus_R_full = H_R_full.T
                    H_full_list.append((minus_R_cart, H_minus_R_full))
                else:
                    if PRINTOUT:
                        print(f"Warning: H at {minus_R} missing for pair with {R}")

        # ============================================================
        # 2. OVERLAP CONSTRUCTION
        # ============================================================
        if R in S_R_dict:
            S_raw_R = np.tril(S_R_dict[R])  # Enforce lower triangle

            if is_origin:
                # Origin: S(0) = Lower + StrictLower^T (symmetric)
                S_R_full = S_raw_R + np.tril(S_raw_R, -1).T
                S_full_list.append((R_cart, S_R_full))
            else:
                if minus_R in S_R_dict:
                    S_raw_mR = np.tril(S_R_dict[minus_R])

                    # S(R) = Lower(R) + StrictLower(-R)^T
                    S_mR_nodiag = S_raw_mR.copy()
                    np.fill_diagonal(S_mR_nodiag, 0)
                    S_R_full = S_raw_R + S_mR_nodiag.T

                    S_full_list.append((R_cart, S_R_full))

                    # S(-R) = S(R)^T (real symmetric for non-SOC)
                    S_minus_R_full = S_R_full.T
                    S_full_list.append((minus_R_cart, S_minus_R_full))
                else:
                    if PRINTOUT:
                        print(f"Warning: S at {minus_R} missing for pair with {R}")

    return H_full_list, S_full_list
