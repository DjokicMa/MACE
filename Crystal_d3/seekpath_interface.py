"""
SeeKPath Interface Module for MACE

This module provides accurate band structure k-path generation using the
official seekpath library. It handles:
- Structure extraction from CRYSTAL output files
- Conversion to/from seekpath format
- Proper handling of parametric k-points
- Discontinuity detection and labeling

Author: Marcus Djokic
Institution: Michigan State University, Mendoza Group

Based on:
- SeeKPath library (https://github.com/giovannipizzi/seekpath)
- HPKOT paper: Hinuma et al., Comp. Mat. Sci. 128, 140 (2017)
"""

import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from fractions import Fraction

# Check if seekpath is available
try:
    import seekpath
    SEEKPATH_AVAILABLE = True
except ImportError:
    SEEKPATH_AVAILABLE = False
    print("Note: seekpath library not installed. Install with: pip install seekpath")


# =============================================================================
# Cell Parameter Conversion
# =============================================================================

def cell_params_to_vectors(a: float, b: float, c: float,
                           alpha_deg: float, beta_deg: float, gamma_deg: float) -> np.ndarray:
    """
    Convert cell parameters (a, b, c, alpha, beta, gamma) to 3x3 lattice vector matrix.

    Uses standard crystallographic convention:
    - a vector along x-axis
    - b vector in xy-plane
    - c vector completes right-handed system

    Args:
        a, b, c: Lattice constants in Angstroms
        alpha_deg, beta_deg, gamma_deg: Angles in degrees

    Returns:
        3x3 numpy array of lattice vectors (rows are vectors)
    """
    # Convert to radians
    alpha = np.radians(alpha_deg)
    beta = np.radians(beta_deg)
    gamma = np.radians(gamma_deg)

    # Vector a along x
    va = np.array([a, 0.0, 0.0])

    # Vector b in xy-plane
    vb = np.array([b * np.cos(gamma), b * np.sin(gamma), 0.0])

    # Vector c
    cx = c * np.cos(beta)
    cy = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
    cz = np.sqrt(max(0.0, c**2 - cx**2 - cy**2))  # max to avoid numerical issues
    vc = np.array([cx, cy, cz])

    return np.array([va, vb, vc])


# =============================================================================
# CRYSTAL Output Parsing
# =============================================================================

# A computed geometry block - "GEOMETRY FOR WAVE FUNCTION", each optimisation
# step, "FINAL OPTIMIZED GEOMETRY" - starts with this line, then the cell
# (A B C ALPHA BETA GAMMA), "ATOMS IN THE ASYMMETRIC UNIT n - ATOMS IN THE
# UNIT CELL: m" and the m atoms (T/F rows). A centred lattice adds the
# "TRANSFORMATION MATRIX PRIMITIVE-CRYSTALLOGRAPHIC CELL" W and the
# "CRYSTALLOGRAPHIC CELL" parameters. The input echo before it ("LATTICE
# PARAMETERS ... - PRIMITIVE CELL") is the cell given in input, which is not
# the computed one after SUPERCEL or other geometry editing, nor the
# optimised one.
_BLOCK_START = re.compile(r'^\s*PRIMITIVE CELL - CENTRING CODE')
_UNIT_CELL_ATOMS = re.compile(r'ATOMS IN THE UNIT CELL:\s*(\d+)')


def _six_numbers(line: str) -> Optional[List[float]]:
    parts = line.split()
    if len(parts) < 6:
        return None
    try:
        return [float(x) for x in parts[:6]]
    except ValueError:
        return None


def _params_after(lines: List[str], start: int, stop: int) -> Optional[Dict[str, float]]:
    """Cell parameters on the line after the first A B C ALPHA BETA GAMMA header."""
    for j in range(start, min(stop, len(lines) - 1)):
        if 'ALPHA' in lines[j] and 'GAMMA' in lines[j]:
            values = _six_numbers(lines[j + 1])
            if values is None:
                return None
            return dict(zip(('a', 'b', 'c', 'alpha', 'beta', 'gamma'), values))
    return None


def _atom_row(line: str):
    parts = line.split()
    if len(parts) < 7 or parts[1] not in ('T', 'F') or not parts[0].isdigit():
        return None
    try:
        return int(parts[2]), [float(parts[4]), float(parts[5]), float(parts[6])]
    except ValueError:
        return None


def _geometry_blocks(lines: List[str]) -> List[Dict[str, Any]]:
    """Every computed geometry block of a CRYSTAL output, in order.

    Each block holds its line index ('line'), the cell parameters
    ('params'), the atom count CRYSTAL prints ('n_atoms') and the atoms read
    ('numbers', 'positions'), W when printed ('W', with the conventional
    vectors a, b, c = the columns of W in terms of the primitive ones: rows
    A_conv = W^T A_prim, as the DIRECT LATTICE VECTORS of C- and R-centred
    outputs show), the CRYSTALLOGRAPHIC CELL parameters when printed
    ('conventional') and whether FINAL OPTIMIZED GEOMETRY heads it ('final').
    """
    starts = [i for i, line in enumerate(lines) if _BLOCK_START.match(line)]
    blocks = []
    for n, start in enumerate(starts):
        stop = starts[n + 1] if n + 1 < len(starts) else len(lines)
        block = {
            'line': start,
            'final': any('FINAL OPTIMIZED GEOMETRY' in lines[j]
                         for j in range(max(0, start - 6), start)),
            'params': _params_after(lines, start + 1, min(start + 4, stop)),
            'n_atoms': None, 'numbers': [], 'positions': [],
            'W': None, 'conventional': None,
        }
        blocks.append(block)
        k = start + 1
        while k < min(start + 8, stop):
            match = _UNIT_CELL_ATOMS.search(lines[k])
            if match:
                block['n_atoms'] = int(match.group(1))
                break
            k += 1
        if block['n_atoms'] is None:
            continue
        k += 1
        # the column header and the line of stars
        while k < min(stop, start + 14) and _atom_row(lines[k]) is None:
            k += 1
        while k < stop:
            row = _atom_row(lines[k])
            if row is None:
                break
            block['numbers'].append(row[0])
            block['positions'].append(row[1])
            k += 1
        for q in range(k, min(k + 16, stop)):
            if 'TRANSFORMATION MATRIX PRIMITIVE-CRYSTALLOGRAPHIC CELL' in lines[q] and q + 1 < stop:
                try:
                    values = [float(x) for x in lines[q + 1].split()[:9]]
                except ValueError:
                    values = []
                if len(values) == 9:
                    block['W'] = np.array(values).reshape(3, 3)
            elif re.match(r'^\s*CRYSTALLOGRAPHIC CELL', lines[q]):
                block['conventional'] = _params_after(lines, q + 1, min(q + 4, stop))
                break
    return blocks


def select_geometry_block(blocks: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """The block BAND's fort.9 holds, and which one it is ('final' or 'first').

    After an optimisation that is the block after the last FINAL OPTIMIZED
    GEOMETRY. Otherwise it is the first computed block (GEOMETRY FOR WAVE
    FUNCTION). An optimisation that stopped before converging prints no final
    block; its fort.9 holds the last step, but the first block is taken, as
    before.
    """
    finals = [b for b in blocks if b['final']]
    if finals:
        return finals[-1], 'final'
    if blocks:
        return blocks[0], 'first'
    return None, None


def expansion_before(lines: List[str], stop: int) -> Optional[np.ndarray]:
    """The SUPERCEL "EXPANSION MATRIX OF PRIMITIVE CELL" printed before line ``stop``.

    Rows E1, E2, E3 as printed: the supercell vectors are E times the
    primitive ones (rows), A_super = E A_prim, so a point's fractions are
    f_super = E f_prim.
    """
    matrix = None
    for i in range(min(stop, len(lines))):
        if 'EXPANSION MATRIX OF PRIMITIVE CELL' not in lines[i]:
            continue
        rows = []
        for line in lines[i + 1:i + 6]:
            parts = line.split()
            if len(parts) == 4 and re.fullmatch(r'E\d', parts[0]):
                try:
                    rows.append([float(x) for x in parts[1:]])
                except ValueError:
                    break
            elif rows:
                break
        if len(rows) == 3:
            matrix = np.array(rows)
    return matrix


def _ran_scelphono(content: str) -> bool:
    """True for a SCELPHONO supercell rather than a SUPERCEL one.

    CRYSTAL23 does not echo the input keywords, and prints the same
    "EXPANSION MATRIX OF PRIMITIVE CELL" for both; SCELPHONO (manual sec.
    4.21 p.73) then reorders the atoms, small cell first, under this line
    (measured on CRYSTAL23 with a TESTGEOM deck).
    """
    return 'ATOMS IN THE SUPERCELL REORDERED FOR PHONON CALCULATION' in content


def parse_crystal_structure(output_file: str) -> Optional[Dict[str, Any]]:
    """
    Extract full crystal structure from CRYSTAL output file.

    The cell and the atoms are read from one computed geometry block: the
    final optimised geometry after an optimisation, otherwise the first
    computed geometry (the cell fort.9 holds; a SUPERCEL supercell, say).
    When the output has no such block, or its atom count does not match the
    atoms read, the earlier parser is used with a warning.

    Args:
        output_file: Path to CRYSTAL .out file

    Returns:
        Dictionary with:
        - 'cell': 3x3 numpy array of lattice vectors
        - 'positions': Nx3 numpy array of fractional coordinates
        - 'numbers': List of atomic numbers
        - 'geometry': 'final' or 'first' (the block read), 'legacy' for
          the earlier parser
        - 'supercell': 'none', 'supercel' or 'scelphono'
        - 'expansion': the expansion matrix E (A_super = E A_prim) or None
        - 'W': the printed primitive-to-crystallographic matrix or None
        Or None if parsing fails
    """
    if not Path(output_file).exists():
        print(f"Error: Output file not found: {output_file}")
        return None

    with open(output_file, 'r') as f:
        content = f.read()
    lines = content.split('\n')

    block, which = select_geometry_block(_geometry_blocks(lines))
    if block is not None and block['params'] is not None and block['n_atoms']:
        if len(block['numbers']) == block['n_atoms']:
            p = block['params']
            expansion = expansion_before(lines, block['line'])
            supercell = 'none'
            if expansion is not None:
                supercell = 'scelphono' if _ran_scelphono(content) else 'supercel'
            return {
                'cell': cell_params_to_vectors(p['a'], p['b'], p['c'],
                                               p['alpha'], p['beta'], p['gamma']),
                'positions': np.array(block['positions']),
                'numbers': block['numbers'],
                'geometry': which,
                'supercell': supercell,
                'expansion': expansion,
                'W': block['W'],
            }
        print(f"WARNING: the {which} geometry block of {Path(output_file).name} lists "
              f"{len(block['numbers'])} atoms where CRYSTAL prints {block['n_atoms']}; "
              f"reading the structure as before")
    else:
        print(f"WARNING: no computed geometry block found in {Path(output_file).name}; "
              f"reading the structure as before")
    structure = _legacy_structure(content, lines)
    if structure is not None:
        structure.update({'geometry': 'legacy', 'supercell': 'none',
                          'expansion': None, 'W': None})
    return structure


def _legacy_structure(content: str, lines: List[str]) -> Optional[Dict[str, Any]]:
    """The structure as read before computed geometry blocks were used."""
    # Extract cell parameters
    cell_params = _extract_cell_parameters(content, lines)
    if cell_params is None:
        print("Error: Could not extract cell parameters")
        return None

    # Extract atomic positions
    positions, numbers = _extract_atomic_positions(content, lines)
    if positions is None or len(positions) == 0:
        print("Error: Could not extract atomic positions")
        return None

    # Convert cell parameters to vectors
    cell = cell_params_to_vectors(
        cell_params['a'], cell_params['b'], cell_params['c'],
        cell_params['alpha'], cell_params['beta'], cell_params['gamma']
    )

    return {
        'cell': cell,
        'positions': np.array(positions),
        'numbers': numbers
    }


def _extract_cell_parameters(content: str, lines: List[str]) -> Optional[Dict[str, float]]:
    """Extract lattice parameters from CRYSTAL output."""

    # Try PRIMITIVE CELL section first (most common in optimized structures)
    params_match = re.search(
        r'PRIMITIVE CELL.*?LATTICE PARAMETERS.*?\n\s*A\s+B\s+C\s+ALPHA\s+BETA\s+GAMMA.*?\n\s*'
        r'([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)',
        content, re.DOTALL | re.IGNORECASE
    )

    if params_match:
        return {
            'a': float(params_match.group(1)),
            'b': float(params_match.group(2)),
            'c': float(params_match.group(3)),
            'alpha': float(params_match.group(4)),
            'beta': float(params_match.group(5)),
            'gamma': float(params_match.group(6))
        }

    # Try generic LATTICE PARAMETERS
    params_match = re.search(
        r'LATTICE PARAMETERS.*?\n\s*A\s+B\s+C\s+ALPHA\s+BETA\s+GAMMA.*?\n\s*'
        r'([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)',
        content, re.DOTALL | re.IGNORECASE
    )

    if params_match:
        return {
            'a': float(params_match.group(1)),
            'b': float(params_match.group(2)),
            'c': float(params_match.group(3)),
            'alpha': float(params_match.group(4)),
            'beta': float(params_match.group(5)),
            'gamma': float(params_match.group(6))
        }

    return None


def _extract_atomic_positions(content: str, lines: List[str]) -> Tuple[Optional[List], Optional[List]]:
    """Extract atomic positions and numbers from CRYSTAL output."""

    positions = []
    numbers = []

    # Look for ATOMS IN THE ASYMMETRIC UNIT section
    # Format: index T/F atom_num symbol x y z
    in_atom_section = False
    past_header = False

    for i, line in enumerate(lines):
        if 'ATOMS IN THE ASYMMETRIC UNIT' in line:
            in_atom_section = True
            past_header = False
            continue

        if in_atom_section:
            # Skip header lines
            if '***' in line:
                past_header = True
                continue

            if not past_header:
                continue

            # Check for end of section
            if not line.strip() or 'INFORMATION' in line or 'TRANSFORMATION' in line:
                break

            parts = line.split()
            if len(parts) >= 7 and parts[1] in ['T', 'F']:
                try:
                    atom_num = int(parts[2])
                    x = float(parts[4])
                    y = float(parts[5])
                    z = float(parts[6])

                    numbers.append(atom_num)
                    positions.append([x, y, z])
                except (ValueError, IndexError):
                    continue

    if not positions:
        # Try alternative format for optimized geometries
        return _extract_positions_alternative(content, lines)

    return positions, numbers


def _extract_positions_alternative(content: str, lines: List[str]) -> Tuple[Optional[List], Optional[List]]:
    """Alternative position extraction for different output formats."""

    positions = []
    numbers = []

    # Look for FINAL OPTIMIZED GEOMETRY section
    final_geom_idx = None
    for i, line in enumerate(lines):
        if 'FINAL OPTIMIZED GEOMETRY' in line:
            final_geom_idx = i
            break

    if final_geom_idx is None:
        # Try GEOMETRY FOR WAVE FUNCTION
        for i, line in enumerate(lines):
            if 'GEOMETRY FOR WAVE FUNCTION' in line:
                final_geom_idx = i
                break

    if final_geom_idx is None:
        return None, None

    # Search for coordinates after this marker
    in_atom_section = False
    for i in range(final_geom_idx, min(final_geom_idx + 500, len(lines))):
        line = lines[i]

        if 'ATOMS IN THE ASYMMETRIC UNIT' in line:
            in_atom_section = True
            continue

        if in_atom_section and '***' in line:
            continue

        if in_atom_section:
            if not line.strip():
                break

            parts = line.split()
            if len(parts) >= 7 and parts[1] in ['T', 'F']:
                try:
                    atom_num = int(parts[2])
                    x = float(parts[4])
                    y = float(parts[5])
                    z = float(parts[6])

                    numbers.append(atom_num)
                    positions.append([x, y, z])
                except (ValueError, IndexError):
                    continue

    return positions if positions else None, numbers if numbers else None


# =============================================================================
# SeeKPath Interface
# =============================================================================

def get_seekpath_bandpath(cell: np.ndarray, positions: np.ndarray, numbers: List[int],
                          with_time_reversal: bool = True) -> Dict[str, Any]:
    """
    Get band path from seekpath library.

    Args:
        cell: 3x3 numpy array of lattice vectors
        positions: Nx3 array of fractional coordinates
        numbers: List of atomic numbers
        with_time_reversal: Use time reversal symmetry (default True)

    Returns:
        seekpath result dictionary with keys:
        - point_coords: Dict[str, List[float]]
        - path: List[Tuple[str, str]]
        - has_inversion_symmetry: bool
        - bravais_lattice: str
        - bravais_lattice_extended: str
        - primitive_lattice: np.ndarray
        - reciprocal_primitive_lattice: np.ndarray
    """
    if not SEEKPATH_AVAILABLE:
        raise ImportError("seekpath library not installed")

    structure = (cell, positions, numbers)

    result = seekpath.get_path(
        structure,
        with_time_reversal=with_time_reversal,
        recipe='hpkot',
        threshold=1e-7,
        symprec=1e-5
    )

    return result


# =============================================================================
# From SeeK-path's cell to CRYSTAL's
# =============================================================================
#
# SeeK-path gives its points as fractions of the reciprocal vectors of ITS
# standardised primitive cell (HPKOT). BAND reads a segment end as fractions
# of the reciprocal vectors of CRYSTAL's own primitive cell (CRYSTAL23 manual
# p.309: I1/ISS b1 + I2/ISS b2 + I3/ISS b3), which is the cell given in input
# reduced by CRYSTAL's matrices (Appendix A.5). The two cells describe the
# same lattice but are not the same cell: a C-centred orthorhombic cell is
# (a/2, b/2, 0), (-a/2, b/2, 0), c in CRYSTAL and (a/2, -b/2, 0),
# (a/2, b/2, 0), c in SeeK-path; an A-centred one, a triclinic one or a
# primitive orthorhombic one not in a < b < c order also differ.
#
# With the rows of A the CRYSTAL primitive vectors and of A_sp SeeK-path's,
# A_sp = N A for an integer N of determinant 1. A point k has fractional
# coordinates f_i = k . a_i / 2pi in a basis, so f_sp = N f, and the point
# SeeK-path names is f = N^-1 f_sp in CRYSTAL's basis (as rows,
# f = f_sp N^-T). Its Cartesian vector f B = f_sp B_sp is unchanged, B and
# B_sp being the reciprocal vectors (rows) with B = N^T B_sp.

def crystal_cell_matrix(seekpath_result: Dict[str, Any], cell) -> np.ndarray:
    """N with SeeK-path's primitive vectors = N @ ``cell`` (vectors as rows).

    ``cell`` is the cell SeeK-path was given (CRYSTAL's primitive cell, in
    any orientation). SeeK-path returns its cell rotated by
    ``rotation_matrix`` R, so A_sp = N (A R^T). N is rounded to integers
    when it is one; it is not when SeeK-path found a smaller primitive cell
    than the one given (a supercell given as P1, say).
    """
    asp = np.array(seekpath_result['primitive_lattice'], dtype=float)
    rot = np.array(seekpath_result['rotation_matrix'], dtype=float)
    n = asp @ np.linalg.inv(np.array(cell, dtype=float) @ rot.T)
    rounded = np.round(n)
    if np.allclose(n, rounded, atol=1e-3) and abs(abs(np.linalg.det(rounded)) - 1) < 1e-9:
        return rounded
    if not np.all(np.isfinite(n)) or abs(np.linalg.det(n)) < 1e-6:
        raise ValueError("SeeK-path's primitive cell cannot be related to the CRYSTAL cell")
    return n


def _same_path_by_symmetry(original: Dict[str, Any], moved: Dict[str, Any],
                           path, rotations, time_reversal: bool) -> bool:
    """True when every original segment is a symmetry image of its moved one.

    A rotation W of the crystal (fractional, x' = W x) takes a point with
    fractional row f to f W^-1; over the whole group that is the set
    {f W}. Time reversal adds -f, and a reciprocal lattice vector G added
    to both ends of a segment changes nothing. When for each segment some
    operation and G map its two moved ends onto its two original ends, the
    original segment runs through points equivalent to those of the right
    one, one for one, and has the same bands.
    """
    signs = (1, -1) if time_reversal else (1,)
    ops = [np.array(w, dtype=float) for w in rotations]
    for a, b in path:
        fa, fb = np.array(moved[a], dtype=float), np.array(moved[b], dtype=float)
        oa, ob = np.array(original[a], dtype=float), np.array(original[b], dtype=float)
        found = False
        for w in ops:
            for s in signs:
                da = s * (fa @ w) - oa
                db = s * (fb @ w) - ob
                g = np.round(da)
                if np.allclose(da, g, atol=1e-8) and np.allclose(db, g, atol=1e-8):
                    found = True
                    break
            if found:
                break
        if not found:
            return False
    return True


def to_crystal_basis(seekpath_result: Dict[str, Any], cell, positions, numbers,
                     with_time_reversal: bool = True) -> Dict[str, Any]:
    """SeeK-path's result with its points given in CRYSTAL's reciprocal basis.

    ``cell``, ``positions`` and ``numbers`` are the structure SeeK-path was
    given: CRYSTAL's primitive cell and the atoms in it. Each point becomes
    N^-1 f_sp (see crystal_cell_matrix) and ``reciprocal_primitive_lattice``
    becomes CRYSTAL's (B = N^T B_sp), so every point keeps its Cartesian
    vector. When each segment of SeeK-path's own numbers, read in CRYSTAL's
    basis, is a symmetry image of the right segment (a cubic cell whose axes
    SeeK-path only relabels, say), those numbers already give the right
    bands and are kept as they are, so such decks do not change.
    """
    n = crystal_cell_matrix(seekpath_result, cell)
    m = np.linalg.inv(n)
    integer = np.allclose(m, np.round(m), atol=1e-9)
    if integer:
        m = np.round(m)
    original = seekpath_result['point_coords']
    moved = {}
    for label, coords in original.items():
        f = m @ np.array(coords, dtype=float)
        f[np.abs(f) < 1e-12] = 0.0
        moved[label] = [float(x) for x in f]

    out = dict(seekpath_result)
    out['crystal_cell_matrix'] = n.tolist()
    labels = _path_points(seekpath_result)
    if all(np.allclose(moved[lab], original[lab], atol=1e-8) for lab in labels):
        return out
    try:
        import spglib
        rotations = spglib.get_symmetry(
            (np.array(cell, dtype=float), np.array(positions, dtype=float), list(numbers)),
            symprec=1e-5)['rotations']
    except Exception:
        rotations = [np.eye(3, dtype=int)]
    if _same_path_by_symmetry(original, moved, seekpath_result['path'], rotations,
                              with_time_reversal):
        return out

    out['point_coords'] = moved
    recip = seekpath_result.get('reciprocal_primitive_lattice')
    if recip is not None:
        out['reciprocal_primitive_lattice'] = (n.T @ np.array(recip, dtype=float)).tolist()
    print("  Note: SeeK-path's points rewritten in the reciprocal basis of CRYSTAL's "
          "primitive cell (SeeK-path's standard cell differs)"
          + ("" if integer else "; SeeK-path found a smaller primitive cell"))
    return out


# =============================================================================
# Format Conversion
# =============================================================================

def convert_seekpath_label(label: str) -> str:
    """
    Convert seekpath label to CRYSTAL-compatible ASCII.

    Args:
        label: SeeKPath label (e.g., 'GAMMA', 'SIGMA_0')

    Returns:
        CRYSTAL-compatible label

    Note:
        We keep GAMMA as-is (not abbreviated to 'G') for consistency with other
        Greek letters (SIGMA, LAMBDA, DELTA) and to avoid confusion with the
        parametric 'G' point that exists in tI2 and mC2 lattices.
    """
    # No conversions needed - keep all labels as-is
    # GAMMA stays GAMMA (like SIGMA stays SIGMA)
    # This avoids collision with the 'G' point in tI2/mC2 lattices
    return label


# BAND gives each segment end as integers in units of 1/ISS (CRYSTAL23 manual
# p.309), and fort.25 writes them again with format 6I3 (App. D, p.447): an
# end coordinate of 1000 or more prints there as asterisks.
FORT25_MAX_ISS = 999

# A lattice-dependent point may be moved to the nearest k/ISS by at most half
# the spacing of the k points CRYSTAL computes along the path (manual p.309,
# note 4: the density depends only on NSUB, the total number of points) at
# this NSUB - MACE's largest default (workflow BAND steps).
REFERENCE_NSUB = 10000


def parametric_point_names(ext_bravais: str) -> Optional[set]:
    """Names of the lattice-dependent points of a SeeK-path lattice.

    SeeK-path defines each point in hpkot/band_path_data/<lattice>/points.txt
    either by fixed fractions ("1/2 0 1/2") or by k-vector parameters
    ("X X 0"); the latter are parametric. None when the definitions cannot
    be read.
    """
    try:
        from seekpath.hpkot.tools import get_path_data
        _, points_def, _ = get_path_data(ext_bravais[:3])
    except Exception:
        return None
    return {name for name, xyz in points_def.items()
            if any(ch.isalpha() for expr in xyz for ch in expr)}


def _fixed_fraction(value: float) -> Optional[Fraction]:
    frac = Fraction(value).limit_denominator(100)
    return frac if abs(float(frac) - value) < 1e-9 else None


def _path_points(seekpath_result: Dict[str, Any]) -> List[str]:
    seen = []
    for pair in seekpath_result['path']:
        for label in pair:
            if label not in seen:
                seen.append(label)
    return seen


def _is_parametric(label: str, coords, parametric: Optional[set]) -> bool:
    if parametric is not None and label.rstrip("'") not in parametric:
        # SeeK-path fixes it; a coordinate that is not a small fraction
        # would still have to be approximated
        return any(_fixed_fraction(c) is None for c in coords)
    if parametric is not None:
        return True
    # Without SeeK-path's definitions: fixed when every coordinate is a
    # small fraction
    return any(_fixed_fraction(c) is None for c in coords)


def rational_iss_step(seekpath_result: Dict[str, Any], parametric: Optional[set] = None) -> int:
    """Lowest common denominator of the fixed (rational) points on the path."""
    from math import gcd
    if parametric is None:
        parametric = parametric_point_names(seekpath_result.get('bravais_lattice_extended', ''))
    step = 1
    for label in _path_points(seekpath_result):
        coords = seekpath_result['point_coords'][label]
        if _is_parametric(label, coords, parametric):
            continue
        for c in coords:
            den = _fixed_fraction(c).denominator
            step = step * den // gcd(step, den)
    return step


def choose_band_iss(seekpath_result: Dict[str, Any],
                    parametric: Optional[set] = None) -> Tuple[int, float]:
    """ISS for a SeeK-path band path, and how far it moves the points.

    The fixed points of the path set ISS to the lowest common multiple of
    their denominators (or its first multiple from 4) and are written exactly. Lattice-dependent points are
    written as k/ISS with ISS a multiple of that, at most FORT25_MAX_ISS,
    taking the ISS that moves them least, when every point then moves by at
    most half the path's k-point spacing at REFERENCE_NSUB. Otherwise they are
    held exactly as before (denominators up to 10**6). The displacement is
    measured in Cartesian reciprocal space (1/Angstrom, 2*pi included) with
    seekpath's reciprocal primitive lattice, or in fractional units without
    one. ``parametric`` names the lattice-dependent points when the caller
    knows them (the static route without SeeK-path's definitions).
    """
    point_coords = seekpath_result['point_coords']
    if parametric is None:
        parametric = parametric_point_names(seekpath_result.get('bravais_lattice_extended', ''))
    step = rational_iss_step(seekpath_result, parametric)
    # d3_kpoints.scale_kpoint_segments, which every deck's path goes through,
    # never writes an ISS below 4; start from the first multiple of the
    # fixed points' denominators that it keeps.
    smallest = -(-4 // step) * step
    moving = [point_coords[label] for label in _path_points(seekpath_result)
              if _is_parametric(label, point_coords[label], parametric)]
    if not moving:
        return smallest, 0.0

    recip = np.array(seekpath_result.get('reciprocal_primitive_lattice', np.eye(3)), dtype=float)
    points = np.array(moving, dtype=float)
    length = sum(float(np.linalg.norm((np.array(point_coords[b]) - np.array(point_coords[a])) @ recip))
                 for a, b in seekpath_result['path'])
    tolerance = 0.5 * length / REFERENCE_NSUB

    best_iss, best_shift = None, None
    for iss in range(smallest, FORT25_MAX_ISS + 1, step):
        shift = float(np.linalg.norm((np.round(points * iss) / iss - points) @ recip, axis=1).max())
        if best_shift is None or shift < best_shift - 1e-15:
            best_iss, best_shift = iss, shift
    if best_iss is not None and best_shift <= tolerance:
        return best_iss, best_shift

    # Keep the exact representation: the largest denominator of the moving
    # coordinates (below 10**6), raised to a multiple of the fixed points'.
    largest = max(Fraction(float(c)).limit_denominator(1000000).denominator
                  for xyz in moving for c in xyz)
    iss = -(-max(largest, smallest) // step) * step
    shift = float(np.linalg.norm((np.round(points * iss) / iss - points) @ recip, axis=1).max())
    return iss, shift


def convert_to_mace_format(seekpath_result: Dict[str, Any],
                           shrink_factor: int = 16) -> Tuple[List[List[int]], List[str], Dict[str, Any]]:
    """
    Convert seekpath output to MACE's expected format.

    Args:
        seekpath_result: Result from get_seekpath_bandpath()
        shrink_factor: no longer used; ISS is chosen from the path's points
            (choose_band_iss). Kept for callers.

    Returns:
        Tuple of:
        - segments: List of [x1,y1,z1, x2,y2,z2] as integers
        - labels: List of k-point labels with '|' discontinuity markers
        - kpath_info: Dict with metadata
    """
    point_coords = seekpath_result['point_coords']
    path = seekpath_result['path']

    # ISS as small as the fixed points allow, and within fort.25's three
    # digits when the lattice-dependent points can be moved by less than
    # half a k-point spacing.
    shrink_factor, point_shift = choose_band_iss(seekpath_result)
    if point_shift > 0:
        print(f"  Note: ISS {shrink_factor}; lattice-dependent points moved by at most "
              f"{point_shift:.2e} to the nearest 1/{shrink_factor}")

    segments = []
    labels = []
    prev_end_label = None

    for i, (start_label, end_label) in enumerate(path):
        # Check for discontinuity (current start doesn't match previous end)
        if prev_end_label is not None and start_label != prev_end_label:
            labels.append('|')

        # Add start label only if first segment or after discontinuity
        if i == 0 or (labels and labels[-1] == '|'):
            labels.append(convert_seekpath_label(start_label))

        # Add end label
        labels.append(convert_seekpath_label(end_label))

        # Get coordinates
        start_coord = point_coords[start_label]
        end_coord = point_coords[end_label]

        # Scale to integers
        scaled_segment = [
            int(round(start_coord[0] * shrink_factor)),
            int(round(start_coord[1] * shrink_factor)),
            int(round(start_coord[2] * shrink_factor)),
            int(round(end_coord[0] * shrink_factor)),
            int(round(end_coord[1] * shrink_factor)),
            int(round(end_coord[2] * shrink_factor)),
        ]
        segments.append(scaled_segment)

        prev_end_label = end_label

    # Build kpath info
    kpath_info = {
        'source': 'seekpath',
        'bravais_lattice': seekpath_result.get('bravais_lattice', 'unknown'),
        'bravais_lattice_extended': seekpath_result.get('bravais_lattice_extended', 'unknown'),
        'has_inversion': seekpath_result.get('has_inversion_symmetry', False),
        'shrink_factor': shrink_factor,
        'max_point_shift': point_shift,
        'n_segments': len(segments),
        'n_discontinuities': labels.count('|')
    }

    return segments, labels, kpath_info


def calculate_minimum_shrink(point_coords: Dict[str, List[float]], max_denom: int = 1000000) -> int:
    """
    Calculate minimum shrink factor to represent all k-point coordinates as integers.

    For band structure D3 files, the SHRINK factor is only used for coordinate
    representation and has NO computational cost (unlike SCF SHRINK). Therefore,
    we use a very high max_denom to ensure exact integer representation of all
    parametric k-points.

    The limit_denominator() call is still needed to avoid floating-point artifacts
    (e.g., Fraction(0.1) = 3602879701896397/36028797018963968 due to binary
    representation). A limit of 1,000,000 is effectively unlimited for any
    reasonable crystallographic k-point coordinate.

    Args:
        point_coords: Dict mapping labels to [x, y, z] coordinates
        max_denom: Maximum denominator for fraction approximation (default 1000000)

    Returns:
        Minimum shrink factor (always even, minimum 8)
    """
    required_denom = 1

    for label, coords in point_coords.items():
        for coord in coords:
            if abs(coord) > 1e-10:
                # Find exact fraction representation (limited to avoid FP artifacts)
                try:
                    frac = Fraction(coord).limit_denominator(max_denom)
                    required_denom = max(required_denom, frac.denominator)
                except (ValueError, ZeroDivisionError):
                    pass

    # Round up to even number
    shrink = required_denom
    if shrink % 2 == 1:
        shrink += 1

    # Minimum of 8 for robustness
    return max(8, shrink)


# =============================================================================
# Main Interface
# =============================================================================

def get_accurate_bandpath(
    output_file: str,
    shrink_factor: int = 16,
    with_time_reversal: bool = True
) -> Tuple[List[List[int]], List[str], Dict[str, Any]]:
    """
    Get accurate band path using seekpath library.

    This is the main interface function. It:
    1. Parses the CRYSTAL output file to extract structure
    2. Calls seekpath to get the correct band path
    3. Converts to MACE format with proper discontinuity markers

    Args:
        output_file: Path to CRYSTAL .out file
        shrink_factor: SHRINK factor for integer coordinates (may be auto-adjusted)
        with_time_reversal: Whether to use time reversal symmetry

    Returns:
        Tuple of:
        - segments: List of [x1,y1,z1, x2,y2,z2] integer coordinates
        - labels: List of k-point labels with '|' discontinuity markers
        - kpath_info: Dict with metadata (bravais_lattice, has_inversion, etc.)

    Raises:
        ImportError: If seekpath is not installed
        ValueError: If structure cannot be parsed
    """
    if not SEEKPATH_AVAILABLE:
        raise ImportError(
            "seekpath library not installed. Install with: pip install seekpath"
        )

    # Parse structure from CRYSTAL output
    print(f"  Parsing structure from: {Path(output_file).name}")
    structure = parse_crystal_structure(output_file)

    if structure is None:
        raise ValueError(f"Could not parse structure from {output_file}")

    print(f"  Found {len(structure['numbers'])} atoms")

    # Get band path from seekpath
    print("  Computing band path with seekpath...")
    result = get_seekpath_bandpath(
        structure['cell'],
        structure['positions'],
        structure['numbers'],
        with_time_reversal=with_time_reversal
    )
    # SeeK-path's points are in its own cell's reciprocal basis; BAND reads
    # them in CRYSTAL's.
    result = to_crystal_basis(result, structure['cell'], structure['positions'],
                              structure['numbers'], with_time_reversal)

    print(f"  Bravais lattice: {result.get('bravais_lattice_extended', 'unknown')}")
    print(f"  Inversion symmetry: {result.get('has_inversion_symmetry', False)}")
    print(f"  Path segments: {len(result['path'])}")

    # Convert to MACE format
    segments, labels, kpath_info = convert_to_mace_format(result, shrink_factor)

    print(f"  Discontinuities detected: {kpath_info['n_discontinuities']}")
    # The cell the path is in: a supercell's (SUPERCEL or SCELPHONO) when
    # the output computed one, with its expansion of the primitive cell.
    kpath_info['supercell'] = structure.get('supercell', 'none')
    if structure.get('expansion') is not None:
        kpath_info['expansion'] = np.array(structure['expansion']).tolist()

    return segments, labels, kpath_info


def get_bandpath_labels_only(
    output_file: str,
    with_time_reversal: bool = True
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Get band path labels only (for CRYSTAL label-based BAND input).

    Args:
        output_file: Path to CRYSTAL .out file
        with_time_reversal: Whether to use time reversal symmetry

    Returns:
        Tuple of:
        - labels: List of k-point labels with '|' discontinuity markers
        - kpath_info: Dict with metadata
    """
    segments, labels, kpath_info = get_accurate_bandpath(
        output_file, shrink_factor=16, with_time_reversal=with_time_reversal
    )
    return labels, kpath_info


def get_point_coordinates(
    output_file: str,
    with_time_reversal: bool = True
) -> Tuple[Dict[str, List[float]], Dict[str, Any]]:
    """
    Get k-point coordinates dictionary.

    Useful for debugging or custom path construction.

    Args:
        output_file: Path to CRYSTAL .out file
        with_time_reversal: Whether to use time reversal symmetry

    Returns:
        Tuple of:
        - point_coords: Dict mapping labels to [x, y, z] fractional coordinates
        - kpath_info: Dict with metadata
    """
    if not SEEKPATH_AVAILABLE:
        raise ImportError("seekpath library not installed")

    structure = parse_crystal_structure(output_file)
    if structure is None:
        raise ValueError(f"Could not parse structure from {output_file}")

    result = get_seekpath_bandpath(
        structure['cell'],
        structure['positions'],
        structure['numbers'],
        with_time_reversal=with_time_reversal
    )
    result = to_crystal_basis(result, structure['cell'], structure['positions'],
                              structure['numbers'], with_time_reversal)

    kpath_info = {
        'source': 'seekpath',
        'bravais_lattice': result.get('bravais_lattice', 'unknown'),
        'bravais_lattice_extended': result.get('bravais_lattice_extended', 'unknown'),
        'has_inversion': result.get('has_inversion_symmetry', False),
    }

    return result['point_coords'], kpath_info


# =============================================================================
# Testing / Standalone Usage
# =============================================================================

def print_bandpath_info(output_file: str):
    """
    Print detailed band path information for a CRYSTAL output file.

    Useful for debugging and verification.
    """
    print(f"\n{'='*60}")
    print(f"Band Path Analysis: {Path(output_file).name}")
    print('='*60)

    try:
        segments, labels, kpath_info = get_accurate_bandpath(output_file)

        print(f"\nBravais Lattice: {kpath_info['bravais_lattice_extended']}")
        print(f"Has Inversion: {kpath_info['has_inversion']}")
        print(f"Shrink Factor: {kpath_info['shrink_factor']}")

        print(f"\nPath Labels ({len([l for l in labels if l != '|'])} points, "
              f"{kpath_info['n_discontinuities']} discontinuities):")
        print("  " + "-".join(labels))

        print(f"\nSegments ({len(segments)}):")
        for i, seg in enumerate(segments):
            print(f"  {i+1}: [{seg[0]:3d}, {seg[1]:3d}, {seg[2]:3d}] -> "
                  f"[{seg[3]:3d}, {seg[4]:3d}, {seg[5]:3d}]")

        # Also print fractional coordinates
        point_coords, _ = get_point_coordinates(output_file)
        print(f"\nK-Point Coordinates (fractional):")
        for label, coords in sorted(point_coords.items()):
            print(f"  {label:12s}: [{coords[0]:8.5f}, {coords[1]:8.5f}, {coords[2]:8.5f}]")

    except ImportError as e:
        print(f"\nError: {e}")
    except ValueError as e:
        print(f"\nError: {e}")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python seekpath_interface.py <crystal_output.out>")
        print("\nThis will analyze the band path for the given CRYSTAL output file.")
        sys.exit(1)

    for output_file in sys.argv[1:]:
        print_bandpath_info(output_file)
