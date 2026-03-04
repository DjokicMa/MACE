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

def parse_crystal_structure(output_file: str) -> Optional[Dict[str, Any]]:
    """
    Extract full crystal structure from CRYSTAL output file.

    Args:
        output_file: Path to CRYSTAL .out file

    Returns:
        Dictionary with:
        - 'cell': 3x3 numpy array of lattice vectors
        - 'positions': Nx3 numpy array of fractional coordinates
        - 'numbers': List of atomic numbers
        Or None if parsing fails
    """
    if not Path(output_file).exists():
        print(f"Error: Output file not found: {output_file}")
        return None

    with open(output_file, 'r') as f:
        content = f.read()
    lines = content.split('\n')

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


def convert_to_mace_format(seekpath_result: Dict[str, Any],
                           shrink_factor: int = 16) -> Tuple[List[List[int]], List[str], Dict[str, Any]]:
    """
    Convert seekpath output to MACE's expected format.

    Args:
        seekpath_result: Result from get_seekpath_bandpath()
        shrink_factor: SHRINK factor for scaling to integers

    Returns:
        Tuple of:
        - segments: List of [x1,y1,z1, x2,y2,z2] as integers
        - labels: List of k-point labels with '|' discontinuity markers
        - kpath_info: Dict with metadata
    """
    point_coords = seekpath_result['point_coords']
    path = seekpath_result['path']

    # Calculate minimum shrink if needed
    min_shrink = calculate_minimum_shrink(point_coords)
    if shrink_factor < min_shrink:
        print(f"  Note: Adjusting shrink factor from {shrink_factor} to {min_shrink} for accuracy")
        shrink_factor = min_shrink

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

    print(f"  Bravais lattice: {result.get('bravais_lattice_extended', 'unknown')}")
    print(f"  Inversion symmetry: {result.get('has_inversion_symmetry', False)}")
    print(f"  Path segments: {len(result['path'])}")

    # Convert to MACE format
    segments, labels, kpath_info = convert_to_mace_format(result, shrink_factor)

    print(f"  Discontinuities detected: {kpath_info['n_discontinuities']}")

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
