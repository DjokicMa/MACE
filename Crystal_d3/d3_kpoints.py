"""
K-point Data and Path Generation for CRYSTAL D3 Calculations

Author: Marcus Djokic
Institution: Michigan State University, Mendoza Group

This module contains:
- K-point coordinate dictionaries for all crystal systems
- Band path definitions and templates
- Path generation functions for different formats (labels, vectors, literature, SeeK-path)
- Crystal system classification utilities

Based on:
- CRYSTAL Tables 14.1 and 14.2 for k-point coordinates
- Setyawan & Curtarolo (2010) for comprehensive paths
- SeeK-path methodology for extended Bravais lattice paths
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import re
from pathlib import Path
import numpy as np


# Centrosymmetric space groups (with inversion symmetry)
CENTROSYMMETRIC_SPACE_GROUPS = {
    2, 10, 11, 12, 13, 14, 15, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 
    63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 83, 84, 85, 86, 87, 88, 123, 124, 125, 126, 
    127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 147, 148, 
    162, 163, 164, 165, 166, 167, 175, 176, 191, 192, 193, 194, 200, 201, 202, 203, 204, 205, 
    206, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230
}


# CRYSTAL-supported high-symmetry k-point labels from Tables 14.1 and 14.2
CRYSTAL_KPOINT_LABELS = {
    "P Cubic": {
        "G": "Γ (0, 0, 0)",
        "M": "(1/2, 1/2, 0)",
        "R": "(1/2, 1/2, 1/2)",
        "X": "(0, 1/2, 0)"
    },
    "FC Cubic": {
        "G": "Γ (0, 0, 0)",
        "X": "(1/2, 0, 1/2)",
        "L": "(1/2, 1/2, 1/2)",
        "W": "(1/2, 1/4, 3/4)"
    },
    "BC Cubic": {
        "G": "Γ (0, 0, 0)",
        "H": "(1/2, -1/2, 1/2)",
        "P": "(1/4, 1/4, 1/4)",
        "N": "(0, 0, 1/2)"
    },
    "Hexagonal or P Trigonal": {
        "G": "Γ (0, 0, 0)",
        "M": "(1/2, 0, 0)",
        "K": "(1/3, 1/3, 0)",
        "A": "(0, 0, 1/2)",
        "L": "(1/2, 0, 1/2)",
        "H": "(1/3, 1/3, 1/2)"
    },
    "Rhombohedral (R Trigonal)": {
        "G": "Γ (0, 0, 0)",
        "T": "(1/2, 1/2, -1/2)",
        "F": "(0, 1/2, 1/2)",
        "L": "(0, 0, 1/2)"
    },
    "P Monoclinic": {
        "G": "Γ (0, 0, 0)",
        "A": "(1/2, -1/2, 0)",
        "B": "(1/2, 0, 0)",
        "C": "(0, 1/2, 1/2)",
        "D": "(1/2, 0, 1/2)",
        "E": "(1/2, -1/2, 1/2)",
        "Y": "(0, 1/2, 0)",
        "Z": "(0, 0, 1/2)"
    },
    "AC Monoclinic": {
        "G": "Γ (0, 0, 0)",
        "A": "(1/2, 0, 0)",
        "Y": "(0, 1/2, 1/2)",
        "M": "(1/2, 1/2, 1/2)"
    },
    "P Orthorhombic": {
        "G": "Γ (0, 0, 0)",
        "S": "(1/2, 1/2, 0)",
        "T": "(0, 1/2, 1/2)",
        "U": "(1/2, 0, 1/2)",
        "R": "(1/2, 1/2, 1/2)",
        "X": "(1/2, 0, 0)",
        "Y": "(0, 1/2, 0)",
        "Z": "(0, 0, 1/2)"
    },
    "FC Orthorhombic": {
        "G": "Γ (0, 0, 0)",
        "Z": "(1/2, 1/2, 0)",
        "Y": "(1/2, 0, 1/2)",
        "T": "(1, 1/2, 1/2)"
    },
    "AC Orthorhombic": {
        "G": "Γ (0, 0, 0)",
        "S": "(0, 1/2, 0)",
        "T": "(1/2, 1/2, 1/2)",
        "R": "(0, 1/2, 1/2)",
        "Y": "(1/2, 1/2, 0)",
        "Z": "(0, 0, 1/2)"
    },
    "BC Orthorhombic": {
        "G": "Γ (0, 0, 0)",
        "S": "(1/2, 0, 0)",
        "T": "(0, 0, 1/2)",
        "R": "(0, 1/2, 0)",
        "X": "(1/2, -1/2, 1/2)",
        "W": "(1/4, 1/4, 1/4)"
    },
    "P Tetragonal": {
        "G": "Γ (0, 0, 0)",
        "M": "(1/2, 1/2, 0)",
        "R": "(0, 1/2, 1/2)",
        "A": "(1/2, 1/2, 1/2)",
        "X": "(0, 1/2, 0)",
        "Z": "(0, 0, 1/2)"
    },
    "BC Tetragonal": {
        "G": "Γ (0, 0, 0)",
        "M": "(1/2, 1/2, -1/2)",
        "P": "(1/2, 1/2, 1/2)",
        "X": "(0, 0, 1/2)"
    }
}


def get_crystal_supported_labels(crystal_system: str, lattice_type: str = "P") -> Dict[str, str]:
    """
    Get CRYSTAL-supported k-point labels for a given crystal system and lattice type.
    
    Args:
        crystal_system: Crystal system (CUBIC, HEXAGONAL, TETRAGONAL, etc.)
        lattice_type: Lattice centering (P, F, I, C, R, etc.)
        
    Returns:
        Dictionary of label: description
    """
    # Map crystal system and lattice type to CRYSTAL label key
    system_map = {
        ("CUBIC", "P"): "P Cubic",
        ("CUBIC", "F"): "FC Cubic",
        ("CUBIC", "I"): "BC Cubic",
        ("HEXAGONAL", "P"): "Hexagonal or P Trigonal",
        ("TRIGONAL", "P"): "Hexagonal or P Trigonal",
        ("TRIGONAL", "R"): "Rhombohedral (R Trigonal)",
        ("MONOCLINIC", "P"): "P Monoclinic",
        ("MONOCLINIC", "C"): "AC Monoclinic",
        ("MONOCLINIC", "A"): "AC Monoclinic",
        ("ORTHORHOMBIC", "P"): "P Orthorhombic",
        ("ORTHORHOMBIC", "F"): "FC Orthorhombic",
        ("ORTHORHOMBIC", "C"): "AC Orthorhombic",
        ("ORTHORHOMBIC", "A"): "AC Orthorhombic",
        ("ORTHORHOMBIC", "I"): "BC Orthorhombic",
        ("TETRAGONAL", "P"): "P Tetragonal",
        ("TETRAGONAL", "I"): "BC Tetragonal",
    }
    
    # Normalize crystal system
    crystal_system = crystal_system.upper()
    lattice_type = lattice_type.upper()
    
    # Get the key for CRYSTAL_KPOINT_LABELS
    key = system_map.get((crystal_system, lattice_type))
    
    if key and key in CRYSTAL_KPOINT_LABELS:
        return CRYSTAL_KPOINT_LABELS[key]
    else:
        # Return default cubic labels as fallback
        return CRYSTAL_KPOINT_LABELS["P Cubic"]


# Cell parameter analysis functions for extended Bravais lattice determination
def determine_triclinic_variant(a: float, b: float, c: float, 
                               alpha: float, beta: float, gamma: float) -> str:
    """
    Determine whether triclinic cell should use aP2 or aP3.
    Based on SeeK-path's Niggli reduction criteria.
    
    Args:
        a, b, c: Lattice parameters in Angstroms
        alpha, beta, gamma: Lattice angles in degrees
        
    Returns:
        "aP2" or "aP3" based on cell analysis
    """
    # Convert to radians
    alpha_rad = np.radians(alpha)
    beta_rad = np.radians(beta) 
    gamma_rad = np.radians(gamma)
    
    # Check aP2 conditions
    # aP2 is used when all angles have cosines > 0 (all angles < 90°)
    # or all angles have cosines < 0 (all angles > 90°)
    cos_alpha = np.cos(alpha_rad)
    cos_beta = np.cos(beta_rad)
    cos_gamma = np.cos(gamma_rad)
    
    if (cos_alpha > 0 and cos_beta > 0 and cos_gamma > 0) or \
       (cos_alpha < 0 and cos_beta < 0 and cos_gamma < 0):
        return "aP2"
    else:
        return "aP3"


def determine_orthorhombic_f_variant(a: float, b: float, c: float) -> str:
    """
    Determine which oF variant based on axis lengths.
    
    The variant depends on which axis is shortest after standardization.
    
    Args:
        a, b, c: Lattice parameters in Angstroms
        
    Returns:
        "oF1", "oF2", or "oF3" based on shortest axis
    """
    # Find which axis is shortest
    axes = [(a, 'a'), (b, 'b'), (c, 'c')]
    axes.sort(key=lambda x: x[0])
    
    shortest_axis = axes[0][1]
    
    if shortest_axis == 'a':
        return "oF1"
    elif shortest_axis == 'b':
        return "oF2"
    else:  # c is shortest
        return "oF3"


def determine_orthorhombic_i_variant(a: float, b: float, c: float) -> str:
    """
    Determine which oI variant based on axis relationships.
    
    Args:
        a, b, c: Lattice parameters in Angstroms
        
    Returns:
        "oI1", "oI2", or "oI3" based on axis ordering
    """
    # Sort axes to determine relationships
    axes = [(a, 'a'), (b, 'b'), (c, 'c')]
    axes.sort(key=lambda x: x[0])
    
    # oI1: Default when axes are in standard order
    # oI2: When specific axis relationships exist
    # oI3: Special case for certain symmetries
    
    # For now, use simplified logic
    # TODO: Implement full SeeK-path criteria
    return "oI1"


def determine_orthorhombic_s_variant(a: float, b: float, c: float) -> str:
    """
    Determine which oS (base-centered orthorhombic) variant.
    
    Args:
        a, b, c: Lattice parameters in Angstroms
        
    Returns:
        "oS1" or "oS2" based on cell analysis
    """
    # oS1: Standard base-centered
    # oS2: Alternative centering with different axis relationships
    
    # Simplified logic for now
    # TODO: Implement full criteria
    return "oS1"


def determine_tetragonal_i_variant(a: float, c: float) -> str:
    """
    Determine tI variant based on c/a ratio.
    
    Args:
        a: a=b lattice parameter in Angstroms
        c: c lattice parameter in Angstroms
        
    Returns:
        "tI1" if c/a < 1, "tI2" if c/a > 1
    """
    if c/a < 1.0:
        return "tI1"
    else:
        return "tI2"


def determine_hexagonal_r_variant(a: float, c: float) -> str:
    """
    Determine hR variant based on c/a ratio in hexagonal setting.
    
    Args:
        a: a=b lattice parameter in Angstroms  
        c: c lattice parameter in Angstroms
        
    Returns:
        "hR1" or "hR2" based on c/a ratio
    """
    # hR1: Standard rhombohedral
    # hR2: Alternative with different c/a ratio
    
    # Use c/a ratio as criterion
    if c/a < np.sqrt(6):  # sqrt(6) ≈ 2.449
        return "hR1"
    else:
        return "hR2"


def determine_cubic_f_variant(sg: int) -> str:
    """
    Determine cF variant based on space group.
    
    Args:
        sg: Space group number
        
    Returns:
        "cF1" or "cF2" based on space group
    """
    # cF1: Standard FCC (e.g., Fm-3m)
    # cF2: Alternative FCC groups
    
    # Space groups with cF2
    cF2_groups = [196, 202, 203, 209, 210, 216, 219, 220]
    
    if sg in cF2_groups:
        return "cF2"
    else:
        return "cF1"


def determine_cubic_i_variant(sg: int) -> str:
    """
    Determine cI variant based on space group.
    
    Args:
        sg: Space group number
        
    Returns:
        "cI1" or "cI2" based on space group
    """
    # cI1: Standard BCC
    # cI2: Alternative BCC groups
    
    # Space groups with cI2
    cI2_groups = [199, 204, 206, 211, 214, 217, 220]
    
    if sg in cI2_groups:
        return "cI2"
    else:
        return "cI1"


def determine_monoclinic_variant(sg: int, a: float, b: float, c: float,
                                beta: float, unique_axis: str = 'b') -> str:
    """
    Determine monoclinic variant based on cell parameters and unique axis.
    
    Args:
        sg: Space group number
        a, b, c: Lattice parameters
        beta: Monoclinic angle (degrees)
        unique_axis: Which axis is unique ('b' standard, 'c' alternative)
        
    Returns:
        Monoclinic variant symbol
    """
    # For primitive monoclinic
    if sg <= 11:  # P lattice
        if unique_axis == 'b':
            # Check angle to determine mP1 vs alternatives
            if beta > 90:
                return "mP1"
            else:
                return "mP1"  # Could be different variant
        else:
            return "mP1"  # c-unique setting
    else:  # C/A centered
        return "mS1"


def extract_and_process_shrink(out_file: str, base_name: str, 
                              input_dir: Path, config: Dict[str, Any],
                              default_shrink: int = 16) -> int:
    """Extract shrink factor from D12 file or calculate from lattice parameters.
    
    Args:
        out_file: Path to output file for lattice parameter extraction
        base_name: Base name of the calculation for finding D12 file
        input_dir: Directory containing input files
        config: Configuration dictionary (may contain existing shrink value)
        default_shrink: Default shrink value if no extraction possible
        
    Returns:
        Shrink factor (always even for cleaner k-paths)
    """
    shrink = config.get("shrink", default_shrink)
    
    # Try to extract shrink from original D12 file first
    d12_path = Path(out_file).with_suffix('.d12')
    if not d12_path.exists():
        # Try to find D12 file with similar name
        d12_files = list(input_dir.glob(f"{base_name}*.d12"))
        if d12_files:
            d12_path = d12_files[0]
    
    if d12_path.exists():
        # Extract shrink from D12
        with open(d12_path, 'r') as f:
            d12_content = f.read()
        
        # Match SHRINK line and following lines
        shrink_match = re.search(r'^SHRINK\s*\n\s*(\d+)(?:\s+(\d+))?\s*\n(?:\s*(\d+)\s+(\d+)\s+(\d+))?', 
                                d12_content, re.MULTILINE)
        if shrink_match:
            first_val = int(shrink_match.group(1))
            second_val = int(shrink_match.group(2)) if shrink_match.group(2) else None
            
            if first_val == 0 and shrink_match.group(3):
                # SHRINK 0 format with k-points on next line
                k1 = int(shrink_match.group(3))
                k2 = int(shrink_match.group(4))
                k3 = int(shrink_match.group(5))
                # Use maximum k value for band structures
                original_shrink = max(k1, k2, k3)
                print(f"    Found SHRINK 0 format with k-points: {k1} {k2} {k3}")
            elif second_val is not None:
                # Standard SHRINK IS IPMG format
                original_shrink = first_val  # Use IS value
            else:
                # Single value format
                original_shrink = first_val
            
            # Check for valid shrink value
            if original_shrink <= 0:
                print(f"    Warning: Invalid shrink factor {original_shrink} in D12 file, will use calculated value")
            else:
                # Round to nearest even number for cleaner k-paths
                if original_shrink % 2 == 1:
                    shrink = original_shrink + 1
                    print(f"    Using shrink factor {shrink} (rounded from {original_shrink} to even number)")
                else:
                    shrink = original_shrink
                    print(f"    Using shrink factor {shrink} from original D12 file")
                return shrink
    
    # Try to extract SHRINK from output file before calculating from lattice
    if out_file and Path(out_file).exists():
        with open(out_file, 'r') as f:
            content = f.read()
        
        # Look for SHRINK in output file (e.g., "SHRINK. FACT.(MONKH.)   12 12 12")
        out_shrink_match = re.search(r'SHRINK\.\s*FACT\.\(MONKH\.\)\s*(\d+)\s+(\d+)\s+(\d+)', content)
        if out_shrink_match:
            k1 = int(out_shrink_match.group(1))
            k2 = int(out_shrink_match.group(2))
            k3 = int(out_shrink_match.group(3))
            original_shrink = max(k1, k2, k3)
            print(f"    Found SHRINK in output file: {k1} {k2} {k3}")
            
            if original_shrink > 0:
                # Round to nearest even number for cleaner k-paths
                if original_shrink % 2 == 1:
                    shrink = original_shrink + 1
                    print(f"    Using shrink factor {shrink} (rounded from {original_shrink} to even number)")
                else:
                    shrink = original_shrink
                    print(f"    Using shrink factor {shrink} from output file")
                return shrink
    
    # If we couldn't extract shrink and it's still default, calculate it
    if shrink == default_shrink and out_file and Path(out_file).exists():
        # Try to extract lattice parameters from output file
        with open(out_file, 'r') as f:
            content = f.read()
        
        # Look for lattice parameters
        lattice_match = re.search(r'LATTICE PARAMETERS.*?\n\s*A\s+B\s+C\s+ALPHA\s+BETA\s+GAMMA.*?\n\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)', content, re.DOTALL)
        if not lattice_match:
            # Try PRIMITIVE CELL pattern
            lattice_match = re.search(r'PRIMITIVE CELL.*?\n\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)', content, re.DOTALL)
        
        if lattice_match:
            a = float(lattice_match.group(1))
            b = float(lattice_match.group(2))
            c = float(lattice_match.group(3))
            
            # Calculate shrink factors using a*k > 60 rule for band structures
            ka = max(2, int(60.0 / a))
            kb = max(2, int(60.0 / b))
            kc = max(2, int(60.0 / c))
            
            # Use maximum for uniform sampling
            shrink = max(ka, kb, kc)
            
            # Round to nearest even number
            if shrink % 2 == 1:
                shrink = shrink + 1
            
            print(f"    Calculated shrink factor {shrink} from a*k > 60 rule (rounded to even)")
            return shrink
    
    # Ensure shrink is never 0 and is even
    if shrink <= 0:
        shrink = default_shrink
        print(f"    Warning: Invalid shrink factor detected, using default {shrink}")
    
    if shrink % 2 == 1:
        shrink = shrink + 1
        print(f"    Using shrink factor {shrink} (rounded to even number)")
    
    return shrink


def scale_kpoint_segments(frac_segments: List[List[float]], shrink: int) -> List[List[int]]:
    """Scale fractional k-point coordinates to integers based on shrink factor.
    
    Args:
        frac_segments: List of k-point segments with fractional coordinates
        shrink: Shrink factor for scaling
        
    Returns:
        List of k-point segments with integer coordinates
    """
    # Safeguard against invalid shrink values
    if shrink <= 0:
        print(f"WARNING: Invalid shrink factor {shrink}, using default 16")
        shrink = 16
    
    coord_segments = []
    for seg in frac_segments:
        scaled_seg = [int(round(coord * shrink)) for coord in seg]
        coord_segments.append(scaled_seg)
    return coord_segments


# High-symmetry point paths for different crystal systems
# Based on the d3_input templates from create_band_d3.py
BAND_PATHS = {
    # Triclinic (SG 1-2)
    "triclinic": ["V", "Y", "G", "Z", "T", "R", "G", "X", "U", "G"],
    
    # Monoclinic (SG 3-15)
    "monoclinic_simple": ["A", "G", "B", "C", "G", "D", "E", "G", "Y", "Z", "G"],
    "monoclinic_ac": ["A", "G", "Y", "M", "G"],
    
    # Orthorhombic (SG 16-74)
    "orthorhombic_simple": ["S", "G", "T", "U", "G", "R", "X", "G", "Y", "Z", "G"],
    "orthorhombic_ab": ["S", "G", "T", "R", "G", "Y", "Z", "G"],
    "orthorhombic_bc": ["S", "G", "T", "R", "G", "X", "W", "G"],
    "orthorhombic_fc": ["Z", "G", "Y", "T", "G"],
    
    # Tetragonal (SG 75-142)
    "tetragonal_simple": ["M", "G", "R", "A", "G", "X", "Z", "G"],
    "tetragonal_bc": ["M", "G", "P", "X", "G"],
    
    # Hexagonal (SG 143-194)
    "hexagonal": ["M", "G", "K", "A", "G", "L", "H", "G"],
    
    # Rhombohedral (R lattice in hexagonal setting)
    "rhombohedral": ["T", "G", "F", "L", "G"],
    
    # Cubic (SG 195-230)
    "cubic_simple": ["M", "G", "R", "X", "G"],
    "cubic_fc": ["X", "G", "L", "W", "G"],
    "cubic_bc": ["H", "G", "P", "N", "G"]
}


# K-point coordinate dictionaries for all crystal systems
# Based on standard crystallographic conventions and CRYSTAL d3 templates

KPOINT_COORDINATES = {
    # Cubic systems (from Table 14.1)
    "cubic_simple": {
        "G": [0.0, 0.0, 0.0],    # Gamma
        "M": [0.5, 0.5, 0.0],    # [1/2, 1/2, 0]
        "R": [0.5, 0.5, 0.5],    # [1/2, 1/2, 1/2]
        "X": [0.0, 0.5, 0.0],    # [0, 1/2, 0]
    },
    "cubic_fc": {
        "G": [0.0, 0.0, 0.0],    # Gamma
        "X": [0.5, 0.0, 0.5],    # [1/2, 0, 1/2]
        "L": [0.5, 0.5, 0.5],    # [1/2, 1/2, 1/2]
        "W": [0.5, 0.25, 0.75],  # [1/2, 1/4, 3/4]
        # K and U not in Table 14.1 - commonly used points
        "K": [3/8, 3/8, 3/4],    # [3/8, 3/8, 3/4]
        "U": [5/8, 0.25, 5/8],   # [5/8, 1/4, 5/8]
    },
    "cubic_bc": {
        "G": [0.0, 0.0, 0.0],    # Gamma
        "H": [0.5, -0.5, 0.5],   # [1/2, -1/2, 1/2]
        "P": [0.25, 0.25, 0.25], # [1/4, 1/4, 1/4]
        "N": [0.0, 0.0, 0.5],    # [0, 0, 1/2]
    },
    
    # Hexagonal/P Trigonal system (from Table 14.1)
    "hexagonal": {
        "G": [0.0, 0.0, 0.0],    # Gamma
        "M": [0.5, 0.0, 0.0],    # [1/2, 0, 0]
        "K": [1/3, 1/3, 0.0],    # [1/3, 1/3, 0]
        "A": [0.0, 0.0, 0.5],    # [0, 0, 1/2]
        "L": [0.5, 0.0, 0.5],    # [1/2, 0, 1/2]
        "H": [1/3, 1/3, 0.5],    # [1/3, 1/3, 1/2]
    },
    
    # Tetragonal systems (from Table 14.2)
    "tetragonal_simple": {
        "G": [0.0, 0.0, 0.0],    # Gamma
        "M": [0.5, 0.5, 0.0],    # [1/2, 1/2, 0]
        "R": [0.0, 0.5, 0.5],    # [0, 1/2, 1/2]
        "A": [0.5, 0.5, 0.5],    # [1/2, 1/2, 1/2]
        "X": [0.0, 0.5, 0.0],    # [0, 1/2, 0]
        "Z": [0.0, 0.0, 0.5],    # [0, 0, 1/2]
    },
    "tetragonal_bc": {
        "G": [0.0, 0.0, 0.0],    # Gamma
        "M": [0.5, 0.5, 0.0],    # [1/2, 1/2, 0]
        "P": [0.25, 0.25, 0.0],  # [1/4, 1/4, 0]
        "X": [0.0, 0.5, 0.0],    # [0, 1/2, 0]
        "N": [0.0, 0.5, 0.5],    # [0, 1/2, 1/2]
        "Z": [0.0, 0.0, 0.5],    # [0, 0, 1/2]
        "R": [0.0, 0.5, 0.5],    # Same as N in some conventions
    },
    
    # Orthorhombic systems (from Table 14.2)
    "orthorhombic_simple": {
        "G": [0.0, 0.0, 0.0],    # Gamma
        "S": [0.5, 0.5, 0.0],    # [1/2, 1/2, 0]
        "T": [0.0, 0.5, 0.5],    # [0, 1/2, 1/2]
        "U": [0.5, 0.0, 0.5],    # [1/2, 0, 1/2]
        "R": [0.5, 0.5, 0.5],    # [1/2, 1/2, 1/2]
        "X": [0.5, 0.0, 0.0],    # [1/2, 0, 0]
        "Y": [0.0, 0.5, 0.0],    # [0, 1/2, 0]
        "Z": [0.0, 0.0, 0.5],    # [0, 0, 1/2]
    },
    "orthorhombic_fc": {
        "G": [0.0, 0.0, 0.0],    # Gamma
        "Z": [0.5, 0.5, 0.0],    # [1/2, 1/2, 0]
        "Y": [0.5, 0.0, 0.0],    # [1/2, 0, 0]
        "T": [0.5, 0.5, 0.5],    # [1/2, 1/2, 1/2]
        "X": [0.0, 0.5, 0.0],    # [0, 1/2, 0]
        "U": [0.0, 0.5, 0.5],    # [0, 1/2, 1/2]
        "S": [0.5, 0.0, 0.5],    # [1/2, 0, 1/2]
        "R": [0.0, 0.0, 0.5],    # [0, 0, 1/2]
    },
    "orthorhombic_ab": {
        "G": [0.0, 0.0, 0.0],    # Gamma
        "S": [0.5, 0.0, 0.0],    # [1/2, 0, 0]
        "T": [0.5, 0.5, 0.0],    # [1/2, 1/2, 0]
        "R": [0.0, 0.5, 0.5],    # [0, 1/2, 1/2]
        "Y": [0.0, 0.5, 0.0],    # [0, 1/2, 0]
        "Z": [0.0, 0.0, 0.5],    # [0, 0, 1/2]
        "X": [0.5, 0.0, 0.5],    # [1/2, 0, 1/2]
        "U": [0.5, 0.5, 0.5],    # [1/2, 1/2, 1/2]
    },
    "orthorhombic_bc": {
        "G": [0.0, 0.0, 0.0],    # Gamma
        "S": [0.0, 0.5, 0.0],    # [0, 1/2, 0]
        "T": [0.5, 0.5, 0.0],    # [1/2, 1/2, 0]
        "R": [0.5, 0.0, 0.5],    # [1/2, 0, 1/2]
        "X": [0.5, 0.0, 0.0],    # [1/2, 0, 0]
        "W": [0.5, 0.5, 0.5],    # [1/2, 1/2, 1/2]
        "Y": [0.0, 0.5, 0.5],    # [0, 1/2, 1/2]
        "Z": [0.0, 0.0, 0.5],    # [0, 0, 1/2]
    },
    
    # Monoclinic systems (from Table 14.2)
    "monoclinic_simple": {
        "G": [0.0, 0.0, 0.0],    # Gamma
        "A": [0.5, 0.5, 0.0],    # [1/2, 1/2, 0]
        "B": [0.0, 0.5, 0.0],    # [0, 1/2, 0]
        "C": [0.0, 0.5, 0.5],    # [0, 1/2, 1/2]
        "D": [0.5, 0.0, 0.5],    # [1/2, 0, 1/2]
        "E": [0.5, 0.5, 0.5],    # [1/2, 1/2, 1/2]
        "Y": [0.5, 0.0, 0.0],    # [1/2, 0, 0]
        "Z": [0.0, 0.0, 0.5],    # [0, 0, 1/2]
    },
    "monoclinic_ac": {
        "G": [0.0, 0.0, 0.0],    # Gamma
        "A": [0.5, 0.0, 0.0],    # [1/2, 0, 0]
        "Y": [0.0, 0.5, 0.0],    # [0, 1/2, 0]
        "M": [0.5, 0.5, 0.0],    # [1/2, 1/2, 0]
        "C": [0.5, 0.5, 0.5],    # [1/2, 1/2, 1/2]
        "D": [0.5, 0.0, 0.5],    # [1/2, 0, 1/2]
        "E": [0.0, 0.5, 0.5],    # [0, 1/2, 1/2]
        "Z": [0.0, 0.0, 0.5],    # [0, 0, 1/2]
    },
    
    # Triclinic system (from Table 14.2)
    "triclinic": {
        "G": [0.0, 0.0, 0.0],    # Gamma
        "V": [0.5, 0.5, 0.0],    # [1/2, 1/2, 0]
        "Y": [0.0, 0.5, 0.0],    # [0, 1/2, 0]
        "Z": [0.0, 0.0, 0.5],    # [0, 0, 1/2]
        "T": [0.0, 0.5, 0.5],    # [0, 1/2, 1/2]
        "R": [0.5, 0.5, 0.5],    # [1/2, 1/2, 1/2]
        "X": [0.5, 0.0, 0.0],    # [1/2, 0, 0]
        "U": [0.5, 0.0, 0.5],    # [1/2, 0, 1/2]
    },
    
    # Rhombohedral (R lattice) (from Table 14.2)
    "rhombohedral": {
        "G": [0.0, 0.0, 0.0],    # Gamma
        "T": [0.5, 0.5, 0.5],    # [1/2, 1/2, 1/2]
        "F": [0.5, 0.5, 0.0],    # [1/2, 1/2, 0]
        "L": [0.5, 0.0, 0.0],    # [1/2, 0, 0]
        "Z": [0.5, 0.5, -0.5],   # [1/2, 1/2, -1/2]
    },
}


# CRYSTAL23 valid k-point labels from Tables 14.1 and 14.2
CRYSTAL23_VALID_LABELS = {
    # P Cubic (Table 14.1)
    "cubic_simple": {"M", "R", "X"},
    # FC Cubic (Table 14.1)
    "cubic_fc": {"X", "L", "W"},
    # BC Cubic (Table 14.1)
    "cubic_bc": {"H", "P", "N"},
    # Hexagonal or P Trigonal (Table 14.1)
    "hexagonal": {"M", "K", "A", "L", "H"},
    # Rhombohedral (R Trigonal) (Table 14.1)
    "rhombohedral": {"T", "F", "L"},
    # P Monoclinic (Table 14.1)
    "monoclinic_simple": {"A", "B", "C", "D", "E", "Y", "Z"},
    # AC Monoclinic (Table 14.1)
    "monoclinic_ac": {"A", "Y", "M"},
    # P Orthorhombic (Table 14.2)
    "orthorhombic_simple": {"S", "T", "U", "R", "X", "Y", "Z"},
    # FC Orthorhombic (Table 14.2)
    "orthorhombic_fc": {"Z", "Y", "T"},
    # AC Orthorhombic (Table 14.2)
    "orthorhombic_ab": {"S", "T", "R", "Y", "Z"},
    # BC Orthorhombic (Table 14.2)
    "orthorhombic_bc": {"S", "T", "R", "X", "W"},
    # P Tetragonal (Table 14.2)
    "tetragonal_simple": {"M", "R", "A", "X", "Z"},
    # BC Tetragonal (Table 14.2)
    "tetragonal_bc": {"M", "P", "X"},
    # Triclinic (derived from coordinates table)
    "triclinic": {"V", "Y", "Z", "T", "R", "X", "U"},
}


def validate_kpoint_labels_for_crystal23(labels: List[str], space_group: int, lattice_type: str) -> Tuple[bool, List[str]]:
    """
    Validate if all k-point labels are supported by CRYSTAL23. If any are invalid,
    convert the entire path to coordinates (CRYSTAL doesn't allow mixing).

    Args:
        labels: List of k-point labels
        space_group: Space group number
        lattice_type: Lattice centering type

    Returns:
        Tuple of (all_valid, validated_path) where:
        - all_valid: True if all labels are valid, False if coordinates were used
        - validated_path: Either original labels or all coordinates
    """
    crystal_system = get_crystal_system_from_space_group(space_group, lattice_type)

    if crystal_system not in CRYSTAL23_VALID_LABELS:
        print(f"Warning: Crystal system '{crystal_system}' not in CRYSTAL23 validation table")
        return True, labels  # Fall back to original behavior

    if crystal_system not in KPOINT_COORDINATES:
        print(f"Warning: Crystal system '{crystal_system}' coordinates not available")
        return True, labels

    # Check if all labels (except | and G) are valid
    all_valid = True
    invalid_labels = []

    for label in labels:
        if label in ["|", "G", "GAMMA"]:
            continue  # Skip discontinuity markers and Gamma

        if label not in CRYSTAL23_VALID_LABELS[crystal_system]:
            all_valid = False
            invalid_labels.append(label)

    if all_valid:
        return True, labels

    # If any label is invalid, convert ALL to coordinates
    print(f"Warning: Invalid CRYSTAL23 k-point labels found: {invalid_labels}")
    print(f"Converting entire path to coordinates for {crystal_system}")

    kpoint_dict = KPOINT_COORDINATES[crystal_system]
    coordinate_path = []

    for label in labels:
        if label == "|":
            coordinate_path.append(label)
        elif label in kpoint_dict:
            coords = kpoint_dict[label]
            coord_str = f"{coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f}"
            coordinate_path.append(coord_str)
        else:
            print(f"Error: K-point label '{label}' not found in coordinate dictionary")
            # Use origin as fallback
            coordinate_path.append("0.000000 0.000000 0.000000")

    return False, coordinate_path


def get_crystal_system_from_space_group(space_group: int, lattice_type: str = "P") -> str:
    """Determine crystal system and lattice centering from space group number."""
    
    if 1 <= space_group <= 2:
        return "triclinic"
    elif 3 <= space_group <= 15:
        # Monoclinic - check for special centering
        if lattice_type in ["A", "C"]:
            return "monoclinic_ac"
        else:
            return "monoclinic_simple"
    elif 16 <= space_group <= 74:
        # Orthorhombic - check centering
        if lattice_type == "F":
            return "orthorhombic_fc"
        elif lattice_type in ["A", "B"]:
            return "orthorhombic_ab"
        elif lattice_type == "C":
            return "orthorhombic_bc"
        else:
            return "orthorhombic_simple"
    elif 75 <= space_group <= 142:
        # Tetragonal
        if lattice_type in ["I", "B"]:
            return "tetragonal_bc"
        else:
            return "tetragonal_simple"
    elif 143 <= space_group <= 194:
        # Hexagonal/Trigonal
        if lattice_type == "R":
            return "rhombohedral"
        else:
            return "hexagonal"
    elif 195 <= space_group <= 230:
        # Cubic
        if lattice_type == "F":
            return "cubic_fc"
        elif lattice_type in ["I", "B"]:
            return "cubic_bc"
        else:
            return "cubic_simple"
    else:
        # Default to triclinic
        return "triclinic"


def get_band_path_from_symmetry(space_group: int, lattice_type: str = "P") -> List[str]:
    """Get appropriate band path based on space group and lattice type."""
    
    crystal_system = get_crystal_system_from_space_group(space_group, lattice_type)
    
    # Return the appropriate path, defaulting to cubic simple if not found
    return BAND_PATHS.get(crystal_system, ["G", "X", "M", "G", "R"])


def get_kpoint_coordinates_from_labels(labels: List[str], space_group: int, lattice_type: str) -> List[List[float]]:
    """Convert high-symmetry labels to k-point coordinates based on crystal system."""
    
    # Determine crystal system from space group
    crystal_system = get_crystal_system_from_space_group(space_group, lattice_type)
    
    # Get appropriate k-point dictionary
    if crystal_system in KPOINT_COORDINATES:
        kpoint_dict = KPOINT_COORDINATES[crystal_system]
    else:
        print(f"\nWarning: K-point coordinates not defined for crystal system '{crystal_system}'")
        print("Please use manual k-point entry (option 4) instead.")
        return []
    
    # Convert label pairs to coordinate segments
    segments = []
    for i in range(len(labels) - 1):
        start_label = labels[i]
        end_label = labels[i + 1]
        
        if start_label in kpoint_dict and end_label in kpoint_dict:
            start = kpoint_dict[start_label]
            end = kpoint_dict[end_label]
            segments.append(start + end)
        else:
            # For unknown labels, warn user
            print(f"\nWarning: Unknown k-point label '{start_label}' or '{end_label}' for {crystal_system}")
            print("Please use manual k-point entry (option 4) instead.")
            return []
    
    return segments


def get_literature_path_labels(space_group: int, lattice_type: str) -> List[str]:
    """Get literature k-path labels based on crystal system.
    
    Based on Setyawan & Curtarolo, Computational Materials Science 49, 299 (2010)
    """
    crystal_system = get_crystal_system_from_space_group(space_group, lattice_type)
    
    # Define comprehensive paths from literature
    literature_paths = {
        "cubic_simple": ["G", "X", "M", "G", "R", "X", "M", "R"],
        "cubic_fc": ["G", "X", "W", "K", "G", "L", "U", "W", "L", "K", "U", "X"],
        "cubic_bc": ["G", "H", "N", "G", "P", "H", "P", "N"],
        "hexagonal": ["G", "M", "K", "G", "A", "L", "H", "A", "L", "M", "K", "H"],
        "tetragonal_simple": ["G", "X", "M", "G", "Z", "R", "A", "Z", "X", "R", "M", "A"],
        "tetragonal_bc": ["G", "X", "M", "G", "Z", "P", "N", "Z", "M", "X", "P"],
        "orthorhombic_simple": ["G", "X", "S", "Y", "G", "Z", "U", "R", "T", "Z", "Y", "T", "U", "X", "S", "R"],
        "orthorhombic_fc": ["G", "Y", "T", "Z", "G", "X", "S", "R", "U", "X", "T", "Y", "S", "U", "Z", "R"],
        "orthorhombic_ab": ["G", "X", "S", "R", "G", "T", "Y", "Z", "G"],
        "orthorhombic_bc": ["G", "X", "S", "R", "G", "T", "W", "Z", "G", "Y"],
        "monoclinic_simple": ["G", "Y", "A", "B", "G", "C", "D", "E", "Z"],
        "monoclinic_ac": ["G", "A", "Y", "M", "G", "C", "D", "E", "Z"],
        "triclinic": ["X", "G", "Y", "L", "G", "Z", "N", "G", "M", "R", "G"],
        "rhombohedral": ["G", "L", "B", "B1", "Z", "G", "X", "Q", "F", "P1", "Z", "L", "P"]
    }
    
    # Get path for this system, default to standard path if not found
    return literature_paths.get(crystal_system, get_band_path_from_symmetry(space_group, lattice_type))


def get_literature_kpath_vectors(space_group: int, lattice_type: str) -> List[List[float]]:
    """Get comprehensive literature k-paths using vectors for all points.
    
    Based on Setyawan & Curtarolo, Computational Materials Science 49, 299 (2010)
    These paths include points not in CRYSTAL's label tables.
    """
    
    # Literature k-point coordinates (including non-CRYSTAL points)
    # These are system-specific and will be added based on crystal system
    lit_kpoints = {}
    
    # Get crystal system
    crystal_system = get_crystal_system_from_space_group(space_group, lattice_type)
    
    # Get base k-points
    if crystal_system in KPOINT_COORDINATES:
        all_kpoints = KPOINT_COORDINATES[crystal_system].copy()
    else:
        all_kpoints = {}
    
    # Add system-specific literature k-points
    if crystal_system == "cubic_fc":
        lit_kpoints = {
            "K": [3/8, 3/8, 3/4],
            "U": [5/8, 1/4, 5/8],
        }
    elif crystal_system == "triclinic":
        # For triclinic, add missing points from literature
        lit_kpoints = {
            "L": [0.5, 0.0, 0.0],    # Same as X
            "M": [0.0, 0.5, 0.5],    # Same as T  
            "N": [0.5, 0.5, 0.0],    # Same as V
        }
    elif crystal_system == "rhombohedral":
        lit_kpoints = {
            "L": [0.5, 0.0, 0.0],
            "B": [0.5, 0.5, 0.0],
            "B1": [0.0, 0.5, 0.5],
            "F": [0.5, 0.5, 0.5],
            "P": [0.75, 0.25, 0.75],
            "P1": [0.25, 0.25, 0.25],
            "Q": [0.75, 0.25, 0.0],
        }
    elif crystal_system in ["tetragonal_bc", "tetragonal_simple"]:
        lit_kpoints = {
            "N": [0.0, 0.5, 0.0],
            "S": [0.5, 0.5, 0.0],
            "S0": [0.5 + 0.25, 0.25, 0.0],  # ζ, ζ, 0 point
            "Z": [0.5, 0.5, 0.5],
        }
    
    # Add literature-specific points
    all_kpoints.update(lit_kpoints)
    
    # Define comprehensive paths from literature
    literature_paths = {
        "cubic_simple": ["G", "X", "M", "G", "R", "X", "M", "R"],
        "cubic_fc": ["G", "X", "W", "K", "G", "L", "U", "W", "L", "K", "U", "X"],
        "cubic_bc": ["G", "H", "N", "G", "P", "H", "P", "N"],
        "hexagonal": ["G", "M", "K", "G", "A", "L", "H", "A", "L", "M", "K", "H"],
        "tetragonal_simple": ["G", "X", "M", "G", "Z", "R", "A", "Z", "X", "R", "M", "A"],
        "tetragonal_bc": ["G", "X", "M", "G", "Z", "P", "N", "Z", "M", "X", "P"],
        "orthorhombic_simple": ["G", "X", "S", "Y", "G", "Z", "U", "R", "T", "Z", "Y", "T", "U", "X", "S", "R"],
        "orthorhombic_fc": ["G", "Y", "T", "Z", "G", "X", "S", "R", "U", "X", "T", "Y", "S", "U", "Z", "R"],
        "orthorhombic_ab": ["G", "X", "S", "R", "G", "T", "Y", "Z", "G"],
        "orthorhombic_bc": ["G", "X", "S", "R", "G", "T", "W", "Z", "G", "Y"],
        "monoclinic_simple": ["G", "Y", "A", "B", "G", "C", "D", "E", "Z"],
        "monoclinic_ac": ["G", "A", "Y", "M", "G", "C", "D", "E", "Z"],
        "triclinic": ["X", "G", "Y", "L", "G", "Z", "N", "G", "M", "R", "G"],
        "rhombohedral": ["G", "L", "B", "B1", "Z", "G", "X", "Q", "F", "P1", "Z", "L", "P"]
    }
    
    # Get path for this system
    path_labels = literature_paths.get(crystal_system, ["G", "X", "M", "G"])
    
    # Convert to coordinate segments
    segments = []
    for i in range(len(path_labels) - 1):
        start_label = path_labels[i]
        end_label = path_labels[i + 1]
        
        if start_label in all_kpoints and end_label in all_kpoints:
            start = all_kpoints[start_label]
            end = all_kpoints[end_label]
            segments.append(start + end)
        else:
            # Skip segments with undefined points
            print(f"Warning: Skipping segment {start_label}-{end_label} (undefined points)")
    
    return segments


def get_extended_bravais(sg: int, lat: str, 
                        a: float = None, b: float = None, c: float = None,
                        alpha: float = None, beta: float = None, gamma: float = None) -> str:
    """Determine extended Bravais lattice symbol from space group and lattice type.
    
    Enhanced version with cell parameter analysis when available.
    Based on SeeK-path methodology for comprehensive k-path determination.
    Returns symbols like aP1, aP2, cF1, etc.
    
    Args:
        sg: Space group number
        lat: Lattice type (P, C, F, I, R, etc.)
        a, b, c: Lattice parameters in Angstroms (optional)
        alpha, beta, gamma: Lattice angles in degrees (optional)
        
    Returns:
        Extended Bravais lattice symbol
    """
    
    # Triclinic
    if sg == 1:
        return "aP2"  # Without inversion
    elif sg == 2:
        # Distinguish aP2 vs aP3 based on cell parameters if available
        if all(p is not None for p in [a, b, c, alpha, beta, gamma]):
            return determine_triclinic_variant(a, b, c, alpha, beta, gamma)
        else:
            return "aP2"  # Default
        
    # Monoclinic
    elif 3 <= sg <= 15:
        if lat == "P":
            # Could use cell parameters to refine
            if all(p is not None for p in [a, b, c, beta]):
                return determine_monoclinic_variant(sg, a, b, c, beta)
            else:
                return "mP1"
        elif lat in ["A", "C"]:
            return "mS1"
        else:
            return "mP1"
            
    # Orthorhombic
    elif 16 <= sg <= 74:
        if lat == "P":
            return "oP1"
        elif lat == "C":
            # Distinguish oS1 vs oS2 based on cell parameters
            if all(p is not None for p in [a, b, c]):
                return determine_orthorhombic_s_variant(a, b, c)
            else:
                return "oS1"
        elif lat == "F":
            # Distinguish oF1 vs oF2 vs oF3 based on axis lengths
            if all(p is not None for p in [a, b, c]):
                return determine_orthorhombic_f_variant(a, b, c)
            else:
                return "oF1"
        elif lat == "I":
            # Distinguish oI1 vs oI2 vs oI3
            if all(p is not None for p in [a, b, c]):
                return determine_orthorhombic_i_variant(a, b, c)
            else:
                return "oI1"
        else:
            return "oP1"
            
    # Tetragonal
    elif 75 <= sg <= 142:
        if lat == "P":
            return "tP1"
        elif lat == "I":
            # Distinguish tI1 vs tI2 based on c/a ratio
            if a is not None and c is not None:
                return determine_tetragonal_i_variant(a, c)
            else:
                return "tI1"
        else:
            return "tP1"
            
    # Trigonal/Rhombohedral
    elif 143 <= sg <= 167:
        if lat == "P":
            return "hP1"
        elif lat == "R":
            # Distinguish hR1 vs hR2 based on c/a ratio
            if a is not None and c is not None:
                return determine_hexagonal_r_variant(a, c)
            else:
                return "hR1"
        else:
            return "hP1"
            
    # Hexagonal
    elif 168 <= sg <= 194:
        return "hP1"
        
    # Cubic
    elif 195 <= sg <= 230:
        if lat == "P":
            return "cP1"
        elif lat == "F":
            # Distinguish cF1 vs cF2 based on space group
            return determine_cubic_f_variant(sg)
        elif lat == "I":
            # Distinguish cI1 vs cI2 based on space group
            return determine_cubic_i_variant(sg)
        else:
            return "cP1"
            
    # Default
    return "aP1"


# SeeK-path data for extended Bravais lattices
# Stored as fractional coordinates that will be scaled by shrink factor
seekpath_data = {
    "aP2": {
        "segments": [
            [0.0, 0.0, 0.0, 0.5, 0.0, 0.0],    # Γ → X
            [0.0, 0.5, 0.0, 0.0, 0.0, 0.0],    # Y → Γ
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.5],    # Γ → Z
            [0.5, 0.5, 0.0, 0.0, 0.5, 0.0],    # N → Y
            [0.5, 0.0, 0.5, 0.0, 0.0, 0.5],    # M → Z
            [0.5, 0.0, 0.0, 0.5, 0.0, 0.5],    # X → M
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.0]     # R → N
        ],
        "labels": ["G", "X", "|", "Y", "G", "Z", "|", "N", "Y", "|", "M", "Z", "|", "X", "M", "|", "R", "N"]
    },
    "cF1": {
        "segments": [
            [0.0, 0.0, 0.0, 0.5, 0.0, 0.5],    # Γ → X
            [0.5, 0.0, 0.5, 0.5, 0.25, 0.75],  # X → W
            [0.5, 0.25, 0.75, 0.375, 0.375, 0.75], # W → K
            [0.375, 0.375, 0.75, 0.0, 0.0, 0.0],    # K → Γ
            [0.0, 0.0, 0.0, 0.5, 0.5, 0.5],    # Γ → L
            [0.5, 0.5, 0.5, 0.625, 0.25, 0.625], # L → U
            [0.625, 0.25, 0.625, 0.5, 0.25, 0.75], # U → W
            [0.5, 0.5, 0.5, 0.375, 0.375, 0.75], # L → K
            [0.625, 0.25, 0.625, 0.5, 0.0, 0.5]  # U → X
        ],
        "labels": ["G", "X", "W", "K", "G", "L", "U", "W", "|", "L", "K", "|", "U", "X"]
    },
    "cF2": {
        "segments": [
            [0.0, 0.0, 0.0, 0.5, 0.0, 0.5],    # Γ → X
            [0.5, 0.0, 0.5, 0.625, 0.25, 0.625], # X → U
            [0.625, 0.25, 0.625, 0.375, 0.375, 0.75], # U → K'
            [0.375, 0.375, 0.75, 0.0, 0.0, 0.0],    # K' → Γ
            [0.0, 0.0, 0.0, 0.5, 0.5, 0.5],    # Γ → L
            [0.5, 0.5, 0.5, 0.5, 0.25, 0.75],  # L → W
            [0.5, 0.25, 0.75, 0.5, 0.0, 0.5],  # W → X
            [0.375, 0.375, 0.75, 0.5, 0.25, 0.75], # K' → W
            [0.625, 0.25, 0.625, 0.5, 0.5, 0.5]  # U → L
        ],
        "labels": ["G", "X", "U", "K'", "G", "L", "W", "X", "|", "K'", "W", "|", "U", "L"]
    },
    "hR1": {
        "segments": [
            [0.0, 0.0, 0.0, 0.5, 0.0, -0.5],   # Γ → L
            [0.5, 0.0, -0.5, 0.5, 0.5, 0.0],   # L → B1
            [0.5, 0.0, -0.5, 0.5, 0.5, -0.5],  # L → B
            [0.5, 0.5, -0.5, 0.0, 0.0, -0.5],  # B → Z
            [0.0, 0.0, -0.5, 0.0, 0.0, 0.0],   # Z → Γ
            [0.0, 0.0, 0.0, 0.5, 0.5, 0.5],    # Γ → X
            [0.5, 0.5, 0.5, 0.0, 0.5, 0.0],    # X → Q
            [0.0, 0.5, 0.0, 0.5, 0.5, 0.0],    # Q → F
            [0.5, 0.5, 0.0, 0.25, 0.25, -0.5], # F → P1
            [0.25, 0.25, -0.5, 0.0, 0.0, -0.5], # P1 → Z
            [0.5, 0.0, -0.5, 0.0, 0.0, -0.5]   # L → Z
        ],
        "labels": ["G", "L", "B1", "|", "L", "B", "Z", "G", "X", "Q", "F", "P1", "Z", "|", "L", "Z"]
    },
    "hR2": {
        "segments": [
            [0.0, 0.0, 0.0, 0.25, 0.25, 0.25], # Γ → P
            [0.25, 0.25, 0.25, 0.0, 0.0, -0.5], # P → Z
            [0.0, 0.0, -0.5, 0.5, 0.5, 0.0],   # Z → Q
            [0.5, 0.5, 0.0, 0.0, 0.0, 0.0],    # Q → Γ
            [0.0, 0.0, 0.0, 0.5, 0.5, 0.5],    # Γ → F
            [0.5, 0.5, 0.5, 0.25, 0.25, 0.25], # F → P
            [0.25, 0.25, 0.25, 0.0, 0.5, 0.25], # P → Q1
            [0.0, 0.5, 0.25, 0.5, 0.0, -0.5],  # Q1 → L
            [0.5, 0.0, -0.5, 0.0, 0.0, -0.5],  # L → Z
            [0.5, 0.5, 0.5, 0.5, 0.0, -0.5]    # F → L
        ],
        "labels": ["G", "P", "Z", "Q", "G", "F", "P", "Q1", "L", "Z", "|", "F", "L"]
    },
    "aP1": {
        # Default fallback for triclinic without specific variant
        "segments": [
            [0.0, 0.0, 0.0, 0.5, 0.0, 0.0],    # Γ → X
            [0.0, 0.5, 0.0, 0.0, 0.0, 0.0],    # Y → Γ
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.5],    # Γ → Z
            [0.5, 0.5, 0.5, 0.0, 0.0, 0.0],    # R → Γ
            [0.0, 0.0, 0.0, 0.0, 0.5, 0.5],    # Γ → T
            [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],    # U → Γ
            [0.0, 0.0, 0.0, 0.5, 0.5, 0.0]     # Γ → V
        ],
        "labels": ["G", "X", "|", "Y", "G", "|", "Z", "|", "R", "G", "|", "T", "|", "U", "G", "|", "V"]
    },
    "aP3": {
        # Triclinic P-1 with inversion (variant 3)
        "segments": [
            [0.0, 0.0, 0.0, 0.5, 0.0, 0.0],    # Γ → X
            [0.0, 0.5, 0.0, 0.0, 0.0, 0.0],    # Y → Γ
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.5],    # Γ → Z
            [-0.5, -0.5, 0.5, 0.0, 0.0, 0.0],  # R₂ → Γ
            [0.0, 0.0, 0.0, 0.0, -0.5, 0.5],   # Γ → T₂
            [-0.5, 0.0, 0.5, 0.0, 0.0, 0.0],   # U₂ → Γ
            [0.0, 0.0, 0.0, 0.5, -0.5, 0.0]    # Γ → V₂
        ],
        "labels": ["G", "X", "|", "Y", "G", "|", "Z", "|", "R_2", "G", "|", "T_2", "|", "U_2", "G", "|", "V_2"]
    },
    "cP1": {
        # Cubic primitive with inversion
        "segments": [
            [0.0, 0.0, 0.0, 0.0, 0.5, 0.0],    # Γ → X
            [0.0, 0.5, 0.0, 0.5, 0.5, 0.0],    # X → M
            [0.5, 0.5, 0.0, 0.0, 0.0, 0.0],    # M → Γ
            [0.0, 0.0, 0.0, 0.5, 0.5, 0.5],    # Γ → R
            [0.5, 0.5, 0.5, 0.0, 0.5, 0.0],    # R → X
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.0],    # R → M
            [0.5, 0.5, 0.0, 0.5, 0.0, 0.0]     # M → X₁
        ],
        "labels": ["G", "X", "M", "G", "R", "X", "|", "R", "M", "|", "M", "X_1"]
    },
    "cP2": {
        # Cubic primitive with inversion (variant 2)
        "segments": [
            [0.0, 0.0, 0.0, 0.0, 0.5, 0.0],    # Γ → X
            [0.0, 0.5, 0.0, 0.5, 0.5, 0.0],    # X → M
            [0.5, 0.5, 0.0, 0.0, 0.0, 0.0],    # M → Γ
            [0.0, 0.0, 0.0, 0.5, 0.5, 0.5],    # Γ → R
            [0.5, 0.5, 0.5, 0.0, 0.5, 0.0],    # R → X
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.0]     # R → M
        ],
        "labels": ["G", "X", "M", "G", "R", "X", "|", "M", "R"]
    },
    "cI1": {
        # Cubic body-centered with inversion
        "segments": [
            [0.0, 0.0, 0.0, 0.5, -0.5, 0.5],   # Γ → H
            [0.5, -0.5, 0.5, 0.0, 0.0, 0.5],   # H → N
            [0.0, 0.0, 0.5, 0.0, 0.0, 0.0],    # N → Γ
            [0.0, 0.0, 0.0, 0.25, 0.25, 0.25], # Γ → P
            [0.25, 0.25, 0.25, 0.5, -0.5, 0.5], # P → H
            [0.25, 0.25, 0.25, 0.0, 0.0, 0.5]  # P → N
        ],
        "labels": ["G", "H", "N", "G", "P", "H", "|", "P", "N"]
    },
    "hP1": {
        # Hexagonal primitive with inversion
        "segments": [
            [0.0, 0.0, 0.0, 0.5, 0.0, 0.0],    # Γ → M
            [0.5, 0.0, 0.0, 1/3, 1/3, 0.0],    # M → K
            [1/3, 1/3, 0.0, 0.0, 0.0, 0.0],    # K → Γ
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.5],    # Γ → A
            [0.0, 0.0, 0.5, 0.5, 0.0, 0.5],    # A → L
            [0.5, 0.0, 0.5, 1/3, 1/3, 0.5],    # L → H
            [1/3, 1/3, 0.5, 0.0, 0.0, 0.5],    # H → A
            [0.5, 0.0, 0.5, 0.5, 0.0, 0.0],    # L → M
            [1/3, 1/3, 0.5, 1/3, 1/3, 0.0],    # H → K
            [1/3, 1/3, 0.0, 1/3, 1/3, -0.5]    # K → H₂
        ],
        "labels": ["G", "M", "K", "G", "A", "L", "H", "A", "|", "L", "M", "|", "H", "K", "|", "K", "H_2"]
    },
    "hP2": {
        # Hexagonal primitive with inversion (variant 2)
        "segments": [
            [0.0, 0.0, 0.0, 0.5, 0.0, 0.0],    # Γ → M
            [0.5, 0.0, 0.0, 1/3, 1/3, 0.0],    # M → K
            [1/3, 1/3, 0.0, 0.0, 0.0, 0.0],    # K → Γ
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.5],    # Γ → A
            [0.0, 0.0, 0.5, 0.5, 0.0, 0.5],    # A → L
            [0.5, 0.0, 0.5, 1/3, 1/3, 0.5],    # L → H
            [1/3, 1/3, 0.5, 0.0, 0.0, 0.5],    # H → A
            [0.5, 0.0, 0.5, 0.5, 0.0, 0.0],    # L → M
            [1/3, 1/3, 0.5, 1/3, 1/3, 0.0]     # H → K
        ],
        "labels": ["G", "M", "K", "G", "A", "L", "H", "A", "|", "L", "M", "|", "H", "K"]
    },
    "mP1": {
        # Monoclinic primitive with inversion
        "segments": [
            [0.0, 0.0, 0.0, 0.0, 0.5, 0.0],    # Γ → Z
            [0.0, 0.5, 0.0, 0.0, 0.5, 0.5],    # Z → D
            [0.0, 0.5, 0.5, 0.0, 0.0, 0.5],    # D → B
            [0.0, 0.0, 0.5, 0.0, 0.0, 0.0],    # B → Γ
            [0.0, 0.0, 0.0, -0.5, 0.0, 0.5],   # Γ → A
            [-0.5, 0.0, 0.5, -0.5, 0.5, 0.5],  # A → E
            [-0.5, 0.5, 0.5, 0.0, 0.5, 0.0],   # E → Z
            [0.0, 0.5, 0.0, -0.5, 0.5, 0.0],   # Z → C₂
            [-0.5, 0.5, 0.0, -0.5, 0.0, 0.0],  # C₂ → Y₂
            [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0]    # Y₂ → Γ
        ],
        "labels": ["G", "Z", "D", "B", "G", "A", "E", "Z", "C_2", "Y_2", "G"]
    },
    "oP1": {
        # Orthorhombic primitive with inversion
        "segments": [
            [0.0, 0.0, 0.0, 0.5, 0.0, 0.0],    # Γ → X
            [0.5, 0.0, 0.0, 0.5, 0.5, 0.0],    # X → S
            [0.5, 0.5, 0.0, 0.0, 0.5, 0.0],    # S → Y
            [0.0, 0.5, 0.0, 0.0, 0.0, 0.0],    # Y → Γ
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.5],    # Γ → Z
            [0.0, 0.0, 0.5, 0.5, 0.0, 0.5],    # Z → U
            [0.5, 0.0, 0.5, 0.5, 0.5, 0.5],    # U → R
            [0.5, 0.5, 0.5, 0.0, 0.5, 0.5],    # R → T
            [0.0, 0.5, 0.5, 0.0, 0.0, 0.5],    # T → Z
            [0.5, 0.0, 0.0, 0.5, 0.0, 0.5],    # X → U
            [0.0, 0.5, 0.0, 0.0, 0.5, 0.5],    # Y → T
            [0.5, 0.5, 0.0, 0.5, 0.5, 0.5]     # S → R
        ],
        "labels": ["G", "X", "S", "Y", "G", "Z", "U", "R", "T", "Z", "|", "Y", "T", "|", "X", "U", "|", "S", "R"]
    },
    "tP1": {
        # Tetragonal primitive with inversion
        "segments": [
            [0.0, 0.0, 0.0, 0.0, 0.5, 0.0],    # Γ → X
            [0.0, 0.5, 0.0, 0.5, 0.5, 0.0],    # X → M
            [0.5, 0.5, 0.0, 0.0, 0.0, 0.0],    # M → Γ
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.5],    # Γ → Z
            [0.0, 0.0, 0.5, 0.0, 0.5, 0.5],    # Z → R
            [0.0, 0.5, 0.5, 0.5, 0.5, 0.5],    # R → A
            [0.5, 0.5, 0.5, 0.0, 0.0, 0.5],    # A → Z
            [0.0, 0.5, 0.0, 0.0, 0.5, 0.5],    # X → R
            [0.5, 0.5, 0.0, 0.5, 0.5, 0.5]     # M → A
        ],
        "labels": ["G", "X", "M", "G", "Z", "R", "A", "Z", "|", "X", "R", "|", "M", "A"]
    },
    "tI1": {
        # Tetragonal body-centered with inversion
        "segments": [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.5],    # Γ → X
            [0.0, 0.0, 0.5, -0.5, 0.5, 0.5],   # X → M
            [-0.5, 0.5, 0.5, 0.0, 0.0, 0.0],   # M → Γ
            [0.0, 0.0, 0.0, 0.342, 0.342, -0.342], # Γ → Z (approximate for general IS)
            [-0.342, 0.658, 0.342, -0.5, 0.5, 0.5], # Z₀ → M
            [0.0, 0.0, 0.5, 0.25, 0.25, 0.25], # X → P
            [0.25, 0.25, 0.25, 0.0, 0.5, 0.0], # P → N
            [0.0, 0.5, 0.0, 0.0, 0.0, 0.0]     # N → Γ
        ],
        "labels": ["G", "X", "M", "G", "Z", "Z_0", "M", "|", "X", "P", "N", "G"]
    },
    "tI2": {
        # Tetragonal body-centered with inversion (variant 2)
        "segments": [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.5],    # Γ → X
            [0.0, 0.0, 0.5, 0.25, 0.25, 0.25], # X → P
            [0.25, 0.25, 0.25, 0.0, 0.5, 0.0], # P → N
            [0.0, 0.5, 0.0, 0.0, 0.0, 0.0],    # N → Γ
            [0.0, 0.0, 0.0, 0.5, 0.5, -0.5],   # Γ → M
            [0.5, 0.5, -0.5, 0.392, 0.608, -0.392], # M → S (approximate)
            [-0.392, 0.392, 0.392, 0.0, 0.0, 0.0], # S₀ → Γ
            [0.0, 0.0, 0.5, -0.285, 0.285, 0.5], # X → R (approximate)
            [0.5, 0.5, -0.285, 0.5, 0.5, -0.5] # G → M
        ],
        "labels": ["G", "X", "P", "N", "G", "M", "S", "S_0", "G", "|", "X", "R", "|", "G", "M"]
    },
    "mS1": {
        # Monoclinic C-centered with inversion (mC1 in SeeK-path)
        # Note: fractional coordinates here, will be scaled by shrink
        "segments": [
            [0.0, 0.0, 0.0, 0.388, 0.388, 0.0],    # Γ → C (approx)
            [-0.388, 0.612, 0.0, -0.5, 0.5, 0.0],  # C₂ → Y₂
            [-0.5, 0.5, 0.0, 0.0, 0.0, 0.0],       # Y₂ → Γ
            [0.0, 0.0, 0.0, -0.5, 0.5, 0.5],       # Γ → M₂
            [-0.5, 0.5, 0.5, -0.368, 0.632, 0.5],  # M₂ → D (approx)
            [0.368, 0.368, 0.5, 0.0, 0.0, 0.5],    # D₂ → A
            [0.0, 0.0, 0.5, 0.0, 0.0, 0.0],        # A → Γ
            [0.0, 0.5, 0.5, 0.0, 0.0, 0.0],        # L₂ → Γ
            [0.0, 0.0, 0.0, 0.0, 0.5, 0.0]         # Γ → V₂
        ],
        "labels": ["G", "C", "C_2", "Y_2", "G", "M_2", "D", "D_2", "A", "G", "|", "L_2", "G", "|", "V_2"]
    },
    "oS1": {
        # Orthorhombic C-centered with inversion (oC1 in SeeK-path)
        # Note: fractional coordinates here
        "segments": [
            [0.0, 0.0, 0.0, -0.5, 0.5, 0.0],       # Γ → Y
            [-0.5, 0.5, 0.0, -0.446, 0.554, 0.0],  # Y → C₀ (approx)
            [0.446, 0.446, 0.0, 0.0, 0.0, 0.0],    # Σ₀ → Γ
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.5],        # Γ → Z
            [0.0, 0.0, 0.5, 0.446, 0.446, 0.5],    # Z → A₀ (approx)
            [-0.446, 0.554, 0.5, -0.5, 0.5, 0.5],  # E₀ → T
            [-0.5, 0.5, 0.5, -0.5, 0.5, 0.0],      # T → Y
            [0.0, 0.0, 0.0, 0.0, 0.5, 0.0],        # Γ → S
            [0.0, 0.5, 0.0, 0.0, 0.5, 0.5],        # S → R
            [0.0, 0.5, 0.5, 0.0, 0.0, 0.5],        # R → Z
            [0.0, 0.0, 0.5, -0.5, 0.5, 0.5]        # Z → T
        ],
        "labels": ["G", "Y", "C_0", "SIGMA_0", "G", "Z", "A_0", "E_0", "T", "Y", "|", "G", "S", "R", "Z", "T"]
    },
    "oI1": {
        # Orthorhombic body-centered with inversion
        "segments": [
            [0.0, 0.0, 0.0, 0.5, 0.5, -0.5],       # Γ → X
            [0.5, 0.5, -0.5, 0.304, 0.696, -0.304], # X → F₂ (approx)
            [-0.304, 0.304, 0.304, 0.0, 0.0, 0.0], # Σ₀ → Γ
            [0.0, 0.0, 0.0, 0.314, -0.314, 0.314], # Γ → Y₀ (approx)
            [0.814, 0.314, -0.314, 0.5, 0.5, -0.5], # U₀ → X (approx)
            [0.0, 0.0, 0.0, 0.0, 0.5, 0.0],        # Γ → R
            [0.0, 0.5, 0.0, 0.25, 0.25, 0.25],     # R → W
            [0.25, 0.25, 0.25, 0.5, 0.0, 0.0],     # W → S
            [0.5, 0.0, 0.0, 0.0, 0.0, 0.0],        # S → Γ
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.5],        # Γ → T
            [0.0, 0.0, 0.5, 0.25, 0.25, 0.25]      # T → W
        ],
        "labels": ["G", "X", "F_2", "SIGMA_0", "G", "Y_0", "U_0", "X", "|", "G", "R", "W", "S", "G", "T", "W"]
    },
    "oF1": {
        # Orthorhombic face-centered with inversion
        # Note: using approximate fractional coordinates
        "segments": [
            [0.0, 0.0, 0.0, 0.5, 0.0, 0.5],        # Γ → Y
            [0.5, 0.0, 0.5, 1.0, 0.5, 0.5],        # Y → T
            [1.0, 0.5, 0.5, 0.5, 0.5, 0.0],        # T → Z
            [0.5, 0.5, 0.0, 0.0, 0.0, 0.0],        # Z → Γ
            [0.0, 0.0, 0.0, 0.0, 0.3, 0.3],        # Γ → Σ₀ (approx)
            [1.0, 0.7, 0.7, 1.0, 0.5, 0.5],        # U₀ → T (approx)
            [0.5, 0.0, 0.5, 0.5, 0.227, 0.773],    # Y → C₀ (approx)
            [0.5, 0.773, 0.227, 0.5, 0.5, 0.0],    # A₀ → Z (approx)
            [0.0, 0.0, 0.0, 0.5, 0.5, 0.5]         # Γ → L
        ],
        "labels": ["G", "Y", "T", "Z", "G", "SIGMA_0", "U_0", "T", "|", "Y", "C_0", "A_0", "Z", "|", "G", "L"]
    },
    
    # Triclinic lattices
    "aP1": {
        # Triclinic primitive without inversion (P1)
        "segments": [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],      # Γ → X
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],      # Y → Γ
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],      # Γ → Z
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],      # R → Γ
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],      # Γ → T
            [1.0, 0.0, 1.0, 0.0, 0.0, 0.0],      # U → Γ
            [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],      # Γ → V
            [0.0, 0.0, 0.0, -1.0, 0.0, 0.0],     # Γ → X'
            [0.0, -1.0, 0.0, 0.0, 0.0, 0.0],     # Y' → Γ
            [0.0, 0.0, 0.0, 0.0, 0.0, -1.0],     # Γ → Z'
            [-1.0, -1.0, -1.0, 0.0, 0.0, 0.0],   # R' → Γ
            [0.0, 0.0, 0.0, 0.0, -1.0, -1.0],    # Γ → T'
            [-1.0, 0.0, -1.0, 0.0, 0.0, 0.0],    # U' → Γ
            [0.0, 0.0, 0.0, -1.0, -1.0, 0.0]     # Γ → V'
        ],
        "labels": ["G", "X", "Y", "G", "G", "Z", "R", "G", "G", "T", "U", "G", "G", "V", "G", "X'", "Y'", "G", "G", "Z'", "R'", "G", "G", "T'", "U'", "G", "G", "V'"]
    },
    "aP3": {
        # Triclinic primitive with inversion (P-1)
        "segments": [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],      # Γ → X
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],      # Y → Γ
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],      # Γ → Z
            [-1.0, -1.0, 1.0, 0.0, 0.0, 0.0],    # R₂ → Γ
            [0.0, 0.0, 0.0, 0.0, -1.0, 1.0],     # Γ → T₂
            [-1.0, 0.0, 1.0, 0.0, 0.0, 0.0],     # U₂ → Γ
            [0.0, 0.0, 0.0, 1.0, -1.0, 0.0]      # Γ → V₂
        ],
        "labels": ["G", "X", "Y", "G", "G", "Z", "R_2", "G", "G", "T_2", "U_2", "G", "G", "V_2"]
    },
    
    # Cubic lattices
    "cP1": {
        # Cubic primitive with inversion (Pm-3)
        "segments": [
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],      # Γ → X
            [0.0, 1.0, 0.0, 1.0, 1.0, 0.0],      # X → M
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],      # M → Γ
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],      # Γ → R
            [1.0, 1.0, 1.0, 0.0, 1.0, 0.0],      # R → X
            [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],      # R → M
            [1.0, 1.0, 0.0, 1.0, 0.0, 0.0]       # M → X₁
        ],
        "labels": ["G", "X", "M", "G", "R", "X", "|", "R", "M", "|", "M", "X_1"]
    },
    "cP2": {
        # Cubic primitive with inversion (Pm-3m)
        "segments": [
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],      # Γ → X
            [0.0, 1.0, 0.0, 1.0, 1.0, 0.0],      # X → M
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],      # M → Γ
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],      # Γ → R
            [1.0, 1.0, 1.0, 0.0, 1.0, 0.0],      # R → X
            [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]       # R → M
        ],
        "labels": ["G", "X", "M", "G", "R", "X", "|", "R", "M"]
    },
    "cI1": {
        # Cubic body-centered with inversion (Im-3m)
        "segments": [
            [0.0, 0.0, 0.0, 2.0, -2.0, 2.0],     # Γ → H
            [2.0, -2.0, 2.0, 0.0, 0.0, 2.0],     # H → N
            [0.0, 0.0, 2.0, 0.0, 0.0, 0.0],      # N → Γ
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],      # Γ → P
            [1.0, 1.0, 1.0, 2.0, -2.0, 2.0],     # P → H
            [1.0, 1.0, 1.0, 0.0, 0.0, 2.0]       # P → N
        ],
        "labels": ["G", "H", "N", "G", "P", "H", "|", "P", "N"]
    },
    
    # Hexagonal lattices
    "hP1": {
        # Hexagonal primitive with inversion (P-31m)
        "segments": [
            [0.0, 0.0, 0.0, 3.0, 0.0, 0.0],      # Γ → M
            [3.0, 0.0, 0.0, 2.0, 2.0, 0.0],      # M → K
            [2.0, 2.0, 0.0, 0.0, 0.0, 0.0],      # K → Γ
            [0.0, 0.0, 0.0, 0.0, 0.0, 3.0],      # Γ → A
            [0.0, 0.0, 3.0, 3.0, 0.0, 3.0],      # A → L
            [3.0, 0.0, 3.0, 2.0, 2.0, 3.0],      # L → H
            [2.0, 2.0, 3.0, 0.0, 0.0, 3.0],      # H → A
            [3.0, 0.0, 3.0, 3.0, 0.0, 0.0],      # L → M
            [2.0, 2.0, 3.0, 2.0, 2.0, 0.0],      # H → K
            [2.0, 2.0, 0.0, 2.0, 2.0, -3.0]      # K → H₂
        ],
        "labels": ["G", "M", "K", "G", "A", "L", "H", "A", "|", "L", "M", "|", "H", "K", "|", "K", "H_2"]
    },
    "hP2": {
        # Hexagonal primitive with inversion (P6/mmm)
        "segments": [
            [0.0, 0.0, 0.0, 3.0, 0.0, 0.0],      # Γ → M
            [3.0, 0.0, 0.0, 2.0, 2.0, 0.0],      # M → K
            [2.0, 2.0, 0.0, 0.0, 0.0, 0.0],      # K → Γ
            [0.0, 0.0, 0.0, 0.0, 0.0, 3.0],      # Γ → A
            [0.0, 0.0, 3.0, 3.0, 0.0, 3.0],      # A → L
            [3.0, 0.0, 3.0, 2.0, 2.0, 3.0],      # L → H
            [2.0, 2.0, 3.0, 0.0, 0.0, 3.0],      # H → A
            [3.0, 0.0, 3.0, 3.0, 0.0, 0.0],      # L → M
            [2.0, 2.0, 3.0, 2.0, 2.0, 0.0]       # H → K
        ],
        "labels": ["G", "M", "K", "G", "A", "L", "H", "A", "|", "L", "M", "|", "H", "K"]
    },
    
    # Monoclinic lattices
    "mP1": {
        # Monoclinic primitive with inversion (P2_1/m)
        "segments": [
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],      # Γ → Z
            [0.0, 1.0, 0.0, 0.0, 1.0, 1.0],      # Z → D
            [0.0, 1.0, 1.0, 0.0, 0.0, 1.0],      # D → B
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],      # B → Γ
            [0.0, 0.0, 0.0, -1.0, 0.0, 1.0],     # Γ → A
            [-1.0, 0.0, 1.0, -1.0, 1.0, 1.0],    # A → E
            [-1.0, 1.0, 1.0, 0.0, 1.0, 0.0],     # E → Z
            [0.0, 1.0, 0.0, -1.0, 1.0, 0.0],     # Z → C₂
            [-1.0, 1.0, 0.0, -1.0, 0.0, 0.0],    # C₂ → Y₂
            [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0]      # Y₂ → Γ
        ],
        "labels": ["G", "Z", "D", "B", "G", "A", "E", "Z", "C_2", "Y_2", "G"]
    },
    
    # Orthorhombic lattices
    "oP1": {
        # Orthorhombic primitive with inversion (Pmmm)
        "segments": [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],      # Γ → X
            [1.0, 0.0, 0.0, 1.0, 1.0, 0.0],      # X → S
            [1.0, 1.0, 0.0, 0.0, 1.0, 0.0],      # S → Y
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],      # Y → Γ
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],      # Γ → Z
            [0.0, 0.0, 1.0, 1.0, 0.0, 1.0],      # Z → U
            [1.0, 0.0, 1.0, 1.0, 1.0, 1.0],      # U → R
            [1.0, 1.0, 1.0, 0.0, 1.0, 1.0],      # R → T
            [0.0, 1.0, 1.0, 0.0, 0.0, 1.0],      # T → Z
            [1.0, 0.0, 0.0, 1.0, 0.0, 1.0],      # X → U
            [0.0, 1.0, 0.0, 0.0, 1.0, 1.0],      # Y → T
            [1.0, 1.0, 0.0, 1.0, 1.0, 1.0]       # S → R
        ],
        "labels": ["G", "X", "S", "Y", "G", "Z", "U", "R", "T", "Z", "|", "X", "U", "|", "Y", "T", "|", "S", "R"]
    },
    
    # Tetragonal lattices (already present but adding tP1)
    "tP1": {
        # Tetragonal primitive with inversion (P4/mmm)
        "segments": [
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],      # Γ → X
            [0.0, 1.0, 0.0, 1.0, 1.0, 0.0],      # X → M
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],      # M → Γ
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],      # Γ → Z
            [0.0, 0.0, 1.0, 0.0, 1.0, 1.0],      # Z → R
            [0.0, 1.0, 1.0, 1.0, 1.0, 1.0],      # R → A
            [1.0, 1.0, 1.0, 0.0, 0.0, 1.0],      # A → Z
            [0.0, 1.0, 0.0, 0.0, 1.0, 1.0],      # X → R
            [1.0, 1.0, 0.0, 1.0, 1.0, 1.0]       # M → A
        ],
        "labels": ["G", "X", "M", "G", "Z", "R", "A", "Z", "|", "X", "R", "|", "M", "A"]
    },
    "tI1": {
        # Tetragonal body-centered with inversion (I4/m)
        "segments": [
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],           # Γ → X
            [0.0, 0.0, 1.0, -1.0, 1.0, 1.0],          # X → M
            [-1.0, 1.0, 1.0, 0.0, 0.0, 0.0],         # M → Γ
            [0.0, 0.0, 0.0, 0.683436, 0.683436, -0.683436],  # Γ → Z (826/1206)
            [-0.683436, 1.313030, 0.683436, -1.0, 1.0, 1.0], # Z₀ → M (scaled)
            [0.0, 0.0, 1.0, 0.5, 0.5, 0.5],          # X → P
            [0.5, 0.5, 0.5, 0.0, 1.0, 0.0],          # P → N
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]           # N → Γ
        ],
        "labels": ["G", "X", "M", "G", "Z", "Z_0", "M", "|", "X", "P", "N", "G"]
    },
    "tI2": {
        # Tetragonal body-centered with inversion (I4/mmm)
        "segments": [
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],          # Γ → X
            [0.0, 0.0, 1.0, 0.5, 0.5, 0.5],          # X → P
            [0.5, 0.5, 0.5, 0.0, 1.0, 0.0],          # P → N
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],          # N → Γ
            [0.0, 0.0, 0.0, 1.0, 1.0, -1.0],         # Γ → M
            [1.0, 1.0, -1.0, 0.784078, 1.212885, -0.784078], # M → S (scaled)
            [-0.784078, 0.784078, 0.784078, 0.0, 0.0, 0.0], # S₀ → Γ
            [0.0, 0.0, 1.0, -0.569274, 0.569274, 1.0], # X → R (scaled)
            [1.0, 1.0, -0.569274, 1.0, 1.0, -1.0]     # G → M
        ],
        "labels": ["G", "X", "P", "N", "G", "M", "S", "S_0", "G", "|", "X", "R", "|", "G", "M"]
    },
    
    # Monoclinic C-centered lattices
    "mC1": {
        # Base-centered monoclinic with inversion (C2/c)
        "segments": [
            [0.0, 0.0, 0.0, 0.388212, 0.388212, 0.0],     # Γ → C (scaled 102700/264680)
            [-0.388212, 0.612109, 0.0, -0.5, 0.5, 0.0],   # C₂ → Y₂
            [-0.5, 0.5, 0.0, 0.0, 0.0, 0.0],              # Y₂ → Γ
            [0.0, 0.0, 0.0, -0.5, 0.5, 0.5],              # Γ → M₂
            [-0.5, 0.5, 0.5, -0.367305, 0.632695, 0.5],   # M₂ → D (scaled)
            [0.367305, 0.367305, 0.5, 0.0, 0.0, 0.5],     # D₂ → A
            [0.0, 0.0, 0.5, 0.0, 0.0, 0.0],               # A → Γ
            [0.0, 0.5, 0.5, 0.0, 0.0, 0.0],               # L₂ → Γ
            [0.0, 0.0, 0.0, 0.0, 0.5, 0.0]                # Γ → V₂
        ],
        "labels": ["G", "C", "C_2", "Y_2", "G", "M_2", "D", "D_2", "A", "G", "|", "L_2", "G", "|", "G", "V_2"]
    },
    "mC2": {
        # Base-centered monoclinic with inversion (C2/c)
        "segments": [
            [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],      # Γ → Y
            [1.0, 1.0, 0.0, 1.0, 1.0, 1.0],      # Y → M
            [1.0, 1.0, 1.0, 0.0, 0.0, 1.0],      # M → A
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],      # A → Γ
            [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],      # L₂ → Γ
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]       # Γ → V₂
        ],
        "labels": ["G", "Y", "M", "A", "G", "|", "L_2", "G", "|", "G", "V_2"]
    },
    "mC3": {
        # Base-centered monoclinic with inversion (C2/m)
        "segments": [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.5],              # Γ → A
            [0.0, 0.0, 0.5, 0.410211, 0.410211, 0.5],    # A → I₂ (scaled 233/568)
            [-0.410211, 0.589789, 0.5, -0.5, 0.5, 0.5],  # I → M₂
            [-0.5, 0.5, 0.5, 0.0, 0.0, 0.0],             # M₂ → Γ
            [0.0, 0.0, 0.0, 0.5, 0.5, 0.0],              # Γ → Y
            [0.0, 0.5, 0.5, 0.0, 0.0, 0.0],              # L₂ → Γ
            [0.0, 0.0, 0.0, 0.0, 0.5, 0.0]               # Γ → V₂
        ],
        "labels": ["G", "A", "I_2", "I", "M_2", "G", "Y", "|", "L_2", "G", "|", "G", "V_2"]
    },
    
    # Orthorhombic C-centered lattices
    "oC1": {
        # Base-centered orthorhombic with inversion (Cmmm)
        "segments": [
            [0.0, 0.0, 0.0, -0.5, 0.5, 0.0],             # Γ → Y
            [-0.5, 0.5, 0.0, -0.446177, 0.553823, 0.0],  # Y → C₀ (scaled 572/1282)
            [0.446177, 0.446177, 0.0, 0.0, 0.0, 0.0],    # Σ₀ → Γ
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.5],              # Γ → Z
            [0.0, 0.0, 0.5, 0.446177, 0.446177, 0.5],    # Z → A₀
            [-0.446177, 0.553823, 0.5, -0.5, 0.5, 0.5],  # E₀ → T
            [-0.5, 0.5, 0.5, -0.5, 0.5, 0.0],            # T → Y
            [0.0, 0.0, 0.0, 0.0, 0.5, 0.0],              # Γ → S
            [0.0, 0.5, 0.0, 0.0, 0.5, 0.5],              # S → R
            [0.0, 0.5, 0.5, 0.0, 0.0, 0.5],              # R → Z
            [0.0, 0.0, 0.5, -0.5, 0.5, 0.5]              # Z → T
        ],
        "labels": ["G", "Y", "C_0", "SIGMA_0", "G", "Z", "A_0", "E_0", "T", "Y", "|", "G", "S", "R", "Z", "T"]
    },
    "oC2": {
        # Base-centered orthorhombic with inversion (Cmcm)
        "segments": [
            [0.0, 0.0, 0.0, 0.5, 0.5, 0.0],              # Γ → Y
            [0.5, 0.5, 0.0, 0.353413, 0.646587, 0.0],    # Y → F₀ (scaled 238/674)
            [-0.353413, 0.353413, 0.0, 0.0, 0.0, 0.0],   # Δ₀ → Γ
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.5],              # Γ → Z
            [0.0, 0.0, 0.5, -0.353413, 0.353413, 0.5],   # Z → B₀
            [0.353413, 0.646587, 0.5, 0.5, 0.5, 0.5],    # G₀ → T
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.0],              # T → Y
            [0.0, 0.0, 0.0, 0.0, 0.5, 0.0],              # Γ → S
            [0.0, 0.5, 0.0, 0.0, 0.5, 0.5],              # S → R
            [0.0, 0.5, 0.5, 0.0, 0.0, 0.5],              # R → Z
            [0.0, 0.0, 0.5, 0.5, 0.5, 0.5]               # Z → T
        ],
        "labels": ["G", "Y", "F_0", "DELTA_0", "G", "Z", "B_0", "G_0", "T", "Y", "|", "G", "S", "R", "Z", "T"]
    },
    
    # Orthorhombic I-centered lattices
    "oI1": {
        # Body-centered orthorhombic with inversion (Immm)
        "segments": [
            [0.0, 0.0, 0.0, 0.5, 0.5, -0.5],             # Γ → X
            [0.5, 0.5, -0.5, 0.322277, 0.677723, -0.322277], # X → F₂ (scaled 58984/183012)
            [-0.322277, 0.322277, 0.322277, 0.0, 0.0, 0.0], # Σ₀ → Γ
            [0.0, 0.0, 0.0, 0.463622, -0.463622, 0.463622], # Γ → Y₀ (scaled 84862/183012)
            [0.536378, 0.463622, -0.463622, 0.5, 0.5, -0.5], # U₀ → X
            [0.0, 0.0, 0.0, 0.0, 0.5, 0.0],              # Γ → R
            [0.0, 0.5, 0.0, 0.25, 0.25, 0.25],           # R → W
            [0.25, 0.25, 0.25, 0.5, 0.0, 0.0],           # W → S
            [0.5, 0.0, 0.0, 0.0, 0.0, 0.0],              # S → Γ
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.5],              # Γ → T
            [0.0, 0.0, 0.5, 0.25, 0.25, 0.25]            # T → W
        ],
        "labels": ["G", "X", "F_2", "SIGMA_0", "G", "Y_0", "U_0", "X", "|", "G", "R", "W", "S", "G", "T", "W"]
    },
    "oI3": {
        # Body-centered orthorhombic with inversion (Imma)
        "segments": [
            [0.0, 0.0, 0.0, 0.5, -0.5, 0.5],             # Γ → X
            [0.5, -0.5, 0.5, 0.345870, -0.345870, 0.654130], # X → F₀ (scaled 939275/2716140)
            [-0.345870, 0.345870, 0.345870, 0.0, 0.0, 0.0], # Σ₀ → Γ
            [0.0, 0.0, 0.0, 0.482915, 0.482915, -0.482915], # Γ → Λ₀ (scaled 1311379/2716140)
            [0.517085, -0.482915, 0.482915, 0.5, -0.5, 0.5], # G₀ → X
            [0.0, 0.0, 0.0, 0.0, 0.5, 0.0],              # Γ → R
            [0.0, 0.5, 0.0, 0.25, 0.25, 0.25],           # R → W
            [0.25, 0.25, 0.25, 0.5, 0.0, 0.0],           # W → S
            [0.5, 0.0, 0.0, 0.0, 0.0, 0.0],              # S → Γ
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.5],              # Γ → T
            [0.0, 0.0, 0.5, 0.25, 0.25, 0.25]            # T → W
        ],
        "labels": ["G", "X", "F_0", "SIGMA_0", "G", "LAMBDA_0", "G_0", "X", "|", "G", "R", "W", "S", "G", "T", "W"]
    },
    
    # Orthorhombic F-centered lattices (simplified with fractional coordinates)
    "oF1": {
        # Face-centered orthorhombic with inversion (Fmmm)
        "segments": [
            [0.0, 0.0, 0.0, 0.5, 0.0, 0.5],              # Γ → Y
            [0.5, 0.0, 0.5, 1.0, 0.5, 0.5],              # Y → T
            [1.0, 0.5, 0.5, 0.5, 0.5, 0.0],              # T → Z
            [0.5, 0.5, 0.0, 0.0, 0.0, 0.0],              # Z → Γ
            [0.0, 0.0, 0.0, 0.0, 0.3, 0.3],              # Γ → Σ₀
            [1.0, 0.7, 0.7, 1.0, 0.5, 0.5],              # U₀ → T
            [0.5, 0.0, 0.5, 0.5, 0.228633, 0.728633],    # Y → C₀
            [0.5, 0.771367, 0.271367, 0.5, 0.5, 0.0],    # A₀ → Z
            [0.0, 0.0, 0.0, 0.5, 0.5, 0.5]               # Γ → L
        ],
        "labels": ["G", "Y", "T", "Z", "G", "SIGMA_0", "U_0", "T", "|", "Y", "C_0", "A_0", "Z", "|", "G", "L"]
    },
    "oF3": {
        # Face-centered orthorhombic with inversion (Fmmm)
        "segments": [
            [0.0, 0.0, 0.0, 0.5, 0.0, 0.5],              # Γ → Y
            [0.5, 0.0, 0.5, 0.5, 0.241963, 0.741963],    # Y → C₀
            [0.5, 0.758037, 0.258037, 0.5, 0.5, 0.0],    # A₀ → Z
            [0.5, 0.5, 0.0, 0.782831, 0.5, 0.282831],    # Z → B₀
            [0.217169, 0.5, 0.717169, 0.0, 0.5, 0.5],    # D₀ → T
            [0.0, 0.5, 0.5, 0.224916, 0.724916, 0.5],    # T → G₀
            [0.775084, 0.275084, 0.5, 0.5, 0.0, 0.5],    # H₀ → Y
            [0.0, 0.5, 0.5, 0.0, 0.0, 0.0],              # T → Γ
            [0.0, 0.0, 0.0, 0.5, 0.5, 0.0],              # Γ → Z
            [0.0, 0.0, 0.0, 0.5, 0.5, 0.5]               # Γ → L
        ],
        "labels": ["G", "Y", "C_0", "A_0", "Z", "B_0", "D_0", "T", "G_0", "H_0", "Y", "|", "T", "G", "|", "G", "Z", "|", "G", "L"]
    },
    
    # Without inversion symmetry entries (selected examples) - using shrink factor scaling
    "mS1": {
        # Monoclinic base-centered without inversion (Cc)
        "segments": [
            [0.0, 0.0, 0.0, 0.386966, 0.386966, 0.0],     # Γ → C (scaled)
            [-0.386966, 0.613034, 0.0, -0.5, 0.5, 0.0],   # C₂ → Y₂
            [-0.5, 0.5, 0.0, 0.0, 0.0, 0.0],              # Y₂ → Γ
            [0.0, 0.0, 0.0, -0.5, 0.5, 0.5],              # Γ → M₂
            [-0.5, 0.5, 0.5, -0.366050, 0.633950, 0.5],   # M₂ → D
            [0.366050, 0.366050, 0.5, 0.0, 0.0, 0.5],     # D₂ → A
            [0.0, 0.0, 0.5, 0.0, 0.0, 0.0],               # A → Γ
            [0.0, 0.5, 0.5, 0.0, 0.0, 0.0],               # L₂ → Γ
            [0.0, 0.0, 0.0, 0.0, 0.5, 0.0],               # Γ → V₂
            [0.0, 0.0, 0.0, -0.386966, -0.386966, 0.0],   # Γ → C'
            [0.386966, -0.613034, 0.0, 0.5, -0.5, 0.0],   # C₂' → Y₂'
            [0.5, -0.5, 0.0, 0.0, 0.0, 0.0],              # Y₂' → Γ
            [0.0, 0.0, 0.0, 0.5, -0.5, -0.5],             # Γ → M₂'
            [0.5, -0.5, -0.5, 0.366050, -0.633950, -0.5], # M₂' → D'
            [-0.366050, -0.366050, -0.5, 0.0, 0.0, -0.5], # D₂' → A'
            [0.0, 0.0, -0.5, 0.0, 0.0, 0.0],              # A' → Γ
            [0.0, -0.5, -0.5, 0.0, 0.0, 0.0],             # L₂' → Γ
            [0.0, 0.0, 0.0, 0.0, -0.5, 0.0]               # Γ → V₂'
        ],
        "labels": ["G", "C", "C_2", "Y_2", "G", "M_2", "D", "D_2", "A", "G", "|", "L_2", "G", "|", "G", "V_2", "|", "G", "C'", "C_2'", "Y_2'", "G", "M_2'", "D'", "D_2'", "A'", "G", "|", "L_2'", "G", "|", "G", "V_2'"]
    },
    "oS1": {
        # Orthorhombic base-centered without inversion (Cmc2_1)
        "segments": [
            [0.0, 0.0, 0.0, -0.5, 0.5, 0.0],             # Γ → Y
            [-0.5, 0.5, 0.0, -0.281380, 0.718620, 0.0],  # Y → C₀
            [0.281380, 0.281380, 0.0, 0.0, 0.0, 0.0],    # Σ₀ → Γ
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.5],              # Γ → Z
            [0.0, 0.0, 0.5, 0.281380, 0.281380, 0.5],    # Z → A₀
            [-0.281380, 0.718620, 0.5, -0.5, 0.5, 0.5],  # E₀ → T
            [-0.5, 0.5, 0.5, -0.5, 0.5, 0.0],            # T → Y
            [0.0, 0.0, 0.0, 0.0, 0.5, 0.0],              # Γ → S
            [0.0, 0.5, 0.0, 0.0, 0.5, 0.5],              # S → R
            [0.0, 0.5, 0.5, 0.0, 0.0, 0.5],              # R → Z
            [0.0, 0.0, 0.5, -0.5, 0.5, 0.5],             # Z → T
            [0.0, 0.0, 0.0, 0.5, -0.5, 0.0],             # Γ → Y'
            [0.5, -0.5, 0.0, 0.281380, -0.718620, 0.0],  # Y' → C₀'
            [-0.281380, -0.281380, 0.0, 0.0, 0.0, 0.0],  # Σ₀' → Γ
            [0.0, 0.0, 0.0, 0.0, 0.0, -0.5],             # Γ → Z'
            [0.0, 0.0, -0.5, -0.281380, -0.281380, -0.5], # Z' → A₀'
            [0.281380, -0.718620, -0.5, 0.5, -0.5, -0.5], # E₀' → T'
            [0.5, -0.5, -0.5, 0.5, -0.5, 0.0],           # T' → Y'
            [0.0, 0.0, 0.0, 0.0, -0.5, 0.0],             # Γ → S'
            [0.0, -0.5, 0.0, 0.0, -0.5, -0.5],           # S' → R'
            [0.0, -0.5, -0.5, 0.0, 0.0, -0.5],           # R' → Z'
            [0.0, 0.0, -0.5, 0.5, -0.5, -0.5]            # Z' → T'
        ],
        "labels": ["G", "Y", "C_0", "SIGMA_0", "G", "Z", "A_0", "E_0", "T", "Y", "|", "G", "S", "R", "Z", "T", "|", "G", "Y'", "C_0'", "SIGMA_0'", "G", "Z'", "A_0'", "E_0'", "T'", "Y'", "|", "G", "S'", "R'", "Z'", "T'"]
    },
    "oI2": {
        # Body-centered orthorhombic without inversion (Ima2)
        "segments": [
            [0.0, 0.0, 0.0, -0.5, 0.5, 0.5],             # Γ → X
            [-0.5, 0.5, 0.5, -0.372123, 0.372123, 0.627877], # X → U₂
            [0.372123, -0.372123, 0.372123, 0.0, 0.0, 0.0], # Y₀ → Γ
            [0.0, 0.0, 0.0, 0.376527, 0.376527, -0.376527], # Γ → Λ₀
            [-0.376527, 0.623473, 0.376527, -0.5, 0.5, 0.5], # G₂ → X
            [0.0, 0.0, 0.0, 0.0, 0.5, 0.0],              # Γ → R
            [0.0, 0.5, 0.0, 0.25, 0.25, 0.25],           # R → W
            [0.25, 0.25, 0.25, 0.5, 0.0, 0.0],           # W → S
            [0.5, 0.0, 0.0, 0.0, 0.0, 0.0],              # S → Γ
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.5],              # Γ → T
            [0.0, 0.0, 0.5, 0.25, 0.25, 0.25],           # T → W
            [0.0, 0.0, 0.0, 0.5, -0.5, -0.5],            # Γ → X'
            [0.5, -0.5, -0.5, 0.372123, -0.372123, -0.627877], # X' → U₂'
            [-0.372123, 0.372123, -0.372123, 0.0, 0.0, 0.0], # Y₀' → Γ
            [0.0, 0.0, 0.0, -0.376527, -0.376527, 0.376527], # Γ → Λ₀'
            [0.376527, -0.623473, -0.376527, 0.5, -0.5, -0.5], # G₂' → X'
            [0.0, 0.0, 0.0, 0.0, -0.5, 0.0],             # Γ → R'
            [0.0, -0.5, 0.0, -0.25, -0.25, -0.25],       # R' → W'
            [-0.25, -0.25, -0.25, -0.5, 0.0, 0.0],       # W' → S'
            [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0],             # S' → Γ
            [0.0, 0.0, 0.0, 0.0, 0.0, -0.5],             # Γ → T'
            [0.0, 0.0, -0.5, -0.25, -0.25, -0.25]        # T' → W'
        ],
        "labels": ["G", "X", "U_2", "Y_0", "G", "LAMBDA_0", "G_2", "X", "|", "G", "R", "W", "S", "G", "T", "W", "|", "G", "X'", "U_2'", "Y_0'", "G", "LAMBDA_0'", "G_2'", "X'", "|", "G", "R'", "W'", "S'", "G", "T'", "W'"]
    },
    "oF2": {
        # Face-centered orthorhombic without inversion (Fdd2)
        "segments": [
            [0.0, 0.0, 0.0, 0.0, 0.5, 0.5],              # Γ → T
            [0.0, 0.5, 0.5, 0.5, 0.5, 1.0],              # T → Z
            [0.5, 0.5, 1.0, 0.5, 0.0, 0.5],              # Z → Y
            [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],              # Y → Γ
            [0.0, 0.0, 0.0, 0.339041, 0.339041, 0.0],    # Γ → Λ₀
            [0.660959, 0.660959, 1.0, 0.5, 0.5, 1.0],    # Q₀ → Z
            [0.0, 0.5, 0.5, 0.195580, 0.695580, 0.5],    # T → G₀
            [0.804420, 0.304420, 0.5, 0.5, 0.0, 0.5],    # H₀ → Y
            [0.0, 0.0, 0.0, 0.5, 0.5, 0.5],              # Γ → L
            [0.0, 0.0, 0.0, 0.0, -0.5, -0.5],            # Γ → T'
            [0.0, -0.5, -0.5, -0.5, -0.5, -1.0],         # T' → Z'
            [-0.5, -0.5, -1.0, -0.5, 0.0, -0.5],         # Z' → Y'
            [-0.5, 0.0, -0.5, 0.0, 0.0, 0.0],            # Y' → Γ
            [0.0, 0.0, 0.0, -0.339041, -0.339041, 0.0],  # Γ → Λ₀'
            [-0.660959, -0.660959, -1.0, -0.5, -0.5, -1.0], # Q₀' → Z'
            [0.0, -0.5, -0.5, -0.195580, -0.695580, -0.5], # T' → G₀'
            [-0.804420, -0.304420, -0.5, -0.5, 0.0, -0.5], # H₀' → Y'
            [0.0, 0.0, 0.0, -0.5, -0.5, -0.5]            # Γ → L'
        ],
        "labels": ["G", "T", "Z", "Y", "G", "LAMBDA_0", "Q_0", "Z", "|", "T", "G_0", "H_0", "Y", "|", "G", "L", "|", "G", "T'", "Z'", "Y'", "G", "LAMBDA_0'", "Q_0'", "Z'", "|", "T'", "G_0'", "H_0'", "Y'", "|", "G", "L'"]
    },
    
    # Non-inversion versions for Bravais lattices
    "aP2_noinv": {
        # Triclinic P1 without inversion (aP2)
        "segments": [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],      # Γ → X
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],      # Y → Γ
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],      # Γ → Z
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],      # R → Γ
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],      # Γ → T
            [1.0, 0.0, 1.0, 0.0, 0.0, 0.0],      # U → Γ
            [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],      # Γ → V
            [0.0, 0.0, 0.0, -1.0, 0.0, 0.0],     # Γ → X'
            [0.0, -1.0, 0.0, 0.0, 0.0, 0.0],     # Y' → Γ
            [0.0, 0.0, 0.0, 0.0, 0.0, -1.0],     # Γ → Z'
            [-1.0, -1.0, -1.0, 0.0, 0.0, 0.0],   # R' → Γ
            [0.0, 0.0, 0.0, 0.0, -1.0, -1.0],    # Γ → T'
            [-1.0, 0.0, -1.0, 0.0, 0.0, 0.0],    # U' → Γ
            [0.0, 0.0, 0.0, -1.0, -1.0, 0.0]     # Γ → V'
        ],
        "labels": ["G", "X", "|", "Y", "G", "|", "G", "Z", "|", "R", "G", "|", "G", "T", "|", "U", "G", "|", "G", "V", "|", "G", "X'", "|", "Y'", "G", "|", "G", "Z'", "|", "R'", "G", "|", "G", "T'", "|", "U'", "G", "|", "G", "V'"]
    },
    "aP3_noinv": {
        # Triclinic P1 without inversion (aP3)
        "segments": [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],      # Γ → X
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],      # Y → Γ
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],      # Γ → Z
            [-1.0, -1.0, 1.0, 0.0, 0.0, 0.0],    # R₂ → Γ
            [0.0, 0.0, 0.0, 0.0, -1.0, 1.0],     # Γ → T₂
            [-1.0, 0.0, 1.0, 0.0, 0.0, 0.0],     # U₂ → Γ
            [0.0, 0.0, 0.0, 1.0, -1.0, 0.0],     # Γ → V₂
            [0.0, 0.0, 0.0, -1.0, 0.0, 0.0],     # Γ → X'
            [0.0, -1.0, 0.0, 0.0, 0.0, 0.0],     # Y' → Γ
            [0.0, 0.0, 0.0, 0.0, 0.0, -1.0],     # Γ → Z'
            [1.0, 1.0, -1.0, 0.0, 0.0, 0.0],     # R₂' → Γ
            [0.0, 0.0, 0.0, 0.0, 1.0, -1.0],     # Γ → T₂'
            [1.0, 0.0, -1.0, 0.0, 0.0, 0.0],     # U₂' → Γ
            [0.0, 0.0, 0.0, -1.0, 1.0, 0.0]      # Γ → V₂'
        ],
        "labels": ["G", "X", "|", "Y", "G", "|", "G", "Z", "|", "R_2", "G", "|", "G", "T_2", "|", "U_2", "G", "|", "G", "V_2", "|", "G", "X'", "|", "Y'", "G", "|", "G", "Z'", "|", "R_2'", "G", "|", "G", "T_2'", "|", "U_2'", "G", "|", "G", "V_2'"]
    },
    "cF1_noinv": {
        # Cubic face-centered without inversion (F23)
        "segments": [
            [0.0, 0.0, 0.0, 0.5, 0.0, 0.5],      # Γ → X
            [0.5, 0.0, 0.5, 0.625, 0.25, 0.625], # X → U
            [0.375, 0.375, 0.75, 0.0, 0.0, 0.0], # K → Γ
            [0.0, 0.0, 0.0, 0.5, 0.5, 0.5],      # Γ → L
            [0.5, 0.5, 0.5, 0.5, 0.25, 0.75],    # L → W
            [0.5, 0.25, 0.75, 0.5, 0.0, 0.5],    # W → X
            [0.5, 0.0, 0.5, 0.75, 0.25, 0.5],    # X → W₂
            [0.0, 0.0, 0.0, -0.5, 0.0, -0.5],    # Γ → X'
            [-0.5, 0.0, -0.5, -0.625, -0.25, -0.625], # X' → U'
            [-0.375, -0.375, -0.75, 0.0, 0.0, 0.0], # K' → Γ
            [0.0, 0.0, 0.0, -0.5, -0.5, -0.5],   # Γ → L'
            [-0.5, -0.5, -0.5, -0.5, -0.25, -0.75], # L' → W'
            [-0.5, -0.25, -0.75, -0.5, 0.0, -0.5], # W' → X'
            [-0.5, 0.0, -0.5, -0.75, -0.25, -0.5] # X' → W₂'
        ],
        "labels": ["G", "X", "U", "K", "G", "L", "W", "X", "W_2", "|", "G", "X'", "U'", "K'", "G", "L'", "W'", "X'", "W_2'"]
    },
    "cF2_noinv": {
        # Cubic face-centered without inversion (F-43m)
        "segments": [
            [0.0, 0.0, 0.0, 0.5, 0.0, 0.5],      # Γ → X
            [0.5, 0.0, 0.5, 0.625, 0.25, 0.625], # X → U
            [0.375, 0.375, 0.75, 0.0, 0.0, 0.0], # K → Γ
            [0.0, 0.0, 0.0, 0.5, 0.5, 0.5],      # Γ → L
            [0.5, 0.5, 0.5, 0.5, 0.25, 0.75],    # L → W
            [0.5, 0.25, 0.75, 0.5, 0.0, 0.5],    # W → X
            [0.0, 0.0, 0.0, -0.5, 0.0, -0.5],    # Γ → X'
            [-0.5, 0.0, -0.5, -0.625, -0.25, -0.625], # X' → U'
            [-0.375, -0.375, -0.75, 0.0, 0.0, 0.0], # K' → Γ
            [0.0, 0.0, 0.0, -0.5, -0.5, -0.5],   # Γ → L'
            [-0.5, -0.5, -0.5, -0.5, -0.25, -0.75], # L' → W'
            [-0.5, -0.25, -0.75, -0.5, 0.0, -0.5] # W' → X'
        ],
        "labels": ["G", "X", "U", "K", "G", "L", "W", "X", "|", "G", "X'", "U'", "K'", "G", "L'", "W'", "X'"]
    },
    "cI1_noinv": {
        # Cubic body-centered without inversion (I-43m)
        "segments": [
            [0.0, 0.0, 0.0, 0.5, -0.5, 0.5],     # Γ → H
            [0.5, -0.5, 0.5, 0.0, 0.0, 0.5],     # H → N
            [0.0, 0.0, 0.5, 0.0, 0.0, 0.0],      # N → Γ
            [0.0, 0.0, 0.0, 0.25, 0.25, 0.25],   # Γ → P
            [0.25, 0.25, 0.25, 0.5, -0.5, 0.5],  # P → H
            [0.25, 0.25, 0.25, 0.0, 0.0, 0.5],   # P → N
            [0.0, 0.0, 0.0, -0.5, 0.5, -0.5],    # Γ → H'
            [-0.5, 0.5, -0.5, 0.0, 0.0, -0.5],   # H' → N'
            [0.0, 0.0, -0.5, 0.0, 0.0, 0.0],     # N' → Γ
            [0.0, 0.0, 0.0, -0.25, -0.25, -0.25], # Γ → P'
            [-0.25, -0.25, -0.25, -0.5, 0.5, -0.5], # P' → H'
            [-0.25, -0.25, -0.25, 0.0, 0.0, -0.5] # P' → N'
        ],
        "labels": ["G", "H", "N", "G", "P", "H", "|", "P", "N", "|", "G", "H'", "N'", "G", "P'", "H'", "|", "P'", "N'"]
    },
    "cP1_noinv": {
        # Cubic primitive without inversion (P2_13)
        "segments": [
            [0.0, 0.0, 0.0, 0.0, 0.5, 0.0],      # Γ → X
            [0.0, 0.5, 0.0, 0.5, 0.5, 0.0],      # X → M
            [0.5, 0.5, 0.0, 0.0, 0.0, 0.0],      # M → Γ
            [0.0, 0.0, 0.0, 0.5, 0.5, 0.5],      # Γ → R
            [0.5, 0.5, 0.5, 0.0, 0.5, 0.0],      # R → X
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.0],      # R → M
            [0.5, 0.5, 0.0, 0.5, 0.0, 0.0],      # M → X₁
            [0.0, 0.0, 0.0, 0.0, -0.5, 0.0],     # Γ → X'
            [0.0, -0.5, 0.0, -0.5, -0.5, 0.0],   # X' → M'
            [-0.5, -0.5, 0.0, 0.0, 0.0, 0.0],    # M' → Γ
            [0.0, 0.0, 0.0, -0.5, -0.5, -0.5],   # Γ → R'
            [-0.5, -0.5, -0.5, 0.0, -0.5, 0.0],  # R' → X'
            [-0.5, -0.5, -0.5, -0.5, -0.5, 0.0], # R' → M'
            [-0.5, -0.5, 0.0, -0.5, 0.0, 0.0]    # M' → X₁'
        ],
        "labels": ["G", "X", "M", "G", "R", "X", "|", "R", "M", "|", "M", "X_1", "|", "G", "X'", "M'", "G", "R'", "X'", "|", "R'", "M'", "|", "M'", "X_1'"]
    },
    "cP2_noinv": {
        # Cubic primitive without inversion (P-43m)
        "segments": [
            [0.0, 0.0, 0.0, 0.0, 0.5, 0.0],      # Γ → X
            [0.0, 0.5, 0.0, 0.5, 0.5, 0.0],      # X → M
            [0.5, 0.5, 0.0, 0.0, 0.0, 0.0],      # M → Γ
            [0.0, 0.0, 0.0, 0.5, 0.5, 0.5],      # Γ → R
            [0.5, 0.5, 0.5, 0.0, 0.5, 0.0],      # R → X
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.0],      # R → M
            [0.0, 0.0, 0.0, 0.0, -0.5, 0.0],     # Γ → X'
            [0.0, -0.5, 0.0, -0.5, -0.5, 0.0],   # X' → M'
            [-0.5, -0.5, 0.0, 0.0, 0.0, 0.0],    # M' → Γ
            [0.0, 0.0, 0.0, -0.5, -0.5, -0.5],   # Γ → R'
            [-0.5, -0.5, -0.5, 0.0, -0.5, 0.0],  # R' → X'
            [-0.5, -0.5, -0.5, -0.5, -0.5, 0.0]  # R' → M'
        ],
        "labels": ["G", "X", "M", "G", "R", "X", "|", "R", "M", "|", "G", "X'", "M'", "G", "R'", "X'", "|", "R'", "M'"]
    },
    "hP1_noinv": {
        # Hexagonal primitive without inversion (P3)
        "segments": [
            [0.0, 0.0, 0.0, 0.5, 0.0, 0.0],      # Γ → M
            [0.5, 0.0, 0.0, 1/3, 1/3, 0.0],      # M → K
            [1/3, 1/3, 0.0, 0.0, 0.0, 0.0],      # K → Γ
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.5],      # Γ → A
            [0.0, 0.0, 0.5, 0.5, 0.0, 0.5],      # A → L
            [0.5, 0.0, 0.5, 1/3, 1/3, 0.5],      # L → H
            [1/3, 1/3, 0.5, 0.0, 0.0, 0.5],      # H → A
            [0.5, 0.0, 0.5, 0.5, 0.0, 0.0],      # L → M
            [1/3, 1/3, 0.5, 1/3, 1/3, 0.0],      # H → K
            [1/3, 1/3, 0.0, 1/3, 1/3, -0.5],     # K → H₂
            [0.0, 0.0, 0.0, -0.5, 0.0, 0.0],     # Γ → M'
            [-0.5, 0.0, 0.0, -1/3, -1/3, 0.0],   # M' → K'
            [-1/3, -1/3, 0.0, 0.0, 0.0, 0.0],    # K' → Γ
            [0.0, 0.0, 0.0, 0.0, 0.0, -0.5],     # Γ → A'
            [0.0, 0.0, -0.5, -0.5, 0.0, -0.5],   # A' → L'
            [-0.5, 0.0, -0.5, -1/3, -1/3, -0.5], # L' → H'
            [-1/3, -1/3, -0.5, 0.0, 0.0, -0.5],  # H' → A'
            [-0.5, 0.0, -0.5, -0.5, 0.0, 0.0],   # L' → M'
            [-1/3, -1/3, -0.5, -1/3, -1/3, 0.0], # H' → K'
            [-1/3, -1/3, 0.0, -1/3, -1/3, 0.5]   # K' → H₂'
        ],
        "labels": ["G", "M", "K", "G", "A", "L", "H", "A", "|", "L", "M", "|", "H", "K", "|", "K", "H_2", "|", "G", "M'", "K'", "G", "A'", "L'", "H'", "A'", "|", "L'", "M'", "|", "H'", "K'", "|", "K'", "H_2'"]
    },
    "hP2_noinv": {
        # Hexagonal primitive without inversion (P-6m2)
        "segments": [
            [0.0, 0.0, 0.0, 0.5, 0.0, 0.0],      # Γ → M
            [0.5, 0.0, 0.0, 1/3, 1/3, 0.0],      # M → K
            [1/3, 1/3, 0.0, 0.0, 0.0, 0.0],      # K → Γ
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.5],      # Γ → A
            [0.0, 0.0, 0.5, 0.5, 0.0, 0.5],      # A → L
            [0.5, 0.0, 0.5, 1/3, 1/3, 0.5],      # L → H
            [1/3, 1/3, 0.5, 0.0, 0.0, 0.5],      # H → A
            [0.5, 0.0, 0.5, 0.5, 0.0, 0.0],      # L → M
            [1/3, 1/3, 0.5, 1/3, 1/3, 0.0],      # H → K
            [0.0, 0.0, 0.0, -0.5, 0.0, 0.0],     # Γ → M'
            [-0.5, 0.0, 0.0, -1/3, -1/3, 0.0],   # M' → K'
            [-1/3, -1/3, 0.0, 0.0, 0.0, 0.0],    # K' → Γ
            [0.0, 0.0, 0.0, 0.0, 0.0, -0.5],     # Γ → A'
            [0.0, 0.0, -0.5, -0.5, 0.0, -0.5],   # A' → L'
            [-0.5, 0.0, -0.5, -1/3, -1/3, -0.5], # L' → H'
            [-1/3, -1/3, -0.5, 0.0, 0.0, -0.5],  # H' → A'
            [-0.5, 0.0, -0.5, -0.5, 0.0, 0.0],   # L' → M'
            [-1/3, -1/3, -0.5, -1/3, -1/3, 0.0]  # H' → K'
        ],
        "labels": ["G", "M", "K", "G", "A", "L", "H", "A", "|", "L", "M", "|", "H", "K", "|", "G", "M'", "K'", "G", "A'", "L'", "H'", "A'", "|", "L'", "M'", "|", "H'", "K'"]
    },
    "hR1_noinv": {
        # Rhombohedral without inversion (R3m)
        "segments": [
            [0.0, 0.0, 0.0, 0.5, 0.5, 0.5],              # Γ → T
            [0.5, 0.5, 0.5, 0.744565, 0.255435, 0.5],    # T → H₂
            [0.5, -0.255435, 0.255435, 0.5, 0.0, 0.0],   # H₀ → L
            [0.5, 0.0, 0.0, 0.0, 0.0, 0.0],              # L → Γ
            [0.0, 0.0, 0.0, 0.377717, -0.377717, 0.0],   # Γ → S₀
            [0.622283, 0.0, 0.377717, 0.5, 0.0, 0.5],    # S₂ → F
            [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],              # F → Γ
            [0.0, 0.0, 0.0, -0.5, -0.5, -0.5],           # Γ → T'
            [-0.5, -0.5, -0.5, -0.744565, -0.255435, -0.5], # T' → H₂'
            [-0.5, 0.255435, -0.255435, -0.5, 0.0, 0.0], # H₀' → L'
            [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0],             # L' → Γ
            [0.0, 0.0, 0.0, -0.377717, 0.377717, 0.0],   # Γ → S₀'
            [-0.622283, 0.0, -0.377717, -0.5, 0.0, -0.5], # S₂' → F'
            [-0.5, 0.0, -0.5, 0.0, 0.0, 0.0]             # F' → Γ
        ],
        "labels": ["G", "T", "H_2", "H_0", "L", "G", "S_0", "S_2", "F", "G", "|", "G", "T'", "H_2'", "H_0'", "L'", "G", "S_0'", "S_2'", "F'", "G"]
    },
    "hR2_noinv": {
        # Rhombohedral without inversion (R3m variant 2)
        "segments": [
            [0.0, 0.0, 0.0, 0.5, 0.0, 0.0],              # Γ → L
            [0.5, 0.0, 0.0, 0.5, -0.5, 0.5],             # L → T
            [0.5, -0.5, 0.5, 0.302174, -0.697826, 0.302174], # T → P₀
            [0.302174, 0.302174, 0.302174, 0.0, 0.0, 0.0], # P₂ → Γ
            [0.0, 0.0, 0.0, 0.5, -0.5, 0.0],             # Γ → F
            [0.0, 0.0, 0.0, -0.5, 0.0, 0.0],             # Γ → L'
            [-0.5, 0.0, 0.0, -0.5, 0.5, -0.5],           # L' → T'
            [-0.5, 0.5, -0.5, -0.302174, 0.697826, -0.302174], # T' → P₀'
            [-0.302174, -0.302174, -0.302174, 0.0, 0.0, 0.0], # P₂' → Γ
            [0.0, 0.0, 0.0, -0.5, 0.5, 0.0]              # Γ → F'
        ],
        "labels": ["G", "L", "T", "P_0", "P_2", "G", "F", "|", "G", "L'", "T'", "P_0'", "P_2'", "G", "F'"]
    },
    "mC1_noinv": {
        # Monoclinic C-centered without inversion (Cc)
        "segments": [
            [0.0, 0.0, 0.0, 0.386966, 0.386966, 0.0],    # Γ → C
            [-0.386966, 0.613034, 0.0, -0.5, 0.5, 0.0],  # C₂ → Y₂
            [-0.5, 0.5, 0.0, 0.0, 0.0, 0.0],             # Y₂ → Γ
            [0.0, 0.0, 0.0, -0.5, 0.5, 0.5],             # Γ → M₂
            [-0.5, 0.5, 0.5, -0.366050, 0.633950, 0.5],  # M₂ → D
            [0.366050, 0.366050, 0.5, 0.0, 0.0, 0.5],    # D₂ → A
            [0.0, 0.0, 0.5, 0.0, 0.0, 0.0],              # A → Γ
            [0.0, 0.5, 0.5, 0.0, 0.0, 0.0],              # L₂ → Γ
            [0.0, 0.0, 0.0, 0.0, 0.5, 0.0],              # Γ → V₂
            [0.0, 0.0, 0.0, -0.386966, -0.386966, 0.0],  # Γ → C'
            [0.386966, -0.613034, 0.0, 0.5, -0.5, 0.0],  # C₂' → Y₂'
            [0.5, -0.5, 0.0, 0.0, 0.0, 0.0],             # Y₂' → Γ
            [0.0, 0.0, 0.0, 0.5, -0.5, -0.5],            # Γ → M₂'
            [0.5, -0.5, -0.5, 0.366050, -0.633950, -0.5], # M₂' → D'
            [-0.366050, -0.366050, -0.5, 0.0, 0.0, -0.5], # D₂' → A'
            [0.0, 0.0, -0.5, 0.0, 0.0, 0.0],             # A' → Γ
            [0.0, -0.5, -0.5, 0.0, 0.0, 0.0],            # L₂' → Γ
            [0.0, 0.0, 0.0, 0.0, -0.5, 0.0]              # Γ → V₂'
        ],
        "labels": ["G", "C", "C_2", "Y_2", "G", "M_2", "D", "D_2", "A", "G", "|", "L_2", "G", "|", "G", "V_2", "|", "G", "C'", "C_2'", "Y_2'", "G", "M_2'", "D'", "D_2'", "A'", "G", "|", "L_2'", "G", "|", "G", "V_2'"]
    },
    "mC2_noinv": {
        # Monoclinic C-centered without inversion (C2)
        "segments": [
            [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],      # Γ → Y
            [1.0, 1.0, 0.0, 1.0, 1.0, 1.0],      # Y → M
            [1.0, 1.0, 1.0, 0.0, 0.0, 1.0],      # M → A
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],      # A → Γ
            [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],      # L₂ → Γ
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],      # Γ → V₂
            [0.0, 0.0, 0.0, -1.0, -1.0, 0.0],    # Γ → Y'
            [-1.0, -1.0, 0.0, -1.0, -1.0, -1.0], # Y' → M'
            [-1.0, -1.0, -1.0, 0.0, 0.0, -1.0],  # M' → A'
            [0.0, 0.0, -1.0, 0.0, 0.0, 0.0],     # A' → Γ
            [0.0, -1.0, -1.0, 0.0, 0.0, 0.0],    # L₂' → Γ
            [0.0, 0.0, 0.0, 0.0, -1.0, 0.0]      # Γ → V₂'
        ],
        "labels": ["G", "Y", "M", "A", "G", "|", "L_2", "G", "|", "G", "V_2", "|", "G", "Y'", "M'", "A'", "G", "|", "L_2'", "G", "|", "G", "V_2'"]
    },
    "mC3_noinv": {
        # Monoclinic C-centered without inversion (Cm)
        "segments": [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.5],              # Γ → A
            [0.0, 0.0, 0.5, 0.389211, 0.389211, 0.5],    # A → I₂
            [-0.389211, 0.610789, 0.5, -0.5, 0.5, 0.5],  # I → M₂
            [-0.5, 0.5, 0.5, 0.0, 0.0, 0.0],             # M₂ → Γ
            [0.0, 0.0, 0.0, 0.5, 0.5, 0.0],              # Γ → Y
            [0.0, 0.5, 0.5, 0.0, 0.0, 0.0],              # L₂ → Γ
            [0.0, 0.0, 0.0, 0.0, 0.5, 0.0],              # Γ → V₂
            [0.0, 0.0, 0.0, 0.0, 0.0, -0.5],             # Γ → A'
            [0.0, 0.0, -0.5, -0.389211, -0.389211, -0.5], # A' → I₂'
            [0.389211, -0.610789, -0.5, 0.5, -0.5, -0.5], # I' → M₂'
            [0.5, -0.5, -0.5, 0.0, 0.0, 0.0],            # M₂' → Γ
            [0.0, 0.0, 0.0, -0.5, -0.5, 0.0],            # Γ → Y'
            [0.0, -0.5, -0.5, 0.0, 0.0, 0.0],            # L₂' → Γ
            [0.0, 0.0, 0.0, 0.0, -0.5, 0.0]              # Γ → V₂'
        ],
        "labels": ["G", "A", "I_2", "I", "M_2", "G", "Y", "|", "L_2", "G", "|", "G", "V_2", "|", "G", "A'", "I_2'", "I'", "M_2'", "G", "Y'", "|", "L_2'", "G", "|", "G", "V_2'"]
    },
    "mP1_noinv": {
        # Monoclinic primitive without inversion (Pc)
        "segments": [
            [0.0, 0.0, 0.0, 0.0, 0.5, 0.0],      # Γ → Z
            [0.0, 0.5, 0.0, 0.0, 0.5, 0.5],      # Z → D
            [0.0, 0.5, 0.5, 0.0, 0.0, 0.5],      # D → B
            [0.0, 0.0, 0.5, 0.0, 0.0, 0.0],      # B → Γ
            [0.0, 0.0, 0.0, -0.5, 0.0, 0.5],     # Γ → A
            [-0.5, 0.0, 0.5, -0.5, 0.5, 0.5],    # A → E
            [-0.5, 0.5, 0.5, 0.0, 0.5, 0.0],     # E → Z
            [0.0, 0.5, 0.0, -0.5, 0.5, 0.0],     # Z → C₂
            [-0.5, 0.5, 0.0, -0.5, 0.0, 0.0],    # C₂ → Y₂
            [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0],     # Y₂ → Γ
            [0.0, 0.0, 0.0, 0.0, -0.5, 0.0],     # Γ → Z'
            [0.0, -0.5, 0.0, 0.0, -0.5, -0.5],   # Z' → D'
            [0.0, -0.5, -0.5, 0.0, 0.0, -0.5],   # D' → B'
            [0.0, 0.0, -0.5, 0.0, 0.0, 0.0],     # B' → Γ
            [0.0, 0.0, 0.0, 0.5, 0.0, -0.5],     # Γ → A'
            [0.5, 0.0, -0.5, 0.5, -0.5, -0.5],   # A' → E'
            [0.5, -0.5, -0.5, 0.0, -0.5, 0.0],   # E' → Z'
            [0.0, -0.5, 0.0, 0.5, -0.5, 0.0],    # Z' → C₂'
            [0.5, -0.5, 0.0, 0.5, 0.0, 0.0],     # C₂' → Y₂'
            [0.5, 0.0, 0.0, 0.0, 0.0, 0.0]       # Y₂' → Γ
        ],
        "labels": ["G", "Z", "D", "B", "G", "A", "E", "Z", "C_2", "Y_2", "G", "|", "G", "Z'", "D'", "B'", "G", "A'", "E'", "Z'", "C_2'", "Y_2'", "G"]
    },
    "oA1_noinv": {
        # Orthorhombic A-centered without inversion (Amm2)
        "segments": [
            [0.0, 0.0, 0.0, -0.5, 0.5, 0.0],             # Γ → Y
            [-0.5, 0.5, 0.0, -0.276078, 0.723922, 0.0],  # Y → C₀
            [0.276078, 0.276078, 0.0, 0.0, 0.0, 0.0],    # Σ₀ → Γ
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.5],              # Γ → Z
            [0.0, 0.0, 0.5, 0.276078, 0.276078, 0.5],    # Z → A₀
            [-0.276078, 0.723922, 0.5, -0.5, 0.5, 0.5],  # E₀ → T
            [-0.5, 0.5, 0.5, -0.5, 0.5, 0.0],            # T → Y
            [0.0, 0.0, 0.0, 0.0, 0.5, 0.0],              # Γ → S
            [0.0, 0.5, 0.0, 0.0, 0.5, 0.5],              # S → R
            [0.0, 0.5, 0.5, 0.0, 0.0, 0.5],              # R → Z
            [0.0, 0.0, 0.5, -0.5, 0.5, 0.5],             # Z → T
            [0.0, 0.0, 0.0, 0.5, -0.5, 0.0],             # Γ → Y'
            [0.5, -0.5, 0.0, 0.276078, -0.723922, 0.0],  # Y' → C₀'
            [-0.276078, -0.276078, 0.0, 0.0, 0.0, 0.0],  # Σ₀' → Γ
            [0.0, 0.0, 0.0, 0.0, 0.0, -0.5],             # Γ → Z'
            [0.0, 0.0, -0.5, -0.276078, -0.276078, -0.5], # Z' → A₀'
            [0.276078, -0.723922, -0.5, 0.5, -0.5, -0.5], # E₀' → T'
            [0.5, -0.5, -0.5, 0.5, -0.5, 0.0],           # T' → Y'
            [0.0, 0.0, 0.0, 0.0, -0.5, 0.0],             # Γ → S'
            [0.0, -0.5, 0.0, 0.0, -0.5, -0.5],           # S' → R'
            [0.0, -0.5, -0.5, 0.0, 0.0, -0.5],           # R' → Z'
            [0.0, 0.0, -0.5, 0.5, -0.5, -0.5]            # Z' → T'
        ],
        "labels": ["G", "Y", "C_0", "SIGMA_0", "G", "Z", "A_0", "E_0", "T", "Y", "|", "G", "S", "R", "Z", "T", "|", "G", "Y'", "C_0'", "SIGMA_0'", "G", "Z'", "A_0'", "E_0'", "T'", "Y'", "|", "G", "S'", "R'", "Z'", "T'"]
    },
    "oA2_noinv": {
        # Orthorhombic A-centered without inversion (Amm2 variant 2)
        "segments": [
            [0.0, 0.0, 0.0, 0.5, 0.5, 0.0],              # Γ → Y
            [0.5, 0.5, 0.0, 0.335185, 0.664815, 0.0],    # Y → F₀
            [-0.335185, 0.335185, 0.0, 0.0, 0.0, 0.0],   # Δ₀ → Γ
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.5],              # Γ → Z
            [0.0, 0.0, 0.5, -0.335185, 0.335185, 0.5],   # Z → B₀
            [0.335185, 0.664815, 0.5, 0.5, 0.5, 0.5],    # G₀ → T
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.0],              # T → Y
            [0.0, 0.0, 0.0, 0.0, 0.5, 0.0],              # Γ → S
            [0.0, 0.5, 0.0, 0.0, 0.5, 0.5],              # S → R
            [0.0, 0.5, 0.5, 0.0, 0.0, 0.5],              # R → Z
            [0.0, 0.0, 0.5, 0.5, 0.5, 0.5],              # Z → T
            [0.0, 0.0, 0.0, -0.5, -0.5, 0.0],            # Γ → Y'
            [-0.5, -0.5, 0.0, -0.335185, -0.664815, 0.0], # Y' → F₀'
            [0.335185, -0.335185, 0.0, 0.0, 0.0, 0.0],   # Δ₀' → Γ
            [0.0, 0.0, 0.0, 0.0, 0.0, -0.5],             # Γ → Z'
            [0.0, 0.0, -0.5, 0.335185, -0.335185, -0.5], # Z' → B₀'
            [-0.335185, -0.664815, -0.5, -0.5, -0.5, -0.5], # G₀' → T'
            [-0.5, -0.5, -0.5, -0.5, -0.5, 0.0],         # T' → Y'
            [0.0, 0.0, 0.0, 0.0, -0.5, 0.0],             # Γ → S'
            [0.0, -0.5, 0.0, 0.0, -0.5, -0.5],           # S' → R'
            [0.0, -0.5, -0.5, 0.0, 0.0, -0.5],           # R' → Z'
            [0.0, 0.0, -0.5, -0.5, -0.5, -0.5]           # Z' → T'
        ],
        "labels": ["G", "Y", "F_0", "DELTA_0", "G", "Z", "B_0", "G_0", "T", "Y", "|", "G", "S", "R", "Z", "T", "|", "G", "Y'", "F_0'", "DELTA_0'", "G", "Z'", "B_0'", "G_0'", "T'", "Y'", "|", "G", "S'", "R'", "Z'", "T'"]
    },
    "oC1_noinv": {
        # Orthorhombic C-centered without inversion (Cmc2_1)
        "segments": [
            [0.0, 0.0, 0.0, -0.5, 0.5, 0.0],             # Γ → Y
            [-0.5, 0.5, 0.0, -0.281380, 0.718620, 0.0],  # Y → C₀
            [0.281380, 0.281380, 0.0, 0.0, 0.0, 0.0],    # Σ₀ → Γ
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.5],              # Γ → Z
            [0.0, 0.0, 0.5, 0.281380, 0.281380, 0.5],    # Z → A₀
            [-0.281380, 0.718620, 0.5, -0.5, 0.5, 0.5],  # E₀ → T
            [-0.5, 0.5, 0.5, -0.5, 0.5, 0.0],            # T → Y
            [0.0, 0.0, 0.0, 0.0, 0.5, 0.0],              # Γ → S
            [0.0, 0.5, 0.0, 0.0, 0.5, 0.5],              # S → R
            [0.0, 0.5, 0.5, 0.0, 0.0, 0.5],              # R → Z
            [0.0, 0.0, 0.5, -0.5, 0.5, 0.5],             # Z → T
            [0.0, 0.0, 0.0, 0.5, -0.5, 0.0],             # Γ → Y'
            [0.5, -0.5, 0.0, 0.281380, -0.718620, 0.0],  # Y' → C₀'
            [-0.281380, -0.281380, 0.0, 0.0, 0.0, 0.0],  # Σ₀' → Γ
            [0.0, 0.0, 0.0, 0.0, 0.0, -0.5],             # Γ → Z'
            [0.0, 0.0, -0.5, -0.281380, -0.281380, -0.5], # Z' → A₀'
            [0.281380, -0.718620, -0.5, 0.5, -0.5, -0.5], # E₀' → T'
            [0.5, -0.5, -0.5, 0.5, -0.5, 0.0],           # T' → Y'
            [0.0, 0.0, 0.0, 0.0, -0.5, 0.0],             # Γ → S'
            [0.0, -0.5, 0.0, 0.0, -0.5, -0.5],           # S' → R'
            [0.0, -0.5, -0.5, 0.0, 0.0, -0.5],           # R' → Z'
            [0.0, 0.0, -0.5, 0.5, -0.5, -0.5]            # Z' → T'
        ],
        "labels": ["G", "Y", "C_0", "SIGMA_0", "G", "Z", "A_0", "E_0", "T", "Y", "|", "G", "S", "R", "Z", "T", "|", "G", "Y'", "C_0'", "SIGMA_0'", "G", "Z'", "A_0'", "E_0'", "T'", "Y'", "|", "G", "S'", "R'", "Z'", "T'"]
    },
    "oC2_noinv": {
        # Orthorhombic C-centered without inversion (Cmc2_1 variant 2)
        "segments": [
            [0.0, 0.0, 0.0, 0.5, 0.5, 0.0],              # Γ → Y
            [0.5, 0.5, 0.0, 0.346380, 0.653620, 0.0],    # Y → F₀
            [-0.346380, 0.346380, 0.0, 0.0, 0.0, 0.0],   # Δ₀ → Γ
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.5],              # Γ → Z
            [0.0, 0.0, 0.5, -0.346380, 0.346380, 0.5],   # Z → B₀
            [0.346380, 0.653620, 0.5, 0.5, 0.5, 0.5],    # G₀ → T
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.0],              # T → Y
            [0.0, 0.0, 0.0, 0.0, 0.5, 0.0],              # Γ → S
            [0.0, 0.5, 0.0, 0.0, 0.5, 0.5],              # S → R
            [0.0, 0.5, 0.5, 0.0, 0.0, 0.5],              # R → Z
            [0.0, 0.0, 0.5, 0.5, 0.5, 0.5],              # Z → T
            [0.0, 0.0, 0.0, -0.5, -0.5, 0.0],            # Γ → Y'
            [-0.5, -0.5, 0.0, -0.346380, -0.653620, 0.0], # Y' → F₀'
            [0.346380, -0.346380, 0.0, 0.0, 0.0, 0.0],   # Δ₀' → Γ
            [0.0, 0.0, 0.0, 0.0, 0.0, -0.5],             # Γ → Z'
            [0.0, 0.0, -0.5, 0.346380, -0.346380, -0.5], # Z' → B₀'
            [-0.346380, -0.653620, -0.5, -0.5, -0.5, -0.5], # G₀' → T'
            [-0.5, -0.5, -0.5, -0.5, -0.5, 0.0],         # T' → Y'
            [0.0, 0.0, 0.0, 0.0, -0.5, 0.0],             # Γ → S'
            [0.0, -0.5, 0.0, 0.0, -0.5, -0.5],           # S' → R'
            [0.0, -0.5, -0.5, 0.0, 0.0, -0.5],           # R' → Z'
            [0.0, 0.0, -0.5, -0.5, -0.5, -0.5]           # Z' → T'
        ],
        "labels": ["G", "Y", "F_0", "DELTA_0", "G", "Z", "B_0", "G_0", "T", "Y", "|", "G", "S", "R", "Z", "T", "|", "G", "Y'", "F_0'", "DELTA_0'", "G", "Z'", "B_0'", "G_0'", "T'", "Y'", "|", "G", "S'", "R'", "Z'", "T'"]
    },
    "oF1_noinv": {
        # Orthorhombic face-centered without inversion (Fmm2)
        "segments": [
            [0.0, 0.0, 0.0, 0.5, 0.0, 0.5],              # Γ → Y
            [0.5, 0.0, 0.5, 1.0, 0.5, 0.5],              # Y → T
            [1.0, 0.5, 0.5, 0.5, 0.5, 0.0],              # T → Z
            [0.5, 0.5, 0.0, 0.0, 0.0, 0.0],              # Z → Γ
            [0.0, 0.0, 0.0, 0.0, 0.3, 0.3],              # Γ → Σ₀
            [1.0, 0.7, 0.7, 1.0, 0.5, 0.5],              # U₀ → T
            [0.5, 0.0, 0.5, 0.5, 0.228633, 0.728633],    # Y → C₀
            [0.5, 0.771367, 0.271367, 0.5, 0.5, 0.0],    # A₀ → Z
            [0.0, 0.0, 0.0, 0.5, 0.5, 0.5],              # Γ → L
            [0.0, 0.0, 0.0, -0.5, 0.0, -0.5],            # Γ → Y'
            [-0.5, 0.0, -0.5, -1.0, -0.5, -0.5],         # Y' → T'
            [-1.0, -0.5, -0.5, -0.5, -0.5, 0.0],         # T' → Z'
            [-0.5, -0.5, 0.0, 0.0, 0.0, 0.0],            # Z' → Γ
            [0.0, 0.0, 0.0, 0.0, -0.3, -0.3],            # Γ → Σ₀'
            [-1.0, -0.7, -0.7, -1.0, -0.5, -0.5],        # U₀' → T'
            [-0.5, 0.0, -0.5, -0.5, -0.228633, -0.728633], # Y' → C₀'
            [-0.5, -0.771367, -0.271367, -0.5, -0.5, 0.0], # A₀' → Z'
            [0.0, 0.0, 0.0, -0.5, -0.5, -0.5]           # Γ → L'
        ],
        "labels": ["G", "Y", "T", "Z", "G", "SIGMA_0", "U_0", "T", "|", "Y", "C_0", "A_0", "Z", "|", "G", "L", "|", "G", "Y'", "T'", "Z'", "G", "SIGMA_0'", "U_0'", "T'", "|", "Y'", "C_0'", "A_0'", "Z'", "|", "G", "L'"]
    },
    "oF3_noinv": {
        # Orthorhombic face-centered without inversion (Fmm2 variant 3)
        "segments": [
            [0.0, 0.0, 0.0, 0.5, 0.0, 0.5],              # Γ → Y
            [0.5, 0.0, 0.5, 0.5, 0.241963, 0.741963],    # Y → C₀
            [0.5, 0.758037, 0.258037, 0.5, 0.5, 0.0],    # A₀ → Z
            [0.5, 0.5, 0.0, 0.782831, 0.5, 0.282831],    # Z → B₀
            [0.217169, 0.5, 0.717169, 0.0, 0.5, 0.5],    # D₀ → T
            [0.0, 0.5, 0.5, 0.224916, 0.724916, 0.5],    # T → G₀
            [0.775084, 0.275084, 0.5, 0.5, 0.0, 0.5],    # H₀ → Y
            [0.0, 0.5, 0.5, 0.0, 0.0, 0.0],              # T → Γ
            [0.0, 0.0, 0.0, 0.5, 0.5, 0.0],              # Γ → Z
            [0.0, 0.0, 0.0, 0.5, 0.5, 0.5],              # Γ → L
            [0.0, 0.0, 0.0, -0.5, 0.0, -0.5],            # Γ → Y'
            [-0.5, 0.0, -0.5, -0.5, -0.241963, -0.741963], # Y' → C₀'
            [-0.5, -0.758037, -0.258037, -0.5, -0.5, 0.0], # A₀' → Z'
            [-0.5, -0.5, 0.0, -0.782831, -0.5, -0.282831], # Z' → B₀'
            [-0.217169, -0.5, -0.717169, 0.0, -0.5, -0.5], # D₀' → T'
            [0.0, -0.5, -0.5, -0.224916, -0.724916, -0.5], # T' → G₀'
            [-0.775084, -0.275084, -0.5, -0.5, 0.0, -0.5], # H₀' → Y'
            [0.0, -0.5, -0.5, 0.0, 0.0, 0.0],            # T' → Γ
            [0.0, 0.0, 0.0, -0.5, -0.5, 0.0],            # Γ → Z'
            [0.0, 0.0, 0.0, -0.5, -0.5, -0.5]           # Γ → L'
        ],
        "labels": ["G", "Y", "C_0", "A_0", "Z", "B_0", "D_0", "T", "G_0", "H_0", "Y", "|", "T", "G", "|", "G", "Z", "|", "G", "L", "|", "G", "Y'", "C_0'", "A_0'", "Z'", "B_0'", "D_0'", "T'", "G_0'", "H_0'", "Y'", "|", "T'", "G", "|", "G", "Z'", "|", "G", "L'"]
    },
    "oI1_noinv": {
        # Orthorhombic body-centered without inversion (Imm2)
        "segments": [
            [0.0, 0.0, 0.0, 0.5, 0.5, -0.5],             # Γ → X
            [0.5, 0.5, -0.5, 0.268084, 0.731916, -0.268084], # X → F₂
            [-0.268084, 0.268084, 0.268084, 0.0, 0.0, 0.0], # Σ₀ → Γ
            [0.0, 0.0, 0.0, 0.277064, -0.277064, 0.277064], # Γ → Y₀
            [0.722936, 0.277064, -0.277064, 0.5, 0.5, -0.5], # U₀ → X
            [0.0, 0.0, 0.0, 0.0, 0.5, 0.0],              # Γ → R
            [0.0, 0.5, 0.0, 0.25, 0.25, 0.25],           # R → W
            [0.25, 0.25, 0.25, 0.5, 0.0, 0.0],           # W → S
            [0.5, 0.0, 0.0, 0.0, 0.0, 0.0],              # S → Γ
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.5],              # Γ → T
            [0.0, 0.0, 0.5, 0.25, 0.25, 0.25],           # T → W
            [0.0, 0.0, 0.0, -0.5, -0.5, 0.5],            # Γ → X'
            [-0.5, -0.5, 0.5, -0.268084, -0.731916, 0.268084], # X' → F₂'
            [0.268084, -0.268084, -0.268084, 0.0, 0.0, 0.0], # Σ₀' → Γ
            [0.0, 0.0, 0.0, -0.277064, 0.277064, -0.277064], # Γ → Y₀'
            [-0.722936, -0.277064, 0.277064, -0.5, -0.5, 0.5], # U₀' → X'
            [0.0, 0.0, 0.0, 0.0, -0.5, 0.0],             # Γ → R'
            [0.0, -0.5, 0.0, -0.25, -0.25, -0.25],       # R' → W'
            [-0.25, -0.25, -0.25, -0.5, 0.0, 0.0],       # W' → S'
            [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0],             # S' → Γ
            [0.0, 0.0, 0.0, 0.0, 0.0, -0.5],             # Γ → T'
            [0.0, 0.0, -0.5, -0.25, -0.25, -0.25]        # T' → W'
        ],
        "labels": ["G", "X", "F_2", "SIGMA_0", "G", "Y_0", "U_0", "X", "|", "G", "R", "W", "S", "G", "T", "W", "|", "G", "X'", "F_2'", "SIGMA_0'", "G", "Y_0'", "U_0'", "X'", "|", "G", "R'", "W'", "S'", "G", "T'", "W'"]
    },
    "oI3_noinv": {
        # Orthorhombic body-centered without inversion (Imm2 variant 3)
        "segments": [
            [0.0, 0.0, 0.0, 0.5, -0.5, 0.5],             # Γ → X
            [0.5, -0.5, 0.5, 0.340078, -0.340078, 0.659922], # X → F₀
            [-0.340078, 0.340078, 0.340078, 0.0, 0.0, 0.0], # Σ₀ → Γ
            [0.0, 0.0, 0.0, 0.351681, 0.351681, -0.351681], # Γ → Λ₀
            [-0.351681, 0.648319, 0.351681, 0.5, -0.5, 0.5], # G₀ → X
            [0.0, 0.0, 0.0, 0.0, 0.5, 0.0],              # Γ → R
            [0.0, 0.5, 0.0, 0.25, 0.25, 0.25],           # R → W
            [0.25, 0.25, 0.25, 0.5, 0.0, 0.0],           # W → S
            [0.5, 0.0, 0.0, 0.0, 0.0, 0.0],              # S → Γ
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.5],              # Γ → T
            [0.0, 0.0, 0.5, 0.25, 0.25, 0.25],           # T → W
            [0.0, 0.0, 0.0, -0.5, 0.5, -0.5],            # Γ → X'
            [-0.5, 0.5, -0.5, -0.340078, 0.340078, -0.659922], # X' → F₀'
            [0.340078, -0.340078, -0.340078, 0.0, 0.0, 0.0], # Σ₀' → Γ
            [0.0, 0.0, 0.0, -0.351681, -0.351681, 0.351681], # Γ → Λ₀'
            [0.351681, -0.648319, -0.351681, -0.5, 0.5, -0.5], # G₀' → X'
            [0.0, 0.0, 0.0, 0.0, -0.5, 0.0],             # Γ → R'
            [0.0, -0.5, 0.0, -0.25, -0.25, -0.25],       # R' → W'
            [-0.25, -0.25, -0.25, -0.5, 0.0, 0.0],       # W' → S'
            [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0],             # S' → Γ
            [0.0, 0.0, 0.0, 0.0, 0.0, -0.5],             # Γ → T'
            [0.0, 0.0, -0.5, -0.25, -0.25, -0.25]        # T' → W'
        ],
        "labels": ["G", "X", "F_0", "SIGMA_0", "G", "LAMBDA_0", "G_0", "X", "|", "G", "R", "W", "S", "G", "T", "W", "|", "G", "X'", "F_0'", "SIGMA_0'", "G", "LAMBDA_0'", "G_0'", "X'", "|", "G", "R'", "W'", "S'", "G", "T'", "W'"]
    },
    "oP1_noinv": {
        # Orthorhombic primitive without inversion (Pmm2)
        "segments": [
            [0.0, 0.0, 0.0, 0.5, 0.0, 0.0],      # Γ → X
            [0.5, 0.0, 0.0, 0.5, 0.5, 0.0],      # X → S
            [0.5, 0.5, 0.0, 0.0, 0.5, 0.0],      # S → Y
            [0.0, 0.5, 0.0, 0.0, 0.0, 0.0],      # Y → Γ
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.5],      # Γ → Z
            [0.0, 0.0, 0.5, 0.5, 0.0, 0.5],      # Z → U
            [0.5, 0.0, 0.5, 0.5, 0.5, 0.5],      # U → R
            [0.5, 0.5, 0.5, 0.0, 0.5, 0.5],      # R → T
            [0.0, 0.5, 0.5, 0.0, 0.0, 0.5],      # T → Z
            [0.5, 0.0, 0.0, 0.5, 0.0, 0.5],      # X → U
            [0.0, 0.5, 0.0, 0.0, 0.5, 0.5],      # Y → T
            [0.5, 0.5, 0.0, 0.5, 0.5, 0.5],      # S → R
            [0.0, 0.0, 0.0, -0.5, 0.0, 0.0],     # Γ → X'
            [-0.5, 0.0, 0.0, -0.5, -0.5, 0.0],   # X' → S'
            [-0.5, -0.5, 0.0, 0.0, -0.5, 0.0],   # S' → Y'
            [0.0, -0.5, 0.0, 0.0, 0.0, 0.0],     # Y' → Γ
            [0.0, 0.0, 0.0, 0.0, 0.0, -0.5],     # Γ → Z'
            [0.0, 0.0, -0.5, -0.5, 0.0, -0.5],   # Z' → U'
            [-0.5, 0.0, -0.5, -0.5, -0.5, -0.5], # U' → R'
            [-0.5, -0.5, -0.5, 0.0, -0.5, -0.5], # R' → T'
            [0.0, -0.5, -0.5, 0.0, 0.0, -0.5],   # T' → Z'
            [-0.5, 0.0, 0.0, -0.5, 0.0, -0.5],   # X' → U'
            [0.0, -0.5, 0.0, 0.0, -0.5, -0.5],   # Y' → T'
            [-0.5, -0.5, 0.0, -0.5, -0.5, -0.5]  # S' → R'
        ],
        "labels": ["G", "X", "S", "Y", "G", "Z", "U", "R", "T", "Z", "|", "Y", "T", "|", "X", "U", "|", "S", "R", "|", "G", "X'", "S'", "Y'", "G", "Z'", "U'", "R'", "T'", "Z'", "|", "Y'", "T'", "|", "X'", "U'", "|", "S'", "R'"]
    },
    "tI1_noinv": {
        # Tetragonal body-centered without inversion (I-4)
        "segments": [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.5],              # Γ → X
            [0.0, 0.0, 0.5, -0.5, 0.5, 0.5],             # X → M
            [-0.5, 0.5, 0.5, 0.0, 0.0, 0.0],             # M → Γ
            [0.0, 0.0, 0.0, 0.473721, 0.473721, -0.473721], # Γ → Z
            [-0.473721, 0.526279, 0.473721, -0.5, 0.5, 0.5], # Z₀ → M
            [0.0, 0.0, 0.5, 0.25, 0.25, 0.25],           # X → P
            [0.25, 0.25, 0.25, 0.0, 0.5, 0.0],           # P → N
            [0.0, 0.5, 0.0, 0.0, 0.0, 0.0],              # N → Γ
            [0.0, 0.0, 0.0, 0.0, 0.0, -0.5],             # Γ → X'
            [0.0, 0.0, -0.5, 0.5, -0.5, -0.5],           # X' → M'
            [0.5, -0.5, -0.5, 0.0, 0.0, 0.0],            # M' → Γ
            [0.0, 0.0, 0.0, -0.473721, -0.473721, 0.473721], # Γ → Z'
            [0.473721, -0.526279, -0.473721, 0.5, -0.5, -0.5], # Z₀' → M'
            [0.0, 0.0, -0.5, -0.25, -0.25, -0.25],       # X' → P'
            [-0.25, -0.25, -0.25, 0.0, -0.5, 0.0],       # P' → N'
            [0.0, -0.5, 0.0, 0.0, 0.0, 0.0]              # N' → Γ
        ],
        "labels": ["G", "X", "M", "G", "Z", "Z_0", "M", "|", "X", "P", "N", "G", "|", "G", "X'", "M'", "G", "Z'", "Z_0'", "M'", "|", "X'", "P'", "N'", "G"]
    },
    "tI2_noinv": {
        # Tetragonal body-centered without inversion (I4mm)
        "segments": [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.5],              # Γ → X
            [0.0, 0.0, 0.5, 0.25, 0.25, 0.25],           # X → P
            [0.25, 0.25, 0.25, 0.0, 0.5, 0.0],           # P → N
            [0.0, 0.5, 0.0, 0.0, 0.0, 0.0],              # N → Γ
            [0.0, 0.0, 0.0, 0.5, 0.5, -0.5],             # Γ → M
            [0.5, 0.5, -0.5, 0.346320, 0.653680, -0.346320], # M → S
            [-0.346320, 0.346320, 0.346320, 0.0, 0.0, 0.0], # S₀ → Γ
            [0.0, 0.0, 0.5, -0.192640, 0.192640, 0.5],   # X → R
            [0.5, 0.5, -0.192640, 0.5, 0.5, -0.5],       # G → M
            [0.0, 0.0, 0.0, 0.0, 0.0, -0.5],             # Γ → X'
            [0.0, 0.0, -0.5, -0.25, -0.25, -0.25],       # X' → P'
            [-0.25, -0.25, -0.25, 0.0, -0.5, 0.0],       # P' → N'
            [0.0, -0.5, 0.0, 0.0, 0.0, 0.0],             # N' → Γ
            [0.0, 0.0, 0.0, -0.5, -0.5, 0.5],            # Γ → M'
            [-0.5, -0.5, 0.5, -0.346320, -0.653680, 0.346320], # M' → S'
            [0.346320, -0.346320, -0.346320, 0.0, 0.0, 0.0], # S₀' → Γ
            [0.0, 0.0, -0.5, 0.192640, -0.192640, -0.5], # X' → R'
            [-0.5, -0.5, 0.192640, -0.5, -0.5, 0.5]      # G' → M'
        ],
        "labels": ["G", "X", "P", "N", "G", "M", "S", "S_0", "G", "|", "X", "R", "|", "G", "M", "|", "G", "X'", "P'", "N'", "G", "M'", "S'", "S_0'", "G", "|", "X'", "R'", "|", "G'", "M'"]
    },
    "tP1_noinv": {
        # Tetragonal primitive without inversion (P-4m2)
        "segments": [
            [0.0, 0.0, 0.0, 0.0, 0.5, 0.0],      # Γ → X
            [0.0, 0.5, 0.0, 0.5, 0.5, 0.0],      # X → M
            [0.5, 0.5, 0.0, 0.0, 0.0, 0.0],      # M → Γ
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.5],      # Γ → Z
            [0.0, 0.0, 0.5, 0.0, 0.5, 0.5],      # Z → R
            [0.0, 0.5, 0.5, 0.5, 0.5, 0.5],      # R → A
            [0.5, 0.5, 0.5, 0.0, 0.0, 0.5],      # A → Z
            [0.0, 0.5, 0.0, 0.0, 0.5, 0.5],      # X → R
            [0.5, 0.5, 0.0, 0.5, 0.5, 0.5],      # M → A
            [0.0, 0.0, 0.0, 0.0, -0.5, 0.0],     # Γ → X'
            [0.0, -0.5, 0.0, -0.5, -0.5, 0.0],   # X' → M'
            [-0.5, -0.5, 0.0, 0.0, 0.0, 0.0],    # M' → Γ
            [0.0, 0.0, 0.0, 0.0, 0.0, -0.5],     # Γ → Z'
            [0.0, 0.0, -0.5, 0.0, -0.5, -0.5],   # Z' → R'
            [0.0, -0.5, -0.5, -0.5, -0.5, -0.5], # R' → A'
            [-0.5, -0.5, -0.5, 0.0, 0.0, -0.5],  # A' → Z'
            [0.0, -0.5, 0.0, 0.0, -0.5, -0.5],   # X' → R'
            [-0.5, -0.5, 0.0, -0.5, -0.5, -0.5]  # M' → A'
        ],
        "labels": ["G", "X", "M", "G", "Z", "R", "A", "Z", "|", "X", "R", "|", "M", "A", "|", "G", "X'", "M'", "G", "Z'", "R'", "A'", "Z'", "|", "X'", "R'", "|", "M'", "A'"]
    }
}


def extract_lattice_parameters_from_output(out_file: str) -> Optional[Dict[str, float]]:
    """Extract lattice parameters from CRYSTAL output file.
    
    Args:
        out_file: Path to CRYSTAL output file
        
    Returns:
        Dictionary with keys: a, b, c, alpha, beta, gamma
        or None if extraction fails
    """
    if not out_file or not Path(out_file).exists():
        return None
        
    try:
        with open(out_file, 'r') as f:
            content = f.read()
            
        # Look for lattice parameters section
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
            
    except Exception:
        pass
        
    return None


def has_inversion_symmetry(space_group: int) -> bool:
    """Quick check if space group has inversion symmetry based on space group number."""
    return space_group in CENTROSYMMETRIC_SPACE_GROUPS


def detect_inversion_from_crystal_output(output_file: str) -> Tuple[bool, str]:
    """Parse CRYSTAL output to determine inversion symmetry.
    
    Returns:
        (has_inversion, detection_method)
    """
    try:
        with open(output_file, 'r') as f:
            content = f.read()
        
        # Method 1: Explicit centrosymmetric statement
        if 'SPACE GROUP (CENTROSYMMETRIC)' in content:
            return True, 'explicit_centrosymmetric'
        
        # Method 2: Check symmetry operators for inversion
        symmop_section = re.search(
            r'SYMMOPS.*?V\s+INV.*?\n(.*?)(?=\n\n|\Z)', 
            content, re.DOTALL
        )
        if symmop_section:
            operators = symmop_section.group(1).strip().split('\n')
            for op in operators:
                if '-1.00  0.00  0.00  0.00 -1.00  0.00' in op and '0.00 -1.00' in op:
                    return True, 'inversion_operator'
        
        # Method 3: Space group number lookup (if explicit statement not found)
        sg_match = re.search(r'SPACE GROUP.*?N[o.]?\s*(\d+)', content)
        if sg_match:
            sg_num = int(sg_match.group(1))
            if sg_num in CENTROSYMMETRIC_SPACE_GROUPS:
                return True, 'space_group_number'
            else:
                return False, 'space_group_number'
        
        return False, 'unknown'
        
    except Exception:
        return False, 'file_error'


def get_seekpath_full_kpath(space_group: int, lattice_type: str, out_file: Optional[str] = None) -> Tuple[List[List[float]], Dict[str, Any]]:
    """Get comprehensive SeeK-path k-paths with extended Bravais lattice notation.
    
    Based on SeeK-path (https://seekpath.materialscloud.io/)
    These paths include extended Bravais lattice symbols and primed points for 
    non-centrosymmetric groups.
    
    Args:
        space_group: Space group number
        lattice_type: Lattice type (P, C, F, I, R, etc.)
        out_file: Optional CRYSTAL output file to extract cell parameters
        
    Returns:
        Tuple of (segments, kpath_info) where:
            segments: List of k-path segments as fractional coordinates
            kpath_info: Dict with inversion symmetry and source information
    """
    
    # Try to extract lattice parameters if output file provided
    lattice_params = None
    has_inversion = has_inversion_symmetry(space_group)  # Default from space group
    
    if out_file:
        lattice_params = extract_lattice_parameters_from_output(out_file)
        
        # Detect inversion from output file
        detected_inv, method = detect_inversion_from_crystal_output(out_file)
        if method != 'unknown' and method != 'file_error':
            has_inversion = detected_inv
            print(f"  Detected {'centrosymmetric' if has_inversion else 'non-centrosymmetric'} "
                  f"structure via {method}")
    
    # Get extended Bravais symbol with cell parameters if available
    if lattice_params:
        ext_bravais = get_extended_bravais(
            space_group, lattice_type,
            lattice_params['a'], lattice_params['b'], lattice_params['c'],
            lattice_params['alpha'], lattice_params['beta'], lattice_params['gamma']
        )
    else:
        ext_bravais = get_extended_bravais(space_group, lattice_type)
    
    # Select appropriate k-path based on inversion
    if has_inversion:
        lookup_key = ext_bravais  # Use base version for centrosymmetric
    else:
        lookup_key = f"{ext_bravais}_noinv"  # Use non-inversion version
        if lookup_key not in seekpath_data:
            print(f"  WARNING: Non-centrosymmetric path not available for {ext_bravais}")
            print("  Using centrosymmetric path (may miss important features)")
            lookup_key = ext_bravais
    
    # Prepare k-path info dictionary
    kpath_info = {
        "has_inversion": has_inversion,
        "extended_bravais": ext_bravais,
        "lookup_key": lookup_key
    }
    
    # Get path data if available
    if lookup_key in seekpath_data:
        return seekpath_data[lookup_key]["segments"], kpath_info
    else:
        # Fallback to literature path first
        print(f"\nSeeK-path data not available for {lookup_key}")
        
        # Try literature path vectors
        lit_segments = get_literature_kpath_vectors(space_group, lattice_type)
        if lit_segments:
            print("Using literature k-path (Setyawan & Curtarolo 2010) instead")
            kpath_info["source"] = "literature"
            return lit_segments, kpath_info
        
        # If no literature path, fall back to standard path
        print("Using standard path instead")
        kpath_info["source"] = "default"
        return get_kpoint_coordinates_from_labels(
            get_band_path_from_symmetry(space_group, lattice_type),
            space_group, lattice_type
        ), kpath_info

def unicode_to_ascii_kpoint(label: str) -> str:
    """Convert Unicode k-point labels to ASCII equivalents for CRYSTAL compatibility."""
    # Common conversions
    conversions = {
        "Γ": "G",
        "Σ": "SIGMA",
        "Λ": "LAMBDA",
        "Δ": "DELTA",
        "Ξ": "XI",
        "Π": "PI",
        "∑": "SIGMA",
        "λ": "LAMBDA",
        "δ": "DELTA",
        "ξ": "XI",
        "π": "PI",
        # Add primed versions
        "Γ'": "G'",
        "Σ'": "SIGMA'",
        "Λ'": "LAMBDA'",
        "Δ'": "DELTA'",
        # Subscripts become regular numbers
        "₀": "0",
        "₁": "1",
        "₂": "2",
        "₃": "3",
        "₄": "4",
        "₅": "5",
        "₆": "6",
        "₇": "7",
        "₈": "8",
        "₉": "9",
    }
    
    result = label
    for unicode_char, ascii_char in conversions.items():
        result = result.replace(unicode_char, ascii_char)
    
    return result

def get_seekpath_labels(space_group: int, lattice_type: str, out_file: Optional[str] = None) -> List[str]:
    """Get the k-point labels for SeeK-path.
    
    Returns the high-symmetry point labels for the SeeK-path including
    discontinuity markers (|).
    
    Args:
        space_group: Space group number
        lattice_type: Lattice type (P, C, F, I, R, etc.)
        out_file: Optional CRYSTAL output file to extract cell parameters
        
    Returns:
        List of k-point labels
    """
    # Try to extract lattice parameters if output file provided
    lattice_params = None
    has_inversion = has_inversion_symmetry(space_group)  # Default from space group
    
    if out_file:
        lattice_params = extract_lattice_parameters_from_output(out_file)
        
        # Detect inversion from output file
        detected_inv, method = detect_inversion_from_crystal_output(out_file)
        if method != 'unknown' and method != 'file_error':
            has_inversion = detected_inv
    
    # Get extended Bravais symbol with cell parameters if available
    if lattice_params:
        ext_bravais = get_extended_bravais(
            space_group, lattice_type,
            lattice_params['a'], lattice_params['b'], lattice_params['c'],
            lattice_params['alpha'], lattice_params['beta'], lattice_params['gamma']
        )
    else:
        ext_bravais = get_extended_bravais(space_group, lattice_type)
    
    # Select appropriate k-path based on inversion
    if has_inversion:
        lookup_key = ext_bravais  # Use base version for centrosymmetric
    else:
        lookup_key = f"{ext_bravais}_noinv"  # Use non-inversion version
        if lookup_key not in seekpath_data:
            lookup_key = ext_bravais  # Fall back to centrosymmetric
    
    # Get path labels if available
    if lookup_key in seekpath_data and "labels" in seekpath_data[lookup_key]:
        return seekpath_data[lookup_key]["labels"]
    else:
        # Fall back to literature path labels first
        return get_literature_path_labels(space_group, lattice_type)


# Band structure templates for different crystal systems
# These are the .band template files referenced in create_band_d3.py
BAND_TEMPLATES = {
    "cubic": {
        "name": "cubic",
        "description": "Simple cubic lattice",
        "path": ["M", "G", "R", "X", "G"],
        "shrink": 0,
        "npoints": 500,
        "bands": [1, 8],
        "plot": True,
        "print_eig": False
    },
    "cubic2": {
        "name": "cubic2", 
        "description": "Face-centered cubic",
        "path": ["X", "G", "L", "W", "G"],
        "shrink": 0,
        "npoints": 500,
        "bands": [1, 8],
        "plot": True,
        "print_eig": False
    },
    "cubic3": {
        "name": "cubic3",
        "description": "Body-centered cubic",
        "path": ["H", "G", "P", "N", "G"],
        "shrink": 0,
        "npoints": 500,
        "bands": [1, 8],
        "plot": True,
        "print_eig": False
    },
    "hexagonal": {
        "name": "hexagonal",
        "description": "Hexagonal lattice",
        "path": ["M", "G", "K", "A", "G", "L", "H", "G"],
        "shrink": 0,
        "npoints": 1000,
        "bands": [1, 8],
        "plot": True,
        "print_eig": False
    },
    "hexagonal2": {
        "name": "hexagonal2",
        "description": "Rhombohedral lattice",
        "path": ["T", "G", "F", "L", "G"],
        "shrink": 0,
        "npoints": 500,
        "bands": [1, 8],
        "plot": True,
        "print_eig": False
    },
    "tetragonal": {
        "name": "tetragonal",
        "description": "Simple tetragonal",
        "path": ["M", "G", "R", "A", "G", "X", "Z", "G"],
        "shrink": 0,
        "npoints": 1000,
        "bands": [1, 8],
        "plot": True,
        "print_eig": False
    },
    "tetragonal2": {
        "name": "tetragonal2",
        "description": "Body-centered tetragonal",
        "path": ["M", "G", "P", "X", "G"],
        "shrink": 0,
        "npoints": 500,
        "bands": [1, 8],
        "plot": True,
        "print_eig": False
    },
    "orthorhombic": {
        "name": "orthorhombic",
        "description": "Simple orthorhombic",
        "path": ["S", "G", "T", "U", "G", "R", "X", "G", "Y", "Z", "G"],
        "shrink": 0,
        "npoints": 1000,
        "bands": [1, 8],
        "plot": True,
        "print_eig": False
    },
    "orthorhombic2": {
        "name": "orthorhombic2",
        "description": "Face-centered orthorhombic",
        "path": ["Z", "G", "Y", "T", "G"],
        "shrink": 0,
        "npoints": 400,
        "bands": [1, 8],
        "plot": True,
        "print_eig": False
    },
    "orthorhombic3": {
        "name": "orthorhombic3",
        "description": "Base-centered orthorhombic (a)",
        "path": ["S", "G", "T", "R", "G", "Y", "Z", "G"],
        "shrink": 0,
        "npoints": 600,
        "bands": [1, 8],
        "plot": True,
        "print_eig": False
    },
    "orthorhombic4": {
        "name": "orthorhombic4",
        "description": "Base-centered orthorhombic (c)",
        "path": ["S", "G", "T", "R", "G", "X", "W", "G"],
        "shrink": 0,
        "npoints": 600,
        "bands": [1, 8],
        "plot": True,
        "print_eig": False
    },
    "monoclinic": {
        "name": "monoclinic",
        "description": "Simple monoclinic",
        "path": ["A", "G", "B", "C", "G", "D", "E", "G", "Y", "Z", "G"],
        "shrink": 0,
        "npoints": 1000,
        "bands": [1, 8],
        "plot": True,
        "print_eig": False
    },
    "monoclinic2": {
        "name": "monoclinic2",
        "description": "Base-centered monoclinic",
        "path": ["A", "G", "Y", "M", "G"],
        "shrink": 0,
        "npoints": 400,
        "bands": [1, 8],
        "plot": True,
        "print_eig": False
    },
    "triclinic": {
        "name": "triclinic",
        "description": "Triclinic lattice",
        "path": ["V", "Y", "G", "Z", "T", "R", "G", "X", "U", "G"],
        "shrink": 0,
        "npoints": 1000,
        "bands": [1, 8],
        "plot": True,
        "print_eig": False
    }
}