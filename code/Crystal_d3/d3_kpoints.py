"""
K-point Data and Path Generation for CRYSTAL D3 Calculations

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
        shrink_match = re.search(r'^SHRINK\s*\n\s*(\d+)', d12_content, re.MULTILINE)
        if shrink_match:
            original_shrink = int(shrink_match.group(1))
            # Round to nearest even number for cleaner k-paths
            if original_shrink % 2 == 1:
                shrink = original_shrink + 1
                print(f"    Using shrink factor {shrink} (rounded from {original_shrink} to even number)")
            else:
                shrink = original_shrink
                print(f"    Using shrink factor {shrink} from original D12 file")
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
    
    # Ensure shrink is even
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


def get_extended_bravais(sg: int, lat: str) -> str:
    """Determine extended Bravais lattice symbol from space group and lattice type.
    
    Based on SeeK-path methodology for comprehensive k-path determination.
    Returns symbols like aP1, aP2, cF1, etc.
    """
    
    # Triclinic
    if sg == 1:
        return "aP2"  # Without inversion
    elif sg == 2:
        # Need to distinguish aP2 vs aP3 based on cell parameters
        # Default to aP2 for now
        return "aP2"
        
    # Monoclinic
    elif 3 <= sg <= 15:
        if lat == "P":
            # Need to check unique axis and angles
            # Default to mP1 for now
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
            # Need to distinguish oS1 vs oS2 based on centering
            return "oS1"
        elif lat == "F":
            # Need to distinguish oF1 vs oF2 vs oF3
            return "oF1"
        elif lat == "I":
            # Need to distinguish oI1 vs oI2 vs oI3
            return "oI1"
        else:
            return "oP1"
            
    # Tetragonal
    elif 75 <= sg <= 142:
        if lat == "P":
            return "tP1"
        elif lat == "I":
            # Need to distinguish tI1 vs tI2
            return "tI1"
        else:
            return "tP1"
            
    # Trigonal/Rhombohedral
    elif 143 <= sg <= 167:
        if lat == "P":
            return "hP1"
        elif lat == "R":
            # Need to distinguish hR1 vs hR2
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
            # Need to distinguish cF1 vs cF2
            return "cF1"
        elif lat == "I":
            # Need to distinguish cI1 vs cI2
            return "cI1"
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
    # Add more SeeK-path data as needed...
}


def get_seekpath_full_kpath(space_group: int, lattice_type: str, out_file: Optional[str] = None) -> List[List[float]]:
    """Get comprehensive SeeK-path k-paths with extended Bravais lattice notation.
    
    Based on SeeK-path (https://seekpath.materialscloud.io/)
    These paths include extended Bravais lattice symbols and primed points for 
    non-centrosymmetric groups.
    """
    
    # Get extended Bravais symbol
    ext_bravais = get_extended_bravais(space_group, lattice_type)
    
    # Get path data if available
    if ext_bravais in seekpath_data:
        return seekpath_data[ext_bravais]["segments"]
    else:
        # Fallback to standard path
        print(f"\nSeeK-path data not available for {ext_bravais}")
        print("Using standard path instead")
        return get_kpoint_coordinates_from_labels(
            get_band_path_from_symmetry(space_group, lattice_type),
            space_group, lattice_type
        )

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

def get_seekpath_labels(space_group: int, lattice_type: str) -> List[str]:
    """Get the k-point labels for SeeK-path.
    
    Returns the high-symmetry point labels for the SeeK-path including
    discontinuity markers (|).
    """
    # Get extended Bravais symbol
    ext_bravais = get_extended_bravais(space_group, lattice_type)
    
    # Get path labels if available
    if ext_bravais in seekpath_data and "labels" in seekpath_data[ext_bravais]:
        return seekpath_data[ext_bravais]["labels"]
    else:
        # Return generic SeeK-path indicator
        return ["SeeK-path"]


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