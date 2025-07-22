"""
D3 Configuration Module for CRYSTAL Properties Calculations

This module contains configuration functions and templates for various D3 calculation types:
- BAND: Electronic band structure
- DOSS: Density of states  
- BOLTZTRA: Transport properties
- ECH3/ECHG: Charge density
- POT3/POTC: Electrostatic potential
"""

import os
import re
from typing import Dict, List, Optional, Tuple, Any, Union
import sys
from pathlib import Path
# Add Crystal_d12 to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "Crystal_d12"))
from d12_interactive import yes_no_prompt
from d12_constants import generate_k_points


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
        "M": [0.5, 0.5, -0.5],   # [1/2, 1/2, -1/2]
        "P": [0.5, 0.5, 0.5],    # [1/2, 1/2, 1/2]
        "X": [0.0, 0.0, 0.5],    # [0, 0, 1/2]
        # Common additions
        "N": [0.5, 0.0, 0.0],    # Common point
        "S": [0.0, 0.5, 0.5],    # Common point
        "Z": [0.0, 0.0, 0.5],    # Same as X
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
        "Y": [0.5, 0.0, 0.5],    # [1/2, 0, 1/2]
        "T": [1.0, 0.5, 0.5],    # [1, 1/2, 1/2]
        # Common additions
        "L": [0.5, 0.5, 0.5],    # Common point
        "X": [0.5, 0.5, 0.0],    # Same as Z
        "W": [0.25, 0.75, 0.25], # Common point
        "U": [0.0, 0.5, 0.0],    # Common point
        "K": [0.0, 0.0, 0.5],    # Common point
    },
    "orthorhombic_ab": {
        "G": [0.0, 0.0, 0.0],    # Gamma
        "S": [0.0, 0.5, 0.0],    # [0, 1/2, 0]
        "T": [0.5, 0.5, 0.5],    # [1/2, 1/2, 1/2]
        "R": [0.0, 0.5, 0.5],    # [0, 1/2, 1/2]
        "Y": [0.5, 0.5, 0.0],    # [1/2, 1/2, 0]
        "Z": [0.0, 0.0, 0.5],    # [0, 0, 1/2]
        # Common additions
        "X": [0.5, 0.0, 0.0],    # Common point
        "U": [0.5, 0.0, 0.5],    # Common point
    },
    "orthorhombic_bc": {
        "G": [0.0, 0.0, 0.0],    # Gamma
        "S": [0.5, 0.0, 0.0],    # [1/2, 0, 0]
        "T": [0.0, 0.0, 0.5],    # [0, 0, 1/2]
        "R": [0.0, 0.5, 0.0],    # [0, 1/2, 0]
        "X": [0.5, -0.5, 0.5],   # [1/2, -1/2, 1/2]
        "W": [0.25, 0.25, 0.25], # [1/4, 1/4, 1/4]
        # Common additions
        "Y": [0.0, 0.5, 0.0],    # Same as R
        "Z": [0.0, 0.0, 0.5],    # Same as T
        "U": [0.5, 0.0, 0.5],    # Common point
    },
    
    # Monoclinic systems (from Table 14.1)
    "monoclinic_simple": {
        "G": [0.0, 0.0, 0.0],    # Gamma
        "A": [0.5, -0.5, 0.0],   # [1/2, -1/2, 0]
        "B": [0.5, 0.0, 0.0],    # [1/2, 0, 0]
        "C": [0.0, 0.5, 0.5],    # [0, 1/2, 1/2]
        "D": [0.5, 0.0, 0.5],    # [1/2, 0, 1/2]
        "E": [0.5, -0.5, 0.5],   # [1/2, -1/2, 1/2]
        "Y": [0.0, 0.5, 0.0],    # [0, 1/2, 0]
        "Z": [0.0, 0.0, 0.5],    # [0, 0, 1/2]
        # Common additions
        "M1": [0.5, 0.5, 0.5],   # Not in table but commonly used
        "H": [0.0, 0.5, 0.0],    # Same as Y
        "X": [0.0, 0.5, 0.5],    # Same as C
    },
    "monoclinic_ac": {
        "G": [0.0, 0.0, 0.0],    # Gamma
        "A": [0.5, 0.0, 0.0],    # [1/2, 0, 0]
        "Y": [0.0, 0.5, 0.5],    # [0, 1/2, 1/2]
        "M": [0.5, 0.5, 0.5],    # [1/2, 1/2, 1/2]
        # Common additions from P monoclinic mapping
        "C": [0.0, 0.0, 0.5],    # Common point
        "D": [0.5, 0.0, 0.5],    # Common point
        "B": [0.5, 0.5, 0.0],    # Common point
        "E": [0.0, 0.5, 0.5],    # Same as Y
        "Z": [0.5, 0.5, 0.5],    # Same as M
    },
    
    # Triclinic system
    "triclinic": {
        "G": [0.0, 0.0, 0.0],
        "X": [0.5, 0.0, 0.0],
        "Y": [0.0, 0.5, 0.0],
        "Z": [0.0, 0.0, 0.5],
        "R": [0.5, 0.5, 0.5],
        # Additional points from your templates
        "V": [0.5, 0.5, 0.0],    # Same as M
        "T": [0.0, 0.5, 0.5],    # Same as N
        "U": [0.5, 0.0, 0.5],    # Same as P
        # Alternatives
        "M": [0.5, 0.5, 0.0],
        "N": [0.0, 0.5, 0.5],
        "P": [0.5, 0.0, 0.5],
        "L": [0.5, 0.5, 0.5],  # Alternative label for R
    },
    
    # Rhombohedral (R Trigonal) system (from Table 14.1)
    "rhombohedral": {
        "G": [0.0, 0.0, 0.0],    # Gamma
        "T": [0.5, 0.5, -0.5],   # [1/2, 1/2, -1/2]
        "F": [0.0, 0.5, 0.5],    # [0, 1/2, 1/2]
        "L": [0.0, 0.0, 0.5],    # [0, 0, 1/2]
    }
}


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
    lit_kpoints = {
        # Cubic FCC additions
        "K": [3/8, 3/8, 3/4],
        "U": [5/8, 1/4, 5/8],
        
        # Tetragonal BC additions  
        "N": [0.0, 0.5, 0.0],
        "S": [0.5, 0.5, 0.0],
        "S0": [0.5 + 0.25, 0.25, 0.0],  # ζ, ζ, 0 point
        "Z": [0.5, 0.5, 0.5],
        
        # Orthorhombic additions
        "A": [0.5, 0.5 + 0.25, 0.0],  # ζ point for oP
        "A1": [0.5, 0.5 - 0.25, 0.0], 
        "X1": [0.25, 0.0, 0.0],
        "Y1": [0.0, 0.25, 0.0],
        "L": [0.5, 0.5, 0.5],
        "L1": [0.5, 0.0, 0.0],
        "L2": [0.0, 0.5, 0.0],
        
        # Monoclinic additions
        "H": [0.0, 0.5, -0.5],
        "H1": [0.0, 0.5, 0.5],
        "M1": [0.5, 0.5, 0.5],
        "X": [0.0, 0.5, 0.0],
        
        # Rhombohedral additions
        "B": [0.5, 0.0, 0.0],
        "B1": [0.0, 0.5, 0.5],
        "P": [0.5 + 0.25, 0.25, 0.25],  # 1-ζ, ζ, ζ
        "P1": [0.25, 0.25, 0.25],       # ζ, ζ, ζ
        "Q": [0.5, 0.5, 0.0],
        "X": [0.5, 0.0, -0.5],
        "Z": [0.5, 0.5, 0.0],
    }
    
    # Merge with existing CRYSTAL points
    crystal_system = get_crystal_system_from_space_group(space_group, lattice_type)
    base_kpoints = KPOINT_COORDINATES.get(crystal_system, {}).copy()
    
    # Add literature points
    all_kpoints = {**base_kpoints, **lit_kpoints}
    
    # Define comprehensive literature paths
    literature_paths = {
        "cubic_fc": ["G", "X", "W", "K", "G", "L", "U", "W", "L", "K", "U", "X"],
        "cubic_bc": ["G", "H", "N", "G", "P", "H", "P", "N"],
        "tetragonal_bc": ["G", "X", "M", "G", "Z", "P", "N", "Z", "M", "X", "P"],
        "orthorhombic_simple": ["G", "X", "S", "Y", "G", "Z", "U", "R", "T", "Z", "Y", "T", "U", "X", "S", "R"],
        "orthorhombic_fc": ["G", "Y", "T", "Z", "G", "X", "A1", "Y", "T", "X1", "X", "A", "Z", "L", "G"],
        "orthorhombic_bc": ["G", "X", "L", "T", "W", "R", "X1", "Z", "G", "Y", "S", "W", "L1", "Y", "Y1", "Z"],
        "monoclinic_simple": ["G", "Y", "H", "C", "E", "M1", "A", "X", "G", "Z", "D", "B", "A", "D", "Y", "H1"],
        "rhombohedral": ["G", "L", "B1", "B", "Z", "G", "X", "Q", "F", "P1", "Z", "L", "P"],
    }
    
    # Get the path for this crystal system
    path_labels = literature_paths.get(crystal_system)
    if not path_labels:
        return []  # No literature path available
    
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


def get_seekpath_full_kpath(space_group: int, lattice_type: str, out_file: Optional[str] = None) -> List[List[float]]:
    """Get comprehensive SeeK-path k-paths with extended Bravais lattice notation.
    
    Based on SeeK-path (https://seekpath.materialscloud.io/)
    These paths include extended Bravais lattice symbols and primed points for 
    non-centrosymmetric groups.
    """
    
    # Extended Bravais lattice determination
    # This maps space group + lattice type to extended Bravais symbol
    def get_extended_bravais(sg: int, lat: str) -> str:
        # Triclinic
        if sg == 1:
            return "aP2"  # Without inversion
        elif sg == 2:
            # Need to distinguish aP2 vs aP3 based on cell parameters
            # Default to aP2 for now
            return "aP2"
            
        # Monoclinic
        elif 3 <= sg <= 15:
            if sg in [5, 8, 9]:  # Without inversion
                if lat == "C":
                    if sg == 5:
                        return "mC2"
                    elif sg == 8:
                        return "mC3"
                    else:  # sg == 9
                        return "mC1"
                else:
                    return "mP1"
            else:  # With inversion
                if lat == "C":
                    if 12 <= sg <= 15:
                        return "mC1"
                    else:
                        return "mC2"
                else:
                    return "mP1"
                    
        # Orthorhombic
        elif 16 <= sg <= 74:
            if sg in [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]:
                # Without inversion
                if lat == "P":
                    return "oP1"
                elif lat in ["C", "A"]:
                    if sg == 36:
                        return "oC1"
                    elif sg == 38:
                        return "oA1"
                    else:
                        return "oC2"
                elif lat == "F":
                    if sg == 42:
                        return "oF1"
                    elif sg == 43:
                        return "oF2"
                    else:
                        return "oF3"
                elif lat == "I":
                    if sg == 44:
                        return "oI1"
                    elif sg == 46:
                        return "oI2"
                    else:
                        return "oI3"
            else:
                # With inversion
                if lat == "P":
                    return "oP1"
                elif lat in ["C", "A"]:
                    if 63 <= sg <= 68:
                        return "oC2"
                    else:
                        return "oC1"
                elif lat == "F":
                    if sg == 69:
                        return "oF1"
                    else:
                        return "oF3"
                elif lat == "I":
                    if sg == 71:
                        return "oI1"
                    elif sg == 74:
                        return "oI3"
                    else:
                        return "oI1"
                        
        # Tetragonal
        elif 75 <= sg <= 142:
            if sg in [75, 76, 77, 78, 79, 80, 81, 82, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122]:
                # Without inversion
                if lat == "P":
                    return "tP1"
                else:  # I
                    if sg == 82:
                        return "tI1"
                    else:
                        return "tI2"
            else:
                # With inversion
                if lat == "P":
                    return "tP1"
                else:  # I
                    if sg in [87, 88]:
                        return "tI1"
                    else:
                        return "tI2"
                        
        # Hexagonal/Trigonal
        elif 143 <= sg <= 194:
            if lat == "R":
                if sg in [146, 148, 155, 160, 161, 166, 167]:
                    # Without inversion
                    if sg == 160:
                        return "hR1"
                    else:
                        return "hR2"
                else:
                    # With inversion
                    if sg == 166:
                        return "hR1"
                    else:
                        return "hR2"
            else:  # P
                if sg in [143, 144, 145, 149, 150, 151, 152, 153, 154, 156, 157, 158, 159, 162, 163, 164, 165, 168, 169, 170, 171, 172, 173, 174, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190]:
                    # Without inversion
                    if sg in [162, 163, 164, 165]:
                        return "hP1"
                    else:
                        return "hP2"
                else:
                    # With inversion
                    if sg in [176, 192, 193, 194]:
                        return "hP2"
                    else:
                        return "hP1"
                        
        # Cubic
        elif 195 <= sg <= 230:
            if sg in [195, 196, 197, 198, 199, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220]:
                # Without inversion
                if lat == "P":
                    if sg in [195, 198, 207, 208, 212, 213]:
                        return "cP1"
                    else:
                        return "cP2"
                elif lat == "F":
                    if sg in [196, 209, 210]:
                        return "cF1"
                    else:
                        return "cF2"
                else:  # I
                    return "cI1"
            else:
                # With inversion
                if lat == "P":
                    if sg in [200, 201, 205]:
                        return "cP1"
                    else:
                        return "cP2"
                elif lat == "F":
                    if sg in [202, 203]:
                        return "cF1"
                    else:
                        return "cF2"
                else:  # I
                    return "cI1"
        
        # Default
        return "cP1"
    
    # Get extended Bravais symbol
    ext_bravais = get_extended_bravais(space_group, lattice_type)
    
    # SeeK-path full path data
    # Format: segments as fractional coordinates [x1, y1, z1, x2, y2, z2]
    # These will be scaled by the user's chosen shrink factor
    seekpath_data = {
        "aP2": {
            "segments": [
                [0.0, 0.0, 0.0, 0.5, 0.0, 0.0],    # Γ → X
                [0.0, 0.5, 0.0, 0.0, 0.0, 0.0],    # Y → Γ
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5],    # Γ → Z
                [0.5, 0.5, 0.5, 0.0, 0.0, 0.0],    # R → Γ
                [0.0, 0.0, 0.0, 0.0, 0.5, 0.5],    # Γ → T
                [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],    # U → Γ
                [0.0, 0.0, 0.0, 0.5, 0.5, 0.0],    # Γ → V
            ]
        },
        "aP3": {
            "segments": [
                [0.0, 0.0, 0.0, 0.5, 0.0, 0.0],     # Γ → X
                [0.0, 0.5, 0.0, 0.0, 0.0, 0.0],     # Y → Γ
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5],     # Γ → Z
                [-0.5, -0.5, 0.5, 0.0, 0.0, 0.0],   # R2 → Γ
                [0.0, 0.0, 0.0, 0.0, -0.5, 0.5],    # Γ → T2
                [-0.5, 0.0, 0.5, 0.0, 0.0, 0.0],    # U2 → Γ
                [0.0, 0.0, 0.0, 0.5, -0.5, 0.0],    # Γ → V2
            ]
        },
        "cF1": {
            "segments": [
                [0.0, 0.0, 0.0, 0.5, 0.0, 0.5],    # Γ → X
                [0.5, 0.0, 0.5, 0.625, 0.25, 0.625],    # X → U
                [0.375, 0.375, 0.75, 0.0, 0.0, 0.0],    # K → Γ
                [0.0, 0.0, 0.0, 0.5, 0.5, 0.5],    # Γ → L
                [0.5, 0.5, 0.5, 0.5, 0.25, 0.75],    # L → W
                [0.5, 0.25, 0.75, 0.5, 0.0, 0.5],    # W → X
                [0.5, 0.0, 0.5, 0.75, 0.25, 0.5],    # X → W2
            ]
        },
        "cF2": {
            "segments": [
                [0.0, 0.0, 0.0, 0.5, 0.0, 0.5],    # Γ → X
                [0.5, 0.0, 0.5, 0.625, 0.25, 0.625],    # X → U
                [0.375, 0.375, 0.75, 0.0, 0.0, 0.0],    # K → Γ
                [0.0, 0.0, 0.0, 0.5, 0.5, 0.5],    # Γ → L
                [0.5, 0.5, 0.5, 0.5, 0.25, 0.75],    # L → W
                [0.5, 0.25, 0.75, 0.5, 0.0, 0.5],    # W → X
            ]
        },
        "cI1": {
            "segments": [
                [0.0, 0.0, 0.0, 0.5, -0.5, 0.5],   # Γ → H
                [0.5, -0.5, 0.5, 0.0, 0.0, 0.5],   # H → N
                [0.0, 0.0, 0.5, 0.0, 0.0, 0.0],    # N → Γ
                [0.0, 0.0, 0.0, 0.25, 0.25, 0.25],    # Γ → P
                [0.25, 0.25, 0.25, 0.5, -0.5, 0.5],   # P → H
                [0.25, 0.25, 0.25, 0.0, 0.0, 0.5],    # P → N
            ]
        },
        "cP1": {
            "segments": [
                [0.0, 0.0, 0.0, 0.0, 0.5, 0.0],    # Γ → X
                [0.0, 0.5, 0.0, 0.5, 0.5, 0.0],    # X → M
                [0.5, 0.5, 0.0, 0.0, 0.0, 0.0],    # M → Γ
                [0.0, 0.0, 0.0, 0.5, 0.5, 0.5],    # Γ → R
                [0.5, 0.5, 0.5, 0.0, 0.5, 0.0],    # R → X
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.0],    # R → M
                [0.5, 0.5, 0.0, 0.5, 0.0, 0.0],    # M → X1
            ]
        },
        "cP2": {
            "segments": [
                [0.0, 0.0, 0.0, 0.0, 0.5, 0.0],    # Γ → X
                [0.0, 0.5, 0.0, 0.5, 0.5, 0.0],    # X → M
                [0.5, 0.5, 0.0, 0.0, 0.0, 0.0],    # M → Γ
                [0.0, 0.0, 0.0, 0.5, 0.5, 0.5],    # Γ → R
                [0.5, 0.5, 0.5, 0.0, 0.5, 0.0],    # R → X
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.0],    # R → M
            ]
        },
        "hP1": {
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
                [1/3, 1/3, 0.0, 1/3, 1/3, -0.5],   # K → H2
            ]
        },
        "hP2": {
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
            ]
        },
        "tP1": {
            "segments": [
                [0.0, 0.0, 0.0, 0.0, 0.5, 0.0],    # Γ → X
                [0.0, 0.5, 0.0, 0.5, 0.5, 0.0],    # X → M
                [0.5, 0.5, 0.0, 0.0, 0.0, 0.0],    # M → Γ
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5],    # Γ → Z
                [0.0, 0.0, 0.5, 0.0, 0.5, 0.5],    # Z → R
                [0.0, 0.5, 0.5, 0.5, 0.5, 0.5],    # R → A
                [0.5, 0.5, 0.5, 0.0, 0.0, 0.5],    # A → Z
                [0.0, 0.5, 0.0, 0.0, 0.5, 0.5],    # X → R
                [0.5, 0.5, 0.0, 0.5, 0.5, 0.5],    # M → A
            ]
        },
        "oP1": {
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
                [0.5, 0.5, 0.0, 0.5, 0.5, 0.5],    # S → R
            ]
        },
    }
    
    # Add more data for other systems based on the user's examples
    # For now, implementing a subset
    
    # Get the path data for this extended Bravais symbol
    path_data = seekpath_data.get(ext_bravais)
    if not path_data:
        # Fallback to basic path
        print(f"Warning: No SeeK-path data for {ext_bravais}, using standard path")
        return get_literature_kpath_vectors(space_group, lattice_type)
    
    # The segments are in fractional coordinates, but CRYSTAL needs them 
    # scaled by the shrink factor. Since we don't know the shrink factor here,
    # we return the fractional coordinates and let the caller handle scaling
    return path_data["segments"]


def get_crystal_system_from_space_group(space_group: int, lattice_type: str) -> str:
    """Determine the crystal system key from space group and lattice type."""
    
    # Triclinic
    if space_group <= 2:
        return "triclinic"
    
    # Monoclinic
    elif 3 <= space_group <= 15:
        if lattice_type in ["C", "A", "B"]:
            return "monoclinic_ac"
        return "monoclinic_simple"
    
    # Orthorhombic
    elif 16 <= space_group <= 74:
        if lattice_type == "F":
            return "orthorhombic_fc"
        elif lattice_type == "I":
            return "orthorhombic_bc"
        elif lattice_type in ["C", "A", "B"]:
            return "orthorhombic_ab"
        return "orthorhombic_simple"
    
    # Tetragonal
    elif 75 <= space_group <= 142:
        if lattice_type == "I":
            return "tetragonal_bc"
        return "tetragonal_simple"
    
    # Trigonal/Hexagonal
    elif 143 <= space_group <= 194:
        if lattice_type == "R":
            return "rhombohedral"
        return "hexagonal"
    
    # Cubic
    elif 195 <= space_group <= 230:
        if lattice_type == "F":
            return "cubic_fc"
        elif lattice_type == "I":
            return "cubic_bc"
        return "cubic_simple"
    
    return "triclinic"  # Default fallback


def get_band_path_from_symmetry(space_group: int, lattice_type: str) -> List[str]:
    """Get the appropriate band path based on space group and lattice type."""
    
    # Triclinic
    if space_group <= 2:
        return BAND_PATHS["triclinic"]
    
    # Monoclinic
    elif 3 <= space_group <= 15:
        if lattice_type == "P":
            return BAND_PATHS["monoclinic_simple"]
        elif lattice_type in ["C", "A", "B"]:
            return BAND_PATHS["monoclinic_ac"]
    
    # Orthorhombic
    elif 16 <= space_group <= 74:
        if lattice_type == "P":
            return BAND_PATHS["orthorhombic_simple"]
        elif lattice_type in ["C", "A"]:
            return BAND_PATHS["orthorhombic_ab"]
        elif lattice_type == "I":
            return BAND_PATHS["orthorhombic_bc"]
        elif lattice_type == "F":
            return BAND_PATHS["orthorhombic_fc"]
    
    # Tetragonal
    elif 75 <= space_group <= 142:
        if lattice_type == "P":
            return BAND_PATHS["tetragonal_simple"]
        elif lattice_type == "I":
            return BAND_PATHS["tetragonal_bc"]
    
    # Trigonal/Hexagonal
    elif 143 <= space_group <= 167:
        if lattice_type == "P":
            return BAND_PATHS["hexagonal"]
        elif lattice_type == "R":
            return BAND_PATHS["rhombohedral"]
    
    # Hexagonal
    elif 168 <= space_group <= 194:
        return BAND_PATHS["hexagonal"]
    
    # Cubic
    elif space_group >= 195:
        if lattice_type == "P":
            return BAND_PATHS["cubic_simple"]
        elif lattice_type == "F":
            return BAND_PATHS["cubic_fc"]
        elif lattice_type == "I":
            return BAND_PATHS["cubic_bc"]
    
    # Default fallback
    return ["G", "X", "M", "G"]


def configure_band_calculation(out_file: Optional[str] = None) -> Dict[str, Any]:
    """Configure BAND calculation settings interactively."""
    print("\n=== BAND STRUCTURE CONFIGURATION ===")
    
    band_config = {}
    
    # Don't ask for title in shared mode - it will be set per material
    # Title will be automatically set in CRYSTALOptToD3.py to include material name
    
    # Path definition method
    print("\nBand path definition:")
    print("1: Automatic - Use standard path based on crystal symmetry from output file")
    print("2: Template selection - Choose from common band paths")
    print("3: Custom labels - Specify path using labels (G, X, M, etc.)")
    print("4: Fractional coordinates - Specify path using k-point vectors")
    
    path_method = input("Select method (1-4) [1]: ").strip() or "1"
    
    if path_method == "1":
        # True automatic path from output file
        if out_file:
            # Extract space group from output file
            import re
            space_group = 1
            lattice_type = 'P'
            
            try:
                with open(out_file, 'r') as f:
                    content = f.read()
                    
                # Find space group number
                sg_match = re.search(r'SPACE GROUP.*?NUMBER:\s*(\d+)', content)
                if sg_match:
                    space_group = int(sg_match.group(1))
                
                # Find lattice type from space group symbol
                sg_symbol_match = re.search(r'SPACE GROUP.*?:\s+([A-Z]\s*[\-/0-9\s]*[A-Z0-9]*)', content)
                if sg_symbol_match:
                    symbol = sg_symbol_match.group(1).strip()
                    if symbol:
                        lattice_type = symbol[0]
                        
                # Get appropriate band path
                path_labels = get_band_path_from_symmetry(space_group, lattice_type)
                print(f"\n✓ Detected space group {space_group} ({lattice_type} lattice)")
                print(f"  Suggested path: {' → '.join(path_labels)}")
                
            except Exception as e:
                print(f"\nWarning: Could not extract symmetry from output file: {e}")
                print("Defaulting to simple cubic path")
                path_labels = BAND_PATHS["cubic_simple"]
        else:
            print("\nWarning: No output file provided, using default cubic path")
            path_labels = BAND_PATHS["cubic_simple"]
        
        # Now ask for format preference
        print("\nBand path format:")
        print("1: High-symmetry labels (CRYSTAL-compatible subset)")
        print("2: K-point vectors (fractional coordinates)")
        print("3: SeeK-path full paths (comprehensive with extended points)")
        
        format_choice = input("Select format (1-3) [1]: ").strip() or "1"
        
        if format_choice == "1":
            # Use labels - CRYSTAL-compatible subset
            band_config["path_method"] = "labels"
            band_config["shrink"] = 0
            band_config["path"] = path_labels
            band_config["auto_path"] = True  # Mark that this was auto-detected
        elif format_choice == "2":
            # Convert labels to k-point vectors
            band_config["path_method"] = "coordinates"
            
            # Calculate appropriate shrink based on lattice parameters
            # Try to extract lattice parameters from output file
            a, b, c = 10.0, 10.0, 10.0  # Default fallback
            if out_file:
                try:
                    with open(out_file, 'r') as f:
                        content = f.read()
                    # Look for lattice parameters
                    import re
                    # Pattern for PRIMITIVE CELL or LATTICE PARAMETERS
                    param_match = re.search(r'LATTICE PARAMETERS.*?\n\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)', content, re.DOTALL)
                    if not param_match:
                        param_match = re.search(r'PRIMITIVE CELL.*?\n\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)', content, re.DOTALL)
                    if param_match:
                        a = float(param_match.group(1))
                        b = float(param_match.group(2))
                        c = float(param_match.group(3))
                except:
                    pass
            
            # Calculate shrink factors using a*k > 60 rule for band structures
            ka = max(2, int(60.0 / a))
            kb = max(2, int(60.0 / b))
            kc = max(2, int(60.0 / c))
            
            # Use maximum for uniform sampling
            shrink = max(ka, kb, kc)
            band_config["shrink"] = shrink
            band_config["auto_path"] = True  # Mark that this was auto-detected
            
            # Get k-point coordinates for the labels (returns fractional coordinates)
            frac_segments = get_kpoint_coordinates_from_labels(path_labels, space_group, lattice_type)
            
            # Scale fractional coordinates by shrink factor to get integers
            coord_segments = []
            for seg in frac_segments:
                scaled_seg = [int(round(coord * shrink)) for coord in seg]
                coord_segments.append(scaled_seg)
            
            band_config["segments"] = coord_segments
            print(f"\n✓ Converted path to k-point vectors (shrink={shrink})")
        else:
            # SeeK-path full paths with comprehensive k-points
            band_config["path_method"] = "coordinates"
            
            # Calculate appropriate shrink based on lattice parameters
            # Try to extract lattice parameters from output file
            a, b, c = 10.0, 10.0, 10.0  # Default fallback
            if out_file:
                try:
                    with open(out_file, 'r') as f:
                        content = f.read()
                    # Look for lattice parameters
                    import re
                    # Pattern for PRIMITIVE CELL or LATTICE PARAMETERS
                    param_match = re.search(r'LATTICE PARAMETERS.*?\n\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)', content, re.DOTALL)
                    if not param_match:
                        param_match = re.search(r'PRIMITIVE CELL.*?\n\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)', content, re.DOTALL)
                    if param_match:
                        a = float(param_match.group(1))
                        b = float(param_match.group(2))
                        c = float(param_match.group(3))
                except:
                    pass
            
            # Calculate shrink factors using a*k > 60 rule for band structures
            ka = max(2, int(60.0 / a))
            kb = max(2, int(60.0 / b))
            kc = max(2, int(60.0 / c))
            
            # Use maximum for uniform sampling
            shrink = max(ka, kb, kc)
            # Round to nearest available k-value
            ks = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 18, 20, 24, 30, 36, 40, 45, 48, 60, 72, 80, 90, 96]
            shrink = min([k for k in ks if k >= shrink] or [shrink])
            
            print(f"\n✓ Calculated shrink factor: {shrink} (based on lattice parameters)")
            band_config["shrink"] = shrink
            band_config["auto_path"] = True
            band_config["seekpath_full"] = True
            
            # Get SeeK-path full path based on extended Bravais lattice (in fractional coordinates)
            frac_segments = get_seekpath_full_kpath(space_group, lattice_type, out_file)
            if frac_segments:
                # Scale fractional coordinates by shrink factor
                coord_segments = []
                for seg in frac_segments:
                    scaled_seg = [int(round(coord * shrink)) for coord in seg]
                    coord_segments.append(scaled_seg)
                band_config["segments"] = coord_segments
                print(f"✓ Using SeeK-path full k-path with {len(coord_segments)} segments")
            else:
                # Fallback to standard path
                print("\nSeeK-path full path not available, using standard path")
                coord_segments = get_kpoint_coordinates_from_labels(path_labels, space_group, lattice_type)
                band_config["segments"] = coord_segments
    
    elif path_method == "2":
        # Template selection
        band_config["path_method"] = "labels"
        band_config["shrink"] = 0
        
        print("\nSelect crystal system:")
        print("1: Cubic")
        print("2: Hexagonal")
        print("3: Tetragonal")
        print("4: Orthorhombic")
        print("5: Monoclinic")
        print("6: Triclinic")
        
        system = input("Select system (1-6) [1]: ").strip() or "1"
        
        if system == "1":
            print("\nCubic lattice type:")
            print("1: Simple cubic (M-G-R-X-G)")
            print("2: FCC (K-G-L-U-W-L-K-W-X-K)")
            print("3: BCC (P-G-N-P-H-N-G-H)")
            cubic_type = input("Select type (1-3) [1]: ").strip() or "1"
            if cubic_type == "1":
                band_config["path"] = BAND_PATHS["cubic_simple"]
            elif cubic_type == "2":
                band_config["path"] = BAND_PATHS["cubic_fc"]
            else:
                band_config["path"] = BAND_PATHS["cubic_bc"]
        elif system == "2":
            band_config["path"] = BAND_PATHS["hexagonal"]
        elif system == "3":
            band_config["path"] = BAND_PATHS.get("tetragonal_simple", ["G", "X", "M", "G", "Z", "R", "A", "Z"])
        elif system == "4":
            band_config["path"] = BAND_PATHS.get("orthorhombic_simple", ["G", "X", "S", "Y", "G", "Z", "U", "R", "T", "Z"])
        elif system == "5":
            band_config["path"] = BAND_PATHS.get("monoclinic_simple", ["G", "Y", "H", "C", "E", "M1", "A", "X", "G"])
        else:
            band_config["path"] = BAND_PATHS.get("triclinic", ["G", "X", "M", "G", "Y", "L", "G"])
    
    elif path_method == "3":
        band_config["path_method"] = "labels"
        band_config["shrink"] = 0
        
        # Show available templates
        print("\nAvailable band path templates:")
        print("1: Cubic simple (M-G-R-X-G)")
        print("2: Cubic FCC (G-X-W-K-G-L-U-W-L-K)")
        print("3: Cubic BCC (G-H-N-G-P-H)")
        print("4: Tetragonal simple")
        print("5: Orthorhombic simple")
        print("6: Hexagonal")
        print("7: Custom path")
        
        template_choice = input("Select template (1-7) [7]: ").strip() or "7"
        
        if template_choice == "1":
            band_config["path"] = BAND_PATHS["cubic_simple"]
        elif template_choice == "2":
            band_config["path"] = BAND_PATHS["cubic_fc"]
        elif template_choice == "3":
            band_config["path"] = BAND_PATHS["cubic_bc"]
        elif template_choice == "4":
            band_config["path"] = BAND_PATHS["tetragonal_simple"]
        elif template_choice == "5":
            band_config["path"] = BAND_PATHS["orthorhombic_simple"]
        elif template_choice == "6":
            band_config["path"] = BAND_PATHS["hexagonal"]
        else:
            # Custom path
            print("\nEnter band path as space-separated labels (e.g., G X M G):")
            print("Note: Use G for Gamma point")
            path_str = input("Path: ").strip()
            band_config["path"] = path_str.split() if path_str else ["G", "X", "M", "G"]
    
    elif path_method == "4":
        # Fractional coordinates
        band_config["path_method"] = "coordinates"
        shrink = int(input("\nShrink factor for k-points [16]: ") or 16)
        band_config["shrink"] = shrink
        
        n_segments = int(input("Number of path segments: "))
        segments = []
        
        print("\nEnter path segments:")
        print("Format options:")
        print("  1. Fractional coordinates: x1 y1 z1 x2 y2 z2 (e.g., 0.0 0.0 0.0 0.5 0.0 0.0)")
        print("  2. Integer coordinates: x1 y1 z1 x2 y2 z2 (e.g., 0 0 0 8 0 0 for shrink=16)")
        format_type = input("\nCoordinate format (1=fractional, 2=integer) [1]: ").strip() or "1"
        
        if format_type == "1":
            print("\nEnter segments as fractional coordinates (will be scaled by shrink factor):")
            for i in range(n_segments):
                segment_str = input(f"Segment {i+1}: ").strip()
                try:
                    coords = [float(x) for x in segment_str.split()]
                    if len(coords) == 6:
                        # Scale fractional to integer
                        scaled_coords = [int(round(coord * shrink)) for coord in coords]
                        segments.append(scaled_coords)
                    else:
                        print("Warning: Expected 6 numbers, skipping segment")
                except ValueError:
                    print("Warning: Invalid input, skipping segment")
        else:
            print(f"\nEnter segments as integer coordinates (pre-scaled for shrink={shrink}):")
            for i in range(n_segments):
                segment_str = input(f"Segment {i+1}: ").strip()
                try:
                    coords = [int(x) for x in segment_str.split()]
                    if len(coords) == 6:
                        segments.append(coords)
                    else:
                        print("Warning: Expected 6 integers, skipping segment")
                except ValueError:
                    print("Warning: Invalid input, skipping segment")
        
        band_config["segments"] = segments
    
    
    # Number of k-points
    n_points = int(input("\nTotal number of k-points along path [1000]: ") or 1000)
    band_config["n_points"] = n_points
    
    # Band range
    print("\nBand range selection:")
    print("1: All bands")
    print("2: Around Fermi level (~20 valence + ~30 conduction bands)")
    print("3: Custom range")
    
    band_range = input("Select option (1-3) [1]: ").strip() or "1"
    
    if band_range == "1":
        band_config["first_band"] = 1
        # Get actual number of bands from output file if available
        if out_file:
            n_ao = get_band_info_from_output(out_file)[0]
            band_config["last_band"] = n_ao if n_ao > 0 else 9999
        else:
            band_config["last_band"] = 9999  # Large number when no output file
    elif band_range == "2":
        # Auto-detect from output file
        if out_file:
            n_ao, first_band, last_band = get_band_info_from_output(out_file)
            band_config["first_band"] = first_band
            band_config["last_band"] = last_band
            print(f"\n✓ Auto-detected band range: {first_band} to {last_band}")
        else:
            # Default to reasonable range
            band_config["first_band"] = 1
            band_config["last_band"] = 100
    else:
        first = int(input("First band index: "))
        last = int(input("Last band index: "))
        band_config["first_band"] = first
        band_config["last_band"] = last
    
    # Output options - always generate plot for BAND calculations
    band_config["plot"] = True  # Always generate plot file for band structure
    
    print_eigenvalues = yes_no_prompt("\nPrint eigenvalues in output?", "yes")
    band_config["print_eigenvalues"] = print_eigenvalues
    
    return band_config


def configure_doss_calculation(out_file: Optional[str] = None) -> Dict[str, Any]:
    """Configure DOSS (Density of States) calculation settings."""
    print("\n=== DENSITY OF STATES CONFIGURATION ===")
    
    doss_config = {}
    
    # Projection type
    print("\nDOS projection type:")
    print("1: Total DOS only")
    print("2: Total + element-projected DOS (e.g., C, N, O)")
    print("3: Total + element + orbital-projected DOS (e.g., C_s, C_p, N_s, N_p)")
    print("4: Custom atom projections")
    
    proj_type = input("Select projection type (1-4) [3]: ").strip() or "3"
    
    if proj_type == "1":
        doss_config["npro"] = 0
    
    elif proj_type == "2":
        # Element projection only
        doss_config["project_orbital_types"] = False
        doss_config["element_only"] = True
        
    elif proj_type == "3":
        # Element + orbital projection (default)
        doss_config["project_orbital_types"] = True
        doss_config["element_only"] = False
    
    elif proj_type == "4":
        # Custom atom projection
        print("\nAtom selection for projection:")
        print("1: All atoms")
        print("2: By element")
        print("3: By atom indices")
        
        atom_selection = input("Select method (1-3) [1]: ").strip() or "1"
        
        if atom_selection == "1":
            doss_config["project_all_atoms"] = True
            # Number will be determined from structure
        
        elif atom_selection == "2":
            elements = input("Enter element symbols (space-separated, e.g., C O): ").strip().split()
            doss_config["project_elements"] = elements
        
        else:
            indices = input("Enter atom indices (space-separated, e.g., 1 2 5): ").strip()
            doss_config["project_atoms"] = [int(x) for x in indices.split()]
    
    # Energy range
    print("\nEnergy range for DOS calculation:")
    print("1: All bands (default)")
    print("2: Energy window (specify min/max in hartree)")
    print("3: Band range (specify first/last band indices)")
    
    range_type = input("Select range type (1-3) [1]: ").strip() or "1"
    
    if range_type == "2":
        # Energy window - requires BMI/BMA parameters
        print("\nEnergy window (in hartree, must be in a band gap):")
        e_min = float(input("Minimum energy (BMI) [-0.7]: ") or -0.7)
        e_max = float(input("Maximum energy (BMA) [0.4]: ") or 0.4)
        doss_config["energy_window"] = (e_min, e_max)
    elif range_type == "3":
        # Band range
        first = int(input("First band index [1]: ") or 1)
        last = int(input("Last band index [-1 for all]: ") or -1)
        doss_config["band_range"] = (first, last)
    else:
        # All bands (default)
        doss_config["all_bands"] = True
    
    # Number of energy points
    n_points = int(input("\nNumber of energy points [1000]: ") or 1000)
    doss_config["n_points"] = n_points
    
    # Polynomial order for DOS expansion
    print("\nLegendre polynomial expansion order:")
    print("(Higher values = smoother DOS, typical: 14-25)")
    npol = int(input("Polynomial order [18]: ") or 18)
    doss_config["npol"] = npol
    
    # Output format
    print("\nOutput format:")
    print("1: Standard CRYSTAL format (fort.25)")
    print("2: DOSS.DAT format")
    print("3: Both formats")
    
    output_format = input("Select format (1-3) [2]: ").strip() or "2"
    doss_config["output_format"] = int(output_format)
    
    # Smearing
    use_smearing = yes_no_prompt("\nApply Gaussian smearing to DOS?", "yes")
    if use_smearing:
        sigma = float(input("Smearing width (eV) [0.1]: ") or 0.1)
        doss_config["smearing"] = sigma
    
    return doss_config


def configure_transport_calculation() -> Dict[str, Any]:
    """Configure BOLTZTRA transport properties calculation."""
    print("\n=== TRANSPORT PROPERTIES CONFIGURATION ===")
    
    transport_config = {}
    
    # Temperature range
    print("\nTemperature range for transport calculations:")
    t_min = float(input("Minimum temperature (K) [100]: ") or 100)
    t_max = float(input("Maximum temperature (K) [800]: ") or 800)
    t_step = float(input("Temperature step (K) [50]: ") or 50)
    transport_config["temperature_range"] = (t_min, t_max, t_step)
    
    # Chemical potential range
    print("\nChemical potential range:")
    print("(Relative to Fermi level)")
    mu_min = float(input("Minimum μ (eV) [-2.0]: ") or -2.0)
    mu_max = float(input("Maximum μ (eV) [2.0]: ") or 2.0)
    mu_step = float(input("μ step (eV) [0.01]: ") or 0.01)
    transport_config["mu_range"] = (mu_min, mu_max, mu_step)
    
    # Transport distribution function range
    print("\nTransport distribution function energy range:")
    tdf_min = float(input("Minimum energy (eV) [-5.0]: ") or -5.0)
    tdf_max = float(input("Maximum energy (eV) [5.0]: ") or 5.0)
    tdf_step = float(input("Energy step (eV) [0.01]: ") or 0.01)
    transport_config["tdf_range"] = (tdf_min, tdf_max, tdf_step)
    
    # Relaxation time
    print("\nRelaxation time approximation:")
    tau = float(input("Electron lifetime (fs) [10]: ") or 10)
    transport_config["relaxation_time"] = tau
    
    # Smearing
    print("\nSmearing for distribution function:")
    smear = float(input("Smearing coefficient [0.0]: ") or 0.0)
    transport_config["smearing"] = smear
    
    if smear > 0:
        print("\nSmearing type:")
        print("-2: Cold smearing (Marzari-Vanderbilt)")
        print("-1: Fermi-Dirac")
        print(" 0: Methfessel-Paxton order 0")
        print(" 1-10: Methfessel-Paxton higher orders")
        
        smear_type = int(input("Smearing type [0]: ") or 0)
        transport_config["smearing_type"] = smear_type
    
    return transport_config


def configure_charge_density_calculation() -> Dict[str, Any]:
    """Configure ECH3 (3D charge density) or ECHG (2D charge density) calculation."""
    print("\n=== CHARGE DENSITY CONFIGURATION ===")
    
    density_config = {}
    
    # Dimension
    print("\nCharge density calculation type:")
    print("1: 3D grid (ECH3)")
    print("2: 2D map (ECHG)")
    print("3: 1D line profile")
    
    dim_type = input("Select type (1-3) [1]: ").strip() or "1"
    
    if dim_type == "1":
        density_config["type"] = "ECH3"
        
        # Number of points
        n_points = int(input("\nNumber of grid points along first direction [100]: ") or 100)
        density_config["n_points"] = n_points
        
        # For non-periodic directions
        print("\nGrid definition for non-periodic directions:")
        print("1: Scale from atomic coordinates")
        print("2: Explicit range")
        
        grid_method = input("Select method (1-2) [1]: ").strip() or "1"
        
        if grid_method == "1":
            scale = float(input("Scaling factor [3.0]: ") or 3.0)
            density_config["scale"] = scale
        else:
            # Will need to ask for explicit ranges based on dimensionality
            density_config["use_range"] = True
    
    elif dim_type == "2":
        density_config["type"] = "ECHG"
        
        # Derivative order
        print("\nDerivative order:")
        print("0: Charge density only")
        print("1: Charge density + gradient")
        print("2: Charge density + gradient + Laplacian")
        
        ider = int(input("Select order (0-2) [0]: ") or 0)
        density_config["derivative_order"] = ider
        
        # Map plane will be defined later
        print("\nMap plane will be defined by three points (A, B, C)")
        density_config["need_map_points"] = True
    
    else:
        density_config["type"] = "ECHG"
        density_config["derivative_order"] = 0
        density_config["line_profile"] = True
        print("\nLine profile will be defined by two points")
    
    # Output format
    print("\nOutput format:")
    print("1: CRYSTAL format (fort.31)")
    print("2: Gaussian CUBE format")
    print("3: Both formats")
    
    output_format = input("Select format (1-3) [2]: ").strip() or "2"
    density_config["output_format"] = int(output_format)
    
    # Spin density (if applicable)
    calc_spin = yes_no_prompt("\nCalculate spin density (if spin-polarized)?", "yes")
    density_config["spin_density"] = calc_spin
    
    return density_config


def configure_potential_calculation() -> Dict[str, Any]:
    """Configure POT3 (3D potential) or POTC (2D potential) calculation."""
    print("\n=== ELECTROSTATIC POTENTIAL CONFIGURATION ===")
    
    potential_config = {}
    
    # Type
    print("\nElectrostatic potential calculation type:")
    print("1: 3D grid (POT3)")
    print("2: At specific points (POTC)")
    print("3: Plane-averaged (2D slabs only)")
    print("4: Volume-averaged (2D slabs only)")
    
    pot_type = input("Select type (1-4) [1]: ").strip() or "1"
    
    if pot_type == "1":
        potential_config["type"] = "POT3"
        
        # Number of points
        n_points = int(input("\nNumber of grid points along first direction [100]: ") or 100)
        potential_config["n_points"] = n_points
        
        # Penetration tolerance
        itol = int(input("Penetration tolerance (suggested: 5) [5]: ") or 5)
        potential_config["itol"] = itol
        
        # Grid definition (similar to charge density)
        print("\nGrid definition for non-periodic directions:")
        print("1: Scale from atomic coordinates")
        print("2: Explicit range")
        
        grid_method = input("Select method (1-2) [1]: ").strip() or "1"
        
        if grid_method == "1":
            scale = float(input("Scaling factor [3.0]: ") or 3.0)
            potential_config["scale"] = scale
        else:
            potential_config["use_range"] = True
    
    else:
        potential_config["type"] = "POTC"
        potential_config["ica"] = int(pot_type) - 2
        
        if pot_type == "2":
            # Points
            print("\nCalculation points:")
            print("1: At atomic positions")
            print("2: At custom points")
            print("3: Read from POTC.INP file")
            
            point_method = input("Select method (1-3) [1]: ").strip() or "1"
            
            if point_method == "1":
                potential_config["at_atoms"] = True
                all_atoms = yes_no_prompt("Calculate at all atoms?", "yes")
                potential_config["all_atoms"] = all_atoms
            
            elif point_method == "2":
                n_points = int(input("Number of points: "))
                potential_config["n_points"] = n_points
                potential_config["custom_points"] = True
            
            else:
                potential_config["read_file"] = True
        
        elif pot_type in ["3", "4"]:
            # Plane/volume averaged (2D only)
            z_min = float(input("\nMinimum z position (bohr): "))
            z_max = float(input("Maximum z position (bohr): "))
            n_planes = int(input("Number of planes [100]: ") or 100)
            
            potential_config["z_range"] = (z_min, z_max)
            potential_config["n_planes"] = n_planes
            
            if pot_type == "4":
                thickness = float(input("Volume slice thickness (bohr): "))
                potential_config["slice_thickness"] = thickness
    
    return potential_config


def get_band_info_from_output(out_file: str) -> Tuple[int, int, int]:
    """Extract band information from CRYSTAL output file."""
    n_ao = 0
    n_electrons = 0
    
    if os.path.exists(out_file):
        with open(out_file, 'r') as f:
            content = f.read()
            
            # Find number of AOs
            import re
            ao_match = re.search(r'NUMBER OF AO\s+(\d+)', content)
            if ao_match:
                n_ao = int(ao_match.group(1))
            
            # Find number of electrons
            elec_match = re.search(r'N\. OF ELECTRONS PER UNIT CELL\s+(\d+)', content)
            if elec_match:
                n_electrons = int(elec_match.group(1))
    
    # Estimate band range
    if n_electrons > 0:
        n_occupied = n_electrons // 2
        first_band = max(1, n_occupied - 20)
        last_band = min(n_ao, n_occupied + 30)
    else:
        first_band = 1
        last_band = n_ao if n_ao > 0 else 100
    
    return n_ao, first_band, last_band


def configure_d3_calculation(calc_type: str, out_file: Optional[str] = None) -> Dict[str, Any]:
    """Main function to configure any D3 calculation type."""
    
    calc_type = calc_type.upper()
    
    if calc_type == "BAND":
        return configure_band_calculation(out_file)
    elif calc_type == "DOSS":
        return configure_doss_calculation(out_file)
    elif calc_type == "TRANSPORT":
        return configure_transport_calculation()
    elif calc_type in ["CHARGE", "ECH3", "ECHG"]:
        return configure_charge_density_calculation()
    elif calc_type in ["POTENTIAL", "POT3", "POTC"]:
        return configure_potential_calculation()
    elif calc_type == "CHARGE+POTENTIAL":
        # This is handled specially in CRYSTALOptToD3.py
        # Return empty config since it will configure each part separately
        return {}
    else:
        raise ValueError(f"Unknown calculation type: {calc_type}")


# ==============================================================================
# DOSS Utilities (merged from d3_doss_utils.py)
# ==============================================================================

def extract_shrink_from_d12(d12_file: str, target_product: float = 70.0) -> List[str]:
    """Extract SHRINK parameters from D12 file for NEWK.
    
    Args:
        d12_file: Path to D12 file
        target_product: Target value for a*ka, b*kb, c*kc (default 70 for DOSS)
    
    Returns:
        List of SHRINK lines
    """
    shrink_lines = []
    
    if not Path(d12_file).exists():
        return ["48 48\n"]  # Default fallback
    
    with open(d12_file, 'r') as f:
        lines = f.readlines()
    
    # Try to find SHRINK parameters
    found_shrink = False
    for i, line in enumerate(lines):
        if "SHRINK" in line:
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                parts = next_line.split()
                
                if len(parts) == 2:
                    # Could be symmetric (ka kb) or unsymmetric first line (0 kb)
                    val1, val2 = map(int, parts)
                    
                    if val1 == 0:
                        # Unsymmetric format: 0 kb followed by ka ka kc
                        shrink_lines.append(next_line + "\n")
                        if i + 2 < len(lines):
                            second_line = lines[i + 2].strip()
                            shrink_lines.append(second_line + "\n")
                            found_shrink = True
                    else:
                        # Symmetric format: ka kb (only one line needed)
                        shrink_lines.append(next_line + "\n")
                        found_shrink = True
                elif len(parts) == 3:
                    # Direct format: ka kb kc (only one line needed)
                    shrink_lines.append(next_line + "\n")
                    found_shrink = True
            break
    
    # If no SHRINK found but we can extract lattice parameters, calculate appropriate values
    if not found_shrink:
        # Try to extract lattice parameters
        a, b, c = None, None, None
        dimensionality = "CRYSTAL"
        spacegroup = 1
        
        for i, line in enumerate(lines):
            if "CRYSTAL" in line:
                dimensionality = "CRYSTAL"
            elif "SLAB" in line:
                dimensionality = "SLAB"
            elif "POLYMER" in line:
                dimensionality = "POLYMER"
            elif "MOLECULE" in line:
                dimensionality = "MOLECULE"
                
            # Look for lattice parameters (various possible formats)
            if any(x in line for x in ["PRIMITIVE CELL", "LATTICE PARAMETERS", "DIRECT LATTICE"]):
                # Next few lines should have parameters
                for j in range(i+1, min(i+5, len(lines))):
                    param_line = lines[j].strip()
                    numbers = re.findall(r"[-+]?\d*\.?\d+", param_line)
                    if len(numbers) >= 3:
                        try:
                            a = float(numbers[0])
                            b = float(numbers[1]) if len(numbers) > 1 else a
                            c = float(numbers[2]) if len(numbers) > 2 else a
                            break
                        except:
                            pass
                            
        # If we found lattice parameters, calculate appropriate SHRINK
        if a and b and c:
            # Scale up k-points for DOSS (target_product instead of 40)
            ka, kb, kc = generate_k_points(a, b, c, dimensionality, spacegroup)
            
            # Scale up for DOSS
            scale_factor = target_product / 40.0
            ka = int(ka * scale_factor)
            kb = int(kb * scale_factor)
            kc = int(kc * scale_factor)
            
            # Ensure reasonable values
            ks = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 18, 20, 24, 30, 36, 40, 45, 48, 60, 72, 80, 90, 96]
            ka = min([k for k in ks if k >= ka] or [ka])
            kb = min([k for k in ks if k >= kb] or [kb])
            kc = min([k for k in ks if k >= kc] or [kc])
            
            if dimensionality == "CRYSTAL":
                shrink_lines = [f"{ka} {kb}\n"]
            elif dimensionality == "SLAB":
                shrink_lines = [f"{ka} {kb}\n"]
            else:
                shrink_lines = [f"{ka} {ka}\n"]
                
    return shrink_lines if shrink_lines else ["48 48\n"]


def parse_basis_set_info(out_file: str) -> Tuple[List[List[str]], int]:
    """Parse basis set information from CRYSTAL output file."""
    with open(out_file, 'r') as f:
        lines = f.readlines()
    
    data_list = []
    n_ao = 0
    
    # Find LOCAL ATOMIC FUNCTIONS BASIS SET section
    basis_start = -1
    for i, line in enumerate(lines):
        if "LOCAL ATOMIC FUNCTIONS BASIS SET" in line:
            basis_start = i
            break
        if "NUMBER OF AO" in line:
            parts = line.split()
            for j, part in enumerate(parts):
                if part == "AO" and j + 1 < len(parts):
                    n_ao = int(parts[j + 1])
    
    if basis_start == -1:
        return [], n_ao
    
    # Parse basis set data
    for i in range(basis_start + 4, len(lines)):
        line = lines[i].strip()
        if "PROCESS" in line:
            continue
        if "*******" in line or "INFORMATION" in line or not line:
            break
        
        # Split line and filter empty strings
        parts = [x for x in line.split() if x]
        if parts:
            data_list.append(parts)
    
    return data_list, n_ao


def get_atoms_and_shells(data_list: List[List[str]]) -> Tuple[List[str], Dict[str, Dict[str, int]]]:
    """Extract atom types and count shells by type."""
    atoms = []
    atom_types = []
    atoms_shells = {}
    
    # First pass: identify atoms
    for line in data_list:
        if len(line) == 5:  # Atom line format
            atom_type = line[1]
            atoms.append(atom_type)
            if atom_type not in atom_types:
                atom_types.append(atom_type)
    
    # Second pass: count shells for each atom type
    for atom_type in atom_types:
        shells = {"S": 0, "SP": 0, "P": 0, "D": 0, "F": 0}
        
        # Find this atom type in data
        for i, line in enumerate(data_list):
            if len(line) == 5 and line[1] == atom_type:
                # Count shells for this atom
                j = i + 1
                while j < len(data_list) and len(data_list[j]) < 5:
                    shell_line = data_list[j]
                    if shell_line[-1] in shells:
                        if len(shell_line) == 2 and "-" in shell_line[0]:
                            # Range format like "1-4"
                            start, end = map(int, shell_line[0].split("-"))
                            shells[shell_line[-1]] += end - start + 1
                        elif len(shell_line) == 3:
                            # Format like "1-4 6"
                            start = int(shell_line[0].split("-")[0])
                            end = int(shell_line[1])
                            shells[shell_line[-1]] += end - start + 1
                        else:
                            shells[shell_line[-1]] += 1
                    j += 1
                break  # Only need to find first instance of this atom type
        
        atoms_shells[atom_type] = shells
    
    return atoms, atoms_shells


def generate_orbital_indices(atoms: List[str], atoms_shells: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, List[int]]]:
    """Generate orbital indices for each atom type and shell type."""
    total_shells = {}
    
    # Initialize structure
    for atom_type in atoms_shells:
        total_shells[atom_type] = {"S": [], "SP": [], "P": [], "D": [], "F": []}
    
    # Assign indices
    ao_index = 1
    for atom in atoms:
        shells = atoms_shells[atom]
        
        # S orbitals (1 function each)
        for _ in range(shells["S"]):
            total_shells[atom]["S"].append(ao_index)
            ao_index += 1
        
        # SP orbitals (4 functions each: 1s + 3p)
        for _ in range(shells["SP"]):
            total_shells[atom]["SP"].append(ao_index)
            ao_index += 4
            
        # P orbitals (3 functions each)
        for _ in range(shells["P"]):
            total_shells[atom]["P"].append(ao_index)
            ao_index += 3
            
        # D orbitals (5 functions each for Cartesian, 6 for spherical)
        for _ in range(shells["D"]):
            total_shells[atom]["D"].append(ao_index)
            ao_index += 5  # Assuming Cartesian d orbitals
            
        # F orbitals (7 functions each for spherical, 10 for Cartesian)
        for _ in range(shells["F"]):
            total_shells[atom]["F"].append(ao_index)
            ao_index += 7  # Assuming spherical f orbitals
    
    return total_shells


def create_doss_projections(total_shells: Dict[str, Dict[str, List[int]]], 
                           element_only: bool = False) -> List[str]:
    """Create DOSS projection lines for each element and orbital type.
    
    Args:
        total_shells: Dictionary of orbital indices by element and type
        element_only: If True, only create element projections (not orbital breakdown)
    
    Returns:
        List of projection lines
    """
    doss_lines = []
    
    # Sort atoms for consistent output
    sorted_atoms = sorted(total_shells.keys())
    
    for atom in sorted_atoms:
        atom_total_indices = []
        
        if not element_only:
            # S orbitals
            if total_shells[atom]["S"]:
                indices = " ".join(str(idx) for idx in total_shells[atom]["S"])
                doss_lines.append(f"{len(total_shells[atom]['S'])} {indices} #{atom} S")
                atom_total_indices.extend(total_shells[atom]["S"])
            
            # SP orbitals
            if total_shells[atom]["SP"]:
                indices = " ".join(str(idx) for idx in total_shells[atom]["SP"])
                doss_lines.append(f"{len(total_shells[atom]['SP'])} {indices} #{atom} SP")
                atom_total_indices.extend(total_shells[atom]["SP"])
            
            # P orbitals
            if total_shells[atom]["P"]:
                indices = " ".join(str(idx) for idx in total_shells[atom]["P"])
                doss_lines.append(f"{len(total_shells[atom]['P'])} {indices} #{atom} P")
                atom_total_indices.extend(total_shells[atom]["P"])
            
            # D orbitals
            if total_shells[atom]["D"]:
                indices = " ".join(str(idx) for idx in total_shells[atom]["D"])
                doss_lines.append(f"{len(total_shells[atom]['D'])} {indices} #{atom} D")
                atom_total_indices.extend(total_shells[atom]["D"])
            
            # F orbitals
            if total_shells[atom]["F"]:
                indices = " ".join(str(idx) for idx in total_shells[atom]["F"])
                doss_lines.append(f"{len(total_shells[atom]['F'])} {indices} #{atom} F")
                atom_total_indices.extend(total_shells[atom]["F"])
        else:
            # Collect all indices for element-only projection
            for orbital_type in ["S", "SP", "P", "D", "F"]:
                atom_total_indices.extend(total_shells[atom][orbital_type])
        
        # All orbitals for this atom
        if atom_total_indices:
            all_indices = " ".join(str(idx) for idx in sorted(atom_total_indices))
            doss_lines.append(f"{len(atom_total_indices)} {all_indices} #{atom} all")
    
    return doss_lines