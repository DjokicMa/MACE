#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CIF to D12 Converter for CRYSTAL23
----------------------------------
This script converts CIF files to D12 input files for CRYSTAL23 with multiple options
for calculation type, basis sets, functionals, and other computational parameters.

DESCRIPTION:
    This tool automates the process of converting CIF files to D12 input files for CRYSTAL23
    quantum chemical calculations. It allows customization of calculation types (single point,
    geometry optimization, frequency), basis sets, DFT functionals, and many other parameters.

REQUIRED PACKAGES:
    - numpy
    - ase (Atomic Simulation Environment)
    - spglib (Optional, for symmetry detection)

INSTALLATION:
    Using conda:
        conda install -c conda-forge numpy ase spglib

    Using pip:
        pip install numpy ase spglib

USAGE:
    1. Basic usage (interactive mode):
       python NewCifToD12.py --cif_dir /path/to/cif/files

    2. Save options for batch processing:
       python NewCifToD12.py --save_options --options_file my_settings.json

    3. Run in batch mode with saved options:
       python NewCifToD12.py --batch --options_file my_settings.json --cif_dir /path/to/cif/files

    4. Specify output directory:
       python NewCifToD12.py --cif_dir /path/to/cif/files --output_dir /path/to/output

CONFIGURATION:
    ** IMPORTANT: Before running, modify the path constants at the top of this script **

    DEFAULT_DZ_PATH = "./full.basis.doublezeta/"  # Path to double-zeta basis set directory
    DEFAULT_TZ_PATH = "./full.basis.triplezeta/"  # Path to triple-zeta basis set directory
    DEFAULT_TVZ_PATH = "./TVZP-rev2/"             # Path to TVZP-rev2 basis set directory

    Update these paths to point to your basis set directories on your system.

AUTHOR:
    Original script by Marcus Djokic, improved and extended by AI assistance
"""

import os
import sys
import glob
import argparse
import numpy as np
from ase.io import read

# Path constants for external basis sets - MODIFY THESE TO MATCH YOUR SYSTEM
DEFAULT_DZ_PATH = (
    "./full.basis.doublezeta/"  # Default path to double-zeta basis set directory
)
DEFAULT_TZ_PATH = (
    "./full.basis.triplezeta/"  # Default path to triple-zeta basis set directory
)
DEFAULT_TVZ_PATH = "./TVZP-rev2/"  # Default path to TVZP-rev2 basis set directory

# Try to import spglib for symmetry operations
try:
    import spglib

    SPGLIB_AVAILABLE = True
except ImportError:
    SPGLIB_AVAILABLE = False
    print("Warning: spglib not found. Symmetry reduction features will be limited.")
    print("Install spglib for full symmetry functionality: pip install spglib")


class Element:
    """Element atomic numbers for easy reference"""

    H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P = list(range(1, 16))
    S, Cl, Ar, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn = list(range(16, 31))
    Ga, Ge, As, Se, Br, Kr, Rb, Sr, Y, Zr, Nb, Mo, Tc, Ru = list(range(31, 45))
    Rh, Pd, Ag, Cd, In, Sn, Sb, Te, I, Xe, Cs, Ba, La, Ce = list(range(45, 59))
    Pr, Nd, Pm, Sm, Eu, Gd, Tb, Dy, Ho, Er, Tm, Yb, Lu, Hf = list(range(59, 73))
    Ta, W, Re, Os, Ir, Pt, Au, Hg, Tl, Pb, Bi, Po, At, Rn = list(range(73, 87))
    Fr, Ra, Ac, Th, Pa, U, Np, Pu, Am, Cm, Bk, Cf, Es, Fm = list(range(87, 101))
    Md, No, Lr, Rf, Db, Sg, Bh, Hs, Mt, Ds, Rg, Cn, Uut = list(range(101, 114))
    Fl, Uup, Lv, Uus, Uuo = list(range(114, 119))


# Dictionary mapping element symbols to atomic numbers
ELEMENT_SYMBOLS = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
    "Cs": 55,
    "Ba": 56,
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Pm": 61,
    "Sm": 62,
    "Eu": 63,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Lu": 71,
    "Hf": 72,
    "Ta": 73,
    "W": 74,
    "Re": 75,
    "Os": 76,
    "Ir": 77,
    "Pt": 78,
    "Au": 79,
    "Hg": 80,
}

# Space groups with multiple origin settings that need special handling
MULTI_ORIGIN_SPACEGROUPS = {
    # F-43m family
    216: {"name": "F-43m", "default": "Origin 2 (ITA)", "crystal_code": "0 0 0"},
    # P-43n family
    218: {"name": "P-43n", "default": "Origin 2 (ITA)", "crystal_code": "0 0 0"},
    # Cubic groups with inversion center
    221: {"name": "Pm-3m", "default": "Origin 2 (ITA)", "crystal_code": "0 0 0"},
    225: {"name": "Fm-3m", "default": "Origin 2 (ITA)", "crystal_code": "0 0 0"},
    227: {
        "name": "Fd-3m",
        "default": "Origin 2 (ITA)",
        "crystal_code": "0 0 0",
        "alt": "Origin 1",
        "alt_crystal_code": "0 0 1",
        "default_pos": (0.125, 0.125, 0.125),
        "alt_pos": (0.0, 0.0, 0.0),
    },
    228: {"name": "Fd-3c", "default": "Origin 2 (ITA)", "crystal_code": "0 0 0"},
    229: {"name": "Im-3m", "default": "Origin 2 (ITA)", "crystal_code": "0 0 0"},
    230: {"name": "Ia-3d", "default": "Origin 2 (ITA)", "crystal_code": "0 0 0"},
}

RHOMBOHEDRAL_SPACEGROUPS = [146, 148, 155, 160, 161, 166, 167]

# Elements that require ECPs (Effective Core Potentials)
ECP_ELEMENTS = [
    37,
    38,
    39,
    40,
    41,
    42,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    66,
    67,
    68,
    69,
    70,
    71,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
]

# Available DFT functionals
DFT_FUNCTIONALS = [
    "B3LYP",
    "PBE0",
    "PBESOL0",
    "HSE06",
    "BLYP",
    "PBE",
    "B97",
    "PW1PW",
    "M06",
    "HSEsol",
    "LC-wPBE",
]

# Functionals available for D3 dispersion correction
D3_FUNCTIONALS = [
    "BLYP",
    "PBE",
    "B97",
    "B3LYP",
    "PBE0",
    "PW1PW",
    "M06",
    "HSE06",
    "HSEsol",
    "LC-wPBE",
]

# Available internal basis sets
INTERNAL_BASIS_SETS = [
    "STO-3G",
    "STO-6G",
    "POB-DZVP",
    "POB-DZVPP",
    "POB-TZVP",
    "POB-DZVP-REV2",
    "POB-TZVP-REV2",
]

# Available SCF convergence methods
SCF_METHODS = ["DIIS", "ANDERSON", "BROYDEN"]

# Available DFT grid sizes
DFT_GRIDS = {
    "1": "OLDGRID",  # Old default grid from CRYSTAL09, pruned (55,434)
    "2": "DEFAULT",  # Default grid in CRYSTAL23
    "3": "LGRID",  # Large grid, pruned (75,434)
    "4": "XLGRID",  # Extra large grid (default)
    "5": "XXLGRID",  # Extra extra large grid, pruned (99,1454)
    "6": "XXXLGRID",  # Ultra extra extra large grid, pruned (150,1454)
    "7": "HUGEGRID",  # Ultra extra extra large grid for SCAN, pruned (300,1454)
}

# Available optimization types
OPT_TYPES = {"1": "FULLOPTG", "2": "CVOLOPT", "3": "CELLONLY", "4": "ATOMONLY"}

# Default geom optimization settings
DEFAULT_OPT_SETTINGS = {
    "TOLDEG": 0.00003,  # RMS of the gradient
    "TOLDEX": 0.00012,  # RMS of the displacement
    "TOLDEE": 7,  # Energy difference between two steps (10^-n)
    "MAXCYCLE": 800,  # Max number of optimization steps
    "MAXTRADIUS": 0.25,  # Max displacement (default, optional)
}

# Default frequency calculation settings
DEFAULT_FREQ_SETTINGS = {
    "NUMDERIV": 2,  # Numerical derivative level
    "TOLINTEG": "12 12 12 12 24",  # Tighter tolerance for frequencies
    "TOLDEE": 12,  # Tighter SCF convergence for frequencies
}

# Default tolerance settings
DEFAULT_TOLERANCES = {
    "TOLINTEG": "7 7 7 7 14",  # Default integration tolerances
    "TOLDEE": 7,  # SCF energy tolerance (exponent)
}


def format_crystal_float(value):
    """
    Format a floating point value in a way that CRYSTAL23 can interpret.
    - Use decimal format to avoid scientific notation issues

    Args:
        value (float): The value to format

    Returns:
        str: The formatted value
    """
    if isinstance(value, int):
        return str(value)

    abs_value = abs(value)
    if abs_value == 0.0:
        return "0.0"
    elif abs_value < 0.0001:
        # For very small values, use a decimal with enough precision (avoid scientific notation)
        return (
            f"{value:.10f}".rstrip("0").rstrip(".")
            if "." in f"{value:.10f}"
            else f"{value:.1f}"
        )
    else:
        # Use decimal format with appropriate precision
        return (
            f"{value:.8f}".rstrip("0").rstrip(".")
            if "." in f"{value:.8f}"
            else f"{value:.1f}"
        )


def get_user_input(prompt, options, default=None):
    """
    Get validated user input from a list of options

    Args:
        prompt (str): The prompt to display to the user
        options (list or dict): Valid options
        default (str, optional): Default value

    Returns:
        str: Valid user input
    """
    if isinstance(options, dict):
        opt_str = "\n".join([f"{key}: {value}" for key, value in options.items()])
        valid_inputs = options.keys()
    else:
        opt_str = "\n".join([f"{i + 1}: {opt}" for i, opt in enumerate(options)])
        valid_inputs = [str(i + 1) for i in range(len(options))]

    default_str = f" (default: {default})" if default else ""

    while True:
        print(f"\n{prompt}{default_str}:\n{opt_str}")
        choice = input("Enter your choice: ").strip()

        if choice == "" and default:
            return default

        if choice in valid_inputs:
            return choice

        print(f"Invalid input. Please choose from {', '.join(valid_inputs)}")


def yes_no_prompt(prompt, default="yes"):
    """
    Prompt for a yes/no response

    Args:
        prompt (str): The prompt to display
        default (str): Default value ('yes' or 'no')

    Returns:
        bool: True for yes, False for no
    """
    valid = {"yes": True, "y": True, "no": False, "n": False}
    if default == "yes":
        prompt += " [Y/n] "
    elif default == "no":
        prompt += " [y/N] "
    else:
        raise ValueError(f"Invalid default value: {default}")

    while True:
        choice = input(prompt).lower() or default
        if choice in valid:
            return valid[choice]
        print("Please respond with 'yes' or 'no' (or 'y' or 'n').")


def read_basis_file(basis_dir, atomic_number):
    """
    Read a basis set file for a given element

    Args:
        basis_dir (str): Directory containing basis set files
        atomic_number (int): Element atomic number

    Returns:
        str: Content of the basis set file
    """
    try:
        with open(os.path.join(basis_dir, str(atomic_number)), "r") as f:
            return f.read()
    except FileNotFoundError:
        print(
            f"Warning: Basis set file for element {atomic_number} not found in {basis_dir}"
        )
        return ""


def unique_elements(element_list):
    """Get unique elements from a list, sorted"""
    unique_list = []
    for element in element_list:
        if element not in unique_list:
            unique_list.append(element)
    return sorted(unique_list)


def parse_cif(cif_file):
    """
    Parse a CIF file to extract crystallographic data

    Args:
        cif_file (str): Path to the CIF file

    Returns:
        dict: Extracted crystallographic data
    """
    try:
        # Try reading with ASE first
        atoms = read(cif_file, format="cif")
        cell_params = atoms.get_cell_lengths_and_angles()
        a, b, c = cell_params[:3]
        alpha, beta, gamma = cell_params[3:]

        # Get atomic positions and symbols
        positions = atoms.get_scaled_positions()
        symbols = atoms.get_chemical_symbols()

        # Get space group number if available
        spacegroup = None
        cif_symmetry_name = None

        # First try to get from ASE info
        if hasattr(atoms, "info") and "spacegroup" in atoms.info:
            spacegroup = atoms.info["spacegroup"].no

        # If not available, try to parse from CIF file directly
        if spacegroup is None:
            with open(cif_file, "r") as f:
                cif_content = f.read()

                # Look for International Tables number
                import re

                sg_match = re.search(
                    r"_symmetry_Int_Tables_number\s+(\d+)", cif_content
                )
                if sg_match:
                    spacegroup = int(sg_match.group(1))

                # Also get the H-M symbol if available for reference
                hm_match = re.search(
                    r'_symmetry_space_group_name_H-M\s+[\'"](.*?)[\'"]', cif_content
                )
                if hm_match:
                    cif_symmetry_name = hm_match.group(1)

        # If still not found, prompt user
        if spacegroup is None:
            print(f"Warning: Space group not found in {cif_file}")
            if cif_symmetry_name:
                print(f"Found Hermann-Mauguin symbol: {cif_symmetry_name}")
            spacegroup = int(input("Please enter the space group number: "))

        # Convert symbols to atomic numbers
        atomic_numbers = [ELEMENT_SYMBOLS.get(sym, 0) for sym in symbols]

        return {
            "a": a,
            "b": b,
            "c": c,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "spacegroup": spacegroup,
            "cif_symmetry_name": cif_symmetry_name,
            "atomic_numbers": atomic_numbers,
            "symbols": symbols,
            "positions": positions,
            "name": os.path.basename(cif_file).replace(".cif", ""),
        }

    except Exception as e:
        # If ASE fails, use manual parsing
        print(f"ASE parsing failed: {e}")
        print("Falling back to manual parsing...")

        with open(cif_file, "r") as f:
            contents = f.readlines()

        # Initialize variables
        data = {
            "a": None,
            "b": None,
            "c": None,
            "alpha": None,
            "beta": None,
            "gamma": None,
            "spacegroup": None,
            "atomic_numbers": [],
            "symbols": [],
            "positions": [],
            "name": os.path.basename(cif_file).replace(".cif", ""),
        }

        # Counters for parsing
        sym_counter = 0
        atom_counter = 0
        a_counter = b_counter = c_counter = 0
        alpha_counter = beta_counter = gamma_counter = 0
        atom_list = []

        # Parse CIF file
        for line in contents:
            words = line.split()
            for i, word in enumerate(words):
                # Get lattice parameters
                if word == "_cell_length_a":
                    a_counter = 1
                elif a_counter == 1:
                    data["a"] = float(word)
                    a_counter = 0

                if word == "_cell_length_b":
                    b_counter = 1
                elif b_counter == 1:
                    data["b"] = float(word)
                    b_counter = 0

                if word == "_cell_length_c":
                    c_counter = 1
                elif c_counter == 1:
                    data["c"] = float(word)
                    c_counter = 0

                # Get unit cell angles
                if word == "_cell_angle_alpha":
                    alpha_counter = 1
                elif alpha_counter == 1:
                    data["alpha"] = float(word)
                    alpha_counter = 0

                if word == "_cell_angle_beta":
                    beta_counter = 1
                elif beta_counter == 1:
                    data["beta"] = float(word)
                    beta_counter = 0

                if word == "_cell_angle_gamma":
                    gamma_counter = 1
                elif gamma_counter == 1:
                    data["gamma"] = float(word)
                    gamma_counter = 0

                # Get space group
                if (
                    word == "_symmetry_Int_Tables_number"
                    or word == "_space_group_IT_number"
                ):
                    sym_counter = 1
                elif sym_counter == 1:
                    data["spacegroup"] = int(word)
                    sym_counter = 0

                # Get atom data
                if word == "_atom_site_occupancy":
                    atom_counter = 1
                elif word == "loop_":
                    atom_counter = 0
                elif atom_counter == 1:
                    atom_list.append(word)

        # Process atom data
        true_index = 0
        index = 0
        atom_name = []
        h = []
        k = []
        l = []

        for i in atom_list:
            if index == 1:
                atom_name.append(i)
            if index == 2:
                h.append(float(i))
            if index == 3:
                k.append(float(i))
            if index == 4:
                l.append(float(i))

            true_index += 1
            index += 1
            if index == 8:
                index = 0

        # Convert atom names to atomic numbers
        atomic_numbers = []
        for name in atom_name:
            try:
                atom = getattr(Element, name)
                atomic_numbers.append(int(atom))
            except (AttributeError, ValueError):
                atomic_numbers.append(ELEMENT_SYMBOLS.get(name, 0))

        # Create fractional coordinates
        positions = []
        for i in range(len(h)):
            positions.append([h[i], k[i], l[i]])

        data["atomic_numbers"] = atomic_numbers
        data["symbols"] = atom_name
        data["positions"] = positions

        return data


def generate_unit_cell_line(spacegroup, a, b, c, alpha, beta, gamma):
    """
    Generate the unit cell line for a CRYSTAL23 input file based on space group

    Args:
        spacegroup (int): Space group number
        a, b, c (float): Lattice parameters
        alpha, beta, gamma (float): Cell angles

    Returns:
        str: Unit cell line for CRYSTAL23 input
    """
    if spacegroup >= 1 and spacegroup <= 2:  # Triclinic
        return f"{a:.8f} {b:.8f} {c:.8f} {alpha:.6f} {beta:.6f} {gamma:.6f} #a,b,c,alpha,beta,gamma Triclinic"
    elif spacegroup >= 3 and spacegroup <= 15:  # Monoclinic
        return f"{a:.8f} {b:.8f} {c:.8f} {beta:.6f} #a,b,c,beta Monoclinic alpha = gamma = 90"
    elif spacegroup >= 16 and spacegroup <= 74:  # Orthorhombic
        return f"{a:.8f} {b:.8f} {c:.8f} #a,b,c Orthorhombic alpha = beta = gamma = 90"
    elif spacegroup >= 75 and spacegroup <= 142:  # Tetragonal
        return f"{a:.8f} {c:.8f} #a=b,c Tetragonal alpha = beta = gamma = 90"
    elif spacegroup >= 143 and spacegroup <= 167:  # Trigonal
        return f"{a:.8f} {c:.8f} #a=b,c Trigonal alpha = beta = 90, gamma = 120"
    elif spacegroup >= 168 and spacegroup <= 194:  # Hexagonal
        return f"{a:.8f} {c:.8f} #a=b,c Hexagonal alpha = beta = 90, gamma = 120"
    elif spacegroup >= 195 and spacegroup <= 230:  # Cubic
        return f"{a:.8f} #a=b=c cubic alpha = beta = gamma = 90"
    else:
        raise ValueError(f"Invalid space group: {spacegroup}")


def generate_k_points(a, b, c, dimensionality, spacegroup):
    """
    Generate Monkhorst-Pack k-point grid based on cell parameters

    Args:
        a, b, c (float): Cell parameters
        dimensionality (str): CRYSTAL, SLAB, POLYMER, or MOLECULE
        spacegroup (int): Space group number

    Returns:
        tuple: ka, kb, kc values for shrinking factor
    """
    ks = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 18, 20, 24, 30, 36, 40, 45, 48, 60]

    # Initialize defaults
    ka = kb = kc = 1

    # Find appropriate values based on cell dimensions
    for k in ks:
        if k * a > 40.0 and k * a < 80.0 and ka == 1:
            ka = k
        if k * b > 40.0 and k * b < 80.0 and kb == 1:
            kb = k
        if k * c > 40.0 and k * c < 80.0 and kc == 1:
            kc = k

    # Adjust based on dimensionality
    if dimensionality == "SLAB":
        kc = 1
    elif dimensionality == "POLYMER":
        kb = kc = 1
    elif dimensionality == "MOLECULE":
        ka = kb = kc = 1

    # Ensure reasonable values
    if ka == 1 and dimensionality not in ["POLYMER", "MOLECULE"]:
        ka = 12
    if kb == 1 and dimensionality not in ["POLYMER", "MOLECULE"]:
        kb = 12
    if kc == 1 and dimensionality not in ["SLAB", "POLYMER", "MOLECULE"]:
        kc = 12

    # For non-P1 symmetry, try to use consistent k-points
    if spacegroup != 1 and dimensionality == "CRYSTAL":
        # For high symmetry systems, use a consistent k-point mesh
        k_values = [k for k in [ka, kb, kc] if k > 1]
        if k_values:
            k_avg = round(sum(k_values) / len(k_values))
            k_avg = min([k for k in ks if k >= k_avg] or [k_avg])

            # Apply the common k value according to crystal system
            if spacegroup >= 195 and spacegroup <= 230:  # Cubic
                ka = kb = kc = k_avg
            elif (
                spacegroup >= 75 and spacegroup <= 194
            ):  # Tetragonal, Trigonal, Hexagonal
                ka = kb = k_avg
            elif spacegroup >= 16 and spacegroup <= 74:  # Orthorhombic
                # Keep different values but round to nearest in ks list
                ka = min([k for k in ks if k >= ka] or [ka])
                kb = min([k for k in ks if k >= kb] or [kb])
                kc = min([k for k in ks if k >= kc] or [kc])

    return ka, kb, kc


def get_calculation_options():
    """Gather calculation options from user"""
    options = {}

    # Ask about symmetry handling
    symmetry_options = {
        "1": "CIF",  # Use symmetry as defined in the CIF file (trust the file)
        "2": "SPGLIB",  # Use spglib to analyze symmetry (may detect different symmetry)
        "3": "P1",  # Use P1 symmetry (all atoms explicit, no symmetry)
    }
    symmetry_choice = get_user_input(
        "Select symmetry handling method", symmetry_options, "1"
    )
    options["symmetry_handling"] = symmetry_options[symmetry_choice]

    # If using spglib symmetry analysis
    if options["symmetry_handling"] == "SPGLIB" and SPGLIB_AVAILABLE:
        # Ask about symmetry tolerance
        tolerance_options = {
            "1": 1e-3,  # Loose tolerance - more forgiving of deviations
            "2": 1e-5,  # Default tolerance
            "3": 1e-7,  # Strict tolerance - requires high precision
        }
        tolerance_choice = get_user_input(
            "Select symmetry detection tolerance", tolerance_options, "2"
        )
        options["symmetry_tolerance"] = tolerance_options[tolerance_choice]

        # Ask about asymmetric unit reduction
        reduce_atoms = yes_no_prompt(
            "Reduce structure to asymmetric unit using spglib?", "yes"
        )
        options["reduce_to_asymmetric"] = reduce_atoms

    # For trigonal space groups, ask about axis representation
    trigonal_axes_options = {
        "1": "AUTO",  # Use setting as detected in CIF
        "2": "HEXAGONAL_AXES",  # Force hexagonal axes
        "3": "RHOMBOHEDRAL_AXES",  # Force rhombohedral axes
    }
    trigonal_axes_choice = get_user_input(
        "For trigonal space groups (143-167), which axes do you prefer?",
        trigonal_axes_options,
        "1",
    )
    options["trigonal_axes"] = trigonal_axes_options[trigonal_axes_choice]

    # For space groups with multiple origins, especially 227 (Fd-3m)
    origin_options = {
        "1": "AUTO",  # Use origin as detected in CIF
        "2": "STANDARD",  # Force standard origin (ITA Origin 2) - CRYSTAL "0 0 0"
        "3": "ALTERNATE",  # Force alternate origin (ITA Origin 1) - CRYSTAL "0 0 1"
    }
    origin_choice = get_user_input(
        "For space groups with multiple origins (e.g., 227-Fd-3m):\n"
        "  Standard (ITA Origin 2, CRYSTAL '0 0 0'): Si at (1/8,1/8,1/8), 36 operators\n"
        "  Alternate (ITA Origin 1, CRYSTAL '0 0 1'): Si at (0,0,0), 24 operators",
        origin_options,
        "1",
    )
    options["origin_setting"] = origin_options[origin_choice]

    # Get dimensionality
    dimensionality_options = {
        "1": "CRYSTAL",
        "2": "SLAB",
        "3": "POLYMER",
        "4": "MOLECULE",
    }
    dimensionality_choice = get_user_input(
        "Select the dimensionality of the system", dimensionality_options, "1"
    )
    options["dimensionality"] = dimensionality_options[dimensionality_choice]

    # Get dimensionality
    dimensionality_options = {
        "1": "CRYSTAL",
        "2": "SLAB",
        "3": "POLYMER",
        "4": "MOLECULE",
    }
    dimensionality_choice = get_user_input(
        "Select the dimensionality of the system", dimensionality_options, "1"
    )
    options["dimensionality"] = dimensionality_options[dimensionality_choice]

    # Special handling for cubic/rhombohedral structures
    if options["dimensionality"] == "CRYSTAL":
        # Ask about origin choice for cubic structures
        origin_options = {
            "1": "0 0 0",  # Default origin
            "2": "0 0 1",  # Alternative origin for certain space groups (like 227 diamond)
        }
        origin_choice = get_user_input(
            "Select origin setting (use 0 0 1 for certain space groups like 227 diamond with origin 2)",
            origin_options,
            "1",
        )
        options["origin_choice"] = origin_options[origin_choice]

        # Ask about rhombohedral setting
        use_rhombohedral = yes_no_prompt(
            "Use rhombohedral setting for applicable space groups (e.g., 146, 148, 155, etc.)? (0 1 0)",
            "no",
        )
        options["use_rhombohedral"] = use_rhombohedral

    # Get calculation type
    calc_options = {
        "1": "SP",  # Single Point
        "2": "OPT",  # Geometry Optimization
        "3": "FREQ",  # Frequency Calculation
    }
    calc_choice = get_user_input("Select calculation type", calc_options, "1")
    options["calculation_type"] = calc_options[calc_choice]

    # If geometry optimization, get optimization type
    if options["calculation_type"] == "OPT":
        opt_choice = get_user_input("Select optimization type", OPT_TYPES, "1")
        options["optimization_type"] = OPT_TYPES[opt_choice]

        # Ask for default or custom optimization settings
        use_default_opt = yes_no_prompt(
            "Use default optimization settings? (TOLDEG=0.00003, TOLDEX=0.00012, TOLDEE=7, MAXCYCLE=800)",
            "yes",
        )

        if not use_default_opt:
            custom_settings = {}
            custom_settings["TOLDEG"] = float(
                input("Enter TOLDEG (RMS of gradient, default 0.00003): ") or 0.00003
            )
            custom_settings["TOLDEX"] = float(
                input("Enter TOLDEX (RMS of displacement, default 0.00012): ")
                or 0.00012
            )
            custom_settings["TOLDEE"] = int(
                input("Enter TOLDEE (energy difference exponent, default 7): ") or 7
            )
            custom_settings["MAXCYCLE"] = int(
                input("Enter MAXCYCLE (max optimization steps, default 800): ") or 800
            )
            options["optimization_settings"] = custom_settings
        else:
            options["optimization_settings"] = DEFAULT_OPT_SETTINGS.copy()

        # Ask about MAXTRADIUS
        use_maxtradius = yes_no_prompt(
            "Set maximum step size (MAXTRADIUS) for geometry optimization?", "no"
        )

        if use_maxtradius:
            if "optimization_settings" not in options:
                options["optimization_settings"] = DEFAULT_OPT_SETTINGS.copy()
            maxtradius = float(
                input("Enter MAXTRADIUS (max displacement, default 0.25): ") or 0.25
            )
            options["optimization_settings"]["MAXTRADIUS"] = maxtradius

    # If frequency calculation, ask about numerical derivative level
    if options["calculation_type"] == "FREQ":
        use_default_freq = yes_no_prompt(
            "Use default frequency calculation settings? (NUMDERIV=2, TOLINTEG=12 12 12 12 24, TOLDEE=12)",
            "yes",
        )

        if not use_default_freq:
            custom_settings = {}
            custom_settings["NUMDERIV"] = int(
                input("Enter NUMDERIV (numerical derivative level, default 2): ") or 2
            )
            options["freq_settings"] = custom_settings
        else:
            options["freq_settings"] = DEFAULT_FREQ_SETTINGS.copy()

        # For frequency calculations, use tighter default tolerances
        if "tolerances" not in options:
            options["tolerances"] = {
                "TOLINTEG": DEFAULT_FREQ_SETTINGS["TOLINTEG"],
                "TOLDEE": DEFAULT_FREQ_SETTINGS["TOLDEE"],
            }

    # Get basis set type
    basis_options = {"1": "EXTERNAL", "2": "INTERNAL"}
    basis_choice = get_user_input("Select basis set type", basis_options, "1")
    options["basis_set_type"] = basis_options[basis_choice]

    # Get specific basis set
    if options["basis_set_type"] == "EXTERNAL":
        external_basis_options = {
            "1": DEFAULT_DZ_PATH,  # Default double-zeta path
            "2": DEFAULT_TZ_PATH,  # Default triple-zeta path
            "3": DEFAULT_TVZ_PATH,  # Default TVZP-rev2 path
        }

        basis_dir_choice = get_user_input(
            f"Select external basis set directory (current defaults: 1={DEFAULT_DZ_PATH}, 2={DEFAULT_TZ_PATH}, 3={DEFAULT_TVZ_PATH})",
            external_basis_options,
            "1",
        )

        # Allow user to override the default path if needed
        selected_path = external_basis_options[basis_dir_choice]
        custom_path = input(
            f"Use this path ({selected_path}) or enter a custom path (press Enter to use default): "
        )

        if custom_path:
            options["basis_set"] = custom_path
        else:
            options["basis_set"] = selected_path
    else:
        internal_basis_options = {
            str(i + 1): bs for i, bs in enumerate(INTERNAL_BASIS_SETS)
        }
        internal_basis_choice = get_user_input(
            "Select internal basis set",
            internal_basis_options,
            "6",  # Default to POB-TZVP-REV2
        )
        options["basis_set"] = internal_basis_options[internal_basis_choice]

    # Get functional
    functional_options = {str(i + 1): func for i, func in enumerate(DFT_FUNCTIONALS)}
    functional_choice = get_user_input(
        "Select DFT functional",
        functional_options,
        "1",  # Default to B3LYP
    )
    options["dft_functional"] = functional_options[functional_choice]

    # Check if dispersion correction is available
    if options["dft_functional"] in D3_FUNCTIONALS:
        use_dispersion = yes_no_prompt(
            f"Add D3 dispersion correction to {options['dft_functional']}?", "no"
        )
        options["use_dispersion"] = use_dispersion
    else:
        print(
            f"Note: D3 dispersion correction not available for {options['dft_functional']}"
        )
        options["use_dispersion"] = False

    # Get DFT grid
    grid_choice = get_user_input(
        "Select DFT integration grid",
        DFT_GRIDS,
        "4",  # Default to XLGRID
    )
    options["dft_grid"] = DFT_GRIDS[grid_choice]

    # Ask about spin polarization
    is_spin_polarized = yes_no_prompt("Use spin-polarized calculation?", "no")
    options["is_spin_polarized"] = is_spin_polarized

    # Ask about Fermi surface smearing for metals
    use_smearing = yes_no_prompt(
        "Use Fermi surface smearing for metallic systems?", "no"
    )
    options["use_smearing"] = use_smearing

    if use_smearing:
        smearing_width = float(
            input(
                "Enter smearing width in hartree (recommended: 0.001-0.02, default 0.01): "
            )
            or 0.01
        )
        options["smearing_width"] = smearing_width

    # Get tolerance settings (if not already set by FREQ)
    if "tolerances" not in options:
        use_default_tol = yes_no_prompt(
            "Use default tolerance settings? (TOLINTEG=7 7 7 7 14, TOLDEE=7)", "yes"
        )

        if not use_default_tol:
            custom_tol = {}
            tolinteg = input(
                "Enter TOLINTEG values (5 integers separated by spaces, default 7 7 7 7 14): "
            )
            tolinteg = tolinteg if tolinteg else "7 7 7 7 14"
            custom_tol["TOLINTEG"] = tolinteg

            toldee = input("Enter TOLDEE value (integer, default 7): ")
            toldee = int(toldee) if toldee else 7
            custom_tol["TOLDEE"] = toldee

            options["tolerances"] = custom_tol
        else:
            options["tolerances"] = DEFAULT_TOLERANCES.copy()

    # Get SCF convergence method
    scf_options = {str(i + 1): method for i, method in enumerate(SCF_METHODS)}
    scf_choice = get_user_input(
        "Select SCF convergence method",
        scf_options,
        "1",  # Default to DIIS
    )
    options["scf_method"] = scf_options[scf_choice]

    # Ask about SCF MAXCYCLE
    use_default_scf_maxcycle = yes_no_prompt("Use default SCF MAXCYCLE (800)?", "yes")

    if not use_default_scf_maxcycle:
        options["scf_maxcycle"] = int(input("Enter SCF MAXCYCLE value: ") or 800)
    else:
        options["scf_maxcycle"] = 800

    # Ask about FMIXING
    use_default_fmixing = yes_no_prompt("Use default FMIXING (30%)?", "yes")

    if not use_default_fmixing:
        options["fmixing"] = int(
            input("Enter FMIXING percentage (0-100, default 30): ") or 30
        )
    else:
        options["fmixing"] = 30

    return options


def create_d12_file(cif_data, output_file, options):
    """
    Create a D12 input file for CRYSTAL23 from CIF data

    Args:
        cif_data (dict): Parsed CIF data
        output_file (str): Output file path
        options (dict): Calculation options

    Returns:
        None
    """
    # Extract CIF data
    a = cif_data["a"]
    b = cif_data["b"]
    c = cif_data["c"]
    alpha = cif_data["alpha"]
    beta = cif_data["beta"]
    gamma = cif_data["gamma"]
    spacegroup = cif_data["spacegroup"]
    atomic_numbers = cif_data["atomic_numbers"]
    symbols = cif_data["symbols"]
    positions = cif_data["positions"]

    # Extract options
    dimensionality = options["dimensionality"]
    calculation_type = options["calculation_type"]
    optimization_type = options.get("optimization_type", None)
    optimization_settings = options.get("optimization_settings", DEFAULT_OPT_SETTINGS)
    freq_settings = options.get("freq_settings", DEFAULT_FREQ_SETTINGS)
    basis_set_type = options["basis_set_type"]
    basis_set = options["basis_set"]
    dft_functional = options["dft_functional"]
    use_dispersion = options["use_dispersion"]
    is_spin_polarized = options["is_spin_polarized"]
    tolerances = options["tolerances"]
    scf_method = options["scf_method"]
    scf_maxcycle = options.get("scf_maxcycle", 800)
    fmixing = options.get("fmixing", 30)
    dft_grid = options.get("dft_grid", "XLGRID")
    use_smearing = options.get("use_smearing", False)
    smearing_width = options.get("smearing_width", 0.01)

    # Determine crystal settings
    trigonal_setting = cif_data.get("trigonal_setting", None)
    origin_setting = options.get("origin_setting", "AUTO")

    # Determine space group origin settings
    origin_directive = "0 0 0"  # Default value

    # Handle space groups with multiple origin choices
    if spacegroup in MULTI_ORIGIN_SPACEGROUPS:
        spg_info = MULTI_ORIGIN_SPACEGROUPS[spacegroup]

        # Handle origin setting
        if origin_setting == "STANDARD":
            origin_directive = spg_info["crystal_code"]
            print(
                f"Using standard origin setting ({spg_info['default']}) for space group {spacegroup} ({spg_info['name']})"
            )
            print(f"CRYSTAL directive: {origin_directive}")

        elif origin_setting == "ALTERNATE" and "alt_crystal_code" in spg_info:
            origin_directive = spg_info["alt_crystal_code"]
            print(
                f"Using alternate origin setting ({spg_info['alt']}) for space group {spacegroup} ({spg_info['name']})"
            )
            print(f"CRYSTAL directive: {origin_directive}")

        elif (
            origin_setting == "AUTO" and spacegroup == 227
        ):  # Special handling for Fd-3m
            # Try to detect based on atom positions
            std_pos = spg_info.get("default_pos", (0.125, 0.125, 0.125))
            alt_pos = spg_info.get("alt_pos", (0.0, 0.0, 0.0))

            # Check if any atoms are near the standard position
            std_detected = False
            alt_detected = False

            for pos in positions:
                # Check for atoms near standard position (1/8, 1/8, 1/8)
                if (
                    abs(pos[0] - std_pos[0]) < 0.01
                    and abs(pos[1] - std_pos[1]) < 0.01
                    and abs(pos[2] - std_pos[2]) < 0.01
                ):
                    std_detected = True

                # Check for atoms near alternate position (0, 0, 0)
                if (
                    abs(pos[0] - alt_pos[0]) < 0.01
                    and abs(pos[1] - alt_pos[1]) < 0.01
                    and abs(pos[2] - alt_pos[2]) < 0.01
                ):
                    alt_detected = True

            if alt_detected and not std_detected:
                # If only alternate position atoms found, use alternate origin
                origin_directive = spg_info["alt_crystal_code"]
                print(
                    f"Detected alternate origin ({spg_info['alt']}) for space group 227 (atoms at {alt_pos})"
                )
                print(
                    f"Using CRYSTAL directive: {origin_directive} (fewer symmetry operators with translational components)"
                )
            else:
                # Default to standard origin
                origin_directive = spg_info["crystal_code"]
                print(
                    f"Using standard origin ({spg_info['default']}) for space group 227"
                )
                print(f"CRYSTAL directive: {origin_directive}")

    # Handle trigonal space groups for rhombohedral axes directive
    use_rhombohedral_axes = False
    if is_trigonal(spacegroup) and trigonal_setting == "rhombohedral_axes":
        use_rhombohedral_axes = True
        print(f"Using rhombohedral axes (0 1 0) for trigonal space group {spacegroup}")

    # Generate functional string with D3 correction if requested
    functional = dft_functional
    if use_dispersion and dft_functional in D3_FUNCTIONALS:
        functional += "-D3"

    # Open output file
    with open(output_file, "w") as f:
        # Write title
        print(os.path.basename(output_file).replace(".d12", ""), file=f)

        # Write structure type and space group
        if dimensionality == "CRYSTAL":
            print("CRYSTAL", file=f)

            # Handle specific origin settings for space groups
            if spacegroup in MULTI_ORIGIN_SPACEGROUPS:
                print(origin_directive, file=f)
            # Handle rhombohedral axes for trigonal space groups
            elif is_trigonal(spacegroup) and use_rhombohedral_axes:
                print("0 1 0", file=f)  # Use rhombohedral axes setting
            else:
                print("0 0 0", file=f)  # Default: use standard setting

            print(spacegroup, file=f)
            print(
                generate_unit_cell_line(spacegroup, a, b, c, alpha, beta, gamma), file=f
            )
        elif dimensionality == "SLAB":
            print("SLAB", file=f)
            print(spacegroup, file=f)
            print(f"{a:.8f} {b:.8f} {gamma:.6f}", file=f)
        elif dimensionality == "POLYMER":
            print("POLYMER", file=f)
            print(spacegroup, file=f)
            print(f"{a:.8f}", file=f)
        elif dimensionality == "MOLECULE":
            print("MOLECULE", file=f)
            print("1", file=f)  # C1 symmetry for molecules

        # Write atomic positions
        print(str(len(atomic_numbers)), file=f)

        for i in range(len(atomic_numbers)):
            atomic_number = atomic_numbers[i]

            # Add 200 to atomic number if ECP is required
            if atomic_number in ECP_ELEMENTS:
                atomic_number += 200

            # Write with different format depending on dimensionality (increased precision)
            if dimensionality == "CRYSTAL":
                print(
                    f"{atomic_number} {positions[i][0]:.10f} {positions[i][1]:.10f} {positions[i][2]:.10f} Biso 1.000000 {symbols[i]}",
                    file=f,
                )
            elif dimensionality == "SLAB":
                print(
                    f"{atomic_number} {positions[i][0]:.10f} {positions[i][1]:.10f} {positions[i][2]:.10f} Biso 1.000000 {symbols[i]}",
                    file=f,
                )
            elif dimensionality == "POLYMER":
                print(
                    f"{atomic_number} {positions[i][0]:.10f} {positions[i][1]:.10f} {positions[i][2]:.10f} Biso 1.000000 {symbols[i]}",
                    file=f,
                )
            elif dimensionality == "MOLECULE":
                print(
                    f"{atomic_number} {positions[i][0]:.10f} {positions[i][1]:.10f} {positions[i][2]:.10f} Biso 1.000000 {symbols[i]}",
                    file=f,
                )

        # Write calculation-type specific parameters
        if calculation_type == "SP":
            # For single point, just end the geometry block
            print("END", file=f)
        elif calculation_type == "OPT":
            # For geometry optimization
            print("OPTGEOM", file=f)

            if optimization_type:
                print(optimization_type, file=f)

            print("MAXCYCLE", file=f)
            print(optimization_settings["MAXCYCLE"], file=f)
            print("TOLDEG", file=f)
            print(format_crystal_float(optimization_settings["TOLDEG"]), file=f)
            print("TOLDEX", file=f)
            print(format_crystal_float(optimization_settings["TOLDEX"]), file=f)
            print("TOLDEE", file=f)
            print(optimization_settings["TOLDEE"], file=f)

            # Add MAXTRADIUS if specified
            if "MAXTRADIUS" in optimization_settings:
                print("MAXTRADIUS", file=f)
                print(format_crystal_float(optimization_settings["MAXTRADIUS"]), file=f)

            print("ENDOPT", file=f)
            print("END", file=f)
        elif calculation_type == "FREQ":
            # For frequency calculation
            print("FREQCALC", file=f)
            print("NUMDERIV", file=f)
            print(freq_settings["NUMDERIV"], file=f)
            print("END", file=f)  # End of FREQCALC block
            print("END", file=f)  # End of geometry block

        # Write basis set
        if basis_set_type == "EXTERNAL":
            # Get unique elements
            unique_atoms = unique_elements(atomic_numbers)

            # Include basis sets for each unique element
            for atomic_number in unique_atoms:
                basis_content = read_basis_file(basis_set, atomic_number)
                print(basis_content, end="", file=f)

            # Only add 99 0 line for external basis sets
            print("99 0", file=f)
            print("END", file=f)
        else:  # Internal basis set
            print(f"BASISSET", file=f)
            print(f"{basis_set}", file=f)
            print("END", file=f)

        # Write DFT section
        print("DFT", file=f)

        if is_spin_polarized:
            print("SPIN", file=f)

        print(functional, file=f)

        # Add DFT grid size only if not default
        if dft_grid != "DEFAULT":
            print(dft_grid, file=f)

        print("ENDDFT", file=f)

        # Write SCF parameters
        # Tolerance settings
        print("TOLINTEG", file=f)
        print(tolerances["TOLINTEG"], file=f)

        print("TOLDEE", file=f)
        print(tolerances["TOLDEE"], file=f)

        # Shrinking factors for k-point sampling
        ka, kb, kc = generate_k_points(a, b, c, dimensionality, spacegroup)
        n_shrink = max(ka, kb, kc) * 2

        print("SHRINK", file=f)
        print(f"0 {n_shrink}", file=f)

        if dimensionality == "CRYSTAL":
            print(f"{ka} {kb} {kc}", file=f)
        elif dimensionality == "SLAB":
            print(f"{ka} {kb} 1", file=f)
        elif dimensionality == "POLYMER":
            print(f"{ka} 1 1", file=f)
        elif dimensionality == "MOLECULE":
            print("0 0 0", file=f)  # No k-points for molecule

        # Add Fermi surface smearing for metallic systems if requested
        if use_smearing:
            print("SMEAR", file=f)
            print(f"{smearing_width:.6f}", file=f)

        # SCF settings
        print("SCFDIR", file=f)  # Use SCF direct algorithm

        # Add BIPOSIZE and EXCHSIZE for large systems
        if len(atomic_numbers) > 5:
            print("BIPOSIZE", file=f)
            print("110000000", file=f)
            print("EXCHSIZE", file=f)
            print("110000000", file=f)

        # Maximum SCF cycles
        print("MAXCYCLE", file=f)
        print(scf_maxcycle, file=f)  # Use the specified SCF MAXCYCLE

        # Fock/KS matrix mixing
        print("FMIXING", file=f)
        print(fmixing, file=f)

        # SCF convergence method
        print(scf_method, file=f)

        if scf_method == "DIIS":
            print("HISTDIIS", file=f)
            print("100", file=f)

        # Print options
        print("PPAN", file=f)  # Print Mulliken population analysis

        # End of input deck
        print("END", file=f)


def process_cifs(cif_directory, options, output_directory=None):
    """
    Process all CIF files in a directory

    Args:
        cif_directory (str): Directory containing CIF files
        options (dict): Calculation options
        output_directory (str, optional): Output directory for D12 files

    Returns:
        None
    """
    if output_directory is None:
        output_directory = cif_directory

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Find all CIF files
    cif_files = glob.glob(os.path.join(cif_directory, "*.cif"))

    if not cif_files:
        print(f"No CIF files found in {cif_directory}")
        return

    print(f"Found {len(cif_files)} CIF files to process")

    # Process each CIF file
    for cif_file in cif_files:
        base_name = os.path.basename(cif_file).replace(".cif", "")

        # Generate output filename
        dimensionality = options["dimensionality"]
        calc_type = options["calculation_type"]
        functional = options["dft_functional"]
        if options["use_dispersion"]:
            functional += "-D3"

        symmetry_tag = "P1" if options["symmetry_handling"] == "P1" else "symm"

        if options["basis_set_type"] == "EXTERNAL":
            basis_name = os.path.basename(options["basis_set"].rstrip("/"))
        else:
            basis_name = options["basis_set"]

        output_name = f"{base_name}_{dimensionality}_{calc_type}_{symmetry_tag}_{functional}_{basis_name}.d12"
        output_file = os.path.join(output_directory, output_name)

        try:
            print(f"Processing {cif_file}...")

            # Parse CIF file
            cif_data = parse_cif(cif_file)

            # Apply symmetry handling
            if options["symmetry_handling"] == "P1":
                # If P1 symmetry requested, override the spacegroup
                cif_data["spacegroup"] = 1
                print("Using P1 symmetry (no symmetry operations, all atoms explicit)")
            elif options["symmetry_handling"] == "FULL":
                # If rhombohedral space group, handle setting conversions
                if is_rhombohedral(cif_data["spacegroup"]):
                    # Detect current setting
                    current_setting = detect_cell_setting(cif_data)
                    print(
                        f"Detected rhombohedral structure in {current_setting} setting"
                    )

                    # Apply setting conversion if needed
                    if "rhomb_setting" in options:
                        target_setting = options["rhomb_setting"]

                        if target_setting == "AUTO":
                            # Keep as is
                            print(
                                f"Using detected {current_setting} setting as specified by AUTO option"
                            )
                        elif (
                            target_setting == "HEXAGONAL"
                            and current_setting == "rhombohedral"
                        ):
                            # Convert from rhombohedral to hexagonal
                            print("Converting from rhombohedral to hexagonal setting")
                            a_hex, c_hex, new_positions = rhomb_to_hex(
                                cif_data["a"], cif_data["alpha"], cif_data["positions"]
                            )

                            # Update cif_data
                            cif_data["a"] = a_hex
                            cif_data["b"] = a_hex
                            cif_data["c"] = c_hex
                            cif_data["alpha"] = 90.0
                            cif_data["beta"] = 90.0
                            cif_data["gamma"] = 120.0
                            cif_data["positions"] = new_positions

                        elif (
                            target_setting == "RHOMBOHEDRAL"
                            and current_setting == "hexagonal"
                        ):
                            # Convert from hexagonal to rhombohedral
                            print("Converting from hexagonal to rhombohedral setting")
                            a_rhomb, alpha_rhomb, new_positions = hex_to_rhomb(
                                cif_data["a"], cif_data["c"], cif_data["positions"]
                            )

                            # Update cif_data
                            cif_data["a"] = a_rhomb
                            cif_data["b"] = a_rhomb
                            cif_data["c"] = a_rhomb
                            cif_data["alpha"] = alpha_rhomb
                            cif_data["beta"] = alpha_rhomb
                            cif_data["gamma"] = alpha_rhomb
                            cif_data["positions"] = new_positions

                # If full symmetry requested and spglib is available, try to reduce to asymmetric unit
                if SPGLIB_AVAILABLE and options.get("reduce_to_asymmetric", True):
                    cif_data = reduce_to_asymmetric_unit(cif_data)

            # Create D12 file
            create_d12_file(cif_data, output_file, options)

            print(f"Created {output_file}")

        except Exception as e:
            print(f"Error processing {cif_file}: {e}")
            continue


def hex_to_rhomb(a, c, positions):
    """
    Convert hexagonal cell parameters and positions to rhombohedral

    Args:
        a (float): Hexagonal a parameter
        c (float): Hexagonal c parameter
        positions (list): Atomic positions in hexagonal setting

    Returns:
        tuple: (a_rhomb, alpha_rhomb, new_positions)
    """
    import math

    # Calculate rhombohedral cell parameter
    a_rhomb = math.sqrt((a**2 / 3.0) + (c**2 / 9.0))

    # Calculate rhombohedral angle
    cos_alpha = (c**2 - 2 * a**2) / (c**2 + 4 * a**2)
    alpha_rhomb = math.degrees(math.acos(cos_alpha))

    # Transformation matrix from hex to rhomb
    T = np.array(
        [[2 / 3, -1 / 3, -1 / 3], [1 / 3, 1 / 3, -2 / 3], [1 / 3, 1 / 3, 1 / 3]]
    )

    # Transform positions
    new_positions = []
    for pos in positions:
        new_pos = np.dot(T, np.array(pos))
        # Ensure coordinates are within [0,1)
        new_pos = new_pos % 1.0
        new_positions.append(new_pos.tolist())

    return a_rhomb, alpha_rhomb, new_positions


def rhomb_to_hex(a_rhomb, alpha_rhomb, positions):
    """
    Convert rhombohedral cell parameters and positions to hexagonal

    Args:
        a_rhomb (float): Rhombohedral a parameter
        alpha_rhomb (float): Rhombohedral alpha angle
        positions (list): Atomic positions in rhombohedral setting

    Returns:
        tuple: (a_hex, c_hex, new_positions)
    """
    import math

    # Convert angle to radians
    alpha_rad = math.radians(alpha_rhomb)

    # Calculate hexagonal cell parameters
    a_hex = a_rhomb * math.sqrt(2 - 2 * math.cos(alpha_rad))
    c_hex = a_rhomb * math.sqrt(3 * (1 + 2 * math.cos(alpha_rad)))

    # Transformation matrix from rhomb to hex
    T = np.array([[1, 0, 1], [-1, 1, 1], [0, -1, 1]]) * (1 / 3)

    # Transform positions
    new_positions = []
    for pos in positions:
        new_pos = np.dot(T, np.array(pos))
        # Ensure coordinates are within [0,1)
        new_pos = new_pos % 1.0
        new_positions.append(new_pos.tolist())

    return a_hex, c_hex, new_positions


def is_trigonal(spacegroup):
    """
    Check if space group is trigonal

    Args:
        spacegroup (int): Space group number

    Returns:
        bool: True if trigonal, False otherwise
    """
    return 143 <= spacegroup <= 167


def is_hexagonal(spacegroup):
    """
    Check if space group is hexagonal

    Args:
        spacegroup (int): Space group number

    Returns:
        bool: True if hexagonal, False otherwise
    """
    return 168 <= spacegroup <= 194


def is_cubic(spacegroup):
    """
    Check if space group is cubic

    Args:
        spacegroup (int): Space group number

    Returns:
        bool: True if cubic, False otherwise
    """
    return 195 <= spacegroup <= 230


def detect_trigonal_setting(cif_data):
    """
    Detect whether a trigonal structure is in hexagonal or rhombohedral axes

    Args:
        cif_data (dict): Parsed CIF data

    Returns:
        str: 'hexagonal_axes' or 'rhombohedral_axes'
    """
    # Check if it's a trigonal space group
    if 143 <= cif_data["spacegroup"] <= 167:
        # Determine which setting based on cell parameters
        if (
            abs(cif_data["alpha"] - 90) < 1e-3
            and abs(cif_data["beta"] - 90) < 1e-3
            and abs(cif_data["gamma"] - 120) < 1e-3
        ):
            # Alpha ~ 90, beta ~ 90, gamma ~ 120 indicates hexagonal axes
            return "hexagonal_axes"
        elif (
            abs(cif_data["alpha"] - cif_data["beta"]) < 1e-3
            and abs(cif_data["beta"] - cif_data["gamma"]) < 1e-3
        ):
            # Alpha = beta = gamma != 90 indicates rhombohedral axes
            return "rhombohedral_axes"

    # Default to hexagonal axes
    return "hexagonal_axes"


def reduce_to_asymmetric_unit(cif_data):
    """
    Reduce the structure to its asymmetric unit using spglib if available

    Args:
        cif_data (dict): Parsed CIF data

    Returns:
        dict: Modified CIF data with only asymmetric unit atoms
    """
    if not SPGLIB_AVAILABLE:
        print("Warning: spglib not available, cannot reduce to asymmetric unit.")
        print("Using all atoms from the CIF file.")
        return cif_data

    try:
        # Create a cell structure for spglib
        lattice = [[cif_data["a"], 0, 0], [0, cif_data["b"], 0], [0, 0, cif_data["c"]]]

        # If non-orthogonal cell, need to convert to cartesian
        if cif_data["alpha"] != 90 or cif_data["beta"] != 90 or cif_data["gamma"] != 90:
            # This is a simplified approach; a proper conversion would be more complex
            print(
                "Warning: Non-orthogonal cell detected. Symmetry reduction may not be accurate."
            )

        positions = cif_data["positions"]
        numbers = cif_data["atomic_numbers"]

        cell = (lattice, positions, numbers)

        # Get spacegroup data
        spacegroup = spglib.get_spacegroup(cell, symprec=1e-5)
        print(f"Detected space group: {spacegroup}")

        # Get symmetrized cell with dataset
        dataset = spglib.get_symmetry_dataset(cell, symprec=1e-5)

        # Get unique atoms (asymmetric unit)
        unique_indices = []
        for i in range(len(numbers)):
            if i in [dataset["equivalent_atoms"][i] for i in range(len(numbers))]:
                unique_indices.append(i)

        # Create new cif_data with only asymmetric unit atoms
        new_cif_data = cif_data.copy()
        new_cif_data["atomic_numbers"] = [numbers[i] for i in unique_indices]
        new_cif_data["symbols"] = [cif_data["symbols"][i] for i in unique_indices]
        new_cif_data["positions"] = [positions[i] for i in unique_indices]

        print(
            f"Reduced from {len(numbers)} atoms to {len(new_cif_data['atomic_numbers'])} atoms in asymmetric unit."
        )
        return new_cif_data

    except Exception as e:
        print(f"Error during symmetry reduction: {e}")
        print("Using all atoms from the CIF file.")
        return cif_data


def print_summary(options):
    """Print a summary of the selected options"""
    print("\n--- Selected Options Summary ---")
    for key, value in options.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for subkey, subvalue in value.items():
                print(f"  - {subkey}: {subvalue}")
        else:
            print(f"{key}: {value}")
    print("-------------------------------\n")


def main():
    """Main function to run the script"""
    parser = argparse.ArgumentParser(
        description="Convert CIF files to D12 input files for CRYSTAL23"
    )
    parser.add_argument(
        "--cif_dir", type=str, default="./", help="Directory containing CIF files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for D12 files (default: same as CIF directory)",
    )
    parser.add_argument(
        "--batch", action="store_true", help="Run in batch mode using saved options"
    )
    parser.add_argument(
        "--save_options",
        action="store_true",
        help="Save options to file for batch mode",
    )
    parser.add_argument(
        "--options_file",
        type=str,
        default="cif2d12_options.txt",
        help="File to save/load options for batch mode",
    )

    args = parser.parse_args()

    if args.batch:
        # Load options from file
        try:
            import json

            with open(args.options_file, "r") as f:
                options = json.load(f)
            print(f"Loaded options from {args.options_file}")
            print_summary(options)
        except Exception as e:
            print(f"Error loading options from {args.options_file}: {e}")
            print("Please run the script without --batch to create options file first")
            return
    else:
        # Get options interactively
        print("CIF to D12 Converter for CRYSTAL23")
        print("==================================")
        options = get_calculation_options()
        print_summary(options)

        # Save options if requested
        if args.save_options:
            try:
                import json

                with open(args.options_file, "w") as f:
                    json.dump(options, f, indent=2)
                print(f"Saved options to {args.options_file}")
            except Exception as e:
                print(f"Error saving options to {args.options_file}: {e}")

    # Process CIF files
    process_cifs(args.cif_dir, options, args.output_dir)


if __name__ == "__main__":
    main()
