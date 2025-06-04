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

    DEFAULT_DZ_PATH = "./full.basis.doublezeta/"  # Path to DZVP-REV2 external basis set
    DEFAULT_TZ_PATH = "./full.basis.triplezeta/"  # Path to TZVP-REV2 external basis set

    Update these paths to point to your basis set directories on your system.

AUTHOR:
    Original script by Marcus Djokic
    Enhanced with comprehensive features by Marcus Djokic with AI assistance
"""

import os
import sys
import glob
import argparse
import numpy as np
from ase.io import read
import json

# Path constants for external basis sets - MODIFY THESE TO MATCH YOUR SYSTEM
# Note: These are the REV2 versions of the basis sets
DEFAULT_DZ_PATH = "./full.basis.doublezeta/"  # DZVP-REV2 external basis set directory
DEFAULT_TZ_PATH = "./full.basis.triplezeta/"  # TZVP-REV2 external basis set directory

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
    "Tl": 81,
    "Pb": 82,
    "Bi": 83,
    "Po": 84,
    "At": 85,
    "Rn": 86,
    "Fr": 87,
    "Ra": 88,
    "Ac": 89,
    "Th": 90,
    "Pa": 91,
    "U": 92,
    "Np": 93,
    "Pu": 94,
    "Am": 95,
    "Cm": 96,
    "Bk": 97,
    "Cf": 98,
    "Es": 99,
    "Fm": 100,
}

# Reverse mapping for element symbols
ATOMIC_NUMBER_TO_SYMBOL = {v: k for k, v in ELEMENT_SYMBOLS.items()}

# Space groups with multiple origin settings that need special handling
MULTI_ORIGIN_SPACEGROUPS = {
    216: {"name": "F-43m", "default": "Origin 2 (ITA)", "crystal_code": "0 0 0"},
    218: {"name": "P-43n", "default": "Origin 2 (ITA)", "crystal_code": "0 0 0"},
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

# Elements that require ECPs in external basis sets (DZVP-REV2 and TZVP-REV2)
ECP_ELEMENTS_EXTERNAL = [
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
    81,
    82,
    83,
    89,
    90,
    91,
    92,
    93,
    94,
    95,
    96,
    97,
    98,
    99,
]

# Note: Noble gases (He, Ne, Ar, Kr, Xe, Rn) are missing from ECP list
# Note: Element 43 (Tc) has full core available

# Available internal basis sets with element ranges and core types
INTERNAL_BASIS_SETS = {
    # Standard basis sets (original 7)
    "STO-3G": {
        "description": "Pople's standard minimal basis set (3 Gaussian function contractions)",
        "elements": list(range(1, 54)),  # H to I
        "all_electron": list(range(1, 54)),
        "ecp_elements": [],
        "standard": True,
    },
    "STO-6G": {
        "description": "Pople's standard minimal basis set (6 Gaussian function contractions)",
        "elements": list(range(1, 37)),  # H to Kr
        "all_electron": list(range(1, 37)),
        "ecp_elements": [],
        "standard": True,
    },
    "POB-DZVP": {
        "description": "POB Double-ζ + polarization basis set",
        "elements": list(range(1, 36)) + [49, 74],  # H to Br, In, W
        "all_electron": list(range(1, 19)),  # H to Ar
        "ecp_elements": list(range(19, 36)) + [49, 74],  # K onwards need ECP
        "standard": True,
    },
    "POB-DZVPP": {
        "description": "POB Double-ζ + double set of polarization functions",
        "elements": list(range(1, 36)) + [49, 83],  # H to Br, In, Bi
        "all_electron": list(range(1, 19)),  # H to Ar
        "ecp_elements": list(range(19, 36)) + [49, 83],  # K onwards need ECP
        "standard": True,
    },
    "POB-TZVP": {
        "description": "POB Triple-ζ + polarization basis set",
        "elements": list(range(1, 36)) + [49, 83],  # H to Br, In, Bi
        "all_electron": list(range(1, 19)),  # H to Ar
        "ecp_elements": list(range(19, 36)) + [49, 83],  # K onwards need ECP
        "standard": True,
    },
    "POB-DZVP-REV2": {
        "description": "POB-REV2 Double-ζ + polarization basis set",
        "elements": list(range(1, 36)),  # H to Br
        "all_electron": list(range(1, 19)),  # H to Ar
        "ecp_elements": list(range(19, 36)),  # K onwards need ECP
        "standard": True,
    },
    "POB-TZVP-REV2": {
        "description": "POB-REV2 Triple-ζ + polarization basis set",
        "elements": list(range(1, 36))
        + list(range(37, 54))
        + [55, 56]
        + list(range(72, 85)),  # H-Br, Rb-I, Cs, Ba, Hf-At
        "all_electron": list(range(1, 19)),  # H to Ar
        "ecp_elements": list(range(19, 36))
        + list(range(37, 54))
        + [55, 56]
        + list(range(72, 85)),  # K onwards need ECP
        "standard": True,
    },
    # Additional basis sets
    "MINIS": {
        "description": "Minimal basis set; primarily for testing and preliminary calculations",
        "elements": list(range(1, 37)),  # H to Kr
        "all_electron": list(range(1, 37)),
        "ecp_elements": [],
        "standard": False,
    },
    "6-31G*": {
        "description": "Split-valence double-zeta with polarization",
        "elements": list(range(1, 31)),  # H to Zn
        "all_electron": list(range(1, 31)),
        "ecp_elements": [],
        "standard": False,
    },
    "def2-SV(P)": {
        "description": "Split-valence with polarization on heavy atoms",
        "elements": list(range(1, 87)),  # H to Rn
        "all_electron": list(range(1, 37)),  # H to Kr
        "ecp_elements": list(range(37, 87)),  # Rb onwards need def2-ECP
        "standard": False,
    },
    "def2-SVP": {
        "description": "Split-valence with polarization; widely used",
        "elements": list(range(1, 87)),  # H to Rn
        "all_electron": list(range(1, 37)),  # H to Kr
        "ecp_elements": list(range(37, 87)),  # Rb onwards need def2-ECP
        "standard": False,
    },
    "def-TZVP": {
        "description": "Triple-zeta valence with polarization",
        "elements": list(range(1, 87)),  # H to Rn
        "all_electron": list(range(1, 37)),  # H to Kr
        "ecp_elements": list(range(37, 87)),  # Rb onwards need def2-ECP
        "standard": False,
    },
    "def2-TZVP": {
        "description": "Enhanced triple-zeta valence with polarization",
        "elements": list(range(1, 87)),  # H to Rn
        "all_electron": list(range(1, 37)),  # H to Kr
        "ecp_elements": list(range(37, 87)),  # Rb onwards need def2-ECP
        "standard": False,
    },
}

# Functional categories with complete descriptions
FUNCTIONAL_CATEGORIES = {
    "HF": {
        "name": "Hartree-Fock Methods",
        "description": "Wave function based methods (no DFT)",
        "functionals": ["RHF", "UHF", "HF-3C", "HFsol-3C"],
        "basis_requirements": {"HF-3C": "MINIX", "HFsol-3C": "SOLMINIX"},
        "descriptions": {
            "RHF": "Restricted Hartree-Fock (closed shell)",
            "UHF": "Unrestricted Hartree-Fock (open shell)",
            "HF-3C": "Minimal basis HF with D3, gCP, and SRB corrections",
            "HFsol-3C": "HF-3C revised for inorganic solids",
        },
    },
    "3C": {
        "name": "3c Composite Methods (DFT)",
        "description": "DFT composite methods with semi-classical corrections (require specific basis sets)",
        "functionals": [
            # Molecular crystal oriented
            "PBEh-3C",
            "HSE-3C",
            "B97-3C",
            # Solid state oriented
            "PBEsol0-3C",
            "HSEsol-3C",
        ],
        "basis_requirements": {
            "PBEh-3C": "def2-mSVP",
            "HSE-3C": "def2-mSVP",
            "B97-3C": "mTZVP",
            "PBEsol0-3C": "sol-def2-mSVP",
            "HSEsol-3C": "sol-def2-mSVP",
        },
        "descriptions": {
            "PBEh-3C": "Modified PBE hybrid (42% HF) with D3 and gCP",
            "HSE-3C": "Screened exchange hybrid optimized for molecular solids",
            "B97-3C": "GGA functional with D3 and SRB corrections",
            "PBEsol0-3C": "PBEsol0 hybrid for solids with D3 and gCP",
            "HSEsol-3C": "HSEsol with semi-classical corrections for solids",
        },
    },
    "MGGA": {
        "name": "meta-GGA Functionals",
        "description": "Functionals that depend on kinetic energy density",
        "functionals": [
            # SCAN family
            "SCAN",
            "r2SCAN",
            "SCAN0",
            "r2SCANh",
            "r2SCAN0",
            "r2SCAN50",
            # Minnesota functionals
            "M05",
            "M052X",
            "M06",
            "M062X",
            "M06HF",
            "M06L",
            "revM06",
            "revM06L",
            "MN15",
            "MN15L",
            # Becke95 correlation based
            "B1B95",
            "mPW1B95",
            "mPW1B1K",
            "PW6B95",
            "PWB6K",
        ],
        "descriptions": {
            # SCAN family
            "SCAN": "Strongly Constrained and Appropriately Normed",
            "r2SCAN": "Regularized SCAN with improved numerical stability",
            "SCAN0": "SCAN hybrid (25% HF)",
            "r2SCANh": "r2SCAN hybrid (10% HF)",
            "r2SCAN0": "r2SCAN hybrid (25% HF)",
            "r2SCAN50": "r2SCAN hybrid (50% HF)",
            # Minnesota functionals
            "M05": "Minnesota 2005 hybrid (28% HF)",
            "M052X": "M05 with doubled HF exchange (56% HF)",
            "M06": "Minnesota 2006 hybrid (27% HF)",
            "M062X": "M06 with doubled HF exchange (54% HF)",
            "M06HF": "Full HF exchange meta-GGA (100% HF)",
            "M06L": "Local meta-GGA for main-group thermochemistry",
            "revM06": "Revised M06 (40.41% HF)",
            "revM06L": "Revised M06L with improved performance",
            "MN15": "Minnesota 2015 hybrid (44% HF)",
            "MN15L": "Minnesota 2015 local functional",
            # Becke95 based
            "B1B95": "One-parameter hybrid with Becke95 correlation (28% HF)",
            "mPW1B95": "Modified PW91 with B95 correlation (31% HF)",
            "mPW1B1K": "Modified PW91 with B95 correlation (44% HF)",
            "PW6B95": "6-parameter functional (28% HF)",
            "PWB6K": "6-parameter functional for kinetics (46% HF)",
        },
    },
    "HYBRID": {
        "name": "Hybrid Functionals (including range-separated)",
        "description": "Global and range-separated hybrid functionals",
        "functionals": [
            # B3 family
            "B3LYP",
            "B3PW",
            "CAM-B3LYP",
            # PBE family
            "PBE0",
            "PBESOL0",
            "PBE0-13",
            # HSE family
            "HSE06",
            "HSEsol",
            # mPW family
            "mPW1PW91",
            "mPW1K",
            # WC family
            "B1WC",
            "WC1LYP",
            # B97 family
            "B97H",
            "wB97",
            "wB97X",
            # Other global hybrids
            "SOGGA11X",
            # Short-range corrected
            "SC-BLYP",
            # Middle-range corrected
            "HISS",
            # Long-range corrected
            "RSHXLDA",
            "LC-wPBE",
            "LC-wPBEsol",
            "LC-wBLYP",
            "LC-BLYP",
            "LC-PBE",
        ],
        "descriptions": {
            # B3 family
            "B3LYP": "Becke 3-parameter hybrid (20% HF)",
            "B3PW": "Becke 3-parameter with PW91 correlation (20% HF)",
            "CAM-B3LYP": "Coulomb-attenuating method B3LYP",
            # PBE family
            "PBE0": "PBE hybrid (25% HF)",
            "PBESOL0": "PBEsol hybrid for solids (25% HF)",
            "PBE0-13": "PBE0 with 1/3 HF exchange (33.33% HF)",
            # HSE family
            "HSE06": "Heyd-Scuseria-Ernzerhof screened hybrid",
            "HSEsol": "HSE for solids",
            # mPW family
            "mPW1PW91": "Modified PW91 hybrid (25% HF)",
            "mPW1K": "Modified PW91 for kinetics (42.8% HF)",
            # WC family
            "B1WC": "One-parameter WC hybrid (16% HF)",
            "WC1LYP": "WC exchange with LYP correlation (16% HF)",
            # B97 family
            "B97H": "Re-parameterized B97 hybrid",
            "wB97": "Head-Gordon's range-separated functional",
            "wB97X": "wB97 with short-range HF exchange",
            # Other
            "SOGGA11X": "Second-order GGA hybrid (40.15% HF)",
            "SC-BLYP": "Short-range corrected BLYP",
            "HISS": "Middle-range corrected functional",
            "RSHXLDA": "Long-range corrected LDA",
            "LC-wPBE": "Long-range corrected PBE",
            "LC-wPBEsol": "Long-range corrected PBEsol",
            "LC-wBLYP": "Long-range corrected BLYP",
            "LC-BLYP": "Long-range corrected BLYP (CAM-style)",
            "LC-PBE": "Long-range corrected PBE",
        },
    },
    "GGA": {
        "name": "GGA Functionals",
        "description": "Generalized Gradient Approximation functionals",
        "functionals": [
            # Becke/LYP
            "BLYP",
            # PBE family
            "PBE",
            "PBESOL",
            # PW family
            "PWGGA",
            # Others
            "SOGGA",
            "WCGGA",
            "B97",
        ],
        "descriptions": {
            "BLYP": "Becke 88 exchange + Lee-Yang-Parr correlation",
            "PBE": "Perdew-Burke-Ernzerhof",
            "PBESOL": "PBE revised for solids",
            "PWGGA": "Perdew-Wang 1991 GGA",
            "SOGGA": "Second-order GGA",
            "WCGGA": "Wu-Cohen GGA",
            "B97": "Becke's 1997 GGA functional",
        },
    },
    "LDA": {
        "name": "LDA/LSD Functionals",
        "description": "Local (Spin) Density Approximation functionals",
        "functionals": ["SVWN", "LDA", "VBH"],
        "descriptions": {
            "SVWN": "Slater exchange + VWN5 correlation",
            "LDA": "Local Density Approximation (Dirac-Slater)",
            "VBH": "von Barth-Hedin LSD functional",
        },
    },
}

# Functionals available for D3 dispersion correction
# Note: PW1PW is the D3 notation for mPW1PW91
D3_FUNCTIONALS = [
    "BLYP",
    "PBE",
    "B97",
    "B3LYP",
    "PBE0",
    "mPW1PW91",
    "M06",
    "HSE06",
    "HSEsol",
    "LC-wPBE",
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

# Default recommended settings
DEFAULT_SETTINGS = {
    "symmetry_handling": "CIF",
    "symmetry_tolerance": 1e-5,
    "reduce_to_asymmetric": False,
    "trigonal_axes": "AUTO",
    "origin_setting": "AUTO",
    "dimensionality": "CRYSTAL",
    "calculation_type": "OPT",
    "optimization_type": "FULLOPTG",
    "optimization_settings": DEFAULT_OPT_SETTINGS.copy(),
    "basis_set_type": "INTERNAL",
    "basis_set": "POB-TZVP-REV2",
    "method": "DFT",
    "dft_functional": "HSE06",
    "use_dispersion": True,
    "dft_grid": "XLGRID",
    "is_spin_polarized": True,
    "use_smearing": False,
    "tolerances": DEFAULT_TOLERANCES.copy(),
    "scf_method": "DIIS",
    "scf_maxcycle": 800,
    "fmixing": 30,
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


def display_default_settings():
    """Display the default settings in a formatted way"""
    print("\n" + "=" * 70)
    print("DEFAULT RECOMMENDED SETTINGS FOR CRYSTAL23")
    print("=" * 70)

    print("\n### SYMMETRY SETTINGS ###")
    print(f"Symmetry handling: CIF (Use symmetry as defined in the CIF file)")
    print(f"Trigonal axes: AUTO (Use setting as detected in CIF)")
    print(f"High symmetry space groups: AUTO (Use origin as detected in CIF)")

    print("\n### CALCULATION SETTINGS ###")
    print(f"Dimensionality: CRYSTAL (3D periodic system)")
    print(f"Calculation type: OPT (Geometry optimization)")
    print(f"Optimization type: FULLOPTG (Full geometry optimization)")
    print(f"Optimization parameters:")
    print(f"  - TOLDEG: {DEFAULT_OPT_SETTINGS['TOLDEG']} (RMS of gradient)")
    print(f"  - TOLDEX: {DEFAULT_OPT_SETTINGS['TOLDEX']} (RMS of displacement)")
    print(f"  - TOLDEE: {DEFAULT_OPT_SETTINGS['TOLDEE']} (Energy convergence)")
    print(f"  - MAXCYCLE: {DEFAULT_OPT_SETTINGS['MAXCYCLE']} (Max optimization steps)")
    print(f"  - MAXTRADIUS: Not set (default)")

    print("\n### BASIS SET AND DFT SETTINGS ###")
    print(f"Method: DFT")
    print(f"Basis set type: INTERNAL")
    print(f"Basis set: POB-TZVP-REV2 (Triple-ζ + polarization, revised)")
    print(f"DFT functional: HSE06-D3 (Screened hybrid with D3 dispersion)")
    print(f"DFT grid: XLGRID (Extra large grid)")
    print(f"Spin polarized: Yes")
    print(f"Fermi smearing: No")

    print("\n### SCF SETTINGS ###")
    print(
        f"Tolerances: TOLINTEG={DEFAULT_TOLERANCES['TOLINTEG']}, TOLDEE={DEFAULT_TOLERANCES['TOLDEE']}"
    )
    print(f"SCF method: DIIS")
    print(f"SCF max cycles: 800")
    print(f"FMIXING: 30%")

    print("\n### EXAMPLE D12 OUTPUT ###")
    print("-" * 70)
    print("EXAMPLE_STRUCTURE")
    print("CRYSTAL")
    print("0 0 0")
    print("225")
    print("5.46500 #a=b=c cubic alpha = beta = gamma = 90")
    print("2")
    print("11 0.0000000000 0.0000000000 0.0000000000 Biso 1.000000 Na")
    print("17 0.5000000000 0.5000000000 0.5000000000 Biso 1.000000 Cl")
    print("OPTGEOM")
    print("FULLOPTG")
    print("MAXCYCLE")
    print("800")
    print("TOLDEG")
    print("0.00003")
    print("TOLDEX")
    print("0.00012")
    print("TOLDEE")
    print("7")
    print("ENDOPT")
    print("END")
    print("BASISSET")
    print("POB-TZVP-REV2")
    print("END")
    print("DFT")
    print("SPIN")
    print("HSE06-D3")
    print("XLGRID")
    print("ENDDFT")
    print("TOLINTEG")
    print("7 7 7 7 14")
    print("TOLDEE")
    print("7")
    print("SHRINK")
    print("0 24")
    print("12 12 12")
    print("SCFDIR")
    print("BIPOSIZE")
    print("110000000")
    print("EXCHSIZE")
    print("110000000")
    print("MAXCYCLE")
    print("800")
    print("FMIXING")
    print("30")
    print("DIIS")
    print("HISTDIIS")
    print("100")
    print("PPAN")
    print("END")
    print("-" * 70)


def select_method():
    """
    Select calculation method (HF or DFT)

    Returns:
        str: 'HF' or 'DFT'
    """
    method_options = {"1": "DFT", "2": "HF"}

    print("\nSelect calculation method:")
    print("1: DFT - Density Functional Theory")
    print("2: HF - Hartree-Fock")

    method_choice = get_user_input("Select method", method_options, "1")
    return method_options[method_choice]


def select_functional(method):
    """
    Select functional/method by category

    Args:
        method (str): 'HF' or 'DFT'

    Returns:
        tuple: (functional, basis_set_requirement) or (functional, None)
    """
    if method == "HF":
        # For HF methods, directly show the options
        hf_info = FUNCTIONAL_CATEGORIES["HF"]
        functional_options = {
            str(i + 1): func for i, func in enumerate(hf_info["functionals"])
        }

        print(f"\nAvailable {hf_info['name']}:")
        for key, func in functional_options.items():
            desc = hf_info["descriptions"].get(func, "")
            if func in ["HF-3C", "HFsol-3C"]:
                basis = hf_info["basis_requirements"][func]
                print(f"{key}: {func} - {desc} (requires {basis} basis set)")
            else:
                print(f"{key}: {func} - {desc}")

        functional_choice = get_user_input("Select HF method", functional_options, "1")
        selected_functional = functional_options[functional_choice]

        # Check if it has basis set requirement
        if selected_functional in hf_info.get("basis_requirements", {}):
            required_basis = hf_info["basis_requirements"][selected_functional]
            return selected_functional, required_basis

        return selected_functional, None

    else:  # DFT
        # First, select DFT category
        dft_categories = {k: v for k, v in FUNCTIONAL_CATEGORIES.items() if k != "HF"}
        category_options = {}

        for i, (key, info) in enumerate(dft_categories.items(), 1):
            category_options[str(i)] = key
            print(f"\n{i}. {info['name']}")
            print(f"   {info['description']}")
            # Show appropriate examples for each category
            if key == "HYBRID":
                print(f"   Examples: B3LYP, PBE0, HSE06, LC-wPBE")
            elif key == "3C":
                print(f"   Examples: PBEh-3C, HSE-3C, B97-3C")
            else:
                print(f"   Examples: {', '.join(info['functionals'][:4])}")

        category_choice = get_user_input(
            "Select functional category", category_options, "2"
        )  # Default to HYBRID
        selected_category = category_options[category_choice]

        # Then select specific functional
        category_info = dft_categories[selected_category]
        functional_options = {
            str(i + 1): func for i, func in enumerate(category_info["functionals"])
        }

        print(f"\nAvailable {category_info['name']}:")
        for key, func in functional_options.items():
            # Build the description string
            desc_parts = []

            # Add functional description if available
            if (
                "descriptions" in category_info
                and func in category_info["descriptions"]
            ):
                desc_parts.append(category_info["descriptions"][func])

            # Add D3 support indicator
            if func in D3_FUNCTIONALS:
                desc_parts.append("[D3✓]")

            # Add basis requirement for 3C methods
            if selected_category == "3C" and "basis_requirements" in category_info:
                basis = category_info["basis_requirements"].get(func, "")
                desc_parts.append(f"(requires {basis})")

            # Print the functional with description
            if desc_parts:
                print(f"{key}: {func} - {' '.join(desc_parts)}")
            else:
                print(f"{key}: {func}")

        functional_choice = get_user_input(
            f"Select {category_info['name']}", functional_options, "1"
        )
        selected_functional = functional_options[functional_choice]

        # Check if functional has basis set requirement
        if selected_category == "3C" and "basis_requirements" in category_info:
            required_basis = category_info["basis_requirements"].get(
                selected_functional
            )
            return selected_functional, required_basis

        return selected_functional, None


def check_basis_set_compatibility(basis_set, atomic_numbers, basis_set_type="INTERNAL"):
    """
    Check if the selected basis set is compatible with all elements in the structure

    Args:
        basis_set (str): Name of the basis set
        atomic_numbers (list): List of atomic numbers in the structure
        basis_set_type (str): "INTERNAL" or "EXTERNAL"

    Returns:
        tuple: (is_compatible, missing_elements_list)
    """
    missing_elements = []

    if basis_set_type == "INTERNAL":
        if basis_set in INTERNAL_BASIS_SETS:
            available_elements = INTERNAL_BASIS_SETS[basis_set]["elements"]
            for atom_num in set(atomic_numbers):
                if atom_num not in available_elements:
                    missing_elements.append(atom_num)
    else:  # EXTERNAL
        # For external basis sets, check if elements need ECP
        # External basis sets go up to element 99, but some need ECP
        for atom_num in set(atomic_numbers):
            if atom_num > 99:
                missing_elements.append(atom_num)

    return len(missing_elements) == 0, missing_elements


def get_element_info_string(basis_set):
    """
    Get a string describing which elements are available for a basis set

    Args:
        basis_set (str): Name of the internal basis set

    Returns:
        str: Description of available elements and core treatment
    """
    if basis_set not in INTERNAL_BASIS_SETS:
        return ""

    bs_info = INTERNAL_BASIS_SETS[basis_set]
    elements = bs_info["elements"]
    all_electron = bs_info.get("all_electron", [])
    ecp_elements = bs_info.get("ecp_elements", [])

    # Create element range descriptions
    def get_range_string(elem_list):
        if not elem_list:
            return ""

        ranges = []
        start = elem_list[0]
        end = elem_list[0]

        for i in range(1, len(elem_list)):
            if elem_list[i] == end + 1:
                end = elem_list[i]
            else:
                if start == end:
                    ranges.append(f"{ATOMIC_NUMBER_TO_SYMBOL.get(start, start)}")
                else:
                    ranges.append(
                        f"{ATOMIC_NUMBER_TO_SYMBOL.get(start, start)}-{ATOMIC_NUMBER_TO_SYMBOL.get(end, end)}"
                    )
                start = end = elem_list[i]

        # Add the last range
        if start == end:
            ranges.append(f"{ATOMIC_NUMBER_TO_SYMBOL.get(start, start)}")
        else:
            ranges.append(
                f"{ATOMIC_NUMBER_TO_SYMBOL.get(start, start)}-{ATOMIC_NUMBER_TO_SYMBOL.get(end, end)}"
            )

        return ", ".join(ranges)

    # Build description
    elem_str = get_range_string(elements)

    # Add core treatment info
    if not ecp_elements:
        core_str = "All-electron"
    elif not all_electron:
        core_str = "ECP only"
    else:
        ae_str = get_range_string(all_electron)
        ecp_str = get_range_string(ecp_elements)
        core_str = f"All-electron ({ae_str}), ECP ({ecp_str})"

    return f"Elements: {elem_str} | Core: {core_str}"


def get_calculation_options():
    """Gather calculation options from user"""
    options = {}

    # First, show default settings and ask if user wants to use them
    display_default_settings()
    use_defaults = yes_no_prompt("\nDo you want to use these default settings?", "yes")

    if use_defaults:
        # Use all default settings
        options = DEFAULT_SETTINGS.copy()
        options["optimization_settings"] = DEFAULT_OPT_SETTINGS.copy()
        options["tolerances"] = DEFAULT_TOLERANCES.copy()
    else:
        # Custom settings - go through each option

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

        # For space groups with multiple origins
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

        # Get calculation type
        calc_options = {
            "1": "SP",  # Single Point
            "2": "OPT",  # Geometry Optimization
            "3": "FREQ",  # Frequency Calculation
        }
        calc_choice = get_user_input("Select calculation type", calc_options, "2")
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
                    input("Enter TOLDEG (RMS of gradient, default 0.00003): ")
                    or 0.00003
                )
                custom_settings["TOLDEX"] = float(
                    input("Enter TOLDEX (RMS of displacement, default 0.00012): ")
                    or 0.00012
                )
                custom_settings["TOLDEE"] = int(
                    input("Enter TOLDEE (energy difference exponent, default 7): ") or 7
                )
                custom_settings["MAXCYCLE"] = int(
                    input("Enter MAXCYCLE (max optimization steps, default 800): ")
                    or 800
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
                    input("Enter NUMDERIV (numerical derivative level, default 2): ")
                    or 2
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

        # Select method (HF or DFT)
        options["method"] = select_method()

        # Select functional/method (before basis set)
        functional, required_basis = select_functional(options["method"])

        if options["method"] == "HF":
            options["hf_method"] = functional
        else:
            options["dft_functional"] = functional

            # Check if dispersion correction is available immediately after functional selection
            if functional in D3_FUNCTIONALS:
                use_dispersion = yes_no_prompt(
                    f"Add D3 dispersion correction to {functional}?", "yes"
                )
                options["use_dispersion"] = use_dispersion
            else:
                # Check if functional already includes dispersion (3C methods)
                if functional in [
                    "PBEh-3C",
                    "HSE-3C",
                    "B97-3C",
                    "PBEsol0-3C",
                    "HSEsol-3C",
                ]:
                    print(
                        f"Note: {functional} already includes dispersion corrections."
                    )
                    options["use_dispersion"] = False
                else:
                    print(
                        f"Note: D3 dispersion correction not available for {functional}"
                    )
                    options["use_dispersion"] = False

        # If functional has required basis set, use it automatically
        if required_basis:
            print(
                f"\nNote: {functional} requires {required_basis} basis set. Using it automatically."
            )
            options["basis_set_type"] = "INTERNAL"  # 3C methods use internal basis sets
            options["basis_set"] = required_basis
        else:
            # Get basis set type
            basis_options = {"1": "EXTERNAL", "2": "INTERNAL"}

            print("\nSelect basis set type:")
            print("1: EXTERNAL - Full-core and ECP basis sets (DZVP-REV2, TZVP-REV2)")
            print("   Note: Elements 37-99 (except Tc) use ECP, up to Es (Z=99)")
            print("2: INTERNAL - All-electron basis sets with limited element coverage")

            basis_choice = get_user_input("Select basis set type", basis_options, "2")
            options["basis_set_type"] = basis_options[basis_choice]

            # Get specific basis set
            if options["basis_set_type"] == "EXTERNAL":
                external_basis_options = {
                    "1": DEFAULT_DZ_PATH,  # DZVP-REV2
                    "2": DEFAULT_TZ_PATH,  # TZVP-REV2
                }

                print(f"\nSelect external basis set directory:")
                print(f"1: DZVP-REV2 ({DEFAULT_DZ_PATH})")
                print("   Full-core: H-Kr, Tc")
                print("   ECP: Rb-Es (except noble gases)")
                print(f"2: TZVP-REV2 ({DEFAULT_TZ_PATH})")
                print("   Full-core: H-Kr, Tc")
                print("   ECP: Rb-Es (except noble gases)")

                basis_dir_choice = get_user_input(
                    "Select external basis set", external_basis_options, "2"
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
                # Internal basis sets
                internal_basis_options = {}
                print("\nAvailable internal basis sets:")

                # First show standard basis sets
                print("\n--- STANDARD BASIS SETS ---")
                option_num = 1
                standard_options = []
                for bs_name, bs_info in INTERNAL_BASIS_SETS.items():
                    if bs_info.get("standard", False):
                        internal_basis_options[str(option_num)] = bs_name
                        standard_options.append(option_num)
                        element_info = get_element_info_string(bs_name)
                        print(f"{option_num}: {bs_name} - {bs_info['description']}")
                        print(f"   {element_info}")
                        option_num += 1

                # Then show additional basis sets
                print("\n--- ADDITIONAL BASIS SETS ---")
                for bs_name, bs_info in INTERNAL_BASIS_SETS.items():
                    if not bs_info.get("standard", False):
                        internal_basis_options[str(option_num)] = bs_name
                        element_info = get_element_info_string(bs_name)
                        print(f"{option_num}: {bs_name} - {bs_info['description']}")
                        print(f"   {element_info}")
                        option_num += 1

                # Default to POB-TZVP-REV2 (should be option 7 in standard sets)
                default_option = "7"
                internal_basis_choice = get_user_input(
                    "Select internal basis set", internal_basis_options, default_option
                )
                options["basis_set"] = internal_basis_options[internal_basis_choice]

        # Get DFT grid (only for DFT)
        if options["method"] == "DFT":
            grid_choice = get_user_input(
                "Select DFT integration grid", DFT_GRIDS, "4"
            )  # Default to XLGRID
            options["dft_grid"] = DFT_GRIDS[grid_choice]

        # Ask about spin polarization
        is_spin_polarized = yes_no_prompt("Use spin-polarized calculation?", "yes")
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
            "Select SCF convergence method", scf_options, "1"
        )  # Default to DIIS
        options["scf_method"] = scf_options[scf_choice]

        # Ask about SCF MAXCYCLE
        use_default_scf_maxcycle = yes_no_prompt(
            "Use default SCF MAXCYCLE (800)?", "yes"
        )

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

    # Check basis set compatibility
    is_compatible, missing_elements = check_basis_set_compatibility(
        options["basis_set"], atomic_numbers, options["basis_set_type"]
    )

    if not is_compatible:
        print(
            f"\nWARNING: The selected basis set '{options['basis_set']}' does not support all elements in your structure!"
        )
        print(
            f"Missing elements: {', '.join([f'{ATOMIC_NUMBER_TO_SYMBOL.get(z, z)} (Z={z})' for z in missing_elements])}"
        )
        if not yes_no_prompt("Continue anyway?", "no"):
            print("Aborting D12 file creation.")
            return

    # Extract options
    dimensionality = options["dimensionality"]
    calculation_type = options["calculation_type"]
    optimization_type = options.get("optimization_type", None)
    optimization_settings = options.get("optimization_settings", DEFAULT_OPT_SETTINGS)
    freq_settings = options.get("freq_settings", DEFAULT_FREQ_SETTINGS)
    basis_set_type = options["basis_set_type"]
    basis_set = options["basis_set"]
    method = options["method"]
    is_spin_polarized = options["is_spin_polarized"]
    tolerances = options["tolerances"]
    scf_method = options["scf_method"]
    scf_maxcycle = options.get("scf_maxcycle", 800)
    fmixing = options.get("fmixing", 30)
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
    if is_trigonal(spacegroup):
        trigonal_axes = options.get("trigonal_axes", "AUTO")
        if trigonal_axes == "RHOMBOHEDRAL_AXES":
            use_rhombohedral_axes = True
            print(
                f"Using rhombohedral axes (0 1 0) for trigonal space group {spacegroup}"
            )
        elif trigonal_axes == "AUTO":
            # Try to detect from CIF
            trigonal_setting = detect_trigonal_setting(cif_data)
            if trigonal_setting == "rhombohedral_axes":
                use_rhombohedral_axes = True
                print(
                    f"Detected rhombohedral axes setting for space group {spacegroup}"
                )

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
            if basis_set_type == "EXTERNAL" and atomic_number in ECP_ELEMENTS_EXTERNAL:
                atomic_number += 200
            elif basis_set_type == "INTERNAL" and basis_set in INTERNAL_BASIS_SETS:
                # Check if element needs ECP in this internal basis set
                if atomic_number in INTERNAL_BASIS_SETS[basis_set].get(
                    "ecp_elements", []
                ):
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

        # Handle HF and DFT methods differently
        if method == "HF":
            hf_method = options.get("hf_method", "RHF")

            # Handle 3C methods and basis sets
            if hf_method in ["HF-3C", "HFsol-3C"]:
                # These are HF methods with corrections, write basis set
                print(f"BASISSET", file=f)
                print(f"{basis_set}", file=f)
                print("END", file=f)

                # Add 3C corrections
                if hf_method == "HF-3C":
                    print("HF3C", file=f)
                    print("END", file=f)
                elif hf_method == "HFsol-3C":
                    print("HFSOL3C", file=f)
                    print("END", file=f)
            else:
                # Standard HF methods (RHF, UHF)
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

                # For UHF, add the UHF keyword
                if hf_method == "UHF":
                    print("UHF", file=f)

        else:  # DFT method
            dft_functional = options.get("dft_functional", "")
            use_dispersion = options.get("use_dispersion", False)
            dft_grid = options.get("dft_grid", "XLGRID")

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

            # Handle special functional keywords
            if dft_functional in [
                "PBEh-3C",
                "HSE-3C",
                "B97-3C",
                "PBEsol0-3C",
                "HSEsol-3C",
            ]:
                # These are standalone keywords in CRYSTAL23
                print(
                    f"{dft_functional.replace('-', '')}", file=f
                )  # Remove hyphen for CRYSTAL23
            elif dft_functional == "mPW1PW91" and use_dispersion:
                print("PW1PW-D3", file=f)
            else:
                # Standard functional
                if use_dispersion and dft_functional in D3_FUNCTIONALS:
                    print(f"{dft_functional}-D3", file=f)
                else:
                    print(f"{dft_functional}", file=f)

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


def is_trigonal(spacegroup):
    """
    Check if space group is trigonal

    Args:
        spacegroup (int): Space group number

    Returns:
        bool: True if trigonal, False otherwise
    """
    return 143 <= spacegroup <= 167


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

        # Method identifier
        if options["method"] == "HF":
            method_name = options.get("hf_method", "RHF")
        else:
            method_name = options.get("dft_functional", "")
            if options.get("use_dispersion"):
                method_name += "-D3"

        symmetry_tag = "P1" if options["symmetry_handling"] == "P1" else "symm"

        if options["basis_set_type"] == "EXTERNAL":
            basis_name = os.path.basename(options["basis_set"].rstrip("/"))
        else:
            basis_name = options["basis_set"]

        output_name = f"{base_name}_{dimensionality}_{calc_type}_{symmetry_tag}_{method_name}_{basis_name}.d12"
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
            elif options["symmetry_handling"] == "SPGLIB":
                # If spglib symmetry requested and reduction is enabled
                if SPGLIB_AVAILABLE and options.get("reduce_to_asymmetric", True):
                    cif_data = reduce_to_asymmetric_unit(cif_data)

            # Create D12 file
            create_d12_file(cif_data, output_file, options)

            print(f"Created {output_file}")

        except Exception as e:
            print(f"Error processing {cif_file}: {e}")
            continue


def print_summary(options):
    """Print a summary of the selected options"""
    print("\n--- Selected Options Summary ---")

    # Method and functional
    if options.get("method") == "HF":
        print(f"Method: Hartree-Fock ({options.get('hf_method', 'RHF')})")
    else:
        functional = options.get("dft_functional", "")
        if options.get("use_dispersion"):
            functional += "-D3"
        print(f"Method: DFT")
        print(f"Functional: {functional}")

    # Basic settings
    print(f"Dimensionality: {options.get('dimensionality', 'CRYSTAL')}")
    print(f"Calculation type: {options.get('calculation_type', 'SP')}")
    if options.get("calculation_type") == "OPT":
        print(f"Optimization type: {options.get('optimization_type', 'FULLOPTG')}")

    # Symmetry settings
    print(f"Symmetry handling: {options.get('symmetry_handling', 'CIF')}")
    if options.get("symmetry_handling") == "SPGLIB":
        print(f"  - Tolerance: {options.get('symmetry_tolerance', 1e-5)}")
        print(f"  - Reduce to asymmetric: {options.get('reduce_to_asymmetric', False)}")

    # Basis set
    print(f"Basis set type: {options.get('basis_set_type', 'INTERNAL')}")
    print(f"Basis set: {options.get('basis_set', 'N/A')}")

    # Additional settings
    if options.get("method") == "DFT":
        print(f"DFT grid: {options.get('dft_grid', 'XLGRID')}")
    print(f"Spin polarized: {options.get('is_spin_polarized', False)}")
    if options.get("use_smearing"):
        print(f"Fermi smearing: Yes (width={options.get('smearing_width', 0.01)})")

    # Tolerances
    if "tolerances" in options:
        print(f"Tolerances:")
        for key, value in options["tolerances"].items():
            print(f"  - {key}: {value}")

    # SCF settings
    print(f"SCF method: {options.get('scf_method', 'DIIS')}")
    print(f"SCF maxcycle: {options.get('scf_maxcycle', 800)}")
    print(f"FMIXING: {options.get('fmixing', 30)}%")

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
        default="cif2d12_options.json",
        help="File to save/load options for batch mode",
    )

    args = parser.parse_args()

    if args.batch:
        # Load options from file
        try:
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
        print("Enhanced by Marcus Djokic with AI assistance")
        print("")
        options = get_calculation_options()
        print_summary(options)

    # Always ask about saving options (unless in batch mode)
    if not args.batch:
        save_options = yes_no_prompt(
            "Save these options for future batch processing?", "yes"
        )
        if save_options or args.save_options:
            try:
                with open(args.options_file, "w") as f:
                    json.dump(options, f, indent=2)
                print(f"Saved options to {args.options_file}")
            except Exception as e:
                print(f"Error saving options to {args.options_file}: {e}")

    # Process CIF files
    process_cifs(args.cif_dir, options, args.output_dir)


if __name__ == "__main__":
    main()
