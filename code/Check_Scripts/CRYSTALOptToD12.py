#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CRYSTAL17/23 Optimization Output to D12 Converter
-------------------------------------------------
This script extracts optimized geometries from CRYSTAL17/23 output files
and creates new D12 input files for follow-up calculations.

DESCRIPTION:
    Takes the optimized geometry from CRYSTAL17/23 output files and creates
    new D12 input files with updated coordinates. The script attempts to
    preserve the original calculation settings and allows the user to
    modify them interactively.

USAGE:
    1. Single file processing:
       python CRYSTALOptToD12.py --out-file file.out --d12-file file.d12

    2. Process all files in a directory:
       python CRYSTALOptToD12.py --directory /path/to/files

    3. Batch processing with shared settings:
       python CRYSTALOptToD12.py --directory /path/to/files --shared-settings

    4. Specify output directory:
       python CRYSTALOptToD12.py --directory /path/to/files --output-dir /path/to/output

    5. Save/load settings:
       python CRYSTALOptToD12.py --save-options --options-file settings.json

AUTHOR:
    New entirely reworked script by Marcus Djokic
    Based on prior versions written by Wangwei Lan, Kevin Lucht, Danny Maldonado, Marcus Djokic
"""

import os
import sys
import re
import argparse
import json
from pathlib import Path


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

# Space group symbol to number mapping (Hermann-Mauguin symbols)
SPACEGROUP_SYMBOLS = {
    # Triclinic
    "P 1": 1,
    "P -1": 2,
    # Monoclinic
    "P 2": 3,
    "P 21": 4,
    "C 2": 5,
    "P M": 6,
    "P C": 7,
    "C M": 8,
    "C C": 9,
    "P 2/M": 10,
    "P 21/M": 11,
    "C 2/M": 12,
    "P 2/C": 13,
    "P 21/C": 14,
    "C 2/C": 15,
    # Orthorhombic
    "P 2 2 2": 16,
    "P 2 2 21": 17,
    "P 21 21 2": 18,
    "P 21 21 21": 19,
    "C 2 2 21": 20,
    "C 2 2 2": 21,
    "F 2 2 2": 22,
    "I 2 2 2": 23,
    "I 21 21 21": 24,
    "P M M 2": 25,
    "P M C 21": 26,
    "P C C 2": 27,
    "P M A 2": 28,
    "P C A 21": 29,
    "P N C 2": 30,
    "P M N 21": 31,
    "P B A 2": 32,
    "P N A 21": 33,
    "P N N 2": 34,
    "C M M 2": 35,
    "C M C 21": 36,
    "C C C 2": 37,
    "A M M 2": 38,
    "A B M 2": 39,
    "A M A 2": 40,
    "A B A 2": 41,
    "F M M 2": 42,
    "F D D 2": 43,
    "I M M 2": 44,
    "I B A 2": 45,
    "I M A 2": 46,
    "P M M M": 47,
    "P N N N": 48,
    "P C C M": 49,
    "P B A N": 50,
    "P M M A": 51,
    "P N N A": 52,
    "P M N A": 53,
    "P C C A": 54,
    "P B A M": 55,
    "P C C N": 56,
    "P B C M": 57,
    "P N N M": 58,
    "P M M N": 59,
    "P B C N": 60,
    "P B C A": 61,
    "P N M A": 62,
    "C M C M": 63,
    "C M C A": 64,
    "C M M M": 65,
    "C C C M": 66,
    "C M M A": 67,
    "C C C A": 68,
    "F M M M": 69,
    "F D D D": 70,
    "I M M M": 71,
    "I B A M": 72,
    "I B C A": 73,
    "I M M A": 74,
    # Tetragonal
    "P 4": 75,
    "P 41": 76,
    "P 42": 77,
    "P 43": 78,
    "I 4": 79,
    "I 41": 80,
    "P -4": 81,
    "I -4": 82,
    "P 4/M": 83,
    "P 42/M": 84,
    "P 4/N": 85,
    "P 42/N": 86,
    "I 4/M": 87,
    "I 41/A": 88,
    "P 4 2 2": 89,
    "P 4 21 2": 90,
    "P 41 2 2": 91,
    "P 41 21 2": 92,
    "P 42 2 2": 93,
    "P 42 21 2": 94,
    "P 43 2 2": 95,
    "P 43 21 2": 96,
    "I 4 2 2": 97,
    "I 41 2 2": 98,
    "P 4 M M": 99,
    "P 4 B M": 100,
    "P 42 C M": 101,
    "P 42 N M": 102,
    "P 4 C C": 103,
    "P 4 N C": 104,
    "P 42 M C": 105,
    "P 42 B C": 106,
    "I 4 M M": 107,
    "I 4 C M": 108,
    "I 41 M D": 109,
    "I 41 C D": 110,
    "P -4 2 M": 111,
    "P -4 2 C": 112,
    "P -4 21 M": 113,
    "P -4 21 C": 114,
    "P -4 M 2": 115,
    "P -4 C 2": 116,
    "P -4 B 2": 117,
    "P -4 N 2": 118,
    "I -4 M 2": 119,
    "I -4 C 2": 120,
    "I -4 2 M": 121,
    "I -4 2 D": 122,
    "P 4/M M M": 123,
    "P 4/M C C": 124,
    "P 4/N B M": 125,
    "P 4/N N C": 126,
    "P 4/M B M": 127,
    "P 4/M N C": 128,
    "P 4/N M M": 129,
    "P 4/N C C": 130,
    "P 42/M M C": 131,
    "P 42/M C M": 132,
    "P 42/N B C": 133,
    "P 42/N N M": 134,
    "P 42/M B C": 135,
    "P 42/M N M": 136,
    "P 42/N M C": 137,
    "P 42/N C M": 138,
    "I 4/M M M": 139,
    "I 4/M C M": 140,
    "I 41/A M D": 141,
    "I 41/A C D": 142,
    # Trigonal
    "P 3": 143,
    "P 31": 144,
    "P 32": 145,
    "R 3": 146,
    "P -3": 147,
    "R -3": 148,
    "P 3 1 2": 149,
    "P 3 2 1": 150,
    "P 31 1 2": 151,
    "P 31 2 1": 152,
    "P 32 1 2": 153,
    "P 32 2 1": 154,
    "R 3 2": 155,
    "P 3 M 1": 156,
    "P 3 1 M": 157,
    "P 3 C 1": 158,
    "P 3 1 C": 159,
    "R 3 M": 160,
    "R 3 C": 161,
    "P -3 1 M": 162,
    "P -3 1 C": 163,
    "P -3 M 1": 164,
    "P -3 C 1": 165,
    "R -3 M": 166,
    "R -3 C": 167,
    # Hexagonal
    "P 6": 168,
    "P 61": 169,
    "P 65": 170,
    "P 62": 171,
    "P 64": 172,
    "P 63": 173,
    "P -6": 174,
    "P 6/M": 175,
    "P 63/M": 176,
    "P 6 2 2": 177,
    "P 61 2 2": 178,
    "P 65 2 2": 179,
    "P 62 2 2": 180,
    "P 64 2 2": 181,
    "P 63 2 2": 182,
    "P 6 M M": 183,
    "P 6 C C": 184,
    "P 63 C M": 185,
    "P 63 M C": 186,
    "P -6 M 2": 187,
    "P -6 C 2": 188,
    "P -6 2 M": 189,
    "P -6 2 C": 190,
    "P 6/M M M": 191,
    "P 6/M C C": 192,
    "P 63/M C M": 193,
    "P 63/M M C": 194,
    # Cubic
    "P 2 3": 195,
    "F 2 3": 196,
    "I 2 3": 197,
    "P 21 3": 198,
    "I 21 3": 199,
    "P M 3": 200,
    "P N 3": 201,
    "F M 3": 202,
    "F D 3": 203,
    "I M 3": 204,
    "P A 3": 205,
    "I A 3": 206,
    "P 4 3 2": 207,
    "P 42 3 2": 208,
    "F 4 3 2": 209,
    "F 41 3 2": 210,
    "I 4 3 2": 211,
    "P 43 3 2": 212,
    "P 41 3 2": 213,
    "I 41 3 2": 214,
    "P -4 3 M": 215,
    "F -4 3 M": 216,
    "I -4 3 M": 217,
    "P -4 3 N": 218,
    "F -4 3 C": 219,
    "I -4 3 D": 220,
    "P M 3 M": 221,
    "P N 3 N": 222,
    "P M 3 N": 223,
    "P N 3 M": 224,
    "F M 3 M": 225,
    "F M 3 C": 226,
    "F D 3 M": 227,
    "F D 3 C": 228,
    "I M 3 M": 229,
    "I A 3 D": 230,
}

# Alternative spellings and common variations
SPACEGROUP_ALTERNATIVES = {
    "P1": 1,
    "P-1": 2,
    "P2": 3,
    "P21": 4,
    "C2": 5,
    "PM": 6,
    "PC": 7,
    "CM": 8,
    "CC": 9,
    "P2/M": 10,
    "P21/M": 11,
    "C2/M": 12,
    "P2/C": 13,
    "P21/C": 14,
    "C2/C": 15,
    # Add more as needed
}

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
}

# Functionals available for D3 dispersion correction
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


class CrystalOutputParser:
    """Enhanced parser for CRYSTAL17/23 output files"""

    def __init__(self, output_file):
        self.output_file = output_file
        self.data = {
            "primitive_cell": [],
            "conventional_cell": [],
            "coordinates": [],
            "spacegroup": None,
            "dimensionality": "CRYSTAL",
            "calculation_type": None,
            "functional": None,
            "basis_set": None,
            "basis_set_type": "INTERNAL",
            "tolerances": {},
            "scf_settings": {},
            "spin_polarized": False,
            "dft_grid": None,
            "dispersion": False,
            "k_points": None,
            "origin_setting": "0 0 0",
            "is_3c_method": False,
            "smearing": False,
            "smearing_width": None,
        }

    def parse(self):
        """Parse the output file and extract all relevant data"""
        with open(self.output_file, "r") as f:
            content = f.read()
            lines = content.split("\n")

        # Extract dimensionality first
        self._extract_dimensionality(lines)

        # Extract optimized geometry
        self._extract_geometry(content)

        # Extract calculation settings
        self._extract_settings(lines)

        return self.data

    def _extract_dimensionality(self, lines):
        """Extract system dimensionality"""
        for i, line in enumerate(lines):
            if "CRYSTAL" in line and "CALCULATION" in line:
                self.data["dimensionality"] = "CRYSTAL"
            elif "SLAB CALCULATION" in line:
                self.data["dimensionality"] = "SLAB"
            elif "POLYMER CALCULATION" in line:
                self.data["dimensionality"] = "POLYMER"
            elif "MOLECULE CALCULATION" in line:
                self.data["dimensionality"] = "MOLECULE"

            # Also check input echo section
            if "*                               CRYSTAL" in line:
                # Look for the dimensionality in the next few lines
                for j in range(i, min(i + 10, len(lines))):
                    if "CRYSTAL - PROPERTIES OF THE CRYSTALLINE STATE" in lines[j]:
                        self.data["dimensionality"] = "CRYSTAL"
                        break
                    elif "SLAB" in lines[j]:
                        self.data["dimensionality"] = "SLAB"
                        break
                    elif "POLYMER" in lines[j]:
                        self.data["dimensionality"] = "POLYMER"
                        break
                    elif "MOLECULE" in lines[j]:
                        self.data["dimensionality"] = "MOLECULE"
                        break

    def _extract_geometry(self, content):
        """Extract optimized geometry from output"""
        lines = content.split("\n")

        # Find FINAL OPTIMIZED GEOMETRY section
        final_geom_idx = None
        for i, line in enumerate(lines):
            if "FINAL OPTIMIZED GEOMETRY" in line and "DIMENSIONALITY" in line:
                final_geom_idx = i
                break

        if final_geom_idx is None:
            # Try alternative patterns
            for i, line in enumerate(lines):
                if "OPT END - CONVERGED" in line:
                    # Look for the geometry after this
                    for j in range(i, min(i + 50, len(lines))):
                        if (
                            "LATTICE PARAMETERS" in lines[j]
                            and "PRIMITIVE CELL" in lines[j]
                        ):
                            final_geom_idx = j - 2
                            break
                    break

        if final_geom_idx is None:
            # For single point calculations, look for initial geometry
            for i, line in enumerate(lines):
                if "GEOMETRY FOR WAVE FUNCTION" in line:
                    final_geom_idx = i
                    break

        if final_geom_idx is None:
            raise ValueError("Could not find geometry in output file")

        # Extract space group
        self._extract_spacegroup(lines)

        # Extract cell parameters
        self._extract_cell_parameters(lines, final_geom_idx)

        # Extract atomic coordinates
        self._extract_coordinates(lines, final_geom_idx)

    def _extract_spacegroup(self, lines):
        """Extract space group number from Hermann-Mauguin symbol"""
        for i, line in enumerate(lines):
            if "SPACE GROUP" in line and ":" in line:
                # Extract the space group symbol after the colon
                parts = line.split(":")
                if len(parts) >= 2:
                    sg_symbol = parts[1].strip()

                    # Try to match in our symbol dictionary
                    if sg_symbol in SPACEGROUP_SYMBOLS:
                        self.data["spacegroup"] = SPACEGROUP_SYMBOLS[sg_symbol]
                        return

                    # Try alternative spellings
                    if sg_symbol in SPACEGROUP_ALTERNATIVES:
                        self.data["spacegroup"] = SPACEGROUP_ALTERNATIVES[sg_symbol]
                        return

                    # Try with spaces removed
                    sg_no_space = sg_symbol.replace(" ", "")
                    if sg_no_space in SPACEGROUP_ALTERNATIVES:
                        self.data["spacegroup"] = SPACEGROUP_ALTERNATIVES[sg_no_space]
                        return

                    print(
                        f"Warning: Could not find space group number for symbol '{sg_symbol}'"
                    )

    def _extract_cell_parameters(self, lines, start_idx):
        """Extract unit cell parameters"""
        # Handle different dimensionalities
        if self.data["dimensionality"] == "MOLECULE":
            # No cell parameters for molecules
            return

        # Look for PRIMITIVE CELL
        for i in range(start_idx, min(start_idx + 100, len(lines))):
            if "PRIMITIVE CELL" in lines[i] and "LATTICE PARAMETERS" in lines[i]:
                # Skip to parameter line
                for j in range(i + 1, i + 10):
                    if (
                        "ALPHA" in lines[j]
                        and "BETA" in lines[j]
                        and "GAMMA" in lines[j]
                    ):
                        params = lines[j + 1].split()
                        if len(params) >= 6:
                            self.data["primitive_cell"] = params[:6]
                        break
                break

        # Look for CRYSTALLOGRAPHIC CELL
        for i in range(start_idx, min(start_idx + 100, len(lines))):
            if "CRYSTALLOGRAPHIC CELL" in lines[i] and "VOLUME" in lines[i]:
                # Skip to parameter line
                for j in range(i + 1, i + 10):
                    if (
                        "ALPHA" in lines[j]
                        and "BETA" in lines[j]
                        and "GAMMA" in lines[j]
                    ):
                        params = lines[j + 1].split()
                        if len(params) >= 6:
                            self.data["conventional_cell"] = params[:6]
                        break
                break

        # If no conventional cell found, use primitive
        if not self.data["conventional_cell"] and self.data["primitive_cell"]:
            self.data["conventional_cell"] = self.data["primitive_cell"]

    def _extract_coordinates(self, lines, start_idx):
        """Extract atomic coordinates with symmetry information"""
        coordinates = []

        # Look for coordinates based on dimensionality
        if self.data["dimensionality"] == "MOLECULE":
            # Look for Cartesian coordinates
            for i in range(start_idx, min(start_idx + 200, len(lines))):
                if "CARTESIAN COORDINATES" in lines[i] and "PRIMITIVE CELL" in lines[i]:
                    # Skip the header lines
                    j = i + 3
                    while j < len(lines):
                        parts = lines[j].split()
                        if len(parts) >= 6 and parts[1].isdigit():
                            try:
                                coord = {
                                    "atom_number": parts[2],
                                    "x": parts[3],
                                    "y": parts[4],
                                    "z": parts[5],
                                    "is_unique": True,  # Molecules typically have all atoms unique
                                }
                                coordinates.append(coord)
                            except:
                                break
                        else:
                            break
                        j += 1
                    break
        else:
            # Look for coordinates with T/F symmetry markers
            found_coords = False

            # First try to find ATOMS IN THE ASYMMETRIC UNIT section
            for i in range(start_idx, min(start_idx + 200, len(lines))):
                if "ATOMS IN THE ASYMMETRIC UNIT" in lines[i]:
                    # Skip header lines
                    j = i + 3
                    while j < len(lines):
                        parts = lines[j].split()
                        if len(parts) >= 7:
                            # Check for T/F marker
                            if parts[1] in ["T", "F"]:
                                try:
                                    coord = {
                                        "atom_number": parts[2],
                                        "x": parts[4],
                                        "y": parts[5],
                                        "z": parts[6],
                                        "is_unique": parts[1]
                                        == "T",  # True if T, False if F
                                    }
                                    coordinates.append(coord)
                                except:
                                    break
                            else:
                                break
                        elif len(parts) < 6 or not lines[j].strip():
                            break
                        j += 1
                    found_coords = True
                    break

            # If not found, try COORDINATES IN THE CRYSTALLOGRAPHIC CELL
            if not found_coords:
                for i in range(start_idx, min(start_idx + 200, len(lines))):
                    if "COORDINATES IN THE CRYSTALLOGRAPHIC CELL" in lines[i]:
                        # Skip header lines
                        j = i + 3
                        while j < len(lines):
                            parts = lines[j].split()
                            if len(parts) >= 7 and (parts[1] == "T" or parts[1] == "F"):
                                coord = {
                                    "atom_number": parts[2],
                                    "x": parts[4],
                                    "y": parts[5],
                                    "z": parts[6],
                                    "is_unique": parts[1] == "T",
                                }
                                coordinates.append(coord)
                            elif len(parts) < 6 or not lines[j].strip():
                                break
                            j += 1
                        break

        self.data["coordinates"] = coordinates

    def _extract_settings(self, lines):
        """Extract calculation settings from output"""
        # Extract DFT functional and check for 3C methods
        self._extract_functional(lines)

        # Extract basis set
        self._extract_basis_set(lines)

        # Extract tolerances
        self._extract_tolerances(lines)

        # Extract SCF settings
        self._extract_scf_settings(lines)

        # Extract k-points
        self._extract_kpoints(lines)

        # Check for spin polarization
        self.data["spin_polarized"] = any(
            "SPIN POLARIZED" in line or "UNRESTRICTED" in line or "SPIN" in line
            for line in lines
        )

        # Extract DFT grid
        self._extract_dft_grid(lines)

        # Check for dispersion correction
        self._extract_dispersion(lines)

        # Check for smearing
        self._extract_smearing(lines)

    def _extract_functional(self, lines):
        """Extract DFT functional including 3C methods"""

        # First check for specific functional patterns in ENERGY EXPRESSION line
        for line in lines:
            if "ENERGY EXPRESSION" in line:
                # Parse the functional from this line
                if "HARTREE+FOCK" in line and "EXCH*0.20000" in line:
                    if "BECKE" in line and "LYP" in line:
                        self.data["functional"] = "B3LYP"
                        return
                    elif "BECKE" in line and "PW91" in line:
                        self.data["functional"] = "B3PW"
                        return
                elif "HARTREE+FOCK" in line and "EXCH*0.25000" in line:
                    if "PBE" in line:
                        self.data["functional"] = "PBE0"
                        return
                elif "HARTREE+FOCK" in line and not "EXCH*" in line:
                    self.data["functional"] = "HF"
                    return

        # Check for 3C methods
        for line in lines:
            # Check for 3C methods
            if "HF-3C" in line or "HF3C" in line:
                self.data["functional"] = "HF-3C"
                self.data["is_3c_method"] = True
                return
            elif "PBEH-3C" in line or "PBEH3C" in line:
                self.data["functional"] = "PBEh-3C"
                self.data["is_3c_method"] = True
                return
            elif "HSE-3C" in line or "HSE3C" in line:
                self.data["functional"] = "HSE-3C"
                self.data["is_3c_method"] = True
                return
            elif "B97-3C" in line or "B973C" in line:
                self.data["functional"] = "B97-3C"
                self.data["is_3c_method"] = True
                return
            elif "HFSOL-3C" in line or "HFSOL3C" in line:
                self.data["functional"] = "HFsol-3C"
                self.data["is_3c_method"] = True
                return
            elif "PBESOL0-3C" in line or "PBESOL03C" in line:
                self.data["functional"] = "PBEsol0-3C"
                self.data["is_3c_method"] = True
                return
            elif "HSESOL-3C" in line or "HSESOL3C" in line:
                self.data["functional"] = "HSEsol-3C"
                self.data["is_3c_method"] = True
                return

        # Check for functional in DFT PARAMETERS section
        functional_found = False
        for i, line in enumerate(lines):
            if "KOHN-SHAM HAMILTONIAN" in line:
                # Look at the next few lines for functional info
                for j in range(i + 1, min(i + 10, len(lines))):
                    if "(EXCHANGE)" in lines[j] and "[CORRELATION]" in lines[j]:
                        # Parse functional info
                        if "BECKE 88" in lines[j] and "LEE-YANG-PARR" in lines[j]:
                            self.data["functional"] = "B3LYP"
                            functional_found = True
                            break
                        elif "PBE" in lines[j]:
                            # Check if it's a hybrid
                            for k in range(j, min(j + 5, len(lines))):
                                if "HYBRID EXCHANGE" in lines[k]:
                                    self.data["functional"] = "PBE0"
                                    functional_found = True
                                    break
                            if not functional_found:
                                self.data["functional"] = "PBE"
                                functional_found = True
                            break

                if functional_found:
                    break

        # If no DFT functional found, check if it's Hartree-Fock
        if not functional_found and not self.data["is_3c_method"]:
            for line in lines:
                if "TYPE OF CALCULATION" in line:
                    if "RESTRICTED CLOSED SHELL" in line and "KOHN-SHAM" not in line:
                        self.data["functional"] = "HF"
                        break
                    elif "UNRESTRICTED OPEN SHELL" in line and "KOHN-SHAM" not in line:
                        self.data["functional"] = "UHF"
                        break

    def _extract_basis_set(self, lines):
        """Extract basis set information"""
        # Look for "Loading internal basis set:" pattern
        for line in lines:
            if "Loading internal basis set:" in line:
                # Extract basis set name after the colon
                parts = line.split(":")
                if len(parts) >= 2:
                    basis_name = parts[1].strip()
                    self.data["basis_set"] = basis_name
                    self.data["basis_set_type"] = "INTERNAL"
                    return

        # For 3C methods, basis set is determined by the method
        if self.data["is_3c_method"] and self.data["functional"]:
            for category in FUNCTIONAL_CATEGORIES.values():
                if "basis_requirements" in category:
                    if self.data["functional"] in category["basis_requirements"]:
                        self.data["basis_set"] = category["basis_requirements"][
                            self.data["functional"]
                        ]
                        self.data["basis_set_type"] = "INTERNAL"
                        return

        # Look for basis set in output
        for i, line in enumerate(lines):
            # Check for internal basis sets
            for basis_name in INTERNAL_BASIS_SETS.keys():
                if basis_name in line and "BASIS SET" in line:
                    self.data["basis_set"] = basis_name
                    self.data["basis_set_type"] = "INTERNAL"
                    return

            # Check for external basis indication
            if "LOCAL ATOMIC FUNCTIONS BASIS SET" in line:
                self.data["basis_set_type"] = "EXTERNAL"
                # External basis set - will need to extract from d12 file
                self.data["basis_set"] = "EXTERNAL"
                return

    def _extract_tolerances(self, lines):
        """Extract tolerance settings"""
        for i, line in enumerate(lines):
            # Look for TOLINTEG pattern
            if "INFORMATION **** TOLINTEG ****" in line:
                # Look for the values in subsequent lines
                for j in range(i + 1, min(i + 5, len(lines))):
                    if "COULOMB AND EXCHANGE SERIES TOLERANCES MODIFIED" in lines[j]:
                        # Parse the next lines for actual values
                        for k in range(j + 1, min(j + 10, len(lines))):
                            if "COULOMB OVERLAP TOL" in lines[k]:
                                # Extract T1 value
                                match = re.search(
                                    r"\(T1\)\s*10\*\*\s*(-?\d+)", lines[k]
                                )
                                if match:
                                    t1 = int(match.group(1))
                                    # Look for other values
                                    t2 = t3 = t4 = t5 = t1  # Default to same as T1

                                    # Extract other tolerances
                                    if (
                                        k + 1 < len(lines)
                                        and "COULOMB PENETRATION TOL" in lines[k + 1]
                                    ):
                                        match = re.search(
                                            r"\(T2\)\s*10\*\*\s*(-?\d+)", lines[k + 1]
                                        )
                                        if match:
                                            t2 = int(match.group(1))

                                    if (
                                        k + 2 < len(lines)
                                        and "EXCHANGE OVERLAP TOL" in lines[k + 2]
                                    ):
                                        match = re.search(
                                            r"\(T3\)\s*10\*\*\s*(-?\d+)", lines[k + 2]
                                        )
                                        if match:
                                            t3 = int(match.group(1))

                                    if (
                                        k + 3 < len(lines)
                                        and "EXCHANGE PSEUDO OVP (F(G))" in lines[k + 3]
                                    ):
                                        match = re.search(
                                            r"\(T4\)\s*10\*\*\s*(-?\d+)", lines[k + 3]
                                        )
                                        if match:
                                            t4 = int(match.group(1))

                                    if (
                                        k + 4 < len(lines)
                                        and "EXCHANGE PSEUDO OVP (P(G))" in lines[k + 4]
                                    ):
                                        match = re.search(
                                            r"\(T5\)\s*10\*\*\s*(-?\d+)", lines[k + 4]
                                        )
                                        if match:
                                            t5 = int(match.group(1))

                                    # Format as string
                                    self.data["tolerances"]["TOLINTEG"] = (
                                        f"{-t1} {-t2} {-t3} {-t4} {-t5}"
                                    )
                                    break
                        break

            # Look for TOLDEE
            elif "INFORMATION **** TOLDEE ****" in line:
                # Extract TOLDEE value
                match = re.search(r"SCF TOL ON TOTAL ENERGY SET TO\s*(\d+)", line)
                if match:
                    self.data["tolerances"]["TOLDEE"] = int(match.group(1))

    def _extract_scf_settings(self, lines):
        """Extract SCF settings"""
        for i, line in enumerate(lines):
            # SCF method - check for DIIS explicitly
            if "INFORMATION **** DIIS ****" in line and "DIIS FOR SCF ACTIVE" in line:
                self.data["scf_settings"]["method"] = "DIIS"
            elif "ANDERSON" in line and "MIXING" in line:
                self.data["scf_settings"]["method"] = "ANDERSON"
            elif "BROYDEN" in line and "MIXING" in line:
                self.data["scf_settings"]["method"] = "BROYDEN"

            # MAXCYCLE
            if "INFORMATION **** MAXCYCLE ****" in line:
                match = re.search(r"MAX NUMBER OF SCF CYCLES SET TO\s*(\d+)", line)
                if match:
                    self.data["scf_settings"]["maxcycle"] = int(match.group(1))

            # FMIXING
            if "FMIXING" in line and "SET TO" in line:
                match = re.search(r"FOCK/KS MATRIX MIXING SET TO\s*(\d+)\s*%", line)
                if match:
                    self.data["scf_settings"]["fmixing"] = int(match.group(1))

    def _extract_kpoints(self, lines):
        """Extract k-point information"""
        for i, line in enumerate(lines):
            if "SHRINK. FACT.(MONKH.)" in line:
                # Extract k-point mesh from the same line
                parts = line.split()
                for j, part in enumerate(parts):
                    if "SHRINK." in part:
                        # k-points should be the next 3 numbers
                        if j + 3 < len(parts):
                            try:
                                k1 = int(parts[j + 2])
                                k2 = int(parts[j + 3])
                                k3 = int(parts[j + 4])
                                self.data["k_points"] = f"{k1} {k2} {k3}"
                                return
                            except:
                                pass

    def _extract_dft_grid(self, lines):
        """Extract DFT grid information"""
        for line in lines:
            # Check for grid setting in information lines
            if "NEW DEFAULT: DFT INTEGRATION GRID INCREASED TO" in line:
                for grid_key, grid_name in DFT_GRIDS.items():
                    if grid_name in line:
                        self.data["dft_grid"] = grid_name
                        return

            # Also check for SIZE OF GRID
            if "SIZE OF GRID" in line:
                match = re.search(r"SIZE OF GRID\s*=\s*(\d+)", line)
                if match:
                    grid_size = int(match.group(1))
                    # Map grid sizes to grid names (approximate)
                    if grid_size < 1000:
                        self.data["dft_grid"] = "OLDGRID"
                    elif grid_size < 2000:
                        self.data["dft_grid"] = "LGRID"
                    elif grid_size < 3000:
                        self.data["dft_grid"] = "XLGRID"
                    else:
                        self.data["dft_grid"] = "XXLGRID"

    def _extract_dispersion(self, lines):
        """Extract dispersion correction information"""
        for line in lines:
            if "PERFORM LATEST DISPERSION CORRECTION DFT-D3" in line:
                self.data["dispersion"] = True
                return
            elif "D3 DISPERSION ENERGY" in line:
                self.data["dispersion"] = True
                return
            elif "GRIMME D3" in line or "DFT-D3" in line:
                self.data["dispersion"] = True
                return

    def _extract_smearing(self, lines):
        """Extract Fermi smearing information"""
        for line in lines:
            if "SMEARING" in line or "FERMI" in line:
                if "WIDTH" in line:
                    match = re.search(r"WIDTH.*?([\d.]+)", line)
                    if match:
                        self.data["smearing"] = True
                        self.data["smearing_width"] = float(match.group(1))


class CrystalInputParser:
    """Enhanced parser for CRYSTAL17/23 input files"""

    def __init__(self, input_file):
        self.input_file = input_file
        self.data = {
            "spacegroup": None,
            "dimensionality": "CRYSTAL",
            "basis_set_type": "INTERNAL",
            "basis_set": None,
            "basis_set_path": None,
            "scf_method": "DIIS",
            "scf_maxcycle": 800,
            "fmixing": 30,
            "k_points": None,
            "origin_setting": "0 0 0",
            "optimization_settings": {},
            "freq_settings": {},
            "external_basis_info": [],
            "external_basis_data": [],  # Store the actual basis set data
        }

    def parse(self):
        """Parse the input file"""
        with open(self.input_file, "r") as f:
            lines = f.readlines()

        # Extract dimensionality and space group
        for i, line in enumerate(lines):
            if line.strip() in ["CRYSTAL", "SLAB", "POLYMER", "MOLECULE"]:
                self.data["dimensionality"] = line.strip()
                # Next lines should have origin and space group for CRYSTAL
                if self.data["dimensionality"] == "CRYSTAL" and i + 2 < len(lines):
                    self.data["origin_setting"] = lines[i + 1].strip()
                    try:
                        self.data["spacegroup"] = int(lines[i + 2].strip())
                    except:
                        pass
                elif self.data["dimensionality"] in ["SLAB", "POLYMER"] and i + 1 < len(
                    lines
                ):
                    try:
                        self.data["spacegroup"] = int(lines[i + 1].strip())
                    except:
                        pass
                break

        # Extract basis set
        self._extract_basis_set(lines)

        # Extract optimization settings
        self._extract_optimization_settings(lines)

        # Extract SCF settings
        for i, line in enumerate(lines):
            if "MAXCYCLE" in line and "OPTGEOM" not in lines[max(0, i - 5) : i]:
                if i + 1 < len(lines):
                    try:
                        self.data["scf_maxcycle"] = int(lines[i + 1].strip())
                    except:
                        pass
            elif "FMIXING" in line:
                if i + 1 < len(lines):
                    try:
                        self.data["fmixing"] = int(lines[i + 1].strip())
                    except:
                        pass
            elif line.strip() in ["DIIS", "ANDERSON", "BROYDEN"]:
                self.data["scf_method"] = line.strip()

        # Extract k-points
        for i, line in enumerate(lines):
            if "SHRINK" in line:
                if i + 2 < len(lines) and self.data["dimensionality"] == "CRYSTAL":
                    self.data["k_points"] = lines[i + 2].strip()
                break

        return self.data

    def _extract_basis_set(self, lines):
        """Extract basis set information"""
        # First find END of geometry section
        geom_end = None
        for i, line in enumerate(lines):
            if line.strip() == "END" and i > 0:
                # This should be the END after geometry
                geom_end = i
                break

        if geom_end is None:
            return

        # Check what comes after geometry END
        for i in range(geom_end + 1, len(lines)):
            if lines[i].strip() == "BASISSET":
                self.data["basis_set_type"] = "INTERNAL"
                if i + 1 < len(lines):
                    self.data["basis_set"] = lines[i + 1].strip()
                return
            elif "99 0" in lines[i]:
                # External basis set
                self.data["basis_set_type"] = "EXTERNAL"
                # Extract all basis set data between geometry END and 99 0
                for j in range(geom_end + 1, i):
                    line = lines[j].strip()
                    if line and not line.startswith("#"):  # Skip comments
                        self.data["external_basis_data"].append(line)
                return

    def _extract_optimization_settings(self, lines):
        """Extract optimization settings if present"""
        in_optgeom = False
        for i, line in enumerate(lines):
            if "OPTGEOM" in line:
                in_optgeom = True
            elif "ENDOPT" in line:
                in_optgeom = False
            elif in_optgeom:
                if line.strip() in OPT_TYPES.values():
                    self.data["optimization_settings"]["type"] = line.strip()
                elif "MAXCYCLE" in line and i + 1 < len(lines):
                    try:
                        self.data["optimization_settings"]["MAXCYCLE"] = int(
                            lines[i + 1].strip()
                        )
                    except:
                        pass
                elif "TOLDEG" in line and i + 1 < len(lines):
                    try:
                        self.data["optimization_settings"]["TOLDEG"] = float(
                            lines[i + 1].strip()
                        )
                    except:
                        pass
                elif "TOLDEX" in line and i + 1 < len(lines):
                    try:
                        self.data["optimization_settings"]["TOLDEX"] = float(
                            lines[i + 1].strip()
                        )
                    except:
                        pass
                elif "TOLDEE" in line and i + 1 < len(lines):
                    try:
                        self.data["optimization_settings"]["TOLDEE"] = int(
                            lines[i + 1].strip()
                        )
                    except:
                        pass
                elif "MAXTRADIUS" in line and i + 1 < len(lines):
                    try:
                        self.data["optimization_settings"]["MAXTRADIUS"] = float(
                            lines[i + 1].strip()
                        )
                    except:
                        pass


def display_current_settings(settings):
    """Display current calculation settings"""
    print("\n" + "=" * 70)
    print("EXTRACTED CALCULATION SETTINGS")
    print("=" * 70)

    print(f"\nDimensionality: {settings.get('dimensionality', 'CRYSTAL')}")
    print(f"Space group: {settings.get('spacegroup', 'N/A')}")
    print(f"Origin setting: {settings.get('origin_setting', '0 0 0')}")

    if settings.get("functional"):
        func = settings["functional"]
        if settings.get("dispersion"):
            func += "-D3"
        print(f"DFT functional: {func}")
        if settings.get("is_3c_method"):
            print(f"  (3c composite method)")
    else:
        print(f"Method: Hartree-Fock")

    print(
        f"Basis set: {settings.get('basis_set', 'N/A')} ({settings.get('basis_set_type', 'INTERNAL')})"
    )
    print(f"Spin polarized: {settings.get('spin_polarized', False)}")

    if settings.get("dft_grid"):
        print(f"DFT grid: {settings.get('dft_grid', 'XLGRID')}")

    if settings.get("tolerances"):
        print(
            f"Tolerances: TOLINTEG={settings['tolerances'].get('TOLINTEG', 'N/A')}, "
            f"TOLDEE={settings['tolerances'].get('TOLDEE', 'N/A')}"
        )

    if settings.get("scf_settings"):
        print(f"SCF method: {settings['scf_settings'].get('method', 'DIIS')}")
        print(f"SCF max cycles: {settings['scf_settings'].get('maxcycle', 800)}")
        print(f"FMIXING: {settings['scf_settings'].get('fmixing', 30)}%")

    if settings.get("k_points"):
        print(f"K-points: {settings['k_points']}")

    if settings.get("smearing"):
        print(f"Fermi smearing: Yes (width={settings.get('smearing_width', 0.01)})")

    print("=" * 70)


def select_functional():
    """Select DFT functional by category"""
    # First, select category - ordered to match NewCifToD12.py
    category_options = {}
    ordered_categories = ["HF", "LDA", "GGA", "HYBRID", "MGGA", "3C"]

    for i, key in enumerate(ordered_categories, 1):
        info = FUNCTIONAL_CATEGORIES[key]
        category_options[str(i)] = key
        print(f"\n{i}. {info['name']}")
        print(f"   {info['description']}")
        # Show appropriate examples for each category
        if key == "HYBRID":
            print(f"   Examples: B3LYP, PBE0, HSE06, LC-wPBE")
        elif key == "3C":
            print(f"   Examples: HF-3C, PBEh-3C, HSE-3C, B97-3C")
        else:
            print(f"   Examples: {', '.join(info['functionals'][:4])}")

    category_choice = get_user_input(
        "Select functional category", category_options, "4"
    )  # Default to HYBRID
    selected_category = category_options[category_choice]

    # Then select specific functional
    category_info = FUNCTIONAL_CATEGORIES[selected_category]
    functional_options = {
        str(i + 1): func for i, func in enumerate(category_info["functionals"])
    }

    print(f"\nAvailable {category_info['name']}:")
    for key, func in functional_options.items():
        # Build the description string
        desc_parts = []

        # Add functional description if available
        if "descriptions" in category_info and func in category_info["descriptions"]:
            desc_parts.append(category_info["descriptions"][func])

        # Add D3 support indicator
        if func in D3_FUNCTIONALS:
            desc_parts.append("[D3✓]")

        # Add basis requirement for 3C methods
        if selected_category in ["3C", "HF"] and "basis_requirements" in category_info:
            basis = category_info["basis_requirements"].get(func, "")
            if basis:
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
    if "basis_requirements" in category_info:
        required_basis = category_info["basis_requirements"].get(selected_functional)
        return selected_functional, required_basis

    return selected_functional, None


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CRYSTAL17/23 Optimization Output to D12 Converter
-------------------------------------------------
This script extracts optimized geometries from CRYSTAL17/23 output files
and creates new D12 input files for follow-up calculations.

DESCRIPTION:
    Takes the optimized geometry from CRYSTAL17/23 output files and creates
    new D12 input files with updated coordinates. The script attempts to
    preserve the original calculation settings and allows the user to
    modify them interactively.

USAGE:
    1. Single file processing:
       python CRYSTALOptToD12.py --out-file file.out --d12-file file.d12

    2. Process all files in a directory:
       python CRYSTALOptToD12.py --directory /path/to/files

    3. Batch processing with shared settings:
       python CRYSTALOptToD12.py --directory /path/to/files --shared-settings

    4. Specify output directory:
       python CRYSTALOptToD12.py --directory /path/to/files --output-dir /path/to/output

    5. Save/load settings:
       python CRYSTALOptToD12.py --save-options --options-file settings.json

AUTHOR:
    Original script by Marcus Djokic
    Contributions by Wangwei Lan, Kevin Lucht, Danny Maldonado
"""

# ... [Keep all imports and other code until get_calculation_options] ...


def get_calculation_options(current_settings, shared_mode=False):
    """Get calculation options from user

    Args:
        current_settings: Current settings extracted from files
        shared_mode: If True, only ask for calculation settings to be shared

    Returns:
        dict: Options for the calculation
    """
    options = current_settings.copy()

    if not shared_mode:
        # Display current settings
        display_current_settings(current_settings)

        # Ask if user wants to keep current settings
        keep_settings = yes_no_prompt(
            "\nKeep these settings for the new calculation?", "yes"
        )
    else:
        # In shared mode, we'll modify calculation settings
        keep_settings = False

    if keep_settings:
        # Just ask for calculation type
        calc_options = {
            "1": "SP",  # Single Point
            "2": "OPT",  # Geometry Optimization
            "3": "FREQ",  # Frequency Calculation
        }
        calc_choice = get_user_input(
            "Select calculation type for the new input", calc_options, "1"
        )
        options["calculation_type"] = calc_options[calc_choice]

        # If OPT selected, warn that geometry is already optimized
        if options["calculation_type"] == "OPT":
            print(
                "\nWarning: The geometry is already optimized. Are you sure you want to run another optimization?"
            )
            if not yes_no_prompt("Continue with geometry optimization?", "no"):
                options["calculation_type"] = "SP"

            # If still OPT, get optimization settings
            if options["calculation_type"] == "OPT":
                opt_choice = get_user_input("Select optimization type", OPT_TYPES, "1")
                options["optimization_type"] = OPT_TYPES[opt_choice]

                # Use default optimization settings
                options["optimization_settings"] = DEFAULT_OPT_SETTINGS.copy()

        # If FREQ, set frequency settings
        if options["calculation_type"] == "FREQ":
            options["freq_settings"] = DEFAULT_FREQ_SETTINGS.copy()
            # Override tolerances for frequency calculations
            options["tolerances"] = {
                "TOLINTEG": DEFAULT_FREQ_SETTINGS["TOLINTEG"],
                "TOLDEE": DEFAULT_FREQ_SETTINGS["TOLDEE"],
            }
    else:
        # Allow full customization
        if shared_mode:
            print("\nDefine shared calculation settings for all files:")
        else:
            print("\nCustomize calculation settings:")

        # Calculation type
        calc_options = {
            "1": "SP",  # Single Point
            "2": "OPT",  # Geometry Optimization
            "3": "FREQ",  # Frequency Calculation
        }
        calc_choice = get_user_input("Select calculation type", calc_options, "1")
        options["calculation_type"] = calc_options[calc_choice]

        # Optimization settings if OPT
        if options["calculation_type"] == "OPT":
            opt_choice = get_user_input("Select optimization type", OPT_TYPES, "1")
            options["optimization_type"] = OPT_TYPES[opt_choice]

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
                maxtradius = float(
                    input("Enter MAXTRADIUS (max displacement, default 0.25): ") or 0.25
                )
                options["optimization_settings"]["MAXTRADIUS"] = maxtradius

        # Frequency settings if FREQ
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
            options["tolerances"] = {
                "TOLINTEG": DEFAULT_FREQ_SETTINGS["TOLINTEG"],
                "TOLDEE": DEFAULT_FREQ_SETTINGS["TOLDEE"],
            }

        # Method selection
        method_options = {"1": "DFT", "2": "HF"}
        print("\nSelect calculation method:")
        print("1: DFT - Density Functional Theory")
        print("2: HF - Hartree-Fock")

        # Default to DFT if current functional is DFT, otherwise HF
        default_method = (
            "1" if options.get("functional") not in ["HF", "UHF", None] else "2"
        )
        method_choice = get_user_input("Select method", method_options, default_method)

        if method_options[method_choice] == "HF":
            # HF method
            change_functional = yes_no_prompt(
                f"Change HF method (current: {options.get('functional', 'RHF')})?",
                "no",  # Changed default to "no" for both modes
            )
            if change_functional:
                functional, required_basis = select_functional()
                options["functional"] = functional

                # If functional has required basis set, use it automatically
                if required_basis:
                    print(
                        f"\nNote: {functional} requires {required_basis} basis set. Using it automatically."
                    )
                    options["basis_set_type"] = "INTERNAL"
                    options["basis_set"] = required_basis
                    options["is_3c_method"] = True
                else:
                    options["is_3c_method"] = False
        else:
            # DFT method
            change_functional = yes_no_prompt(
                f"Change DFT functional (current: {options.get('functional', 'N/A')})?",
                "no",  # Changed default to "no" for both modes
            )
            if change_functional:
                functional, required_basis = select_functional()
                options["functional"] = functional

                # If functional has required basis set, use it automatically
                if required_basis:
                    print(
                        f"\nNote: {functional} requires {required_basis} basis set. Using it automatically."
                    )
                    options["basis_set_type"] = "INTERNAL"
                    options["basis_set"] = required_basis
                    options["is_3c_method"] = True
                else:
                    options["is_3c_method"] = False

                    # Check if dispersion correction is available
                    if functional in D3_FUNCTIONALS:
                        options["dispersion"] = yes_no_prompt(
                            f"Add D3 dispersion correction to {functional}?",
                            "yes" if options.get("dispersion") else "no",
                        )
                    else:
                        options["dispersion"] = False

        # Basis set (if not determined by functional)
        if not (options.get("is_3c_method") and options.get("basis_set")):
            change_basis = yes_no_prompt(
                f"Change basis set (current: {options.get('basis_set', 'N/A')})?",
                "no",  # Changed default to "no" for both modes
            )
            if change_basis:
                basis_type = get_user_input(
                    "Basis set type", {"1": "EXTERNAL", "2": "INTERNAL"}, "2"
                )

                if basis_type == "2":
                    internal_basis_options = {}
                    print("\nAvailable internal basis sets:")

                    # First show standard basis sets
                    print("\n--- STANDARD BASIS SETS ---")
                    option_num = 1
                    for bs_name, bs_info in INTERNAL_BASIS_SETS.items():
                        if bs_info.get("standard", False):
                            internal_basis_options[str(option_num)] = bs_name
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

                    basis_choice = get_user_input(
                        "Select internal basis set", internal_basis_options, "7"
                    )
                    options["basis_set"] = internal_basis_options[basis_choice]
                    options["basis_set_type"] = "INTERNAL"
                else:
                    options["basis_set_type"] = "EXTERNAL"
                    print("\nExternal basis set options:")
                    print("1: DZVP-REV2 (./full.basis.doublezeta/)")
                    print("2: TZVP-REV2 (./full.basis.triplezeta/)")
                    print("3: Custom path")
                    print("4: Use basis from original file")

                    external_choice = get_user_input(
                        "Select external basis set",
                        {"1": "DZ", "2": "TZ", "3": "Custom", "4": "Original"},
                        "4" if not shared_mode else "2",
                    )

                    if external_choice == "1":
                        options["basis_set_path"] = "./full.basis.doublezeta/"
                    elif external_choice == "2":
                        options["basis_set_path"] = "./full.basis.triplezeta/"
                    elif external_choice == "3":
                        options["basis_set_path"] = input(
                            "Enter path to external basis set directory: "
                        ).strip()
                    else:
                        # Use external basis from original file
                        options["use_original_external_basis"] = True

        # DFT grid
        if options.get("functional") and options["functional"] not in [
            "HF",
            "RHF",
            "UHF",
        ]:
            change_grid = yes_no_prompt(
                f"Change DFT integration grid (current: {options.get('dft_grid', 'XLGRID')})?",
                "no",
            )
            if change_grid:
                grid_choice = get_user_input(
                    "Select DFT integration grid", DFT_GRIDS, "4"
                )
                options["dft_grid"] = DFT_GRIDS[grid_choice]

        # Spin polarization - UPDATED DEFAULT TO YES
        options["spin_polarized"] = yes_no_prompt(
            "Use spin-polarized calculation?",
            "yes",  # Changed default to "yes"
        )

        # Fermi smearing
        options["smearing"] = yes_no_prompt(
            "Use Fermi surface smearing for metallic systems?",
            "yes" if options.get("smearing") else "no",
        )

        if options["smearing"]:
            default_width = options.get("smearing_width", 0.01)
            width = input(
                f"Enter smearing width in hartree (recommended: 0.001-0.02, default {default_width}): "
            ).strip()
            options["smearing_width"] = float(width) if width else default_width

        # Tolerances (if not already set by FREQ)
        if options["calculation_type"] != "FREQ":
            change_tol = yes_no_prompt("Change tolerance settings?", "no")
            if change_tol:
                if "tolerances" not in options:
                    options["tolerances"] = {}

                tolinteg = input(
                    "Enter TOLINTEG values (5 integers, default '7 7 7 7 14'): "
                ).strip()
                options["tolerances"]["TOLINTEG"] = (
                    tolinteg if tolinteg else "7 7 7 7 14"
                )

                toldee = input("Enter TOLDEE value (integer, default 7): ").strip()
                options["tolerances"]["TOLDEE"] = int(toldee) if toldee else 7

        # SCF settings
        change_scf = yes_no_prompt("Change SCF settings?", "no")
        if change_scf:
            scf_methods = {"1": "DIIS", "2": "ANDERSON", "3": "BROYDEN"}
            scf_choice = get_user_input("Select SCF method", scf_methods, "1")

            if "scf_settings" not in options:
                options["scf_settings"] = {}

            options["scf_settings"]["method"] = scf_methods[scf_choice]

            maxcycle = input("Enter SCF MAXCYCLE (default 800): ").strip()
            options["scf_settings"]["maxcycle"] = int(maxcycle) if maxcycle else 800

            fmixing = input("Enter FMIXING percentage (default 30): ").strip()
            options["scf_settings"]["fmixing"] = int(fmixing) if fmixing else 30

    # Symmetry handling section
    if options.get("dimensionality") != "MOLECULE":
        if shared_mode:
            # In shared mode, ask generically without specific counts
            print("\nSymmetry handling for all files:")
            sym_options = {
                "1": "Write only unique atoms (asymmetric unit) when available",
                "2": "Write all atoms",
            }

            sym_choice = get_user_input(
                "How should atoms be written in the new inputs?",
                sym_options,
                "1",  # Default to writing only unique atoms
            )

            options["write_only_unique"] = sym_choice == "1"
        else:
            # In single file mode, show specific information
            has_symmetry_info = any(
                coord.get("is_unique") is not None
                for coord in options.get("coordinates", [])
            )

            if has_symmetry_info:
                unique_count = sum(
                    1
                    for coord in options["coordinates"]
                    if coord.get("is_unique", True)
                )
                total_count = len(options["coordinates"])

                print(f"\nSymmetry information detected:")
                print(f"  Unique atoms (T): {unique_count}")
                print(f"  Total atoms: {total_count}")

                sym_options = {
                    "1": "Write only unique atoms (asymmetric unit)",
                    "2": "Write all atoms",
                }

                sym_choice = get_user_input(
                    "How should atoms be written in the new input?",
                    sym_options,
                    "1",  # Default to writing only unique atoms
                )

                options["write_only_unique"] = sym_choice == "1"
            else:
                options["write_only_unique"] = False
    else:
        options["write_only_unique"] = False

    return options


def generate_unit_cell_line(spacegroup, cell_params, dimensionality):
    """Generate the unit cell line for CRYSTAL23 input"""
    if dimensionality == "MOLECULE":
        return ""  # No unit cell for molecules

    a, b, c, alpha, beta, gamma = [float(x) for x in cell_params[:6]]

    if dimensionality == "SLAB":
        return f"{a:.8f} {b:.8f} {gamma:.6f}"
    elif dimensionality == "POLYMER":
        return f"{a:.8f}"
    elif dimensionality == "CRYSTAL":
        if spacegroup >= 1 and spacegroup <= 2:  # Triclinic
            return f"{a:.8f} {b:.8f} {c:.8f} {alpha:.6f} {beta:.6f} {gamma:.6f}"
        elif spacegroup >= 3 and spacegroup <= 15:  # Monoclinic
            return f"{a:.8f} {b:.8f} {c:.8f} {beta:.6f}"
        elif spacegroup >= 16 and spacegroup <= 74:  # Orthorhombic
            return f"{a:.8f} {b:.8f} {c:.8f}"
        elif spacegroup >= 75 and spacegroup <= 142:  # Tetragonal
            return f"{a:.8f} {c:.8f}"
        elif spacegroup >= 143 and spacegroup <= 167:  # Trigonal
            return f"{a:.8f} {c:.8f}"
        elif spacegroup >= 168 and spacegroup <= 194:  # Hexagonal
            return f"{a:.8f} {c:.8f}"
        elif spacegroup >= 195 and spacegroup <= 230:  # Cubic
            return f"{a:.8f}"
        else:
            raise ValueError(f"Invalid space group: {spacegroup}")

    return ""


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


def write_d12_file(output_file, geometry_data, settings, external_basis_data=None):
    """Write new D12 file with optimized geometry and settings"""

    with open(output_file, "w") as f:
        # Title
        title = output_file.replace(".d12", "")
        f.write(f"{title}\n")

        # Structure section
        dimensionality = settings.get("dimensionality", "CRYSTAL")
        f.write(f"{dimensionality}\n")

        if dimensionality == "CRYSTAL":
            # Handle space groups with multiple origins
            spacegroup = settings.get("spacegroup", 1)
            origin_setting = settings.get("origin_setting", "0 0 0")

            # Check if this space group has special origin settings
            if spacegroup in MULTI_ORIGIN_SPACEGROUPS:
                spg_info = MULTI_ORIGIN_SPACEGROUPS[spacegroup]
                f.write(f"{spg_info['crystal_code']}\n")
            else:
                f.write(f"{origin_setting}\n")

            f.write(f"{spacegroup}\n")
        elif dimensionality in ["SLAB", "POLYMER"]:
            f.write(f"{settings.get('spacegroup', 1)}\n")
        elif dimensionality == "MOLECULE":
            f.write("1\n")  # C1 symmetry

        # Unit cell parameters (if not molecule)
        if dimensionality != "MOLECULE" and geometry_data.get("conventional_cell"):
            cell_line = generate_unit_cell_line(
                settings.get("spacegroup", 1),
                geometry_data["conventional_cell"],
                dimensionality,
            )
            if cell_line:
                f.write(f"{cell_line}\n")

        # Atomic coordinates - filter based on symmetry preference
        coords = geometry_data["coordinates"]

        # Filter coordinates if requested
        if settings.get("write_only_unique", False):
            coords_to_write = [c for c in coords if c.get("is_unique", True)]
        else:
            coords_to_write = coords

        f.write(f"{len(coords_to_write)}\n")

        for atom in coords_to_write:
            atom_num = int(atom["atom_number"])
            # Add 200 to atomic number if ECP is required for external basis sets
            if (
                settings.get("basis_set_type") == "EXTERNAL"
                and atom_num in ECP_ELEMENTS_EXTERNAL
            ):
                atom_num += 200
            elif settings.get("basis_set_type") == "INTERNAL" and settings.get(
                "basis_set"
            ):
                # Check if element needs ECP in this internal basis set
                basis_set = settings["basis_set"]
                if basis_set in INTERNAL_BASIS_SETS:
                    ecp_elements = INTERNAL_BASIS_SETS[basis_set].get(
                        "ecp_elements", []
                    )
                    if int(atom["atom_number"]) in ecp_elements:
                        atom_num += 200

            symbol = ATOMIC_NUMBER_TO_SYMBOL.get(int(atom["atom_number"]), "X")
            f.write(
                f"{atom_num} {atom['x']} {atom['y']} {atom['z']} Biso 1.000000 {symbol}\n"
            )

        # Calculation-specific section
        if settings["calculation_type"] == "OPT":
            f.write("OPTGEOM\n")
            f.write(f"{settings.get('optimization_type', 'FULLOPTG')}\n")

            opt_settings = settings.get("optimization_settings", DEFAULT_OPT_SETTINGS)
            f.write("MAXCYCLE\n")
            f.write(f"{opt_settings.get('MAXCYCLE', 800)}\n")
            f.write("TOLDEG\n")
            f.write(f"{format_crystal_float(opt_settings.get('TOLDEG', 0.00003))}\n")
            f.write("TOLDEX\n")
            f.write(f"{format_crystal_float(opt_settings.get('TOLDEX', 0.00012))}\n")
            f.write("TOLDEE\n")
            f.write(f"{opt_settings.get('TOLDEE', 7)}\n")

            if "MAXTRADIUS" in opt_settings:
                f.write("MAXTRADIUS\n")
                f.write(f"{format_crystal_float(opt_settings['MAXTRADIUS'])}\n")

            f.write("ENDOPT\n")
        elif settings["calculation_type"] == "FREQ":
            f.write("FREQCALC\n")
            freq_settings = settings.get("freq_settings", DEFAULT_FREQ_SETTINGS)
            f.write("NUMDERIV\n")
            f.write(f"{freq_settings.get('NUMDERIV', 2)}\n")
            f.write("END\n")

        f.write("END\n")

        # Handle basis sets and method section
        functional = settings.get("functional", "")
        method = "HF" if functional in ["HF", "RHF", "UHF"] else "DFT"

        # Handle 3C methods and basis sets
        if functional in ["HF-3C", "HFsol-3C"]:
            # These are HF methods with corrections, write basis set but no DFT block
            f.write("BASISSET\n")
            f.write(f"{settings['basis_set']}\n")
            f.write("END\n")

            # Add 3C corrections
            if functional == "HF-3C":
                f.write("HF3C\n")
                f.write("END\n")
            elif functional == "HFsol-3C":
                f.write("HFSOL3C\n")
                f.write("END\n")
        elif functional in ["PBEh-3C", "HSE-3C", "B97-3C", "PBEsol0-3C", "HSEsol-3C"]:
            # DFT 3C methods
            f.write("BASISSET\n")
            f.write(f"{settings['basis_set']}\n")
            f.write("END\n")

            f.write("DFT\n")
            if settings.get("spin_polarized"):
                f.write("SPIN\n")

            # Write 3C method (remove hyphen for CRYSTAL23)
            f.write(f"{functional.replace('-', '')}\n")
            f.write("ENDDFT\n")
        else:
            # Standard basis set and method handling
            if settings.get("basis_set_type") == "EXTERNAL":
                # Write external basis set data
                if settings.get("use_original_external_basis") and external_basis_data:
                    # Use the external basis from the original file
                    for line in external_basis_data:
                        f.write(f"{line}\n")
                elif settings.get("basis_set_path"):
                    # Read basis sets from specified path
                    f.write(
                        f"# External basis set from: {settings['basis_set_path']}\n"
                    )
                    unique_atoms = set()
                    for atom in coords:
                        unique_atoms.add(int(atom["atom_number"]))

                    # Read basis set files
                    for atom_num in sorted(unique_atoms):
                        basis_file = os.path.join(
                            settings["basis_set_path"], str(atom_num)
                        )
                        if os.path.exists(basis_file):
                            with open(basis_file, "r") as bf:
                                f.write(bf.read())
                        else:
                            print(
                                f"Warning: Basis set file not found for element {atom_num}"
                            )

                f.write("99 0\n")
                f.write("END\n")
            else:
                # Internal basis set
                f.write("BASISSET\n")
                f.write(f"{settings.get('basis_set', 'POB-TZVP-REV2')}\n")
                f.write("END\n")

            # Write method section
            if method == "HF":
                # Handle HF methods
                if functional == "UHF":
                    f.write("UHF\n")
                # RHF is default, no keyword needed
            else:
                # Write DFT section
                f.write("DFT\n")

                if settings.get("spin_polarized"):
                    f.write("SPIN\n")

                # Write functional
                if functional == "mPW1PW91" and settings.get("dispersion"):
                    f.write("PW1PW-D3\n")
                else:
                    # Standard functional
                    if settings.get("dispersion") and functional in D3_FUNCTIONALS:
                        f.write(f"{functional}-D3\n")
                    else:
                        f.write(f"{functional}\n")

                # Add DFT grid if specified
                if settings.get("dft_grid") and settings["dft_grid"] != "DEFAULT":
                    f.write(f"{settings['dft_grid']}\n")

                f.write("ENDDFT\n")

        # SCF parameters
        # Tolerances
        tolerances = settings.get("tolerances", DEFAULT_TOLERANCES)
        f.write("TOLINTEG\n")
        f.write(f"{tolerances.get('TOLINTEG', '7 7 7 7 14')}\n")
        f.write("TOLDEE\n")
        f.write(f"{tolerances.get('TOLDEE', 7)}\n")

        # K-points (for periodic systems)
        if dimensionality != "MOLECULE":
            if settings.get("k_points"):
                # Use extracted k-points
                f.write("SHRINK\n")
                f.write("0 24\n")  # Default IS value
                f.write(f"{settings['k_points']}\n")
            elif geometry_data.get("conventional_cell"):
                # Generate k-points based on cell size
                a, b, c = [float(x) for x in geometry_data["conventional_cell"][:3]]
                ka, kb, kc = generate_k_points(
                    a, b, c, dimensionality, settings.get("spacegroup", 1)
                )
                n_shrink = max(ka, kb, kc) * 2

                f.write("SHRINK\n")
                f.write(f"0 {n_shrink}\n")

                if dimensionality == "CRYSTAL":
                    f.write(f"{ka} {kb} {kc}\n")
                elif dimensionality == "SLAB":
                    f.write(f"{ka} {kb} 1\n")
                elif dimensionality == "POLYMER":
                    f.write(f"{ka} 1 1\n")
            else:
                # Default k-points
                f.write("SHRINK\n")
                f.write("0 24\n")
                if dimensionality == "CRYSTAL":
                    f.write("8 8 8\n")
                elif dimensionality == "SLAB":
                    f.write("8 8 1\n")
                elif dimensionality == "POLYMER":
                    f.write("8 1 1\n")

        # Fermi smearing
        if settings.get("smearing"):
            f.write("SMEAR\n")
            f.write(f"{settings.get('smearing_width', 0.01):.6f}\n")

        # SCF settings
        f.write("SCFDIR\n")

        # Add BIPOSIZE and EXCHSIZE for large systems
        if len(coords) > 5:
            f.write("BIPOSIZE\n")
            f.write("110000000\n")
            f.write("EXCHSIZE\n")
            f.write("110000000\n")

        # SCF convergence
        scf_settings = settings.get("scf_settings", {})
        f.write("MAXCYCLE\n")
        f.write(f"{scf_settings.get('maxcycle', 800)}\n")

        f.write("FMIXING\n")
        f.write(f"{scf_settings.get('fmixing', 30)}\n")

        scf_method = scf_settings.get("method", "DIIS")
        f.write(f"{scf_method}\n")

        if scf_method == "DIIS":
            f.write("HISTDIIS\n")
            f.write("100\n")

        # Print options
        f.write("PPAN\n")  # Print Mulliken population analysis

        # End of input
        f.write("END\n")


def process_files(output_file, input_file=None, shared_settings=None):
    """Process CRYSTAL output and input files

    Args:
        output_file: Path to .out file
        input_file: Path to .d12 file (optional)
        shared_settings: Pre-defined settings to use (optional)

    Returns:
        tuple: (success, settings_used)
    """

    # Parse output file
    print(f"\nParsing output file: {output_file}")
    out_parser = CrystalOutputParser(output_file)
    try:
        out_data = out_parser.parse()
    except Exception as e:
        print(f"Error parsing output file: {e}")
        return False, None

    # Parse input file if provided
    settings = out_data.copy()
    external_basis_data = []

    if input_file and os.path.exists(input_file):
        print(f"Parsing input file: {input_file}")
        in_parser = CrystalInputParser(input_file)
        try:
            in_data = in_parser.parse()

            # Merge data, preferring output file data but filling in gaps
            for key, value in in_data.items():
                if key not in settings or settings[key] is None:
                    settings[key] = value
                elif key == "scf_settings":
                    # Merge SCF settings
                    if "scf_settings" not in settings:
                        settings["scf_settings"] = {}
                    settings["scf_settings"].update(value)

            # Store external basis data
            external_basis_data = in_data.get("external_basis_data", [])
        except Exception as e:
            print(f"Warning: Error parsing input file: {e}")
            print("Continuing with output file data only")

    # Set defaults if not found
    if not settings.get("spacegroup"):
        if settings["dimensionality"] == "MOLECULE":
            settings["spacegroup"] = 1
        else:
            print("Warning: Space group not found. Defaulting to P1")
            settings["spacegroup"] = 1

    if not settings.get("basis_set"):
        settings["basis_set"] = "POB-TZVP-REV2"
        settings["basis_set_type"] = "INTERNAL"

    # Set default tolerances if not found
    if not settings.get("tolerances"):
        settings["tolerances"] = DEFAULT_TOLERANCES.copy()

    # Set default SCF settings if not found
    if not settings.get("scf_settings"):
        settings["scf_settings"] = {
            "method": "DIIS",
            "maxcycle": 800,
            "fmixing": 30,
        }

    # Get user options or use shared settings
    if shared_settings:
        # Merge shared settings with current settings
        options = settings.copy()
        # Override with shared settings (except geometry-specific data)
        for key, value in shared_settings.items():
            if key not in [
                "coordinates",
                "primitive_cell",
                "conventional_cell",
                "spacegroup",
                "dimensionality",
                "origin_setting",
            ]:
                options[key] = value
    else:
        options = get_calculation_options(settings)

    # Create output filename
    base_name = os.path.splitext(output_file)[0]
    calc_type = options["calculation_type"]
    functional = options.get("functional", "HF")
    if options.get("dispersion"):
        functional += "-D3"

    new_filename = f"{base_name}_{calc_type.lower()}_{functional}_optimized.d12"

    # Write new D12 file
    print(f"\nWriting new D12 file: {new_filename}")
    write_d12_file(new_filename, out_data, options, external_basis_data)

    print(f"\nSuccessfully created {new_filename}")

    return True, options


def find_file_pairs(directory):
    """Find matching .out and .d12 file pairs in a directory

    Returns:
        list: List of tuples (out_file, d12_file or None)
    """
    pairs = []

    # Find all .out files
    out_files = [f for f in os.listdir(directory) if f.endswith(".out")]

    for out_file in out_files:
        base_name = out_file[:-4]  # Remove .out extension
        d12_file = f"{base_name}.d12"

        full_out_path = os.path.join(directory, out_file)
        full_d12_path = (
            os.path.join(directory, d12_file)
            if os.path.exists(os.path.join(directory, d12_file))
            else None
        )

        pairs.append((full_out_path, full_d12_path))

    return pairs


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Convert CRYSTAL17/23 optimization output to new D12 input files"
    )
    parser.add_argument("--out-file", type=str, help="CRYSTAL output file (.out)")
    parser.add_argument(
        "--d12-file", type=str, help="Original CRYSTAL input file (.d12)"
    )
    parser.add_argument(
        "--directory",
        type=str,
        default=".",
        help="Directory containing files (default: current directory)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for new D12 files (default: same as input)",
    )
    parser.add_argument(
        "--shared-settings",
        action="store_true",
        help="Apply the same calculation settings to all files",
    )
    parser.add_argument(
        "--save-options", action="store_true", help="Save options to file"
    )
    parser.add_argument(
        "--options-file",
        type=str,
        default="crystal_opt_settings.json",
        help="File to save/load options",
    )

    args = parser.parse_args()

    print("CRYSTAL17/23 Optimization Output to D12 Converter")
    print("=" * 55)
    print("Enhanced version matching NewCifToD12.py configurations")
    print("New entirely reworked script by Marcus Djokic")
    print(
        "Based on old versions by Wangwei Lan, Kevin Lucht, Danny Maldonado, Marcus Djokic"
    )
    print("")

    # Create output directory if specified
    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Single file processing
    if args.out_file:
        if not os.path.exists(args.out_file):
            print(f"Error: Output file {args.out_file} not found")
            return

        success, options = process_files(args.out_file, args.d12_file)

        if success and args.save_options:
            with open(args.options_file, "w") as f:
                # Convert non-serializable items
                save_options = {}
                for k, v in options.items():
                    if k not in ["coordinates", "primitive_cell", "conventional_cell"]:
                        save_options[k] = v
                json.dump(save_options, f, indent=2)
            print(f"Settings saved to {args.options_file}")

    else:
        # Directory processing
        file_pairs = find_file_pairs(args.directory)

        if not file_pairs:
            print(f"No .out files found in {args.directory}")
            return

        print(f"Found {len(file_pairs)} output file(s) to process")

        # If shared settings requested, get them once
        shared_settings = None
        if args.shared_settings and len(file_pairs) > 1:
            print("\n" + "=" * 70)
            print("SHARED SETTINGS MODE")
            print("=" * 70)
            print("Define calculation settings to apply to all files.")
            print(
                "Note: Geometry, symmetry, and space group info will be preserved from each file."
            )

            # Use first file as template for getting settings
            first_out, first_d12 = file_pairs[0]
            print(f"\nUsing {os.path.basename(first_out)} as template for settings...")

            # Parse first file to get baseline settings
            out_parser = CrystalOutputParser(first_out)
            try:
                out_data = out_parser.parse()
                settings = out_data.copy()

                if first_d12:
                    in_parser = CrystalInputParser(first_d12)
                    try:
                        in_data = in_parser.parse()
                        for key, value in in_data.items():
                            if key not in settings or settings[key] is None:
                                settings[key] = value
                    except:
                        pass

                # Get shared settings
                shared_settings = get_calculation_options(settings, shared_mode=True)

                print("\n" + "=" * 70)
                print("Shared settings defined. These will be applied to all files.")
                print("=" * 70)

            except Exception as e:
                print(f"Error getting shared settings: {e}")
                return

        # Process all file pairs
        success_count = 0
        for out_file, d12_file in file_pairs:
            print(f"\n{'=' * 70}")
            print(f"Processing: {os.path.basename(out_file)}")
            if d12_file:
                print(f"With input: {os.path.basename(d12_file)}")
            else:
                print("No corresponding .d12 file found")
            print("=" * 70)

            success, options = process_files(out_file, d12_file, shared_settings)
            if success:
                success_count += 1

        print(f"\n{'=' * 70}")
        print(
            f"Processing complete: {success_count}/{len(file_pairs)} files processed successfully"
        )
        print("=" * 70)

        # Save options if requested
        if args.save_options and shared_settings:
            with open(args.options_file, "w") as f:
                save_options = {}
                for k, v in shared_settings.items():
                    if k not in ["coordinates", "primitive_cell", "conventional_cell"]:
                        save_options[k] = v
                json.dump(save_options, f, indent=2)
            print(f"\nShared settings saved to {args.options_file}")


if __name__ == "__main__":
    main()
