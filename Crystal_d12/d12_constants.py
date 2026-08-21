#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D12 Constants and Configuration Module for CRYSTAL23
----------------------------------------------------
This module contains all constants, data dictionaries, and configuration
data used across the D12 creation system. It consolidates constants from:
- d12creation.py
- d12_config_common.py

This centralization improves maintainability and reduces duplication.

Author: Marcus Djokic
Institution: Michigan State University, Mendoza Group
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple


# ============================================================
# Element Data
# ============================================================

@dataclass
class Element:
    """Represents a chemical element with its properties"""
    symbol: str
    number: int
    mass: float
    
    def __str__(self):
        return self.symbol
    
    def __repr__(self):
        return f"Element({self.symbol}, {self.number}, {self.mass})"


# Element symbols by atomic number
ELEMENT_SYMBOLS = {
    1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne',
    11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca',
    21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn',
    31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr', 37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr',
    41: 'Nb', 42: 'Mo', 43: 'Tc', 44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn',
    51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 55: 'Cs', 56: 'Ba', 57: 'La', 58: 'Ce', 59: 'Pr', 60: 'Nd',
    61: 'Pm', 62: 'Sm', 63: 'Eu', 64: 'Gd', 65: 'Tb', 66: 'Dy', 67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb',
    71: 'Lu', 72: 'Hf', 73: 'Ta', 74: 'W', 75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au', 80: 'Hg',
    81: 'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At', 86: 'Rn', 87: 'Fr', 88: 'Ra', 89: 'Ac', 90: 'Th',
    91: 'Pa', 92: 'U', 93: 'Np', 94: 'Pu', 95: 'Am', 96: 'Cm', 97: 'Bk', 98: 'Cf', 99: 'Es', 100: 'Fm',
    101: 'Md', 102: 'No', 103: 'Lr', 104: 'Rf', 105: 'Db', 106: 'Sg', 107: 'Bh', 108: 'Hs', 109: 'Mt',
    110: 'Ds', 111: 'Rg', 112: 'Cn', 113: 'Nh', 114: 'Fl', 115: 'Mc', 116: 'Lv', 117: 'Ts', 118: 'Og'
}

# Reverse mapping: symbol -> atomic number
SYMBOL_TO_NUMBER = {v: k for k, v in ELEMENT_SYMBOLS.items()}

# Reverse mapping for convenience
ATOMIC_NUMBER_TO_SYMBOL = ELEMENT_SYMBOLS  # Maps atomic number to symbol


# ============================================================
# High-Symmetry k-point Paths for Band Structure Calculations
# ============================================================

# High-symmetry paths for different crystal systems
# Based on standard conventions (Setyawan & Curtarolo, Comp. Mat. Sci. 49, 299 (2010))
HIGH_SYMMETRY_PATHS = {
    "cubic_fc": {  # Face-centered cubic (FCC)
        "labels": ["X", "G", "L", "W", "G"],
        "label_path": [
            "X G",
            "G L", 
            "L W",
            "W G"
        ],
        "coordinates": {
            "G": [0.0, 0.0, 0.0],    # Gamma
            "X": [0.5, 0.0, 0.5],
            "L": [0.5, 0.5, 0.5],
            "W": [0.5, 0.25, 0.75],
            "K": [0.375, 0.375, 0.75],
            "U": [0.625, 0.25, 0.625]
        },
        "coord_path": [
            [6, 0, 6, 0, 0, 0],      # X → Γ
            [0, 0, 0, 6, 6, 6],      # Γ → L
            [6, 6, 6, 6, 3, 9],      # L → W
            [6, 3, 9, 0, 0, 0]       # W → Γ
        ]
    },
    "cubic_bc": {  # Body-centered cubic (BCC)
        "labels": ["G", "H", "N", "G", "P", "H"],
        "label_path": [
            "G H",
            "H N",
            "N G",
            "G P",
            "P H"
        ],
        "coordinates": {
            "G": [0.0, 0.0, 0.0],
            "H": [0.5, -0.5, 0.5],
            "P": [0.25, 0.25, 0.25],
            "N": [0.0, 0.0, 0.5]
        },
        "coord_path": [
            [0, 0, 0, 4, -4, 4],     # Γ → H
            [4, -4, 4, 0, 0, 4],     # H → N
            [0, 0, 4, 0, 0, 0],      # N → Γ
            [0, 0, 0, 2, 2, 2],      # Γ → P
            [2, 2, 2, 4, -4, 4]      # P → H
        ]
    },
    "cubic_simple": {  # Simple cubic
        "labels": ["G", "X", "M", "G", "R", "X"],
        "label_path": [
            "G X",
            "X M",
            "M G",
            "G R",
            "R X",
            "M R"
        ],
        "coordinates": {
            "G": [0.0, 0.0, 0.0],
            "X": [0.5, 0.0, 0.0],
            "M": [0.5, 0.5, 0.0],
            "R": [0.5, 0.5, 0.5]
        },
        "coord_path": [
            [0, 0, 0, 2, 0, 0],      # Γ → X
            [2, 0, 0, 2, 2, 0],      # X → M
            [2, 2, 0, 0, 0, 0],      # M → Γ
            [0, 0, 0, 2, 2, 2],      # Γ → R
            [2, 2, 2, 2, 0, 0],      # R → X
            [2, 2, 0, 2, 2, 2]       # M → R
        ]
    },
    "hexagonal": {  # Hexagonal
        "labels": ["G", "M", "K", "G", "A", "L", "H", "A"],
        "label_path": [
            "G M",
            "M K",
            "K G",
            "G A",
            "A L",
            "L H",
            "H A"
        ],
        "coordinates": {
            "G": [0.0, 0.0, 0.0],
            "M": [0.5, 0.0, 0.0],
            "K": [1.0/3.0, 1.0/3.0, 0.0],
            "A": [0.0, 0.0, 0.5],
            "L": [0.5, 0.0, 0.5],
            "H": [1.0/3.0, 1.0/3.0, 0.5]
        },
        "coord_path": [
            [0, 0, 0, 3, 0, 0],      # Γ → M
            [3, 0, 0, 2, 2, 0],      # M → K
            [2, 2, 0, 0, 0, 0],      # K → Γ
            [0, 0, 0, 0, 0, 3],      # Γ → A
            [0, 0, 3, 3, 0, 3],      # A → L
            [3, 0, 3, 2, 2, 3],      # L → H
            [2, 2, 3, 0, 0, 3]       # H → A
        ]
    },
    "tetragonal": {  # Tetragonal
        "labels": ["G", "X", "M", "G", "Z", "R", "A", "Z"],
        "label_path": [
            "G X",
            "X M",
            "M G",
            "G Z",
            "Z R",
            "R A",
            "A Z"
        ],
        "coordinates": {
            "G": [0.0, 0.0, 0.0],
            "X": [0.5, 0.0, 0.0],
            "M": [0.5, 0.5, 0.0],
            "Z": [0.0, 0.0, 0.5],
            "R": [0.5, 0.0, 0.5],
            "A": [0.5, 0.5, 0.5]
        },
        "coord_path": [
            [0, 0, 0, 2, 0, 0],      # Γ → X
            [2, 0, 0, 2, 2, 0],      # X → M
            [2, 2, 0, 0, 0, 0],      # M → Γ
            [0, 0, 0, 0, 0, 2],      # Γ → Z
            [0, 0, 2, 2, 0, 2],      # Z → R
            [2, 0, 2, 2, 2, 2],      # R → A
            [2, 2, 2, 0, 0, 2]       # A → Z
        ]
    },
    "orthorhombic": {  # Orthorhombic
        "labels": ["G", "X", "S", "Y", "G", "Z", "U", "R", "T", "Z"],
        "label_path": [
            "G X",
            "X S",
            "S Y", 
            "Y G",
            "G Z",
            "Z U",
            "U R",
            "R T",
            "T Z"
        ],
        "coordinates": {
            "G": [0.0, 0.0, 0.0],
            "X": [0.5, 0.0, 0.0],
            "Y": [0.0, 0.5, 0.0],
            "Z": [0.0, 0.0, 0.5],
            "U": [0.5, 0.0, 0.5],
            "T": [0.0, 0.5, 0.5],
            "S": [0.5, 0.5, 0.0],
            "R": [0.5, 0.5, 0.5]
        }
    },
    "monoclinic": {  # Monoclinic
        "labels": ["G", "Y", "H", "C", "E", "M1", "A", "X", "G"],
        "label_path": [
            "G Y",
            "Y H",
            "H C",
            "C E",
            "E M1",
            "M1 A",
            "A X",
            "X G"
        ],
        "coordinates": {
            "G": [0.0, 0.0, 0.0],
            "Y": [0.0, 0.5, 0.0],
            "H": [0.0, 0.5, 0.5],
            "Z": [0.0, 0.0, 0.5],
            "A": [0.5, 0.0, 0.0],
            "C": [0.5, 0.5, 0.0],
            "D": [0.5, 0.0, 0.5],
            "E": [0.5, 0.5, 0.5]
        }
    },
    "triclinic": {  # Triclinic
        "labels": ["G", "X", "Y", "Z", "G", "R", "S", "T", "U", "V", "W", "G"],
        "label_path": [
            "G X",
            "X Y",
            "Y Z",
            "Z G",
            "G R",
            "R S", 
            "S T",
            "T U",
            "U V",
            "V W",
            "W G"
        ],
        "coordinates": {
            "G": [0.0, 0.0, 0.0],
            "X": [0.5, 0.0, 0.0],
            "Y": [0.0, 0.5, 0.0],
            "Z": [0.0, 0.0, 0.5],
            "R": [0.5, 0.5, 0.0],
            "S": [0.5, 0.0, 0.5],
            "T": [0.0, 0.5, 0.5],
            "U": [0.5, 0.5, 0.5]
        }
    }
}

# Mapping from space group to crystal system/path type
SPACEGROUP_TO_PATH = {
    # Triclinic (1-2)
    1: "triclinic", 2: "triclinic",
    # Monoclinic (3-15)
    **{i: "monoclinic" for i in range(3, 16)},
    # Orthorhombic (16-74)
    **{i: "orthorhombic" for i in range(16, 75)},
    # Tetragonal (75-142)
    **{i: "tetragonal" for i in range(75, 143)},
    # Trigonal (143-167)
    **{i: "hexagonal" for i in range(143, 168)},  # Using hexagonal path for trigonal
    # Hexagonal (168-194)
    **{i: "hexagonal" for i in range(168, 195)},
    # Cubic (195-230)
    **{i: "cubic_fc" if i in [225, 216, 227, 228] else 
       "cubic_bc" if i in [229, 230] else "cubic_simple" 
       for i in range(195, 231)}
}

# ============================================================
# Space Group Data
# ============================================================

# Space group symbols - maps number to Hermann-Mauguin symbol
SPACEGROUP_SYMBOLS = {
    1: 'P1', 2: 'P-1', 3: 'P2', 4: 'P21', 5: 'C2', 6: 'Pm', 7: 'Pc', 8: 'Cm', 9: 'Cc',
    10: 'P2/m', 11: 'P21/m', 12: 'C2/m', 13: 'P2/c', 14: 'P21/c', 15: 'C2/c',
    16: 'P222', 17: 'P2221', 18: 'P21212', 19: 'P212121', 20: 'C2221', 21: 'C222',
    22: 'F222', 23: 'I222', 24: 'I212121', 25: 'Pmm2', 26: 'Pmc21', 27: 'Pcc2',
    28: 'Pma2', 29: 'Pca21', 30: 'Pnc2', 31: 'Pmn21', 32: 'Pba2', 33: 'Pna21',
    34: 'Pnn2', 35: 'Cmm2', 36: 'Cmc21', 37: 'Ccc2', 38: 'Amm2', 39: 'Aem2',
    40: 'Ama2', 41: 'Aea2', 42: 'Fmm2', 43: 'Fdd2', 44: 'Imm2', 45: 'Iba2',
    46: 'Ima2', 47: 'Pmmm', 48: 'Pnnn', 49: 'Pccm', 50: 'Pban', 51: 'Pmma',
    52: 'Pnna', 53: 'Pmna', 54: 'Pcca', 55: 'Pbam', 56: 'Pccn', 57: 'Pbcm',
    58: 'Pnnm', 59: 'Pmmn', 60: 'Pbcn', 61: 'Pbca', 62: 'Pnma', 63: 'Cmcm',
    64: 'Cmce', 65: 'Cmmm', 66: 'Cccm', 67: 'Cmme', 68: 'Ccce', 69: 'Fmmm',
    70: 'Fddd', 71: 'Immm', 72: 'Ibam', 73: 'Ibca', 74: 'Imma', 75: 'P4',
    76: 'P41', 77: 'P42', 78: 'P43', 79: 'I4', 80: 'I41', 81: 'P-4', 82: 'I-4',
    83: 'P4/m', 84: 'P42/m', 85: 'P4/n', 86: 'P42/n', 87: 'I4/m', 88: 'I41/a',
    89: 'P422', 90: 'P4212', 91: 'P4122', 92: 'P41212', 93: 'P4222', 94: 'P42212',
    95: 'P4322', 96: 'P43212', 97: 'I422', 98: 'I4122', 99: 'P4mm', 100: 'P4bm',
    101: 'P42cm', 102: 'P42nm', 103: 'P4cc', 104: 'P4nc', 105: 'P42mc', 106: 'P42bc',
    107: 'I4mm', 108: 'I4cm', 109: 'I41md', 110: 'I41cd', 111: 'P-42m', 112: 'P-42c',
    113: 'P-421m', 114: 'P-421c', 115: 'P-4m2', 116: 'P-4c2', 117: 'P-4b2', 118: 'P-4n2',
    119: 'I-4m2', 120: 'I-4c2', 121: 'I-42m', 122: 'I-42d', 123: 'P4/mmm', 124: 'P4/mcc',
    125: 'P4/nbm', 126: 'P4/nnc', 127: 'P4/mbm', 128: 'P4/mnc', 129: 'P4/nmm',
    130: 'P4/ncc', 131: 'P42/mmc', 132: 'P42/mcm', 133: 'P42/nbc', 134: 'P42/nnm',
    135: 'P42/mbc', 136: 'P42/mnm', 137: 'P42/nmc', 138: 'P42/ncm', 139: 'I4/mmm',
    140: 'I4/mcm', 141: 'I41/amd', 142: 'I41/acd', 143: 'P3', 144: 'P31', 145: 'P32',
    146: 'R3', 147: 'P-3', 148: 'R-3', 149: 'P312', 150: 'P321', 151: 'P3112',
    152: 'P3121', 153: 'P3212', 154: 'P3221', 155: 'R32', 156: 'P3m1', 157: 'P31m',
    158: 'P3c1', 159: 'P31c', 160: 'R3m', 161: 'R3c', 162: 'P-31m', 163: 'P-31c',
    164: 'P-3m1', 165: 'P-3c1', 166: 'R-3m', 167: 'R-3c', 168: 'P6', 169: 'P61',
    170: 'P65', 171: 'P62', 172: 'P64', 173: 'P63', 174: 'P-6', 175: 'P6/m',
    176: 'P63/m', 177: 'P622', 178: 'P6122', 179: 'P6522', 180: 'P6222', 181: 'P6422',
    182: 'P6322', 183: 'P6mm', 184: 'P6cc', 185: 'P63cm', 186: 'P63mc', 187: 'P-6m2',
    188: 'P-6c2', 189: 'P-62m', 190: 'P-62c', 191: 'P6/mmm', 192: 'P6/mcc',
    193: 'P63/mcm', 194: 'P63/mmc', 195: 'P23', 196: 'F23', 197: 'I23', 198: 'P213',
    199: 'I213', 200: 'Pm-3', 201: 'Pn-3', 202: 'Fm-3', 203: 'Fd-3', 204: 'Im-3',
    205: 'Pa-3', 206: 'Ia-3', 207: 'P432', 208: 'P4232', 209: 'F432', 210: 'F4132',
    211: 'I432', 212: 'P4332', 213: 'P4132', 214: 'I4132', 215: 'P-43m', 216: 'F-43m',
    217: 'I-43m', 218: 'P-43n', 219: 'F-43c', 220: 'I-43d', 221: 'Pm-3m', 222: 'Pn-3n',
    223: 'Pm-3n', 224: 'Pn-3m', 225: 'Fm-3m', 226: 'Fm-3c', 227: 'Fd-3m', 228: 'Fd-3c',
    229: 'Im-3m', 230: 'Ia-3d'
}

# Create reverse mapping from space group number to symbol
SPACEGROUP_NUMBER_TO_SYMBOL = SPACEGROUP_SYMBOLS

# Create reverse mapping from symbol to number
SPACEGROUP_SYMBOL_TO_NUMBER = {symbol: number for number, symbol in SPACEGROUP_SYMBOLS.items()}

# Alternative space group notations (including CRYSTAL output format with spaces)
SPACEGROUP_ALTERNATIVES = {
    # Monoclinic unique axis b settings
    "P121": 3, "P1211": 3,
    "P1211": 4, "P1211": 4,
    "C121": 5, "C1211": 5,
    "P1m1": 6, "P11m": 6,
    "P1c1": 7, "P11a": 7, "P11n": 7, "P11b": 7,
    "C1m1": 8, "C11m": 8, "A1m1": 8, "I1m1": 8,
    "C1c1": 9, "C11b": 9, "A1n1": 9, "I1a1": 9, "A1a1": 9, "C1n1": 9, "I1c1": 9, "B11n": 9,
    "P12/m1": 10, "P112/m": 10,
    "P121/m1": 11, "P1121/m": 11,
    "C12/m1": 12, "C112/m": 12, "A12/m1": 12, "I12/m1": 12,
    "P12/c1": 13, "P112/a": 13, "P112/n": 13, "P112/b": 13,
    "P121/c1": 14, "P121/a1": 14, "P121/n1": 14, "P121/b1": 14, "P1121/a": 14, "P1121/n": 14, "P1121/b": 14,
    "C12/c1": 15, "C112/b": 15, "A12/n1": 15, "I12/a1": 15, "A12/a1": 15, "C12/n1": 15, "I12/c1": 15, "B112/n": 15,
    # Orthorhombic
    "Pnm21": 31,
    "Pcm21": 26,
    "Pbn21": 33,
    "Aem2": 39, "Abm2": 39,
    "Aea2": 41, "Aba2": 41,
    "Cmce": 64, "Cmca": 64,
    "Ccce": 68, "Ccca": 68,
    # Tetragonal
    "P-421c": 114, "P-42c": 114,
    # Hexagonal/Trigonal
    "H3": 146, "H-3": 148, "H32": 155, "H3m": 160, "H3c": 161, "H-3m": 166, "H-3c": 167,
    # Origin choice 2
    "P-42m:2": 111, "P-42c:2": 112, "P-421m:2": 113, "P-421c:2": 114,
    "P-4m2:2": 115, "P-4c2:2": 116, "P-4b2:2": 117, "P-4n2:2": 118,
    "P4/mcc:2": 124, "P4/nbm:2": 125, "P4/nnc:2": 126, "P4/mbm:2": 127,
    "P4/mnc:2": 128, "P4/nmm:2": 129, "P4/ncc:2": 130, "P42/mcm:2": 132,
    "P42/nbc:2": 133, "P42/nnm:2": 134, "P42/mbc:2": 135, "P42/mnm:2": 136,
    "P42/nmc:2": 137, "P42/ncm:2": 138, "Pbcn:2": 60,
    # More variations for cubic space groups
    "PM3M": 221, "PM-3M": 221, "Pm-3m": 221,
    "PN3N": 222, "PN-3N": 222, "Pn-3n": 222,
    "PM3N": 223, "PM-3N": 223, "Pm-3n": 223,
    "PN3M": 224, "PN-3M": 224, "Pn-3m": 224,
    "FM3M": 225, "FM-3M": 225, "Fm-3m": 225,
    "FM3C": 226, "FM-3C": 226, "Fm-3c": 226,
    "FD3M": 227, "FD-3M": 227, "Fd-3m": 227,
    "FD3C": 228, "FD-3C": 228, "Fd-3c": 228,
    "IM3M": 229, "IM-3M": 229, "Im-3m": 229,
    "IA3D": 230, "IA-3D": 230, "Ia-3d": 230,
    # Common alternate orthorhombic notations
    "PMC21": 26, "PCA21": 29, "PNA21": 33, "PMN21": 31,
    "CMCM": 63, "CMCE": 64, "CMMM": 65, "CCCM": 66,
    "CMME": 67, "CCCE": 68, "FMMM": 69, "FDDD": 70,
    "IMMM": 71, "IBAM": 72, "IBCA": 73, "IMMA": 74,
    # CRYSTAL output format with spaces (complete set)
    # Triclinic
    "P 1": 1, "P -1": 2,
    # Monoclinic
    "P 2": 3, "P 21": 4, "C 2": 5, "P M": 6, "P C": 7, "C M": 8, "C C": 9,
    "P 2/M": 10, "P 21/M": 11, "C 2/M": 12, "P 2/C": 13, "P 21/C": 14, "C 2/C": 15,
    # Orthorhombic
    "P 2 2 2": 16, "P 2 2 21": 17, "P 21 21 2": 18, "P 21 21 21": 19,
    "C 2 2 21": 20, "C 2 2 2": 21, "F 2 2 2": 22, "I 2 2 2": 23, "I 21 21 21": 24,
    "P M M 2": 25, "P M C 21": 26, "P C C 2": 27, "P M A 2": 28, "P C A 21": 29,
    "P N C 2": 30, "P M N 21": 31, "P B A 2": 32, "P N A 21": 33, "P N N 2": 34,
    "C M M 2": 35, "C M C 21": 36, "C C C 2": 37, "A M M 2": 38, "A E M 2": 39,
    "A M A 2": 40, "A E A 2": 41, "F M M 2": 42, "F D D 2": 43, "I M M 2": 44,
    "I B A 2": 45, "I M A 2": 46, "P M M M": 47, "P N N N": 48, "P C C M": 49,
    "P B A N": 50, "P M M A": 51, "P N N A": 52, "P M N A": 53, "P C C A": 54,
    "P B A M": 55, "P C C N": 56, "P B C M": 57, "P N N M": 58, "P M M N": 59,
    "P B C N": 60, "P B C A": 61, "P N M A": 62, "C M C M": 63, "C M C E": 64,
    "C M M M": 65, "C C C M": 66, "C M M E": 67, "C C C E": 68, "F M M M": 69,
    "F D D D": 70, "I M M M": 71, "I B A M": 72, "I B C A": 73, "I M M A": 74,
    # Old notation (A instead of E) - CRYSTAL often outputs these
    "A B M 2": 39, "A B A 2": 41, "C M C A": 64, "C M M A": 67, "C C C A": 68,
    # Tetragonal
    "P 4": 75, "P 41": 76, "P 42": 77, "P 43": 78, "I 4": 79, "I 41": 80,
    "P -4": 81, "I -4": 82, "P 4/M": 83, "P 42/M": 84, "P 4/N": 85, "P 42/N": 86,
    "I 4/M": 87, "I 41/A": 88, "P 4 2 2": 89, "P 42 1 2": 90, "P 41 2 2": 91,
    "P 41 21 2": 92, "P 42 2 2": 93, "P 42 21 2": 94, "P 43 2 2": 95, "P 43 21 2": 96,
    "I 4 2 2": 97, "I 41 2 2": 98, "P 4 M M": 99, "P 4 B M": 100, "P 42 C M": 101,
    "P 42 N M": 102, "P 4 C C": 103, "P 4 N C": 104, "P 42 M C": 105, "P 42 B C": 106,
    "I 4 M M": 107, "I 4 C M": 108, "I 41 M D": 109, "I 41 C D": 110,
    "P -4 2 M": 111, "P -4 2 C": 112, "P -4 21 M": 113, "P -4 21 C": 114,
    "P -4 M 2": 115, "P -4 C 2": 116, "P -4 B 2": 117, "P -4 N 2": 118,
    "I -4 M 2": 119, "I -4 C 2": 120, "I -4 2 M": 121, "I -4 2 D": 122,
    "P 4/M M M": 123, "P 4/M C C": 124, "P 4/N B M": 125, "P 4/N N C": 126,
    "P 4/M B M": 127, "P 4/M N C": 128, "P 4/N M M": 129, "P 4/N C C": 130,
    "P 42/M M C": 131, "P 42/M C M": 132, "P 42/N B C": 133, "P 42/N N M": 134,
    "P 42/M B C": 135, "P 42/M N M": 136, "P 42/N M C": 137, "P 42/N C M": 138,
    "I 4/M M M": 139, "I 4/M C M": 140, "I 41/A M D": 141, "I 41/A C D": 142,
    # Trigonal
    "P 3": 143, "P 31": 144, "P 32": 145, "R 3": 146, "P -3": 147, "R -3": 148,
    "P 3 1 2": 149, "P 3 2 1": 150, "P 31 1 2": 151, "P 31 2 1": 152,
    "P 32 1 2": 153, "P 32 2 1": 154, "R 3 2": 155, "P 3 M 1": 156, "P 3 1 M": 157,
    "P 3 C 1": 158, "P 3 1 C": 159, "R 3 M": 160, "R 3 C": 161,
    "P -3 1 M": 162, "P -3 1 C": 163, "P -3 M 1": 164, "P -3 C 1": 165,
    "R -3 M": 166, "R -3 C": 167,
    # Hexagonal
    "P 6": 168, "P 61": 169, "P 65": 170, "P 62": 171, "P 64": 172, "P 63": 173,
    "P -6": 174, "P 6/M": 175, "P 63/M": 176, "P 6 2 2": 177, "P 61 2 2": 178,
    "P 65 2 2": 179, "P 62 2 2": 180, "P 64 2 2": 181, "P 63 2 2": 182,
    "P 6 M M": 183, "P 6 C C": 184, "P 63 C M": 185, "P 63 M C": 186,
    "P -6 M 2": 187, "P -6 C 2": 188, "P -6 2 M": 189, "P -6 2 C": 190,
    "P 6/M M M": 191, "P 6/M C C": 192, "P 63/M C M": 193, "P 63/M M C": 194,
    # Cubic
    "P 2 3": 195, "F 2 3": 196, "I 2 3": 197, "P 21 3": 198, "I 21 3": 199,
    "P M 3": 200, "P N 3": 201, "F M 3": 202, "F D 3": 203, "I M 3": 204,
    "P A 3": 205, "I A 3": 206, "P 4 3 2": 207, "P 42 3 2": 208, "F 4 3 2": 209,
    "F 41 3 2": 210, "I 4 3 2": 211, "P 43 3 2": 212, "P 41 3 2": 213, "I 41 3 2": 214,
    "P -4 3 M": 215, "F -4 3 M": 216, "I -4 3 M": 217, "P -4 3 N": 218,
    "F -4 3 C": 219, "I -4 3 D": 220, "P M 3 M": 221, "P N 3 N": 222,
    "P M 3 N": 223, "P N 3 M": 224, "F M 3 M": 225, "F M 3 C": 226,
    "F D 3 M": 227, "F D 3 C": 228, "I M 3 M": 229, "I A 3 D": 230,
}

# Space groups with multiple origin choices
MULTI_ORIGIN_SPACEGROUPS = {
    48: {"name": "Pnnn", "default": "Origin 2", "crystal_code": "0 0 0", "alt": "Origin 1", "alt_crystal_code": "0 0 1"},
    50: {"name": "Pban", "default": "Origin 2", "crystal_code": "0 0 0", "alt": "Origin 1", "alt_crystal_code": "0 0 1"},
    59: {"name": "Pmmn", "default": "Origin 2", "crystal_code": "0 0 0", "alt": "Origin 1", "alt_crystal_code": "0 0 1"},
    60: {"name": "Pbcn", "default": "Origin 2", "crystal_code": "0 0 0", "alt": "Origin 1", "alt_crystal_code": "0 1 0"},
    68: {"name": "Ccce", "default": "Origin 2", "crystal_code": "0 0 0", "alt": "Origin 1", "alt_crystal_code": "0 0 1"},
    70: {"name": "Fddd", "default": "Origin 2", "crystal_code": "0 0 0", "alt": "Origin 1", "alt_crystal_code": "0 0 1"},
    85: {"name": "P4/n", "default": "Origin 2", "crystal_code": "0 0 0", "alt": "Origin 1", "alt_crystal_code": "0 0 1"},
    86: {"name": "P4_2/n", "default": "Origin 2", "crystal_code": "0 0 0", "alt": "Origin 1", "alt_crystal_code": "0 0 1"},
    88: {"name": "I4_1/a", "default": "Origin 2", "crystal_code": "0 0 0", "alt": "Origin 1", "alt_crystal_code": "0 0 1"},
    125: {"name": "P4/nbm", "default": "Origin 2", "crystal_code": "0 0 0", "alt": "Origin 1", "alt_crystal_code": "0 0 1"},
    126: {"name": "P4/nnc", "default": "Origin 2", "crystal_code": "0 0 0", "alt": "Origin 1", "alt_crystal_code": "0 0 1"},
    129: {"name": "P4/nmm", "default": "Origin 2", "crystal_code": "0 0 0", "alt": "Origin 1", "alt_crystal_code": "0 0 1"},
    130: {"name": "P4/ncc", "default": "Origin 2", "crystal_code": "0 0 0", "alt": "Origin 1", "alt_crystal_code": "0 0 1"},
    133: {"name": "P4_2/nbc", "default": "Origin 2", "crystal_code": "0 0 0", "alt": "Origin 1", "alt_crystal_code": "0 0 1"},
    134: {"name": "P4_2/nnm", "default": "Origin 2", "crystal_code": "0 0 0", "alt": "Origin 1", "alt_crystal_code": "0 0 1"},
    137: {"name": "P4_2/nmc", "default": "Origin 2", "crystal_code": "0 0 0", "alt": "Origin 1", "alt_crystal_code": "0 0 1"},
    138: {"name": "P4_2/ncm", "default": "Origin 2", "crystal_code": "0 0 0", "alt": "Origin 1", "alt_crystal_code": "0 0 1"},
    141: {"name": "I4_1/amd", "default": "Origin 2", "crystal_code": "0 0 0", "alt": "Origin 1", "alt_crystal_code": "0 0 1"},
    142: {"name": "I4_1/acd", "default": "Origin 2", "crystal_code": "0 0 0", "alt": "Origin 1", "alt_crystal_code": "0 0 1"},
    201: {"name": "Pn-3", "default": "Origin 2", "crystal_code": "0 0 0", "alt": "Origin 1", "alt_crystal_code": "0 0 1"},
    203: {"name": "Fd-3", "default": "Origin 2", "crystal_code": "0 0 0", "alt": "Origin 1", "alt_crystal_code": "0 0 1"},
    222: {"name": "Pn-3n", "default": "Origin 2", "crystal_code": "0 0 0", "alt": "Origin 1", "alt_crystal_code": "0 0 1"},
    224: {"name": "Pn-3m", "default": "Origin 2", "crystal_code": "0 0 0", "alt": "Origin 1", "alt_crystal_code": "0 0 1"},
    227: {"name": "Fd-3m", "default": "Origin 2", "crystal_code": "0 0 0", "alt": "Origin 1", "alt_crystal_code": "0 0 1"},
    228: {"name": "Fd-3c", "default": "Origin 2", "crystal_code": "0 0 0", "alt": "Origin 1", "alt_crystal_code": "0 0 1"}
}

# Rhombohedral space groups (can be expressed in hexagonal or rhombohedral axes)
RHOMBOHEDRAL_SPACEGROUPS = [146, 148, 155, 160, 161, 166, 167]


# ============================================================
# Layer groups (SLAB) and rod groups (POLYMER)
# ============================================================

# The record after the SLAB keyword is a LAYER group and the record after
# POLYMER a ROD group - CRYSTAL's own IGR numbering, NOT the 3D space group.
# Appendix A.2 (manual page 421) lists the 80 layer groups, Appendix A.3
# (pages 422-424) the 99 rod groups; manual L434 confirms the counts: "230
# space groups, 80 layer groups, 99 rod groups, 45 point groups are available
# (Appendix A)".
#
# Both appendices also print the corresponding 3D space group, which is what
# makes the reverse lookup below possible. A.2's header states the one
# restriction on it: "The number of the space group is written in parentheses
# when the orientation of the symmetry operators does not correspond to the
# first setting in the I. T." A.3 repeats that sentence and adds that "the
# symmetry operators are generated for the space groups (principal axis z) and
# then rotated by 90 degrees through y, to have the polymer axis along x
# (CRYSTAL convention)" - a relabelling of axes, which leaves the
# International Tables NUMBER unchanged.
#
# The reverse lookup is NOT one-to-one and the parentheses alone do not make it
# one. A space group can appear on several rows, at most one of them
# unparenthesised, and those rows are the same group type in different
# orientations: A.2 rows 23 and 27 are both C2v^1 with N = 25 and N = (25) -
# Pmm2 has its 2-fold along the surface normal, P2mm in the plane - and a space
# group number cannot tell them apart. The same holds for A.2 25/(25) siblings
# at N = 28 and N = 51, and for A.3 N = 25 (rows 20, 24, 26) and N = 26 (rows
# 21, 22). So the map keeps a space group only when the appendix lists exactly
# ONE row for it AND that row is unparenthesised.
#
# Even then the map is only a NECESSARY condition. Being unparenthesised is a
# property of CRYSTAL's table, not of the caller's structure: it says the
# table's operators are in the International Tables first setting, and the
# lookup is therefore correct only if the incoming cell is in that setting too,
# with the non-periodic direction along c (SLAB) or the chain along a (POLYMER,
# manual note 18 at L1232-1233). Nothing in this converter enforces or checks
# that - the CIF/spglib path reads only the space-group NUMBER and writes the
# CIF's own cell back out - which is why callers of these maps must also run
# check_layer_group_cell and the origin-freedom guards below.
#
# The rows are transcribed from the PDF appendix; the plain-text dump of the
# manual interleaves the two printed columns of A.2 and cannot be used. The
# space-group column keeps the manual's parentheses verbatim so that the
# first-setting flag is derived from the manual's own notation rather than
# re-asserted here. Layer groups 17 and 18 both print "P2/b11" in A.2, which
# reads as a typographical slip - they are distinguished only by C2h^4/(13)
# against C2h^5/(14) - and are kept exactly as printed. (A.3's row 78 has a
# similar oddity, "P6_6" in the Hermann-Mauguin column against C6^6 and space
# group 173; that column is not carried here, only A.3's "polymer" symbol,
# which reads P6_3.) None of the symbol strings are used to make a decision.

# (IGR, Hermann-Mauguin, Schoenflies, space group as printed)
LAYER_GROUP_ROWS = (
    # Oblique lattices (P)
    (1, "P1", "C1^1", "1"),
    (2, "P-1", "Ci^1", "2"),
    (3, "P112", "C2^1", "(3)"),
    (4, "P11m", "Cs^1", "(6)"),
    (5, "P11a", "Cs^2", "(7)"),
    (6, "P112/m", "C2h^1", "(10)"),
    (7, "P112/a", "C2h^4", "(13)"),
    # Rectangular lattices (P or C)
    (8, "P211", "C2^1", "(3)"),
    (9, "P2_111", "C2^2", "(4)"),
    (10, "C211", "C2^3", "(5)"),
    (11, "Pm11", "Cs^1", "(6)"),
    (12, "Pb11", "Cs^2", "(7)"),
    (13, "Cm11", "Cs^3", "(8)"),
    (14, "P2/m11", "C2h^1", "(10)"),
    (15, "P2_1/m11", "C2h^2", "(11)"),
    (16, "C2/m11", "C2h^3", "(12)"),
    (17, "P2/b11", "C2h^4", "(13)"),
    (18, "P2/b11", "C2h^5", "(14)"),
    (19, "P222", "D2^1", "16"),
    (20, "P22_12", "D2^2", "(17)"),
    (21, "P2_12_12", "D2^3", "18"),
    (22, "C222", "D2^6", "21"),
    (23, "Pmm2", "C2v^1", "25"),
    (24, "Pma2", "C2v^4", "28"),
    (25, "Pba2", "C2v^8", "32"),
    (26, "Cmm2", "C2v^11", "35"),
    (27, "P2mm", "C2v^1", "(25)"),
    (28, "P2_1am", "C2v^2", "(26)"),
    (29, "P2_1ma", "C2v^2", "(26)"),
    (30, "P2mb", "C2v^4", "(28)"),
    (31, "P2_1mn", "C2v^7", "(31)"),
    (32, "P2aa", "C2v^3", "(27)"),
    (33, "P2_1ab", "C2v^5", "(29)"),
    (34, "P2an", "C2v^6", "(30)"),
    (35, "C2mm", "C2v^1", "(38)"),
    (36, "C2mb", "C2v^5", "(39)"),
    (37, "Pmmm", "D2h^1", "47"),
    (38, "Pmam", "D2h^5", "(51)"),
    (39, "Pmma", "D2h^5", "51"),
    (40, "Pmmn", "D2h^13", "59"),
    (41, "Pbam", "D2h^9", "55"),
    (42, "Pmaa", "D2h^3", "(49)"),
    (43, "Pman", "D2h^7", "(53)"),
    (44, "Pbma", "D2h^11", "(57)"),
    (45, "Pbaa", "D2h^8", "(54)"),
    (46, "Pban", "D2h^4", "50"),
    (47, "Cmmm", "D2h^19", "65"),
    (48, "Cmma", "D2h^21", "67"),
    # Square lattices (P)
    (49, "P4", "C4^1", "75"),
    (50, "P-4", "S4^1", "81"),
    (51, "P4/m", "C4h^1", "83"),
    (52, "P4/n", "C4h^3", "85"),
    (53, "P422", "D4^1", "89"),
    (54, "P42_12", "D4^2", "90"),
    (55, "P4mm", "C4v^1", "99"),
    (56, "P4bm", "C4v^2", "100"),
    (57, "P-42m", "D2d^1", "111"),
    (58, "P-42_1m", "D2d^3", "113"),
    (59, "P-4m2", "D2d^5", "115"),
    (60, "P-4b2", "D2d^7", "117"),
    (61, "P4/mmm", "D4h^1", "123"),
    (62, "P4/nbm", "D4h^3", "125"),
    (63, "P4/mbm", "D4h^5", "127"),
    (64, "P4/nmm", "D4h^7", "129"),
    # Hexagonal lattices (P)
    (65, "P3", "C3^1", "143"),
    (66, "P-3", "C3i^1", "147"),
    (67, "P312", "D3^1", "149"),
    (68, "P321", "D3^2", "150"),
    (69, "P3m1", "C3v^1", "156"),
    (70, "P31m", "C3v^2", "157"),
    (71, "P-31m", "D3d^1", "162"),
    (72, "P-3m1", "D3d^3", "164"),
    (73, "P6", "C6^1", "168"),
    (74, "P-6", "C3h^1", "174"),
    (75, "P6/m", "C6h^1", "175"),
    (76, "P622", "D6^1", "177"),
    (77, "P6mm", "C6v^1", "183"),
    (78, "P-6m2", "D3h^1", "187"),
    (79, "P-62m", "D3h^3", "189"),
    (80, "P6/mmm", "D6h^1", "191"),
)

# (IGR, "polymer" symbol along x, Schoenflies, space group as printed).
# A.3 also prints the Hermann-Mauguin symbol in the z-axis setting; it is
# documentation only and is not carried here.
ROD_GROUP_ROWS = (
    (1, "P1", "C1^1", "1"),
    (2, "P-1", "Ci^1", "2"),
    (3, "P211", "C2^1", "(3)"),
    (4, "P2_111", "C2^2", "(4)"),
    (5, "P121", "C2^1", "(3)"),
    (6, "P112", "C2^1", "(3)"),
    (7, "Pm11", "Cs^1", "(6)"),
    (8, "P1m1", "Cs^1", "(6)"),
    (9, "P1a1", "Cs^2", "(7)"),
    (10, "P11m", "Cs^1", "(6)"),
    (11, "P11a", "Cs^2", "(7)"),
    (12, "P2/m11", "C2h^1", "(10)"),
    (13, "P2_1/m11", "C2h^2", "(11)"),
    (14, "P12/m1", "C2h^1", "(10)"),
    (15, "P12/a1", "C2h^4", "(13)"),
    (16, "P112/m", "C2h^1", "(10)"),
    (17, "P112/a", "C2h^4", "(13)"),
    (18, "P222", "D2^1", "16"),
    (19, "P2_122", "D2^2", "17"),
    (20, "P2mm", "C2v^1", "25"),
    (21, "P2_1am", "C2v^2", "26"),
    (22, "P2_1ma", "C2v^2", "(26)"),
    (23, "P2aa", "C2v^3", "27"),
    (24, "Pm2m", "C2v^1", "(25)"),
    (25, "Pm2a", "C2v^4", "(28)"),
    (26, "Pmm2", "C2v^1", "(25)"),
    (27, "Pma2", "C2v^4", "(28)"),
    (28, "Pmmm", "D2h^1", "47"),
    (29, "P2/m2/a2/a", "D2h^3", "49"),
    (30, "P2_1/m2/m2/a", "D2h^5", "(51)"),
    (31, "P2_1/m2/a2/m", "D2h^5", "(51)"),
    (32, "P4", "C4^1", "75"),
    (33, "P4_1", "C4^2", "76"),
    (34, "P4_2", "C4^3", "77"),
    (35, "P4_3", "C4^4", "78"),
    (36, "P-4", "S4^1", "81"),
    (37, "P4/m", "C4h^1", "83"),
    (38, "P4_2/m", "C4h^2", "84"),
    (39, "P422", "D4^1", "89"),
    (40, "P4_122", "D4^3", "91"),
    (41, "P4_222", "D4^5", "93"),
    (42, "P4_322", "D4^7", "95"),
    (43, "P4mm", "C4v^1", "99"),
    (44, "P4_2am", "C4v^3", "101"),
    (45, "P4aa", "C4v^5", "103"),
    (46, "P4_2ma", "C4v^7", "105"),
    (47, "P-42m", "D2d^1", "111"),
    (48, "P-42a", "D2d^2", "112"),
    (49, "P-4m2", "D2d^5", "115"),
    (50, "P-4a2", "D2d^6", "116"),
    (51, "P4/mmm", "D4h^1", "123"),
    (52, "P4/m2/a2/a", "D4h^2", "124"),
    (53, "P4_2/m2/m2/a", "D4h^9", "131"),
    (54, "P4_2/m2/a2/m", "D4h^10", "132"),
    (55, "P3", "C3^1", "143"),
    (56, "P3_1", "C3^2", "144"),
    (57, "P3_2", "C3^3", "145"),
    (58, "P-3", "C3i^1", "147"),
    (59, "P312", "D3^1", "149"),
    (60, "P3_112", "D3^3", "151"),
    (61, "P3_212", "D3^5", "153"),
    (62, "P321", "D3^2", "150"),
    (63, "P3_121", "D3^4", "152"),
    (64, "P3_221", "D3^6", "154"),
    (65, "P3m1", "C3v^1", "156"),
    (66, "P3a1", "C3v^3", "158"),
    (67, "P31m", "C3v^2", "157"),
    (68, "P31a", "C3v^4", "159"),
    (69, "P-31m", "D3d^1", "162"),
    (70, "P-31a", "D3d^2", "163"),
    (71, "P-3m1", "D3d^3", "164"),
    (72, "P-3a1", "D3d^4", "165"),
    (73, "P6", "C6^1", "168"),
    (74, "P6_1", "C6^2", "169"),
    (75, "P6_5", "C6^3", "170"),
    (76, "P6_2", "C6^4", "171"),
    (77, "P6_4", "C6^5", "172"),
    (78, "P6_3", "C6^6", "173"),
    (79, "P-6", "C3h^1", "174"),
    (80, "P6/m", "C6h^1", "175"),
    (81, "P6_3/m", "C6h^2", "176"),
    (82, "P622", "D6^1", "177"),
    (83, "P6_122", "D6^2", "178"),
    (84, "P6_522", "D6^3", "179"),
    (85, "P6_222", "D6^4", "180"),
    (86, "P6_422", "D6^5", "181"),
    (87, "P6_322", "D6^6", "182"),
    (88, "P6mm", "C6v^1", "183"),
    (89, "P6aa", "C6v^2", "184"),
    (90, "P6_3am", "C6v^3", "185"),
    (91, "P6_3ma", "C6v^4", "186"),
    (92, "P-6m2", "D3h^1", "187"),
    (93, "P-6a2", "D3h^2", "188"),
    (94, "P-62m", "D3h^3", "189"),
    (95, "P-62a", "D3h^4", "190"),
    (96, "P6/mmm", "D6h^1", "191"),
    (97, "P6/m2/a2/a", "D6h^2", "192"),
    (98, "P6_3/m2/a2/m", "D6h^3", "193"),
    (99, "P6_3/m2/m2/a", "D6h^4", "194"),
)


def _appendix_a_maps(rows):
    """Split an appendix A.2/A.3 transcription into the two lookups needed.

    Returns (igr_by_spacegroup, igrs_by_spacegroup).

    The second is every row the appendix prints for a space group, in IGR
    order; the first keeps only the space groups with exactly ONE such row
    whose number the manual prints WITHOUT parentheses. Both conditions are
    needed. Parentheses mean "the orientation of the symmetry operators does
    not correspond to the first setting in the I. T." (A.2/A.3 header), so a
    parenthesised row cannot be inverted; and a space group with several rows
    has several orientations of the same group type, which its number cannot
    distinguish (A.2 N = 25 is C2v^1 as both Pmm2 and P2mm).
    """
    candidates = {}
    first_setting = set()
    for igr, _symbol, _schoenflies, number in rows:
        spacegroup = int(number.strip("()"))
        candidates.setdefault(spacegroup, []).append(igr)
        if not number.startswith("("):
            first_setting.add(igr)
    by_spacegroup = {
        spacegroup: igrs[0]
        for spacegroup, igrs in candidates.items()
        if len(igrs) == 1 and igrs[0] in first_setting
    }
    return by_spacegroup, {
        spacegroup: tuple(igrs) for spacegroup, igrs in sorted(candidates.items())
    }


# 3D space group -> layer group (45 of the 230) / rod group (75 of the 230),
# plus every candidate row the appendix prints, used to explain a refusal.
LAYER_GROUP_FROM_SPACEGROUP, LAYER_GROUP_CANDIDATES = _appendix_a_maps(
    LAYER_GROUP_ROWS
)
ROD_GROUP_FROM_SPACEGROUP, ROD_GROUP_CANDIDATES = _appendix_a_maps(ROD_GROUP_ROWS)

# Layer groups in which every symmetry operation preserves the SIGN of z.
#
# A SLAB atom record gives "z in Angstrom, x, y in fractional units" (manual
# L1021-1022, and again at L29023-29024) - z is a Cartesian, NON-periodic
# coordinate measured from the layer group's own origin, unlike the fractional
# z of a 3D deck which is only defined modulo the c translation. Every layer
# group containing an operation that maps z to -z (an inversion centre, a
# mirror or glide perpendicular to z, a 2-fold axis lying in the plane, S4,
# sigma_h) pins that origin at z = 0 and CRYSTAL builds the other half of the
# slab from it: the manual's own diamond (100) deck (L29193-29207) lists five
# atoms at z = 0.44625 .. 4.01625 under layer group 39 for a slab its title
# calls "ten layers slab".
#
# The groups below are the ones with no such operation - C1; the Cn with the
# axis along z (P112, P4, P3, P6); the Cnv with the axis along z (Pmm2, Pma2,
# Pba2, Cmm2, P4mm, P4bm, P3m1, P31m, P6mm); and the Cs whose mirror/glide is
# perpendicular to x and therefore contains z (Pm11, Pb11, Cm11). For these the
# z origin is free and any offset may be written.
#
# For the rest, the offset cannot be recovered from the coordinates either: the
# atoms handed to a converter are an ASYMMETRIC UNIT, which sits on one side of
# the mirror rather than straddling it (diamond's five atoms are all at z > 0,
# and so are corundum's six under layer group 7). A "is it centred on zero"
# test would therefore reject correct input as often as wrong input. The only
# safe automatic answer is to map a 3D space group to a layer group in this set
# and refuse otherwise, leaving the two-sided groups to an explicit request in
# which the caller also asserts where the z origin is.
LAYER_GROUPS_POLAR_IN_Z = frozenset(
    {1, 3, 11, 12, 13, 23, 24, 25, 26, 49, 55, 56, 65, 69, 70, 73, 77}
)

# Rod groups that leave y and z unconstrained.
#
# A POLYMER atom record gives "y,z in Angstrom, x in fractional units" (manual
# L1019-1020): y and z are Cartesian distances from the ROD AXIS. Every rod
# group this converter can auto-map - i.e. every unparenthesised A.3 row other
# than P1, starting with the inversion centre at IGR 2 - has at least one
# operation that fixes the axis at y = z = 0, so the manual's polymer decks
# carry signed coordinates about it - (SN)x "16 0.0 -0.844969 0.0" (L29220),
# the water polymer "1 0.032558 0.836088 -0.400375" and "8 0.5 -1.370589 0."
# (L29234-29239), formamide "8 -7.548E-2 5.302E-3 0.7665" (L29255-29260).
# The claim is deliberately scoped to the auto-mappable rows: it is NOT true of
# A.3 as a whole. IGR 7 (Pm11) is a mirror perpendicular to x and leaves both y
# and z free, and 8-11 each pin only one of the two. All of those are printed
# in parentheses, so they can never be auto-mapped and never reach this set.
# Those same decks show why the offset cannot be checked for either: (SN)x's
# two asymmetric-unit atoms sit at y = -0.844969 and y = 0.667077, nowhere near
# symmetric about the axis, because the screw axis generates the partners.
ROD_GROUPS_FREE_OF_AXIS_OPERATIONS = frozenset({1})

# How far (fractional) a structure's symmetry axis may sit from the in-plane
# origin before an automatically mapped layer group is refused.
#
# CRYSTAL places the layer group's rotation axis at the in-plane origin and
# expands the asymmetric unit about it, so an offset structure expands into a
# different slab without any error. spglib's origin_shift measures the offset;
# a correctly placed structure returns exactly 0.0 (measured on a honeycomb
# with the 6-fold axis on the cell origin), so this only has to absorb
# numerical noise, not real displacement - it is deliberately an order of
# magnitude tighter than the lattice-class tolerances below.
LAYER_GROUP_ORIGIN_TOL_FRAC = 1e-4

# Tolerances for the 2D lattice-class cross-check (check_layer_group_cell).
#
# Angles: real optimized slabs in this project's corpus land a few thousandths
# of a degree off the ideal value - test/SP/4LG_FSI_2x2_AA_opt_sp.d12 carries
# gamma = 120.003561 - so the check must be looser than that while still
# refusing a genuinely different lattice.
LAYER_GROUP_ANGLE_TOL_DEG = 0.05
# Lengths: the same deck has |a - b| / a = 2.1e-5. NOTE this tolerance is NOT
# tied to the spglib symprec that produced the space group in the first place
# (NewCifToD12 default 1e-5, overridable through options["symmetry_tolerance"]),
# so raising symprec far enough to force, say, a hexagonal assignment on a cell
# that is not hexagonal to 1e-3 will be refused here. That is deliberate: the
# deck, not spglib, is what CRYSTAL reads.
LAYER_GROUP_LENGTH_RTOL = 1e-3


# ============================================================
# Basis Set and Element Data
# ============================================================

# Elements with ECPs in DZVP-REV2 and TZVP-REV2 external basis sets
ECP_ELEMENTS_EXTERNAL = [
    # 4th row (all use ECP)
    37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
    # 5th row (all use ECP)
    55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
    71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85,
    # 6th row (all use ECP)
    87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99
    # Note: Tc (43), Kr (36), Xe (54), Rn (86) are full-core in external sets
]

# Internal basis sets available in CRYSTAL
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
        + list(range(72, 85)),
        "all_electron": list(range(1, 19)),  # H to Ar
        "ecp_elements": list(range(19, 36))
        + list(range(37, 54))
        + [55, 56]
        + list(range(72, 85)),
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


# ============================================================
# Functional and Method Data
# ============================================================

# Functional categories with detailed information
FUNCTIONAL_CATEGORIES = {
    "LDA": {
        "name": "LDA/LSD Functionals",
        "description": "Local (Spin) Density Approximation functionals",
        "functionals": ["SVWN", "VBH"],
        "descriptions": {
            "SVWN": "Slater exchange + VWN5 correlation",
            "VBH": "von Barth-Hedin LSD functional"
        }
    },
    "GGA": {
        "name": "GGA Functionals",
        "description": "Generalized Gradient Approximation functionals",
        "functionals": ["BLYP", "PBE", "PBESOL", "PWGGA", "SOGGA", "WCGGA", "B97"],
        "descriptions": {
            "BLYP": "Becke 88 exchange + Lee-Yang-Parr correlation",
            "PBE": "Perdew-Burke-Ernzerhof",
            "PBESOL": "PBE revised for solids",
            "PWGGA": "Perdew-Wang 1991 GGA",
            "SOGGA": "Second-order GGA",
            "WCGGA": "Wu-Cohen GGA",
            "B97": "Becke's 1997 GGA functional"
        }
    },
    "HYBRID": {
        "name": "Hybrid Functionals (including range-separated)",
        "description": "Global and range-separated hybrid functionals",
        "functionals": [
            "B3LYP",
            "B3PW",
            "CAM-B3LYP",
            "PBE0",
            "PBESOL0",
            "PBE0-13",
            "HSE06",
            "HSEsol",
            "mPW1PW91",
            "mPW1K",
            "B1WC",
            "WC1LYP",
            "B97H",
            "wB97",
            "wB97X",
            "SOGGA11X",
            "SC-BLYP",
            "HISS",
            "RSHXLDA",
            "LC-wPBE",
            "LC-wPBEsol",
            "LC-wBLYP",
            "LC-BLYP",
            "LC-PBE",
        ],
        "descriptions": {
            "B3LYP": "Becke 3-parameter hybrid (20% HF)",
            "B3PW": "Becke 3-parameter with PW91 correlation (20% HF)",
            "CAM-B3LYP": "Coulomb-attenuating method B3LYP",
            "PBE0": "PBE hybrid (25% HF)",
            "PBESOL0": "PBEsol hybrid for solids (25% HF)",
            "PBE0-13": "PBE0 with 1/3 HF exchange (33.33% HF)",
            "HSE06": "Heyd-Scuseria-Ernzerhof screened hybrid",
            "HSEsol": "HSE for solids",
            "mPW1PW91": "Modified PW91 hybrid (25% HF)",
            "mPW1K": "Modified PW91 for kinetics (42.8% HF)",
            "B1WC": "One-parameter WC hybrid (16% HF)",
            "WC1LYP": "WC exchange with LYP correlation (16% HF)",
            "B97H": "Re-parameterized B97 hybrid",
            "wB97": "Head-Gordon's range-separated functional",
            "wB97X": "wB97 with short-range HF exchange",
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
        "description": "Include kinetic energy density",
        "functionals": [
            "SCAN", "r2SCAN", "SCAN0", "r2SCANh", "r2SCAN0", "r2SCAN50",
            "M05", "M052X", "M06", "M062X", "M06HF", "M06L", 
            "revM06", "revM06L", "MN15", "MN15L", "B1B95", "mPW1B95", 
            "mPW1B1K", "PW6B95", "PWB6K"
        ],
        "descriptions": {
            "SCAN": "Strongly Constrained and Appropriately Normed",
            "r2SCAN": "Regularized SCAN with improved numerical stability",
            "SCAN0": "SCAN hybrid (25% HF)",
            "r2SCANh": "r2SCAN hybrid (10% HF)",
            "r2SCAN0": "r2SCAN hybrid (25% HF)",
            "r2SCAN50": "r2SCAN hybrid (50% HF)",
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
            "B1B95": "One-parameter hybrid with Becke95 correlation (28% HF)",
            "mPW1B95": "Modified PW91 with B95 correlation (31% HF)",
            "mPW1B1K": "Modified PW91 with B95 correlation (44% HF)",
            "PW6B95": "6-parameter functional (28% HF)",
            "PWB6K": "6-parameter functional for kinetics (46% HF)"
        }
    },
    "3C": {
        "name": "3c Composite Methods (DFT)",
        "description": "DFT composite methods with semi-classical corrections (require specific basis sets)",
        "functionals": ["PBEH3C", "HSE3C", "B973C", "PBESOL03C", "HSESOL3C"],
        "basis_requirements": {
            "PBEH3C": "def2-mSVP",
            "HSE3C": "def2-mSVP",
            "B973C": "mTZVP",
            "PBESOL03C": "SOLDEF2MSVP",
            "HSESOL3C": "SOLDEF2MSVP",
        },
        "descriptions": {
            "PBEH3C": "Modified PBE hybrid (42% HF) with D3 and gCP",
            "HSE3C": "Screened exchange hybrid optimized for molecular solids",
            "B973C": "GGA functional with D3 and SRB corrections",
            "PBESOL03C": "PBEsol0 hybrid for solids with D3 and gCP",
            "HSESOL3C": "HSEsol with semi-classical corrections for solids",
        },
    },
    "HF": {
        "name": "Hartree-Fock Methods",
        "description": "Wave function based methods (no DFT)",
        "functionals": ["RHF", "UHF", "HF3C", "HFSOL3C"],
        "basis_requirements": {"HF3C": "MINIX", "HFSOL3C": "SOLMINIX"},
        "descriptions": {
            "RHF": "Restricted Hartree-Fock (closed shell)",
            "UHF": "Unrestricted Hartree-Fock (open shell)",
            "HF3C": "Minimal basis HF with D3, gCP, and SRB corrections",
            "HFSOL3C": "HF3C revised for inorganic solids",
        },
    }
}

# Functionals supporting D3 dispersion correction
D3_FUNCTIONALS = [
    # GGA
    "BLYP", "PBE", "B97",
    # Hybrid
    "B3LYP", "PBE0", "HSE06", "HSEsol", "mPW1PW91", "LC-wPBE",
    # meta-GGA
    "M06"
]

# Functional keyword mapping for CRYSTAL

# ============================================================
# SCF and DFT Settings
# ============================================================

# Available SCF methods
SCF_METHODS = ["RHF", "UHF", "ROHF"]

# DFT grid options (correct CRYSTAL options from backup)
DFT_GRIDS = {
    "1": "OLDGRID",
    "2": "DEFAULT",  
    "3": "LGRID",
    "4": "XLGRID",
    "5": "XXLGRID",
    "6": "XXXLGRID",
    "7": "HUGEGRID"
}


# ============================================================
# Default Tolerances and Settings
# ============================================================

# Default SCF tolerances
DEFAULT_TOLERANCES = {
    "TOLINTEG": "7 7 7 7 14",
    "TOLDEE": 7,
}

# Default optimization settings
DEFAULT_OPT_SETTINGS = {
    "type": "FULLOPTG",
    "maxcycle": 800,  # Updated to match what's shown in prompts
    "convergence": "Standard",
    "toldeg": 0.0003,
    "toldex": 0.0012,
    "toldee": 7,
}

# Default frequency settings
DEFAULT_FREQ_SETTINGS = {
    "NUMDERIV": 2,
    "TOLINTEG": "9 9 9 11 38",
    "TOLDEE": 11,
}

# Default general settings
DEFAULT_SETTINGS = {
    "dimensionality": "CRYSTAL",
    "k_points": 8,
    "method": "DFT",  # Added for compatibility
    "method_type": "DFT",
    "dft_functional": "HSE06",  # For DFT calculations
    "functional": "HSE06",
    "basis_set": "POB-TZVP-REV2",
    "basis_set_type": "INTERNAL",
    "dft_grid": "XLGRID",
    "use_dispersion": True,  # Added for compatibility 
    "dispersion": True,
    "is_spin_polarized": True,  # Added for compatibility
    "spin_polarized": True,
    "shrink": [8, 8],
    "scf_maxcycle": 800,  # Added for compatibility
    "maxcycle": 800,
    "fmixing": 30,
    "optimization_settings": DEFAULT_OPT_SETTINGS,
    "tolerances": DEFAULT_TOLERANCES,
    "calculation_type": "OPT",  # Added default calculation type
    "optimization_type": "FULLOPTG",  # Added default optimization type
    "scf_method": "DIIS",  # Added default SCF method
    "symmetry_handling": "CIF",  # Added default symmetry handling
}


# ============================================================
# Optimization Settings
# ============================================================

# Optimization types
OPT_TYPES = {
    "1": "FULLOPTG",
    "2": "CELLONLY", 
    "3": "ATOMONLY",  # INTONLY is not a CRYSTAL keyword (manual sec. 7.3.1)
    "4": "ITATOCEL",
    "5": "CVOLOPT"
}


# ============================================================
# Frequency Calculation Settings
# ============================================================

# Advanced frequency settings
ADVANCED_FREQ_SETTINGS = {
    "SUPERCELL": [2, 2, 2],
    "MODES": "ALL",
    "PRINT": 1,
    "INTENS": True,
    "RAMAN": False,
    "IR_METHOD": "BERRY",  # BERRY, WANNIER, or CPHF
    "TEMP": 298.15,
    "PRESSURE": 0.101325,  # MPa (1 atm)
}

# Frequency calculation templates
FREQ_TEMPLATES = {
    "basic": {
        "numderiv": 2,
        "mode": "GAMMA",
        "intensities": False,
        "raman": False,
    },
    "ir_spectrum": {
        "numderiv": 2,
        "mode": "GAMMA",
        "intensities": True,
        "ir_method": "CPHF",
        "irspec": True,
        "spec_range": [0, 4000],
        "resolution": 16,
        "lorentz_width": 8,
    },
    "raman_spectrum": {
        "numderiv": 2,
        "mode": "GAMMA",
        "intensities": True,
        "ir_method": "CPHF",
        "raman": True,
        "cphf_max_iter": 30,
        "cphf_tolerance": 6,
        "ramspec": True,
        "spec_range": [0, 4000],
        "resolution": 16,
        "lorentz_width": 8,
        "laser_wavelength": 532,
        "temperature": 298.15,
    },
    "ir_raman": {
        "numderiv": 2,
        "mode": "GAMMA",
        "intensities": True,
        "ir_method": "CPHF",
        "raman": True,
        "cphf_max_iter": 30,
        "cphf_tolerance": 6,
        "irspec": True,
        "ramspec": True,
        "spec_range": [0, 4000],
        "resolution": 16,
        "lorentz_width": 8,
        "laser_wavelength": 532,
        "temperature": 298.15,
    },
    "thermodynamics": {
        "numderiv": 2,
        "mode": "GAMMA",
        "intensities": False,
        "thermo": True,
        "temprange": (20, 0, 400),
    },
    "phonon_bands": {
        "numderiv": 2,
        "mode": "DISPERSION",
        "dispersion": True,
        "scelphono": [2, 2, 2],
        "bands": {
            "shrink": 16,
            "npoints": 100,
            "path": "AUTO",
        },
    },
    "phonon_dos": {
        "numderiv": 2,
        "mode": "DISPERSION",
        "dispersion": True,
        "scelphono": [2, 2, 2],
        "pdos": {
            "max_freq": 2000,
            "nbins": 200,
            "projected": True,
        },
    },
}

# Common functionals for easy selection
COMMON_FUNCTIONALS = [
    "PBE", "PBE0", "B3LYP", "HSE06", "PBEsol", "PBEsol0"
]

# Print options
PRINT_OPTIONS = {
    "1": "PRINTOUT - Extended printout",
    "2": "PPAN - Mulliken analysis",
    "3": "PELF - Electron localization function",
    "4": "PDOS - Projected density of states",
    "5": "PRHO - Electron density",
    "6": "PBAND - Band structure",
}

# Dispersion options
DISPERSION_OPTIONS = {
    "NONE": "No dispersion correction",
    "D3": "Grimme D3 with zero damping",
    "D3BJ": "Grimme D3 with Becke-Johnson damping",
}

# Smearing options
SMEARING_OPTIONS = {
    "FERMI": "Fermi-Dirac smearing",
    "GAUSS": "Gaussian smearing",
    "MP": "Methfessel-Paxton smearing",
}

# DFT grid options for better organization
DFT_GRID_OPTIONS = ["XLGRID", "LGRID", "GRID", "SMALLGRID"]


# ============================================================
# Configuration Functions (from d12_config_common.py)
# ============================================================

def configure_tolerances(shared_mode: bool = False, calculation_type: str = None) -> Dict[str, Any]:
    """
    Configure integral and SCF tolerances.
    
    Args:
        shared_mode: If True, configuration will be used for multiple files
        calculation_type: Type of calculation (SP, OPT, FREQ) to provide appropriate recommendations
        
    Returns:
        Dictionary with tolerance settings
    """
    # yes_no_prompt is already defined above
    
    tolerances = {}
    
    print("\n=== SCF CONVERGENCE SETTINGS ===")
    
    # Menu-based selection matching CRYSTALOptToD12's approach
    if calculation_type == "FREQ":
        print("\nSelect SCF convergence level (FREQ calculations require tighter tolerances):")
        print("1: Standard - TOLINTEG: 7 7 7 7 14, TOLDEE: 7")
        print("2: Tight - TOLINTEG: 8 8 8 9 24, TOLDEE: 9 (recommended for FREQ)")
        print("3: Very tight - TOLINTEG: 9 9 9 11 38, TOLDEE: 11 (default for FREQ)")
        print("4: Custom")
        
        choice = input("Select tolerance level (1-4) [3]: ").strip()
        if not choice:
            choice = "3"  # Default to very tight for FREQ
    else:
        # SP/OPT calculations
        print("\nSelect SCF convergence level:")
        print("1: Standard - TOLINTEG: 7 7 7 7 14, TOLDEE: 7 (default for OPT/SP)")
        print("2: Tight - TOLINTEG: 8 8 8 9 24, TOLDEE: 9 (higher precision)")
        print("3: Very tight - TOLINTEG: 9 9 9 11 38, TOLDEE: 11 (ultra-high precision)")
        print("4: Custom")
        
        choice = input("Select tolerance level (1-4) [1]: ").strip()
        if not choice:
            choice = "1"  # Default to standard for SP/OPT
    
    # Process the choice
    if choice == "1":
        tolerances["TOLINTEG"] = "7 7 7 7 14"
        tolerances["TOLDEE"] = 7
    elif choice == "2":
        tolerances["TOLINTEG"] = "8 8 8 9 24"
        tolerances["TOLDEE"] = 9
    elif choice == "3":
        tolerances["TOLINTEG"] = "9 9 9 11 38"
        tolerances["TOLDEE"] = 11
    elif choice == "4":
        # Custom tolerances
        print("\nTOLINTEG controls integral accuracy (5 integers):")
        print("  - Higher values = more accurate but slower")
        print("  - Standard: 7 7 7 7 14")
        print("  - Tight: 8 8 8 9 24")
        print("  - Very tight: 9 9 9 11 38")
        print("  - Ultra tight: 10 10 10 12 40")
        
        tolinteg_input = input("Enter TOLINTEG values (5 integers) [7 7 7 7 14]: ").strip()
        if tolinteg_input:
            tolerances["TOLINTEG"] = tolinteg_input
        else:
            tolerances["TOLINTEG"] = "7 7 7 7 14"
        
        print("\nTOLDEE controls SCF convergence (energy threshold):")
        print("  - Value N means convergence at 10^-N Hartree")
        print("  - Default: 7 (10^-7 Ha)")
        print("  - Tight: 9 (10^-9 Ha)")
        print("  - Very tight: 11 (10^-11 Ha)")
        
        toldee_input = input("Enter TOLDEE value (integer) [7]: ").strip()
        if toldee_input:
            try:
                tolerances["TOLDEE"] = int(toldee_input)
            except ValueError:
                print("Invalid input, using default value of 7")
                tolerances["TOLDEE"] = 7
        else:
            tolerances["TOLDEE"] = 7
    else:
        # Invalid choice, use defaults
        print("Invalid choice, using default tolerances.")
        if calculation_type == "FREQ":
            tolerances["TOLINTEG"] = "9 9 9 11 38"
            tolerances["TOLDEE"] = 11
        else:
            tolerances["TOLINTEG"] = "7 7 7 7 14"
            tolerances["TOLDEE"] = 7
    
    return tolerances


def configure_scf_settings(shared_mode: bool = False) -> Dict[str, Any]:
    """
    Configure SCF convergence settings.
    
    Args:
        shared_mode: If True, configuration will be used for multiple files
        
    Returns:
        Dictionary with SCF settings
    """
    
    
    scf_settings = {}
    
    print("\n=== SCF SETTINGS ===")
    
    # MAXCYCLE
    print("\nMaximum SCF cycles:")
    print("  - Default: 800 (recommended)")
    print("  - Increase for difficult convergence")
    
    maxcycle_input = input("MAXCYCLE [800]: ").strip()
    if maxcycle_input:
        try:
            scf_settings["maxcycle"] = int(maxcycle_input)
        except ValueError:
            print("Invalid input, using default of 800")
            scf_settings["maxcycle"] = 800
    else:
        scf_settings["maxcycle"] = 800
    
    # FMIXING
    print("\nFMIXING percentage:")
    print("  - Controls mixing of old and new density matrices")
    print("  - Default: 30 (30%)")
    print("  - Lower values = more stable but slower convergence")
    
    fmixing_input = input("FMIXING [30]: ").strip()
    if fmixing_input:
        try:
            fmixing = int(fmixing_input)
            if 0 <= fmixing <= 100:
                scf_settings["fmixing"] = fmixing
            else:
                print("Value out of range, using default of 30")
                scf_settings["fmixing"] = 30
        except ValueError:
            print("Invalid input, using default of 30")
            scf_settings["fmixing"] = 30
    else:
        scf_settings["fmixing"] = 30
    
    # SCF mixing scheme
    print("\nSCF mixing scheme:")
    mixing_options = {
        "1": "DIIS",      # Default - Direct Inversion in Iterative Subspace
        "2": "NODIIS",    # Simple mixing
        "3": "ANDERSON",  # Anderson mixing
        "4": "BROYDEN",   # Broyden mixing
    }
    
    mixing_choice = get_user_input(
        "Select SCF mixing scheme",
        mixing_options,
        "1"
    )
    
    scf_method = mixing_options[mixing_choice]
    scf_settings["method"] = scf_method  # Always set for compatibility
    
    # Level shifting
    use_levshift = yes_no_prompt(
        "\nUse level shifting (helps convergence for metals/small gaps)?",
        "no"
    )
    
    if use_levshift:
        print("\nLevel shifting moves virtual orbitals up in energy")
        print("  - Helps SCF convergence for metallic/small-gap systems")
        print("  - Default: 5 Ha shift, locked for 20 cycles")
        
        shift_input = input("Shift value in Hartree [5]: ").strip()
        lock_input = input("Lock cycles [20]: ").strip()
        
        try:
            shift = float(shift_input) if shift_input else 5.0
            lock = int(lock_input) if lock_input else 20
            scf_settings["levshift"] = (shift, lock)
        except ValueError:
            print("Invalid input, using defaults")
            scf_settings["levshift"] = (5.0, 20)
    
    # Ask about SMEAR (fermi smearing for metallic systems)
    use_smear = yes_no_prompt(
        "\nUse SMEAR (Fermi smearing for metallic systems)?",
        "no"
    )
    
    if use_smear:
        print("\nSMEAR helps SCF convergence for metals/small-gap systems")
        print("  - Typical values: 0.005-0.02 Hartree")
        print("  - Default: 0.01 Hartree")
        
        smear_input = input("SMEAR value in Hartree [0.01]: ").strip()
        try:
            smear_value = float(smear_input) if smear_input else 0.01
            scf_settings["smear"] = smear_value
        except ValueError:
            print("Invalid input, using default of 0.01")
            scf_settings["smear"] = 0.01
    
    return scf_settings


def select_basis_set(elements: List[int], method: str = "DFT", 
                    functional: Optional[str] = None,
                    shared_mode: bool = False) -> Dict[str, Any]:
    """
    Select basis set based on elements present and method.
    
    Args:
        elements: List of atomic numbers
        method: Calculation method (HF or DFT)
        functional: DFT functional name (for 3C methods)
        shared_mode: If True, configuration will be used for multiple files
        
    Returns:
        Dictionary with basis set configuration
    """
    
    
    basis_config = {}
    
    # Check if functional requires specific basis
    if functional:
        for category, info in FUNCTIONAL_CATEGORIES.items():
            if functional in info.get("functionals", []):
                if "basis_requirements" in info and functional in info["basis_requirements"]:
                    required_basis = info["basis_requirements"][functional]
                    print(f"\nNote: {functional} requires {required_basis} basis set.")
                    basis_config["basis_set_type"] = "INTERNAL"
                    basis_config["basis_set"] = required_basis
                    return basis_config
    
    # Check element compatibility
    heavy_elements = [z for z in elements if z > 86]
    
    print("\n=== BASIS SET SELECTION ===")
    
    if heavy_elements:
        print(f"\nWarning: Heavy elements detected (Z > 86): {heavy_elements}")
        print("Limited basis set options available.")
    
    # Basis set options
    basis_options = {"1": "EXTERNAL", "2": "INTERNAL"}
    
    print("\nBasis set type:")
    print("1: EXTERNAL - Full-core and ECP basis sets (recommended)")
    print("   - DZVP-REV2 / TZVP-REV2")
    print("   - Consistent quality across periodic table")
    print("   - ECPs for elements 37-99")
    print("2: INTERNAL - CRYSTAL built-in basis sets")
    print("   - Various options with different coverage")
    print("   - Some limitations for heavy elements")
    
    basis_choice = get_user_input("Select basis set type", basis_options, "2")
    basis_config["basis_set_type"] = basis_options[basis_choice]
    
    if basis_config["basis_set_type"] == "EXTERNAL":
        # External basis set selection
        # Try to import paths from mace_config
        try:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from mace_config import DEFAULT_DZ_PATH, DEFAULT_TZ_PATH
            external_options = {
                "1": DEFAULT_DZ_PATH,  # DZVP-REV2
                "2": DEFAULT_TZ_PATH,  # TZVP-REV2
            }
        except ImportError:
            # Fallback to local paths
            external_options = {
                "1": "./basis_sets/full.basis.doublezeta/",  # DZVP-REV2
                "2": "./basis_sets/full.basis.triplezeta/",  # TZVP-REV2
            }
        
        print("\nExternal basis set:")
        print("1: DZVP-REV2 - Double-zeta + polarization")
        print("   - Good balance of speed and accuracy")
        print("2: TZVP-REV2 - Triple-zeta + polarization")
        print("   - Higher accuracy, more expensive")
        print("3: Custom path - Specify your own basis set directory")

        # Add custom option
        external_options["3"] = "CUSTOM"

        external_choice = get_user_input(
            "Select external basis set",
            external_options,
            "2"
        )

        if external_choice == "3":
            # Custom basis set path
            print("\nCustom external basis set:")
            print("Enter the full path to your basis set directory.")
            print("This directory should contain numbered files (1, 6, 8, etc.) for each element.")
            print("Example: /path/to/my/basis_sets/custom_basis/")

            while True:
                custom_path = input("Basis set directory path: ").strip()
                if not custom_path:
                    print("Please enter a valid path.")
                    continue

                custom_path = Path(custom_path)
                if not custom_path.exists():
                    print(f"Error: Path does not exist: {custom_path}")
                    retry = input("Try again? [Y/n]: ").strip().lower()
                    if retry == 'n':
                        print("Falling back to TZVP-REV2")
                        basis_config["basis_set"] = external_options["2"]
                        break
                    continue

                if not custom_path.is_dir():
                    print(f"Error: Path is not a directory: {custom_path}")
                    continue

                # Check if it contains some basis files
                basis_files = list(custom_path.glob("[0-9]*"))
                if not basis_files:
                    print(f"Warning: No numbered basis files found in {custom_path}")
                    confirm = input("Use this path anyway? [y/N]: ").strip().lower()
                    if confirm != 'y':
                        continue

                # Ensure path ends with /
                custom_path_str = str(custom_path) + ("/" if not str(custom_path).endswith("/") else "")
                basis_config["basis_set"] = custom_path_str
                basis_config["basis_set_path"] = custom_path_str  # Also set basis_set_path for write_d12_file
                print(f"Using custom basis set: {basis_config['basis_set']}")
                break
        else:
            basis_config["basis_set"] = external_options[external_choice]
            basis_config["basis_set_path"] = external_options[external_choice]  # Also set basis_set_path for write_d12_file
        
    else:
        # Internal basis set selection
        print("\nAvailable internal basis sets:")
        
        # Filter basis sets by element compatibility. The old rule was a guess
        # (max_z <= 36 or a name whitelist) and got it wrong in both directions:
        # it hid POB-TZVP-REV2, the only internal set that carries Pb/Bi/Cs, and
        # it kept sets that do not have the requested elements at all. Ask the
        # measured coverage tables instead (VERIFIED_INTERNAL_BASIS_ELEMENTS).
        structure_elements = sorted(set(elements)) if elements else []
        compatible_basis = {}
        for bs_name, bs_info in INTERNAL_BASIS_SETS.items():
            is_compatible, _missing = check_basis_set_compatibility(
                bs_name, structure_elements, "INTERNAL"
            )
            if is_compatible:
                compatible_basis[bs_name] = bs_info

        if not compatible_basis:
            # Every internal set is missing at least one element. Offering the
            # menu anyway would write a deck CRYSTAL rejects at LoadBa time.
            missing_str = ", ".join(
                str(ELEMENT_SYMBOLS.get(z, z)) for z in structure_elements
            )
            print("\nNo internal basis set covers all elements in this structure:")
            print(f"   {missing_str}")
            print("Switching to an EXTERNAL basis set (TZVP-REV2),")
            print("which provides ECPs for elements 37-99.")
            try:
                import sys
                from pathlib import Path
                sys.path.insert(0, str(Path(__file__).parent.parent))
                from mace_config import DEFAULT_TZ_PATH
                fallback_path = DEFAULT_TZ_PATH
            except ImportError:
                fallback_path = "./basis_sets/full.basis.triplezeta/"
            basis_config["basis_set_type"] = "EXTERNAL"
            basis_config["basis_set"] = fallback_path
            basis_config["basis_set_path"] = fallback_path
            return basis_config

        # Show standard basis sets first
        print("\n--- STANDARD BASIS SETS ---")
        option_num = 1
        internal_options = {}
        
        for bs_name, bs_info in compatible_basis.items():
            if bs_info.get("standard", False):
                internal_options[str(option_num)] = bs_name
                element_info = get_element_info_string(bs_name)
                print(f"{option_num}: {bs_name} - {bs_info['description']}")
                print(f"   {element_info}")
                option_num += 1
        
        # Then additional basis sets
        print("\n--- ADDITIONAL BASIS SETS ---")
        for bs_name, bs_info in compatible_basis.items():
            if not bs_info.get("standard", False):
                internal_options[str(option_num)] = bs_name
                element_info = get_element_info_string(bs_name)
                print(f"{option_num}: {bs_name} - {bs_info['description']}")
                print(f"   {element_info}")
                option_num += 1
        
        # Default to POB-TZVP-REV2 when it survived the filter - that is what the
        # old hardcoded "7" resolved to on the full menu. A fixed number cannot
        # be used any more: get_user_input returns the default verbatim without
        # a membership test, so "7" on a shorter menu raises KeyError below.
        default_choice = next(
            (num for num, name in internal_options.items()
             if name == "POB-TZVP-REV2"),
            "1",
        )

        internal_choice = get_user_input(
            "Select internal basis set",
            internal_options,
            default_choice
        )
        basis_config["basis_set"] = internal_options[internal_choice]
    
    return basis_config


def configure_dft_grid(functional: str, shared_mode: bool = False) -> Optional[str]:
    """
    Configure DFT integration grid.
    
    Args:
        functional: DFT functional name
        shared_mode: If True, configuration will be used for multiple files
        
    Returns:
        Grid keyword or None
    """
    
    
    # 3C methods have their own optimized grids
    if "-3C" in functional or functional.endswith("3C"):
        return None
    
    print("\n=== DFT INTEGRATION GRID ===")
    print("Integration grid quality affects accuracy and speed")
    
    print("\nAvailable grids:")
    print("1: OLDGRID - Old default grid from CRYSTAL09, pruned (55,434)")
    print("2: DEFAULT - Default grid in CRYSTAL23")
    print("3: LGRID - Large grid, pruned (75,434)")
    print("4: XLGRID - Extra large grid (default)")
    print("5: XXLGRID - Extra extra large grid, pruned (99,1454)")
    print("6: XXXLGRID - Ultra extra extra large grid, pruned (150,1454)")
    print("7: HUGEGRID - Ultra extra extra large grid for SCAN, pruned (300,1454)")
    
    grid_choice = get_user_input(
        "Select integration grid",
        DFT_GRIDS,
        "4"  # Default to XLGRID as in original
    )
    
    return DFT_GRIDS[grid_choice]


def configure_dispersion(functional: str, shared_mode: bool = False) -> Dict[str, Any]:
    """
    Configure dispersion correction settings.
    
    Args:
        functional: DFT functional name
        shared_mode: If True, configuration will be used for multiple files
        
    Returns:
        Dictionary with dispersion settings
    """
    # yes_no_prompt is already defined above
    
    dispersion_config = {}
    
    # Check if functional already includes dispersion
    if "-3C" in functional or functional.endswith("3C") or "-D3" in functional:
        if "-D3" in functional:
            print(f"\nNote: {functional} already has D3 dispersion selected.")
            dispersion_config["use_dispersion"] = True
        else:
            print(f"\nNote: {functional} already includes dispersion corrections.")
            dispersion_config["use_dispersion"] = False
        return dispersion_config
    
    # Check if functional supports D3 (strip -D3 if present)
    base_functional = functional.replace("-D3", "")
    if base_functional not in D3_FUNCTIONALS:
        print(f"\nNote: D3 dispersion not parameterized for {functional}")
        dispersion_config["use_dispersion"] = False
        return dispersion_config
    
    # Ask about dispersion
    print("\n=== DISPERSION CORRECTION ===")
    print(f"D3 dispersion correction is available for {functional}")
    print("Recommended for:")
    print("  - Van der Waals interactions")
    print("  - Molecular crystals")
    print("  - Layered materials")
    print("  - Adsorption studies")
    
    use_d3 = yes_no_prompt(
        f"Add D3 dispersion correction to {functional}?",
        "yes"
    )
    
    dispersion_config["use_dispersion"] = use_d3
    
    if use_d3:
        # Ask about D3 variant
        print("\nD3 variants:")
        print("1: D3(0) - Original D3 with zero damping")
        print("2: D3(BJ) - Becke-Johnson damping (recommended)")
        
        d3_variant = input("Select D3 variant (1-2) [2]: ").strip() or "2"
        
        if d3_variant == "2":
            dispersion_config["d3_version"] = "D3BJ"
        else:
            dispersion_config["d3_version"] = "D3"
    
    return dispersion_config


def configure_spin_polarization(shared_mode: bool = False) -> Dict[str, Any]:
    """
    Configure spin polarization settings.
    
    Args:
        shared_mode: If True, configuration will be used for multiple files
        
    Returns:
        Dictionary with spin settings
    """
    # yes_no_prompt is already defined above
    
    spin_config = {}
    
    print("\n=== SPIN POLARIZATION ===")
    
    use_spin = yes_no_prompt(
        "Use spin-polarized calculation?",
        "yes"
    )
    
    spin_config["spin_polarized"] = use_spin
    
    if use_spin:
        print("\nSPINLOCK options (number of unpaired electrons, nα-nβ):")
        print("  - Enter 0 for automatic spin optimization")
        print("  - Enter positive integer for fixed spin multiplicity (e.g., 2 for triplet)")
        print("  - Enter a negative integer for a net beta-electron excess")
        print("    (this is a CELL TOTAL, not an ordering: CRYSTAL has no way to")
        print("     express an antiferromagnetic sublattice from SPINLOCK alone)")
        
        spinlock_input = input("SPINLOCK value (nα-nβ) [0]: ").strip()
        
        if spinlock_input:
            try:
                spinlock = int(spinlock_input)
                if spinlock != 0:
                    spin_config["spinlock"] = spinlock
            except ValueError:
                print("Invalid input, using automatic spin")
    
    return spin_config


def configure_smearing(system_type: str = "insulator", 
                      shared_mode: bool = False) -> Dict[str, Any]:
    """
    Configure Fermi smearing for metallic systems.
    
    Args:
        system_type: Type of system (metal/semiconductor/insulator)
        shared_mode: If True, configuration will be used for multiple files
        
    Returns:
        Dictionary with smearing settings
    """
    # yes_no_prompt is already defined above
    
    smear_config = {}
    
    if system_type == "insulator":
        print("\nInsulating system - Fermi smearing not needed")
        smear_config["enabled"] = False
        return smear_config
    
    print("\n=== FERMI SMEARING ===")
    print("Fermi smearing helps SCF convergence for metals")
    
    if system_type == "metal":
        default_smear = "yes"
        print("Metallic system detected - smearing recommended")
    else:
        default_smear = "no"
        print("Small gap semiconductor - smearing optional")
    
    use_smear = yes_no_prompt(
        "Enable Fermi smearing?",
        default_smear
    )
    
    smear_config["enabled"] = use_smear
    
    if use_smear:
        print("\nSmearing width (Hartree):")
        print("  - Typical: 0.001-0.01 Ha")
        print("  - Larger values = easier convergence but less accurate")
        print("  - Must extrapolate to zero smearing for final energy")
        
        width_input = input("Smearing width [0.005]: ").strip()
        
        if width_input:
            try:
                width = float(width_input)
                smear_config["width"] = width
            except ValueError:
                print("Invalid input, using default of 0.005")
                smear_config["width"] = 0.005
        else:
            smear_config["width"] = 0.005
    
    return smear_config


# ============================================================
# Utility Functions
# ============================================================

# Back-navigation primitive. Falls back to plain input() if menu_nav is unavailable,
# so these helpers keep working even outside the MACE tree. nav_read only intercepts
# 'b'/'back' when a flow is wrapped by run_with_back AND the valid_set excludes 'b'
# (so a menu whose options literally include 'b' is unaffected); otherwise it behaves
# exactly like input().
try:
    from menu_nav import nav_read as _nav_read
except Exception:  # pragma: no cover - defensive fallback
    def _nav_read(prompt="", valid_set=None):
        return input(prompt)


def get_user_input(prompt: str, options: Any, default: Optional[str] = None) -> str:
    """
    Get validated user input from a list of options

    Args:
        prompt: The prompt to display to the user
        options: Valid options (list or dict)
        default: Default value

    Returns:
        Valid user input
    """
    if isinstance(options, dict):
        opt_str = "\n".join([f"{key}: {value}" for key, value in options.items()])
        valid_inputs = list(options.keys())
    else:
        opt_str = "\n".join([f"{i + 1}: {opt}" for i, opt in enumerate(options)])
        valid_inputs = [str(i + 1) for i in range(len(options))]

    default_str = f" (default: {default})" if default else ""

    while True:
        print(f"\n{prompt}{default_str}:\n{opt_str}")
        choice = _nav_read("Enter your choice: ", valid_set=valid_inputs).strip()
        
        if choice == "" and default:
            return default
        
        if choice in valid_inputs:
            return choice
        
        print(f"Invalid input. Please choose from {', '.join(valid_inputs)}")


def yes_no_prompt(prompt: str, default: str = "yes") -> bool:
    """
    Prompt for a yes/no response
    
    Args:
        prompt: The prompt to display
        default: Default value ('yes' or 'no')
        
    Returns:
        True for yes, False for no
    """
    valid = {"yes": True, "y": True, "no": False, "n": False}
    if default == "yes":
        prompt += " [Y/n] "
    elif default == "no":
        prompt += " [y/N] "
    else:
        raise ValueError(f"Invalid default value: {default}")
    
    while True:
        choice = _nav_read(prompt, valid_set=valid).lower() or default
        if choice in valid:
            return valid[choice]
        print("Please respond with 'yes' or 'no' (or 'y' or 'n').")


def get_valid_input(prompt: str, valid_values: List[str], 
                   default: Optional[str] = None) -> str:
    """Get validated user input from a list of valid values"""
    while True:
        value = _nav_read(prompt, valid_set=valid_values).strip()
        if not value and default:
            return default
        if value in valid_values:
            return value
        print(f"Invalid input. Please choose from: {', '.join(valid_values)}")


def safe_float(value: str, default: float) -> float:
    """Safely convert string to float with default"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value: str, default: int) -> int:
    """Safely convert string to int with default"""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _is_rhombohedral_axes(a: float, b: float, c: float,
                          alpha: float, beta: float, gamma: float) -> bool:
    """Whether a trigonal cell is given in rhombohedral (not hexagonal) axes.

    Same predicate, same tolerance and same hexagonal-first ordering as
    NewCifToD12.detect_trigonal_setting, so the two cannot disagree about the
    same numbers.
    """
    if (abs(alpha - 90) < 1e-3 and abs(beta - 90) < 1e-3
            and abs(gamma - 120) < 1e-3):
        # alpha ~ 90, beta ~ 90, gamma ~ 120 indicates hexagonal axes
        return False
    if abs(alpha - beta) < 1e-3 and abs(beta - gamma) < 1e-3:
        # alpha = beta = gamma != 90 indicates rhombohedral axes
        return True
    return False


def layer_group_lattice(layer_group: int) -> str:
    """2D lattice type of a layer group (manual Appendix A.2, page 421).

    A.2 prints the 80 layer groups under four lattice headings, in contiguous
    blocks: oblique (P) 1-7, rectangular (P or C) 8-48, square (P) 49-64,
    hexagonal (P) 65-80. Those blocks are also what fixes how many values the
    SLAB cell record carries, so this function and the SLAB branch of
    generate_unit_cell_line below must not drift apart.
    """
    if 1 <= layer_group <= 7:
        return "oblique"
    if 8 <= layer_group <= 48:
        return "rectangular"
    if 49 <= layer_group <= 64:
        return "square"
    if 65 <= layer_group <= 80:
        return "hexagonal"
    raise ValueError(f"Invalid layer group: {layer_group}")


def check_layer_group_cell(layer_group: int, a: float, b: float,
                           gamma: float) -> Optional[str]:
    """Refusal message when a SLAB cell contradicts its layer group's lattice.

    The minimal set printed after the layer group is "a,[b],[gamma]", with b
    "for rectangular lattices only" and the angle for "triclinic lattices only"
    (manual L999-1002), so a square or hexagonal group prints a alone and
    CRYSTAL derives b and gamma from the group itself. The manual's diamond
    (100) deck (L29193-29196) prints "2.52437 2.52437" - two values with
    a == b - which is the proof that the count follows the lattice class and
    not the numbers.

    This is not a defensive cross-check on an otherwise sound map: nothing
    upstream puts the cell into the International Tables first setting (see the
    note over LAYER_GROUP_FROM_SPACEGROUP), so for an automatically mapped
    group this is the only place the chosen group is ever compared against the
    actual cell, and callers must treat the message as a refusal there. When
    the caller named the layer group itself the same message is worth emitting
    as a warning rather than a refusal - a relaxed cell a tenth of a degree off
    its ideal angle is still the group the caller says it is, and there would
    otherwise be no way to convert it at all.

    The check covers ONLY the in-plane lattice class. It does NOT catch a group
    of the right lattice class in the wrong orientation - A.2's 25/(25),
    28/(28) and 51/(51) sibling pairs are all rectangular - which is why the
    space-group map must independently require that the appendix list exactly
    one candidate. It also says nothing about c, alpha or beta - a cell whose c
    axis is not perpendicular to the layer is flattened by the fractional-z to
    Cartesian-z conversion in the callers, and that is a separate question this
    does not settle - and nothing about the z origin, for which see
    LAYER_GROUPS_POLAR_IN_Z.

    Returns None when the deck may be written.
    """
    lattice = layer_group_lattice(layer_group)
    mean_ab = 0.5 * (a + b)
    lengths_equal = (
        mean_ab > 0 and abs(a - b) / mean_ab <= LAYER_GROUP_LENGTH_RTOL
    )

    def angle_is(target):
        return abs(gamma - target) <= LAYER_GROUP_ANGLE_TOL_DEG

    required = None
    if lattice == "rectangular" and not angle_is(90.0):
        required = "gamma = 90 degrees"
    elif lattice == "square" and not (angle_is(90.0) and lengths_equal):
        required = "a = b and gamma = 90 degrees"
    elif lattice == "hexagonal" and not (angle_is(120.0) and lengths_equal):
        required = "a = b and gamma = 120 degrees"

    if required is None:
        return None
    return (
        f"Layer group {layer_group} sits on a {lattice} 2D lattice (manual "
        f"Appendix A.2), which requires {required}, but the cell is "
        f"a={a:.6f}, b={b:.6f}, gamma={gamma:.6f}. CRYSTAL reads only the "
        f"minimal set of lattice vectors for that class, so it would build a "
        f"different cell from the one given."
    )


def generate_unit_cell_line(spacegroup: int, cell_params: List[float],
                           dimensionality: str,
                           use_rhombohedral_axes: Optional[bool] = None) -> str:
    """Generate the unit cell line for CRYSTAL23 input

    use_rhombohedral_axes selects the cell a rhombohedral space group is written
    in (CRYSTAL's IFHR flag). None means "infer it from the cell parameters".
    """
    if dimensionality == "MOLECULE":
        return ""  # No unit cell for molecules
    
    a, b, c, alpha, beta, gamma = [float(x) for x in cell_params[:6]]
    
    if dimensionality == "SLAB":
        # How many values the minimal set has is fixed by the layer group's 2D
        # lattice type: the SLAB record is "a,[b],[gamma]" with b for
        # rectangular lattices only and the angle for oblique lattices only,
        # and Appendix A.2 partitions the 80 layer groups into oblique 1-7,
        # rectangular 8-48, square 49-64 and hexagonal 65-80. Printing three
        # values for a square or hexagonal layer group makes CRYSTAL consume
        # the following line as coordinates.
        if 1 <= spacegroup <= 7:  # Oblique
            return f"{a:.8f} {b:.8f} {gamma:.6f}"
        elif 8 <= spacegroup <= 48:  # Rectangular (P or C)
            return f"{a:.8f} {b:.8f}"
        elif 49 <= spacegroup <= 80:  # Square (49-64), hexagonal (65-80)
            return f"{a:.8f}"
        else:
            raise ValueError(f"Invalid layer group: {spacegroup}")
    elif dimensionality == "POLYMER":
        return f"{a:.8f}"
    elif dimensionality == "CRYSTAL":
        if spacegroup >= 1 and spacegroup <= 2:  # Triclinic
            return f"{a:.8f} {b:.8f} {c:.8f} {alpha:.6f} {beta:.6f} {gamma:.6f}"
        elif spacegroup >= 3 and spacegroup <= 15:  # Monoclinic
            # CRYSTAL interprets monoclinic space-group NUMBERS in the standard
            # International Tables "unique axis b" setting, so the cell line must
            # carry beta (the a^c angle) with alpha = gamma = 90. A cell that is
            # actually unique-axis-a or unique-axis-c (alpha or gamma is the
            # non-orthogonal angle) cannot be expressed by swapping the printed
            # angle: CRYSTAL would still build a b-unique cell and read it as
            # beta, silently producing a wrong lattice. Detect that case and
            # refuse loudly rather than emit garbage. b-unique cells (every real
            # MP/spglib monoclinic structure) are unchanged: this still returns
            # "a b c beta".
            d_alpha, d_beta, d_gamma = (
                abs(alpha - 90.0), abs(beta - 90.0), abs(gamma - 90.0)
            )
            if d_alpha > 1e-2 and d_alpha >= d_beta and d_alpha >= d_gamma:
                raise ValueError(
                    f"Monoclinic cell looks unique-axis-a (alpha={alpha:.4f}, "
                    f"beta={beta:.4f}, gamma={gamma:.4f}); CRYSTAL space group "
                    f"{spacegroup} expects the standard unique-axis-b setting "
                    f"(beta != 90, alpha = gamma = 90). Standardize the structure "
                    f"to unique-axis-b (e.g. via spglib) before generating the .d12."
                )
            if d_gamma > 1e-2 and d_gamma >= d_beta and d_gamma >= d_alpha:
                raise ValueError(
                    f"Monoclinic cell looks unique-axis-c (alpha={alpha:.4f}, "
                    f"beta={beta:.4f}, gamma={gamma:.4f}); CRYSTAL space group "
                    f"{spacegroup} expects the standard unique-axis-b setting "
                    f"(beta != 90, alpha = gamma = 90). Standardize the structure "
                    f"to unique-axis-b (e.g. via spglib) before generating the .d12."
                )
            return f"{a:.8f} {b:.8f} {c:.8f} {beta:.6f}"
        elif spacegroup >= 16 and spacegroup <= 74:  # Orthorhombic
            return f"{a:.8f} {b:.8f} {c:.8f}"
        elif spacegroup >= 75 and spacegroup <= 142:  # Tetragonal
            return f"{a:.8f} {c:.8f}"
        elif spacegroup >= 143 and spacegroup <= 167:  # Trigonal
            # A rhombohedral group may be written in either cell, and the two
            # lines are different quantities: IFHR=0 is the hexagonal cell
            # (a,c), IFHR=1 the rhombohedral one (a,alpha). Emitting "a c" for
            # a cell given in rhombohedral axes hands CRYSTAL the a length
            # where it expects alpha. Non-rhombohedral trigonal groups have
            # only the hexagonal cell, so they are untouched.
            if spacegroup in RHOMBOHEDRAL_SPACEGROUPS:
                rhombohedral = use_rhombohedral_axes
                if rhombohedral is None:
                    rhombohedral = _is_rhombohedral_axes(
                        a, b, c, alpha, beta, gamma
                    )
                if rhombohedral:
                    return f"{a:.8f} {alpha:.6f}"
            return f"{a:.8f} {c:.8f}"
        elif spacegroup >= 168 and spacegroup <= 194:  # Hexagonal
            return f"{a:.8f} {c:.8f}"
        elif spacegroup >= 195 and spacegroup <= 230:  # Cubic
            return f"{a:.8f}"
        else:
            raise ValueError(f"Invalid space group: {spacegroup}")
    
    return ""


def read_basis_file(basis_dir: str, atomic_number: int, basis_set_type: str = "EXTERNAL") -> str:
    """
    Read a basis set file for a given element.

    For EXTERNAL basis sets, ECP elements (Z >= 37) have their basis files
    named with +200 offset (e.g., Te=52 -> file "252", Pb=82 -> file "282").

    Args:
        basis_dir: Directory containing basis set files
        atomic_number: Element atomic number (original, without +200)
        basis_set_type: "EXTERNAL" or "INTERNAL" - determines file naming

    Returns:
        Content of the basis set file
    """
    import os

    # For EXTERNAL basis sets, ECP elements use +200 naming convention
    file_number = atomic_number
    if basis_set_type == "EXTERNAL" and atomic_number in ECP_ELEMENTS_EXTERNAL:
        file_number = atomic_number + 200

    basis_file_path = os.path.join(basis_dir, str(file_number))

    try:
        with open(basis_file_path, "r") as f:
            return f.read()
    except FileNotFoundError:
        # Also try without +200 offset as fallback (for non-standard directories)
        if file_number != atomic_number:
            try:
                with open(os.path.join(basis_dir, str(atomic_number)), "r") as f:
                    return f.read()
            except FileNotFoundError:
                pass

        print(
            f"Warning: Basis set file for element {atomic_number} (Z={atomic_number}) "
            f"not found in {basis_dir} (tried: {file_number})"
        )
        return ""


def get_element_info_string(basis_name: str) -> str:
    """
    Get a formatted string describing element coverage for a basis set.
    
    Args:
        basis_name: Name of the basis set
        
    Returns:
        Formatted string with element information
    """
    if basis_name not in INTERNAL_BASIS_SETS:
        return "Unknown basis set"
    
    bs_info = INTERNAL_BASIS_SETS[basis_name]
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
                    ranges.append(f"{ELEMENT_SYMBOLS.get(start, start)}")
                else:
                    ranges.append(
                        f"{ELEMENT_SYMBOLS.get(start, start)}-{ELEMENT_SYMBOLS.get(end, end)}"
                    )
                start = end = elem_list[i]
        
        # Add the last range
        if start == end:
            ranges.append(f"{ELEMENT_SYMBOLS.get(start, start)}")
        else:
            ranges.append(
                f"{ELEMENT_SYMBOLS.get(start, start)}-{ELEMENT_SYMBOLS.get(end, end)}"
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


def format_crystal_float(value: float) -> str:
    """
    Format a float value for CRYSTAL input according to its specific rules.
    CRYSTAL requires scientific notation for values outside certain ranges.
    
    Args:
        value: Float value to format
        
    Returns:
        Formatted string
    """
    if abs(value) < 1e-10:
        return "0.0"
    elif 0.0001 <= abs(value) < 10000:
        # For values in this range, use regular decimal notation
        return f"{value:.6f}".rstrip('0').rstrip('.')
    else:
        # For very small or large values, use scientific notation
        return f"{value:.6E}"


def generate_k_points(a: float, b: float, c: float, dimensionality: str, spacegroup: int) -> Tuple[int, int, int]:
    """
    Generate Monkhorst-Pack k-point grid based on cell parameters

    Args:
        a, b, c: Cell parameters in Angstroms
        dimensionality: CRYSTAL, SLAB, POLYMER, or MOLECULE
        spacegroup: Space group number

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


# Element coverage of CRYSTAL23's INTERNAL basis sets, measured rather than
# assumed. The manual documents no per-element ranges, and the 3c basis sets
# (def2-mSVP, mTZVP, SOLDEF2MSVP, MINIX, SOLMINIX) are not listed in its
# internal-library table at all -- so these were mapped by running CRYSTAL23
# itself on a one-atom cell for every element 1-99 and recording which loaded.
#
# Two distinct failure modes were seen, and BOTH are excluded here:
#   "Basis set is not implemented for requested element: N" -- simply absent.
#   "ERROR **** LoadBa **** UNIT CELL NOT NEUTRAL"          -- the element is
#      present but its shell charges disagree with the effective nuclear
#      charge (an ECP core mismatch in CRYSTAL's own library). Affects
#      def2-mSVP and MINIX at 81-85, and mTZVP at 79-86. Unusable either way.
#
# This is why a lead perovskite under HSE-3c/def2-mSVP dies with a neutrality
# error on a cell that is perfectly neutral: Pb (82) falls in that broken band.
#
# Regenerate with tests/basis_coverage/scan_basis.sh (see AUTHORSHIP notes).
# Measured against CRYSTAL/23-intel-2023a.
VERIFIED_INTERNAL_BASIS_ELEMENTS = {
    # 3c composite-method basis sets (undocumented in the manual)
    "def2-mSVP": list(range(1, 81)) + [86],
    "MINIX": list(range(1, 81)) + [86],
    "mTZVP": list(range(1, 58)) + list(range(72, 79)),
    "SOLDEF2MSVP": list(range(1, 54)),
    "SOLMINIX": list(range(1, 54)),
    # General-purpose internal sets
    "STO-3G": list(range(1, 54)),
    "STO-6G": list(range(1, 37)),
    "POB-DZVP": list(range(1, 43)) + list(range(44, 54)) + [74, 83],
    "POB-DZVPP": [1] + list(range(3, 10)) + list(range(11, 18))
    + list(range(19, 36)) + [49, 83],
    "POB-DZVP-REV2": [1] + list(range(3, 10)) + list(range(11, 18))
    + list(range(19, 36)),
    "POB-TZVP": list(range(1, 39)) + list(range(40, 43))
    + list(range(44, 54)) + [83],
    "POB-TZVP-REV2": [1] + list(range(3, 10)) + list(range(11, 18))
    + list(range(19, 36)) + list(range(37, 43)) + list(range(44, 54))
    + [55, 56] + list(range(72, 85)),
}


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
        available_elements = _internal_basis_elements(basis_set)
        if available_elements is not None:
            for atom_num in set(atomic_numbers):
                if atom_num not in available_elements:
                    missing_elements.append(atom_num)
    else:  # EXTERNAL
        # An external basis set is a DIRECTORY of per-element files. The only
        # honest check is whether the file this run would actually read exists:
        # read_basis_file() returns "" for a missing file, which silently drops
        # that element's basis block out of the deck.
        for atom_num in set(atomic_numbers):
            if atom_num > 99 or not _external_basis_file_exists(basis_set, atom_num):
                missing_elements.append(atom_num)

    return len(missing_elements) == 0, sorted(missing_elements)


def _internal_basis_elements(basis_set):
    """Elements an internal basis set actually supports, or None if unknown.

    Returning None (rather than an empty set) keeps an unrecognised basis name
    from being reported as "every element is missing"; the caller treats None
    as "cannot verify" and passes the deck through.
    """
    if not isinstance(basis_set, str):
        # Callers may pass None when no basis has been chosen yet; that is
        # "cannot verify", not "every element is missing".
        return None
    # Measured coverage wins over the hand-written INTERNAL_BASIS_SETS ranges:
    # several of those were wrong in the dangerous direction (POB-TZVP-REV2
    # claimed He/Ne/Ar/Tc, which CRYSTAL rejects outright).
    for table in (VERIFIED_INTERNAL_BASIS_ELEMENTS, INTERNAL_BASIS_SETS):
        entry = table.get(basis_set)
        if entry is not None:
            return set(entry["elements"] if isinstance(entry, dict) else entry)
        # CRYSTAL keywords are case-insensitive and the 3c names are mixed
        # case (def2-mSVP, mTZVP), so a user-typed name may not match exactly.
        lowered = basis_set.lower()
        for name, value in table.items():
            if name.lower() == lowered:
                return set(value["elements"] if isinstance(value, dict) else value)
    return None


def _external_basis_file_exists(basis_dir, atomic_number):
    """Whether read_basis_file() would find a file for this element.

    Mirrors read_basis_file's lookup exactly, including its +200 ECP naming and
    its fallback to the un-offset name.
    """
    import os

    if not basis_dir or not os.path.isdir(basis_dir):
        # Not a usable directory - let the existing "basis file not found"
        # warning in read_basis_file speak rather than blaming the elements.
        return True
    candidates = [atomic_number]
    if atomic_number in ECP_ELEMENTS_EXTERNAL:
        candidates.insert(0, atomic_number + 200)
    return any(os.path.isfile(os.path.join(basis_dir, str(n))) for n in candidates)
