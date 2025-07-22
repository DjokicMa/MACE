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
FUNCTIONAL_KEYWORD_MAP = {
    "PBESOL": "PBESOLXC",
    "SOGGA": "SOGGAXC", 
    "VBH": "VBHLYP",
    "mPWPW": "PWPW",
    "LC-wPBE": "WCGGA-PBE",
    "LC-BLYP": "WCGGA"
}


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
    "3": "INTONLY",
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
    max_z = max(elements) if elements else 1
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
        
        external_choice = get_user_input(
            "Select external basis set",
            external_options,
            "2"
        )
        basis_config["basis_set"] = external_options[external_choice]
        
    else:
        # Internal basis set selection
        print("\nAvailable internal basis sets:")
        
        # Filter basis sets by element compatibility
        compatible_basis = {}
        for bs_name, bs_info in INTERNAL_BASIS_SETS.items():
            # Simple compatibility check - could be improved
            if max_z <= 36 or "ECP" in bs_name or bs_name in ["POB-DZVP", "POB-TZVP"]:
                compatible_basis[bs_name] = bs_info
        
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
        
        internal_choice = get_user_input(
            "Select internal basis set",
            internal_options,
            "7"  # Default to POB-TZVP if available
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
        print("  - Enter -1 for antiferromagnetic initial guess")
        
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
        choice = input("Enter your choice: ").strip()
        
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
        choice = input(prompt).lower() or default
        if choice in valid:
            return valid[choice]
        print("Please respond with 'yes' or 'no' (or 'y' or 'n').")


def get_valid_input(prompt: str, valid_values: List[str], 
                   default: Optional[str] = None) -> str:
    """Get validated user input from a list of valid values"""
    while True:
        value = input(prompt).strip()
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


def generate_unit_cell_line(spacegroup: int, cell_params: List[float], 
                           dimensionality: str) -> str:
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


def read_basis_file(basis_dir: str, atomic_number: int) -> str:
    """
    Read a basis set file for a given element
    
    Args:
        basis_dir: Directory containing basis set files
        atomic_number: Element atomic number
        
    Returns:
        Content of the basis set file
    """
    import os
    try:
        with open(os.path.join(basis_dir, str(atomic_number)), "r") as f:
            return f.read()
    except FileNotFoundError:
        print(
            f"Warning: Basis set file for element {atomic_number} not found in {basis_dir}"
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
