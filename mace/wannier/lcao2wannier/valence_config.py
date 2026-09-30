"""
Valence orbital configuration lookup table for LCAO-PDWF band selection.

Provides the angular momentum channels (l-values) considered "valence"
for each element, used to build the target orbital mask in the
LCAO projectability-disentangled Wannier function approach.

Two configurations are provided per element:
  - 'standard': minimal valence set matching typical pseudopotential
    PAO configurations (SSSP/PseudoDojo/GBRV conventions). Suitable
    for band interpolation and transport properties.
  - 'extended': includes semi-core states that may hybridize with
    valence bands. Needed for GW, DMFT, or when semi-core states
    overlap energetically with valence bands.

The l-values follow the convention: 0=s, 1=p, 2=d, 3=f.

Sources:
  - VASP PAW pseudopotential recommendations (vasp.at/wiki)
  - SSSP v1.3 efficiency/precision libraries (materialscloud.org)
  - PseudoDojo v0.4 (pseudo-dojo.org)
  - NIST Atomic Spectra Database for ground state configurations
  - Standard solid-state chemistry textbook conventions

Note: This table is the author's compilation from the above public
sources, adapted for LCAO basis set classification. It is not a
verbatim reproduction of any single source.
"""

from typing import Dict, Set, Tuple, Union, Optional, List
import numpy as np


# Element symbols indexed by atomic number
ELEMENT_SYMBOLS = [
    '',  # index 0 placeholder
    'H',  'He',
    'Li', 'Be', 'B',  'C',  'N',  'O',  'F',  'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P',  'S',  'Cl', 'Ar',
    'K',  'Ca', 'Sc', 'Ti', 'V',  'Cr', 'Mn', 'Fe',
    'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se',
    'Br', 'Kr',
    'Rb', 'Sr', 'Y',  'Zr', 'Nb', 'Mo', 'Tc', 'Ru',
    'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te',
    'I',  'Xe',
    'Cs', 'Ba',
    'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd',
    'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W',  'Re', 'Os', 'Ir', 'Pt', 'Au',
    'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
]

# Reverse lookup: symbol -> Z
ELEMENT_Z = {sym: z for z, sym in enumerate(ELEMENT_SYMBOLS) if sym}


# ============================================================================
# VALENCE CONFIGURATION TABLE
# ============================================================================
# Format: Z -> (standard_l_set, extended_l_set, n_valence_standard, notes)
#
# standard_l_set: set of l-values for minimal valence (matches typical PP)
# extended_l_set: set of l-values including semi-core states
# n_valence_standard: number of valence electrons in standard config
#
# The l-values indicate which angular momentum channels should be
# classified as "target" when the first radial function of that channel
# on each atom is selected. For the LCAO-PDWF approach, this determines
# num_wann = sum over atoms of (2l+1) for each l in the target set
# (doubled for SOC spinor calculations).
# ============================================================================

VALENCE_CONFIG: Dict[int, Tuple[Set[int], Set[int], int, str]] = {
    # ---- Period 1 ----
    1:  ({0},       {0},          1,  "H: 1s1"),
    2:  ({0},       {0},          2,  "He: 1s2"),

    # ---- Period 2 ----
    3:  ({0},       {0, 1},       1,  "Li: 2s1; extended adds 1s semi-core"),
    4:  ({0},       {0, 1},       2,  "Be: 2s2; extended adds 1s or 2p unoccupied"),
    5:  ({0, 1},    {0, 1},       3,  "B: 2s2 2p1"),
    6:  ({0, 1},    {0, 1},       4,  "C: 2s2 2p2"),
    7:  ({0, 1},    {0, 1},       5,  "N: 2s2 2p3"),
    8:  ({0, 1},    {0, 1},       6,  "O: 2s2 2p4"),
    9:  ({0, 1},    {0, 1},       7,  "F: 2s2 2p5"),
    10: ({0, 1},    {0, 1},       8,  "Ne: 2s2 2p6"),

    # ---- Period 3 ----
    11: ({0},       {0, 1},       1,  "Na: 3s1; extended adds 2p semi-core"),
    12: ({0, 1},    {0, 1},       2,  "Mg: 3s2; p included for crystal field polarization"),
    13: ({0, 1},    {0, 1},       3,  "Al: 3s2 3p1"),
    14: ({0, 1},    {0, 1},       4,  "Si: 3s2 3p2"),
    15: ({0, 1},    {0, 1},       5,  "P: 3s2 3p3"),
    16: ({0, 1},    {0, 1},       6,  "S: 3s2 3p4"),
    17: ({0, 1},    {0, 1},       7,  "Cl: 3s2 3p5"),
    18: ({0, 1},    {0, 1},       8,  "Ar: 3s2 3p6"),

    # ---- Period 4 ----
    19: ({0},       {0, 1},       1,  "K: 4s1; extended adds 3p semi-core"),
    20: ({0, 1},    {0, 1},       2,  "Ca: 4s2; p for polarization, extended adds 3p"),

    # 3d transition metals: standard = s,d; extended adds p semi-core (3p)
    21: ({0, 2},    {0, 1, 2},   3,   "Sc: 3d1 4s2"),
    22: ({0, 2},    {0, 1, 2},   4,   "Ti: 3d2 4s2"),
    23: ({0, 2},    {0, 1, 2},   5,   "V: 3d3 4s2"),
    24: ({0, 2},    {0, 1, 2},   6,   "Cr: 3d5 4s1"),
    25: ({0, 2},    {0, 1, 2},   7,   "Mn: 3d5 4s2"),
    26: ({0, 2},    {0, 1, 2},   8,   "Fe: 3d6 4s2"),
    27: ({0, 2},    {0, 1, 2},   9,   "Co: 3d7 4s2"),
    28: ({0, 2},    {0, 1, 2},  10,   "Ni: 3d8 4s2"),
    29: ({0, 2},    {0, 1, 2},  11,   "Cu: 3d10 4s1"),
    30: ({0, 2},    {0, 1, 2},  12,   "Zn: 3d10 4s2"),

    # Post-3d main group
    31: ({0, 1},    {0, 1, 2},   3,   "Ga: 4s2 4p1; extended adds 3d semi-core"),
    32: ({0, 1},    {0, 1, 2},   4,   "Ge: 4s2 4p2; extended adds 3d"),
    33: ({0, 1},    {0, 1, 2},   5,   "As: 4s2 4p3; extended adds 3d"),
    34: ({0, 1},    {0, 1, 2},   6,   "Se: 4s2 4p4; extended adds 3d"),
    35: ({0, 1},    {0, 1, 2},   7,   "Br: 4s2 4p5"),
    36: ({0, 1},    {0, 1, 2},   8,   "Kr: 4s2 4p6"),

    # ---- Period 5 ----
    37: ({0},       {0, 1},       1,  "Rb: 5s1; extended adds 4p"),
    38: ({0, 1},    {0, 1},       2,  "Sr: 5s2; p for polarization"),

    # 4d transition metals: standard = s,d; extended adds p semi-core (4p)
    39: ({0, 2},    {0, 1, 2},   3,   "Y: 4d1 5s2"),
    40: ({0, 2},    {0, 1, 2},   4,   "Zr: 4d2 5s2"),
    41: ({0, 2},    {0, 1, 2},   5,   "Nb: 4d4 5s1"),
    42: ({0, 2},    {0, 1, 2},   6,   "Mo: 4d5 5s1"),
    43: ({0, 2},    {0, 1, 2},   7,   "Tc: 4d5 5s2"),
    44: ({0, 2},    {0, 1, 2},   7,   "Ru: 4d7 5s1"),
    45: ({0, 2},    {0, 1, 2},   9,   "Rh: 4d8 5s1"),
    46: ({0, 2},    {0, 1, 2},  10,   "Pd: 4d10"),
    47: ({0, 2},    {0, 1, 2},  11,   "Ag: 4d10 5s1"),
    48: ({0, 2},    {0, 1, 2},  12,   "Cd: 4d10 5s2"),

    # Post-4d main group
    49: ({0, 1},    {0, 1, 2},   3,   "In: 5s2 5p1; extended adds 4d"),
    50: ({0, 1},    {0, 1, 2},   4,   "Sn: 5s2 5p2; extended adds 4d"),
    51: ({0, 1},    {0, 1, 2},   5,   "Sb: 5s2 5p3; extended adds 4d"),
    52: ({0, 1},    {0, 1, 2},   6,   "Te: 5s2 5p4; extended adds 4d"),
    53: ({0, 1},    {0, 1, 2},   7,   "I: 5s2 5p5"),
    54: ({0, 1},    {0, 1, 2},   8,   "Xe: 5s2 5p6"),

    # ---- Period 6 ----
    55: ({0},       {0, 1},       1,  "Cs: 6s1; extended adds 5p"),
    56: ({0, 1},    {0, 1},       2,  "Ba: 6s2; p for polarization"),

    # Lanthanides
    57: ({0, 1, 2},    {0, 1, 2, 3},   3,   "La: 5d1 6s2; f empty in ground state"),
    58: ({0, 2, 3},    {0, 1, 2, 3},   4,   "Ce: 4f1 5d1 6s2"),
    59: ({0, 2, 3},    {0, 1, 2, 3},   5,   "Pr: 4f3 6s2"),
    60: ({0, 2, 3},    {0, 1, 2, 3},   6,   "Nd: 4f4 6s2"),
    61: ({0, 2, 3},    {0, 1, 2, 3},   7,   "Pm: 4f5 6s2"),
    62: ({0, 2, 3},    {0, 1, 2, 3},   8,   "Sm: 4f6 6s2"),
    63: ({0, 2, 3},    {0, 1, 2, 3},   9,   "Eu: 4f7 6s2"),
    64: ({0, 2, 3},    {0, 1, 2, 3},  10,   "Gd: 4f7 5d1 6s2"),
    65: ({0, 2, 3},    {0, 1, 2, 3},  11,   "Tb: 4f9 6s2"),
    66: ({0, 2, 3},    {0, 1, 2, 3},  12,   "Dy: 4f10 6s2"),
    67: ({0, 2, 3},    {0, 1, 2, 3},  13,   "Ho: 4f11 6s2"),
    68: ({0, 2, 3},    {0, 1, 2, 3},  14,   "Er: 4f12 6s2"),
    69: ({0, 2, 3},    {0, 1, 2, 3},  15,   "Tm: 4f13 6s2"),
    70: ({0, 2, 3},    {0, 1, 2, 3},  16,   "Yb: 4f14 6s2"),
    71: ({0, 2, 3},    {0, 1, 2, 3},  17,   "Lu: 4f14 5d1 6s2"),

    # 5d transition metals: standard = s,d; extended adds p semi-core (5p)
    72: ({0, 2},    {0, 1, 2},   4,   "Hf: 5d2 6s2"),
    73: ({0, 2},    {0, 1, 2},   5,   "Ta: 5d3 6s2"),
    74: ({0, 2},    {0, 1, 2},   6,   "W: 5d4 6s2"),
    75: ({0, 2},    {0, 1, 2},   7,   "Re: 5d5 6s2"),
    76: ({0, 2},    {0, 1, 2},   8,   "Os: 5d6 6s2"),
    77: ({0, 2},    {0, 1, 2},   9,   "Ir: 5d7 6s2"),
    78: ({0, 2},    {0, 1, 2},  10,   "Pt: 5d9 6s1"),
    79: ({0, 2},    {0, 1, 2},  11,   "Au: 5d10 6s1"),
    80: ({0, 2},    {0, 1, 2},  12,   "Hg: 5d10 6s2"),

    # Post-5d main group (includes SOC-relevant heavy elements)
    81: ({0, 1},    {0, 1, 2},   3,   "Tl: 6s2 6p1; extended adds 5d"),
    82: ({0, 1},    {0, 1, 2},   4,   "Pb: 6s2 6p2; extended adds 5d"),
    83: ({0, 1},    {0, 1, 2},   5,   "Bi: 6s2 6p3; extended adds 5d"),
    84: ({0, 1},    {0, 1, 2},   6,   "Po: 6s2 6p4"),
    85: ({0, 1},    {0, 1, 2},   7,   "At: 6s2 6p5"),
    86: ({0, 1},    {0, 1, 2},   8,   "Rn: 6s2 6p6"),
}


# ============================================================================
# CORE SHELL COUNTING TABLE
# ============================================================================
# Number of core PRINCIPAL shells per l-channel for all-electron basis sets.
# This equals the number of radial functions to skip before reaching the
# first valence radial in that l-channel.
#
# Derived from the noble gas core configuration of each element:
#   Period 1: no core
#   Period 2: [He] core = 1s
#   Period 3: [Ne] core = 1s 2s 2p
#   Period 4: [Ar] core = 1s 2s 2p 3s 3p
#   Period 5: [Kr] core = 1s 2s 2p 3s 3p 3d 4s 4p
#   Period 6: [Xe] core = adds 5s 5p to [Kr]4d10
#
# For ECP/pseudopotential atoms, set n_core_radials = 0 since the
# core is removed by the ECP and the first radial IS valence.
#
# Format: Z -> {l: n_core_principal_shells}
# ============================================================================

CORE_SHELLS: Dict[int, Dict[int, int]] = {
    # Period 1: no core
    1:  {},                          # H
    2:  {},                          # He

    # Period 2: [He] core = 1s -> 1 core s-shell
    3:  {0: 1},                      # Li
    4:  {0: 1},                      # Be
    5:  {0: 1},                      # B
    6:  {0: 1},                      # C
    7:  {0: 1},                      # N
    8:  {0: 1},                      # O
    9:  {0: 1},                      # F
    10: {0: 1},                      # Ne

    # Period 3: [Ne] core = 1s 2s 2p -> 2 core s-shells, 1 core p-shell
    11: {0: 2, 1: 1},               # Na
    12: {0: 2, 1: 1},               # Mg
    13: {0: 2, 1: 1},               # Al
    14: {0: 2, 1: 1},               # Si
    15: {0: 2, 1: 1},               # P
    16: {0: 2, 1: 1},               # S
    17: {0: 2, 1: 1},               # Cl
    18: {0: 2, 1: 1},               # Ar

    # Period 4 main group: [Ar] core = 1s 2s 2p 3s 3p
    19: {0: 3, 1: 2},               # K
    20: {0: 3, 1: 2},               # Ca

    # 3d transition metals: [Ar] core (no core d-shells)
    21: {0: 3, 1: 2},               # Sc
    22: {0: 3, 1: 2},               # Ti
    23: {0: 3, 1: 2},               # V
    24: {0: 3, 1: 2},               # Cr
    25: {0: 3, 1: 2},               # Mn
    26: {0: 3, 1: 2},               # Fe
    27: {0: 3, 1: 2},               # Co
    28: {0: 3, 1: 2},               # Ni
    29: {0: 3, 1: 2},               # Cu
    30: {0: 3, 1: 2},               # Zn

    # Post-3d: [Ar]3d10 core -> 3d becomes core
    31: {0: 3, 1: 2, 2: 1},         # Ga
    32: {0: 3, 1: 2, 2: 1},         # Ge
    33: {0: 3, 1: 2, 2: 1},         # As
    34: {0: 3, 1: 2, 2: 1},         # Se
    35: {0: 3, 1: 2, 2: 1},         # Br
    36: {0: 3, 1: 2, 2: 1},         # Kr

    # Period 5: [Kr] core = [Ar]3d10 4s 4p
    37: {0: 4, 1: 3, 2: 1},         # Rb
    38: {0: 4, 1: 3, 2: 1},         # Sr

    # 4d transition metals: [Kr] core (no core 4d-shells)
    39: {0: 4, 1: 3, 2: 1},         # Y
    40: {0: 4, 1: 3, 2: 1},         # Zr
    41: {0: 4, 1: 3, 2: 1},         # Nb
    42: {0: 4, 1: 3, 2: 1},         # Mo
    43: {0: 4, 1: 3, 2: 1},         # Tc
    44: {0: 4, 1: 3, 2: 1},         # Ru
    45: {0: 4, 1: 3, 2: 1},         # Rh
    46: {0: 4, 1: 3, 2: 1},         # Pd
    47: {0: 4, 1: 3, 2: 1},         # Ag
    48: {0: 4, 1: 3, 2: 1},         # Cd

    # Post-4d: [Kr]4d10 core
    49: {0: 4, 1: 3, 2: 2},         # In
    50: {0: 4, 1: 3, 2: 2},         # Sn
    51: {0: 4, 1: 3, 2: 2},         # Sb
    52: {0: 4, 1: 3, 2: 2},         # Te
    53: {0: 4, 1: 3, 2: 2},         # I
    54: {0: 4, 1: 3, 2: 2},         # Xe

    # Period 6: [Xe] core = [Kr]4d10 5s 5p
    55: {0: 5, 1: 4, 2: 2},         # Cs
    56: {0: 5, 1: 4, 2: 2},         # Ba

    # Lanthanides: [Xe] core (no 4f in core by default)
    57: {0: 5, 1: 4, 2: 2},         # La
    58: {0: 5, 1: 4, 2: 2},         # Ce
    59: {0: 5, 1: 4, 2: 2},         # Pr
    60: {0: 5, 1: 4, 2: 2},         # Nd
    61: {0: 5, 1: 4, 2: 2},         # Pm
    62: {0: 5, 1: 4, 2: 2},         # Sm
    63: {0: 5, 1: 4, 2: 2},         # Eu
    64: {0: 5, 1: 4, 2: 2},         # Gd
    65: {0: 5, 1: 4, 2: 2},         # Tb
    66: {0: 5, 1: 4, 2: 2},         # Dy
    67: {0: 5, 1: 4, 2: 2},         # Ho
    68: {0: 5, 1: 4, 2: 2},         # Er
    69: {0: 5, 1: 4, 2: 2},         # Tm
    70: {0: 5, 1: 4, 2: 2},         # Yb
    71: {0: 5, 1: 4, 2: 2},         # Lu

    # 5d transition metals: [Xe]4f14 core -> 4f IS core
    72: {0: 5, 1: 4, 2: 2, 3: 1},   # Hf
    73: {0: 5, 1: 4, 2: 2, 3: 1},   # Ta
    74: {0: 5, 1: 4, 2: 2, 3: 1},   # W
    75: {0: 5, 1: 4, 2: 2, 3: 1},   # Re
    76: {0: 5, 1: 4, 2: 2, 3: 1},   # Os
    77: {0: 5, 1: 4, 2: 2, 3: 1},   # Ir
    78: {0: 5, 1: 4, 2: 2, 3: 1},   # Pt
    79: {0: 5, 1: 4, 2: 2, 3: 1},   # Au
    80: {0: 5, 1: 4, 2: 2, 3: 1},   # Hg

    # Post-5d: [Xe]4f14 5d10 core
    81: {0: 5, 1: 4, 2: 3, 3: 1},   # Tl
    82: {0: 5, 1: 4, 2: 3, 3: 1},   # Pb
    83: {0: 5, 1: 4, 2: 3, 3: 1},   # Bi
    84: {0: 5, 1: 4, 2: 3, 3: 1},   # Po
    85: {0: 5, 1: 4, 2: 3, 3: 1},   # At
    86: {0: 5, 1: 4, 2: 3, 3: 1},   # Rn
}


def get_core_shell_count(
    element: Union[str, int],
    l: int,
) -> int:
    """
    Get the number of core radial functions for a given element and l-channel.

    For all-electron basis sets, this is the number of radial functions
    of angular momentum l that describe core orbitals and must be skipped
    to reach the first valence radial.

    Parameters
    ----------
    element : str or int
        Element symbol or atomic number.
    l : int
        Angular momentum quantum number.

    Returns
    -------
    int
        Number of core radial functions to skip.
    """
    if isinstance(element, str):
        z = ELEMENT_Z.get(element)
        if z is None:
            return 0
    else:
        z = int(element)

    core_shells = CORE_SHELLS.get(z, {})
    return core_shells.get(l, 0)


# ============================================================================
# Convenience: transition metal s,p,d config for transport/topology
# ============================================================================

TM_WITH_P: Dict[int, Set[int]] = {
    z: {0, 1, 2} for z in range(21, 31)   # 3d: Sc-Zn
}
TM_WITH_P.update({
    z: {0, 1, 2} for z in range(39, 49)   # 4d: Y-Cd
})
TM_WITH_P.update({
    z: {0, 1, 2} for z in range(72, 81)   # 5d: Hf-Hg
})


# ============================================================================
# API functions
# ============================================================================

def get_valence_l(
    element: Union[str, int],
    extended: bool = False,
    include_tm_p: bool = False,
) -> Set[int]:
    """
    Get the set of valence angular momentum channels for an element.

    Parameters
    ----------
    element : str or int
        Element symbol (e.g., 'Sn') or atomic number (e.g., 50).
    extended : bool
        If True, include semi-core states. Default: False.
    include_tm_p : bool
        If True and element is a transition metal, include p-channel
        even in standard mode. Useful for transport/topology.
        Ignored if extended=True (which already includes p).

    Returns
    -------
    set of int
        Set of l-values (0=s, 1=p, 2=d, 3=f) for the valence manifold.

    Raises
    ------
    KeyError
        If element is not in the lookup table.
    """
    if isinstance(element, str):
        z = ELEMENT_Z.get(element)
        if z is None:
            raise KeyError(f"Unknown element symbol: {element}")
    else:
        z = int(element)

    if z not in VALENCE_CONFIG:
        raise KeyError(
            f"Element Z={z} not in valence configuration table. "
            f"Table covers Z=1 to Z={max(VALENCE_CONFIG.keys())}."
        )

    standard_l, extended_l, n_val, notes = VALENCE_CONFIG[z]

    if extended:
        return set(extended_l)

    result = set(standard_l)

    # Optionally add p to transition metals
    if include_tm_p and z in TM_WITH_P:
        result = result | TM_WITH_P[z]

    return result


def get_num_target_orbitals(
    element: Union[str, int],
    extended: bool = False,
    include_tm_p: bool = False,
    has_soc: bool = False,
) -> int:
    """
    Get the number of target orbitals for one atom of this element.

    This is sum of (2l+1) for each l in the valence set,
    doubled if SOC (spinor basis).
    """
    l_set = get_valence_l(element, extended=extended,
                          include_tm_p=include_tm_p)
    n_spatial = sum(2 * l + 1 for l in l_set)
    return n_spatial * 2 if has_soc else n_spatial


def compute_num_wann(
    atoms: List[Union[str, int]],
    extended: bool = False,
    include_tm_p: bool = False,
    has_soc: bool = False,
) -> int:
    """
    Compute total num_wann for a list of atoms in the unit cell.

    Examples
    --------
    >>> compute_num_wann(['Mg', 'B', 'B'])
    12
    >>> compute_num_wann(['Mg', 'B', 'B'], has_soc=True)
    24
    >>> compute_num_wann(['Sn', 'Sn'], has_soc=True)
    16
    """
    return sum(
        get_num_target_orbitals(atom, extended=extended,
                                include_tm_p=include_tm_p,
                                has_soc=has_soc)
        for atom in atoms
    )


def build_target_mask(
    basis_shells,
    extended: bool = False,
    include_tm_p: bool = False,
    has_soc: bool = False,
    verbose: bool = False,
    uses_ecp: Optional[Dict[int, bool]] = None,
    all_radials: Optional[bool] = None,
) -> np.ndarray:
    """
    Build the boolean target orbital mask from Crystal23 basis set info.

    For all-electron atoms, core radial functions are skipped using the
    CORE_SHELLS table before selecting the first valence radial. For ECP
    atoms (core removed by pseudopotential), the first radial IS valence.

    Parameters
    ----------
    basis_shells : list of ShellInfo or list of dict
        Parsed basis shell information. Each entry must have attributes
        or keys: 'atom_index', 'atom_Z', 'l', 'ao_indices'.
    extended : bool
        Include semi-core target orbitals.
    include_tm_p : bool
        Include p for transition metals.
    has_soc : bool
        If True, expand mask to spinor basis using block convention
        (spin-up block then spin-down block, matching Crystal23 SOC).
    verbose : bool
        Print classification details.
    uses_ecp : dict, optional
        Maps atom_index -> bool indicating whether the atom uses an ECP.
        If None, auto-detected from the basis set (atoms with fewer
        shells than expected for their Z are assumed to use ECPs).
    all_radials : bool or None
        If True, include ALL non-core radials of each valence l-channel
        as target (not just the first valence radial). This is essential
        for all-electron multi-zeta basis sets where valence band character
        is distributed across multiple radial functions. If False, include
        only the first valence radial (standard PDWF behavior, suitable
        for ECP/pseudopotential basis sets). If None (default), auto-detect:
        use all_radials=True for all-electron atoms, False for ECP atoms.

    Returns
    -------
    np.ndarray, dtype=bool
        Boolean mask of shape (num_ao,) or (2*num_ao,) for SOC.
    """
    # Support both dict and dataclass (ShellInfo) interfaces
    def _get(shell, key):
        if isinstance(shell, dict):
            return shell[key]
        return getattr(shell, key)

    if uses_ecp is None:
        uses_ecp = {}

    # Auto-detect ECP usage: if an atom has no core shells in the basis
    # but CORE_SHELLS says it should, it's likely using an ECP.
    # We detect this by checking if the number of s-shells is fewer than
    # the expected core s-shells + 1 (for the valence s).
    if not uses_ecp:
        uses_ecp = _auto_detect_ecp(basis_shells, _get)

    # Determine total number of AOs
    all_ao = set()
    for shell in basis_shells:
        all_ao.update(_get(shell, 'ao_indices'))
    num_ao = max(all_ao) + 1

    target_mask = np.zeros(num_ao, dtype=bool)

    # Track radial index for each (atom_index, l) pair
    # This counts how many shells of this (atom, l) we've seen
    radial_count = {}

    for shell in basis_shells:
        atom_idx = _get(shell, 'atom_index')
        atom_z = _get(shell, 'atom_Z')
        l = _get(shell, 'l')
        ao_idx = _get(shell, 'ao_indices')

        # Get valence l-channels for this element
        valence_l = get_valence_l(atom_z, extended=extended,
                                  include_tm_p=include_tm_p)

        key = (atom_idx, l)
        radial_idx = radial_count.get(key, 0)
        radial_count[key] = radial_idx + 1

        # Is this l in the valence set?
        if l not in valence_l:
            classification = "COMPLEMENT (polarization)"
        else:
            # Determine how many core radials to skip
            is_ecp = uses_ecp.get(atom_idx, False)
            if is_ecp:
                n_core = 0  # ECP: no core in basis, first radial IS valence
            else:
                core_shells = CORE_SHELLS.get(atom_z, {})
                n_core = core_shells.get(l, 0)

            # Determine whether to use all-radial mode for this atom
            if all_radials is None:
                # Auto: all-radials for all-electron, single for ECP
                use_all = not is_ecp
            else:
                use_all = all_radials

            if radial_idx < n_core:
                classification = "COMPLEMENT (core)"
            elif radial_idx == n_core:
                # This is the first valence radial: always TARGET
                target_mask[list(ao_idx)] = True
                classification = "TARGET"
            elif use_all:
                # All-radial mode: include additional valence radials
                target_mask[list(ao_idx)] = True
                classification = "TARGET (multi-zeta)"
            else:
                classification = "COMPLEMENT (multi-zeta)"

        if verbose:
            sym = ELEMENT_SYMBOLS[atom_z] if atom_z < len(ELEMENT_SYMBOLS) else f"Z={atom_z}"
            l_name = 'spdfgh'[l] if l < 6 else f'l={l}'
            print(f"  Atom {atom_idx} ({sym}), l={l_name}, radial {radial_idx}, "
                  f"AOs {min(ao_idx)}-{max(ao_idx)}: {classification}")

    if has_soc:
        # Block convention: [spin-up block | spin-down block]
        # Crystal23 SOC builds [[H_aa, H_ab], [H_ba, H_bb]] so
        # eigenvectors are [C_up(1..N), C_dn(1..N)]
        target_mask_soc = np.concatenate([target_mask, target_mask])
        return target_mask_soc

    return target_mask


def _auto_detect_ecp(basis_shells, _get) -> Dict[int, bool]:
    """
    Auto-detect which atoms use ECPs based on shell count vs expected core.

    For all-electron atoms, the number of s-shells should be at least
    n_core_s + 1 (core s-shells plus at least one valence s). If an atom
    has fewer s-shells than expected, it likely uses an ECP.

    Parameters
    ----------
    basis_shells : list
        Parsed basis shell information.
    _get : callable
        Accessor function for shell attributes.

    Returns
    -------
    dict
        Maps atom_index -> bool (True = uses ECP).
    """
    # Count s-shells per atom
    s_shell_count = {}
    atom_z_map = {}

    for shell in basis_shells:
        atom_idx = _get(shell, 'atom_index')
        atom_z = _get(shell, 'atom_Z')
        l = _get(shell, 'l')
        atom_z_map[atom_idx] = atom_z

        if l == 0:
            s_shell_count[atom_idx] = s_shell_count.get(atom_idx, 0) + 1

    uses_ecp = {}
    for atom_idx, z in atom_z_map.items():
        core_s = CORE_SHELLS.get(z, {}).get(0, 0)
        n_s = s_shell_count.get(atom_idx, 0)
        # If fewer s-shells than core_s + 1 (need at least 1 valence s),
        # the atom is using an ECP that removed core shells
        if core_s > 0 and n_s < core_s + 1:
            uses_ecp[atom_idx] = True
        else:
            uses_ecp[atom_idx] = False

    return uses_ecp


def summarize_config(
    atoms: List[Union[str, int]],
    extended: bool = False,
    include_tm_p: bool = False,
    has_soc: bool = False,
) -> str:
    """Return a human-readable summary of the target configuration."""
    lines = []
    lines.append("LCAO-PDWF Target Configuration Summary")
    lines.append("=" * 50)
    lines.append(f"Mode: {'extended (semi-core)' if extended else 'standard'}"
                 f"{' + TM p-channel' if include_tm_p else ''}")
    lines.append(f"SOC: {'Yes (spinor)' if has_soc else 'No'}")
    lines.append("")

    total = 0
    for i, atom in enumerate(atoms):
        if isinstance(atom, int):
            z = atom
            sym = ELEMENT_SYMBOLS[z] if z < len(ELEMENT_SYMBOLS) else f"Z={z}"
        else:
            sym = atom
            z = ELEMENT_Z[atom]

        l_set = get_valence_l(z, extended=extended, include_tm_p=include_tm_p)
        l_names = ', '.join('spdfgh'[l] for l in sorted(l_set))
        n = get_num_target_orbitals(z, extended=extended,
                                    include_tm_p=include_tm_p,
                                    has_soc=has_soc)
        total += n

        _, _, n_val, notes = VALENCE_CONFIG[z]
        lines.append(f"  Atom {i}: {sym} (Z={z})")
        lines.append(f"    Valence channels: {l_names}")
        lines.append(f"    Target AOs: {n}")
        lines.append(f"    Config: {notes}")

    lines.append("")
    lines.append(f"Total num_wann: {total}")
    lines.append("=" * 50)

    return '\n'.join(lines)
