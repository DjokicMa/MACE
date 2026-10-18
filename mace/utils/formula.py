"""Chemical formulas, for the CLI and the GUI alike: the formula materials.db
stores (``mace`` order), the reduced formula and Z the structure view and the
Database view show (Pauling ``electronegativity`` order, as pymatgen writes
them), the Hill formula of an ion fragment (``hill``) and of the structure
view (ase's ``hill-ase``), and ase's ``metal`` order the slab builder names
terminations with.

Standard library only: the GUI and the database code import it without ase.
"""
from __future__ import annotations

import re
from collections import Counter
from math import gcd
from typing import Dict, Iterable, Mapping, Optional, Tuple

# The elements H-Lr in atomic-number order (ELEMENTS[z - 1])
ELEMENTS = tuple(
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu "
    "Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba "
    "La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi "
    "Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr".split())

# Pauling electronegativities (CRC Handbook values, as pymatgen tabulates
# them); the noble gases without one sort last. Used only to order formulas.
PAULING = dict(zip(ELEMENTS, [
    2.20, 9, 0.98, 1.57, 2.04, 2.55, 3.04, 3.44, 3.98, 9, 0.93, 1.31, 1.61, 1.90, 2.19,
    2.58, 3.16, 9, 0.82, 1.00, 1.36, 1.54, 1.63, 1.66, 1.55, 1.83, 1.88, 1.91, 1.90,
    1.65, 1.81, 2.01, 2.18, 2.55, 2.96, 3.00, 0.82, 0.95, 1.22, 1.33, 1.6, 2.16, 1.9,
    2.2, 2.28, 2.20, 1.93, 1.69, 1.78, 1.96, 2.05, 2.1, 2.66, 2.6, 0.79, 0.89, 1.10,
    1.12, 1.13, 1.14, 1.13, 1.17, 1.2, 1.2, 1.1, 1.22, 1.23, 1.24, 1.25, 1.1, 1.27,
    1.3, 1.5, 2.36, 1.9, 2.2, 2.20, 2.28, 2.54, 2.00, 1.62, 2.33, 2.02, 2.0, 2.2, 2.2,
    0.7, 0.9, 1.1, 1.3, 1.5, 1.38, 1.36, 1.28, 1.13, 1.28, 1.3, 1.3, 1.3, 1.3, 1.3,
    1.3, 1.3]))

# materials.db's order: elements that conventionally come last (electronegative,
# in decreasing electropositivity so SiO2 / TiO2 / CaCO3 read naturally)
NONMETAL_ORDER = ['B', 'C', 'Si', 'Ge', 'N', 'P', 'As', 'Sb', 'H',
                  'Te', 'Se', 'S', 'At', 'I', 'Br', 'Cl', 'F', 'O']
# ...the others before them by atomic number, over the table materials.db
# formulas have always used (H-Rn); anything heavier sorts after those
_MACE_Z = {s: z for z, s in enumerate(ELEMENTS[:86], start=1)}

# ase.formula's non-metals (its "metal" order puts them last, alphabetically)
ASE_NON_METALS = ('H', 'He', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Si', 'P', 'S', 'Cl', 'Ar', 'Ge', 'As',
                  'Se', 'Br', 'Kr', 'Sb', 'Te', 'I', 'Xe', 'Po', 'At', 'Rn')

ORDERS = ("mace", "electronegativity", "hill", "hill-ase", "metal")


def count_symbols(symbols: Iterable[str]) -> Dict[str, int]:
    """{symbol: count} of a list of element symbols."""
    return dict(Counter(symbols))


def _ordered(counts: Mapping[str, int], order: str):
    if order == "mace":
        cations = sorted((e for e in counts if e not in NONMETAL_ORDER), key=lambda e: _MACE_Z.get(e, 999))
        return cations + [e for e in NONMETAL_ORDER if e in counts]
    if order == "electronegativity":
        return sorted(counts, key=lambda e: (PAULING.get(e, 9), e))
    if order == "hill":
        if "C" not in counts:
            return sorted(counts)
        return ["C"] + (["H"] if "H" in counts else []) + sorted(e for e in counts if e not in ("C", "H"))
    if order == "hill-ase":
        return [e for e in "CH" if e in counts] + sorted(e for e in counts if e not in ("C", "H"))
    if order == "metal":
        return sorted(e for e in counts if e not in ASE_NON_METALS) + sorted(e for e in counts if e in ASE_NON_METALS)
    raise ValueError(f"order is one of {', '.join(ORDERS)}")


def format_formula(counts: Mapping[str, int], order: str = "mace") -> str:
    """The formula of ``counts`` ({symbol: count}), a count of 1 left out:
    ``mace`` (materials.db: cations by atomic number, then the anions in a
    fixed electronegativity order), ``electronegativity`` (ascending Pauling,
    as pymatgen), ``hill`` (C, H, then alphabetical; all alphabetical with no
    carbon), ``hill-ase`` (ase's: C and H first even with no carbon, so HF,
    not FH) or ``metal`` (ase's: metals alphabetically, then non-metals
    alphabetically)."""
    return "".join(e + (str(counts[e]) if counts[e] != 1 else "") for e in _ordered(counts, order))


def reduce_counts(counts: Mapping[str, int]) -> Tuple[Dict[str, int], int]:
    """(counts divided by their greatest common divisor, that divisor Z)."""
    z = 0
    for c in counts.values():
        z = gcd(z, c)
    if not z:
        return {}, 0
    return {e: c // z for e, c in counts.items()}, z


def reduced_formula(counts: Mapping[str, int], order: str = "electronegativity") -> Tuple[str, int]:
    """(reduced formula, Z): SiO2, GaAs, CsPbI3, NaHCO3 with the number of
    formula units; a single element reads C with Z = 8; ("", 0) for none."""
    reduced, z = reduce_counts(counts)
    return format_formula(reduced, order), z


def parse_formula(formula: str) -> Optional[Dict[str, int]]:
    """'F4Li4' (a cell formula, as materials.db stores it) -> {'F': 4, 'Li': 4}.
    None when the text is not a plain formula of known elements (brackets,
    charges, ...)."""
    if not formula or not re.fullmatch(r"(?:[A-Z][a-z]?\d*)+", formula):
        return None
    counts: Dict[str, int] = {}
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        if el not in PAULING:
            return None
        counts[el] = counts.get(el, 0) + (int(n) if n else 1)
    return counts
