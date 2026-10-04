"""Molecular fragments, known ions and the Mulliken lines charged runs are checked with.

Standard library only (like calc_detection.py): Crystal_d12 scripts load it
when ``mace`` is not importable, and the output side (property_extractor)
uses the same code.

Fragments are found by connectivity, not atom order: two atoms are bonded when
their distance is below ``scale`` x the sum of their covalent radii (Cordero et
al., Dalton Trans. 2008, 2832 - the table ase uses). In a periodic cell the
neighbour search runs over image shifts along the periodic directions only,
and a union-find that carries each atom's integer image offset marks a
fragment "extended" when it is bonded to its own image (a framework or chain,
not a molecule).

The cations of ION_TABLE are never bonded to anything: in an ion/solvent
cluster a Li+ sits ~1.9-2.0 A from the solvent's O, inside the covalent-radius
cut-off, and must stay its own fragment. That is why ``classify(auto=True)``
refuses what looks like an ionic solid (lone O/F/... atoms, or a table metal
next to an extended fragment): there, removing the metal's bonds invents ions.
"""

import math
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

# Cordero et al. 2008 covalent radii (A), H-Rn; C is the sp3 value, Mn/Fe/Co
# the low-spin ones (as in ase.data.covalent_radii).
COVALENT_RADII = {
    "H": 0.31, "He": 0.28, "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66,
    "F": 0.57, "Ne": 0.58, "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05,
    "Cl": 1.02, "Ar": 1.06, "K": 2.03, "Ca": 1.76, "Sc": 1.70, "Ti": 1.60, "V": 1.53, "Cr": 1.39,
    "Mn": 1.39, "Fe": 1.32, "Co": 1.26, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22, "Ga": 1.22, "Ge": 1.20,
    "As": 1.19, "Se": 1.20, "Br": 1.20, "Kr": 1.16, "Rb": 2.20, "Sr": 1.95, "Y": 1.90, "Zr": 1.75,
    "Nb": 1.64, "Mo": 1.54, "Tc": 1.47, "Ru": 1.46, "Rh": 1.42, "Pd": 1.39, "Ag": 1.45, "Cd": 1.44,
    "In": 1.42, "Sn": 1.39, "Sb": 1.39, "Te": 1.38, "I": 1.39, "Xe": 1.40, "Cs": 2.44, "Ba": 2.15,
    "La": 2.07, "Ce": 2.04, "Pr": 2.03, "Nd": 2.01, "Pm": 1.99, "Sm": 1.98, "Eu": 1.98, "Gd": 1.96,
    "Tb": 1.94, "Dy": 1.92, "Ho": 1.92, "Er": 1.89, "Tm": 1.90, "Yb": 1.87, "Lu": 1.87, "Hf": 1.75,
    "Ta": 1.70, "W": 1.62, "Re": 1.51, "Os": 1.44, "Ir": 1.41, "Pt": 1.36, "Au": 1.36, "Hg": 1.32,
    "Tl": 1.45, "Pb": 1.46, "Bi": 1.48, "Po": 1.40, "At": 1.50, "Rn": 1.50,
}

SYMBOLS = ["X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P",
           "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
           "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh",
           "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
           "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re",
           "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn"]
ATOMIC_NUMBER = {s: z for z, s in enumerate(SYMBOLS) if z}

# Elements that are not metals; a lone one of these left as its own fragment
# means the structure is an ionic solid (oxide, fluoride, ...), not a cluster.
NON_METALS = {"H", "He", "B", "C", "N", "O", "F", "Ne", "Si", "P", "S", "Cl", "Ar", "As", "Se",
              "Br", "Kr", "Te", "I", "Xe", "At", "Rn"}
NOBLE_GASES = {"He", "Ne", "Ar", "Kr", "Xe", "Rn"}

# Known ions. Cations: one atom, charge removed from the valence s shell.
# Anions: exact formula, the coordination ("degrees") of the named centre
# atoms, and the terminal atoms - (element, bonded-to element, count), each
# with exactly one bond - over which the extra electron(s) are spread on the
# valence p shell (the group's FSI- recipe: +0.25 e on each sulfonyl O).
ION_TABLE = {
    "Li+": {"formula": {"Li": 1}, "charge": 1},
    "Na+": {"formula": {"Na": 1}, "charge": 1},
    "K+": {"formula": {"K": 1}, "charge": 1},
    "Mg2+": {"formula": {"Mg": 1}, "charge": 2},
    "Ca2+": {"formula": {"Ca": 1}, "charge": 2},
    "Zn2+": {"formula": {"Zn": 1}, "charge": 2},
    "FSI-": {"formula": {"F": 2, "N": 1, "O": 4, "S": 2}, "charge": -1,
             "degrees": {"N": 2, "S": 4}, "terminal": ("O", "S", 4)},
    "TFSI-": {"formula": {"C": 2, "F": 6, "N": 1, "O": 4, "S": 2}, "charge": -1,
              "degrees": {"N": 2, "S": 4, "C": 4}, "terminal": ("O", "S", 4)},
    "OTf-": {"formula": {"C": 1, "F": 3, "O": 3, "S": 1}, "charge": -1,
             "degrees": {"S": 4, "C": 4}, "terminal": ("O", "S", 3)},
    "ClO4-": {"formula": {"Cl": 1, "O": 4}, "charge": -1,
              "degrees": {"Cl": 4}, "terminal": ("O", "Cl", 4)},
    "NO3-": {"formula": {"N": 1, "O": 3}, "charge": -1,
             "degrees": {"N": 3}, "terminal": ("O", "N", 3)},
    "PF6-": {"formula": {"F": 6, "P": 1}, "charge": -1,
             "degrees": {"P": 6}, "terminal": ("F", "P", 6)},
    "BF4-": {"formula": {"B": 1, "F": 4}, "charge": -1,
             "degrees": {"B": 4}, "terminal": ("F", "B", 4)},
}
TABLE_METALS = {next(iter(v["formula"])) for v in ION_TABLE.values() if v["charge"] > 0}

AUTO_SOLID_MESSAGE = ("auto ion charges are for molecular ions/clusters; this looks like an ionic "
                      "solid")


class IonChargeError(ValueError):
    """A charge set-up MACE will not write, with the reason."""


def element_symbol(label: str) -> str:
    """'LI' / 'li' / 'Li' / 'Li1' -> 'Li'."""
    m = re.match(r"([A-Za-z]{1,2})", str(label).strip())
    if not m:
        return str(label)
    s = m.group(1)
    s = s[0].upper() + s[1:].lower()
    if s not in ATOMIC_NUMBER and s[0] in ATOMIC_NUMBER:
        return s[0]
    return s


def formula_string(counts: Dict[str, int]) -> str:
    """Hill order: C, H, then alphabetical; e.g. F2NO4S2."""
    keys = sorted(counts)
    if "C" in counts:
        keys = ["C"] + (["H"] if "H" in counts else []) + [k for k in keys if k not in ("C", "H")]
    return "".join(f"{k}{counts[k] if counts[k] > 1 else ''}" for k in keys)


# ----------------------------------------------------------------- geometry

def _inverse3(m):
    a, b, c = m
    det = (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
           + a[2] * (b[0] * c[1] - b[1] * c[0]))
    if abs(det) < 1e-12:
        raise IonChargeError("the cell matrix is singular")
    inv = [[(b[1] * c[2] - b[2] * c[1]) / det, (a[2] * c[1] - a[1] * c[2]) / det, (a[1] * b[2] - a[2] * b[1]) / det],
           [(b[2] * c[0] - b[0] * c[2]) / det, (a[0] * c[2] - a[2] * c[0]) / det, (a[2] * b[0] - a[0] * b[2]) / det],
           [(b[0] * c[1] - b[1] * c[0]) / det, (a[1] * c[0] - a[0] * c[1]) / det, (a[0] * b[1] - a[1] * b[0]) / det]]
    return inv


def cell_matrix(a, b, c, alpha, beta, gamma):
    """Lattice vectors (rows, A) from cell parameters, a along x, b in the xy plane."""
    al, be, ga = (math.radians(x) for x in (alpha, beta, gamma))
    va = (a, 0.0, 0.0)
    vb = (b * math.cos(ga), b * math.sin(ga), 0.0)
    cx = c * math.cos(be)
    cy = c * (math.cos(al) - math.cos(be) * math.cos(ga)) / math.sin(ga)
    cz = math.sqrt(max(c * c - cx * cx - cy * cy, 0.0))
    return [va, vb, (cx, cy, cz)]


def _radius(symbol: str) -> float:
    try:
        return COVALENT_RADII[symbol]
    except KeyError:
        raise IonChargeError(f"no covalent radius for element {symbol!r}")


def find_fragments(symbols: Sequence[str], xyz: Sequence[Sequence[float]], cell=None,
                   periodic_dirs: Sequence[int] = (), scale: float = 1.15,
                   exclude_metals: bool = True) -> List[dict]:
    """Connected fragments of a structure.

    symbols: element symbols; xyz: Cartesian coordinates (A); cell: three
    lattice vectors (rows, A) when periodic_dirs is not empty - a SLAB passes
    (0, 1), a POLYMER (0,), a 3D cell (0, 1, 2); the non-periodic rows only
    have to make the matrix invertible. Atoms of TABLE_METALS are never
    bonded when exclude_metals.

    Returns fragments ordered by their first atom, each
    {"atoms": [0-based indices], "formula": str, "extended": bool,
     "neighbours": {i: [j, ...]}} (neighbours by bond, images included).
    """
    syms = [element_symbol(s) for s in symbols]
    n = len(syms)
    pts = [tuple(float(v) for v in p) for p in xyz]
    if len(pts) != n:
        raise IonChargeError("symbols and coordinates differ in length")
    radii = [_radius(s) for s in syms]
    max_cut = scale * 2 * max(radii) if radii else 0.0

    periodic = tuple(sorted(set(periodic_dirs)))
    shifts = [(0, 0, 0)]
    frac = None
    if periodic:
        if cell is None:
            raise IonChargeError("a periodic structure needs its cell")
        cell = [tuple(float(x) for x in row) for row in cell]
        inv = _inverse3(cell)  # columns: reciprocal vectors / 2pi
        # fractional coordinates, wrapped along the periodic directions
        frac = []
        for p in pts:
            f = [sum(p[k] * inv[k][d] for k in range(3)) for d in range(3)]
            for d in periodic:
                f[d] -= math.floor(f[d])
            frac.append(f)
        pts = [tuple(sum(f[d] * cell[d][k] for d in range(3)) for k in range(3)) for f in frac]
        ranges = []
        for d in range(3):
            if d in periodic:
                recip = math.sqrt(sum(inv[k][d] ** 2 for k in range(3)))
                width = 1.0 / recip
                ranges.append(range(-(math.ceil(max_cut / width) + 1), math.ceil(max_cut / width) + 2))
            else:
                ranges.append(range(0, 1))
        shifts = [(i, j, k) for i in ranges[0] for j in ranges[1] for k in ranges[2]]
        shift_vecs = {s: tuple(sum(s[d] * cell[d][k] for d in range(3)) for k in range(3)) for s in shifts}
    else:
        shift_vecs = {(0, 0, 0): (0.0, 0.0, 0.0)}

    parent = list(range(n))
    offset = [(0, 0, 0)] * n  # image of atom i that lies in its parent's copy
    extended_roots = set()

    def find(i):
        if parent[i] == i:
            return i, (0, 0, 0)
        root, off = find(parent[i])
        total = tuple(a + b for a, b in zip(offset[i], off))
        parent[i] = root
        offset[i] = total
        return root, total

    neighbors: Dict[int, List[int]] = {i: [] for i in range(n)}
    for i in range(n):
        if exclude_metals and syms[i] in TABLE_METALS:
            continue
        for j in range(i, n):
            if exclude_metals and syms[j] in TABLE_METALS:
                continue
            cut = scale * (radii[i] + radii[j])
            cut2 = cut * cut
            for s in shifts:
                if i == j and s == (0, 0, 0):
                    continue
                v = shift_vecs[s]
                d2 = sum((pts[j][k] + v[k] - pts[i][k]) ** 2 for k in range(3))
                if d2 > cut2 or d2 < 1e-8:
                    continue
                neighbors[i].append(j)
                if i != j:
                    neighbors[j].append(i)
                # j at image s is bonded to i at image 0
                ri, oi = find(i)
                rj, oj = find(j)
                want = tuple(a + b for a, b in zip(s, oi))  # translation putting j in ri's copy
                if ri != rj:
                    parent[rj] = ri
                    offset[rj] = tuple(w - b for w, b in zip(want, oj))
                    if rj in extended_roots:
                        extended_roots.add(ri)
                elif want != oj:
                    extended_roots.add(ri)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i)[0], []).append(i)
    frags = []
    for root, atoms in groups.items():
        counts = Counter(syms[i] for i in atoms)
        frags.append({"atoms": sorted(atoms), "formula": formula_string(counts),
                      "extended": root in extended_roots,
                      "neighbors": {i: neighbors[i] for i in atoms}})
    frags.sort(key=lambda f: f["atoms"][0])
    return frags


# ----------------------------------------------------------- classification

def classify_fragment(fragment: dict, symbols: Sequence[str]) -> Optional[str]:
    """The ION_TABLE name a fragment matches, or None (neutral)."""
    if fragment.get("extended"):
        return None
    syms = [element_symbol(s) for s in symbols]
    atoms = fragment["atoms"]
    counts = Counter(syms[i] for i in atoms)
    nb = fragment["neighbors"]
    for name, ion in ION_TABLE.items():
        if dict(counts) != ion["formula"]:
            continue
        if any(len(nb[i]) != deg for el, deg in ion.get("degrees", {}).items()
               for i in atoms if syms[i] == el):
            continue
        term = ion.get("terminal")
        if term:
            el, to, count = term
            ends = [i for i in atoms if syms[i] == el and len(nb[i]) == 1 and syms[nb[i][0]] == to]
            if len(ends) != count:
                continue
        return name
    return None


# Valences an atom of a closed-shell neutral molecule can have (hypervalent
# S/P/N/halogens included, so DMSO, sulfones, nitro groups and SO2F2 pass).
VALENCES = {"H": (1,), "B": (3,), "C": (4,), "N": (3, 5), "O": (2,), "F": (1,), "Si": (4,),
            "P": (3, 5), "S": (2, 4, 6), "Cl": (1, 3, 5, 7), "Br": (1, 3, 5, 7), "I": (1, 3, 5, 7),
            "Se": (2, 4, 6), "As": (3, 5), "Ge": (4,), "Sn": (4,)}
_FREE_VALENCE_BUDGET = 200000


def free_valence(fragment: dict, symbols: Sequence[str]) -> Optional[int]:
    """Valences a fragment leaves unpaired at best, or None when it cannot be judged.

    Every bond found has order 1-3; an atom's bond orders must add up to one
    of its VALENCES. 0 means some Lewis structure makes the fragment a
    closed-shell neutral molecule (EC, water, SO2F2, DMSO, benzene); more
    means it cannot be one - an anion such as CO3 2- (2), OH- or a
    carboxylate (1) - or a radical. None: an element without a VALENCES entry,
    an atom with more bonds than its largest valence (a bond-length artefact),
    an extended fragment, or the search budget ran out.
    """
    if fragment.get("extended"):
        return None
    syms = [element_symbol(s) for s in symbols]
    atoms = fragment["atoms"]
    nb = fragment["neighbors"]
    residues = {}  # atom -> sorted allowed (valence - bonds)
    for i in atoms:
        opts = VALENCES.get(syms[i])
        if opts is None:
            return None
        deg = len(set(nb[i]))
        if len(nb[i]) != deg:  # bonded to its own image twice: not a molecule
            return None
        allowed = sorted(v - deg for v in opts if v >= deg)
        if not allowed:
            return None
        residues[i] = allowed
    # extra bond order (0-2 per bond) only where both ends have room
    room = {i for i in atoms if residues[i][-1] > 0}
    edges = sorted({(min(i, j), max(i, j)) for i in room for j in nb[i] if j in room and j != i})
    last_edge = {}
    for k, (i, j) in enumerate(edges):
        last_edge[i] = k
        last_edge[j] = k
    load = {i: 0 for i in atoms}

    def leftover(i):
        for r in residues[i]:
            if r >= load[i]:
                return r - load[i]
        return None

    base = 0
    for i in atoms:
        if i not in last_edge:
            base += residues[i][0]
    best = [None]
    nodes = [0]

    def search(k, partial):
        nodes[0] += 1
        if nodes[0] > _FREE_VALENCE_BUDGET:
            raise _Budget()
        if best[0] is not None and partial >= best[0]:
            return
        if k == len(edges):
            best[0] = partial
            return
        i, j = edges[k]
        for x in (2, 1, 0):
            if load[i] + x > residues[i][-1] or load[j] + x > residues[j][-1]:
                continue
            load[i] += x
            load[j] += x
            add, ok = 0, True
            for a in (i, j):
                if last_edge[a] == k:
                    left = leftover(a)
                    if left is None:
                        ok = False
                        break
                    add += left
            if ok:
                search(k + 1, partial + add)
            load[i] -= x
            load[j] -= x
            if best[0] == 0:
                return

    try:
        search(0, base)
    except _Budget:
        return None
    return best[0]


class _Budget(Exception):
    pass


def terminal_atoms(fragment: dict, symbols: Sequence[str], ion: str) -> List[int]:
    """The atoms an ion's charge is put on: the cation itself, or the anion's terminal atoms."""
    syms = [element_symbol(s) for s in symbols]
    spec = ION_TABLE[ion]
    if "terminal" not in spec:
        return list(fragment["atoms"])
    el, to, _ = spec["terminal"]
    nb = fragment["neighbors"]
    return [i for i in fragment["atoms"] if syms[i] == el and len(nb[i]) == 1 and syms[nb[i][0]] == to]


def classify(symbols: Sequence[str], xyz, cell=None, periodic_dirs: Sequence[int] = (),
             scale: float = 1.15, auto: bool = True) -> List[dict]:
    """Fragments with their ion assignment.

    Each fragment dict gains "ion" (ION_TABLE name or None), "charge" (formal,
    0 for neutral) and "sites" (0-based atoms carrying the charge). With
    auto=True, structures that look like ionic solids are refused
    (IonChargeError, AUTO_SOLID_MESSAGE).
    """
    syms = [element_symbol(s) for s in symbols]
    frags = find_fragments(syms, xyz, cell, periodic_dirs, scale)
    if auto:
        lone = sorted({syms[f["atoms"][0]] for f in frags
                       if len(f["atoms"]) == 1 and syms[f["atoms"][0]] in NON_METALS
                       and syms[f["atoms"][0]] not in NOBLE_GASES and syms[f["atoms"][0]] != "H"})
        if lone:
            raise IonChargeError(f"{AUTO_SOLID_MESSAGE}: lone {', '.join(lone)} atom(s) are bonded "
                                 f"to nothing ({'/'.join(sorted(TABLE_METALS))} are never bonded)")
        # a table metal touching an extended fragment or a bare non-metal atom
        bare_or_ext = set()
        for f in frags:
            if f["extended"] or (len(f["atoms"]) == 1 and syms[f["atoms"][0]] not in TABLE_METALS):
                bare_or_ext.update(f["atoms"])
        if bare_or_ext:
            touch = find_fragments(syms, xyz, cell, periodic_dirs, scale, exclude_metals=False)
            for f in touch:
                for i in f["atoms"]:
                    if syms[i] in TABLE_METALS and any(j in bare_or_ext for j in f["neighbors"][i]):
                        raise IonChargeError(f"{AUTO_SOLID_MESSAGE}: {syms[i]} (atom {i + 1}) is bonded "
                                             "to an extended framework")
    for f in frags:
        ion = classify_fragment(f, syms)
        f["ion"] = ion
        f["charge"] = ION_TABLE[ion]["charge"] if ion else 0
        f["sites"] = terminal_atoms(f, syms, ion) if ion else []
    return frags


def format_fragment_report(frags: List[dict], symbols: Sequence[str]) -> List[str]:
    """One line per kind of fragment: count x formula -> ion and charge."""
    kinds = Counter((f["formula"], f.get("ion"), f.get("charge", 0), f["extended"]) for f in frags)
    lines = []
    for (formula, ion, charge, ext), count in sorted(kinds.items(), key=lambda kv: kv[0][0]):
        what = ion if ion else ("extended (neutral)" if ext else "neutral")
        lines.append(f"{count} x {formula}: {what}" + (f" ({charge:+d} each)" if charge else ""))
    return lines


# -------------------------------------------------------------- output text

_MULLIKEN_COUNT = re.compile(
    r"(?:^[ \t]*(ALPHA\+BETA|ALPHA-BETA)[ \t]+ELECTRONS[ \t]*\n)?"
    r"[ \t]*MULLIKEN POPULATION ANALYSIS[^\n]*?NO\. OF ELECTRONS[ \t]+([-\d.]+)",
    re.MULTILINE)


def mulliken_electron_count(text: str) -> Optional[float]:
    """The electron count of the last Mulliken population print, or None.

    This is the real count: DOPING leaves the header's "N. OF ELECTRONS PER
    CELL" at the value before doping. The ALPHA-BETA print's line is the
    spin (nalpha - nbeta), never the count, so it is skipped.
    """
    count = None
    for match in _MULLIKEN_COUNT.finditer(text):
        if match.group(1) == "ALPHA-BETA":
            continue
        try:
            count = float(match.group(2))
        except ValueError:
            continue
    return count


_PP_VALENCE = re.compile(r"ATOMIC NUMBER\s+(\d+)\s*,\s*NUCLEAR CHARGE\s+([-\d.]+)")


def valence_charges(text: str) -> Dict[int, float]:
    """{atomic number: valence charge} of the pseudopotential atoms in a CRYSTAL .out.

    Read from the PSEUDOPOTENTIAL INFORMATION block, e.g. "ATOMIC NUMBER  82,
    NUCLEAR CHARGE 22.000, GIVEN IN INPUT". The key is the plain atomic
    number (Z % 100). All-electron atoms are not listed.
    """
    found = {}
    for z, charge in _PP_VALENCE.findall(text):
        try:
            found[int(z) % 100] = float(charge)
        except ValueError:
            continue
    return found


def valence_of(z: int, valence: Optional[Dict] = None) -> Optional[float]:
    """The charge a Mulliken population of an atom with Z-column label z is measured against.

    The pseudopotential charge from ``valence`` (valence_charges) when it
    lists the element; otherwise Z for an all-electron label (Z, 100 + Z or
    1000 + Z), and None for a 200 + Z or 300 + Z label - a valence-only (ECP)
    atom whose valence charge is not known, which must be flagged rather than
    measured against Z. CRYSTAL takes a pseudopotential only on a label above
    200 (manual p.25: NAT < 200 or > 1000 all-electron, NAT = Z + 200 or
    Z + 300 valence only), so a 1xx label never carries one.
    """
    label = int(z)
    z = label % 100
    for key in (z, str(z)):
        if valence and key in valence:
            return float(valence[key])
    if 200 <= label < 1000:
        return None
    return float(z)


VALENCE_UNKNOWN_FLAG = ("pseudopotential atom whose valence charge the .out does not give "
                        "(no PSEUDOPOTENTIAL INFORMATION line for it): charge not computed")


_MULLIKEN_HEAD = re.compile(
    r"(?:^[ \t]*(ALPHA\+BETA|ALPHA-BETA)[ \t]+ELECTRONS[ \t]*\n)?"
    r"[ \t]*MULLIKEN POPULATION ANALYSIS[^\n]*?NO\. OF ELECTRONS[ \t]+[-\d.]+[^\n]*\n",
    re.MULTILINE)
_ATOM_ROW = re.compile(r"^\s*(\d+)\s+([A-Z]{1,2})\s+(\d+)\s+(-?\d+\.\d+)")


def mulliken_atoms(text: str) -> Tuple[List[dict], List[dict]]:
    """The last Mulliken atom table of each kind: (populations, spins).

    Each list holds {"atom": label, "element": symbol, "z": Z column,
    "value": CHARGE column} - the electron population for the ALPHA+BETA (or
    unlabelled) print, the spin population for the ALPHA-BETA print (empty
    for closed-shell runs).
    """
    pops: List[dict] = []
    spins: List[dict] = []
    for head in _MULLIKEN_HEAD.finditer(text):
        rows = []
        started = False
        for line in text[head.end():].splitlines():
            m = _ATOM_ROW.match(line)
            if m:
                started = True
                rows.append({"atom": int(m.group(1)), "element": element_symbol(m.group(2)),
                             "z": int(m.group(3)), "value": float(m.group(4))})
                continue
            stripped = line.strip()
            if not stripped or stripped.startswith("ATOM"):
                if started and not stripped:
                    # a blank line ends the A.O. table once rows began, unless
                    # the next line is a continuation (rows are separated by
                    # wrapped population lines, never blank lines)
                    break
                continue
            if started and re.match(r"^[-\d.\s]+$", stripped):
                continue  # wrapped A.O. populations
            if started:
                break
        if head.group(1) == "ALPHA-BETA":
            spins = rows
        else:
            pops = rows
    return pops, spins


_DIMENSIONALITY = re.compile(r"DIMENSIONALITY OF THE SYSTEM\s+(\d)")
_LATTICE_VALUES = re.compile(r"^\s*A\s+B\s+C\s+ALPHA\s+BETA\s+GAMMA\s*$")
_ATOM_LINE = re.compile(r"^\s*\d+\s+[TF]\s+(\d+)\s+([A-Z]{1,2})\s+(\S+)\s+(\S+)\s+(\S+)\s*$")


def final_structure(text: str) -> Optional[dict]:
    """The last geometry an .out prints: atoms (Cartesian A), cell and periodic directions.

    Read from the last "ATOMS IN THE ASYMMETRIC UNIT ... ATOMS IN THE UNIT
    CELL" table (every atom of the cell, in the order the Mulliken analysis
    uses), its column header (X/A fractional, X(ANGSTROM) Cartesian) and the
    lattice parameters printed above it. None when there is no such table.
    """
    lines = text.splitlines()
    idx = [i for i, l in enumerate(lines) if "ATOMS IN THE ASYMMETRIC UNIT" in l]
    if not idx:
        return None
    start = idx[-1]
    dim = 3
    lattice = None
    for j in range(start, max(start - 40, -1), -1):
        if lattice is None and _LATTICE_VALUES.match(lines[j]) and j + 1 < len(lines):
            try:
                lattice = [float(x) for x in lines[j + 1].split()[:6]]
            except ValueError:
                lattice = None
        m = _DIMENSIONALITY.search(lines[j])
        if m:
            dim = int(m.group(1))
            break
    header = lines[start + 1].split()[1:] if start + 1 < len(lines) else []
    frac_cols = [("/" in h) for h in header[:3]]
    symbols, zs, coords = [], [], []
    for line in lines[start + 3:]:
        m = _ATOM_LINE.match(line)
        if not m:
            if symbols:
                break
            continue
        try:
            vals = [float(m.group(k)) for k in (3, 4, 5)]
        except ValueError:
            return None
        zs.append(int(m.group(1)))
        symbols.append(element_symbol(m.group(2)))
        coords.append(vals)
    if not symbols:
        return None
    periodic = {3: (0, 1, 2), 2: (0, 1), 1: (0,), 0: ()}.get(dim, ())
    cell = None
    if periodic:
        if not lattice or len(lattice) < 6:
            return None
        cell = cell_matrix(*lattice)
    xyz = []
    for vals in coords:
        p = [0.0, 0.0, 0.0]
        for d in range(3):
            if d < len(frac_cols) and frac_cols[d] and cell:
                for k in range(3):
                    p[k] += vals[d] * cell[d][k]
            else:
                p[d] += vals[d]
        xyz.append(tuple(p))
    return {"symbols": symbols, "z": zs, "xyz": xyz, "cell": cell, "periodic_dirs": periodic}


FRAGMENT_DEVIATION_FLAG = 0.5  # e; a heuristic, not a test


def fragment_charges(text: str, force: bool = False) -> Dict[str, object]:
    """Per-fragment Mulliken charge and spin of a run, when the run was charged.

    Computed when the Mulliken count differs from the sum of valence charges,
    a CHARGE NORMALIZATION FACTOR other than 1 was printed (a CHEMOD guess was
    renormalised; ignored after a GUESSP density-matrix restart), CRYSTAL warned "UNIT CELL NOT NEUTRAL", or force=True;
    otherwise {}. Returns {"electron_count": {...}, "fragment_charges": [...]}.
    A fragment's charge is the sum over its atoms of (valence charge -
    Mulliken population); a deviation of more than FRAGMENT_DEVIATION_FLAG
    from the formal charge of a known ion is flagged (electron delocalised).
    """
    count = mulliken_electron_count(text)
    if count is None:
        return {}
    pops, spins = mulliken_atoms(text)
    if not pops:
        return {}
    pp = valence_charges(text)
    ref = [valence_of(a["z"], pp) for a in pops]
    unknown = [f"{a['element']} (atom {a['atom']})" for a, r in zip(pops, ref) if r is None]
    zsum = None if unknown else sum(ref)
    factors = [float(x) for x in re.findall(r"CHARGE NORMALIZATION FACTOR\s+([-\d.]+)", text)]
    # After a density-matrix restart (GUESSP) CRYSTAL renormalises the
    # projected density every cycle and does not read CHEMOD, so the factor
    # says nothing about charge; net charge and UNIT CELL NOT NEUTRAL still do.
    if "RESTART FROM A PREVIOUS DENSITY MATRIX" in text:
        factors = []
    renormalized = any(abs(f - 1.0) > 1e-6 for f in factors)
    not_neutral = "UNIT CELL NOT NEUTRAL" in text
    net = None if zsum is None else round(zsum - count, 6)
    if not (force or (net is not None and abs(net) > 1e-3) or renormalized or not_neutral):
        return {}
    result: Dict[str, object] = {"electron_count": {
        "mulliken": count, "valence_sum": zsum, "net_charge": net,
        "charge_normalization_factors": sorted({f for f in factors if abs(f - 1.0) > 1e-6}),
        "unit_cell_not_neutral": not_neutral}}
    if unknown:
        result["electron_count"]["valence_unknown"] = unknown
    geom = final_structure(text)
    if geom is None or len(geom["symbols"]) != len(pops):
        return result
    try:
        frags = classify(geom["symbols"], geom["xyz"], geom["cell"], geom["periodic_dirs"], auto=False)
    except IonChargeError:
        return result
    spin_of = {a["atom"]: a["value"] for a in spins}
    entries = []
    for f in frags:
        s = sum(spin_of.get(i + 1, 0.0) for i in f["atoms"]) if spins else None
        if any(ref[i] is None for i in f["atoms"]):
            entries.append({"formula": f["formula"], "atoms": [i + 1 for i in f["atoms"]], "ion": f["ion"],
                            "extended": f["extended"], "formal_charge": f["charge"],
                            "mulliken_charge": None, "spin": None if s is None else round(s, 4) + 0.0,
                            "deviation": None, "flag": VALENCE_UNKNOWN_FLAG})
            continue
        q = sum(ref[i] - pops[i]["value"] for i in f["atoms"])
        dev = q - f["charge"]
        flag = None
        if f["ion"] and abs(dev) > FRAGMENT_DEVIATION_FLAG:
            flag = f"electron delocalised from {f['ion']} (Mulliken {q:+.2f} vs formal {f['charge']:+d})"
        elif not f["ion"] and abs(q) > FRAGMENT_DEVIATION_FLAG:
            flag = f"neutral fragment carries {q:+.2f} e"
        entries.append({"formula": f["formula"], "atoms": [i + 1 for i in f["atoms"]], "ion": f["ion"],
                        "extended": f["extended"], "formal_charge": f["charge"],
                        "mulliken_charge": round(q, 4) + 0.0, "spin": None if s is None else round(s, 4) + 0.0,
                        "deviation": round(dev, 4) + 0.0, "flag": flag})
    result["fragment_charges"] = entries
    return result


def format_fragment_charges(props: Dict[str, object]) -> List[str]:
    """The "Fragments" report section (empty without fragment_charges)."""
    entries = props.get("fragment_charges")
    if not entries:
        return []
    count = props.get("electron_count") or {}
    net = count.get("net_charge", 0)
    lines = ["Fragments" + (f" (net charge {'n/a' if net is None else format(net, '+g')}, "
                            f"{count.get('mulliken', 0):g} electrons)" if count else "")]
    for e in entries:
        name = e["ion"] or ("extended" if e["extended"] else "neutral")
        spin = "" if e["spin"] is None else f", spin {e['spin']:+.2f}"
        atoms = e["atoms"]
        where = f"atom {atoms[0]}" if len(atoms) == 1 else f"atoms {atoms[0]}..{atoms[-1]}" \
            if atoms == list(range(atoms[0], atoms[-1] + 1)) else f"{len(atoms)} atoms"
        lines.append(f"  {e['formula']:<12} {name:<10} formal {e['formal_charge']:+d}  "
                     + ("Mulliken n/a" if e["mulliken_charge"] is None
                        else f"Mulliken {e['mulliken_charge']:+.3f}") + f"{spin}  ({where})"
                     + (f"  <- {e['flag']}" if e["flag"] else ""))
    return lines
