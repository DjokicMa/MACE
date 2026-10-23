"""UFF / UFF4MOF atom typing, bond orders and the list of bonded terms.

Everything is built once from the starting geometry (a force field's
topology is fixed for a run):

1. Bonds: atom pairs closer than ``BOND_SCALE`` x the sum of ase's
   covalent radii (metal-metal pairs: ``METAL_METAL_SCALE``), across
   periodic images, each with its image shift.
2. Rings of 3 to 6 atoms (non-metals), and which 5- and 6-rings are
   aromatic (planar, every ring atom sp2-like).
3. Atom types. Main-group atoms by element, neighbors and rings (C_3, C_2,
   C_R, C_1, N_3, N_R, N_2, N_1, O_3, O_2, O_R, O_1, S_3+2, ...).
   Carboxylate carbon is C_R with C-O bond order 1.5 (resonance). Metals and
   every other element by coordination number and shape (tetrahedral or
   square planar at four, octahedral at six, ...). In UFF4MOF mode the MOF
   types come first: Zn3f2 (Zn4O), Cu4+2/Zn4+2 (paddlewheels), Zr8f4, the
   mu3/mu4 oxygens O_2_z / O_3_f, metal-oxygen bond order 0.5 and metal-metal
   0.25, as Addicoat et al. and lammps-interface type them.
4. Bond orders: aromatic 1.5, carboxylate 1.5, amide C-N 1.41, then double
   and triple bonds where the types need them, 1 otherwise.
5. Terms: bonds, angles (with the angle's Fourier order from the center's
   type), torsions about bonds between sp2/sp3 main-group atoms, inversions
   at sp2 C, N, O and at P/As/Sb/Bi, and the 1-2 and 1-3 pairs the van der
   Waals sum leaves out.

Every term stores its atoms with integer image shifts relative to its first
or central atom, so the energy code never wraps coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import uff_params as P

BOND_SCALE = 1.15
METAL_METAL_SCALE = 1.05
_SYMBOLS = None


def _symbols():
    global _SYMBOLS
    if _SYMBOLS is None:
        from ase.data import chemical_symbols
        _SYMBOLS = chemical_symbols
    return _SYMBOLS


_NONMETALS = {1, 2, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18, 32, 33, 34, 35, 36, 51, 52, 53, 54, 85, 86}
GROUP6 = {8, 16, 34, 52, 84}


def is_metal(z: int) -> bool:
    return int(z) not in _NONMETALS


Shift = Tuple[int, int, int]
ZERO: Shift = (0, 0, 0)


def _sub(a: Shift, b: Shift) -> Shift:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a: Shift, b: Shift) -> Shift:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


@dataclass
class Topology:
    mode: str                                  # "uff" or "uff4mof"
    types: List[str]
    neighbors: List[List[Tuple[int, Shift]]]   # per atom: (j, shift of j relative to i)
    bonds: List[Tuple[int, int, Shift, float]] # (i, j, shift of j, bond order)
    angles: List[Tuple[int, int, int, Shift, Shift]]     # (a, b center, c, shift a, shift c) rel. to b
    torsions: List[Tuple[int, int, int, int, Shift, Shift, Shift, float, int, float]]
    # (i, j, k, l, shift i, shift k, shift l) rel. to j, V (kcal/mol, divided), n, cos term
    inversions: List[Tuple[int, int, int, int, Shift, Shift, Shift, float, float, float, float]]
    # (i, j center, k, l, shift i, k, l rel. to j, K, C0, C1, C2)
    excluded: List[Tuple[int, int, Shift]]     # 1-2 and 1-3 pairs, i < j (or i == j with shift > 0)
    warnings: List[str] = field(default_factory=list)
    rings: List[List[int]] = field(default_factory=list)
    aromatic: List[bool] = field(default_factory=list)

    def counts(self) -> Dict[str, int]:
        return {"atoms": len(self.types), "bonds": len(self.bonds), "angles": len(self.angles),
                "torsions": len(self.torsions), "inversions": len(self.inversions)}

    def type_counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for t in self.types:
            out[t] = out.get(t, 0) + 1
        return dict(sorted(out.items()))


def table(mode: str) -> Dict[str, tuple]:
    return P.UFF4MOF_DATA if mode == "uff4mof" else P.UFF_DATA


# --------------------------------------------------------------------------- bonds

def find_bonds(atoms, scale: float = BOND_SCALE, metal_scale: float = METAL_METAL_SCALE):
    """Neighbors with image shifts: list per atom of (j, shift).

    UFF connectivity for atom typing (plain covalent radii, metal-metal pairs at
    their own scale), kept apart on purpose from the display rule that the
    viewer, Clean and mace analyze share (mace.gui.structures.bond_pairs): the
    UFF types and bonded terms are parametrized on this connectivity."""
    from ase.data import covalent_radii
    from ase.neighborlist import neighbor_list
    z = atoms.numbers
    rad = covalent_radii[z]
    n = len(atoms)
    nbrs: List[List[Tuple[int, Shift]]] = [[] for _ in range(n)]
    if n < 2 and not any(atoms.pbc):
        return nbrs
    ii, jj, SS, dd = neighbor_list("ijSd", atoms, list(scale * rad), self_interaction=False)
    metal = np.array([is_metal(x) for x in z])
    for i, j, s, d in zip(ii, jj, SS, dd):
        if metal[i] and metal[j] and d > metal_scale * (rad[i] + rad[j]):
            continue
        if z[i] == 1 and z[j] == 1 and d > 0.9:          # H...H contacts are not bonds (H2 is 0.74 Å)
            continue
        nbrs[int(i)].append((int(j), (int(s[0]), int(s[1]), int(s[2]))))
    return nbrs


# --------------------------------------------------------------------------- rings

def _ring_key(path) -> tuple:
    """One key for a ring however it was found: its (atom, shift) set, translated so
    that each image of its lowest atom in turn is at no shift; the least of those."""
    s = min(x for x, _ in path)
    return min(tuple(sorted((x, _sub(sx, t)) for x, sx in path)) for a, t in path if a == s)


def find_rings(nbrs, z, max_size: int = 6) -> List[List[Tuple[int, Shift]]]:
    """Rings of 3..max_size distinct non-metal atom images (each once), as
    (atom, shift relative to the ring's first atom) lists. An atom may come back
    in another image (graphene's primitive cell has two atoms and 6-rings)."""
    rings, seen = [], set()
    n = len(nbrs)
    ok = [not is_metal(int(zz)) and int(zz) != 1 for zz in z]
    for s in range(n):
        if not ok[s] or len(nbrs[s]) < 2:
            continue
        stack = [(s, ZERO, [(s, ZERO)])]
        while stack:
            a, sa, path = stack.pop()
            for b, sab in nbrs[a]:
                if not ok[b]:
                    continue
                sb = _add(sa, sab)
                if b == s and sb == ZERO:
                    if len(path) >= 3:
                        key = _ring_key(path)
                        if key not in seen:
                            seen.add(key)
                            rings.append(list(path))
                    continue
                if b < s or len(path) >= max_size or (b, sb) in path:
                    continue
                stack.append((b, sb, path + [(b, sb)]))
    return rings


def _planar(points: np.ndarray, tol: float = 0.12) -> bool:
    c = points - points.mean(axis=0)
    _, sv, vt = np.linalg.svd(c)
    return float(np.abs(c @ vt[-1]).max()) < tol


def aromatic_rings(atoms, nbrs, rings) -> List[bool]:
    z = atoms.numbers
    pos, cell = atoms.positions, atoms.cell.array
    out = []
    for ring in rings:
        size = len(ring)
        if size not in (5, 6):
            out.append(False)
            continue
        good = True
        for a, _ in ring:
            deg, e = len(nbrs[a]), int(z[a])
            if e == 6 and deg != 3 and not (size == 5 and deg == 2):
                good = False
            elif e == 7 and deg not in (2, 3):
                good = False
            elif e in (8, 16, 34) and (deg != 2 or size != 5):
                good = False
            elif e == 5 and deg != 3:
                good = False
            elif e not in (5, 6, 7, 8, 16, 34):
                good = False
        if good:
            pts = np.array([pos[a] + np.array(s) @ cell for a, s in ring])
            good = _planar(pts)
        if good and size == 5:
            # a 5-ring needs a donor (N-H, O, S, or a carbanion C): cyclopentane-like rings are never planar anyway
            good = any(int(z[a]) in (7, 8, 16, 34) for a, _ in ring) or all(len(nbrs[a]) == 3 for a, _ in ring)
        out.append(bool(good))
    return out


# --------------------------------------------------------------------------- geometry helpers

def _vec(atoms, i: int, j: int, s: Shift) -> np.ndarray:
    return atoms.positions[j] + np.array(s, float) @ atoms.cell.array - atoms.positions[i]


def _angles_at(atoms, i, nb) -> List[float]:
    vs = [_vec(atoms, i, j, s) for j, s in nb]
    out = []
    for a in range(len(vs)):
        for b in range(a + 1, len(vs)):
            c = vs[a] @ vs[b] / (np.linalg.norm(vs[a]) * np.linalg.norm(vs[b]) + 1e-12)
            out.append(float(np.degrees(np.arccos(np.clip(c, -1, 1)))))
    return out


def _square_planar(atoms, i, nb) -> bool:
    return len(nb) == 4 and sum(a > 150.0 for a in _angles_at(atoms, i, nb)) >= 2


# --------------------------------------------------------------------------- typing

def _el(z: int) -> str:
    return _symbols()[int(z)]


def _prefix(t: str) -> str:
    return t[:2].rstrip("_")


def _digit(t: str) -> str:
    return t[2] if len(t) > 2 else ""


def _generic_type(el: str, cn: int, planar4: bool, mode: str, warn: List[str], idx: int) -> str:
    tab = table(mode)
    base = el + "_" if len(el) == 1 else el
    if mode == "uff4mof":
        suffix = {2: "1f1", 3: "2f2", 4: "4f2" if planar4 else "3f2", 5: "4+2", 6: "6f3", 8: "8f4"}.get(cn)
        if suffix:
            for t in (base + suffix, base + suffix.replace("f", "+")):
                if t in tab:
                    return t
            if cn == 5 and base + "4f2" in tab:
                return base + "4f2"
    cands = [t for t in tab if _prefix(t) == el]
    if not cands:
        raise KeyError(el)
    want = {0: "1", 1: "1", 2: "1", 3: "2", 4: "4" if planar4 else "3", 5: "5", 6: "6", 7: "8", 8: "8"}.get(cn, "6")
    exact = [t for t in cands if _digit(t) == want]
    if exact:
        return exact[0]
    if want == "2":
        r = [t for t in cands if _digit(t) == "R"]
        if r:
            return r[0]
    order = {"1": 1, "2": 2, "R": 2, "3": 3, "4": 4, "5": 5, "6": 6, "8": 8}
    target = order.get(want, 6)
    best = min(cands, key=lambda t: (abs(order.get(_digit(t), 9) - target), cands.index(t)))
    if len(cands) > 1 or _digit(best) not in ("", want):
        warn.append(f"atom {idx + 1} ({el}, {cn} neighbors): no {el} type for that coordination; using {best}")
    return best


def assign_types(atoms, nbrs, rings, arom, mode: str = "uff4mof") -> Tuple[List[str], List[str], set]:
    """(types, warnings, carboxylate carbons)."""
    z = [int(x) for x in atoms.numbers]
    n = len(z)
    tab = table(mode)
    in_arom = [False] * n
    for ring, a in zip(rings, arom):
        if a:
            for i, _ in ring:
                in_arom[i] = True
    warn: List[str] = []
    types: List[Optional[str]] = [None] * n
    metal = [is_metal(x) for x in z]

    def nz(i):
        return [z[j] for j, _ in nbrs[i]]

    # carboxylate carbons (UFF4MOF's resonance convention, C_R with two C-O bonds of order 1.5): sp2 C with
    # two O neighbors, each bonded only to carbon or metals; in plain UFF only when an O is bonded to a metal
    # (an isolated carboxylate is typed like any conjugated group, as RDKit does)
    carbox = set()
    for i in range(n):
        if z[i] == 6 and len(nbrs[i]) == 3:
            os_ = [j for j, _ in nbrs[i] if z[j] == 8]
            if len(os_) == 2 and all(all(z[k] == 6 or metal[k] for k, _ in nbrs[o]) for o in os_) and \
                    (mode == "uff4mof" or any(metal[k] for o in os_ for k, _ in nbrs[o])):
                carbox.add(i)
    for i in range(n):
        e, deg, el = z[i], len(nbrs[i]), _el(z[i])
        nb = nbrs[i]
        try:
            if e == 1:
                types[i] = "H_b" if deg == 2 and all(z[j] == 5 for j, _ in nb) else "H_"
            elif e == 6:
                if in_arom[i] or i in carbox:
                    types[i] = "C_R"
                elif deg >= 4:
                    types[i] = "C_3"
                elif deg == 3:
                    types[i] = "C_2"
                elif deg == 2:
                    types[i] = "C_1" if max(_angles_at(atoms, i, nb)) > 160.0 else "C_2"
                else:
                    types[i] = "C_1"
            elif e == 7:
                if in_arom[i]:
                    types[i] = "N_R"
                elif deg >= 4:
                    types[i] = "N_3"
                elif deg == 3:
                    types[i] = "N_3"            # N_R when conjugated (conjugate(), after the bond orders)
                elif deg == 2:
                    types[i] = "N_1" if max(_angles_at(atoms, i, nb)) > 160.0 else "N_2"
                else:
                    types[i] = "N_1"
            elif e == 8:
                ns = set(nz(i))
                nmet = sum(metal[j] for j, _ in nb)
                if in_arom[i]:
                    types[i] = "O_R"
                elif deg == 1:
                    j = nb[0][0]
                    if z[j] == 1:
                        types[i] = "O_3"
                    elif z[j] == 6 and len(nbrs[j]) == 1:
                        types[i] = "O_1"                    # carbon monoxide
                    else:
                        types[i] = "O_2"
                elif ns <= {14, 13} and deg == 2:
                    types[i] = "O_3_z"
                elif any(j in carbox for j, _ in nb):
                    types[i] = "O_2"
                elif mode == "uff4mof" and nmet:
                    if deg == 2:
                        types[i] = "O_2"
                    elif deg == 3:
                        allm = nmet == 3
                        types[i] = ("O_3_f" if ns == {40} else "O_2_z") if allm else "O_2"
                    else:
                        types[i] = "O_3_f"
                else:
                    types[i] = "O_3"
            elif e == 16:
                if in_arom[i]:
                    types[i] = "S_R"
                elif deg <= 1:
                    types[i] = "S_2"
                elif deg == 2:
                    types[i] = "S_3+2"
                elif deg == 3:
                    types[i] = "S_3+4"
                else:
                    types[i] = "S_3+6"
            elif e == 15:
                types[i] = "P_3+3" if deg <= 3 else "P_3+5"
            elif e == 5:
                types[i] = "B_3" if deg >= 4 else "B_2"
            elif e in (9, 53):
                types[i] = el + "_"
            elif e in (17, 35):
                types[i] = el
            else:
                types[i] = _generic_type(el, deg, _square_planar(atoms, i, nb), mode, warn, i)
            if types[i] not in tab:
                types[i] = _generic_type(el, deg, _square_planar(atoms, i, nb), mode, warn, i)
        except KeyError:
            raise ValueError(f"atom {i + 1}: UFF has no parameters for {el}")
    return types, warn, carbox


# --------------------------------------------------------------------------- bond orders

def bond_orders(atoms, nbrs, types, rings, arom, carbox, mode) -> Dict[Tuple[int, int, Shift], float]:
    """Bond order per bond key (i, j, shift) with i < j, or i == j and shift > 0."""
    z = [int(x) for x in atoms.numbers]
    metal = [is_metal(x) for x in z]
    n = len(z)
    order: Dict[Tuple[int, int, Shift], float] = {}

    key = _pair_key

    for i in range(n):
        for j, s in nbrs[i]:
            order[key(i, j, s)] = 1.0
    arom_bonds = set()
    for ring, a in zip(rings, arom):
        if not a:
            continue
        for k in range(len(ring)):
            (i, si), (j, sj) = ring[k], ring[(k + 1) % len(ring)]
            arom_bonds.add(key(i, j, _sub(sj, si)))
    for k in arom_bonds:
        order[k] = 1.5
    for (i, j, s) in list(order):
        ti, tj = types[i], types[j]
        if (i in carbox and z[j] == 8) or (j in carbox and z[i] == 8):
            order[(i, j, s)] = 1.5
        elif metal[i] or metal[j]:
            if mode == "uff4mof":
                order[(i, j, s)] = 0.25 if (metal[i] and metal[j]) else 0.5
    # double and triple bonds where the types need them
    need = [0] * n
    for i in range(n):
        t = types[i]
        if any(order[key(i, j, s)] != 1.0 for j, s in nbrs[i]):
            continue                                        # aromatic, carboxylate, metal-bonded: settled
        if t in ("C_2", "N_2", "O_2", "S_2", "B_2"):
            need[i] = 1
        elif t in ("C_1", "N_1", "O_1"):
            need[i] = 2
    # triples first (nitriles, alkynes, CO), then terminal O/S doubles, then the rest
    for i in range(n):
        if need[i] == 2:
            for j, s in nbrs[i]:
                if need[j] == 2 and (len(nbrs[i]) <= 2 and len(nbrs[j]) <= 2):
                    if types[i] == "C_1" and types[j] == "C_1" and (len(nbrs[i]) == 2 or len(nbrs[j]) == 2) or \
                            {types[i], types[j]} <= {"C_1", "N_1", "O_1"}:
                        order[key(i, j, s)] = 3.0
                        need[i] = need[j] = 0
                        break
    for i in range(n):
        if need[i] == 2:          # cumulated (allene, CO2): two doubles
            for j, s in nbrs[i]:
                if z[j] in (6, 7, 8, 16) and order[key(i, j, s)] == 1.0:
                    order[key(i, j, s)] = 2.0
                    need[j] = max(0, need[j] - 1)
            need[i] = 0
    terminal = [i for i in range(n) if need[i] == 1 and len(nbrs[i]) == 1]
    for i in terminal:
        j, s = nbrs[i][0]
        if z[j] in (6, 7, 15, 16) and not metal[j] and (need[j] >= 1 or types[j] in ("S_3+4", "S_3+6", "P_3+5")):
            order[key(i, j, s)] = 2.0
            need[i] = 0
            if need[j] >= 1:
                need[j] -= 1
    for i in range(n):
        if need[i] != 1:
            continue
        cands = [(j, s) for j, s in nbrs[i] if need[j] == 1]
        if cands:
            j, s = min(cands, key=lambda c: sum(need[k] == 1 for k, _ in nbrs[c[0]]))
            order[key(i, j, s)] = 2.0
            need[i] = need[j] = 0
    return order


# --------------------------------------------------------------------------- conjugation

def conjugate(nbrs, z, types, orders, arom_atoms) -> List[str]:
    """RDKit's resonance types: an sp2 C/N/O atom on a conjugated bond, and
    an N or O lone pair next to one (amide and aniline N, acid and phenol O),
    gets the resonance type (C_R, N_R, O_R).

    A single bond is conjugated when both its atoms are unsaturated (a bond
    of order 1.5 or more) or one is unsaturated and the other a lone-pair
    donor (N with up to 3, O with up to 2 neighbors, neither unsaturated);
    a multiple bond when it is aromatic, touches a conjugated single bond, or
    shares an sp atom with another multiple bond (CO2). Bonds to metals never
    count. The bond orders stay as assigned (1, 1.5, 2, 3)."""
    n = len(z)
    metal = [is_metal(x) for x in z]
    key = _pair_key

    def bo(i, j, s):
        return orders[key(i, j, s)]
    unsat = [any(bo(i, j, s) >= 1.5 and not metal[j] for j, s in nbrs[i]) for i in range(n)]
    donor = [(z[i] == 7 and len(nbrs[i]) <= 3 or z[i] == 8 and len(nbrs[i]) <= 2) and not unsat[i] and not metal[i]
             for i in range(n)]
    conj_single = set()
    for i in range(n):
        if metal[i]:
            continue
        for j, s in nbrs[i]:
            if metal[j] or bo(i, j, s) != 1.0:
                continue
            if (unsat[i] or donor[i]) and (unsat[j] or donor[j]) and not (donor[i] and donor[j]):
                conj_single.add(key(i, j, s))
    conj_atom = [False] * n
    for k in conj_single:
        conj_atom[k[0]] = conj_atom[k[1]] = True
    for i in range(n):
        if metal[i]:
            continue
        for j, s in nbrs[i]:
            if metal[j] or bo(i, j, s) < 1.5:
                continue
            on = (arom_atoms[i] and arom_atoms[j]) or any(
                key(a, b, t) in conj_single for a in (i, j) for b, t in nbrs[a])
            if not on:
                for a in (i, j):
                    if _digit(types[a]) == "1" and sum(bo(a, b, t) >= 1.5 for b, t in nbrs[a]) >= 2:
                        on = True
            if on:
                conj_atom[i] = conj_atom[j] = True
    out = list(types)
    for i in range(n):
        if not conj_atom[i] or z[i] not in (6, 7, 8):
            continue
        t = out[i]
        if t in ("C_2", "N_2", "O_2") or (donor[i] and t in ("N_3", "O_3")):
            out[i] = t[:2] + "R"
    return out


# --------------------------------------------------------------------------- terms

def _hyb(t: str, z: int) -> str:
    if is_metal(z):
        return "metal"
    d = _digit(t)
    return {"3": "sp3", "2": "sp2", "R": "sp2", "1": "sp"}.get(d, "other")


def _inversion_params(zc: int, t: str, nb_types: List[str], nb_deg: List[int], mode: str = "uff"):
    """(K, C0, C1, C2) for one of the three inversion terms at a center, or None.
    An sp2 carbon bound to an sp2 oxygen gets K = 50 (RDKit's rule; UFF4MOF,
    as lammps-interface, only for a terminal carbonyl O, not a metal-bound
    carboxylate O)."""
    if zc in (6, 7, 8):
        if _digit(t) not in ("2", "R"):
            return None
        k = 6.0
        if zc == 6 and any(tt[:2] == "O_" and _digit(tt) in ("2", "R") and (mode == "uff" or d == 1)
                           for tt, d in zip(nb_types, nb_deg)):
            k = 50.0
        return k / 3.0, 1.0, -1.0, 0.0
    w0 = {15: 84.4339, 33: 86.9735, 51: 87.7047, 83: 90.0}.get(zc)
    if w0 is None:
        return None
    w = np.radians(w0)
    c2 = 1.0
    c1 = -4.0 * np.cos(w)
    c0 = -c1 * np.cos(w) + c2 * np.cos(2.0 * w)
    return 22.0 / 3.0, float(c0), float(c1), float(c2)


def build(atoms, mode: str = "uff4mof") -> Topology:
    if mode not in ("uff", "uff4mof"):
        raise ValueError("mode is uff or uff4mof")
    z = [int(x) for x in atoms.numbers]
    n = len(z)
    nbrs = find_bonds(atoms)
    rings = find_rings(nbrs, z)
    arom = aromatic_rings(atoms, nbrs, rings)
    types, warn, carbox = assign_types(atoms, nbrs, rings, arom, mode)
    orders = bond_orders(atoms, nbrs, types, rings, arom, carbox, mode)
    arom_atoms = [False] * n
    for ring, a_ in zip(rings, arom):
        if a_:
            for i, _ in ring:
                arom_atoms[i] = True
    tab = table(mode)
    keep = set(carbox) | {j for c in carbox for j, _ in nbrs[c] if z[j] == 8}   # carboxylates stay C_R / O_2
    types = [t2 if (t2 in tab and i not in keep) else t1
             for i, (t1, t2) in enumerate(zip(types, conjugate(nbrs, z, types, orders, arom_atoms)))]

    def bo(i, j, s):
        if i < j or (i == j and s > ZERO):
            return orders[(i, j, s)]
        return orders[(j, i, (-s[0], -s[1], -s[2]))]

    bonds = [(i, j, s, o) for (i, j, s), o in orders.items()]
    angles = []
    for b in range(n):
        nb = nbrs[b]
        for x in range(len(nb)):
            for y in range(x + 1, len(nb)):
                (a, sa), (c, sc) = nb[x], nb[y]
                angles.append((a, b, c, sa, sc))
    torsions = []
    for (j, k, sjk, o) in bonds:
        hj, hk = _hyb(types[j], z[j]), _hyb(types[k], z[k])
        if hj not in ("sp2", "sp3") or hk not in ("sp2", "sp3"):
            continue
        here = []
        for i, sji in nbrs[j]:
            if i == k and sji == sjk:
                continue
            for l, skl in nbrs[k]:
                sl = _add(sjk, skl)
                if l == j and sl == ZERO:
                    continue
                if l == i and sl == sji:
                    continue                                  # a three-membered ring
                here.append((i, l, sji, sl))
        if not here:
            continue
        tj, tk = P.UFF4MOF_DATA[types[j]] if mode == "uff4mof" else P.UFF_DATA[types[j]], \
            P.UFF4MOF_DATA[types[k]] if mode == "uff4mof" else P.UFF_DATA[types[k]]
        for i, l, sji, sl in here:
            end_sp2 = _hyb(types[i], z[i]) == "sp2" or _hyb(types[l], z[l]) == "sp2"
            V, nn, cterm = _torsion_params(o, z[j], z[k], hj, hk, tj, tk, end_sp2)
            torsions.append((i, j, k, l, sji, sjk, sl, V / len(here), nn, cterm))
    inversions = []
    for j in range(n):
        if len(nbrs[j]) != 3:
            continue
        nb = nbrs[j]
        p = _inversion_params(z[j], types[j], [types[a] for a, _ in nb], [len(nbrs[a]) for a, _ in nb], mode)
        if p is None:
            continue
        (a, sa), (b, sb), (c, sc) = nb
        for (i, si), (k, sk), (l, sl) in (((a, sa), (b, sb), (c, sc)), ((a, sa), (c, sc), (b, sb)),
                                          ((b, sb), (c, sc), (a, sa))):
            inversions.append((i, j, k, l, si, sk, sl) + p)
    excl = set()
    for i in range(n):
        for j, s in nbrs[i]:
            excl.add(_pair_key(i, j, s))
    for (a, b, c, sa, sc) in angles:
        if a == c and sa == sc:
            continue
        excl.add(_pair_key(a, c, _sub(sc, sa)))
    return Topology(mode=mode, types=list(types), neighbors=nbrs, bonds=bonds, angles=angles, torsions=torsions,
                    inversions=inversions, excluded=sorted(excl), warnings=warn,
                    rings=[[a for a, _ in r] for r in rings], aromatic=arom)


def _pair_key(i, j, s):
    if i < j or (i == j and s > ZERO):
        return (i, j, s)
    return (j, i, (-s[0], -s[1], -s[2]))


def _torsion_params(order, zj, zk, hj, hk, pj, pk, end_sp2):
    """(V kcal/mol, n, cos term) as RDKit's UFF: E = V/2 (1 - cterm cos(n phi))."""
    vi, vj = pj[6], pk[6]
    ui, uj = pj[7], pk[7]

    def eq17():
        return 5.0 * np.sqrt(ui * uj) * (1.0 + 4.18 * np.log(order))
    if hj == "sp3" and hk == "sp3":
        V, n, c = np.sqrt(vi * vj), 3, -1.0
        if order == 1.0 and zj in GROUP6 and zk in GROUP6:
            v2 = 2.0 if zj == 8 else 6.8
            v3 = 2.0 if zk == 8 else 6.8
            V, n, c = np.sqrt(v2 * v3), 2, -1.0
        return float(V), n, c
    if hj == "sp2" and hk == "sp2":
        return float(eq17()), 2, 1.0
    V, n, c = 1.0, 6, 1.0
    if order == 1.0:
        if (hj == "sp3" and zj in GROUP6 and zk not in GROUP6) or (hk == "sp3" and zk in GROUP6 and zj not in GROUP6):
            return float(eq17()), 2, -1.0
        if end_sp2:
            return 2.0, 3, -1.0
    return V, n, c
