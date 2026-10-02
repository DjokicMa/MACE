"""Geometry of a CRYSTAL output, defect-site resolution and PBC atom matching.

Atom labels in a supercell are CRYSTAL's own numbering, so they are read from
a CRYSTAL output (the pristine supercell run) or given by the user; they are
never guessed. The tables are printed right after the geometry is set up, so a
still-running pristine output can be read as soon as they appear.

Tables read (last occurrence of each):

    DIRECT LATTICE VECTORS CARTESIAN COMPONENTS (ANGSTROM)
    CARTESIAN COORDINATES - PRIMITIVE CELL
    ATOMS IN THE ASYMMETRIC UNIT ... - ATOMS IN THE UNIT CELL: N
        (label, T/F, Z, symbol, X/A Y/B Z/C; for slabs Z is in Angstrom)
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from mace.defects import DefectError
from mace.defects.deck_edit import HostInfo, element_number, element_symbol

#: Distance within which a fractional site must match an atom (Angstrom).
SITE_MATCH_TOL = 0.3
#: Closest an inserted atom may come to an existing one (Angstrom).
MIN_INSERT_DISTANCE = 0.8
#: Layer groups with a centred (c) rectangular lattice.
_CENTRED_LAYER_GROUPS = {10, 13, 18, 22, 26, 35, 36, 47, 48}

_LATTICE = "DIRECT LATTICE VECTORS CARTESIAN COMPONENTS (ANGSTROM)"
_CART = "CARTESIAN COORDINATES - PRIMITIVE CELL"
_ASYM = re.compile(r"ATOMS IN THE ASYMMETRIC UNIT\s+(\d+)\s+-\s+ATOMS IN THE UNIT CELL:\s+(\d+)")
_CART_ROW = re.compile(r"^\s*(\d+)\s+(\d+)\s+([A-Za-z]{1,2})\s+(\S+)\s+(\S+)\s+(\S+)\s*$")
_ASYM_ROW = re.compile(r"^\s*(\d+)\s+([TF])\s+(\d+)\s+([A-Za-z]{1,2})\s+(\S+)\s+(\S+)\s+(\S+)\s*$")


@dataclass
class Atom:
    label: int
    z: int
    symbol: str
    cart: np.ndarray
    frac: Optional[np.ndarray] = None
    asym: Optional[bool] = None

    @property
    def element(self) -> str:
        return "XX" if self.symbol.upper() == "XX" else self.symbol.capitalize()


@dataclass
class Geometry:
    lattice: np.ndarray          # rows a, b, c in Angstrom
    atoms: List[Atom]
    dimensionality: int
    n_electrons_header: Optional[int] = None

    def atom(self, label: int) -> Atom:
        for a in self.atoms:
            if a.label == label:
                return a
        raise DefectError(f"atom {label} is not in the output (it lists 1..{len(self.atoms)})")

    def by_element(self, element: str) -> List[Atom]:
        return [a for a in self.atoms if a.element.lower() == element.lower()]


def _last(lines: List[str], marker: str) -> Optional[int]:
    idx = None
    for i, line in enumerate(lines):
        if marker in line:
            idx = i
    return idx


def parse_geometry(text: str) -> Optional[Geometry]:
    """The supercell geometry of a CRYSTAL output, or None if the tables are not (all) there yet."""
    lines = text.splitlines()
    i_lat = _last(lines, _LATTICE)
    i_cart = _last(lines, _CART)
    asym_idx = None
    for i, line in enumerate(lines):
        if _ASYM.search(line):
            asym_idx = i
    if i_lat is None or i_cart is None:
        return None
    try:
        lat = np.array([[float(v) for v in lines[i_lat + 2 + k].split()[:3]] for k in range(3)])
    except (ValueError, IndexError):
        return None
    atoms: List[Atom] = []
    for line in lines[i_cart + 4:]:
        m = _CART_ROW.match(line)
        if not m:
            break
        atoms.append(Atom(int(m.group(1)), int(m.group(2)), m.group(3),
                          np.array([float(m.group(k)) for k in (4, 5, 6)])))
    if not atoms:
        return None
    if asym_idx is not None:
        n_cell = int(_ASYM.search(lines[asym_idx]).group(2))
        if n_cell != len(atoms):
            return None
        frac = {}
        for line in lines[asym_idx + 3:asym_idx + 3 + n_cell]:
            m = _ASYM_ROW.match(line)
            if not m:
                break
            frac[int(m.group(1))] = (m.group(2) == "T", np.array([float(m.group(k)) for k in (5, 6, 7)]))
        for a in atoms:
            if a.label in frac:
                a.asym, a.frac = frac[a.label]
    dim = 3
    m = None
    for m in re.finditer(r"DIMENSIONALITY OF THE SYSTEM\s+(\d)", text):
        pass
    if m:
        dim = int(m.group(1))
    elif abs(lat[2][2] - 500.0) < 1e-6:
        dim = 2
    n = None
    for n in re.finditer(r"N\. OF ELECTRONS PER CELL\s+(\d+)", text):
        pass
    return Geometry(lat, atoms, dim, int(n.group(1)) if n else None)


# ---------------------------------------------------------------------------
# Periodic distances
# ---------------------------------------------------------------------------

def min_image(vec: np.ndarray, lattice: np.ndarray, dim: int) -> np.ndarray:
    """Shortest periodic image of a Cartesian vector (only a, b are periodic in 2D)."""
    vec = np.asarray(vec, float)
    if dim == 2:
        lat2 = lattice[:2, :2]
        f = np.linalg.solve(lat2.T, vec[:2])
        f -= np.round(f)
        base = np.array([*(f @ lat2), vec[2]])
        best = base
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                cand = base + i * np.r_[lattice[0][:2], 0] + j * np.r_[lattice[1][:2], 0]
                if np.linalg.norm(cand) < np.linalg.norm(best):
                    best = cand
        return best
    f = np.linalg.solve(lattice.T, vec)
    f -= np.round(f)
    base = f @ lattice
    best = base
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            for k in (-1, 0, 1):
                cand = base + i * lattice[0] + j * lattice[1] + k * lattice[2]
                if np.linalg.norm(cand) < np.linalg.norm(best):
                    best = cand
    return best


def pbc_distance(geom: Geometry, a: Sequence[float], b: Sequence[float]) -> float:
    return float(np.linalg.norm(min_image(np.asarray(b) - np.asarray(a), geom.lattice, geom.dimensionality)))


def neighbours(geom: Geometry, center: Sequence[float], cutoff: float,
               exclude: Sequence[int] = ()) -> List[Tuple[int, float]]:
    out = []
    for a in geom.atoms:
        if a.label in exclude:
            continue
        d = pbc_distance(geom, center, a.cart)
        if d <= cutoff:
            out.append((a.label, d))
    return sorted(out, key=lambda t: (t[1], t[0]))


def nearest_distance(geom: Geometry, center: Sequence[float], exclude: Sequence[int] = ()) -> Tuple[int, float]:
    best = (0, float("inf"))
    for a in geom.atoms:
        if a.label in exclude:
            continue
        d = pbc_distance(geom, center, a.cart)
        if d < best[1]:
            best = (a.label, d)
    return best


def match_atoms(geom_a: Geometry, geom_b: Geometry, tol: float = SITE_MATCH_TOL) -> Dict[int, int]:
    """Labels of geom_a mapped to the atom of geom_b at the same position (PBC of geom_b).

    Used after ATOMREMO renumbering: a defect cell's label k is not the
    pristine label k once an atom has been removed.
    """
    out = {}
    for a in geom_a.atoms:
        best = None
        for b in geom_b.atoms:
            d = pbc_distance(geom_b, a.cart, b.cart)
            if d <= tol and (best is None or d < best[1]):
                best = (b.label, d)
        if best:
            out[a.label] = best[0]
    return out


def far_atom(geom: Geometry, from_label: int, element: str) -> int:
    """The atom of ``element`` farthest from atom ``from_label`` (PBC); ties -> lowest label."""
    center = geom.atom(from_label).cart
    best = None
    for a in geom.by_element(element):
        if a.label == from_label:
            continue
        d = pbc_distance(geom, center, a.cart)
        if best is None or d > best[1] + 1e-6:
            best = (a.label, d)
    if best is None:
        raise DefectError(f"no other {element} atom in the supercell for the far-atom recipe")
    return best[0]


# ---------------------------------------------------------------------------
# Sites
# ---------------------------------------------------------------------------

def _inverse_int(matrix) -> np.ndarray:
    return np.linalg.inv(np.array(matrix, float))


def host_cell(host: HostInfo, keyword: str, matrix, geom: Geometry) -> np.ndarray:
    """The host deck's cell in the frame of the supercell output (rows, Angstrom).

    SUPERCON expands the crystallographic cell and SUPERCEL the primitive one
    (p.75), so the host's cell is M^-1 times the supercell vectors. A
    fractional site is given in the host deck's cell, which for SUPERCEL is
    only the expanded cell when the host lattice is primitive.
    """
    D = geom.lattice
    if keyword.upper() == "SUPERCEL":
        centred = False
        if host.geometry_kind == "CRYSTAL":
            from mace.defects.deck_edit import centring_multiplicity
            centred = centring_multiplicity(host) != 1
        else:
            try:
                centred = int(host.spacegroup.split()[0]) in _CENTRED_LAYER_GROUPS
            except ValueError:
                centred = False
        if centred:
            raise DefectError("--site-frac / fractional --insert with SUPERCEL on a centred host: "
                              "SUPERCEL expands the primitive cell, not the host's crystallographic "
                              "cell; use SUPERCON (3D), --site-label, or --cart")
    if host.dimensionality == 2:
        inv = _inverse_int(matrix)
        ab = inv @ D[:2]
        return np.vstack([ab, D[2]])
    return _inverse_int(matrix) @ D


def frac_to_cart(host: HostInfo, cell: np.ndarray, frac: Sequence[float]) -> np.ndarray:
    f = np.asarray(frac, float)
    if host.dimensionality == 2:
        # deck coordinates of a slab: fractional x, y and z in Angstrom
        return f[0] * cell[0] + f[1] * cell[1] + np.array([0.0, 0.0, f[2]])
    return f @ cell


def host_sites(host: HostInfo, element: str) -> List[str]:
    z = element_number(element)
    return [f"host atom {i + 1}: {element_symbol(zc)} {x:g} {y:g} {zz:g}"
            for i, (zc, x, y, zz) in enumerate(host.atoms) if zc % 100 == z]


def resolve_site(host: HostInfo, geom: Optional[Geometry], element: str, *, label: Optional[int] = None,
                 frac: Optional[Sequence[float]] = None, keyword: str = "SUPERCEL",
                 matrix=None) -> Tuple[Optional[int], str, List[str]]:
    """The supercell label of the defect site; (None, reason, notes) when it needs the pristine output."""
    notes: List[str] = []
    z = element_number(element)
    if not any(zc % 100 == z for zc, *_ in host.atoms):
        raise DefectError(f"{element} is not in the host")
    if label is not None:
        if geom is None:
            notes.append(f"WARNING: site label {label} was not checked against a CRYSTAL output "
                         f"(it must be a {element} in the supercell's numbering)")
            return label, "label (unchecked)", notes
        atom = geom.atom(label)
        if atom.element.lower() != element.lower():
            raise DefectError(f"atom {label} of the pristine supercell is {atom.element}, not {element}")
        return label, "label", notes
    if frac is not None:
        if geom is None:
            return None, "needs the pristine output (fractional site)", notes
        cell = host_cell(host, keyword, matrix, geom)
        cart = frac_to_cart(host, cell, frac)
        best = None
        for a in geom.by_element(element):
            d = pbc_distance(geom, cart, a.cart)
            if best is None or d < best[1]:
                best = (a.label, d)
        if best is None or best[1] > SITE_MATCH_TOL:
            raise DefectError(f"no {element} within {SITE_MATCH_TOL} A of fractional site "
                              f"{tuple(frac)} (Cartesian {np.round(cart, 4).tolist()}); nearest "
                              f"{element}: {best}")
        notes.append(f"site {tuple(frac)} of the host cell -> supercell atom {best[0]} "
                     f"({best[1]:.3f} A away)")
        return best[0], "fractional", notes
    sites = host_sites(host, element)
    if len(sites) > 1:
        raise DefectError(f"{element} occupies {len(sites)} inequivalent sites of the host; choose one "
                          "with --site-frac x,y,z (host cell) or --site-label N:\n  " + "\n  ".join(sites))
    if geom is None:
        return None, "needs the pristine output (element site)", notes
    first = min(a.label for a in geom.by_element(element))
    notes.append(f"{element} has one site in the host, so every image is equivalent: using "
                 f"supercell atom {first}")
    return first, "element", notes


def insertion_point(host: HostInfo, geom: Geometry, coords: Sequence[float], cart: bool,
                    keyword: str, matrix) -> np.ndarray:
    """Cartesian (supercell frame) position of an inserted atom, refused if too close to an atom."""
    if cart:
        point = np.asarray(coords, float)
    else:
        point = frac_to_cart(host, host_cell(host, keyword, matrix, geom), coords)
    label, d = nearest_distance(geom, point)
    if d < MIN_INSERT_DISTANCE:
        raise DefectError(f"the inserted atom would be {d:.2f} A from atom {label}: closer than "
                          f"{MIN_INSERT_DISTANCE} A to an existing atom")
    return point
