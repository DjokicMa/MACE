"""Lattice-dependent SeeK-path (HPKOT) k-points computed from the cell.

Without the seekpath library MACE writes the static paths of d3_kpoints'
``seekpath_data``, whose lattice-dependent points were evaluated once, for
some other cell. For the lattices below this module evaluates them for the
actual cell, from the conventional cell parameters CRYSTAL prints, and gives
them in the reciprocal basis of CRYSTAL's primitive cell, as the library
route does (seekpath_interface.crystal_cell_matrix / to_crystal_basis). It
needs numpy only.

The k-vector parameters, point coordinates and paths are transcribed from
SeeK-path 2.2 (hpkot/band_path_data/<lattice>/k_vector_parameters.txt,
points.txt and path.txt; MIT License, Copyright (c) 2016 Giovanni Pizzi,
Yoyo Hinuma and contributors, https://github.com/giovannipizzi/seekpath),
which implements Y. Hinuma, G. Pizzi, Y. Kumagai, F. Oba and I. Tanaka,
"Band structure diagram paths based on crystallography", Comp. Mat. Sci.
128, 140 (2017). The lattice-variant rules are SeeK-path's
(hpkot.get_path). The tests compare every entry with SeeK-path's files and
results.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np


class NotInHpkotSetting(ValueError):
    """The cell is not in the setting HPKOT's formulas are written for."""


# Each entry: the k-vector parameters in evaluation order (expressions in a,
# b, c, cosbeta, sinbeta and earlier parameters), the points as SeeK-path's
# points.txt writes them, and the path (path.txt).
HPKOT_POINTS: Dict[str, Dict] = {
    "tI1": {
        "parameters": [("H", "(1+c*c/a/a)/4")],
        "points": """
GAMMA 0 0 0
M -1/2 1/2 1/2
X 0 0 1/2
P 1/4 1/4 1/4
Z H H -H
Z_0 -H 1-H H
N 0 1/2 0
""",
        "path": "GAMMA-X X-M M-GAMMA GAMMA-Z Z_0-M X-P P-N N-GAMMA",
    },
    "tI2": {
        "parameters": [("H", "(1+a*a/c/c)/4"), ("Z", "a*a/2/c/c")],
        "points": """
GAMMA 0 0 0
M 1/2 1/2 -1/2
X 0 0 1/2
P 1/4 1/4 1/4
N 0 1/2 0
S_0 -H H H
S H 1-H -H
R -Z Z 1/2
G 1/2 1/2 -Z
""",
        "path": "GAMMA-X X-P P-N N-GAMMA GAMMA-M M-S S_0-GAMMA X-R G-M",
    },
    "hR1": {
        "parameters": [("D", "a*a/4/c/c"), ("Y", "5/6-2*D"), ("N", "1/3+D")],
        "points": """
GAMMA 0 0 0
T 1/2 1/2 1/2
L 1/2 0 0
L_2 0 -1/2 0
L_4 0 0 -1/2
F 1/2 0 1/2
F_2 1/2 1/2 0
S_0 N -N 0
S_2 1-N 0 N
S_4 N 0 -N
S_6 1-N N 0
H_0 1/2 -1+Y 1-Y
H_2 Y 1-Y 1/2
H_4 Y 1/2 1-Y
H_6 1/2 1-Y -1+Y
M_0 N -1+Y N
M_2 1-N 1-Y 1-N
M_4 Y N N
M_6 1-N 1-N 1-Y
M_8 N N -1+Y
""",
        "path": "GAMMA-T T-H_2 H_0-L L-GAMMA GAMMA-S_0 S_2-F F-GAMMA",
    },
    "hR2": {
        "parameters": [("Z", "1/6-c*c/9/a/a"), ("H", "1/2-2*Z"), ("N", "1/2+Z")],
        "points": """
GAMMA 0 0 0
T 1/2 -1/2 1/2
P_0 H -1+H H
P_2 H H H
R_0 1-H -H -H
M 1-N -N 1-N
M_2 N -1+N -1+N
L 1/2 0 0
F 1/2 -1/2 0
""",
        "path": "GAMMA-L L-T T-P_0 P_2-GAMMA GAMMA-F",
    },
    "oF1": {
        "parameters": [("J", "(1+a*a/b/b-a*a/c/c)/4"), ("H", "(1+a*a/b/b+a*a/c/c)/4")],
        "points": """
GAMMA 0 0 0
T 1 1/2 1/2
Z 1/2 1/2 0
Y 1/2 0 1/2
SIGMA_0 0 H H
U_0 1 1-H 1-H
A_0 1/2 1/2+J J
C_0 1/2 1/2-J 1-J
L 1/2 1/2 1/2
""",
        "path": "GAMMA-Y Y-T T-Z Z-GAMMA GAMMA-SIGMA_0 U_0-T Y-C_0 A_0-Z GAMMA-L",
    },
    "oC2": {
        "parameters": [("X", "(1+b*b/a/a)/4")],
        "points": """
GAMMA 0 0 0
Y 1/2 1/2 0
T 1/2 1/2 1/2
T_2 1/2 1/2 -1/2
Z 0 0 1/2
Z_2 0 0 -1/2
S 0 1/2 0
R 0 1/2 1/2
R_2 0 1/2 -1/2
DELTA_0 -X X 0
F_0 X 1-X 0
B_0 -X X 1/2
B_2 -X X -1/2
G_0 X 1-X 1/2
G_2 X 1-X -1/2
""",
        "path": "GAMMA-Y Y-F_0 DELTA_0-GAMMA GAMMA-Z Z-B_0 G_0-T T-Y GAMMA-S S-R R-Z Z-T",
    },
    "mC1": {
        "parameters": [("Z", "(2+a/c*cosbeta)/4/sinbeta/sinbeta"),
                       ("H", "1/2-2*Z*c*cosbeta/a"),
                       ("S", "3/4-b*b/4/a/a/sinbeta/sinbeta"),
                       ("P", "S-(3/4-S)*a*cosbeta/c")],
        "points": """
GAMMA 0 0 0
Y_2 -1/2 1/2 0
Y_4 1/2 -1/2 0
A 0 0 1/2
M_2 -1/2 1/2 1/2
V 1/2 0 0
V_2 0 1/2 0
L_2 0 1/2 1/2
C 1-S 1-S 0
C_2 -1+S S 0
C_4 S -1+S 0
D -1+P P 1/2
D_2 1-P 1-P 1/2
E -1+Z 1-Z 1-H
E_2 -Z Z H
E_4 Z -Z 1-H
""",
        "path": "GAMMA-C C_2-Y_2 Y_2-GAMMA GAMMA-M_2 M_2-D D_2-A A-GAMMA L_2-GAMMA GAMMA-V_2",
    },
}

# MACE's static lookup names every C-centred monoclinic lattice "mS1"; its
# entry is SeeK-path's mC1 path with the primed copy SeeK-path appends
# without time reversal.
ALIASES = {"mS1": "mC1"}

# HPKOT's primitive vectors in terms of its conventional ones,
# (a_P, b_P, c_P) = (a, b, c) P (SeeK-path hpkot.spg_mapping.get_P_matrix,
# HPKOT Table 3); as rows, A_hp = P^T A_conv.
HPKOT_P = {
    "tI": np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]]) / 2,
    "oF": np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]) / 2,
    "hR": np.array([[2, -1, -1], [1, 1, -2], [1, 1, 1]]) / 3,
    "oC": np.array([[1, 1, 0], [-1, 1, 0], [0, 0, 2]]) / 2,
    "mC": np.array([[1, -1, 0], [1, 1, 0], [0, 0, 2]]) / 2,
}

# CRYSTAL's "TRANSFORMATION MATRIX PRIMITIVE-CRYSTALLOGRAPHIC CELL" W as its
# outputs print it, by centring (CRYSTAL23 manual, Appendix A, Table A.1
# p.426, the "-> P" matrices; R is the hexagonal-axes cell): the
# conventional vectors are the columns of W in terms of the primitive ones,
# so as rows A_conv = W^T A_prim and A_prim = (W^T)^-1 A_conv.
CRYSTAL_W = {
    "A": np.array([[1, 0, 0], [0, 1, 1], [0, -1, 1]]),
    "B": np.array([[1, 0, -1], [0, 1, 0], [1, 0, 1]]),
    "C": np.array([[1, 1, 0], [-1, 1, 0], [0, 0, 1]]),
    "F": np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]]),
    "I": np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]),
    "R": np.array([[1, 0, 1], [-1, 1, 1], [0, -1, 1]]),
}

# The centring each lattice family's conventional cell has in CRYSTAL
FAMILY_CENTRING = {"tI": "I", "oF": "F", "hR": "R", "oC": "C", "mC": "C"}


def hpkot_key(key: str) -> Optional[str]:
    """The HPKOT_POINTS entry a static lookup key uses, or None."""
    base = key.split("_noinv")[0]
    base = ALIASES.get(base, base)
    return base if base in HPKOT_POINTS else None


def _cell(params: Dict[str, float]):
    return (params["a"], params["b"], params["c"],
            params["alpha"], params["beta"], params["gamma"])


def conventional_vectors(params: Dict[str, float]) -> np.ndarray:
    """Rows a, b, c: a along x, b in the xy plane (as seekpath_interface)."""
    a, b, c, al, be, ga = _cell(params)
    al, be, ga = np.radians([al, be, ga])
    cx = c * np.cos(be)
    cy = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
    cz = np.sqrt(max(0.0, c * c - cx * cx - cy * cy))
    return np.array([[a, 0.0, 0.0], [b * np.cos(ga), b * np.sin(ga), 0.0], [cx, cy, cz]])


def hpkot_variant(family: str, params: Dict[str, float]) -> str:
    """SeeK-path's extended Bravais lattice for a conventional cell (hpkot.get_path)."""
    a, b, c, _, beta, _ = _cell(params)
    cosbeta = np.cos(np.radians(beta))
    if family == "tI":
        return "tI1" if c <= a else "tI2"
    if family == "oF":
        if 1 / a ** 2 > 1 / b ** 2 + 1 / c ** 2:
            return "oF1"
        if 1 / c ** 2 > 1 / a ** 2 + 1 / b ** 2:
            return "oF2"
        return "oF3"
    if family == "hR":
        return "hR1" if np.sqrt(3) * a <= np.sqrt(2) * c else "hR2"
    if family == "oC":
        return "oC1" if a <= b else "oC2"
    if family == "mC":
        if b < a * np.sqrt(1 - cosbeta ** 2):
            return "mC1"
        return "mC2" if -a * cosbeta / c + a * a * (1 - cosbeta ** 2) / b ** 2 <= 1 else "mC3"
    raise ValueError(family)


def check_setting(family: str, params: Dict[str, float]) -> None:
    """Raise NotInHpkotSetting unless the cell is in the setting SeeK-path standardises to.

    SeeK-path evaluates its formulas in spglib's standardised conventional
    cell. CRYSTAL's conventional cell is the same cell when: tI a = b and
    right angles; hR hexagonal axes (a = b, gamma = 120); oF a < b < c (spglib
    orders the axes of these groups so); oC right angles; mC b unique,
    beta > 90 and the a-c mesh reduced as spglib reduces it
    (|a c cos beta| <= a^2/2 and <= c^2).
    """
    a, b, c, al, be, ga = _cell(params)

    def right(*angles):
        return all(abs(x - 90.0) < 1e-3 for x in angles)

    if family == "tI" and not (abs(a - b) < 1e-4 and right(al, be, ga)):
        raise NotInHpkotSetting("the conventional cell is not tetragonal with a = b")
    if family == "hR" and not (abs(a - b) < 1e-4 and right(al, be) and abs(ga - 120.0) < 1e-3):
        raise NotInHpkotSetting("the conventional cell is not in hexagonal axes")
    if family == "oF":
        if not right(al, be, ga):
            raise NotInHpkotSetting("the conventional cell is not orthorhombic")
        if not a < b < c:
            raise NotInHpkotSetting(
                f"the conventional axes are not in SeeK-path's order a < b < c "
                f"(a = {a:g}, b = {b:g}, c = {c:g})")
    if family == "oC" and not right(al, be, ga):
        raise NotInHpkotSetting("the conventional cell is not orthorhombic")
    if family == "mC":
        if not right(al, ga):
            raise NotInHpkotSetting("the monoclinic axis is not b")
        if be <= 90.0:
            raise NotInHpkotSetting(f"beta = {be:g} is not obtuse")
        ac = abs(a * c * np.cos(np.radians(be)))
        if ac > a * a / 2 + 1e-9 or ac > c * c + 1e-9:
            raise NotInHpkotSetting("the a-c mesh is not reduced as SeeK-path's standard cell is")


def evaluate(entry_key: str, params: Dict[str, float]) -> Dict[str, List[float]]:
    """Every point of an entry, in HPKOT's primitive reciprocal basis."""
    entry = HPKOT_POINTS[entry_key]
    a, b, c, _, beta, _ = _cell(params)
    cosbeta = np.cos(np.radians(beta))
    names = {"a": a, "b": b, "c": c, "cosbeta": cosbeta, "sinbeta": np.sqrt(1 - cosbeta ** 2)}
    for name, expr in entry["parameters"]:
        names[name] = eval(expr, {"__builtins__": {}}, names)  # trusted constants above
    points = {}
    for line in entry["points"].strip().splitlines():
        label, *xyz = line.split()
        points[label] = [float(eval(v, {"__builtins__": {}}, names)) for v in xyz]
    return points


def parametric_names(entry_key: str) -> set:
    """Names of the entry's lattice-dependent points (coordinates with a parameter)."""
    names = set()
    for line in HPKOT_POINTS[entry_key]["points"].strip().splitlines():
        label, *xyz = line.split()
        if any(ch.isalpha() for expr in xyz for ch in expr):
            names.add(label)
    return names


def crystal_reciprocal(conv_params: Dict[str, float], centering: str, w=None) -> np.ndarray:
    """Reciprocal vectors (rows, 2 pi included) of CRYSTAL's primitive cell."""
    return 2 * np.pi * np.linalg.inv(crystal_primitive(conv_params, centering, w)).T


def hpkot_standard_primitive(key: str, conv_params: Dict[str, float]) -> np.ndarray:
    """HPKOT's primitive vectors (rows) built from a conventional cell."""
    family = HPKOT_P[key[:2]]
    return family.T @ conventional_vectors(conv_params)


def crystal_primitive(conv_params: Dict[str, float], centering: str, w=None) -> np.ndarray:
    """CRYSTAL's primitive vectors (rows): (W^T)^-1 A_conv, W printed or Table A.1."""
    w = CRYSTAL_W[centering] if w is None else np.array(w, dtype=float)
    return np.linalg.inv(np.array(w, dtype=float).T) @ conventional_vectors(conv_params)


def crystal_points(lookup_key: str, conv_params: Dict[str, float],
                   w=None) -> Tuple[Dict[str, List[float]], str]:
    """The entry's points for this cell in CRYSTAL's reciprocal basis, and its HPKOT key.

    A_crystal = (W^T)^-1 A_conv (W as printed, or Table A.1), A_hp from
    HPKOT's P; N = A_hp A_crystal^-1 (seekpath_interface.crystal_cell_matrix)
    and each point is N^-1 f_hp, as on the library route. A primed point (the
    time-reversed copy SeeK-path adds without inversion and time reversal)
    is -f. Raises NotInHpkotSetting when the cell is not in HPKOT's setting,
    is another HPKOT variant, or N is not an integer matrix of determinant 1.
    """
    key = hpkot_key(lookup_key)
    if key is None:
        raise NotInHpkotSetting(f"no lattice-dependent formulas for {lookup_key}")
    family = key[:2]
    check_setting(family, conv_params)
    variant = hpkot_variant(family, conv_params)
    if variant != key:
        raise NotInHpkotSetting(f"SeeK-path takes {variant} for this cell, not {key}")
    a_hp = hpkot_standard_primitive(key, conv_params)
    a_crystal = crystal_primitive(conv_params, FAMILY_CENTRING[family], w)
    try:
        from seekpath_interface import crystal_cell_matrix
    except ImportError:  # pragma: no cover - package-relative import
        from .seekpath_interface import crystal_cell_matrix
    try:
        n = crystal_cell_matrix({"primitive_lattice": a_hp, "rotation_matrix": np.eye(3)}, a_crystal)
    except ValueError as exc:
        raise NotInHpkotSetting(str(exc))
    if not (np.allclose(n, np.round(n)) and abs(abs(np.linalg.det(n)) - 1) < 1e-9):
        raise NotInHpkotSetting("CRYSTAL's primitive cell is not an integer setting of HPKOT's")
    m = np.round(np.linalg.inv(n))
    points = {}
    for label, f in evaluate(key, conv_params).items():
        g = m @ np.array(f)
        g[np.abs(g) < 1e-12] = 0.0
        points[label] = [float(x) for x in g]
        if label != "GAMMA":
            points[label + "'"] = [0.0 if x == 0 else -float(x) for x in g]
    return points, key
