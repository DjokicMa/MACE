"""CRYSTAL's fort.34 (EXTERNAL geometry, also .gui / .f34 / .ext): read with its symmetry.

The format (CRYSTAL23 manual, EXTERNAL, pp. 450-451), all lengths in Å, Cartesian:

    dim  centering crystal-type          dimensionality 0-3, then two codes
    a1 a2 a3                             the three lattice vectors; a non-periodic
    b1 b2 b3                             direction holds 500 Å (c of a SLAB, b and c
    c1 c2 c3                             of a POLYMER, all three of a MOLECULE)
    nsym                                 the symmetry operators, 4 records each:
    R11 R12 R13                          the 3x3 rotation (Cartesian)
    R21 R22 R23
    R31 R32 R33
    t1 t2 t3                             and the translation (Cartesian, Å)
    nirr                                 the atoms of the asymmetric unit
    Z x y z                              (Z: CRYSTAL's numbering, 2xx an ECP)
    sg nops                              (optional) the space group number and its operator count

The cell is the orbit of the listed atoms under the operators, de-duplicated
(ase's reader refuses any file with more than the identity). CRYSTAL23 writes
the operators for a column vector, ``x' = R x + t`` (checked on a real EXTPRT
run, P3_1 Se: tests/data/fort34/Se_P31_crystal23.f34); a file written the other
way round (``x' = x R + t``) is read too: of the two, the one that maps the
generated cell onto itself is used, the column reading when both do.

Screw axes (the 1.1.6 check): the two readings differ only for an operator whose
rotation is not its own transpose, paired with a translation. R^T = R^-1, so
reading a 3_1 operator the other way round gives 3_2 (likewise 4_1/4_3, 6_1/6_5,
6_2/6_4). From the whole cell (CRYSTAL23 lists every atom) the wrong reading
still closes, but only by adding the mirror image's atoms, so the reading that
generates fewer atoms is used. From an asymmetric unit alone both readings give
the same number of atoms, the two enantiomorphic structures: the file's trailing
record (space group number) decides through spglib for a 3D cell; without it
(or for a layer or rod group) the column reading is kept and the result says
``convention_ambiguous`` with both groups, so the reader can warn.
CRYSTALpytools writes pymatgen's ``rotation_matrix`` (which acts as ``R x``)
row by row, i.e. the column reading too.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

#: Å; CRYSTAL marks a non-periodic lattice direction with a 500 Å vector
NONPERIODIC = 500.0
#: Å; two generated atoms closer than this are the same atom
DEDUP_TOL = 1e-3
#: Å; the generated cell must map onto itself to within this under every operator
CLOSURE_TOL = 1e-2


class Fort34Error(ValueError):
    """A fort.34 that cannot be read, said in words (which record, what is wrong)."""


def _floats(line: str, n: int, what: str, lineno: int) -> List[float]:
    parts = line.replace("D", "E").replace("d", "e").split()
    try:
        vals = [float(x) for x in parts[:n]]
    except ValueError:
        raise Fort34Error(f"line {lineno} ({what}) is not {n} numbers: {line.strip()!r}") from None
    if len(vals) < n:
        raise Fort34Error(f"line {lineno} ({what}) has {len(vals)} numbers, {n} expected: {line.strip()!r}")
    return vals


def _int(line: str, what: str, lineno: int) -> int:
    try:
        return int(float(line.split()[0]))
    except (ValueError, IndexError):
        raise Fort34Error(f"line {lineno} ({what}) is not a count: {line.strip()!r}") from None


def parse(text: str) -> Dict[str, Any]:
    """The records of a fort.34, unexpanded: dimensionality, lattice (3x3),
    rotations (n,3,3), translations (n,3), numbers and positions of the
    asymmetric unit (Cartesian Å)."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 6:
        raise Fort34Error(f"only {len(lines)} records; a fort.34 needs a header, 3 lattice vectors, "
                          "the symmetry operators and the atoms")
    head = lines[0].split()
    try:
        dim = int(float(head[0]))
    except (ValueError, IndexError):
        raise Fort34Error(f"line 1 (dimensionality) is not a number: {lines[0].strip()!r}") from None
    if dim not in (0, 1, 2, 3):
        raise Fort34Error(f"line 1: dimensionality {dim}, expected 0 to 3")
    lattice = np.array([_floats(lines[i], 3, "lattice vector", i + 1) for i in (1, 2, 3)])
    i = 4
    nsym = _int(lines[i], "number of symmetry operators", i + 1)
    if nsym < 1:
        raise Fort34Error(f"line {i + 1}: {nsym} symmetry operators (at least the identity expected)")
    i += 1
    need = i + 4 * nsym + 1
    if len(lines) < need:
        raise Fort34Error(f"line {i}: {nsym} symmetry operators (4 records each) announced, "
                          f"but the file ends after {len(lines)} records")
    rots, trans = [], []
    for k in range(nsym):
        r = [_floats(lines[i + j], 3, f"rotation of operator {k + 1}", i + j + 1) for j in range(3)]
        t = _floats(lines[i + 3], 3, f"translation of operator {k + 1}", i + 4)
        rots.append(r)
        trans.append(t)
        i += 4
    nirr = _int(lines[i], "number of atoms", i + 1)
    i += 1
    if nirr < 1 or len(lines) < i + nirr:
        raise Fort34Error(f"line {i}: {nirr} atoms announced, {max(0, len(lines) - i)} atom records follow")
    numbers, pos = [], []
    for k in range(nirr):
        parts = lines[i + k].split()
        try:
            numbers.append(int(float(parts[0])))
        except (ValueError, IndexError):
            raise Fort34Error(f"line {i + k + 1} (atom {k + 1}) has no atomic number: {lines[i + k].strip()!r}") from None
        pos.append(_floats(" ".join(parts[1:]), 3, f"position of atom {k + 1}", i + k + 1))
    declared = None
    k = i + nirr
    if k < len(lines):
        parts = lines[k].split()
        try:
            declared = int(float(parts[0])) if parts else None
        except ValueError:
            declared = None
    rots = np.array(rots, float)
    dev = np.abs(np.einsum("nij,nkj->nik", rots, rots) - np.eye(3)).max(axis=(1, 2))
    bad = np.nonzero(dev > 1e-3)[0]
    if len(bad):
        raise Fort34Error(f"symmetry operator {bad[0] + 1} is not a rotation (not orthogonal): "
                          f"{rots[bad[0]].round(4).tolist()}")
    periodic = lattice[:dim]
    if dim and abs(np.linalg.det(lattice)) < 1e-6:
        raise Fort34Error("the lattice vectors are linearly dependent (zero cell volume)")
    if dim and np.linalg.matrix_rank(periodic, tol=1e-6) < dim:
        raise Fort34Error(f"the {dim} periodic lattice vectors are linearly dependent")
    return {"dim": dim, "header": head, "lattice": lattice, "rotations": rots,
            "translations": np.array(trans, float), "numbers": numbers,
            "positions": np.array(pos, float), "declared_group": declared}


def _wrap(cart: np.ndarray, lattice: np.ndarray, dim: int) -> np.ndarray:
    """Cartesian positions with the periodic fractional coordinates folded into [0, 1)."""
    if dim == 0:
        return cart.copy()
    frac = np.linalg.solve(lattice.T, cart.T).T
    frac[:, :dim] -= np.floor(frac[:, :dim] + 1e-9)
    return frac @ lattice


def _dist(a: np.ndarray, b: np.ndarray, lattice: np.ndarray, dim: int) -> np.ndarray:
    """Minimum-image distances between every row of ``a`` and of ``b`` (Å)."""
    d = a[:, None, :] - b[None, :, :]
    if dim:
        f = np.linalg.solve(lattice.T, d.reshape(-1, 3).T).T
        f[:, :dim] -= np.round(f[:, :dim])
        d = (f @ lattice).reshape(d.shape)
    return np.linalg.norm(d, axis=2)


def _apply(rots, trans, cart, row: bool) -> np.ndarray:
    """Every operator on every position: (nsym, natoms, 3)."""
    if row:
        return np.einsum("aj,njk->nak", cart, rots) + trans[:, None, :]
    return np.einsum("nkj,aj->nak", rots, cart) + trans[:, None, :]


def _expand(rec: Dict[str, Any], row: bool) -> Tuple[np.ndarray, List[int], float]:
    lat, dim = rec["lattice"], rec["dim"]
    rots, trans = rec["rotations"], rec["translations"]
    out_pos: List[np.ndarray] = []
    out_z: List[int] = []
    for z, p in zip(rec["numbers"], rec["positions"]):
        images = _wrap(_apply(rots, trans, p[None, :], row)[:, 0, :], lat, dim)
        for x in images:
            if out_pos:
                same = [k for k in range(len(out_pos)) if out_z[k] == z]
                if same and _dist(x[None], np.array([out_pos[k] for k in same]), lat, dim).min() < DEDUP_TOL:
                    continue
            out_pos.append(x)
            out_z.append(z)
    cart = np.array(out_pos)
    # closure: every operator maps the cell onto itself (same element)
    worst = 0.0
    zs = np.array(out_z)
    moved = _apply(rots, trans, cart, row)
    for z in set(out_z):
        m = zs == z
        d = _dist(moved[:, m, :].reshape(-1, 3), cart[m], lat, dim)
        worst = max(worst, float(d.min(axis=1).max()))
    return cart, out_z, worst


def _same_cell(a, b, lat, dim) -> bool:
    """Do two expansions hold the same atoms (any order, minimum image)?"""
    (ca, za), (cb, zb) = a, b
    if len(za) != len(zb) or sorted(za) != sorted(zb):
        return False
    zb = np.asarray(zb)
    for x, z in zip(ca, za):
        m = zb == z
        if _dist(x[None], cb[m], lat, dim).min() >= DEDUP_TOL * 10:
            return False
    return True


def _spacegroup_number(cart, numbers, lat) -> int:
    try:
        import spglib
        frac = np.linalg.solve(lat.T, np.asarray(cart).T).T
        ds = spglib.get_symmetry_dataset((lat, frac % 1.0, [int(z) % 100 or int(z) for z in numbers]), symprec=1e-3)
        n = ds["number"] if isinstance(ds, dict) else getattr(ds, "number", None)
        return int(n) if n else 0
    except Exception:
        return 0


def expand(rec: Dict[str, Any]) -> Dict[str, Any]:
    """The full cell of a parse() result: {cart (n,3), numbers, nsym, nirr,
    convention[, convention_ambiguous, groups]}. Raises Fort34Error when the
    operators do not map the atoms onto themselves either way round (a damaged file)."""
    best = None
    tries = {}
    for row in (True, False):
        cart, nums, worst = _expand(rec, row)
        if worst < CLOSURE_TOL:
            tries[row] = (cart, nums)
        if best is None or worst < best:
            best = worst
    if tries:
        lat, dim = rec["lattice"], rec["dim"]
        row = False if False in tries else True
        out = {"nsym": len(rec["rotations"]), "nirr": len(rec["numbers"])}
        if len(tries) == 2 and not _same_cell(tries[True], tries[False], lat, dim):
            # an enantiomorphic screw axis read both ways (see the module doc)
            n_row, n_col = len(tries[True][1]), len(tries[False][1])
            groups = {r: (_spacegroup_number(*tries[r], lat) if dim == 3 else 0) for r in (True, False)}
            want = rec.get("declared_group")
            match = [r for r in (True, False) if want and groups[r] == want]
            if n_row != n_col:
                row = n_row < n_col            # the wrong reading of a whole cell adds the mirror image's atoms
            elif len(match) == 1:
                row = match[0]
            else:
                out.update(convention_ambiguous=True, groups={"row": groups[True], "column": groups[False]})
        cart, nums = tries[row]
        out.update(cart=cart, numbers=nums, convention="row" if row else "column")
        return out
    raise Fort34Error(f"its {len(rec['rotations'])} symmetry operators do not map the "
                      f"{len(rec['numbers'])} atoms of its asymmetric unit onto one cell "
                      f"(an atom lands {best:.3f} Å from any other): the operators or the atoms are damaged")


def read(path) -> Dict[str, Any]:
    """A fort.34 file: parse() + expand(), with ``dim``, ``lattice`` and ``pbc``."""
    path = Path(path)
    try:
        text = path.read_text(errors="replace")
    except OSError as exc:
        raise Fort34Error(f"cannot open it: {exc.strerror or exc}") from exc
    rec = parse(text)
    full = expand(rec)
    full.update(dim=rec["dim"], lattice=rec["lattice"], header=rec["header"],
                pbc=[i < rec["dim"] for i in range(3)])
    return full
