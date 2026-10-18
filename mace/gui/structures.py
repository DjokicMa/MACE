"""Structures for the GUI: read CIF / .d12 / CRYSTAL .out, symmetry, comparison.

Everything here returns plain JSON-able dicts, so the browser only draws.
The readers reuse what MACE already has: ase for CIF, ``CrystalInputParser``
for decks and ``seekpath_interface._geometry_blocks`` for every geometry a
CRYSTAL output prints (initial, each OPT point, final).

Positions CRYSTAL prints for the non-periodic directions of a SLAB, POLYMER
or MOLECULE are Cartesian (Angstrom); they are kept Cartesian here and the
cell vector along such a direction is only a drawing aid (``pbc`` False).
"""
from __future__ import annotations

import contextlib
import io
import re
import sys
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
for _sub in ("Crystal_d12", "Crystal_d3"):
    _p = str(_REPO / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Two atoms are bonded (drawn, tabulated, cleaned) when closer than
# BOND_SCALE x the sum of their covalent radii, metal-metal pairs excepted.
BOND_SCALE = 1.2

# CRYSTAL's dummy length for a non-periodic direction
_DUMMY = 500.0

# CRYSTAL23 prints "MOLECULAR CALCULATION" for a molecule (all 331 corpus molecule outputs)
_DIM_BANNERS = (("MOLECULAR CALCULATION", 0), ("POLYMER CALCULATION", 1),
                ("SLAB CALCULATION", 2), ("CRYSTAL CALCULATION", 3))
_DIM_NAMES = {3: "CRYSTAL", 2: "SLAB", 1: "POLYMER", 0: "MOLECULE"}

_ENERGY = re.compile(r"TOTAL ENERGY\((?:DFT|HF)\)\(AU\)\(\s*\d+\)\s+(\S+)")


def _symbols():
    from ase.data import chemical_symbols
    return chemical_symbols


def cellpar_to_cell(params: Dict[str, float]) -> np.ndarray:
    from ase.geometry import cellpar_to_cell as _c
    return np.asarray(_c([params["a"], params["b"], params["c"],
                          params["alpha"], params["beta"], params["gamma"]]))


def _lattice(cell: np.ndarray) -> Dict[str, float]:
    from ase.geometry import cell_to_cellpar
    a, b, c, al, be, ga = cell_to_cellpar(cell)
    return {"a": a, "b": b, "c": c, "alpha": al, "beta": be, "gamma": ga,
            "volume": float(abs(np.linalg.det(cell)))}


def symmetry(cell, frac, numbers, pbc=(True, True, True),
             symprec: float = 1e-3, with_ops: bool = False) -> Dict[str, Any]:
    """Space group from spglib (3D only; low-dimensional cells are reported
    as such, since a layer/rod group needs CRYSTAL's own tables)."""
    if not all(pbc):
        return {"available": False,
                "note": "spglib space groups are for 3D cells; CRYSTAL reports "
                        "the layer/rod group in the output"}
    try:
        import spglib
        from spglib_compat import attribute_dataset
    except Exception as exc:  # pragma: no cover - spglib is a requirement
        return {"available": False, "note": f"spglib unavailable: {exc}"}
    ds = attribute_dataset(spglib.get_symmetry_dataset(
        (np.asarray(cell), np.asarray(frac) % 1.0, list(numbers)), symprec=symprec))
    if ds is None:
        return {"available": False, "note": "spglib found no symmetry"}
    get = lambda name: getattr(ds, name)  # noqa: E731
    wyck = list(get("wyckoffs"))
    equiv = [int(x) for x in get("equivalent_atoms")]
    try:
        site = [str(x) for x in get("site_symmetry_symbols")]
    except AttributeError:   # a very old spglib
        site = []
    # spglib counts the operations and an orbit's atoms in the cell it was
    # given; the tables (and the user) count them in the conventional cell.
    # So a supercell or a primitive cell is scaled by the volume ratio: MgO
    # stays 192 operations and 4a, as a 10x10x10 or a primitive cell too.
    try:
        ratio = abs(float(np.linalg.det(np.asarray(get("std_lattice"), float)))) / abs(float(np.linalg.det(np.asarray(cell, float))))
    except Exception:
        ratio = 1.0
    if not np.isfinite(ratio) or ratio <= 0:
        ratio = 1.0
    count: Dict[int, int] = {}
    for e in equiv:
        count[e] = count.get(e, 0) + 1
    n_ops = int(len(get("rotations")))
    out = {
        "available": True,
        "number": int(get("number")),
        "international": str(get("international")),
        "hall": str(get("hall")),
        "pointgroup": str(get("pointgroup")),
        "n_operations": n_ops,
        "n_operations_conventional": max(1, int(round(n_ops * ratio))),
        "multiplicity": [max(1, int(round(count[e] * ratio))) for e in equiv],
        "unique_atoms": len(set(equiv)),
        "wyckoffs": wyck,
        "equivalent_atoms": equiv,
        "site_symmetry": site,
        "symprec": symprec,
    }
    if with_ops:     # the operations themselves, for the browser's orbit previews
        out["rotations"] = np.asarray(get("rotations")).astype(int).tolist()
        out["translations"] = np.asarray(get("translations"), float).tolist()
    return out


def _gated_symmetry(cell, frac, numbers, pbc):
    """symmetry() through the GUI's compute gate (section 9: the core cap, and
    one big structure at a time, since spglib needs tens of MB on a large
    supercell). Never called from inside the gate (the editor passes its own)."""
    from .compute import GATE, HEAVY_ATOMS
    return GATE.run(None, lambda: symmetry(cell, frac, numbers, pbc), heavy=len(numbers) > HEAVY_ATOMS)


SYMMETRY_INLINE_MAX = 20000     # atoms: spglib on more (minutes, gigabytes) is a task the user asks for


def _deferred_symmetry(n: int) -> Dict[str, Any]:
    return {"available": False, "not_computed": True,
            "note": f"Not computed above {SYMMETRY_INLINE_MAX:,} atoms ({n:,} here): spglib would take minutes"}


def _payload(cell, frac_or_none, cart, numbers, pbc, source, extra=None, symmetry_data=None):
    """The structure as JSON-able lists; ``symmetry_data`` is a symmetry() result the
    caller already has (the editor's), so spglib does not run twice."""
    cell = np.asarray(cell, float)
    cart = np.asarray(cart, float)
    numbers = [int(n) for n in numbers]
    if frac_or_none is None:
        frac = np.linalg.solve(cell.T, cart.T).T
    else:
        frac = np.asarray(frac_or_none, float)
    sym = _symbols()
    out = {
        "source": source,
        "cell": cell.tolist(),
        "pbc": [bool(p) for p in pbc],
        "dimensionality": _DIM_NAMES[int(sum(pbc))],
        "numbers": numbers,
        "symbols": [sym[n] for n in numbers],
        "radii": [float(r) for r in _covalent(numbers)],
        "metal": [bool(m) for m in _is_metal(numbers)],
        "cart": cart.tolist(),
        "frac": frac.tolist(),
        "lattice": _lattice(cell),
        "formula": _formula(numbers),
        "reduced_formula": reduced_formula(numbers)[0],
        "z": reduced_formula(numbers)[1],
        "bond_scale": BOND_SCALE,
        "symmetry": symmetry_data if symmetry_data is not None else (
            _deferred_symmetry(len(numbers)) if len(numbers) > SYMMETRY_INLINE_MAX and all(pbc)
            else _gated_symmetry(cell, frac, numbers, pbc)),
        "origin": [0.0, 0.0, 0.0],
    }
    if extra:
        out.update(extra)
    return out


def reduced_formula(numbers: Sequence[int]) -> Tuple[str, int]:
    """(reduced formula, Z): elements in ascending Pauling electronegativity,
    as pymatgen writes them (SiO2, GaAs, CsPbI3, NaHCO3), with the number of
    formula units in the cell. A single element reads C with Z = 8.
    (mace.utils.formula, which the CLI uses too)"""
    from mace.utils.formula import count_symbols, reduced_formula as _reduced
    sym = _symbols()
    return _reduced(count_symbols(sym[int(n)] for n in numbers))


def reduce_formula_text(formula: str) -> Optional[Tuple[str, int]]:
    """'F4Li4' (a cell formula, as materials.db stores it) -> ('LiF', 4).
    None when the text is not a plain formula (brackets, charges, ...)."""
    from mace.utils.formula import parse_formula, reduced_formula as _reduced
    counts = parse_formula(formula)
    return _reduced(counts) if counts else None


def _formula(numbers: Sequence[int]) -> str:
    """The cell's full formula in Hill order, as ase writes it (C and H first)."""
    from mace.utils.formula import count_symbols, format_formula
    sym = _symbols()
    return format_formula(count_symbols(sym[int(n)] for n in numbers), "hill-ase")


# --------------------------------------------------------------------------
# readers
# --------------------------------------------------------------------------

def _p1_cif_atoms(path: Path):
    """A P1 CIF with full occupancies, read without ase's symmetry expansion:
    ase.io.read checks every atom against every other one for duplicates,
    which takes minutes for a few thousand atoms (8000: 190 s), and with one
    operation and no partial sites it gives the very same atoms. None for
    anything else, which then takes ase's own path."""
    try:
        from ase.io.cif import parse_cif
        from mace.utils.cif_symbols import normalize_block   # "BI" is bismuth, not boron
        blocks = [b for b in map(normalize_block, parse_cif(str(path))) if b.has_structure()]
        if not blocks:
            return None
        b = blocks[-1]                                # ase.io.read's default: the last one
        if len(b.get_spacegroup(True).get_symop()) != 1:
            return None
        occ = b.get("_atom_site_occupancy")
        if occ is not None and any(abs(float(x) - 1.0) > 1e-6 for x in occ):
            return None
        return b.get_unsymmetrized_structure()
    except Exception:
        return None


def cif_deck_origin(path: Path, atoms):
    """``atoms`` as cif2d12 writes them for the deck (the GUI and
    ``mace preopt`` both read CIFs through this).

    Without listed operators ase expands a CIF in origin choice 1. For the
    24 space groups with two origins cif2d12 reads which origin the CIF
    uses (its tags, else its atoms: NewCifToD12.resolve_cif_origin) and
    expands the records in that one. Same rule here, so the GUI shows the
    8-atom diamond and the 4-atom FeSe the decks describe, not ase's
    overlapping atoms. A CIF whose origin cannot be told raises
    ValueError, as cif2d12 refuses it."""
    sg = atoms.info.get("spacegroup")
    no = getattr(sg, "no", None)
    if no not in _TWO_ORIGINS:
        return atoms
    try:
        import NewCifToD12 as cif2d12
    except Exception:
        return atoms
    try:
        if cif2d12._cif_lists_symops(str(path)):
            return atoms
        records = cif2d12._parse_cif_atom_site_records(str(path))
        if not records:
            return atoms
        cellpar = atoms.cell.cellpar()
        with contextlib.redirect_stdout(io.StringIO()):
            choice, _ = cif2d12.resolve_cif_origin(str(path), no, records, cellpar,
                                                   None, False)
    except cif2d12.AmbiguousOriginError as exc:
        raise ValueError(f"{Path(path).name}: {exc}") from None
    except Exception:
        return atoms
    if choice is None or choice == getattr(sg, "setting", 1):
        return atoms
    # The same expansion cif2d12 writes, rebuilt with ase's crystal so the
    # atoms keep the group and spacegroup_kinds that mace preopt writes its
    # CIF from.
    from ase.spacegroup import crystal
    return crystal(records[0], records[1], spacegroup=no, setting=choice,
                   cellpar=cellpar, onduplicates="keep")


def read_cif(path: Path) -> Dict[str, Any]:
    import warnings
    from mace.utils.cif_symbols import read_cif as _read_cif
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        atoms = _p1_cif_atoms(path)
        if atoms is None:
            atoms = cif_deck_origin(path, _read_cif(path))
    return _payload(atoms.cell.array, atoms.get_scaled_positions(wrap=False),
                    atoms.positions, atoms.numbers, (True, True, True),
                    {"kind": "cif", "path": str(path)})


_TWO_ORIGINS = frozenset({48, 50, 59, 68, 70, 85, 86, 88, 125, 126, 129, 130, 133, 134,
                          137, 138, 141, 142, 201, 203, 222, 224, 227, 228})


_RHOMBOHEDRAL = frozenset({146, 148, 155, 160, 161, 166, 167})


def _deck_spacegroup(path: Path, parsed: Dict[str, Any]) -> int:
    """The deck's space group number; IFLAG = 1 decks give a symbol instead,
    which the parser leaves as None in ``spacegroup``."""
    if parsed.get("spacegroup") is not None:
        return int(parsed["spacegroup"])
    from d12_constants import spacegroup_number_from_symbol
    lines = Path(path).read_text(errors="replace").splitlines()
    for i, line in enumerate(lines[1:], 1):
        if line.strip() == "CRYSTAL" and i + 2 < len(lines):
            number = spacegroup_number_from_symbol(lines[i + 2].split("#", 1)[0].strip())
            if number:
                return number
            break
    raise ValueError(f"cannot read the geometry of {Path(path).name}: "
                     "its space group record is not a number or a symbol CRYSTAL reads")


def external_keyword(path: Path) -> Optional[str]:
    """EXTERNAL / DLVINPUT when the deck reads its geometry from fort.34."""
    from mace.submission.external_geometry import deck_keyword
    return deck_keyword(path)


def _missing_external_message(path: Path) -> str:
    from mace.submission.external_geometry import missing_fort34_message
    msg = missing_fort34_message(path)
    if msg is None:     # the staged file is there; d12_atoms still reads only inline geometry
        return (f"{Path(path).name} reads its geometry from an external file "
                f"(fort.34); open that file to see the structure")
    return msg + " Or open the .out of its run to see the geometry CRYSTAL used."


class OpenInstead(ValueError):
    """The file cannot be drawn, but ``path`` (a file next to it, e.g. the
    .out of a SLAB deck's run) shows the same structure: the server names it
    and the GUI opens it instead."""

    def __init__(self, message: str, path: Path):
        super().__init__(message)
        self.path = Path(path)


def d12_atoms(path: Path):
    """A 3D deck's structure as ase Atoms: the asymmetric unit expanded by the
    space group with ase's ``crystal`` (which records the site of each atom in
    ``spacegroup_kinds`` and the group in ``info["spacegroup"]``).

    CRYSTAL's default origin (IFSO 0) is ITA origin choice 2 where a group
    has two; ``0 0 1`` asks for choice 1.

    IFSO > 1 adds the record IX IY IZ, "non-standard shift of the origin ...
    in fractions of the crystallographic cell lattice vectors times 24"
    (manual p.19). CRYSTAL23 applies the group's operations about the point
    -s (s = IX IY IZ / 24) of the deck's frame: an atom at x has the image
    R(x + s) + t - s, which is what TESTGEOM prints (P-1, s = 1/8 0 0: the
    inversion image of x = 0.1 is -0.35). So the atoms are expanded at x + s
    and the result moved back by s, keeping the deck's own frame.
    """
    from d12_parsers import CrystalInputParser
    from ase.spacegroup import crystal

    keyword = external_keyword(path)
    if keyword:
        raise ValueError(_missing_external_message(path))
    parsed = CrystalInputParser(str(path)).parse()
    dim = parsed.get("dimensionality", "CRYSTAL")
    if dim != "CRYSTAL":
        out = Path(path).with_suffix(".out")
        if out.is_file():
            raise OpenInstead(f"{Path(path).name} is a {dim} deck, which MACE GUI does not draw from the deck; "
                              f"{out.name}, the output of its run, shows the structure", out)
        raise ValueError("MACE GUI reads 3D (CRYSTAL) decks; open the .out of a "
                         "SLAB/POLYMER/MOLECULE run instead")
    if "cell_parameters" not in parsed or "atoms" not in parsed:
        why = parsed.get("geometry_unparsed") or "no CRYSTAL geometry block after the title"
        raise ValueError(f"cannot read the geometry of {Path(path).name}: {why}")
    cp = parsed["cell_parameters"]
    sg = _deck_spacegroup(path, parsed)
    numbers = [int(a["atomic_number"]) % 100 for a in parsed["atoms"]]
    basis = [[a["x"], a["y"], a["z"]] for a in parsed["atoms"]]
    origin = (parsed.get("origin_setting") or "0 0 0").split()
    shift = None
    if parsed.get("origin_shift") is not None:
        shift = np.asarray(parsed["origin_shift"], float) / 24.0
        basis = (np.asarray(basis, float) + shift).tolist()
    # ase's setting 2 is origin choice 2 only for the groups that have two
    # origins; for a rhombohedral group it means rhombohedral axes, and
    # CRYSTAL's default there is hexagonal (ase setting 1) unless the deck's
    # IFHR is 1 (its cell record is then a, alpha of the rhombohedral cell)
    if sg in _TWO_ORIGINS:
        settings = [1, 2] if origin[-1:] == ["1"] else [2, 1]
    elif parsed.get("rhombohedral_axes") and sg in _RHOMBOHEDRAL:
        settings = [2]
    else:
        settings = [1]
    last = None
    for setting in settings:
        try:
            atoms = crystal(numbers, basis=basis, spacegroup=sg, setting=setting,
                            cellpar=[cp["a"], cp["b"], cp["c"],
                                     cp["alpha"], cp["beta"], cp["gamma"]],
                            primitive_cell=False, onduplicates="error")
            break
        except Exception as exc:  # setting unavailable or overlapping atoms
            last = exc
    else:
        raise ValueError(f"cannot expand space group {sg}: {last}")
    if shift is not None:
        atoms.set_scaled_positions(np.mod(atoms.get_scaled_positions(wrap=False) - shift, 1.0))
    return atoms, sg


def read_d12(path: Path) -> Dict[str, Any]:
    """A 3D deck: asymmetric unit + space group expanded with ase (d12_atoms).

    An EXTERNAL / DLVINPUT deck has no geometry of its own: it is drawn from
    the file its job would stage as fort.34, else from its run's .out, and the
    payload's source says which (``geometry_from``, ``geometry_staged``)."""
    if external_keyword(path):
        return _read_external_d12(Path(path))
    atoms, sg = d12_atoms(path)
    source = {"kind": "d12", "path": str(path), "deck_spacegroup": sg}
    # records that edit this geometry (a supercell, substituted atoms, ...):
    # CRYSTAL runs the edited structure; the drawing is the one before them
    from mace.preopt import deckscan
    edits = deckscan.scan(Path(path).read_text(errors="replace"))["geometry_edits"]
    if edits:
        source["geometry_edits"] = edits
    return _payload(atoms.cell.array, atoms.get_scaled_positions(wrap=False),
                    atoms.positions, atoms.numbers, (True, True, True), source)


def companion_stamp(path: Path) -> Tuple:
    """For an EXTERNAL / DLVINPUT deck, the (name, mtime_ns, size) of every
    file its drawing may come from (and of its job script, which names the
    staged one), None for a missing one; () for any other file. Part of the
    parse-cache key, so a changed .ext or .out redraws the deck."""
    path = Path(path)
    if path.suffix.lower() != ".d12" or not external_keyword(path):
        return ()
    from mace.submission.external_geometry import UNSTAGED_SUFFIXES, fort34_sources
    files = fort34_sources(path) + [path.with_suffix(x) for x in UNSTAGED_SUFFIXES] + [
        path.parent / "fort.34", path.with_suffix(".out"), path.with_suffix(".sh")]
    out = []
    for f in dict.fromkeys(files):
        try:
            st = f.stat()
            out.append((str(f), int(st.st_mtime_ns), int(st.st_size)))
        except OSError:
            out.append((str(f), None))
    return ("companions",) + tuple(out)


def _read_external_d12(path: Path) -> Dict[str, Any]:
    from mace.submission.external_geometry import fort34_source, unstaged_geometry_files
    staged = fort34_source(path)
    if staged is not None:
        candidates = [(staged, True)]
    else:
        candidates = [(p, False) for p in unstaged_geometry_files(path)]
    out = path.with_suffix(".out")
    if out.is_file():
        candidates.append((out, False))
    if not candidates:
        raise ValueError(_missing_external_message(path))
    src, is_staged = candidates[0]
    if src.suffix.lower() == ".out":
        data = read_out(src, -1)
    else:
        try:
            data = read_any(src, "fort34", box=False)
        except Exception as exc:  # ase's fort.34 reader: any malformed record
            raise ValueError(f"{path.name} reads its geometry from {src.name} (fort.34), "
                             f"which cannot be read: {exc}") from exc
    data["source"] = {"kind": "d12", "path": str(path), "geometry_from": str(src),
                      "geometry_staged": is_staged,
                      "geometry_keyword": external_keyword(path)}
    return data


_DIM_LINE = re.compile(r"DIMENSIONALITY OF THE SYSTEM\s+(\d)\s*$")


def _out_dimensionality(lines: List[str]) -> int:
    """The banner (``MOLECULAR CALCULATION`` ...) when CRYSTAL printed one, else
    the first ``DIMENSIONALITY OF THE SYSTEM n`` line, else 3. Runs restarted
    from EXTERNAL or GUESSP data print no banner; the dimensionality line is
    printed with every geometry (where both exist, they agree)."""
    for line in lines[:400]:
        for banner, dim in _DIM_BANNERS:
            if banner in line and "PCRYSTAL" not in line:
                return dim
    for line in lines[:400]:
        if "DIMENSIONALITY OF THE SYSTEM" in line:
            m = _DIM_LINE.search(line)
            if m and int(m.group(1)) <= 3:
                return int(m.group(1))
    return 3


# A molecule (dimensionality 0) prints no PRIMITIVE CELL block: its geometry is
# the "ATOMS IN THE ASYMMETRIC UNIT" table in Angstrom under one of these headers.
MOLECULE_BLOCK_START = re.compile(
    r"^\s*(GEOMETRY FOR WAVE FUNCTION|FINAL OPTIMIZED GEOMETRY) - DIMENSIONALITY OF THE SYSTEM\s+0\s*$")
_UNIT_CELL_ATOMS = re.compile(r"ATOMS IN THE UNIT CELL:\s*(\d+)")
_MPI_NOISE = re.compile(r"\s*PROCESS\s+\d+\s+OF\s+\d+\s+WORKING\s*")


def molecule_blocks(lines: List[str]) -> List[Dict[str, Any]]:
    """Every molecule geometry table of a CRYSTAL output, in order, as blocks
    shaped like ``seekpath_interface._geometry_blocks``' (``line``, ``params``,
    ``numbers``, ``positions`` in Angstrom, ``final``, ``opt_point``,
    ``n_atoms``) plus ``complete``: the table has ended (the blank line after
    its last row) with as many rows as CRYSTAL announced. Atom rows are read
    by the same ``_atom_row`` the periodic parser uses. A table cut by the end
    of ``lines`` (a run still writing it) is returned with ``complete`` False."""
    from seekpath_interface import _atom_row
    out: List[Dict[str, Any]] = []
    n = len(lines)
    i = 0
    while i < n:
        m = MOLECULE_BLOCK_START.match(lines[i])
        if not m:
            i += 1
            continue
        block = {"line": i, "params": {"a": _DUMMY, "b": _DUMMY, "c": _DUMMY, "alpha": 90.0, "beta": 90.0,
                                       "gamma": 90.0},
                 "numbers": [], "positions": [], "final": m.group(1).startswith("FINAL"),
                 "opt_point": False, "n_atoms": None, "complete": False, "molecule": True}
        j = i + 1
        header = False
        while j < n and j < i + 8:
            mm = _UNIT_CELL_ATOMS.search(lines[j])
            if mm:
                block["n_atoms"] = int(mm.group(1))
            if "X(ANGSTROM)" in lines[j]:
                header = True
                j += 1
                break
            j += 1
        if not header:
            i += 1
            continue
        while j < n and lines[j].lstrip().startswith("*"):
            j += 1
        while j < n:
            row = _atom_row(lines[j])
            if row is None and "PROCESS" in lines[j] and j + 1 < n:
                # another MPI rank's "PROCESS n OF m WORKING" written into the middle of a row: the
                # rest of the row went to the next line (seen in real Pcrystal outputs)
                fixed = _MPI_NOISE.sub(" ", lines[j]).rstrip() + " " + lines[j + 1].strip()
                row = _atom_row(fixed)
                if row is not None:
                    j += 1
            if row is None:
                break
            block["numbers"].append(row[0])
            block["positions"].append(row[1])
            j += 1
        ended = j < n            # a line after the last row: the table is over
        block["complete"] = bool(ended and block["numbers"]
                                 and (block["n_atoms"] is None or len(block["numbers"]) >= block["n_atoms"]))
        block["end_line"] = j
        out.append(block)
        i = j
    return out


def _block_structure(block: Dict[str, Any], dim: int) -> Optional[Dict[str, Any]]:
    params = block.get("params")
    if not params or not block["numbers"]:
        return None
    pos = np.asarray(block["positions"], float)
    numbers = [n % 100 for n in block["numbers"]]
    pbc = [i < dim for i in range(3)]
    if dim == 3:
        cell = cellpar_to_cell(params)
        return {"cell": cell, "frac": pos, "cart": pos @ cell, "numbers": numbers,
                "pbc": pbc}
    # low-dimensional: periodic directions fractional, the rest Cartesian.
    # A non-periodic cell vector spans just the atoms (a SLAB has no c): the
    # viewer draws the in-plane cell at the slab's faces, and adds vacuum
    # padding only when asked, so the vacuum direction never dominates.
    p = dict(params)
    span = np.ptp(pos[:, dim:], axis=0) if dim < 3 else []
    for i, key in enumerate(("a", "b", "c")):
        if i >= dim:
            p[key] = float(max(span[i - dim], 0.5)) if len(span) else 0.5
    if dim == 0:
        p.update(alpha=90.0, beta=90.0, gamma=90.0)
    cell = cellpar_to_cell(p)
    cart = np.zeros_like(pos)
    for i in range(3):
        if i < dim:
            cart += np.outer(pos[:, i], cell[i])
        else:
            cart[:, i] += pos[:, i]
    # the drawn box starts at the lowest atom along each non-periodic axis
    origin = np.zeros(3)
    for i in range(dim, 3):
        origin += cell[i] / np.linalg.norm(cell[i]) * float(
            (cart @ (cell[i] / np.linalg.norm(cell[i]))).min())
    return {"cell": cell, "frac": None, "cart": cart, "numbers": numbers, "pbc": pbc,
            "origin": origin}


def read_out_frames(path: Path) -> Dict[str, Any]:
    """Every geometry block of a CRYSTAL output, with the SCF energy of each."""
    from seekpath_interface import _geometry_blocks

    text = Path(path).read_text(errors="replace")
    lines = text.splitlines()
    dim = _out_dimensionality(lines)
    blocks = _geometry_blocks(lines)
    frames = []
    for n, block in enumerate(blocks):
        stop = blocks[n + 1]["line"] if n + 1 < len(blocks) else len(lines)
        energy = None
        for j in range(block["line"], stop):
            m = _ENERGY.search(lines[j])
            if m:
                try:
                    energy = float(m.group(1))
                except ValueError:
                    pass
                break
        s = _block_structure(block, dim)
        if s is None:
            continue
        label = ("final" if block["final"] else
                 "OPT point" if block["opt_point"] else "initial")
        frames.append({"structure": s, "energy_au": energy, "label": label,
                       "line": block["line"] + 1})
    if not frames:
        # a molecule: no PRIMITIVE CELL block, its Angstrom tables instead (only whole ones)
        mol = [b for b in molecule_blocks(lines) if b["complete"]]
        if mol:
            dim = 0
        for n, block in enumerate(mol):
            stop = mol[n + 1]["line"] if n + 1 < len(mol) else len(lines)
            energy = None
            for j in range(block["line"], stop):
                m = _ENERGY.search(lines[j])
                if m:
                    try:
                        energy = float(m.group(1))
                    except ValueError:
                        pass
                    break
            s = _block_structure(block, 0)
            if s is not None:
                frames.append({"structure": s, "energy_au": energy, "label": "final" if block["final"] else "initial",
                               "line": block["line"] + 1})
    status = None
    for line in reversed(lines):
        if "OPT END" in line:
            status = line.strip()
            break
    return {"frames": frames, "dimensionality": _DIM_NAMES[dim], "opt_status": status,
            "is_opt": any(f["label"] != "initial" for f in frames)}


def frame_payload(frame: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
    s = frame["structure"]
    return _payload(s["cell"], s["frac"], s["cart"], s["numbers"], s["pbc"], source,
                    {"energy_au": frame["energy_au"], "frame_label": frame["label"],
                     "origin": np.asarray(s.get("origin", np.zeros(3))).tolist()})


def read_out(path: Path, frame: int = -1) -> Dict[str, Any]:
    data = read_out_frames(path)
    if not data["frames"]:
        raise ValueError("No geometry in this output: CRYSTAL stopped before printing one (see the end of the output)")
    f = data["frames"][frame]
    out = frame_payload(f, {"kind": "out", "path": str(path)})
    out["n_frames"] = len(data["frames"])
    out["opt_status"] = data["opt_status"]
    return out


class UnreadableStructure(ValueError):
    """A parser stopped on the file (a missing record, a bad number): said in
    words, as a ValueError, so the server answers 400 with the reason instead
    of a parser's bare KeyError text."""


def _unreadable(path: Path, exc: Exception) -> UnreadableStructure:
    detail = str(exc).strip("'\"") or type(exc).__name__
    if isinstance(exc, KeyError):
        detail = f"no {detail} in it"
    return UnreadableStructure(f"cannot read the structure in {Path(path).name}: {detail}")


def read_structure(path: Path, frame: int = -1) -> Dict[str, Any]:
    """The structure in a file, as the payload the viewer draws."""
    try:
        return _read_structure(path, frame)
    except (KeyError, TypeError, IndexError, AttributeError) as exc:
        raise _unreadable(path, exc) from exc


def _read_structure(path: Path, frame: int = -1) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".cif":
        return read_cif(path)
    if suffix == ".d12":
        return read_d12(path)
    if suffix in (".out", ".o", ".log"):
        return read_out(path, frame)
    from .filekind import kind_by_name
    kind = kind_by_name(path.name)
    if kind in ("xyz", "poscar"):
        with open(path, "rb") as fh:
            fmt = sniff(path.name, fh.read(4096)) or ("poscar" if kind == "poscar" else "xyz")
        return read_any(path, fmt, box=False)
    if kind in ASE_ONLY:
        return read_any(path, kind, box=False)
    raise ValueError(f"not a structure file: {path.name}")


# molecular and other formats ase reads (plan E10): explorer kind -> ase format
ASE_ONLY = {"mol": "mol", "pdb": "proteindatabank", "gen": "gen", "fort34": "crystal"}


# --------------------------------------------------------------------------
# before / after comparison (VESTA-style)
# --------------------------------------------------------------------------

def _covalent(numbers):
    from ase.data import covalent_radii
    return np.array([covalent_radii[n] for n in numbers])


# H, He, B, C, N, O, F, Ne, Si, P, S, Cl, Ar, Ge, As, Se, Br, Kr, Sb, Te, I,
# Xe, At, Rn: everything else counts as a metal for drawing bonds.
_NONMETALS = frozenset({1, 2, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18, 32, 33, 34,
                        35, 36, 51, 52, 53, 54, 85, 86})


def _is_metal(numbers):
    """Metal-metal pairs get no bond, as VESTA draws ionic solids: covalent
    radii alone would join the cations of MgO, NaCl or LiF (Mg-Mg 2.98 A
    is inside 1.2 x 2 x 1.41 A)."""
    return np.array([int(n) not in _NONMETALS for n in numbers], bool)


def _neighbor_pairs(cell, cart, numbers, pbc, scale=BOND_SCALE, max_pairs=4000):
    """(i, j, image) with |r_j + image.cell - r_i| < scale * (R_i + R_j)."""
    cell = np.asarray(cell)
    rad = _covalent(numbers)
    metal = _is_metal(numbers)
    both_metal = metal[:, None] & metal[None, :]
    rng = [(-1, 0, 1) if p else (0,) for p in pbc]
    pairs = []
    for img in product(*rng):
        shift = np.asarray(img) @ cell
        d = np.linalg.norm(cart[None, :, :] + shift - cart[:, None, :], axis=2)
        cut = scale * (rad[:, None] + rad[None, :])
        ii, jj = np.nonzero((d < cut) & (d > 0.1) & ~both_metal)
        for i, j in zip(ii, jj):
            if (i, j, img) < (j, i, tuple(-x for x in img)):
                pairs.append((int(i), int(j), img))
                if len(pairs) >= max_pairs:
                    return pairs
    return pairs


def _kabsch(x0, x1):
    """``x1`` moved onto ``x0``: centroids together, then the proper rotation
    that fits them best (Kabsch)."""
    m0, m1 = x0.mean(axis=0), x1.mean(axis=0)
    a, b = x0 - m0, x1 - m1
    if len(a) >= 3:
        u, _, vt = np.linalg.svd(b.T @ a)
        d = 1.0 if np.linalg.det(u @ vt) >= 0 else -1.0
        b = b @ (u @ np.diag([1.0, 1.0, d]) @ vt)
    return b + m0


def align_molecule(x0, x1):
    """``x1`` in ``x0``'s frame without its rigid translation and rotation. The
    fit is done again without the atoms that clearly moved, so one moved atom
    does not spread a little of its move over all the others."""
    x0, x1 = np.asarray(x0, float), np.asarray(x1, float)
    if not len(x0):
        return x1
    y = _kabsch(x0, x1)
    d = np.linalg.norm(y - x0, axis=1)
    keep = d <= max(0.05, 3.0 * float(np.median(d)))
    if 3 <= keep.sum() < len(d):
        m0, m1 = x0[keep].mean(axis=0), x1[keep].mean(axis=0)
        a, b = x0[keep] - m0, x1[keep] - m1
        u, _, vt = np.linalg.svd(b.T @ a)
        sgn = 1.0 if np.linalg.det(u @ vt) >= 0 else -1.0
        y = (x1 - m1) @ (u @ np.diag([1.0, 1.0, sgn]) @ vt) + m0
    return y


def _nearest_shared_images(before, after):
    """``after``'s Cartesian positions, each atom moved by whole lattice
    vectors along the axes periodic in BOTH ends to its image nearest its
    ``before`` position (a slab against its own 3D CIF: an atom wrapped by +a
    in one of them is not a displacement). The common shift of the whole
    structure along each such axis (a different cell origin) is kept, so only
    a per-atom wrap is undone."""
    x1 = np.asarray(after["cart"], float)
    shared = [i for i in range(3) if before.get("pbc") and after.get("pbc")
              and bool(before["pbc"][i]) and bool(after["pbc"][i])]
    cell = after.get("cell")
    if not shared or not len(x1) or cell is None or len(cell) != 3:
        return x1
    c1 = np.asarray(cell, float)
    if not all(np.linalg.norm(c1[i]) > 1e-8 for i in shared):
        return x1
    d = (x1 - np.asarray(before["cart"], float)) @ np.linalg.pinv(c1)
    n = np.zeros_like(d)
    for i in shared:
        ang = 2 * np.pi * d[:, i]
        common = np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()) / (2 * np.pi)
        n[:, i] = np.round(d[:, i] - common)
    return x1 - n @ c1


def _compare_molecule(before, after) -> Dict[str, Any]:
    """compare() when either end is a molecule (no periodic direction) or the
    two differ in periodicity (a CIF's box against a MOLECULE output's
    bounding box): Cartesian displacements once the rigid translation and
    rotation are taken out; there is no cell to compare. Along the axes
    periodic in both ends (a slab against its 3D CIF) each atom is first
    taken at its nearest periodic image."""
    x0 = np.asarray(before["cart"], float)
    x1 = align_molecule(x0, _nearest_shared_images(before, after))
    disp = x1 - x0
    dist = np.linalg.norm(disp, axis=1)
    sym = _symbols()
    numbers = [int(n) for n in before["numbers"]]
    bonds = {}
    for i, j, _img in _neighbor_pairs(np.eye(3), x0, numbers, (False, False, False)):
        d0, d1 = float(np.linalg.norm(x0[j] - x0[i])), float(np.linalg.norm(x1[j] - x1[i]))
        a, b = sorted((sym[numbers[i]], sym[numbers[j]]))
        row = bonds.setdefault((a, b, round(d0, 3), round(d1, 3)), {"pair": f"{a}-{b}", "before": d0, "after": d1,
                                                                    "delta": d1 - d0, "count": 0, "example": [i + 1, j + 1]})
        row["count"] += 1
    return {
        "displacements": [{"atom": k + 1, "symbol": sym[numbers[k]], "vector": disp[k].tolist(), "length": float(dist[k])}
                          for k in range(len(numbers))],
        "before_cart": x0.tolist(),
        "after_unwrapped_cart": x1.tolist(),
        "max_displacement": float(dist.max()) if len(dist) else 0.0,
        "rms_displacement": float(np.sqrt((dist ** 2).mean())) if len(dist) else 0.0,
        "cell": [],
        "bonds": sorted(bonds.values(), key=lambda r: (r["pair"], r["before"])),
        "aligned": "molecule",
    }


def compare(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    """Displacements, cell-parameter and bond-length changes between two
    structures with the same atoms in the same order.

    Displacements are the change of fractional coordinates expressed in the
    final cell (non-periodic directions: Cartesian), so a cell change alone
    displaces nothing ("internal": the relaxation relative to the cell).
    ``displacements_cart`` are r_final - r_initial in Angstrom with the cell
    change included and the mean shift removed (see below); the GUI shows
    them by default when ``cell_changed``. A molecule, or two ends that
    differ in periodicity, is compared in Cartesian coordinates with its rigid
    translation and rotation taken out (_compare_molecule, which has no cell,
    so no ``displacements_cart`` and no "relative to the cell" choice).

    ``before``/``after`` are the internal frame structures (cell, frac or
    None, cart, numbers, pbc). Each atom of ``after`` is taken at the periodic
    image nearest its ``before`` position, so a wrap across the cell is not a
    displacement.
    """
    if list(before["numbers"]) != list(after["numbers"]):
        raise ValueError("the two structures do not hold the same atoms in the same order")
    pbc = before["pbc"]
    if not any(pbc) or [bool(v) for v in pbc] != [bool(v) for v in after["pbc"]]:
        return _compare_molecule(before, after)
    c0, c1 = np.asarray(before["cell"]), np.asarray(after["cell"])
    x0, x1 = np.asarray(before["cart"]), np.asarray(after["cart"])
    # fractional along periodic axes, Cartesian elsewhere
    f0 = np.linalg.solve(c0.T, x0.T).T
    f1 = np.linalg.solve(c1.T, x1.T).T
    # draw inside the cell: wrap the initial atoms into [0, 1) along the
    # periodic axes and move each final atom by the same lattice vector
    shift = np.zeros_like(f1)
    for i in range(3):
        if pbc[i]:
            home = np.floor(f0[:, i] + 1e-9)
            f0[:, i] -= home
            shift[:, i] = np.round(f1[:, i] - home - f0[:, i]) + home
    x0 = f0 @ c0
    f1u = f1 - shift
    x1u = f1u @ c1
    # The atomic relaxation, measured in the final cell: origin- and
    # wrap-independent. (A Cartesian difference x1 - x0 would mix in the cell
    # strain, which grows with the distance from the cell origin; the strain
    # is reported separately in the cell table.)
    disp = (f1u - f0) @ c1
    dist = np.linalg.norm(disp, axis=1)
    # The Cartesian displacement, cell change included: each final atom at the
    # image nearest its initial position (x1u), minus the mean shift, so the
    # answer does not depend on where the cell origin sits (a cell that only
    # shrinks moves the atoms toward their common center, not toward the
    # origin). Atoms fixed in fractional terms by symmetry (graphite's layers
    # along c) move here and not above.
    dcart = x1u - x0
    if len(dcart):
        dcart = dcart - dcart.mean(axis=0)
    dist_cart = np.linalg.norm(dcart, axis=1)
    sym = _symbols()
    numbers = [int(n) for n in before["numbers"]]

    l0, l1 = _lattice(c0), _lattice(c1)
    keys = [k for k, p in (("a", pbc[0]), ("b", pbc[1]), ("c", pbc[2])) if p]
    keys += {3: ["alpha", "beta", "gamma", "volume"], 2: ["gamma"]}.get(sum(pbc), [])
    cell_rows = []
    for k in keys:
        v0, v1 = l0[k], l1[k]
        cell_rows.append({"param": k, "before": v0, "after": v1, "delta": v1 - v0,
                          "percent": (v1 - v0) / v0 * 100 if v0 else None})

    bonds = {}
    for i, j, img in _neighbor_pairs(c0, x0, numbers, pbc):
        d0 = float(np.linalg.norm(x0[j] + np.asarray(img) @ c0 - x0[i]))
        d1 = float(np.linalg.norm(x1u[j] + np.asarray(img) @ c1 - x1u[i]))
        a, b = sorted((sym[numbers[i]], sym[numbers[j]]))
        key = (a, b, round(d0, 3), round(d1, 3))
        row = bonds.setdefault(key, {"pair": f"{a}-{b}", "before": d0, "after": d1,
                                     "delta": d1 - d0, "count": 0, "example": [i + 1, j + 1]})
        row["count"] += 1
    bond_rows = sorted(bonds.values(), key=lambda r: (r["pair"], r["before"]))

    return {
        "displacements": [{"atom": k + 1, "symbol": sym[numbers[k]],
                           "vector": disp[k].tolist(), "length": float(dist[k])}
                          for k in range(len(numbers))],
        "before_cart": x0.tolist(),
        "after_unwrapped_cart": x1u.tolist(),
        "max_displacement": float(dist.max()) if len(dist) else 0.0,
        "rms_displacement": float(np.sqrt((dist ** 2).mean())) if len(dist) else 0.0,
        "displacements_cart": [{"atom": k + 1, "symbol": sym[numbers[k]],
                                "vector": dcart[k].tolist(), "length": float(dist_cart[k])}
                               for k in range(len(numbers))],
        "max_displacement_cart": float(dist_cart.max()) if len(dist_cart) else 0.0,
        "rms_displacement_cart": float(np.sqrt((dist_cart ** 2).mean())) if len(dist_cart) else 0.0,
        # the cell changed (a length or the volume by more than 1e-6 relative, an angle by 1e-6 degrees)
        "cell_changed": any(abs(r["delta"]) > (1e-6 * abs(r["before"]) if r["param"] in ("a", "b", "c", "volume") else 1e-6)
                            for r in cell_rows),
        "cell": cell_rows,
        "bonds": bond_rows,
    }


def trajectory(path: Path) -> Dict[str, Any]:
    """Frames of an output plus the initial-vs-last comparison."""
    data = read_out_frames(path)
    frames = data["frames"]
    if not frames:
        raise ValueError("No geometry in this output: CRYSTAL stopped before printing one (see the end of the output)")
    source = {"kind": "out", "path": str(path)}
    payloads = [frame_payload(f, source) for f in frames]
    result = {
        "path": str(path),
        "opt_status": data["opt_status"],
        "is_opt": data["is_opt"],
        "dimensionality": data["dimensionality"],
        "frames": payloads,
        "energies_au": [f["energy_au"] for f in frames],
        "labels": [f["label"] for f in frames],
    }
    if len(frames) > 1:
        result["comparison"] = compare(frames[0]["structure"], frames[-1]["structure"])
    return result


class MatchError(ValueError):
    """Two structures that cannot be compared atom by atom (shown with the reason)."""


# position matching builds one distance matrix per element: past this many atoms of one
# element (a 3000 x 3000 x 3 float64 difference is 216 MB) the order is taken as it is
MATCH_MAX_PER_ELEMENT = 3000


def match_atoms(before: Dict[str, Any], after: Dict[str, Any]) -> Tuple[List[int], float, str]:
    """The order of ``after``'s atoms that pairs each with one of ``before``'s.

    Two files can list the same species in the same sequence and still put
    the atoms in a different order (two CIF writers, a d12 against its run),
    so the atoms are paired by position whenever the counts allow: species by
    species, each ``after`` atom to the nearest ``before`` atom of its element
    under the minimum image, measured in fractional coordinates (so two files
    in different Cartesian orientations still pair) with ``after``'s metric
    (the Hungarian assignment when scipy is here, else greedy). When that
    pairing is the identity, or an element has more than
    MATCH_MAX_PER_ELEMENT atoms, the order is by index.
    Returns (order, max mismatch in Å, "index" | "position")."""
    n0, n1 = list(before["numbers"]), list(after["numbers"])
    if len(n0) != len(n1):
        raise MatchError(f"{len(n0)} atoms vs {len(n1)}: compare a supercell with Build › Supercell first")
    if sorted(n0) != sorted(n1):
        sym = _symbols()
        from collections import Counter
        a, b = Counter((sym[z] if 0 < z < len(sym) else str(z)) for z in n0), Counter((sym[z] if 0 < z < len(sym) else str(z)) for z in n1)
        raise MatchError(f"the two structures need the same atoms: {' '.join(f'{k}{v}' for k, v in sorted(a.items()))} vs "
                         f"{' '.join(f'{k}{v}' for k, v in sorted(b.items()))}")
    from collections import Counter
    if max(Counter(n0).values(), default=0) > MATCH_MAX_PER_ELEMENT:
        if n0 == n1:
            return list(range(len(n0))), 0.0, "index"
        raise MatchError(f"too many atoms of one element to match by position (over {MATCH_MAX_PER_ELEMENT}); "
                         f"list the atoms in the same order in both files")
    c1 = np.asarray(after["cell"], float)
    periodic = all(after["pbc"]) and all(before["pbc"]) and abs(np.linalg.det(c1)) > 1e-8
    if periodic:
        x0, x1 = np.asarray(before["frac"], float), np.asarray(after["frac"], float)
    else:
        # a molecule (or a box against a bounding box): positions about each one's center
        x0, x1 = np.asarray(before["cart"], float), np.asarray(after["cart"], float)
        if len(x0):
            x0, x1 = x0 - x0.mean(axis=0), x1 - x1.mean(axis=0)
    order = [-1] * len(n0)
    worst = 0.0
    same_seq = n0 == n1
    cost_found = cost_index = 0.0
    for z in sorted(set(n0)):
        i0 = [i for i, v in enumerate(n0) if v == z]
        i1 = [i for i, v in enumerate(n1) if v == z]
        d = x1[i1][None, :, :] - x0[i0][:, None, :]          # before x after
        if periodic:
            d -= np.round(d)
            d = d @ c1
        dist = np.linalg.norm(d, axis=2)
        del d
        try:
            from scipy.optimize import linear_sum_assignment
            r, c = linear_sum_assignment(dist)
            pairs = list(zip(r.tolist(), c.tolist()))
        except ImportError:                                 # greedy: closest pairs first
            pairs, used_r, used_c = [], set(), set()
            for flat in np.argsort(dist, axis=None):
                rr, cc = divmod(int(flat), dist.shape[1])
                if rr in used_r or cc in used_c:
                    continue
                pairs.append((rr, cc))
                used_r.add(rr)
                used_c.add(cc)
        for rr, cc in pairs:
            order[i0[rr]] = i1[cc]
            worst = max(worst, float(dist[rr, cc]))
            cost_found += float(dist[rr, cc])
        if same_seq:
            cost_index += float(np.trace(dist))
    # the files' own order wins whenever it pairs the atoms as well as any other order does
    # (a relaxation that moves atoms far stays by index rather than being re-paired)
    if order == list(range(len(n0))) or (same_seq and cost_index <= cost_found + 1e-6 * len(n0)):
        idx = list(range(len(n0)))
        if same_seq and order != idx:
            worst = _index_worst(x0, x1, c1 if periodic else None)
        return idx, worst, "index"
    return order, worst, "position"


def _index_worst(x0, x1, cell) -> float:
    d = x1 - x0
    if cell is not None:
        d -= np.round(d)
        d = d @ cell
    return float(np.linalg.norm(d, axis=1).max()) if len(d) else 0.0


def compare_payloads(a: Dict[str, Any], b: Dict[str, Any], match: str = "auto") -> Dict[str, Any]:
    """compare() for two read_structure() payloads (e.g. two CIFs); the atoms of
    ``b`` are matched to ``a``'s by index, or by position when their order differs."""
    def internal(p):
        return {"cell": p["cell"], "frac": p["frac"], "cart": p["cart"],
                "numbers": p["numbers"], "pbc": p["pbc"]}
    ia, ib = internal(a), internal(b)
    if match == "index":
        order, worst, how = list(range(len(ib["numbers"]))), 0.0, "index"
    else:
        order, worst, how = match_atoms(ia, ib)
    if how == "position":
        ib = {"cell": ib["cell"], "frac": [ib["frac"][k] for k in order], "cart": [ib["cart"][k] for k in order],
              "numbers": [ib["numbers"][k] for k in order], "pbc": ib["pbc"]}
    out = compare(ia, ib)
    out["match"] = how
    out["max_mismatch"] = worst
    out["order"] = order if how == "position" else None
    return out


def same_crystal(a: Dict[str, Any], b: Dict[str, Any], tol: float = 0.01) -> bool:
    """True when ``b`` can be the primitive cell of ``a`` (or the same cell): the same
    composition ratio and volume per atom within ``tol``, and the same space group
    when spglib can tell (a centered .d12 against the cell CRYSTAL ran)."""
    from collections import Counter
    na, nb = Counter(a["numbers"]), Counter(b["numbers"])
    if not na or set(na) != set(nb) or len(a["numbers"]) % len(b["numbers"]):
        return False
    k = len(a["numbers"]) // len(b["numbers"])
    if any(na[z] != k * nb[z] for z in na):
        return False
    va, vb = abs(np.linalg.det(np.asarray(a["cell"], float))), abs(np.linalg.det(np.asarray(b["cell"], float)))
    if vb <= 0 or abs(va / vb - k) > tol * k:
        return False
    g = symmetry_change(a, b)
    return not (g["before"] and g["after"] and g["before"] != g["after"])


def symmetry_change(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """The space group of each end ("Fm-3m (225) → Fm-3m (225)"), when both are 3D and small enough."""
    out = {}
    for k, p in (("before", a), ("after", b)):
        y = p.get("symmetry") if isinstance(p.get("symmetry"), dict) else None
        if not y or not y.get("available"):
            if all(p["pbc"]) and len(p["numbers"]) <= 2000:
                try:
                    y = symmetry(p["cell"], p["frac"], p["numbers"], p["pbc"])
                except Exception:
                    y = None
        out[k] = f"{y['international']} ({y['number']})" if y and y.get("available") else None
    return out


# --------------------------------------------------------------------------
# dropped and pasted files (Phase 4, plan B8/B9)
# --------------------------------------------------------------------------

FORMAT_NAMES = {"cif": "CIF", "poscar": "POSCAR", "xyz": "XYZ", "extxyz": "extXYZ", "d12": "d12", "out": "CRYSTAL output"}
_FLOAT = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")


def _floats(line: str, n: int) -> bool:
    parts = line.split()
    return len(parts) >= n and all(_FLOAT.match(p) for p in parts[:n])


def sniff(name: str, head: bytes) -> Optional[str]:
    """The format of a dropped or pasted file: the extension when it has a
    known one, else what the first lines look like. None when neither tells."""
    base = Path(str(name)).name
    low = base.lower()
    for ext, fmt in ((".cif", "cif"), (".d12", "d12"), (".out", "out"), (".log", "out"), (".extxyz", "extxyz"),
                     (".xyz", "xyz"), (".vasp", "poscar"), (".poscar", "poscar")):
        if low.endswith(ext):
            return fmt
    if re.match(r"^(POSCAR|CONTCAR)([._-]|$)", base):
        return "poscar"
    text = head.decode("utf-8", "replace")
    lines = [ln.rstrip() for ln in text.splitlines()]
    if re.search(r"^\s*data_", text, re.M) or "_cell_length_a" in text:
        return "cif"
    if len(lines) >= 5 and _floats(lines[1], 1) and all(_floats(lines[k], 3) for k in (2, 3, 4)):
        return "poscar"
    if lines and lines[0].strip().isdigit():
        return "extxyz" if len(lines) > 1 and 'Lattice="' in lines[1] else "xyz"
    if re.search(r"^\s*(CRYSTAL|SLAB|POLYMER|MOLECULE)\s*$", text, re.M) and re.search(r"^\s*END", text, re.M):
        return "d12"
    if "CRYSTAL" in text and ("EEEEEEEEEE" in text or "TOTAL ENERGY" in text or "PROCESS" in text):
        return "out"
    return None


def read_any(path: Path, fmt: str, box: bool = True) -> Dict[str, Any]:
    """A dropped or pasted file read with the format sniff() found. A plain XYZ
    has no cell: with ``box`` it gets one with 10 Å of vacuum between images,
    else it is a molecule (view-only)."""
    try:
        return _read_any(path, fmt, box)
    except (KeyError, TypeError, IndexError, AttributeError) as exc:
        raise _unreadable(path, exc) from exc


def _read_any(path: Path, fmt: str, box: bool = True) -> Dict[str, Any]:
    import warnings
    path = Path(path)
    if fmt == "cif":
        return read_cif(path)
    if fmt == "d12":
        return read_d12(path)
    if fmt == "out":
        return read_out(path, -1)
    if fmt not in ("poscar", "xyz", "extxyz") and fmt not in ASE_ONLY:
        raise ValueError(f"no reader for {fmt}")
    import ase.io
    ase_fmt = {"poscar": "vasp", "xyz": "xyz", "extxyz": "extxyz"}.get(fmt) or ASE_ONLY[fmt]
    if fmt == "mol" and path.suffix.lower() == ".sdf":
        ase_fmt = "sdf"
    source = str(path)
    if fmt == "poscar":
        # "BI S" on the species line: ase looks it up exactly (KeyError).
        import io
        from mace.utils.cif_symbols import normalize_poscar_species
        text = path.read_text(errors="replace")
        fixed = normalize_poscar_species(text)
        if fixed != text:
            source = io.StringIO(fixed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        atoms = ase.io.read(source, format=ase_fmt)
    numbers = [int(n) for n in atoms.numbers]
    has_cell = atoms.cell.rank == 3 and abs(atoms.cell.volume) > 1e-6
    if not has_cell and box:
        atoms.center(vacuum=5.0)               # 5 Å each side: 10 Å between periodic images
        atoms.pbc = True
        has_cell = True
    if has_cell:
        return _payload(atoms.cell.array, atoms.get_scaled_positions(wrap=False), atoms.positions, numbers,
                        (True, True, True), {"kind": fmt, "path": str(path)})
    # no cell: a molecule, drawn in the box that holds it
    span = np.ptp(atoms.positions, axis=0) if len(atoms) else np.ones(3)
    cell = np.diag(np.maximum(span, 0.5))
    cart = atoms.positions - atoms.positions.min(0)
    out = _payload(cell, None, cart, numbers, (False, False, False), {"kind": fmt, "path": str(path)},
                   symmetry_data={"available": False, "note": "No cell: a molecule file has none"})
    out["no_cell"] = True
    return out


# --------------------------------------------------------------------------
# dimensionality (plan E10): a hint from the vacuum, and the cell CRYSTAL's
# SLAB / POLYMER / MOLECULE decks need, written by the CLI's own writer
# --------------------------------------------------------------------------
DIM_GAP = 8.0          # Å of vacuum that marks a non-periodic direction (Settings › structure.vacuumGap)
_AXIS = "abc"


def _vacuum_along(cell: np.ndarray, frac: np.ndarray, i: int) -> float:
    """The widest empty slab between atom layers along lattice vector i (Å, measured
    perpendicular to the other two vectors): the vacuum that direction holds."""
    if len(frac) == 0:
        return 0.0
    f = np.sort(np.mod(frac[:, i], 1.0))
    gaps = np.diff(np.concatenate([f, [f[0] + 1.0]]))
    j, k = [x for x in range(3) if x != i]
    n = np.cross(cell[j], cell[k])
    height = abs(float(np.dot(cell[i], n))) / max(1e-12, float(np.linalg.norm(n)))
    return float(gaps.max() * height)


def dimensionality_hint(cell, cart, pbc=(True, True, True), gap: float = DIM_GAP) -> Dict[str, Any]:
    """What a structure looks like: {dim, vacuum_axes, vacuum (Å per axis), oriented}.
    A molecule in a box has vacuum along all three cell vectors, a slab along exactly
    one, a polymer along two; a molecular crystal (no vacuum) stays CRYSTAL. Only a
    suggestion: the file never changes."""
    cart = np.asarray(cart, float).reshape(-1, 3)
    if not all(pbc):
        dim = int(sum(bool(p) for p in pbc))
        return {"dim": _DIM_NAMES[dim], "vacuum_axes": [_AXIS[i] for i in range(3) if not pbc[i]], "vacuum": {},
                "oriented": True, "from": "pbc"}
    cell = np.asarray(cell, float)
    frac = np.linalg.solve(cell.T, cart.T).T
    vac = {_AXIS[i]: round(_vacuum_along(cell, frac, i), 2) for i in range(3)}
    axes = [a for a in _AXIS if vac[a] >= gap]
    dim = {0: "CRYSTAL", 1: "SLAB", 2: "POLYMER", 3: "MOLECULE"}[len(axes)]
    # CRYSTAL's SLAB has its vacuum along c, a POLYMER its chain along a
    oriented = (dim == "SLAB" and axes == ["c"]) or (dim == "POLYMER" and axes == ["b", "c"]) or dim in ("CRYSTAL", "MOLECULE")
    return {"dim": dim, "vacuum_axes": axes, "vacuum": vac, "oriented": bool(oriented), "from": "vacuum"}


class OrientError(ValueError):
    pass


def _rotation_to(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The rotation taking a to x and the a-b plane to x-y (b in the +y half)."""
    e1 = a / np.linalg.norm(a)
    n = np.cross(a, b)
    if np.linalg.norm(n) < 1e-9:
        raise OrientError("the two periodic vectors are parallel")
    e3 = n / np.linalg.norm(n)
    e2 = np.cross(e3, e1)
    return np.vstack([e1, e2, e3])          # rows: new x, y, z in old coordinates


def _unwrap_through_vacuum(cell: np.ndarray, cart: np.ndarray, axes) -> np.ndarray:
    """The atoms of a periodic cell made whole along ``axes`` (lattice indices): along
    each, the widest empty gap between atoms becomes the cell boundary, so a molecule
    (or a chain's cross-section) stored wrapped across a face is in one piece again.
    Lattice translations only: every periodic image is the same structure."""
    if len(cart) == 0:
        return cart
    frac = np.linalg.solve(cell.T, cart.T).T
    for i in axes:
        f = np.mod(frac[:, i], 1.0)
        order = np.sort(f)
        gaps = np.diff(np.concatenate([order, [order[0] + 1.0]]))
        k = int(np.argmax(gaps))
        cut = order[k]
        # atoms above the widest gap's lower edge move down by one period: the gap is now at the face
        frac[:, i] = np.where(f > cut + 1e-12, f - 1.0, f) if k < len(order) - 1 else f
    return frac @ cell


def standardize_for_dim(numbers, cart, cell, pbc, dim: str, gap: float = DIM_GAP) -> Dict[str, Any]:
    """The cell and positions the CLI's coordinate writer handles for a SLAB,
    POLYMER or MOLECULE deck, by a rigid motion only (every distance kept):

    * MOLECULE: the Cartesian positions centered in an orthogonal cube of side
      (largest extent + 2 x gap), so fractional x a = Cartesian x;
    * SLAB: the two periodic vectors become a, b with a along x and the plane in
      x-y, and c is rebuilt perpendicular to it (thickness + gap), so CRYSTAL's
      Cartesian z is the true one;
    * POLYMER: the periodic vector becomes a along x, b and c orthogonal to it and
      to each other (extent + gap).

    The periodic vectors are the ones ``pbc`` marks, else (a 3D cell) the ones
    without vacuum; a cell whose vacuum is not along one (SLAB) or two (POLYMER)
    of its vectors is refused. Returns {cell, cart, moved, note}."""
    numbers = [int(n) for n in numbers]
    cart = np.asarray(cart, float).reshape(-1, 3)
    cell = np.asarray(cell, float)
    dim = str(dim).upper()
    if dim == "CRYSTAL":
        return {"cell": cell, "cart": cart, "moved": False, "note": ""}
    if dim == "MOLECULE":
        if all(pbc) and abs(np.linalg.det(cell)) > 1e-9:
            # a molecule in a periodic box (a CIF) may be stored wrapped across the box's faces
            cart = _unwrap_through_vacuum(cell, cart, range(3))
        span = np.ptp(cart, axis=0) if len(cart) else np.zeros(3)
        side = float(max(span.max(), 1.0) + 2 * gap)
        box = np.eye(3) * side
        center = (cart.max(axis=0) + cart.min(axis=0)) / 2 if len(cart) else np.zeros(3)
        new = cart - center + side / 2
        moved = not (np.allclose(cell, box, atol=1e-6) and np.allclose(new, cart, atol=1e-6))
        return {"cell": box, "cart": new, "moved": bool(moved), "note": f"Molecule centered in a {side:.1f} Å cubic box"}
    want = 2 if dim == "SLAB" else 1
    per = [i for i in range(3) if pbc[i]] if not all(pbc) else None
    if per is None:
        hint = dimensionality_hint(cell, cart, pbc, gap)
        vac = [_AXIS.index(a) for a in hint["vacuum_axes"]]
        if len(vac) != 3 - want:
            if dim == "SLAB":
                raise OrientError("MACE cannot orient this cell for a SLAB deck: the vacuum is not along one cell vector. "
                                  "Rotate the cell so the vacuum is along c, or cut the slab with SLABCUT from the bulk.")
            raise OrientError("MACE cannot orient this cell for a POLYMER deck: the chain must run along one cell vector, "
                              "with vacuum along the other two.")
        per = [i for i in range(3) if i not in vac]
    if len(per) != want:
        raise OrientError(f"a {dim} needs {want} periodic direction{'s' if want > 1 else ''}, this structure has {len(per)}")
    if dim == "SLAB":
        a, b = cell[per[0]], cell[per[1]]
        if np.dot(np.cross(a, b), cell[3 - per[0] - per[1]]) < 0 and all(pbc):
            a, b = b, a                    # keep the deck right-handed with c up
        R = _rotation_to(a, b)
        new = cart @ R.T
        A, B = R @ a, R @ b
        # one layer: unwrap along z through the vacuum, so the slab is in one piece
        z = new[:, 2]
        thick = float(np.ptp(z)) if len(z) else 0.0
        if all(pbc):
            old_c = cell[3 - per[0] - per[1]]
            h = float(np.dot(R @ old_c, [0, 0, 1]))
            if abs(h) > 1e-9 and thick > abs(h) - gap:
                # the layer straddles the cell boundary: move atoms by the old c to one side of the vacuum
                f = np.mod((z - z.min()) / abs(h), 1.0)
                order = np.sort(f)
                gaps = np.diff(np.concatenate([order, [order[0] + 1]]))
                cut = order[int(np.argmax(gaps))]
                shift = np.where(f > cut + 1e-9, -1.0, 0.0) * np.sign(h)
                new = new + np.outer(shift, R @ old_c)
                thick = float(np.ptp(new[:, 2]))
        c_len = thick + gap
        newcell = np.vstack([A, B, [0.0, 0.0, c_len]])
        new[:, 2] += -new[:, 2].min() + gap / 2
        return {"cell": newcell, "cart": new, "moved": True,
                "note": f"Oriented for CRYSTAL: vacuum along z, c = {c_len:.2f} Å perpendicular to the layer"}
    # POLYMER
    if all(pbc):
        # the chain may be stored wrapped across the faces of its vacuum directions
        cart = _unwrap_through_vacuum(cell, cart, [i for i in range(3) if i not in per])
    p = cell[per[0]]
    other = cell[(per[0] + 1) % 3]
    if np.linalg.norm(np.cross(p, other)) < 1e-9:
        other = np.array([0.0, 0.0, 1.0]) if abs(p[2]) < 0.9 * np.linalg.norm(p) else np.array([0.0, 1.0, 0.0])
    R = _rotation_to(p, other)
    new = cart @ R.T
    span = np.ptp(new[:, 1:], axis=0) if len(new) else np.zeros(2)
    b_len, c_len = float(span[0] + gap), float(span[1] + gap)
    new[:, 0] -= new[:, 0].min()              # the chain starts at x = 0 (a whole number of periods: a rigid shift)
    new[:, 1] += -new[:, 1].min() + gap / 2
    new[:, 2] += -new[:, 2].min() + gap / 2
    newcell = np.array([[np.linalg.norm(p), 0, 0], [0, b_len, 0], [0, 0, c_len]])
    return {"cell": newcell, "cart": new, "moved": True,
            "note": f"Oriented for CRYSTAL: the chain along x, b and c orthogonal ({b_len:.1f}, {c_len:.1f} Å)"}
