"""Parent geometry -> phonopy supercells (needs phonopy)."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

WARN_FOLDED = "FOLDED_BANDS"


def normalise_dim(dim: Sequence[int]) -> np.ndarray:
    """3 ints -> diag(3x3); 9 ints -> 3x3 (row-major, phonopy's DIM order)."""
    vals = [int(v) for v in dim]
    if len(vals) == 3:
        m = np.diag(vals)
    elif len(vals) == 9:
        m = np.array(vals).reshape(3, 3)
    else:
        raise ValueError("--dim takes 3 or 9 integers")
    if round(abs(np.linalg.det(m))) < 1:
        raise ValueError("--dim matrix is singular")
    return m.astype(int)


def dim_tag(m: np.ndarray) -> str:
    """Short tag for file names: '222' for diag(2,2,2), else the 9 ints."""
    if np.count_nonzero(m - np.diag(np.diag(m))) == 0:
        return "".join(str(int(v)) for v in np.diag(m))
    return "m" + "".join(str(int(v)) for v in m.flatten())


def read_parent_cell(out_path: Path):
    """(PhonopyAtoms, conv_numbers) of the parent's final geometry."""
    from phonopy.interface.crystal import read_crystal

    with contextlib.redirect_stdout(io.StringIO()):
        cell, conv = read_crystal(str(out_path))
    if conv is None:
        raise ValueError(f"{out_path}: phonopy could not read a geometry from this output")
    return cell, list(conv)


def make_phonopy(cell, dim: np.ndarray):
    from phonopy import Phonopy

    return Phonopy(cell, supercell_matrix=dim, primitive_matrix="P", calculator="crystal")


def generate(phonon, amplitude: Optional[float], pm: str) -> None:
    kw = {}
    if amplitude is not None:
        kw["distance"] = amplitude
    if pm == "yes":
        kw["is_plusminus"] = True
    elif pm == "no":
        kw["is_plusminus"] = False
    else:
        kw["is_plusminus"] = "auto"
    phonon.generate_displacements(**kw)


def supercell_conv_numbers(phonon, conv_numbers: Sequence[int]) -> List[int]:
    """CRYSTAL conventional atomic numbers (e.g. 282 for an ECP Pb) for every
    supercell atom, mapped through phonopy's own supercell-to-unit-cell map."""
    sc = phonon.supercell
    s2u = np.asarray(sc.s2u_map)
    u2u = sc.u2u_map
    out = [int(conv_numbers[u2u[int(s)]]) for s in s2u]
    nums = list(sc.numbers)
    bad = [i for i, (c, z) in enumerate(zip(out, nums)) if c % 100 != z]
    if bad:
        raise AssertionError(
            f"conventional atomic number does not match the element for supercell atom(s) "
            f"{[b + 1 for b in bad[:5]]}")
    return out


#: spglib space-group numbers of the rhombohedral (R) groups.
RHOMBOHEDRAL_GROUPS = frozenset({146, 148, 155, 160, 161, 166, 167})

CRY_SYM_CAVEAT = (
    "--cry-sym: symmetry operators of each displaced cell are written into its .ext "
    "(phonopy's CRY_SYM). phonopy warns that some space groups do not work this way "
    "(e.g. R-3m). Measured on CRYSTAL23 for diamond (Fd-3m): the same frequencies as "
    "P1 to 0.04 cm-1 at about a third of the core hours. collect still checks that "
    "the geometry CRYSTAL printed is each displaced cell.")


def cry_sym_warnings(cell) -> List[str]:
    """The CRY_SYM caveat, plus a stronger one for a rhombohedral parent."""
    out = [CRY_SYM_CAVEAT]
    try:
        import spglib
        ds = spglib.get_symmetry_dataset((cell.cell, cell.scaled_positions, cell.numbers),
                                         symprec=1e-5)
    except Exception:
        return out
    if ds is None:
        return out
    number = getattr(ds, "number", None)
    symbol = getattr(ds, "international", None)
    if number is None and isinstance(ds, dict):
        number, symbol = ds.get("number"), ds.get("international")
    if number in RHOMBOHEDRAL_GROUPS:
        out.append(f"the parent is {symbol} (No. {number}), a rhombohedral group like the "
                   "R-3m phonopy warns about: compare with a run without --cry-sym before "
                   "trusting these phonons")
    return out


def folded_warning(cell) -> Optional[Tuple[str, str]]:
    """Warn when spglib finds a smaller primitive cell than the parent's cell."""
    try:
        import spglib
        res = spglib.standardize_cell(
            (cell.cell, cell.scaled_positions, cell.numbers),
            to_primitive=True, no_idealize=True, symprec=1e-5)
    except Exception:
        return None
    if not res:
        return None
    n_prim = len(res[2])
    n = len(cell.numbers)
    if n_prim < n:
        return (WARN_FOLDED,
                f"the parent cell holds {n} atoms but a primitive cell has {n_prim} "
                "(P1 or SUPERCEL parent?): phonon bands are folded into the smaller "
                "Brillouin zone of the parent cell")
    return None
