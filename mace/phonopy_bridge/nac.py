"""Non-analytical correction from a CRYSTAL Gamma FREQCALC with INTENS + INTCPHF.

phonopy-crystal-born reads the dielectric tensor ("FR.(eV) COMP. ALPHA(Re,Im)
EPSILON(Re,Im)") and the Born charge tensors ("ATOMIC BORN CHARGE TENSOR") and
writes them for CRYSTAL's asymmetric-unit ("T") atoms only. phonopy's
parse_BORN then hands them to ITS independent atoms and spreads them by ITS
symmetry - so both programs must agree on the cell, its orientation (the
tensors are cartesian), the atom order and the asymmetric unit. Every one of
those is checked here before the BORN text is used.
"""

from __future__ import annotations

import contextlib
import io
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

from . import parent as P

POSITION_TOLERANCE = 1e-3     # Angstrom

_ASYM_ROW = re.compile(r"^\s*(\d+)\s+([TF])\s+(\d+)\s+([A-Za-z]{1,2})\s+\S+\s+\S+\s+\S+")


class NacError(ValueError):
    pass


def asymmetric_flags(out_text: str) -> List[str]:
    """'T'/'F' per atom from the LAST 'ATOMS IN THE ASYMMETRIC UNIT' list."""
    lines = out_text.splitlines()
    idx = [i for i, ln in enumerate(lines) if "ATOMS IN THE ASYMMETRIC UNIT" in ln]
    if not idx:
        return []
    flags = []
    j = idx[-1] + 1
    while j < len(lines) and (not _ASYM_ROW.match(lines[j])):
        j += 1
        if j > idx[-1] + 4:
            return []
    while j < len(lines):
        m = _ASYM_ROW.match(lines[j])
        if not m:
            break
        flags.append(m.group(2))
        j += 1
    return flags


def run_crystal_born(freq_out: Path) -> str:
    """phonopy-crystal-born, in-process, on ``freq_out``; its stdout text."""
    from phonopy.scripts import phonopy_crystal_born as pcb

    buf = io.StringIO()
    old = sys.argv
    sys.argv = ["phonopy-crystal-born", str(freq_out)]
    try:
        with contextlib.redirect_stdout(buf):
            pcb.run()
    except SystemExit:
        pass
    except Exception as exc:          # NameError etc. on an output it cannot read
        raise NacError(f"phonopy-crystal-born could not read {freq_out}: {exc!r}") from exc
    finally:
        sys.argv = old
    return buf.getvalue()


def _check_geometry(freq_out: Path, unitcell) -> None:
    from .structure import read_parent_cell

    cell, _ = read_parent_cell(freq_out)
    lat, ref = np.array(cell.cell), np.array(unitcell.cell)
    if np.abs(lat - ref).max() > POSITION_TOLERANCE:
        raise NacError(
            f"{freq_out.name}: cartesian lattice vectors differ from the phonopy unit cell "
            f"by {np.abs(lat - ref).max():.2e} A (Born and dielectric tensors are cartesian, "
            "so the cells must have the same orientation)")
    if list(cell.symbols) != list(unitcell.symbols):
        raise NacError(f"{freq_out.name}: atoms {list(cell.symbols)} do not match the phonopy "
                       f"unit cell {list(unitcell.symbols)} (count or order)")
    d = (np.array(cell.positions) - np.array(unitcell.positions)) @ np.linalg.inv(ref)
    d -= np.round(d)
    dist = np.linalg.norm(d @ ref, axis=1)
    if dist.max() > POSITION_TOLERANCE:
        raise NacError(f"{freq_out.name}: atom {int(np.argmax(dist)) + 1} is {dist.max():.2e} A "
                       "from its position in the phonopy unit cell")


def _check_method(freq_out: Path, man: Dict) -> None:
    d12 = freq_out.with_suffix(".d12")
    if not d12.is_file():
        raise NacError(f"{freq_out.name}: no {d12.name} next to it, so its method and basis "
                       "cannot be compared with the phonopy parent's")
    want = man.get("method_lines")
    got = P.method_lines(P.split_parent_deck(d12.read_text(errors="replace")))
    if want is not None and got != want:
        raise NacError(f"{d12.name}: functional/basis records differ from the phonopy parent's "
                       f"({sorted(k for k in set(got) | set(want) if got.get(k) != want.get(k))})")


def nac_params_from_freq(freq_out: Path, phonon, man: Dict):
    """(nac_params, BORN text) after every consistency check passed."""
    from phonopy.file_IO import parse_BORN_from_strings
    from phonopy.structure.symmetry import Symmetry

    freq_out = Path(freq_out)
    text = freq_out.read_text(errors="replace")
    if "ATOMIC BORN CHARGE TENSOR" not in text or "EPSILON" not in text:
        raise NacError(f"{freq_out.name}: no Born charges / dielectric tensor - run a Gamma "
                       "FREQCALC with INTENS and INTCPHF")
    d12 = freq_out.with_suffix(".d12")
    d12_text = d12.read_text(errors="replace") if d12.is_file() else ""
    if re.search(r"^\s*(SCELPHONO|DISPERSI)\s*$", text + "\n" + d12_text, re.M):
        raise NacError(f"{freq_out.name}: a SCELPHONO/DISPERSI run is a supercell; use a plain "
                       "Gamma FREQCALC with INTENS + INTCPHF")
    if not P.normal_termination(text):
        raise NacError(f"{freq_out.name}: did not terminate normally")

    unitcell = phonon.unitcell
    _check_geometry(freq_out, unitcell)

    flags = asymmetric_flags(text)
    t_atoms = [i for i, f in enumerate(flags) if f == "T"]
    sym = Symmetry(phonon.primitive, symprec=1e-5)
    indep = [int(i) for i in sym.get_independent_atoms()]
    if t_atoms != indep:
        raise NacError(
            f"{freq_out.name}: CRYSTAL's asymmetric unit (atoms {[i + 1 for i in t_atoms]}) "
            f"is not phonopy's independent atoms ({[i + 1 for i in indep]}); the Born charges "
            "would land on the wrong atoms")
    _check_method(freq_out, man)

    born = run_crystal_born(freq_out)
    if not born.strip():
        raise NacError(f"phonopy-crystal-born wrote nothing for {freq_out.name}")
    params = parse_BORN_from_strings(born, phonon.primitive, symprec=1e-5)
    if params.get("factor") is None:
        # BORN says "default": the unit conversion of the calculator (eV, Angstrom).
        from phonopy.interface.calculator import get_calculator_physical_units
        params["factor"] = get_calculator_physical_units("crystal").nac_factor
    return params, born
