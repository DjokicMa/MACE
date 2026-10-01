"""Decide whether a chained step may restart from its predecessor's density matrix.

CRYSTAL23 manual, GUESSP (pp. 114-115): the density matrix written to fort.9
at the end of a previous SCF run is read from fort.20 as the SCF guess. "The
two cases ... must have same symmetry, and same number of atoms, basis
functions and shells. Atoms and shells must be in the same order. The program
does not check the 1:1 old-new correspondence." Different geometrical
parameters, computational conditions or exponents are allowed.

Because CRYSTAL does not check, a wrong matrix is not refused - it is silently
used. So everything here answers "no" unless both decks can be read and agree:

* the geometry frame (dimensionality, space/layer/rod/point group record, atom
  count, atomic numbers in order, any symmetry-editing records after the atom
  list) - the manual's "same symmetry, same atoms in the same order". The
  lattice parameters and coordinates may differ: the follow-up is built from
  the predecessor's final geometry, and the manual allows a changed geometry.
* the basis set (the BASISSET name, or the whole external basis block) - the
  manual's "same basis functions and shells". A basis change needs GUESDUAL
  (p. 113), which MACE does not write.
* the spin treatment (UHF / ROHF / DFT SPIN) - a closed-shell matrix as the
  guess of a spin-polarized run, or the reverse, is not described in the
  manual.

Refused outright:

* ATOMSPIN in the follow-up: it "is used to compute the density matrix as
  superposition of atomic densities ... it does not work with GUESSP" (p. 99).
* TWOCOMPON in either deck: a 2c SCF restarts with GUESSPSO / GUESSP[NOSO]
  (p. 173), not with a 1c GUESSP.

The functional may differ (the manual's "computational conditions"); a
3c method that also changes the basis is caught by the basis comparison.
"""
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Records after the atom list that end the part of the geometry block that
# defines the structure.
_GEOMETRY_STOP = {"OPTGEOM", "FREQCALC", "END", "ENDG", "ENDGEOM", "BASISSET"}
# Lines that close the geometry block or one of its sub-blocks.
_BLOCK_ENDS = {"END", "ENDG", "ENDGEOM", "ENDOPT", "ENDFREQ"}
_SPIN_KEYS = {"UHF", "ROHF", "SPIN"}
_BASIS_END = re.compile(r"^99\s+0$")


def _lines(deck_text: str) -> List[str]:
    return [line.strip() for line in deck_text.splitlines()]


def _norm(line: str) -> Tuple[str, ...]:
    """Tokens of a record, numbers compared as numbers (2.0 == 2.)."""
    out = []
    for tok in line.split():
        try:
            out.append(repr(float(tok)))
        except ValueError:
            out.append(tok.upper())
    return tuple(out)


def geometry_frame(deck_text: str) -> Optional[tuple]:
    """What of the geometry fixes the density matrix layout, or None.

    None means "could not read it", which callers treat as a mismatch.
    """
    lines = _lines(deck_text)
    try:
        dim = lines[1].upper()
        i = 2
        if dim == "CRYSTAL":
            flags = lines[i].split()
            header = [_norm(lines[i]), _norm(lines[i + 1])]
            i += 2
            if len(flags) >= 3 and int(flags[2]) > 1:
                # IFSO > 1: an origin-shift record follows the space group
                header.append(_norm(lines[i]))
                i += 1
            i += 1  # lattice parameters
        elif dim in ("SLAB", "POLYMER"):
            header = [_norm(lines[i])]
            i += 2  # group, lattice parameters
        elif dim == "MOLECULE":
            header = [_norm(lines[i])]
            i += 1
        else:
            return None  # EXTERNAL and anything else: nothing to compare
        natoms = int(lines[i].split()[0])
        atoms = tuple(int(lines[i + 1 + k].split()[0]) for k in range(natoms))
        i += 1 + natoms
        extra = []
        while i < len(lines) and lines[i].upper() not in _GEOMETRY_STOP:
            if lines[i]:
                extra.append(_norm(lines[i]))
            i += 1
        if i >= len(lines):
            return None
    except (IndexError, ValueError):
        return None
    return (dim, tuple(header), natoms, atoms, tuple(extra))


def basis_signature(deck_text: str) -> Optional[tuple]:
    """The deck's basis set: its BASISSET name or its external block, or None."""
    lines = _lines(deck_text)
    upper = [line.upper() for line in lines]
    if "BASISSET" in upper:
        k = upper.index("BASISSET")
        return ("BASISSET", upper[k + 1]) if k + 1 < len(upper) else None
    end = next((k for k, line in enumerate(lines) if _BASIS_END.match(line)), None)
    if end is None:
        return None
    start = max((k for k in range(end) if upper[k] in _BLOCK_ENDS), default=None)
    if start is None:
        return None
    block = [_norm(line) for line in lines[start + 1:end + 1] if line]
    # Basis-block records between "99 0" and its END (GHOSTS and the like).
    k = end + 1
    while k < len(lines) and upper[k] != "END":
        if lines[k]:
            block.append(_norm(lines[k]))
        k += 1
    return ("EXTERNAL", tuple(block))


def spin_treatment(deck_text: str) -> Tuple[str, ...]:
    return tuple(sorted({line for line in (l.upper() for l in _lines(deck_text))
                         if line in _SPIN_KEYS}))


def refusal(source_deck: str, target_deck: str) -> Optional[str]:
    """Why ``target_deck`` may NOT restart from ``source_deck``'s matrix, or None."""
    src_up = {line.upper() for line in _lines(source_deck)}
    tgt_up = {line.upper() for line in _lines(target_deck)}
    if "TWOCOMPON" in src_up or "TWOCOMPON" in tgt_up:
        return "a two-component (TWOCOMPON) deck restarts with GUESSPSO/GUESSPNOSO, not GUESSP"
    if "ATOMSPIN" in tgt_up:
        return "the deck sets ATOMSPIN, which does not work with GUESSP"
    if "SCFDIR" not in tgt_up:
        return "no SCFDIR record to place GUESSP before"
    src_frame, tgt_frame = geometry_frame(source_deck), geometry_frame(target_deck)
    if src_frame is None or tgt_frame is None:
        return "could not read the symmetry and atom list of both decks"
    if src_frame != tgt_frame:
        return "the symmetry or the atom list differs from the previous step"
    src_basis, tgt_basis = basis_signature(source_deck), basis_signature(target_deck)
    if src_basis is None or tgt_basis is None:
        return "could not read the basis set of both decks"
    if src_basis != tgt_basis:
        return "the basis set differs from the previous step"
    if spin_treatment(source_deck) != spin_treatment(target_deck):
        return "the spin treatment (UHF/ROHF/SPIN) differs from the previous step"
    return None


def add_guessp(deck_text: str) -> str:
    """``deck_text`` with a GUESSP record before SCFDIR (where the d12 writer
    puts it). A deck that already asks for GUESSP is returned unchanged."""
    lines = deck_text.splitlines(keepends=True)
    if any(line.strip().upper() == "GUESSP" for line in lines):
        return deck_text
    for k, line in enumerate(lines):
        if line.strip().upper() == "SCFDIR":
            ending = line[len(line.rstrip("\r\n")):] or "\n"
            lines.insert(k, "GUESSP" + ending)
            return "".join(lines)
    return deck_text


def predecessor_f9(calc: Dict) -> Optional[Path]:
    """The non-empty density-matrix file a completed step left, or None.

    The job script saves fort.9 as $DIR/$JOB.f9, next to $JOB.out. An empty one
    is what an aborted run leaves behind and is not a guess.
    """
    out = Path(calc.get("output_file") or "")
    candidates = [out.with_suffix(".f9")] if out.name else []
    if calc.get("work_dir") and out.name:
        candidates.append(Path(calc["work_dir"]) / f"{out.stem}.f9")
    for f9 in candidates:
        if f9.is_file() and f9.stat().st_size > 0:
            return f9
    return None


def script_stages_f20(script_text: str) -> bool:
    """Whether a job script stages $JOB.f20 as fort.20 (and strips GUESSP when
    it has nothing to stage). Scripts from before that staging existed run
    GUESSP with no fort.20, and CRYSTAL stops."""
    return '"$DIR/$JOB.f20"' in script_text


def stage(f9: Path, step_dir: Path, job_name: str) -> Path:
    """Copy the predecessor's matrix to where the job script looks for it."""
    dest = Path(step_dir) / f"{job_name}.f20"
    shutil.copy2(f9, dest)
    return dest
