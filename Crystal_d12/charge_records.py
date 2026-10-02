"""Carry a parent deck's electron-count records (CHEMOD, CHARGED, DOPING) into a derived deck.

A charged or ionic system is set up in CRYSTAL23 by records that change how many
electrons the cell holds or where they start:

    CHEMOD   (basis-set input, manual p.80)  per-atom shell charges of the guess,
             "NC" then NC records "LA CH(1) ... CH(NS)"
    CHARGED  (basis-set input, manual p.80)  allow a non-neutral periodic cell
    DOPING   (SCF input, manual p.102)       add (or with a negative value, remove)
             electrons; a compensating background is added when needed

A deck derived from such a parent (SP, FREQ, ... from an OPT) must keep them, or
it silently runs a different charge state. MACE writes derived decks from the
parent's settings, which do not model these records, so they are copied here
from the parent deck text after the derived deck is written.

Where they go is what the group's Li/FSI/EC cluster decks use, which CRYSTAL23
ran (HSEsol-3c, BASISSET SOLDEF2MSVP): the basis-set records right after the
BASISSET name line (or after the "99 0" that ends an explicit basis), and DOPING
inside the SCF block, before its final END.

This module is standard library only.
"""

import re
from typing import Dict, List, Optional, Tuple


def _records(text: str) -> List[str]:
    return text.splitlines()


def extract_charge_records(parent_text: str) -> Dict[str, List[str]]:
    """The CHEMOD, CHARGED and DOPING records of a deck, as their original lines.

    Returns a dict with keys "CHEMOD", "CHARGED", "DOPING"; each value is the
    list of lines making up that record (empty when absent). Line 1 (the title)
    is never read as a keyword.
    """
    lines = _records(parent_text)
    found = {"CHEMOD": [], "CHARGED": [], "DOPING": []}
    i = 1
    while i < len(lines):
        key = lines[i].strip().upper()
        if key == "CHEMOD" and not found["CHEMOD"] and i + 1 < len(lines):
            try:
                n = int(lines[i + 1].split()[0])
            except (ValueError, IndexError):
                i += 1
                continue
            found["CHEMOD"] = lines[i:i + 2 + n]
            i += 2 + n
            continue
        if key == "CHARGED" and not found["CHARGED"]:
            found["CHARGED"] = [lines[i]]
        elif key == "DOPING" and not found["DOPING"] and i + 1 < len(lines):
            found["DOPING"] = lines[i:i + 2]
            i += 2
            continue
        i += 1
    return found


def chemod_labels(chemod: List[str]) -> List[int]:
    """The atom labels a CHEMOD record names."""
    labels = []
    for line in chemod[2:]:
        try:
            labels.append(int(line.split()[0]))
        except (ValueError, IndexError):
            pass
    return labels


def _geometry_atom_count(lines: List[str]) -> Optional[int]:
    """Number of atoms in a deck's geometry block (the record after the symmetry line)."""
    dims = {"CRYSTAL": 3, "SLAB": 2, "POLYMER": 2, "MOLECULE": 2}
    for i, line in enumerate(lines[1:], start=1):
        key = line.strip().upper()
        if key in dims:
            # CRYSTAL: IFLAG IFHR IFSO / space group / cell / natoms
            # SLAB, POLYMER, MOLECULE: group / cell (none for MOLECULE) / natoms
            j = i + 1
            if key == "CRYSTAL":
                j += 3
            elif key in ("SLAB", "POLYMER"):
                j += 2
            else:
                j += 1
            try:
                return int(lines[j].split()[0])
            except (ValueError, IndexError):
                return None
    return None


def _basis_insert_index(lines: List[str]) -> Optional[int]:
    """Where basis-set input keywords go: after the BASISSET name line, or after "99 0"."""
    for i, line in enumerate(lines[1:], start=1):
        if line.strip().upper() == "BASISSET" and i + 1 < len(lines):
            return i + 2
    last = None
    for i, line in enumerate(lines):
        if re.fullmatch(r"\s*99\s+0\s*", line):
            last = i
    return None if last is None else last + 1


def carry_charge_records(child_text: str, parent_text: str) -> Tuple[str, List[str]]:
    """The child deck with the parent's charge records added, and messages for the user.

    Records the child already has are left alone (an explicit template wins).
    CHEMOD is only copied when both decks list the same number of atoms, since
    its atom labels are positions in that list; otherwise a warning is returned
    and CHEMOD is left out. DOPING and CHARGED do not depend on atom order.
    """
    parent = extract_charge_records(parent_text)
    if not any(parent.values()):
        return child_text, []
    lines = _records(child_text)
    trailing_newline = child_text.endswith("\n")
    child_has = extract_charge_records(child_text)
    notes = []

    basis_add: List[str] = []
    if parent["CHEMOD"] and not child_has["CHEMOD"]:
        n_parent = _geometry_atom_count(_records(parent_text))
        n_child = _geometry_atom_count(lines)
        if n_parent is not None and n_parent == n_child:
            basis_add += parent["CHEMOD"]
            notes.append(f"kept the parent's CHEMOD (atoms {', '.join(map(str, chemod_labels(parent['CHEMOD'])))})")
        else:
            notes.append(
                "WARNING: the parent deck has CHEMOD but the new deck lists "
                f"{n_child} atoms against the parent's {n_parent}, so its atom labels "
                "cannot be trusted; CHEMOD was NOT copied - add it by hand")
    if parent["CHARGED"] and not child_has["CHARGED"]:
        basis_add += parent["CHARGED"]
        notes.append("kept the parent's CHARGED")
    if basis_add:
        at = _basis_insert_index(lines)
        if at is None:
            notes.append("WARNING: no basis-set block found; CHEMOD/CHARGED were NOT copied")
        else:
            lines[at:at] = basis_add

    if parent["DOPING"] and not child_has["DOPING"]:
        last_end = max((i for i, l in enumerate(lines) if l.strip().upper() == "END"), default=None)
        if last_end is None:
            notes.append("WARNING: no final END found; DOPING was NOT copied")
        else:
            lines[last_end:last_end] = parent["DOPING"]
            notes.append(f"kept the parent's DOPING {parent['DOPING'][1].strip()}")

    out = "\n".join(lines)
    return out + ("\n" if trailing_newline else ""), notes
