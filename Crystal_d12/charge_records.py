"""Carry a parent deck's electron-count records (CHEMOD, CHARGED, DOPING) into a derived deck.

A charged or ionic system is set up in CRYSTAL23 by records that change how many
electrons the cell holds or where they start:

    CHEMOD   (basis-set input, manual p.80)  per-atom shell charges of the guess,
             "NC" then NC records "LA CH(1) ... CH(NS)"
    CHARGED  (basis-set input, manual p.80)  allow a non-neutral periodic cell
    DOPING   (SCF input, manual p.102)       add (or with a negative value, remove)
             electrons; a compensating background is added when needed
    GHOSTS   (basis-set input, manual p.82)  atoms kept as basis functions only
             (a vacancy with its basis, as `mace defects --ghost` writes)

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


def _numeric_line(line: str) -> bool:
    """True for a line of numbers only (a CHEMOD label or shell-charge record)."""
    tokens = line.split()
    if not tokens:
        return False
    try:
        for t in tokens:
            float(t.upper().replace("D", "E"))
    except ValueError:
        return False
    return True


def chemod_end(lines: List[str], i: int) -> Optional[int]:
    """Index just past the CHEMOD record starting at lines[i], or None if it is malformed.

    CRYSTAL reads each atom's record (label, then its shell charges) free
    format, so the label may share a line with the charges or stand on a line
    of its own, the charges following; both layouts occur. The record is the
    count line plus every line of numbers after it (at least one per atom).
    """
    try:
        n = int(lines[i + 1].split()[0])
    except (ValueError, IndexError):
        return None
    j = i + 2
    while j < len(lines) and _numeric_line(lines[j]):
        j += 1
    return max(j, i + 2 + n)


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
            end = chemod_end(lines, i)
            if end is None:
                i += 1
                continue
            found["CHEMOD"] = lines[i:end]
            i = end
            continue
        if key == "CHARGED" and not found["CHARGED"]:
            found["CHARGED"] = [lines[i]]
        elif key == "DOPING" and not found["DOPING"] and i + 1 < len(lines):
            found["DOPING"] = lines[i:i + 2]
            i += 2
            continue
        i += 1
    return found


def extract_ghosts(text: str) -> List[str]:
    """The GHOSTS record of a deck (basis-set input, manual p.82) as its lines, or [].

    "GHOSTS", the count NA, then NA atom labels, free format (one or several
    lines). A record that does not read that way is not returned.
    """
    lines = _records(text)
    for i in range(1, len(lines)):
        if lines[i].strip().upper() != "GHOSTS":
            continue
        try:
            n = int(lines[i + 1].split()[0])
        except (ValueError, IndexError):
            return []
        labels, j = [], i + 2
        while len(labels) < n and j < len(lines) and _numeric_line(lines[j]):
            labels += lines[j].split()
            j += 1
        return lines[i:j] if len(labels) == n else []
    return []


def ghost_labels(ghosts: List[str]) -> List[int]:
    """The atom labels a GHOSTS record names."""
    return [int(t) for line in ghosts[2:] for t in line.split()]


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


LIBRARY_END_MESSAGE = (
    "the deck has an END right after the BASISSET library name (and its CHEMOD/CHARGED): "
    "with a library basis there is no basis-set block, so that END ends block 3 and CRYSTAL23 "
    "silently ignores the DFT block and every keyword after it (the run falls back to HF, "
    "measured on CRYSTAL23) - remove that END")


def library_basis_end(lines: List[str]) -> Optional[int]:
    """Index of an END that directly follows a BASISSET library name, or None.

    With "BASISSET <library>" there is no basis-set block: the lines after the
    name are already block 3 (CHEMOD and CHARGED are still read there,
    measured on CRYSTAL23). An END after the name and its CHEMOD/CHARGED
    records - "closing" CHEMOD as an explicit basis would - ends block 3, and
    the DFT block, TOLINTEG and everything after it are ignored without an
    error. Only an END followed by more input counts.
    """
    for i, line in enumerate(lines[1:], start=1):
        if line.strip().upper() != "BASISSET" or i + 1 >= len(lines):
            continue
        j = i + 2
        while j < len(lines):
            key = lines[j].strip().upper()
            if key == "CHEMOD":
                j = chemod_end(lines, j)
                if j is None:
                    return None
                continue
            if key == "CHARGED":
                j += 1
                continue
            break
        if j < len(lines) and lines[j].strip().upper() == "END" \
                and any(l.strip() for l in lines[j + 1:]):
            return j
        return None
    return None


class ChargeCarryError(ValueError):
    """The child deck cannot get the parent's charge state (the reason)."""


def _ic():
    """ion_charges (deck reading, basis parsing), imported only when needed."""
    import ion_charges
    return ion_charges


def _num(text: str):
    from decimal import Decimal
    return Decimal(text.strip().split()[0])


def _fmt_doping(value) -> str:
    from decimal import Decimal
    value = Decimal(str(value))
    if value == value.to_integral_value():
        return str(int(value))
    return _ic().fmt_charge(value)


def _raw_basis_blocks(text: str) -> Optional[Dict[int, Tuple[str, ...]]]:
    """{atom label: its explicit basis block, line by line}, or None if not readable."""
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines[1:], start=1):
        if line.strip().upper() in ("END", "ENDGEOM") and i + 1 < len(lines) \
                and re.fullmatch(r"\s*\d+\s+\d+\s*", lines[i + 1]):
            start = i + 1
            break
    if start is None:
        return None
    blocks: Dict[int, Tuple[str, ...]] = {}
    i = start
    try:
        while i < len(lines):
            z, nshell = (int(t) for t in lines[i].split()[:2])
            if z == 99 and nshell == 0:
                return blocks
            j = i + 1
            head = lines[j].strip().upper()
            if head in ("HAYWLC", "HAYWSC", "BARTHE", "DURAND"):
                j += 1
            elif head == "INPUT":
                j += 2 + sum(int(float(x)) for x in lines[j + 1].split()[1:7])
            for _ in range(nshell):
                rec = lines[j].split()
                j += 1 + (int(float(rec[2])) if int(rec[0]) == 0 else 0)
            blocks[z] = tuple(" ".join(l.split()) for l in lines[i:j])
            i = j
    except (ValueError, IndexError):
        return None
    return None


def _basis_section(text: str) -> Optional[Tuple[str, ...]]:
    """An explicit basis input as text: the lines after the geometry's END up to "99 0"."""
    lines = text.splitlines()
    end = _basis_insert_index(lines)
    if end is None:
        return None
    start = None
    for i in range(end - 1, 0, -1):
        if lines[i].strip().upper() in ("END", "ENDGEOM"):
            start = i + 1
            break
    if start is None:
        return None
    return tuple(" ".join(l.split()) for l in lines[start:end])


def deck_basis_name(text: str) -> Optional[str]:
    """The internal library a deck loads (BASISSET name), or None."""
    lines = text.splitlines()
    for i, line in enumerate(lines[1:], start=1):
        if line.strip().upper() == "BASISSET" and i + 1 < len(lines):
            return lines[i + 1].strip().split()[0] if lines[i + 1].strip() else None
    return None


def _norm(name: str) -> str:
    return re.sub(r"[^0-9A-Z]", "", str(name).upper())


def same_basis(parent_text: str, child_text: str, labels: List[int]) -> bool:
    """True when the child loads the basis the parent's CHEMOD atoms were written for.

    The same internal library (BASISSET name), or identical explicit blocks
    for the atom types CHEMOD names (labels are positions in the atom list).
    """
    p_name, c_name = deck_basis_name(parent_text), deck_basis_name(child_text)
    if p_name or c_name:
        return bool(p_name and c_name and _norm(p_name) == _norm(c_name))
    p_blocks, c_blocks = _raw_basis_blocks(parent_text), _raw_basis_blocks(child_text)
    if p_blocks is None or c_blocks is None:
        # not readable block by block: the same basis only if the whole
        # basis input (geometry END to "99 0") is the same
        p_sec, c_sec = _basis_section(parent_text), _basis_section(child_text)
        return p_sec is not None and p_sec == c_sec
    try:
        geom = _ic().read_deck_geometry(parent_text)
        kinds = {geom.z_labels[label - 1] for label in labels}
    except Exception:
        kinds = set(p_blocks)
    return all(k in p_blocks and p_blocks.get(k) == c_blocks.get(k) for k in kinds)


def explicit_chemod_deficit(parent_text: str, chemod: List[str]):
    """Electrons the parent's CHEMOD removes against its explicit basis (None if not explicit)."""
    from decimal import Decimal
    if deck_basis_name(parent_text) or not chemod:
        return None
    ic = _ic()
    try:
        blocks = ic.parse_basis_blocks(parent_text)
        geom = ic.read_deck_geometry(parent_text)
        deficit = Decimal(0)
        for line in chemod[2:]:
            tok = line.split()
            label = int(tok[0])
            z = geom.z_labels[label - 1]
            block = blocks.get(z) or blocks.get(z % 100)
            deficit += block.valence - sum((Decimal(t) for t in tok[1:]), Decimal(0))
        return deficit
    except Exception:
        return None


def out_electron_offset(out_text: str):
    """Electrons a run had beyond neutral, from its .out: Mulliken count - sum of valence charges.

    None when the .out has no Mulliken print. The header count is never used:
    DOPING leaves it at the count before doping.
    """
    from decimal import Decimal
    frag = _ic().frag
    count = frag.mulliken_electron_count(out_text)
    if count is None:
        return None
    pops, _ = frag.mulliken_atoms(out_text)
    if not pops:
        return None
    pp = frag.valence_charges(out_text)
    ref = [frag.valence_of(a["z"], pp) for a in pops]
    if any(r is None for r in ref):
        return None  # an ECP atom whose valence charge the .out does not give
    zsum = sum(ref)
    offset = Decimal(str(round(count - zsum, 6)))
    return Decimal(0) if abs(offset) < Decimal("0.001") else offset


def parent_electron_offset(parent_text: str, parent_out_text: Optional[str] = None):
    """Electrons the parent ran with beyond neutral: from its .out, else from its records."""
    from decimal import Decimal
    if parent_out_text:
        offset = out_electron_offset(parent_out_text)
        if offset is not None:
            return offset
    rec = extract_charge_records(parent_text)
    offset = _num(rec["DOPING"][1]) if rec["DOPING"] else Decimal(0)
    deficit = explicit_chemod_deficit(parent_text, rec["CHEMOD"])
    if deficit:
        offset -= deficit
    return offset


def _child_electrons(child_text: str):
    """(electron count, dimensionality) of a deck, or (None, None) when it cannot be
    counted (symmetry, geometry editing)."""
    from decimal import Decimal
    ic = _ic()
    try:
        geom = ic.read_deck_geometry(child_text)
    except Exception:
        return None, None
    if geom.symmetric:
        return None, None
    rec = extract_charge_records(child_text)
    if deck_basis_name(child_text):
        total = Decimal(sum(z % 100 for z in geom.z_labels))
    else:
        try:
            blocks = ic.parse_basis_blocks(child_text)
        except (ValueError, IndexError):
            return None, None
        total = Decimal(0)
        for z in geom.z_labels:
            block = blocks.get(z) or blocks.get(z % 100)
            total += block.valence if block else Decimal(z % 100)
        deficit = explicit_chemod_deficit(child_text, rec["CHEMOD"])
        if deficit:
            total -= deficit
    # A ghost atom keeps its basis functions but loses its nucleus and electrons.
    for label in ghost_labels(extract_ghosts(child_text)):
        try:
            z = geom.z_labels[label - 1]
        except IndexError:
            return None, None
        if deck_basis_name(child_text):
            total -= z % 100
        else:
            block = blocks.get(z) or blocks.get(z % 100)
            total -= block.valence if block else Decimal(z % 100)
    if rec["DOPING"]:
        total += _num(rec["DOPING"][1])
    return total, geom.dimensionality


def ensure_open_shell(child_text: str, parent_text: Optional[str]) -> Tuple[str, List[str]]:
    """Give a closed-shell child with an odd electron count the parent's SPIN /
    SPINLOCK, or refuse it (ChargeCarryError) - a molecule; a periodic cell
    only gets a warning, since it can hold an odd count as a half-filled band."""
    electrons, dim = _child_electrons(child_text)
    if electrons is None or electrons != electrons.to_integral_value() or int(electrons) % 2 == 0:
        return child_text, []
    ic = _ic()
    child = ic.deck_method(child_text)
    # an open-shell odd count with an HSE-type screened hybrid (measured to
    # crash on CRYSTAL23, 0D and 3D); the functional is kept, with a warning
    hybrid = [w for w in (ic.screened_hybrid_odd_warning(child),) if w]
    if child.spin and (child.method == "HF" or child.spinlock):
        return child_text, hybrid
    parent = ic.deck_method(parent_text) if parent_text else None
    lines = child_text.splitlines()
    notes = []
    if child.method == "DFT" and parent is not None and parent.spin:
        upper = [l.strip().upper() for l in lines]
        if not child.spin and "ENDDFT" in upper:
            lines.insert(upper.index("ENDDFT"), "SPIN")
            notes.append("kept the parent's SPIN (odd electron count)")
        if parent.spinlock and not child.spinlock:
            end = max(i for i, l in enumerate(lines) if l.strip().upper() == "END")
            lines[end:end] = ["SPINLOCK", f"{parent.spinlock[0]} {parent.spinlock[1]}"]
            notes.append(f"kept the parent's SPINLOCK {parent.spinlock[0]} {parent.spinlock[1]}")
        if not (parent.spinlock or child.spinlock):
            notes.append("WARNING: odd electron count with SPIN but no SPINLOCK, as in the parent")
        return "\n".join(lines) + ("\n" if child_text.endswith("\n") else ""), notes + hybrid
    if child.spin:
        return child_text, hybrid
    if dim != "MOLECULE":
        return child_text, [f"WARNING: the new deck holds {int(electrons)} electrons per cell (odd) "
                            "and is closed shell (a half-filled band); the parent had no "
                            "SPIN/SPINLOCK to keep"]
    raise ChargeCarryError(
        f"the new deck holds {int(electrons)} electrons (odd) but is closed shell; the parent "
        "had no SPIN/SPINLOCK to keep - give the child SPIN + SPINLOCK (or UHF)")


def _chemod_numbers(chemod: List[str]) -> List[float]:
    """A CHEMOD record's numbers after its count, whatever the line layout."""
    return [float(t.upper().replace("D", "E")) for line in chemod[2:] for t in line.split()]


def _setup_matches(setup, parent: Dict[str, List[str]]) -> bool:
    """True when an ion_charges set-up writes exactly the parent's charge records."""
    from decimal import Decimal
    mine = setup.chemod_lines()
    if not mine or len(mine) < 2 or mine[1].strip() != parent["CHEMOD"][1].split()[0]:
        return False
    a, b = _chemod_numbers(mine), _chemod_numbers(parent["CHEMOD"])
    if len(a) != len(b) or any(abs(x - y) > 1e-6 for x, y in zip(a, b)):
        return False
    if bool(setup.charged) != bool(parent["CHARGED"]):
        return False
    parent_doping = _num(parent["DOPING"][1]) if parent["DOPING"] else None
    if (setup.doping is None) != (parent_doping is None):
        return False
    return setup.doping is None or abs(Decimal(setup.doping) - parent_doping) < Decimal("1e-6")


def regenerate_chemod(child_text: str, parent_text: str,
                      parent_out_text: Optional[str] = None):
    """The child with the parent's CHEMOD worked out again for the child's basis.

    Only when the parent's CHEMOD (with its CHARGED/DOPING) is exactly what
    MACE's ion charges (ion_charges.plan_for_deck, ion_charges auto) write
    for the parent deck: then the same set-up is planned for the child's
    geometry and basis, and its records keep the parent's net charge.
    Returns (child text, notes), or (None, the reason it was not done).
    """
    ic = _ic()
    parent = extract_charge_records(parent_text)
    stripped_parent = ic.strip_charge_records(parent_text)
    offset = parent_electron_offset(parent_text, parent_out_text)
    variants = []
    for chemod in ("auto", "yes"):
        variants.append({"ion_charges": "auto", "chemod": chemod})
        if offset:
            variants.append({"ion_charges": "auto", "chemod": chemod, "net_charge": -offset})
    match = None
    for opts in variants:
        try:
            setup = ic.plan_for_deck(stripped_parent, **opts)
        except Exception:
            continue
        if setup is not None and _setup_matches(setup, parent):
            match = (opts, setup)
            break
    if match is None:
        return None, ("it is not the CHEMOD MACE's ion charges write for the parent, so "
                      "MACE cannot work out its equivalent")
    opts, parent_setup = match
    stripped_child = ic.strip_charge_records(child_text)
    try:
        setup = ic.plan_for_deck(stripped_child, parent_spinlock=ic.deck_method(parent_text).spinlock,
                                 **opts)
        if setup is None or not setup.chemod:
            return None, "the new basis takes no CHEMOD for these ions"
        if setup.net_charge != parent_setup.net_charge:
            return None, (f"the new deck's ions give a net charge of {setup.net_charge}, "
                          f"the parent's {parent_setup.net_charge}")
        text = ic.apply_charge_setup(stripped_child, setup)
    except Exception as e:
        return None, str(e)
    labels = ", ".join(str(label) for label, _ in setup.chemod)
    notes = [f"the parent's CHEMOD (MACE's ion charges) was worked out again for the new basis "
             f"(atoms {labels})"]
    if setup.charged:
        notes.append("kept the parent's net charge with CHARGED (CHEMOD sets the count in an "
                     "explicit basis)")
    if setup.doping is not None:
        notes.append(f"kept the parent's net charge as DOPING {setup.doping_lines()[1]}")
    return text, notes + list(setup.warnings)


def carry_charge_records(child_text: str, parent_text: str,
                         parent_out_text: Optional[str] = None) -> Tuple[str, List[str]]:
    """The child deck with the parent's charge state, and messages for the user.

    Records the child already has are left alone (an explicit template wins).
    CHEMOD is copied only when both decks list the same atoms (its labels are
    positions in that list) and load the same basis (its shell charges are
    per shell of that basis). With another basis, a CHEMOD that MACE's ion
    charges wrote for the parent is worked out again for the new basis
    (regenerate_chemod); any other is left out with a warning, and a
    charge it carried (an explicit basis with a non-neutral CHEMOD, measured to
    set the electron count) becomes DOPING, taken from the parent's Mulliken
    count when its .out is given. CHARGED goes with CHEMOD. DOPING is kept.
    GHOSTS is copied, ahead of CHEMOD, when both decks list the same atoms.
    An odd count in a closed-shell child gets the parent's SPIN/SPINLOCK or
    is refused (ChargeCarryError).
    """
    parent = extract_charge_records(parent_text)
    parent_ghosts = extract_ghosts(parent_text)
    if not any(parent.values()) and not parent_ghosts:
        return child_text, []
    lines = _records(child_text)
    trailing_newline = child_text.endswith("\n")
    child_has = extract_charge_records(child_text)
    notes = []
    n_parent = _geometry_atom_count(_records(parent_text))
    if parent_out_text and "SUPERCELL GENERATED" in parent_out_text:
        # The parent built a supercell (SUPERCEL/SUPERCON): its atom labels
        # are the supercell's, which opt2d12 writes out atom by atom in P1.
        cell = re.findall(r"ATOMS IN THE UNIT CELL:\s+(\d+)", parent_out_text)
        if cell:
            n_parent = int(cell[-1])
    n_child = _geometry_atom_count(lines)
    if library_basis_end(_records(parent_text)) is not None:
        notes.append("WARNING: the parent deck has an END right after its BASISSET library name "
                     "(and CHEMOD), which ends block 3 there: CRYSTAL23 ignored the parent's DFT "
                     "block and the keywords after it, so the parent ran HF - check its energy "
                     "before using it")
    if library_basis_end(lines) is not None:
        raise ChargeCarryError(LIBRARY_END_MESSAGE.replace("the deck", "the new deck", 1))

    carry_chemod = False
    if parent["CHEMOD"] and not child_has["CHEMOD"]:
        labels = chemod_labels(parent["CHEMOD"])
        if n_parent is None or n_parent != n_child:
            notes.append(
                "WARNING: the parent deck has CHEMOD but the new deck lists "
                f"{n_child} atoms against the parent's {n_parent}, so its atom labels "
                "cannot be trusted; CHEMOD was NOT copied - add it by hand")
        elif not same_basis(parent_text, child_text, labels):
            if not any(child_has.values()):
                regenerated, why = regenerate_chemod(child_text, parent_text, parent_out_text)
                if regenerated is not None:
                    out, spin_notes = ensure_open_shell(regenerated, parent_text)
                    return out, notes + why + spin_notes
            else:
                why = "the new deck already has its own charge records"
            notes.append(
                "WARNING: the parent's CHEMOD gives shell charges for its own basis set, and "
                f"the new deck loads a different one; CHEMOD was NOT copied ({why}) - set it "
                "for the new basis with --ion-charges auto (opt2d12 or mace ions)")
        else:
            carry_chemod = True

    basis_add: List[str] = []
    if parent_ghosts and not extract_ghosts(child_text):
        # GHOSTS names atoms by their label, as CHEMOD does: kept only when
        # the child lists the same atoms.
        if n_parent is None or n_parent != n_child:
            notes.append(
                "WARNING: the parent deck has GHOSTS but the new deck lists "
                f"{n_child} atoms against the parent's {n_parent}, so its atom labels "
                "cannot be trusted; GHOSTS was NOT copied - add it by hand")
        else:
            basis_add += parent_ghosts
            notes.append(f"kept the parent's GHOSTS (atoms {', '.join(map(str, ghost_labels(parent_ghosts)))})")
    if carry_chemod:
        basis_add += parent["CHEMOD"]
        notes.append(f"kept the parent's CHEMOD (atoms {', '.join(map(str, chemod_labels(parent['CHEMOD'])))})")
    dropped_chemod = bool(parent["CHEMOD"]) and not carry_chemod and not child_has["CHEMOD"]
    if parent["CHARGED"] and not child_has["CHARGED"]:
        if dropped_chemod:
            notes.append("left out the parent's CHARGED (it went with its CHEMOD)")
        else:
            basis_add += parent["CHARGED"]
            notes.append("kept the parent's CHARGED")
    if basis_add:
        at = _basis_insert_index(lines)
        if at is None:
            notes.append("WARNING: no basis-set block found; GHOSTS/CHEMOD/CHARGED were NOT copied")
        else:
            lines[at:at] = basis_add

    doping = None
    if not child_has["DOPING"]:
        if dropped_chemod and explicit_chemod_deficit(parent_text, parent["CHEMOD"]):
            # the parent's charge was in its explicit CHEMOD: carry it as DOPING
            offset = parent_electron_offset(parent_text, parent_out_text)
            if offset:
                doping = ["DOPING", _fmt_doping(offset)]
                notes.append(f"the parent's charge (set by its CHEMOD) is DOPING {doping[1]} here")
        elif parent["DOPING"]:
            doping = list(parent["DOPING"])
            notes.append(f"kept the parent's DOPING {parent['DOPING'][1].strip()}")
    if doping:
        last_end = max((i for i, l in enumerate(lines) if l.strip().upper() == "END"), default=None)
        if last_end is None:
            notes.append("WARNING: no final END found; DOPING was NOT copied")
        else:
            lines[last_end:last_end] = doping
            if n_parent is not None and n_child is not None and n_parent != n_child:
                notes.append(f"WARNING: DOPING {doping[1].strip()} was copied unchanged into a deck "
                             f"with {n_child} atoms (the parent has {n_parent}); check it is still "
                             "the charge you want")

    out = "\n".join(lines) + ("\n" if trailing_newline else "")
    out, spin_notes = ensure_open_shell(out, parent_text)
    return out, notes + spin_notes


def _ecp_valence_unknown(out_text: str) -> bool:
    """True when the .out's Mulliken print has an ECP atom with no known valence charge."""
    frag = _ic().frag
    if frag.mulliken_electron_count(out_text) is None:
        return False
    pops, _ = frag.mulliken_atoms(out_text)
    pp = frag.valence_charges(out_text)
    return any(frag.valence_of(a["z"], pp) is None for a in pops)


def out_charge_signs(out_text: str) -> List[str]:
    """What in a .out shows a charged or ionic run, without a Mulliken print.

    CHEMOD ("ELECTRONIC CONFIGURATION MODIFIED IN n ATOMS"), CHARGED
    ("CHARGED CELL"), "UNIT CELL NOT NEUTRAL" and a renormalised guess (a
    CHARGE NORMALIZATION FACTOR other than 1). DOPING prints nothing
    (measured on CRYSTAL23), so a doped run with none of these shows no sign.
    """
    text = out_text or ""
    signs = []
    if "ELECTRONIC CONFIGURATION MODIFIED" in text:
        signs.append("CHEMOD")
    if "CHARGED CELL" in text:
        signs.append("CHARGED")
    if "UNIT CELL NOT NEUTRAL" in text:
        signs.append("UNIT CELL NOT NEUTRAL")
    for value in re.findall(r"CHARGE NORMALIZATION FACTOR\s+([-\d.]+)", text):
        try:
            if abs(float(value) - 1.0) > 1e-6:
                signs.append("a renormalised guess")
                break
        except ValueError:
            continue
    return signs


def charge_from_parent_out(child_text: str, out_text: str) -> Tuple[str, List[str]]:
    """A child written from a .out alone (no parent deck) gets DOPING from its Mulliken count.

    Without a Mulliken print the charge state cannot be recovered and the
    deck is left as written: a warning says so when the .out shows a sign of
    a charged run (out_charge_signs), otherwise an informational note (a
    neutral FREQ parent, the usual case, is not warned about).
    """
    if extract_charge_records(child_text)["DOPING"]:
        return child_text, []
    offset = out_electron_offset(out_text)
    if offset is None and _ecp_valence_unknown(out_text):
        return child_text, ["WARNING: no parent deck, and the .out does not give the valence charge "
                            "of its pseudopotential atom(s), so the parent's electron count (and any "
                            "DOPING) cannot be recovered from its Mulliken print - check the new deck"]
    if offset is None:
        signs = out_charge_signs(out_text)
        if signs:
            return child_text, [f"WARNING: no parent deck and no Mulliken electron count in the .out, "
                                f"which shows a charged or ionic run ({', '.join(signs)}), so the "
                                "parent's charge state (CHEMOD/DOPING) cannot be recovered - check "
                                "the new deck"]
        return child_text, ["no parent deck and no Mulliken electron count in the .out; it shows "
                            "no sign of a charged run, so the new deck is neutral (DOPING leaves "
                            "no trace in an .out: add it by hand if the parent had one)"]
    if not offset:
        return child_text, []
    lines = child_text.splitlines()
    last_end = max((i for i, l in enumerate(lines) if l.strip().upper() == "END"), default=None)
    if last_end is None:
        return child_text, ["WARNING: no final END found; DOPING was NOT added"]
    lines[last_end:last_end] = ["DOPING", _fmt_doping(offset)]
    out = "\n".join(lines) + ("\n" if child_text.endswith("\n") else "")
    notes = [f"the parent ran with {_fmt_doping(offset)} electron(s) beyond neutral "
             f"(Mulliken count): DOPING {_fmt_doping(offset)}"]
    out, spin_notes = ensure_open_shell(out, None)
    return out, notes + spin_notes
