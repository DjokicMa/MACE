"""The total charge and spin multiplicity an ML potential is told (OrbMol-v2,
UMA's omol head), from the structure's real settings, never a silent default.

In order:

1. an explicit value: ``mace calc --charge Q``, ``--multiplicity M`` (or
   ``--uhf N``, M = N + 1), the GUI's fields;
2. the file the structure came from: a .d12 deck (no CHARGED, CHEMOD or
   DOPING: neutral; SPINLOCK n: n unpaired electrons) or a CRYSTAL .out (the
   charge from N. OF ELECTRONS PER CELL and the atomic numbers when no
   effective core potential hides core electrons; the spin from ALPHA-BETA
   ELECTRONS LOCKED TO, else the last TOTAL ATOMIC SPINS of a spin-polarized
   run);
3. otherwise charge 0 and the lowest multiplicity the electron count allows
   (1 for an even count, 2 for an odd one), said so in the result, with a
   warning for an odd-electron or open-shell system.

ATOMSPIN sets starting atomic spins, not the total, so it is never read as
the multiplicity. Every result records the charge and spin used and where
each came from.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

_ELECTRONS = re.compile(r"N\. OF ELECTRONS PER CELL\s+(\d+)")
_LOCKED = re.compile(r"ALPHA-BETA ELECTRONS LOCKED TO\s+(-?\d+)")
_FLOAT = re.compile(r"^-?\d+\.\d+$")


def needs(model: Dict[str, Any], task: Optional[str] = None) -> bool:
    """Whether ``model`` (with its task head) takes a charge and spin."""
    cs = model.get("charge_spin") or []
    return "*" in cs or (task is not None and task in cs)


def default_task(model: Dict[str, Any], atoms) -> Optional[str]:
    """The task head for this structure's dimensionality (UMA: omat for a
    solid, a slab or a polymer, omol for a molecule)."""
    td = model.get("task_default")
    if not td:
        return None
    return td[str(int(sum(bool(x) for x in atoms.pbc)))]


def _deck(path: Path) -> Dict[str, Any]:
    from mace.preopt import deckscan
    text = path.read_text(errors="replace")
    scan = deckscan.scan(text)
    out: Dict[str, Any] = {"charge": None, "uhf": None, "what": path.name, "notes": []}
    if scan["charge_records"]:
        out["notes"].append(f"the deck is charged ({'/'.join(scan['charge_records'])}) but states no net charge MACE "
                            "can read: give it (--charge)")
    else:
        out["charge"] = 0
    lines = [ln.strip() for ln in text.splitlines()]
    for i, ln in enumerate(lines):
        if ln.upper() == "SPINLOCK" and i + 1 < len(lines):
            tok = lines[i + 1].split()
            try:
                out["uhf"] = abs(int(tok[0]))
            except (IndexError, ValueError):
                pass
            break
    return out


def _out(path: Path) -> Dict[str, Any]:
    text = path.read_text(errors="replace")
    out: Dict[str, Any] = {"charge": None, "uhf": None, "what": path.name, "notes": []}
    m = _ELECTRONS.search(text)
    if m:
        try:
            from mace.gui.structures import read_structure
            nums = [int(n) for n in read_structure(path, 0)["numbers"]]
        except Exception:
            nums = []
        if nums and all(n < 100 for n in nums):
            out["charge"] = sum(nums) - int(m.group(1))
        elif nums:
            out["notes"].append("effective core potentials hide core electrons: the charge is not read from the output")
    m = _LOCKED.search(text)
    if m:
        out["uhf"] = abs(int(m.group(1)))
    else:
        spins = _last_total_spins(text)
        if spins is not None:
            out["uhf"] = abs(int(round(spins)))
    return out


def _last_total_spins(text: str) -> Optional[float]:
    """The sum of the last TOTAL ATOMIC SPINS table (Mulliken), or None."""
    lines = text.splitlines()
    idx = [i for i, ln in enumerate(lines) if ln.strip().startswith("TOTAL ATOMIC SPINS")]
    if not idx:
        return None
    vals: List[float] = []
    for ln in lines[idx[-1] + 1:]:
        tok = ln.split()
        if not tok or not all(_FLOAT.match(t) for t in tok):
            break
        vals += [float(t) for t in tok]
    return sum(vals) if vals else None


def from_file(path) -> Optional[Dict[str, Any]]:
    """{charge, uhf, what, notes} read from the structure's file, or None
    (a CIF, an xyz, ...: no charge or spin in it)."""
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    suffix = p.suffix.lower()
    try:
        if suffix == ".d12":
            return _deck(p)
        if suffix in (".out", ".o", ".log"):
            return _out(p)
    except Exception:
        return None
    return None


def resolve(atoms, *, charge: Optional[int] = None, uhf: Optional[int] = None,
            multiplicity: Optional[int] = None, file_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """{charge, spin (the multiplicity, 2S+1), uhf, charge_source, spin_source,
    warnings}; ValueError for a multiplicity the electron count cannot have."""
    warnings: List[str] = []
    fi = file_info or {}
    if charge is not None:
        q, qsrc = int(charge), "given"
    elif fi.get("charge") is not None:
        q, qsrc = int(fi["charge"]), f"from {fi.get('what', 'the file')}"
    else:
        q, qsrc = 0, "assumed (no charge given or found): 0"
        warnings += list(fi.get("notes") or [])
    nel = int(sum(int(z) for z in atoms.numbers)) - q
    if multiplicity is not None:
        mult, ssrc = int(multiplicity), "given"
    elif uhf is not None:
        mult, ssrc = int(uhf) + 1, "given (unpaired electrons + 1)"
    elif fi.get("uhf") is not None:
        mult, ssrc = int(fi["uhf"]) + 1, f"from {fi.get('what', 'the file')}"
    else:
        mult = 2 if nel % 2 else 1
        ssrc = f"assumed: the lowest multiplicity for {nel} electrons"
    if nel < 0:
        raise ValueError(f"a charge of {q} leaves {nel} electrons")
    if mult < 1 or (nel + mult - 1) % 2:
        raise ValueError(f"a spin multiplicity of {mult} is impossible with {nel} electrons "
                         f"(charge {q}): it must be {'even' if nel % 2 else 'odd'}")
    if nel % 2:
        warnings.append(f"an odd number of electrons ({nel}): an open shell (multiplicity {mult}); check the charge "
                        "and spin")
    elif mult > 1:
        warnings.append(f"an open-shell state (multiplicity {mult})")
    return {"charge": q, "spin": mult, "uhf": mult - 1, "charge_source": qsrc, "spin_source": ssrc,
            "electrons": nel, "warnings": warnings}
