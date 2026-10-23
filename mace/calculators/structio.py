"""Reading the structure of a ``mace calc`` input.

CIF, .d12, xyz/extxyz and POSCAR go through ``mace preopt``'s reader (the
CIF's own space group and site map are kept, so an optimized structure can
be written back in that group; a group-1 SLAB/POLYMER/MOLECULE deck and an
extended XYZ keep their pbc; a deck whose geometry block edits the cell,
such as SUPERCEL or ATOMREMO, is refused with the same words). Every other format the GUI opens
(CRYSTAL outputs at any frame, xyz/extxyz, POSCAR, fort.34, gen, pdb, mol)
goes through the GUI's reader; ase trajectories through ase.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

from .base import CalcError


def atoms_from_payload(p):
    from ase import Atoms
    import numpy as np
    nums = [int(n) % 100 or int(n) for n in p["numbers"]]     # ECP atomic numbers (2xx) are the element
    pbc = [bool(x) for x in p.get("pbc", [True, True, True])]
    if all(pbc):
        return Atoms(numbers=nums, cell=np.asarray(p["cell"], float), scaled_positions=np.asarray(p["frac"], float),
                     pbc=True)
    return Atoms(numbers=nums, cell=np.asarray(p["cell"], float), positions=np.asarray(p["cart"], float), pbc=pbc)


def read(path: Path, frame: int = -1) -> Tuple[Any, Optional[Any], Optional[Any]]:
    """(atoms, spacegroup or None, site kinds or None); ``atoms.info["source_path"]``
    names the file (an ML model that takes a charge and spin reads them there)."""
    atoms, sg, kinds = _read(Path(path), frame)
    try:
        atoms.info["source_path"] = str(Path(path).resolve())
    except Exception:
        pass
    return atoms, sg, kinds


def _read(path: Path, frame: int = -1) -> Tuple[Any, Optional[Any], Optional[Any]]:
    path = Path(path)
    if not path.is_file():
        raise CalcError(f"{path} is not a file", kind="usage")
    suffix = path.suffix.lower()
    try:
        low = path.name.lower()
        if suffix in (".cif", ".d12", ".xyz", ".extxyz", ".vasp", ".poscar") or low.startswith(("poscar", "contcar")):
            from mace.preopt import deckscan
            from mace.preopt.cli import read_input
            atoms, sg, kinds, scan = read_input(path)
            if scan and scan.get("geometry_edits"):
                raise CalcError(deckscan.refusal_text(path.name, scan["geometry_edits"]), kind="usage")
            return atoms, sg, kinds
        if suffix == ".traj":
            from ase.io import read as aread
            return aread(str(path), index=frame), None, None
        from mace.gui import structures
        return atoms_from_payload(structures.read_structure(path, frame)), None, None
    except CalcError:
        raise
    except Exception as exc:
        raise CalcError(f"cannot read {path.name}: {(str(exc).strip().splitlines() or [type(exc).__name__])[0]}",
                        kind="usage")
