"""Slabs, polymers and molecules (pbc per axis) for the codes that want a 3D cell.

A non-3D structure keeps its ``pbc``: along a non-periodic axis its cell
vector is only a drawing box (a MOLECULE deck's box can be a few Å; a SLAB
deck's c can be 0.5 Å), so nothing may wrap or see an image there. The
springs, UFF and EQeq honor ``pbc`` themselves. xtb, DFTB+ and some ML
potentials (CHGNet builds a 3D crystal) do not, so ``Boxed`` hands them a
copy whose non-periodic vectors span the atoms plus ``VACUUM`` Å
(``mace.gui.structures.boxed``: same atom order, directions kept), with the
energy, forces and charges passed back unchanged and the stress rescaled to
the real cell's volume (the virial is the same in both cells), so a strain
filter on the periodic directions sees consistent numbers.

``measure`` is the size a relaxation may change: the volume of a 3D cell,
the area of a slab, the length of a polymer, nothing for a molecule.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

VACUUM = 20.0     # Å: more than every cutoff of the ML potentials and GFN-FF's 10 Å minimum

# The ML backends that run a molecule as a molecule: pbc off, its cell (zero for a plain XYZ) unused,
# as mace-torch's and orb-models' neighbor lists follow pbc per axis and fairchem's omol head is for
# molecules. CHGNet and MatterSim build their graph from a 3D lattice, so a molecule reaches them in a
# box with VACUUM Å around it (``Boxed``), as a slab or polymer reaches every ML model.
OPEN_BOUNDARY_BACKENDS = frozenset({"mace_mp", "orb", "fairchem"})


def dimensionality(atoms) -> int:
    return int(sum(bool(x) for x in atoms.pbc))


def is_low_dim(atoms) -> bool:
    return not all(bool(x) for x in atoms.pbc)


def molecule_as_is(model, atoms) -> bool:
    """A molecule (no pbc) for an ML model that runs one without a cell."""
    return not any(bool(x) for x in atoms.pbc) and (model.get("backend") in OPEN_BOUNDARY_BACKENDS
                                                     or bool(model.get("charge_spin")))


def boxed(atoms, vacuum: float = VACUUM):
    from mace.gui.structures import boxed as _boxed
    return _boxed(atoms, vacuum)


def measure(atoms) -> Optional[float]:
    """Volume (3D), area (slab), length (polymer) of the periodic part; None for a molecule."""
    pbc = [bool(x) for x in atoms.pbc]
    cell = atoms.cell.array
    per = [cell[i] for i in range(3) if pbc[i]]
    if len(per) == 3:
        return float(abs(np.linalg.det(cell)))
    if len(per) == 2:
        return float(np.linalg.norm(np.cross(per[0], per[1])))
    if len(per) == 1:
        return float(np.linalg.norm(per[0]))
    return None


def strain_mask(atoms):
    """The 3x3 strain mask of the periodic directions (None: the cell cannot be relaxed)."""
    from mace.gui.clean import periodic_mask
    return periodic_mask(atoms)


def _base():
    try:
        from ase.calculators.calculator import Calculator, all_changes
    except ImportError:
        return object, ()
    return Calculator, all_changes


_Calculator, _ALL = _base()


class Boxed(_Calculator):
    """``inner`` computed on a boxed copy of a slab, polymer or molecule."""

    def __init__(self, inner, vacuum: float = VACUUM, **kw):
        super().__init__(**kw)
        self.inner = inner
        self.vacuum = float(vacuum)
        self.implemented_properties = list(getattr(inner, "implemented_properties", ["energy", "forces"]))
        for k in ("info", "mace_info", "mace_engine", "mace_workdir"):
            if hasattr(inner, k):
                setattr(self, k, getattr(inner, k))

    def calculate(self, atoms=None, properties=("energy",), system_changes=_ALL):
        super().calculate(atoms, properties, system_changes)
        a = self.atoms
        b = boxed(a, self.vacuum)
        b.calc = self.inner
        want = [p for p in properties if p in self.implemented_properties]
        self.inner.calculate(b, want or ["energy"], system_changes)
        r = self.inner.results
        for k in ("energy", "free_energy", "forces", "charges"):
            if k in r:
                self.results[k] = r[k]
        if "stress" in r:
            v_real = abs(np.linalg.det(a.cell.array))
            v_box = abs(np.linalg.det(b.cell.array))
            if v_real > 1e-9:
                self.results["stress"] = np.asarray(r["stress"], float) * (v_box / v_real)

    def __getattr__(self, name):          # calls, ncalls, close ... of the wrapped calculator
        if name in ("inner",):
            raise AttributeError(name)
        return getattr(self.inner, name)

    def close(self) -> None:
        close = getattr(self.inner, "close", None)
        if close is not None:
            close()


class Whole(Boxed):
    """``inner``, a code run without a cell (xtb on a molecule), computed on a
    copy of a molecule in a periodic box made whole across the box's faces
    (lattice translations only, ``mace.gui.structures._unwrap_through_vacuum``)
    with ``pbc`` off: the raw wrapped coordinates would split it. Energy,
    forces and charges come back unchanged (a translation by a lattice vector
    moves no force); a molecule has no stress."""

    def calculate(self, atoms=None, properties=("energy",), system_changes=_ALL):
        _Calculator.calculate(self, atoms, properties, system_changes)
        from mace.gui.structures import _unwrap_through_vacuum
        a = self.atoms
        b = a.copy()
        b.positions = _unwrap_through_vacuum(np.array(a.cell.array, float), np.array(a.positions, float), range(3))
        b.pbc = False
        b.calc = self.inner
        want = [p for p in properties if p in self.implemented_properties and p != "stress"]
        self.inner.calculate(b, want or ["energy"], system_changes)
        r = self.inner.results
        for k in ("energy", "free_energy", "forces", "charges"):
            if k in r:
                self.results[k] = r[k]
