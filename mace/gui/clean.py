"""Clean geometry: a rough periodic spring pre-relaxation (not physics).

This tidies a hand-built or hand-edited structure before it goes to CRYSTAL:
atoms dropped a little off where they belong move back toward sensible bond
lengths. It knows nothing about bonding beyond covalent radii, so treat the
result as a starting geometry, never as a relaxed structure.

The model (ase calculator ``SpringModel``):

* bonded pairs: the bonds the viewer draws in the starting structure
  (``structures.bond_pairs``, the twin of geom/core.js's bond rule: 1.15 x the
  sum of the bond radii, metal-metal pairs only in a structure of metals only,
  or the document's own rule from Build › Calculate Bonds…). ``bonds_held``
  (Clean › Bonds held) picks them: "shown" (that rule, the default), "lenient"
  (1.2 x the sum of the bond radii, metal-metal pairs in a structure of metals
  only: it also holds bonds stretched a little past what the view draws) or
  "none" (no springs, the repulsion only). The topology is fixed for the run;
  each bond is a harmonic spring toward the sum of the covalent radii;
* every other pair closer than ``rep_scale`` x the sum of the radii pushes
  apart with a soft harmonic repulsion. Bond springs alone leave zero-energy
  motions (6 in conventional Si, 9 in conventional MgO or NaCl, from a rank
  check of the rigidity matrix); the second-neighbor repulsion removes them,
  which ``floppy_modes`` checks;
* FIRE moves the atoms; ``keep_symmetry`` adds ``ase.constraints.FixSymmetry``
  so the space group survives; ``relax_cell`` (off by default) also relaxes
  the cell, which responds to the bond springs only (the repulsion is there
  to stiffen the atom positions; its pressure would blow the cell up).
  Covalent radii underestimate ionic bonds (Na-Cl 2.68 A against
  2.82 A measured), so relaxing the cell shrinks an ionic crystal by ~5 %.

Slabs, polymers and molecules (1.1.6): the atoms keep their ``pbc``, so no
bond or repulsion reaches across a non-periodic direction (its cell vector is
only the drawing box). There is no space group to keep (FixSymmetry is off and
the report says so), and ``relax_cell`` relaxes the periodic directions only
(a strain mask), when they lie along Cartesian axes the way CRYSTAL orients a
SLAB (a, b in the xy plane) or a POLYMER (a along x); a molecule has no cell.

Adapter hooks: a better pre-relaxation is any ase calculator with forces,
and ``ADAPTERS`` maps a method name to a factory ``factory(atoms, **opts) ->
ase Calculator``; ``clean(..., method=name, calc_opts=opts)`` runs it with the
same FIRE loop, FixSymmetry, cell filter and cancel hook as the springs, and
calls the calculator's ``close()`` (when it has one) when the run ends. The
tiers themselves (GFN-FF through the ``xtb`` program, ML potentials in their
own interpreter) live in ``mace.preopt``, which registers them here when
they are installed; the spring model stays the default and the fallback.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from . import structures as S

K_BOND = 10.0       # eV / A^2
K_REP = 3.0         # eV / A^2
REP_SCALE = 1.8     # repulsion range, x the sum of the covalent radii
ADAPTERS: Dict[str, Callable[..., Any]] = {}   # name -> factory(atoms, **opts); see module doc


NOTHING = ("nothing to clean: no bonded pairs (a lone atom, or atoms the bond rule leaves unbonded; "
           "Build › Calculate Bonds… sets the rule)")


class NothingToClean(ValueError):
    """No bonded pairs: a lone atom, or atoms the bond rule leaves unbonded."""


def bonded_pairs(atoms, rule: Optional[Dict[str, Any]] = None) -> List[Tuple[int, int, Tuple[int, int, int]]]:
    """(i, j, image) for each bond, each pair once, as the viewer draws them: ``rule`` is
    ``structures.bond_rule`` of the document's settings (None: the defaults)."""
    return S.bond_pairs(atoms.cell.array, atoms.positions, atoms.numbers, atoms.pbc, **(rule or {}))


# Clean › Bonds held (mace preopt --bonds): the bonds the springs hold
BONDS_HELD = ("shown", "lenient", "none")
LENIENT_TOLERANCE = S.BOND_SCALE      # 1.2: Clean's tolerance before 1.1.6, the bond table's


def check_bonds_held(value: Any) -> str:
    """``value`` as a Bonds held mode ("shown" for None or ""); a ValueError names the modes."""
    v = "shown" if value in (None, "") else str(value)
    if v not in BONDS_HELD:
        raise ValueError(f"bonds held is one of {', '.join(BONDS_HELD)} (not {value!r})")
    return v


def held_bonds(atoms, bonds_held: str = "shown",
               rule: Optional[Dict[str, Any]] = None) -> List[Tuple[int, int, Tuple[int, int, int]]]:
    """The bonds the springs hold under Bonds held: "shown" the view's (``rule``: the document's
    criteria), "lenient" the same rule at 1.2 x the sum of the bond radii with the default
    criteria, "none" no bond."""
    mode = check_bonds_held(bonds_held)
    if mode == "none":
        return []
    if mode == "lenient":
        return S.bond_pairs(atoms.cell.array, atoms.positions, atoms.numbers, atoms.pbc, tolerance=LENIENT_TOLERANCE)
    return bonded_pairs(atoms, rule)


def _calculator_base():
    # ase is optional for the GUI (only the editor needs it): without it the
    # server must still import this module; clean() then fails on its own import
    try:
        from ase.calculators.calculator import Calculator, all_changes
    except ImportError:
        return object, ()
    return Calculator, all_changes


Calculator, _ALL_CHANGES = _calculator_base()


class SpringModel(Calculator):
    """Bond springs + short-range repulsion; energy, forces and stress."""

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, bonds, k_bond=K_BOND, k_rep=K_REP, rep_scale=REP_SCALE, **kw):
        super().__init__(**kw)
        self.bonds = list(bonds)
        self.bond_keys = {(i, j, s) for i, j, s in self.bonds}
        self.bond_keys |= {(j, i, tuple(-x for x in s)) for i, j, s in self.bonds}
        self.k_bond, self.k_rep, self.rep_scale = k_bond, k_rep, rep_scale

    def calculate(self, atoms=None, properties=("energy",), system_changes=_ALL_CHANGES):
        from ase.neighborlist import neighbor_list
        super().calculate(atoms, properties, system_changes)
        a = self.atoms
        pos, cell = a.positions, a.cell.array
        rad = S._covalent(a.numbers)
        n = len(a)
        e = 0.0
        f = np.zeros((n, 3))
        virial = np.zeros((3, 3))

        def pair(i, j, d, r, k, r0, half=False, stress=True):
            nonlocal e, virial
            de = k * (r - r0)                     # dE/dr
            e_ = 0.5 * k * (r - r0) ** 2
            g = de * d / r                        # dE/d(d)
            w = 0.5 if half else 1.0
            e += w * e_
            f[i] += w * g
            f[j] -= w * g
            if stress:
                virial += w * np.outer(g, d)

        if self.bonds:
            ii = np.array([b[0] for b in self.bonds])
            jj = np.array([b[1] for b in self.bonds])
            sh = np.array([b[2] for b in self.bonds], float)
            D = pos[jj] + sh @ cell - pos[ii]
            for (i, j), d in zip(zip(ii, jj), D):
                pair(i, j, d, np.linalg.norm(d), self.k_bond, rad[i] + rad[j])
        ii, jj, SS, D = neighbor_list("ijSD", a, list(self.rep_scale * rad))
        for i, j, s, d in zip(ii, jj, SS, D):
            if (int(i), int(j), tuple(int(x) for x in s)) in self.bond_keys:
                continue
            rc = self.rep_scale * (rad[i] + rad[j])
            r = np.linalg.norm(d)
            if r < rc:     # each pair is listed twice (i->j and j->i)
                pair(i, j, d, r, self.k_rep, rc, half=True, stress=False)
        self.results["energy"] = e
        self.results["forces"] = f
        vol = abs(np.linalg.det(cell))
        stress = virial / vol if vol > 1e-9 else np.zeros((3, 3))
        stress = 0.5 * (stress + stress.T)
        self.results["stress"] = np.array([stress[0, 0], stress[1, 1], stress[2, 2],
                                           stress[1, 2], stress[0, 2], stress[0, 1]])


def floppy_modes(atoms, bonds=None, with_repulsion=True, rep_scale=REP_SCALE) -> int:
    """Zero-energy motions of the springs at this geometry, beyond the 3
    rigid translations: 3N - rank of the rigidity matrix (one row per bond
    and, with ``with_repulsion``, per repelling pair; fixed cell)."""
    from ase.neighborlist import neighbor_list
    bonds = bonds if bonds is not None else bonded_pairs(atoms)
    pos, cell = atoms.positions, atoms.cell.array
    rows = [(i, j, np.asarray(s, float)) for i, j, s in bonds]
    if with_repulsion:
        keys = {(i, j, tuple(s)) for i, j, s in bonds}
        keys |= {(j, i, tuple(-x for x in s)) for i, j, s in bonds}
        rad = S._covalent(atoms.numbers)
        for i, j, sh in zip(*neighbor_list("ijS", atoms, list(rep_scale * rad))):
            key = (int(i), int(j), tuple(int(x) for x in sh))
            if key not in keys and key < (key[1], key[0], tuple(-x for x in key[2])):
                rows.append((int(i), int(j), np.asarray(sh, float)))
    n = len(atoms)
    R = np.zeros((len(rows), 3 * n))
    for k, (i, j, sh) in enumerate(rows):
        d = pos[j] + sh @ cell - pos[i]
        u = d / np.linalg.norm(d)
        R[k, 3 * j:3 * j + 3] += u
        R[k, 3 * i:3 * i + 3] -= u
    rank = np.linalg.matrix_rank(R, tol=1e-6) if len(rows) else 0
    return 3 * n - rank - 3


def periodic_mask(atoms) -> Optional[np.ndarray]:
    """The 3x3 strain mask that relaxes only the periodic directions of a slab or
    polymer, or None when they do not lie along Cartesian axes (or there are none).
    A 3D cell: all ones."""
    pbc = [bool(x) for x in atoms.pbc]
    if all(pbc):
        return np.ones((3, 3))
    if not any(pbc):
        return None
    cell = atoms.cell.array
    axes = set()
    for k in range(3):
        if pbc[k]:
            axes |= {j for j in range(3) if abs(cell[k][j]) > 1e-6}
    if len(axes) != sum(pbc):
        return None
    # the non-periodic vectors must not lean into the periodic plane's axes beyond what their own carry
    for k in range(3):
        if not pbc[k] and any(abs(cell[k][j]) > 1e-6 for j in axes):
            return None
    m = np.zeros((3, 3))
    for i in axes:
        for j in axes:
            m[i, j] = 1.0
    return m


def _cell_filter(atoms, mask=None):
    try:
        from ase.filters import FrechetCellFilter
        return FrechetCellFilter(atoms, mask=mask)
    except ImportError:   # ase < 3.23
        try:
            from ase.filters import ExpCellFilter
        except ImportError:
            from ase.constraints import ExpCellFilter
        return ExpCellFilter(atoms, mask=mask)


def clean(atoms, keep_symmetry: bool = True, relax_cell: bool = False,
          symprec: float = 1e-3, fmax: float = 1e-3, max_steps: int = 2000,
          time_limit: float = 10.0, method: str = "springs",
          should_stop: Optional[Callable[[], bool]] = None,
          calc_opts: Optional[Dict[str, Any]] = None,
          factory: Optional[Callable[..., Any]] = None,
          bond_rule: Optional[Dict[str, Any]] = None, bonds_held: str = "shown") -> Dict[str, Any]:
    """Pre-relax a copy of ``atoms``; returns the new atoms and a report.
    ``bond_rule`` (``structures.bond_rule``): the bonds the springs hold, as the
    document's viewer draws them (None: the default rule); ``bonds_held``
    (held_bonds) picks that rule, the lenient one or none.
    ``should_stop`` (the GUI's cancel check) is called once per FIRE step.
    ``calc_opts`` go to an adapter's factory (never to the spring model).
    ``factory`` is the calculator factory of this one run (mace.preopt passes
    its tier's own), taken over ``ADAPTERS[method]``: runs in other threads
    never share or swap it."""
    from ase.optimize import FIRE

    if len(atoms) == 0:
        raise NothingToClean("the structure has no atoms")
    notes: List[str] = []
    mask = None
    if not all(atoms.pbc):
        dim = S._DIM_NAMES[int(sum(bool(x) for x in atoms.pbc))].lower()
        if keep_symmetry:
            keep_symmetry = False
            notes.append(f"a {dim} has no space group to keep: symmetry was not constrained")
        if relax_cell:
            mask = periodic_mask(atoms)
            if mask is None:
                relax_cell = False
                notes.append(f"the cell was kept: a {dim} has no periodic directions along Cartesian axes to relax"
                             if any(atoms.pbc) else "the cell was kept: a molecule has no cell")
            else:
                notes.append(f"only the periodic directions of the {dim} were relaxed")
    bonds_held = check_bonds_held(bonds_held)
    bonds = held_bonds(atoms, bonds_held, bond_rule)
    if method != "springs":
        factory = factory or ADAPTERS.get(method)
        if factory is None:
            raise ValueError(f"no clean method {method!r} here (only the spring model)")
    elif not bonds and bonds_held != "none":         # None: the repulsion alone, asked for
        raise NothingToClean(NOTHING)
    work = atoms.copy()
    work.calc = (SpringModel(bonds) if method == "springs" else factory(work, **(calc_opts or {})))
    try:
        if keep_symmetry:
            from ase.constraints import FixSymmetry
            work.set_constraint(FixSymmetry(work, symprec=symprec))
        target = _cell_filter(work, mask if mask is not None and not np.all(mask) else None) if relax_cell else work
        opt = FIRE(target, logfile=None)
        start = time.monotonic()
        steps = 0
        converged = False
        e0 = float(work.get_potential_energy())
        for converged in opt.irun(fmax=fmax, steps=max_steps):
            steps += 1
            if converged or time.monotonic() - start > time_limit:
                break
            if should_stop is not None and should_stop():
                break
        work.set_constraint()
        out = work.copy()
        out.calc = None
        return {"atoms": out, "converged": bool(converged), "steps": steps,
                "energy_before": e0, "energy_after": float(work.get_potential_energy()),
                "timed_out": (not converged) and time.monotonic() - start > time_limit,
                "n_bonds": len(bonds), "bonds_held": bonds_held, "method": method,
                **({"notes": notes} if notes else {})}
    finally:
        close = getattr(work.calc, "close", None)
        if callable(close):
            close()
