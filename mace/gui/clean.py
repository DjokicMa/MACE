"""Clean geometry: a rough periodic spring pre-relaxation (not physics).

This tidies a hand-built or hand-edited structure before it goes to CRYSTAL:
atoms dropped a little off where they belong move back toward sensible bond
lengths. It knows nothing about bonding beyond covalent radii, so treat the
result as a starting geometry, never as a relaxed structure.

The model (ase calculator ``SpringModel``):

* bonded pairs: the pairs closer than ``structures.BOND_SCALE`` x the sum of
  their covalent radii in the starting structure (metal-metal pairs excepted,
  as the viewer draws them). The topology is fixed for the run; each bond is
  a harmonic spring toward the sum of the covalent radii;
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


class NothingToClean(ValueError):
    """No bonded pairs: an all-metal cell or a lone atom."""


def bonded_pairs(atoms, scale: float = S.BOND_SCALE) -> List[Tuple[int, int, Tuple[int, int, int]]]:
    """(i, j, image) for each bond, each pair once, as the viewer draws them."""
    from ase.neighborlist import neighbor_list
    rad = S._covalent(atoms.numbers)
    metal = S._is_metal(atoms.numbers)
    ii, jj, SS = neighbor_list("ijS", atoms, list(scale * rad))
    out = []
    for i, j, sh in zip(ii, jj, SS):
        if metal[i] and metal[j]:
            continue
        key = (int(i), int(j), tuple(int(x) for x in sh))
        if (key[0], key[1], key[2]) < (key[1], key[0], tuple(-x for x in key[2])):
            out.append(key)
    return out


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


def _cell_filter(atoms):
    try:
        from ase.filters import FrechetCellFilter
        return FrechetCellFilter(atoms)
    except ImportError:   # ase < 3.23
        try:
            from ase.filters import ExpCellFilter
        except ImportError:
            from ase.constraints import ExpCellFilter
        return ExpCellFilter(atoms)


def clean(atoms, keep_symmetry: bool = True, relax_cell: bool = False,
          symprec: float = 1e-3, fmax: float = 1e-3, max_steps: int = 2000,
          time_limit: float = 10.0, method: str = "springs",
          should_stop: Optional[Callable[[], bool]] = None,
          calc_opts: Optional[Dict[str, Any]] = None,
          factory: Optional[Callable[..., Any]] = None) -> Dict[str, Any]:
    """Pre-relax a copy of ``atoms``; returns the new atoms and a report.
    ``should_stop`` (the GUI's cancel check) is called once per FIRE step.
    ``calc_opts`` go to an adapter's factory (never to the spring model).
    ``factory`` is the calculator factory of this one run (mace.preopt passes
    its tier's own), taken over ``ADAPTERS[method]``: runs in other threads
    never share or swap it."""
    from ase.optimize import FIRE

    if len(atoms) == 0:
        raise NothingToClean("the structure has no atoms")
    bonds = bonded_pairs(atoms)
    if method != "springs":
        factory = factory or ADAPTERS.get(method)
        if factory is None:
            raise ValueError(f"no clean method {method!r} here (only the spring model)")
    elif not bonds:
        raise NothingToClean("nothing to clean: no bonded pairs (an all-metal cell or a lone "
                             "atom; the spring model draws no metal-metal bonds)")
    work = atoms.copy()
    work.calc = (SpringModel(bonds) if method == "springs" else factory(work, **(calc_opts or {})))
    try:
        if keep_symmetry:
            from ase.constraints import FixSymmetry
            work.set_constraint(FixSymmetry(work, symprec=symprec))
        target = _cell_filter(work) if relax_cell else work
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
                "n_bonds": len(bonds), "method": method}
    finally:
        close = getattr(work.calc, "close", None)
        if callable(close):
            close()
