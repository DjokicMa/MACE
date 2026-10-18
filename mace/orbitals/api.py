"""High-level helpers shared by `mace plotting`, the GUI and the database index."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .basis import CONVENTION
from .cube import cache_dir_for, cached_cube_matches, cube_name, write_cube
from .dataset import HARTREE_EV, OrbitalRun, OrbitalsError, default_isovalue

# Warn before writing more than this in one request (HPCC: 1M-file quota per user).
WARN_CUBE_FILES = 50
WARN_CUBE_BYTES = 500 * 1024 * 1024
CUBE_BYTES_PER_POINT = 13.2        # "%13.5E" plus line breaks

DIM_NAMES = {0: "molecule", 1: "polymer (1D)", 2: "slab (2D)", 3: "crystal (3D)"}


def open_run(path, out_path=None) -> OrbitalRun:
    return OrbitalRun.from_path(path, out_path=out_path)


def describe(run: OrbitalRun, k_index: Optional[int] = None) -> Dict:
    """JSON-ready index of a run: files, k points, and the levels of one k point."""
    dim = run.dimensionality()
    ks = run.kpoints()
    usable = [e for e in ks if e.usable]
    sel = k_index if k_index is not None else (usable[0].index if usable else None)
    info = {
        "root": run.root,
        "dimensionality": dim,
        "system": DIM_NAMES.get(dim, f"{dim}D"),
        "molecule": run.molecule,
        "files": [f.name for f in run.files],
        "n_files": len(run.files),
        "bytes": run.total_bytes(),
        "out": Path(run.out.path).name if run.out and run.out.path else None,
        "kpoints": [e.to_dict() for e in ks],
        "selected": sel,
        "notes": caveats(run),
    }
    if run.out is not None:
        info["electrons"] = run.out.electrons
        info["calc_type"] = run.out.calc_type
        if run.out.vbm:
            info["vbm"] = {"band": run.out.vbm[0], "k": run.out.vbm[1], "e": run.out.vbm[2]}
        if run.out.cbm:
            info["cbm"] = {"band": run.out.cbm[0], "k": run.out.cbm[1], "e": run.out.cbm[2]}
        if run.out.efermi is not None:
            info["efermi"] = run.out.efermi
    if sel is not None:
        kd = run.load(sel)
        info["k"] = kd.summary()
    return info


def caveats(run: OrbitalRun) -> List[str]:
    dim = run.dimensionality()
    notes = []
    if dim == 0:
        notes.append("Molecular orbitals: real, plotted as +/- lobes.")
    else:
        notes.append("Bloch orbitals: at Γ and real (zone-boundary) k points ψ is real up to a sign.")
        if dim >= 2:
            notes.append("Complex k points of 2D/3D systems are not offered: CRYSTAL23's files fail the "
                         "orthonormality check there.")
        else:
            notes.append("Complex k points of polymers are fine: view Re ψ, Im ψ or |ψ|².")
    notes.append("Degenerate orbitals mix arbitrarily; sum the set for a gauge-free |ψ|².")
    return notes


def estimate_cube_bytes(npoints: int, natoms: int = 0) -> int:
    return int(npoints * CUBE_BYTES_PER_POINT + 300 + 60 * natoms)


def quota_warning(n_files: int, n_bytes: int) -> Optional[str]:
    if n_files > WARN_CUBE_FILES or n_bytes > WARN_CUBE_BYTES:
        return (f"this writes {n_files} cube file(s), about {n_bytes / 1e6:.0f} MB; HPCC home and scratch have a "
                f"1M-file quota and cubes are large: ask only for the orbitals you need")
    return None


def make_cube(run: OrbitalRun, k_index: int, spin: str, mos: Sequence, part: str = "re",
              cells=(1, 1, 1), quality: str = "normal", degenerate_sum: bool = False,
              out_dir=None, force: bool = False, max_points: int = 4_000_000) -> Tuple[Path, Dict]:
    """Write (or reuse) the cube for one request in ``<job>_orbitals/``; returns (path, meta)."""
    kd = run.load(k_index)
    sp = spin if spin in kd.spins else _spin_alias(kd, spin)
    ns = sorted({kd.resolve(sp, m) for m in mos})
    if degenerate_sum:
        ns = sorted({x for n in ns for x in kd.degenerate_set(sp, n)})
    eff_part = "abs2" if len(ns) > 1 else part
    cells = tuple(cells) if not run.molecule else (1, 1, 1)
    folder = Path(out_dir) if out_dir else cache_dir_for(run.files[0].parent, run.root)
    name = cube_name(run.root, k_index, sp, ns, eff_part, cells, molecule=run.molecule, quality=quality)
    path = folder / name
    # reuse a cached cube only when it holds these MOs under the current basis convention
    # (cubes written before the spherical-f fix, or for another MO set, are recomputed)
    if path.exists() and not force and cached_cube_matches(path, ns, CONVENTION):
        return path, {"cached": True, "mos": ns, "part": eff_part, "spin": sp}
    grid, data, meta = run.grid(k_index, sp, ns, part=eff_part, cells=cells, quality=quality, max_points=max_points)
    folder.mkdir(parents=True, exist_ok=True)
    Z, pos = run.atoms_bohr(k_index)
    pos, Z = _atoms_in_box(run, kd, grid, Z, pos)
    lv = kd.levels(sp)
    if len(ns) == 1:
        e = lv[ns[0] - 1]
        title = (f"{run.root} {kd.entry.label} {sp} MO {ns[0]} E={e.energy:.6f} Ha "
                 f"({e.energy * HARTREE_EV:.3f} eV) occ={e.occupation:g} {eff_part}")
    else:
        title = f"{run.root} {kd.entry.label} {sp} MOs {ns[0]}-{ns[-1]} sum|psi|^2"
    fields = {"k": kd.entry.index, "kfrac": ",".join(f"{x:.4f}" for x in kd.entry.frac), "spin": sp,
              "mo": "-".join(str(n) for n in ns), "part": eff_part, "cells": "x".join(str(c) for c in cells),
              "e_ha": ",".join(f"{lv[n - 1].energy:.6f}" for n in ns), "check": meta.get("status", kd.status),
              "conv": CONVENTION, "dim": run.dimensionality()}
    write_cube(path, grid, data, Z, pos, title, fields)
    meta.update({"cached": False, "isovalue": default_isovalue(data, eff_part != "abs2")})
    return path, meta


def _spin_alias(kd, spin):
    a = {"A": "ALPHA", "B": "BETA", "ALPHA": "ALPHA", "BETA": "BETA", "AB": "ALPHA+BETA"}.get(str(spin).upper())
    if a in kd.spins:
        return a
    if len(kd.spins) == 1:
        return kd.spins[0]
    raise OrbitalsError(f"spin {spin!r} not in {kd.spins}")


def _atoms_in_box(run, kd, grid, Z, pos):
    """Atoms (and their periodic images) inside the grid box, for drawing."""
    if not kd.basis.periodic:
        return pos, Z
    A = kd.basis.lattice_axes()
    W = np.linalg.inv(grid.vecs)
    ns = kd.basis.cells_within(float(np.linalg.norm(np.array(grid.shape) @ np.abs(grid.vecs))) + 1.0)
    shape = np.array(grid.shape)
    outp, outz = [], []
    for n in ns:
        p = pos + n @ A
        t = (p - grid.origin) @ W
        ok = np.all((t >= -0.5) & (t <= shape - 0.5), axis=1)
        outp.extend(p[ok])
        outz.extend(Z[ok])
    if not outp:
        return pos, Z
    return np.array(outp), np.array(outz)
