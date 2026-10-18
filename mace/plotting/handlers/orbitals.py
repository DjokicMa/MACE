"""Orbital (CRYSTAL23 ORBITALS / Molden) plotting handler for mace plotting.

An ORBITALS run leaves ``.molden`` files (one per k point for periodic
systems). This handler picks the orbital, k point and spin, builds a Gaussian
cube in the job's ``<job>_orbitals/`` folder with :mod:`mace.orbitals` (Bloch
lattice sum, CRYSTAL's atom images, the orthonormality self-check), then hands
the cube to the cube handler, so the rendering, formats and publication
styling are the same as for ECH3/POT3 cubes.

Author: Marcus Djokic
Institution: Michigan State University, Mendoza Group
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..prompts import get_string_input, select_option, yes_no_prompt
from .. import outcome
from ..registry import PlotKind, PlotterEntry, register
from . import cube as cube_handler

HARTREE_EV = 27.211386245988
_DEFAULT_CELLS = {0: (1, 1, 1), 1: (3, 1, 1), 2: (2, 2, 1), 3: (1, 1, 1)}


def is_molden(path: str) -> bool:
    try:
        with open(path, "r", errors="replace") as fh:
            head = fh.read(400)
    except OSError:
        return False
    return "[molden format]" in head.lower()


def _runs(files: List[str]) -> List[Path]:
    """One representative file per ORBITALS run (folder + root)."""
    from mace.orbitals.outfile import split_name
    seen, reps = set(), []
    for f in sorted(files):
        p = Path(f)
        key = (str(p.parent.resolve()), split_name(p.name)["root"] or p.stem)
        if key not in seen:
            seen.add(key)
            reps.append(p)
    return reps


def _parse_cells(text) -> Tuple[int, int, int]:
    if text is None:
        return None
    if isinstance(text, (tuple, list)):
        vals = [int(x) for x in text]
    else:
        vals = [int(x) for x in re.split(r"[x,\s]+", str(text).strip().lower()) if x]
    if not vals or any(v < 1 or v > 12 for v in vals):
        raise ValueError(f"cells must be 1-12 per axis, like 2x2x1 (got {text!r})")
    while len(vals) < 3:
        vals.append(1)
    return tuple(vals[:3])


# =============================================================================
# Configuration
# =============================================================================

def configure_orbitals_plot(interactive: bool = True) -> Dict[str, Any]:
    """Orbital choice (resolved per run when interactive) plus the cube view settings."""
    config: Dict[str, Any] = {
        "k": None,              # None -> the VBM's k point when usable, else the first usable one
        "spin": None,           # None -> first channel (ALPHA for open shell)
        "mos": ["HOMO", "LUMO"],
        "part": "re",           # re | im | abs2
        "degenerate_sum": False,
        "cells": None,          # None -> per dimensionality
        "quality": "normal",
        "list_only": False,
        "interactive": interactive,
        "cube": cube_handler.configure_cube_plot(interactive=False),
    }
    return config


def _config_from_args(args) -> Dict[str, Any]:
    cfg = configure_orbitals_plot(interactive=False)
    cfg["cube"] = cube_handler._cube_config_from_args(args)
    if getattr(args, "mo", None):
        cfg["mos"] = [m.strip() for m in str(args.mo).split(",") if m.strip()]
    if getattr(args, "k", None) is not None:
        cfg["k"] = int(args.k)
    if getattr(args, "orb_spin", None):
        cfg["spin"] = args.orb_spin
    if getattr(args, "part", None):
        cfg["part"] = args.part
    cfg["degenerate_sum"] = bool(getattr(args, "degenerate_sum", False))
    if getattr(args, "cells", None):
        cfg["cells"] = _parse_cells(args.cells)
    if getattr(args, "quality", None):
        cfg["quality"] = args.quality
    cfg["list_only"] = bool(getattr(args, "list_orbitals", False))
    return cfg


# =============================================================================
# Level table
# =============================================================================

def _default_k(run) -> Optional[int]:
    usable = [e for e in run.kpoints() if e.usable]
    if not usable:
        return None
    if run.out is not None and run.out.vbm is not None:
        k = run.out.vbm[1]
        if any(e.index == k for e in usable):
            return k
    return usable[0].index


def print_run_summary(run) -> None:
    from mace.orbitals.api import DIM_NAMES, caveats
    dim = run.dimensionality()
    print(f"\n  ORBITALS run {run.root}: {DIM_NAMES.get(dim, dim)}, {len(run.files)} file(s), "
          f"{_size(run.total_bytes())}")
    if not run.molecule:
        print("  k points (from the .out):" if run.out else
              "  k points (no ORBITALS .out next to the files: only Γ is usable):")
        for e in run.kpoints():
            tag = "R" if e.real else "C"
            state = "" if e.usable else f"  -- not available: {e.reason}"
            print(f"    {e.index:3d}  [{tag}] {e.label}{state}")
    for note in caveats(run):
        print(f"  Note: {note}")


def _size(n: int) -> str:
    return f"{n / 1e6:.1f} MB" if n >= 1e6 else f"{n / 1e3:.0f} KB"


def print_levels(kd, spin: str, window: int = 5) -> None:
    """The levels around the HOMO. A metal with E_F in the .out is listed relative to E_F
    (as the GUI does) with CRYSTAL's raw occupation, which is not an electron count there."""
    lv = kd.levels(spin)
    h = kd.homo(spin)
    metal = kd.metallic(spin)
    use_ef = metal and kd.efermi is not None
    if use_ef:
        ref, ref_txt = kd.efermi, "the Fermi level E_F (from the .out)"
    else:
        ref = lv[h - 1].energy if h else 0.0
        ref_txt = "the highest occupied level" if h else "0"
    lo = max(1, (h or 1) - window)
    hi = min(len(lv), (h or 0) + window + 1)
    closed = spin == "ALPHA+BETA"
    print(f"\n  Levels, spin {spin}" + (" (fractional occupations: metal)" if metal else "")
          + f"; energies relative to {ref_txt}:")
    ecol = "E-E_F (eV)  " if use_ef else "E-E_ref (eV)"
    print(f"      #   E (Ha)      {ecol}  {'occupation' if metal else 'electrons'}  ")
    groups = {}
    for l in lv:
        groups.setdefault(l.group, []).append(l.index)
    for l in lv[lo - 1:hi]:
        # a metal's CRYSTAL Occup (bcc Fe: up to 2.13 per spin channel) is printed as is
        e_count = l.occupation if metal else l.occupation * (2 if (closed and kd.max_occ == 1.0) else 1)
        mark = ""
        if h and l.index == h:
            mark = "HOMO" if not use_ef else "<E_F"
        elif h and l.index == h + 1:
            mark = "LUMO" if not use_ef else ">E_F"
        deg = groups[l.group]
        degtxt = f"  degenerate {deg[0]}-{deg[-1]}" if len(deg) > 1 else ""
        print(f"    {l.index:4d}  {l.energy:10.6f}  {(l.energy - ref) * HARTREE_EV:10.3f}    {e_count:5.2f}   "
              f"{mark:4s}{degtxt}")


# =============================================================================
# Rendering
# =============================================================================

def _ask_run_choices(run, kd, config) -> Dict[str, Any]:
    """Interactive step: k point, spin, orbitals, part, degenerate sum, cells."""
    choice = dict(config)
    if not run.molecule:
        usable = [e for e in run.kpoints() if e.usable]
        opts = [f"{e.index}: {e.label} [{'R' if e.real else 'C'}]" for e in usable]
        default = next((i for i, e in enumerate(usable, 1) if e.index == kd.entry.index), 1)
        i = select_option("Which k point?", opts, default=default)
        choice["k"] = usable[i - 1].index
        kd = run.load(choice["k"])
    if len(kd.spins) > 1:
        i = select_option("Which spin channel?", [s.capitalize() for s in kd.spins], default=1)
        choice["spin"] = kd.spins[i - 1]
    spin = choice.get("spin") or kd.spins[0]
    print_levels(kd, spin)
    while True:
        raw = get_string_input("  Orbitals (comma-separated: HOMO, LUMO, HOMO-1, or numbers)", "HOMO,LUMO")
        mos = [m.strip() for m in raw.split(",") if m.strip()]
        try:
            ns = [kd.resolve(spin, m) for m in mos]
            break
        except ValueError as exc:
            print(f"  {exc}")
    choice["mos"] = mos
    degenerate = any(len(kd.degenerate_set(spin, n)) > 1 for n in ns)
    if degenerate:
        print("  Some of these orbitals belong to degenerate sets: a single member depends on an")
        print("  arbitrary mixing inside the set. Summing the set gives a well-defined |ψ|².")
        choice["degenerate_sum"] = yes_no_prompt("  Sum each degenerate set (shows |ψ|²)?", "yes")
    parts = ["Re ψ (signed lobes)", "|ψ|² (density)"] + ([] if kd.entry.real else ["Im ψ (signed lobes)"])
    i = select_option("What to show?", parts, default=1)
    choice["part"] = ["re", "abs2", "im"][i - 1]
    if not run.molecule:
        dflt = "x".join(str(c) for c in _DEFAULT_CELLS.get(run.dimensionality(), (1, 1, 1)))
        while True:
            raw = get_string_input("  Cells to draw along the periodic axes (e.g. 2x2x1)", dflt)
            try:
                choice["cells"] = _parse_cells(raw)
                break
            except ValueError as exc:
                print(f"  {exc}")
    choice["cube"] = cube_handler.configure_cube_plot(interactive=True)
    return choice


def plot_orbitals(files: List[str], config: Dict[str, Any], output_dir: str = ".") -> List[str]:
    """Build the requested orbital cubes for every ORBITALS run in ``files`` and render them."""
    from mace.orbitals import api
    from mace.orbitals.dataset import OrbitalsError
    from mace.orbitals.molden import MoldenError

    outputs: List[str] = []
    for rep in _runs(files):
        try:
            run = api.open_run(rep)
        except (OrbitalsError, MoldenError, OSError) as exc:
            print(f"  {rep.name}: {exc}")
            outcome.failed(rep.name, str(exc))
            continue
        print_run_summary(run)
        k = config.get("k") if config.get("k") is not None else _default_k(run)
        if run.molecule:
            k = 1
        if k is None:
            print("  No usable k point in this run.")
            outcome.failed(rep.name, "no usable k point")
            continue
        try:
            e = run.entry(k)
        except OrbitalsError as exc:
            print(f"  {exc}")
            outcome.failed(rep.name, str(exc))
            continue
        if not e.usable:
            print(f"  k point {k} {e.label} is not available: {e.reason}")
            outcome.failed(rep.name, f"k point {k} is not available")
            continue
        kd = run.load(k)
        if kd.status != "ok":
            worst = max(kd.check.values()) if kd.check else float("nan")
            print(f"  Self-check: max |C^H S C - 1| = {worst:.2g} ({kd.status})")
            if kd.status == "fail":
                print("  The file fails the orthonormality check; not drawing it.")
                outcome.failed(rep.name, "fails the orthonormality check")
                continue
        else:
            print(f"  Self-check passed (max |C^H S C - 1| = {max(kd.check.values()):.1e}).")
        choice = dict(config)
        if config.get("interactive"):
            choice = _ask_run_choices(run, kd, config)
            kd = run.load(choice["k"]) if choice.get("k") else kd
        spin = choice.get("spin") or kd.spins[0]
        try:
            spin = api._spin_alias(kd, spin)
        except OrbitalsError as exc:
            print(f"  {exc}")
            outcome.failed(rep.name, str(exc))
            continue
        if choice.get("list_only"):
            print_levels(kd, spin, window=10**6)
            continue
        part = choice.get("part", "re")
        if part == "im" and kd.entry.real:
            print("  Im ψ is zero at Γ, at real k points and for molecules: showing Re ψ instead.")
            part = "re"
        cells = choice.get("cells") or _DEFAULT_CELLS.get(run.dimensionality(), (1, 1, 1))
        try:
            requests = _requests(kd, spin, choice.get("mos") or ["HOMO"], bool(choice.get("degenerate_sum")))
        except (OrbitalsError, ValueError) as exc:
            print(f"  {exc}")
            outcome.failed(rep.name, str(exc))
            continue
        warn = api.quota_warning(len(requests), 0)
        if warn:
            print(f"  Warning: {warn}")
        cubes, isos = [], []
        for mos in requests:
            try:
                path, meta = api.make_cube(run, kd.entry.index, spin, mos, part=part, cells=cells,
                                           quality=choice.get("quality", "normal"))
            except (OrbitalsError, ValueError) as exc:
                print(f"  {exc}")
                outcome.failed(rep.name, str(exc))
                continue
            if meta.get("cached"):
                print(f"  Reusing {path.parent.name}/{path.name}")
            else:
                print(f"  Wrote {path.parent.name}/{path.name}")
            cubes.append(str(path))
            isos.append(meta.get("isovalue"))
        cube_cfg = dict(choice.get("cube") or cube_handler.configure_cube_plot(interactive=False))
        for path, iso in zip(cubes, isos):
            cfg = dict(cube_cfg)
            if not cfg.get("iso"):
                if iso is None:
                    iso = _isovalue_from_cube(path)
                if iso:
                    v = float(f"{iso:.3g}")
                    # signed Re/Im psi: both lobes (+ blue, - red, as the GUI); |psi|^2: one surface
                    cfg["iso"] = [v] if _is_density(path) else [-v, v]
            outputs.extend(cube_handler.plot_cube([path], cfg, output_dir))
    return outputs


def _is_density(path: str) -> bool:
    return Path(path).stem.endswith("_abs2") or "_abs2_" in Path(path).stem


def _isovalue_from_cube(path: str) -> Optional[float]:
    from mace.orbitals.cube import cube_meta, read_cube
    from mace.orbitals.dataset import default_isovalue
    try:
        _, data, *_rest, c2 = read_cube(path)
    except (OSError, ValueError):
        return None
    meta = cube_meta(c2) or {}
    return default_isovalue(data, meta.get("part", "re") != "abs2")


def _requests(kd, spin, mos, degenerate_sum) -> List[List[int]]:
    """One cube per requested orbital (or per degenerate set when summing)."""
    out: List[List[int]] = []
    for m in mos:
        n = kd.resolve(spin, m)
        req = kd.degenerate_set(spin, n) if degenerate_sum else [n]
        if req not in out:
            out.append(req)
    return out


# =============================================================================
# Registration
# =============================================================================

def register_orbitals_plotter() -> None:
    register(PlotterEntry(
        kind=PlotKind.ORBITALS,
        flag="orbitals",
        label="orbital",
        handler=plot_orbitals,
        configure=configure_orbitals_plot,
        config_from_args=_config_from_args,
        patterns=["*.molden"],
        sniff=is_molden,
        accepts_energy_range=False,
        progress_tmpl="\n  Plotting orbitals from {n} Molden file(s)...",
        not_found_msg="  No ORBITALS (.molden) files found.",
        menu_tmpl="Plot orbitals ({n} Molden files)",
        discovery_label="Orbitals (ORBITALS / Molden)",
    ))


register_orbitals_plotter()
