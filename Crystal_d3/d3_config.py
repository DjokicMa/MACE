#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D3 Configuration Management Module
============================================================

Author: Marcus Djokic
Institution: Michigan State University, Mendoza Group

This module provides JSON-based configuration management for CRYSTAL D3 calculations.
It allows saving, loading, and applying configuration settings for various D3 calculation types.

Features:
    - Save D3 calculation settings to JSON files
    - Load settings from JSON files for batch processing
    - Support for all D3 calculation types (BAND, DOSS, ECH3, POT3, etc.)
    - Validate configuration completeness and compatibility
    - Default configurations for common calculation types

Usage:
    # Save configuration to file
    save_d3_config(config_dict, "my_band_config.json")
    
    # Load configuration from file
    config = load_d3_config("my_band_config.json")
    
    # Apply configuration to D3 generation
    d3_generator = CRYSTALOptToD3(...)
    d3_generator.apply_config(config)
"""

import json
import os
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path


# The keys of an ORBITALS configuration. kmesh: band_edges (default), gamma,
# all_real or custom (with "is"; a 2D/3D IS with complex k points is refused:
# CRYSTAL23's complex-k files there are unusable). ICAR and ILOC are not
# settings: ICAR has no effect on the files and ILOC is always 0.
ORBITALS_CONFIG_KEYS = ("calculation_type", "kmesh", "is")
ORBITALS_KMESH_MODES = ("band_edges", "gamma", "all_real", "custom")

# DOSS energy points (NPT, the second number of the DOSS record). CRYSTAL
# spreads them evenly from the bottom of band INZB to the top of band IFNB, or
# over BMI..BMA (CRYSTAL23 manual, DOSS, p.320, which sets no upper limit).
# The setting is the energy step: NPT = ceil(span / step) within
# DOSS_NPTS_RANGE, unless a number of points is given (honored as it is).
DOSS_DEFAULT_STEP_EV = 0.02          # the Standard resolution: the one line to change the default
DOSS_STEP_PRESETS = {"screening": 0.2, "standard": DOSS_DEFAULT_STEP_EV, "fine": 0.005}
DOSS_RESOLUTIONS = ("screening", "standard", "fine", "step", "points")
# CRYSTAL's real maximum: the manual sets no limit, but DOSS.DAT's header prints NPT + 2 in five
# digits; probed on the MSU HPCC (CRYSTAL23, diamond, 2026-10-23) 60000 points ran and 200000
# wrote "NEPTS *****", which no reader takes. The step rule goes up to it, and a number of
# points is honored up to it.
DOSS_MAX_NPTS = 99997
DOSS_NPTS_RANGE = (1000, DOSS_MAX_NPTS)   # the step rule's least and most
# above this many points (an all-bands range: core levels to the highest virtual bands) the deck covers a
# window around E_F instead, from the same step, on every route (Marcus, 2026-10-23); "all_bands": True
# (opt2d3's No, --doss-window all) keeps every band
DOSS_WINDOW_OFFER_NPTS = 5000
# the window, eV relative to E_F: the whole valence band of a carbon or oxide material (diamond's is
# about 21 eV wide) and the low conduction bands, without the core levels and the high virtual bands;
# also the default of opt2d3's "Energy window around Fermi level". The default of a 3D or 2D material; a
# 1D one takes bands by index instead (doss_band_index_range, below)
DOSS_DEFAULT_WINDOW_EV = (-20.0, 15.0)
# coarser than this, the DOS-derived band-edge properties (DOS effective masses,
# band-edge character) lose the edge: warned
DOSS_COARSE_STEP_EV = 0.05
# A band range's span is first printed by the DOSS run (the SCF output and
# fort.9 hold no band energies), so it is estimated from the atoms: from the
# deepest core level (~ -(Z - 0.75 Z^0.47)^2 / 2 Hartree, the 1s level; an ECP
# atom's lowest semicore band ~ -5 Hartree) to the highest virtual bands
# (~ +20 Hartree). The example suite's all-bands runs: C -10.1 (estimate -9.1),
# N -14.3 (-13.2), Si -66.2 (-65.1) Hartree; tops 15.3-25.6 Hartree.
DOSS_TOP_BAND_HA = 20.0
DOSS_ECP_BOTTOM_HA = -5.0
_HARTREE_EV = 27.211386245988


def doss_core_level_ha(z: int) -> float:
    """The approximate lowest band of an atom (Hartree): its 1s level, or an ECP's semicore band (Z > 200)."""
    z = int(z)
    if z > 200:
        return DOSS_ECP_BOTTOM_HA
    z = z % 100 if z > 100 else z            # CRYSTAL's 1xx: the same element with another basis
    return -0.5 * (z - 0.75 * z ** 0.47) ** 2 if z > 0 else 0.0


def doss_span_estimate_ev(atomic_numbers) -> Optional[float]:
    """The estimated span (eV) of all bands for the atoms' conventional atomic numbers (CRYSTAL's
    geometry listing: 6 for C, 247 for Ag with an ECP); None without atoms."""
    zs = [int(z) for z in atomic_numbers or () if int(z) > 0]
    if not zs:
        return None
    bottom = min(min(doss_core_level_ha(z) for z in zs), -0.5)
    return (DOSS_TOP_BAND_HA - bottom) * _HARTREE_EV


def doss_atomic_numbers(out_text: str) -> List[int]:
    """The atoms' conventional atomic numbers from the first geometry listing of a CRYSTAL output
    ("   1 T   6 C ...", "   1 T 247 AG ..." for an ECP atom)."""
    import re
    zs, last = [], 0
    for m in re.finditer(r"^\s*(\d+)\s+[TF]\s+(\d+)\s+[A-Z][A-Za-z]?\s+", out_text or "", re.MULTILINE):
        idx = int(m.group(1))
        if idx <= last:
            break
        zs.append(int(m.group(2)))
        last = idx
    return zs


def doss_span(config: Optional[Dict[str, Any]], atomic_numbers=None) -> Tuple[Optional[float], bool]:
    """(span in eV, estimated) of a DOSS configuration: an energy window's width (Hartree in the
    configuration), else the estimate for all bands from the atoms (an upper bound for a band range)."""
    win = (config or {}).get("energy_window")
    if win:
        return abs(float(win[1]) - float(win[0])) * _HARTREE_EV, False
    return doss_span_estimate_ev(atomic_numbers), True


def doss_resolution_menu(span_ev: Optional[float], estimated: bool = True) -> List[Tuple[str, str]]:
    """The resolution choices (name, text) with the points and step each gives for this span."""
    def grid(step):
        n = doss_npts(span_ev, None, step)
        if not span_ev:
            return f"{n} points"
        about = "~" if estimated else ""
        return f"{n} points, {about}{span_ev / n:.3g} eV apart"
    return [("screening", f"Screening - {DOSS_STEP_PRESETS['screening']} eV step ({grid(DOSS_STEP_PRESETS['screening'])})"),
            ("standard", f"Standard - {DOSS_STEP_PRESETS['standard']} eV step ({grid(DOSS_STEP_PRESETS['standard'])})"),
            ("fine", f"Fine - {DOSS_STEP_PRESETS['fine']} eV step ({grid(DOSS_STEP_PRESETS['fine'])})"),
            ("step", "Step - another energy step (eV)"),
            ("points", "Points - a number of energy points, as given (e.g. 1000 to screen many structures)")]


def doss_step_ev(config: Optional[Dict[str, Any]] = None) -> float:
    """The energy step a DOSS configuration asks for: its resolution's preset (screening, standard,
    fine), else energy_step_ev, else the Standard step."""
    config = config or {}
    name = str(config.get("energy_resolution") or "").lower()
    if name in DOSS_STEP_PRESETS:
        return DOSS_STEP_PRESETS[name]
    step = config.get("energy_step_ev")
    if step not in (None, "", "auto"):
        step = float(step)
        if step > 0:
            return step
    return DOSS_DEFAULT_STEP_EV


def doss_explicit_points(config: Optional[Dict[str, Any]] = None) -> Optional[int]:
    """The number of energy points a configuration names (an int), else None ("auto", absent)."""
    n = (config or {}).get("n_points")
    if n is None or str(n).strip().lower() in ("", "auto"):
        return None
    return int(n)


def doss_npts(span_ev: Optional[float], n_points: Any = None, step: Optional[float] = None) -> int:
    """The NPT of a DOSS deck. An explicit n_points is honored as given (a screening run may want
    1000 points over hundreds of eV). Otherwise ceil(span / step), the Standard step by default,
    within DOSS_NPTS_RANGE; the most points when the span is not known."""
    if n_points is not None and str(n_points).strip().lower() not in ("", "auto"):
        return int(n_points)
    import math
    lo, hi = DOSS_NPTS_RANGE
    step = float(step) if step else DOSS_DEFAULT_STEP_EV
    try:
        span = float(span_ev)
    except (TypeError, ValueError):
        return hi
    if not math.isfinite(span) or span <= 0 or step <= 0:
        return hi
    return min(max(math.ceil(span / step - 1e-9), lo), hi)


def doss_grid_note(span_ev: Optional[float], npts: int, explicit: bool, estimated: bool) -> Tuple[str, Optional[str]]:
    """What the deck's grid is (one line) and a warning when its step is coarser than DOSS_COARSE_STEP_EV."""
    if not span_ev:
        return f"{npts} energy points", None
    step = span_ev / npts
    about = "~" if estimated else ""
    line = f"{npts} energy points, {about}{step:.3g} eV apart over {about}{span_ev:.0f} eV"
    if step <= DOSS_COARSE_STEP_EV:
        return line, None
    why = (f"the {npts} points given give" if explicit else
           f"{npts} points, the most MACE writes, give" if npts >= DOSS_NPTS_RANGE[1] else "the chosen step gives")
    return line, (f"{why} a {about}{step:.2g} eV energy step: the DOS effective masses and band-edge "
                  f"character need {DOSS_COARSE_STEP_EV} eV or finer, and are skipped or unreliable for this run")


def doss_window_offer(span_ev: Optional[float], npts: int, step: float) -> Optional[str]:
    """The warning that offers a window around E_F: when an all-bands grid at this step needs more than
    DOSS_WINDOW_OFFER_NPTS points; None otherwise."""
    if not span_ev or npts <= DOSS_WINDOW_OFFER_NPTS:
        return None
    return (f"This DOS needs {npts} points for a {step:g} eV step over all bands (~{span_ev:.0f} eV, "
            f"including core levels)")


def doss_window_ha(lo_ev: float, hi_ev: float) -> Tuple[float, float]:
    """An energy window in eV relative to E_F as the configuration holds it (Hartree relative to E_F;
    the deck writer adds E_F: CRYSTAL's BMI/BMA are absolute)."""
    return float(lo_ev) / _HARTREE_EV, float(hi_ev) / _HARTREE_EV


# The default by periodic dimensionality (above DOSS_WINDOW_OFFER_NPTS points over all bands). CRYSTAL
# needs an energy window's BMI and BMA in band gaps: on the MSU HPCC (CRYSTAL23, 2026-10-23) a 1D
# polyacetylene DOSS whose BMA fell inside a band stopped with "double free or corruption" (with NPRO 0
# too), while a BMA in a gap and the same bands by index ran and agreed with the all-bands DOS within 0.7%.
# 3D and 2D: the window DOSS_DEFAULT_WINDOW_EV, its bounds moved outward into the nearest gaps when a BAND
# run of the same SCF gives the band extents (doss_snap_window). 1D: bands by index, never energies
# (doss_band_index_range): from the first band above the core to the conduction bands. 0D: no DOSS.
# How far a moved bound stays from a band edge (a BAND path need not reach a band's extreme on the DOSS
# k mesh), and at most how far into a gap it goes
DOSS_SNAP_MARGIN_EV = 0.2
DOSS_SNAP_MAX_EV = 1.0
# a bound is moved only to a gap this close: a 3D material's conduction bands overlap into one continuum
# (diamond's from E_F+6 to E_F+175 eV), and the suite's 3D and 2D windows ran with bounds inside bands
DOSS_SNAP_REACH_EV = 5.0
# The band-index form's span, for its NPT: a bound on the valence and conduction bands it covers. The
# suite's valence + as many conduction bands measured 36.1 (polyacetylene) to 48.8 eV (SOD; BWIDTH,
# HPCC 2026-10-23); a BAND run of the SCF beside it gives the real span (+ DOSS_SNAP_MAX_EV)
DOSS_BAND_INDEX_SPAN_EV = 50.0
# at least this many conduction bands by default (a material with few valence orbitals)
DOSS_MIN_CONDUCTION_BANDS = 4
# a band-index DOS should reach this far above E_F (the window's top): the post-run note says when not
DOSS_REACH_EV = DOSS_DEFAULT_WINDOW_EV[1]
_NOBLE_GAS_Z = (2, 10, 18, 36, 54, 86, 118)


def crystal_core_electrons(z: int) -> int:
    """The core electrons CRYSTAL counts for an all-electron atom ("CORE ELECTRONS PER CELL"): the noble
    gas core, plus the filled d shell of a p-block element after a transition series and the 4f shell
    from Hf on (Br: 28; the test corpus's outputs agree for H to Pb)."""
    z = int(z) % 100 if int(z) > 100 else int(z)
    core = max([n for n in _NOBLE_GAS_Z if n < z] or [0])
    if 31 <= z <= 36 or 49 <= z <= 54 or 81 <= z <= 86 or 113 <= z <= 118:
        core += 10
    if 72 <= z <= 86 or z >= 104:
        core += 14
    return core


def valence_shell_orbitals(z: int) -> int:
    """The minimal valence shell's orbitals of an atom: 1 for H and He, s + p for a main-group atom, s + p
    + d for a transition metal, + f for a lanthanide or actinide."""
    z = int(z) % 100 if int(z) > 100 else int(z)
    if z <= 2:
        return 1
    if 58 <= z <= 70 or 90 <= z <= 102:
        return 16
    if 21 <= z <= 30 or 39 <= z <= 48 or 57 <= z <= 80 or 89 <= z <= 112:
        return 9
    return 4


def doss_band_counts(out_text: str) -> Optional[Dict[str, int]]:
    """{core, occupied, n_ao, ...} bands per cell of a CRYSTAL SCF output, None when it prints no counts.
    core is "CORE ELECTRONS PER CELL" / 2 plus, for an ECP atom (CRYSTAL counts none of its electrons as
    core), the semicore orbitals its pseudopotential keeps beyond the valence shell (Ag with 19
    electrons: 4s 4p; Pb with 22: 5s 5p 5d), so the bands start where they would all-electron."""
    import re
    import math
    text = out_text or ""
    ao = re.search(r"NUMBER OF AO\s+(\d+)", text)
    nel = re.search(r"N\. OF ELECTRONS PER CELL\s+(\d+)", text)
    core = re.search(r"CORE ELECTRONS PER CELL\s+(\d+)", text)
    if not (ao and nel and core):
        return None
    charges = {int(z): float(q) for z, q in re.findall(r"ATOMIC NUMBER\s+(\d+), NUCLEAR CHARGE\s+([\d.]+)", text)}
    zs = doss_atomic_numbers(text)
    semicore = 0
    for zc in zs:
        if zc > 200 and (zc % 100) in charges:
            z = zc % 100
            semicore += max(0, int(round(charges[z])) - (z - crystal_core_electrons(z))) // 2
    return {"core": int(core.group(1)) // 2 + semicore, "occupied": math.ceil(int(nel.group(1)) / 2),
            "n_ao": int(ao.group(1)), "semicore": semicore,
            "valence_orbitals": sum(valence_shell_orbitals(z) for z in zs)}


def doss_conduction_bands(config: Optional[Dict[str, Any]]) -> Optional[int]:
    """The conduction bands a configuration asks for (conduction_bands, --doss-conduction-bands): an int,
    else None (auto)."""
    n = (config or {}).get("conduction_bands")
    if n is None or str(n).strip().lower() in ("", "auto"):
        return None
    return int(n)


def doss_band_index_range(counts: Dict[str, int], conduction: Optional[int] = None) -> Tuple[int, int]:
    """(first, last) band of the band-index DOS (the 1D default): from the first band above the core to
    the highest occupied band + ``conduction`` bands; by default to the highest of the minimal valence
    shells' bands, the valence bands' count again above them, and DOSS_MIN_CONDUCTION_BANDS (polyacetylene:
    3-12; diamond, h-BN, graphene: 3-10; SOD: 43-138), never past the last band."""
    core, occ, n_ao = counts["core"], counts["occupied"], counts["n_ao"]
    first = core + 1
    if conduction is not None:
        last = occ + int(conduction)
    else:
        last = max(core + counts.get("valence_orbitals", 0), occ + (occ - core), occ + DOSS_MIN_CONDUCTION_BANDS)
    last = min(n_ao, last) if n_ao > 0 else last
    return first, max(last, first)


def doss_snap_window(bmi: float, bma: float, extents: List[Tuple[float, float]],
                     fermi: float = 0.0) -> Tuple[float, float, List[str]]:
    """BMI and BMA (Hartree, absolute) moved outward into the nearest gaps of the bands ``extents`` (each
    band's (lowest, highest) energy, Hartree, absolute), at least DOSS_SNAP_MARGIN_EV from a band edge and
    at most DOSS_SNAP_MAX_EV into the gap, and only to a gap within DOSS_SNAP_REACH_EV; never inward.
    Returns (bmi, bma, each bound inside bands: moved or kept, and why, in eV from ``fermi``)."""
    margin, most = DOSS_SNAP_MARGIN_EV / _HARTREE_EV, DOSS_SNAP_MAX_EV / _HARTREE_EV

    def ev(e):
        return f"E_F{(e - fermi) * _HARTREE_EV:+.2f} eV"
    merged: List[List[float]] = []
    for lo, hi in sorted((min(a, b), max(a, b)) for a, b in extents):
        if merged and lo <= merged[-1][1] + 2 * margin:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    moved = []
    reach = DOSS_SNAP_REACH_EV / _HARTREE_EV
    for k, (lo, hi) in enumerate(merged):
        if lo - margin < bmi < hi + margin:
            gap = lo - merged[k - 1][1] if k else float("inf")
            new = lo - min(max(gap / 2, margin), most)
            if bmi - new > reach:
                moved.append(f"the lower bound {ev(bmi)} is inside the bands from {ev(lo)} to {ev(hi)}, with no "
                             f"gap within {DOSS_SNAP_REACH_EV:g} eV below it: kept")
            else:
                moved.append(f"the lower bound {ev(bmi)} was inside the bands from {ev(lo)} to {ev(hi)}: "
                             f"moved down to {ev(new)}, in the gap below them")
                bmi = new
        if lo - margin < bma < hi + margin:
            gap = merged[k + 1][0] - hi if k + 1 < len(merged) else float("inf")
            new = hi + min(max(gap / 2, margin), most)
            if new - bma > reach:
                moved.append(f"the upper bound {ev(bma)} is inside the bands from {ev(lo)} to {ev(hi)}, with no "
                             f"gap within {DOSS_SNAP_REACH_EV:g} eV above it: kept")
            else:
                moved.append(f"the upper bound {ev(bma)} was inside the bands from {ev(lo)} to {ev(hi)}: "
                             f"moved up to {ev(new)}, in the gap above them")
                bma = new
    return bmi, bma, moved


# TRANSPORT's k mesh (the NEWK before BOLTZTRA). The transport distribution sums band velocities over
# the NEWK k points on 0.01 eV bins (CRYSTAL23 manual, eq. 14.4, p.314), so the SCF's mesh, which
# converges energies, leaves it spiky. The mesh is chosen by a k-point spacing (1/Angstrom, 2*pi
# included) along the periodic reciprocal vectors: IS = the largest ceil(|b_i| / spacing), never below
# the parent's IS; ISP = 2 IS, as for DOSS. The presets depend on the periodic dimensionality: measured
# on the HPCC (2026-10-23) for the example suite's five materials against their densest run, at 300 K
# over the chemical potentials where σ is at least 1e-3 of its peak, Standard is the coarsest spacing
# with σ within 5% and S within 5 µV/K (mean): 1D polyacetylene IS 288 (0.0089), 2D h-BN IS 128-192
# and graphene IS 192 (0.0154), 3D SOD IS 10 (0.10) and diamond IS 48 (0.064, its densest affordable
# run: IS 36 was 10% off in σ). A 1D or 2D run takes seconds even then; a 3D one minutes (diamond IS 48
# under 2 with BOLTZTRA's fort.70 scratch files on the node's disk: mace/submission/submit_prop.sh).
TRANSPORT_K_STANDARD = {3: 0.064, 2: 0.015, 1: 0.009}    # 1/Angstrom, by periodic dimensionality: the default
TRANSPORT_K_FINE = {3: 0.045, 2: 0.010, 1: 0.006}
TRANSPORT_K_PRESETS = {"screening": None, "standard": TRANSPORT_K_STANDARD, "fine": TRANSPORT_K_FINE}
TRANSPORT_K_RESOLUTIONS = ("screening", "standard", "fine", "spacing", "is")


def _spacing_for(value, dim: int) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, dict):
        return float(value.get(int(dim), value[3]))
    return float(value)


def reciprocal_lengths(out_file: str) -> Optional[List[float]]:
    """|b_i| (1/Angstrom, 2*pi included) of the output's periodic directions (3D, 2D or 1D); None without a cell."""
    try:
        import numpy as np
        try:
            import d3_bz
        except ImportError:
            from Crystal_d3 import d3_bz
        st = d3_bz.structure_of(str(out_file))
    except Exception:
        return None
    if st is None or not st.get("dim"):
        return None
    b = 2 * np.pi * np.linalg.inv(np.asarray(st["cell"], dtype=float)).T
    return [float(x) for x in np.linalg.norm(b, axis=1)[:int(st["dim"])]]


def transport_k_spacing(config: Optional[Dict[str, Any]], dim: int = 3) -> Optional[float]:
    """The k spacing a TRANSPORT configuration asks for at this periodic dimensionality (None: the parent's
    mesh, or an explicit IS)."""
    c = config or {}
    name = str(c.get("k_resolution") or "standard").lower()
    if name in TRANSPORT_K_PRESETS:
        return _spacing_for(TRANSPORT_K_PRESETS[name], dim)
    if name == "spacing" and c.get("k_spacing"):
        return float(c["k_spacing"])
    return None if name == "is" else _spacing_for(TRANSPORT_K_STANDARD, dim)


def transport_standard_is(recip: Optional[List[float]]) -> Optional[int]:
    """The Standard IS of a cell (its periodic |b_i|), or None without one."""
    import math
    if not recip:
        return None
    sp = _spacing_for(TRANSPORT_K_STANDARD, len(recip))
    return max(math.ceil(b / sp - 1e-9) for b in recip)


def transport_mesh(config: Optional[Dict[str, Any]], parent_is: int,
                   recip: Optional[List[float]]) -> Tuple[int, int, str]:
    """(IS, ISP, why) of a TRANSPORT deck's NEWK: an explicit IS as given; else the spacing's IS (the
    largest ceil(|b_i| / spacing) over the periodic directions), never below the parent's IS."""
    import math
    c = config or {}
    name = str(c.get("k_resolution") or "standard").lower()
    if name == "is" and c.get("newk_is"):
        n = int(c["newk_is"])
        why = f"IS {n} as given" + (f" (below the parent's IS {parent_is})" if n < parent_is else "")
        return n, 2 * n, why
    spacing = transport_k_spacing(c, len(recip) if recip else 3)
    if spacing is None:
        return parent_is, 2 * parent_is, f"the parent SCF's mesh (IS {parent_is}, Screening)"
    if not recip:
        return parent_is, 2 * parent_is, (f"the parent SCF's mesh (IS {parent_is}): no cell in the output to "
                                          f"apply the {spacing:g} 1/Å spacing to")
    need = max(math.ceil(b / spacing - 1e-9) for b in recip)
    n = max(need, parent_is)
    label = name.capitalize() if name in TRANSPORT_K_PRESETS else "spacing"
    why = (f"{label}: {spacing:g} 1/Å or finer along |b| = {', '.join(f'{b:.3f}' for b in recip)} 1/Å"
           + (f" (IS {need}; the parent's IS {parent_is} is finer)" if parent_is > need else
              f"; the parent SCF's IS is {parent_is}"))
    return n, 2 * n, why


# --transport-k / --transport-is of mace opt2d3
_TRANSPORT_OVERRIDE: Dict[str, Any] = {}


def set_transport_override(k: Any = None, newk_is: Any = None) -> Dict[str, Any]:
    _TRANSPORT_OVERRIDE.clear()
    if newk_is is not None:
        _TRANSPORT_OVERRIDE.update(k_resolution="is", newk_is=int(newk_is))
    elif k is not None:
        name = str(k).strip().lower()
        if name in TRANSPORT_K_PRESETS:
            _TRANSPORT_OVERRIDE.update(k_resolution=name)
        else:
            _TRANSPORT_OVERRIDE.update(k_resolution="spacing", k_spacing=float(k))
    return dict(_TRANSPORT_OVERRIDE)


def transport_override() -> Dict[str, Any]:
    return dict(_TRANSPORT_OVERRIDE)


def apply_transport_override(config: Dict[str, Any]) -> Dict[str, Any]:
    if not _TRANSPORT_OVERRIDE:
        return config
    out = {k: v for k, v in config.items() if k not in ("k_resolution", "k_spacing", "newk_is")}
    out.update(_TRANSPORT_OVERRIDE)
    return out


def transport_k_menu(parent_is: Optional[int], recip: Optional[List[float]]) -> List[Tuple[str, str]]:
    """The TRANSPORT k-mesh choices (name, text), each with the IS it gives for this output."""
    def mesh(name):
        if parent_is is None:
            return ""
        n, _, _ = transport_mesh({"k_resolution": name}, parent_is, recip)
        return f", IS {n}"
    dim = len(recip) if recip else 3
    std, fine = _spacing_for(TRANSPORT_K_STANDARD, dim), _spacing_for(TRANSPORT_K_FINE, dim)
    return [("screening", f"Screening - the parent SCF's mesh{mesh('screening')}"),
            ("standard", f"Standard - {std:g} 1/Å k spacing{mesh('standard')}"),
            ("fine", f"Fine - {fine:g} 1/Å k spacing{mesh('fine')}"),
            ("spacing", "Spacing - another k spacing (1/Å)"),
            ("is", "IS - a shrinking factor, as given")]


def doss_keeps_all_bands(config: Optional[Dict[str, Any]]) -> bool:
    """The configuration opts out of the default window around E_F (opt2d3's No, --doss-window all;
    all_bands_confirmed: the 1.1.6 window branch's name)."""
    c = config or {}
    return bool(c.get("all_bands") or c.get("all_bands_confirmed"))


# --doss-step / --doss-points / --doss-window of mace opt2d3: the resolution of every DOSS deck of the run
_DOSS_OVERRIDE: Dict[str, Any] = {}
# --non-interactive: no question is asked (the window default applies as it does on every route)
_NON_INTERACTIVE = {"on": False}


def set_non_interactive(on: bool = True) -> None:
    _NON_INTERACTIVE["on"] = bool(on)


def non_interactive() -> bool:
    return _NON_INTERACTIVE["on"]


def set_doss_override(step: Any = None, points: Any = None, window: Any = None,
                      conduction: Any = None) -> Dict[str, Any]:
    """Set the run's DOSS resolution (a step in eV or a preset name, or a number of points), window
    (eV relative to E_F) and conduction bands (a 1D material's bands by index)."""
    _DOSS_OVERRIDE.clear()
    if conduction is not None:
        _DOSS_OVERRIDE["conduction_bands"] = int(conduction)
    if window is not None and str(window[0]).strip().lower() == "all":
        _DOSS_OVERRIDE["all_bands"] = True
    elif window is not None:
        _DOSS_OVERRIDE["energy_window"] = doss_window_ha(*window)
    if points is not None:
        _DOSS_OVERRIDE.update(energy_resolution="points", n_points=int(points))
    elif step is not None:
        name = str(step).strip().lower()
        if name in DOSS_STEP_PRESETS:
            _DOSS_OVERRIDE.update(energy_resolution=name, n_points="auto")
        else:
            _DOSS_OVERRIDE.update(energy_resolution="step", energy_step_ev=float(step), n_points="auto")
    return dict(_DOSS_OVERRIDE)


def doss_override() -> Dict[str, Any]:
    return dict(_DOSS_OVERRIDE)


def apply_doss_override(config: Dict[str, Any]) -> Dict[str, Any]:
    """The configuration with the run's --doss-step / --doss-points / --doss-window /
    --doss-conduction-bands applied (a copy when it changes)."""
    if not _DOSS_OVERRIDE:
        return config
    out = dict(config)
    if set(_DOSS_OVERRIDE) - {"energy_window", "all_bands", "conduction_bands"}:
        out.pop("energy_step_ev", None)
    if "energy_window" in _DOSS_OVERRIDE:
        out.pop("band_range", None)
        out.pop("all_bands", None)
    if "all_bands" in _DOSS_OVERRIDE:
        out.pop("energy_window", None)
    out.update(_DOSS_OVERRIDE)
    return out


def clean_config_for_saving(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean configuration for saving by preserving 'auto' settings.
    
    Removes calculated values that should be recalculated for each material,
    keeping only the settings that indicate 'auto' behavior.
    
    Args:
        config: Original configuration dictionary
        
    Returns:
        dict: Cleaned configuration suitable for saving
    """
    import copy
    cleaned = copy.deepcopy(config)
    
    calc_type = cleaned.get("calculation_type", "")
    
    if calc_type == "BAND":
        # For BAND calculations, preserve auto indicators
        if cleaned.get("auto_path", False):
            # Remove calculated path values, keep only the auto indicators
            cleaned["path"] = "auto"
            cleaned["labels"] = "auto"
            if "path_labels" in cleaned:
                del cleaned["path_labels"]
            if "segments" in cleaned:
                del cleaned["segments"]
            # If path was auto, shrink should be auto too (unless it's labels mode)
            if cleaned.get("path_method") == "coordinates":
                cleaned["shrink"] = "auto"
        
        # Keep shrink as "auto" if it was auto-detected
        if cleaned.get("shrink") == "auto" or cleaned.get("auto_shrink", False):
            cleaned["shrink"] = "auto"
        
        # Keep bands as "auto" if it was auto-detected
        if cleaned.get("bands") == "auto" or cleaned.get("auto_bands", False):
            cleaned["bands"] = "auto"
            if "band_range" in cleaned and cleaned.get("bands") == "auto":
                del cleaned["band_range"]
            # Remove specific band numbers when using auto
            if "first_band" in cleaned:
                del cleaned["first_band"]
            if "last_band" in cleaned:
                del cleaned["last_band"]
    
    elif calc_type == "DOSS":
        # For DOSS, if projections were auto-generated, don't save them
        if cleaned.get("project_orbital_types", False):
            # Remove the specific calculated projections
            if "projections" in cleaned:
                cleaned["projections"] = []
            # Set projection_type based on the configuration
            if cleaned.get("element_only", False):
                cleaned["projection_type"] = 2
            elif cleaned.get("include_element_totals", True):
                cleaned["projection_type"] = 3
            else:
                cleaned["projection_type"] = 4
        elif cleaned.get("project_all_atoms", False):
            cleaned["projection_type"] = 5
            # Remove specific atom projections if using all atoms
            if "project_atoms" in cleaned:
                del cleaned["project_atoms"]
        elif "project_atoms" in cleaned:
            cleaned["projection_type"] = 5
        elif "npro" in cleaned and cleaned["npro"] == 0:
            cleaned["projection_type"] = 1
        
        # For manual projections (type 6), keep the projections
        if cleaned.get("projection_type") == 6:
            # Keep manual projections as-is
            pass
    
    elif calc_type == "ORBITALS":
        # The k mesh is chosen per material from the mode (band edges, Γ, all
        # real points): only the mode and a custom IS are kept.
        for key in [k for k in cleaned if k not in ORBITALS_CONFIG_KEYS]:
            del cleaned[key]

    elif calc_type == "TRANSPORT":
        # For TRANSPORT, if using Fermi reference, save relative range
        if cleaned.get("mu_reference") == "fermi":
            # Keep the relative range, remove absolute values
            if "mu_relative_range" in cleaned:
                mu_min_rel, mu_max_rel = cleaned["mu_relative_range"]
                mu_step = cleaned.get("mu_range", (0, 0, 0.01))[2]
                cleaned["mu_range"] = (mu_min_rel, mu_max_rel, mu_step)
                cleaned["mu_range_type"] = "auto_fermi"
            if "tdf_relative_range" in cleaned:
                tdf_lo, tdf_hi = cleaned["tdf_relative_range"]
                cleaned["tdf_range"] = (tdf_lo, tdf_hi, cleaned.get("tdf_range", (0, 0, 0.01))[2])
    
    # Remove any internal processing flags
    keys_to_remove = ["auto_shrink", "auto_bands", "auto_labels", "mu_relative_range", "tdf_relative_range"]
    for key in keys_to_remove:
        if key in cleaned:
            del cleaned[key]
    
    return cleaned


def save_d3_config(config: Dict[str, Any], filename: str, overwrite: bool = True) -> bool:
    """
    Save D3 configuration to JSON file.
    
    Args:
        config: Dictionary containing D3 configuration settings
        filename: Path to save the JSON file
        overwrite: Whether to overwrite existing file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Check if file exists and overwrite is False
        if os.path.exists(filename) and not overwrite:
            response = input(f"{filename} exists. Overwrite? [y/N]: ").strip().lower()
            if response != 'y':
                print("Save canceled.")
                return False
        
        # Clean configuration before saving
        cleaned_config = clean_config_for_saving(config)
        
        # Add metadata
        config_with_metadata = {
            "version": "1.0",
            "type": "d3_configuration",
            "calculation_type": cleaned_config.get("calculation_type", "unknown"),
            "configuration": cleaned_config
        }
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(config_with_metadata, f, indent=2)
        
        print(f"\nD3 configuration saved to: {filename}")
        return True
        
    except Exception as e:
        print(f"Error saving D3 configuration: {e}")
        return False


EXAMPLE_CONFIGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example_configs")


def resolve_d3_config_path(name: str) -> str:
    """Find a D3 configuration file named on the command line.

    Tried in order: the path as given; for a relative path, the same path
    under this install's Crystal_d3/example_configs; and for a relative path
    ending in "example_configs/<file>" (one written against another working
    directory, e.g. a code checkout given as a relative path), this
    install's example_configs/<file>. The absolute path of the first that
    exists is returned, else ``name`` unchanged. No other path is reduced to
    its bare file name, which could load a different file.
    """
    candidates = [name]
    if not os.path.isabs(name):
        candidates.append(os.path.join(EXAMPLE_CONFIGS_DIR, name))
        parts = os.path.normpath(name).split(os.sep)
        if len(parts) >= 2 and parts[-2] == "example_configs":
            candidates.append(os.path.join(EXAMPLE_CONFIGS_DIR, parts[-1]))
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    return name


def load_d3_config(filename: str) -> Optional[Dict[str, Any]]:
    """
    Load D3 configuration from JSON file.
    
    Args:
        filename: Path to the JSON file to load
        
    Returns:
        dict: Configuration dictionary or None if error
    """
    filename = resolve_d3_config_path(filename)
    try:
        if not os.path.exists(filename):
            print(f"Error: Configuration file {filename} not found.")
            return None
        
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Check if it's a D3 configuration file
        if data.get("type") != "d3_configuration":
            print(f"Error: {filename} is not a valid D3 configuration file.")
            return None
        
        config = data.get("configuration", {})
        print(f"\nLoaded D3 configuration from: {filename}")
        print(f"Calculation type: {config.get('calculation_type', 'unknown')}")
        
        return config
        
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {filename}: {e}")
        return None
    except Exception as e:
        print(f"Error loading D3 configuration: {e}")
        return None


def validate_d3_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate D3 configuration for completeness and compatibility.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        tuple: (is_valid, error_messages)
    """
    errors = []
    
    # Check for calculation type
    if "calculation_type" not in config:
        errors.append("Missing calculation_type")
    
    calc_type = config.get("calculation_type", "")
    
    # Validate based on calculation type
    if calc_type == "BAND":
        if "n_points" not in config:
            errors.append("Missing required field for BAND: n_points")
        # The band range may be given as bands="auto"/"fermi", a band_range,
        # or a first_band/last_band pair — planner-written expert configs use
        # the pair with last_band=null meaning "all bands" (the generator
        # resolves it to n_ao per material). Requiring the literal 'bands'
        # key rejected every planner-saved BAND config, and the run then
        # exited 0 with no D3 file.
        if not any(k in config for k in ("bands", "band_range", "first_band")):
            errors.append("Missing required field for BAND: bands")
        # an edited k-path (d3_band_path): its shape and labels; whether its
        # points exist is checked against each material's computed path
        if config.get("band_path") is not None:
            from d3_band_path import validate as _validate_band_path
            path_errors, _ = _validate_band_path(config["band_path"])
            errors += [f"band_path: {e}" for e in path_errors]
    
    elif calc_type == "DOSS":
        # energy_range is only present in energy-window mode; the default
        # all-bands configs saved by configure_doss_calculation don't have
        # it, and requiring it rejected every saved DOSS config on reload
        required = ["projection_type", "n_points"]
        for key in required:
            if key not in config:
                errors.append(f"Missing required field for DOSS: {key}")
        res = config.get("energy_resolution")
        if res is not None and str(res).lower() not in DOSS_RESOLUTIONS:
            errors.append(f"energy_resolution must be one of {', '.join(DOSS_RESOLUTIONS)}")
        n = config.get("n_points")
        if n is not None and str(n).strip().lower() not in ("", "auto"):
            try:
                if not 2 <= int(n) <= DOSS_MAX_NPTS:
                    raise ValueError
            except (TypeError, ValueError):
                errors.append(f"n_points must be auto or a whole number from 2 to {DOSS_MAX_NPTS} (DOSS.DAT's "
                              "header prints NPT + 2 in five digits)")
        step = config.get("energy_step_ev")
        if step is not None:
            try:
                if float(step) <= 0:
                    raise ValueError
            except (TypeError, ValueError):
                errors.append("energy_step_ev must be a step in eV larger than 0")
        cb = config.get("conduction_bands")
        if cb is not None and str(cb).strip().lower() not in ("", "auto"):
            try:
                if int(cb) < 1 or int(cb) != float(cb):
                    raise ValueError
            except (TypeError, ValueError):
                errors.append("conduction_bands must be auto or a whole number of bands, at least 1")
    
    elif calc_type in ["CHARGE", "POTENTIAL", "CHARGE+POTENTIAL"]:
        if calc_type == "CHARGE+POTENTIAL":
            if "charge_config" not in config or "potential_config" not in config:
                errors.append("Missing charge_config or potential_config for CHARGE+POTENTIAL")
        else:
            required = ["type", "n_points"]
            for key in required:
                if key not in config:
                    errors.append(f"Missing required field for {calc_type}: {key}")
    
    elif calc_type == "WANNIER":
        required = ["wannier_functions", "plot_bands"]
        for key in required:
            if key not in config:
                errors.append(f"Missing required field for WANNIER: {key}")
    
    elif calc_type == "DENSITY_MATRIX":
        if "shells" not in config:
            errors.append("Missing shells specification for DENSITY_MATRIX")
    
    elif calc_type == "RAMAN":
        required = ["raman_modes", "temperature", "wavelength"]
        for key in required:
            if key not in config:
                errors.append(f"Missing required field for RAMAN: {key}")
    
    elif calc_type == "DFT_EXCHANGE":
        if "exchange_options" not in config:
            errors.append("Missing exchange_options for DFT_EXCHANGE")
    
    elif calc_type == "ORBITALS":
        kmesh = config.get("kmesh", "band_edges")
        if kmesh not in ORBITALS_KMESH_MODES:
            errors.append(f"ORBITALS kmesh must be one of {', '.join(ORBITALS_KMESH_MODES)} "
                          f"(got {kmesh!r})")
        if kmesh == "custom":
            is_ = config.get("is")
            if not isinstance(is_, int) or isinstance(is_, bool) or is_ < 1:
                errors.append(f"ORBITALS kmesh custom needs a whole number is >= 1 (got {is_!r})")
        iloc = config.get("iloc", config.get("ILOC", 0))
        if str(iloc).strip() not in ("0", "", "None", "False") or config.get("wannier"):
            errors.append("ORBITALS ILOC 1 (Wannier functions) is not supported: use the "
                          "Wannier90 / lcao2wannier route")

    elif calc_type == "TRANSPORT":
        res = config.get("k_resolution")
        if res is not None and str(res).lower() not in TRANSPORT_K_RESOLUTIONS:
            errors.append(f"k_resolution must be one of {', '.join(TRANSPORT_K_RESOLUTIONS)}")
        if config.get("k_spacing") is not None:
            try:
                if float(config["k_spacing"]) <= 0:
                    raise ValueError
            except (TypeError, ValueError):
                errors.append("k_spacing must be a k spacing in 1/Å larger than 0")
        if config.get("newk_is") is not None:
            try:
                if int(config["newk_is"]) < 1:
                    raise ValueError
            except (TypeError, ValueError):
                errors.append("newk_is must be a shrinking factor of at least 1")
        # configure_transport_calculation emits temperature_range/mu_range;
        # "transport_type" is not a key it ever produces
        required = ["temperature_range"]
        for key in required:
            if key not in config:
                errors.append(f"Missing required field for TRANSPORT: {key}")
    
    else:
        errors.append(f"Unknown calculation type: {calc_type}")
    
    is_valid = len(errors) == 0
    return is_valid, errors


def get_default_d3_config(calc_type: str) -> Dict[str, Any]:
    """
    Get default configuration for a specific D3 calculation type.
    
    Args:
        calc_type: Type of calculation (BAND, DOSS, etc.)
        
    Returns:
        dict: Default configuration for the calculation type
    """
    defaults = {
        "BAND": {
            "calculation_type": "BAND",
            "n_points": 100,
            "bands": "auto",
            "path": "auto",
            "labels": "auto",
            "shrink": "auto"
        },
        "DOSS": {
            "calculation_type": "DOSS",
            "projection_type": 1,  # Total DOS
            "energy_range": "bands",
            "band_range": [0, 999],
            "energy_resolution": "standard",
            "n_points": "auto",
            "print_integrated": True,
            "output_format": 0,
            "projections": []
        },
        "CHARGE": {
            "calculation_type": "CHARGE",
            "type": "ECH3",
            "n_points": 100,
            "scale": 3,
            "use_range": False
        },
        "POTENTIAL": {
            "calculation_type": "POTENTIAL",
            "type": "POT3",
            "n_points": 100,
            "scale": 3,
            "use_range": False
        },
        "CHARGE+POTENTIAL": {
            "calculation_type": "CHARGE+POTENTIAL",
            "charge_config": {
                "type": "ECH3",
                "n_points": 100,
                "scale": 3,
                "use_range": False
            },
            "potential_config": {
                "type": "POT3",
                "n_points": 100,
                "scale": 3,
                "use_range": False
            }
        },
        "WANNIER": {
            "calculation_type": "WANNIER",
            "wannier_functions": "all",
            "plot_bands": True,
            "localization_method": "boys",
            "max_iterations": 100
        },
        "DENSITY_MATRIX": {
            "calculation_type": "DENSITY_MATRIX",
            "shells": "all",
            "print_overlap": True,
            "print_kinetic": False
        },
        "RAMAN": {
            "calculation_type": "RAMAN",
            "raman_modes": "all",
            "temperature": 298.15,
            "wavelength": 532.0,
            "polarization": "unpolarized"
        },
        "DFT_EXCHANGE": {
            "calculation_type": "DFT_EXCHANGE",
            "exchange_options": {
                "exact_exchange": 0.25,
                "range_separation": False
            }
        },
        "ORBITALS": {
            "calculation_type": "ORBITALS",
            "kmesh": "band_edges"
        },
        "TRANSPORT": {
            "calculation_type": "TRANSPORT",
            "transport_type": "conductivity",
            "temperature_range": [100, 1000],
            "n_temperatures": 10,
            "carrier_concentration": "auto"
        }
    }
    
    return defaults.get(calc_type, {"calculation_type": calc_type})


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries, with override_config taking precedence.
    
    Args:
        base_config: Base configuration
        override_config: Configuration to override base settings
        
    Returns:
        dict: Merged configuration
    """
    import copy
    
    result = copy.deepcopy(base_config)
    
    for key, value in override_config.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def print_d3_config_summary(config: Dict[str, Any]) -> None:
    """
    Print a summary of the D3 configuration.
    
    Args:
        config: Configuration dictionary to summarize
    """
    print("\n" + "="*60)
    print("D3 CONFIGURATION SUMMARY")
    print("="*60)
    
    calc_type = config.get("calculation_type", "Unknown")
    print(f"Calculation Type: {calc_type}")
    print("-"*60)
    
    if calc_type == "BAND":
        print(f"Number of points: {config.get('n_points', 'N/A')}")
        print(f"Band range: {config.get('bands', 'N/A')}")
        print(f"Path type: {config.get('path', 'N/A')}")
        
    elif calc_type == "DOSS":
        proj_type = config.get("projection_type", 0)
        proj_names = {
            1: "Total DOS only",
            2: "Projected on AO shells", 
            3: "Element and orbital projections",
            4: "Orbital projections only (no element totals)",
            5: "Projected on atoms",
            6: "Manual orbital projections"
        }
        print(f"Projection type: {proj_names.get(proj_type, 'Unknown')}")
        print(f"Energy range: {config.get('energy_range', 'N/A')}")
        if doss_explicit_points(config) is not None:
            print(f"Number of points: {config.get('n_points')} (as given)")
        else:
            print(f"Energy resolution: {config.get('energy_resolution') or 'standard'} "
                  f"({doss_step_ev(config):g} eV step; the points follow from the span)")
        
        if proj_type == 6 and "projections" in config:
            print("Manual projections:")
            for proj in config["projections"]:
                print(f"  - {proj}")
    
    elif calc_type in ["CHARGE", "POTENTIAL"]:
        print(f"Type: {config.get('type', 'N/A')}")
        print(f"Number of points: {config.get('n_points', 'N/A')}")
        print(f"Scale factor: {config.get('scale', 'N/A')}")
        
    elif calc_type == "CHARGE+POTENTIAL":
        if "charge_config" in config:
            print("Charge density:")
            print(f"  Type: {config['charge_config'].get('type', 'N/A')}")
            print(f"  Points: {config['charge_config'].get('n_points', 'N/A')}")
        if "potential_config" in config:
            print("Electrostatic potential:")
            print(f"  Type: {config['potential_config'].get('type', 'N/A')}")
            print(f"  Points: {config['potential_config'].get('n_points', 'N/A')}")
    
    elif calc_type == "ORBITALS":
        labels = {"band_edges": "Γ and the real k points of the band edges",
                  "gamma": "Γ only (IS 1)", "all_real": "All real k points (IS 2)",
                  "custom": f"Custom IS {config.get('is', 'N/A')}"}
        kmesh = config.get("kmesh", "band_edges")
        print(f"k points: {labels.get(kmesh, kmesh)}")
        print("Output: every orbital, as Molden files (<deck>_orb*.molden)")

    elif calc_type == "WANNIER":
        print(f"Functions: {config.get('wannier_functions', 'N/A')}")
        print(f"Plot bands: {config.get('plot_bands', 'N/A')}")
        print(f"Localization: {config.get('localization_method', 'N/A')}")
        
    elif calc_type == "RAMAN":
        print(f"Modes: {config.get('raman_modes', 'N/A')}")
        print(f"Temperature: {config.get('temperature', 'N/A')} K")
        print(f"Wavelength: {config.get('wavelength', 'N/A')} nm")
    
    print("="*60)


def create_d3_config_from_interactive(calc_type: str, input_file: str = None) -> Dict[str, Any]:
    """
    Create D3 configuration by calling the interactive configuration.
    
    Args:
        calc_type: Type of D3 calculation
        input_file: Path to input file (for context)
        
    Returns:
        dict: Configuration dictionary
    """
    from d3_interactive import configure_d3_calculation
    
    # Get configuration through interactive prompts
    config = configure_d3_calculation(calc_type, input_file)
    
    # Ensure calculation_type is set
    if "calculation_type" not in config:
        config["calculation_type"] = calc_type
    
    return config


def save_d3_options_prompt(config: Dict[str, Any], default_filename: str = None, 
                         skip_prompt: bool = False) -> bool:
    """
    Interactive prompt to save D3 configuration options.
    
    Args:
        config: Configuration dictionary to save
        default_filename: Default filename to suggest
        skip_prompt: If True, skip the initial save prompt and proceed directly
        
    Returns:
        bool: True if saved successfully
    """
    from d12_constants import yes_no_prompt
    
    if skip_prompt or yes_no_prompt("\nSave these D3 options to file for future use?", "yes"):
        if default_filename is None:
            calc_type = config.get("calculation_type", "d3").lower()
            default_filename = f"d3_{calc_type}_config.json"
        
        filename = input(f"Enter filename (default: {default_filename}): ").strip()
        if not filename:
            filename = default_filename
        
        # Ensure .json extension
        if not filename.endswith('.json'):
            filename += '.json'
        
        return save_d3_config(config, filename)
    
    return False


def list_available_d3_configs(directory: str = ".") -> List[str]:
    """
    List available D3 configuration files in a directory.
    
    Args:
        directory: Directory to search for config files
        
    Returns:
        list: List of available configuration files
    """
    config_files = []
    
    try:
        for file in os.listdir(directory):
            if file.endswith('.json') and 'd3' in file.lower():
                filepath = os.path.join(directory, file)
                try:
                    # Try to load and validate it's a D3 config
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    if data.get("type") == "d3_configuration":
                        config_files.append(file)
                except:
                    pass
    except Exception as e:
        print(f"Error listing configuration files: {e}")
    
    return sorted(config_files)


def select_d3_config_file(directory: str = ".") -> Optional[str]:
    """
    Interactive selection of D3 configuration file.
    
    Args:
        directory: Directory to search for config files
        
    Returns:
        str: Selected filename or None
    """
    configs = list_available_d3_configs(directory)
    
    if not configs:
        print("No D3 configuration files found in current directory.")
        return None
    
    print("\nAvailable D3 configuration files:")
    for i, config in enumerate(configs, 1):
        print(f"{i}. {config}")
    
    try:
        choice = int(input("\nSelect configuration file (number): ").strip())
        if 1 <= choice <= len(configs):
            return configs[choice - 1]
        else:
            print("Invalid selection.")
            return None
    except (ValueError, KeyboardInterrupt):
        return None