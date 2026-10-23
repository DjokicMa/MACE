"""Derived properties of a CHARGE+POTENTIAL run (CRYSTAL ECH3/POT3) from its
cubes beside the .out (``<stem>_POT.CUBE``, ``<stem>_DENS.CUBE``,
``<stem>_SPIN.CUBE``); the numbers are mace/properties/volumetric.py's.

* potential_profile: the plane-averaged and macroscopic-averaged
  electrostatic potential energy of an electron along the stacking axis
  (a slab's normal, a bulk's c), and for a slab the vacuum level on each
  side, the step across it and the work functions E_vac - E_F (E_F: the
  FERMI ENERGY the .out prints, 3 decimals in hartree: +-0.014 eV);
* cell_dipole: the dipole moment of a slab (along its normal) from its
  vacuum step, of a polymer (across its axis) or a molecule as CRYSTAL
  prints it for the parent SCF (else from the nuclear charges and the
  density cube, on a fine grid only), and for a slab the potential step
  4 pi p / A it sets up
  between the two vacuum regions (what a dipole correction of a periodic
  code would add; CRYSTAL's slab is not periodic along its normal and
  needs none);
* bader_charges: Bader charges, basin volumes and (SPIN cube) spin
  moments per atom, MACE's own grid implementation (volumetric.bader).
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from mace.properties.registry import Output, Sibling, Skip, derived

_NE = re.compile(r"N\.\s*OF ELECTRONS PER CELL\s+(\d+)")


def cube_finder(token: str):
    """find(output_file): the run's ``<stem>_<TOKEN>.CUBE`` (CRYSTAL's name), any case."""
    def find(out: Path) -> Optional[Path]:
        out = Path(out)
        want = f"{out.stem}_{token}.CUBE"
        p = out.with_name(want)
        if p.is_file():
            return p
        try:
            for q in out.parent.iterdir():
                if q.name.upper() == want.upper() and q.is_file():
                    return q
        except OSError:
            return None
        return None
    return find


_NAT = re.compile(r"N\.\s*OF ATOMS PER CELL\s+(\d+)")
_RUN_SUFFIX = re.compile(r"_charge\+potential\d*$", re.IGNORECASE)


def parent_scf(out: Path) -> Optional[Path]:
    """The SCF run (.out) the properties ran from: ``<base>.out``, ``<base>_sp.out`` or ``<base>_opt.out``
    (``<base>``: the stem without ``_charge+potential``; opt2d3's names) in the run's folder, else in a
    folder beside it (a workflow's ``step_NNN_SP/``, ``SP/`` beside ``CHARGE+POTENTIAL/``): the first
    name that matches one file (or several identical copies). Whether it is the same system is checked
    by the caller."""
    import filecmp
    out = Path(out)
    base = _RUN_SUFFIX.sub("", out.stem)
    if base == out.stem:
        return None
    ranks = [f"{base}{s}{e}" for s in ("", "_sp", "_SP", "_opt", "_OPT") for e in (".out", ".log")]
    try:
        dirs = [out.parent] + sorted(d for d in out.parent.parent.iterdir() if d.is_dir() and d != out.parent)[:2000]
    except OSError:
        dirs = [out.parent]
    for name in ranks:
        hits = [d / name for d in dirs if (d / name).is_file()]
        if not hits:
            continue
        try:
            same = all(filecmp.cmp(hits[0], h, shallow=False) for h in hits[1:])
        except OSError:
            same = False
        return hits[0] if same else None
    return None


def printed_parent_dipole(ctx, g) -> Optional[Dict[str, Any]]:
    """CRYSTAL's printed dipole of this run (a properties run normally prints none) or of its parent SCF
    (:func:`parent_scf`) when that run has the cube's atom count and the run's electron count:
    {"debye": {axis: D}, "source": file name}, or None."""
    from mace.properties.defs.scf_summary import printed_dipole
    d = printed_dipole(ctx.content)
    if d:
        return {"debye": d, "source": Path(ctx.output_file).name}
    scf = ctx.sibling("SCF")
    if scf is None:
        return None
    try:
        text = Path(scf).read_text(errors="replace")
    except OSError:
        return None
    d = printed_dipole(text)
    nat, ne, ne_run = _NAT.findall(text), electrons(text), electrons(ctx.content)
    if not d or not nat or int(nat[-1]) != len(g.positions) or (ne is not None and ne_run is not None
                                                                and abs(ne - ne_run) > 0.5):
        return None
    return {"debye": d, "source": Path(scf).name}


def electrons(text: str) -> Optional[float]:
    """The cell's electron count the .out prints (N. OF ELECTRONS PER CELL), last occurrence."""
    m = _NE.findall(text)
    return float(m[-1]) if m else None


def valence(text: str, g) -> List[float]:
    """Each cube atom's nuclear charge: the pseudopotential's valence charge where the .out gives one
    (ion_fragments.valence_charges), else the cube's charge column (Z for an all-electron atom)."""
    from mace.utils.ion_fragments import valence_charges
    table = valence_charges(text)
    return [float(table.get(int(z) % 100, ch)) for z, ch in zip(g.numbers, g.charges)]


def _checked_counts(ctx, g):
    """(valence charges, electron count) of the run; Skip when they cannot be right."""
    zs = valence(ctx.content, g)
    ne = electrons(ctx.content)
    if ne is not None and abs(sum(zs) - ne) > 2.5:
        raise Skip(f"the cube's nuclear charges sum to {sum(zs):g} and the cell has {ne:g} electrons: a "
                   "pseudopotential whose valence charge the .out does not give")
    return zs, ne


def fermi_au(ctx) -> Optional[float]:
    s = ctx.value("charge_potential")
    if isinstance(s, dict) and s.get("fermi_energy_au") is not None:
        return float(s["fermi_energy_au"])
    return None


def _round_list(a, nd=6):
    return [None if v is None else round(float(v), nd) for v in a]


def _load(path: Path):
    from mace.properties import volumetric as V
    try:
        return V.load(path)
    except (OSError, ValueError) as e:
        raise Skip(f"{Path(path).name}: {e}")


# ------------------------------------------------------------------ potential along the stacking axis
@derived("potential_profile", calc_types=("CHARGE+POTENTIAL",), category="charge_potential",
         group="Potential and work function", version="1",
         siblings=(Sibling("POT", find=cube_finder("POT"), description="the run's <stem>_POT.CUBE (POT3)"),),
         outputs=(
             Output("potential_planar_average", "JSON", "json", blob=True,
                    label="Potential V(z): planar and macroscopic average",
                    description="the electron's electrostatic potential energy -V (eV) averaged over the lattice "
                                "planes along the stacking axis (a slab's normal, a bulk's c) and its running "
                                "mean over one interlayer period (macroscopic average), with z in Å, the vacuum "
                                "levels and the Fermi energy"),
             Output("vacuum_level_top_ev", "eV", label="E<sub>vac</sub> (top)", precision=5,
                    description="potential energy of an electron in the vacuum above a slab (mean over the "
                                "grid's last 1 Å)"),
             Output("vacuum_level_bottom_ev", "eV", label="E<sub>vac</sub> (bottom)", precision=5,
                    description="potential energy of an electron in the vacuum below a slab"),
             Output("potential_vacuum_step_ev", "eV", label="ΔE<sub>vac</sub> (top − bottom)", precision=4,
                    description="vacuum level above a slab minus below it: the potential step of its dipole"),
             Output("work_function_top_ev", "eV", label="Φ (top)", precision=4,
                    description="E_vac(top) - E_F of a slab (E_F as the .out prints it; an insulator's is the "
                                "top of its valence band)"),
             Output("work_function_bottom_ev", "eV", label="Φ (bottom)", precision=4,
                    description="E_vac(bottom) - E_F of a slab"),
             Output("work_function_ev", "eV", label="Φ (mean of both sides)", precision=4,
                    description="the mean of a slab's two work functions (both are the same for a symmetric "
                                "slab)"),
         ),
         source="<stem>_POT.CUBE (POT3) and the .out's FERMI ENERGY, via mace.properties.volumetric")
def _potential_profile(ctx):
    """Plane and macroscopic averages of the potential; vacuum levels and work functions of a slab."""
    import numpy as np
    from mace.properties import volumetric as V
    g = _load(ctx.sibling("POT"))
    if V.stacking_axis(g) is None:
        raise Skip("a polymer or a molecule has no stacking axis to average along")
    prof = V.potential_profile(g, fermi_au(ctx))
    curve: Dict[str, Any] = {
        "axis": prof["axis"], "units": {"z": "Å", "energy": "eV"}, "z_A": _round_list(prof["z_A"], 5),
        "planar_ev": _round_list(prof["planar_ev"]),
        "macroscopic_ev": _round_list(prof["macroscopic_ev"]) if prof["macroscopic_ev"] is not None else None,
        "window_A": prof["window_A"], "atoms_z_A": prof["atoms_z_A"], "periodic": prof["periodic"],
        "energy": "electron potential energy -V (V: CRYSTAL's POT3 electrostatic potential)",
        "cube": Path(ctx.sibling("POT")).name}
    vals: Dict[str, Any] = {"potential_planar_average": curve}
    if not prof["periodic"]:
        vac = prof["vacuum"]
        curve["vacuum"] = vac
        curve["step_ev"] = prof["step_ev"]
        vals.update(vacuum_level_top_ev=vac["top"]["level_ev"], vacuum_level_bottom_ev=vac["bottom"]["level_ev"],
                    potential_vacuum_step_ev=prof["step_ev"])
        if "fermi_ev" in prof:
            curve["fermi_ev"] = prof["fermi_ev"]
            top, bot = vac["top"]["work_function_ev"], vac["bottom"]["work_function_ev"]
            vals.update(work_function_top_ev=top, work_function_bottom_ev=bot,
                        work_function_ev=0.5 * (top + bot))
        warn = [f"the potential is not flat in the {s} vacuum ({vac[s]['drift_ev']:.3f} eV over the last Å, "
                f"{vac[s]['distance_A']:.1f} Å from the outermost atom)" for s in ("top", "bottom")
                if not vac[s]["converged"]]
        if warn:
            curve["warnings"] = warn
    else:
        curve["mean_ev"] = prof["mean_ev"]
    return vals


# ------------------------------------------------------------------ dipole
@derived("cell_dipole", calc_types=("CHARGE+POTENTIAL",), category="charge_potential", group="Dipole",
         version="1",
         siblings=(Sibling("POT", required=False, find=cube_finder("POT"),
                           description="the run's <stem>_POT.CUBE: a slab's dipole from its vacuum step"),
                   Sibling("DENS", required=False, find=cube_finder("DENS"),
                           description="the run's <stem>_DENS.CUBE: a molecule's or polymer's dipole when its "
                                       "parent SCF is not beside the run"),
                   Sibling("SCF", required=False, find=parent_scf,
                           description="the parent SCF's .out (<base>.out, <base>_sp.out or <base>_opt.out, beside "
                                       "the run or in a folder beside its folder): a molecule's or polymer's "
                                       "dipole as CRYSTAL prints it")),
         outputs=(
             Output("cell_dipole_debye", "D", label="|p| (cell)", precision=5,
                    description="dipole moment of the cell's charge along its non-periodic directions: a slab's "
                                "from the potential step between its vacuum regions (p = -A dU / 4 pi), a "
                                "molecule's (all three) or a polymer's (across it) as CRYSTAL prints it for the "
                                "parent SCF, else from the nuclear charges and the ECH3 density about the center "
                                "of nuclear charge (only on a fine grid: spacing <= 0.16 bohr, nuclear-cusp "
                                "correction <= 0.1 e)"),
             Output("cell_dipole_x_debye", "D", label="p<sub>x</sub> (cell)", precision=5,
                    description="x component of the cell's dipole moment (physical sign: toward the positive charge)"),
             Output("cell_dipole_y_debye", "D", label="p<sub>y</sub> (cell)", precision=5,
                    description="y component of the cell's dipole moment"),
             Output("cell_dipole_z_debye", "D", label="p<sub>z</sub> (cell)", precision=5,
                    description="z component of the cell's dipole moment (a slab's normal)"),
             Output("cell_dipole_correction_ev", "eV", label="ΔV<sub>dip</sub> = −4πp/A", precision=4,
                    description="a slab's dipole correction: the step -4 pi p_z / A of the electron's potential "
                                "energy from below the slab to above it, which a periodic code's dipole correction "
                                "adds (CRYSTAL's slab has open boundaries and shows it as is)"),
             Output("cell_dipole", "JSON", "json", label="Cell dipole (details)",
                    description="the vector in e·bohr and Debye, how it was obtained, the grid's electron count, "
                                "its nuclear-cusp correction and scale, the net charge, the area"),
         ),
         source="a slab: <stem>_POT.CUBE's vacuum step; a molecule or polymer: the parent SCF's DIPOLE MOMENT "
                "line, else <stem>_DENS.CUBE and the cube's nuclei on a fine grid (mace.properties.volumetric)")
def _cell_dipole(ctx):
    """Dipole moment of the cell; for a slab from its vacuum potential step (the dipole correction)."""
    import math
    from mace.properties import volumetric as V
    pot, dens = ctx.sibling("POT"), ctx.sibling("DENS")
    if pot is None and dens is None:
        raise Skip("no POT or DENS cube beside the run")
    g = _load(pot if pot is not None else dens)
    if g.dimensionality == 3:
        raise Skip("a 3D cell has no non-periodic direction: its dipole is not defined")
    if g.dimensionality == 2:
        if pot is None:
            raise Skip("a slab's dipole is read from its vacuum potential step: no POT cube beside the run")
        prof = V.potential_profile(g)
        A = V.area(g)
        pz = -(prof["step_ev"] / V.HARTREE_EV) * A / (4 * math.pi)
        vec = [0.0, 0.0, pz * V.DEBYE_PER_AU]
        detail = {"method": "the potential step between the vacuum regions (POT cube): p = -A dU / 4 pi",
                  "vector_au": [0.0, 0.0, round(pz, 8)], "vector_debye": _round_list(vec, 6),
                  "area_bohr2": A, "step_ev": prof["step_ev"], "vacuum": prof["vacuum"], "cube": Path(pot).name}
        return {"cell_dipole_debye": abs(vec[2]), "cell_dipole_x_debye": 0.0, "cell_dipole_y_debye": 0.0,
                "cell_dipole_z_debye": vec[2], "cell_dipole_correction_ev": prof["step_ev"], "cell_dipole": detail}
    # a molecule or polymer: CRYSTAL's own dipole of the parent SCF when it is beside the run (exact), else
    # the density cube's, which only a fine grid gives right (EC 5.29 vs 5.34 D at 0.16 bohr; LiFSI-DEC
    # 4.97 vs 7.38 D at 0.48 bohr, where the cusps are off by several electrons)
    pr = printed_parent_dipole(ctx, g)
    if pr is not None:
        vec = [float(pr["debye"].get(a, 0.0)) for a in ("x", "y", "z")]
        vals = {"cell_dipole_debye": math.sqrt(sum(v * v for v in vec))}
        for k, n in enumerate(("x", "y", "z")):
            vals[f"cell_dipole_{n}_debye"] = vec[k]
        vals["cell_dipole"] = {
            "method": "CRYSTAL's DIPOLE MOMENT ALONG NON PERIODIC DIRECTIONS of the parent SCF",
            "source": pr["source"], "axes": sorted(pr["debye"]),
            "vector_au": _round_list([v / V.DEBYE_PER_AU for v in vec], 8), "vector_debye": _round_list(vec, 6),
            "dimensionality": g.dimensionality}
        return vals
    if dens is None:
        raise Skip("a molecule's or polymer's dipole needs the parent SCF's printed dipole or the density: "
                   "neither is beside the run")
    if g.kind != "density":
        g = _load(dens)
    zs, ne = _checked_counts(ctx, g)
    try:
        d = V.dipole(g, zs, ne)
    except ValueError as e:
        raise Skip(str(e))
    if not d["reliable"]:
        raise Skip(f"the parent SCF's printed dipole is not beside the run, and the density cube is too coarse "
                   f"for one (spacing {d['spacing_bohr']:.3f} bohr, nuclear-cusp correction "
                   f"{d['core_correction']:+.3f} e; needs <= {V.DIPOLE_MAX_SPACING_BOHR:g} bohr and "
                   f"<= {V.DIPOLE_MAX_CORE_CORRECTION:g} e)")
    vec = d["vector_debye"]
    vals: Dict[str, Any] = {"cell_dipole_debye": d["magnitude_debye"]}
    for k, n in enumerate(("x", "y", "z")):
        vals[f"cell_dipole_{n}_debye"] = float(vec[k])
    detail = {"method": "nuclear charges and the ECH3 density (nuclear cusps corrected), about the center of "
                        "nuclear charge",
              "vector_au": _round_list(d["vector_au"], 8), "vector_debye": _round_list(vec, 6),
              "electrons_grid": round(d["electrons_grid"], 6), "core_correction": round(d["core_correction"], 6),
              "electrons": d["electrons"], "scale": round(d["scale"], 8), "net_charge": round(d["net_charge"], 6),
              "nuclear_charge": d["nuclear_charge"], "dimensionality": g.dimensionality, "cube": Path(dens).name,
              "source": Path(dens).name, "spacing_bohr": round(d["spacing_bohr"], 4)}
    if abs(d["net_charge"]) > 0.05:
        detail["note"] = "a charged cell: its dipole moment depends on the origin (here the center of nuclear charge)"
    vals["cell_dipole"] = detail
    return vals


# ------------------------------------------------------------------ Bader
def _summary(numbers, values) -> Dict[str, Dict[str, float]]:
    from mace.utils.formula import ELEMENTS
    by: Dict[str, List[float]] = defaultdict(list)
    for z, v in zip(numbers, values):
        by[ELEMENTS[int(z) - 1] if 0 < int(z) <= len(ELEMENTS) else str(z)].append(float(v))
    return {el: {"count": len(v), "min": min(v), "max": max(v), "mean": sum(v) / len(v)} for el, v in sorted(by.items())}


@derived("bader_charges", calc_types=("CHARGE+POTENTIAL",), category="charge_potential", group="Bader charges",
         version="1",
         siblings=(Sibling("DENS", find=cube_finder("DENS"), description="the run's <stem>_DENS.CUBE (ECH3)"),
                   Sibling("SPIN", required=False, find=cube_finder("SPIN"),
                           description="the run's <stem>_SPIN.CUBE: spin moments of the same basins")),
         outputs=(
             Output("bader_charge_min", "e", label="q<sub>Bader</sub> min", precision=4,
                    description="most negative Bader charge of the cell's atoms"),
             Output("bader_charge_max", "e", label="q<sub>Bader</sub> max", precision=4,
                    description="most positive Bader charge of the cell's atoms"),
             Output("bader_spin_total", "μB", label="Σ m<sub>Bader</sub>", precision=4, extensive=True,
                    description="sum of the atoms' Bader spin moments (the SPIN cube over each basin)"),
             Output("bader_spin_max_abs", "μB", label="|m<sub>Bader</sub>| max", precision=4,
                    description="largest Bader spin moment of an atom, in magnitude"),
             Output("bader_electrons_grid", "electrons", label="N<sub>e</sub> on the grid", precision=6,
                    description="electrons the density grid holds (the basins are scaled to the run's count)"),
             Output("bader_nonnuclear_maxima", "dimensionless", "int", label="Non-nuclear maxima", precision=6,
                    description="density maxima farther than 1 bohr from every atom (given to the nearest atom)"),
             Output("bader_charges", "JSON", "json", blob=True, label="Bader charges per atom",
                    description="per atom: element, Bader charge, population, basin volume (Å³) and spin moment; "
                                "per element: count, min, max, mean; the grid, the scale and the vacuum threshold"),
         ),
         source="<stem>_DENS.CUBE (+ _SPIN.CUBE): MACE's grid Bader partition (on-grid ascent with near-grid "
                "refinement), mace.properties.volumetric.bader")
def _bader_charges(ctx):
    """Bader charges, basin volumes and spin moments per atom from the density cube."""
    from mace.properties import volumetric as V
    g = _load(ctx.sibling("DENS"))
    zs, ne = _checked_counts(ctx, g)
    spin = None
    if ctx.sibling("SPIN") is not None:
        try:
            s = V.load(ctx.sibling("SPIN"))
            spin = s if s.data.shape == g.data.shape else None
        except (OSError, ValueError):
            spin = None
        g = _load(ctx.sibling("DENS"))          # the reader keeps one grid: the density again
    try:
        r = V.bader(g, spin=spin, valence=zs, electrons=ne)
    except ValueError as e:
        raise Skip(str(e))
    from mace.utils.formula import ELEMENTS
    atoms = []
    for i, z in enumerate(r["numbers"]):
        a = {"atom": i + 1, "element": ELEMENTS[int(z) - 1] if 0 < int(z) <= len(ELEMENTS) else str(z),
             "z": int(z), "charge": round(float(r["charges"][i]), 6),
             "population": round(float(r["populations"][i]), 6), "volume_A3": round(float(r["volumes_A3"][i]), 5)}
        if "spins" in r:
            a["spin"] = round(float(r["spins"][i]), 6)
        atoms.append(a)
    detail: Dict[str, Any] = {
        "atoms": atoms, "by_element": {"charge": _summary(r["numbers"], r["charges"])},
        "electrons": ne, "electrons_grid": round(r["electrons_grid"], 6), "electrons_vacuum": round(r["electrons_vacuum"], 6),
        "scale": round(r["scale"], 8), "maxima": r["maxima"], "nonnuclear_maxima": r["nonnuclear_maxima"],
        "vacuum_density_e_bohr3": r["vacuum_density"], "grid": r["shape"], "dv_bohr3": r["dv_bohr3"],
        "method": "on-grid steepest ascent (Henkelman et al. 2006) refined by near-grid trajectories "
                  "(Tang et al. 2009) on the basin boundaries; MACE's own implementation",
        "cube": Path(ctx.sibling("DENS")).name}
    vals: Dict[str, Any] = {"bader_charge_min": float(min(r["charges"])), "bader_charge_max": float(max(r["charges"])),
                            "bader_electrons_grid": r["electrons_grid"],
                            "bader_nonnuclear_maxima": r["nonnuclear_maxima"], "bader_charges": detail}
    if "spins" in r:
        detail["by_element"]["spin"] = _summary(r["numbers"], r["spins"])
        detail["spin_total_grid"] = round(r["spin_total"], 6)
        vals["bader_spin_total"] = float(sum(r["spins"]))
        vals["bader_spin_max_abs"] = float(max(abs(float(v)) for v in r["spins"]))
    if ne and abs(r["electrons_grid"] - ne) > 0.01 * ne:
        detail["warning"] = (f"the grid holds {r['electrons_grid']:.2f} of the run's {ne:g} electrons: a coarse grid "
                             "misses part of each nuclear cusp, so the charges are approximate")
    return vals
