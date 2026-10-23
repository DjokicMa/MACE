"""Summaries of an SCF run (SP, OPT, FREQ) the extractor's own keys do not hold:

* mulliken_summary: the Mulliken charges and spin populations per element
  (min, max, mean) and the cell's total spin (magnetization), from the
  extractor's processed_atomic_charges (population_analysis_processor: the
  charges against each atom's valence charge);
* dipole_moment: the dipole moment CRYSTAL prints for a molecule, polymer
  or slab (DIPOLE MOMENT ALONG NON PERIODIC DIRECTIONS, the Debye line,
  whose sign is the physical one; the A.U. line counts the electron charge
  as +1), and for a slab the potential step it sets up across the slab;
* packing_fraction: the fraction of a 3D cell inside the atoms' van der
  Waals spheres (1 - the geometric void fraction of the porosity tool,
  mace.analysis.porosity, the same distance field and radii) and inside
  their covalent spheres.

The density is the extractor's own key ``density`` (g/cm³).
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any, Dict, List

from mace.properties.registry import Output, Skip, derived

#: runs that end with an SCF of their own (a standalone CPHF run is an SCF, then the CPHF block)
SCF_RUNS = ("SP", "OPT", "FREQ", "CPHF")
from mace.constants import HARTREE_TO_EV as HARTREE_EV  # noqa: E402
DEBYE_PER_AU = 2.541746473
from mace.constants import BOHR_TO_ANGSTROM as BOHR  # noqa: E402  # ------------------------------------------------------------------ Mulliken
def _stats(v: List[float]) -> Dict[str, float]:
    return {"min": min(v), "max": max(v), "mean": sum(v) / len(v)}


def mulliken(rows) -> Dict[str, Any]:
    """{by_element: {el: {count, charge: {min, max, mean}, spin: {...}}}, charges, spins} of the
    processor's atomic charges (an ECP atom without a valence charge has no charge)."""
    charges: Dict[str, List[float]] = defaultdict(list)
    spins: Dict[str, List[float]] = defaultdict(list)
    count: Dict[str, int] = defaultdict(int)
    allq, alls = [], []
    for r in rows or []:
        el = str(r.get("element") or "?")
        el = el[:1].upper() + el[1:].lower()
        count[el] += 1
        q = r.get("mulliken_charge")
        if q is not None:
            charges[el].append(float(q))
            allq.append(float(q))
        s = r.get("spin_density")
        if s is not None:
            spins[el].append(float(s))
            alls.append(float(s))
    by = {}
    for el in sorted(count):
        e: Dict[str, Any] = {"count": count[el]}
        if charges[el]:
            e["charge"] = _stats(charges[el])
        if spins[el]:
            e["spin"] = _stats(spins[el])
        by[el] = e
    return {"by_element": by, "charges": allq, "spins": alls}


@derived("mulliken_summary", calc_types=SCF_RUNS, category="population_analysis", group="Populations",
         requires=("processed_atomic_charges",), version="1",
         outputs=(
             Output("mulliken_charge_min", "e", label="q<sub>Mulliken</sub> min", precision=4,
                    description="most negative Mulliken charge of the cell's atoms (valence charge minus population)"),
             Output("mulliken_charge_max", "e", label="q<sub>Mulliken</sub> max", precision=4,
                    description="most positive Mulliken charge of the cell's atoms"),
             Output("mulliken_spin_total", "μB", label="Total magnetization Σm", precision=5, extensive=True,
                    description="sum of the atoms' Mulliken spin populations (alpha - beta) of the cell"),
             Output("mulliken_spin_abs_total", "μB", label="Σ|m|", precision=5, extensive=True,
                    description="sum of the magnitudes of the atoms' spin populations (nonzero for an "
                                "antiferromagnet whose total vanishes)"),
             Output("mulliken_spin_max_abs", "μB", label="|m| max", precision=5,
                    description="largest spin population of an atom, in magnitude"),
             Output("mulliken_by_element", "JSON", "json", label="Mulliken charge and spin per element",
                    description="per element: count, and min, max and mean of the charge and the spin population"),
         ),
         source="processed_atomic_charges (the .out's Mulliken tables, population_analysis_processor)")
def _mulliken_summary(ctx):
    """Mulliken charge and spin per element; total magnetization."""
    rows = ctx.value("processed_atomic_charges")
    if isinstance(rows, str):
        import json
        rows = json.loads(rows)
    m = mulliken(rows)
    if not m["charges"] and not m["spins"]:
        raise Skip("no Mulliken charge or spin population")
    vals: Dict[str, Any] = {"mulliken_by_element": m["by_element"]}
    if m["charges"]:
        vals.update(mulliken_charge_min=min(m["charges"]), mulliken_charge_max=max(m["charges"]))
    if m["spins"]:
        vals.update(mulliken_spin_total=sum(m["spins"]), mulliken_spin_abs_total=sum(abs(s) for s in m["spins"]),
                    mulliken_spin_max_abs=max(abs(s) for s in m["spins"]))
    return vals


# ------------------------------------------------------------------ the dipole CRYSTAL prints
_DIP = re.compile(r"DIPOLE MOMENT ALONG NON PERIODIC DIRECTIONS:\s*\n\s*:::\s*A\.U\.\s*([^\n]*)\n\s*:::\s*DEBYE\s*([^\n]*)")
_NUM = re.compile(r"[-+]?\d*\.\d+(?:[EeDd][-+]?\d+)?")


def printed_dipole(text: str) -> Dict[str, float]:
    """{axis: Debye} of the last DIPOLE MOMENT ALONG NON PERIODIC DIRECTIONS block (the Debye line)."""
    found = None
    for found in _DIP.finditer(text):
        pass
    if found is None:
        return {}
    axes = re.findall(r"\(([XYZ])\)", found.group(1))
    vals = [float(v.replace("D", "E").replace("d", "e")) for v in _NUM.findall(found.group(2))]
    if not axes or len(vals) != len(axes):
        return {}
    return {a.lower(): v for a, v in zip(axes, vals)}


@derived("dipole_moment", calc_types=SCF_RUNS, category="electronic", group="Dipole", version="1",
         outputs=(
             Output("dipole_moment_debye", "D", label="|p|", precision=5,
                    description="dipole moment CRYSTAL prints for a molecule (along its non-periodic directions "
                                "for a polymer or a slab)"),
             Output("dipole_moment_x_debye", "D", label="p<sub>x</sub>", precision=5,
                    description="x component of the printed dipole moment (Debye: toward the positive charge)"),
             Output("dipole_moment_y_debye", "D", label="p<sub>y</sub>", precision=5,
                    description="y component of the printed dipole moment"),
             Output("dipole_moment_z_debye", "D", label="p<sub>z</sub>", precision=5,
                    description="z component of the printed dipole moment (a slab's normal)"),
             Output("dipole_potential_step_ev", "eV", label="ΔV<sub>dip</sub> = −4πp<sub>z</sub>/A", precision=4,
                    description="a slab's dipole as the step of an electron's potential energy from below the slab "
                                "to above it, -4 pi p_z / A (the work functions of its two sides differ by it)"),
         ),
         source=".out: DIPOLE MOMENT ALONG NON PERIODIC DIRECTIONS (the last SCF; the Debye line) and, for a "
                "slab, primitive_cell_area")
def _dipole_moment(ctx):
    """The dipole moment CRYSTAL prints (molecules, polymers, slabs)."""
    d = printed_dipole(ctx.content)
    if not d:
        raise Skip("no DIPOLE MOMENT line (a 3D cell, or a run that did not print it)")
    vals: Dict[str, Any] = {f"dipole_moment_{a}_debye": v for a, v in d.items()}
    vals["dipole_moment_debye"] = math.sqrt(sum(v * v for v in d.values()))
    area = ctx.value("primitive_cell_area")
    if set(d) == {"z"} and area:
        pz = d["z"] / DEBYE_PER_AU                          # e*bohr
        vals["dipole_potential_step_ev"] = -4.0 * math.pi * pz / (float(area) / BOHR ** 2) * HARTREE_EV
    return vals


# ------------------------------------------------------------------ packing fraction
#: the packing fraction's grid: the porosity tool's spacing, fewer points for a large cell
PACKING_SPACING = 0.15
PACKING_MAX_POINTS = 1_000_000


def packing_fractions(st, spacing: float = PACKING_SPACING, max_points: int = PACKING_MAX_POINTS) -> Dict[str, Any]:
    """{"vdw": fraction of the cell inside the van der Waals spheres (CCDC radii, the porosity tool's),
    "covalent": inside the covalent spheres (Cordero et al. 2008, ase), "grid", "spacing_A"}: overlaps
    counted once, on mace.analysis.porosity's grid and exact distance field."""
    import numpy as np
    from ase.data import covalent_radii
    from mace.analysis import porosity as P
    st = st.wrapped()
    shape, h = P._grid_shape(st, spacing, max_points)
    cart = P._grid_frac(shape) @ st.cell
    out: Dict[str, Any] = {"grid": list(shape), "spacing_A": round(h, 4)}
    for name, R in (("vdw", P.atom_radii(st, "ccdc")),
                    ("covalent", np.array([covalent_radii[int(z)] for z in st.numbers], float))):
        field = P.SurfaceDistance(st, R)
        inside = 0
        for k in range(0, len(cart), 400_000):
            inside += int((field(cart[k:k + 400_000], frac_in_cell=True) < 0).sum())
        out[name] = inside / len(cart)
    return out


@derived("packing_fraction", calc_types=SCF_RUNS, category="structural", group="Structure", version="1",
         outputs=(
             Output("packing_fraction_vdw", "dimensionless", label="Packing fraction (van der Waals)", precision=4,
                    description="fraction of the cell inside the atoms' van der Waals spheres (CCDC radii), "
                                "overlaps counted once: 1 - the porosity tool's geometric void fraction; the "
                                "packing coefficient of a molecular crystal (1 for a dense solid)"),
             Output("packing_fraction_covalent", "dimensionless", label="Packing fraction (covalent)", precision=4,
                    description="fraction of the cell inside the atoms' covalent spheres (Cordero radii), overlaps "
                                "counted once (0.34 for diamond)"),
         ),
         source="the final geometry of the .out; mace.analysis.porosity's distance field on a grid of 0.15 Å "
                "(at most 10⁶ points)")
def _packing_fraction(ctx):
    """Packing fractions of a 3D cell (van der Waals and covalent spheres)."""
    if (ctx.dimensionality or 3) != 3:
        raise Skip("a slab, polymer or molecule fills a box whose vacuum is a choice of the model")
    from mace.analysis import structure
    try:
        st = structure.load(ctx.output_file)
    except Exception as e:                         # noqa: BLE001
        raise Skip(f"no structure in the output: {e}")
    if st.n == 0 or not all(st.pbc):
        raise Skip("no 3D cell in the output")
    try:
        pf = packing_fractions(st)
    except ValueError as e:                        # an element without a radius
        raise Skip(str(e))
    return {"packing_fraction_vdw": pf["vdw"], "packing_fraction_covalent": pf["covalent"]}
