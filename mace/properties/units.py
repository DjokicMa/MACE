"""Display units of the stored properties: what the Properties tab (and
``mace analyze --list-properties``) can show a value in.

Display only: the database keeps every key, value and unit as the
extraction wrote them. For each key this module says

* its **dimension** (energy, length, pressure, ...) and the stored unit
  within it, from the key's unit string (:data:`STORED`), with a few keys
  whose unit string alone would mislead (:data:`KEY_DIMENSION`: a 1D carrier
  concentration in cm⁻¹ is not a wavenumber);
* its **basis**: none, per cell or per atom (``eV/atom``,
  ``states/(Hartree·cell)``);
* whether it is **extensive** (per cell: a total energy), which the Properties
  tab offers per atom and per formula unit as well. The classification is
  explicit: property_keys.EXTENSIVE for the extractor's keys,
  ``Output(extensive=True)`` for a derived property's;
* the **same quantity** stored under other keys in other units
  (property_keys.SAME_QUANTITY, ``Output(same_as=...)``): the tab shows one
  row (the most precise key) with a unit picker, not one row per unit.

Every factor comes from mace/constants.py: HARTREE_TO_EV and
BOHR_TO_ANGSTROM (the values the extractor converts with) and the exact SI
defining constants (CODATA 2018). ``scale`` is the size of one unit in the
dimension's base unit, so ``value_in_b = value_in_a * scale_a / scale_b``;
the GUI does the same arithmetic with the scales :func:`catalog` gives it.

Standard library only (the GUI's server imports it).
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

from mace.constants import (BOHR_TO_ANGSTROM, BOLTZMANN_J_K, CALORIE_J, ELEMENTARY_CHARGE_C, EV_TO_CM1, EV_TO_KJ_MOL,
                            HARTREE_TO_EV, HARTREE_TO_J, SPEED_OF_LIGHT_M_S, THZ_TO_CM1)
from mace.utils import property_keys

# ------------------------------------------------------------------ derived factors
#: 1 eV in kJ/mol, 1 eV in cm⁻¹, 1 THz in cm⁻¹, 1 Hartree in J (mace/constants.py)
EV_KJ_MOL, EV_CM, THZ_CM, HARTREE_J = EV_TO_KJ_MOL, EV_TO_CM1, THZ_TO_CM1, HARTREE_TO_J
#: 1 meV in K (10⁻³ e / k_B)
MEV_K = 1e-3 * ELEMENTARY_CHARGE_C / BOLTZMANN_J_K
#: 1 Hartree in J
HARTREE_J = HARTREE_TO_EV * ELEMENTARY_CHARGE_C
#: 1 debye in C·m (10⁻²¹ / c, exact by definition)
DEBYE_C_M = 1e-21 / SPEED_OF_LIGHT_M_S
#: 1 e·Å in D
E_ANG_D = ELEMENTARY_CHARGE_C * 1e-10 / DEBYE_C_M
#: 1 Hartree/Bohr³ in GPa
HARTREE_BOHR3_GPA = HARTREE_J / (BOHR_TO_ANGSTROM * 1e-10) ** 3 / 1e9

# ------------------------------------------------------------------ the units of each dimension
# (unit id, scale in the base unit, label per atom/f.u./cell: a template with
# {b}, or a dict by basis; None: "<id>/{b}"). The first unit is the base.
_DIMS: Dict[str, Tuple[str, List[Tuple[str, float, Any]]]] = {
    "energy": ("Energy", [
        ("eV", 1.0, None), ("meV", 1e-3, None), ("Hartree", HARTREE_TO_EV, None),
        ("mHartree", HARTREE_TO_EV / 1000.0, None), ("Ry", HARTREE_TO_EV / 2.0, None),
        ("kJ/mol", 1.0 / EV_KJ_MOL, "kJ/mol per {b}"), ("kcal/mol", CALORIE_J / EV_KJ_MOL, "kcal/mol per {b}"),
        ("cm⁻¹", 1.0 / EV_CM, None)]),
    "length": ("Length", [("Å", 1.0, None), ("Bohr", BOHR_TO_ANGSTROM, None), ("nm", 10.0, None),
                          ("pm", 0.01, None)]),
    "area": ("Area", [("Å²", 1.0, None), ("Bohr²", BOHR_TO_ANGSTROM ** 2, None), ("nm²", 100.0, None)]),
    "volume": ("Volume", [("Å³", 1.0, None), ("Bohr³", BOHR_TO_ANGSTROM ** 3, None), ("nm³", 1000.0, None)]),
    "inverse_length": ("Inverse length", [("Å⁻¹", 1.0, None), ("bohr⁻¹", 1.0 / BOHR_TO_ANGSTROM, None),
                                          ("nm⁻¹", 0.1, None)]),
    "pressure": ("Pressure", [("GPa", 1.0, None), ("MPa", 1e-3, None), ("kbar", 0.1, None), ("bar", 1e-4, None),
                              ("Mbar", 100.0, None), ("atm", 101325e-9, None),
                              ("Hartree/Bohr³", HARTREE_BOHR3_GPA, None)]),
    "inverse_pressure": ("Compressibility", [("TPa⁻¹", 1.0, None), ("GPa⁻¹", 1000.0, None)]),
    "force": ("Force", [("eV/Å", 1.0, None), ("Hartree/Bohr", HARTREE_TO_EV / BOHR_TO_ANGSTROM, None)]),
    "dipole": ("Dipole moment", [("D", 1.0, None), ("e·Å", E_ANG_D, None),
                                 ("e·Bohr", E_ANG_D * BOHR_TO_ANGSTROM, None)]),
    "frequency": ("Frequency", [("cm⁻¹", 1.0, None), ("THz", THZ_CM, None), ("meV", EV_CM / 1000.0, None)]),
    "characteristic_temperature": ("Characteristic temperature", [("K", 1.0, None), ("meV", MEV_K, None)]),
    "density": ("Density", [("g/cm³", 1.0, None), ("kg/m³", 1e-3, None)]),
    "time": ("Time", [("fs", 1.0, None), ("ps", 1000.0, None)]),
    "duration": ("Duration", [("seconds", 1.0, None), ("min", 60.0, None), ("h", 3600.0, None)]),
    "velocity": ("Velocity", [("km/s", 1.0, None), ("m/s", 1e-3, None), ("Å/fs", 100.0, None)]),
    "angle": ("Angle", [("degrees", 1.0, None), ("rad", 180.0 / math.pi, None)]),
    # energy per kelvin of a cell (an entropy, a heat capacity); J/(mol·K): per mole of cells
    "entropy": ("Entropy", [
        ("meV/K", 1.0, {"cell": "meV/(cell·K)", "atom": "meV/(atom·K)", "f.u.": "meV/(f.u.·K)"}),
        ("mHartree/K", HARTREE_TO_EV, {"cell": "mHartree/(cell·K)", "atom": "mHartree/(atom·K)",
                                       "f.u.": "mHartree/(f.u.·K)"}),
        ("J/(mol·K)", 1.0 / EV_KJ_MOL, {"cell": "J/(mol·K)", "atom": "J/(mol·K) per atom",
                                        "f.u.": "J/(mol·K) per f.u."})]),
    "dos": ("Density of states", [("states/eV", 1.0, "states/(eV·{b})"),
                                  ("states/Hartree", 1.0 / HARTREE_TO_EV, "states/(Hartree·{b})")]),
    "surface_energy": ("Surface energy", [
        ("J/m²", 1.0, None), ("mJ/m²", 1e-3, None), ("eV/Å²", ELEMENTARY_CHARGE_C * 1e20, None),
        ("meV/Å²", ELEMENTARY_CHARGE_C * 1e17, None),
        ("Hartree/Bohr²", HARTREE_J / (BOHR_TO_ANGSTROM * 1e-10) ** 2, None)]),
    "power_factor": ("Power factor", [("W/(m·K²)", 1.0, None), ("mW/(m·K²)", 1e-3, None),
                                      ("μW/(cm·K²)", 1e-4, None)]),
    "seebeck": ("Seebeck coefficient", [("μV/K", 1.0, None), ("mV/K", 1000.0, None), ("V/K", 1e6, None)]),
    "mobility": ("Mobility", [("cm²/(V·s)", 1.0, None), ("m²/(V·s)", 1e4, None)]),
    "magnetic_moment": ("Magnetic moment", [("μB", 1.0, None)]),
}

#: a stored unit string -> (dimension, unit id, basis). Basis None: the unit
#: names none (an extensive key is then per cell).
STORED: Dict[str, Tuple[str, str, Optional[str]]] = {}
for _dim, (_t, _units) in _DIMS.items():
    for _u, _s, _per in _units:
        STORED.setdefault(_u, (_dim, _u, None))
STORED.update({
    # energies with a basis
    "eV/atom": ("energy", "eV", "atom"), "meV/cell": ("energy", "meV", "cell"),
    # the entropy units name the cell
    "mHartree/(cell·K)": ("entropy", "mHartree/K", "cell"), "meV/(cell·K)": ("entropy", "meV/K", "cell"),
    "J/(mol·K)": ("entropy", "J/(mol·K)", "cell"),
    "states/(Hartree·cell)": ("dos", "states/Hartree", "cell"), "states/(eV·atom)": ("dos", "states/eV", "atom"),
    "cm⁻¹": ("frequency", "cm⁻¹", None),
})
del STORED["K"]          # a temperature in K is shown in K (a Debye temperature: KEY_DIMENSION)

#: keys whose unit string alone would give the wrong dimension or basis
KEY_DIMENSION: Dict[str, Tuple[str, str, Optional[str]]] = {
    # a carrier concentration per cm of a 1D cell, not a wavenumber
    "intrinsic_carrier_concentration_300k_cm1": ("carrier_concentration_1d", "cm⁻¹", None),
    # a temperature that stands for an energy (k_B Θ_D)
    "debye_temperature_elastic_k": ("characteristic_temperature", "K", None),
    "debye_temperature_phonon_k": ("characteristic_temperature", "K", None),
    "debye_temperature_phonon_log_k": ("characteristic_temperature", "K", None),
    "debye_temperature_sound_k": ("characteristic_temperature", "K", None),
    # per atom although the unit says eV
    "energy_per_atom_ev": ("energy", "eV", "atom"),
}

#: units with a dimension but one display unit (no picker): unit -> dimension
FIXED: Dict[str, str] = {
    "K": "temperature", "1/K": "thermal_expansion", "e": "charge", "electrons": "electron_count",
    "m_e": "mass", "W/(m·K)": "thermal_conductivity", "pm/V": "nonlinear_optics", "pC/N": "piezoelectric_strain",
    "C/m²": "piezoelectric_stress", "N/m": "modulus_2d", "m/N": "compressibility_2d",
    "mJ/(mol·K²)": "sommerfeld", "cm⁻³": "carrier_concentration_3d", "cm⁻²": "carrier_concentration_2d",
    "cycles": "count", "bytes": "size",
}
#: unit strings that are no physical unit (no dimension, no picker)
UNITLESS = frozenset({"", "dimensionless", "category", "code", "boolean", "JSON", "number", "coordinates"})

BASES = ("atom", "f.u.")

#: categories whose energies are levels (a gap, a band edge, E_F, a work
#: function): a preset may give them their own unit ("energy_level": the
#: chemistry preset keeps them in eV while it shows the energies in kJ/mol)
LEVEL_CATEGORIES = frozenset({"electronic", "band_structure", "density_of_states", "orbitals", "charge_potential",
                              "transport"})

#: blanket unit choices: dimension -> unit id (a dimension not named keeps the
#: stored unit; "energy_level": the energies of LEVEL_CATEGORIES, else "energy")
PRESETS: List[Dict[str, Any]] = [
    {"id": "stored", "label": "As stored", "units": {}},
    {"id": "hartree", "label": "Hartree atomic units", "units": {
        "energy": "Hartree", "energy_level": "Hartree", "length": "Bohr", "area": "Bohr²", "volume": "Bohr³", "inverse_length": "bohr⁻¹",
        "pressure": "Hartree/Bohr³", "force": "Hartree/Bohr", "dipole": "e·Bohr", "entropy": "mHartree/K",
        "dos": "states/Hartree", "surface_energy": "Hartree/Bohr²"}},
    {"id": "solid", "label": "eV/Å (solid state)", "units": {
        "energy": "eV", "energy_level": "eV", "length": "Å", "area": "Å²", "volume": "Å³", "inverse_length": "Å⁻¹", "pressure": "GPa",
        "force": "eV/Å", "dipole": "e·Å", "frequency": "THz", "entropy": "meV/K", "dos": "states/eV",
        "surface_energy": "J/m²", "characteristic_temperature": "K"}},
    {"id": "chemistry", "label": "kJ/mol (chemistry)", "units": {
        "energy": "kJ/mol", "energy_level": "eV", "length": "Å", "area": "Å²", "volume": "Å³", "inverse_length": "Å⁻¹",
        "pressure": "GPa", "force": "eV/Å", "dipole": "D", "frequency": "cm⁻¹", "entropy": "J/(mol·K)",
        "dos": "states/eV", "surface_energy": "J/m²"}},
]


def _unit(dim: str, uid: str) -> Optional[Tuple[str, float, Any]]:
    d = _DIMS.get(dim)
    return next((u for u in d[1] if u[0] == uid), None) if d else None


def label(dim: str, uid: str, basis: Optional[str] = None) -> str:
    """A unit as shown: ``eV``, ``eV/atom``, ``kJ/mol per f.u.``, ``meV/(cell·K)``."""
    u = _unit(dim, uid)
    if u is None:
        return uid
    if not basis:
        return uid
    per = u[2]
    if isinstance(per, dict):
        return per.get(basis, f"{uid} per {basis}")
    return (per or "{u}/{b}").format(u=uid, b=basis)


def scale(dim: str, uid: str) -> float:
    u = _unit(dim, uid)
    if u is None:
        raise KeyError(f"no unit {uid!r} of {dim!r}")
    return u[1]


def convert(value: float, dim: str, src: str, dst: str) -> float:
    """``value`` in unit ``src`` of dimension ``dim``, in unit ``dst``."""
    if src == dst:
        return value
    return value * scale(dim, src) / scale(dim, dst)


def units_of(dim: str) -> List[str]:
    d = _DIMS.get(dim)
    return [u[0] for u in d[1]] if d else []


def dimensions() -> Dict[str, Tuple[str, List[Tuple[str, float, Any]]]]:
    return _DIMS


# ------------------------------------------------------------------ what a key is
def _output(name: str):
    try:
        from mace.properties import registry
    except ImportError:                                # pragma: no cover - the package is always there
        return None
    return registry.output_of(name)


def is_extensive(name: str) -> bool:
    """Per cell: the extractor's key in property_keys.EXTENSIVE, a derived
    output declared ``extensive=True``."""
    if name in property_keys.EXTENSIVE:
        return True
    o = _output(name)
    return bool(o is not None and o.extensive)


def same_as(name: str) -> Optional[str]:
    """The key the Properties tab shows this key's quantity under (None: itself)."""
    if name in property_keys.SAME_QUANTITY:
        return property_keys.SAME_QUANTITY[name]
    o = _output(name)
    return o.same_as or None if o is not None else None


def info(name: str, unit: Optional[str] = None) -> Dict[str, Any]:
    """{dim, preset_dim, unit_id, basis, extensive, same_as, convertible,
    full_precision} of a key (``unit``: the stored unit string, else the
    registry's). ``dim`` None: no physical unit, or a JSON/text/boolean value.
    ``preset_dim``: what a preset sets the key by (``energy_level`` for an
    energy of LEVEL_CATEGORIES, else the dimension). ``full_precision``: an
    energy shown to CRYSTAL's printed precision (:func:`energy_decimals`): an
    energy of the system (not a level), or any energy stored in Hartree."""
    k = property_keys.key(name)
    unit = unit if unit is not None else (k.unit if k else "")
    typ = k.type if k else ""
    out: Dict[str, Any] = {"dim": None, "preset_dim": None, "unit_id": None, "basis": None, "extensive": False,
                           "same_as": same_as(name), "convertible": False, "full_precision": False}
    if typ not in ("float", "int") or unit in UNITLESS:
        return out
    hit = KEY_DIMENSION.get(name) or STORED.get(unit)
    if hit is None:
        if unit in FIXED:
            out.update(dim=FIXED[unit], unit_id=unit)
        return out
    dim, uid, basis = hit
    ext = is_extensive(name) and typ == "float"
    level = dim == "energy" and k is not None and k.category in LEVEL_CATEGORIES
    out.update(dim=dim, preset_dim="energy_level" if level else dim, unit_id=uid, basis=basis, extensive=ext)
    out["convertible"] = dim in _DIMS and (len(units_of(dim)) > 1 or ext)
    out["full_precision"] = dim == "energy" and typ == "float" and (not level or uid in ("Hartree", "mHartree"))
    return out


# ------------------------------------------------------------------ how many digits
#: CRYSTAL prints its energies in Hartree to about 10 decimals (TOTAL ENERGY,
#: the components, the FREQ thermodynamics): the resolution an energy is shown
#: to, in whatever unit it is shown in
ENERGY_RESOLUTION_HARTREE = 1e-10


def energy_decimals(uid: str) -> int:
    """Decimals that show an energy in unit ``uid`` to ENERGY_RESOLUTION_HARTREE
    (Hartree 10, Ry 10, eV 9, mHartree 7, kJ/mol 7, kcal/mol 8, meV 6, cm⁻¹ 5)."""
    res = ENERGY_RESOLUTION_HARTREE * HARTREE_TO_EV / scale("energy", uid)
    return max(0, math.ceil(-math.log10(res) - 1e-9))


def display_decimals(name: str, unit: Optional[str] = None) -> Optional[int]:
    """Decimals of a key's value as stored (None: its significant digits, as before)."""
    i = info(name, unit)
    return energy_decimals(i["unit_id"]) if i["full_precision"] else None


def known(unit: str) -> bool:
    """A unit string this module has a decision for."""
    return unit in UNITLESS or unit in STORED or unit in FIXED


# ------------------------------------------------------------------ duplicates
def same_quantity_groups() -> Dict[str, List[str]]:
    """{the shown key: [it, then the keys of the same quantity in other units]}."""
    from mace.properties import registry
    registry.load()
    groups: Dict[str, List[str]] = {}
    names = list(property_keys.KEYS) + list(property_keys.EXTRA)
    for n in names:
        p = same_as(n)
        if p:
            groups.setdefault(p, [p]).append(n)
    return groups


# a unit suffix of a key name, and the dimension it names (find_unit_duplicates)
_NAME_UNITS = ("_au", "_ev", "_kj_mol", "_j_m2", "_ev_ang2", "_mhartree_cell_k", "_mev_cell_k", "_j_mol_k",
               "_ev_per_atom", "_mev", "_gpa", "_mpa", "_ang", "_ang2", "_ang3", "_bohr", "_debye", "_cm", "_thz",
               "_fs", "_ps", "_s", "_km_s", "_m_s", "_k", "_deg")


def find_unit_duplicates() -> List[List[str]]:
    """Keys that look like one quantity in several units, found from the
    names and units alone (no declaration): the same name once a unit suffix
    is taken off, the same dimension and basis, different units. The test
    suite checks every such group is declared (SAME_QUANTITY, same_as)."""
    from mace.properties import registry
    registry.load()
    by: Dict[Tuple[str, str, Optional[str]], List[Tuple[str, str]]] = {}
    for n, k in list(property_keys.KEYS.items()) + list(property_keys.EXTRA.items()):
        i = info(n, k.unit)
        if not i["dim"] or i["dim"] not in _DIMS:
            continue
        stem = n
        for suf in sorted(_NAME_UNITS, key=len, reverse=True):
            if n.endswith(suf):
                stem = n[: -len(suf)]
                break
        by.setdefault((stem, i["dim"], i["basis"]), []).append((n, i["unit_id"]))
    out = []
    for (_stem, _dim, _b), keys in sorted(by.items()):
        if len(keys) > 1 and len({u for _, u in keys}) == len(keys):
            out.append(sorted(n for n, _ in keys))
    return out


# ------------------------------------------------------------------ normalization
def counts(props: Dict[str, Any], cell_formula: Optional[str] = None) -> Dict[str, Any]:
    """{atoms, formula_units, formula} of a run's values (what per atom and
    per formula unit divide by): atoms_in_unit_cell, and the reduced formula
    of its composition (its atoms, else its Mulliken table, else
    ``cell_formula``: the materials table's formula of the cell, F4Li4) with
    Z, the formula units in the cell (the EOS's formula_units without one).
    A composition whose atom count is not atoms_in_unit_cell is not used."""
    out: Dict[str, Any] = {"atoms": None, "formula_units": None, "formula": None}
    n = props.get("atoms_in_unit_cell")
    if isinstance(n, (int, float)) and not isinstance(n, bool) and n > 0:
        out["atoms"] = int(n)
    try:
        from mace.properties import reference
        from mace.utils.formula import parse_formula, reduced_formula
        comp = reference.composition(props)
        if not comp and cell_formula:
            c = parse_formula(cell_formula)
            if c and (out["atoms"] is None or sum(c.values()) == out["atoms"]):
                comp = c
        if comp:
            f, z = reduced_formula(dict(comp))
            if z:
                out.update(formula=f, formula_units=int(z))
    except Exception:                                  # noqa: BLE001 - no composition: no per formula unit
        pass
    fu = props.get("formula_units")
    if out["formula_units"] is None and isinstance(fu, (int, float)) and not isinstance(fu, bool) and fu > 0:
        out["formula_units"] = int(fu)
    return out


# ------------------------------------------------------------------ the GUI's rows
def annotate(rows: List[Dict[str, Any]], props_by_calc: Optional[Dict[Any, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Add each Properties-tab row's unit facts in place and return the
    catalog the tab converts with.

    Each row gets ``dim``, ``unit_id``, ``basis``, ``extensive``,
    ``convertible``; an extensive row its run's ``atoms``,
    ``formula_units`` and ``formula``. Of the keys that hold one quantity in
    several units, the row of the most precise one present (per run) gets
    ``also`` (the others' keys) and the others ``merged_into`` (its key):
    the tab shows that one row. Every row stays in the list (the database's
    rows, one for one)."""
    props_by_calc = props_by_calc or {}
    cnt: Dict[Any, Dict[str, Any]] = {}
    present: Dict[Tuple[Any, str], List[Dict[str, Any]]] = {}
    used = set()
    for r in rows:
        i = info(r["name"], r.get("unit"))
        r.update(dim=i["dim"], preset_dim=i["preset_dim"], unit_id=i["unit_id"], basis=i["basis"], extensive=i["extensive"],
                 full_precision=i["full_precision"],
                 convertible=i["convertible"] and isinstance(r.get("value"), (int, float))
                 and not isinstance(r.get("value"), bool))
        if r["convertible"]:
            used.add(i["dim"])
        if r["extensive"]:
            cid = r.get("calc_id")
            if cid not in cnt:
                p = props_by_calc.get(cid) or {}
                cnt[cid] = counts(p, p.get("_cell_formula"))
            r.update(cnt[cid])
        head = i["same_as"] or r["name"]
        present.setdefault((r.get("calc_id"), head), []).append(r)
    groups = same_quantity_groups() if any(len(g) > 1 for g in present.values()) else {}
    for (_cid, head), group in present.items():
        if len(group) < 2:
            continue
        order = [head] + [n for n in (groups.get(head) or []) if n != head]
        rank = {n: k for k, n in enumerate(order)}
        group.sort(key=lambda r: rank.get(r["name"], len(rank)))
        keep = next((r for r in group if r.get("convertible")), group[0])
        keep["also"] = [r["name"] for r in group if r is not keep]
        for r in group:
            if r is not keep:
                r["merged_into"] = keep["name"]
    return catalog(used)


def catalog(dims: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """The units the GUI offers: {dimensions: {dim: {label, units: [{id,
    label, per, scale, decimals}]}}, presets, bases}. ``per``: the unit's label
    per atom and per formula unit; ``decimals``: an energy's decimals in it
    (:func:`energy_decimals`, the rows marked ``full_precision``)."""
    names = list(_DIMS) if dims is None else [d for d in _DIMS if d in set(dims)]
    out = {}
    for d in names:
        title, units = _DIMS[d]
        out[d] = {"label": title, "units": [{"id": u, "scale": s, "label": u,
                                             "per": {b: label(d, u, b) for b in ("cell",) + BASES},
                                             "decimals": energy_decimals(u) if d == "energy" else None}
                                            for u, s, _ in units]}
    return {"dimensions": out, "presets": PRESETS, "bases": list(BASES)}


def describe_units(name: str) -> str:
    """The units a key can be shown in, for the inventory ("" when none)."""
    i = info(name)
    if not i["convertible"]:
        return ""
    us = units_of(i["dim"])
    text = ", ".join(label(i["dim"], u, i["basis"]) for u in us)
    if i["extensive"]:
        text += "; per atom, per formula unit"
    return text
