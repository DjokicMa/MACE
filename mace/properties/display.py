"""How a database key is shown: its label (plotly markup, ``<sub>``/``<sup>``,
for the GUI; :func:`plain` for CSV and the terminal), its group, unit,
precision, and whether it is a derived property (which one, at which
version). Derived keys take their Output declaration; the extractor's own
keys their property_keys entry, with a label for the common ones and a
readable form of the name for the rest.

Standard library only (the GUI's server imports it).
"""
from __future__ import annotations

import html
import re
from typing import Any, Dict, List, Optional

from mace.utils import property_keys

# category -> display group, in the order the Properties tab lists them
GROUPS = {
    "energy": "Energy", "electronic": "Electronic", "band_structure": "Band structure",
    "density_of_states": "Density of states", "thermodynamics": "Thermodynamics", "frequency": "Vibrations",
    "mechanical": "Mechanical", "transport": "Transport", "lattice": "Lattice", "structural": "Structure",
    "crystallographic": "Symmetry", "population_analysis": "Populations", "optimization": "Optimization",
    "charge_potential": "Charge and potential", "orbitals": "Orbitals", "scf_settings": "SCF settings",
    "computational": "Run", "other": "Other",
}

# labels of the extractor's most-read keys (the rest: label_of_name)
LABELS = {
    "total_energy": "E<sub>total</sub>", "total_energy_corrected": "E<sub>total</sub> + D3 + gCP",
    "total_energy_plus_d3": "E<sub>total</sub> + D3", "d3_dispersion_energy": "E<sub>D3</sub>",
    "gcp_energy": "E<sub>gCP</sub>", "band_gap": "Band gap", "direct_band_gap": "Direct gap",
    "indirect_band_gap": "Indirect gap", "fermi_energy": "E<sub>F</sub>",
    "zero_point_energy": "Zero-point energy E<sub>0</sub>", "electronic_energy": "E<sub>L</sub> (electronic)",
    "thermal_energy": "E<sub>T</sub> (thermal)", "pv_term": "PV", "entropy_term": "TS",
    "free_energy_correction": "E<sub>T</sub> + PV − TS", "gibbs_free_energy": "Gibbs free energy G",
    "enthalpy": "Enthalpy H = G + TS", "entropy": "Entropy S", "heat_capacity": "Heat capacity C<sub>v</sub>",
    "thermodynamic_temperature": "T (thermodynamics)", "thermodynamic_pressure": "P (thermodynamics)",
    "primitive_cell_volume": "V<sub>cell</sub> (primitive)", "crystallographic_cell_volume": "V<sub>cell</sub> (conventional)",
    "density": "Density ρ", "bulk_modulus_hill": "Bulk modulus B<sub>H</sub>",
    "bulk_modulus_voigt": "Bulk modulus B<sub>V</sub>", "bulk_modulus_reuss": "Bulk modulus B<sub>R</sub>",
    "shear_modulus_hill": "Shear modulus G<sub>H</sub>", "shear_modulus_voigt": "Shear modulus G<sub>V</sub>",
    "shear_modulus_reuss": "Shear modulus G<sub>R</sub>", "youngs_modulus": "Young's modulus E",
    "poisson_ratio": "Poisson's ratio ν", "elastic_anisotropy_universal": "Anisotropy A<sup>U</sup>",
    "debye_temperature_elastic": "Debye temperature Θ<sub>D</sub>",
    "transport_seebeck_max": "|S|<sub>max</sub>", "transport_power_factor_max_avg": "PF<sub>max</sub> (average)",
    "transport_zt_electronic_max_avg": "ZT<sub>e,max</sub> (average)", "transport_zt_total_max_avg": "ZT<sub>max</sub> (average)",
    "dielectric_constant_electronic": "ε<sub>∞</sub>", "refractive_index_electronic": "n",
    "lowest_frequency": "Lowest frequency", "highest_frequency": "Highest frequency",
    "cpu_time_total": "CPU time", "wall_time": "Wall time",
    # the quick estimate beside the accurate m_c / m_d (effective_masses) and the DOSS run's rough DOS mass
    "real_electron_effective_mass": "Effective mass, quick estimate: electrons (one band along the path)",
    "real_hole_effective_mass": "Effective mass, quick estimate: holes (one band along the path)",
    "real_average_effective_mass": "Effective mass, quick estimate: mean of electrons and holes (one band along "
                                   "the path)",
}

# display groups of extractor keys shown with a derived property's group
GROUP_OF = {f"real_{c}_effective_mass": "Effective masses" for c in ("electron", "hole", "average")}

# unit suffixes a key name carries (the unit is shown on its own)
_SUFFIXES = ("_kj_mol", "_j_mol_k", "_mhartree_cell_k", "_mev_cell_k", "_uv_per_k", "_per_k", "_km_s",
             "_cm2_vs", "_gpa", "_ang3", "_ang", "_mpa", "_au", "_ev", "_cm", "_fs", "_s", "_k")


def stem_of(name: str) -> str:
    """The name without its unit suffix (``gibbs_free_energy_kj_mol`` -> ``gibbs_free_energy``)."""
    for suf in _SUFFIXES:
        if name.endswith(suf) and len(name) > len(suf) + 1:
            return name[: -len(suf)]
    return name


def label_of_name(name: str) -> str:
    s = stem_of(name).replace("_", " ").strip()
    return s[:1].upper() + s[1:]


def plain(label: str) -> str:
    """A label without markup, for CSV and the terminal: ``m<sub>c</sub>`` ->
    ``m_c``, ``Å<sup>3</sup>`` -> ``Å^3``."""
    s = re.sub(r"<sub>(.*?)</sub>", lambda m: "_" + m.group(1), label or "")
    s = re.sub(r"<sup>(.*?)</sup>", lambda m: "^" + m.group(1), s)
    s = re.sub(r"<[^>]+>", "", s)
    return html.unescape(s)


def describe(name: str) -> Dict[str, Any]:
    """{name, label, label_plain, unit, type, category, group, precision,
    description, derived (the property name or None), version, source, blob}."""
    from mace.properties import registry
    k = property_keys.key(name)
    out = registry.output_of(name)
    prop = registry.owner_of(name)
    if out is not None and prop is not None:
        label = out.label or label_of_name(name)
        cat = out.category or prop.category
        return {"name": name, "label": label, "label_plain": plain(label), "unit": out.unit, "type": out.type,
                "category": cat, "group": out.group or prop.group or GROUPS.get(cat, cat),
                "precision": out.precision, "description": out.description or prop.description,
                "derived": prop.name, "version": prop.version, "source": prop.source,
                "interactive": prop.interactive, "blob": out.blob}
    cat = k.category if k else "other"
    label = LABELS.get(stem_of(name)) or LABELS.get(name) or label_of_name(name)
    return {"name": name, "label": label, "label_plain": plain(label), "unit": k.unit if k else "",
            "type": k.type if k else "", "category": cat,
            "group": GROUP_OF.get(name) or GROUPS.get(cat, cat.replace("_", " ").title()),
            "precision": 6, "description": k.description if k else "", "derived": None, "version": None,
            "source": "", "interactive": False, "blob": property_keys.is_blob(name)}


def group_order(group: str) -> int:
    order = list(GROUPS.values())
    return order.index(group) if group in order else len(order)


def format_value(value: Any, precision: int = 6, decimals: Optional[int] = None) -> str:
    """A value as the Properties tab and the CSV show it: a float to
    ``precision`` significant digits, or to ``decimals`` decimals (an energy
    at CRYSTAL's printed precision: mace/properties/units.display_decimals;
    the GUI formats a converted value the same way, prop_units.js), JSON as a
    short summary."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float) and decimals is not None:
        return f"{value:.{max(0, int(decimals))}f}"
    if isinstance(value, float):
        return f"{value:.{max(1, int(precision))}g}"
    if isinstance(value, (list, dict)):
        n = len(value)
        return f"{'list' if isinstance(value, list) else 'table'} of {n} item{'s' if n != 1 else ''}"
    return str(value)


def rows_sorted(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(rows, key=lambda r: (group_order(r.get("group", "")), r.get("group", ""), r.get("name", "")))


def unit_text(unit: Optional[str]) -> str:
    """The unit as shown (none for dimensionless, category, boolean, JSON)."""
    return "" if unit in (None, "", "dimensionless", "category", "code", "boolean", "JSON", "number") else unit
