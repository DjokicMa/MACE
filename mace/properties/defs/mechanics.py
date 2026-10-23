"""The mechanics results' scalars in the database: a thin wrapper around
``mace.elastic.properties.all_properties(analysis.analyze(text))``, the
numbers the Mechanical Properties Results document shows (its
``views.document`` runs the same ``analyze``), so the database and the
GUI share one code path.

ELASTIC: directional extremes of E, G, nu and the linear compressibility,
sound and group velocity extremes, anisotropy indices, Pugh and Cauchy,
Chen and Tian hardness, Clarke and Cahill minimum thermal conductivity,
piezoelectric extremes, the 2D anisotropy. EOS: each fit's residual RMS
and largest residual, the formula units. QHA: S, C_v, C_p and the
Grueneisen parameters near 300 K. CPHF: principal refractive indices,
birefringence, optic axial angle, SHG d_eff. FREQ with INTCPHF: Born
charge range, acoustic-sum residual, eps_0, eps_inf, the largest LO-TO
splitting.

The database keys are all_properties' names in lower case (``qL`` ->
``ql``). The Voigt/Reuss/Hill moduli, Young's modulus, Poisson's ratio,
A^U and the elastic Debye temperature are the extractor's own keys and are
not repeated (``anisotropy_universal`` is ``elastic_anisotropy_universal``).
Every value all_properties gives, with its unit, is also kept whole in
``mechanics_derived`` (JSON), the ones without a key of their own too.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from mace.properties.registry import Output, Skip, derived

# (name, unit as stored, unit all_properties gives, label, group, description)
_E = "Elastic"
_S = "Elastic: sound"
_EOS = "Equation of state"
_Q = "Quasi-harmonic"
_O = "Optics"
_B = "Dielectric (Born charges)"
SPEC: List[Tuple[str, str, str, str, str, str]] = []


def _add(name, unit, given, label, group, desc):
    SPEC.append((name, unit, given, label, group, desc))


for q, lab in (("youngs_modulus", "E"), ("shear_modulus", "G")):
    for k in ("min", "max"):
        _add(f"{q}_{k}", "GPa", "GPa", f"{lab}<sub>{k}</sub>", _E, f"{k}imum of the directional {q.replace('_', ' ')}")
        _add(f"{q}_{k}_2d", "N/m", "N/m", f"{lab}<sub>{k}</sub> (2D)", _E,
             f"{k}imum of a 2D crystal's directional {q.replace('_', ' ')} (per length)")
for k in ("min", "max"):
    _add(f"linear_compressibility_{k}", "TPa⁻¹", "TPa^-1", f"β<sub>{k}</sub>", _E, f"{k}imum of the linear compressibility")
    _add(f"linear_compressibility_{k}_2d", "m/N", "m/N", f"β<sub>{k}</sub> (2D)", _E,
         f"{k}imum of a 2D crystal's linear compressibility")
    _add(f"poisson_ratio_{k}", "dimensionless", "", f"ν<sub>{k}</sub>", _E, f"{k}imum of the directional Poisson's ratio")
for mode, lab in (("ql", "qL"), ("qs1", "qS1"), ("qs2", "qS2")):
    for k in ("min", "max"):
        _add(f"sound_velocity_{mode}_{k}_km_s", "km/s", "km/s", f"v<sub>{lab},{k}</sub>", _S,
             f"{k}imum over directions of the {lab} phase velocity (Christoffel)")
        _add(f"group_velocity_{mode}_{k}_km_s", "km/s", "km/s", f"v<sub>g,{lab},{k}</sub>", _S,
             f"{k}imum over directions of the {lab} group velocity")
for name, label, desc in (("anisotropy_zener", "A<sub>Z</sub>", "Zener anisotropy 2C44/(C11 - C12)"),
                          ("anisotropy_chung_buessem", "A<sub>CB</sub>", "Chung-Buessem anisotropy"),
                          ("youngs_modulus_ratio", "E<sub>max</sub>/E<sub>min</sub>", "Young's modulus anisotropy ratio"),
                          ("shear_modulus_ratio", "G<sub>max</sub>/G<sub>min</sub>", "shear modulus anisotropy ratio"),
                          ("poisson_ratio_ratio", "ν<sub>max</sub>/ν<sub>min</sub>", "Poisson's ratio anisotropy ratio"),
                          ("pugh_ratio", "B/G", "Pugh's ratio B_H/G_H (ductile above 1.75)"),
                          ("anisotropy_2d_su", "A<sub>SU</sub> (2D)", "the 2D anisotropy index of a 2D crystal"),
                          ("piezo_coupling_k2_max", "k<sup>2</sup><sub>max</sub>",
                           "largest electromechanical coupling k^2 over directions")):
    _add(name, "dimensionless", "", label, _E, desc)
_add("ductile_pugh", "boolean", "", "Ductile (Pugh)", _E, "B_H/G_H above 1.75")
for rel in ("c12_minus_c44", "c12_minus_c66", "c13_minus_c55", "c23_minus_c44", "c14_minus_c56", "c25_minus_c46",
            "c36_minus_c45"):
    a, b = rel.split("_minus_")
    _add(f"cauchy_{rel}", "GPa", "GPa", f"{a.upper()} − {b.upper()}", _E,
         f"Cauchy relation {a.upper()} - {b.upper()} (the Cauchy pressure of a cubic crystal)")
_add("vickers_hardness_chen_gpa", "GPa", "GPa", "H<sub>V</sub> (Chen)", _E, "Vickers hardness, Chen's model")
_add("vickers_hardness_tian_gpa", "GPa", "GPa", "H<sub>V</sub> (Tian)", _E, "Vickers hardness, Tian's model")
_add("min_thermal_conductivity_clarke", "W/(m·K)", "W/(m K)", "κ<sub>min</sub> (Clarke)", _E,
     "Clarke's minimum thermal conductivity")
_add("min_thermal_conductivity_cahill", "W/(m·K)", "W/(m K)", "κ<sub>min</sub> (Cahill)", _E,
     "Cahill's minimum thermal conductivity")
_add("piezo_d_longitudinal_max", "pC/N", "pC/N", "|d<sub>L</sub>|<sub>max</sub>", _E,
     "largest longitudinal piezoelectric strain coefficient over directions")
_add("piezo_e_longitudinal_max", "C/m²", "C/m^2", "|e<sub>L</sub>|<sub>max</sub>", _E,
     "largest longitudinal piezoelectric stress coefficient over directions")
for fit in ("murnaghan", "birch_murnaghan", "poirier_tarantola", "vinet"):
    for k, what in (("rms", "RMS"), ("max", "largest")):
        _add(f"eos_{fit}_residual_{k}", "meV/cell", "meV/cell", f"{fit.replace('_', '-').title()} residual {what}", _EOS,
             f"{what} residual E_computed - E_fit of the {fit.replace('_', '-')} fit")
_add("formula_units", "dimensionless", "", "Formula units Z", _EOS, "formula units in the cell (the EOS enthalpy is per formula unit)")
_add("qha_entropy_300k", "J/(mol·K)", "J/(mol K)", "S (300 K, QHA)", _Q, "entropy along the QHA V(T) near 300 K")
_add("qha_cv_300k", "J/(mol·K)", "J/(mol K)", "C<sub>V</sub> (300 K, QHA)", _Q, "isochoric heat capacity near 300 K")
_add("qha_cp_300k", "J/(mol·K)", "J/(mol K)", "C<sub>P</sub> (300 K, QHA)", _Q, "isobaric heat capacity near 300 K")
_add("qha_gamma_thermodynamic_300k", "dimensionless", "", "γ<sub>th</sub> (300 K)", _Q,
     "thermodynamic Grüneisen parameter alpha_V B_T V_m / C_V near 300 K")
_add("qha_gamma_mode_300k", "dimensionless", "", "γ (300 K, modes)", _Q, "CRYSTAL's mode-averaged Grüneisen parameter near 300 K")
for k in (1, 2, 3):
    _add(f"refractive_index_principal_{k}", "dimensionless", "", f"n<sub>{k}</sub>", _O, f"principal refractive index {k} of eps_inf")
_add("birefringence", "dimensionless", "", "Δn", _O, "birefringence n_max - n_min")
_add("birefringence_signed", "dimensionless", "", "Δn (signed)", _O, "signed birefringence n_e - n_o of a uniaxial crystal")
_add("optic_axial_angle_deg", "degrees", "deg", "2V", _O, "optic axial angle of a biaxial crystal")
for name, label, desc in (("shg_d_longitudinal_max", "|d<sub>L</sub>|<sub>max</sub> (SHG)", "largest longitudinal SHG coefficient"),
                          ("shg_d_magnitude_max", "|d|<sub>max</sub> (SHG)", "largest SHG d_eff magnitude"),
                          ("shg_d_powder_longitudinal", "d<sub>powder,L</sub> (SHG)", "powder average of the longitudinal SHG d"),
                          ("shg_d_powder_magnitude", "d<sub>powder</sub> (SHG)", "powder average of the SHG d magnitude")):
    _add(name, "pm/V", "pm/V", label, _O, desc)
_add("born_charge_min", "e", "e", "Z*<sub>min</sub>", _B, "smallest isotropic Born effective charge")
_add("born_charge_max", "e", "e", "Z*<sub>max</sub>", _B, "largest isotropic Born effective charge")
_add("born_charge_sum_residual", "e", "e", "|ΣZ*| (acoustic sum)", _B, "largest component of the Born charges' sum (0 exactly)")
_add("dielectric_constant_static", "dimensionless", "", "ε<sub>0</sub>", _B, "static dielectric constant (trace/3) of a FREQ run with INTCPHF")
_add("dielectric_constant_electronic_freq", "dimensionless", "", "ε<sub>∞</sub> (FREQ)", _B,
     "electronic dielectric constant (trace/3) of a FREQ run with INTCPHF")
_add("lo_to_splitting_max", "cm⁻¹", "cm^-1", "LO−TO<sub>max</sub>", _B, "largest computed LO-TO splitting")

#: database key -> (unit given by all_properties)
GIVEN = {s[0]: s[2] for s in SPEC}
#: all_properties names the extractor already stores under its own key
SKIP = {"anisotropy_universal": "elastic_anisotropy_universal"}


def db_key(name: str, unit: str) -> str:
    """all_properties' name as a database key: lower case, ``_2d`` for a 2D crystal's per-length moduli."""
    k = name.lower()
    if unit in ("N/m", "m/N") and k + "_2d" in GIVEN:
        k += "_2d"
    return k


def keys_of(allp: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """{database key: value} of all_properties' result (the declared keys whose unit matches)."""
    out: Dict[str, Any] = {}
    for name, v in allp.items():
        if name in SKIP:
            continue
        k = db_key(name, v.get("unit", ""))
        if k in GIVEN and GIVEN[k] == v.get("unit", ""):
            out[k] = v["value"]
    return out


def values(text: str) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """({database key: value} of the declared keys, all_properties as given)."""
    from mace.elastic import analysis as A
    from mace.elastic import properties as P
    allp = P.all_properties(A.analyze(text), text)
    return keys_of(allp), allp


_MECH = ("ELASTIC", "EOS", "QHA", "CPHF", "FREQ")


@derived("mechanics_derived", calc_types=_MECH, category="mechanical", group="Mechanical", version="1",
         outputs=tuple(Output(n, u, "bool" if u == "boolean" else "float", label=lab, group=grp, description=d,
                              precision=5, category="electronic" if grp in (_O, _B) else
                              ("thermodynamics" if grp == _Q else ""))
                       for n, u, _g, lab, grp, d in SPEC)
         + (Output("mechanics_derived", "JSON", "json", label="Mechanics results (all derived scalars)",
                   description="every scalar mace.elastic.properties.all_properties gives for the run, "
                               "{name: {value, unit}}, as the Mechanical Properties Results show them"),),
         source="mace.elastic.properties.all_properties(mace.elastic.analysis.analyze(.out)): the Mechanical "
                "Properties Results' numbers")
def _mechanics(ctx):
    """The mechanics results' derived scalars (elastic, EOS, QHA, optics, Born charges)."""
    text = ctx.content
    if ctx.calc_type == "FREQ" and "ATOMIC BORN CHARGE TENSOR" not in text:
        raise Skip("a FREQ run without INTCPHF has no Born charges or dielectric response")
    got, allp = values(text)
    if not allp:
        raise Skip("no elastic, EOS, QHA, CPHF or Born-charge results in the output")
    got["mechanics_derived"] = allp
    return got
