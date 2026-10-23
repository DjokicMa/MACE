"""The scalar quantities the results documents of ELASTIC, EOS, QHA, CPHF and FREQ (INTCPHF) runs
derive, as {name: {"value", "unit"}}: one flat list for a property registry (materials.db) and the
GUI to share. Every number is computed in mace.elastic (``analysis.analyze``, ``tensor.derived``),
never in the GUI's routes; this module only names them.

``elastic_properties(data)``: from ``analysis.analyze(text)`` (its ``derived``);
``eos_properties(data)``: the fits' residuals (``eos.quality``); ``qha_properties(data)``: S, C_v,
C_p and gamma near 300 K (``qha.thermo``, ``qha.gruneisen``); ``optics_properties(data)``: the
principal indices, birefringence and SHG d_eff (``optics``); ``born_properties(data)``: Born charges,
eps_0 and the LO-TO splitting of a FREQ run with INTCPHF (``born``).

What the documents show flagged is not stored: the derived numbers of a mechanically unstable C
(``derived.unstable``), a Chen hardness its formula puts below zero, and the class-dependent optics
(principal n, birefringence, 2V) and the LO-TO splitting of an eps_inf or eps_0 that breaks the crystal
symmetry, has no real index, or comes from a CPHF that was unstable or did not converge.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


def _put(out: Dict[str, Dict[str, Any]], name: str, value, unit: str = "") -> None:
    if isinstance(value, bool) or (isinstance(value, (int, float)) and value == value):
        out[name] = {"value": value, "unit": unit}


def elastic_properties(data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """The directional extremes, anisotropy indices, character, hardness, thermal and piezoelectric
    numbers of an ELASTIC run (tensor.derived)."""
    out: Dict[str, Dict[str, Any]] = {}
    der = data.get("derived") or {}
    if der.get("unstable"):
        return out                     # C not positive definite: every derived number is unphysical
    for q, prefix in (("youngs", "youngs_modulus"), ("compressibility", "linear_compressibility"),
                      ("shear", "shear_modulus"), ("poisson", "poisson_ratio")):
        e = (der.get("directional") or {}).get(q) or {}
        for k in ("min", "max"):
            if e.get(k):
                _put(out, f"{prefix}_{k}", e[k]["value"], e.get("unit", ""))
    for name, s in ((der.get("directional") or {}).get("sound") or {}).items():
        for k in ("min", "max"):
            _put(out, f"sound_velocity_{name}_{k}_km_s", s[k]["value"], "km/s")
            _put(out, f"group_velocity_{name}_{k}_km_s", s["group"][k]["value"], "km/s")
    idx = der.get("indices") or {}
    for k, name in (("universal", "anisotropy_universal"), ("zener", "anisotropy_zener"),
                    ("chung_buessem", "anisotropy_chung_buessem"), ("e_ratio", "youngs_modulus_ratio"),
                    ("g_ratio", "shear_modulus_ratio"), ("nu_ratio", "poisson_ratio_ratio")):
        _put(out, name, idx.get(k))
    ch = der.get("character") or {}
    _put(out, "pugh_ratio", ch.get("pugh_ratio"))
    _put(out, "ductile_pugh", ch.get("ductile"))
    for c in ch.get("cauchy") or []:
        _put(out, "cauchy_" + c["relation"].replace(" − ", "_minus_").lower(), c["value"], "GPa")
    hv = ch.get("hardness") or {}
    if not hv.get("chen_clamped"):
        _put(out, "vickers_hardness_chen_gpa", hv.get("chen_gpa"), "GPa")
    _put(out, "vickers_hardness_tian_gpa", hv.get("tian_gpa"), "GPa")
    th = der.get("thermal") or {}
    _put(out, "min_thermal_conductivity_clarke", th.get("clarke_w_mk"), "W/(m K)")
    _put(out, "min_thermal_conductivity_cahill", th.get("cahill_w_mk"), "W/(m K)")
    pz = der.get("piezo") or {}
    if pz.get("longitudinal") and pz["longitudinal"].get("max"):
        _put(out, f"piezo_{pz['kind']}_longitudinal_max", abs(pz["longitudinal"]["max"]["value"]), pz["unit"])
        cp = pz.get("coupling") or {}
        if cp.get("available"):
            _put(out, "piezo_coupling_k2_max", cp["k2_max"]["value"])
    t2 = der.get("two_d") or {}
    _put(out, "anisotropy_2d_su", t2.get("anisotropy_su"))
    return out


def eos_properties(data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Each EOS fit's residual RMS and largest residual (views.eos_quality, in analyze()'s eos.quality)
    and the formula units of the cell (the enthalpy is per formula unit)."""
    out: Dict[str, Dict[str, Any]] = {}
    for k, q in ((data.get("eos") or {}).get("quality") or {}).items():
        _put(out, f"eos_{k}_residual_rms", q["rms_mev"], "meV/cell")
        _put(out, f"eos_{k}_residual_max", q["max_abs_mev"], "meV/cell")
    if data.get("eos"):
        _put(out, "formula_units", data.get("formula_units"))
    return out


def nearest_300k(temperatures) -> Optional[int]:
    """The index of the temperature nearest 300 K, None when none is within 15 K: the row the
    *_300k values come from (the GUI's QHA tiles read the same one)."""
    T = list(temperatures or [])
    if not T:
        return None
    i = min(range(len(T)), key=lambda k: abs(T[k] - 300.0))
    return i if abs(T[i] - 300.0) <= 15.0 else None


def qha_properties(data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """The QHA thermodynamics along V(T) nearest 300 K (qha.thermo) and CRYSTAL's mode Grueneisen
    parameter there (qha.gruneisen_table)."""
    out: Dict[str, Dict[str, Any]] = {}
    q = data.get("qha") or {}
    th = q.get("thermo")
    if th:
        i = nearest_300k(th["temperature_k"])
        if i is not None:
            _put(out, "qha_entropy_300k", th["entropy_j_mol_k"][i], "J/(mol K)")
            _put(out, "qha_cv_300k", th["cv_j_mol_k"][i], "J/(mol K)")
            _put(out, "qha_cp_300k", th["cp_j_mol_k"][i], "J/(mol K)")
            _put(out, "qha_gamma_thermodynamic_300k", th["gamma_th"][i])
    g = q.get("gruneisen")
    if g:
        i = nearest_300k(g["temperature_k"])
        if i is not None:
            _put(out, "qha_gamma_mode_300k", g["gamma"][i])
    return out


def _cphf_failed(opt: Dict[str, Any]) -> bool:
    """The CPHF CRYSTAL printed was unstable (DIIS test past CPHF_DIIS_UNSTABLE) or stopped unconverged:
    its eps_inf is not the crystal's response, whatever the symmetry allows (Ag1Cl5, monoclinic: eps 777)."""
    diag = opt.get("cphf") or {}
    return bool(diag.get("unstable") or diag.get("unconverged"))


def optics_properties(data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """The principal refractive indices and birefringence of eps_inf, and the SHG d_eff (optics.summary)."""
    out: Dict[str, Dict[str, Any]] = {}
    opt = data.get("optics") or {}
    pr = opt.get("principal")
    if pr and not (pr.get("symmetry_mismatch") or pr.get("unphysical") or _cphf_failed(opt)):
        for k, n in enumerate(pr["n"]):
            _put(out, f"refractive_index_principal_{k + 1}", n)
        _put(out, "birefringence", pr["birefringence"])
        if pr.get("birefringence_signed") is not None:
            _put(out, "birefringence_signed", pr["birefringence_signed"])
        if pr.get("optic_axial_angle_deg") is not None:
            _put(out, "optic_axial_angle_deg", pr["optic_axial_angle_deg"], "deg")
    sh = opt.get("shg") or {}
    if sh.get("zero") is False:
        _put(out, "shg_d_longitudinal_max", abs(sh["longitudinal"]["max"]["value"]), "pm/V")
        _put(out, "shg_d_magnitude_max", sh["magnitude"]["max"]["value"], "pm/V")
        _put(out, "shg_d_powder_longitudinal", sh["powder_longitudinal"], "pm/V")
        _put(out, "shg_d_powder_magnitude", sh["powder_magnitude"], "pm/V")
    return out


def born_properties(data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """A FREQ run with INTCPHF (born.dielectric_response): eps_inf and eps_0 (trace/3), the range of
    the Born charges, the acoustic-sum residual and the largest computed LO-TO splitting."""
    out: Dict[str, Dict[str, Any]] = {}
    b = data.get("born") or {}
    if not b:
        return out
    zs = [x["isotropic"] for x in b["born_charges"]]
    _put(out, "born_charge_min", min(zs), "e")
    _put(out, "born_charge_max", max(zs), "e")
    _put(out, "born_charge_sum_residual", max(abs(v) for row in b["charge_sum"] for v in row), "e")
    _put(out, "dielectric_constant_static", b.get("epsilon_0_iso"))
    _put(out, "dielectric_constant_electronic_freq", b.get("epsilon_inf_iso"))
    split = [l - t for r in b.get("lo_to") or [] for t, l in zip(r["to_cm"], r["lo_cm"])]
    opt = data.get("optics") or {}
    pr = opt.get("principal") or {}
    broken = (b.get("epsilon_0_symmetry_mismatch") or pr.get("symmetry_mismatch") or pr.get("unphysical")
              or _cphf_failed(opt))
    if split and not broken:                 # the splitting along q needs q.eps_inf.q of the crystal
        _put(out, "lo_to_splitting_max", max(split), "cm^-1")
    return out


def all_properties(data: Dict[str, Any], text: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Every derived scalar of a run this module knows (``text``: the output, for the quantities read
    beyond analyze())."""
    return {**elastic_properties(data), **eos_properties(data), **qha_properties(data), **optics_properties(data),
            **born_properties(data)}
