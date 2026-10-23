"""What a QHA run prints beyond its zero-pressure table (CRYSTAL23 manual 9.3; Erba, J. Chem. Phys.
141, 124115 (2014)), and the thermodynamic functions derived from it.

Read from the output:

* ``THERMAL PROPERTIES FROM GRUNEISEN PARAMETERS``: T, alpha, V, the Grueneisen parameter gamma
  and Cp - Cv of CRYSTAL's mode-Grueneisen model;
* each volume's ``HARMONIC VIBRATIONAL CONTRIBUTIONS``: EL and E0 (hartree/cell), and per
  temperature ET and TS (hartree/cell), the entropy S and the heat capacity C_v (J/(mol K), per
  mole of cells), with the volume printed before them;
* ``PRESSURE-VOLUME-TEMPERATURE RELATION``: F(V, T) and G on a volume grid at each temperature
  (CRYSTAL's own fit of the static energy and of the vibrational free energy).

Derived (zero pressure), at each temperature every volume prints:

* F_vib(V_i, T) = E0 + ET - TS and F(V_i, T) = EL + F_vib, then F_vib, S and C_v interpolated to the
  equilibrium volume V(T) of the zero-pressure table with a polynomial of the order CRYSTAL uses
  (its ORDER OF POLYNOMIALS, 2 by default) through the computed volumes, and EL with one of the
  highest order the volumes allow (up to 3);
* C_p = C_v + (C_p - C_v), the latter from the zero-pressure table;
* the thermodynamic Grueneisen parameter gamma_th = alpha_V B_T V_m / C_v (V_m the molar volume of
  the cell), next to CRYSTAL's mode-averaged gamma; left out where C_v is under a fifth of its
  largest value (alpha is then tiny and the ratio numerical noise);
* the F(V) family: F(V) at several temperatures from the PVT table (else the polynomial through the
  computed volumes) with each curve's minimum, which traces V(T).

Numpy only.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional

from . import analysis as A

N_AVOGADRO = 6.02214076e23
from mace.constants import HARTREE_TO_KJ_MOL as HARTREE_KJ_MOL  # noqa: E402
HARTREE_MEV = 27211.386245988

_GRUN_HEAD = re.compile(r"^\s*THERMAL PROPERTIES FROM GRUNEISEN PARAMETERS\s*$", re.M)
_PVT_HEAD = re.compile(r"^\s*PRESSURE-VOLUME-TEMPERATURE RELATION\s*$", re.M)
_HARM_HEAD = re.compile(r"^\s*HARMONIC VIBRATIONAL CONTRIBUTIONS TO THERMODYNAMIC FUNCTIONS", re.M)
_ORDER = re.compile(r"ORDER OF POLYNOMIALS:\s*(\d+)")
_AT_T = re.compile(r"AT \(T =\s*(" + A._FLOAT + r")\s*K")
_VOL_LINE = re.compile(r"PRIMITIVE CELL - CENTRING CODE\s+\S+\s+VOLUME=\s*(" + A._FLOAT + r")")


def _row(name: str, block: str, col: int) -> Optional[float]:
    m = re.search(r"^\s*" + re.escape(name) + r"\s*:\s*(" + A._FLOAT + r")\s+(" + A._FLOAT + r")\s+(" + A._FLOAT + r")\s*$",
                  block, re.M)
    return A._f(m.group(col)) if m else None


def gruneisen_table(text: str) -> Optional[Dict[str, List[float]]]:
    """CRYSTAL's Grueneisen-model table: {temperature_k, alpha_per_k, volume_ang3, gamma, cp_minus_cv_j_mol_k}."""
    m = A._last(_GRUN_HEAD, text)
    if not m:
        return None
    rows = A._number_table(text, m.end(), 5)
    if not rows:
        return None
    keys = ("temperature_k", "alpha_per_k", "volume_ang3", "gamma", "cp_minus_cv_j_mol_k")
    return {k: [r[i] for r in rows] for i, k in enumerate(keys)}


def harmonic_blocks(text: str) -> List[Dict[str, Any]]:
    """Each volume's harmonic thermodynamics, sorted by volume: {volume_ang3, el_au, e0_au, temperature_k,
    et_au, ts_au, entropy_j_mol_k, cv_j_mol_k} ([] for an output without them)."""
    heads = list(_HARM_HEAD.finditer(text))
    out = []
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        stop = text.find("QUASI-HARMONIC APPROXIMATION", m.end(), end)
        block = text[m.end():stop if stop > 0 else end]
        vol = None
        for vm in _VOL_LINE.finditer(text, 0, m.start()):
            vol = A._f(vm.group(1))
        head = block.split("THERMODYNAMIC FUNCTIONS WITH VIBRATIONAL CONTRIBUTIONS")[0]
        b: Dict[str, Any] = {"volume_ang3": vol, "el_au": _row("EL", head, 1), "e0_au": _row("E0", head, 1),
                             "temperature_k": [], "et_au": [], "ts_au": [], "entropy_j_mol_k": [], "cv_j_mol_k": []}
        for part in block.split("THERMODYNAMIC FUNCTIONS WITH VIBRATIONAL CONTRIBUTIONS")[1:]:
            t = _AT_T.search(part)
            vals = (_row("ET", part, 1), _row("TS", part, 1), _row("ENTROPY", part, 3), _row("HEAT CAPACITY", part, 3))
            if t and None not in vals:
                b["temperature_k"].append(A._f(t.group(1)))
                for k, v in zip(("et_au", "ts_au", "entropy_j_mol_k", "cv_j_mol_k"), vals):
                    b[k].append(v)
        if b["temperature_k"] and vol is not None and b["el_au"] is not None and b["e0_au"] is not None:
            out.append(b)
    out.sort(key=lambda x: x["volume_ang3"])
    return out


def pvt_table(text: str) -> Dict[float, Dict[str, List[float]]]:
    """CRYSTAL's PVT relation by temperature: {T: {volume_ang3, pressure_gpa, f_au, g_au}}."""
    m = _PVT_HEAD.search(text)
    if not m:
        return {}
    out: Dict[float, Dict[str, List[float]]] = {}
    for r in A._number_table(text, m.end(), 5, max_rows=100000):
        t = out.setdefault(r[0], {"volume_ang3": [], "pressure_gpa": [], "f_au": [], "g_au": []})
        for k, v in zip(("volume_ang3", "pressure_gpa", "f_au", "g_au"), r[1:]):
            t[k].append(v)
    return out


def _order(text: str) -> int:
    m = _ORDER.search(text, text.find("QUASI-HARMONIC APPROXIMATION") if "QUASI-HARMONIC APPROXIMATION" in text else 0)
    return int(m.group(1)) if m else 2


def _common_temperatures(blocks: List[Dict[str, Any]]) -> List[float]:
    if not blocks:
        return []
    common = set(blocks[0]["temperature_k"])
    for b in blocks[1:]:
        common &= set(b["temperature_k"])
    return sorted(common)


def _at(blocks, key: str, t: float) -> List[float]:
    return [b[key][b["temperature_k"].index(t)] for b in blocks]


def thermo(text: str, qha: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """F, S, C_v, C_p and gamma_th along the zero-pressure V(T), at the temperatures every volume prints
    (None without the harmonic blocks or with fewer than three volumes)."""
    import numpy as np
    blocks = harmonic_blocks(text)
    if len(blocks) < 3 or not qha:
        return None
    order = min(_order(text), len(blocks) - 1)
    vols = np.array([b["volume_ang3"] for b in blocks])
    T_q = qha["temperature_k"]
    rows = {k: [] for k in ("temperature_k", "volume_ang3", "f_au", "f_kj_mol", "entropy_j_mol_k", "cv_j_mol_k",
                            "cp_j_mol_k", "gamma_th")}
    for t in _common_temperatures(blocks):
        if t not in T_q:
            continue
        i = T_q.index(t)
        v = qha["volume_ang3"][i]
        el, e0 = np.array([b["el_au"] for b in blocks]), np.array([b["e0_au"] for b in blocks])
        fvib = e0 + np.array(_at(blocks, "et_au", t)) - np.array(_at(blocks, "ts_au", t))
        ev = lambda y: float(np.polyval(np.polyfit(vols, y, order), v))     # noqa: E731
        F = float(np.polyval(np.polyfit(vols, el, min(3, len(vols) - 1)), v)) + ev(fvib)
        S, Cv = ev(np.array(_at(blocks, "entropy_j_mol_k", t))), ev(np.array(_at(blocks, "cv_j_mol_k", t)))
        cp = Cv + qha["cp_minus_cv_j_mol_k"][i]
        vm = v * 1e-30 * N_AVOGADRO
        # undefined where alpha prints as zero or C_v vanishes (the lowest temperatures)
        alpha = qha["alpha_v_per_k"][i]
        gam = alpha * qha["bulk_modulus_t_gpa"][i] * 1e9 * vm / Cv if Cv > 1e-2 and alpha > 0 else None
        for k, x in (("temperature_k", t), ("volume_ang3", v), ("f_au", F), ("f_kj_mol", F * HARTREE_KJ_MOL),
                     ("entropy_j_mol_k", S), ("cv_j_mol_k", Cv), ("cp_j_mol_k", cp), ("gamma_th", gam)):
            rows[k].append(x)
    if not rows["temperature_k"]:
        return None
    # where C_v is under a fifth of its largest value alpha is tiny and gamma_th is numerical noise
    cmax = max(rows["cv_j_mol_k"])
    rows["gamma_th"] = [g if g is not None and c >= 0.2 * cmax else None for g, c in zip(rows["gamma_th"], rows["cv_j_mol_k"])]
    rows["volumes_ang3"] = vols.tolist()
    rows["polynomial_order"] = order
    return rows


def free_energy_family(text: str, qha: Optional[Dict[str, Any]] = None, n_curves: int = 7) -> Optional[Dict[str, Any]]:
    """F(V) at up to ``n_curves`` temperatures (the first, the last and evenly between) with each curve's
    minimum: from the PVT table, else a polynomial through the computed volumes' F(V_i, T)
    (None when the output has neither)."""
    import numpy as np
    pvt = pvt_table(text)
    blocks = harmonic_blocks(text)
    source = "pvt" if pvt else "harmonic" if len(blocks) >= 3 else None
    if source is None:
        return None
    temps = sorted(pvt) if pvt else _common_temperatures(blocks)
    if not temps:
        return None
    pick = sorted({temps[round(k * (len(temps) - 1) / max(1, n_curves - 1))] for k in range(min(n_curves, len(temps)))})
    curves = []
    order = min(_order(text), max(1, len(blocks) - 1))
    for t in pick:
        if pvt:
            V, F = np.array(pvt[t]["volume_ang3"]), np.array(pvt[t]["f_au"])
        else:
            # the static energy through every computed volume (a polynomial of one order less than their
            # number), the vibrational free energy with CRYSTAL's order, as its QHA module fits them apart
            vols = np.array([b["volume_ang3"] for b in blocks])
            el = np.array([b["el_au"] for b in blocks])
            fv = np.array([b["e0_au"] for b in blocks]) + np.array(_at(blocks, "et_au", t)) - np.array(_at(blocks, "ts_au", t))
            V = np.linspace(vols.min(), vols.max(), 61)
            F = np.polyval(np.polyfit(vols, el, min(3, len(vols) - 1)), V) + np.polyval(np.polyfit(vols, fv, order), V)
        k = int(np.argmin(F))
        vmin, fmin = float(V[k]), float(F[k])
        if 0 < k < len(V) - 1:                  # the parabola through the lowest point and its neighbors
            c = np.polyfit(V[k - 1:k + 2], F[k - 1:k + 2], 2)
            if c[0] > 0:
                vmin = float(-c[1] / (2 * c[0]))
                fmin = float(np.polyval(c, vmin))
        curves.append({"temperature_k": t, "volume_ang3": V.tolist(), "f_au": F.tolist(), "v_min": vmin, "f_min_au": fmin,
                       "interior": bool(0 < k < len(V) - 1)})
    points = []
    if blocks:
        for t in pick:
            if all(t in b["temperature_k"] for b in blocks):
                points.append({"temperature_k": t, "volume_ang3": [b["volume_ang3"] for b in blocks],
                               "f_au": [b["el_au"] + b["e0_au"] + b["et_au"][b["temperature_k"].index(t)]
                                        - b["ts_au"][b["temperature_k"].index(t)] for b in blocks]})
    return {"source": source, "curves": curves, "points": points}


def summary(text: str, qha: Dict[str, Any]) -> Dict[str, Any]:
    """The additions analyze() files under qha: {gruneisen, thermo} when the output prints them."""
    out: Dict[str, Any] = {}
    g = gruneisen_table(text)
    if g:
        out["gruneisen"] = g
    th = thermo(text, qha)
    if th:
        out["thermo"] = th
    return out
