"""Thermodynamics of a FREQ run at every temperature (and pressure) it printed.

The extractor's scalar keys (thermal_energy_*, entropy_term_*,
gibbs_free_energy_*, enthalpy_*, entropy_*, heat_capacity_*,
thermodynamic_temperature_k) are those of the FIRST "AT (T = ..., P = ...)"
block of the output. A FREQCALC with TEMPERAT (or PRESSURE) prints one block
per temperature (and pressure); this property keeps all of them as one curve,
with the same names per point, so G(T), H(T), S(T) and Cv(T) are in the
database. A run that printed one block gets nothing here (its scalars say it
all).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from mace.properties.registry import Output, Skip, derived

_NUM = r"([-+]?\d+\.\d+(?:[EeDd][-+]?\d+)?|[-+]?\d+(?:[EeDd][-+]?\d+)?)"
_AT = re.compile(r"THERMODYNAMIC FUNCTIONS (?:WITH VIBRATIONAL CONTRIBUTIONS|TAKING INTO ACCOUNT MOLECULAR)"
                 r".*?AT \(T =\s*([\d.]+)\s*K,\s*P =\s*([\d.EeDd+-]+)\s*MPA\)", re.DOTALL)
_ROW = {
    "thermal_energy": r"ET",
    "pv_term": r"PV",
    "entropy_term": r"TS",
    "free_energy_correction": r"ET\+PV-TS",
    "gibbs_free_energy": r"EL\+E0\+ET\+PV-TS",
}
_OTHER = {"entropy": r"ENTROPY", "heat_capacity": r"HEAT CAPACITY"}
_E_UNITS = ("au", "ev", "kj_mol")
_S_UNITS = ("mhartree_cell_k", "mev_cell_k", "j_mol_k")
UNITS = {"temperature_k": "K", "pressure_mpa": "MPa", "au": "Hartree", "ev": "eV", "kj_mol": "kJ/mol",
         "mhartree_cell_k": "mHartree/(cell·K)", "mev_cell_k": "meV/(cell·K)", "j_mol_k": "J/(mol·K)"}


def _f(x: str) -> float:
    return float(x.replace("D", "E").replace("d", "e"))


def _three(text: str, label: str):
    m = re.search(r"^\s*" + label + r"\s*:\s*" + r"\s+".join([_NUM] * 3), text, re.MULTILINE)
    return [_f(v) for v in m.groups()] if m else None


def points(text: str) -> List[Dict[str, Any]]:
    """Every thermodynamics block of a FREQ output: {temperature_k,
    pressure_mpa, <name>_<unit> ...} with the names of the extractor's scalar
    keys (enthalpy = G + TS, per unit, as the extractor builds it)."""
    hits = list(_AT.finditer(text))
    out = []
    for i, m in enumerate(hits):
        end = hits[i + 1].start() if i + 1 < len(hits) else min(len(text), m.end() + 4000)
        block = text[m.end():end]
        p: Dict[str, Any] = {"temperature_k": float(m.group(1)), "pressure_mpa": _f(m.group(2))}
        cut = block.find("OTHER THERMODYNAMIC FUNCTIONS")
        energies = block if cut < 0 else block[:cut]
        for name, label in _ROW.items():
            v = _three(energies, label)
            if v:
                p.update({f"{name}_{u}": x for u, x in zip(_E_UNITS, v)})
        if "gibbs_free_energy_au" in p and "entropy_term_au" in p:
            for u in _E_UNITS:
                p[f"enthalpy_{u}"] = p[f"gibbs_free_energy_{u}"] + p[f"entropy_term_{u}"]
        if cut >= 0:
            for name, label in _OTHER.items():
                v = _three(block[cut:], label)
                if v:
                    p.update({f"{name}_{u}": x for u, x in zip(_S_UNITS, v)})
        out.append(p)
    return out


@derived("thermodynamics_vs_temperature", calc_types=("FREQ",), category="thermodynamics",
         group="Thermodynamics", version="1",
         outputs=(Output("thermodynamics_vs_temperature", "JSON", "json", blob=True,
                         label="G, H, S, C<sub>v</sub> vs T",
                         description="every temperature (and pressure) the FREQ run printed (TEMPERAT, PRESSURE): "
                                     "ET, PV, TS, ET+PV-TS, G, H = G + TS, S and Cv per point, in each of "
                                     "CRYSTAL's units; the scalar keys are the first point's"),
                  Output("thermodynamic_n_points", "dimensionless", "int", label="Temperatures printed",
                         precision=6,
                         description="(T, P) points of thermodynamics_vs_temperature")),
         source="the .out's THERMODYNAMIC FUNCTIONS ... AT (T = .., P = ..) blocks")
def _thermodynamics_vs_temperature(ctx):
    """Every temperature and pressure of a FREQ run's thermodynamics."""
    pts = points(ctx.content)
    if len(pts) < 2:
        raise Skip("one temperature printed (the scalar keys hold it)")
    names = sorted({k for p in pts for k in p if k not in ("temperature_k", "pressure_mpa")})
    units = {"temperature_k": "K", "pressure_mpa": "MPa"}
    for n in names:
        units[n] = next(UNITS[u] for u in (*_E_UNITS, *_S_UNITS) if n.endswith("_" + u))
    curve = {"points": pts, "units": units,
             "zero_point_energy_au": ctx.value("zero_point_energy_au"),
             "electronic_energy_au": ctx.value("electronic_energy_au")}
    return {"thermodynamics_vs_temperature": curve, "thermodynamic_n_points": len(pts)}
