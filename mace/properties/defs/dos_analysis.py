"""Derived properties of a DOSS run, from its DOSS.DAT as the GUI's DOS plot
reads it (mace/gui/results.dos_data: energies in eV from E_F, every
projection in states/eV/cell with the label of the deck, "Ti (d)", and the
spin channels as the plot shows them).

* Sommerfeld coefficient of a metal from the extractor's N(E_F)
  (dos_at_fermi_total, both spins, states/eV/cell):
  gamma = (pi^2 / 3) k_B^2 N(E_F) = 2.3571 mJ/(mol K^2) per state/eV, per
  mole of primitive cells (and per mole of atoms);
* d-band center of each transition-metal d projection, in the two
  conventions of the literature: the first moment of the projected DOS
  (Hammer-Norskov) over the DOSS energy window up to E_F + D_BAND_MAX_EV
  (occupied and empty d states; the cutoff keeps the basis set's diffuse d
  functions out, which put Ag4Cl2S's full-range center at +3.8 eV), and the
  first moment of the occupied part only (up to E_F); with the filling (the
  share below E_F) and width (the square root of the second central moment)
  over the first window;
* orbital/atom character of the band edges: each projection's share of
  the total DOS within EDGE_WINDOW_EV below the VBM and above the CBM (the
  edges of the extractor's DOS gap, dos_structure), or within
  FERMI_WINDOW_EV of E_F for a metal. Per element the orbital projections
  are used ("Ti (d)"), else its all-orbital one; a deck that projects
  nothing gives nothing;
* carrier concentrations against the Fermi level at 300 K, n(mu) =
  int_CB g f and p(mu) = int_VB g (1 - f), per cm^3 (3D), cm^2 (2D) or cm
  (1D) of the primitive cell, with the intrinsic level (n = p) and
  carrier concentration n_i. The DOS inside the gap (the extractor's DOS
  gap) is set to zero: CRYSTAL's Fourier-Legendre DOS leaks a tail into
  the gap that would otherwise outweigh exp(-E_g / 2kT);
* density-of-states effective masses of electrons and holes, from
  mace.transport.mass.dos_mass (mace tau's DOS fit of the integrated DOS
  near the edge): every band and every equivalent valley at the edge. A
  fit with R^2 below MIN_R2 is kept in the details only. Rough: the DOS
  near an edge is CRYSTAL's smoothed one (NPOL Legendre terms, the energy
  grid); the accurate masses are the BAND run's m_c and m_d
  (effective_masses).

The DOS of a spin-polarized run is per channel (alpha, beta) when the two
differ, and summed otherwise (the plot's rule, presets.dos_spin_polarized);
scalars are of both spins together, the details keep each channel.
"""
from __future__ import annotations

import functools
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mace.properties.registry import Output, Sibling, Skip, derived

from mace.constants import BOHR_TO_ANGSTROM, HARTREE_TO_EV as HARTREE_EV  # noqa: E402
K_B_EV = 8.617333262e-5
FARADAY = 96485.33212                    # e * N_A, J/(mol eV)
BOHR_CM = BOHR_TO_ANGSTROM * 1e-8
#: mJ/(mol K^2) per state/eV: (pi^2/3) k_B^2 e N_A
GAMMA_PER_STATE = math.pi ** 2 / 3.0 * K_B_EV ** 2 * FARADAY * 1000.0
EDGE_WINDOW_EV = 0.5
FERMI_WINDOW_EV = 0.25
TEMPERATURE_K = 300.0
#: a DOS mass fit below this R^2 is not stored
MIN_R2 = 0.9
#: the d-band moments use E <= E_F + D_BAND_MAX_EV: above it the d projection of an LCAO run is that of
#: the basis set's diffuse and polarization d functions (Ag4Cl2S: 17 of its 49 Ag d states lie above
#: E_F + 10 eV), not of the metal's d band (a d0 oxide's empty t2g and eg lie within 2-8 eV)
D_BAND_MAX_EV = 10.0
TRANSITION_METALS = frozenset(
    "Sc Ti V Cr Mn Fe Co Ni Cu Zn Y Zr Nb Mo Tc Ru Rh Pd Ag Cd La Hf Ta W Re Os Ir Pt Au Hg "
    "Ac Rf Db Sg Bh Hs Mt Ds Rg Cn".split())


@functools.lru_cache(maxsize=8)
def _read(path: str, mtime: float, size: int) -> Dict[str, Any]:
    import numpy as np
    from mace.gui import results
    from mace.plotting.presets import parse_projection
    d = results.dos_data(Path(path), reference="fermi")
    E = np.asarray(d["energy"], float)
    chans = ("alpha", "beta") if d["spin_polarized"] else ("total",)
    projs = []
    for p in d["projections"]:
        # Mulliken projections can dip below zero: kept as they are (the shares add up to the total)
        up = np.asarray(p["up"], float)
        vals = {"total": up}
        if d["spin_polarized"]:
            dn = -np.asarray(p["down"], float)
            vals = {"alpha": up, "beta": dn, "total": up + dn}
        projs.append({"label": p["label"].strip(), **parse_projection(p["label"]), "dos": vals})
    return {"file": Path(path).name, "energy": E, "channels": chans, "spin_polarized": bool(d["spin_polarized"]),
            "projections": projs}


def read(dos_file: Path) -> Dict[str, Any]:
    """The DOSS.DAT as the GUI's DOS plot reads it (cached per file): energy (eV from E_F), channels,
    projections [{label, element, orbital, dos: {channel: states/eV/cell}}], the total last."""
    p = Path(dos_file)
    st = p.stat()
    return _read(str(p), st.st_mtime, st.st_size)


def _trapz(y, x) -> float:
    import numpy as np
    f = getattr(np, "trapezoid", None) or np.trapz
    return float(f(y, x)) if len(x) > 1 else 0.0


def _total(d) -> Dict[str, Any]:
    return d["projections"][-1]["dos"]


def _named(d) -> List[Dict[str, Any]]:
    """The projections that name an element: its orbital projections, else its all-orbital one."""
    named = [p for p in d["projections"][:-1] if p.get("element")]
    out = []
    for el in dict.fromkeys(p["element"] for p in named):
        orb = [p for p in named if p["element"] == el and p["orbital"] in ("s", "p", "d", "f")]
        out += orb or [p for p in named if p["element"] == el and p["orbital"] == "all"][:1]
    return out


def _name(p) -> str:
    from mace.plotting.presets import legend_name
    return legend_name(p["label"])


def d_band(d) -> Dict[str, Any]:
    """{element: {channel: center, filling, width, occupied center, states, electrons}} of each
    transition-metal d projection."""
    import numpy as np
    keep = d["energy"] <= D_BAND_MAX_EV
    E = d["energy"][keep]
    occ = E <= 0.0
    out: Dict[str, Any] = {}
    for p in d["projections"][:-1]:
        if p.get("orbital") != "d" or p.get("element") not in TRANSITION_METALS:
            continue
        rec = {}
        for ch, rho in p["dos"].items():
            rho = rho[keep]
            w = _trapz(rho, E)
            if w <= 1e-6:
                continue
            c = _trapz(E * rho, E) / w
            wo = _trapz(rho[occ], E[occ])
            rec[ch] = {"center_ev": c, "filling": wo / w,
                       "width_ev": math.sqrt(max(_trapz((E - c) ** 2 * rho, E) / w, 0.0)),
                       "occupied_center_ev": (_trapz(E[occ] * rho[occ], E[occ]) / wo) if wo > 1e-6 else None,
                       "states": w, "electrons": wo}
        if rec:
            out[p["element"]] = rec
    return out


def composition(d, lo: float, hi: float) -> Dict[str, Any]:
    """Each named projection's share of the total DOS between ``lo`` and ``hi`` (eV from E_F), per channel."""
    E = d["energy"]
    m = (E >= lo) & (E <= hi)
    if m.sum() < 2:
        return {}
    tot = _total(d)
    out: Dict[str, Any] = {"window_ev": [lo, hi], "channels": {}}
    for ch, t in tot.items():
        wt = _trapz(t[m], E[m])
        if wt <= 1e-9:
            continue
        shares = {_name(p): _trapz(p["dos"][ch][m], E[m]) / wt for p in _named(d)}
        by_el: Dict[str, float] = {}
        for p in _named(d):
            by_el[p["element"]] = by_el.get(p["element"], 0.0) + shares[_name(p)]
        out["channels"][ch] = {"states": wt, "shares": dict(sorted(shares.items(), key=lambda t: -t[1])),
                               "by_element": dict(sorted(by_el.items(), key=lambda t: -t[1])),
                               "unassigned": 1.0 - sum(shares.values())}
    return out


def _edges(ctx) -> Tuple[Optional[float], Optional[float]]:
    ds = ctx.value("dos_structure") or {}
    if not isinstance(ds, dict):
        return None, None
    return ds.get("vbm_ev"), ds.get("cbm_ev")


def _cell(ctx, dim: int) -> Tuple[float, str]:
    """(size of the primitive cell in cm^dim, unit of a carrier concentration)."""
    from mace.transport import mass
    lat = mass.lattice_from_output(ctx.output_file)
    if lat is None:
        raise Skip("no lattice vectors in the output (DIRECT LATTICE VECTOR COMPONENTS)")
    size = mass._cell_size(lat, dim) * BOHR_CM ** dim
    return size, {3: "cm⁻³", 2: "cm⁻²", 1: "cm⁻¹"}[dim]


def carriers(E, g, vbm: float, cbm: float, T: float = TEMPERATURE_K, mus=None):
    """(n(mu), p(mu)) per cell for the total DOS ``g`` (states/eV/cell) with the gap (vbm, cbm) emptied."""
    import numpy as np
    kT = K_B_EV * T
    g = np.where((E > vbm) & (E < cbm), 0.0, np.clip(g, 0.0, None))
    cb, vb = E >= cbm, E <= vbm
    mus = np.atleast_1d(np.asarray(mus, float))
    n = np.array([_trapz(g[cb] * np.exp(-np.logaddexp(0.0, (E[cb] - mu) / kT)), E[cb]) for mu in mus])
    p = np.array([_trapz(g[vb] * np.exp(-np.logaddexp(0.0, (mu - E[vb]) / kT)), E[vb]) for mu in mus])
    return n, p


def intrinsic_level(E, g, vbm: float, cbm: float, T: float = TEMPERATURE_K) -> Optional[float]:
    """The Fermi level where n = p (bisection on log n - log p, which rises with mu)."""
    import numpy as np

    def h(mu):
        n, p = carriers(E, g, vbm, cbm, T, [mu])
        if n[0] <= 0 or p[0] <= 0:
            return None
        return math.log(n[0]) - math.log(p[0])
    lo, hi = vbm, cbm
    hl, hh = h(lo), h(hi)
    if hl is None or hh is None or not (hl < 0 < hh):
        return None
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        hm = h(mid)
        if hm is None:
            return None
        if hm < 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-7:
            break
    return float(np.mean([lo, hi]))


def _atoms(ctx) -> Optional[int]:
    n = ctx.value("atoms_in_unit_cell")
    if n:
        return int(n)
    m = re.search(r"N\. OF ATOMS PER CELL\s+(\d+)", ctx.content or "")
    return int(m.group(1)) if m else None


def _dos_file(ctx):
    p = ctx.sibling("DOSS")
    try:
        return read(p)
    except (OSError, ValueError, IndexError, KeyError) as e:
        raise Skip(f"the DOSS file could not be read: {e}")


_SIB = (Sibling("DOSS", description="the run's DOSS.DAT"),)
_SRC = "DOSS.DAT via mace/gui/results.dos_data (the GUI's DOS plot reader, projection labels from the .d3)"


@derived("sommerfeld_coefficient", calc_types=("DOSS",), category="density_of_states", group="Fermi level",
         requires=("dos_at_fermi_total",), version="1",
         source="dos_at_fermi_total (N(E_F), both spins, states/eV/cell) of the extractor",
         outputs=(Output("sommerfeld_gamma_mj_mol_k2", "mJ/(mol·K²)", label="Sommerfeld γ (per mol cells)",
                         precision=4, description="electronic heat capacity coefficient (π²/3) k_B² N(E_F), "
                                                  "per mole of primitive cells (a metal)"),
                  Output("sommerfeld_gamma_per_atom_mj_mol_k2", "mJ/(mol·K²)", label="Sommerfeld γ (per mol atoms)",
                         precision=4, description="(π²/3) k_B² N(E_F) per mole of atoms")))
def _sommerfeld(ctx):
    """Sommerfeld coefficient of a metal from N(E_F)."""
    if not ctx.value("dos_is_metallic"):
        raise Skip("not a metal (the DOS has a gap at E_F): no electronic heat capacity linear in T")
    g = GAMMA_PER_STATE * float(ctx.value("dos_at_fermi_total"))
    n = _atoms(ctx)
    return {"sommerfeld_gamma_mj_mol_k2": g, "sommerfeld_gamma_per_atom_mj_mol_k2": g / n if n else None}


@derived("d_band_center", calc_types=("DOSS",), category="density_of_states", group="d band", siblings=_SIB,
         version="2", source=_SRC,
         outputs=(Output("d_band_center_ev", "eV", label="ε<sub>d</sub> d-band center (to E<sub>F</sub> + 10 eV)",
                         precision=4,
                         description="first moment of the transition metal's d projected DOS, occupied and empty "
                                     "states, over the DOSS window up to E_F + 10 eV (the cutoff keeps the basis "
                                     "set's diffuse d functions out), from E_F (the metal with the most d states "
                                     "when there are several)"),
                  Output("d_band_center_occupied_ev", "eV", label="ε<sub>d</sub> d-band center (occupied, to "
                         "E<sub>F</sub>)", precision=4,
                         description="first moment of the occupied part of the d projected DOS (from the bottom "
                                     "of the DOSS window up to E_F), from E_F: the other convention of the "
                                     "literature"),
                  Output("d_band_filling", "dimensionless", label="d-band filling (to E<sub>F</sub> + 10 eV)",
                         precision=4,
                         description="share of the d projected DOS below E_F, over the DOSS window up to "
                                     "E_F + 10 eV"),
                  Output("d_band_width_ev", "eV", label="d-band width (2nd moment, to E<sub>F</sub> + 10 eV)",
                         precision=4,
                         description="square root of the second central moment of the d projected DOS up to "
                                     "E_F + 10 eV"),
                  Output("d_band_element", "category", "text", label="d-band element",
                         description="the transition metal the d-band scalars are of"),
                  Output("d_band", "JSON", "json", blob=True, label="d band (details)",
                         description="per transition metal and spin channel: center, occupied-part center, "
                                     "filling, width, d states and d electrons in the window")))
def _d_band_center(ctx):
    """d-band center, filling and width of each transition-metal d projection."""
    d = _dos_file(ctx)
    db = d_band(d)
    if not db:
        raise Skip("no transition-metal d projection in the DOSS deck")
    el = max(db, key=lambda e: db[e]["total"]["states"])
    t = db[el]["total"]
    E = d["energy"]
    return {"d_band_center_ev": t["center_ev"], "d_band_center_occupied_ev": t["occupied_center_ev"],
            "d_band_filling": t["filling"], "d_band_width_ev": t["width_ev"], "d_band_element": el,
            "d_band": {"window_ev": [float(E[0]), float(min(E[-1], D_BAND_MAX_EV))],
                       "spin_polarized": d["spin_polarized"],
                       "elements": db}}


@derived("band_edge_character", calc_types=("DOSS",), category="density_of_states", group="Band-edge character",
         siblings=_SIB, version="1", source=_SRC + "; the edges of dos_structure",
         outputs=(Output("vbm_character", "category", "text", label="VBM character",
                         description=f"projection with the largest share of the DOS within {EDGE_WINDOW_EV} eV "
                                     "below the VBM"),
                  Output("vbm_character_fraction", "dimensionless", label="VBM character share", precision=3,
                         description="its share of the total DOS in that window"),
                  Output("cbm_character", "category", "text", label="CBM character",
                         description=f"projection with the largest share of the DOS within {EDGE_WINDOW_EV} eV "
                                     "above the CBM"),
                  Output("cbm_character_fraction", "dimensionless", label="CBM character share", precision=3,
                         description="its share of the total DOS in that window"),
                  Output("fermi_level_character", "category", "text", label="E<sub>F</sub> character",
                         description=f"projection with the largest share of the DOS within ±{FERMI_WINDOW_EV} eV "
                                     "of E_F (a metal)"),
                  Output("fermi_level_character_fraction", "dimensionless", label="E<sub>F</sub> character share",
                         precision=3, description="its share of the total DOS in that window"),
                  Output("band_edge_composition", "JSON", "json", blob=True, label="Band-edge composition",
                         description="per edge and spin channel: each projection's and each element's share of the "
                                     "DOS in the edge window, and the share no projection covers")))
def _band_edge_character(ctx):
    """Orbital and atom character of the band edges from the projected DOS."""
    d = _dos_file(ctx)
    if not _named(d):
        raise Skip("the DOSS deck projects on no element (only the total DOS)")
    vbm, cbm = _edges(ctx)
    res: Dict[str, Any] = {"spin_polarized": d["spin_polarized"]}
    vals: Dict[str, Any] = {}
    if ctx.value("dos_is_metallic"):
        res["fermi"] = composition(d, -FERMI_WINDOW_EV, FERMI_WINDOW_EV)
        keys = {"fermi": "fermi_level_character"}
    elif vbm is None or cbm is None:
        raise Skip("no band edges in the DOS (dos_structure found no gap window)")
    else:
        res["vbm"] = composition(d, vbm - EDGE_WINDOW_EV, vbm)
        res["cbm"] = composition(d, cbm, cbm + EDGE_WINDOW_EV)
        keys = {"vbm": "vbm_character", "cbm": "cbm_character"}
    for edge, key in keys.items():
        t = (res.get(edge) or {}).get("channels", {}).get("total")
        if t and t["shares"]:
            name, share = next(iter(t["shares"].items()))
            vals[key] = name
            vals[key + "_fraction"] = share
    if not vals:
        step = float(d["energy"][1] - d["energy"][0]) if len(d["energy"]) > 1 else 0.0
        raise Skip(f"the total DOS in the edge windows is zero or negative (energy grid {step:.3g} eV)")
    vals["band_edge_composition"] = res
    return vals


def _carrier_outputs():
    out = []
    for dim, suf, unit in ((3, "cm3", "cm⁻³"), (2, "cm2", "cm⁻²"), (1, "cm1", "cm⁻¹")):
        out.append(Output(f"intrinsic_carrier_concentration_300k_{suf}", unit,
                          label=f"n<sub>i</sub> (300 K, {dim}D)", precision=4,
                          description=f"intrinsic carrier concentration n = p at 300 K, per {unit[:2]} "
                                      f"of a {dim}D cell"))
    return tuple(out)


@derived("carrier_concentration", calc_types=("DOSS",), category="density_of_states", group="Carriers",
         siblings=_SIB, version="1", source=_SRC + "; the gap of dos_structure; the cell of the .out",
         outputs=(Output("intrinsic_fermi_level_300k_ev", "eV", label="E<sub>F,i</sub> (300 K)", precision=4,
                         description="Fermi level where n = p at 300 K, from E_F of the run (CRYSTAL's E_F of an "
                                     "insulator is the valence-band top)"),)
         + _carrier_outputs()
         + (Output("carrier_concentration_300k", "JSON", "json", blob=True, label="n, p vs E<sub>F</sub> (300 K)",
                   description="electron and hole concentrations against the Fermi level at 300 K, from 0.3 eV "
                               "below the VBM to 0.3 eV above the CBM"),))
def _carrier_concentration(ctx):
    """Electron and hole concentrations against the Fermi level at 300 K."""
    import numpy as np
    dim = ctx.dimensionality or 3
    if dim == 0:
        raise Skip("a molecule has no carrier concentration")
    if ctx.value("dos_is_metallic"):
        raise Skip("a metal: the carriers do not follow the Fermi level through a gap")
    vbm, cbm = _edges(ctx)
    if vbm is None or cbm is None or cbm <= vbm:
        raise Skip("no gap in the DOS (dos_structure has no band edges)")
    d = _dos_file(ctx)
    E = d["energy"]
    g = _total(d)["total"]
    size, unit = _cell(ctx, dim)
    span = (cbm - vbm) + 0.6
    step = max(0.01, span / 300.0)
    mus = np.arange(vbm - 0.3, cbm + 0.3 + step / 2, step)
    n, p = carriers(E, g, vbm, cbm, TEMPERATURE_K, mus)
    sig = lambda a: [float(f"{v:.4g}") for v in a]          # noqa: E731
    vals: Dict[str, Any] = {}
    mu_i = intrinsic_level(E, g, vbm, cbm)
    curve = {"temperature_k": TEMPERATURE_K, "dimensionality": dim, "vbm_ev": vbm, "cbm_ev": cbm,
             "cell_size": size, "cell_size_unit": {3: "cm³", 2: "cm²", 1: "cm"}[dim],
             "mu_ev": [round(float(m), 4) for m in mus], "n": sig(n / size), "p": sig(p / size),
             "units": {"mu_ev": "eV from E_F of the run", "n": unit, "p": unit}}
    if mu_i is not None:
        ni, pi = carriers(E, g, vbm, cbm, TEMPERATURE_K, [mu_i])
        vals["intrinsic_fermi_level_300k_ev"] = mu_i
        vals[f"intrinsic_carrier_concentration_300k_cm{dim}"] = float(math.sqrt(ni[0] * pi[0])) / size
        curve["intrinsic"] = {"mu_ev": mu_i, "n": float(ni[0]) / size, "p": float(pi[0]) / size}
    vals["carrier_concentration_300k"] = curve
    return vals


@derived("dos_effective_mass", calc_types=("DOSS",), category="density_of_states", group="DOS mass",
         siblings=_SIB, version="1",
         source="DOSS.DAT via mace.transport.mass.dos_mass (mace tau's DOS fit, the integrated DOS near the edge)",
         outputs=(Output("dos_effective_mass_electron", "m_e",
                         label="DOS mass m<sub>d</sub>, electrons (rough, from the smoothed DOS)",
                         precision=4, description="rough density-of-states mass of electrons at the CBM, fitted to "
                                                  "CRYSTAL's smoothed DOS (sensitive to NPOL and the energy grid). "
                                                  "Use the BAND run's effective_mass_electron_md (accurate)"),
                  Output("dos_effective_mass_hole", "m_e",
                         label="DOS mass m<sub>d</sub>, holes (rough, from the smoothed DOS)", precision=4,
                         description="rough density-of-states mass of holes at the VBM, fitted to CRYSTAL's "
                                     "smoothed DOS (sensitive to NPOL and the energy grid). Use the BAND run's "
                                     "effective_mass_hole_md (accurate)"),
                  Output("dos_effective_masses", "JSON", "json", blob=True, label="DOS masses, rough (details)",
                         description="per carrier: m_d, R², points and window of the fit, onset, NPOL, warnings")))
def _dos_effective_mass(ctx):
    """Density-of-states effective masses from the DOS near the band edges."""
    import contextlib
    import io
    from mace.transport import mass
    dim = ctx.dimensionality or 3
    if dim == 0:
        raise Skip("a molecule has no band edges")
    if ctx.value("dos_is_metallic"):
        raise Skip("a metal has no band edge to fit")
    vals: Dict[str, Any] = {}
    detail: Dict[str, Any] = {"dimensionality": dim}
    for carrier in ("electron", "hole"):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                r = mass.dos_mass(ctx.sibling("DOSS"), carrier, dim)
        except (mass.MassError, OSError, ValueError) as e:
            detail[carrier] = {"error": str(e)}
            continue
        detail[carrier] = {k: (Path(v).name if k == "file" else v) for k, v in r.items()}
        # mace tau prints any fit with its R^2; a stored mass has to be a fit (SOD on a 0.25 eV grid: R^2 0)
        if r["r2"] < MIN_R2 or not 0.01 <= r["m_d"] <= 100.0:
            detail[carrier]["rejected"] = (f"R^2 {r['r2']:.2f} (< {MIN_R2}) or m_d {r['m_d']:.3g} m_e outside "
                                           "0.01..100: not a parabolic band edge on this grid")
            continue
        vals[f"dos_effective_mass_{carrier}"] = r["m_d"]
    if not vals:
        raise Skip("; ".join(f"{c}: {detail[c].get('error') or detail[c].get('rejected')}"
                             for c in ("electron", "hole")))
    vals["dos_effective_masses"] = detail
    return vals
