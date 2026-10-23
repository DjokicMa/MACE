"""The results document of an ELASTIC, EOS, QHA or CPHF run: what the GUI
opens such an output as, and what ``mace analyze elastic --plot`` draws.

The numbers are ``analysis.analyze``'s (``mace analyze elastic``); this
module adds what a document shows besides them:

* ``settings``: the run's key settings for the Inspector (the module, its
  sub-keywords, NUMDERIV and STEPSIZE as the output prints them, the EOS
  volume range, the QHA temperatures, the CPHF order and wavelengths, the
  functional and the basis), each with where it was read from;
* ``views``: the plots the run gives, by kind;
* ``figure``: one plot's curves as plain arrays (the GUI draws them with
  plotly, the CLI with matplotlib, so both show the same numbers).

Directional moduli (Nye, Physical Properties of Crystals, 1985, ch. 8):
with a = (n1^2, n2^2, n3^2, n2 n3, n1 n3, n1 n2) for a unit direction n and
S = C^-1 in Voigt notation (engineering shear strains),
1/E(n) = a^T S a and the linear compressibility beta(n) = a^T S h,
h = (1, 1, 1, 0, 0, 0) (Nye eq. 8.37; Cairns & Goodwin, PCCP 17, 20449
(2015)). A slab (2D) takes a = (c^2, s^2, c s), h = (1, 1, 0) on its
in-plane block xx, yy, xy.

Equations of state (the four CRYSTAL fits, manual ch. 12, p. 284), E(V) and
P(V) = -dE/dV with V0, E0, B0, B0' as CRYSTAL prints them:

* Murnaghan (1944): P = B0/B' [(V0/V)^B' - 1]
* Birch-Murnaghan (1947), third order: f = (V0/V)^(2/3),
  E = E0 + 9 V0 B0/16 {(f - 1)^3 B' + (f - 1)^2 (6 - 4 f)}
* Poirier-Tarantola (1998): x = ln(V0/V), E = E0 + B0 V0 x^2/2 (1 + (B' - 2) x/3)
* Vinet (1987): eta = (V/V0)^(1/3),
  E = E0 + 2 B0 V0/(B' - 1)^2 {2 - [5 + 3 B'(eta - 1) - 3 eta] exp(-3/2 (B' - 1)(eta - 1))}

Standard library and numpy only.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import analysis as A
from . import tensor as T

KINDS = ("ELASTIC", "EOS", "QHA", "CPHF", "FREQ")
GPA_A3_HARTREE = 1e-21 / A.HARTREE_J            # GPa * A^3 in hartree
HARTREE_MEV = 27211.386245988
EOS_LABELS = {"murnaghan": "Murnaghan 1944", "birch_murnaghan": "Birch-Murnaghan 1947",
              "poirier_tarantola": "Poirier-Tarantola 1998", "vinet": "Vinet 1987"}
VOIGT = ("xx", "yy", "zz", "yz", "xz", "xy")
SHAPES = ("surface", "planes")
SHAPED = ("youngs", "compressibility", "shear", "poisson", "piezo", "shg")       # a 3D run: --shape surface|planes

#: every plot a run can give: key -> (kind, label, what it draws)
PLOTS: Dict[str, Tuple[str, str, str]] = {
    "youngs": ("ELASTIC", "Young's modulus", "the directional Young's modulus E(n)"),
    "compressibility": ("ELASTIC", "Linear compressibility", "the directional linear compressibility beta(n)"),
    "shear": ("ELASTIC", "Shear modulus", "the directional shear modulus G(n, m): its minimum and maximum over m perpendicular to n"),
    "poisson": ("ELASTIC", "Poisson's ratio", "Poisson's ratio nu(n, m): its minimum and maximum over m perpendicular to n"),
    "sound": ("ELASTIC", "Sound velocities", "the phase and group velocities of the three acoustic waves (Christoffel) in the xy, xz and yz planes"),
    "piezo": ("ELASTIC", "Piezoelectric", "the longitudinal piezoelectric coefficient d_L(n) (e_L(n) when the run gives no d)"),
    "eos": ("EOS", "E(V)", "the energy of each volume with the four fitted equations of state"),
    "pv": ("EOS", "P(V)", "the pressure of the four fitted equations of state"),
    "bp": ("EOS", "B(P)", "the bulk modulus of each fit along its pressure"),
    "vp": ("EOS", "V/V<sub>0</sub>(P)", "the volume of each fit along its pressure, relative to its V0"),
    "residuals": ("EOS", "Residuals", "the computed energies minus each fit, with the RMS of each"),
    "enthalpy": ("EOS", "Enthalpy", "H(P) = E + PV per formula unit; with other phases (--compare), H - H_ref and the transition pressures"),
    "volume": ("QHA", "V(T)", "the equilibrium volume at zero pressure"),
    "vratio": ("QHA", "V(T)/V<sub>0</sub>", "the volume and lattice parameters relative to the lowest temperature"),
    "expansion": ("QHA", "Thermal expansion", "the volume and linear thermal expansion coefficients"),
    "bulk": ("QHA", "B(T)", "the isothermal and adiabatic bulk moduli"),
    "heat": ("QHA", "Heat capacity", "Cp and Cv along V(T) when the output has each volume's harmonic thermodynamics, Cp - Cv, each volume's Cv"),
    "gamma": ("QHA", "Grüneisen γ(T)", "CRYSTAL's mode Grueneisen parameter and the thermodynamic one, alpha B V / Cv"),
    "free": ("QHA", "F(T)", "the Helmholtz free energy along V(T)"),
    "entropy": ("QHA", "S(T)", "the vibrational entropy along V(T)"),
    "fv": ("QHA", "F(V)", "the free energy against volume at several temperatures, each curve's minimum traced"),
    "epsilon": ("CPHF", "ε(ω)", "the dielectric tensor at each frequency (DYNAMIC)"),
    "refractive": ("CPHF", "n(ω)", "the refractive index (and extinction coefficient) of each diagonal component (DYNAMIC)"),
    "absorption": ("CPHF", "Absorption", "the absorption coefficient 2 omega kappa / c of each diagonal component (DYNAMIC with DAMPING)"),
    "indicatrix": ("CPHF", "Index ellipsoid", "the refractive-index ellipsoid: semi-axes n along the principal axes of eps"),
    "shg": ("CPHF", "SHG |d(n)|", "the second-harmonic polarization |d_ijk n_j n_k| for a pump polarized along n"),
}


# ------------------------------------------------------------------ kind
def kind_of(data: Dict[str, Any]) -> Optional[str]:
    """ELASTIC, EOS, QHA or CPHF from what analyze() found (None: nothing)."""
    if any(k in data for k in ("elastic", "piezo_e", "piezo_d", "pockels", "piezo_optic")):
        return "ELASTIC"
    if "qha" in data:
        return "QHA"
    if "born" in data:
        return "FREQ"
    if "eos" in data:
        return "EOS"
    if any(k in data for k in ("dielectric", "beta", "gamma")):
        return "CPHF"
    return None


# ------------------------------------------------------------------ directional moduli
def _a3(n) -> List[float]:
    x, y, z = n
    return [x * x, y * y, z * z, y * z, x * z, x * y]


def _stiffness(data: Dict[str, Any]):
    """(C, dim, unit): the stiffness the directional plots use. 3D: the 6x6 in
    GPa. 2D: the in-plane block (xx, yy, xy) in the moduli's unit (N/m when
    the run gives hartree and the area is known)."""
    import numpy as np
    el = data.get("elastic")
    if not el:
        return None
    dim = data.get("dimensionality") or 3
    C = np.array(el["matrix"], dtype=float)
    if dim == 3 and C.shape == (6, 6):
        return (C + C.T) / 2.0, 3, el.get("unit") or "GPa"
    if dim == 2:
        mod = data.get("moduli") or {}
        if mod.get("c2d"):
            return np.array(mod["c2d"], dtype=float), 2, mod.get("unit") or el.get("unit") or ""
        if C.shape == (6, 6):
            C = C[np.ix_([0, 1, 5], [0, 1, 5])]
        if C.shape == (3, 3):
            return (C + C.T) / 2.0, 2, el.get("unit") or ""
    return None


def _units(quantity: str, dim: int, unit: str) -> Tuple[str, float]:
    """(the unit drawn, as plot text, the factor from 1/S or S h to it)."""
    if quantity in ("youngs", "shear"):
        return html_unit(unit), 1.0
    if quantity == "poisson":
        return "", 1.0
    if dim == 3:
        return "TPa<sup>−1</sup>", 1000.0                 # S in GPa^-1
    return ("m/N" if unit == "N/m" else f"{html_unit(unit)}<sup>−1</sup>" if unit else ""), 1.0


def html_unit(u: str) -> str:
    """A unit as plot text: TPa^-1 -> TPa<sup>−1</sup>, C/m^2 -> C/m<sup>2</sup>, A^3 -> Å<sup>3</sup>."""
    u = str(u or "").replace("⁻¹", "^-1").replace("²", "^2").replace("³", "^3")
    u = re.sub(r"\bA\^3", "Å^3", u)
    return re.sub(r"\^(-?\d+)", lambda m: "<sup>" + m.group(1).replace("-", "−") + "</sup>", u)


def plain(text: str) -> str:
    """Plot text as plain text (CSV and TSV): E<sub>min</sub> -> E_min, Å<sup>3</sup> -> Å^3, − -> -."""
    t = re.sub(r"<sub>(.*?)</sub>", r"_\1", str(text))
    t = re.sub(r"<sup>(.*?)</sup>", r"^\1", t)
    return re.sub(r"<[^>]+>", "", t).replace("−", "-")


def _a_rows(S, dirs):
    """(a per direction, h) as arrays: the Voigt weights of unit directions (N x 3)."""
    import numpy as np
    d = np.asarray(dirs, dtype=float).reshape(-1, 3)
    x, y, z = d[:, 0], d[:, 1], d[:, 2]
    if len(S) == 6:
        return np.stack([x * x, y * y, z * z, y * z, x * z, x * y], axis=1), np.array([1.0, 1, 1, 0, 0, 0])
    return np.stack([x * x, y * y, x * y], axis=1), np.array([1.0, 1, 0])


def directional_values(S, quantity: str, dirs):
    """E(n) (1/a^T S a) or beta(n) (a^T S h) of the compliance S (6x6, or 3x3 for a slab's
    in-plane block) along each unit direction of ``dirs`` (N x 3; a slab's lie in xy)."""
    import numpy as np
    S = np.asarray(S, dtype=float)
    a, h = _a_rows(S, dirs)
    if quantity == "youngs":
        with np.errstate(divide="ignore"):
            return 1.0 / np.einsum("ni,ij,nj->n", a, S, a)
    return a @ (S @ h)


def directional(S, quantity: str, n) -> float:
    """E(n) or beta(n) along one unit direction n."""
    return float(directional_values(S, quantity, [list(n) + [0.0] * (3 - len(n))])[0])


direction_label = T.direction_label


def _round(x, digits=5):
    return [_round(v, digits) for v in x] if isinstance(x, (list, tuple)) else (
        None if x is None or not math.isfinite(x) else float(f"{x:.{digits}g}"))


def _extrema(S, quantity: str, dim: int, scale: float, lattice: bool = False) -> Dict[str, Any]:
    """The smallest and largest value over directions and where (tensor.extremes)."""
    import numpy as np
    S = np.asarray(S, dtype=float)
    out = T.extremes(lambda d: directional_values(S, quantity, d) * scale, dim, lattice=lattice)
    out.pop("ratio", None)
    return out


def _sphere(n_theta: int, n_phi: int, full: bool = True):
    import numpy as np
    t, p = np.meshgrid(np.linspace(0.0, math.pi, n_theta), np.linspace(0.0, 2 * math.pi, n_phi), indexing="ij")
    return t, np.stack([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)], axis=-1)


def _surf(v, n) -> Dict[str, Any]:
    """A closed surface at distance |v| along each direction n (v signed, kept as the color)."""
    import numpy as np
    r = np.abs(v)
    out = {c: _round((r * n[..., k]).tolist()) for k, c in enumerate("xyz")}
    out["value"] = _round(v.tolist())
    return out


SYMBOLS = {"youngs": "E", "compressibility": "β", "shear": "G", "poisson": "ν", "sound": "v"}
NAMES = {"youngs": "Young's modulus", "compressibility": "Linear compressibility", "shear": "Shear modulus",
         "poisson": "Poisson's ratio", "sound": "Sound velocity"}
DIRECTIONAL = ("youngs", "compressibility", "shear", "poisson", "sound", "piezo")


def _piezo_tensor(data: Dict[str, Any]):
    import numpy as np
    for key, kind in (("piezo_d", "d"), ("piezo_e", "e")):
        t = data.get(key)
        if t and np.shape(t["matrix"]) == (3, 6) and any(abs(x) > 1e-12 for row in t["matrix"] for x in row):
            return np.array(t["matrix"], dtype=float), kind
    return None


def directional_figure(data: Dict[str, Any], quantity: str, shape: str = "surface",
                       n_theta: int = 46, n_phi: int = 91, n_polar: int = 361) -> Dict[str, Any]:
    """The surface or the principal-plane cuts of a directional property: E(n), beta(n), the min and max
    of G(n, m) and nu(n, m) over m perpendicular to n (ELATE's envelopes), the Christoffel phase and group
    velocities (cuts only), and the longitudinal piezoelectric coefficient."""
    import numpy as np
    if quantity == "piezo":
        return piezo_figure(data, shape, n_theta, n_phi, n_polar)
    got = T.compliance_of(data)
    if got is None:
        raise ValueError("no elastic constants in this output")
    St, Ct, dim, unit, C, S = got
    shown, scale = _units(quantity, dim, unit)
    sym = SYMBOLS[quantity]
    if quantity == "sound":
        shown, shape = "km/s", "planes"
    label = f"{NAMES[quantity]} {sym}" + (f" ({shown})" if shown else "")
    der = (data.get("derived") or {}).get("directional") or T.derived(data).get("directional") or {}
    ex = der.get(quantity) or {}
    out: Dict[str, Any] = {"view": quantity, "dim": dim, "unit": shown, "label": label, "symbol": sym,
                           "extrema": ({k: ex[k] for k in ("min", "max", "isotropic") if k in ex} if quantity != "sound" else {}),
                           "title": PLOTS[quantity][1] + (" (in-plane)" if dim == 2 else "")}
    if quantity in ("youngs", "compressibility"):
        out["extrema"] = _extrema(np.linalg.inv(C), quantity, dim, scale, T.is_cubic(data))
    if quantity == "compressibility":
        out["negative"] = out["extrema"]["min"]["value"] is not None and out["extrema"]["min"]["value"] < 0
    if quantity == "poisson":
        out["negative"] = bool(ex.get("min") and ex["min"]["value"] is not None and ex["min"]["value"] < 0)
    if quantity == "sound":
        rho = data.get("density_g_cm3")
        if not rho:
            raise ValueError("no density in this output (the sound velocities need it)")
        out["note"] = ("Phase velocities of the quasi-longitudinal (qL) and the fast and slow quasi-shear (qS1, qS2) waves "
                       "from the Christoffel equation, solid; the group velocity |v<sub>g</sub>| of each, dotted")
    if dim == 3 and shape == "surface":
        t, n = _sphere(n_theta, n_phi)
        flat = n.reshape(-1, 3)
        if quantity in ("youngs", "compressibility"):
            v = (directional_values(S, quantity, flat) * scale).reshape(t.shape)
            out.update(shape="surface", surface=_surf(v, n))
            return out
        sp = T.shear_poisson_extremes(St, flat)
        q = "g" if quantity == "shear" else "nu"
        env = {k: _surf(sp[f"{q}_{k}"].reshape(t.shape), n) for k in ("min", "max")}
        out.update(shape="surface", envelopes=env)
        return out
    planes = ([("xy", (0, 1)), ("xz", (0, 2)), ("yz", (1, 2))] if dim == 3 else [("xy", (0, 1))])
    deg = np.linspace(0.0, 360.0, n_polar)
    cuts = []
    for name, (i, j) in planes:
        n = np.zeros((len(deg), 3))
        n[:, i], n[:, j] = np.cos(np.radians(deg)), np.sin(np.radians(deg))
        cut: Dict[str, Any] = {"plane": name, "theta_deg": _round(deg.tolist(), 6)}
        if quantity in ("youngs", "compressibility"):
            cut["r"] = _round((directional_values(S, quantity, n) * scale).tolist())
        elif quantity in ("shear", "poisson"):
            sp = T.shear_poisson_extremes(St, n)
            q = "g" if quantity == "shear" else "nu"
            if dim == 2:
                cut["r"] = _round(sp[f"{q}_min"].tolist())
            else:
                cut["curves"] = [{"name": f"{sym}<sub>min</sub>", "key": "min", "r": _round(sp[f"{q}_min"].tolist())},
                                 {"name": f"{sym}<sub>max</sub>", "key": "max", "r": _round(sp[f"{q}_max"].tolist())}]
        else:
            ch = T.christoffel(Ct, n, data["density_g_cm3"])
            cut["curves"] = ([{"name": m, "key": m, "r": _round(ch["phase"][:, k].tolist())} for k, m in enumerate(("qL", "qS1", "qS2"))]
                             + [{"name": f"{m} group", "key": f"{m}_group", "dash": "dot", "r": _round(ch["group"][:, k].tolist())}
                                for k, m in enumerate(("qL", "qS1", "qS2"))])
        cuts.append(cut)
    out.update(shape="planes", planes=cuts)
    return out


def piezo_figure(data: Dict[str, Any], shape: str = "surface", n_theta: int = 46, n_phi: int = 91,
                 n_polar: int = 361) -> Dict[str, Any]:
    """The longitudinal piezoelectric coefficient d_L(n) (or e_L(n)) as a surface (|d_L|, colored by its
    sign) or principal-plane cuts."""
    import numpy as np
    got = _piezo_tensor(data)
    if got is None:
        raise ValueError("no non-zero piezoelectric tensor in this output")
    M, kind = got
    unit = "pC/N" if kind == "d" else "C/m<sup>2</sup>"
    sym = f"{kind}<sub>L</sub>"
    pz = (data.get("derived") or {}).get("piezo") or T.piezo(data)
    out: Dict[str, Any] = {"view": "piezo", "dim": data.get("dimensionality") or 3, "unit": unit, "symbol": sym,
                           "label": f"Longitudinal piezoelectric coefficient {sym} ({unit})", "kind": kind,
                           "extrema": {k: pz["longitudinal"][k] for k in ("min", "max", "isotropic")},
                           "title": PLOTS["piezo"][1], "negative": True,
                           "note": (f"{sym}(n) = {kind}<sub>ijk</sub> n<sub>i</sub> n<sub>j</sub> n<sub>k</sub>: the strain (d) or "
                                    "polarization (e) along n per unit field (stress) along n")}
    if shape == "surface":
        t, n = _sphere(n_theta, n_phi)
        v = T.piezo_longitudinal(M, kind, n.reshape(-1, 3)).reshape(t.shape)
        out.update(shape="surface", surface=_surf(v, n))
        return out
    deg = np.linspace(0.0, 360.0, n_polar)
    c, s_ = np.cos(np.radians(deg))[:, None], np.sin(np.radians(deg))[:, None]
    e = np.eye(3)
    planes = [("xy", e[0], e[1]), ("xz", e[0], e[2]), ("yz", e[1], e[2])]
    best = max(abs(T.piezo_longitudinal(M, kind, c * a + s_ * b)).max() for _, a, b in planes)
    mx = np.array(out["extrema"]["max"]["direction"], dtype=float)
    lat = T.is_cubic(data)
    if best < 0.9 * abs(out["extrema"]["max"]["value"] or 0.0):
        # the largest coefficient is off the principal planes (a cubic -43m crystal's <111>): the plane
        # through z (x when it lies along z) and that direction, named by its normal
        a = e[2] if abs(mx[2]) < 0.99 else e[0]
        b = mx - (mx @ a) * a
        b /= np.linalg.norm(b)
        normal = np.cross(a, b)
        lab = T.direction_label(normal, lattice=lat)
        # a cubic crystal's plane by its Miller indices, else by its Cartesian normal
        lab = lab.replace("[", "(").replace("]", ")") if lat else f"⊥ {lab}"
        planes.append((lab, b, a))
        # a principal plane where the coefficient vanishes says nothing: only the cuts that show it
        planes = [p for p in planes if abs(T.piezo_longitudinal(M, kind, c * p[1] + s_ * p[2])).max() > 1e-9 * best + 1e-12
                  or p is planes[-1]]
        out["note"] += f"; the cut through z and {out['extrema']['max']['label']}, where it is largest"
    cuts = []
    for name, a, b in planes:
        n = c * a + s_ * b
        cuts.append({"plane": name, "theta_deg": _round(deg.tolist(), 6), "r": _round(T.piezo_longitudinal(M, kind, n).tolist()),
                     "axes": [T.direction_label(a, lattice=lat), T.direction_label(b, lattice=lat)]})
    out.update(shape="planes", planes=cuts)
    return out


# ------------------------------------------------------------------ equations of state
def eos_energy(name: str, V: float, f: Dict[str, float]) -> float:
    """E(V) in hartree of one CRYSTAL fit (V in A^3)."""
    V0, E0, B0, Bp = f["volume_ang3"], f["energy_au"], f["bulk_modulus_gpa"], f["bulk_modulus_prime"]
    k = B0 * GPA_A3_HARTREE                         # GPa -> hartree / A^3
    if name == "murnaghan":
        return E0 + k * V / Bp * ((V0 / V) ** Bp / (Bp - 1.0) + 1.0) - k * V0 / (Bp - 1.0)
    if name == "birch_murnaghan":
        x = (V0 / V) ** (2.0 / 3.0)
        return E0 + 9.0 * V0 * k / 16.0 * ((x - 1.0) ** 3 * Bp + (x - 1.0) ** 2 * (6.0 - 4.0 * x))
    if name == "poirier_tarantola":
        x = math.log(V0 / V)
        return E0 + k * V0 * x * x / 2.0 * (1.0 + (Bp - 2.0) * x / 3.0)
    if name == "vinet":
        eta = (V / V0) ** (1.0 / 3.0)
        return E0 + 2.0 * k * V0 / (Bp - 1.0) ** 2 * (
            2.0 - (5.0 + 3.0 * Bp * (eta - 1.0) - 3.0 * eta) * math.exp(-1.5 * (Bp - 1.0) * (eta - 1.0)))
    raise ValueError(f"unknown equation of state {name}")


def eos_pressure(name: str, V: float, f: Dict[str, float]) -> float:
    """P(V) = -dE/dV in GPa of one CRYSTAL fit."""
    V0, B0, Bp = f["volume_ang3"], f["bulk_modulus_gpa"], f["bulk_modulus_prime"]
    if name == "murnaghan":
        return B0 / Bp * ((V0 / V) ** Bp - 1.0)
    if name == "birch_murnaghan":
        x = (V0 / V) ** (1.0 / 3.0)
        return 1.5 * B0 * (x ** 7 - x ** 5) * (1.0 + 0.75 * (Bp - 4.0) * (x * x - 1.0))
    if name == "poirier_tarantola":
        x = math.log(V0 / V)
        return B0 * (V0 / V) * x * (1.0 + (Bp - 2.0) * x / 2.0)
    if name == "vinet":
        eta = (V / V0) ** (1.0 / 3.0)
        return 3.0 * B0 * (1.0 - eta) / eta ** 2 * math.exp(1.5 * (Bp - 1.0) * (1.0 - eta))
    raise ValueError(f"unknown equation of state {name}")


def _eos_grid(eos: Dict[str, Any], n: int = 161) -> List[float]:
    vs = [p[0] for p in eos.get("points") or []] or [f["volume_ang3"] for f in (eos.get("fits") or {}).values()]
    lo, hi = min(vs), max(vs)
    pad = 0.02 * (hi - lo or lo)
    lo, hi = lo - pad, hi + pad
    return [lo + (hi - lo) * k / (n - 1) for k in range(n)]


def eos_bulk(name: str, V: float, f: Dict[str, float]) -> float:
    """B(V) = -V dP/dV in GPa of one fit (central difference of eos_pressure; B(V0) = B0)."""
    h = 1e-5 * V
    return -V * (eos_pressure(name, V + h, f) - eos_pressure(name, V - h, f)) / (2 * h)


def eos_quality(eos: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Per fit: the residuals E_computed - E_fit (meV/cell) of every computed point, their RMS and
    largest magnitude, and the RMS relative to the spread of the computed energies. The fit is
    evaluated from V0, E0, B0, B' as CRYSTAL prints them (4, 8, 2 and 2 decimals): their rounding
    adds up to ~0.01 meV."""
    pts = eos.get("points") or []
    out: Dict[str, Dict[str, Any]] = {}
    if len(pts) < 2:
        return out
    span = (max(p[1] for p in pts) - min(p[1] for p in pts)) * HARTREE_MEV or 1.0
    for k, f in (eos.get("fits") or {}).items():
        res = [(p[1] - eos_energy(k, p[0], f)) * HARTREE_MEV for p in pts]
        rms = math.sqrt(sum(r * r for r in res) / len(res))
        rel = rms / span
        out[k] = {"residuals_mev": _round(res, 6), "rms_mev": _round(rms, 6), "max_abs_mev": _round(max(abs(r) for r in res), 6),
                  "rms_relative": _round(rel, 4), "quality": "good" if rel < 1e-3 else "fair" if rel < 1e-2 else "poor"}
    return out


def _fit_key(fits: Dict[str, Any], fit: Optional[str]) -> str:
    if fit:
        if fit not in fits:
            raise ValueError(f"no {fit} fit in this output (it has {', '.join(fits)})")
        return fit
    return "birch_murnaghan" if "birch_murnaghan" in fits else next(iter(fits))


def enthalpy_curve(data: Dict[str, Any], fit: str, n: int = 241) -> Dict[str, List[float]]:
    """H(P) = E + PV per formula unit along one fit, over the computed volumes (2 % margin):
    {p_gpa, h_ha_fu, v_ang3_fu}, by increasing pressure."""
    eos = data.get("eos") or {}
    f = eos["fits"][fit]
    z = data.get("formula_units") or 1
    rows = []
    for v in _eos_grid(eos, n):
        p = eos_pressure(fit, v, f)
        rows.append((p, (eos_energy(fit, v, f) + p * v * GPA_A3_HARTREE) / z, v / z))
    rows.sort()
    return {"p_gpa": [r[0] for r in rows], "h_ha_fu": [r[1] for r in rows], "v_ang3_fu": [r[2] for r in rows]}


def _interp(xs: Sequence[float], ys: Sequence[float], x: float) -> float:
    import bisect
    i = min(max(bisect.bisect_left(xs, x), 1), len(xs) - 1)
    t = (x - xs[i - 1]) / ((xs[i] - xs[i - 1]) or 1.0)
    return ys[i - 1] + t * (ys[i] - ys[i - 1])


#: |H(other) - H(ref)| below this (meV per formula unit) is no difference: the EOS fits' energies carry
#: 1e-8 hartree (0.3 meV) at best, and two identical runs give exactly 0 at every pressure
DH_TOL_MEV = 1e-2


def transitions(ref: Dict[str, Any], other: Dict[str, Any], fit: Optional[str] = None, n: int = 401) -> Dict[str, Any]:
    """The enthalpy difference H(other) - H(ref) per formula unit along P over the pressures both
    fits cover, and the pressures where it changes sign (linear interpolation between grid points):
    where it goes from one side of +-DH_TOL_MEV to the other, so that two runs that agree everywhere
    (``same``: |dH| within the tolerance over the whole range) have no transition.
    Raises ValueError when the two runs differ in composition."""
    fa, fb = ref.get("formula"), other.get("formula")
    if fa and fb and fa != fb:
        raise ValueError(f"the phases differ in composition ({fa} and {fb}): enthalpies compare per formula unit of one composition")
    fits_a, fits_b = (ref.get("eos") or {}).get("fits") or {}, (other.get("eos") or {}).get("fits") or {}
    if not fits_a or not fits_b:
        raise ValueError("both runs need the equation-of-state fits (an EOS run)")
    key = _fit_key(fits_a, fit)
    if key not in fits_b:
        raise ValueError(f"the other run has no {key} fit")
    a, b = enthalpy_curve(ref, key), enthalpy_curve(other, key)
    lo, hi = max(a["p_gpa"][0], b["p_gpa"][0]), min(a["p_gpa"][-1], b["p_gpa"][-1])
    out: Dict[str, Any] = {"fit": key, "p_range_gpa": [_round(lo, 6), _round(hi, 6)], "p_gpa": [], "dh_mev_fu": [], "crossings": []}
    if hi <= lo:
        out["overlap"] = False
        return out
    out["overlap"] = True
    ps = [lo + (hi - lo) * k / (n - 1) for k in range(n)]
    dh = [(_interp(b["p_gpa"], b["h_ha_fu"], p) - _interp(a["p_gpa"], a["h_ha_fu"], p)) * HARTREE_MEV for p in ps]
    out["same"] = all(abs(x) <= DH_TOL_MEV for x in dh)
    # the points beyond the tolerance, and a crossing between two of them of opposite sign (any points
    # within the tolerance between them are where the curves touch)
    side = [(k, x) for k, x in enumerate(dh) if abs(x) > DH_TOL_MEV]
    for (i, a), (j, b) in zip(side, side[1:]):
        if a * b < 0:
            k = next((m for m in range(i + 1, j + 1) if dh[m - 1] * dh[m] <= 0), j)
            x0, x1 = dh[k - 1], dh[k]
            p = ps[k - 1] + (ps[k] - ps[k - 1]) * x0 / ((x0 - x1) or 1.0)
            # above the crossing the phase with the lower enthalpy is stable
            out["crossings"].append({"pressure_gpa": _round(p, 6), "stable_above": "other" if b < a else "reference"})
    out["p_gpa"], out["dh_mev_fu"] = _round(ps, 7), _round(dh, 7)
    return out


def eos_figure(data: Dict[str, Any], view: str, compare: Optional[Sequence[Tuple[str, Dict[str, Any]]]] = None,
               fit: Optional[str] = None, name: str = "") -> Dict[str, Any]:
    """One EOS plot. ``compare``: (label, analyze() of another EOS run) for the enthalpy difference."""
    eos = data.get("eos") or {}
    fits = eos.get("fits") or {}
    pts = eos.get("points") or []
    if not fits and not pts:
        raise ValueError("no equation-of-state results in this output")
    grid = _eos_grid(eos)
    series = []
    if view == "eos":
        ref = min(p[1] for p in pts) if pts else min(f["energy_au"] for f in fits.values())
        for k, f in fits.items():
            series.append({"name": EOS_LABELS.get(k, k), "key": k, "role": "fit", "x": _round(grid, 8),
                           "y": _round([(eos_energy(k, v, f) - ref) * HARTREE_MEV for v in grid], 7)})
        if pts:
            series.append({"name": "computed", "key": "points", "role": "points", "x": [p[0] for p in pts],
                           "y": _round([(p[1] - ref) * HARTREE_MEV for p in pts], 8),
                           "e_au": [p[1] for p in pts]})
        return {"view": "eos", "title": "E(V)", "x_label": "V (Å<sup>3</sup>)", "y_label": "E − E<sub>min</sub> (meV/cell)",
                "series": series, "reference_au": ref, "note": f"Energies relative to the lowest computed point, {ref:.8f} Ha"}
    if view == "pv":
        for k, f in fits.items():
            series.append({"name": EOS_LABELS.get(k, k), "key": k, "role": "fit", "x": _round(grid, 8),
                           "y": _round([eos_pressure(k, v, f) for v in grid], 7)})
        return {"view": "pv", "title": "P(V)", "x_label": "V (Å<sup>3</sup>)", "y_label": "P (GPa)", "series": series,
                "note": "Pressure of each fit, P = −dE/dV"}
    if view in ("bp", "vp"):
        for k, f in fits.items():
            P = [eos_pressure(k, v, f) for v in grid]
            y = [eos_bulk(k, v, f) for v in grid] if view == "bp" else [v / f["volume_ang3"] for v in grid]
            series.append({"name": EOS_LABELS.get(k, k), "key": k, "role": "fit", "x": _round(P, 7), "y": _round(y, 7)})
        if view == "bp":
            return {"view": "bp", "title": "B(P)", "x_label": "P (GPa)", "y_label": "B (GPa)", "series": series,
                    "note": "Bulk modulus of each fit along its pressure, B = −V dP/dV (B = B<sub>0</sub> at P = 0, slope B′ there)"}
        return {"view": "vp", "title": "V/V<sub>0</sub>(P)", "x_label": "P (GPa)", "y_label": "V/V<sub>0</sub>", "series": series,
                "note": "Volume of each fit along its pressure, relative to that fit's V<sub>0</sub>"}
    if view == "residuals":
        q = eos_quality(eos)
        if not q:
            raise ValueError("no computed points to compare the fits with")
        for k, r in q.items():
            series.append({"name": f"{EOS_LABELS.get(k, k)} (RMS {r['rms_mev']:.3g} meV)", "key": k, "role": "curve",
                           "marker": True, "x": [p[0] for p in pts], "y": r["residuals_mev"]})
        return {"view": "residuals", "title": "Fit residuals", "x_label": "V (Å<sup>3</sup>)",
                "y_label": "E<sub>computed</sub> − E<sub>fit</sub> (meV/cell)", "series": series, "zero_line": True,
                "quality": q, "note": "Residual of each computed point from each fit (the fits as CRYSTAL prints V<sub>0</sub>, E<sub>0</sub>, B<sub>0</sub>, B′)"}
    if view != "enthalpy":
        raise ValueError(f"unknown EOS plot {view}")
    key = _fit_key(fits, fit)
    z = data.get("formula_units") or 1
    fu = f" ({data['formula']}, Z = {z})" if data.get("formula") else ""
    if not compare:
        e0 = fits[key]["energy_au"] / z
        for k in fits:
            c = enthalpy_curve(data, k)
            series.append({"name": EOS_LABELS.get(k, k), "key": k, "role": "fit", "x": _round(c["p_gpa"], 7),
                           "y": _round([(h - e0) * HARTREE_MEV / 1000.0 for h in c["h_ha_fu"]], 8)})
        return {"view": "enthalpy", "title": "H(P)", "x_label": "P (GPa)", "y_label": "H − E<sub>0</sub> (eV/f.u.)",
                "series": series, "fit": key, "formula": data.get("formula"), "formula_units": z,
                "note": f"Enthalpy H = E + PV per formula unit{fu}, relative to E<sub>0</sub> of the {EOS_LABELS.get(key, key)} fit; "
                        "Add phase… compares another structure's"}
    name0 = _sub(name) or "this run"
    series.append({"name": f"{name0} (reference)", "key": "reference", "role": "curve", "dash": "dot", "x": [], "y": []})
    found = []
    for label, other in compare:
        t = transitions(data, other, key)
        series.append({"name": _sub(label), "key": label, "role": "curve", "x": t["p_gpa"], "y": t["dh_mev_fu"]})
        found.append({"phase": label, "formula": other.get("formula"), **{k: t[k] for k in ("fit", "p_range_gpa", "overlap", "crossings")},
                      "same": bool(t.get("same"))})
    xs = [x for s_ in series for x in s_["x"]]
    if xs:
        series[0]["x"], series[0]["y"] = [min(xs), max(xs)], [0.0, 0.0]

    def said(f):
        if f["crossings"]:
            return ", ".join(f"P<sub>t</sub> = {c['pressure_gpa']:.2f} GPa" for c in f["crossings"])
        if not f["overlap"]:
            return "no common pressure range"
        if f["same"]:
            return f"the same enthalpy (within {DH_TOL_MEV:g} meV/f.u.) at every pressure: no transition"
        return f"no crossing between {f['p_range_gpa'][0]:.1f} and {f['p_range_gpa'][1]:.1f} GPa"
    note = "; ".join(f"{f['phase']}: {said(f)}" for f in found)
    return {"view": "enthalpy", "title": "ΔH(P)", "x_label": "P (GPa)", "y_label": "ΔH = H − H<sub>ref</sub> (meV/f.u.)",
            "series": series, "fit": key, "formula": data.get("formula"), "formula_units": z, "transitions": found,
            "markers": [c["pressure_gpa"] for f in found for c in f["crossings"]],
            "note": f"Enthalpy of each phase minus this run's, per formula unit{fu}, {EOS_LABELS.get(key, key)} fits: {note}"}


def _sub(name: str) -> str:
    return re.sub(r"\.(out|log)$", "", name)


# ------------------------------------------------------------------ QHA
_HARM_HEAD = re.compile(r"^\s*HARMONIC VIBRATIONAL CONTRIBUTIONS TO THERMODYNAMIC FUNCTIONS", re.M)
_AT_T = re.compile(r"AT \(T =\s*(" + A._FLOAT + r")\s*K")
_CV_ROW = re.compile(r"^\s*HEAT CAPACITY\s*:\s*(" + A._FLOAT + r")\s+(" + A._FLOAT + r")\s+(" + A._FLOAT + r")\s*$", re.M)
_VOL_LINE = re.compile(r"PRIMITIVE CELL - CENTRING CODE\s+\S+\s+VOLUME=\s*(" + A._FLOAT + r")")


def harmonic_heat_capacity(text: str) -> List[Dict[str, Any]]:
    """The harmonic C_v(T) (J/(mol K)) CRYSTAL prints for each volume of a QHA
    run, with that volume (the last cell printed before its frequencies);
    [] for an output without them."""
    heads = list(_HARM_HEAD.finditer(text))
    out = []
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        stop = text.find("QUASI-HARMONIC APPROXIMATION", m.end(), end)
        block = text[m.end():stop if stop > 0 else end]
        vol = None
        for vm in _VOL_LINE.finditer(text, 0, m.start()):
            vol = A._f(vm.group(1))
        ts, cv = [], []
        for part in block.split("THERMODYNAMIC FUNCTIONS WITH VIBRATIONAL CONTRIBUTIONS")[1:]:
            t, c = _AT_T.search(part), _CV_ROW.search(part)
            if t and c:
                ts.append(A._f(t.group(1)))
                cv.append(A._f(c.group(3)))
        if ts:
            out.append({"volume_ang3": vol, "temperature_k": ts, "cv_j_mol_k": cv})
    out.sort(key=lambda b: b["volume_ang3"] or 0.0)
    return out


def qha_figure(data: Dict[str, Any], view: str, text: Optional[str] = None) -> Dict[str, Any]:
    q = data.get("qha")
    if not q:
        raise ValueError("no quasi-harmonic results in this output")
    T = q["temperature_k"]
    th = q.get("thermo") or {}
    if view == "volume":
        return {"view": view, "title": "V(T)", "x_label": "T (K)", "y_label": "V (Å<sup>3</sup>)",
                "series": [{"name": "V (P = 0)", "role": "curve", "x": T, "y": q["volume_ang3"]}],
                "note": "Equilibrium volume of the primitive cell at zero pressure (zero-point energy included)"}
    if view == "vratio":
        v0 = q["volume_ang3"][0]
        series = [{"name": f"V/V(T<sub>0</sub>)", "role": "curve", "x": T, "y": _round([v / v0 for v in q["volume_ang3"]], 9)}]
        lat = q.get("lattice")
        if lat:
            for g in _axis_groups(lat):
                x0 = lat[f"{g[0]}_ang"][0]
                series.append({"name": " = ".join(f"{ax}/{ax}(T<sub>0</sub>)" for ax in g), "role": "curve", "dash": "dash",
                               "x": lat["temperature_k"], "y": _round([x / x0 for x in lat[f"{g[0]}_ang"]], 9)})
        return {"view": view, "title": "V(T)/V(T<sub>0</sub>)", "x_label": "T (K)", "y_label": "relative to T<sub>0</sub>",
                "series": series, "note": f"Volume and lattice parameters relative to their values at T<sub>0</sub> = {T[0]:g} K "
                                          "(zero pressure, zero-point energy included)"}
    if view == "expansion":
        series = [{"name": "α<sub>V</sub>", "role": "curve", "x": T, "y": q["alpha_v_per_k"]}]
        lat = q.get("lattice")
        if lat:
            for g in _axis_groups(lat):
                series.append({"name": " = ".join(f"α<sub>{ax}</sub>" for ax in g), "role": "curve", "x": lat["temperature_k"],
                               "y": lat[f"alpha_{g[0]}_per_k"], "dash": "dash"})
        return {"view": view, "title": "Thermal expansion", "x_label": "T (K)", "y_label": "α (K<sup>−1</sup>)", "series": series,
                "exponent": True, "note": "Volume thermal expansion α<sub>V</sub> and the linear α of the conventional cell's a, b, c "
                                          "(zero pressure; one curve per distinct axis)"}
    if view == "bulk":
        return {"view": view, "title": "B(T)", "x_label": "T (K)", "y_label": "B (GPa)",
                "note": "Isothermal and adiabatic bulk moduli at zero pressure",
                "series": [{"name": "B<sub>T</sub> (isothermal)", "role": "curve", "x": T, "y": q["bulk_modulus_t_gpa"]},
                           {"name": "B<sub>S</sub> (adiabatic)", "role": "curve", "x": T, "y": q["bulk_modulus_s_gpa"], "dash": "dash"}]}
    if view == "gamma":
        series = []
        g = q.get("gruneisen")
        if g:
            series.append({"name": "γ (CRYSTAL, mode Grüneisen)", "role": "curve", "x": g["temperature_k"], "y": g["gamma"]})
        if th:
            pts = [(t, x) for t, x in zip(th["temperature_k"], th["gamma_th"]) if x is not None]
            if pts:
                series.append({"name": "γ<sub>th</sub> = α<sub>V</sub> B<sub>T</sub> V/C<sub>V</sub>", "role": "curve", "marker": True,
                               "dash": "dash", "x": [p[0] for p in pts], "y": _round([p[1] for p in pts], 6)})
        if not series:
            raise ValueError("no Grüneisen parameter in this output")
        return {"view": view, "title": "Grüneisen γ(T)", "x_label": "T (K)", "y_label": "γ", "series": series,
                "note": "CRYSTAL's mode-averaged Grüneisen parameter, and the thermodynamic γ<sub>th</sub> from the zero-pressure "
                        "α<sub>V</sub>, B<sub>T</sub>, V and the harmonic C<sub>V</sub> at V(T) (where C<sub>V</sub> is at least a fifth of its "
                        "largest value)"}
    if view in ("free", "entropy"):
        if not th:
            raise ValueError("no per-volume harmonic thermodynamics in this output (F and S need them)")
        if view == "free":
            f0 = th["f_kj_mol"][0]
            return {"view": view, "title": "F(T)", "x_label": "T (K)", "y_label": "F − F(T<sub>0</sub>) (kJ/mol)",
                    "series": [{"name": "F (P = 0)", "role": "curve", "marker": len(th["temperature_k"]) < 30,
                                "x": th["temperature_k"], "y": _round([f - f0 for f in th["f_kj_mol"]], 8)}],
                    "note": f"Helmholtz free energy EL + E0 + ET − TS along V(T), per mole of cells, relative to "
                            f"T<sub>0</sub> = {th['temperature_k'][0]:g} K ({th['f_au'][0]:.8f} Ha/cell); "
                            f"order-{th['polynomial_order']} polynomial through the {len(th['volumes_ang3'])} computed volumes"}
        return {"view": view, "title": "S(T)", "x_label": "T (K)", "y_label": "S (J/(mol·K))",
                "series": [{"name": "S (P = 0)", "role": "curve", "marker": len(th["temperature_k"]) < 30,
                            "x": th["temperature_k"], "y": _round(th["entropy_j_mol_k"], 8)}],
                "note": "Vibrational entropy along V(T), per mole of cells (each volume's harmonic S, interpolated to V(T))"}
    if view == "fv":
        from . import qha as Q
        fam = Q.free_energy_family(text or "")
        if not fam:
            raise ValueError("this output prints neither the PVT relation nor each volume's harmonic thermodynamics")
        ref = min(c["f_min_au"] for c in fam["curves"])
        series = [{"name": f"{c['temperature_k']:g} K", "key": f"T{c['temperature_k']:g}", "role": "curve",
                   "x": _round(c["volume_ang3"], 8), "y": _round([(f - ref) * HARTREE_MEV for f in c["f_au"]], 8)}
                  for c in fam["curves"]]
        series.append({"name": "minimum F(V) (V(T))", "key": "minima", "role": "points", "line": True,
                       "x": _round([c["v_min"] for c in fam["curves"]], 8), "y": _round([(c["f_min_au"] - ref) * HARTREE_MEV for c in fam["curves"]], 8)})
        if fam["points"]:
            series.append({"name": "computed volumes", "key": "computed", "role": "points", "hollow": True,
                           "x": [v for p in fam["points"] for v in p["volume_ang3"]],
                           "y": _round([(f - ref) * HARTREE_MEV for p in fam["points"] for f in p["f_au"]], 8)})
        src = ("CRYSTAL's PRESSURE-VOLUME-TEMPERATURE table" if fam["source"] == "pvt"
               else "a polynomial through the computed volumes' F")
        return {"view": view, "title": "F(V)", "x_label": "V (Å<sup>3</sup>)", "y_label": "F − F<sub>min</sub> (meV/cell)",
                "series": series, "minima": [{"temperature_k": c["temperature_k"], "volume_ang3": c["v_min"]} for c in fam["curves"]],
                "note": f"Helmholtz free energy against volume at {len(fam['curves'])} temperatures ({src}); the dots trace each "
                        "curve's minimum, the zero-pressure V(T)"}
    series = []
    if th:
        series += [{"name": "C<sub>p</sub>", "role": "curve", "x": th["temperature_k"], "y": _round(th["cp_j_mol_k"], 8)},
                   {"name": "C<sub>v</sub> at V(T)", "role": "curve", "dash": "dash", "x": th["temperature_k"], "y": _round(th["cv_j_mol_k"], 8)}]
    series.append({"name": "C<sub>p</sub> − C<sub>v</sub>", "role": "curve", "x": T, "y": q["cp_minus_cv_j_mol_k"]})
    note = "C<sub>p</sub> − C<sub>v</sub> from the QHA (zero pressure); the output prints no C<sub>v</sub> of the expanded cell"
    for b in harmonic_heat_capacity(text or ""):
        series.append({"name": f"C<sub>v</sub> (harmonic, V = {b['volume_ang3']:.3f} Å<sup>3</sup>)", "role": "curve", "dash": "dot",
                       "x": b["temperature_k"], "y": b["cv_j_mol_k"]})
        note = "C<sub>p</sub> − C<sub>v</sub> from the QHA; C<sub>v</sub> of each volume from its harmonic frequencies"
    if th:
        note = ("C<sub>v</sub> interpolated to V(T) and C<sub>p</sub> = C<sub>v</sub> + (C<sub>p</sub> − C<sub>v</sub>), per mole of cells; "
                "each computed volume's harmonic C<sub>v</sub> dotted")
    return {"view": "heat", "title": "Heat capacity", "x_label": "T (K)", "y_label": "C (J/(mol·K))", "series": series,
            "note": note}


def _axis_groups(lat: Dict[str, List[float]]) -> List[List[str]]:
    """The conventional axes that expand alike (a cubic cell: one group a = b = c)."""
    groups: List[List[str]] = []
    for ax in "abc":
        same = next((g for g in groups if lat[f"alpha_{g[0]}_per_k"] == lat[f"alpha_{ax}_per_k"]), None)
        if same:
            same.append(ax)
        else:
            groups.append([ax])
    return groups


# ------------------------------------------------------------------ CPHF
def dielectric_spectrum(text: str) -> List[Dict[str, Any]]:
    """Every row of the polarizability / dielectric table, all frequencies: one
    table, or one table per frequency (it stops at a frequency and component
    seen before: a PHOTOELA run prints the same table again per strained cell)."""
    rows: List[Dict[str, Any]] = []
    seen = set()
    for m in A._DIELECTRIC_HEAD.finditer(text):
        got = False
        for line in text[m.end():].splitlines()[1:]:
            r = A._DIELECTRIC_ROW.match(line)
            if r:
                key = (r.group(1), r.group(2))
                if key in seen:
                    return rows
                seen.add(key)
                v = [A._f(x) for x in (r.group(1), *r.group(3, 4, 5, 6, 7))]
                rows.append({"frequency_ev": v[0], "component": r.group(2), "alpha": v[1], "alpha_im": v[2],
                             "epsilon": v[3], "epsilon_im": v[4], "chi1": v[5]})
                got = True
            elif got and line.strip():
                break
    return rows


def frequencies(rows: Sequence[Dict[str, Any]]) -> List[float]:
    return sorted({r["frequency_ev"] for r in rows})


def tensor3(rows: Sequence[Dict[str, Any]], key: str) -> List[List[float]]:
    """The symmetric 3x3 tensor of one frequency's six independent components."""
    m = [[0.0] * 3 for _ in range(3)]
    for r in rows:
        i, j = "XYZ".index(r["component"][0]), "XYZ".index(r["component"][1])
        m[i][j] = m[j][i] = r[key]
    return m


def cphf_figure(data: Dict[str, Any], text: str) -> Dict[str, Any]:
    rows = dielectric_spectrum(text)
    fr = frequencies(rows)
    if len(fr) < 2:
        raise ValueError("a static CPHF run: ε at one frequency (DYNAMIC gives ε(ω))")
    series = []
    for comp in ("XX", "YY", "ZZ", "XY", "XZ", "YZ"):
        pts = sorted((r["frequency_ev"], r["epsilon"], r["epsilon_im"]) for r in rows if r["component"] == comp)
        if not pts or all(abs(p[1]) < 1e-8 and abs(p[2]) < 1e-8 for p in pts):
            continue
        series.append({"name": f"Re ε<sub>{comp.lower()}</sub>", "role": "curve", "x": [p[0] for p in pts], "y": [p[1] for p in pts]})
        if any(abs(p[2]) > 1e-8 for p in pts):
            series.append({"name": f"Im ε<sub>{comp.lower()}</sub>", "role": "curve", "dash": "dash", "x": [p[0] for p in pts],
                           "y": [p[2] for p in pts]})
    return {"view": "epsilon", "title": "ε(ω)", "x_label": "ħω (eV)", "y_label": "ε", "series": series,
            "note": "CPHF/KS DYNAMIC: the dielectric tensor at each frequency (Im ε with DAMPING)"}


def optics_figure(data: Dict[str, Any], view: str, shape: str = "surface") -> Dict[str, Any]:
    """n(omega) and the absorption of a DYNAMIC run, the index ellipsoid, and the SHG surface or cuts
    (mace.elastic.optics)."""
    import numpy as np
    from . import optics as O
    opt = data.get("optics") or {}
    if view in ("refractive", "absorption"):
        dy = opt.get("dynamic")
        if not dy:
            raise ValueError("a static CPHF run: one frequency (DYNAMIC gives n(ω))")
        series = []
        for comp, c in dy["components"].items():
            if view == "refractive":
                series.append({"name": f"n<sub>{comp}</sub>", "role": "curve", "x": c["frequency_ev"], "y": c["n"]})
                if dy["absorbing"]:
                    series.append({"name": f"κ<sub>{comp}</sub>", "role": "curve", "dash": "dash", "x": c["frequency_ev"], "y": c["kappa"]})
            else:
                series.append({"name": f"α<sub>{comp}</sub>", "role": "curve", "x": c["frequency_ev"], "y": c["absorption_cm"]})
        if view == "refractive":
            return {"view": view, "title": "n(ω)", "x_label": "ħω (eV)", "y_label": "n, κ" if dy["absorbing"] else "n", "series": series,
                    "note": "Complex refractive index n + iκ = √ε of each diagonal component" + (
                        "" if dy["absorbing"] else " (ε is real: no DAMPING, so κ = 0 and n only)")}
        return {"view": view, "title": "Absorption", "x_label": "ħω (eV)", "y_label": "α (cm<sup>−1</sup>)", "series": series,
                "exponent": True, "note": "Absorption coefficient α = 2ωκ/c of each diagonal component (from Im ε, DAMPING)"}
    if view == "indicatrix":
        pr = opt.get("principal")
        if not pr:
            raise ValueError("no dielectric tensor in this output")
        lo, hi = int(np.argmin(pr["n"])), int(np.argmax(pr["n"]))
        ext = {"min": {"value": pr["n"][lo], "direction": pr["axes"][lo], "label": pr["axis_labels"][lo]},
               "max": {"value": pr["n"][hi], "direction": pr["axes"][hi], "label": pr["axis_labels"][hi]},
               "isotropic": pr["optical_class"] == "isotropic"}
        return {"view": view, "dim": 3, "shape": "surface", "surface": O.indicatrix(pr), "unit": "", "symbol": "n",
                "label": "Refractive index n", "title": PLOTS[view][1], "extrema": ext, "glyph": True,
                "note": (f"Index ellipsoid, semi-axes n = {', '.join(f'{x:.4f}' for x in pr['n'])} along "
                         f"{', '.join(pr['axis_labels'])}: {pr['optical_class']}" + (
                             f", Δn = {pr['birefringence']:.4g}" if pr["optical_class"] != "isotropic" else ""))}
    s = opt.get("shg") or {}
    if s.get("zero") is not False:
        raise ValueError("no non-zero second-order susceptibility in this output (a centrosymmetric crystal or no THIRD/FOURTH)")
    d = np.array(s["tensor"], dtype=float)
    out: Dict[str, Any] = {"view": view, "dim": 3, "unit": "pm/V", "symbol": "|d(n)|", "label": "SHG |d(n)| (pm/V)",
                           "title": PLOTS[view][1], "extrema": {k: s["magnitude"][k] for k in ("min", "max")},
                           "note": "|d<sub>ijk</sub> n<sub>j</sub> n<sub>k</sub>|: the second-harmonic polarization for a pump polarized along n "
                                   "(Kleinman symmetry; the run's frequency, no phase matching)"}
    out["extrema"]["isotropic"] = False
    if shape == "surface":
        t, n = _sphere(46, 91)
        v = O.shg_values(d, n.reshape(-1, 3), False).reshape(t.shape)
        out.update(shape="surface", surface=_surf(v, n))
        return out
    deg = np.linspace(0.0, 360.0, 361)
    cuts = []
    for name, (i, j) in (("xy", (0, 1)), ("xz", (0, 2)), ("yz", (1, 2))):
        n = np.zeros((len(deg), 3))
        n[:, i], n[:, j] = np.cos(np.radians(deg)), np.sin(np.radians(deg))
        cuts.append({"plane": name, "theta_deg": _round(deg.tolist(), 6),
                     "curves": [{"name": "|d(n)|", "key": "magnitude", "r": _round(O.shg_values(d, n, False).tolist())},
                                {"name": "d<sub>L</sub>(n)", "key": "longitudinal", "dash": "dash", "r": _round(O.shg_values(d, n, True).tolist())}]})
    out.update(shape="planes", planes=cuts)
    return out


# ------------------------------------------------------------------ views and figures
def views(data: Dict[str, Any], text: str = "") -> List[Dict[str, str]]:
    """The plots this run gives (Results, the tables, comes first in the GUI)."""
    kind = kind_of(data)
    out = []
    if kind == "ELASTIC":
        dim = data.get("dimensionality") or 3
        if T.compliance_of(data) is not None:
            out += ["youngs", "compressibility", "shear", "poisson"]
            if dim == 3 and data.get("density_g_cm3"):
                out.append("sound")
        if dim == 3 and _piezo_tensor(data) is not None:
            out.append("piezo")
    elif kind == "EOS":
        eos = data.get("eos") or {}
        if eos.get("points") or eos.get("fits"):
            out.append("eos")
        if eos.get("fits"):
            out += ["pv", "bp", "vp"]
            if len(eos.get("points") or []) > 1:
                out.append("residuals")
            out.append("enthalpy")
    elif kind == "QHA":
        q = data.get("qha") or {}
        out += ["volume", "vratio", "expansion", "bulk", "heat"]
        if q.get("gruneisen") or q.get("thermo"):
            out.append("gamma")
        if q.get("thermo"):
            out += ["free", "entropy"]
        from . import qha as Q
        if Q.free_energy_family(text or "") is not None:
            out.append("fv")
    elif kind == "CPHF":
        opt = data.get("optics") or {}
        if len(frequencies(dielectric_spectrum(text))) > 1:
            out += ["epsilon", "refractive"]
            if (opt.get("dynamic") or {}).get("absorbing"):
                out.append("absorption")
        if opt.get("principal"):
            out.append("indicatrix")
        if (opt.get("shg") or {}).get("zero") is False:
            out.append("shg")
    return [{"key": k, "label": PLOTS[k][1], "what": PLOTS[k][2]} for k in out]


def figure(data: Dict[str, Any], view: str, text: str = "", shape: str = "surface",
           compare: Optional[Sequence[Tuple[str, Dict[str, Any]]]] = None, fit: Optional[str] = None,
           name: str = "") -> Dict[str, Any]:
    """One plot's data (``views`` names the ones a run has). ``compare``, ``fit``: the enthalpy plot's
    other phases ((label, analyze() of their output)) and the fit it uses (Birch-Murnaghan by default);
    ``name``: this output's file name (the enthalpy plot's reference)."""
    if view not in PLOTS:
        raise ValueError(f"unknown plot {view!r}: one of {', '.join(PLOTS)}")
    if shape not in SHAPES:
        raise ValueError(f"shape is {' or '.join(SHAPES)}")
    have = [v["key"] for v in views(data, text)]
    if view not in have:
        raise ValueError(f"this output has no {plain(PLOTS[view][1])} plot" + (f" (it has {', '.join(have)})" if have else ""))
    if view in DIRECTIONAL:
        return directional_figure(data, view, shape)
    if PLOTS[view][0] == "EOS":
        return eos_figure(data, view, compare, fit, name)
    if view == "epsilon":
        return cphf_figure(data, text)
    if view in ("refractive", "absorption", "indicatrix", "shg"):
        return optics_figure(data, view, shape)
    return qha_figure(data, view, text)


def table(fig: Dict[str, Any]) -> Dict[str, Any]:
    """The figure's data as {columns, rows} (CSV): one row per point."""
    unit = f" ({fig['unit']})" if fig.get("unit") else ""
    if fig.get("glyph"):                      # the index ellipsoid: its points and the index there
        s = fig["surface"]
        rows = [[s["x"][i][j], s["y"][i][j], s["z"][i][j], v] for i, row in enumerate(s["value"]) for j, v in enumerate(row)]
        return {"columns": ["x", "y", "z", "n"], "rows": rows}
    if fig.get("shape") == "surface":
        env = fig.get("envelopes")
        s = env["min"] if env else fig["surface"]
        rows = []
        for i, row in enumerate(s["value"]):
            for j, v in enumerate(row):
                # the direction is the grid's: (theta_i, phi_j) as the surface was drawn
                t = math.pi * i / (len(s["value"]) - 1)
                p = 2 * math.pi * j / (len(row) - 1)
                n = [round(math.sin(t) * math.cos(p), 5) + 0.0, round(math.sin(t) * math.sin(p), 5) + 0.0, round(math.cos(t), 5) + 0.0]
                rows.append(n + ([v, env["max"]["value"][i][j]] if env else [v]))
        cols = ([f"{fig['symbol']}_min{unit}", f"{fig['symbol']}_max{unit}"] if env else [f"{fig['symbol']}{unit}"])
        return {"columns": ["n_x", "n_y", "n_z"] + [plain(c) for c in cols], "rows": rows}
    if fig.get("shape") == "planes":
        rows = []
        first = fig["planes"][0]
        if "curves" in first:
            for c in fig["planes"]:
                rows += [[c["plane"], t] + [k["r"][i] for k in c["curves"]] for i, t in enumerate(c["theta_deg"])]
            names = [plain(f"{k['name']}{unit}") for k in first["curves"]]
            return {"columns": ["plane", "angle (deg)"] + names, "rows": rows}
        for c in fig["planes"]:
            rows += [[c["plane"], t, r] for t, r in zip(c["theta_deg"], c["r"])]
        return {"columns": ["plane", "angle (deg)", plain(f"{fig['symbol']}{unit}")], "rows": rows}
    rows = []
    for s in fig["series"]:
        rows += [[plain(s["name"]), x, y] for x, y in zip(s["x"], s["y"])]
    return {"columns": ["series", plain(fig["x_label"]), plain(fig["y_label"])], "rows": rows}


# ------------------------------------------------------------------ settings
_STEP = re.compile(r"PERFORMED WITH\s+(\d+)\s+POINTS AND\s*\n\s*A STEP OF\s+(" + A._FLOAT + r")")
_QHA_T = re.compile(r"TEMPERATURE FOR THERMO ANALYSIS\s+(" + A._FLOAT + r")\s*KELVIN\s*-\s*(" + A._FLOAT
                    + r")\s*KELVIN\s*\(POINTS\s+(\d+)\)")
_QHA_N = re.compile(r"^\s*NUMBER OF POINTS:\s*(\d+)\s*$", re.M)
_MODULE_KEYS = ("ELASTCON", "ELAPIEZO", "PIEZOCON", "PIEZOCP", "PHOTOELA", "EOS", "QHA", "CPHF")
_SHOWN_FLAGS = ("CLAMPION", "NUCHESS", "HESSNUM2", "MODEPART", "PREOPTGEOM", "PRINT", "RESTART", "RESTART2",
                "AWESOME", "THIRD", "FOURTH")


def deck_block(d12: Path) -> Optional[Dict[str, Any]]:
    """The module block of a deck ({mode, ...} as d12_calc_elastic.parse_block reads it), or None."""
    try:
        lines = d12.read_text(errors="replace").splitlines()
    except OSError:
        return None
    import sys
    repo = Path(__file__).resolve().parents[2]
    if str(repo / "Crystal_d12") not in sys.path:
        sys.path.insert(0, str(repo / "Crystal_d12"))
    try:
        import d12_calc_elastic as E
    except Exception:                       # the deck tools are not importable: the output still tells
        return None
    for i, ln in enumerate(lines):
        k = ln.strip().upper()
        if k in _MODULE_KEYS:
            try:
                block, _ = E.parse_block(lines, i)
            except Exception:
                return {"mode": k}
            flags = [f for f in _SHOWN_FLAGS if block.get(f.lower()) is True or block.get("restart") == f]
            return {**block, "flags": flags}
    return None


def _module_of_output(text: str, data: Dict[str, Any]) -> Optional[str]:
    kind = kind_of(data)
    if kind == "ELASTIC":
        if "pockels" in data or "piezo_optic" in data:
            return "PHOTOELA"
        if ("piezo_e" in data or "piezo_d" in data) and "elastic" in data:
            return "ELAPIEZO"
        if "elastic" not in data:
            return "PIEZOCON"
        return "ELASTCON"
    return kind


def settings(text: str, path: Optional[Path], data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The run's key settings: [{label, value, source}] (source: out, d12 or default)."""
    rows: List[Dict[str, Any]] = []

    def add(label, value, source):
        if value not in (None, ""):
            rows.append({"label": label, "value": str(value), "source": source})
    block = None
    d12 = None
    if path is not None:
        cand = Path(path).with_suffix(".d12")
        if cand.is_file():
            d12 = cand
            block = deck_block(cand)
    mode = (block or {}).get("mode") or _module_of_output(text, data)
    add("Module", mode, "d12" if block else "out")
    if block and block.get("flags"):
        add("Options", " ".join(block["flags"]), "d12")
    if mode in ("ELASTCON", "ELAPIEZO", "PIEZOCON", "PHOTOELA"):
        m = _STEP.search(text)
        if m:
            add("NUMDERIV (points per strain)", int(m.group(1)), "out")
            add("STEPSIZE (strain step)", f"{A._f(m.group(2)):g}", "out")
        else:
            add("NUMDERIV (points per strain)", (block or {}).get("numderiv") or 3, "d12" if (block or {}).get("numderiv") else "default")
            if (block or {}).get("stepsize"):
                add("STEPSIZE (strain step)", block["stepsize"], "d12")
        if block and block.get("deform"):
            add("Strains (DEFORM)", " ".join(VOIGT[i] for i, f in enumerate(block["deform"]) if str(f) == "1"), "d12")
        if block and block.get("clampion"):
            add("Nuclear relaxation", "none (clamped ions)", "d12")
        elif block and block.get("nuchess"):
            add("Nuclear relaxation", "internal-strain Hessian (NUCHESS)", "d12")
        elif "elastic" in data:
            add("Nuclear relaxation", "atoms relaxed at each strain", "default" if block else "out")
        if (block or {}).get("dynamic"):
            add("Wavelength (DYNAMIC)", f"{block['dynamic']} nm", "d12")
    if mode == "EOS":
        pts = (data.get("eos") or {}).get("points") or []
        v0 = data.get("volume_ang3")
        if pts and v0:
            vs = [p[0] for p in pts]
            add("Volume range", f"{min(vs) / v0:.3f} – {max(vs) / v0:.3f} × V (input {v0:.4f} Å³)", "out")
            add("Volumes", f"{len(vs)} ({min(vs):.4f} – {max(vs):.4f} Å³)", "out")
        if block and block.get("range"):
            r = block["range"]
            add("RANGE", " ".join(str(x) for x in r), "d12")
        if block and block.get("prange"):
            add("PRANGE (printed pressures, GPa)", " ".join(str(x) for x in block["prange"]), "d12")
    if mode == "QHA":
        m = _QHA_T.search(text)
        if m:
            add("Temperatures", f"{A._f(m.group(1)):g} – {A._f(m.group(2)):g} K, {m.group(3)} points", "out")
        m = _QHA_N.search(text)
        if m:
            add("Volumes", m.group(1), "out")
        elif block and block.get("points"):
            add("Volumes", block["points"], "d12")
        if block and block.get("qstep"):
            add("Volume step", f"{block['qstep']} %", "d12")
        if block and block.get("scelphono"):
            add("Phonon supercell (SCELPHONO)", " ".join(str(x) for x in block["scelphono"]), "d12")
    if mode == "CPHF":
        order = ["polarizability α, ε"]
        if "beta" in data or (block or {}).get("third") or (block or {}).get("fourth"):
            order.append("β")
        if "gamma" in data or (block or {}).get("fourth"):
            order.append("γ")
        add("Response", ", ".join(order), "out" if ("beta" in data or "gamma" in data or not block) else "d12")
        fr = frequencies(dielectric_spectrum(text))
        if len(fr) > 1:
            add("Frequencies (DYNAMIC)", f"{len(fr)}, {fr[0]:.3f} – {fr[-1]:.3f} eV", "out")
        elif fr:
            add("Frequency", "static (ω = 0)" if fr[0] == 0 else f"{fr[0]:.3f} eV", "out")
        if block and block.get("damping"):
            add("DAMPING", f"{block['damping']} Ha", "d12")
        if block and block.get("tolalpha"):
            add("TOLALPHA", block["tolalpha"], "d12")
    dim = data.get("dimensionality")
    add("Periodicity", {3: "3D crystal", 2: "2D slab", 1: "1D polymer", 0: "0D molecule"}.get(dim), "out")
    if path is not None:
        from mace.gui import runinfo
        try:
            add("Functional", runinfo._functional(text, Path(path)), "d12" if d12 else "out")
            add("Basis", runinfo._basis(Path(path)), "d12")
        except Exception:                   # a deck MACE cannot read: the output's own rows stay
            pass
    return rows


# ------------------------------------------------------------------ Born stability
def born_criteria(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Each Born stability condition of the run's crystal system, [{criterion, value, unit, ok}]
    (Mouhat & Coudert, PRB 90, 224104 (2014), eqs. 12-15 and 19; any system: every eigenvalue of
    C positive, their eq. 2). A slab: C11 > 0, C11 C22 > C12^2, C66 > 0 of its in-plane block."""
    import numpy as np
    el = data.get("elastic")
    if not el:
        return []
    got = _stiffness(data)
    if got is None:
        return []
    C, dim, unit = got
    out: List[Dict[str, Any]] = []

    def add(text, value, u=None):
        out.append({"criterion": text, "value": float(value), "unit": unit if u is None else u, "ok": bool(value > 0)})
    if dim == 2:
        c11, c22, c12, c66 = C[0, 0], C[1, 1], C[0, 1], C[2, 2]
        add("C11 > 0", c11)
        add("C11 C22 − C12² > 0", c11 * c22 - c12 * c12, f"({unit})²")
        add("C66 > 0", c66)
        return out
    c = lambda i, j: C[i - 1, j - 1]                          # noqa: E731 (Voigt indices as printed)
    system = (el.get("system") or "").upper()
    if system == "CUBIC":
        add("C11 − C12 > 0", c(1, 1) - c(1, 2))
        add("C11 + 2 C12 > 0", c(1, 1) + 2 * c(1, 2))
        add("C44 > 0", c(4, 4))
    elif system == "HEXAGONAL" or system.startswith("TETRAGONAL"):
        add("C11 − |C12| > 0", c(1, 1) - abs(c(1, 2)))
        add("C33 (C11 + C12) − 2 C13² > 0", c(3, 3) * (c(1, 1) + c(1, 2)) - 2 * c(1, 3) ** 2, f"{unit}²")
        add("C44 > 0", c(4, 4))
        if system.startswith("TETRAGONAL"):
            add("C66 > 0", c(6, 6))
            if abs(c(1, 6)) > 1e-6:
                add("C66 (C11 − C12) − 2 C16² > 0", c(6, 6) * (c(1, 1) - c(1, 2)) - 2 * c(1, 6) ** 2, f"{unit}²")
    elif system == "ORTHORHOMBIC":
        add("C11 > 0", c(1, 1))
        add("C11 C22 − C12² > 0", c(1, 1) * c(2, 2) - c(1, 2) ** 2, f"{unit}²")
        add("det of the normal block > 0", c(1, 1) * c(2, 2) * c(3, 3) + 2 * c(1, 2) * c(1, 3) * c(2, 3)
            - c(1, 1) * c(2, 3) ** 2 - c(2, 2) * c(1, 3) ** 2 - c(3, 3) * c(1, 2) ** 2, f"{unit}³")
        for k in (4, 5, 6):
            add(f"C{k}{k} > 0", c(k, k))
    eig = np.linalg.eigvalsh(C)
    add("smallest eigenvalue of C > 0", float(eig.min()))
    return out


# ------------------------------------------------------------------ the document
def document(text: str, path: Optional[Path] = None, epsilon=None) -> Dict[str, Any]:
    """Everything the results document shows: analyze() plus kind, settings, views
    and, for CPHF, the tensors as 3x3 matrices. ``epsilon``: as analyze()'s."""
    data = A.analyze(text, epsilon)
    name = Path(path).name if path is not None else ""
    out: Dict[str, Any] = {"name": name, "kind": kind_of(data), "data": data, "report": A.report(data, name),
                           "empty": not data}
    if not data:
        out.update(settings=settings(text, path, data) if path is not None else [], views=[])
        return out
    out["settings"] = settings(text, path, data)
    out["views"] = views(data, text)
    de = data.get("dielectric")
    if de:
        rows = dielectric_spectrum(text) or de["components"]
        f0 = frequencies(rows)[0]
        first = [r for r in rows if r["frequency_ev"] == f0]
        out["cphf"] = {"frequency_ev": f0, "epsilon": tensor3(first, "epsilon"), "alpha": tensor3(first, "alpha"),
                       "chi1": tensor3(first, "chi1"), "n_frequencies": len(frequencies(rows))}
        if (data.get("optics") or {}).get("principal"):
            out["cphf"]["principal"] = data["optics"]["principal"]
    g = (data.get("qha") or {}).get("gruneisen")
    if g:
        # the row of CRYSTAL's Grueneisen table the database's qha_gamma_mode_300k comes from
        from .properties import nearest_300k
        out["gruneisen_300k"] = nearest_300k(g.get("temperature_k"))
    out["born"] = born_criteria(data)
    der = data.get("derived") or {}
    if der.get("directional"):
        out["directional"] = der["directional"]
    return out


def document_file(path, epsilon=None) -> Dict[str, Any]:
    p = Path(path)
    return document(p.read_text(errors="replace"), p, epsilon)
