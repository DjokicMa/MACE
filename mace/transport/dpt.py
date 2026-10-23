"""Deformation-potential (acoustic-phonon) relaxation times and mobilities.

Longitudinal acoustic phonons scatter a carrier through the shift of its
band edge with strain, E1 = dE_edge/d(epsilon) (Bardeen & Shockley, Phys. Rev.
80, 72 (1950)). In the elastic, equipartition limit and for a parabolic
band of effective mass m*, the phonon-limited mobilities are

3D  mu = 2 sqrt(2 pi) e hbar^4 C_ii / (3 m*^(5/2) (k_B T)^(3/2) E1^2)
        C_ii: the longitudinal elastic constant along the transport
        direction (Pa) - Bardeen & Shockley (1950), eq. (54);
2D  mu = e hbar^3 C_2D / (k_B T m* m_d E1^2),  m_d = sqrt(m_x m_y)
        C_2D: the in-plane stretching modulus (N/m) - Takagi et al., IEEE
        Trans. Electron Devices 41, 2357 (1994); Qiao et al., Nat. Commun. 5,
        4475 (2014);
1D  mu = e hbar^2 C_1D / ((2 pi k_B T)^(1/2) m*^(3/2) E1^2)
        C_1D: the stretching modulus per unit length (J/m) - Beleznay, Bogar
        & Ladik, J. Chem. Phys. 119, 5690 (2003).

The relaxation time follows from mu = e <<tau>> / m*: <<tau>> = mu m* / e,
the momentum-relaxation time averaged over the non-degenerate carrier
distribution - the constant tau that gives the same conductivity, i.e.
the value for BOLTZTRA's RELAXTIM.

Two masses. The scattering rate follows the density of final states, so
tau takes the density-of-states mass m_d, and mu = e <<tau>> / m_c the
conductivity (band) mass m_c along the transport direction (Herring &
Vogt, Phys. Rev. 101, 944 (1956); in this form for DFT, e.g. Xi et al.,
Nanoscale 4, 4348 (2012)):

3D  mu = 2 sqrt(2 pi) e hbar^4 C_ii / (3 m_c m_d^(3/2) (k_B T)^(3/2) E1^2),
    <<tau>> = 2 sqrt(2 pi) hbar^4 C_ii / (3 m_d^(3/2) (k_B T)^(3/2) E1^2)
    (m_d of one valley, (m1 m2 m3)^(1/3): the acoustic scattering is
    intravalley);
2D  mu = e hbar^3 C_2D / (k_B T m_c m_d E1^2), tau = hbar^3 C_2D / (k_B T m_d E1^2);
1D  mu = e hbar^2 C_1D / ((2 pi k_B T)^(1/2) m_c m_d^(1/2) E1^2), one band:
    m_c = m_d = m, tau = hbar^2 C_1D / ((2 pi k_B T)^(1/2) m^(1/2) E1^2).

An isotropic band has m_c = m_d = m*, the forms above. So RELAXTIM depends
on m_d alone; m_c changes only the mobility printed with it.

For 3D the scattering rate is energy dependent (it follows the density of
states): 1/tau(E) = D^2 k_B T m_d^(3/2) sqrt(2E) / (pi hbar^4 C_ii), i.e.
tau(E) = tau_0 (E / k_B T)^(-1/2) with tau_0 = pi hbar^4 C_ii /
(sqrt(2) E1^2 m_d^(3/2) (k_B T)^(3/2)), E measured from the band edge
(Lundstrom, Fundamentals of Carrier Transport, 2nd ed., sec. 2.8). Its
Maxwell-Boltzmann average <<tau>> = tau_0 Gamma(2)/Gamma(5/2) =
4 tau_0 / (3 sqrt(pi)) reproduces the 3D mobility above. In 2D the parabolic
density of states is constant, so tau does not depend on E.

What the estimate leaves out (stated in every provenance): optical-phonon,
polar (Froehlich), impurity and boundary scattering, degenerate statistics
and non-parabolicity. It is the acoustic-phonon limit, an upper bound on
tau where other mechanisms matter.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

E_CHARGE = 1.602176634e-19        # C
HBAR = 1.054571817e-34            # J s
K_B = 1.380649e-23                # J/K
M_E = 9.1093837015e-31            # kg

REFERENCES = {
    3: "J. Bardeen & W. Shockley, Phys. Rev. 80, 72 (1950)",
    2: "S. Takagi et al., IEEE Trans. Electron Devices 41, 2357 (1994); J. Qiao et al., Nat. Commun. 5, 4475 (2014)",
    1: "F. B. Beleznay, F. Bogar & J. Ladik, J. Chem. Phys. 119, 5690 (2003)",
}
FORMULAS = {
    3: "mu = 2 sqrt(2 pi) e hbar^4 C / (3 m* m_d^(3/2) (kB T)^(3/2) E1^2), m_d = m* unless given; tau = mu m*/e",
    2: "mu = e hbar^3 C2D / (kB T m* m_d E1^2), m_d = sqrt(m_x m_y) unless given; tau = mu m*/e",
    1: "mu = e hbar^2 C1D / ((2 pi kB T)^(1/2) m* m_d^(1/2) E1^2), m_d = m* unless given; tau = mu m*/e",
}
C_UNITS = {3: "GPa", 2: "N/m", 1: "J/m"}
LIMITS = ("acoustic-phonon (deformation-potential) scattering only: optical, polar (Froehlich), impurity "
          "and boundary scattering, degenerate statistics and non-parabolicity are left out; the result is "
          "an upper bound on tau where those matter")


class TauError(ValueError):
    """An input the estimate cannot use (the message says which and why)."""


def _positive(name: str, v: Optional[float]) -> float:
    if v is None:
        raise TauError(f"{name} is required")
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise TauError(f"{name} must be a number, not {v!r}")
    if not math.isfinite(x) or x <= 0:
        raise TauError(f"{name} must be positive (got {v})")
    return x


def dp_mobility(dim: int, m_star: float, e1_ev: float, c: float, temperature_k: float = 300.0,
                m_other: Optional[float] = None, m_dos: Optional[float] = None) -> Dict[str, float]:
    """Deformation-potential mobility and relaxation time.

    dim: 3, 2 or 1. m_star: the transport-direction effective mass in
    electron masses. e1_ev: the deformation potential in eV (its sign does
    not matter). c: the elastic constant in GPa (3D), N/m (2D) or J/m (1D).
    m_other (2D): the in-plane mass across the transport direction (for
    m_d; default m_star, an isotropic band). m_dos: the density-of-states
    mass m_d itself (default m_star, or sqrt(m* m_other) in 2D; in 1D it
    differs from m* only for several bands at the edge). m_star is then the
    conductivity mass m_c.

    Returns {"mobility_cm2_Vs", "tau_s", "tau_fs"}.
    """
    if dim not in (1, 2, 3):
        raise TauError("dim is 1, 2 or 3")
    m = _positive("m*", m_star) * M_E
    e1 = abs(float(e1_ev)) * E_CHARGE
    if e1 == 0:
        raise TauError("E1 must be non-zero")
    cc = _positive("C", c)
    t = _positive("T", temperature_k)
    kt = K_B * t
    md = _positive("m_d", m_dos) * M_E if m_dos is not None else None
    if dim == 3:
        cc *= 1e9                                            # GPa -> Pa
        md = md or m
        mu = 2.0 * math.sqrt(2.0 * math.pi) * E_CHARGE * HBAR ** 4 * cc / (3.0 * m * md ** 1.5 * kt ** 1.5 * e1 ** 2)
    elif dim == 2:
        md = md or math.sqrt(m * (_positive("m_other", m_other) * M_E if m_other is not None else m))
        mu = E_CHARGE * HBAR ** 3 * cc / (kt * m * md * e1 ** 2)
    else:
        md = md or m
        mu = E_CHARGE * HBAR ** 2 * cc / (math.sqrt(2.0 * math.pi * kt) * m * md ** 0.5 * e1 ** 2)
    tau = mu * m / E_CHARGE
    return {"mobility_cm2_Vs": mu * 1e4, "tau_s": tau, "tau_fs": tau * 1e15}


def tau_from_mobility(mobility_cm2_vs: float, m_star: float) -> Dict[str, float]:
    """tau = mu m*/e from a mobility (measured, or from another method)."""
    mu = _positive("mobility", mobility_cm2_vs) * 1e-4
    m = _positive("m*", m_star) * M_E
    tau = mu * m / E_CHARGE
    return {"mobility_cm2_Vs": mu * 1e4, "tau_s": tau, "tau_fs": tau * 1e15}


def tau0_3d(m_star: float, e1_ev: float, c_gpa: float, temperature_k: float = 300.0) -> float:
    """tau_0 (s) of tau(E) = tau_0 (E/kBT)^(-1/2), 3D parabolic band."""
    m = _positive("m*", m_star) * M_E
    e1 = abs(float(e1_ev)) * E_CHARGE
    kt = K_B * _positive("T", temperature_k)
    c = _positive("C", c_gpa) * 1e9
    return math.pi * HBAR ** 4 * c / (math.sqrt(2.0) * e1 ** 2 * m ** 1.5 * kt ** 1.5)


def tau_energy_table(m_star: float, e1_ev: float, c_gpa: float, temperature_k: float = 300.0,
                     energies_kt: Optional[List[float]] = None) -> List[Dict[str, float]]:
    """tau(E) at E = x kBT above the band edge (3D), in fs; m_star here is m_d."""
    t0 = tau0_3d(m_star, e1_ev, c_gpa, temperature_k)
    kt_ev = K_B * temperature_k / E_CHARGE
    xs = energies_kt or [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
    return [{"energy_kT": x, "energy_eV": x * kt_ev, "tau_fs": t0 * x ** -0.5 * 1e15} for x in xs]


def estimate(dim: int, m_star: float, e1_ev: float, c: float, temperature_k: float = 300.0,
             m_other: Optional[float] = None, sources: Optional[Dict[str, str]] = None,
             carrier: Optional[str] = None, axis: Optional[str] = None,
             m_dos: Optional[float] = None) -> Dict[str, Any]:
    """dp_mobility with its provenance (and tau(E) in 3D): what MACE reports."""
    res = dp_mobility(dim, m_star, e1_ev, c, temperature_k, m_other, m_dos)
    src = dict(sources or {})
    out: Dict[str, Any] = {
        **res,
        "method": "deformation-potential theory (acoustic-phonon scattering)",
        "formula": FORMULAS[dim], "reference": REFERENCES[dim], "limits": LIMITS,
        "dimensionality": dim, "temperature_k": float(temperature_k),
        "inputs": {
            "m_star": {"value": float(m_star), "unit": "m_e", "source": src.get("m_star", "given")},
            "E1": {"value": abs(float(e1_ev)), "unit": "eV", "source": src.get("E1", "given")},
            "C": {"value": float(c), "unit": C_UNITS[dim], "source": src.get("C", "given")},
        },
        "relaxtim": f"RELAXTIM\n{res['tau_fs']:.4g}",
    }
    if dim == 2 and m_other is not None and m_dos is None:
        out["inputs"]["m_other"] = {"value": float(m_other), "unit": "m_e", "source": src.get("m_other", "given")}
    if m_dos is not None:
        out["inputs"]["m_d"] = {"value": float(m_dos), "unit": "m_e", "source": src.get("m_dos", "given")}
    if carrier:
        out["carrier"] = carrier
    if axis:
        out["axis"] = axis
    if dim == 3:
        out["tau_of_energy"] = tau_energy_table(m_dos if m_dos is not None else m_star, e1_ev, c, temperature_k)
        out["tau_of_energy_formula"] = "tau(E) = tau_0 (E/kB T)^(-1/2), <<tau>> = 4 tau_0 / (3 sqrt(pi))"
    return out


def estimate_from_mobility(mobility_cm2_vs: float, m_star: float, sources: Optional[Dict[str, str]] = None
                           ) -> Dict[str, Any]:
    res = tau_from_mobility(mobility_cm2_vs, m_star)
    src = dict(sources or {})
    return {**res, "method": "tau = mu m*/e from a mobility", "formula": "tau = mu m* / e",
            "reference": "Drude relation; mu from the source named in inputs",
            "limits": "as good as the mobility and m*: a measured Hall mobility includes every scattering mechanism",
            "inputs": {"mobility": {"value": float(mobility_cm2_vs), "unit": "cm^2/(V s)",
                                    "source": src.get("mobility", "given")},
                       "m_star": {"value": float(m_star), "unit": "m_e", "source": src.get("m_star", "given")}},
            "relaxtim": f"RELAXTIM\n{res['tau_fs']:.4g}"}


def report(res: Dict[str, Any]) -> str:
    """The plain-text result with its provenance."""
    lines = [f"tau = {res['tau_fs']:.4g} fs   (mobility {res['mobility_cm2_Vs']:.4g} cm^2/(V s))"]
    if res.get("carrier") or res.get("axis"):
        lines[0] += f"   [{' '.join(x for x in (res.get('carrier'), res.get('axis')) if x)}]"
    lines.append(f"Method:    {res['method']}")
    lines.append(f"Formula:   {res['formula']}")
    lines.append(f"Reference: {res['reference']}")
    if "temperature_k" in res:
        lines.append(f"T:         {res['temperature_k']:g} K")
    lines.append("Inputs:")
    for k, v in res["inputs"].items():
        lines.append(f"  {k:9s} {v['value']:.6g} {v['unit']:9s} from {v['source']}")
    if res.get("tau_of_energy"):
        lines.append("Energy-resolved tau(E) (3D parabolic band, E above the band edge):")
        for row in res["tau_of_energy"]:
            lines.append(f"  E = {row['energy_kT']:5.2f} kT ({row['energy_eV'] * 1000:7.2f} meV)   tau = {row['tau_fs']:9.4g} fs")
    lines.append(f"Limits:    {res['limits']}")
    lines.append("For BOLTZTRA (opt2d3 TRANSPORT deck):  " + res["relaxtim"].replace("\n", " "))
    return "\n".join(lines)
