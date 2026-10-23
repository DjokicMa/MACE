"""Optical quantities of a CPHF/KS run (CRYSTAL23 manual 10.2) from its dielectric tensor and first
hyperpolarizability.

* Principal values and axes of eps_inf (the eigenvectors of the symmetric 3x3 tensor), the principal
  refractive indices n_i = sqrt(eps_i) and the index ellipsoid (indicatrix) x^2/n1^2 + y^2/n2^2 +
  z^2/n3^2 = 1 in the principal frame; the optical class (isotropic, uniaxial: two equal n, biaxial)
  and the birefringence Delta n = n_max - n_min; a uniaxial crystal's sign, positive when the
  extraordinary index n_e (the distinct one) exceeds the ordinary n_o.
* A DYNAMIC run: the complex refractive index n + i kappa = sqrt(eps' + i eps'') of each diagonal
  component, n = sqrt((|eps| + eps')/2), kappa = sqrt((|eps| - eps')/2), and the absorption
  coefficient alpha = 2 omega kappa / c (cm^-1) when eps has an imaginary part (DAMPING).
* SHG: the second-order tensor d_ijk (pm/V, CRYSTAL's d(MKS) column, d = chi(2)/2, Kleinman-
  symmetric as printed: d_ijk is the same for every order of i, j, k). For a pump polarized along n
  the second-harmonic polarization is P_i = d_ijk n_j n_k: its magnitude |d(n)| and its component
  along n, the longitudinal d_L(n) = d_ijk n_i n_j n_k. d_eff of common configurations: the pump and
  the analyzed harmonic both along x, y or z (d_11, d_22, d_33), the direction of the largest d_L
  (one polarization, any orientation), the largest |d(n)|, and the orientational (Kurtz-Perry
  powder) averages <d_L^2>^(1/2) and <|d(n)|^2>^(1/2) over the sphere. No phase matching or dispersion
  is included: the tensor is the run's (static, or its one frequency). A beta table CRYSTAL printed
  as an overflow or with a diverged response (analysis.parse's ``diverged``), or without every
  independent component, gives no d and no conclusion (``zero`` None): only a complete, converged
  table of zeros means a centrosymmetric crystal.
* The crystal symmetry check: the space group's crystal family fixes the optical class (cubic:
  isotropic; tetragonal, hexagonal and trigonal: uniaxial, Nye, Physical Properties of Crystals,
  ch. XIII). A tensor less symmetric than that (a CPHF that did not converge to the crystal's
  response) is reported with a warning, and its class-dependent results (principal n, birefringence,
  2V, optic sign) are not stored in the database (properties.optics_properties).

Numpy only.
"""
from __future__ import annotations

import itertools
import math
from typing import Any, Dict, List, Optional, Sequence

from . import tensor as T

HBAR_C_EV_CM = 1.973269804e-5          # eV cm


def _r(x, digits=10):
    return None if x is None or not math.isfinite(x) else float(f"{x:.{digits}g}")


#: the optical class each crystal family requires (CRYSTAL's CRYSTAL FAMILY line; trigonal is HEXAGONAL)
FAMILY_CLASS = {"CUBIC": "isotropic", "TETRAGONAL": "uniaxial", "HEXAGONAL": "uniaxial", "TRIGONAL": "uniaxial",
                "RHOMBOHEDRAL": "uniaxial"}
_RANK = {"isotropic": 0, "uniaxial": 1, "biaxial": 2}


def tensor_class(eps: Sequence[Sequence[float]], tol: float = 2e-3) -> str:
    """isotropic, uniaxial or biaxial: how many principal values agree within ``tol`` of the largest
    (orientation-free; looser than principal()'s, so that a converged tensor's last digits pass)."""
    import numpy as np
    E = np.array(eps, dtype=float)
    w = np.linalg.eigvalsh((E + E.T) / 2.0)
    scale = max(float(np.abs(w).max()), 1e-12)
    same01, same12 = w[1] - w[0] <= tol * scale, w[2] - w[1] <= tol * scale
    return "isotropic" if same01 and same12 else "uniaxial" if same01 or same12 else "biaxial"


def symmetry_mismatch(eps: Sequence[Sequence[float]], family: Optional[str], tol: float = 2e-3) -> Optional[Dict[str, str]]:
    """{expected, found} when the tensor is less symmetric than the crystal family requires (a cubic
    crystal's eps not isotropic, a tetragonal, hexagonal or trigonal one's biaxial); None when it
    agrees, or for a family that constrains nothing (orthorhombic, monoclinic, triclinic) or none known.
    A tensor more symmetric than required (an accidental degeneracy) is no error."""
    exp = FAMILY_CLASS.get((family or "").upper())
    if not exp:
        return None
    found = tensor_class(eps, tol)
    return {"expected": exp, "found": found} if _RANK[found] > _RANK[exp] else None


def principal(eps: Sequence[Sequence[float]], tol: float = 1e-4, lattice: bool = False) -> Dict[str, Any]:
    """Principal values and axes of a dielectric tensor (ascending), the refractive indices, the optical
    class, the birefringence and, for a uniaxial crystal, its sign and optic axis. A non-positive
    principal value (no real index): ``unphysical``, without a class. ``lattice``: direction_label's."""
    import numpy as np
    E = np.array(eps, dtype=float)
    E = (E + E.T) / 2.0
    w, v = np.linalg.eigh(E)
    n = np.sqrt(np.clip(w, 0.0, None))
    axes = [[round(float(x), 6) + 0.0 for x in v[:, k]] for k in range(3)]
    out: Dict[str, Any] = {"epsilon": [_r(x) for x in w], "n": [_r(x) for x in n], "axes": axes,
                           "axis_labels": [T.direction_label(a, lattice=lattice) for a in axes],
                           "birefringence": _r(float(n.max() - n.min()))}
    if float(w.min()) <= 0.0:
        out.update(optical_class="unphysical", unphysical=True)
        return out
    scale = max(float(n.max()), 1e-12)
    same01, same12 = abs(n[1] - n[0]) <= tol * scale, abs(n[2] - n[1]) <= tol * scale
    if same01 and same12:
        out["optical_class"] = "isotropic"
    elif same01 or same12:
        k = 2 if same01 else 0                 # the distinct (extraordinary) index
        no, ne = float(n[1]), float(n[k])
        out.update(optical_class="uniaxial", n_o=_r(no), n_e=_r(ne), birefringence_signed=_r(ne - no),
                   sign="positive" if ne > no else "negative", optic_axis=axes[k],
                   optic_axis_label=T.direction_label(axes[k], lattice=lattice))
    else:
        out["optical_class"] = "biaxial"
        # the optic axes lie in the plane of the largest and smallest index, at V_z from the largest one's axis:
        # tan V_z = (n_z/n_x) sqrt((n_y^2 - n_x^2)/(n_z^2 - n_y^2)) (Born & Wolf, Principles of Optics, 15.3);
        # 2V is the acute angle, bisected by n_z (positive) or by n_x (negative)
        nx, ny, nz = (float(x) for x in n)
        vz = 2 * math.degrees(math.atan(nz / nx * math.sqrt((ny ** 2 - nx ** 2) / (nz ** 2 - ny ** 2))))
        out["optic_axial_angle_deg"] = _r(vz if vz <= 90 else 180 - vz)
        out["sign"] = "positive" if vz <= 90 else "negative"
    return out


def indicatrix(pr: Dict[str, Any], n_theta: int = 37, n_phi: int = 73) -> Dict[str, Any]:
    """The index ellipsoid as a surface: semi-axes n_i along the principal axes."""
    import numpy as np
    t, p = np.meshgrid(np.linspace(0.0, math.pi, n_theta), np.linspace(0.0, 2 * math.pi, n_phi), indexing="ij")
    u = np.stack([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)], axis=-1)       # principal-frame unit sphere
    R = np.array(pr["axes"], dtype=float)                                                  # rows: the principal axes
    pts = (u * np.array(pr["n"], dtype=float)) @ R
    # the index for light polarized along each point's direction (the radius of the ellipsoid there)
    r = np.linalg.norm(pts, axis=-1)
    return {"x": pts[..., 0].round(6).tolist(), "y": pts[..., 1].round(6).tolist(), "z": pts[..., 2].round(6).tolist(),
            "value": r.round(6).tolist()}


def complex_index(eps_re: float, eps_im: float) -> Dict[str, float]:
    m = math.hypot(eps_re, eps_im)
    return {"n": math.sqrt(max(0.0, (m + eps_re) / 2)), "kappa": math.sqrt(max(0.0, (m - eps_re) / 2))}


def dynamic(rows: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """n(omega), kappa(omega) and the absorption coefficient of each diagonal component of a DYNAMIC
    run's table (None for a single frequency)."""
    freqs = sorted({r["frequency_ev"] for r in rows})
    if len(freqs) < 2:
        return None
    out: Dict[str, Any] = {"frequency_ev": freqs, "components": {}, "absorbing": False}
    for comp in ("XX", "YY", "ZZ"):
        pts = sorted((r["frequency_ev"], r["epsilon"], r["epsilon_im"]) for r in rows if r["component"] == comp)
        if not pts:
            continue
        ci = [complex_index(e, i) for _, e, i in pts]
        absorb = any(abs(i) > 1e-8 for _, _, i in pts)
        out["absorbing"] = out["absorbing"] or absorb
        out["components"][comp.lower()] = {
            "frequency_ev": [p[0] for p in pts], "n": [_r(c["n"], 8) for c in ci], "kappa": [_r(c["kappa"], 8) for c in ci],
            "absorption_cm": [_r(2 * p[0] * c["kappa"] / HBAR_C_EV_CM, 8) for p, c in zip(pts, ci)]}
    return out


# ------------------------------------------------------------------ SHG
def shg_tensor(beta: Dict[str, Any], key: str = "mks"):
    """d_ijk (3x3x3, pm/V by default) from the beta block's components (XXX, XXY, ...): each listed
    component given to every order of its indices (Kleinman symmetry, as CRYSTAL symmetrizes it)."""
    import numpy as np
    d = np.zeros((3, 3, 3))
    for r in beta.get("components") or []:
        c = r["component"]
        if len(c) != 3:
            continue
        idx = ["XYZ".index(x) for x in c]
        for p in set(itertools.permutations(idx)):
            d[p] = r[key]
    return d


def shg_values(d, n, longitudinal: bool):
    """|d(n)| = |d_ijk n_j n_k| or d_L(n) = d_ijk n_i n_j n_k along unit directions n."""
    import numpy as np
    n = T._unit(n)
    P = np.einsum("ijk,nj,nk->ni", d, n, n)
    return np.einsum("ni,ni->n", P, n) if longitudinal else np.linalg.norm(P, axis=1)


def _sphere_mean(f) -> float:
    """The mean of f over the unit sphere (a 1-degree theta/phi grid weighted by sin(theta))."""
    import numpy as np
    t, p = np.meshgrid(np.radians(np.arange(0.5, 180.0, 1.0)), np.radians(np.arange(0.5, 360.0, 1.0)), indexing="ij")
    n = np.stack([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)], axis=-1).reshape(-1, 3)
    w = np.sin(t).reshape(-1)
    return float((f(n) * w).sum() / w.sum())


def shg(beta: Optional[Dict[str, Any]], lattice: bool = False) -> Dict[str, Any]:
    """d_eff of the common configurations and the directional extremes (pm/V); {"zero": True} for a
    centrosymmetric crystal (every component of a complete, converged table below 1e-6 pm/V); "zero"
    None with ``diverged`` (overflow or a diverged response) or ``incomplete`` (not every independent
    component read): no d and no conclusion."""
    import numpy as np
    if not beta or not beta.get("components"):
        return {}
    if beta.get("diverged"):
        return {"zero": None, "diverged": True, "unit": "pm/V"}
    if beta.get("complete") is False:
        return {"zero": None, "incomplete": True, "unit": "pm/V"}
    d = shg_tensor(beta)
    if float(np.abs(d).max()) < 1e-6:
        return {"zero": True, "unit": "pm/V"}
    lon = T.extremes(lambda x: shg_values(d, x, True), 3, full=True, lattice=lattice)
    mag = T.extremes(lambda x: shg_values(d, x, False), 3, lattice=lattice)
    return {"zero": False, "unit": "pm/V", "tensor": d.round(8).tolist(),
            "d_axes": [_r(float(d[i, i, i])) for i in range(3)],
            "longitudinal": {k: lon[k] for k in ("min", "max")}, "magnitude": {k: mag[k] for k in ("min", "max")},
            "powder_longitudinal": _r(math.sqrt(_sphere_mean(lambda x: shg_values(d, x, True) ** 2))),
            "powder_magnitude": _r(math.sqrt(_sphere_mean(lambda x: shg_values(d, x, False) ** 2)))}


def _fmt(values) -> str:
    return ", ".join(f"{x:.4g}" for x in values)


def summary(text: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """What analyze() files under ``optics``: the principal analysis of the first frequency's eps (with
    the crystal symmetry check), the CPHF convergence CRYSTAL printed (``cphf``), the dynamic n, kappa
    and absorption, the SHG d_eff, and ``warnings``."""
    from . import analysis as A
    from .views import dielectric_spectrum, frequencies, tensor3
    out: Dict[str, Any] = {}
    warnings: List[str] = []
    lattice = T.is_cubic(data)
    de = data.get("dielectric")
    if de:
        rows = dielectric_spectrum(text) or de["components"]
        f0 = frequencies(rows)[0]
        eps = tensor3([r for r in rows if r["frequency_ev"] == f0], "epsilon")
        pr = {"frequency_ev": f0, **principal(eps, lattice=lattice)}
        out["principal"] = pr
        if pr.get("unphysical"):
            warnings.append(f"eps_inf has a non-positive principal value ({_fmt(pr['epsilon'])}): the CPHF tensor is "
                            "unphysical; its principal indices, optical class and 2V are not stored")
        else:
            mm = symmetry_mismatch(eps, data.get("crystal_family"))
            if mm:
                pr.update(symmetry_mismatch=True, expected_class=mm["expected"])
                warnings.append(f"eps_inf is {mm['found']} (principal values {_fmt(pr['epsilon'])}) but the space group "
                                f"{data.get('space_group') or ''} is {data['crystal_family'].lower()}, so it must be "
                                f"{mm['expected']}: the CPHF tensor breaks the crystal symmetry (not the crystal's "
                                "response); its principal indices, optical class, 2V and optic sign are not stored")
        diag = A.cphf_diagnostics(text)
        if diag:
            out["cphf"] = diag
            if diag.get("unstable"):
                warnings.append(f"The CPHF DIIS test reached {diag['max_diis_test']:.3g} (a stable run stays below 10): "
                                "the response equations were unstable; its principal indices, optical class, 2V "
                                "and LO-TO splitting are not stored")
            if diag.get("unconverged"):
                warnings.append(f"The CPHF along {', '.join(diag['unconverged'])} stopped at the cycle limit "
                                f"({diag['max_cycles']}) without converging; its principal indices, optical class, "
                                "2V and LO-TO splitting are not stored")
        dy = dynamic(rows)
        if dy:
            out["dynamic"] = dy
    s = shg(data.get("beta"), lattice)
    if s:
        out["shg"] = s
    if warnings:
        out["warnings"] = warnings
    return out
