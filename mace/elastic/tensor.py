"""Directional and derived elastic properties of a stiffness tensor, with the
conventions of ELATE (Gaillac, Pullumbi, Coudert, J. Phys.: Condens. Matter
28, 275201 (2016)).

The Voigt compliance S (engineering shear strains) becomes the four-index
S_ijkl with S_ijkl = S_mn times 1/2 for each of m, n that is a shear index
(4, 5, 6); the stiffness C_ijkl = C_mn without factors. For unit vectors n
and m (m perpendicular to n):

* Young's modulus           1/E(n) = S_ijkl n_i n_j n_k n_l
* linear compressibility    beta(n) = S_ijkk n_i n_j
* shear modulus             1/G(n, m) = 4 S_ijkl n_i m_j n_k m_l
* Poisson's ratio           nu(n, m) = -E(n) S_ijkl n_i n_j m_k m_l

G and nu depend on two directions. For a given n both are quadratic forms in
m = cos(chi) u + sin(chi) v (u, v an orthonormal pair perpendicular to n),
so their minimum and maximum over m are the eigenvalues of a 2x2 matrix:
exact, with the m that gives each (ELATE scans chi numerically).

* Christoffel: Gamma_ik = C_ijkl n_j n_l, rho v^2 its eigenvalues (the
  quasi-longitudinal and two quasi-shear phase velocities), and the group
  velocity v_g,l = C_ijkl p_i p_k n_j / (rho v) of the polarization p
  (Musgrave, Crystal Acoustics, 1970; Jaeken & Cottenier, Comput. Phys.
  Commun. 207, 445 (2016)). With C in GPa and rho in g/cm^3, v is in km/s.

Indices and estimates (3D, Hill moduli K and G in GPa):

* universal anisotropy A^U = 5 G_V/G_R + K_V/K_R - 6 (Ranganathan &
  Ostoja-Starzewski, PRL 101, 055504 (2008)); Zener A = 2 C44/(C11 - C12)
  (cubic); Chung-Buessem A_C = (G_V - G_R)/(G_V + G_R) (J. Appl. Phys. 38,
  2535 (1967)); the ratios E_max/E_min, G_max/G_min, nu_max/nu_min;
* Pugh's ratio K/G, ductile above 1.75 (Pugh, Philos. Mag. 45, 823 (1954));
* Cauchy relations C12 = C66, C13 = C55, C23 = C44 (cubic: the Cauchy
  pressure C12 - C44; Pettifor, Mater. Sci. Technol. 8, 345 (1992));
* Vickers hardness: Chen, H_V = 2 (k^2 G)^0.585 - 3, k = G/K (Chen, Niu,
  Li, Li, Intermetallics 19, 1275 (2011); a soft solid, where the formula
  goes negative, gets 0, flagged and not stored); Tian, H_V = 0.92 k^1.137 G^0.708
  (Tian, Xu, Zhao, Int. J. Refract. Met. Hard Mater. 33, 93 (2012));
* minimum thermal conductivity: Clarke, k = 0.87 k_B M_a^(-2/3) E^(1/2)
  rho^(1/6), M_a the mean atomic mass (Clarke, Surf. Coat. Technol. 163-164,
  67 (2003)); Cahill, k = (pi/6)^(1/3) k_B n^(2/3) (v_l + 2 v_t), n the
  number of atoms per volume (Cahill, Watson, Pohl, PRB 46, 6131 (1992)).

Piezoelectric (3x6 Voigt tensors as CRYSTAL prints them): the longitudinal
coefficients d_L(n) = d_ijk n_i n_j n_k (d_ijk = d_iJ/2 for a shear J) and
e_L(n) = e_ijk n_i n_j n_k (e_ijk = e_iJ). The electromechanical coupling
along n, k^2(n) = d_L(n)^2 / (s_L(n) eps0 eps_L(n)) with s_L = 1/E(n) and
eps_L = n.eps.n, is the IEEE longitudinal factor (k33 along z) generalized
to any direction; eps must be the stress-free static permittivity eps^T,
which an ELAPIEZO run does not print: it is given (--epsilon), or the run's
own electronic eps_inf is used, which gives an upper bound (eps^T >= eps_inf).

A slab (2D): the in-plane block (xx, yy, xy) in N/m; E(theta), nu(theta)
and G(theta) with m the in-plane normal to n, and the 2D anisotropy index
A^SU = sqrt[(K_V/K_R - 1)^2 + 2 (G_V/G_R - 1)^2] (Li, Shao, Gao, Liu,
Extreme Mech. Lett. 34, 100615 (2020)).

Directions are Cartesian (x, y, z): a symmetry direction is labeled as a
combination of the axes (x+y), or as the lattice direction [u v w] in a
cubic crystal, where the two agree; any other direction by its components.

A mechanically unstable C (not positive definite) gives all of these, flagged
``unstable``: they are unphysical and the database does not store them.

Numpy only.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

PAIRS = {6: ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)), 3: ((0, 0), (1, 1), (0, 1))}
K_B = 1.380649e-23
EPS0 = 8.8541878128e-12
AMU_KG = 1.66053906660e-27
DUCTILE_PUGH = 1.75
HARDNESS_REFS = {"chen": "Chen, Niu, Li, Li, Intermetallics 19, 1275 (2011)",
                 "tian": "Tian, Xu, Zhao, Int. J. Refract. Met. Hard Mater. 33, 93 (2012)"}
THERMAL_REFS = {"clarke": "Clarke, Surf. Coat. Technol. 163-164, 67 (2003)",
                "cahill": "Cahill, Watson, Pohl, Phys. Rev. B 46, 6131 (1992)"}


# ------------------------------------------------------------------ tensors
def voigt_to_tensor(M, compliance: bool):
    """The four-index tensor of a Voigt matrix (6x6 in 3D, 3x3 xx, yy, xy for a slab)."""
    import numpy as np
    M = np.asarray(M, dtype=float)
    pairs = PAIRS[len(M)]
    d = 3 if len(M) == 6 else 2
    T = np.zeros((d, d, d, d))
    for a, (i, j) in enumerate(pairs):
        for b, (k, l) in enumerate(pairs):
            f = ((0.5 if i != j else 1.0) * (0.5 if k != l else 1.0)) if compliance else 1.0
            for p, q in ((i, j), (j, i)):
                for r, s in ((k, l), (l, k)):
                    T[p, q, r, s] = M[a, b] * f
    return T


def _unit(n, dim: int = 0):
    """Unit rows of n; ``dim`` 2: their xy part (a slab's in-plane directions)."""
    import numpy as np
    n = np.asarray(n, dtype=float)
    n = n.reshape(-1, n.shape[-1])
    if dim == 2 and n.shape[1] == 3:
        n = n[:, :2]
    return n / np.linalg.norm(n, axis=1, keepdims=True)


def youngs(St, n):
    import numpy as np
    n = _unit(n, St.shape[0])
    with np.errstate(divide="ignore"):
        return 1.0 / np.einsum("ijkl,ni,nj,nk,nl->n", St, n, n, n, n)


def compressibility(St, n):
    import numpy as np
    n = _unit(n, St.shape[0])
    return np.einsum("ijkk,ni,nj->n", St, n, n)


def shear(St, n, m):
    import numpy as np
    n, m = _unit(n, St.shape[0]), _unit(m, St.shape[0])
    with np.errstate(divide="ignore"):
        return 1.0 / (4.0 * np.einsum("ijkl,ni,nj,nk,nl->n", St, n, m, n, m))


def poisson(St, n, m):
    import numpy as np
    n, m = _unit(n, St.shape[0]), _unit(m, St.shape[0])
    return -youngs(St, n) * np.einsum("ijkl,ni,nj,nk,nl->n", St, n, n, m, m)


def perp_basis(n):
    """(u, v): an orthonormal pair perpendicular to each unit n (N x 3)."""
    import numpy as np
    n = _unit(n)
    ref = np.where(np.abs(n[:, 2:3]) < 0.9, np.array([[0.0, 0.0, 1.0]]), np.array([[1.0, 0.0, 0.0]]))
    u = np.cross(n, ref)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    return u, np.cross(n, u)


def _sym2(M, u, v):
    """The extremes of m.M.m over m = cos(chi) u + sin(chi) v: (low, high, m_low, m_high)."""
    import numpy as np
    a = np.einsum("nj,njl,nl->n", u, M, u)
    c = np.einsum("nj,njl,nl->n", v, M, v)
    b = 0.5 * (np.einsum("nj,njl,nl->n", u, M, v) + np.einsum("nj,njl,nl->n", v, M, u))
    mean, r = 0.5 * (a + c), np.sqrt((0.5 * (a - c)) ** 2 + b * b)
    chi = 0.5 * np.arctan2(2.0 * b, a - c)              # the m of the larger value
    m_hi = np.cos(chi)[:, None] * u + np.sin(chi)[:, None] * v
    m_lo = -np.sin(chi)[:, None] * u + np.cos(chi)[:, None] * v
    return mean - r, mean + r, m_lo, m_hi


def shear_poisson_extremes(St, n) -> Dict[str, Any]:
    """Per direction n: G_min, G_max, nu_min, nu_max over every m perpendicular to n, and those m.
    A slab (2D St): the one in-plane m, so min = max."""
    import numpy as np
    if St.shape[0] == 2:
        n = _unit(n, 2)
        m = np.stack([-n[:, 1], n[:, 0], np.zeros(len(n))], axis=1)
        g, nu = shear(St, n, m), poisson(St, n, m)
        return {"g_min": g, "g_max": g, "nu_min": nu, "nu_max": nu, "m_g_min": m, "m_g_max": m,
                "m_nu_min": m, "m_nu_max": m}
    n = _unit(n)
    u, v = perp_basis(n)
    Mg = np.einsum("njkl,nk->njl", np.tensordot(n, St, axes=([1], [0])), n)
    lo, hi, m_lo, m_hi = _sym2(Mg, u, v)
    with np.errstate(divide="ignore"):
        g_max, g_min = 1.0 / (4.0 * lo), 1.0 / (4.0 * hi)
    Mn = np.einsum("njkl,nj->nkl", np.tensordot(n, St, axes=([1], [0])), n)
    lo2, hi2, mn_lo, mn_hi = _sym2(Mn, u, v)
    E = youngs(St, n)
    return {"g_min": g_min, "g_max": g_max, "nu_min": -E * hi2, "nu_max": -E * lo2,
            "m_g_min": m_hi, "m_g_max": m_lo, "m_nu_min": mn_hi, "m_nu_max": mn_lo}


def christoffel(Ct, n, rho: float, group: bool = True) -> Dict[str, Any]:
    """Phase velocities (km/s, C in GPa and rho in g/cm^3) of the three acoustic waves along each n,
    fastest first (quasi-longitudinal, fast and slow quasi-shear), their polarizations, and the
    group velocity magnitudes."""
    import numpy as np
    n = _unit(n)
    B = np.tensordot(n, Ct, axes=([1], [1]))              # C_ijkl n_j: (N, i, k, l)
    G = np.einsum("nikl,nl->nik", B, n)
    lam, vec = np.linalg.eigh(G)
    lam, vec = lam[:, ::-1], vec[:, :, ::-1]
    v = np.sqrt(np.clip(lam, 0.0, None) / rho)
    out = {"phase": v, "polarization": vec}
    if group:
        vg = np.einsum("nikl,nim,nkm->nlm", B, vec, vec)
        with np.errstate(divide="ignore", invalid="ignore"):
            vg = vg / (rho * v[:, None, :])
        out["group_vector"] = vg
        out["group"] = np.linalg.norm(vg, axis=1)
    return out


# ------------------------------------------------------------------ directions
def sphere_grid(step_deg: float = 0.5, full: bool = False):
    """Unit directions on a theta/phi grid: the upper hemisphere (the quantities here are even in
    n), or the full sphere."""
    import numpy as np
    tmax = 180.0 if full else 90.0
    t, p = np.meshgrid(np.radians(np.arange(0.0, tmax + step_deg / 2, step_deg)),
                       np.radians(np.arange(0.0, 360.0, step_deg)), indexing="ij")
    return np.stack([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)], axis=-1).reshape(-1, 3)


def low_index(dim: int = 3, full: bool = False):
    """The [uvw] directions with |u|, |v|, |w| <= 3, normalized (the upper half unless ``full``)."""
    import numpy as np
    r = range(-3, 4)
    pts = set()
    for i in r:
        for j in r:
            for k in (r if dim == 3 else (0,)):
                if (i, j, k) == (0, 0, 0):
                    continue
                g = math.gcd(math.gcd(abs(i), abs(j)), abs(k))
                t = (i // g, j // g, k // g)
                if not full:
                    first = next(x for x in (t[2], t[1], t[0]) if x != 0) if dim == 3 else next(x for x in (t[1], t[0]) if x != 0)
                    if first < 0:
                        t = tuple(-x for x in t)
                pts.add(t)
    a = np.array(sorted(pts), dtype=float)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def plane_grid(step_deg: float = 0.25, full: bool = False):
    import numpy as np
    k = np.radians(np.arange(0.0, 360.0 if full else 180.0, step_deg))
    return np.stack([np.cos(k), np.sin(k), np.zeros_like(k)], axis=1)


def _refine(f, n0, dim: int):
    """A pattern search from n0 that lowers f (the extremes of a smooth function off the grid)."""
    import numpy as np
    if dim == 2:
        a = math.atan2(n0[1], n0[0])
        best = float(f(np.array([[math.cos(a), math.sin(a), 0.0]]))[0])
        step = math.radians(0.25)
        while step > 1e-6:
            moved = False
            for s in (step, -step):
                val = float(f(np.array([[math.cos(a + s), math.sin(a + s), 0.0]]))[0])
                if val < best - 1e-12 * max(1.0, abs(best)):
                    a, best, moved = a + s, val, True
                    break
            if not moved:
                step /= 2
        return np.array([math.cos(a), math.sin(a), 0.0]), best
    t, p = math.acos(max(-1.0, min(1.0, n0[2]))), math.atan2(n0[1], n0[0])
    vec = lambda t, p: np.array([[math.sin(t) * math.cos(p), math.sin(t) * math.sin(p), math.cos(t)]])   # noqa: E731
    best = float(f(vec(t, p))[0])
    step = math.radians(0.5)
    while step > 1e-6:
        moved = False
        for dt, dp in ((step, 0), (-step, 0), (0, step), (0, -step)):
            val = float(f(vec(t + dt, p + dp))[0])
            if val < best - 1e-12 * max(1.0, abs(best)):
                t, p, best, moved = t + dt, p + dp, val, True
                break
        if not moved:
            step /= 2
    return vec(t, p)[0], best


def is_cubic(data: Dict[str, Any]) -> bool:
    """A 3D cubic crystal: its conventional axes lie along x, y and z (CRYSTAL's standard orientation),
    so a Cartesian direction is the lattice direction with the same indices."""
    if (data.get("dimensionality") or 3) != 3:
        return False
    system = ((data.get("elastic") or {}).get("system") or data.get("crystal_family") or "").upper()
    return system == "CUBIC"


def direction_label(d: Sequence[float], tol: float = 1e-3, lattice: bool = False) -> str:
    """A unit direction (Cartesian) as text. When integers up to 3 give it within ``tol`` (a symmetry
    direction; the first nonzero component made positive): ``lattice`` (a cubic crystal, see is_cubic)
    the lattice direction [u v w], else the Cartesian combination (z, x+y, 2x−z). Any other direction
    (an extreme off the symmetry directions): its Cartesian components (x, y, z), never rounded to
    integers. Only a cubic crystal's Cartesian and lattice indices agree."""
    norm = math.sqrt(sum(float(x) ** 2 for x in d)) or 1.0
    d = [float(x) / norm for x in d]
    first = next((x for x in d if abs(x) > 1e-6), 1.0)
    if first < 0:
        d = [-x for x in d]
    big = max(abs(x) for x in d) or 1.0
    for k in range(1, 4):
        ints = [round(x / big * k) for x in d]
        norm = math.sqrt(sum(i * i for i in ints)) or 1.0
        if all(abs(i / norm - x) <= tol for i, x in zip(ints, d)):
            g = 0
            for i in ints:
                g = math.gcd(g, abs(i))
            ints = [i // (g or 1) for i in ints]
            if lattice:
                return "[" + " ".join(str(i).replace("-", "−") for i in ints) + "]"
            terms = []
            for i, ax in zip(ints, "xyz"):
                if i:
                    terms.append(("−" if i < 0 else "+" if terms else "") + ("" if abs(i) == 1 else str(abs(i))) + ax)
            return "".join(terms)
    return "(" + ", ".join(f"{x:.3f}".replace("-", "−") for x in d) + ")"


def _r(x, digits=10):
    return None if x is None or not math.isfinite(x) else float(f"{x:.{digits}g}")


def extremes(f, dim: int, full: bool = False, pair=None, lattice: bool = False) -> Dict[str, Any]:
    """The smallest and largest value of f (directions -> values) and where: a 0.5-degree grid plus the
    low-index directions, the best refined by a pattern search. ``pair(n)``: the second direction m
    that goes with the value at n, as {"min": m, "max": m}. ``lattice``: direction_label's."""
    import numpy as np
    low = low_index(dim, full)
    dirs = np.vstack([low, sphere_grid(1.0, full) if dim == 3 else plane_grid(0.5, full)])
    v = f(dirs)
    finite = np.isfinite(v)
    if not finite.any():
        return {}
    out: Dict[str, Any] = {}
    span = float(np.nanmax(np.abs(v[finite]))) or 1e-30
    for key, sign in (("min", 1.0), ("max", -1.0)):
        w = np.where(finite, sign * v, np.inf)
        i = int(np.argmin(w))
        same = np.flatnonzero(np.abs(w - w[i]) <= 1e-9 * span)
        # of the equivalent directions (a cubic crystal's eight <111>), the low-index one with the most positive components
        score = (10.0 * (dirs[same] > 1e-6).sum(axis=1) + dirs[same].sum(axis=1) + 1e-3 * dirs[same][:, 0]
                 - 100.0 * (same >= len(low)))
        i = int(same[int(np.argmax(score))])
        n, best = _refine(lambda d: sign * f(d), dirs[i], dim)
        if best > sign * v[i] - 1e-9 * span:              # the grid point is the extreme (a symmetry direction)
            n, best = dirs[i], sign * v[i]
        d = [round(float(x), 4) + 0.0 for x in n]
        e: Dict[str, Any] = {"value": _r(sign * best), "direction": d, "label": direction_label(d, lattice=lattice)}
        if pair is not None:
            m = pair(np.array([n]))[key][0]
            md = [round(float(x), 4) + 0.0 for x in m]
            e.update(direction2=md, label2=direction_label(md, lattice=lattice))
        out[key] = e
    lo, hi = out["min"]["value"], out["max"]["value"]
    out["isotropic"] = bool(lo is not None and hi is not None and abs(hi - lo) <= 1e-4 * max(abs(hi), abs(lo), 1e-30))
    out["ratio"] = _r(hi / lo) if lo and hi is not None and lo > 0 else None
    return out


# ------------------------------------------------------------------ the derived numbers
def compliance_of(data: Dict[str, Any]):
    """(S as four-index, C as four-index, dim, unit): the run's stiffness (GPa in 3D, the moduli's unit
    for a slab) and its inverse; None without constants."""
    import numpy as np
    el = data.get("elastic")
    if not el:
        return None
    dim = data.get("dimensionality") or 3
    C = np.array(el["matrix"], dtype=float)
    unit = el.get("unit") or "GPa"
    if dim == 3 and C.shape == (6, 6):
        C = (C + C.T) / 2.0
    elif dim == 2:
        mod = data.get("moduli") or {}
        if mod.get("c2d"):
            C, unit = np.array(mod["c2d"], dtype=float), mod.get("unit") or unit
        elif C.shape == (6, 6):
            C = C[np.ix_([0, 1, 5], [0, 1, 5])]
        if C.shape != (3, 3):
            return None
        C = (C + C.T) / 2.0
    else:
        return None
    S = np.linalg.inv(C)
    return voigt_to_tensor(S, True), voigt_to_tensor(C, False), (3 if len(C) == 6 else 2), unit, C, S


def anisotropy_2d(C, S) -> Dict[str, Any]:
    """Voigt and Reuss bounds of a slab's in-plane block and A^SU (Li et al. 2020)."""
    kv = (C[0, 0] + C[1, 1] + 2 * C[0, 1]) / 4.0
    gv = (C[0, 0] + C[1, 1] - 2 * C[0, 1] + 4 * C[2, 2]) / 8.0
    kr = 1.0 / (S[0, 0] + S[1, 1] + 2 * S[0, 1])
    gr = 2.0 / (S[0, 0] + S[1, 1] - 2 * S[0, 1] + S[2, 2])
    return {"area_modulus_voigt": _r(kv), "area_modulus_reuss": _r(kr), "shear_modulus_voigt": _r(gv),
            "shear_modulus_reuss": _r(gr), "anisotropy_su": _r(math.sqrt((kv / kr - 1) ** 2 + 2 * (gv / gr - 1) ** 2))}


def hardness(K: float, G: float) -> Dict[str, Any]:
    """Vickers hardness estimates in GPa from the Hill K and G (GPa)."""
    if not (K and G and K > 0 and G > 0):
        return {}
    k = G / K
    chen = 2.0 * (k * k * G) ** 0.585 - 3.0
    out = {"chen_gpa": _r(max(chen, 0.0)), "tian_gpa": _r(0.92 * k ** 1.137 * G ** 0.708)}
    if chen < 0:
        # a soft solid (small G or G/K) is outside the range of Chen's fit: its -3 GPa offset gives a
        # negative hardness, shown as 0 with the formula's value and not stored
        out.update(chen_clamped=True, chen_formula_gpa=_r(chen))
    return out


def min_thermal_conductivity(E_gpa: float, density_g_cm3: float, n_atoms: int, volume_ang3: float,
                             v_l_km_s: float, v_t_km_s: float) -> Dict[str, Any]:
    """Clarke's and Cahill's minimum thermal conductivity, W/(m K)."""
    rho = density_g_cm3 * 1000.0
    vol = volume_ang3 * 1e-30
    m_a = rho * vol / n_atoms                              # kg per atom
    n = n_atoms / vol
    return {"clarke_w_mk": _r(0.87 * K_B * m_a ** (-2.0 / 3.0) * math.sqrt(E_gpa * 1e9) * rho ** (1.0 / 6.0)),
            "cahill_w_mk": _r((math.pi / 6.0) ** (1.0 / 3.0) * K_B * n ** (2.0 / 3.0) * (v_l_km_s + 2 * v_t_km_s) * 1000.0),
            "mean_atomic_mass_amu": _r(m_a / AMU_KG)}


def cauchy(C, system: str) -> List[Dict[str, Any]]:
    """The Cauchy relations of C (Voigt, GPa): the Cauchy pressure C12 - C44 of a cubic crystal, else
    C12 - C66, C13 - C55, C23 - C44 (and the three off-diagonal ones when the system has them)."""
    c = lambda i, j: float(C[i - 1, j - 1])               # noqa: E731
    if system == "CUBIC":
        return [{"relation": "C12 − C44", "value": _r(c(1, 2) - c(4, 4))}]
    rows = [("C12 − C66", c(1, 2) - c(6, 6)), ("C13 − C55", c(1, 3) - c(5, 5)), ("C23 − C44", c(2, 3) - c(4, 4))]
    for (a, b), (x, y) in (((1, 4), (5, 6)), ((2, 5), (4, 6)), ((3, 6), (4, 5))):
        if abs(c(a, b)) > 1e-6 or abs(c(x, y)) > 1e-6:
            rows.append((f"C{a}{b} − C{x}{y}", c(a, b) - c(x, y)))
    return [{"relation": r, "value": _r(v)} for r, v in rows]


# ------------------------------------------------------------------ piezoelectric
def piezo_longitudinal(P, kind: str, n):
    """d_L(n) (kind 'd', Voigt d in pC/N) or e_L(n) (kind 'e', C/m^2) along unit directions n."""
    import numpy as np
    n = _unit(n)
    x, y, z = n[:, 0], n[:, 1], n[:, 2]
    f = 1.0 if kind == "d" else 2.0
    a = np.stack([x * x, y * y, z * z, f * y * z, f * x * z, f * x * y], axis=1)
    return np.einsum("ni,ij,nj->n", n, np.asarray(P, dtype=float), a)


def coupling(dL, E_gpa, epsL):
    """k^2 along n from d_L (pC/N), E(n) (GPa) and the relative permittivity along n."""
    import numpy as np
    return (np.asarray(dL) * 1e-12) ** 2 * np.asarray(E_gpa) * 1e9 / (EPS0 * np.asarray(epsL))


def epsilon_tensor(eps) -> Optional[List[List[float]]]:
    """A relative permittivity given as one number, three (diagonal) or nine values; None otherwise."""
    if eps is None:
        return None
    vals = [float(x) for x in (eps if isinstance(eps, (list, tuple)) else str(eps).replace(",", " ").split())]
    if len(vals) == 1:
        vals = vals * 3
    if len(vals) == 3:
        return [[vals[0], 0.0, 0.0], [0.0, vals[1], 0.0], [0.0, 0.0, vals[2]]]
    if len(vals) == 9:
        return [vals[0:3], vals[3:6], vals[6:9]]
    raise ValueError("epsilon is one value, three (xx yy zz) or nine")


def piezo(data: Dict[str, Any], St=None, epsilon=None) -> Dict[str, Any]:
    """The longitudinal piezoelectric coefficient's extremes and, with S and a permittivity, k^2."""
    import numpy as np
    P = data.get("piezo_d")
    kind = "d"
    if not P:
        P, kind = data.get("piezo_e"), "e"
    if not P or np.shape(P["matrix"]) != (3, 6):
        return {}
    M = np.array(P["matrix"], dtype=float)
    lat = is_cubic(data)
    out: Dict[str, Any] = {"kind": kind, "unit": "pC/N" if kind == "d" else "C/m^2",
                           "longitudinal": extremes(lambda d: np.abs(piezo_longitudinal(M, kind, d)), 3, full=False, lattice=lat)}
    if not out["longitudinal"] or not out["longitudinal"]["max"]["value"]:
        out["zero"] = True
        return out
    # the sign along the extreme direction (d_L is odd in n: d_L(-n) = -d_L(n))
    mx = out["longitudinal"]["max"]
    mx["signed"] = _r(float(piezo_longitudinal(M, kind, [mx["direction"]])[0]))
    out["axes"] = [_r(float(piezo_longitudinal(M, kind, [[1.0 if i == j else 0.0 for j in range(3)]])[0])) for i in range(3)]
    if kind != "d" or St is None:
        return out
    eps, source = None, None
    if epsilon is not None:
        eps, source = np.array(epsilon_tensor(epsilon)), "given"
    else:
        de = data.get("dielectric")
        if de and de.get("components"):
            m = np.zeros((3, 3))
            for r in de["components"]:
                if r["frequency_ev"] == de["components"][0]["frequency_ev"]:
                    i, j = "XYZ".index(r["component"][0]), "XYZ".index(r["component"][1])
                    m[i, j] = m[j, i] = r["epsilon"]
            eps, source = m, "electronic"
    if eps is None:
        out["coupling"] = {"available": False}
        return out

    def k2(d):
        d = _unit(d)
        return coupling(piezo_longitudinal(M, "d", d), youngs(St, d), np.einsum("ni,ij,nj->n", d, eps, d))
    ex = extremes(k2, 3, lattice=lat)
    out["coupling"] = {"available": True, "epsilon": eps.tolist(), "epsilon_source": source,
                       "k2_axes": [_r(float(k2([[1.0 if i == j else 0.0 for j in range(3)]])[0])) for i in range(3)],
                       "k2_max": ex.get("max"), "k_max": _r(math.sqrt(max(0.0, ex["max"]["value"]))) if ex.get("max") else None}
    return out


def derived(data: Dict[str, Any], epsilon=None) -> Dict[str, Any]:
    """Every number this module derives from the run's tensors ({} without them)."""
    import numpy as np
    out: Dict[str, Any] = {}
    got = compliance_of(data)
    if got is not None:
        St, Ct, dim, unit, C, S = got
        lat = is_cubic(data)
        b_unit, b_scale = (("TPa^-1", 1000.0) if dim == 3 else ("m/N" if unit == "N/m" else f"{unit}^-1", 1.0))
        sp = _memo(lambda d: shear_poisson_extremes(St, d))
        ext = {"youngs": {"unit": unit, **extremes(lambda d: youngs(St, d), dim, lattice=lat)},
               "compressibility": {"unit": b_unit, **extremes(lambda d: compressibility(St, d) * b_scale, dim, lattice=lat)}}
        ext["shear"] = {"unit": unit, **_pair_extremes(sp, "g", dim, lat)}
        ext["poisson"] = {"unit": "", **_pair_extremes(sp, "nu", dim, lat)}
        nu = ext["poisson"]
        if nu.get("min") and nu["min"]["value"] is not None:
            nu["ratio"] = _r(nu["max"]["value"] / nu["min"]["value"]) if nu["min"]["value"] > 0 else None
            nu["auxetic"] = bool(nu["min"]["value"] < 0)
        out["directional"] = ext
        mod = data.get("moduli") or {}
        if mod.get("born_stable") is False:
            # C is not positive definite: every number below is that of an unphysical tensor (E(n) of
            # -1e8 GPa and the like); shown flagged, never stored (properties.elastic_properties)
            out["unstable"] = True
        if dim == 3:
            rho = data.get("density_g_cm3")
            if rho:
                sound = {}
                ch = _memo(lambda d: christoffel(Ct, d, rho))
                for k, name in enumerate(("qL", "qS1", "qS2")):
                    sound[name] = {"unit": "km/s", **extremes(lambda d, k=k: ch(d)["phase"][:, k], 3, lattice=lat)}
                    sound[name]["group"] = {"unit": "km/s", **extremes(lambda d, k=k: ch(d)["group"][:, k], 3, lattice=lat)}
                ext["sound"] = sound
            idx: Dict[str, Any] = {}
            if "universal_anisotropy" in mod:
                idx["universal"] = _r(mod["universal_anisotropy"])
            system = ((data.get("elastic") or {}).get("system") or "").upper()
            if system == "CUBIC" and abs(C[0, 0] - C[0, 1]) > 1e-9:
                idx["zener"] = _r(2 * C[3, 3] / (C[0, 0] - C[0, 1]))
            gv, gr = mod.get("shear_modulus_voigt_gpa"), mod.get("shear_modulus_reuss_gpa")
            if gv and gr:
                idx["chung_buessem"] = _r((gv - gr) / (gv + gr))
            for k, q in (("e_ratio", "youngs"), ("g_ratio", "shear"), ("nu_ratio", "poisson")):
                idx[k] = ext[q].get("ratio")
            out["indices"] = idx
            K, G = mod.get("bulk_modulus_hill_gpa"), mod.get("shear_modulus_hill_gpa")
            ch: Dict[str, Any] = {"system": system.lower() or None}
            if K and G:
                ch.update(pugh_ratio=_r(K / G), ductile=bool(K / G > DUCTILE_PUGH), pugh_threshold=DUCTILE_PUGH)
            ch["cauchy"] = cauchy(C, system)
            ch["hardness"] = {**hardness(K, G), "references": HARDNESS_REFS} if K and G else {}
            out["character"] = ch
            th: Dict[str, Any] = {}
            if all(data.get(k) for k in ("density_g_cm3", "n_atoms", "volume_ang3")) and mod.get("youngs_modulus_gpa") \
                    and mod.get("sound_velocity_longitudinal_km_s"):
                th = min_thermal_conductivity(mod["youngs_modulus_gpa"], data["density_g_cm3"], data["n_atoms"],
                                              data["volume_ang3"], mod["sound_velocity_longitudinal_km_s"],
                                              mod["sound_velocity_transverse_km_s"])
                th["references"] = THERMAL_REFS
            if mod.get("debye_temperature_k"):
                th["debye_temperature_k"] = _r(mod["debye_temperature_k"])
            if th:
                out["thermal"] = th
            pz = piezo(data, St, epsilon)
            if pz:
                out["piezo"] = pz
        else:
            out["two_d"] = {**anisotropy_2d(C, S), "unit": unit, "e_ratio": ext["youngs"].get("ratio"),
                            "g_ratio": ext["shear"].get("ratio"), "nu_ratio": ext["poisson"].get("ratio")}
    elif data.get("piezo_e") or data.get("piezo_d"):
        pz = piezo(data, None, epsilon)
        if pz:
            out["piezo"] = pz
    return out


def _memo(fn):
    """fn of a direction array, kept for the last large array it was given (the grid every extreme
    of the same quantity starts from)."""
    last: Dict[str, Any] = {}

    def g(d):
        if len(d) < 64:
            return fn(d)
        key = (d.shape, hash(d.tobytes()))
        if last.get("key") != key:
            last.update(key=key, value=fn(d))
        return last["value"]
    return g


def _pair_extremes(sp, q: str, dim: int, lattice: bool = False) -> Dict[str, Any]:
    """The global min of q_min(n) and max of q_max(n), with n and the m that goes with each."""
    lo = extremes(lambda d: sp(d)[f"{q}_min"], dim, pair=lambda d: {"min": sp(d)[f"m_{q}_min"], "max": sp(d)[f"m_{q}_min"]},
                  lattice=lattice)
    hi = extremes(lambda d: sp(d)[f"{q}_max"], dim, pair=lambda d: {"min": sp(d)[f"m_{q}_max"], "max": sp(d)[f"m_{q}_max"]},
                  lattice=lattice)
    if not lo or not hi:
        return {}
    out = {"min": lo["min"], "max": hi["max"]}
    a, b = out["min"]["value"], out["max"]["value"]
    out["isotropic"] = bool(a is not None and b is not None and abs(b - a) <= 1e-4 * max(abs(a), abs(b), 1e-30))
    out["ratio"] = _r(b / a) if a and b is not None and a > 0 else None
    return out
