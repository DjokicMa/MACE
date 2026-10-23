"""Whole-pattern fitting of a measured powder pattern: Pawley and Rietveld.

Both fit y(2theta) = sum_f I_f PV(2theta; 2theta_f + zero, FWHM_f, eta)
(+ the K-alpha2 copy) + background, by weighted least squares
(w = 1/sigma^2), with

- the lattice through a symmetry-adapted strain of the starting cell: the
  cell is A (I + e) with e restricted to the strains the point group leaves
  invariant (cubic 1 parameter, tetragonal/hexagonal 2, orthorhombic 3,
  monoclinic 4, triclinic 6), so constraints hold in any setting;
- Caglioti widths (U, V, W), a pseudo-Voigt mixing eta, a zero shift and a
  Chebyshev background.

Pawley: every reflection family (an orbit of the Laue group) has its own
intensity, >= 0; systematic absences of the space group are left out when
a structure is given. Rietveld: I_f = scale * m_f * |F_f|^2 * LP * PO_f from
the structure, refining the scale, an overall B and, on request, the
atomic coordinates along the directions their site symmetry allows, and a
March-Dollase preferred orientation along a chosen [hkl].

The linear parameters (intensities or scale, and background) are solved
exactly at every step (variable projection), so the nonlinear search only
sees the cell, the zero, the profile and the structure parameters. Not
modeled: anomalous dispersion, peak asymmetry, sample displacement,
absorption, anisotropic broadening, occupancies and anisotropic ADPs.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import pxrd
from .structure import Structure


# ---------------------------------------------------------------------------
# symmetry helpers
# ---------------------------------------------------------------------------
def symmetry_ops(st: Structure, symprec: float = 1e-3) -> Tuple[np.ndarray, np.ndarray, Optional[Dict]]:
    """(rotations, translations, dataset) of the structure (fractional)."""
    import spglib
    ds = spglib.get_symmetry_dataset((st.cell, st.frac, st.numbers), symprec=symprec)
    if ds is None:
        return np.eye(3, dtype=int)[None], np.zeros((1, 3)), None
    rot = np.asarray(ds.rotations if hasattr(ds, "rotations") else ds["rotations"])
    tr = np.asarray(ds.translations if hasattr(ds, "translations") else ds["translations"])
    return rot, tr, ds


def lattice_ops(cell: np.ndarray, symprec: float = 1e-3) -> np.ndarray:
    """The point group of the lattice itself (the holohedry), fractional."""
    import spglib
    sym = spglib.get_symmetry((cell, [[0, 0, 0]], [1]), symprec=symprec)
    return np.asarray(sym["rotations"]) if sym else np.eye(3, dtype=int)[None]


def strain_basis(cell: np.ndarray, rotations: np.ndarray) -> np.ndarray:
    """Orthonormal basis (k, 3, 3) of the symmetric strains every rotation
    leaves invariant."""
    A = np.asarray(cell, float)
    Ainv_T = np.linalg.inv(A).T
    Rc = [A.T @ R @ Ainv_T for R in rotations]
    basis = []
    for i in range(3):
        for j in range(i, 3):
            E = np.zeros((3, 3))
            E[i, j] = E[j, i] = 1.0
            P = sum(r @ E @ r.T for r in Rc) / len(Rc)
            basis.append(P[np.triu_indices(3)])
    M = np.array(basis).T                 # (6, 6)
    u, s, vt = np.linalg.svd(M)
    k = int((s > 1e-8 * max(s.max(), 1)).sum())
    out = []
    for c in range(k):
        v = u[:, c]
        E = np.zeros((3, 3))
        E[np.triu_indices(3)] = v
        E = E + E.T - np.diag(np.diag(E))
        out.append(E / np.linalg.norm(E))
    return np.array(out)


def strained(cell: np.ndarray, basis: np.ndarray, coef: np.ndarray) -> np.ndarray:
    e = np.tensordot(coef, basis, axes=1) if len(coef) else np.zeros((3, 3))
    return np.asarray(cell, float) @ (np.eye(3) + e)


def crystal_system(n_strain: int, rotations: np.ndarray) -> str:
    n = len({tuple(np.asarray(r).ravel()) for r in rotations})
    return {1: "cubic", 2: "tetragonal/hexagonal", 3: "orthorhombic", 4: "monoclinic", 6: "triclinic"}.get(
        n_strain, f"{n_strain} cell parameters") if n else "triclinic"


def absent(H: np.ndarray, rotations: np.ndarray, translations: np.ndarray) -> np.ndarray:
    """Systematic absences: h with h R = h and h.t not an integer for some op."""
    out = np.zeros(len(H), bool)
    for R, t in zip(rotations, translations):
        fixed = np.all(H @ R == H, axis=1)
        ph = H @ t
        out |= fixed & (np.abs(ph - np.round(ph)) > 1e-3)
    return out


def families(H: np.ndarray, rotations: np.ndarray) -> np.ndarray:
    """A family id per reflection: its orbit under the Laue group."""
    rots = list(rotations) + [-np.asarray(r) for r in rotations]
    key = {}
    fam = np.empty(len(H), int)
    for n, h in enumerate(H):
        orbit = sorted({tuple(int(v) for v in h @ R) for R in rots})
        k = orbit[-1]
        if k not in key:
            key[k] = len(key)
        fam[n] = key[k]
    return fam


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------
class _Positions:
    """Atomic coordinates as free parameters along the directions each site
    symmetry allows; symmetry-equivalent atoms move together."""

    def __init__(self, st: Structure, rotations, translations, dataset):
        self.x0 = st.frac.copy()
        n = st.n
        if dataset is None:
            eq = np.arange(n)
        else:
            eq = np.asarray(dataset.equivalent_atoms if hasattr(dataset, "equivalent_atoms") else dataset["equivalent_atoms"])
        self.blocks = []        # (rep, basis (3, k))
        self.map = []           # per atom: (block index, R) with delta_j = R delta_rep
        reps = sorted(set(int(e) for e in eq))
        bidx = {}
        for r in reps:
            site = [R for R, t in zip(rotations, translations)
                    if np.allclose(((R @ self.x0[r] + t - self.x0[r]) + 0.5) % 1 - 0.5, 0, atol=1e-3)]
            P = sum(np.asarray(R, float) for R in site) / max(len(site), 1)
            u, s, vt = np.linalg.svd(P)
            B = u[:, s > 0.5]
            bidx[r] = len(self.blocks)
            self.blocks.append((r, B))
        for j in range(n):
            r = int(eq[j])
            Rj = np.eye(3)
            for R, t in zip(rotations, translations):
                if np.allclose(((R @ self.x0[r] + t - self.x0[j]) + 0.5) % 1 - 0.5, 0, atol=1e-3):
                    Rj = np.asarray(R, float)
                    break
            self.map.append((bidx[r], Rj))
        self.n = sum(B.shape[1] for _, B in self.blocks)

    def apply(self, p: np.ndarray) -> np.ndarray:
        deltas = []
        k = 0
        for r, B in self.blocks:
            deltas.append(B @ p[k:k + B.shape[1]])
            k += B.shape[1]
        out = self.x0.copy()
        for j, (b, R) in enumerate(self.map):
            out[j] = self.x0[j] + R @ deltas[b]
        return out


def _cheb(x: np.ndarray, n: int) -> np.ndarray:
    t = 2 * (x - x.min()) / max(x.max() - x.min(), 1e-12) - 1
    return np.polynomial.chebyshev.chebvander(t, max(n - 1, 0))


def _stats(y, yc, w, n_par):
    r = y - yc
    rp = float(np.abs(r).sum() / max(np.abs(y).sum(), 1e-30))
    rwp = float(np.sqrt((w * r * r).sum() / max((w * y * y).sum(), 1e-30)))
    rexp = float(np.sqrt(max(len(y) - n_par, 1) / max((w * y * y).sum(), 1e-30)))
    chi2 = float((w * r * r).sum() / max(len(y) - n_par, 1))
    return {"Rp": rp, "Rwp": rwp, "Rexp": rexp, "chi2": chi2, "GOF": rwp / rexp if rexp > 0 else float("nan")}


def fit(pattern: Dict, method: str = "pawley", structure: Optional[Structure] = None,
        cell: Optional[np.ndarray] = None, wavelength="CuKa", two_theta_range: Optional[Tuple[float, float]] = None,
        exclude: Sequence[Tuple[float, float]] = (), n_background: int = 6, refine_positions: bool = False,
        preferred_orientation: Optional[Sequence[int]] = None, lattice_symmetry: Optional[str] = None,
        max_nfev: int = 400, progress: Optional[Callable[[float, str], None]] = None,
        check: Optional[Callable[[], None]] = None) -> Dict:
    """Pawley or Rietveld fit; see the module docstring."""
    from scipy.optimize import least_squares, lsq_linear
    say = progress or (lambda f, m="": None)
    chk = check or (lambda: None)
    method = method.lower()
    if method not in ("pawley", "rietveld"):
        raise ValueError("method is pawley or rietveld")
    l1, l2, ratio, label = pxrd.source(wavelength)
    x = np.asarray(pattern["two_theta"], float)
    y = np.asarray(pattern["intensity"], float)
    sig = np.asarray(pattern.get("sigma") if pattern.get("sigma") is not None else np.sqrt(np.clip(y, 1, None)), float)
    m = np.ones(len(x), bool)
    if two_theta_range:
        m &= (x >= two_theta_range[0]) & (x <= two_theta_range[1])
    for lo, hi in exclude:
        m &= ~((x >= lo) & (x <= hi))
    x, y, sig = x[m], y[m], sig[m]
    if len(x) < 50:
        raise ValueError("fewer than 50 points to fit")
    w = 1.0 / np.clip(sig, 1e-6, None) ** 2

    # the starting structure / cell and its symmetry
    if structure is not None:
        st = pxrd.conventional(structure)
        rot, tr, ds = symmetry_ops(st)
        cell0 = st.cell.copy()
    else:
        if method == "rietveld":
            raise ValueError("Rietveld needs a structure")
        if cell is None:
            raise ValueError("Pawley needs a starting cell or a structure")
        cell0 = np.asarray(cell, float)
        st, ds = None, None
        rot, tr = lattice_ops(cell0), np.zeros((1, 3))
    if lattice_symmetry == "triclinic":
        lat_rot = np.eye(3, dtype=int)[None]
    else:
        lat_rot = rot
    basis = strain_basis(cell0, lat_rot)
    pos = _Positions(st, rot, tr, ds) if (method == "rietveld" and refine_positions) else None

    # reflections from the starting cell, a little past the range so strain cannot push new ones in
    tmax = min(float(x.max()) + 3.0, 179.0)
    d_min = l1 / (2 * math.sin(math.radians(tmax) / 2)) * 0.97
    H, _ = pxrd.hkl_list(cell0, d_min)
    if structure is not None:
        H = H[~absent(H, rot, tr)]
    fam = families(H, rot)
    nf = int(fam.max()) + 1 if len(fam) else 0
    rep = np.zeros((nf, 3), int)
    mult = np.bincount(fam, minlength=nf)
    for i in range(len(H)):
        rep[fam[i]] = H[i]
    po_dir = np.asarray(preferred_orientation, float) if preferred_orientation is not None else None

    # starting widths from the measured peaks
    try:
        pk = pxrd.find_peaks(x, y, prominence=0.05)["peaks"]
        fw = np.median([p["fwhm"] for p in pk[:10] if p.get("fwhm")]) if pk else 0.1
    except Exception:
        fw = 0.1
    W0 = float(np.clip(fw, 0.02, 1.0)) ** 2
    bgX = _cheb(x, n_background)

    # parameter vector: [strain (k), zero, U, V, W, eta, (B), (positions), (po r)]
    k = len(basis)
    names = [f"strain{i}" for i in range(k)] + ["zero", "U", "V", "W", "eta"]
    p0 = [0.0] * k + [0.0, 0.0, 0.0, W0, 0.5]
    lo_b = [-0.05] * k + [-0.5, -0.1, -0.1, 1e-5, 0.0]
    hi_b = [0.05] * k + [0.5, 0.5, 0.1, 1.0, 1.0]
    if method == "rietveld":
        names.append("B")
        p0.append(0.5)
        lo_b.append(-1.0)
        hi_b.append(8.0)
        if pos is not None:
            names += [f"xyz{i}" for i in range(pos.n)]
            p0 += [0.0] * pos.n
            lo_b += [-0.05] * pos.n
            hi_b += [0.05] * pos.n
        if po_dir is not None:
            names.append("po_r")
            p0.append(1.0)
            lo_b.append(0.3)
            hi_b.append(3.0)
    p0, lo_b, hi_b = np.array(p0), np.array(lo_b), np.array(hi_b)
    lin = {"x": None}

    def unpack(p):
        q = dict(zip(names, p))
        coef = np.array([q[f"strain{i}"] for i in range(k)])
        return q, strained(cell0, basis, coef)

    def columns(p):
        q, A = unpack(p)
        rec = np.linalg.inv(A).T
        g = np.linalg.norm(rep @ rec, axis=1)
        d = 1.0 / g
        s1 = l1 / (2 * d)
        ok = s1 < 1
        tt = np.full(nf, np.nan)
        tt[ok] = np.degrees(2 * np.arcsin(s1[ok]))
        f = pxrd.fwhm_caglioti(np.nan_to_num(tt, nan=90.0), q["U"], q["V"], q["W"], None, l1)
        eta = q["eta"]
        inwin = ok & (tt > x.min() - 3 * f) & (tt < x.max() + 3 * f)
        cols = np.zeros((len(x), nf))
        lp = pxrd.lorentz_polarization(np.nan_to_num(tt, nan=90.0))
        for j in np.flatnonzero(inwin):
            c = tt[j] + q["zero"]
            sl = np.abs(x - c) < 15 * f[j]
            cols[sl, j] = pxrd.pseudo_voigt(x[sl], c, f[j], eta)
            if l2 and ratio > 0:
                s2 = l2 / (2 * d[j])
                if s2 < 1:
                    c2 = math.degrees(2 * math.asin(s2)) + q["zero"]
                    sl2 = np.abs(x - c2) < 15 * f[j]
                    cols[sl2, j] += ratio * pxrd.pseudo_voigt(x[sl2], c2, f[j], eta)
        return q, A, tt, d, lp, inwin, cols

    def model(p):
        q, A, tt, d, lp, inwin, cols = columns(p)
        sw = np.sqrt(w)
        if method == "pawley":
            use = np.flatnonzero(inwin & (cols.max(0) > 0))
            M = np.hstack([cols[:, use], bgX])
            lb = np.r_[np.zeros(len(use)), np.full(bgX.shape[1], -np.inf)]
            r = lsq_linear(M * sw[:, None], y * sw, bounds=(lb, np.inf), lsmr_tol="auto", method="bvls" if M.shape[1] < 400 else "trf")
            I = np.zeros(nf)
            I[use] = r.x[:len(use)]
            bgc = r.x[len(use):]
            yc = cols @ I + bgX @ bgc
            return yc, bgX @ bgc, {"I": I, "tt": tt, "d": d, "A": A, "q": q}
        # Rietveld
        stp = Structure(A, pos.apply(np.array([q[f"xyz{i}"] for i in range(pos.n)])) if pos else st.frac, st.numbers)
        F2 = pxrd.structure_factors(stp, rep, d, b_iso=q["B"])
        I = mult * F2 * lp
        if po_dir is not None:
            r_ = q["po_r"]
            rec = np.linalg.inv(A).T
            gv = rep @ rec
            gd = po_dir @ rec
            # March-Dollase, averaged over the family members (one representative here)
            cosa = np.abs(gv @ gd) / np.linalg.norm(gv, axis=1) / max(np.linalg.norm(gd), 1e-12)
            sin2 = 1 - cosa ** 2
            I = I * (r_ ** 2 * cosa ** 2 + sin2 / r_) ** -1.5
        I[~inwin] = 0
        prof = cols @ I
        M = np.column_stack([prof, bgX])
        lb = np.r_[0.0, np.full(bgX.shape[1], -np.inf)]
        r = lsq_linear(M * sw[:, None], y * sw, bounds=(lb, np.inf), method="bvls")
        scale, bgc = r.x[0], r.x[1:]
        yc = scale * prof + bgX @ bgc
        return yc, bgX @ bgc, {"I": scale * I, "tt": tt, "d": d, "A": A, "q": q, "scale": scale, "F2": F2,
                                "frac": stp.frac}

    nfev = {"n": 0}

    def resid(p):
        chk()
        nfev["n"] += 1
        if nfev["n"] % 10 == 0:
            say(min(0.95, 0.1 + 0.85 * nfev["n"] / max_nfev), f"{method}: {nfev['n']} evaluations")
        yc, _, _ = model(p)
        return (y - yc) * np.sqrt(w)

    # stage 1: zero, cell and W; stage 2: everything
    say(0.05, f"{method}: cell and zero")
    free1 = np.array([n.startswith("strain") or n in ("zero", "W", "scale", "B") for n in names])

    def sub(fr, base):
        def f(v):
            p = base.copy()
            p[fr] = v
            return resid(p)
        return f

    r1 = least_squares(sub(free1, p0), p0[free1], bounds=(lo_b[free1], hi_b[free1]), x_scale="jac",
                       max_nfev=max_nfev // 3, diff_step=1e-4)
    p1 = p0.copy()
    p1[free1] = r1.x
    say(0.4, f"{method}: all parameters")
    r2 = least_squares(resid, p1, bounds=(lo_b, hi_b), x_scale="jac", max_nfev=max_nfev, diff_step=1e-4)
    p = r2.x
    yc, bg, info = model(p)
    n_lin = (int((info["I"] > 0).sum()) if method == "pawley" else 1) + bgX.shape[1]
    stats = _stats(y, yc, w, len(p) + n_lin)
    # esds from the Jacobian of the nonlinear problem
    esd = np.full(len(p), np.nan)
    try:
        J = r2.jac
        cov = np.linalg.pinv(J.T @ J) * stats["chi2"]
        esd = np.sqrt(np.clip(np.diag(cov), 0, None))
    except Exception:
        cov = None
    A = info["A"]
    lat = Structure(A, np.zeros((1, 3)), [1]).lattice()
    lat["volume"] = float(abs(np.linalg.det(A)))
    lat_esd = {}
    if cov is not None and k:
        base = np.array(list(lat.values()))
        Jl = []
        for i in range(k):
            dp = p.copy()
            h = 1e-6
            dp[i] += h
            _, A2 = unpack(dp)
            l2_ = Structure(A2, np.zeros((1, 3)), [1]).lattice()
            l2_["volume"] = float(abs(np.linalg.det(A2)))
            Jl.append((np.array(list(l2_.values())) - base) / h)
        Jl = np.array(Jl).T
        C = cov[:k, :k]
        lat_esd = dict(zip(lat.keys(), np.sqrt(np.clip(np.diag(Jl @ C @ Jl.T), 0, None)).tolist()))
    q = info["q"]
    ticks = []
    for j in np.argsort(info["tt"]):
        if np.isfinite(info["tt"][j]) and x.min() <= info["tt"][j] + q["zero"] <= x.max():
            ticks.append({"two_theta": float(info["tt"][j] + q["zero"]), "hkl": rep[j].tolist(),
                          "d": float(info["d"][j]), "intensity": float(info["I"][j]), "multiplicity": int(mult[j])})
    out = {
        "method": method, "source": label, "wavelength": l1, "converged": bool(r2.success),
        "message": str(r2.message), "nfev": nfev["n"],
        "lattice": lat, "lattice_esd": lat_esd, "cell": A.tolist(), "cell_start": cell0.tolist(),
        "n_cell_parameters": k, "zero": float(q["zero"]),
        "profile": {"U": float(q["U"]), "V": float(q["V"]), "W": float(q["W"]), "eta": float(q["eta"])},
        "parameters": {n: {"value": float(v), "esd": float(e)} for n, v, e in zip(names, p, esd)},
        "statistics": stats,
        "two_theta": x.tolist(), "observed": y.tolist(), "calculated": yc.tolist(), "background": bg.tolist(),
        "difference": (y - yc).tolist(), "reflections": ticks,
    }
    if method == "rietveld":
        out["scale"] = float(info["scale"])
        out["B"] = float(q["B"])
        if pos is not None:
            out["frac"] = info["frac"].tolist()
            out["symbols"] = st.symbols
        if po_dir is not None:
            out["preferred_orientation"] = {"hkl": list(preferred_orientation), "r": float(q["po_r"])}
    say(1.0, "done")
    return out


def synthetic_pattern(st: Structure, wavelength="CuKa", two_theta_range=(10, 100), step=0.02, U=0.004, V=-0.002,
                      W=0.004, eta=0.4, b_iso=0.5, zero=0.0, background=50.0, counts=20000.0, noise=True,
                      seed: int = 0) -> Dict:
    """A measured-like pattern of a structure (Poisson noise), for testing a fit."""
    l1, l2, ratio, _ = pxrd.source(wavelength)
    peaks = pxrd.reflections(st, wavelength, (two_theta_range[0] - 2, two_theta_range[1] + 2), b_iso=b_iso,
                             min_rel=1e-5)
    x = np.arange(two_theta_range[0], two_theta_range[1] + step / 2, step)
    y = pxrd.profile(peaks, x, wavelength, U, V, W, eta, zero=zero)
    y = y / max(y.max(), 1e-12) * counts + background * (1 + 0.3 * np.cos(np.radians(x)))
    if noise:
        y = np.random.default_rng(seed).poisson(np.clip(y, 0, None)).astype(float)
    return {"two_theta": x, "intensity": y, "sigma": np.sqrt(np.clip(y, 1, None)), "name": "synthetic"}
