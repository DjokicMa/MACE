"""Powder X-ray diffraction: the pattern of a crystal structure, and reading
and analyzing measured patterns.

Simulation (kinematic, Bragg-Brentano by default):

    I(hkl) = |F(hkl)|^2 * LP(2theta) * exp(-2 B s^2)     s = sin(theta)/lambda = 1/(2d)
    F(hkl) = sum_j f_j(s) * occ_j * exp(2 pi i h.x_j)
    LP     = (1 + cos^2(2theta) cos^2(2theta_m)) / (sin^2(theta) cos(theta) (1 + cos^2(2theta_m)))

with Waasmaier-Kirfel atomic form factors (_xray_data.py; no anomalous
dispersion). Every reflection inside the 2theta range is enumerated, so the
multiplicity is implicit; reflections at the same 2theta merge into one
peak (equivalent planes, and accidental overlaps such as Si 333/511, as an
instrument sees them). Profiles are pseudo-Voigt with the Caglioti width
FWHM^2 = U tan^2(theta) + V tan(theta) + W (plus Scherrer size broadening
when a crystallite size is given), with the K-alpha2 line where the source
has one.

Measured patterns: two- or three-column text (2theta, intensity[, sigma];
.xy .xye .dat .txt .csv, comments and headers skipped) and PANalytical
.xrdml. Peak search: SNIP background, Savitzky-Golay smoothing, scipy
find_peaks, then a pseudo-Voigt fit of each peak.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .structure import Structure

# (K-alpha1, K-alpha2, I(K-alpha2)/I(K-alpha1)); Angstrom (Hoelzer et al. 1997, ITC Vol. C)
SOURCES: Dict[str, Tuple[float, Optional[float], float]] = {
    "CuKa": (1.540598, 1.544426, 0.5),
    "CuKa1": (1.540598, None, 0.0),
    "MoKa": (0.709319, 0.713609, 0.5),
    "MoKa1": (0.709319, None, 0.0),
    "CoKa": (1.788996, 1.792835, 0.5),
    "CrKa": (2.289760, 2.293663, 0.5),
    "FeKa": (1.936042, 1.939980, 0.5),
    "AgKa": (0.559421, 0.563813, 0.5),
}
DEFAULT_SOURCE = "CuKa"


def source(wavelength) -> Tuple[float, Optional[float], float, str]:
    """A source name (CuKa, MoKa1, ...) or a wavelength in Angstrom ->
    (lambda1, lambda2 or None, ratio, label)."""
    if wavelength is None:
        wavelength = DEFAULT_SOURCE
    if isinstance(wavelength, (int, float)):
        return float(wavelength), None, 0.0, f"{float(wavelength):g} A"
    key = {k.lower(): k for k in SOURCES}.get(str(wavelength).strip().lower().replace("-", "").replace("alpha", "a"))
    if key:
        l1, l2, r = SOURCES[key]
        return l1, l2, r, key
    try:
        v = float(wavelength)
    except ValueError:
        raise ValueError(f"unknown source {wavelength!r}: one of {', '.join(SOURCES)} or a wavelength in Angstrom") from None
    if not 0.05 < v < 5:
        raise ValueError("wavelength must be between 0.05 and 5 Angstrom")
    return v, None, 0.0, f"{v:g} A"


def form_factor(symbols: Sequence[str], s: np.ndarray) -> np.ndarray:
    """f0 for each symbol at each s = sin(theta)/lambda: shape (n_atoms, n_s)."""
    from ._xray_data import WK_F0
    s2 = np.asarray(s, float) ** 2
    out = np.empty((len(symbols), len(s2)))
    cache = {}
    for i, sym in enumerate(symbols):
        if sym not in cache:
            c = WK_F0.get(sym)
            if c is None:
                raise ValueError(f"no X-ray form factor for {sym}")
            a, c0, b = np.array(c[:5]), c[5], np.array(c[6:])
            cache[sym] = (a[:, None] * np.exp(-b[:, None] * s2[None, :])).sum(0) + c0
        out[i] = cache[sym]
    return out


def hkl_list(cell: np.ndarray, d_min: float) -> Tuple[np.ndarray, np.ndarray]:
    """Every (h, k, l) != 0 with d >= d_min, and d."""
    rec = np.linalg.inv(cell).T          # rows: reciprocal vectors without 2 pi
    # |h_i| = |g . a_i| <= |g| |a_i| = |a_i| / d
    hmax = [int(math.floor(np.linalg.norm(cell[i]) / d_min)) + 1 for i in range(3)]
    rng = [np.arange(-m, m + 1) for m in hmax]
    H = np.stack(np.meshgrid(*rng, indexing="ij"), -1).reshape(-1, 3)
    H = H[np.any(H != 0, axis=1)]
    g = H @ rec
    q = np.linalg.norm(g, axis=1)
    keep = q <= 1.0 / d_min + 1e-12
    return H[keep], 1.0 / q[keep]


def lorentz_polarization(two_theta_deg: np.ndarray, mono_2theta_deg: float = 0.0) -> np.ndarray:
    tt = np.radians(two_theta_deg)
    th = tt / 2
    cm = math.cos(math.radians(mono_2theta_deg)) ** 2
    return (1 + np.cos(tt) ** 2 * cm) / (np.sin(th) ** 2 * np.cos(th) * (1 + cm))


def structure_factors(st: Structure, H: np.ndarray, d: np.ndarray, b_iso=0.0,
                      occupancy: Optional[np.ndarray] = None) -> np.ndarray:
    """|F|^2 for each reflection."""
    s = 1.0 / (2 * d)
    syms = st.symbols
    uniq = sorted(set(syms))
    fz = form_factor(uniq, s)
    idx = {u: i for i, u in enumerate(uniq)}
    f = fz[[idx[x] for x in syms]]                  # (n_atoms, n_refl)
    if isinstance(b_iso, dict):
        B = np.array([float(b_iso.get(x, b_iso.get("*", 0.0))) for x in syms])
    else:
        B = np.full(len(syms), float(b_iso))
    f = f * np.exp(-B[:, None] * s[None, :] ** 2)
    if occupancy is not None:
        f = f * np.asarray(occupancy, float)[:, None]
    phase = 2 * np.pi * (st.frac @ H.T)             # (n_atoms, n_refl)
    F = (f * np.exp(1j * phase)).sum(0)
    return (F * F.conj()).real


def _group(two_theta: np.ndarray, tol: float = 1e-4) -> List[np.ndarray]:
    order = np.argsort(two_theta)
    groups, cur = [], [order[0]] if len(order) else []
    for a, b in zip(order[:-1], order[1:]):
        if two_theta[b] - two_theta[a] <= tol:
            cur.append(b)
        else:
            groups.append(np.array(cur))
            cur = [b]
    if cur:
        groups.append(np.array(cur))
    return groups


def conventional(st: Structure, symprec: float = 1e-3) -> Structure:
    """The spglib standard conventional cell (same crystal), so peaks carry
    the usual Miller indices; the structure itself when spglib finds nothing."""
    try:
        import spglib
        r = spglib.standardize_cell((st.cell, st.frac, st.numbers), to_primitive=False, no_idealize=True,
                                    symprec=symprec)
    except Exception:
        r = None
    if not r:
        return st
    cell, frac, nums = r
    return Structure(cell, frac, nums, st.pbc, st.name, dict(st.meta))


def reflections(st: Structure, wavelength=DEFAULT_SOURCE, two_theta_range=(5.0, 90.0), b_iso=0.0,
                mono_2theta: float = 0.0, min_rel: float = 1e-3, lam: Optional[float] = None,
                standardize: bool = True) -> List[Dict]:
    """The peaks of a structure at the K-alpha1 (or the single) wavelength:
    [{two_theta, d, intensity (max 100), hkl, families, multiplicity, F2}].
    With ``standardize`` the Miller indices refer to the conventional cell."""
    st.require_3d("Powder diffraction")
    if standardize:
        st = conventional(st)
    l1 = lam if lam is not None else source(wavelength)[0]
    lo, hi = two_theta_range
    hi = min(float(hi), 179.0)
    d_min = l1 / (2 * math.sin(math.radians(hi) / 2))
    H, d = hkl_list(st.cell, d_min)
    sin_t = l1 / (2 * d)
    ok = sin_t < 1
    H, d, sin_t = H[ok], d[ok], sin_t[ok]
    tt = np.degrees(2 * np.arcsin(sin_t))
    ok = (tt >= lo) & (tt <= hi)
    H, d, tt = H[ok], d[ok], tt[ok]
    if not len(H):
        return []
    F2 = structure_factors(st, H, d, b_iso)
    I = F2 * lorentz_polarization(tt, mono_2theta)
    out = []
    for g in _group(tt):
        tot = float(I[g].sum())
        hk = H[g]
        rep = tuple(int(x) for x in max(map(tuple, hk.tolist())))
        fams = sorted({tuple(sorted(np.abs(x).tolist(), reverse=True)) for x in hk.tolist()}, reverse=True)
        out.append({"two_theta": float(tt[g].mean()), "d": float(d[g].mean()), "raw": tot, "hkl": list(rep),
                    "families": [list(f) for f in fams], "multiplicity": int(len(g)),
                    "F2": float(F2[g].mean())})
    top = max(p["raw"] for p in out) or 1.0
    res = []
    for p in out:
        p["intensity"] = 100.0 * p.pop("raw") / top
        if p["intensity"] >= 100.0 * min_rel:
            res.append(p)
    return res


# ---------------------------------------------------------------------------
# profiles
# ---------------------------------------------------------------------------
def fwhm_caglioti(two_theta, U, V, W, size_nm=None, lam=1.5406):
    th = np.radians(np.asarray(two_theta, float)) / 2
    t = np.tan(th)
    f2 = np.clip(U * t * t + V * t + W, 1e-8, None)
    f = np.sqrt(f2)
    if size_nm:
        # Scherrer, K = 0.9: beta (rad) = K lambda / (L cos theta)
        f = f + np.degrees(0.9 * lam / (size_nm * 10.0 * np.cos(th)))
    return f


def pseudo_voigt(x: np.ndarray, x0, fwhm, eta) -> np.ndarray:
    """Area-normalized pseudo-Voigt (unit integral)."""
    g = fwhm / 2
    dx = x - x0
    lor = (1 / np.pi) * g / (dx * dx + g * g)
    sig = fwhm / (2 * math.sqrt(2 * math.log(2)))
    gau = np.exp(-dx * dx / (2 * sig * sig)) / (sig * math.sqrt(2 * np.pi))
    return eta * lor + (1 - eta) * gau


def profile(peaks: Sequence[Dict], two_theta: np.ndarray, wavelength=DEFAULT_SOURCE, U=0.0, V=0.0, W=0.01,
            eta=0.5, size_nm=None, include_ka2=True, zero: float = 0.0) -> np.ndarray:
    """A continuous pattern (max 100) from a reflections() list; ``zero``
    shifts every line (an instrument's zero error)."""
    l1, l2, ratio, _ = source(wavelength)
    x = np.asarray(two_theta, float)
    y = np.zeros_like(x)
    for p in peaks:
        f = float(fwhm_caglioti(p["two_theta"], U, V, W, size_nm, l1))
        win = 12 * f
        sl = (x > p["two_theta"] - win - 1.5) & (x < p["two_theta"] + win + 1.5)
        y[sl] += p["intensity"] * pseudo_voigt(x[sl], p["two_theta"] + zero, f, eta)
        if include_ka2 and l2 and ratio > 0:
            s2 = l2 / (2 * p["d"])
            if s2 < 1:
                t2 = math.degrees(2 * math.asin(s2))
                # its own window: at high angle Kα2 lies degrees above Kα1 (Ag Kα near 150°: about 4.7°)
                sl = (x > t2 - win - 1.5) & (x < t2 + win + 1.5)
                y[sl] += ratio * p["intensity"] * pseudo_voigt(x[sl], t2 + zero, f, eta)
    m = y.max()
    return y * (100.0 / m) if m > 0 else y


def simulate(st: Structure, wavelength=DEFAULT_SOURCE, two_theta_range=(5.0, 90.0), step=0.02, b_iso=0.0,
             U=0.0, V=0.0, W=0.01, eta=0.5, size_nm=None, mono_2theta=0.0, min_rel=1e-3) -> Dict:
    """Stick list and profile of a structure."""
    l1, l2, ratio, label = source(wavelength)
    peaks = reflections(st, wavelength, two_theta_range, b_iso, mono_2theta, min_rel)
    x = np.arange(two_theta_range[0], two_theta_range[1] + step / 2, step)
    y = profile(peaks, x, wavelength, U, V, W, eta, size_nm)
    return {"source": label, "wavelength": l1, "wavelength2": l2, "ka2_ratio": ratio,
            "range": list(two_theta_range), "step": step, "peaks": peaks,
            "two_theta": x.tolist(), "intensity": y.tolist(),
            "profile": {"U": U, "V": V, "W": W, "eta": eta, "size_nm": size_nm}, "b_iso": b_iso,
            "formula": st.formula(), "lattice": st.lattice()}


# ---------------------------------------------------------------------------
# measured patterns
# ---------------------------------------------------------------------------
_NUM = re.compile(r"^[\s]*[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?([\s,;]+[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?)+\s*$")


def read_pattern_text(text: str, name: str = "pattern") -> Dict:
    """Two/three-column text or an .xrdml document -> {two_theta, intensity, sigma}."""
    if "<xrdMeasurements" in text[:4000] or name.lower().endswith(".xrdml"):
        return _read_xrdml(text)
    xs, ys, ss = [], [], []
    for line in text.splitlines():
        t = line.strip()
        if not t or t[0] in "#!;%'\"" or not _NUM.match(t):
            continue
        parts = [float(v) for v in re.split(r"[\s,;]+", t) if v]
        if len(parts) < 2:
            continue
        xs.append(parts[0])
        ys.append(parts[1])
        ss.append(parts[2] if len(parts) > 2 else float("nan"))
    if len(xs) < 10:
        raise ValueError("no pattern found: expected columns 2theta intensity [sigma]")
    x, y, s = np.array(xs), np.array(ys), np.array(ss)
    order = np.argsort(x)
    x, y, s = x[order], y[order], s[order]
    if np.isnan(s).all():
        s = np.sqrt(np.clip(y, 1.0, None))
    else:
        s = np.where(np.isnan(s) | (s <= 0), np.sqrt(np.clip(y, 1.0, None)), s)
    return {"two_theta": x, "intensity": y, "sigma": s, "name": name}


def _read_xrdml(text: str) -> Dict:
    import xml.etree.ElementTree as ET
    root = ET.fromstring(text)
    ns = {"x": root.tag.split("}")[0].strip("{")} if root.tag.startswith("{") else {}
    q = (lambda p: p.replace("/", "/x:").replace("//x:", "//x:")) if ns else (lambda p: p)
    pts = root.find(".//x:dataPoints" if ns else ".//dataPoints", ns)
    if pts is None:
        raise ValueError("xrdml: no dataPoints")
    start = end = None
    for pos in pts.findall("x:positions" if ns else "positions", ns):
        if pos.get("axis") == "2Theta":
            s = pos.find("x:startPosition" if ns else "startPosition", ns)
            e = pos.find("x:endPosition" if ns else "endPosition", ns)
            lst = pos.find("x:listPositions" if ns else "listPositions", ns)
            if lst is not None:
                xs = np.array([float(v) for v in lst.text.split()])
                start = None
            elif s is not None and e is not None:
                start, end = float(s.text), float(e.text)
    node = pts.find("x:intensities" if ns else "intensities", ns)
    if node is None:
        node = pts.find("x:counts" if ns else "counts", ns)
    y = np.array([float(v) for v in node.text.split()])
    if start is not None:
        xs = np.linspace(start, end, len(y))
    return {"two_theta": xs, "intensity": y, "sigma": np.sqrt(np.clip(y, 1.0, None)), "name": "xrdml"}


def read_pattern(path) -> Dict:
    p = Path(path)
    if p.stat().st_size > 50 * 1024 * 1024:
        raise ValueError("pattern file over 50 MB")
    return read_pattern_text(p.read_text(errors="replace"), p.name)


def snip_background(y: np.ndarray, width: int = 40) -> np.ndarray:
    """SNIP (Ryan et al. 1988) on the LLS-transformed signal."""
    v = np.log(np.log(np.sqrt(np.clip(y, 0, None) + 1) + 1) + 1)
    n = len(v)
    for k in range(1, max(2, int(width)) + 1):
        a = v.copy()
        a[k:n - k] = np.minimum(v[k:n - k], 0.5 * (v[:n - 2 * k] + v[2 * k:]))
        v = a
    return (np.exp(np.exp(v) - 1) - 1) ** 2 - 1


def find_peaks(two_theta, intensity, prominence: float = 0.02, min_separation: float = 0.08,
               background: Optional[np.ndarray] = None, smooth: bool = True, fit: bool = True) -> Dict:
    """Peaks of a measured pattern: [{two_theta, height, area, fwhm, eta}] by
    decreasing height; ``prominence`` is relative to the strongest peak."""
    from scipy.signal import find_peaks as _fp, savgol_filter
    x = np.asarray(two_theta, float)
    y = np.asarray(intensity, float)
    step = float(np.median(np.diff(x)))
    bg = snip_background(y, max(5, int(1.5 / step))) if background is None else np.asarray(background, float)
    net = y - bg
    ys = savgol_filter(net, max(5, (int(0.1 / step) // 2) * 2 + 1), 3) if smooth and len(y) > 15 else net
    top = float(ys.max()) if ys.max() > 0 else 1.0
    idx, props = _fp(ys, prominence=prominence * top, distance=max(1, int(min_separation / step)),
                     height=prominence * top)
    peaks = []
    for i in idx:
        p = {"two_theta": float(x[i]), "two_theta_max": float(x[i]), "height": float(net[i]), "fwhm": None,
             "area": None, "eta": None}
        if fit:
            p.update(_fit_peak(x, net, i, step))
            if abs(p["two_theta"] - p["two_theta_max"]) > max(0.5 * (p.get("fwhm") or 0), 2 * step):
                p["two_theta"] = p["two_theta_max"]          # a fit pulled onto a neighbor: keep the maximum
        peaks.append(p)
    peaks.sort(key=lambda p: -p["height"])
    # one peak found twice (a fit drifting onto its neighbor): keep the stronger
    kept = []
    for p in peaks:
        if all(abs(p["two_theta"] - q["two_theta"]) >= min_separation for q in kept):
            kept.append(p)
    peaks = kept
    return {"peaks": peaks, "background": bg, "step": step}


def _fit_peak(x, y, i, step) -> Dict:
    from scipy.optimize import least_squares
    half = y[i] / 2
    l = i
    while l > 0 and y[l] > half:
        l -= 1
    r = i
    while r < len(y) - 1 and y[r] > half:
        r += 1
    fw = max((r - l) * step, 2 * step)
    sl = slice(max(0, i - int(2.5 * fw / step) - 2), min(len(x), i + int(2.5 * fw / step) + 3))
    xs, ys = x[sl], y[sl]
    if len(xs) < 5:
        return {"fwhm": fw}
    area0 = y[i] * fw * 1.2

    def res(p):
        x0, A, f, eta, c = p
        return A * pseudo_voigt(xs, x0, f, eta) + c - ys

    try:
        r_ = least_squares(res, [x[i], area0, fw, 0.5, 0.0],
                           bounds=([x[i] - fw, 0, step / 2, 0, -abs(y[i])], [x[i] + fw, np.inf, 10 * fw + 1, 1, abs(y[i])]))
        x0, A, f, eta, c = r_.x
        return {"two_theta": float(x0), "area": float(A), "fwhm": float(f), "eta": float(eta),
                "height": float(A * pseudo_voigt(np.array([x0]), x0, f, eta)[0])}
    except Exception:            # pragma: no cover - keep the raw maximum
        return {"fwhm": fw}


def strip_ka2(peaks: Sequence[Dict], wavelength=DEFAULT_SOURCE, tol: float = 0.03) -> List[Dict]:
    """Peaks without the K-alpha2 satellites of stronger ones (for indexing)."""
    l1, l2, ratio, _ = source(wavelength)
    if not l2 or ratio <= 0:
        return list(peaks)
    out = []
    for p in peaks:
        sat = False
        for q in peaks:
            if q is p or q["height"] < 1.3 * p["height"]:
                continue
            s2 = l2 / l1 * math.sin(math.radians(q.get("two_theta_max", q["two_theta"])) / 2)
            t2 = math.degrees(2 * math.asin(s2)) if s2 < 1 else None
            if t2 is not None and min(abs(t2 - p["two_theta"]), abs(t2 - p.get("two_theta_max", p["two_theta"]))) \
                    <= tol + 0.0005 * q["two_theta"]:
                sat = True
                break
        if not sat:
            out.append(p)
    return out


def d_spacing(two_theta, lam: float) -> np.ndarray:
    return lam / (2 * np.sin(np.radians(np.asarray(two_theta, float)) / 2))


def compare_patterns(x_a, y_a, x_b, y_b) -> Dict[str, float]:
    """Similarity of two patterns on a common grid: Pearson r and Rwp-like
    misfit after scaling b onto a."""
    x = np.asarray(x_a, float)
    yb = np.interp(x, np.asarray(x_b, float), np.asarray(y_b, float))
    ya = np.asarray(y_a, float)
    k = float((ya * yb).sum() / max((yb * yb).sum(), 1e-30))
    r = float(np.corrcoef(ya, yb)[0, 1]) if ya.std() > 0 and yb.std() > 0 else 0.0
    rwp = float(np.sqrt(((ya - k * yb) ** 2).sum() / max((ya * ya).sum(), 1e-30)))
    return {"pearson": r, "scale": k, "rwp": rwp}
