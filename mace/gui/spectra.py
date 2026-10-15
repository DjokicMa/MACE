"""IR and Raman spectra for the GUI's plots (plan B5, T9).

Nothing is re-parsed: ``spectra_api.read_spectrum`` reads the file (as
``mace plotting --ir/--raman`` does), ``RAMAN_MIN_COLS`` names what each
Raman view needs, and ``group_files_by_material`` / ``average_spectrum``
average the conformers as ``mace plotting --spectra --average`` does. This
module only names the columns and shapes the arrays into JSON.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

# IRSPEC.DAT (CRYSTAL23 manual, p.453): frequency (cm-1), wavelength (nm), the raw absorbance; a
# periodic run adds the classical and the two Rayleigh-scattering absorbances (columns 4-6) and one
# reflectance per inequivalent polarization direction (7 on). The wavelength is a second x axis.
IR_COLUMNS = ["absorbance", "absorbance_classical", "absorbance_rayleigh_1", "absorbance_rayleigh_2"]
IR_FIRST = 2                                           # 0-based column of the raw absorbance
RAMAN_COLUMNS = ["total", "parallel", "perpendicular", "xx", "xy", "xz", "yy", "yz", "zz"]
MAX_POINTS = 20000


def kind_of(name: str) -> Optional[str]:
    up = name.upper()
    if up.endswith("IRSPEC.DAT"):
        return "ir"
    if up.endswith("RAMSPEC.DAT"):
        return "raman"
    return None


def _shape(name: str, kind: str, data) -> Dict[str, Any]:
    import numpy as np
    from mace.plotting.handlers.spectra_api import RAMAN_MIN_COLS
    if data.size == 0 or data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"{name}: needs 2 columns at least, found {0 if data.size == 0 else (data.shape[1] if data.ndim == 2 else 1)}")
    if kind == "ir" and data.shape[1] < 3:
        raise ValueError(f"{name}: an IR spectrum needs 3 columns, found {data.shape[1]}")
    if kind == "raman" and data.shape[1] < RAMAN_MIN_COLS["total"]:
        raise ValueError(f"{name}: a Raman spectrum needs {RAMAN_MIN_COLS['total']} columns, found {data.shape[1]}")
    step = max(1, len(data) // MAX_POINTS)
    d = data[::step]
    cols: Dict[str, List[float]] = {}
    for j in range(IR_FIRST if kind == "ir" else 1, d.shape[1]):
        if kind == "ir":
            k = j - IR_FIRST
            label = IR_COLUMNS[k] if k < len(IR_COLUMNS) else f"reflectance_{k - len(IR_COLUMNS) + 1}"
        else:
            label = RAMAN_COLUMNS[j - 1] if j - 1 < len(RAMAN_COLUMNS) else f"column {j + 1}"
        cols[label] = [float(v) for v in np.round(d[:, j], 7)]
    out = {"x": [float(v) for v in np.round(d[:, 0], 5)], "columns": cols, "n_points": int(len(data)),
           "ylabel": "Absorbance" if kind == "ir" else "Raman activity"}
    if kind == "ir":
        out["wavelength_nm"] = [float(v) for v in np.round(d[:, 1], 5)]
    return out


def read(path: Path) -> Dict[str, Any]:
    from mace.plotting.handlers.spectra_api import read_spectrum, group_files_by_material
    kind = kind_of(path.name)
    if kind is None:
        raise ValueError(f"{path.name} is not an IRSPEC.DAT or RAMSPEC.DAT file")
    out = _shape(path.name, kind, read_spectrum(str(path)))
    out.update(kind=kind, name=path.name)
    # the conformers of the same material beside it (conf-style names), for "average"
    same = [str(p) for p in sorted(path.parent.iterdir()) if p.is_file() and kind_of(p.name) == kind]
    groups = group_files_by_material(same)
    for mat, files in groups.items():
        if str(path) in files and len(files) > 1:
            out["group"] = [Path(f).name for f in files]
            out["material"] = mat
    return out


def average(path: Path) -> Dict[str, Any]:
    from mace.plotting.handlers.spectra_api import group_files_by_material, average_spectrum, RAMAN_MIN_COLS
    kind = kind_of(path.name)
    same = [str(p) for p in sorted(path.parent.iterdir()) if p.is_file() and kind_of(p.name) == kind]
    groups = group_files_by_material(same)
    files = next((fs for fs in groups.values() if str(path) in fs), None)
    if not files or len(files) < 2:
        raise ValueError("no conformers of this material beside it to average")
    res = average_spectrum(files, 3 if kind == "ir" else RAMAN_MIN_COLS["total"])
    if res is None:
        raise ValueError("none of the conformers' spectra could be read")
    data, n = res
    out = _shape(path.name, kind, data)
    out.update(kind=kind, name=f"average of {n} conformers", n_used=n)
    return out


# ---------------------------------------------------------------- broadened from the modes (plan E4)
# CRYSTAL's own broadening (IRSPEC/RAMSPEC DAMPFAC): a full width at half maximum of 8 cm-1 by default.
# (mace plotting --freq's viewer uses gamma = 8 as a half width, FWHM 16; its look is not changed.)
DEFAULT_FWHM = 8.0
SHAPES = ("lorentzian", "gaussian", "pseudo-voigt")
XMAX_LIMIT = 20000.0                 # cm-1: above any vibrational frequency


def dampfac(out: Path, kind: str) -> Optional[float]:
    """The DAMPFAC the run's deck gives its IRSPEC (kind ir) or RAMSPEC (raman)
    block, from <stem>.d12 or MACE's <stem>_temp.d12 beside the output."""
    block = "IRSPEC" if kind == "ir" else "RAMSPEC"
    for cand in (out.with_suffix(".d12"), out.with_name(out.stem + "_temp.d12")):
        try:
            lines = [ln.strip().upper() for ln in cand.read_text(errors="replace").splitlines()]
        except OSError:
            continue
        inside = False
        for i, ln in enumerate(lines):
            if ln == block:
                inside = True
            elif inside and ln == "END":
                inside = False
            elif inside and ln == "DAMPFAC" and i + 1 < len(lines):
                try:
                    v = float(lines[i + 1].split()[0])
                except (ValueError, IndexError):
                    return None
                return v if 0 < v < 1000 else None
    return None


def active_peaks(parsed: Dict[str, Any], kind: str) -> List[Dict[str, Any]]:
    """The IR- or Raman-active real modes with a nonzero intensity, one peak per
    degenerate set. CRYSTAL prints one row per set ("4-   6") whose INTENS already
    holds the degeneracy (INTENS_n = ... d_n (Z_x^2 + Z_y^2 + Z_z^2), as its output
    says); mace plotting's parser gives each mode of the set that row, so the set's
    first mode stands for it here and the others are skipped (else a triply
    degenerate peak would count three times)."""
    key, act = ("ir_intens", "ir_active") if kind == "ir" else ("raman_intens", "raman_active")
    out = []
    skip = 0
    for m in parsed["modes"]:
        if skip:
            skip -= 1
            continue
        d = int(m.get("degeneracy") or 1)
        skip = max(0, d - 1)            # the set's partners follow it in the table's order
        inten = m.get(key) or 0.0
        if m.get(act) and not m.get("imaginary") and m["freq"] > 1 and inten > 0:
            out.append({"mode": m["mode"], "freq": float(m["freq"]), "irrep": m.get("irrep"), "intens": float(inten),
                        "degeneracy": d})
    return out


def from_modes(parsed: Dict[str, Any], kind: str, shape: str = "lorentzian", fwhm: float = DEFAULT_FWHM,
               eta: float = 1.0, xmin: float = 0.0, xmax: Optional[float] = None) -> Dict[str, Any]:
    """An IR or Raman curve broadened from the mode table: IR Lorentzian (or
    Gaussian), Raman pseudo-Voigt with weight ``eta`` on the Lorentzian (1 =
    pure Lorentzian, CRYSTAL's VOIGT default). Every line has the full width
    ``fwhm`` and unit area times its intensity, as CRYSTAL's SPEC.DAT curves."""
    import numpy as np
    if kind not in ("ir", "raman"):
        raise ValueError("kind is ir or raman")
    if not (np.isfinite(fwhm) and 0.5 <= fwhm <= 500):
        raise ValueError("fwhm must be between 0.5 and 500 cm-1")
    if not np.isfinite(eta):
        raise ValueError("eta must be a number from 0 to 1")
    if xmax is not None and not (np.isfinite(xmax) and 10 <= xmax <= XMAX_LIMIT):
        # the grid is x step min(fwhm / 8, 1): an unbounded xmax would be an unbounded array
        raise ValueError(f"xmax must be between 10 and {XMAX_LIMIT:g} cm-1")
    eta = 1.0 if kind == "ir" and shape != "gaussian" else 0.0 if shape == "gaussian" else float(min(max(eta, 0.0), 1.0))
    peaks = active_peaks(parsed, kind)
    if not peaks:
        raise ValueError("This output has no IR or Raman intensities (run FREQCALC with INTENS)" if kind == "ir" or not
                         any(m.get("raman_intens") for m in parsed["modes"]) else "No Raman-active mode with an intensity")
    f = np.array([p["freq"] for p in peaks])
    I = np.array([p["intens"] for p in peaks])
    hi = float(xmax) if xmax is not None else float(f.max() + 6 * fwhm + 100)
    step = min(fwhm / 8.0, 1.0)
    x = np.arange(max(0.0, float(xmin)), hi + step, step)
    half = fwhm / 2.0
    y = np.zeros_like(x)
    sig2 = fwhm * fwhm / (8 * np.log(2))                         # Gaussian: FWHM = 2 sqrt(2 ln 2) sigma
    for k in range(0, len(f), 256):                              # in blocks: bounded memory
        d = x[None, :] - f[k:k + 256, None]
        L = (half / np.pi) / (d * d + half * half)
        G = np.exp(-d * d / (2 * sig2)) / np.sqrt(2 * np.pi * sig2)
        y += (I[k:k + 256, None] * (eta * L + (1 - eta) * G)).sum(axis=0)
    step_out = max(1, len(x) // MAX_POINTS)
    col = "absorbance" if kind == "ir" else "total"
    label = {"lorentzian": "Lorentzian", "gaussian": "Gaussian", "pseudo-voigt": f"pseudo-Voigt (η {eta:g})"}.get(shape, shape)
    if kind == "raman" and shape != "gaussian":
        label = "Lorentzian" if eta == 1.0 else f"pseudo-Voigt (η {eta:g})"
    return {"x": [float(v) for v in np.round(x[::step_out], 4)], "columns": {col: [float(v) for v in np.round(y[::step_out], 7)]},
            "n_points": int(len(x)), "kind": kind, "source": "modes", "fwhm": float(fwhm), "shape": shape, "eta": eta,
            "shape_label": label, "n_active": len(peaks),
            "ylabel": "IR absorbance (from the modes)" if kind == "ir" else "Raman intensity (relative, max 1000)",
            "peaks": peaks}
