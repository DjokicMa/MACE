"""Band structure and DOS data for the GUI's interactive (plotly) plots.

Nothing is re-parsed here: the BAND.DAT reader and the k-path reader are
``ipBANDS_V2.read_band_data`` / ``parse_kpoint_path_and_segments`` and the DOS
readers are ``ipDOS_V2``'s, the same code ``mace plotting --band/--dos`` runs.
This module only reshapes their arrays into JSON for the browser:

* energies in eV; with a POTC run beside the data (``<material>_POTC.POTC.DAT``
  and ``<material>_POTC.out``) the plotters shift everything onto the vacuum
  scale, so the Fermi level sits at that shift (``fermi``), not at 0;
* CRYSTAL writes two spin blocks even for a closed-shell run; when the beta
  block equals the alpha one (bands) or its negative (DOS) the run is
  reported closed-shell and only one set is sent;
* a path discontinuity (``U|K``) gets a break (null) so no line joins the
  two sides.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_PLOTTING = Path(__file__).resolve().parents[2] / "Plotting"
BAND_SUFFIXES = (".BAND.DAT", ".band.band.dat", "_band.dat")
DOS_SUFFIXES = (".DOSS.DAT", "_doss.dat")
MAX_BANDS = 400            # per spin, after windowing
WINDOW_EV = 25.0           # bands that never come within this of E_F are dropped


def _plotters():
    os.environ.setdefault("MPLBACKEND", "Agg")
    if str(_PLOTTING) not in sys.path:
        sys.path.insert(0, str(_PLOTTING))
    import ipBANDS_V2
    import ipDOS_V2
    return ipBANDS_V2, ipDOS_V2


def _stem(name: str, suffixes) -> str:
    for suffix in suffixes:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def _material(stem: str, tag: str) -> str:
    return stem[: -len(tag)] if stem.endswith(tag) else stem


def _potc(directory: Path, material: str) -> Tuple[str, str, bool]:
    dat = directory / f"{material}_POTC.POTC.DAT"
    out = directory / f"{material}_POTC.out"
    return str(dat), str(out), dat.is_file() and out.is_file()


def _plotly_label(label: str) -> str:
    """'$Σ_{0}$' (matplotlib mathtext) -> 'Σ<sub>0</sub>' (plotly)."""
    label = label.replace("$", "")
    return re.sub(r"_\{([^}]*)\}", r"<sub>\1</sub>", label)


def _with_breaks(x: np.ndarray, ys: np.ndarray, breaks: List[float]):
    """Insert a null between two points at the same x on a discontinuous tick."""
    cut = [i + 1 for i in range(len(x) - 1)
           if any(abs(x[i] - b) < 1e-6 and abs(x[i + 1] - b) < 1e-6 for b in breaks)]
    if not cut:
        return x.tolist(), ys.T.tolist()
    xs = np.insert(x.astype(float), cut, np.nan)
    yy = np.insert(ys.astype(float), cut, np.nan, axis=0)
    clean = lambda a: [None if np.isnan(v) else float(v) for v in a]  # noqa: E731
    return clean(xs), [clean(col) for col in yy.T] if yy.ndim == 2 else clean(yy)


def _window(bands: np.ndarray, fermi: float) -> Tuple[np.ndarray, bool]:
    keep = [j for j in range(bands.shape[1])
            if bands[:, j].max() > fermi - WINDOW_EV and bands[:, j].min() < fermi + WINDOW_EV]
    truncated = len(keep) > MAX_BANDS
    if truncated:   # the bands nearest the Fermi level
        keep = sorted(keep, key=lambda j: abs(bands[:, j].mean() - fermi))[:MAX_BANDS]
        keep.sort()
    return bands[:, keep], truncated


def band_data(path: Path) -> Dict[str, Any]:
    path = Path(path)
    ipb, _ = _plotters()
    stem = _stem(path.name, BAND_SUFFIXES)
    material = _material(stem, "_band")
    dat, out, has_potc = _potc(path.parent, material)
    E, B, Eb, Bb, ticks, maxV, _kidx, _np = ipb.read_band_data(
        str(path), dat if has_potc else None, out if has_potc else None)
    n = len(E)
    labels: List[str] = []
    for cand in (path.parent / f"{stem}.d3", path.parent / f"{material}_band.d3",
                 path.parent / f"{material}.d3"):
        if cand.is_file():
            all_labels, _segs, _clean = ipb.parse_kpoint_path_and_segments(str(cand))
            labels = [_plotly_label(lab) for lab in (all_labels or [])]
            break
    # each spin block repeats the tick spec: keep the first copy
    ticks = [float(t) for t in ticks]
    half = len(ticks) // 2
    if half and len(ticks) % 2 == 0 and np.allclose(ticks[:half], ticks[half:]):
        ticks = ticks[:half]
    if len(labels) != len(ticks):
        labels = labels[: len(ticks)] + [""] * max(0, len(ticks) - len(labels))
    breaks = [t for t, lab in zip(ticks, labels) if "|" in lab]
    spin = bool(np.any(Bb != 0)) and not np.array_equal(B, Bb)
    fermi = float(maxV)
    up, trunc_up = _window(B, fermi)
    x, up_cols = _with_breaks(E[:n], up[:n], breaks)
    result = {
        "kind": "band", "name": path.name, "x": x, "up": up_cols,
        "down": None, "spin_polarized": spin, "fermi": fermi,
        "reference": "vacuum" if has_potc else "fermi",
        "ticks": ticks, "labels": labels, "truncated": trunc_up,
        "n_kpoints": n, "n_bands": int(B.shape[1]),
    }
    if spin:
        down, trunc_down = _window(Bb, fermi)
        _, result["down"] = _with_breaks(Eb[:n], down[:n], breaks)
        result["truncated"] = trunc_up or trunc_down
    return result


def _doss_header(path: Path) -> Tuple[Optional[int], int]:
    with open(path) as fh:
        head = fh.readline().split()
    try:
        nproj = int(head[head.index("NPROJ") + 1])
    except (ValueError, IndexError):
        nproj = None
    try:
        nspin = int(head[head.index("NSPIN") + 1])
    except (ValueError, IndexError):
        nspin = 1
    return nproj, nspin


def dos_data(path: Path) -> Dict[str, Any]:
    path = Path(path)
    _, ipd = _plotters()
    stem = _stem(path.name, DOS_SUFFIXES)
    material = _material(stem, "_doss")
    dat, out, has_potc = _potc(path.parent, material)
    nproj, nspin = _doss_header(path)
    labels: Optional[List[str]] = None
    for cand in (path.parent / f"{stem}.d3", path.parent / f"{material}_doss.d3"):
        if cand.is_file():
            labels = [ipd.doss_label(r).strip() for r in ipd.read_doss_projection_labels(str(cand))]
            # a projection on every orbital of an element reads "C" in the
            # .d3: in a legend beside "C (s)", "C (p)" that looks like a typo
            labels = [f"{lab} (all)" if re.fullmatch(r"[A-Z][a-z]?", lab) else lab
                      for lab in labels]
            break
    if labels is None or (nproj is not None and len(labels) != nproj):
        if nproj is None:
            raise ValueError(f"{path.name}: no NPROJ in the header and no .d3 beside it")
        labels = [f"projection {k}" for k in range(1, nproj)] + ["Total"]
    maxV = ipd.vacuum_shift(dat, out) if has_potc else 0
    cols = [np.asarray(c, float) for c in ipd.read_doss_columns(str(path), len(labels) + 1, maxV)]
    n_all = len(cols[0])
    half = n_all // 2 if nspin == 2 and n_all % 2 == 0 else n_all
    energy = cols[0][:half]
    up = [c[:half] for c in cols[1:]]
    down = [c[half:] for c in cols[1:]] if half < n_all else None
    spin = down is not None and not all(np.array_equal(d, -u) for u, d in zip(up, down))
    return {
        "kind": "dos", "name": path.name, "energy": energy.tolist(),
        "fermi": float(maxV), "reference": "vacuum" if has_potc else "fermi",
        "spin_polarized": spin,
        "projections": [{"label": lab, "up": u.tolist(),
                         "down": (d.tolist() if spin else None)}
                        for lab, u, d in zip(labels, up, down or up)],
    }


def partner(path: Path, kind: str) -> Optional[Path]:
    """The DOSS file of the same run as a BAND file (and the reverse)."""
    if kind == "band":
        material = _material(_stem(path.name, BAND_SUFFIXES), "_band")
        names = (f"{material}_doss.DOSS.DAT", f"{material}.DOSS.DAT", f"{material}_doss.dat")
    else:
        material = _material(_stem(path.name, DOS_SUFFIXES), "_doss")
        names = (f"{material}_band.BAND.DAT", f"{material}.BAND.DAT", f"{material}_band.dat",
                 f"{material}.band.band.dat")
    for name in names:
        if (path.parent / name).is_file():
            return path.parent / name
    return None


# ---------------------------------------------------------------- transport (plan E8)
TRANSPORT_QUANTITIES = {
    # file token: (label, unit shown, factor from the file's SI unit)
    "SEEBECK": ("Seebeck coefficient", "µV/K", 1e6),
    "SIGMA": ("Electrical conductivity σ", "S/m", 1.0),
    "SIGMAS": ("σS (conductivity × Seebeck)", "A/(m·K)", 1.0),
    "KAPPA": ("Electronic thermal conductivity κ", "W/(m·K)", 1.0),
    "TDF": ("Transport distribution function", "1/ħ²·eV·fs/Å", 1.0),
}
_TRANSPORT_RE = re.compile(r"^(?P<stem>.+)\.(?P<q>SEEBECK|SIGMAS|SIGMA|KAPPA|TDF)\.dat$", re.I)


def transport_data(path: Path) -> Dict[str, Any]:
    """A BOLTZTRA .dat file for the transport plot: per temperature, the chemical
    potential and the average of the tensor's diagonal (both from the database
    extractor's own ``_parse_transport_dat``), plus the xx, yy, zz components
    read from the same rows; the other quantities of the run beside it."""
    from mace.utils.property_extractor import CrystalPropertyExtractor
    m = _TRANSPORT_RE.match(path.name)
    if not m:
        raise ValueError(f"{path.name} is not a BOLTZTRA .SEEBECK/.SIGMA/.SIGMAS/.KAPPA/.TDF.dat file")
    q = m.group("q").upper()
    label, unit, factor = TRANSPORT_QUANTITIES[q]
    if q == "TDF":
        return _tdf_data(path, m.group("stem"), label, unit)
    file_unit, rows = CrystalPropertyExtractor._parse_transport_dat(path)
    if not rows:
        raise ValueError(f"{path.name}: no data rows")
    # the tensor components, from the same data lines (column layout as the extractor reads it)
    comps: Dict[float, Dict[str, List[float]]] = {}
    header = None
    for line in path.read_text(errors="ignore").splitlines():
        s = line.strip()
        if s.startswith("#") and "Mu(eV)" in s:
            header = s.lstrip("#").split()
            continue
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        try:
            vals = [float(x) for x in parts]
        except ValueError:
            continue
        if len(vals) < 4 or not all(np.isfinite(vals)):
            continue
        t = vals[3:]
        diag = (t[0], t[4], t[8]) if len(t) == 9 else (t[0], t[3], t[5]) if len(t) == 6 else (t[0], t[1], t[2]) if len(t) >= 3 else (t[0],) * 3
        c = comps.setdefault(vals[1], {"xx": [], "yy": [], "zz": []})
        for k, v in zip(("xx", "yy", "zz"), diag):
            c[k].append(v * factor)
    temps: Dict[float, Dict[str, List[float]]] = {}
    for T, mu, n, avg in rows:
        d = temps.setdefault(T, {"mu": [], "avg": [], "carriers": []})
        d["mu"].append(mu)
        d["avg"].append(avg * factor)
        d["carriers"].append(n)
    blocks = []
    for T in sorted(temps):
        b = temps[T]
        c = comps.get(T, {})
        blocks.append({"T": T, "mu": b["mu"], "avg": b["avg"], "carriers": b["carriers"],
                       **{k: c.get(k) for k in ("xx", "yy", "zz") if c.get(k) and len(c[k]) == len(b["mu"])}})
    return {"name": path.name, "quantity": q, "label": label, "unit": unit, "file_unit": file_unit, "blocks": blocks,
            "siblings": _transport_siblings(path, m.group("stem")), "header": header, "x": "mu"}


def _transport_siblings(path: Path, stem: str) -> Dict[str, str]:
    siblings = {}
    for qq in TRANSPORT_QUANTITIES:
        for cand in (path.with_name(f"{stem}.{qq}.dat"), path.with_name(f"{stem}.{qq.lower()}.dat")):
            if cand.is_file():
                siblings[qq] = cand.name
                break
    return siblings


def _tdf_data(path: Path, stem: str, label: str, unit: str) -> Dict[str, Any]:
    """The TDF file: energy, then the symmetric tensor xx xy xz yy yz zz (no temperature)."""
    E, comp = [], {"xx": [], "yy": [], "zz": []}
    for line in path.read_text(errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        try:
            v = [float(x) for x in s.split()]
        except ValueError:
            continue
        if len(v) < 7 or not all(np.isfinite(v)):
            continue
        E.append(v[0])
        comp["xx"].append(v[1]); comp["yy"].append(v[4]); comp["zz"].append(v[6])
    if not E:
        raise ValueError(f"{path.name}: no data rows")
    avg = [(a + b + c) / 3 for a, b, c in zip(comp["xx"], comp["yy"], comp["zz"])]
    return {"name": path.name, "quantity": "TDF", "label": label, "unit": unit, "file_unit": None, "x": "energy",
            "blocks": [{"T": None, "mu": E, "avg": avg, **comp}], "siblings": _transport_siblings(path, stem), "header": None}
