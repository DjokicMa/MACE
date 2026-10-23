"""The plot gallery: many outputs' plots of one kind at once.

One module for the terminal and the browser. ``mace plotting --gallery KIND
FILES/FOLDER`` and the GUI's Plot Gallery both

* collect the set with :func:`collect` (a folder is searched recursively, a
  file of another kind brings the file of this kind of the same material
  beside it), capped at ``MAX_ITEMS`` with a message saying so;
* read each pane with :func:`pane` (the readers ``mace plotting`` and the
  GUI's plot documents use: ``mace.gui.results``, ``mace.gui.spectra``,
  ``mace.utils.out_convergence``, ``mace.gui.structures``), one JSON-able
  description per pane that matplotlib (here) and plotly (the browser) draw;
* caption the set with :func:`captions` (material id, formula, band gap and
  energy from the materials.db beside the files, when there is one), written
  as CSV by :func:`captions_csv`;
* export the grid as one figure with :func:`render_figure` (the GUI runs
  ``mace plotting --gallery`` itself for its Export, so both make the same
  file).

A pane is ``{"path", "name", "title", "kind", "panels": [panel], "error"}``;
a panel is ``{"series": [{"x", "y", "name", "role"}], "xlabel", "ylabel",
"xlim", "ylim", "xlog", "ylog", "energy" ("x" | "y" | None, the axis the
synchronized energy window applies to), "hlines", "vlines", "ticks",
"ticklabels", "atoms", "cell", "width"}``.
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import sqlite3
from pathlib import Path
from typing import Callable, Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import quote

from mace.utils import plot_labels as L

KINDS = ("band", "dos", "bandos", "ir", "raman", "opt", "scf", "structure")
LABELS = {"band": "Band structure", "dos": "Density of states", "bandos": "Band structure + DOS",
          "ir": "IR spectrum", "raman": "Raman spectrum", "opt": "Optimization energy",
          "scf": "SCF convergence", "structure": "Structure"}
MAX_ITEMS = 200            # panes in one gallery (the GUI's and the CLI's default)
SCAN_CAP = 20000           # files looked at when searching folders
EWINDOW = (-6.0, 6.0)      # the synchronized energy window, relative to the Fermi level (mace plotting --band's)
REFERENCES = ("auto", "fermi", "vacuum")    # mace plotting --energy-reference
MAX_POINTS = 2000          # per series in a pane: a pane is small
MAX_COLS = 8
from mace.constants import HARTREE_TO_EV as HARTREE_EV  # noqa: E402
FIGURE_FORMATS = ("svg", "png", "pdf")
CAPTION_COLUMNS = ("n", "material_id", "formula", "band_gap_eV", "energy_eV", "file")
_GAP_NAMES = ("band_gap_ev", "band_gap", "electronic_band_gap", "gap_ev")
_ENERGY_NAMES = ("total_energy_ev", "total_energy_au", "total_energy", "final_energy_ev", "final_energy")
_SKIP_DIRS = {"backups", "__pycache__", "node_modules"}

_BAND = (".BAND.DAT", ".band.band.dat", "_band.dat")
_DOS = (".DOSS.DAT", "_doss.dat")
_OUT = (".out", ".log", ".o")
_STRUCT = (".cif", ".d12", ".out")


class GalleryError(ValueError):
    """A set or option the gallery cannot use (the message says why)."""


# ---------------------------------------------------------------- the set

def _natural(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(p))]


def _is_band(name: str) -> bool:
    return name.endswith(_BAND)


def _is_dos(name: str) -> bool:
    return name.endswith(_DOS)


def _head(path: Path, n: int = 512 * 1024) -> bytes:
    try:
        with open(path, "rb") as fh:
            return fh.read(n)
    except OSError:
        return b""


def _is_crystal_out(path: Path) -> bool:
    """A CRYSTAL output: its banner, or (a trimmed output, a job log of the
    run) the first geometry header CRYSTAL prints."""
    if not path.name.lower().endswith(_OUT):
        return False
    try:
        from mace.utils.db_build import is_crystal_output
        if is_crystal_output(path):
            return True
    except (OSError, ImportError):
        pass
    head = _head(path, 64 * 1024)
    return b"EEEEEEEEEE STARTING" in head and b"CALCULATION" in head


def _is_opt(path: Path) -> bool:
    return _is_crystal_out(path) and b"OPTGEOM" in _head(path)


def _lone_f25(path: Path, want: str) -> bool:
    """A fort.25 of BAND (``band``) or DOSS (``dos``) blocks with no BAND.DAT/DOSS.DAT of the same run
    beside it (where the DAT exists it is the file the gallery draws)."""
    from . import f25
    try:
        return f25.is_f25_name(path.name) and f25.plot_kind(path, want) is not None and not f25.has_dat_twin(path)
    except OSError:
        return False


def matches(path: Path, kind: str) -> bool:
    """Whether ``path`` is a file this gallery kind plots (by name; an output
    by its CRYSTAL banner and, for ``opt``, its OPTGEOM)."""
    name = path.name
    if kind in ("band", "bandos"):
        return _is_band(name) or _lone_f25(path, "band")
    if kind == "dos":
        return _is_dos(name) or _lone_f25(path, "dos")
    if kind in ("ir", "raman"):
        up = name.upper()
        return up.endswith("IRSPEC.DAT") if kind == "ir" else up.endswith("RAMSPEC.DAT")
    if kind == "opt":
        return _is_opt(path)
    if kind == "scf":
        return _is_crystal_out(path)
    if kind == "structure":
        low = name.lower()
        if low.endswith(".out"):
            return _is_crystal_out(path)
        return low.endswith((".cif", ".d12"))
    raise GalleryError(f"the gallery kind is one of {', '.join(KINDS)}")


def material_of(path: Path) -> str:
    try:
        from mace.utils.db_build import material_id_from_filename
        return material_id_from_filename(path)
    except Exception:
        return path.name.split(".")[0]


def _walk(folder: Path, budget: List[int]) -> Iterable[Path]:
    """The files below ``folder``, sorted, hidden folders and backups left out;
    stops after SCAN_CAP files (``budget`` counts them)."""
    try:
        entries = sorted(os.scandir(folder), key=lambda e: _natural(Path(e.name)))
    except OSError:
        return
    dirs = []
    for e in entries:
        if budget[0] <= 0:
            return
        if e.name.startswith("."):
            continue
        try:
            if e.is_dir(follow_symlinks=False):
                if e.name not in _SKIP_DIRS:
                    dirs.append(Path(e.path))
            elif e.is_file():
                budget[0] -= 1
                yield Path(e.path)
        except OSError:
            continue
    for d in dirs:
        yield from _walk(d, budget)


def _related(path: Path, kind: str) -> List[Path]:
    """For a file of another kind (an output picked for its band structure):
    the files of this kind beside it with the same material id."""
    try:
        sibs = sorted(path.parent.iterdir(), key=_natural)
    except OSError:
        return []
    mid = material_of(path)
    stem = path.name.split(".")[0]
    out = []
    for s in sibs:
        if s == path or not s.is_file() or not matches(s, kind):
            continue
        if material_of(s) == mid or s.name.startswith(stem):
            out.append(s)
    return out


def collect(inputs: Sequence[Any], kind: str, cap: int = MAX_ITEMS,
            inside: Optional[Callable[[Path], bool]] = None, page: int = 1) -> Dict[str, Any]:
    """The files of the set, in order: ``{"items": [Path], "total", "capped",
    "message", "scan_capped", "page", "pages", "offset"}``. ``inputs`` are files and folders.
    ``cap`` is the page size and ``page`` the page wanted (1-based): a set larger than one page is
    read a page at a time. ``inside``: a file it refuses is left out (the GUI: a symbolic link out
    of the served folder)."""
    if kind not in KINDS:
        raise GalleryError(f"the gallery kind is one of {', '.join(KINDS)}")
    cap = max(1, int(cap))
    found: List[Path] = []
    seen = set()
    budget = [SCAN_CAP]

    def add(p: Path) -> None:
        if inside is not None and not inside(p):
            return
        try:
            key = p.resolve()
        except (OSError, RuntimeError):
            key = p
        if key not in seen:
            seen.add(key)
            found.append(p)

    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            for f in _walk(p, budget):
                if matches(f, kind):
                    add(f)
        elif p.is_file():
            if matches(p, kind):
                add(p)
            elif kind in ("bandos",) and _is_dos(p.name):
                other = _partner(p, "dos")
                if other is not None:
                    add(other)
            else:
                for f in _related(p, kind):
                    add(f)
    total = len(found)
    return _paged(found, total, cap, page, kind, budget[0] <= 0)


def _paged(found: List[Path], total: int, cap: int, page: int, kind: str, scan_capped: bool = False) -> Dict[str, Any]:
    """One page of ``found`` (``cap`` per page) with the message that says which part of the set it is."""
    pages = max(1, -(-total // cap))
    page = max(1, min(int(page or 1), pages))
    offset = (page - 1) * cap
    items = found[offset:offset + cap]
    msg = None
    if total > cap:
        if page == 1:
            msg = (f"Showing the first {cap} of {total} files. Page through the set, or narrow it (a subfolder, "
                   "a selection or a database filter).")
        else:
            msg = f"Showing files {offset + 1}-{offset + len(items)} of {total} (page {page} of {pages})."
    if scan_capped:
        msg = ((msg + " ") if msg else "") + (f"The search stopped after {SCAN_CAP} files; pick a smaller folder "
                                              "to be sure every file is included.")
    if not items:
        msg = f"No {LABELS[kind].lower()} files in this set."
    return {"items": items, "total": total, "capped": total > cap, "message": msg, "scan_capped": scan_capped,
            "page": page, "pages": pages, "offset": offset, "page_size": cap}


def _partner(path: Path, kind: str) -> Optional[Path]:
    from mace.gui import results
    try:
        return results.partner(path, kind)
    except Exception:
        return None


# ---------------------------------------------------------------- panes

def _thin(xs: List[Any], ys: List[Any], limit: int = MAX_POINTS) -> Tuple[List[Any], List[Any]]:
    n = len(xs)
    if n <= limit:
        return list(xs), list(ys)
    step = int(math.ceil(n / limit))
    keep = sorted(set(range(0, n, step)) | {n - 1} | {i for i, v in enumerate(xs) if v is None}
                  | {i for i, v in enumerate(ys) if v is None})
    return [xs[i] for i in keep], [ys[i] for i in keep]


def _r(v, nd=5):
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else round(float(v), nd)


def _joined(x: List[Any], cols: List[List[Any]], lo: float, hi: float) -> Tuple[List[Any], List[Any]]:
    """Every band as one series, joined by a null between bands (one line
    object per pane, not one per band); bands outside [lo, hi] left out."""
    X: List[Any] = []
    Y: List[Any] = []
    for col in cols:
        vals = [v for v in col if v is not None]
        if not vals or max(vals) < lo or min(vals) > hi:
            continue
        if X:
            X.append(None)
            Y.append(None)
        X.extend(_r(v) for v in x)
        Y.extend(_r(v) for v in col)
    return X, Y


def _reference(d: Dict[str, Any], reference: str, shift: float) -> Tuple[float, float, bool]:
    """(what to subtract from the reader's energies, where E_F then is, vacuum
    scale?) for --energy-reference and --fermi-shift, as mace plotting --band and
    --dos draw them: "fermi" takes a POTC run off the vacuum scale, "vacuum"
    needs one; the shift moves the data, E_F's line and the window stay."""
    vac = d.get("reference") == "vacuum"
    if reference == "vacuum" and not vac:
        raise GalleryError("--energy-reference vacuum needs the run's POTC files beside the data")
    base = float(d["fermi"]) if (reference == "fermi" and vac) else 0.0
    return base + float(shift or 0.0), float(d["fermi"]) - base, vac and reference != "fermi"


def _elabel(vac: bool, shift: float) -> str:
    """The energy axis as mace plotting --band and the GUI's band plot title it (plot_labels.energy_title)."""
    return L.with_unit(L.energy_title("vacuum" if vac else "fermi", shift), "eV")


def _moved(vals: List[Any], off: float) -> List[Any]:
    return vals if not off else [None if v is None else v - off for v in vals]


def _band_panel(path: Path, ewin: Tuple[float, float], shift: float = 0.0,
                reference: str = "auto") -> Dict[str, Any]:
    from mace.gui import results
    from . import presets
    d = results.band_data(path)
    if d.get("domain") == "phonon":
        raise ValueError("phonon bands are drawn in the Plot viewer and by mace plotting --band, not in the gallery")
    metal = presets.is_metallic(d)           # Band style Auto: overlay for a metal, else single
    off, f, vac = _reference(d, reference, shift)
    if off:
        d = {**d, "up": [_moved(c, off) for c in d["up"]],
             "down": [_moved(c, off) for c in d["down"]] if d.get("down") else d.get("down")}
    lo, hi = f + ewin[0] - 2, f + ewin[1] + 2
    series = []
    x, y = _joined(d["x"], d["up"], lo, hi)
    series.append({"x": x, "y": y, "name": "α" if d.get("down") else "bands", "role": "up"})
    if d.get("down"):
        x, y = _joined(d["x"], d["down"], lo, hi)
        series.append({"x": x, "y": y, "name": "β", "role": "down"})
    xs = [v for v in d["x"] if v is not None]
    return {"series": series, "xlabel": "", "ylabel": _elabel(vac, shift),
            "xlim": [min(xs), max(xs)] if xs else None, "ylim": [f + ewin[0], f + ewin[1]], "energy": "y",
            "fermi": f, "eflines": [f], "vlines": list(d.get("ticks") or []), "ticks": list(d.get("ticks") or []),
            "ticklabels": list(d.get("labels") or []), "width": 3,
            # one set of bands (closed shell, or alpha = beta): drawn in the neutral band color, not spin up's
            "closed": not d.get("down"), "spin_identical": bool(d.get("spin_identical")), "metal": metal}


def _dos_series(d: Dict[str, Any], f: float, ewin: Tuple[float, float], rotate: bool,
                dos_colors: str = "auto", smooth: float = 0.0) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """(the series, the panel's DOS look): the total and the projections the DOS plot draws by default
    (presets.pick_projections: the 2-4 carrying the DOS in the window, largest first), each with its
    projection label (``proj``) so the GUI colors it in the theme as the CLI does on paper; spin down
    as the reader gives it (negative: the mirrored half). ``smooth``: the Gaussian sigma in eV of
    --dos-smooth (display only, presets.gauss_smooth). The look: dos_mode, dos_spin (both spins
    drawn), dos_side (mirrored or one), dos_weights, dos_order (the projections by weight: their
    projection-palette order), dos_max (the largest |DOS| drawn in the window) and dos_smooth."""
    from . import presets
    energy = d["energy"]
    lo, hi = f + ewin[0], f + ewin[1]
    keep = [i for i, e in enumerate(energy) if lo - 1 <= e <= hi + 1] or list(range(len(energy)))
    inwin = [i for i in keep if lo <= energy[i] <= hi] or keep
    projs = d["projections"]
    spin = bool(d.get("spin_polarized", any(pr.get("down") is not None for pr in projs)))
    weights = presets.dos_weights(energy, projs, lo, hi)
    look = presets.dos_styles([pr["label"] for pr in projs], dos_colors or "auto", show_total=True, spin=spin,
                              weights=weights)
    pk = presets.pick_projections(look["lines"], weights)
    by = {pr["label"]: (pr, st) for pr, st in zip(projs, look["lines"])}
    # the total (last in the file) first, then the picked projections, largest first
    order = [(projs[-1], look["lines"][-1])] + [by[lab] for lab in pk["shown"] if lab in by]
    sigma = presets.smooth_sigma(smooth)
    series, top = [], 0.0
    for k, (pr, st) in enumerate(order):
        role = "total" if k == 0 else "projection"
        for suffix, key in (("", "up"), (" β", "down")):
            vals = pr.get(key)
            if vals is None:
                continue
            if sigma:
                vals = presets.gauss_smooth(vals, energy, sigma)
            top = max([top] + [abs(vals[i] or 0.0) for i in inwin])
            e = [_r(energy[i]) for i in keep]
            v = [_r(vals[i], 6) for i in keep]
            e, v = _thin(e, v)
            series.append({"x": v if rotate else e, "y": e if rotate else v, "proj": pr["label"], "spin": key,
                           "name": (st["name"] if k else "Total") + suffix, "role": role})
    return series, {"dos_mode": look["mode"], "dos_spin": spin, "dos_side": "mirrored" if spin else "one",
                    "dos_colors": dos_colors or "auto", "dos_weights": {k: round(w, 6) for k, w in weights.items()},
                    "dos_order": pk["order"], "dos_max": _r(top, 6) or 0.0, "dos_smooth": sigma}


def _dos_panel(path: Path, ewin: Tuple[float, float], rotate: bool = False, shift: float = 0.0,
               reference: str = "auto", dos_colors: str = "auto", smooth: float = 0.0) -> Dict[str, Any]:
    """A DOS panel (the 1.1.6 DOS layout, as plotly_bands/bandfig.js draw it): ``eflines`` E_F
    (dotted), ``baseline`` DOS = 0 (in ink), ``nodata`` the energies the DOS run did not compute
    (presets.dos_window_gaps, shaded) and the DOS look of _dos_series."""
    from mace.gui import results
    from . import presets
    d = results.dos_data(path)
    if d.get("domain") == "phonon":
        raise ValueError("a phonon DOS is drawn in the Plot viewer and by mace plotting --dos, not in the gallery")
    off, f, vac = _reference(d, reference, shift)
    if off:
        d = {**d, "energy": _moved(d["energy"], off),
             "window": [w - off for w in d["window"]] if d.get("window") else None}
    series, look = _dos_series(d, f, ewin, rotate, dos_colors, smooth)
    look["nodata"] = presets.dos_window_gaps(d.get("window"), f + ewin[0], f + ewin[1])
    elabel = _elabel(vac, shift)
    if rotate:
        return {"series": series, "xlabel": "DOS", "ylabel": "", "ylim": [f + ewin[0], f + ewin[1]], "energy": "y",
                "fermi": f, "eflines": [f], "baseline": 0.0, "width": 1, **look}
    return {"series": series, "xlabel": elabel, "ylabel": "DOS (states/eV)", "xlim": [f + ewin[0], f + ewin[1]],
            "energy": "x", "fermi": f, "eflines": [f], "baseline": 0.0, "width": 1, **look}


def _spectrum_panel(path: Path) -> Dict[str, Any]:
    from mace.gui import spectra
    d = spectra.read(path)
    col = "absorbance" if d["kind"] == "ir" else "total"
    ys = d["columns"].get(col) or next(iter(d["columns"].values()))
    x, y = _thin([_r(v, 3) for v in d["x"]], [_r(v, 6) for v in ys])
    return {"series": [{"x": x, "y": y, "name": col, "role": "total"}], "xlabel": L.with_unit("Wavenumber", "cm-1"),
            "ylabel": d.get("ylabel") or "", "energy": None, "width": 1}


def _opt_panel(path: Path) -> Dict[str, Any]:
    from mace.utils import out_convergence as oc
    res = oc.parse(path)
    if not res.get("is_opt"):
        raise GalleryError(f"{path.name} is not a geometry optimization")
    rows = oc.energy_along_run(res, relative_to="final")
    if not rows:
        raise GalleryError(f"{path.name} has no optimization step with an energy yet")
    x = [r["point"] for r in rows if not r["final"]] or [rows[-1]["point"]]
    y = [_r(r["de_au"] * HARTREE_EV * 1000, 4) for r in rows if not r["final"]] or [0.0]
    end = (res.get("opt") or {}).get("end") or {}
    return {"series": [{"x": x, "y": y, "name": f"E {L.MINUS} {L.SYMBOLS['E_final']}", "role": "total", "markers": True}],
            "xlabel": "Optimization step", "ylabel": L.with_unit(f"E {L.MINUS} {L.SYMBOLS['E_final']}", "meV"), "energy": None, "hlines": [0.0],
            "status": end.get("category") or end.get("status"), "width": 1}


def _scf_panel(path: Path) -> Dict[str, Any]:
    from mace.utils import out_convergence as oc
    res = oc.parse(path)
    blocks = [b for b in (res.get("scf") or {}).get("detail") or [] if b.get("cycles")]
    if not blocks:
        raise GalleryError(f"{path.name} prints no SCF cycles")
    b = blocks[-1]
    pts = [(c["cyc"], abs(c["detot"])) for c in b["cycles"] if c.get("detot")]
    if not pts:
        raise GalleryError(f"{path.name} prints no SCF energy changes")
    return {"series": [{"x": [p[0] for p in pts], "y": [_r(p[1], 14) for p in pts], "name": "|ΔE|", "role": "total",
                        "markers": True}],
            "xlabel": "SCF cycle", "ylabel": "|ΔE| (Ha)", "ylog": True, "energy": None,
            "note": f"last of {len(blocks)} SCF runs" if len(blocks) > 1 else None,
            "ended": b.get("ended"), "width": 1}


_COLORS: Optional[Dict[str, str]] = None


def element_color(sym: str) -> str:
    global _COLORS
    if _COLORS is None:
        try:
            data = json.loads((Path(__file__).resolve().parents[1] / "gui" / "static" / "element_colors.json").read_text())
            _COLORS = dict(data["schemes"][data.get("default", "vesta")])
            _COLORS["_"] = data.get("fallback", "#ff1493")
        except (OSError, ValueError, KeyError):
            _COLORS = {"_": "#888888"}
    return _COLORS.get(sym, _COLORS["_"])


def _structure_panel(path: Path) -> Dict[str, Any]:
    """The atoms seen along the cell's c axis (along z for a molecule),
    drawn back to front, with the cell's outline."""
    import numpy as np
    from mace.gui import structures
    s = structures.read_structure(path)
    cart = np.asarray(s["cart"], float)
    cell = np.asarray(s["cell"], float)
    periodic = any(s.get("pbc") or [])
    if periodic:
        a, b = cell[0], cell[1]
        ex = a / (np.linalg.norm(a) or 1.0)
        n = np.cross(a, b)
        n = n / (np.linalg.norm(n) or 1.0)
        ey = np.cross(n, ex)
    else:
        ex, ey, n = np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])
    order = np.argsort(cart @ n) if len(cart) else []
    atoms = {"x": [], "y": [], "r": [], "color": [], "symbol": []}
    for i in order:
        atoms["x"].append(_r(cart[i] @ ex, 3))
        atoms["y"].append(_r(cart[i] @ ey, 3))
        atoms["r"].append(_r(s["radii"][i], 3))
        atoms["color"].append(element_color(s["symbols"][i]))
        atoms["symbol"].append(s["symbols"][i])
    outline = None
    if periodic and len(cell) >= 2:
        corners = [np.zeros(3), cell[0], cell[0] + cell[1], cell[1], np.zeros(3)]
        outline = {"x": [_r(c @ ex, 3) for c in corners], "y": [_r(c @ ey, 3) for c in corners]}
    return {"series": [], "atoms": atoms, "cell": outline, "xlabel": "", "ylabel": "", "energy": None, "equal": True,
            "formula": s.get("reduced_formula") or s.get("formula"), "width": 1}


def pane(path: Any, kind: str, ewin: Tuple[float, float] = EWINDOW, shift: float = 0.0,
         reference: str = "auto", dos_colors: str = "auto", dos_smooth: float = 0.0,
         partner: Any = None, dos_shift: float = 0.0) -> Dict[str, Any]:
    """One pane's description; an unreadable file gives ``error`` (the grid
    shows it in the pane), never an exception. ``shift``/``reference``: the
    band/DOS energy zero (mace plotting --fermi-shift/--energy-reference);
    ``dos_colors``: which DOS projections (mace plotting --dos-colors); ``dos_smooth``: the
    DOS's Gaussian smoothing in eV, 0 off (--dos-smooth). ``partner``: a band + DOS pane's DOS
    from another folder (--partner), ``dos_shift`` (eV) added to its energies (its E_F on the bands')."""
    path = Path(path)
    if kind not in KINDS:
        raise GalleryError(f"the gallery kind is one of {', '.join(KINDS)}")
    if reference not in REFERENCES:
        raise GalleryError(f"the energy reference is one of {', '.join(REFERENCES)}")
    eref = {"shift": float(shift or 0.0), "reference": reference}
    ewin = (float(ewin[0]), float(ewin[1]))
    if not ewin[0] < ewin[1]:
        raise GalleryError("the energy window's lower limit is below its upper limit")
    out: Dict[str, Any] = {"path": str(path), "name": path.name, "title": material_of(path), "kind": kind,
                           "panels": [], "error": None}
    try:
        if kind == "band":
            out["panels"] = [_band_panel(path, ewin, **eref)]
        elif kind == "dos":
            out["panels"] = [_dos_panel(path, ewin, dos_colors=dos_colors, smooth=dos_smooth, **eref)]
        elif kind == "bandos":
            out["panels"] = [_band_panel(path, ewin, **eref)]
            other = Path(partner) if partner else _partner(path, "band")
            if other is not None:
                # the DOS drawn at E - s: its energies + dos_shift are drawn with s - dos_shift
                dref = {**eref, "shift": eref["shift"] - float(dos_shift or 0.0)}
                out["panels"].append(_dos_panel(other, ewin, rotate=True, dos_colors=dos_colors, smooth=dos_smooth, **dref))
                out["panels"][1]["metal"] = out["panels"][0].get("metal", False)
                out["partner"] = str(other)
            else:
                out["note"] = "No DOSS file of this run beside it"
        elif kind in ("ir", "raman"):
            out["panels"] = [_spectrum_panel(path)]
        elif kind == "opt":
            out["panels"] = [_opt_panel(path)]
        elif kind == "scf":
            out["panels"] = [_scf_panel(path)]
        else:
            out["panels"] = [_structure_panel(path)]
    except Exception as exc:          # any reader failure: the pane says so, the others draw
        out["panels"] = []
        out["error"] = str(exc).strip().splitlines()[0][:300] if str(exc).strip() else type(exc).__name__
    return out


def titles(paths: Sequence[Any]) -> List[str]:
    """The pane titles: the material id, or the file's name where two files
    of the set share a material id (an OPT and its restart)."""
    ids = [material_of(Path(p)) for p in paths]
    out = []
    for p, mid in zip(paths, ids):
        if ids.count(mid) > 1:
            name = Path(p).name
            out.append(name if len(name) <= 48 else name[:45] + "…")
        else:
            out.append(mid)
    return out


def shared_limits(panes: Sequence[Dict[str, Any]]) -> List[Dict[str, Optional[List[float]]]]:
    """Per panel position: the union of the panes' data ranges on the axes
    the energy window does not fix (shared axes)."""
    out: List[Dict[str, Optional[List[float]]]] = []
    for pn in panes:
        for k, pl in enumerate(pn.get("panels") or []):
            while len(out) <= k:
                out.append({"x": None, "y": None})
            for ax in ("x", "y"):
                if pl.get("energy") == ax or pl.get("equal"):
                    continue
                if pl.get("dos_side"):          # a DOS axis: its largest |DOS| (dos_axis makes the range)
                    cur = out[k][ax]
                    top = float(pl.get("dos_max") or 0.0)
                    out[k][ax] = [0.0, top] if cur is None else [min(cur[0], 0.0), max(cur[1], top)]
                    continue
                vals = [v for s in pl.get("series") or [] for v in s.get(ax) or [] if v is not None]
                if pl.get(ax + "log"):
                    vals = [v for v in vals if v > 0]
                if not vals:
                    continue
                lo, hi = min(vals), max(vals)
                cur = out[k][ax]
                out[k][ax] = [lo, hi] if cur is None else [min(cur[0], lo), max(cur[1], hi)]
    return out


def dos_axis(pl: Dict[str, Any], shared: Optional[Sequence[float]] = None) -> List[float]:
    """A DOS panel's DOS-axis range (plotly_bands.dos_range's rule): the largest |DOS| drawn (the
    shared one across the grid with shared axes) plus 8%, symmetric about 0 when the spins are
    mirrored, from 0 otherwise. The GUI's gallery.js dosAxis is the same."""
    m = float(shared[1]) if shared else float(pl.get("dos_max") or 0.0)
    m = m * 1.08 if m > 0 else 1.0
    return [-m, m] if pl.get("dos_side") == "mirrored" else [0.0, m]


# ---------------------------------------------------------------- captions

def find_db(path: Path, stop: Optional[Path] = None, levels: int = 3) -> Optional[Path]:
    """The materials.db in the file's folder or up to ``levels`` folders
    above it (not above ``stop``)."""
    d = Path(path).resolve().parent
    for _ in range(levels + 1):
        c = d / "materials.db"
        if c.is_file():
            return c
        if stop is not None and d == Path(stop).resolve():
            break
        if d.parent == d:
            break
        d = d.parent
    return None


def _open_ro(db: Path) -> Optional[sqlite3.Connection]:
    try:
        conn = sqlite3.connect(f"file:{quote(str(Path(db).resolve()))}?mode=ro", uri=True, timeout=10)
        conn.execute("SELECT 1 FROM materials LIMIT 1")
        return conn
    except sqlite3.Error:
        return None


def _to_ev(name: str, value: float, unit: Optional[str]) -> float:
    u = (unit or "").strip().lower()
    if u in ("ha", "hartree", "au", "a.u.", "hartree/cell") or (not u and name.endswith("_au")):
        return value * HARTREE_EV
    return value


def _prop(conn, material: str, calc: Optional[str], names) -> Optional[float]:
    """The first of ``names`` the material has, in eV (the calculation's own
    row first, else the newest)."""
    for name in names:
        rows = conn.execute("SELECT property_value, calc_id, property_unit FROM properties WHERE material_id = ? "
                            "AND property_name = ? AND property_value IS NOT NULL ORDER BY extracted_at DESC",
                            (material, name)).fetchall()
        if not rows:
            continue
        mine = [r for r in rows if calc and r[1] == calc]
        r = (mine or rows)[0]
        return _to_ev(name, float(r[0]), r[2])
    return None


_INDEX_ROWS = 20000


def _run_key(p: Path) -> Tuple[str, str]:
    """A run's files share their folder and the name before the first dot
    (X_band.out, X_band.BAND.DAT, X_band.d3)."""
    try:
        d = str(p.parent.resolve())
    except OSError:
        d = str(p.parent)
    return d, p.name.split(".")[0]


def _run_index(conn, db: Path) -> Dict[Any, Tuple[str, str]]:
    """{run key: (calc_id, material_id)} for the calculations whose recorded
    output or input is found now (a path relative to the database's folder,
    or moved with it: db_build.file_candidates)."""
    from mace.utils.db_build import file_candidates
    out: Dict[Any, Tuple[str, str]] = {}
    try:
        rows = conn.execute("SELECT calc_id, material_id, output_file, input_file, work_dir FROM calculations "
                            "ORDER BY created_at LIMIT ?", (_INDEX_ROWS,)).fetchall()
    except sqlite3.Error:
        return out
    for calc, mat, outf, inf, wd in rows:
        for name in (outf, inf):
            for c in file_candidates(name, Path(db).parent, wd)[:4]:
                if c.is_file():
                    out[_run_key(c)] = (calc, mat)
                    break
    return out


def _lookup(index: Dict[Any, Tuple[str, str]], p: Path) -> Tuple[Optional[str], Optional[str]]:
    d, stem = _run_key(p)
    hit = index.get((d, stem))
    for tail in ("_temp", "_IRSPEC", "_RAMSPEC"):
        if hit is None and stem.endswith(tail):
            hit = index.get((d, stem[:-len(tail)]))
    return hit if hit else (None, None)


def captions(items: Sequence[Any], db: Optional[Path] = None, stop: Optional[Path] = None,
             rel=None, start: int = 1) -> Dict[str, Any]:
    """One caption row per item (``CAPTION_COLUMNS``); ``db`` is the
    database to read, else the materials.db found beside each file. ``rel``
    turns a path into the text the ``file`` column shows."""
    conns: Dict[Path, Optional[sqlite3.Connection]] = {}
    indexes: Dict[Path, Dict[Any, Tuple[str, str]]] = {}
    found: Dict[Path, Optional[Path]] = {}
    rows = []
    used = set()
    try:
        for n, raw in enumerate(items, start=start):
            p = Path(raw)
            mid = material_of(p)
            row = {"n": n, "material_id": mid, "formula": None, "band_gap_eV": None, "energy_eV": None,
                   "file": rel(p) if rel else str(p)}
            where = db
            if where is None:
                if p.parent not in found:
                    found[p.parent] = find_db(p, stop)
                where = found[p.parent]
            if where is not None:
                if where not in conns:
                    conns[where] = _open_ro(where)
                conn = conns[where]
                if conn is not None:
                    try:
                        if where not in indexes:
                            indexes[where] = _run_index(conn, where)
                        calc, known = _lookup(indexes[where], p)
                        if calc is None and p.name.lower().endswith(_OUT):
                            from mace.utils.db_build import calc_for_output
                            try:
                                calc = calc_for_output(conn, p)
                            except sqlite3.Error:
                                calc = None
                            if calc:
                                r = conn.execute("SELECT material_id FROM calculations WHERE calc_id = ?", (calc,)).fetchone()
                                known = r[0] if r else None
                        if known:
                            row["material_id"] = mid = known
                        r = conn.execute("SELECT formula FROM materials WHERE material_id = ?", (mid,)).fetchone()
                        if r:
                            row["formula"] = r[0]
                            used.add(str(where))
                        g = _prop(conn, mid, calc, _GAP_NAMES)
                        e = _prop(conn, mid, calc, _ENERGY_NAMES)
                        row["band_gap_eV"] = None if g is None else round(g, 4)
                        row["energy_eV"] = None if e is None else round(e, 6)
                        if g is not None or e is not None:
                            used.add(str(where))
                    except sqlite3.Error:
                        pass
            rows.append(row)
    finally:
        for c in conns.values():
            if c is not None:
                c.close()
    return {"columns": list(CAPTION_COLUMNS), "rows": rows, "databases": sorted(used)}


def captions_csv(rows: Sequence[Dict[str, Any]]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(CAPTION_COLUMNS)
    for r in rows:
        w.writerow(["" if r.get(c) is None else r.get(c) for c in CAPTION_COLUMNS])
    return buf.getvalue()


# ---------------------------------------------------------------- the figure

def _mpl_label(s: str) -> str:
    return L.mathtext(s)


def _mpl_rgba(color: str):
    """presets.rgba()'s "rgba(r,g,b,a)" as matplotlib's (r, g, b, a); another color as it is."""
    m = re.match(r"rgba\((\d+),(\d+),(\d+),([\d.]+)\)$", str(color).replace(" ", ""))
    if not m:
        return color
    return (int(m[1]) / 255, int(m[2]) / 255, int(m[3]) / 255, float(m[4]))


def _draw_panel(ax, pl: Dict[str, Any], shared: Optional[Dict[str, Any]], small: bool, style: str = "default",
                spin_look: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """One panel in the plot style ``style`` (presets.py): spin up solid and spin down in its line
    style in the spin pair, DOS projections in their DOS colors (the GUI's Gallery draws the same).
    A DOS panel in the 1.1.6 DOS layout (plotly_bands.band_figure): spin down mirrored with
    magnitudes on both sides and "↑ up"/"down ↓" labels, the totals filled in the spin pair (a
    closed-shell total as a neutral fill under an ink line), projection colors clear of the spin
    pair, spin down dashed, a DOS = 0 baseline, a dotted E_F and the no-data band shaded.
    ``spin_look``: the spin scheme, custom colors, spin down's line style and the closed-shell band
    color (--spin-colors, --spin-up-color, --spin-down-color, --spin-down-style, --band-color).
    Returns what the legend names (_legend): {"spin2", "closed", "keys": [(name, fill, edge)], "ink", "bands", "dash"}."""
    import numpy as np
    from . import presets
    sl = spin_look or {}
    family = "band" if pl.get("energy") else "spectra" if str(pl.get("xlabel") or "").startswith("Wavenumber") else "scf"
    spin = presets.spin_colors(style, "band", scheme=sl.get("spin_colors"), up=sl.get("spin_up_color"),
                               down=sl.get("spin_down_color"))
    down_dash = sl.get("spin_down_style") or presets.SPIN_DOWN_DASH
    ink = "#000000" if style in ("publication", "grayscale") else "#222222"
    side = pl.get("dos_side")
    both = side == "mirrored"
    look = None
    projs = [s["proj"] for s in pl.get("series") or [] if s.get("proj") and s.get("spin") == "up"]
    if projs:
        look = {x["label"]: x for x in presets.dos_styles(projs, pl.get("dos_mode") or "auto", presets.element_colors(),
                                                           preset=style, spin=bool(pl.get("dos_spin")),
                                                           weights=pl.get("dos_weights"))["lines"]}
        # DOS colors Auto: the preset's projection colors by weight, clear of the spin pair when both spins are drawn
        if side and (pl.get("dos_colors") or "auto") == "auto":
            ppal = presets.projection_palette(style, [spin["up"], spin["down"]] if both else None)
            for i, lab in enumerate(pl.get("dos_order") or [] if ppal else []):
                if lab in look:
                    look[lab] = {**look[lab], "color": ppal[i % len(ppal)], "dash": "solid"}
    # a closed-shell run's bands and DOS total, by --band-style as the plotly figure draws them (plotly_bands.look_parts):
    # single: the pair's first color (or --band-color), overlay: the first color with the second dashed on top,
    # ink: the preset's neutral color; auto: overlay for a metal, else single
    bstyle = presets.band_style(sl.get("band_style"), {"fermi": 0, "up": [[-1.0, 1.0]]} if pl.get("metal") else None)
    custom = sl.get("band_color")
    if bstyle == "ink":
        bands = presets.band_color(style, "band", "light", None, ink)
    elif bstyle == "single" and custom and presets._HEXCOLOR.match(custom):
        bands = custom.lower()
    else:
        bands = spin["up"]
    fills = presets.spin_fills(spin, style)
    nfill = presets.neutral_fill(bands, style)
    rot = pl.get("energy") == "y"
    keys: List[Tuple[str, Any, Any]] = []
    k = 0
    for s in pl.get("series") or []:
        x = np.array([np.nan if v is None else v for v in s["x"]], float)
        y = np.array([np.nan if v is None else v for v in s["y"]], float)
        role, ls = s.get("role"), "-"
        down = s.get("spin") == "down" or role == "down"
        fill = None
        if role == "projection" and look and s.get("proj") in look:
            c, ls = look[s["proj"]]["color"], presets.mpl_dash(look[s["proj"]]["dash"])
        elif role == "projection":
            c = presets.series_style(style, family, k)["color"]
            k += 1
        elif role == "up" and pl.get("closed"):
            c = bands
        elif role in ("up", "down"):
            c = spin[role]
        elif side:                      # a DOS total: in the spin pair and filled, or neutral (closed shell)
            c = (spin["down"] if down else spin["up"]) if both else bands
            fill = (fills["down"] if down else fills["up"]) if both else nfill
        else:
            c = ink
        if down:
            ls = presets.mpl_dash(down_dash) if ls == "-" else ls
        if fill:
            vals = np.nan_to_num(x if rot else y)
            band_fill = ax.fill_betweenx(y, 0, vals) if rot else ax.fill_between(x, 0, vals)
            band_fill.set(color=_mpl_rgba(fill), linewidth=0, zorder=1)
        if side and not down and role in ("total", "projection"):
            keys.append((s.get("name") or "", _mpl_rgba(presets.rgba(ink, 0.12) if both else presets.rgba(bands, 0.22))
                         if role == "total" else c, ("#888888" if both else bands) if role == "total" else c))
        ax.plot(x, y, color=c, lw=0.7 if role in ("up", "down") else 1.0 if role == "total" and side else 0.9,
                linestyle=ls, marker="o" if s.get("markers") else None, ms=2.5, zorder=2)
        if role == "up" and pl.get("closed") and bstyle == "overlay":      # the second color dashed on top
            ax.plot(x, y, color=spin["down"], lw=0.7, linestyle=presets.mpl_dash("dash" if down_dash == "solid" else down_dash),
                    zorder=2.1)
    at = pl.get("atoms")
    if at and at["x"]:
        r = np.array(at["r"], float)
        ax.scatter(at["x"], at["y"], s=(r * 12) ** 2 + 4, c=at["color"], edgecolors="#333333", linewidths=0.3,
                   zorder=3)
        if pl.get("cell"):
            ax.plot(pl["cell"]["x"], pl["cell"]["y"], color="#888888", lw=0.6, zorder=1)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xticks([])
        ax.set_yticks([])
    for v in pl.get("hlines") or []:
        ax.axhline(v, color="#999999", lw=0.5, ls="--", zorder=0)
    for v in pl.get("eflines") or []:       # E_F dotted: dashed reads as spin down
        (ax.axhline if pl.get("energy") == "y" else ax.axvline)(v, color="#666666", lw=0.7, ls=":", zorder=1.5)
    if pl.get("baseline") is not None:
        (ax.axvline if rot else ax.axhline)(pl["baseline"], color=ink, lw=0.8, zorder=1.5)
    for g0, g1 in pl.get("nodata") or []:
        (ax.axhspan if rot else ax.axvspan)(g0, g1, color=ink, alpha=0.05, lw=0, zorder=0)
        win = pl.get("xlim") or [g0, g1]
        if not rot and not small and g1 - g0 >= 0.15 * abs(win[1] - win[0]):
            ax.text((g0 + g1) / 2, 0.5, presets.DOS_LAYOUT["no_data"], rotation=90, ha="center", va="center",
                    fontsize=5, color="#888888", transform=ax.get_xaxis_transform())
    for v in pl.get("vlines") or []:
        ax.axvline(v, color="#bbbbbb", lw=0.4, zorder=0)
    if pl.get("ylog"):
        ax.set_yscale("log")
    if pl.get("xlog"):
        ax.set_xscale("log")
    for ax_name in ("x", "y"):
        lim = pl.get(ax_name + "lim")
        if shared and shared.get(ax_name) and pl.get("energy") != ax_name:
            lim = shared[ax_name]
        if side and pl.get("energy") != ax_name:            # the DOS axis: symmetric when mirrored, |DOS| ticks
            lim = dos_axis(pl, (shared or {}).get(ax_name))
            if both:
                tk = presets.mirrored_ticks(lim[0], lim[1])
                (ax.set_xticks if ax_name == "x" else ax.set_yticks)(tk["vals"])
                (ax.set_xticklabels if ax_name == "x" else ax.set_yticklabels)(tk["text"])
        if lim and lim[0] != lim[1]:
            (ax.set_xlim if ax_name == "x" else ax.set_ylim)(lim[0], lim[1])
    fs = 5 if small else 6
    if both:                                # each spin's half named over it, in its color
        if rot:
            tr = ax.get_xaxis_transform()
            ax.text(0, 1.01, " ↑ up", ha="left", va="bottom", fontsize=fs, color=spin["up"], fontweight="bold", transform=tr)
            ax.text(0, 1.01, "down ↓ ", ha="right", va="bottom", fontsize=fs, color=spin["down"], fontweight="bold",
                    transform=tr)
        else:
            ax.text(0.01, 0.97, "↑ up", ha="left", va="top", fontsize=fs, color=spin["up"], fontweight="bold",
                    transform=ax.transAxes)
            ax.text(0.01, 0.03, "down ↓", ha="left", va="bottom", fontsize=fs, color=spin["down"], fontweight="bold",
                    transform=ax.transAxes)
    if side and pl.get("dos_smooth"):
        ax.text(0.99, 0.01, f"smoothed, σ = {presets.js_num(pl['dos_smooth'])} eV", ha="right", va="bottom", fontsize=fs,
                color="#666666", transform=ax.transAxes)
    if pl.get("ticks"):
        ax.set_xticks(pl["ticks"])
        labels = [_mpl_label(t) for t in pl.get("ticklabels") or []]
        # a long path (a molecule's, every point a tick) has no room for its labels in a pane
        ax.set_xticklabels(labels if labels and len(labels) <= 12 else [""] * len(pl["ticks"]),
                           fontsize=5 if small else 7)
    ax.set_xlabel(_mpl_label(pl.get("xlabel") or ""), fontsize=6 if small else 8)
    ax.set_ylabel(_mpl_label(pl.get("ylabel") or ""), fontsize=6 if small else 8)
    ax.tick_params(labelsize=5 if small else 7, length=2)
    spin2 = both if side else (not pl.get("closed") if pl.get("energy") == "y" and "closed" in pl else None)
    return {"spin2": spin2, "identical": bool(pl.get("spin_identical")), "keys": keys, "ink": ink, "bands": bands,
            "up": spin["up"], "down": spin["down"], "dash": down_dash}


def _legend(fig, ax, infos: Sequence[Dict[str, Any]], band_dos: bool, small: bool) -> None:
    """One pane's legend (plotly_bands' two titled groups): Spin (in the colors drawn, spin up solid and spin
    down in its line style, or the one closed-shell key) and Projections (color swatches); in the
    right margin beside a band + DOS pane, inside a DOS-alone pane where it covers no curve
    (presets.legend_spot), else outside."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from . import presets
    from .plotly_bands import spin_keys
    spin2 = [i["spin2"] for i in infos if i["spin2"] is not None]
    if not spin2:
        return
    first = infos[0]
    words = "α = β (non-magnetic)" if any(i["identical"] for i in infos) else "α = β (closed shell)"
    handles: List[Any] = [Patch(visible=False)]
    labels = ["Spin"]
    for kk in spin_keys(any(spin2), {"spin": "both"}, first["dash"], words, first):
        handles.append(Line2D([], [], color=kk.get("color") or first["ink"], lw=1.4,
                              linestyle=presets.mpl_dash(kk["dash"])))
        labels.append(kk["name"])
    keys = [k for i in infos for k in i["keys"]]
    if keys:
        handles.append(Patch(visible=False))
        labels.append("Projections")
        for name, fill, edge in keys:
            handles.append(Patch(facecolor=fill, edgecolor=edge, lw=0.6))
            labels.append(name)
    fs = 5 if small else 6.5
    kw = {"fontsize": fs, "frameon": True, "framealpha": 0.88, "handlelength": 1.8, "borderaxespad": 0.3}
    if band_dos:
        lg = ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.03, 1.0), **kw)
    else:
        lines = [(ln.get_xdata(), ln.get_ydata()) for ln in ax.get_lines() if len(ln.get_xdata()) > 2]
        box = {"fw": min(0.9, (46 + 6.6 * max(len(x) for x in labels)) / 260), "fh": min(0.9, (10 + 19 * len(labels)) / 220)}
        spot = presets.legend_spot([([None if v != v else float(v) for v in x], [None if v != v else float(v) for v in y])
                                    for x, y in lines], ax.get_xlim(), ax.get_ylim(), box["fw"], box["fh"])
        if spot == "outside":
            lg = ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0), **kw)
        else:
            lg = ax.legend(handles, labels, loc="upper right" if spot == "top" else "lower right", **kw)
    for t in lg.get_texts():
        if t.get_text() in ("Spin", "Projections"):
            t.set_fontweight("bold")
            t.set_color("#555555")


def render_figure(panes: Sequence[Dict[str, Any]], out: Any, cols: int = 3, shared_axes: bool = True,
                  title: Optional[str] = None, dpi: int = 150, style: str = "default",
                  size: Optional[Tuple[Optional[float], Optional[float]]] = None,
                  spin_look: Optional[Dict[str, Any]] = None, legend: bool = False, start: int = 1) -> Path:
    """The grid as one figure (SVG, PNG or PDF by the file's suffix): pane
    *n* titled "n. material id" (counted from ``start``: a later page goes on from the
    page before, as its captions), an unreadable one saying why. ``size``: (width, height) in
    inches, either None for the grid's own (mace plotting --width/--height). ``legend``: each
    band/DOS pane with its grouped legend (_legend; the gallery's grid has none, as in the GUI)."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(out)
    fmt = out.suffix.lower().lstrip(".")
    if fmt not in FIGURE_FORMATS:
        raise GalleryError(f"the figure is one of {', '.join('.' + f for f in FIGURE_FORMATS)}")
    if not panes:
        raise GalleryError("nothing to draw")
    cols = max(1, min(int(cols), MAX_COLS, len(panes)))
    rows = int(math.ceil(len(panes) / cols))
    small = cols >= 4
    w, h = 3.4, 2.6
    fw, fh = cols * w, rows * h + (0.4 if title else 0)
    if size:
        fw, fh = size[0] or fw, size[1] or fh
    fig = plt.figure(figsize=(fw, fh))
    gs = fig.add_gridspec(rows, cols, hspace=0.55, wspace=0.35)
    lims = shared_limits(panes) if shared_axes else []
    for i, pn in enumerate(panes):
        cell = gs[i // cols, i % cols]
        panels = pn.get("panels") or []
        head = f"{i + start}. {pn.get('title') or pn.get('name')}"
        if not panels:
            ax = fig.add_subplot(cell)
            ax.set_axis_off()
            ax.text(0.5, 0.5, pn.get("error") or "Nothing to draw", ha="center", va="center", wrap=True, fontsize=7,
                    color="#a33", transform=ax.transAxes)
            ax.set_title(head, fontsize=7 if small else 9)
            continue
        sub = cell.subgridspec(1, len(panels), width_ratios=[p.get("width", 1) for p in panels], wspace=0.05)
        first = None
        infos = []
        for k, pl in enumerate(panels):
            ax = fig.add_subplot(sub[0, k], sharey=first if (first is not None and pl.get("energy") == "y") else None)
            infos.append(_draw_panel(ax, pl, lims[k] if k < len(lims) else None, small, style, spin_look))
            if k == 0:
                first = ax
                ax.set_title(head, fontsize=7 if small else 9, loc="left")
            elif pl.get("energy") == "y":
                ax.tick_params(labelleft=False)
                ax.set_xticks([])
        if legend:
            _legend(fig, ax, infos, len(panels) > 1, small)
    if title:
        fig.suptitle(title, fontsize=11)
    out.parent.mkdir(parents=True, exist_ok=True)
    # text stays text in an SVG (editable, searchable), whatever the matplotlibrc says
    with matplotlib.rc_context({"svg.fonttype": "none"}):
        fig.savefig(out, format=fmt, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def command(kind: str, inputs: Sequence[str], cols: int, out: str, shared_axes: bool = True,
            ewin: Tuple[float, float] = EWINDOW, caption_csv: Optional[str] = None,
            shift: float = 0.0, reference: str = "auto", style: str = "default",
            dos_colors: str = "auto", spin_colors: str = "auto", spin_up_color: Optional[str] = None,
            spin_down_color: Optional[str] = None, spin_down_style: str = "dash", dos_smooth: float = 0.0,
            band_color: Optional[str] = None, explicit: bool = False, page: int = 1, page_size: int = 0,
            band_style: Optional[str] = None) -> List[str]:
    """The ``mace plotting --gallery`` arguments (after ``mace``) for these
    choices: the GUI shows it and runs it for Export (in the GUI's plot style,
    DOS colors and spin look, when they are not the defaults). ``explicit``:
    the defaults too, so a mace-folder.toml's [plots] values do not replace
    them (the GUI's grid does not read the file's values the CLI would)."""
    a = ["plotting", "--gallery", kind, *[str(x) for x in inputs], "--cols", str(int(cols)), "-o", str(out)]
    if page_size and int(page_size) != MAX_ITEMS:
        a += ["--max", str(int(page_size))]
    if page and int(page) > 1:
        a += ["--page", str(int(page))]
    if not shared_axes:
        a.append("--independent-axes")
    if (style and style != "default") or explicit:
        a += ["--style", style or "default"]
    if kind in ("dos", "bandos") and ((dos_colors and dos_colors != "auto") or explicit):
        a += ["--dos-colors", dos_colors or "auto"]
    if kind in ("dos", "bandos") and (dos_smooth or explicit):
        from . import presets
        if presets.smooth_sigma(dos_smooth) or explicit:
            a += [f"--dos-smooth={presets.js_num(presets.smooth_sigma(dos_smooth))}"]
    if kind in ("band", "dos", "bandos"):
        from . import presets
        spin = presets.style_flags({"spin_colors": spin_colors or "auto", "spin_up_color": spin_up_color,
                                    "spin_down_color": spin_down_color, "spin_down_style": spin_down_style or "dash",
                                    "band_color": band_color, "band_style": band_style or "auto"}, explicit=explicit)
        a += [f for f in spin if f.startswith(("--spin-colors", "--spin-up-color", "--spin-down-color",
                                                "--spin-down-style", "--spin-down-solid", "--band-color", "--band-style"))]
    if kind in ("band", "dos", "bandos"):            # always: the line reproduces the figure in any folder
        a += [f"--e-lower={float(ewin[0]):g}", f"--e-upper={float(ewin[1]):g}"]
        if shift or explicit:
            a.append(f"--fermi-shift={float(shift or 0):g}")
        if reference != "auto" or explicit:
            a.append(f"--energy-reference={reference}")
    if caption_csv:
        a += ["--caption-csv", str(caption_csv)]
    return a


def run_cli(args) -> int:
    """``mace plotting --gallery KIND FILES/FOLDER``: the grid as one figure
    (``-o``: a file, or a folder for gallery_<kind>.svg) and the caption
    table on screen (``--caption-csv`` writes it)."""
    import sys
    kind = args.gallery
    inputs = list(getattr(args, "files", None) or []) or [getattr(args, "directory", ".") or "."]
    missing = [x for x in inputs if not Path(x).exists()]
    if missing:
        print(f"❌ Not found: {', '.join(missing)}", file=sys.stderr)
        return 2
    cap = getattr(args, "max", None) or MAX_ITEMS
    got = collect(inputs, kind, cap=cap, page=getattr(args, "page", None) or 1)
    if got["message"]:
        print(f"  {got['message']}")
    if not got["items"]:
        return 1
    o = Path(getattr(args, "output", ".") or ".")
    fmt = getattr(args, "format", None)
    if o.suffix.lower().lstrip(".") in FIGURE_FORMATS:
        target = o
    else:
        target = o / f"gallery_{kind}.{fmt if fmt in FIGURE_FORMATS else 'svg'}"
    ewin = (args.e_lower, args.e_upper) if getattr(args, "e_lower", None) is not None else EWINDOW
    panes = []
    for n, (p, t) in enumerate(zip(got["items"], titles(got["items"])), start=got["offset"] + 1):
        pn = pane(p, kind, ewin, getattr(args, "fermi_shift", 0.0) or 0.0,
                  getattr(args, "energy_reference", None) or "auto",
                  dos_colors=getattr(args, "dos_colors", None) or "auto",
                  dos_smooth=getattr(args, "dos_smooth", None) or 0.0)
        pn["title"] = t
        if pn["error"]:
            print(f"  ⚠️  {n}. {p.name}: {pn['error']}")
        panes.append(pn)
    cols = getattr(args, "cols", None) or min(3, len(panes))
    spin_look = {"spin_colors": getattr(args, "spin_colors", None), "spin_up_color": getattr(args, "spin_up_color", None),
                 "spin_down_color": getattr(args, "spin_down_color", None),
                 "spin_down_style": getattr(args, "spin_style", None), "band_color": getattr(args, "band_color", None),
                 "band_style": getattr(args, "band_style", None)}
    render_figure(panes, target, cols=cols, shared_axes=not getattr(args, "independent_axes", False),
                  title=f"{LABELS[kind]} ({len(panes)} files)", style=getattr(args, "style", None) or "default",
                  spin_look=spin_look,
                  size=(getattr(args, "width", None), getattr(args, "height", None)), start=got["offset"] + 1)
    cap_rows = captions(got["items"], db=Path(args.db) if getattr(args, "db", None) else None, start=got["offset"] + 1)
    print(f"\n  {'#':>3}  {'Material':<40} {'Formula':<14} {'Gap (eV)':>9} {'Energy (eV)':>16}")
    for r in cap_rows["rows"]:
        print(f"  {r['n']:>3}  {str(r['material_id'])[:40]:<40} {str(r['formula'] or '—')[:14]:<14} "
              f"{'—' if r['band_gap_eV'] is None else format(r['band_gap_eV'], '.3f'):>9} "
              f"{'—' if r['energy_eV'] is None else format(r['energy_eV'], '.4f'):>16}")
    if getattr(args, "caption_csv", None):
        Path(args.caption_csv).write_text(captions_csv(cap_rows["rows"]))
        print(f"\n  Captions: {args.caption_csv}")
    print(f"\n  Gallery: {target} ({len(panes)} panes, {min(int(cols), MAX_COLS)} columns)")
    return 0
