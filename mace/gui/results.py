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
  reported closed-shell and only one set is sent; beta bands within 10 meV of
  alpha (a non-magnetic spin-polarized run) are sent once too, flagged
  ``spin_identical``;
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
    os.environ["MPLBACKEND"] = "Agg"        # the server never draws through X (DISPLAY may be X11 forwarding)
    if str(_PLOTTING) not in sys.path:
        sys.path.insert(0, str(_PLOTTING))
    import ipBANDS_V2
    import ipDOS_V2
    return ipBANDS_V2, ipDOS_V2


def _stem(name: str, suffixes) -> str:
    if name.lower().endswith((".f25", ".fort.25")):          # a fort.25: its name without the extension
        return name[:-4] if name.lower().endswith(".f25") else name[:-8]
    for suffix in suffixes:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def _material(stem: str, tag: str) -> str:
    return stem[: -len(tag)] if stem.endswith(tag) else stem


def _potc(directory: Path, material: str, reference: str = "auto") -> Tuple[str, str, bool]:
    """The run's POTC files and whether the energies go on the vacuum scale: ``reference``
    auto (when both files are there), fermi (never) or vacuum (both files needed: mace
    plotting --energy-reference)."""
    dat = directory / f"{material}_POTC.POTC.DAT"
    out = directory / f"{material}_POTC.out"
    have = dat.is_file() and out.is_file()
    if reference == "vacuum" and not have:
        raise ValueError(f"--energy-reference vacuum needs the run's POTC files beside the data ({dat.name} and {out.name})")
    return str(dat), str(out), have and reference != "fermi"


def _plotly_label(label: str) -> str:
    """'$Σ_{0}$' (matplotlib mathtext) -> 'Σ<sub>0</sub>' (plotly)."""
    label = label.replace("$", "")
    return re.sub(r"_\{([^}]*)\}", r"<sub>\1</sub>", label)


def _with_breaks(x: np.ndarray, ys: np.ndarray, after: List[int]):
    """Insert a null after each point in ``after`` (a discontinuous tick, see
    ``ipBANDS_V2.path_break_indices``) so no line joins the two sides."""
    cut = [i + 1 for i in after if 0 <= i < len(x) - 1]
    if not cut:
        return x.tolist(), ys.T.tolist()
    xs = np.insert(x.astype(float), cut, np.nan)
    yy = np.insert(ys.astype(float), cut, np.nan, axis=0)
    clean = lambda a: [None if np.isnan(v) else float(v) for v in a]  # noqa: E731
    return clean(xs), [clean(col) for col in yy.T] if yy.ndim == 2 else clean(yy)


def _near_count(bands: np.ndarray, fermi: float) -> int:
    """The bands that come within WINDOW_EV of E_F (what _window keeps for the GUI before its cap)."""
    mx, mn = bands.max(axis=0), bands.min(axis=0)
    return int(np.count_nonzero((mx > fermi - WINDOW_EV) & (mn < fermi + WINDOW_EV)))


def _window(bands: np.ndarray, fermi: float, window: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, bool]:
    """The bands to send. The GUI (``window`` None): those that come within WINDOW_EV of E_F, at most
    MAX_BANDS (a browser's cost), the ones that come nearest E_F first. mace plotting (``window``: the
    requested energies, eV around ``fermi`` and the Fermi shift): every band that enters it or the GUI's
    +/-WINDOW_EV, no cap (a file has no browser cost; the 400 cap once dropped bands inside a wide
    window, 1.1.6 review F4-4)."""
    lo, hi = fermi - WINDOW_EV, fermi + WINDOW_EV
    if window is not None:      # the requested window, and the GUI's +/-25 eV (an HTML figure zooms out)
        lo, hi = min(lo, fermi + window[0]), max(hi, fermi + window[1])
    mx, mn = bands.max(axis=0), bands.min(axis=0)
    keep = [j for j in range(bands.shape[1]) if mx[j] > lo and mn[j] < hi]
    truncated = window is None and len(keep) > MAX_BANDS
    if truncated:   # the bands that come nearest the Fermi level (0 for one that crosses it)
        near = np.where((mn <= fermi) & (mx >= fermi), 0.0, np.minimum(abs(mn - fermi), abs(mx - fermi)))
        keep = sorted(sorted(keep, key=lambda j: near[j])[:MAX_BANDS])
    return bands[:, keep], truncated


def _fermi_note(path: Path) -> str:
    """What the band plot says when CRYSTAL wrote no Fermi level (-1.0, a spin-locked SCF): the one
    mace.utils.fermi_level derived from the bands, or that there is none."""
    from mace.utils import fermi_level
    try:
        fl = fermi_level.band_fermi(path)
    except Exception:
        return ""
    return fl.note if fl.derived or (not fl.known and fermi_level.is_placeholder(fl.file_au)) else ""


def _from_f25(path: Path, kind: str, reference: str, window, domain: str, run: int) -> Dict[str, Any]:
    """A fort.25's BAND or DOSS blocks as the data of a BAND.DAT/DOSS.DAT of the same run: converted into a
    scratch folder beside copies of the run's small files (mace/plotting/f25.py) and read by the same code."""
    import tempfile
    from mace.plotting import f25
    info = f25.describe(path, domain)
    with tempfile.TemporaryDirectory(prefix="mace-f25-") as tmp:
        dat, _ = f25.materialize(path, tmp, "BAND" if kind == "band" else "DOSS", domain, run=run)
        data = band_data(dat, reference, window) if kind == "band" else dos_data(dat, reference)
    data["name"] = path.name
    data["source"] = "f25"
    note = _fermi_note(path) if kind == "band" else ""
    if note:                  # the converted DAT holds the corrected E_F: the note is the f25's
        data["fermi_note"] = note
    data["n_runs"] = info.get("n_runs", 1)
    return data


def band_data(path: Path, reference: str = "auto", window: Optional[Tuple[float, float]] = None,
              domain: str = "auto", run: int = -1) -> Dict[str, Any]:
    """The band plot's data. ``window`` (eV around E_F, Fermi shift included): mace plotting's
    requested energies, every band in it (see _window); None: the GUI's. A fort.25 (``*.f25``) is read
    through its blocks (``domain``: electronic or phonon, auto from the run; ``run``: which of the runs
    the file holds, the last by default)."""
    path = Path(path)
    from mace.plotting import f25 as _f25
    if _f25.is_f25_name(path.name):
        return _from_f25(path, "band", reference, window, domain, run)
    ipb, _ = _plotters()
    stem = _stem(path.name, BAND_SUFFIXES)
    material = _material(stem, "_band")
    dat, out, has_potc = _potc(path.parent, material, reference)
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
    if not labels and _domain(path) == "phonon":        # a phonon fort.25 has no .d3: the DAT's panel labels
        from mace.plotting import f25
        labels = [_plotly_label(lab) for lab in f25.dat_labels(path)]
    if len(labels) != len(ticks):
        labels = labels[: len(ticks)] + [""] * max(0, len(ticks) - len(labels))
    breaks = ipb.path_break_indices(E[:n], ticks, labels)
    state, spin_diff = ipb.spin_state(B, Bb)
    spin = state == "polarized"
    fermi = float(maxV)
    up, trunc_up = _window(B, fermi, window)
    # the GUI's bands that never come within WINDOW_EV of E_F (not loaded; a
    # view of a wider window notes them, mace plotting draws them)
    beyond = 0 if window is not None else int(B.shape[1] - _near_count(B, fermi))
    x, up_cols = _with_breaks(E[:n], up[:n], breaks)
    result = {
        "kind": "band", "name": path.name, "x": x, "up": up_cols,
        "down": None, "spin_polarized": spin, "fermi": fermi,
        # a spin-polarized run whose beta bands match alpha (non-magnetic): one set
        "spin_identical": state == "identical", "spin_max_diff": spin_diff,
        "reference": "vacuum" if has_potc else "fermi",
        "ticks": ticks, "labels": labels, "truncated": trunc_up,
        "n_kpoints": n, "n_bands": int(B.shape[1]),
        "window_ev": WINDOW_EV, "n_beyond": beyond,
        "domain": _domain(path), "has_potc": Path(dat).is_file() and Path(out).is_file(),
    }
    note = _fermi_note(path)
    if note:
        result["fermi_note"] = note
    if spin:
        down, trunc_down = _window(Bb, fermi, window)
        _, result["down"] = _with_breaks(Eb[:n], down[:n], breaks)
        result["truncated"] = trunc_up or trunc_down
    return result


def _doss_header(path: Path) -> Tuple[Optional[int], int]:
    _, ipd = _plotters()
    return ipd.read_doss_header(str(path))


def _domain(dat: Path) -> str:
    """"phonon" for the data of a phonon fort.25 (its converted DAT says so), else "electronic"."""
    from mace.plotting import f25
    return f25.dat_domain(dat)


def dos_data(path: Path, reference: str = "auto", domain: str = "auto", run: int = -1) -> Dict[str, Any]:
    path = Path(path)
    from mace.plotting import f25 as _f25
    if _f25.is_f25_name(path.name):
        return _from_f25(path, "dos", reference, None, domain, run)
    _, ipd = _plotters()
    stem = _stem(path.name, DOS_SUFFIXES)
    material = _material(stem, "_doss")
    dat, out, has_potc = _potc(path.parent, material, reference)
    nproj, nspin = _doss_header(path)
    labels: Optional[List[str]] = None
    # the run's DOSS deck: the same candidates as the Related view's projections (related.py), so a
    # DOSS.DAT under a generic name (fort.25-style folders) still names its projections ("Mg s", "O p")
    from . import related
    cands = [path.parent / f"{stem}.d3", path.parent / f"{material}_doss.d3"]
    for st in related.stems(path):
        cands += [path.parent / f"{st}.d3", path.parent / f"{related.material_of(st)}_doss.d3"]
    for cand in dict.fromkeys(cands):
        if cand.is_file():
            labels = [ipd.doss_label(r).strip() for r in ipd.read_doss_projection_labels(str(cand))]
            # a projection on every orbital of an element reads "C" in the
            # .d3: in a legend beside "C (s)", "C (p)" that looks like a typo
            labels = [f"{lab} (all)" if re.fullmatch(r"[A-Z][a-z]?", lab) else lab
                      for lab in labels] or None
            # a deck of another kind (no DOSS block) or of another run (another count): the next one
            if labels and (nproj is None or len(labels) == nproj):
                break
            labels = None
    if labels is None or (nproj is not None and len(labels) != nproj):
        if nproj is None:
            raise ValueError(f"{path.name}: no NPROJ in the header and no .d3 beside it")
        labels = [f"projection {k}" for k in range(1, nproj)] + ["Total"]
    maxV = ipd.vacuum_shift(dat, out) if has_potc else 0
    cols = ipd.read_doss_columns(str(path), len(labels) + 1, maxV)
    n_all = len(cols[0])
    half = n_all // 2 if nspin == 2 and n_all % 2 == 0 else n_all
    # the bin at a window edge that holds the states outside the window (same as mace plotting --dos)
    cols, pileup = ipd.suppress_edge_pileup(cols, n_all // half)
    cols = [np.asarray(c, float) for c in cols]
    energy = cols[0][:half]
    up = [c[:half] for c in cols[1:]]
    down = [c[half:] for c in cols[1:]] if half < n_all else None
    # one rule with mace plotting (presets.dos_spin_polarized, B2): spin-polarized unless spin down is spin
    # up's mirror image (a closed-shell run, which CRYSTAL writes as two blocks too)
    from mace.plotting.presets import dos_spin_polarized
    spin = dos_spin_polarized(up, down)
    # an unrestricted run with alpha = beta (MACE's SPIN default makes most runs unrestricted): each
    # block holds one spin, so the one-sided curve is their sum, alpha + |beta| (the total DOS); a
    # mirrored view draws half of it on each side (bandfig.js upOf/downOf)
    summed = down is not None and not spin
    if summed:
        up = [u - d for u, d in zip(up, down)]
    # a band-index DOS (a 1D material's default) that stops below E_F + 15 eV (mace.utils.doss_reach)
    from mace.utils import doss_reach
    reach = doss_reach.cached_note(path)
    return {
        "reach_note": reach, "notes": [reach] if reach else [],
        "kind": "dos", "name": path.name, "energy": energy.tolist(), "domain": _domain(path),
        "has_potc": Path(dat).is_file() and Path(out).is_file(),
        "fermi": float(maxV), "reference": "vacuum" if has_potc else "fermi",
        "spin_polarized": spin, "spin_summed": summed,
        # the computed DOS window: no data beyond it (the plot shades that region)
        "window": [float(energy[0]), float(energy[-1])] if len(energy) else None,
        "edge_pileup": int(pileup),
        "projections": [{"label": lab, "up": u.tolist(),
                         "down": (d.tolist() if spin else None)}
                        for lab, u, d in zip(labels, up, down or up)],
    }


def _d3_tools():
    d3 = Path(__file__).resolve().parents[2] / "Crystal_d3"
    if str(d3) not in sys.path:
        sys.path.insert(0, str(d3))
    import d3_band_path
    import d3_bz
    return d3_band_path, d3_bz


def band_path_errors(band_path: Any):
    """(errors, warnings) of an edited k-path (d3_band_path.validate)."""
    kp, _ = _d3_tools()
    return kp.validate(band_path)


def bz_data(path: Path, inside=None) -> Dict[str, Any]:
    """The Brillouin zone view's data (Crystal_d3/d3_bz.py): the zone of the run's
    cell, its high-symmetry points and the computed path. ``path`` is a BAND.DAT
    (the cell from the run's .out beside it, the path from its .d3) or a CRYSTAL
    output (the cell; the path comes from the deck the editor has). ``inside``:
    whether a file beside it may be read (the GUI: inside the served folder)."""
    path = Path(path)
    ok = inside or (lambda f: True)
    kp, bz = _d3_tools()
    out = bz.output_for(str(path))
    if out is not None and out != path and not ok(out):
        out = None
    if out is None:
        raise FileNotFoundError(f"no CRYSTAL output beside {path.name} to read the cell from")
    segments: List[Dict[str, Any]] = []
    if path.suffix != ".out":
        # the run's path: its .d3, else a fort.25's segment records (a phonon run's: its FREQCALC BANDS)
        info = kpath_info(path, inside)
        segments = [s for s in info["segments"] if s.get("k_from") is not None] if info else []
    data = bz.bz_view(str(out), segments)
    data["output"] = out.name
    return data


def _band_deck(path: Path, ok) -> Optional[Path]:
    stem = _stem(path.name, BAND_SUFFIXES)
    material = _material(stem, "_band")
    for cand in (path.parent / f"{stem}.d3", path.parent / f"{material}_band.d3", path.parent / f"{material}.d3"):
        if cand.is_file() and ok(cand):
            return cand
    return None


_DAT_POINT = re.compile(r"^#\s+(\d+)\s+\((-?\d+),(-?\d+),(-?\d+)\)/(\d+)")


def kpath_info(path: Path, inside=None) -> Optional[Dict[str, Any]]:
    """The k-path a band structure was computed along, for the Inspector: {source (the file it is read from),
    title, iss (the shrinking factor the segment ends are integers of), points (k points along the path),
    lines (segments), segments [{from, to, k_from, k_to (fractional, None when unknown)}], domain}. Read from
    the run's BAND deck (.d3), else a fort.25's segment records (a phonon run's ISS from its FREQCALC BANDS in
    the .d12 beside it), else a BAND.DAT's header; None when the file says nothing about its path."""
    from mace.plotting import f25
    path = Path(path)
    ok = inside or (lambda f: True)
    kp, _ = _d3_tools()
    deck = path if path.suffix == ".d3" else _band_deck(path, ok)
    if deck is not None:
        parsed = kp.parse_band_deck(deck.read_text(errors="replace"))
        if parsed:
            head = parsed["header"]
            try:
                points = int(head[2])
            except (ValueError, IndexError):
                points = None
            return {"source": deck.name, "title": parsed["title"], "iss": parsed["shrink"] or None, "points": points,
                    "lines": len(parsed["segments"]), "domain": "electronic", "labels_mode": parsed["labels_mode"],
                    "segments": [{"from": s["from"], "to": s["to"], "k_from": s.get("k_from"), "k_to": s.get("k_to")}
                                 for s in parsed["segments"]]}
    if f25.is_f25_name(path.name):
        try:
            blocks = f25.read_blocks(path, ("BAND",))
            domain = f25.domain_of(path)
        except Exception:
            return None
        blocks = [b for b in blocks if len(b.ints) >= 6]
        # an open-shell run writes the path once per spin: the second half repeats the first
        half = len(blocks) // 2
        if half and len(blocks) % 2 == 0 and [b.ints[:6] for b in blocks[:half]] == [b.ints[:6] for b in blocks[half:]]:
            blocks = blocks[:half]
        if not blocks:
            return None
        iss = f25._iss_of(path) if domain == "phonon" else None
        # a phonon path's names: the deck title's, as the band plot labels its ticks (f25._phonon_labels)
        named = f25._title_labels(f25._deck_of(path), [tuple(blocks[0].ints[:3])] + [tuple(b.ints[3:6]) for b in blocks]) \
            if domain == "phonon" else None
        names = [["GAMMA" if n == "Γ" else re.sub(r"[${}]", "", n) for n in lab.split("|")] for lab in named or []]
        segs = []
        for i, b in enumerate(blocks):
            a, e = b.ints[:3], b.ints[3:6]
            segs.append({"from": names[i][-1] if names else "GAMMA" if not any(a) else f"({a[0]},{a[1]},{a[2]})",
                         "to": names[i + 1][0] if names else "GAMMA" if not any(e) else f"({e[0]},{e[1]},{e[2]})",
                         "k_from": [x / iss for x in a] if iss else None, "k_to": [x / iss for x in e] if iss else None,
                         "ints": [list(a), list(e)]})
        return {"source": path.name, "title": None, "iss": iss, "points": int(sum(b.ncol for b in blocks)),
                "lines": len(segs), "domain": domain, "labels_mode": False, "segments": segs}
    if path.name.endswith(BAND_SUFFIXES):
        try:
            head = [ln for ln in path.read_text(errors="replace").splitlines()[:64] if ln.startswith("#")]
        except OSError:
            return None
        m = re.search(r"NKPT\s+(\d+)", head[0]) if head else None
        pts = [_DAT_POINT.match(ln) for ln in head]
        pts = [p for p in pts if p and int(p.group(5)) > 0]          # a malformed "/0" header point says nothing
        iss = int(pts[0].group(5)) if pts else None
        ks = [[int(p.group(k)) / int(p.group(5)) for k in (2, 3, 4)] for p in pts]
        segs = [{"from": "GAMMA" if not any(a) else "", "to": "GAMMA" if not any(b) else "", "k_from": a, "k_to": b}
                for a, b in zip(ks, ks[1:])]
        if not m and not segs:
            return None
        return {"source": path.name, "title": None, "iss": iss, "points": int(m.group(1)) if m else None,
                "lines": len(segs) or None, "domain": _domain(path) if path.name.endswith(".BAND.DAT") else "electronic",
                "labels_mode": False, "segments": segs}
    return None


def partner(path: Path, kind: str) -> Optional[Path]:
    """The DOSS file of the same run as a BAND file (and the reverse)."""
    from mace.plotting import f25
    if f25.is_f25_name(path.name) and f25.plot_kind(path, "band") and f25.plot_kind(path, "dos"):
        return path                    # a phonon run's fort.25 holds the band structure and the DOS
    if kind == "band":
        material = _material(_stem(path.name, BAND_SUFFIXES), "_band")
        names = (f"{material}_doss.DOSS.DAT", f"{material}.DOSS.DAT", f"{material}_doss.dat", f"{material}_doss.f25")
    else:
        material = _material(_stem(path.name, DOS_SUFFIXES), "_doss")
        names = (f"{material}_band.BAND.DAT", f"{material}.BAND.DAT", f"{material}_band.dat",
                 f"{material}.band.band.dat", f"{material}_band.f25")
    for name in names:
        c = path.parent / name
        if c.is_file():
            if name.endswith(".f25") and not _f25_is(c, "dos" if kind == "band" else "band"):
                continue
            return c
    return None


def _f25_is(path: Path, kind: str) -> bool:
    from mace.plotting import f25
    return f25.plot_kind(path, kind) is not None


# ---------------------------------------------------------------- a band + DOS pair from two folders (1.1.6)
PAIR_FERMI_TOL_EV = 0.05        # E_F of the two runs agree within this (a DOSS output prints it to 1 mHa)
PAIR_CELL_TOL = 0.01            # and their periodic lattice lengths within 1 %
from mace.constants import HARTREE_TO_EV as HARTREE_EV  # noqa: E402
_FERMI_RE = (re.compile(r"EFERMI\s+(-?\d*\.\d+(?:E[+-]?\d+)?)"),
             re.compile(r"FERMI ENERGY\s+(-?\d*\.\d+(?:E[+-]?\d+)?)"),
             re.compile(r"TOP OF VALENCE BANDS.*EIG\s+(-?\d*\.\d+(?:E[+-]?\d+)?)"))


def run_output(path: Path, kind: str) -> Optional[Path]:
    """The CRYSTAL output of the run a band or DOS file comes from: its .out beside it, else the SCF run's."""
    path = Path(path)
    if kind == "band":
        _, bz = _d3_tools()
        out = bz.output_for(str(path))
        return Path(out) if out is not None else None
    stem = _stem(path.name, DOS_SUFFIXES)
    material = _material(stem, "_doss")
    for name in (f"{stem}.out", f"{material}_doss.out", f"{material}_sp.out", f"{material}_opt.out",
                 f"{material}.out", f"{material}_SP.out", f"{material}_OPT.out"):
        if (path.parent / name).is_file():
            return path.parent / name
    return None


def _fermi_ev(path: Path, out: Optional[Path]) -> Optional[float]:
    """The run's Fermi energy (eV, absolute): a fort.25's header, else the last one its output prints;
    CRYSTAL's -1.0 Ha placeholder (a spin-locked SCF) is replaced by the run's derived E_F, as everywhere
    (mace.utils.fermi_level.run_fermi), else there is none."""
    from mace.plotting import f25
    from mace.utils import fermi_level
    printed = None
    if f25.is_f25_name(path.name):
        try:
            blocks = f25.read_blocks(path, ("BAND", "DOSS"))
            if blocks:
                printed = float(blocks[-1].ef)
        except Exception:
            pass
    if printed is None and out is not None:
        try:
            text = out.read_text(errors="replace")
        except OSError:
            text = ""
        for rx in _FERMI_RE:
            hits = rx.findall(text)
            if hits:
                printed = float(hits[-1])
                break
    if printed is None:
        return None
    if fermi_level.is_placeholder(printed):
        value = fermi_level.run_fermi(out, printed).value_au if out is not None else None
        return None if value is None else value * HARTREE_EV
    return printed * HARTREE_EV


def _run_facts(path: Path, kind: str, inside=None) -> Dict[str, Any]:
    """What a pair is checked by: the formula and periodic lattice lengths of the run's output, and its E_F."""
    ok = inside or (lambda f: True)
    out = run_output(path, kind)
    if out is not None and not ok(out):
        out = None
    domain = half_domain(path)
    facts: Dict[str, Any] = {"file": path.name, "out": out.name if out else None, "formula": None,
                             "lengths": None, "domain": domain, "supercell": None,
                             "fermi_ev": None if domain == "phonon" else _fermi_ev(path, out)}
    if out is not None:
        facts.update(_out_facts(out))
    if domain == "phonon":
        facts["supercell"] = _supercell(path, out, ok)
    return facts


def half_domain(path: Path) -> str:
    """"phonon" for a phonon run's band or DOS file (a fort.25 of a FREQCALC run, a DAT written from one),
    else "electronic"."""
    from mace.plotting import f25
    path = Path(path)
    return f25.quick_domain(path) if f25.is_f25_name(path.name) else f25.dat_domain(path)


_EXPANSION = re.compile(r"EXPANSION MATRIX OF PRIMITIVE CELL\s*\n\s*E1([^\n]*)\n\s*E2([^\n]*)\n\s*E3([^\n]*)")


def _supercell(path: Path, out: Optional[Path], ok) -> Optional[List[List[int]]]:
    """A phonon run's supercell (its SCELPHONO expansion matrix, the rows of the periodic directions): from the
    output's EXPANSION MATRIX OF PRIMITIVE CELL, else the run's .d12 beside the file; None when neither says."""
    if out is not None:
        try:
            m = _EXPANSION.search(out.read_text(errors="replace"))
        except OSError:
            m = None
        if m:
            rows = [[int(round(float(x))) for x in g.split()[:3]] for g in m.groups()]
            return [r for r in rows if any(r)] or None
    stem = _stem(path.name, BAND_SUFFIXES + DOS_SUFFIXES)
    deck = path.with_name(stem + ".d12")
    if not (deck.is_file() and ok(deck)):
        return None
    try:
        lines = [ln.strip() for ln in deck.read_text(errors="replace").splitlines()]
    except OSError:
        return None
    if "SCELPHONO" not in lines:
        return None
    rows = []
    for ln in lines[lines.index("SCELPHONO") + 1:]:
        try:
            rows.append([int(x) for x in ln.split()])
        except ValueError:
            break
    return rows or None


def supercell_text(rows: List[List[int]]) -> str:
    """2×2×2 for a diagonal expansion (a slab's 2×2×1, a polymer's 2×1×1), else its rows."""
    n = len(rows)
    if all(len(r) >= n and all(r[j] == 0 for j in range(len(r)) if j != i) for i, r in enumerate(rows)):
        return "×".join([str(r[i]) for i, r in enumerate(rows)] + ["1"] * (3 - n))
    return "(" + "; ".join(" ".join(str(x) for x in r) for r in rows) + ")"


_ATOM_ROW = re.compile(r"^\s*\d+\s+\d+\s+([A-Z][A-Z]?)\s+\d+\s+-?\d")


def _out_facts(out: Path) -> Dict[str, Any]:
    """An output's reduced formula and periodic lattice lengths (Å): a properties run prints the direct lattice
    vectors and its ATOM N.AT. SHELL table, an SCF run its geometry. CRYSTAL's formal 500 Å of a slab, polymer
    or molecule is not a period."""
    from collections import Counter
    from functools import reduce
    from math import gcd
    _, bz = _d3_tools()
    facts: Dict[str, Any] = {"formula": None, "lengths": None}
    cell = bz.printed_cell(str(out))
    symbols: List[str] = []
    try:
        lines = out.read_text(errors="replace").splitlines()
    except OSError:
        lines = []
    for n, line in enumerate(lines):
        if "ATOM N.AT.  SHELL" in line:
            symbols = []
            for row in lines[n + 2:]:
                m = _ATOM_ROW.match(row)
                if not m:
                    break
                symbols.append(m.group(1).capitalize())
    if cell is None or not symbols:
        try:
            s = bz.structure_of(str(out))
        except Exception:
            s = None
        if s is not None:
            cell = s["cell"] if cell is None else cell
            st = s.get("structure") or {}
            if not symbols and st.get("numbers"):
                from ase.data import chemical_symbols
                symbols = [chemical_symbols[int(z) % 100] for z in st["numbers"]]
    if cell is not None:
        facts["lengths"] = [float(x) for x in np.linalg.norm(np.asarray(cell, float), axis=1) if x < 499.0]
    if symbols:
        c = Counter(symbols)
        g = reduce(gcd, c.values())
        order = sorted(c, key=lambda el: (0, el) if "C" in c and el in ("C", "H") else (1, el))     # Hill order
        facts["formula"] = "".join(f"{el}{c[el] // g if c[el] // g > 1 else ''}" for el in order)
    return facts


def pair_check(band: Path, dos: Path, inside=None) -> Dict[str, Any]:
    """Whether a band file and a DOS file from another folder are one system: the same reduced formula
    (refused when not), the same periodic lattice lengths within 1 % and E_F within 0.05 eV (warned). With
    E_F further apart the DOS is drawn on the band run's E_F: ``dos_shift`` (eV) is added to its energies.
    A phonon dispersion pairs only with a phonon DOS (an electronic half with a phonon one is refused), of the
    same supercell when both runs say (refused when not); the two share zero frequency, there is no E_F."""
    b, d = _run_facts(Path(band), "band", inside), _run_facts(Path(dos), "dos", inside)
    out: Dict[str, Any] = {"band": b, "dos": d, "error": None, "warnings": [], "notes": [], "dos_shift": 0.0,
                           "domain": b["domain"]}
    if b["domain"] != d["domain"]:
        what = {("band", "phonon"): "a phonon dispersion", ("band", "electronic"): "an electronic band structure",
                ("dos", "phonon"): "a phonon DOS", ("dos", "electronic"): "an electronic DOS"}
        out["error"] = (f"{b['file']} is {what['band', b['domain']]} and {d['file']} {what['dos', d['domain']]}: "
                        f"electronic bands and phonons are never one figure")
        return out
    if b["formula"] and d["formula"] and b["formula"] != d["formula"]:
        out["error"] = (f"{b['file']} is {b['formula']} ({b['out']}) and {d['file']} is {d['formula']} ({d['out']}): "
                        f"not the same system")
        return out
    if b["lengths"] and d["lengths"]:
        if len(b["lengths"]) != len(d["lengths"]):
            out["error"] = f"{b['file']} is {len(b['lengths'])}D and {d['file']} {len(d['lengths'])}D: not the same system"
            return out
        worst = max(abs(x - y) / max(x, 1e-9) for x, y in zip(b["lengths"], d["lengths"]))
        if worst > PAIR_CELL_TOL:
            out["warnings"].append("The cells differ: " + ", ".join(f"{x:.3f}" for x in b["lengths"]) + " Å against " +
                                   ", ".join(f"{y:.3f}" for y in d["lengths"]) + f" Å ({100 * worst:.1f} %)")
    if not (b["formula"] and d["formula"]):
        missing = [f["file"] for f in (b, d) if not f["formula"]]
        out["notes"].append(f"No CRYSTAL output beside {' and '.join(missing)}: the system could not be checked")
    if b["domain"] == "phonon":
        if b["supercell"] and d["supercell"] and b["supercell"] != d["supercell"]:
            out["error"] = (f"{b['file']} was computed in a {supercell_text(b['supercell'])} supercell and {d['file']} in a "
                            f"{supercell_text(d['supercell'])} one: not the same phonon calculation")
            return out
        out["notes"].append("Phonon frequencies: both halves share zero frequency and the plot's unit (no Fermi level)"
                            + (f"; {supercell_text(b['supercell'])} supercell" if b["supercell"] and d["supercell"] else ""))
        return out
    if b["fermi_ev"] is not None and d["fermi_ev"] is not None:
        delta = d["fermi_ev"] - b["fermi_ev"]
        if abs(delta) > PAIR_FERMI_TOL_EV:
            out["dos_shift"] = delta
            out["warnings"].append(f"The Fermi levels differ by {delta:+.3f} eV ({b['fermi_ev']:.3f} eV for the bands, "
                                   f"{d['fermi_ev']:.3f} eV for the DOS): the DOS is drawn on the band run's E_F")
    else:
        out["notes"].append("A Fermi level was not found: each half is drawn on its own run's E_F")
    return out


# ---------------------------------------------------------------- transport (plan E8)
# the raw files' quantities: file token -> (label, unit shown, factor from the file's SI unit); the
# derived ones (PF, κₑ, ZT, L, σ/τ, PF/τ) and the reader are mace.utils.transport_plot, shared with
# mace plotting --transport
from mace.utils import transport_plot as _TP  # noqa: E402

TRANSPORT_QUANTITIES = {k: (_TP.QUANTITIES[k]["label"], _TP.QUANTITIES[k]["unit"], _TP.QUANTITIES[k]["factor"])
                        for k in _TP.FILE_KINDS}
_TRANSPORT_RE = _TP.NAME_RE


def transport_data(path: Path, quantity: Optional[str] = None, kappa_l=None, tau_fs=None, inside=None) -> Dict[str, Any]:
    """A BOLTZTRA run's quantity for the transport plot (default: the opened file's own), per
    temperature against the chemical potential (the TDF against energy): the average of the
    diagonal and its xx, yy, zz, from the database extractor's own rows (an open-shell run's alpha
    and beta blocks summed), and which quantities the run's files beside it allow
    (``available``). See mace.utils.transport_plot.quantity_data."""
    from mace.utils import transport_mesh
    return transport_mesh.add_note(_TP.quantity_data(path, quantity, kappa_l, tau_fs, inside), path)


def _transport_siblings(path: Path, stem: str) -> Dict[str, str]:
    """The run's transport files beside ``path``, by kind (any case: MACE's runs write .DAT)."""
    return {k: p.name for k, p in _TP.run_files(path).items()}


def _transport_out(path: Path, stem: str, inside=None) -> Optional[Path]:
    """The BOLTZTRA .out of a run's .dat files: ``<stem>.out``, else the one transport .out beside them."""
    return _TP.find_out(path, stem, inside)


def transport_zt(path: Path, kappa_l=None, tau_fs=None, inside=None) -> Dict[str, Any]:
    """The figure of merit of a BOLTZTRA run against μ, per temperature: the
    electronic ZTₑ = S²σT/κₑ and, with a lattice κ_L (W/(m·K): a value, or a
    T,kappa_L CSV; see mace.utils.transport_zt), the total S²σT/(κₑ + κ_L) at
    the run's relaxation time or ``tau_fs``. The same computation as
    ``mace analyze --extract-properties --kappa-lattice`` (transport_zt.zt_points
    on the extractor's own rows), so the peak is the CLI's
    ``transport_zt_total_max_avg`` to the last digit. A kappa_L or tau MACE refuses
    raises KappaLatticeError; a total it cannot give (a temperature outside the
    κ_L table, no RELAXTIM in the .out) is None with the reason in ``refused``.
    ``inside(path) -> bool``: the files beside ``path`` it may read (the GUI's folder)."""
    from mace.utils import transport_zt as Z
    from mace.utils.property_extractor import CrystalPropertyExtractor
    m = _TRANSPORT_RE.match(path.name)
    if not m:
        raise ValueError(f"{path.name} is not a BOLTZTRA .dat file")
    stem = m.group("stem")
    sib = {q: n for q, n in _transport_siblings(path, stem).items() if inside is None or inside(path.with_name(n))}
    need = [q for q in ("SEEBECK", "SIGMA", "KAPPA") if q not in sib]
    if need:
        raise ValueError(f"ZT needs the run's SEEBECK, SIGMA and KAPPA .dat files; {', '.join(need)} is missing")
    kl = kappa_l if isinstance(kappa_l, Z.KappaLattice) or kappa_l is None else Z.parse_kappa_lattice(kappa_l)
    tau_fs = Z.check_tau_fs(tau_fs)
    out = _transport_out(path, stem, inside)
    out_text = out.read_text(errors="ignore") if out else None
    # the tensors averaged over the axes the run is periodic along (a slab: in
    # plane), as the database's *_avg values
    axes = CrystalPropertyExtractor.periodic_axes(out_text)
    rows = {q: CrystalPropertyExtractor._parse_transport_dat(path.with_name(sib[q]), axes)[1]
            for q in ("SEEBECK", "SIGMA", "KAPPA")}
    if not rows["SEEBECK"]:
        raise ValueError(f"{sib['SEEBECK']}: no data rows")
    tau_run = Z.read_relaxation_time_fs(out_text) if out else None
    refused = None
    tau = scale = tau_source = None
    if kl is not None:
        try:
            tau, scale, tau_source = Z.resolve_tau(tau_run, tau_fs, out.name if out else f"no {stem}.out beside the data:")
        except Z.KappaLatticeError as e:
            refused = str(e) if out else (f"no {stem}.out beside the data, so no relaxation time: SIGMA.dat and KAPPA.dat "
                                          f"are at the run's RELAXTIM and the total ZT depends on it (κ_L carries no τ)")
    r = Z.zt_points(rows["SEEBECK"], rows["SIGMA"], rows["KAPPA"], kl if refused is None else None, scale or 1.0)
    if kl is not None and refused is None:
        missing = Z.uncovered(kl, r["temperatures"])
        if missing:
            refused = (f"the κ_L table {Path(kl.path).name} covers {kl.range_text()}, and the run has "
                       f"{', '.join(f'{t:g} K' for t in missing)}: no run-wide peak (MACE does not extrapolate)")
    blocks = []
    for T in r["temperatures"]:
        b = r["blocks"][T]
        blocks.append({"T": T, "mu": b["mu"], "zt_e": b["zt_e"], "zt_total": b["zt_total"], "kappa_l": b["kappa_l"],
                       "refused": b["refused"] or (refused if kl is not None and b["zt_total"] is None else None)})
    return {"name": path.name, "stem": stem, "out": out.name if out else None,
            "kappa_l_source": kl.source() if kl is not None else None,
            "kappa_l_label": kl.label() if kl is not None else None,
            "tau_fs": tau, "tau_run_fs": tau_run, "tau_source": tau_source, "refused": refused,
            "blocks": blocks, "zt_e_max": r["zt_e_max"],
            "zt_total_max": r["zt_total_max"] if kl is not None and refused is None else None}
