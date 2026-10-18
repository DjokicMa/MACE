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


def band_data(path: Path, reference: str = "auto", window: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
    """The band plot's data. ``window`` (eV around E_F, Fermi shift included): mace plotting's
    requested energies, every band in it (see _window); None: the GUI's."""
    path = Path(path)
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
    }
    if spin:
        down, trunc_down = _window(Bb, fermi, window)
        _, result["down"] = _with_breaks(Eb[:n], down[:n], breaks)
        result["truncated"] = trunc_up or trunc_down
    return result


def _doss_header(path: Path) -> Tuple[Optional[int], int]:
    _, ipd = _plotters()
    return ipd.read_doss_header(str(path))


def dos_data(path: Path, reference: str = "auto") -> Dict[str, Any]:
    path = Path(path)
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
    return {
        "kind": "dos", "name": path.name, "energy": energy.tolist(),
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
        stem = _stem(path.name, BAND_SUFFIXES)
        material = _material(stem, "_band")
        for cand in (path.parent / f"{stem}.d3", path.parent / f"{material}_band.d3", path.parent / f"{material}.d3"):
            if cand.is_file() and ok(cand):
                parsed = kp.parse_band_deck(cand.read_text(errors="replace"))
                if parsed:
                    segments = parsed["segments"]
                break
    data = bz.bz_view(str(out), segments)
    data["output"] = out.name
    return data


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
    # KAPPA.dat: the manual's κₑ (CRYSTAL23 Eq. 14.3) is the zero-field
    # coefficient κ₀; the electronic κₑ at zero current is κ₀ − T·σ·S²
    "KAPPA": ("Zero-field thermal coefficient κ₀ (the manual's κₑ, Eq. 14.3)", "W/(m·K)", 1.0),
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
    # rows with the tensor diagonal: an open-shell file's alpha and beta blocks
    # summed (CRYSTAL23 manual p. 315), as the extractor and the ZT tab read it
    file_unit, trows, axes = CrystalPropertyExtractor._parse_transport_dat_tensor(path, with_axes=True)
    if not trows:
        raise ValueError(f"{path.name}: no data rows")
    header = None
    for line in path.read_text(errors="ignore").splitlines():
        s = line.strip()
        if s.startswith("#") and "Mu(eV)" in s:
            header = s.lstrip("#").split()
            break
    temps: Dict[float, Dict[str, List[float]]] = {}
    for T, mu, n, diag in trows:
        d = temps.setdefault(T, {"mu": [], "avg": [], "carriers": [], "xx": [], "yy": [], "zz": []})
        d["mu"].append(mu)
        d["avg"].append(sum(diag) / len(diag) * factor)
        d["carriers"].append(n)
        for k, v in zip(axes, diag):  # a slab's file: xx and yy only
            d[k].append(v * factor)
    comps = temps
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


def _transport_out(path: Path, stem: str, inside=None) -> Optional[Path]:
    """The BOLTZTRA .out of a run's .dat files: ``<stem>.out``, else the one transport .out beside them."""
    from mace.utils.calc_detection import is_transport_output
    ok = inside or (lambda f: True)
    exact = path.with_name(f"{stem}.out")
    if exact.is_file():
        return exact if ok(exact) else None
    cands = []
    for c in sorted(path.parent.glob("*.out")):
        if not ok(c):
            continue
        try:
            with c.open(errors="ignore") as fh:
                if is_transport_output(fh.read(2_000_000)):
                    cands.append(c)
        except OSError:
            continue
    return cands[0] if len(cands) == 1 else None


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
