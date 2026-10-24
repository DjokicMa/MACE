"""mace plotting's band structure and DOS figures, drawn by plotly as mace gui draws them.

The GUI builds its band/DOS figure in ``mace/gui/static/js/docs/bandfig.js`` and styles it with
``plotpresets.js styleFigure``; this module is their port, function for function, on the same
data (``mace.gui.results.band_data`` / ``dos_data``: the readers both sides use) and the same
shared presets (``presets.py``, which the GUI reads as generated JSON). The tokens the GUI reads
from tokens.css are ``TOKENS`` here, at the Light plot background (a paper figure stays white);
``tests/test_plotly_bands.py`` runs bandfig.js in node and compares its figure JSON with this
one's, and checks ``TOKENS`` against tokens.css. A change to either side is a change to both.

The files are written by kaleido (``write``); one kaleido process serves every figure of a run.
"""
from __future__ import annotations

import copy
import json
import math
import os
import re
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import presets as P
from mace.utils import plot_labels as L

from mace.constants import CALORIE_J, EV_TO_KJ_MOL, HARTREE_TO_EV as HARTREE_EV  # noqa: E402
ENERGY_UNITS = {"eV": 1, "meV": 1000, "Ha": 1 / HARTREE_EV, "kJ/mol": EV_TO_KJ_MOL, "kcal/mol": EV_TO_KJ_MOL / CALORIE_J}
# the units as the axes draw them (plot_labels.UNITS: cm<sup>−1</sup>)
UNIT_LABEL = {u: L.unit(u) for u in ("eV", "meV", "Ha", "kJ/mol", "kcal/mol", "cm-1", "THz")}
# phonon frequencies (a phonon fort.25's band and DOS): value in the unit = value in eV x factor (plotbase.js PHONON_UNITS)
CM_PER_MEV = 8.065543937
CM_PER_THZ = 33.35640952
PHONON_UNITS = {"cm-1": 1000 * CM_PER_MEV, "THz": 1000 * CM_PER_MEV / CM_PER_THZ, "meV": 1000}

# tokens.css at a light plot (the :root block; --line is --border): what the GUI's band plots read
FONT_SANS = 'system-ui, -apple-system, "Segoe UI", Roboto, "Noto Sans", "Helvetica Neue", Arial, sans-serif'
TOKENS = {
    "--text": "#15181e", "--text-2": "#545b67", "--text-3": "#636a76", "--line": "#e3e6eb",
    "--border-strong": "#848b97", "--accent": "#2563eb", "--plot-bg": "#ffffff", "--surface": "#ffffff",
    "--font-sans": FONT_SANS,
}
# a preset's --plot-ink on a light plot (tokens.css [data-plot-preset] blocks; else --text)
INK = {"publication": "#000000", "grayscale": "#000000"}

# the figure sizes (CSS px, as the GUI pane draws them) and the PNG scale: 300 dpi at 96 px per inch
# the band and band + DOS figures are the GUI plot pane's size at a 1400x900 window (756x421); a DOS alone is square
SIZES = {"band": (760, 420), "dos": (520, 520), "bandos": (760, 420)}
PNG_DPI = 300                  # a PNG's default resolution; the layout is in CSS px, 96 to the inch
PNG_SCALE = PNG_DPI / 96
FORMATS = ("png", "svg", "pdf")

# the GUI's band/DOS options (plot.js DEFAULTS.bands) at their defaults
# (hidden None: the projections drawn are the 2-4 that carry the DOS, presets.pick_projections)
DEFAULT_OPTS = {"emin": -6, "emax": 6, "shift": 0, "spin": "both", "width": 1.5, "hidden": None, "unit": "eV",
                "ylog": False, "xlog": False, "xr": None, "yr": None,
                "fermi_line": True, "k_labels": True, "spin_labels": True}


def phonon_window(band: Optional[Dict[str, Any]], dos: Optional[Dict[str, Any]]) -> List[float]:
    """bandfig.js phononWindow: the frequencies' range (eV) with 5% around it, an unstable mode kept. The DOS
    counts where it has states (its grid runs to the PDOS input's upper limit, often far above the top mode)."""
    vals: List[float] = []
    if band:
        for col in list(band.get("up") or []) + list(band.get("down") or []):
            vals += [v for v in col if v is not None]
    if dos:
        energy = [v for v in dos.get("energy") or []]
        cols = [c for p in dos.get("projections") or [] for c in (p.get("up"), p.get("down")) if c]
        held = [e for k, e in enumerate(energy) if e is not None and any(k < len(c) and c[k] for c in cols)]
        vals += held or [v for v in energy if v is not None]
    if not vals:
        return [0, 0.1]
    lo, hi = min(vals), max(vals)
    pad = max(0.05 * (hi - lo), 0.001)
    return [min(lo, 0) - pad, hi + pad]


def tokens(pid: Optional[str] = "default") -> Dict[str, str]:
    """The tokens a band plot in preset ``pid`` reads on a light plot: the series palette is the
    preset's own (presets.py; tokens.css has the same light palette per preset)."""
    pid = P.resolve_id(pid, "band")
    pal = P.get(pid, "band").get("palette") or P.PRESETS["default"]["palette"]
    out = dict(TOKENS)
    out["--plot-ink"] = INK.get(pid, TOKENS["--text"])
    for k, c in enumerate(pal):
        out[f"--plot-{k + 1}"] = c
    return out


# ---------------------------------------------------------------- JavaScript arithmetic
def _to_fixed(x: float, n: int) -> float:
    """``+(x).toFixed(n)``: the decimal nearest x's exact value, a tie away from zero."""
    q = Decimal(x).quantize(Decimal(1).scaleb(-n), rounding=ROUND_HALF_UP)
    return float(q)


def _js_round(x: float) -> int:
    return math.floor(x + 0.5)


# ---------------------------------------------------------------- the look (bandfig.js lookOf, lookParts)
def look_of(o: Optional[Dict[str, Any]], base: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = {}
    for opt in P.STYLE_OPTIONS:
        name = opt["name"]
        v = (o or {}).get(name)
        if v is not None:
            out[name] = v
        elif base and base.get(name) is not None:
            out[name] = base[name]
        else:
            out[name] = opt["default"]
    return out


def look_parts(env: Dict[str, Any], band: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
    look = env.get("look") or {}
    preset = P.resolve_id(look.get("style") or "default", "band")
    spin = P.spin_colors(preset, "band", env.get("tone") or "light", look.get("spin_colors"),
                         look.get("spin_up_color"), look.get("spin_down_color"))
    spin["dash"] = look.get("spin_down_style") or "dash"
    css = env.get("css") or (lambda n: "")
    # a run that is not spin-polarized: its one set of bands and one-sided DOS total in the band style's
    # color: single the pair's first color (or --band-color), overlay the first color, ink the plot's ink
    spin["style"] = P.band_style(look.get("band_style"), band)
    custom = look.get("band_color")
    if spin["style"] == "ink":
        spin["bands"] = P.band_color(preset, "band", env.get("tone") or "light", None, css("--plot-ink") or css("--text") or "#000000")
    elif spin["style"] == "single" and custom and P._HEXCOLOR.match(custom):
        spin["bands"] = custom.lower()
    else:
        spin["bands"] = spin["up"]
    return preset, spin, look


def pal(css: Callable[[str], str]) -> List[str]:
    return [css(f"--plot-{k + 1}") for k in range(10)]


def k_label(label: Any) -> str:
    return str(label or "").replace("GAMMA", "Γ")


DASH_WORD = {"dash": "dashed", "dot": "dotted", "dashdot": "dash-dot", "solid": "solid"}


# ---------------------------------------------------------------- DOS (bandfig.js dosLook, dosRange)
def _finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def dos_look(dos: Dict[str, Any], env: Dict[str, Any], o: Dict[str, Any]) -> Dict[str, Any]:
    css = env["css"]
    pl = pal(css)
    preset, _spin, look = look_parts(env)
    labels = [p["label"] for p in dos["projections"]]
    bg = css("--plot-bg") or "#ffffff"
    want = look.get("dos_spin") or "auto"
    sp = bool(dos.get("spin_polarized"))
    if want == "auto":
        side = "mirrored" if sp else "one"
    elif want == "overlaid":
        side = "overlaid" if sp else "one"
    else:
        side = "mirrored"
    ef, shift = dos.get("fermi") or 0, o.get("shift") or 0
    emin = o["emin"] if _finite(o.get("emin")) else -6
    emax = o["emax"] if _finite(o.get("emax")) else 6
    weights = P.dos_weights(dos["energy"], dos["projections"], ef + shift + emin, ef + shift + emax)
    out = P.dos_styles(labels, look.get("dos_colors") or "auto", env.get("element_colors") or {}, preset=preset,
                       show_total=look.get("dos_total") is not False, bg=bg, palette=pl[1:] + pl[:1],
                       spin=side != "one", weights=weights)
    out["side"] = side
    out["byLabel"] = {line["label"]: line for line in out["lines"]}
    # the projections drawn by default (o["hidden"] None): the 2-4 carrying the DOS in the window, the
    # rest hidden (presets.pick_projections; bandfig.js dosLook)
    pk = P.pick_projections(out["lines"], weights)
    out["order"] = pk["order"]
    hidden = o.get("hidden")
    out["hidden"] = list(hidden) if isinstance(hidden, list) else [lab for lab in pk["order"] if lab not in pk["shown"]]
    # DOS colors Auto: the preset's projection colors, largest first, clear of the spin pair when both spins are drawn
    if (look.get("dos_colors") or "auto") == "auto":
        ppal = P.projection_palette(preset, [_spin["up"], _spin["down"]] if side != "one" else None, bg)
        if ppal:
            for k, lab in enumerate(pk["order"]):
                out["byLabel"][lab]["color"] = ppal[k % len(ppal)]
                out["byLabel"][lab]["dash"] = "solid"
    return out


def up_of(dos: Dict[str, Any], p: Dict[str, Any], side: Optional[str]) -> List[Any]:
    """The right-hand curve: an alpha = beta run's summed total (results.dos_data spin_summed) is
    split in half when both spins are drawn."""
    if dos.get("spin_summed") and side not in (None, "one"):
        return [v / 2 for v in p["up"]]
    return p["up"]


def down_of(dos: Dict[str, Any], p: Dict[str, Any]) -> List[Any]:
    if dos.get("spin_polarized") and p.get("down"):
        return p["down"]
    if dos.get("spin_summed"):
        return [-v / 2 for v in p["up"]]
    return [-v for v in p["up"]]


def dos_range(dos: Dict[str, Any], lo: float, hi: float, opts: Dict[str, Any], look: Optional[Dict[str, Any]]) -> List[float]:
    m = 0.0
    hidden = (look["hidden"] if look else opts.get("hidden")) or []
    projs = []
    for p in dos["projections"]:
        if p["label"] in hidden or (look and p["label"] in look["byLabel"] and not look["byLabel"][p["label"]]["show"]):
            continue
        projs.append((up_of(dos, p, look["side"] if look else None), down_of(dos, p)))
    for i, e in enumerate(dos["energy"]):
        if e < lo or e > hi:
            continue
        for up, down in projs:
            if opts.get("spin") != "down":
                m = max(m, abs(up[i] or 0))
            if (not look or look["side"] != "one") and opts.get("spin") != "up":
                m = max(m, abs(down[i] or 0))
    m = m * 1.08 if m > 0 else 1
    side = look["side"] if look else ("mirrored" if dos.get("spin_polarized") else "one")
    if side == "mirrored":
        sp = opts.get("spin")
        return [-m, m] if sp == "both" else [-m, 0] if sp == "down" else [0, m]
    return [0, m]


# ---------------------------------------------------------------- bands
# bandfig.js shiftText and energyTitle: labels.js / plot_labels (' + 0.50 eV'; E − E<sub>F</sub>, or with a Fermi
# shift s, which moves the data by -s, E − (E<sub>F</sub> + s), E<sub>vac</sub> on the vacuum scale), as the
# matplotlib figures' energy_axis_label
shift_text = L.shift_text
energy_title = L.energy_title


def join_bands(x: Sequence[Any], bands: Sequence[Sequence[Any]], shift: float):
    xs, ys, idx = [], [], []
    for k, b in enumerate(bands):
        for i in range(len(x)):
            xs.append(x[i])
            ys.append(None if b[i] is None else b[i] - shift)
            idx.append(k)
        xs.append(None)
        ys.append(None)
        idx.append(-1)
    return xs, ys, idx


TICK_FONT = 12


def label_width(html: str) -> float:
    sub = len(re.sub(r"<[^>]+>", "", "".join(re.findall(r"<sub>.*?</sub>", html))))
    main = len(re.sub(r"<[^>]+>", "", re.sub(r"<sub>.*?</sub>", "", html)))
    return (main * 0.62 + sub * 0.45) * TICK_FONT + 6


def k_ticks(ticks: Sequence[float], labels: Sequence[str], rng: Sequence[float], width_px: float):
    vals: List[float] = []
    text: List[str] = []
    for i, t in enumerate(ticks):
        lab = k_label(labels[i])
        k = len(vals) - 1
        if k >= 0 and abs(t - vals[k]) < 1e-5:
            if text[k] != lab:
                text[k] = f"{text[k]}|{lab}"
            continue
        vals.append(t)
        text.append(lab)
    x0, x1 = (rng[0], rng[1]) if rng[0] <= rng[1] else (rng[1], rng[0])
    span = (x1 - x0) or 1
    right = [-math.inf, -math.inf]
    rows = 1
    out = []
    for i, lab in enumerate(text):
        if vals[i] < x0 - 1e-9 or vals[i] > x1 + 1e-9 or not lab:
            out.append(lab)
            continue
        px = (vals[i] - x0) / span * width_px
        half = label_width(lab) / 2
        if px - half >= right[0] + 4:
            row = 0
        elif px - half >= right[1] + 4:
            row = 1
        else:
            row = 0 if right[0] <= right[1] else 1
        right[row] = px + half
        if row:
            rows = 2
        out.append(f" <br>{lab}" if row else lab)
    return vals, out, rows


def band_tick_layout(band, rng, dos, width, r=14):
    w = max(200, ((width or 900) - 58 - r) * (0.7 if dos else 1))
    vals, text, rows = k_ticks(band["ticks"], band["labels"], rng, w)
    return {"xaxis.tickvals": vals, "xaxis.ticktext": text, "margin.b": 42 + (rows - 1) * 15}


# ---------------------------------------------------------------- the k-path as edited (bandfig.js editedBand)
def _kp():
    d3 = Path(__file__).resolve().parents[2] / "Crystal_d3"
    if str(d3) not in sys.path:
        sys.path.insert(0, str(d3))
    import d3_band_path
    return d3_band_path


_GREEK = {"GAMMA": "Γ", "DELTA": "Δ", "SIGMA": "Σ", "LAMBDA": "Λ", "PI": "Π"}


def k_display(label: Any) -> str:
    """kpath.js display: "GAMMA" -> "Γ", "F_0" -> "F<sub>0</sub>" (HTML-escaped)."""
    esc = lambda t: t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")  # noqa: E731
    s = _kp().canonical_label(label)
    m = re.match(r"^([^_]*)(?:_(.*))?$", s, re.S)
    base = m.group(1) if m else s
    sub = m.group(2) if m else None
    base = re.sub(r"^(GAMMA|DELTA|SIGMA|LAMBDA|PI)", lambda w: _GREEK[w.group(1)], base)
    return esc(base) + (f"<sub>{esc(sub)}</sub>" if sub is not None else "")


def path_segments(band: Dict[str, Any]):
    tokens: List[str] = []
    for i, lab in enumerate(band["labels"]):
        lab = lab or f"P{i + 1}"
        parts = str(lab).split("|")
        if len(parts) > 1:
            tokens += [parts[0], "|", parts[1]]
        else:
            tokens.append(parts[0])
    segs = _kp().from_tokens(tokens)
    return segs if len(segs) == len(band["ticks"]) - 1 else None


def slice_of(band: Dict[str, Any], n: int) -> Tuple[int, int]:
    def near(t):
        best, bd = -1, math.inf
        for i, x in enumerate(band["x"]):
            if x is not None and abs(x - t) < bd:
                bd, best = abs(x - t), i
        return best
    a = near(band["ticks"][n])
    b = near(band["ticks"][n + 1])
    if n > 0 and "|" in str(band["labels"][n]):
        a += 1
        while a < b and band["x"][a] is None:
            a += 1
    return a, b


def edited_band(band: Optional[Dict[str, Any]], edit: Optional[List[Dict[str, Any]]]):
    """The band data along an edited k-path (mace plotting --band-path), laid out as the GUI does."""
    if not band or not edit:
        return band
    computed = path_segments(band)
    if not computed:
        return band
    kp = _kp()
    can = kp.canonical_label
    used = set()
    x: List[Any] = []
    up = [[] for _ in band["up"]]
    down = [[] for _ in band["down"]] if band.get("down") else None
    ticks: List[float] = []
    labels: List[str] = []
    x0, prev = 0.0, None
    seg_x = {}
    for ei, s in enumerate(edit):
        if s.get("hidden"):
            continue
        n = next((i for i, c in enumerate(computed) if i not in used and can(c["from"]) == can(s["from"]) and can(c["to"]) == can(s["to"])), -1)
        rev = False
        if n < 0:
            n = next((i for i, c in enumerate(computed) if i not in used and can(c["from"]) == can(s["to"]) and can(c["to"]) == can(s["from"])), -1)
            rev = n >= 0
        if n < 0:
            continue
        used.add(n)
        a, b = slice_of(band, n)
        idx = [i for i in range(a, b + 1) if band["x"][i] is not None]
        if rev:
            idx = idx[::-1]
        xa = band["x"][idx[0]]
        cont = prev is not None and kp.joined(prev, s)
        if prev is not None and not cont:
            x.append(None)
            for u in up:
                u.append(None)
            if down is not None:
                for u in down:
                    u.append(None)
            labels[-1] = f"{labels[-1]}|{k_display(s['from'])}"
        if prev is None:
            ticks.append(0)
            labels.append(k_display(s["from"]))
        for k, i in enumerate(idx):
            if cont and k == 0:
                continue
            x.append(x0 + abs(band["x"][i] - xa))
            for j, col in enumerate(band["up"]):
                up[j].append(col[i])
            if down is not None:
                for j, col in enumerate(band["down"]):
                    down[j].append(col[i])
        xs0 = x0
        x0 += abs(band["x"][idx[-1]] - xa)
        seg_x[ei] = [xs0, x0]
        ticks.append(x0)
        labels.append(k_display(s["to"]))
        prev = s
    if not x:
        return band
    return {**band, "x": x, "up": up, "down": down, "ticks": ticks, "labels": labels, "edited": True, "segX": seg_x}


# ---------------------------------------------------------------- the figure (bandfig.js bandFigure)
def legend_top(**more) -> Dict[str, Any]:
    return {"orientation": "h", "x": 0, "xanchor": "left", "y": 1, "yanchor": "bottom", "font": {"size": 11}, **more}


def legend_side(**more) -> Dict[str, Any]:
    return {"orientation": "v", "x": 1, "xanchor": "left", "y": 1, "yanchor": "top", "font": {"size": 11}, "tracegroupgap": 0, **more}


def group_title(css: Callable[[str], str]) -> Dict[str, Any]:
    """A legend group's title font (bandfig.js GROUP_TITLE): the Spin and Projections groups."""
    return {"size": 11, "color": css("--text-2")}


def spin_keys(spin2: bool, o: Dict[str, Any], spin_dash: str, closed: Optional[str] = None,
              spin: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """The legend's Spin keys (bandfig.js spinKeys): [{"name", "dash", "color"(, "closed")}]: spin up solid
    in the pair's first color and spin down in its second and line style (o["spin"]: one of them), or one
    closed-shell key in the band color; ``spin``: the pair drawn {"up", "down", "bands"} (None: no colors)."""
    sp = spin or {}
    if not spin2:
        return [{"name": closed or "α = β (closed shell)", "dash": "solid", "closed": True, "color": sp.get("bands")}]
    out = []
    if o.get("spin") != "down":
        out.append({"name": "spin up (α)", "dash": "solid", "color": sp.get("up")})
    if o.get("spin") != "up":
        out.append({"name": "spin down (β)", "dash": spin_dash, "color": sp.get("down")})
    return out


def half_keys(band: Optional[Dict[str, Any]], dos: Optional[Dict[str, Any]], side: Optional[str], o: Dict[str, Any],
              spin: Dict[str, Any], closed_words: str) -> Dict[str, Any]:
    """A band + DOS figure's legend keys (bandfig.js halfKeys): {"spin", "bands"}: the DOS's keys as the
    Spin group and, when the bands are drawn in other colors or line styles, the bands' own keys (else
    "bands" None: one shared group)."""
    bk = spin_keys(bool(band.get("spin_polarized")), o, spin["dash"], closed_words, spin) if band else None
    if not dos:
        return {"spin": bk, "bands": None}
    dk = spin_keys(side != "one", o, spin["dash"], closed_words, spin)
    if not bk:
        return {"spin": dk, "bands": None}
    # a closed-shell Overlay draws the pair itself: it matches a DOS drawn in the pair
    same = (not band.get("spin_polarized") and spin["style"] == "overlay" and side != "one") or (
        len(bk) == len(dk) and all(a["color"] == b["color"] and a["dash"] == b["dash"] for a, b in zip(bk, dk)))
    if same:
        return {"spin": bk if (band.get("spin_polarized") or side == "one") else dk, "bands": None}
    return {"spin": dk, "bands": bk}


def sigma_text(sigma_ev: float, f: float, unit: str) -> str:
    """A DOS smoothing width (eV) in the plot's unit (``f``: the unit per eV, ``unit``: its label): as set on
    an eV axis, else to three significant digits: "0.05 eV", "24.2 cm⁻¹" (bandfig.js sigmaText)."""
    v = float(f"{sigma_ev * f:.3g}") if f and f != 1 else sigma_ev
    return f"{P.js_num(v)} {unit or 'eV'}"


def legend_box(names: Sequence[str], groups: int, width: Optional[float], height: Optional[float]) -> Dict[str, float]:
    """A DOS-alone legend's box as a fraction of the plot {"fw", "fh"} (bandfig.js legendBox)."""
    chars = max([0] + [len(re.sub(r"<[^>]+>", "", str(n))) for n in names])
    wpx, hpx = 46 + 6.6 * chars, 10 + 19 * (len(names) + groups)
    pw, ph = max(100, (width or 520) - 72), max(100, (height or 420) - 52)
    return {"fw": min(0.9, wpx / pw), "fh": min(0.9, hpx / ph)}


def _drop_none(d: Dict[str, Any]) -> Dict[str, Any]:
    """JSON.stringify leaves out a key whose value is undefined: the same keys on both sides."""
    return {k: v for k, v in d.items() if v is not None or k in ("x", "y")}


def band_figure(band: Optional[Dict[str, Any]], dos: Optional[Dict[str, Any]], o: Dict[str, Any],
                env: Dict[str, Any]) -> Dict[str, Any]:
    """{"traces", "layout", "legendBase", "absShown", "dosMode"}: bandfig.js bandFigure's figure.
    ``env``: {"css": token function, "look", "tone", "paper", "element_colors", "width", "height",
    "marginR", "fermiKey"} (as bandfig.js; presets are this package's)."""
    css = env["css"]
    preset, spin, look_ = look_parts(env, band)
    abs_shown = False
    dos_mode = None
    ref_d = band or dos
    # a phonon fort.25 (domain "phonon"): see bandfig.js bandFigure
    ph = ref_d.get("domain") == "phonon"
    ref, ef, shift = ref_d.get("reference"), 0 if ph else ref_d.get("fermi") or 0, 0 if ph else o.get("shift") or 0
    # a frequency unit (cm-1, THz) on an electronic plot: eV, the factor and the label alike
    u_name = (o.get("unit") if o.get("unit") in PHONON_UNITS else "cm-1") if ph else \
        (o.get("unit") if o.get("unit") in ENERGY_UNITS else "eV")
    f = (PHONON_UNITS if ph else ENERGY_UNITS).get(u_name) or 1
    unit = UNIT_LABEL.get(u_name) or "eV"
    ds = 1 / f if ph else 1
    tf = lambda y: None if y is None else y * f  # noqa: E731
    # Plot options › Reference/Labels: the dotted E_F line (and its key), the k-point labels, the spin halves' words
    show_ef, show_k, show_spin = (not ph) and o.get("fermi_line") is not False, o.get("k_labels") is not False, o.get("spin_labels") is not False
    # the energy axis is never logarithmic (energies are signed): "ylog" is the DOS axis (GUI parity, bandfig.js)
    log_e = False
    lg = (lambda ys: [None if (y is None or y == 0) else abs(y) for y in ys]) if log_e else (lambda ys: ys)  # noqa: E731
    abs_shown = log_e
    grid = css("--line")
    traces: List[Dict[str, Any]] = []
    shapes: List[Dict[str, Any]] = []
    annotations: List[Dict[str, Any]] = []
    keys: List[Dict[str, Any]] = []
    ink = css("--plot-ink") or css("--text")
    yr = o.get("yr") or ([v * f for v in phonon_window(band, dos)] if ph else [(ef + o["emin"]) * f, (ef + o["emax"]) * f])
    fermi_line = {"color": css("--text-2"), "width": 1.2, "dash": "dot"}     # dotted: dashed reads as spin down
    yaxis = {"title": {"text": f"{'|' if log_e else ''}{'Frequency' if ph else energy_title(ref, shift)}{'|' if log_e else ''} ({unit})"},
             "type": "log" if log_e else "linear"}
    if log_e:
        yaxis["autorange"] = True
    else:
        yaxis["range"] = yr
    yaxis.update({"gridcolor": grid, "zeroline": False, "linecolor": grid, "mirror": True, "showline": True,
                  "showspikes": True, "spikemode": "across", "spikethickness": 1, "spikecolor": css("--text-3"), "spikedash": "dot"})
    lay: Dict[str, Any] = {
        "margin": {"l": 58, "r": 14, "t": 10, "b": 42, "autoexpand": False}, "showlegend": True, "hovermode": "closest",
        "dragmode": "zoom", "legend": legend_top(), **(env.get("paper") or {}),
        "font": {"color": css("--text"), "size": 12, "family": css("--font-sans")},
        "modebar": {"bgcolor": "rgba(0,0,0,0)", "color": css("--text-2"), "activecolor": css("--accent")},
        "yaxis": yaxis,
    }
    look = dos_look(dos, env, o) if dos else None
    side = look["side"] if look else None
    if band:
        lay["xaxis"] = {"domain": [0, 0.7] if dos else [0, 1],
                        "range": [band["ticks"][0], band["ticks"][-1]], "showgrid": True, "gridcolor": css("--border-strong"),
                        "zeroline": False, "showline": True, "mirror": True, "linecolor": grid, "showspikes": True,
                        "spikemode": "across", "spikethickness": 1, "spikecolor": css("--text-3"), "spikedash": "dot", "tickangle": 0}
        if o.get("xr"):
            lay["xaxis"]["range"] = o["xr"]
        tl = band_tick_layout(band, lay["xaxis"]["range"], dos, env.get("width"), env.get("marginR", 14))
        lay["xaxis"]["tickvals"] = tl["xaxis.tickvals"]
        lay["xaxis"]["ticktext"] = tl["xaxis.ticktext"] if show_k else ["" for _ in tl["xaxis.ticktext"]]
        lay["margin"]["b"] = tl["margin.b"] if show_k else 42
        if o.get("spin") != "down" or not band.get("spin_polarized"):
            xs, ys, idx = join_bands(band["x"], band["up"], shift)
            ys = lg([tf(y) for y in ys])
            traces.append({"x": xs, "y": ys, "customdata": idx, "xaxis": "x", "yaxis": "y", "mode": "lines", "type": "scatter",
                           "name": "spin up (α)" if band.get("spin_polarized") else "α = β (non-magnetic)" if band.get("spin_identical") else "bands",
                           "showlegend": False, "meta": "up", "legendrank": 1,
                           "line": {"color": spin["up"] if band.get("spin_polarized") else spin["bands"], "width": o["width"],
                                    "simplify": False}, "hoverinfo": "none"})
            # Band style overlay (a closed-shell run): the same bands again, dashed, in the pair's second color
            if not band.get("spin_polarized") and spin["style"] == "overlay":
                traces.append({"x": xs, "y": ys, "customdata": idx, "xaxis": "x", "yaxis": "y", "mode": "lines", "type": "scatter",
                               "name": "bands (overlay)", "showlegend": False, "meta": "overlay", "legendrank": 2,
                               "line": {"color": spin["down"], "width": o["width"],
                                        "dash": "dash" if spin["dash"] == "solid" else spin["dash"], "simplify": False},
                               "hoverinfo": "none"})
        if band.get("spin_polarized") and band.get("down") and o.get("spin") != "up":
            xs, ys, idx = join_bands(band["x"], band["down"], shift)
            ys = lg([tf(y) for y in ys])
            traces.append({"x": xs, "y": ys, "customdata": idx, "xaxis": "x", "yaxis": "y", "mode": "lines", "type": "scatter",
                           "name": "spin down (β)", "showlegend": False, "meta": "down", "legendrank": 2,
                           "line": {"color": spin["down"], "width": o["width"], "dash": spin["dash"], "simplify": False}, "hoverinfo": "none"})
        if not log_e and show_ef:
            shapes.append({"type": "line", "xref": "x domain", "x0": 0, "x1": 1, "yref": "y", "y0": ef * f, "y1": ef * f,
                           "editable": True, "name": "fermi", "line": dict(fermi_line)})
    smooth = P.smooth_sigma(look_.get("dos_smooth"))
    curves: List[Tuple[List[Any], List[Any]]] = []
    if dos:
        ax, xr = ("xaxis2", "x2") if band else ("xaxis", "x")
        dlog = bool(o.get("ylog"))
        axis = {"domain": [0.73, 1] if band else [0, 1],
                "title": {"text": f"|DOS| (states/{unit}, log)" if dlog else f"DOS (states/{unit if ph else 'eV'})"}, "gridcolor": grid,
                "zeroline": True, "zerolinecolor": ink, "zerolinewidth": 1.2, "showline": True, "mirror": True, "linecolor": grid,
                "type": "log" if dlog else "linear"}
        if dlog:
            axis["autorange"] = True
            abs_shown = True
        else:
            axis["range"] = [v * ds for v in dos_range(dos, shift + yr[0] / f, shift + yr[1] / f, o, look)]
        lay[ax] = axis
        if not band and o.get("xr"):
            lay[ax]["range"] = o["xr"]
        rng = lay[ax].get("range")
        if side == "mirrored" and not dlog and rng[0] < 0 < rng[1]:
            tk = P.mirrored_ticks(rng[0], rng[1])
            lay[ax]["tickvals"], lay[ax]["ticktext"] = tk["vals"], tk["text"]
        dos_mode = look["mode"]
        dl = "legend"
        both = side != "one"
        fills = P.spin_fills(spin, preset)
        nfill = P.neutral_fill(spin["bands"], preset)
        ys = [tf(e - shift) for e in dos["energy"]]
        sm = (lambda xs: P.gauss_smooth(xs, dos["energy"], smooth)) if smooth else (lambda xs: xs)  # noqa: E731
        # Log DOS: a zero (a gap) is no point, and the total is not filled: plotly joins a "tozerox" fill's
        # segments across the gap (a wedge drawn over the gap), and zero is off a log axis anyway.
        dx0 = (lambda xs: [None if (v is None or v == 0) else abs(v) for v in xs]) if o.get("ylog") else (lambda xs: xs)  # noqa: E731
        dx = (lambda xs: [None if v is None else v * ds for v in dx0(xs)]) if ph else dx0  # noqa: E731
        pos = lambda xs: [None if v is None else abs(v) for v in xs]  # noqa: E731
        drawn = [(p, k) for k, p in enumerate(dos["projections"])
                 if look["byLabel"].get(p["label"]) and look["byLabel"][p["label"]]["show"] and p["label"] not in look["hidden"]]
        drawn.sort(key=lambda t: 0 if look["byLabel"][t[0]["label"]]["total"] else 1)
        for p, k in drawn:
            st = look["byLabel"][p["label"]]
            total = st["total"]
            lw = o["width"] + 0.2 if total else o["width"] - 0.2
            col = (spin["up"] if both else spin["bands"]) if total else st["color"]
            dcol = spin["down"] if (total and both) else col
            dash = st.get("dash") or "solid"
            name = st["name"]
            if o.get("spin") != "down":
                x = dx(sm(up_of(dos, p, side)))
                traces.append(_drop_none({"x": x, "y": ys, "xaxis": xr, "yaxis": "y", "mode": "lines", "type": "scatter", "meta": "dos",
                                          "name": f"{name} ↑" if both else name, "legendgroup": p["label"], "legend": dl,
                                          "showlegend": False, "legendrank": 10 + k,
                                          "line": {"color": col, "width": lw, "dash": dash, "simplify": False},
                                          "fill": "tozerox" if total and not dlog else "none",
                                          "fillcolor": (fills["up"] if both else nfill) if total and not dlog else None,
                                          "hoverinfo": "none"}))
                curves.append((x, ys))
            if both and o.get("spin") != "up":
                x = dx(sm(pos(down_of(dos, p)) if side == "overlaid" else down_of(dos, p)))
                mir = total and side == "mirrored" and not dlog
                traces.append(_drop_none({"x": x, "y": ys, "xaxis": xr, "yaxis": "y", "mode": "lines", "type": "scatter",
                                          "meta": "dos", "name": f"{name} ↓", "legendgroup": p["label"], "legend": dl,
                                          "showlegend": False, "legendrank": 10 + k,
                                          "line": {"color": dcol, "width": lw, "dash": spin["dash"] if dash == "solid" else dash, "simplify": False},
                                          "opacity": 0.65 if (side == "overlaid" and dash != "solid") else None,
                                          "fill": "tozerox" if mir else "none", "fillcolor": fills["down"] if mir else None,
                                          "hoverinfo": "none"}))
                curves.append((x, ys))
        rank = {lab: i for i, lab in enumerate(look.get("order") or [])}
        sw = sorted(drawn, key=lambda t: (0 if look["byLabel"][t[0]["label"]]["total"] else 1, rank.get(t[0]["label"], 999), t[1]))
        for i, (p, _k) in enumerate(sw):
            st = look["byLabel"][p["label"]]
            fill = (P.rgba(ink, 0.12) if both else P.rgba(spin["bands"], 0.22)) if st["total"] else st["color"]
            edge = (css("--border-strong") if both else spin["bands"]) if st["total"] else st["color"]
            keys.append(_drop_none({"x": [None], "y": [None], "xaxis": "x", "yaxis": "y", "mode": "markers", "type": "scatter",
                                    "name": st["name"], "meta": "key-proj", "legend": "legend", "legendgroup": "proj",
                                    "legendrank": 100 + i, "customdata": [p["label"]], "hoverinfo": "skip", "showlegend": True,
                                    "marker": {"symbol": "square", "size": 11, "color": fill, "line": {"color": edge, "width": 1}},
                                    "legendgrouptitle": {"text": "Phonon DOS" if ph else "Projections", "font": group_title(css)} if i == 0 else None}))
        if side == "mirrored" and o.get("spin") == "both" and not dlog and show_spin:
            z = (0 - rng[0]) / (rng[1] - rng[0])     # beside the zero line, each on its own side
            annotations.append({"xref": f"{xr} domain", "yref": "paper", "x": z, "y": 1, "xanchor": "left", "xshift": 3, "yanchor": "bottom",
                                "showarrow": False, "text": "<b>↑ up</b>", "font": {"size": 11, "color": spin["up"]}, "name": "spin-up-label"})
            annotations.append({"xref": f"{xr} domain", "yref": "paper", "x": z, "y": 1, "xanchor": "right", "xshift": -3, "yanchor": "bottom",
                                "showarrow": False, "text": "<b>down ↓</b>", "font": {"size": 11, "color": spin["down"]}, "name": "spin-down-label"})
            lay["margin"]["t"] = max(lay["margin"]["t"], 46 if (env.get("width") or 900) < 640 else 24)
        if not log_e and show_ef:
            shapes.append({"type": "line", "xref": f"{xr} domain", "x0": 0, "x1": 1, "yref": "y", "y0": ef * f, "y1": ef * f,
                           "editable": not band, "name": "fermi2", "line": dict(fermi_line)})
        y0, y1 = min(yr[0], yr[1]), max(yr[0], yr[1])
        win = [(w - shift) * f for w in dos["window"]] if dos.get("window") else None
        for g0, g1 in P.dos_window_gaps(win, y0, y1):
            shapes.append({"type": "rect", "xref": f"{xr} domain", "x0": 0, "x1": 1, "yref": "y", "y0": g0, "y1": g1,
                           "line": {"width": 0}, "fillcolor": P.rgba(ink, 0.05), "layer": "below", "name": "no-data"})
            if not band and g1 - g0 >= 0.06 * (y1 - y0):
                low = g0 <= y0
                annotations.append({"xref": f"{xr} domain", "yref": "y", "x": 0.5, "y": g1 if low else g0, "xanchor": "center",
                                    "yanchor": "top" if low else "bottom", "showarrow": False, "text": P.DOS_LAYOUT["no_data"],
                                    "font": {"size": 9, "color": css("--text-3")}, "name": "no-data"})
        if smooth:
            # the width in the axis's unit (set in eV and applied on the readers' eV scale, a phonon DOS's too)
            sig = sigma_text(smooth, f, unit)
            annotations.append({"xref": f"{xr} domain", "yref": "y domain", "x": 0.99, "y": 0.01, "xanchor": "right", "yanchor": "bottom",
                                "showarrow": False,
                                "text": f"smoothed, σ = {sig}" if band else f"Gaussian smoothing σ = {sig} (display only)",
                                "font": {"size": 10, "color": css("--text-2")}, "name": "dos-smooth"})
    # the Spin group (keys in the colors and line styles drawn) and, beside it, the bands' own group when
    # the bands are drawn otherwise than the DOS; phonons have no spin: no Spin group (the DOS's own group only)
    closed_words = "α = β (non-magnetic)" if band and not band.get("spin_polarized") and band.get("spin_identical") else "α = β (closed shell)"
    grouped = bool(dos)
    hk = {"spin": [], "bands": None} if ph else half_keys(band, dos, side, o, spin, closed_words)

    def key_of(k: Dict[str, Any], i: int, group: str, title: str, meta: str) -> Dict[str, Any]:
        return _drop_none({"x": [None], "y": [None], "xaxis": "x", "yaxis": "y", "mode": "lines", "type": "scatter", "name": k["name"],
                           "meta": meta, "legend": "legend", "legendgroup": group if grouped else None,
                           "legendrank": (50 if group == "bands" else 1) + i, "hoverinfo": "skip", "showlegend": True,
                           "line": {"color": k.get("color") or ink, "width": 2, "dash": k["dash"]},
                           "legendgrouptitle": {"text": title, "font": group_title(css)} if grouped and i == 0 else None})
    keys[:0] = ([key_of(k, i, "spin", "Spin", "key-spin") for i, k in enumerate(hk["spin"])]
                + [key_of(k, i, "bands", "Bands", "key-band") for i, k in enumerate(hk["bands"] or [])])
    if env.get("fermiKey") and not log_e and show_ef:
        keys.append({"x": [None], "y": [None], "xaxis": "x", "yaxis": "y", "mode": "lines", "type": "scatter", "meta": "key-fermi",
                     "name": L.fermi_key(shift), "legend": "legend", "line": dict(fermi_line), "hoverinfo": "skip",
                     "legendrank": 5000})
    traces += keys
    if not dos:
        # a phonon dispersion alone: one set of branches, no legend
        lay["legend"] = legend_top() if ph else legend_top(title={"text": "Spin", "side": "left", "font": group_title(css)})
        if not keys:
            lay["showlegend"] = False
    elif band:
        lay["legend"] = legend_side(traceorder="grouped", tracegroupgap=8)
    else:
        groups = len({k["legendgroup"] for k in keys if k.get("legendgroup")})
        box = legend_box([k["name"] for k in keys], groups, env.get("width"), env.get("height"))
        rng = lay["xaxis"].get("range")
        spot = P.legend_spot(curves, rng, yr, box["fw"], box["fh"]) if rng else "outside"
        if spot == "outside":
            lay["legend"] = legend_side(traceorder="grouped", tracegroupgap=8)
        else:
            lay["legend"] = {"orientation": "v", "x": 0.99, "xanchor": "right",
                             "y": 0.99 if spot == "top" else 0.08 if smooth else 0.01, "yanchor": "top" if spot == "top" else "bottom",
                             "font": {"size": 11}, "traceorder": "grouped", "tracegroupgap": 8,
                             "bgcolor": P.rgba(css("--plot-bg") or "#ffffff", 0.88), "bordercolor": grid, "borderwidth": 1}
    lay["shapes"] = shapes
    if annotations:
        lay["annotations"] = annotations
    return {"traces": traces, "layout": lay, "legendBase": {"t": lay["margin"]["t"], "r": lay["margin"]["r"]},
            "absShown": abs_shown, "dosMode": dos_mode}


def _js_num(a: float) -> str:
    """A number as JavaScript prints it in a template string (0.18, 0.09, 1)."""
    if float(a).is_integer():
        return str(int(a))
    return repr(float(a))


# ---------------------------------------------------------------- the preset's styling (plotpresets.js styleFigure)
def style_figure(traces: List[Dict[str, Any]], layout: Dict[str, Any], family: str, pid: Optional[str],
                 tok: Callable[[str], str]):
    pid = P.resolve_id(pid, family)
    if not layout or pid == "default":
        return traces, layout
    pr = P.get(pid, family)
    inky = pid in ("publication", "grayscale")
    ink = (inky and tok("--plot-ink")) or tok("--text")
    base = (layout.get("font") or {}).get("size") or 12
    fd = pr.get("font_delta") or 0
    outside = pr["ticks"] == "outside"
    nogrid = pr["grid"] == "none"
    layout["font"] = {**(layout.get("font") or {}), "color": ink, "size": base + fd}
    for k in list(layout.keys()):
        ax = layout[k]
        if not re.fullmatch(r"[xy]axis\d*", k) or not isinstance(ax, dict):
            continue
        if ax.get("showline") is not False and not ax.get("overlaying"):
            ax.update({"showline": True, "linecolor": ink, "linewidth": _to_fixed(1.0 * pr["frame"], 2)})
        if ax.get("ticks") != "" and outside:
            ax.update({"ticks": "outside", "ticklen": _js_round(4 * pr["frame"]), "tickwidth": _to_fixed(0.9 * pr["frame"], 2), "tickcolor": ink})
        if not ((ax.get("tickfont") or {}).get("size")):
            ax["tickfont"] = {**(ax.get("tickfont") or {}), "color": ink, "size": base + fd + 1}
        if nogrid:
            if ax.get("showgrid") is True:
                ax.update({"gridcolor": ink, "gridwidth": 0.6})
            else:
                ax["showgrid"] = False
        if ax.get("zeroline") and inky:
            ax.update({"zerolinecolor": ink, "zerolinewidth": 0.8})
        if isinstance(ax.get("title"), dict):
            t = ax["title"]
            ax["title"] = {**t, "font": {**(t.get("font") or {}), "color": ink, "size": base + fd + 2}}
    gray = bool(pr.get("dashes")) and family in ("spectra", "scf")
    n = 0
    for t in traces or []:
        if not t:
            continue
        ln = t.get("line")
        if ln and isinstance(ln.get("width"), (int, float)) and ln["width"] > 0 and t.get("mode") != "markers":
            t["line"] = {**ln, "width": _to_fixed(ln["width"] * pr["line"], 2)}
        if gray and t.get("type") == "scatter" and not t.get("meta") and "lines" in (t.get("mode") or "lines") \
                and isinstance(t.get("x"), list) and len(t["x"]) > 1:
            t["line"] = {**(t.get("line") or {}), "color": tok(f"--plot-{(n % 10) + 1}"), "dash": pr["dashes"][n % len(pr["dashes"])]}
            n += 1
        mk = t.get("marker")
        if mk and isinstance(mk.get("size"), (int, float)) and fd > 2:
            t["marker"] = {**mk, "size": _js_round(mk["size"] * 1.4)}
        gt = t.get("legendgrouptitle")
        if gt and gt.get("font"):
            t["legendgrouptitle"] = {**gt, "font": {**gt["font"], "size": base + fd, **({"color": ink} if inky else {})}}
    for k in list(layout.keys()):
        if re.fullmatch(r"legend\d*", k) and layout[k]:
            lg = layout[k]
            layout[k] = lg = {**lg, "font": {**(lg.get("font") or {}), "color": ink, "size": base + fd}}
            if lg.get("title") and lg["title"].get("font"):
                lg["title"] = {**lg["title"], "font": {**lg["title"]["font"], "size": base + fd, **({"color": ink} if inky else {})}}
    if isinstance(layout.get("annotations"), list):
        layout["annotations"] = [{**a, "font": {**a["font"], "size": a["font"]["size"] + fd}}
                                 if a and isinstance(a.get("font"), dict) and isinstance(a["font"].get("size"), (int, float)) else a
                                 for a in layout["annotations"]]
    m = layout.get("margin")
    if m and isinstance(m.get("t"), (int, float)) and m["t"] < 16 + 2 * fd:
        layout["margin"] = m = {**m, "t": 16 + 2 * fd}
    if fd > 2 and m and isinstance(m.get("l"), (int, float)):
        layout["margin"] = {**m, "l": m["l"] + 3 * fd, "b": (m.get("b") or 40) + 2 * fd}
    return traces, layout


# ---------------------------------------------------------------- mace plotting's figures
def opts_for(e_lower: float = -6, e_upper: float = 6, shift: float = 0.0, unit: str = "eV",
             width: float = 1.5) -> Dict[str, Any]:
    o = copy.deepcopy(DEFAULT_OPTS)
    o.update({"emin": float(e_lower), "emax": float(e_upper), "shift": float(shift or 0), "unit": unit or "eV", "width": width})
    return o


def env_for(look: Dict[str, Any], width: float, element_colors: Optional[Dict[str, str]] = None,
            fermi_key: bool = True, height: Optional[float] = None) -> Dict[str, Any]:
    pid = P.resolve_id(look.get("style") or "default", "band")
    tk = tokens(pid)
    return {"css": lambda n: tk.get(n, ""), "look": look, "tone": "light",
            "paper": {"paper_bgcolor": "#ffffff", "plot_bgcolor": "#ffffff"},
            "element_colors": P.element_colors() if element_colors is None else element_colors,
            "width": width, "height": height, "marginR": 14, "fermiKey": fermi_key}


def figure(band: Optional[Dict[str, Any]], dos: Optional[Dict[str, Any]], look: Dict[str, Any],
           o: Dict[str, Any], size: Optional[Tuple[int, int]] = None,
           element_colors: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """The plotly figure (a dict) mace plotting writes: the GUI's band, DOS or band + DOS figure in
    ``look`` (STYLE_OPTIONS names), on a white page, its legends fitted by plotly's margins."""
    kind = "bandos" if band and dos else "band" if band else "dos"
    w, h = size or SIZES[kind]
    env = env_for(look, w, element_colors, height=h)
    fig = band_figure(band, dos, o, env)
    pid = P.resolve_id(look.get("style") or "default", "band")
    tk = tokens(pid)
    traces, layout = style_figure(fig["traces"], fig["layout"], "band", pid, lambda n: tk.get(n, ""))
    # a file: no hover, the legends push the margins (the GUI fits them from the legends as drawn)
    layout["margin"]["autoexpand"] = True
    layout["width"], layout["height"] = w, h
    for k in ("hovermode", "dragmode", "modebar"):
        layout.pop(k, None)
    for s in layout.get("shapes", []):
        s.pop("editable", None)
    # plotly's margin push sets the top legend right on the frame (the GUI's fitLegends leaves the base
    # margin between them): a see-through legend never hides the frame's top line
    for k in [k for k in layout if re.fullmatch(r"legend\d*", k)]:
        if layout[k].get("x", 0) >= 1 or layout[k].get("y", 0) >= 1:     # an inside legend (a DOS alone) keeps its box
            layout[k] = {**layout[k], "bgcolor": "rgba(0,0,0,0)"}
    if (layout.get("legend") or {}).get("orientation") == "h":
        layout["legend"]["y"] = 1 + 6 / max(1, h - 60)
    return {"data": traces, "layout": layout}


_KALEIDO_OK: Optional[bool] = None
_KALEIDO_WHY = ""


# the pair requirements.txt pins: kaleido 0.2.1 needs no Chrome; plotly 7 dropped kaleido 0.x (it
# raises "Image export requires the Kaleido package, v1.0.0 or greater") and 6.2 deprecated it.
# Tested with plotly 5.24.1, 6.0.1, 6.1.0 and 6.1.2.
PLOTLY_PIN = "plotly>=5.0.0,<6.2"
KALEIDO_PIN = "kaleido==0.2.1"
INSTALL_LINE = f'pip install "{PLOTLY_PIN}" "{KALEIDO_PIN}"'


def _why_not(e: BaseException) -> str:
    """What stops the export, and how to fix it: the exception's first non-empty line (plotly 7's
    message starts with a newline, so the note once read "RuntimeError: ") and the install line."""
    text = next((ln.strip() for ln in str(e).splitlines() if ln.strip()), "")
    why = f"{type(e).__name__}: {text}"[:200] if text else type(e).__name__
    try:
        import plotly
        version = plotly.__version__
    except Exception:
        version = None
    if version and int(version.split(".")[0]) >= 7:
        return (f"{why}. plotly {version} needs kaleido 1 (which needs Chrome); MACE uses kaleido 0.2.1. "
                f"Fix: {INSTALL_LINE}")
    return f"{why}. Fix: {INSTALL_LINE}"


def available() -> Tuple[bool, str]:
    """(plotly and kaleido import and a test export works, why not and how to fix it)."""
    global _KALEIDO_OK, _KALEIDO_WHY
    if _KALEIDO_OK is None:
        try:
            import plotly.io as pio  # noqa: F401
            import kaleido  # noqa: F401
            pio.to_image({"data": [{"type": "scatter", "x": [0, 1], "y": [0, 1]}], "layout": {"width": 40, "height": 30}},
                         format="png", width=40, height=30)
            _KALEIDO_OK = True
        except Exception as e:      # missing package, plotly 7 with kaleido 0.2.1, no Chromium, a sandbox
            _KALEIDO_OK, _KALEIDO_WHY = False, _why_not(e)
    return _KALEIDO_OK, _KALEIDO_WHY


def write(fig: Dict[str, Any], path: Path, fmt: str, dpi: Optional[int] = None) -> Path:
    """Write ``fig`` as png (``dpi``, default 300), svg or pdf; one kaleido process serves every call.
    The layout is in CSS px (96 to the inch), so a PNG is drawn at dpi/96 times the layout and says its
    resolution (a pHYs chunk; kaleido writes none, so LaTeX and Word placed a 7.9 in figure 25-33 in
    wide): it opens at the figure's size, layout px / 96 inches."""
    import plotly.io as pio
    lay = fig["layout"]
    dpi = int(dpi or PNG_DPI)
    pio.write_image(fig, str(path), format=fmt, width=lay["width"], height=lay["height"],
                    scale=dpi / 96 if fmt == "png" else 1, validate=False)
    if fmt == "png":
        set_png_dpi(path, dpi)
    return Path(path)


def set_png_dpi(path: Path, dpi: int) -> None:
    """Write ``dpi`` into the PNG's pHYs chunk (dots per meter), replacing any; the pixels are untouched."""
    import struct
    import zlib
    data = Path(path).read_bytes()
    sig, pos, chunks = data[:8], 8, []
    while pos < len(data):
        n = struct.unpack(">I", data[pos:pos + 4])[0]
        kind = data[pos + 4:pos + 8]
        end = pos + 12 + n
        if kind != b"pHYs":
            chunks.append((kind, data[pos:end]))
        pos = end
    ppm = int(round(dpi / 0.0254))
    body = struct.pack(">IIB", ppm, ppm, 1)
    phys = struct.pack(">I", len(body)) + b"pHYs" + body + struct.pack(">I", zlib.crc32(b"pHYs" + body) & 0xffffffff)
    out = [sig]
    for i, (kind, raw) in enumerate(chunks):
        out.append(raw)
        if kind == b"IHDR":
            out.append(phys)
    Path(path).write_bytes(b"".join(out))


def to_json(fig: Dict[str, Any]) -> str:
    return json.dumps(fig, ensure_ascii=False)
