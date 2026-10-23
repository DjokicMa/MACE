"""Plot presets, spin styling and DOS coloring: one definition for ``mace
plotting`` and ``mace gui`` (1.1.6).

A preset is a small data object (``PRESETS``): the series palette on a light
figure, line widths and styles, fonts, grid and ticks, the map colormaps and
isosurface colors, the spin pair and the DOS mode. Each plot family (band &
DOS, spectra, maps & volumes, SCF & run charts) can use its own preset; the
GUI's Settings › Appearance › Plot style sets one for all of them, or one per
family. ``mace plotting --style`` picks one for a CLI figure.

The GUI draws on any theme: its series colors come from tokens.css (the
``[data-plot-preset]`` blocks hold each preset's light and dark palette), the
rest from the copy of ``SHARED`` in ``mace/gui/static/js/plotpresets.js``
(tests/test_plot_presets.py keeps the two equal). More presets, and users' own
saved ones, are entries of the same shape (``register``).

Standard library only.
"""
from __future__ import annotations

import colorsys
import copy
import json
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# SPIN COLORS: named schemes, one pair per plot background (light, dark), the
# same in mace gui and mace plotting (Marcus, 2026-10-16). A preset names its
# scheme ("spin_scheme") or has a pair of its own ("spin": Print grayscale,
# Colorblind safe); the user's choice (Spin colors, --spin-colors) overrides it.
# "custom" is the user's own pair (Spin up/down color, --spin-up-color and
# --spin-down-color). Spin down (beta) is dashed in every scheme by default
# (colorblind safety); --spin-down-solid / "beta dashed" off turns that off.
# ---------------------------------------------------------------------------
SPIN_SCHEMES = {
    "mace": {"label": "MACE pink/cyan",
             "light": {"up": "#e0218a", "down": "#00a0a4"}, "dark": {"up": "#fa26a0", "down": "#2ff3e0"}},
    "blue-red": {"label": "Blue/red",
                 "light": {"up": "#1f4e9c", "down": "#c0392b"}, "dark": {"up": "#7aa7f0", "down": "#f08a7c"}},
    "black-vermilion": {"label": "Black/vermilion",
                        "light": {"up": "#000000", "down": "#d55e00"}, "dark": {"up": "#ffffff", "down": "#e8803a"}},
}
DEFAULT_SPIN_SCHEME = "blue-red"   # Marcus 2026-10-16 (final): blue/red; MACE pink/cyan stays a named scheme
SPIN_COLOR_CHOICES = ("auto", "mace", "blue-red", "black-vermilion", "custom")   # auto: the plot style's
SPIN_COLORS = {tone: dict(SPIN_SCHEMES[DEFAULT_SPIN_SCHEME][tone]) for tone in ("light", "dark")}
SPIN_DOWN_DASH = "dash"           # spin down: dashed, full width (T3)

FAMILIES = ("band", "spectra", "maps", "scf")
FAMILY_LABELS = {"band": "Band & DOS", "spectra": "Spectra", "maps": "Maps & volumes", "scf": "SCF & run charts"}
DOS_MODES = ("auto", "total", "element", "orbital", "element-orbital")
DASHES = ("solid", "dash", "dot", "dashdot", "longdash", "longdashdot")
SPIN_DOS = ("auto", "mirrored", "overlaid")

# s/p/d/f in by-orbital mode: Okabe-Ito hues, the same for every element
ORBITAL_COLORS = {"s": "#0072b2", "p": "#d55e00", "d": "#009e73", "f": "#cc79a7", "all": "#555555"}
# element x orbital: the line style that tells the orbitals apart (with the shade)
ORBITAL_DASH = {"all": "solid", "s": "solid", "p": "dash", "d": "dot", "f": "dashdot"}
# the automatic DOS mode (DOS colors: Auto)
AUTO_DOS = {"orbital_max_elements": 1, "element_orbital_max_elements": 3, "max_lines": 12}

_GRAY = {"palette": ["#000000", "#4d4d4d", "#767676", "#262626", "#626262", "#3a3a3a", "#555555", "#6e6e6e", "#1a1a1a", "#808080"],
         "dashes": ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]}

# the DOS projection colors (the 2-4 projections a DOS draws by default, DOS colors Auto): hues that keep
# clear of every spin scheme's pair, so a projection never reads as a spin (projection_palette drops the
# ones too near the active pair's hues); Print grayscale keeps its grays (None)
_PROJ = ["#2f5fd0", "#e08a00", "#7b4fb5", "#8a6a2a", "#2e8b57", "#5b6b7b"]
_OKABE_PROJ = ["#0072b2", "#d55e00", "#009e73", "#cc79a7", "#c98500", "#882255", "#999933", "#666666"]

# the presets: "palette" is the light-figure series palette (mace plotting's; the GUI's is
# tokens.css, per theme); "line" scales line widths; "font" is mace plotting's label size in pt
# (the GUI adds font_delta px to its own); "spin_scheme" a SPIN_SCHEMES id, or "spin" a pair of its own;
# "dos" the DOS mode, the total's fill opacity and "proj", the projection colors (see _PROJ);
# "bands" the color {light, dark} of a run's one set of bands and its one-sided DOS total when the run
# is not spin-polarized (None: the plot's ink, its text color); "maps"/"iso" are the
# map colormaps and the isosurface (+, -) colors of spin and difference volumes.
PRESETS: Dict[str, Dict[str, Any]] = {
    "default": {
        "label": "Default", "families": list(FAMILIES),
        "desc": "MACE's own look: the theme's colors, a light grid",
        "palette": ["#e5484d", "#30a46c", "#f59e0b", "#8b5cf6", "#0891b2", "#db2777", "#65a30d", "#ea580c", "#64748b", "#2563eb"],
        "line": 1.0, "font": 11, "tick_font": 10, "font_delta": 0, "grid": "theme", "ticks": "theme", "frame": 1.0,
        "dashes": None, "markers": None, "spin": None, "spin_scheme": "blue-red",
        "bands": None,
        "dos": {"mode": "auto", "total_fill": 0.18, "proj": list(_PROJ)},
        "maps": {"sequential": "Viridis", "diverging": "RdBu", "contours": "both"},
        "iso": {"plus": "#2563eb", "minus": "#dc2626"},
    },
    "publication": {
        "label": "Publication", "families": list(FAMILIES),
        "desc": "A figure for a paper: Okabe-Ito after black, black frame, ticks outside, no grid, heavier lines",
        "palette": ["#000000", "#0072b2", "#d55e00", "#009e73", "#cc79a7", "#c98500", "#3d9ad1", "#999933", "#666666", "#882255"],
        "line": 1.25, "font": 12, "tick_font": 11, "font_delta": 1, "grid": "none", "ticks": "outside", "frame": 1.25,
        "dashes": None, "markers": None,
        "spin": None, "spin_scheme": "black-vermilion",
        "bands": {"light": "#000000", "dark": "#ffffff"},
        "dos": {"mode": "auto", "total_fill": 0.18, "proj": list(_OKABE_PROJ)},
        "maps": {"sequential": "Viridis", "diverging": "RdBu", "contours": "both"},
        "iso": {"plus": "#0072b2", "minus": "#d55e00"},
    },
    "vesta": {
        "label": "VESTA-like", "families": ["maps"],
        "desc": "VESTA's look for sections and isosurfaces: a rainbow section map with black contour lines, "
                "yellow (+) and cyan (-) isosurfaces. Maps & volumes only (other plots draw Default)",
        "palette": None, "line": 1.0, "font": 11, "tick_font": 10, "font_delta": 0, "grid": "none", "ticks": "outside", "frame": 1.0,
        "dashes": None, "markers": None, "spin": None, "spin_scheme": "blue-red",
        "bands": None,
        "dos": {"mode": "auto", "total_fill": 0.18, "proj": list(_PROJ)},
        "maps": {"sequential": "vesta", "diverging": "vesta", "contours": "both"},
        "iso": {"plus": "#e6c200", "minus": "#00a0c8"},
    },
    "presentation": {
        "label": "Presentation", "families": list(FAMILIES),
        "desc": "Readable from the back of a room: large text, thick lines, the theme's colors",
        "palette": ["#e5484d", "#30a46c", "#f59e0b", "#8b5cf6", "#0891b2", "#db2777", "#65a30d", "#ea580c", "#64748b", "#2563eb"],
        "line": 2.0, "font": 16, "tick_font": 14, "font_delta": 5, "grid": "theme", "ticks": "outside", "frame": 1.6,
        "dashes": None, "markers": None, "spin": None, "spin_scheme": "blue-red",
        "bands": None,
        "dos": {"mode": "auto", "total_fill": 0.18, "proj": list(_PROJ)},
        "maps": {"sequential": "Viridis", "diverging": "RdBu", "contours": "filled"},
        "iso": {"plus": "#2563eb", "minus": "#dc2626"},
    },
    "grayscale": {
        "label": "Print grayscale", "families": list(FAMILIES),
        "desc": "For black-and-white print: black and grays, told apart by line style and markers",
        "palette": _GRAY["palette"],
        "line": 1.15, "font": 11, "tick_font": 10, "font_delta": 0, "grid": "none", "ticks": "outside", "frame": 1.2,
        "dashes": _GRAY["dashes"], "markers": ["circle", "square", "triangle-up", "diamond", "triangle-down", "cross"],
        "spin": {"light": {"up": "#000000", "down": "#767676"}, "dark": {"up": "#ffffff", "down": "#a6a6a6"}}, "spin_scheme": None,
        "bands": {"light": "#000000", "dark": "#ffffff"},
        "dos": {"mode": "auto", "total_fill": 0.25, "proj": None},
        "maps": {"sequential": "Greys", "diverging": "gray", "contours": "both"},  # british-ok: a colormap name
        "iso": {"plus": "#4d4d4d", "minus": "#bfbfbf"},
    },
    "colorblind": {
        "label": "Colorblind safe", "families": list(FAMILIES),
        "desc": "Okabe-Ito colors that every common color-vision deficiency tells apart; cividis maps",
        "palette": ["#0072b2", "#d55e00", "#009e73", "#cc79a7", "#c98500", "#3d9ad1", "#000000", "#999933", "#666666", "#882255"],
        "line": 1.0, "font": 11, "tick_font": 10, "font_delta": 0, "grid": "theme", "ticks": "theme", "frame": 1.0,
        "dashes": None, "markers": None,
        "spin": {"light": {"up": "#0072b2", "down": "#d55e00"}, "dark": {"up": "#56b4e9", "down": "#e8803a"}}, "spin_scheme": None,
        "bands": {"light": "#000000", "dark": "#ffffff"},
        "dos": {"mode": "auto", "total_fill": 0.18, "proj": list(_OKABE_PROJ)},
        "maps": {"sequential": "Cividis", "diverging": "RdBu", "contours": "both"},
        "iso": {"plus": "#0072b2", "minus": "#d55e00"},
    },
}
ORDER = ["default", "publication", "vesta", "presentation", "grayscale", "colorblind"]

# ---------------------------------------------------------------------------
# The plot style options: one name per option for the GUI's controls (Settings
# and each plot's Plot options › Style), mace plotting's flags and a folder's
# mace-folder.toml [plots] keys (plots.<name>). "gui" is the GUI setting that
# holds the user's default; a plot's own choice is its option of the same name.
# ---------------------------------------------------------------------------
# How a closed-shell run's bands are drawn (1.1.6 design, Marcus 2026-10-18): single (one solid line in the
# color pair's first color), overlay (that line with a dashed line in the second color on top, the look of a
# spin-polarized plot) or ink (the plot's neutral ink); auto: single when the bands
# have a gap at E_F, overlay for a metal (is_metallic). A spin-polarized run always draws up solid, down dashed.
BAND_STYLES = ("auto", "single", "overlay", "ink")
# a band crosses E_F (a metal) when it reaches this far (eV) above and below it
METAL_TOL = 0.01

STYLE_OPTIONS = [
    {"name": "style", "label": "Plot style", "cli": "--style", "gui": "plots.style", "choices": list(ORDER), "default": "default"},
    {"name": "spin_colors", "label": "Color pair", "cli": "--spin-colors", "gui": "plots.spinColors",
     "choices": list(SPIN_COLOR_CHOICES), "default": "auto"},
    {"name": "spin_up_color", "label": "Spin up color", "cli": "--spin-up-color", "gui": "plots.spinUp", "choices": None, "default": ""},
    {"name": "spin_down_color", "label": "Spin down color", "cli": "--spin-down-color", "gui": "plots.spinDown", "choices": None, "default": ""},
    {"name": "band_color", "label": "Band color", "cli": "--band-color", "gui": "plots.bandColor", "choices": None, "default": ""},
    # a closed-shell run's bands (BAND_STYLES): auto is single with a gap, overlay for a metal (is_metallic)
    {"name": "band_style", "label": "Band style", "cli": "--band-style", "gui": "plots.bandStyle", "choices": list(BAND_STYLES), "default": "auto"},
    {"name": "spin_down_style", "label": "β line", "cli": "--spin-down-style", "gui": "plots.spinDownDash",
     "choices": ["dash", "dot", "dashdot", "solid"], "default": "dash"},
    {"name": "dos_colors", "label": "DOS projections", "cli": "--dos-colors", "gui": "plots.dosColors", "choices": list(DOS_MODES), "default": "auto"},
    {"name": "dos_spin", "label": "DOS spin layout", "cli": "--dos-spin", "gui": "plots.spinDos", "choices": list(SPIN_DOS), "default": "auto"},
    {"name": "dos_total", "label": "DOS total", "cli": "--no-dos-total", "gui": "plots.dosTotal", "choices": [True, False], "default": True},
    # Gaussian smoothing of the drawn DOS (display only): sigma in eV, 0 = off; on: DOS_SMOOTH_SIGMA
    {"name": "dos_smooth", "label": "DOS smoothing", "cli": "--dos-smooth", "gui": "plots.dosSmooth", "choices": None, "default": 0},
]
STYLE_NAMES = [o["name"] for o in STYLE_OPTIONS]
STYLE_DEFAULTS = {o["name"]: o["default"] for o in STYLE_OPTIONS}


def style_flags(look: Dict[str, Any], explicit: bool = False) -> List[str]:
    """mace plotting's flags for a style (STYLE_OPTIONS names), only the values that are not the
    defaults: what the GUI's Copy CLI command and Publication Figure pass (bandfig.js styleFlags).
    ``explicit``: the defaults too, so a mace-folder.toml's [plots] values do not replace them."""
    out: List[str] = []
    v = {**STYLE_DEFAULTS, **{k: x for k, x in (look or {}).items() if x is not None}}
    if v["style"] != "default" or explicit:
        out.append(f"--style={v['style']}")
    scheme = v["spin_colors"] or "auto"
    if scheme != "auto" or explicit:
        out.append(f"--spin-colors={scheme}")
    if scheme in ("auto", "custom"):
        for k in ("spin_up_color", "spin_down_color"):
            if v[k] and _HEXCOLOR.match(v[k]):
                out.append(f"--{k.replace('_', '-')}={v[k].lower()}")
    if v.get("band_color") and _HEXCOLOR.match(v["band_color"]):
        out.append(f"--band-color={v['band_color'].lower()}")
    if (v.get("band_style") or "auto") != "auto" or explicit:
        out.append(f"--band-style={v.get('band_style') or 'auto'}")
    if v["spin_down_style"] == "solid":
        out.append("--spin-down-solid")
    elif v["spin_down_style"] != "dash" or explicit:
        out.append(f"--spin-down-style={v['spin_down_style']}")
    if v["dos_colors"] != "auto" or explicit:
        out.append(f"--dos-colors={v['dos_colors']}")
    if v["dos_spin"] != "auto" or explicit:
        out.append(f"--dos-spin={v['dos_spin']}")
    if v["dos_total"] is False:
        out.append("--no-dos-total")
    elif explicit:
        out.append("--dos-total")
    sm = smooth_sigma(v.get("dos_smooth"))
    if sm or explicit:
        out.append(f"--dos-smooth={js_num(sm)}")
    return out


def js_num(x: float) -> str:
    """A number as JavaScript writes it in a string (0.05, 1, 0.126): the same text on both sides."""
    x = float(x)
    if x.is_integer():
        return str(int(x))
    return repr(x)


def smooth_sigma(v: Any) -> float:
    """The DOS smoothing width (eV) of a style value: 0 (off) for anything that is not a positive number."""
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
        return 0
    return float(v)


# The DOS layout (1.1.6 design): the projections drawn by default (those carrying at least "frac" of the
# DOS in the window, at least "min", at most "max"), the smoothing width when it is turned on, the spin
# fills' and the closed-shell total fill's opacity (times the preset's total_fill), the hue distance a
# projection color keeps from a spin color, and the words over the energies the DOS run did not compute.
DOS_SMOOTH_SIGMA = 0.05
DOS_LAYOUT = {"pick_min": 2, "pick_max": 4, "pick_frac": 0.04, "smooth_sigma": DOS_SMOOTH_SIGMA,
              "spin_fill": 0.7, "neutral_fill": 0.5, "hue_gap": 30, "chroma": 0.25,
              "no_data": "outside the DOS window (no data)"}

# what both sides share (plotpresets.js holds the same JSON between its markers)
SHARED = {"spin": SPIN_COLORS, "spin_schemes": SPIN_SCHEMES, "default_spin_scheme": DEFAULT_SPIN_SCHEME,
          "spin_color_choices": list(SPIN_COLOR_CHOICES), "spin_down_dash": SPIN_DOWN_DASH, "style_options": STYLE_OPTIONS, "families": list(FAMILIES), "order": ORDER,
          "presets": PRESETS, "orbital_colors": ORBITAL_COLORS, "orbital_dash": ORBITAL_DASH,
          "shade_span": 0.36, "auto_dos": AUTO_DOS, "dos_modes": list(DOS_MODES), "dos_layout": DOS_LAYOUT,
          "band_styles": list(BAND_STYLES), "metal_tol": METAL_TOL}


def shared_json() -> str:
    return json.dumps(SHARED, sort_keys=True)


def register(pid: str, preset: Dict[str, Any], base: str = "default") -> Dict[str, Any]:
    """Add a preset (a user's saved one): ``preset`` over a copy of ``base``."""
    if not re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", pid or ""):
        raise ValueError("a preset id is lowercase letters, digits, - and _")
    out = copy.deepcopy(PRESETS[base])
    out.update(copy.deepcopy(preset))
    PRESETS[pid] = out
    if pid not in ORDER:
        ORDER.append(pid)
    return out


def get(pid: Optional[str], family: str = "band") -> Dict[str, Any]:
    """The preset ``pid`` as ``family`` draws it: Default for an unknown id, or for a family
    the preset does not cover (VESTA-like is for maps and volumes)."""
    p = PRESETS.get(pid or "default") or PRESETS["default"]
    if family not in p["families"]:
        p = PRESETS["default"]
    return p


def resolve_id(pid: Optional[str], family: str = "band") -> str:
    p = PRESETS.get(pid or "default")
    return pid if p and family in p["families"] else "default"


_HEXCOLOR = re.compile(r"^#[0-9a-fA-F]{6}$")


def spin_colors(pid: Optional[str], family: str = "band", tone: str = "light", scheme: Optional[str] = None,
                up: Optional[str] = None, down: Optional[str] = None) -> Dict[str, str]:
    """The spin pair {"up", "down"} a plot draws: ``scheme`` (a SPIN_SCHEMES id) when given, else the
    preset's own (its spin_scheme, or its own pair). ``up``/``down`` (#rrggbb) are the custom colors:
    they apply with scheme "custom", or "auto"/None (a color given alone means custom)."""
    p = get(pid, family)
    tone = tone if tone in ("light", "dark") else "light"
    if scheme in SPIN_SCHEMES:
        pair = SPIN_SCHEMES[scheme]
    elif p.get("spin_scheme") in SPIN_SCHEMES:
        pair = SPIN_SCHEMES[p["spin_scheme"]]
    else:
        pair = p.get("spin") or SPIN_SCHEMES[DEFAULT_SPIN_SCHEME]
    out = dict(pair.get(tone) or pair["light"])
    if scheme in (None, "", "auto", "custom"):
        if up and _HEXCOLOR.match(up):
            out["up"] = up.lower()
        if down and _HEXCOLOR.match(down):
            out["down"] = down.lower()
    return out


def band_color(pid: Optional[str], family: str = "band", tone: str = "light", custom: Optional[str] = None,
               ink: str = "#000000") -> str:
    """The color of a run's one set of bands (closed shell, or alpha = beta) and of its one-sided DOS
    total: ``custom`` (#rrggbb, --band-color), else the preset's "bands", else ``ink`` (the plot's text
    color). A spin-polarized run keeps the spin pair."""
    if custom and _HEXCOLOR.match(custom):
        return custom.lower()
    own = get(pid, family).get("bands")
    tone = tone if tone in ("light", "dark") else "light"
    if own:
        return own.get(tone) or own["light"]
    return ink


def is_metallic(band: Optional[Dict[str, Any]]) -> bool:
    """A band structure without a gap at E_F (plotpresets.js isMetallic): one of its bands (either spin)
    reaches METAL_TOL eV above and below the Fermi energy (band["fermi"], the energies' own scale)."""
    if not band:
        return False
    ef = band.get("fermi") or 0
    for block in (band.get("up") or [], band.get("down") or []):
        for b in block:
            v = [e for e in b if e is not None]
            if v and min(v) < ef - METAL_TOL and max(v) > ef + METAL_TOL:
                return True
    return False


def band_style(style: Optional[str], band: Optional[Dict[str, Any]] = None) -> str:
    """The band style drawn (single, overlay or ink): ``style`` (BAND_STYLES), auto: overlay for a metal, else single."""
    if style in ("single", "overlay", "ink"):
        return style
    return "overlay" if is_metallic(band) else "single"


# ---------------------------------------------------------------------------
# colors
# ---------------------------------------------------------------------------
def _rgb(h: str):
    h = h.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def _hex(rgb) -> str:
    return "#" + "".join(f"{max(0, min(255, math.floor(c * 255 + 0.5))):02x}" for c in rgb)


def luminance(h: str) -> float:
    def ch(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (ch(c) for c in _rgb(h))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: str, b: str) -> float:
    la, lb = luminance(a), luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def readable(color: str, bg: str = "#ffffff", need: float = 2.0) -> str:
    """``color`` darkened (on a light background) or lightened (on a dark one) in HLS lightness
    until it has ``need``:1 contrast on ``bg``: a light element color (S yellow, H white) reads
    as a line."""
    if contrast(color, bg) >= need:
        return color.lower()
    h, l, s = colorsys.rgb_to_hls(*_rgb(color))
    dark_bg = luminance(bg) < 0.18
    for _ in range(100):
        l = min(1.0, l + 0.01) if dark_bg else max(0.0, l - 0.01)
        c = _hex(colorsys.hls_to_rgb(h, l, s))
        if contrast(c, bg) >= need:
            return c
    return "#ffffff" if dark_bg else "#000000"


ORBITAL_RANK = {"s": 0.0, "p": 1 / 3, "d": 2 / 3, "f": 1.0}
SHADE_SPAN = 0.36


def shade(color: str, orbital: str, bg: str = "#ffffff", need: float = 2.0, rank: Optional[float] = None) -> str:
    """A shade of ``color``'s hue for one orbital (element x orbital): s strongest against the
    background, then p, d, f, spread over SHADE_SPAN of HLS lightness from the faintest shade
    still readable (``need``:1 on ``bg``); "all" is the color itself, made readable. ``rank``
    (0 strongest .. 1 faintest) overrides the orbital's own (dos_styles: by weight)."""
    if rank is None and orbital not in ORBITAL_RANK:
        return readable(color, bg, need)
    h, _l, s = colorsys.rgb_to_hls(*_rgb(color))
    dark_bg = luminance(bg) < 0.18
    # the faintest readable lightness of this hue: the highest one (light bg) or the lowest (dark)
    edge = 0.0 if not dark_bg else 1.0
    for i in range(101):
        l = (100 - i) / 100 if not dark_bg else i / 100
        if contrast(_hex(colorsys.hls_to_rgb(h, l, s)), bg) >= need:
            edge = l
            break
    t = ORBITAL_RANK[orbital] if rank is None else rank
    # spread down (up) from the edge, never to black or white: the hue stays visible
    if not dark_bg:
        lo = max(0.15, edge - SHADE_SPAN)
        l = edge - (edge - lo) * (1 - t)
    else:
        hi = min(0.9, edge + SHADE_SPAN)
        l = edge + (hi - edge) * (1 - t)
    return _hex(colorsys.hls_to_rgb(h, l, s))


# ---------------------------------------------------------------------------
# DOS projections: names, the automatic mode and each line's color and style
# ---------------------------------------------------------------------------
_ORB_RE = re.compile(r"^\s*([A-Z][a-z]?)\s*(?:\((all|s|p|d|f)\))?\s*$")


def parse_projection(label: str) -> Dict[str, Optional[str]]:
    """A projection label ("C (p)", "Mg (all)", "Mg", "Total", "projection 3") as
    {"element", "orbital"}: element None for the total or an unnamed projection."""
    if label.strip() == "Total":
        return {"element": None, "orbital": "total"}
    m = _ORB_RE.match(label)
    if not m:
        return {"element": None, "orbital": None}
    return {"element": m.group(1), "orbital": m.group(2) or "all"}


def legend_name(label: str) -> str:
    """The legend's name of a projection: "Mg s", "O p", "Mg" (all its orbitals), "Total"."""
    p = parse_projection(label)
    if p["element"] is None:
        return label.strip()
    return p["element"] if p["orbital"] == "all" else f"{p['element']} {p['orbital']}"


def auto_dos_mode(projections: Iterable[Dict[str, Any]]) -> str:
    """DOS colors Auto: one element -> by orbital; two or three -> element x orbital (unless that
    is more than AUTO_DOS max_lines lines); more -> by element. No named projection -> total."""
    projections = list(projections)
    els = [p["element"] for p in projections if p.get("element")]
    elements = list(dict.fromkeys(els))
    if not elements:
        return "total"
    if len(elements) <= AUTO_DOS["orbital_max_elements"]:
        return "orbital"
    orb_lines = sum(1 for p in projections if p.get("element") and p.get("orbital") not in ("all", None))
    if len(elements) <= AUTO_DOS["element_orbital_max_elements"] and orb_lines <= AUTO_DOS["max_lines"]:
        return "element-orbital"
    return "element"


DARK_GRAYS = ("#ffffff", "#bdbdbd", "#8c8c8c", "#e0e0e0", "#a6a6a6", "#d0d0d0", "#b0b0b0", "#9a9a9a", "#eeeeee", "#949494")
RANK_STEPS = (0.0, 1 / 3, 2 / 3, 1.0)       # a shade per place: strongest first
DASH_STEPS = ("solid", "dash", "dot", "dashdot")


def dos_weights(energy: Sequence[float], projections: Sequence[Dict[str, Any]], lo: float, hi: float) -> Dict[str, float]:
    """Each projection's weight between ``lo`` and ``hi`` (eV, as ``energy``): the sum of |up| and
    |down| over the points in that window (dos_styles ranks an element's orbitals by it)."""
    out = {}
    for p in projections:
        w = 0.0
        down = p.get("down")
        for i, e in enumerate(energy):
            if e is None or e < lo or e > hi:
                continue
            w += abs(p["up"][i] or 0.0)
            if down is not None:
                w += abs(down[i] or 0.0)
        out[p["label"]] = w
    return out


def _places(parsed: List[Dict[str, Any]], weights: Optional[Dict[str, float]]) -> Dict[str, int]:
    """Each orbital projection's place among its element's orbitals: by weight, largest first
    (the dominant line gets the strongest shade and a solid line); without weights s, p, d, f."""
    order = {"s": 0, "p": 1, "d": 2, "f": 3}
    out: Dict[str, int] = {}
    by_el: Dict[str, List[Dict[str, Any]]] = {}
    for p in parsed:
        if p["element"] and p["orbital"] in order:
            by_el.setdefault(p["element"], []).append(p)
    for items in by_el.values():
        if weights:
            items = sorted(items, key=lambda p: (-(weights.get(p["label"]) or 0.0), order[p["orbital"]]))
        else:
            items = sorted(items, key=lambda p: order[p["orbital"]])
        for k, p in enumerate(items):
            out[p["label"]] = k
    return out


def dos_styles(labels: Sequence[str], mode: str = "auto", element_colors: Optional[Dict[str, str]] = None,
               preset: Optional[str] = None, show_total: bool = True, bg: str = "#ffffff",
               palette: Optional[Sequence[str]] = None, spin: bool = False,
               weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Which projections a DOS draws and how: ``{"mode": the mode used, "lines": [{"label",
    "name", "color", "dash", "show", "total"}]}`` in the order of ``labels``.

    * total: the total only;
    * element: each element's all-orbital projection, in its element color (made readable);
    * orbital: each orbital projection in its orbital's hue (``ORBITAL_COLORS``), elements told
      apart by line style;
    * element-orbital: each orbital projection a shade of its element's color, also by line style;
    * Grayscale (its palette has no hue): every line black or gray, told apart by line style.
    An element's orbitals take their shades and line styles by ``weights`` (dos_weights), the
    largest strongest and solid, so the dominant line (Cr d of CrO2) is never the faintest or
    dotted; without weights in the order s, p, d, f. ``spin``: both spins are drawn, and spin
    down is the dashed line: every projection is solid, told apart by color (the elements of
    orbital mode by a shade of the orbital's hue). An unnamed projection ("projection 3") takes
    the series palette. ``show_total``: the total is drawn too (in every mode but total, where it
    always is)."""
    P = get(preset, "band")
    pal = list(palette or P.get("palette") or PRESETS["default"]["palette"])
    gray = bool(P.get("dashes"))
    parsed = [dict(parse_projection(lab), label=lab) for lab in labels]
    used = auto_dos_mode(parsed) if mode in (None, "", "auto") else mode
    if used not in DOS_MODES:
        used = "element"
    has_all = {p["element"] for p in parsed if p["element"] and p["orbital"] == "all"}
    elements = list(dict.fromkeys(p["element"] for p in parsed if p["element"]))
    place = _places(parsed, weights)
    ecol = {}
    for k, el in enumerate(elements):
        base = (element_colors or {}).get(el) or pal[k % len(pal)]
        ecol[el] = readable(base, bg)
    edash = {el: DASHES[k % len(DASHES)] for k, el in enumerate(elements)}
    eplace = {el: k for k, el in enumerate(elements)}
    lines, unnamed = [], 0
    for k, p in enumerate(parsed):
        line = {"label": p["label"], "name": legend_name(p["label"]), "total": p["orbital"] == "total",
                "color": None, "dash": "solid", "show": False}
        el, orb = p["element"], p["orbital"]
        n = place.get(p["label"], 0)
        if line["total"]:
            line["show"] = used == "total" or show_total
            line["color"] = "#000000" if luminance(bg) > 0.18 else "#ffffff"
        elif el is None:                                   # an unnamed projection: shown unless total only
            line["show"] = used != "total"
            line["color"] = pal[unnamed % len(pal)]
            unnamed += 1
        elif used == "element":
            # the element's all-orbital projection; without one, its orbitals (in its color, by style)
            line["show"] = orb == "all" or el not in has_all
            line["color"] = ecol[el] if (orb == "all" or not spin) else shade(ecol[el], orb, bg, rank=RANK_STEPS[min(n, 3)])
            line["dash"] = "solid" if orb == "all" or spin else (DASH_STEPS[min(n, 3)] if weights else ORBITAL_DASH.get(orb, "solid"))
        elif used == "orbital":
            line["show"] = orb != "all"
            base = ORBITAL_COLORS.get(orb, "#555555")
            if spin and len(elements) > 1:
                line["color"] = shade(base, orb, bg, rank=RANK_STEPS[min(eplace[el], 3)])
            else:
                line["color"] = readable(base, bg)
                line["dash"] = edash[el] if len(elements) > 1 else "solid"
        elif used == "element-orbital":
            line["show"] = orb != "all"
            if weights:
                line["color"] = shade(ecol[el], orb, bg, rank=RANK_STEPS[min(n, 3)])
                line["dash"] = "solid" if spin else DASH_STEPS[min(n, 3)]
            else:
                line["color"] = shade(ecol[el], orb, bg)
                line["dash"] = "solid" if spin else ORBITAL_DASH.get(orb, "solid")
        if gray and not line["total"] and line["show"]:
            m = sum(1 for x in lines if x["show"] and not x["total"])
            if luminance(bg) > 0.18:
                line["color"] = P["palette"][(m + 1) % len(P["palette"])]
            else:       # on a dark plot: light grays (one gray when the dashes tell the lines apart)
                line["color"] = DARK_GRAYS[(m + 1) % len(DARK_GRAYS)] if spin else "#d9d9d9"
            line["dash"] = "solid" if spin else P["dashes"][m % len(P["dashes"])]
        lines.append(line)
    return {"mode": used, "lines": lines}


# ---------------------------------------------------------------------------
# the DOS layout (1.1.6 design): shared by mace gui (plotpresets.js, the same functions by the same
# names in camelCase), mace plotting's plotly figure (plotly_bands.py), the gallery and the matplotlib
# figures. Plain data in, plain data out.
# ---------------------------------------------------------------------------
def rgba(color: str, a: float) -> str:
    """``#rrggbb`` at opacity ``a`` (rounded to 3 places) as plotly's rgba(); another color as it is."""
    if not isinstance(color, str) or not _HEXCOLOR.match(color):
        return color
    r, g, b = (int(color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{js_num(_round3(a))})"


def _round3(x: float) -> float:
    return math.floor(x * 1000 + 0.5) / 1000


def spin_fills(pair: Dict[str, str], pid: Optional[str] = "default", family: str = "band") -> Dict[str, str]:
    """The light fills under the spin up and spin down totals: each spin's own color (the active pair) at
    the preset's total_fill times DOS_LAYOUT["spin_fill"]."""
    a = _round3((get(pid, family)["dos"].get("total_fill") or 0.18) * DOS_LAYOUT["spin_fill"])
    return {"up": rgba(pair["up"], a), "down": rgba(pair["down"], a)}


def neutral_fill(color: str, pid: Optional[str] = "default", family: str = "band") -> str:
    """The light fill under a closed-shell run's total: its neutral band color at the preset's total_fill
    times DOS_LAYOUT["neutral_fill"]."""
    return rgba(color, _round3((get(pid, family)["dos"].get("total_fill") or 0.18) * DOS_LAYOUT["neutral_fill"]))


def _hue_chroma(color: str):
    r, g, b = _rgb(color)
    mx, mn = max(r, g, b), min(r, g, b)
    c = mx - mn
    if c == 0:
        return 0.0, 0.0
    if mx == r:
        h = ((g - b) / c) % 6
    elif mx == g:
        h = (b - r) / c + 2
    else:
        h = (r - g) / c + 4
    return h * 60, c


def projection_palette(pid: Optional[str] = "default", avoid: Optional[Sequence[str]] = None, bg: str = "#ffffff",
                       family: str = "band") -> Optional[List[str]]:
    """The colors of a DOS's projections, largest first: the preset's "proj" colors without those whose
    hue is within DOS_LAYOUT["hue_gap"] degrees of a (chromatic) color in ``avoid`` (the active spin pair),
    each made readable on ``bg``. None: the preset keeps dos_styles' colors (Print grayscale)."""
    pool = get(pid, family)["dos"].get("proj")
    if not pool:
        return None
    gap, chroma = DOS_LAYOUT["hue_gap"], DOS_LAYOUT["chroma"]
    spins = [_hue_chroma(c) for c in (avoid or []) if isinstance(c, str) and _HEXCOLOR.match(c)]
    spins = [h for h, c in spins if c >= chroma]

    def clear(c):
        h, ch = _hue_chroma(c)
        if ch < chroma:
            return True
        return all(min(abs(h - s), 360 - abs(h - s)) >= gap for s in spins)
    out = [c for c in pool if clear(c)] or list(pool)
    return [readable(c, bg) for c in out]


def pick_projections(lines: Sequence[Dict[str, Any]], weights: Dict[str, float], nmin: Optional[int] = None,
                     nmax: Optional[int] = None, frac: Optional[float] = None) -> Dict[str, List[str]]:
    """Which projections a DOS draws by default: of ``lines`` (dos_styles' lines; the shown, non-total
    ones count), those carrying at least ``frac`` of the DOS in the window (``weights``: dos_weights; the
    Total's weight, else the sum), at least ``nmin``, at most ``nmax``, largest first. {"order": every
    candidate label by weight, "shown": the ones drawn}; the rest are the default hidden projections."""
    nmin = DOS_LAYOUT["pick_min"] if nmin is None else nmin
    nmax = DOS_LAYOUT["pick_max"] if nmax is None else nmax
    frac = DOS_LAYOUT["pick_frac"] if frac is None else frac
    cand = [ln["label"] for ln in lines if ln.get("show") and not ln.get("total")]
    w = {lab: float((weights or {}).get(lab) or 0.0) for lab in cand}
    tot = next((float((weights or {}).get(ln["label"]) or 0.0) for ln in lines if ln.get("total")), 0.0) or sum(w.values())
    order = [lab for _k, lab in sorted(enumerate(cand), key=lambda t: (-w[t[1]], t[0]))]
    shown = [lab for k, lab in enumerate(order) if k < nmin or w[lab] >= frac * tot][:nmax]
    return {"order": order, "shown": shown}


def gauss_smooth(y: Sequence[Any], energy: Sequence[float], sigma: float) -> List[float]:
    """``y`` convolved with a normalized Gaussian of width ``sigma`` (eV) on ``energy``'s median step,
    out to 4 sigma; points beyond the data count as zero (display only: the DOS as drawn)."""
    vals = [0.0 if v is None else float(v) for v in y]
    if not sigma or sigma <= 0 or len(vals) < 3:
        return vals
    steps = sorted(energy[i + 1] - energy[i] for i in range(len(energy) - 1))
    m = len(steps) // 2
    de = steps[m] if len(steps) % 2 else (steps[m - 1] + steps[m]) / 2
    if not de or de <= 0:
        return vals
    n = int(math.ceil(4 * sigma / de))
    k = [math.exp(-0.5 * ((j - n) * de / sigma) ** 2) for j in range(2 * n + 1)]
    ks = 0.0
    for v in k:
        ks += v
    k = [v / ks for v in k]
    out = []
    size = len(vals)
    for i in range(size):
        acc = 0.0
        for j in range(max(0, n - i), min(2 * n + 1, size - i + n)):
            acc += vals[i + j - n] * k[j]
        out.append(acc)
    return out


def dos_window_gaps(window: Optional[Sequence[float]], lo: float, hi: float) -> List[List[float]]:
    """The energy ranges between ``lo`` and ``hi`` that the DOS run did not compute (outside its
    ``window``, dos_data's [first, last] energy): [[y0, y1], ...], each shaded and labeled
    DOS_LAYOUT["no_data"] (no data, not zero states)."""
    if not window or len(window) != 2:
        return []
    out = []
    if window[0] > lo:
        out.append([lo, min(window[0], hi)])
    if window[1] < hi:
        out.append([max(window[1], lo), hi])
    return [g for g in out if g[1] > g[0]]


def legend_spot(curves: Sequence[Tuple[Sequence[Any], Sequence[Any]]], xr: Sequence[float], yr: Sequence[float],
                fw: float, fh: float) -> str:
    """Where a DOS-alone figure's legend goes: "top" (inside, top right) when no curve enters the box
    ``fw`` x ``fh`` (fractions of the plot) there, else "bottom" (inside, bottom right), else "outside"
    (right of the plot). ``curves``: (x, y) pairs in data coordinates; ``xr``, ``yr`` the axis ranges."""
    x0, x1 = min(xr), max(xr)
    y0, y1 = min(yr), max(yr)
    xcut = x1 - fw * (x1 - x0)
    for spot, lo, hi in (("top", y1 - fh * (y1 - y0), y1), ("bottom", y0, y0 + fh * (y1 - y0))):
        hit = False
        for xs, ys in curves:
            for x, y in zip(xs, ys):
                if x is not None and y is not None and lo <= y <= hi and x > xcut:
                    hit = True
                    break
            if hit:
                break
        if not hit:
            return spot
    return "outside"


def nice_step(m: float) -> float:
    """A tick step that puts about three ticks between 0 and ``m``."""
    for s in (0.01, 0.02, 0.025, 0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10, 20, 25, 50, 100, 200, 250, 500):
        if m / s <= 3.2:
            return s
    return 1000


def mirrored_ticks(lo: float, hi: float) -> Dict[str, List[Any]]:
    """A mirrored DOS axis's ticks: magnitudes on both sides of zero ({"vals", "text"})."""
    m = max(abs(lo), abs(hi))
    step = nice_step(m)
    k0, k1 = int(math.ceil(lo / step - 1e-9)), int(math.floor(hi / step + 1e-9))
    vals = [_round_step(k * step) for k in range(k0, k1 + 1)]
    return {"vals": vals, "text": [js_num(abs(v)) for v in vals]}


def _round_step(x: float) -> float:
    return math.floor(x * 1e6 + 0.5) / 1e6


# ---------------------------------------------------------------------------
# spin state of a DOS (one rule for the GUI and mace plotting, B2)
# ---------------------------------------------------------------------------
DOS_SPIN_REL = 1e-6


def dos_spin_polarized(up: Sequence[Sequence[float]], down: Optional[Sequence[Sequence[float]]]) -> bool:
    """True when the run's spin-down DOS is not spin-up's mirror image: CRYSTAL writes both blocks
    even for a closed-shell run, the second the exact negative of the first. A closed-shell run is
    drawn one-sided, a spin-polarized one mirrored (unless the user asks otherwise).
    ``up``/``down``: the columns of each block (down as written, negative)."""
    if down is None:
        return False
    peak = 0.0
    diff = 0.0
    for u, d in zip(up, down):
        for a, b in zip(u, d):
            peak = max(peak, abs(a))
            diff = max(diff, abs(a + b))
    return bool(diff > DOS_SPIN_REL * (peak or 1.0))


# ---------------------------------------------------------------------------
# mace plotting (matplotlib)
# ---------------------------------------------------------------------------
MPL_DASH = {"solid": "-", "dash": (0, (5, 2)), "dot": (0, (1, 1.5)), "dashdot": (0, (5, 1.5, 1, 1.5)),
            "longdash": (0, (9, 3)), "longdashdot": (0, (9, 2, 1, 2))}
MPL_MARKER = {"circle": "o", "square": "s", "triangle-up": "^", "diamond": "D", "triangle-down": "v", "cross": "x"}


def mpl_dash(name: Optional[str]):
    return MPL_DASH.get(name or "solid", "-")


def mpl_rc(pid: Optional[str], family: str = "band") -> Dict[str, Any]:
    """matplotlib rcParams of a preset (fonts, frame, ticks, grid) for mace plotting's figures."""
    p = get(pid, family)
    rc = {"font.size": p["font"], "axes.labelsize": p["font"], "xtick.labelsize": p["tick_font"],
          "ytick.labelsize": p["tick_font"], "legend.fontsize": max(8, p["tick_font"] - 1),
          "axes.linewidth": 1.2 * p["frame"], "axes.edgecolor": "black", "axes.labelweight": "normal"}
    if p["ticks"] == "outside":
        rc.update({"xtick.direction": "out", "ytick.direction": "out"})
    return rc


def series_style(pid: Optional[str], family: str, k: int) -> Dict[str, Any]:
    """The k-th series of a figure: {"color", "dash", "marker"} (Grayscale: by line style and marker)."""
    p = get(pid, family)
    pal = p.get("palette") or PRESETS["default"]["palette"]
    out = {"color": pal[k % len(pal)], "dash": "solid", "marker": None}
    if p.get("dashes"):
        out["dash"] = p["dashes"][k % len(p["dashes"])]
    if p.get("markers"):
        out["marker"] = p["markers"][k % len(p["markers"])]
    return out


def element_colors(name: Optional[str] = None) -> Dict[str, str]:
    """The active element palette (mace plotting --palette, else the saved default, else VESTA)."""
    try:
        from mace.utils import element_palettes
        return dict(element_palettes.load_colors(name) or {})
    except Exception:
        return {}


JS_DATA = "mace/gui/static/js/plotpresets_data.js"


def js_module() -> str:
    """mace/gui/static/js/plotpresets_data.js: ``SHARED`` for the GUI (regenerate after a change:
    ``python -m mace.plotting.presets --js``; tests/test_plot_presets.py checks it is current)."""
    return ("/* GENERATED by `python -m mace.plotting.presets --js` from mace/plotting/presets.py:\n"
            " * the plot presets, spin colors and DOS coloring rules the GUI shares with mace plotting.\n"
            " * Do not edit here. */\n"
            + _mark_names(f"export const SHARED = {json.dumps(SHARED, sort_keys=True, indent=1, ensure_ascii=False)};\n"))


def _mark_names(text: str) -> str:
    """A colormap's own name ("Greys") marked for tests/test_american_english.py."""   # british-ok: a colormap name
    return "".join(line + (" // british-ok: a colormap name" if '"Greys"' in line else "") + "\n"
                   for line in text.splitlines())


if __name__ == "__main__":   # pragma: no cover
    import sys
    from pathlib import Path
    if "--js" in sys.argv:
        out = Path(__file__).resolve().parents[2] / JS_DATA
        out.write_text(js_module())
        print(f"wrote {out}")
    else:
        print(shared_json())
