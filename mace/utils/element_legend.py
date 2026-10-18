"""Element legends for figures: one function for ``mace plotting --legend``
and the GUI's Export Element Legend (and the legend in exported images).

``legend()`` takes structure files (CIF, CRYSTAL .d12 and .out, POSCAR,
XYZ, ... read the way the GUI reads them) or folders of them, or symbols
directly, collects the elements (one structure, or the union of a batch),
colors them with an element palette (mace/utils/element_palettes.py: the
saved default, VESTA when none is saved, or the one named) and writes:

- SVG: clean, editable vector; each element is a ``<g id="legend-Fe">``
  with its swatch (circle or square) and its label as real ``<text>``, so
  the legend drops into Inkscape, Illustrator or PowerPoint and stays
  editable;
- PNG: the same layout drawn by matplotlib, at a DPI, transparent or on a
  background;
- a table (TSV, CSV or JSON): symbol, name, Z, hex color and RGB, for
  building a legend in another program.

Layouts: vertical, horizontal or a grid of N columns; elements sorted by
atomic number (default), by first appearance or alphabetically.
"""
from __future__ import annotations

import csv
import io
import json
import math
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Callable, Any, Dict, Iterable, List, Optional, Sequence, Tuple
from xml.sax.saxutils import escape

from . import element_palettes as EP

FORMATS = ("svg", "png", "tsv", "csv", "json")
LAYOUTS = ("vertical", "horizontal", "grid")
SHAPES = ("circle", "square")
LABELS = ("symbol", "name", "both")
ORDERS = ("z", "first", "alpha")
FONT = "Helvetica, Arial, sans-serif"
# text colors by what the legend sits on: dark text for a light background, light for a dark one
TEXT = {"dark": "#000000", "light": "#ffffff"}
# a PNG legend's total size (it is drawn in memory: RGBA, 4 bytes a pixel)
MAX_PNG_PIXELS = 50_000_000
STRUCTURE_SUFFIXES = (".cif", ".d12", ".out", ".log", ".xyz", ".extxyz", ".vasp", ".poscar", ".gen", ".pdb", ".mol")
_POSCAR = re.compile(r"^(POSCAR|CONTCAR)([._-]|$)")


@dataclass
class LegendOptions:
    layout: str = "vertical"
    columns: int = 4                  # grid only
    shape: str = "circle"
    swatch: float = 18.0              # px (1/96 in)
    font_size: float = 14.0           # px
    label: str = "symbol"
    text: str = "dark"                # "dark", "light" or "#rrggbb"
    background: Optional[str] = None  # None: transparent; else "#rrggbb"
    outline: bool = True              # a thin dark ring around each swatch (white H on white paper)
    order: str = "z"
    dpi: int = 300                    # PNG only

    def checked(self, size: Optional[Tuple[float, float]] = None) -> "LegendOptions":
        """A copy with every value in range; ValueError names the first bad one.
        With `size` (the layout's width and height in px at 96 dpi) the PNG
        that size makes at o.dpi is checked too: at most MAX_PNG_PIXELS."""
        o = LegendOptions(**asdict(self))
        for name, allowed in (("layout", LAYOUTS), ("shape", SHAPES), ("label", LABELS), ("order", ORDERS)):
            if getattr(o, name) not in allowed:
                raise ValueError(f"{name} must be one of {', '.join(allowed)}")
        o.columns = int(o.columns)
        if not 1 <= o.columns <= 30:
            raise ValueError("columns must be 1 to 30")
        o.swatch, o.font_size, o.dpi = float(o.swatch), float(o.font_size), int(o.dpi)
        if not 4 <= o.swatch <= 200:
            raise ValueError("swatch must be 4 to 200 px")
        if not 4 <= o.font_size <= 120:
            raise ValueError("font size must be 4 to 120 px")
        if not 30 <= o.dpi <= 2400:
            raise ValueError("dpi must be 30 to 2400")
        if o.text not in TEXT and EP.normalize_color(o.text) is None:
            raise ValueError("text must be dark, light or #rrggbb")
        if o.background in ("", "none", "transparent"):
            o.background = None
        if o.background is not None:
            bg = {"white": "#ffffff", "black": "#000000"}.get(o.background, o.background)
            if EP.normalize_color(bg) is None:
                raise ValueError("background must be none, white, black or #rrggbb")
            o.background = EP.normalize_color(bg)
        if size is not None:
            w, h = (float(x) * o.dpi / 96 for x in size)
            if w * h > MAX_PNG_PIXELS:
                fit = max(30, int(96 * math.sqrt(MAX_PNG_PIXELS / (float(size[0]) * float(size[1])))))
                raise ValueError(f"the PNG would be {w:.0f} x {h:.0f} px ({w * h / 1e6:.0f} megapixels; the limit is "
                                 f"{MAX_PNG_PIXELS / 1e6:.0f}): use at most {fit} dpi, a smaller font or swatch, "
                                 f"or SVG")
        return o

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "LegendOptions":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in names}).checked()


# ---------------------------------------------------------------- elements
def _names() -> Sequence[str]:
    try:
        from ase.data import atomic_names
        return atomic_names
    except ImportError:          # pragma: no cover - ase is a MACE dependency
        return [""] * 119


def atomic_number(sym: str) -> int:
    return EP.SYMBOLS.index(sym) + 1 if sym in EP.SYMBOLS else 0


def is_structure_file(p: Path) -> bool:
    return p.suffix.lower() in STRUCTURE_SUFFIXES or bool(_POSCAR.match(p.name))


def expand(paths: Iterable[Any], recursive: bool = False, inside: Optional[Callable[[Path], bool]] = None) -> List[Path]:
    """Files as given; a folder gives its structure files (sorted; hidden ones skipped; with ``inside``,
    the ones it accepts: the GUI leaves out a symbolic link out of the served folder)."""
    out: List[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            it = p.rglob("*") if recursive else p.iterdir()
            out += sorted(f for f in it if f.is_file() and is_structure_file(f)
                          and not any(part.startswith(".") for part in f.relative_to(p).parts)
                          and (inside is None or inside(f)))
        else:
            out.append(p)
    return out


def symbols_of(path: Path) -> List[str]:
    """The element symbols of one structure file, in file order (the GUI's reader)."""
    from mace.gui.structures import read_structure
    s = read_structure(Path(path))
    return [str(x) for x in s["symbols"]]


def collect(paths: Iterable[Any], order: str = "z", recursive: bool = False,
            inside: Optional[Callable[[Path], bool]] = None
            ) -> Tuple[List[str], Dict[str, List[str]], List[Tuple[str, str]]]:
    """(elements, per-file elements, skipped files with why): the union over every file."""
    seen: List[str] = []
    per: Dict[str, List[str]] = {}
    skipped: List[Tuple[str, str]] = []
    for f in expand(paths, recursive, inside):
        try:
            syms = symbols_of(f)
        except Exception as e:      # noqa: BLE001 - one unreadable file never stops the batch
            skipped.append((str(f), str(e) or type(e).__name__))
            continue
        per[str(f)] = sorted(set(syms), key=syms.index)
        for s in syms:
            if s not in seen:
                seen.append(s)
    return sort_symbols(seen, order), per, skipped


def sort_symbols(symbols: Iterable[str], order: str = "z") -> List[str]:
    seen: List[str] = []
    for s in symbols:
        s = str(s).strip()
        s = s[:1].upper() + s[1:].lower()
        if s and s not in seen:
            seen.append(s)
    if order == "z":
        return sorted(seen, key=lambda s: (atomic_number(s) or 999, s))
    if order == "alpha":
        return sorted(seen)
    return seen


def palette_colors(palette: Optional[str] = None, colors: Optional[Dict[str, str]] = None) -> Tuple[str, Dict[str, str]]:
    """(palette id, {symbol: hex}): given colors (the GUI's unsaved edit) over the named palette,
    else the saved default, else VESTA. The legacy preset is its full table here."""
    user = EP.read_user()
    pid = EP.find_id(palette, user) if palette else (EP.default_id(user) or EP.presets()["default"])
    out = EP.resolve(pid, user)
    if colors:
        out = {**out, **EP.clean_colors(colors)}
    return pid, out


def rows(symbols: Sequence[str], colors: Dict[str, str]) -> List[Dict[str, Any]]:
    names = _names()
    out = []
    for s in symbols:
        z = atomic_number(s)
        hx = colors.get(s) or EP.fallback_color()
        out.append({"symbol": s, "name": names[z] if 0 < z < len(names) else s, "z": z, "hex": hx,
                    "rgb": [int(hx[i:i + 2], 16) for i in (1, 3, 5)]})
    return out


# ---------------------------------------------------------------- layout (shared by SVG and PNG)
def _label(r: Dict[str, Any], o: LegendOptions) -> Tuple[str, str]:
    """(bold part, plain part) of a row's label."""
    if o.label == "symbol":
        return r["symbol"], ""
    if o.label == "name":
        return "", r["name"]
    return r["symbol"], f" {r['name']}"


def _text_width(bold: str, plain: str, f: float) -> float:
    # an average advance of about 0.6 em (a little more in bold), with room for wider fonts (DejaVu Sans)
    return (0.72 * len(bold) + 0.62 * len(plain)) * f


def layout(table: List[Dict[str, Any]], o: LegendOptions) -> Dict[str, Any]:
    """Every item's swatch center and text origin, and the picture's size, in px."""
    s, f = o.swatch, o.font_size
    pad, gap, colgap, rowgap = round(0.5 * s, 2), round(0.45 * s, 2), round(1.2 * s, 2), round(0.35 * s, 2)
    n = len(table)
    cols = 1 if o.layout == "vertical" else max(1, n) if o.layout == "horizontal" else min(o.columns, max(1, n))
    nrows = -(-n // cols) if n else 0
    item_h = max(s, 1.25 * f)
    widths = [0.0] * cols
    for k, r in enumerate(table):
        widths[k % cols] = max(widths[k % cols], s + gap + _text_width(*_label(r, o), f))
    items = []
    for k, r in enumerate(table):
        c, row = k % cols, k // cols
        x0 = pad + sum(widths[:c]) + c * colgap
        cy = pad + row * (item_h + rowgap) + item_h / 2
        items.append({"row": r, "cx": x0 + s / 2, "cy": cy, "tx": x0 + s + gap, "label": _label(r, o)})
    width = pad * 2 + sum(widths) + max(0, cols - 1) * colgap
    height = pad * 2 + nrows * item_h + max(0, nrows - 1) * rowgap
    return {"items": items, "width": round(max(width, 2 * pad), 2), "height": round(max(height, 2 * pad), 2)}


def _n(x: float) -> str:
    return f"{x:.2f}".rstrip("0").rstrip(".")


def text_color(o: LegendOptions) -> str:
    return TEXT.get(o.text) or EP.normalize_color(o.text) or TEXT["dark"]


# ---------------------------------------------------------------- outputs
def svg(table: List[Dict[str, Any]], o: LegendOptions, title: str = "Element legend") -> str:
    lay = layout(table, o)
    W, H, s, txt = lay["width"], lay["height"], o.swatch, text_color(o)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{_n(W)}" height="{_n(H)}" viewBox="0 0 {_n(W)} {_n(H)}">',
           f"<title>{escape(title)}</title>"]
    if o.background:
        out.append(f'<rect id="legend-background" width="{_n(W)}" height="{_n(H)}" fill="{o.background}"/>')
    out.append(f'<g id="element-legend" font-family="{FONT}" font-size="{_n(o.font_size)}" fill="{txt}">')
    ring = ' stroke="#000000" stroke-opacity="0.35" stroke-width="1"' if o.outline else ""
    for it in lay["items"]:
        r = it["row"]
        out.append(f'<g id="legend-{r["symbol"]}" data-element="{r["symbol"]}" data-z="{r["z"]}">')
        if o.shape == "circle":
            out.append(f'<circle cx="{_n(it["cx"])}" cy="{_n(it["cy"])}" r="{_n(s / 2)}" fill="{r["hex"]}"{ring}/>')
        else:
            out.append(f'<rect x="{_n(it["cx"] - s / 2)}" y="{_n(it["cy"] - s / 2)}" width="{_n(s)}" height="{_n(s)}" '
                       f'rx="{_n(s * 0.12)}" fill="{r["hex"]}"{ring}/>')
        bold, plain = it["label"]
        spans = (f'<tspan font-weight="bold">{escape(bold)}</tspan>' if bold else "") + (escape(plain) if plain else "")
        # the baseline a third of the font size below the swatch's center: centered for Helvetica-like fonts
        out.append(f'<text x="{_n(it["tx"])}" y="{_n(it["cy"] + o.font_size * 0.35)}" xml:space="preserve">{spans}</text>')
        out.append("</g>")
    out += ["</g>", "</svg>"]
    return "\n".join(out) + "\n"


def png(table: List[Dict[str, Any]], o: LegendOptions) -> bytes:
    """The SVG's layout drawn by matplotlib at o.dpi (transparent unless o.background)."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.patches import Circle, FancyBboxPatch
    lay = layout(table, o)
    o.checked(size=(lay["width"], lay["height"]))
    W, H, s = lay["width"], lay["height"], o.swatch
    fig = Figure(figsize=(W / 96, H / 96), dpi=o.dpi)
    FigureCanvasAgg(fig)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    if o.background:
        fig.patch.set_facecolor(o.background)
    ec = (0, 0, 0, 0.35) if o.outline else "none"
    pt = 72 / 96          # px -> pt
    for it in lay["items"]:
        r = it["row"]
        if o.shape == "circle":
            ax.add_patch(Circle((it["cx"], it["cy"]), s / 2, facecolor=r["hex"], edgecolor=ec, linewidth=1 * pt))
        else:
            ax.add_patch(FancyBboxPatch((it["cx"] - s / 2, it["cy"] - s / 2), s, s, boxstyle=f"round,pad=0,rounding_size={s * 0.12}",
                                        facecolor=r["hex"], edgecolor=ec, linewidth=1 * pt))
        bold, plain = it["label"]
        x = it["tx"]
        kw = dict(va="baseline", ha="left", color=text_color(o), fontsize=o.font_size * pt,
                  family="sans-serif")
        y = it["cy"] + o.font_size * 0.35
        if bold:
            t = ax.text(x, y, bold, fontweight="bold", **kw)
            # the plain part starts where the bold one really ends (display px -> layout px)
            x += t.get_window_extent(fig.canvas.get_renderer()).width * 96 / o.dpi
        if plain:
            ax.text(x, y, plain, **kw)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=o.dpi, transparent=o.background is None,
                facecolor=o.background if o.background else "none")
    return buf.getvalue()


def table_text(table: List[Dict[str, Any]], fmt: str) -> str:
    if fmt == "json":
        return json.dumps([{**r, "rgb": list(r["rgb"])} for r in table], indent=2) + "\n"
    buf = io.StringIO()
    w = csv.writer(buf, delimiter="\t" if fmt == "tsv" else ",", lineterminator="\n")
    w.writerow(["symbol", "name", "z", "hex", "r", "g", "b"])
    for r in table:
        w.writerow([r["symbol"], r["name"], r["z"], r["hex"], *r["rgb"]])
    return buf.getvalue()


@dataclass
class Legend:
    """What legend() made: the content of the requested format and what it shows."""
    format: str
    content: bytes
    symbols: List[str]
    palette: str
    rows: List[Dict[str, Any]]
    width: float
    height: float
    per_file: Dict[str, List[str]] = field(default_factory=dict)
    skipped: List[Tuple[str, str]] = field(default_factory=list)


def legend(paths: Optional[Iterable[Any]] = None, *, symbols: Optional[Sequence[str]] = None, fmt: str = "svg",
           palette: Optional[str] = None, colors: Optional[Dict[str, str]] = None,
           options: Optional[Any] = None, recursive: bool = False, title: str = "Element legend",
           inside: Optional[Callable[[Path], bool]] = None) -> Legend:
    """The one entry point (CLI and GUI): the legend of the union of the elements
    in `paths` (files and folders) and `symbols` (an unsaved structure), in `fmt`."""
    if fmt not in FORMATS:
        raise ValueError(f"format must be one of {', '.join(FORMATS)}")
    o = options if isinstance(options, LegendOptions) else LegendOptions.from_dict(options)
    o = o.checked()
    per: Dict[str, List[str]] = {}
    skipped: List[Tuple[str, str]] = []
    syms: List[str] = []
    if paths is not None:
        syms, per, skipped = collect(paths, "first", recursive, inside)
    if symbols is not None:
        bad = [s for s in sort_symbols(symbols, "first") if s not in EP.SYMBOLS]
        if bad:
            raise ValueError(f"not element symbols: {', '.join(bad)}")
        syms = syms + list(symbols)
    syms = sort_symbols(syms, o.order)
    if not syms:
        raise ValueError("no elements found" + (f" ({len(skipped)} file(s) could not be read)" if skipped else ""))
    pid, cmap = palette_colors(palette, colors)
    table = rows(syms, cmap)
    lay = layout(table, o)
    if fmt == "png":
        o.checked(size=(lay["width"], lay["height"]))     # before anything is drawn
    if fmt == "svg":
        content = svg(table, o, title).encode()
    elif fmt == "png":
        content = png(table, o)
    else:
        content = table_text(table, fmt).encode()
    return Legend(fmt, content, syms, pid, table, lay["width"], lay["height"], per, skipped)
