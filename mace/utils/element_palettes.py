"""Element color palettes: one loader for the GUI and the CLI plotters.

Built-in presets live in ``mace/gui/static/element_colors.json`` (the
VESTA and Jmol colors of pymatgen, MIT, and MACE's own colorblind-safe
scheme; see ``element_colors.LICENSE.txt``). Your own palettes and your
default live in ``$XDG_CONFIG_HOME/mace/element_colors.json`` (default
``~/.config/mace/element_colors.json``), written by the GUI (View ›
Element Colors…) and read by both the GUI and ``mace plotting``:

.. code-block:: json

    {
      "version": 1,
      "rev": 3,
      "default": "user:My oxides",
      "palettes": {
        "My oxides": {"base": "vesta", "colors": {"O": "#d55e00", "Ti": "#56b4e9"}}
      }
    }

``default`` is a palette id: a preset name (``vesta``, ``jmol``, ``cvd``,
``legacy``)
or ``user:<name>`` for one of ``palettes``. A user palette stores only the
elements it changes; every other element takes its ``base`` preset's color
(so "reset to preset" for an element just drops it from ``colors``).
Colors are ``#rrggbb``. The file may be edited by hand; a bad entry is
skipped (with a warning), never fatal.

An exported palette (the GUI's Export…) is a self-contained file::

    {"mace_element_palette": 1, "name": "...", "base": "vesta",
     "colors": {...changed...}, "elements": {...every element...}}

Only the standard library is used here, so the CLI can import it cheaply.
"""
from __future__ import annotations

import json
import os
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PRESETS_PATH = Path(__file__).resolve().parents[1] / "gui" / "static" / "element_colors.json"
FILE_NAME = "element_colors.json"
VERSION = 1
EXPORT_KEY = "mace_element_palette"
USER_PREFIX = "user:"
# the colors mace plotting drew before 1.1.6: in the CLI each plotter keeps its own
# table exactly; in the GUI the preset is the structure SVG plotter's table
LEGACY = "legacy"
MAX_PALETTES = 50
MAX_NAME = 60
MAX_BYTES = 256_000
_HEX6 = re.compile(r"^#[0-9a-fA-F]{6}$")
_HEX3 = re.compile(r"^#[0-9a-fA-F]{3}$")
_warned: set = set()

# Z = 1..118
SYMBOLS: Tuple[str, ...] = tuple((
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr "
    "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb "
    "Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr "
    "Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og").split())
_SYMSET = frozenset(SYMBOLS)


def config_path() -> Path:
    """``$XDG_CONFIG_HOME/mace/element_colors.json`` (beside the GUI's gui.json)."""
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return Path(base) / "mace" / FILE_NAME


def _warn(msg: str) -> None:
    if msg not in _warned:
        _warned.add(msg)
        print(f"mace: {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------- presets
@lru_cache(maxsize=1)
def presets() -> Dict[str, Any]:
    """{"default", "fallback", "schemes": {id: {sym: hex}}, "labels": {id: text}}."""
    data = json.loads(PRESETS_PATH.read_text())
    labels = data.get("labels") or {}
    data["labels"] = {k: labels.get(k, k) for k in data["schemes"]}
    return data


def preset_ids() -> List[str]:
    p = presets()
    order = [p["default"]] + sorted(k for k in p["schemes"] if k != p["default"])
    return order


def fallback_color() -> str:
    return presets()["fallback"]


# ---------------------------------------------------------------- checks
def normalize_color(value: Any) -> Optional[str]:
    """'#rrggbb' (lower case) from '#rgb' or '#rrggbb'; None if not a color."""
    if not isinstance(value, str):
        return None
    v = value.strip()
    if _HEX3.match(v):
        v = "#" + "".join(c * 2 for c in v[1:])
    return v.lower() if _HEX6.match(v) else None


def clean_name(name: Any) -> str:
    if not isinstance(name, str):
        raise ValueError("a palette name must be text")
    n = " ".join(name.split())
    if not n:
        raise ValueError("a palette name cannot be empty")
    if len(n) > MAX_NAME:
        raise ValueError(f"a palette name has at most {MAX_NAME} characters")
    if any(ord(c) < 32 for c in n):
        raise ValueError("a palette name cannot hold control characters")
    return n


def clean_colors(colors: Any) -> Dict[str, str]:
    """{symbol: '#rrggbb'}; ValueError naming the first bad entry."""
    if colors is None:
        return {}
    if not isinstance(colors, dict):
        raise ValueError("colors must be an object of element: '#rrggbb'")
    out = {}
    for sym, c in colors.items():
        if sym not in _SYMSET:
            raise ValueError(f"{sym!r} is not an element symbol")
        hx = normalize_color(c)
        if hx is None:
            raise ValueError(f"the color of {sym} must be '#rrggbb', not {c!r}")
        out[sym] = hx
    return out


def clean_palette(obj: Any) -> Dict[str, Any]:
    """A stored palette: {"base": preset id, "colors": {...}}; ValueError if unusable."""
    if not isinstance(obj, dict):
        raise ValueError("a palette must be an object")
    base = obj.get("base") or presets()["default"]
    if base not in presets()["schemes"]:
        raise ValueError(f"base must be one of {', '.join(preset_ids())}")
    colors = clean_colors(obj.get("colors"))
    # a color that equals the preset's is not a change: kept out, so reset-to-preset stays exact
    preset = presets()["schemes"][base]
    return {"base": base, "colors": {s: c for s, c in sorted(colors.items()) if preset.get(s) != c}}


# ---------------------------------------------------------------- the user file
def _empty() -> Dict[str, Any]:
    return {"version": VERSION, "rev": 0, "default": None, "palettes": {}}


def normalize_doc(data: Any, where: str = "") -> Dict[str, Any]:
    """The usable part of a stored file: bad palettes and a bad default are
    dropped (with a warning), never fatal."""
    out = _empty()
    if not isinstance(data, dict):
        return out
    rev = data.get("rev")
    out["rev"] = rev if isinstance(rev, int) and not isinstance(rev, bool) else 0
    pals = data.get("palettes") if isinstance(data.get("palettes"), dict) else {}
    for name, pal in list(pals.items())[:MAX_PALETTES]:
        try:
            out["palettes"][clean_name(name)] = clean_palette(pal)
        except ValueError as e:
            _warn(f"{where or config_path()}: palette {name!r} skipped: {e}")
    d = data.get("default")
    if isinstance(d, str) and is_known(d, out):
        out["default"] = d
    elif d not in (None, ""):
        _warn(f"{where or config_path()}: default {d!r} is not a known palette; using the built-in default")
    return out


def read_user(path: Optional[Path] = None) -> Dict[str, Any]:
    """The user's palettes file, normalized; an absent or unreadable file is empty."""
    p = Path(path) if path else config_path()
    try:
        data = json.loads(p.read_text())
    except FileNotFoundError:
        return _empty()
    except (OSError, ValueError) as e:
        _warn(f"{p} is not readable ({e}); using the built-in element colors")
        return _empty()
    return normalize_doc(data, str(p))


def is_known(pid: str, user: Optional[Dict[str, Any]] = None) -> bool:
    if pid in presets()["schemes"]:
        return True
    if isinstance(pid, str) and pid.startswith(USER_PREFIX):
        return pid[len(USER_PREFIX):] in ((user or {}).get("palettes") or {})
    return False


def palette_ids(user: Optional[Dict[str, Any]] = None) -> List[str]:
    user = read_user() if user is None else user
    return preset_ids() + [USER_PREFIX + n for n in sorted(user["palettes"], key=str.lower)]


def label_of(pid: str) -> str:
    if pid.startswith(USER_PREFIX):
        return pid[len(USER_PREFIX):]
    return presets()["labels"].get(pid, pid)


def find_id(name: str, user: Optional[Dict[str, Any]] = None) -> str:
    """A palette id from what a person types: an id ('vesta', 'user:Mine'),
    a user palette's name ('Mine') or a preset's label, any case."""
    user = read_user() if user is None else user
    if is_known(name, user):
        return name
    want = " ".join(str(name).split()).lower()
    for pid in palette_ids(user):
        if want in (pid.lower(), label_of(pid).lower()):
            return pid
    raise ValueError(f"no element color palette named {name!r}; "
                     f"choose one of: {', '.join(palette_ids(user))}")


def resolve(pid: str, user: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """Every element's color in palette `pid` (KeyError if unknown)."""
    p = presets()
    if pid in p["schemes"]:
        return dict(p["schemes"][pid])
    user = read_user() if user is None else user
    if not is_known(pid, user):
        raise KeyError(pid)
    pal = user["palettes"][pid[len(USER_PREFIX):]]
    return {**p["schemes"][pal["base"]], **pal["colors"]}


def default_id(user: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """The saved default, or None when none was ever saved."""
    user = read_user() if user is None else user
    return user.get("default")


def load_colors(name: Optional[str] = None, user: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, str]]:
    """The CLI's entry point: `name` (a --palette value), else the saved
    default, else the built-in default (VESTA, as the GUI), as {symbol:
    '#rrggbb'}. None for the "legacy" preset: each plotter then draws its
    own pre-1.1.6 colors exactly. ValueError for an unknown name."""
    user = read_user() if user is None else user
    pid = find_id(name, user) if name else (default_id(user) or presets()["default"])
    return None if pid == LEGACY else resolve(pid, user)


def color_for(colors: Dict[str, str], symbol: str) -> str:
    """One element's color in a resolved palette, any case ('FE', 'fe', 'Fe')."""
    s = str(symbol).strip()
    s = s[:1].upper() + s[1:].lower()
    return colors.get(s) or fallback_color()


def darker(hex_color: str, f: float = 0.5) -> str:
    """The same hue, darker (outlines of 2D atom discs)."""
    c = normalize_color(hex_color) or fallback_color()
    return "#" + "".join(f"{max(0, min(255, round(int(c[i:i + 2], 16) * f))):02x}" for i in (1, 3, 5))


# ---------------------------------------------------------------- export / import
def export_doc(pid: str, user: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    user = read_user() if user is None else user
    if pid in presets()["schemes"]:
        base, colors, name = pid, {}, label_of(pid)
    else:
        if not is_known(pid, user):
            raise KeyError(pid)
        name = pid[len(USER_PREFIX):]
        base, colors = user["palettes"][name]["base"], dict(user["palettes"][name]["colors"])
    return {EXPORT_KEY: 1, "name": name, "base": base, "colors": colors,
            "elements": resolve(pid, user)}


def parse_import(obj: Any) -> Tuple[str, Dict[str, Any]]:
    """(name, palette) from an exported file. `colors` (the changes) wins; a
    file with only `elements` (another program's table) becomes a palette
    with every differing element over its base."""
    if not isinstance(obj, dict) or obj.get(EXPORT_KEY) != 1:
        raise ValueError(f"not a MACE element palette file (no \"{EXPORT_KEY}\": 1)")
    name = clean_name(obj.get("name") or "Imported")
    base = obj.get("base") or presets()["default"]
    if base not in presets()["schemes"]:
        base = presets()["default"]
    colors = dict(clean_colors(obj.get("elements")))
    colors.update(clean_colors(obj.get("colors")))
    return name, clean_palette({"base": base, "colors": colors})


def unique_name(name: str, taken) -> str:
    if name not in taken:
        return name
    k = 2
    while f"{name} ({k})" in taken:
        k += 1
    return f"{name} ({k})"


# ---------------------------------------------------------------- changes (pure)
def apply_op(cur: Optional[Dict[str, Any]], op: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """The file after one change, and what the change made ({"id": ...}).

    ops: {"op": "save", "name", "base", "colors", "replace": bool}
         {"op": "delete", "name"}
         {"op": "rename", "name", "to"}
         {"op": "default", "id"}            (id null: no saved default)
         {"op": "import", "palette": <exported file>}
    ValueError for a bad request (nothing changes)."""
    doc = normalize_doc(cur or {})
    pals = dict(doc["palettes"])
    kind = op.get("op") if isinstance(op, dict) else None
    made: Dict[str, Any] = {}
    if kind == "save":
        name = clean_name(op.get("name"))
        if name not in pals and len(pals) >= MAX_PALETTES:
            raise ValueError(f"at most {MAX_PALETTES} palettes; delete one first")
        if name in pals and not op.get("replace", True):
            raise ValueError(f"a palette named {name!r} exists")
        pals[name] = clean_palette({"base": op.get("base"), "colors": op.get("colors")})
        made["id"] = USER_PREFIX + name
    elif kind == "delete":
        name = clean_name(op.get("name"))
        if name not in pals:
            raise ValueError(f"no palette named {name!r}")
        del pals[name]
        if doc["default"] == USER_PREFIX + name:
            doc["default"] = None
    elif kind == "rename":
        name, to = clean_name(op.get("name")), clean_name(op.get("to"))
        if name not in pals:
            raise ValueError(f"no palette named {name!r}")
        if to != name and to in pals:
            raise ValueError(f"a palette named {to!r} exists")
        pals = {(to if k == name else k): v for k, v in pals.items()}
        if doc["default"] == USER_PREFIX + name:
            doc["default"] = USER_PREFIX + to
        made["id"] = USER_PREFIX + to
    elif kind == "default":
        pid = op.get("id")
        if pid not in (None, "") and not is_known(pid, {"palettes": pals}):
            raise ValueError(f"no palette {pid!r}")
        doc["default"] = pid or None
        made["id"] = doc["default"]
    elif kind == "import":
        name, pal = parse_import(op.get("palette"))
        if len(pals) >= MAX_PALETTES:
            raise ValueError(f"at most {MAX_PALETTES} palettes; delete one first")
        name = unique_name(name, pals)
        pals[name] = pal
        made["id"] = USER_PREFIX + name
    else:
        raise ValueError("op must be save, delete, rename, default or import")
    doc["palettes"] = pals
    doc["rev"] = doc["rev"] + 1
    doc["version"] = VERSION
    if len(json.dumps(doc)) > MAX_BYTES:
        raise ValueError("the palettes file would be too large")
    return doc, made


# ---------------------------------------------------------------- the CLI's choice for this run
_choice: Optional[str] = None
_active_cache: Tuple[Any, Optional[Dict[str, str]]] = (None, None)


def choose(name: Optional[str]) -> Optional[str]:
    """Remember `mace plotting --palette NAME` for this run (None: the saved
    default). Returns the palette id; ValueError for an unknown name."""
    global _choice, _active_cache
    _choice = find_id(name) if name else None
    _active_cache = (None, None)
    return _choice


def active_colors() -> Optional[Dict[str, str]]:
    """The colors the CLI plotters draw atoms in this run: the --palette
    choice, else the saved default, else VESTA; None for "legacy" (the
    plotter's own pre-1.1.6 colors).
    Cached until the file changes, so a plotter may call it per atom."""
    global _active_cache
    try:
        st = config_path().stat()
        stamp = (st.st_mtime_ns, st.st_size)
    except OSError:
        stamp = None
    key = (_choice, stamp)
    if _active_cache[0] != key:
        try:
            _active_cache = (key, load_colors(_choice))
        except (ValueError, KeyError) as e:
            _warn(f"element colors: {e}")
            _active_cache = (key, None)
    return _active_cache[1]
