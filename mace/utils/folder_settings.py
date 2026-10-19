"""Per-folder settings: ``mace-folder.toml`` in a project folder.

A project folder (one you share with a group, or keep in git) can carry its
own display, plot, Calculation Setup, database and submit defaults. They
override your own defaults while you work in that folder:

* ``mace gui`` reads the file of the folder it serves; its Settings dialog
  shows where each value comes from and saves to this folder or to your
  own defaults (``$XDG_CONFIG_HOME/mace/gui.json``);
* CLI commands run inside the folder read the same file for the settings
  that have a CLI form: ``mace plotting`` (the band/DOS energy window),
  ``mace opt2d12`` / ``mace opt2d3`` (a preset config file) and ``mace
  convert`` (cif2d12's options file for ``--batch``), and ``mace submit``
  (job size and walltime per job kind, within what ``site.toml`` allows).

Precedence, per value: a change made in this GUI session or a CLI flag >
this folder's file > your own defaults (gui.json; site.toml for cluster
resources) > MACE's built-in values. Environment variables that set site
values (``MACE_SLURM_*``) and site.toml's ``[resources.<TYPE>]`` tables stay
above the folder's file, and site.toml's ``[resources.properties]`` stays
above its plain ``[submit]`` values (docs/cli.md lists the full order).

Where the file is found: the folder itself, then each parent folder up to
(not including) your home folder or the file-system root; the nearest file
wins (files are not merged). ``MACE_FOLDER_SETTINGS=none`` turns folder
settings off; ``MACE_FOLDER_SETTINGS=/path/mace-folder.toml`` uses that file.

The file holds data only: it is TOML with a fixed set of keys (``KEYS``),
each checked for its type and range. An unknown key or a bad value is
ignored with a warning, never an error. Paths in it are relative to the
file's folder and must stay inside it (no absolute paths, no ``..``, no
symbolic link that leads out). A file that others may change (writable by
everyone, or in a folder writable by everyone, such as a scratch root) or
a symbolic link is ignored. Nothing in it is ever run.

Standard library only: ``mace submit``'s job-script generators import it
on a cluster's module Python.
"""
from __future__ import annotations

import json
import math
import os
import re
import stat
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

FILENAME = "mace-folder.toml"
ENV = "MACE_FOLDER_SETTINGS"
MAX_BYTES = 64_000

# the DB view's columns (mace/gui/static/js/docs/db.js COLS)
DB_COLUMNS = ("material_id", "formula", "space_group", "dimensionality", "n_calcs", "n_completed", "status",
              "updated_at")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ()+._/-]{0,59}$")
FILE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
PARTITION_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
# A walltime as MACE takes it, here and in site.toml (site.py's _valid_walltime uses these):
# D-HH:MM:SS, D-HH:MM, D-HH or HH:MM:SS, at most 366 days. Never SLURM's MM and MM:SS forms:
# "12:00" (meant as 12 hours) is 12 minutes to SLURM.
WALLTIME_RE = re.compile(r"^(?:[0-9]{1,3}-[0-9]{1,2}(?::[0-5][0-9](?::[0-5][0-9])?)?|[0-9]{1,4}:[0-5][0-9]:[0-5][0-9])$")
MAX_WALLTIME_S = 366 * 86400


def _k(type_: str, gui: Optional[str] = None, help: str = "", **kw) -> Dict[str, Any]:
    return {"type": type_, "gui": gui, "help": help, **kw}


# Every key a folder file may hold: "<table>.<key>" -> its type, the GUI
# setting it sets (gui: the key in settings.js DEFAULTS) and its CLI use.
KEYS: Dict[str, Dict[str, Any]] = {
    # [viewer]: how structures are drawn (independent of the light/dark theme)
    # the element palette presets (mace/gui/static/element_colors.json and "legacy"; element_palettes.preset_ids());
    # a palette of a user's own lives in that user's ~/.config/mace/element_colors.json, not in a shared folder
    "viewer.color_scheme": _k("choice", "colorScheme", "Element colors (a preset palette)",
                              choices=("vesta", "jmol", "cvd", "legacy")),
    "viewer.atom_scale": _k("number", "atomScale", "Atom scale (ball and stick)", range=(0.05, 1.5)),
    "viewer.bond_radius": _k("number", "bondRadius", "Bond radius in Å", range=(0.01, 1.0)),
    "viewer.bond_tolerance": _k("number", "bondTolerance", "Drawn bonds: × the sum of covalent radii",
                                range=(1.0, 1.4)),
    "viewer.bonds_between_metals": _k("bool", "bondMetalMetal", "Draw bonds between metal atoms"),
    "viewer.labels": _k("choice", "labels", "Atom labels", choices=("off", "element", "index", "both", "wyckoff")),
    "viewer.projection": _k("choice", "projection", "Camera projection", choices=("orthographic", "perspective")),
    "viewer.default_view": _k("choice", "viewport.defaultView", "How a structure first shows",
                              choices=("standard", "a", "c")),
    "viewer.show_axes": _k("bool", "showAxes", "Axis triad"),
    "viewer.show_cell": _k("bool", "showCell", "Unit cell"),
    "viewer.show_boundary": _k("bool", "showBoundary", "Boundary atoms"),
    "viewer.show_polyhedra": _k("bool", "showPolyhedra", "Coordination polyhedra"),
    "viewer.display_from": _k("triple", "displayLo", "Drawn range start, fractional a, b, c", range=(-5.0, 10.0)),
    "viewer.display_to": _k("triple", "displayHi", "Drawn range end, fractional a, b, c", range=(-5.0, 10.0)),
    "viewer.bonded_outside": _k("choice", "bondedOutside", "Atoms outside the range bonded to atoms inside it",
                                choices=("off", "neighbors", "molecules")),
    # [units]
    "units.coordinates": _k("choice", "coordMode", "Fractional or Cartesian coordinates", choices=("frac", "cart")),
    "units.energy": _k("choice", "energyUnit", "Energy unit of plots and run charts",
                       choices=("eV", "meV", "Ha", "kJ/mol", "kcal/mol")),
    # [plots]
    "plots.relative_to": _k("choice", "plots.relTo", "Run charts: energies relative to",
                            choices=("first", "final", "minimum", "absolute")),
    "plots.segments": _k("choice", "plots.segments", "FINALRUN segments", choices=("continuous", "separate")),
    "plots.log_delta_e": _k("bool", "plots.logDeltaE", "Run energy charts open on the log |ΔE| view"),
    "plots.energy_per": _k("choice", "plots.per", "Energy per cell, atom or formula unit",
                           choices=("cell", "atom", "fu")),
    "plots.line_width": _k("number", "plots.lineWidth", "Line width in px", range=(0.5, 6.0)),
    "plots.energy_window": _k("pair", "plots.energyWindow", "Band/DOS energy window in eV around the Fermi level "
                              "(mace plotting --e-lower/--e-upper)", range=(-100.0, 100.0), cli=True),
    "plots.fermi_shift": _k("number", "plots.fermiShift", "Band/DOS Fermi shift in eV (mace plotting --fermi-shift)",
                            range=(-50.0, 50.0), cli=True),
    # 1.1.6: the plot style of every plot family, the DOS colors and spin down's line style (mace gui's
    # Settings and mace plotting --style/--dos-colors/--spin-style)
    "plots.style": _k("choice", "plots.style", "Plot style of every plot (mace plotting --style)",
                      choices=("default", "publication", "vesta", "presentation", "grayscale", "colorblind"), cli=True),
    "plots.dos_colors": _k("choice", "plots.dosColors", "DOS colors (mace plotting --dos-colors)",
                           choices=("auto", "total", "element", "orbital", "element-orbital"), cli=True),
    "plots.spin_down_style": _k("choice", "plots.spinDownDash", "Spin down's line style (mace plotting --spin-down-style; "
                                "--spin-down-solid)", choices=("dash", "dot", "dashdot", "solid"), cli=True),
    # the spin colors (presets.py SPIN_SCHEMES; auto: the plot style's), the DOS spin layout and total,
    # the energy zero and mace plotting's renderer: the same names as the GUI's Style controls and the flags
    "plots.spin_colors": _k("choice", "plots.spinColors", "Spin colors (mace plotting --spin-colors)",
                            choices=("auto", "mace", "blue-red", "black-vermilion", "custom"), cli=True),
    "plots.spin_up_color": _k("color", "plots.spinUp", "Spin up's custom color (mace plotting --spin-up-color)", cli=True),
    "plots.spin_down_color": _k("color", "plots.spinDown", "Spin down's custom color (mace plotting --spin-down-color)", cli=True),
    "plots.band_color": _k("color", "plots.bandColor", "The bands and one-sided DOS total of a run that is not "
                           "spin-polarized (mace plotting --band-color)", cli=True),
    "plots.dos_spin": _k("choice", "plots.spinDos", "DOS spin layout (mace plotting --dos-spin)",
                         choices=("auto", "mirrored", "overlaid"), cli=True),
    "plots.dos_total": _k("bool", "plots.dosTotal", "Draw the DOS total (mace plotting --dos-total/--no-dos-total)", cli=True),
    "plots.dos_smooth": _k("number", "plots.dosSmooth", "DOS smoothing: Gaussian sigma in eV, display only; 0 = off "
                           "(mace plotting --dos-smooth)", range=(0.0, 1.0), cli=True),
    "plots.energy_reference": _k("choice", None, "Band/DOS energy zero (mace plotting --energy-reference)",
                                 choices=("auto", "fermi", "vacuum"), cli=True),
    # electrostatic potential colors (1.1.7): chemistry red = negative (the default), physics red = positive;
    # 1.1.6's viewer.potential_colors reads as this key (RENAMED)
    "plots.potential_colors": _k("choice", "volume.mepColors", "Electrostatic potential colors: chemistry "
                                 "(red = negative) or physics (red = positive) (mace plotting --potential-colors)",
                                 choices=("chemistry", "physics"), cli=True),
    "plots.backend": _k("choice", None, "mace plotting's band/DOS renderer (--backend; default plotly when it can export)",
                        choices=("plotly", "matplotlib"), cli=True),
    "plots.ir_line_shape": _k("choice", "spectra.lineShape", "IR line shape", choices=("lorentzian", "gaussian")),
    "plots.imaginary_noise_cm": _k("number", "analysis.imagNoiseCm", "Imaginary modes below this (cm⁻¹) are noise",
                                   range=(0.0, 200.0)),
    # [setup]: Calculation Setup and the deck CLIs
    "setup.functional": _k("name", "functional", "Calculation Setup's and mace convert's functional for a deck without a parent ('-D3' turns D3 on)"),
    "setup.basis_set": _k("name", "basisSet", "Calculation Setup's and mace convert's internal basis set for a deck without a parent"),
    "setup.example_config": _k("name", "exampleConfig", "Calculation Setup's example config"),
    "setup.options_file_name": _k("filename", "optionsFileName", "Save Settings' file name"),
    "setup.opt2d12_config": _k("path", None, "mace opt2d12: --config-file when none is given", cli=True),
    "setup.opt2d3_config": _k("path", None, "mace opt2d3: --config-file when none is given", cli=True),
    "setup.cif2d12_options": _k("path", None, "mace convert --batch: --options_file when none is given", cli=True),
    # [database]
    "database.file": _k("path", "defaultDb", "What Database View opens"),
    "database.columns": _k("columns", "database.columns", "Database View's columns, in order"),
    "database.refresh_seconds": _k("choice", "database.refreshSeconds", "Refresh the Database view (0 = off)",
                                   choices=(0, 15, 30, 60)),
    # [submit]: merged with site.toml's rules (its [folders] table says what a folder may set)
    "submit.partition": _k("partition", None, "#SBATCH -p: one of site.toml's [folders] partitions", cli=True),
    "submit.ntasks": _k("int", None, "#SBATCH --ntasks of CRYSTAL jobs (over site.toml's [resources] ntasks)",
                        range=(1, 4096), cli=True),
    "submit.walltime": _k("walltime", None, "Walltime of CRYSTAL jobs, and of properties jobs when neither "
                          "[submit.properties] nor site.toml's [resources.properties] sets one", cli=True),
    "submit.properties.ntasks": _k("int", None, "#SBATCH --ntasks of properties jobs (over site.toml's "
                                   "[resources.properties] ntasks)", range=(1, 4096), cli=True),
    "submit.properties.walltime": _k("walltime", None, "Walltime of properties jobs (over site.toml's "
                                     "[resources.properties] walltime)", cli=True),
}
# keys of an earlier version: old name -> (new name, {old value: new value})
RENAMED: Dict[str, Tuple[str, Dict[Any, Any]]] = {
    # 1.1.6's "cli" was red = positive (mace plotting's colors then)
    "viewer.potential_colors": ("plots.potential_colors", {"cli": "physics", "chemistry": "chemistry"}),
}
GUI_KEYS: Dict[str, str] = {spec["gui"]: key for key, spec in KEYS.items() if spec["gui"]}
TABLES = sorted({k.rsplit(".", 1)[0] for k in KEYS})

TEMPLATE = """\
# MACE folder settings: defaults for everyone who works in this folder.
# They override each user's own defaults while MACE works here (mace gui and
# the CLI commands run in this folder or below it); a GUI change made in a
# session, or a CLI flag, still wins. Safe to commit to git or share with a group:
# the file holds data only, and paths in it stay inside this folder.
# Keys MACE does not know are ignored with a warning. Reference: docs/gui.md.

[viewer]
# color_scheme = "vesta"            # element colors: "vesta", "jmol", "cvd" or "legacy"
# atom_scale = 0.4                  # ball and stick, 0.05 to 1.5
# bond_tolerance = 1.15             # drawn bonds: x the sum of covalent radii, 1.0 to 1.4
# show_cell = true
# projection = "orthographic"       # or "perspective"

[units]
# energy = "eV"                     # eV, meV, Ha, kJ/mol, kcal/mol

[plots]
# energy_window = [-6.0, 6.0]       # band/DOS window in eV (mace plotting --e-lower/--e-upper)
# fermi_shift = 0.0                 # band/DOS Fermi shift in eV (mace plotting --fermi-shift)
# style = "publication"             # plot style: default, publication, vesta, presentation, grayscale, colorblind
# dos_colors = "auto"               # auto, total, element, orbital, element-orbital
# spin_down_style = "dash"          # dash, dot, dashdot, solid (spin down's line)
# spin_colors = "auto"              # auto (the style's), mace, blue-red, black-vermilion, custom
# band_color = "#000000"            # a run that is not spin-polarized: its bands and one-sided DOS total
# dos_spin = "auto"                 # auto, mirrored, overlaid
# dos_smooth = 0.05                 # DOS smoothing: Gaussian sigma in eV (display only); 0 = off
# backend = "plotly"                # mace plotting's band/DOS renderer: plotly or matplotlib
# log_delta_e = false               # run energy charts open on the log |dE| view
# relative_to = "final"             # first, final, minimum, absolute

[setup]
# functional = "HSE06"              # Calculation Setup's defaults (GUI)
# basis_set = "POB-TZVP-REV2"
# opt2d12_config = "presets/sp.json"   # mace opt2d12 --config-file, when none is given
# opt2d3_config = "presets/band.json"  # mace opt2d3 --config-file, when none is given
# cif2d12_options = "presets/cif.json" # mace convert --batch --options_file, when none is given

[database]
# columns = ["material_id", "formula", "space_group", "status", "updated_at"]

[submit]
# what site.toml's [folders] table allows (mace site shows it):
# ntasks = 32                       # CRYSTAL jobs
# walltime = "2-00:00:00"
# partition = "general"             # one of site.toml's [folders] partitions

[submit.properties]
# ntasks = 16
# walltime = "4:00:00"
"""


# ---------------------------------------------------------------- the checks
def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def check_value(key: str, value: Any) -> Optional[str]:
    """Why ``value`` cannot be ``key``'s (a reason), or None. Paths are checked
    for their form here; ``check_path`` checks where they lead."""
    spec = KEYS.get(key)
    if spec is None:
        return "is not a folder setting"
    t = spec["type"]
    if t == "bool":
        return None if isinstance(value, bool) else "must be true or false"
    if t == "color":
        return None if isinstance(value, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", value) else 'must be a color like "#1a2b3c"' 
    if t == "choice":
        ok = [c for c in spec["choices"] if c == value and type(c) is type(value)]
        return None if ok else "must be one of " + ", ".join(json.dumps(c) for c in spec["choices"])
    if t in ("number", "int"):
        lo, hi = spec["range"]
        if t == "int" and (isinstance(value, bool) or not isinstance(value, int)):
            return f"must be a whole number from {lo} to {hi}"
        if not _num(value) or not lo <= value <= hi:
            return f"must be a number from {lo:g} to {hi:g}"
        return None
    if t in ("pair", "triple"):
        n = 2 if t == "pair" else 3
        lo, hi = spec["range"]
        if not isinstance(value, list) or len(value) != n or not all(_num(x) and lo <= x <= hi for x in value):
            return f"must be a list of {n} numbers from {lo:g} to {hi:g}"
        if t == "pair" and value[1] - value[0] < 0.1:
            return "must be [low, high] with high above low"
        return None
    if t == "name":
        return None if isinstance(value, str) and (value == "" or NAME_RE.match(value)) and ".." not in value \
            else "must be a name (letters, digits, spaces and ( ) + . _ / -)"
    if t == "filename":
        return None if isinstance(value, str) and FILE_NAME_RE.match(value) and ".." not in value \
            else "must be a file name without a folder (letters, digits, . _ -)"
    if t == "path":
        return path_form(value)
    if t == "columns":
        if (not isinstance(value, list) or not value or len(value) > len(DB_COLUMNS)
                or len(set(map(str, value))) != len(value) or not all(c in DB_COLUMNS for c in value)):
            return "must be a list of columns from " + ", ".join(DB_COLUMNS)
        return None
    if t == "partition":
        return None if isinstance(value, str) and PARTITION_RE.match(value) else "must be a partition name"
    if t == "walltime":
        return None if isinstance(value, str) and walltime_seconds(value) \
            else ('must be a walltime such as "2:00:00", "1-00" or "2-00:00:00" (D-HH:MM:SS, D-HH or HH:MM:SS, '
                  'at most 366 days; not "12:00", which SLURM reads as 12 minutes)')
    return "has an unknown type"  # pragma: no cover


def path_form(value: Any) -> Optional[str]:
    """A path inside the folder: relative, no '..', no '~' or '$'."""
    if not isinstance(value, str) or not value or len(value) > 500:
        return "must be a path inside this folder"
    if value.startswith(("/", "~", "\\")) or "$" in value or "\\" in value or "\x00" in value:
        return "must be a relative path inside this folder"
    if ".." in value.split("/"):
        return "must not leave this folder ('..')"
    return None


def check_path(folder: Path, value: str) -> Optional[str]:
    """Why ``value`` (relative to ``folder``) leads out of it, or None."""
    why = path_form(value)
    if why:
        return why
    try:
        base = folder.resolve()
        real = (folder / value).resolve()
    except (OSError, RuntimeError):
        return "cannot be resolved"
    if real != base and base not in real.parents:
        return "leads out of this folder (a symbolic link)"
    return None


def walltime_seconds(value: str) -> Optional[int]:
    """Seconds of a walltime in the forms WALLTIME_RE takes (hours of a D-HH
    form at most 23, at most 366 days), else None."""
    v = (value or "").strip() if isinstance(value, str) else ""
    if not WALLTIME_RE.fullmatch(v):
        return None
    days, rest = (v.split("-", 1) if "-" in v else ("0", v))
    nums = [int(p) for p in rest.split(":")] + [0, 0]
    h, m, s = nums[:3]
    if "-" in v and h > 23:
        return None
    secs = int(days) * 86400 + h * 3600 + m * 60 + s
    return secs if 0 < secs <= MAX_WALLTIME_S else None


# ---------------------------------------------------------------- finding the file
def _home() -> Optional[Path]:
    try:
        return Path(os.path.expanduser("~")).resolve()
    except (OSError, RuntimeError):
        return None


# A process that is not "in" a folder (mace gui's server: its folder is the
# one it serves, not where it was started) turns the working-folder default off.
CWD_DISCOVERY = True
# A command that works on another folder's files (`mace submit DIR/deck` run
# from elsewhere) looks from that folder instead of the working folder, so it
# reads the file the job's completion callback (run in DIR) reads.
DISCOVERY_START: Optional[Path] = None


def use_folder(folder: Optional[os.PathLike]) -> None:
    """Look for the folder file from ``folder`` instead of the working folder
    (None: the working folder again). MACE_FOLDER_SETTINGS still wins."""
    global DISCOVERY_START
    DISCOVERY_START = Path(folder) if folder is not None else None


def find(start: Optional[os.PathLike] = None, env: Optional[Dict[str, str]] = None) -> Optional[Path]:
    """The folder file for ``start`` (default: the working folder): the
    MACE_FOLDER_SETTINGS file, or the nearest mace-folder.toml in ``start``
    or a parent, below the home folder and the file-system root."""
    e = os.environ if env is None else env
    pinned = (e.get(ENV) or "").strip()
    if pinned.lower() == "none":
        return None
    if pinned:
        p = Path(pinned).expanduser()
        return p if p.name == FILENAME or p.suffix == ".toml" else None
    if start is None and DISCOVERY_START is not None:
        start = DISCOVERY_START
    if start is None and not CWD_DISCOVERY:
        return None
    try:
        here = Path(start if start is not None else os.getcwd()).resolve()
    except (OSError, RuntimeError):
        return None
    home = _home()
    for d in [here, *here.parents]:
        if d == Path(d.anchor) or (home is not None and d == home):
            break
        f = d / FILENAME
        try:
            if os.path.lexists(f):
                return f
        except OSError:
            return None
    return None


@dataclass
class FolderSettings:
    """A folder file's checked values (``values``: key -> value, only good
    ones), what was wrong with it (``warnings``) and where it is."""
    path: Optional[Path] = None
    values: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    refused: bool = False             # the whole file was ignored (see warnings)

    @property
    def folder(self) -> Optional[Path]:
        return self.path.parent if self.path else None

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def gui(self, root: Optional[Path] = None) -> Dict[str, Any]:
        """The GUI settings this file sets: {settings.js key: value}. A path
        becomes relative to ``root`` (the served folder); one outside it is left out."""
        out = {}
        for key, v in self.values.items():
            g = KEYS[key]["gui"]
            if not g:
                continue
            if KEYS[key]["type"] == "path":
                if root is None or self.folder is None:
                    continue
                try:
                    v = (self.folder / v).resolve().relative_to(Path(root).resolve()).as_posix()
                except (ValueError, OSError):
                    continue
            out[g] = v
        return out

    def cli_only(self) -> Dict[str, Any]:
        return {k: v for k, v in self.values.items() if not KEYS[k]["gui"]}


def _refusal(path: Path) -> Optional[str]:
    try:
        lst = os.lstat(path)
    except OSError as exc:
        return f"cannot be read ({exc.strerror})"
    if stat.S_ISLNK(lst.st_mode):
        return "is a symbolic link; MACE reads a regular file only"
    if not stat.S_ISREG(lst.st_mode):
        return "is not a regular file"
    if lst.st_mode & stat.S_IWOTH:
        return "can be changed by everyone (chmod o-w it)"
    try:
        dst = os.stat(path.parent)
        if dst.st_mode & stat.S_IWOTH:
            return "is in a folder everyone can write to (a scratch root?); keep it in your project folder"
    except OSError:
        pass
    if lst.st_size > MAX_BYTES:
        return f"is larger than {MAX_BYTES // 1000} kB"
    return None


def _flatten(data: Dict[str, Any], prefix: str = "") -> List[Tuple[str, Any]]:
    out = []
    for k, v in data.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict) and (key in TABLES or any(t.startswith(key + ".") for t in TABLES)) and key not in KEYS:
            out += _flatten(v, key)
        else:
            out.append((key, v))
    return out


def parse(text: str, path: Path) -> FolderSettings:
    """The checked values of a folder file's ``text``."""
    fs = FolderSettings(path=path)
    try:
        import tomllib
    except ImportError:          # Python < 3.11
        fs.refused = True
        fs.warnings.append(f"{path}: needs Python 3.11 or newer (tomllib) to read; ignored")
        return fs
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        fs.refused = True
        fs.warnings.append(f"{path}: not valid TOML ({exc}); the whole file is ignored")
        return fs
    unknown = []
    for key, v in _flatten(data):
        if key in RENAMED:
            new, values = RENAMED[key]
            fs.warnings.append(f"{path}: {key} is now {new}; read as {new} = {json.dumps(values.get(v, v))}")
            key, v = new, values.get(v, v)
        if key not in KEYS:
            unknown.append(key)
            continue
        why = check_value(key, v)
        if why is None and KEYS[key]["type"] == "path":
            why = check_path(path.parent, v)
        if why:
            fs.warnings.append(f"{path}: {key} {why}; ignored")
            continue
        fs.values[key] = v
    if unknown:
        fs.warnings.append(f"{path}: unknown keys ignored: {', '.join(sorted(unknown))}")
    return fs


_cache: Dict[str, Any] = {}
_cache_lock = threading.Lock()


def read(path: Path) -> FolderSettings:
    """The checked folder file at ``path`` (cached by size and mtime)."""
    why = _refusal(path)
    if why:
        return FolderSettings(path=path, refused=True, warnings=[f"{path} {why}; it is ignored"])
    st = os.stat(path)
    key = (str(path), st.st_size, st.st_mtime_ns)
    with _cache_lock:
        if _cache.get("key") == key:
            return _cache["fs"]
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return FolderSettings(path=path, refused=True, warnings=[f"{path} cannot be read ({exc}); it is ignored"])
    fs = parse(text, path)
    with _cache_lock:
        _cache.update(key=key, fs=fs)
    return fs


def load(start: Optional[os.PathLike] = None, env: Optional[Dict[str, str]] = None) -> FolderSettings:
    """The folder settings that apply in ``start`` (default: the working
    folder); an empty FolderSettings when there is no file."""
    p = find(start, env)
    if p is None:
        return FolderSettings()
    if not os.path.lexists(p):
        return FolderSettings(warnings=[f"{ENV} names {p}, which does not exist"])
    return read(p)


_said: set = set()


def warn_once(fs: FolderSettings, out=None) -> None:
    """Print each warning of ``fs`` once per process (stderr)."""
    for w in fs.warnings:
        if w not in _said:
            _said.add(w)
            print(f"MACE: {w}", file=out or sys.stderr, flush=True)


def note(text: str, out=None) -> None:
    if text not in _said:
        _said.add(text)
        print(f"MACE: {text}", file=out or sys.stderr, flush=True)


# ---------------------------------------------------------------- the CLI's uses
def plot_defaults(fs: FolderSettings) -> Dict[str, Any]:
    """argparse defaults for ``mace plotting`` (``parser.set_defaults``)."""
    out = {}
    win = fs.get("plots.energy_window")
    if win:
        out["e_lower"], out["e_upper"] = float(win[0]), float(win[1])
    for key, dest in (("plots.style", "style"), ("plots.dos_colors", "dos_colors"), ("plots.spin_down_style", "spin_style"),
                      ("plots.spin_colors", "spin_colors"), ("plots.spin_up_color", "spin_up_color"),
                      ("plots.spin_down_color", "spin_down_color"), ("plots.band_color", "band_color"),
                      ("plots.dos_spin", "spin_dos"),
                      ("plots.dos_total", "dos_total"), ("plots.dos_smooth", "dos_smooth"), ("plots.fermi_shift", "fermi_shift"),
                      ("plots.energy_reference", "energy_reference"), ("plots.backend", "backend"),
                      ("plots.potential_colors", "potential_colors"),
                      ("units.energy", "units")):
        if fs.get(key) is not None:
            out[dest] = fs.get(key)
    return out


def _given(argv: Iterable[str], *flags: str) -> bool:
    for a in argv:
        for f in flags:
            if a == f or (f.startswith("--") and a.startswith(f + "=")):
                return True
    return False


def preset_args(command: str, argv: List[str], fs: FolderSettings) -> Tuple[List[str], Optional[str]]:
    """(argv with the folder's preset added, a note) for ``mace opt2d12``,
    ``mace opt2d3`` and ``mace convert``; argv unchanged when the command
    names its own file, saves one, or only asks for help."""
    if not fs.values or _given(argv, "-h", "--help"):
        return argv, None
    folder = fs.folder
    if command == "opt2d12":
        key, flag = "setup.opt2d12_config", "--config-file"
        if _given(argv, "--config-file", "--save-options"):
            return argv, None
    elif command == "opt2d3":
        key, flag = "setup.opt2d3_config", "--config-file"
        if _given(argv, "--config-file", "-c", "--save-config", "--list-configs"):
            return argv, None
    elif command == "convert":
        key, flag = "setup.cif2d12_options", "--options_file"
        if not _given(argv, "--batch") or _given(argv, "--options_file", "--save_options"):
            return argv, None
    else:
        return argv, None
    rel = fs.get(key)
    if not rel:
        return argv, None
    target = folder / rel
    if not target.is_file():
        return argv, f"{fs.path}: {key} = {rel!r} does not exist; not used"
    return argv + [flag, str(target)], f"using {flag} {target} ({key} in {fs.path}; pass {flag} to use another)"


# Calculation Setup's defaults for a deck without a parent (cif2d12 from a
# CIF; opt2d12 only where the parent's functional is unusable). mace gui's
# Setup sets these variables to its effective values (session > folder >
# user); set, even empty, they replace the folder file for that run.
SETUP_ENV = {"setup.functional": "MACE_SETUP_FUNCTIONAL", "setup.basis_set": "MACE_SETUP_BASIS_SET"}


def setup_defaults(env: Optional[Dict[str, str]] = None, start: Optional[os.PathLike] = None
                   ) -> Dict[str, Tuple[str, str]]:
    """{"functional"|"basis_set": (value, where it comes from)} for a deck
    without a parent: the MACE_SETUP_* variables when either is set, else
    the folder file's [setup] functional / basis_set; {} when none."""
    e = os.environ if env is None else env
    out: Dict[str, Tuple[str, str]] = {}
    if any(var in e for var in SETUP_ENV.values()):
        for key, var in SETUP_ENV.items():
            v = (e.get(var) or "").strip()
            if v and check_value(key, v) is None:
                out[key.split(".", 1)[1]] = (v, "mace gui's settings")
        return out
    fs = load(start, env)
    warn_once(fs)
    for key in SETUP_ENV:
        v = fs.get(key)
        if v:
            out[key.split(".", 1)[1]] = (v, str(fs.path))
    return out


# ---------------------------------------------------------------- the site layer (mace submit)
SITE_ALLOW_DEFAULT = ("ntasks", "walltime", "partition")


def submit_values(fs: FolderSettings) -> Dict[str, Any]:
    return {k[len("submit."):]: v for k, v in fs.values.items() if k.startswith("submit.")}


# ---------------------------------------------------------------- writing (the GUI's Save to this folder)
_write_lock = threading.Lock()


def toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        r = repr(v)
        return r if any(c in r for c in ".eEn") else r + ".0"
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, list):
        return "[" + ", ".join(toml_value(x) for x in v) + "]"
    raise ValueError(f"cannot write {type(v).__name__} values")


_HEADER = re.compile(r"^\s*\[\s*([A-Za-z0-9_.\- ]+?)\s*\]\s*(#.*)?$")


def _key_of(line: str) -> Optional[str]:
    m = re.match(r"^\s*([A-Za-z0-9_-]+)\s*=", line)
    return m.group(1) if m else None


def _value_end(lines: List[str], i: int) -> Tuple[int, str]:
    """(index of the last line of the value that starts on line i, the
    trailing comment of that line or ''): brackets and strings are followed."""
    depth, j, quote = 0, i, None
    text = lines[i].split("=", 1)[1]
    while True:
        k = 0
        comment = ""
        while k < len(text):
            c = text[k]
            if quote:
                if c == "\\" and quote == '"':
                    k += 2
                    continue
                if c == quote:
                    quote = None
            elif c in "\"'":
                quote = c
            elif c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
            elif c == "#":
                comment = text[k:].rstrip("\n")
                break
            k += 1
        if depth <= 0 or j + 1 >= len(lines):
            return j, comment
        j += 1
        text = lines[j]


def edit_text(text: str, sets: Dict[str, Any], unsets: Iterable[str]) -> str:
    """``text`` with these keys set or removed, every other line (comments
    included) kept. A key goes under its table's header (added at the end
    when missing), after the commented example of the same key if there is one."""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    todo = [(k, True, v) for k, v in sets.items()] + [(k, False, None) for k in unsets if k not in sets]
    for key, is_set, value in todo:
        table, name = key.rsplit(".", 1)
        start = end = None
        for i, l in enumerate(lines):
            m = _HEADER.match(l)
            if not m:
                continue
            if start is not None:
                end = i
                break
            if m.group(1).replace(" ", "") == table:
                start = i
        if start is not None and end is None:
            end = len(lines)
        hit = None
        if start is not None:
            for i in range(start + 1, end):
                if _key_of(lines[i]) == name:
                    hit = i
                    break
        if not is_set:
            if hit is not None:
                last, _ = _value_end(lines, hit)
                del lines[hit:last + 1]
            continue
        new = f"{name} = {toml_value(value)}"
        if hit is not None:
            last, comment = _value_end(lines, hit)
            lines[hit:last + 1] = [new + (f"  {comment.strip()}" if comment else "")]
            continue
        if start is None:
            while lines and lines[-1].strip() == "":
                lines.pop()
            lines += ["", f"[{table}]", new]
            continue
        at = None
        for i in range(start + 1, end):
            if re.match(rf"^\s*#\s*{re.escape(name)}\s*=", lines[i]):
                at = i + 1
                break
        if at is None:
            at = end
            while at - 1 > start and lines[at - 1].strip() == "":
                at -= 1
        lines.insert(at, new)
    return "\n".join(lines) + "\n"


def write(path: Path, sets: Dict[str, Any], unsets: Iterable[str] = ()) -> FolderSettings:
    """Set and remove keys in the folder file at ``path`` (created from
    TEMPLATE when missing), keeping its comments. Every value is checked
    first (ValueError names the bad one); the result is read back, and a
    key the edit could not place (written another way by hand, e.g. as an
    inline table) is a ValueError, with the file left as it was."""
    unsets = list(unsets)
    for k, v in sets.items():
        why = check_value(k, v)
        if why is None and KEYS[k]["type"] == "path":
            why = check_path(path.parent, v)
        if why:
            raise ValueError(f"{k} {why}")
    for k in unsets:
        if k not in KEYS:
            raise ValueError(f"{k} is not a folder setting")
    with _write_lock:
        exists = os.path.lexists(path)
        if exists:
            why = _refusal(path)
            if why:
                raise ValueError(f"{path} {why}")
            old = path.read_text(encoding="utf-8")
            mode = stat.S_IMODE(os.stat(path).st_mode)
        else:
            old, mode = TEMPLATE, 0o644
        new = edit_text(old, sets, unsets)
        if len(new.encode("utf-8")) > MAX_BYTES:
            raise ValueError(f"{path.name} would be larger than {MAX_BYTES // 1000} kB")
        back = parse(new, path)
        if back.refused:
            raise ValueError(f"{path.name} could not be edited here; edit it by hand ({back.warnings[0]})")
        for k, v in sets.items():
            if back.values.get(k) != v:
                raise ValueError(f"{k} is written another way in {path.name}; edit it by hand")
        for k in unsets:
            if k in back.values:
                raise ValueError(f"{k} is written another way in {path.name}; edit it by hand")
        # a fresh, unpredictable temp file (O_EXCL): never one planted at a guessable name
        import tempfile
        fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
        try:
            os.fchmod(fd, mode & ~stat.S_IWOTH)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(new)
            os.replace(tmp, path)
        finally:
            if os.path.lexists(tmp):
                os.unlink(tmp)
    return read(path)


# ---------------------------------------------------------------- where a save goes (GUI and `mace site set --folder`)
def save_target(folder: os.PathLike, env: Optional[Dict[str, str]] = None) -> Tuple[Path, Optional[str]]:
    """(the file a save for ``folder`` writes, why it may not be written or
    None). The GUI's Save to this folder and ``mace site set --folder`` both
    ask here: the file found for the folder is edited as it is, but never one
    above the folder (that file is read there; it is edited from its own
    folder), never one MACE refuses to read (a symbolic link, a file or a
    folder everyone can change, bad TOML), never one in a place MACE does
    not look (the home folder or the file-system root), and never a
    read-only one."""
    folder = Path(folder)
    fs = load(folder, env=env)
    target = fs.path if fs.path is not None else folder / FILENAME
    try:
        here = folder.resolve()
        inside = target.parent.resolve() == here
    except (OSError, RuntimeError):
        return target, f"{folder} cannot be resolved"
    if not inside:
        return target, (f"The folder settings come from {target}, above this folder: edit that file, "
                        "or open its folder")
    if fs.refused:
        return target, f"{target.name} is not used ({'; '.join(fs.warnings)}); fix it by hand first"
    if fs.path is None:
        if here == Path(here.anchor) or here == _home():
            return target, (f"MACE does not read a {FILENAME} in {'your home folder' if here == _home() else '/'}; "
                            "keep it in a project folder")
        try:
            if os.stat(here).st_mode & stat.S_IWOTH:
                return target, ("This folder is one everyone can write to (a scratch root?): MACE ignores a "
                                f"{FILENAME} there; keep it in your project folder")
        except OSError as exc:
            return target, f"{folder} cannot be read ({exc.strerror})"
        if not os.access(here, os.W_OK):
            return target, "This folder is read-only for you"
    elif not os.access(target, os.W_OK):
        return target, f"{target} is read-only for you"
    return target, None


def value_from_text(key: str, text: str) -> Any:
    """A value for ``key`` from the shell's text (``mace site set --folder``):
    true/false, a number, a list as "a,b" or "[a, b]", else the text. The
    result still goes through check_value."""
    spec = KEYS.get(key)
    if spec is None:
        raise ValueError(f"{key} is not a folder setting")
    t, s = spec["type"], text.strip()
    if t == "bool":
        low = s.lower()
        if low in ("true", "yes", "on", "1"):
            return True
        if low in ("false", "no", "off", "0"):
            return False
        return s
    if t == "choice":
        for c in spec["choices"]:
            if str(c) == s:
                return c
        return s
    if t in ("number", "int"):
        try:
            return int(s) if re.fullmatch(r"[-+]?\d+", s) else float(s)
        except ValueError:
            return s
    if t in ("pair", "triple", "columns"):
        parts = [p for p in re.split(r"[,\s]+", s.strip("[]() ")) if p]
        if t == "columns":
            return [p.strip("\"'") for p in parts]
        try:
            return [float(p) for p in parts]
        except ValueError:
            return s
    return text
