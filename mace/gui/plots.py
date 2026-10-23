"""Results plots for the GUI, made by ``mace plotting`` itself.

The band, DOS and spectra plotters chdir into the data file's directory and
write their PNG/SVG beside it. The GUI never writes into the browsed
directory, so it copies the data file (and the small files of the same run
the DOS plotter reads: .d3, .out) into a folder of its own cache directory
and runs ``mace plotting --band|--dos|--ir|--raman`` there, as a cancelable
``plot`` task (clirun: nice 10, one CPU slot, SIGTERM then SIGKILL). Those
folders count against the one memory budget (cache.DiskCache): the cache
directory is under /tmp, often RAM.
"""
from __future__ import annotations

import hashlib
import os
import json
import re
import math
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from . import clirun

# kind -> (mace plotting flag, data-file suffixes that name the run)
KINDS = {
    "band": ("--band", (".BAND.DAT", ".band.band.dat", "_band.dat")),
    "dos": ("--dos", (".DOSS.DAT", "_doss.dat")),
    # the band structure and its run's DOS in one figure (what the GUI shows when it has both)
    "bandos": ("--band-dos", (".BAND.DAT", ".band.band.dat", "_band.dat", ".DOSS.DAT", "_doss.dat")),
    # a phonon dispersion and its phonon DOS in one figure (a phonon run's fort.25 files; Add Phonon DOS…)
    "phbandos": ("--phonon-band-dos", (".BAND.DAT", ".band.band.dat", "_band.dat", ".DOSS.DAT", "_doss.dat")),
    "ir": ("--ir", ("_IRSPEC.DAT", ".IRSPEC.DAT", "IRSPEC.DAT")),
    "raman": ("--raman", ("_RAMSPEC.DAT", ".RAMSPEC.DAT", "RAMSPEC.DAT")),
    # a BOLTZTRA file of any kind; the run's other files beside it are copied along (transport_files)
    "transport": ("--transport", ()),
}
FORMATS = ("svg", "png", "pdf")
_SIBLING_SUFFIXES = (".d3", ".out", ".d12")
_SIBLING_MAX = 20 * 1024 * 1024


def kind_of(path: Path) -> Optional[str]:
    """'band' / 'dos' / 'ir' / 'raman' when mace plotting's registry classifies the file so."""
    try:
        from mace.plotting import handlers  # noqa: F401  (registers the plotters)
        from mace.plotting.detect import classify_file
        kind = classify_file(str(path))
    except Exception:
        kind = None
    if kind is None:
        return None
    return {"band": "band", "dos": "dos", "ir": "ir", "raman": "raman", "transport": "transport"}.get(kind.value)


def _run_stem(name: str, kind: str) -> str:
    from mace.plotting import f25
    if kind == "transport":
        from mace.utils import transport_plot
        return transport_plot.split_name(Path(name))[0]
    for suffix in KINDS[kind][1]:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    if f25.is_f25_name(name):
        return f25._stem(Path(name))                  # x_band.fort.25 -> x_band, not x_band.fort
    return Path(name).stem


BAND_PATH_FILE = "band_path.json"
PARTNER_DIR = "partner"           # a band + DOS pair's other half (from another folder) in the cache folder


# the plots with an energy window (mace plotting --e-lower/--e-upper)
ENERGY_KINDS = ("band", "dos", "bandos", "phbandos")
# the band + DOS figures (a pair's other half from another folder: --partner)
PAIR_KINDS = ("bandos", "phbandos")


def energy_window(get, folder_window=None) -> Optional[Tuple[float, float]]:
    """The window a band/DOS plot is drawn with: ``emin``/``emax`` as given (the
    page sends the one on screen), else the served folder's ``[plots]
    energy_window``, else None (the CLI's own default)."""
    lo, hi = get("emin"), get("emax")
    if lo in (None, "") and hi in (None, ""):
        return (float(folder_window[0]), float(folder_window[1])) if folder_window else None
    try:
        lo, hi = float(lo), float(hi)
    except (TypeError, ValueError):
        raise ValueError("the energy window is two numbers (eV)")
    if not (math.isfinite(lo) and math.isfinite(hi)) or not lo < hi or abs(lo) > 200 or abs(hi) > 200:
        raise ValueError("the energy window runs from a lower to a higher energy, within ±200 eV")
    return lo, hi


# mace plotting --energy-reference (auto: the vacuum scale with the run's POTC files, as the GUI's plot)
REFERENCES = ("auto", "fermi", "vacuum")
SHIFT_MAX = 50.0          # Settings' Fermi shift range (eV)


def energy_reference(get) -> Tuple[float, str]:
    """(Fermi shift in eV, energy reference) of a band/DOS plot: ``shift`` (the
    one on screen: dragged or typed) and ``reference`` as given, else 0 and auto
    (mace plotting --fermi-shift/--energy-reference)."""
    s, ref = get("shift"), get("reference")
    try:
        s = 0.0 if s in (None, "") else float(s)
    except (TypeError, ValueError):
        raise ValueError("the Fermi shift is a number (eV)")
    if not math.isfinite(s) or abs(s) > SHIFT_MAX:
        raise ValueError(f"the Fermi shift is within ±{SHIFT_MAX:g} eV")
    ref = ref or "auto"
    if ref not in REFERENCES:
        raise ValueError(f"the energy reference is one of {', '.join(REFERENCES)}")
    return s, ref


_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


UNITS = ("eV", "meV", "Ha", "kJ/mol", "kcal/mol", "cm-1", "THz")
# the names the 1.1.6 preview's page sent (before presets.STYLE_OPTIONS)
_OLD_NAMES = {"spin_style": "spin_down_style", "spin_up": "spin_up_color", "spin_down": "spin_down_color", "spin_dos": "dos_spin"}


def look_args(kind: str, get, explicit: bool = False) -> list:
    """The plot's look on screen as mace plotting flags (1.1.6): the presets.py STYLE_OPTIONS names
    (style, spin_colors, spin_up_color, spin_down_color, band_color, spin_down_style, dos_colors, dos_spin,
    dos_total), as presets.style_flags writes them (only values that are not the defaults; a
    spectrum takes the style only), and for a band/DOS plot ``unit`` (--units). Anything else is refused.
    ``explicit`` (the served folder has a mace-folder.toml, which the child reads): the defaults too, so
    the file's [plots] and [units] values do not replace what the screen shows (as Copy CLI Command)."""
    from mace.plotting import presets
    look = {}
    for opt in presets.STYLE_OPTIONS:
        v = get(opt["name"])
        if v in (None, ""):
            old = next((o for o, n in _OLD_NAMES.items() if n == opt["name"]), None)
            v = get(old) if old else None
        if v in (None, ""):
            continue
        name = opt["name"]
        if name == "dos_total":
            look[name] = str(v).lower() not in ("0", "false")
            continue
        if name == "dos_smooth":
            try:
                sm = float(v)
            except (TypeError, ValueError):
                raise ValueError("dos_smooth is a sigma in eV from 0 to 1")
            if not 0 <= sm <= 1:
                raise ValueError("dos_smooth is a sigma in eV from 0 to 1")
            look[name] = sm
            continue
        if name in ("spin_up_color", "spin_down_color", "band_color"):
            if not _HEX.match(str(v)):
                raise ValueError(f"{name} is a color like #1a2b3c")
            look[name] = str(v)
            continue
        if v not in opt["choices"]:
            raise ValueError(f"{name} is one of " + ", ".join(map(str, opt["choices"])))
        look[name] = v
    if kind not in ENERGY_KINDS:
        style = look.get("style", "default")
        return [f"--style={style}"] if explicit else presets.style_flags({"style": style})
    out = presets.style_flags(look, explicit=explicit)
    # a fort.25: its domain (electronic or phonon) and which of its runs (--f25-domain, --f25-run)
    dom, run = get("domain"), get("run")
    if dom not in (None, "", "auto"):
        if dom not in ("electronic", "phonon"):
            raise ValueError("domain is one of auto, electronic, phonon")
        out.append(f"--f25-domain={dom}")
    if run not in (None, "", 0, "0", -1, "-1"):
        try:
            if int(run) < 1:
                raise ValueError
        except (TypeError, ValueError):
            raise ValueError("run is a whole number from 1")
        out.append(f"--f25-run={int(run)}")
    # the Reference and Labels menus' switches (on by default): --no-fermi-line, --no-k-labels, --no-spin-labels
    for name in ("fermi_line", "k_labels", "spin_labels"):
        if str(get(name)).lower() in ("0", "false"):
            out.append("--no-" + name.replace("_", "-"))
    # Log Y on a DOS plot, Log DOS on a band + DOS plot: the DOS axis only (band energies stay linear)
    if kind in ("dos",) + PAIR_KINDS and str(get("log_dos")).lower() in ("1", "true"):
        out.append("--log-dos")
    unit = get("unit")
    if explicit and unit in (None, ""):
        unit = "eV"
    if unit not in (None, "") and (explicit or unit != "eV"):
        if unit not in UNITS:
            raise ValueError("unit is one of " + ", ".join(UNITS))
        out.append(f"--units={unit}")
    return out


def transport_args(get, kappa_path=None) -> Tuple[list, list]:
    """(mace plotting flags, files to copy along) of a transport plot as the screen shows it:
    ``quantity`` (--quantity), ``component``, ``temperature`` (a value or all), ``x`` (mu or carriers),
    ``kappa_l`` (--kappa-lattice: a value, or a CSV ``kappa_path(text)`` finds inside the folder,
    copied along and passed by its name) and ``tau_fs`` (--tau-fs). Each checked here."""
    from mace.utils import transport_plot as T
    from mace.utils import transport_zt as Z
    out, files = [], []
    q = get("quantity")
    if q not in (None, ""):
        out.append(f"--quantity={T.CLI_OF[T.quantity_key(q)]}")
    c = get("component")
    if c not in (None, ""):
        if c not in T.COMPONENTS:
            raise ValueError("component is one of " + ", ".join(T.COMPONENTS))
        out.append(f"--component={c}")
    t = get("temperature")
    if t not in (None, ""):
        if str(t).lower() != "all":
            try:
                if not math.isfinite(float(t)) or float(t) <= 0:
                    raise ValueError
            except (TypeError, ValueError):
                raise ValueError("temperature is a number of K or all")
            t = f"{float(t):g}"
        out.append(f"--temperature={str(t).lower()}")
    x = get("x")
    if x not in (None, ""):
        if x not in T.X_AXES:
            raise ValueError("x is one of " + ", ".join(T.X_AXES))
        out.append(f"--x-axis={x}")
    kl = str(get("kappa_l") or "").strip()
    if kl:
        try:
            Z.parse_kappa_lattice(float(kl))
            out.append(f"--kappa-lattice={float(kl):g}")
        except ValueError:
            f = kappa_path(kl) if kappa_path else None
            if f is None or not Path(f).is_file():
                raise ValueError(f"kappa_l {kl!r} is neither a number nor a CSV file in the folder")
            Z.parse_kappa_lattice(str(f))
            out.append(f"--kappa-lattice={Path(f).name}")
            files.append(Path(f))
    tau = str(get("tau_fs") or "").strip()
    if tau:
        out.append(f"--tau-fs={Z.check_tau_fs(tau):g}")
    return out, files


def transport_files(path: Path, inside=None) -> list:
    """The run's other transport files, .out and .d3 beside a BOLTZTRA file (mace plotting --transport
    finds them beside the one it is given, as the GUI's plot)."""
    from mace.utils import transport_plot as T
    ok = inside or (lambda f: True)
    stem = T.split_name(path)[0]
    files = [f for f in T.run_files(path, ok).values() if f != path]
    for suffix in (".out", ".d3"):
        f = T.beside(path, stem, suffix, ok)
        if f is not None:
            files.append(f)
    return [f for f in files if f.stat().st_size < 10 * _SIBLING_MAX]


def _window_args(kind: str, ewin, shift: float = 0.0, reference: str = "auto", explicit: bool = False) -> list:
    """``explicit``: the shift and the reference even at 0 and auto (a mace-folder.toml's
    fermi_shift or energy_reference would replace them)."""
    if kind not in ENERGY_KINDS:
        return []
    out = [f"--e-lower={float(ewin[0]):g}", f"--e-upper={float(ewin[1]):g}"] if ewin else []
    if shift or explicit:
        out.append(f"--fermi-shift={float(shift or 0):g}")
    if explicit or (reference and reference != "auto"):
        reference = reference or "auto"
        out.append(f"--energy-reference={reference}")
    return out


def render_key(path: Path, kind: str, fmt: str = "", band_path: Optional[list] = None, ewin=None,
               shift: float = 0.0, reference: str = "auto", look: Optional[list] = None,
               explicit: bool = False, extra_files=None, partner: Optional[Path] = None) -> str:
    st = path.stat()
    edit = json.dumps(band_path, sort_keys=True) if band_path else ""
    win = " ".join(_window_args(kind, ewin, shift, reference, explicit) + list(look or [])
                   + [f"{Path(f).name}:{Path(f).stat().st_mtime_ns}:{Path(f).stat().st_size}" for f in extra_files or []])
    other = f"|{partner.resolve()}|{partner.stat().st_mtime_ns}" if partner is not None else ""
    return hashlib.sha1(f"{path.resolve()}|{st.st_mtime_ns}|{st.st_size}|{kind}|{fmt}|{edit}|{win}{other}".encode()).hexdigest()[:16]


def command(path: Path, kind: str, fmt: str = "", band_path: Optional[list] = None, ewin=None,
            shift: float = 0.0, reference: str = "auto", look: Optional[list] = None,
            explicit: bool = False, partner: Optional[Path] = None) -> str:
    """The command a user runs in the data file's folder for the same figure;
    a band/DOS plot always names its window (the server's cache folder has no
    mace-folder.toml, so the window is passed, never looked up), and its
    Fermi shift and energy reference when they are not the defaults; a band +
    DOS pair from two folders its other half (--partner, relative to the folder)."""
    import shlex
    other = [f"--partner {shlex.quote(os.path.relpath(partner, path.parent))}"] if partner is not None else []
    return " ".join([f"mace plotting {KINDS[kind][0]} {shlex.quote(path.name)}"] + other + ([f"--format {fmt}"] if fmt else [])
                    + ([f"--band-path {BAND_PATH_FILE}"] if band_path else [])
                    + _window_args(kind, ewin, shift, reference, explicit) + list(look or []))


def _potc_files(path: Path, kind: str) -> list:
    """The run's POTC files beside a band/DOS data file (both or none, as the
    GUI's readers take them): with them mace plotting draws the vacuum scale."""
    if kind not in ENERGY_KINDS or kind == "phbandos":       # (a phonon plot has no vacuum scale)
        return []
    stem = _run_stem(path.name, kind)
    tag = "_band" if (kind == "band" or (kind == "bandos" and path.name.endswith(KINDS["band"][1]))) else "_doss"
    material = stem[: -len(tag)] if stem.endswith(tag) else stem
    files = [path.parent / f"{material}_POTC.POTC.DAT", path.parent / f"{material}_POTC.out"]
    return files if all(f.is_file() and f.stat().st_size < _SIBLING_MAX for f in files) else []


def render(path: Path, kind: str, disk, ctx: Any = None, fmt: str = "", band_path: Optional[list] = None,
           ewin: Optional[Tuple[float, float]] = None, env: Optional[Dict[str, str]] = None,
           shift: float = 0.0, reference: str = "auto", look: Optional[list] = None,
           explicit: bool = False, inside=None, extra_files=None, partner: Optional[Path] = None) -> Dict[str, Any]:
    """Plot ``path`` with mace plotting into the disk cache; the image files it made.
    ``look``: the plot style flags (look_args), so the figure is drawn as the screen shows it.
    ``band_path``: the band plot's edited k-path (written as band_path.json and passed
    as --band-path, what a CLI user passes for the same figure). ``ewin``: the
    band/DOS energy window (--e-lower/--e-upper); ``shift``/``reference``: its
    Fermi shift and energy zero (--fermi-shift/--energy-reference); ``env``: the
    child's environment (clirun.cli_env() by default); ``inside``: whether a file
    of the same run is inside the served folder (only those are copied along:
    a partner or sibling that is a symlink out of it is never read); ``extra_files``: files the
    flags name (a transport plot's κ_L table), copied in by name. ``partner``: a band + DOS
    pair's other half from another folder (Add DOS…, Add Band…: --partner)."""
    inside = inside or (lambda f: True)
    if kind not in KINDS:
        raise ValueError(f"cannot render {kind!r} plots")
    if fmt and fmt not in FORMATS:
        raise ValueError(f"format is one of {', '.join(FORMATS)}")
    # (a band + DOS figure draws its bands along the edited path too: mace plotting --band-dos --band-path)
    if band_path is not None and kind not in ("band",) + PAIR_KINDS:
        band_path = None
    if kind not in ENERGY_KINDS:
        ewin, shift, reference = None, 0.0, "auto"
    if partner is not None and (kind not in PAIR_KINDS or not inside(partner)):
        raise ValueError(f"{partner.name}: a partner is the other half of a band + DOS figure, inside the folder")
    key = render_key(path, kind, fmt, band_path, ewin, shift, reference, look, explicit, extra_files, partner)
    work = disk.acquire(key)
    ok = False
    try:
        done = work / ".done"
        if not done.exists():
            for p in list(work.iterdir()):
                shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink()
            if ctx is not None:
                ctx.progress(0.1, "Copying the data files")
            shutil.copy2(path, work / path.name)
            # band + DOS: the partner of the same run too (mace plotting --band-dos finds it beside the file)
            datas = [(path, work)]
            if kind in PAIR_KINDS and partner is not None:
                # the other half from another folder, with its run's files, in a folder of its own
                (work / PARTNER_DIR).mkdir()
                shutil.copy2(partner, work / PARTNER_DIR / partner.name)
                datas.append((partner, work / PARTNER_DIR))
            elif kind in PAIR_KINDS:
                from . import results
                from mace.plotting import f25
                is_band = path.name.endswith(KINDS["band"][1]) or (f25.is_f25_name(path.name) and f25.plot_kind(path) == "band")
                other = results.partner(path, "band" if is_band else "dos")
                if other is not None and not inside(other):
                    other = None
                if other is None:
                    raise ValueError(f"{path.name}: no band structure and DOS of the same run beside each other")
                shutil.copy2(other, work / other.name)
                datas.append((other, work))
            # a transport file: the run's other files (its derived quantities need them), .out and .d3
            if kind == "transport":
                for sib in transport_files(path, inside):
                    shutil.copy2(sib, work / sib.name)
            for f in extra_files or []:
                shutil.copy2(f, work / Path(f).name)
            for data, dest in datas:
                stem = _run_stem(data.name, kind)
                for sib in data.parent.iterdir():
                    if (sib.is_file() and sib.name.startswith(stem) and sib not in [x for x, _ in datas]
                            and sib.suffix in _SIBLING_SUFFIXES and inside(sib)
                            and sib.stat().st_size < _SIBLING_MAX):
                        shutil.copy2(sib, dest / sib.name)
            if partner is not None:
                # the partner's run output under the SCF run's name (its E_F and cell check the pair)
                from . import results
                is_band = partner.name.endswith(KINDS["band"][1]) or results._f25_is(partner, "band") and \
                    not results._f25_is(partner, "dos")
                out = results.run_output(partner, "band" if is_band else "dos")
                if out is not None and inside(out) and not (work / PARTNER_DIR / out.name).exists() \
                        and out.stat().st_size < _SIBLING_MAX:
                    shutil.copy2(out, work / PARTNER_DIR / out.name)
            # a POTC run: the vacuum scale, as the GUI's plot draws it
            for sib in _potc_files(path, kind):
                if inside(sib):
                    shutil.copy2(sib, work / sib.name)
            if ctx is not None:
                ctx.progress(0.3, f"Running {command(path, kind, fmt, band_path, ewin, shift, reference, look, explicit, partner)}")
            args = ["--no-banner", "plotting", KINDS[kind][0], str(work / path.name)]
            if partner is not None:
                args += ["--partner", str(work / PARTNER_DIR / partner.name)]
            if fmt:
                args += ["--format", fmt]
            if band_path:
                (work / BAND_PATH_FILE).write_text(json.dumps({"band_path": band_path}, indent=1))
                args += ["--band-path", str(work / BAND_PATH_FILE)]
            args += _window_args(kind, ewin, shift, reference, explicit) + list(look or [])
            result = clirun.run(clirun.mace_argv(*args), work, ctx, timeout=300, env=env)
            (work / "log.txt").write_text(result.log)
            done.write_text(str(result.returncode))
        if ctx is not None:
            ctx.progress(0.9, "Reading the images")
        images = sorted(p.name for p in work.iterdir() if p.suffix.lower() in (".svg", ".png", ".pdf"))
        log = (work / "log.txt").read_text() if (work / "log.txt").exists() else ""
        ok = True
        return {"key": key, "images": images, "log": log.replace(str(work) + "/", "")[-4000:],
                "command": command(path, kind, fmt, band_path, ewin, shift, reference, look, explicit, partner)}
    finally:
        # a canceled or failed render leaves nothing half-made behind
        disk.release(key, remove=not ok)


def cached_file(cache: Path, key: str, name: str) -> Optional[Path]:
    """A file render() produced, or None (no path tricks)."""
    if not key.isalnum() or "/" in name or "\\" in name or name.startswith("."):
        return None
    target = cache / key / name
    return target if target.is_file() else None
