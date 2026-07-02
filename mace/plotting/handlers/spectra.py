"""IR / Raman spectra plotting handlers for mace plotting (Phase 4).

Wires the pure :mod:`spectra_api` functions to discovery, interactive config and
the registry. Two registry kinds — SPECTRA_IR (``*IRSPEC.DAT``) and
SPECTRA_RAMAN (``*RAMSPEC.DAT``) — discovered by filename suffix (glob), so no
content sniffing is needed.

Averaging mirrors the original scripts: only conf-style files (``-confN_``) are
averaged, grouped per material; non-conf files are always plotted individually.
Output is matplotlib raster/vector (png/svg/pdf) — there is no interactive HTML
for these 2D spectra, so a requested ``--format html`` falls back to png.

Author: Marcus Djokic
Institution: Michigan State University, Mendoza Group
"""
import re
from pathlib import Path
from typing import Any, Dict, List

from . import spectra_api as api
from ..prompts import configure_output_formats, select_option, yes_no_prompt
from ..registry import PlotKind, PlotterEntry, register

_CONF_RE = re.compile(r"-conf\d+_")
_IMAGE_FORMATS = {"png", "svg", "pdf"}


def _resolve_formats(args) -> List[str]:
    """Map the shared --format to a matplotlib-capable format (html -> png)."""
    fmt = getattr(args, "format", None)
    if fmt in _IMAGE_FORMATS:
        return [fmt]
    return ["png"]


def _stem(path: str) -> str:
    name = Path(path).name
    for ext in (".DAT", ".dat"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return name


# =============================================================================
# IR
# =============================================================================

def configure_ir_spectra(interactive: bool = True) -> Dict[str, Any]:
    config: Dict[str, Any] = {"average": False, "ir_column": 2, "formats": ["png"]}
    if not interactive:
        return config

    print("\n" + "-" * 50)
    print("  IR SPECTRUM CONFIGURATION")
    print("-" * 50)
    print("  IRSPEC.DAT -> absorbance vs wavenumber.")
    config["average"] = yes_no_prompt(
        "Average conf-style configs per material (one curve each)?", "no")
    config["formats"] = configure_output_formats(interactive=True)
    return config


def _ir_config_from_args(args) -> Dict[str, Any]:
    cfg = configure_ir_spectra(interactive=False)
    cfg["average"] = bool(getattr(args, "average", False))
    cfg["ir_column"] = getattr(args, "ir_column", 2)
    cfg["formats"] = _resolve_formats(args)
    return cfg


def plot_ir_spectra(files: List[str], config: Dict[str, Any], output_dir: str = ".") -> List[str]:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out: List[str] = []
    ir_column = config.get("ir_column", 2)
    formats = config.get("formats") or ["png"]

    conf = [f for f in files if _CONF_RE.search(Path(f).name)]
    nonconf = [f for f in files if f not in conf]

    individual = files
    if config.get("average") and conf:
        # Group by full path (dedup exact duplicates only): keying by basename
        # silently collapsed same-named conf files from different directories.
        groups = api.group_files_by_material(sorted(set(conf)))
        for material, paths in sorted(groups.items()):
            res = api.average_spectrum(paths, min_cols=ir_column + 1)
            if not res:
                continue
            avg, n = res
            prefix = str(Path(output_dir) / f"{material}_IRSPEC_avg")
            print(f"\n  IR (avg {n}): {material}")
            new = api.render_ir(avg, prefix, formats=formats, ir_column=ir_column,
                                title_suffix=f" - {material} (avg {n})")
            for p in new:
                print(f"    Generated: {Path(p).name}")
            out += new
        individual = nonconf

    for f in individual:
        print(f"\n  IR: {Path(f).name}")
        new = api.render_ir(api.read_spectrum(f), str(Path(output_dir) / _stem(f)),
                            formats=formats, ir_column=ir_column)
        if not new:
            print(f"    Skipped (needs > {ir_column} columns).")
        for p in new:
            print(f"    Generated: {Path(p).name}")
        out += new

    return out


# =============================================================================
# Raman
# =============================================================================

def configure_raman_spectra(interactive: bool = True) -> Dict[str, Any]:
    config: Dict[str, Any] = {"raman_mode": "all", "average": False, "formats": ["png"]}
    if not interactive:
        return config

    print("\n" + "-" * 50)
    print("  RAMAN SPECTRUM CONFIGURATION")
    print("-" * 50)
    print("  RAMSPEC.DAT -> intensity vs wavenumber.")
    mode_choice = select_option(
        "Which intensities?",
        [
            "Total only",
            "Total + parallel + perpendicular (powder polarizations)",
            "All: par/perp + the six single-crystal directions",
        ],
        default=3,
    )
    config["raman_mode"] = ["total", "par_perp", "all"][mode_choice - 1]
    config["average"] = yes_no_prompt(
        "Average conf-style configs per material?", "no")
    config["formats"] = configure_output_formats(interactive=True)
    return config


def _raman_config_from_args(args) -> Dict[str, Any]:
    cfg = configure_raman_spectra(interactive=False)
    cfg["average"] = bool(getattr(args, "average", False))
    cfg["raman_mode"] = getattr(args, "raman_mode", None) or "all"
    cfg["formats"] = _resolve_formats(args)
    return cfg


def plot_raman_spectra(files: List[str], config: Dict[str, Any], output_dir: str = ".") -> List[str]:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out: List[str] = []
    mode = config.get("raman_mode", "all")
    min_cols = api.RAMAN_MIN_COLS.get(mode, 10)
    formats = config.get("formats") or ["png"]

    conf = [f for f in files if _CONF_RE.search(Path(f).name)]
    nonconf = [f for f in files if f not in conf]

    individual = files
    if config.get("average") and conf:
        # Group by full path (dedup exact duplicates only) — see IR branch.
        groups = api.group_files_by_material(sorted(set(conf)))
        for material, paths in sorted(groups.items()):
            res = api.average_spectrum(paths, min_cols=min_cols)
            if not res:
                continue
            avg, n = res
            prefix = str(Path(output_dir) / f"{material}_RAMSPEC_avg")
            print(f"\n  Raman (avg {n}, {mode}): {material}")
            new = api.render_raman(avg, prefix, mode=mode, formats=formats,
                                   title_suffix=f" - {material} (avg {n})")
            for p in new:
                print(f"    Generated: {Path(p).name}")
            out += new
        individual = nonconf

    for f in individual:
        print(f"\n  Raman ({mode}): {Path(f).name}")
        new = api.render_raman(api.read_spectrum(f), str(Path(output_dir) / _stem(f)),
                               mode=mode, formats=formats)
        if not new:
            print(f"    Skipped (needs >= {min_cols} columns for mode '{mode}').")
        for p in new:
            print(f"    Generated: {Path(p).name}")
        out += new

    return out


# =============================================================================
# Registration
# =============================================================================

def register_spectra_plotters() -> None:
    register(PlotterEntry(
        kind=PlotKind.SPECTRA_IR,
        flag="ir",
        label="IR spectrum",
        handler=plot_ir_spectra,
        configure=configure_ir_spectra,
        config_from_args=_ir_config_from_args,
        patterns=["*IRSPEC.DAT", "*irspec.dat"],
        accepts_energy_range=False,
        progress_tmpl="\n  Plotting {n} IR spectrum/spectra...",
        not_found_msg="  No IRSPEC.DAT files found.",
        menu_tmpl="Plot IR spectra ({n} files)",
        discovery_label="IR Spectrum",
    ))
    register(PlotterEntry(
        kind=PlotKind.SPECTRA_RAMAN,
        flag="raman",
        label="Raman spectrum",
        handler=plot_raman_spectra,
        configure=configure_raman_spectra,
        config_from_args=_raman_config_from_args,
        patterns=["*RAMSPEC.DAT", "*ramspec.dat"],
        accepts_energy_range=False,
        progress_tmpl="\n  Plotting {n} Raman spectrum/spectra...",
        not_found_msg="  No RAMSPEC.DAT files found.",
        menu_tmpl="Plot Raman spectra ({n} files)",
        discovery_label="Raman Spectrum",
    ))


register_spectra_plotters()
