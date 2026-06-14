"""FREQ vibrational-mode plotting handler for mace plotting.

Drives the relocated ``engines.vibmode_viewer`` engine through a constructed argv
the same way the cube/legacy handlers do. The engine drops into an interactive
``input()`` loop or ``fig.show()`` unless told otherwise, so this handler ALWAYS
passes one of ``--list`` / ``--mode N`` / ``--all`` plus a save flag
(``--save`` for HTML, ``--gif`` for animation), keeping it fully headless.

Discovery uses the C1 content gate: the FREQ entry's ``sniff`` is
``detect.is_freq_output`` (phonon banner + a real NORMAL MODES block), so the
hundreds of ordinary ``.out`` runs are not mistaken for FREQ.

Per H5: ``--list`` short-circuits (prints the mode table, no render); the default
when no specific mode is requested is a SINGLE representative mode (the lowest
real mode), not every mode — ``--all-modes`` is the explicit opt-in.

Each file renders into its own ``output_dir/<stem>/`` subdirectory so multi-file
batches don't collide on the engine's fixed output names
(``all_modes_viewer.html`` / ``mode_<n>_<freq>cm-1.html``).

Author: Marcus Djokic
Institution: Michigan State University, Mendoza Group
"""
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .. import detect
from ..prompts import (
    get_float_input,
    select_option,
    yes_no_prompt,
)
from ..registry import PlotKind, PlotterEntry, register

_RENDER_EXTS = {".html", ".gif"}


def _engine():
    from ..engines import vibmode_viewer as vib
    return vib


# =============================================================================
# Mode selection
# =============================================================================

def _representative_mode(out_file: str) -> Optional[int]:
    """Lowest real (non-acoustic, non-imaginary) mode number, for the default
    single-mode render. Falls back to the first parsed mode; None if none."""
    vib = _engine()
    try:
        parser = vib.Crystal23FreqParser(out_file)
    except (OSError, Exception):  # noqa: BLE001 - parser is third-party-ish
        return None
    modes = getattr(parser, "modes", None) or []
    for mode in modes:
        if mode.get("freq", 0.0) > 1.0 and not mode.get("imaginary"):
            return int(mode["mode"])
    return int(modes[0]["mode"]) if modes else None


# =============================================================================
# Configuration
# =============================================================================

def configure_freq_plot(interactive: bool = True) -> Dict[str, Any]:
    """Configure FREQ vibrational-mode visualization parameters."""
    config: Dict[str, Any] = {
        "list_modes": False,
        "mode": None,           # specific mode number; None -> representative
        "all_modes": False,
        "amplitude": 1.0,
        "frames": 30,
        "gif": False,
        "gif_fps": 20,
        "static": False,
        "normalize": False,
        "formats": ["html"],
    }

    if not interactive:
        return config

    print("\n" + "-" * 50)
    print("  VIBRATIONAL MODE CONFIGURATION")
    print("-" * 50)
    print("  Each mode is an atomic-displacement pattern at a given frequency.")
    print("  Tip: choose 'List the mode table' first to see mode numbers, irreps,")
    print("  imaginary (soft) modes and IR/Raman activity before picking one.")

    choice = select_option(
        "What would you like to do?",
        [
            "Render one mode (animated 3D)",
            "List the mode table only (no render)",
            "Render every mode into a single HTML",
        ],
        default=1,
    )
    if choice == 2:
        config["list_modes"] = True
        return config

    if choice == 3:
        config["all_modes"] = True
    else:
        if yes_no_prompt("Use the lowest real mode automatically (representative)?", "yes"):
            config["mode"] = None
        else:
            config["mode"] = int(get_float_input("  Mode number (from the --list-modes table)", 1))

        out_choice = select_option(
            "  Output style:",
            [
                "Interactive animated HTML (recommended)",
                "Animated GIF (shareable, static file)",
                "Static HTML with displacement arrows (no animation)",
            ],
            default=1,
        )
        if out_choice == 2:
            config["gif"] = True
        elif out_choice == 3:
            config["static"] = True

    if yes_no_prompt("Configure advanced options?", "no"):
        config["amplitude"] = get_float_input("  Displacement amplitude (visual exaggeration)", 1.0)
        config["normalize"] = yes_no_prompt("  Normalize displacements for visibility?", "no")
        if config["gif"]:
            config["gif_fps"] = int(get_float_input("  GIF frames per second", 20))
        config["frames"] = int(get_float_input("  Animation frame count", 30))

    return config


def _freq_config_from_args(args) -> Dict[str, Any]:
    cfg = configure_freq_plot(interactive=False)
    cfg["list_modes"] = bool(getattr(args, "list_modes", False))
    cfg["all_modes"] = bool(getattr(args, "all_modes", False))
    if getattr(args, "mode", None) is not None:
        cfg["mode"] = int(args.mode)
    cfg["amplitude"] = getattr(args, "amplitude", cfg["amplitude"])
    cfg["gif"] = bool(getattr(args, "gif", False))
    cfg["gif_fps"] = getattr(args, "gif_fps", cfg["gif_fps"])
    cfg["static"] = bool(getattr(args, "static", False))
    cfg["normalize"] = bool(getattr(args, "normalize", False))
    frames = getattr(args, "frames", None)
    if frames:
        cfg["frames"] = int(frames)
    return cfg


# =============================================================================
# Rendering
# =============================================================================

def _run_engine(argv: List[str]) -> None:
    vib = _engine()
    argv_backup = sys.argv
    try:
        sys.argv = ["vibmode_viewer.py", *argv]
        vib.main()
    except (Exception, SystemExit) as e:  # engine sys.exit()s on --list/--all/error
        # SystemExit(0) is the normal --list/--all exit; only note real errors.
        if isinstance(e, SystemExit) and (e.code in (0, None)):
            return
        print(f"    Error: {e}")
    finally:
        sys.argv = argv_backup


def _collect_new(out_dir: Path, before: set) -> List[str]:
    after = {p for p in out_dir.rglob("*") if p.suffix.lower() in _RENDER_EXTS}
    return sorted(str(p) for p in (after - before))


def plot_freq(files: List[str], config: Dict[str, Any], output_dir: str = ".") -> List[str]:
    """Render FREQ vibrational modes. ``list_modes`` short-circuits to the table."""
    output_files: List[str] = []

    for out_file in files:
        name = Path(out_file).name

        if config.get("list_modes"):
            print(f"\n  Modes in {name}:")
            _run_engine([out_file, "--list"])
            continue

        file_out = Path(output_dir) / Path(out_file).stem
        file_out.mkdir(parents=True, exist_ok=True)
        before = {p for p in file_out.rglob("*") if p.suffix.lower() in _RENDER_EXTS}

        argv = [out_file, "--output-dir", str(file_out),
                "--amplitude", str(config.get("amplitude", 1.0)),
                "--frames", str(config.get("frames", 30))]
        if config.get("normalize"):
            argv.append("--normalize")
        if config.get("static"):
            argv.append("--static")

        if config.get("all_modes"):
            print(f"\n  Rendering all modes: {name}")
            argv.append("--all")
        else:
            mode = config.get("mode") or _representative_mode(out_file)
            if mode is None:
                print(f"    No vibrational modes found in {name}; skipping.")
                continue
            print(f"\n  Rendering mode {mode}: {name}")
            argv += ["--mode", str(mode)]
            if config.get("gif"):
                argv += ["--gif", "--gif-fps", str(config.get("gif_fps", 20))]
            else:
                argv.append("--save")

        _run_engine(argv)
        new = _collect_new(file_out, before)
        for p in new:
            print(f"    Generated: {Path(p).name}")
        output_files += new

    return output_files


# =============================================================================
# Registration
# =============================================================================

def register_freq_plotter() -> None:
    register(PlotterEntry(
        kind=PlotKind.FREQ,
        flag="freq",
        label="vibrational mode",
        handler=plot_freq,
        configure=configure_freq_plot,
        config_from_args=_freq_config_from_args,
        patterns=["*.out", "*.OUT"],
        sniff=detect.is_freq_output,
        accepts_energy_range=False,
        progress_tmpl="\n  Rendering vibrational modes for {n} file(s)...",
        not_found_msg="  No FREQ (.out with normal modes) files found.",
        menu_tmpl="Render vibrational modes ({n} files)",
        discovery_label="FREQ Vibrational Mode",
    ))


register_freq_plotter()
