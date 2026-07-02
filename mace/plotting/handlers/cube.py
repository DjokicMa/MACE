"""Cube (ECH3/POT3 volumetric) plotting handler for mace plotting.

Drives the relocated ``engines.crystal_cubeviz_plotly`` engine the same way the
legacy band/DOS handlers drive theirs: build an argv, point it at an explicit
``--save`` path so the engine *writes* HTML/PNG instead of ``fig.show()``, call
``main()`` inside a SystemExit-safe guard, and collect the files that appeared.

The engine resolves the cube sub-type (density / potential / spin / difference)
itself from the comment + filename, so this handler does not re-derive it — it
just forwards quantity sub-flags as the engine's own arithmetic/quantity options
(single source of truth, integration-plan C2). The ~66 engine flags are reachable
through ``engine_args`` (escape hatch), screened so output-controlling flags can
never override the ``--save`` writer discipline (integration-plan L4).

Author: Marcus Djokic
Institution: Michigan State University, Mendoza Group
"""
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..prompts import (
    configure_plotly_formats,
    get_float_input,
    get_string_input,
    select_option,
    yes_no_prompt,
)
from ..registry import PlotKind, PlotterEntry, register

# Engine flags that write output or open a browser: never honored from the
# user's --engine-args escape hatch, so mace's explicit --save path always wins.
_OUTPUT_CONTROLLING = frozenset({
    "--save", "--save-png", "--save-svg", "--save-pdf", "--save-gif",
    "--output-cube", "--visualize-result",
    "-i", "--interactive", "--iso-browse", "--slice-browse",
})

# mace --format -> engine save flag + file extension.
_FORMAT_FLAG = {
    "html": ("--save", "html"),
    "png": ("--save-png", "png"),
    "svg": ("--save-svg", "svg"),
    "pdf": ("--save-pdf", "pdf"),
}


def _engine_main():
    """Import the cube engine lazily (plotly/numpy only needed at render time)."""
    from ..engines import crystal_cubeviz_plotly as ccp
    return ccp.main


def _screen_engine_args(tokens: List[str]) -> Tuple[List[str], List[str]]:
    """Split escape-hatch tokens into (kept, rejected). A rejected output-control
    flag drops itself; following non-flag operands are left in place (argparse in
    the engine ignores stray values it doesn't expect only if attached — so we
    drop just the flag token and let the user see the warning)."""
    kept, rejected = [], []
    for tok in tokens:
        if tok in _OUTPUT_CONTROLLING:
            rejected.append(tok)
        else:
            kept.append(tok)
    return kept, rejected


# =============================================================================
# Configuration
# =============================================================================

def configure_cube_plot(interactive: bool = True) -> Dict[str, Any]:
    """Configure cube visualization parameters."""
    config: Dict[str, Any] = {
        "view": "iso",          # iso | slice | slice-all
        "iso": None,            # None -> engine auto-recommends isovalues
        "colorscale": None,
        "alpha": 0.7,
        "show_atoms": True,
        "bonds": False,
        "publication": False,
        "scale": None,          # None | 'log' | 'linear'
        "clip": 99.5,
        "slice_axis": "z",
        "slice_pos": None,
        "operation": None,      # None | 'diff'
        "formats": ["html"],
        "engine_args": [],
    }

    if not interactive:
        return config

    print("\n" + "-" * 50)
    print("  CUBE VISUALIZATION CONFIGURATION")
    print("-" * 50)
    print("  Cube sub-type (density / potential / spin / difference) is detected")
    print("  automatically from each file; choose how to render it below.")

    view_choice = select_option(
        "How would you like to view the volume?",
        [
            "3D isosurface   - shells at constant value (best for densities/orbitals)",
            "Single 2D slice - one plane through the volume (best for ESP maps)",
            "Slice grid      - many parallel slices at once (explore the volume)",
        ],
        default=1,
    )
    config["view"] = ["iso", "slice", "slice-all"][view_choice - 1]

    if config["view"] == "iso":
        if yes_no_prompt("Auto-select isovalues (recommended)?", "yes"):
            config["iso"] = None
        else:
            while True:
                raw = get_string_input("  Isovalues, comma-separated (e.g. 0.001,0.01)", "0.001,0.01")
                try:
                    config["iso"] = [float(x) for x in raw.split(",") if x.strip()]
                    break
                except ValueError:
                    print(f"  '{raw}' is not a comma-separated list of numbers; try again.")
    else:
        axis_choice = select_option(
            "  Slice/stack axis:",
            ["z (recommended for slabs)", "y", "x"],
            default=1,
        )
        config["slice_axis"] = ["z", "y", "x"][axis_choice - 1]
        if config["view"] == "slice":
            if yes_no_prompt("  Use the best/middle slice automatically?", "yes"):
                config["slice_pos"] = None          # engine picks the best slice
            else:
                config["slice_pos"] = int(get_float_input("  Slice index (0-based)", 0))

    config["show_atoms"] = yes_no_prompt("Overlay atoms?", "yes")
    if config["show_atoms"]:
        config["bonds"] = yes_no_prompt("  Draw bonds between atoms?", "no")

    config["formats"] = configure_plotly_formats(interactive=True)

    if yes_no_prompt("Configure advanced appearance?", "no"):
        cs = get_string_input("  Plotly colorscale (blank = auto, e.g. Viridis/RdBu)", "")
        config["colorscale"] = cs or None
        scale_choice = select_option(
            "  Color scaling:",
            ["auto (engine decides)", "log (wide dynamic range)", "linear"],
            default=1,
        )
        config["scale"] = [None, "log", "linear"][scale_choice - 1]
        config["clip"] = get_float_input("  Color clip percentile (trim outliers)", 99.5)
        config["publication"] = yes_no_prompt("  Publication styling (clean axes, no title)?", "no")

    return config


def _cube_config_from_args(args) -> Dict[str, Any]:
    cfg = configure_cube_plot(interactive=False)

    # View: --slice / --slice-all override the default iso view.
    if getattr(args, "slice", None):
        cfg["view"] = "slice"
        cfg["slice_axis"] = str(args.slice[0]).lower()
        cfg["slice_pos"] = int(args.slice[1])
    elif getattr(args, "slice_all", None):
        cfg["view"] = "slice-all"
        cfg["slice_axis"] = str(args.slice_all).lower()
    elif getattr(args, "view", None):
        cfg["view"] = args.view

    iso = getattr(args, "iso", None)
    if iso:
        try:
            cfg["iso"] = [float(x) for x in str(iso).split(",") if x.strip()]
        except ValueError:
            # argparse-style usage error, not a traceback
            print(f"Error: --iso expects comma-separated numbers (e.g. 0.001,0.01), got '{iso}'")
            raise SystemExit(2)

    if getattr(args, "diff", None):
        cfg["operation"] = "diff"

    cfg["colorscale"] = getattr(args, "colorscale", None) or None
    # NB: --alpha is owned by the band group (different default/semantics); cube
    # opacity stays at its engine default and is reachable via --engine-args.
    cfg["show_atoms"] = not getattr(args, "no_atoms", False)
    cfg["bonds"] = bool(getattr(args, "bonds", False))
    cfg["publication"] = bool(getattr(args, "publication", False))
    if getattr(args, "log", False):
        cfg["scale"] = "log"
    elif getattr(args, "linear", False):
        cfg["scale"] = "linear"
    cfg["clip"] = getattr(args, "clip", cfg["clip"])

    fmt = getattr(args, "format", None)
    if fmt:
        cfg["formats"] = [fmt]

    ea = getattr(args, "engine_args", None)
    if ea:
        import shlex
        cfg["engine_args"] = shlex.split(ea) if isinstance(ea, str) else list(ea)

    return cfg


# =============================================================================
# Rendering
# =============================================================================

def _common_argv(config: Dict[str, Any]) -> List[str]:
    """Engine flags shared across views (not the per-file filename/save)."""
    argv: List[str] = []
    if config["view"] == "iso":
        if config.get("iso"):
            argv += ["--iso", ",".join(str(v) for v in config["iso"])]
    elif config["view"] == "slice":
        pos = config.get("slice_pos")
        # A non-numeric position makes the engine pick its own best/middle slice
        # (CubeFile.get_best_visualization_slice) instead of a hard index-0 edge.
        argv += ["--slice", config.get("slice_axis", "z"),
                 str(pos) if pos is not None else "auto"]
    elif config["view"] == "slice-all":
        argv += ["--slice-all", config.get("slice_axis", "z")]

    if config.get("colorscale"):
        argv += ["--cmap", config["colorscale"]]
    argv += ["--alpha", str(config.get("alpha", 0.7))]
    if not config.get("show_atoms", True):
        argv += ["--no-atoms"]
    if config.get("bonds"):
        argv += ["--bonds"]
    if config.get("publication"):
        argv += ["--publication"]
    if config.get("scale") == "log":
        argv += ["--log-scale"]
    elif config.get("scale") == "linear":
        argv += ["--linear-scale"]
    argv += ["--clip", str(config.get("clip", 99.5))]

    engine_args = config.get("engine_args") or []
    if engine_args:
        kept, rejected = _screen_engine_args(list(engine_args))
        if rejected:
            print(f"    Ignoring output-controlling engine-args (mace owns output): {' '.join(rejected)}")
        argv += kept
    return argv


def _save_flags(base: str, output_dir: str, view: str, formats: List[str]) -> List[Tuple[str, str]]:
    """(engine_flag, path) pairs for each requested format."""
    pairs = []
    for fmt in formats:
        flag, ext = _FORMAT_FLAG.get(fmt, ("--save", "html"))
        path = str(Path(output_dir) / f"{base}.{view}.{ext}")
        pairs.append((flag, path))
    return pairs


def _run_engine(argv: List[str]) -> None:
    main = _engine_main()
    argv_backup = sys.argv
    try:
        sys.argv = ["crystal_cubeviz_plotly.py", *argv]
        main()
    except (Exception, SystemExit) as e:  # engines sys.exit() on bad input
        # Same contract as the freq handler: SystemExit(0/None) is a normal
        # engine exit (e.g. the all-zeros warning path), not an error; a bare
        # integer exit code means the engine already printed its message.
        if isinstance(e, SystemExit):
            if e.code in (0, None):
                return
            print(f"    Engine aborted (exit {e.code}) — see message above.")
        else:
            print(f"    Error: {e}")
    finally:
        sys.argv = argv_backup


def plot_cube(files: List[str], config: Dict[str, Any], output_dir: str = ".") -> List[str]:
    """Render cube files. ``operation='diff'`` subtracts exactly two cubes into a
    single figure; otherwise each file is rendered independently."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_files: List[str] = []
    common = _common_argv(config)
    view = config.get("view", "iso")
    formats = config.get("formats") or ["html"]

    if config.get("operation") == "diff":
        if len(files) != 2:
            print(f"  --diff requires exactly 2 cubes, got {len(files)}.")
            return []
        base = f"{Path(files[0]).stem}_minus_{Path(files[1]).stem}"
        print(f"\n  Difference: {Path(files[0]).name} - {Path(files[1]).name}")
        for flag, path in _save_flags(base, output_dir, "diff", formats):
            _run_engine(["--subtract", files[0], files[1], "--align-grids", *common, flag, path])
            if Path(path).exists():
                output_files.append(path)
                print(f"    Generated: {Path(path).name}")
        return output_files

    for cube_file in files:
        base = Path(cube_file).stem
        print(f"\n  Plotting cube: {Path(cube_file).name}")
        for flag, path in _save_flags(base, output_dir, view, formats):
            _run_engine([cube_file, *common, flag, path])
            if Path(path).exists():
                output_files.append(path)
                print(f"    Generated: {Path(path).name}")

    return output_files


# =============================================================================
# Registration
# =============================================================================

def register_cube_plotter() -> None:
    register(PlotterEntry(
        kind=PlotKind.CUBE,
        flag="cube",
        label="cube volumetric",
        handler=plot_cube,
        configure=configure_cube_plot,
        config_from_args=_cube_config_from_args,
        patterns=["*.cube", "*.CUBE"],
        accepts_energy_range=False,
        progress_tmpl="\n  Plotting {n} cube file(s)...",
        not_found_msg="  No cube (.CUBE) files found.",
        menu_tmpl="Plot cube volumetrics ({n} files)",
        discovery_label="Cube Volumetric",
    ))


register_cube_plotter()
