"""
MACE Plotting Module - Main Entry Point

Provides interactive and command-line interfaces for plotting CRYSTAL outputs.

Dispatch is registry-driven: discovery, the interactive menu, and CLI routing
all iterate ``registry.entries()`` rather than hard-coding band/DOS/structure.
Adding a new visualization (cube / FREQ / IR / Raman) is a single
``register()`` call in ``handlers/`` plus its mode flag in ``create_parser``.

The band/DOS/structure plotters themselves live in ``handlers/legacy.py``
(moved verbatim from this file); the underlying engines under ``Plotting/``
are untouched.

Author: Marcus Djokic
Institution: Michigan State University, Mendoza Group
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Any

from . import handlers  # noqa: F401  (import side-effect: registers all plotters)
from . import detect
from .registry import PlotKind, REGISTRY, entries, get
from .prompts import select_option, yes_no_prompt


# Legacy string keys for the back-compat discover_plottable_files() shim.
_LEGACY_KEY = {PlotKind.BAND: 'band', PlotKind.DOS: 'dos', PlotKind.STRUCTURE: 'cif'}


# =============================================================================
# File Discovery
# =============================================================================

def discover_plottable_files(directory: str = ".") -> Dict[str, List[str]]:
    """Back-compat shim returning the legacy ``{'band','dos','cif'}`` dict.

    Discovery itself is now registry-driven (:func:`detect.discover`); this
    wrapper preserves the old string-keyed return shape for any caller that
    still expects it.
    """
    by_kind = detect.discover(directory)
    results: Dict[str, List[str]] = {'band': [], 'dos': [], 'cif': []}
    for kind, key in _LEGACY_KEY.items():
        results[key] = by_kind.get(kind, [])
    return results


def print_discovered_files(by_kind: Dict[PlotKind, List[str]]) -> None:
    """Print a summary of discovered plottable files (registry-driven)."""
    print("\n" + "=" * 60)
    print("  DISCOVERED FILES")
    print("=" * 60)

    total = sum(len(v) for v in by_kind.values())
    if total == 0:
        print("  No plottable files found in current directory.")
        print("\n  Supported file types:")
        print("    - Band structure: *.band.band.dat, *.BAND.DAT")
        print("    - DOS: *_doss.DOSS.DAT, *.DOSS.DAT")
        print("    - Structures: *.cif")
        return

    for entry in entries():
        flist = by_kind.get(entry.kind, [])
        if not flist:
            continue
        print(f"\n  {entry.discovery_label} Files ({len(flist)}):")
        for f in flist[:5]:
            print(f"    - {Path(f).name}")
        if len(flist) > 5:
            print(f"    ... and {len(flist) - 5} more")

    print()


# =============================================================================
# Main Interactive Interface
# =============================================================================

def run_interactive(directory: str = ".") -> None:
    """Run the interactive plotting interface (registry-driven)."""
    print("\n" + "=" * 60)
    print("  MACE PLOTTING")
    print("=" * 60)
    print("  Interactive plotting for CRYSTAL calculation outputs")

    by_kind = detect.discover(directory)
    print_discovered_files(by_kind)

    total_files = sum(len(v) for v in by_kind.values())
    if total_files == 0:
        return

    # Build menu from whichever kinds are present, in registration order.
    present = [e for e in entries() if by_kind.get(e.kind)]
    options: List[str] = []
    option_map: Dict[int, Any] = {}

    for entry in present:
        options.append(entry.menu_tmpl.format(n=len(by_kind[entry.kind])))
        option_map[len(options)] = entry.kind

    if len(present) > 1:
        options.append("Plot all available")
        option_map[len(options)] = 'all'

    options.append("Exit")
    option_map[len(options)] = 'exit'

    while True:
        choice = select_option("What would you like to plot?", options, default=1)
        action = option_map[choice]

        if action == 'exit':
            print("\n  Exiting MACE plotting.")
            return

        if action == 'all':
            for entry in present:
                config = entry.configure(interactive=True)
                entry.handler(by_kind[entry.kind], config, directory)
        else:
            entry = REGISTRY[action]
            config = entry.configure(interactive=True)
            entry.handler(by_kind[entry.kind], config, directory)

        print("\n" + "-" * 40)
        if not yes_no_prompt("Plot something else?", "no"):
            print("\n  Done!")
            return


# =============================================================================
# Command-Line Interface
# =============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for mace plotting."""
    parser = argparse.ArgumentParser(
        prog='mace plotting',
        description='Plot CRYSTAL calculation outputs (bands, DOS, structures, '
                    'cube volumetrics, FREQ vibrational modes, IR/Raman spectra)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mace plotting                          # Interactive mode in current directory
  mace plotting --band                   # Plot all band structures with defaults
  mace plotting --dos                    # Plot all DOS with defaults
  mace plotting --structure              # Visualize all CIF files
  mace plotting --all                    # Plot everything discovered with defaults

  mace plotting --cube                   # Render every cube (.CUBE) in the directory
  mace plotting --cube dens.CUBE --iso 0.001,0.01
  mace plotting --esp pot.CUBE --view slice --slice z 30
  mace plotting --diff after.CUBE before.CUBE   # density difference

  mace plotting --freq mol_freq.out --list-modes
  mace plotting --freq mol_freq.out --mode 7 --gif
  mace plotting --freq mol_freq.out --all-modes

  mace plotting --ir                     # plot every *IRSPEC.DAT
  mace plotting --raman --raman-mode all # Raman incl. single-crystal directions
  mace plotting --spectra --average      # IR + Raman, conf configs averaged
"""
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Run in interactive mode (default if no other options)'
    )
    mode_group.add_argument(
        '--band',
        action='store_true',
        help='Plot band structures'
    )
    mode_group.add_argument(
        '--dos',
        action='store_true',
        help='Plot density of states'
    )
    mode_group.add_argument(
        '--structure',
        action='store_true',
        help='Visualize crystal structures from CIF files'
    )
    mode_group.add_argument(
        '--cube',
        action='store_true',
        help='Plot cube volumetric data (ECH3/POT3 density, potential, spin)'
    )
    mode_group.add_argument(
        '--freq', '--vibmodes',
        dest='freq',
        action='store_true',
        help='Render FREQ vibrational normal modes'
    )
    mode_group.add_argument(
        '--ir',
        action='store_true',
        help='Plot IR spectra (*IRSPEC.DAT)'
    )
    mode_group.add_argument(
        '--raman',
        action='store_true',
        help='Plot Raman spectra (*RAMSPEC.DAT)'
    )
    mode_group.add_argument(
        '--spectra',
        action='store_true',
        help='Plot both IR and Raman spectra (umbrella)'
    )
    mode_group.add_argument(
        '--all',
        action='store_true',
        help='Plot all available outputs with defaults'
    )

    # Explicit input file(s); when omitted, files are auto-discovered in -d.
    parser.add_argument(
        'files',
        nargs='*',
        metavar='FILE',
        help='Explicit input file(s); omit to auto-discover in --directory'
    )
    parser.add_argument(
        '--format',
        choices=['html', 'png', 'svg', 'pdf'],
        help='Output format for cube/FREQ plots (default: html)'
    )

    # Common options
    parser.add_argument(
        '-d', '--directory',
        default='.',
        help='Working directory (default: current)'
    )
    parser.add_argument(
        '-o', '--output',
        default='.',
        help='Output directory for generated plots'
    )

    # Band options
    band_group = parser.add_argument_group('band structure options')
    band_group.add_argument(
        '--e-lower',
        type=float,
        default=-8.0,
        help='Lower energy limit in eV (default: -8)'
    )
    band_group.add_argument(
        '--e-upper',
        type=float,
        default=8.0,
        help='Upper energy limit in eV (default: 8)'
    )
    band_group.add_argument(
        '--alpha',
        type=float,
        default=0.3,
        help='Transparency for band stacking (default: 0.3)'
    )
    band_group.add_argument(
        '--no-gaps',
        action='store_true',
        help='Remove gaps at k-path discontinuities'
    )

    # DOS options
    dos_group = parser.add_argument_group('DOS options')
    dos_group.add_argument(
        '--projection',
        choices=['both', 'total', 'orbital', 'tm_orb'],
        default='both',
        help='Projection type (default: both)'
    )
    dos_group.add_argument(
        '--no-composition',
        action='store_true',
        help='Disable VB/CB composition labels'
    )

    # Structure options
    struct_group = parser.add_argument_group('structure visualization options')
    struct_group.add_argument(
        '--supercell',
        type=int,
        nargs=3,
        default=[2, 2, 2],
        metavar=('NX', 'NY', 'NZ'),
        help='Supercell dimensions (default: 2 2 2)'
    )
    struct_group.add_argument(
        '--bond-cutoff',
        type=float,
        default=1.9,
        help='Bond cutoff in Angstroms (default: 1.9)'
    )
    struct_group.add_argument(
        '--color-by-coord',
        action='store_true',
        help='Color atoms by coordination number'
    )

    # Cube visualization options
    cube_group = parser.add_argument_group('cube visualization options')
    cube_group.add_argument('--density', '--charge', dest='density', action='store_true',
                            help='Cube: select density quantity (implies --cube)')
    cube_group.add_argument('--esp', '--potential', dest='esp', action='store_true',
                            help='Cube: select electrostatic potential (implies --cube)')
    cube_group.add_argument('--spin', action='store_true',
                            help='Cube: select spin density (implies --cube)')
    cube_group.add_argument('--diff', nargs=2, metavar=('A', 'B'),
                            help='Cube difference A - B (implies --cube)')
    cube_group.add_argument('--iso',
                            help='Cube isovalue(s), comma-separated (e.g. 0.001,0.01)')
    cube_group.add_argument('--view', choices=['iso', 'slice', 'slice-all'],
                            help='Cube view (default: iso)')
    cube_group.add_argument('--slice', nargs=2, metavar=('AXIS', 'POS'),
                            help='Single 2D slice: axis (x/y/z) and integer index')
    cube_group.add_argument('--slice-all', metavar='AXIS',
                            help='Grid of slices along AXIS (x/y/z)')
    cube_group.add_argument('--colorscale', help='Plotly colorscale name')
    cube_group.add_argument('--no-atoms', action='store_true', help='Cube: do not draw atoms')
    cube_group.add_argument('--bonds', action='store_true', help='Cube: draw bonds')
    cube_group.add_argument('--publication', action='store_true', help='Cube: publication styling')
    cube_group.add_argument('--log', action='store_true', help='Cube: log color scale')
    cube_group.add_argument('--linear', action='store_true', help='Cube: linear color scale')
    cube_group.add_argument('--clip', type=float, default=99.5,
                            help='Cube: color clip percentile (default: 99.5)')
    cube_group.add_argument('--engine-args',
                            help='Advanced: extra cube-engine flags (shlex-split; '
                                 'output-controlling flags are screened out)')

    # Vibrational mode options ( --all-modes is deliberately NOT in mode_group,
    # so it never clashes with the top-level --all; see H5 ).
    freq_group = parser.add_argument_group('vibrational mode options')
    freq_group.add_argument('--mode', type=int, help='FREQ: render this mode number')
    freq_group.add_argument('--all-modes', action='store_true',
                            help='FREQ: render every mode into one HTML (explicit opt-in)')
    freq_group.add_argument('--list-modes', action='store_true',
                            help='FREQ: list the mode table only (no render)')
    freq_group.add_argument('--amplitude', type=float, default=1.0,
                            help='FREQ: displacement amplitude (default: 1.0)')
    freq_group.add_argument('--gif', action='store_true',
                            help='FREQ: export an animated GIF instead of HTML')
    freq_group.add_argument('--gif-fps', type=int, default=20,
                            help='FREQ: GIF frames per second (default: 20)')
    freq_group.add_argument('--static', action='store_true',
                            help='FREQ: static view with arrows (no animation)')
    freq_group.add_argument('--normalize', action='store_true',
                            help='FREQ: normalize displacements for visibility')
    freq_group.add_argument('--frames', type=int,
                            help='FREQ: animation frame count (default: 30)')

    # Spectra (IR / Raman) options
    spectra_group = parser.add_argument_group('spectra (IR / Raman) options')
    spectra_group.add_argument('--raman-mode', choices=['total', 'par_perp', 'all'],
                               help="Raman intensities: total / par_perp / all (default: all)")
    spectra_group.add_argument('--average', action='store_true',
                               help='Average conf-style (-confN_) spectra per material')
    spectra_group.add_argument('--ir-column', type=int, default=2,
                               help='IR intensity column index (default: 2 = absorbance)')

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for mace plotting.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:])

    Returns:
        Exit code (0 for success, 1 if a requested kind has no files).
    """
    parser = create_parser()

    if argv is None:
        argv = sys.argv[1:]

    args = parser.parse_args(argv)
    files_pos = list(getattr(args, 'files', []) or [])

    # H4: --diff owns exactly its two operands; extra positionals are an error.
    if getattr(args, 'diff', None) and files_pos:
        parser.error('--diff takes exactly two cube files; '
                     'remove the extra positional argument(s)')

    # A mode flag pins the kind (mutually exclusive, so at most one matches);
    # cube sub-flags (--density/--esp/--spin/--diff) imply --cube.
    pinned = next((e for e in entries() if getattr(args, e.flag, False)), None)
    if pinned is None and _cube_implied(args):
        pinned = get(PlotKind.CUBE)

    if pinned is not None:
        return _dispatch_pinned(pinned, args, files_pos)

    # --spectra umbrella: both IR and Raman.
    if getattr(args, 'spectra', False):
        return _dispatch_kinds([PlotKind.SPECTRA_IR, PlotKind.SPECTRA_RAMAN],
                               args, files_pos,
                               empty_msg="  No spectra (IRSPEC/RAMSPEC .DAT) files found.")

    if args.all:
        by_kind = detect.discover(args.directory)
        for entry in entries():
            files = by_kind.get(entry.kind, [])
            if not files:
                continue
            config = entry.configure(interactive=False)
            if entry.accepts_energy_range:
                config['e_lower'] = args.e_lower
                config['e_upper'] = args.e_upper
            print(entry.progress_tmpl.format(n=len(files)))
            entry.handler(files, config, args.output)
        return 0

    # Explicit files with no mode flag: classify each and dispatch by kind.
    if files_pos:
        return _dispatch_positional(args, files_pos)

    # Interactive mode (default)
    run_interactive(args.directory)
    return 0


def _cube_implied(args) -> bool:
    """True when a cube sub-flag is present without an explicit mode flag."""
    return bool(getattr(args, 'density', False) or getattr(args, 'esp', False)
                or getattr(args, 'spin', False) or getattr(args, 'diff', None))


def _filter_cube_subtype(files: List[str], args) -> List[str]:
    """Narrow discovered cubes to a requested sub-type by the trailing filename
    token (``_DENS``/``_POT``/``_SPIN``.CUBE). Filename-level only — the engine
    remains the authority on the data-level sub-type (integration-plan C2)."""
    token = ('dens' if getattr(args, 'density', False)
             else 'pot' if getattr(args, 'esp', False)
             else 'spin' if getattr(args, 'spin', False) else None)
    if token is None:
        return files
    return [f for f in files if Path(f).name.lower().endswith(f'_{token}.cube')]


def _config_for(entry, args):
    return (entry.config_from_args(args)
            if entry.config_from_args else entry.configure(interactive=False))


def _dispatch_pinned(entry, args, files_pos: List[str]) -> int:
    """Run a single pinned plotter: explicit files win, else discover its kind."""
    if entry.kind == PlotKind.CUBE and getattr(args, 'diff', None):
        files = list(args.diff)
    elif files_pos:
        files = files_pos
    else:
        files = detect.discover(args.directory).get(entry.kind, [])
        if entry.kind == PlotKind.CUBE:
            files = _filter_cube_subtype(files, args)

    if not files:
        print(entry.not_found_msg)
        return 1

    config = _config_for(entry, args)
    print(entry.progress_tmpl.format(n=len(files)))
    entry.handler(files, config, args.output)
    return 0


def _dispatch_kinds(kinds, args, files_pos: List[str], empty_msg: str) -> int:
    """Dispatch several kinds at once (the --spectra umbrella). Explicit files
    are classified/routed by content; otherwise each kind is discovered."""
    if files_pos:
        return _dispatch_positional(args, files_pos)

    by_kind = detect.discover(args.directory)
    any_done = False
    for kind in kinds:
        entry = get(kind)
        files = by_kind.get(kind, [])
        if not files:
            continue
        config = _config_for(entry, args)
        print(entry.progress_tmpl.format(n=len(files)))
        entry.handler(files, config, args.output)
        any_done = True

    if not any_done:
        print(empty_msg)
        return 1
    return 0


def _dispatch_positional(args, files_pos: List[str]) -> int:
    """Classify explicit files by content/name and dispatch grouped by kind."""
    groups: Dict[PlotKind, List[str]] = {}
    for f in files_pos:
        kind = detect.classify_file(f)
        if kind is None:
            print(f"  Skipping unrecognized file: {Path(f).name} "
                  f"(force with --cube/--freq/--band/--dos)")
            continue
        groups.setdefault(kind, []).append(f)

    if not groups:
        print("  No recognized plottable files.")
        return 1

    for entry in entries():               # registration order
        files = groups.get(entry.kind)
        if not files:
            continue
        config = _config_for(entry, args)
        print(entry.progress_tmpl.format(n=len(files)))
        entry.handler(files, config, args.output)
    return 0


if __name__ == '__main__':
    sys.exit(main())
