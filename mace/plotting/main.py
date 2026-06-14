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
from .registry import PlotKind, REGISTRY, entries
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
        description='Plot CRYSTAL calculation outputs (bands, DOS, structures)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mace plotting                     # Interactive mode in current directory
  mace plotting --band              # Plot all band structures with defaults
  mace plotting --dos               # Plot all DOS with defaults
  mace plotting --structure         # Visualize all CIF files
  mace plotting --all               # Plot everything with defaults
  mace plotting --band -8 8         # Band plot with custom energy range
  mace plotting --dos -8 8 both     # DOS plot with projection type
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
        '--all',
        action='store_true',
        help='Plot all available outputs with defaults'
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

    # Single-kind mode (--band / --dos / --structure). Mode flags are mutually
    # exclusive, so at most one matches.
    for entry in entries():
        if getattr(args, entry.flag, False):
            by_kind = detect.discover(args.directory)
            files = by_kind.get(entry.kind, [])
            if not files:
                print(entry.not_found_msg)
                return 1
            config = (entry.config_from_args(args)
                      if entry.config_from_args else entry.configure(interactive=False))
            print(entry.progress_tmpl.format(n=len(files)))
            entry.handler(files, config, args.output)
            return 0

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

    # Interactive mode (default)
    run_interactive(args.directory)
    return 0


if __name__ == '__main__':
    sys.exit(main())
