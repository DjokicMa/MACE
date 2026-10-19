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

# Force a non-interactive backend for file output BEFORE any plotter pulls
# in pyplot: on headless HPC nodes matplotlib auto-picks a GUI backend (GTK)
# and dies at savefig with "gdk-pixbuf-error-quark: Couldn't recognize the
# image file format" (real case: login-node band plot, phase-5 QA).
# setdefault keeps an explicit user MPLBACKEND override working.
os.environ.setdefault("MPLBACKEND", "Agg")

from . import handlers  # noqa: F401  (import side-effect: registers all plotters)
from . import detect, outcome
from .registry import PlotKind, REGISTRY, entries, get
from .prompts import select_option, yes_no_prompt
from mace.utils import folder_settings as _fs


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

def _dos_smooth_ev(text: str) -> float:
    """--dos-smooth's value: a Gaussian sigma in eV from 0 (off) to 1."""
    try:
        v = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number (a sigma in eV, e.g. 0.05)")
    if not 0 <= v <= 1:
        raise argparse.ArgumentTypeError(f"{text} is outside 0 to 1 eV (0 turns the smoothing off)")
    return v


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
  mace plotting --band-dos               # Each band structure with its DOS, one figure (as the GUI)
  mace plotting --band --backend matplotlib   # the figures of MACE 1.1.5 and earlier
  mace plotting --structure              # Visualize all CIF files
  mace plotting --all                    # Plot everything discovered with defaults

  mace plotting --cube                   # Render every cube (.CUBE) in the directory
  mace plotting --cube dens.CUBE --iso 0.001,0.01
  mace plotting --esp pot.CUBE --view slice --slice z 30
  mace plotting --diff after.CUBE before.CUBE   # density difference

  mace plotting --orbitals                         # HOMO and LUMO of the ORBITALS run here
  mace plotting --orbitals si_orb_00001_K000_real.molden --mo HOMO --degenerate-sum
  mace plotting --orbitals --list-orbitals         # k points, levels, self-check
  mace plotting --freq mol_freq.out --list-modes
  mace plotting --freq mol_freq.out --mode 7 --gif
  mace plotting --freq mol_freq.out --all-modes

  mace plotting --ir                     # plot every *IRSPEC.DAT
  mace plotting --raman --raman-mode all # Raman incl. single-crystal directions
  mace plotting --spectra --average      # IR + Raman, conf configs averaged

  mace plotting --legend MgO.cif         # element legend SVG in the GUI's palette
  mace plotting --legend runs/ --format png --dpi 600 --legend-layout grid
  mace plotting --gallery band BAND/ --cols 4 -o bands.svg   # many band structures in one grid
  mace plotting --gallery bandos run1/ run2/ --caption-csv captions.csv
  mace plotting --gallery opt OPT/ -o opt.png               # energy along each optimization
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
        '--band-dos',
        dest='band_dos',
        action='store_true',
        help="Plot each band structure with its run's DOS in one figure on a shared energy axis "
             "(mace gui's band + DOS view): FILE is a BAND or a DOSS file, its partner sits beside it"
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
        '--orbitals',
        action='store_true',
        help='Plot orbitals from a CRYSTAL ORBITALS run (.molden files): choose the orbital, '
             'k point and spin; a cube is built in <job>_orbitals/ and drawn like other cubes'
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
        '--bz',
        action='store_true',
        help='Draw the first Brillouin zone with the k-path (BAND.DAT/.d3/.out; --band-path for an edit)'
    )
    mode_group.add_argument(
        '--all',
        action='store_true',
        help='Plot all available outputs with defaults'
    )
    mode_group.add_argument(
        '--legend',
        action='store_true',
        help='Write an element legend (SVG, PNG or a TSV/CSV/JSON table) for the elements of the '
             'given structure files or folders (their union), in the element palette (--palette)'
    )
    mode_group.add_argument(
        '--gallery',
        metavar='KIND',
        choices=['band', 'dos', 'bandos', 'ir', 'raman', 'opt', 'scf', 'structure'],
        help='Many outputs in one grid figure: band, dos, bandos (band + DOS), ir, raman, '
             'opt (energy along the optimization), scf (SCF convergence) or structure'
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
        choices=['html', 'png', 'svg', 'pdf', 'tsv', 'csv', 'json'],
        help='Output format: cube/FREQ html (default), png, svg, pdf; band/DOS/band-DOS png, svg, pdf or html '
             '(default: png and svg); --legend: svg (default), png, tsv, csv or json'
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
        help='Output directory for generated plots (band/DOS: beside the data when not given)'
    )

    # Plot styles (1.1.6, mace/plotting/presets.py STYLE_OPTIONS: the same names as the GUI's Style controls
    # and a folder's mace-folder.toml [plots] keys)
    from .presets import ORDER, DOS_MODES, SPIN_COLOR_CHOICES, DOS_SMOOTH_SIGMA
    style_group = parser.add_argument_group('plot style options (band, DOS, band + DOS, gallery, spectra, cube maps)')
    style_group.add_argument('--backend', choices=['plotly', 'matplotlib'], default=None,
                             help='Band/DOS renderer: plotly draws the GUI\'s figure (default, when plotly and kaleido '
                                  'can export here), matplotlib the figures of MACE 1.1.5 and earlier')
    style_group.add_argument('--style', choices=list(ORDER), default=None,
                             help='Plot style: default, publication (Okabe-Ito after black, no grid, ticks outside), '
                                  'vesta (cube maps and isosurfaces: VESTA-like; other plots draw default), '
                                  'presentation (large text, thick lines), grayscale (print: line styles and markers), '
                                  'colorblind (Okabe-Ito). Default: default')
    style_group.add_argument('--spin-colors', choices=list(SPIN_COLOR_CHOICES), default=None,
                             help="Band/DOS spin colors: auto (the style's: blue/red in default and presentation, "
                                  "black/vermilion in publication), mace (pink/cyan), blue-red, black-vermilion, or "
                                  "custom (--spin-up-color/--spin-down-color). Default: auto")
    style_group.add_argument('--potential-colors', choices=['chemistry', 'physics'], default=None,
                             help='Cube maps and isosurfaces: electrostatic potential colors, chemistry (default: '
                                  'red = negative, electron-rich; blue = positive) or physics (red = positive), '
                                  'as the GUI\'s Potential colors')
    style_group.add_argument('--spin-up-color', metavar='HEX', default=None,
                             help="Band/DOS: spin up's color, #rrggbb (a custom pair; default: the spin colors')")
    style_group.add_argument('--spin-down-color', metavar='HEX', default=None,
                             help="Band/DOS: spin down's color, #rrggbb (a custom pair; default: the spin colors')")
    style_group.add_argument('--band-color', metavar='HEX', default=None,
                             help="Band/DOS: the color of a run that is not spin-polarized (closed shell, or α = β): its "
                                  "bands and one-sided DOS total, #rrggbb (default: the style's, the plot's text color; "
                                  "a spin-polarized run keeps the spin colors)")
    style_group.add_argument('--spin-down-style', '--spin-style', dest='spin_style',
                             choices=['dash', 'dot', 'dashdot', 'solid'], default=None,
                             help="Band/DOS: spin down's (β's) line style, spin up solid (default: dash)")
    style_group.add_argument('--spin-down-solid', dest='spin_style', action='store_const', const='solid',
                             help='Band/DOS: spin down solid too (the spins told apart by color only)')
    style_group.add_argument('--dos-colors', choices=list(DOS_MODES), default=None,
                             help="DOS: which projections and their colors: auto (one element by orbital, two or three "
                                  "element x orbital, more by element; default), total, element, orbital, element-orbital")
    style_group.add_argument('--no-dos-total', dest='dos_total', action='store_false', default=None,
                             help='DOS: without the total')
    style_group.add_argument('--dos-total', dest='dos_total', action='store_true', default=None,
                             help="DOS: with the total (the default; overrides a mace-folder.toml's dos_total = false)")
    style_group.add_argument('--dos-spin', '--spin-dos', dest='spin_dos', choices=['auto', 'mirrored', 'overlaid'], default=None,
                             help='DOS spin layout: auto (a spin-polarized run mirrored, a closed-shell run one-sided; '
                                  'default), mirrored or overlaid')
    style_group.add_argument('--dos-smooth', dest='dos_smooth', metavar='EV', type=_dos_smooth_ev, default=None,
                             help=f'DOS: Gaussian smoothing of the drawn curves (display only, labeled in the plot), '
                                  f'sigma in eV, e.g. {DOS_SMOOTH_SIGMA:g}; 0: off (default: off)')
    style_group.add_argument('--units', choices=['eV', 'meV', 'Ha', 'kJ/mol', 'kcal/mol'], default=None,
                             help='Band/DOS energy unit (plotly backend; default: eV)')
    style_group.add_argument('--width', type=float, default=None, metavar='IN',
                             help='Band/DOS/band + DOS figure width in inches (default: plotly the GUI pane\'s size, '
                                  'band and band + DOS 7.9 x 4.4 in, DOS 5.4 x 5.4 in; matplotlib 3.5 x 5 in, a band '
                                  'figure wider with many k labels; matplotlib widens a figure for its legend)')
    style_group.add_argument('--height', type=float, default=None, metavar='IN',
                             help='Band/DOS/band + DOS figure height in inches (default: see --width)')
    style_group.add_argument('--dpi', type=int, default=None,
                             help='PNG resolution, dots per inch, written into the PNG with the size: band/DOS/band + DOS '
                                  '(default: 300; --backend matplotlib band and DOS 600) and --legend (default: 300)')

    # Gallery options (mace/plotting/gallery.py, shared with the GUI's Plot Gallery)
    gallery_group = parser.add_argument_group('gallery options (--gallery KIND FILES/FOLDER)')
    gallery_group.add_argument('--cols', type=int, default=None,
                               help='Gallery: columns of the grid (default: 3, at most 8)')
    gallery_group.add_argument('--independent-axes', action='store_true',
                               help='Gallery: each pane its own axis ranges (default: shared; the energy '
                                    'window of band/DOS panes is always --e-lower/--e-upper)')
    gallery_group.add_argument('--caption-csv', metavar='PATH',
                               help='Gallery: also write the caption table (material, formula, gap, energy) as CSV')
    gallery_group.add_argument('--max', type=int, default=None,
                               help='Gallery: at most this many panes (default: 200)')
    gallery_group.add_argument('--db', metavar='PATH',
                               help='Gallery: the materials.db for the captions (default: the one beside the files)')

    # Band options
    band_group = parser.add_argument_group('band structure options')
    band_group.add_argument(
        '--e-lower',
        type=float,
        default=-6.0,
        help='Lower energy limit in eV, from the reference (default: -6, as the GUI)'
    )
    band_group.add_argument(
        '--e-upper',
        type=float,
        default=6.0,
        help='Upper energy limit in eV, from the reference (default: 6, as the GUI)'
    )
    band_group.add_argument(
        '--energy-reference',
        choices=['auto', 'fermi', 'vacuum'],
        default='auto',
        help="Band/DOS energy zero: auto (default) is the vacuum scale when the run's POTC files "
             '(<material>_POTC.POTC.DAT and _POTC.out) are beside the data, else the Fermi level; '
             'fermi: E - E_F even then; vacuum: the vacuum scale (needs the POTC files)'
    )
    band_group.add_argument(
        '--fermi-shift',
        type=float,
        default=0.0,
        metavar='EV',
        help='Band/DOS: move the energies by -EV, so the energy EV above E_F is the reference '
             '(the dotted line; the GUI\'s Fermi shift); the window stays around it (default: 0)'
    )
    band_group.add_argument(
        '--alpha',
        type=float,
        default=1.0,
        help='Band line opacity; below 1 overlapping bands stack darker (default: 1, opaque)'
    )
    band_group.add_argument(
        '--no-gaps',
        action='store_true',
        help='Remove gaps at k-path discontinuities'
    )
    band_group.add_argument(
        '--band-path',
        metavar='FILE',
        help='Plot along an edited k-path: a JSON band_path (as mace gui and opt2d3 write it)'
    )

    # DOS options
    dos_group = parser.add_argument_group('DOS options')
    dos_group.add_argument(
        '--projection',
        choices=['both', 'total', 'orbital', 'tm_orb'],
        default=None,
        help='Projection type of --backend matplotlib (default: the projections of --dos-colors, as the GUI)'
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

    # Element colors (structures, FREQ modes, cube atom overlays), shared with the GUI
    struct_group.add_argument(
        '--palette',
        metavar='NAME',
        help='Element colors for atoms (structure, FREQ modes, cube overlays): a preset '
             '(vesta, jmol, cvd, legacy) or a palette saved in the GUI (View > Element Colors). '
             'Default: the default palette saved there, else VESTA (as the GUI). '
             'legacy: the colors mace plotting used before 1.1.6'
    )

    # Element legend options (mace/utils/element_legend.py, shared with the GUI)
    lg = parser.add_argument_group('element legend options (--legend)')
    lg.add_argument('--legend-layout', choices=['vertical', 'horizontal', 'grid'], default='vertical',
                    help='One column, one row, or a grid (default: vertical)')
    lg.add_argument('--legend-columns', type=int, default=4, metavar='N', help='Grid: columns (default: 4)')
    lg.add_argument('--legend-shape', choices=['circle', 'square'], default='circle', help='Swatch shape (default: circle)')
    lg.add_argument('--legend-swatch', type=float, default=18, metavar='PX', help='Swatch size in px, 1/96 in (default: 18)')
    lg.add_argument('--legend-font-size', type=float, default=14, metavar='PX', help='Font size in px (default: 14)')
    lg.add_argument('--legend-label', choices=['symbol', 'name', 'both'], default='symbol',
                    help='Symbol, full name or both (default: symbol)')
    lg.add_argument('--legend-text', default='dark', metavar='COLOR',
                    help='Text color: dark (for light backgrounds, default), light (for dark ones) or #rrggbb')
    lg.add_argument('--legend-background', default='none', metavar='COLOR',
                    help='none (transparent, default), white, black or #rrggbb')
    lg.add_argument('--legend-order', choices=['z', 'first', 'alpha'], default='z',
                    help='Atomic number (default), first appearance or alphabetical')
    lg.add_argument('--legend-no-outline', action='store_true', help='No thin ring around the swatches')
    lg.add_argument('--legend-recursive', action='store_true', help='Folders: include their subfolders')

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
    cube_group.add_argument('--clip', type=float, default=99.0,
                            help='Cube slices: the color range runs from the (100 - CLIP)th to the CLIPth percentile, '
                                 'symmetric about 0 for spin and potential, so nuclear spikes do not take the scale (default: 99)')
    cube_group.add_argument('--color-range', metavar='LO,HI', default=None,
                            help='Cube slices: the color range outright (log10 units for a log-scale density)')
    cube_group.add_argument('--engine-args',
                            help='Advanced: extra cube-engine flags (shlex-split; '
                                 'output-controlling flags are screened out)')

    # Vibrational mode options ( --all-modes is deliberately NOT in mode_group,
    # so it never clashes with the top-level --all; see H5 ).
    orb_group = parser.add_argument_group('orbital options (--orbitals; the cube options above apply too)')
    orb_group.add_argument('--mo', metavar='LIST',
                           help='Orbitals: comma-separated HOMO, LUMO, HOMO-1, LUMO+2 or MO numbers '
                                '(default: HOMO,LUMO)')
    orb_group.add_argument('--k', type=int, metavar='N',
                           help='Orbitals: k point number from the ORBITALS .out list (default: the '
                                'valence-band top, else Γ)')
    orb_group.add_argument('--orb-spin', choices=['alpha', 'beta'],
                           help='Orbitals: spin channel of an open-shell run (default: alpha)')
    orb_group.add_argument('--part', choices=['re', 'im', 'abs2'],
                           help='Orbitals: Re ψ (default), Im ψ (complex k only) or |ψ|²')
    orb_group.add_argument('--degenerate-sum', action='store_true',
                           help='Orbitals: sum |ψ|² over each degenerate set (gauge-free)')
    orb_group.add_argument('--cells', metavar='NxMxL',
                           help='Orbitals: cells to draw along the periodic axes (default: 3 for '
                                'polymers, 2x2 for slabs, 1x1x1 for crystals)')
    orb_group.add_argument('--quality', choices=['preview', 'normal', 'fine'],
                           help='Orbitals: grid spacing 0.40 / 0.25 / 0.15 bohr (default: normal)')
    orb_group.add_argument('--list-orbitals', action='store_true',
                           help='Orbitals: print the k points and levels, draw nothing')

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
        Exit code: 0 for success; 1 if a requested kind has no files or any requested figure
        was not drawn (each plotter goes on to the next file, outcome records the failures).
    """
    outcome.reset()
    rc = _main(argv)
    bad = outcome.failures()
    if bad:
        print(f"\n  {len(bad)} figure(s) not drawn:", file=sys.stderr)
        for name, why in bad:
            print(f"    {name}: {why}" if why else f"    {name}", file=sys.stderr)
        return rc or 1
    return rc


def _main(argv: Optional[List[str]] = None) -> int:
    parser = create_parser()

    if argv is None:
        argv = sys.argv[1:]

    # a project folder's mace-folder.toml [plots] (the energy window) is the
    # default here; a flag on the command line still wins
    folder = _folder_defaults(parser, argv)

    args = parser.parse_args(argv)
    # -o given: band/DOS figures go there (B8); without it they go beside the data, as before
    args.output_given = _fs._given(argv, '-o', '--output') or any(a.startswith('-o') and len(a) > 2 for a in argv)
    if getattr(args, 'band', False) or getattr(args, 'dos', False) or getattr(args, 'band_dos', False):
        # the energy window runs from a lower to a higher energy, as the GUI requires
        # (equal limits drew a zero-height plot; inverted ones were swapped silently)
        if not args.e_lower < args.e_upper:
            parser.error(f"--e-lower ({args.e_lower:g}) must be below --e-upper ({args.e_upper:g})")
        # -o is the folder the figures go in (named <material>.BANDS.<format>, ...): a figure
        # file name made a folder of that name
        out = Path(args.output or '.')
        if args.output_given and out.suffix.lower() in ('.svg', '.png', '.pdf', '.html') and not out.is_dir():
            parser.error(f"-o {args.output}: for --band/--dos/--band-dos, -o names the folder the figures are "
                         f"written in (as <material>.BANDS.<format>); pass a folder, and --format for the format")
    if folder:
        used = [f"--{k.replace('_', '-')} {v:g}" if isinstance(v, (int, float)) else f"--{k.replace('_', '-')} {v}"
                for k, v in folder[1].items() if getattr(args, k, None) == v
                and not any(a == f"--{k.replace('_', '-')}" or a.startswith(f"--{k.replace('_', '-')}=") for a in argv)]
        if used:
            _fs.note(f"using {' '.join(used)} from {folder[0]} (pass the option to use another value)")
    files_pos = list(getattr(args, 'files', []) or [])

    # the element palette for every atom drawn in this run (one loader with the GUI)
    from mace.utils import element_palettes
    try:
        element_palettes.choose(getattr(args, 'palette', None))
    except ValueError as e:
        parser.error(str(e))

    if getattr(args, 'legend', False):
        from .legend import run_legend
        return run_legend(args, files_pos, parser)
    if args.format in ('tsv', 'csv', 'json'):
        parser.error(f'--format {args.format} is for --legend only')

    # H4: --diff owns exactly its two operands; extra positionals are an error.
    if getattr(args, 'diff', None) and files_pos:
        parser.error('--diff takes exactly two cube files; '
                     'remove the extra positional argument(s)')

    if getattr(args, 'gallery', None):
        from . import gallery
        try:
            return gallery.run_cli(args)
        except gallery.GalleryError as exc:
            print(f"❌ {exc}", file=sys.stderr)
            return 2

    # A mode flag pins the kind (mutually exclusive, so at most one matches);
    # cube sub-flags (--density/--esp/--spin/--diff) imply --cube.
    pinned = next((e for e in entries() if getattr(args, e.flag, False)), None)
    if pinned is None and _cube_implied(args):
        pinned = get(PlotKind.CUBE)

    if pinned is not None:
        return _dispatch_pinned(pinned, args, files_pos)

    if getattr(args, 'bz', False):
        from .handlers import bz
        return bz.main(args)

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
                config['fermi_shift'] = args.fermi_shift
                config['energy_reference'] = args.energy_reference
                from .handlers.legacy import _size_config, _style_config
                config.update(_style_config(args))
                config.update(_size_config(args))
            print(entry.progress_tmpl.format(n=len(files)))
            entry.handler(files, config, _output_for(entry, args))
        return 0

    # Explicit files with no mode flag: classify each and dispatch by kind.
    if files_pos:
        return _dispatch_positional(args, files_pos)

    # Interactive mode (default)
    run_interactive(args.directory)
    return 0


def _output_for(entry, args):
    """The handler's output directory: band and DOS figures go beside the data unless -o is given."""
    if entry.kind in (PlotKind.BAND, PlotKind.DOS) and not getattr(args, 'output_given', True):
        return None
    return args.output


def _folder_defaults(parser, argv):
    """(the folder file, the defaults it sets) after setting them on
    ``parser``; None without a folder file (nothing changes)."""
    fs = _fs.load()
    _fs.warn_once(fs)
    defaults = _fs.plot_defaults(fs)
    if not defaults:
        return None
    parser.set_defaults(**defaults)
    return fs.path, defaults


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
    entry.handler(files, config, _output_for(entry, args))
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
        entry.handler(files, config, _output_for(entry, args))
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
        entry.handler(files, config, _output_for(entry, args))
    return 0


if __name__ == '__main__':
    sys.exit(main())
