"""
MACE Plotting Module - Main Entry Point

Provides interactive and command-line interfaces for plotting CRYSTAL outputs.

Author: Marcus Djokic
Institution: Michigan State University, Mendoza Group
"""

import os
import sys
import glob
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any


# =============================================================================
# Interactive Prompts (matching MACE style)
# =============================================================================

def yes_no_prompt(prompt: str, default: str = "yes") -> bool:
    """Get yes/no response from user with validation."""
    default_char = "Y/n" if default.lower() == "yes" else "y/N"

    while True:
        response = input(f"{prompt} [{default_char}]: ").strip().lower()

        if not response:
            return default.lower() == "yes"

        if response in ["y", "yes", "true", "1"]:
            return True
        elif response in ["n", "no", "false", "0"]:
            return False
        else:
            print(f"  Invalid response '{response}'. Please enter yes/no (y/n).")


def select_option(prompt: str, options: List[str], default: int = 1) -> int:
    """Display numbered options and get user selection."""
    print(f"\n{prompt}")
    for i, option in enumerate(options, 1):
        marker = " *" if i == default else ""
        print(f"  {i}. {option}{marker}")

    while True:
        response = input(f"\nSelect option [1-{len(options)}] (default: {default}): ").strip()

        if not response:
            return default

        try:
            choice = int(response)
            if 1 <= choice <= len(options):
                return choice
        except ValueError:
            pass

        print(f"  Invalid selection. Please enter 1-{len(options)}.")


def get_float_input(prompt: str, default: float) -> float:
    """Get a float value from user with default."""
    while True:
        response = input(f"{prompt} [{default}]: ").strip()

        if not response:
            return default

        try:
            return float(response)
        except ValueError:
            print(f"  Invalid number. Please enter a numeric value.")


def get_string_input(prompt: str, default: str = "") -> str:
    """Get a string value from user with optional default."""
    if default:
        response = input(f"{prompt} [{default}]: ").strip()
        return response if response else default
    else:
        return input(f"{prompt}: ").strip()


# =============================================================================
# File Discovery
# =============================================================================

def discover_plottable_files(directory: str = ".") -> Dict[str, List[str]]:
    """
    Discover files that can be plotted in the given directory.

    Returns:
        Dictionary with keys 'band', 'dos', 'cif' mapping to lists of files.
    """
    directory = Path(directory)

    results = {
        'band': [],
        'dos': [],
        'cif': [],
    }

    # Band structure files (*.band.band.dat or *.BAND.DAT)
    band_patterns = ['*.band.band.dat', '*.BAND.DAT', '*_band.dat']
    for pattern in band_patterns:
        results['band'].extend(glob.glob(str(directory / pattern)))

    # DOS files (*_doss.DOSS.DAT or *.DOSS.DAT)
    dos_patterns = ['*_doss.DOSS.DAT', '*.DOSS.DAT', '*_doss.dat']
    for pattern in dos_patterns:
        results['dos'].extend(glob.glob(str(directory / pattern)))

    # CIF files
    results['cif'] = glob.glob(str(directory / '*.cif'))

    # Remove duplicates and sort
    for key in results:
        results[key] = sorted(list(set(results[key])))

    return results


def print_discovered_files(files: Dict[str, List[str]]) -> None:
    """Print a summary of discovered plottable files."""
    print("\n" + "=" * 60)
    print("  DISCOVERED FILES")
    print("=" * 60)

    total = sum(len(v) for v in files.values())
    if total == 0:
        print("  No plottable files found in current directory.")
        print("\n  Supported file types:")
        print("    - Band structure: *.band.band.dat, *.BAND.DAT")
        print("    - DOS: *_doss.DOSS.DAT, *.DOSS.DAT")
        print("    - Structures: *.cif")
        return

    if files['band']:
        print(f"\n  Band Structure Files ({len(files['band'])}):")
        for f in files['band'][:5]:
            print(f"    - {Path(f).name}")
        if len(files['band']) > 5:
            print(f"    ... and {len(files['band']) - 5} more")

    if files['dos']:
        print(f"\n  DOS Files ({len(files['dos'])}):")
        for f in files['dos'][:5]:
            print(f"    - {Path(f).name}")
        if len(files['dos']) > 5:
            print(f"    ... and {len(files['dos']) - 5} more")

    if files['cif']:
        print(f"\n  CIF Structure Files ({len(files['cif'])}):")
        for f in files['cif'][:5]:
            print(f"    - {Path(f).name}")
        if len(files['cif']) > 5:
            print(f"    ... and {len(files['cif']) - 5} more")

    print()


# =============================================================================
# Band Structure Plotting
# =============================================================================

def configure_output_formats(interactive: bool = True, default_formats: List[str] = None) -> List[str]:
    """
    Configure output file formats.

    Args:
        interactive: Whether to prompt user
        default_formats: Default formats if not interactive

    Returns:
        List of format strings (e.g., ['svg', 'png'])
    """
    if default_formats is None:
        default_formats = ['png', 'svg']

    if not interactive:
        return default_formats

    print("\n  Output format options:")
    print("    1. PNG + SVG (default)")
    print("    2. SVG only (scalable)")
    print("    3. PNG only (600 DPI)")
    print("    4. PDF only (vector)")
    print("    5. All formats (PNG + SVG + PDF)")

    choice = get_string_input("  Select format(s) [1-5]", "1")

    format_map = {
        '1': ['png', 'svg'],
        '2': ['svg'],
        '3': ['png'],
        '4': ['pdf'],
        '5': ['png', 'svg', 'pdf'],
    }

    return format_map.get(choice, ['png', 'svg'])


def configure_band_plot(interactive: bool = True) -> Dict[str, Any]:
    """
    Configure band structure plotting parameters.

    Returns:
        Dictionary of configuration options.
    """
    config = {
        'e_lower': -2.0,
        'e_upper': 5.0,
        'alpha': 0.3,
        'segments': None,
        'auto_width': True,
        'gap_width': 0.05,
        'no_gaps': True,
        'spin_up_color': '#fa26a0',
        'spin_down_color': '#2ff3e0',
        'formats': ['png', 'svg'],
    }

    if not interactive:
        return config

    print("\n" + "-" * 40)
    print("  BAND STRUCTURE CONFIGURATION")
    print("-" * 40)

    # Energy range
    if yes_no_prompt("Use default energy range (-2 to +5 eV)?", "yes"):
        pass
    else:
        config['e_lower'] = get_float_input("  Lower energy limit (eV)", -2.0)
        config['e_upper'] = get_float_input("  Upper energy limit (eV)", 5.0)

    # Output formats
    config['formats'] = configure_output_formats(interactive=True)

    # Advanced options
    if yes_no_prompt("Configure advanced options?", "no"):
        config['alpha'] = get_float_input("  Transparency (0=opaque, 1=transparent)", 0.3)
        config['auto_width'] = yes_no_prompt("  Auto-adjust figure width?", "yes")
        config['no_gaps'] = yes_no_prompt("  Hide gaps at path discontinuities?", "yes")

        if yes_no_prompt("  Customize spin colors?", "no"):
            config['spin_up_color'] = get_string_input("    Spin-up color (hex)", "#fa26a0")
            config['spin_down_color'] = get_string_input("    Spin-down color (hex)", "#2ff3e0")

    return config


def plot_bands(files: List[str], config: Dict[str, Any], output_dir: str = ".") -> List[str]:
    """
    Plot band structures for the given files.

    Returns:
        List of generated output files.
    """
    # Import the band plotting module
    plotting_dir = Path(__file__).parent.parent.parent / "Plotting"
    if str(plotting_dir) not in sys.path:
        sys.path.insert(0, str(plotting_dir))

    try:
        import ipBANDS_V2 as band_plotter
    except ImportError as e:
        print("  Error: Could not import band plotting module.")
        print(f"  Expected location: {plotting_dir / 'ipBANDS_V2.py'}")
        print(f"  Import error: {e}")
        import traceback
        traceback.print_exc()
        return []

    output_files = []
    original_dir = os.getcwd()

    for band_file in files:
        band_path = Path(band_file)
        work_dir = band_path.parent

        print(f"\n  Plotting: {band_path.name}")

        argv_backup = sys.argv
        try:
            os.chdir(work_dir)

            # Build command-line arguments for the band plotter
            sys.argv = [
                'ipBANDS_V2.py',
                str(config['e_lower']),
                str(config['e_upper']),
            ]

            # Always forward alpha: the config default (0.3) differs from the
            # plotter's own default (1.0), so gating on != 0.3 silently dropped
            # the configured transparency and rendered opaque.
            sys.argv.extend(['--alpha', str(config['alpha'])])
            if config['auto_width']:
                sys.argv.append('--auto-width')
            if config['no_gaps']:
                sys.argv.append('--no-gaps')
            if config['gap_width'] != 0.05:
                sys.argv.extend(['--gap-width', str(config['gap_width'])])
            if config['spin_up_color'] != '#fa26a0':
                sys.argv.extend(['--spin-up-color', config['spin_up_color']])
            if config['spin_down_color'] != '#2ff3e0':
                sys.argv.extend(['--spin-down-color', config['spin_down_color']])
            if config.get('formats'):
                sys.argv.extend(['--formats', ','.join(config['formats'])])

            # Run the plotter
            band_plotter.main()

            # Find generated files
            base_name = band_path.name.replace('.band.band.dat', '').replace('.BAND.DAT', '')
            formats = config.get('formats', ['svg'])
            for fmt in formats:
                out_file = work_dir / f"{base_name}.BANDS.{fmt}"
                if out_file.exists():
                    output_files.append(str(out_file))
                    print(f"    Generated: {out_file.name}")

        except (Exception, SystemExit) as e:
            # The legacy plotters call sys.exit() on bad input; SystemExit is not
            # an Exception, so without catching it one bad file aborted the whole
            # batch and the mace process.
            print(f"    Error: {e}")
        finally:
            sys.argv = argv_backup
            os.chdir(original_dir)

    return output_files


# =============================================================================
# DOS Plotting
# =============================================================================

def configure_dos_plot(interactive: bool = True) -> Dict[str, Any]:
    """
    Configure DOS plotting parameters.

    Returns:
        Dictionary of configuration options.
    """
    config = {
        'e_lower': -2.0,
        'e_upper': 5.0,
        'projection_type': 'orbital',
        'x_scale': 'auto',
        'element_mode': 'all',
        'composition': True,
        'vb_range': 0.5,
        'cb_range': 0.5,
        'formats': ['png', 'svg'],
    }

    if not interactive:
        return config

    print("\n" + "-" * 40)
    print("  DOS CONFIGURATION")
    print("-" * 40)

    # Energy range
    if yes_no_prompt("Use default energy range (-2 to +5 eV)?", "yes"):
        pass
    else:
        config['e_lower'] = get_float_input("  Lower energy limit (eV)", -2.0)
        config['e_upper'] = get_float_input("  Upper energy limit (eV)", 5.0)

    # Projection type
    proj_choice = select_option(
        "Select projection type:",
        [
            "orbital - Show orbital projections (s, p, d, f)",
            "both - Show total and orbital projections",
            "total - Show only total DOS by element",
            "tm_orb - Transition metals orbital, others total",
        ],
        default=1
    )
    config['projection_type'] = ['orbital', 'both', 'total', 'tm_orb'][proj_choice - 1]

    # Output formats
    config['formats'] = configure_output_formats(interactive=True)

    # Advanced options
    if yes_no_prompt("Configure advanced options?", "no"):
        config['composition'] = yes_no_prompt("  Show VB/CB composition labels?", "yes")

        if config['composition']:
            config['vb_range'] = get_float_input("  VB analysis range (eV below VBM)", 0.5)
            config['cb_range'] = get_float_input("  CB analysis range (eV above CBM)", 0.5)

        scale_choice = select_option(
            "  X-axis scaling mode:",
            [
                "auto - Full range automatic",
                "upper_half - Scale based on conduction band",
                "lower_half - Scale based on valence band",
                "fermi:2 - Focus around Fermi level",
            ],
            default=1
        )
        config['x_scale'] = ['auto', 'upper_half', 'lower_half', 'fermi:2'][scale_choice - 1]

    return config


def plot_dos(files: List[str], config: Dict[str, Any], output_dir: str = ".") -> List[str]:
    """
    Plot DOS for the given files.

    Returns:
        List of generated output files.
    """
    # 'tm_orb' (transition-metal orbital, others total) is an ipDOS *element_mode*,
    # not a proj_type. Both the interactive menu and the CLI expose it as a
    # projection choice, so map it back here; otherwise ipDOS_V2 rejects
    # proj_type='tm_orb' with sys.exit(1). Copy the config so we don't mutate the
    # caller's dict.
    if config.get('projection_type') == 'tm_orb':
        config = {**config, 'projection_type': 'total', 'element_mode': 'tm_orb'}

    # Import the DOS plotting module
    plotting_dir = Path(__file__).parent.parent.parent / "Plotting"
    if str(plotting_dir) not in sys.path:
        sys.path.insert(0, str(plotting_dir))

    try:
        import ipDOS_V2 as dos_plotter
    except ImportError:
        print("  Error: Could not import DOS plotting module.")
        print(f"  Expected location: {plotting_dir / 'ipDOS_V2.py'}")
        return []

    output_files = []
    original_dir = os.getcwd()

    for dos_file in files:
        dos_path = Path(dos_file)
        work_dir = dos_path.parent

        print(f"\n  Plotting: {dos_path.name}")

        argv_backup = sys.argv
        try:
            os.chdir(work_dir)

            # Build command-line arguments for the DOS plotter
            sys.argv = [
                'ipDOS_V2.py',
                str(config['e_lower']),
                str(config['e_upper']),
                config['projection_type'],
                config['x_scale'],
                config['element_mode'],
            ]

            if not config['composition']:
                sys.argv.append('--no-composition')
            if config['vb_range'] != 0.5:
                sys.argv.extend(['--vb-range', str(config['vb_range'])])
            if config['cb_range'] != 0.5:
                sys.argv.extend(['--cb-range', str(config['cb_range'])])
            if config.get('formats'):
                sys.argv.extend(['--formats', ','.join(config['formats'])])

            # Run the plotter
            dos_plotter.main()

            # Find generated files
            base_name = dos_path.name.replace('_doss.DOSS.DAT', '').replace('.DOSS.DAT', '')
            formats = config.get('formats', ['svg'])
            for fmt in formats:
                out_file = work_dir / f"{base_name}.DOSS.{fmt}"
                if out_file.exists():
                    output_files.append(str(out_file))
                    print(f"    Generated: {out_file.name}")

        except (Exception, SystemExit) as e:
            # The legacy plotters call sys.exit() on bad input; SystemExit is not
            # an Exception, so without catching it one bad file aborted the whole
            # batch and the mace process.
            print(f"    Error: {e}")
        finally:
            sys.argv = argv_backup
            os.chdir(original_dir)

    return output_files


# =============================================================================
# Structure Visualization
# =============================================================================

def configure_structure_plot(interactive: bool = True) -> Dict[str, Any]:
    """
    Configure structure visualization parameters.

    Returns:
        Dictionary of configuration options.
    """
    config = {
        'supercell': (2, 2, 2),
        'canvas_size': 800,
        'bond_cutoff': 1.9,
        'color_by_coord': True,
        'parallel_jobs': max(1, os.cpu_count() - 1),
    }

    if not interactive:
        return config

    print("\n" + "-" * 40)
    print("  STRUCTURE VISUALIZATION CONFIGURATION")
    print("-" * 40)

    # Supercell
    if yes_no_prompt("Use default supercell (2x2x2)?", "yes"):
        pass
    else:
        nx = int(get_float_input("  Supercell X dimension", 2))
        ny = int(get_float_input("  Supercell Y dimension", 2))
        nz = int(get_float_input("  Supercell Z dimension", 2))
        config['supercell'] = (nx, ny, nz)

    # Advanced options
    if yes_no_prompt("Configure advanced options?", "no"):
        config['canvas_size'] = int(get_float_input("  Canvas size (pixels)", 800))
        config['bond_cutoff'] = get_float_input("  Bond cutoff (Angstroms)", 1.9)
        config['color_by_coord'] = yes_no_prompt("  Color atoms by coordination number?", "yes")

    return config


def plot_structures(files: List[str], config: Dict[str, Any], output_dir: str = ".") -> List[str]:
    """
    Generate structure visualizations for the given CIF files.

    Returns:
        List of generated output files.
    """
    # Import the structure plotting module
    plotting_dir = Path(__file__).parent.parent.parent / "Plotting"
    if str(plotting_dir) not in sys.path:
        sys.path.insert(0, str(plotting_dir))

    try:
        import plottingCIFs as struct_plotter
    except ImportError:
        print("  Error: Could not import structure plotting module.")
        print(f"  Expected location: {plotting_dir / 'plottingCIFs.py'}")
        return []

    output_files = []
    output_path = Path(output_dir)

    # Create output directory if needed
    svg_dir = output_path / "structure_svgs"
    svg_dir.mkdir(exist_ok=True)

    print(f"\n  Processing {len(files)} CIF file(s)...")
    print(f"  Output directory: {svg_dir}")

    try:
        # Use the module's processing function
        for cif_file in files:
            cif_path = Path(cif_file)
            print(f"\n  Processing: {cif_path.name}")

            try:
                # Process the CIF file
                result = struct_plotter.process_cif(
                    str(cif_path),
                    str(svg_dir),
                    config['supercell'],
                    config['canvas_size'],
                    config['color_by_coord'],
                    config['bond_cutoff']
                )

                if result:
                    for view_file in result:
                        output_files.append(view_file)
                        print(f"    Generated: {Path(view_file).name}")

            except Exception as e:
                print(f"    Error: {e}")

    except Exception as e:
        print(f"  Error during structure processing: {e}")

    return output_files


# =============================================================================
# Main Interactive Interface
# =============================================================================

def run_interactive(directory: str = ".") -> None:
    """
    Run the interactive plotting interface.

    Args:
        directory: Working directory to search for plottable files.
    """
    print("\n" + "=" * 60)
    print("  MACE PLOTTING")
    print("=" * 60)
    print("  Interactive plotting for CRYSTAL calculation outputs")

    # Discover files
    files = discover_plottable_files(directory)
    print_discovered_files(files)

    total_files = sum(len(v) for v in files.values())
    if total_files == 0:
        return

    # Build options based on available files
    options = []
    option_map = {}

    if files['band']:
        options.append(f"Plot band structures ({len(files['band'])} files)")
        option_map[len(options)] = 'band'

    if files['dos']:
        options.append(f"Plot DOS ({len(files['dos'])} files)")
        option_map[len(options)] = 'dos'

    if files['cif']:
        options.append(f"Visualize structures ({len(files['cif'])} files)")
        option_map[len(options)] = 'cif'

    if len(option_map) > 1:
        options.append("Plot all available")
        option_map[len(options)] = 'all'

    options.append("Exit")
    option_map[len(options)] = 'exit'

    # Main selection loop
    while True:
        choice = select_option("What would you like to plot?", options, default=1)
        action = option_map[choice]

        if action == 'exit':
            print("\n  Exiting MACE plotting.")
            return

        if action == 'all':
            # Plot everything
            if files['band']:
                config = configure_band_plot(interactive=True)
                plot_bands(files['band'], config, directory)

            if files['dos']:
                config = configure_dos_plot(interactive=True)
                plot_dos(files['dos'], config, directory)

            if files['cif']:
                config = configure_structure_plot(interactive=True)
                plot_structures(files['cif'], config, directory)

        elif action == 'band':
            config = configure_band_plot(interactive=True)
            plot_bands(files['band'], config, directory)

        elif action == 'dos':
            config = configure_dos_plot(interactive=True)
            plot_dos(files['dos'], config, directory)

        elif action == 'cif':
            config = configure_structure_plot(interactive=True)
            plot_structures(files['cif'], config, directory)

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
    """
    Main entry point for mace plotting.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:])

    Returns:
        Exit code (0 for success)
    """
    parser = create_parser()

    if argv is None:
        argv = sys.argv[1:]

    args = parser.parse_args(argv)

    # Determine mode
    if args.band:
        # Band plotting with CLI options
        files = discover_plottable_files(args.directory)
        if not files['band']:
            print("  No band structure files found.")
            return 1

        config = {
            'e_lower': args.e_lower,
            'e_upper': args.e_upper,
            'alpha': args.alpha,
            'no_gaps': args.no_gaps,
            'auto_width': True,
            'gap_width': 0.05,
            'spin_up_color': '#fa26a0',
            'spin_down_color': '#2ff3e0',
            'segments': None,
        }

        print(f"\n  Plotting {len(files['band'])} band structure(s)...")
        plot_bands(files['band'], config, args.output)

    elif args.dos:
        # DOS plotting with CLI options
        files = discover_plottable_files(args.directory)
        if not files['dos']:
            print("  No DOS files found.")
            return 1

        config = {
            'e_lower': args.e_lower,
            'e_upper': args.e_upper,
            'projection_type': args.projection,
            'x_scale': 'auto',
            'element_mode': 'all',
            'composition': not args.no_composition,
            'vb_range': 0.5,
            'cb_range': 0.5,
        }

        print(f"\n  Plotting {len(files['dos'])} DOS file(s)...")
        plot_dos(files['dos'], config, args.output)

    elif args.structure:
        # Structure visualization with CLI options
        files = discover_plottable_files(args.directory)
        if not files['cif']:
            print("  No CIF files found.")
            return 1

        config = {
            'supercell': tuple(args.supercell),
            'canvas_size': 800,
            'bond_cutoff': args.bond_cutoff,
            'color_by_coord': args.color_by_coord,
            'parallel_jobs': max(1, os.cpu_count() - 1),
        }

        print(f"\n  Visualizing {len(files['cif'])} structure(s)...")
        plot_structures(files['cif'], config, args.output)

    elif args.all:
        # Plot everything with defaults
        files = discover_plottable_files(args.directory)

        if files['band']:
            config = configure_band_plot(interactive=False)
            config['e_lower'] = args.e_lower
            config['e_upper'] = args.e_upper
            print(f"\n  Plotting {len(files['band'])} band structure(s)...")
            plot_bands(files['band'], config, args.output)

        if files['dos']:
            config = configure_dos_plot(interactive=False)
            config['e_lower'] = args.e_lower
            config['e_upper'] = args.e_upper
            print(f"\n  Plotting {len(files['dos'])} DOS file(s)...")
            plot_dos(files['dos'], config, args.output)

        if files['cif']:
            config = configure_structure_plot(interactive=False)
            print(f"\n  Visualizing {len(files['cif'])} structure(s)...")
            plot_structures(files['cif'], config, args.output)

    else:
        # Interactive mode (default)
        run_interactive(args.directory)

    return 0


if __name__ == '__main__':
    sys.exit(main())
