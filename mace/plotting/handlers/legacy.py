"""Legacy plotters (band / DOS / structure) for mace plotting.

These ``configure_*`` / ``plot_*`` functions are moved verbatim from
``main.py`` (Phase 0 registry refactor). The ONLY change versus the originals
is the ``Plotting/`` directory depth — this module lives one package level
deeper than the old ``main.py``, so the relative path is recomputed. The
underlying engines (``ipBANDS_V2`` / ``ipDOS_V2`` / ``plottingCIFs``) are
untouched and still imported lazily inside each ``plot_*`` call.

At import time this module registers the three legacy plotters into the
registry, so ``import mace.plotting.handlers`` is all that's needed to make
band/DOS/structure discoverable and dispatchable.

Author: Marcus Djokic
Institution: Michigan State University, Mendoza Group
"""
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from ..prompts import (
    configure_output_formats,
    get_float_input,
    get_string_input,
    select_option,
    yes_no_prompt,
)
from ..registry import PlotKind, PlotterEntry, register

# Repo-root/Plotting (this file: mace/plotting/handlers/legacy.py -> parents[3] = repo root).
_PLOTTING_DIR = Path(__file__).resolve().parents[3] / "Plotting"


# =============================================================================
# Band Structure Plotting
# =============================================================================

def configure_band_plot(interactive: bool = True) -> Dict[str, Any]:
    """Configure band structure plotting parameters."""
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

    if yes_no_prompt("Use default energy range (-2 to +5 eV)?", "yes"):
        pass
    else:
        config['e_lower'] = get_float_input("  Lower energy limit (eV)", -2.0)
        config['e_upper'] = get_float_input("  Upper energy limit (eV)", 5.0)

    config['formats'] = configure_output_formats(interactive=True)

    if yes_no_prompt("Configure advanced options?", "no"):
        config['alpha'] = get_float_input("  Transparency (0=opaque, 1=transparent)", 0.3)
        config['auto_width'] = yes_no_prompt("  Auto-adjust figure width?", "yes")
        config['no_gaps'] = yes_no_prompt("  Hide gaps at path discontinuities?", "yes")

        if yes_no_prompt("  Customize spin colors?", "no"):
            config['spin_up_color'] = get_string_input("    Spin-up color (hex)", "#fa26a0")
            config['spin_down_color'] = get_string_input("    Spin-down color (hex)", "#2ff3e0")

    return config


def plot_bands(files: List[str], config: Dict[str, Any], output_dir: str = ".") -> List[str]:
    """Plot band structures for the given files."""
    plotting_dir = _PLOTTING_DIR
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

            band_plotter.main()

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


def _band_config_from_args(args) -> Dict[str, Any]:
    return {
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


# =============================================================================
# DOS Plotting
# =============================================================================

def configure_dos_plot(interactive: bool = True) -> Dict[str, Any]:
    """Configure DOS plotting parameters."""
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

    if yes_no_prompt("Use default energy range (-2 to +5 eV)?", "yes"):
        pass
    else:
        config['e_lower'] = get_float_input("  Lower energy limit (eV)", -2.0)
        config['e_upper'] = get_float_input("  Upper energy limit (eV)", 5.0)

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

    config['formats'] = configure_output_formats(interactive=True)

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
    """Plot DOS for the given files."""
    # 'tm_orb' (transition-metal orbital, others total) is an ipDOS *element_mode*,
    # not a proj_type. Both the interactive menu and the CLI expose it as a
    # projection choice, so map it back here; otherwise ipDOS_V2 rejects
    # proj_type='tm_orb' with sys.exit(1). Copy the config so we don't mutate the
    # caller's dict.
    if config.get('projection_type') == 'tm_orb':
        config = {**config, 'projection_type': 'total', 'element_mode': 'tm_orb'}

    plotting_dir = _PLOTTING_DIR
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

            dos_plotter.main()

            base_name = dos_path.name.replace('_doss.DOSS.DAT', '').replace('.DOSS.DAT', '')
            formats = config.get('formats', ['svg'])
            for fmt in formats:
                out_file = work_dir / f"{base_name}.DOSS.{fmt}"
                if out_file.exists():
                    output_files.append(str(out_file))
                    print(f"    Generated: {out_file.name}")

        except (Exception, SystemExit) as e:
            print(f"    Error: {e}")
        finally:
            sys.argv = argv_backup
            os.chdir(original_dir)

    return output_files


def _dos_config_from_args(args) -> Dict[str, Any]:
    return {
        'e_lower': args.e_lower,
        'e_upper': args.e_upper,
        'projection_type': args.projection,
        'x_scale': 'auto',
        'element_mode': 'all',
        'composition': not args.no_composition,
        'vb_range': 0.5,
        'cb_range': 0.5,
    }


# =============================================================================
# Structure Visualization
# =============================================================================

def configure_structure_plot(interactive: bool = True) -> Dict[str, Any]:
    """Configure structure visualization parameters."""
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

    if yes_no_prompt("Use default supercell (2x2x2)?", "yes"):
        pass
    else:
        nx = int(get_float_input("  Supercell X dimension", 2))
        ny = int(get_float_input("  Supercell Y dimension", 2))
        nz = int(get_float_input("  Supercell Z dimension", 2))
        config['supercell'] = (nx, ny, nz)

    if yes_no_prompt("Configure advanced options?", "no"):
        config['canvas_size'] = int(get_float_input("  Canvas size (pixels)", 800))
        config['bond_cutoff'] = get_float_input("  Bond cutoff (Angstroms)", 1.9)
        config['color_by_coord'] = yes_no_prompt("  Color atoms by coordination number?", "yes")

    return config


def plot_structures(files: List[str], config: Dict[str, Any], output_dir: str = ".") -> List[str]:
    """Generate structure visualizations for the given CIF files."""
    plotting_dir = _PLOTTING_DIR
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

    svg_dir = output_path / "structure_svgs"
    svg_dir.mkdir(exist_ok=True)

    print(f"\n  Processing {len(files)} CIF file(s)...")
    print(f"  Output directory: {svg_dir}")

    try:
        for cif_file in files:
            cif_path = Path(cif_file)
            print(f"\n  Processing: {cif_path.name}")

            try:
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


def _structure_config_from_args(args) -> Dict[str, Any]:
    return {
        'supercell': tuple(args.supercell),
        'canvas_size': 800,
        'bond_cutoff': args.bond_cutoff,
        'color_by_coord': args.color_by_coord,
        'parallel_jobs': max(1, os.cpu_count() - 1),
    }


# =============================================================================
# Registration
# =============================================================================

def register_legacy_plotters() -> None:
    """Register band / DOS / structure into the registry (idempotent)."""
    register(PlotterEntry(
        kind=PlotKind.BAND,
        flag='band',
        label='band structure',
        handler=plot_bands,
        configure=configure_band_plot,
        config_from_args=_band_config_from_args,
        patterns=['*.band.band.dat', '*.BAND.DAT', '*_band.dat'],
        accepts_energy_range=True,
        progress_tmpl="\n  Plotting {n} band structure(s)...",
        not_found_msg="  No band structure files found.",
        menu_tmpl="Plot band structures ({n} files)",
        discovery_label="Band Structure",
    ))
    register(PlotterEntry(
        kind=PlotKind.DOS,
        flag='dos',
        label='DOS',
        handler=plot_dos,
        configure=configure_dos_plot,
        config_from_args=_dos_config_from_args,
        patterns=['*_doss.DOSS.DAT', '*.DOSS.DAT', '*_doss.dat'],
        accepts_energy_range=True,
        progress_tmpl="\n  Plotting {n} DOS file(s)...",
        not_found_msg="  No DOS files found.",
        menu_tmpl="Plot DOS ({n} files)",
        discovery_label="DOS",
    ))
    register(PlotterEntry(
        kind=PlotKind.STRUCTURE,
        flag='structure',
        label='structure',
        handler=plot_structures,
        configure=configure_structure_plot,
        config_from_args=_structure_config_from_args,
        patterns=['*.cif'],
        accepts_energy_range=False,
        progress_tmpl="\n  Visualizing {n} structure(s)...",
        not_found_msg="  No CIF files found.",
        menu_tmpl="Visualize structures ({n} files)",
        discovery_label="CIF Structure",
    ))


register_legacy_plotters()
