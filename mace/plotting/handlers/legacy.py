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
from .. import outcome
from ..registry import PlotKind, PlotterEntry, register

# Repo-root/Plotting (this file: mace/plotting/handlers/legacy.py -> parents[3] = repo root).
_PLOTTING_DIR = Path(__file__).resolve().parents[3] / "Plotting"


# =============================================================================
# Band Structure Plotting
# =============================================================================

def _interactive_window(config: Dict[str, Any]):
    """The energy range the interactive questions offer: a project folder's
    mace-folder.toml [plots] energy_window when there is one (said once),
    else the built-in -6 to +6 eV (the CLI and GUI default); ``config`` takes it as its default."""
    from mace.utils import folder_settings
    fs = folder_settings.load()
    folder_settings.warn_once(fs)
    win = folder_settings.plot_defaults(fs)
    if win:
        config['e_lower'], config['e_upper'] = win['e_lower'], win['e_upper']
        folder_settings.note(f"energy range {win['e_lower']:g} to {win['e_upper']:+g} eV from {fs.path}")
    return config['e_lower'], config['e_upper']


def configure_band_plot(interactive: bool = True) -> Dict[str, Any]:
    """Configure band structure plotting parameters."""
    config = {
        'e_lower': -6.0,
        'e_upper': 6.0,
        # opaque lines; spin colors None: the plot style's pair (presets.py SPIN_COLORS by default)
        'alpha': 1.0,
        'segments': None,
        'auto_width': True,
        'gap_width': 0.05,
        'no_gaps': True,
        'spin_up_color': None,
        'spin_down_color': None,
        'formats': ['png', 'svg'],
        **_style_defaults(),
    }

    if not interactive:
        return config

    print("\n" + "-" * 40)
    print("  BAND STRUCTURE CONFIGURATION")
    print("-" * 40)

    lo, hi = _interactive_window(config)
    if yes_no_prompt(f"Use default energy range ({lo:g} to {hi:+g} eV)?", "yes"):
        pass
    else:
        config['e_lower'] = get_float_input("  Lower energy limit (eV)", lo)
        config['e_upper'] = get_float_input("  Upper energy limit (eV)", hi)

    config['formats'] = configure_output_formats(interactive=True)

    if yes_no_prompt("Configure advanced options?", "no"):
        config['alpha'] = get_float_input("  Opacity (1 = opaque; below 1 overlapping bands stack)", 1.0)
        config['auto_width'] = yes_no_prompt("  Auto-adjust figure width?", "yes")
        config['no_gaps'] = yes_no_prompt("  Hide gaps at path discontinuities?", "yes")

        if yes_no_prompt("  Customize spin colors?", "no"):
            from ..presets import spin_colors
            pair = spin_colors(config.get('style'))
            config['spin_up_color'] = get_string_input("    Spin-up color (hex)", pair['up'])
            config['spin_down_color'] = get_string_input("    Spin-down color (hex)", pair['down'])

    return config


def _style_defaults() -> Dict[str, Any]:
    """The plot style, DOS colors and spin styling (presets.STYLE_OPTIONS: --style, --spin-colors,
    --spin-down-style, --dos-colors, --dos-spin, --no-dos-total), the energy unit and the renderer:
    the built-in defaults (the flags override them; a folder's [plots] sets them as defaults)."""
    return {'style': 'default', 'dos_colors': None, 'dos_total': True, 'spin_dos': 'auto', 'spin_style': 'dash',
            'spin_colors': None, 'band_color': None, 'band_style': None, 'dos_smooth': None, 'units': None, 'backend': None}


def _style_argv(config: Dict[str, Any], dos: bool) -> List[str]:
    """--style/--spin-style/--band-color (and for a DOS --dos-colors/--no-total/--spin-dos) for the
    plotter, only when they are not the defaults."""
    out = []
    if config.get('band_color'):
        out += ['--band-color', config['band_color']]
    if config.get('style', 'default') != 'default':
        out += ['--style', config['style']]
    if config.get('spin_style', 'dash') != 'dash':
        out += ['--spin-style', config['spin_style']]
    if dos:
        if config.get('dos_colors'):
            out += ['--dos-colors', config['dos_colors']]
        if config.get('dos_total') is False:
            out.append('--no-total')
        if config.get('spin_dos', 'auto') != 'auto':
            out += ['--spin-dos', config['spin_dos']]
        from ..presets import smooth_sigma
        if smooth_sigma(config.get('dos_smooth')):
            out += ['--dos-smooth', repr(float(config['dos_smooth']))]
    return out


def _out_dir(data_file: Path, output_dir) -> Path:
    """Where a band/DOS figure goes: -o when given (B8), else beside the data (the plotter's folder)."""
    if output_dir in (None, ""):
        return data_file.parent
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def _backend(config: Dict[str, Any]) -> str:
    from ..bandos_cli import choose_backend
    return choose_backend(config.get('backend'))


def plot_bands(files: List[str], config: Dict[str, Any], output_dir: str = None) -> List[str]:
    """Plot band structures for the given files: into ``output_dir`` (mace plotting -o) when
    given, else beside each data file. Drawn by plotly as mace gui draws them (bandos_cli),
    or by ipBANDS_V2 (matplotlib: --backend matplotlib, or no plotly/kaleido here)."""
    if _backend(config) == 'plotly':
        from ..bandos_cli import plot
        return plot('band', files, config, output_dir)
    from ..bandos_cli import formats_of, spin_pair
    config = {**config, 'formats': formats_of(config, 'matplotlib')}
    pair = spin_pair(config)
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
        outcome.failed("band plotter", str(e))
        return []

    output_files = []
    original_dir = os.getcwd()

    for band_file in files:
        band_path = Path(band_file).resolve()
        work_dir = band_path.parent
        out_dir = _out_dir(band_path, output_dir)

        print(f"\n  Plotting: {band_path.name}")
        if _no_potc(band_path, config):
            continue

        argv_backup = sys.argv
        try:
            os.chdir(work_dir)

            sys.argv = [
                'ipBANDS_V2.py',
                str(config['e_lower']),
                str(config['e_upper']),
            ]

            # Always forward alpha: gating on a default once silently dropped the
            # configured transparency
            sys.argv.extend(['--alpha', str(config['alpha'])])
            # this file only (the plotter otherwise draws every BAND.DAT of the folder), into out_dir
            sys.argv.extend(['--file', band_path.name, '--output-dir', str(out_dir)])
            if config['auto_width'] and not config.get('width'):     # --width: that width
                sys.argv.append('--auto-width')
            if config['no_gaps']:
                sys.argv.append('--no-gaps')
            if config['gap_width'] != 0.05:
                sys.argv.extend(['--gap-width', str(config['gap_width'])])
            # the spin pair of the style and --spin-colors (custom colors over it), as the GUI's
            sys.argv.extend(['--spin-up-color', pair['up'], '--spin-down-color', pair['down']])
            sys.argv.extend(_style_argv(config, dos=False))
            sys.argv.extend(_size_argv(config, band=True))
            if config.get('formats'):
                sys.argv.extend(['--formats', ','.join(config['formats'])])
            if config.get('band_path'):
                sys.argv.extend(['--band-path', str(Path(config['band_path']).resolve())])
            sys.argv.extend(_reference_argv(config))

            band_plotter.main()

            from ..bandos_cli import band_stem
            base_name = band_stem(band_path)      # the name ipBANDS_V2 writes (fe_band.BAND.DAT -> fe)
            formats = config.get('formats', ['svg'])
            wrote = False
            for fmt in formats:
                out_file = out_dir / f"{base_name}.BANDS.{fmt}"
                if out_file.exists():
                    output_files.append(str(out_file))
                    print(f"    Generated: {out_file.name}")
                    wrote = True
            if not wrote:
                print("    Error: no band figure was written")
                outcome.failed(band_path.name, "no figure written")

        except (Exception, SystemExit) as e:
            # The legacy plotters call sys.exit() on bad input; SystemExit is not
            # an Exception, so without catching it one bad file aborted the whole
            # batch and the mace process.
            print(f"    Error: {e}")
            outcome.failed(band_path.name, str(e))
        finally:
            sys.argv = argv_backup
            os.chdir(original_dir)

    return output_files


def _no_potc(data_file: Path, config: Dict[str, Any]) -> bool:
    """--energy-reference vacuum without the run's POTC files beside the data:
    says so (the file is skipped, as a plotter error skips it)."""
    if config.get('energy_reference') != 'vacuum':
        return False
    import re
    material = re.sub(r'([._]band\.band\.dat|_doss\.DOSS\.DAT)$', '', data_file.name, flags=re.IGNORECASE)
    need = [data_file.parent / f"{material}_POTC.POTC.DAT", data_file.parent / f"{material}_POTC.out"]
    if all(p.is_file() for p in need):
        return False
    print(f"    Error: --energy-reference vacuum needs the run's POTC files beside the data "
          f"({need[0].name} and {need[1].name})")
    outcome.failed(data_file.name, "no POTC files for --energy-reference vacuum")
    return True


def _reference_argv(config: Dict[str, Any]) -> List[str]:
    """--fermi-shift / --energy-reference for the plotter, only when they are
    not the defaults (0 eV, auto): without them its command line is as before."""
    out = []
    if config.get('fermi_shift'):
        out += ['--fermi-shift', repr(float(config['fermi_shift']))]
    if config.get('energy_reference', 'auto') != 'auto':
        out += ['--energy-reference', str(config['energy_reference'])]
    return out


def _style_config(args) -> Dict[str, Any]:
    out = _style_defaults()
    for k in out:
        v = getattr(args, k, None)
        if v is not None:
            out[k] = v
    for k in ('spin_up_color', 'spin_down_color'):
        if getattr(args, k, None):
            out[k] = getattr(args, k)
    fmt = getattr(args, 'format', None)
    if fmt in ('png', 'svg', 'pdf', 'html'):
        out['formats'] = [fmt]
    return out


def _size_config(args) -> Dict[str, Any]:
    """--width/--height (inches) and --dpi of a band/DOS/band + DOS figure (None: the backend's own)."""
    out = {}
    for k in ('width', 'height', 'dpi'):
        v = getattr(args, k, None)
        if v is not None:
            if v <= 0:
                raise SystemExit(f"mace plotting: --{k} must be positive")
            out[k] = v
    return out


def _size_argv(config: Dict[str, Any], band: bool) -> List[str]:
    """The matplotlib plotters' --fixed-width/--fixed-height/--dpi."""
    out = []
    if config.get('width'):
        out += ['--fixed-width', str(config['width'])]
    if config.get('height'):
        out += ['--fixed-height', str(config['height'])]
    if config.get('dpi'):
        out += ['--dpi', str(int(config['dpi']))]
    return out


def _reference_config(args) -> Dict[str, Any]:
    return {'fermi_shift': float(getattr(args, 'fermi_shift', 0.0) or 0.0),
            'energy_reference': getattr(args, 'energy_reference', None) or 'auto'}


def _band_config_from_args(args) -> Dict[str, Any]:
    return {
        'e_lower': args.e_lower,
        'e_upper': args.e_upper,
        'alpha': args.alpha,
        'no_gaps': args.no_gaps,
        'auto_width': True,
        'gap_width': 0.05,
        'spin_up_color': getattr(args, 'spin_up_color', None),
        'spin_down_color': getattr(args, 'spin_down_color', None),
        'segments': None,
        **_style_config(args),
        # absolute: the plotter runs in the data file's directory
        'band_path': str(Path(args.band_path).resolve()) if getattr(args, 'band_path', None) else None,
        **_reference_config(args),
        **_size_config(args),
    }


# =============================================================================
# DOS Plotting
# =============================================================================

def configure_dos_plot(interactive: bool = True) -> Dict[str, Any]:
    """Configure DOS plotting parameters."""
    config = {
        'e_lower': -6.0,
        'e_upper': 6.0,
        'projection_type': 'auto',
        'x_scale': 'auto',
        'element_mode': 'all',
        'composition': True,
        'vb_range': 0.5,
        'cb_range': 0.5,
        'formats': ['png', 'svg'],
        **_style_defaults(),
    }

    if not interactive:
        return config

    print("\n" + "-" * 40)
    print("  DOS CONFIGURATION")
    print("-" * 40)

    lo, hi = _interactive_window(config)
    if yes_no_prompt(f"Use default energy range ({lo:g} to {hi:+g} eV)?", "yes"):
        pass
    else:
        config['e_lower'] = get_float_input("  Lower energy limit (eV)", lo)
        config['e_upper'] = get_float_input("  Upper energy limit (eV)", hi)

    mode = select_option(
        "DOS colors (which projections, as mace gui's DOS colors):",
        [
            "auto - one element by orbital, two or three element x orbital, more by element",
            "total - the total only",
            "element - each element",
            "orbital - each orbital (s, p, d, f)",
            "element-orbital - each element's orbitals",
        ],
        default=1
    )
    config['dos_colors'] = ['auto', 'total', 'element', 'orbital', 'element-orbital'][mode - 1]

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


def plot_dos(files: List[str], config: Dict[str, Any], output_dir: str = None) -> List[str]:
    """Plot DOS for the given files: into ``output_dir`` (mace plotting -o) when given, else
    beside each data file."""
    # 'tm_orb' (transition-metal orbital, others total) is an ipDOS *element_mode*,
    # not a proj_type. Both the interactive menu and the CLI expose it as a
    # projection choice, so map it back here; otherwise ipDOS_V2 rejects
    # proj_type='tm_orb' with sys.exit(1). Copy the config so we don't mutate the
    # caller's dict.
    if _backend(config) == 'plotly':
        from ..bandos_cli import plot
        return plot('dos', files, config, output_dir)
    from ..bandos_cli import formats_of, spin_pair
    if config.get('projection_type') in (None, 'auto'):
        # no --projection: the projections and colors of the GUI (--dos-colors, auto by default)
        config = {**config, 'projection_type': 'both', 'dos_colors': config.get('dos_colors') or 'auto'}
    config = {**config, 'formats': formats_of(config, 'matplotlib')}
    if config.get('projection_type') == 'tm_orb':
        config = {**config, 'projection_type': 'total', 'element_mode': 'tm_orb'}
    pair = spin_pair(config)

    plotting_dir = _PLOTTING_DIR
    if str(plotting_dir) not in sys.path:
        sys.path.insert(0, str(plotting_dir))

    try:
        import ipDOS_V2 as dos_plotter
    except ImportError:
        print("  Error: Could not import DOS plotting module.")
        print(f"  Expected location: {plotting_dir / 'ipDOS_V2.py'}")
        outcome.failed("DOS plotter", "could not import ipDOS_V2")
        return []

    output_files = []
    original_dir = os.getcwd()

    for dos_file in files:
        dos_path = Path(dos_file).resolve()
        work_dir = dos_path.parent
        out_dir = _out_dir(dos_path, output_dir)

        print(f"\n  Plotting: {dos_path.name}")
        if _no_potc(dos_path, config):
            continue

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
            sys.argv.extend(_reference_argv(config))
            sys.argv.extend(_style_argv(config, dos=True))
            sys.argv.extend(_size_argv(config, band=False))
            # the spin pair of the style and --spin-colors (custom colors over it): the total's two lines
            sys.argv.extend(['--spin-up-color', pair['up'], '--spin-down-color', pair['down']])
            # this file only (the plotter otherwise draws every DOSS.DAT of the folder), into out_dir
            sys.argv.extend(['--file', dos_path.name, '--output-dir', str(out_dir)])

            dos_plotter.main()

            base_name = dos_path.name.replace('_doss.DOSS.DAT', '').replace('.DOSS.DAT', '')
            formats = config.get('formats', ['svg'])
            wrote = False
            for fmt in formats:
                out_file = out_dir / f"{base_name}.DOSS.{fmt}"
                if out_file.exists():
                    output_files.append(str(out_file))
                    print(f"    Generated: {out_file.name}")
                    wrote = True
            if not wrote:
                print("    Error: no DOS figure was written")
                outcome.failed(dos_path.name, "no figure written")

        except (Exception, SystemExit) as e:
            print(f"    Error: {e}")
            outcome.failed(dos_path.name, str(e))
        finally:
            sys.argv = argv_backup
            os.chdir(original_dir)

    return output_files


def _dos_config_from_args(args) -> Dict[str, Any]:
    return {
        'e_lower': args.e_lower,
        'e_upper': args.e_upper,
        # no --projection: the GUI's projections (--dos-colors); matplotlib only
        'projection_type': args.projection or 'auto',
        'projection_given': args.projection is not None,
        'x_scale': 'auto',
        'element_mode': 'all',
        'composition': not args.no_composition,
        'vb_range': 0.5,
        'cb_range': 0.5,
        **_reference_config(args),
        **_style_config(args),
        **_size_config(args),
    }


# =============================================================================
# Band structure + DOS (one figure, a shared energy axis: mace gui's band + DOS view)
# =============================================================================

def configure_band_dos_plot(interactive: bool = True) -> Dict[str, Any]:
    """Configure the band + DOS figure (the band options and the DOS colors)."""
    config = {**configure_band_plot(False), 'dos_colors': None}
    if not interactive:
        return config
    print("\n" + "-" * 40)
    print("  BAND STRUCTURE + DOS CONFIGURATION")
    print("-" * 40)
    lo, hi = _interactive_window(config)
    if not yes_no_prompt(f"Use default energy range ({lo:g} to {hi:+g} eV)?", "yes"):
        config['e_lower'] = get_float_input("  Lower energy limit (eV)", lo)
        config['e_upper'] = get_float_input("  Upper energy limit (eV)", hi)
    config['formats'] = configure_output_formats(interactive=True)
    return config


def plot_band_dos(files: List[str], config: Dict[str, Any], output_dir: str = None) -> List[str]:
    """One band + DOS figure per run ({material}.BAND_DOS.fmt): each file (a BAND or a DOSS
    file) with its partner of the same run. plotly draws mace gui's figure; with matplotlib
    (--backend matplotlib, or no plotly/kaleido) the gallery's band + DOS pane."""
    from .. import bandos_cli
    if _backend(config) == 'plotly':
        return bandos_cli.plot('bandos', files, config, output_dir)
    from .. import gallery
    made, seen = [], set()
    # the matplotlib band + DOS figure is the gallery's pane: it always draws the
    # DOS total, the spin layout of the run and eV. A value it cannot draw is
    # refused (exit 1, no figure), never silently dropped.
    cannot = []
    if config.get('dos_total') is False:
        cannot.append('--no-dos-total (or [plots] dos_total = false)')
    if config.get('spin_dos') not in (None, 'auto'):
        cannot.append(f"--dos-spin {config['spin_dos']} (or [plots] dos_spin)")
    if config.get('units') not in (None, 'eV'):
        cannot.append(f"--units {config['units']} (or [units] energy)")
    if cannot:
        why = ("the matplotlib band + DOS figure cannot draw " + ", ".join(cannot)
               + "; use --backend plotly (needs plotly and kaleido), or leave them out"
               + " (--dos-total, --dos-spin auto and --units eV override a mace-folder.toml)")
        print(f"\n  Error: {why}")
        for f in files:
            outcome.failed(Path(f).name, why)
        return made
    fmts = [f for f in bandos_cli.formats_of(config, 'matplotlib')]
    for f in files:
        band_file, dos_file = bandos_cli.pair_of(Path(f).resolve())
        print(f"\n  Plotting: {Path(f).name}")
        if band_file is None or dos_file is None:
            print("    No partner of the same run beside it (a band + DOS figure needs both): skipped")
            outcome.failed(Path(f).name, "no band or DOS partner of the same run")
            continue
        if band_file in seen:
            continue
        seen.add(band_file)
        try:
            # --dos-smooth / [plots] dos_smooth: the gallery's DOS half smoothed as the DOS figure
            import inspect
            smooth = {'dos_smooth': config.get('dos_smooth') or 0.0} \
                if 'dos_smooth' in inspect.signature(gallery.pane).parameters else {}
            pane = gallery.pane(band_file, 'bandos', (config['e_lower'], config['e_upper']), config.get('fermi_shift', 0.0),
                                config.get('energy_reference') or 'auto', config.get('dos_colors') or 'auto', **smooth)
            if pane.get('error'):
                # an unreadable file or a missing POTC: the pane would draw the message as the figure
                raise ValueError(pane['error'])
            out_dir = _out_dir(band_file, output_dir)
            for fmt in fmts:
                target = out_dir / f"{bandos_cli.material_of(band_file)}.BAND_DOS.{fmt}"
                gallery.render_figure([pane], target, cols=1, dpi=config.get('dpi') or 300, style=config.get('style') or 'default',
                                      size=(config.get('width'), config.get('height')), legend=True,
                                      spin_look={'spin_colors': config.get('spin_colors'),
                                                 'spin_up_color': config.get('spin_up_color'),
                                                 'spin_down_color': config.get('spin_down_color'),
                                                 'spin_down_style': config.get('spin_style'),
                                                 'band_color': config.get('band_color')})
                made.append(str(target))
                print(f"    Generated: {target.name}")
        except Exception as e:
            print(f"    Error: {e}")
            outcome.failed(Path(f).name, str(e))
    return made


def _band_dos_config_from_args(args) -> Dict[str, Any]:
    return {**_band_config_from_args(args), 'dos_colors': getattr(args, 'dos_colors', None)}


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
        'parallel_jobs': max(1, (os.cpu_count() or 2) - 1),
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
        outcome.failed("structure plotter", "could not import plottingCIFs")
        return []

    output_files = []
    output_path = Path(output_dir)

    svg_dir = output_path / "structure_svgs"
    svg_dir.mkdir(parents=True, exist_ok=True)  # -o may name a nested, not-yet-created dir

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

                else:
                    outcome.failed(cif_path.name, "no figure written")

            except Exception as e:
                print(f"    Error: {e}")
                outcome.failed(cif_path.name, str(e))

    except Exception as e:
        print(f"  Error during structure processing: {e}")
        outcome.failed("structures", str(e))

    return output_files


def _structure_config_from_args(args) -> Dict[str, Any]:
    return {
        'supercell': tuple(args.supercell),
        'canvas_size': 800,
        'bond_cutoff': args.bond_cutoff,
        'color_by_coord': args.color_by_coord,
        'parallel_jobs': max(1, (os.cpu_count() or 2) - 1),
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
    from ..bandos_cli import has_partner
    register(PlotterEntry(
        kind=PlotKind.BANDDOS,
        flag='band_dos',
        label='band structure + DOS',
        handler=plot_band_dos,
        configure=configure_band_dos_plot,
        config_from_args=_band_dos_config_from_args,
        # a band file with the DOS of the same run beside it (the pair is one figure)
        patterns=['*.band.band.dat', '*.BAND.DAT', '*_band.dat'],
        sniff=has_partner,
        accepts_energy_range=True,
        progress_tmpl="\n  Plotting {n} band structure + DOS figure(s)...",
        not_found_msg="  No band structure with its DOS beside it found.",
        menu_tmpl="Plot band structure + DOS, one figure ({n} runs)",
        discovery_label="Band Structure + DOS",
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
