"""Pure IR/Raman spectra read/render/average functions (Phase 4).

Lifted from ``plotIRRAM.py`` — the verified superset of the five standalone
spectra scripts (``irspec_plot``/``ramspec_plot``/``irspec_average``/
``ramspec_average`` are subsets). Their ``read_*`` bodies are byte-identical, so
they collapse to a single :func:`read_spectrum`. The matplotlib plot bodies are
preserved verbatim in spirit; only the I/O plumbing changed, applying the §2.4
robustness fixes:

* column gate relaxed from ``!= N`` to "has at least the columns this view
  needs" — extra columns are tolerated, too-few are skipped (not crashed);
* no ``os.getcwd()`` — callers pass ``out_prefix`` / file paths;
* no ``input()`` — rendering is fully headless (matplotlib ``Agg``);
* output format parametrized (png / svg / pdf via ``savefig``).

These are pure functions: no registry, no prompts. The handler module
(``handlers/spectra.py``) wires them to discovery, config and the registry.

IRSPEC.DAT columns:  wavenumber, frequency, absorbance.
RAMSPEC.DAT columns: wavenumber, total, parallel, perpendicular, then the six
single-crystal directions xx,xy,xz,yy,yz,zz.

Author: Marcus Djokic
Institution: Michigan State University, Mendoza Group
"""
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

SINGLE_CRYSTAL_LABELS = ['xx', 'xy', 'xz', 'yy', 'yz', 'zz']
SINGLE_CRYSTAL_COLORS = ['steelblue', 'darkorange', 'seagreen',
                         'mediumpurple', 'crimson', 'goldenrod']

# Minimum column count each Raman view needs.
RAMAN_MIN_COLS = {'total': 2, 'par_perp': 4, 'all': 10}


# --------------------------------------------------------------------------- #
# Reading / grouping (shared, identical across all 5 source scripts)
# --------------------------------------------------------------------------- #

def read_spectrum(file_path: str) -> np.ndarray:
    """Read an IRSPEC/RAMSPEC .DAT file into a 2D float array.

    Skips blank and ``#`` comment lines and any row that does not parse as
    floats. Returns an empty array for an empty/unparseable file — and for a
    missing/unreadable one (with a printed error): spectra was the only
    handler kind that let OSError escape as a raw traceback.
    """
    data = []
    try:
        with open(file_path, 'r', errors='replace') as fh:
            for line in fh:
                clean = line.strip()
                if clean == '' or clean.startswith('#'):
                    continue
                try:
                    data.append([float(x) for x in clean.split()])
                except ValueError:
                    continue
    except OSError as e:
        print(f"    Error reading {os.path.basename(str(file_path))}: {e}")
        return np.array([])
    return np.array(data)


def get_material_name(file_name: str) -> Optional[str]:
    """Material name = everything before ``-confN_`` (None if not conf-style)."""
    m = re.match(r'^(.+?)-conf\d+_', os.path.basename(file_name))
    return m.group(1) if m else None


def group_files_by_material(files: List[str]) -> Dict[str, List[str]]:
    """Group conf-style files by material; non-conf files are dropped."""
    groups: Dict[str, List[str]] = {}
    for f in files:
        mat = get_material_name(f)
        if mat is None:
            continue
        groups.setdefault(mat, []).append(f)
    return groups


def average_spectrum(file_paths: List[str], min_cols: int
                     ) -> Optional[Tuple[np.ndarray, int]]:
    """Average several spectra (same material) into one data array.

    Files are read in sorted order; the first valid file fixes the reference
    shape, and any later file whose shape differs (ragged grid / wrong column
    count, or fewer than ``min_cols`` columns) is skipped rather than crashing
    the average. Returns ``(avg_data, n_used)`` with the same column layout as
    the inputs, or ``None`` if nothing valid was found.
    """
    stack: List[np.ndarray] = []
    ref_shape: Optional[tuple] = None
    for fp in sorted(file_paths):
        d = read_spectrum(fp)
        if d.size == 0 or d.ndim != 2 or d.shape[1] < min_cols:
            continue
        if ref_shape is None:
            ref_shape = d.shape
        if d.shape != ref_shape:
            continue
        stack.append(d)
    if not stack:
        return None
    return np.mean(np.stack(stack), axis=0), len(stack)


# --------------------------------------------------------------------------- #
# Rendering (matplotlib, headless Agg)
# --------------------------------------------------------------------------- #

def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def _save(plt, fig, out_prefix: str, suffix: str, formats) -> List[str]:
    paths = []
    for fmt in formats:
        path = f"{out_prefix}{suffix}.{fmt}"
        fig.savefig(path, dpi=300)
        paths.append(path)
    plt.close(fig)
    return paths


def _decorate(ax, title, ylabel='Intensity'):
    ax.set_xlabel('Wavenumber (cm$^{-1}$)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.invert_xaxis()
    ax.legend()
    ax.grid(True, alpha=0.3)


def render_ir(data: np.ndarray, out_prefix: str, formats=('png',),
              ir_column: int = 2, title_suffix: str = '') -> List[str]:
    """Plot IR absorbance vs wavenumber. ``ir_column`` selects the intensity
    column (default 2 = absorbance). Returns the saved file paths."""
    if data.size == 0 or data.ndim != 2 or data.shape[1] <= ir_column:
        return []
    plt = _plt()
    wavenumber = data[:, 0]
    absorbance = data[:, ir_column]
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(wavenumber, absorbance, color='steelblue', linewidth=1.5, label='Absorbance')
    _decorate(ax, 'IR Absorbance Spectrum' + title_suffix, ylabel='Absorbance')
    x_min = np.floor(wavenumber.min() / 1000) * 1000
    x_max = np.ceil(wavenumber.max() / 1000) * 1000
    ax.set_xticks(np.arange(x_min, x_max + 1000, 1000))
    fig.tight_layout()
    return _save(plt, fig, out_prefix, '_absorbance', formats)


def render_raman(data: np.ndarray, out_prefix: str, mode: str = 'all',
                 formats=('png',), title_suffix: str = '') -> List[str]:
    """Plot a Raman spectrum.

    ``mode`` is one of:
      * ``total``    — total polycrystalline intensity (needs >=2 cols)
      * ``par_perp`` — total + parallel + perpendicular (needs >=4 cols)
      * ``all``      — par/perp panel + the six single-crystal directions
                       (needs >=10 cols)
    Returns ``[]`` (not a crash) if the data lacks the columns the mode needs.
    """
    need = RAMAN_MIN_COLS.get(mode, 10)
    if data.size == 0 or data.ndim != 2 or data.shape[1] < need:
        return []
    plt = _plt()
    wn = data[:, 0]

    if mode == 'total':
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.plot(wn, data[:, 1], color='steelblue', linewidth=1.5, label='Total')
        _decorate(ax, 'Raman Spectrum - Total' + title_suffix)
        fig.tight_layout()
        return _save(plt, fig, out_prefix, '_total', formats)

    if mode == 'par_perp':
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.plot(wn, data[:, 1], color='steelblue', linewidth=1.5, label='Total')
        ax.plot(wn, data[:, 2], color='darkorange', linewidth=1.5, label='Parallel')
        ax.plot(wn, data[:, 3], color='seagreen', linewidth=1.5, label='Perpendicular')
        _decorate(ax, 'Raman Spectrum - Total, Parallel, Perpendicular' + title_suffix)
        fig.tight_layout()
        return _save(plt, fig, out_prefix, '_par_perp', formats)

    # mode == 'all'
    fig, axes = plt.subplots(2, 1, figsize=(8, 10))
    axes[0].plot(wn, data[:, 1], color='steelblue', linewidth=1.5, label='Total')
    axes[0].plot(wn, data[:, 2], color='darkorange', linewidth=1.5, label='Parallel')
    axes[0].plot(wn, data[:, 3], color='seagreen', linewidth=1.5, label='Perpendicular')
    _decorate(axes[0], 'Raman Spectrum - Total, Parallel, Perpendicular' + title_suffix)
    for i in range(6):
        axes[1].plot(wn, data[:, 4 + i], color=SINGLE_CRYSTAL_COLORS[i],
                     linewidth=1.5, label=SINGLE_CRYSTAL_LABELS[i])
    _decorate(axes[1], 'Raman Spectrum - Single Crystal Directions' + title_suffix)
    fig.tight_layout()
    return _save(plt, fig, out_prefix, '_all', formats)
