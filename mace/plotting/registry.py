"""Registry spine for ``mace plotting``.

The registry is the single source of truth for "what can mace plot and how".
Each visualization (band / DOS / structure today; cube / FREQ / IR / Raman in
later phases) registers one :class:`PlotterEntry`. Discovery (``detect.py``),
the interactive menu, and CLI dispatch (``main.py``) all iterate the registry
rather than hard-coding each kind, so adding a new visualization is a single
``register()`` call.

This module has no internal dependencies (stdlib only) to keep the import
graph acyclic: ``registry`` <- ``detect`` / ``handlers`` <- ``main``.

Author: Marcus Djokic
Institution: Michigan State University, Mendoza Group
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class PlotKind(Enum):
    """Kinds of CRYSTAL output mace plotting can visualize.

    BAND/DOS/STRUCTURE are wired in Phase 0. CUBE/FREQ/SPECTRA_*/UNKNOWN are
    declared now (harmless) so later phases only add registrations, not enum
    members.
    """

    BAND = "band"
    DOS = "dos"
    STRUCTURE = "structure"
    CUBE = "cube"
    FREQ = "freq"
    SPECTRA_IR = "ir"
    SPECTRA_RAMAN = "raman"
    UNKNOWN = "unknown"


# Handler signature: (files, config, output_dir) -> list of generated paths.
Handler = Callable[[List[str], Dict[str, Any], str], List[str]]
# Configure signature: (interactive) -> config dict.
Configure = Callable[[bool], Dict[str, Any]]


@dataclass
class PlotterEntry:
    """One registered visualization.

    Attributes
    ----------
    kind, flag, label:
        Identity. ``flag`` is the argparse dest (e.g. ``"band"`` for
        ``--band``); ``label`` is a short human noun used in messages.
    handler:
        ``(files, config, output_dir) -> [generated paths]``.
    configure:
        ``(interactive) -> config dict`` for interactive / ``--all`` use.
    config_from_args:
        Builds the config dict directly from parsed CLI args for single-flag
        mode (``--band`` etc.). Falls back to ``configure(False)`` if None.
    patterns:
        Glob patterns for filename-based discovery (band/DOS/structure).
    sniff:
        Optional content classifier ``(path) -> bool`` for later phases where
        extension/glob cannot disambiguate (cube sub-types, FREQ ``.out``).
    accepts_energy_range:
        Whether ``--all`` should override e_lower/e_upper from CLI args
        (true for band/DOS, false for structure) — preserves legacy behavior.
    progress_tmpl, not_found_msg, menu_tmpl:
        Console strings, ``{n}`` = file count. Kept per-entry to preserve the
        exact wording of the legacy CLI.
    """

    kind: PlotKind
    flag: str
    label: str
    handler: Handler
    configure: Configure
    config_from_args: Optional[Callable[[Any], Dict[str, Any]]] = None
    patterns: Optional[List[str]] = None
    sniff: Optional[Callable[[str], bool]] = None
    accepts_energy_range: bool = False
    progress_tmpl: str = "  Plotting {n} file(s)..."
    not_found_msg: str = "  No files found."
    menu_tmpl: str = "Plot ({n} files)"
    discovery_label: str = "Plottable"  # section header in print_discovered_files


# Insertion-ordered (dict preserves order in py3.7+); legacy order is
# band, dos, structure.
REGISTRY: "Dict[PlotKind, PlotterEntry]" = {}


def register(entry: PlotterEntry) -> None:
    """Register (or replace) the plotter for ``entry.kind``."""
    REGISTRY[entry.kind] = entry


def get(kind: PlotKind) -> Optional[PlotterEntry]:
    return REGISTRY.get(kind)


def entries() -> List[PlotterEntry]:
    """All registered entries, in registration order."""
    return list(REGISTRY.values())
