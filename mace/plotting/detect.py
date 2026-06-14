"""Registry-driven discovery / classification for mace plotting.

``discover()`` reproduces the legacy ``discover_plottable_files`` behavior
exactly (same glob patterns, dedup, sort) but iterates the registry instead of
hard-coding band/DOS/CIF. ``classify_file()`` maps a single path to its
:class:`PlotKind`.

Phase 0 uses filename globs only (the ``patterns`` field). Later phases add
content sniffers (the ``sniff`` field) for cube sub-types and FREQ ``.out``
files, which share extensions and cannot be told apart by glob alone.

Author: Marcus Djokic
Institution: Michigan State University, Mendoza Group
"""
from __future__ import annotations

import fnmatch
import glob
from pathlib import Path
from typing import Dict, List, Optional

from .registry import PlotKind, entries


def discover(directory: str = ".") -> Dict[PlotKind, List[str]]:
    """Discover plottable files under ``directory``, grouped by PlotKind.

    Mirrors the legacy glob behavior: each registered entry's ``patterns`` are
    globbed (case-sensitive, as on the original code path), results deduped and
    sorted. Sniff-based entries (later phases) are handled after globbing.
    """
    base = Path(directory)
    results: Dict[PlotKind, List[str]] = {}

    for entry in entries():
        found: List[str] = []
        if entry.patterns:
            for pattern in entry.patterns:
                found.extend(glob.glob(str(base / pattern)))
        results[entry.kind] = sorted(set(found))

    # Content-sniff entries (cube / FREQ / spectra): added in later phases.
    # They scan candidate files whose extension is ambiguous and assign by
    # content. No-op while no entry defines ``sniff``.
    _apply_sniffers(base, results)

    return results


def _apply_sniffers(base: Path, results: Dict[PlotKind, List[str]]) -> None:
    """Placeholder for Phase 3+ content-sniff discovery. No-op until an entry
    defines a ``sniff`` callable."""
    sniffers = [e for e in entries() if e.sniff is not None]
    if not sniffers:
        return
    # Implemented in Phase 3 (cube/FREQ). Kept explicit so the discovery
    # contract is visible now and later phases only fill this in.
    for entry in sniffers:
        results.setdefault(entry.kind, [])


def classify_file(path: str) -> Optional[PlotKind]:
    """Classify a single file path to its PlotKind, or None if unrecognized.

    Filename patterns first (cheap), then content sniffers (later phases).
    """
    p = Path(path)
    name = p.name

    for entry in entries():
        if entry.patterns:
            for pattern in entry.patterns:
                if fnmatch.fnmatch(name, pattern):
                    return entry.kind

    for entry in entries():
        if entry.sniff is not None:
            try:
                if p.is_file() and entry.sniff(str(p)):
                    return entry.kind
            except OSError:
                continue

    return None
