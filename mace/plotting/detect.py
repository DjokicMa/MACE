"""Registry-driven discovery / classification for mace plotting.

``discover()`` reproduces the legacy ``discover_plottable_files`` behavior
exactly for glob-only kinds (band/DOS/CIF/cube/spectra) but iterates the registry
instead of hard-coding them. ``classify_file()`` maps a single path to its
:class:`PlotKind`.

Two matching signals per entry:

* ``patterns`` — filename globs (cheap extension/suffix gate). Cube (``*.CUBE``)
  and spectra (``*IRSPEC.DAT`` / ``*RAMSPEC.DAT``) are fully disambiguated here.
* ``sniff`` — a content predicate ``(path) -> bool`` for kinds that share an
  extension with unrelated files. FREQ is the case that needs it: ``.out`` is
  shared by hundreds of SCF/OPT/BAND runs, so an entry with ``patterns=['*.out']``
  uses ``patterns`` only as a candidate filter and ``sniff`` as the real gate.

An entry with both must satisfy *both*: the pattern narrows to candidates, the
sniff confirms content. An entry with patterns only matches on the pattern; an
entry with sniff only (no patterns) is not auto-discovered (nothing to glob).

Author: Marcus Djokic
Institution: Michigan State University, Mendoza Group
"""
from __future__ import annotations

import fnmatch
import glob
from pathlib import Path
from typing import Dict, List, Optional

from .registry import PlotKind, PlotterEntry, entries

# FREQ .out signature: the phonon-calc banner AND a real normal-modes block.
# Gating on the modes marker (C1 eigenvector gate) means an aborted freq run that
# printed the banner but produced no modes classifies UNKNOWN, not FREQ.
_FREQ_SIGNATURE = "CALCULATION OF PHONON FREQUENCIES AT THE GAMMA POINT"
_FREQ_MODES_MARKER = "NORMAL MODES NORMALIZED"

# Defensive bound: never slurp a pathological multi-hundred-MB file into memory.
_MAX_SNIFF_BYTES = 256 * 1024 * 1024


def is_freq_output(path: str) -> bool:
    """True iff ``path`` is a CRYSTAL FREQ run with a real normal-modes block.

    Reads the file once (utf-8, errors replaced); both markers must be present.
    Returns False (never raises) for missing/unreadable/oversize files so it is
    safe to call over a whole directory during discovery.
    """
    try:
        p = Path(path)
        if not p.is_file() or p.stat().st_size > _MAX_SNIFF_BYTES:
            return False
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return _FREQ_SIGNATURE in text and _FREQ_MODES_MARKER in text


def _safe_sniff(entry: PlotterEntry, path: str) -> bool:
    """Run an entry's sniff predicate, swallowing OSError -> False."""
    try:
        return bool(entry.sniff(path))
    except OSError:
        return False


def discover(directory: str = ".") -> Dict[PlotKind, List[str]]:
    """Discover plottable files under ``directory``, grouped by PlotKind.

    For each registered entry: glob its ``patterns`` (case-sensitive, as the
    original code path), dedup + sort, then — if the entry defines a ``sniff`` —
    keep only the candidates whose content passes the gate. Entries with no
    ``patterns`` cannot be globbed and yield an empty bucket.
    """
    base = Path(directory)
    results: Dict[PlotKind, List[str]] = {}

    for entry in entries():
        found: List[str] = []
        if entry.patterns:
            for pattern in entry.patterns:
                found.extend(glob.glob(str(base / pattern)))
        found = sorted(set(found))
        if entry.sniff is not None:
            found = [f for f in found if _safe_sniff(entry, f)]
        results[entry.kind] = found

    return results


def classify_file(path: str) -> Optional[PlotKind]:
    """Classify a single file path to its PlotKind, or None if unrecognized.

    An entry matches when its filename pattern matches (if it has patterns) AND
    its content sniff passes (if it has a sniff). A ``.out`` that matches the
    FREQ glob but fails the FREQ content gate is therefore *not* FREQ.
    """
    p = Path(path)
    name = p.name

    for entry in entries():
        if entry.patterns is not None:
            if not any(fnmatch.fnmatch(name, pat) for pat in entry.patterns):
                continue
        elif entry.sniff is None:
            continue  # entry has neither signal — cannot match
        if entry.sniff is not None:
            if not (p.is_file() and _safe_sniff(entry, str(p))):
                continue
        return entry.kind

    return None
