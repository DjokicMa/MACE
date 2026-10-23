"""Does a band-index DOSS reach far enough above E_F? Read from its own files (1.1.6).

A 1D material's DOSS takes its bands by index by default (Crystal_d3/d3_config.doss_band_index_range:
CRYSTAL needs an energy window's bounds in band gaps), so how far the DOS reaches above E_F depends on
how high its last band goes, which no deck writer can know before the run. A run whose DOSS deck names
its bands by index (INZB and IFNB positive) and whose DOSS.DAT stops below E_F + DOSS_REACH_EV (the
window's top, 15 eV) gets a note naming the flag that adds conduction bands. The note is the DOS plot's
(GUI, mace plotting --dos) and the extractor's (doss_reach_note in materials.db).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

from mace.constants import HARTREE_TO_EV as HARTREE_EV
_SUFFIXES = (".DOSS.DAT", "_doss.dat")


def _d3_config():
    d3 = str(Path(__file__).resolve().parents[2] / "Crystal_d3")
    if d3 not in sys.path:
        sys.path.insert(0, d3)
    import d3_config
    return d3_config


def deck_band_range(deck: Path) -> Optional[Tuple[int, int]]:
    """(INZB, IFNB) of a DOSS deck that names its bands by index, else None (an energy window, no DOSS)."""
    try:
        lines = [l.strip() for l in Path(deck).read_text(errors="replace").splitlines()]
    except OSError:
        return None
    for i, line in enumerate(lines):
        if line.upper() == "DOSS" and i + 1 < len(lines):
            parts = lines[i + 1].split()
            try:
                first, last = int(parts[2]), int(parts[3])
            except (IndexError, ValueError):
                return None
            return (first, last) if first > 0 and last > 0 else None
    return None


def _deck_of(dat: Path) -> Optional[Path]:
    """The DOSS deck of the run whose DOSS.DAT this is (<stem>.d3 beside it, else the folder's one DOSS deck)."""
    name = dat.name
    stem = next((name[: -len(s)] for s in _SUFFIXES if name.lower().endswith(s.lower())), None)
    if stem:
        cand = dat.with_name(stem + ".d3")
        if cand.is_file():
            return cand
    decks = [d for d in sorted(dat.parent.glob("*.d3")) if deck_band_range(d) is not None
             or "DOSS" in d.read_text(errors="replace").upper().split()]
    return decks[0] if len(decks) == 1 else None


def top_ev(dat: Path) -> Optional[float]:
    """The highest energy of a DOSS.DAT, eV from E_F (CRYSTAL writes E - E_F in Hartree)."""
    try:
        text = Path(dat).read_text(errors="replace")
    except OSError:
        return None
    top = None
    for line in text.splitlines():
        if line and line[0] not in "#@" and line.strip():
            try:
                e = float(line.split()[0])
            except ValueError:
                continue
            top = e if top is None else max(top, e)
    return None if top is None else top * HARTREE_EV


def reach_note(dat: Path) -> Optional[str]:
    """Why a band-index DOS stops too low above E_F, or None."""
    dat = Path(dat)
    deck = _deck_of(dat)
    rng = deck_band_range(deck) if deck is not None else None
    if rng is None:
        return None
    top = top_ev(dat)
    reach = _d3_config().DOSS_REACH_EV
    if top is None or top >= reach:
        return None
    return (f"the DOS stops at E_F{top:+.1f} eV, below E_F+{reach:g} eV: its deck takes bands {rng[0]} to {rng[1]} "
            f"by index (a 1D material's default), and the higher conduction bands are not in it. Rerun the DOSS "
            f"deck with more of them (mace opt2d3 --doss-conduction-bands N)")


_CACHE: dict = {}


def cached_note(path: Path) -> Optional[str]:
    """reach_note, once per DOSS.DAT state (the GUI asks on every redraw)."""
    p = Path(path)
    try:
        key = (str(p.resolve()), p.stat().st_mtime_ns)
    except OSError:
        return None
    if key not in _CACHE:
        try:
            _CACHE[key] = reach_note(p)
        except Exception:
            _CACHE[key] = None
    return _CACHE[key]


def add_note(data: dict, path: Path) -> dict:
    """The DOS plot's data with the reach note among its notes (and as reach_note)."""
    note = cached_note(path)
    data["reach_note"] = note
    if note and note not in data.get("notes", []):
        data.setdefault("notes", []).append(note)
    return data
