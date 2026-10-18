"""``mace plotting --bz``: the first Brillouin zone with the k-path in it.

The geometry is Crystal_d3/d3_bz.py's, the one the mace gui view draws: the
zone of the run's cell (its .out), the high-symmetry points, and the path of
the run's .d3 as arrows (or the path as edited, ``--band-path FILE``: hidden
segments faint, jumps dashed). Input: BAND.DAT / .d3 / .out files, or every
BAND.DAT in the directory.
"""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path
from typing import List, Optional

_D3 = Path(__file__).resolve().parents[3] / "Crystal_d3"
_PLOTTING = Path(__file__).resolve().parents[3] / "Plotting"


def _tools():
    for d in (_D3, _PLOTTING):
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
    import d3_band_path
    import d3_bz
    return d3_band_path, d3_bz


def _deck_for(path: Path) -> Optional[Path]:
    if path.suffix == ".d3":
        return path
    name = path.name
    for suf in (".BAND.DAT", ".band.band.dat", "_band.dat", ".out"):
        if name.endswith(suf):
            name = name[: -len(suf)]
            break
    material = name[:-5] if name.endswith("_band") else name
    for cand in (f"{name}.d3", f"{material}_band.d3", f"{material}.d3"):
        if (path.parent / cand).is_file():
            return path.parent / cand
    return None


def plot_bz(files: List[str], band_path: Optional[str], output_dir: str, formats=("png", "svg")) -> List[str]:
    kp, bz = _tools()
    edit = None
    if band_path:
        import ipBANDS_V2
        edit = ipBANDS_V2.load_band_path(band_path)
    written: List[str] = []
    for f in files:
        p = Path(f)
        out = bz.output_for(str(p))
        if out is None:
            print(f"  {p.name}: no CRYSTAL output beside it to read the cell from")
            continue
        deck = _deck_for(p)
        computed = []
        if deck is not None:
            parsed = kp.parse_band_deck(deck.read_text(errors="replace"))
            computed = parsed["segments"] if parsed else []
        segs = computed
        if edit is not None:
            resolved = kp.resolve([dict(s, hidden=False) for s in edit], computed)
            segs = [dict(r, **({"hidden": True} if s.get("hidden") else {})) for r, s in zip(resolved, edit)]
        view = bz.bz_view(str(out), segs)
        stem = p.name
        for suf in (".BAND.DAT", ".band.band.dat", "_band.dat", ".d3", ".out"):
            if stem.endswith(suf):
                stem = stem[: -len(suf)]
                break
        stem = stem[:-5] if stem.endswith("_band") else stem
        title = f"{stem}: first Brillouin zone" + (" (edited k-path)" if edit is not None else "")
        os.makedirs(output_dir, exist_ok=True)
        for w in bz.plot_bz(view, title, str(Path(output_dir) / stem), formats):
            print(f"    Generated: {Path(w).name}")
            written.append(w)
    return written


def main(args) -> int:
    files = list(getattr(args, "files", []) or [])
    if not files:
        d = getattr(args, "directory", ".")
        files = sorted(glob.glob(os.path.join(d, "*.BAND.DAT")) + glob.glob(os.path.join(d, "*_band.dat")))
    if not files:
        print("  No band structure files found (give a BAND.DAT, .d3 or .out).")
        return 1
    fmt = getattr(args, "format", None)
    formats = (fmt,) if fmt in ("png", "svg", "pdf") else ("png", "svg")
    print(f"\n  Drawing the Brillouin zone of {len(files)} run(s)...")
    return 0 if plot_bz(files, getattr(args, "band_path", None), getattr(args, "output", "."), formats) else 1
