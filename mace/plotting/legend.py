"""``mace plotting --legend``: an element legend for figures (SVG, PNG or a
table), from the elements of the given structure files or folders (the
union over all of them), in the GUI's element palette. The work is done by
mace/utils/element_legend.py, which the GUI's Export Element Legend uses
too, so both write the same file for the same choices."""
from __future__ import annotations

import glob
from pathlib import Path
from typing import List

from mace.utils import element_legend as L

EXT = {"svg": ".svg", "png": ".png", "tsv": ".tsv", "csv": ".csv", "json": ".json"}


def _inputs(files_pos: List[str], directory: str) -> List[str]:
    out: List[str] = []
    for f in files_pos or [directory]:
        if any(ch in f for ch in "*?[") and not Path(f).exists():
            out += sorted(glob.glob(f, recursive=True))     # a quoted pattern: "runs/*/*.out"
        else:
            out.append(f)
    return out


def options_from_args(args) -> dict:
    return {"layout": args.legend_layout, "columns": args.legend_columns, "shape": args.legend_shape,
            "swatch": args.legend_swatch, "font_size": args.legend_font_size, "label": args.legend_label,
            "text": args.legend_text, "background": args.legend_background, "order": args.legend_order,
            "outline": not args.legend_no_outline, "dpi": args.dpi or 300}


def run_legend(args, files_pos: List[str], parser) -> int:
    fmt = args.format or "svg"
    if fmt not in L.FORMATS:
        parser.error(f"--legend writes {', '.join(L.FORMATS)}, not {fmt}")
    inputs = _inputs(files_pos, args.directory)
    try:
        res = L.legend(inputs, fmt=fmt, palette=args.palette, options=options_from_args(args),
                       recursive=args.legend_recursive)
    except ValueError as e:
        print(f"  Error: {e}")
        return 1
    out = Path(args.output)
    if out.suffix.lower() not in EXT.values():
        out = out / f"element_legend{EXT[fmt]}"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(res.content)
    for f, why in res.skipped:
        print(f"  Skipped {f}: {why}")
    print(f"  {len(res.symbols)} element(s) from {len(res.per_file)} file(s): {' '.join(res.symbols)}")
    print(f"  Palette: {res.palette}")
    print(f"  Wrote {out}")
    return 0
