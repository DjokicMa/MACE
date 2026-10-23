"""`mace slice`: a plane through a cube file (1.1.6), the CLI side of the GUI's
Volume document Slice section (js/docs/slice_widget.js): a lattice plane of
the grid (--axis a|b|c, exact grid values), a Miller plane (--hkl H K L) or
the plane normal to a direction (--uvw U V W), at a position across the box,
optionally tilted and twisted. The geometry and the sampling are the GUI's
(mace/gui/volume.py plane_geometry / plane_sample), so the picture is the
map the GUI shows. Writes a PNG/SVG/PDF map (matplotlib) or a CSV table.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Optional, Sequence


def build_parser():
    p = argparse.ArgumentParser(
        prog="mace slice",
        description=("A plane through a CRYSTAL cube (density, potential, spin, a difference or a Wannier orbital): "
                     "a lattice plane of the grid, a Miller plane (hkl) or the plane normal to a direction [uvw], "
                     "anywhere across the cell and turned as you like. The GUI's Volume document draws the same "
                     "plane (Slice section, with a mini unit-cell preview you can drag)."),
        epilog=("examples:\n"
                "  mace slice MgO_DENS.CUBE --hkl 1 1 0                # (110) through the middle, MgO_DENS_110.png\n"
                "  mace slice MgO_DENS.CUBE --hkl 1 1 1 --pos 0.25 -o p.svg\n"
                "  mace slice slab_POT.CUBE --uvw 0 0 1 --pos 0.6 --colors physics\n"
                "  mace slice MgO_DENS.CUBE --axis c --index 40 -o c40.csv   # the grid's own plane as a table\n"
                "  mace slice MgO_DENS.CUBE --hkl 1 0 0 --tilt 15 --twist -10"),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cube", help="a .CUBE / *_CUBE.DAT file")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--axis", choices=("a", "b", "c"), help="a lattice plane of the grid along a, b or c (exact values)")
    g.add_argument("--hkl", nargs=3, type=int, metavar=("H", "K", "L"), help="a Miller plane (default: 0 0 1)")
    g.add_argument("--uvw", nargs=3, type=int, metavar=("U", "V", "W"), help="the plane normal to the direction [uvw]")
    p.add_argument("--index", type=int, help="with --axis: the grid plane (default: the middle one)")
    p.add_argument("--pos", type=float, default=0.5, help="position across the box along the normal, 0 to 1 (default 0.5)")
    p.add_argument("--tilt", type=float, default=0.0, help="turn about the plane's first in-plane axis, degrees")
    p.add_argument("--twist", type=float, default=0.0, help="turn about its second in-plane axis, degrees")
    p.add_argument("-o", "--output", help="output: .png (default), .svg, .pdf or .csv")
    p.add_argument("--contours", type=int, default=12, help="contour levels (default 12)")
    p.add_argument("--colors", choices=("chemistry", "physics"), default=None,
                   help="a potential's colors: chemistry red = negative (default, as the GUI), physics red = positive")
    p.add_argument("--dpi", type=int, default=150)
    return p


def plane_of(path: str, axis=None, index=None, hkl=None, uvw=None, pos=0.5, tilt=0.0, twist=0.0):
    """{x, y, z (rows along y; NaN outside), outline, title, type, signed} of the plane asked for."""
    import numpy as np
    from mace.gui import volume as V
    if axis:
        c = V.read(path, skip_vacuum=True)
        n = c.data.shape["abc".index(axis)]
        i = n // 2 if index is None else int(index)
        r = V.slice_job(path=path, axis=axis, index=i)
        z = np.array([[np.nan if v is None else v for v in row] for row in r["cart"]], float).T
        P = np.asarray(r["corners"], float)
        outline = [[P[0, 0], P[1, 0], P[3, 0], P[2, 0]], [P[0, 1], P[1, 1], P[3, 1], P[2, 1]]]
        title = f"Plane {axis} = {i} of {n} ({r['position_A']:.3f} Å from the grid's origin)"
        return {"x": np.asarray(r["x"]), "y": np.asarray(r["y"]), "z": z, "outline": outline, "title": title,
                "type": r["type"], "tag": f"{axis}{i}", **_axes(axis), **_quantity(c)}
    mode, ints = ("uvw", uvw) if uvw else ("hkl", hkl or (0, 0, 1))
    c = V.read(path, skip_vacuum=True)
    r = V.plane_of_cube(c, mode, ints, pos, tilt, twist)
    ix = " ".join(str(x) for x in r["ints"])
    head = f"({ix})" if mode == "hkl" else f"normal to [{ix}]"
    turned = f", turned {tilt:g}° / {twist:g}°" if tilt or twist else ""
    title = f"{head}{turned} at {pos:.3f}: {r['position_A']:.3f} Å along the normal of {r['dmax'] - r['dmin']:.3f} Å"
    if mode == "hkl":
        title += f" (h x + k y + l z = {r['miller_t']:.3f}, d = {r['spacing_A']:.3f} Å)"
    tag = ("" if mode == "hkl" else "uvw") + "".join(str(x) if x >= 0 else f"m{-x}" for x in r["ints"])
    return {"x": np.asarray(r["x"]), "y": np.asarray(r["y"]), "z": r["values"], "outline": r["outline"], "title": title,
            "type": r["type"], "tag": tag, **_axes(None), **_quantity(c)}


def _axes(axis) -> dict:
    """The in-plane axes' titles (plot markup, mace.utils.plot_labels), as the GUI's slice names them: a grid
    plane runs along its first cell axis and across it (⊥) in the plane; an (hkl) or [uvw] plane along
    e1 and e2."""
    from mace.utils import plot_labels as L
    if axis:
        a = [n for n in "abc" if n != axis][0]
        return {"xlabel": L.with_unit(f"Along {a}", "A"), "ylabel": L.with_unit(f"⊥ {a}", "A")}
    return {"xlabel": L.with_unit(f"Along {L.SYMBOLS['e_1']}", "A"), "ylabel": L.with_unit(f"Along {L.SYMBOLS['e_2']}", "A")}


def _quantity(c) -> dict:
    """The colorbar's title: the cube's quantity and the values' unit (plot markup)."""
    from mace.gui import volume as V
    return {"label": V._label(c.data_type), "unit": V._engine().cube_value_unit(c)}


def write(res, out: Path, contours: int = 12, colors: Optional[str] = None, dpi: int = 150, name: str = "") -> Path:
    import numpy as np
    if out.suffix.lower() == ".csv":
        with open(out, "w") as fh:
            fh.write(f"# {name}: {res['title']}\n# x_A,y_A,value\n")
            for j, y in enumerate(res["y"]):
                for i, x in enumerate(res["x"]):
                    v = res["z"][j, i]
                    if math.isfinite(v):
                        fh.write(f"{x:.5f},{y:.5f},{v:.8g}\n")
        return out
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = np.ma.masked_invalid(res["z"])
    t = str(res["type"])
    fin = z.compressed()
    signed = fin.size and fin.min() < 0 < fin.max() and t != "density"
    fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=dpi)
    if signed:
        # robust limits: the spikes at the nuclei do not take the whole scale (the GUI's map spans the full +/- max)
        m = float(np.percentile(np.abs(fin), 99.5)) or float(np.abs(fin).max()) or 1.0
        # + blue, − red (spin, differences, the chemistry potential colors); physics: red = positive
        cmap = "RdBu_r" if (t.startswith("potential") and colors == "physics") else "RdBu"
        levels = np.linspace(-m, m, max(3, contours) + 1)
    else:
        cmap = "viridis"
        lo, hi = (float(np.percentile(fin, 0.5)), float(np.percentile(fin, 99.5))) if fin.size else (0.0, 1.0)
        levels = np.linspace(lo, hi if hi > lo else lo + 1e-12, max(3, contours) + 1)
    cs = ax.contourf(res["x"], res["y"], z, levels=levels, cmap=cmap, extend="both")
    ax.contour(res["x"], res["y"], z, levels=cs.levels, colors="k", linewidths=0.3, alpha=0.5)
    U, W = res["outline"]
    ax.plot(list(U) + [U[0]], list(W) + [W[0]], color="0.3", lw=0.8)
    from mace.utils import plot_labels as L
    cb = fig.colorbar(cs, ax=ax, shrink=0.85)
    if res.get("label"):
        cb.set_label(L.mathtext(L.with_unit(res["label"], res.get("unit"))))
    ax.set_aspect("equal")
    ax.set_xlabel(L.mathtext(res["xlabel"]))
    ax.set_ylabel(L.mathtext(res["ylabel"]))
    short = name if len(name) <= 70 else "…" + name[-69:]
    ax.set_title(f"{short}\n{res['title']}", fontsize=8, wrap=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    p = Path(args.cube)
    if not p.is_file():
        print(f"mace slice: {p}: no such file", file=sys.stderr)
        return 1
    try:
        res = plane_of(str(p), axis=args.axis, index=args.index, hkl=args.hkl, uvw=args.uvw, pos=args.pos,
                       tilt=args.tilt, twist=args.twist)
        out = Path(args.output) if args.output else p.with_name(f"{p.stem}_{res['tag']}.png")
        if out.suffix.lower() not in (".png", ".svg", ".pdf", ".csv"):
            out = out.with_suffix(".png")
        write(res, out, contours=args.contours, colors=args.colors, dpi=args.dpi, name=p.name)
    except ValueError as exc:
        print(f"mace slice: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {out}: {res['title']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
