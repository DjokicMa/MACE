"""`mace transform`: rotate and translate atoms of a structure file, as the GUI's Rotate and Move tools do.

The command-line side of the viewport gizmos (1.1.6): a turn by an exact angle about an axis (x, y, z, a cell
axis or reciprocal axis, or a lattice direction [u v w]) through the pivot the GUI's Rotate panel offers: the
selection's centroid or center of mass (the atoms made whole across the cell's faces first, as the GUI does), the
cell's center, the origin, an atom or a given point; then a displacement in Å or in fractions of the cell axes.
The cell is unchanged; the result is written with ase (CIF by default).
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

USAGE = "mace transform INPUT [--atoms LIST] [--rotate DEG --axis AXIS] [--about centroid|com|cell|origin|atom:N|X,Y,Z] " \
        "[--translate DX DY DZ | --translate-frac DA DB DC] [-o OUTPUT] [--no-wrap] [--force]"


class TransformError(ValueError):
    pass


def parse_atoms(text: Optional[str], symbols: Sequence[str]) -> List[int]:
    """0-based indices from "1,5-8", element symbols ("O", "Mg,O") or "all" (blank: all). Sorted, unique."""
    n = len(symbols)
    if text is None or not str(text).strip() or str(text).strip().lower() == "all":
        return list(range(n))
    out = set()
    for part in re.split(r"[\s,;]+", str(text).strip()):
        if not part:
            continue
        m = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if m:
            a = int(m.group(1)); b = int(m.group(2) or a)
            if a < 1 or b < a or b > n:
                raise TransformError(f"--atoms {part}: the structure has atoms 1 to {n}")
            out.update(range(a - 1, b))
            continue
        sym = part[:1].upper() + part[1:].lower()
        hits = [i for i, s in enumerate(symbols) if s == sym]
        if not hits:
            raise TransformError(f"--atoms {part}: no such atom index or element in the structure")
        out.update(hits)
    return sorted(out)


def _usable(cell, pbc) -> bool:
    c = np.asarray(cell, float)
    return c.shape == (3, 3) and any(pbc) and abs(np.linalg.det(c)) > 1e-9


def axis_vector(text: str, cell, pbc) -> np.ndarray:
    """A unit axis: x, y, z; a, b, c (cell rows); a*, b*, c* (reciprocal rows); or three numbers: a lattice
    direction [u v w] for a periodic cell, a Cartesian vector otherwise (as the GUI's Rotate panel reads it)."""
    k = str(text).strip()
    if k in ("x", "y", "z"):
        return np.eye(3)["xyz".index(k)]
    usable = _usable(cell, pbc)
    c = np.asarray(cell, float)
    if re.fullmatch(r"[abc]\*?", k):
        if not usable:
            raise TransformError(f"--axis {k} needs a cell")
        v = np.linalg.inv(c).T["abc".index(k[0])] if k.endswith("*") else c["abc".index(k[0])]
        return v / np.linalg.norm(v)
    t = k.strip("[]")
    parts = [p for p in re.split(r"[\s,;]+", t) if p]
    if len(parts) == 1 and re.fullmatch(r"(-?\d){3}", parts[0]):
        parts = re.findall(r"-?\d", parts[0])          # "110", "1-10": one digit per index
    try:
        v = np.array([float(p) for p in parts], float)
    except ValueError:
        v = np.zeros(0)
    if v.shape != (3,) or not np.all(np.isfinite(v)) or np.linalg.norm(v) < 1e-12:
        raise TransformError(f"--axis {text!r}: x, y, z, a, b, c, a*, b*, c* or three numbers, not all zero")
    if usable:
        v = v @ c
    return v / np.linalg.norm(v)


def whole(frac: np.ndarray, cell, pbc, idx: Sequence[int]) -> np.ndarray:
    """The picked atoms' fractional coordinates made whole across the cell's faces: each at its image nearest an
    atom already placed (periodic axes only; above 400 atoms, nearest the first). gizmo_math.js wholeSelection."""
    c = np.asarray(cell, float)
    per = np.array([bool(p) for p in pbc])
    out = [frac[idx[0]].copy()]
    for i in idx[1:]:
        f = frac[i]
        refs = out[:1] if len(out) > 400 else out
        best, bl = None, math.inf
        for r in refs:
            d = f - r
            d = np.where(per, d - np.round(d), d)
            L = float(np.sum((d @ c) ** 2))
            if L < bl:
                bl, best = L, r + d
        out.append(best)
    return np.array(out)


BBOX_PAD = 1.0      # structures.py / geom/core.js: the bounding box's 1 Å on each side
PIVOTS = ("centroid", "com", "cell", "origin")
ABOUT_HELP = "centroid, com, cell, origin, atom:N or three Cartesian numbers x,y,z (Å)"


def cell_center(cell, pbc, cart, origin=None) -> np.ndarray:
    """The middle of the cell, (½, ½, ½); along a slab's, polymer's or molecule's non-periodic axes, the middle of
    the drawing box the GUI's default Bounding box vacuum draws (the atoms' extent plus 1 Å each side, centered on
    them: geom/core.js boxCorners)."""
    C = np.asarray(cell, float)
    cart = np.asarray(cart, float).reshape(-1, 3)
    O = np.zeros(3) if origin is None else np.asarray(origin, float).copy()
    if C.shape != (3, 3) or abs(np.linalg.det(C)) < 1e-9:
        return (cart.min(axis=0) + cart.max(axis=0)) / 2 if len(cart) else np.zeros(3)
    V = C.copy()
    rec = np.linalg.inv(C).T
    for k in range(3):
        if pbc[k]:
            continue
        u = C[k] / np.linalg.norm(C[k])
        nu = rec[k] / np.linalg.norm(rec[k])
        cos = float(u @ nu) or 1.0
        h = cart @ nu if len(cart) else np.zeros(1)
        lo, hi = float(h.min()), float(h.max())
        L = hi - lo + 2 * BBOX_PAD
        O = O + u * ((lo + hi) / 2 - L / 2 - float(O @ nu)) / cos
        V[k] = u * L / cos
    return O + V.sum(axis=0) / 2


def pivot_point(about: str, payload: dict, idx: Sequence[int], pts: np.ndarray, fw: Optional[np.ndarray]):
    """(the pivot in Å, its name) of a turn of the atoms idx (pts: their Cartesian positions made whole, fw: their
    fractions, None without a cell), as the GUI's Rotate panel (viewport/gizmo_math.js pivotPoint):
      centroid  the atoms' centroid;
      com       their center of mass (standard atomic masses: ase.data.atomic_masses);
      cell      the middle of the cell (cell_center);
      origin    (0, 0, 0);
      atom:N    atom N (1-based; it need not be picked): where the picked atoms have it, else its image nearest
                their centroid (periodic axes only);
      x,y,z     that point (Å)."""
    key = str(about).strip().lower()
    cell = np.asarray(payload["cell"], float)
    pbc = [bool(p) for p in payload.get("pbc", (True, True, True))]
    if key == "centroid":
        return pts.mean(axis=0), "the centroid"
    if key == "com":
        from ase.data import atomic_masses
        m = np.array([float(atomic_masses[int(payload["numbers"][i]) % 100]) for i in idx])
        return (pts * m[:, None]).sum(axis=0) / m.sum(), "the center of mass"
    if key == "cell":
        return cell_center(cell, pbc, payload["cart"], payload.get("origin")), "the cell center"
    if key == "origin":
        return np.zeros(3), "the origin"
    m = re.fullmatch(r"atom:(\d+)", key)
    if m:
        cart = np.asarray(payload["cart"], float)
        n, i = len(cart), int(m.group(1)) - 1
        if not 0 <= i < n:
            raise TransformError(f"--about {about}: the structure has atoms 1 to {n}")
        symbols = payload.get("symbols") or _symbols(payload["numbers"])
        name = f"{symbols[i]}{i + 1}"
        if i in idx:
            return pts[list(idx).index(i)].copy(), name
        if fw is not None:
            frac = np.asarray(payload["frac"], float) if payload.get("frac") is not None else cart @ np.linalg.inv(cell)
            fc = fw.mean(axis=0)
            d = frac[i] - fc
            d = np.where(pbc, d - np.floor(d + 0.5), d)
            return (fc + d) @ cell, name
        return cart[i].copy(), name
    try:
        point = np.array([float(x) for x in re.split(r"[\s,;]+", key) if x], float)
    except ValueError:
        point = np.zeros(0)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise TransformError(f"--about: {ABOUT_HELP}")
    return point, "the point"


def rotation_matrix(axis: np.ndarray, deg: float) -> np.ndarray:
    """The right-handed turn by deg about the unit axis (counterclockwise seen from its tip)."""
    u = np.asarray(axis, float) / np.linalg.norm(axis)
    a = math.radians(deg)
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * (K @ K)


def transform(payload: dict, atoms: Optional[str] = None, rotate: float = 0.0, axis: Optional[str] = None,
              about: str = "centroid", translate=None, translate_frac=None, wrap: bool = True) -> dict:
    """The payload with the picked atoms turned (about the pivot), then displaced. Returns a new payload dict
    (cell, frac, cart, numbers, pbc, symbols) and a summary."""
    cell = np.asarray(payload["cell"], float)
    pbc = [bool(p) for p in payload.get("pbc", (True, True, True))]
    usable = _usable(cell, pbc)
    cart = np.asarray(payload["cart"], float)
    if payload.get("frac") is not None and usable:
        frac = np.asarray(payload["frac"], float)
    elif usable:
        frac = cart @ np.linalg.inv(cell)
    else:
        frac = None
    symbols = payload.get("symbols") or _symbols(payload["numbers"])
    idx = parse_atoms(atoms, symbols)
    if not idx:
        raise TransformError("no atoms picked")
    if usable:
        fw = whole(frac, cell, pbc, idx)
        pts = fw @ cell
    else:
        fw = None
        pts = cart[idx].copy()
    new = pts.copy()
    summary = []
    if rotate:
        if axis is None:
            raise TransformError("--rotate needs --axis")
        u = axis_vector(axis, cell, pbc)
        pivot, pname = pivot_point(about, {**payload, "pbc": pbc, "symbols": symbols, "frac": frac}, idx, pts, fw)
        R = rotation_matrix(u, rotate)
        new = (new - pivot) @ R.T + pivot
        summary.append(f"turned {len(idx)} atom{'s' * (len(idx) != 1)} by {rotate:g}° about {axis} "
                       f"({u[0]:.4f}, {u[1]:.4f}, {u[2]:.4f}) through {pname} at ({pivot[0]:.4f}, {pivot[1]:.4f}, {pivot[2]:.4f}) Å")
    if translate is not None and translate_frac is not None:
        raise TransformError("give --translate (Å) or --translate-frac, not both")
    if translate is not None:
        d = np.asarray(translate, float)
        new = new + d
        summary.append(f"moved them by ({d[0]:g}, {d[1]:g}, {d[2]:g}) Å")
    if translate_frac is not None:
        if not usable:
            raise TransformError("--translate-frac needs a cell")
        df = np.asarray(translate_frac, float)
        new = new + df @ cell
        summary.append(f"moved them by ({df[0]:g}, {df[1]:g}, {df[2]:g}) of a, b, c")
    if not summary:
        raise TransformError("nothing to do: give --rotate/--axis and/or --translate/--translate-frac")
    out_cart = cart.copy()
    if usable:
        # each atom keeps the cell image it had: its displacement added to where it was
        disp = (new - pts) @ np.linalg.inv(cell)
        out_frac = frac.copy()
        out_frac[idx] = frac[idx] + disp
        if wrap:
            for i in idx:
                for k in range(3):
                    if pbc[k]:
                        out_frac[i, k] -= math.floor(out_frac[i, k] + 1e-12)
                        if abs(out_frac[i, k] - 1.0) < 1e-12:
                            out_frac[i, k] = 0.0
        out_cart = out_frac @ cell
    else:
        out_cart[idx] = new
        out_frac = None
    return {"cell": cell.tolist(), "frac": None if out_frac is None else out_frac.tolist(), "cart": out_cart.tolist(),
            "numbers": list(payload["numbers"]), "pbc": pbc, "symbols": symbols, "indices": idx,
            "summary": "; ".join(summary)}


def _symbols(numbers) -> List[str]:
    from ase.data import chemical_symbols
    return [chemical_symbols[int(n) % 100] for n in numbers]


def to_atoms(p: dict):
    from ase import Atoms
    return Atoms(symbols=p["symbols"], positions=p["cart"], cell=p["cell"], pbc=p["pbc"])


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="mace transform", usage=USAGE,
                                 description="Rotate and/or translate atoms of a structure (the GUI's Rotate and Move "
                                             "tools from the command line). The rotation comes first.")
    ap.add_argument("input", help="a CIF, D12, CRYSTAL output or any structure file the GUI opens")
    ap.add_argument("--atoms", default=None, help='1-based indices, ranges and elements: "1,5-8", "O", "all" (default all)')
    ap.add_argument("--rotate", type=float, default=0.0, metavar="DEG", help="turn by DEG degrees (counterclockwise seen from the axis's tip)")
    ap.add_argument("--axis", default=None, help='x, y, z, a, b, c, a*, b*, c*, or "u v w" (a lattice direction; Cartesian for a molecule)')
    ap.add_argument("--about", default="centroid", metavar="PIVOT",
                    help="the pivot of --rotate, as the GUI's Rotate panel: centroid (of the atoms, made whole across the "
                         "cell's faces; the default), com (their center of mass), cell (the cell's center; the middle of "
                         "the atoms along a slab's or molecule's open directions), origin, atom:N (atom N, 1-based; it "
                         "need not be in --atoms) or x,y,z in Å")
    ap.add_argument("--translate", type=float, nargs=3, metavar=("DX", "DY", "DZ"), default=None, help="then move by this many Å")
    ap.add_argument("--translate-frac", type=float, nargs=3, metavar=("DA", "DB", "DC"), default=None,
                    help="then move by these fractions of the cell axes")
    ap.add_argument("-o", "--output", default=None, help="the file to write (default <stem>_transformed.cif; the format from its extension)")
    ap.add_argument("--no-wrap", action="store_true", help="keep moved atoms where they land instead of wrapping them into the cell")
    ap.add_argument("--force", action="store_true", help="overwrite OUTPUT")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    src = Path(args.input)
    if not src.is_file():
        print(f"Error: {src}: no such file", file=sys.stderr)
        return 2
    out = Path(args.output) if args.output else src.with_name(f"{src.stem}_transformed.cif")
    if out.exists() and not args.force:
        print(f"Error: {out} exists (use --force to overwrite)", file=sys.stderr)
        return 2
    try:
        from mace.gui.structures import read_structure
        payload = read_structure(src)
        r = transform(payload, atoms=args.atoms, rotate=args.rotate, axis=args.axis, about=args.about,
                      translate=args.translate, translate_frac=args.translate_frac, wrap=not args.no_wrap)
    except TransformError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except (ValueError, OSError) as exc:
        print(f"Error: {src}: {exc}", file=sys.stderr)
        return 2
    from ase.io import write
    atoms = to_atoms(r)
    if not all(atoms.pbc) and not out.name.lower().endswith((".xyz", ".extxyz")):
        # a slab, polymer or molecule in a format with no pbc (CIF, POSCAR): a box with vacuum, as mace supercell
        from mace.gui.structures import boxed
        atoms = boxed(atoms)
    elif not all(atoms.pbc):          # an extended XYZ keeps the pbc, with the vacuum vectors as the cell
        from mace.gui.structures import nonperiodic_info, vacuum_cell
        atoms.info.update(nonperiodic_info(payload))
        atoms = vacuum_cell(atoms)
    write(str(out), atoms)
    print(f"{src.name}: {r['summary']}")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
