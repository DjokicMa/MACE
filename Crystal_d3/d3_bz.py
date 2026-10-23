"""The first Brillouin zone of a CRYSTAL cell and the k-path in it.

One geometry for ``mace plotting --bz`` and the Brillouin zone view of
``mace gui`` (the k-path editor's). The zone is the Wigner-Seitz cell of the
reciprocal lattice: the Voronoi cell of the origin (scipy.spatial.Voronoi)
among the lattice points nearest it. A slab's zone is the 2D one of its two
periodic vectors, a polymer's the segment [-b1/2, b1/2].

k-points are fractional in the basis of CRYSTAL's primitive reciprocal
vectors, the basis of the BAND deck's segment records; reciprocal vectors
include 2*pi and are in 1/Angstrom. The high-symmetry points are the path's
own (from the deck, so the labels are the band plot's) and, for a 3D cell
with seekpath, the other points seekpath names for the lattice, rewritten in
CRYSTAL's basis as the deck's are (seekpath_interface.get_point_coordinates).
"""
from __future__ import annotations

import contextlib
import io
import itertools
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def reciprocal(cell: Sequence[Sequence[float]]) -> np.ndarray:
    """Rows b1, b2, b3 with a_i . b_j = 2*pi delta_ij."""
    a = np.asarray(cell, dtype=float)
    return 2 * np.pi * np.linalg.inv(a).T


def _order_polygon(pts: np.ndarray, normal: np.ndarray) -> List[int]:
    c = pts.mean(axis=0)
    u = pts[0] - c
    if np.linalg.norm(u) < 1e-12:
        u = pts[1] - c
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    ang = [np.arctan2(np.dot(p - c, v), np.dot(p - c, u)) for p in pts]
    return list(np.argsort(ang))


def _reduced(b: np.ndarray) -> np.ndarray:
    """A shorter basis of the same lattice (pairwise Gauss reduction, a few
    passes): the Voronoi cell needs the lattice points nearest the origin."""
    b = b.copy()
    for _ in range(12):
        changed = False
        for i, j in itertools.permutations(range(len(b)), 2):
            m = round(np.dot(b[i], b[j]) / np.dot(b[j], b[j]))
            if m:
                b[i] = b[i] - m * b[j]
                changed = True
        if not changed:
            break
    return b


def brillouin_zone(cell: Sequence[Sequence[float]], dim: int = 3) -> Dict[str, Any]:
    """{"dim", "recip": rows b1..b3, "vertices": [[x,y,z]], "faces": [[i...]] (ordered
    polygons, 3D), "edges": [[i, j]]}, in 1/Angstrom."""
    from scipy.spatial import Voronoi

    rec = reciprocal(cell)
    out: Dict[str, Any] = {"dim": dim, "recip": rec.tolist()}
    if dim == 1:
        v = np.array([-rec[0] / 2, rec[0] / 2])
        out.update(vertices=v.tolist(), faces=[], edges=[[0, 1]])
        return out
    if dim == 2:
        b = rec[:2]
        e1 = b[0] / np.linalg.norm(b[0])
        n = np.cross(b[0], b[1])
        n = n / np.linalg.norm(n)
        e2 = np.cross(n, e1)
        to2 = lambda p: np.array([np.dot(p, e1), np.dot(p, e2)])   # noqa: E731
        basis = _reduced(np.array([to2(b[0]), to2(b[1])]))
        pts = np.array([i * basis[0] + j * basis[1] for i in range(-3, 4) for j in range(-3, 4)])
        vor = Voronoi(pts)
        origin = int(np.argmin(np.linalg.norm(pts, axis=1)))
        region = vor.regions[vor.point_region[origin]]
        poly = np.array([vor.vertices[k] for k in region])
        order = np.argsort(np.arctan2(poly[:, 1], poly[:, 0]))
        verts = [(p[0] * e1 + p[1] * e2) for p in poly[order]]
        m = len(verts)
        out.update(vertices=[list(map(float, v)) for v in verts], faces=[list(range(m))],
                   edges=[[k, (k + 1) % m] for k in range(m)])
        return out
    basis = _reduced(rec)
    rng = range(-2, 3)
    pts = np.array([i * basis[0] + j * basis[1] + k * basis[2] for i in rng for j in rng for k in rng])
    vor = Voronoi(pts)
    origin = int(np.argmin(np.linalg.norm(pts, axis=1)))
    used: Dict[int, int] = {}
    verts: List[List[float]] = []
    faces: List[List[int]] = []
    for (p, q), ridge in zip(vor.ridge_points, vor.ridge_vertices):
        if origin not in (p, q) or -1 in ridge:
            continue
        other = q if p == origin else p
        normal = pts[other] - pts[origin]
        normal = normal / np.linalg.norm(normal)
        poly = np.array([vor.vertices[k] for k in ridge])
        face = []
        for k in np.array(ridge)[_order_polygon(poly, normal)]:
            if k not in used:
                used[k] = len(verts)
                verts.append([float(x) for x in vor.vertices[k]])
            face.append(used[k])
        faces.append(face)
    edges = sorted({tuple(sorted((f[k], f[(k + 1) % len(f)]))) for f in faces for k in range(len(f))})
    out.update(vertices=verts, faces=faces, edges=[list(e) for e in edges])
    return out


def cartesian(k: Sequence[float], recip: Sequence[Sequence[float]]) -> List[float]:
    return [float(x) for x in np.asarray(k, dtype=float) @ np.asarray(recip, dtype=float)]


def inside(cart: Sequence[float], recip: Sequence[Sequence[float]], dim: int = 3, tol: float = 1e-6) -> bool:
    """Whether a Cartesian k lies in the closed first zone (no reciprocal lattice point is nearer)."""
    rec = np.asarray(recip, dtype=float)[:dim]
    k = np.asarray(cart, dtype=float)
    for n in itertools.product(range(-2, 3), repeat=dim):
        if not any(n):
            continue
        g = np.asarray(n, dtype=float) @ rec
        if np.dot(k, g) > np.dot(g, g) / 2 + tol * max(1.0, np.dot(g, g)):
            return False
    return True


# ---------------------------------------------------------------- CRYSTAL outputs
def structure_of(out_file: str) -> Optional[Dict[str, Any]]:
    """The output's cell (rows, Angstrom) and dimensionality, as the k-path code reads them."""
    import seekpath_interface
    import d3_kpoints
    with contextlib.redirect_stdout(io.StringIO()):
        st = seekpath_interface.parse_crystal_structure(str(out_file))
        dim = d3_kpoints.output_dimensionality(str(out_file))
    if st is None or "cell" not in st:
        cell = printed_cell(out_file)
        return None if cell is None else {"cell": cell, "dim": _dim_of(cell, dim), "structure": None}
    return {"cell": np.asarray(st["cell"], dtype=float), "dim": _dim_of(st["cell"], dim), "structure": st}


def _dim_of(cell, dim) -> int:
    """The output's dimensionality; a properties run (BAND, DOSS) does not print SLAB or POLYMER
    CALCULATION, but its lattice vectors carry CRYSTAL's formal 500 Å along each non-periodic axis
    (a slab's zone is 2D, a polymer's 1D)."""
    d = int(dim if dim is not None else 3)
    if d == 3:
        lengths = np.linalg.norm(np.asarray(cell, dtype=float), axis=1)
        n = int(np.sum(lengths < 499.0))
        if n < 3 and all(x >= 499.0 for x in lengths[n:]):
            return n
    return d


_VECTORS = ("DIRECT LATTICE VECTOR COMPONENTS (ANGSTROM)", "DIRECT LATTICE VECTORS CARTESIAN COMPONENTS (ANGSTROM)")


def printed_cell(out_file: str) -> Optional[np.ndarray]:
    """The last printed direct lattice vectors (a properties run's output prints
    them, with no geometry block to read the structure from)."""
    try:
        lines = Path(out_file).read_text(errors="replace").splitlines()
    except OSError:
        return None
    cell = None
    for n, line in enumerate(lines):
        if any(t in line for t in _VECTORS):
            rows = []
            for ln in lines[n + 1:n + 6]:
                parts = ln.split()
                try:
                    vals = [float(x) for x in parts]
                except ValueError:
                    continue
                if len(vals) == 3:
                    rows.append(vals)
                if len(rows) == 3:
                    break
            if len(rows) == 3:
                cell = np.array(rows)
    return cell


def seekpath_points(out_file: str) -> Dict[str, List[float]]:
    """seekpath's high-symmetry points of a 3D cell in CRYSTAL's reciprocal basis ({} without seekpath)."""
    try:
        import seekpath_interface
        with contextlib.redirect_stdout(io.StringIO()):
            pc, _ = seekpath_interface.get_point_coordinates(str(out_file))
        return {k: [float(x) for x in v] for k, v in pc.items()}
    except Exception:
        return {}


def bz_view(out_file: str, segments: Sequence[Dict[str, Any]], dim: Optional[int] = None) -> Dict[str, Any]:
    """Everything the BZ views draw: the zone, the points (label, fractional and
    Cartesian k, whether the path visits it) and the path's segments in Cartesian k.

    ``segments`` are d3_band_path segments with fractional k (a deck's, or the
    edited path resolved against it)."""
    import d3_band_path as kp
    s = structure_of(out_file)
    if s is None:
        raise ValueError(f"no cell in {Path(out_file).name}")
    d = dim or s["dim"]
    bz = brillouin_zone(s["cell"], d)
    rec = bz["recip"]
    pts: Dict[str, Dict[str, Any]] = {}
    for seg in segments:
        for end in ("from", "to"):
            k = seg.get(f"k_{end}")
            lab = kp.canonical_label(seg[end])
            if k is not None and lab not in pts:
                pts[lab] = {"k": [float(x) for x in k], "cart": cartesian(k, rec), "on_path": True}
    if d == 3:
        for lab, k in seekpath_points(out_file).items():
            lab = kp.canonical_label(lab)
            if lab not in pts:
                pts[lab] = {"k": k, "cart": cartesian(k, rec), "on_path": False}
    # a point outside the first zone is the same state as one on it (they differ by a reciprocal
    # lattice vector), but a path drawn to it leaves the zone: the views say so
    for p in pts.values():
        p["inside"] = inside(p["cart"], rec, d)
    path = []
    for seg in segments:
        if seg.get("k_from") is None or seg.get("k_to") is None:
            continue
        path.append({"from": kp.canonical_label(seg["from"]), "to": kp.canonical_label(seg["to"]),
                     "a": cartesian(seg["k_from"], rec), "b": cartesian(seg["k_to"], rec),
                     **({"hidden": True} if seg.get("hidden") else {})})
    return {"bz": bz, "points": pts, "path": path}


def output_for(data_file: str) -> Optional[Path]:
    """The CRYSTAL output of a BAND.DAT / .d3 / .out: the file itself, the run's
    .out beside it, or the SCF run it started from (<material>_sp/opt .out)."""
    p = Path(data_file)
    if p.suffix == ".out":
        return p
    name = p.name
    for suf in (".BAND.DAT", ".band.band.dat", "_band.dat", ".d3", ".f25"):
        if name.endswith(suf):
            name = name[: -len(suf)]
            break
    material = name[:-5] if name.endswith("_band") else name
    for cand in (f"{name}.out", f"{material}_band.out", f"{material}_sp.out", f"{material}_opt.out",
                 f"{material}.out", f"{material}_SP.out", f"{material}_OPT.out"):
        if (p.parent / cand).is_file():
            return p.parent / cand
    return None


# ---------------------------------------------------------------- a figure (mace plotting --bz)
def plot_bz(view: Dict[str, Any], title: str, out_stem: str, formats=("png", "svg"),
            elev: float = 24, azim: float = -62) -> List[str]:
    """The zone, its reciprocal axes, the labeled points and the path (arrows; hidden
    segments faint, jumps dashed) as a matplotlib 3D figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import d3_band_path as kp

    bz = view["bz"]
    V = np.array(bz["vertices"])
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    for i, j in bz["edges"]:
        ax.plot(*zip(V[i], V[j]), color="0.35", lw=1.0)
    rec = np.array(bz["recip"])
    for n, b in enumerate(rec[: bz["dim"]]):
        b = b * 0.62
        ax.quiver(0, 0, 0, *b, color="0.55", arrow_length_ratio=0.08, lw=0.8)
        ax.text(*(b * 1.06), f"$b_{n + 1}$", color="0.45", fontsize=9)
    prev = None
    for seg in view["path"]:
        a, b = np.array(seg["a"]), np.array(seg["b"])
        if prev is not None and (prev["to"] != seg["from"] or np.linalg.norm(np.array(prev["b"]) - a) > 1e-6):
            ax.plot(*zip(prev["b"], a), color="#2563eb", lw=0.9, ls="--", alpha=0.6)
        hidden = seg.get("hidden")
        ax.quiver(*a, *(b - a), color="#2563eb", alpha=0.25 if hidden else 1.0, lw=2.0,
                  arrow_length_ratio=0.12, linestyle=":" if hidden else "-")
        if not hidden:
            prev = seg
    for lab, p in view["points"].items():
        c = p["cart"]
        on = p["on_path"]
        ax.scatter(*c, color="#d97706" if on else "0.5", s=18 if on else 8, depthshade=False)
        if on or bz["dim"] == 3:
            ax.text(*c, " " + kp_label_tex(lab), fontsize=10 if on else 8, color="black" if on else "0.4")
    # each axis as long as the zone is (a flat zone stays flat), the reciprocal axes inside the frame
    ext = np.maximum(np.abs(np.vstack([V, rec[: bz["dim"]] * 0.7])).max(axis=0), 0.05 * np.abs(V).max()) * 1.12
    ax.set_xlim(-ext[0], ext[0]); ax.set_ylim(-ext[1], ext[1]); ax.set_zlim(-ext[2], ext[2])
    ax.set_box_aspect(tuple(ext))
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=11)
    written = []
    for fmt in formats:
        path = f"{out_stem}.BZ.{fmt}"
        fig.savefig(path, dpi=300 if fmt == "png" else None, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written


def kp_label_tex(label: str) -> str:
    """GAMMA -> $\\Gamma$, F_0 -> $F_{0}$, DELTA_0 -> $\\Delta_{0}$ (matplotlib mathtext)."""
    greek = {"GAMMA": r"\Gamma", "DELTA": r"\Delta", "SIGMA": r"\Sigma", "LAMBDA": r"\Lambda", "PI": r"\Pi"}
    base, _, sub = label.partition("_")
    base = greek.get(base, base.replace("'", r"^{\prime}"))
    return f"${base}_{{{sub}}}$" if sub else f"${base}$"
