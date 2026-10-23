"""Pore geometry of a crystal: MACE's own grid algorithm, and Zeo++ when it
is installed (mace.analysis.zeopp).

Model: hard-sphere atoms (radii.py: CCDC/Bondi or UFF) and a spherical
probe of radius r_p (an adsorbate's kinetic radius, PROBES). On a grid over
the cell, d(x) is the distance from x to the nearest atom *surface*
(|x - r_i| - R_i over the atom and all of its periodic images), computed
exactly with one k-d tree per distinct radius.

- Di, the largest included sphere: 2 max d, refined off-grid by a local
  maximization of the continuous d(x).
- The probe-center region C(r_p) = {x : d(x) >= r_p} is split into
  connected components with periodic boundaries; following each component
  across the cell faces gives the lattice vectors it repeats along. A
  component that connects to its own periodic image is a channel (its
  dimensionality is the rank of those vectors); one that does not is an
  isolated pocket the probe cannot reach from outside: inaccessible.
  Along a non-periodic direction (a slab, a molecule) a component that
  reaches the open boundary is accessible.
- Df, the largest free sphere: the largest r for which C(r) still has a
  channel (bisection), per direction a, b, c too; Dif is the largest
  included sphere along that free path.
- Volumes (Zeo++ conventions): AV/NAV, the probe-center accessible and
  non-accessible volumes (Zeo++ -vol); POAV/PONAV, the volume the probe
  sphere itself can occupy (Zeo++ -volpo); the geometric void fraction
  (probe 0).
- ASA/NASA, accessible surface area: points on spheres of radius R_i + r_p
  around each atom, kept when no other atom's sphere covers them, split by
  the component of C they touch (Zeo++ -sa).
- Geometric density from the cell mass.

Accuracy is set by the grid spacing (default 0.15 A, coarser for large
cells so the grid stays near ``max_points``): Di is refined off the grid,
Df and the volumes carry an error of about one spacing; tests compare
against Zeo++ and the IZA zeolite values.
"""
from __future__ import annotations

import base64
import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import mesh_quality
from .structure import Structure

# kinetic diameters (Breck, Zeolite Molecular Sieves, 1974), probe radius = half
PROBES: Dict[str, Dict[str, float]] = {
    "He": {"kinetic_diameter": 2.60, "label": "Helium"},
    "H2": {"kinetic_diameter": 2.89, "label": "Hydrogen"},
    "H2O": {"kinetic_diameter": 2.65, "label": "Water"},
    "CO2": {"kinetic_diameter": 3.30, "label": "Carbon dioxide"},
    "Ar": {"kinetic_diameter": 3.40, "label": "Argon"},
    "N2": {"kinetic_diameter": 3.64, "label": "Nitrogen"},
    "CH4": {"kinetic_diameter": 3.80, "label": "Methane"},
}
DEFAULT_PROBE = "N2"
DEFAULT_SPACING = 0.15
DEFAULT_MAX_POINTS = 4_000_000
DEFAULT_SURFACE_SAMPLES = 2000       # points per atom sphere for the surface area


def probe_radius(probe) -> Tuple[float, str]:
    """A preset name (N2, CO2, ...) or a radius in Angstrom -> (radius, label)."""
    if probe is None:
        probe = DEFAULT_PROBE
    if isinstance(probe, (int, float)):
        r = float(probe)
    else:
        s = str(probe).strip()
        key = {k.lower(): k for k in PROBES}.get(s.lower())
        if key:
            return PROBES[key]["kinetic_diameter"] / 2.0, key
        try:
            r = float(s)
        except ValueError:
            raise ValueError(f"unknown probe {probe!r}: one of {', '.join(PROBES)} or a radius in Angstrom") from None
    if not (0.0 <= r <= 10.0):
        raise ValueError("probe radius must be between 0 and 10 Angstrom")
    return r, f"{r:g} A"


def atom_radii(st: Structure, radii="ccdc", overrides: Optional[Dict[str, float]] = None) -> np.ndarray:
    from . import _radii_data as RD
    sym = st.symbols
    if isinstance(radii, dict):
        table = {k: float(v) for k, v in radii.items()}
    elif str(radii).lower() == "uff":
        table = {k: x / 2 ** (1 / 6) / 2 for k, x in RD.UFF_X.items()}
    elif str(radii).lower() == "ccdc":
        table = dict(RD.CCDC)
    else:
        raise ValueError("radii: ccdc, uff or a {element: radius} table")
    if overrides:
        table.update({k: float(v) for k, v in overrides.items()})
    missing = sorted({s for s in sym if s not in table})
    if missing:
        raise ValueError(f"no radius for {', '.join(missing)}; give one with --radius EL=R")
    return np.array([table[s] for s in sym], float)


# ---------------------------------------------------------------------------
# the distance field
# ---------------------------------------------------------------------------
class SurfaceDistance:
    """d(x) = min_i |x - r_i| - R_i over the atoms and their images; exact."""

    def __init__(self, st: Structure, radii: np.ndarray, pad: float = 8.0):
        self.st = st.wrapped()
        self.radii = np.asarray(radii, float)
        self.inv = np.linalg.inv(self.st.cell)
        self.groups = sorted(set(np.round(self.radii, 6)))
        self._build(pad)

    def _build(self, pad: float):
        from scipy.spatial import cKDTree
        self.pad = pad
        cart, idx, _ = self.st.images(pad)
        self.trees = []
        for r in self.groups:
            m = np.isclose(self.radii[idx], r, atol=1e-6)
            self.trees.append((float(r), cKDTree(cart[m]) if m.any() else None))

    def wrap_frac(self, frac: np.ndarray) -> np.ndarray:
        f = np.array(frac, float)
        for i in range(3):
            if self.st.pbc[i]:
                f[..., i] -= np.floor(f[..., i])
        return f

    def __call__(self, cart: np.ndarray, frac_in_cell: bool = False) -> np.ndarray:
        """d at Cartesian points (wrapped into the home cell first)."""
        cart = np.atleast_2d(cart)
        if not frac_in_cell:
            cart = self.wrap_frac(cart @ self.inv) @ self.st.cell
        while True:
            best = np.full(len(cart), np.inf)
            nearest = np.full(len(cart), np.inf)
            for r, tree in self.trees:
                if tree is None:
                    continue
                dist, _ = tree.query(cart, k=1, workers=1)
                np.minimum(best, dist - r, out=best)
                np.minimum(nearest, dist, out=nearest)
            # a nearest center farther than the padding may be beaten by an image outside it
            bad = nearest > self.pad - 1e-9
            if not bad.any() or self.pad > 200:
                return best
            self._build(self.pad * 2)


def open_box(st: Structure, pad: float) -> Tuple[Structure, bool]:
    """A slab, polymer or molecule in a box that holds every atom with at
    least ``pad`` of vacuum along each non-periodic direction (the grid
    covers only the box). A CRYSTAL slab or polymer comes with a dummy
    vector there (0.5 A, or atoms far outside it): the box is rebuilt
    from the atoms, perpendicular to the periodic directions, and its
    corner kept in meta["origin"] so Cartesian output (the pore surfaces)
    lands on the atoms. A box that already holds the atoms is kept.
    Returns (structure, rebuilt)."""
    if all(st.pbc):
        return st, False
    cell = np.array(st.cell, float)
    origin = np.array(st.meta.get("origin", (0.0, 0.0, 0.0)), float)
    open_ax = [k for k in range(3) if not st.pbc[k]]
    if abs(np.linalg.det(cell)) > 1e-6:
        widths = 1.0 / np.linalg.norm(np.linalg.inv(cell).T, axis=1)
        f = st.frac
        if all(f[:, k].min() * widths[k] >= pad - 1e-6 and (1.0 - f[:, k].max()) * widths[k] >= pad - 1e-6
               for k in open_ax):
            return st, False
    cart = st.frac @ cell + origin
    per = [k for k in range(3) if st.pbc[k]]
    # unit directions for the open axes, perpendicular to the periodic ones
    if len(per) == 2:
        n = np.cross(cell[per[0]], cell[per[1]])
        dirs = [n / np.linalg.norm(n)]
    elif len(per) == 1:
        a = cell[per[0]] / np.linalg.norm(cell[per[0]])
        e1 = np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0])
        e1 /= np.linalg.norm(e1)
        dirs = [e1, np.cross(a, e1)]
    else:
        dirs = [np.eye(3)[k] for k in range(3)]
    new = cell.copy()
    o = np.zeros(3)
    for k, u in zip(open_ax, dirs):
        p = cart @ u
        lo, hi = float(p.min()) - pad, float(p.max()) + pad
        new[k] = u * (hi - lo)
        o += u * lo
    if np.linalg.det(new) < 0:            # keep a right-handed box
        k = open_ax[-1]
        o += new[k]
        new[k] = -new[k]
    frac = np.linalg.solve(new.T, (cart - o).T).T
    meta = dict(st.meta)
    meta["origin"] = o.tolist()
    return Structure(new, frac, st.numbers, st.pbc, st.name, meta), True


def _grid_shape(st: Structure, spacing: float, max_points: int) -> Tuple[Tuple[int, int, int], float]:
    lengths = np.linalg.norm(st.cell, axis=1)
    h = float(spacing)
    while True:
        shape = tuple(max(8, int(math.ceil(L / h))) for L in lengths)
        if np.prod(shape, dtype=float) <= max_points or h > 2.0:
            return shape, h
        h *= 1.1


def _grid_frac(shape) -> np.ndarray:
    axes = [np.arange(n) / n for n in shape]
    g = np.stack(np.meshgrid(*axes, indexing="ij"), -1)
    return g.reshape(-1, 3)


# ---------------------------------------------------------------------------
# periodic connected components
# ---------------------------------------------------------------------------
class Components:
    """Connected components of a boolean grid with periodic faces where the
    structure is periodic. ``dim[c]``: the rank of the lattice vectors along
    which component c repeats (0 = a closed pocket); ``open[c]``: it reaches
    a non-periodic boundary; ``axes[c]``: the cell axes it percolates along."""

    def __init__(self, mask: np.ndarray, pbc: Sequence[bool]):
        from scipy import ndimage
        lab, n = ndimage.label(mask)
        self.raw = lab
        self.n_raw = n
        parent = np.arange(n + 1)
        off = np.zeros((n + 1, 3), int)
        adj: Dict[int, List[Tuple[int, np.ndarray]]] = {}
        self_cycles: Dict[int, List[np.ndarray]] = {}
        for ax in range(3):
            if not pbc[ax] or mask.shape[ax] < 2:
                continue
            a = np.take(lab, -1, axis=ax).ravel()
            b = np.take(lab, 0, axis=ax).ravel()
            m = (a > 0) & (b > 0)
            if not m.any():
                continue
            pairs = np.unique(np.stack([a[m], b[m]], 1), axis=0)
            e = np.zeros(3, int)
            e[ax] = 1
            for x, y in pairs:
                x, y = int(x), int(y)
                adj.setdefault(x, []).append((y, e))
                adj.setdefault(y, []).append((x, -e))
        # traverse: offsets, roots, cycle vectors
        root = np.zeros(n + 1, int)
        seen = np.zeros(n + 1, bool)
        cycles: Dict[int, List[np.ndarray]] = {}
        for s in range(1, n + 1):
            if seen[s]:
                continue
            seen[s] = True
            root[s] = s
            stack = [s]
            cyc = []
            while stack:
                u = stack.pop()
                for v, e in adj.get(u, ()):
                    if not seen[v]:
                        seen[v] = True
                        root[v] = s
                        off[v] = off[u] + e
                        stack.append(v)
                    else:
                        c = off[u] + e - off[v]
                        if c.any():
                            cyc.append(c)
            cycles[s] = cyc
        self.root = root
        roots = sorted(cycles)
        self.roots = roots
        self.dim: Dict[int, int] = {}
        self.axes: Dict[int, Tuple[bool, bool, bool]] = {}
        for r in roots:
            cv = np.array(cycles[r]) if cycles[r] else np.zeros((0, 3), int)
            self.dim[r] = int(np.linalg.matrix_rank(cv)) if len(cv) else 0
            self.axes[r] = tuple(bool(len(cv) and np.any(cv[:, i] != 0)) for i in range(3))
        # touching an open (non-periodic) boundary
        self.open: Dict[int, bool] = {r: False for r in roots}
        for ax in range(3):
            if pbc[ax]:
                continue
            for face in (np.take(lab, 0, axis=ax), np.take(lab, -1, axis=ax)):
                for l in np.unique(face[face > 0]):
                    self.open[int(root[l])] = True
        self.label = root[lab]          # every voxel -> its component root (0 = not in the mask)

    def accessible_roots(self) -> List[int]:
        return [r for r in self.roots if self.dim[r] > 0 or self.open[r]]


# ---------------------------------------------------------------------------
# the Voronoi network: Df and accessibility off the grid
# ---------------------------------------------------------------------------
class _UF:
    """Union-find on network nodes that also tracks the lattice offset of each
    node to its root, so joining a component to its own image is seen as a
    cycle vector (the component repeats along it)."""

    def __init__(self, n):
        self.p = np.arange(n)
        self.o = np.zeros((n, 3), int)
        self.cyc = {}

    def find(self, x):
        path = []
        while self.p[x] != x:
            path.append(x)
            x = self.p[x]
        root = x
        # compress, accumulating offsets from the top of the path down
        acc = np.zeros(3, int)
        for y in reversed(path):
            acc = acc + self.o[y]
            self.o[y] = acc
            self.p[y] = root
        return root

    def union(self, u, v, s):
        ru, rv = self.find(u), self.find(v)
        ou, ov = self.o[u] if u != ru else np.zeros(3, int), self.o[v] if v != rv else np.zeros(3, int)
        if ru == rv:
            c = ou + s - ov
            if c.any():
                b = self.cyc.setdefault(ru, [])
                if np.linalg.matrix_rank(np.array(b + [c])) > len(b):
                    b.append(c)
                    return c
            return None
        self.p[rv] = ru
        self.o[rv] = ou + s - ov
        b = self.cyc.pop(rv, [])
        new = None
        for c in b:
            mine = self.cyc.setdefault(ru, [])
            if np.linalg.matrix_rank(np.array(mine + [c])) > len(mine):
                mine.append(c)
                new = c
        return new

    def basis(self, x):
        return self.cyc.get(self.find(x), [])


class VoronoiNetwork:
    """Nodes and edges of the Voronoi decomposition of the atom centers
    (with periodic images), mapped back into the cell. For atoms of equal
    radii the edges are the paths of largest clearance, so the largest free
    sphere follows exactly; for unequal radii each edge's narrowest point is
    re-centered in its window (a 2D maximization of d), which recovers most
    of the difference (Zeo++ uses its -ha sphere subdivision for this)."""

    def __init__(self, st: Structure, field: SurfaceDistance, pad: float, refine_window: float = 0.4):
        from scipy.spatial import Voronoi, cKDTree
        if not all(st.pbc):
            raise ValueError("the Voronoi network is for 3D crystals")
        self.st, self.field = st, field
        inv = np.linalg.inv(st.cell)
        widths = 1.0 / np.linalg.norm(inv.T, axis=1)
        # an edge can run a whole cell before its far end: images out to one
        # cell width plus the clearance, so that far vertex is exact too
        pad = min(max(pad, float(widths.max()) + pad), 60.0)
        cart, _, _ = st.images(pad)
        vor = Voronoi(cart)
        fv = vor.vertices @ inv
        # a vertex is exact when every atom within its circumsphere is among the images
        rho, _ = cKDTree(cart).query(vor.vertices, k=1)
        out = np.maximum(0.0, np.maximum(-fv, fv - 1)) * widths
        ok = np.linalg.norm(out, axis=1) + rho < pad - 1e-6
        shift = np.floor(fv + 1e-9).astype(int)
        wrapped = fv - shift
        wrapped[wrapped >= 1 - 1e-7] = 0.0
        key = np.round(wrapped * 1e5).astype(np.int64)
        key %= 100000
        uniq, inverse = np.unique(key, axis=0, return_inverse=True)
        inverse = inverse.ravel()
        self.node_frac = uniq / 1e5
        self.node_d = field(self.node_frac @ st.cell, frac_in_cell=True)
        # edges from the ridge polygons
        ea, eb = [], []
        # qhull does not promise the vertices of a ridge in cyclic order, so
        # every pair of a face's vertices is joined: the true polygon edges
        # are among them, and a chord is a real (narrower) path, never a wrong one
        for poly in vor.ridge_vertices:
            q = [x for x in poly if x >= 0 and ok[x]]
            k = len(q)
            for i in range(k):
                for j in range(i + 1, k):
                    ea.append(q[i])
                    eb.append(q[j])
        ea, eb = np.array(ea, int), np.array(eb, int)
        keep = ok[ea] & ok[eb] & (np.all(shift[ea] == 0, axis=1) | np.all(shift[eb] == 0, axis=1))
        ea, eb = ea[keep], eb[keep]
        u, v = inverse[ea], inverse[eb]
        s = shift[eb] - shift[ea]
        # canonical direction and dedupe
        first_nz = np.take_along_axis(s, np.argmax(s != 0, axis=1)[:, None], 1).ravel()
        flip = (u > v) | ((u == v) & (first_nz < 0))
        u2 = np.where(flip, v, u)
        v2 = np.where(flip, u, v)
        s2 = np.where(flip[:, None], -s, s)
        rec = np.unique(np.column_stack([u2, v2, s2]), axis=0)
        rec = rec[~((rec[:, 0] == rec[:, 1]) & ~rec[:, 2:].any(1))]
        self.eu, self.ev, self.es = rec[:, 0], rec[:, 1], rec[:, 2:5]
        pu = self.node_frac[self.eu] @ st.cell
        pv = (self.node_frac[self.ev] + self.es) @ st.cell
        self.cap, self.mid = self._segment_min(pu, pv)
        self.refine_window = refine_window
        self.refined = 0
        self._done = np.zeros(len(self.cap), bool)

    def _segment_min(self, a, b, n=24, iters=24):
        """min of d along straight segments a->b (vectorized golden section)."""
        t = np.linspace(0, 1, n)
        pts = a[:, None, :] + (b - a)[:, None, :] * t[None, :, None]
        dd = self.field(pts.reshape(-1, 3)).reshape(len(a), n)
        k = np.argmin(dd, axis=1)
        lo = t[np.maximum(k - 1, 0)]
        hi = t[np.minimum(k + 1, n - 1)]
        g = (5 ** 0.5 - 1) / 2
        f = lambda tt: self.field(a + (b - a) * tt[:, None])
        x1, x2 = hi - g * (hi - lo), lo + g * (hi - lo)
        f1, f2 = f(x1), f(x2)
        for _ in range(iters):
            left = f1 < f2                       # the minimum is in [lo, x2]
            nlo = np.where(left, lo, x1)
            nhi = np.where(left, x2, hi)
            nx1 = np.where(left, nhi - g * (nhi - nlo), x2)
            nx2 = np.where(left, x1, nlo + g * (nhi - nlo))
            fnew = f(np.where(left, nx1, nx2))
            f1, f2 = np.where(left, fnew, f2), np.where(left, f1, fnew)
            lo, hi, x1, x2 = nlo, nhi, nx1, nx2
        tm = 0.5 * (lo + hi)
        vals = np.minimum(f(tm), dd.min(1))
        return vals, a + (b - a) * tm[:, None]

    def refine(self, below: float, limit: int = 4000):
        """Re-center the narrowest point of each edge whose clearance is just
        under ``below`` (within refine_window) in its window plane, and keep the
        better of the straight and bent paths: only those edges can change a
        decision taken at ``below``."""
        from scipy.optimize import minimize
        idx = np.flatnonzero((self.cap >= below - self.refine_window) & (self.cap < below) & ~self._done)
        idx = idx[np.argsort(-self.cap[idx])][:limit]
        self._done[idx] = True
        st = self.st
        for e in idx:
            pu = self.node_frac[self.eu[e]] @ st.cell
            pv = (self.node_frac[self.ev[e]] + self.es[e]) @ st.cell
            t = pv - pu
            L = np.linalg.norm(t)
            if L < 1e-6:
                continue
            t /= L
            e1 = np.cross(t, [1.0, 0, 0] if abs(t[0]) < 0.9 else [0, 1.0, 0])
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(t, e1)
            b0 = self.mid[e]
            g = lambda x: -float(self.field((b0 + x[0] * e1 + x[1] * e2)[None, :])[0])
            r = minimize(g, np.zeros(2), method="Nelder-Mead",
                         options={"xatol": 1e-4, "fatol": 1e-6, "initial_simplex": [[0, 0], [0.1, 0], [0, 0.1]]})
            b1 = b0 + r.x[0] * e1 + r.x[1] * e2
            c1, _ = self._segment_min(pu[None, :], b1[None, :])
            c2, _ = self._segment_min(b1[None, :], pv[None, :])
            c = min(float(c1[0]), float(c2[0]))
            if c > self.cap[e]:
                self.cap[e] = c
                self.refined += 1

    def percolation(self):
        """Edges by decreasing capacity: the first that closes a cycle with a
        nonzero lattice vector gives the free sphere; per axis the same."""
        order = np.argsort(-self.cap)
        uf = _UF(len(self.node_d))
        first, axes = None, [None, None, None]
        for e in order:
            c = uf.union(int(self.eu[e]), int(self.ev[e]), self.es[e])
            if c is None:
                continue
            cap = float(self.cap[e])
            if first is None:
                first = (cap, int(self.eu[e]))
            for i in range(3):
                if axes[i] is None and c[i] != 0:
                    axes[i] = cap
            if all(x is not None for x in axes):
                break
        return first, axes

    def core(self, r: float) -> np.ndarray:
        """Nodes of the subnetwork with clearance >= r left after pruning dead
        ends (nodes with one edge, repeatedly): the part a free sphere passes
        through rather than visits."""
        m = self.cap >= r
        eu, ev = self.eu[m], self.ev[m]
        n = len(self.node_d)
        alive = np.ones(n, bool)
        while True:
            em = alive[eu] & alive[ev]
            deg = np.bincount(eu[em], minlength=n) + np.bincount(ev[em], minlength=n)
            dead = alive & (deg <= 1)
            if not dead.any():
                break
            alive &= ~dead
        deg = np.bincount(eu[alive[eu] & alive[ev]], minlength=n) + np.bincount(ev[alive[eu] & alive[ev]], minlength=n)
        return alive & (deg >= 2)

    def components(self, r: float):
        """Union-find of the subnetwork with clearance >= r: (root per node,
        {root: cycle basis})."""
        uf = _UF(len(self.node_d))
        for e in np.flatnonzero(self.cap >= r):
            uf.union(int(self.eu[e]), int(self.ev[e]), self.es[e])
        roots = np.array([uf.find(i) for i in range(len(self.node_d))])
        return roots, {int(k): v for k, v in uf.cyc.items()}


# ---------------------------------------------------------------------------
# the analysis
# ---------------------------------------------------------------------------
def _fib_sphere(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


def _refine_max(field: SurfaceDistance, start: np.ndarray, step: float) -> Tuple[float, np.ndarray]:
    from scipy.optimize import minimize
    st = field.st
    open_ax = [k for k in range(3) if not st.pbc[k]]

    def f(x):
        # along a non-periodic direction the search stays in the box: outside
        # it d only grows (open vacuum), and the maximum would run away
        if open_ax:
            fr = x @ field.inv
            fr[open_ax] = np.clip(fr[open_ax], 0.0, 1.0)
            x = fr @ st.cell
        return -float(field(x[None, :])[0])
    best_x, best = start, -f(start)
    try:
        r = minimize(f, start, method="Nelder-Mead",
                     options={"xatol": 1e-4, "fatol": 1e-6, "initial_simplex": start + np.vstack([np.zeros(3), np.eye(3) * step])})
        if -r.fun > best:
            best, best_x = -r.fun, r.x
            if open_ax:
                fr = best_x @ field.inv
                fr[open_ax] = np.clip(fr[open_ax], 0.0, 1.0)
                best_x = fr @ st.cell
    except Exception:          # pragma: no cover - scipy failure keeps the grid value
        pass
    return best, best_x


def _grid_free_sphere(st, d3, h, chk):
    """Df, Df per axis and Dif from percolation of the grid (bisection)."""
    def perco(r):
        c = Components(d3 >= r, st.pbc)
        roots = c.accessible_roots()
        axes = [any(c.axes[x][i] for x in roots) for i in range(3)]
        return bool(roots), axes, c, roots

    dmax = float(d3.max())
    steps = int(math.ceil(math.log2(max(dmax, 1e-3) / (h / 8)))) + 1
    ok0, at_zero, _, _ = perco(0.0)
    if not ok0:
        return 0.0, [0.0, 0.0, 0.0], 0.0
    lo, hi = 0.0, dmax
    for _ in range(steps):
        chk()
        mid = 0.5 * (lo + hi)
        if perco(mid)[0]:
            lo = mid
        else:
            hi = mid
    df_r = lo
    _, at_df, cdf, rdf = perco(df_r)
    axes = [0.0, 0.0, 0.0]
    for i in range(3):
        if not st.pbc[i] or not at_zero[i]:
            continue
        if at_df[i]:
            axes[i] = 2 * df_r
            continue
        a_lo, a_hi = 0.0, df_r
        for _ in range(steps):
            chk()
            mid = 0.5 * (a_lo + a_hi)
            if perco(mid)[1][i]:
                a_lo = mid
            else:
                a_hi = mid
        axes[i] = 2 * a_lo
    path = np.isin(cdf.label, rdf)
    dif = 2 * float(d3[path].max()) if path.any() else 0.0
    return 2 * df_r, axes, dif


# ---------------------------------------------------------------------------
# per feature: every channel and pocket on its own (the GUI's per-feature table)
# ---------------------------------------------------------------------------
NO_FEATURE = 65535                     # a pore mesh vertex next to no feature (uint16)


def _feature_grid(lab: np.ndarray, feat_sets: List[set], n_channels: int, rest_acc: List[int],
                  rest_nac: List[int]) -> Tuple[np.ndarray, int, List[bool]]:
    """Every voxel of the probe-center region -> its feature (channels
    0..n_channels-1, then the pockets; -1 outside the region). A grid piece
    no network node landed in (its class decided by the caller) joins the
    nearest feature of that class; with none of that class, it is a feature
    of its own (``extra``: their accessibility, numbered after the pockets)."""
    from scipy import ndimage
    lut = np.full(int(lab.max()) + 1, -1, np.int32)
    for k, gs in enumerate(feat_sets):
        for g in gs:
            if 0 < g < len(lut) and lut[g] < 0:
                lut[g] = k
    base = lut[lab]
    n = len(feat_sets)
    extra: List[bool] = []
    for pieces, acc in ((rest_acc, True), (rest_nac, False)):
        if not pieces:
            continue
        mine = (base >= 0) & ((base < n_channels) if acc else (base >= n_channels))
        if mine.any():
            _, ind = ndimage.distance_transform_edt(~mine, return_indices=True)
            near = base[ind[0], ind[1], ind[2]]
            del ind
            for r in pieces:
                lut[r] = int(np.bincount(near[lab == r]).argmax())
        else:
            for r in pieces:
                lut[r] = n + len(extra)
                extra.append(acc)
    lut[0] = -1
    return lut[lab], n, extra


class _NearestFeature:
    """The feature of the nearest center voxel of a class (accessible or
    not) for any grid point or Cartesian point: the occupiable shell, the
    surface samples and the pore mesh vertices join a feature this way."""

    def __init__(self, fid, acc_mask, nac_mask, shape, pbc, field, origin):
        self.fid, self.masks, self.shape, self.pbc, self.field = fid, {True: acc_mask, False: nac_mask}, shape, pbc, field
        self.origin = np.asarray(origin, float)
        self._near: Dict[bool, Optional[np.ndarray]] = {}

    def grid(self, accessible: bool) -> Optional[np.ndarray]:
        if accessible not in self._near:
            from scipy import ndimage
            m = self.masks[accessible]
            if not m.any():
                self._near[accessible] = None
            else:
                _, ind = ndimage.distance_transform_edt(~m, return_indices=True)
                self._near[accessible] = self.fid[ind[0], ind[1], ind[2]]
        return self._near[accessible]

    def at_points(self, cart: np.ndarray, accessible: bool) -> np.ndarray:
        """Cartesian points of the analysis frame (no origin) -> feature id (-1: none)."""
        cart = np.asarray(cart, float).reshape(-1, 3)
        near = self.grid(accessible)
        if near is None or not len(cart):
            return np.full(len(cart), -1, np.int64)
        fr = self.field.wrap_frac(cart @ self.field.inv)
        ijk = np.rint(fr * np.array(self.shape)).astype(int)
        for ax in range(3):
            ijk[:, ax] = ijk[:, ax] % self.shape[ax] if self.pbc[ax] else np.clip(ijk[:, ax], 0, self.shape[ax] - 1)
        return near[ijk[:, 0], ijk[:, 1], ijk[:, 2]].astype(np.int64)

    def mesh_ids(self, positions: np.ndarray, accessible: bool) -> np.ndarray:
        """Pore mesh vertices (Cartesian, origin included) -> uint16 feature ids (NO_FEATURE: none)."""
        ids = self.at_points(np.asarray(positions, float) - self.origin, accessible)
        return np.where(ids < 0, NO_FEATURE, ids).astype("<u2")


def _features(st, field, d3, gf, h, fid, n_feat, n_channels, channels, extra, vox, near, occ_acc, occ_nac,
              w_acc, w_nac, r_p) -> List[Dict]:
    """The per-feature rows: probe-center and probe-occupiable volume, surface
    area, the largest included sphere (and where it sits), the centroid. The
    channels' volumes and areas add up to AV, POAV and ASA, the pockets' to
    NAV, PONAV and NASA."""
    from scipy import ndimage
    nf = n_feat + len(extra)
    if nf == 0:
        return []
    flat = fid.ravel()
    sel = flat >= 0
    ids = flat[sel]
    count = np.bincount(ids, minlength=nf)
    occ = np.zeros(nf)
    for mask, acc in ((occ_acc, True), (occ_nac, False)):
        if mask.any():
            g = near.grid(acc)
            if g is not None:
                k = g[mask]
                occ += np.bincount(k[k >= 0], minlength=nf)[:nf]
    area = np.zeros(nf)
    for samples, acc in ((w_acc, True), (w_nac, False)):
        pts = [q for q, _ in samples if len(q)]
        if not pts:
            continue
        w = np.concatenate([np.full(len(q), a) for q, a in samples if len(q)])
        k = near.at_points(np.concatenate(pts), acc)
        ok = k >= 0
        area += np.bincount(k[ok], weights=w[ok], minlength=nf)[:nf]
    # the centroid: a circular mean along the periodic axes (a cage around a cell corner sits at that corner)
    g = gf[sel]
    cen = np.zeros((nf, 3))
    safe = np.maximum(count, 1)
    for ax in range(3):
        if st.pbc[ax]:
            c = np.bincount(ids, weights=np.cos(2 * np.pi * g[:, ax]), minlength=nf)
            s_ = np.bincount(ids, weights=np.sin(2 * np.pi * g[:, ax]), minlength=nf)
            cen[:, ax] = (np.arctan2(s_, c) / (2 * np.pi)) % 1.0
        else:
            cen[:, ax] = np.bincount(ids, weights=g[:, ax], minlength=nf) / safe
    origin = np.array([float(x) for x in st.meta.get("origin", (0.0, 0.0, 0.0))])
    present = [i for i in range(nf) if count[i] > 0]
    tops = ndimage.maximum_position(d3, labels=fid + 1, index=[i + 1 for i in present]) if present else []
    di = {}
    for i, pos in zip(present, tops):
        x0 = (np.array(pos, float) / np.array(d3.shape)) @ st.cell
        v, x = _refine_max(field, x0, h)
        v = max(v, float(d3[tuple(pos)]))
        di[i] = (2 * max(v, 0.0), [float(a) + float(o) for a, o in zip(x, origin)])
    out = []
    for i in range(nf):
        if i < n_channels:
            kind, acc, dim, axes = "channel", True, int(channels[i]["dim"]), [bool(a) for a in channels[i]["axes"]]
        elif i < n_feat:
            kind, acc, dim, axes = "pocket", False, 0, [False, False, False]
        else:
            acc = extra[i - n_feat]
            kind, dim, axes = ("channel" if acc else "pocket"), 0, [False, False, False]
        Di, Dc = di.get(i, (0.0, None))
        row = {"id": i, "kind": kind, "accessible": acc, "dim": dim, "axes": axes,
               "volume_A3": float(count[i]) * vox, "occupiable_A3": float(occ[i]) * vox, "area_A2": float(area[i]),
               "Di": Di, "Di_center_cart": Dc,
               "centroid_cart": [float(a) + float(o) for a, o in zip(cen[i] @ st.cell, origin)] if count[i] else None,
               "n_points": int(count[i])}
        if i >= n_feat:
            row["fragment"] = True
        out.append(row)
    return out


def analyze(st: Structure, probe=DEFAULT_PROBE, radii="ccdc", radius_overrides: Optional[Dict[str, float]] = None,
            spacing: float = DEFAULT_SPACING, max_points: int = DEFAULT_MAX_POINTS,
            surface_samples: int = DEFAULT_SURFACE_SAMPLES, render: bool = False, render_max: int = 72,
            progress: Optional[Callable[[float, str], None]] = None, check: Optional[Callable[[], None]] = None,
            mesh: Optional[Dict] = None, keep_remesh: bool = False) -> Dict:
    """Pore metrics of ``st`` for one probe. See the module docstring.

    ``mesh``: the drawn surfaces' quality (mesh_quality.pores: a preset and
    knobs; None: Standard with ``render_max``). ``keep_remesh``: the result
    also holds ``_remesh(mesh, progress)``, which draws the surfaces again at
    another quality without the analysis (the GUI keeps it, never JSON)."""
    say = progress or (lambda f, m="": None)
    chk = check or (lambda: None)
    r_p, probe_label = probe_radius(probe)
    if st.n == 0:
        raise ValueError("the structure has no atoms")
    R = atom_radii(st, radii, radius_overrides)
    vac = max(5.0, float(R.max()) + r_p + 1.0)
    st, reboxed = open_box(st, vac)
    st = st.wrapped()
    shape, h = _grid_shape(st, spacing, max_points)
    say(0.02, f"grid {shape[0]}x{shape[1]}x{shape[2]}")
    field = SurfaceDistance(st, R)
    gf = _grid_frac(shape)
    cart = gf @ st.cell
    d = np.empty(len(cart))
    chunk = 400_000
    for k in range(0, len(cart), chunk):
        chk()
        d[k:k + chunk] = field(cart[k:k + chunk], frac_in_cell=True)
        say(0.05 + 0.35 * min(1.0, (k + chunk) / len(cart)), "distance field")
    d3 = d.reshape(shape)
    vox = st.volume / d.size
    warnings: List[str] = []
    if st.dim < 3:
        warnings.append(f"{st.dim}D structure: volumes, fractions, Di and Df refer to the whole box, vacuum included")
    if reboxed:
        warnings.append(f"the box along the non-periodic direction(s) is the atoms plus {vac:.1f} A of vacuum on each side")

    # Di: the largest included sphere, refined off the grid
    top = np.argsort(d)[-5:]
    di_r, di_x = -np.inf, None
    for t in top:
        v, x = _refine_max(field, cart[t], h)
        if v > di_r:
            di_r, di_x = v, x
    di = 2 * max(di_r, 0.0)
    say(0.45, "channels")
    chk()

    # the Voronoi network (3D): the free sphere and which pores connect
    net = None
    if st.dim == 3:
        say(0.47, "Voronoi network")
        try:
            net = VoronoiNetwork(st, field, pad=max(6.0, di_r + float(R.max()) + 2.0))
        except Exception as exc:          # qhull trouble: the grid alone decides
            warnings.append(f"Voronoi network unavailable ({type(exc).__name__}); free sphere from the grid")
            net = None
    chk()

    # Df: the largest free sphere
    say(0.55, "largest free sphere")
    df_grid, df_axes_grid, dif_grid = _grid_free_sphere(st, d3, h, chk)
    if net is not None:
        first, axes = net.percolation()
        for _ in range(3):              # refining can raise the answer; settle it
            if first is None:
                break
            before = first[0]
            net.refine(before + 1e-9)
            first, axes = net.percolation()
            if first is None or first[0] <= before + 1e-9:
                break
        # an edge's clearance is negative where it runs through overlapping
        # atom spheres (a dense crystal): no free path then, and Df is 0 as on
        # the grid, never a negative diameter
        if first is not None and first[0] > 0:
            df = 2 * first[0]
            df_axes = [2 * a if a is not None and a > 0 else 0.0 for a in axes]
            roots, _ = net.components(first[0] - 1e-9)
            core = net.core(first[0] - 1e-9) & (roots == roots[first[1]])
            dif = 2 * float(net.node_d[core].max()) if core.any() else df
        else:
            df, df_axes, dif = 0.0, [0.0, 0.0, 0.0], 0.0
        df_method = "voronoi"
    else:
        df, df_axes, dif, df_method = df_grid, df_axes_grid, dif_grid, "grid"
    chk()

    # the probe-center region at r_p: channels (accessible) and pockets
    comps = Components(d3 >= r_p, st.pbc)
    lab = comps.label
    vox_of = lambda r: float((lab == r).sum()) * vox
    channels, pockets = [], []
    # the grid pieces of each feature (channels first, then pockets): the per-feature table
    feat_sets: List[set] = []
    rest_acc: List[int] = []            # pieces no network node landed in, by their class verdict
    rest_nac: List[int] = []
    if net is not None:
        net.refine(r_p)
        roots, cyc = net.components(r_p)
        live = net.node_d >= r_p
        node_ijk = np.rint(net.node_frac * np.array(shape)).astype(int) % np.array(shape)
        node_lab = lab[node_ijk[:, 0], node_ijk[:, 1], node_ijk[:, 2]]
        acc_roots, pocket_sets = set(), []
        for nr in np.unique(roots[live]):
            members = live & (roots == nr)
            glabs = set(int(x) for x in np.unique(node_lab[members]) if x > 0)
            basis = cyc.get(int(nr), [])
            if basis:
                acc_roots |= glabs
                if glabs:
                    vb = np.array(basis)
                    channels.append({"dim": int(np.linalg.matrix_rank(vb)),
                                     "axes": [bool(np.any(vb[:, i] != 0)) for i in range(3)], "open": False,
                                     "volume_A3": sum(vox_of(g) for g in glabs)})
                    claimed = set().union(*feat_sets) if feat_sets else set()
                    feat_sets.append(glabs - claimed)
            elif glabs:
                pocket_sets.append(glabs)
        # a pocket is a set of grid pieces one closed network component reaches;
        # pieces that also belong to a channel are channel, and pockets that
        # share a piece are one pocket
        merged: List[set] = []
        for ps in pocket_sets:
            ps = ps - acc_roots
            if not ps:
                continue
            hit = [m for m in merged if m & ps]
            for m in hit:
                merged.remove(m)
                ps |= m
            merged.append(ps)
        nac_roots = set().union(*merged) if merged else set()
        pockets = [{"volume_A3": sum(vox_of(g) for g in m)} for m in merged]
        feat_sets.extend(set(m) for m in merged)
        nac_roots -= acc_roots
        # grid pieces no network node landed in (fragments of a pore cut by the
        # grid): they take the verdict of the nearest classified voxel
        rest = [r for r in comps.roots if r not in acc_roots and r not in nac_roots]
        if rest:
            from scipy import ndimage
            cls = np.zeros(shape, np.int8)
            cls[np.isin(lab, list(acc_roots))] = 1
            cls[np.isin(lab, list(nac_roots))] = 2
            if (cls > 0).any():
                _, ind = ndimage.distance_transform_edt(cls == 0, return_indices=True)
                near = cls[ind[0], ind[1], ind[2]]
                for r in rest:
                    m = lab == r
                    vote = np.bincount(near[m], minlength=3)
                    (acc_roots if vote[1] >= vote[2] else nac_roots).add(r)
                    (rest_acc if vote[1] >= vote[2] else rest_nac).append(r)
            else:
                nac_roots |= set(rest)
                rest_nac.extend(rest)
    else:
        acc_roots = set(comps.accessible_roots())
        chan_sets, pocket_sets_g = [], []
        for r in comps.roots:
            if r in acc_roots:
                channels.append({"dim": comps.dim[r], "axes": list(comps.axes[r]), "open": comps.open[r],
                                 "volume_A3": vox_of(r)})
                chan_sets.append({r})
            else:
                pockets.append({"volume_A3": vox_of(r)})
                pocket_sets_g.append({r})
        feat_sets = chan_sets + pocket_sets_g
    in_c = lab > 0
    acc_mask = np.isin(lab, list(acc_roots)) if acc_roots else np.zeros(shape, bool)
    nac_mask = in_c & ~acc_mask
    chan_dim = max([max(c["dim"], 1) for c in channels], default=0)
    fid, n_feat, extra = _feature_grid(lab, feat_sets, len(channels), rest_acc, rest_nac)
    say(0.7, "volumes")
    chk()

    # volumes: probe-center (AV/NAV) and probe-occupiable (POAV/PONAV)
    void0 = float((d >= 0).sum()) * vox
    av = float(acc_mask.sum()) * vox
    nav = float(nac_mask.sum()) * vox
    from scipy.spatial import cKDTree

    def on_mask(points, mask):
        """Whether any of the 8 grid corners around each point is in mask."""
        fp = field.wrap_frac(points @ field.inv)
        base = np.floor(fp * np.array(shape)).astype(int)
        hit = np.zeros(len(fp), bool)
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    ijk = base + np.array([dx, dy, dz])
                    for ax in range(3):
                        ijk[:, ax] = ijk[:, ax] % shape[ax] if st.pbc[ax] else np.clip(ijk[:, ax], 0, shape[ax] - 1)
                    hit |= mask[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
        return hit

    def grid_centers(mask):
        """Grid points of a center class near the pore wall (with images)."""
        near = mask & (d3 < 2 * r_p + 2 * h)
        cf = gf[near.ravel()]
        if not len(cf):
            return np.zeros((0, 3))
        tmp = Structure(st.cell, cf, np.ones(len(cf), int), st.pbc)
        return tmp.images(r_p + 2 * h)[0]

    gc_acc, gc_nac = grid_centers(acc_mask), grid_centers(nac_mask)

    # the boundary of the probe-center region, sampled exactly: points on the
    # spheres of radius R_i + r_p no other atom's sphere covers. They give the
    # accessible surface area and, with the grid centers, the volume the probe
    # sphere occupies (a point within r_p of the boundary of C is in it).
    say(0.75, "surface area")
    unit = _fib_sphere(max(20, int(surface_samples)))
    home = st.cart
    big = SurfaceDistance(st, R + r_p)
    t_acc = cKDTree(gc_acc) if len(gc_acc) else None
    t_nac = cKDTree(gc_nac) if len(gc_nac) else None
    asa = nasa = 0.0
    unresolved = 0
    s_acc, s_nac = [], []
    w_acc, w_nac = [], []                # each sphere sample's area (the per-feature surface areas)
    for i in range(st.n):
        chk() if i % 200 == 0 else None
        rr = R[i] + r_p
        pts = home[i] + rr * unit
        free = big(pts) > -1e-7 * max(1.0, rr)
        if not free.any():
            continue
        q = pts[free]
        acc_hit = on_mask(q, acc_mask)
        nac_hit = on_mask(q, nac_mask) & ~acc_hit
        loose = ~acc_hit & ~nac_hit
        if loose.any():
            # a thin part of C the grid missed: the nearer class of grid center decides
            da = t_acc.query(q[loose], k=1, distance_upper_bound=2 * h)[0] if t_acc is not None else np.full(loose.sum(), np.inf)
            dn = t_nac.query(q[loose], k=1, distance_upper_bound=2 * h)[0] if t_nac is not None else np.full(loose.sum(), np.inf)
            li = np.flatnonzero(loose)
            acc_hit[li[(da <= dn) & np.isfinite(da)]] = True
            nac_hit[li[(dn < da) & np.isfinite(dn)]] = True
            unresolved += int((~np.isfinite(da) & ~np.isfinite(dn)).sum())
        area = 4 * math.pi * rr * rr / len(unit)
        asa += area * float(acc_hit.sum())
        nasa += area * float(nac_hit.sum())
        s_acc.append(q[acc_hit])
        s_nac.append(q[nac_hit])
        w_acc.append((q[acc_hit], area))
        w_nac.append((q[nac_hit], area))
    # where two of those spheres meet, C has a re-entrant edge (a probe in a
    # crevice touching both atoms): sample those circles too
    ii, jj, sh, dist = st.neighbor_pairs(2 * (float(R.max()) + r_p))
    first_nz = np.take_along_axis(sh, np.argmax(sh != 0, axis=1)[:, None], 1).ravel()
    keep = (ii < jj) | ((ii == jj) & (first_nz > 0))
    keep &= dist < R[ii] + R[jj] + 2 * r_p
    if keep.any():
        a1 = home[ii[keep]]
        a2 = (st.frac[jj[keep]] + sh[keep]) @ st.cell
        r1 = R[ii[keep]] + r_p
        r2 = R[jj[keep]] + r_p
        L = dist[keep]
        u = (a2 - a1) / L[:, None]
        x = (L * L + r1 * r1 - r2 * r2) / (2 * L)
        rc = np.sqrt(np.clip(r1 * r1 - x * x, 0, None))
        e1 = np.cross(u, np.where(np.abs(u[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]]))
        e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
        e2 = np.cross(u, e1)
        m = max(16, int(math.ceil(2 * math.pi * float(rc.max()) / 0.08)))
        th = np.linspace(0, 2 * math.pi, m, endpoint=False)
        circ = (a1 + x[:, None] * u)[:, None, :] + rc[:, None, None] * (
            np.cos(th)[None, :, None] * e1[:, None, :] + np.sin(th)[None, :, None] * e2[:, None, :])
        circ = circ.reshape(-1, 3)
        circ = circ[big(circ) > -1e-6]
        if len(circ):
            ca = on_mask(circ, acc_mask)
            cn = on_mask(circ, nac_mask) & ~ca
            s_acc.append(circ[ca])
            s_nac.append(circ[cn])
    say(0.85, "occupiable volume")
    chk()
    shell = (d3 >= 0) & (d3 < r_p)
    shell_idx = np.flatnonzero(shell.ravel())

    def occupiable(center_mask, grid_pts, samples):
        out = center_mask.copy()
        pts = [grid_pts] if len(grid_pts) else []
        if samples:
            sp = np.concatenate(samples) if len(samples) > 1 else samples[0]
            if len(sp):
                f = field.wrap_frac(sp @ field.inv)
                tmp = Structure(st.cell, f, np.ones(len(f), int), st.pbc)
                pts.append(tmp.images(r_p + h)[0])
        if not pts:
            return out, None
        tree = cKDTree(np.concatenate(pts))
        if len(shell_idx):
            dist, _ = tree.query(cart[shell_idx], k=1, distance_upper_bound=r_p + 1e-9)
            out.ravel()[shell_idx[np.isfinite(dist)]] = True
        return out, tree

    occ_acc, tree_acc = occupiable(acc_mask, gc_acc, s_acc)
    occ_nac, tree_nac = occupiable(nac_mask, gc_nac, s_nac)
    occ_nac &= ~occ_acc
    poav = float(occ_acc.sum()) * vox
    ponav = float(occ_nac.sum()) * vox
    say(0.88, "per-feature table")
    chk()
    near = _NearestFeature(fid, acc_mask, nac_mask, shape, st.pbc, field, st.meta.get("origin", (0.0, 0.0, 0.0)))
    features = _features(st, field, d3, gf, h, fid, n_feat, len(channels), channels, extra, vox, near,
                         occ_acc, occ_nac, w_acc, w_nac, r_p)

    mass_g = st.mass_amu / 6.02214076e23
    V = st.volume
    dens = st.density

    def per(vol):
        return {"A3": vol, "fraction": vol / V, "cm3_g": vol * 1e-24 / mass_g}

    def area(a):
        return {"A2": a, "m2_cm3": a / V * 1e4, "m2_g": a * 1e-20 / mass_g}

    out = {
        "method": "native-grid",
        "probe": {"name": probe_label, "radius": r_p},
        "radii": radii if isinstance(radii, str) else "custom",
        "grid": {"shape": list(shape), "spacing": h, "points": int(d.size)},
        "cell_volume_A3": V, "mass_amu": st.mass_amu, "density_g_cm3": dens,
        "Di": di, "Df": df, "Dif": dif, "Df_axes": df_axes, "Df_method": df_method, "Df_grid": df_grid,
        "Di_center_cart": [float(x) + float(o) for x, o in zip(di_x, st.meta.get("origin", (0.0, 0.0, 0.0)))],
        "void_fraction_geometric": void0 / V,
        "AV": per(av), "NAV": per(nav), "POAV": per(poav), "PONAV": per(ponav),
        "ASA": area(asa), "NASA": area(nasa),
        "channels": channels, "pockets": pockets, "features": features,
        "channel_dimensionality": chan_dim,
        "n_channels": len(channels), "n_pockets": len(pockets),
        "warnings": warnings,
        "dimensionality": st.dim,
    }
    if di_r < r_p:
        out["warnings"].append(f"the probe ({probe_label}, r = {r_p:.2f} A) fits nowhere: Di/2 = {di_r:.2f} A")
    if render:
        say(0.9, "render data")
        knobs = mesh_quality.pores(mesh)
        if mesh is None:
            knobs["grid"] = int(render_max)
        out["render"] = _render_fields(st, shape, field, acc_mask, nac_mask, tree_acc, tree_nac, r_p, knobs,
                                       feature_of=near.mesh_ids)
        if keep_remesh:
            def remesh(spec, progress=None):
                (progress or (lambda f, m="": None))(0.1, "surfaces")
                return _render_fields(st, shape, field, acc_mask, nac_mask, tree_acc, tree_nac, r_p,
                                      mesh_quality.pores(spec), feature_of=near.mesh_ids)
            out["_remesh"] = remesh
    say(1.0, "done")
    return out


def _render_fields(st, shape, field, acc_mask, nac_mask, tree_acc, tree_nac, r_p, knobs, feature_of=None) -> Dict:
    """Two scalar fields on a uniform coarse grid that includes the far faces
    (so an isosurface closes): positive inside the volume the probe sphere can
    occupy, accessible and inaccessible; the surface at 0 is the pore wall a
    probe touches. Each coarse point takes the channel label of the nearest
    analysis-grid point. ``knobs``: mesh_quality.pores (grid, upsample,
    smooth, max_triangles)."""
    if not isinstance(knobs, dict):              # a bare render_max (the grid), as before presets
        knobs = {**mesh_quality.pores(None), "grid": int(knobs)}
    up = max(1, int(knobs.get("upsample", 1)))
    nc = [max(4, min(int(n) * up, int(knobs["grid"]))) for n in shape]
    axes = [np.arange(m + 1) / m for m in nc]
    F = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    pts = F @ st.cell
    dd = field(pts, frac_in_cell=True)
    ijk = np.rint(F * np.array(shape)).astype(int)
    for ax in range(3):
        ijk[:, ax] = ijk[:, ax] % shape[ax] if st.pbc[ax] else np.clip(ijk[:, ax], 0, shape[ax] - 1)

    def fieldof(mask, tree):
        inside = mask[ijk[:, 0], ijk[:, 1], ijk[:, 2]] & (dd >= r_p - 1e-9)
        v = np.where(dd < 0, dd - r_p, -r_p).astype(float)
        v[inside] = dd[inside]
        rest = ~inside & (dd >= 0)
        if tree is not None and rest.any():
            dist, _ = tree.query(pts[rest], k=1)
            v[rest] = r_p - dist
        return np.clip(v, -2 * r_p - 1, None).astype("<f4")

    acc = fieldof(acc_mask, tree_acc)
    nac = fieldof(nac_mask, tree_nac)
    shp = [m + 1 for m in nc]
    origin = [float(x) for x in st.meta.get("origin", (0.0, 0.0, 0.0))]
    voxels = np.array([st.cell[i] / nc[i] for i in range(3)])
    return {"shape": shp, "origin": origin, "voxels": voxels.tolist(),
            "accessible": base64.b64encode(acc.tobytes()).decode(),
            "inaccessible": base64.b64encode(nac.tobytes()).decode(),
            "has_inaccessible": bool(nac_mask.any()), "has_accessible": bool(acc_mask.any()),
            "level": 0.0, "encoding": "float32 little-endian, C order (a index slowest)",
            "meshes": _tagged(pore_meshes(acc.reshape(shp), nac.reshape(shp), st.pbc, voxels, origin,
                                          smooth=float(knobs["smooth"]), max_triangles=int(knobs["max_triangles"]),
                                          feature_of=feature_of), knobs, nc)}


def _tagged(meshes: Dict, knobs: Dict, nc) -> Dict:
    """The meshes with the quality they were made at (the GUI's Inspector shows it)."""
    tri = sum(int(m["n_triangles"]) for m in (meshes.get("accessible"), meshes.get("inaccessible")) if m)
    meshes["quality"] = {**knobs, "render_grid": [int(n) for n in nc], "triangles": tri}
    return meshes


# the drawn surfaces only (never the numbers): a Gaussian of this many voxels rounds off the
# grid's steps, and a cell's surfaces stop at this many triangles (a coarser grid beyond it)
RENDER_SMOOTH = 0.7
RENDER_MAX_TRIANGLES = 600_000
# the grid steps the surface's regions connect along: the edges of the tetrahedra
# marching_tetrahedra cuts each cube into (and their opposites)
_KUHN = [np.array(d) for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1))]


def pore_meshes(acc: np.ndarray, nac: np.ndarray, pbc: Sequence[bool], voxels: np.ndarray, origin: Sequence[float],
                smooth: float = RENDER_SMOOTH, max_triangles: int = RENDER_MAX_TRIANGLES,
                feature_of: Optional[Callable[[np.ndarray, bool], np.ndarray]] = None) -> Dict:
    """Triangle meshes of the two render fields (grids that include the far
    faces), as the 3D view draws them:

    - accessible: the channels, cut at the cell faces (they go on forever);
    - inaccessible: every pocket once and whole, closed, in the periodic
      image that puts its center in the cell (a cage at a corner is drawn
      around that corner), with its pocket number per vertex.

    Positions are Cartesian (A), normals point out of the pore (toward the
    atoms) and triangles wind counterclockwise seen from that side. Base64:
    positions float32, normals int8 (x127), faces uint32, pocket uint16,
    little-endian. ``feature_of(positions, accessible)``, when given, numbers
    each vertex with the feature (analyze()'s ``features``) it bounds:
    ``feature``, uint16 (65535: none), and ``features``, the ids present."""
    from scipy import ndimage
    pbc = tuple(bool(p) for p in pbc)
    vox = np.asarray(voxels, float)
    o = np.asarray(origin, float)
    inv_t = np.linalg.inv(vox).T
    flip = np.linalg.det(vox) < 0

    def base(F, step):
        # the periodic grid (a periodic axis's far face repeats its near one), smoothed; a coarser
        # one (``step`` times the spacing) for a surface over the triangle budget
        A = np.asarray(F, float)[tuple(slice(0, -1) if p else slice(None) for p in pbc)]
        scale = np.ones(3)
        if step > 1:
            coords = []
            for ax in range(3):
                n = A.shape[ax]
                m = max(4, n // step) if pbc[ax] else max(4, (n - 1) // step + 1)
                c = np.arange(m) * (n / m if pbc[ax] else (n - 1) / (m - 1))
                scale[ax] = c[1] - c[0]
                coords.append(c)
            A = ndimage.map_coordinates(A, np.meshgrid(*coords, indexing="ij"), order=1, mode="grid-wrap")
        if smooth > 0:
            k = int((A > 0).sum())
            A = ndimage.gaussian_filter(A, smooth, mode=["wrap" if p else "nearest" for p in pbc])
            if 0 < k < A.size:
                # smoothing rounds a pore and also shrinks it: the level that keeps its volume
                top = np.partition(A.ravel(), A.size - k - 1)[A.size - k - 1:]
                A = A - 0.5 * (top[0] + top[1:].min())
        return A, scale

    def grad(A, periodic):
        return np.stack([(np.roll(A, -1, ax) - np.roll(A, 1, ax)) / 2 if periodic[ax] else np.gradient(A, axis=ax)
                         for ax in range(3)], -1)

    def close(A):
        # the far faces again: the periodic grid plus its first layer at the end of each periodic axis
        for ax in range(3):
            if pbc[ax]:
                A = np.concatenate([A, np.take(A, [0], axis=ax)], axis=ax)
        return A

    def surface(F, G, shift, v_scale):
        from .isosurface import interpolate, marching_tetrahedra
        verts, faces, edge = marching_tetrahedra(F, 0.0)
        if not len(faces):
            return None
        g = (interpolate(G, edge) / v_scale) @ inv_t
        n = -g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
        if flip:
            faces = faces[:, ::-1]
        return {"p": o + (verts + shift) * v_scale @ vox, "n": n, "f": faces}

    def pack(parts, pockets=None, accessible=True):
        parts = [p for p in parts if p is not None]
        if not parts:
            return None
        P = np.concatenate([p["p"] for p in parts]).astype("<f4")
        N = np.clip(np.rint(np.concatenate([p["n"] for p in parts]) * 127), -127, 127).astype("i1")
        offs = np.cumsum([0] + [len(p["p"]) for p in parts])
        Fc = np.concatenate([p["f"] + k for p, k in zip(parts, offs)]).astype("<u4")
        b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
        out = {"n_vertices": int(len(P)), "n_triangles": int(len(Fc)), "positions": b64(P), "normals": b64(N), "faces": b64(Fc)}
        if pockets is not None:
            out["pocket"] = b64(np.concatenate([np.full(len(p["p"]), k, "<u2") for p, k in zip(parts, pockets)]))
            out["n_pockets"] = int(max(pockets) + 1)
        if feature_of is not None:
            fv = np.asarray(feature_of(P.astype(float), accessible)).astype("<u2")
            out["feature"] = b64(fv)
            out["features"] = sorted(int(x) for x in np.unique(fv) if x != NO_FEATURE)
        return out

    def channels(A, step):
        if not (A > 0).any():
            return None
        return surface(close(A), close(grad(A, pbc)), 0.0, step)

    def pockets(A, step):
        pos = A > 0
        st3 = np.zeros((3, 3, 3), bool)
        st3[1, 1, 1] = True
        for d in _KUHN:
            st3[tuple(1 + d)] = st3[tuple(1 - d)] = True
        lab, n = ndimage.label(pos, st3)
        if not n:
            return [], []
        shape = np.array(A.shape)
        # join the pieces across the periodic faces, with the image each one sits in
        parent = list(range(n + 1))
        rel = [np.zeros(3, int) for _ in range(n + 1)]
        wraps = set()

        def find(x):
            if parent[x] == x:
                return x, np.zeros(3, int)
            r, ro = find(parent[x])
            parent[x], rel[x] = r, rel[x] + ro
            return r, rel[x]
        idx = [np.arange(s) for s in A.shape]
        for d in _KUHN:
            if not any(d[ax] and pbc[ax] for ax in range(3)):
                continue
            cross = np.zeros(A.shape, bool)
            for ax in range(3):
                if d[ax] and pbc[ax]:
                    sl = [None, None, None]
                    sl[ax] = slice(None)
                    cross |= (idx[ax] == A.shape[ax] - 1)[tuple(sl)]
            for ax in range(3):            # off the grid along an open axis: no neighbor
                if d[ax] and not pbc[ax]:
                    sl = [None, None, None]
                    sl[ax] = slice(None)
                    cross &= (idx[ax] < A.shape[ax] - 1)[tuple(sl)]
            p = np.argwhere(pos & cross)
            if not len(p):
                continue
            q = p + d
            w = q // shape
            q = q % shape
            ok = pos[q[:, 0], q[:, 1], q[:, 2]]
            rows = np.unique(np.column_stack([lab[tuple(p[ok].T)], lab[tuple(q[ok].T)], w[ok]]), axis=0)
            for a, b, *ww in rows:
                ra, oa = find(a)
                rb, ob = find(b)
                if ra == rb:
                    if not np.array_equal(ob, oa + ww):
                        wraps.add(ra)     # joined to its own image: not a pocket (drawn cut at the faces)
                else:
                    parent[rb], rel[rb] = ra, oa + np.array(ww) - ob
        roots = {}
        for L in range(1, n + 1):
            r, off = find(L)
            roots.setdefault(r, []).append((L, off.copy()))
        flat = lab.ravel()
        order = np.argsort(flat, kind="stable")
        starts = np.searchsorted(flat[order], np.arange(n + 2))
        G0 = grad(A, pbc)
        floor = float(min(A.min(), 0.0)) - 1e-3
        meshes, ids = [], []
        cut = np.zeros(A.shape, bool)
        for r, members in roots.items():
            if r in wraps:
                for L, _ in members:
                    cut |= lab == L
                continue
            U = np.concatenate([np.column_stack(np.unravel_index(order[starts[L]:starts[L + 1]], A.shape)) + off * shape
                                for L, off in members])
            # the image with its center in the cell (a cage centered on a corner: around that corner)
            c = U.mean(0) / shape
            s = np.where(pbc, -np.floor(c + 1e-3), 0).astype(int)
            U = U + s * shape
            lo, hi = U.min(0) - 2, U.max(0) + 3
            box = np.stack(np.meshgrid(*[np.arange(a, b) for a, b in zip(lo, hi)], indexing="ij"), -1)
            g = box.reshape(-1, 3)
            inside = np.ones(len(g), bool)
            gm = g.copy()
            for ax in range(3):
                if pbc[ax]:
                    gm[:, ax] %= shape[ax]
                else:
                    inside &= (g[:, ax] >= 0) & (g[:, ax] < shape[ax])
                    gm[:, ax] = np.clip(g[:, ax], 0, shape[ax] - 1)
            V = np.where(inside, A[gm[:, 0], gm[:, 1], gm[:, 2]], floor).reshape(box.shape[:3])
            mine = np.zeros(V.shape, bool)
            mine[tuple((U - lo).T)] = True
            # another pocket's (or another image's) volume in the box is outside this one
            V = np.where(mine | (V <= 0), V, -1e-3)
            surf = surface(V, np.stack(np.gradient(V), -1), lo, step)
            if surf is not None:
                meshes.append(surf)
                ids.append(len(ids))
        if cut.any():
            # a region joined to its own image is cut at the faces like the channels
            V = np.where(cut | (A <= 0), A, -1e-3)
            surf = surface(close(V), close(G0), 0.0, step)
            if surf is not None:
                meshes.append(surf)
                ids.append(len(ids))
        return meshes, ids

    # the pockets always on the full grid (a coarser one could join one to its own image through
    # a window too narrow for the probe); the channels coarser while over the triangle budget
    pk, ids = pockets(*base(nac, 1))
    left = max_triangles - sum(len(m["f"]) for m in pk)
    for step in (1, 2, 3):
        a = channels(*base(acc, step))
        if a is None or len(a["f"]) <= left or step == 3:
            break
    return {"accessible": pack([a]), "inaccessible": pack(pk, ids, accessible=False) if pk else None, "step": step}


def compare_probes(st: Structure, probes: Sequence = tuple(PROBES), **kw) -> List[Dict]:
    """The same structure for several probes (no render data)."""
    return [analyze(st, probe=p, **{**kw, "render": False}) for p in probes]
