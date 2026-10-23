"""Volumetric analysis of a CHARGE+POTENTIAL run's cubes (CRYSTAL ECH3 and
POT3: ``<stem>_DENS.CUBE``, ``<stem>_POT.CUBE``, ``<stem>_SPIN.CUBE``) for the
derived-property registry (mace/properties/defs/charge_potential.py).

The cube is read by mace plotting's ``CubeFile`` through
``mace.gui.volume.read`` (the Volume document's reader), with the vacuum
crop off, and cut to one period along each periodic axis
(``volume.grid_period``: a grid that repeats its first plane as its last
counts it once). Everything here is computed on that full grid:

* :func:`planar_average`: the potential averaged over the lattice planes
  normal to the stacking axis (a slab's non-periodic axis, a bulk's c), and
  its macroscopic average (Baldereschi, Baroni and Resta, PRL 61, 734, 1988:
  a running mean over one interlayer period); the vacuum levels of a slab
  and its work functions E_vac - E_F;
* :func:`dipole`: the dipole moment of the charge in the cell along its
  non-periodic directions, from the nuclear charges and the density (a
  slab's is better read from its vacuum step, -A dU / 4 pi: the defs do);
* :func:`core_corrections`: what the grid's sum misses or over-counts at
  each nucleus (a grid a tenth of a bohr apart does not resolve the core's
  cusp), from Slater core densities sampled on the same points; the dipole
  and the Bader populations take it out at the nucleus;
* :func:`bader`: Bader's atoms in molecules partition of the density (zero
  flux surfaces), MACE's own implementation of the grid algorithms of
  Henkelman, Arnaldsson and Jonsson (Comput. Mater. Sci. 36, 354, 2006: the
  on-grid steepest ascent) and Tang, Sanville and Henkelman (J. Phys.:
  Condens. Matter 21, 084204, 2009: near-grid trajectories with a
  correction vector, which remove the grid's directional bias). No Bader
  program is shipped, called or fetched.

Units: the cube's are atomic (bohr, e/bohr^3, hartree/e); positions are
returned in Angstrom, energies in eV. The potential CRYSTAL writes is the
electrostatic potential of the charge distribution (positive at the nuclei);
an electron's energy is minus it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from mace.constants import BOHR_TO_ANGSTROM as BOHR, HARTREE_TO_EV as HARTREE_EV  # noqa: E402  (BOHR: Å per bohr)
DEBYE_PER_AU = 2.541746473       # one e*bohr in Debye
#: density below which a grid point is vacuum (e/bohr^3): no basin is followed there
VACUUM_DENSITY = 1e-4
#: largest grid the Bader partition takes (points); a larger one is skipped
BADER_MAX_POINTS = 40_000_000
#: largest cube read here (bytes): a larger one is skipped
MAX_BYTES = 2_000_000_000


@dataclass
class Grid:
    """One period of a cube along each periodic axis, in atomic units."""
    data: Any                    # (n0, n1, n2) float64
    voxel: Any                   # (3, 3): the step vectors (rows), bohr
    origin: Any                  # (3,): bohr
    periodic: List[bool]
    dv: float                    # bohr^3 per point (one period / points along a periodic axis)
    cell: Any                    # (3, 3): the lattice vectors along the periodic axes, bohr (rows; others: the grid's span)
    numbers: List[int]
    charges: List[float]         # the cube's nuclear-charge column
    positions: Any               # (natoms, 3): bohr
    kind: str
    path: Path
    duplicate_planes: List[str] = field(default_factory=list)
    #: periodic axes whose points do not divide the period evenly (a thinned cube): no Bader, no dipole
    uneven: List[str] = field(default_factory=list)

    @property
    def dimensionality(self) -> int:
        return int(sum(self.periodic))

    @property
    def shape(self):
        return self.data.shape


_CACHE: Dict[Any, Grid] = {}


def load(path) -> Grid:
    """The cube at ``path`` (mace.gui.volume.read, no vacuum crop), one period per periodic axis."""
    import numpy as np
    from mace.gui import volume
    p = Path(path)
    st = p.stat()
    if st.st_size > MAX_BYTES:
        raise ValueError(f"{p.name}: {st.st_size / 1e9:.1f} GB is larger than the {MAX_BYTES / 1e9:.0f} GB read here")
    key = (str(p.resolve()), st.st_mtime_ns, st.st_size)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    c = volume.read(str(p), skip_vacuum=True)
    d = np.asarray(c.data, dtype=np.float64)
    per = list(volume._periodic_axes(c))
    period = volume.grid_period(c)
    dup = ["abc"[k] for k in range(3) if per[k] and period[k] < d.shape[k]]
    d = np.ascontiguousarray(d[:period[0], :period[1], :period[2]])
    vv = np.asarray(c.voxel_vectors, float)
    cp = getattr(c, "cell_parameters", None) or {}
    cell = np.zeros((3, 3))
    uneven = []
    scale = 1.0
    for k in range(3):
        u = vv[k] / np.linalg.norm(vv[k])
        if per[k] and cp.get("abc"[k]):
            length = float(cp["abc"[k]])
            cell[k] = u * length
            n, h = d.shape[k], float(np.linalg.norm(vv[k]))
            if abs(length / h - n) > 1e-3 * n:
                # the points do not divide the period evenly (a cube thinned to every 2nd point keeps
                # N points over N - 1/2 steps): each point stands for length / N of it
                uneven.append("abc"[k])
                scale *= (length / n) / h
        else:
            cell[k] = vv[k] * d.shape[k]
    dv = abs(float(np.linalg.det(vv))) * scale
    charges = []
    try:
        with open(p) as fh:
            fh.readline(), fh.readline()
            nat = abs(int(fh.readline().split()[0]))
            for _ in range(3):
                fh.readline()
            for _ in range(nat):
                charges.append(float(fh.readline().split()[1]))
    except (OSError, ValueError, IndexError):
        charges = [float(z) for z in c.atomic_numbers]
    g = Grid(d, vv, np.asarray(c.origin, float), per, dv, cell, [int(z) for z in c.atomic_numbers], charges,
             np.asarray(c.atomic_positions, float).reshape(-1, 3), str(c.data_type), p, dup, uneven)
    _CACHE.clear()                     # one grid at a time: a slab's cube is tens of millions of values
    _CACHE[key] = g
    return g


# ------------------------------------------------------------------ stacking axis and planar average
def stacking_axis(g: Grid) -> Optional[int]:
    """The axis a planar average runs along: a slab's non-periodic one, a bulk's c; None for a polymer
    or a molecule (no lattice planes to average over)."""
    if g.dimensionality == 2:
        return g.periodic.index(False)
    if g.dimensionality == 3:
        return 2
    return None


def _normal(g: Grid, axis: int):
    import numpy as np
    a, b = [g.voxel[k] for k in range(3) if k != axis]
    n = np.cross(a, b)
    n /= np.linalg.norm(n)
    if float(np.dot(n, g.voxel[axis])) < 0:
        n = -n
    return n


def heights(g: Grid, axis: int):
    """The atoms' heights along the stacking direction (Angstrom), on the grid's own scale."""
    import numpy as np
    n = _normal(g, axis)
    return (g.positions @ n) * BOHR


def planar_average(g: Grid, axis: int):
    """(z in Angstrom, the plane averages of the grid's values) along ``axis``. The plane average is the
    mean over a plane's points: they sample one period of each in-plane axis evenly."""
    import numpy as np
    others = tuple(k for k in range(3) if k != axis)
    prof = g.data.mean(axis=others)
    n = _normal(g, axis)
    step = float(np.dot(g.voxel[axis], n))
    z = (float(np.dot(g.origin, n)) + step * np.arange(prof.size)) * BOHR
    return z, prof


def layer_period(h: Sequence[float], period: Optional[float] = None, gap: float = 0.7) -> Optional[float]:
    """The interlayer distance (Angstrom) of atoms at heights ``h``: atoms closer than ``gap`` in height are
    one layer; the median distance between neighboring layers (across the cell's boundary too when
    ``period``, a periodic stacking axis's length, is given). None for a single layer."""
    import numpy as np
    x = np.sort(np.asarray(h, float))
    if period:
        x = np.sort(np.mod(x, period))
    if x.size < 2:
        return None
    layers = [[x[0]]]
    for v in x[1:]:
        if v - layers[-1][-1] > gap:
            layers.append([v])
        else:
            layers[-1].append(v)
    centers = np.array([np.mean(L) for L in layers])
    if period and len(centers) > 1 and centers[0] + period - centers[-1] <= gap:
        centers = centers[:-1]
    d = list(np.diff(centers))
    if period:
        d.append(centers[0] + period - centers[-1])
    d = [v for v in d if v > gap]
    if not d or (len(centers) < 2 and not period):
        return None
    return float(np.median(d))


def macroscopic_average(z, prof, window: float, periodic: bool):
    """The running mean of ``prof`` over ``window`` Angstrom (a periodic profile wraps; a slab's is held at
    its end values)."""
    import numpy as np
    step = float(z[1] - z[0]) if len(z) > 1 else 1.0
    w = max(1, int(round(window / step)))
    if w <= 1 or w >= len(prof):
        return np.asarray(prof, float).copy()
    kern = np.ones(w) / w
    left, right = w // 2, w - 1 - w // 2
    pad = np.pad(np.asarray(prof, float), (left, right), mode="wrap" if periodic else "edge")
    return np.convolve(pad, kern, mode="valid")


def potential_profile(g: Grid, fermi_au: Optional[float] = None, vacuum_width: float = 1.0) -> Dict[str, Any]:
    """The potential along the stacking axis: the plane and macroscopic averages of the electron's
    potential energy U = -V (eV), and for a slab the vacuum level on each side (the mean of U over the
    outermost ``vacuum_width`` Angstrom of the grid), how flat it is there, the step across the slab and,
    with the Fermi energy, the work functions E_vac - E_F."""
    import numpy as np
    axis = stacking_axis(g)
    if axis is None:
        raise ValueError("a polymer or a molecule has no stacking axis")
    z, v = planar_average(g, axis)
    u = -v * HARTREE_EV
    h = heights(g, axis)
    periodic = g.periodic[axis]
    length = float(len(z) * (z[1] - z[0])) if periodic and len(z) > 1 else None
    window = layer_period(h, length)
    macro = macroscopic_average(z, u, window, periodic) if window else None
    out: Dict[str, Any] = {"axis": "abc"[axis], "z_A": z, "planar_ev": u, "macroscopic_ev": macro,
                           "window_A": window, "atoms_z_A": [float(h.min()), float(h.max())] if h.size else None,
                           "periodic": bool(periodic)}
    if periodic:
        out["mean_ev"] = float(u.mean())
        return out
    step = float(z[1] - z[0])
    k = max(2, int(round(vacuum_width / step)))
    sides = {}
    for side, sl, sl2, edge in (("bottom", slice(0, k), slice(k, 2 * k), float(z[0])),
                                ("top", slice(len(u) - k, len(u)), slice(len(u) - 2 * k, len(u) - k), float(z[-1]))):
        level = float(u[sl].mean())
        drift = abs(level - float(u[sl2].mean()))
        clear = (float(h.min()) - edge) if side == "bottom" else (edge - float(h.max()))
        sides[side] = {"level_ev": level, "drift_ev": drift, "distance_A": clear,
                       "converged": bool(drift < 0.02 and clear >= 3.0)}
    out["vacuum"] = sides
    out["step_ev"] = sides["top"]["level_ev"] - sides["bottom"]["level_ev"]
    if fermi_au is not None:
        ef = float(fermi_au) * HARTREE_EV
        out["fermi_ev"] = ef
        for side in ("top", "bottom"):
            sides[side]["work_function_ev"] = sides[side]["level_ev"] - ef
    return out


def _even(g: Grid) -> None:
    if g.uneven:
        raise ValueError(f"the cube's points do not divide the cell evenly along {', '.join(g.uneven)} (a thinned "
                         "cube): the grid sums need CRYSTAL's own even grid")


# ------------------------------------------------------------------ the nuclear cusps
def core_shells(z: int):
    """(n, electrons, Slater exponent) of the closed core shells of element ``z`` (Slater's rules):
    1s from Li on, 2s2p from Na, 3s3p from K, 3d from Ga."""
    out = []
    if z > 2:
        out.append((1, 2, z - 0.30))
    if z > 10:
        out.append((2, 8, (z - 4.15) / 2.0))
    if z > 18:
        out.append((3, 8, (z - 11.25) / 3.0))
    if z > 30:
        out.append((3, 10, (z - 21.15) / 3.0))
    return out


def core_corrections(g: Grid, valence: Optional[Sequence[float]] = None):
    """The electrons the grid's sum over-counts (negative: misses) at each nucleus: a grid a tenth of a
    bohr apart cannot resolve the core density's cusp, so its sum near a nucleus is off by up to a few
    tenths of an electron. Each atom's core shells as Slater densities, N (2z)^(2n+1) r^(2n-2)
    e^(-2zr) / (4 pi (2n)!), summed over the grid's points around the nucleus (as the grid samples
    the real core) minus their exact count; subtracting it from the atom's basin removes the error the
    core carries (the valence density is smooth on the grid). An atom with a pseudopotential (valence
    below Z) has no core in the density and no correction."""
    import numpy as np
    zs = list(valence) if valence is not None else list(g.charges)
    inv = np.linalg.inv(g.voxel)
    steps = np.linalg.norm(g.voxel, axis=1)
    out = np.zeros(len(g.positions))
    for i, (z, R) in enumerate(zip(g.numbers, g.positions)):
        z = int(z)
        shells = core_shells(z)
        if not shells or (i < len(zs) and zs[i] < z - 0.5):
            continue
        rmax = max((2 * n + 12) / (2.0 * zeta) for n, _, zeta in shells)
        f = (R - g.origin) @ inv
        c = np.rint(f).astype(int)
        half = [min(40, int(math.ceil(rmax / s)) + 1) for s in steps]
        rng = [np.arange(c[k] - half[k], c[k] + half[k] + 1) for k in range(3)]
        I, J, K = np.meshgrid(*rng, indexing="ij")
        ijk = np.stack([I.ravel(), J.ravel(), K.ravel()], axis=1).astype(float)
        r = np.linalg.norm(g.origin + ijk @ g.voxel - R, axis=1)
        s = 0.0
        exact = 0.0
        for n, ne, zeta in shells:
            a = 2.0 * zeta
            rho = ne * a ** (2 * n + 1) / (4 * math.pi * math.factorial(2 * n)) * r ** (2 * n - 2) * np.exp(-a * r)
            s += float(rho.sum()) * g.dv
            exact += ne
        out[i] = s - exact
    return out


# ------------------------------------------------------------------ dipole
def area(g: Grid) -> Optional[float]:
    """The slab's in-plane cell area (bohr^2), else None."""
    import numpy as np
    if g.dimensionality != 2:
        return None
    a, b = [g.cell[k] for k in range(3) if g.periodic[k]]
    return float(np.linalg.norm(np.cross(a, b)))


def first_moment(g: Grid):
    """(electrons on the grid, their first moment sum rho r dV in e*bohr) from the 1D marginals."""
    import numpy as np
    d = g.data
    n = float(d.sum()) * g.dv
    m = float(d.sum()) * g.origin
    for k in range(3):
        marg = d.sum(axis=tuple(j for j in range(3) if j != k))
        m = m + float(np.dot(np.arange(marg.size), marg)) * g.voxel[k]
    return n, m * g.dv


#: a molecule's or polymer's grid dipole is kept only on a grid this fine (bohr between points) and with a
#: nuclear-cusp correction this small (e): coarser grids give EC 5.90 for 5.34 D, LiFSI-DEC 4.97 for 7.38 D
DIPOLE_MAX_SPACING_BOHR = 0.16
DIPOLE_MAX_CORE_CORRECTION = 0.1


def dipole(g: Grid, valence: Optional[Sequence[float]] = None, electrons: Optional[float] = None) -> Dict[str, Any]:
    """The dipole moment of the cell's charge along its non-periodic directions (physical sign: from the
    negative to the positive charge), about the center of the nuclear charge.

    ``valence``: the nuclear charge of each atom (the valence charge of a pseudopotential atom; default:
    the cube's charge column). ``electrons``: the cell's electron count. The grid's sum misses or
    over-counts part of each nuclear cusp (core_corrections): that error sits at the nucleus and is
    taken out there; what remains is scaled to the run's count. ``reliable``: the grid is fine enough
    (DIPOLE_MAX_SPACING_BOHR, DIPOLE_MAX_CORE_CORRECTION) for the result to be trusted."""
    import numpy as np
    _even(g)
    zs = np.asarray(valence if valence is not None else g.charges, float)
    if zs.size != len(g.positions):
        raise ValueError("one nuclear charge per atom is needed")
    q = float(zs.sum())
    r0 = (zs[:, None] * g.positions).sum(axis=0) / q
    n_grid, m = first_moment(g)
    delta = core_corrections(g, zs)
    n_c = n_grid - float(delta.sum())
    m_c = m - (delta[:, None] * g.positions).sum(axis=0)
    f = (float(electrons) / n_c) if electrons else 1.0
    p = -(m_c - n_c * r0) * f                          # sum Z (R - r0) = 0
    # keep the non-periodic part
    per = [g.cell[k] / np.linalg.norm(g.cell[k]) for k in range(3) if g.periodic[k]]
    if len(per) == 2:
        nrm = np.cross(per[0], per[1])
        nrm /= np.linalg.norm(nrm)
        if nrm[2] < 0:
            nrm = -nrm
        p = float(np.dot(p, nrm)) * nrm
    elif len(per) == 1:
        p = p - float(np.dot(p, per[0])) * per[0]
    elif len(per) == 3:
        raise ValueError("a 3D cell has no non-periodic direction")
    out = {"vector_au": p, "magnitude_au": float(np.linalg.norm(p)), "vector_debye": p * DEBYE_PER_AU,
           "magnitude_debye": float(np.linalg.norm(p)) * DEBYE_PER_AU, "electrons_grid": n_grid,
           "core_correction": float(delta.sum()), "electrons": float(electrons) if electrons else n_c,
           "nuclear_charge": q, "net_charge": q - (float(electrons) if electrons else n_c), "scale": f}
    out["spacing_bohr"] = float(np.linalg.norm(g.voxel, axis=1).max())
    out["reliable"] = (out["spacing_bohr"] <= DIPOLE_MAX_SPACING_BOHR
                       and abs(out["core_correction"]) <= DIPOLE_MAX_CORE_CORRECTION)
    A = area(g)
    if A:
        pz = float(np.dot(p, nrm))
        out["normal_au"] = pz
        out["area_bohr2"] = A
        # the potential energy of an electron above the slab minus below it: -4 pi p / A (hartree)
        out["step_ev"] = -4.0 * math.pi * pz / A * HARTREE_EV
    return out


# ------------------------------------------------------------------ Bader
_OFFSETS = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)]


def _shift(a, off, periodic, fill):
    """b[x] = a[x + off] (wrapping along periodic axes, ``fill`` past a non-periodic edge)."""
    import numpy as np
    b = a
    for ax, o in enumerate(off):
        if o == 0:
            continue
        if periodic[ax]:
            b = np.roll(b, -o, axis=ax)
        else:
            b = np.roll(b, -o, axis=ax)
            idx = [slice(None)] * 3
            idx[ax] = slice(-1, None) if o > 0 else slice(0, 1)
            if b is a:
                b = b.copy()
            b[tuple(idx)] = fill
    return b


def _flat(ijk, shape, periodic):
    """Flat indices of integer grid points (n, 3), wrapped along periodic axes, clipped otherwise."""
    import numpy as np
    out = np.empty(len(ijk), dtype=np.int64)
    ii = []
    for ax in range(3):
        v = ijk[:, ax]
        v = np.mod(v, shape[ax]) if periodic[ax] else np.clip(v, 0, shape[ax] - 1)
        ii.append(v)
    out[:] = (ii[0] * shape[1] + ii[1]) * shape[2] + ii[2]
    return out


def _unflat(flat, shape):
    import numpy as np
    i = flat // (shape[1] * shape[2])
    r = flat - i * shape[1] * shape[2]
    j = r // shape[2]
    k = r - j * shape[2]
    return np.stack([i, j, k], axis=1)


def _ongrid_pointers(rho, voxel, periodic):
    """For each point the flat index of its steepest-ascent neighbor among the 26 (gain per distance);
    a point with no higher neighbor points at itself."""
    import numpy as np
    shape = rho.shape
    best = np.zeros(shape)
    arg = np.full(shape, -1, dtype=np.int8)
    for t, off in enumerate(_OFFSETS):
        dist = float(np.linalg.norm(np.asarray(off, float) @ voxel))
        nb = _shift(rho, off, periodic, -np.inf)
        gain = (nb - rho) / dist
        better = gain > best
        best[better] = gain[better]
        arg[better] = t
    idx = np.arange(rho.size, dtype=np.int64).reshape(shape)
    ptr = idx.copy()
    offs = np.asarray(_OFFSETS, dtype=np.int64)
    has = arg >= 0
    src = idx[has]
    ijk = _unflat(src, shape) + offs[arg[has].astype(np.int64)]
    ptr[has] = _flat(ijk, shape, periodic)
    return ptr.ravel()


def _follow(ptr):
    """Pointer jumping: every point's terminal (a point that points at itself)."""
    import numpy as np
    p = ptr.copy()
    for _ in range(64):
        q = p[p]
        if np.array_equal(q, p):
            return p
        p = q
    return p


def _nearest_atom(points_bohr, g: Grid):
    """(index of the nearest atom, its distance in bohr) of each Cartesian point, with the periodic images."""
    import numpy as np
    per = [k for k in range(3) if g.periodic[k]]
    shifts = [np.zeros(3)]
    if per:
        import itertools
        rng = [(-1, 0, 1) if k in per else (0,) for k in range(3)]
        shifts = [np.asarray(s, float) @ g.cell for s in itertools.product(*rng)]
    best_d = np.full(len(points_bohr), np.inf)
    best_i = np.zeros(len(points_bohr), dtype=np.int64)
    for s in shifts:
        pos = g.positions + s
        d = np.linalg.norm(points_bohr[:, None, :] - pos[None, :, :], axis=2)
        i = d.argmin(axis=1)
        dd = d[np.arange(len(d)), i]
        upd = dd < best_d
        best_d[upd] = dd[upd]
        best_i[upd] = i[upd]
    return best_i, best_d


def _cart(ijk, g: Grid):
    return g.origin + ijk.astype(float) @ g.voxel


def _boundary(labels, periodic):
    """Points with a face neighbor in another basin."""
    import numpy as np
    b = np.zeros(labels.shape, bool)
    for ax in range(3):
        for o in (-1, 1):
            off = [0, 0, 0]
            off[ax] = o
            nb = _shift(labels, tuple(off), periodic, -2)
            b |= (nb != labels) & (nb != -2)
    return b


def _refine(rho, labels, voxel, periodic, todo, max_steps):
    """Near-grid trajectories (Tang, Sanville and Henkelman 2009) from the points ``todo`` (flat indices):
    each step follows the density gradient to the nearest grid point, a correction vector carrying what
    the rounding left out; a trajectory ends at a point whose label is trusted (not in ``todo``) and takes
    it. Returns the new labels of ``todo``."""
    import numpy as np
    shape = rho.shape
    flat_rho = rho.ravel()
    flat_lab = labels.ravel()
    trusted = np.ones(rho.size, bool)
    trusted[todo] = False
    G = voxel @ voxel.T                              # metric of the index coordinates
    Ginv = np.linalg.inv(G)
    cur = _unflat(todo, shape)
    corr = np.zeros((len(todo), 3))
    out = np.full(len(todo), -1, dtype=np.int64)
    active = np.arange(len(todo))
    e = np.eye(3, dtype=np.int64)
    for _ in range(max_steps):
        if active.size == 0:
            break
        c = cur[active]
        f0 = flat_rho[_flat(c, shape, periodic)]
        grad = np.empty((len(active), 3))
        for ax in range(3):
            fp = flat_rho[_flat(c + e[ax], shape, periodic)]
            fm = flat_rho[_flat(c - e[ax], shape, periodic)]
            grad[:, ax] = 0.5 * (fp - fm)
        step = grad @ Ginv                            # the ascent direction in index units
        mx = np.abs(step).max(axis=1)
        ok = mx > 0
        step[ok] /= mx[ok, None]
        rstep = np.rint(step)
        corr[active] += step - rstep
        bump = np.rint(corr[active])
        rstep += bump
        corr[active] -= bump
        nxt = c + rstep.astype(np.int64)
        f1 = flat_rho[_flat(nxt, shape, periodic)]
        # no ascent along the gradient (at a maximum, or a rounding into a lower point): the on-grid step
        bad = ~ok | (f1 <= f0)
        if bad.any():
            cb = c[bad]
            fb0 = f0[bad]
            best = np.zeros(len(cb))
            bstep = np.zeros((len(cb), 3), dtype=np.int64)
            for off in _OFFSETS:
                o = np.asarray(off, dtype=np.int64)
                dist = float(np.linalg.norm(o.astype(float) @ voxel))
                fn = flat_rho[_flat(cb + o, shape, periodic)]
                gain = (fn - fb0) / dist
                upd = gain > best
                best[upd] = gain[upd]
                bstep[upd] = o
            nxt[bad] = cb + bstep
            corr[active[bad]] = 0.0
            at_max = best <= 0
            if at_max.any():
                ids = active[bad][at_max]
                out[ids] = flat_lab[_flat(cb[at_max], shape, periodic)]
        for ax in range(3):
            if periodic[ax]:
                nxt[:, ax] = np.mod(nxt[:, ax], shape[ax])
            else:
                nxt[:, ax] = np.clip(nxt[:, ax], 0, shape[ax] - 1)
        cur[active] = nxt
        fl = _flat(nxt, shape, periodic)
        done = trusted[fl] & (out[active] < 0)
        out[active[done]] = flat_lab[fl[done]]
        active = active[out[active] < 0]
    if active.size:                                   # did not settle: keep the on-grid label
        out[active] = flat_lab[todo[active]]
    return out


def _ties(rho, labels, periodic, rel: float = 1e-6):
    """Points on a mirror plane of the grid: their two neighbors along an axis are in different basins
    and equally dense, so the ascent's choice between them is a tie. Such a point counts half to each
    (a symmetric pair of atoms then gets equal shares). Returns (mask of the split points, the two
    basins of each, their flat indices)."""
    import numpy as np
    shape = rho.shape
    mask = np.zeros(rho.size, bool)
    la, lb, idx = [], [], []
    for ax in range(3):
        e = [0, 0, 0]
        e[ax] = 1
        plus = _shift(labels, tuple(e), periodic, -2).ravel()
        minus = _shift(labels, tuple(-v for v in e), periodic, -2).ravel()
        rp = _shift(rho, tuple(e), periodic, np.nan).ravel()
        rm = _shift(rho, tuple(-v for v in e), periodic, np.nan).ravel()
        tie = ((plus != minus) & (plus >= 0) & (minus >= 0) & ~mask & (labels.ravel() >= 0)
               & (np.abs(rp - rm) <= rel * np.abs(rho.ravel())))
        k = np.flatnonzero(tie)
        mask[k] = True
        la.append(plus[k])
        lb.append(minus[k])
        idx.append(k)
    cat = (lambda xs: np.concatenate(xs).astype(np.int64))
    return mask, cat(la), cat(lb), cat(idx)


def bader(g: Grid, spin: Optional[Grid] = None, valence: Optional[Sequence[float]] = None,
          electrons: Optional[float] = None, vacuum: float = VACUUM_DENSITY, passes: int = 3) -> Dict[str, Any]:
    """Bader charges of the density grid ``g``: each point is assigned to the density maximum its
    steepest-ascent path reaches (on-grid, then near-grid trajectories for the points on the basins'
    boundaries, repeated until no label changes or ``passes`` times); each maximum belongs to the
    nearest atom (a maximum farther than 1 bohr from every atom is a non-nuclear attractor, counted).
    Points below ``vacuum`` e/bohr^3 are vacuum.

    ``valence``: each atom's nuclear (valence) charge, default the cube's charge column.
    ``electrons``: the cell's electron count. Each basin's population is its grid sum less its
    nucleus's cusp error (core_corrections), the rest scaled to the run's count (the grid's own
    total is reported). ``spin``: the SPIN cube on the same grid, integrated over the same basins.
    """
    import numpy as np
    _even(g)
    rho = g.data
    if rho.size > BADER_MAX_POINTS:
        raise ValueError(f"{rho.size} grid points: more than the {BADER_MAX_POINTS} the Bader partition takes here")
    nat = len(g.positions)
    if nat == 0:
        raise ValueError("the cube lists no atoms")
    shape, per = rho.shape, g.periodic
    ptr = _ongrid_pointers(rho, g.voxel, per)
    term = _follow(ptr)
    flat = rho.ravel()
    vac = flat < vacuum
    maxima = np.unique(term[~vac])
    atom_of, dist = _nearest_atom(_cart(_unflat(maxima, shape), g), g)
    lab_of_max = np.full(rho.size, -1, dtype=np.int64)
    lab_of_max[maxima] = atom_of
    labels = lab_of_max[term]
    labels[vac] = -1
    labels = labels.reshape(shape)
    changed_total = 0
    for _ in range(max(0, passes)):
        bnd = _boundary(labels, per).ravel() & ~vac
        todo = np.flatnonzero(bnd)
        if todo.size == 0:
            break
        new = _refine(rho, labels, g.voxel, per, todo, max_steps=4 * max(shape))
        lf = labels.ravel()
        diff = new != lf[todo]
        changed = int(diff.sum())
        lf[todo] = new
        labels = lf.reshape(shape)
        changed_total += changed
        if changed == 0:
            break
    lf = labels.ravel()
    zs = np.asarray(valence if valence is not None else g.charges, float)
    split = _ties(rho, labels, per)

    def integrate(values):
        w = np.where(split[0], 0.0, 1.0)
        own = lf >= 0
        tot = np.bincount(lf[own], weights=(values * w)[own], minlength=nat)[:nat]
        for lab in (split[1], split[2]):
            tot += np.bincount(lab, weights=0.5 * values[split[3]], minlength=nat)[:nat]
        return tot
    pop_grid = integrate(flat) * g.dv
    vol = integrate(np.ones_like(flat)) * g.dv * BOHR ** 3
    n_grid = float(flat.sum()) * g.dv
    n_vac = float(flat[vac].sum()) * g.dv
    delta = core_corrections(g, zs)                   # each nucleus's cusp error, in its own basin
    pop = pop_grid - delta
    scale = 1.0
    if electrons:
        inside = float(pop.sum())
        scale = (float(electrons) - n_vac) / inside if inside > 0 else 1.0
    pop_s = pop * scale
    charges = zs - pop_s
    out: Dict[str, Any] = {
        "charges": charges, "populations": pop_s, "volumes_A3": vol, "numbers": list(g.numbers),
        "electrons_grid": n_grid, "electrons_vacuum": n_vac, "scale": scale,
        "core_correction": float(delta.sum()), "core_corrections": delta,
        "maxima": int(maxima.size), "nonnuclear_maxima": int((dist > 1.0).sum()),
        "refined_points": int(changed_total), "tie_points": int(split[3].size), "vacuum_density": vacuum,
        "shape": list(shape), "dv_bohr3": g.dv,
    }
    if spin is not None:
        s = np.asarray(spin.data, float)
        if s.shape != rho.shape:
            raise ValueError(f"the SPIN cube's grid {s.shape} is not the density's {rho.shape}")
        sf = s.ravel()
        out["spins"] = integrate(sf) * spin.dv
        out["spin_total"] = float(sf.sum()) * spin.dv
    return out
