"""Evaluate orbitals (molecular or Bloch) on regular grids.

psi_k(r) = sum_R exp(+2 pi i k.n) sum_mu c_mu phi_mu(r - tau_mu - R)

The sum runs over every atom image whose functions reach the box. For each
image the work is limited to the sub-block of grid indices inside the image's
cutoff sphere (computed from the inverse voxel vectors), so the cost grows
with the box volume times the number of images that overlap it, not with
(cells x points).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from .basis import CART, Basis

# A basis function is cut off where its most diffuse primitive drops below this.
VALUE_TOL = 1e-9


@dataclass
class GridSpec:
    origin: np.ndarray       # (3,) bohr
    vecs: np.ndarray         # (3, 3) voxel step vectors as rows (bohr)
    shape: Tuple[int, int, int]

    @property
    def npoints(self) -> int:
        return int(np.prod(self.shape))

    @property
    def voxel_volume(self) -> float:
        return float(abs(np.linalg.det(self.vecs)))

    def points(self) -> np.ndarray:
        I = np.indices(self.shape).reshape(3, -1).T
        return self.origin + I @ self.vecs


def _shell_cutoff(exps: np.ndarray, coefs: np.ndarray, l: int, tol: float = VALUE_TOL) -> float:
    """Radius (bohr) beyond which every primitive of the shell is below tol."""
    r = 0.0
    for a, c in zip(exps, coefs):
        c = abs(c)
        if c == 0.0:
            continue
        # solve c r^l exp(-a r^2) = tol by fixed-point on r^2 = (ln(c/tol) + l ln r)/a
        r2 = max(math.log(max(c, 1e-300) / tol) / a, 1e-6)
        for _ in range(20):
            rr = math.sqrt(r2)
            nxt = max((math.log(max(c, 1e-300) / tol) + l * math.log(max(rr, 1e-6))) / a, 1e-6)
            if abs(nxt - r2) < 1e-6:
                break
            r2 = nxt
        r = max(r, math.sqrt(r2))
    return r


def build_box(basis: Basis, cells: Sequence[int] = (1, 1, 1), spacing: float = 0.25,
              pad: float = 5.0, max_points: Optional[int] = None, center_cells: bool = True) -> GridSpec:
    """Grid for a molecule (atoms +- pad) or n_i cells along each periodic axis.

    Periodic axes: ``ceil(|a_i|/spacing)`` points per cell, no duplicated edge
    point (the usual periodic cube). Non-periodic axes: the atoms' extent along
    that Cartesian direction +- pad. With ``max_points`` the spacing grows until
    the grid fits.
    """
    cells = tuple(int(max(1, c)) for c in cells)
    while True:
        spec = _box(basis, cells, spacing, pad, center_cells)
        if max_points is None or spec.npoints <= max_points:
            return spec
        spacing *= (spec.npoints / max_points) ** (1.0 / 3.0) * 1.02


def _box(basis: Basis, cells, spacing, pad, center_cells) -> GridSpec:
    pos = basis.pos
    A = basis.lattice_axes()
    per = basis.periodic
    vecs = np.zeros((3, 3))
    shape = [0, 0, 0]
    origin = np.zeros(3)
    if not per:
        lo = pos.min(axis=0) - pad
        hi = pos.max(axis=0) + pad
        for i in range(3):
            n = int(math.ceil((hi[i] - lo[i]) / spacing)) + 1
            vecs[i, i] = (hi[i] - lo[i]) / (n - 1)
            shape[i] = n
        return GridSpec(lo, vecs, tuple(shape))
    for i in range(3):
        if i in per:
            n = max(2, int(math.ceil(np.linalg.norm(A[i]) / spacing)))
            vecs[i] = A[i] / n
            shape[i] = n * cells[i]
        else:
            lo, hi = pos[:, i].min() - pad, pos[:, i].max() + pad
            n = int(math.ceil((hi - lo) / spacing)) + 1
            vecs[i, i] = (hi - lo) / (n - 1)
            shape[i] = n
            origin[i] = lo
    if center_cells:
        # start the periodic box half a supercell before the atoms' centroid
        M = np.array([A[i] for i in per])
        cen = pos.mean(axis=0)
        frac = np.linalg.lstsq(M.T, cen, rcond=None)[0]
        shift = np.zeros(3)
        for j, i in enumerate(per):
            shift += (frac[j] - cells[i] / 2.0) * A[i]
        nonper = [i for i in range(3) if i not in per]
        # remove any component of the shift along non-periodic Cartesian directions
        for i in nonper:
            shift[i] = 0.0
        origin = origin + shift
    return GridSpec(origin, vecs, tuple(shape))


def evaluate(basis: Basis, C: np.ndarray, grid: GridSpec, kfrac=None,
             tol: float = VALUE_TOL) -> np.ndarray:
    """psi values (m, n0, n1, n2), complex, for the columns of C (nbf, m)."""
    C = np.asarray(C)
    m = C.shape[1]
    psi = np.zeros((m,) + tuple(grid.shape), complex)
    if m == 0:
        return psi
    k = np.zeros(3) if kfrac is None else np.asarray(kfrac, float)
    A = basis.lattice_axes()
    W = np.linalg.inv(grid.vecs)            # cartesian -> fractional grid index (columns)
    wn = np.linalg.norm(W, axis=0)          # index change per bohr along each grid axis (upper bound)
    shape = np.array(grid.shape)

    # per atom: shells, cutoffs
    atoms = {}
    for sh in basis.shells:
        rc = _shell_cutoff(sh["exps"], sh["coefs"] * max(1.0, float(np.abs(C[sh["off"]:sh["off"] + sh["n"]]).max())),
                           sh["l"], tol)
        atoms.setdefault(sh["atom"], []).append((sh, rc))
    rc_max = max(rc for lst in atoms.values() for _, rc in lst)

    # candidate cells: images of any atom that can touch the box
    if basis.periodic:
        corners = np.array([[i, j, l] for i in (0, 1) for j in (0, 1) for l in (0, 1)]) * (shape - 1)
        cpts = grid.origin + corners @ grid.vecs
        dia = np.linalg.norm(cpts.max(axis=0) - cpts.min(axis=0))
        reach = dia + rc_max + float(np.max(np.linalg.norm(basis.pos - basis.pos.mean(axis=0), axis=1)))
        box_c = cpts.mean(axis=0)
        base = basis.cells_within(reach + np.linalg.norm(box_c - basis.pos.mean(axis=0)) + 1.0)
    else:
        base = np.zeros((1, 3), int)
    Rs = base @ A
    phases = np.exp(2j * math.pi * (base @ k))
    real_phase = bool(np.all(np.abs(phases.imag) < 1e-12))
    acc_dtype = float if (real_phase and not np.iscomplexobj(C)) else complex

    for a, shl in atoms.items():
        rc_atom = max(rc for _, rc in shl)
        centers = basis.pos[a][None, :] + Rs                       # (ncell, 3)
        t = (centers - grid.origin) @ W                            # fractional index of each image
        h = rc_atom * wn
        lo = np.maximum(np.ceil(t - h).astype(int), 0)
        hi = np.minimum(np.floor(t + h).astype(int), shape - 1)
        ok = np.all(hi >= lo, axis=1)
        for ci in np.nonzero(ok)[0]:
            l0, h0 = lo[ci], hi[ci]
            i0 = np.arange(l0[0], h0[0] + 1)
            i1 = np.arange(l0[1], h0[1] + 1)
            i2 = np.arange(l0[2], h0[2] + 1)
            # displacement from the image center for the sub-block, built by broadcasting
            base_pt = grid.origin + l0 @ grid.vecs - centers[ci]
            d = (base_pt[None, None, None, :]
                 + (i0 - l0[0])[:, None, None, None] * grid.vecs[0]
                 + (i1 - l0[1])[None, :, None, None] * grid.vecs[1]
                 + (i2 - l0[2])[None, None, :, None] * grid.vecs[2])
            d = d.reshape(-1, 3)
            r2 = np.einsum("ij,ij->i", d, d)
            sel = np.nonzero(r2 < rc_atom * rc_atom)[0]
            if sel.size == 0:
                continue
            ds, r2s = d[sel], r2[sel]
            acc = np.zeros((m, sel.size), acc_dtype)
            for sh, rc in shl:
                inner = r2s < rc * rc
                if not inner.any():
                    continue
                if inner.all():
                    dd, rr = ds, r2s
                    idx = None
                else:
                    idx = np.nonzero(inner)[0]
                    dd, rr = ds[idx], r2s[idx]
                rad = np.exp(-np.outer(sh["exps"], rr)).T @ sh["coefs"]        # (np,)
                carts = CART[sh["l"]]
                mon = np.empty((len(carts), rr.size))
                for j, (e0, e1, e2) in enumerate(carts):
                    v = rad.copy()
                    if e0:
                        v *= dd[:, 0] ** e0
                    if e1:
                        v *= dd[:, 1] ** e1
                    if e2:
                        v *= dd[:, 2] ** e2
                    mon[j] = v
                phi = sh["T"] @ mon                                             # (nfunc, np)
                contrib = C[sh["off"]:sh["off"] + sh["n"]].T @ phi              # (m, np)
                if idx is None:
                    acc += contrib
                else:
                    acc[:, idx] += contrib
            ph = phases[ci]
            sub = psi[:, l0[0]:h0[0] + 1, l0[1]:h0[1] + 1, l0[2]:h0[2] + 1].reshape(m, -1)
            sub[:, sel] += (ph.real if real_phase else ph) * acc
            psi[:, l0[0]:h0[0] + 1, l0[1]:h0[1] + 1, l0[2]:h0[2] + 1] = sub.reshape(
                (m, len(i0), len(i1), len(i2)))
    return psi
