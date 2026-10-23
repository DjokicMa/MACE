"""Isosurfaces of a scalar grid by marching tetrahedra (numpy only).

Each grid cube is cut into the six tetrahedra of the Kuhn (Freudenthal)
triangulation, the ones around its main diagonal, which neighboring cubes
share face for face. A surface vertex sits on a tetrahedron edge, and two
tetrahedra that share an edge share its vertex, so the surface of a region
that stays inside the grid is closed: every triangle edge belongs to exactly
two triangles. Marching cubes has ambiguous cube faces that can leave holes;
tetrahedra have none.

The surface is oriented: triangles wind counterclockwise seen from outside
the region (the side where the field is below the level), in index space.
"""
from __future__ import annotations

import itertools
from typing import Tuple

import numpy as np

# cube corner c = x + 2y + 4z
_CORNER = np.array([[c & 1, (c >> 1) & 1, (c >> 2) & 1] for c in range(8)])
# the six tetrahedra: 0 -> one axis -> two axes -> 7, so the corners of each are ordered (each
# corner's offset is at least the one before it in every component)
_TETS = []
for _perm in itertools.permutations(range(3)):
    _p, _path = np.zeros(3, int), [0]
    for _ax in _perm:
        _p[_ax] = 1
        _path.append(int(_p[0] + 2 * _p[1] + 4 * _p[2]))
    _TETS.append(_path)
_TETS = np.array(_TETS)


def _cases():
    """For each of the 16 inside/outside patterns of a tetrahedron's corners, its triangles as
    lists of three (inside corner, outside corner) edges."""
    out = []
    for code in range(16):
        ins = [k for k in range(4) if code >> k & 1]
        outs = [k for k in range(4) if not code >> k & 1]
        if len(ins) == 1:
            tris = [[(ins[0], o) for o in outs]]
        elif len(ins) == 3:
            tris = [[(i, outs[0]) for i in ins]]
        elif len(ins) == 2:
            q = [(ins[0], outs[0]), (ins[0], outs[1]), (ins[1], outs[1]), (ins[1], outs[0])]
            tris = [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]
        else:
            tris = []
        out.append(tris)
    return out


_CASES = _cases()


def marching_tetrahedra(F: np.ndarray, level: float = 0.0) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """The surface where ``F`` crosses ``level``, ``F > level`` inside.

    Returns ``(verts, faces, (a, b, t))``: the vertices in grid-index
    coordinates, triangles as vertex indices, and for each vertex the flat
    indices of the two grid points of its edge and the fraction ``t`` from
    ``a`` to ``b`` (to interpolate anything else given on the grid, such as
    the gradient).
    """
    F = np.asarray(F, float)
    if F.ndim != 3 or min(F.shape) < 2:
        return np.zeros((0, 3)), np.zeros((0, 3), int), (np.zeros(0, int), np.zeros(0, int), np.zeros(0))
    # a grid point exactly at the level counts as outside, a little below it (no vertex on a corner)
    F = np.where(F == level, level - 1e-9 * max(1.0, abs(level)), F)
    inside = F > level
    nx, ny, nz = F.shape
    # only the cubes the surface crosses
    c8 = np.stack([inside[x:nx - 1 + x, y:ny - 1 + y, z:nz - 1 + z] for x, y, z in _CORNER], -1)
    cnt = c8.sum(-1)
    base = np.argwhere((cnt > 0) & (cnt < 8))
    flatF = F.ravel()
    strides = np.array([ny * nz, nz, 1])
    keys, tvals, tris_xyz = [], [], []
    for tet in _TETS:
        corner_xyz = base[:, None, :] + _CORNER[tet][None, :, :]            # (n, 4, 3)
        corner_flat = corner_xyz @ strides                                     # (n, 4)
        vals = flatF[corner_flat]
        code = ((vals > level) * (1 << np.arange(4))).sum(1)
        for c in range(1, 15):
            sel = np.flatnonzero(code == c)
            if not len(sel):
                continue
            cf, cx, cv = corner_flat[sel], corner_xyz[sel], vals[sel]
            ins = [k for k in range(4) if c >> k & 1]
            outs = [k for k in range(4) if not c >> k & 1]
            # inside to outside, the way the surface faces
            w = cx[:, outs].mean(1) - cx[:, ins].mean(1)
            for tri in _CASES[c]:
                kk, tt, pp = [], [], []
                for i, o in tri:
                    s, e = (i, o) if i < o else (o, i)       # the edge from its lower corner (tetrahedron order)
                    d = _CORNER[tet[e]] - _CORNER[tet[s]]
                    kk.append(cf[:, s] * 8 + int(d[0] + 2 * d[1] + 4 * d[2]))
                    t = (level - cv[:, s]) / (cv[:, e] - cv[:, s])
                    tt.append(t)
                    # the edge's middle, not the vertex: the winding is the same for any vertex
                    # on the edge, and a vertex near a corner would make the triangle degenerate
                    pp.append(cx[:, s] + 0.5 * d[None, :])
                n = np.cross(pp[1] - pp[0], pp[2] - pp[0])
                flip = (n * w).sum(1) < 0
                kk[1], kk[2] = np.where(flip, kk[2], kk[1]), np.where(flip, kk[1], kk[2])
                tt[1], tt[2] = np.where(flip, tt[2], tt[1]), np.where(flip, tt[1], tt[2])
                keys.append(np.stack(kk, 1))
                tvals.append(np.stack(tt, 1))
    if not keys:
        return np.zeros((0, 3)), np.zeros((0, 3), int), (np.zeros(0, int), np.zeros(0, int), np.zeros(0))
    K = np.concatenate(keys)
    T = np.concatenate(tvals)
    uk, first, faces = np.unique(K.ravel(), return_index=True, return_inverse=True)
    faces = faces.reshape(-1, 3)
    # in grid order (the vertices are), so a run of triangles stays in one place of the grid
    faces = faces[np.argsort(faces.min(1), kind="stable")]
    t = T.ravel()[first]
    a = uk // 8
    dcode = uk % 8
    d = np.stack([dcode & 1, (dcode >> 1) & 1, (dcode >> 2) & 1], 1)
    a_xyz = np.stack(np.unravel_index(a, F.shape), 1)
    b = (a_xyz + d) @ strides
    verts = a_xyz + t[:, None] * d
    return verts, faces, (a, b, t)


def interpolate(G: np.ndarray, edge: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """Values given on the grid (``G``: grid shape + trailing components) at the vertices."""
    a, b, t = edge
    g = G.reshape(-1, *G.shape[3:])
    tt = t.reshape(-1, *([1] * (g.ndim - 1)))
    return (1 - tt) * g[a] + tt * g[b]


def edges_shared(faces: np.ndarray) -> np.ndarray:
    """How many triangles share each undirected edge (2 everywhere on a closed surface)."""
    f = np.asarray(faces)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    e.sort(1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    return counts
