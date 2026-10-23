"""A periodic cube's isosurface joins across the cells of a supercell (1.1.6).

Marcus on HPCC: "cube files dont connect in supercells, the meshes are
disconnected completely at the UC boundary". The page drew one surface per
cell: 3Dmol marched one cell's grid and smoothed it as an open mesh, and the
cells were set the grid's points times its voxel apart, a voxel more than a
closed CRYSTAL grid's cell (100 planes span 99 voxels), so every cell's surface
stopped a voxel or more short of its neighbor's.

gui/volume.py now sends a periodic axis as whole voxels of one period plus the
closing plane (``meta['wrap']``, the cell from CRYSTAL's header) and
js/viewport/periodic_surface.js pads that grid across the faces, marches and
smooths it with 3Dmol's own MarchingCube and cuts it back to the cell. Here the
surfaces of a 2x2x2 supercell are made the page's way (3Dmol's vendored
MarchingCube in node) and every edge near an inner face of the supercell is
shared by two triangles: no seam. Synthetic grids (closed and open, a skewed
cell, both signs, every smoothing a preset uses, strided and refined), and the
example suite's diamond density and graphene slab.
"""
import itertools
import json
import lzma
import shutil
import subprocess
import types
from pathlib import Path

import numpy as np
import pytest

from mace.gui import volume as V

REPO = Path(__file__).resolve().parents[1]
SUITE = REPO / "examples" / "suite"
B = 0.529177210903

NODE_SCRIPT = r"""
import { createRequire } from "module";
import fs from "fs";
const require = createRequire(import.meta.url);
globalThis.window = globalThis; globalThis.self = globalThis;
globalThis.document = { createElement: () => ({ getContext: () => null, style: {} }), addEventListener() {}, querySelector() { return null; } };
if (!globalThis.navigator) globalThis.navigator = { userAgent: "node" };
const [vendor, module, input] = process.argv.slice(2);
require(vendor);
const MC = globalThis.$3Dmol.MarchingCube;
const PS = await import(module);
const out = [];
for (const J of JSON.parse(fs.readFileSync(input))) {
  const data = Float32Array.from(J.data), L = J.lattice, cells = J.cells;
  const tiles = [];
  let mesh = null;
  if (!J.naive) mesh = PS.periodicMesh(data, J.shape, J.wrap, J.iso, J.smooth, MC);
  for (let a = 0; a < cells[0]; a++) for (let b = 0; b < cells[1]; b++) for (let c = 0; c < cells[2]; c++) {
    const o = [0, 1, 2].map((j) => J.origin[j] + a * L[0][j] + b * L[1][j] + c * L[2][j]);
    if (!J.naive) { tiles.push({ V: PS.placed(mesh, o, J.voxels).map((p) => [p.x, p.y, p.z]), F: mesh.faces }); continue; }
    // what 1.1.5's page drew: 3Dmol's addIsosurface on the cell's grid, the cell moved by the points times the voxel
    const bits = new Uint8Array(data.length), [nx, ny, nz] = J.shape, W = J.voxels;
    for (let i = 0; i < data.length; i++) if ((J.iso >= 0 ? data[i] - J.iso : J.iso - data[i]) > 0) bits[i] = 2;
    const M = new globalThis.$3Dmol.Matrix4(W[0][0], W[1][0], W[2][0], o[0], W[0][1], W[1][1], W[2][1], o[1], W[0][2], W[1][2], W[2][2], o[2], 0, 0, 0, 1);
    const verts = [], faces = [];
    MC.march(bits, verts, faces, { fulltable: true, voxel: false, unitCube: { x: 1, y: 1, z: 1 }, origin: { x: o[0], y: o[1], z: o[2] }, matrix: M, nX: nx, nY: ny, nZ: nz });
    if (J.smooth > 0) MC.laplacianSmooth(J.smooth, verts, faces);
    tiles.push({ V: verts.map((p) => [p.x, p.y, p.z]), F: faces });
  }
  out.push(seams(tiles, L, cells, J.origin));
}
console.log(JSON.stringify(out));

/** Vertices merged by position (1e-5 Å, neighbors searched), edges used by one triangle only, sorted by
 * where they lie: by an outer face of the supercell (the surface is cut there), by an inner face (a seam),
 * or elsewhere (a hole). */
function seams(tiles, L, cells, org) {
  const Li = inv3(L), grid = new Map(), pts = [], edges = new Map();
  const vid = (p) => {
    const q = p.map((x) => Math.round(x * 1e4));
    for (const d of [[0, 0, 0], ...[-1, 0, 1].flatMap((i) => [-1, 0, 1].flatMap((j) => [-1, 0, 1].map((k) => [i, j, k])))]) {
      const hit = grid.get(q.map((x, k) => x + d[k]).join(","));
      if (hit !== undefined && Math.hypot(...[0, 1, 2].map((k) => pts[hit][k] - p[k])) < 1e-5) return hit;
    }
    grid.set(q.join(","), pts.length); pts.push(p);
    return pts.length - 1;
  };
  let triangles = 0;
  for (const { V, F } of tiles) for (let f = 0; f + 2 < F.length; f += 3) {
    triangles++;
    const t = [vid(V[F[f]]), vid(V[F[f + 1]]), vid(V[F[f + 2]])];
    for (const [u, v] of [[t[0], t[1]], [t[1], t[2]], [t[2], t[0]]]) {
      const k = u < v ? `${u}_${v}` : `${v}_${u}`;
      edges.set(k, (edges.get(k) || 0) + 1);
    }
  }
  const r = { triangles, outer: 0, seam: 0, hole: 0 };
  for (const [k, n] of edges) {
    if (n !== 1) continue;
    const [u, v] = k.split("_").map(Number), mid = [0, 1, 2].map((j) => (pts[u][j] + pts[v][j]) / 2 - org[j]);
    const f = [0, 1, 2].map((j) => mid[0] * Li[0][j] + mid[1] * Li[1][j] + mid[2] * Li[2][j]);
    if (f.some((x, j) => Math.abs(x) < 0.1 || Math.abs(x - cells[j]) < 0.1)) r.outer++;
    else if (f.some((x, j) => cells[j] > 1 && Math.abs(x - Math.round(x)) < 0.1)) r.seam++;
    else r.hole++;
  }
  return r;
}
function inv3(m) {
  const [a, b, c] = m, d = a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0]);
  return [[b[1] * c[2] - b[2] * c[1], -(a[1] * c[2] - a[2] * c[1]), a[1] * b[2] - a[2] * b[1]],
    [-(b[0] * c[2] - b[2] * c[0]), a[0] * c[2] - a[2] * c[0], -(a[0] * b[2] - a[2] * b[0])],
    [b[0] * c[1] - b[1] * c[0], -(a[0] * c[1] - a[1] * c[0]), a[0] * b[1] - a[1] * b[0]]].map((r) => r.map((x) => x / d));
}
"""


def _cube(n=24, a=8.0, closing=True, skew=False, signed=False, atoms=((0.03, 0.5, 0.5), (0.5, 0.97, 0.2), (0.6, 0.4, 0.99))):
    """A periodic CRYSTAL-like grid (bohr): Gaussians at atoms on or near the cell's faces, summed over the
    neighboring images; ``closing``: the grid repeats its first plane as its last (CRYSTAL's ECH3/POT3)."""
    L = np.diag([a, a, a]) if not skew else np.array([[a, 0, 0], [a / 2, a * 3 ** .5 / 2, 0], [0, 0, a]])
    m = n + 1 if closing else n
    f = np.stack(np.meshgrid(*[np.arange(m) / n] * 3, indexing="ij"), -1)
    d = np.zeros((m, m, m))
    for i, at in enumerate(atoms):
        sign = -1.0 if signed and i % 2 else 1.0
        for img in itertools.product((-1, 0, 1), repeat=3):
            r = (f - np.asarray(at) - np.asarray(img)) @ L
            d += sign * np.exp(-(r ** 2).sum(-1) / 0.8)
    vv = L / n
    return types.SimpleNamespace(data=d, voxel_vectors=vv, origin=np.zeros(3), nvoxels=np.array(d.shape),
                                 cell_parameters={k: float(np.linalg.norm(L[i])) for i, k in enumerate("abc")},
                                 lattice_vectors=vv * np.array(d.shape)[:, None], data_type="spin" if signed else "density")


def _case(c, quality="medium", iso=0.5, smooth=1, naive=False, points=None):
    """One input of the node script: the grid the page gets and the cell it repeats it by. ``naive``:
    what the page drew before (the file's grid, repeated by its points times its voxel)."""
    pbc = V._periodic_axes(c)
    if naive:
        g, vox, org = np.asarray(c.data, "<f4"), np.asarray(c.voxel_vectors) * B, np.asarray(c.origin) * B
        m = {"shape": list(g.shape), "voxels": vox.tolist(), "origin": org.tolist(), "wrap": [False] * 3}
        lat = np.asarray(c.lattice_vectors, float) * B
    else:
        g, m = V.decimate(c, quality, points=points)
        lat = V.cell_lattice(c, m)
    return {"data": g.ravel().tolist(), "shape": m["shape"], "voxels": m["voxels"], "origin": m["origin"], "wrap": m["wrap"],
            "lattice": lat.tolist(), "iso": iso, "smooth": smooth, "naive": naive, "cells": [2 if p else 1 for p in pbc]}


def _run(cases, tmp_path):
    script = tmp_path / "seams.mjs"
    script.write_text(NODE_SCRIPT)
    inp = tmp_path / "cases.json"
    inp.write_text(json.dumps(cases))
    vendor = REPO / "mace" / "gui" / "static" / "vendor" / "3dmol" / "3Dmol-min.js"
    module = (REPO / "mace" / "gui" / "static" / "js" / "viewport" / "periodic_surface.js").as_uri()
    p = subprocess.run(["node", str(script), str(vendor), module, str(inp)], capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr[-2000:]
    return json.loads(p.stdout.strip().splitlines()[-1])


needs_node = pytest.mark.skipif(not shutil.which("node"), reason="node not installed")


# ---------------------------------------------------------------- the grid the page gets
def test_a_periodic_axis_is_one_period_plus_the_closing_plane():
    for closing in (True, False):
        c = _cube(n=24, closing=closing)
        assert V.wrap_axes(c) == [(24, 24.0)] * 3
        g, m = V.decimate(c, "medium")
        assert m["wrap"] == [True] * 3 and list(g.shape) == [25, 25, 25] and m["stride"] == 1
        # the cell's own planes, unchanged, and the first again as the last
        assert np.array_equal(g[:24, :24, :24], np.asarray(c.data[:24, :24, :24], "<f4"))
        for ax in range(3):
            assert np.array_equal(np.take(g, 0, axis=ax), np.take(g, 24, axis=ax))
        # the cell the page repeats by is CRYSTAL's (8 bohr), not the points times the voxel
        assert np.allclose(np.linalg.norm(V.cell_lattice(c, m), axis=1), 8.0 * B, atol=1e-6)


def test_strided_and_odd_period_grids_still_close():
    # a budget below the grid: every 2nd point of a 51-plane period cannot land on the face, so the
    # period is resampled to 26 planes (linear, across the face too) and closed
    c = _cube(n=51, closing=True)
    g, m = V.decimate(c, "low", points=50_000)
    assert m["stride"] == 2 and list(g.shape) == [27, 27, 27] and m["wrap"] == [True] * 3
    assert np.allclose(np.linalg.norm(np.asarray(m["voxels"]), axis=1) * 26, 8.0 * B, atol=1e-6)
    for ax in range(3):
        assert np.array_equal(np.take(g, 0, axis=ax), np.take(g, 26, axis=ax))
    # the example suite keeps every 2nd point of CRYSTAL's closed grid: 50 planes over 49.5 voxels
    d = _cube(n=99, closing=True)
    d.data = d.data[::2, ::2, ::2]
    d.voxel_vectors = d.voxel_vectors * 2
    d.nvoxels = np.array(d.data.shape)
    (planes, period), = set(V.wrap_axes(d))
    assert planes == 50 and abs(period - 49.5) < 1e-9
    g, m = V.decimate(d, "medium")
    assert list(g.shape) == [51, 51, 51]
    assert np.allclose(np.linalg.norm(V.cell_lattice(d, m), axis=1), 8.0 * B, atol=1e-6)


def test_non_periodic_axes_are_as_before():
    # a slab's c (the formal 500 A) and a molecule are cropped and strided as in 1.1.5; nothing wraps
    c = _cube(n=24)
    c.cell_parameters = {"a": 8.0, "b": 8.0, "c": 500 / B}
    assert [w is not None for w in V.wrap_axes(c)] == [True, True, False]
    g, m = V.decimate(c, "medium")
    assert m["wrap"] == [True, True, False] and list(g.shape) == [25, 25, 25]
    assert np.allclose(np.asarray(m["voxels"])[2], c.voxel_vectors[2] * B)
    c = _cube(n=48)
    c.cell_parameters = {"a": 0.0, "b": 0.0, "c": 0.0}
    g, m = V.decimate(c, "low", points=50_000)
    assert m["wrap"] == [False] * 3 and m["stride"] == 2 and list(g.shape) == [25, 25, 25]
    assert np.array_equal(g, np.asarray(c.data[::2, ::2, ::2], "<f4"))


def test_a_refined_periodic_grid_keeps_the_cell():
    pytest.importorskip("scipy")
    c = _cube(n=24, closing=True)
    std, _ = V.decimate(c, "medium")
    g, m = V.decimate(c, "publication")
    assert m["upsample"] == 2 and list(g.shape) == [49, 49, 49]
    # the cell's own points kept, the face closed, the cell the same
    assert np.allclose(g[::2, ::2, ::2], std, atol=1e-5)
    assert np.array_equal(g[0], g[48])
    assert np.allclose(V.cell_lattice(c, m), V.cell_lattice(c, V.decimate(c, "medium")[1]), atol=1e-8)


# ---------------------------------------------------------------- the surfaces of a 2x2x2 supercell
@needs_node
def test_the_cells_surfaces_join_without_a_seam(tmp_path):
    cases, names = [], []
    for closing, skew in ((True, False), (False, False), (True, True)):
        c = _cube(closing=closing, skew=skew)
        for smooth in (0, 1, 2, 3):                       # every preset's smoothing passes (and the Advanced 0)
            cases.append(_case(c, smooth=smooth))
            names.append(f"closing={closing} skew={skew} smooth={smooth}")
    signed = _cube(signed=True)
    for iso in (0.3, -0.3):                                # a signed grid's two lobes
        cases.append(_case(signed, iso=iso))
        names.append(f"signed iso={iso}")
    cases.append(_case(_cube(n=51), quality="low", points=50_000))   # strided, resampled across the face
    names.append("strided")
    # 1.1.5's tiling of the same grids: the check sees its seams
    naive = [_case(_cube(closing=closing), smooth=1, naive=True) for closing in (True, False)]
    got = _run(cases + naive, tmp_path)
    for name, r in zip(names, got):
        assert r["triangles"] > 1000 and r["seam"] == 0 and r["hole"] == 0, (name, r)
    for r in got[len(cases):]:
        assert r["seam"] > 100, r


@needs_node
def test_a_refined_grid_joins_too(tmp_path):
    pytest.importorskip("scipy")
    got = _run([_case(_cube(), quality="publication", smooth=3), _case(_cube(skew=True), quality="high", smooth=2)], tmp_path)
    for r in got:
        assert r["triangles"] > 1000 and r["seam"] == 0 and r["hole"] == 0, r


def _suite_cube(tmp_path, entry, name):
    pytest.importorskip("plotly", reason="the cube engine imports plotly")
    xz = SUITE / entry / "CHARGE+POTENTIAL" / f"{name}.CUBE.xz"
    path = tmp_path / f"{name}.CUBE"
    path.write_bytes(lzma.decompress(xz.read_bytes()))
    return V.read(str(path))


@needs_node
def test_example_suite_diamond_and_graphene_join(tmp_path):
    """Real CRYSTAL23 grids from the example suite: diamond's density (3D, every 2nd point: 50 planes over
    49.5 voxels) and graphene's (a slab: a and b wrap, the vacuum c does not) at their default levels."""
    diamond = _suite_cube(tmp_path, "2-dense-3d/diamond", "diamond_charge+potential_DENS")
    graphene = _suite_cube(tmp_path, "3-slab-2d/graphene", "graphene_charge+potential_DENS")
    assert V.decimate(graphene, "medium")[1]["wrap"] == [True, True, False]
    cases = [_case(diamond, q, iso=V.default_level(diamond), smooth=s) for q, s in (("low", 1), ("medium", 1), ("high", 2))]
    cases.append(_case(graphene, "medium", iso=V.default_level(graphene), smooth=1))
    got = _run(cases + [_case(diamond, "medium", iso=V.default_level(diamond), smooth=1, naive=True)], tmp_path)
    for r in got[:-1]:
        assert r["triangles"] > 1000 and r["seam"] == 0 and r["hole"] == 0, r
    assert got[-1]["seam"] > 100, got[-1]
