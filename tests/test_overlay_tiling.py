"""Structure Analysis overlays over the display range (1.1.6).

The pore surfaces (accessible channels, inaccessible pockets), Miller planes
and a net were drawn in one unit cell only, even with the structure shown over
a 3x3x3 supercell: sodalite's beta cages sat in one cell of 27. Each overlay of
the cell is now drawn in every cell of the display range's cell box, as the
atoms are: its 3Dmol shape is built once for the cell and each other cell draws
a clone of the built mesh (sharing its geometry and material) moved by its
lattice vector (js/viewport/cell_tiles.js, shape_cache.keepBuilt).

Here the real overlay code (js/viewport/pores.js createOverlay) draws real pore
meshes (mace.analysis.porosity) through the vendored 3Dmol in node, in every
mesh style: one copy per cell of the box, at the right lattice vector, none
along a slab's open axis, built once, the pockets' colors shared by every copy,
the channels joined across the cell faces (also the wireframe's coarsened
vertices), and no pocket face doubled.
"""
import json
import lzma
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
JS = REPO / "mace" / "gui" / "static" / "js" / "viewport"
VENDOR = REPO / "mace" / "gui" / "static" / "vendor" / "3dmol" / "3Dmol-min.js"
SOD_OUT = REPO / "examples" / "suite" / "1-porous" / "SOD" / "OPT" / "SOD_opt.out.xz"

pytestmark = pytest.mark.skipif(not shutil.which("node"), reason="node not installed")

NODE_SCRIPT = r"""
import { createRequire } from "module";
import fs from "fs";
const require = createRequire(import.meta.url);
globalThis.window = globalThis; globalThis.self = globalThis;
globalThis.document = { createElement: () => ({ getContext: () => null, style: {} }), addEventListener() {}, removeEventListener() {},
  querySelector() { return null; }, getElementById() { return null; } };
if (!globalThis.navigator) globalThis.navigator = { userAgent: "node" };
const [vendor, jsdir, input] = process.argv.slice(2);
require(vendor);
const $3 = globalThis.$3Dmol;
const T = await import(jsdir + "/cell_tiles.js");
const P = await import(jsdir + "/pores.js");
// a 3Dmol viewer as the overlays use it: its shapes made as GLViewer makes them
const viewer = () => {
  const vw = { shapes: [], render() {}, removeShape(s) { vw.shapes = vw.shapes.filter((x) => x !== s); } };
  for (const k of ["addCustom", "addSphere", "addCylinder"]) {
    const m = k === "addCustom" ? "addCustom" : k;
    vw[k] = (spec) => { const s = new $3.GLShape(spec); s[m](spec); s.finalize(); vw.shapes.push(s); return s; };
  }
  return vw;
};
const group = () => ({ children: [], add(o) { o.parent = this; this.children.push(o); }, remove(o) { this.children = this.children.filter((x) => x !== o); o.parent = undefined; } });
const tr = (m) => [m.matrix.elements[12], m.matrix.elements[13], m.matrix.elements[14]];
const out = { tiles: [], cases: [] };
const J = JSON.parse(fs.readFileSync(input));
// the cells of a box, pure
for (const t of J.tiles) {
  const r = T.tileShifts(t.cell, t.pbc, t.box);
  out.tiles.push({ cells: r.cells, shifts: r.shifts, total: r.total, capped: r.capped,
    far: T.farFaceTiles(r, t.frac || [], t.pbc).cells });
}
for (const C of J.cases) {
  const vw = viewer(), pal = ["#e69f00", "#56b4e9", "#009e73", "#f0e442", "#0072b2", "#d55e00", "#cc79a7"];
  const css = (k) => (k.startsWith("--pore-pocket-") ? pal[Number(k.slice(14)) - 1] : k === "--pore-acc" ? "#3366cc" : k === "--pore-nac" ? "#cc3333" : "#888888");
  const O = P.createOverlay({ viewer: () => vw, activeDocId: () => 1, cssVar: css, docRev: null,
    display: () => ({ cell: C.cell, pbc: C.pbc, box: C.box }) });
  const key = O.show(C.kind, C.data, C.opts, 1, null);
  const it = O.get(key), L = it.tiles.cell || C.cell;
  const res = { name: C.name, n: it.tiles.shifts.length, shifts: it.tiles.shifts, shapes: [], list: O.list(1)[0].cells };
  for (const s of it.shapes) {
    const g = group();
    s.globj(g);
    const obj = s.renderedShapeObj, kids = obj.children;
    const meshes = kids.filter((m) => m.geometry === kids[0].geometry);
    s.globj(g);                                  // a second paint: nothing built again
    const again = s.renderedShapeObj === obj && obj.children.length === kids.length;
    // the cell's vertices (the shared geometry), and its triangles
    const V = [], cen = [];
    for (const gg of s.geo.geometryGroups) {
      const A = gg.vertexArray;
      for (let i = 0; i < gg.vertices; i++) V.push([A[3 * i], A[3 * i + 1], A[3 * i + 2]]);
      const F = gg.faceArray, base = V.length - gg.vertices;
      if (C.faces) for (let f = 0; f + 2 < gg.faceidx; f += 3) {
        const a = V[base + F[f]], b = V[base + F[f + 1]], c = V[base + F[f + 2]];
        cen.push([a, b, c]);
      }
    }
    res.shapes.push({ pass: s.macePass || "", wire: !!s.wireframe, children: kids.length, meshes: meshes.length,
      shared: kids.every((m) => m.geometry === kids[0].geometry || m.geometry === kids[1 % kids.length].geometry) && kids.every((m) => m.material === kids[0].material || m.material === kids[1 % kids.length].material),
      offsets: meshes.map(tr), auto: kids.some((m) => m.matrixAutoUpdate), again, color: s.color === undefined ? null : String(s.color),
      seam: C.seam ? seam(V, L, C.pbc) : null, doubled: C.faces ? doubled(cen, meshes.map(tr)) : null,
      colors: C.colors ? groupColors(s) : null, inside: inside(V, meshes.map(tr), s.boundingSphere) });
  }
  out.cases.push(res);
}
console.log(JSON.stringify(out));

/* The vertices on a periodic face of the cell that the next cell's copy of the opposite face does not meet
 * (1e-3 Å): 0 when the copies join. */
function seam(V, L, pbc) {
  const [a, b, c] = L, det = a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0]);
  const Li = [[b[1] * c[2] - b[2] * c[1], -(a[1] * c[2] - a[2] * c[1]), a[1] * b[2] - a[2] * b[1]],
    [-(b[0] * c[2] - b[2] * c[0]), a[0] * c[2] - a[2] * c[0], -(a[0] * b[2] - a[2] * b[0])],
    [b[0] * c[1] - b[1] * c[0], -(a[0] * c[1] - a[1] * c[0]), a[0] * b[1] - a[1] * b[0]]].map((r) => r.map((x) => x / det));
  const o = J.origin || [0, 0, 0];
  const fr = (p) => [0, 1, 2].map((k) => (p[0] - o[0]) * Li[0][k] + (p[1] - o[1]) * Li[1][k] + (p[2] - o[2]) * Li[2][k]);
  const key = (p) => p.map((x) => Math.round(x * 1000)).join(",");
  let face = 0, miss = 0;
  for (let k = 0; k < 3; k++) {
    if (!pbc[k]) continue;
    const lo = new Set();
    for (const p of V) if (fr(p)[k] < 1e-4) { lo.add(key(p)); for (const d of [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]) lo.add(key(p.map((x, j) => x + d[j] * 1e-3))); }
    for (const p of V) {
      if (fr(p)[k] <= 1 - 1e-4) continue;
      face++;
      const q = p.map((x, j) => x - L[k][j]);
      if (!lo.has(key(q))) miss++;
    }
  }
  return { face, miss };
}
/* Triangles of one copy that another copy draws at the same place (their corners to 1e-3 Å). A sliver of no
 * area (marching tetrahedra's, where the field is 0 on a grid point of a face) draws nothing and is left out. */
function doubled(tri, shifts) {
  const seen = new Map();
  let n = 0;
  const area = ([a, b, c]) => {
    const u = [0, 1, 2].map((k) => b[k] - a[k]), w = [0, 1, 2].map((k) => c[k] - a[k]);
    return Math.hypot(u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]) / 2;
  };
  const real = tri.filter((abc) => area(abc) > 1e-6);
  shifts.forEach((t, i) => {
    for (const abc of real) {
      const k = abc.map((p) => p.map((x, j) => Math.round((x + t[j]) * 1000)).join(",")).sort().join(";");
      const was = seen.get(k);
      if (was === undefined) seen.set(k, i); else if (was !== i) n++;
    }
  });
  return n;
}
/* Every copy's vertices inside the shape's bounding sphere (the slab, fog and zoom see them all). */
function inside(V, shifts, bs) {
  const c = bs.center;
  let far = 0;
  for (const t of shifts) for (const p of V) far = Math.max(far, Math.hypot(p[0] + t[0] - c.x, p[1] + t[1] - c.y, p[2] + t[2] - c.z));
  return far <= bs.radius + 1e-3;
}
/* Each geometry group's vertex color (the pockets' own colors, shared by every copy). */
function groupColors(s) {
  return s.geo.geometryGroups.map((g) => (g.colorArray && g.vertices ? [g.colorArray[0], g.colorArray[1], g.colorArray[2]].map((x) => Math.round(x * 255)) : null));
}
"""


def _run(payload, tmp_path):
    script = tmp_path / "tiles.mjs"
    script.write_text(NODE_SCRIPT)
    inp = tmp_path / "in.json"
    inp.write_text(json.dumps(payload))
    p = subprocess.run(["node", str(script), str(VENDOR), JS.as_uri(), str(inp)], capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stderr[-3000:]
    return json.loads(p.stdout.strip().splitlines()[-1])


def _lattice_shifts(cell, cells):
    return [list(np.asarray(c) @ np.asarray(cell)) for c in cells]


# ---------------------------------------------------------------- the cells of a box
def test_one_tile_per_cell_of_the_display_box(tmp_path):
    skew = [[7.576, 0.0, 0.0], [-2.525, 7.143, 0.0], [-2.525, -3.571, 6.186]]
    cubic = [[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]]
    cases = [
        {"cell": skew, "pbc": [True] * 3, "box": {"lo": [0, 0, 0], "hi": [3, 3, 3]}},
        {"cell": sum(skew, []), "pbc": [True] * 3, "box": {"lo": [-1, 0, 0], "hi": [1, 1, 2]}},          # a flat cell, a box from -1
        {"cell": cubic, "pbc": [True, True, False], "box": {"lo": [0, 0, 0], "hi": [2, 2, 3]}},            # a slab: none along c
        {"cell": cubic, "pbc": [True, False, False], "box": {"lo": [0, 0, 0], "hi": [3, 3, 3]}},           # a polymer: along a only
        {"cell": cubic, "pbc": [True] * 3, "box": {"lo": [0, 0, 0], "hi": [5, 5, 5]}},                    # over the limit
        {"cell": cubic, "pbc": [True] * 3, "box": None},                                                  # no box: the cell
        # a polygon on the far c face: drawn in the last layer of cells only (the next cell's c = 0 plane)
        {"cell": cubic, "pbc": [True] * 3, "box": {"lo": [0, 0, 0], "hi": [2, 2, 2]}, "frac": [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]},
    ]
    r = _run({"tiles": cases, "cases": []}, tmp_path)["tiles"]
    full = lambda lo, hi: [[a, b, c] for a in range(lo[0], hi[0]) for b in range(lo[1], hi[1]) for c in range(lo[2], hi[2])]
    assert r[0]["cells"] == full([0, 0, 0], [3, 3, 3]) and r[0]["total"] == 27 and not r[0]["capped"]
    assert np.allclose(r[0]["shifts"], _lattice_shifts(skew, r[0]["cells"]))
    assert r[1]["cells"] == full([-1, 0, 0], [1, 1, 2])
    assert np.allclose(r[1]["shifts"], _lattice_shifts(skew, r[1]["cells"]))
    assert r[2]["cells"] == full([0, 0, 0], [2, 2, 1])
    assert r[3]["cells"] == [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
    assert r[4]["total"] == 125 and r[4]["capped"] and len(r[4]["cells"]) == 64
    assert r[5]["cells"] == [[0, 0, 0]] and r[5]["shifts"] == [[0, 0, 0]]
    assert r[6]["far"] == [c for c in full([0, 0, 0], [2, 2, 2]) if c[2] == 1]


# ---------------------------------------------------------------- real pore meshes
def _structures():
    from ase.spacegroup import crystal
    from mace.analysis import structure
    lta = structure.from_atoms(crystal(["O", "O", "O", "Si"], basis=[(0, 0.2122, 0.5), (0.1103, 0.1103, 0.3384), (0, 0.2967, 0.2967),
                                                                    (0, 0.1823, 0.3684)], spacegroup=221, cellpar=[11.919] * 3 + [90] * 3))
    return lta


@pytest.fixture(scope="module")
def sod_suite(tmp_path_factory):
    """The example suite's sodalite (its CRYSTAL optimization, primitive cell: one beta cage)."""
    pytest.importorskip("scipy")
    from mace.analysis import structure
    p = tmp_path_factory.mktemp("sod") / "SOD_opt.out"
    p.write_bytes(lzma.decompress(SOD_OUT.read_bytes()))
    return structure.load(p)


STYLES = [("solid", {}), ("flat", {"shading": "flat"}), ("single", {"sides": "single"}), ("double", {"sides": "double"}),
          ("wireframe", {"style": "wireframe"}), ("edges", {"style": "edges"}), ("points", {"style": "points"})]


@pytest.mark.parametrize("quality", ["standard", "publication"])
def test_sodalite_pockets_over_3x3x3(sod_suite, tmp_path, quality):
    pytest.importorskip("scipy")
    from mace.analysis import porosity
    st = sod_suite
    r = porosity.analyze(st, probe="N2", spacing=0.25, render=True, mesh={"quality": quality})
    rd = r["render"]
    assert rd["meshes"]["inaccessible"]["n_pockets"] == 1 and rd["meshes"]["accessible"] is None
    cell = st.cell.tolist()
    box = {"lo": [0, 0, 0], "hi": [3, 3, 3]}
    styles = STYLES if quality == "standard" else STYLES[:1]
    cases = [{"name": n, "kind": "pores", "data": rd, "opts": {"colorPockets": True, **o}, "cell": cell, "pbc": [True] * 3, "box": box,
              "faces": n == "solid", "colors": n == "solid"} for n, o in styles]
    out = _run({"tiles": [], "cases": cases, "origin": rd["origin"]}, tmp_path)["cases"]
    cells = [[a, b, c] for a in range(3) for b in range(3) for c in range(3)]
    want = np.asarray(_lattice_shifts(cell, cells))
    for c in out:
        assert c["n"] == 27 and c["list"] == 27, c["name"]
        assert np.allclose(c["shifts"], want, atol=1e-9), c["name"]
        assert c["shapes"], c["name"]
        for s in c["shapes"]:
            # one built mesh, 27 copies at the lattice vectors, sharing its geometry and material, built once
            assert s["meshes"] == 27 and s["shared"] and s["again"] and not s["auto"] and s["inside"], (c["name"], s)
            assert np.allclose(s["offsets"], want, atol=1e-5), (c["name"], s["pass"])
    solid = out[0]["shapes"][0]
    # the pocket's own color (Color pockets individually), in the shared geometry: the same in every cell
    assert solid["color"] is None and all(x == [230, 159, 0] for x in solid["colors"]), solid["colors"][:3]
    # each cage whole once per cell: no triangle drawn twice over the 27 copies
    assert solid["doubled"] == 0


def _sod_primitive():
    """IZA sodalite in its primitive (rhombohedral, 109.47 degree) cell."""
    from ase.spacegroup import crystal
    from mace.analysis import structure
    st = structure.from_atoms(crystal(["Si", "O"], basis=[(0.25, 0.5, 0.0), (0.1467, 0.5, 0.8533)], spacegroup=229,
                                      cellpar=[8.965] * 3 + [90] * 3))
    P = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]]) * st.cell[0, 0] / 2
    f = np.mod(st.cart @ np.linalg.inv(P) + 1e-6, 1.0) - 1e-6
    _, keep = np.unique(np.round(f, 4) % 1.0, axis=0, return_index=True)
    return structure.Structure(P, f[keep], st.numbers[keep], (True, True, True))


@pytest.mark.parametrize("which", ["LTA N2", "SOD primitive, r = 1 A"])
def test_channels_join_across_the_cell_faces(tmp_path, which):
    pytest.importorskip("scipy")
    pytest.importorskip("ase")
    from mace.analysis import porosity
    # LTA's alpha cages (a cubic cell); sodalite open to a 1 A probe in its skewed primitive cell
    st, probe = (_structures(), "N2") if which == "LTA N2" else (_sod_primitive(), 1.0)
    rd = porosity.analyze(st, probe=probe, spacing=0.25, render=True)["render"]
    assert rd["meshes"]["accessible"]
    cell = st.cell.tolist()
    box = {"lo": [0, 0, 0], "hi": [2, 2, 2]}
    cases = [{"name": n, "kind": "pores", "data": rd, "opts": {"inaccessible": False, **o}, "cell": cell, "pbc": [True] * 3, "box": box,
              "seam": True, "faces": n == "solid"} for n, o in STYLES if n in ("solid", "wireframe", "edges")]
    out = _run({"tiles": [], "cases": cases, "origin": rd["origin"]}, tmp_path)["cases"]
    for c in out:
        assert c["n"] == 8
        for s in c["shapes"]:
            assert s["meshes"] == 8 and s["shared"], (c["name"], s)
            # every vertex on a face of the cell meets the next cell's copy of the opposite face (the wireframe's
            # merged vertices too): the channels run on without a gap
            assert s["seam"]["face"] > 20 and s["seam"]["miss"] == 0, (c["name"], s["pass"], s["seam"])
            if s["doubled"] is not None:
                assert s["doubled"] == 0


def test_a_slab_has_no_copies_along_its_vacuum(tmp_path):
    pytest.importorskip("scipy")
    from ase.build import fcc111
    from mace.analysis import porosity, structure
    at = fcc111("Cu", size=(2, 2, 3), vacuum=6.0)
    at.pbc = (True, True, False)
    st = structure.from_atoms(at)
    rd = porosity.analyze(st, probe="He", spacing=0.3, render=True)["render"]
    cell = st.cell.tolist()
    case = {"name": "slab", "kind": "pores", "data": rd, "opts": {}, "cell": cell, "pbc": [True, True, False],
            "box": {"lo": [0, 0, 0], "hi": [2, 2, 3]}}
    out = _run({"tiles": [], "cases": [case], "origin": rd["origin"]}, tmp_path)["cases"][0]
    assert out["n"] == 4 and all(abs(t[2]) < 1e-9 for t in out["shifts"])
    for s in out["shapes"]:
        assert s["meshes"] == 4


def test_net_and_planes_are_tiled_and_the_crystal_shape_is_not(tmp_path):
    cubic = [[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0]]
    box = {"lo": [0, 0, 0], "hi": [2, 2, 2]}
    net = {"frac": [[0, 0, 0]], "edges": [[0, 0, [1, 0, 0]], [0, 0, [0, 1, 0]], [0, 0, [0, 0, 1]]], "cell": cubic}
    # (001) at 0: the planes c = 0 and c = 1 of the cell (planes_in_box), so c = 1 is drawn in the top layer only
    planes = [{"n": 0, "polygon": [[0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 4, 0]]}, {"n": 1, "polygon": [[0, 0, 4], [4, 0, 4], [4, 4, 4], [0, 4, 4]]}]
    mesh = {"vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], "triangles": [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], "cell": cubic}
    cases = [{"name": "net", "kind": "net", "data": net, "opts": {}, "cell": cubic, "pbc": [True] * 3, "box": box},
             {"name": "planes", "kind": "planes", "data": planes, "opts": {}, "cell": cubic, "pbc": [True] * 3, "box": box},
             {"name": "mesh", "kind": "mesh", "data": mesh, "opts": {}, "cell": cubic, "pbc": [True] * 3, "box": box}]
    out = {c["name"]: c for c in _run({"tiles": [], "cases": cases}, tmp_path)["cases"]}
    assert out["net"]["n"] == 8 and all(s["meshes"] == 8 for s in out["net"]["shapes"]) and len(out["net"]["shapes"]) == 4
    p0, p1 = out["planes"]["shapes"]
    assert p0["meshes"] == 8 and p1["meshes"] == 4
    assert sorted(round(t[2], 6) for t in p1["offsets"]) == [4.0] * 4
    assert out["mesh"]["n"] == 1 and all(s["meshes"] == 1 for s in out["mesh"]["shapes"])
