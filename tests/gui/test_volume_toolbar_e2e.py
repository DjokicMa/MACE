"""The toolbar's display options work in the volume views (cube, orbitals)
(1.1.6: they were grayed out with "The volume view has its own display
options").

Supercell, unit cell, bonds, labels, boundary atoms, axes and View Along are
enabled for a charge-density cube and for an orbital, each does what it does
in the structure view (read back from what 3Dmol draws), and a supercell
replicates the isosurface with the atoms. An orbital's Cells field is the
toolbar's display range, not a second control. A volume of a molecule has no
cell: the options that need one say why. Chromium and Firefox. The cubes are
real CRYSTAL23 output from the test/ corpus (skipped without it); the
orbitals are the real runs under tests/data/orbitals.
"""
import shutil
import time
from pathlib import Path

import pytest

import guiharness as H

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("plotly", reason="the cube engine imports plotly")
np = pytest.importorskip("numpy")

CORPUS = H.REPO_ROOT / "test" / "ECH3POT3"
MOL = "DEC_MOLECULE_OPT_symm_HSESOL3C_SOLDEF2MSVP_opt_HSESOL3C_optimized_sp_HSESOL3C_optimized_charge+potential"
SLAB = "4LG_FSI_TopMiddle_2x2_ABAB_charge+potential"
DATA = Path(__file__).resolve().parents[1] / "data" / "orbitals"
VV = "window.MACE.state.viewers['viewer-volume']"
SI = "orb/si_g/si_g_00001_K000_real.molden"
SLAB_ORB = "orb/slab/slab_f_00003_K200_real.molden"


BULK_CELL = (8.0, 8.0, 20.0)            # bohr: a tall c that is still periodic (a slab's is the formal 500 A)


def write_bulk_cube(path):
    """A 3D-periodic CRYSTAL-style cube: the cell on the second comment line, two atoms near
    the cell's center and a density that falls off well inside the cell."""
    n = (16, 16, 32)
    step = [BULK_CELL[k] / n[k] for k in range(3)]
    atoms = [(8, (0.5, 0.5, 0.45)), (22, (0.5, 0.5, 0.6))]
    g = np.indices(n).reshape(3, -1).T * step
    rho = np.zeros(len(g))
    for _, f in atoms:
        c = np.array(f) * BULK_CELL
        rho += np.exp(-((g - c) ** 2).sum(1) / (2 * 0.7 ** 2))
    lines = [" Charge density - 3D GRID - GAUSSIAN CUBE FORMAT bulk",
             "  {:.8f}  {:.8f}  {:.8f}  90.000000  90.000000  90.000000".format(*BULK_CELL),
             "{:5d}  0.000000  0.000000  0.000000".format(len(atoms))]
    lines += ["{:5d}  {:.6f}  {:.6f}  {:.6f}".format(n[k], *[step[k] if j == k else 0.0 for j in range(3)]) for k in range(3)]
    lines += ["{:5d}  {:.6f}  {:.6f}  {:.6f}  {:.6f}".format(z, float(z), *(np.array(f) * BULK_CELL)) for z, f in atoms]
    vals = rho.reshape(n)
    for i in range(n[0]):
        for j in range(n[1]):
            row = vals[i, j]
            lines += [" ".join("{:.5E}".format(x) for x in row[k:k + 6]) for k in range(0, n[2], 6)]
    path.write_text("\n".join(lines) + "\n")


def write_face_cube(path, signed=False, n=20, a=8.0):
    """A 3D-periodic CRYSTAL-style cube whose density crosses the cell's faces: atoms on and near
    them, the Gaussians summed over the neighboring images, and the closing plane (n + 1 planes
    over the cell, as CRYSTAL writes ECH3). ``signed``: the second atom's lobe is negative (a spin)."""
    import itertools
    atoms = [(0.02, 0.5, 0.5), (0.5, 0.97, 0.03), (0.6, 0.4, 0.99)]
    f = np.stack(np.meshgrid(*[np.arange(n + 1) / n] * 3, indexing="ij"), -1)
    rho = np.zeros((n + 1,) * 3)
    for i, at in enumerate(atoms):
        for img in itertools.product((-1, 0, 1), repeat=3):
            rho += (-1.0 if signed and i == 1 else 1.0) * np.exp(-(((f - np.array(at) - img) * a) ** 2).sum(-1) / 1.2)
    kind = "Spin density" if signed else "Charge density"
    lines = [f" {kind} - 3D GRID - GAUSSIAN CUBE FORMAT face",
             "  {:.8f}  {:.8f}  {:.8f}  90.000000  90.000000  90.000000".format(a, a, a),
             "{:5d}  0.000000  0.000000  0.000000".format(len(atoms))]
    lines += ["{:5d}  {:.6f}  {:.6f}  {:.6f}".format(n + 1, *[a / n if j == k else 0.0 for j in range(3)]) for k in range(3)]
    lines += ["    6  6.000000  {:.6f}  {:.6f}  {:.6f}".format(*(np.array(at) * a)) for at in atoms]
    for i in range(n + 1):
        for j in range(n + 1):
            row = rho[i, j]
            lines += [" ".join("{:.5E}".format(x) for x in row[k:k + 6]) for k in range(0, n + 1, 6)]
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture(scope="module", params=["chromium", "firefox"])
def app(request, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("voltoolbar")
    root = H.examples_copy(tmp)
    shutil.copytree(DATA, root / "orb")
    (root / "vol").mkdir()
    if (CORPUS / f"{SLAB}_DENS.CUBE").is_file():
        shutil.copyfile(CORPUS / f"{SLAB}_DENS.CUBE", root / "vol" / "slab_DENS.CUBE")
        shutil.copyfile(CORPUS / f"{MOL}_DENS.CUBE", root / "vol" / "mol_DENS.CUBE")
        shutil.copyfile(CORPUS / f"{MOL}_POT.CUBE", root / "vol" / "mol_POT.CUBE")
    write_bulk_cube(root / "vol" / "bulk_DENS.cube")
    write_face_cube(root / "vol" / "face_DENS.cube")
    write_face_cube(root / "vol" / "face_SPIN.cube", signed=True)
    write_face_cube(root / "vol" / "face_POT.cube", signed=True)
    proc, base, _ = H.start_server(root, tmp / "xdg", MACE_GUI_TEST_HOOKS="1", MACE_GUI_MAX_WORKERS="2")
    try:
        drv = H.browser(request.param, width=1280, height=800)
    except BaseException:
        proc.terminate()
        proc.wait(timeout=15)
        raise
    try:
        H.load(drv, base)
        drv.execute_script("window.__mace.settings.set('reduceMotion', 'off'); window.__mace.settings.set('volume.quality', 'low')")
        yield drv
    finally:
        drv.quit()
        proc.terminate()
        proc.wait(timeout=15)


@pytest.fixture
def drv(app, request):
    failed = request.session.testsfailed
    yield app
    if request.session.testsfailed > failed:
        H.save_failure(app, request.node.name)
    app.execute_script("""window.__mace.closeAll(); const s = window.__mace.settings;
      s.set('showCell', true); s.set('labels', 'off'); s.set('projection', 'orthographic'); s.set('showAxes', true);
      s.set('showBoundary', true); s.set('displayLo', [0, 0, 0]); s.set('displayHi', [1, 1, 1]);
      const b = document.getElementById('opt-bonds'); if (b && !b.checked) window.__mace.run('view.bonds');""")
    time.sleep(0.3)


def corpus(drv):
    if not (CORPUS / f"{SLAB}_DENS.CUBE").is_file():
        pytest.skip("test/ corpus not present (gitignored)")


def st(drv):
    return drv.execute_script("return window.__mace.volumeView.state()")


def open_vol(drv, path, orbital=False):
    kind = drv.execute_async_script("const d = arguments[1]; window.__mace.open(arguments[0]).then((x) => d(x && x.kind), (e) => d('error: ' + e.message))", path)
    assert kind == "volume", kind
    H.wait(drv, """const s = window.__mace.volumeView && window.__mace.volumeView.state();
      return !!(s && s.path === arguments[0] && s.surfaces > 0 && s.display && (!arguments[1] || (s.orbital && !s.orbital.busy)))""", 90, path, orbital)
    drv.execute_script("window.__mace.settings.set('showBoundary', false)")        # each site once: the counts below
    time.sleep(0.6)


def run(drv, cmd):
    drv.execute_script("window.__mace.run(arguments[0])", cmd)
    time.sleep(0.5)


def reason(drv, cmd):
    return drv.execute_script("return window.__mace.menuReason(arguments[0])", cmd)


def drawn_atoms(drv):
    return drv.execute_script(f"return {VV}.getModel() ? {VV}.getModel().selectedAtoms({{}}).length : 0")


def wait_orbital(drv, cond):
    H.wait(drv, "const s = window.__mace.volumeView.state(); return !!(s && s.orbital && !s.orbital.busy && s.surfaces > 0 && (" + cond + "))", 90)
    time.sleep(0.5)
    return st(drv)


ENABLED = ["view.cell", "view.bonds", "view.labelsCycle", "view.boundary", "view.axes", "view.alongA", "view.alongBStar", "view.range2",
           "view.rangeCycle", "view.rangeCustom", "view.labels.element", "view.reset", "view.projection", "view.rotateLeft"]


def test_every_option_is_enabled_for_a_periodic_cube(drv):
    corpus(drv)
    open_vol(drv, "vol/slab_DENS.CUBE")
    for cmd in ENABLED:
        assert reason(drv, cmd) == "", (cmd, reason(drv, cmd))
    for cmd in ["view.cell", "view.bonds", "view.labelsCycle", "view.boundary", "view.projection"]:
        H.wait(drv, "return document.querySelector(`#toolbar [data-cmd='${arguments[0]}']`).getAttribute('aria-disabled') === 'false'", 5, cmd)
    assert drv.execute_script("return document.getElementById('tb-range-label').textContent") == "1×1×1"


def test_a_molecule_has_no_cell_and_says_why(drv):
    corpus(drv)
    open_vol(drv, "vol/mol_DENS.CUBE")
    for cmd in ["view.range2", "view.rangeCycle", "view.rangeCustom", "view.boundary", "view.alongA", "view.alongCStar"]:
        assert reason(drv, cmd).startswith("This volume is of a molecule"), (cmd, reason(drv, cmd))
    for cmd in ["view.bonds", "view.labelsCycle", "view.axes", "view.reset"]:
        assert reason(drv, cmd) == "", (cmd, reason(drv, cmd))
    n = st(drv)["atoms"]
    assert st(drv)["display"]["atoms"] == n == drawn_atoms(drv)
    # bonds and labels work for a molecule; the potential cube of it is the same view
    run(drv, "view.bonds")
    assert st(drv)["display"]["bonds"] is False
    assert drv.execute_script(f"return {VV}.getModel().selectedAtoms({{}}).every((a) => !(a.style && a.style.stick))")
    run(drv, "view.bonds")
    assert drv.execute_script(f"return {VV}.getModel().selectedAtoms({{}}).some((a) => a.style && a.style.stick)")
    run(drv, "view.labels.both")
    assert drv.execute_script(f"return {VV}.labels.length") == n
    run(drv, "view.labels.off")
    assert drv.execute_script(f"return {VV}.labels.length") == 0
    open_vol(drv, "vol/mol_POT.CUBE")
    assert reason(drv, "view.bonds") == "" and reason(drv, "view.range2").startswith("This volume is of a molecule")
    assert not H.errors(drv), H.errors(drv)


def test_cube_cell_bonds_labels_axes_and_view(drv):
    corpus(drv)
    open_vol(drv, "vol/slab_DENS.CUBE")
    s0 = st(drv)
    n = s0["atoms"]
    assert s0["display"]["cell"] is True and s0["display"]["atoms"] == n == drawn_atoms(drv)
    # Show Unit Cell: the 12 edges, and the Inspector's Cell box is the same setting
    shapes = drv.execute_script(f"return {VV}.shapes.length")
    assert drv.execute_script("return document.getElementById('vo-cell').checked")
    run(drv, "view.cell")
    assert st(drv)["display"]["cell"] is False
    assert drv.execute_script(f"return {VV}.shapes.length") == shapes - 12
    assert not drv.execute_script("return document.getElementById('vo-cell').checked")
    drv.execute_script("document.getElementById('vo-cell').click()")
    time.sleep(0.4)
    assert st(drv)["display"]["cell"] is True and drv.execute_script("return window.__mace.settings.get('showCell')") is True
    assert drv.execute_script(f"return {VV}.shapes.length") == shapes
    # bonds
    assert drv.execute_script(f"return {VV}.getModel().selectedAtoms({{}}).some((a) => a.style && a.style.stick)")
    run(drv, "view.bonds")
    assert drv.execute_script(f"return {VV}.getModel().selectedAtoms({{}}).every((a) => !(a.style && a.style.stick))")
    run(drv, "view.bonds")
    # labels: the cell's own atoms
    run(drv, "view.labels.element")
    assert drv.execute_script(f"return {VV}.labels.length") == n
    run(drv, "view.labelsCycle")                       # element -> index
    assert st(drv)["display"]["labels"] == "index"
    run(drv, "view.labels.off")
    assert drv.execute_script(f"return {VV}.labels.length") == 0
    # axes: the triad on the volume view follows Show Axes and the camera
    tri = "document.getElementById('axis-triad-vol')"
    assert drv.execute_script(f"const t = {tri}; return !!t && t.style.display !== 'none' && t.querySelectorAll('.ax').length === 3")
    run(drv, "view.axes")
    time.sleep(0.2)
    assert drv.execute_script(f"return {tri}.style.display") == "none"
    run(drv, "view.axes")
    time.sleep(0.2)
    assert drv.execute_script(f"return {tri}.style.display") != "none"
    # View Along a turns the camera to a (and the triad with it); Reset View goes back
    q0 = drv.execute_script(f"return {VV}.getView().slice(4)")
    run(drv, "view.alongA")
    time.sleep(0.4)
    q1 = drv.execute_script(f"return {VV}.getView().slice(4)")
    assert not np.allclose(q0, q1)
    ax = drv.execute_script(f"return [...{tri}.querySelectorAll('.ax-a line')].map((l) => [+l.getAttribute('x2'), +l.getAttribute('y2')])")
    assert ax and np.hypot(*ax[0]) < 1.0, ax                           # a points at the viewer: no length on screen
    run(drv, "view.reset")
    assert np.allclose(drv.execute_script(f"return {VV}.getView().slice(4)"), q0, atol=1e-6)
    # a view to c*, rotate 90 degrees: the camera moves
    run(drv, "view.rotateLeft")
    assert not np.allclose(drv.execute_script(f"return {VV}.getView().slice(4)"), q0)
    assert not H.errors(drv), H.errors(drv)


def test_cube_supercell_replicates_the_surface_with_the_atoms(drv):
    corpus(drv)
    open_vol(drv, "vol/slab_DENS.CUBE")
    s1 = st(drv)
    n, v1 = s1["atoms"], s1["vertices"]
    assert s1["periodic"] and s1["surfaces"] == 1
    run(drv, "view.range2")
    s = st(drv)
    # the slab is periodic in a and b: 2x2 cells, the c axis is the vacuum
    assert s["display"]["atoms"] == 4 * n == drawn_atoms(drv), s["display"]
    assert s["display"]["images"] == 4 and s["surfaces"] == 4 and s["vertices"] == 4 * v1
    # the toolbar says what is drawn (2x2x1), the tooltip why, as the structure view does
    assert drv.execute_script("return document.getElementById('tb-range-label').textContent") == "2×2×1"
    assert "not periodic along c" in drv.execute_script("return document.getElementById('tb-range').dataset.tip")
    assert on_screen(drv)["bad"] == 0
    # the surface of every image is where its atoms are: one surface vertex lies within 1.2 A of 99% of the drawn atoms
    near = drv.execute_script(f"""const V = window.__mace.volumeView.vertices(), A = {VV}.getModel().selectedAtoms({{}});
      let hit = 0; for (const a of A) {{ let best = 1e9;
        for (let i = 0; i < V.length; i += 3) {{ const d = Math.hypot(V[i] - a.x, V[i+1] - a.y, V[i+2] - a.z); if (d < best) best = d; }}
        if (best < 1.2) hit++; }} return hit / A.length;""")
    assert near > 0.99, near
    # the cell box grows with the range (12 edges, once)
    run(drv, "view.range1")
    s = st(drv)
    assert s["display"]["atoms"] == n and s["surfaces"] == 1 and s["vertices"] == v1
    # Cycle Display Range (1 -> 2) and Boundary Atoms: more atoms on the faces, never fewer
    run(drv, "view.rangeCycle")
    assert st(drv)["display"]["images"] == 4
    off = st(drv)["display"]["atoms"]
    run(drv, "view.boundary")
    assert drv.execute_script("return window.__mace.settings.get('showBoundary')") is True or drv.execute_script(
        "return document.getElementById('opt-boundary').checked")
    assert st(drv)["display"]["atoms"] >= off
    assert not H.errors(drv), H.errors(drv)


def on_screen(drv):
    """The drawn atoms whose centers fall outside the volume view's canvas."""
    return drv.execute_script(f"""const v = {VV}, r = v.getCanvas().getBoundingClientRect();
      const A = v.getModel() ? v.getModel().selectedAtoms({{}}) : [];
      const P = A.map((a) => v.modelToScreen({{x: a.x, y: a.y, z: a.z}}));
      return {{n: P.length, bad: P.filter((p) => p.x < r.left || p.x > r.right || p.y < r.top || p.y > r.bottom).length}}""")


def per_axis(drv, cell):
    """The cells the drawn atoms and the isosurface vertices occupy along a, b and c (an orthogonal cell from 0)."""
    return drv.execute_script(f"""const cell = arguments[0], V = window.__mace.volumeView.vertices(), A = {VV}.getModel().selectedAtoms({{}});
      const cells = (pts) => [0, 1, 2].map((k) => new Set(pts.map((p) => Math.floor(p[k] / cell[k]))).size);
      const vs = []; for (let i = 0; i < V.length; i += 3) vs.push([V[i], V[i + 1], V[i + 2]]);
      return {{atoms: cells(A.map((a) => [a.x, a.y, a.z])), surfaces: cells(vs)}}""", cell)


def test_bulk_cube_range_repeats_all_three_axes_and_fits(drv):
    # 1.1.6: a new range kept the camera, so a 3x3x3 of a cube showed as the cells that fit (3x2, or 3x3x1)
    open_vol(drv, "vol/bulk_DENS.cube")
    s = st(drv)
    assert s["periodic"] and s["surfaces"] == 1 and s["display"]["atoms"] == 2
    cell = [x * 0.529177210903 for x in BULK_CELL]
    for along in (None, "view.alongA"):
        if along:
            run(drv, "view.range1")
            run(drv, along)
        for n in (2, 3):
            run(drv, f"view.range{n}")
            s = st(drv)
            assert s["display"]["images"] == n ** 3 and s["surfaces"] == n ** 3 and not s["display"]["capped"], s["display"]
            assert s["display"]["atoms"] == 2 * n ** 3 == drawn_atoms(drv)
            # the tall c is periodic: the label is the range, no note
            assert drv.execute_script("return document.getElementById('tb-range-label').textContent") == f"{n}×{n}×{n}"
            assert drv.execute_script("return document.getElementById('tb-range').dataset.tip") == "Display range"
            c = per_axis(drv, cell)
            assert c["atoms"] == [n, n, n] and c["surfaces"] == [n, n, n], c
            # every cell is on screen (the camera follows the range)
            o = on_screen(drv)
            assert o["n"] == 2 * n ** 3 and o["bad"] == 0, (along, n, o)
    assert not H.errors(drv), H.errors(drv)


def seams(drv):
    """The drawn surfaces' triangles merged by position, and their edges used by one triangle only that
    lie by an inner face of the display range (a seam between cells) or inside a cell (a hole), and the
    vertices two cells share drawn in two colors."""
    return drv.execute_script(f"""const v = {VV}, L = window.__mace.activeDoc().meta.lattice, D = window.__mace.volumeView.state().display;
      const R = D.range, cells = Array.isArray(R) ? R : R.hi.map((x, k) => x - R.lo[k]), o = D.box.origin;
      const inv = (M) => {{ const [a, b, c] = M, d = a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0]);
        return [[b[1] * c[2] - b[2] * c[1], -(a[1] * c[2] - a[2] * c[1]), a[1] * b[2] - a[2] * b[1]],
          [-(b[0] * c[2] - b[2] * c[0]), a[0] * c[2] - a[2] * c[0], -(a[0] * b[2] - a[2] * b[0])],
          [b[0] * c[1] - b[1] * c[0], -(a[0] * c[1] - a[1] * c[0]), a[0] * b[1] - a[1] * b[0]]].map((r) => r.map((x) => x / d)); }};
      const Li = inv(L), grid = new Map(), pts = [], edges = new Map();
      const near = [-1, 0, 1].flatMap((i) => [-1, 0, 1].flatMap((j) => [-1, 0, 1].map((k) => [i, j, k])));
      // a vertex two cells share has one color too (a potential's colors read across the faces)
      const cols = [];
      let colorMismatch = 0;
      const vid = (p, c) => {{
        const q = p.map((x) => Math.round(x * 1e3));
        for (const d of near) {{
          const h = grid.get(q.map((x, k) => x + d[k]).join(","));
          if (h !== undefined && Math.hypot(pts[h][0] - p[0], pts[h][1] - p[1], pts[h][2] - p[2]) < 1e-4) {{
            if (Math.max(...c.map((x, k) => Math.abs(x - cols[h][k]))) > 1e-3) colorMismatch++;
            return h;
          }}
        }}
        grid.set(q.join(","), pts.length); pts.push(p); cols.push(c); return pts.length - 1;
      }};
      let triangles = 0;
      for (const s of v.shapes) for (const g of (s && s.geo && s.geo.geometryGroups) || []) {{
        const P = g.vertexArray, F = g.faceArray, C = g.colorArray;
        for (let f = 0; f + 2 < g.faceidx; f += 3) {{
          triangles++;
          const t = [0, 1, 2].map((j) => {{ const i = 3 * F[f + j]; return vid([P[i], P[i + 1], P[i + 2]], C ? [C[i], C[i + 1], C[i + 2]] : [0, 0, 0]); }});
          for (const [a, b] of [[t[0], t[1]], [t[1], t[2]], [t[2], t[0]]]) {{ const k = a < b ? a + "_" + b : b + "_" + a; edges.set(k, (edges.get(k) || 0) + 1); }}
        }}
      }}
      const r = {{ triangles, seam: 0, hole: 0, outer: 0, cells, colorMismatch }};
      for (const [k, n] of edges) {{
        if (n !== 1) continue;
        const [a, b] = k.split("_").map(Number), mid = [0, 1, 2].map((j) => (pts[a][j] + pts[b][j]) / 2 - o[j]);
        const f = [0, 1, 2].map((j) => mid[0] * Li[0][j] + mid[1] * Li[1][j] + mid[2] * Li[2][j]);
        if (f.some((x, j) => Math.abs(x) < 0.1 || Math.abs(x - cells[j]) < 0.1)) r.outer++;
        else if (f.some((x) => Math.abs(x - Math.round(x)) < 0.1)) r.seam++;
        else r.hole++;
      }}
      return r;""")


def test_cube_surfaces_join_across_the_cells_of_a_supercell(drv):
    """1.1.6 (Marcus on HPCC): a cube's surfaces did not connect in a supercell, a gap at every cell
    boundary. A density and a spin whose lobes cross the cell's faces, at 2x2x2: every triangle edge by
    an inner face is shared (no seam), at each mesh quality and in flat shading, for both signs, and
    the density colored by a potential has one color where two cells meet."""
    import importlib.util
    def pick(id_, value):
        e = H.wait(drv, f"return document.getElementById('{id_}')", 10)
        drv.execute_script("arguments[0].value = arguments[1]; arguments[0].dispatchEvent(new Event('change', {bubbles: true}))", e, value)

    for path, levels in (("vol/face_DENS.cube", 1), ("vol/face_SPIN.cube", 2)):
        open_vol(drv, path)
        run(drv, "view.range2")
        H.wait(drv, "return window.__mace.volumeView.state().surfaces === arguments[0]", 30, 8 * levels)
        assert drv.execute_script("return window.__mace.activeDoc().meta.wrap") == [True, True, True]
        presets = ("draft", "standard", "high", "publication") if levels == 1 else ("standard",)
        for preset in presets:
            pick("vo-quality", preset)
            H.wait(drv, "const s = window.__mace.volumeView.state(); return s.mesh.quality === arguments[0] && !s.mesh.loading && s.surfaces === arguments[1]",
                   120, preset, 8 * levels)
            time.sleep(0.3)
            r = seams(drv)
            assert r["cells"] == [2, 2, 2] and r["triangles"] > 1000 and r["seam"] == 0 and r["hole"] == 0, (path, preset, r)
        if levels == 1 and importlib.util.find_spec("scipy"):
            # Color by Potential: the potential read across the faces (no pale seam where the mesh passes the grid)
            drv.execute_script("document.querySelector('#volume-host [data-act=mep]').click()")
            sel = H.wait(drv, "return document.querySelector('#cube-pick')", 10)
            drv.execute_script("arguments[0].value = 'vol/face_POT.cube'", sel)
            drv.execute_script("document.getElementById('cube-ok').click()")
            H.wait(drv, "const s = window.__mace.volumeView.state(); return s.color === 'vol/face_POT.cube' && s.vmin !== null && s.surfaces === 8", 90)
            time.sleep(0.3)
            r = seams(drv)
            assert r["seam"] == 0 and r["hole"] == 0 and r["colorMismatch"] == 0, (path, "potential colors", r)
        pick("vo-shading", "flat")
        H.wait(drv, "return window.__mace.volumeView.state().surfaces > 0", 10)
        time.sleep(0.3)
        r = seams(drv)
        assert r["seam"] == 0 and r["hole"] == 0, (path, "flat", r)
        for style in ("edges", "points", "wireframe", "solid"):
            pick("vo-style", style)
            H.wait(drv, "return window.__mace.volumeView.state().surfaces > 0", 10)
        pick("vo-shading", "smooth")
        run(drv, "view.range1")
        drv.execute_script("window.__mace.closeAll()")
        time.sleep(0.3)
    assert not H.errors(drv), H.errors(drv)


def test_orbital_cells_are_the_toolbars_range(drv):
    open_vol(drv, SI, orbital=True)
    s = st(drv)
    o = s["orbital"]
    # the file's default (2x2x2 for a 3D cell) is the document's display range: one control
    assert o["cells"] == [2, 2, 2] and drv.execute_script("return document.getElementById('tb-range-label').textContent") == "2×2×2"
    assert [drv.execute_script("return +document.querySelectorAll('#volume-host .orb-cell')[arguments[0]].value", k) for k in range(3)] == [2, 2, 2]
    # the atoms the server lists for the box are the ones the range draws (each site once)
    assert s["display"]["atoms"] == s["atoms"] == drawn_atoms(drv)
    assert reason(drv, "view.range3") == "" and reason(drv, "view.alongA") == "" and reason(drv, "view.boundary") == ""
    # the toolbar: 3x3x3 is a new grid of 3 cells, the field follows
    run(drv, "view.range3")
    s = wait_orbital(drv, "s.orbital.cells[0] === 3")
    assert s["orbital"]["cells"] == [3, 3, 3] and s["display"]["atoms"] == s["atoms"] == drawn_atoms(drv)
    assert [drv.execute_script("return +document.querySelectorAll('#volume-host .orb-cell')[arguments[0]].value", k) for k in range(3)] == [3, 3, 3]
    assert drv.execute_script("return document.getElementById('tb-range-label').textContent") == "3×3×3"
    # the field: it writes the range, the toolbar follows
    cell = drv.execute_script("return document.querySelector('#volume-host .orb-cell[data-axis=\"0\"]')")
    drv.execute_script("arguments[0].value = '1'; arguments[0].dispatchEvent(new Event('change', {bubbles: true}))", cell)
    s = wait_orbital(drv, "s.orbital.cells[0] === 1")
    assert s["orbital"]["cells"] == [1, 3, 3]
    assert drv.execute_script("return document.getElementById('tb-range-label').textContent") == "1×3×3"
    # more than the server draws: brought back to its limit, and said so
    drv.execute_script("window.__mace.display.setCells(window.__mace.activeDoc(), [6, 3, 3])")
    s = wait_orbital(drv, "s.orbital.cells[0] === 4")
    assert s["orbital"]["cells"] == [4, 3, 3]
    assert drv.execute_script("return document.getElementById('tb-range-label').textContent") == "4×3×3"
    # cell, bonds, labels and axes work on the orbital too
    sh = drv.execute_script(f"return {VV}.shapes.length")
    run(drv, "view.cell")
    assert drv.execute_script(f"return {VV}.shapes.length") == sh - 12
    run(drv, "view.cell")
    run(drv, "view.labels.element")
    assert drv.execute_script(f"return {VV}.labels.length") > 0
    run(drv, "view.labels.off")
    assert drv.execute_script("const t = document.getElementById('axis-triad-vol'); return !!t && t.style.display !== 'none'")
    assert not H.errors(drv), H.errors(drv)


def test_slab_orbital_range_keeps_the_vacuum_axis_at_one(drv):
    open_vol(drv, SLAB_ORB, orbital=True)
    s = st(drv)
    assert s["orbital"]["cells"] == [2, 2, 1] and s["display"]["atoms"] == s["atoms"] == drawn_atoms(drv)
    run(drv, "view.range3")
    s = wait_orbital(drv, "s.orbital.cells[0] === 3")
    assert s["orbital"]["cells"] == [3, 3, 1] and s["display"]["atoms"] == s["atoms"]
    assert drv.execute_script("return document.getElementById('tb-range-label').textContent") == "3×3×1"
    assert "not periodic along c" in drv.execute_script("return document.getElementById('tb-range').dataset.tip")
    assert not H.errors(drv), H.errors(drv)
