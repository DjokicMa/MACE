"""Coordination analysis and polyhedra (plan T11, B9).

The core functions run in the page (imported from js/geom/core.js) on the
demo CIFs: MgO, Si, hBN, LiF, NaCl. Shift+P toggles one polyhedron mesh per
center element; the Coordination tab lists the unique sites with their
Wyckoff letters; above 2000 drawn atoms polyhedra stay off and say why; the
worker builds the 4096-atom set within its budget."""
import os
import time

import pytest

import guiharness as H
import p4util as U

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")
from selenium.webdriver.common.keys import Keys  # noqa: E402

CORE = "const core = await import(document.querySelector('script[src*=\"/js/shell.js\"]').src.replace('shell.js', 'geom/core.js'));"


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    from ase.io import read, write
    tmp = tmp_path_factory.mktemp("p5coord")
    root = H.examples_copy(tmp)
    (root / "big").mkdir()
    write(str(root / "big" / "MgO_888.cif"), read(str(root / "structures" / "MgO.cif")).repeat((8, 8, 8)))
    a = U.start_app(tmp, root=root)
    try:
        yield a
    finally:
        U.stop_app(a)


@pytest.fixture
def fresh(app):
    U.reset(app[0])
    return app


def coord(drv, path, sites=None):
    """[{i, cn, nearest, neighbors, econ}] for the CIF's unique sites (or `sites`), and each center's hull facets."""
    U.open_doc(drv, path)
    return drv.execute_async_script("""const done = arguments[arguments.length - 1];
      (async () => {""" + CORE + """
        const s = window.__mace.doc().structure;
        const z = Uint8Array.from(s.symbols.map((x) => core.zOf(x))), frac = Float64Array.from(s.frac.flat());
        const radii = Float32Array.from(s.radii), metal = Uint8Array.from(s.metal.map(Number));
        const sites = arguments[0] || [...new Set(s.symmetry.equivalent_atoms)];
        const rows = core.coordination(s.cell, frac, z, radii, metal, 1.15, sites);
        const adj = core.bondedShells(s.cell, frac, radii, metal, 1.15);
        const hulls = sites.map((i) => core.convexHull(adj[i].map((x) => x.vec)).facets.length);
        const disp = window.__mace.viewport.scene().display;
        const poly = core.polyhedra(s.cell, frac, z, radii, metal, 1.15, disp.cart, disp.base);
        return { rows: rows.map((r) => ({ i: r.i, sym: s.symbols[r.i], cn: r.cn, nearest: r.nearest, econ: r.econ,
          neighbors: r.neighbors.map((x) => core.SYMBOLS[x.z] + 'x' + x.count) })), hulls, faces: poly.faces, meshes: poly.meshes.map((m) => core.SYMBOLS[m.z]) };
      })().then(done, (e) => done({ error: String(e) }));""", sites)


def test_core_cn_on_the_demo_structures(fresh):
    drv = fresh[0]
    r = coord(drv, "structures/MgO.cif")
    by = {x["sym"]: x for x in r["rows"]}
    assert by["Mg"]["cn"] == 6 and abs(by["Mg"]["nearest"] - 2.106) < 1e-3 and by["Mg"]["neighbors"] == ["Ox6"]
    assert by["O"]["cn"] == 6 and by["O"]["neighbors"] == ["Mgx6"]
    assert abs(by["Mg"]["econ"] - 6) < 1e-6
    assert r["meshes"] == ["Mg"] and all(f["faces"] == 8 for f in r["faces"])     # octahedra around Mg only
    si = coord(drv, "structures/Si.cif")
    assert all(x["cn"] == 4 and abs(x["nearest"] - 2.35) < 0.01 for x in si["rows"]), si
    assert all(h == 4 for h in si["hulls"])                                        # the Si tetrahedron
    assert si["meshes"] == []                                                      # Si-Si: no cation polyhedra
    bn = coord(drv, "structures/hBN.cif")
    assert {x["sym"]: x["cn"] for x in bn["rows"]}["B"] == 3
    for path, cation in (("structures/LiF.cif", "Li"), ("structures/NaCl.cif", "Na")):
        r = coord(drv, path)
        assert r["meshes"] == [cation] and all(f["cn"] == 6 and f["faces"] == 8 for f in r["faces"]), (path, r)
    U.no_errors(drv)


def test_shift_p_and_the_coordination_tab(fresh):
    drv = fresh[0]
    U.open_doc(drv, "structures/MgO.cif")
    assert drv.execute_script("return window.__mace.viewport.polyhedra().meshes") == 0
    drv.execute_script("document.body.focus()")
    U.keys(drv, Keys.SHIFT, "P")
    H.wait(drv, "const p = window.__mace.viewport.polyhedra(); return p.on && p.meshes === 1", 20)
    p = drv.execute_script("return window.__mace.viewport.polyhedra()")
    assert p["shapes"] == 2 and p["count"] >= 4                    # one mesh and one edge set for Mg
    U.shot(drv, "polyhedra_mgo")
    U.keys(drv, Keys.SHIFT, "P")
    H.wait(drv, "const p = window.__mace.viewport.polyhedra(); return !p.on && p.shapes === 0", 20)
    # the Coordination tab: one row per unique site, with its Wyckoff letter
    drv.execute_script("window.__mace.run('tool.coordination')")
    H.wait(drv, "return document.querySelectorAll('.bottom-pane.coordination .coord-table tbody tr').length === 2", 20)
    rows = drv.execute_script("return [...document.querySelectorAll('.bottom-pane.coordination .coord-table tbody tr')].map((r) => [...r.cells].map((c) => c.textContent))")
    assert rows[0][:3] == ["Mg1", "4a", "6"] and rows[1][:3] == ["O5", "4b", "6"], rows
    assert rows[0][6] == "O×6" and rows[0][7].startswith("6 O 2.106")
    U.shot(drv, "coordination_tab")
    # the inspector line for one selected atom
    drv.execute_script("window.__mace.editor.select([0])")
    H.wait(drv, "const e = document.querySelector('#inspector .ins-cn'); return e && e.textContent === 'CN 6 (O×6, 2.106 Å)'", 10)
    # Export Table writes the shown rows
    assert drv.execute_script("return window.__mace.menuReason('analysis.exportCsv')") == ""
    U.no_errors(drv)


def test_lod_keeps_polyhedra_off_above_2000_atoms(fresh):
    drv = fresh[0]
    drv.execute_script("window.__mace.settings.set('showPolyhedra', true)")
    try:
        U.open_doc(drv, "big/MgO_888.cif")
        H.wait(drv, "const p = window.__mace.viewport.polyhedra(); return p && p.skipped", 30)
        assert "Polyhedra are off above 2,000 drawn atoms" in drv.execute_script("return window.__mace.viewport.banner()")
        assert drv.execute_script("return window.__mace.viewport.polyhedra().shapes") == 0
        # the worker's polyhedra for all 4096 atoms (the bench row: ≤100 ms in the worker)
        ms = drv.execute_async_script("""const done = arguments[arguments.length - 1];
          (async () => {""" + CORE + """
            const W = await import(document.querySelector('script[src*="/js/shell.js"]').src.replace('shell.js', 'geom/client.js'));
            const s = window.__mace.doc().structure, d = window.__mace.viewport.scene().display;
            const args = () => ({ cell: s.cell, frac: Float64Array.from(s.frac.flat()), z: Uint8Array.from(s.symbols.map((x) => core.zOf(x))),
              radii: Float32Array.from(s.radii), metal: Uint8Array.from(s.metal.map(Number)), tol: 1.15, dispCart: d.cart.slice(), dispBase: Uint32Array.from(d.base) });
            await W.job("polyhedra", args());
            const t = [];
            for (let k = 0; k < 3; k++) { const r = await W.job("polyhedra", args()); t.push(r.ms !== undefined ? r.ms : -1); if (!r.count) throw new Error('no polyhedra'); }
            return t;
          })().then(done, (e) => done(String(e)));""")
        assert isinstance(ms, list), ms
        print("polyhedra_4096_ms (worker)", ms)
        # the 100 ms budget is a benchmark row (MACE_GUI_BENCH=1, an idle machine);
        # on a shared or loaded one only a gross slowdown fails this test
        assert min(ms) <= (100 if os.environ.get("MACE_GUI_BENCH") == "1" else 1000), ms
    finally:
        drv.execute_script("window.__mace.settings.set('showPolyhedra', false)")
