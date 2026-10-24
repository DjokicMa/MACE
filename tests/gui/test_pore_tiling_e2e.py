"""Pore surfaces over a supercell in a real browser (1.1.6).

Structure Analysis › Porosity drew sodalite's beta cage in one cell only, even
with the structure shown 3x3x3 (Marcus: a single cell does not show the
framework bonded). The example suite's sodalite (examples/suite/1-porous/SOD,
the cubic cell: a beta cage on its corner and one at its center) is drawn
3x3x3 here: the pocket overlay has one copy per cell at the cell's lattice
vector, built once (a paint rebuilds nothing), the pocket colors shared by
every copy, in every style and at every mesh quality, and nothing logs an
error. The pore surfaces are made at their own default, Preview (Settings ›
Viewport › Pore surface mesh quality), whatever the other surfaces' Mesh
quality says; a quality picked for the overlay is kept for its next result,
and Export Image's Publication surfaces come and go. Screenshots (1400x900)
and the draw times go to MACE_GUI_SHOTS."""
import json
import os
import shutil
from pathlib import Path

import pytest

import guiharness as H
import p4util as U

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("scipy")

SOD_CIF = Path(__file__).resolve().parents[2] / "examples" / "suite" / "1-porous" / "SOD" / "structure" / "SOD.cif"


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("poretile_e2e")
    root = H.examples_copy(tmp)
    (root / "structures").mkdir(exist_ok=True)
    shutil.copy(SOD_CIF, root / "structures" / "SOD_suite.cif")
    a = U.start_app(tmp, width=1400, height=900, root=root)
    try:
        yield a
    finally:
        U.stop_app(a)


def porosity(drv, path, probe="N2", spacing="0.25"):
    U.open_doc(drv, path)
    drv.execute_script("window.__mace.run('analysis.structure.porosity')")
    H.wait(drv, "const d = document.getElementById('analysis-panel'); return d && d.open && !!d.querySelector('.an-form')", 15)
    drv.execute_script("document.querySelector('#analysis-panel [data-p=probe]').value = arguments[0];"
                       "document.querySelector('#analysis-panel [data-p=spacing]').value = arguments[1];"
                       "window.__prev = window.__maceAnalysis.results().porosity || null;"
                       "document.querySelector('#analysis-panel .an-form button[type=submit]').click()", probe, spacing)
    H.wait(drv, "const r = window.__maceAnalysis.results().porosity; return !!r && r !== window.__prev", 120)
    drv.execute_script("const d = document.getElementById('analysis-panel'); if (d && d.open) d.querySelector('[data-close]').click()")


# the pore overlay's shapes: their copies (the built mesh and its clones), where each is, and whether they share it
COPIES_JS = """
const it = window.__maceAnalysis.overlay.find('pores'), v = window.MACE.state.viewers['viewer-main'];
v.render();
const s = window.__mace.doc().structure, c = s.cell.length === 9 ? [s.cell.slice(0, 3), s.cell.slice(3, 6), s.cell.slice(6)] : s.cell;
return { cell: c, cells: it.tiles.cells, shifts: it.tiles.shifts, shapes: it.shapes.map((sh) => {
  const k = sh.renderedShapeObj ? sh.renderedShapeObj.children : [];
  return { pass: sh.macePass || '', n: k.length, at: k.map((m) => [m.matrix.elements[12], m.matrix.elements[13], m.matrix.elements[14]]),
    shared: k.every((m) => m.geometry === k[0].geometry && m.material === k[0].material), color: sh.color === undefined ? null : String(sh.color) };
}) };
"""

# a redraw (the structure layer drew again: every shape made and built anew, then one paint), and paints of the built
# scene (what a turn of the view costs): ms, each until the frame is drawn (a pixel read waits for the GL); and that
# those paints rebuild nothing
TIME_JS = """
const v = window.MACE.state.viewers['viewer-main'], $3 = window.$3Dmol, O = window.__maceAnalysis.overlay;
const gl = v.renderer.getContext(), px = new Uint8Array(4), sync = () => gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
sync();
const t0 = performance.now();
O.redraw();
sync();
const t1 = performance.now();
const geos = new Set(O.find('pores').shapes.map((s) => s.geo)), fin = $3.GLShape.finalizeGeo;
let rebuilt = 0;
$3.GLShape.finalizeGeo = function (g) { if (geos.has(g)) rebuilt++; return fin.call(this, g); };
const t2 = performance.now();
try { for (let i = 0; i < 5; i++) { v.render(); sync(); } } finally { $3.GLShape.finalizeGeo = fin; }
const t3 = performance.now();
const m = O.find('pores').data.meshes;
return { redraw: t1 - t0, paint: (t3 - t2) / 5, rebuilt, triangles: m.quality ? m.quality.triangles : null, cells: O.find('pores').tiles.shifts.length };
"""

def check_copies(drv, n):
    r = drv.execute_script(COPIES_JS)
    C = r["cell"]
    want = [[sum(i[k] * C[k][j] for k in range(3)) for j in range(3)] for i in r["cells"]]
    assert len(r["cells"]) == n and sorted(map(tuple, r["cells"])) == sorted({tuple(c) for c in r["cells"]})
    for a, b in zip(r["shifts"], want):
        assert max(abs(x - y) for x, y in zip(a, b)) < 1e-6, (a, b)
    assert r["shapes"], r
    for s in r["shapes"]:
        # one copy per displayed cell, at its lattice vector, all drawing the one built mesh
        assert s["n"] == n and s["shared"], s
        for a, b in zip(s["at"], want):
            assert max(abs(x - y) for x, y in zip(a, b)) < 1e-4, (s["pass"], a, b)
    return r


MADE = "return window.__maceAnalysis.overlay.find('pores').data.meshes.quality.quality"


def test_sodalite_pockets_over_3x3x3(app):
    drv = app[0]
    # the pore surfaces' own mesh quality (Preview unless chosen), whatever the other surfaces' setting says
    drv.execute_script("window.__mace.settings.set('mesh.quality', 'high')")
    default = drv.execute_script("return window.__mace.settings.get('pores.quality')")
    assert default == "preview"
    porosity(drv, "structures/SOD_suite.cif")
    H.wait(drv, "return window.__maceAnalysis.overlay.count('pores') >= 1", 15)
    assert drv.execute_script(MADE) == default
    assert drv.execute_script("return window.__maceAnalysis.overlay.list()[0].opts.quality") == default
    drv.execute_script("window.__mace.settings.set('mesh.quality', 'standard')")
    rec = drv.execute_script("const r = window.__maceAnalysis.results().porosity; return {f: r.features.length, "
                             "n: r.render.meshes.inaccessible.n_pockets, acc: !!r.render.meshes.accessible}")
    assert rec == {"f": 2, "n": 2, "acc": False}, rec           # the two beta cages, closed to N2
    check_copies(drv, 1)
    times = {f"{default}, 1 cell": drv.execute_script(TIME_JS)}
    U.shot(drv, "pore_tiling_sod_1x1x1")
    # 3x3x3: the cages in each of the 27 cells
    drv.execute_script("window.__mace.display.setCells(window.__mace.activeDoc(), 3)")
    H.wait(drv, "const it = window.__maceAnalysis.overlay.find('pores'); return !!it && it.tiles.shifts.length === 27", 15)
    H.wait(drv, "return window.__mace.viewport.displayed() > 36 * 27", 15)
    check_copies(drv, 27)
    key = drv.execute_script("return window.__maceAnalysis.overlay.find('pores').key")
    assert drv.execute_script("return window.__maceAnalysis.overlay.list()[0].cells") == 27
    times[default] = drv.execute_script(TIME_JS)
    U.shot(drv, "pore_tiling_sod_3x3x3")
    # Color pockets individually: the copies share the colored mesh, so a pocket has its color in every cell
    drv.execute_script("window.__maceAnalysis.overlay.override(arguments[0], {colorPockets: true})", key)
    r = check_copies(drv, 27)
    assert all(s["color"] is None for s in r["shapes"]), r["shapes"]
    U.shot(drv, "pore_tiling_sod_3x3x3_pocket_colors")
    drv.execute_script("window.__maceAnalysis.overlay.override(arguments[0], {colorPockets: null})", key)
    # every mesh style
    for style in ({"style": "wireframe"}, {"style": "edges"}, {"style": "points"}, {"shading": "flat"}, {"sides": "double"}):
        drv.execute_script("window.__maceAnalysis.overlay.override(arguments[0], arguments[1])", key, style)
        check_copies(drv, 27)
        drv.execute_script("window.__maceAnalysis.overlay.override(arguments[0], arguments[1])", key, {k: None for k in style})
    # every preset (the Inspector's choice for this overlay), Publication included: the server's mesh, tiled the same way
    for q in ("draft", "standard", "publication"):
        drv.execute_script("window.__maceAnalysis.overlay.override(arguments[0], {quality: arguments[1]})", key, q)
        drv.execute_async_script("const done = arguments[2]; window.__maceAnalysis.overlay.meshed(arguments[0]).then(() => done(true))", key, q)
        H.wait(drv, MADE + " === arguments[0]", 120, q)
        check_copies(drv, 27)
        times[q] = drv.execute_script(TIME_JS)
    U.shot(drv, "pore_tiling_sod_3x3x3_publication")
    for q, t in times.items():
        assert t["rebuilt"] == 0, (q, t)                           # a paint draws the built meshes, no rebuild
    target = os.environ.get("MACE_GUI_SHOTS")
    if target:
        (Path(target) / "pore_tiling_times.json").write_text(json.dumps(times, indent=1))
    # the overlay's choice is remembered: a new result of this structure keeps it
    porosity(drv, "structures/SOD_suite.cif")
    H.wait(drv, "return window.__maceAnalysis.overlay.list()[0].opts.quality === 'publication'", 15)
    H.wait(drv, MADE + " === 'publication'", 120)
    # back to the default; Export Image's Publication surfaces for the picture only, then the view's own again
    drv.execute_script("window.__maceAnalysis.overlay.override(arguments[0], {quality: null})", key)
    H.wait(drv, MADE + " === arguments[0]", 60, default)
    during = drv.execute_async_script("""const done = arguments[0], O = window.__maceAnalysis.overlay;
        O.withQuality('publication', async () => O.find('pores').data.meshes.quality.quality).then(done)""")
    assert during == "publication"
    H.wait(drv, MADE + " === arguments[0]", 60, default)
    check_copies(drv, 27)
    drv.execute_script("window.__mace.display.setCells(window.__mace.activeDoc(), 1)")
    H.wait(drv, "return window.__maceAnalysis.overlay.find('pores').tiles.shifts.length === 1", 15)
    assert not H.errors(drv), H.errors(drv)
