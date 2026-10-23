"""Analysis › Structure Analysis in a real browser: the menu commands open
the panel lazily on a structure, Porosity draws the pores in the 3D view
(sodalite's cages are closed to helium: inaccessible pockets only) and keeps
them through a redraw, results and pores stay with the structure they were
computed for (switching documents, also while a run is going), the powder
pattern, topology and morphology tabs answer, and nothing logs an error."""
import time

import pytest

import guiharness as H
import p4util as U

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")
pytest.importorskip("scipy")

IDS = ["analysis.structure.porosity", "analysis.structure.pxrd", "analysis.structure.fit", "analysis.structure.topology",
       "analysis.structure.polyhedra", "analysis.structure.bonds", "analysis.structure.morphology",
       "analysis.structure.planes"]


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("analysis_e2e")
    root = H.examples_copy(tmp)
    from ase.io import write
    from ase.spacegroup import crystal
    sod = crystal(["Si", "O"], basis=[(0.25, 0.5, 0.0), (0.1467, 0.5, 0.8533)], spacegroup=229, cellpar=[8.965] * 3 + [90] * 3)
    write(str(root / "structures" / "SOD.cif"), sod)
    lta = crystal(["O", "O", "O", "Si"], basis=[(0, 0.2122, 0.5), (0.1103, 0.1103, 0.3384), (0, 0.2967, 0.2967), (0, 0.1823, 0.3684)],
                  spacegroup=221, cellpar=[11.919] * 3 + [90] * 3)
    write(str(root / "structures" / "LTA.cif"), lta)
    a = U.start_app(tmp, root=root)
    try:
        yield a
    finally:
        U.stop_app(a)


def wait_panel(drv):
    H.wait(drv, "const d = document.getElementById('analysis-panel'); return d && d.open && !!d.querySelector('.an-form')", 15)


def click_run(drv):
    drv.execute_script("document.querySelector('#analysis-panel .an-form button[type=submit]').click()")


def wait_out(drv, timeout=120):
    H.wait(drv, "const o = document.querySelector('#analysis-panel .an-out'); return o && o.children.length > 0", timeout)


def test_commands_need_a_structure_and_open_lazily(app):
    drv = app[0]
    U.reset(drv)
    for cid in IDS:
        assert drv.execute_script("return window.__mace.menuReason(arguments[0])", cid), cid
    names = drv.execute_script("return performance.getEntriesByType('resource').map((r) => r.name)")
    assert not any("analysis_panel.js" in n for n in names), "the panel loaded at boot"
    U.open_doc(drv, "structures/SOD.cif")
    for cid in IDS:
        assert drv.execute_script("return window.__mace.menuReason(arguments[0])", cid) == "", cid


def test_porosity_draws_inaccessible_pockets(app):
    drv = app[0]
    U.open_doc(drv, "structures/SOD.cif")
    drv.execute_script("window.__mace.run('analysis.structure.porosity')")
    wait_panel(drv)
    drv.execute_script("const s = document.querySelector('#analysis-panel [data-p=probe]'); s.value = 'He';"
                       "document.querySelector('#analysis-panel [data-p=spacing]').value = '0.25'")
    click_run(drv)
    wait_out(drv)
    text = drv.execute_script("return document.querySelector('#analysis-panel .an-out').textContent")
    assert "2 pocket(s)" in text and "0 channel system(s)" in text
    assert drv.execute_script("return document.querySelector('#analysis-panel [data-o=show]').textContent").startswith("Redraw")
    count = lambda k: drv.execute_script("return window.__maceAnalysis.overlay.count(arguments[0])", k)
    assert count("pores") == 1                     # the inaccessible surface only
    # a redraw of the structure (a display change) keeps the overlay
    drv.execute_script("window.__mace.run('view.cell'); window.__mace.run('view.cell')")
    time.sleep(0.5)
    assert count("pores") == 1
    U.shot(drv, "analysis_pores")
    assert not H.errors(drv)


def test_other_tabs_answer(app):
    drv = app[0]
    U.open_doc(drv, "structures/Si.cif")
    drv.execute_script("window.__mace.run('analysis.structure.pxrd')")
    wait_panel(drv)
    click_run(drv)
    wait_out(drv)
    assert drv.execute_script("return !!document.querySelector('#analysis-panel svg.an-chart')")
    assert "(1 1 1)" in drv.execute_script("return document.querySelector('#analysis-panel .an-out table').textContent")
    drv.execute_script("document.querySelector('#analysis-panel .an-tab[data-tab=topology]').click()")
    click_run(drv)
    wait_out(drv)
    assert "dia" in drv.execute_script("return document.querySelector('#analysis-panel .an-out').textContent")
    cite = drv.execute_script("const e = document.querySelector('#analysis-panel [data-o=citation]');"
                              "return e && [e.textContent, [...e.querySelectorAll('a')].map((a) => a.href)]")
    assert cite and "O'Keeffe" in cite[0] and "doi:10.1021/ar800124u" in cite[0]     # RCSR is credited
    assert "https://doi.org/10.1021/ar800124u" in cite[1]
    drv.execute_script("document.querySelector('#analysis-panel [data-o=net]').click()")
    time.sleep(0.3)
    assert drv.execute_script("return window.__maceAnalysis.overlay.count('net')") == 8 + 16    # 8 vertices, 16 edges
    drv.execute_script("document.querySelector('#analysis-panel .an-tab[data-tab=morphology]').click()")
    click_run(drv)
    wait_out(drv)
    assert "(1 1 1)" in drv.execute_script("return document.querySelector('#analysis-panel .an-out').textContent")
    time.sleep(0.3)
    assert drv.execute_script("return window.__maceAnalysis.overlay.count('mesh')") == 1
    U.shot(drv, "analysis_morphology")
    assert not H.errors(drv)


def doc_index(drv, path):
    return drv.execute_script("return window.__mace.docs().findIndex((d) => d.path && d.path.endsWith(arguments[0]))", path)


def test_pores_stay_with_their_structure(app):
    drv = app[0]
    U.open_doc(drv, "structures/SOD.cif")
    drv.execute_script("window.__mace.run('analysis.structure.porosity')")
    wait_panel(drv)
    drv.execute_script("document.querySelector('#analysis-panel [data-p=probe]').value = 'N2';"
                       "document.querySelector('#analysis-panel [data-p=spacing]').value = '0.25'")
    click_run(drv)
    H.wait(drv, "const r = window.__maceAnalysis.results().porosity; return !!r && r.probe.name === 'N2' && r.grid.spacing > 0.2", 120)
    method = drv.execute_script("const m = document.querySelector('#analysis-panel [data-k=method]'); return m ? m.textContent : ''")
    assert "SOD.cif" in method and "MACE native engine" in method and "probe N2, r = 1.820" in method
    count = lambda: drv.execute_script("return window.__maceAnalysis.overlay.count('pores')")
    assert count() == 1
    # another structure: none of sodalite's numbers or pores on it
    U.open_doc(drv, "structures/LTA.cif")
    time.sleep(0.5)
    assert count() == 0
    assert drv.execute_script("return document.querySelector('#analysis-panel [data-k=doc]').textContent") == "Structure: LTA.cif"
    assert drv.execute_script("return document.querySelector('#analysis-panel .an-out').children.length") == 0
    # a run on LTA that lands while sodalite is shown: kept for LTA, drawn only there
    lta = drv.execute_script("return window.__mace.doc().id")
    click_run(drv)
    time.sleep(0.2)
    drv.execute_script("window.__mace.activate(arguments[0])", doc_index(drv, "structures/SOD.cif"))
    H.wait(drv, "const r = window.__maceAnalysis.resultsFor(arguments[0]).porosity; return !!r", 120, lta)
    time.sleep(0.5)
    # each document keeps its own pores: LTA's wait for LTA, sodalite's own are drawn here
    assert drv.execute_script("return window.__maceAnalysis.overlay.docOf('pores')") == lta and count() == 1
    assert drv.execute_script("return window.__maceAnalysis.overlay.find('pores').doc") == \
        drv.execute_script("return window.__mace.doc().id")
    text = drv.execute_script("return document.querySelector('#analysis-panel .an-out').textContent")
    assert "SOD.cif" in text and "2 pocket(s)" in text and "LTA.cif" not in text
    drv.execute_script("window.__mace.activate(arguments[0])", doc_index(drv, "structures/LTA.cif"))
    time.sleep(0.8)
    text = drv.execute_script("return document.querySelector('#analysis-panel .an-out').textContent")
    assert "LTA.cif" in text and "1 channel system(s), dimensionality 3; 1 pocket(s)" in text
    assert count() == 2                    # the alpha cages (accessible) and the beta cages (closed)
    U.shot(drv, "analysis_pores_lta")
    assert not H.errors(drv)


def pixels(url):
    """(green-dominant share, the image) of a PNG data URL: the bluish green of the channels."""
    import base64
    import io
    from PIL import Image
    im = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")
    px = list(im.getdata())
    return sum(1 for r, g, b in px if g > r + 40 and g > b) / len(px), im


def export_png(drv):
    drv.execute_script("window.__mace.run('file.exportImage')")
    d = H.wait(drv, "const d = document.getElementById('export-image'); return d && d.open && d", 10)
    drv.execute_script("arguments[0].querySelector('[data-k=triad]').checked = false;"
                       "arguments[0].dispatchEvent(new Event('change', {bubbles: true}))", d)
    return drv.execute_async_script("""const done = arguments[0];
        window.__maceExportHook = (url) => done(url);
        document.querySelector('#export-image [data-v="export"]').click();""")


def test_pore_meshes_and_their_controls(app):
    """LTA's alpha cages (one channel system, cut at the cell faces) and beta cage (a closed
    pocket) as meshes: translucent without writing depth, the controls under the results change
    them and are kept in Settings (gui.json, checked by the server), Settings changes them back,
    the PNG export draws them, the SVG note says they are left out, and nothing logs an error."""
    drv, base, xdg = app[0], app[1], app[3]
    U.open_doc(drv, "structures/LTA.cif")
    drv.execute_script("window.__mace.run('analysis.structure.porosity')")
    wait_panel(drv)
    drv.execute_script("document.querySelector('#analysis-panel [data-p=probe]').value = 'N2';"
                       "document.querySelector('#analysis-panel [data-p=spacing]').value = '0.25'")
    drv.execute_script("window.__prev = window.__maceAnalysis.results().porosity || null")
    click_run(drv)
    H.wait(drv, "const r = window.__maceAnalysis.results().porosity; return !!r && r !== window.__prev && r.grid.spacing > 0.2", 120)
    H.wait(drv, "return window.__maceAnalysis.overlay.count('pores') === 2", 30)
    # the meshes travel instead of the two fields; one closed pocket
    got = drv.execute_script("const m = window.__maceAnalysis.results().porosity.render;"
                             "return [!!m.meshes.accessible, m.meshes.inaccessible.n_pockets, 'accessible' in m, 'inaccessible' in m]")
    assert got == [True, 1, False, False]
    SHAPES = """const v = window.MACE.state.viewers['viewer-main'];
        return v.shapes.filter((s) => s && s.opacity < 1).map((s) => [s.opacity, !!s.wireframe,
          (s.renderedShapeObj ? s.renderedShapeObj.children : []).every((c) => !c.material || c.material.depthWrite === false)])"""
    shapes = drv.execute_script(SHAPES)
    # the far walls of both surfaces and their faint near walls, none writing depth
    assert len(shapes) == 4 and all(s[2] for s in shapes), shapes
    assert sorted(round(s[0], 3) for s in shapes) == [0.15, 0.15, 0.5, 0.5]
    with_pores, _ = pixels(export_png(drv))
    q = lambda o: f"#analysis-panel [data-o={o}]"
    ctl = lambda o, js: drv.execute_script(f"const e = document.querySelector('{q(o)}'); {js}")
    # Inaccessible pockets off: the channels only
    ctl("inaccessible", "e.checked = false; e.dispatchEvent(new Event('change', {bubbles: true}))")
    assert drv.execute_script("return window.__maceAnalysis.overlay.count('pores')") == 1
    ctl("inaccessible", "e.checked = true; e.dispatchEvent(new Event('change', {bubbles: true}))")
    ctl("colorPockets", "e.checked = true; e.dispatchEvent(new Event('change', {bubbles: true}))")
    ctl("opacity", "e.value = '0.3'; e.dispatchEvent(new Event('input', {bubbles: true}))")
    ctl("style", "e.value = 'wireframe'; e.dispatchEvent(new Event('change', {bubbles: true}))")
    shapes = drv.execute_script(SHAPES)
    assert len(shapes) == 2 and all(s[1] and s[0] == pytest.approx(0.3) for s in shapes), shapes
    get = lambda k: drv.execute_script("return window.__mace.settings.get(arguments[0])", k)
    assert [get(k) for k in ("pores.inaccessible", "pores.colorPockets", "pores.opacity", "pores.style")] == [True, True, 0.3, "wireframe"]
    drv.execute_async_script("const d = arguments[0]; window.__mace.settings.flush().then(d)")
    import json
    stored = json.loads((xdg / "mace" / "gui.json").read_text())["settings"]
    assert {k: stored.get(k) for k in ("pores.colorPockets", "pores.opacity", "pores.style")} == \
        {"pores.colorPockets": True, "pores.opacity": 0.3, "pores.style": "wireframe"}
    # the server checks them as the browser makes them, and refuses others
    st, _, body = U.req(base, "api/settings", data={"settings": {**stored, "pores.opacity": 2, "pores.style": "dots", "pores.accessible": "yes"}})
    assert st == 200 and sorted(body["rejected"]) == ["pores.accessible", "pores.opacity", "pores.style"]
    defaults = drv.execute_script("const d = window.__mace.settings.defaults(); return Object.fromEntries(Object.entries(d).filter(([k]) => k.startsWith('pores.')))")
    from mace.gui import settings as S
    # 1.1.6: shading, walls and the near walls while dragging joined the five
    assert len(defaults) == 8 and all(S.check(k, v) is None for k, v in defaults.items()), defaults
    # Settings › Viewport changes the surfaces and the controls back
    drv.execute_script("window.__mace.settings.set('pores.style', 'solid'); window.__mace.settings.set('pores.opacity', 0.5);"
                       "window.__mace.settings.set('pores.colorPockets', false)")
    time.sleep(0.3)
    assert len(drv.execute_script(SHAPES)) == 4
    assert drv.execute_script(f"return [document.querySelector('{q('style')}').value, document.querySelector('{q('opacity')}').value,"
                              f" document.querySelector('{q('colorPockets')}').checked]") == ["solid", "0.5", False]
    # the PNG export draws the surfaces (the channels' green), and none without them
    ctl("accessible", "e.checked = false; e.dispatchEvent(new Event('change', {bubbles: true}))")
    without, _ = pixels(export_png(drv))
    ctl("accessible", "e.checked = true; e.dispatchEvent(new Event('change', {bubbles: true}))")
    assert with_pores > 0.01 and without < 0.002, (with_pores, without)
    # SVG has no meshes: its note says so
    drv.execute_script("window.__mace.run('file.exportImage')")
    d = H.wait(drv, "const d = document.getElementById('export-image'); return d && d.open && d", 10)
    drv.execute_script("const s = arguments[0].querySelector('[data-k=format]'); s.value = 'svg'; s.dispatchEvent(new Event('change', {bubbles: true}))", d)
    assert "export PNG to keep them" in drv.execute_script("return arguments[0].querySelector('.svg-note').textContent", d)
    drv.execute_script("arguments[0].close()", d)
    for k in defaults:
        drv.execute_script("window.__mace.settings.reset(arguments[0])", k)
    U.shot(drv, "analysis_pore_meshes")
    assert not H.errors(drv), H.errors(drv)


TASK = """return [...window.__mace.tasks().values()].filter((t) => t.title === arguments[0])
    .sort((a, b) => (b.t0 || 0) - (a.t0 || 0)).map((t) => [t.id, t.state])[0] || null"""


def test_runs_go_on_in_the_background(app):
    """A run goes on until it is canceled: another tab of the panel, another document and back,
    the Tasks tab lists it with its progress and a Cancel button, and the result lands on its own
    document; Cancel (in the Tasks tab) stops it."""
    drv = app[0]
    U.open_doc(drv, "structures/LTA.cif")
    U.open_doc(drv, "structures/SOD.cif")
    sod = drv.execute_script("return window.__mace.doc().id")
    drv.execute_script("window.__mace.run('analysis.structure.porosity')")
    wait_panel(drv)
    # a fine grid: several seconds
    drv.execute_script("document.querySelector('#analysis-panel [data-p=probe]').value = 'CO2';"
                       "document.querySelector('#analysis-panel [data-p=spacing]').value = '0.08'")
    click_run(drv)
    H.wait(drv, "const t = [...window.__mace.tasks().values()].find((t) => t.title === 'Porosity: SOD' && t.state === 'running'); return !!t", 20)
    # the Tasks tab: the run, its progress and a Cancel button
    drv.execute_script("document.querySelector('.bottom-tab[data-bottom=tasks]').click()")
    H.wait(drv, "return [...document.querySelectorAll('.bottom-pane.tasks .task-row')].some((r) => r.textContent.includes('Porosity: SOD')"
                " && r.querySelector('[data-act=cancel]') && r.querySelector('[role=progressbar]'))", 10)
    # another tab of the panel and back: the status line and its Cancel are there again
    drv.execute_script("document.querySelector('#analysis-panel .an-tab[data-tab=pxrd]').click()")
    drv.execute_script("document.querySelector('#analysis-panel .an-tab[data-tab=porosity]').click()")
    st = drv.execute_script("const s = document.querySelector('#analysis-panel [data-k=status]'); return [s.textContent, !!s.querySelector('.an-cancel')]")
    assert st[1] and st[0].strip(), st
    # another document while it runs: it goes on
    drv.execute_script("window.__mace.activate(arguments[0])", doc_index(drv, "structures/LTA.cif"))
    time.sleep(1.0)
    assert drv.execute_script(TASK, "Porosity: SOD")[1] in ("running", "done")
    H.wait(drv, "const r = window.__maceAnalysis.resultsFor(arguments[0]).porosity; return !!r && r.probe.name === 'CO2' && r.grid.spacing < 0.1", 180, sod)
    assert drv.execute_script(TASK, "Porosity: SOD")[1] == "done"
    # back to sodalite: its result is shown
    drv.execute_script("window.__mace.activate(arguments[0])", doc_index(drv, "structures/SOD.cif"))
    H.wait(drv, "const o = document.querySelector('#analysis-panel .an-out'); return o && o.textContent.includes('probe CO2')", 10)
    # Cancel in the Tasks tab stops a run; the panel says so and keeps the earlier result
    drv.execute_script("document.querySelector('#analysis-panel [data-p=spacing]').value = '0.07'")
    click_run(drv)
    H.wait(drv, "const t = [...window.__mace.tasks().values()].find((t) => t.title === 'Porosity: SOD' && t.state === 'running'); return !!t", 20)
    H.wait(drv, "const b = [...document.querySelectorAll('.bottom-pane.tasks .task-row')].find((r) => r.textContent.includes('Porosity: SOD')"
                " && r.querySelector('[data-act=cancel]')); if (b) b.querySelector('[data-act=cancel]').click(); return !!b", 10)
    H.wait(drv, "const s = document.querySelector('#analysis-panel [data-k=status]'); return s && s.textContent.includes('Canceled')", 30)
    assert drv.execute_script(TASK, "Porosity: SOD")[1] == "canceled"
    assert drv.execute_script("return window.__maceAnalysis.resultsFor(arguments[0]).porosity.grid.spacing", sod) == pytest.approx(0.08, abs=0.01)
    assert not H.errors(drv), H.errors(drv)
