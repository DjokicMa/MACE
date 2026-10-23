"""The bottom panel's Analysis tab and Inspector › 3D Overlays in a real
browser (1.1.6): a porosity result is listed with the settings it was
computed with, its key numbers and its per-feature table (every channel and
pocket, whose volumes and surfaces add up to the totals), a click on a
feature highlights and frames it, the result's actions work (show/hide,
command line, JSON/CSV, run again, delete), the 3D Overlays section shows,
restyles, makes see-through, hides and deletes each overlay on its own, and
nothing logs an error. Screenshots (1400x900) go to MACE_GUI_SHOTS."""
import json
import time

import pytest

import guiharness as H
import p4util as U

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")
pytest.importorskip("scipy")


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("anres_e2e")
    root = H.examples_copy(tmp)
    from ase.io import write
    from ase.spacegroup import crystal
    sod = crystal(["Si", "O"], basis=[(0.25, 0.5, 0.0), (0.1467, 0.5, 0.8533)], spacegroup=229, cellpar=[8.965] * 3 + [90] * 3)
    write(str(root / "structures" / "SOD.cif"), sod)
    lta = crystal(["O", "O", "O", "Si"], basis=[(0, 0.2122, 0.5), (0.1103, 0.1103, 0.3384), (0, 0.2967, 0.2967), (0, 0.1823, 0.3684)],
                  spacegroup=221, cellpar=[11.919] * 3 + [90] * 3)
    write(str(root / "structures" / "LTA.cif"), lta)
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


def tab(drv):
    drv.execute_script("document.querySelector('.bottom-tab[data-bottom=analysis]').click()")
    H.wait(drv, "return !!document.querySelector('.bottom-pane.analysis .anr-detail h3')", 15)


def close_panel(drv):
    drv.execute_script("const d = document.getElementById('analysis-panel'); if (d && d.open) d.querySelector('[data-close]').click()")


def test_porosity_result_in_the_analysis_tab(app):
    drv, downloads = app[0], app[5]
    porosity(drv, "structures/LTA.cif")
    close_panel(drv)
    # a taller bottom panel: the per-feature table in sight
    drv.execute_script("document.querySelector('.shell').style.setProperty('--h-bottom', '400px')")
    tab(drv)
    det = ".bottom-pane.analysis .anr-detail"
    text = drv.execute_script(f"return document.querySelector('{det}').textContent")
    assert "Porosity of LTA.cif" in text
    for s in ("Probe N2, r = 1.820 Å", "Atomic radii CCDC", "Grid spacing 0.250 Å", "Engine MACE native"):
        assert s in " ".join(text.split()), s
    assert drv.execute_script(f"return document.querySelector('{det} .anr-cli').textContent") == \
        "mace analyze porosity structures/LTA.cif --spacing 0.25"
    # the alpha cages (one 3D channel system) and the beta cage (a closed pocket), each a row
    rows = drv.execute_script(f"return [...document.querySelectorAll('{det} .anr-feat')].map((r) => [...r.cells].map((c) => c.textContent.trim()))")
    assert [r[1] for r in rows] == ["Channel", "Pocket"] and rows[0][2].startswith("3D") and rows[1][2] == "0D"
    # the features add up to the totals (POAV, PONAV, ASA, NASA), and the footer says so
    sums = drv.execute_script("""const r = window.__maceAnalysis.results().porosity, F = r.features;
        const s = (acc, k) => F.filter((f) => f.accessible === acc).reduce((t, f) => t + f[k], 0);
        return [s(true, 'occupiable_A3') - r.POAV.A3, s(false, 'occupiable_A3') - r.PONAV.A3, s(true, 'area_A2') - r.ASA.A2,
                s(false, 'area_A2') - r.NASA.A2, s(true, 'volume_A3') - r.AV.A3, s(false, 'volume_A3') - r.NAV.A3]""")
    assert all(abs(x) < 1e-6 for x in sums), sums
    foot = drv.execute_script(f"return document.querySelector('{det} tfoot').textContent")
    assert "Total, 1 channel" in foot and "Total, 1 pocket" in foot and "= POAV, ASA" in foot and "= PONAV, NASA" in foot
    # a click on the pocket: drawn in the accent color, the rest dimmed; again: back
    drv.execute_script(f"document.querySelectorAll('{det} .anr-feat')[1].click()")
    H.wait(drv, f"return document.querySelectorAll('{det} .anr-feat')[1].getAttribute('aria-selected') === 'true'", 5)
    hl = "return window.MACE.state.viewers['viewer-main'].shapes.filter((s) => s && s.maceHighlight).length"
    assert drv.execute_script("return window.__maceAnalysis.overlay.find('pores').hl") == 1
    assert drv.execute_script(hl) >= 1
    U.shot(drv, "analysis_tab_porosity_features")
    drv.execute_script(f"document.querySelectorAll('{det} .anr-feat')[1].click()")
    time.sleep(0.3)
    assert drv.execute_script("return window.__maceAnalysis.overlay.find('pores').hl") is None and drv.execute_script(hl) == 0
    # Hide / Show in 3D
    act = lambda a: drv.execute_script(f"document.querySelector('{det} [data-a={a}]').click()")
    act("overlay")
    H.wait(drv, "return window.__maceAnalysis.overlay.count('pores') === 0", 5)
    act("overlay")
    H.wait(drv, "return window.__maceAnalysis.overlay.count('pores') === 2", 5)
    # JSON and CSV: the settings with the numbers; the CSV one row per feature
    act("json")
    act("csv")
    H.wait(drv, "return true", 1)
    end = time.time() + 10
    ready = lambda f: (downloads / f).exists() and (downloads / f).stat().st_size > 0
    while time.time() < end and not (ready("LTA_porosity.json") and ready("LTA_porosity.csv")):
        time.sleep(0.2)
    time.sleep(0.3)
    js = json.loads((downloads / "LTA_porosity.json").read_text())
    assert js["tool"] == "porosity" and js["settings"]["Probe"] == "N2, r = 1.820 Å" and "render" not in js["result"]
    assert js["command"] == "mace analyze porosity structures/LTA.cif --spacing 0.25" and len(js["result"]["features"]) == 2
    csv = (downloads / "LTA_porosity.csv").read_text().splitlines()
    data = [ln for ln in csv if not ln.startswith("#")]
    assert data[0].startswith("feature,type,accessible") and len(data) == 3 and any("Probe: N2" in ln for ln in csv)
    # Run Again: the same tool and settings, a second result listed
    act("rerun")
    H.wait(drv, "return document.querySelectorAll('.bottom-pane.analysis .anr-row').length === 2", 120)
    assert drv.execute_script("return window.__maceAnalysis.store.forDoc(window.__mace.doc()).map((r) => r.params.spacing)") == ["0.25", "0.25"]
    # Delete Result: one row left
    act("delete")
    H.wait(drv, "return document.querySelectorAll('.bottom-pane.analysis .anr-row').length === 1", 5)
    assert not H.errors(drv), H.errors(drv)


def test_other_tools_are_listed_with_their_tables(app):
    drv = app[0]
    # another open document with a result of its own: All open documents lists it beside Si's
    U.open_doc(drv, "structures/LTA.cif")
    drv.execute_script("window.__mace.run('analysis.structure.pxrd')")
    H.wait(drv, "const d = document.getElementById('analysis-panel'); return d && d.open && !!d.querySelector('.an-form')", 15)
    drv.execute_script("window.__prev = window.__maceAnalysis.results().pxrd || null;"
                       "document.querySelector('#analysis-panel .an-form button[type=submit]').click()")
    H.wait(drv, "const r = window.__maceAnalysis.results().pxrd; return !!r && r !== window.__prev", 60)
    close_panel(drv)
    U.open_doc(drv, "structures/Si.cif")
    drv.execute_script("window.__mace.run('analysis.structure.pxrd')")
    H.wait(drv, "const d = document.getElementById('analysis-panel'); return d && d.open && !!d.querySelector('.an-form')", 15)
    drv.execute_script("document.querySelector('#analysis-panel .an-form button[type=submit]').click()")
    H.wait(drv, "return !!window.__maceAnalysis.results().pxrd", 60)
    drv.execute_script("document.querySelector('#analysis-panel .an-tab[data-tab=topology]').click()")
    drv.execute_script("document.querySelector('#analysis-panel .an-form button[type=submit]').click()")
    H.wait(drv, "return !!window.__maceAnalysis.results().topology", 60)
    close_panel(drv)
    tab(drv)
    rows = drv.execute_script("return [...document.querySelectorAll('.bottom-pane.analysis .anr-row')].map((r) => r.querySelector('b').textContent)")
    assert rows == ["Topology", "Powder pattern"]
    det = ".bottom-pane.analysis .anr-detail"
    assert "dia" in drv.execute_script(f"return document.querySelector('{det}').textContent")
    drv.execute_script("document.querySelectorAll('.bottom-pane.analysis .anr-row')[1].click()")
    H.wait(drv, f"return document.querySelector('{det} h3').textContent.startsWith('Powder pattern')", 5)
    text = " ".join(drv.execute_script(f"return document.querySelector('{det}').textContent").split())
    assert "Radiation CuKa, λ = 1.54" in text and "(1 1 1)" in text
    # All open documents: LTA's result too
    drv.execute_script("document.querySelector('.bottom-pane.analysis [data-scope=all]').click()")
    docs = drv.execute_script("return [...document.querySelectorAll('.bottom-pane.analysis .anr-row .anr-doc')].map((e) => e.textContent)")
    assert "LTA.cif" in docs and "Si.cif" in docs
    drv.execute_script("document.querySelector('.bottom-pane.analysis [data-scope=doc]').click()")
    assert not H.errors(drv), H.errors(drv)


SHAPES = """const v = window.MACE.state.viewers['viewer-main'];
    return v.shapes.filter((s) => s && s.opacity < 1 && !s.macePass).map((s) => [Math.round(100 * s.opacity) / 100, !!s.wireframe,
      s.color && s.color.getHex ? '#' + s.color.getHex().toString(16).padStart(6, '0') : s.color || null])"""


def test_overlay_manager(app):
    drv = app[0]
    porosity(drv, "structures/LTA.cif")
    drv.execute_script("document.querySelector('#analysis-panel .an-tab[data-tab=topology]').click()")
    drv.execute_script("document.querySelector('#analysis-panel .an-form button[type=submit]').click()")
    H.wait(drv, "return !!window.__maceAnalysis.results().topology", 60)
    drv.execute_script("document.querySelector('#analysis-panel [data-o=net]').click()")
    close_panel(drv)
    sec = '#inspector .insp-section[data-key="overlays"]'
    H.wait(drv, f"return document.querySelectorAll('{sec} .ovm-row').length === 2", 10)
    drv.execute_script(f"const s = document.querySelector('{sec}'); if (s.classList.contains('collapsed')) s.querySelector('.insp-head').click(); s.scrollIntoView()")
    names = drv.execute_script(f"return [...document.querySelectorAll('{sec} .ovm-name')].map((b) => b.textContent.trim())")
    assert names[0].startswith("Net") and names[1].startswith("Pores · N2, r 1.82 Å"), names
    count = lambda k: drv.execute_script("return window.__maceAnalysis.overlay.count(arguments[0])", k)
    assert count("pores") == 2 and count("net") > 0
    # pick the pores: their own controls
    drv.execute_script(f"document.querySelectorAll('{sec} .ovm-name')[1].click()")
    H.wait(drv, f"return !!document.querySelector('{sec} .ovm-edit [data-o=style]')", 5)
    ctl = lambda o, js: drv.execute_script(f"const e = document.querySelector('{sec} .ovm-edit [data-o={o}]'); {js}")
    ctl("style", "e.value = 'wireframe'; e.dispatchEvent(new Event('change', {bubbles: true}))")
    ctl("opacity", "e.value = '0.3'; e.dispatchEvent(new Event('input', {bubbles: true}))")
    shapes = drv.execute_script(SHAPES)
    assert len(shapes) == 2 and all(s[1] and s[0] == pytest.approx(0.3) for s in shapes), shapes
    # only this overlay: Settings keep their defaults
    assert drv.execute_script("return [window.__mace.settings.get('pores.style'), window.__mace.settings.get('pores.opacity')]") == ["solid", 0.5]
    ctl("nacColor", "e.value = '#ff00ff'; e.dispatchEvent(new Event('change', {bubbles: true}))")
    assert any(s[2] == "#ff00ff" for s in drv.execute_script(SHAPES))
    drv.execute_script(f"const s = document.querySelector('{sec}'), i = document.getElementById('inspector');"
                       "i.scrollTop += s.getBoundingClientRect().top - i.getBoundingClientRect().top - 48")
    U.shot(drv, "inspector_overlay_manager")
    # the pocket on its own: left out, then back
    drv.execute_script(f"const e = document.querySelector('{sec} .ovm-edit [data-f=\"1\"]'); e.checked = false; e.dispatchEvent(new Event('change', {{bubbles: true}}))")
    assert count("pores") == 1
    drv.execute_script(f"const e = document.querySelector('{sec} .ovm-edit [data-f=\"1\"]'); e.checked = true; e.dispatchEvent(new Event('change', {{bubbles: true}}))")
    assert count("pores") == 2
    # Reset: the Settings' look again
    drv.execute_script(f"document.querySelector('{sec} .ovm-edit [data-a=reset]').click()")
    shapes = drv.execute_script(SHAPES)
    assert len(shapes) == 2 and all(not s[1] and s[0] == pytest.approx(0.5) for s in shapes), shapes
    # visibility: the eye hides and shows one overlay, the other stays
    eye = lambda i: drv.execute_script(f"document.querySelectorAll('{sec} .ovm-eye')[{i}].click()")
    eye(1)
    H.wait(drv, "return window.__maceAnalysis.overlay.count('pores') === 0", 5)
    assert count("net") > 0
    eye(1)
    H.wait(drv, "return window.__maceAnalysis.overlay.count('pores') === 2", 5)
    # Delete: the overlay and its mesh data go, the result stays (without its surfaces)
    drv.execute_script(f"document.querySelectorAll('{sec} .ovm-row [data-a=delete]')[1].click()")
    H.wait(drv, f"return document.querySelectorAll('{sec} .ovm-row').length === 1", 5)
    assert count("pores") == 0 and count("net") > 0
    got = drv.execute_script("const r = window.__maceAnalysis.results().porosity; return [!!r, 'render' in r, r.render_deleted]")
    assert got == [True, False, True]
    tab(drv)
    assert drv.execute_script("return document.querySelector('.bottom-pane.analysis .anr-detail [data-a=overlay]')") is None or \
        drv.execute_script("const b = [...document.querySelectorAll('.bottom-pane.analysis .anr-row')].find((r) => r.textContent.includes('Porosity')); b.click();"
                           "return document.querySelector('.bottom-pane.analysis .anr-detail [data-a=overlay]').disabled")
    assert not H.errors(drv), H.errors(drv)


def test_analyze_structures_batch(app):
    """Three structure files picked in the Explorer, right-click › Analyze Structures…: one task per file,
    at most the chosen number at a time; the Analysis tab's Batches show the summary (the columns of
    `mace analyze porosity FILE... --csv`), Export CSV writes it, and each file's result is listed with
    that file's document once it is open."""
    drv, downloads = app[0], app[5]
    files = ["structures/Si.cif", "structures/MgO.cif", "structures/SOD.cif"]
    drv.execute_script("""for (const p of arguments[0]) {
        const b = document.querySelector(`#file-list .file[data-path="${p}"]`);
        b.dispatchEvent(new MouseEvent('click', {bubbles: true, ctrlKey: true}));
      }""", files)
    H.wait(drv, "return window.MACE.state.selected.size === 3", 5)
    drv.execute_script("""const b = document.querySelector('#file-list .file[data-path="structures/SOD.cif"]'), r = b.getBoundingClientRect();
        b.dispatchEvent(new MouseEvent('contextmenu', {bubbles: true, cancelable: true, button: 2, clientX: r.left + 8, clientY: r.top + 4}))""")
    item = H.wait(drv, "return [...document.querySelectorAll('.menu-item')].find((m) => m.textContent.includes('Analyze Structures'))", 5)
    drv.execute_script("arguments[0].click()", item)
    dlg = H.wait(drv, "const d = document.getElementById('analysis-batch'); return d && d.open && d", 10)
    assert drv.execute_script("return arguments[0].querySelector('[data-k=count]').textContent", dlg).startswith("3 of ")
    # every field of the tool's form is named by its label (for=, or the label around it)
    unlabeled = drv.execute_script("""const d = arguments[0];
        return [...d.querySelectorAll('[data-p]')].filter((e) => !e.closest('label') && !(e.id && d.querySelector(`label[for="${e.id}"]`)))
          .map((e) => e.dataset.p)""", dlg)
    assert unlabeled == [], unlabeled
    drv.execute_script("""const d = arguments[0];
        d.querySelector('[data-p=probe]').value = 'He'; d.querySelector('[data-p=spacing]').value = '0.3';
        d.querySelector('[data-p=spacing]').dispatchEvent(new Event('change', {bubbles: true}));
        d.querySelector('[data-k=jobs]').value = '2';""", dlg)
    cli = drv.execute_script("return arguments[0].querySelector('[data-k=cli]').textContent", dlg)
    assert cli == "Command line: mace analyze porosity FILE... --probe He --spacing 0.3 --csv porosity_summary.csv", cli
    drv.execute_script("arguments[0].querySelector('[data-k=run]').click()", dlg)
    # at most two of this batch's tasks run at once
    most = 0
    end = time.time() + 180
    while time.time() < end:
        st = drv.execute_script("return window.__maceAnalysis.store.batches()[0].rows.map((r) => r.state)")
        most = max(most, st.count("running"))
        if all(s in ("done", "failed", "canceled") for s in st):
            break
        time.sleep(0.05)
    assert st == ["done", "done", "done"] and most <= 2, (st, most)
    # the Analysis tab, on the batch: one row per structure
    H.wait(drv, "return !!document.querySelector('.bottom-pane.analysis .anr-batch')", 10)
    assert drv.execute_script("return document.querySelector('.bottom-pane.analysis [data-scope=batch]').getAttribute('aria-selected')") == "true"
    rows = drv.execute_script("return [...document.querySelectorAll('.bottom-pane.analysis .anr-batch tbody tr')].map((r) => [...r.cells].map((c) => c.textContent.trim()))")
    head = drv.execute_script("return [...document.querySelectorAll('.bottom-pane.analysis .anr-batch thead th')].map((c) => c.textContent)")
    assert [r[head.index("file")] for r in rows] == files and all(r[0] == "Done" for r in rows)
    assert {"probe", "Di", "Df", "POAV_A3", "n_pockets"} <= set(head)
    sod = rows[2]
    assert sod[head.index("probe")] == "He" and sod[head.index("n_pockets")] == "2"
    drv.execute_script("document.querySelector('.shell').style.setProperty('--h-bottom', '400px')")
    U.shot(drv, "analysis_batch_summary")
    drv.execute_script("document.querySelector('.bottom-pane.analysis .anr-detail [data-a=csv]').click()")
    end = time.time() + 10
    while time.time() < end and not ((downloads / "porosity_summary.csv").exists() and (downloads / "porosity_summary.csv").read_text().count("\n") >= 4):
        time.sleep(0.2)
    lines = (downloads / "porosity_summary.csv").read_text().splitlines()
    assert lines[0].startswith("file,probe,probe_radius") and [ln.split(",")[0] for ln in lines[1:]] == files
    # the same columns as the command line's summary
    from mace.analysis.cli import summary_row
    assert lines[0].split(",") == list(summary_row("porosity", {}, "x").keys())
    # Open on a row: that structure's document
    drv.execute_script("document.querySelectorAll('.bottom-pane.analysis .anr-batch [data-open]')[2].click()")
    H.wait(drv, "const d = window.__mace.doc(); return !!d && d.path === 'structures/SOD.cif'", 20)
    # each file's document lists its result
    U.open_doc(drv, "structures/MgO.cif")
    got = drv.execute_script("return window.__maceAnalysis.store.forDoc(window.__mace.doc()).map((r) => [r.tool, !!r.batch, r.result.probe.name])")
    assert got == [["porosity", True, "He"]]
    assert not H.errors(drv), H.errors(drv)


def test_a_picked_feature_and_a_batch_overlay_go_with_their_result(app):
    """A picked channel's highlight goes when the scope (or the row, or the document) changes: the table forgot
    it, the 3D view kept it. A batch result's surfaces go with it when it is deleted while its structure is
    not in front (they stayed on that document)."""
    drv = app[0]
    porosity(drv, "structures/SOD.cif", probe="He", spacing="0.3")
    close_panel(drv)
    tab(drv)
    drv.execute_script("document.querySelector('.bottom-pane.analysis [data-scope=doc]').click()")    # the batch test left Batches
    det = ".bottom-pane.analysis .anr-detail"
    H.wait(drv, f"return document.querySelectorAll('{det} .anr-feat').length > 0", 15)
    drv.execute_script(f"document.querySelectorAll('{det} .anr-feat')[0].click()")
    H.wait(drv, "const it = window.__maceAnalysis.overlay.find('pores'); return !!it && it.hl != null", 5)
    drv.execute_script("document.querySelector('.bottom-pane.analysis [data-scope=all]').click()")
    H.wait(drv, "const it = window.__maceAnalysis.overlay.find('pores'); return !!it && it.hl === null", 5)
    # a batch result drawn on SOD.cif, deleted from another document
    sod = drv.execute_script("return window.__mace.doc().id")
    drv.execute_script("window.__maceAnalysis.store.batch('porosity', {probe: 'He', spacing: '0.3', render: true}, ['structures/SOD.cif'], 1)")
    H.wait(drv, "const b = window.__maceAnalysis.store.batches()[0]; return b.rows.length === 1 && b.rows[0].state === 'done'", 120)
    rid = drv.execute_script("return window.__maceAnalysis.store.batches()[0].rows[0].rec")
    assert drv.execute_script("const A = window.__maceAnalysis.store; return A.show(A.get(arguments[0]))", rid)
    H.wait(drv, "const it = window.__maceAnalysis.overlay.find('pores', arguments[0]); return !!it && it.meta.rec === arguments[1]", 5, sod, rid)
    U.open_doc(drv, "structures/MgO.cif")
    drv.execute_script("window.__maceAnalysis.store.remove(arguments[0])", rid)
    assert drv.execute_script("return window.__maceAnalysis.overlay.find('pores', arguments[0])", sod) is None
    assert not H.errors(drv), H.errors(drv)



def test_a_pore_overlay_is_built_once_and_drawn_once_per_paint(app):
    """3Dmol rebuilt every shape's mesh on each render() (a hover, a click, a zoom step) and drew every
    translucent mesh twice per paint (a depth pre-pass, then the color): the pores, drawn without writing
    depth, keep their mesh across renders and skip that pre-pass (viewport/shape_cache.js)."""
    drv = app[0]
    porosity(drv, "structures/LTA.cif")
    close_panel(drv)
    H.wait(drv, "return window.__maceAnalysis.overlay.count('pores') === 2", 10)
    r = drv.execute_script("""const v = window.MACE.state.viewers['viewer-main'], $3 = window.$3Dmol;
      const sh = window.__maceAnalysis.overlay.find('pores').shapes, geos = new Set(sh.map((s) => s.geo));
      const fin = $3.GLShape.finalizeGeo;
      let n = 0;
      $3.GLShape.finalizeGeo = function (g) { if (geos.has(g)) n++; return fin.call(this, g); };
      try { for (let i = 0; i < 5; i++) v.render(); } finally { $3.GLShape.finalizeGeo = fin; }
      const meshes = sh.flatMap((s) => s.renderedShapeObj.children);
      const noDepth = meshes.every((m) => m.material.depthWrite === false);
      // the renderer's per-paint draw list entry for each mesh: no depth pre-pass
      const prepass = meshes.filter((m) => { const o = { object: m }; v.renderer.unrollBufferMaterial(o); return !!o.transparentDepth; }).length;
      sh.forEach((s) => { s.hidden = true; });
      v.render();
      const gone = sh.every((s) => !s.renderedShapeObj);
      sh.forEach((s) => { s.hidden = false; });
      v.render();
      return {shapes: sh.length, n, meshes: meshes.length, noDepth, prepass, gone, back: sh.every((s) => !!s.renderedShapeObj)};""")
    assert r["shapes"] == 4 and r["meshes"] == 4, r
    assert r["n"] == 0, r                        # no mesh rebuilt by 5 renders
    assert r["prepass"] == 0, r                  # each translucent mesh drawn once per paint
    assert r["noDepth"] and r["gone"] and r["back"], r
    assert not H.errors(drv), H.errors(drv)
