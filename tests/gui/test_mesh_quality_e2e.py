"""Mesh quality and mesh style in a real browser (1.1.6): a preset picked for
one overlay re-meshes that overlay only (the net next to it keeps its
shapes), going back is instant, every style draws without an error and an
unchanged one is never rebuilt, Export Image at Publication quality puts the
on-screen mesh back, Settings › Mesh quality is what a new result is made
at and survives a reload, the optional "drop the faint near walls while
dragging" hides them during a drag only, and a cube's quality and styles
work in the Volume view."""
import shutil
import time

import pytest

import guiharness as H
import p4util as U

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")
pytest.importorskip("scipy")
from selenium.webdriver.common.action_chains import ActionChains  # noqa: E402
from selenium.webdriver.common.by import By  # noqa: E402

CORPUS = H.REPO_ROOT / "test" / "ECH3POT3"
MOL = "DEC_MOLECULE_OPT_symm_HSESOL3C_SOLDEF2MSVP_opt_HSESOL3C_optimized_sp_HSESOL3C_optimized_charge+potential"


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("mesh_e2e")
    root = H.examples_copy(tmp)
    from ase.io import write
    from ase.spacegroup import crystal
    lta = crystal(["O", "O", "O", "Si"], basis=[(0, 0.2122, 0.5), (0.1103, 0.1103, 0.3384), (0, 0.2967, 0.2967), (0, 0.1823, 0.3684)],
                  spacegroup=221, cellpar=[11.919] * 3 + [90] * 3)
    write(str(root / "structures" / "LTA.cif"), lta)
    if (CORPUS / f"{MOL}_DENS.CUBE").is_file():
        (root / "vol").mkdir()
        shutil.copyfile(CORPUS / f"{MOL}_DENS.CUBE", root / "vol" / "mol_DENS.CUBE")
    a = U.start_app(tmp, width=1400, height=900, root=root, MACE_GUI_MAX_WORKERS="2")
    try:
        yield a
    finally:
        U.stop_app(a)


def porosity(drv, path="structures/LTA.cif", ready=None):
    """Pores of LTA from Analyze › Porosity; `ready()` runs once the panel (and window.__maceAnalysis) is up."""
    U.open_doc(drv, path)
    drv.execute_script("window.__mace.run('analysis.structure.porosity')")
    H.wait(drv, "const d = document.getElementById('analysis-panel'); return d && d.open && !!d.querySelector('.an-form')", 15)
    if ready:
        ready()
    drv.execute_script("document.querySelector('#analysis-panel [data-p=probe]').value = 'N2';"
                       "document.querySelector('#analysis-panel [data-p=spacing]').value = '0.25';"
                       "window.__prev = window.__maceAnalysis.results().porosity || null;"
                       "document.querySelector('#analysis-panel .an-form button[type=submit]').click()")
    H.wait(drv, "const r = window.__maceAnalysis.results().porosity; return !!r && r !== window.__prev", 120)
    H.wait(drv, "return window.__maceAnalysis.overlay.count('pores') > 0", 10)


def close_panel(drv):
    drv.execute_script("const d = document.getElementById('analysis-panel'); if (d && d.open) d.querySelector('[data-close]').click()")


def pores(drv):
    """The pore overlay's Inspector entry: {key, madeAt, remeshing, opts, ...} (no mesh data); None before
    any (the analysis module loads with the first Analyze panel: a test run on its own has none yet)."""
    return drv.execute_script("""if (!window.__maceAnalysis) return null;
      const O = window.__maceAnalysis.overlay, it = O.find('pores');
      const o = it && O.list(it.doc).find((x) => x.key === it.key);
      return o ? { key: o.key, madeAt: o.madeAt, remeshing: o.remeshing, opts: o.opts, over: o.over, drawn: o.drawn } : null;""")


def wait_made(drv, quality, timeout=120):
    H.wait(drv, """const O = window.__maceAnalysis.overlay, it = O.find('pores'); const o = it && O.list(it.doc).find((x) => x.key === it.key);
      return !!o && !o.remeshing && (o.madeAt || {}).quality === arguments[0]""", timeout, quality)
    return pores(drv)


REBUILDS = """const v = window.MACE.state.viewers['viewer-main'], $3 = window.$3Dmol, O = window.__maceAnalysis.overlay;
  const sh = O.find(arguments[0]).shapes, geos = new Set(sh.map((s) => s.geo)), fin = $3.GLShape.finalizeGeo;
  let n = 0;
  $3.GLShape.finalizeGeo = function (g) { if (geos.has(g)) n++; return fin.call(this, g); };
  try { for (let i = 0; i < 3; i++) v.render(); } finally { $3.GLShape.finalizeGeo = fin; }
  return n;"""


def test_a_preset_remeshes_that_overlay_only_and_back_is_instant(app):
    drv = app[0]
    porosity(drv)
    drv.execute_script("document.querySelector('#analysis-panel .an-tab[data-tab=topology]').click()")
    drv.execute_script("document.querySelector('#analysis-panel .an-form button[type=submit]').click()")
    H.wait(drv, "return !!window.__maceAnalysis.results().topology", 60)
    drv.execute_script("document.querySelector('#analysis-panel [data-o=net]').click()")
    close_panel(drv)
    H.wait(drv, "return window.__maceAnalysis.overlay.count('net') > 0", 10)
    p0 = pores(drv)
    assert p0["madeAt"]["quality"] == "standard" and p0["madeAt"]["triangles"] > 1000, p0
    drv.execute_script("window.__net = window.__maceAnalysis.overlay.find('net').shapes.slice()")
    # the Inspector's Mesh quality select (the section's editor of the pores)
    sec = '#inspector .insp-section[data-key="overlays"]'
    drv.execute_script(f"const s = document.querySelector('{sec}'); if (s.classList.contains('collapsed')) s.querySelector('.insp-head').click(); s.scrollIntoView()")
    H.wait(drv, f"return document.querySelectorAll('{sec} .ovm-row').length === 2", 10)
    pick = f"[...document.querySelectorAll('{sec} .ovm-name')].find((b) => b.textContent.startsWith('Pores')).click()"
    drv.execute_script(pick)
    sel = H.wait(drv, f"return document.querySelector('{sec} .ovm-edit [data-o=quality]')", 5)
    assert sel.get_attribute("value") == "standard"
    drv.execute_script("arguments[0].value = 'draft'; arguments[0].dispatchEvent(new Event('change', {bubbles: true}))", sel)
    H.wait(drv, f"const e = document.querySelector('{sec} .ovm-mesh'); return !!e && /Re-meshing at Draft|Made at Draft/.test(e.textContent)", 10)
    p1 = wait_made(drv, "draft")
    assert p1["madeAt"]["triangles"] < p0["madeAt"]["triangles"] and p1["madeAt"]["render_grid"][0] < p0["madeAt"]["render_grid"][0], (p0, p1)
    # the net was not touched: the same shapes, still in the viewer
    assert drv.execute_script("""const v = window.MACE.state.viewers['viewer-main'], now = window.__maceAnalysis.overlay.find('net').shapes;
      return now.length === window.__net.length && now.every((s, i) => s === window.__net[i] && v.shapes.includes(s))""")
    # High, then Standard again: kept, so no request and no wait
    drv.execute_script("window.__maceAnalysis.overlay.override(arguments[0], {quality: 'high'})", p0["key"])
    p2 = wait_made(drv, "high")
    assert p2["madeAt"]["triangles"] > p0["madeAt"]["triangles"], (p0, p2)
    back = drv.execute_script("""const O = window.__maceAnalysis.overlay; O.override(arguments[0], {quality: null});
      const o = O.list(O.get(arguments[0]).doc).find((x) => x.key === arguments[0]); return [o.madeAt.quality, o.madeAt.triangles, o.remeshing]""", p0["key"])
    assert back == ["standard", p0["madeAt"]["triangles"], False], back
    # a net's spheres follow the quality too (3Dmol's sphere detail)
    nv = lambda: drv.execute_script("return window.__maceAnalysis.overlay.find('net').shapes.reduce((n, s) => n + s.geo.vertices, 0)")
    n0 = nv()
    drv.execute_script("const O = window.__maceAnalysis.overlay; O.override(O.find('net').key, {quality: 'publication'})")
    assert nv() > n0
    drv.execute_script("const O = window.__maceAnalysis.overlay; O.override(O.find('net').key, {quality: null})")
    U.no_errors(drv)


def test_every_style_draws_and_an_unchanged_one_is_never_rebuilt(app):
    drv = app[0]
    if not pores(drv):
        porosity(drv)
        close_panel(drv)
    key = pores(drv)["key"]
    looks = [{"style": "solid"}, {"style": "solid", "shading": "flat"}, {"style": "solid", "sides": "single"},
             {"style": "solid", "sides": "double"}, {"style": "wireframe"}, {"style": "edges"}, {"style": "edges", "shading": "flat"},
             {"style": "points"}]
    prev = None
    for lk in looks:
        r = drv.execute_script("""const O = window.__maceAnalysis.overlay, v = window.MACE.state.viewers['viewer-main'];
          O.override(arguments[0], {style: null, shading: null, sides: null, ...arguments[1]});
          const sh = O.get(arguments[0]).shapes;
          window.__styleShapes = window.__styleShapes || [];
          const old = window.__styleShapes.filter((s) => v.shapes.includes(s)).length;
          window.__styleShapes = sh.slice();
          return {n: sh.length, surfaces: O.count('pores'), stale: old, faces: sh.reduce((t, s) => t + s.geo.geometryGroups.reduce((m, g) => m + g.faceidx, 0), 0),
            wire: sh.filter((s) => s.wireframe).length, near: sh.filter((s) => s.macePass === 'near').length, edges: sh.filter((s) => s.macePass === 'edges').length}""",
                              key, lk)
        assert r["n"] > 0 and r["surfaces"] > 0 and r["faces"] > 0 and r["stale"] == 0, (lk, r)
        if lk == {"style": "solid"}:
            assert r["near"] > 0 and r["wire"] == 0, r
        if lk.get("sides") in ("single", "double"):
            assert r["near"] == 0, (lk, r)
        if lk["style"] == "wireframe":
            assert r["wire"] == r["n"], r
        if lk["style"] == "edges":
            assert r["edges"] > 0 and r["wire"] == r["edges"], r
        assert drv.execute_script(REBUILDS, "pores") == 0, lk
        prev = r
    assert prev is not None
    drv.execute_script("window.__maceAnalysis.overlay.override(arguments[0], {style: null, shading: null, sides: null})", key)
    U.no_errors(drv)


def test_export_at_publication_puts_the_view_mesh_back(app):
    drv, dl = app[0], app[5]
    if not pores(drv):
        porosity(drv)
        close_panel(drv)
    p0 = pores(drv)
    assert p0["madeAt"]["quality"] == "standard"
    drv.execute_script("""window.__seen = []; window.__unseen = window.__maceAnalysis.overlay.onChange((why, it) => {
        if (it && it.kind === 'pores' && why === 'meshed') window.__seen.push(it.data.meshes.quality.quality); });
      window.__shot = null; window.__maceExportHook = (u) => { window.__shot = u; };""")
    drv.execute_script("window.__mace.run('file.exportImage')")
    d = H.wait(drv, "const d = document.getElementById('export-image'); return d && d.open && d", 10)
    box = d.find_element(By.CSS_SELECTOR, '[data-k="remesh"]')
    assert box.is_displayed() and not box.is_selected()
    drv.execute_script("arguments[0].click()", box)
    drv.execute_script("arguments[0].click()", d.find_element(By.CSS_SELECTOR, '[data-v="export"]'))
    H.wait(drv, "return !!window.__shot", 180)
    assert drv.execute_script("return window.__seen") == ["publication"]
    p1 = pores(drv)
    assert p1["madeAt"] == p0["madeAt"] and "quality" not in p1["over"], (p0, p1)
    drv.execute_script("window.__unseen(); window.__maceExportHook = null")
    U.no_errors(drv)


def test_settings_mesh_quality_is_what_a_new_result_is_made_at_and_it_persists(app):
    drv, base = app[0], app[1]
    drv.execute_script("window.__mace.settings.set('mesh.quality', 'draft')")
    porosity(drv, ready=lambda: drv.execute_script(
        """window.__asked = 0; window.__unask = window.__maceAnalysis.overlay.onChange((why) => { if (why === 'remesh') window.__asked++; });"""))
    close_panel(drv)
    p = wait_made(drv, "draft", 30)
    assert p["opts"]["quality"] == "draft" and drv.execute_script("return window.__asked") == 0, p
    drv.execute_script("window.__unask()")
    H.load(drv, base)
    assert drv.execute_script("return window.__mace.settings.get('mesh.quality')") == "draft"
    drv.execute_script("window.__mace.settings.set('mesh.quality', 'standard')")
    U.no_errors(drv)


def test_dropping_the_near_walls_while_dragging_is_optional(app):
    drv = app[0]
    porosity(drv)
    close_panel(drv)
    cv = drv.find_element(By.CSS_SELECTOR, "#viewer-main canvas")
    near = "return window.__maceAnalysis.overlay.find('pores').shapes.filter((s) => s.macePass === 'near').map((s) => !!s.hidden)"
    assert drv.execute_script(near) and not any(drv.execute_script(near))

    def during():
        ActionChains(drv).move_to_element(cv).click_and_hold().move_by_offset(30, 5).move_by_offset(30, 5).perform()
        try:
            return drv.execute_script(near)
        finally:
            ActionChains(drv).release().perform()

    assert not any(during())                    # off by default
    drv.execute_script("window.__mace.settings.set('pores.dropNearWhileDragging', true)")
    h = during()
    assert h and all(h), h
    H.wait(drv, "return window.__maceAnalysis.overlay.find('pores').shapes.filter((s) => s.macePass === 'near').every((s) => !s.hidden)", 5)
    drv.execute_script("window.__mace.settings.set('pores.dropNearWhileDragging', false)")
    U.no_errors(drv)


def test_a_cube_mesh_quality_and_styles(app):
    drv = app[0]
    if not (CORPUS / f"{MOL}_DENS.CUBE").is_file():
        pytest.skip("test/ corpus not present (gitignored)")
    drv.execute_script("window.__mace.settings.set('volume.quality', 'low')")
    drv.execute_script("window.MACE.state.filter = 'all'; window.MACE.state.text = ''; window.MACE.renderFiles()")
    el = H.wait(drv, "return document.querySelector('#file-list .file[data-path=\"vol/mol_DENS.CUBE\"]')", 10)
    drv.execute_script("arguments[0].scrollIntoView({block: 'center'})", el)
    ActionChains(drv).move_to_element(el).click().perform()
    H.wait(drv, "const s = window.__mace.volumeView && window.__mace.volumeView.state(); return s && s.path === 'vol/mol_DENS.CUBE' && s.surfaces > 0", 90)
    st = drv.execute_script("return window.__mace.volumeView.state()")
    assert st["mesh"]["quality"] == "standard" and st["mesh"]["grid"] == "low", st["mesh"]
    shape0, v0 = st["shape"], st["vertices"]

    def pick(id_, value):
        e = H.wait(drv, f"return document.getElementById('{id_}')", 10)
        drv.execute_script("arguments[0].value = arguments[1]; arguments[0].dispatchEvent(new Event('change', {bubbles: true}))", e, value)

    pick("vo-quality", "publication")
    H.wait(drv, "const s = window.__mace.volumeView.state(); return s.mesh.grid === 'publication' && !s.mesh.loading && s.surfaces > 0", 120)
    st = drv.execute_script("return window.__mace.volumeView.state()")
    assert st["shape"][0] > shape0[0] and st["vertices"] > v0, (shape0, st["shape"])
    pick("vo-quality", "draft")
    H.wait(drv, "const s = window.__mace.volumeView.state(); return s.mesh.grid === 'low' && !s.mesh.loading && s.surfaces > 0", 60)
    for style, shading in (("points", "smooth"), ("edges", "smooth"), ("solid", "flat"), ("wireframe", "smooth"), ("solid", "smooth")):
        pick("vo-style", style)
        pick("vo-shading", shading)
        H.wait(drv, "return window.__mace.volumeView.state().surfaces > 0", 10)
    U.no_errors(drv)
