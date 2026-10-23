"""The 3D view's Style and Lighting presets (Inspector › Rendering, 1.1.7):
each style draws what it says (Space-filling has no bonds, Wireframe thin
sticks only, Grayscale grays with outlines, VESTA-like VESTA's colors and
radii whatever Settings choose, Polyhedral one octahedron per Ti in the
drawn cells of TiPbO3, Magnetic colors following the sign of the Mulliken
moments with a legend), styles that cannot apply are grayed out with the
reason, each
lighting preset sets its values (Glossy dark: a specular highlight on a black
background), the two dropdowns combine without touching each other's keys,
and a document saved with the old Rendering preset keeps it as its lighting.
Real browser, software GL; Firefox or Chromium (MACE_GUI_BROWSER)."""
import re
import shutil
import time

import pytest

import guiharness as H
import realdata

pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")

from selenium.webdriver.common.action_chains import ActionChains  # noqa: E402
from selenium.webdriver.support.ui import Select  # noqa: E402

PRESETS_JS = H.REPO_ROOT / "mace" / "gui" / "static" / "js" / "viewport" / "presets.js"
STYLES = ["default", "vesta", "spacefill", "wireframe", "grayscale", "polyhedral", "magnetic"]
LIGHTS = ["default", "flat", "publication", "presentation", "glossy"]

# a final ALPHA-BETA Mulliken block for the two Si atoms of the demo single point: one up, one down
SPIN_BLOCK = """
 MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
     ALPHA-BETA ELECTRONS
 MULLIKEN POPULATION ANALYSIS - NO. OF ELECTRONS    0.000000

  ATOM    Z CHARGE  A.O. POPULATION

   1 SI  14  0.800  0.100  0.200  0.300  0.200
   2 SI  14 -0.800 -0.100 -0.200 -0.300 -0.200

"""


def _optional(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except pytest.skip.Exception:
        return None


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("presets")
    root = H.examples_copy(tmp)
    (root / "spin").mkdir()
    afm = (root / "outputs" / "Si_sp.out").read_text() + SPIN_BLOCK
    (root / "spin" / "afm_sp.out").write_text(afm)
    real = not H.remote()
    if real:
        _optional(realdata.copy, root / "real", "tipbo3.cif")
        p = _optional(realdata.first, "OPT/Ti19O30_mp-760075_*_optimized.out", "ALPHA-BETA ELECTRONS")
        if p:
            shutil.copyfile(p, root / "real" / "tio_spin.out")
        c = _optional(realdata.first, "_synthetic_cubes/cubic3D_DENS.cube")
        if c:
            shutil.copyfile(c, root / "real" / "cubic_DENS.cube")
    proc, base, _ = H.start_server(root, tmp / "xdg")
    drv = H.browser(width=1400, height=900)
    try:
        H.load(drv, base)
        if (root / "real" / "tio_spin.out").is_file():
            # the run's final geometry as a CIF of the same name: the structure view of a spin run
            cif = drv.execute_async_script("""const d = arguments[0];
              window.MACE.api('structure/export', {body: {source: {path: 'real/tio_spin.out'}, format: 'CIF', cif_mode: 'p1'}, blob: true})
                .then((r) => r.blob.text()).then(d, (e) => d('ERR ' + e.message))""")
            if not cif.startswith("ERR"):
                (root / "real" / "tio_spin.cif").write_text(cif)
                H.load(drv, base)                  # the file list again, with it
        yield drv, root, base
    finally:
        drv.quit()
        proc.terminate()
        proc.wait(timeout=15)


@pytest.fixture
def fresh(app):
    drv = app[0]
    drv.execute_script("window.__mace.closeAll(); document.querySelectorAll('dialog[open]').forEach((d) => d.close());")
    return app


def need(root, rel):
    if not (root / rel).is_file():
        pytest.skip(f"{rel}: test/ corpus file not present")


def open_doc(drv, path, kind="structure"):
    got = drv.execute_async_script(
        "const done = arguments[1]; window.__mace.open(arguments[0]).then((d) => done(d && d.kind), (e) => done('error: ' + e.message))", path)
    assert got == kind, (got, drv.execute_script("return [...document.querySelectorAll('#toasts > *')].map((t) => t.textContent)"))
    drv.execute_script("const m = window.__mace.display, a = window.__mace.activeDoc(); ['range', 'render', 'style'].forEach((g) => m.reset(a, g));")
    H.wait(drv, "return !!document.querySelector('#inspector .insp-section[data-key=render] #dp-stylePreset')", 15)
    time.sleep(0.8)


def expand(drv):
    sec = drv.find_element("css selector", "#inspector .insp-section[data-key=render]")
    if "collapsed" in sec.get_attribute("class"):
        el = sec.find_element("css selector", ".insp-head")
        drv.execute_script("arguments[0].scrollIntoView({block: 'center'})", el)
        ActionChains(drv).move_to_element(el).click().perform()
        time.sleep(0.2)
    return drv.find_element("css selector", "#inspector .insp-section[data-key=render]")


def choose(drv, which, value, settle=1.0):
    """Pick a preset in the Inspector's Style or Lighting dropdown, as a user does."""
    sel = expand(drv).find_element("id", "dp-stylePreset" if which == "style" else "dp-renderPreset")
    drv.execute_script("arguments[0].scrollIntoView({block: 'center'})", sel)
    Select(sel).select_by_value(value)
    time.sleep(settle)


def values(drv):
    return drv.execute_script("return window.__mace.display.of(window.__mace.activeDoc())")


def option(drv, value):
    return drv.execute_script("""const o = [...document.getElementById('dp-stylePreset').options].find((x) => x.value === arguments[0]);
      return {disabled: o.disabled, title: o.title, text: o.textContent}""", value)


def atom_styles(drv):
    """The 3Dmol style keys of every atom of the structure layer."""
    return drv.execute_script("""const m = window.__mace.viewport.scene().layers.structure;
      return m.selectedAtoms({}).map((a) => Object.keys(a.style).sort().join('+'));""")


def rgb(hexstr):
    n = int(hexstr[1:], 16)
    return n >> 16, (n >> 8) & 255, n & 255


# ---------------------------------------------------------------- the table is the one definition
def test_one_definition_reaches_settings_server_and_docs():
    src = PRESETS_JS.read_text()
    lighting = re.findall(r'id: "([a-z]+)", label: "([^"]+)"', src[src.index("LIGHTING ="):src.index("STYLES =")])
    styles = re.findall(r'id: "([a-z]+)", label: "([^"]+)"', src[src.index("STYLES ="):src.index("LIGHTING_KEYS")])
    assert [i for i, _ in lighting] == LIGHTS and [i for i, _ in styles] == STYLES
    from mace.gui import settings as S
    assert S._CHOICES["renderPreset"] == tuple(LIGHTS) + ("custom",)
    assert S._CHOICES["stylePreset"] == tuple(STYLES) + ("custom",)
    for k, v in (("renderPreset", "presentation"), ("renderPreset", "glossy"), ("stylePreset", "polyhedral"), ("atomColor", "spin")):
        assert S.check(k, v) is None, (k, v)
    assert S.check("stylePreset", "cartoon")
    docs = (H.REPO_ROOT / "docs" / "gui.md").read_text()
    for _, label in lighting + styles:
        assert f"| {label} |" in docs, label


def test_spins_endpoint_reads_the_final_block(tmp_path):
    from mace.gui import related
    out = tmp_path / "afm_sp.out"
    out.write_text((H.EXAMPLES / "outputs" / "Si_sp.out").read_text() + SPIN_BLOCK)
    r = related.spins(out, inside=lambda p: p.name)
    assert r["state"] == "ok" and r["values"] == [0.8, -0.8] and r["symbols"] == ["Si", "Si"] and not r["closed"]
    # a CIF beside it reads the output of the same name; a closed-shell output says why
    (tmp_path / "afm_sp.cif").write_text("data_x\n")
    assert related.spins(tmp_path / "afm_sp.cif", inside=lambda p: p.name)["values"] == [0.8, -0.8]
    r = related.spins(H.EXAMPLES / "outputs" / "Si_sp.out", inside=lambda p: p.name)
    assert r["state"] == "none" and "ALPHA-BETA" in r["reason"]
    # SPINLOCK's "ALPHA-BETA ELECTRONS LOCKED TO" line is not a Mulliken block
    lock = tmp_path / "lock.out"
    lock.write_text(" ALPHA-BETA ELECTRONS LOCKED TO   2 FOR  15 SCF CYCLES\n   1 FE  26  2.485\n   2 FE  26  2.870\n")
    assert related.spins(lock, inside=lambda p: p.name)["state"] == "none"


# ---------------------------------------------------------------- styles
def test_each_style_sets_only_its_own_keys(fresh):
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    choose(drv, "light", "presentation")
    lighting = {k: v for k, v in values(drv).items() if k in drv.execute_script("return window.__mace.display.GROUPS.render")}
    for s in ["vesta", "spacefill", "wireframe", "grayscale", "polyhedral", "default"]:
        choose(drv, "style", s)
        v = values(drv)
        want = drv.execute_script("return window.__mace.display.STYLE_PRESETS[arguments[0]]", s)
        assert v["stylePreset"] == s and all(v[k] == x for k, x in want.items()), (s, v)
        assert {k: v[k] for k in lighting} == lighting, s          # the lighting stays
    assert not H.errors(drv)


def test_space_filling_has_no_bonds(fresh):
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    assert drv.execute_script("return window.__mace.viewport.scene().pairs.length") > 0
    choose(drv, "style", "spacefill")
    assert drv.execute_script("return window.__mace.viewport.scene().pairs") is None
    assert set(atom_styles(drv)) == {"sphere"}
    st = drv.execute_script("return window.__mace.atomStyle(0)")
    assert st["scale"] == 0.95
    # van der Waals radii: O's sphere is 0.95 × 3Dmol's 1.52 Å
    o = drv.execute_script("const s = window.__mace.doc().structure; return s.symbols.indexOf('O')")
    assert drv.execute_script("return window.__mace.atomStyle(arguments[0]).radius", o) == pytest.approx(0.95 * 1.52, abs=1e-3)


def test_wireframe_draws_thin_sticks_only(fresh):
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    choose(drv, "style", "wireframe")
    assert set(atom_styles(drv)) == {"stick"}
    r = drv.execute_script("const m = window.__mace.viewport.scene().layers.structure; return m.selectedAtoms({})[0].style.stick.radius")
    assert r == pytest.approx(0.06)
    assert drv.execute_script("return window.__mace.viewport.scene().pairs.length") > 0


def test_grayscale_is_gray_with_outlines(fresh):
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    choose(drv, "style", "grayscale")
    n = drv.execute_script("return window.__mace.doc().structure.symbols.length")
    colors = {drv.execute_script("return window.__mace.atomStyle(arguments[0]).color", i) for i in range(n)}
    assert all(len(set(rgb(c))) == 1 for c in colors), colors
    assert len(colors) == 2                                    # Mg and O stay apart
    look = drv.execute_script("const v = window.MACE.state.viewers['viewer-main']; v.paintNow(); return JSON.parse(v.maceLook)")
    assert look[3] is True                                     # the outline, under the Default lighting
    assert values(drv)["outline"] is False                      # ...which the style did not change


def test_vesta_like_pins_vesta_whatever_settings_choose(fresh):
    drv = fresh[0]
    drv.execute_script("window.__mace.settings.set('colorScheme', 'jmol'); window.__mace.settings.set('radiusSet', 'covalent')")
    try:
        open_doc(drv, "structures/MgO.cif")
        vesta = drv.execute_script("return window.MACE.paletteMap('vesta').O")
        jmol = drv.execute_script("return window.MACE.paletteMap('jmol').O")
        assert vesta.lower() != jmol.lower()
        assert drv.execute_script("return window.MACE.elementColor('O')").lower() == jmol.lower()
        choose(drv, "style", "vesta")
        assert drv.execute_script("return window.MACE.elementColor('O')").lower() == vesta.lower()
        assert drv.execute_script("return window.$3Dmol.GLModel.vdwRadii.O") == pytest.approx(0.74)      # VESTA's atomic radius
        o = drv.execute_script("const s = window.__mace.doc().structure; return s.symbols.indexOf('O')")
        assert drv.execute_script("return window.__mace.atomStyle(arguments[0]).color", o).lower() == vesta.lower()
    finally:
        drv.execute_script("window.__mace.settings.reset('colorScheme'); window.__mace.settings.reset('radiusSet')")


def test_polyhedral_draws_one_octahedron_per_ti_in_the_range(fresh):
    """Since 1.1.6 Pb bonds to O out to the sum of the Shannon ionic radii (heavy-element bonds), so in this
    TiPbO3 (I4/m) both cations are centers (their ligands are O): TiO6 octahedra and PbO5 square pyramids
    (Pb, with fewer ligands than Ti, is no A site here). A center gets its polyhedron once, inside the drawn
    cells: the boundary copies on the far faces draw none (1.1.6)."""
    drv, root = fresh[0], fresh[1]
    need(root, "real/tipbo3.cif")
    open_doc(drv, "real/tipbo3.cif")
    choose(drv, "style", "polyhedral", settle=0.3)
    H.wait(drv, "const p = window.__mace.viewport.scene().polyhedra(); return p.on && p.count > 0", 20)
    p = drv.execute_script("return window.__mace.viewport.scene().polyhedra()")
    cations = drv.execute_script("return window.__mace.doc().structure.symbols.filter((x) => x === 'Ti' || x === 'Pb').length")
    assert p["count"] == cations and p["meshes"] == 2, (p, cations)    # one mesh per center element
    # TiO6 octahedra (8 faces) and PbO5 square pyramids (5 faces)
    assert {(f["cn"], f["faces"]) for f in p["faces"]} == {(6, 8), (5, 5)}, p["faces"]
    assert p["shapes"] == 2 * p["meshes"]                            # each mesh and its edges
    # the drawn range's periodic images get theirs: 2x2x2 has 8 times the cell's
    drv.execute_script("window.__mace.display.setCells(window.__mace.activeDoc(), 2)")
    H.wait(drv, "const p = window.__mace.viewport.scene().polyhedra(); return p.count > arguments[0]", 20, p["count"])
    assert drv.execute_script("return window.__mace.viewport.scene().polyhedra().count") == 8 * cations
    assert not H.errors(drv)


def test_polyhedral_and_wireframe_stay_responsive_on_a_3x3x3_supercell(fresh):
    drv, root = fresh[0], fresh[1]
    need(root, "real/tio_spin.cif")
    open_doc(drv, "real/tio_spin.cif")
    assert drv.execute_script("return window.__mace.doc().structure.symbols.length") == 49
    for s in ("wireframe", "polyhedral"):
        choose(drv, "style", s, settle=0.3)
        drv.execute_script("window.__mace.display.setCells(window.__mace.activeDoc(), 3)")
        H.wait(drv, "return window.__mace.viewState().displayedAtoms > 1300", 20)
        if s == "polyhedral":
            H.wait(drv, "const p = window.__mace.viewport.scene().polyhedra(); return p.count > 400", 30)
        t = drv.execute_script("""const v = window.MACE.state.viewers['viewer-main'];
          let t0 = performance.now(); window.__mace.viewport.scene().redrawStructure(); const draw = performance.now() - t0;
          t0 = performance.now(); for (let k = 0; k < 5; k++) v.paintNow(); return {draw, paint: (performance.now() - t0) / 5};""")
        print(s, "3x3x3", t)
        assert t["draw"] < 1500 and t["paint"] < 500, (s, t)           # software GL on a shared machine
        drv.execute_script("window.__mace.display.setCells(window.__mace.activeDoc(), 1)")
        time.sleep(0.6)


def _spin_check(drv):
    """Each atom's drawn color against the sign of its moment: red for up, blue for down, white for none."""
    sp = drv.execute_script("return window.__mace.activeDoc().spins")
    assert sp["state"] == "ok", sp
    n = len(sp["values"])
    hits = 0
    for i in range(n):
        m = sp["values"][i]
        r, g, b = rgb(drv.execute_script("return window.__mace.atomStyle(arguments[0]).color", i))
        gap = 20 if abs(m) > 0.25 * sp["max"] else 0          # a small moment: a pale tint, of the right hue
        if m > 0.05 * sp["max"]:
            assert r > b + gap, (i, m, (r, g, b))
            hits += 1
        elif m < -0.05 * sp["max"]:
            assert b > r + gap, (i, m, (r, g, b))
            hits += 1
        else:
            assert min(r, g, b) > 200, (i, m, (r, g, b))
    assert hits >= 2
    legend = drv.execute_script("const l = document.querySelector('#viewer-main .vp-spin-legend'); return l && [l.textContent, l.getAttribute('aria-label')]")
    assert legend and "μB" in legend[0] and "spin down" in legend[1], legend
    return sp


def test_magnetic_colors_follow_the_sign_of_the_moments(fresh):
    drv = fresh[0]
    open_doc(drv, "spin/afm_sp.out")
    H.wait(drv, "const s = window.__mace.activeDoc().spins; return s && s.state === 'ok'", 10)
    assert not option(drv, "magnetic")["disabled"]
    choose(drv, "style", "magnetic")
    sp = _spin_check(drv)
    assert sp["values"] == [0.8, -0.8]
    note = expand(drv).find_element("css selector", ".dp-style-info").text
    assert "afm_sp.out" in note and "blue spin down" in note, note
    # another style: the legend goes
    choose(drv, "style", "default")
    assert not drv.execute_script("return document.querySelector('#viewer-main .vp-spin-legend')")
    assert not H.errors(drv)


def test_magnetic_on_a_real_spin_run(fresh):
    drv, root = fresh[0], fresh[1]
    need(root, "real/tio_spin.cif")
    open_doc(drv, "real/tio_spin.cif")
    H.wait(drv, "const s = window.__mace.activeDoc().spins; return s && s.state !== 'loading'", 15)
    choose(drv, "style", "magnetic")
    sp = _spin_check(drv)
    assert min(sp["values"]) < -0.5 and max(sp["values"]) > 0.5


def test_magnetic_is_grayed_out_without_spins(fresh):
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    H.wait(drv, "const s = window.__mace.activeDoc().spins; return s && s.state !== 'loading'", 10)
    o = option(drv, "magnetic")
    assert o["disabled"] and "No CRYSTAL output" in o["title"] and "unavailable" in o["text"], o
    na = expand(drv).find_element("css selector", ".dp-style-na").text
    assert "Magnetic" in na and "No CRYSTAL output" in na, na
    open_doc(drv, "outputs/Si_sp.out")
    H.wait(drv, "const s = window.__mace.activeDoc().spins; return s && s.state !== 'loading'", 10)
    o = option(drv, "magnetic")
    assert o["disabled"] and "ALPHA-BETA" in o["title"], o


def test_styles_in_the_optimization_view(fresh):
    """The OPT view draws the style (Space-filling: spheres, no bonds); Polyhedral is grayed out there."""
    drv = fresh[0]
    open_doc(drv, "outputs/Ag2Br3_opt_unfinished.out", kind="trajectory")
    o = option(drv, "polyhedral")
    assert o["disabled"] and "structure view only" in o["title"] and "optimization view" in o["title"], o
    keys = """const v = window.MACE.state.viewers['viewer-before'], out = new Set();
      v.getModelList().forEach((m) => m.selectedAtoms({}).forEach((a) => out.add(Object.keys(a.style).sort().join('+'))));
      return [...out]"""
    assert "sphere+stick" in drv.execute_script(keys)
    choose(drv, "style", "spacefill", settle=1.5)
    assert drv.execute_script(keys) == ["sphere"]
    assert not H.errors(drv)


def test_styles_in_the_volume_view(fresh):
    drv, root = fresh[0], fresh[1]
    need(root, "real/cubic_DENS.cube")
    open_doc(drv, "real/cubic_DENS.cube", kind="volume")
    o = option(drv, "polyhedral")
    assert o["disabled"] and "volume view" in o["title"], o
    choose(drv, "style", "grayscale", settle=1.5)
    cols = drv.execute_script("""const m = window.MACE.state.viewers['viewer-volume'].getModel(0); if (!m) return [];
      return m.selectedAtoms({}).map((a) => window.$3Dmol.getColorFromStyle(a, a.style.sphere).getHex())""")
    assert cols and all(((c >> 16) & 255) == ((c >> 8) & 255) == (c & 255) for c in cols), cols


# ---------------------------------------------------------------- lighting
@pytest.mark.parametrize("preset", LIGHTS)
def test_lighting_presets_set_their_values(fresh, preset):
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    choose(drv, "style", "spacefill")
    choose(drv, "light", preset)
    want = drv.execute_script("return window.__mace.display.PRESETS[arguments[0]]", preset)
    v = values(drv)
    assert v["renderPreset"] == preset and all(v[k] == x for k, x in want.items()), v
    assert v["stylePreset"] == "spacefill" and v["atomDraw"] == "spacefill"      # the style stays
    spec = drv.execute_script("const v = window.MACE.state.viewers['viewer-main']; v.paintNow(); return v.renderer.maceSpecular")
    assert spec == pytest.approx(want["shininess"])
    if preset == "glossy":
        corner = drv.execute_script("""const v = window.MACE.state.viewers['viewer-main']; v.paintNow();
          const c = document.createElement('canvas'), g = c.getContext('2d'), cv = v.getCanvas(); c.width = cv.width; c.height = cv.height;
          return new Promise((ok) => { const img = new Image(); img.onload = () => { g.drawImage(img, 0, 0); ok([...g.getImageData(1, 1, 1, 1).data].slice(0, 3)); }; img.src = cv.toDataURL(); });""")
        assert max(corner) <= 3, corner                                        # black (Chromium dithers by 1)
        # the highlight: the whitest pixel (its weakest channel) is whiter than the same view without shininess
        bright = """const v = window.MACE.state.viewers['viewer-main']; v.paintNow();
          const c = document.createElement('canvas'), g = c.getContext('2d'), cv = v.getCanvas(); c.width = cv.width; c.height = cv.height;
          return new Promise((ok) => { const img = new Image(); img.onload = () => { g.drawImage(img, 0, 0);
            const d = g.getImageData(0, 0, c.width, c.height).data; let n = 0; for (let k = 0; k < d.length; k += 4) n = Math.max(n, Math.min(d[k], d[k + 1], d[k + 2])); ok(n); }; img.src = cv.toDataURL(); });"""
        shiny = drv.execute_script(bright)
        drv.execute_script("window.__mace.display.set(window.__mace.activeDoc(), 'shininess', 0)")
        time.sleep(0.6)
        matte = drv.execute_script(bright)
        assert shiny > matte + 40, (shiny, matte)


def test_custom_style_and_combination(fresh):
    drv = fresh[0]
    drv.execute_script("window.__mace.settings.reset('styleCustom')")
    open_doc(drv, "structures/MgO.cif")
    choose(drv, "style", "spacefill")
    choose(drv, "light", "glossy")
    # an edit makes the style Custom and keeps it; the lighting is untouched
    el = expand(drv).find_element("css selector", "input[data-k=atomScale]")
    drv.execute_script("arguments[0].value = 0.7; arguments[0].dispatchEvent(new Event('change', {bubbles: true}))", el)
    time.sleep(0.5)
    v = values(drv)
    assert v["stylePreset"] == "custom" and v["renderPreset"] == "glossy" and v["atomDraw"] == "spacefill"
    assert drv.execute_script("return window.__mace.settings.get('styleCustom')")["atomScale"] == 0.7
    choose(drv, "style", "default")
    assert values(drv)["atomScale"] == 0.4
    choose(drv, "style", "custom")
    v = values(drv)
    assert v["atomScale"] == 0.7 and v["atomDraw"] == "spacefill" and v["renderPreset"] == "glossy"
    # Reset of the style leaves the lighting
    drv.execute_script("arguments[0].click()", expand(drv).find_element("css selector", "[data-reset-group=style]"))
    time.sleep(0.5)
    v = values(drv)
    assert v["stylePreset"] == "default" and v["renderPreset"] == "glossy"
    drv.execute_script("window.__mace.settings.reset('styleCustom')")


def test_save_as_default_and_settings_show_both(fresh):
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    choose(drv, "style", "grayscale")
    choose(drv, "light", "flat")
    for g in ("style", "render"):
        drv.execute_script("arguments[0].click()", expand(drv).find_element("css selector", f"[data-save-group={g}]"))
        time.sleep(0.3)
    try:
        assert drv.execute_script("return [window.__mace.settings.get('stylePreset'), window.__mace.settings.get('atomColor'), window.__mace.settings.get('renderPreset')]") == ["grayscale", "gray", "flat"]
        open_doc(drv, "structures/NaCl.cif")
        v = values(drv)
        assert v["stylePreset"] == "grayscale" and v["renderPreset"] == "flat"
        rows = drv.execute_async_script("""const d = arguments[0];
          import(document.querySelector('script[src*="/js/shell.js"]').src.replace('shell.js', 'lazy/settings_schema.js'))
            .then((m) => d(m.SCHEMA.filter((r) => ['stylePreset', 'renderPreset'].includes(r.key)).map((r) => [r.key, r.label, r.options.map((o) => o[0])])))""")
        assert rows == [["renderPreset", "Lighting preset", LIGHTS + ["custom"]], ["stylePreset", "Style preset", STYLES + ["custom"]]], rows
    finally:
        for k in ("stylePreset", "atomColor", "styleOutline", "renderPreset", "lightStrength", "ambient", "depthCue", "atomScale", "bondRadius",
                  "atomDraw", "atomRadii", "showPolyhedra", "outline", "outlineWidth", "ambientOcclusion", "background", "shininess", "lightDirection"):
            drv.execute_script("window.__mace.settings.reset(arguments[0])", k)


def test_old_rendering_preset_migrates_to_the_lighting(fresh):
    """A document saved before 1.1.7 (renderPreset and its values, an own atom size) reopens
    with that preset as its Lighting, and its atom size as a Custom style."""
    drv, root, base = fresh
    open_doc(drv, "structures/LiF.cif")
    drv.execute_async_script("""const d = arguments[0], M = window.__mace, a = M.activeDoc();
      M.docprefs.put('display', a, Object.assign({}, M.display.PRESETS.presentation, {renderPreset: 'presentation', atomScale: 0.7}));
      M.docprefs.flush().then(() => d(1));""")
    for k in ("shininess", "stylePreset"):        # what the old state never had
        drv.execute_script("const M = window.__mace, a = M.activeDoc(), o = M.docprefs.get('display', a); delete o[arguments[0]]; M.docprefs.put('display', a, o)", k)
    drv.execute_async_script("window.__mace.docprefs.flush().then(() => arguments[0](1))")
    drv.execute_script("localStorage.clear()")
    H.load(drv, base)
    got = drv.execute_async_script("const done = arguments[1]; window.__mace.open(arguments[0]).then((d) => done(d && d.kind))", "structures/LiF.cif")
    assert got == "structure"
    H.wait(drv, "return window.__mace.display.of(window.__mace.activeDoc()).renderPreset === 'presentation'", 10)
    H.wait(drv, "return !!document.querySelector('#inspector #dp-renderPreset')", 15)
    time.sleep(0.6)
    assert expand(drv).find_element("id", "dp-renderPreset").get_attribute("value") == "presentation"
    assert expand(drv).find_element("id", "dp-stylePreset").get_attribute("value") == "custom"
    v = values(drv)
    assert v["background"] == "black" and v["atomScale"] == 0.7 and v["atomDraw"] == "auto"
    assert drv.execute_script("return window.MACE.viewerBg()") == drv.execute_script("return window.MACE.cssVar('--viewport-black')")
    drv.execute_script("const m = window.__mace.display, a = window.__mace.activeDoc(); ['render', 'style'].forEach((g) => m.reset(a, g));")
    assert not H.errors(drv)
