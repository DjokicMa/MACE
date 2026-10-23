"""Coordination polyhedra as crystallographers draw them (1.1.6, VESTA's
polyhedral style), in a real browser on a server started through mace_cli.

A perovskite draws its TiO6 octahedron only (the Sr A site stays a ball,
with its Sr-O sticks); rutile draws one octahedron per Ti in the cell, not one
more at each corner copy; a center's sticks to its anions go, the center atom
is small and the vertex anions smaller. Every other atom and bond is drawn in
the chosen style (ball and stick, sticks, wireframe, space filling): pyrite's
S-S between the FeS6 octahedra, olivine's Mg-Si between its MgO6 and SiO4. The Inspector's Centers, Face Opacity, Edges and
Center Atom change the drawing, persist per document (gui-docprefs.json)
and as Settings defaults (gui.json), and the server, settings.js and the
Settings schema agree on them. The SVG export draws the faces too."""
import json
import time

import pytest

import guiharness as H

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")

POLY = "const p = window.__mace.viewport.scene().polyhedra(); return p && p.on && p.count > 0"
SETTINGS_JS = "import(document.querySelector('script[src*=\"/js/shell.js\"]').src.replace('shell.js', 'settings.js'))"


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    from ase.io import write
    from ase.spacegroup import crystal
    tmp = tmp_path_factory.mktemp("poly")
    root = H.examples_copy(tmp)
    (root / "poly").mkdir()
    write(str(root / "poly" / "SrTiO3.cif"), crystal(["Sr", "Ti", "O"], [(0, 0, 0), (.5, .5, .5), (.5, .5, 0)], spacegroup=221,
                                                      cellpar=[3.905] * 3 + [90] * 3))
    write(str(root / "poly" / "rutile.cif"), crystal(["Ti", "O"], [(0, 0, 0), (.3048, .3048, 0)], spacegroup=136,
                                                     cellpar=[4.594, 4.594, 2.959, 90, 90, 90]))
    write(str(root / "poly" / "pyrite.cif"), crystal(["Fe", "S"], [(0, 0, 0), (.385, .385, .385)], spacegroup=205, cellpar=[5.417] * 3 + [90] * 3))
    write(str(root / "poly" / "forsterite.cif"), crystal(["Mg", "Mg", "Si", "O", "O", "O"], [(0, 0, 0), (.2774, .25, .9915), (.0940, .25, .4262),
          (.0913, .25, .7657), (.4474, .25, .2215), (.1628, .0331, .2777)], spacegroup=62, cellpar=[10.19, 5.98, 4.75, 90, 90, 90]))
    # SrTiO3 moved by 0.49995 along a: the Ti center at x = 0.99995, just inside the cell
    from ase import Atoms
    write(str(root / "poly" / "SrTiO3_edge.cif"), Atoms("SrTiO3", cell=[3.905] * 3, pbc=True, scaled_positions=[
        (0.49995, 0, 0), (0.99995, .5, .5), (0.99995, .5, 0), (0.99995, 0, .5), (0.49995, .5, .5)]))
    xdg = tmp / "xdg"
    proc, base, _ = H.start_server(root, xdg)
    drv = H.firefox(1400, 900)
    try:
        H.load(drv, base)
        yield drv, base, root, xdg
    finally:
        drv.quit()
        proc.terminate()
        proc.wait(timeout=15)


@pytest.fixture
def fresh(app):
    drv = app[0]
    drv.execute_script("window.__mace.closeAll(); window.__mace.settings.resetAll(); window.__maceErrors.length = 0;"
                       "document.querySelectorAll('dialog[open]').forEach((d) => d.close());")
    return app


def open_poly(drv, path):
    kind = drv.execute_async_script(
        "const done = arguments[1]; window.__mace.open(arguments[0]).then((d) => done(d && d.kind))", path)
    assert kind == "structure"
    H.wait(drv, "return window.__mace.viewport && window.__mace.viewport.displayed() > 0", 20)
    drv.execute_script("const m = window.__mace.display, a = window.__mace.activeDoc(); m.reset(a, 'style');")
    time.sleep(0.3)
    drv.execute_script("window.__mace.run('view.polyhedra')")
    H.wait(drv, POLY, 30)
    time.sleep(0.3)
    return drv.execute_script("return window.__mace.viewport.scene().polyhedra()")


def drawn(drv):
    """[(symbol, drawn radius, image)] of every drawn atom, and the drawn bonds as symbol pairs."""
    return drv.execute_script("""const sc = window.__mace.viewport.scene(), d = sc.display, s = window.__mace.doc().structure;
      const atoms = Array.from(d.base, (b, k) => [s.symbols[b], sc.drawnRadii[k], d.image[k]]);
      const p = sc.pairs || [], bonds = [];
      for (let q = 0; q < p.length; q += 2) bonds.push([s.symbols[d.base[p[q]]], s.symbols[d.base[p[q + 1]]]].sort().join('-'));
      return {atoms, bonds};""")


def poly_section(drv):
    sec = drv.find_element("css selector", "#inspector .insp-section[data-key=render]")
    if "collapsed" in sec.get_attribute("class"):
        drv.execute_script("arguments[0].scrollIntoView({block: 'center'})", sec)
        sec.find_element("css selector", ".insp-head").click()
        time.sleep(0.2)
    H.wait(drv, "return !!document.querySelector('#inspector [data-poly-center]')", 20)
    return drv.find_element("css selector", "#inspector .dp-poly")


def test_settings_agree_and_round_trip(fresh):
    from mace.gui import settings as S
    drv, base, root, xdg = fresh
    for k, v in (("polyCenters", "auto"), ("polyCenters", "Ti,Si"), ("polyCenters", ""), ("polyOpacity", 0.5),
                 ("polyOpacity", 1), ("polyEdges", False), ("polyCenterAtom", "hidden"), ("polyCenterAtom", "full")):
        assert S.check(k, v) is None, (k, v)
    for k, v in (("polyCenters", "ti"), ("polyCenters", "Ti;Si"), ("polyCenters", 3), ("polyOpacity", 0), ("polyOpacity", 1.5),
                 ("polyOpacity", True), ("polyEdges", "yes"), ("polyCenterAtom", "large")):
        assert S.check(k, v), (k, v)
    # settings.js: the same defaults and the same checks (a bad stored value is dropped)
    js = drv.execute_async_script("const d = arguments[0]; " + SETTINGS_JS + """.then((m) => d({
        defaults: Object.fromEntries(['polyCenters', 'polyOpacity', 'polyEdges', 'polyCenterAtom'].map((k) => [k, m.DEFAULTS[k]])),
        bad: m.migrate({polyCenters: 'ti', polyOpacity: 5, polyEdges: 'no', polyCenterAtom: 'large'}),
        good: m.migrate({polyCenters: 'Ti,Sr', polyOpacity: 0.3, polyEdges: false, polyCenterAtom: 'hidden'})}))""")
    assert js["defaults"] == {"polyCenters": "auto", "polyOpacity": 0.5, "polyEdges": True, "polyCenterAtom": "small"}
    assert js["bad"] == {} and js["good"] == {"polyCenters": "Ti,Sr", "polyOpacity": 0.3, "polyEdges": False, "polyCenterAtom": "hidden"}
    schema = drv.execute_async_script("const d = arguments[0]; import(document.querySelector('script[src*=\"/js/shell.js\"]').src"
                                      ".replace('shell.js', 'lazy/settings_schema.js')).then((m) => d(m.SCHEMA.map((r) => r.key)))")
    assert {"polyCenters", "polyOpacity", "polyEdges", "polyCenterAtom"} <= set(schema)
    # Settings defaults reach gui.json, and only valid ones are kept
    drv.execute_script("const s = window.__mace.settings; s.set('polyOpacity', 0.7); s.set('polyEdges', false); s.set('polyCenterAtom', 'full');")
    drv.execute_async_script("const d = arguments[0]; window.__mace.settings.flush().then(d)")
    saved = json.loads((xdg / "mace" / "gui.json").read_text())["settings"]
    assert saved["polyOpacity"] == 0.7 and saved["polyEdges"] is False and saved["polyCenterAtom"] == "full", saved
    # a document's own values reach gui-docprefs.json (the server checks them as settings)
    open_poly(drv, "poly/rutile.cif")
    drv.execute_script("window.__mace.display.set(window.__mace.activeDoc(), {polyOpacity: 0.3, polyCenters: ''})")
    time.sleep(1.2)                       # the 500 ms debounce, then the POST
    data = json.loads((xdg / "mace" / "gui-docprefs.json").read_text())
    doc = [v for docs in data["kinds"]["display"].values() for p, v in docs.items() if p.endswith("poly/rutile.cif")]
    assert doc and doc[0]["values"]["polyOpacity"] == 0.3 and doc[0]["values"]["polyCenters"] == "", doc
    H.wait(drv, "return window.__mace.viewport.scene().polyhedra().count === 0", 20)
    drv.execute_script("window.__mace.display.reset(window.__mace.activeDoc(), 'style')")
    assert not H.errors(drv), H.errors(drv)


def test_perovskite_draws_bo6_only(fresh):
    drv = fresh[0]
    p = open_poly(drv, "poly/SrTiO3.cif")
    # one TiO6 octahedron: no SrO12 (the A site shares faces with it), no anion-centered polyhedra
    assert p["centers"] == ["Ti"] and p["candidates"] == ["Ti", "Sr"] and p["spacers"] == ["Sr"], p
    assert p["count"] == 1 and p["meshes"] == 1 and [(f["cn"], f["faces"]) for f in p["faces"]] == [(6, 8)], p
    d = drawn(drv)
    # the polyhedron replaces Ti's sticks; the A site keeps its own (1.1.6: they went too)
    assert "O-Ti" not in d["bonds"] and d["bonds"].count("O-Sr") == 24, d["bonds"]
    # all six corners are the drawn O (3.905 Å, half a hundredth, missed three of them and drew a ball over each)
    assert sum(1 for x in d["atoms"] if x[0] == "O" and x[1] > 0) == 6 and p["outside"] == 0, p
    r = {sym: max(x[1] for x in d["atoms"] if x[0] == sym) for sym in ("Sr", "Ti", "O")}
    full = drv.execute_script("return Object.fromEntries(['Sr', 'Ti', 'O'].map((e) => [e, window.MACE.sphereScale(true) * $3Dmol.GLModel.vdwRadii[e]]))")
    # a small center (0.45), smaller vertex anions (0.6), the A site as it is
    assert abs(r["Ti"] / full["Ti"] - 0.45) < 0.01 and abs(r["O"] / full["O"] - 0.6) < 0.01 and abs(r["Sr"] / full["Sr"] - 1) < 0.01, (r, full)
    # the A site's note in the Inspector
    poly_section(drv)
    note = drv.find_element("css selector", "#inspector .dp-poly-note")
    assert note.is_displayed() and note.text.startswith("Sr left out"), note.text
    # rutile: one octahedron per Ti in the cell, none at the corner copies (they are not drawn either)
    p = open_poly(drv, "poly/rutile.cif")
    assert p["centers"] == ["Ti"] and p["count"] == 2, p
    d = drawn(drv)
    assert sum(1 for s, rad, img in d["atoms"] if s == "Ti" and rad > 0) == 2
    assert sum(1 for s, rad, img in d["atoms"] if s == "Ti" and img) == 7 and "O-Ti" not in d["bonds"]
    assert p["outside"] > 0                                            # the corners outside the cell have their O
    assert not H.errors(drv), H.errors(drv)


def test_inspector_controls_change_the_polyhedra(fresh):
    drv = fresh[0]
    open_poly(drv, "poly/SrTiO3.cif")
    sec = poly_section(drv)
    # Centers: Sr checked draws its cuboctahedron too; the document stores its elements
    sr = sec.find_element("css selector", "[data-poly-center=Sr]")
    assert not sr.is_selected() and sec.find_element("css selector", "[data-poly-center=Ti]").is_selected()
    drv.execute_script("arguments[0].scrollIntoView({block: 'center'})", sr)
    sr.click()
    H.wait(drv, "return window.__mace.viewport.scene().polyhedra().centers.length === 2", 20)
    p = drv.execute_script("return window.__mace.viewport.scene().polyhedra()")
    assert sorted((f["cn"], f["faces"]) for f in p["faces"]) == [(6, 8), (12, 14)], p["faces"]
    assert drv.execute_script("return window.__mace.display.of(window.__mace.activeDoc()).polyCenters") == "Ti,Sr"
    # unchecked again it is the default ("auto"), not a stored list
    sec = poly_section(drv)
    drv.execute_script("arguments[0].click()", sec.find_element("css selector", "[data-poly-center=Sr]"))
    H.wait(drv, "return window.__mace.viewport.scene().polyhedra().centers.length === 1", 20)
    assert drv.execute_script("return window.__mace.display.overrides(window.__mace.activeDoc()).polyCenters") is None
    # Face Opacity
    sec = poly_section(drv)
    drv.execute_script("const s = arguments[0].querySelector('[data-k=polyOpacity]'); s.value = 0.8; s.dispatchEvent(new Event('change', {bubbles: true}))", sec)
    H.wait(drv, "return window.__mace.viewport.scene().polyhedra().opacity === 0.8", 20)
    # Edges off: the face mesh alone
    sec = poly_section(drv)
    drv.execute_script("arguments[0].click()", sec.find_element("css selector", "[data-k=polyEdges]"))
    H.wait(drv, "const p = window.__mace.viewport.scene().polyhedra(); return p.edges === false && p.shapes === p.meshes", 20)
    # Center Atom hidden: Ti is not drawn (and cannot be picked)
    sec = poly_section(drv)
    drv.execute_script("const s = arguments[0].querySelector('[data-k=polyCenterAtom]'); s.value = 'hidden'; s.dispatchEvent(new Event('change', {bubbles: true}))", sec)
    H.wait(drv, """const sc = window.__mace.viewport.scene(), s = window.__mace.doc().structure;
      return Array.from(sc.display.base).every((b, k) => s.symbols[b] !== 'Ti' || sc.drawnRadii[k] === 0)""", 20)
    # none of these made the Style Custom (the Polyhedra checkbox did, as before)
    o = drv.execute_script("return window.__mace.display.overrides(window.__mace.activeDoc())")
    assert o["polyOpacity"] == 0.8 and o["polyEdges"] is False and o["polyCenterAtom"] == "hidden", o
    # the SVG export draws the faces toward the viewer, and no hidden center
    drv.execute_script("window.__svg = null; window.__maceExportHook = (t, kind) => { if (kind === 'svg') window.__svg = t; };")
    r = drv.execute_async_script("const d = arguments[0]; window.__mace.exportSvg({}).then((n) => d({n, text: window.__svg}), (e) => d({error: String(e.message || e)}))")
    assert "error" not in r, r
    assert 1 <= r["text"].count("<polygon") <= 8 and 'fill-opacity="0.8"' in r["text"]
    assert "<title>Ti2</title>" not in r["text"]
    drv.execute_script("window.__mace.display.reset(window.__mace.activeDoc(), 'style')")
    assert not H.errors(drv), H.errors(drv)


def test_an_edit_in_the_middle_of_the_atoms_draws_them_again(fresh):
    """The polyhedra are cached by every drawn position and element (the key held only the first and the last
    coordinate): Zr in place of rutile's second Ti, neither the first nor the last atom drawn, is a center now."""
    drv = fresh[0]
    p = open_poly(drv, "poly/rutile.cif")
    assert p["centers"] == ["Ti"] and p["count"] == 2, p
    drv.execute_script("window.__mace.run('build.symModeP1')")
    drv.execute_script("const x = window.__mace.editor; x.E.api.edit(x.doc(), 'replace', {indices: [1], element: 'Zr'})")
    H.wait(drv, "return window.__mace.editor.doc().structure.symbols[1] === 'Zr'", 30)
    H.wait(drv, "const p = window.__mace.viewport.scene().polyhedra(); return p && p.candidates && p.candidates.includes('Zr')", 30)
    assert not H.errors(drv), H.errors(drv)


def test_a_center_just_below_one_keeps_its_polyhedron_without_boundary_atoms(fresh):
    """Without boundary atoms a center at x = 0.99995 is drawn once, there: its polyhedron is drawn with it
    (the range test dropped centers from 1 - 1e-4 to 1)."""
    drv = fresh[0]
    drv.execute_script("window.__mace.settings.set('showBoundary', false)")
    p = open_poly(drv, "poly/SrTiO3_edge.cif")
    assert drv.execute_script("return window.__mace.viewport.displayed()") == 5
    assert p["centers"] == ["Ti"] and p["count"] == 1, p
    drv.execute_script("window.__mace.settings.set('showBoundary', true)")
    H.wait(drv, "return window.__mace.viewport.displayed() > 5", 20)
    H.wait(drv, "const p = window.__mace.viewport.scene().polyhedra(); return p && p.count === 1", 20)
    assert not H.errors(drv), H.errors(drv)


def set_style(drv, style):
    drv.execute_script("const el = document.getElementById('opt-style'); el.value = arguments[0]; el.dispatchEvent(new Event('change'))", style)
    time.sleep(0.6)


def drawn_styles(drv):
    """Per drawn atom: symbol, drawn radius and the 3Dmol style keys; the bonds as symbol pairs; the polyhedra's
    corner balls outside the range (their scale, by style)."""
    return drv.execute_script("""const sc = window.__mace.viewport.scene(), d = sc.display, s = window.__mace.doc().structure;
      const m = sc.layers.structure, by = new Map(m.selectedAtoms({}).map((a) => [a.index, a.style || {}]));
      const atoms = Array.from(d.base, (b, k) => { const st = by.get(k) || {};
        return [s.symbols[b], sc.drawnRadii[k], Object.keys(st).sort().join('+'), !!(st.stick && st.stick.showNonBonded)]; });
      const p = sc.pairs || [], bonds = [];
      for (let q = 0; q < p.length; q += 2) bonds.push([s.symbols[d.base[p[q]]], s.symbols[d.base[p[q + 1]]]].sort().join('-'));
      const g = sc.polyGeometry();
      return {atoms, bonds, extra: g ? g.extra.length : 0, ballScale: g ? (g.ballScale === undefined ? null : g.ballScale) : null,
        sphere: window.MACE.sphereScale(true)};""")


@pytest.mark.parametrize("name,kept,gone", [
    ("SrTiO3", {"O-Sr": 24}, ("O-Ti",)),                  # the A site's bonds
    ("pyrite", {"S-S": 1}, ("Fe-S",)),                    # a bond between the vertices of two octahedra
    ("forsterite", {"Mg-Si": 7}, ("Mg-O", "O-Si")),       # the contacts between neighboring MgO6 and SiO4
])
def test_the_rest_of_the_structure_keeps_the_chosen_style(fresh, name, kept, gone):
    """Polyhedra on: only a center's sticks to its anions go. Every other bond is drawn, in ball and stick,
    sticks and wireframe alike; every atom but a left-out one (a center's boundary copy, a hidden center) is
    drawn in that style, and a left-out atom is drawn in none (it stayed a stick joint in Sticks). In Sticks
    an atom whose bonds the polyhedron took (a corner anion) stays a ball of the stick radius (it vanished)."""
    drv = fresh[0]
    open_poly(drv, f"poly/{name}.cif")
    for style in ("ballstick", "stick", "wireframe", "spacefill"):
        set_style(drv, style)
        H.wait(drv, POLY, 30)
        d = drawn_styles(drv)
        if style == "spacefill":
            assert d["bonds"] == [], (style, d["bonds"])
        else:
            for pair, n in kept.items():
                assert d["bonds"].count(pair) == n, (style, pair, d["bonds"])
            assert not [b for b in d["bonds"] if b in gone], (style, d["bonds"])
        want = {"ballstick": "sphere+stick", "stick": "stick", "wireframe": "line+sphere", "spacefill": "sphere"}[style]
        shown = [a for a in d["atoms"] if a[1] > 0]
        assert shown and all(a[2] == want for a in shown), (style, {a[2] for a in shown})
        assert all(a[2] == "" for a in d["atoms"] if a[1] == 0), (style, [a for a in d["atoms"] if a[1] == 0 and a[2]])
        if style == "stick":
            assert all(a[3] for a in shown), "a bondless atom vanished in Sticks"
        # the corners outside the range: balls as large as the drawn vertex anions in this style
        if d["extra"]:
            ball = {"ballstick": d["sphere"] * 0.6, "wireframe": 0.06, "spacefill": 0.57, "stick": None}[style]
            assert (d["ballScale"] is None) if ball is None else abs(d["ballScale"] - ball) < 1e-3, (style, d["ballScale"])
    set_style(drv, "ballstick")
    assert not H.errors(drv), H.errors(drv)
