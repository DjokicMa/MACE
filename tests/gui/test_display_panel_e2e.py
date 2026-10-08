"""The Inspector's "Range & boundary" and "Rendering" groups (polish A3, A5,
A6), with real clicks in a fresh browser profile on a server started
through mace_cli, on real structures from test/ (copied under neutral names).

Per document: one document's range, boundary atoms or look never changes
another's; Settings hold the defaults, which documents without their own
value follow live; the values come back after a reload, a close and reopen,
and on a new local port in a fresh browser (the server's gui-docprefs.json).
View only: nothing reaches the served folder, a CIF or a deck."""
import json
import time

import pytest

import guiharness as H
import realdata
from test_geom_range import ase_bonds, bfs_molecules, oracle_range

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")

from selenium.webdriver.common.action_chains import ActionChains  # noqa: E402
from selenium.webdriver.common.keys import Keys  # noqa: E402


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("dpanel")
    root = H.examples_copy(tmp)
    if not H.remote():
        realdata.copy(root / "real", "agte.cif", "tipbo3.cif", "molcrystal.cif")
    xdg = tmp / "xdg"
    proc, base, _ = H.start_server(root, xdg)
    drv = H.firefox(1400, 900)
    st = {"proc": proc, "base": base}
    try:
        H.load(drv, base)
        yield drv, st, root, xdg, tmp
    finally:
        drv.quit()
        st["proc"].terminate()
        st["proc"].wait(timeout=15)


@pytest.fixture
def fresh(app):
    drv = app[0]
    drv.execute_script("window.__mace.closeAll(); document.querySelectorAll('dialog[open]').forEach((d) => d.close());"
                       "document.querySelectorAll('#toasts > *').forEach((t) => t.remove());")
    return app


def type_field(drv, sel, text):
    """Click a Range field, select all and type, then Tab; the Inspector can be drawn
    again just before (a busy machine): the field is looked up again then."""
    from selenium.common.exceptions import StaleElementReferenceException
    for attempt in range(5):
        try:
            el = expand(drv, "range").find_element("css selector", sel)
            click(drv, el)
            el.send_keys(Keys.CONTROL, "a")
            el.send_keys(text, Keys.TAB)
            return
        except StaleElementReferenceException:
            if attempt == 4:
                raise
            time.sleep(0.3)


def click(drv, el):
    """A real click, after scrolling the Inspector to the control (as a user would)."""
    drv.execute_script("arguments[0].scrollIntoView({block: 'center'})", el)
    time.sleep(0.05)
    ActionChains(drv).move_to_element(el).pause(0.05).click().perform()


def open_doc(drv, path):
    kind = drv.execute_async_script(
        "const done = arguments[1]; window.__mace.open(arguments[0]).then((d) => done(d && d.kind))", path)
    assert kind == "structure", drv.execute_script("return [...document.querySelectorAll('#toasts > *')].map((t) => t.textContent)")
    H.wait(drv, "return !!document.querySelector('#inspector .insp-section[data-key=range] .dp')", 10)
    time.sleep(0.6)


def clean(drv, *paths):
    """Start a test with these documents at their defaults (earlier tests changed them on this server)."""
    for p in paths:
        open_doc(drv, p)
        drv.execute_script("const m = window.__mace.display, a = window.__mace.activeDoc(); m.reset(a, 'range'); m.reset(a, 'render');")
    drv.execute_script("window.__mace.closeAll()")
    time.sleep(0.3)


def pick(drv, sel_el, value):
    from selenium.webdriver.support.ui import Select
    drv.execute_script("arguments[0].scrollIntoView({block: 'center'})", sel_el)
    Select(sel_el).select_by_value(value)
    time.sleep(0.6)


def section(drv, key):
    return drv.find_element("css selector", f"#inspector .insp-section[data-key={key}]")


def expand(drv, key):
    sec = section(drv, key)
    if "collapsed" in sec.get_attribute("class"):
        click(drv, sec.find_element("css selector", ".insp-head"))
        time.sleep(0.2)
    return section(drv, key)


def drawn(drv):
    return drv.execute_script("return window.__mace.viewState().displayedAtoms")


def drawn_set(drv):
    return set(map(tuple, drv.execute_script("""const d = window.__mace.viewport.scene().display;
        return Array.from(d.base, (b, k) => [b, d.shift[3 * k], d.shift[3 * k + 1], d.shift[3 * k + 2]]);""")))


def payload(drv, path):
    return drv.execute_async_script(
        "const d = arguments[1]; window.MACE.api('structure?path=' + encodeURIComponent(arguments[0])).then(d)", path)


def wait_redraw(drv, before, timeout=2):
    H.wait(drv, "return window.__mace.viewState().displayedAtoms !== arguments[0]", timeout, before)
    time.sleep(0.2)


def quick(drv, name):
    click(drv, expand(drv, "range").find_element("css selector", f"[data-quick='{name}']"))
    time.sleep(0.5)


def test_cell_plus_10_and_typed_range(fresh):
    drv = fresh[0]
    clean(drv, "real/agte.cif")
    open_doc(drv, "real/agte.cif")
    s = payload(drv, "real/agte.cif")
    n0 = drawn(drv)
    quick(drv, "cell10")
    wait_redraw(drv, n0)
    rng = {"lo": [-0.1] * 3, "hi": [1.1] * 3}
    assert drawn_set(drv) == oracle_range(s, rng, True)
    assert drv.find_element("id", "tb-range-label").text == "−0.1–1.1"
    # type 0 / 2 in the b row, Tab: redrawn within 500 ms
    n1 = drawn(drv)
    type_field(drv, "[data-lo='1']", "0")
    type_field(drv, "[data-hi='1']", "2")
    t0 = time.time()
    H.wait(drv, "return window.__mace.viewState().displayedAtoms !== arguments[0]", 2, n1)
    assert time.time() - t0 < 0.5 + 0.3
    time.sleep(0.4)
    assert drawn_set(drv) == oracle_range(s, {"lo": [-0.1, 0, -0.1], "hi": [1.1, 2, 1.1]}, True)
    assert not H.errors(drv), H.errors(drv)


def test_whole_molecules_and_extended_networks(fresh):
    drv = fresh[0]
    clean(drv, "real/molcrystal.cif", "real/tipbo3.cif")
    open_doc(drv, "real/molcrystal.cif")
    s = payload(drv, "real/molcrystal.cif")
    pick(drv, expand(drv, "range").find_element("id", "dp-bondedOutside"), "molecules")
    want = bfs_molecules(oracle_range(s, [1, 1, 1], True), ase_bonds(s))
    assert drawn_set(drv) == want
    # and nothing is clipped (polish A2) with the grown set
    from test_viewport_clip_e2e import check
    assert check(drv, "viewer-main")["clipped"] == 0
    open_doc(drv, "real/tipbo3.cif")
    t = payload(drv, "real/tipbo3.cif")
    pick(drv, expand(drv, "range").find_element("id", "dp-bondedOutside"), "molecules")
    note = expand(drv, "range").find_element("css selector", ".dp-bonded-note").text
    assert "no discrete molecules" in note or "extended network" in note, note
    assert drawn_set(drv) == oracle_range(t, [1, 1, 1], True)


def test_boundary_is_one_control_per_document(fresh):
    drv, st = fresh[0], fresh[1]
    clean(drv, "real/agte.cif", "real/tipbo3.cif")
    open_doc(drv, "real/agte.cif")
    assert drv.execute_script("return document.querySelectorAll('#opt-boundary').length") == 1
    assert drv.execute_script("return !!document.querySelector('#inspector .insp-section[data-key=range] #opt-boundary')")
    assert not drv.execute_script("return !!document.querySelector('#inspector .insp-section[data-key=display] #opt-boundary')")
    default = drv.execute_script("return window.__mace.settings.get('showBoundary')")
    n0 = drawn(drv)
    click(drv, drv.find_element("id", "opt-boundary"))
    wait_redraw(drv, n0)
    assert drv.execute_script("return window.__mace.settings.get('showBoundary')") == default
    open_doc(drv, "real/tipbo3.cif")
    assert drv.execute_script("return window.__mace.viewState().boundary") == default
    assert drv.execute_script("return document.getElementById('opt-boundary').checked") == default


def test_per_document_range_reload_and_reopen(fresh):
    drv, st = fresh[0], fresh[1]
    clean(drv, "real/agte.cif", "real/tipbo3.cif")
    open_doc(drv, "real/agte.cif")
    open_doc(drv, "real/tipbo3.cif")
    quick(drv, "3")
    assert drv.find_element("id", "tb-range-label").text == "3×3×3"
    # the override dot shows on the changed control only
    dots = drv.execute_script("return [...document.querySelectorAll('#inspector [data-key=range] .ovr-dot')].map((d) => d.dataset.unset)")
    assert dots == ["range"], dots
    drv.execute_script("window.__mace.activate(0)")
    time.sleep(0.6)
    assert drv.find_element("id", "tb-range-label").text == "1×1×1"
    assert not drv.execute_script("return document.querySelectorAll('#inspector [data-key=range] .ovr-dot').length")
    # a reload, then a close and reopen: tipbo3 is still at 3
    drv.execute_async_script("const d = arguments[0]; window.__mace.settings.flush().then(() => d(1))")
    time.sleep(0.8)
    H.load(drv, st["base"])
    open_doc(drv, "real/tipbo3.cif")
    H.wait(drv, "return document.getElementById('tb-range-label').textContent === '3×3×3'", 5)
    drv.execute_script("window.__mace.closeAll()")
    open_doc(drv, "real/tipbo3.cif")
    H.wait(drv, "return document.getElementById('tb-range-label').textContent === '3×3×3'", 5)
    # Reset brings it back to the default
    click(drv, expand(drv, "range").find_element("css selector", "[data-reset-group=range]"))
    H.wait(drv, "return document.getElementById('tb-range-label').textContent === '1×1×1'", 5)


def test_new_local_port_and_fresh_browser(fresh):
    """The overrides live on the server: a server on another port (same folder
    and XDG dir) and a browser with empty localStorage get them back."""
    H.local_only("restarts the server (the HPCC check re-tunnels on another port by hand)")
    drv, st, root, xdg, tmp = fresh
    open_doc(drv, "real/agte.cif")
    quick(drv, "cell10")
    time.sleep(1.0)                      # the 500 ms debounce, then the POST
    data = json.loads((xdg / "mace" / "gui-docprefs.json").read_text())
    assert any("real/agte.cif" in docs for docs in data["kinds"]["display"].values())
    st["proc"].terminate()
    st["proc"].wait(timeout=15)
    proc, base, _ = H.start_server(root, xdg, port=H.free_port())
    st["proc"], st["base"] = proc, base
    drv2 = H.firefox(1400, 900)
    try:
        H.load(drv2, base)
        open_doc(drv2, "real/agte.cif")
        H.wait(drv2, "return document.getElementById('tb-range-label').textContent === '−0.1–1.1'", 5)
    finally:
        drv2.quit()
    H.load(drv, base)


def test_settings_default_applies_live_unless_overridden(fresh):
    H.local_only("reads gui.json")
    drv, st, root, xdg, _ = fresh
    clean(drv, "real/agte.cif", "real/tipbo3.cif")
    open_doc(drv, "real/agte.cif")
    open_doc(drv, "real/tipbo3.cif")
    # tipbo3: its own atom size
    sec = expand(drv, "range")
    slider = sec.find_element("css selector", "input[data-k=atomScale]")
    drv.execute_script("arguments[0].value = 0.8; arguments[0].dispatchEvent(new Event('change', {bubbles: true}))", slider)
    time.sleep(0.5)
    scale = lambda: drv.execute_script("return window.__mace.viewState().sphereScale")  # noqa: E731
    own = scale()
    drv.execute_script("window.__mace.settings.set('atomScale', 0.3)")
    time.sleep(0.6)
    assert scale() == own                                  # its own value stays
    drv.execute_script("window.__mace.activate(0)")
    time.sleep(0.6)
    assert scale() == pytest.approx(0.3 * 0.7, abs=1e-6)  # agte follows the default
    drv.execute_script("window.__mace.activate(1)")
    time.sleep(0.6)
    click(drv, expand(drv, "range").find_element("css selector", "[data-reset-group=range]"))
    time.sleep(0.6)
    assert scale() == pytest.approx(0.3 * 0.7, abs=1e-6)  # Reset: follows again
    # Save as Default: the document's range becomes the default, in gui.json, no warning
    quick(drv, "cell10")
    click(drv, expand(drv, "range").find_element("css selector", "[data-save-group=range]"))
    drv.execute_async_script("const d = arguments[0]; window.__mace.settings.flush().then(() => d(1))")
    stored = json.loads((xdg / "mace" / "gui.json").read_text())["settings"]
    assert stored["displayLo"] == [-0.1, -0.1, -0.1] and stored["displayHi"] == [1.1, 1.1, 1.1]
    toasts = drv.execute_script("return [...document.querySelectorAll('#toasts > *')].map((t) => [t.className, t.textContent])")
    assert any("Saved as the default" in t for _, t in toasts) and not any("warning" in c for c, _ in toasts), toasts
    drv.execute_script("window.__mace.settings.resetAll()")
    drv.execute_async_script("const d = arguments[0]; window.__mace.settings.flush().then(() => d(1))")


def test_view_only(fresh):
    """Range, boundary and molecules changes write nothing into the folder,
    and the exported CIF of the document is the same bytes."""
    H.local_only("hashes the served folder")
    drv, st, root = fresh[0], fresh[1], fresh[2]
    clean(drv, "real/molcrystal.cif")
    before = H.dir_md5(root)
    open_doc(drv, "real/molcrystal.cif")
    cif = lambda: drv.execute_async_script("""const d = arguments[0];
        window.__mace.sourceOf ? 0 : 0;
        window.MACE.api('structure/export', {body: {source: {path: 'real/molcrystal.cif'}, format: 'CIF', cif_mode: 'p1'}, blob: true})
          .then((r) => r.blob.text()).then(d, (e) => d('ERR ' + e.message))""")  # noqa: E731
    c0 = cif()
    quick(drv, "cell10")
    click(drv, drv.find_element("id", "opt-boundary"))
    pick(drv, expand(drv, "range").find_element("id", "dp-bondedOutside"), "molecules")
    assert cif() == c0 and not c0.startswith("ERR")
    assert H.dir_md5(root) == before


def test_polyhedra_shortcut_is_per_document(fresh):
    drv = fresh[0]
    clean(drv, "real/agte.cif", "real/tipbo3.cif")
    open_doc(drv, "real/agte.cif")
    open_doc(drv, "real/tipbo3.cif")
    drv.execute_script("document.querySelector('#viewer-main').closest('.viewer-wrap').focus()")
    ActionChains(drv).key_down(Keys.SHIFT).send_keys("p").key_up(Keys.SHIFT).perform()
    time.sleep(0.5)
    assert drv.execute_script("return window.__mace.ctx && window.__mace.settings.get('showPolyhedra')") is False
    on = drv.execute_script("return window.__mace.viewport.scene().polyhedra().on")
    drv.execute_script("window.__mace.activate(0)")
    time.sleep(0.6)
    assert drv.execute_script("return window.__mace.viewport.scene().polyhedra().on") != on


# ---------------------------------------------------------------- Rendering (A6)
PIX = """const v = window.MACE.state.viewers['viewer-main']; v.paintNow();
  const cv = v.getCanvas(), url = cv.toDataURL('image/png');
  return new Promise((ok) => { const img = new Image(); img.onload = () => {
    const c = document.createElement('canvas'); c.width = cv.width; c.height = cv.height;
    const g = c.getContext('2d'); g.drawImage(img, 0, 0);
    const d = g.getImageData(0, 0, c.width, c.height).data; let h = 0;
    for (let k = 0; k < d.length; k += 4 * 7) h = (h * 31 + d[k] + 3 * d[k + 1] + 7 * d[k + 2]) >>> 0;
    ok({hash: h, corner: [d[0], d[1], d[2]], dark: (() => { let n = 0; for (let k = 0; k < d.length; k += 16) if (d[k] + d[k + 1] + d[k + 2] < 200) n++; return n; })()}); };
    img.src = url; });"""


def pix(drv):
    return drv.execute_script(PIX)


def test_each_rendering_control_changes_the_picture_and_reset_restores(fresh):
    drv = fresh[0]
    clean(drv, "real/tipbo3.cif")
    open_doc(drv, "real/tipbo3.cif")
    sec = expand(drv, "render")
    base = pix(drv)["hash"]
    sliders = {"depthCue": 0.8, "lightStrength": 1.5, "ambient": 0.4, "ambientOcclusion": 0.8}
    for k, v in sliders.items():
        el = expand(drv, "render").find_element("css selector", f"input[data-k={k}]")
        drv.execute_script("arguments[0].value = arguments[1]; arguments[0].dispatchEvent(new Event('change', {bubbles: true}))", el, v)
        time.sleep(0.5)
        h = pix(drv)["hash"]
        assert h != base, k
        click(drv, expand(drv, "render").find_element("css selector", "[data-reset-group=render]"))
        time.sleep(0.5)
        assert pix(drv)["hash"] == base, f"Reset after {k}"
    for d in ("upper-left", "top"):
        click(drv, expand(drv, "render").find_element("css selector", f"[data-dir='{d}']"))
        time.sleep(0.5)
        assert pix(drv)["hash"] != base, d
        assert drv.execute_script("return window.__mace.settings.get('ambient')") == 0       # the default is untouched
        click(drv, expand(drv, "render").find_element("css selector", "[data-reset-group=render]"))
        time.sleep(0.5)
    # the preset becomes Custom after an edit
    el = expand(drv, "render").find_element("css selector", "input[data-k=lightStrength]")
    drv.execute_script("arguments[0].value = 1.2; arguments[0].dispatchEvent(new Event('change', {bubbles: true}))", el)
    time.sleep(0.4)
    assert expand(drv, "render").find_element("id", "dp-renderPreset").get_attribute("value") == "custom"
    click(drv, expand(drv, "render").find_element("css selector", "[data-reset-group=render]"))


@pytest.mark.parametrize("preset", ["flat", "publication", "presentation"])
def test_presets_set_their_values(fresh, preset):
    drv = fresh[0]
    clean(drv, "real/agte.cif")
    open_doc(drv, "real/agte.cif")
    pick(drv, expand(drv, "render").find_element("id", "dp-renderPreset"), preset)
    want = drv.execute_script("return window.__mace.display.PRESETS[arguments[0]]", preset)
    got = drv.execute_script("return window.__mace.display.of(window.__mace.activeDoc())")
    for k, v in want.items():
        assert got[k] == v, (k, got[k], v)
    assert got["renderPreset"] == preset
    if preset == "publication":
        p = pix(drv)
        assert p["corner"] == [255, 255, 255], p          # white background (the PNG export reads this canvas)
        # the outline's dark edge ring: more dark pixels than the same look without it
        drv.execute_script("window.__mace.display.set(window.__mace.activeDoc(), 'outline', false)")
        time.sleep(0.5)
        plain = pix(drv)
        drv.execute_script("window.__mace.display.set(window.__mace.activeDoc(), 'outline', true)")
        time.sleep(0.5)
        assert p["dark"] > plain["dark"] + 50, (p, plain)
        # SVG export stays flat: the atoms' fills are the element colors
        drv.execute_script("window.__svg = null; window.__maceExportHook = (t, kind) => { if (kind === 'svg') window.__svg = t; };")
        r = drv.execute_async_script("const d = arguments[0]; window.__mace.exportSvg({}).then((n) => d({n, text: window.__svg}), (e) => d({error: String(e.message || e)}))")
        assert "error" not in r, r
        import re
        fills = set(re.findall(r'<circle[^>]*fill="(#[0-9a-fA-F]{6})"', r["text"]))
        colors = drv.execute_script("return ['Ag', 'Te'].map((e) => window.MACE.elementColor(e).toLowerCase())")
        assert fills and {f.lower() for f in fills} <= set(colors), (fills, colors)
    click(drv, expand(drv, "render").find_element("css selector", "[data-reset-group=render]"))


def test_custom_range_dialog_takes_from_and_to(fresh):
    """Toolbar › Display range › Custom…: from/to per axis with the Inspector's
    rules (−5..10, the end at least 0.05 above the start), for this document."""
    drv = fresh[0]
    clean(drv, "real/agte.cif")
    open_doc(drv, "real/agte.cif")
    s = payload(drv, "real/agte.cif")
    click(drv, drv.find_element("id", "tb-range"))
    H.wait(drv, "return !!document.querySelector('.menu [data-cmd-id=\"view.rangeCustom\"]')", 5)
    click(drv, drv.find_element("css selector", '.menu [data-cmd-id="view.rangeCustom"]'))
    H.wait(drv, "const d = document.getElementById('range-custom'); return !!(d && d.open)", 5)
    dlg = drv.find_element("id", "range-custom")
    # it opens at the document's range (the unit cell)
    assert [dlg.find_element("css selector", f"[data-lo='{k}']").get_attribute("value") for k in range(3)] == ["0"] * 3
    assert [dlg.find_element("css selector", f"[data-hi='{k}']").get_attribute("value") for k in range(3)] == ["1"] * 3
    # each field says which end it is (a row of "from" over a row of "to")
    assert [x.text.strip() for x in dlg.find_elements("css selector", "label.f")] == \
        ["a from", "b from", "c from", "a to", "b to", "c to"]

    def typed(sel, text):
        el = dlg.find_element("css selector", sel)
        click(drv, el)
        el.send_keys(Keys.CONTROL, "a")
        el.send_keys(text)

    # an end below its start is refused, and the dialog stays open
    typed("[data-hi='0']", "0.02")
    click(drv, dlg.find_element("css selector", "[data-act='apply']"))
    time.sleep(0.2)
    assert dlg.is_displayed() and "0.05" in dlg.find_element("css selector", ".set-err").text
    n0 = drawn(drv)
    for k in range(3):
        typed(f"[data-lo='{k}']", "-0.1")
        typed(f"[data-hi='{k}']", "1.1")
    click(drv, dlg.find_element("css selector", "[data-act='apply']"))
    H.wait(drv, "return !document.getElementById('range-custom').open", 5)
    wait_redraw(drv, n0)
    assert drawn_set(drv) == oracle_range(s, {"lo": [-0.1] * 3, "hi": [1.1] * 3}, True)
    assert drv.find_element("id", "tb-range-label").text == "−0.1–1.1"
    eff = drv.execute_script("const a = window.__mace.activeDoc(), v = window.__mace.display.of(a); return [v.displayLo, v.displayHi]")
    assert eff == [[-0.1] * 3, [1.1] * 3]
    assert not H.errors(drv), H.errors(drv)
