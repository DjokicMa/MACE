"""A middle click in a 3D viewer pans or zooms; it never pastes (Linux's
primary selection). Headless Chromium on Linux runs the real primary-selection
paste for a WebDriver middle click: text selected with the mouse in a field,
then a middle click in the viewer, would open Import from Clipboard. Both
browsers also get a middle press and release in the viewer followed at once
by a paste event. Ctrl+V and a middle click on a text field still paste, and
the middle and right drags still pan and turn the view."""
import time

import pytest

import guiharness as H

pytestmark = [pytest.mark.gui, pytest.mark.guismoke]
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")                 # the server reads the structures with it

from selenium.webdriver.common.action_chains import ActionChains  # noqa: E402

CIF = "data_x\n_cell_length_a 4\n_cell_length_b 4\n_cell_length_c 4\n_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"

PROBE_JS = """const i = document.createElement('input'); i.id = arguments[0]; i.value = arguments[1];
  i.style = 'position:fixed;left:10px;top:' + arguments[2] + 'px;width:300px;height:30px;z-index:99999;font-size:16px';
  document.body.appendChild(i);"""

# a middle press and release on the canvas, as the browser sends them, then the paste
MIDDLE_THEN_PASTE_JS = """const cv = document.querySelector(arguments[0] + ' canvas'), r = cv.getBoundingClientRect();
  const at = {clientX: r.left + 120, clientY: r.top + 90, button: 1, buttons: 4, bubbles: true, cancelable: true, composed: true};
  const prevented = [];
  for (const [C, type] of [[PointerEvent, 'pointerdown'], [MouseEvent, 'mousedown'], [PointerEvent, 'pointerup'],
                           [MouseEvent, 'mouseup'], [MouseEvent, 'auxclick']]) {
    const e = new C(type, {...at, pointerId: 1, pointerType: 'mouse', buttons: type.endsWith('down') ? 4 : 0});
    cv.dispatchEvent(e);
    prevented.push(e.defaultPrevented);
  }
  const dt = new DataTransfer(); dt.setData('text/plain', arguments[1]);
  (document.activeElement || document.body).dispatchEvent(new ClipboardEvent('paste', {clipboardData: dt, bubbles: true, cancelable: true}));
  return {prevented, hook: window.__mace.onPaste(document.activeElement || document.body, arguments[1])};"""


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("midpaste")
    root = H.examples_copy(tmp)
    proc, base, _ = H.start_server(root, tmp / "xdg", MACE_GUI_TEST_HOOKS="1")
    try:
        drv = H.browser()
    except BaseException:
        proc.terminate()
        proc.wait(timeout=15)
        raise
    try:
        H.load(drv, base)
        yield drv
    finally:
        drv.quit()
        proc.terminate()
        proc.wait(timeout=15)


@pytest.fixture
def drv(app, request):
    app.execute_script("window.__mace.closeAll(); document.querySelectorAll('dialog[open]').forEach((d) => d.close());"
                       "document.querySelectorAll('#probe, #probe2').forEach((e) => e.remove());"
                       "window.__maceErrors.length = 0; performance.clearResourceTimings();")
    failed = request.session.testsfailed
    yield app
    if request.session.testsfailed > failed:          # the call's report is in before teardown
        H.save_failure(app, request.node.name)


def open_doc(drv, path):
    kind = drv.execute_async_script("const d = arguments[1]; window.__mace.open(arguments[0]).then((x) => d(x && x.kind))", path)
    time.sleep(0.6)
    return kind


def state(drv):
    """What a paste could change: the open documents, parse requests, the atoms, the Import dialog."""
    return drv.execute_script("""const d = document.getElementById('import-clipboard');
      return {docs: window.__mace.docs().map((x) => x.title), dialog: !!(d && d.open),
              parse: performance.getEntriesByType('resource').filter((r) => r.name.includes('api/parse')).length,
              atoms: window.__mace.viewState().displayedAtoms}""")


def select_probe(drv, text):
    """`text` in a field, selected with a real mouse drag: on Linux that is the primary selection."""
    drv.execute_script(PROBE_JS, "probe", text, 10)
    el = drv.find_element("id", "probe")
    ActionChains(drv).move_to_element_with_offset(el, -145, 0).click_and_hold().move_to_element_with_offset(el, 145, 0).release().perform()
    time.sleep(0.2)
    return el


def is_chromium(drv):
    return drv.capabilities.get("browserName", "").lower() in ("chrome", "chromium", "chrome-headless-shell")


def test_middle_click_in_viewer_pastes_nothing(drv):
    assert open_doc(drv, "structures/MgO.cif") == "structure"
    H.wait(drv, "return window.__mace.viewState().displayedAtoms > 0", 20)
    select_probe(drv, "data_x primary selection")    # a field drops the newlines: one line
    before = state(drv)
    H.click_at(drv, 120, 90, button=1)                  # real: Chromium pastes the primary selection here
    time.sleep(0.6)
    assert state(drv) == before
    assert drv.execute_script("return document.getElementById('probe').value") == "data_x primary selection"
    # the events themselves: the release, its mouseup and auxclick are canceled; the paste is ignored
    r = drv.execute_script(MIDDLE_THEN_PASTE_JS, "#viewer-main", CIF)
    assert r["prevented"][2:] == [True, True, True], r
    assert r["hook"] is False, r
    time.sleep(0.6)
    assert state(drv) == before
    # Ctrl+V right after a middle click: its paste opens Import from Clipboard
    drv.execute_script(MIDDLE_THEN_PASTE_JS.replace("const dt", "return {prevented, hook: null}; const dt"), "#viewer-main", CIF)
    drv.execute_script("document.body.dispatchEvent(new KeyboardEvent('keydown', {key: 'v', code: 'KeyV', ctrlKey: true, bubbles: true, cancelable: true}))")
    assert drv.execute_script("return window.__mace.onPaste(document.body, arguments[0])", CIF) is True
    H.wait(drv, "const d = document.getElementById('import-clipboard'); return d && d.open", 10)
    assert not H.errors(drv), H.errors(drv)


def test_middle_click_on_text_field_still_pastes(drv):
    assert open_doc(drv, "structures/MgO.cif") == "structure"
    H.wait(drv, "return window.__mace.viewState().displayedAtoms > 0", 20)
    select_probe(drv, "PRIMARY")
    H.click_at(drv, 120, 90, button=1)
    time.sleep(0.6)                                     # past the viewer's window
    drv.execute_script(PROBE_JS, "probe2", "", 60)
    el = drv.find_element("id", "probe2")
    ActionChains(drv).click(el).perform()
    from selenium.webdriver.common.actions.action_builder import ActionBuilder
    ab = ActionBuilder(drv)
    ab.pointer_action.move_to(el).pointer_down(1).pointer_up(1)
    ab.perform()
    time.sleep(0.4)
    if is_chromium(drv):                                # Chromium runs the primary-selection paste headless
        assert drv.execute_script("return document.getElementById('probe2').value") == "PRIMARY"
    # a paste on a text field is the field's (the page's handler leaves it)
    assert drv.execute_script("return window.__mace.onPaste(document.getElementById('probe2'), 'x')") is False
    assert not drv.execute_script("const d = document.getElementById('import-clipboard'); return !!(d && d.open)")
    assert not H.errors(drv), H.errors(drv)


def _model_point_under(drv, x, y):
    return drv.execute_script("""const c = window.__mace.viewport.camera().calibrate();
        return [0, 1, 2].map((k) => c.center[k] + c.right[k] * (arguments[0] - c.centerPx[0]) / c.s
                                    - c.up[k] * (arguments[1] - c.centerPx[1]) / c.s)""", x, y)


def test_drags_still_pan_orbit_and_zoom(drv):
    assert open_doc(drv, "structures/MgO.cif") == "structure"
    H.wait(drv, "return window.__mace.viewState().displayedAtoms > 0", 20)
    canvas = H.viewer_canvas(drv)
    view = "return window.__mace.viewState().view"
    H.click_at(drv, 120, 90, button=1)
    # middle-drag pans: the model point under the pointer follows it
    p0 = _model_point_under(drv, 200, 120)
    H.drag(drv, canvas, [(200, 120), (260, 160)], button=1)
    time.sleep(0.2)
    s = drv.execute_script("return window.__mace.viewport.camera().screenOf(arguments[0])", p0)
    assert abs(s[0] - 260) < 2 and abs(s[1] - 160) < 2, s
    # right-drag turns it
    v0 = drv.execute_script(view)
    H.drag(drv, canvas, [(200, 120), (280, 150)], button=2)
    time.sleep(0.2)
    v1 = drv.execute_script(view)
    assert v1[4:] != v0[4:] and v1[:4] == v0[:4]
    # 3Dmol's own viewer (an optimization's frames): its middle-drag still zooms, and still pastes nothing
    assert open_doc(drv, "outputs/Ag2Br3_opt_unfinished.out") == "trajectory"
    H.wait(drv, "return !!(window.MACE.state.traj && window.MACE.state.viewers['viewer-before'])", 30)
    time.sleep(0.4)
    before = state(drv)
    v = "return window.MACE.state.viewers['viewer-before'].getView()"
    g0 = drv.execute_script(v)
    gl = drv.find_element("css selector", "#viewer-before canvas")
    H.drag(drv, gl, [(150, 100), (150, 160)], button=1)
    time.sleep(0.3)
    g1 = drv.execute_script(v)
    assert g1 != g0
    r = drv.execute_script(MIDDLE_THEN_PASTE_JS, "#viewer-before", CIF)
    assert r["prevented"][1:] == [True, True, True, True], r
    assert r["hook"] is False
    time.sleep(0.4)
    assert state(drv) == before
    assert not H.errors(drv), H.errors(drv)
