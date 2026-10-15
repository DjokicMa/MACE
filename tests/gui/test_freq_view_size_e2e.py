"""The mode view keeps its size while it plays and turns (1.1.5 report, at
UI scale 110 %: "it shows the vib mode view in the center but there are
vertical and horizontal scroll bars, and if you rotate or press play it zooms
in and out").

The structure views' 280 px floor made the mode view's viewer taller than
the room under its head, so its panel scrolled; a scrollbar took width or
height from the 3D view, which resized it, which took the scrollbar away
again. Here every animation frame of a playing mode, through a real
left-drag that turns it, is recorded in the page: no panel or viewer between
the 3D view and its tab scrolls, the canvas stays its viewer's size, and the
camera's zoom never changes. UI scale 100, 110 and 125 %, devicePixelRatio 1
and 2, Chromium and Firefox; the corpus file is copied in at run time
(skipped without the test/ corpus).
"""
import shutil
import time

import pytest

import guiharness as H

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")

CORPUS = H.REPO_ROOT / "test" / "FREQ"
DIA = "2_dia2_opt_rev1_freq_B3LYP-D3-D3_optimized.out"

# started, then read back when done (the page records while WebDriver drags)
RECORD = """const n = arguments[0], el = document.getElementById('viewer-freq'), tab = document.getElementById('tab-freq');
  const v = window.MACE.state.viewers['viewer-freq'], log = [];
  const f0 = window.__maceFreqFrames;
  window.__sizeRec = null;
  const step = () => {
    const cv = el.querySelector('canvas'), r = el.getBoundingClientRect(), z = r.width / el.offsetWidth;
    const scroll = [];
    for (let e = el; e; e = e.parentElement) {
      const cs = getComputedStyle(e), clip = cs.overflowX !== 'visible' || cs.overflowY !== 'visible';
      if (clip && (e.scrollHeight > e.clientHeight || e.scrollWidth > e.clientWidth))
        scroll.push([e.id || e.className, e.scrollWidth, e.clientWidth, e.scrollHeight, e.clientHeight]);
      if (e === tab) break;
    }
    log.push({scroll, own: [el.offsetWidth, el.offsetHeight], canvas: [cv.offsetWidth, cv.offsetHeight, cv.style.width, cv.style.height],
      W: [v.WIDTH, v.HEIGHT], zoom: v.getView()[3], q: v.getView().slice(4), z});
    if (log.length < n) requestAnimationFrame(step); else window.__sizeRec = {log, frames: window.__maceFreqFrames - f0};
  };
  requestAnimationFrame(step);"""


@pytest.fixture(scope="module", params=[("chromium", 1), ("firefox", 1), ("chromium", 2), ("firefox", 2)],
                ids=["chromium-dpr1", "firefox-dpr1", "chromium-dpr2", "firefox-dpr2"])
def app(request, tmp_path_factory):
    if not (CORPUS / DIA).is_file():
        pytest.skip("test/ corpus not present (gitignored)")
    name, dpr = request.param
    tmp = tmp_path_factory.mktemp("freqsize")
    root = H.examples_copy(tmp)
    (root / "freq").mkdir()
    shutil.copy2(CORPUS / DIA, root / "freq" / DIA)
    proc, base, _ = H.start_server(root, tmp / "xdg", MACE_GUI_TEST_HOOKS="1")
    kw = {}
    if dpr != 1:
        kw = {"args": (f"--force-device-scale-factor={dpr}",)} if name == "chromium" else {"prefs": {"layout.css.devPixelsPerPx": str(float(dpr))}}
    try:
        drv = H.browser(name, width=1280, height=800, **kw)
    except BaseException:
        proc.terminate()
        proc.wait(timeout=15)
        raise
    try:
        H.load(drv, base)
        got = drv.execute_script("return window.devicePixelRatio")
        if abs(got - dpr) > 0.01:
            pytest.skip(f"this browser would not run at devicePixelRatio {dpr} (got {got})")
        drv.execute_script("window.__mace.settings.set('reduceMotion', 'off')")
        # the mode loop pauses in a window without focus (a background window); some headless
        # browsers never give the page focus (Chrome for Testing 140), so the page is told it has it
        drv.execute_script("document.hasFocus = () => true")
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
    app.execute_script("window.__mace.closeAll(); window.__mace.settings.set('uiScale', 100)")
    time.sleep(0.3)


def drag(drv, x, y):
    from selenium.webdriver.common.actions.action_builder import ActionBuilder
    ab = ActionBuilder(drv)
    p = ab.pointer_action
    p.move_to_location(int(x), int(y))
    p.pointer_down()
    for k in range(1, 13):
        p.move_to_location(int(x + 5 * k), int(y + 2 * k))
        p.pause(0.04)
    p.pointer_up()
    ab.perform()


@pytest.mark.parametrize("scale", [100, 110, 125])
def test_mode_view_never_scrolls_or_pumps_while_playing_and_turning(drv, scale):
    drv.execute_script("window.__mace.settings.set('uiScale', arguments[0])", scale)
    time.sleep(0.5)
    drv.execute_async_script("const done = arguments[1]; window.__mace.openModes(arguments[0]).then(() => done(1))", f"freq/{DIA}")
    H.wait(drv, "const v = window.__mace.freqView, s = v && v.state(); return s && s.mode", 60)
    H.wait(drv, "return window.__mace.freqView.playing()", 10)
    time.sleep(1.0)                  # settled: the first fit, the bottom panel, the inspector
    drv.execute_script(RECORD, 120)
    res = H.wait(drv, "return window.__sizeRec", 30)
    log = res["log"]
    assert res["frames"] > 3, res["frames"]           # it played
    for e in log:
        assert not e["scroll"], (scale, e)
        assert e["canvas"][0] == e["own"][0] and e["canvas"][1] == e["own"][1], (scale, e)
        assert e["W"] == e["own"], (scale, e)
    assert len({e["zoom"] for e in log}) == 1, (scale, sorted({e["zoom"] for e in log}))
    assert len({tuple(e["own"]) for e in log}) == 1, (scale, {tuple(e["own"]) for e in log})
    # now turn it (a real left-drag, 3Dmol's own mouse) while it plays: still no scrollbar, the zoom unchanged
    r = drv.execute_script("const r = document.getElementById('viewer-freq').getBoundingClientRect(); return [r.left, r.top, r.width, r.height]")
    drv.execute_script(RECORD, 90)
    drag(drv, r[0] + r[2] * 0.3, r[1] + r[3] * 0.3)
    res2 = H.wait(drv, "return window.__sizeRec", 30)
    log2 = res2["log"]
    assert len({tuple(e["q"]) for e in log2}) > 1                    # it turned
    for e in log2:
        assert not e["scroll"], (scale, e)
        assert e["canvas"][0] == e["own"][0] and e["canvas"][1] == e["own"][1], (scale, e)
    assert {e["zoom"] for e in log2} == {log[0]["zoom"]}, (scale, sorted({e["zoom"] for e in log2}))
    assert not H.errors(drv), H.errors(drv)
