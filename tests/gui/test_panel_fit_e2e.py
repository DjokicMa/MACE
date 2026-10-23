"""The Structure Analysis panel fits the window (1.1.6), in headless Firefox:

* it opens at about a third of the window (380 px at 1280, 600 px at 1920),
  over the Inspector first, so most of the 3D view stays in sight;
* a panel dragged on a wide window stays whole inside a narrower one, when
  the window shrinks and when the panel opens again (it was cut at the right
  edge: the porosity values and the Probe radius hint);
* its left edge widens it (pointer and arrow keys), double-click gives its
  own width back;
* the Probe radius field shows its whole placeholder.
"""
import time

import pytest

import guiharness as H
import p4util as U

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")

RECT = "const r = document.getElementById('analysis-panel').getBoundingClientRect(); return [r.left, r.right, r.width, innerWidth]"


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    a = U.start_app(tmp_path_factory.mktemp("panelfit"))
    try:
        yield a
    finally:
        U.stop_app(a)


def _open(drv):
    drv.execute_script("window.__mace.run('analysis.structure.porosity')")
    H.wait(drv, "const d = document.getElementById('analysis-panel'); return d && d.open && !!d.querySelector('.an-form')", 30)
    time.sleep(0.3)


def test_the_panel_width_follows_the_window(app):
    drv = app[0]
    for w, h, lo, hi in [(1280, 800, 380, 420), (1400, 900, 420, 460), (1920, 1080, 600, 600)]:
        U.reset(drv, w, h)
        U.open_doc(drv, "structures/MgO.cif")
        _open(drv)
        left, right, width, win = drv.execute_script(RECT)
        assert lo <= round(width) <= hi and right <= win, (w, width, right)
        # the hint of Probe radius is whole
        assert drv.execute_script("const i = document.querySelector('#analysis-panel [data-p=probe_radius]');"
                                  "const c = document.createElement('canvas').getContext('2d'); c.font = getComputedStyle(i).font;"
                                  "return c.measureText(i.placeholder).width < i.clientWidth")
        drv.execute_script("document.getElementById('analysis-panel').close()")
    U.no_errors(drv)


def test_a_moved_panel_stays_inside_a_narrower_window(app):
    from selenium.webdriver.common.action_chains import ActionChains
    drv = app[0]
    U.reset(drv, 1920, 1080)
    U.open_doc(drv, "structures/MgO.cif")
    _open(drv)
    head = drv.find_element("css selector", "#analysis-panel .dlg-head h2")
    ActionChains(drv).click_and_hold(head).move_by_offset(-6, 0).release().perform()
    assert drv.execute_script("return document.getElementById('analysis-panel').style.left") != ""
    # the window shrinks while it is open, then it opens again
    H.set_viewport(drv, 1400, 900)
    H.wait(drv, "const r = document.getElementById('analysis-panel').getBoundingClientRect(); return r.right <= innerWidth + 0.5", 10)
    drv.execute_script("document.getElementById('analysis-panel').close()")
    _open(drv)
    left, right, width, win = drv.execute_script(RECT)
    assert left >= 0 and right <= win + 0.5, (left, right, win)
    U.no_errors(drv)


def test_the_left_edge_widens_the_panel(app):
    from selenium.webdriver.common.action_chains import ActionChains
    drv = app[0]
    U.reset(drv, 1400, 900)
    U.open_doc(drv, "structures/MgO.cif")
    _open(drv)
    w0 = drv.execute_script(RECT)[2]
    edge = drv.find_element("css selector", "#analysis-panel .dlg-edge")
    assert edge.get_attribute("role") == "separator"
    ActionChains(drv).click_and_hold(edge).move_by_offset(-150, 0).release().perform()
    left, right, w1, win = drv.execute_script(RECT)
    assert abs(w1 - (w0 + 150)) < 3 and right <= win + 0.5, (w0, w1, right)
    # arrow keys, as the other splitters
    drv.execute_script("const e = document.querySelector('#analysis-panel .dlg-edge'); e.focus();"
                       "e.dispatchEvent(new KeyboardEvent('keydown', {key: 'ArrowLeft', bubbles: true}))")
    assert drv.execute_script(RECT)[2] > w1 + 10
    # double-click: its own width again
    ActionChains(drv).double_click(edge).perform()
    assert abs(drv.execute_script(RECT)[2] - w0) < 1
    U.no_errors(drv)
