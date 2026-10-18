"""Hiding and showing the Explorer, the Inspector and the bottom panel
(Firefox and Chromium, UI scale 100% and 110%, repo data only): each hides
from its header's x, its key, the View menu, the status bar's toggle and a
splitter dragged shut; a hidden one leaves a tab on its edge that brings it
back (a real click, the pointer mapped through the zoom); Hide All Panels
brings back exactly the panels that were open, with a Show Panels chip; the
state survives a reload; the how-to-get-it-back toast comes once. With
MACE_PANEL_SHOTS set, screenshots at 110% go there."""
import os
import time
from pathlib import Path

import pytest

import guiharness as H
import p4util as U

pytestmark = [pytest.mark.gui, pytest.mark.guismoke]
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")
from selenium.webdriver.common.actions.action_builder import ActionBuilder  # noqa: E402
from selenium.webdriver.common.keys import Keys  # noqa: E402

W, HT = 1500, 900
PANELS = ("explorer", "inspector", "bottom")


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("panels")
    root = H.examples_copy(tmp)
    proc, base, _ = H.start_server(root, tmp / "xdg", MACE_GUI_TEST_HOOKS="1", MACE_GUI_NO_WARMUP="1")
    yield root, base
    proc.terminate()
    proc.wait(timeout=30)


@pytest.fixture(scope="module", params=["firefox", "chromium"])
def browser(request, served):
    root, base = served
    try:
        drv = H.browser(request.param, width=W, height=HT)
    except Exception as e:                       # noqa: BLE001
        if request.param == "chromium":
            pytest.skip(f"no Chromium: {e}")
        raise
    try:
        H.load(drv, base)
        yield drv, request.param
    finally:
        drv.quit()


def show_all(drv):
    drv.execute_script("const S = window.__mace; for (const n of ['explorer', 'inspector', 'bottom'])"
                       " if (!document.getElementById('shell').classList.contains('no-' + n)) continue; else S.run('view.' + n);")


@pytest.fixture(params=[100, 110])
def fresh(browser, request):
    drv, name = browser
    U.reset(drv, W, HT)
    drv.execute_script("sessionStorage.removeItem('mace.gui.panelHint'); window.__mace.profile.setUi('hints', {})")
    drv.execute_script("if (window.__mace.ctx().zenMode) window.__mace.run('view.zen')")
    if drv.execute_script("return document.getElementById('panels-exit') && !document.getElementById('panels-exit').hidden"):
        drv.execute_script("window.__mace.run('view.hideAllPanels')")
    show_all(drv)
    # a structure open: the bottom panel shows (a document with none, or none open, hides it by itself)
    drv.execute_async_script("const done = arguments[1]; window.__mace.open(arguments[0]).then(() => done(true))", "structures/MgO.cif")
    drv.execute_script("window.__mace.settings.set('uiScale', arguments[0])", request.param)
    time.sleep(0.4)
    drv.execute_script("document.querySelectorAll('#toasts .toast2').forEach((t) => t.remove())")
    try:
        yield drv, name, request.param
    finally:
        drv.execute_script("window.__mace.settings.set('uiScale', 100)")
        show_all(drv)


def shown(drv, n):
    return drv.execute_script("return !document.getElementById('shell').classList.contains('no-' + arguments[0])", n)


def wait_shown(drv, n, on=True):
    H.wait(drv, "return document.getElementById('shell').classList.contains('no-' + arguments[0]) === !arguments[1]", 5, n, on)


def click_el(drv, css):
    """A real click at the element's center (screen px, as the zoomed page reports them)."""
    x, y = drv.execute_script("const r = document.querySelector(arguments[0]).getBoundingClientRect();"
                              " return [r.left + r.width / 2, r.top + r.height / 2]", css)
    hit = drv.execute_script("const e = document.elementFromPoint(arguments[0], arguments[1]); return !!e && !!e.closest(arguments[2])", x, y, css)
    assert hit, f"{css} is covered at its own center"
    ab = ActionBuilder(drv)
    ab.pointer_action.move_to_location(int(x), int(y)).click()
    ab.perform()
    time.sleep(0.25)


def toasts(drv):
    return drv.execute_script("return [...document.querySelectorAll('#toasts .toast2 .msg')].map((m) => m.textContent)")


def test_each_panel_hides_and_comes_back_by_every_path(fresh):
    drv, name, scale = fresh
    for n in PANELS:
        # the header's x hides it; the edge tab (aria-expanded false, labeled with the key) brings it back
        click_el(drv, f'.panel-hide[data-panel="{n}"]')
        wait_shown(drv, n, False)
        edge = f'.edge-tab[data-panel="{n}"]'
        assert drv.execute_script("const e = document.querySelector(arguments[0]); return !e.hidden && e.getAttribute('aria-expanded') === 'false'"
                                  " && e.getAttribute('aria-controls') === arguments[1] && /^Show /.test(e.getAttribute('aria-label'))", edge, n)
        click_el(drv, edge)
        wait_shown(drv, n, True)
        assert drv.execute_script("return document.querySelector(arguments[0]).hidden", edge)
        # the status bar's toggle (aria-pressed) both ways
        click_el(drv, f"#st-panel-{n}")
        wait_shown(drv, n, False)
        assert drv.execute_script("return document.getElementById(arguments[0]).getAttribute('aria-pressed')", f"st-panel-{n}") == "false"
        click_el(drv, f"#st-panel-{n}")
        wait_shown(drv, n, True)
    # the keys (Ctrl+B, Ctrl+Alt+B, Ctrl+J) and the View menu's checkmarks
    for n, combo in (("explorer", (Keys.CONTROL, "b")), ("inspector", (Keys.CONTROL, Keys.ALT, "b")), ("bottom", (Keys.CONTROL, "j"))):
        drv.execute_script("document.body.focus()")
        U.keys(drv, *combo)
        wait_shown(drv, n, False)
        assert drv.execute_script("return window.__mace.menus && window.__mace.lastCommand().id") == f"view.{n}"
        U.keys(drv, *combo)
        wait_shown(drv, n, True)
    # the toast came once, for the first panel hidden
    hints = [t for t in toasts(drv) if "bring it back" in t]
    assert len(hints) == 1 and hints[0].startswith("Explorer hidden"), toasts(drv)
    assert not H.errors(drv), H.errors(drv)


def test_dont_show_again_is_remembered(fresh):
    drv, name, scale = fresh
    drv.execute_script("window.__mace.run('view.bottom')")
    btn = H.wait(drv, "return [...document.querySelectorAll('#toasts .toast2 button')].find((b) => b.textContent === \"Don't show again\")", 5)
    btn.click()
    H.wait(drv, "return (window.__mace.profile.ui('hints', {}) || {}).panelHidden === true", 5)
    drv.execute_script("sessionStorage.removeItem('mace.gui.panelHint'); window.__mace.run('view.bottom'); window.__mace.run('view.explorer')")
    time.sleep(0.3)
    assert not [t for t in toasts(drv) if "bring it back" in t and t.startswith("Explorer")]


def test_a_splitter_dragged_shut_hides_its_panel(fresh):
    drv, name, scale = fresh
    w0 = drv.execute_script("return getComputedStyle(document.getElementById('shell')).getPropertyValue('--w-explorer')")
    x, y = drv.execute_script("const r = document.getElementById('split-explorer').getBoundingClientRect(); return [r.left + r.width / 2, r.top + r.height / 2]")
    ab = ActionBuilder(drv)
    p = ab.pointer_action
    p.move_to_location(int(x), int(y)).pointer_down()
    for k in range(1, 9):
        p.move_to_location(max(1, int(x - k * 40)), int(y))
    p.pointer_up()
    ab.perform()
    wait_shown(drv, "explorer", False)
    click_el(drv, '.edge-tab[data-panel="explorer"]')
    wait_shown(drv, "explorer", True)
    assert drv.execute_script("return getComputedStyle(document.getElementById('shell')).getPropertyValue('--w-explorer')") == w0


def test_a_canceled_or_lost_splitter_drag_lets_go(fresh):
    """A drag the browser cancels (a touch turned into a scroll, a dialog taking the pointer) or whose
    pointer capture is lost ends there: no stuck .dragging, and moving the pointer no longer resizes.
    Canceled: the panel goes back to its size; lost capture: it keeps the size reached."""
    drv, name, scale = fresh
    width = lambda: drv.execute_script("return document.getElementById('explorer').getBoundingClientRect().width")  # noqa: E731
    drv.execute_script("const s = document.getElementById('split-explorer'); s.addEventListener('pointerdown', (e) => { window.__pid = e.pointerId; }, true)")
    def release():
        ab = ActionBuilder(drv)
        ab.pointer_action.pointer_up()
        ab.perform()
        time.sleep(0.2)
    for how in ("cancel", "lost"):
        w0 = width()
        x, y = drv.execute_script("const r = document.getElementById('split-explorer').getBoundingClientRect(); return [r.left + r.width / 2, r.top + r.height / 2]")
        ab = ActionBuilder(drv)
        ab.pointer_action.move_to_location(int(x), int(y)).pointer_down().move_to_location(int(x + 30), int(y))
        try:
            ab.perform()
            time.sleep(0.25)
            w1 = width()
            assert w1 > w0 + 10, (how, w0, w1)
            if how == "cancel":
                drv.execute_script("document.getElementById('split-explorer').dispatchEvent(new PointerEvent('pointercancel', {bubbles: true, pointerId: window.__pid}))")
            else:
                drv.execute_script("document.getElementById('split-explorer').releasePointerCapture(window.__pid)")
            time.sleep(0.2)
            # the button still held: moving on must not resize (a browser may report the lost capture
            # only with the next pointer event)
            ab = ActionBuilder(drv)
            ab.pointer_action.move_to_location(int(x + 80), int(y)).move_to_location(int(x + 120), int(y))
            ab.perform()
            time.sleep(0.25)
            assert not drv.execute_script("return document.getElementById('split-explorer').classList.contains('dragging')"), how
            w2 = width()
        finally:
            release()
        assert abs(w2 - (w0 if how == "cancel" else w1)) <= 1, (how, w0, w1, w2)
        assert abs(width() - w2) <= 1, how                                  # the release changes nothing either
    assert not H.errors(drv), H.errors(drv)


def test_hide_all_panels_round_trip_and_reload(fresh):
    drv, name, scale = fresh
    drv.execute_script("window.__mace.run('view.inspector')")             # the Inspector was already closed
    wait_shown(drv, "inspector", False)
    U.keys(drv, Keys.CONTROL, Keys.SHIFT, "f")
    for n in PANELS:
        wait_shown(drv, n, False)
    assert drv.execute_script("return !document.getElementById('panels-exit').hidden")
    assert drv.execute_script("return [...document.querySelectorAll('.edge-tab')].filter((e) => !e.hidden).length") == 3
    shots = os.environ.get("MACE_PANEL_SHOTS")
    if shots and scale == 110:
        Path(shots).mkdir(parents=True, exist_ok=True)
        drv.execute_script("document.querySelectorAll('#toasts .toast2').forEach((t) => t.remove())")
        time.sleep(0.3)
        drv.save_screenshot(str(Path(shots) / f"hide_all_panels_{name}_110.png"))
    # a reload keeps it
    time.sleep(1.5)
    drv.refresh()
    H.wait(drv, "return document.documentElement.dataset.appReady === '1'", 60)
    for n in PANELS:
        wait_shown(drv, n, False)
    assert drv.execute_script("return !document.getElementById('panels-exit').hidden")
    # the chip brings back exactly what was open: the Explorer and the bottom panel, not the Inspector
    click_el(drv, "#panels-exit")
    wait_shown(drv, "explorer", True)
    wait_shown(drv, "bottom", True)
    assert not shown(drv, "inspector")
    assert drv.execute_script("return document.getElementById('panels-exit').hidden")
    if shots and scale == 110:
        drv.save_screenshot(str(Path(shots) / f"panels_back_inspector_edge_tab_{name}_110.png"))
    assert not H.errors(drv), H.errors(drv)


def test_shortcuts_are_free_and_not_reserved(fresh):
    drv, name, scale = fresh
    for cmd, key in (("view.explorer", "Mod+B"), ("view.inspector", "Mod+Alt+B"), ("view.bottom", "Mod+J"), ("view.hideAllPanels", "Mod+Shift+F")):
        assert drv.execute_script("return window.__mace.keymap.keysOf(arguments[0])", cmd) == [key]
        assert not drv.execute_script("return window.__mace.keymap.isReserved(arguments[0])", key)
    conflicts = drv.execute_script("return window.__mace.keymap.allConflicts()")
    assert not [c for c in (conflicts or []) if "hideAllPanels" in str(c)], conflicts


def test_edge_tabs_over_a_split_and_the_narrow_inspector_drawer(fresh):
    drv, name, scale = fresh
    drv.execute_async_script("const done = arguments[0]; window.__maceGroups.openToSide('structures/Si.cif').then(() => done(true))")
    try:
        # with two editor groups, the Inspector's edge tab sits over the right group's frame and still takes the click
        drv.execute_script("window.__maceGroups.focusGroup('main'); window.__mace.run('view.inspector')")
        wait_shown(drv, "inspector", False)
        click_el(drv, '.edge-tab[data-panel="inspector"]')
        wait_shown(drv, "inspector", True)
    finally:
        drv.execute_script("window.__maceGroups.closeGroup(true)")
    # a narrow window: the Inspector is a drawer, shut until asked for; its edge tab opens it
    H.set_viewport(drv, 1000, HT)
    try:
        H.wait(drv, "return document.getElementById('shell').classList.contains('no-inspector')", 5)
        click_el(drv, '.edge-tab[data-panel="inspector"]')
        H.wait(drv, "return document.getElementById('shell').classList.contains('drawer-open')", 5)
    finally:
        drv.execute_script("if (document.getElementById('shell').classList.contains('drawer-open')) window.__mace.run('view.inspector')")
        H.set_viewport(drv, W, HT)
        time.sleep(0.3)
    assert not H.errors(drv), H.errors(drv)


def test_the_bottom_panels_close_button_covers_no_tab(fresh):
    """At a width where the bottom tabs overflow, the ×'s box overlaps no tab's visible part (the strip's
    end padding existed only at the scroll end, so the last visible tab ran under it: "PROBLE✕")."""
    drv = fresh[0]
    try:
        H.set_viewport(drv, 900, 700)
        time.sleep(0.4)
        hit = drv.execute_script("""const x = document.querySelector('#bottom > .panel-hide').getBoundingClientRect();
          const strip = document.querySelector('#bottom > .bottom-tabs').getBoundingClientRect();
          return [...document.querySelectorAll('#bottom .bottom-tab')].filter((t) => t.offsetParent).map((t) => {
            const r = t.getBoundingClientRect(), right = Math.min(r.right, strip.right);
            return right > x.left + 0.5 && r.left < x.right ? t.textContent.trim() : null; }).filter(Boolean);""")
        assert hit == [], hit
    finally:
        H.set_viewport(drv, W, HT)
