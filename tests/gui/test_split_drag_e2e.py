"""Two editor groups, polish (1.1.6): in headless Firefox and Chromium, real
pointer drags and real keys, at UI scale 100% and 110% (the right group's
points mapped through its frame and the CSS zoom, as window.maceZoom says):

- the Inspector, the bottom panel and the selection commands follow the
  focused group, whether the focus moved by a click or by the keyboard
  (Focus Other Group, Focus Left/Right Group);
- a tab dragged from the right group onto the left group's tab strip goes in
  at that place; onto the left group's editor area it goes at the end; a tab
  dragged within a strip reorders it; Esc cancels a drag;
- the right group closes when its last tab is closed or moved out; closing
  the left group's last tab brings the right group's documents left;
- a tab dragged onto the right half of a single group splits the editor;
- the right group's order and active tab come back after a reload;
- Shift+Left/Right moves a focused tab, Move to Other Group from the keyboard,
  each tab strip names its group;
- a right group opened after a theme change has the window's theme (and
  follows a later change, where the build passes settings to the pane).

Repo data only (examples/gui and the generic CRYSTAL23 fixtures)."""
import time

import pytest

import guiharness as H
import p4util as U
from test_parity_fixtures import unpack

pytestmark = [pytest.mark.gui, pytest.mark.guismoke]
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")
from selenium.webdriver.common.action_chains import ActionChains  # noqa: E402
from selenium.webdriver.common.actions.action_builder import ActionBuilder  # noqa: E402
from selenium.webdriver.common.by import By  # noqa: E402
from selenium.webdriver.common.keys import Keys  # noqa: E402

W, HT = 1500, 900


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("split_drag")
    root = H.examples_copy(tmp)
    unpack(root / "parity")
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


@pytest.fixture
def fresh(browser):
    drv, name = browser
    drv.execute_script("const g = window.__maceGroups; if (g && g.isOpen()) g.closeGroup(true)")
    H.wait(drv, "return !window.__maceGroups.isOpen()", 10)
    U.reset(drv, W, HT)
    drv.execute_script("window.__macePane.docs().forEach((d) => window.__macePane.close(d.id, true))")
    drv.execute_script("window.__mace.settings.set('reduceMotion', 'off')")
    try:
        yield drv
    except BaseException:
        H.save_failure(drv, f"split_drag_{name}")
        raise
    finally:
        drv.execute_script("window.__mace.settings.set('uiScale', 100)")


# ---------------------------------------------------------------- helpers
def side(drv, js):
    """Run js in the right group's page (`p` its __macePane, `M` its __mace, `W` its window)."""
    return drv.execute_script("const f = document.querySelector('.group-frame'); if (!f) return null; "
                              "const p = f.contentWindow.__macePane, M = f.contentWindow.__mace, W = f.contentWindow; " + js)


def wait_side(drv, js, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        try:
            r = side(drv, js)
        except Exception:                        # noqa: BLE001 (the frame is loading)
            r = None
        if r:
            return r
        time.sleep(0.2)
    raise AssertionError(f"right group: timed out waiting for {js}")


def titles(drv, where="main"):
    js = "return [...D.querySelectorAll('#doc-tabs .doc-tab .t')].map((t) => t.textContent)"
    return drv.execute_script("const D = document; " + js) if where == "main" else (side(drv, "const D = W.document; " + js) or [])


def open_main(drv, path):
    drv.execute_async_script("const done = arguments[1]; window.__mace.open(arguments[0]).then(() => done(true), () => done(false))", path)
    H.wait(drv, "const d = window.__mace.activeDoc(); return d && d.path === arguments[0]", 60, path)


def open_side(drv, path):
    """Open to the Side, to the end (the group takes the focus once the document is in)."""
    drv.execute_async_script("const done = arguments[1]; window.__maceGroups.openToSide(arguments[0]).then(() => done(true), () => done(false))", path)
    wait_side(drv, f"const a = p.active(); return a && a.path === {path!r}")


def tab_point(drv, title, where="main", at=0.5):
    """A point across a tab (at: 0 its left edge, 0.5 its center), in the window's viewport px (the right
    group's mapped through its frame and its zoom)."""
    if where == "main":
        return drv.execute_script(
            "const t = [...document.querySelectorAll('#doc-tabs .doc-tab')].find((x) => x.querySelector('.t').textContent === arguments[0]);"
            "const r = t.getBoundingClientRect(); return [r.left + r.width * arguments[1], r.top + r.height / 2]", title, at)
    return side(drv, "const fr = f.getBoundingClientRect(), k = fr.width / W.innerWidth;"
                     "const t = [...W.document.querySelectorAll('#doc-tabs .doc-tab')].find((x) => x.querySelector('.t').textContent === "
                     f"{title!r}); const r = t.getBoundingClientRect();"
                     f"return [fr.left + (r.left + r.width * {at}) * k, fr.top + (r.top + r.height / 2) * k]")


def rect(drv, sel):
    return drv.execute_script("const r = document.querySelector(arguments[0]).getBoundingClientRect();"
                              "return {left: r.left, top: r.top, right: r.right, bottom: r.bottom, width: r.width, height: r.height}", sel)


def drag(drv, a, b, steps=12, release=True):
    """A real pointer drag between two viewport points (the browser routes it into the frame and back)."""
    ab = ActionBuilder(drv)
    p = ab.pointer_action
    p.move_to_location(int(a[0]), int(a[1]))
    p.pointer_down()
    for i in range(1, steps + 1):
        p.move_to_location(int(a[0] + (b[0] - a[0]) * i / steps), int(a[1] + (b[1] - a[1]) * i / steps))
    p.pause(0.25)
    if release:
        p.pointer_up()
    ab.perform()
    time.sleep(0.3)


def no_errors(drv):
    errs = [e for e in H.errors(drv) if "ResizeObserver loop" not in e]
    assert not errs, errs
    errs = [e for e in (side(drv, "return W.__maceErrors || []") or []) if "ResizeObserver loop" not in e]
    assert not errs, errs


def focused(drv):
    return drv.execute_script("return window.__maceGroups.state().focused")


def in_frame(drv, fn):
    drv.switch_to.frame(drv.find_element(By.CSS_SELECTOR, ".group-frame"))
    try:
        return fn()
    finally:
        drv.switch_to.default_content()


PANELS = """const f = document.querySelector('.group-frame'), W = f.contentWindow, fr = f.getBoundingClientRect();
  const shown = (w, e) => !!e && w.getComputedStyle(e).display !== 'none' && e.getBoundingClientRect().width > 0;
  return {cover: Math.round(fr.right) === Math.round(document.getElementById('inspector').getBoundingClientRect().right),
    inert: document.getElementById('inspector').inert,
    paneInsp: shown(W, W.document.getElementById('inspector')),
    paneSections: [...W.document.querySelectorAll('#inspector .insp-section')].map((x) => x.dataset.key),
    winSections: [...document.querySelectorAll('#inspector .insp-section')].map((x) => x.dataset.key),
    paneTab: (W.document.querySelector('.bottom-tab.active') || {}).dataset.bottom || null,
    winTab: (document.querySelector('.bottom-tab.active') || {}).dataset.bottom || null,
    status: !!document.getElementById('st-group')};"""


def panels(drv, cond, timeout=20):
    """The PANELS reading once cond (a Python predicate on it) holds; the last reading in the failure."""
    end, g = time.time() + timeout, None
    while time.time() < end:
        g = drv.execute_script(PANELS)
        if cond(g):
            return g
        time.sleep(0.2)
    raise AssertionError(f"panels: {g}, groups: {drv.execute_script('return window.__maceGroups.state()')}")


# ---------------------------------------------------------------- focus follows
def test_inspector_bottom_panel_and_selection_follow_the_keyboard_focus(fresh):
    drv = fresh
    open_main(drv, "structures/MgO.cif")
    open_side(drv, "parity/water_freq.out")
    wait_side(drv, "const a = p.active(); return a && a.kind === 'freq'", 90)
    # the explorer's Open to the Side gave the right group the focus: its Inspector and its bottom panel
    g = panels(drv, lambda g: g["cover"] and g["paneInsp"])
    # (then, read-only, the Structure and Cell of the FREQ run: the panel context, 1.1.6)
    assert g["inert"] and g["paneSections"][:2] == ["mode", "scf"] and g["paneTab"] == "modes" and g["status"], g
    assert g["winTab"] == "atoms", g                                  # the left group keeps its own bottom tab
    # Mod+Alt+\ (pressed in the right group): the window's Inspector is the left structure's again
    in_frame(drv, lambda: U.keys(drv, Keys.CONTROL, Keys.ALT, "\\"))
    g = panels(drv, lambda g: not g["cover"] and not g["paneInsp"])
    assert focused(drv) == "main" and not g["inert"] and not g["status"], g
    assert g["winSections"] and "mode" not in g["winSections"], g
    # Focus Right Group from the palette's command (no default key): the right group's again
    drv.execute_script("window.__mace.run('view.focusRightGroup')")
    panels(drv, lambda g: g["cover"] and g["paneInsp"])
    assert focused(drv) == "side"
    drv.execute_script("window.__mace.run('view.focusLeftGroup')")
    H.wait(drv, "return window.__maceGroups.state().focused === 'main'", 10)
    no_errors(drv)


def test_select_all_acts_on_the_focused_group(fresh):
    drv = fresh
    open_main(drv, "structures/MgO.cif")
    open_side(drv, "structures/Si.cif")
    wait_side(drv, "return M.editor && M.viewport && M.viewport.displayed() > 0", 60)
    H.wait(drv, "return window.__maceGroups.state().focused === 'side'", 10)
    # Mod+A in the window, while the right group has the focus: the right group's atoms are selected
    drv.execute_script("document.body.focus()")
    drv.execute_script("window.__maceGroups.focusGroup('side')")
    drv.execute_script("window.__mace.run('select.all')")
    n_side = wait_side(drv, "const s = M.editor.selection(); return s.length > 0 && s.length")
    assert n_side == side(drv, "return M.doc().structure.numbers.length"), n_side
    assert drv.execute_script("return window.__mace.editor ? window.__mace.editor.selection().length : 0") == 0
    # the keyboard moves the focus left (pressed in the right group): Select All now selects MgO's atoms
    in_frame(drv, lambda: U.keys(drv, Keys.CONTROL, Keys.ALT, "\\"))
    H.wait(drv, "return window.__maceGroups.state().focused === 'main'", 10)
    drv.execute_script("window.__mace.run('select.all')")
    H.wait(drv, "return window.__mace.editor.selection().length === window.__mace.doc().structure.numbers.length", 10)
    no_errors(drv)


# ---------------------------------------------------------------- dragging tabs
@pytest.mark.parametrize("scale", [100, 110])
def test_drag_a_tab_back_reorder_and_collapse(fresh, scale):
    drv = fresh
    drv.execute_script("window.__mace.settings.set('uiScale', arguments[0])", scale)
    if scale != 100:
        H.wait(drv, "return Math.abs(window.maceZoom.of(document.getElementById('doc-tabs')) - arguments[0] / 100) < 0.01", 10, scale)
    open_main(drv, "structures/MgO.cif")
    open_main(drv, "structures/Si.cif")
    open_side(drv, "structures/NaCl.cif")
    open_side(drv, "structures/LiF.cif")
    assert titles(drv) == ["MgO.cif", "Si.cif"] and titles(drv, "side") == ["NaCl.cif", "LiF.cif"]
    # within the right group's strip: LiF before NaCl
    drag(drv, tab_point(drv, "LiF.cif", "side"), tab_point(drv, "NaCl.cif", "side", 0.25))
    wait_side(drv, "return p.docs().map((d) => d.title).join() === 'LiF.cif,NaCl.cif'", 10)
    # LiF from the right group onto the left strip, between MgO and Si
    mg, si = tab_point(drv, "MgO.cif"), tab_point(drv, "Si.cif")
    gap = rect(drv, "#doc-tabs .doc-tab")["right"]
    drag(drv, tab_point(drv, "LiF.cif", "side"), (gap + 2, (mg[1] + si[1]) / 2))
    H.wait(drv, "return window.__macePane.docs().map((d) => d.title).join() === 'MgO.cif,LiF.cif,Si.cif'", 30)
    assert titles(drv, "side") == ["NaCl.cif"]
    assert drv.execute_script("return window.__mace.activeDoc().title") == "LiF.cif"
    assert focused(drv) == "main"
    # within the left strip: Si to the front
    drag(drv, tab_point(drv, "Si.cif"), tab_point(drv, "MgO.cif", at=0.25))
    H.wait(drv, "return window.__macePane.docs().map((d) => d.title).join() === 'Si.cif,MgO.cif,LiF.cif'", 10)
    # a left tab onto the right group's strip, before NaCl
    drag(drv, tab_point(drv, "MgO.cif"), tab_point(drv, "NaCl.cif", "side", 0.25))
    wait_side(drv, "return p.docs().map((d) => d.title).join() === 'MgO.cif,NaCl.cif'", 30)
    # Esc cancels a drag: nothing moves
    a = tab_point(drv, "NaCl.cif", "side")
    views = rect(drv, "#views")
    drag(drv, a, (views["left"] + views["width"] / 2, views["top"] + views["height"] / 2), release=False)
    assert drv.execute_script("return !!document.querySelector('body > .group-drop.over')")
    in_frame(drv, lambda: ActionChains(drv).send_keys(Keys.ESCAPE).perform())
    H.wait(drv, "return !document.body.classList.contains('tab-dragging')", 5)
    ActionBuilder(drv).clear_actions()
    ab = ActionBuilder(drv)
    ab.pointer_action.pointer_up()
    ab.perform()
    time.sleep(0.3)
    assert titles(drv, "side") == ["MgO.cif", "NaCl.cif"]
    # the right group's tabs onto the left editor area: each goes at the end; the last one closes the group
    for t in ["MgO.cif", "NaCl.cif"]:
        views = rect(drv, "#views")
        drag(drv, tab_point(drv, t, "side"), (views["left"] + views["width"] / 2, views["top"] + views["height"] / 2))
        H.wait(drv, "return window.__macePane.docs().some((d) => d.title === arguments[0])", 30, t)
    try:
        H.wait(drv, "return !window.__maceGroups.isOpen() && !document.querySelector('.group-frame')", 20)
    except AssertionError:
        raise AssertionError(f"the right group did not close: left {titles(drv)}, right {titles(drv, 'side')}, "
                             f"{drv.execute_script('return window.__maceGroups.state()')}") from None
    assert titles(drv) == ["Si.cif", "LiF.cif", "MgO.cif", "NaCl.cif"]
    # one group: a tab dragged onto its right half splits the editor
    views = rect(drv, "#views")
    drag(drv, tab_point(drv, "Si.cif"), (views["right"] - 40, views["top"] + views["height"] / 2))
    wait_side(drv, "return p.docs().map((d) => d.title).join() === 'Si.cif'", 60)
    assert titles(drv) == ["LiF.cif", "MgO.cif", "NaCl.cif"]
    # closing the right group's last tab (a real click on its x) closes the group
    x = side(drv, "const fr = f.getBoundingClientRect(), k = fr.width / W.innerWidth;"
                  "const r = W.document.querySelector('#doc-tabs .doc-tab [data-close]').getBoundingClientRect();"
                  "return [fr.left + (r.left + r.width / 2) * k, fr.top + (r.top + r.height / 2) * k]")
    ab = ActionBuilder(drv)
    ab.pointer_action.move_to_location(int(x[0]), int(x[1])).click()
    ab.perform()
    H.wait(drv, "return !window.__maceGroups.isOpen() && !document.querySelector('.group-frame')", 10)
    assert titles(drv) == ["LiF.cif", "MgO.cif", "NaCl.cif"]
    no_errors(drv)


def test_closing_the_left_groups_last_tab_brings_the_right_group_left(fresh):
    drv = fresh
    open_main(drv, "structures/MgO.cif")
    open_side(drv, "structures/Si.cif")
    open_side(drv, "structures/NaCl.cif")
    side(drv, "p.reopen({path: 'structures/Si.cif'})")                      # Si active on the right
    wait_side(drv, "return p.active().title === 'Si.cif'")
    ActionChains(drv).move_to_element(drv.find_element(By.CSS_SELECTOR, "#doc-tabs .doc-tab [data-close]")).click().perform()
    H.wait(drv, "return !window.__maceGroups.isOpen() && window.__macePane.docs().length === 2", 30)
    assert titles(drv) == ["Si.cif", "NaCl.cif"]
    assert drv.execute_script("return window.__mace.activeDoc().title") == "Si.cif"
    no_errors(drv)


# ---------------------------------------------------------------- persistence, keyboard, labels, theme
def test_the_right_groups_order_and_active_tab_come_back(fresh):
    drv = fresh
    open_main(drv, "structures/MgO.cif")
    open_side(drv, "structures/Si.cif")
    open_side(drv, "structures/NaCl.cif")
    side(drv, "p.reorder(p.docs().find((d) => d.title === 'NaCl.cif').id, 0)")
    side(drv, "p.reopen({path: 'structures/Si.cif'})")
    wait_side(drv, "return p.active().title === 'Si.cif' && p.docs().map((d) => d.title).join() === 'NaCl.cif,Si.cif'")
    drv.execute_script("window.__maceGroups.focusGroup('main')")                # persists the split
    time.sleep(1.5)                                                               # the profile write
    drv.refresh()
    H.wait(drv, "return document.documentElement.dataset.appReady === '1'", 60)
    wait_side(drv, "return p && p.docs().length === 2 && p.active() && p.active().title === 'Si.cif'", 60)
    assert titles(drv, "side") == ["NaCl.cif", "Si.cif"]
    assert focused(drv) == "main"
    no_errors(drv)


def test_keyboard_moves_tabs_and_strips_name_their_group(fresh):
    drv = fresh
    open_main(drv, "structures/MgO.cif")
    open_main(drv, "structures/Si.cif")
    assert drv.execute_script("return document.getElementById('doc-tabs').getAttribute('aria-label')") == "Open documents"
    # Shift+Left on the focused tab moves it; the arrows alone go to the next tab
    drv.execute_script("document.querySelector('#doc-tabs .doc-tab.active').focus()")
    ActionChains(drv).key_down(Keys.SHIFT).send_keys(Keys.ARROW_LEFT).key_up(Keys.SHIFT).perform()
    H.wait(drv, "return [...document.querySelectorAll('#doc-tabs .doc-tab .t')].map((t) => t.textContent).join() === 'Si.cif,MgO.cif'", 5)
    assert drv.execute_script("return document.activeElement.classList.contains('doc-tab') && document.activeElement.textContent.includes('Si.cif')")
    ActionChains(drv).send_keys(Keys.ARROW_RIGHT).perform()
    H.wait(drv, "return window.__mace.activeDoc().title === 'MgO.cif'", 5)
    # Move to Other Group from the tab's menu, by the keyboard (Shift+F10, the arrows, Enter)
    drv.execute_script("document.querySelector('#doc-tabs .doc-tab.active').focus()")
    t = drv.find_element(By.CSS_SELECTOR, "#doc-tabs .doc-tab.active")
    ActionChains(drv).context_click(t).perform()
    item = H.wait(drv, "return document.querySelector(\".menu .menu-item[data-cmd-id='tab.moveToOtherGroup']\")", 5)
    k = drv.execute_script("const m = [...document.querySelectorAll('.menu')].pop(); "
                           "return [...m.querySelectorAll('.menu-item:not([disabled])')].findIndex((b) => b.dataset.cmdId === 'tab.moveToOtherGroup')")
    assert item is not None and k >= 0
    ac = ActionChains(drv)
    focused_k = drv.execute_script("const m = [...document.querySelectorAll('.menu')].pop(); const items = [...m.querySelectorAll('.menu-item:not([disabled])')];"
                                   " return items.indexOf(document.activeElement)")
    for _ in range(k - max(0, focused_k)):
        ac.send_keys(Keys.ARROW_DOWN)
    ac.send_keys(Keys.ENTER).perform()
    wait_side(drv, "return p.docs().map((d) => d.title).join() === 'MgO.cif'", 60)
    assert drv.execute_script("return document.getElementById('doc-tabs').getAttribute('aria-label')") == "Left editor group: open documents"
    assert side(drv, "return W.document.getElementById('doc-tabs').getAttribute('aria-label')") == "Right editor group: open documents"
    assert drv.execute_script("return document.querySelector('.group-frame').getAttribute('aria-label')") == "Right editor group"
    # in the right group: Move Tab Left / Right have nothing to do with one tab; Move to Other Group brings it back
    side(drv, "M.run('tab.moveToOtherGroup')")
    H.wait(drv, "return !window.__maceGroups.isOpen() && window.__macePane.docs().length === 2", 30)
    assert drv.execute_script("return document.getElementById('doc-tabs').getAttribute('aria-label')") == "Open documents"
    no_errors(drv)


def test_the_right_group_has_the_windows_theme(fresh):
    drv = fresh
    drv.execute_script("window.__mace.settings.set('theme', 'dark')")
    try:
        open_main(drv, "structures/MgO.cif")
        open_side(drv, "structures/Si.cif")
        theme = drv.execute_script("return document.documentElement.getAttribute('data-theme')")
        wait_side(drv, f"return W.document.documentElement.getAttribute('data-theme') === {theme!r}", 10)
        if not side(drv, "return typeof p.adoptSettings === 'function'"):
            pytest.skip("this build does not pass a later settings change to the right group")
        drv.execute_script("window.__mace.settings.set('theme', 'light')")
        theme = drv.execute_script("return document.documentElement.getAttribute('data-theme')")
        wait_side(drv, f"return W.document.documentElement.getAttribute('data-theme') === {theme!r}", 10)
    finally:
        drv.execute_script("window.__mace.settings.set('theme', 'light')")
