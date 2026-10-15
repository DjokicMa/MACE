"""Two editor-group bugs (Firefox and Chromium, repo data only):

- a document still opening to the side took the focus back to the right
  group after the user had moved it to the left one;
- a tab dragged once left its text selected, and the next drag from it
  started the browser's own drag (Chromium), which canceled the GUI's: the
  tab never reached the right half that splits the editor."""
import time

import pytest

import guiharness as H
import p4util as U

pytestmark = [pytest.mark.gui, pytest.mark.guismoke]
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")
from selenium.webdriver.common.actions.action_builder import ActionBuilder  # noqa: E402

W, HT = 1500, 900


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("split_bugs")
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
        yield drv
    finally:
        drv.quit()


@pytest.fixture
def fresh(browser):
    drv = browser
    drv.execute_script("const g = window.__maceGroups; if (g && g.isOpen()) g.closeGroup(true)")
    H.wait(drv, "return !window.__maceGroups.isOpen()", 10)
    U.reset(drv, W, HT)
    drv.execute_script("window.__macePane.docs().forEach((d) => window.__macePane.close(d.id, true))")
    return drv


def open_main(drv, path):
    drv.execute_async_script("const done = arguments[1]; window.__mace.open(arguments[0]).then(() => done(true), () => done(false))", path)
    H.wait(drv, "const d = window.__mace.activeDoc(); return d && d.path === arguments[0]", 60, path)


def tab_center(drv, title):
    return drv.execute_script(
        "const t = [...document.querySelectorAll('#doc-tabs .doc-tab')].find((x) => x.querySelector('.t').textContent === arguments[0]);"
        "const r = t.getBoundingClientRect(); return [r.left + r.width / 2, r.top + r.height / 2]", title)


def drag(drv, a, b, steps=12):
    ab = ActionBuilder(drv)
    p = ab.pointer_action
    p.move_to_location(int(a[0]), int(a[1]))
    p.pointer_down()
    for i in range(1, steps + 1):
        p.move_to_location(int(a[0] + (b[0] - a[0]) * i / steps), int(a[1] + (b[1] - a[1]) * i / steps))
    p.pause(0.25)
    p.pointer_up()
    ab.perform()
    time.sleep(0.3)


def test_a_slow_open_to_the_side_leaves_the_focus_where_the_user_moved_it(fresh):
    drv = fresh
    open_main(drv, "structures/MgO.cif")
    # the right group is up; the next document opens slowly there (its read held back)
    drv.execute_async_script("const done = arguments[0]; window.__maceGroups.openToSide('structures/Si.cif').then(() => done(true))")
    drv.execute_script(
        "const W = document.querySelector('.group-frame').contentWindow, p = W.__macePane, orig = p.reopen;"
        "window.__release = null; const gate = new Promise((r) => { window.__release = r; });"
        "p.reopen = (x) => gate.then(() => orig(x));"
        "window.__opened = window.__maceGroups.openToSide('structures/NaCl.cif').then(() => true)")
    H.wait(drv, "return window.__maceGroups.state().focused === 'side'", 10)
    # meanwhile the user moves the focus to the left group (Focus Other Group)
    drv.execute_script("window.__mace.run('view.focusOtherGroup')")
    H.wait(drv, "return window.__maceGroups.state().focused === 'main'", 10)
    drv.execute_script("window.__release()")
    drv.execute_async_script("const done = arguments[0]; window.__opened.then(done)")
    time.sleep(0.3)
    H.wait(drv, "const f = document.querySelector('.group-frame'); const a = f.contentWindow.__macePane.active(); return a && a.path === 'structures/NaCl.cif'", 30)
    assert drv.execute_script("return window.__maceGroups.state().focused") == "main"
    assert not [e for e in H.errors(drv) if "ResizeObserver loop" not in e]


def test_a_second_drag_of_a_tab_still_splits_the_editor(fresh):
    drv = fresh
    open_main(drv, "structures/MgO.cif")
    open_main(drv, "structures/Si.cif")
    # a first drag along the tab strip (it lands nowhere): it must not leave the tabs' text selected
    a = tab_center(drv, "Si.cif")
    drag(drv, a, tab_center(drv, "MgO.cif"))
    assert drv.execute_script("return String(getSelection()).trim()") == ""
    # the second drag, onto the right half of the editor, splits it
    views = drv.execute_script("const r = document.getElementById('views').getBoundingClientRect(); return [r.right, r.top + r.height / 2]")
    drag(drv, tab_center(drv, "Si.cif"), (views[0] - 40, views[1]))
    H.wait(drv, "const f = document.querySelector('.group-frame'); const p = f && f.contentWindow.__macePane;"
                " return p && p.docs().some((d) => d.path === 'structures/Si.cif')", 60)
    assert [d["path"] for d in drv.execute_script("return window.__macePane.docs()")] == ["structures/MgO.cif"]
    assert not [e for e in H.errors(drv) if "ResizeObserver loop" not in e]
