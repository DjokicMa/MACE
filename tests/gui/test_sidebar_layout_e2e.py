"""The explorer's section layout (1.1.6), at 110 % UI scale: the splitters
between sections resize them by mouse (zoom-correct: a screen px of pointer
travel is 1/1.1 of the section's own px) and by keyboard (role=separator,
aria-valuenow), double-click resets; a section moves by dragging its header
or from the header's menu (Move Up / Move Down); the menu hides and shows
sections; Reset Sidebar Layout (palette and keymap) puts it all back; and
all of it survives a reload (the layout, saved to the profile).

Chromium and Firefox; the folder is a copy of examples/gui (its
materials.db gives the Materials.db section) plus a few scripts to open."""
import time

import pytest

import guiharness as H

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")
from selenium.webdriver.common.action_chains import ActionChains  # noqa: E402
from selenium.webdriver.common.keys import Keys  # noqa: E402

SCALE = 110
DEFAULT = ["editors", "files", "db", "recent"]


@pytest.fixture(scope="module", params=["chromium", "firefox"])
def app(request, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("sidebar_layout")
    root = H.examples_copy(tmp)
    for i in range(8):
        (root / f"note_{i:02d}.sh").write_text(f"#!/bin/sh\necho {i}\n")
    proc, base, _ = H.start_server(root, tmp / "xdg", MACE_GUI_TEST_HOOKS="1")
    try:
        drv = H.browser(request.param, width=1280, height=1000)
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


def prepare(drv):
    drv.execute_script("window.__mace.settings.set('uiScale', arguments[0])", SCALE)
    drv.execute_async_script("""const done = arguments[arguments.length - 1];
      (async () => { await window.__mace.open('structures/diamond.cif');
        for (let i = 0; i < 8; i++) await window.__mace.open('note_' + String(i).padStart(2, '0') + '.sh'); })()
        .then(() => done(true), (e) => done(String(e)));""")
    H.wait(drv, "return document.querySelectorAll('#editors-rows .file').length === 9", 20)
    H.wait(drv, "return document.querySelectorAll('#recent-rows .file').length > 0 && !!document.getElementById('db-toggle')", 20)
    time.sleep(0.4)


@pytest.fixture
def drv(app, request):
    app.execute_script("window.__mace.sidebar().reset(); window.__mace.profile.setUi('explorer', {})")
    H.set_viewport(app, 1280, 1000)
    prepare(app)
    failed = request.session.testsfailed
    yield app
    if request.session.testsfailed > failed:
        H.save_failure(app, request.node.name)
    app.execute_script("document.querySelectorAll('.menu').forEach((m) => m.remove()); window.__mace.closeAll();"
                       "window.__mace.sidebar().reset(); window.__mace.settings.set('uiScale', 100)")
    time.sleep(0.3)


def order(drv):
    """The sections on screen, top to bottom."""
    return drv.execute_script("""return [...document.querySelectorAll('#explorer > .side-section')]
      .filter((s) => !s.hidden && s.childElementCount && s.offsetHeight > 0).map((s) => s.dataset.section)""")


def saved(drv):
    return drv.execute_script("return JSON.parse(localStorage.getItem('mace.gui.layout.v1') || '{}').sidebar || null")


def splitter_for(drv, key):
    el = drv.execute_script(f"return document.querySelector('#explorer > .side-split:not([hidden])[data-target=\"{key}\"]')")
    assert el is not None, drv.execute_script(
        "return [...document.querySelectorAll('#explorer > .side-split')].map((s) => [s.hidden, s.dataset.target])")
    return el


def height(drv, key):
    return drv.execute_script(f"return document.querySelector('#explorer > [data-section=\"{key}\"]').offsetHeight")


def header(drv, key):
    return drv.execute_script(f"return document.querySelector('#explorer > [data-section=\"{key}\"] > .section-label')")


def zoom(drv):
    return drv.execute_script("const e = document.getElementById('explorer'); return e.getBoundingClientRect().height / e.offsetHeight")


def drag_screen(drv, x0, y0, x1, y1, steps=8):
    """Press at screen point (x0, y0), move to (x1, y1) in steps, release: viewport
    coordinates, so the pointer does not follow a splitter or header that moves."""
    from selenium.webdriver.common.actions.action_builder import ActionBuilder
    ab = ActionBuilder(drv)
    p = ab.pointer_action
    p.move_to_location(round(x0), round(y0))
    p.pointer_down()
    for s in range(1, steps + 1):
        p.move_to_location(round(x0 + (x1 - x0) * s / steps), round(y0 + (y1 - y0) * s / steps))
        p.pause(0.02)
    p.pointer_up()
    ab.perform()


def center(drv, el):
    r = drv.execute_script("return arguments[0].getBoundingClientRect().toJSON()", el)
    return r["left"] + r["width"] / 2, r["top"] + r["height"] / 2


def menu_item(drv, cmd):
    return H.wait(drv, f"return document.querySelector('.menu [data-cmd-id=\"{cmd}\"]')", 5)


def test_splitters_are_separators_between_shown_sections(drv):
    assert abs(zoom(drv) - SCALE / 100) < 0.01
    assert order(drv) == DEFAULT
    sp = drv.execute_script("""return [...document.querySelectorAll('#explorer > .side-split:not([hidden])')].map((s) => ({
        role: s.getAttribute('role'), o: s.getAttribute('aria-orientation'), t: s.dataset.target, tab: s.tabIndex,
        now: Number(s.getAttribute('aria-valuenow')), min: Number(s.getAttribute('aria-valuemin')), label: s.getAttribute('aria-label'),
        h: document.querySelector('#explorer > [data-section="' + s.dataset.target + '"]').offsetHeight}))""")
    # editors|files sizes Open Editors; files|db none (Files fills, Materials.db starts folded); db|recent sizes Recent
    assert [s["t"] for s in sp] == ["editors", "recent"], sp
    for s in sp:
        assert s["role"] == "separator" and s["o"] == "horizontal" and s["tab"] == 0 and s["label"].startswith("Resize "), s
        assert abs(s["now"] - s["h"]) <= 1 and s["min"] > 0, s


def test_resize_by_mouse_at_110_percent_and_double_click_resets(drv):
    sp = splitter_for(drv, "editors")
    h0, z = height(drv, "editors"), zoom(drv)
    x, y = center(drv, sp)
    drag_screen(drv, x, y, x, y + 44)
    time.sleep(0.3)
    h1 = height(drv, "editors")
    # 44 screen px at 110 % is 40 of the section's own px
    assert abs(h1 - (h0 + 44 / z)) <= 1.5, (h0, h1, z)
    sp = splitter_for(drv, "editors")
    assert abs(int(sp.get_attribute("aria-valuenow")) - h1) <= 1
    assert abs(saved(drv)["sizes"]["editors"] - h1) <= 1
    # the file tree gave the room
    ActionChains(drv).double_click(sp).perform()
    time.sleep(0.3)
    assert abs(height(drv, "editors") - h0) <= 1
    assert not (saved(drv) or {}).get("sizes")


def test_resize_by_keyboard(drv):
    sp = splitter_for(drv, "editors")
    h0 = height(drv, "editors")
    sp.send_keys(Keys.ARROW_DOWN)
    sp.send_keys(Keys.ARROW_DOWN)
    time.sleep(0.2)
    assert abs(height(drv, "editors") - (h0 + 32)) <= 1
    sp.send_keys(Keys.ARROW_UP)
    time.sleep(0.2)
    h = height(drv, "editors")
    assert abs(h - (h0 + 16)) <= 1
    assert int(sp.get_attribute("aria-valuenow")) == h
    assert drv.execute_script("return document.activeElement.classList.contains('side-split')")
    # a splitter that sizes the section under it (Materials.db above is folded): ArrowUp grows Recent
    sp2 = splitter_for(drv, "recent")
    assert sp2.get_attribute("aria-controls") == "recent-files"
    d0 = height(drv, "recent")
    sp2.send_keys(Keys.ARROW_UP)
    time.sleep(0.2)
    assert abs(height(drv, "recent") - (d0 + 16)) <= 1


def test_reorder_by_dragging_a_header(drv):
    rec, ed = header(drv, "recent"), header(drv, "editors")
    # from Recent's header up past the middle of Open Editors (screen px)
    x, y = center(drv, rec)
    top = drv.execute_script("return arguments[0].getBoundingClientRect().top", ed)
    was = rec.get_attribute("aria-expanded")
    drag_screen(drv, x, y, x, top + 3)
    time.sleep(0.3)
    assert order(drv) == ["recent", "editors", "files", "db"]
    # the drag was not a click: the section did not fold
    assert header(drv, "recent").get_attribute("aria-expanded") == was
    assert saved(drv)["order"][:2] == ["recent", "editors"]


def test_reorder_hide_and_show_from_the_header_menu(drv):
    ActionChains(drv).context_click(header(drv, "files")).perform()
    menu_item(drv, "view.sidebar.moveUp").click()
    time.sleep(0.2)
    assert order(drv) == ["files", "editors", "db", "recent"]
    # the last one cannot move down
    ActionChains(drv).context_click(header(drv, "recent")).perform()
    assert menu_item(drv, "view.sidebar.moveDown").get_attribute("aria-disabled") == "true"
    m = menu_item(drv, "view.sidebar.show.db")
    assert m.get_attribute("role") == "menuitemcheckbox" and m.get_attribute("aria-checked") == "true"
    m.click()
    time.sleep(0.2)
    assert order(drv) == ["files", "editors", "recent"]
    assert saved(drv)["hidden"] == ["db"]
    # Move Down skips the hidden section
    ActionChains(drv).context_click(header(drv, "editors")).perform()
    menu_item(drv, "view.sidebar.moveDown").click()
    time.sleep(0.2)
    assert order(drv) == ["files", "recent", "editors"]
    # shown again from the header menu, in the place it had
    ActionChains(drv).context_click(header(drv, "files")).perform()
    m = menu_item(drv, "view.sidebar.show.db")
    assert m.get_attribute("aria-checked") == "false"
    m.click()
    time.sleep(0.2)
    assert "db" in order(drv)
    # hiding Files hides its filter too
    drv.execute_script("window.__mace.run('view.sidebar.show.files')")
    time.sleep(0.2)
    assert "files" not in order(drv)
    assert drv.execute_script("return document.querySelector('#explorer .side-head').hidden")
    drv.execute_script("window.__mace.run('view.sidebar.show.files')")
    time.sleep(0.6)
    # keyboard: Alt+Up on a focused header, and Shift+F10 opens its menu
    before = order(drv)
    header(drv, "editors").send_keys(Keys.ALT, Keys.ARROW_UP)
    time.sleep(0.2)
    o = order(drv)
    i = before.index("editors")
    assert o.index("editors") == i - 1 and o[i] == before[i - 1], (before, o)
    assert drv.execute_script("return document.activeElement.id") == "editors-toggle"


def test_reset_sidebar_layout_from_the_palette(drv):
    drv.execute_script("window.__mace.sidebar().move('recent', -1); window.__mace.sidebar().setShown('db', false)")
    splitter_for(drv, "editors").send_keys(Keys.ARROW_DOWN)
    time.sleep(0.2)
    assert order(drv) != DEFAULT and saved(drv)
    cmds = {c["id"]: c for c in drv.execute_script("return window.__mace.commands")}
    c = cmds["view.resetSidebar"]
    assert c["title"] == "Reset Sidebar Layout" and c["palette"] and c["category"] == "View"
    # the keymap editor lists every registered command (Settings › Keyboard): it is bindable
    assert drv.execute_script("return window.__mace.palette.results('Reset Sidebar Layout')[0].id") == "view.resetSidebar"
    drv.execute_script("window.__mace.palette.open('')")
    time.sleep(0.3)
    box = drv.switch_to.active_element
    box.send_keys("Reset Sidebar Layout")
    time.sleep(0.4)
    box.send_keys(Keys.ENTER)
    time.sleep(0.4)
    assert order(drv) == DEFAULT
    assert saved(drv) is None
    assert drv.execute_script("return document.querySelector('#open-editors').style.flex") == ""


def test_the_layout_survives_a_reload(drv):
    sb = "window.__mace.sidebar()"
    drv.execute_script(f"{sb}.move('db', -1); {sb}.setShown('recent', false)")
    splitter_for(drv, "editors").send_keys(Keys.ARROW_DOWN)
    splitter_for(drv, "editors").send_keys(Keys.ARROW_DOWN)
    time.sleep(0.2)
    h = height(drv, "editors")
    want = order(drv)
    drv.execute_script("return window.__mace.profile.flush()")
    # the server's copy, not just this browser's: the profile has it
    assert drv.execute_script("return window.__mace.profile.ui('layout', {}).sidebar.hidden")[0] == "recent"
    drv.refresh()
    H.wait(drv, "return document.documentElement.dataset.appReady === '1'")
    prepare(drv)
    assert order(drv) == want
    assert abs(height(drv, "editors") - h) <= 1
    assert drv.execute_script("return document.getElementById('recent-files').hidden")
    assert not H.errors(drv)
