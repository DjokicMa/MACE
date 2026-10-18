"""Expand all / Collapse all for the right-hand panel (1.1.6).

- the Inspector's header has Expand All Sections and Collapse All Sections
  buttons (labeled, aria-expanded kept on every section head); real clicks,
  the default keys Ctrl+Alt+] / Ctrl+Alt+[, and Shift-click on a section
  head open or shut every section; the state is remembered across a reload;
- Calculation Setup has the same two buttons in its header, and the keys
  reach it while it replaces the Inspector.

The sharp 3D canvas at a UI scale above 100 % is tested once, with 1.1.5's
UI-scale tests (test_ui_scale_pointer_e2e.py).
"""
import time

import pytest

import guiharness as H

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")                 # the server reads the structures with it

from selenium.webdriver.common.action_chains import ActionChains  # noqa: E402
from selenium.webdriver.common.keys import Keys  # noqa: E402

SECTIONS = """return [...document.querySelectorAll('#inspector .insp-section')].map((s) =>
  [s.dataset.key, s.classList.contains('collapsed'), s.querySelector('.insp-head').getAttribute('aria-expanded')])"""
SETUP_SECTIONS = """return [...document.querySelectorAll('#inspector .su-sec')].filter((s) => s.dataset.sec !== 'Advanced').map((s) =>
  [s.dataset.sec, s.classList.contains('collapsed'), s.querySelector('.su-sec-h').getAttribute('aria-expanded')])"""


@pytest.fixture(scope="module", params=["chromium", "firefox"])
def app(request, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("sections")
    root = H.examples_copy(tmp)
    proc, base, _ = H.start_server(root, tmp / "xdg")
    try:
        drv = H.browser(request.param, width=1400, height=900)
    except BaseException:
        proc.terminate()
        proc.wait(timeout=15)
        raise
    try:
        H.load(drv, base)
        yield drv, base
    finally:
        drv.quit()
        proc.terminate()
        proc.wait(timeout=15)


@pytest.fixture
def drv(app, request):
    d, _ = app
    failed = request.session.testsfailed
    yield d
    if request.session.testsfailed > failed:
        H.save_failure(d, request.node.name)
    d.execute_script("if (window.__mace.ctx().setupOpen) window.__mace.run('calc.setup');"
                     "window.__mace.closeAll(); window.__mace.settings.set('uiScale', 100)")
    time.sleep(0.3)


def open_doc(drv, path):
    kind = drv.execute_async_script("const d = arguments[1]; window.__mace.open(arguments[0]).then((x) => d(x && x.kind))", path)
    time.sleep(0.6)
    return kind


def click(drv, css, *mods):
    el = drv.find_element("css selector", css)
    ac = ActionChains(drv)
    for m in mods:
        ac.key_down(m)
    ac.move_to_element(el).click()
    for m in reversed(mods):
        ac.key_up(m)
    ac.perform()
    time.sleep(0.2)


def keys(drv, *combo):
    ac = ActionChains(drv)
    for k in combo[:-1]:
        ac.key_down(k)
    ac.send_keys(combo[-1])
    for k in reversed(combo[:-1]):
        ac.key_up(k)
    ac.perform()
    time.sleep(0.3)


def all_state(rows):
    """True when every section is open, False when every one is shut, else None."""
    shut = {r[1] for r in rows}
    assert all(r[2] == ("false" if r[1] else "true") for r in rows), rows      # aria-expanded follows
    return None if len(shut) != 1 else not shut.pop()


def test_inspector_expand_and_collapse_all(app, drv):
    _, base = app
    assert open_doc(drv, "structures/MgO.cif") == "structure"
    H.wait(drv, "return document.querySelectorAll('#inspector .insp-section').length >= 4", 10)
    labels = drv.execute_script("return [...document.querySelectorAll('#inspector .insp-bar button')].map((b) => b.getAttribute('aria-label'))")
    assert labels == ["Expand All Sections", "Collapse All Sections", "Hide Inspector (Ctrl+Alt+B)"]   # the × last, in the bar
    # the commands: in the registry with their keys, and in View › Inspector Sections
    cmds = {c["id"]: c["keys"] for c in drv.execute_script("return window.__mace.commands")}
    assert cmds["view.expandSections"] == ["Mod+Alt+]"] and cmds["view.collapseSections"] == ["Mod+Alt+["]
    view = next(m for m in drv.execute_script("return window.__mace.menus()") if m["label"] == "View")
    assert {"label": "Inspector Sections"} in view["items"]

    click(drv, "#inspector .insp-bar [data-all='shut']")
    assert all_state(drv.execute_script(SECTIONS)) is False
    click(drv, "#inspector .insp-bar [data-all='open']")
    assert all_state(drv.execute_script(SECTIONS)) is True

    # the keys, from the viewport (the page, not a text field, has the focus)
    drv.find_element("css selector", "#viewer-main").click()
    keys(drv, Keys.CONTROL, Keys.ALT, "[")
    assert all_state(drv.execute_script(SECTIONS)) is False
    keys(drv, Keys.CONTROL, Keys.ALT, "]")
    assert all_state(drv.execute_script(SECTIONS)) is True

    # a plain click still toggles one; Shift-click toggles all the way that one goes
    click(drv, "#inspector .insp-section[data-key='cell'] .insp-head")
    rows = drv.execute_script(SECTIONS)
    assert all_state(rows) is None and dict((r[0], r[1]) for r in rows)["cell"] is True
    click(drv, "#inspector .insp-section[data-key='symmetry'] .insp-head", Keys.SHIFT)
    assert all_state(drv.execute_script(SECTIONS)) is False

    # remembered: shut after a reload
    time.sleep(0.8)
    H.load(drv, base)
    assert open_doc(drv, "structures/MgO.cif") == "structure"
    H.wait(drv, "return document.querySelectorAll('#inspector .insp-section').length >= 4", 10)
    assert all_state(drv.execute_script(SECTIONS)) is False
    click(drv, "#inspector .insp-bar [data-all='open']")
    assert all_state(drv.execute_script(SECTIONS)) is True
    assert not H.errors(drv), H.errors(drv)


def test_one_section_has_a_bar_without_the_section_buttons(drv):
    drv.execute_script("window.__mace.run('view.database')")
    H.wait(drv, "const d = window.__mace.activeDoc(); return d && d.kind === 'db'", 20)
    time.sleep(0.3)
    assert drv.execute_script("return document.querySelectorAll('#inspector .insp-section').length") == 1
    # the header (title and the hide ×) is always there; Expand/Collapse All only with two or more sections
    assert drv.execute_script("return !!document.querySelector('#inspector > .insp-bar .panel-hide')")
    assert not drv.execute_script("return !!document.querySelector('#inspector .insp-bar [data-all]')")
    assert drv.execute_script("return window.__mace.menuReason('view.expandSections')")


def test_setup_expand_and_collapse_all(drv):
    assert open_doc(drv, "structures/Si.cif") == "structure"
    drv.execute_script("window.__mace.run('calc.setup')")
    H.wait(drv, "return document.querySelectorAll('#inspector .su-sec').length >= 3", 30)
    H.wait(drv, "return !!document.querySelector('#inspector .su-head [data-act=collapseAll]')", 10)
    click(drv, "#inspector .su-head [data-act='collapseAll']")
    assert all_state(drv.execute_script(SETUP_SECTIONS)) is False
    click(drv, "#inspector .su-head [data-act='expandAll']")
    assert all_state(drv.execute_script(SETUP_SECTIONS)) is True
    # the keys reach Setup while it replaces the Inspector
    drv.find_element("css selector", "#viewer-main").click()
    keys(drv, Keys.CONTROL, Keys.ALT, "[")
    assert all_state(drv.execute_script(SETUP_SECTIONS)) is False
    # Shift-click on one section head opens them all
    click(drv, "#inspector .su-sec:not([data-sec='Advanced']) .su-sec-h", Keys.SHIFT)
    assert all_state(drv.execute_script(SETUP_SECTIONS)) is True
    assert not H.errors(drv), H.errors(drv)
