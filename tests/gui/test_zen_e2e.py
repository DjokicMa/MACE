"""Zen Mode always has a visible way out (polish A8): the chip at the top
right, Esc when nothing else wants it, the viewport menu, and the same after
a reload in Zen. Real clicks and real keys, in a fresh browser profile."""
import time

import pytest

import guiharness as H

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")                 # the server reads the structures with it

from selenium.webdriver.common.action_chains import ActionChains  # noqa: E402
from selenium.webdriver.common.keys import Keys  # noqa: E402

ZEN = "return document.getElementById('shell').classList.contains('zen')"
CHROME_HIDDEN = """return ['#menubar', '#toolbar', '#activity', '#statusbar'].every((s) => {
  const el = document.querySelector(s); return !el || getComputedStyle(el).display === 'none'; })"""
CHIP = "const c = document.getElementById('zen-exit'); return !!c && !c.hidden && c.getBoundingClientRect().width > 0"


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("zen")
    proc, base, _ = H.start_server(H.examples_copy(tmp), tmp / "xdg")
    drv = H.firefox(1280, 800)
    try:
        H.load(drv, base)
        yield drv, base
    finally:
        drv.quit()
        proc.terminate()
        proc.wait(timeout=15)


@pytest.fixture
def fresh(app):
    drv = app[0]
    drv.execute_script("window.__mace.closeAll(); document.querySelectorAll('dialog[open]').forEach((d) => d.close());"
                       "if (document.getElementById('shell').classList.contains('zen')) window.__mace.run('view.zen');"
                       "document.querySelectorAll('#toasts > *').forEach((t) => t.remove());")
    drv.execute_async_script("const done = arguments[0]; window.__mace.open('structures/MgO.cif').then(() => done(1))")
    time.sleep(0.5)
    drv.execute_script("document.body.focus()")
    return app


def key(drv, *combo):
    ac = ActionChains(drv)
    for k in combo[:-1]:
        ac.key_down(k)
    ac.send_keys(combo[-1])
    for k in reversed(combo[:-1]):
        ac.key_up(k)
    ac.perform()
    time.sleep(0.2)


def enter_by_menu(drv):
    top = [b for b in drv.find_elements("css selector", ".menu-top") if b.text == "View"][0]
    ActionChains(drv).move_to_element(top).click().perform()
    item = H.wait(drv, "return document.querySelector('.menu [data-cmd-id=\"view.zen\"]')", 3)
    # the View menu is taller than this window since 1.1.6: it scrolls, so bring the item into view
    drv.execute_script("arguments[0].scrollIntoView({block: 'nearest'})", item)
    w = item.size["width"]
    ActionChains(drv).move_to_element_with_offset(item, int(-w / 2 + 16), 0).pause(0.05).click().perform()
    H.wait(drv, ZEN, 3)


def test_enter_by_menu_and_leave_by_chip(fresh):
    drv = fresh[0]
    enter_by_menu(drv)
    assert drv.execute_script(CHROME_HIDDEN)
    assert drv.execute_script(CHIP)
    # the hint says how to leave, with the effective shortcut
    toast = H.wait(drv, "return [...document.querySelectorAll('#toasts > *')].map((t) => t.textContent).join(' ')", 3)
    assert "Esc" in toast and "Zen Mode" in toast
    ActionChains(drv).move_to_element(drv.find_element("css selector", "#zen-exit")).click().perform()
    H.wait(drv, "return !document.getElementById('shell').classList.contains('zen')", 3)
    assert drv.execute_script("return document.getElementById('zen-exit').hidden")


def test_shortcut_in_and_esc_out(fresh):
    drv = fresh[0]
    key(drv, Keys.CONTROL, Keys.ALT, "z")
    assert drv.execute_script(ZEN)
    key(drv, Keys.ESCAPE)
    assert not drv.execute_script(ZEN)


def test_esc_clears_the_selection_first(fresh):
    drv = fresh[0]
    key(drv, Keys.CONTROL, Keys.ALT, "z")
    drv.execute_script("window.__mace.editor.select([0])")
    drv.execute_script("document.body.focus()")
    key(drv, Keys.ESCAPE)
    assert drv.execute_script(ZEN) and drv.execute_script("return window.__mace.editor.selection().length") == 0
    key(drv, Keys.ESCAPE)
    assert not drv.execute_script(ZEN)


def test_esc_in_a_dialog_closes_only_the_dialog(fresh):
    drv = fresh[0]
    key(drv, Keys.CONTROL, Keys.ALT, "z")
    drv.execute_script("window.__mace.run('file.settings')")
    H.wait(drv, "return !!document.querySelector('dialog#settings[open]')", 10)
    key(drv, Keys.ESCAPE)
    H.wait(drv, "return !document.querySelector('dialog#settings[open]')", 3)
    assert drv.execute_script(ZEN)
    # the palette still opens in Zen
    key(drv, Keys.F1)
    H.wait(drv, "return window.__mace.palette.isOpen()", 3)
    key(drv, Keys.ESCAPE)
    assert drv.execute_script(ZEN)
    key(drv, Keys.ESCAPE)
    assert not drv.execute_script(ZEN)


def test_reload_in_zen_keeps_the_chip(fresh):
    drv, base = fresh[0], fresh[1]
    key(drv, Keys.CONTROL, Keys.ALT, "z")
    H.load(drv, base)
    assert drv.execute_script(ZEN) and drv.execute_script(CHIP)
    assert "dim" not in drv.execute_script("return document.getElementById('zen-exit').className")
    ActionChains(drv).move_to_element(drv.find_element("css selector", "#zen-exit")).click().perform()
    H.wait(drv, "return !document.getElementById('shell').classList.contains('zen')", 3)


def test_chip_fades_and_comes_back(fresh):
    drv = fresh[0]
    key(drv, Keys.CONTROL, Keys.ALT, "z")
    ActionChains(drv).move_by_offset(0, 0).perform()
    H.wait(drv, "return document.getElementById('zen-exit').classList.contains('dim')", 6)
    chip = drv.find_element("css selector", "#zen-exit")
    ActionChains(drv).move_to_element(chip).perform()
    H.wait(drv, "return getComputedStyle(document.getElementById('zen-exit')).opacity === '1'", 3)
    key(drv, Keys.ESCAPE)


def test_viewport_menu_leads_out(fresh):
    drv = fresh[0]
    key(drv, Keys.CONTROL, Keys.ALT, "z")
    canvas = drv.find_element("css selector", "#viewer-main canvas")
    ActionChains(drv).move_to_element_with_offset(canvas, -canvas.size["width"] // 2 + 10, -canvas.size["height"] // 2 + 40) \
        .context_click().perform()
    item = H.wait(drv, "return document.querySelector('.menu [data-cmd-id=\"view.exitZen\"]')", 3)
    w = item.size["width"]
    ActionChains(drv).move_to_element_with_offset(item, int(-w / 2 + 16), 0).pause(0.05).click().perform()
    H.wait(drv, "return !document.getElementById('shell').classList.contains('zen')", 3)
