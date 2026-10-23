"""Change preferences, close the tab, stop the server, start it again on
another port, open a fresh browser: everything comes back (polish P7, the
acceptance test of Marcus's request: "restart the server on a different port
and a fresh browser profile -> everything restored").

The last change (UI scale) is made right before the tab is left, with no
explicit flush: the page's leave-flush has to carry it. The server is then
stopped with SIGTERM (and, in the second variant, SIGINT as Ctrl+C does).
Unsaved drafts and Zen Mode stay with the first browser by design. Repo data
only (examples/gui), so it runs in the CI smoke job; the restored side runs in
Firefox and in Chromium."""
import json
import signal
import time
import urllib.request

import pytest

import guiharness as H

pytestmark = [pytest.mark.gui, pytest.mark.guismoke]
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")                 # the server reads the structures with it

from selenium.webdriver.common.action_chains import ActionChains  # noqa: E402


def click(drv, el):
    """A real click; given a CSS selector, the element is found again if a re-render replaced it."""
    from selenium.common.exceptions import StaleElementReferenceException
    for attempt in range(3):
        target = css(drv, el) if isinstance(el, str) else el
        try:
            drv.execute_script("arguments[0].scrollIntoView({block: 'center'})", target)
            ActionChains(drv).move_to_element(target).pause(0.05).click().perform()
            return
        except StaleElementReferenceException:
            if not isinstance(el, str):
                raise
            time.sleep(0.2)


def css(drv, sel):
    return drv.find_element("css selector", sel)


def open_doc(drv, path):
    return drv.execute_async_script("const d = arguments[1]; window.__mace.open(arguments[0]).then((x) => d(x && x.kind))", path)


def settings_click(drv, key, val):
    """A real click on a Settings dialog's segmented choice."""
    drv.execute_script("window.__mace.run('file.settings')")
    H.wait(drv, "return !!document.querySelector('dialog#settings[open]')", 10)
    sec = drv.execute_script("""const r = document.querySelector(`#settings [data-row="${arguments[0]}"]`); return !!r""", key)
    if not sec:
        drv.execute_script("document.querySelector('#settings [data-sec=appearance]').click()")
    click(drv, css(drv, f'#settings button[data-key="{key}"][data-val="{val}"]'))
    time.sleep(0.1)


def close_dialogs(drv):
    drv.execute_script("document.querySelectorAll('dialog[open]').forEach((d) => d.close())")


def first_paint(base):
    """The <html> tag the server sends (what the browser paints first)."""
    req = urllib.request.Request(base, headers={"X-Mace-Token": H.TOKEN})
    with urllib.request.urlopen(req, timeout=20) as r:
        import re
        return re.search(r"<html[^>]*>", r.read().decode()).group(0)


@pytest.mark.parametrize("stop,browser_b", [(signal.SIGTERM, "firefox"), (signal.SIGINT, "chromium")],
                         ids=["sigterm-firefox", "sigint-chromium"])
def test_restart_on_another_port_restores_everything(tmp_path, stop, browser_b):
    root, xdg = H.examples_copy(tmp_path), tmp_path / "xdg"
    proc, base, _ = H.start_server(root, xdg)
    drv = H.firefox(1400, 900)
    try:
        H.load(drv, base)
        # theme Dark with the menu bar's theme button (cycles system > light > dark)
        for _ in range(3):
            if drv.execute_script("return window.__mace.settings.get('theme')") == "dark":
                break
            click(drv, css(drv, "#theme-btn"))
            time.sleep(0.15)
        assert drv.execute_script("return window.__mace.settings.get('theme')") == "dark"
        settings_click(drv, "density", "compact")
        drv.execute_script("document.querySelector('#settings [data-sec=units]').click()")
        energy = css(drv, '#settings select[data-key="energyUnit"]')
        from selenium.webdriver.support.ui import Select
        Select(energy).select_by_value("Ha")
        drv.execute_script("document.querySelector('#settings [data-sec=viewport]').click()")
        drv.execute_script("const r = document.querySelector('#settings input[data-key=atomScale]'); r.value = 0.5; r.dispatchEvent(new Event('change', {bubbles: true}))")
        close_dialogs(drv)
        # the explorer wider (a real drag of its splitter)
        sp = css(drv, "#split-explorer")
        ActionChains(drv).move_to_element(sp).click_and_hold().move_by_offset(60, 0).pause(0.1).release().perform()
        time.sleep(0.3)
        w_explorer = drv.execute_script("return parseInt(getComputedStyle(document.getElementById('shell')).getPropertyValue('--w-explorer'))")
        assert w_explorer > 280
        # two Recent files, one pinned; MgO at 2x2x2 (its own range); the Symmetry section collapsed
        assert open_doc(drv, "structures/Si.cif") == "structure"
        assert open_doc(drv, "structures/MgO.cif") == "structure"
        # 1.1.6: the Range section's N x M x L picker (its presets) replaced the 2x2x2 button
        H.wait(drv, "return !!document.querySelector('#inspector .cells-preset')", 10)
        # (picked by script: a toast of the opens above may cover the select at the window's bottom right)
        drv.execute_script("""const s = document.querySelector('#inspector .cells-preset'); s.value = '2,2,2';
          s.dispatchEvent(new Event('change', {bubbles: true}))""")
        H.wait(drv, "return document.getElementById('tb-range-label').textContent === '2×2×2'", 5)
        click(drv, '#inspector .insp-section[data-key="symmetry"] .insp-head')
        drv.execute_script("window.__mace.run('file.recent.pin', {path: 'structures/Si.cif'})")
        # the Clean method is remembered (Springs is always available)
        drv.execute_script("window.__mace.profile.remember('cleanMethod', 'springs')")
        # an unsaved edit (a draft: stays in this browser)
        drv.execute_script("window.__mace.editor.select([0]); window.__mace.run('edit.delete')")
        H.wait(drv, "return window.__mace.docs().some((d) => d.dirty)", 10)
        # the Tasks tab in the bottom panel (a real click), then Zen Mode (this browser only)
        click(drv, '.bottom-tab[data-bottom="tasks"]')
        drv.execute_script("window.__mace.run('view.zen')")
        time.sleep(0.8)                              # the debounced saves of the changes above
        # the LAST change, then leave at once: no flush, no wait for the 500 ms debounce
        drv.execute_script("window.__mace.settings.set('uiScale', 110)")
        drv.execute_script("window.onbeforeunload = null; addEventListener('beforeunload', (e) => e.stopImmediatePropagation(), true);")
        drv.get("about:blank")
    finally:
        drv.quit()
    # the server drains the writes it has received; the page's pagehide keepalive may still be on its
    # way when the browser quits: give it up to 3 s to arrive (the app sends it again on the next load)
    end = time.time() + 3
    while time.time() < end:
        try:
            if json.loads((xdg / "mace" / "gui.json").read_text())["settings"].get("uiScale") == 110:
                break
        except (OSError, ValueError, KeyError):
            pass
        time.sleep(0.1)
    proc.send_signal(stop)
    proc.wait(timeout=20)
    stored = json.loads((xdg / "mace" / "gui.json").read_text())
    assert stored["settings"].get("uiScale") == 110, stored["settings"]

    proc, base, _ = H.start_server(root, xdg, port=H.free_port())
    drv = H.chromium(1400, 900) if browser_b == "chromium" else H.firefox(1400, 900)
    try:
        html = first_paint(base)
        assert 'data-theme="dark"' in html and "--ui-scale:1.1" in html and 'data-density="compact"' in html, html
        H.load(drv, base)
        g = lambda k: drv.execute_script("return window.__mace.settings.get(arguments[0])", k)  # noqa: E731
        assert (g("theme"), g("uiScale"), g("density"), g("energyUnit"), g("atomScale")) == ("dark", 110, "compact", "Ha", 0.5)
        assert drv.execute_script("return document.documentElement.getAttribute('data-theme')") == "dark"
        w = drv.execute_script("return parseInt(getComputedStyle(document.getElementById('shell')).getPropertyValue('--w-explorer'))")
        assert w == w_explorer
        rec = drv.execute_script("return window.__mace.profile.recentFiles().map((x) => [x.path, x.pinned])")
        assert rec[:2] == [["structures/Si.cif", True], ["structures/MgO.cif", False]], rec
        assert drv.execute_script("return window.__mace.profile.memory('cleanMethod')") == "springs"
        assert drv.execute_script("return document.querySelector('.bottom-tab.active').dataset.bottom") == "tasks"
        # MgO opens at its own 2x2x2, with the override dot; the Symmetry section stays collapsed
        assert open_doc(drv, "structures/MgO.cif") == "structure"
        H.wait(drv, "return document.getElementById('tb-range-label').textContent === '2×2×2'", 10)
        assert drv.execute_script("return !!document.querySelector('#inspector [data-unset=range]')")
        assert "collapsed" in css(drv, '#inspector .insp-section[data-key="symmetry"]').get_attribute("class")
        # by design not here: the draft (this browser only) and Zen Mode
        assert not drv.execute_script("return document.getElementById('shell').classList.contains('zen')")
        assert drv.execute_script("const b = document.getElementById('draft-banner'); return !b || b.hidden")
        assert not H.errors(drv), H.errors(drv)
    finally:
        drv.quit()
        proc.terminate()
        proc.wait(timeout=15)
