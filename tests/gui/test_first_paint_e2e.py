"""First paint from the server's profile in a browser that has been here
before (polish P3 e2e b and c), with real clicks, a fresh Firefox profile and
a server started through mace_cli.

A fresh browser is covered by test_profile_restart_e2e (first paint on a new
port) and test_shell_api::test_index_carries_server_prefs (the <html> tag,
no ETag, no 304). These two are the cases a fresh browser cannot catch:

(b) the same browser comes back after another tab changed the theme: the page
    is fetched again (200, never a cached copy) and paints the new theme;
(c) a change the server did not get (it answered 503) stays this browser's
    value on the next load, and reaches gui.json once the server takes it.

boot.js records what it painted in window.__maceFirstPaint before any module
runs, so "first paint" is read exactly, not raced against /api/boot."""
import json
import time
import urllib.request

import pytest

import guiharness as H

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")

from selenium.webdriver.common.action_chains import ActionChains  # noqa: E402


@pytest.fixture
def served(tmp_path):
    H.local_only("toggles a test hook and reads gui.json")
    root, xdg = H.examples_copy(tmp_path), tmp_path / "xdg"
    proc, base, _ = H.start_server(root, xdg, MACE_GUI_TEST_HOOKS="1")
    drv = H.firefox(1280, 860)
    try:
        yield drv, base, xdg
    finally:
        drv.quit()
        proc.terminate()
        proc.wait(timeout=15)


def post(base, route, body):
    req = urllib.request.Request(base + "api/" + route, data=json.dumps(body).encode(), method="POST",
                                 headers={"X-Mace-Token": H.TOKEN, "Content-Type": "application/json",
                                          "X-Requested-With": "mace-gui"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())


def click(drv, el):
    drv.execute_script("arguments[0].scrollIntoView({block: 'center'})", el)
    ActionChains(drv).move_to_element(el).pause(0.05).click().perform()


def first_paint(drv):
    return drv.execute_script("return window.__maceFirstPaint")


def settings_choice(drv, key, value):
    """Settings › Appearance: a real click on the segmented control's choice."""
    click(drv, drv.find_element("id", "settings-btn"))
    H.wait(drv, "return document.getElementById('settings') && document.getElementById('settings').open", 10)
    sel = f'#settings [data-key="{key}"][data-val="{value}"]'
    H.wait(drv, f"return !!document.querySelector('{sel}')", 10)
    click(drv, drv.find_element("css selector", sel))
    H.wait(drv, f"return document.querySelector('{sel}').getAttribute('aria-checked') === 'true'", 5)
    drv.execute_script("document.querySelectorAll('dialog[open]').forEach((d) => d.close())")


def nav_status(drv):
    """The page's own HTTP status and whether it came from the browser cache."""
    return drv.execute_script("""const n = performance.getEntriesByType('navigation')[0] || {};
        return {status: n.responseStatus === undefined ? null : n.responseStatus, transfer: n.transferSize}""")


def stored(xdg):
    p = xdg / "mace" / "gui.json"
    return json.loads(p.read_text())["settings"] if p.exists() else {}


def test_returning_browser_paints_what_another_tab_chose(served):
    drv, base, xdg = served
    H.load(drv, base)
    # tab 1: Dark with the menu bar's theme button (system > light > dark)
    for _ in range(3):
        if drv.execute_script("return window.__mace.settings.get('theme')") == "dark":
            break
        click(drv, drv.find_element("id", "theme-btn"))
        time.sleep(0.15)
    assert drv.execute_script("return window.__mace.settings.get('theme')") == "dark"
    H.wait(drv, "return !localStorage.getItem('mace.gui.settings.pending') || localStorage.getItem('mace.gui.settings.pending') === '[]'", 10)
    assert stored(xdg).get("theme") == "dark"
    drv.refresh()
    H.wait(drv, "return document.documentElement.dataset.appReady === '1'")
    assert first_paint(drv)["theme"] == "dark"
    tab1 = drv.current_window_handle
    # tab 2, same browser profile (shared localStorage): Light through the Settings dialog
    drv.switch_to.new_window("tab")
    H.load(drv, base)
    settings_choice(drv, "theme", "light")
    H.wait(drv, "return !localStorage.getItem('mace.gui.settings.pending') || localStorage.getItem('mace.gui.settings.pending') === '[]'", 10)
    assert stored(xdg).get("theme") == "light"
    drv.close()
    drv.switch_to.window(tab1)
    # tab 1 reloads: the page comes from the server again (200), and paints Light at once
    drv.refresh()
    H.wait(drv, "return document.documentElement.dataset.appReady === '1'")
    fp = first_paint(drv)
    assert fp["theme"] == "light" and fp["source"] == "server", fp
    st = nav_status(drv)
    if st["status"] is not None:
        assert st["status"] == 200, st
    assert not H.errors(drv), H.errors(drv)


def test_a_change_the_server_missed_wins_on_the_next_load(served):
    drv, base, xdg = served
    H.load(drv, base)
    post(base, "test/profile/fail", {"on": True})
    try:
        settings_choice(drv, "uiScale", "130")
        H.wait(drv, "return [...document.querySelectorAll('#toasts > *')].some((t) => /saved in this browser only/.test(t.textContent))", 10)
        assert "uiScale" not in stored(xdg)
        assert "uiScale" in json.loads(drv.execute_script("return localStorage.getItem('mace.gui.settings.pending')") or "[]")
        # reload while the server still refuses saves (the leave-flush cannot deliver it either):
        # the server's page says 100%, this browser's unsent 130% is what is painted
        drv.refresh()
        H.wait(drv, "return document.documentElement.dataset.appReady === '1'")
        assert "uiScale" not in stored(xdg)
        fp = first_paint(drv)
        assert fp["scale"] == "1.3" and fp["source"] == "server", fp
        assert drv.execute_script("return window.__mace.settings.get('uiScale')") == 130
    finally:
        post(base, "test/profile/fail", {"on": False})
    drv.refresh()
    H.wait(drv, "return document.documentElement.dataset.appReady === '1'")
    assert first_paint(drv)["scale"] == "1.3"
    # the leave-flush or the boot merge sends it again; the server then holds it
    t0 = time.time()
    while time.time() - t0 < 10 and stored(xdg).get("uiScale") != 130:
        time.sleep(0.2)
    assert stored(xdg).get("uiScale") == 130
    assert drv.execute_script("return window.__mace.settings.get('uiScale')") == 130
