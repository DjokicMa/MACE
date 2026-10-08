"""Copy Diagnostics and Report a Problem with real clicks in a fresh headless
Firefox: the Help menu prefetches the bundle, so Copy Diagnostics copies at
once; a click before the bundle is ready (the route delayed by a test hook)
still copies or opens the dialog with the text selected; Save as File
downloads mace-gui-diagnostics.txt; About has the same button. Plus a
smoke of Copy Diagnostics then Sign Out in Chromium and Firefox."""
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

import guiharness as H

pytestmark = pytest.mark.gui
REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "test"
HEADER = "MACE GUI diagnostics. Paths, file and folder names, tokens, user and host names were removed"


def start(tmp, **env):
    root = tmp / "calc"
    root.mkdir(exist_ok=True)
    src = CORPUS / "OPT" / "1_dia_opt_rev1.out"
    if src.is_file():
        shutil.copy(src, root / src.name)
    e = H.env(tmp / "xdg", MACE_GUI_NO_WARMUP="1", MACE_GUI_TEST_HOOKS="1", **env)
    p = H.ServerProcess([sys.executable, str(H.MACE_CLI), "--no-banner", "gui", "--dir", str(root), "--port", "0",
                         "--token", H.TOKEN], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True, env=e, start_new_session=True,
                        preexec_fn=H.die_with_pytest)
    for _ in range(40):
        m = re.search(H.URL_RE, p.stdout.readline())
        if m:
            return p, m.group(1)
    raise AssertionError("no URL")


def menu_item(drv, top, cmd, timeout=15):
    """The visible item ``cmd`` of the menu-bar menu ``top``, opening that
    menu as often as it takes. A menu closes when the window loses focus (the
    window's blur, as native menus do), which a busy machine running several
    headless browsers can cause at any moment, so a single click on the menu
    title followed by a long wait for the item was flaky; this waits for the
    item and opens the menu again whenever it is not open."""
    find = "const e = document.querySelector(`.menu-item[data-cmd-id='${arguments[0]}']`); return e && e.offsetParent ? e : null"
    is_open = ("const b = [...document.querySelectorAll('.menu-top')].find(x => x.textContent === arguments[0]); "
               "return !!b && b.getAttribute('aria-expanded') === 'true' && !!document.querySelector(`.menu[data-top='${b.dataset.menu}']`)")
    end = time.time() + timeout
    while time.time() < end:
        e = drv.execute_script(find, cmd)
        if e:
            return e
        if not drv.execute_script(is_open, top):
            [b for b in drv.find_elements("css selector", ".menu-top") if b.text == top][0].click()
        time.sleep(0.3)
    raise AssertionError(f"menu item {cmd} of {top} not shown within {timeout} s")


def click_menu_item(drv, top, cmd):
    """Click ``cmd`` in menu ``top``; an open menu is rebuilt when the window's
    state changes, so a stale item is looked up again."""
    from selenium.common.exceptions import StaleElementReferenceException
    for attempt in range(5):
        try:
            menu_item(drv, top, cmd).click()
            return
        except StaleElementReferenceException:
            if attempt == 4:
                raise
            time.sleep(0.2)


def help_item(drv, cmd):
    return menu_item(drv, "Help", cmd)


def test_copy_after_prefetch_and_report_save(tmp_path):
    p, base = start(tmp_path)
    dl = tmp_path / "downloads"
    dl.mkdir()
    drv = H.firefox(download_dir=dl)
    try:
        H.load(drv, base)
        item = help_item(drv, "help.copyDiagnostics")
        time.sleep(1.5)                                       # the prefetch the menu started
        item.click()
        copied = H.wait(drv, "return window.__maceLastCopy", 10)
        assert copied.startswith(HEADER)
        assert H.TOKEN not in copied and "MACE: " in copied and "== Browser" in copied
        assert H.wait(drv, "return [...document.querySelectorAll('#toasts .toast2')].some(t => "
                           "t.textContent.includes('Diagnostics copied'))", 5)
        help_item(drv, "help.report").click()
        H.wait(drv, "const t = document.getElementById('diag-text'); return t && t.value.startsWith('MACE GUI diagnostics')", 15)
        drv.find_element("id", "diag-save").click()
        f = dl / "mace-gui-diagnostics.txt"
        for _ in range(50):
            if f.exists() and f.stat().st_size:
                break
            time.sleep(0.2)
        assert f.read_text().startswith(HEADER)
        assert drv.find_element("id", "diag-issue").get_attribute("href").endswith("/issues/new")
        # About has the button too
        drv.execute_script("document.querySelectorAll('dialog[open]').forEach(d => d.close()); window.__maceLastCopy = null")
        help_item(drv, "help.about").click()
        time.sleep(1.0)
        drv.find_element("id", "about-diag").click()
        assert H.wait(drv, "return window.__maceLastCopy", 10).startswith(HEADER)
    finally:
        drv.quit()
        p.terminate()
        p.wait(15)


def test_click_before_the_bundle_is_ready(tmp_path):
    p, base = start(tmp_path, MACE_GUI_TEST_DIAG_DELAY="3")
    drv = H.firefox()
    try:
        H.load(drv, base)
        help_item(drv, "help.copyDiagnostics").click()          # at once: the prefetch is still waiting
        H.wait(drv, """const t = document.getElementById('diag-text');
            return (window.__maceLastCopy && window.__maceLastCopy.startsWith('MACE GUI diagnostics'))
                || (t && t.value.startsWith('MACE GUI diagnostics') && document.activeElement === t
                    && t.selectionEnd - t.selectionStart === t.value.length)""", 20)
    finally:
        drv.quit()
        p.terminate()
        p.wait(15)


@pytest.mark.parametrize("browser", ["chromium", "firefox"])
def test_smoke_sign_out_after_copy(tmp_path, browser):
    make = getattr(H, browser, None)
    if make is None:
        pytest.skip(f"no {browser} helper in guiharness")
    p, base = start(tmp_path)
    drv = make()
    try:
        H.load(drv, base)
        item = help_item(drv, "help.copyDiagnostics")
        time.sleep(1.5)
        item.click()
        assert H.wait(drv, "return window.__maceLastCopy", 10).startswith(HEADER)
        # the window losing focus closes an open menu (what made this test flaky
        # on a busy machine): the File menu is opened again
        menu_item(drv, "File", "file.signOut")
        drv.execute_script("window.dispatchEvent(new Event('blur'))")
        assert not drv.execute_script("return document.querySelectorAll('.menu').length")
        click_menu_item(drv, "File", "file.signOut")
        H.wait(drv, "const d = document.getElementById('sign-out'); return d && d.open", 10)
        drv.find_element("id", "so-now").click()
        H.wait(drv, "return !!document.getElementById('stopped-page')", 10)
        p.wait(10)
        assert p.returncode == 0
    finally:
        drv.quit()
        if p.poll() is None:
            p.terminate()
            p.wait(15)
