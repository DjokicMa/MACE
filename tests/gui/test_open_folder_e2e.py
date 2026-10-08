"""File > Open Folder with real clicks in a fresh headless Firefox (temp
copies of real test/ files, a fake HOME): menu, root chip, folder, Open; the
Explorer lists the new folder's files; a modified document asks first and its
draft comes back in the first folder; the Explorer header button and the
command search; a second tab follows the switch."""
import shutil
import time
from pathlib import Path

import pytest

import guiharness as H

pytestmark = pytest.mark.gui
REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "test"
A_FILES = ["OPT/1_dia_opt_rev1.out", "CIFs/1_dia_opt_rev1.cif"]
B_FILES = ["OPT/1_dia_opt_BULK_OPTGEOM.out", "OPT/1_dia_opt_BULK_OPTGEOM.d12"]


@pytest.fixture
def setup(tmp_path):
    if not all((CORPUS / f).is_file() for f in A_FILES + B_FILES):
        pytest.skip("test/ corpus not present")
    home = tmp_path / "home"
    (home / ".ssh").mkdir(parents=True)
    a, b = home / "calcA", home / "calcB"
    a.mkdir()
    b.mkdir()
    for f in A_FILES:
        shutil.copy(CORPUS / f, a / Path(f).name)
    for f in B_FILES:
        shutil.copy(CORPUS / f, b / Path(f).name)
    import re
    import subprocess
    import sys
    e = H.env(tmp_path / "xdg", HOME=str(home), MACE_GUI_NO_WARMUP="1")
    proc = H.ServerProcess([sys.executable, str(H.MACE_CLI), "--no-banner", "gui", "--dir", str(a), "--port", "0",
                            "--token", H.TOKEN], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, env=e, start_new_session=True,
                           preexec_fn=H.die_with_pytest)
    base = None
    for _ in range(40):
        m = re.search(H.URL_RE, proc.stdout.readline())
        if m:
            base = m.group(1)
            break
    drv = H.firefox()
    try:
        yield drv, base, home, a, b
    finally:
        drv.quit()
        proc.terminate()
        proc.wait(15)


def explorer_paths(drv):
    return drv.execute_script("return [...document.querySelectorAll('#file-list .file')].map(e => e.dataset.path)")


def wait_explorer(drv, names, timeout=30):
    H.wait(drv, "return document.documentElement.dataset.appReady === '1'", 60)
    H.wait(drv, """const want = arguments[0]; const got = [...document.querySelectorAll('#file-list .file')]
        .map(e => e.dataset.path); return want.every(n => got.includes(n)) && got.length === want.length""",
           timeout, sorted(names))


def menu_item(drv, menu, cmd):
    [b for b in drv.find_elements("css selector", ".menu-top") if b.text == menu][0].click()
    el = H.wait(drv, "return document.querySelector(`.menu-item[data-cmd-id='${arguments[0]}']`)", 10, cmd)
    el.click()


def choose_in_dialog(drv, folder_name, chip="Home"):
    H.wait(drv, "const d = document.getElementById('open-folder'); return d && d.open && "
                "document.querySelectorAll('#of-roots [data-root]').length > 0", 15)
    [c for c in drv.find_elements("css selector", "#of-roots [data-root]") if c.text == chip][0].click()
    li = H.wait(drv, "return [...document.querySelectorAll('#of-list li[data-dir]')].find(l => l.dataset.dir === "
                     "arguments[0])", 10, folder_name)
    li.click()
    drv.find_element("css selector", "#of-open").click()


@H.needs_ase
def test_open_folder_switches_and_drafts_come_back(setup):
    drv, base, home, a, b = setup
    H.load(drv, base)
    wait_explorer(drv, ["1_dia_opt_rev1.cif", "1_dia_opt_rev1.out"])
    # a modified structure in folder A
    drv.execute_script("window.__mace.settings.set('autosaveDrafts', 1)")
    drv.execute_async_script("const done = arguments[1]; window.__mace.open(arguments[0]).then(() => done())",
                             "1_dia_opt_rev1.cif")
    H.wait(drv, "return window.__mace.viewport && window.__mace.viewport.displayed() > 0", 30)
    drv.execute_script("window.__mace.run('build.symModeP1')")
    drv.execute_script("const E = window.__mace.editor.E; E.api.edit(window.__mace.editor.doc(), 'replace', "
                       "{indices: [0], element: 'Si'}, {})")
    H.wait(drv, "const e = window.__mace.editor.doc(); return e && e.dirty && e.pending === 0", 20)
    # File > Open Folder…: the modified document asks first
    menu_item(drv, "File", "file.openFolder")
    choose_in_dialog(drv, "calcB")
    H.wait(drv, "const d = document.getElementById('dirty-first'); return d && d.open", 10)
    assert "unsaved changes" in drv.find_element("css selector", "#dirty-first .dlg-body").text
    drv.find_element("css selector", '#dirty-first [data-r="discard"]').click()
    wait_explorer(drv, ["1_dia_opt_BULK_OPTGEOM.d12", "1_dia_opt_BULK_OPTGEOM.out"])
    assert "calcB" in drv.execute_script("return document.getElementById('root-chip').title")
    # Recent folders are the profile's (gui.json), so they also come back on a new port
    H.wait(drv, "return window.__mace.profile.recentFolders().some((p) => p.endsWith('/calcB'))", 15)
    # back to A through the folder name in the Explorer's header: the draft is offered again
    drv.find_element("css selector", "#root-chip").click()
    choose_in_dialog(drv, "calcA")
    wait_explorer(drv, ["1_dia_opt_rev1.cif", "1_dia_opt_rev1.out"])
    H.wait(drv, "const b = document.getElementById('draft-banner'); return b && !b.hidden && "
                "b.textContent.includes('1_dia_opt_rev1.cif')", 15)
    assert not H.errors(drv)


def test_refusals_show_in_the_dialog_and_the_palette_finds_it(setup):
    drv, base, home, a, b = setup
    H.load(drv, base)
    wait_explorer(drv, ["1_dia_opt_rev1.cif", "1_dia_opt_rev1.out"])
    # the command search finds it (no default key)
    from selenium.webdriver.common.keys import Keys
    drv.execute_script("window.__mace.palette.open('')")
    box = drv.find_element("css selector", "#palette input")
    box.send_keys("open folder")
    time.sleep(0.3)
    box.send_keys(Keys.ENTER)
    H.wait(drv, "const d = document.getElementById('open-folder'); return d && d.open", 15)
    inp = drv.find_element("css selector", "#of-path")
    inp.send_keys(str(home / ".ssh"))
    drv.find_element("css selector", "#of-open").click()
    err = H.wait(drv, "const e = document.getElementById('of-error'); return e && !e.hidden && e.textContent", 10)
    assert "Hidden folders cannot be opened here" in err
    inp.clear()
    inp.send_keys(str(home))
    drv.find_element("css selector", "#of-open").click()
    H.wait(drv, "const e = document.getElementById('of-error'); return e && e.textContent.includes("
                "'not your home folder itself')", 10)


def test_second_tab_follows_the_switch(setup):
    drv, base, home, a, b = setup
    H.load(drv, base)
    wait_explorer(drv, ["1_dia_opt_rev1.cif", "1_dia_opt_rev1.out"])
    first = drv.current_window_handle
    drv.switch_to.new_window("tab")
    drv.get(base)
    wait_explorer(drv, ["1_dia_opt_rev1.cif", "1_dia_opt_rev1.out"])
    menu_item(drv, "File", "file.openFolder")
    choose_in_dialog(drv, "calcB")
    wait_explorer(drv, ["1_dia_opt_BULK_OPTGEOM.d12", "1_dia_opt_BULK_OPTGEOM.out"])
    drv.switch_to.window(first)
    H.wait(drv, "const d = document.getElementById('folder-changed'); return (d && d.open) || "
                "[...document.querySelectorAll('#file-list .file')].some(e => e.dataset.path.includes('BULK'))", 30)
    wait_explorer(drv, ["1_dia_opt_BULK_OPTGEOM.d12", "1_dia_opt_BULK_OPTGEOM.out"], 30)
