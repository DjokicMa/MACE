"""File History and the protected-kind refusal in headless Firefox
(PLAN_analysis C19-d, C20-d, C20-e): real clicks and keys for the actions
under test (Save, Save As, the Explorer context menu, the palette, the
dialog's buttons, the toast's Undo); the edits themselves go through the
editor's API (not under test). A fresh Firefox profile each time."""
import hashlib
import shutil
import time
from pathlib import Path

import pytest

import guiharness as H
import p4util as U

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
from selenium.webdriver.common.action_chains import ActionChains  # noqa: E402
from selenium.webdriver.common.keys import Keys  # noqa: E402


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def edit(drv, op, args):
    drv.execute_script("const E = window.__mace.editor.E; E.api.edit(window.__mace.editor.doc(), arguments[0], arguments[1], {})", op, args)
    U.wait_idle(drv)


def wait_saved(drv):
    H.wait(drv, "const e = window.__mace.editor.doc(); return e && !e.dirty && e.pending === 0", 15)


def file_row(drv, rel):
    """The Explorer row of `rel`, scrolled into view. The list re-renders when
    the folder changes, so a row found a moment earlier can be replaced:
    found again then."""
    from selenium.common.exceptions import StaleElementReferenceException
    for attempt in range(5):
        row = drv.find_element("css selector", f'#file-list .file[data-path="{rel}"]')
        try:
            drv.execute_script("arguments[0].scrollIntoView({block: 'center'})", row)
            return row
        except StaleElementReferenceException:
            if attempt == 4:
                raise
            time.sleep(0.2)


def row_still(drv, rel, polls=3):
    """Wait until the row stays at one place (within 2 px), under the pointer, for `polls` checks 0.2 s apart."""
    find = "return document.querySelector(`#file-list .file[data-path=\"${arguments[0]}\"]`)"
    js = (find.replace("return ", "const f = ") + "; if (!f) return null; const r = f.getBoundingClientRect();"
          " const e = document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2);"
          " return e && e.closest('.file') === f ? r.y : null")
    last, same, end = None, 0, time.time() + 10
    while time.time() < end and same < polls:
        if drv.execute_script(js, rel) is None:
            drv.execute_script("arguments[0] && arguments[0].scrollIntoView({block: 'center'})", drv.execute_script(find, rel))
        y = drv.execute_script(js, rel)
        same = same + 1 if y is not None and last is not None and abs(y - last) <= 2 else 0
        last = y
        time.sleep(0.2)
    assert same >= polls, f"the Explorer row {rel} kept moving"


def click_menu_item(drv, cmd):
    H.wait(drv, "return !!document.querySelector(`.menu .menu-item[data-cmd-id='${arguments[0]}']`)", 5, cmd)
    drv.find_element("css selector", f".menu .menu-item[data-cmd-id='{cmd}']").click()


@pytest.fixture
def app(tmp_path):
    root = H.examples_copy(tmp_path)
    (root / "structures" / "Hist.cif").write_bytes((root / "structures" / "MgO.cif").read_bytes())
    shutil.copyfile(H.EXAMPLES / "outputs" / "Si_sp.out", root / "structures" / "x.out")
    a = U.start_app(tmp_path, root=root)
    try:
        yield a
    finally:
        try:
            U.stop_app(a)
        except Exception:          # the restart test stopped it already
            pass


def test_save_as_onto_an_output_is_refused_in_the_dialog(app):
    pytest.importorskip("ase", reason="the server reads and writes structures with ase")
    drv, base, root = app[0], app[1], app[2]
    before = U.tree_md5(root)
    file_row(drv, "structures/Hist.cif").click()
    H.wait(drv, "return window.__mace.editor && window.__mace.editor.doc()", 30)
    drv.execute_script("document.body.focus()")
    U.keys(drv, Keys.CONTROL, Keys.SHIFT, "s")
    H.wait(drv, "return document.querySelector('#save-as') && document.querySelector('#save-as').open", 10)
    name = drv.find_element("css selector", "#save-as [data-k='name']")
    name.clear()
    name.send_keys("x.out")
    drv.find_element("css selector", "#save-as [data-v='save']").click()
    H.wait(drv, "return /calculation output; MACE never writes over outputs/.test(document.querySelector('#save-as .err').textContent)", 10)
    assert drv.execute_script("return document.querySelector('#save-as').open")
    drv.find_element("css selector", "#save-as [data-v='cancel']").click()
    assert U.tree_md5(root) == before
    U.no_errors(drv)


def test_file_history_lists_compares_restores_and_undoes(app, tmp_path):
    pytest.importorskip("ase", reason="the server reads and writes structures with ase")
    drv, base, root, xdg, proc = app[0], app[1], app[2], app[3], app[4]
    target = root / "structures" / "Hist.cif"
    v0 = target.read_bytes()
    drv.execute_script("window.__mace.settings.set('confirmOverwrite', false)")
    file_row(drv, "structures/Hist.cif").click()
    H.wait(drv, "return window.__mace.editor && window.__mace.editor.doc()", 30)
    drv.execute_script("window.__mace.run('build.symModeP1')")
    edit(drv, "replace", {"indices": [0], "element": "Ca"})
    drv.execute_script("document.body.focus()")
    U.keys(drv, Keys.CONTROL, "s")
    wait_saved(drv)
    v1 = target.read_bytes()
    edit(drv, "replace", {"indices": [0], "element": "Sr"})
    U.keys(drv, Keys.CONTROL, "s")
    wait_saved(drv)
    v2 = target.read_bytes()
    assert len({v0, v1, v2}) == 3
    # the save toast: Undo puts the previous version back (and the document reloads)
    H.wait(drv, "return [...document.querySelectorAll('#toasts button')].some((b) => b.textContent === 'Undo')", 5)
    undo = [b for b in drv.find_elements("css selector", "#toasts button") if b.text == "Undo"][-1]
    undo.click()
    H.wait(drv, "return [...document.querySelectorAll('#toasts .msg')].some((m) => /Restored the previous/.test(m.textContent))", 15)
    assert target.read_bytes() == v1
    # File History from the Explorer's context menu (a real right-click)
    # (after a save the Explorer is drawn again, and the Recent list above it can grow: the row
    # moves for a moment, so the click waits until the row stays put under the pointer)
    file_row(drv, "structures/Hist.cif")
    row_still(drv, "structures/Hist.cif")
    ActionChains(drv).context_click(drv.find_element("css selector", '#file-list .file[data-path="structures/Hist.cif"]')).perform()
    click_menu_item(drv, "explorer.history")
    H.wait(drv, "return document.querySelector('#file-history') && document.querySelector('#file-history').open", 10)
    assert "Hist.cif" in drv.find_element("css selector", "#file-history").text.splitlines()[0]
    H.wait(drv, "return document.querySelectorAll('#file-history .hist-row').length >= 3", 10)
    rows = drv.execute_script("return [...document.querySelectorAll('#file-history .hist-row')].map((r) => r.textContent)")
    assert len(rows) == 3, rows             # before save 1, before save 2, before the Undo
    # Compare with Current: the oldest version (v0) beside the current file (v1)
    drv.find_elements("css selector", "#file-history .hist-row [data-act=compare]")[-1].click()
    H.wait(drv, "return document.querySelectorAll('#file-history .hist-pre').length === 2", 10)
    texts = drv.execute_script("return [...document.querySelectorAll('#file-history .hist-pre')].map((p) => p.textContent)")
    assert texts[0] == v0.decode() and texts[1] == v1.decode()
    drv.find_element("css selector", "#file-history [data-act=back]").click()
    # Restore This Version: the oldest bytes come back, and the current version is kept first
    drv.find_elements("css selector", "#file-history .hist-row [data-act=restore]")[-1].click()
    H.wait(drv, "return document.querySelectorAll('#file-history .hist-row').length === 4", 15)
    assert sha(target) == hashlib.sha256(v0).hexdigest()
    drv.find_element("css selector", "#file-history [data-close]").click()
    # the palette's File: Show History acts on the active document
    drv.execute_script("document.body.focus()")
    U.keys(drv, Keys.F1)
    box = drv.find_element("css selector", "#palette input")
    box.send_keys("Show History")
    time.sleep(0.3)
    box.send_keys(Keys.ENTER)
    H.wait(drv, "return document.querySelectorAll('#file-history .hist-row').length === 4", 10)
    U.no_errors(drv)

    # C20-e: after a restart on another port and in a fresh browser profile, the history is still there
    U.stop_app(app)
    port = H.free_port()
    proc2, base2, _ = H.start_server(root, xdg, port=port, MACE_GUI_TEST_HOOKS="1")
    drv2 = H.firefox()
    try:
        H.load(drv2, base2)
        ActionChains(drv2).context_click(file_row(drv2, "structures/Hist.cif")).perform()
        click_menu_item(drv2, "explorer.history")
        H.wait(drv2, "return document.querySelectorAll('#file-history .hist-row').length === 4", 15)
        assert base2 != base
    finally:
        drv2.quit()
        proc2.terminate()
        proc2.wait(timeout=15)
