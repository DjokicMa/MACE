"""Recent files in headless Firefox (plan T10, criterion 5): kept per served
folder under the server's dir_hash, newest first, pinned ones on top, a
missing file grayed and removable, the old key moved over once, and the
explorer's collapsible RECENT section. Since the 1.1.5 polish round (P2) the
list lives in the profile on the server (gui.json recent.files.<dir hash>),
so these tests read it there."""
import json
import time

import pytest

import guiharness as H
import p4util as U

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")                 # the server reads the demo structures with it


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    a = U.start_app(tmp_path_factory.mktemp("p4recent"))
    try:
        yield a
    finally:
        U.stop_app(a)


@pytest.fixture
def fresh(app):
    U.reset(app[0])
    app[0].execute_script("Object.keys(localStorage).filter((k) => k.startsWith('mace.gui.recent') || k === 'mace.gui.explorer.recentOpen').forEach((k) => localStorage.removeItem(k));"
                          "window.__mace.profile.setRecentFiles([]); window.__mace.profile.setUi('explorer', {});")
    app[0].execute_async_script("const d = arguments[0]; window.__mace.profile.flush().then(() => d(1))")
    app[0].execute_script("window.__mace.renderRecent()")
    return app


def dir_hash(drv):
    return drv.execute_script("return window.MACE.state.info.dir_hash")


def reload_files(drv):
    drv.execute_async_script("const done = arguments[0]; window.MACE.loadFiles().then(() => { window.__mace.renderRecent(); done(1); }, (e) => done(e.message))")


def open_recent_menu(drv):
    drv.execute_script("""const b = [...document.querySelectorAll('.menu-top')].find((x) => x.textContent === 'File');
        b.dispatchEvent(new PointerEvent('pointerdown', {bubbles: true, button: 0}))""")
    H.wait(drv, "return document.querySelectorAll('.menu').length > 0")
    drv.execute_script("""const r = [...document.querySelectorAll('.menu .menu-item')].find((x) => x.textContent.includes('Open Recent'));
        r.dispatchEvent(new PointerEvent('pointerenter', {bubbles: true})); r.click()""")
    H.wait(drv, "return document.querySelectorAll('.menu').length > 1", 5)
    return drv.execute_script("""const m = [...document.querySelectorAll('.menu')].pop();
        return [...m.querySelectorAll('.menu-item')].map((x) => x.querySelector('.lbl').textContent)""")


def test_open_recent_after_reload(fresh):
    drv, base = fresh[0], fresh[1]
    for p in ("structures/MgO.cif", "structures/Si.cif", "structures/hBN.cif"):
        U.open_doc(drv, p)
    H.load(drv, base)
    H.wait(drv, "return document.documentElement.dataset.appReady === '1'", 20)
    items = open_recent_menu(drv)
    assert items[:3] == ["structures/hBN.cif", "structures/Si.cif", "structures/MgO.cif"], items
    assert items[-1] == "Clear Recently Opened"
    U.shot(drv, "open_recent_menu")
    drv.execute_script("""const m = [...document.querySelectorAll('.menu')].pop();
        [...m.querySelectorAll('.menu-item')].find((x) => x.textContent.includes('structures/Si.cif')).click()""")
    H.wait(drv, "return window.__mace.doc() && window.__mace.doc().title === 'Si.cif'", 20)
    drv.execute_async_script("const d = arguments[0]; window.__mace.profile.flush().then(() => d(1))")
    prof = drv.execute_async_script("const d = arguments[0]; window.MACE.api('profile').then(d)")
    stored = prof["recent"]["files"][dir_hash(drv)]["items"]
    assert [x["path"] for x in stored][:3] == ["structures/Si.cif", "structures/hBN.cif", "structures/MgO.cif"]
    assert all(set(x) == {"path", "t", "pinned"} for x in stored)
    U.no_errors(drv)


def test_recent_pin_remove_missing(fresh):
    drv, root = fresh[0], fresh[2]
    extra = root / "structures" / "Gone.cif"
    extra.write_text((root / "structures" / "MgO.cif").read_text())
    try:
        reload_files(drv)
        H.wait(drv, "return !!window.MACE.fileByPath('structures/Gone.cif')", 20)
        for p in ("structures/Gone.cif", "structures/MgO.cif", "structures/NaCl.cif"):
            U.open_doc(drv, p)
        drv.execute_script("window.__mace.closeAll()")
        # pin MgO from its context menu in the explorer's RECENT section
        drv.execute_script("""const b = document.querySelector('#recent-files [data-recent="structures/MgO.cif"]');
            const r = b.getBoundingClientRect(); b.dispatchEvent(new MouseEvent('contextmenu', {bubbles: true, clientX: r.left + 5, clientY: r.top + 5}))""")
        H.wait(drv, "return document.querySelectorAll('.menu').length > 0", 5)
        labels = drv.execute_script("return [...document.querySelector('.menu').querySelectorAll('.menu-item .lbl')].map((x) => x.textContent)")
        assert labels == ["Open", "Pin", "Remove from Recent", "Copy Path"], labels
        drv.execute_script("[...document.querySelectorAll('.menu .menu-item')].find((x) => x.textContent.includes('Pin')).click()")
        rows = drv.execute_script("return [...document.querySelectorAll('#recent-files [data-recent]')].map((b) => [b.dataset.recent, !!b.querySelector('.pin')])")
        assert rows[0] == ["structures/MgO.cif", True] and [r[0] for r in rows] == ["structures/MgO.cif", "structures/NaCl.cif", "structures/Gone.cif"], rows
        # the file goes away: grayed "Missing", Open disabled, Remove works
        extra.unlink()
        reload_files(drv)
        H.wait(drv, "return !window.MACE.fileByPath('structures/Gone.cif')", 20)
        H.wait(drv, "const b = document.querySelector('#recent-files [data-recent=\"structures/Gone.cif\"]'); return b && b.classList.contains('missing')", 10)
        assert "Missing" in drv.find_element("css selector", '#recent-files [data-recent="structures/Gone.cif"]').text
        drv.execute_script("""const b = document.querySelector('#recent-files [data-recent="structures/Gone.cif"]');
            b.dispatchEvent(new MouseEvent('contextmenu', {bubbles: true, clientX: 50, clientY: 300}))""")
        H.wait(drv, "return document.querySelectorAll('.menu').length > 0", 5)
        assert drv.execute_script("return [...document.querySelectorAll('.menu .menu-item')].find((x) => x.textContent.includes('Open')).getAttribute('aria-disabled')") == "true"
        drv.execute_script("[...document.querySelectorAll('.menu .menu-item')].find((x) => x.textContent.includes('Remove from Recent')).click()")
        rows = drv.execute_script("return [...document.querySelectorAll('#recent-files [data-recent]')].map((b) => b.dataset.recent)")
        assert rows == ["structures/MgO.cif", "structures/NaCl.cif"], rows
        # pinned entries don't count toward recentLimit
        drv.execute_script("window.__mace.settings.set('recentLimit', 1)")
        U.open_doc(drv, "structures/LiF.cif")
        rows = drv.execute_script("return [...document.querySelectorAll('#recent-files [data-recent]')].map((b) => b.dataset.recent)")
        assert rows == ["structures/MgO.cif", "structures/LiF.cif"], rows
    finally:
        if extra.exists():
            extra.unlink()
    U.no_errors(drv)


def test_recent_key_migrates_once(fresh):
    drv, base = fresh[0], fresh[1]
    old = drv.execute_script("""let h = 0; for (const ch of window.MACE.state.info.root) h = (h * 31 + ch.charCodeAt(0)) | 0;
        return 'mace.gui.recent.' + (h >>> 0).toString(36)""")
    # a browser from before: the old key, and no Recent list in the profile yet
    drv.execute_script("localStorage.setItem(arguments[0], JSON.stringify(['structures/LiF.cif', 'structures/MgO.cif']));"
                       "localStorage.removeItem('mace.gui.profile.migrated'); localStorage.removeItem('mace.gui.profile.cache.v1');"
                       "localStorage.removeItem('mace.gui.recent.' + window.MACE.state.info.dir_hash)", old)
    drv.execute_async_script("const d = arguments[0]; window.MACE.api('profile', {body: {recent: {files: {[window.MACE.state.info.dir_hash]: null}}}}).then(() => d(1))")
    H.load(drv, base)
    H.wait(drv, "return document.documentElement.dataset.appReady === '1'", 20)
    got = drv.execute_script("return window.__mace.profile.recentFiles()")
    assert [x["path"] for x in got] == ["structures/LiF.cif", "structures/MgO.cif"]
    drv.execute_async_script("const d = arguments[0]; window.__mace.profile.flush().then(() => d(1))")
    prof = drv.execute_async_script("const d = arguments[0]; window.MACE.api('profile').then(d)")
    assert [x["path"] for x in prof["recent"]["files"][dir_hash(drv)]["items"]] == ["structures/LiF.cif", "structures/MgO.cif"]
    # once: an old key written again later is not merged over the profile's list
    drv.execute_script("localStorage.setItem(arguments[0], JSON.stringify(['structures/Si.cif']))", old)
    H.load(drv, base)
    H.wait(drv, "return document.documentElement.dataset.appReady === '1'", 20)
    got2 = drv.execute_script("return window.__mace.profile.recentFiles()")
    assert [x["path"] for x in got2] == ["structures/LiF.cif", "structures/MgO.cif"]
    drv.execute_script("localStorage.removeItem(arguments[0])", old)


def test_recent_section_in_explorer(fresh):
    drv, base = fresh[0], fresh[1]
    assert drv.execute_script("return document.querySelector('#recent-files').children.length") == 0
    U.open_doc(drv, "structures/NaCl.cif")
    H.wait(drv, "return !!document.querySelector('#recent-files [data-recent=\"structures/NaCl.cif\"]')", 5)
    assert drv.execute_script("return document.querySelector('#recent-toggle').textContent.trim()") == "Recent"
    # collapsed, and still collapsed after a reload
    drv.execute_script("document.querySelector('#recent-toggle').click()")
    assert drv.execute_script("return document.querySelector('#recent-rows').hidden && document.querySelector('#recent-toggle').getAttribute('aria-expanded') === 'false'")
    H.load(drv, base)
    H.wait(drv, "return !!document.querySelector('#recent-toggle')", 20)
    assert drv.execute_script("return document.querySelector('#recent-rows').hidden")
    drv.execute_script("document.querySelector('#recent-toggle').click()")
    drv.execute_script("document.querySelector('#recent-files [data-recent=\"structures/NaCl.cif\"]').click()")
    H.wait(drv, "return window.__mace.doc() && window.__mace.doc().title === 'NaCl.cif'", 20)
    U.no_errors(drv)


def test_built_doc_enters_recent_only_when_saved(fresh):
    drv, root = fresh[0], fresh[2]
    U.open_doc(drv, "structures/MgO.cif")
    drv.execute_script("window.__mace.closeAll()")
    before = drv.execute_script("return [...document.querySelectorAll('#recent-files [data-recent]')].map((b) => b.dataset.recent)")
    drv.execute_script("""const dt = new DataTransfer(); dt.items.add(new File([arguments[0]], 'Dropped_LiF.cif'));
        document.body.dispatchEvent(new DragEvent('drop', {dataTransfer: dt, bubbles: true, cancelable: true}))""",
                       (root / "structures" / "LiF.cif").read_text())
    H.wait(drv, "return window.__mace.doc() && window.__mace.doc().title === 'Dropped_LiF.cif'", 20)
    time.sleep(0.2)
    assert drv.execute_script("return [...document.querySelectorAll('#recent-files [data-recent]')].map((b) => b.dataset.recent)") == before
    drv.execute_script("window.__mace.run('file.saveAs')")
    H.wait(drv, "const d = document.getElementById('save-as'); return d && d.open", 10)
    drv.execute_script("document.querySelector('#save-as [data-v=\"save\"]').click()")
    H.wait(drv, "return !window.__mace.doc().dirty", 10)
    rows = drv.execute_script("return [...document.querySelectorAll('#recent-files [data-recent]')].map((b) => b.dataset.recent)")
    assert rows[0] == "Dropped_LiF.cif", rows
    (root / "Dropped_LiF.cif").unlink()


def test_recent_reset_elsewhere_is_not_brought_back(fresh):
    """Settings > Advanced > Reset Recent in another browser (here: the route
    itself), then this browser loads the page again: its cached list must not
    come back (and so must not be uploaded again on the next change)."""
    from test_shell_api import req
    drv, base = fresh[0], fresh[1]
    drv.execute_script("window.__mace.profile.setRecentFiles([{path: 'structures/MgO.cif', t: Date.now(), pinned: false}])")
    drv.execute_async_script("const d = arguments[0]; window.__mace.profile.flush().then(() => d(1))")
    assert [x["path"] for x in drv.execute_script("return window.__mace.profile.recentFiles()")] == ["structures/MgO.cif"]
    assert req(base, "api/profile/reset", data={"sections": ["recent"]})[0] == 200
    H.load(drv, base)
    assert drv.execute_script("return window.__mace.profile.recentFiles()") == []
    drv.execute_async_script("const d = arguments[0]; window.__mace.profile.flush().then(() => d(1))")
    stored = json.loads(req(base, "api/profile")[2])
    assert stored["recent"]["files"] == {}
