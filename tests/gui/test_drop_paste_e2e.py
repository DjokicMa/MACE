"""Drag and drop, Import from Clipboard, Paste, Cut and Duplicate in headless
Firefox (plan T9, criterion 4). WebDriver can't drive a real OS drag or the
system clipboard: drops dispatch synthetic DragEvents carrying a
DataTransfer built in the page, which reach the same window listeners a
real drop does; pastes call the page's paste handler with the text (Firefox
empties a synthetic ClipboardEvent's DataTransfer)."""
import time

import pytest

import guiharness as H
import p4util as U

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    a = U.start_app(tmp_path_factory.mktemp("p4drop"))
    try:
        yield a
    finally:
        U.stop_app(a)


@pytest.fixture
def fresh(app):
    U.reset(app[0])
    return app


DROP_JS = """
const files = arguments[0], phase = arguments[1];
const dt = new DataTransfer();
for (const [name, text, size] of files) {
  const f = size ? new File([new Uint8Array(size)], name) : new File([text], name, {type: 'text/plain'});
  dt.items.add(f);
}
window.__dt = dt;
for (const t of phase) document.body.dispatchEvent(new DragEvent(t, {dataTransfer: dt, bubbles: true, cancelable: true}));
"""


def drop(drv, files, phases=("dragenter", "dragover", "drop")):
    drv.execute_script(DROP_JS, [list(f) + [0] * (3 - len(f)) for f in files], list(phases))


def parse_requests(drv):
    return drv.execute_script("return performance.getEntriesByType('resource').filter((r) => r.name.includes('api/parse')).length")


def paste(drv, text, target="document.activeElement || document.body"):
    """The page's paste handler with this clipboard text. Firefox empties the
    DataTransfer of a synthetic ClipboardEvent, so the test hands the same
    handler an event-like object (target, clipboardData, preventDefault)."""
    return drv.execute_script(f"return window.__mace.onPaste({target}, arguments[0])", text)


def wait_row(drv, label, timeout=15):
    """The History's last row is `label` and the edit queue is idle."""
    H.wait(drv, f"const e = window.__mace.editor.doc(); return e && e.rows[e.rows.length - 1].label === {label!r} && e.pending === 0 && !e.waiting", timeout)


def cif(name):
    return (H.EXAMPLES / "structures" / name).read_text()


def test_drop_cif_overlay_untitled_dirty_md5(fresh):
    drv, base, root = fresh[0], fresh[1], fresh[2]
    before, files0 = U.tree_md5(root), U.req(base, "api/files")[2]
    drop(drv, [("NaCl_copy.cif", cif("NaCl.cif"))], ("dragenter", "dragover"))
    assert drv.execute_script("const o = document.getElementById('drop-overlay'); return !!o && !o.hidden && getComputedStyle(o).display !== 'none'")
    assert "Drop to open (CIF, POSCAR, XYZ, d12, out)" in drv.find_element("id", "drop-overlay").text
    U.shot(drv, "drop_overlay")
    drv.execute_script("document.body.dispatchEvent(new DragEvent('drop', {dataTransfer: window.__dt, bubbles: true, cancelable: true}))")
    H.wait(drv, "return window.__mace.doc() && window.__mace.doc().title === 'NaCl_copy.cif'", 20)
    assert drv.execute_script("return document.getElementById('drop-overlay').hidden")
    d = U.doc(drv)
    assert d["untitled"] and d["dirty"] and d["n"] == 8 and d["sg"] == 225, d
    tip = drv.execute_script("return document.querySelector('.doc-tab.active').title")
    assert "Not saved in the folder yet" in tip, tip
    H.wait(drv, "return document.body.textContent.includes('Not saved in the folder yet')", 5)
    assert U.tree_md5(root) == before and U.req(base, "api/files")[2] == files0
    assert U.req(base, "api/test/parse_tmp")[2]["files"] == []
    U.no_errors(drv)


def test_overlay_hides_on_escape_and_leave(fresh):
    drv = fresh[0]
    drop(drv, [("a.cif", "x")], ("dragenter",))
    assert not drv.execute_script("return document.getElementById('drop-overlay').hidden")
    from selenium.webdriver.common.keys import Keys
    U.keys(drv, Keys.ESCAPE)
    assert drv.execute_script("return document.getElementById('drop-overlay').hidden")
    drop(drv, [("a.cif", "x")], ("dragenter", "dragleave"))
    assert drv.execute_script("return document.getElementById('drop-overlay').hidden")
    assert parse_requests(drv) == 0


def test_drop_over_50mb_toast_no_request(fresh):
    drv = fresh[0]
    drop(drv, [("huge.xyz", "", 50 * 1024 * 1024 + 1)])
    H.wait(drv, "return [...document.querySelectorAll('#toasts > *')].some((t) => t.textContent.includes('over 50 MB'))", 5)
    assert parse_requests(drv) == 0
    drv.execute_script("window.__dt = null")


def test_drop_unsupported_folder_and_over_8(fresh):
    drv = fresh[0]
    drop(drv, [("picture.png", "png")])
    H.wait(drv, "return [...document.querySelectorAll('#toasts > *')].some((t) => t.textContent.includes(\"MACE can't open .png files\"))", 5)
    # a folder: the browser gives a zero-size File with no type
    drv.execute_script("""const dt = new DataTransfer(); dt.items.add(new File([], 'somefolder'));
        document.body.dispatchEvent(new DragEvent('drop', {dataTransfer: dt, bubbles: true, cancelable: true}))""")
    H.wait(drv, "return [...document.querySelectorAll('#toasts > *')].some((t) => t.textContent.includes(\"Folders can't be dropped\"))", 5)
    assert parse_requests(drv) == 0
    text = cif("MgO.cif")
    drop(drv, [(f"MgO_{k}.cif", text) for k in range(12)])
    H.wait(drv, "return [...document.querySelectorAll('#toasts > *')].some((t) => t.textContent.includes('Opened 8 of 12 files; drop the rest again'))", 5)
    H.wait(drv, "return window.__mace.docs().filter((d) => /^MgO_\\d+\\.cif$/.test(d.title)).length === 8", 60)
    assert parse_requests(drv) == 8
    U.no_errors(drv)


def test_drop_name_clash_disambiguated(fresh):
    drv, base, root = fresh[0], fresh[1], fresh[2]
    p = root / "structures" / "MgO.cif"
    md5 = U.tree_md5(p.parent)
    for k in (1, 2):
        drop(drv, [("MgO.cif", cif("MgO.cif"))])
        want = "MgO.cif (dropped)" if k == 1 else "MgO.cif (dropped 2)"
        H.wait(drv, f"return window.__mace.doc() && window.__mace.doc().title === '{want}'", 20)
    drv.execute_script("window.__mace.closeAll()")
    drop(drv, [("MgO.cif", cif("MgO.cif"))])
    H.wait(drv, "return window.__mace.doc() && window.__mace.doc().title === 'MgO.cif (dropped)'", 20)
    drv.execute_script("window.__mace.run('file.saveAs')")
    H.wait(drv, "const d = document.getElementById('save-as'); return d && d.open", 10)
    assert drv.execute_script("return document.querySelector('#save-as [data-k=\"name\"]').value") == "MgO_dropped.cif"
    # saving over the folder's MgO.cif asks first, and names it
    drv.execute_script("""const d = document.querySelector('#save-as'); d.querySelector('[data-k="dir"]').value = 'structures';
        d.querySelector('[data-k="name"]').value = 'MgO.cif'; d.querySelector('[data-v="save"]').click()""")
    H.wait(drv, "const d = document.getElementById('save-replace'); return d && d.open", 10)
    assert "structures/MgO.cif already exists" in drv.find_element("id", "save-replace").text
    assert U.tree_md5(p.parent) == md5
    drv.execute_script("document.querySelector('#save-replace [data-v=\"cancel\"]').click()")
    time.sleep(0.3)
    assert U.tree_md5(p.parent) == md5
    U.no_errors(drv)


def test_dropped_doc_save_as_default_name(fresh):
    drv, root = fresh[0], fresh[2]
    drop(drv, [("POSCAR_LiF", "LiF\n1.0\n4.0 0 0\n0 4.0 0\n0 0 4.0\nLi F\n1 1\nDirect\n0 0 0\n0.5 0.5 0.5\n")])
    H.wait(drv, "return window.__mace.doc() && window.__mace.doc().title === 'POSCAR_LiF'", 20)
    assert U.doc(drv)["n"] == 2
    drv.execute_script("window.__mace.run('file.save')")           # untitled: Save goes to Save As
    H.wait(drv, "const d = document.getElementById('save-as'); return d && d.open", 10)
    assert drv.execute_script("return document.querySelector('#save-as [data-k=\"name\"]').value") == "POSCAR_LiF.cif"
    drv.execute_script("document.querySelector('#save-as [data-v=\"save\"]').click()")
    H.wait(drv, "return !window.__mace.doc().dirty", 10)
    assert (root / "POSCAR_LiF.cif").exists()
    (root / "POSCAR_LiF.cif").unlink()


def test_parse_errors_shown(fresh):
    drv = fresh[0]
    drop(drv, [("broken.cif", "data_x\n_cell_length_a 4\n_cell_length_b oops\nloop_\n_atom_site_label\n")])
    H.wait(drv, "return [...document.querySelectorAll('#toasts > *')].some((t) => t.textContent.includes(\"Couldn't read broken.cif as CIF\"))", 10)
    probs = drv.execute_script("return window.__mace.problems().map((p) => [p.source, p.message, p.action])")
    assert any(p[0] == "broken.cif" and "Couldn't read broken.cif" in p[1] and p[2] == "Open as Text" for p in probs), probs
    drv.execute_script("window.__mace.runProblemAction('broken.cif')")
    H.wait(drv, "return window.__mace.doc() && window.__mace.doc().kind === 'text' && window.__mace.doc().title === 'broken.cif'", 10)
    drv.execute_script("window.__maceErrors.length = 0")


def test_import_clipboard_dialog_fallback(fresh):
    """Mod+Shift+V's paste event carries the text into the dialog; its live
    detection names the format; Import opens an untitled tab."""
    drv = fresh[0]
    poscar = "MgO\n1.0\n4.2 0 0\n0 4.2 0\n0 0 4.2\nMg O\n4 4\nDirect\n0 0 0\n0 .5 .5\n.5 0 .5\n.5 .5 0\n.5 .5 .5\n.5 0 0\n0 .5 0\n0 0 .5\n"
    drv.execute_script("document.body.dispatchEvent(new KeyboardEvent('keydown', {key: 'V', code: 'KeyV', ctrlKey: true, shiftKey: true, bubbles: true, cancelable: true}))")
    paste(drv, poscar)
    H.wait(drv, "const d = document.getElementById('import-clipboard'); return d && d.open", 10)
    assert drv.find_element("id", "import-detect").text == "Looks like POSCAR, 8 atoms"
    U.shot(drv, "import_clipboard")
    # editing the text updates it; an XYZ shows the box option
    drv.execute_script("""const t = document.getElementById('import-text'); t.value = '3\\nwater\\nO 0 0 0\\nH .76 .59 0\\nH -.76 .59 0\\n';
        t.dispatchEvent(new Event('input'))""")
    assert drv.find_element("id", "import-detect").text == "Looks like XYZ, 3 atoms"
    assert drv.execute_script("return !document.getElementById('import-box-row').hidden")
    drv.execute_script("const t = document.getElementById('import-text'); t.value = arguments[0]; t.dispatchEvent(new Event('input'))", poscar)
    drv.execute_script("document.querySelector('#import-clipboard [data-act=\"import\"]').click()")
    H.wait(drv, "return window.__mace.doc() && /^Clipboard \\d+$/.test(window.__mace.doc().title)", 20)
    d = U.doc(drv)
    assert d["n"] == 8 and d["untitled"] and d["sg"] == 225, d
    assert drv.execute_script("const d = document.getElementById('import-clipboard'); return !d || !d.open")
    # the menu entry with an unreadable clipboard opens the empty dialog
    drv.execute_script("navigator.clipboard.readText = () => Promise.reject(new Error('denied')); window.__mace.run('file.importClipboard')")
    H.wait(drv, "const d = document.getElementById('import-clipboard'); return d && d.open", 10)
    assert "could not be read" in drv.find_element("id", "import-detect").text
    assert drv.execute_script("return document.activeElement.id") == "import-text"
    drv.execute_script("document.querySelector('#import-clipboard [data-act=\"cancel\"]').click()")
    U.no_errors(drv)


def test_mod_v_does_not_prevent_default_keydown(fresh):
    drv = fresh[0]
    U.open_doc(drv, "structures/MgO.cif")
    for shift in (False, True):
        prevented = drv.execute_script("""const ev = new KeyboardEvent('keydown', {key: 'v', code: 'KeyV', ctrlKey: true, shiftKey: arguments[0], bubbles: true, cancelable: true});
            document.querySelector('.viewer-wrap').dispatchEvent(ev); return ev.defaultPrevented""", shift)
        assert prevented is False


def test_paste_into_text_input_left_alone(fresh):
    drv = fresh[0]
    U.open_doc(drv, "structures/MgO.cif")
    drv.execute_script("const i = document.createElement('input'); i.id = 'p4-in'; document.body.appendChild(i); i.focus()")
    rows0 = U.ed_state(drv)["rows"]
    assert paste(drv, "1\nX: element, fractional x y z, Cartesian x y z (Å)\nO 0.1 0.1 0.1  0 0 0\n", "document.getElementById('p4-in')") is False
    time.sleep(0.4)
    assert U.ed_state(drv)["rows"] == rows0
    assert not drv.execute_script("const d = document.getElementById('import-clipboard'); return d && d.open")
    drv.execute_script("document.getElementById('p4-in').remove()")


def test_paste_cut_duplicate_one_entry_each(fresh):
    drv = fresh[0]
    U.open_doc(drv, "structures/NaCl.cif")
    drv.execute_script("window.__mace.run('build.symModeP1')")
    U.wait_idle(drv)
    # Phase 3's Copy block for atoms 1 and 2, as the clipboard would hold it
    drv.execute_script("window.__mace.editor.select([0, 1], 'replace')")
    block = drv.execute_script("""const e = window.__mace.editor.doc(), s = e.structure;
        return '2\\nNaCl: element, fractional x y z, Cartesian x y z (Å)\\n' + [0, 1].map((i) =>
          s.symbols[i] + ' ' + s.frac[i].map((x) => ((x + 0.13) % 1).toFixed(5)).join(' ') + '   0 0 0').join('\\n') + '\\n'""")
    n0 = U.ed_state(drv)["n"]
    drv.execute_script("document.querySelector('.viewer-wrap').focus()")
    assert paste(drv, block) is True
    wait_row(drv, "Paste 2 Atoms")
    st = U.ed_state(drv)
    assert st["n"] == n0 + 2 and st["rows"][-1] == "Paste 2 Atoms", st["rows"]
    H.wait(drv, f"return [...window.__mace.editor.doc().selection].sort((a, b) => a - b).join() === '{n0},{n0 + 1}'", 5)
    # Cut: copy, then one delete step
    drv.execute_script("window.__mace.run('edit.cut')")
    wait_row(drv, "Cut 2 Atoms")
    st = U.ed_state(drv)
    assert st["n"] == n0 and st["rows"][-1] == "Cut 2 Atoms", st["rows"]
    # Duplicate: two copies, one entry, selected (and grabbed when the viewport has focus)
    drv.execute_script("window.__mace.editor.select([0], 'replace')")
    drv.execute_script("window.__mace.run('edit.duplicate')")
    wait_row(drv, "Duplicate 1 Atom")
    st = U.ed_state(drv)
    assert st["n"] == n0 + 1 and st["rows"][-1] == "Duplicate 1 Atom", st["rows"]
    from selenium.webdriver.common.keys import Keys
    U.keys(drv, Keys.ESCAPE)                            # Esc keeps the copies at the offset
    assert U.ed_state(drv)["n"] == n0 + 1
    U.no_errors(drv)


def test_paste_text_not_a_block_opens_import(fresh):
    drv = fresh[0]
    U.open_doc(drv, "structures/MgO.cif")
    paste(drv, "2\n\nH 0 0 0\nH 0 0 0.74\n")
    H.wait(drv, "const d = document.getElementById('import-clipboard'); return d && d.open", 10)
    assert drv.find_element("id", "import-detect").text == "Looks like XYZ, 2 atoms"
    drv.execute_script("document.querySelector('#import-clipboard [data-act=\"cancel\"]').click()")


def test_paste_sym_orbit_preview_confirm(fresh):
    """In Sym mode a pasted general-position atom brings its whole orbit
    (192 in Fm-3m): more than 3x the copied count asks first (D7)."""
    drv = fresh[0]
    U.open_doc(drv, "structures/MgO.cif")
    drv.execute_script("window.__mace.run('build.symModeSym')")
    U.wait_idle(drv)
    n0 = U.ed_state(drv)["n"]
    block = "1\nMgO: element, fractional x y z, Cartesian x y z (Å)\nO 0.11 0.07 0.03   0 0 0\n"
    drv.execute_script("document.querySelector('.viewer-wrap').focus()")
    paste(drv, block)
    H.wait(drv, "const d = document.getElementById('paste-confirm'); return d && d.open", 10)
    txt = drv.find_element("id", "paste-confirm").text
    assert "1 copied atom become 192" in txt, txt
    drv.execute_script("document.querySelector('#paste-confirm [data-r=\"cancel\"]').click()")
    time.sleep(0.4)
    assert U.ed_state(drv)["n"] == n0
    paste(drv, block)
    H.wait(drv, "const d = document.getElementById('paste-confirm'); return d && d.open", 10)
    drv.execute_script("document.querySelector('#paste-confirm [data-r=\"ok\"]').click()")
    wait_row(drv, "Paste 1 Atom")
    st = U.ed_state(drv)
    assert st["n"] == n0 + 192 and st["rows"][-1] == "Paste 1 Atom", st["rows"]
    U.no_errors(drv)
