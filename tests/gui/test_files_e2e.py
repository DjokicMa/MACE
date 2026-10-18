"""The file tools in headless Firefox (plan T3, T9, T10): packed structures
decoded exactly, Export Structure as browser downloads that never write
into the served folder, held view-only documents claimed back after a
reload, the "changed on disk" banner, and hover prefetch."""
import time
import warnings
from pathlib import Path

import pytest

import guiharness as H
import p4util as U

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
from selenium.webdriver.common.action_chains import ActionChains  # noqa: E402
ase_io = pytest.importorskip("ase.io")
np = pytest.importorskip("numpy")


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("p4files")
    a = U.start_app(tmp)
    try:
        yield a
    finally:
        U.stop_app(a)


@pytest.fixture
def fresh(app):
    U.reset(app[0])
    for p in Path(app[5]).glob("*"):
        p.unlink()
    return app


def test_packed_round_trip_exact(fresh):
    """Python encodes, the page decodes: the same doubles as ase reads (0 ulp)."""
    drv = fresh[0]
    U.open_doc(drv, "structures/Si.cif")
    url = drv.execute_script("""return performance.getEntriesByType('resource').map((r) => r.name)
        .filter((n) => n.includes('api/structure?') && n.includes('packed=1'))""")
    assert url, "the page asked for the packed payload"
    frac = drv.execute_script("return window.__mace.doc().structure.frac")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = ase_io.read(str(H.EXAMPLES / "structures" / "Si.cif")).get_scaled_positions(wrap=False)
    assert np.asarray(frac).tobytes() == np.asarray(ref, float).tobytes()
    s = drv.execute_script("const s = window.__mace.doc().structure; return [s.symbols, s.numbers.length, s.cart[1], s.symmetry.number]")
    assert s[0] == ["Si"] * 8 and s[1] == 8 and s[3] == 227
    # the formula units survive packing (the atomic numbers travel as zn, not z)
    zf = drv.execute_script("const s = window.__mace.doc().structure; return [s.reduced_formula, s.z]")
    assert zf == ["Si", 8], zf
    U.no_errors(drv)


def wait_download(folder, name, timeout=15):
    end = time.time() + timeout
    p = Path(folder) / name
    while time.time() < end:
        busy = [x for x in Path(folder).iterdir() if x.suffix in (".part", ".crdownload")]   # Firefox, Chromium
        if p.exists() and p.stat().st_size > 0 and not busy:
            return p
        time.sleep(0.1)
    raise AssertionError(f"no download {name}: {[x.name for x in Path(folder).iterdir()]}")


def menu_pick(drv, top, sub, cmd):
    """Top menu › submenu › entry with real pointer input, as a student does it:
    a click on the top menu, a hover on the submenu's row, then right into the
    submenu along that row and down to the entry (a diagonal would cross other
    rows), and a click. Rows are measured afresh: an open menu may redraw."""
    ActionChains(drv).move_to_element(H.wait(drv, "return [...document.querySelectorAll('#menus .menu-top')]"
                                                  ".find((b) => b.textContent.trim() === arguments[0])", 5, top)).click().perform()
    row = H.wait(drv, "return [...document.querySelectorAll('.menu .menu-item[aria-haspopup]')]"
                      ".find((b) => b.textContent.trim() === arguments[0])", 5, sub)
    ActionChains(drv).move_to_element(row).pause(0.05).perform()
    r, it = H.wait(drv, """const row = [...document.querySelectorAll('.menu .menu-item[aria-haspopup]')]
            .find((b) => b.textContent.trim() === arguments[0]);
        const it = document.querySelector('.menu [data-cmd-id="' + arguments[1] + '"]');
        return row && it && [row.getBoundingClientRect().toJSON(), it.getBoundingClientRect().toJSON()]""", 5, sub, cmd)
    x0, y0 = int(r["left"] + r["width"] / 2), int(r["top"] + r["height"] / 2)      # where the pointer is
    x1, y1 = int(it["left"] + 16), int(it["top"] + it["height"] / 2)
    ActionChains(drv).move_by_offset(x1 - x0, 0).pause(0.05).move_by_offset(0, y1 - y0).pause(0.05).click().perform()


def test_export_formats_download_and_no_write(fresh):
    drv, base, root, _, _, downloads = fresh
    before = U.tree_md5(root)
    U.open_doc(drv, "structures/NaCl.cif")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = ase_io.read(str(H.EXAMPLES / "structures" / "NaCl.cif"))
    for cmd, name, reader in (("file.export.CIF", "NaCl.cif", "cif"), ("file.export.POSCAR", "NaCl.vasp", "vasp"),
                              ("file.export.XYZ", "NaCl.xyz", "xyz"), ("file.export.extXYZ", "NaCl.extxyz", "extxyz")):
        # each export is a real click on its menu entry, as a student makes it: Chromium-based
        # browsers (Chrome, Brave, Edge) let a page download a second file only after a user
        # input since the first; four scripted runs in a row are blocked from the second on
        menu_pick(drv, "File", "Export Structure", cmd)
        p = wait_download(downloads, name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = ase_io.read(str(p), format=reader)
        assert len(a) == len(ref) and sorted(a.get_chemical_symbols()) == sorted(ref.get_chemical_symbols()), cmd
    assert U.tree_md5(root) == before                      # downloads never touch --dir
    # the P1 CIF is what Save As writes for the same document
    cif = (Path(downloads) / "NaCl.cif").read_bytes()
    s = U.req(base, "api/structure?path=structures/NaCl.cif")[2]
    st, _, out = U.post(base, "api/structure/save", {"open": {"path": "structures/NaCl.cif", "expected": s["file"]},
                                                     "path": "structures/NaCl_saveas.cif", "format": "cif"})
    assert st == 200 and (root / "structures" / "NaCl_saveas.cif").read_bytes() == cif
    (root / "structures" / "NaCl_saveas.cif").unlink()
    U.no_errors(drv)


def test_submenu_stays_open_on_a_slow_move_into_it(fresh):
    """A pointer that slides along File › Export Structure into its submenu (many
    small moves, as a hand makes them) keeps both menus open: crossing the row's
    label and icon used to rebuild the submenu 120 ms later, and the old
    submenu's own hover timer then closed every menu, so the entry under the
    pointer was gone before the click (any browser; Firefox's input showed it)."""
    drv = fresh[0]
    U.open_doc(drv, "structures/NaCl.cif")
    ActionChains(drv).move_to_element(H.wait(drv, "return [...document.querySelectorAll('#menus .menu-top')]"
                                                  ".find((b) => b.textContent.trim() === 'File')", 5)).click().perform()
    row = H.wait(drv, "return [...document.querySelectorAll('.menu .menu-item[aria-haspopup]')]"
                      ".find((b) => b.textContent.trim() === 'Export Structure')", 5)
    ac = ActionChains(drv).move_to_element_with_offset(row, -row.size["width"] // 2 + 8, 0)   # its left end
    ac.pause(0.3).perform()                                                              # the submenu opens
    H.wait(drv, "return document.querySelectorAll('.menu').length === 2", 5)
    r, it = drv.execute_script("""const it = document.querySelector('.menu [data-cmd-id="file.export.CIF"]');
        return [arguments[0].getBoundingClientRect().toJSON(), it.getBoundingClientRect().toJSON()]""", row)
    x, y = int(r["left"] + 8), int(r["top"] + r["height"] / 2)
    x1 = int(it["left"] + it["width"] / 2)
    ac = ActionChains(drv, duration=16)                     # a hand's speed: 6 px every 16 ms
    for _ in range((x1 - x) // 6):                          # rightwards along the row, into the submenu
        ac.move_by_offset(6, 0)
    ac.pause(0.5).perform()
    assert drv.execute_script("return [...document.querySelectorAll('.menu')].map((m) => m.getAttribute('aria-label'))") == \
        ["File", "Export Structure"]
    downloads = fresh[5]
    ActionChains(drv).click().perform()
    wait_download(downloads, "NaCl.cif")
    U.no_errors(drv)


def test_export_qe_disabled_with_reason(fresh):
    drv = fresh[0]
    U.open_doc(drv, "structures/MgO.cif")
    assert drv.execute_script("return window.__mace.menuReason('file.export.QE')") == "Needs the Quantum ESPRESSO backend"
    assert drv.execute_script("return window.__mace.menuReason('file.export.CIF')") == ""
    tips = drv.execute_script("""const c = window.__mace.commands.filter((x) => x.id.startsWith('file.export.'));
        return c.map((x) => x.title)""")
    assert tips == ["CIF (P1)", "CIF (Symmetrized)", "POSCAR", "XYZ", "extXYZ", "Quantum ESPRESSO Input"]


def test_held_claim_on_reload_keeps_doc(fresh):
    """A view-only held structure open in a tab survives a reload: the
    reloaded page claims it back and reopens it."""
    drv, base = fresh[0], fresh[1]
    H.wait(drv, "return !!window.__mace.clientId()", 10)
    hid = drv.execute_async_script("""const done = arguments[0];
        window.MACE.api('test/held/put', {body: {atoms: 6000}}).then((r) => done(r.held), (e) => done('ERR ' + e.message))""")
    assert hid.startswith("h"), hid
    drv.execute_async_script("const done = arguments[1]; window.__mace.openHeld(arguments[0], 'Held 6000').then(() => done(1))", hid)
    H.wait(drv, "return window.__mace.doc() && window.__mace.doc().title === 'Held 6000' && window.__mace.doc().held", 10)
    assert U.doc(drv)["n"] == 6000
    H.load(drv, base)                                          # the reload
    H.wait(drv, "return window.__mace.docs().some((d) => d.title === 'Held 6000')", 20)
    assert U.req(base, "api/test/held")[2]["pinned"] >= 1
    # closing the tab releases it
    drv.execute_script("window.__mace.closeAll()")
    end = time.time() + 5
    while time.time() < end and U.req(base, f"api/held?id={hid}")[0] == 200:
        time.sleep(0.1)
    assert U.req(base, f"api/held?id={hid}")[0] == 404
    U.no_errors(drv)


def test_changed_on_disk_banner_reload(fresh):
    drv, base, root = fresh[0], fresh[1], fresh[2]
    U.open_doc(drv, "structures/LiF.cif")
    p = root / "structures" / "LiF.cif"
    original = p.read_text()
    try:
        drv.execute_script("window.dispatchEvent(new Event('focus'))")
        time.sleep(0.5)
        assert drv.execute_script("return document.querySelector('#disk-banner').hidden")
        p.write_text(original + "\n# touched\n")
        drv.execute_script("window.__mace.checkDisk()")
        H.wait(drv, "return !document.querySelector('#disk-banner').hidden", 5)
        assert "LiF.cif changed on disk." in drv.execute_script("return document.querySelector('#disk-banner').textContent")
        U.shot(drv, "changed_on_disk_banner")
        drv.execute_script("document.querySelector('#disk-banner [data-v=reload]').click()")
        H.wait(drv, "return document.querySelector('#disk-banner').hidden && window.__mace.doc() && window.__mace.doc().title === 'LiF.cif'", 10)
        st = drv.execute_script("return window.__mace.doc().data ? null : 1")
        assert U.doc(drv)["n"] == 8
        # Ignore keeps it quiet for that version of the file
        p.write_text(original + "\n# touched again\n")
        drv.execute_script("window.__mace.checkDisk()")
        H.wait(drv, "return !document.querySelector('#disk-banner').hidden", 5)
        drv.execute_script("document.querySelector('#disk-banner [data-v=ignore]').click()")
        drv.execute_script("window.__mace.checkDisk()")
        time.sleep(0.6)
        assert drv.execute_script("return document.querySelector('#disk-banner').hidden")
    finally:
        p.write_text(original)
    U.no_errors(drv)


def test_changed_on_disk_checked_right_after_the_page_loads(fresh):
    """The first focus check after a page load runs, however soon it comes: the
    2 s throttle measured from performance.now(), which starts at 0 on a load,
    with 0 meaning 'never checked', so every check in a page's first 2 s was
    dropped (and the test hook's diskCheckAt = 0 too: a flaky banner test in
    Chromium, which loads the page fast enough to be inside that window)."""
    drv, base, root = fresh[0], fresh[1], fresh[2]
    p = root / "structures" / "LiF.cif"
    original = p.read_text()
    try:
        H.load(drv, base)
        U.open_doc(drv, "structures/LiF.cif", wait_atoms=False)
        p.write_text(original + "\n# touched\n")
        age = drv.execute_script("window.dispatchEvent(new Event('focus')); return performance.now()")
        H.wait(drv, "return !document.querySelector('#disk-banner').hidden", 5)
        print(f"checked {age:.0f} ms after the page load")
    finally:
        p.write_text(original)
    U.no_errors(drv)


def test_prefetch_on_hover_one_request(fresh):
    drv = fresh[0]
    drv.execute_script("window.__mace.closeAll()")
    U.open_doc(drv, "structures/MgO.cif")         # the editor is loaded
    drv.execute_script("performance.clearResourceTimings()")
    from selenium.webdriver.common.action_chains import ActionChains
    row = drv.find_element("css selector", '#file-list .file[data-path="structures/diamond.cif"]')
    # in the middle of the list: at its bottom edge the Recent section's header can cover a row
    drv.execute_script("arguments[0].scrollIntoView({block: 'center'})", row)
    time.sleep(0.2)
    ActionChains(drv).move_to_element(row).perform()
    time.sleep(0.5)
    other = drv.find_element("css selector", '#file-list .file[data-path="structures/hBN.cif"]')
    ActionChains(drv).move_to_element(other).pause(0.05).move_to_element(row).perform()
    time.sleep(0.5)
    hits = drv.execute_script("""return performance.getEntriesByType('resource')
        .filter((r) => r.name.includes('api/structure?') && r.name.includes('diamond')).map((r) => r.transferSize)""")
    assert len(hits) == 1, hits
    drv.execute_script("performance.clearResourceTimings()")
    U.open_doc(drv, "structures/diamond.cif")
    sizes = drv.execute_script("""return performance.getEntriesByType('resource')
        .filter((r) => r.name.includes('api/structure?') && r.name.includes('diamond')).map((r) => r.transferSize)""")
    assert sizes == [], sizes                                           # the open used the prefetched one
    assert U.doc(drv)["n"] == 8
    U.no_errors(drv)


def test_trajectory_doc_uses_summary_and_steps(fresh):
    """An OPT output opens from /api/out/summary (never /api/trajectory);
    scrubbing to a step asks for that step (and its neighbors, prefetched)
    once, and Edit This Step uses it."""
    drv = fresh[0]
    drv.execute_script("window.__mace.closeAll(); performance.clearResourceTimings()")
    drv.execute_script("document.dispatchEvent(new CustomEvent('mace-open', {detail: 'outputs/Ag2Br3_opt_unfinished.out'}))")
    H.wait(drv, "return window.__mace.doc() && window.__mace.doc().kind === 'trajectory'", 20)
    names = lambda: drv.execute_script("return performance.getEntriesByType('resource').map((r) => r.name).filter((n) => n.includes('/api/'))")
    H.wait(drv, "return !!window.MACE.state.traj && window.MACE.state.traj.frames.length === 6", 10)
    n = names()
    assert any("api/out/summary?" in x for x in n) and not any("api/trajectory?" in x for x in n), n
    assert not any("api/out/step?" in x for x in n), n
    assert drv.execute_script("return window.MACE.state.traj.frames.map((f) => !!f)") == [True, False, False, False, False, True]
    drv.execute_script("performance.clearResourceTimings(); const el = document.querySelector('#o-frame'); el.value = 2; el.dispatchEvent(new Event('input'))")
    H.wait(drv, "return !!window.MACE.state.traj.frames[2] && window.MACE.state.cmp && document.querySelector('#o-frame-out').textContent.startsWith('2 ')", 10)
    time.sleep(0.4)
    steps = sorted(int(x.split("step=")[1].split("&")[0]) for x in names() if "api/out/step?" in x)
    assert steps == [1, 2, 3], steps
    drv.execute_script("performance.clearResourceTimings(); const el = document.querySelector('#o-frame'); el.value = 3; el.dispatchEvent(new Event('input'))")
    H.wait(drv, "return document.querySelector('#o-frame-out').textContent.startsWith('3 ')", 10)
    time.sleep(0.4)
    steps = sorted(int(x.split("step=")[1].split("&")[0]) for x in names() if "api/out/step?" in x)
    assert steps == [4], steps                          # 2 and 3 were already here
    U.no_errors(drv)
