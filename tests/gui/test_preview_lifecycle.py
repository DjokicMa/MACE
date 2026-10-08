"""The builders' shared frame and the preview lifecycle (plan T3b, B10, B11)
in headless Firefox, on the test-only Identity builder (MACE_GUI_TEST_HOOKS)
so the frame is proven before any real builder: lazy and small, focus and
Esc, the Result control remembered, This structure off above 5000 atoms,
editing off while a preview is open, Mod+Z discards, Enter and Esc scoped
to the panel, the preview kept with its document across tab switches,
closing discards, and a slow build shown in the dialog or handed to Tasks."""
import gzip
import time
from pathlib import Path

import pytest

import guiharness as H
import p4util as U

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")
from selenium.webdriver.common.keys import Keys  # noqa: E402

STATIC = Path(__file__).resolve().parents[2] / "mace" / "gui" / "static"


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("p4frame")
    a = U.start_app(tmp)
    try:
        yield a
    finally:
        U.stop_app(a)


@pytest.fixture
def fresh(app):
    U.reset(app[0])
    drv = app[0]
    drv.execute_script("try { Object.keys(localStorage).filter((k) => k.startsWith('mace.gui.builder.')).forEach((k) => localStorage.removeItem(k)); } catch (e) {}")
    return app


def open_identity(drv, **opts):
    drv.execute_async_script("const done = arguments[1]; window.__mace.openTestBuilder(arguments[0]).then(() => done(1), (e) => done(e.message))", opts)
    H.wait(drv, "return !!document.querySelector('dialog.builder-dialog[open]')", 10)
    time.sleep(0.2)


def frame(drv):
    return drv.execute_script("return window.__mace.builderFrame()")


def models(drv):
    return drv.execute_script("return window.__mace.viewport.models()")


def test_builder_frame_lazy_and_small(fresh):
    drv = fresh[0]
    lazy = drv.execute_script("""return performance.getEntriesByType('resource').filter((r) => r.name.includes('builder_util') || r.name.includes('builders.css')).length""")
    assert lazy == 0                                       # nothing of it at boot
    size = len(gzip.compress((STATIC / "js" / "lazy" / "builder_util.js").read_bytes(), 9))
    css = len(gzip.compress((STATIC / "css" / "builders.css").read_bytes(), 9))
    assert size <= 8 * 1024, size                          # the frame alone: <= 8 KB gzipped
    assert size + css <= 20 * 1024, (size, css)


def test_builder_frame_preview_ghost_and_esc(fresh):
    drv = fresh[0]
    U.open_doc(drv, "structures/MgO.cif")
    opener = drv.execute_script("document.querySelector('#toolbar [data-cmd=\"file.open\"]').focus(); return document.activeElement.dataset.cmd")
    open_identity(drv)
    H.wait(drv, "const m = window.__mace.viewport.models(); return m && m.ghost === 8", 10)
    m = models(drv)
    assert m["ghost"] == 8 and abs(m["opacity"] - 0.3) < 1e-6, m
    # a parameter change asks again (debounced), the ghost stays current
    drv.execute_script("const i = document.querySelector('.builder-dialog [data-k=delay]'); i.value = '100'; i.dispatchEvent(new Event('input', {bubbles: true}))")
    time.sleep(0.6)
    assert models(drv)["ghost"] == 8
    U.keys(drv, Keys.ESCAPE)
    H.wait(drv, "return !document.querySelector('dialog.builder-dialog[open]')", 5)
    m = models(drv)
    assert m["ghost"] == 0 and not m["cell"], m
    assert drv.execute_script("return document.activeElement.dataset.cmd") == opener
    assert not drv.execute_script("return window.__mace.ctx().previewing")
    U.no_errors(drv)


def test_builder_frame_fits_1024x640(fresh):
    drv = fresh[0]
    H.set_viewport(drv, 1024, 640)
    U.open_doc(drv, "structures/MgO.cif")
    open_identity(drv)
    st = drv.execute_script("""const d = document.querySelector('dialog.builder-dialog[open]'), r = d.getBoundingClientRect();
        const clipped = [...d.querySelectorAll('button, label, .bld-readout, h2')].filter((el) => el.offsetParent && el.scrollWidth > el.clientWidth + 1).map((el) => el.textContent.trim());
        const off = [...d.querySelectorAll('button, input')].filter((el) => { const b = el.getBoundingClientRect(); return el.offsetParent && (b.right > innerWidth || b.bottom > innerHeight || b.left < 0 || b.top < 0); }).length;
        return {top: r.top, bottom: r.bottom, right: r.right, left: r.left, W: innerWidth, H: innerHeight, clipped, off}""")
    assert st["bottom"] <= st["H"] and st["right"] <= st["W"] and st["left"] >= 0, st
    assert not st["clipped"] and st["off"] == 0, st
    U.shot(drv, "builder_frame_1024x640")
    U.keys(drv, Keys.ESCAPE)
    H.set_viewport(drv, 1280, 800)


def test_builder_frame_result_control_remembered(fresh):
    drv, base = fresh[0], fresh[1]
    U.open_doc(drv, "structures/MgO.cif")
    open_identity(drv)
    assert frame(drv)["result"] == "new"
    drv.execute_script("document.querySelector('.builder-dialog .bld-result [data-r=inplace]').click()")
    assert frame(drv)["result"] == "inplace"
    assert drv.execute_script("return document.querySelector('.builder-dialog .bld-name').hidden")
    U.keys(drv, Keys.ESCAPE)
    H.load(drv, base)
    U.open_doc(drv, "structures/MgO.cif")
    open_identity(drv)
    assert frame(drv)["result"] == "inplace"
    checked = drv.execute_script("return document.querySelector('.builder-dialog .bld-result [aria-checked=true]').dataset.r")
    assert checked == "inplace"
    U.keys(drv, Keys.ESCAPE)


def test_builder_frame_in_place_disabled_over_5000(fresh):
    drv = fresh[0]
    U.open_doc(drv, "structures/MgO.cif")
    open_identity(drv, n=85184)
    b = drv.execute_script("const b = document.querySelector('.builder-dialog .bld-result [data-r=inplace]'); return {dis: b.getAttribute('aria-disabled'), why: b.title}")
    assert b["dis"] == "true" and b["why"] == "In place is limited to 5,000 atoms; this makes 85,184", b
    drv.execute_script("document.querySelector('.builder-dialog .bld-result [data-r=inplace]').click()")
    assert frame(drv)["result"] == "new"
    U.keys(drv, Keys.ESCAPE)


def test_previewing_disables_edits_with_reason(fresh):
    drv = fresh[0]
    U.open_doc(drv, "structures/MgO.cif")
    open_identity(drv)
    for cmd in ("edit.delete", "edit.replaceElement", "tool.move", "build.addAtom", "edit.editCell"):
        why = drv.execute_script("return window.__mace.menuReason(arguments[0])", cmd)
        assert why == "Accept or discard the preview first", (cmd, why)
    assert drv.execute_script("return window.__mace.ctx().previewing")
    tb = drv.execute_script("const b = document.querySelector('#toolbar [data-cmd=\"tool.move\"]'); return [b.getAttribute('aria-disabled'), b.dataset.reason]")
    assert tb == ["true", "Accept or discard the preview first"], tb
    U.shot(drv, "previewing_reason")
    # the viewport still orbits
    v0 = drv.execute_script("return window.MACE.state.viewers['viewer-main'].getView()")
    drv.execute_script("window.__mace.viewport.camera().orbit(30, 10)")
    assert drv.execute_script("return window.MACE.state.viewers['viewer-main'].getView()") != v0
    U.keys(drv, Keys.ESCAPE)
    assert drv.execute_script("return window.__mace.menuReason('edit.editCell')") == ""


def test_mod_z_discards_preview(fresh):
    """Mod+Z closes the preview and never undoes a committed entry."""
    drv = fresh[0]
    U.open_doc(drv, "structures/MgO.cif")
    drv.execute_script("window.__mace.run('build.symModeP1'); window.__mace.editor.select([0], 'replace')")
    drv.execute_script("window.__mace.run('edit.delete')")
    U.wait_idle(drv)
    rows = U.ed_state(drv)["rows"]
    open_identity(drv)
    drv.execute_script("document.querySelector('#viewer-main').closest('.viewer-wrap').focus()")
    U.keys(drv, Keys.CONTROL, "z")
    H.wait(drv, "return !document.querySelector('dialog.builder-dialog[open]')", 5)
    st = U.ed_state(drv)
    assert st["rows"] == rows and st["n"] == 7 and st["pos"] == len(rows) - 1, st      # the delete stays
    assert models(drv)["ghost"] == 0
    # with no preview Mod+Z undoes again
    drv.execute_script("document.querySelector('#viewer-main').closest('.viewer-wrap').focus()")
    U.keys(drv, Keys.CONTROL, "z")
    U.wait_idle(drv)
    assert U.ed_state(drv)["n"] == 8


def test_esc_and_enter_scoped_to_panel(fresh):
    drv = fresh[0]
    U.open_doc(drv, "structures/MgO.cif")
    drv.execute_script("window.__mace.run('tool.lasso')")
    open_identity(drv)
    # Esc in the viewport closes the panel, not the tool
    drv.execute_script("document.querySelector('#viewer-main').closest('.viewer-wrap').focus()")
    U.keys(drv, Keys.ESCAPE)
    H.wait(drv, "return !document.querySelector('dialog.builder-dialog[open]')", 5)
    assert drv.execute_script("return window.__mace.viewState().tool") == "lasso"
    U.keys(drv, Keys.ESCAPE)                                   # now Esc leaves the tool
    assert drv.execute_script("return window.__mace.viewState().tool") == "select"
    # Enter in a field of the panel creates (a new tab)
    n0 = len(drv.execute_script("return window.__mace.docs()"))
    open_identity(drv)
    drv.execute_script("document.querySelector('.builder-dialog [data-k=delay]').focus()")
    U.keys(drv, Keys.ENTER)
    H.wait(drv, f"return window.__mace.docs().length === {n0 + 1}", 10)
    d = U.doc(drv)
    assert d["title"] == "MgO_copy" and d["dirty"] and d["untitled"] and d["n"] == 8, d
    assert not drv.execute_script("return document.querySelector('dialog.builder-dialog[open]')")
    U.no_errors(drv)


def test_proposal_survives_tab_switch(fresh):
    drv = fresh[0]
    U.open_doc(drv, "structures/MgO.cif")
    open_identity(drv)
    H.wait(drv, "return window.__mace.viewport.models().ghost === 8", 10)
    U.open_doc(drv, "structures/Si.cif")
    time.sleep(0.4)
    assert drv.execute_script("return document.querySelector('dialog.builder-dialog').hidden")
    assert models(drv)["ghost"] == 0 and not drv.execute_script("return window.__mace.ctx().previewing")
    drv.execute_script("window.__mace.activate(0)")
    H.wait(drv, "return !document.querySelector('dialog.builder-dialog').hidden && window.__mace.viewport.models().ghost === 8", 10)
    assert drv.execute_script("return window.__mace.ctx().previewing")
    U.keys(drv, Keys.ESCAPE)


def test_panel_close_and_tab_close_discard(fresh):
    drv = fresh[0]
    U.open_doc(drv, "structures/MgO.cif")
    open_identity(drv)
    drv.execute_script("document.querySelector('dialog.builder-dialog [data-close]').click()")
    H.wait(drv, "return !document.querySelector('dialog.builder-dialog[open]')", 5)
    assert models(drv)["ghost"] == 0 and not drv.execute_script("return window.__mace.ctx().previewing")
    open_identity(drv)
    drv.execute_script("window.__mace.run('file.closeTab')")          # no question: a preview is not unsaved work
    time.sleep(0.4)
    assert not drv.execute_script("return document.querySelector('dialog.builder-dialog[open]')")
    assert drv.execute_script("return window.__mace.docs().length") == 0
    U.no_errors(drv)


def test_detached_build_progress_in_dialog_then_tab(fresh):
    drv = fresh[0]
    U.open_doc(drv, "structures/MgO.cif")
    open_identity(drv, delay=2500, n=3000)          # 3000 > 2000 atoms: the pool, with progress
    drv.execute_script("document.querySelector('.builder-dialog .bld-go').click()")
    H.wait(drv, "const f = window.__mace.builderFrame(); return f && f.busy && /Building… \\d+%/.test(f.goLabel)", 10)
    lbl = frame(drv)["goLabel"]
    H.wait(drv, "const f = window.__mace.builderFrame(); return f && /Building… ([1-9]\\d?)%/.test(f.goLabel)", 10)
    U.shot(drv, "detached_build_progress")
    assert drv.execute_script("return document.querySelector('.builder-dialog .bld-params').disabled")
    H.wait(drv, "return window.__mace.docs().length === 2", 20)
    assert U.doc(drv)["title"] == "MgO_copy"
    assert not drv.execute_script("return document.querySelector('dialog.builder-dialog[open]')")
    assert lbl.startswith("Building…")
    U.no_errors(drv)


def test_detached_build_cancel_in_dialog(fresh):
    drv = fresh[0]
    U.open_doc(drv, "structures/MgO.cif")
    open_identity(drv, delay=8000, n=3000)
    drv.execute_script("document.querySelector('.builder-dialog .bld-go').click()")
    H.wait(drv, "const f = window.__mace.builderFrame(); return f && f.busy", 10)
    time.sleep(0.6)
    drv.execute_script("document.querySelector('.builder-dialog [data-act=cancel]').click()")
    H.wait(drv, "const f = window.__mace.builderFrame(); return f && !f.busy", 10)
    assert drv.execute_script("return window.__mace.docs().length") == 1
    assert not drv.execute_script("return document.querySelector('.builder-dialog .bld-params').disabled")
    assert drv.execute_script("return window.__mace.tasks().some((t) => t.state === 'canceled')")
    U.keys(drv, Keys.ESCAPE)


def test_detached_build_handed_to_tasks_on_close(fresh):
    drv = fresh[0]
    U.open_doc(drv, "structures/MgO.cif")
    open_identity(drv, delay=2500, n=3000)
    drv.execute_script("document.querySelector('.builder-dialog .bld-go').click()")
    H.wait(drv, "const f = window.__mace.builderFrame(); return f && f.busy", 10)
    drv.execute_script("document.querySelector('dialog.builder-dialog [data-close]').click()")
    time.sleep(0.3)
    assert not drv.execute_script("return document.querySelector('dialog.builder-dialog[open]')")
    assert drv.execute_script("return window.__mace.tasks().some((t) => t.state === 'running' || t.state === 'queued')")
    H.wait(drv, "return [...document.querySelectorAll('#toasts > *')].some((t) => t.textContent.includes('Identity ready'))", 20)
    assert drv.execute_script("return window.__mace.docs().length") == 1          # no focus stolen
    drv.execute_script("[...document.querySelectorAll('#toasts > *')].find((t) => t.textContent.includes('Identity ready')).querySelector('button').click()")
    H.wait(drv, "return window.__mace.docs().length === 2", 10)
    assert U.doc(drv)["title"] == "MgO_copy"
    U.no_errors(drv)
