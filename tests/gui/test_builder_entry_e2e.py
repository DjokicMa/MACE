"""The builders' commands, menus and other entry points (plan T12): every
Phase 4 id is real (enabled on a structure, with a reason elsewhere), each
dialog loads lazily and stays small, keeps focus and Esc, fits 1024x640, the
classic panel's buttons and the Inspector's Cell section open the dialogs,
the Result control is the same everywhere, and In place is one History
entry for each builder."""
import gzip
import time
from pathlib import Path

import pytest

import guiharness as H
import p4util as U

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")

STATIC = H.REPO_ROOT / "mace" / "gui" / "static"
P4_IDS = ["build.supercell", "build.slab", "build.defect.vacancy", "build.defect.substitution", "build.defect.interstitial",
          "build.defect.antisite", "build.transform", "build.symmetrize", "build.clean"]
DIALOGS = {"build.supercell": ("supercell", "supercell_dialog.js"), "build.slab": ("slab", "slab_dialog.js"),
           "build.defect.vacancy": ("defect", "defect_dialog.js"), "build.transform": ("transform", "transform_dialog.js"),
           "build.symmetrize": ("symmetrize", "symmetrize_dialog.js"), "build.clean": ("clean", "clean_panel.js")}


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    a = U.start_app(tmp_path_factory.mktemp("p4entry"))
    try:
        yield a
    finally:
        U.stop_app(a)


@pytest.fixture
def fresh(app):
    U.reset(app[0])
    return app


def wait_frame(drv, fid, timeout=15):
    H.wait(drv, f"const f = window.__mace.builderFrame(); return f && f.id === '{fid}'", timeout)
    time.sleep(0.2)


def close_frame(drv):
    from selenium.webdriver.common.keys import Keys
    drv.execute_script("const d = document.querySelector('dialog.builder-dialog[open] [data-act=cancel]'); if (d) d.click()")
    H.wait(drv, "return !document.querySelector('dialog.builder-dialog[open]')", 5)


def test_build_commands_not_later(fresh):
    drv = fresh[0]
    cmds = {c["id"]: c for c in drv.execute_script("return window.__mace.commands")}
    for cid in P4_IDS:
        assert cid in cmds and not cmds[cid]["soon"], cid
        assert drv.execute_script("return window.__mace.menuReason(arguments[0])", cid), cid      # welcome: a reason
    U.open_doc(drv, "structures/MgO.cif")
    for cid in P4_IDS:
        assert drv.execute_script("return window.__mace.menuReason(arguments[0])", cid) == "", cid


def test_builder_dialogs_lazy_and_under_20kb(fresh):
    drv = fresh[0]
    frame = len(gzip.compress((STATIC / "js" / "lazy" / "builder_util.js").read_bytes(), 9))
    css = len(gzip.compress((STATIC / "css" / "builders.css").read_bytes(), 9))
    for cid, (fid, f) in DIALOGS.items():
        own = len(gzip.compress((STATIC / "js" / "lazy" / f).read_bytes(), 9))
        assert own + frame + css <= 20 * 1024, (f, own, frame, css)
    names = drv.execute_script("return performance.getEntriesByType('resource').map((r) => r.name)")
    assert not any(f in n for n in names for _, f in DIALOGS.values()), "a builder dialog loaded at boot"


@pytest.mark.parametrize("cid", list(DIALOGS))
def test_builder_dialogs_focus_return_and_esc(fresh, cid):
    from selenium.webdriver.common.keys import Keys
    drv = fresh[0]
    U.open_doc(drv, "structures/MgO.cif")
    drv.execute_script("document.querySelector('.viewer-wrap').focus()")
    drv.execute_script("window.__mace.run(arguments[0])", cid)
    wait_frame(drv, DIALOGS[cid][0])
    assert drv.execute_script("return !!document.activeElement.closest('dialog.builder-dialog')")
    drv.execute_script("document.activeElement.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', bubbles: true, cancelable: true}))")
    H.wait(drv, "return !document.querySelector('dialog.builder-dialog[open]')", 5)
    H.wait(drv, "return !!document.activeElement.closest('.viewer-wrap')", 5)
    assert not drv.execute_script("return window.__mace.ctx().previewing")


@pytest.mark.parametrize("cid", list(DIALOGS))
def test_builder_dialogs_fit_1024x640(fresh, cid):
    drv = fresh[0]
    H.set_viewport(drv, 1024, 640)
    U.open_doc(drv, "structures/MgO.cif")
    drv.execute_script("window.__mace.run(arguments[0])", cid)
    wait_frame(drv, DIALOGS[cid][0])
    r = drv.execute_script("const r = document.querySelector('dialog.builder-dialog[open]').getBoundingClientRect(); return [r.left, r.top, r.right, r.bottom, innerWidth, innerHeight]")
    assert r[0] >= 0 and r[1] >= 0 and r[2] <= r[4] and r[3] <= r[5], r
    foot = drv.execute_script("const r = document.querySelector('dialog.builder-dialog[open] .bld-go').getBoundingClientRect(); return r.bottom <= innerHeight && r.width > 0")
    assert foot
    close_frame(drv)


def test_classic_buttons_open_new_dialogs(fresh):
    drv = fresh[0]
    U.open_doc(drv, "structures/MgO.cif")
    H.wait(drv, "return !!document.querySelector('#inspector #e-supercell')", 10)
    labels = drv.execute_script("return ['#e-supercell', '#e-primitive', '#e-symmetrize', '#e-clean'].map((s) => document.querySelector(s).textContent.trim())")
    assert labels == ["Make", "Primitive", "Symmetrize", "Clean"], labels         # the Phase 3 labels stay
    drv.execute_script("document.querySelector('#e-n1').value = 3; document.querySelector('#e-n2').value = 3; document.querySelector('#e-n3').value = 1")
    for sel, fid in (("#e-supercell", "supercell"), ("#e-primitive", "transform"), ("#e-symmetrize", "symmetrize"), ("#e-clean", "clean")):
        drv.execute_script("document.querySelector(arguments[0]).click()", sel)
        wait_frame(drv, fid)
        if fid == "supercell":
            n = drv.execute_script("return [...document.querySelectorAll('dialog.builder-dialog[open] [data-n]')].map((x) => Number(x.value))")
            assert n == [3, 3, 1], n
        close_frame(drv)
    assert U.ed_state(drv)["rows"] == ["Opened MgO.cif"]          # nothing ran without a Create or Apply


def test_inspector_cell_buttons_open_builders(fresh):
    drv = fresh[0]
    U.open_doc(drv, "structures/MgO.cif")
    H.wait(drv, "return !!document.querySelector('#inspector .ins-build')", 10)
    btns = drv.execute_script("return [...document.querySelectorAll('#inspector .insp-section[data-key=cell] button.tb-btn-text')].map((b) => b.textContent.trim())")
    assert btns == ["Supercell…", "Transform…", "Edit Cell…"], btns
    find = "[...document.querySelectorAll('#inspector .ins-build')].find((b) => b.textContent.trim() === arguments[0])"
    for label, fid in (("Supercell…", "supercell"), ("Transform…", "transform")):
        # the closed dialog's preview ends, and the editable Cell section comes back, on the next animation frame
        H.wait(drv, f"return !!{find}", 10, label)
        drv.execute_script("[...document.querySelectorAll('#inspector .ins-build')].find((b) => b.textContent.trim() === arguments[0]).click()", label)
        wait_frame(drv, fid)
        close_frame(drv)


def test_builder_result_control_same_in_every_dialog(fresh):
    drv = fresh[0]
    U.open_doc(drv, "structures/MgO.cif")
    seen = {}
    for cid in ("build.supercell", "build.defect.vacancy", "build.transform", "build.symmetrize", "build.slab"):
        drv.execute_script("window.__mace.run(arguments[0])", cid)
        wait_frame(drv, DIALOGS[cid][0])
        seen[cid] = drv.execute_script("return [...document.querySelectorAll('dialog.builder-dialog[open] .bld-result [data-r]')].map((b) => [b.dataset.r, b.textContent.trim()])")
        close_frame(drv)
    for cid in ("build.supercell", "build.defect.vacancy", "build.transform", "build.symmetrize"):
        assert seen[cid] == [["new", "New tab"], ["inplace", "This structure"]], (cid, seen[cid])
    assert seen["build.slab"] == [["new", "New tab"]]                   # a slab never replaces its bulk
    # remembered per dialog
    drv.execute_script("window.__mace.run('build.supercell')")
    wait_frame(drv, "supercell")
    drv.execute_script("document.querySelector('dialog.builder-dialog[open] [data-r=inplace]').click()")
    close_frame(drv)
    drv.execute_script("window.__mace.run('build.supercell')")
    wait_frame(drv, "supercell")
    assert drv.execute_script("return window.__mace.builderFrame().result") == "inplace"
    drv.execute_script("document.querySelector('dialog.builder-dialog[open] [data-r=new]').click()")
    close_frame(drv)


@pytest.mark.parametrize("cid,setup,label", [
    ("build.supercell", None, "Supercell"),
    ("build.transform", None, "Transform to Primitive"),
    ("build.symmetrize", None, "Symmetrize"),
    ("build.defect.vacancy", "pick", "Vacancy"),
])
def test_builder_result_is_one_history_entry_in_place(fresh, cid, setup, label):
    drv = fresh[0]
    U.open_doc(drv, "structures/NaCl.cif")
    drv.execute_script("window.__mace.run(arguments[0])", cid)
    wait_frame(drv, DIALOGS[cid][0])
    drv.execute_script("const b = document.querySelector('dialog.builder-dialog[open] [data-r=inplace]'); if (b) b.click()")
    if setup == "pick":
        H.wait(drv, "return document.querySelectorAll('dialog.builder-dialog[open] li[data-i]').length > 0", 15)
        drv.execute_script("document.querySelector('dialog.builder-dialog[open] li[data-i]').click()")
    H.wait(drv, "const f = window.__mace.builderFrame(); return f && !f.goDisabled && f.result === 'inplace'", 30)
    drv.execute_script("document.querySelector('dialog.builder-dialog[open] .bld-go').click()")
    H.wait(drv, "const e = window.__mace.editor.doc(); return e && e.rows.length === 2 && e.pending === 0 && !e.waiting", 30)
    rows = U.ed_state(drv)["rows"]
    assert rows[0] == "Opened NaCl.cif" and rows[1].startswith(label), rows
    assert drv.execute_script("return window.__mace.docs().length") == 1
    U.no_errors(drv)
