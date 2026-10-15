"""The builder dialogs in headless Firefox (plan T4-T8, T12; spec §7 Phase 4
criteria 1-3): Supercell, Slab from Bulk, Point Defect, Transform Cell,
Symmetrize and Clean Geometry, each with its preview in the viewport, and
the entry points that open them."""
import time
from pathlib import Path

import pytest

import guiharness as H
import p4util as U

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")
from selenium.webdriver.common.keys import Keys  # noqa: E402


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("p4build")
    a = U.start_app(tmp)
    try:
        yield a
    finally:
        U.stop_app(a)


@pytest.fixture
def fresh(app):
    U.reset(app[0])
    app[0].execute_script("try { Object.keys(localStorage).filter((k) => k.startsWith('mace.gui.builder.')).forEach((k) => localStorage.removeItem(k)); } catch (e) {}")
    return app


def dialog_open(drv, id_):
    H.wait(drv, f"return !!document.querySelector('dialog.builder-dialog[data-builder=\"{id_}\"][open]')", 10)
    time.sleep(0.2)


def frame(drv):
    return drv.execute_script("return window.__mace.builderFrame()")


def rev_of(drv, title):
    return drv.execute_script("""const d = window.__mace.docs().find((x) => x.title === arguments[0]);
        const doc = window.__mace.activeDoc(); return d ? d.id : null""", title)


def test_supercell_dialog_mgo_2x2x2_new_tab(fresh):
    """Criterion 1: a 2×2×2 supercell of MgO opens a new tab "MgO_2x2x2" with
    64 atoms and Fm-3m; the MgO tab is unchanged; one build request; nothing written."""
    drv, base, root = fresh[0], fresh[1], fresh[2]
    before = U.tree_md5(root)
    U.open_doc(drv, "structures/MgO.cif")
    mgo = drv.execute_script("const d = window.__mace.activeDoc(); return {id: d.id, n: d.data.numbers.length}")
    drv.execute_script("document.querySelector('#viewer-main').closest('.viewer-wrap').focus()")
    U.keys(drv, Keys.CONTROL, Keys.ALT, "p")
    dialog_open(drv, "supercell")
    H.wait(drv, "const m = window.__mace.viewport.models(); return m && m.ghost === 64 && m.cell", 10)
    f = frame(drv)
    assert f["name"] == "MgO_2x2x2" and f["result"] == "new" and "64 atoms" in f["readout"], f
    U.shot(drv, "supercell_dialog_preview")
    drv.execute_script("performance.clearResourceTimings()")
    drv.execute_script("document.querySelector('.builder-dialog .bld-go').click()")
    H.wait(drv, "return window.__mace.docs().length === 2 && window.__mace.doc().title === 'MgO_2x2x2'", 20)
    time.sleep(0.5)
    d = U.doc(drv)
    assert d["n"] == 64 and d["dirty"] and d["untitled"], d
    H.wait(drv, "return document.querySelector('#st-sg') && document.querySelector('#st-sg').textContent.startsWith('Fm-3m (225)')", 10)
    assert "Supercell 2×2×2 of MgO.cif" in drv.execute_script("return document.querySelector('#s-prov').textContent")
    rows = drv.execute_script("return window.__mace.editor.doc().rows.map((r) => r.label)")
    assert rows == ["Supercell 2×2×2 of MgO.cif"], rows
    builds = drv.execute_script("return performance.getEntriesByType('resource').filter((r) => r.name.includes('api/build/supercell')).length")
    assert builds == 1
    tip = drv.execute_script("return document.querySelector('.doc-tab.active').title")
    assert "Supercell 2×2×2 of MgO.cif" in tip
    # a draft was written at creation, so a reload offers it back
    assert drv.execute_script("return Object.keys(localStorage).some((k) => k.startsWith('mace.gui.draft.') && localStorage.getItem(k).includes('MgO_2x2x2'))")
    # the MgO tab is unchanged
    drv.execute_script("window.__mace.activate(0)")
    time.sleep(0.3)
    d0 = U.doc(drv)
    assert d0["title"] == "MgO.cif" and d0["n"] == 8 and not d0["dirty"]
    assert U.tree_md5(root) == before
    U.shot(drv, "supercell_new_tab")
    U.no_errors(drv)


def test_supercell_matrix_tab_and_presets(fresh):
    drv = fresh[0]
    U.open_doc(drv, "structures/hBN.cif")
    drv.execute_script("window.__mace.run('build.supercell')")
    dialog_open(drv, "supercell")
    assert drv.execute_script("return !!document.querySelector('.builder-dialog [data-preset=r30]')")    # hexagonal: √3 offered
    drv.execute_script("document.querySelector('.builder-dialog [data-preset=r30]').click()")
    time.sleep(0.4)
    f = frame(drv)
    assert f["name"] == "hBN_r3xr3x1" and "12 atoms" in f["readout"], f
    assert drv.execute_script("return document.querySelector('.builder-dialog [data-k=det]').textContent") == "3"
    # a determinant of 0 blocks Create, with the reason inline
    drv.execute_script("const i = document.querySelector('.builder-dialog [data-m=\"22\"]'); i.value = '0'; i.dispatchEvent(new Event('input', {bubbles: true}))")
    time.sleep(0.3)
    f = frame(drv)
    assert f["error"] == "The matrix must have a positive determinant" and f["goDisabled"], f
    U.shot(drv, "supercell_matrix_det0")
    U.keys(drv, Keys.ESCAPE)
    U.open_doc(drv, "structures/MgO.cif")
    drv.execute_script("window.__mace.run('build.supercell')")
    dialog_open(drv, "supercell")
    assert not drv.execute_script("return !!document.querySelector('.builder-dialog [data-preset=r30]')")
    drv.execute_script("document.querySelector('.builder-dialog [data-preset=r45]').click()")
    time.sleep(0.3)
    assert frame(drv)["name"] == "MgO_r2xr2x1"
    # more than 500 atoms: the CRYSTAL cost warning; more than 5000: view-only
    drv.execute_script("document.querySelector('.builder-dialog [data-tab=diag]').click(); const i = document.querySelector('.builder-dialog [data-n=\"0\"]'); i.value = '10'; i.dispatchEvent(new Event('input', {bubbles: true}))")
    time.sleep(0.5)
    ro = frame(drv)["readout"]
    assert "8,000 atoms" in ro and "CRYSTAL cost grows about as N³" in ro and "Opens view-only" in ro, ro
    assert "Preview shows the cell only above 2,000 atoms" in ro
    m = drv.execute_script("return window.__mace.viewport.models()")
    assert m["cell"] and m["ghost"] == 0, m
    U.keys(drv, Keys.ESCAPE)
    U.no_errors(drv)


def test_supercell_in_place_one_history_entry(fresh):
    drv = fresh[0]
    U.open_doc(drv, "structures/NaCl.cif")
    drv.execute_script("window.__mace.run('build.supercell')")
    dialog_open(drv, "supercell")
    drv.execute_script("document.querySelector('.builder-dialog .bld-result [data-r=inplace]').click()")
    drv.execute_script("const l = document.querySelector('.builder-dialog [data-k=link]'); l.checked = false; l.dispatchEvent(new Event('change', {bubbles: true}))")
    drv.execute_script("const ns = document.querySelectorAll('.builder-dialog [data-n]'); ns[0].value = '1'; ns[1].value = '1'; ns[2].value = '2'; ns[2].dispatchEvent(new Event('input', {bubbles: true}))")
    time.sleep(0.4)
    drv.execute_script("document.querySelector('.builder-dialog .bld-go').click()")
    U.wait_idle(drv)
    st = U.ed_state(drv)
    assert st["n"] == 16 and st["rows"][-1] == "Supercell 1×1×2" and len(st["rows"]) == 2, st
    assert len(drv.execute_script("return window.__mace.docs()")) == 1
    U.no_errors(drv)


# --- T5 slab ----------------------------------------------------------------------------

def test_slab_dialog_new_tab_named(fresh):
    """Criterion 2 in the page: Slab from Bulk on MgO, (001), 5 layers, 15 Å:
    the preview pane shows the slab next to the bulk; Create opens MgO_001_5L
    with c ≈ 15 + 4·2.106 Å; the bulk tab is unchanged."""
    drv, root = fresh[0], fresh[2]
    before = U.tree_md5(root)
    U.open_doc(drv, "structures/MgO.cif")
    drv.execute_script("window.__mace.run('build.slab')")
    dialog_open(drv, "slab")
    H.wait(drv, "return window.__maceSlabPane && window.__maceSlabPane.n === 20", 15)
    assert drv.execute_script("return document.querySelector('#viewer-main').closest('.viewer-wrap').classList.contains('has-preview-pane')")
    label = drv.execute_script("return window.__maceSlabPane.label")
    assert label == "Preview: MgO (001), 5 layers, MgO-terminated", label
    f = frame(drv)
    assert f["name"] == "MgO_001_5L" and "5 layers" in f["readout"] and "20 atoms" in f["readout"], f
    assert "Vacuum is ignored by CRYSTAL SLAB (2D)" in drv.execute_script("return document.querySelector('.builder-dialog').textContent")
    U.shot(drv, "slab_dialog_pane")
    drv.execute_script("document.querySelector('.builder-dialog .bld-go').click()")
    H.wait(drv, "return window.__mace.docs().length === 2 && window.__mace.doc().title === 'MgO_001_5L'", 20)
    s = drv.execute_script("const s = window.__mace.doc().structure; return {c: s.lattice.c, n: s.numbers.length, pbc: s.pbc}")
    assert abs(s["c"] - 23.424) < 0.01 and s["n"] == 20 and all(s["pbc"]), s
    assert not drv.execute_script("return document.querySelector('#viewer-main').closest('.viewer-wrap').classList.contains('has-preview-pane')")
    rows = drv.execute_script("return window.__mace.editor.doc().rows.map((r) => r.label)")
    assert rows == ["Slab (001), 5 layers of MgO.cif"], rows
    drv.execute_script("window.__mace.activate(0)")
    time.sleep(0.3)
    assert U.doc(drv)["n"] == 8 and not U.doc(drv)["dirty"]
    assert U.tree_md5(root) == before
    U.no_errors(drv)


def test_slab_never_replaces_bulk(fresh):
    drv = fresh[0]
    U.open_doc(drv, "structures/MgO.cif")
    drv.execute_script("window.localStorage.setItem('mace.gui.builder.slab.result', 'inplace')")
    drv.execute_script("window.__mace.run('build.slab')")
    dialog_open(drv, "slab")
    segs = drv.execute_script("return [...document.querySelectorAll('.builder-dialog .bld-result [data-r]')].map((b) => [b.dataset.r, b.disabled])")
    assert segs == [["new", True]], segs
    tip = drv.execute_script("return document.querySelector('.builder-dialog .bld-result').dataset.desc")
    assert tip == "A slab never replaces the bulk document"
    assert frame(drv)["result"] == "new"
    # (111): two terminations, each with its thumbnail
    drv.execute_script("const h = document.querySelectorAll('.builder-dialog [data-h]'); h[0].value = '1'; h[1].value = '1'; h[2].dispatchEvent(new Event('input', {bubbles: true}))")
    H.wait(drv, "return document.querySelectorAll('.builder-dialog [data-k=terms] li[data-t]').length === 2", 10)
    assert drv.execute_script("return document.querySelectorAll('.builder-dialog [data-k=terms] svg.thumb circle').length") > 0
    drv.execute_script("document.querySelectorAll('.builder-dialog [data-k=terms] li[data-t]')[1].click()")
    H.wait(drv, "return window.__maceSlabPane && /-terminated$/.test(window.__maceSlabPane.label)", 10)
    U.shot(drv, "slab_111_terminations")
    U.keys(drv, Keys.ESCAPE)
    time.sleep(0.3)
    assert not drv.execute_script("return document.querySelector('#viewer-main').closest('.viewer-wrap').classList.contains('has-preview-pane')")
    assert U.doc(drv)["n"] == 8 and len(drv.execute_script("return window.__mace.docs()")) == 1
    U.no_errors(drv)


# --- T6 point defects -------------------------------------------------------------------

def defect_ready(drv):
    H.wait(drv, "return document.querySelectorAll('.builder-dialog [data-k=sites] li[data-i]').length >= 2", 15)
    time.sleep(0.4)


def set_n(drv, n):
    drv.execute_script("const r = document.querySelector('.builder-dialog [name=dsize][value=hand]'); r.checked = true; r.dispatchEvent(new Event('change', {bubbles: true}));"
                       "const ns = document.querySelectorAll('.builder-dialog [data-n]'); arguments[0].forEach((v, k) => { ns[k].value = String(v); });"
                       "ns[2].dispatchEvent(new Event('input', {bubbles: true}))", n)
    time.sleep(0.5)


def test_defect_dialog_names_supercell_and_charge(fresh):
    drv, root = fresh[0], fresh[2]
    before = U.tree_md5(root)
    U.open_doc(drv, "structures/MgO.cif")
    drv.execute_script("window.__mace.run('build.defect.vacancy')")
    dialog_open(drv, "defect")
    defect_ready(drv)
    f = frame(drv)
    assert f["name"] == "MgO_3x3x3_V_Mg_4a", f                   # the image-distance supercell, the first site
    H.wait(drv, "return /nearest defect image 12\\.6/.test(window.__mace.builderFrame().readout)", 10)
    assert "215 atoms" in frame(drv)["readout"]
    m = drv.execute_script("return window.__mace.viewport.models()")
    assert m["halo"] >= 1 and m["vacancies"] == 1 and m["cell"], m
    drv.execute_script("document.querySelectorAll('.builder-dialog [data-k=sites] li[data-i]')[1].click()")
    time.sleep(0.4)
    assert frame(drv)["name"] == "MgO_3x3x3_V_O_4b"
    set_n(drv, [2, 2, 2])
    assert frame(drv)["name"] == "MgO_2x2x2_V_O_4b"
    set_n(drv, [1, 1, 1])
    f = frame(drv)
    assert f["name"] == "MgO_V_O_4b", f
    H.wait(drv, "return /interact strongly/.test(window.__mace.builderFrame().readout)", 10)
    amber = drv.execute_script("return [...document.querySelectorAll('.builder-dialog .bld-readout .warn')].map((e) => getComputedStyle(e).color)")
    assert amber, "the short image distance is amber"
    U.shot(drv, "defect_dialog_1x1x1_amber")
    set_n(drv, [2, 2, 2])
    drv.execute_script("const c = document.querySelector('.builder-dialog [data-k=charge]'); c.value = '2'; c.dispatchEvent(new Event('input', {bubbles: true}))")
    time.sleep(0.3)
    drv.execute_script("document.querySelector('.builder-dialog .bld-go').click()")
    H.wait(drv, "return window.__mace.docs().length === 2 && window.__mace.doc().title === 'MgO_2x2x2_V_O_4b'", 20)
    time.sleep(0.5)
    assert U.doc(drv)["n"] == 63
    prov = drv.execute_script("return document.querySelector('#s-prov').textContent")
    assert "Vacancy at O (4b) in MgO.cif" in prov and "Defect charge for Setup: +2" in prov, prov
    assert drv.execute_script("""return Object.keys(localStorage).some((k) => { if (!k.startsWith('mace.gui.draft.')) return false;
        const d = JSON.parse(localStorage.getItem(k)); return d.name === 'MgO_2x2x2_V_O_4b' && d.meta && d.meta.hints && d.meta.hints.charge === 2; })""")
    assert U.tree_md5(root) == before
    U.no_errors(drv)


def test_defect_pick_from_viewport_selects_row(fresh):
    drv = fresh[0]
    U.open_doc(drv, "structures/MgO.cif")
    drv.execute_script("window.__mace.run('build.defect.substitution')")
    dialog_open(drv, "defect")
    defect_ready(drv)
    # a click on an O atom in the viewport selects the O row
    i = drv.execute_script("const s = window.__mace.doc().structure; return s.symbols.indexOf('O')")
    p = H.atom_screen(drv, i)
    H.click_at(drv, p["x"], p["y"])
    time.sleep(0.5)
    sel = drv.execute_script("return [...document.querySelectorAll('.builder-dialog [data-k=sites] li[aria-selected=true]')].map((l) => l.textContent.trim())")
    picked = drv.execute_script("return window.__mace.editor.selection()")
    if picked == [i]:
        assert sel and sel[0].startswith("O"), sel
    else:          # the click hit an atom in front of it: the row follows whatever was picked
        sym = drv.execute_script("return window.__mace.doc().structure.symbols[arguments[0]]", picked[0])
        assert sel and sel[0].startswith(sym), (sel, sym)
    U.keys(drv, Keys.ESCAPE)


def test_one_structure_per_site_opens_n_tabs(fresh):
    drv = fresh[0]
    U.open_doc(drv, "structures/MgO.cif")
    drv.execute_script("window.__mace.run('build.defect.vacancy')")
    dialog_open(drv, "defect")
    defect_ready(drv)
    set_n(drv, [1, 1, 1])
    drv.execute_script("const c = document.querySelector('.builder-dialog [data-k=per]'); c.checked = true; c.dispatchEvent(new Event('change', {bubbles: true}))")
    time.sleep(0.4)
    assert "2 structures" in frame(drv)["readout"]
    m = drv.execute_script("return window.__mace.viewport.models()")
    assert m["halo"] >= 2, m
    drv.execute_script("document.querySelector('.builder-dialog .bld-go').click()")
    H.wait(drv, "return window.__mace.docs().length === 3", 20)
    titles = [d["title"] for d in drv.execute_script("return window.__mace.docs()")]
    assert titles == ["MgO.cif", "MgO_V_Mg_4a", "MgO_V_O_4b"], titles
    U.no_errors(drv)


def test_atom_menu_defect_entries_preselect_site(fresh):
    drv = fresh[0]
    U.open_doc(drv, "structures/MgO.cif")
    drv.execute_script("window.__mace.editor.select([5], 'replace')")
    assert drv.execute_script("return window.__mace.menuReason('atom.vacancyHere')") == ""
    drv.execute_script("window.__mace.run('atom.antisite')")
    dialog_open(drv, "defect")
    defect_ready(drv)
    tab = drv.execute_script("return document.querySelector('.builder-dialog [role=tab][aria-selected=true]').dataset.tab")
    assert tab == "antisite"
    f = frame(drv)
    assert f["name"].endswith("_Mg_O_4b"), f
    U.keys(drv, Keys.ESCAPE)


# --- T7 transform cell, symmetrize -------------------------------------------------------

def write_distorted(root, name, delta):
    """MgO with one O moved by delta (fractional), written into the test's own copy of the folder."""
    import warnings
    import numpy as np
    import ase.io
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = ase.io.read(str(Path(root) / "structures" / "MgO.cif"))
        f = a.get_scaled_positions()
        f[4] += np.asarray(delta)
        a.set_scaled_positions(f)
        ase.io.write(str(Path(root) / "structures" / name), a, format="cif")


def test_transform_dialog_primitive_in_place(fresh):
    drv = fresh[0]
    U.open_doc(drv, "structures/MgO.cif")
    drv.execute_script("window.__mace.run('build.transform')")
    dialog_open(drv, "transform")
    assert frame(drv)["result"] == "inplace"
    H.wait(drv, "return /8 → 2 atoms · a 4\\.212 → 2\\.978 Å · Fm-3m kept/.test(window.__mace.builderFrame().readout)", 10)
    m = drv.execute_script("return window.__mace.viewport.models()")
    assert m["cell"] and m["ghost"] == 2 and m["labels"] == 3, m          # the new cell, its atoms, a′ b′ c′
    U.shot(drv, "transform_dialog_preview")
    drv.execute_script("document.querySelector('.builder-dialog .bld-go').click()")
    U.wait_idle(drv)
    st = U.ed_state(drv)
    assert st["n"] == 2 and st["rows"][-1] == "Transform to Primitive" and len(st["rows"]) == 2, st
    U.no_errors(drv)


def test_symmetrize_accept_one_entry_cancel_none(fresh):
    drv, root = fresh[0], fresh[2]
    write_distorted(root, "MgO_d.cif", (0.001, 0.0007, 0.0004))
    drv.execute_script("window.MACE.loadFiles()")
    time.sleep(0.5)
    U.open_doc(drv, "structures/MgO_d.cif")
    drv.execute_script("window.__mace.run('build.symmetrize')")
    dialog_open(drv, "symmetrize")
    H.wait(drv, "return document.querySelectorAll('.builder-dialog [data-k=sweep] li[data-i]').length === 13", 15)
    first = drv.execute_script("return document.querySelector('.builder-dialog [data-k=sweep] li[data-i]').textContent")
    assert "P1 (1)" in first, first
    bold = drv.execute_script("return document.querySelectorAll('.builder-dialog [data-k=sweep] li.change b').length")
    assert bold >= 1
    drv.execute_script("document.querySelector('.builder-dialog [data-k=sweep] li[data-i=\"9\"]').click()")        # 1e-2
    H.wait(drv, "return /Fm-3m \\(225\\)/.test(window.__mace.builderFrame().readout) && /max shift/.test(window.__mace.builderFrame().readout)", 10)
    m = drv.execute_script("return window.__mace.viewport.models()")
    assert m["ghost"] == 8 and m["arrows"] >= 1, m
    U.shot(drv, "symmetrize_dialog_sweep")
    rows = U.ed_state(drv)["rows"]
    U.keys(drv, Keys.ESCAPE)                                                    # Cancel: no entry
    time.sleep(0.4)
    assert U.ed_state(drv)["rows"] == rows
    drv.execute_script("window.__mace.run('build.symmetrize')")
    dialog_open(drv, "symmetrize")
    H.wait(drv, "return document.querySelectorAll('.builder-dialog [data-k=sweep] li[data-i]').length === 13", 15)
    drv.execute_script("document.querySelector('.builder-dialog [data-k=sweep] li[data-i=\"9\"]').click()")
    H.wait(drv, "return /max shift/.test(window.__mace.builderFrame().readout)", 10)
    drv.execute_script("document.querySelector('.builder-dialog [data-k=sweep]').closest('dialog').querySelector('[data-k=tol]').focus()")
    U.keys(drv, Keys.ENTER)                                                     # Accept: one entry
    U.wait_idle(drv)
    st = U.ed_state(drv)
    assert st["rows"][-1] == "Symmetrize at 0.01 Å" and len(st["rows"]) == len(rows) + 1, st["rows"]
    H.wait(drv, "return document.querySelector('#st-sg').textContent.startsWith('Fm-3m (225)')", 10)
    U.no_errors(drv)


# --- T8 Clean geometry (criterion 3) ------------------------------------------------------

def test_clean_accept_one_entry(fresh):
    drv, root = fresh[0], fresh[2]
    write_distorted(root, "MgO_c.cif", (0.047, 0.0, 0.0))                      # one O moved 0.2 Å
    drv.execute_script("window.MACE.loadFiles()")
    time.sleep(0.5)
    U.open_doc(drv, "structures/MgO_c.cif")
    drv.execute_script("document.querySelector('#viewer-main').closest('.viewer-wrap').focus()")
    U.keys(drv, Keys.CONTROL, Keys.ALT, "g")          # Ctrl+Shift+G is Firefox's (plan T16)
    dialog_open(drv, "clean")
    H.wait(drv, "const f = window.__mace.builderFrame(); return f && !f.goDisabled", 60)
    ro = frame(drv)["readout"]
    import re
    assert re.search(r"max shift \d+\.\d{3} Å · RMS \d+\.\d{3} Å · \d+ bonds? changed · group \S+ → \S+", ro), ro
    assert "Spring model (rough pre-relaxation, not physics)" in drv.execute_script("return document.querySelector('.builder-dialog').textContent")
    m = drv.execute_script("return window.__mace.viewport.models()")
    assert m["dim"] and m["ghost"] == 8 and m["arrows"] >= 1, m
    op = drv.execute_script("return Number(getComputedStyle(document.documentElement).getPropertyValue('--ghost-opacity'))")
    assert abs(op - 0.3) < 1e-9
    U.shot(drv, "clean_panel_preview")
    rows = U.ed_state(drv)["rows"]
    drv.execute_script("document.querySelector('.builder-dialog .bld-go').click()")
    H.wait(drv, f"const e = window.__mace.editor.doc(); return e.rows.length === {len(rows) + 1} && e.pending === 0", 30)
    st = U.ed_state(drv)
    assert st["rows"][-1] == "Clean Geometry", st["rows"]
    assert not drv.execute_script("return window.__mace.viewport.models().dim")
    U.no_errors(drv)


def test_clean_discard_no_entry(fresh):
    drv, root = fresh[0], fresh[2]
    write_distorted(root, "MgO_c2.cif", (0.047, 0.0, 0.0))
    drv.execute_script("window.MACE.loadFiles()")
    time.sleep(0.5)
    U.open_doc(drv, "structures/MgO_c2.cif")
    before = U.ed_state(drv)
    drv.execute_script("window.__mace.run('build.clean')")
    dialog_open(drv, "clean")
    H.wait(drv, "const f = window.__mace.builderFrame(); return f && !f.goDisabled", 60)
    U.keys(drv, Keys.ESCAPE)
    time.sleep(0.5)
    after = U.ed_state(drv)
    assert after["rows"] == before["rows"] and after["frac"] == before["frac"], (before["rows"], after["rows"])
    assert not drv.execute_script("return window.__mace.viewport.models().dim")
    U.no_errors(drv)


def test_classic_clean_button_opens_panel(fresh):
    drv = fresh[0]
    U.open_doc(drv, "structures/MgO.cif")
    drv.execute_script("window.__mace.run('build.editStructure')")
    H.wait(drv, "return !!document.querySelector('#inspector #e-clean')", 10)
    assert drv.execute_script("return document.querySelector('#e-clean').textContent.trim()").startswith("Clean")
    drv.execute_script("document.querySelector('#e-clean').click()")
    dialog_open(drv, "clean")
    U.keys(drv, Keys.ESCAPE)


def test_clean_on_metal_shows_message(fresh):
    drv = fresh[0]
    drv.execute_async_script("""const done = arguments[0];
        window.MACE.api('edit/from_structure', {body: {name: 'Cu', cell: [[3.61,0,0],[0,3.61,0],[0,0,3.61]], numbers: [29,29,29,29],
          frac: [[0,0,0],[0,.5,.5],[.5,0,.5],[.5,.5,0]]}}).then((r) => window.__mace.openUntitled(r, 'Cu')).then(() => done(1))""")
    H.wait(drv, "return window.__mace.viewport.displayed() > 0", 10)
    drv.execute_script("window.__mace.run('build.clean')")
    dialog_open(drv, "clean")
    H.wait(drv, "return /no bonded pairs/.test(window.__mace.builderFrame().error)", 20)
    f = frame(drv)
    assert f["goDisabled"] and f["error"].startswith("Nothing to clean: no bonded pairs"), f
    assert drv.execute_script("return window.__mace.viewport.models().ghost") == 0
    U.shot(drv, "clean_on_metal")
    U.keys(drv, Keys.ESCAPE)


def test_supercell_dialog_on_a_slab_repeats_a_and_b(fresh):
    """A SLAB output: n3 is 1 and fixed, the 2×2 build is made (a held view of a
    slab), and a matrix that touches c says why it cannot be built."""
    drv = fresh[0]
    U.open_doc(drv, "outputs/graphene_slab_sp.out")
    n0 = U.doc(drv)["n"]
    assert n0 > 0
    drv.execute_script("window.__mace.run('build.supercell')")
    dialog_open(drv, "supercell")
    vals = drv.execute_script("return [...document.querySelectorAll('.builder-dialog [data-n]')].map((i) => [i.value, i.disabled])")
    assert vals == [["2", False], ["2", False], ["1", True]], vals
    H.wait(drv, f"const f = window.__mace.builderFrame(); return f && f.readout.includes('{4 * n0} atoms')", 10)
    assert frame(drv)["name"].endswith("_2x2x1")
    drv.execute_script("document.querySelector('.builder-dialog [data-tab=matrix]').click()")
    drv.execute_script("""const i = document.querySelector('.builder-dialog [data-m="02"]'); i.value = '1';
        i.dispatchEvent(new Event('input', {bubbles: true}))""")
    time.sleep(0.4)
    f = frame(drv)
    assert "periodic along a and b only" in f["error"] and f["goDisabled"], (f, drv.execute_script("return [...document.querySelectorAll('.builder-dialog [data-m]')].map((i) => i.value)"))
    U.no_errors(drv)
