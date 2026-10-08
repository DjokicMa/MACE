"""Layout checks from the Phase 4 visual review, in headless Firefox: the
explorer's names keep their ends next to the file details, the builder
dialogs keep Cancel and the primary button on one footer row, a builder's
preview stays in sight (and the view comes back on Cancel), the Tasks rows
share their columns, and the run and slab-preview views fill their panes."""
import time

import pytest

import guiharness as H
import p4util as U

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")
from selenium.webdriver.common.keys import Keys  # noqa: E402


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    a = U.start_app(tmp_path_factory.mktemp("p4layout"))
    try:
        yield a
    finally:
        U.stop_app(a)


@pytest.fixture
def fresh(app):
    U.reset(app[0])
    return app


def dialog_open(drv, id_):
    H.wait(drv, f"return !!document.querySelector('dialog.builder-dialog[data-builder=\"{id_}\"][open]')", 10)
    time.sleep(0.2)


@pytest.mark.parametrize("size", [(1280, 800), (1920, 1080)])
def test_explorer_names_keep_their_end_beside_details(fresh, size):
    """A long name loses its middle, never its end, when the formula and atom
    count sit beside it: the last part is drawn whole, inside the row."""
    drv = fresh[0]
    U.reset(drv, *size)
    H.wait(drv, "return document.querySelectorAll('#file-list .file .file-det').length > 0", 30)
    bad = drv.execute_script("""return [...document.querySelectorAll('#file-list .file')].map((b) => {
        // a short name is all head (no tail): it is never cut; a long one keeps its tail whole
        const head = b.querySelector('.mid-a'), tail = b.querySelector('.mid-b'), last = tail.textContent ? tail : head;
        const r = b.getBoundingClientRect(), t = last.getBoundingClientRect();
        const det = b.querySelector('.file-det'), d = det ? det.getBoundingClientRect() : null;
        return {name: b.dataset.path, tail: tail.textContent, ok: t.width > 0 && last.scrollWidth <= Math.ceil(t.width) + 1
          && t.right <= r.right + 0.5 && (!d || d.width === 0 || t.right <= d.left + 0.5)};
      }).filter((x) => !x.ok)""")
    assert not bad, bad
    # the details show whole or not at all (no "Ag2Br3 · 3…", no lone "A")
    cut = drv.execute_script("""return [...document.querySelectorAll('#file-list .file-det')]
        .filter((d) => getComputedStyle(d).visibility !== 'hidden' && d.scrollWidth > d.clientWidth + 1).map((d) => d.textContent)""")
    assert not cut, cut


@pytest.mark.parametrize("builder", ["supercell", "slab", "defect", "transform", "symmetrize"])
def test_builder_footer_one_row_cancel_then_primary(fresh, builder):
    drv = fresh[0]
    U.open_doc(drv, "structures/MgO.cif")
    drv.execute_script(f"window.__mace.run('build.{'defect.vacancy' if builder == 'defect' else builder}')")
    dialog_open(drv, builder)
    geo = drv.execute_script("""const d = document.querySelector('dialog.builder-dialog[open]');
        const c = d.querySelector('[data-act=cancel]').getBoundingClientRect(), g = d.querySelector('.bld-go').getBoundingClientRect();
        const foot = d.querySelector('.dlg-foot').getBoundingClientRect(), dr = d.getBoundingClientRect();
        return {sameRow: Math.abs(c.top - g.top) < 1, order: c.right <= g.left, right: dr.right - g.right,
                footH: foot.height, resultInBody: !!d.querySelector('.dlg-body .bld-result') || !d.querySelector('.bld-result'),
                nameInBody: !!d.querySelector('.dlg-body .bld-name')}""")
    assert geo["sameRow"] and geo["order"] and geo["right"] < 40 and geo["footH"] < 64, geo
    assert geo["resultInBody"] and geo["nameInBody"], geo
    U.keys(drv, Keys.ESCAPE)
    U.no_errors(drv)


def test_supercell_preview_in_sight_and_view_restored(fresh):
    """The 2×2×2 preview reaches past the default view of MgO: the view zooms
    out so every ghost atom and the new cell lie in the part of the viewport
    the dialog leaves free; Cancel puts the view back."""
    drv = fresh[0]
    U.open_doc(drv, "structures/MgO.cif")
    v0 = drv.execute_script("return window.MACE.viewer('viewer-main').getView()")
    drv.execute_script("window.__mace.run('build.supercell')")
    dialog_open(drv, "supercell")
    H.wait(drv, "const m = window.__mace.viewport.models(); return m && m.ghost === 64 && m.cell", 15)
    time.sleep(0.4)
    out = drv.execute_script("""const v = window.MACE.viewer('viewer-main'), cv = v.getCanvas(), r = cv.getBoundingClientRect();
        const dlg = document.querySelector('dialog.builder-dialog[open]').getBoundingClientRect();
        const right = Math.min(r.right, dlg.left);
        const gid = window.__mace.viewport.modelIds().ghost, atoms = v.getModel(gid).selectedAtoms({});
        const pts = atoms.map((a) => v.modelToScreen({x: a.x, y: a.y, z: a.z}));
        return {bad: pts.filter((p) => p.x < r.left || p.x > right || p.y < r.top || p.y > r.bottom).length, n: pts.length}""")
    assert out["n"] == 64 and out["bad"] == 0, out
    U.keys(drv, Keys.ESCAPE)
    time.sleep(0.4)
    v1 = drv.execute_script("return window.MACE.viewer('viewer-main').getView()")
    assert all(abs(a - b) < 1e-6 for a, b in zip(v0, v1)), (v0, v1)
    U.no_errors(drv)


def test_tasks_rows_share_columns(fresh):
    drv, base = fresh[0], fresh[1]
    U.open_doc(drv, "structures/MgO.cif")
    drv.execute_script("window.__mace.run('view.tasks')")
    U.post(base, "api/build/identity", {"source": {"path": "structures/MgO.cif"}, "params": {"delay_ms": 50, "fake_n": 9000},
                                         "open_as": {"name": "quick"}})
    U.post(base, "api/build/identity", {"source": {"path": "structures/MgO.cif"}, "params": {"delay_ms": 4000, "fake_n": 9000},
                                         "open_as": {"name": "slow"}})
    H.wait(drv, "return document.querySelectorAll('.task-row').length >= 2 && !!document.querySelector('.task-row.s-running')", 15)
    xs = drv.execute_script("return [...document.querySelectorAll('.task-row .task-bar')].map((b) => [Math.round(b.getBoundingClientRect().left), Math.round(b.getBoundingClientRect().width)])")
    assert len(xs) >= 2 and len({x for x, _ in xs}) == 1 and len({w for _, w in xs}) == 1, xs
    U.no_errors(drv)


def test_run_view_fills_its_pane(fresh):
    """The run's overlay view: the cell's standard view, its drawn atoms and
    cell filling most of the limiting side of the canvas, all inside it."""
    drv = fresh[0]
    U.open_doc(drv, "outputs/Ag2Br3_opt_unfinished.out", wait_atoms=False)
    H.wait(drv, "return !!(window.MACE.state.traj && window.MACE.state.viewers['viewer-before'])")
    time.sleep(1.0)
    out = drv.execute_script("""const v = window.MACE.state.viewers['viewer-before'], cv = v.getCanvas(), r = cv.getBoundingClientRect();
        const atoms = v.getModel(0).selectedAtoms({});
        let x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9, bad = 0;
        for (const a of atoms) { const p = v.modelToScreen({x: a.x, y: a.y, z: a.z});
          if (p.x < r.left || p.x > r.right || p.y < r.top || p.y > r.bottom) bad++;
          x0 = Math.min(x0, p.x); x1 = Math.max(x1, p.x); y0 = Math.min(y0, p.y); y1 = Math.max(y1, p.y); }
        return {bad, fill: Math.max((x1 - x0) / r.width, (y1 - y0) / r.height), n: atoms.length}""")
    assert out["n"] > 0 and out["bad"] == 0 and out["fill"] > 0.45, out
    U.no_errors(drv)


def test_slab_preview_fills_the_free_part_of_the_viewport(fresh):
    drv = fresh[0]
    U.open_doc(drv, "structures/MgO.cif")
    drv.execute_script("window.__mace.run('build.slab')")
    dialog_open(drv, "slab")
    H.wait(drv, "return window.__maceSlabPane && window.__maceSlabPane.n === 20", 15)
    time.sleep(0.4)
    out = drv.execute_script("""const v = window.MACE.viewer('viewer-slab-preview'), cv = v.getCanvas(), r = cv.getBoundingClientRect();
        const dlg = document.querySelector('dialog.builder-dialog[open]').getBoundingClientRect();
        const right = Math.min(r.right, dlg.left);
        const atoms = v.getModel(0).selectedAtoms({});
        let y0 = 1e9, y1 = -1e9, bad = 0;
        for (const a of atoms) { const p = v.modelToScreen({x: a.x, y: a.y, z: a.z});
          if (p.x < r.left || p.x > right || p.y < r.top || p.y > r.bottom) bad++;
          y0 = Math.min(y0, p.y); y1 = Math.max(y1, p.y); }
        return {bad, h: (y1 - y0) / r.height, n: atoms.length}""")
    # 5 planes over 8.4 Å of a 23.4 Å cell drawn at 85% of the height: about a third of it
    assert out["n"] >= 20 and out["bad"] == 0 and out["h"] > 0.2, out
    U.keys(drv, Keys.ESCAPE)
    U.no_errors(drv)


def test_symmetrize_warns_when_a_supercell_would_shrink(fresh):
    """Symmetrize writes spglib's conventional cell: on MgO 2×2×2 that is the
    8-atom cell, and the readout says so before Accept (as Save As refuses a
    symmetrized CIF of a supercell)."""
    drv = fresh[0]
    U.open_doc(drv, "structures/MgO.cif")
    drv.execute_script("window.__mace.run('build.supercell')")
    dialog_open(drv, "supercell")
    drv.execute_script("document.querySelector('.builder-dialog .bld-go').click()")
    H.wait(drv, "return window.__mace.doc() && window.__mace.doc().title === 'MgO_2x2x2'", 20)
    drv.execute_script("window.__mace.run('build.symmetrize')")
    dialog_open(drv, "symmetrize")
    H.wait(drv, "const r = document.querySelector('.builder-dialog .bld-readout'); return r && /from 64/.test(r.textContent)", 20)
    ro = drv.execute_script("return document.querySelector('.builder-dialog .bld-readout').textContent")
    assert "8 atoms (from 64)" in ro and "a supercell loses its repeats" in ro, ro
    U.keys(drv, Keys.ESCAPE)
    U.no_errors(drv)


def test_payload_defers_symmetry_above_20000_atoms():
    """A file over 20,000 atoms opens without spglib (an 85k-atom CIF took
    97 s and 1.6 GB in the request thread); the inspector offers Compute Now."""
    import numpy as np
    from mace.gui import structures as S
    n = S.SYMMETRY_INLINE_MAX + 1
    t0 = time.perf_counter()
    p = S._payload(np.eye(3) * 60.0, np.random.default_rng(1).random((n, 3)), np.zeros((n, 3)), [12] * n, (True, True, True),
                   {"kind": "cif", "path": "big.cif"})
    assert p["symmetry"]["not_computed"] and not p["symmetry"]["available"] and "20,000" in p["symmetry"]["note"]
    assert time.perf_counter() - t0 < 10


def test_compute_now_on_a_big_built_structure(fresh):
    drv, base = fresh[0], fresh[1]
    U.open_doc(drv, "structures/MgO.cif")
    drv.execute_script("window.__mace.run('build.supercell')")
    dialog_open(drv, "supercell")
    # 14×14×14 (21,952 atoms; the diagonal tab stops at 10): the matrix tab
    drv.execute_script("""document.querySelector('.builder-dialog [data-tab=matrix]').click();
        for (const k of ['00', '11', '22']) { const i = document.querySelector(`.builder-dialog [data-m="${k}"]`); i.value = '14';
          i.dispatchEvent(new Event('input', {bubbles: true})); }""")
    time.sleep(0.6)
    drv.execute_script("document.querySelector('.builder-dialog .bld-go').click()")
    H.wait(drv, "return window.__mace.docs().length === 2 && window.__mace.doc().structure && window.__mace.doc().structure.numbers.length === 21952", 120)
    # the Cell card is the built cell's (14 × 4.212 Å), not its parent's
    H.wait(drv, "return /58\\.968/.test(document.querySelector('#s-lattice').textContent)", 20)
    H.wait(drv, "return [...document.querySelectorAll('.sym-compute')].some((b) => b.offsetParent)", 20)
    drv.execute_script("[...document.querySelectorAll('.sym-compute')].find((b) => b.offsetParent).click()")
    H.wait(drv, "const d = window.__mace.doc(); return d && d.structure && d.structure.symmetry && d.structure.symmetry.number === 225", 180)
    U.no_errors(drv)
