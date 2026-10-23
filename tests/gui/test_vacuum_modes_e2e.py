"""Slabs, polymers and molecules in the browser (1.1.6): the five Vacuum modes of
Inspector › Display (Bounding box, Open, File-defined, CRYSTAL 500 Å, Custom), per
document, drawing the box and spacing a custom display range along the
non-periodic axis; the editor (Move, Add, Delete) on a slab in P1, with Sym mode
refused; and Build › Supercell repeating c at the vacuum spacing. Firefox."""
import time

import pytest

import guiharness as H

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")

SLAB = "outputs/graphene_slab_sp.out"
VM = "window.MACE.state.viewers['viewer-main']"


@pytest.fixture(scope="module")
def drv(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("vacmodes")
    root = H.examples_copy(tmp)
    proc, base, _ = H.start_server(root, tmp / "xdg", MACE_GUI_TEST_HOOKS="1", MACE_GUI_MAX_WORKERS="2")
    try:
        d = H.browser("firefox", width=1280, height=800)
    except BaseException:
        proc.terminate()
        proc.wait(timeout=15)
        raise
    try:
        H.load(d, base)
        d.execute_script("const s = window.__mace.settings; s.set('reduceMotion', 'on'); s.set('showBoundary', false)")
        yield d
    finally:
        d.quit()
        proc.terminate()
        proc.wait(timeout=15)


def run(drv, cmd):
    drv.execute_script("window.__mace.run(arguments[0])", cmd)
    time.sleep(0.6)


def custom(drv, hi):
    run(drv, "view.rangeCustom")
    H.wait(drv, "const d = document.getElementById('range-custom'); return !!(d && d.open)", 5)
    drv.execute_script("""const d = document.getElementById('range-custom');
      arguments[0].forEach((x, k) => { d.querySelector(`[data-hi="${k}"]`).value = String(x); });
      d.querySelector('[data-act="apply"]').click();""", hi)
    H.wait(drv, "return !document.getElementById('range-custom').open", 5)
    time.sleep(1.0)


def zs(drv):
    return sorted(drv.execute_script(f"return {VM}.getModel().selectedAtoms({{}}).map((a) => a.z)"))


def pick_mode(drv, mode, gap=None):
    drv.execute_script("""const s = document.getElementById('opt-vacuum-mode'); s.value = arguments[0];
      s.dispatchEvent(new Event('change'));
      if (arguments[1] !== null) { const g = document.getElementById('opt-vacuum-gap'); g.value = String(arguments[1]);
        g.dispatchEvent(new Event('change')); }""", mode, gap)
    time.sleep(0.8)


def open_slab(drv):
    kind = drv.execute_async_script("const d = arguments[1]; window.__mace.open(arguments[0]).then((x) => d(x && x.kind))", SLAB)
    assert kind == "structure"
    H.wait(drv, f"const v = {VM}; return !!(v && v.getModel())", 30)
    time.sleep(0.8)


def slab_open(drv):
    """The slab, opened here when this test runs without the ones before it."""
    if drv.execute_script("const d = window.__mace.doc(); return !!d && d.path === arguments[0]", SLAB):
        return
    open_slab(drv)
    H.wait(drv, "return !!(window.__mace.editor && window.__mace.editor.doc())", 30)


def test_five_modes_space_a_custom_range_along_c(drv):
    open_slab(drv)
    assert drv.execute_script("return document.getElementById('opt-vacuum-mode').disabled") is False
    assert drv.execute_script("return document.getElementById('opt-vacuum-mode').value") == "bbox"
    n0 = len(zs(drv))
    custom(drv, [1, 1, 3])
    # graphene is flat: Bounding box = 1 Å each side, so the layers are 2 Å apart
    z = zs(drv)
    assert len(z) == 3 * n0 and sorted(set(round(x - z[0], 3) for x in z)) == [0.0, 2.0, 4.0], z
    pick_mode(drv, "gap", 12)
    z = zs(drv)
    assert sorted(set(round(x - z[0], 3) for x in z)) == [0.0, 12.0, 24.0], z
    assert "12 Å custom" in drv.execute_script("return document.getElementById('tb-range').dataset.tip")
    assert drv.execute_script("return document.getElementById('opt-vacuum-gap').hidden") is False
    pick_mode(drv, "crystal")
    z = zs(drv)
    assert sorted(set(round(x - z[0], 3) for x in z)) == [0.0, 500.0, 1000.0], z
    pick_mode(drv, "cell")                    # the output's formal 500 Å c
    assert sorted(set(round(x - z[0], 3) for x in zs(drv))) == [0.0, 500.0, 1000.0]
    pick_mode(drv, "open")
    assert len(zs(drv)) == n0                 # nothing repeats along an open direction
    assert "Open vacuum" in drv.execute_script("return document.getElementById('tb-range').dataset.tip")
    # per document: the Settings default is untouched
    assert drv.execute_script("return window.__mace.settings.get('viewport.vacuumMode')") == "bbox"
    pick_mode(drv, "bbox")
    run(drv, "view.range1")
    assert not H.errors(drv), H.errors(drv)


def test_slab_edits_in_p1(drv):
    slab_open(drv)
    reason = "return window.__mace.menuReason(arguments[0])"
    assert drv.execute_script("return window.__mace.ctx().docEditable") is True
    assert "needs a 3D crystal" in drv.execute_script(reason, "build.symModeSym")
    z0 = drv.execute_script("return window.__mace.doc().structure.cart[0][2]")
    drv.execute_script("const E = window.__mace.editor.E; E.api.edit(window.__mace.editor.doc(), 'move', "
                       "{indices: [0], delta: [0, 0, 3], wrap: true})")
    H.wait(drv, "const e = window.__mace.editor.doc(); return e && e.pending === 0 && !e.waiting && !!e.srv.session", 20)
    s = drv.execute_script("const s = window.__mace.editor.doc().structure; return {z: s.cart[0][2], pbc: s.pbc}")
    assert s["pbc"] == [True, True, False] and abs(s["z"] - (z0 + 3)) < 1e-6, s     # c never wraps
    drv.execute_script("const E = window.__mace.editor.E; E.api.edit(window.__mace.editor.doc(), 'add', "
                       "{element: 'H', cart: [0, 0, 6]})")
    H.wait(drv, "const e = window.__mace.editor.doc(); return e.pending === 0 && !e.waiting && e.n === 3", 20)
    assert drv.execute_script("return window.__mace.editor.doc().structure.dimensionality") == "SLAB"
    # Clean Geometry on the slab: springs, no space group, the pbc kept
    r = drv.execute_async_script("""const done = arguments[0], e = window.__mace.editor.doc();
      window.MACE.api('clean', {body: {session: e.srv.session, base_rev: e.srv.rev, keep_symmetry: true, relax_cell: false,
        time_limit: 2}, withStatus: true}).then((x) => done({status: x.status, data: x.data}), (err) => done({err: String(err.message)}))""")
    assert "err" not in r, r
    if r["status"] == 202:                    # a pool task: its result when it is done
        tid = r["data"]["task"]["id"]
        for _ in range(60):
            res = drv.execute_async_script("""const done = arguments[1];
              window.MACE.api('tasks/' + arguments[0] + '/result', {withStatus: true}).then((x) => done({status: x.status, data: x.data}),
                (err) => done({status: err.status || 0, err: String(err.message)}))""", tid)
            if res["status"] == 200:
                r = res
                break
            time.sleep(0.5)
    assert r["status"] == 200 and (r["data"].get("proposal_id") or r["data"].get("nothing_to_clean")), r
    if r["data"].get("proposal_id"):
        assert r["data"]["report"]["converged"] in (True, False)
    assert not H.errors(drv), H.errors(drv)


def test_supercell_repeats_c_at_the_vacuum_spacing(drv):
    slab_open(drv)
    pick_mode(drv, "gap", 10)
    drv.execute_script("window.__mace.run('build.supercell')")
    H.wait(drv, "return !!document.querySelector('[data-k=\"link\"]')", 10)
    drv.execute_script("""const box = document.querySelector('[data-k="link"]').closest('[data-pane="diag"]');
      const set = (k, v) => { const i = box.querySelector(`[data-n="${k}"]`); i.value = String(v);
        i.dispatchEvent(new Event('input', { bubbles: true })); i.dispatchEvent(new Event('change', { bubbles: true })); };
      document.querySelector('[data-k="link"]').checked = false; document.querySelector('[data-k="link"]').dispatchEvent(new Event('change'));
      set(0, 1); set(1, 1); set(2, 2);""")
    time.sleep(1.5)
    txt = drv.execute_script("const d = document.querySelector('[data-k=\"link\"]').closest('dialog, .panel, section, div.bld'); return (d || document.body).textContent")
    assert "repeated 2× at the 10 Å custom vacuum spacing" in txt, txt[-1500:]
    n = drv.execute_script("return window.__mace.doc().structure.numbers.length")
    drv.execute_script("""const dlg = document.querySelector('[data-k="link"]').closest('dialog');
      [...dlg.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Create').click();""")
    H.wait(drv, f"const d = window.__mace.doc(); return !!(d && d.structure && d.structure.numbers.length === {2 * n})", 30)
    s = drv.execute_script("const s = window.__mace.doc().structure; return {pbc: s.pbc, z: s.cart.map((c) => c[2])}")
    assert s["pbc"] == [True, True, False]
    z = sorted(s["z"])
    h = z[n - 1] - z[0]
    assert abs((z[-1] - z[0]) - (2 * h + 10)) < 1e-4, (z, h)            # two copies, 10 Å of vacuum between
    assert not H.errors(drv), H.errors(drv)
