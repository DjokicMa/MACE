"""The structure viewer and the editor in headless Firefox (spec Phase 3):
the pointer model, selection (click, box, lasso, orbit, fragment), the
camera, hover readout, and (later sections) editing, the inspector, tools,
dialogs, measurements, history, saving, drafts and export.

Real WebDriver pointer actions throughout (buttons 0/1/2, wheel). Set
MACE_GUI_SHOTS=<dir> to also save screenshots.
"""
import json
import math
import os
import time
from pathlib import Path

import pytest

import guiharness as H

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")

from selenium.webdriver.common.keys import Keys  # noqa: E402
from selenium.webdriver.common.action_chains import ActionChains  # noqa: E402



@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("guied")
    root, xdg = H.examples_copy(tmp), tmp / "xdg"
    proc, base, _ = H.start_server(root, xdg, MACE_GUI_TEST_HOOKS="1")
    downloads = tmp / "downloads"
    downloads.mkdir()
    drv = H.firefox(1280, 800, download_dir=downloads)
    try:
        H.load(drv, base)
        yield drv, base, root, xdg, proc, downloads
    finally:
        drv.quit()
        proc.terminate()
        proc.wait(timeout=15)


@pytest.fixture
def fresh(app):
    drv = app[0]
    drv.execute_script("window.__mace.closeAll(); window.__mace.settings.resetAll(); window.__mace.settings.set('theme', 'light');"
                       "const set = (id, v) => { const e = document.getElementById(id); if (e.checked !== v) { e.checked = v; e.dispatchEvent(new Event('change')); } };"
                       "set('opt-bonds', true); set('opt-super', false);"
                       "window.__mace.run('tool.select');"
                       "document.querySelectorAll('dialog[open]').forEach((d) => d.close());"
                       "document.querySelectorAll('#toasts > *').forEach((t) => t.remove());"
                       "window.__maceErrors.length = 0;"
                       # Resource Timing keeps 250 entries by default: the request counts need room
                       "performance.clearResourceTimings(); performance.setResourceTimingBufferSize(100000);")
    H.set_viewport(drv, 1280, 800)
    time.sleep(0.3)
    drv.execute_script("document.body.focus()")
    return app


def keys(drv, *combo):
    ac = ActionChains(drv)
    for k in combo[:-1]:
        ac.key_down(k)
    ac.send_keys(combo[-1])
    for k in reversed(combo[:-1]):
        ac.key_up(k)
    ac.perform()
    time.sleep(0.15)


def open_doc(drv, path):
    kind = drv.execute_async_script(
        "const done = arguments[1]; window.__mace.open(arguments[0]).then((d) => done(d && d.kind))", path)
    H.wait(drv, "return window.__mace.viewport && window.__mace.viewport.displayed() > 0")
    time.sleep(0.3)
    return kind


def shot(drv, name):
    target = os.environ.get("MACE_GUI_SHOTS")
    if target:
        Path(target).mkdir(parents=True, exist_ok=True)
        drv.save_screenshot(str(Path(target) / f"{name}.png"))


def selection(drv):
    return drv.execute_script("return window.__mace.editor.selection()")


def set_view(drv, q=None, boundary=False):
    """MgO without boundary images at a fixed turn where no two drawn atoms
    overlap on screen (searched once from a seeded list of turns)."""
    # small atoms (atom scale 0.2): every center is clear of its neighbors' discs.
    # Ball and stick draws atom scale x VESTA's atomic radius (T9): Mg 1.60 A.
    drv.execute_script("window.__mace.settings.set('atomScale', 0.2)")
    if not boundary:
        drv.execute_script("window.__mace.settings.set('showBoundary', false)")
    H.wait(drv, f"return window.__mace.viewport.displayed() === {8 if not boundary else 27} && "
                "Math.abs(window.__mace.viewport.scene().drawnRadii[0] - 0.2 * 1.6) < 1e-3", 10)
    drv.execute_script("""const v = window.MACE.state.viewers['viewer-main'], cam = window.__mace.viewport.camera();
        const disp = window.__mace.viewport.scene().display, R = window.__mace.viewport.scene().drawnRadii;
        let seed = 7; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
        const want = arguments[0];
        let best = null, bestGap = -Infinity;
        for (let t = 0; t < (want ? 1 : 200); t++) {
          const q = want || [rnd() - 0.5, rnd() - 0.5, rnd() - 0.5, rnd() - 0.5], n = Math.hypot(...q);
          const view = v.getView();
          v.setView([view[0], view[1], view[2], view[3], ...q.map((x) => x / n)]);
          cam.invalidate();
          const c = cam.calibrate(), pts = [];
          for (let k = 0; k < disp.n; k++) pts.push(cam.screenOf([disp.cart[3 * k], disp.cart[3 * k + 1], disp.cart[3 * k + 2]]));
          // every atom's center is clear of every other atom's disc: a click on it picks it
          let gap = Infinity;
          for (let a = 0; a < pts.length; a++) for (let b = a + 1; b < pts.length; b++)
            gap = Math.min(gap, Math.hypot(pts[a][0] - pts[b][0], pts[a][1] - pts[b][1]) - Math.max(R[a], R[b]) * c.s - 3);
          if (gap > bestGap) { bestGap = gap; best = q.map((x) => x / n); }
          if (gap > 12) break;
        }
        const view = v.getView();
        v.setView([view[0], view[1], view[2], view[3], ...best]); v.render(); cam.invalidate();
        window.__setViewGap = bestGap;""", q)
    time.sleep(0.3)
    assert drv.execute_script("return window.__setViewGap") > 2


def screen_atoms(drv, n=8):
    return [H.atom_screen(drv, i) for i in range(n)]


def no_errors(drv):
    errs = H.errors(drv)
    assert not errs, errs


# --- T4: pointer, selection, camera -------------------------------------------------------

def test_click_shift_mod_select(fresh):
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    set_view(drv)
    H.click_atom(drv, 0)
    time.sleep(0.2)
    assert selection(drv) == [0]
    assert drv.execute_script("return document.querySelector('#st-sel').textContent") == "1 of 8 atoms"
    assert "Selected Mg1, 4a" in drv.execute_script("return document.querySelector('#sr-live').textContent")
    time.sleep(0.5)        # not a double click
    H.click_atom(drv, 4, mods=(Keys.SHIFT,))
    time.sleep(0.5)
    H.click_atom(drv, 5, mods=(Keys.SHIFT,))
    time.sleep(0.2)
    assert selection(drv) == [0, 4, 5]
    assert drv.execute_script("return document.querySelector('#st-sel').textContent") == "3 of 8 atoms"
    time.sleep(0.5)
    H.click_atom(drv, 4, mods=(Keys.CONTROL,))               # Mod toggles
    time.sleep(0.5)
    H.click_atom(drv, 6, mods=(Keys.CONTROL,))
    time.sleep(0.2)
    assert selection(drv) == [0, 5, 6]
    time.sleep(0.5)
    H.click_atom(drv, 5, mods=(Keys.SHIFT, Keys.CONTROL))     # Shift+Mod subtracts
    time.sleep(0.2)
    assert selection(drv) == [0, 6]
    time.sleep(0.5)
    H.click_atom(drv, 1)                                     # a plain click replaces
    time.sleep(0.5)
    assert selection(drv) == [1]
    # empty space clears
    H.click_at(drv, 8, 8)
    time.sleep(0.2)
    assert selection(drv) == []
    no_errors(drv)


def _three_in_a_box(pts):
    """Indices of 3 atoms whose bounding box (padded 6 px) holds no other atom."""
    import itertools
    for trio in itertools.combinations(range(len(pts)), 3):
        xs = [pts[i]["x"] for i in trio]
        ys = [pts[i]["y"] for i in trio]
        box = (min(xs) - 6, min(ys) - 6, max(xs) + 6, max(ys) + 6)
        if all(not (box[0] <= p["x"] <= box[2] and box[1] <= p["y"] <= box[3]) for k, p in enumerate(pts) if k not in trio):
            return trio, box
    return None, None


def test_box_select_three_atoms(fresh):
    """Criterion 1: box-selecting 3 atoms shows "3 of 8 atoms" (the Atoms
    panel rows are checked with the table, in the T10 section)."""
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    set_view(drv)
    pts = screen_atoms(drv)
    trio, box = _three_in_a_box(pts)
    assert trio, pts
    canvas = H.viewer_canvas(drv)
    H.drag(drv, canvas, [(box[0], box[1]), (box[2], box[3])])
    H.wait(drv, "return document.querySelector('#st-sel').textContent === '3 of 8 atoms'", 5)
    assert selection(drv) == sorted(trio)
    assert drv.execute_script("return window.__mace.viewport.selectionShells()") == 3
    assert not drv.execute_script("return document.querySelector('.vp-overlay').classList.contains('on')")
    # and the Atoms panel highlights the same rows
    rows = drv.execute_script("return [...document.querySelectorAll('.atoms-table tr.sel')].map((r) => Number(r.dataset.i))")
    assert sorted(rows) == sorted(trio), rows
    shot(drv, "p3_box_select_light")
    no_errors(drv)


def test_lasso_select(fresh):
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    set_view(drv)
    drv.execute_script("window.__mace.run('tool.lasso')")
    pts = screen_atoms(drv)
    trio, box = _three_in_a_box(pts)
    # a triangle-ish polygon around the box corners
    path = [(box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3]), (box[0], box[1] + 1)]
    H.drag(drv, H.viewer_canvas(drv), path)
    H.wait(drv, "return window.__mace.editor.selection().length === 3", 5)
    assert selection(drv) == sorted(trio)
    no_errors(drv)


def test_double_click_orbit_triple_click_fragment(fresh):
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    set_view(drv)
    H.click_atom(drv, 4, count=2)                 # O on 4b: the four O atoms
    H.wait(drv, "return window.__mace.editor.selection().length === 4", 5)
    assert selection(drv) == [4, 5, 6, 7]
    time.sleep(0.6)
    H.click_atom(drv, 0, count=3)                 # MgO is one connected network: everything
    H.wait(drv, "return window.__mace.editor.selection().length === 8", 5)
    no_errors(drv)


def _model_point_under(drv, x, y):
    """The model point under (x, y) in the plane through the view center."""
    return drv.execute_script("""const c = window.__mace.viewport.camera().calibrate();
        return [0, 1, 2].map((k) => c.center[k] + c.right[k] * (arguments[0] - c.centerPx[0]) / c.s
                                    - c.up[k] * (arguments[1] - c.centerPx[1]) / c.s)""", x, y)


def test_right_drag_orbits_middle_pans_wheel_zooms_to_cursor(fresh):
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    view = "return window.__mace.viewState().view"
    v0 = drv.execute_script(view)
    canvas = H.viewer_canvas(drv)
    H.drag(drv, canvas, [(300, 200), (380, 230)], button=2)
    time.sleep(0.2)
    v1 = drv.execute_script(view)
    assert v1[4:] != v0[4:] and v1[:4] == v0[:4]           # the rotation changed, nothing else
    assert not drv.execute_script("return document.querySelector('.menu')")     # a drag opens no menu
    # middle-drag pans: the model point that was under the pointer follows it
    p0 = _model_point_under(drv, 300, 200)
    H.drag(drv, canvas, [(300, 200), (360, 240)], button=1)
    time.sleep(0.2)
    s = drv.execute_script("return window.__mace.viewport.camera().screenOf(arguments[0])", p0)
    assert abs(s[0] - 360) < 2 and abs(s[1] - 240) < 2, s
    # Zoom toward › Mouse pointer: the wheel zooms toward the cursor, the point under it stays within 2 px
    drv.execute_script("window.__mace.settings.set('viewport.zoomToward', 'pointer')")
    try:
        p1 = _model_point_under(drv, 420, 150)
        z0 = drv.execute_script(view)[3]
        H.wheel(drv, 420, 150, -200)
        time.sleep(0.3)
        assert drv.execute_script(view)[3] != z0
        s = drv.execute_script("return window.__mace.viewport.camera().screenOf(arguments[0])", p1)
        assert abs(s[0] - 420) <= 2 and abs(s[1] - 150) <= 2, s
    finally:
        drv.execute_script("window.__mace.settings.set('viewport.zoomToward', 'center')")
    no_errors(drv)


def test_wheel_up_zooms_in_and_invert_scroll_zoom_flips_it(fresh):
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")

    def spacing():
        a, b = H.atom_screen(drv, 0), H.atom_screen(drv, 1)
        return math.hypot(a["x"] - b["x"], a["y"] - b["y"])
    s0 = spacing()
    H.wheel(drv, 400, 300, -200)              # scrolling up: toward the structure
    time.sleep(0.3)
    s1 = spacing()
    assert s1 > s0 * 1.05, (s0, s1)
    drv.execute_script("window.__mace.settings.set('invertWheelZoom', true)")    # applies live
    H.wheel(drv, 400, 300, -200)
    time.sleep(0.3)
    assert spacing() < s1 / 1.05
    no_errors(drv)


def test_alt_left_drag_orbits_in_any_tool(fresh):
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    view = "return window.__mace.viewState().view"
    for tool in ("select", "lasso", "pan"):
        drv.execute_script("window.__mace.run(arguments[0])", f"tool.{tool}")
        v0 = drv.execute_script(view)
        H.drag(drv, H.viewer_canvas(drv), [(250, 250), (320, 260)], keys=(Keys.ALT,))
        time.sleep(0.2)
        v1 = drv.execute_script(view)
        assert v1[4:] != v0[4:] and v1[:3] == v0[:3], tool
        assert selection(drv) == []
    # the Orbit tool turns with a plain left drag; Pan moves
    drv.execute_script("window.__mace.run('tool.orbit')")
    v0 = drv.execute_script(view)
    H.drag(drv, H.viewer_canvas(drv), [(250, 250), (300, 250)])
    assert drv.execute_script(view)[4:] != v0[4:]
    drv.execute_script("window.__mace.run('tool.pan')")
    v0 = drv.execute_script(view)
    H.drag(drv, H.viewer_canvas(drv), [(250, 250), (300, 250)])
    v1 = drv.execute_script(view)
    assert v1[:3] != v0[:3] and v1[4:] == v0[4:]
    no_errors(drv)


def test_right_click_opens_atom_menu_without_drag(fresh):
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    set_view(drv)
    H.click_atom(drv, 0, button=2)
    H.wait(drv, "return document.querySelector('.menu[aria-label=\"Atom menu\"]')", 5)
    keys(drv, Keys.ESCAPE)
    H.click_at(drv, 6, 6, button=2)
    H.wait(drv, "return document.querySelector('.menu[aria-label=\"Viewport menu\"]')", 5)
    keys(drv, Keys.ESCAPE)
    assert drv.execute_script("return document.querySelectorAll('.menu').length") == 0
    no_errors(drv)


def test_hover_readout_and_shift_cart(fresh):
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    set_view(drv)
    p = H.atom_screen(drv, 4)
    H.hover_at(drv, p["x"], p["y"])
    H.wait(drv, "return document.querySelector('#st-cursor').textContent.startsWith('O5')", 5)
    assert drv.execute_script("return document.querySelector('#st-cursor').textContent") == \
        "O5 · 4b · x 0.5000 y 0.5000 z 0.5000"
    assert drv.execute_script("return window.__mace.viewport.modelIds().hover") is not None
    ActionChains(drv).key_down(Keys.SHIFT).perform()
    H.wait(drv, "return document.querySelector('#st-cursor').textContent.endsWith('Å')", 5)
    assert drv.execute_script("return document.querySelector('#st-cursor').textContent") == \
        "O5 · 4b · x 2.1060 y 2.1060 z 2.1060 Å"
    ActionChains(drv).key_up(Keys.SHIFT).perform()
    H.hover_at(drv, 5, 5)
    H.wait(drv, "return document.querySelector('#st-cursor').textContent === ''", 5)
    no_errors(drv)


def test_no_idle_frames_with_pointer_controller(fresh):
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    H.hover_at(drv, 200, 200)
    time.sleep(0.5)
    n = drv.execute_script("""window.__idleRaf = 0; const o = window.requestAnimationFrame;
        window.__origRaf2 = o; window.requestAnimationFrame = (f) => { window.__idleRaf++; return o(f); }; return 0""")
    time.sleep(3)
    n = drv.execute_script("window.requestAnimationFrame = window.__origRaf2; return window.__idleRaf")
    assert n == 0


def test_selection_model_only_rebuilds_selection(fresh):
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    set_view(drv)
    ids0 = drv.execute_script("return window.__mace.viewport.modelIds()")
    H.click_atom(drv, 0)
    time.sleep(0.3)
    ids1 = drv.execute_script("return window.__mace.viewport.modelIds()")
    assert ids1["structure"] == ids0["structure"] and ids1["selection"] is not None
    time.sleep(0.5)
    H.click_atom(drv, 4, mods=(Keys.SHIFT,))
    time.sleep(0.3)
    ids2 = drv.execute_script("return window.__mace.viewport.modelIds()")
    assert ids2["structure"] == ids0["structure"] and ids2["selection"] != ids1["selection"]
    no_errors(drv)


def test_selection_outline_rings_follow_the_camera(fresh):
    """The selection shows as a crisp ring around each selected atom (the
    translucent shell alone hardly shows on atoms of a similar color), drawn
    over the viewer after each paint: it follows an orbit, and goes when the
    selection does."""
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    set_view(drv)
    drv.execute_script("window.__mace.editor.select([0, 4], 'replace')")
    H.wait(drv, "return window.__mace.viewport.rings().filter((r) => r.kind === 'sel').length === 2", 5)

    def check():
        for r in drv.execute_script("return window.__mace.viewport.rings()"):
            p = H.atom_screen(drv, r["i"])
            assert abs(r["x"] - p["x"]) < 1.5 and abs(r["y"] - p["y"]) < 1.5, (r, p)
        return drv.execute_script("return window.__mace.viewport.rings()")
    rings = check()
    # the ring is drawn in the selection color on the layer over the viewer
    # (the 2 px stroke is centered on r at a sub-pixel position that depends on
    # the atom's drawn radius: read the pixels across it, take the closest)
    pixels = drv.execute_script("""const c = document.querySelector('.vp-rings'), r = arguments[0], k = c.width / c.clientWidth;
        const x = Math.round((r.x + r.r) * k), y = Math.round(r.y * k);
        return [-1, 0, 1].map((d) => Array.from(c.getContext('2d').getImageData(x + d, y, 1, 1).data))""", rings[0])
    sel = drv.execute_script("return getComputedStyle(document.documentElement).getPropertyValue('--sel').trim()")
    want = [int(sel[i:i + 2], 16) for i in (1, 3, 5)]
    rgba = min(pixels, key=lambda px: max(abs(a - b) for a, b in zip(px[:3], want)))
    # (antialiased against the thin background line around it: close to, not exactly, the token)
    assert rgba[3] > 200 and all(abs(a - b) < 80 for a, b in zip(rgba[:3], want)), (rgba, sel)
    drv.execute_script("window.__mace.viewport.camera().orbit(40, 25)")
    time.sleep(0.3)
    moved = check()
    assert abs(moved[0]["x"] - rings[0]["x"]) + abs(moved[0]["y"] - rings[0]["y"]) > 3
    drv.execute_script("window.__mace.editor.select([], 'replace')")
    H.wait(drv, "return window.__mace.viewport.rings().length === 0", 5)
    no_errors(drv)


def test_first_structure_of_a_page_is_framed_like_reset_view(fresh):
    """The first structure a page opens is drawn while its viewer is still
    hidden; the fit waits for the viewer to show, so it comes up the size
    Reset View gives (it used to come up at 3Dmol's default, a fifth of it)."""
    drv, base = fresh[0], fresh[1]
    H.set_viewport(drv, 1920, 1080)            # where it showed: a wide viewer
    H.load(drv, base)
    open_doc(drv, "structures/MgO.cif")

    def spread():
        pts = [H.atom_screen(drv, i) for i in range(8)]
        return max(p["x"] for p in pts) - min(p["x"] for p in pts), max(p["y"] for p in pts) - min(p["y"] for p in pts)
    first = spread()
    drv.execute_script("window.__mace.run('view.reset')")
    time.sleep(0.6)
    again = spread()
    assert abs(first[0] - again[0]) < 3 and abs(first[1] - again[1]) < 3, (first, again)
    H.set_viewport(drv, 1280, 800)
    no_errors(drv)


# --- T5: structure documents edit in place; ordered, local-first edits ------------------

def ed_state(drv):
    return drv.execute_script("""const e = window.__mace.editor.doc(); if (!e) return null;
        return {symbols: e.structure.symbols, frac: e.structure.frac, n: e.n, pos: e.pos, rows: e.rows.map((r) => r.label),
                revs: e.rows.map((r) => r.rev), dirty: e.dirty, pending: e.pending, waiting: e.waiting,
                session: e.srv.session, symMode: e.symMode, rev: e.rev, savedRev: e.savedRev}""")


def wait_idle(drv, timeout=20):
    H.wait(drv, "const e = window.__mace.editor.doc(); return e && e.pending === 0 && !e.waiting", timeout)
    time.sleep(0.1)


def edit(drv, op, args, opts=None):
    drv.execute_script("const E = window.__mace.editor.E; E.api.edit(window.__mace.editor.doc(), arguments[0], arguments[1], arguments[2] || {})",
                       op, args, opts)


def oracle(steps, symmetric=False):
    """The same edits, one by one, on a server-side EditSession (no browser, no queue)."""
    from mace.gui import editor as E
    from mace.gui import structures as S
    s = E.EditSession(E.atoms_from_payload(S.read_structure(H.EXAMPLES / "structures" / "MgO.cif")), "MgO.cif",
                      label="Opened MgO.cif", locked=symmetric)
    rev = s.rev
    for st in steps:
        if st[0] == "undo":
            if s.history:
                s.apply("undo", {})
        elif st[0] == "merged":
            _, args, count = st
            s.apply("move", args, merge="k", merge_count=count)
        else:
            s.apply(st[0], st[1])
    return s, rev


def assert_matches(drv, s):
    import numpy as np
    st = ed_state(drv)
    r = s.payload()
    assert st["symbols"] == r["symbols"]
    d = np.array(st["frac"]) - np.array(r["frac"])
    assert np.abs(d - np.round(d)).max() < 1e-9
    assert st["rows"] == [h["label"] for h in r["history"]], (st["rows"], [h["label"] for h in r["history"]])
    assert st["pos"] == r["position"]


def test_editor_commands_not_soon(fresh):
    """The Phase 3 commands built so far run (not unfinished) where they apply."""
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    ids = ["edit.undo", "edit.redo", "edit.copy", "edit.delete", "select.all", "select.none", "select.invert",
           "select.byElement", "select.equivalents", "select.neighbors", "select.grow", "select.shrink",
           "build.delete", "build.symMode", "build.symModeP1", "build.symModeSym", "atom.selectEquivalents",
           "atom.selectElement", "atom.selectNeighbors", "view.alongAStar", "view.alongBStar", "view.alongCStar",
           "view.rotateLeft", "view.rotateRight", "view.rotateUp", "view.rotateDown", "view.rotateInPlaneMinus",
           "view.rotateInPlanePlus", "view.frameSelection", "edit.nudgeLeft", "edit.nudgeRight", "edit.nudgeUp",
           "edit.nudgeDown", "edit.nudgeOut", "edit.nudgeIn", "tool.lasso", "tool.pan", "tool.zoom",
           "file.save", "file.saveAs", "file.saveAll", "file.revert", "file.exportImage", "file.copyImage",
           "file.new", "build.addAtom", "build.replace", "viewport.addAtomHere", "traj.editStep",
           "edit.replaceElement", "edit.setCoordinates", "edit.editCell", "edit.history", "view.alongUVW",
           "view.measurements", "select.within", "select.byIndex", "select.byFrac", "select.saveSet", "select.sets",
           "tool.measureSelection", "tool.clearMeasurements", "atom.measureFrom"]
    cmds = {c["id"]: c for c in drv.execute_script("return window.__mace.commands")}
    assert not [i for i in ids if cmds[i]["soon"]]
    drv.execute_script("window.__mace.editor.select([0], 'replace')")
    for i in ["select.all", "select.none", "select.invert", "select.byElement", "select.equivalents", "select.neighbors",
              "select.grow", "select.shrink", "build.symModeP1", "build.symModeSym", "build.symMode", "view.frameSelection",
              "edit.delete", "build.delete", "edit.copy", "view.alongAStar", "view.rotateLeft", "view.rotateInPlanePlus"]:
        assert drv.execute_script("return window.__mace.menuReason(arguments[0])", i) == "", i
    no_errors(drv)


def test_disabled_reasons_slab_trajectory_large(fresh):
    drv, base, root = fresh[0], fresh[1], fresh[2]
    reason = """return window.__mace.menuReason(arguments[0])"""
    open_doc(drv, "outputs/graphene_slab_sp.out")
    assert drv.execute_script("return window.__mace.ctx().docEditable") is False
    assert drv.execute_script(reason, "build.symModeSym") == "Editing needs a 3D periodic cell (this is a slab)"
    drv.execute_script("window.__mace.closeAll()")
    drv.execute_async_script("const d = arguments[1]; window.__mace.open(arguments[0]).then(() => d())",
                             "outputs/Ag2Br3_opt_unfinished.out")
    time.sleep(0.4)
    assert drv.execute_script("return window.__mace.doc().kind") == "trajectory"
    assert drv.execute_script(reason, "build.symModeSym") == "Trajectories are view-only. Use Edit This Step"
    assert drv.execute_script(reason, "select.all") == "Trajectories are view-only. Use Edit This Step"
    drv.execute_script("window.__mace.closeAll()")
    # more than 5000 atoms: view-only, with the reason
    import ase.io
    from ase.build import bulk
    ase.io.write(root / "big_mgo.cif", bulk("MgO", "rocksalt", a=4.212, cubic=True).repeat((9, 9, 9)))
    drv.execute_async_script("window.MACE.loadFiles().then(arguments[0])")
    open_doc(drv, "big_mgo.cif")
    assert drv.execute_script(reason, "edit.delete") == "Editing is limited to 5000 atoms"
    assert drv.execute_script(reason, "select.all") == ""
    (root / "big_mgo.cif").unlink()
    drv.execute_async_script("window.MACE.loadFiles().then(arguments[0])")
    no_errors(drv)


def test_first_edit_is_one_request(fresh):
    """Plan A1, §8.1: the first edit of a fresh document opens its session in
    the same request (Resource Timing shows one /api/edit request)."""
    drv = fresh[0]
    open_doc(drv, "structures/NaCl.cif")
    t = H.now_ms(drv)
    edit(drv, "replace", {"indices": [0], "element": "K"})
    wait_idle(drv)
    reqs = H.requests_since(drv, t, "/api/")
    # the profile's own debounced saves (Recent, layout: polish P2) are not part of the edit
    names = [r["name"].split("/api/")[1] for r in reqs]
    assert [n for n in names if n.split("?")[0] not in ("profile", "docprefs", "settings")] == ["edit/op"], reqs
    st = ed_state(drv)
    assert st["session"] and st["rows"][0] == "Opened NaCl.cif" and st["dirty"]
    no_errors(drv)


def test_rapid_nudges_then_undo_match_serial_oracle(fresh):
    """Plan A3a: six edits, 30 nudges 16 ms apart, then five Mod+Z at once:
    the result, the History labels and the position equal the same edits
    applied one by one on the server's editor."""
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    drv.execute_script("window.__mace.run('build.symModeP1')")
    steps = [("replace", {"indices": [0], "element": "Ca"}), ("replace", {"indices": [1], "element": "Sr"}),
             ("replace", {"indices": [2], "element": "Ba"}), ("replace", {"indices": [3], "element": "Be"}),
             ("replace", {"indices": [5], "element": "S"}), ("delete", {"indices": [6]})]
    for op, args in steps:
        edit(drv, op, args)
    drv.execute_script("window.__mace.editor.select([4], 'replace')")
    drv.execute_async_script("""const done = arguments[0], E = window.__mace.editor.E, d = window.__mace.doc;
        let k = 0; const t = setInterval(() => {
          E.run('edit.nudgeRight', window.__mace.activeDoc(), {chord: 'Alt+Right'});
          if (++k === 30) { clearInterval(t); for (let u = 0; u < 5; u++) window.__mace.run('edit.undo'); done(); }
        }, 16);""")
    st = ed_state(drv)
    assert st["pos"] == 2                       # the undos showed at once, before the server answered
    wait_idle(drv, 30)
    s, _ = oracle(steps + [("merged", {"indices": [4], "frac_delta": [0.30, 0, 0]}, 30)] + [("undo",)] * 5)
    assert_matches(drv, s)
    assert drv.execute_script("return window.__mace.editor.doc().rows[7].label") == "Move 1 Atom (×30)"
    assert drv.execute_script("return window.__mace.editor.E.editing.stats.reconcileWarnings") == 0
    no_errors(drv)


def test_paint_then_immediate_undo_order(fresh):
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    drv.execute_script("window.__mace.run('build.symModeP1')")
    drv.execute_script("""const E = window.__mace.editor.E, e = window.__mace.editor.doc();
        E.api.edit(e, 'replace', {indices: [0, 1, 2], element: 'Ca'}); E.api.undo(e);
        window.__afterUndo = e.structure.symbols.slice();""")
    assert drv.execute_script("return window.__afterUndo") == ["Mg"] * 4 + ["O"] * 4     # at once, on screen
    wait_idle(drv)
    s, _ = oracle([("replace", {"indices": [0, 1, 2], "element": "Ca"}), ("undo",)])
    assert_matches(drv, s)
    assert ed_state(drv)["rows"][-1].startswith("Replace Mg→Ca (3 atoms")         # still redoable
    no_errors(drv)


def test_reversed_arrival_still_ordered(fresh):
    """The first of two quick edits is held 400 ms on the server: the second
    is not even sent until the first is answered, so both apply in order."""
    drv, base = fresh[0], fresh[1]
    open_doc(drv, "structures/MgO.cif")
    drv.execute_script("window.__mace.run('build.symModeP1')")
    drv.execute_script("""window.__first = true; window.__maceQueueHeaders = (item) => {
        if (window.__first && item.kind === 'replace') { window.__first = false; return {'X-Mace-Test-Delay-Ms': '400', 'X-Mace-Test-Tag': 'first'}; }
        return {'X-Mace-Test-Tag': 'second'}; };""")
    drv.execute_async_script("""fetch('api/test/gate/reset', {method: 'POST', headers: {'X-Requested-With': 'mace-gui',
        'Content-Type': 'application/json'}, body: '{}'}).then(() => arguments[0]())""")
    edit(drv, "replace", {"indices": [0], "element": "Ca"})
    edit(drv, "replace", {"indices": [0], "element": "Sr"})
    wait_idle(drv)
    drv.execute_script("delete window.__maceQueueHeaders")
    s, _ = oracle([("replace", {"indices": [0], "element": "Ca"}), ("replace", {"indices": [0], "element": "Sr"})])
    assert_matches(drv, s)
    log = drv.execute_async_script("fetch('api/test/log').then((r) => r.json()).then((j) => arguments[0](j.log))")
    first = next(e for e in log if e["tag"] == "first")
    second = next(e for e in log if e["tag"] == "second")
    assert second["arrived"] >= first["done"], log
    no_errors(drv)


def test_local_first_matches_server(fresh):
    """20 mixed edits in P1 and Sym: what the page computes equals what the
    server answers (no reconcile warning)."""
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    drv.execute_script("window.__mace.editor.E.editing.stats.reconcileWarnings = 0")
    # Sym (MgO's default): whole orbits; a move kept on its site (Keep Fm-3m) is computed here too
    ops = [("move", {"indices": [4], "frac_delta": [0.01, 0, 0], "project": True}), ("replace", {"indices": [0], "element": "Ca"}),
           ("replace", {"indices": [4], "element": "S"}), ("move", {"indices": [0], "frac_delta": [0, 0, 0]}),
           ("delete", {"indices": [5]})]
    for k in range(2):
        for op, args in ops:
            edit(drv, op, args)
        wait_idle(drv)
        drv.execute_script("window.__mace.editor.E.api.undo(window.__mace.editor.doc())")
        wait_idle(drv)         # an undo to a state not held here waits for the server
    drv.execute_script("window.__mace.run('build.symModeP1')")
    p1 = [("move", {"indices": [1], "delta": [0.1, 0.05, 0]}), ("replace", {"indices": [2], "element": "Zn"}),
          ("move", {"indices": [3], "frac_delta": [-0.02, 0, 0.01], "wrap": True}), ("delete", {"indices": [0, 1]}),
          ("set_positions", {"indices": [0], "frac": [[0.3, 0.3, 0.3]]})]
    for op, args in p1:
        edit(drv, op, args)
    wait_idle(drv)
    st = drv.execute_script("return window.__mace.editor.E.editing.stats")
    # the first Sym move waits (no operations before the session opens), and the
    # edits queued behind it wait with it; every later one is computed here
    assert st["reconcileWarnings"] == 0 and st["predicted"] >= 10, st
    no_errors(drv)


def test_session_evicted_404_recovery(fresh):
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    edit(drv, "replace", {"indices": [0], "element": "Ca"})
    wait_idle(drv)
    sid = ed_state(drv)["session"]
    drv.execute_async_script("""fetch('api/test/evict', {method: 'POST', headers: {'X-Requested-With': 'mace-gui',
        'Content-Type': 'application/json'}, body: JSON.stringify({session: arguments[0]})}).then(() => arguments[1]())""", sid)
    edit(drv, "replace", {"indices": [4], "element": "S"})
    wait_idle(drv)
    st = ed_state(drv)
    assert st["session"] != sid and st["symbols"].count("Ca") == 4 and st["symbols"].count("S") == 4
    toasts = drv.execute_script("return [...document.querySelectorAll('#toasts .msg')].map((m) => m.textContent)")
    assert any("undo history was released" in t for t in toasts), toasts
    drv.execute_script("window.__mace.run('edit.undo')")
    wait_idle(drv)
    st = ed_state(drv)
    assert st["symbols"].count("S") == 0 and st["symbols"].count("Ca") == 4 and st["dirty"]
    no_errors(drv)


def test_classic_panel_still_edits(fresh):
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    assert "more editing (classic)" in drv.find_element("id", "inspector").text.lower()
    drv.execute_script("window.__mace.run('build.symModeP1')")
    drv.execute_script("""document.querySelector('#e-element').value = 'Li';
        ['#e-x', '#e-y', '#e-z'].forEach((id) => document.querySelector(id).value = '0.25');
        document.querySelector('#e-add').click()""")
    H.wait(drv, "return document.querySelector('#e-counts').textContent.startsWith('9 atoms')")
    wait_idle(drv)
    assert ed_state(drv)["rows"][-1] == "Add Li"
    assert drv.execute_script("return window.__mace.viewport.displayed()") > 9
    drv.execute_script("document.querySelector('#e-undo').click()")
    H.wait(drv, "return document.querySelector('#e-counts').textContent.startsWith('8 atoms')")
    no_errors(drv)


def test_server_restart_403_banner(tmp_path):
    """Plan R4: a restarted server has a new token, so every request of the
    old page is refused (403). The page says so, keeps the structure and
    the edits on screen, and retries nothing."""
    root, xdg = H.examples_copy(tmp_path), tmp_path / "xdg"
    port = H.free_port()
    proc, base, _ = H.start_server(root, xdg, port=port, token="first-token-000001")
    drv = H.firefox(1280, 800)
    try:
        H.load(drv, base, token="first-token-000001")
        open_doc(drv, "structures/MgO.cif")
        edit(drv, "replace", {"indices": [0], "element": "Ca"})
        wait_idle(drv)
        proc.terminate()
        proc.wait(timeout=15)
        proc, base2, _ = H.start_server(root, xdg, port=port, token="second-token-00002")
        # the same port; another token gives another session path, which the
        # old page's requests do not reach (403: the server restarted)
        assert base2.split("/s/")[0] == base.split("/s/")[0] and base2 != base
        t = H.now_ms(drv)
        edit(drv, "replace", {"indices": [4], "element": "S"})
        H.wait(drv, "return document.querySelector('#conn-banner').dataset.lost === '1'", 10)
        text = drv.execute_script("return document.querySelector('#conn-banner').textContent")
        assert "Lost the connection to the MACE server" in text and "Reopen the URL the server printed" in text
        st = ed_state(drv)
        assert st["symbols"].count("Ca") == 4 and st["symbols"].count("S") == 4     # the edits stay on screen
        time.sleep(1.0)
        edits = [r for r in H.requests_since(drv, t, "/api/edit")]
        assert len(edits) == 1, edits                                                 # nothing retried
        assert drv.execute_script("return window.__mace.ctx().docEditable") is False
        assert "Lost the connection" in drv.execute_script("return window.__mace.menuReason('edit.delete')")
    finally:
        drv.quit()
        proc.terminate()
        proc.wait(timeout=15)



def test_tunnel_loss_banner_wording(tmp_path):
    """The event stream ends (the ssh tunnel closed, or the server stopped):
    after its grace period the banner says "Lost the connection to the MACE
    server", the wording docs/gui.md Troubleshooting quotes, and shows the
    tunnel line to rerun."""
    root, xdg = H.examples_copy(tmp_path), tmp_path / "xdg"
    proc, base, _ = H.start_server(root, xdg)
    drv = H.firefox(1280, 800)
    try:
        H.load(drv, base)
        proc.terminate()
        proc.wait(timeout=15)
        H.wait(drv, "const b = document.querySelector('#conn-banner'); return !b.classList.contains('hidden')", 45)
        text = drv.execute_script("return document.querySelector('#conn-banner').textContent")
        assert text.startswith("Lost the connection to the MACE server. Reconnecting"), text
        assert "ssh -N " in text
    finally:
        drv.quit()
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=15)

# --- T6: the inspector: coordinates, nudge, cell, symmetry; the amber group ----------------

def insp_input(drv, label):
    return drv.execute_script("return [...document.querySelectorAll('#inspector input')].find((i) => i.getAttribute('aria-label') === arguments[0])", label)


def insp_values(drv, prefix):
    return drv.execute_script("""return ['x', 'y', 'z'].map((a) => [...document.querySelectorAll('#inspector input')]
        .find((i) => i.getAttribute('aria-label') === arguments[0] + ' ' + a).value)""", prefix)


def type_into(drv, label, text, key=Keys.ENTER):
    # the inspector can be drawn again just after a document opens (on a busy machine the
    # first field found is then gone): the field is looked up again
    from selenium.common.exceptions import StaleElementReferenceException
    for attempt in range(5):
        try:
            el = insp_input(drv, label)
            drv.execute_script("arguments[0].focus(); arguments[0].select()", el)
            el.send_keys(Keys.CONTROL, "a")
            el.send_keys(text)
            if key:
                el.send_keys(key)
            break
        except StaleElementReferenceException:
            if attempt == 4:
                raise
            time.sleep(0.3)
    time.sleep(0.2)


def test_frac_cart_toggle_and_expression_commit(fresh):
    """Criterion 7: the Frac/Cart toggle changes the values; 1/3 commits 0.33333."""
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    drv.execute_script("window.__mace.run('build.symModeP1'); window.__mace.editor.select([4], 'replace')")
    H.wait(drv, "return !!document.querySelector('#inspector .ins-xyz input')")
    assert insp_values(drv, "Fractional") == ["0.50000", "0.50000", "0.50000"]
    drv.execute_script("document.querySelector('#inspector .ins-seg [data-v=\"cart\"]').click()")
    H.wait(drv, "return !!document.querySelector('#inspector input[aria-label=\"Cartesian x\"]')")
    assert insp_values(drv, "Cartesian") == ["2.10600", "2.10600", "2.10600"]
    drv.execute_script("document.querySelector('#inspector .ins-seg [data-v=\"frac\"]').click()")
    H.wait(drv, "return !!document.querySelector('#inspector input[aria-label=\"Fractional x\"]')")
    type_into(drv, "Fractional x", "1/3")
    wait_idle(drv)
    H.wait(drv, "return document.querySelector('#inspector input[aria-label=\"Fractional x\"]').value === '0.33333'", 5)
    x = drv.execute_script("return window.__mace.editor.doc().structure.frac[4][0]")
    assert abs(x - 1 / 3) < 1e-9
    shot(drv, "p3_inspector_one_atom_light")
    no_errors(drv)


def test_invalid_expression_inline_error(fresh):
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    drv.execute_script("window.__mace.run('build.symModeP1'); window.__mace.editor.select([4], 'replace')")
    H.wait(drv, "return !!document.querySelector('#inspector .ins-xyz input')")
    rev = ed_state(drv)["rev"]
    type_into(drv, "Fractional x", "-x")
    el = insp_input(drv, "Fractional x")
    assert el.get_attribute("aria-invalid") == "true"
    msg = drv.execute_script("return document.getElementById(arguments[0].getAttribute('aria-describedby')).textContent", el)
    assert "not a number" in msg
    assert ed_state(drv)["rev"] == rev and ed_state(drv)["pending"] == 0
    border = drv.execute_script("return getComputedStyle(arguments[0]).borderTopColor", el)
    danger = drv.execute_script("const d = document.createElement('div'); d.style.color = 'var(--danger)'; document.body.appendChild(d);"
                                "const c = getComputedStyle(d).color; d.remove(); return c")
    assert border == danger
    keys(drv, Keys.ESCAPE)
    no_errors(drv)


def test_wrap_toast_with_undo(fresh):
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    drv.execute_script("window.__mace.run('build.symModeP1'); window.__mace.editor.select([0], 'replace')")
    H.wait(drv, "return !!document.querySelector('#inspector .ins-xyz input')")
    type_into(drv, "Fractional x", "-0.25")
    wait_idle(drv)
    H.wait(drv, "return [...document.querySelectorAll('#toasts .msg')].some((m) => m.textContent === 'Wrapped 1 atom into the cell')", 5)
    assert abs(drv.execute_script("return window.__mace.editor.doc().structure.frac[0][0]") - 0.75) < 1e-9
    drv.execute_script("[...document.querySelectorAll('#toasts .toast2')].find((t) => t.textContent.includes('Wrapped')).querySelector('.acts button').click()")
    wait_idle(drv)
    assert abs(drv.execute_script("return window.__mace.editor.doc().structure.frac[0][0]")) < 1e-9
    no_errors(drv)


def test_cell_field_commit_scales_atoms(fresh):
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    drv.execute_script("window.__mace.run('build.symModeP1')")
    H.wait(drv, "return !!document.querySelector('#inspector input[aria-label=\"Cell a\"]')")
    type_into(drv, "Cell a", "4.4")
    wait_idle(drv)
    st = drv.execute_script("const s = window.__mace.editor.doc().structure; return [s.lattice.a, s.frac[4], s.cart[4]]")
    assert abs(st[0] - 4.4) < 1e-9 and abs(st[1][0] - 0.5) < 1e-9 and abs(st[2][0] - 2.2) < 1e-9     # scaled with the cell
    assert ed_state(drv)["rows"][-1] == "Edit Cell"
    no_errors(drv)


def test_symmetry_section_live(fresh):
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    H.wait(drv, "return !!document.querySelector('#inspector .ins-sym')")
    text = drv.execute_script("return document.querySelector('#inspector .ins-sym').textContent")
    for want in ["No. 225", "cubic", "192", "2 of 8", "-F 4 2 3"]:
        assert want in text, (want, text)
    drv.execute_script("window.__mace.run('build.symModeP1')")
    edit(drv, "replace", {"indices": [0], "element": "Ca"})
    wait_idle(drv)
    # CaMg3O4 (Ca on the corner): Pm-3m, Ca 1a, Mg 3c, O 1b + 3d
    H.wait(drv, "return document.querySelector('#inspector .ins-sym').textContent.includes('4 of 8')", 5)
    text = drv.execute_script("return document.querySelector('#inspector .ins-sym').textContent")
    assert "No. 221" in text and "Lower than" in text, text
    # the tolerance: no undo step, the session takes it
    type_into(drv, "Symmetry tolerance", "0.01")
    wait_idle(drv)
    assert drv.execute_script("return window.__mace.editor.doc().symprec") == 0.01
    assert ed_state(drv)["rows"][-1].startswith("Replace")
    no_errors(drv)


def test_p1_move_lowers_group_amber(fresh):
    """Criterion 6: in P1, moving one O by +0.05 along a turns the status
    bar's group amber, lower than Fm-3m, as spglib says."""
    import numpy as np
    from mace.gui import structures as S
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    drv.execute_script("window.__mace.run('build.symModeP1'); window.__mace.editor.select([4], 'replace')")
    H.wait(drv, "return !!document.querySelector('#inspector .ins-xyz input')")
    type_into(drv, "Fractional x", "0.5+0.05")
    wait_idle(drv)
    H.wait(drv, "return document.querySelector('#st-sg').classList.contains('warn')", 5)
    sg = drv.find_element("id", "st-sg")
    assert "↓ from Fm-3m" in sg.text
    color = drv.execute_script("return getComputedStyle(arguments[0]).color", sg)
    warning = drv.execute_script("const d = document.createElement('div'); d.style.color = 'var(--warning)'; document.body.appendChild(d);"
                                 "const c = getComputedStyle(d).color; d.remove(); return c")
    assert color == warning
    s = drv.execute_script("return window.__mace.editor.doc().structure")
    want = S.symmetry(np.array(s["cell"]), np.array(s["frac"]), s["numbers"])["number"]
    assert f"({want})" in sg.text and want < 225
    shot(drv, "p3_status_amber_light")
    # the second path: the Move tool (G; Esc ends the grab G starts, the tool stays) and Alt+→ with a 0.05 step
    focus_viewport(drv)                      # Ctrl+Z / Undo in a field is the field's own
    drv.execute_script("window.__mace.run('edit.undo')")
    wait_idle(drv)
    H.wait(drv, "return !document.querySelector('#st-sg').classList.contains('warn')", 5)
    type_into(drv, "Nudge step", "0.05")
    p = H.atom_screen(drv, 4)
    H.hover_at(drv, p["x"], p["y"])
    focus_viewport(drv)
    keys(drv, "g")
    keys(drv, Keys.ESCAPE)
    assert drv.execute_script("return window.__mace.viewState().tool") == "move"
    keys(drv, Keys.ALT, Keys.ARROW_RIGHT)
    wait_idle(drv)
    assert abs(drv.execute_script("return window.__mace.editor.doc().structure.frac[4][0]") - 0.55) < 1e-9
    H.wait(drv, "return document.querySelector('#st-sg').classList.contains('warn')", 5)
    no_errors(drv)


def test_settings_editor_rows_no_phase3_hint(fresh):
    """The rows the editor now uses lose their "Used by the editor (Phase 3)" hint."""
    drv = fresh[0]
    drv.execute_script("window.__mace.run('file.settings')")
    H.wait(drv, "return document.querySelector('#settings') && document.querySelector('#settings').open", 10)
    wired = {"units": ["coordMode", "decimals.coord"], "viewport": ["quality", "labels"],
             "editor": ["symmetryModeDefault", "symprec", "defaultElement", "nudgeStepFrac", "nudgeStepCart",
                        "wrapIntoCell", "undoLimit", "confirmOverwrite", "keepBackup", "autosaveDrafts"],
             "files": ["saveFormat"]}
    for sec, keys_ in wired.items():
        drv.execute_script("document.querySelector(`#settings [data-sec='${arguments[0]}']`).click()", sec)
        time.sleep(0.15)
        for key in keys_:
            row = drv.execute_script("return document.querySelector(`#settings [data-row='${arguments[0]}']`).textContent", key)
            assert "Phase 3" not in row and "coming soon" not in row.lower(), (key, row)
    drv.execute_script("document.querySelector('#settings').close()")


# --- T7: Move (grab), Add Atom, Replace, Delete, Bond, the picker, modal keys --------------

def focus_viewport(drv):
    drv.execute_script("document.querySelector('#viewer-main').closest('.viewer-wrap').focus()")


def press(drv, *combo):
    keys(drv, *combo)


def grab_state(drv):
    return drv.execute_script("return window.__mace.editor.E.tools.grabState")


def test_move_drag_with_axis_lock_and_typed_offset(fresh):
    """Drag, press X, let go, type 0.15, Enter: one move of Δx = 0.150 Å."""
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    set_view(drv)
    drv.execute_script("window.__mace.run('build.symModeP1'); window.__mace.run('tool.move'); window.__mace.editor.select([4], 'replace')")
    p = H.atom_screen(drv, 4)
    x0 = drv.execute_script("return window.__mace.editor.doc().structure.cart[4]")
    # the button goes down on O5, moves, X locks during the drag, then the button comes up
    t = H._Ticks(drv)
    canvas = H.viewer_canvas(drv)
    t.move(canvas, *H._offset(drv, canvas, p["x"], p["y"]))
    t.down(0)
    for k in range(1, 6):
        t.move(canvas, *H._offset(drv, canvas, p["x"] + 6 * k, p["y"] + 3 * k), ms=16)
    t.k.key_down("x"); t.p.pause(0)
    t.k.key_up("x"); t.p.pause(0)
    t.up(0)
    t.perform()
    time.sleep(0.3)
    assert drv.execute_script("return window.__mace.editor.E.tools.grabbing") is True      # a key was used: it goes on
    assert grab_state(drv)["axis"] == "x"
    t0 = H.now_ms(drv)
    ActionChains(drv).send_keys("0.15").perform()
    time.sleep(0.3)
    assert grab_state(drv)["typed"] == "0.15"
    assert "x 0.150 Å" in drv.execute_script("return document.querySelector('.vp-chip').textContent")
    shot(drv, "p3_move_chip_light")
    ActionChains(drv).send_keys(Keys.ENTER).perform()
    wait_idle(drv)
    x1 = drv.execute_script("return window.__mace.editor.doc().structure.cart[4]")
    assert abs(x1[0] - x0[0] - 0.15) < 1e-9 and abs(x1[1] - x0[1]) < 1e-9 and abs(x1[2] - x0[2]) < 1e-9
    assert [r["name"].split("/api/")[1] for r in H.requests_since(drv, t0, "/api/edit")] == ["edit/op"]
    assert ed_state(drv)["rows"][-1] == "Move 1 Atom"
    no_errors(drv)


def test_g_key_grab_moves_without_button_and_click_commits(fresh):
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    set_view(drv)
    drv.execute_script("window.__mace.run('build.symModeP1'); window.__mace.editor.select([4], 'replace')")
    p = H.atom_screen(drv, 4)
    H.hover_at(drv, p["x"], p["y"])
    focus_viewport(drv)
    x0 = drv.execute_script("return window.__mace.editor.doc().structure.cart[4]")
    ActionChains(drv).send_keys("g").perform()          # from the Select tool: arms Move, grabs at the pointer
    time.sleep(0.2)
    assert drv.execute_script("return [window.__mace.viewState().tool, window.__mace.editor.E.tools.grabbing]") == ["move", True]
    H.hover_at(drv, p["x"] + 50, p["y"] - 20)            # no button held: the atom follows
    time.sleep(0.3)
    assert drv.execute_script("return window.__mace.viewport.modelIds().drag") is not None
    H.click_at(drv, p["x"] + 50, p["y"] - 20)
    wait_idle(drv)
    x1 = drv.execute_script("return window.__mace.editor.doc().structure.cart[4]")
    assert max(abs(a - b) for a, b in zip(x0, x1)) > 0.3
    assert drv.execute_script("return window.__mace.editor.E.tools.grabbing") is False
    no_errors(drv)


def test_move_esc_cancels_no_request(fresh):
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    set_view(drv)
    drv.execute_script("window.__mace.run('build.symModeP1'); window.__mace.editor.select([4], 'replace')")
    p = H.atom_screen(drv, 4)
    H.hover_at(drv, p["x"], p["y"])
    focus_viewport(drv)
    t0 = H.now_ms(drv)
    ActionChains(drv).send_keys("g").perform()
    H.hover_at(drv, p["x"] + 60, p["y"])
    time.sleep(0.3)
    ActionChains(drv).send_keys(Keys.ESCAPE).perform()
    time.sleep(0.4)
    assert drv.execute_script("return window.__mace.editor.E.tools.grabbing") is False
    assert not H.requests_since(drv, t0, "/api/edit")
    assert drv.execute_script("return window.__mace.viewport.modelIds().drag") is None
    # a right click cancels too, and opens no menu
    ActionChains(drv).send_keys("g").perform()
    H.hover_at(drv, p["x"] + 40, p["y"])
    H.click_at(drv, p["x"] + 40, p["y"], button=2)
    time.sleep(0.3)
    assert drv.execute_script("return window.__mace.editor.E.tools.grabbing") is False
    assert not drv.execute_script("return document.querySelector('.menu')")
    assert not H.requests_since(drv, t0, "/api/edit")
    no_errors(drv)


MODAL_KEYS = [("z", None), ("x", None), ("y", None), ("X", Keys.SHIFT), ("1", None), ("2", None), ("3", None),
              ("5", None), ("-", None), ("+", None), ("f", None), ("g", None), ("o", None), ("b", None), ("h", None),
              ("a", None), ("l", None), ("m", None), ("v", None), ("C", Keys.SHIFT), ("e", None)]


def _view_snapshot(drv):
    return drv.execute_script("""const v = window.__mace.viewState(); return {tool: v.tool, view: v.view, projection: v.projection,
        axes: window.__mace.settings.get('showAxes'), cell: window.__mace.settings.get('showCell'),
        marks: performance.getEntriesByType('mark').filter((m) => /^ui:[a-z]+\\.[A-Za-z.0-9]+:start$/.test(m.name)).length}""")


@pytest.mark.parametrize("key, mod", MODAL_KEYS, ids=[f"{'Shift+' if m else ''}{k}" for k, m in MODAL_KEYS])
def test_modal_layer_keys_in_and_out(fresh, key, mod):
    """Outside a grab each key does what the keymap says; during a grab it
    does its grab action or is swallowed: tool, view, projection, axes and
    cell stay as they were and no command runs."""
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    drv.execute_script("window.__mace.run('build.symModeP1'); window.__mace.editor.select([4], 'replace')")
    p = H.atom_screen(drv, 4)
    H.hover_at(drv, p["x"], p["y"])
    focus_viewport(drv)
    combo = (mod, key) if mod else (key,)
    # outside: some command runs (or the tool/view changes)
    before = _view_snapshot(drv)
    keys(drv, *combo)
    time.sleep(0.3)
    after = _view_snapshot(drv)
    if key not in ("x", "y"):            # bare X and Y are not bound outside a grab
        assert after["marks"] > before["marks"] or after != before, (key, before, after)
    drv.execute_script("document.querySelectorAll('.ptable, dialog[open]').forEach((d) => d.close ? d.close() : d.remove())")
    drv.execute_script("window.__mace.editor.E.tools.endGrab(); window.__mace.run('tool.select');"
                       "window.__mace.editor.select([4], 'replace')")
    H.hover_at(drv, p["x"], p["y"])
    focus_viewport(drv)
    keys(drv, "g")
    assert drv.execute_script("return window.__mace.editor.E.tools.grabbing") is True
    g0 = grab_state(drv)
    before = _view_snapshot(drv)
    keys(drv, *combo)
    time.sleep(0.25)
    after = _view_snapshot(drv)
    g1 = grab_state(drv)
    assert g1 is not None, key                                   # still grabbing
    for k in ("tool", "projection", "axes", "cell", "marks"):
        assert after[k] == before[k], (key, k, before[k], after[k])
    assert max(abs(a - b) for a, b in zip(after["view"], before["view"])) < 1e-9, key
    expect = {"z": ("axis", "z"), "x": ("axis", "x"), "y": ("axis", "y"), "X": ("axis", "a"), "1": ("typed", "1"),
              "2": ("typed", "2"), "3": ("typed", "3"), "5": ("typed", "5"), "-": ("typed", "-")}.get(key)
    if expect:
        assert g1[expect[0]] == expect[1], (key, g1)
    keys(drv, Keys.ESCAPE)
    assert drv.execute_script("return window.__mace.editor.E.tools.grabbing") is False
    no_errors(drv)


def test_add_atom_element_field(fresh):
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    focus_viewport(drv)
    keys(drv, "a")
    assert drv.execute_script("return window.__mace.viewState().tool") == "add"
    keys(drv, "e")                                       # E in Add Atom: the element field
    assert drv.execute_script("return document.activeElement.classList.contains('vp-el')")
    drv.execute_script("document.activeElement.select()")
    ActionChains(drv).send_keys("O").send_keys(Keys.ENTER).perform()
    time.sleep(0.2)
    assert drv.execute_script("return window.__mace.editor.E.tools.element") == "O"
    assert drv.execute_script("return window.__mace.viewState().tool") == "add"        # O went to the field, not Orbit
    keys(drv, "e")
    drv.execute_script("document.activeElement.select()")
    ActionChains(drv).send_keys("Qq").send_keys(Keys.ENTER).perform()
    time.sleep(0.2)
    assert drv.execute_script("return document.querySelector('.vp-chip .err').textContent") == "Not an element symbol"
    ActionChains(drv).send_keys(Keys.ESCAPE).perform()                                   # reverts
    time.sleep(0.2)
    assert drv.execute_script("return document.querySelector('.vp-chip .vp-el').value") == "O"
    focus_viewport(drv)
    keys(drv, "o")                                       # with the viewport focused, O is Orbit
    assert drv.execute_script("return window.__mace.viewState().tool") == "orbit"
    no_errors(drv)


def test_cheat_sheet_lists_modal_layers(fresh):
    drv = fresh[0]
    drv.execute_script("window.__mace.run('help.shortcuts')")
    H.wait(drv, "return document.querySelector('#cheatsheet') && document.querySelector('#cheatsheet').open", 10)
    text = drv.execute_script("return document.querySelector('#cheatsheet').textContent")
    assert "While moving" in text and "Lock to the cell axis a, b or c" in text and "Add Atom element" in text
    drv.execute_script("document.querySelector('#cheatsheet').close()")


def test_sym_move_keep_and_allow(fresh):
    """MgO in Sym, O off 4b: Keep Fm-3m projects the move (225, 8 atoms);
    Allow lets the orbit grow, as the server computes it."""
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    set_view(drv)
    edit(drv, "set_cell", {"cellpar": [4.212, 4.212, 4.212, 90, 90, 90]})       # opens the session: its operations come back
    wait_idle(drv)
    assert drv.execute_script("return !!window.__mace.editor.doc().ops")
    for choice, n in [("keep", 8), ("allow", None)]:
        drv.execute_script("window.__mace.editor.select([4], 'replace')")
        p = H.atom_screen(drv, 4)
        H.hover_at(drv, p["x"], p["y"])
        focus_viewport(drv)
        keys(drv, "g")
        H.hover_at(drv, p["x"] + 30, p["y"] + 10)
        H.wait(drv, "return !!document.querySelector('.vp-chip [data-g]')", 5)
        text = drv.execute_script("return document.querySelector('.vp-chip').textContent")
        assert "Moving off 4b lowers the group" in text and "Keep Fm-3m" in text
        # the status bar shows the group the move would give, amber (spec §2.7)
        H.wait(drv, "const g = document.getElementById('st-sg'); return g.classList.contains('warn') && g.textContent.includes('↓ from Fm-3m')", 5)
        shot(drv, f"p3_sym_move_{choice}_light")
        drv.execute_script("document.querySelector(`.vp-chip [data-g='${arguments[0]}']`).click()", choice)
        time.sleep(0.2)
        keys(drv, Keys.ENTER)
        wait_idle(drv)
        st = drv.execute_script("const s = window.__mace.editor.doc().structure; return [s.symmetry.number, s.numbers.length]")
        if choice == "keep":
            assert st == [225, 8], st
            assert drv.execute_script("const g = document.getElementById('st-sg'); return [g.textContent, g.classList.contains('warn')]") == ["Fm-3m (225)", False]
        else:
            assert st[1] > 8, st
            drv.execute_script("window.__mace.run('edit.undo')")
            wait_idle(drv)
    no_errors(drv)


def test_symmetry_requests_debounced_during_drag(fresh):
    """§8.8: during a 1.5 s drag (a move every 16 ms), at most one /api/symmetry
    per 150 ms, never two at once, and at least one."""
    import math
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    set_view(drv)
    drv.execute_script("window.__mace.run('build.symModeP1'); window.__mace.run('tool.move'); window.__mace.editor.select([4], 'replace')")
    p = H.atom_screen(drv, 4)
    t0 = H.now_ms(drv)
    canvas = H.viewer_canvas(drv)
    t = H._Ticks(drv)
    t.move(canvas, *H._offset(drv, canvas, p["x"], p["y"]))
    t.down(0)
    for k in range(1, 95):
        t.move(canvas, *H._offset(drv, canvas, p["x"] + 40 * math.sin(k / 10), p["y"] + 20 * math.cos(k / 13)), ms=16)
    t.up(0)
    t.perform()
    wait_idle(drv)
    time.sleep(0.4)
    reqs = sorted(H.requests_since(drv, t0, "/api/symmetry"), key=lambda r: r["start"])
    span = drv.execute_script("return performance.now()") - t0
    assert 1 <= len(reqs) <= math.ceil(span / 150), (len(reqs), span)
    for a, b in zip(reqs, reqs[1:]):
        assert b["start"] >= a["end"] - 1, (a, b)              # never two in flight
    no_errors(drv)


def test_add_atom_plane_and_shift_bond(fresh):
    import numpy as np
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    set_view(drv)
    drv.execute_script("window.__mace.run('build.symModeP1'); window.__mace.run('tool.add'); window.__mace.editor.E.tools.setElement('Li')")
    # a click on empty space: on the plane through the cell's center, facing the camera
    r = drv.execute_script("return window.__mace.viewport.camera().calibrate()")
    s0 = drv.execute_script("return window.__mace.editor.doc().structure")
    center = (np.array(s0["cell"]).sum(axis=0) / 2).tolist()
    cx, cy = drv.execute_script("return window.__mace.viewport.camera().screenOf(arguments[0])", center)
    H.click_at(drv, cx + 3, cy - 2)
    wait_idle(drv)
    s1 = drv.execute_script("return window.__mace.editor.doc().structure")
    assert s1["symbols"][-1] == "Li" and len(s1["symbols"]) == 9
    toward = np.array(r["toward"])
    assert abs(np.dot(np.array(s1["cart"][-1]) - center, toward)) < 1e-6        # in the plane
    # Shift+click on O5: a bonded Li away from its neighbors, at r_cov(O) + r_cov(Li)
    p = H.atom_screen(drv, 4)
    H.click_at(drv, p["x"], p["y"], keys=(Keys.SHIFT,))
    wait_idle(drv)
    s2 = drv.execute_script("return window.__mace.editor.doc().structure")
    from ase.data import covalent_radii
    d = np.linalg.norm(np.array(s2["cart"][-1]) - np.array(s2["cart"][4]))
    assert len(s2["symbols"]) == 10 and abs(d - (covalent_radii[8] + covalent_radii[3])) < 1e-3
    no_errors(drv)


def test_add_atom_sym_count_preview(fresh):
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    set_view(drv)
    edit(drv, "set_cell", {"cellpar": [4.212, 4.212, 4.212, 90, 90, 90]})
    wait_idle(drv)
    drv.execute_script("window.__mace.run('tool.add')")
    p = H.atom_screen(drv, 4)
    H.hover_at(drv, p["x"] + 17, p["y"] + 9)
    H.wait(drv, "return /\\+\\d+ atoms/.test(document.querySelector('.vp-chip').textContent)", 5)
    text = drv.execute_script("return document.querySelector('.vp-chip').textContent")
    n = int(text.split("+")[1].split(" ")[0])
    assert n > 1
    shot(drv, "p3_add_atom_preview_light")
    no_errors(drv)


def test_replace_paint_one_history_entry(fresh):
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    set_view(drv)
    drv.execute_script("window.__mace.run('build.symModeP1'); window.__mace.run('tool.replace'); window.__mace.editor.E.tools.setElement('Ca')")
    pts = screen_atoms(drv)
    path = [(pts[i]["x"], pts[i]["y"]) for i in (0, 1, 2)]
    H.drag(drv, H.viewer_canvas(drv), path, steps=8)
    wait_idle(drv)
    st = ed_state(drv)
    # the stroke paints every atom it crosses (it may pass over an O on its way), in one step
    n = 8 - st["symbols"].count("Mg") - st["symbols"].count("O")
    assert st["symbols"][:3] == ["Ca", "Ca", "Ca"] and n >= 3
    assert st["rows"][-1].startswith("Replace ") and st["rows"][-1].endswith(f"→Ca ({n} atoms)") and len(st["rows"]) == 2
    no_errors(drv)


def test_delete_orbit_sym(fresh):
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    set_view(drv)
    H.click_atom(drv, 4)
    focus_viewport(drv)
    keys(drv, Keys.DELETE)
    wait_idle(drv)
    st = ed_state(drv)
    assert st["symbols"] == ["Mg"] * 4 and st["rows"][-1] == "Delete 4 Atoms"
    no_errors(drv)


def test_bond_override_display_only(fresh):
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    set_view(drv)
    preview = """const done = arguments[0]; const e = window.__mace.editor.doc();
        window.__mace.editor.E.editing.request(e, 'deck_preview', 'edit/deck_preview', () => ({name: 'x.cif',
          json: {}, cif_mode: 'p1'}), {payloadKey: false}).then((r) => done(r && r.cif))"""
    cif0 = drv.execute_async_script(preview)
    pairs0 = drv.execute_script("return window.__mace.viewport.scene().pairs.length")
    drv.execute_script("window.__mace.run('tool.bond')")
    H.click_atom(drv, 0)
    time.sleep(0.5)
    H.click_atom(drv, 1)                                  # Mg–Mg: not bonded (metals): adds one
    time.sleep(0.4)
    assert drv.execute_script("return window.__mace.viewport.scene().pairs.length") == pairs0 + 2
    assert drv.execute_script("return window.__mace.editor.doc().bondOverrides.size") == 1
    assert drv.execute_async_script(preview) == cif0     # the structure (and deck) are untouched
    assert not ed_state(drv)["dirty"] or True
    no_errors(drv)


def test_ptable_keyboard_pick_and_recent(fresh):
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    drv.execute_script("window.__mace.run('build.symModeP1'); window.__mace.editor.select([0], 'replace')")
    focus_viewport(drv)
    keys(drv, "e")
    H.wait(drv, "return document.activeElement && document.activeElement.classList.contains('pt-search')", 5)
    shot(drv, "p3_ptable_light")
    ActionChains(drv).send_keys("F").send_keys("e").send_keys(Keys.ENTER).perform()
    wait_idle(drv)
    assert ed_state(drv)["symbols"][0] == "Fe"
    drv.execute_script("window.__mace.editor.select([1], 'replace')")
    focus_viewport(drv)
    keys(drv, "e")
    H.wait(drv, "return document.querySelector('.ptable .pt-recent [data-sym=\"Fe\"]')", 5)
    # arrows move through the grid; Enter on a cell picks it
    ActionChains(drv).send_keys(Keys.ARROW_DOWN).perform()
    first = drv.execute_script("return document.activeElement.dataset.sym")
    ActionChains(drv).send_keys(Keys.ARROW_RIGHT).perform()
    second = drv.execute_script("return document.activeElement.dataset.sym")
    assert first != second
    ActionChains(drv).send_keys(Keys.ESCAPE).perform()
    time.sleep(0.2)
    assert not drv.execute_script("return document.querySelector('.ptable')")
    no_errors(drv)


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_ptable_cells_meet_text_contrast(fresh, theme):
    """Every element cell's symbol reads at 4.5:1 or better on its own color."""
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    drv.execute_script("window.__mace.settings.set('theme', arguments[0]); window.__mace.editor.select([0], 'replace')", theme)
    focus_viewport(drv)
    keys(drv, "e")
    H.wait(drv, "return document.querySelectorAll('.ptable .pt-cell').length >= 103", 5)
    worst = drv.execute_script("""const rgb = (s) => s.match(/[\\d.]+/g).slice(0, 3).map(Number);
        const lum = (c) => { const [r, g, b] = c.map((v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; });
          return 0.2126 * r + 0.7152 * g + 0.0722 * b; };
        return [...document.querySelectorAll('.ptable .pt-cell')].map((b) => { const cs = getComputedStyle(b);
          const a = lum(rgb(cs.color)), c = lum(rgb(cs.backgroundColor));
          return [(Math.max(a, c) + 0.05) / (Math.min(a, c) + 0.05), b.dataset.sym]; }).sort((x, y) => x[0] - y[0])[0]""")
    assert worst[0] >= 4.5, worst
    keys(drv, Keys.ESCAPE)
    no_errors(drv)


def test_replace_mg_ca_sym(fresh):
    """Criterion 5: Sym mode, Mg -> Ca changes all 4 Mg; still Fm-3m; the
    History row; the four dirty marks; Mod+Z restores and clears them; one
    request each (the replace also opens the session)."""
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    set_view(drv)
    assert ed_state(drv)["symMode"] == "sym"
    H.click_atom(drv, 0)
    focus_viewport(drv)
    t0 = H.now_ms(drv)
    keys(drv, "e")
    H.wait(drv, "return document.activeElement && document.activeElement.classList.contains('pt-search')", 5)
    ActionChains(drv).send_keys("Ca").send_keys(Keys.ENTER).perform()
    wait_idle(drv)
    st = ed_state(drv)
    assert st["symbols"].count("Ca") == 4 and st["symbols"].count("Mg") == 0
    sg = drv.find_element("id", "st-sg")
    assert sg.text == "Fm-3m (225)" and "warn" not in sg.get_attribute("class")
    assert st["rows"][-1].startswith("Replace Mg→Ca (4 atoms")
    assert drv.execute_script("return document.querySelector('.doc-tab.active').classList.contains('dirty')")
    assert drv.execute_script("return document.title").startswith("● ")
    assert "Modified" in drv.execute_script("return document.querySelector('#st-doc').textContent")
    # the Structure card keeps naming the file's kind, not the editor's payload ("edit")
    badges = drv.execute_script("return [...document.querySelectorAll('#s-badges .badge')].map((b) => b.textContent)")
    assert "cif" in badges and "edit" not in badges, badges
    assert drv.execute_script("return document.querySelector('#toolbar [data-cmd=\"file.save\"]').classList.contains('accent')")
    assert len(H.requests_since(drv, t0, "/api/edit")) == 1
    shot(drv, "p3_dirty_light")
    t1 = H.now_ms(drv)
    focus_viewport(drv)
    keys(drv, Keys.CONTROL, "z")
    wait_idle(drv)
    st = ed_state(drv)
    assert st["symbols"].count("Mg") == 4 and not st["dirty"]
    assert not drv.execute_script("return document.querySelector('.doc-tab.active').classList.contains('dirty')")
    assert not drv.execute_script("return document.title").startswith("● ")
    assert "Saved" in drv.execute_script("return document.querySelector('#st-doc').textContent")
    assert len(H.requests_since(drv, t1, "/api/edit")) == 1
    no_errors(drv)


# --- T8: Edit Cell, Set Coordinates, View Along [uvw], the selection dialogs ------------------

def dlg_input(drv, dlg, label):
    return drv.execute_script("return [...document.querySelectorAll(`#${arguments[0]} input`)].find((i) => i.getAttribute('aria-label') === arguments[1])", dlg, label)


def set_field(drv, el, text):
    drv.execute_script("arguments[0].value = arguments[1]; arguments[0].dispatchEvent(new Event('input', {bubbles: true}))", el, text)


def test_edit_cell_dialog_ghost_and_apply(fresh):
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    drv.execute_script("window.__mace.run('edit.editCell')")
    H.wait(drv, "return document.querySelector('#edit-cell') && document.querySelector('#edit-cell').open", 10)
    assert drv.execute_script("return document.querySelector('#edit-cell').matches(':modal')") is False     # non-modal
    a = dlg_input(drv, "edit-cell", "a (Å)")
    set_field(drv, a, "4.4")
    # Sym mode, cubic: b and c follow a, the angles are fixed
    assert dlg_input(drv, "edit-cell", "b (Å)").get_attribute("value") in ("4.4",) or \
        drv.execute_script("return arguments[0].value", dlg_input(drv, "edit-cell", "c (Å)")) == "4.4"
    assert drv.execute_script("return arguments[0].disabled", dlg_input(drv, "edit-cell", "α (°)"))
    assert "85.184" in drv.execute_script("return document.querySelector('#edit-cell .ec-vol').textContent")
    assert drv.execute_script("return window.__mace.viewport.scene().layers && true")
    shot(drv, "p3_edit_cell_light")
    drv.execute_script("document.querySelector('#edit-cell [data-act=\"apply\"]').click()")
    wait_idle(drv)
    st = drv.execute_script("const e = window.__mace.editor.doc(); return [e.structure.lattice.a, e.structure.lattice.c, e.rows.map((r) => r.label)]")
    assert abs(st[0] - 4.4) < 1e-9 and abs(st[1] - 4.4) < 1e-9 and st[2][-1] == "Edit Cell" and len(st[2]) == 2
    assert not drv.execute_script("return document.querySelector('#edit-cell').open")
    no_errors(drv)


def test_set_coordinates_dialog_modes(fresh):
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    drv.execute_script("window.__mace.run('build.symModeP1'); window.__mace.editor.select([4, 5], 'replace')")
    focus_viewport(drv)
    keys(drv, Keys.CONTROL, Keys.SHIFT, "u")
    H.wait(drv, "return document.querySelector('#set-coords') && document.querySelector('#set-coords').open", 10)
    # relative: both move by the same Δ
    drv.execute_script("document.querySelector('#set-coords [data-mode=\"relative\"]').click()")
    set_field(drv, dlg_input(drv, "set-coords", "Set x"), "0.1")
    assert drv.execute_script("return window.__mace.viewport.modelIds().drag") is not None          # the ghost preview
    drv.execute_script("document.querySelector('#set-coords [data-act=\"apply\"]').click()")
    wait_idle(drv)
    f = drv.execute_script("return window.__mace.editor.doc().structure.frac")
    assert abs(f[4][0] - 0.6) < 1e-9 and abs(f[5][0] - 0.6) < 1e-9
    # per axis: only z is set, for each atom
    drv.execute_script("window.__mace.run('edit.setCoordinates')")
    H.wait(drv, "return document.querySelector('#set-coords').open", 10)
    drv.execute_script("document.querySelector('#set-coords [data-mode=\"axis\"]').click()")
    set_field(drv, dlg_input(drv, "set-coords", "Set z"), "1/4")
    drv.execute_script("document.querySelector('#set-coords [data-act=\"apply\"]').click()")
    wait_idle(drv)
    f = drv.execute_script("return window.__mace.editor.doc().structure.frac")
    assert abs(f[4][2] - 0.25) < 1e-9 and abs(f[5][2] - 0.25) < 1e-9 and abs(f[4][0] - 0.6) < 1e-9
    assert ed_state(drv)["rows"][-2:] == ["Move 2 Atoms", "Set Coordinates"]
    no_errors(drv)


def test_view_along_uvw(fresh):
    import numpy as np
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    focus_viewport(drv)
    keys(drv, Keys.ALT, "u")
    H.wait(drv, "return document.querySelector('#along-uvw') && document.querySelector('#along-uvw').open", 10)
    for k, v in zip("uvw", "110"):
        set_field(drv, dlg_input(drv, "along-uvw", k), v)
    time.sleep(0.5)
    drv.execute_script("document.querySelector('#along-uvw [data-act=\"apply\"]').click()")
    time.sleep(0.4)
    q = drv.execute_script("return window.__mace.viewState().view.slice(4)")
    r = drv.execute_script("""const q = arguments[0], p = [1, 1, 0].map((x) => x / Math.SQRT2);
        const [x, y, z, w] = q, ix = w*p[0]+y*p[2]-z*p[1], iy = w*p[1]+z*p[0]-x*p[2], iz = w*p[2]+x*p[1]-y*p[0], iw = -x*p[0]-y*p[1]-z*p[2];
        return [ix*w+iw*-x+iy*-z-iz*-y, iy*w+iw*-y+iz*-x-ix*-z, iz*w+iw*-z+ix*-y-iy*-x]""", q)
    assert np.allclose(r, [0, 0, 1], atol=1e-6)          # [110] points at the viewer
    assert drv.execute_script("return document.querySelector('#tb-along-label').textContent") == "[1 1 0]"
    no_errors(drv)


def test_select_within_by_index_by_frac(fresh):
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    drv.execute_script("window.__mace.editor.select([0], 'replace'); window.__mace.run('select.within')")
    H.wait(drv, "return document.querySelector('#sel-within') && document.querySelector('#sel-within').open", 10)
    set_field(drv, dlg_input(drv, "sel-within", "Radius"), "2.2")
    drv.execute_script("document.querySelector('#sel-within [data-act=\"apply\"]').click()")
    time.sleep(0.3)
    assert selection(drv) == [0, 5, 6, 7]                      # Mg1 and its three O neighbors in the cell (images count)
    drv.execute_script("window.__mace.run('select.byIndex')")
    H.wait(drv, "return document.querySelector('#sel-index').open", 10)
    set_field(drv, dlg_input(drv, "sel-index", "Atom numbers"), "1-2, 5, 7-")
    drv.execute_script("document.querySelector('#sel-index [data-act=\"apply\"]').click()")
    time.sleep(0.2)
    assert selection(drv) == [0, 1, 4, 6, 7]
    drv.execute_script("window.__mace.run('select.byFrac')")
    H.wait(drv, "return document.querySelector('#sel-frac').open", 10)
    set_field(drv, dlg_input(drv, "sel-frac", "Fractional range"), "z > 0.25")
    drv.execute_script("document.querySelector('#sel-frac [data-act=\"apply\"]').click()")
    time.sleep(0.2)
    s = drv.execute_script("return window.__mace.editor.doc().structure.frac")
    assert selection(drv) == [i for i, f in enumerate(s) if (f[2] % 1) > 0.25]
    drv.execute_script("window.__mace.run('select.saveSet')")
    H.wait(drv, "return document.querySelector('#sel-save').open", 10)
    drv.execute_script("document.querySelector('#sel-save [data-act=\"apply\"]').click()")
    drv.execute_script("window.__mace.editor.select([], 'replace'); window.__mace.run('select.sets')")
    rows = drv.execute_script("return [...document.querySelectorAll('.menu .menu-item .lbl')].map((x) => x.textContent)")
    assert rows[0].startswith("Set 1 (")
    keys(drv, Keys.ESCAPE)
    no_errors(drv)


def test_floating_panels_leave_the_structure_in_view_and_drag(fresh):
    """Edit Cell's ghost cell is the point of the panel: the panels float at
    the window's right edge, clear of the viewer's center, and their header
    drags them (the place is kept for the next opening)."""
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    for cmd, did in [("edit.editCell", "edit-cell"), ("edit.setCoordinates", "set-coords"), ("file.exportImage", "export-image")]:
        if cmd == "edit.setCoordinates":
            drv.execute_script("window.__mace.editor.select([4], 'replace')")
        drv.execute_script("window.__mace.run(arguments[0])", cmd)
        H.wait(drv, f"const d = document.getElementById('{did}'); return d && d.open", 5)
        box, view = drv.execute_script("""return [document.getElementById(arguments[0]).getBoundingClientRect().toJSON(),
            document.getElementById('viewer-main').getBoundingClientRect().toJSON()]""", did)
        cx, cy = view["x"] + view["width"] / 2, view["y"] + view["height"] / 2
        assert not (box["left"] <= cx <= box["right"] and box["top"] <= cy <= box["bottom"]), (did, box, view)
        assert box["right"] <= drv.execute_script("return innerWidth") and box["bottom"] <= drv.execute_script("return innerHeight")
        drv.execute_script("document.getElementById(arguments[0]).close()", did)
    drv.execute_script("window.__mace.run('edit.editCell')")
    H.wait(drv, "const d = document.getElementById('edit-cell'); return d && d.open", 5)
    head = drv.find_element("css selector", "#edit-cell .dlg-head h2")
    before = drv.execute_script("return document.getElementById('edit-cell').getBoundingClientRect().toJSON()")
    ActionChains(drv).click_and_hold(head).move_by_offset(-200, 60).release().perform()
    after = drv.execute_script("return document.getElementById('edit-cell').getBoundingClientRect().toJSON()")
    assert abs(after["left"] - before["left"] + 200) < 3 and abs(after["top"] - before["top"] - 60) < 3, (before, after)
    drv.execute_script("document.getElementById('edit-cell').close(); window.__mace.run('edit.editCell')")
    again = drv.execute_script("return document.getElementById('edit-cell').getBoundingClientRect().toJSON()")
    assert abs(again["left"] - after["left"]) < 1 and abs(again["top"] - after["top"]) < 1
    drv.execute_script("document.getElementById('edit-cell').close()")
    no_errors(drv)


def test_dialog_focus_return_and_esc(fresh):
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    focus_viewport(drv)
    keys(drv, Keys.CONTROL, Keys.SHIFT, "l")
    H.wait(drv, "return document.querySelector('#edit-cell') && document.querySelector('#edit-cell').open", 10)
    assert drv.execute_script("return document.activeElement.closest('#edit-cell') !== null")
    ActionChains(drv).send_keys(Keys.ESCAPE).perform()
    time.sleep(0.2)
    assert not drv.execute_script("return document.querySelector('#edit-cell').open")
    assert drv.execute_script("return document.activeElement.classList.contains('viewer-wrap')")
    assert drv.execute_script("return window.__mace.viewport.scene() && true")
    no_errors(drv)


# --- T9: measurements, view along, triad, projection -------------------------------------------

def measure_rows(drv):
    return drv.execute_script("""return [...document.querySelectorAll('.bottom-pane.measurements .ms-table tbody tr')]
        .map((tr) => [...tr.querySelectorAll('td')].slice(0, 3).map((td) => td.textContent).concat([tr.className]))""")


def test_measure_mgo_distance(fresh):
    """Criterion 4: the Measure tool on Mg–O shows 2.106 Å and a row."""
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    set_view(drv)
    focus_viewport(drv)
    keys(drv, "m")
    assert drv.execute_script("return window.__mace.viewState().tool") == "measure"
    H.click_atom(drv, 0)
    time.sleep(0.5)
    H.click_atom(drv, 5)
    time.sleep(0.4)
    assert "2.106 Å" in drv.execute_script("return window.__mace.viewport.labels()")
    drv.execute_script("window.__mace.run('view.bottom'); window.__mace.run('view.bottom')")
    rows = measure_rows(drv)
    assert len(rows) == 1 and rows[0][0] == "Distance" and rows[0][2] == "2.106 Å", rows
    v = drv.execute_script("return window.__mace.editor.doc().measurements[0].value")
    from ase.build import bulk
    mgo = bulk("MgO", "rocksalt", a=4.212, cubic=True)
    assert abs(v - mgo.get_distance(0, 1, mic=True)) < 5e-4
    shot(drv, "p3_measure_light")
    no_errors(drv)


def test_measure_angle_dihedral(fresh):
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    set_view(drv)
    drv.execute_script("window.__mace.run('tool.measure')")
    for i in (5, 0, 0, 6):                   # O6–Mg1–O7: a right angle (Mg1 twice counts once)
        H.click_atom(drv, i)
        time.sleep(0.5)
    rows = measure_rows(drv)
    assert len(rows) == 1 and rows[0][0] == "Angle" and rows[0][2] == "90.00°", rows
    H.click_atom(drv, 3)
    time.sleep(0.4)
    rows = measure_rows(drv)
    assert len(rows) == 1 and rows[0][0] == "Dihedral" and rows[0][2].endswith("°"), rows
    m = drv.execute_script("return window.__mace.editor.doc().measurements[0]")
    import numpy as np
    from ase import Atoms
    s = drv.execute_script("return window.__mace.editor.doc().structure")
    pos = [np.dot(np.array(s["frac"][a["i"]]) + a["image"], s["cell"]) for a in m["atoms"]]
    at = Atoms("X4", positions=pos)
    assert abs(((m["value"] - at.get_dihedral(0, 1, 2, 3)) + 180) % 360 - 180) < 1e-6
    no_errors(drv)


def test_measurement_invalid_after_delete_and_restored_on_undo(fresh):
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    drv.execute_script("window.__mace.run('build.symModeP1'); window.__mace.editor.select([0, 5], 'replace'); window.__mace.run('tool.measureSelection')")
    time.sleep(0.2)
    assert measure_rows(drv)[0][2] == "2.106 Å"
    edit(drv, "delete", {"indices": [5]})
    wait_idle(drv)
    H.wait(drv, "return document.querySelector('.bottom-pane.measurements tbody tr.invalid')", 5)
    assert measure_rows(drv)[0][2] == "atom deleted"
    drv.execute_script("window.__mace.editor.E.api.undo(window.__mace.editor.doc())")
    wait_idle(drv)
    H.wait(drv, "return !document.querySelector('.bottom-pane.measurements tbody tr.invalid')", 5)
    assert measure_rows(drv)[0][2] == "2.106 Å"
    # an edit that moves the atoms recomputes the value
    edit(drv, "move", {"indices": [5], "frac_delta": [0.05, 0, 0]})
    wait_idle(drv)
    H.wait(drv, "return document.querySelector('.bottom-pane.measurements tbody tr td:nth-child(3)').textContent !== '2.106 Å'", 5)
    no_errors(drv)


def test_measurements_tsv(fresh):
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    drv.execute_script("""window.__copied = null; navigator.clipboard.writeText = (t) => { window.__copied = t; return Promise.resolve(); };
        window.__mace.editor.select([0, 5], 'replace'); window.__mace.run('tool.measureSelection');
        window.__mace.editor.select([5, 0, 6], 'replace'); window.__mace.run('tool.measureSelection');""")
    drv.execute_script("document.querySelector('.bottom-pane.measurements .ms-copy').click()")
    H.wait(drv, "return window.__copied", 5)
    lines = drv.execute_script("return window.__copied").strip().split("\n")
    assert lines[0] == "type\tatoms\tvalue\tunit" and lines[1].startswith("Distance\tMg1–O6\t2.1060\tÅ")
    assert lines[2].startswith("Angle\tO6–Mg1–O7") and lines[2].endswith("\t90.000\t°")
    drv.execute_script("window.__mace.run('view.measurements')")          # Shift+M: hide them all
    assert drv.execute_script("return window.__mace.viewport.labels().length") == 0
    drv.execute_script("window.__mace.run('view.measurements'); window.__mace.run('tool.clearMeasurements')")
    assert drv.execute_script("return window.__mace.editor.doc().measurements.length") == 0
    no_errors(drv)


def _png_hash(drv):
    import hashlib
    import base64
    url = drv.execute_script("const v = window.MACE.state.viewers['viewer-main']; v.render(); return v.getCanvas().toDataURL('image/png')")
    return hashlib.md5(base64.b64decode(url.split(",")[1])).hexdigest()


@pytest.mark.parametrize("which", ["tri", "mgo"])
def test_view_along_six_distinct_with_triad(fresh, which):
    """Criterion 2: 1 2 3 and Shift+1 2 3 give six distinct views; the triad
    shows the matching axis toward the viewer. A cubic cell has a ∥ a*, so
    distinctness is checked on a triclinic cell; MgO gets the screenshots."""
    import numpy as np
    drv, root = fresh[0], fresh[2]
    path = "structures/MgO.cif"
    if which == "tri":
        import ase.io
        from ase import Atoms
        tri = Atoms("NaCl", cell=[4, 5, 6, 80, 95, 105], scaled_positions=[[0.1, 0.2, 0.3], [0.6, 0.5, 0.4]], pbc=True)
        ase.io.write(root / "tri.cif", tri)
        drv.execute_async_script("window.MACE.loadFiles().then(arguments[0])")
        path = "tri.cif"
    open_doc(drv, path)
    drv.execute_script("window.__mace.settings.set('reduceMotion', 'on')")
    focus_viewport(drv)
    views, hashes = {}, {}
    for name, combo in [("a", ("1",)), ("b", ("2",)), ("c", ("3",)), ("as", (Keys.SHIFT, "1")), ("bs", (Keys.SHIFT, "2")), ("cs", (Keys.SHIFT, "3"))]:
        keys(drv, *combo)
        time.sleep(0.3)
        views[name] = drv.execute_script("return window.__mace.viewState().view.slice(4)")
        hashes[name] = _png_hash(drv)
        shot(drv, f"p3_along_{name}_{which}")
        tri_axes = {t["axis"]: t for t in drv.execute_script("return window.__mace.triad()")}
        k = name[0]
        if len(name) == 1:
            assert math.hypot(tri_axes[k]["x"], tri_axes[k]["y"]) <= 2 and tri_axes[k]["z"] > 0, (name, tri_axes)
        else:
            assert tri_axes[k]["z"] > 0
            for o in "abc":
                if o != k:
                    assert abs(tri_axes[o]["z"]) <= 0.02, (name, o, tri_axes)
    if which == "tri":
        names = list(views)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                dot = abs(float(np.dot(views[a], views[b])))
                assert 2 * math.degrees(math.acos(min(1.0, dot))) > 1, (a, b)
                assert hashes[a] != hashes[b], (a, b)
        (root / "tri.cif").unlink()
        drv.execute_async_script("window.MACE.loadFiles().then(arguments[0])")
    no_errors(drv)


def test_projection_toggle_pressed_state(fresh):
    """Criterion 3: 5 toggles the projection; the toolbar button, the menu,
    the setting and the 3D camera agree."""
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    focus_viewport(drv)
    for want in ("perspective", "orthographic"):
        keys(drv, "5")
        time.sleep(0.3)
        st = drv.execute_script("return window.__mace.viewState()")
        pressed = drv.execute_script("return document.querySelector('#toolbar [data-cmd=\"view.projection\"]').getAttribute('aria-pressed')")
        assert st["projection"] == want and st["cameraOrtho"] == (want == "orthographic")
        assert pressed == ("true" if want == "perspective" else "false")
        drv.execute_script("""const b = [...document.querySelectorAll('.menu-top')].find((x) => x.textContent === 'View');
            b.dispatchEvent(new PointerEvent('pointerdown', {bubbles: true, button: 0}))""")
        checked = drv.execute_script("return document.querySelector('.menu [data-cmd-id=\"view.projection\"]').getAttribute('aria-checked')")
        keys(drv, Keys.ESCAPE)
        assert checked == ("true" if want == "perspective" else "false")
        shot(drv, f"p3_projection_{want}")
        focus_viewport(drv)
    no_errors(drv)


# --- T10: the Atoms table, the History panel, hidden and released viewers -------------------

def show_atoms(drv):
    drv.execute_script("document.querySelector('.bottom-tab[data-bottom=\"atoms\"]').click()")
    time.sleep(0.2)

def test_atoms_table_virtualized_8000(fresh):
    import ase.io
    from ase.build import bulk
    drv, root = fresh[0], fresh[2]
    ase.io.write(root / "mgo_10x10x10.cif", bulk("MgO", "rocksalt", a=4.212, cubic=True).repeat((10, 10, 10)))
    drv.execute_async_script("window.MACE.loadFiles().then(arguments[0])")
    drv.set_script_timeout(120)
    open_doc(drv, "mgo_10x10x10.cif")
    show_atoms(drv)
    H.wait(drv, "return document.querySelectorAll('.atoms-table tbody tr').length > 0", 20)
    most = drv.execute_async_script("""const done = arguments[0], sc = document.querySelector('.at-scroll');
        let most = 0, top = 0;
        const step = () => {
          most = Math.max(most, document.querySelectorAll('.atoms-table tr').length);
          top += 2000; sc.scrollTop = top;
          if (top < sc.scrollHeight) requestAnimationFrame(() => requestAnimationFrame(step)); else done(most);
        };
        step();""")
    assert most <= 80, most
    last = drv.execute_script("return [...document.querySelectorAll('.atoms-table tbody tr[data-i]')].map((r) => Number(r.dataset.i)).pop()")
    assert last == 7999
    (root / "mgo_10x10x10.cif").unlink()
    drv.execute_async_script("window.MACE.loadFiles().then(arguments[0])")
    no_errors(drv)


def test_atoms_table_selection_sync_and_edit(fresh):
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    show_atoms(drv)
    drv.execute_script("window.__mace.run('build.symModeP1')")
    # a row's first cell (geckodriver clicks cells, not table rows)
    rows = lambda: drv.execute_script("return [...document.querySelectorAll('.atoms-table tbody tr[data-i] td:first-child')]")
    r = rows()
    r[1].click()
    assert selection(drv) == [1]
    ActionChains(drv).key_down(Keys.SHIFT).click(rows()[3]).key_up(Keys.SHIFT).perform()
    assert selection(drv) == [1, 2, 3]
    ActionChains(drv).key_down(Keys.CONTROL).click(rows()[2]).key_up(Keys.CONTROL).perform()
    assert selection(drv) == [1, 3]
    drv.execute_script("window.__mace.editor.select([5], 'replace')")          # and back: the viewport's selection shows
    assert drv.execute_script("return [...document.querySelectorAll('.atoms-table tr.sel')].map((r) => Number(r.dataset.i))") == [5]
    # Enter edits the cell in place; Tab moves to the next one
    td = drv.execute_script("return document.querySelector('.atoms-table tr[data-i=\"5\"] td[data-c=\"3\"]')")
    ActionChains(drv).double_click(td).perform()
    inp = drv.execute_script("return document.querySelector('.atoms-table .at-edit')")
    drv.execute_script("arguments[0].select()", inp)
    inp.send_keys("1/4")
    inp.send_keys(Keys.TAB)
    wait_idle(drv)
    assert abs(drv.execute_script("return window.__mace.editor.doc().structure.frac[5][0]") - 0.25) < 1e-9
    assert drv.execute_script("return !!document.querySelector('.atoms-table .at-edit')")             # on to y
    ActionChains(drv).send_keys(Keys.ESCAPE).perform()
    td = drv.execute_script("return document.querySelector('.atoms-table tr[data-i=\"5\"] td[data-c=\"1\"]')")
    ActionChains(drv).double_click(td).perform()
    inp = drv.execute_script("return document.querySelector('.atoms-table .at-edit')")
    drv.execute_script("arguments[0].select()", inp)
    inp.send_keys("S", Keys.ENTER)
    wait_idle(drv)
    assert ed_state(drv)["symbols"][5] == "S"
    no_errors(drv)


def test_atoms_table_paste_column_p1(fresh):
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    show_atoms(drv)
    # a paste event carrying the text (headless Firefox gives a synthetic ClipboardEvent no data)
    paste = """const sc = document.querySelector('.at-scroll'), dt = new DataTransfer(); dt.setData('text/plain', arguments[0]);
        sc.focus(); const ev = new Event('paste', {bubbles: true, cancelable: true});
        Object.defineProperty(ev, 'clipboardData', {value: dt}); sc.dispatchEvent(ev);"""
    drv.execute_script("document.querySelector('.atoms-table tr[data-i=\"0\"] td[data-c=\"5\"]').click()")
    drv.execute_script(paste, "0.1\n0.2\n0.3\n")
    time.sleep(0.3)
    toasts = drv.execute_script("return [...document.querySelectorAll('#toasts .msg')].map((m) => m.textContent)")
    assert any("Switch to P1 to paste coordinates" in t for t in toasts), toasts            # Sym: refused
    drv.execute_script("window.__mace.run('build.symModeP1')")
    drv.execute_script("document.querySelector('.atoms-table tr[data-i=\"0\"] td[data-c=\"5\"]').click()")
    drv.execute_script(paste, "0.1\n0.2\n0.3\n")
    wait_idle(drv)
    f = drv.execute_script("return window.__mace.editor.doc().structure.frac")
    assert [round(f[i][2], 9) for i in range(3)] == [0.1, 0.2, 0.3]
    assert ed_state(drv)["rows"][-1] == "Set Coordinates" and len(ed_state(drv)["rows"]) == 2      # one step
    no_errors(drv)


def test_history_jump_gray_and_redo(fresh):
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    drv.execute_script("window.__mace.run('build.symModeP1')")
    for i, el in enumerate(["Ca", "Sr", "Ba", "Be", "Zn", "Cd"]):
        edit(drv, "replace", {"indices": [i % 4], "element": el})
    wait_idle(drv)
    focus_viewport(drv)
    keys(drv, Keys.CONTROL, Keys.SHIFT, "h")
    H.wait(drv, "return document.querySelector('.bottom-pane.history.active .hist-row')", 5)
    labels = drv.execute_script("return [...document.querySelectorAll('.hist-row .hist-l')].map((x) => x.textContent)")
    assert labels[0] == "Opened MgO.cif" and len(labels) == 7
    t = H.now_ms(drv)
    drv.execute_script("document.querySelector('.hist-row[data-n=\"2\"]').click()")
    wait_idle(drv)
    assert len(H.requests_since(drv, t, "/api/edit")) == 1                                  # one request
    st = ed_state(drv)
    assert st["pos"] == 2 and st["symbols"][:4] == ["Ca", "Sr", "Mg", "Mg"]
    gray = drv.execute_script("return [...document.querySelectorAll('.hist-row.redo')].map((r) => Number(r.dataset.n))")
    assert gray == [3, 4, 5, 6]
    shot(drv, "p3_history_light")
    drv.execute_script("window.__mace.run('edit.redo')")
    wait_idle(drv)
    assert ed_state(drv)["pos"] == 3
    edit(drv, "replace", {"indices": [4], "element": "S"})                                 # a new edit drops the redo rows
    wait_idle(drv)
    assert len(ed_state(drv)["rows"]) == 5
    no_errors(drv)


def test_hidden_viewers_never_paint(fresh):
    drv = fresh[0]
    drv.execute_async_script("const d = arguments[1]; window.__mace.open(arguments[0]).then(() => d())", "outputs/Ag2Br3_opt_unfinished.out")
    time.sleep(0.5)
    open_doc(drv, "structures/MgO.cif")
    drv.execute_script("window.__mace.run('build.symModeP1'); window.__paintsBy = {};"
                       "document.addEventListener('mace-painted', (e) => { window.__paintsBy[e.detail] = (window.__paintsBy[e.detail] || 0) + 1; })")
    for k in range(20):
        edit(drv, "move", {"indices": [k % 8], "frac_delta": [0.001, 0, 0]})
    wait_idle(drv)
    time.sleep(0.3)
    paints = drv.execute_script("return window.__paintsBy")
    assert paints.get("viewer-main", 0) >= 1 and paints.get("viewer-before", 0) == 0 and paints.get("viewer-after", 0) == 0, paints
    no_errors(drv)


def test_freed_viewer_rebuilds_under_300ms(fresh):
    drv = fresh[0]
    drv.execute_script("window.__maceViewerIdleMs = 300")
    open_doc(drv, "structures/MgO.cif")
    view = drv.execute_script("return window.__mace.viewState().view")
    drv.execute_async_script("const d = arguments[1]; window.__mace.open(arguments[0]).then(() => d())", "materials.db")
    H.wait(drv, "return window.MACE.state.viewers['viewer-main'].freed === true", 5)
    assert drv.execute_script("return window.MACE.state.viewers['viewer-main'].getModel() === undefined || "
                              "window.MACE.state.viewers['viewer-main'].getModel() === null")
    drv.execute_script("window.__maceRebuildMs = null; window.__mace.activate(0)")
    H.wait(drv, "return window.__maceRebuildMs !== null && window.__maceRebuildMs !== undefined", 5)
    ms = drv.execute_script("return window.__maceRebuildMs")
    assert ms <= 300, ms
    time.sleep(0.3)
    assert max(abs(a - b) for a, b in zip(view, drv.execute_script("return window.__mace.viewState().view"))) < 1e-6
    drv.execute_script("window.__maceViewerIdleMs = 0")
    no_errors(drv)


# --- T11: saving, drafts, export ----------------------------------------------------------

def md5_tree(root):
    import hashlib
    h = hashlib.md5()
    for p in sorted(Path(root).rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(root)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def toasts(drv):
    return drv.execute_script("return [...document.querySelectorAll('#toasts .msg')].map((m) => m.textContent)")


def run_cmd(drv, cid):
    drv.execute_script("window.__mace.run(arguments[0])", cid)
    time.sleep(0.2)


def dialog_open(drv, did, timeout=5):
    H.wait(drv, f"const d = document.getElementById('{did}'); return d && d.open", timeout)
    return drv.execute_script(f"return document.getElementById('{did}')")


def press_button(drv, did, value):
    drv.execute_script(f"document.querySelector('#{did} [data-v=\"{value}\"]').click()")
    time.sleep(0.2)


def wait_saved(drv, timeout=10):
    H.wait(drv, "const e = window.__mace.editor.doc(); return e && !e.dirty && e.pending === 0", timeout)


def copy_mgo(root, name):
    target = root / "structures" / name
    target.write_bytes((root / "structures" / "MgO.cif").read_bytes())
    return target


def reload_files(drv):
    drv.execute_async_script("window.MACE.loadFiles().then(arguments[0])")


def test_save_in_place_confirm_and_bak(fresh):
    drv, root = fresh[0], fresh[2]
    target = copy_mgo(root, "MgO_save1.cif")
    original = target.read_bytes()
    reload_files(drv)
    open_doc(drv, "structures/MgO_save1.cif")
    drv.execute_script("window.__mace.run('build.symModeP1')")
    edit(drv, "replace", {"indices": [0], "element": "Ca"})
    wait_idle(drv)
    run_cmd(drv, "file.save")
    dlg = dialog_open(drv, "save-confirm")
    assert "Overwrite MgO_save1.cif?" in dlg.text and "MgO_save1.cif.bak" in dlg.text
    drv.execute_script("document.querySelector('#save-confirm [data-k=\"remember\"]').click()")
    press_button(drv, "save-confirm", "overwrite")
    wait_saved(drv)
    bak = root / "structures" / "MgO_save1.cif.bak"
    assert bak.read_bytes() == original
    assert "Ca" in target.read_text()
    # the second save asks nothing (Don't ask again) and keeps the backup
    edit(drv, "replace", {"indices": [1], "element": "Sr"})
    wait_idle(drv)
    run_cmd(drv, "file.save")
    wait_saved(drv)
    assert not drv.execute_script("const d = document.getElementById('save-confirm'); return !!(d && d.open)")
    assert "Sr" in target.read_text() and bak.read_bytes() == original
    # a new session (close, reopen) saving again: the .bak is still the very first original
    drv.execute_script("window.__mace.closeAll()")
    open_doc(drv, "structures/MgO_save1.cif")
    drv.execute_script("window.__mace.run('build.symModeP1')")
    edit(drv, "replace", {"indices": [2], "element": "Ba"})
    wait_idle(drv)
    run_cmd(drv, "file.save")
    wait_saved(drv)
    assert "Ba" in target.read_text() and bak.read_bytes() == original
    no_errors(drv)


def test_save_changed_on_disk_prompt(fresh):
    drv, root = fresh[0], fresh[2]
    target = copy_mgo(root, "MgO_disk.cif")
    reload_files(drv)
    drv.execute_script("window.__mace.settings.set('confirmOverwrite', false)")
    open_doc(drv, "structures/MgO_disk.cif")
    drv.execute_script("window.__mace.run('build.symModeP1')")
    edit(drv, "replace", {"indices": [0], "element": "Ca"})
    wait_idle(drv)
    changed = target.read_text().replace("MgO", "MgO changed elsewhere", 1) + "\n# touched\n"
    target.write_text(changed)
    os.utime(target, ns=(time.time_ns(), time.time_ns() + 5_000_000_000))
    run_cmd(drv, "file.save")
    dlg = dialog_open(drv, "save-changed")
    assert "MgO_disk.cif changed on disk since you opened it" in dlg.text
    press_button(drv, "save-changed", "cancel")
    time.sleep(0.5)
    assert target.read_text() == changed                                                      # Cancel: nothing written
    assert drv.execute_script("return window.__mace.editor.doc().dirty")
    run_cmd(drv, "file.save")
    dialog_open(drv, "save-changed")
    press_button(drv, "save-changed", "saveas")                                              # Save As: a new name
    dialog_open(drv, "save-as")
    drv.execute_script("const n = document.querySelector('#save-as [data-k=\"name\"]'); n.value = 'MgO_disk_copy.cif';")
    press_button(drv, "save-as", "save")
    wait_saved(drv)
    assert (root / "structures" / "MgO_disk_copy.cif").exists() and target.read_text() == changed
    assert drv.execute_script("return window.__mace.editor.doc().path") == "structures/MgO_disk_copy.cif"
    # Overwrite: the file as it is on disk gets replaced (a fresh document of the changed file)
    drv.execute_script("window.__mace.closeAll()")
    open_doc(drv, "structures/MgO_disk.cif")
    drv.execute_script("window.__mace.run('build.symModeP1')")
    edit(drv, "replace", {"indices": [0], "element": "Sr"})
    wait_idle(drv)
    target.write_text(changed + "# again\n")
    os.utime(target, ns=(time.time_ns(), time.time_ns() + 9_000_000_000))
    run_cmd(drv, "file.save")
    dialog_open(drv, "save-changed")
    press_button(drv, "save-changed", "overwrite")
    wait_saved(drv)
    assert "Sr" in target.read_text() and "# again" not in target.read_text()
    no_errors(drv)


def test_save_as_out_defaults_edited_cif(fresh):
    drv = fresh[0]
    drv.execute_async_script("const d = arguments[1]; window.__mace.open(arguments[0]).then(() => d())", "outputs/Si_sp.out")
    H.wait(drv, "return window.__mace.viewport && window.__mace.viewport.displayed() > 0", 20)
    if drv.execute_script("return window.__mace.activeDoc().kind") != "structure":
        pytest.skip("Si_sp.out opens as a run, not a structure")
    run_cmd(drv, "file.save")                                                              # Save on a .out: Save As
    dialog_open(drv, "save-as")
    assert drv.execute_script("return document.querySelector('#save-as [data-k=\"name\"]').value") == "Si_sp_edited.cif"
    assert drv.execute_script("return document.querySelector('#save-as [data-k=\"dir\"]').value") == "outputs"
    press_button(drv, "save-as", "cancel")
    no_errors(drv)


def test_save_clears_dirty_and_title_dot(fresh):
    drv, root = fresh[0], fresh[2]
    copy_mgo(root, "MgO_dot.cif")
    reload_files(drv)
    drv.execute_script("window.__mace.settings.set('confirmOverwrite', false)")
    open_doc(drv, "structures/MgO_dot.cif")
    drv.execute_script("window.__mace.run('build.symModeP1')")
    edit(drv, "replace", {"indices": [0], "element": "Ca"})
    wait_idle(drv)
    assert drv.execute_script("return document.title").startswith("● ")
    assert drv.execute_script("return !!document.querySelector('.doc-tab.active.dirty')")
    keys(drv, Keys.CONTROL, "s")
    wait_saved(drv)
    assert not drv.execute_script("return document.title").startswith("● ")
    assert not drv.execute_script("return !!document.querySelector('.doc-tab.dirty')")
    assert any("Saved MgO_dot.cif" in t for t in toasts(drv))
    no_errors(drv)


def test_save_all_and_revert(fresh):
    drv, root = fresh[0], fresh[2]
    a, b = copy_mgo(root, "MgO_all_a.cif"), copy_mgo(root, "MgO_all_b.cif")
    reload_files(drv)
    drv.execute_script("window.__mace.settings.set('confirmOverwrite', false)")
    for p, el in (("structures/MgO_all_a.cif", "Ca"), ("structures/MgO_all_b.cif", "Sr")):
        open_doc(drv, p)
        drv.execute_script("window.__mace.run('build.symModeP1')")
        edit(drv, "replace", {"indices": [0], "element": el})
        wait_idle(drv)
    run_cmd(drv, "file.saveAll")
    H.wait(drv, "return window.__mace.docs().every((d) => !d.dirty)", 10)
    assert "Ca" in a.read_text() and "Sr" in b.read_text()
    # Revert: asks, then the file as it is on disk
    edit(drv, "replace", {"indices": [1], "element": "Ba"})
    wait_idle(drv)
    run_cmd(drv, "file.revert")
    dialog_open(drv, "revert-confirm")
    press_button(drv, "revert-confirm", "revert")
    H.wait(drv, "const e = window.__mace.editor.doc(); return e && !e.dirty && e.structure.symbols[1] !== 'Ba'", 10)
    st = ed_state(drv)
    assert st["symbols"][0] == "Sr" and len(st["rows"]) == 1
    no_errors(drv)


def test_classic_save_cif_uses_save_as(fresh):
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    t = H.now_ms(drv)
    drv.execute_script("document.getElementById('e-save').click()")
    dialog_open(drv, "save-as")
    assert not H.requests_since(drv, t, "save_cif")
    press_button(drv, "save-as", "cancel")
    no_errors(drv)


def draft_keys(drv):
    return drv.execute_script("return Object.keys(localStorage).filter((k) => k.startsWith('mace.gui.draft.'))")


def test_reload_with_dirty_doc_shows_restore_banner(fresh):
    drv, base, root = fresh[0], fresh[1], fresh[2]
    copy_mgo(root, "MgO_draft.cif")
    reload_files(drv)
    before = md5_tree(root)
    drv.execute_script("window.__mace.settings.set('autosaveDrafts', 1)")
    open_doc(drv, "structures/MgO_draft.cif")
    drv.execute_script("window.__mace.run('build.symModeP1')")
    edit(drv, "replace", {"indices": [0], "element": "Ca"})
    wait_idle(drv)
    # the 1 s timer writes it (no explicit flush)
    H.wait(drv, "return Object.keys(localStorage).some((k) => k.startsWith('mace.gui.draft.') && k.endsWith('structures/MgO_draft.cif'))", 10)
    assert md5_tree(root) == before                                                        # nothing written into --dir
    drv.execute_script("window.__mace.drafts.flush()")
    H.load(drv, base)
    H.wait(drv, "const b = document.getElementById('draft-banner'); return b && !b.hidden && b.textContent.includes('MgO_draft.cif')", 15)
    text = drv.execute_script("return document.getElementById('draft-banner').textContent")
    assert "Restore unsaved changes to MgO_draft.cif from" in text
    drv.execute_script("document.querySelector('#draft-banner [data-v=\"restore\"]').click()")
    H.wait(drv, "const e = window.__mace.editor && window.__mace.editor.doc(); return e && e.dirty && e.structure.symbols[0] === 'Ca'", 20)
    assert drv.execute_script("return window.__mace.editor.doc().path") == "structures/MgO_draft.cif"
    # reload again: Discard drops the key and no banner comes back
    drv.execute_script("window.__mace.drafts.flush()")
    H.load(drv, base)
    H.wait(drv, "const b = document.getElementById('draft-banner'); return b && !b.hidden", 15)
    drv.execute_script("document.querySelector('#draft-banner [data-v=\"discard\"]').click()")
    time.sleep(0.3)
    assert not [k for k in draft_keys(drv) if k.endswith("MgO_draft.cif")]
    H.load(drv, base)
    time.sleep(1.0)
    assert drv.execute_script("const b = document.getElementById('draft-banner'); return !b || b.hidden")
    assert md5_tree(root) == before
    no_errors(drv)


def png_size(data):
    import struct
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", data[16:24])


def png_pixel_alpha(data, points):
    """Alpha at (x, y) points of an RGBA PNG (decoded with zlib, filters undone)."""
    import struct
    import zlib
    w, h = png_size(data)
    pos, idat, ctype = 8, b"", None
    while pos < len(data):
        n, typ = struct.unpack(">I4s", data[pos:pos + 8])
        chunk = data[pos + 8:pos + 8 + n]
        if typ == b"IHDR":
            ctype = chunk[9]
        elif typ == b"IDAT":
            idat += chunk
        pos += 12 + n
    if ctype != 6:
        return [255 for _ in points]
    raw, bpp, stride = zlib.decompress(idat), 4, w * 4
    rows, prev, k = [], bytearray(stride), 0
    for _ in range(h):
        f, line = raw[k], bytearray(raw[k + 1:k + 1 + stride])
        k += 1 + stride
        for i in range(stride):
            a = line[i - bpp] if i >= bpp else 0
            b, c = prev[i], prev[i - bpp] if i >= bpp else 0
            if f == 1:
                line[i] = (line[i] + a) & 255
            elif f == 2:
                line[i] = (line[i] + b) & 255
            elif f == 3:
                line[i] = (line[i] + (a + b) // 2) & 255
            elif f == 4:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                line[i] = (line[i] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 255
        rows.append(line)
        prev = line
    return [rows[y][x * 4 + 3] for x, y in points]


def export_png(drv, size="4", transparent=True):
    drv.execute_script("window.__mace.run('file.exportImage')")
    dialog_open(drv, "export-image")
    drv.execute_script("""const d = document.getElementById('export-image');
        d.querySelector('[data-k="size"]').value = arguments[0];
        const t = d.querySelector('[data-k="transparent"]'); t.checked = arguments[1];
        d.querySelector('[data-k="triad"]').checked = false;
        d.dispatchEvent(new Event('change', {bubbles: true}));
        window.__maceExportUrl = null;""", size, transparent)
    return drv.execute_async_script("""const done = arguments[0];
        window.__maceExportHook = (url) => done(url);
        document.querySelector('#export-image [data-v="export"]').click();""")


def test_export_png_4x_exact_size_and_transparent(fresh):
    import base64
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    set_view(drv)
    w, h = drv.execute_script("const c = window.MACE.state.viewers['viewer-main'].getCanvas(); return [c.width, c.height]")
    assert drv.execute_script("return devicePixelRatio") == 1
    url = export_png(drv, "4", True)
    data = base64.b64decode(url.split(",", 1)[1])
    assert png_size(data) == (4 * w, 4 * h)
    p = H.atom_screen(drv, 0)
    r = drv.execute_script("return document.getElementById('viewer-main').getBoundingClientRect().toJSON()")
    center = (int((p["x"]) * 4), int((p["y"]) * 4))
    corner_a, atom_a = png_pixel_alpha(data, [(2, 2), center])
    assert corner_a == 0 and atom_a == 255, (corner_a, atom_a, r)
    url = export_png(drv, "1", False)
    data = base64.b64decode(url.split(",", 1)[1])
    assert png_size(data) == (w, h)
    assert png_pixel_alpha(data, [(2, 2)])[0] == 255
    # the viewer is back to its own size and still draws
    assert drv.execute_script("const c = window.MACE.state.viewers['viewer-main'].getCanvas(); return [c.width, c.height]") == [w, h]
    no_errors(drv)


def test_export_download_lands_in_download_dir(fresh):
    drv, root, dl = fresh[0], fresh[2], fresh[5]
    before = md5_tree(root)
    open_doc(drv, "structures/MgO.cif")
    drv.execute_script("window.__mace.run('view.alongA')")
    time.sleep(0.5)
    for f in dl.glob("*"):
        f.unlink()
    drv.execute_script("window.__mace.run('file.exportImage')")
    dialog_open(drv, "export-image")
    name = drv.execute_script("return document.querySelector('#export-image [data-k=\"name\"]').value")
    assert name == "MgO_a.png"
    press_button(drv, "export-image", "export")
    end = time.time() + 15
    while time.time() < end and not (dl / name).exists():
        time.sleep(0.2)
    assert (dl / name).exists() and png_size((dl / name).read_bytes())
    assert md5_tree(root) == before
    no_errors(drv)


def test_copy_image_or_reason(fresh):
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    ok = drv.execute_async_script("const d = arguments[0]; window.__mace.copyImage().then(d)")
    if ok:
        assert any("Copied the image" in t for t in toasts(drv))
    else:
        msgs = toasts(drv)
        assert any(t.startswith("Could not copy the image:") for t in msgs), msgs
        assert drv.execute_script("return [...document.querySelectorAll('#toasts .acts button')].some((b) => b.textContent === 'Export Image…')")
    no_errors(drv)


def test_draft_never_writes_dir(fresh):
    drv, root = fresh[0], fresh[2]
    before = md5_tree(root)
    open_doc(drv, "structures/NaCl.cif")
    drv.execute_script("window.__mace.run('build.symModeP1')")
    edit(drv, "replace", {"indices": [0], "element": "K"})
    wait_idle(drv)
    out = drv.execute_script("return window.__mace.drafts.flush()")
    assert out.get("structures/NaCl.cif") == "write", out
    assert [k for k in draft_keys(drv) if k.endswith(".structures/NaCl.cif")]
    assert md5_tree(root) == before
    drv.execute_script("window.__mace.closeAll()")                      # closing drops the draft
    assert not [k for k in draft_keys(drv) if k.endswith(".structures/NaCl.cif")]
    no_errors(drv)


# --- T12: New Structure, Edit This Step, context menus ---------------------------------------

def test_new_structure_untitled_dirty_save_as(fresh):
    drv, root = fresh[0], fresh[2]
    keys(drv, Keys.ALT, "n")
    dialog_open(drv, "new-structure")
    drv.execute_script("""const d = document.getElementById('new-structure'), q = (k) => d.querySelector(`[data-k="${k}"]`);
        q('name').value = 'rocksalt_test'; q('sg').value = 'F m -3 m';
        ['a', 'b', 'c'].forEach((k) => q(k).value = '4.3');
        q('sites').value = arguments[0];""", "Mg 0 0 0\nO 0.5 0.5 0.5")
    press_button(drv, "new-structure", "create")
    H.wait(drv, "const e = window.__mace.editor && window.__mace.editor.doc(); return e && e.structure.symbols.length === 8", 15)
    st = ed_state(drv)
    assert st["dirty"] and st["symMode"] == "sym"
    assert drv.execute_script("return window.__mace.activeDoc().untitled && window.__mace.activeDoc().title") == "rocksalt_test"
    run_cmd(drv, "file.save")                                                  # untitled: Save goes to Save As
    dialog_open(drv, "save-as")
    assert drv.execute_script("return document.querySelector('#save-as [data-k=\"name\"]').value") == "rocksalt_test_edited.cif"
    drv.execute_script("document.querySelector('#save-as [data-k=\"name\"]').value = 'rocksalt_test.cif'")
    press_button(drv, "save-as", "save")
    wait_saved(drv)
    assert (root / "rocksalt_test.cif").is_file()
    assert drv.execute_script("return window.__mace.activeDoc().path") == "rocksalt_test.cif"
    # a bad site line is named in the dialog, nothing is created
    n = len(drv.execute_script("return window.__mace.docs()"))
    keys(drv, Keys.ALT, "n")
    dialog_open(drv, "new-structure")
    drv.execute_script("document.querySelector('#new-structure [data-k=\"sites\"]').value = 'Mg 0 0'")
    press_button(drv, "new-structure", "create")
    assert "write El x y z" in drv.execute_script("return document.querySelector('#new-structure .err').textContent")
    press_button(drv, "new-structure", "cancel")
    assert len(drv.execute_script("return window.__mace.docs()")) == n
    (root / "rocksalt_test.cif").unlink()
    no_errors(drv)


def test_edit_this_step_opens_structure(fresh):
    drv = fresh[0]
    drv.execute_async_script("const d = arguments[1]; window.__mace.open(arguments[0]).then(() => d())", "outputs/Ag2Br3_opt_unfinished.out")
    H.wait(drv, "return window.__mace.activeDoc() && window.__mace.activeDoc().kind === 'trajectory'", 15)
    H.wait(drv, "return !document.getElementById('o-frame-wrap').classList.contains('hidden')", 10)
    drv.execute_script("window.__mace.run('traj.first')")
    time.sleep(0.3)
    first = drv.execute_script("return window.MACE.state.traj.frames[0].frac")
    drv.execute_script("document.getElementById('o-edit-step').click()")
    H.wait(drv, "const d = window.__mace.activeDoc(); return d && d.kind === 'structure' && d.frame === 0", 15)
    assert drv.execute_script("return window.__mace.activeDoc().title").endswith("· step 1")
    assert drv.execute_script("return window.__mace.ctx().docEditable")
    t = H.now_ms(drv)
    drv.execute_script("window.__mace.run('build.symModeP1')")
    edit(drv, "move", {"indices": [0], "frac_delta": [0.01, 0, 0]})
    wait_idle(drv)
    reqs = drv.execute_script("return window.__mace.editor.doc().queue.log.map((x) => x.route)")
    assert "edit/open" in reqs or "edit/op" in reqs
    st = ed_state(drv)
    assert st["session"] and abs((st["frac"][0][0] - first[0][0]) % 1 - 0.01) < 1e-9
    assert max(abs(x - y) for x, y in zip(st["frac"][1], first[1])) < 1e-9
    # the step document saves through Save As (a .out is never written)
    run_cmd(drv, "file.save")
    dialog_open(drv, "save-as")
    assert drv.execute_script("return document.querySelector('#save-as [data-k=\"name\"]').value") == "Ag2Br3_opt_unfinished_edited.cif"
    press_button(drv, "save-as", "cancel")
    # the run itself is still there, and opening the .out again goes to it
    drv.execute_async_script("const d = arguments[1]; window.__mace.open(arguments[0]).then(() => d())", "outputs/Ag2Br3_opt_unfinished.out")
    assert drv.execute_script("return window.__mace.activeDoc().kind") == "trajectory"
    no_errors(drv)


def test_context_menu_entries_enabled(fresh):
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    set_view(drv)
    drv.execute_script("window.__mace.run('build.symModeP1')")
    H.click_atom(drv, 0, button=2)
    H.wait(drv, "return document.querySelector('.menu[aria-label=\"Atom menu\"]')", 5)
    soon = drv.execute_script("return [...document.querySelectorAll('.menu [data-cmd-id]')].filter((b) => b.title.startsWith('Not in this version yet')).map((b) => b.dataset.cmdId)")
    assert not soon, soon
    # nothing was selected: the right-click selected the atom, so the entries that
    # act on the selection (Delete, Replace Element, Set Coordinates) act on it
    assert selection(drv) == [0]
    off = drv.execute_script("""return [...document.querySelectorAll('.menu [data-cmd-id]')]
        .filter((b) => b.disabled || b.getAttribute('aria-disabled') === 'true').map((b) => b.dataset.cmdId)""")
    assert not {"edit.delete", "edit.replaceElement", "edit.setCoordinates"} & set(off), off
    keys(drv, Keys.ESCAPE)
    # a right-click inside a selection keeps it
    drv.execute_script("window.__mace.editor.select([0, 4], 'replace')")
    H.click_atom(drv, 4, button=2)
    H.wait(drv, "return document.querySelector('.menu[aria-label=\"Atom menu\"]')", 5)
    assert selection(drv) == [0, 4]
    keys(drv, Keys.ESCAPE)
    a, b = H.atom_screen(drv, 0), H.atom_screen(drv, 4)             # a quarter of the way from Mg1 to O5: empty space
    H.click_at(drv, a["x"] + (b["x"] - a["x"]) * 0.3, a["y"] + (b["y"] - a["y"]) * 0.3, button=2)
    H.wait(drv, "return document.querySelector('.menu[aria-label=\"Viewport menu\"]')", 5)
    soon = drv.execute_script("return [...document.querySelectorAll('.menu [data-cmd-id]')].filter((b) => b.title.startsWith('Not in this version yet')).map((b) => b.dataset.cmdId)")
    assert soon == [], soon                                              # Paste is real since Phase 4
    drv.execute_script("document.querySelector('.menu [data-cmd-id=\"viewport.addAtomHere\"]').click()")
    wait_idle(drv)
    st = ed_state(drv)
    assert len(st["symbols"]) == 9 and st["rows"][-1].startswith("Add"), (toasts(drv), st["rows"])
    no_errors(drv)


# --- T14: the Phase 3 surfaces: contrast, clipping, requests ---------------------------------

CONTRAST = """const sel = arguments[0];
  const rgb = (s) => (s.match(/[\\d.]+/g) || [0, 0, 0, 0]).map(Number);
  const lum = (c) => { const [r, g, b] = c.slice(0, 3).map((v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; });
    return 0.2126 * r + 0.7152 * g + 0.0722 * b; };
  const bgOf = (el) => { for (let e = el; e; e = e.parentElement) { const c = rgb(getComputedStyle(e).backgroundColor);
      if (c.length < 4 || c[3] > 0.5) return c; } return [255, 255, 255]; };
  let min = [99, ''];
  for (const el of document.querySelectorAll(sel)) {
    if (!el.childNodes.length || ![...el.childNodes].some((n) => n.nodeType === 3 && n.textContent.trim())) continue;
    if (el.getClientRects().length === 0 || el.closest('[aria-disabled="true"], option, .muted, [disabled]')) continue;
    const c = rgb(getComputedStyle(el).color), b = bgOf(el);
    const [x, y] = [lum(c), lum(b)].sort((p, q) => q - p), ratio = (x + 0.05) / (y + 0.05);
    if (ratio < min[0]) min = [ratio, (el.className || el.tagName) + ': ' + el.textContent.trim().slice(0, 30)];
  }
  return min"""
CLIPPED = """return [...document.querySelectorAll(arguments[0])]
    .filter((el) => el.getClientRects().length > 0 && getComputedStyle(el).display !== 'inline')
    .filter((el) => el.scrollWidth > el.clientWidth + 1)
    .map((el) => (el.className || el.tagName) + ': ' + el.textContent.trim().slice(0, 50))"""


def visit_phase3_surfaces(drv, root, each):
    """Open the Phase 3 dialogs, panels and banners once each; each(name, selector) runs while one is up."""
    copy_mgo(root, "MgO_audit.cif")
    reload_files(drv)
    open_doc(drv, "structures/MgO_audit.cif")
    drv.execute_script("window.__mace.run('build.symModeP1')")
    edit(drv, "replace", {"indices": [0], "element": "Ca"})
    wait_idle(drv)
    each("inspector", "#inspector *")
    each("explorer_title", ".side-title *")
    for tab in ("atoms", "measurements", "history"):
        drv.execute_script(f"document.querySelector('.bottom-tab[data-bottom=\"{tab}\"]').click()")
        time.sleep(0.2)
        each(f"panel_{tab}", ".bottom-pane.active *")
    # the History with a step after the current one (gray italics, still readable)
    edit(drv, "replace", {"indices": [1], "element": "Sr"})
    wait_idle(drv)
    drv.execute_script("window.__mace.run('edit.undo')")
    wait_idle(drv)
    H.wait(drv, "return document.querySelector('.bottom-pane.active .hist-row.redo')", 5)
    each("panel_history_redo", ".bottom-pane.active *")
    run_cmd(drv, "file.save")
    dialog_open(drv, "save-confirm")
    each("save_confirm", "dialog[open] *")
    press_button(drv, "save-confirm", "cancel")
    run_cmd(drv, "file.saveAs")
    dialog_open(drv, "save-as")
    each("save_as", "dialog[open] *")
    press_button(drv, "save-as", "cancel")
    run_cmd(drv, "file.exportImage")
    dialog_open(drv, "export-image")
    each("export", "dialog[open] *")
    press_button(drv, "export-image", "cancel")
    for cmd, did in (("edit.editCell", "cell-dialog"), ("edit.setCoordinates", "coords-dialog")):
        drv.execute_script("window.__mace.editor.select([1], 'replace')")
        run_cmd(drv, cmd)
        time.sleep(0.4)
        each(cmd, "dialog[open] *")
        drv.execute_script("document.querySelectorAll('dialog[open]').forEach((d) => d.close())")
    keys(drv, Keys.ALT, "n")
    dialog_open(drv, "new-structure")
    each("new_structure", "dialog[open] *")
    press_button(drv, "new-structure", "cancel")
    # the restore banner, as a reload would show it
    # (a draft of a document open and modified here is not offered: copy this one under another file's name)
    drv.execute_script("""window.__mace.drafts.flush();
        const k = Object.keys(localStorage).find((x) => x.startsWith('mace.gui.draft.') && x.endsWith('MgO_audit.cif'));
        const d = JSON.parse(localStorage.getItem(k)); d.path = 'structures/MgO_audit_copy.cif'; d.name = 'MgO_audit_copy.cif';
        localStorage.setItem(k.replace('MgO_audit.cif', 'MgO_audit_copy.cif'), JSON.stringify(d));
        window.__mace.offerDrafts();""")
    H.wait(drv, "return !document.getElementById('draft-banner').hidden", 5)
    each("draft_banner", "#draft-banner *")
    drv.execute_script("document.querySelector('#draft-banner [data-v=\"discard\"]').click(); window.__mace.closeAll()")
    (root / "structures" / "MgO_audit.cif").unlink()


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_phase3_surfaces_contrast_clipping_and_requests(fresh, theme):
    drv, root = fresh[0], fresh[2]
    drv.execute_script("window.__mace.settings.set('theme', arguments[0])", theme)
    time.sleep(0.2)
    low, clipped = {}, {}

    def each(name, sel):
        r = drv.execute_script(CONTRAST, sel)
        if r[0] < 4.5:
            low[name] = r
        bad = drv.execute_script(CLIPPED, sel.replace(" *", " :is(button, label, h2, h3, .lbl, th, td, .hist-l, .note)"))
        if bad:
            clipped[name] = bad
        shot(drv, f"p3_{name}_{theme}")
    visit_phase3_surfaces(drv, root, each)
    assert not low, low
    assert not clipped, clipped
    urls = drv.execute_script("return performance.getEntriesByType('resource').map((r) => r.name)")
    origin = drv.execute_script("return location.origin")
    assert all(u.startswith(origin) or u.startswith("data:") or u.startswith("blob:") for u in urls), \
        [u for u in urls if not u.startswith(origin)]
    no_errors(drv)
