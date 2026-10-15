"""The 3D views turn about a known, visible center (1.1.5 report: "when you
hold right click it's not clear what the center of rotation is, and why isn't
it by default the center of the unit cell?").

3Dmol turns the model about the point at the view's center (minus the view's
translation). The default view now puts the drawn cell's center there (a
slab: its in-plane cell center at the slab's mid-height; a molecule: the
middle of its atoms), Reset View and an edit keep it, and while a drag turns
the view a small crosshair marks it, gone on release. The expected centers
are worked out here from the files with ase, so nothing here shares the code
under test. Chromium and Firefox, real WebDriver pointer actions.
"""
import time

import pytest

import guiharness as H

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
np = pytest.importorskip("numpy")
pytest.importorskip("ase")

CENTER = "const w = window.MACE.state.viewers[arguments[0]].getView(); return [-w[0], -w[1], -w[2]];"
MARK = """const el = document.getElementById(arguments[0]), m = el && el.querySelector('.pivot-mark');
  if (!m) return null;
  const cv = el.querySelector('canvas'), r = cv.getBoundingClientRect(), z = r.width / cv.offsetWidth, mr = m.getBoundingClientRect();
  return {hidden: m.hidden || getComputedStyle(m).display === 'none', x: (mr.left + mr.width / 2 - r.left) / z,
          y: (mr.top + mr.height / 2 - r.top) / z, w: cv.offsetWidth, h: cv.offsetHeight, z,
          left: r.left, top: r.top, sw: r.width, sh: r.height};"""


def _write_inputs(root):
    from ase import Atoms
    from ase.io import write
    d = root / "pivot"
    d.mkdir()
    # a molecule off the origin, no cell
    write(str(d / "water.xyz"), Atoms("OH2", positions=[[3, 4, 5], [3.9572, 4, 5], [2.76, 4.927, 5]]), format="xyz")


@pytest.fixture(scope="module", params=["chromium", "firefox"])
def app(request, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pivot")
    root = H.examples_copy(tmp)
    _write_inputs(root)
    proc, base, _ = H.start_server(root, tmp / "xdg", MACE_GUI_TEST_HOOKS="1")
    try:
        drv = H.browser(request.param, width=1280, height=800)
    except BaseException:
        proc.terminate()
        proc.wait(timeout=15)
        raise
    try:
        H.load(drv, base)
        yield drv, root
    finally:
        drv.quit()
        proc.terminate()
        proc.wait(timeout=15)


@pytest.fixture
def drv(app, request):
    d, _ = app
    failed = request.session.testsfailed
    yield d
    if request.session.testsfailed > failed:
        H.save_failure(d, request.node.name)
    d.execute_script("window.__mace.closeAll(); window.__mace.settings.set('uiScale', 100)")
    H.set_viewport(d, 1280, 800)
    time.sleep(0.3)


def open_doc(drv, path, kind="structure"):
    got = drv.execute_async_script(
        "const d = arguments[1]; window.__mace.open(arguments[0]).then((x) => d(x && x.kind))", path)
    assert got == kind, (path, got)
    if kind == "structure":
        H.wait(drv, "return window.__mace.viewState().displayedAtoms > 0", 30)
    time.sleep(0.6)


def center(drv, viewer="viewer-main"):
    return np.array(drv.execute_script(CENTER, viewer))


def read(path):
    from ase.io import read as aread
    return aread(str(path))


def pointer(drv):
    from selenium.webdriver.common.actions.action_builder import ActionBuilder
    from selenium.webdriver.common.actions.mouse_button import MouseButton
    ab = ActionBuilder(drv)
    return ab, ab.pointer_action, MouseButton


# every change of the marker while a gesture runs: shown (where, in the canvas's own px) or hidden
WATCH = """const el = document.getElementById(arguments[0]), m = el.querySelector('.pivot-mark'), cv = el.querySelector('canvas');
  window.__pivotLog = [];
  const log = () => { const r = cv.getBoundingClientRect(), z = r.width / cv.offsetWidth, mr = m.getBoundingClientRect();
    window.__pivotLog.push(m.hidden ? {shown: false} : {shown: true, x: (mr.left + mr.width / 2 - r.left) / z,
      y: (mr.top + mr.height / 2 - r.top) / z, w: cv.offsetWidth, h: cv.offsetHeight,
      view: window.MACE.state.viewers[arguments[0]].getView()}); };
  if (window.__pivotObs) window.__pivotObs.disconnect();
  window.__pivotObs = new MutationObserver(log);
  window.__pivotObs.observe(m, {attributes: true, attributeFilter: ['hidden', 'style']});"""


def drag_gesture(drv, x, y, button):
    """Press, drag 48 px (in steps, with pauses), release: one real pointer gesture."""
    ab, p, _ = pointer(drv)
    p.move_to_location(int(x), int(y))
    p.pointer_down(button=button)
    for k in range(1, 9):
        p.move_to_location(int(x + 6 * k), int(y + 2 * k))
        p.pause(0.03)
    p.pointer_up(button=button)
    ab.perform()


def check_marker_log(drv, viewer):
    log = drv.execute_script("const l = window.__pivotLog; window.__pivotObs.disconnect(); return l")
    shown = [e for e in log if e["shown"]]
    assert shown, log
    for e in shown:
        # on the rotation center: the view's center (where the cell's center is drawn)
        assert abs(e["x"] - e["w"] / 2) < 2 and abs(e["y"] - e["h"] / 2) < 2, e
    assert not log[-1]["shown"], log                 # gone on release
    assert drv.execute_script(MARK, viewer)["hidden"]
    return shown


def test_crystal_turns_about_its_cell_center(app, drv):
    _, root = app
    open_doc(drv, "structures/MgO.cif")
    atoms = read(root / "structures" / "MgO.cif")
    want = atoms.cell.array.sum(axis=0) / 2
    assert np.allclose(center(drv), want, atol=1e-3), (center(drv), want)
    # an edit keeps the view (no jump); Reset View brings the cell's center back
    drv.execute_script("window.__mace.run('view.rotateLeft')")
    time.sleep(0.5)
    drv.execute_script("const v = window.MACE.state.viewers['viewer-main'], w = v.getView(); v.setView([w[0] + 1, w[1], w[2], ...w.slice(3)])")
    drv.execute_script("window.__mace.run('view.reset')")
    H.wait(drv, f"const w = window.MACE.state.viewers['viewer-main'].getView(); return Math.abs(-w[0] - {want[0]}) < 1e-3", 10)
    assert np.allclose(center(drv), want, atol=1e-3)
    # a 2x2x2 display range: the center of what is drawn
    drv.execute_script("window.__mace.run('view.range2')")
    time.sleep(0.6)
    drv.execute_script("window.__mace.run('view.reset')")
    time.sleep(0.6)
    assert np.allclose(center(drv), 2 * want, atol=1e-3), (center(drv), 2 * want)
    drv.execute_script("window.__mace.run('view.range1')")
    time.sleep(0.4)
    assert not H.errors(drv), H.errors(drv)


def test_slab_turns_about_its_mid_height(app, drv):
    """A SLAB output (periodic in a and b): the in-plane cell's center, at the atoms' mid-height."""
    open_doc(drv, "outputs/graphene_slab_sp.out")
    s = drv.execute_script("const s = window.__mace.doc().structure; return {pbc: s.pbc, cell: s.cell, cart: s.cart}")
    assert s["pbc"] == [True, True, False], s["pbc"]
    C, P = np.array(s["cell"]), np.array(s["cart"])
    assert abs(C[0][2]) < 1e-9 and abs(C[1][2]) < 1e-9        # a and b in the xy plane, the slab along z
    want = (C[0] + C[1]) / 2
    want[2] = (P[:, 2].min() + P[:, 2].max()) / 2
    assert np.allclose(center(drv), want, atol=1e-3), (center(drv), want)
    assert not H.errors(drv), H.errors(drv)


def test_molecule_turns_about_its_atoms_middle(app, drv):
    _, root = app
    open_doc(drv, "pivot/water.xyz")
    # the server may move a molecule (its lowest atom to the origin): the drawn positions, checked against the file's shape
    P = np.array(drv.execute_script("return window.__mace.doc().structure.cart"))
    F = read(root / "pivot" / "water.xyz").positions
    assert np.allclose(P - P[0], F - F[0], atol=1e-4)
    want = (P.min(axis=0) + P.max(axis=0)) / 2
    assert np.allclose(center(drv), want, atol=1e-3), (center(drv), want)
    assert not H.errors(drv), H.errors(drv)


@pytest.mark.parametrize("scale", [100, 125])
def test_right_drag_shows_the_center_until_release(drv, scale):
    open_doc(drv, "structures/MgO.cif")
    drv.execute_script("window.__mace.settings.set('uiScale', arguments[0])", scale)
    time.sleep(0.8)
    m = drv.execute_script(MARK, "viewer-main")
    assert m and m["hidden"], m
    q0 = drv.execute_script("return window.MACE.state.viewers['viewer-main'].getView().slice(4)")
    drv.execute_script(WATCH, "viewer-main")
    from selenium.webdriver.common.actions.mouse_button import MouseButton
    drag_gesture(drv, m["left"] + m["sw"] * 0.3, m["top"] + m["sh"] * 0.3, MouseButton.RIGHT)
    time.sleep(0.3)
    check_marker_log(drv, "viewer-main")
    assert drv.execute_script("return window.MACE.state.viewers['viewer-main'].getView().slice(4)") != q0   # it turned
    assert not drv.execute_script("return document.querySelector('.menu')")     # a drag, not a click: no menu
    assert not H.errors(drv), H.errors(drv)


def test_3dmol_viewer_left_drag_shows_the_center(drv):
    """The comparison viewer keeps 3Dmol's own mouse: a left-drag turns it."""
    open_doc(drv, "outputs/Ag2Br3_opt_unfinished.out", "trajectory")
    H.wait(drv, "const v = window.MACE.state.viewers['viewer-before']; return v && v.getModel() && v.visible()", 30)
    time.sleep(0.5)
    m = drv.execute_script(MARK, "viewer-before")
    assert m and m["hidden"], m
    q0 = drv.execute_script("return window.MACE.state.viewers['viewer-before'].getView().slice(4)")
    drv.execute_script(WATCH, "viewer-before")
    from selenium.webdriver.common.actions.mouse_button import MouseButton
    drag_gesture(drv, m["left"] + m["sw"] * 0.3, m["top"] + m["sh"] * 0.4, MouseButton.LEFT)
    time.sleep(0.3)
    check_marker_log(drv, "viewer-before")
    assert drv.execute_script("return window.MACE.state.viewers['viewer-before'].getView().slice(4)") != q0
    assert not H.errors(drv), H.errors(drv)


def test_center_rotation_commands(app, drv):
    _, root = app
    open_doc(drv, "structures/MgO.cif")
    want = read(root / "structures" / "MgO.cif").cell.array.sum(axis=0) / 2
    zoom0 = drv.execute_script("return window.MACE.state.viewers['viewer-main'].getView()[3]")
    # on the first atom: its position becomes the center, the zoom stays
    drv.execute_script("window.__mace.editor.select([0], 'replace')")
    drv.execute_script("window.__mace.run('view.pivotSelection')")
    time.sleep(0.6)
    p0 = np.array(drv.execute_script("return window.__mace.doc().structure.cart[0]"))
    assert np.allclose(center(drv), p0, atol=1e-3), (center(drv), p0)
    assert abs(drv.execute_script("return window.MACE.state.viewers['viewer-main'].getView()[3]") - zoom0) < 1e-6
    drv.execute_script("window.__mace.run('view.pivotCell')")
    time.sleep(0.6)
    assert np.allclose(center(drv), want, atol=1e-3)
    # both are in the palette and in the empty-space menu
    ids = drv.execute_script("return window.__mace.commands.map((c) => c.id)")
    assert "view.pivotCell" in ids and "view.pivotSelection" in ids
    assert not H.errors(drv), H.errors(drv)
