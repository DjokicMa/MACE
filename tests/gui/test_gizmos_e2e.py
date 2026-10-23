"""The Rotate tool's compass wheels and the Move tool's 3D gizmo (1.1.6), in headless Firefox with real
WebDriver pointer and key actions:

* Rotate (R) draws three axis wheels and an in-plane wheel with a panel; dragging a wheel turns the view about
  that model axis only, by the angle the readout shows; letting go keeps it.
* Snapping is to whole steps of the turn since the drag began: the Angle snap setting (Custom: any angle), Ctrl
  inverting it while held; Shift snaps every angle, Alt keeps only the larger one (Tab: the other).
* The panel's typed angle turns the view, or the selected atoms (one undo step "Rotate N Atoms about Centroid")
  about their centroid; a drag on a wheel turns the atoms too.
* The Pivot (1.1.6): the selection's centroid or center of mass, the cell's center, the origin or a picked atom,
  each document its own (Settings › Viewport › Rotate pivot the default); a marker shows it, the wheels sit on
  it, a typed turn goes about it and the undo step names it; one selected atom about its own centroid gets a hint.
* Move: dragging an arrow moves the selection along that axis only, a square in its plane only; the panel's
  Δx Δy Δz (or Δa Δb Δc) moves it exactly.
* Ctrl held during an ordinary orbit (right drag) inverts the angle snap too.
"""
import math
import os
import time
from pathlib import Path

import pytest

import guiharness as H

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")
np = pytest.importorskip("numpy")

from selenium.webdriver.common.keys import Keys  # noqa: E402

KEYS = ["viewport.rotation", "viewport.angleSnap", "viewport.snapStep", "viewport.rotatePivot", "theme", "background"]
# Ti1, O2, O3 for a selection that straddles the a and b faces (made whole: O2 at x = -0.07, O3 at y = -0.15), C4 to
# pick as the pivot, H5; no coordinate half a cell from another (every image unambiguous)
PIVOT_FRAC = [[0.10, 0.20, 0.30], [0.93, 0.25, 0.32], [0.15, 0.85, 0.28], [0.60, 0.55, 0.50], [0.70, 0.45, 0.65]]


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("gizmos")
    root, xdg = H.examples_copy(tmp), tmp / "xdg"
    from ase import Atoms
    from ase.io import write
    write(str(root / "structures" / "pivot_test.cif"), Atoms("TiO2CH", scaled_positions=PIVOT_FRAC, cell=np.diag([5.0, 6.0, 7.0]), pbc=True))
    proc, base, _ = H.start_server(root, xdg, MACE_GUI_TEST_HOOKS="1")
    drv = H.firefox(1280, 800)
    try:
        H.load(drv, base)
        yield drv, base, root, xdg
    finally:
        drv.quit()
        proc.terminate()
        proc.wait(timeout=15)


@pytest.fixture
def drv(app, request):
    d = app[0]
    d.execute_script("window.__mace.closeAll(); const S = window.__mace.settings, D = S.defaults();"
                     "for (const k of arguments[0]) S.set(k, D[k]); S.set('theme', 'light');"
                     "window.__mace.run('tool.select'); window.__maceErrors.length = 0;"
                     "document.querySelectorAll('dialog[open]').forEach((x) => x.close());", KEYS)
    H.set_viewport(d, 1280, 800)
    time.sleep(0.3)
    d.execute_script("document.body.focus()")
    failed = request.session.testsfailed
    yield d
    if request.session.testsfailed > failed:
        H.save_failure(d, request.node.name)


def open_doc(drv, path="structures/MgO.cif"):
    drv.execute_async_script("const d = arguments[1]; window.__mace.open(arguments[0]).then((x) => d(x && x.kind))", path)
    H.wait(drv, "return window.__mace.viewport && window.__mace.viewport.displayed() > 0", 30)
    time.sleep(0.5)


def shot(drv, name):
    target = os.environ.get("MACE_GUI_SHOTS")
    if target:
        Path(target).mkdir(parents=True, exist_ok=True)
        drv.save_screenshot(str(Path(target) / f"{name}.png"))


def quat(drv):
    return np.array(drv.execute_script("return window.MACE.state.viewers['viewer-main'].getView().slice(4, 8)"), float)


def conj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def qaxis(axis, deg):
    u = np.asarray(axis, float) / np.linalg.norm(axis)
    h = math.radians(deg) / 2
    return np.array([*(u * math.sin(h)), math.cos(h)])


def model_turn(q0, q1):
    """The model-space turn between two views (q1 = q0 * turn): (unit axis, degrees)."""
    r = qmul(conj(q0), q1)
    if r[3] < 0:
        r = -r
    ang = math.degrees(2 * math.atan2(np.linalg.norm(r[:3]), r[3]))
    ax = r[:3] / (np.linalg.norm(r[:3]) or 1)
    return ax, ang


def turn_deg(q0, q1):
    return model_turn(q0, q1)[1]


def tools(drv, expr):
    return drv.execute_script(f"const T = window.__mace.editor.E.tools; return {expr}")


def ed_rows(drv):
    return drv.execute_script("const e = window.__mace.editor.doc(); return e.rows.map((r) => r.label)")


def wait_idle(drv, timeout=20):
    H.wait(drv, "const e = window.__mace.editor.doc(); return e && e.pending === 0 && !e.waiting", timeout)
    time.sleep(0.1)


def cart(drv):
    return np.array(drv.execute_script("return window.__mace.editor.doc().structure.cart"), float)


def path_drag(drv, pts, keys=(), button=0, mid_keys=None, tab_at=None, events=None):
    """Press at pts[0], move through pts, release. keys: held throughout; mid_keys: pressed after the first move
    (Alt at the press would orbit instead); tab_at: press Tab after that many moves; events: {move: [("down" or
    "up", key)]}."""
    el = H.viewer_canvas(drv)
    t = H._Ticks(drv)
    t.move(el, *H._offset(drv, el, *pts[0]))
    for k in keys:
        t.key_down(k)
    t.down(button)
    for n, p in enumerate(pts[1:]):
        t.move(el, *H._offset(drv, el, *p), ms=16)
        if n == 1 and mid_keys:
            for k in mid_keys:
                t.key_down(k)
        if tab_at is not None and n == tab_at:
            t.key_down(Keys.TAB)
            t.key_up(Keys.TAB)
        for kind, k in (events or {}).get(n, ()):
            (t.key_down if kind == "down" else t.key_up)(k)
    t.up(button)
    for k in reversed(list(keys) + list(mid_keys or [])):
        t.key_up(k)
    t.perform()
    time.sleep(0.4)


def line(p0, p1, n=12):
    return [(p0[0] + (p1[0] - p0[0]) * k / n, p0[1] + (p1[1] - p0[1]) * k / n) for k in range(n + 1)]


def geometry(drv):
    return tools(drv, "T.rotateTool.geometry()")


def free_start(g):
    """A point off every wheel (outside the in-plane wheel), from where a drag turns freely."""
    cx, cy = g["center"]
    return (cx - g["Rv"] - 40, cy - 60)


PANEL_RECT = """const p = document.querySelector('.vp-gz-panel:not([hidden])'), c = document.querySelector('#viewer-main canvas');
  if (!p) return null; const a = p.getBoundingClientRect(), b = c.getBoundingClientRect();
  return [a.left - b.left, a.top - b.top, a.right - b.left, a.bottom - b.top];"""


def ring_arc(g, key, others=("x", "y", "z"), drv=None):
    """A run of the wheel's front points that stays clear of the other wheels and of the panel (a drag along it
    hits this one)."""
    pts = g["rings"][key]
    box = drv.execute_script(PANEL_RECT) if drv else None
    other = [p for k in others if k != key for p in g["rings"][k]]
    cx, cy = g["center"]

    def clear(p):
        inside = box and box[0] - 8 <= p[0] <= box[2] + 8 and box[1] - 8 <= p[1] <= box[3] + 8
        return min((math.hypot(p[0] - q[0], p[1] - q[1]) for q in other), default=99) > 14 and \
            abs(math.hypot(p[0] - cx, p[1] - cy) - g["Rv"]) > 12 and not inside
    best, run = [], []
    for p in pts:
        if clear(p):
            run.append(p)
            if len(run) > len(best):
                best = run[:]
        else:
            run = []
    assert len(best) >= 8, (key, len(best))
    return best


def no_errors(drv):
    errs = H.errors(drv)
    assert not errs, errs


# ------------------------------------------------------------------------------------------------- Rotate

def test_rotate_tool_shows_wheels_panel_and_hint(drv):
    open_doc(drv)
    drv.execute_script("window.__mace.run('tool.rotate')")
    time.sleep(0.4)
    assert drv.execute_script("return document.querySelector('.vp-gizmo').classList.contains('on')")
    g = geometry(drv)
    assert g["target"] == "view" and set(g["rings"]) == {"x", "y", "z"}
    assert all(len(v) > 10 for v in g["rings"].values())
    panel = drv.find_element("css selector", '.vp-gz-panel[data-for="rotate"]')
    assert panel.is_displayed()
    assert "Ctrl (held): invert the snap" in panel.get_attribute("textContent")
    assert drv.execute_script("return document.querySelector('.vp-chip').textContent").startswith("Rotate the view")
    # pixels drawn on a White background in the dark theme too (the view's own tokens)
    drv.execute_script("window.__mace.settings.set('theme', 'dark'); window.__mace.settings.set('background', 'white')")
    time.sleep(0.5)
    drawn = drv.execute_script("""const c = document.querySelector('.vp-gizmo'), g = c.getContext('2d');
        const d = g.getImageData(0, 0, c.width, c.height).data; let n = 0; for (let k = 3; k < d.length; k += 4) if (d[k] > 200) n++; return n""")
    assert drawn > 500
    shot(drv, "gizmo_rotate_dark_white")
    # another tool: the wheels and the panel go
    drv.execute_script("window.__mace.run('tool.select')")
    time.sleep(0.3)
    assert not drv.execute_script("return document.querySelector('.vp-gizmo').classList.contains('on')")
    assert not panel.is_displayed()
    no_errors(drv)


@pytest.mark.parametrize("key,axis", [("z", [0, 0, 1]), ("x", [1, 0, 0])])
def test_dragging_a_wheel_turns_the_view_about_that_axis_only(drv, key, axis):
    open_doc(drv)
    drv.execute_script("window.__mace.run('tool.rotate')")
    time.sleep(0.4)
    g = geometry(drv)
    arc = ring_arc(g, key, drv=drv)
    q0 = quat(drv)
    path_drag(drv, arc[: min(len(arc), 30)])
    q1 = quat(drv)
    ax, ang = model_turn(q0, q1)
    assert ang > 5, ang
    assert abs(abs(ax @ np.array(axis)) - 1) < 1e-6, (ax, axis)
    last = tools(drv, "T.rotateTool.last")
    assert last["comps"][0]["name"] == f"About {key}"
    assert abs(abs(last["comps"][0]["deg"]) - ang) < 1e-6
    # the readout gives the angle both ways round, with a compass dial
    dials = drv.find_element("css selector", '.vp-gz-panel[data-for="rotate"] .gz-dials')
    assert "Last turn" in dials.text and "/" in dials.text
    assert drv.find_elements("css selector", ".gz-dials .gz-compass")
    no_errors(drv)


def test_snap_custom_step_and_ctrl_inverts_it(drv):
    open_doc(drv)
    drv.execute_script("window.__mace.run('tool.rotate')")
    time.sleep(0.4)
    s = free_start(geometry(drv))
    horiz = line(s, (s[0] + 130, s[1]))                        # 52° of free turn (0.4°/px)
    # no snap: 52°
    q0 = quat(drv); path_drag(drv, horiz); assert abs(turn_deg(q0, quat(drv)) - 52) < 1e-6
    # Ctrl inverts the off snap: the custom angle's steps (22.5° -> 45°)
    drv.execute_script("window.__mace.settings.set('viewport.snapStep', 22.5)")
    s = free_start(geometry(drv))
    q0 = quat(drv); path_drag(drv, line(s, (s[0] + 130, s[1])), keys=(Keys.CONTROL,))
    assert abs(turn_deg(q0, quat(drv)) - 45) < 1e-6
    # Angle snap: Custom snaps a plain drag; Ctrl lets it go free
    drv.execute_script("window.__mace.settings.set('viewport.angleSnap', 'custom')")
    s = free_start(geometry(drv))
    q0 = quat(drv); path_drag(drv, line(s, (s[0] + 110, s[1])))   # 44° -> 45°
    assert abs(turn_deg(q0, quat(drv)) - 45) < 1e-6
    s = free_start(geometry(drv))
    q0 = quat(drv); path_drag(drv, line(s, (s[0] + 110, s[1])), keys=(Keys.CONTROL,))
    assert abs(turn_deg(q0, quat(drv)) - 44) < 1e-6
    # the panel's step field sets the custom angle (and the snap follows it)
    step = drv.find_element("css selector", ".vp-gz-panel .gz-step")
    step.clear(); step.send_keys("10", Keys.ENTER)
    time.sleep(0.2)
    assert drv.execute_script("const S = window.__mace.settings; return [S.get('viewport.snapStep'), S.get('viewport.angleSnap')]") == [10, "custom"]
    # below Settings' 0.1° minimum: refused in words, the step kept (it was taken, then not saved)
    step.clear(); step.send_keys("0.05", Keys.ENTER)
    time.sleep(0.2)
    assert step.get_attribute("aria-invalid") == "true"
    assert drv.execute_script("return window.__mace.settings.get('viewport.snapStep')") == 10
    no_errors(drv)


def test_shift_snaps_every_angle_alt_keeps_one_tab_the_other(drv):
    open_doc(drv)
    drv.execute_script("window.__mace.run('tool.rotate')")
    time.sleep(0.4)

    def diag(**kw):
        s = free_start(geometry(drv))
        path_drag(drv, line(s, (s[0] + 220, s[1] - 108), 16), **kw)      # ≈ 88°, ≈ -43°
        return [round(c["deg"], 6) for c in tools(drv, "T.rotateTool.last.comps")]
    assert diag() == pytest.approx([88.0, -43.2])
    drv.execute_script("window.__mace.settings.set('viewport.snapStep', 45)")
    assert diag(keys=(Keys.SHIFT,)) == [90, -45]
    assert diag(mid_keys=(Keys.ALT,)) == [90, 0]
    assert diag(mid_keys=(Keys.ALT,), tab_at=5) == [0, -45]
    # Alt pressed again (Alt+Tab is the system's window switcher): the other axis
    assert diag(mid_keys=(Keys.ALT,), events={5: [("up", Keys.ALT)], 8: [("down", Keys.ALT)]}) == [0, -45]
    # Shift+Alt: in-plane only (about the line of sight)
    q0 = quat(drv)
    s = free_start(geometry(drv))
    path_drag(drv, line(s, (s[0] + 60, s[1] + 160), 16), mid_keys=(Keys.SHIFT, Keys.ALT))
    last = tools(drv, "T.rotateTool.last")
    assert [c["name"] for c in last["comps"]] == ["In-plane"] and last["mode"] == "in-plane only"
    ax, ang = model_turn(q0, quat(drv))
    toward = np.array(drv.execute_script("""const q = arguments[0]; return window.MACE.state.viewers['viewer-main'] && (() => {
        const [x, y, z, w] = q.map((v, k) => (k < 3 ? -v : v)); const p = [0, 0, 1];
        const ix = w * p[0] + y * p[2] - z * p[1], iy = w * p[1] + z * p[0] - x * p[2], iz = w * p[2] + x * p[1] - y * p[0], iw = -x * p[0] - y * p[1] - z * p[2];
        return [ix * w + iw * -x + iy * -z - iz * -y, iy * w + iw * -y + iz * -x - ix * -z, iz * w + iw * -z + ix * -y - iy * -x]; })()""", list(q0)), float)
    assert ang > 5 and abs(abs(ax @ toward) - 1) < 1e-6
    no_errors(drv)


def test_typed_angle_turns_the_view(drv):
    open_doc(drv)
    drv.execute_script("window.__mace.run('tool.rotate')")
    time.sleep(0.4)
    q0 = quat(drv)
    sel = drv.find_element("css selector", ".vp-gz-panel .gz-axis")
    from selenium.webdriver.support.ui import Select
    Select(sel).select_by_value("x")
    ang = drv.find_element("css selector", ".vp-gz-panel .gz-angle")
    ang.clear(); ang.send_keys("30")
    drv.find_element("css selector", '.vp-gz-panel [data-act="ccw"]').click()
    time.sleep(0.6)
    want = qmul(q0, qaxis([1, 0, 0], 30))
    q1 = quat(drv)
    assert min(np.abs(q1 - want).max(), np.abs(q1 + want).max()) < 1e-9
    # the reverse button turns back
    drv.find_element("css selector", '.vp-gz-panel [data-act="cw"]').click()
    time.sleep(0.6)
    q2 = quat(drv)
    assert min(np.abs(q2 - q0).max(), np.abs(q2 + q0).max()) < 1e-9
    no_errors(drv)


def wrapped_equal(a, b, cell):
    """Cartesian points equal modulo lattice translations."""
    inv = np.linalg.inv(np.asarray(cell, float))
    d = (np.asarray(a) - np.asarray(b)) @ inv
    d -= np.round(d)
    return np.abs(d @ np.asarray(cell, float)).max()


def test_typed_and_dragged_turns_rotate_the_selected_atoms(drv):
    open_doc(drv)
    drv.execute_script("window.__mace.run('build.symModeP1'); window.__mace.editor.select([0, 4], 'replace');"
                       "window.__mace.run('tool.rotate')")
    time.sleep(0.4)
    assert tools(drv, "T.rotateTool.geometry().target") == "selection"
    assert "Rotate 2 atoms" in drv.execute_script("return document.querySelector('.vp-chip').textContent")
    s = drv.execute_script("return window.__mace.editor.doc().structure")
    c0, cell = np.array(s["cart"], float), np.array(s["cell"], float)
    # typed: 90° about z, about the pair's centroid (the atoms made whole across the cell's faces)
    from selenium.webdriver.support.ui import Select
    Select(drv.find_element("css selector", ".vp-gz-panel .gz-axis")).select_by_value("z")
    ang = drv.find_element("css selector", ".vp-gz-panel .gz-angle")
    ang.clear(); ang.send_keys("90", Keys.ENTER)
    wait_idle(drv)
    assert ed_rows(drv)[-1] == "Rotate 2 Atoms about Centroid"
    c1 = cart(drv)
    # expected: the pair made whole (atom 4 at its image nearest atom 0), turned about its centroid
    f = np.array(s["frac"], float)
    d = f[4] - f[0]
    p0, p4 = c0[0], (f[0] + d - np.round(d)) @ cell
    piv = (p0 + p4) / 2
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)
    assert wrapped_equal(c1[0], piv + R @ (p0 - piv), cell) < 1e-6
    assert wrapped_equal(c1[4], piv + R @ (p4 - piv), cell) < 1e-6
    others = [i for i in range(len(c0)) if i not in (0, 4)]
    assert np.abs(c1[others] - c0[others]).max() < 1e-9
    # dragged: the z wheel turns them again (one more undo step), the view stays
    q0 = quat(drv)
    g = geometry(drv)
    path_drag(drv, ring_arc(g, "z", drv=drv)[:24])
    wait_idle(drv)
    assert ed_rows(drv)[-1] == "Rotate 2 Atoms about Centroid" and len(ed_rows(drv)) >= 3
    assert np.abs(quat(drv) - q0).max() < 1e-12
    c2 = cart(drv)
    assert np.abs(c2[[0, 4]] - c1[[0, 4]]).max() > 0.05
    assert np.abs(c2[[0, 4], 2] - c1[[0, 4], 2]).max() < 1e-6           # about z: z unchanged
    # Esc during a drag cancels: no edit
    n = len(ed_rows(drv))
    arc = ring_arc(geometry(drv), "z", drv=drv)[:20]
    el = H.viewer_canvas(drv)
    t = H._Ticks(drv)
    t.move(el, *H._offset(drv, el, *arc[0])); t.down(0)
    for p in arc[1:]:
        t.move(el, *H._offset(drv, el, *p), ms=16)
    t.key_down(Keys.ESCAPE); t.key_up(Keys.ESCAPE)
    t.up(0)
    t.perform()
    time.sleep(0.5)
    assert len(ed_rows(drv)) == n and np.abs(cart(drv) - c2).max() < 1e-12
    assert drv.execute_script("return window.__mace.viewport.modelIds().drag") is None
    shot(drv, "gizmo_rotate_selection")
    no_errors(drv)


# ------------------------------------------------------------------------------------------------- the Pivot

PANEL = '.vp-gz-panel[data-for="rotate"]'
M_TI, M_O = 47.867, 15.999


def made_whole(frac, sel):
    """The picked atoms' fractions, each at its image nearest an atom already placed (gizmo_math.js wholeSelection)."""
    out = [np.array(frac[sel[0]], float)]
    for i in sel[1:]:
        best = None
        for r in out:
            d = np.array(frac[i], float) - r
            d -= np.round(d)
            if best is None or np.linalg.norm(d) < best[0]:
                best = (np.linalg.norm(d), r + d)
        out.append(best[1])
    return np.array(out)


def marker_arms(drv, cx, cy, r=8):
    """How many of the pivot marker's four crosshair arms (r px from the center) are drawn: opaque and dark (the
    light theme's text color; the wheels are red, green and blue, the view's center a dot)."""
    return drv.execute_script("""const c = document.querySelector('.vp-gizmo'), g = c.getContext('2d'), k = c.width / c.offsetWidth;
        const [x, y, r] = [arguments[0], arguments[1], arguments[2]];
        return [[r, 0], [-r, 0], [0, r], [0, -r]].filter(([dx, dy]) => {
          const d = g.getImageData(Math.floor((x + dx) * k), Math.floor((y + dy) * k), 1, 1).data;
          return d[3] > 200 && d[0] < 100 && d[1] < 100 && d[2] < 100; }).length""", cx, cy, r)


def type_turn(drv, deg="90"):
    from selenium.webdriver.support.ui import Select
    Select(drv.find_element("css selector", f"{PANEL} .gz-axis")).select_by_value("z")
    ang = drv.find_element("css selector", f"{PANEL} .gz-angle")
    ang.clear()
    ang.send_keys(deg, Keys.ENTER)
    wait_idle(drv)


def pick_pivot(drv, value):
    from selenium.webdriver.support.ui import Select
    Select(drv.find_element("css selector", f"{PANEL} .gz-pivot")).select_by_value(value)
    time.sleep(0.3)


def test_each_pivot_turns_the_selection_about_it_with_a_marker_and_a_named_undo_step(drv, app):
    open_doc(drv, "structures/pivot_test.cif")
    drv.execute_script("window.__mace.run('build.symModeP1')")
    wait_idle(drv)
    s = drv.execute_script("return window.__mace.editor.doc().structure")
    assert s["symbols"] == ["Ti", "O", "O", "C", "H"]
    cell, f0, c0 = np.array(s["cell"], float), np.array(s["frac"], float), np.array(s["cart"], float)
    sel = [0, 1, 2]
    drv.execute_script("window.__mace.editor.select(arguments[0], 'replace'); window.__mace.run('tool.rotate')", sel)
    time.sleep(0.5)
    panel = drv.find_element("css selector", PANEL)
    assert panel.find_element("css selector", ".gz-pivot").get_attribute("value") == "centroid"
    assert not panel.find_element("css selector", ".gz-pivot-hint").is_displayed()
    whole = made_whole(f0, sel) @ cell
    fc = made_whole(f0, sel).mean(axis=0)
    dc = f0[3] - fc
    dc -= np.floor(dc + 0.5)
    expect = {
        "centroid": (whole.mean(axis=0), "Centroid"),
        "com": ((M_TI * whole[0] + M_O * whole[1] + M_O * whole[2]) / (M_TI + 2 * M_O), "Center of Mass"),
        "cell": (cell.sum(axis=0) / 2, "Cell Center"),
        "origin": (np.zeros(3), "Origin"),
        "atom": ((fc + dc) @ cell, "C4"),          # C4, not selected: its image nearest the selection's centroid
    }
    assert np.allclose(expect["atom"][0], np.array([-0.40, 0.55, 0.50]) @ cell)
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)
    centers, marked = {}, 0
    for key, (piv, name) in expect.items():
        if key == "atom":
            # Pick, then a click on C4 (not selected; the selection stays)
            drv.find_element("css selector", f"{PANEL} .gz-pick").click()
            time.sleep(0.2)
            assert tools(drv, "T.rotateTool.picking") and \
                panel.find_element("css selector", ".gz-pick").get_attribute("aria-pressed") == "true"
            assert panel.find_element("css selector", ".gz-pivot-atom").text == "Click an atom…"
            H.click_atom(drv, 3)
            time.sleep(0.4)
            assert not tools(drv, "T.rotateTool.picking")
            assert panel.find_element("css selector", ".gz-pivot").get_attribute("value") == "atom"
            assert panel.find_element("css selector", '.gz-pivot option[value="atom"]').text == "Picked atom (C4)"
            assert drv.execute_script("return [...window.__mace.editor.doc().selection].sort()") == sel
        else:
            pick_pivot(drv, key)
        g = geometry(drv)
        assert g["target"] == "selection" and g["pivot"]["mode"] == key and g["pivot"]["name"] == name
        assert g["pivot"]["point"] == pytest.approx(list(piv), abs=1e-6)
        # the wheels sit on the pivot, and its marker is drawn there
        scr = drv.execute_script("return window.__mace.viewport.camera().screenOf(arguments[0])", list(piv))
        assert g["center"] == pytest.approx(scr, abs=0.5)
        centers[key] = g["center"]
        if 20 < g["center"][0] < 1200 and 20 < g["center"][1] < 700:
            assert marker_arms(drv, *g["center"]) == 4, key
            marked += 1
        shot(drv, f"gizmo_rotate_pivot_{key}")
        # a typed 90° about z: about the pivot, one undo step that names it
        type_turn(drv)
        assert ed_rows(drv)[-1] == f"Rotate 3 Atoms about {name}"
        c1 = cart(drv)
        for k, i in enumerate(sel):
            assert wrapped_equal(c1[i], piv + R @ (whole[k] - piv), cell) < 1e-6, (key, i)
        assert np.abs(c1[[3, 4]] - c0[[3, 4]]).max() < 1e-9
        # undo (from the viewport: in the angle field, Ctrl+Z is the field's own)
        drv.execute_script("document.activeElement.blur(); window.__mace.run('edit.undo')")
        wait_idle(drv)
        assert np.abs(cart(drv) - c0).max() < 1e-9
    assert len({tuple(np.round(v, 1)) for v in centers.values()}) == 5 and marked >= 4, (centers, marked)
    # the document keeps its pivot (the server's gui-docprefs.json); Settings' default is untouched
    assert tools(drv, "T.rotateTool.pivot") == "atom:4"
    assert drv.execute_script("return window.__mace.settings.get('viewport.rotatePivot')") == "centroid"
    time.sleep(1.2)                       # the 500 ms debounce, then the POST
    import json
    data = json.loads((app[3] / "mace" / "gui-docprefs.json").read_text())
    own = [v["values"] for docs in data["kinds"]["display"].values() for p, v in docs.items() if p.endswith("pivot_test.cif")]
    assert own and own[0].get("viewport.rotatePivot") == "atom:4", own
    shot(drv, "gizmo_rotate_pivot_atom")
    # the view (nothing selected): about its center, no pivot marker, the pivot kept for later
    drv.execute_script("window.__mace.editor.select([], 'replace')")
    time.sleep(0.3)
    g = geometry(drv)
    assert g["target"] == "view" and g["pivot"] is None
    time.sleep(0.2)
    assert marker_arms(drv, *g["center"]) == 0            # the view's center: a dot, no marker
    no_errors(drv)


def test_one_atom_about_its_own_centroid_gets_a_hint_and_the_default_comes_from_settings(drv, app):
    import json
    # settings.js keeps what settings.py takes (and drops the rest); the Settings dialog lists the key
    js = drv.execute_async_script("const d = arguments[0]; import(document.querySelector('script[src*=\"/js/shell.js\"]').src"
                                  ".replace('shell.js', 'settings.js')).then((m) => d({def: m.DEFAULTS['viewport.rotatePivot'],"
                                  "good: ['com', 'cell', 'atom:12'].map((v) => m.migrate({'viewport.rotatePivot': v})),"
                                  "bad: ['atom:0', 'mass', 7].map((v) => m.migrate({'viewport.rotatePivot': v}))}))")
    assert js["def"] == "centroid" and js["bad"] == [{}, {}, {}]
    assert [x["viewport.rotatePivot"] for x in js["good"]] == ["com", "cell", "atom:12"]
    schema = drv.execute_async_script("const d = arguments[0]; import(document.querySelector('script[src*=\"/js/shell.js\"]').src"
                                      ".replace('shell.js', 'lazy/settings_schema.js')).then((m) => d(m.SCHEMA.filter((r) => r.key === 'viewport.rotatePivot')))")
    assert len(schema) == 1
    drv.execute_script("window.__mace.settings.set('viewport.rotatePivot', 'cell')")
    drv.execute_async_script("const d = arguments[0]; window.__mace.settings.flush().then(d)")
    assert json.loads((app[3] / "mace" / "gui.json").read_text())["settings"]["viewport.rotatePivot"] == "cell"
    open_doc(drv)
    drv.execute_script("window.__mace.run('build.symModeP1'); window.__mace.editor.select([0], 'replace');"
                       "window.__mace.run('tool.rotate')")
    time.sleep(0.5)
    panel = drv.find_element("css selector", PANEL)
    hint = panel.find_element("css selector", ".gz-pivot-hint")
    # Settings › Viewport › Rotate pivot is the default of a document that has none of its own
    assert panel.find_element("css selector", ".gz-pivot").get_attribute("value") == "cell"
    assert tools(drv, "T.rotateTool.geometry().pivot.mode") == "cell"
    assert not hint.is_displayed()
    drv.execute_script("window.__mace.settings.set('viewport.rotatePivot', 'centroid')")
    time.sleep(0.3)
    assert panel.find_element("css selector", ".gz-pivot").get_attribute("value") == "centroid"
    assert hint.is_displayed() and hint.text.lower() == "one atom turns in place; pick another pivot"
    shot(drv, "gizmo_rotate_pivot_hint")
    pick_pivot(drv, "origin")
    assert not hint.is_displayed()
    pick_pivot(drv, "centroid")
    assert hint.is_displayed()
    # the Move tool: its gizmo at the atom, as before
    drv.execute_script("window.__mace.run('tool.move')")
    time.sleep(0.3)
    h = tools(drv, "T.moveGizmo.screenHandles(window.__mace.editor.doc())")
    a = H.atom_screen(drv, 0)
    assert h and abs(h["center"][0] - a["x"]) < 2 and abs(h["center"][1] - a["y"]) < 2, h
    no_errors(drv)


# ------------------------------------------------------------------------------------------------- Move

def min_image(v, cell):
    f = np.asarray(v) @ np.linalg.inv(cell)
    f -= np.round(f)
    return f @ cell


def handles(drv):
    return tools(drv, "T.moveGizmo.screenHandles(window.__mace.editor.doc())")


def test_move_gizmo_arrow_and_plane_constrain_and_panel_moves_exactly(drv):
    open_doc(drv)
    drv.execute_script("window.__mace.run('build.symModeP1'); window.__mace.editor.select([4], 'replace'); window.__mace.run('tool.move')")
    time.sleep(0.4)
    assert not tools(drv, "T.grabbing")
    assert drv.execute_script("return document.querySelector('.vp-gizmo').classList.contains('on')")
    h = handles(drv)
    assert set(h["axes"]) == {"x", "y", "z"} and set(h["planes"]) == {"xy", "yz", "zx"}
    c0 = cart(drv)
    # the x arrow: along x only
    a = h["axes"]["x"]
    b, tip = np.array(a["base"]), np.array(a["tip"])
    start = b + 0.6 * (tip - b)
    u = (tip - b) / np.linalg.norm(tip - b)
    path_drag(drv, line(tuple(start), tuple(start + 30 * u + 20 * np.array([-u[1], u[0]]))))
    wait_idle(drv)
    c1 = cart(drv)
    cell = np.array(drv.execute_script("return window.__mace.editor.doc().structure.cell"), float)
    d = min_image(c1[4] - c0[4], cell)
    # 30 px along an arrow that spans L Å over |tip - base| px (the perpendicular 20 px adds nothing)
    want = 30 / np.linalg.norm(tip - b) * tools(drv, "T.moveGizmo.handles(window.__mace.editor.doc()).L")
    assert abs(d[0] - want) < 0.02 * want and abs(d[1]) < 1e-9 and abs(d[2]) < 1e-9, (d, want)
    assert ed_rows(drv)[-1] == "Move 1 Atom"
    # the xy square: z stays
    h = handles(drv)
    q = np.array(h["planes"]["xy"])
    mid = q.mean(axis=0)
    path_drag(drv, line(tuple(mid), tuple(mid + np.array([35, -30]))))
    wait_idle(drv)
    c2 = cart(drv)
    d = min_image(c2[4] - c1[4], cell)
    assert abs(d[2]) < 1e-9 and math.hypot(d[0], d[1]) > 0.1, d
    # the panel: an exact displacement, previewed as typed, applied by Enter (one step)
    fields = drv.find_elements("css selector", '.vp-gz-panel[data-for="move"] .gz-d')
    fields[0].send_keys("0.25")
    fields[1].send_keys("-0.1")
    time.sleep(0.3)
    assert tools(drv, "T.grabbing") is True
    assert drv.execute_script("return window.__mace.viewport.modelIds().drag") is not None
    fields[1].send_keys(Keys.ENTER)
    wait_idle(drv)
    c3 = cart(drv)
    assert np.abs(min_image(c3[4] - c2[4], cell) - np.array([0.25, -0.1, 0])).max() < 1e-9
    # the cell frame: Δa as a fraction of a
    drv.find_element("css selector", '.vp-gz-panel[data-for="move"] [data-f="cell"]').click()
    time.sleep(0.2)
    assert handles(drv)["frame"] == "cell"
    f0 = np.array(drv.execute_script("return window.__mace.editor.doc().structure.frac[4]"), float)
    field = drv.find_elements("css selector", '.vp-gz-panel[data-for="move"] .gz-d')[0]
    field.send_keys("0.1")
    drv.find_element("css selector", '.vp-gz-panel[data-for="move"] [data-act="apply"]').click()
    wait_idle(drv)
    f1 = np.array(drv.execute_script("return window.__mace.editor.doc().structure.frac[4]"), float)
    df = f1 - f0
    df -= np.round(df - np.array([0.1, 0, 0]))                 # wrapped into the cell
    assert np.abs(df - np.array([0.1, 0, 0])).max() < 1e-9, df
    drv.find_element("css selector", '.vp-gz-panel[data-for="move"] [data-f="cart"]').click()
    shot(drv, "gizmo_move")
    no_errors(drv)


# ------------------------------------------------------------------------------------------------- orbit

def test_ctrl_inverts_the_snap_of_an_ordinary_orbit(drv):
    open_doc(drv)
    drv.execute_script("window.__mace.settings.set('viewport.snapStep', 22.5)")
    c = drv.execute_script("const r = document.querySelector('#viewer-main canvas').getBoundingClientRect(); return [r.width, r.height]")
    mid = (c[0] / 2, c[1] / 2)
    q0 = quat(drv)
    path_drag(drv, line(mid, (mid[0] + 130, mid[1])), keys=(Keys.CONTROL,), button=2)
    drv.execute_script("document.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape'}))")
    assert abs(turn_deg(q0, quat(drv)) - 45) < 1e-6
    drv.execute_script("window.__mace.settings.set('viewport.angleSnap', '30')")
    q0 = quat(drv)
    path_drag(drv, line(mid, (mid[0] + 130, mid[1])), keys=(Keys.CONTROL,), button=2)
    drv.execute_script("document.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape'}))")
    assert abs(turn_deg(q0, quat(drv)) - 52) < 0.6                   # free: the trackball's own increments
    no_errors(drv)


# ------------------------------------------------------------------------------------------------- the toolbar

@pytest.mark.parametrize("size", [(1024, 640), (1280, 800)])
def test_the_rotate_tool_fits_the_toolbar_and_has_its_key(drv, size):
    """The new toolbar button (R) pushes nothing out of the toolbar down to the smallest window (spec §1.1)."""
    open_doc(drv)
    H.set_viewport(drv, *size)
    time.sleep(0.4)
    got = drv.execute_script("""const t = document.querySelector('#toolbar'), r = t.getBoundingClientRect();
      return {over: t.scrollWidth - t.clientWidth,
        cut: [...t.querySelectorAll('button')].filter((b) => b.getBoundingClientRect().right > r.right + 0.5
          || b.scrollWidth > b.clientWidth + 1).map((b) => b.dataset.cmd || b.id || b.textContent),
        page: document.documentElement.scrollWidth - innerWidth}""")
    assert got == {"over": 0, "cut": [], "page": 0}, got
    assert drv.find_element("css selector", '#toolbar [data-cmd="tool.rotate"]').is_displayed()
    cmd = [c for c in drv.execute_script("return window.__mace.commands") if c["id"] == "tool.rotate"][0]
    assert cmd["keys"] == ["R"]
    # R from the viewport arms it
    drv.execute_script("document.querySelector('#viewer-main').closest('.viewer-wrap').focus()")
    from selenium.webdriver.common.action_chains import ActionChains
    ActionChains(drv).send_keys("r").perform()
    time.sleep(0.3)
    assert drv.execute_script("return window.__mace.viewState().tool") == "rotate"
    no_errors(drv)


def test_orbit_drag_reads_out_its_turn_in_the_chip(drv):
    open_doc(drv)
    drv.execute_script("window.__mace.settings.set('viewport.angleSnap', '15'); window.__mace.run('tool.orbit')")
    time.sleep(0.3)
    c = drv.execute_script("const r = document.querySelector('#viewer-main canvas').getBoundingClientRect(); return [r.width, r.height]")
    el = H.viewer_canvas(drv)
    t = H._Ticks(drv)
    t.move(el, *H._offset(drv, el, c[0] / 2, c[1] / 2))
    t.down(0)
    for k in range(1, 11):
        t.move(el, *H._offset(drv, el, c[0] / 2 + 10 * k, c[1] / 2), ms=16)
    t.perform()
    time.sleep(0.3)
    chip = drv.execute_script("return document.querySelector('.vp-chip').textContent")
    assert chip.startswith("Orbit · Horizontal +45.0° / −315.0° · Vertical 0.0° / 0.0° · snap 15°"), chip
    t = H._Ticks(drv)
    t.up(0)
    t.perform()
    time.sleep(0.4)
    chip = drv.execute_script("return document.querySelector('.vp-chip').textContent")
    assert "Horizontal" not in chip, chip
    no_errors(drv)


def test_picked_atom_again_is_this_documents_own(drv):
    """Pivot › Picked atom brings back the atom picked in this document; in another document it starts Pick
    (it took the first document's atom number, another atom or none in this one)."""
    open_doc(drv, "structures/pivot_test.cif")
    drv.execute_script("window.__mace.run('build.symModeP1')")
    wait_idle(drv)
    drv.execute_script("window.__mace.editor.select([0, 1, 2], 'replace'); window.__mace.run('tool.rotate')")
    time.sleep(0.5)
    drv.find_element("css selector", f"{PANEL} .gz-pick").click()
    time.sleep(0.2)
    H.click_atom(drv, 3)
    time.sleep(0.4)
    assert tools(drv, "T.rotateTool.pivot") == "atom:4"
    pick_pivot(drv, "centroid")
    pick_pivot(drv, "atom")
    assert tools(drv, "T.rotateTool.pivot") == "atom:4" and not tools(drv, "T.rotateTool.picking")
    open_doc(drv, "structures/MgO.cif")
    drv.execute_script("window.__mace.editor.select([0, 1], 'replace'); window.__mace.run('tool.rotate')")
    time.sleep(0.5)
    pick_pivot(drv, "atom")
    assert tools(drv, "T.rotateTool.picking") and tools(drv, "T.rotateTool.pivot") != "atom:4"
    no_errors(drv)
