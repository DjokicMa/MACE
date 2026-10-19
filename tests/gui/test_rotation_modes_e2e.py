"""Rotation and zoom modes of the 3D views (Settings › Viewport › Rotation, Angle snap and Zoom toward;
the toolbar's View Along ▾ menu).

* Turntable mode: after any number of diagonal drags the model's up (what was up on screen
  when the turntable took it) still projects straight up the screen, no roll, in the structure view
  and in a 3Dmol run view; a free (trackball) drag rolls it.
* An angle snap turns the view in whole steps: each drag's turn is a multiple of the step, and the
  arrow keys turn by one step (15° without a snap).
* Snap View to Nearest Axis looks along the nearest cell or Cartesian axis.
* Zoom toward › Mouse pointer keeps the model point under the pointer where it is on screen as the wheel
  zooms (the structure view and a 3Dmol run view); View center keeps the view's center there instead,
  and Invert scroll zoom flips the direction in both.
* The settings are saved: a reload keeps them; the toolbar's View Along ▾ menu sets and shows them.
Chromium and Firefox, real WebDriver pointer actions; the quaternions are checked here with numpy.
"""
import time

import pytest

import guiharness as H

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
np = pytest.importorskip("numpy")

KEYS = ["viewport.rotation", "viewport.angleSnap", "viewport.zoomToward", "invertWheelZoom", "uiScale"]
CANVAS = """const el = document.getElementById(arguments[0]), cv = el.querySelector('canvas'), r = cv.getBoundingClientRect();
  return {left: r.left, top: r.top, w: r.width, h: r.height};"""
QUAT = "return window.MACE.state.viewers[arguments[0]].getView().slice(4, 8)"


@pytest.fixture(scope="module", params=["chromium", "firefox"])
def app(request, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("rotmodes")
    root = H.examples_copy(tmp)
    proc, base, _ = H.start_server(root, tmp / "xdg", MACE_GUI_TEST_HOOKS="1")
    try:
        drv = H.browser(request.param, width=1280, height=800)
    except BaseException:
        proc.terminate()
        proc.wait(timeout=15)
        raise
    try:
        H.load(drv, base)
        yield drv, base
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
    d.execute_script("window.__mace.closeAll(); const S = window.__mace.settings, D = S.defaults();"
                     "for (const k of arguments[0]) S.set(k, D[k]);", KEYS)
    time.sleep(0.3)


def open_doc(drv, path, kind="structure"):
    got = drv.execute_async_script(
        "const d = arguments[1]; window.__mace.open(arguments[0]).then((x) => d(x && x.kind))", path)
    assert got == kind, (path, got)
    if kind == "structure":
        H.wait(drv, "return window.__mace.viewState().displayedAtoms > 0", 30)
    time.sleep(0.6)


def setting(drv, key, value):
    drv.execute_script("window.__mace.settings.set(arguments[0], arguments[1])", key, value)


def quat(drv, viewer="viewer-main"):
    return np.array(drv.execute_script(QUAT, viewer), float)


def rot(q, v):
    """v turned by the unit quaternion q = [x, y, z, w] (model -> camera)."""
    u, w = np.array(q[:3]), q[3]
    v = np.asarray(v, float)
    return v + 2 * np.cross(u, np.cross(u, v) + w * v)


def conj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def turn_deg(q0, q1):
    """The angle of the turn from q0 to q1, in degrees."""
    r = qmul(q1, conj(q0))
    return float(np.degrees(2 * np.arccos(min(1.0, abs(r[3])))))


def drag(drv, viewer, dx, dy, button="right", start=(0.5, 0.5), steps=10):
    from selenium.webdriver.common.actions.action_builder import ActionBuilder
    from selenium.webdriver.common.actions.mouse_button import MouseButton
    c = drv.execute_script(CANVAS, viewer)
    x, y = c["left"] + c["w"] * start[0], c["top"] + c["h"] * start[1]
    ab = ActionBuilder(drv)
    p = ab.pointer_action
    b = MouseButton.RIGHT if button == "right" else MouseButton.LEFT
    p.move_to_location(int(x), int(y))
    p.pointer_down(button=b)
    for k in range(1, steps + 1):
        p.move_to_location(int(x + dx * k / steps), int(y + dy * k / steps))
        p.pause(0.02)
    p.pointer_up(button=b)
    ab.perform()
    time.sleep(0.25)
    # a right-click that did not move would open a menu: none here
    drv.execute_script("document.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape'}))")


def roll_of(q, up):
    """How far the model's up has turned off the screen's vertical (its screen x over its in-plane length)."""
    s = rot(q, up)
    return abs(s[0]) / max(1e-9, np.hypot(s[0], s[1]))


DIAGONALS = [(140, 90), (-60, 130), (110, -70), (-150, -40)]


@pytest.mark.parametrize("viewer,button", [("viewer-main", "right"), ("viewer-before", "left")])
def test_turntable_keeps_roll_at_zero(drv, viewer, button):
    if viewer == "viewer-main":
        open_doc(drv, "structures/MgO.cif")
    else:
        open_doc(drv, "outputs/Ag2Br3_opt_unfinished.out", "trajectory")
        H.wait(drv, "const v = window.MACE.state.viewers['viewer-before']; return v && v.getModel() && v.visible()", 30)
        time.sleep(0.5)
    # the free trackball rolls the structure over a few diagonal drags (what the turntable is for)
    q0 = quat(drv, viewer)
    up = rot(conj(q0), [0, 1, 0])
    for dx, dy in DIAGONALS:
        drag(drv, viewer, dx, dy, button)
    assert roll_of(quat(drv, viewer), up) > 0.05
    # Turntable: the up taken at the first drag stays vertical on screen through every drag
    setting(drv, "viewport.rotation", "axes")
    q0 = quat(drv, viewer)
    up = rot(conj(q0), [0, 1, 0])
    for dx, dy in DIAGONALS:
        drag(drv, viewer, dx, dy, button)
        q = quat(drv, viewer)
        assert roll_of(q, up) < 1e-4, (dx, dy, roll_of(q, up))
    assert turn_deg(q0, quat(drv, viewer)) > 10            # it did turn
    assert not H.errors(drv), H.errors(drv)


@pytest.mark.parametrize("mode", ["free", "axes"])
def test_angle_snap_turns_in_whole_steps(drv, mode):
    open_doc(drv, "structures/MgO.cif")
    setting(drv, "viewport.rotation", mode)
    setting(drv, "viewport.angleSnap", "30")
    for dx, dy in [(130, 0), (0, -170), (40, 0), (260, 0)]:
        q0 = quat(drv)
        drag(drv, "viewer-main", dx, dy)
        a = turn_deg(q0, quat(drv))
        want = round(0.4 * abs(dx or dy) / 30) * 30           # 0.4°/px, rounded to whole steps
        assert abs(a - want) < 1e-3, (dx, dy, a, want)
    # a 3Dmol view snaps the same way
    open_doc(drv, "outputs/Ag2Br3_opt_unfinished.out", "trajectory")
    H.wait(drv, "const v = window.MACE.state.viewers['viewer-before']; return v && v.getModel() && v.visible()", 30)
    time.sleep(0.5)
    q0 = quat(drv, "viewer-before")
    drag(drv, "viewer-before", 160, 0, "left")
    a = turn_deg(q0, quat(drv, "viewer-before"))
    assert a > 1 and abs(a / 30 - round(a / 30)) < 1e-4, a
    assert not H.errors(drv), H.errors(drv)


def test_arrow_keys_turn_by_the_snap_step(drv):
    open_doc(drv, "structures/MgO.cif")
    drv.execute_script("document.querySelector('#viewer-main').closest('.viewer-wrap').focus()")
    for snap, want in (("off", 15), ("45", 45), ("90", 90)):
        setting(drv, "viewport.angleSnap", snap)
        q0 = quat(drv)
        drv.execute_script("window.__mace.run('view.rotate15Right')")
        time.sleep(0.5)
        assert abs(turn_deg(q0, quat(drv)) - want) < 1e-3, (snap, turn_deg(q0, quat(drv)))
    # Shift+arrow stays 1°
    q0 = quat(drv)
    drv.execute_script("window.__mace.run('view.rotate15Up', {chord: 'Shift+Up'})")
    time.sleep(0.3)
    assert abs(turn_deg(q0, quat(drv)) - 1) < 1e-3
    assert not H.errors(drv), H.errors(drv)


def test_snap_to_nearest_axis(drv):
    open_doc(drv, "structures/MgO.cif")
    drv.execute_script("window.__mace.run('view.alongB')")
    time.sleep(0.5)
    drag(drv, "viewer-main", 30, 20)                       # a little off b
    drv.execute_script("window.__mace.run('view.snapNearestAxis')")
    time.sleep(0.6)
    q = quat(drv)
    cell = np.array(drv.execute_script("return window.__mace.doc().structure.cell"), float)
    b = cell[1] / np.linalg.norm(cell[1])
    toward = rot(conj(q), [0, 0, 1])
    assert abs(abs(toward @ b) - 1) < 1e-6, (toward, b)
    assert drv.execute_script("return document.getElementById('tb-along-label').textContent") in ("b", "−b")
    # up is an axis too: the view is square to the cell
    up = rot(conj(q), [0, 1, 0])
    assert max(abs(up @ (c / np.linalg.norm(c))) for c in cell) > 1 - 1e-6
    assert not H.errors(drv), H.errors(drv)


# the screen point (page px) of model point p, and of the view's center (the rotation center), in viewer arguments[0]
SCREEN = """const v = window.MACE.state.viewers[arguments[0]], p = arguments[1], w = v.getView();
  const s = v.modelToScreen({x: p[0], y: p[1], z: p[2]}), c = v.modelToScreen({x: -w[0], y: -w[1], z: -w[2]});
  return {p: [s.x, s.y], c: [c.x, c.y], zoom: w[3]};"""
# an atom of the drawn model away from the view's center: the point to zoom toward
ATOM = """const v = window.MACE.state.viewers[arguments[0]], w = v.getView(), c = [-w[0], -w[1], -w[2]];
  const atoms = v.getModel().selectedAtoms({});
  let best = atoms[0], bd = -1;
  for (const a of atoms) { const s = v.modelToScreen(a), o = v.modelToScreen({x: c[0], y: c[1], z: c[2]}), d = Math.hypot(s.x - o.x, s.y - o.y);
    const r = v.getCanvas().getBoundingClientRect();
    if (d > bd && s.x > r.left + 20 && s.x < r.right - 20 && s.y > r.top + 20 && s.y < r.bottom - 20) { bd = d; best = a; } }
  return [best.x, best.y, best.z];"""


def wheel_at(drv, viewer, page_xy, dy):
    from selenium.webdriver.common.action_chains import ActionChains
    from selenium.webdriver.common.actions.wheel_input import ScrollOrigin
    el = drv.find_element("css selector", f"#{viewer} canvas")
    r = drv.execute_script(CANVAS, viewer)
    off = (int(round(page_xy[0] - (r["left"] + r["w"] / 2))), int(round(page_xy[1] - (r["top"] + r["h"] / 2))))
    ActionChains(drv).scroll_from_origin(ScrollOrigin.from_element(el, *off), 0, int(dy)).perform()
    time.sleep(0.4)


@pytest.mark.parametrize("viewer", ["viewer-main", "viewer-before"])
def test_zoom_toward_pointer_or_view_center(drv, viewer):
    if viewer == "viewer-main":
        open_doc(drv, "structures/MgO.cif")
    else:
        open_doc(drv, "outputs/Ag2Br3_opt_unfinished.out", "trajectory")
        H.wait(drv, "const v = window.MACE.state.viewers['viewer-before']; return v && v.getModel() && v.visible()", 30)
        time.sleep(0.5)
    p = drv.execute_script(ATOM, viewer)
    for invert in (False, True):
        setting(drv, "invertWheelZoom", invert)
        # Mouse pointer: the atom under the pointer stays under it (within 2 px), zooming in and out
        setting(drv, "viewport.zoomToward", "pointer")
        for dy in ((200, -200) if invert else (-200, 200)):            # in, then back out
            s0 = drv.execute_script(SCREEN, viewer, p)
            wheel_at(drv, viewer, s0["p"], dy)
            s1 = drv.execute_script(SCREEN, viewer, p)
            assert s1["zoom"] != s0["zoom"]
            assert (s1["zoom"] > s0["zoom"]) == ((dy < 0) != invert)      # up zooms in unless inverted
            assert abs(s1["p"][0] - s0["p"][0]) <= 2 and abs(s1["p"][1] - s0["p"][1]) <= 2, (dy, s0, s1)
        # View center: the center stays put and the atom off it moves (away from the center when zooming in)
        setting(drv, "viewport.zoomToward", "center")
        s0 = drv.execute_script(SCREEN, viewer, p)
        wheel_at(drv, viewer, s0["p"], 200 if invert else -200)
        s1 = drv.execute_script(SCREEN, viewer, p)
        assert s1["zoom"] > s0["zoom"]
        assert abs(s1["c"][0] - s0["c"][0]) <= 1 and abs(s1["c"][1] - s0["c"][1]) <= 1, (s0, s1)
        moved = np.hypot(s1["p"][0] - s0["p"][0], s1["p"][1] - s0["p"][1])
        assert moved > 3, (s0, s1)
        wheel_at(drv, viewer, s0["p"], -200 if invert else 200)              # back out
    assert not H.errors(drv), H.errors(drv)


def test_rotation_settings_persist_and_show_in_the_toolbar(app, drv):
    _, base = app
    open_doc(drv, "structures/MgO.cif")
    item = "return document.querySelector('.menu [data-cmd-id=\"' + arguments[0] + '\"]')"
    # the toolbar's View Along menu sets them, and checks the current ones
    drv.find_element("css selector", "#tb-along").click()
    assert H.wait(drv, item, 5, "view.rotation.free").get_attribute("aria-checked") == "true"
    H.wait(drv, item, 5, "view.rotation.axes").click()
    drv.find_element("css selector", "#tb-along").click()
    H.wait(drv, item, 5, "view.angleSnap.45").click()
    drv.find_element("css selector", "#tb-along").click()
    H.wait(drv, item, 5, "view.zoomToward.pointer").click()
    drv.find_element("css selector", "#tb-along").click()
    assert H.wait(drv, item, 5, "view.rotation.axes").get_attribute("aria-checked") == "true"
    assert H.wait(drv, item, 5, "view.angleSnap.45").get_attribute("aria-checked") == "true"
    assert H.wait(drv, item, 5, "view.zoomToward.pointer").get_attribute("aria-checked") == "true"
    shots = __import__("os").environ.get("MACE_GUI_SHOTS")
    if shots:
        drv.save_screenshot(f"{shots}/rotation_menu.png")
    drv.execute_script("document.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape'}))")
    assert drv.execute_script("const S = window.__mace.settings; return [S.get('viewport.rotation'), S.get('viewport.angleSnap'), S.get('viewport.zoomToward')]") == ["axes", "45", "pointer"]
    # saved: a reload keeps them (the server's copy and the browser's)
    time.sleep(1.0)
    H.load(drv, base)
    H.wait(drv, "return window.__mace && window.__mace.settings", 30)
    assert drv.execute_script("const S = window.__mace.settings; return [S.get('viewport.rotation'), S.get('viewport.angleSnap'), S.get('viewport.zoomToward')]") == ["axes", "45", "pointer"]
    assert not H.errors(drv), H.errors(drv)
