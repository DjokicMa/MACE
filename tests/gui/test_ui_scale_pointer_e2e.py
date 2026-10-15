"""Clicks, hovers, region selections, right-clicks and drags land under the
pointer at every UI scale (Settings > Appearance > UI scale, Ctrl+= / Ctrl+-).

The UI scale is a CSS zoom on the window's shell. Chromium 128+ (Chrome,
Brave, Edge) and Firefox 126+ apply zoom as the standard says: a pointer
event's clientX and getBoundingClientRect() are screen px, while an element's
own lengths (offsetWidth, a canvas's drawing, 3Dmol's WIDTH) are its zoomed
px. Code that subtracted the two picked the atom at zoom x the pointer's
offset: at 110 % a click selected an atom lower and to the right of the
pointer, and the plots' hover and the splitters were off the same way.

Each case uses real WebDriver pointer actions at screen points worked out
from where the thing is drawn (the canvas's screen corner plus its own px
times the zoom), so nothing here shares the code under test. Chromium first
(the browser of the report); Firefox too when it is installed.
"""
import sys
import time

import pytest

import guiharness as H

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")                 # the server reads the structures with it
pytest.importorskip("plotly")
sys.path.insert(0, str(H.REPO_ROOT / "tests"))
from test_gui_results import write_band  # noqa: E402

SCALES = (90, 100, 110, 125)

# the screen point of atom i: the canvas's screen corner plus its own px times the zoom
ATOM_AT = """const cv = document.querySelector('#viewer-main canvas'), r = cv.getBoundingClientRect();
  const z = r.width / cv.offsetWidth, a = window.__mace.viewport.atomScreen(arguments[0]);
  return a && {x: r.left + a.x * z, y: r.top + a.y * z, z, own: [a.x, a.y], w: cv.offsetWidth, h: cv.offsetHeight};"""


@pytest.fixture(scope="module", params=["chromium", "firefox"])
def app(request, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("uiscale")
    root = H.examples_copy(tmp)
    (root / "results").mkdir()
    write_band(root / "results")
    proc, base, _ = H.start_server(root, tmp / "xdg", MACE_GUI_TEST_HOOKS="1")
    try:
        drv = H.browser(request.param, width=1280, height=800)
    except BaseException:
        proc.terminate()
        proc.wait(timeout=15)
        raise
    try:
        H.load(drv, base)
        yield drv
    finally:
        drv.quit()
        proc.terminate()
        proc.wait(timeout=15)


@pytest.fixture
def drv(app, request):
    failed = request.session.testsfailed
    yield app
    if request.session.testsfailed > failed:
        H.save_failure(app, request.node.name)
    app.execute_script("window.__mace.closeAll(); window.__mace.settings.set('uiScale', 100)")
    H.set_viewport(app, 1280, 800)
    time.sleep(0.3)


def scale(drv, s):
    drv.execute_script("window.__mace.settings.set('uiScale', arguments[0])", s)
    time.sleep(0.5)
    z = drv.execute_script("const sh = document.getElementById('shell'); return sh.getBoundingClientRect().width / sh.offsetWidth")
    assert abs(z - s / 100) < 0.01, (s, z)
    # the 3D view settles at its new size (3Dmol resizes on a later frame)
    last = None
    for _ in range(30):
        now = drv.execute_script("const c = document.querySelector('#viewer-main canvas'); return c ? [c.offsetWidth, c.offsetHeight, c.width] : null")
        if now == last:
            break
        last = now
        time.sleep(0.15)


def pointer(drv):
    from selenium.webdriver.common.actions.action_builder import ActionBuilder
    from selenium.webdriver.common.actions.mouse_button import MouseButton
    ab = ActionBuilder(drv)
    return ab, ab.pointer_action, MouseButton


def move(drv, x, y):
    ab, p, _ = pointer(drv)
    p.move_to_location(int(round(x)), int(round(y)))
    ab.perform()


def click(drv, x, y, right=False):
    ab, p, B = pointer(drv)
    p.move_to_location(int(round(x)), int(round(y)))
    p.click(button=B.RIGHT if right else B.LEFT)
    ab.perform()


def drag(drv, path):
    ab, p, _ = pointer(drv)
    p.move_to_location(int(round(path[0][0])), int(round(path[0][1])))
    p.pointer_down()
    for x, y in path[1:]:
        p.move_to_location(int(round(x)), int(round(y)))
    p.pointer_up()
    ab.perform()


def open_structure(drv):
    kind = drv.execute_async_script("const d = arguments[1]; window.__mace.open(arguments[0]).then((x) => d(x && x.kind))",
                                    "structures/MgO.cif")
    assert kind == "structure"
    H.wait(drv, "return window.__mace.viewState().displayedAtoms > 0", 30)
    time.sleep(0.5)


def selection(drv):
    return drv.execute_script("return window.__mace.editor.selection()")


def targets(drv):
    """Atoms drawn in front at their own center, well inside the canvas, at least
    30 own px from the canvas's corner (so a point off by the zoom misses them)."""
    out = []
    n = drv.execute_script("return window.__mace.viewport.displayed()")
    for i in range(n):
        a = drv.execute_script(ATOM_AT, i)
        if not a:
            continue
        x, y = a["own"]
        if not (30 < x < a["w"] - 30 and 30 < y < a["h"] - 30):
            continue
        hit = drv.execute_script("return window.__mace.viewport.pickAt(arguments[0], arguments[1])", x, y)
        if hit and hit["i"] == i:
            out.append((i, a))
    assert len(out) >= 2, out
    return out[:3]


@pytest.mark.parametrize("s", SCALES)
def test_click_box_and_right_click_on_atoms(drv, s):
    open_structure(drv)
    scale(drv, s)
    for i, a in targets(drv):
        # a click selects the atom under the pointer
        drv.execute_script("window.__mace.editor.select && window.__mace.editor.select([], 'replace')")
        click(drv, a["x"], a["y"])
        H.wait(drv, "return window.__mace.editor.selection().length > 0", 5)
        assert selection(drv) == [i], (s, i, a, selection(drv))
        # a small box around it, drawn on screen, holds that atom
        click(drv, a["x"], a["y"])                      # nothing new: the same atom
        d = 7
        drag(drv, [(a["x"] - d, a["y"] - d), (a["x"], a["y"] - d), (a["x"] + d, a["y"]), (a["x"] + d, a["y"] + d)])
        time.sleep(0.3)
        assert i in selection(drv), (s, i, a, selection(drv))
        # a right-click opens the atom's menu where the pointer is
        click(drv, a["x"], a["y"], right=True)
        m = H.wait(drv, "const m = [...document.querySelectorAll('.menu')].pop(); if (!m) return null;"
                        "const r = m.getBoundingClientRect(); return {label: m.getAttribute('aria-label'), left: r.left, top: r.top}", 5)
        assert m["label"] == "Atom menu", (s, i, m)
        assert abs(m["left"] - a["x"]) <= 6 or m["left"] < a["x"], (s, i, a, m)      # it may flip left near the edge
        # closed again (closeAll() would close the document too)
        drv.execute_script("document.body.dispatchEvent(new PointerEvent('pointerdown', {bubbles: true, button: 0}))")
        H.wait(drv, "return !document.querySelector('.menu')", 5)
    assert not H.errors(drv), H.errors(drv)


@pytest.mark.parametrize("s", SCALES)
def test_3dmol_own_mouse_reads_the_canvas_px(drv, s):
    """The run and comparison viewers keep 3Dmol's own mouse handling: its
    pointer, in normalized device coordinates, is where the screen point is."""
    open_structure(drv)
    scale(drv, s)
    got = drv.execute_script("""const cv = document.querySelector('#viewer-main canvas'), V = cv._3dmol_viewer;
      const r = cv.getBoundingClientRect(), sx = r.left + r.width * 0.75, sy = r.top + r.height * 0.25;
      const e = {pageX: sx + scrollX, pageY: sy + scrollY};
      return V.mouseXY(V.getX(e), V.getY(e));""")
    assert abs(got["x"] - 0.5) < 0.01 and abs(got["y"] - 0.5) < 0.01, (s, got)


@pytest.mark.parametrize("s", SCALES)
def test_3d_canvas_draws_at_the_screen_resolution(drv, s):
    """The 3D canvas holds devicePixelRatio x zoom pixels per own px (above
    100 % it was drawn at 1x and stretched, so it looked soft), while its own
    size and 3Dmol's WIDTH stay in own px, devicePixelRatio is left as it was,
    and a click still selects the atom under the pointer."""
    open_structure(drv)
    scale(drv, s)
    w, h, cw, ch, vw, dpr, z = drv.execute_script("""const c = document.querySelector('#viewer-main canvas'), V = c._3dmol_viewer;
      const sh = document.getElementById('shell');
      return [c.offsetWidth, c.offsetHeight, c.width, c.height, V.WIDTH, window.devicePixelRatio,
              sh.getBoundingClientRect().width / sh.offsetWidth]""")
    assert abs(z - s / 100) < 0.01, (s, z)
    # the canvas truncates its size to whole pixels, and the measured zoom is rounded: 2 px of slack
    assert abs(cw - w * dpr * z) <= 2 and abs(ch - h * dpr * z) <= 2, (s, w, h, cw, ch, dpr, z)
    assert vw == w, (s, vw, w)
    i, a = targets(drv)[0]
    drv.execute_script("window.__mace.editor.select && window.__mace.editor.select([], 'replace')")
    click(drv, a["x"], a["y"])
    H.wait(drv, "return window.__mace.editor.selection().length > 0", 5)
    assert selection(drv) == [i], (s, i, a, selection(drv))
    assert not H.errors(drv), H.errors(drv)


@pytest.mark.parametrize("s", SCALES)
def test_plot_hover_is_the_point_under_the_pointer(drv, s):
    drv.execute_async_script("const d = arguments[1]; window.__mace.open(arguments[0]).then(() => d(1))", "results/mgo_band.BAND.DAT")
    H.wait(drv, "const v = window.__mace.plotView, s = v && v.state(); return s && s.drawn", 60)
    H.set_viewport(drv, 1280, 1100)                    # room for the plot above the bottom panel at 125 %
    scale(drv, s)
    time.sleep(0.5)
    pts = drv.execute_script("""const p = document.querySelector('#plot-host .pv-plot'), L = p._fullLayout, t = p.data[0];
      const r = p.getBoundingClientRect(), z = r.width / p.offsetWidth, out = [];
      for (let k = 1; k < t.x.length - 1; k++) {
        if (!Number.isFinite(t.x[k]) || !Number.isFinite(t.y[k])) continue;      // the gaps between path segments
        const x = r.left + (L.xaxis._offset + L.xaxis.l2p(t.x[k])) * z, y = r.top + (L.yaxis._offset + L.yaxis.l2p(t.y[k])) * z;
        const e = document.elementFromPoint(x, y);        // in sight: not under the bottom panel nor an axis's drag strip
        if (e && e.classList.contains('nsewdrag')) out.push([k, x, y]);
      }
      return out;""")
    assert len(pts) >= 2, (s, pts)
    for k, x, y in pts[:: max(1, len(pts) // 3)][:3]:
        move(drv, 5, 5)
        time.sleep(0.15)
        move(drv, x, y)
        # the hovered point is the one under the pointer (a band path repeats a k point:
        # its twin at the same place is as good)
        hov = H.wait(drv, """const p = document.querySelector('#plot-host .pv-plot'), L = p._fullLayout;
          const r = p.getBoundingClientRect(), z = r.width / p.offsetWidth;
          return p._hoverdata && p._hoverdata.length && p._hoverdata.map((h) => [h.pointNumber,
            r.left + (L.xaxis._offset + L.xaxis.l2p(h.x)) * z, r.top + (L.yaxis._offset + L.yaxis.l2p(h.y)) * z])""", 5)
        assert min(abs(hx - x) + abs(hy - y) for _, hx, hy in hov) <= 3, (s, k, x, y, hov)
    assert not H.errors(drv), H.errors(drv)


@pytest.mark.parametrize("s", SCALES)
def test_splitter_follows_the_pointer(drv, s):
    open_structure(drv)
    scale(drv, s)
    sp = drv.execute_script("const r = document.getElementById('split-explorer').getBoundingClientRect();"
                            "return [r.left + r.width / 2, r.top + r.height / 2]")
    drag(drv, [sp, (sp[0] + 20, sp[1]), (sp[0] + 40, sp[1])])
    time.sleep(0.3)
    after = drv.execute_script("const r = document.getElementById('split-explorer').getBoundingClientRect(); return r.left + r.width / 2")
    assert abs(after - (sp[0] + 40)) <= 2, (s, sp, after)
    drv.execute_script("document.getElementById('split-explorer').dispatchEvent(new MouseEvent('dblclick', {bubbles: true}))")


# where a floating element sits next to what opened it, in screen px (all rects from the page)
FLOATS = """const s = arguments[0], z0 = s / 100, done = arguments[arguments.length - 1], M = window.__mace;
const frames = () => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const zoom = (e) => e.getBoundingClientRect().width / e.offsetWidth;
const box = (e) => { const r = e.getBoundingClientRect(); return {left: r.left, right: r.right, top: r.top, bottom: r.bottom, z: zoom(e)}; };
const close = async () => { document.body.dispatchEvent(new PointerEvent('pointerdown', {bubbles: true, button: 0})); await frames(); };
const out = {W: innerWidth, H: innerHeight};
const kind = arguments[1];
(async () => {
  if (kind === 'plot') {
    for (const [sel, pop] of [['.pk-units', '.pk-menu'], ['.pk-export', '.pk-menu'], ['.pk-more', '.pk-pop']]) {
      const b = document.querySelector('#plot-host ' + sel);
      if (!b) continue;
      b.click(); await frames();
      const p = document.querySelector(pop);
      out[sel] = {btn: box(b), pop: p && box(p)};
      await close();
    }
  } else {
    // a menu bar menu under its button, the 3D view's menu in the window's corner, the periodic table
    const b = document.querySelector('.menu-top[data-menu="0"]');
    b.dispatchEvent(new PointerEvent('pointerdown', {bubbles: true, button: 0})); await frames();
    const m = [...document.querySelectorAll('.menu')].pop();
    out.top = {btn: box(b), pop: box(m), maxH: parseFloat(getComputedStyle(m).maxHeight)};
    await close();
    M.contextFor('empty', null, innerWidth - 2, innerHeight - 2); await frames();
    out.corner = {pop: box(document.querySelector('.menu[aria-label="Viewport menu"]'))};
    await close();
    M.run('select.all'); await frames();
    M.run('edit.replaceElement');
    for (let i = 0; i < 100 && !document.querySelector('.ptable'); i++) await sleep(30);
    await frames();
    out.ptable = {pop: box(document.querySelector('.ptable'))};
    document.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', bubbles: true}));
    document.querySelector('.ptable') && document.querySelector('.ptable').dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', bubbles: true}));
    await close();
    M.run('select.none');
  }
  done(out);
})().catch((e) => done({error: String(e && e.stack || e)}));"""


def _inside(o, pop, s):
    assert pop["left"] >= -0.5 and pop["top"] >= -0.5 and pop["right"] <= o["W"] + 0.5 and pop["bottom"] <= o["H"] + 0.5, (s, o)


@pytest.mark.parametrize("s", (100, 110, 125))
def test_plot_menus_and_options_follow_the_ui_scale(drv, s):
    """A plot's Units and Export menus and its Plot Options popover live in the overlay
    layer outside the zoomed shell: they take the UI scale too, right-aligned under
    their button (a 4 px gap, zoomed), inside the window."""
    drv.execute_async_script("const d = arguments[1]; window.__mace.open(arguments[0]).then(() => d(1))", "results/mgo_band.BAND.DAT")
    H.wait(drv, "const v = window.__mace.plotView, s = v && v.state(); return s && s.drawn", 60)
    scale(drv, s)
    time.sleep(0.3)
    o = drv.execute_async_script(FLOATS, s, "plot")
    assert "error" not in o, o
    seen = [k for k in (".pk-units", ".pk-export", ".pk-more") if k in o]
    assert ".pk-export" in seen and ".pk-more" in seen, o
    z = s / 100
    for k in seen:
        b, p = o[k]["btn"], o[k]["pop"]
        assert p, (s, k, o)
        assert abs(b["z"] - z) < 0.01 and abs(p["z"] - z) < 0.01, (s, k, b, p)
        assert abs(p["top"] - (b["bottom"] + 4 * z)) <= 1.5, (s, k, b, p)
        assert abs(p["right"] - b["right"]) <= 1.5 or p["left"] <= 4 * z + 1.5, (s, k, b, p)
        _inside(o, p, s)
    assert not H.errors(drv), H.errors(drv)


@pytest.mark.parametrize("s", (100, 110, 125))
def test_app_menus_and_picker_follow_the_ui_scale(drv, s):
    """The menus and the periodic table share that layer: zoomed like the shell, a menu bar
    menu under its button, the 3D view's menu opened in the window's corner flipped up
    and kept inside it."""
    open_structure(drv)
    scale(drv, s)
    o = drv.execute_async_script(FLOATS, s, "app")
    assert "error" not in o, o
    z = s / 100
    b, p = o["top"]["btn"], o["top"]["pop"]
    assert abs(p["z"] - z) < 0.01, (s, o)
    assert abs(p["left"] - b["left"]) <= 1.5 and abs(p["top"] - (b["bottom"] + 2 * z)) <= 1.5, (s, b, p)
    # a menu is never taller than the window: its own max-height, zoomed, is the window's less 48 px (zoomed)
    assert abs(o["top"]["maxH"] * z - (o["H"] - 48 * z)) <= 2, (s, o["top"], o["H"])
    c = o["corner"]["pop"]
    assert abs(c["z"] - z) < 0.01, (s, c)
    assert abs(c["bottom"] - (o["H"] - 4 * z)) <= 1.5, (s, o)             # flipped above the point, 4 px (zoomed) from the edge
    assert abs(c["right"] - (o["W"] - 4 * z)) <= 1.5, (s, o)              # shifted left, 4 px (zoomed) from the edge
    _inside(o, c, s)
    t = o["ptable"]["pop"]
    assert abs(t["z"] - z) < 0.01, (s, t)
    _inside(o, t, s)
    assert not H.errors(drv), H.errors(drv)
