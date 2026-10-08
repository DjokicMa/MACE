"""No drawn atom is ever cut by the clipping planes, and the depth cueing is
subtle by default (polish A2).

3Dmol set its near and far planes only at the first fit, so a larger display
range, a pan or a zoom cut atoms off, and its default fog (from 40% of the
slab) washed out most of a large cell. viewport/clip.js now sets the planes
from what is drawn before every paint. These tests read the camera's real
near/far and fog after a paint (not a re-derivation of 3Dmol's conventions),
over 50 random rotations, a pan, a zoom in and a zoom out, for the structure,
OPT, compare and normal-mode viewers, with real CRYSTAL data from test/.
"""
import time

import pytest

import guiharness as H
import realdata

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")                 # the server reads the structures with it

from selenium.webdriver.common.action_chains import ActionChains  # noqa: E402

FREQ = "FREQ/2_dia2_opt_rev1_freq_B3LYP-D3-D3_optimized.out"

# For a viewer: every drawn atom's depth interval [d - r, d + r] against the
# camera's [near, far] after one synchronous paint, at the current pose.
CHECK = r"""
const v = window.MACE.state.viewers[arguments[0]];
if (!v.paintNow()) return {error: 'not visible'};
const view = v.getView(), cam = v.camera;
const e = cam.position.z - v.rotationGroup.position.z;
const [qx, qy, qz, qw] = view.slice(4, 8);
const rz = (x, y, z) => { x += view[0]; y += view[1]; z += view[2];
  const ix = qw*x + qy*z - qz*y, iy = qw*y + qz*x - qx*z, iz = qw*z + qx*y - qy*x, iw = -qx*x - qy*y - qz*z;
  return iz*qw + iw*-qz + ix*-qy - iy*-qx; };
const vdw = window.$3Dmol.GLModel.vdwRadii;
const rad = (a) => { const s = a.style || {}; if (s.sphere) return s.sphere.radius || (s.sphere.scale || 1) * (vdw[a.elem] || 1.6);
  return s.stick ? (s.stick.radius || 0.25) : 0.3; };
let n = 0, clipped = 0, dmin = Infinity, dmax = -Infinity, far = null;
for (const m of v.getModelList()) for (const a of m.selectedAtoms({})) {
  const d = e - rz(a.x, a.y, a.z), r = rad(a);
  if (!cam.ortho && d - r < 0) continue;            // behind a perspective camera: not drawn anyway
  n++;
  // a perspective near plane cannot go below 1: an atom crossing it is zoomed into, not clipped by our planes
  const nearOk = d - r >= cam.near - 1e-6 || (!cam.ortho && cam.near <= 1 + 1e-6);
  if (!nearOk || d + r > cam.far + 1e-6) clipped++;
  if (d < dmin) dmin = d;
  if (d > dmax) { dmax = d; far = r; }
}
const fog = v.scene.fog, ss = (a, b, x) => { if (a === b) return 0; const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
return {n, clipped, near: cam.near, far: cam.far, fogNear: fog.near, fogFar: fog.far, disabled: !!v.config.disableFog,
  blendFar: ss(fog.near, fog.far, dmax + far), blendMax: ss(fog.near, fog.far, dmax), last: v.maceClip ? v.maceClip.last : null};
"""

# deterministic pseudo-random unit quaternions (as the diagnosis probe used)
POSES = r"""
const v = window.MACE.state.viewers[arguments[0]], k = arguments[1];
const f = (x) => Math.abs(x % 1);
const u1 = f(Math.sin(k * 12.9898) * 43758.5453), u2 = f(Math.sin(k * 78.233) * 12345.678), u3 = f(Math.sin(k * 39.425) * 9876.54);
const q = [Math.sqrt(1 - u1) * Math.sin(2 * Math.PI * u2), Math.sqrt(1 - u1) * Math.cos(2 * Math.PI * u2),
  Math.sqrt(u1) * Math.sin(2 * Math.PI * u3), Math.sqrt(u1) * Math.cos(2 * Math.PI * u3)];
const w = v.getView(); v.setView([...w.slice(0, 4), ...q]);"""

# move the model in the screen plane by a fraction of the viewport width (what a pan does)
PAN = r"""
const v = window.MACE.state.viewers[arguments[0]], frac = arguments[1];
const w = v.getView(), [x, y, z, ww] = [-w[4], -w[5], -w[6], w[7]];
const rot = (p) => { const [px, py, pz] = p; const ix = ww*px + y*pz - z*py, iy = ww*py + z*px - x*pz, iz = ww*pz + x*py - y*px, iw = -x*px - y*py - z*pz;
  return [ix*ww + iw*-x + iy*-z - iz*-y, iy*ww + iw*-y + iz*-x - ix*-z, iz*ww + iw*-z + ix*-y - iy*-x]; };
const cam = v.camera, width = cam.ortho ? (cam.right - cam.left) : 2 * (cam.position.z - v.rotationGroup.position.z) * Math.tan(Math.PI / 360 * cam.fov);
const d = rot([frac * width, 0, 0]);
v.setView([w[0] + d[0], w[1] + d[1], w[2] + d[2], ...w.slice(3)]);"""


def check(drv, vid):
    r = drv.execute_script(CHECK, vid)
    assert "error" not in r, r
    return r


def sweep(drv, vid, case, poses=50):
    """50 rotations, then a pan by 40% of the view, a zoom in x4 and out x0.25: no atom clipped."""
    bad = []
    for k in range(poses):
        drv.execute_script(POSES, vid, k + 1)
        r = check(drv, vid)
        if r["clipped"]:
            bad.append(("rot", k, r["clipped"], r["n"]))
    base = drv.execute_script("return window.MACE.state.viewers[arguments[0]].getView()", vid)
    for what, js in (("pan", (PAN, 0.4)), ("zoom-in", ("window.MACE.state.viewers[arguments[0]].zoom(4)",)),
                     ("zoom-out", ("window.MACE.state.viewers[arguments[0]].zoom(0.0625)",))):
        drv.execute_script(js[0], vid, *js[1:])
        r = check(drv, vid)
        if r["clipped"]:
            bad.append((what, r["clipped"], r["n"]))
    drv.execute_script("window.MACE.state.viewers[arguments[0]].setView(arguments[1])", vid, base)
    assert not bad, (case, bad)
    assert r["n"] > 0


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("clip")
    root = H.examples_copy(tmp)
    if not H.remote():
        realdata.copy(root / "real", "agte.cif", "molecule.cif", "slab_opt.out",
                      extra={"freq.out": realdata.first(FREQ)})
    proc, base, _ = H.start_server(root, tmp / "xdg")
    drv = H.firefox(1280, 800)
    try:
        H.load(drv, base)
        yield drv, base, root
    finally:
        drv.quit()
        proc.terminate()
        proc.wait(timeout=15)


@pytest.fixture
def fresh(app):
    drv = app[0]
    drv.execute_script("window.__mace.closeAll(); window.__mace.settings.resetAll(); window.__mace.run('view.range1');"
                       "document.querySelectorAll('#toasts > *').forEach((t) => t.remove());")
    time.sleep(0.3)
    return app


def open_doc(drv, path, kind):
    got = drv.execute_async_script(
        "const done = arguments[1]; window.__mace.open(arguments[0]).then((d) => done(d && d.kind))", path)
    assert got == kind
    time.sleep(0.8)


def real_click(drv, el):
    ActionChains(drv).move_to_element(el).pause(0.05).click().perform()


def set_range(drv, n):
    """The reported path: a real click on the toolbar's range menu."""
    real_click(drv, drv.find_element("css selector", "#tb-range"))
    H.wait(drv, "return !!document.querySelector('.menu')", 3)
    item = drv.find_element("css selector", f'.menu [data-cmd-id="view.range{n}"]')
    w = item.size["width"]
    ActionChains(drv).move_to_element_with_offset(item, int(-w / 2 + 16), 0).pause(0.05).click().perform()
    H.wait(drv, "return document.getElementById('tb-range-label').textContent === arguments[0]", 5, f"{n}×{n}×{n}")
    time.sleep(0.6)


def test_structure_range_1_and_3(fresh):
    drv = fresh[0]
    open_doc(drv, "real/agte.cif", "structure")
    sweep(drv, "viewer-main", "agte r1")
    # the reported path: fit at 1x1x1, then 3x3x3 from the toolbar, then rotate
    set_range(drv, 3)
    sweep(drv, "viewer-main", "agte r3")
    assert not H.errors(drv), H.errors(drv)


def test_first_frame_after_a_range_change_and_a_pan(fresh):
    """One paint after the change already has every atom inside: the planes
    are set before 3Dmol turns them into the camera's near/far."""
    drv = fresh[0]
    open_doc(drv, "real/agte.cif", "structure")
    set_range(drv, 3)
    r = check(drv, "viewer-main")                 # check() paints exactly once, then reads
    assert r["clipped"] == 0 and r["n"] > 300, r
    drv.execute_script(PAN, "viewer-main", 0.4)
    r = check(drv, "viewer-main")
    assert r["clipped"] == 0, r


def test_molecule(fresh):
    drv = fresh[0]
    open_doc(drv, "real/molecule.cif", "structure")
    sweep(drv, "viewer-main", "molecule")


def test_opt_and_compare_viewers(fresh):
    drv = fresh[0]
    open_doc(drv, "real/slab_opt.out", "trajectory")
    H.wait(drv, "return !!(window.MACE.state.traj && window.MACE.state.viewers['viewer-before'])", 60)
    n = drv.execute_script("return window.MACE.state.traj.frames.length")
    for step in sorted({0, n // 2, n - 1}):
        drv.execute_script("const el = document.getElementById('o-frame'); if (el) { el.value = String(arguments[0]); el.dispatchEvent(new Event('input')); }", step)
        H.wait(drv, "return !!window.MACE.state.traj.frames[arguments[0]]", 60, step)
        time.sleep(0.5)
        sweep(drv, "viewer-before", f"slab step {step}", poses=20)
    # the comparison, side by side: start on the left, the step on the right
    drv.execute_script("window.MACE.setLook('side')")
    H.wait(drv, "return !!window.MACE.state.viewers['viewer-after']", 10)
    time.sleep(0.6)
    sweep(drv, "viewer-after", "compare side", poses=20)
    sweep(drv, "viewer-before", "compare side (left)", poses=20)
    drv.execute_script("window.MACE.setLook('overlay')")


def test_freq_viewer_at_the_extreme_frame(fresh):
    drv = fresh[0]
    drv.execute_async_script("const done = arguments[1]; window.__mace.openModes(arguments[0]).then(() => done(1))", "real/freq.out")
    H.wait(drv, "const v = window.__mace.freqView, s = v && v.state(); return s && s.mode", 60)
    n = drv.execute_script("return window.__mace.freqView.state().n")
    drv.execute_async_script("const done = arguments[1]; window.__mace.freqView.select(arguments[0]).then(() => done(1))", n - 1)
    drv.execute_script("window.__mace.freqView.play(false)")
    time.sleep(0.5)
    # frame 6 of 24 is sin(pi/2): the largest displacement
    drv.execute_script("const v = window.MACE.state.viewers['viewer-freq']; v.setFrame(6)")
    time.sleep(0.3)
    sweep(drv, "viewer-freq", "freq highest mode", poses=30)


def test_depth_cueing_default_off_and_full(fresh):
    drv = fresh[0]
    open_doc(drv, "real/agte.cif", "structure")
    set_range(drv, 3)
    r = check(drv, "viewer-main")
    assert not r["disabled"]
    # subtle by default: the farthest atom's far side blends at most 20% toward the background
    assert r["blendFar"] <= 0.2, r
    drv.execute_script("window.__mace.settings.set('depthCue', 0)")
    r = check(drv, "viewer-main")
    assert r["disabled"] and r["blendFar"] == 0, r
    drv.execute_script("window.__mace.settings.set('depthCue', 1)")
    r = check(drv, "viewer-main")
    last = r["last"]
    assert not r["disabled"] and abs(r["fogNear"] - last["front"]) < 1e-6 and abs(r["fogFar"] - last["back"]) < 1e-6, r
    assert r["blendFar"] > 0.5, r
    drv.execute_script("window.__mace.settings.set('depthCue', 0.15)")


def test_large_cell_corners_are_drawn(fresh):
    """A screenshot check that does not trust the planes math: AgTe at range
    3 seen along a body diagonal has atoms in all four quadrants."""
    drv = fresh[0]
    open_doc(drv, "real/agte.cif", "structure")
    set_range(drv, 3)
    drv.execute_script("const v = window.MACE.state.viewers['viewer-main']; const w = v.getView();"
                       "v.setView([...w.slice(0, 4), 0.2798, 0.1159, 0.3647, 0.8806]); v.paintNow();")
    time.sleep(0.5)
    counts = drv.execute_script("""const v = window.MACE.state.viewers['viewer-main']; v.paintNow();
        const cv = v.getCanvas(), url = cv.toDataURL('image/png');
        return new Promise((ok) => { const img = new Image(); img.onload = () => {
          const c = document.createElement('canvas'); c.width = cv.width; c.height = cv.height;
          const g = c.getContext('2d'); g.drawImage(img, 0, 0);
          const d = g.getImageData(0, 0, c.width, c.height).data, bg = [d[0], d[1], d[2]], W = c.width, H = c.height, q = [0, 0, 0, 0];
          for (let y = 0; y < H; y += 2) for (let x = 0; x < W; x += 2) { const k = 4 * (y * W + x);
            if (Math.abs(d[k] - bg[0]) + Math.abs(d[k + 1] - bg[1]) + Math.abs(d[k + 2] - bg[2]) > 30) q[(y < H / 2 ? 0 : 2) + (x < W / 2 ? 0 : 1)]++; }
          ok(q); }; img.src = url; });""")
    assert all(c > 0 for c in counts), counts


def test_clip_costs_almost_nothing_per_paint(fresh):
    """The wrapper does O(1) work per paint: the bounds are kept until a model or shape changes."""
    drv = fresh[0]
    open_doc(drv, "real/agte.cif", "structure")
    set_range(drv, 3)
    us = drv.execute_script("""const v = window.MACE.state.viewers['viewer-main'];
        v.setSlabAndFog();
        const proto = Object.getPrototypeOf(v).setSlabAndFog.bind(v);
        const t = (f) => { const t0 = performance.now(); for (let k = 0; k < 20000; k++) f(); return (performance.now() - t0) / 20000 * 1000; };
        const a = t(() => proto()), b = t(() => v.setSlabAndFog());
        return [a, b];""")
    assert us[1] - us[0] < 20, us        # microseconds per paint on top of 3Dmol's own
