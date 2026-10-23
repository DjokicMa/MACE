"""1.1.6 cube views, print patterns and animation export, in the real GUI (headless Firefox).

* A cube's atoms follow the Inspector's Display sliders (Atom size, Bond radius): before 1.1.6 the
  volume view drew fixed small balls, and only a style preset changed them.
* The Slice section: a Miller plane from the common-plane chips, the outline the page draws equals
  the polygon the server samples (one geometry), the mini unit-cell preview moves the plane along its
  normal when dragged and turns it with Shift, an Alt-drag in the 3D view moves it too, and the map
  shows the full grid's plane once the plane stops.
* The plane stays legible on a White background in the dark theme (and the reverse): its outline is at
  least 3:1 on the drawn background.
* Print patterns: a layer over the structure view with a pattern per element ("all"), none when off,
  and Export Image draws it.
* Export Animation: a GIF of the FREQ view's mode (one cycle, 24 frames) and, rendered by the server,
  of an optimization's steps.

The FREQ and OPT outputs are real CRYSTAL23 runs of the test/ corpus (those tests skip without it); the
cube is a small synthetic 3D-periodic one (the corpus has slabs and molecules only).
"""
import shutil
import time

import pytest

import sys

import guiharness as H
import p4util as U

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("plotly", reason="the cube engine imports plotly")
np = pytest.importorskip("numpy")
PIL = pytest.importorskip("PIL.Image")

from selenium.webdriver.common.action_chains import ActionChains  # noqa: E402
from selenium.webdriver.common.keys import Keys  # noqa: E402

CORPUS = H.REPO_ROOT / "test"
FREQ = CORPUS / "FREQ" / "EC_MOLECULE_OPT_symm_HSESOL3C_SOLDEF2MSVP_opt_HSESOL3C_optimized_freq_HSESOL3C_optimized.out"
OPT = CORPUS / "OPT" / "2_dia2_opt_rev1.out"
CELL = (8.0, 8.0, 12.0)            # bohr
sys.path.insert(0, str(H.REPO_ROOT / "tests"))
from test_gui_results import write_band_path  # noqa: E402
GRAPHITE = "GAMMA-M-K-GAMMA-A-L-H-A-|-L-M-|-H-K"
TICKS = (0.0, 0.78265, 1.20908, 2.05632, 2.32554, 3.10818, 3.53462, 4.38186, 4.65597, 4.93007)
HEX_POINTS = {"GAMMA": (0, 0, 0), "M": (0.5, 0, 0), "K": (1 / 3, 1 / 3, 0), "A": (0, 0, 0.5), "L": (0.5, 0, 0.5),
              "H": (1 / 3, 1 / 3, 0.5)}
GRAPHITE_OUT = """ CRYSTAL - PROPERTIES
 DIRECT LATTICE VECTOR COMPONENTS (ANGSTROM)
         2.12415   -1.22638    0.00000
        -0.00000    2.45276    0.00000
         0.00000    0.00000    6.17522
"""


def write_bulk_cube(path):
    """A 3D-periodic CRYSTAL-style cube (the cell on the second comment line), two atoms, a smooth density."""
    n = (16, 16, 24)
    step = [CELL[k] / n[k] for k in range(3)]
    atoms = [(8, (0.5, 0.5, 0.4)), (12, (0.5, 0.5, 0.65))]
    g = np.indices(n).reshape(3, -1).T * step
    rho = np.zeros(len(g))
    for _, f in atoms:
        c = np.array(f) * CELL
        rho += np.exp(-((g - c) ** 2).sum(1) / (2 * 0.8 ** 2))
    lines = [" Charge density - 3D GRID - GAUSSIAN CUBE FORMAT bulk",
             "  {:.8f}  {:.8f}  {:.8f}  90.000000  90.000000  90.000000".format(*CELL),
             "{:5d}  0.000000  0.000000  0.000000".format(len(atoms))]
    lines += ["{:5d}  {:.6f}  {:.6f}  {:.6f}".format(n[k], *[step[k] if j == k else 0.0 for j in range(3)]) for k in range(3)]
    lines += ["{:5d}  {:.6f}  {:.6f}  {:.6f}  {:.6f}".format(z, float(z), *(np.array(f) * CELL)) for z, f in atoms]
    vals = rho.reshape(n)
    for i in range(n[0]):
        for j in range(n[1]):
            row = vals[i, j]
            lines += [" ".join("{:.5E}".format(x) for x in row[k:k + 6]) for k in range(0, n[2], 6)]
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("a03cube")
    root = H.examples_copy(tmp)
    (root / "vol").mkdir()
    write_bulk_cube(root / "vol" / "bulk_DENS.cube")
    mol = CORPUS / "ECH3POT3" / "DEC_MOLECULE_OPT_symm_HSESOL3C_SOLDEF2MSVP_opt_HSESOL3C_optimized_sp_HSESOL3C_optimized_charge+potential_DENS.CUBE"
    if mol.is_file():
        shutil.copyfile(mol, root / "vol" / "mol_DENS.CUBE")
    (root / "gr").mkdir()
    write_band_path(root / "gr", stem="g_band", path=GRAPHITE, ticks=TICKS, points=HEX_POINTS)
    (root / "gr" / "g_band.out").write_text(GRAPHITE_OUT)
    (root / "runs").mkdir()
    for f in (FREQ, OPT):
        if f.is_file():
            shutil.copyfile(f, root / "runs" / f.name)
    a = U.start_app(tmp, root=root, MACE_GUI_MAX_WORKERS="2")
    try:
        a[0].execute_script("window.__mace.settings.set('reduceMotion', 'off'); window.__mace.settings.set('volume.quality', 'low')")
        yield a
    finally:
        U.stop_app(a)


@pytest.fixture
def drv(app, request):
    failed = request.session.testsfailed
    d = app[0]
    yield d
    if request.session.testsfailed > failed:
        H.save_failure(d, request.node.name)
    d.execute_script("""window.__mace.closeAll(); const s = window.__mace.settings;
      s.set('theme', 'light'); s.set('background', 'theme'); s.set('viewport.printPatterns', 'off');
      s.set('displayLo', [0, 0, 0]); s.set('displayHi', [1, 1, 1]);
      document.querySelectorAll('dialog[open]').forEach((x) => x.close());""")
    time.sleep(0.3)


def open_vol(drv, path):
    kind = drv.execute_async_script("const d = arguments[1]; window.__mace.open(arguments[0]).then((x) => d(x && x.kind), (e) => d('error: ' + e.message))", path)
    assert kind == "volume", kind
    H.wait(drv, "const s = window.__mace.volumeView && window.__mace.volumeView.state(); return !!(s && s.path === arguments[0] && s.surfaces > 0 && s.atomStyle)", 90, path)


def st(drv):
    return drv.execute_script("return window.__mace.volumeView.state()")


def lum(h):
    c = [int(h[k:k + 2], 16) / 255 for k in (1, 3, 5)]
    c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def ratio(a, b):
    x, y = lum(a), lum(b)
    return (max(x, y) + 0.05) / (min(x, y) + 0.05)


def hexof(drv, css):
    """A CSS color as #rrggbb (through a canvas)."""
    return drv.execute_script("""const c = document.createElement('canvas').getContext('2d'); c.fillStyle = '#000'; c.fillStyle = arguments[0];
      const v = c.fillStyle; if (v.startsWith('#')) return v; const m = v.match(/\\d+/g); return '#' + m.slice(0, 3).map((x) => (+x).toString(16).padStart(2, '0')).join('');""", css)


def test_cube_atoms_follow_the_inspector_sliders(drv):
    open_vol(drv, "vol/bulk_DENS.cube")
    s0 = st(drv)["atomStyle"]
    assert "sphere" in s0 and "stick" in s0, s0
    # the Inspector's Display section writes the document's own Atom size and Bond radius
    found = drv.execute_script("""const set = (id, v) => { const el = document.getElementById(id); if (!el) return false;
        el.value = String(v); el.dispatchEvent(new Event('input', {bubbles: true})); el.dispatchEvent(new Event('change', {bubbles: true})); return true; };
      return [set('dp-atomScale', 0.8), set('dp-bondRadius', 0.24)];""")
    if not all(found):
        # the Display section is not shown for this layout: the same per-document keys through Settings' defaults
        drv.execute_script("window.__mace.settings.set('atomScale', 0.8); window.__mace.settings.set('bondRadius', 0.24)")
    H.wait(drv, """const a = window.__mace.volumeView.state().atomStyle;
      return a && Math.abs(a.sphere.scale - 2 * arguments[0]) < 1e-6 && Math.abs(a.stick.radius - 0.24) < 1e-6""", 20, s0["sphere"]["scale"])
    drv.execute_script("window.__mace.settings.set('atomScale', 0.4); window.__mace.settings.set('bondRadius', 0.12)")
    assert not H.errors(drv), H.errors(drv)


def test_miller_plane_preview_drag_turn_and_full_grid_map(drv):
    open_vol(drv, "vol/bulk_DENS.cube")
    drv.execute_script("document.querySelector('[data-vo-plane=\"hkl:1,1,0\"]').click()")
    H.wait(drv, """const s = window.__mace.volumeView.state(); return s.slice && s.slice.mode === 'hkl' && s.map && !s.map.preview
      && s.plane && s.plane.shapes >= 4""", 60)
    s = st(drv)
    assert s["slice"]["ints"] == [1, 1, 0] and abs(s["slice"]["pos"] - 0.5) < 1e-9
    # one geometry: the outline drawn in the page is the polygon the server sampled
    a, b = np.array(s["plane"]["polygon"]), np.array(s["slice"]["polygon"])
    assert a.shape == b.shape and a.shape[0] >= 4, (a, b)
    assert np.abs(a - b).max() < 1e-4, (a, b)
    # (110) of a tetragonal box: the normal is along x + y
    n = np.array(s["plane"]["normal"])
    assert abs(abs(n[0]) - abs(n[1])) < 1e-9 and abs(n[2]) < 1e-9
    # a drag along the plane's normal on the preview moves it; the map follows (a preview, then the full grid)
    cv = drv.find_element("id", "vo-preview")
    drv.execute_script("arguments[0].scrollIntoView({block: 'center'})", cv)
    ns = drv.execute_script("return window.__mace.volumeView.previewNormal()")
    assert ns and ns["len"] > 0, ns
    w, h = drv.execute_script("return [arguments[0].clientWidth, arguments[0].clientHeight]", cv)
    ux, uy = ns["sx"] / ns["len"], ns["sy"] / ns["len"]
    H.drag(drv, cv, [(w / 2, h / 2), (w / 2 + 30 * ux, h / 2 + 30 * uy)])
    H.wait(drv, "const s = window.__mace.volumeView.state(); return s.plane.pos > 0.55 && s.slice.pos > 0.55 && s.map && !s.map.preview", 60)
    # Shift-drag turns it (the Untilt button appears enabled), and Untilt squares it again
    H.drag(drv, cv, [(w / 2, h / 2), (w / 2 + 40, h / 2 - 20)], keys=(Keys.SHIFT,))
    H.wait(drv, "const s = window.__mace.volumeView.state(); return s.slice.tilt !== 0 && s.slice.twist !== 0 && !s.map.preview", 60)
    drv.execute_script("document.querySelector('[data-vo=\"planeSquare\"]').click()")
    H.wait(drv, "const s = window.__mace.volumeView.state(); return s.slice.tilt === 0 && s.slice.twist === 0", 60)
    # Alt-drag in the 3D view moves the plane too (and does not turn the camera)
    view0 = drv.execute_script("return window.MACE.state.viewers['viewer-volume'].getView()")
    pos0 = st(drv)["plane"]["pos"]
    vw = drv.find_element("id", "viewer-volume")
    vw_w, vw_h = drv.execute_script("return [arguments[0].clientWidth, arguments[0].clientHeight]", vw)
    H.drag(drv, vw, [(vw_w / 2, vw_h / 2), (vw_w / 2 + 60, vw_h / 2 - 60)], keys=(Keys.ALT,))
    H.wait(drv, "return Math.abs(window.__mace.volumeView.state().plane.pos - arguments[0]) > 0.01", 20, pos0)
    view1 = drv.execute_script("return window.MACE.state.viewers['viewer-volume'].getView()")
    assert np.allclose(view0[4:8], view1[4:8], atol=1e-9), (view0, view1)
    head = drv.execute_script("return document.querySelector('#volume-host .vv-slice-head').textContent")
    assert head.startswith("(1 1 0) at ") and "h x + k y + l z" in head, head
    # a lattice plane of the grid still works as before (exact grid planes, arrow keys)
    drv.execute_script("const s = document.getElementById('vo-axis'); s.value = 'c'; s.dispatchEvent(new Event('change', {bubbles: true}))")
    H.wait(drv, "const s = window.__mace.volumeView.state(); return s.slice && s.slice.axis === 'c' && s.slice.index === 12", 60)
    assert not H.errors(drv), H.errors(drv)


@pytest.mark.parametrize("theme, background", [("dark", "white"), ("light", "black")])
def test_slice_plane_stays_legible_on_a_forced_background(drv, theme, background):
    drv.execute_script("const s = window.__mace.settings; s.set('theme', arguments[0]); s.set('background', arguments[1])", theme, background)
    open_vol(drv, "vol/bulk_DENS.cube")
    drv.execute_script("document.getElementById('vo-slice').click()")
    H.wait(drv, "const s = window.__mace.volumeView.state(); return s.plane && s.plane.shapes >= 4", 60)
    c = st(drv)["plane"]["colors"]
    edge, fill, bg = hexof(drv, c["edge"]), hexof(drv, c["fill"]), hexof(drv, c["bg"])
    assert ratio(edge, bg) >= 3, (edge, bg)
    assert ratio(fill, bg) >= 2, (fill, bg)
    # a custom background of any lightness: the outline is moved until it stands out
    out = drv.execute_script("""return import(document.querySelector('script[src*="/js/shell.js"]').src.replace('shell.js', 'viewport/contrast.js'))
      .then((m) => ['#777777', '#8a8a8a', '#2b2b2b'].map((bg) => [bg, m.legible('#8aa4d6', bg)]))""")
    for bg, col in out:
        assert ratio(hexof(drv, col), bg) >= 3, (bg, col)
    assert not H.errors(drv), H.errors(drv)


def test_print_patterns_layer_and_export_image(drv):
    assert U.open_doc(drv, "structures/MgO.cif") == "structure"
    time.sleep(0.5)
    drv.execute_script("window.__mace.settings.set('viewport.printPatterns', 'all')")
    H.wait(drv, "const p = window.__macePatterns; const s = p && p.stateOf('viewer-main'); return s && s.patterned > 0 && Object.keys(s.map).length === 2", 30)
    m = drv.execute_script("return window.__macePatterns.stateOf('viewer-main').map")
    assert set(m) == {"Mg", "O"} and len(set(m.values())) == 2, m
    # the same rule as mace animate: "auto" patterns only elements of a similar gray
    same = drv.execute_script("return window.__macePatterns.assign(['A', 'A', 'B', 'C'], (s) => ({A: '#808080', B: '#828282', C: '#202020'})[s], 'auto')")
    assert same == {"B": "slash"}, same
    from mace.gui import animexport as AX
    assert AX.assign_patterns(["A", "A", "B", "C"], lambda s: {"A": (128, 128, 128), "B": (130, 130, 130), "C": (32, 32, 32)}[s], "auto") == same
    # Export Image draws the layer into the PNG (it differs from the one without patterns)
    def png(patterns):
        drv.execute_script("window.__maceExportHook = (u) => { window.__lastExport = u; }; window.__lastExport = null; window.__mace.run('file.exportImage')")
        d = H.wait(drv, "const d = document.getElementById('export-image'); return d && d.open && d", 10)
        drv.execute_script("const s = arguments[0].querySelector('[data-k=patterns]'); s.value = arguments[1]; s.dispatchEvent(new Event('change', {bubbles: true}))", d, patterns)
        drv.execute_script("arguments[0].querySelector('[data-v=export]').click()", d)
        return H.wait(drv, "return window.__lastExport", 30)
    with_p, without = png("all"), png("off")
    assert with_p != without
    drv.execute_script("window.__mace.settings.set('viewport.printPatterns', 'off')")
    H.wait(drv, "return window.__macePatterns.stateOf('viewer-main').patterned === 0", 20)
    assert not H.errors(drv), H.errors(drv)


def _export_animation(drv, source=None, fmt="gif"):
    drv.execute_script("window.__animDone = null; window.__maceAnimHook = (r) => { window.__animDone = r; }; window.__mace.run('file.exportAnimation')")
    d = H.wait(drv, "const d = document.getElementById('export-animation'); return d && d.open && d", 20)
    if source:
        drv.execute_script("const s = arguments[0].querySelector('[data-k=source]'); s.value = arguments[1]; s.dispatchEvent(new Event('change', {bubbles: true}))", d, source)
    drv.execute_script("""const q = (k) => arguments[0].querySelector('[data-k=' + k + ']');
      q('format').value = arguments[1]; q('size').value = 'custom'; q('w').value = '320'; q('h').value = '240';
      q('format').dispatchEvent(new Event('change', {bubbles: true})); q('size').dispatchEvent(new Event('change', {bubbles: true}));
      arguments[0].querySelector('[data-v=export]').click()""", d, fmt)
    return H.wait(drv, "return window.__animDone || (document.querySelector('#export-animation .err') || {}).textContent || null", 180)


def test_export_animation_gif_of_a_normal_mode(drv):
    if not FREQ.is_file():
        pytest.skip("test/ corpus not present (gitignored)")
    drv.execute_async_script("const done = arguments[1]; window.__mace.openModes(arguments[0]).then(() => done(1))", f"runs/{FREQ.name}")
    H.wait(drv, "const v = window.__mace.freqView, s = v && v.state(); return s && s.mode && s.display", 60)
    r = _export_animation(drv)
    assert isinstance(r, dict), r
    assert r["format"] == "gif" and r["frames"] == 24 and r["size"] == [320, 240] and r["type"] == "image/gif", r
    caps = drv.execute_async_script("const d = arguments[0]; window.MACE.api('anim/caps').then(d)")
    if caps["webm"]:
        r = _export_animation(drv, fmt="webm")
        assert isinstance(r, dict) and r["format"] == "webm" and r["type"] == "video/webm", r
    assert not H.errors(drv), H.errors(drv)


def test_export_animation_of_a_mode_rendered_by_the_server_keeps_the_views_scale(drv):
    """Physical ×20 in the view: the server draws the same scale (it read ×20 as 0.5 Å × 20 and
    refused amplitudes above 10)."""
    if not FREQ.is_file():
        pytest.skip("test/ corpus not present (gitignored)")
    drv.execute_async_script("const done = arguments[1]; window.__mace.openModes(arguments[0]).then(() => done(1))", f"runs/{FREQ.name}")
    H.wait(drv, "const v = window.__mace.freqView, s = v && v.state(); return s && s.mode && s.display", 60)
    drv.execute_script("const r = document.getElementById('fo-amp'); r.value = 1.30103; r.dispatchEvent(new Event('input', {bubbles: true}))")
    H.wait(drv, "const s = window.__mace.freqView.state(); return s.opts.scale === 'physical' && Math.abs(s.opts.amplitude - 20) < 0.01", 10)
    r = _export_animation(drv, source="file")
    assert isinstance(r, dict), r
    assert r["kind"] == "freq" and r["frames"] == 24 and r["type"] == "image/gif", r
    assert not H.errors(drv), H.errors(drv)


def test_export_animation_rendered_by_the_server_from_an_optimization(drv):
    if not OPT.is_file():
        pytest.skip("test/ corpus not present (gitignored)")
    kind = drv.execute_async_script("const d = arguments[1]; window.__mace.open(arguments[0]).then((x) => d(x && x.kind))", f"runs/{OPT.name}")
    assert kind in ("trajectory", "structure"), kind
    time.sleep(1.0)
    r = _export_animation(drv, source="file")
    assert isinstance(r, dict), r
    assert r["kind"] == "opt" and r["frames"] == 7 and r["type"] == "image/gif", r
    assert not H.errors(drv), H.errors(drv)


SPAN = """const gl = document.querySelector('.po-bz .kp-bz .kp-bz-gl'), cv = gl && gl.querySelector('canvas');
  if (!cv || !cv.width) return null;
  const c = document.createElement('canvas'); c.width = cv.width; c.height = cv.height;
  const g = c.getContext('2d'); g.drawImage(cv, 0, 0);
  const px = g.getImageData(0, 0, c.width, c.height).data, bg = [px[0], px[1], px[2]];
  let x0 = c.width, x1 = -1, y0 = c.height, y1 = -1;
  for (let y = 0; y < c.height; y += 2) for (let x = 0; x < c.width; x += 2) {
    const i = 4 * (y * c.width + x);
    if (Math.abs(px[i] - bg[0]) + Math.abs(px[i + 1] - bg[1]) + Math.abs(px[i + 2] - bg[2]) > 40) { x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
  }
  return x1 < 0 ? [0, 0] : [(x1 - x0) / c.width, (y1 - y0) / c.height];"""


@pytest.mark.parametrize("theme, background", [("dark", "white"), ("light", "black")])
def test_brillouin_zone_fills_its_view_and_keeps_contrast(drv, theme, background):
    drv.execute_script("const s = window.__mace.settings; s.set('theme', arguments[0]); s.set('background', arguments[1])", theme, background)
    U.open_doc(drv, "gr/g_band.BAND.DAT", wait_atoms=False)
    H.wait(drv, "const e = document.querySelector('.po-bz .kp-bz'); return e && e.__bz.state().loaded && e.__bz.state().viewer && e.__bz.state().fitted", 60)
    st = drv.execute_script("return document.querySelector('.po-bz .kp-bz').__bz.state()")
    c = st["colors"]
    bg = hexof(drv, c["bg"])
    assert lum(bg) > 0.8 if background == "white" else lum(bg) < 0.05, bg
    assert ratio(hexof(drv, c["edge"]), bg) >= 3 and ratio(hexof(drv, c["accent"]), bg) >= 3, c
    assert ratio(hexof(drv, c["text"]), bg) >= 4.5, c
    # the zone fills its view at once (it was drawn at about 40 % until zoomed)
    time.sleep(0.5)
    span = H.wait(drv, SPAN, 20)
    assert max(span) > 0.6, span
    assert not H.errors(drv), H.errors(drv)


def test_a_fractional_range_along_a_vacuum_axis_says_what_it_draws(drv):
    """A molecule's cube with a fractional display range: the range the view reports is the copies it draws
    (2 along c for 0..2.4), not 1x1x1 while the surfaces repeat (a 1.1.6 sweep finding)."""
    if not (CORPUS / "ECH3POT3").is_dir():
        pytest.skip("test/ corpus not present (gitignored)")
    open_vol(drv, "vol/mol_DENS.CUBE")
    drv.execute_script("const s = window.__mace.settings; s.set('displayLo', [0, 0, 0]); s.set('displayHi', [1, 1, 2.4])")
    H.wait(drv, "const d = window.__mace.volumeView.state().display; return d && JSON.stringify(d.range) === '[1,1,2]' && d.surfaces === 2", 30)
    drv.execute_script("const s = window.__mace.settings; s.set('displayHi', [1, 1, 1])")
    H.wait(drv, "const d = window.__mace.volumeView.state().display; return d && JSON.stringify(d.range) === '[1,1,1]' && d.surfaces === 1", 30)
    assert not H.errors(drv), H.errors(drv)
