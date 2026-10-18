"""Plot presets per family, DOS colors and spin styling in the GUI (1.1.6), on real runs from the
test/ corpus, in Firefox and Chromium; asserted on the plotly traces and layouts (no screenshots).

* Settings › Appearance › Plot style is the preset of every family, "Same as Plot style" per family
  or a family's own; VESTA-like is for maps and volumes only; a stored 1.1.6-preview "plots.style"
  (default or publication) reads as that preset in every family; an unknown value is dropped;
* DOS colors Auto/Total/By element/By orbital/Element × orbital pick the projections and name them
  ("Ag s", "Br"), and a plot's own choice wins over Settings;
* spin styling on a real spin-polarized run (4-layer AA graphene): both spins drawn, spin down
  dashed over spin up in the interim blue / red pair; Settings change the pair and the dash; the
  spin DOS is mirrored (Auto, B2) or overlaid;
* the plot's PNG/SVG export draws the preset as the screen does;
* B7: Settings › Viewport › Radii changes the radius every view draws (Ti has its own size).
"""
import json
import shutil
import time

import pytest

import guiharness as H
import p4util as U
from conftest import find_data

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("plotly")

AA = "4LG_2x2_AA_opt_HSESOL3C_optimized"
AG = "Ag4Br2S_CRYSTAL_OPT_symm_PBE-D3_full.basis.triplezeta_opt_B3LYP-D3-D3_optimized_sp_B3LYP-D3-D3_optimized"


@pytest.fixture(scope="module", params=["firefox", "chromium"])
def app(request, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("presets")
    root = H.examples_copy(tmp)
    spin, ag = root / "spin", root / "ag"
    spin.mkdir()
    ag.mkdir()
    for pat in (f"BAND/{AA}_band.BAND.DAT", f"BAND/{AA}_band.d3", f"DOSS/{AA}_doss.DOSS.DAT", f"DOSS/{AA}_doss.d3"):
        shutil.copy(find_data(pat), spin)
    for pat in (f"DOSS/{AG}_doss.DOSS.DAT", f"DOSS/{AG}_doss.d3"):
        shutil.copy(find_data(pat), ag)
    real = H.firefox
    if request.param == "chromium":
        H.firefox = lambda w, h, download_dir=None, **kw: H.chromium(w, h, download_dir=download_dir)
    try:
        a = U.start_app(tmp, root=root)
    finally:
        H.firefox = real
    try:
        yield a
    finally:
        U.stop_app(a)


KEYS = ["plots.style", "plots.style.band", "plots.style.spectra", "plots.style.maps", "plots.style.scf", "plots.dosColors",
        "plots.dosTotal", "plots.spinColors", "plots.spinUp", "plots.spinDown", "plots.spinDownDash", "plots.spinDos", "radiusSet", "theme"]


@pytest.fixture
def drv(app, request):
    d = app[0]
    U.reset(d)
    d.execute_script("Object.keys(localStorage).filter((k) => k.startsWith('mace.gui.plot.')).forEach((k) => localStorage.removeItem(k));"
                     "const S = window.__mace.settings, D = S.defaults();"
                     "for (const k of arguments[0]) if (k in D) S.set(k, D[k]);"
                     "S.set('theme', 'light')", KEYS)
    failed = request.session.testsfailed
    yield d
    if request.session.testsfailed > failed:
        H.save_failure(d, request.node.name)


def open_plot(drv, path):
    U.open_doc(drv, path, wait_atoms=False)
    H.wait(drv, "const v = window.__mace.plotView, s = v && v.state(); return s && s.drawn && v.el().dataset.drawn === arguments[0]", 90, path)
    time.sleep(0.3)


def redraw(drv, key, value):
    drv.execute_script("document.querySelector('#plot-host .pv-plot').dataset.drawn = ''; window.__mace.settings.set(arguments[0], arguments[1])", key, value)
    H.wait(drv, "const v = window.__mace.plotView, s = v && v.state(); return s && s.drawn && v.el().dataset.drawn !== ''", 30)
    time.sleep(0.3)


TRACES = """const norm = (x) => { const c = document.createElement('canvas').getContext('2d'); c.fillStyle = '#123456'; c.fillStyle = x; return c.fillStyle; };
  const p = document.querySelector('#plot-host .pv-plot');
  return {traces: p.data.map((t) => ({name: t.name, meta: t.meta || null, color: t.line ? norm(t.line.color) : null, dash: t.line && t.line.dash || 'solid',
    show: t.showlegend !== false, xmin: (t.x || []).reduce((m, v) => (v === null ? m : Math.min(m, v)), Infinity)})),
    x2: p.layout.xaxis2 ? p.layout.xaxis2.range : (p.layout.xaxis || {}).range, ticks: p._fullLayout.xaxis.ticks, font: p._fullLayout.font.size}"""


def test_a_preset_per_family_and_same_for_all(drv):
    S = "return ['band', 'spectra', 'maps', 'scf'].map((f) => window.MACE.plotPreset(f))"
    assert drv.execute_script(S) == ["default"] * 4
    drv.execute_script("window.__mace.settings.set('plots.style', 'colorblind')")
    assert drv.execute_script(S) == ["colorblind"] * 4                     # Same for all
    drv.execute_script("window.__mace.settings.set('plots.style.band', 'grayscale'); window.__mace.settings.set('plots.style.maps', 'vesta')")
    assert drv.execute_script(S) == ["grayscale", "colorblind", "vesta", "colorblind"]
    # VESTA-like is for maps and volumes: elsewhere it draws Default
    drv.execute_script("window.__mace.settings.set('plots.style', 'vesta'); window.__mace.settings.set('plots.style.band', 'same')")
    assert drv.execute_script(S) == ["default", "default", "vesta", "default"]
    # maps and volumes: the 3D view's and the slices' sign colors (+ first) and colormaps
    assert drv.execute_script("return window.MACE.volumeColors()") == ["#e6c200", "#00a0c8"]
    assert drv.execute_script("return window.MACE.mapScale(false).colorscale[0][1]") == "#0000ff"
    drv.execute_script("window.__mace.settings.set('plots.style', 'default'); window.__mace.settings.set('plots.style.maps', 'same')")
    assert drv.execute_script("return window.MACE.mapScale(true)") == {"colorscale": "RdBu", "reversescale": True, "contours": "both"}
    U.no_errors(drv)


def test_stored_plot_style_migrates(drv):
    got = drv.execute_script("""const m = window.__mace.settings.migrate({'plots.style': 'publication', 'plots.style.band': 'bogus',
        'plots.dosColors': 'rainbow', 'plots.spinUp': 'red', 'plots.spinDown': '#00ff00'});
      return m""")
    assert got == {"plots.style": "publication", "plots.spinDown": "#00ff00"}, got
    drv.execute_script("window.__mace.settings.set('plots.style', 'publication')")
    assert drv.execute_script("return ['band', 'spectra', 'maps', 'scf'].map((f) => window.MACE.plotPreset(f))") == ["publication"] * 4
    from mace.gui import settings as S
    assert S.check("plots.style", "grayscale") is None and S.check("plots.style.maps", "vesta") is None
    assert S.check("plots.style.band", "same") is None and S.check("plots.style", "bogus")
    assert S.check("plots.spinUp", "#123abc") is None and S.check("plots.spinUp", "blue")
    assert S.check("plots.dosColors", "element-orbital") is None and S.check("plots.dosTotal", "yes")


def test_dos_colors_pick_and_name_the_projections(drv):
    open_plot(drv, f"ag/{AG}_doss.DOSS.DAT")
    names = lambda: [t["name"] for t in drv.execute_script(TRACES)["traces"] if t["meta"] == "dos"]  # noqa: E731
    # Auto, three elements: element x orbital, named "Ag d"; the 2-4 projections carrying the DOS
    # are drawn (1.1.6), the others hidden (Plot options › Projections)
    assert "Ag d" in names() and "Br p" in names() and "Total" in names() and not any("(" in n for n in names()), names()
    assert "Ag s" not in names() and 3 <= len(names()) <= 5, names()
    redraw(drv, "plots.dosColors", "element")
    assert sorted(names()) == ["Ag", "Br", "S", "Total"], names()
    redraw(drv, "plots.dosColors", "total")
    assert names() == ["Total"]
    redraw(drv, "plots.dosColors", "orbital")
    tr = {t["name"]: t for t in drv.execute_script(TRACES)["traces"] if t["meta"] == "dos"}
    assert tr["Br p"]["color"] == tr["S p"]["color"] == "#d55e00" and tr["Br p"]["dash"] != tr["S p"]["dash"]
    redraw(drv, "plots.dosTotal", False)
    assert "Total" not in names()
    # the plot's own choice (Plot options › DOS colors) over Settings
    drv.execute_script("const i = document.getElementById('po-dosc'); i.value = 'element'; i.dispatchEvent(new Event('change', {bubbles: true}))")
    H.wait(drv, "return document.querySelector('#plot-host .pv-plot').data.some((t) => t.name === 'Br')", 20)
    U.no_errors(drv)


def test_spin_styling_on_a_real_spin_polarized_run(drv):
    open_plot(drv, f"spin/{AA}_band.BAND.DAT")
    tr = {t["meta"]: t for t in drv.execute_script(TRACES)["traces"] if t["meta"] in ("up", "down")}
    # both spins, spin down dashed and drawn over spin up, in the default scheme (presets.py SPIN_SCHEMES: Blue/red)
    assert tr["up"]["color"] == "#1f4e9c" and tr["down"]["color"] == "#c0392b"
    assert tr["up"]["dash"] == "solid" and tr["down"]["dash"] == "dash"
    order = [t["meta"] for t in drv.execute_script(TRACES)["traces"] if t["meta"] in ("up", "down")]
    assert order == ["up", "down"]
    note = drv.find_element("css selector", "#plot-host .pv-note").text
    assert "α solid, β dashed" in note and "meV apart" in note, note
    # the DOS beside it: mirrored (Auto, B2), the total ↑ in spin up's color, ↓ in spin down's
    fig = drv.execute_script(TRACES)
    dos = {t["name"]: t for t in fig["traces"] if t["meta"] == "dos"}
    assert fig["x2"][0] < 0 < fig["x2"][1]
    assert dos["Total ↑"]["color"] == "#1f4e9c" and dos["Total ↓"]["color"] == "#c0392b" and dos["Total ↓"]["xmin"] < 0
    # a named scheme in Settings
    redraw(drv, "plots.spinColors", "mace")
    tr = {t["meta"]: t for t in drv.execute_script(TRACES)["traces"] if t["meta"] in ("up", "down")}
    assert (tr["up"]["color"], tr["down"]["color"]) == ("#e0218a", "#00a0a4")
    redraw(drv, "plots.spinColors", "auto")
    # Settings: another pair and a dotted spin down
    drv.execute_script("window.__mace.settings.set('plots.spinUp', '#000000'); window.__mace.settings.set('plots.spinDown', '#d55e00')")
    redraw(drv, "plots.spinDownDash", "dot")
    tr = {t["meta"]: t for t in drv.execute_script(TRACES)["traces"] if t["meta"] in ("up", "down")}
    assert (tr["up"]["color"], tr["down"]["color"], tr["down"]["dash"]) == ("#000000", "#d55e00", "dot")
    # overlaid: both spins on the right
    redraw(drv, "plots.spinDos", "overlaid")
    fig = drv.execute_script(TRACES)
    dos = {t["name"]: t for t in fig["traces"] if t["meta"] == "dos"}
    assert fig["x2"][0] == 0 and dos["Total ↓"]["xmin"] >= 0
    # Print grayscale: black and gray
    drv.execute_script("window.__mace.settings.set('plots.spinUp', ''); window.__mace.settings.set('plots.spinDown', '')")
    redraw(drv, "plots.style.band", "grayscale")
    tr = {t["meta"]: t for t in drv.execute_script(TRACES)["traces"] if t["meta"] in ("up", "down")}
    assert (tr["up"]["color"], tr["down"]["color"]) == ("#000000", "#767676")
    U.no_errors(drv)


def test_exports_draw_the_preset(drv):
    """The PNG/SVG export is drawn from the figure on screen: Presentation's larger text and thick lines."""
    open_plot(drv, f"spin/{AA}_band.BAND.DAT")
    before = drv.execute_script(TRACES)["font"]
    redraw(drv, "plots.style", "presentation")
    fig = drv.execute_script(TRACES)
    assert fig["font"] > before and fig["ticks"] == "outside"
    svg = drv.execute_async_script("""const d = arguments[0], el = document.querySelector('#plot-host .pv-plot');
      Plotly.toImage(el, {format: 'svg', width: 600, height: 400}).then((u) => d(decodeURIComponent(u.replace(/^data:image\\/svg\\+xml,/, ''))))""")
    assert "stroke-width: 3px" in svg or "stroke-width:3px" in svg.replace(" ", "")
    U.no_errors(drv)


def test_radii_setting_changes_the_spheres(drv):
    """B7: Atomic (VESTA) radii by default, so a transition metal has its own size; van der Waals is the
    pre-1.1.6 table; Atom scale 0.4 draws 40 % of the atomic radius (T9)."""
    U.open_doc(drv, "structures/NaCl.cif", wait_atoms=True)
    R = "const r = $3Dmol.GLModel.vdwRadii; return [r.Na, r.Ti, r.Fe, window.MACE.sphereScale(true)]"
    na, ti, fe, k = drv.execute_script(R)
    assert (na, ti, fe, k) == (1.91, 1.47, 1.26, 0.4)
    drv.execute_script("window.__mace.settings.set('radiusSet', 'vdw')")
    time.sleep(0.5)
    na, ti, fe, k = drv.execute_script(R)
    assert na == pytest.approx(2.27) and ti is None and k == pytest.approx(0.28)
    drv.execute_script("window.__mace.settings.set('radiusSet', 'ionic')")
    time.sleep(0.5)
    assert drv.execute_script(R)[0] == pytest.approx(1.02)
    U.no_errors(drv)
