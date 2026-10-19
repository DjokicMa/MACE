"""A plot's own style (1.1.6, Marcus 2026-10-16): Plot options › Style and the toolbar's Style ▾ menu.

On the spin-polarized Fe run (tests/data/spin_fe) as a band + DOS figure:
* the default spin colors are Blue/red, β dashed;
* Plot options › Style: a scheme recolors the bands and the DOS total at once, undo brings the old
  pair back, Reset Plot Options goes back to Settings; β dashed off makes spin down solid;
  Make This the Default writes the look into Settings and the plot follows Settings again;
* the toolbar's Style ▾ menu picks a preset and the DOS colors through the same options;
* Copy CLI Command writes `mace plotting --band-dos ...` with the plot's style flags;
* a 2D map's color range is robust (1st to 99th percentile); Full and Custom override it;
* a POTM potential map is red at its negative end (chemistry colors), blue with Potential colors › Physics.
"""
import shutil
import time

import pytest

import guiharness as H
import p4util as U
from test_parity_fixtures import unpack  # noqa: E402

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("plotly")

FE = H.REPO_ROOT / "tests" / "data" / "spin_fe"
KEYS = ["plots.style", "plots.style.band", "plots.spinColors", "plots.spinUp", "plots.spinDown", "plots.spinDownDash",
        "plots.dosColors", "plots.spinDos", "plots.dosTotal", "theme"]


@pytest.fixture(scope="module", params=["firefox", "chromium"])
def app(request, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("plotstyle")
    root = H.examples_copy(tmp)
    (root / "fe").mkdir()
    for f in FE.iterdir():
        shutil.copy(f, root / "fe")
    unpack(root / "gp")
    _potm(root / "gp")
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


def drawn(drv):
    H.wait(drv, "const v = window.__mace.plotView, s = v && v.state(); return s && s.drawn && v.el().dataset.drawn !== ''", 30)
    time.sleep(0.3)


FIG = """const norm = (x) => { const c = document.createElement('canvas').getContext('2d'); c.fillStyle = '#123456'; c.fillStyle = x; return c.fillStyle; };
  const p = document.querySelector('#plot-host .pv-plot'), by = {};
  p.data.forEach((t) => { if (t.meta === 'up' || t.meta === 'down' || t.name === 'Total ↑' || t.name === 'Total ↓')
    by[t.meta === 'up' || t.meta === 'down' ? t.meta : t.name] = {color: norm(t.line.color), dash: t.line.dash || 'solid', width: t.line.width}; });
  return {by, x2: p.layout.xaxis2 && p.layout.xaxis2.range, font: p._fullLayout.font.size, opts: window.__mace.plotView.state().opts,
          dosNames: p.data.filter((t) => t.meta === 'dos').map((t) => t.name)};"""


def change(drv, ident, value=None, checked=None):
    drv.execute_script("document.querySelector('#plot-host .pv-plot').dataset.drawn = '';"
                       "const i = document.getElementById(arguments[0]); if (arguments[2] !== null) i.checked = arguments[2]; else i.value = arguments[1];"
                       "i.dispatchEvent(new Event('change', {bubbles: true}))", ident, value, checked)
    drawn(drv)


def test_style_section_recolors_undoes_resets_and_saves(drv):
    open_plot(drv, "fe/fe_band.BAND.DAT")
    f = drv.execute_script(FIG)
    assert (f["by"]["up"]["color"], f["by"]["down"]["color"], f["by"]["down"]["dash"]) == ("#1f4e9c", "#c0392b", "dash")
    assert (f["by"]["Total ↑"]["color"], f["by"]["Total ↓"]["color"]) == ("#1f4e9c", "#c0392b")
    assert drv.execute_script("return !!document.querySelector('.po-style #po-spinc')")
    # a scheme of the plot's own: the bands and the DOS total at once
    change(drv, "po-spinc", "mace")
    f = drv.execute_script(FIG)
    assert (f["by"]["up"]["color"], f["by"]["down"]["color"], f["by"]["Total ↓"]["color"]) == ("#e0218a", "#00a0a4", "#00a0a4")
    assert f["opts"]["spin_colors"] == "mace"
    # undo: the plot's options' own history
    drv.execute_script("document.querySelector('#plot-host .pv-plot').dataset.drawn = ''; window.__mace.plotView.undo()")
    drawn(drv)
    f = drv.execute_script(FIG)
    assert f["by"]["up"]["color"] == "#1f4e9c" and f["opts"]["spin_colors"] is None
    # β dashed off: spin down solid
    change(drv, "po-bdash", checked=False)
    assert drv.execute_script(FIG)["by"]["down"]["dash"] == "solid"
    # Reset Plot Options: back to Settings
    change(drv, "po-style", "publication")
    assert drv.execute_script(FIG)["by"]["up"]["color"] == "#000000"          # Publication: black/vermilion
    drv.execute_script("document.querySelector('#plot-host .pv-plot').dataset.drawn = ''; window.__mace.plotView.reset()")
    drawn(drv)
    f = drv.execute_script(FIG)
    assert f["opts"]["style"] is None and f["opts"]["spin_down_style"] is None and f["by"]["down"]["dash"] == "dash"
    # Make This the Default: Settings take the look, the plot follows them
    change(drv, "po-spinc", "black-vermilion")
    change(drv, "po-dspin", "overlaid")
    drv.execute_script("document.querySelector('#plot-host .pv-plot').dataset.drawn = ''; document.querySelector('[data-po=default]').click()")
    drawn(drv)
    S = drv.execute_script("const S = window.__mace.settings; return {c: S.get('plots.spinColors'), d: S.get('plots.spinDos')}")
    assert S == {"c": "black-vermilion", "d": "overlaid"}
    f = drv.execute_script(FIG)
    assert f["opts"]["spin_colors"] is None and f["opts"]["dos_spin"] is None
    assert f["by"]["up"]["color"] == "#000000" and f["x2"][0] == 0      # overlaid: both spins on the right
    U.no_errors(drv)


def test_toolbar_style_menu_and_cli_command(drv):
    open_plot(drv, "fe/fe_band.BAND.DAT")
    before = drv.execute_script(FIG)["font"]
    drv.find_element("css selector", "#plot-host .pk-style").click()
    items = drv.execute_script("return [...document.querySelectorAll('.pk-menu button')].map((b) => b.dataset.v)")
    assert {"style:presentation", "spin:mace", "bdash", "dos:element", "dspin:mirrored", "dtotal", "cli"} <= set(items), items
    drv.execute_script("document.querySelector('#plot-host .pv-plot').dataset.drawn = '';"
                       "document.querySelector('.pk-menu button[data-v=\"style:presentation\"]').click()")
    drawn(drv)
    f = drv.execute_script(FIG)
    assert f["opts"]["style"] == "presentation" and f["font"] > before
    drv.find_element("css selector", "#plot-host .pk-style").click()
    drv.execute_script("document.querySelector('#plot-host .pv-plot').dataset.drawn = '';"
                       "document.querySelector('.pk-menu button[data-v=\"dos:total\"]').click()")
    drawn(drv)
    f = drv.execute_script(FIG)
    assert f["opts"]["dos_colors"] == "total" and f["dosNames"] and all(n.startswith("Total") for n in f["dosNames"])
    # Copy CLI Command: what mace plotting draws the same figure with
    text = drv.execute_script("return window.__mace.plotView.cliText()")
    assert text.startswith("mace plotting --band-dos fe_band.BAND.DAT --e-lower=-6 --e-upper=6"), text
    assert "--style=presentation" in text and "--dos-colors=total" in text
    U.no_errors(drv)


def test_a_style_pick_by_keyboard_keeps_the_focus_on_style(drv):
    """Enter on Style ▾, the arrows, Enter: the pick draws the toolbar again and the focus is on the new
    Style button (it fell to <body>, so a keyboard user started again from the top)."""
    from selenium.webdriver.common.keys import Keys
    open_plot(drv, "fe/fe_band.BAND.DAT")
    drv.execute_script("document.querySelector('#plot-host .pk-style').focus()")
    U.keys(drv, Keys.ENTER)
    H.wait(drv, "return !!document.querySelector('.pk-menu') && document.activeElement.closest('.pk-menu')", 5)
    U.keys(drv, Keys.ARROW_DOWN)
    U.keys(drv, Keys.ENTER)
    drawn(drv)
    H.wait(drv, "return document.activeElement && document.activeElement.classList.contains('pk-style')", 5)
    U.no_errors(drv)


MAP = """const p = document.querySelector('#plot-host .pv-plot'), t = p.data[0], v = t.z.flat().filter(Number.isFinite);
  return {zmin: t.zmin, zmax: t.zmax, lo: Math.min(...v), hi: Math.max(...v)};"""


def _potm(gp):
    """mgo_echg's map made a POTM potential: values shifted to both signs, the output's keyword POTM."""
    lines = (gp / "mgo_echg.f25").read_text().splitlines()
    n = int(lines[0].split()[1]) * int(lines[0].split()[2])
    rows = (n + 5) // 6
    vals = [float(x) - 0.05 for ln in lines[3:3 + rows] for x in ln.split()]
    body = ["".join(f"{v:12.5E}" for v in vals[i:i + 6]) for i in range(0, n, 6)]
    (gp / "mgo_potm.f25").write_text("\n".join(lines[:3] + body + lines[3 + rows:]) + "\n")
    out = (gp / "mgo_echg.out").read_text().replace("ALPHA+BETA ELECTRONS DENSITY", "").replace(
        "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTT ECHG ", "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTT POTM ")
    (gp / "mgo_potm.out").write_text(out)


# the low end (most negative values) of the map's colorscale as (red, blue)
MAP_LOW = """const p = document.querySelector('#plot-host .pv-plot'), t = p._fullData[0], cs = t.colorscale;
  const c = document.createElement('canvas').getContext('2d'); c.fillStyle = (t.reversescale ? cs[cs.length - 1] : cs[0])[1];
  const h = c.fillStyle; if (h[0] === '#') return [parseInt(h.slice(1, 3), 16), parseInt(h.slice(5, 7), 16)];
  const n = h.match(/[0-9.]+/g).map(Number); return [n[0], n[2]];"""


def test_potential_map_in_chemistry_colors_and_physics_on_request(drv):
    open_plot(drv, "gp/mgo_potm.f25")
    assert drv.execute_script("return document.querySelector('#plot-host .pv-plot').data[0].zmin") < 0
    r, b = drv.execute_script(MAP_LOW)
    assert r > b, (r, b)                                               # negative: red
    U.shot(drv, "map_potential_chemistry")
    drv.execute_script("document.querySelector('#plot-host .pv-plot').dataset.drawn = '';"
                       "window.__mace.settings.set('volume.mepColors', 'physics')")
    drawn(drv)
    r, b = H.wait(drv, MAP_LOW.replace("return [n[0], n[2]];", "return n[2] > n[0] && [n[0], n[2]];")
                  .replace("return [parseInt(h.slice(1, 3), 16), parseInt(h.slice(5, 7), 16)];",
                           "{ const a = [parseInt(h.slice(1, 3), 16), parseInt(h.slice(5, 7), 16)]; return a[1] > a[0] && a; }"), 10)
    assert b > r                                                       # physics: negative blue
    U.shot(drv, "map_potential_physics")
    drv.execute_script("window.__mace.settings.set('volume.mepColors', 'chemistry')")
    # a charge density map is not a potential: the setting does not touch it
    open_plot(drv, "gp/mgo_echg.f25")
    U.no_errors(drv)


def test_map_color_range_is_robust(drv):
    open_plot(drv, "gp/mgo_echg.f25")
    m = drv.execute_script(MAP)
    assert m["lo"] <= m["zmin"] < m["zmax"] < m["hi"]                # the spikes at the nuclei are clipped
    change(drv, "po-crange", "full")
    m = drv.execute_script(MAP)
    assert m["zmin"] == pytest.approx(m["lo"]) and m["zmax"] == pytest.approx(m["hi"])
    change(drv, "po-crange", "custom")
    change(drv, "po-zmin", "-1")
    change(drv, "po-zmax", "0.5")
    m = drv.execute_script(MAP)
    assert (m["zmin"], m["zmax"]) == (-1, 0.5)
    U.no_errors(drv)
