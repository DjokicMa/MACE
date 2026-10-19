"""A band + DOS plot's style controls (1.1.6 preview fix, Marcus 2026-10-18): the toolbar's Style ▾ menu and
Plot options › Style draw one list of controls, grouped by the figure's halves, whichever file was opened.

* a BAND.DAT with its DOSS.DAT partner, and that DOSS.DAT: both the Bands group (spin colors, β line) and
  the DOS group (DOS colors, spin layout, total, smoothing), the same controls in the menu and the Inspector;
* a BAND.DAT alone: the Bands group only; a DOSS.DAT alone: the DOS group only;
* the DOSS.DAT's view of the pair: a Bands control recolors the bands, a DOS control the DOS panel, from the
  menu and the Inspector alike; undo, Make This the Default and Copy CLI Command take both halves.
On the spin-polarized Fe run (tests/data/spin_fe).
"""
import shutil
import time

import pytest

import guiharness as H
import p4util as U

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("plotly")

FE = H.REPO_ROOT / "tests" / "data" / "spin_fe"
KEYS = ["plots.style", "plots.style.band", "plots.spinColors", "plots.spinUp", "plots.spinDown", "plots.spinDownDash",
        "plots.dosColors", "plots.spinDos", "plots.dosTotal", "plots.dosSmooth", "plots.background", "plots.backgroundColor", "theme"]
PAIR_BAND, PAIR_DOS = "fe/fe_band.BAND.DAT", "fe/fe_doss.DOSS.DAT"
CLOSED = "closed/fe_band.BAND.DAT"
BAND_ALONE, DOS_ALONE = "bandonly/fe_band.BAND.DAT", "dosonly/fe_doss.DOSS.DAT"
BANDS = {"po-spinc", "po-bdash"}
DOS = {"po-dosc", "po-dspin", "po-dtotal", "po-dsmooth"}
# the Style menu's value prefix of each Inspector control (plot.js styleControls)
MENU = {"po-style": "style", "po-spinc": "spin", "po-bdash": "bdash", "po-dosc": "dos", "po-dspin": "dspin",
        "po-dtotal": "dtotal", "po-dsmooth": "dsmooth", "po-bandc": "bandc", "po-spinup": "spinup", "po-spindn": "spindn",
        "po-bg": "bg", "po-bgc": "bgc"}


@pytest.fixture(scope="module", params=["firefox", "chromium"])
def app(request, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("plothalves")
    root = H.examples_copy(tmp)
    for d in ("fe", "bandonly", "dosonly"):
        (root / d).mkdir()
    for f in FE.iterdir():
        shutil.copy(f, root / "fe")
    shutil.copy(FE / "fe_band.BAND.DAT", root / "bandonly")
    shutil.copy(FE / "fe_doss.DOSS.DAT", root / "dosonly")
    # a closed-shell run: the Fe bands' α block alone
    (root / "closed").mkdir()
    alpha = (FE / "fe_band.BAND.DAT").read_text().splitlines()[:240]
    alpha[0] = alpha[0].replace("NSPIN     2", "NSPIN     1")
    (root / "closed" / "fe_band.BAND.DAT").write_text("\n".join(alpha) + "\n")
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


def inspector(drv):
    """{group: [control ids]} of Plot options › Style"""
    return drv.execute_script("const out = {}; document.querySelectorAll('.po-style [data-group]').forEach((g) =>"
                              " out[g.dataset.group] = [...g.querySelectorAll('input[id], select[id]')].map((i) => i.id)); return out")


def style_menu(drv):
    """{group: [menu value prefixes]} of the toolbar's Style ▾ menu (closed again)"""
    drv.find_element("css selector", "#plot-host .pk-style").click()
    H.wait(drv, "return !!document.querySelector('.pk-menu')", 5)
    out = drv.execute_script("const out = {plot: []}; let g = 'plot';"
                             "for (const el of document.querySelectorAll('.pk-menu > *')) {"
                             "  if (el.classList.contains('pk-group')) { g = el.dataset.group; out[g] = out[g] || []; }"
                             "  else if (el.dataset.v && !['settings', 'cli'].includes(el.dataset.v)) { const k = el.dataset.v.split(':')[0]; if (!out[g].includes(k)) out[g].push(k); } }"
                             "return out")
    close_menu(drv)
    return out


def close_menu(drv):
    drv.execute_script("const m = document.querySelector('.pk-menu'); if (m) m.close()")


def pick(drv, value):
    drv.find_element("css selector", "#plot-host .pk-style").click()
    H.wait(drv, "return !!document.querySelector('.pk-menu')", 5)
    drv.execute_script("document.querySelector('#plot-host .pv-plot').dataset.drawn = '';"
                       "document.querySelector(`.pk-menu button[data-v=\"${arguments[0]}\"]`).click()", value)
    drawn(drv)


def change(drv, ident, value=None, checked=None):
    drv.execute_script("document.querySelector('#plot-host .pv-plot').dataset.drawn = '';"
                       "const i = document.getElementById(arguments[0]); if (arguments[2] !== null) i.checked = arguments[2]; else i.value = arguments[1];"
                       "i.dispatchEvent(new Event('change', {bubbles: true}))", ident, value, checked)
    drawn(drv)


FIG = """const norm = (x) => { const c = document.createElement('canvas').getContext('2d'); c.fillStyle = '#123456'; c.fillStyle = x; return c.fillStyle; };
  const p = document.querySelector('#plot-host .pv-plot'), by = {};
  p.data.forEach((t) => { if (t.meta === 'up' || t.meta === 'down' || t.name === 'Total ↑' || t.name === 'Total ↓')
    by[t.meta === 'up' || t.meta === 'down' ? t.meta : t.name] = {color: norm(t.line.color), dash: t.line.dash || 'solid'}; });
  return {by, opts: window.__mace.plotView.state().opts, dosNames: p.data.filter((t) => t.meta === 'dos').map((t) => t.name)};"""


def same_controls(ins, menu):
    """the menu offers each Inspector control of each group (Smoothing σ: only while smoothing is on)"""
    assert set(ins) == set(menu), (ins, menu)
    for g, ids in ins.items():
        assert [MENU[i] for i in ids if i in MENU] == [k for k in menu[g] if k != "dsigma"], (g, ids, menu[g])


@pytest.mark.parametrize("path", [PAIR_BAND, PAIR_DOS])
def test_band_with_dos_partner_offers_both_halves(drv, path):
    open_plot(drv, path)
    ins, menu = inspector(drv), style_menu(drv)
    assert set(ins) == {"plot", "bands", "dos"}, ins
    assert BANDS <= set(ins["bands"]) and DOS <= set(ins["dos"]), ins
    assert not (DOS & set(ins["bands"])) and not (BANDS & set(ins["dos"])), ins
    same_controls(ins, menu)
    U.no_errors(drv)


def test_band_alone_offers_only_band_settings(drv):
    open_plot(drv, BAND_ALONE)
    ins, menu = inspector(drv), style_menu(drv)
    assert set(ins) == {"plot", "bands"} and BANDS <= set(ins["bands"]) and not (DOS & set(ins["bands"])), ins
    same_controls(ins, menu)
    U.no_errors(drv)


def test_dos_alone_offers_only_dos_settings(drv):
    open_plot(drv, DOS_ALONE)
    ins, menu = inspector(drv), style_menu(drv)
    # no Bands group: the DOS total's spin colors and β line sit in the DOS group
    assert set(ins) == {"plot", "dos"} and DOS <= set(ins["dos"]), ins
    same_controls(ins, menu)
    U.no_errors(drv)


def test_each_half_is_styled_from_the_dos_files_view(drv):
    """the pair opened from its DOSS.DAT: a Bands pick recolors the bands, a DOS pick the DOS panel;
    undo, Copy CLI Command and Make This the Default take both halves"""
    open_plot(drv, PAIR_DOS)
    f = drv.execute_script(FIG)
    assert f["by"]["up"]["color"] == "#1f4e9c" and f["by"]["down"]["dash"] == "dash"
    # the menu: Bands › β dashed off makes the bands' spin down solid (and the DOS total's)
    pick(drv, "bdash")
    f = drv.execute_script(FIG)
    assert f["opts"]["spin_down_style"] == "solid" and f["by"]["down"]["dash"] == "solid"
    # the menu: DOS › Total only
    pick(drv, "dos:total")
    f = drv.execute_script(FIG)
    assert f["opts"]["dos_colors"] == "total" and f["dosNames"] and all(n.startswith("Total") for n in f["dosNames"])
    # the Inspector: a custom spin pair, then its spin up color from the menu's color picker
    change(drv, "po-spinc", "custom")
    assert "spinup" in style_menu(drv)["bands"]
    drv.find_element("css selector", "#plot-host .pk-style").click()
    drv.execute_script("document.querySelector('.pk-menu button[data-v=\"spinup\"]').click()")
    H.wait(drv, "return !!document.getElementById('pk-colorpick')", 5)
    drv.execute_script("document.querySelector('#plot-host .pv-plot').dataset.drawn = '';"
                       "const i = document.getElementById('pk-colorpick'); i.value = '#00aa00'; i.dispatchEvent(new Event('change', {bubbles: true}))")
    drawn(drv)
    close_menu(drv)
    f = drv.execute_script(FIG)
    assert f["opts"]["spin_up_color"] == "#00aa00" and f["by"]["up"]["color"] == "#00aa00" and f["by"]["Total ↑"]["color"] == "#00aa00"
    assert drv.execute_script("return document.getElementById('po-spinup').value") == "#00aa00"
    # the Inspector: DOS smoothing on, its σ from the menu
    change(drv, "po-dsmooth", checked=True)
    pick(drv, "dsigma:0.1")
    assert drv.execute_script(FIG)["opts"]["dos_smooth"] == 0.1
    assert float(drv.execute_script("return document.getElementById('po-dsigma').value")) == 0.1
    # undo: one step back for each pick, whichever place made it
    drv.execute_script("document.querySelector('#plot-host .pv-plot').dataset.drawn = ''; window.__mace.plotView.undo()")
    drawn(drv)
    assert drv.execute_script(FIG)["opts"]["dos_smooth"] == 0.05
    # Copy CLI Command: both halves' flags
    text = drv.execute_script("return window.__mace.plotView.cliText()")
    assert text.startswith("mace plotting --band-dos fe_band.BAND.DAT"), text
    assert "--dos-colors=total" in text and "--spin-down-solid" in text, text
    # Make This the Default: both halves go to Settings
    drv.execute_script("document.querySelector('#plot-host .pv-plot').dataset.drawn = ''; document.querySelector('[data-po=default]').click()")
    drawn(drv)
    S = drv.execute_script("const S = window.__mace.settings; return {c: S.get('plots.spinColors'), up: S.get('plots.spinUp'),"
                           " d: S.get('plots.spinDownDash'), dc: S.get('plots.dosColors')}")
    assert S == {"c": "custom", "up": "#00aa00", "d": "solid", "dc": "total"}, S
    f = drv.execute_script(FIG)
    assert f["opts"]["spin_colors"] is None and f["opts"]["dos_colors"] is None and f["by"]["up"]["color"] == "#00aa00"
    U.no_errors(drv)


BG = """const p = document.querySelector('#plot-host .pv-plot'), lum = (x) => { const c = document.createElement('canvas').getContext('2d');
  c.fillStyle = x; const n = parseInt(c.fillStyle.slice(1), 16); return (0.299 * (n >> 16) + 0.587 * ((n >> 8) & 255) + 0.114 * (n & 255)) / 255; };
  return {paper: p._fullLayout.paper_bgcolor, plot: p._fullLayout.plot_bgcolor, tick: lum(p._fullLayout.xaxis.tickfont.color),
          surface: p.getAttribute('data-plot-surface'), opts: window.__mace.plotView.state().opts}"""


def test_plot_background_black_custom_theme(drv):
    """Plot options › Style › Background, as the 3D view's: Black draws light text, a light Custom color dark text, Theme as before"""
    open_plot(drv, BAND_ALONE)
    theme = drv.execute_script(BG)
    assert theme["surface"] is None, theme
    change(drv, "po-bg", "black")
    b = drv.execute_script(BG)
    assert b["paper"] == b["plot"] == "#000000" and b["tick"] > 0.5 and b["surface"] == "dark", b
    assert "bg" in style_menu(drv)["plot"]
    change(drv, "po-bg", "custom")
    change(drv, "po-bgc", "#ffffcc")
    c = drv.execute_script(BG)
    assert c["paper"] == c["plot"] == "#ffffcc" and c["tick"] < 0.5 and c["surface"] == "light", c
    assert c["opts"]["background"] == "custom" and c["opts"]["background_color"] == "#ffffcc"
    change(drv, "po-bg", "")
    t = drv.execute_script(BG)
    assert t["paper"] == theme["paper"] and t["surface"] is None and t["opts"]["background"] is None, (t, theme)
    U.no_errors(drv)


INK = """const p = document.querySelector('#plot-host .pv-plot'), rgb = (x) => { const c = document.createElement('canvas').getContext('2d');
  c.fillStyle = x; const n = parseInt(c.fillStyle.slice(1), 16); return [n >> 16, (n >> 8) & 255, n & 255]; };
  const L = (x) => { const v = rgb(x).map((c) => { c /= 255; return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; });
    return 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2]; };
  const bg = p._fullLayout.plot_bgcolor, lines = p.data.filter((t) => t.mode === 'lines' && t.line && t.line.color && !/dos/i.test(t.xaxis || ''));
  const cs = [...new Set(lines.map((t) => rgb(t.line.color).map((c) => c.toString(16).padStart(2, '0')).join('')))];
  const a = L(bg), contrast = Math.min(...lines.map((t) => { const b = L(t.line.color); return (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05); }));
  return {bg, colors: cs, contrast}"""


# presets.py SPIN_SCHEMES["blue-red"]: the default spin pair on a light and a dark plot
BLUE_RED = {"light": {"up": "#1f4e9c", "down": "#c0392b"}, "dark": {"up": "#7aa7f0", "down": "#f08a7c"}}


@pytest.mark.parametrize("theme,bg,tone", [("dark", "white", "light"), ("light", "black", "dark")])
def test_plot_background_sets_the_data_colors(drv, theme, bg, tone):
    """The data colors follow the plot's own background, not the app theme: the closed-shell band ink contrasts
    with it, and the spin pair is the light-plot (white) or dark-plot (black) Blue/red"""
    drv.execute_script("window.__mace.settings.set('theme', arguments[0])", theme)
    open_plot(drv, CLOSED)
    change(drv, "po-bg", bg)
    c = drv.execute_script(INK)
    assert c["colors"] and c["contrast"] >= 3, c
    open_plot(drv, BAND_ALONE)
    change(drv, "po-bg", bg)
    pair = BLUE_RED[tone]
    s = drv.execute_script(INK)
    assert {pair["up"].lstrip("#"), pair["down"].lstrip("#")} <= set(s["colors"]), (s, pair)
    U.no_errors(drv)
