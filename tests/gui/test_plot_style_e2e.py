"""Settings › Appearance › Plot style (1.1.6), in Chromium and Firefox, and the
migration of the preview's Publication theme.

Plot style Publication draws every plot as a figure in any theme and on any
plot background: Okabe–Ito series after a first one in ink (black on a light
plot, white on a dark one), the axes, frame and ticks in ink, outside ticks,
larger tick labels, no decorative grid (the band structure's high-symmetry
k-point lines stay, in ink) and lines a quarter heavier. The PNG/SVG exports
are the figure as drawn. Default leaves every plot as it was.

The Publication theme of the 1.1.6 preview is gone: a stored theme
"publication" loads as Light with Plot style Publication, once and silently,
from gui.json (settings.py, and the <html> tag of first paint), from
localStorage (settings.js) and in boot.js's first frame.
"""
import json
import sys
import time

import pytest

import guiharness as H
import p4util as U

pytestmark = [pytest.mark.gui, pytest.mark.guismoke]

sys.path.insert(0, str(H.REPO_ROOT / "tests"))


# ---------------------------------------------------------------- gui.json (no browser)

def test_server_migrates_the_publication_theme(tmp_path, monkeypatch):
    from mace.gui import settings as S
    from mace.gui import server
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert S.migrate_theme({"theme": "publication"}) == {"theme": "light", "plots.style": "publication"}
    assert S.migrate_theme({"theme": "publication", "plots.style": "default"}) == {"theme": "light", "plots.style": "default"}
    assert S.migrate_theme({"theme": "sepia"}) == {"theme": "sepia"}
    f = tmp_path / "mace" / "gui.json"
    f.parent.mkdir(parents=True)
    f.write_text(json.dumps({"version": 2, "rev": 3, "settings": {"theme": "publication", "uiScale": 110}}))
    got = S.load()
    assert got["theme"] == "light" and got["plots.style"] == "publication" and got["uiScale"] == 110, got
    assert S.load_profile()["settings"]["theme"] == "light"
    html = server._prefs_attrs()
    assert 'data-theme="light"' in html and 'data-plot-style="publication"' in html and 'data-theme="publication"' not in html, html
    # the next write stores the migrated values, once: the file no longer says "publication" for the theme
    S.patch({"settings": {"set": {"density": "compact"}, "unset": []}})
    stored = json.loads(f.read_text())["settings"]
    assert stored["theme"] == "light" and stored["plots.style"] == "publication" and stored["density"] == "compact", stored
    # a plot style the user chose later is theirs
    S.patch({"settings": {"set": {"plots.style": "default"}, "unset": []}})
    assert S.load()["plots.style"] == "default" and S.load()["theme"] == "light"
    assert "publication" not in server.THEMES
    assert S.check("plots.style", "publication") is None and S.check("plots.style", "bold")


# ---------------------------------------------------------------- the browser

@pytest.fixture(scope="module", params=["chromium", "firefox"])
def app(request, tmp_path_factory):
    from test_dos_spin_window import write_doss_spin
    from test_gui_results import write_band_crystal
    tmp = tmp_path_factory.mktemp("plotstyle")
    root = H.examples_copy(tmp)
    (root / "gr").mkdir()
    write_band_crystal(root / "gr", noise=1e-4)
    write_doss_spin(root / "gr")
    # a closed-shell pair: the same bands with the DOS's α block alone (NSPIN 1)
    (root / "grc").mkdir()
    write_band_crystal(root / "grc")
    doss = write_doss_spin(root / "grc").read_text().split("&\n")[0].replace("NSPIN     2", "NSPIN     1")
    (root / "grc" / "gra_doss.DOSS.DAT").write_text(doss + "&\n")
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
    d.execute_script("window.__mace.settings.set('theme', 'light'); window.__mace.settings.set('plots.surface', 'theme');"
                     "window.__mace.settings.set('plots.style', 'default')")
    failed = request.session.testsfailed
    yield d
    if request.session.testsfailed > failed:
        H.save_failure(d, request.node.name)


def open_plot(drv, path):
    U.open_doc(drv, path, wait_atoms=False)
    H.wait(drv, "const v = window.__mace.plotView, s = v && v.state(); return s && s.drawn && v.el().dataset.drawn === arguments[0]", 60, path)
    time.sleep(0.3)


# the figure as plotly holds it: axes, the first band trace, the fonts
FIG = """const el = document.querySelector('#plot-host .pv-plot'), L = el._fullLayout;
  const c = document.createElement('canvas').getContext('2d'), norm = (x) => { c.fillStyle = '#123456'; c.fillStyle = x; return c.fillStyle; };
  const ax = (a) => ({line: norm(a.linecolor), lw: a.linewidth, ticks: a.ticks, tick: norm(a.tickcolor), grid: a.showgrid,
    gridc: a.showgrid ? norm(a.gridcolor) : null, size: a.tickfont.size});
  const band = el.data.find((t) => t.meta === 'up'), dos = el.data.find((t) => t.meta === 'dos' && /^Total/.test(t.name));
  return {x: ax(L.xaxis), y: ax(L.yaxis), x2: L.xaxis2 ? ax(L.xaxis2) : null, font: norm(L.font.color),
    band: norm(band.line.color), bandW: band.line.width, dos: dos ? norm(dos.line.color) : null,
    style: document.documentElement.getAttribute('data-plot-style'), tone: document.documentElement.getAttribute('data-plot-tone')}"""


def redraw(drv, key, value):
    drv.execute_script("document.querySelector('#plot-host .pv-plot').dataset.drawn = ''; window.__mace.settings.set(arguments[0], arguments[1])", key, value)
    H.wait(drv, "const v = window.__mace.plotView, s = v && v.state(); return s && s.drawn && v.el().dataset.drawn !== ''", 30)
    time.sleep(0.3)


def test_publication_plot_style_draws_a_figure(drv):
    open_plot(drv, "gr/gra_band.BAND.DAT")
    before = drv.execute_script(FIG)
    assert before["style"] is None and before["tone"] == "light"
    assert before["band"] != "#000000" and before["x"]["ticks"] != "outside", before
    # Settings › Appearance › Plot style: a real click
    drv.execute_script("window.__mace.run('file.settings')")
    H.wait(drv, "return !!document.querySelector('dialog#settings[open] select[data-key=\"plots.style\"]')", 10)
    el = drv.find_element("css selector", 'dialog#settings select[data-key="plots.style"]')
    drv.execute_script("arguments[0].scrollIntoView({block: 'center'})", el)
    from selenium.webdriver.support.ui import Select
    Select(el).select_by_value("publication")
    H.wait(drv, "return document.documentElement.getAttribute('data-plot-style') === 'publication'", 5)
    drv.execute_script("document.querySelectorAll('dialog[open]').forEach((d) => d.close())")
    H.wait(drv, "const el = document.querySelector('#plot-host .pv-plot'); return el._fullLayout.xaxis.ticks === 'outside'", 30)
    pub = drv.execute_script(FIG)
    assert pub["style"] == "publication" and pub["tone"] == "light"
    assert pub["band"] == "#000000" and pub["font"] == "#000000", pub
    assert pub["bandW"] == pytest.approx(before["bandW"] * 1.25, abs=0.01)
    for a in ("x", "y"):
        assert pub[a]["line"] == "#000000" and pub[a]["ticks"] == "outside" and pub[a]["tick"] == "#000000", (a, pub[a])
        assert pub[a]["size"] > before[a]["size"], (a, pub[a], before[a])
    assert pub["x"]["grid"] and pub["x"]["gridc"] == "#000000"    # the high-symmetry k-points stay, in ink
    assert pub["y"]["grid"] is False                               # the energy grid is decoration: gone
    if pub["dos"]:
        assert pub["dos"] == "#000000"                              # the total DOS: the first series, in ink
    # the SVG export is the figure as drawn
    svg = drv.execute_async_script("""const d = arguments[0], el = document.querySelector('#plot-host .pv-plot');
      Plotly.toImage(el, {format: 'svg', width: 600, height: 400}).then((u) => d(decodeURIComponent(u.replace(/^data:image\\/svg\\+xml,/, ''))))""")
    assert "rgb(0, 0, 0)" in svg
    # a dark plot background: the ink turns white, the series stay Okabe–Ito
    redraw(drv, "plots.surface", "dark")
    dark = drv.execute_script(FIG)
    assert dark["tone"] == "dark" and dark["band"] == "#ffffff" and dark["x"]["line"] == "#ffffff", dark
    # a dark theme with Follow: the same
    drv.execute_script("window.__mace.settings.set('plots.surface', 'theme')")
    redraw(drv, "theme", "dark")
    assert drv.execute_script(FIG)["tone"] == "dark" and drv.execute_script(FIG)["band"] == "#ffffff"
    # back to Default: the plot as it was
    redraw(drv, "theme", "light")
    redraw(drv, "plots.style", "default")
    again = drv.execute_script(FIG)
    assert again["style"] is None and again["band"] == before["band"] and again["x"]["ticks"] == before["x"]["ticks"], again
    U.no_errors(drv)


def test_publication_theme_loads_as_light_with_the_plot_style(app):
    drv, base, root, xdg = app[0], app[1], app[2], app[3]
    # a browser of the 1.1.6 preview: Publication in localStorage, not yet sent to the server
    drv.execute_script("window.__mace.settings.flush && window.__mace.settings.flush()")
    time.sleep(0.5)
    f = xdg / "mace" / "gui.json"
    data = json.loads(f.read_text()) if f.exists() else {"version": 2, "settings": {}}
    data["settings"] = {k: v for k, v in data["settings"].items() if k not in ("theme", "plots.style")}
    f.write_text(json.dumps(data))
    drv.execute_script("localStorage.setItem('mace.gui.settings.v1', JSON.stringify({theme: 'publication'}));"
                       "localStorage.setItem('mace.gui.settings.pending', JSON.stringify(['theme']))")
    H.load(drv, base)
    H.wait(drv, "return document.documentElement.dataset.appReady === '1'", 60)
    assert drv.execute_script("return window.__maceFirstPaint.theme") == "light"
    assert drv.execute_script("return document.documentElement.getAttribute('data-theme')") == "light"
    assert drv.execute_script("return [window.__mace.settings.get('theme'), window.__mace.settings.get('plots.style')]") == ["light", "publication"]
    H.wait(drv, "return document.documentElement.getAttribute('data-plot-style') === 'publication'", 10)
    assert json.loads(drv.execute_script("return localStorage.getItem('mace.gui.settings.v1')"))["theme"] == "light"
    drv.execute_async_script("const d = arguments[0]; window.__mace.settings.flush().then(d, d)")
    stored = json.loads(f.read_text())["settings"]
    assert stored.get("theme") == "light" and stored.get("plots.style") == "publication", stored
    # a gui.json of the preview (another browser chose it): the <html> tag paints Light with the plot style
    data = json.loads(f.read_text())
    data["settings"].update({"theme": "publication"})
    data["settings"].pop("plots.style", None)
    f.write_text(json.dumps(data))
    drv.execute_script("localStorage.removeItem('mace.gui.settings.pending')")
    H.load(drv, base)
    H.wait(drv, "return document.documentElement.dataset.appReady === '1'", 60)
    assert drv.execute_script("return window.__maceFirstPaint.theme") == "light"
    assert drv.execute_script("return [window.__mace.settings.get('theme'), window.__mace.settings.get('plots.style')]") == ["light", "publication"]
    # the Settings › Appearance theme cards: no Publication, no Special purpose group
    drv.execute_script("window.__mace.run('file.settings')")
    H.wait(drv, "return !!document.querySelector('dialog#settings[open] .theme-card')", 10)
    cards = drv.execute_script("return [...document.querySelectorAll('#settings .theme-card')].map((b) => b.dataset.val)")
    groups = drv.execute_script("return [...document.querySelectorAll('#settings .theme-group-h')].map((g) => g.textContent)")
    drv.execute_script("document.querySelectorAll('dialog[open]').forEach((d) => d.close())")
    assert "publication" not in cards and len(cards) == 18 and groups == ["Light & Dark", "Accessibility", "Editor themes"], (cards, groups)
    drv.execute_script("window.__mace.settings.set('plots.style', 'default')")
    U.no_errors(drv)


def test_closed_shell_band_color_reaches_bands_and_dos_total(drv):
    """A closed-shell band + DOS pair with a gap: Band style Single, whose Band color (in Bands; the Color pair
    sits in Colors) recolors the bands and the one-sided DOS total, from the Inspector and the Style menu alike"""
    open_plot(drv, "grc/gra_band.BAND.DAT")
    ids = drv.execute_script("return [...document.querySelectorAll('.po-style [data-group=\"bands\"] [id]')].map((i) => i.id)")
    assert "po-bandm" in ids and "po-bandc" in ids and "po-spinc" not in ids, ids
    drv.execute_script("document.querySelector('#plot-host .pv-plot').dataset.drawn = '';"
                       "const i = document.getElementById('po-bandc'); i.value = '#12ab34'; i.dispatchEvent(new Event('change', {bubbles: true}))")
    H.wait(drv, "const v = window.__mace.plotView, s = v && v.state(); return s && s.drawn && v.el().dataset.drawn !== ''", 30)
    time.sleep(0.3)
    f = drv.execute_script(FIG)
    assert f["band"] == "#12ab34" and f["dos"] == "#12ab34", f
    drv.find_element("css selector", "#plot-host .pk-style").click()
    H.wait(drv, "return !!document.querySelector('.pk-menu')", 5)
    menu = drv.execute_script("return [...document.querySelectorAll('.pk-menu [data-v]')].map((b) => b.dataset.v)")
    drv.execute_script("document.querySelector('.pk-menu').close()")
    assert "bandm:auto" in menu and "bandc" in menu and "bstyle:overlay" in menu and any(v.startswith("spin:") for v in menu), menu
    colors = drv.execute_script("return [...document.querySelectorAll('.po-style [data-group=\"colors\"] [id]')].map((i) => i.id)")
    assert colors == ["po-spinc"], colors
    U.no_errors(drv)
