"""UI themes and element color palettes (1.1.6), in Chromium and Firefox.

Themes: Settings › Appearance has a card per theme; pointing at one
previews it (the page, the 3D view's background and the plots follow) and a
click keeps it; the choice is in gui.json and on the next page's first
paint. View › Theme › Color Theme… lists the themes in the command search:
the arrow keys preview, Esc goes back, Enter keeps.

Element colors: View › Element Colors… edits a palette with a periodic
table and a color picker; the 3D view draws the edit at once; Save writes
~/.config/mace/element_colors.json (the file mace plotting reads), Set as
Default makes it every view's palette, and a new tab with no localStorage
draws it too. Export writes the palette as JSON; Import of that file makes a
second palette with the same colors.
"""
import json
from pathlib import Path
import time

import pytest

import guiharness as H

pytestmark = [pytest.mark.gui, pytest.mark.guismoke]
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")                 # the server reads the structures with it

from selenium.webdriver.common.action_chains import ActionChains  # noqa: E402
from selenium.webdriver.common.keys import Keys  # noqa: E402

COLORS = json.loads((H.REPO_ROOT / "mace" / "gui" / "static" / "element_colors.json").read_text())
TOKEN = "return getComputedStyle(document.documentElement).getPropertyValue(arguments[0]).trim()"
# whether the run plot's text color is the theme's --text (any CSS notation)
PLOT_TEXT_IS_THEMES = """const p = document.querySelector('#runplot-host .js-plotly-plot');
  const got = p && p._fullLayout && p._fullLayout.font.color; if (!got) return false;
  const want = getComputedStyle(document.documentElement).getPropertyValue('--text').trim();
  const c = document.createElement('canvas').getContext('2d'); c.fillStyle = want; const w = c.fillStyle; c.fillStyle = got;
  return c.fillStyle === w"""


@pytest.fixture(scope="module", params=["chromium", "firefox"])
def app(request, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("themes")
    root, xdg = H.examples_copy(tmp), tmp / "xdg"
    proc, base, _ = H.start_server(root, xdg)
    try:
        drv = H.browser(request.param, width=1400, height=900)
    except BaseException:
        proc.terminate()
        proc.wait(timeout=15)
        raise
    try:
        H.load(drv, base)
        yield drv, base, xdg
    finally:
        drv.quit()
        proc.terminate()
        proc.wait(timeout=15)


@pytest.fixture
def drv(app, request):
    d = app[0]
    failed = request.session.testsfailed
    yield d
    if request.session.testsfailed > failed:
        H.save_failure(d, request.node.name)
    d.execute_script("document.querySelectorAll('dialog[open]').forEach((x) => x.close()); window.__mace.closeAll()")


def click(drv, el):
    drv.execute_script("arguments[0].scrollIntoView({block: 'center'})", el)
    ActionChains(drv).move_to_element(el).pause(0.05).click().perform()
    time.sleep(0.15)


def css(drv, sel):
    return drv.find_element("css selector", sel)


def open_doc(drv, path):
    drv.execute_async_script("const d = arguments[1]; window.__mace.open(arguments[0]).then(() => d(1), () => d(0))", path)
    time.sleep(0.6)


def flush(drv):
    drv.execute_async_script("const d = arguments[0]; window.__mace.settings.flush().then(d)")


def gui_json(xdg):
    p = xdg / "mace" / "gui.json"
    return json.loads(p.read_text())["settings"] if p.exists() else {}


def palettes_file(xdg):
    p = xdg / "mace" / "element_colors.json"
    return json.loads(p.read_text()) if p.exists() else {}


def theme(drv):
    return drv.execute_script("return document.documentElement.getAttribute('data-theme')")


def test_theme_cards_preview_keep_and_persist(drv, app):
    _, base, xdg = app
    drv.execute_script("window.__mace.settings.set('theme', 'light')")
    open_doc(drv, "outputs/Ag2Br3_opt_unfinished.out")
    H.wait(drv, "return document.querySelectorAll('#o-energy circle.pt, #o-energy path.pt').length >= 2", 60)
    click(drv, css(drv, "#o-e-plot"))                                      # the run as a Plotly figure
    H.wait(drv, "return window.__mace.runView && window.__mace.runView.state() && window.__mace.runView.state().drawn", 60)
    H.wait(drv, PLOT_TEXT_IS_THEMES, 15)
    click(drv, css(drv, "#settings-btn"))
    H.wait(drv, "return document.getElementById('settings') && document.getElementById('settings').open", 10)
    cards = drv.execute_script("return [...document.querySelectorAll('#settings .theme-card')].map((b) => b.dataset.val)")
    assert cards == ["system", "light", "dark", "sepia", "black", "hc-light", "hc-dark", "cvd",
                     "catppuccin-mocha", "catppuccin-latte", "dracula", "gruvbox-dark", "nord", "one-dark",
                     "solarized-dark", "solarized-light", "tokyo-night", "tokyo-night-storm"], cards
    # in labeled groups, one radio group over all of them
    groups = drv.execute_script("""return [...document.querySelectorAll('#settings .theme-group')].map((g) =>
      [g.querySelector('.theme-group-h').textContent, g.querySelectorAll('.theme-card').length,
       document.getElementById(g.getAttribute('aria-labelledby')) === g.querySelector('.theme-group-h')])""")
    assert groups == [["Light & Dark", 5, True], ["Accessibility", 3, True], ["Editor themes", 10, True]], groups
    assert drv.execute_script("return document.querySelectorAll('#settings [role=radiogroup] .theme-card[role=radio]').length") == 18
    # pointing at a card previews it; moving off goes back
    ActionChains(drv).move_to_element(css(drv, '#settings [data-key="theme"][data-val="hc-dark"]')).perform()
    H.wait(drv, "return document.documentElement.getAttribute('data-theme') === 'hc-dark'", 5)
    assert drv.execute_script(TOKEN, "--bg") == "#000000"
    ActionChains(drv).move_to_element(css(drv, "#settings .dlg-head h2")).perform()
    H.wait(drv, "return document.documentElement.getAttribute('data-theme') === 'light'", 5)
    assert drv.execute_script("return window.__mace.settings.get('theme')") == "light"
    # a click keeps it: the page, the 3D view's background and the plot follow
    click(drv, css(drv, '#settings [data-key="theme"][data-val="sepia"]'))
    assert theme(drv) == "sepia" and drv.execute_script("return window.__mace.settings.get('theme')") == "sepia"
    assert css(drv, '#settings [data-val="sepia"]').get_attribute("aria-checked") == "true"
    assert drv.execute_script(TOKEN, "--bg") == "#e2d3b4"
    H.wait(drv, "return window.__mace.runView.state().drawn", 30)
    H.wait(drv, PLOT_TEXT_IS_THEMES, 15)                                 # the plot's text takes the theme's
    drv.execute_script("document.querySelectorAll('dialog[open]').forEach((x) => x.close())")
    open_doc(drv, "structures/MgO.cif")
    H.wait(drv, "return window.__mace.viewState().background === getComputedStyle(document.documentElement).getPropertyValue('--viewport-bg').trim()", 10)
    assert drv.execute_script("return window.__mace.viewState().background") == "#fbf6ea"
    # persisted: gui.json, the page's first paint, and a reload
    flush(drv)
    assert gui_json(xdg).get("theme") == "sepia"
    drv.refresh()
    H.wait(drv, "return document.documentElement.dataset.appReady === '1'")
    assert drv.execute_script("return window.__maceFirstPaint.theme") == "sepia" and theme(drv) == "sepia"
    assert not H.errors(drv), H.errors(drv)


def test_color_theme_command_previews_with_the_arrow_keys(drv, app):
    drv.execute_script("window.__mace.settings.set('theme', 'sepia')")
    drv.execute_script("window.__mace.run('view.themePicker')")
    H.wait(drv, "return document.getElementById('palette').open", 5)
    rows = drv.execute_script("return [...document.querySelectorAll('#palette .pal-row .t')].map((t) => t.textContent)")
    assert rows[:5] == ["System", "Light", "Dark", "Sepia", "True Black"] and "Paper" not in rows, rows
    assert drv.execute_script("return document.querySelector('#palette .pal-group').textContent") == "Light & Dark"
    # every theme in its group's section (the list draws only what is in view: the model has them all)
    model = drv.execute_script("return window.__mace.palette.results('').map((r) => [r.group, r.label])")
    assert len(model) == 18 and [g for g, _ in model][::5] == ["Light & Dark", "Accessibility", "Editor themes", "Editor themes"], model
    assert sorted({g for g, _ in model}) == ["Accessibility", "Editor themes", "Light & Dark"], model
    assert "Publication" not in [t for _, t in model] and ["Editor themes", "Tokyo Night Storm"] == model[-1], model
    assert drv.execute_script("return document.querySelector('#palette .pal-row[aria-selected=true] .t').textContent") == "Sepia"
    inp = css(drv, "#palette input")
    inp.send_keys(Keys.ARROW_DOWN)                                         # Sepia -> True Black: shown at once
    H.wait(drv, "return document.documentElement.getAttribute('data-theme') === 'black'", 5)
    assert drv.execute_script("return window.__mace.settings.get('theme')") == "sepia"
    inp.send_keys(Keys.ESCAPE)                                             # Esc: back to Sepia
    H.wait(drv, "return document.documentElement.getAttribute('data-theme') === 'sepia'", 5)
    drv.execute_script("window.__mace.run('view.themePicker')")
    H.wait(drv, "return document.getElementById('palette').open", 5)
    inp = css(drv, "#palette input")
    inp.send_keys("high contrast light")
    H.wait(drv, "return document.documentElement.getAttribute('data-theme') === 'hc-light'", 5)
    inp.send_keys(Keys.ENTER)                                              # Enter keeps it
    H.wait(drv, "return window.__mace.settings.get('theme') === 'hc-light'", 5)
    assert theme(drv) == "hc-light"
    # the plain command search: a theme command previews once the arrows reach it
    drv.execute_script("window.__mace.palette.open('>Theme: Publication')")
    time.sleep(0.2)
    assert theme(drv) == "hc-light"                                        # typing alone does not
    drv.execute_script("window.__mace.palette.close()")
    drv.execute_script("window.__mace.settings.set('theme', 'system')")
    assert theme(drv) is None
    assert not H.errors(drv), H.errors(drv)


def test_plot_background_and_3d_overlays_have_their_own_lightness(drv, app):
    """Settings › Appearance › Plot background: a light (dark) plot in a dark (light) theme draws its
    own surface, text and series, and an export has it too; Follow theme is see-through as before.
    A White 3D background in a dark theme gives the 3D view's overlays (the triad) the light set; one
    of the theme's own lightness changes nothing."""
    drv.execute_script("window.__mace.settings.set('theme', 'tokyo-night'); window.__mace.settings.set('plots.surface', 'light')")
    open_doc(drv, "outputs/Ag2Br3_opt_unfinished.out")
    H.wait(drv, "return document.querySelectorAll('#o-energy circle.pt, #o-energy path.pt').length >= 2", 60)
    click(drv, css(drv, "#o-e-plot"))
    H.wait(drv, "return window.__mace.runView && window.__mace.runView.state() && window.__mace.runView.state().drawn", 60)
    assert drv.execute_script("return document.documentElement.getAttribute('data-plot-surface')") == "light"
    lay = "const p = document.querySelector('#runplot-host .js-plotly-plot'); return p && [p._fullLayout.paper_bgcolor, p._fullLayout.font.color, getComputedStyle(p).backgroundColor]"
    H.wait(drv, f"{lay.replace('return p && ', 'const x = p && ')}; return x && x[1] === '#15181e'", 15)
    paper, text, bg = drv.execute_script(lay)
    assert paper == "#ffffff" and text == "#15181e" and bg == "rgb(255, 255, 255)", (paper, text, bg)
    assert drv.execute_script(TOKEN, "--bg") == "#16161e"                   # the window keeps its theme
    drv.execute_script("window.__mace.settings.set('plots.surface', 'theme')")
    H.wait(drv, PLOT_TEXT_IS_THEMES, 15)
    assert drv.execute_script("return document.documentElement.getAttribute('data-plot-surface')") is None
    assert drv.execute_script(lay)[0] in ("rgba(0, 0, 0, 0)", "rgba(0,0,0,0)")
    # the 3D view: White in a dark theme gives the triad the light set; Theme puts it back
    open_doc(drv, "structures/MgO.cif")
    triad = "const l = document.querySelector('#axis-triad line'); return l && getComputedStyle(l).stroke"
    H.wait(drv, triad.replace("return l &&", "return !!l ||"), 10)
    drv.execute_script("window.__mace.settings.set('background', 'white')")
    H.wait(drv, "return document.documentElement.getAttribute('data-vp-surface') === 'light'", 10)
    H.wait(drv, f"{triad.replace('return l && ', 'const c = l && ')}; return ['rgb(220, 38, 38)', 'rgb(21, 128, 61)', 'rgb(37, 99, 235)'].includes(c)", 10)
    assert drv.execute_script("return window.__mace.viewState().background") == "#ffffff"
    drv.execute_script("window.__mace.settings.set('theme', 'light')")      # white in a light theme: the theme's own
    H.wait(drv, "return !document.documentElement.hasAttribute('data-vp-surface')", 10)
    drv.execute_script("window.__mace.settings.set('background', 'black')")
    H.wait(drv, "return document.documentElement.getAttribute('data-vp-surface') === 'dark'", 10)
    drv.execute_script("window.__mace.settings.set('background', 'theme'); window.__mace.settings.set('theme', 'system')")
    H.wait(drv, "return !document.documentElement.hasAttribute('data-vp-surface')", 10)
    assert not H.errors(drv), H.errors(drv)


def test_element_colors_edit_save_default_reload_and_share(drv, app):
    _, base, xdg = app
    open_doc(drv, "structures/MgO.cif")
    vesta_mg = COLORS["schemes"]["vesta"]["Mg"]
    H.wait(drv, "return window.__mace.atomStyle(0).color.toLowerCase() === arguments[0]", 10, vesta_mg)
    drv.execute_script("window.__mace.run('view.elementColors')")
    H.wait(drv, "return document.getElementById('palette-editor') && document.getElementById('palette-editor').open", 10)
    assert len(drv.find_elements("css selector", "#palette-editor .pe-cell")) == 103
    click(drv, css(drv, '#palette-editor .pe-cell[data-sym="Mg"]'))
    assert css(drv, "#pe-sym").text == "Mg" and css(drv, "#pe-hex").get_attribute("value") == vesta_mg
    # a new color: every view draws it at once (the draft), the cell is marked changed
    hexbox = css(drv, "#pe-hex")
    hexbox.clear()
    hexbox.send_keys("#123456", Keys.ENTER)
    H.wait(drv, "return window.__mace.atomStyle(0).color.toLowerCase() === '#123456'", 10)
    assert "mod" in css(drv, '#palette-editor .pe-cell[data-sym="Mg"]').get_attribute("class")
    assert drv.execute_script("return window.MACE.elementColor('Mg')") == "#123456"
    # Reset gives back VESTA's Mg; the picker sets it again
    click(drv, css(drv, "#pe-reset-el"))
    H.wait(drv, "return window.__mace.atomStyle(0).color.toLowerCase() === arguments[0]", 10, vesta_mg)
    drv.execute_script("const c = document.getElementById('pe-color'); c.value = '#123456'; c.dispatchEvent(new Event('input', {bubbles: true}))")
    H.wait(drv, "return window.__mace.atomStyle(0).color.toLowerCase() === '#123456'", 10)
    # Save as New, then Set as Default
    name = css(drv, "#pe-name")
    name.clear()
    name.send_keys("Lab colors")
    click(drv, css(drv, "#pe-save"))
    H.wait(drv, "return document.getElementById('pe-pick').value === 'user:Lab colors'", 10)
    saved = palettes_file(xdg)
    assert saved["palettes"]["Lab colors"] == {"base": "vesta", "colors": {"Mg": "#123456"}}, saved
    click(drv, css(drv, "#pe-default"))
    H.wait(drv, "return window.__mace.settings.get('colorScheme') === 'user:Lab colors'", 10)
    H.wait(drv, "return true", 1)
    flush(drv)
    time.sleep(0.5)
    assert palettes_file(xdg)["default"] == "user:Lab colors"
    assert gui_json(xdg).get("colorScheme") == "user:Lab colors"
    # Export: the palette as JSON (the changes and every element)
    drv.execute_script("window.__exported = null; window.__maceExportHook = (t, kind) => { window.__exported = [kind, t]; }")
    click(drv, css(drv, "#pe-export"))
    kind, text = drv.execute_script("return window.__exported")
    out = json.loads(text)
    assert kind == "palette" and out["mace_element_palette"] == 1 and out["name"] == "Lab colors"
    assert out["colors"] == {"Mg": "#123456"} and len(out["elements"]) == 103 and out["elements"]["O"] == COLORS["schemes"]["vesta"]["O"]
    # Import that file: a second palette, same colors, shown in the editor
    drv.execute_script("""const dt = new DataTransfer(); dt.items.add(new File([arguments[0]], 'lab.json', {type: 'application/json'}));
        const i = document.getElementById('pe-file'); i.files = dt.files; i.dispatchEvent(new Event('change', {bubbles: true}))""", text)
    H.wait(drv, "return document.getElementById('pe-pick').value === 'user:Lab colors (2)'", 10)
    assert palettes_file(xdg)["palettes"]["Lab colors (2)"] == {"base": "vesta", "colors": {"Mg": "#123456"}}
    click(drv, css(drv, "#pe-delete"))
    click(drv, css(drv, '#palette-editor [data-pe="delete-yes"]'))
    H.wait(drv, "return !Object.keys(window.MACE.state.palettes.palettes).includes('Lab colors (2)')", 10)
    # closing keeps the default's colors
    drv.execute_script("document.getElementById('palette-editor').close()")
    H.wait(drv, "return window.__mace.atomStyle(0).color.toLowerCase() === '#123456'", 10)
    # a new tab with no localStorage (another browser): the server's profile and palettes file
    first = drv.current_window_handle
    drv.switch_to.new_window("tab")
    try:
        drv.get(H.login_url(base))
        H.wait(drv, "return document.documentElement.dataset.appReady === '1'")
        drv.execute_script("localStorage.clear()")
        drv.refresh()
        H.wait(drv, "return document.documentElement.dataset.appReady === '1'")
        assert drv.execute_script("return window.__mace.settings.get('colorScheme')") == "user:Lab colors"
        open_doc(drv, "structures/MgO.cif")
        H.wait(drv, "return window.__mace.atomStyle(0).color.toLowerCase() === '#123456'", 10)
        assert drv.execute_script("return document.querySelector('#opt-colors').value") == "user:Lab colors"
        assert not H.errors(drv), H.errors(drv)
    finally:
        drv.close()
        drv.switch_to.window(first)
    # back to the preset default: the file follows Settings › Color scheme
    drv.execute_script("window.__mace.settings.set('colorScheme', 'vesta')")
    H.wait(drv, "return window.MACE.state.palettes.default === 'vesta'", 10)
    assert palettes_file(xdg)["default"] == "vesta"
    assert not H.errors(drv), H.errors(drv)


def test_element_table_is_one_tab_stop_with_arrow_keys(drv, app):
    """The palette editor's table (1.1.6 review): a group, one tab stop (the picked element), the
    arrow keys move along it, Enter picks; each cell is announced with its color and says whether
    it is the one picked."""
    drv.execute_script("window.__mace.run('view.elementColors')")
    H.wait(drv, "return !!document.querySelector('#palette-editor .pe-cell')", 10)
    grid = css(drv, "#palette-editor .pe-grid")
    assert grid.get_attribute("role") == "group"
    stops = drv.execute_script("return [...document.querySelectorAll('#palette-editor .pe-cell')].filter((b) => b.tabIndex >= 0).map((b) => b.dataset.sym)")
    sel = drv.execute_script("return document.querySelector('#palette-editor .pe-cell.sel').dataset.sym")
    assert stops == [sel]
    label = css(drv, f'#palette-editor .pe-cell[data-sym="{sel}"]').get_attribute("aria-label")
    assert label.startswith(sel + " #") and css(drv, f'#palette-editor .pe-cell[data-sym="{sel}"]').get_attribute("aria-pressed") == "true"
    drv.execute_script("document.querySelector('#palette-editor .pe-cell[data-sym=\"Na\"]').focus()")
    ActionChains(drv).send_keys(Keys.ARROW_RIGHT).perform()                 # Na -> Mg
    assert drv.execute_script("return document.activeElement.dataset.sym") == "Mg"
    ActionChains(drv).send_keys(Keys.ARROW_UP).perform()                    # above Mg: Be
    assert drv.execute_script("return document.activeElement.dataset.sym") == "Be"
    ActionChains(drv).send_keys(Keys.ENTER).perform()
    H.wait(drv, "return document.querySelector('#palette-editor .pe-cell[data-sym=\"Be\"]').getAttribute('aria-pressed') === 'true'", 5)
    assert drv.execute_script("return [...document.querySelectorAll('#palette-editor .pe-cell')].filter((b) => b.tabIndex >= 0).map((b) => b.dataset.sym)") == ["Be"]
    assert css(drv, "#pe-sym").text == "Be"


def test_palette_api_checks_its_input(app):
    """The route behind the editor: a bad request is a 400 that names the problem and writes nothing."""
    import urllib.error
    import urllib.request
    _, base, xdg = app

    def post(body):
        r = urllib.request.Request(base + "api/palettes", data=json.dumps(body).encode(), method="POST",
                                   headers={"X-Mace-Token": H.TOKEN, "Content-Type": "application/json", "X-Requested-With": "mace-gui"})
        try:
            with urllib.request.urlopen(r, timeout=20) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    before = palettes_file(xdg)
    for bad in ({"op": "save", "name": "x", "colors": {"Mg": "purple"}}, {"op": "save", "name": "", "colors": {}},
                {"op": "default", "id": "user:nope"}, {"op": "import", "palette": {"name": "x"}}):
        status, r = post(bad)
        assert status == 400 and r.get("error"), (bad, status, r)
    assert palettes_file(xdg) == before
    status, r = post({"op": "save", "name": "API", "base": "jmol", "colors": {"O": "#0F0"}})
    assert status == 200 and r["id"] == "user:API" and r["palettes"]["API"] == {"base": "jmol", "colors": {"O": "#00ff00"}}
    status, r = post({"op": "delete", "name": "API"})
    assert status == 200 and "API" not in r["palettes"]


def side(drv, js):
    """Run js in the right editor group's page (W is its window)."""
    return drv.execute_script("const f = document.querySelector('.group-frame'); if (!f) return null; const W = f.contentWindow; " + js)


def test_the_right_editor_group_follows_theme_and_palette(drv, app):
    """The right group is its own page: it takes the window's theme, its preview and the palette being edited."""
    open_doc(drv, "structures/NaCl.cif")
    open_doc(drv, "structures/MgO.cif")
    drv.execute_script("window.__mace.run('view.splitRight')")
    H.wait(drv, "const f = document.querySelector('.group-frame'); const W = f && f.contentWindow;"
                "return !!(W && W.__macePane && W.document.documentElement.dataset.appReady === '1' && W.__macePane.active())", 60)
    H.wait(drv, "const W = document.querySelector('.group-frame').contentWindow; return W.__mace.atomStyle(0) !== null", 30)
    drv.execute_script("window.__mace.settings.set('theme', 'black')")
    H.wait(drv, "return document.querySelector('.group-frame').contentWindow.document.documentElement.getAttribute('data-theme') === 'black'", 10)
    drv.execute_script("window.__mace.run('view.elementColors')")
    H.wait(drv, "return document.getElementById('palette-editor') && document.getElementById('palette-editor').open", 10)
    side_sym = side(drv, "return W.MACE.state.structure.symbols[0]")
    drv.execute_script("""document.querySelector(`#palette-editor .pe-cell[data-sym="${arguments[0]}"]`).click();
        const c = document.getElementById('pe-color'); c.value = '#2b8c3e'; c.dispatchEvent(new Event('input', {bubbles: true}))""", side_sym)
    H.wait(drv, "const W = document.querySelector('.group-frame').contentWindow; const a = W.__mace.atomStyle(0); return a && a.color.toLowerCase() === '#2b8c3e'", 10)
    drv.execute_script("document.getElementById('palette-editor').close()")
    want = COLORS["schemes"]["vesta"][side_sym]
    H.wait(drv, "const W = document.querySelector('.group-frame').contentWindow; return W.__mace.atomStyle(0).color.toLowerCase() === arguments[0]", 10, want)
    drv.execute_script("window.__mace.settings.set('theme', 'system'); window.__mace.run('view.closeGroup')")
    assert not H.errors(drv), H.errors(drv)


# --- element legends (1.1.6): the dialog, the API (the CLI's function) and the legend in exported images ---

def test_legend_api_is_the_cli_function(app, monkeypatch):
    """/api/legend answers what mace plotting --legend writes for the same files and choices, byte for byte."""
    import base64
    import urllib.error
    import urllib.request
    from mace.utils import element_legend as EL
    _, base, xdg = app
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))

    def post(body):
        r = urllib.request.Request(base + "api/legend", data=json.dumps(body).encode(), method="POST",
                                   headers={"X-Mace-Token": H.TOKEN, "Content-Type": "application/json", "X-Requested-With": "mace-gui"})
        try:
            with urllib.request.urlopen(r, timeout=60) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    served = Path(json.loads(urllib.request.urlopen(urllib.request.Request(base + "api/info", headers={"X-Mace-Token": H.TOKEN}), timeout=20).read())["root"])
    opts = {"layout": "grid", "columns": 2, "label": "both", "background": "white"}
    for fmt in ("svg", "tsv", "json", "png"):
        st, r = post({"paths": ["structures", "outputs/Si_sp.out"], "format": fmt, "options": opts})
        assert st == 200, r
        cli = EL.legend([served / "structures", served / "outputs" / "Si_sp.out"], fmt=fmt, options=opts)
        got = base64.b64decode(r["content"]) if r["encoding"] == "base64" else r["content"].encode()
        assert got == cli.content, fmt
        assert r["symbols"] == cli.symbols and r["files"] == 7
    st, r = post({"symbols": ["Fe", "O"], "palette": "jmol", "format": "svg"})
    assert st == 200 and f'fill="{COLORS["schemes"]["jmol"]["Fe"]}"' in r["content"]
    for bad in ({"paths": ["../../etc"]}, {"paths": ["structures"], "format": "pdf"}, {}, {"symbols": ["Qq"]}):
        st, r = post(bad)
        assert st in (400, 403, 404), (bad, st, r)


def test_export_element_legend_dialog(drv, app):
    import base64
    open_doc(drv, "structures/MgO.cif")
    drv.execute_script("window.__mace.run('file.exportLegend')")
    ready = "const d = document.getElementById('legend-dialog'); return !!(d && d.open && d.dataset.ready && document.getElementById('lg-img').complete)"
    H.wait(drv, ready, 15)
    msg = css(drv, "#lg-msg").text
    assert msg.startswith("2 elements: O Mg"), msg
    # live preview: a grid with names; the preview is the server's SVG
    seq = drv.execute_script("return document.getElementById('legend-dialog').dataset.ready")
    drv.execute_script("""const d = document.getElementById('legend-dialog'), s = (k, v) => { const e = d.querySelector(`[data-k="${k}"]`); e.value = v;
        e.dispatchEvent(new Event('change', {bubbles: true})); }; s('layout', 'grid'); s('label', 'both');""")
    H.wait(drv, f"return document.getElementById('legend-dialog').dataset.ready !== '{seq}'", 10)
    assert not css(drv, "#legend-dialog .lg-cols").get_attribute("hidden")
    drv.execute_script("window.__exported = null; window.__maceExportHook = (t, kind) => { window.__exported = [kind, t]; }")
    click(drv, css(drv, '#legend-dialog [data-v="export"]'))
    H.wait(drv, "return !!window.__exported", 10)
    kind, text = drv.execute_script("return window.__exported")
    assert kind == "legend-svg" and 'id="legend-Mg"' in text and f'fill="{COLORS["schemes"]["vesta"]["Mg"]}"' in text
    assert "Magnesium" in text
    # PNG, then a folder from the Explorer's menu (the union of its structures)
    drv.execute_script("window.__mace.run('file.exportLegend')")
    H.wait(drv, ready, 15)
    drv.execute_script("const e = document.querySelector('#legend-dialog [data-k=format]'); e.value = 'png'; e.dispatchEvent(new Event('change', {bubbles: true}))")
    assert not css(drv, "#legend-dialog .lg-dpi").get_attribute("hidden")
    drv.execute_script("window.__exported = null")
    click(drv, css(drv, '#legend-dialog [data-v="export"]'))
    H.wait(drv, "return !!window.__exported", 10)
    kind, b64 = drv.execute_script("return window.__exported")
    assert kind == "legend-png" and base64.b64decode(b64)[:4] == b"\x89PNG"
    drv.execute_script("window.__mace.run('explorer.legend', {dir: 'structures'})")
    H.wait(drv, "const m = document.getElementById('lg-msg'); return m && /from 6 files/.test(m.textContent)", 15)
    assert css(drv, "#lg-msg").text.startswith("10 elements: Li B C N O F Na Mg Si Cl")
    drv.execute_script("document.getElementById('legend-dialog').close()")
    # the legend in an exported structure image's corner (SVG: its own editable group)
    open_doc(drv, "structures/MgO.cif")
    drv.execute_script("window.__exported = null")
    drv.execute_async_script("const d = arguments[0]; window.__mace.exportSvg({legend: 'top-right'}).then(() => d(1), (e) => d(String(e)))")
    kind, text = drv.execute_script("return window.__exported")
    assert kind == "svg" and 'id="legend-corner"' in text and 'id="legend-O"' in text and 'id="legend-Mg"' in text
    assert not H.errors(drv), H.errors(drv)
