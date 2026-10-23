"""The `mace gui` app shell in headless Firefox (spec Phase 1): menus,
toolbar, tabs, status bar, shortcuts, palette, undo/redo, settings, themes.

Set MACE_GUI_SHOTS=<dir> to also save the screenshot matrix there.
"""
import json
import os
import time
from pathlib import Path

import pytest

import guiharness as H

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")

from selenium.webdriver.common.action_chains import ActionChains  # noqa: E402
from selenium.webdriver.common.keys import Keys  # noqa: E402
from selenium.webdriver.support.ui import Select  # noqa: E402

MENU_ORDER = ["File", "Edit", "View", "Select", "Build", "Tools", "Calculation", "Analysis", "Help"]


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("guishell")
    root, xdg = H.examples_copy(tmp), tmp / "xdg"
    proc, base, _ = H.start_server(root, xdg)
    drv = H.firefox(1280, 800)
    try:
        H.load(drv, base)
        if not drv.execute_script("return !!document.createElement('canvas').getContext('webgl')"):
            pytest.skip("no WebGL in this headless Firefox")
        yield drv, base, root, xdg
    finally:
        drv.quit()
        proc.terminate()
        proc.wait(timeout=15)


@pytest.fixture
def fresh(app):
    """Every test starts on the welcome screen at 1280x800, light theme."""
    drv, base, root, xdg = app
    drv.execute_script("window.__mace.closeAll(); window.__mace.settings.resetAll(); window.__mace.settings.set('theme', 'light');"
                       "const set = (id, v) => { const e = document.getElementById(id); if (e.checked !== v) { e.checked = v; e.dispatchEvent(new Event('change')); } };"
                       "set('opt-bonds', true); set('opt-super', false);"
                       "const st = document.getElementById('opt-style'); if (st.value !== 'ballstick') { st.value = 'ballstick'; st.dispatchEvent(new Event('change')); }"
                       "window.__mace.run('tool.select');"
                       "document.querySelectorAll('dialog[open]').forEach((d) => d.close());"
                       "document.querySelectorAll('#toasts > *').forEach((t) => t.remove());")
    H.set_viewport(drv, 1280, 800)
    time.sleep(0.3)   # the resize event (which closes menus) lands first
    drv.execute_script("document.body.focus()")
    return app


def keys(drv, *combo):
    """Press a chord, e.g. keys(drv, Keys.CONTROL, 'k')."""
    ac = ActionChains(drv)
    for k in combo[:-1]:
        ac.key_down(k)
    ac.send_keys(combo[-1])
    for k in reversed(combo[:-1]):
        ac.key_up(k)
    ac.perform()
    time.sleep(0.15)


def body_focus(drv):
    drv.execute_script("document.activeElement && document.activeElement.blur(); document.body.focus()")


def open_doc(drv, path):
    return drv.execute_async_script(
        "const done = arguments[1]; window.__mace.open(arguments[0]).then((d) => done(d && d.kind))", path)


def shot(drv, name):
    target = os.environ.get("MACE_GUI_SHOTS")
    if target:
        Path(target).mkdir(parents=True, exist_ok=True)
        drv.save_screenshot(str(Path(target) / f"{name}.png"))


def open_menu(drv, label):
    drv.execute_script(
        "const b = [...document.querySelectorAll('.menu-top')].find((x) => x.textContent === arguments[0]);"
        "b.dispatchEvent(new PointerEvent('pointerdown', {bubbles: true, button: 0}))", label)
    H.wait(drv, "return document.querySelectorAll('.menu').length > 0")


# --- 1, 2: menus ----------------------------------------------------------------------

@pytest.mark.parametrize("size", [(1280, 800), (1920, 1080)])
@pytest.mark.parametrize("theme", ["light", "dark"])
def test_menu_bar_order_and_items_match_the_registry(fresh, size, theme):
    drv = fresh[0]
    H.set_viewport(drv, *size)
    drv.execute_script("window.__mace.settings.set('theme', arguments[0])", theme)
    assert open_doc(drv, "structures/MgO.cif") == "structure"
    labels = drv.execute_script("return [...document.querySelectorAll('.menu-top')].map((b) => b.textContent)")
    assert labels == MENU_ORDER
    model = drv.execute_script("return window.__mace.menus()")
    cmds = {c["id"]: c for c in drv.execute_script("return window.__mace.commands")}
    for m in model:
        open_menu(drv, m["label"])
        rows = drv.execute_script("""return [...document.querySelector('.menu').children].map((el) =>
            el.classList.contains('menu-sep') ? '-' : {id: el.dataset.cmdId || null,
              label: el.querySelector('.lbl').textContent, kbd: (el.querySelector('kbd') || {}).textContent || '',
              kbdRight: el.querySelector('kbd') ? el.querySelector('kbd').getBoundingClientRect().right : 0,
              right: el.getBoundingClientRect().right})""")
        assert len(rows) == len(m["items"]), m["label"]
        for it, row in zip(m["items"], rows):
            if it == "-":
                assert row == "-"
            elif isinstance(it, str):
                c = cmds[it]
                # the label comes from the registry, never from the menu
                assert row["id"] == it and row["label"] == c["menuTitle"], (it, row)
                if c["keys"]:
                    shown = drv.execute_script("return window.__mace.formatChord(arguments[0])", c["keys"][0])
                    assert row["kbd"] == shown, it
                    assert row["right"] - row["kbdRight"] < 40, it   # right-aligned
            else:
                assert row["label"] == it["label"]
        shot(drv, f"menu_{m['label']}_{theme}_{size[0]}")
        keys(drv, Keys.ESCAPE)
        assert not drv.execute_script("return document.querySelectorAll('.menu').length")
    assert not H.errors(drv), H.errors(drv)


def test_menus_open_with_the_keyboard(fresh):
    drv = fresh[0]
    body_focus(drv)
    keys(drv, Keys.ALT, "v")       # Alt+letter opens View on Linux
    assert drv.execute_script("return document.querySelector('.menu').getAttribute('aria-label')") == "View"
    keys(drv, Keys.ARROW_RIGHT)    # to the next menu
    assert drv.execute_script("return document.querySelector('.menu').getAttribute('aria-label')") == "Select"
    keys(drv, Keys.ESCAPE)
    assert not drv.execute_script("return document.querySelectorAll('.menu').length")


def test_unbuilt_items_name_their_owner(fresh):
    """An unfinished command says what it arrives with (Phase 4 carry-over d:
    no "Coming soon"); the Build menu's builders are real since Phase 4."""
    drv = fresh[0]
    open_menu(drv, "Help")
    docs = drv.execute_script("return document.querySelector('.menu [data-cmd-id=\"help.docs\"]')")
    assert docs.get_attribute("aria-disabled") != "true" and "Not in this version" not in (docs.get_attribute("title") or "")
    keys(drv, Keys.ESCAPE)
    # Open to the Side arrived with the split editor (plan E11): a real command now
    assert drv.execute_script("return window.__mace.menuReason('explorer.openToSide')") == "Pick a file in the Explorer"
    assert drv.execute_script("return window.__mace.menuReason('file.export.QE')") == "Needs the Quantum ESPRESSO backend"


# --- 3: toolbar and tooltips -------------------------------------------------------------

def test_toolbar_groups_and_every_tooltip(fresh):
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    groups = drv.execute_script("""const t = document.querySelector('#toolbar');
        return {tools: t.querySelectorAll('[role=radiogroup] [data-cmd^="tool."]').length,
          toggles: ['view.cell','view.bonds','view.labelsCycle','view.boundary','view.axes','view.projection']
             .every((id) => t.querySelector(`[data-cmd="${id}"]`)),
          along: !!t.querySelector('#tb-along'), range: !!t.querySelector('#tb-range'),
          sym: t.querySelector('.segment').textContent, clean: !!t.querySelector('[data-cmd="build.clean"]'),
          setup: !!t.querySelector('[data-cmd="calc.setup"]')}""")
    assert groups == {"tools": 11, "toggles": True, "along": True, "range": True, "sym": "P1Sym",
                      "clean": True, "setup": True}
    buttons = drv.find_elements("css selector", "#toolbar [data-cmd], #toolbar [data-tip]")
    cmds = {c["id"]: c for c in drv.execute_script("return window.__mace.commands")}
    for b in buttons:
        ActionChains(drv).move_to_element(b).perform()
        H.wait(drv, "return !document.querySelector('#tooltip').hidden", 3)
        tip = drv.execute_script("return document.querySelector('#tooltip').textContent")
        cid = b.get_attribute("data-cmd")
        if cid:
            c = cmds[cid]
            assert c["title"].split(" (")[0][:12] in tip or "Projection" in tip or "Display range" in tip \
                or "Save" in tip or "Theme" in tip, (cid, tip)
            if c["keys"]:
                assert drv.execute_script("return window.__mace.formatChord(arguments[0])", c["keys"][0]) in tip, (cid, tip)
        else:
            # split buttons: their names and keys come from the registry
            assert b.get_attribute("data-tip") in tip, tip
            if b.get_attribute("id") == "tb-along":
                assert "1 / 2 / 3" in tip and "a* Shift+1" in tip, tip
            if b.get_attribute("id") == "tb-range":
                assert "Shift+R" in tip, tip
    ActionChains(drv).move_by_offset(0, 300).perform()


# --- 4, 5, 10: status bar, documents, welcome ----------------------------------------------

def test_status_bar_for_mgo(fresh):
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    H.wait(drv, "return document.querySelector('#st-sel')")
    text = drv.execute_script("return document.querySelector('#statusbar').innerText")
    assert "Connected" in text and "0 of 8 atoms" in text and "Fm-3m (225)" in text
    assert "a 4.212 b 4.212 c 4.212 Å · 90 90 90°" in text
    assert drv.execute_script("return document.querySelector('#st-conn .st-dot').classList.contains('ok')")


def test_three_documents_and_the_trajectory_timeline(fresh):
    drv = fresh[0]
    assert drv.execute_script("return document.querySelector('#tab-welcome').classList.contains('active')")
    assert open_doc(drv, "structures/MgO.cif") == "structure"
    # its first view is complete: the Display panel (loaded with the first structure) has read the
    # structure's spin moments once (the Magnetic style), which a switch back must not read again
    H.wait(drv, "const d = window.__mace.doc(); return !!d.spins && d.spins.state !== 'loading'", 10)
    assert open_doc(drv, "outputs/Ag2Br3_opt_unfinished.out") == "trajectory"
    assert drv.execute_script("return document.querySelector('#tab-opt').classList.contains('active')")
    assert drv.execute_script("return document.querySelector('.bottom-tab.active').dataset.bottom") == "timeline"
    assert drv.execute_script("return !!document.querySelector('.bottom-pane.timeline.active #o-energy svg')")
    assert open_doc(drv, "materials.db") == "db"
    tabs = drv.execute_script("return [...document.querySelectorAll('.doc-tab .t')].map((t) => t.textContent)")
    assert tabs == ["MgO.cif", "Ag2Br3_opt_unfinished.out", "materials.db"]
    shot(drv, "three_tabs")
    # switching back costs no request for a structure (its payload is kept);
    # a settings or profile save started earlier (Recent, polish P2) is let out first, so it is not counted
    drv.execute_async_script("Promise.all([window.__mace.settings.flush(), window.__mace.profile.flush()]).then(() => arguments[0]())")
    # a finished request's Resource Timing entry is queued a moment later (and the
    # database document's first page and property list are still arriving): let both land
    time.sleep(0.4)
    n = drv.execute_script("return performance.getEntriesByType('resource').length")
    drv.execute_script("window.__mace.activate(0)")
    time.sleep(0.5)
    new = drv.execute_script("return performance.getEntriesByType('resource').slice(arguments[0]).map((r) => r.name)", n)
    assert not [u for u in new if "/api/" in u], new
    keys(drv, Keys.ALT, Keys.PAGE_DOWN)
    assert drv.execute_script("return window.__mace.doc().title") == "Ag2Br3_opt_unfinished.out"
    drv.execute_script("window.__mace.run('file.closeAll')")
    assert drv.execute_script("return document.querySelector('#tab-welcome').classList.contains('active')")
    welcome = drv.find_element("id", "tab-welcome").text.lower()
    assert "recent" in welcome and "mgo.cif" in welcome   # just opened
    assert not H.errors(drv), H.errors(drv)



def test_welcome_says_how_many_more_structures(fresh):
    """The welcome card lists three CIFs; the rest are one click away."""
    drv = fresh[0]
    card = drv.execute_script("""const c = [...document.querySelectorAll('#tab-welcome .welcome-card')]
        .find((x) => x.querySelector('h3').textContent === 'Structures in this folder');
        return [c.querySelectorAll('[data-open]').length, (c.querySelector('[data-more-structures]') || {}).textContent]""")
    assert card == [3, "+3 more in the Explorer"], card
    drv.execute_script("document.querySelector('[data-more-structures]').click()")
    H.wait(drv, "return document.querySelector('.kind-chip[data-f=struct]').classList.contains('active')", 5)
    assert not H.errors(drv), H.errors(drv)


def test_opt_view_keeps_a_usable_3d_view_at_1280x800(fresh):
    """The before/after view keeps at least 150 px at 100 % UI scale with the bottom panel
    open, while the tab (toolbar and legend included) fits without scrolling
    (test_compare_e2e.py checks 110 %, where the tab may scroll a little)."""
    drv = fresh[0]
    open_doc(drv, "outputs/Ag2Br3_opt_unfinished.out")
    H.wait(drv, "return document.querySelector('#opt-viewers').getBoundingClientRect().height > 0", 10)
    h = drv.execute_script("return document.querySelector('#opt-viewers').offsetHeight")
    assert h >= 150, h
    tab = drv.execute_script("const t = document.getElementById('tab-opt'); return t.scrollHeight - t.clientHeight")
    assert tab <= 1, tab
    assert not H.errors(drv), H.errors(drv)


# --- shortcuts ------------------------------------------------------------------------------

def test_shortcuts(fresh):
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    time.sleep(0.5)
    body_focus(drv)
    keys(drv, Keys.CONTROL, Keys.ALT, "k")
    assert drv.execute_script("return document.querySelector('#palette').open")
    keys(drv, Keys.ESCAPE)
    assert not drv.execute_script("return document.querySelector('#palette').open")
    keys(drv, Keys.F1)
    assert drv.execute_script("return document.querySelector('#palette').open")
    keys(drv, Keys.ESCAPE)
    body_focus(drv)
    keys(drv, "?")
    H.wait(drv, "return document.querySelector('#cheatsheet') && document.querySelector('#cheatsheet').open")   # loads on first open
    keys(drv, Keys.ESCAPE)
    body_focus(drv)
    before = drv.execute_script("return window.__mace.viewState()")
    keys(drv, Keys.SHIFT, "b")
    assert drv.execute_script("return window.__mace.viewState().bonds") is (not before["bonds"])
    keys(drv, Keys.SHIFT, "b")
    keys(drv, "1")
    view_a = drv.execute_script("return window.__mace.viewState().view")
    keys(drv, "3")
    view_c = drv.execute_script("return window.__mace.viewState().view")
    assert view_a != view_c and view_a != before["view"]
    keys(drv, "o")    # Orbit tool
    assert drv.execute_script("return window.__mace.viewState().tool") == "orbit"
    keys(drv, Keys.ESCAPE)
    assert drv.execute_script("return window.__mace.viewState().tool") == "select"
    # a single-key shortcut never fires while typing
    drv.find_element("id", "file-filter").click()
    keys(drv, "o")
    assert drv.execute_script("return window.__mace.viewState().tool") == "select"
    assert drv.execute_script("return document.querySelector('#file-filter').value") == "o"
    drv.execute_script("const f = document.querySelector('#file-filter'); f.value = '';"
                       "f.dispatchEvent(new Event('input'))")
    body_focus(drv)
    keys(drv, Keys.CONTROL, "b")
    assert drv.execute_script("return document.querySelector('#shell').classList.contains('no-explorer')")
    keys(drv, Keys.CONTROL, "b")
    keys(drv, Keys.CONTROL, "j")
    assert drv.execute_script("return document.querySelector('#shell').classList.contains('no-bottom')")
    keys(drv, Keys.CONTROL, "j")
    keys(drv, Keys.CONTROL, Keys.ALT, "t")
    assert drv.execute_script("return window.__mace.settings.get('theme')") == "dark"
    assert not H.errors(drv), H.errors(drv)


def test_keymap_parsing_is_layout_safe(fresh):
    drv = fresh[0]
    c = drv.execute_script("return [window.__mace.parseChord('Mod+Shift+Z'), window.__mace.parseChord('?'),"
                           "window.__mace.parseChord('Mod+,'), window.__mace.parseChord('Alt+PageDown')]")
    assert c[0]["code"] == "KeyZ" and c[0]["mod"] and c[0]["shift"]
    assert c[1]["key"] == "?" and c[2]["code"] == "Comma" and c[3]["code"] == "PageDown" and c[3]["alt"]
    reserved = {"Mod+N", "Mod+T", "Mod+W", "Mod+Shift+N", "Mod+Shift+T", "Mod+Shift+W", "Mod+Tab",
                "Mod+L", "Mod+Q", "Mod+Shift+P", "Mod+PageUp", "Mod+PageDown", "F5", "F11", "F12"}
    bound = {k for c in drv.execute_script("return window.__mace.commands") for k in c["keys"]}
    assert not bound & reserved
    assert not [k for k in bound if k.startswith("Ctrl+") and k[5:].isdigit()]


# --- palette ----------------------------------------------------------------------------------

def test_palette_filters_runs_and_returns_focus(fresh):
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    filt = drv.find_element("id", "file-filter")
    filt.click()
    keys(drv, Keys.CONTROL, Keys.ALT, "k")
    box = drv.find_element("css selector", "#palette input")
    assert drv.execute_script("return document.activeElement") == box
    box.send_keys("show bonds")
    first = drv.execute_script("return document.querySelector('.pal-row[aria-selected=true]').dataset.id")
    assert first == "view.bonds"
    kbd = drv.execute_script("return document.querySelector('.pal-row[aria-selected=true] kbd').textContent")
    assert kbd == "Shift+B"
    bonds = drv.execute_script("return window.__mace.viewState().bonds")
    box.send_keys(Keys.ENTER)
    assert drv.execute_script("return window.__mace.viewState().bonds") is (not bonds)
    assert not drv.execute_script("return document.querySelector('#palette').open")
    assert drv.execute_script("return document.activeElement.id") == "file-filter"
    # aliases, files and the reason of an unfinished command
    assert drv.execute_script("return window.__mace.palette.results('gamma')[0].id") == "edit.editCell"
    rows = drv.execute_script("return window.__mace.palette.results('#mgo')")
    assert rows[0]["label"] == "structures/MgO.cif"
    assert not drv.execute_script("return window.__mace.palette.results('documentation')[0].why")


# --- undo / redo ------------------------------------------------------------------------------

def test_undo_redo_and_dirty_state(fresh):
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    # a structure document is edited in place (Phase 3); the classic panel's Add still adds
    # (in P1: MgO opens in Sym mode, where Li at 8c would add the whole orbit)
    H.wait(drv, "return window.__mace.doc() && window.__mace.doc().kind === 'structure' && window.MaceEditor.atoms() === 8")
    drv.execute_script("window.__mace.run('build.symModeP1')")
    drv.execute_script("""document.querySelector('#e-element').value = 'Li';
        ['#e-x', '#e-y', '#e-z'].forEach((id) => document.querySelector(id).value = '0.25');
        document.querySelector('#e-add').click()""")
    H.wait(drv, "return window.MaceEditor.atoms() === 9")
    H.wait(drv, "return window.__mace.doc().dirty")
    time.sleep(0.2)
    assert drv.execute_script("return document.title").startswith("● ")
    assert drv.execute_script("return document.querySelector('.doc-tab.active').classList.contains('dirty')")
    assert "Modified" in drv.execute_script("return document.querySelector('#st-doc').textContent")
    assert not drv.execute_script("return document.querySelector('#toolbar .dirty-dot').hidden")
    body_focus(drv)
    keys(drv, Keys.CONTROL, "z")
    H.wait(drv, "return window.MaceEditor.atoms() === 8")
    H.wait(drv, "return !window.__mace.doc().dirty")
    keys(drv, Keys.CONTROL, Keys.SHIFT, "z")
    H.wait(drv, "return window.MaceEditor.atoms() === 9")
    keys(drv, Keys.CONTROL, "z")
    H.wait(drv, "return window.MaceEditor.atoms() === 8")
    keys(drv, Keys.CONTROL, "y")
    H.wait(drv, "return window.MaceEditor.atoms() === 9")
    # Mod+Z in a text field is the field's own undo, not the structure's
    drv.find_element("id", "file-filter").click()
    keys(drv, Keys.CONTROL, "z")
    time.sleep(0.4)
    assert drv.execute_script("return window.MaceEditor.atoms()") == 9
    drv.execute_script("window.MaceEditor.undo()")
    H.wait(drv, "return window.MaceEditor.atoms() === 8")
    assert not H.errors(drv), H.errors(drv)


# --- settings ---------------------------------------------------------------------------------

def test_settings_persist_across_reload_and_in_the_config_file(fresh):
    drv, base, _, xdg = fresh
    body_focus(drv)
    keys(drv, Keys.CONTROL, ",")
    H.wait(drv, "return document.querySelector('#settings') && document.querySelector('#settings').open")
    drv.execute_script("document.querySelector('#settings [data-key=\"density\"][data-val=\"compact\"]').click()")
    drv.execute_script("window.__mace.settings.set('theme', 'dark')")
    drv.execute_async_script("const done = arguments[0]; window.__mace.settings.flush().then(done)")
    stored = json.loads((xdg / "mace" / "gui.json").read_text())
    assert stored["settings"] == {"density": "compact", "theme": "dark"}   # non-default keys only
    drv.refresh()
    H.wait(drv, "return document.documentElement.dataset.appReady === '1'")
    assert drv.execute_script("return [document.documentElement.dataset.theme, document.documentElement.dataset.density]") == ["dark", "compact"]
    drv.execute_script("window.__mace.settings.set('density', 'comfortable'); window.__mace.settings.set('theme', 'system')")
    drv.execute_async_script("const done = arguments[0]; window.__mace.settings.flush().then(done)")
    assert json.loads((xdg / "mace" / "gui.json").read_text())["settings"] == {}


# --- 6, 7, 8, 9: language, tokens, clipping, theme parity -----------------------------------

def test_dark_scrollbars_are_themed(fresh):
    drv = fresh[0]
    drv.execute_script("window.__mace.settings.set('theme', 'dark')")
    time.sleep(0.3)
    color = drv.execute_script("return getComputedStyle(document.documentElement).scrollbarColor")
    assert color and "255, 255, 255" not in color and color != "auto", color
    track = drv.execute_script("return getComputedStyle(document.querySelector('#file-list')).scrollbarColor")
    assert "255, 255, 255" not in track


@pytest.mark.parametrize("size", [(1280, 800), (1920, 1080)])
def test_no_label_is_clipped(fresh, size):
    drv = fresh[0]
    H.set_viewport(drv, *size)
    for path in ("structures/MgO.cif", "outputs/Ag2Br3_opt_unfinished.out", "materials.db"):
        open_doc(drv, path)
        time.sleep(0.4)
        clipped = drv.execute_script("""
          return [...document.querySelectorAll('button, [role=menuitem], th, .label, .section-label, .insp-head, .st-item')]
            .filter((el) => el.offsetParent !== null && !el.closest('.file, .doc-tab, .nm.mid, #parked'))
            .filter((el) => el.scrollWidth > el.clientWidth + 1)
            .map((el) => (el.id || el.className) + ': ' + el.textContent.trim().slice(0, 40))""")
        assert not clipped, (path, clipped)
    # explorer names: a middle ellipsis, with the full path as the tooltip
    names = drv.execute_script("""return [...document.querySelectorAll('#file-list .file')].map((b) =>
        [b.title.split(' · ')[0], b.querySelector('.mid-a').textContent + b.querySelector('.mid-b').textContent])""")
    assert all(full.endswith(name) for full, name in names)


def test_light_and_dark_have_the_same_layout(fresh):
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    sel = "#menubar, #toolbar, #explorer, #doc-tabs, #inspector, #bottom, #statusbar, .menu-top, #toolbar [data-cmd], .doc-tab, .insp-head, .st-item"
    boxes = {}
    for theme in ("light", "dark"):
        drv.execute_script("window.__mace.settings.set('theme', arguments[0])", theme)
        time.sleep(0.4)
        boxes[theme] = drv.execute_script(
            "return [...document.querySelectorAll(arguments[0])].map((e) => { const r = e.getBoundingClientRect();"
            "return [r.x, r.y, r.width, r.height]; })", sel)
        shot(drv, f"structure_{theme}")
    assert len(boxes["light"]) == len(boxes["dark"])
    worst = max(abs(a - b) for la, da in zip(boxes["light"], boxes["dark"]) for a, b in zip(la, da))
    assert worst <= 1, worst


def test_only_local_requests_and_no_console_errors(fresh):
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    urls = drv.execute_script("return performance.getEntriesByType('resource').map((r) => r.name)")
    origin = drv.execute_script("return location.origin")
    assert urls and all(u.startswith(origin) for u in urls), [u for u in urls if not u.startswith(origin)]
    assert not H.errors(drv), H.errors(drv)


def test_plotly_loads_only_when_a_plot_opens(app, tmp_path):
    """3Dmol after first paint; plotly never, until a band or DOS plot opens."""
    drv = app[0]
    drv.refresh()
    H.wait(drv, "return document.documentElement.dataset.appReady === '1'")
    open_doc(drv, "structures/MgO.cif")
    open_doc(drv, "outputs/Ag2Br3_opt_unfinished.out")
    time.sleep(0.5)
    names = drv.execute_script("return performance.getEntriesByType('resource').map((r) => r.name)")
    assert not [n for n in names if "plotly" in n]
    fcp = drv.execute_script("const p = performance.getEntriesByName('first-contentful-paint')[0]; return p ? p.startTime : null")
    three = drv.execute_script("const r = performance.getEntriesByType('resource').find((x) => x.name.includes('3Dmol')); return r ? r.startTime : null")
    assert three is not None and (fcp is None or three >= fcp - 1)


def test_headless_browser_renders_in_software(app):
    """Spec section 9: tests never load a GPU. WebGL must come from Mesa's
    software rasterizer, not the NVIDIA driver."""
    drv = app[0]
    renderer = H.webgl_renderer(drv).lower()
    assert renderer and any(s in renderer for s in ("llvmpipe", "softpipe", "swiftshader")), renderer


# --- review fixes: layout, triad, overlays, contrast -------------------------------------------

def test_inspector_gets_its_width_back_after_setup(fresh):
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    width = "return document.querySelector('#inspector').getBoundingClientRect().width"
    before = drv.execute_script(width)
    drv.execute_script("window.__mace.run('calc.setup')")
    time.sleep(0.3)
    assert drv.execute_script(width) >= 459
    drv.execute_script("window.__mace.run('calc.setup')")
    time.sleep(0.3)
    assert abs(drv.execute_script(width) - before) <= 1


@pytest.mark.parametrize("size", [(1280, 800), (1920, 1080)])
def test_database_table_fills_the_center(fresh, size):
    drv = fresh[0]
    H.set_viewport(drv, *size)
    open_doc(drv, "materials.db")
    H.wait(drv, "return document.querySelectorAll('#tab-db .dbv-tr[data-id]').length > 0", 20)
    time.sleep(0.3)
    # the virtual table (Phase 5): it fills the center; its cells never wrap (fixed-height rows)
    got = drv.execute_script("""const c = document.querySelector('#tab-db').getBoundingClientRect();
      const t = document.querySelector('#tab-db .dbv-table').getBoundingClientRect();
      const wrapped = [...document.querySelectorAll('#tab-db .dbv-td')].filter((td) => td.scrollHeight > td.clientHeight + 2).length;
      return {center: c.width, card: t.width, wrapped}""")
    assert got["card"] >= 0.9 * (got["center"] - 2 * 20), got
    assert got["wrapped"] == 0, got


def test_axis_triad_labels_clear_the_origin(fresh):
    """An axis seen end-on still gets a readable label, away from the others."""
    drv = fresh[0]
    svg = drv.execute_script("""return window.__mace.triadSvg(
        [{k: 1, x: 0.3, y: -0.4, z: -0.98}, {k: 2, x: -4, y: -21, z: 0.1}, {k: 0, x: 21, y: 6, z: 0.1}],
        () => 'red')""")
    import re
    labels = {m[2]: (float(m[0]), float(m[1])) for m in re.findall(r'<text x="([-\d.]+)" y="([-\d.]+)">(\w)</text>', svg)}
    assert set(labels) == {"a", "b", "c"}
    assert all((x * x + y * y) ** 0.5 >= 14 - 1e-6 for x, y in labels.values()), labels
    assert 'opacity="0.6"' in svg      # the far, end-on axis is drawn fainter
    open_doc(drv, "structures/MgO.cif")
    H.wait(drv, "return document.querySelectorAll('#axis-triad text').length === 3")
    box = drv.execute_script("const r = document.querySelector('#axis-triad').getBoundingClientRect(); return [r.width, r.height]")
    assert box == [80, 80]


def test_tooltips_and_menus_give_way_to_overlays(fresh):
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    b = drv.find_element("css selector", '#toolbar [data-cmd="tool.orbit"]')
    ActionChains(drv).move_to_element(b).perform()
    H.wait(drv, "return !document.querySelector('#tooltip').hidden", 3)
    shown = "return getComputedStyle(document.querySelector('#tooltip')).display !== 'none'"
    assert drv.execute_script(shown)
    drv.execute_script("window.dispatchEvent(new Event('resize'))")
    # hidden for real: the attribute and what is drawn
    assert drv.execute_script("return document.querySelector('#tooltip').hidden")
    assert not drv.execute_script(shown)
    ActionChains(drv).move_by_offset(0, 300).perform()
    # a menu open when a dialog or the palette opens closes first
    drv.find_element("css selector", ".menu-top").click()
    H.wait(drv, "return !!document.querySelector('.menu')", 3)
    drv.execute_script("window.__mace.run('help.about')")
    assert drv.execute_script("return !document.querySelector('.menu') && document.querySelector('#about').open")
    assert drv.execute_script("return document.activeElement === document.querySelector('#about')")
    drv.execute_script("document.querySelector('#about').close()")
    drv.find_element("css selector", ".menu-top").click()
    drv.execute_script("window.__mace.palette.open('')")
    assert drv.execute_script("return !document.querySelector('.menu')")
    drv.execute_script("window.__mace.palette.close()")


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_kind_chips_and_axis_labels_meet_aa_contrast(fresh, theme):
    drv = fresh[0]
    drv.execute_script("window.__mace.settings.set('theme', arguments[0])", theme)
    time.sleep(0.3)
    worst = drv.execute_script("""
      const rgb = (s) => s.match(/[\\d.]+/g).slice(0, 3).map(Number);
      const lum = (c) => { const [r, g, b] = c.map((v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; });
        return 0.2126 * r + 0.7152 * g + 0.0722 * b; };
      const ratio = (a, b) => { const [x, y] = [lum(a), lum(b)].sort((p, q) => q - p); return (x + 0.05) / (y + 0.05); };
      const out = [];
      for (const el of document.querySelectorAll('#file-list .kb')) {
        const cs = getComputedStyle(el);
        out.push([el.textContent, ratio(rgb(cs.color), rgb(cs.backgroundColor))]);
      }
      const root = getComputedStyle(document.documentElement), hex = (h) => [1, 3, 5].map((i) => parseInt(h.trim().slice(i, i + 2), 16));
      for (const k of ['a', 'b', 'c']) out.push(['axis-' + k, ratio(hex(root.getPropertyValue('--axis-' + k)), hex(root.getPropertyValue('--viewport-bg')))]);
      return out.sort((p, q) => p[1] - q[1]).slice(0, 3)""")
    assert worst[0][1] >= 4.5, worst


def test_each_structure_document_keeps_its_camera(fresh):
    """Back on a structure after a trajectory, the view is that structure's:
    no stale fit from the trajectory (it zoomed MgO out to half its size)."""
    drv = fresh[0]
    view = "return window.__mace.viewState().view"
    open_doc(drv, "structures/MgO.cif")
    time.sleep(0.5)
    mgo = drv.execute_script(view)
    open_doc(drv, "outputs/Ag2Br3_opt_unfinished.out")
    time.sleep(0.5)
    drv.execute_script("window.__mace.activate(0)")
    time.sleep(0.6)
    assert max(abs(a - b) for a, b in zip(mgo, drv.execute_script(view))) < 1e-6
    drv.execute_script("window.__mace.run('view.reset')")   # a changed camera is kept too
    time.sleep(0.3)
    turned = drv.execute_script(view)
    open_doc(drv, "structures/NaCl.cif")
    time.sleep(0.5)
    assert abs(drv.execute_script(view)[0] - turned[0]) > 0.5      # NaCl is centered on itself
    drv.execute_script("window.__mace.activate(0)")
    time.sleep(0.6)
    assert max(abs(a - b) for a, b in zip(turned, drv.execute_script(view))) < 1e-6


def test_viewers_paint_once_per_frame_at_device_resolution(fresh):
    """Spec §8.2: render on demand, one paint per frame, no 2x supersampling
    on top of the device pixel ratio (it made a switch cost several 4x paints)."""
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    open_doc(drv, "outputs/Ag2Br3_opt_unfinished.out")
    time.sleep(0.5)
    drv.execute_script("window.__paints = 0; document.addEventListener('mace-painted', () => window.__paints++)")
    for i in range(6):
        drv.execute_async_script("const d = arguments[1]; window.__mace.activate(arguments[0])"
                                 ".then(() => requestAnimationFrame(() => requestAnimationFrame(() => d())))", i % 2)
    assert drv.execute_script("return window.__paints") <= 2 * 6
    sizes = drv.execute_script("""return [...document.querySelectorAll('canvas[id$="-gl"]')]
        .filter((c) => c.clientWidth > 0).map((c) => [c.width, Math.round(c.clientWidth * devicePixelRatio)])""")
    assert sizes and all(abs(w - css) <= 1 for w, css in sizes), sizes
    burst = drv.execute_async_script("""const d = arguments[0], v = window.MACE.state.viewers['viewer-before'];
        window.__paints = 0; for (let k = 0; k < 10; k++) v.render();
        requestAnimationFrame(() => requestAnimationFrame(() => d(window.__paints)))""")
    assert burst == 1


@pytest.mark.parametrize("size", [(1024, 640), (1099, 700), (1100, 700), (1280, 800)])
def test_toolbar_fits_down_to_the_minimum_window(fresh, size):
    """Spec §1.1: 1024x640 is the smallest supported window; nothing in the
    toolbar may be pushed out or squeezed (Setup was cut off at 1024px)."""
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    H.set_viewport(drv, *size)
    time.sleep(0.4)
    got = drv.execute_script("""const t = document.querySelector('#toolbar'), r = t.getBoundingClientRect();
      return {over: t.scrollWidth - t.clientWidth,
        cut: [...t.querySelectorAll('button')].filter((b) => b.getBoundingClientRect().right > r.right + 0.5
          || b.scrollWidth > b.clientWidth + 1).map((b) => b.dataset.cmd || b.id || b.textContent),
        page: document.documentElement.scrollWidth - innerWidth}""")
    assert got == {"over": 0, "cut": [], "page": 0}, got


def test_inspector_is_a_closed_drawer_below_1100px(fresh):
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    shown = "const e = document.querySelector('#inspector'); return [getComputedStyle(e).display !== 'none', getComputedStyle(e).position]"
    H.set_viewport(drv, 1024, 640)
    time.sleep(0.4)
    assert drv.execute_script(shown) == [False, "static"]       # out of the way by default
    drv.execute_script("window.__mace.run('view.inspector')")
    time.sleep(0.2)
    assert drv.execute_script(shown) == [True, "fixed"]         # over the viewport
    drv.find_element("css selector", ".bottom-tabs").click()      # a click in the center shuts it
    time.sleep(0.2)
    assert drv.execute_script(shown)[0] is False
    H.set_viewport(drv, 1280, 800)
    time.sleep(0.4)
    assert drv.execute_script(shown) == [True, "static"]        # docked again, as saved


# --- Phase 2: keymap, reserved keys, mnemonics, browser keys ---------------------------------

# Pairs that share a chord but never in the same context (printed by the test)
CONFLICT_ALLOWLIST = set()


def test_keymap_overrides_and_reserved(fresh):
    drv = fresh[0]
    km = "window.__mace.keymap"
    assert drv.execute_script(f"return {km}.keysOf('edit.duplicate')") == ["Mod+D"]
    rep = drv.execute_script(f"""return {km}.apply([{{command: 'view.alongA', keys: ['7']}},
        {{command: 'edit.redo', keys: ['Mod+Y'], remove: true}}, {{command: 'help.about', keys: ['Mod+W']}},
        {{command: 'no.such', keys: ['X']}}, {{command: 'view.reset', remove: true}}])""")
    assert drv.execute_script(f"return {km}.keysOf('view.alongA')") == ["7"]
    assert drv.execute_script(f"return {km}.keysOf('edit.redo')") == ["Mod+Shift+Z"]       # only Mod+Y removed
    assert drv.execute_script(f"return {km}.keysOf('view.reset')") == []                   # unbound
    assert drv.execute_script(f"return {km}.keysOf('help.about')") == []                   # Mod+W refused
    reasons = {(s["command"], s["reason"]) for s in rep["skipped"]}
    assert any(c == "help.about" and "Ctrl+W closes the browser tab" in r for c, r in reasons), reasons
    assert any(c == "no.such" for c, _ in reasons)
    assert drv.execute_script(f"return {km}.conflicts('7')") == ["view.alongA"]
    drv.execute_script(f"{km}.apply([])")                                                  # reset
    assert drv.execute_script(f"return {km}.keysOf('view.alongA')") == ["1"]
    assert drv.execute_script(f"return {km}.keysOf('view.reset')") == ["Home"]
    assert drv.execute_script(f"return {km}.conflicts('Mod+D')") == ["edit.duplicate"]
    assert drv.execute_script(f"return {km}.conflicts('Ctrl+D')") == ["edit.duplicate"]     # Ctrl and Mod are one key
    assert drv.execute_script(f"return {km}.conflicts('Alt+F')") == ["menu.open.file"]
    # Alt+E is Firefox's Edit menu (measured with real keys, plan T16): reserved, and no Edit mnemonic
    assert drv.execute_script(f"return {km}.isReserved('Alt+E')") and drv.execute_script(f"return {km}.conflicts('Alt+E')") == []
    for chord in ("Mod+W", "Ctrl+W", "Mod+Shift+T", "Mod+Tab", "F5", "Ctrl+1", "Mod+9", "F12"):
        assert drv.execute_script(f"return {km}.isReserved(arguments[0])", chord), chord
    assert not drv.execute_script(f"return {km}.isReserved('Mod+Alt+1')")
    assert drv.execute_script(f"return {km}.riskOf('Ctrl+Shift+I')")
    # the default keymap is conflict free, mnemonics included, apart from context-disjoint pairs
    pairs = drv.execute_script(f"return {km}.allConflicts()")
    print("conflict allowlist:", sorted(CONFLICT_ALLOWLIST))
    assert {tuple(p) for p in pairs} <= CONFLICT_ALLOWLIST, pairs
    # the disjoint pairs the spec relies on are seen as disjoint
    assert not drv.execute_script(f"return {km}.whensOverlap('hasSelection', '!hasSelection')")
    assert not drv.execute_script(f"return {km}.whensOverlap('docKind == structure', 'docKind == plot|db')")
    assert drv.execute_script(f"return {km}.whensOverlap('docKind == structure|trajectory', 'hasDoc')")


def test_menu_mnemonics_conflict_free(fresh):
    """Every top menu's Alt+first-letter is its working mnemonic and nothing
    else's, or on this allowlist (decision D5: Edit and Build give theirs to
    Select by Element and Select Bonded Neighbors; F10 reaches them)."""
    drv = fresh[0]
    allow = {"Edit": "Alt+E is Firefox's Edit menu (plan T16)", "Build": "Alt+B is Firefox's Bookmarks menu (plan T16)"}
    print("mnemonic allowlist:", allow)
    km = "window.__mace.keymap"
    mn = {m["id"]: m for m in drv.execute_script(f"return {km}.pseudo()")}
    for label in MENU_ORDER:
        chord = f"Alt+{label[0]}"
        mid = f"menu.open.{label.lower()}"
        others = drv.execute_script(f"return {km}.conflicts(arguments[0], null, arguments[1])", chord, mid)
        if label in allow:
            assert mn[mid]["keys"] == [], label
        else:
            assert mn[mid]["keys"] == [chord] and others == [], (label, others)
    for letter, label in (("f", "File"), ("v", "View")):
        body_focus(drv)
        keys(drv, Keys.ALT, letter)
        assert drv.execute_script("return document.querySelector('.menu').getAttribute('aria-label')") == label
        keys(drv, Keys.ESCAPE)
    body_focus(drv)
    keys(drv, Keys.F10)
    assert drv.execute_script("return document.activeElement.classList.contains('menu-top') && document.activeElement.textContent") == "File"
    assert not drv.execute_script("return document.querySelectorAll('.menu').length")
    keys(drv, Keys.ARROW_DOWN)
    assert drv.execute_script("return document.querySelector('.menu').getAttribute('aria-label')") == "File"
    keys(drv, Keys.ESCAPE)
    # a mnemonic can be moved like any binding
    drv.execute_script(f"{km}.apply([{{command: 'menu.open.edit', keys: ['Alt+Shift+E']}}])")
    body_focus(drv)
    keys(drv, Keys.ALT, Keys.SHIFT, "e")
    assert drv.execute_script("return document.querySelector('.menu') && document.querySelector('.menu').getAttribute('aria-label')") == "Edit"
    keys(drv, Keys.ESCAPE)
    drv.execute_script(f"{km}.apply([])")


def _keydown(drv, **ev):
    return drv.execute_script("""const t = document.activeElement || document.body;
        const e = new KeyboardEvent('keydown', Object.assign({cancelable: true, bubbles: true}, arguments[0]));
        t.dispatchEvent(e); return e.defaultPrevented""", ev)


def test_browser_nav_keys_never_leave_the_page(fresh):
    drv, base = fresh[0], fresh[1]
    drv.get("about:blank")                     # so Back has somewhere to go
    H.load(drv, base)
    open_doc(drv, "structures/MgO.cif")
    time.sleep(0.4)
    href = drv.execute_script("return location.href")
    drv.execute_script("document.querySelector('#toolbar [data-cmd=\"file.open\"]').focus()")
    toasts = "return [...document.querySelectorAll('#toasts .msg')].map((m) => m.textContent)"
    q0 = drv.execute_script("return window.__mace.viewState().view.slice(4)")
    assert _keydown(drv, key="ArrowLeft", code="ArrowLeft", altKey=True) is True
    time.sleep(0.4)          # Rotate 90° Left (Phase 3) turned the view instead of going Back
    assert drv.execute_script("return window.__mace.viewState().view.slice(4)") != q0
    assert _keydown(drv, key="ArrowRight", code="ArrowRight", altKey=True) is True
    assert _keydown(drv, key="2", code="Digit2", altKey=True) is True                     # Space-Filling, viewport not focused
    assert any("Space-Filling" in t and "viewport" in t for t in drv.execute_script(toasts)), drv.execute_script(toasts)
    assert drv.execute_script("return window.__mace.viewState().style") == "ballstick"
    assert _keydown(drv, key="Backspace", code="Backspace") is True                       # Delete, nothing selected
    # an unbound chord stays the browser's
    assert _keydown(drv, key="Home", code="Home", altKey=True) is False
    # in a text field Backspace edits text, but Alt+Left is still kept from the browser
    drv.execute_script("document.querySelector('#file-filter').focus()")
    assert _keydown(drv, key="Backspace", code="Backspace") is False
    assert _keydown(drv, key="ArrowLeft", code="ArrowLeft", altKey=True) is True
    # and inside the palette, which stops the keys it handles
    drv.execute_script("window.__mace.palette.open('')")
    assert _keydown(drv, key="ArrowLeft", code="ArrowLeft", altKey=True) is True
    assert _keydown(drv, key="Backspace", code="Backspace") is False
    drv.execute_script("window.__mace.palette.close()")
    assert drv.execute_script("return location.href") == href
    # and a real key press
    b = drv.find_element("css selector", '#toolbar [data-cmd="file.open"]')
    drv.execute_script("arguments[0].focus()", b)
    keys(drv, Keys.ALT, Keys.ARROW_LEFT)
    time.sleep(0.5)
    assert drv.execute_script("return location.href") == href
    assert drv.execute_script("return document.documentElement.dataset.appReady") == "1"
    assert not H.errors(drv), H.errors(drv)


def test_beforeunload_guard(fresh):
    drv = fresh[0]
    fire = "return !document.defaultView.dispatchEvent(new Event('beforeunload', {cancelable: true}))"
    open_doc(drv, "structures/MgO.cif")
    assert drv.execute_script(fire) is False                  # nothing modified: never asks
    H.wait(drv, "return window.__mace.doc() && window.__mace.doc().kind === 'structure' && window.MaceEditor.atoms() === 8")
    drv.execute_script("window.__mace.run('build.symModeP1')")
    drv.execute_script("""document.querySelector('#e-element').value = 'Li';
        ['#e-x', '#e-y', '#e-z'].forEach((id) => document.querySelector(id).value = '0.25');
        document.querySelector('#e-add').click()""")
    H.wait(drv, "return window.__mace.doc().dirty")
    assert drv.execute_script(fire) is True                   # modified: the browser asks
    drv.execute_script("window.MaceEditor.undo()")
    H.wait(drv, "return !window.__mace.doc().dirty")
    assert drv.execute_script(fire) is False


# The spec's default keymap (§4.2, §1.1, §1.3), with the documented deviations.
SPEC_KEYMAP = {
    "view.palette": ["F1", "Mod+Alt+K"], "file.quickOpen": ["Mod+P"], "file.goToDoc": ["Mod+E"], "nav.goToAtom": ["Mod+G"],
    "help.shortcuts": ["?", "Mod+/"], "file.settings": ["Mod+,"], "select.exitTool": ["Esc"],
    "file.new": ["Alt+N"], "file.open": ["Mod+O"], "file.save": ["Mod+S"], "file.saveAs": ["Mod+Shift+S"],
    "file.saveAll": ["Mod+Alt+S"], "file.exportImage": ["Mod+Shift+X"], "file.copyImage": ["Mod+Shift+C"],
    "file.importClipboard": ["Mod+Shift+V"], "file.closeTab": ["Alt+W"], "file.nextTab": ["Alt+PageDown"],
    "file.prevTab": ["Alt+PageUp"],
    "edit.undo": ["Mod+Z"], "edit.redo": ["Mod+Shift+Z", "Mod+Y"], "edit.cut": ["Mod+X"], "edit.copy": ["Mod+C"],
    "edit.paste": ["Mod+V"], "edit.duplicate": ["Mod+D"], "edit.delete": ["Delete", "Backspace"],
    "edit.replaceElement": ["E"], "tool.replace": ["E"], "edit.setCoordinates": ["Mod+Shift+T"], "edit.editCell": ["Mod+Shift+L"],
    "edit.history": ["Mod+Shift+H"],
    "select.all": ["Mod+A"], "select.none": ["Mod+Alt+A"], "select.invert": ["Mod+I"], "select.byElement": ["Mod+Alt+E"],
    "select.equivalents": ["Alt+Q"], "select.within": ["Alt+R"], "select.neighbors": ["Mod+Alt+N"], "select.byIndex": ["Alt+I"],
    "select.grow": ["Mod+]"], "select.shrink": ["Mod+["],
    "tool.select": ["V"], "tool.lasso": ["L"], "tool.orbit": ["O"], "tool.pan": ["H"], "tool.zoom": ["Z"],
    "tool.move": ["G"], "tool.add": ["A"], "tool.measure": ["M"], "tool.bond": ["B"],
    "tool.measureSelection": ["Mod+M"], "tool.clearMeasurements": ["Mod+Alt+M"],
    "edit.nudgeLeft": ["Alt+Left"], "edit.nudgeRight": ["Alt+Right"], "edit.nudgeUp": ["Alt+Up"], "edit.nudgeDown": ["Alt+Down"],
    "edit.nudgeOut": ["Alt+PageUp"], "edit.nudgeIn": ["Alt+PageDown"],
    "view.alongA": ["1"], "view.alongB": ["2"], "view.alongC": ["3"], "view.alongAStar": ["Shift+1"],
    "view.alongBStar": ["Shift+2"], "view.alongCStar": ["Shift+3"], "view.alongUVW": ["Alt+U"],
    "view.rotateLeft": ["Alt+Left"], "view.rotateRight": ["Alt+Right"], "view.rotateUp": ["Alt+Up"], "view.rotateDown": ["Alt+Down"],
    "view.rotateInPlaneMinus": ["Alt+["], "view.rotateInPlanePlus": ["Alt+]"],
    "view.rotate15Left": ["Left"], "view.rotate15Right": ["Right"], "view.rotate15Up": ["Up"], "view.rotate15Down": ["Down"],
    "view.reset": ["Home"], "view.frameSelection": ["F"], "view.projection": ["5"], "view.zoomIn3D": ["+"], "view.zoomOut3D": ["-"],
    "view.cell": ["Shift+C"], "view.boundary": ["Shift+A"], "view.bonds": ["Shift+B"], "view.labelsCycle": ["Shift+L"],
    "view.axes": ["Shift+X"], "view.measurements": ["Shift+M"], "view.polyhedra": ["Shift+P"], "view.rangeCycle": ["Shift+R"],
    "view.style.ballstick": ["Alt+1"], "view.style.spacefill": ["Alt+2"], "view.style.stick": ["Alt+3"], "view.style.wireframe": ["Alt+4"],
    "view.explorer": ["Mod+B"], "view.inspector": ["Mod+Alt+B"], "view.bottom": ["Mod+J"], "view.zen": ["Mod+Alt+Z"],
    "view.focusExplorer": ["Mod+Shift+E"], "view.focusViewport": ["Mod+1"], "view.focusInspector": ["Mod+Shift+I"],
    "view.database": ["Mod+Shift+D"], "view.tasks": ["Mod+Shift+Y"], "view.themeCycle": ["Mod+Alt+T"],
    "view.zoomIn": ["Mod+="], "view.zoomOut": ["Mod+-"], "view.zoomReset": ["Mod+0"], "view.inspectorDrawer": ["Mod+Alt+I"],
    "build.supercell": ["Mod+Alt+P"], "build.slab": ["Mod+Alt+H"], "build.clean": ["Mod+Alt+G"], "build.symMode": ["Mod+Alt+Y"],
    "calc.setup": ["Mod+Shift+R"], "calc.preview": ["Mod+Enter"], "calc.saveJson": ["Mod+Alt+J"], "calc.copyCli": ["Mod+Alt+C"],
    "traj.prev": [","], "traj.next": ["."], "traj.first": ["Shift+,"], "traj.last": ["Shift+."], "traj.play": ["Space"],
    "traj.overlay": ["Mod+Alt+O"],
}
# Where MACE differs from the spec's table, and why (each also noted in the code).
KEYMAP_DEVIATIONS = {
    "edit.setCoordinates": (["Mod+Shift+U"], "Mod+Shift+T reopens a closed browser tab (reserved, spec §4.1)"),
    "view.focusViewport": (["Mod+Alt+1"], "Mod+1 is Ctrl+1, browser tab 1 (reserved); the spec's own fallback"),
}
# Shift as 10x (nudge) or 1 degree (15° turns) is spelled out as a second chord
SHIFT_VARIANTS = {"edit.nudge", "view.rotate15"}


def test_default_keymap_matches_spec(fresh):
    drv = fresh[0]
    cmds = {c["id"]: c for c in drv.execute_script("return window.__mace.commands")}
    print("keymap deviations:", {k: v[0] for k, v in KEYMAP_DEVIATIONS.items()})
    for cid, want in SPEC_KEYMAP.items():
        assert cid in cmds, cid
        have = cmds[cid]["keys"]
        if cid in KEYMAP_DEVIATIONS:
            assert have == KEYMAP_DEVIATIONS[cid][0], (cid, have)
            continue
        if any(cid.startswith(p) for p in SHIFT_VARIANTS):
            assert have[0] == want[0] and all("Shift+" in k for k in have[1:]), (cid, have)
            continue
        assert have == want, (cid, have, want)
    menu_ids = set()
    def walk(items):
        for it in items:
            if isinstance(it, str) and it != "-":
                menu_ids.add(it)
    for m in drv.execute_script("return window.__mace.menus()"):
        walk(m["items"])
    assert len(cmds) >= len(menu_ids | set(SPEC_KEYMAP)), (len(cmds), len(menu_ids | set(SPEC_KEYMAP)))


def test_view_keys_turn_zoom_and_center(fresh):
    """The view-only keys that work now: arrows turn 15° (Shift 1°), + and -
    zoom, both only with the viewport focused."""
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    time.sleep(0.5)
    view = "return window.__mace.viewState().view"
    drv.execute_script("document.querySelector('#viewer-main').closest('.viewer-wrap').focus()")
    v0 = drv.execute_script(view)
    keys(drv, Keys.ARROW_LEFT)
    v1 = drv.execute_script(view)
    assert v1[4:] != v0[4:]                        # the rotation changed
    keys(drv, Keys.SHIFT, Keys.ARROW_LEFT)
    v2 = drv.execute_script(view)
    # Shift turns 1°: a smaller change of the quaternion than the 15° step
    d15 = sum((a - b) ** 2 for a, b in zip(v0[4:], v1[4:])) ** 0.5
    d1 = sum((a - b) ** 2 for a, b in zip(v1[4:], v2[4:])) ** 0.5
    assert 0 < d1 < d15 / 5, (d1, d15)
    z0 = drv.execute_script(view)[3]
    keys(drv, "+")
    time.sleep(0.4)
    assert drv.execute_script(view)[3] != z0
    # not in a field
    drv.find_element("id", "file-filter").click()
    v3 = drv.execute_script(view)
    keys(drv, Keys.ARROW_LEFT)
    assert drv.execute_script(view) == v3
    assert not H.errors(drv), H.errors(drv)


def test_trajectory_step_keys(fresh):
    drv = fresh[0]
    assert open_doc(drv, "outputs/Ag2Br3_opt_unfinished.out") == "trajectory"
    time.sleep(0.4)
    n = drv.execute_script("return window.__mace.ctx().trajSteps")
    assert n >= 3
    frame = "return Number(document.querySelector('#o-frame').value)"
    assert drv.execute_script(frame) == n - 1          # opens on the last step
    body_focus(drv)
    keys(drv, ",")
    H.wait(drv, f"return Number(document.querySelector('#o-frame').value) === {n - 2}")
    keys(drv, Keys.SHIFT, ",")
    H.wait(drv, "return Number(document.querySelector('#o-frame').value) === 0")
    keys(drv, ".")
    H.wait(drv, "return Number(document.querySelector('#o-frame').value) === 1")
    keys(drv, Keys.SHIFT, ".")
    H.wait(drv, f"return Number(document.querySelector('#o-frame').value) === {n - 1}")
    for look in ("twotone", "side", "dmap", "overlay"):  # Mod+Alt+O cycles the four looks (plan B7)
        keys(drv, Keys.CONTROL, Keys.ALT, "o")
        assert drv.execute_script("return document.querySelector('.seg-btn.active').dataset.layout") == look
    # a structure has no steps: the key says so, the slider does not move
    open_doc(drv, "structures/MgO.cif")
    assert drv.execute_script("return window.__mace.ctx().trajSteps") == 0
    assert not H.errors(drv), H.errors(drv)


# --- Phase 2: the palette --------------------------------------------------------------------

def _first_row(drv):
    return drv.execute_script("""const r = document.querySelector('.pal-row[data-k="0"]');
        return r && [r.querySelector('.t').textContent, (r.querySelector('kbd') || {}).textContent || '']""")


@pytest.mark.parametrize("start", ["body", "viewport", "toolbar"])
def test_palette_sup_first_with_shortcut(fresh, start):
    """Criterion 1: Ctrl+Alt+K and F1 open it (Firefox keeps Ctrl+K, plan T16); "sup" shows Build: Supercell… first
    with its shortcut; Esc closes it and focus goes back."""
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    time.sleep(0.3)
    focus = {"body": "document.activeElement.blur(); document.body.focus()",
             "viewport": "document.querySelector('#viewer-main').closest('.viewer-wrap').focus()",
             "toolbar": "document.querySelector('#toolbar [data-cmd=\"view.cell\"]').focus()"}[start]
    for opener in ((Keys.CONTROL, Keys.ALT, "k"), (Keys.F1,)):
        drv.execute_script(focus)
        before = drv.execute_script("return document.activeElement")
        keys(drv, *opener)
        H.wait(drv, "return document.querySelector('#palette').open")
        box = drv.find_element("css selector", "#palette input")
        box.send_keys("sup")
        assert _first_row(drv) == ["Build: Supercell…", "Ctrl+Alt+P"]
        if start == "body" and opener[0] == Keys.CONTROL:
            for theme in ("light", "dark"):
                drv.execute_script("window.__mace.settings.set('theme', arguments[0])", theme)
                time.sleep(0.2)
                shot(drv, f"palette_sup_{theme}")
            drv.execute_script("window.__mace.settings.set('theme', 'light')")
        keys(drv, Keys.ESCAPE)
        assert not drv.execute_script("return document.querySelector('#palette').open")
        assert drv.execute_script("return document.activeElement") == before


def test_palette_lists_every_command(fresh):
    """Criterion 2: the palette count equals the registry size."""
    drv = fresh[0]
    ids = drv.execute_script("return window.__mace.palette.results('').filter((r) => r.id).map((r) => r.id)")
    # (a recent file's own menu entries, which act on that file, are not palette commands)
    reg = drv.execute_script("return window.__mace.commands.filter((c) => c.palette).map((c) => c.id)")
    assert len(ids) == len(reg) == len(set(ids))
    assert set(ids) == set(reg)
    # every one is also drawn when scrolled to: the list is virtual, not short
    drv.execute_script("window.__mace.palette.open('')")
    n = drv.execute_script("return document.querySelector('#pal-list').style.height")
    groups = drv.execute_script("return new Set(window.__mace.palette.results('').map((r) => r.group)).size")
    assert int(n.rstrip("px")) == 32 * (len(reg) + groups)
    assert drv.execute_script("return window.__mace.palette.domRows()") <= 40
    drv.execute_script("const s = document.querySelector('.pal-scroll'); s.scrollTop = s.scrollHeight")
    time.sleep(0.2)
    last = drv.execute_script("return [...document.querySelectorAll('.pal-row')].pop().dataset.k")
    assert int(last) == len(reg) - 1
    drv.execute_script("window.__mace.palette.close()")


def test_palette_modes(fresh):
    drv = fresh[0]
    R = "return window.__mace.palette.results(arguments[0])"
    assert drv.execute_script(R, "gamma")[0]["id"] == "edit.editCell"
    assert drv.execute_script(R, ":")[0]["why"]                         # no trajectory
    open_doc(drv, "outputs/Ag2Br3_opt_unfinished.out")
    open_doc(drv, "structures/MgO.cif")
    files = drv.execute_script(R, "#")
    assert len(files) >= 10 and all(r["group"] == "Files" for r in files)
    docs = drv.execute_script(R, "%")
    assert sorted(r["label"] for r in docs) == ["Ag2Br3_opt_unfinished.out", "MgO.cif"]
    atoms = drv.execute_script(R, "@O5")
    assert atoms[0]["label"] == "O5" and "Jump to atom" in atoms[0]["desc"]
    calc = drv.execute_script(R, "= 0.5 0 0.5")
    assert calc[0]["label"] == "2.1060 0.0000 2.1060 Å", calc
    back = drv.execute_script(R, "= c 2.106 0 2.106")
    assert back[0]["label"].startswith("0.50000 0.00000 0.50000"), back
    assert drv.execute_script(R, ":3")[0]["why"]                        # MgO is not a trajectory
    # @ jumps: the view centers on the atom, the readout names it
    drv.execute_script("window.__mace.palette.open('@O5')")
    drv.find_element("css selector", "#palette input").send_keys(Keys.ENTER)
    H.wait(drv, "return document.querySelector('#st-cursor').textContent.startsWith('O5')")
    # Mod+Enter runs and keeps the palette open
    drv.execute_script("window.__mace.palette.open('show bonds')")
    bonds = drv.execute_script("return window.__mace.viewState().bonds")
    ActionChains(drv).key_down(Keys.CONTROL).send_keys(Keys.ENTER).key_up(Keys.CONTROL).perform()
    time.sleep(0.2)
    assert drv.execute_script("return document.querySelector('#palette').open")
    assert drv.execute_script("return window.__mace.viewState().bonds") is (not bonds)
    keys(drv, Keys.ESCAPE)
    # the trajectory's steps
    drv.execute_script("window.__mace.activate(0)")
    time.sleep(0.4)
    assert drv.execute_script(R, ":1")[0]["label"] == "Go to step 1"
    drv.execute_script("window.__mace.palette.open(':1')")
    drv.find_element("css selector", "#palette input").send_keys(Keys.ENTER)
    H.wait(drv, "return document.querySelector('#o-frame').value === '1'")
    assert not H.errors(drv), H.errors(drv)


def test_palette_is_virtual_and_fast(fresh):
    """Spec §8.1: ≤8 ms per keystroke on 500 commands + 2000 atoms (rank and
    DOM). Checked here with a ×3 margin for the software-GL test box; the
    bench enforces the budget itself."""
    drv = fresh[0]
    times = drv.execute_script("return window.__mace.palette.bench(500, 2000)")
    assert all(t["dom"] <= 40 for t in times), times
    assert any(t["rows"] >= 200 for t in times)
    worst = max(t["ms"] for t in times)
    print("palette filter ms per keystroke:", [(t["q"], round(t["ms"], 2)) for t in times])
    assert worst <= 24, times
    assert drv.execute_script("return window.__mace.palette.rankBench(2500)") <= 24
    drv.execute_script("window.__mace.settings.set('devShowCommandIds', true)")
    drv.execute_script("window.__mace.palette.open('sup')")
    assert drv.execute_script("return document.querySelector('.pal-row .pal-id').textContent") == "build.supercell"
    drv.execute_script("window.__mace.palette.close(); window.__mace.settings.set('devShowCommandIds', false)")


# --- Phase 2: settings drive the viewport ------------------------------------------------------

COLORS = json.loads((H.REPO_ROOT / "mace" / "gui" / "static" / "element_colors.json").read_text())
# T0 measured a render-then-read pixel at Mg1 stable over 5 of 5 fresh pages, so the
# pixel checks below are asserted, as a second proof next to the model's own style
PIXELS_STABLE = True


def mg1_pixel(drv):
    """The rendered color at Mg1's center and the drawn disk's radius in px,
    measured outward from the structure's middle (away from other atoms)."""
    return drv.execute_async_script("""const done = arguments[0];
      const v = window.MACE.state.viewers['viewer-main'], s = window.MACE.state.structure;
      const at = window.__mace.atomScreen(0), cv = v.getCanvas(), r = cv.getBoundingClientRect();
      const mid = s.cell.reduce((t, c) => [t[0] + c[0] / 2, t[1] + c[1] / 2, t[2] + c[2] / 2], [0, 0, 0]);
      const m = v.modelToScreen({x: mid[0], y: mid[1], z: mid[2]});
      let dx = at.x - (m.x - r.left), dy = at.y - (m.y - r.top); const n = Math.hypot(dx, dy) || 1; dx /= n; dy /= n;
      v.render();
      const url = cv.toDataURL('image/png'), k = cv.width / cv.clientWidth;
      const img = new Image();
      img.onload = () => {
        const c = document.createElement('canvas'); c.width = cv.width; c.height = cv.height;
        const g = c.getContext('2d'); g.drawImage(img, 0, 0);
        const px = (x, y) => Array.from(g.getImageData(Math.round(x * k), Math.round(y * k), 1, 1).data);
        const bg = px(3, 3), center = px(at.x, at.y);
        let rad = 0;
        for (let d = 1; d < 200; d++) {
          const p = px(at.x + dx * d, at.y + dy * d);
          if (Math.max(...[0, 1, 2].map((j) => Math.abs(p[j] - bg[j]))) <= 12) { rad = d; break; }
        }
        done({center, bg, rad});
      };
      img.src = url;""")
D1_DEFAULT_SCHEME = "vesta"     # decision D1: VESTA stays the default (the spec's §3 table says Jmol)


def old_tilt(drv):
    """The camera the Phase 1-3 default view had (a fixed turn from no
    rotation), where Mg1 is in front: Phase 4's standard view puts the origin
    corner at the back, behind a bond (plan R9: the camera is set explicitly)."""
    drv.execute_script("""const v = window.MACE.state.viewers['viewer-main'], sc = window.__mace.viewport.scene();
        const disp = sc.display, s = window.__mace.doc().structure, pts = [];
        for (let k = 0; k < disp.n; k++) pts.push([disp.cart[3 * k], disp.cart[3 * k + 1], disp.cart[3 * k + 2]]);
        for (const i of [0, 1]) for (const j of [0, 1]) for (const k of [0, 1])
          pts.push([0, 1, 2].map((m) => i * s.cell[0][m] + j * s.cell[1][m] + k * s.cell[2][m]));
        // 3Dmol's setView takes only an Array of the page's own (instanceof Array): built with window.Array
        const view = v.getView();
        v.setView(window.Array.from([...view.slice(0, 4), 0, 0, 0, 1]));
        window.MACE.fitView(v, s, [], 1, pts, {tilt: true});      // the Phase 3 tilt (the fit is tight since Phase 4)
        v.render();""")
    time.sleep(0.3)


def test_settings_drive_the_viewport(fresh):
    drv = fresh[0]
    assert drv.execute_script("return window.__mace.settings.defaults().colorScheme") == D1_DEFAULT_SCHEME
    open_doc(drv, "structures/MgO.cif")
    time.sleep(0.6)
    old_tilt(drv)
    st = "return window.__mace.atomStyle(0)"
    a = drv.execute_script(st)
    assert a["color"].lower() == COLORS["schemes"]["vesta"]["Mg"].lower(), a
    px_vesta = mg1_pixel(drv)
    # atomic radii x atom scale 0.40: VESTA's ball and stick, 40 % of the radius (T9)
    assert abs(a["scale"] - 0.4) < 1e-9
    assert drv.execute_script("return window.__mace.viewState().sphereScale") == 0.4
    drv.execute_script("window.__mace.settings.set('colorScheme', 'jmol')")
    H.wait(drv, "return window.__mace.atomStyle(0).color.toLowerCase() === arguments[0]", 5,
           COLORS["schemes"]["jmol"]["Mg"].lower())
    assert drv.execute_script("return document.querySelector('#opt-colors').value") == "jmol"   # mirrored
    time.sleep(0.3)
    px_jmol = mg1_pixel(drv)
    print("Mg1 pixel VESTA / Jmol:", px_vesta, px_jmol)
    if PIXELS_STABLE:
        assert max(abs(x - y) for x, y in zip(px_vesta["center"][:3], px_jmol["center"][:3])) > 20, (px_vesta, px_jmol)
    drv.execute_script("window.__mace.settings.set('atomScale', 0.6)")
    H.wait(drv, "return Math.abs(window.__mace.atomStyle(0).scale - 0.6) < 1e-9", 5)
    H.wait(drv, "const a = window.__mace.atomStyle(0); return Math.abs(a.radius / arguments[0] - 1.5) < 0.01", 5, a["radius"])
    b = drv.execute_script(st)
    assert a["radius"] and b["radius"] and abs(b["radius"] / a["radius"] - 1.5) < 0.01, (a, b)
    time.sleep(0.3)
    px_big = mg1_pixel(drv)
    print("Mg1 disk radius px at 0.40 / 0.60:", px_jmol["rad"], px_big["rad"])
    if PIXELS_STABLE:
        assert px_jmol["rad"] > 0 and px_big["rad"] >= 1.3 * px_jmol["rad"], (px_jmol, px_big)
    drv.execute_script("window.__mace.settings.set('background', 'black')")
    H.wait(drv, "return window.__mace.viewState().background === getComputedStyle(document.documentElement).getPropertyValue('--viewport-black').trim()", 5)
    # the old checkbox in Inspector › Display writes the setting back
    drv.execute_script("const c = document.querySelector('#opt-colors'); c.value = 'vesta'; c.dispatchEvent(new Event('change'))")
    assert drv.execute_script("return window.__mace.settings.get('colorScheme')") == "vesta"
    # labels: a setting with the checkbox as its mirror
    drv.execute_script("window.__mace.run('view.labelsCycle')")
    assert drv.execute_script("return [window.__mace.settings.get('labels'), document.querySelector('#opt-labels').checked]") == ["element", True]
    # Shift+L cycles all five (Index and Wyckoff came with the editor in Phase 3)
    for want in ["index", "both", "wyckoff", "off"]:
        drv.execute_script("window.__mace.run('view.labelsCycle')")
        assert drv.execute_script("return window.__mace.settings.get('labels')") == want
    assert drv.execute_script("return [window.__mace.settings.get('labels'), document.querySelector('#opt-labels').checked]") == ["off", False]
    # the view toggles that are settings
    drv.execute_script("window.__mace.run('view.cell')")
    assert drv.execute_script("return [window.__mace.settings.get('showCell'), window.__mace.viewState().cell]") == [False, False]
    drv.execute_script("window.__mace.run('view.cell')")
    assert not H.errors(drv), H.errors(drv)


def test_default_look_and_bonds_unchanged(fresh):
    """D2: the spec's bondTolerance 1.15 (the demo drew with 1.2) bonds the
    same pairs in every demo structure; only the stick radius moves 0.11 -> 0.12 Å."""
    drv = fresh[0]
    counts = {}
    for p in ["structures/MgO.cif", "structures/NaCl.cif", "structures/LiF.cif", "structures/Si.cif",
              "structures/diamond.cif", "structures/hBN.cif"]:
        open_doc(drv, p)
        time.sleep(0.3)
        counts[p] = drv.execute_script("""const s = window.MACE.state.structure, at = window.MACE.displayAtoms(s, {supercell: false, boundary: true});
            const n = (sc) => window.MACE.bondsFor(at, s, sc).reduce((t, b) => t + b.length, 0) / 2;
            return [n(1.2), n(0)]""")
    print("bond pairs at 1.2 and at the setting (1.15):", counts)
    assert all(a == b for a, b in counts.values()), counts


def test_old_color_key_migrates(fresh):
    drv, base = fresh[0], fresh[1]
    drv.execute_script("""const s = JSON.parse(localStorage.getItem('mace.gui.settings.v1') || '{}'); delete s.colorScheme;
        localStorage.setItem('mace.gui.settings.v1', JSON.stringify(s)); localStorage.setItem('mace-gui-colors', 'jmol')""")
    drv.refresh()
    H.wait(drv, "return document.documentElement.dataset.appReady === '1'")
    assert drv.execute_script("return window.__mace.settings.get('colorScheme')") == "jmol"
    assert drv.execute_script("return localStorage.getItem('mace-gui-colors')") is None
    assert drv.execute_script("return localStorage.getItem('mace.gui.settings.version')") == "3"   # the schema is at 3 (polish)
    drv.execute_script("window.__mace.settings.resetAll()")
    drv.execute_async_script("const d = arguments[0]; window.__mace.settings.flush().then(d)")


# --- Phase 2: the Settings dialog -------------------------------------------------------------

# Every key of the spec's §3 table (cellColor, defaultDb and the MACE defaults included)
SPEC_SETTINGS = ["theme", "density", "uiScale", "reduceMotion", "colorScheme", "radiusSet", "atomScale", "bondRadius",
                 "bondTolerance", "bondMetalMetal", "bondColoring", "background", "labels", "labelSize", "projection",
                 "quality", "antialias", "showAxes", "showCell", "showBoundary", "cellColor", "coordMode", "lengthUnit",
                 "energyUnit", "decimals.coord", "decimals.length", "decimals.angle", "symmetryModeDefault", "symprec",
                 "defaultElement", "nudgeStepFrac", "nudgeStepCart", "wrapIntoCell", "confirmOverwrite", "keepBackup",
                 "autosaveDrafts", "undoLimit", "explorerExclude", "defaultDb", "recentLimit", "saveFormat", "functional",
                 "basisSet", "exampleConfig", "optionsFileName", "runPreflightOnPreview", "heartbeatSeconds",
                 "devShowCommandIds", "max_workers", "cache_mb", "gpu_device"]


def open_settings(drv, section=None):
    body_focus(drv)
    keys(drv, Keys.CONTROL, ",")
    H.wait(drv, "return document.querySelector('#settings') && document.querySelector('#settings').open")
    if section:
        drv.execute_script("document.querySelector(`#settings [data-sec='${arguments[0]}']`).click()", section)


def stored(xdg):
    p = xdg / "mace" / "gui.json"
    return json.loads(p.read_text())["settings"] if p.exists() else {}


def flush(drv):
    drv.execute_async_script("const d = arguments[0]; window.__mace.settings.flush().then(d)")


def test_settings_schema_covers_defaults(fresh):
    drv = fresh[0]
    keys_ = drv.execute_async_script("""const d = arguments[0];
        import(document.querySelector('script[src*="/js/shell.js"]').src.replace('shell.js', 'lazy/settings_schema.js'))
          .then((m) => d(m.SCHEMA.map((r) => r.key)))""")
    defaults = drv.execute_script("return Object.keys(window.__mace.settings.defaults())")
    # settings with their own UI, not a Settings row: the keymap (Keyboard Shortcuts), the Inspector's
    # Basic | Advanced per view kind (its tab strip) and the activity bar's order and hidden views (its menu)
    # and the bond pair rules (Build › Calculate Bonds…, 1.1.6)
    own_ui = {"keymap", "inspector.tier", "activity.order", "activity.hidden", "activity.visible", "bondPairs"}
    assert sorted(k for k in keys_) == sorted(k for k in defaults if k not in own_ui), set(keys_) ^ set(defaults)
    assert len(keys_) == len(set(keys_))
    assert not set(SPEC_SETTINGS) - set(keys_), set(SPEC_SETTINGS) - set(keys_)


def test_settings_live_and_persistent(fresh):
    """Criterion 4: Ctrl+, opens Settings; theme, color scheme and atom scale
    change the viewport live; they persist after a reload and gui.json holds
    exactly the non-default keys."""
    drv, base, _, xdg = fresh
    drv.execute_script("window.__mace.settings.reset('theme')")      # every setting at its default
    flush(drv)
    assert stored(xdg) == {}
    open_doc(drv, "structures/MgO.cif")
    time.sleep(0.5)
    open_settings(drv)
    # (a) theme
    drv.find_element("css selector", '#settings [data-key="theme"][data-val="dark"]').click()
    assert drv.execute_script("return document.documentElement.dataset.theme") == "dark"
    H.wait(drv, "return window.__mace.viewState().background === getComputedStyle(document.documentElement).getPropertyValue('--viewport-bg').trim()", 5)
    # (b) the color scheme: under D1 VESTA is the default, so the live change goes to Jmol and back
    drv.execute_script("document.querySelector(`#settings [data-sec='viewport']`).click()")
    sel = drv.find_element("css selector", "#set-colorScheme")
    from selenium.webdriver.support.ui import Select
    Select(sel).select_by_value("jmol")
    H.wait(drv, "return window.__mace.atomStyle(0).color.toLowerCase() === arguments[0]", 5, COLORS["schemes"]["jmol"]["Mg"].lower())
    assert drv.execute_script("return window.__mace.viewState().colorScheme") == "jmol"
    # (c) atom scale 0.40 -> 0.60
    r0 = drv.execute_script("return window.__mace.atomStyle(0).radius")
    drv.execute_script("const r = document.querySelector('#set-atomScale'); r.value = '0.6';"
                       "r.dispatchEvent(new Event('input', {bubbles: true})); r.dispatchEvent(new Event('change', {bubbles: true}))")
    H.wait(drv, "return window.__mace.viewState().sphereScale === 0.6", 5)
    # the redraw lands on the next frame: wait for the drawn sphere itself
    H.wait(drv, "const a = window.__mace.atomStyle(0); return Math.abs(a.radius / arguments[0] - 1.5) < 0.01", 5, r0)
    shot(drv, "settings_viewport_dark")
    flush(drv)
    # an edited style value makes the Style preset Custom, as in the Inspector (1.1.6)
    assert stored(xdg) == {"theme": "dark", "colorScheme": "jmol", "atomScale": 0.6, "stylePreset": "custom"}
    Select(drv.find_element("css selector", "#set-colorScheme")).select_by_value(D1_DEFAULT_SCHEME)
    flush(drv)
    assert stored(xdg) == {"theme": "dark", "atomScale": 0.6, "stylePreset": "custom"}     # the default is not stored
    drv.refresh()
    H.wait(drv, "return document.documentElement.dataset.appReady === '1'")
    open_doc(drv, "structures/MgO.cif")
    time.sleep(0.5)
    assert drv.execute_script("return document.documentElement.dataset.theme") == "dark"
    assert drv.execute_script("return window.__mace.atomStyle(0).color.toLowerCase()") == COLORS["schemes"][D1_DEFAULT_SCHEME]["Mg"].lower()
    assert drv.execute_script("return window.__mace.viewState().sphereScale") == 0.6
    # the Reset links
    open_settings(drv, "appearance")
    drv.find_element("css selector", '#settings [data-reset="theme"]').click()
    drv.execute_script("document.querySelector(`#settings [data-sec='viewport']`).click()")
    drv.find_element("css selector", '#settings [data-reset="atomScale"]').click()
    assert drv.execute_script("return document.querySelector('#settings [data-reset=\"atomScale\"]').hidden")
    drv.find_element("css selector", '#settings [data-reset="stylePreset"]').click()
    flush(drv)
    assert stored(xdg) == {}
    drv.execute_script("document.querySelector('#settings').close()")


def test_settings_search_sections_and_errors(fresh):
    drv = fresh[0]
    open_settings(drv)
    assert drv.execute_script("return document.activeElement.id") == "set-search"
    drv.find_element("id", "set-search").send_keys("bond")
    rows = drv.execute_script("return [...document.querySelectorAll('#settings .set-row')].map((r) => r.dataset.row)")
    # "Bonded atoms outside" (polish A5) is found too, and the shortest bond (1.1.6, Calculate Bonds…)
    assert set(rows) == {"bondRadius", "bondTolerance", "bondMin", "bondMetalMetal", "bondColoring", "bondedOutside"}, rows
    counts = drv.execute_script("return [...document.querySelectorAll('.set-nav-item')].map((b) => [b.dataset.sec, b.querySelector('.n').textContent, b.classList.contains('dim')])")
    assert ["viewport", "6", False] in counts and ["editor", "", True] in counts, counts
    shot(drv, "settings_search_bond")
    drv.find_element("id", "set-search").clear()
    drv.find_element("id", "set-search").send_keys("zzz")
    assert "No settings match" in drv.find_element("css selector", "#settings .set-pane").text
    drv.find_element("id", "set-search").clear()
    drv.execute_script("document.querySelector('#set-search').dispatchEvent(new Event('input'))")
    # all 8 sections open; the hint on the editor's rows
    for sec in ("appearance", "viewport", "units", "editor", "keyboard", "files", "mace", "advanced"):
        drv.execute_script("document.querySelector(`#settings [data-sec='${arguments[0]}']`).click()", sec)
        time.sleep(0.15)
        for theme in ("light", "dark"):
            drv.execute_script("window.__mace.settings.set('theme', arguments[0])", theme)
            shot(drv, f"settings_{sec}_{theme}")
        drv.execute_script("window.__mace.settings.set('theme', 'light')")
        assert drv.execute_script("return document.querySelector('#settings .set-h').textContent")
    drv.execute_script("document.querySelector(`#settings [data-sec='editor']`).click()")
    # every editor row is used now (Phase 3); a row for a later phase still says so
    assert "Phase 3" not in drv.find_element("css selector", "#settings .set-pane").text
    drv.execute_script("document.querySelector(`#settings [data-sec='files']`).click()")
    assert "What Database View" in drv.find_element("css selector", "#settings .set-pane").text
    # a number: Enter commits, out of range is clamped with an inline note, not a number is an error
    drv.execute_script("document.querySelector(`#settings [data-sec='viewport']`).click()")
    f = drv.find_element("id", "set-labelSize")
    f.clear(); f.send_keys("40", Keys.ENTER)
    assert drv.execute_script("return window.__mace.settings.get('labelSize')") == 20
    err = drv.execute_script("const f = document.querySelector('#set-labelSize'); return document.getElementById(f.getAttribute('aria-describedby')).textContent")
    assert "9–20" in err and "20" in err
    f = drv.find_element("id", "set-labelSize")
    f.clear(); f.send_keys("abc", Keys.ENTER)
    assert drv.execute_script("return document.querySelector('#set-labelSize').getAttribute('aria-invalid')") == "true"
    assert drv.execute_script("return window.__mace.settings.get('labelSize')") == 20
    shot(drv, "settings_invalid_number")
    drv.execute_script("window.__mace.settings.reset('labelSize')")
    # the CLI's defaults, read from the server
    drv.execute_script("document.querySelector(`#settings [data-sec='mace']`).click()")
    text = drv.find_element("css selector", "#settings .set-pane").text
    assert "CLI default: " in text
    drv.execute_script("document.querySelector('#settings').close()")
    assert not H.errors(drv), H.errors(drv)


def test_settings_fit_the_smallest_window(fresh):
    drv = fresh[0]
    H.set_viewport(drv, 1024, 640)
    time.sleep(0.3)
    open_settings(drv, "viewport")
    got = drv.execute_script("""const d = document.querySelector('#settings').getBoundingClientRect();
        return [d.left >= 0, d.top >= 0, d.right <= innerWidth, d.bottom <= innerHeight,
          document.documentElement.scrollWidth - innerWidth, document.documentElement.scrollHeight - innerHeight]""")
    assert got == [True, True, True, True, 0, 0], got
    shot(drv, "settings_1024x640")
    drv.execute_script("document.querySelector('#settings').close()")


def test_dialogs_load_on_first_open(app):
    """Spec §8.5 / P3: Settings and the cheat sheet are fetched the first time
    they open, once."""
    drv, base = app[0], app[1]
    drv.refresh()
    H.wait(drv, "return document.documentElement.dataset.appReady === '1'")
    lazy = "return performance.getEntriesByType('resource').filter((r) => r.name.includes('/js/lazy/') || r.name.includes('dialogs.css')).map((r) => r.name.split('/').pop())"
    # nothing of either dialog at startup (the Tasks pane's module and its lazy_util.js load at
    # startup when the server already has tasks, e.g. from an earlier test on this server)
    base0 = drv.execute_script(lazy)
    dialogs = {"settings_dialog.js", "settings_schema.js", "dialogs.css", "cheatsheet.js"}
    assert not dialogs & set(base0), base0
    assert set(base0) <= {"lazy_util.js"}, base0
    open_settings(drv)
    first = drv.execute_script(lazy)
    assert {"settings_dialog.js", "settings_schema.js", "dialogs.css"} <= set(first), first
    drv.execute_script("document.querySelector('#settings').close()")
    open_settings(drv)
    assert drv.execute_script(lazy) == first                 # no second request
    drv.execute_script("document.querySelector('#settings').close()")
    # the shortcut sheet: its module on first open, nothing on the second
    assert "cheatsheet.js" not in first
    open_cheatsheet(drv)
    second = drv.execute_script(lazy)
    assert set(second) - set(first) == {"cheatsheet.js"}, second
    drv.execute_script("document.querySelector('#cheatsheet').close()")
    open_cheatsheet(drv)
    assert drv.execute_script(lazy) == second
    drv.execute_script("document.querySelector('#cheatsheet').close()")


# --- Phase 2: the keymap editor (Settings › Keyboard) -------------------------------------------

def open_keymap(drv, filt=""):
    open_settings(drv, "keyboard")
    H.wait(drv, "return !!document.querySelector('#settings .km-list')")
    if filt:
        f = drv.find_element("css selector", "#settings .km-filter")
        f.clear(); f.send_keys(filt)


def record(drv, cid):
    drv.execute_script("document.querySelector(`.km-row[data-id='${arguments[0]}'] .km-keys`).click()", cid)
    H.wait(drv, "return !document.querySelector('.km-rec').hidden && document.activeElement === document.querySelector('.km-rec')")


def rec_key(drv, **ev):
    """A key press in the recorder, synthetic: a real Ctrl+W could close the test window."""
    return drv.execute_script("""const e = new KeyboardEvent('keydown', Object.assign({cancelable: true, bubbles: true}, arguments[0]));
        document.activeElement.dispatchEvent(e); return e.defaultPrevented""", ev)


def rec_msg(drv):
    return drv.execute_script("const m = document.querySelector('.km-msg'); return m ? m.textContent : ''")


def test_reserved_binding_refused(fresh):
    """Criterion 5b: binding Ctrl+W is refused, with the reason."""
    drv = fresh[0]
    open_keymap(drv, "Duplicate")
    record(drv, "edit.duplicate")
    for ev, shown, why in [({"code": "KeyW", "key": "w", "ctrlKey": True}, "Ctrl+W", "closes the browser tab"),
                           ({"code": "Tab", "key": "Tab", "ctrlKey": True}, "Ctrl+Tab", "switches browser tabs"),
                           ({"code": "F5", "key": "F5"}, "F5", "reloads the page"),
                           ({"code": "Digit1", "key": "1", "ctrlKey": True}, "Ctrl+1", "browser tab 1")]:
        assert rec_key(drv, **ev) is True
        msg = rec_msg(drv)
        assert shown in msg and why in msg and "can't be used" in msg, msg
        assert drv.execute_script("return window.__mace.keymap.keysOf('edit.duplicate')") == ["Mod+D"]
        assert drv.execute_script("return !document.querySelector('.km-rec').hidden")     # still waiting
    shot(drv, "keymap_reserved")
    assert drv.execute_script("return window.__mace.settings.get('keymap')") == []
    # a developer-tools key warns, and can be used anyway
    rec_key(drv, code="KeyI", key="I", ctrlKey=True, shiftKey=True)
    assert "developer tools" in rec_msg(drv)
    shot(drv, "keymap_risky_warning")
    drv.execute_script("[...document.querySelectorAll('.km-msg button')].find((b) => b.textContent === 'Use anyway').click()")
    # Ctrl+Shift+I was Focus Inspector's: now a conflict to settle
    assert "is also Focus Inspector" in rec_msg(drv), rec_msg(drv)
    drv.execute_script("[...document.querySelectorAll('.km-msg button')].find((b) => b.textContent === 'Pick another key').click()")
    assert drv.execute_script("return window.__mace.keymap.keysOf('edit.duplicate')") == ["Mod+D"]
    # still recording, the question gone, the panel listening for the next key
    assert drv.execute_script("const r = document.querySelector('.km-rec'); return !r.hidden && document.activeElement === r"
                              " && document.querySelector('.km-msg').textContent === ''")
    # a conflict (Ctrl+D is Duplicate's), replaced
    open_keymap(drv, "Select All")
    record(drv, "select.all")
    rec_key(drv, code="KeyD", key="d", ctrlKey=True)
    assert "Ctrl+D is also Duplicate. Replace it?" in rec_msg(drv)
    shot(drv, "keymap_conflict")
    drv.execute_script("[...document.querySelectorAll('.km-msg button')].find((b) => b.textContent === 'Replace').click()")
    assert drv.execute_script("return window.__mace.keymap.keysOf('select.all')") == ["Mod+D"]
    assert drv.execute_script("return window.__mace.keymap.keysOf('edit.duplicate')") == []
    assert drv.execute_script("return window.__mace.settings.get('keymap')") == [
        {"command": "edit.duplicate", "remove": True}, {"command": "select.all", "keys": ["Mod+D"]}]
    drv.execute_script("document.querySelector('#settings [data-km=\"reset-all\"]').click()")
    assert drv.execute_script("return window.__mace.keymap.keysOf('edit.duplicate')") == ["Mod+D"]
    drv.execute_script("document.querySelector('#settings').close()")


def test_keymap_row_reset_asks_before_taking_a_key_back(fresh):
    """A row's Reset gives the command its default keys back; when another command
    holds one of them now, it asks (Replace / Cancel) instead of binding it twice."""
    drv = fresh[0]
    open_keymap(drv, "Select All")
    record(drv, "select.all")
    rec_key(drv, code="KeyD", key="d", ctrlKey=True)
    drv.execute_script("[...document.querySelectorAll('.km-msg button')].find((b) => b.textContent === 'Replace').click()")
    assert drv.execute_script("return window.__mace.keymap.keysOf('edit.duplicate')") == []
    open_keymap(drv, "Duplicate")
    reset = "document.querySelector(`.km-row[data-id='edit.duplicate'] [data-reset-key]`).click()"
    drv.execute_script(reset)
    H.wait(drv, "return !document.querySelector('.km-rec').hidden")
    msg = rec_msg(drv)
    assert "Ctrl+D" in msg and "Select All" in msg, msg
    # nothing changes until asked: no key is bound twice
    assert drv.execute_script("return window.__mace.keymap.keysOf('edit.duplicate')") == []
    assert drv.execute_script("return window.__mace.keymap.keysOf('select.all')") == ["Mod+D"]
    shot(drv, "keymap_reset_conflict")
    drv.execute_script("[...document.querySelectorAll('.km-msg button')].find((b) => b.textContent === 'Cancel').click()")
    assert drv.execute_script("return document.querySelector('.km-rec').hidden")
    assert drv.execute_script("return window.__mace.keymap.keysOf('edit.duplicate')") == []
    drv.execute_script(reset)
    H.wait(drv, "return !document.querySelector('.km-rec').hidden")
    drv.execute_script("[...document.querySelectorAll('.km-msg button')].find((b) => b.textContent === 'Replace').click()")
    assert drv.execute_script("return window.__mace.keymap.keysOf('edit.duplicate')") == ["Mod+D"]
    assert drv.execute_script("return window.__mace.keymap.keysOf('select.all')") == []
    # a row whose default keys are free resets at once
    open_keymap(drv, "Select All")
    drv.execute_script("document.querySelector(`.km-row[data-id='select.all'] [data-reset-key]`).click()")
    assert drv.execute_script("return window.__mace.keymap.keysOf('select.all')") == ["Mod+A"]
    assert drv.execute_script("return window.__mace.settings.get('keymap')") == []
    drv.execute_script("document.querySelector('#settings').close()")


def test_keymap_editor_warns_browser_nav(fresh):
    drv = fresh[0]
    open_keymap(drv, "Reset View")
    record(drv, "view.reset")
    rec_key(drv, code="ArrowLeft", key="ArrowLeft", altKey=True)
    msg = rec_msg(drv)
    assert "Alt+← is the browser's Back key" in msg and "Use anyway" in msg, msg
    drv.execute_script("[...document.querySelectorAll('.km-msg button')].find((b) => b.textContent === 'Use anyway').click()")
    # not refused: Alt+Left is Nudge's and Rotate 90° Left's, so it asks about those next
    assert "is also Nudge −x and Rotate 90° Left. Replace them?" in rec_msg(drv), rec_msg(drv)
    drv.execute_script("[...document.querySelectorAll('.km-msg button')].find((b) => b.textContent === 'Pick another key').click()")
    assert drv.execute_script("return window.__mace.keymap.keysOf('view.reset')") == ["Home"]
    drv.execute_script("document.querySelector('[data-rec=\"cancel\"]').click(); document.querySelector('#settings').close()")


def test_settings_section_note_and_dependent_control(fresh):
    """A section with rows still waiting gives each its own hint (the editor
    uses most of its rows since Phase 3); Custom background is off until
    Background is Custom color."""
    drv = fresh[0]
    open_settings(drv, "editor")
    pane = "document.querySelector('#settings .set-pane')"
    assert drv.execute_script(f"return {pane}.querySelectorAll('.set-note').length") == 0
    waiting = drv.execute_script(f"return [...{pane}.querySelectorAll('.set-row')].filter((r) => r.querySelector('.set-help')"
                                 " && r.querySelector('.set-help').textContent.includes('Phase 3')).map((r) => r.dataset.row)")
    assert set(waiting) <= {"confirmOverwrite", "keepBackup", "autosaveDrafts"}, waiting
    drv.execute_script("document.querySelector(`#settings [data-sec='viewport']`).click()")
    assert drv.execute_script(f"return {pane}.querySelectorAll('.set-note').length") == 0          # mixed section: per-row hints
    assert drv.execute_script("return document.querySelector('#set-backgroundColor').disabled") is True
    Select(drv.find_element("id", "set-background")).select_by_value("custom")
    H.wait(drv, "return document.querySelector('#set-backgroundColor').disabled === false")
    Select(drv.find_element("id", "set-background")).select_by_value("theme")
    H.wait(drv, "return document.querySelector('#set-backgroundColor').disabled === true")
    drv.execute_script("document.querySelector('#settings').close()")


def test_keymap_recorder_buttons_work_from_the_keyboard(fresh):
    """Enter and Space on the recorder's own buttons press them; they are not
    recorded as the new key (real WebDriver keys, as a keyboard user types)."""
    drv = fresh[0]
    open_keymap(drv, "Duplicate")
    record(drv, "edit.duplicate")
    keys(drv, Keys.TAB)                                                     # to Cancel
    assert drv.execute_script("return document.activeElement.dataset.rec") == "cancel"
    keys(drv, Keys.ENTER)
    assert drv.execute_script("return document.querySelector('.km-rec').hidden")
    assert drv.execute_script("return window.__mace.keymap.keysOf('edit.duplicate')") == ["Mod+D"]
    assert drv.execute_script("return window.__mace.settings.get('keymap')") == []
    # a conflict puts focus on Replace: Enter replaces with the key pressed (Ctrl+D), not with Enter
    open_keymap(drv, "Select All")
    record(drv, "select.all")
    keys(drv, Keys.CONTROL, "d")
    assert "Ctrl+D is also Duplicate. Replace it?" in rec_msg(drv)
    assert drv.execute_script("return document.activeElement.textContent") == "Replace"
    keys(drv, Keys.ENTER)
    assert drv.execute_script("return window.__mace.keymap.keysOf('select.all')") == ["Mod+D"]
    assert drv.execute_script("return window.__mace.keymap.keysOf('edit.duplicate')") == []
    # Space on "Pick another key" keeps recording; the next key is then recorded
    open_keymap(drv, "Show Axes")
    record(drv, "view.axes")
    keys(drv, Keys.CONTROL, "d")
    drv.execute_script("[...document.querySelectorAll('.km-msg button')].find((b) => b.textContent === 'Pick another key').focus()")
    keys(drv, Keys.SPACE)
    assert drv.execute_script("return !document.querySelector('.km-rec').hidden && document.activeElement === document.querySelector('.km-rec')")
    keys(drv, Keys.SHIFT, "q")
    assert drv.execute_script("return window.__mace.keymap.keysOf('view.axes')") == ["Shift+Q"]
    drv.execute_script("document.querySelector('#settings [data-km=\"reset-all\"]').click(); document.querySelector('#settings').close()")


def test_keymap_text_entry_binds_esc_and_backspace(fresh):
    """Bare Esc and Backspace mean cancel and remove in the recorder, so the
    spec's own Esc and Backspace bindings are entered as text."""
    drv = fresh[0]
    open_keymap(drv, "Reset View")
    record(drv, "view.reset")
    assert rec_key(drv, code="Escape", key="Escape") is True                 # bare Esc cancels
    assert drv.execute_script("return document.querySelector('.km-rec').hidden")
    assert drv.execute_script("return document.querySelector('#settings').open")   # and Settings stays open
    record(drv, "view.reset")
    drv.execute_script("document.querySelector('[data-rec=\"text\"]').click()")
    t = drv.find_element("css selector", ".km-text")
    t.send_keys("Esc", Keys.ENTER)
    assert "Esc is also Exit Tool / Select None. Replace it?" in rec_msg(drv), rec_msg(drv)
    drv.execute_script("[...document.querySelectorAll('.km-msg button')].find((b) => b.textContent === 'Replace').click()")
    assert drv.execute_script("return window.__mace.keymap.keysOf('view.reset')") == ["Esc"]
    open_keymap(drv, "Show Axes")
    record(drv, "view.axes")
    drv.execute_script("document.querySelector('[data-rec=\"text\"]').click()")
    drv.find_element("css selector", ".km-text").send_keys("Backspace", Keys.ENTER)
    msg = rec_msg(drv)
    assert "browser's" in msg or "Back" in msg, msg                          # a browser-navigation key: warned
    drv.execute_script("[...document.querySelectorAll('.km-msg button')].find((b) => b.textContent === 'Use anyway').click()")
    assert "Backspace is also Delete. Replace it?" in rec_msg(drv), rec_msg(drv)
    drv.execute_script("[...document.querySelectorAll('.km-msg button')].find((b) => b.textContent === 'Replace').click()")
    assert drv.execute_script("return window.__mace.keymap.keysOf('view.axes')") == ["Backspace"]
    assert drv.execute_script("return window.__mace.keymap.keysOf('edit.delete')") == ["Delete"]
    drv.execute_script("document.querySelector('#settings').close()")
    # pressing them runs the new commands
    open_doc(drv, "structures/MgO.cif")
    time.sleep(0.4)
    body_focus(drv)
    axes = drv.execute_script("return window.__mace.settings.get('showAxes')")
    keys(drv, Keys.BACKSPACE)
    assert drv.execute_script("return window.__mace.settings.get('showAxes')") is (not axes)
    drv.execute_script("document.querySelector('#viewer-main').closest('.viewer-wrap').focus()")
    keys(drv, Keys.ARROW_LEFT)                                                 # turn the camera away first
    turned = drv.execute_script("return window.__mace.viewState().view")
    body_focus(drv)
    keys(drv, Keys.ESCAPE)
    time.sleep(0.3)
    assert drv.execute_script("return window.__mace.viewState().view") != turned   # Esc reset the view
    # Tab in the recorder moves focus on: the recorder is no keyboard trap
    open_keymap(drv, "Reset View")
    record(drv, "view.reset")
    keys(drv, Keys.TAB)
    assert drv.execute_script("return document.activeElement.dataset.rec") == "cancel"
    drv.execute_script("document.querySelector('[data-rec=\"cancel\"]').click(); document.querySelector('#settings [data-km=\"reset-all\"]').click()")
    drv.execute_script("document.querySelector('#settings').close()")


def test_single_keys_ignored_in_inputs(fresh):
    """Criterion 5a: typing v, 1, ? or Shift+B in a field types; it switches
    no tool, view or dialog."""
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    time.sleep(0.4)
    state = "return [window.__mace.viewState().tool, JSON.stringify(window.__mace.viewState().view), window.__mace.viewState().bonds, !!(document.querySelector('#cheatsheet') && document.querySelector('#cheatsheet').open)]"

    def typed(css):
        el = drv.find_element("css selector", css)
        el.click()
        drv.execute_script("arguments[0].value = ''", el)
        before = drv.execute_script(state)
        el.send_keys("v1?")
        ActionChains(drv).key_down(Keys.SHIFT).send_keys("b").key_up(Keys.SHIFT).perform()
        time.sleep(0.2)
        assert drv.execute_script(state) == before, css
        return drv.execute_script("return arguments[0].value", el)
    assert typed("#file-filter") == "v1?B"
    drv.execute_script("const f = document.querySelector('#file-filter'); f.value = ''; f.dispatchEvent(new Event('input'))")
    drv.execute_script("window.__mace.palette.open('')")
    assert typed("#palette input") == "v1?B"
    drv.execute_script("window.__mace.palette.close()")
    open_settings(drv)
    assert typed("#set-search") == "v1?B"
    drv.execute_script("const s = document.querySelector('#set-search'); s.value = ''; s.dispatchEvent(new Event('input'))")
    drv.execute_script("document.querySelector(`#settings [data-sec='viewport']`).click()")
    el = drv.find_element("id", "set-labelSize")
    el.click(); el.send_keys("1")
    assert drv.execute_script(state)[0] == "select"
    drv.execute_script("document.querySelector('#settings').close()")
    drv.execute_script("window.__mace.settings.reset('labelSize')")
    assert not H.errors(drv), H.errors(drv)


# --- Phase 2: the shortcut sheet and rebinding ---------------------------------------------------

def open_cheatsheet(drv):
    body_focus(drv)
    keys(drv, "?")
    H.wait(drv, "return document.querySelector('#cheatsheet') && document.querySelector('#cheatsheet').open")


@pytest.mark.parametrize("size, columns", [((1920, 1080), 4), ((1280, 800), 2)])
def test_cheatsheet_columns(fresh, size, columns):
    """Criterion 3a: four columns at 1920x1080, two at 1280x800."""
    drv = fresh[0]
    H.set_viewport(drv, *size)
    time.sleep(0.3)
    open_cheatsheet(drv)
    heads = drv.execute_script("return [...document.querySelectorAll('#cheatsheet .cheat-col-h')].map((h) => h.textContent)")
    assert heads == ["Global", "File and Edit", "Select and Tools", "View"]
    lefts = drv.execute_script("""return [...new Set([...document.querySelectorAll('#cheatsheet .cheat-col')]
        .filter((c) => c.querySelector('.cheat-row:not([hidden])')).map((c) => Math.round(c.getBoundingClientRect().left)))]""")
    assert len(lefts) == columns, lefts
    for theme in ("light", "dark"):
        drv.execute_script("window.__mace.settings.set('theme', arguments[0])", theme)
        time.sleep(0.2)
        shot(drv, f"cheatsheet_{size[0]}_{theme}")
    drv.execute_script("window.__mace.settings.set('theme', 'light')")
    # every bound command is on it, at its effective keys
    n = drv.execute_script("return document.querySelectorAll('#cheatsheet .cheat-row').length")
    bound = drv.execute_script("return window.__mace.commands.filter((c) => c.keys.length).length")
    fixed = drv.execute_script("return window.__mace.modalRows()")      # Phase 3's keys while moving, fixed
    assert n == bound + fixed
    f = drv.find_element("id", "cheat-filter")
    f.send_keys("view along")
    shown = drv.execute_script("return [...document.querySelectorAll('#cheatsheet .cheat-row:not([hidden])')].map((r) => r.dataset.id)")
    assert shown and all("along" in x.lower() for x in shown), shown
    assert drv.execute_script("return [...document.querySelectorAll('#cheatsheet .cheat-group')].filter((g) => !g.hidden).length") >= 1
    if size[0] == 1920:
        shot(drv, "cheatsheet_filtered")
    f.clear(); f.send_keys(Keys.ESCAPE)
    assert not drv.execute_script("return document.querySelector('#cheatsheet').open")


def pdf_pages(pdf):
    """Pages in a PDF, also when its objects sit in compressed streams (no PDF library needed)."""
    import re
    import zlib
    text = pdf
    for m in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", pdf, re.S):
        try:
            text += zlib.decompress(m.group(1))
        except zlib.error:
            pass
    return len(re.findall(rb"/Type\s*/Page(?![s\w])", text))


def test_cheatsheet_prints_on_one_page(fresh, tmp_path):
    drv = fresh[0]
    H.set_viewport(drv, 1920, 1080)
    open_cheatsheet(drv)
    from selenium.webdriver.common.print_page_options import PrintOptions
    import base64
    import re
    po = PrintOptions()
    po.orientation = "landscape"
    po.page_width, po.page_height = 21.0, 29.7      # A4 (cm), turned by the orientation
    pdf = base64.b64decode(drv.print_page(po))
    target = os.environ.get("MACE_GUI_SHOTS")
    if target:
        Path(target).mkdir(parents=True, exist_ok=True)
        (Path(target) / "cheatsheet_print.pdf").write_bytes(pdf)
    assert pdf_pages(pdf) == 1, pdf_pages(pdf)
    drv.execute_script("document.querySelector('#cheatsheet').close()")


def test_rebind_view_along_a_everywhere(fresh):
    """Criterion 3b: rebinding View Along a to 7 in Settings shows 7 in the
    cheat sheet, the menu, the tooltip and the palette, and 7 works."""
    drv, base, _, xdg = fresh
    open_doc(drv, "structures/MgO.cif")
    time.sleep(0.4)
    open_keymap(drv, "View Along a")
    record(drv, "view.alongA")
    rec_key(drv, code="Digit7", key="7")
    assert drv.execute_script("return window.__mace.keymap.keysOf('view.alongA')") == ["7"]
    drv.execute_script("document.querySelector('#settings').close()")
    flush(drv)
    got = stored(xdg)
    got.pop("theme", None)            # the test fixture's light theme
    assert got == {"keymap": [{"command": "view.alongA", "keys": ["7"]}]}, got

    def everywhere():
        # the cheat sheet
        open_cheatsheet(drv)
        assert drv.execute_script("return document.querySelector('#cheatsheet .cheat-row[data-id=\"view.alongA\"] kbd').textContent") == "7"
        keys(drv, Keys.ESCAPE)
        # View › View Along › a
        open_menu(drv, "View")
        drv.execute_script("[...document.querySelectorAll('.menu .menu-item')].find((b) => b.textContent.includes('View Along')).click()")
        H.wait(drv, "return document.querySelectorAll('.menu').length === 2")
        assert drv.execute_script("return document.querySelectorAll('.menu')[1].querySelector('[data-cmd-id=\"view.alongA\"] kbd').textContent") == "7"
        shot(drv, "rebind_menu_view_along")
        keys(drv, Keys.ESCAPE); keys(drv, Keys.ESCAPE)
        # the toolbar's view-along tooltip
        ActionChains(drv).move_to_element(drv.find_element("id", "tb-along")).perform()
        H.wait(drv, "return !document.querySelector('#tooltip').hidden", 3)
        tip = drv.execute_script("return document.querySelector('#tooltip').textContent")
        assert "7 / 2 / 3" in tip and "a 7" in tip, tip
        shot(drv, "rebind_tooltip")
        ActionChains(drv).move_by_offset(0, 300).perform()
        # the palette
        drv.execute_script("window.__mace.palette.open('view along a')")
        row = drv.execute_script("""const r = document.querySelector('.pal-row[data-id="view.alongA"]');
            return r && [r.querySelector('.t').textContent, r.querySelector('kbd').textContent]""")
        assert row == ["View: View Along a", "7"], row
        shot(drv, "rebind_palette")
        drv.execute_script("window.__mace.palette.close()")
        # 7 views along a; 1 no longer does (views turn at once: the camera's
        # 200 ms animation is off with reduced motion, so each view is final)
        drv.execute_script("window.__mace.settings.set('reduceMotion', 'on')")
        drv.execute_script("document.querySelector('#viewer-main').closest('.viewer-wrap').focus()")
        keys(drv, "3")
        along_c = drv.execute_script("return window.__mace.viewState().view")
        keys(drv, "1")
        assert drv.execute_script("return window.__mace.viewState().view") == along_c
        keys(drv, "7")
        along_a = drv.execute_script("return window.__mace.viewState().view")
        assert along_a != along_c
        drv.execute_script("window.__mace.run('view.alongA')")
        assert drv.execute_script("return window.__mace.viewState().view") == along_a
    everywhere()
    drv.refresh()
    H.wait(drv, "return document.documentElement.dataset.appReady === '1'")
    open_doc(drv, "structures/MgO.cif")
    time.sleep(0.5)
    everywhere()
    drv.execute_script("window.__mace.settings.set('keymap', [])")
    flush(drv)
    assert not H.errors(drv), H.errors(drv)


# --- Phase 2: context menus (spec §2.2) -----------------------------------------------------------

ATOM_MENU = ["Select Equivalents", "Select Element (Mg)", "Select Neighbors", "-", "Replace Element…", "Set Coordinates…",
             "Delete", "-", "Vacancy Here", "Substitute…", "Antisite…", "-", "Measure from Here", "Center View Here", "-", "Copy Coordinates (Fractional)",
             "Copy Coordinates (Cartesian)", "-", "Properties"]
# Export Element Legend… (1.1.6) in both
EMPTY_MENU = ["Add Atom Here", "Paste", "Select All", "-", "View Along", "Snap View to Nearest Axis", "Reset View", "Center Rotation on Unit Cell", "Center Rotation on Selection", "Display Range", "Calculate Bonds…", "-", "Export Image…",
              "Export Element Legend…"]
EXPLORER_MENU = ["Open", "Open to the Side", "Open as Text", "Open as Structure", "Compare With Active", "Plot", "Compare Convergence", "Plot Gallery",
                 "Export Element Legend…", "Phonons with Phonopy…", "Analyze Structures…", "-",
                 "Copy Path", "Reveal in Folder", "Show File History…", "-", "Set Up Calculation", "Submit…", "Run on This Node…", "New Workflow…",
                 "Create or Update Database from This Folder…"]
TAB_MENU = ["Close", "Close Others", "Close to the Right", "-", "Move to Other Group", "Move Left", "Move Right", "-", "Copy Path", "Reveal in Explorer"]


def menu_rows(drv, label):
    H.wait(drv, f"return !!document.querySelector('.menu[aria-label=\"{label}\"]')", 5)
    return drv.execute_script(f"""return [...document.querySelector('.menu[aria-label="{label}"]').children].map((el) =>
        el.classList.contains('menu-sep') ? ['-', '', null] : [el.querySelector('.lbl').textContent,
          (el.querySelector('kbd') || {{}}).textContent || '', el.getAttribute('aria-disabled')])""")


def click_item(drv, label):
    drv.execute_script("[...document.querySelectorAll('.menu .menu-item')].find((b) => b.querySelector('.lbl').textContent === arguments[0]).click()", label)


def stub_clipboard(drv):
    drv.execute_script("window.__copied = null; navigator.clipboard.writeText = (t) => { window.__copied = t; return Promise.resolve(); }")


def test_atom_context_menu(fresh):
    """Criterion 6: a right-click on an atom shows the §2.2 atom menu."""
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    time.sleep(0.6)
    stub_clipboard(drv)
    p = drv.execute_script("return window.__mace.atomScreen(0)")
    canvas = drv.find_element("css selector", "#viewer-main canvas")
    ActionChains(drv).move_to_element_with_offset(canvas, int(p["x"] - p["cx"]), int(p["y"] - p["cy"])).context_click().perform()
    rows = menu_rows(drv, "Atom menu")
    assert [r[0] for r in rows] == ATOM_MENU
    kb = {r[0]: r[1] for r in rows}
    assert kb["Replace Element…"] == "E" and kb["Set Coordinates…"] == "Ctrl+Shift+U" and kb["Delete"] == "Del"
    on = {r[0] for r in rows if r[2] != "true" and r[0] != "-"}
    # Phase 3 enabled the selection entries; the rest arrive with their tools
    assert {"Center View Here", "Copy Coordinates (Fractional)", "Copy Coordinates (Cartesian)", "Properties",
            "Select Equivalents", "Select Element (Mg)", "Select Neighbors"} <= on, on
    for theme in ("light", "dark"):
        drv.execute_script("window.__mace.settings.set('theme', arguments[0])", theme)
        shot(drv, f"ctx_atom_{theme}")
    drv.execute_script("window.__mace.settings.set('theme', 'light')")
    click_item(drv, "Copy Coordinates (Fractional)")
    H.wait(drv, "return window.__copied !== null", 3)
    assert drv.execute_script("return window.__copied") == "0.00000 0.00000 0.00000"
    before = drv.execute_script("return window.__mace.viewState().view")
    ActionChains(drv).move_to_element_with_offset(canvas, int(p["x"] - p["cx"]), int(p["y"] - p["cy"])).context_click().perform()
    menu_rows(drv, "Atom menu")
    click_item(drv, "Center View Here")
    time.sleep(0.3)
    after = drv.execute_script("return window.__mace.viewState().view")
    assert after[:3] != before[:3] and max(abs(x) for x in after[:3]) < 1e-6      # Mg1 sits at the origin
    q = drv.execute_script("return window.__mace.atomScreen(0)")
    assert abs(q["x"] - q["cx"]) < 2 and abs(q["y"] - q["cy"]) < 2                # and is now at the center
    drv.execute_script("window.__mace.contextFor('atom', {atom: {i: 4, image: [0, 0, 0]}})")
    click_item(drv, "Copy Coordinates (Cartesian)")
    H.wait(drv, "return window.__copied && window.__copied !== '0.00000 0.00000 0.00000'", 3)
    want = drv.execute_script("""const s = window.MACE.state.structure;
        return window.__mace.rank.fracToCart(s.frac[4], s.cell).map((x) => x.toFixed(5)).join(' ')""")
    assert drv.execute_script("return window.__copied") == want, want
    assert not H.errors(drv), H.errors(drv)


def test_empty_space_context_menu(fresh):
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    time.sleep(0.5)
    canvas = drv.find_element("css selector", "#viewer-main canvas")
    w, h = canvas.size["width"], canvas.size["height"]
    ActionChains(drv).move_to_element_with_offset(canvas, -w // 2 + 8, -h // 2 + 8).context_click().perform()
    rows = menu_rows(drv, "Viewport menu")
    assert [r[0] for r in rows] == EMPTY_MENU
    shot(drv, "ctx_empty_light")
    click_item(drv, "Reset View")
    assert not drv.execute_script("return document.querySelector('.menu')")
    # near the window's edge the menu flips to stay on screen
    drv.execute_script("window.__mace.contextFor('empty', null, innerWidth - 4, innerHeight - 4)")
    r = drv.execute_script("const r = document.querySelector('.menu').getBoundingClientRect(); return [r.left, r.top, r.right, r.bottom, innerWidth, innerHeight]")
    assert r[0] >= 0 and r[1] >= 0 and r[2] <= r[4] and r[3] <= r[5], r
    shot(drv, "ctx_atom_near_edge")
    keys(drv, Keys.ESCAPE)


def test_explorer_context_menu(fresh):
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    stub_clipboard(drv)
    row = drv.find_element("css selector", '#file-list .file[data-path="structures/NaCl.cif"]')
    # the file list scrolls inside the explorer (the RECENT section shares it): bring the row into its view
    drv.execute_script("arguments[0].scrollIntoView({block: 'center'})", row)
    ActionChains(drv).context_click(row).perform()
    rows = menu_rows(drv, "Explorer menu")
    assert [r[0] for r in rows] == EXPLORER_MENU
    off = {r[0] for r in rows if r[2] == "true"}
    # NaCl is no BAND/DOSS file, nor an output, nor a deck, nor a phonopy set's file
    # (nor an ELASTIC, EOS, QHA or CPHF output, which opens as its results: Open as Structure)
    assert off == {"Open as Structure", "Plot", "Compare Convergence", "Phonons with Phonopy…", "Submit…", "Run on This Node…"}, off
    shot(drv, "ctx_explorer_light")
    click_item(drv, "Copy Path")
    H.wait(drv, "return window.__copied === 'structures/NaCl.cif'", 3)
    ActionChains(drv).context_click(row).perform()
    menu_rows(drv, "Explorer menu")
    click_item(drv, "Open as Text")
    H.wait(drv, "return window.__mace.doc() && window.__mace.doc().kind === 'text'", 5)
    # Open as Text: the windowed viewer (plan B8)
    H.wait(drv, "return document.querySelector('#text-view .tv-rows').textContent.includes('_cell_length_a')", 10)
    # keyboard: Shift+F10 on a focused row
    drv.execute_script("document.querySelector('#file-list .file[data-path=\"structures/Si.cif\"]').focus()")
    keys(drv, Keys.SHIFT, Keys.F10)
    rows = menu_rows(drv, "Explorer menu")
    keys(drv, Keys.ENTER)                                           # the first item: Open
    H.wait(drv, "return window.__mace.doc() && window.__mace.doc().title === 'Si.cif'", 5)
    # Compare With Active: Si (2 atoms) and NaCl (8) cannot be paired atom by atom: the server says so
    drv.execute_script("window.__mace.contextFor('explorer', {path: 'structures/NaCl.cif'})")
    click_item(drv, "Compare With Active")
    H.wait(drv, "return [...document.querySelectorAll('#toasts .toast2.error')].length > 0", 5)
    drv.execute_script("document.querySelectorAll('#toasts > *').forEach((t) => t.remove())")
    # Si.cif and the Si deck hold the same atoms: one Trajectory-style document with the comparison
    open_doc(drv, "structures/Si.cif")
    drv.execute_script("window.__mace.contextFor('explorer', {path: 'outputs/Si_sp.d12'})")
    click_item(drv, "Compare With Active")
    H.wait(drv, "return window.__mace.doc() && window.__mace.doc().title === 'Si.cif ↔ Si_sp.d12'", 10)
    assert drv.execute_script("return document.querySelector('#tab-opt').classList.contains('active')")
    assert not H.errors(drv), H.errors(drv)


def test_context_menu_gives_focus_back(fresh):
    """Shift+F10 then Esc (real keys) returns focus to the explorer row or tab
    the menu was opened for, as a keyboard user expects (spec §5.3)."""
    drv = fresh[0]
    open_doc(drv, "structures/MgO.cif")
    time.sleep(0.3)
    for sel in ("#file-list .file[data-path='structures/NaCl.cif']", ".doc-tab"):
        drv.execute_script("document.querySelector(arguments[0]).focus()", sel)
        keys(drv, Keys.SHIFT, Keys.F10)
        H.wait(drv, "return !!document.querySelector('.menu') && document.activeElement.closest('.menu') !== null")
        keys(drv, Keys.ESCAPE)
        assert not drv.execute_script("return !!document.querySelector('.menu')")
        assert drv.execute_script("return document.activeElement === document.querySelector(arguments[0])", sel), sel
    # running an entry that moves focus nowhere also gives it back
    sel = "#file-list .file[data-path='structures/NaCl.cif']"
    drv.execute_script("navigator.clipboard.writeText = () => Promise.resolve(); document.querySelector(arguments[0]).focus()", sel)
    keys(drv, Keys.SHIFT, Keys.F10)
    drv.execute_script("[...document.querySelectorAll('.menu .menu-item')].find((b) => b.textContent.includes('Copy Path')).click()")
    assert drv.execute_script("return document.activeElement === document.querySelector(arguments[0])", sel)


def test_tab_context_menu(fresh):
    drv = fresh[0]
    for p in ("structures/MgO.cif", "structures/NaCl.cif", "structures/LiF.cif", "structures/Si.cif"):
        open_doc(drv, p)
    stub_clipboard(drv)
    tab = lambda i: drv.find_elements("css selector", ".doc-tab")[i]
    ActionChains(drv).context_click(tab(1)).perform()
    rows = menu_rows(drv, "Tab menu")
    assert [r[0] for r in rows] == TAB_MENU
    shot(drv, "ctx_tab_light")
    click_item(drv, "Copy Path")
    H.wait(drv, "return window.__copied === 'structures/NaCl.cif'", 3)
    ActionChains(drv).context_click(tab(1)).perform()
    menu_rows(drv, "Tab menu")
    click_item(drv, "Reveal in Explorer")
    assert drv.execute_script("return document.activeElement.dataset.path") == "structures/NaCl.cif"
    ActionChains(drv).context_click(tab(2)).perform()
    menu_rows(drv, "Tab menu")
    click_item(drv, "Close to the Right")
    titles = lambda: drv.execute_script("return window.__mace.docs().map((d) => d.title)")
    assert titles() == ["MgO.cif", "NaCl.cif", "LiF.cif"]
    ActionChains(drv).context_click(tab(1)).perform()
    menu_rows(drv, "Tab menu")
    click_item(drv, "Close Others")
    assert titles() == ["NaCl.cif"]
    ActionChains(drv).context_click(tab(0)).perform()
    rows = menu_rows(drv, "Tab menu")
    assert {r[0] for r in rows if r[2] == "true"} == {"Close Others", "Close to the Right", "Move Left", "Move Right"}
    click_item(drv, "Close")
    assert titles() == []
    assert not H.errors(drv), H.errors(drv)


# --- Phase 2: a rejected setting reverts ----------------------------------------------------------

def test_settings_rejected_key_reverts_with_toast(fresh):
    drv = fresh[0]
    open_settings(drv, "advanced")
    drv.execute_script("""const orig = window.fetch;
      window.fetch = (url, init) => {
        // just the request that carries cache_mb (a save still queued from the
        // fixture's reset may go first, and must reach the server as it is)
        // settings are saved as per-key patches through /api/profile (polish P2)
        if (String(url).endsWith('api/profile') && init && init.method === 'POST' && String(init.body).includes('cache_mb')) {
          window.fetch = orig;
          return Promise.resolve(new Response(JSON.stringify({rev: 1, settings: {labelSize: 14, theme: 'light'}, ui: {}, memory: {},
            recent: {files: {}, folders: []}, rejected: {'settings.cache_mb': 'must be a whole number from 64 to 65536 MB'}}),
            {status: 200, headers: {'Content-Type': 'application/json'}}));
        }
        return orig(url, init);
      };""")
    drv.execute_script("window.__mace.settings.set('cache_mb', 128); window.__mace.settings.set('labelSize', 14)")
    flush(drv)
    toasts = drv.execute_script("return [...document.querySelectorAll('#toasts .msg')].map((m) => m.textContent)")
    assert any("Couldn't save Cache limit: must be a whole number from 64 to 65536 MB" in t for t in toasts), toasts
    assert drv.execute_script("return window.__mace.settings.get('cache_mb')") == 512        # reverted
    assert drv.execute_script("return document.querySelector('#set-cache_mb').value") == "512"  # in the dialog too
    assert drv.execute_script("return window.__mace.settings.get('labelSize')") == 14        # the other one stays
    # the server unreachable: kept in this browser, said once
    drv.execute_script("""const orig = window.fetch; window.__origFetch = orig;
      window.fetch = (url, init) => (String(url).endsWith('api/profile') && init && init.method === 'POST')
        ? Promise.reject(new TypeError('NetworkError when attempting to fetch resource.')) : orig(url, init);""")
    drv.execute_script("document.querySelectorAll('#toasts > *').forEach((t) => t.remove())")
    drv.execute_script("window.__mace.settings.set('labelSize', 15)")
    flush(drv)
    flush(drv)
    toasts = drv.execute_script("return [...document.querySelectorAll('#toasts .msg')].map((m) => m.textContent)")
    assert sum("Settings saved in this browser only" in t for t in toasts) == 1, toasts
    assert drv.execute_script("return window.__mace.settings.get('labelSize')") == 15
    drv.execute_script("window.fetch = window.__origFetch; window.__mace.settings.reset('labelSize')")
    flush(drv)
    drv.execute_script("document.querySelector('#settings').close()")


# --- Phase 2: no errors, no clipping, AA contrast in the new surfaces ---------------------------

CLIPPED = """return [...document.querySelectorAll(arguments[0])]
    .filter((el) => (el.offsetParent !== null || getComputedStyle(el).position === 'fixed') && getComputedStyle(el).display !== 'inline')
    .filter((el) => el.scrollWidth > el.clientWidth + 1)
    .map((el) => (el.className || el.tagName) + ': ' + el.textContent.trim().slice(0, 50))"""


def visit_phase2_surfaces(drv, on_each=None):
    """Open every Phase 2 surface once; on_each(name) runs while each is up."""
    open_doc(drv, "structures/MgO.cif")
    time.sleep(0.4)
    each = on_each or (lambda name: None)
    drv.execute_script("window.__mace.palette.open('sup')")
    each("palette")
    drv.execute_script("window.__mace.palette.close()")
    open_settings(drv)
    for sec in ("appearance", "viewport", "units", "editor", "keyboard", "files", "mace", "advanced"):
        drv.execute_script("document.querySelector(`#settings [data-sec='${arguments[0]}']`).click()", sec)
        if sec == "keyboard":
            H.wait(drv, "return !!document.querySelector('#settings .km-row')")
        time.sleep(0.1)
        each(f"settings_{sec}")
    drv.execute_script("document.querySelector('#settings').close()")
    open_cheatsheet(drv)
    each("cheatsheet")
    drv.execute_script("document.querySelector('#cheatsheet').close()")
    for kind, args in (("atom", {"atom": {"i": 0, "image": [0, 0, 0]}}), ("empty", None),
                       ("explorer", {"path": "structures/MgO.cif"}), ("tab", {"docId": drv.execute_script("return window.__mace.docs()[0].id")})):
        drv.execute_script("window.__mace.contextFor(arguments[0], arguments[1])", kind, args)
        each(f"ctx_{kind}")
        keys(drv, Keys.ESCAPE)


def test_phase2_surfaces_only_local_requests_and_no_errors(fresh):
    drv = fresh[0]
    visit_phase2_surfaces(drv)
    urls = drv.execute_script("return performance.getEntriesByType('resource').map((r) => r.name)")
    origin = drv.execute_script("return location.origin")
    assert all(u.startswith(origin) for u in urls), [u for u in urls if not u.startswith(origin)]
    assert not H.errors(drv), H.errors(drv)


@pytest.mark.parametrize("size", [(1280, 800), (1920, 1080), (1024, 640)])
def test_no_label_is_clipped_in_dialogs(fresh, size):
    drv = fresh[0]
    H.set_viewport(drv, *size)
    time.sleep(0.3)
    found = {}

    def check(name):
        sel = ("dialog[open] :is(button, label, .lbl, h2, h3, h4, .set-help, .set-dyn, .km-head span, .km-cmd, .km-src,"
               " .cheat-row .t, .cheat-col-h, .set-nav-item span), .menu .menu-item .lbl, .pal-text")
        bad = drv.execute_script(CLIPPED, sel)
        if bad:
            found[name] = bad
    visit_phase2_surfaces(drv, check)
    assert not found, found


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_settings_and_keymap_text_meet_aa_contrast(fresh, theme):
    drv = fresh[0]
    drv.execute_script("window.__mace.settings.set('theme', arguments[0])", theme)
    time.sleep(0.2)
    worst = {}

    def check(name):
        r = drv.execute_script("""
          const rgb = (s) => (s.match(/[\\d.]+/g) || [0, 0, 0, 0]).map(Number);
          const lum = (c) => { const [r, g, b] = c.slice(0, 3).map((v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; });
            return 0.2126 * r + 0.7152 * g + 0.0722 * b; };
          const bgOf = (el) => { for (let e = el; e; e = e.parentElement) { const c = rgb(getComputedStyle(e).backgroundColor);
              if (c.length < 4 || c[3] > 0.5) return c; } return [255, 255, 255]; };
          let min = [99, ''];
          for (const el of document.querySelectorAll('dialog[open] *')) {
            if (!el.childNodes.length || ![...el.childNodes].some((n) => n.nodeType === 3 && n.textContent.trim())) continue;
            if (el.offsetParent === null || el.closest('[aria-disabled="true"], option, .muted')) continue;
            const cs = getComputedStyle(el), c = rgb(cs.color), b = bgOf(el);
            const [x, y] = [lum(c), lum(b)].sort((p, q) => q - p), ratio = (x + 0.05) / (y + 0.05);
            if (ratio < min[0]) min = [ratio, el.className + ': ' + el.textContent.trim().slice(0, 30)];
          }
          return min""")
        worst[name] = r
    open_settings(drv)
    for sec in ("appearance", "viewport", "keyboard", "advanced"):
        drv.execute_script("document.querySelector(`#settings [data-sec='${arguments[0]}']`).click()", sec)
        if sec == "keyboard":
            H.wait(drv, "return !!document.querySelector('#settings .km-row')")
        time.sleep(0.1)
        check(sec)
    drv.execute_script("document.querySelector('#settings').close()")
    open_cheatsheet(drv)
    check("cheatsheet")
    drv.execute_script("document.querySelector('#cheatsheet').close()")
    low = {k: v for k, v in worst.items() if v[0] < 4.5}
    assert not low, low



def test_settings_server_wins_on_boot(fresh):
    """S2: the file on the server wins over this browser's copy for the keys it holds."""
    drv, base, _, xdg = fresh
    flush(drv)
    drv.execute_script("window.__mace.settings.set('labelSize', 16)")    # this browser: 16
    flush(drv)
    (xdg / "mace").mkdir(parents=True, exist_ok=True)
    (xdg / "mace" / "gui.json").write_text(json.dumps({"version": 1, "settings": {"labelSize": 18, "theme": "dark"}}))
    drv.refresh()
    H.wait(drv, "return document.documentElement.dataset.appReady === '1'")
    assert drv.execute_script("return window.__mace.settings.get('labelSize')") == 18
    assert drv.execute_script("return document.documentElement.dataset.theme") == "dark"
    drv.execute_script("window.__mace.settings.resetAll()")
    flush(drv)


def test_keys_match_by_position_not_by_letter(fresh):
    """S3: an AZERTY keyboard's Q position types "a"; MACE matches the key's
    position (event.code), so it runs the command bound to Q, not to A."""
    drv = fresh[0]
    drv.execute_script("window.__mace.keymap.apply([{command: 'view.reset', keys: ['Alt+Q']}, {command: 'help.about', keys: ['Alt+A']}])")
    open_doc(drv, "structures/MgO.cif")
    time.sleep(0.4)
    body_focus(drv)
    hit = drv.execute_script("""const e = new KeyboardEvent('keydown', {code: 'KeyQ', key: 'a', altKey: true, bubbles: true, cancelable: true});
        document.body.dispatchEvent(e); return e.defaultPrevented""")
    assert hit is True
    assert not drv.execute_script("return !!(document.querySelector('#about') && document.querySelector('#about').open)")
    marks = drv.execute_script("return performance.getEntriesByType('mark').map((m) => m.name).filter((n) => n.startsWith('ui:view.reset') || n.startsWith('ui:help.about'))")
    assert "ui:view.reset:start" in marks and "ui:help.about:start" not in marks, marks
    drv.execute_script("window.__mace.keymap.apply(window.__mace.settings.get('keymap'))")
