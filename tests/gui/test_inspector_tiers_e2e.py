"""The Inspector column in 1.1.6: its Inspector | Setup tabs (a way back from
Calculation Setup besides pressing Setup again), the Basic / Advanced detail
kept per view kind, the N x M x L cells picker that replaced the 2x2x2
checkboxes (structure, optimization and mode views), the FREQ amplitude
(physical by default, labeled when exaggerated) and the activity bar's
hide / reorder. Firefox, headless; the FREQ part copies a corpus file in
(skipped without the gitignored test/ corpus).
"""
import shutil
import time

import pytest

import guiharness as H

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")

CORPUS = H.REPO_ROOT / "test" / "FREQ"
DIA = "2_dia2_opt_rev1_freq_B3LYP-D3-D3_optimized.out"


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("tiers")
    root = H.examples_copy(tmp)
    if (CORPUS / DIA).is_file():
        (root / "freq").mkdir()
        shutil.copy2(CORPUS / DIA, root / "freq" / DIA)
    proc, base, _ = H.start_server(root, tmp / "xdg", MACE_GUI_TEST_HOOKS="1")
    try:
        drv = H.browser("firefox", width=1400, height=900)
    except BaseException:
        proc.terminate()
        proc.wait(timeout=15)
        raise
    try:
        H.load(drv, base)
        drv.execute_script("window.__mace.settings.set('reduceMotion', 'off')")
        yield drv, root
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
    d.execute_script("""const m = window.__mace; if (document.querySelector('#inspector .setup')) m.run('view.inspectorTab');
      m.closeAll(); m.settings.set('inspector.detail', 'remember'); m.settings.set('inspector.tier', {}); m.settings.set('activity.order', []);
      m.settings.set('activity.hidden', []); m.settings.set('activity.visible', true);""")
    time.sleep(0.3)


def js(drv, code, *args):
    return drv.execute_script(code, *args)


def open_doc(drv, path):
    kind = drv.execute_async_script(
        "const done = arguments[1]; window.__mace.open(arguments[0]).then((d) => done(d && d.kind))", path)
    time.sleep(0.6)
    return kind


def run(drv, cmd, args=None):
    js(drv, "window.__mace.run(arguments[0], arguments[1])", cmd, args)
    time.sleep(0.4)


def shown(drv, sel):
    return js(drv, """return [...document.querySelectorAll(arguments[0])].filter((e) => e.offsetParent !== null || getComputedStyle(e).display !== 'none' && e.getClientRects().length).length""", sel)


def test_basic_hides_advanced_and_more_button_shows_them(drv):
    assert open_doc(drv, "structures/MgO.cif") == "structure"
    H.wait(drv, "return !!document.querySelector('#inspector .insp-tabs .insp-tier')", 10)
    # the harness starts every view Advanced (MACE_GUI_INSPECTOR_TIER): everything shows
    assert js(drv, "return document.getElementById('inspector').dataset.inspTier") == "advanced"
    H.wait(drv, "return !!document.querySelector('#inspector .dp-axis')", 10)
    assert shown(drv, "#inspector .dp-axis") == 3
    # Basic for the structure view: the fractional from/to rows and the classic forms hide, a "more" button says so
    js(drv, "document.querySelector('#inspector [data-tier-pick=basic]').click()")
    H.wait(drv, "return document.getElementById('inspector').dataset.inspTier === 'basic'", 5)
    time.sleep(0.3)
    assert shown(drv, "#inspector .dp-axis") == 0
    assert js(drv, "return window.__mace.settings.get('inspector.tier').structure") == "basic"
    more = js(drv, "return [...document.querySelectorAll('#inspector .tier-more')].map((b) => b.textContent)")
    assert more and all(t.endswith("more in Advanced") for t in more), more
    # the cells picker is Basic: it stays
    assert shown(drv, "#inspector [data-cells-picker=dp-cells]") == 1
    # one "more" click: Advanced for this view kind, the rows are back
    js(drv, "document.querySelector('#inspector .tier-more').click()")
    H.wait(drv, "return document.getElementById('inspector').dataset.inspTier === 'advanced'", 5)
    assert shown(drv, "#inspector .dp-axis") == 3
    assert js(drv, "return window.__mace.settings.get('inspector.tier').structure") == "advanced"
    # the palette command toggles it too
    run(drv, "view.inspectorAdvanced")
    assert js(drv, "return document.getElementById('inspector').dataset.inspTier") == "basic"
    assert not H.errors(drv), H.errors(drv)


TIER = "return document.getElementById('inspector').dataset.inspTier"


def tier_is(drv, t):
    H.wait(drv, f"{TIER} === arguments[0]", 5, t)


def test_inspector_detail_always_modes_and_precedence(drv):
    """Settings › Appearance › Inspector detail: Always Basic / Always Advanced beat the per-view
    table and MACE_GUI_INSPECTOR_TIER (the harness's "advanced"); a pick in an Always mode shows
    for the view on screen only and writes neither the setting nor the table."""
    js(drv, "window.__mace.settings.set('inspector.tier', {structure: 'advanced'}); window.__mace.settings.set('inspector.detail', 'basic')")
    assert open_doc(drv, "structures/MgO.cif") == "structure"
    H.wait(drv, "return !!document.querySelector('#inspector .insp-tabs .insp-tier')", 10)
    # Always Basic beats the table's structure: advanced
    tier_is(drv, "basic")
    H.wait(drv, "return !!document.querySelector('#inspector .tier-more')", 10)
    assert "Always Basic" in js(drv, "return document.querySelector('#inspector .insp-tier').title")
    # ... and the server's default for a kind with no entry (trajectory)
    assert open_doc(drv, "outputs/Ag2Br3_opt_unfinished.out") == "trajectory"
    tier_is(drv, "basic")
    # a "more" click: Advanced for this view only; the setting and the table stay
    js(drv, "window.__mace.activate(0)")
    tier_is(drv, "basic")
    H.wait(drv, "return !!document.querySelector('#inspector .tier-more')", 10)
    js(drv, "document.querySelector('#inspector .tier-more').click()")
    tier_is(drv, "advanced")
    assert js(drv, "return window.__mace.commandChecked('view.inspectorAdvanced')") is True
    assert "this view only" in js(drv, "return document.querySelector('#inspector .insp-tier').title")
    assert js(drv, "return window.__mace.settings.get('inspector.detail')") == "basic"
    assert js(drv, "return window.__mace.settings.get('inspector.tier')") == {"structure": "advanced"}
    # another view shows: Basic again, and coming back does not bring the pick back
    js(drv, "window.__mace.activate(1)")
    tier_is(drv, "basic")
    js(drv, "window.__mace.activate(0)")
    tier_is(drv, "basic")
    # View › Advanced Inspector: the same one-view pick, and toggling back clears it
    run(drv, "view.inspectorAdvanced")
    tier_is(drv, "advanced")
    run(drv, "view.inspectorAdvanced")
    tier_is(drv, "basic")
    assert js(drv, "return window.__mace.settings.get('inspector.tier')") == {"structure": "advanced"}
    # Always Advanced applies at once; the switch's Basic is again this view only
    js(drv, "window.__mace.settings.set('inspector.tier', {structure: 'basic', trajectory: 'basic'}); window.__mace.settings.set('inspector.detail', 'advanced')")
    tier_is(drv, "advanced")
    js(drv, "document.querySelector('#inspector [data-tier-pick=basic]').click()")
    tier_is(drv, "basic")
    js(drv, "window.__mace.activate(1)")
    tier_is(drv, "advanced")
    assert js(drv, "return window.__mace.settings.get('inspector.tier')") == {"structure": "basic", "trajectory": "basic"}
    # Remember per view: the table again (it beats the server's "advanced"); an unpicked kind takes the server's
    js(drv, "window.__mace.settings.set('inspector.detail', 'remember')")
    tier_is(drv, "basic")
    js(drv, "window.__mace.activate(0)")
    tier_is(drv, "basic")
    js(drv, "window.__mace.settings.set('inspector.tier', {})")
    tier_is(drv, "advanced")
    # in Remember per view a pick is kept per kind, as before
    js(drv, "document.querySelector('#inspector [data-tier-pick=basic]').click()")
    tier_is(drv, "basic")
    assert js(drv, "return window.__mace.settings.get('inspector.tier')") == {"structure": "basic"}
    assert not H.errors(drv), H.errors(drv)


def test_inspector_detail_in_the_settings_dialog(drv):
    """The Settings row: a radio group under Appearance, set from the keyboard."""
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    assert open_doc(drv, "structures/NaCl.cif") == "structure"
    tier_is(drv, "advanced")
    run(drv, "file.settings")
    H.wait(drv, "return document.querySelector('#settings') && document.querySelector('#settings').open", 10)
    js(drv, "document.querySelector(\"#settings [data-sec='appearance']\").click()")
    H.wait(drv, "return !!document.querySelector(\"#settings [data-key='inspector.detail']\")", 5)
    labels = js(drv, "return [...document.querySelectorAll(\"#settings [data-key='inspector.detail']\")].map((b) => [b.dataset.val, b.textContent, b.getAttribute('aria-checked')])")
    assert labels == [["remember", "Remember per view", "true"], ["basic", "Always Basic", "false"], ["advanced", "Always Advanced", "false"]]
    btn = drv.find_element(By.CSS_SELECTOR, "#settings [data-key='inspector.detail'][data-val='basic']")
    js(drv, "arguments[0].focus()", btn)
    btn.send_keys(Keys.ENTER)
    H.wait(drv, "return window.__mace.settings.get('inspector.detail') === 'basic'", 5)
    tier_is(drv, "basic")
    js(drv, "document.querySelector('#settings').close()")
    assert not H.errors(drv), H.errors(drv)


def test_setup_tab_and_back_to_inspector(drv):
    assert open_doc(drv, "structures/NaCl.cif") == "structure"
    H.wait(drv, "return !!document.querySelector('#inspector .insp-tabs')", 10)
    tab = "#inspector .insp-tabs [data-insp-tab='%s']"
    assert js(drv, f"return document.querySelector(\"{tab % 'inspector'}\").getAttribute('aria-selected')") == "true"
    js(drv, f"document.querySelector(\"{tab % 'setup'}\").click()")
    H.wait(drv, "return !!document.querySelector('#inspector .setup')", 20)
    # in Setup: the tabs stay on top, the Inspector tab is the way back
    assert js(drv, f"return document.querySelector(\"{tab % 'setup'}\").getAttribute('aria-selected')") == "true"
    assert js(drv, "return document.querySelector('#inspector').firstElementChild.classList.contains('insp-tabs')")
    js(drv, f"document.querySelector(\"{tab % 'inspector'}\").click()")
    H.wait(drv, "return !document.querySelector('#inspector .setup') && !!document.querySelector('#inspector .insp-section')", 10)
    # the palette's Show Inspector does the same
    run(drv, "calc.setup")
    H.wait(drv, "return !!document.querySelector('#inspector .setup')", 20)
    run(drv, "view.inspectorTab")
    H.wait(drv, "return !document.querySelector('#inspector .setup')", 10)
    assert not H.errors(drv), H.errors(drv)


def test_cells_picker_sets_any_range(drv):
    assert open_doc(drv, "structures/Si.cif") == "structure"
    H.wait(drv, "return !!document.getElementById('dp-cells-0')", 10)
    for k, v in enumerate([2, 3, 1]):
        js(drv, """const i = document.getElementById('dp-cells-' + arguments[0]); i.value = arguments[1];
          i.dispatchEvent(new Event('change', {bubbles: true}))""", k, v)
        time.sleep(0.2)
    # "=" (same along all) is off until pressed: each axis is its own
    H.wait(drv, "return document.getElementById('tb-range-label').textContent === '2×3×1'", 5)
    assert js(drv, "return window.__mace.display.rangeOf(window.__mace.activeDoc()).hi") == [2, 3, 1]
    # View › Display Range › N × M × L… is checked for a non-cube whole-cell range, Custom… is not
    assert js(drv, "return window.__mace.commandChecked('view.rangeCells')") is True
    assert js(drv, "return window.__mace.commandChecked('view.rangeCustom')") is False
    # the old 2×2×2 checkbox is not on screen
    assert shown(drv, "label:has(> #opt-super)") == 0
    run(drv, "view.rangeCells")
    H.wait(drv, "return !!document.querySelector('#cells-panel[open] #cells-dlg-0')", 10)
    assert js(drv, "return [0, 1, 2].map((k) => document.getElementById('cells-dlg-' + k).value)") == ["2", "3", "1"]
    # "=" pressed: one value sets every periodic direction
    js(drv, "document.querySelector('#cells-panel .cells-link').click()")
    H.wait(drv, "return document.getElementById('tb-range-label').textContent === '2×2×2'", 5)
    js(drv, "document.querySelector('#cells-panel .cells-link').click()")
    js(drv, "document.querySelector('#cells-panel [data-close]').click()")
    assert not H.errors(drv), H.errors(drv)


def test_optimization_view_repeats_cells(drv):
    assert open_doc(drv, "outputs/Ag2Br3_opt_unfinished.out") == "trajectory"
    VB = "window.MACE.state.viewers['viewer-before']"
    H.wait(drv, f"const v = {VB}; return v && v.getModel() && v.visible()", 30)
    time.sleep(0.5)
    count = f"return {VB}.getModelList().reduce((n, m) => n + m.selectedAtoms({{}}).length, 0)"
    n1 = js(drv, count)
    run(drv, "view.range2")
    H.wait(drv, f"{count.replace('return ', 'return (')}) === arguments[0] * 8", 15, n1)
    assert js(drv, "return window.__mace.optCells()") == [2, 2, 2]
    run(drv, "view.range1")
    H.wait(drv, f"{count.replace('return ', 'return (')}) === arguments[0]", 15, n1)
    assert not H.errors(drv), H.errors(drv)


def test_freq_amplitude_is_physical_and_says_so(drv):
    if not (CORPUS / DIA).is_file():
        pytest.skip("test/ corpus not present (gitignored)")
    drv.execute_async_script("const done = arguments[1]; window.__mace.openModes(arguments[0]).then(() => done(1))", f"freq/{DIA}")
    H.wait(drv, "const v = window.__mace.freqView, s = v && v.state(); return s && s.mode && s.display && s.maxDisp", 60)
    time.sleep(0.4)
    s = js(drv, "return window.__mace.freqView.state()")
    assert s["opts"]["scale"] == "physical" and s["drawScale"] == 1
    # CRYSTAL's zero-point amplitude: tenths of an angstrom at most
    assert 0.001 < s["maxDisp"] < 0.3, s["maxDisp"]
    assert "physical amplitude" in js(drv, "return document.querySelector('.fv-mode').textContent")
    # frame 6 of 24 is the turning point: the largest atom moves its printed amplitude, not 0.5 Å
    d = js(drv, """const f = window.MACE.state.viewers['viewer-freq'].getModel().frames, a = f[6], b = f[0];
      return Math.max(...a.map((x, k) => Math.hypot(x.x - b[k].x, x.y - b[k].y, x.z - b[k].z)))""")
    assert abs(d - s["maxDisp"]) < 2e-4, (d, s["maxDisp"])
    # exaggeration ×10 on the log slider: labeled in the header and the note
    js(drv, "const r = document.getElementById('fo-amp'); r.value = 1; r.dispatchEvent(new Event('input', {bubbles: true}))")
    time.sleep(0.3)
    assert "exaggerated ×10" in js(drv, "return document.querySelector('.fv-mode').textContent")
    assert "Exaggerated ×10" in js(drv, "return document.querySelector('.fo-amp-note').textContent")
    # Fit 0.5 Å: the old drawing, named, and kept for the next FREQ
    js(drv, "document.querySelector('[data-fo-scale=normalized]').click()")
    time.sleep(0.4)
    s = js(drv, "return window.__mace.freqView.state()")
    assert s["opts"]["scale"] == "normalized" and abs(s["drawScale"] * s["maxDisp"] - 0.5) < 1e-6
    assert js(drv, "return window.__mace.settings.get('freq.amplitudeScale')") == "normalized"
    js(drv, "window.__mace.settings.set('freq.amplitudeScale', 'physical')")
    # the cells picker replaced the 2×2×2 box
    assert js(drv, "return !document.getElementById('fo-super') && !!document.getElementById('fo-cells-0')")
    n = js(drv, "return window.__mace.freqView.doc().structure.numbers.length")
    for k, v in enumerate([2, 2, 1]):
        js(drv, """const i = document.getElementById('fo-cells-' + arguments[0]); i.value = arguments[1];
          i.dispatchEvent(new Event('change', {bubbles: true}))""", k, v)
        time.sleep(0.4)
    H.wait(drv, "return window.__mace.freqView.state().display.atoms === arguments[0]", 10, 4 * n)
    assert not H.errors(drv), H.errors(drv)


def test_activity_bar_hide_and_reorder(drv):
    ids = lambda: js(drv, "return [...document.querySelectorAll('#activity [data-cmd]')].map((b) => b.dataset.cmd)")
    first = ids()
    assert first[0] == "view.explorer" and first[-1] == "file.settings"
    run(drv, "view.activity.show.view.tasks")
    assert "view.tasks" not in ids()
    run(drv, "view.activity.moveDown", {"id": "view.explorer"})
    assert ids()[1] == "view.explorer"
    # the buttons can be dragged (Settings stays put)
    assert js(drv, "return document.querySelector('#activity [data-cmd=\"view.database\"]').draggable") is True
    assert js(drv, "return document.querySelector('#activity [data-cmd=\"file.settings\"]').draggable") is False
    run(drv, "view.activityBar")
    assert js(drv, "return document.getElementById('shell').classList.contains('no-activity')")
    assert js(drv, "return getComputedStyle(document.getElementById('activity')).display") == "none"
    run(drv, "view.activity.reset")
    assert ids() == first
    assert not js(drv, "return document.getElementById('shell').classList.contains('no-activity')")
    assert not H.errors(drv), H.errors(drv)


def test_panel_header_keeps_underscores_visible(drv):
    assert open_doc(drv, "outputs/Ag2Br3_opt_unfinished.out") == "trajectory"
    H.wait(drv, "return !!document.querySelector('#inspector .insp-bar .pc-doc')", 10)
    # the label's line box holds the underscore's descent (it clipped at line-height 1)
    lh = js(drv, "const e = document.querySelector('#inspector .insp-bar .pc-doc'); return [parseFloat(getComputedStyle(e).lineHeight), parseFloat(getComputedStyle(e).fontSize)]")
    assert lh[0] >= 1.3 * lh[1], lh
    assert not H.errors(drv), H.errors(drv)


def test_split_view_bottom_panel_hides_from_the_right_group(drv):
    """The right editor group's bottom panel had no × (the pane hides the window's panel controls)."""
    assert open_doc(drv, "structures/MgO.cif") == "structure"
    run(drv, "view.splitRight")
    H.wait(drv, """const f = document.querySelector('#group-side iframe'), w = f && f.contentWindow;
      return w && w.document && w.document.documentElement.classList.contains('pane-bottom') && !!w.document.querySelector('#bottom > .pane-x')""", 30)
    pane = "document.querySelector('#group-side iframe').contentWindow"
    assert js(drv, f"const w = {pane}; return w.getComputedStyle(w.document.querySelector('#bottom > .pane-x')).display") != "none"
    js(drv, f"{pane}.document.querySelector('#bottom > .pane-x').click()")
    H.wait(drv, "return document.getElementById('shell').classList.contains('no-bottom')", 10)
    H.wait(drv, f"return !{pane}.document.documentElement.classList.contains('pane-bottom')", 10)
    run(drv, "view.bottom")
    run(drv, "view.closeGroup")
    assert not H.errors(drv), H.errors(drv)
