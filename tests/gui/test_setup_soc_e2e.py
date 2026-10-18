"""Spin-orbit coupling in Calculation Setup (1.1.6), in a real browser.

Run in both browsers (MACE_GUI_BROWSER=firefox|chromium); each test runs at
UI scale 100% and 110%. The SOC section only sends keys of the CLI's own
settings file; so the deck the preview shows is the deck the saved file
writes, and the refusals and warnings are the CLI's words."""
import json
import time

import pytest

import guiharness as H
import p4util as U
import setup_routes as R
from test_setup_e2e import READY, printed_deck, set_field, state
from test_setup_soc_api import HFO_CIF, INAS_CIF, PBTE_CIF, SOC_READY

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")
from selenium.webdriver.common.keys import Keys  # noqa: E402

SOC = '#inspector .su-sec[data-sec="Spin-orbit coupling"]'


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("socsetup")
    root = R.served_root(tmp)
    d = root / "soc"
    d.mkdir()
    for name, text in (("PbTe", PBTE_CIF), ("HfO", HFO_CIF), ("InAs", INAS_CIF)):
        (d / f"{name}.cif").write_text(text)
    a = U.start_app(tmp, root=root)
    try:
        yield a
    finally:
        U.stop_app(a)


@pytest.fixture(params=[100, 110], ids=["scale100", "scale110"])
def fresh(app, request):
    drv = app[0]
    U.reset(drv)
    drv.execute_script("""try { Object.keys(localStorage).filter((k) => k.startsWith('mace.gui.setup.')).forEach((k) => localStorage.removeItem(k)); } catch (e) {}""")
    if drv.execute_script("return window.__mace.ctx().setupOpen"):
        drv.execute_script("window.__mace.run('calc.setup')")
    drv.execute_script("window.__mace.settings.set('uiScale', arguments[0])", request.param)
    time.sleep(0.2)
    yield app
    drv.execute_script("window.__mace.settings.set('uiScale', 100)")


def wait_ready(drv, timeout=180):
    H.wait(drv, READY, timeout)
    time.sleep(0.2)


def open_setup(drv, path):
    U.open_doc(drv, path)
    drv.execute_script("window.__mace.run('calc.setup', {open: true})")
    H.wait(drv, "return !!document.querySelector('#inspector .setup')", 30)
    wait_ready(drv)


def pick_calc(drv, calc):
    drv.execute_script("document.querySelector(`#inspector .su-calc [data-calc='${arguments[0]}']`).click()", calc)
    H.wait(drv, "const s = window.__mace.setup.state(); return s.calc === arguments[0] && s.result && !s.busy && !s.pendingRun", 180, calc)
    time.sleep(0.2)


def answer(drv, answers):
    """The form's rows set one at a time, as a user changes them (each runs the CLI)."""
    for fid, v in answers.items():
        H.wait(drv, "return !!document.querySelector(`#inspector .su-row[data-field='${arguments[0]}'] select, #inspector .su-row[data-field='${arguments[0]}'] input`)", 30, fid)
        set_field(drv, fid, v)
        wait_ready(drv)
        assert state(drv)["answers"].get(fid) == v or json.loads(state(drv)["json"]).get(fid) is not None, fid


def soc_switch(drv):
    """A real click on the SOC switch (scrolled into view first)."""
    sw = drv.find_element("css selector", "#su-soc-on")
    drv.execute_script("arguments[0].scrollIntoView({block: 'center'})", sw)
    sw.click()
    wait_ready(drv)


def soc_on(drv):
    soc_switch(drv)
    H.wait(drv, "const s = window.__mace.setup.state(); return s.patches.soc === true && s.result.soc && s.result.soc.values.soc === true", 60)
    wait_ready(drv)


def schema_soc(drv):
    """The server's schema.soc (the CLI's tables), as the page has it."""
    from test_setup_soc_api import S
    return S.schema()["soc"]


def notes_text(drv):
    return drv.execute_script("const n = document.querySelector('#su-soc-notes'); return n ? n.textContent : ''")


def fits(drv):
    """Nothing in the SOC section wider than the section (no clipped radio text, no sideways scroll)."""
    return drv.execute_script(f"""const s = document.querySelector('{SOC}'); if (!s) return [];
        const r = s.getBoundingClientRect();
        return [...s.querySelectorAll('*')].filter((e) => {{ const b = e.getBoundingClientRect(); return b.width && (b.right > r.right + 1 || b.left < r.left - 1); }})
          .map((e) => e.tagName + '#' + e.id + '.' + e.className).slice(0, 5)""")


def test_pbte_sp_soc_deck_is_the_saved_files_deck(fresh, tmp_path):
    drv, root = fresh[0], fresh[2]
    open_setup(drv, "soc/PbTe.cif")
    pick_calc(drv, "SP")
    answer(drv, SOC_READY)
    assert drv.execute_script("return document.querySelector('#su-soc-on').checked") is False
    soc_on(drv)
    st = state(drv)
    lines = st["deck"].splitlines()
    assert "TWOCOMPON" in lines and "INPSOC" in lines, st["deck"]
    assert json.loads(st["json"])["soc"] is True
    sw = drv.execute_script("const s = document.querySelector('#su-soc-on'); return [s.checked, s.getAttribute('aria-checked'), s.disabled]")
    assert sw == [True, "true", False]
    # the CLI's "SOC deck:" lines, verbatim
    changes = drv.execute_script("return [...document.querySelectorAll('#su-soc-notes .su-soc-changes li')].map((l) => l.textContent)")
    assert changes and changes == drv.execute_script("return window.__mace.setup.state().result.soc.changes"), changes
    assert "What the CLI changed" in notes_text(drv)
    assert drv.execute_script("return document.querySelector('#su-soc-route-inpsoc').checked")
    assert not fits(drv), fits(drv)
    # the Deck document is the preview's deck
    drv.execute_script("window.__mace.run('calc.preview')")
    H.wait(drv, "const d = window.__mace.activeDoc(); return d && d.deck", 120)
    assert drv.execute_script("return window.__mace.activeDoc().text") == st["deck"]
    # Save with Write decks now: the deck on disk is the preview's
    drv.execute_script("window.__mace.run('calc.saveJson')")
    H.wait(drv, "return !!document.querySelector('#su-write')", 10)
    drv.execute_script("const w = document.querySelector('#su-write'); if (!w.checked) w.click();")
    drv.execute_script("const o = document.querySelector('#su-over'); if (!o.checked) o.click();")
    drv.execute_script("window.__mace.run('calc.saveJson')")
    H.wait(drv, "const s = window.__mace.setup.state(); return s && s.saved", 180)
    saved = drv.execute_script("return window.__mace.setup.state().saved")
    assert (root / saved["json"]).read_text() == st["json"]
    decks = [w for w in saved.get("written") or [] if w.endswith(".d12")]
    assert decks, saved
    text = (root / decks[0]).read_text()
    assert "TWOCOMPON" in text.splitlines() and "INPSOC" in text.splitlines()
    assert text == st["deck"]
    # the printed command, run on its own, writes the same deck
    out_rel = decks[0].rsplit("/", 1)[0]
    (root / decks[0]).rename(root / (decks[0] + ".gui"))
    try:
        name, cli = printed_deck(root, tmp_path, saved["command"], out_rel)
    finally:
        (root / (decks[0] + ".gui")).rename(root / decks[0])
    assert (name, cli) == (st["deckName"], st["deck"])
    U.no_errors(drv)


def test_defaults_show_the_clis_refusal(fresh):
    drv = fresh[0]
    open_setup(drv, "soc/PbTe.cif")
    pick_calc(drv, "SP")
    soc_on(drv)
    probs = drv.execute_script("return window.__mace.problems().filter((p) => p.field === 'soc' && p.severity === 'error').map((p) => p.message)")
    assert probs and probs[0].startswith("The CLI wrote no deck:"), probs
    ref = drv.execute_script("const r = document.querySelector('#su-soc-notes .su-soc-refusal'); return r ? r.textContent : ''")
    assert probs[0] in ref
    assert not fits(drv), fits(drv)


def test_opt_switch_is_off_with_the_clis_reason(fresh):
    drv = fresh[0]
    open_setup(drv, "soc/PbTe.cif")
    st = state(drv)
    assert st["calc"] == "OPT"
    why = schema_soc(drv)["not_for"]["OPT"]
    sw = drv.execute_script("""const s = document.querySelector('#su-soc-on');
        return [s.disabled, s.getAttribute('aria-disabled'), document.getElementById(s.getAttribute('aria-describedby')).textContent]""")
    assert sw[0] is True and sw[1] == "true" and why in sw[2], sw


def test_hf_grays_out_stutsc_with_the_clis_reason(fresh):
    drv = fresh[0]
    open_setup(drv, "soc/HfO.cif")
    pick_calc(drv, "SP")
    soc_on(drv)
    why = schema_soc(drv)["stutsc_refused"]["72"]
    r = drv.execute_script("""const s = document.querySelector('#su-soc-route-stutsc'); const id = s.getAttribute('aria-describedby');
        return [s.disabled, s.getAttribute('aria-disabled'), id, id && document.getElementById(id) ? document.getElementById(id).textContent : null,
                !!document.querySelector('#su-soc-route-inpsoc:not([disabled])')]""")
    assert r[0] is True and r[1] == "true" and r[2] == "su-soc-why-stutsc" and why in r[3] and r[4], r
    assert not fits(drv), fits(drv)


def test_undo_and_the_problems_show_action(fresh):
    drv = fresh[0]
    open_setup(drv, "soc/PbTe.cif")
    pick_calc(drv, "SP")
    soc_on(drv)
    # Problems' Show on the refusal (field "soc") focuses inside the SOC section
    drv.execute_script("document.querySelector('.bottom-tab[data-bottom=problems]').click()")
    H.wait(drv, "return [...document.querySelectorAll('.bottom-pane.problems button')].some((b) => b.textContent.trim() === 'Show')", 20)
    drv.execute_script("""const rows = [...document.querySelectorAll('.bottom-pane.problems *')].filter((e) => e.querySelector && e.textContent.includes('The CLI wrote no deck:') && [...e.querySelectorAll('button')].some((b) => b.textContent.trim() === 'Show'));
        const row = rows[rows.length - 1]; [...row.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Show').click();""")
    H.wait(drv, f"return !!document.activeElement.closest('{SOC}')", 10)
    assert drv.execute_script("return document.activeElement.id") == "su-soc-on"
    # Mod+Z: SOC back off (the settings' own undo)
    U.keys(drv, Keys.CONTROL, "z")
    wait_ready(drv)
    H.wait(drv, "const s = window.__mace.setup.state(); return !('soc' in s.patches) && s.result.soc.values.soc !== true", 60)
    wait_ready(drv)
    assert drv.execute_script("return document.querySelector('#su-soc-on').checked") is False
    assert "TWOCOMPON" not in state(drv)["deck"].splitlines()
    # and redo through the panel's own path
    drv.execute_script("window.__mace.setup.redo()")
    H.wait(drv, "const s = window.__mace.setup.state(); return s.patches.soc === true && !s.busy && !s.pendingRun", 60)
    drv.execute_script("window.__mace.setup.undo()")
    H.wait(drv, "const s = window.__mace.setup.state(); return !('soc' in s.patches) && !s.busy && !s.pendingRun", 60)
    U.no_errors(drv)


def test_inas_shows_the_all_electron_warning(fresh):
    drv = fresh[0]
    open_setup(drv, "soc/InAs.cif")
    pick_calc(drv, "SP")
    answer(drv, SOC_READY)
    soc_on(drv)
    warns = drv.execute_script("return [...document.querySelectorAll('#su-soc-notes .su-soc-warn')].map((w) => w.textContent)")
    assert any("As (atomic number 33) is all-electron" in w for w in warns), warns
    assert "TWOCOMPON" in state(drv)["deck"].splitlines()
    assert not fits(drv), fits(drv)


def test_follow_up_badges_name_where_soc_comes_from(fresh):
    """An SP from an .out (opt2d12): the switch is on offer and its value is the parent's;
    BAND (opt2d3) starts from its own defaults, so its off switch is "default", never "from parent"."""
    drv = fresh[0]
    open_setup(drv, "outputs/Si_sp.out")
    pick_calc(drv, "SP")
    sw = drv.execute_script("""const r = document.querySelector('[data-soc-row=soc]');
        return [document.querySelector('#su-soc-on').disabled, r.querySelector('.su-badge').textContent]""")
    assert sw == [False, "from parent .out"], sw
    pick_calc(drv, "BAND")
    why = schema_soc(drv)["not_for"]["BAND"]
    sw = drv.execute_script("""const r = document.querySelector('[data-soc-row=soc]');
        return [document.querySelector('#su-soc-on').disabled, r.querySelector('.su-badge').textContent, r.querySelector('.su-h').textContent]""")
    assert sw[0] is True and sw[1] == "default" and why in sw[2], sw
