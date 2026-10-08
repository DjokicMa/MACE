"""Calculation Setup in headless Firefox (plan T3 browser rows, T4; spec §7 Phase 5 criterion 1).

Mod+Shift+R opens Setup on the active structure; the form is the CLI's
questions; Mod+Enter opens a Deck document whose text is the deck the
printed command writes, byte for byte; Save writes the CLI's own JSON
bytes and Copy CLI Command gives the command; the settings have their own
undo, separate from the structure's; an edited .out or an earlier frame of
a run asks which geometry the deck uses before anything runs."""
import json
import shlex
import shutil
import time
from pathlib import Path

import pytest

import guiharness as H
import p4util as U
import setup_routes as R

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")
pytest.importorskip("ase")
from selenium.webdriver.common.keys import Keys  # noqa: E402


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("p5setup")
    root = R.served_root(tmp)
    a = U.start_app(tmp, root=root)
    try:
        yield a
    finally:
        U.stop_app(a)


@pytest.fixture
def fresh(app):
    drv = app[0]
    U.reset(drv)
    drv.execute_script("""try { Object.keys(localStorage).filter((k) => k.startsWith('mace.gui.setup.')).forEach((k) => localStorage.removeItem(k)); } catch (e) {}
        if (window.__mace.setupOpen && window.__mace.ctx().setupOpen) window.__mace.run('calc.setup');""")
    if drv.execute_script("return window.__mace.ctx().setupOpen"):
        drv.execute_script("window.__mace.run('calc.setup')")
    return app


def state(drv):
    return drv.execute_script("""const m = window.__mace.setup; const s = m && m.state(); if (!s) return null;
        return {mode: s.mode, calc: s.calc, answers: s.answers, json: s.jsonText, busy: !!s.busy, stale: !!s.stale,
                previewSha: s.previewSha, sha: s.result && s.result.json_sha256, deck: s.result && s.result.deck,
                deckName: s.result && s.result.deck_name, form: s.result && s.result.form.map((f) => f.id),
                hist: s.history.length, pos: s.pos, error: s.error || null, command: s.result && s.result.command}""")


READY = """const m = window.__mace.setup, a = window.__mace.activeDoc(); if (!m || !a) return false;
    const subj = m.subject(); if (!subj || subj.id !== (a.setupFor || a.id)) return false;
    const s = m.state(); return s && s.result && !s.busy && !s.pendingRun"""


def wait_ready(drv, timeout=120):
    H.wait(drv, READY, timeout)
    time.sleep(0.2)


def open_setup(drv, path, keys=True):
    U.open_doc(drv, path)
    if keys:
        drv.execute_script("document.querySelector('.views .panel.active .viewer-wrap, .views .panel.active').focus()")
        U.keys(drv, Keys.CONTROL, Keys.SHIFT, "r")
    else:
        drv.execute_script("window.__mace.run('calc.setup', {open: true})")
    H.wait(drv, "return !!document.querySelector('#inspector .setup')", 30)
    wait_ready(drv)


def set_field(drv, fid, value):
    drv.execute_script("""const r = document.querySelector(`#inspector .su-row[data-field="${arguments[0]}"]`);
        const c = r.querySelector('select, input'); if (c.type === 'checkbox') c.checked = !!arguments[1]; else c.value = arguments[1];
        c.dispatchEvent(new Event('change', {bubbles: true}));""", fid, value)


def printed_deck(root, tmp, command, out_rel):
    g = tmp / f"g{len(list(tmp.iterdir()))}"
    shutil.copytree(root, g)
    argv = shlex.split(command)
    out = g / out_rel
    before = {p.name for p in out.iterdir()} if out.is_dir() else set()
    R.run_printed(argv, g)
    new = sorted(p for p in out.iterdir() if p.suffix in (".d12", ".d3") and p.name not in before)
    return new[0].name, new[0].read_text()


def retype(drv, fid, text):
    drv.execute_script("document.querySelectorAll('#toasts > *').forEach((t) => t.remove())")
    inp = drv.find_element("css selector", f"[data-field={fid}] input")
    drv.execute_script("arguments[0].scrollIntoView({block: 'center'})", inp)
    inp.click()
    inp.send_keys(Keys.CONTROL, "a")
    inp.send_keys(Keys.BACKSPACE)
    inp.send_keys(text)


def test_calculation_menu_is_real(fresh):
    drv = fresh[0]
    cmds = {c["id"]: c for c in drv.execute_script("return window.__mace.commands")}
    menu = next(m for m in drv.execute_script("return window.__mace.menus()") if m["label"] == "Calculation")
    ids = [i for i in menu["items"] if isinstance(i, str) and i != "-"]
    assert ids and not [i for i in ids if cmds[i]["soon"]], [i for i in ids if cmds[i]["soon"]]
    assert {"calc.setup", "calc.preview", "calc.saveJson", "calc.preflight", "calc.copyCli"} <= set(ids)
    assert any(isinstance(i, dict) and i["label"] == "Open Example Config" for i in menu["items"])
    assert drv.execute_script("return window.__mace.menuReason('calc.preview')") == "Open Calculation Setup first"


def test_row1_keys_open_setup_and_mod_enter_gives_the_printed_commands_deck(fresh, tmp_path):
    """Criterion 1 (row 1): Mod+Shift+R on MgO.cif, the CLI's defaults (HSE06-D3, SPIN), Mod+Enter
    opens the Deck document; its text is the printed command's deck; Save writes the preview's JSON bytes."""
    drv, base, root = fresh[0], fresh[1], fresh[2]
    open_setup(drv, "structures/MgO.cif")
    st = state(drv)
    assert st["mode"] == "cif2d12" and st["calc"] == "OPT"
    ui_rows = drv.execute_script("return [...document.querySelectorAll('#inspector .su-row')].map((r) => r.dataset.field)")
    assert ui_rows == [f for f in st["form"] if f != "calc_type"] or set(ui_rows) == set(st["form"]) - {"calc_type"}
    shown = drv.execute_script("return document.querySelector('[data-field=functional] select').value")
    assert shown == "HSE06" and "SPIN" in st["deck"].splitlines()
    U.shot(drv, "setup_defaults_mgo")
    drv.execute_script("document.querySelector('#inspector .su-calc button').focus()")
    U.keys(drv, Keys.CONTROL, Keys.ENTER)
    H.wait(drv, "const d = window.__mace.activeDoc(); return d && d.deck", 60)
    deck_doc = drv.execute_script("const d = window.__mace.activeDoc(); return {title: d.title, text: d.text}")
    assert (deck_doc["title"], deck_doc["text"]) == (st["deckName"], st["deck"])
    assert drv.execute_script("return document.querySelector('#text-body').textContent") == st["deck"]
    # save, then run the printed command unchanged
    drv.execute_script("window.__mace.run('calc.saveJson')")
    H.wait(drv, "return !!document.querySelector('#su-name')", 10)
    drv.execute_script("document.querySelector('#su-over').click()")
    drv.execute_script("window.__mace.run('calc.saveJson')")
    H.wait(drv, "const s = window.__mace.setup.state(); return s && s.saved", 60)
    saved = drv.execute_script("return window.__mace.setup.state().saved")
    assert (root / saved["json"]).read_text() == st["json"]
    assert saved["json_sha256"] == st["sha"]
    drv.execute_script("window.__mace.run('calc.copyCli')")
    assert drv.execute_script("return window.__mace.lastCopy()") == saved["command"]
    name, text = printed_deck(root, tmp_path, saved["command"], "setup/MgO")
    assert (name, text) == (st["deckName"], st["deck"])
    U.shot(drv, "setup_saved_mgo")
    U.no_errors(drv)


@pytest.mark.parametrize("calc, row", [("SP", 9), ("FREQ", 10)])
def test_follow_up_rows_keep_the_parent(fresh, tmp_path, calc, row):
    drv, base, root = fresh[0], fresh[1], fresh[2]
    open_setup(drv, "outputs/Si_sp.out")
    st = state(drv)
    assert st["mode"] == "opt2d12"
    drv.execute_script("document.querySelector(`#inspector .su-calc [data-calc='${arguments[0]}']`).click()", calc)
    H.wait(drv, "const s = window.__mace.setup.state(); return s.calc === arguments[0] && s.result && !s.busy", 120, calc)
    st = state(drv)
    assert json.loads(st["json"])["functional"] == "PBE0"           # the parent's
    badge = drv.execute_script("return document.querySelector('[data-field=functional] .su-badge').textContent")
    assert badge == "parent"
    assert st["deckName"] == f"Si_sp_{calc.lower()}_PBE0_optimized.d12"
    # the manual route: the parent's settings ("y")
    d = tmp_path / "m"
    shutil.copytree(root / "outputs", d / "in")
    R.run_cli_manual(["opt2d12", "--out-file", "in/Si_sp.out", "--calc-type", calc, "--save-options", "--options-file", "Jm.json",
                      "--output-dir", "out"], R.YES, d)
    assert next((d / "out").glob("*.d12")).read_text() == st["deck"]
    assert (d / "Jm.json").read_text() == st["json"]
    U.shot(drv, f"setup_follow_{calc}")


@pytest.mark.parametrize("calc", ["BAND", "DOSS"])
def test_properties_rows(fresh, tmp_path, calc):
    """Rows 15, 19: BAND (NSUB 1000) and DOSS from Si_sp.out with its wavefunction; the stand-in note."""
    drv, base, root = fresh[0], fresh[1], fresh[2]
    open_setup(drv, "outputs/Si_sp.out")
    drv.execute_script("document.querySelector(`#inspector .su-calc [data-calc='${arguments[0]}']`).click()", calc)
    H.wait(drv, "const s = window.__mace.setup.state(); return s.calc === arguments[0] && s.result && !s.busy", 120, calc)
    st = state(drv)
    assert st["mode"] == "opt2d3" and st["deckName"] == f"Si_{calc.lower()}.d3"
    if calc == "BAND":
        assert "4 0 1000 1 100 1 0" in st["deck"]
    probs = drv.execute_script("return window.__mace.problems().map((p) => p.message)")
    assert any("empty stand-in for Si_sp.f9" in p for p in probs), probs
    # opt2d3's fields start from its own defaults (not the parent's), and every type reads whole:
    # rows of three under SP / OPT / FREQ, no TRANSP… or POTENT…
    badges = drv.execute_script("return [...document.querySelectorAll('#inspector .su-row .su-badge')].map((b) => b.textContent)")
    assert badges and all(b == "default" for b in badges), badges
    rows = drv.execute_script("return [...document.querySelectorAll('#inspector .su-calc-row')].map((r) => [...r.children].map((b) => [b.textContent, b.scrollWidth <= b.clientWidth]))")
    assert [[t for t, _ in r] for r in rows] == [["SP", "OPT", "FREQ"], ["BAND", "DOSS", "TRANSPORT"], ["CHARGE", "POTENTIAL", "CHG+POT"]]
    assert all(fits for r in rows for _, fits in r), rows
    d = tmp_path / "m"
    shutil.copytree(root / "outputs", d / "in")
    R.run_cli_manual(["opt2d3", "-i", "in/Si_sp.out", "-t", calc, "--save-config", "--options-file", "Jm.json", "--output-dir", "out"],
                     R.ENTER, d)
    assert next((d / "out").glob("*.d3")).read_text() == st["deck"]
    U.shot(drv, f"setup_{calc.lower()}")


def test_session_row7_previews_and_save_asks_for_a_cif(fresh):
    drv = fresh[0]
    U.open_doc(drv, "structures/MgO.cif")
    drv.execute_script("window.__mace.editor.select([4], 'replace')")
    drv.execute_script("window.__mace.run('edit.delete')")
    H.wait(drv, "return window.__mace.doc().dirty", 20)
    drv.execute_script("window.__mace.run('calc.setup', {open: true})")
    wait_ready(drv)
    st = state(drv)
    info = drv.execute_script("const s = window.__mace.setup.state(); return [s.result.source, window.__mace.doc().structure.numbers.length, window.__mace.doc().dirty, window.__mace.setup.subject().id, window.__mace.activeDoc().id]")
    assert st["mode"] == "cif2d12" and st["deckName"] and st["deck"].count("\n8 0") == 0, info
    assert "Not saved in the folder" in drv.find_element("css selector", "#inspector .setup").text
    drv.execute_script("window.__mace.run('calc.saveJson')")
    time.sleep(0.4)
    assert "Save the structure as CIF first" in drv.find_element("css selector", "#inspector .setup").text
    U.shot(drv, "setup_session")


def test_a_file_opened_as_text_is_not_a_setup_subject(fresh):
    """A CIF opened as text (Open as Text) while Setup is open: Setup does not
    try to read a structure from the text document (it used to reject with
    "This document has no structure the server can read")."""
    drv = fresh[0]
    open_setup(drv, "structures/MgO.cif", keys=False)
    drv.execute_script("window.__mace.openAsText('structures/MgO.cif')")
    H.wait(drv, "const a = window.__mace.activeDoc(); return a && a.kind === 'text'", 20)
    time.sleep(1.5)                                   # past the auto-preview delay
    assert drv.execute_script("return window.__mace.setup.state()") is None
    U.no_errors(drv)


def test_a_view_made_from_an_output_is_not_a_setup_subject(fresh):
    """Run Analysis of an output while Setup is open: the Run Analysis view has
    no file of its own, so Setup does not try to follow it (it used to ask the
    server for a deck with no path and showed "no path given")."""
    drv = fresh[0]
    open_setup(drv, "outputs/Ag2Br3_opt_unfinished.out", keys=False)
    drv.execute_script("window.__mace.run('analysis.runPlot')")
    H.wait(drv, "const a = window.__mace.activeDoc(); return a && a.kind === 'runplot'", 30)
    time.sleep(1.5)
    assert drv.execute_script("return window.__mace.setup.state()") is None
    assert "no path given" not in drv.find_element("css selector", "#inspector").text
    U.no_errors(drv)


def test_auto_preview_typing_runs_once_and_cancels_the_older(fresh):
    drv, base = fresh[0], fresh[1]
    open_setup(drv, "structures/NaCl.cif", keys=False)
    assert drv.execute_script("return window.__mace.settings.get('setupAutoPreview')") is True
    n0 = len([t for t in drv.execute_script("return window.__mace.tasks()") if t["kind"] == "deck"])
    drv.execute_script("document.querySelector('[data-field=fmixing] input').focus()")
    inp = drv.find_element("css selector", "[data-field=fmixing] input")
    inp.send_keys(Keys.CONTROL, "a")
    for ch in "45":
        inp.send_keys(ch)
        time.sleep(0.15)
    time.sleep(0.2)
    assert len([t for t in drv.execute_script("return window.__mace.tasks()") if t["kind"] == "deck"]) == n0   # nothing before 600 ms
    wait_ready(drv)
    deck_tasks = [t for t in drv.execute_script("return window.__mace.tasks()") if t["kind"] == "deck"][n0:]
    assert len(deck_tasks) == 1, deck_tasks
    assert "FMIXING\n45" in state(drv)["deck"]
    hist = drv.execute_script("return window.__mace.setup.historyRows().map((h) => h.label)")
    assert hist[-1].startswith("FMIXING") and hist.count(hist[-1]) == 1           # typing merged into one entry
    # a second edit while the first runs: the first is superseded
    drv.execute_script("window.__mace.settings.set('setupAutoPreview', true)")
    retype(drv, "fmixing", "50")
    time.sleep(0.7)
    retype(drv, "fmixing", "55")
    wait_ready(drv)
    ts = [t for t in drv.execute_script("return window.__mace.tasks()") if t["kind"] == "deck"][n0:]
    assert "FMIXING\n55" in state(drv)["deck"]
    assert any(t["state"] == "canceled" for t in ts) or len(ts) >= 2
    # off: typing previews nothing (the JSON still comes from the CLI)
    drv.execute_script("window.__mace.settings.set('setupAutoPreview', false)")
    before = state(drv)["previewSha"]
    retype(drv, "fmixing", "60")
    time.sleep(1.5)
    wait_ready(drv)
    st = state(drv)
    assert st["previewSha"] == before and st["stale"]
    assert "Preview is out of date" in drv.find_element("id", "su-state").text
    drv.execute_script("window.__mace.settings.set('setupAutoPreview', true)")


def test_setup_undo_is_its_own_history(fresh):
    drv = fresh[0]
    open_setup(drv, "structures/LiF.cif", keys=False)
    before = state(drv)["json"]
    struct_hist = drv.execute_script("const e = window.__mace.editor && window.__mace.editor.doc(); return e ? e.rows.length : null")
    drv.execute_script("window.__mace.setup.loadExample(window.MACE.state.templates.find((t) => t.name === 'standard_dft_opt'))")
    H.wait(drv, "return [...document.querySelectorAll('#toasts > *')].some((t) => t.textContent.includes('Loaded standard_dft_opt'))", 10)
    wait_ready(drv)
    assert state(drv)["json"] != before
    assert "Undo" in " ".join(U.toasts(drv))
    U.shot(drv, "setup_example_toast")
    drv.execute_script("document.querySelector('#inspector .su-calc button').focus()")
    U.keys(drv, Keys.CONTROL, "z")
    wait_ready(drv)
    assert state(drv)["json"] == before                         # byte for byte
    U.keys(drv, Keys.CONTROL, Keys.SHIFT, "z")
    wait_ready(drv)
    assert state(drv)["json"] != before
    # the Settings chip in History lists the entries
    drv.execute_script("window.__mace.run('edit.history')")
    drv.execute_script("document.querySelector('.hist-chips [data-hist=settings]').click()")
    rows = drv.execute_script("return [...document.querySelectorAll('.hist-host .hist-row')].map((r) => r.textContent)")
    assert any("Loaded standard_dft_opt" in r for r in rows), rows
    U.shot(drv, "setup_history_settings")
    drv.execute_script("document.querySelector('#inspector .su-calc [aria-checked=true]').focus()")
    U.keys(drv, Keys.CONTROL, "z")
    wait_ready(drv)
    # Mod+Z inside a text field is the field's own undo
    n = state(drv)["hist"]
    drv.execute_script("document.querySelectorAll('#toasts > *').forEach((t) => t.remove())")
    inp = drv.find_element("css selector", "[data-field=maxcycle] input")
    drv.execute_script("arguments[0].scrollIntoView({block: 'center'})", inp)
    inp.click()
    inp.send_keys("9")
    U.keys(drv, Keys.CONTROL, "z")
    assert state(drv)["pos"] == state(drv)["hist"] - 1
    assert drv.execute_script("const e = window.__mace.editor && window.__mace.editor.doc(); return e ? e.rows.length : null") == struct_hist
    wait_ready(drv)
    # typed "9" then the field's own undo: back to where it was, so no step (the redo branch went with the typing)
    labels = drv.execute_script("return window.__mace.setup.historyRows().map((h) => h.label)")
    assert not any(l.startswith("SCF max cycles") for l in labels), labels
    # a reload keeps the settings and their history
    hist = state(drv)["hist"]
    H.load(drv, fresh[1])
    U.open_doc(drv, "structures/LiF.cif")
    drv.execute_script("window.__mace.run('calc.setup', {open: true})")
    wait_ready(drv)
    assert state(drv)["hist"] == hist


def test_edited_output_asks_which_geometry(fresh):
    drv, base = fresh[0], fresh[1]
    U.open_doc(drv, "outputs/Si_sp.out")
    drv.execute_script("window.__mace.editor.select([0], 'replace')")
    drv.execute_script("document.querySelector('#viewer-main').closest('.viewer-wrap').focus()")
    drv.execute_script("window.__mace.run('tool.move')")
    drv.execute_script("window.__mace.run('edit.nudgeRight')")
    H.wait(drv, "return window.__mace.doc().dirty", 20)
    n0 = len([t for t in drv.execute_script("return window.__mace.tasks()") if t["kind"] == "deck"])
    drv.execute_script("window.__mace.run('calc.setup', {open: true})")
    H.wait(drv, "return !!document.querySelector('#inspector .su-guard')", 20)
    # read the text in the page in one step: the inspector can re-render between a
    # find_element and its .text, which leaves a stale element under load
    text = drv.execute_script("const g = document.querySelector('#inspector .su-guard'); return g ? g.innerText : ''")
    assert "opt2d12 uses the final geometry in Si_sp.out, not your edits" in text
    assert drv.execute_script("return document.querySelector('[data-act=preview]').disabled")
    assert "Choose which geometry the deck uses" in drv.find_element("id", "su-state").text
    time.sleep(1)
    assert len([t for t in drv.execute_script("return window.__mace.tasks()") if t["kind"] == "deck"]) == n0
    U.shot(drv, "setup_guard_dirty_out")
    drv.execute_script("document.querySelector('[data-act=guardParent]').click()")
    wait_ready(drv)
    st = state(drv)
    assert st["deckName"] == "Si_sp_sp_PBE0_optimized.d12" and st["deck"]
    drv.execute_script("window.__mace.run('tool.select')")


def test_trajectory_frame_guard(fresh):
    drv = fresh[0]
    U.open_doc(drv, "outputs/Ag2Br3_opt_unfinished.out", wait_atoms=False)
    H.wait(drv, "return window.__mace.activeDoc() && window.__mace.activeDoc().kind === 'trajectory'", 30)
    time.sleep(0.5)
    drv.execute_script("window.__mace.run('traj.first')")
    time.sleep(0.5)
    drv.execute_script("window.__mace.run('calc.setup', {open: true})")
    H.wait(drv, "return !!document.querySelector('#inspector .su-guard')", 20)
    # read the text in the page in one step: the inspector can re-render between a
    # find_element and its .text, which leaves a stale element under load
    text = drv.execute_script("const g = document.querySelector('#inspector .su-guard'); return g ? g.innerText : ''")
    assert "not frame 1 of" in text, text
    drv.execute_script("document.querySelector('[data-act=guardParent]').click()")
    wait_ready(drv)
    assert state(drv)["deckName"].startswith("Ag2Br3_opt_unfinished_sp_")
    U.shot(drv, "setup_guard_frame")


def test_preflight_runs_static_only_here(fresh):
    drv = fresh[0]
    open_setup(drv, "structures/Si.cif", keys=False)
    drv.execute_script("window.__mace.run('calc.preflight')")
    H.wait(drv, "return window.__mace.problems().some((p) => p.source === 'Preflight')", 120)
    msgs = drv.execute_script("return window.__mace.problems().filter((p) => p.source === 'Preflight').map((p) => p.message)")
    assert any("CRYSTAL not found on this machine" in m or "ACCEPTED" in m for m in msgs), msgs
    # at 1280 px with Setup open the bottom tabs overflow the center: the active one (Problems) stays in sight
    drv.execute_script("document.querySelector('.bottom-tab[data-bottom=problems]').click()")
    time.sleep(0.3)
    inside = drv.execute_script("""const b = document.querySelector('.bottom-tab.active'), s = document.querySelector('#bottom-tabs');
        const rb = b.getBoundingClientRect(), rs = s.getBoundingClientRect(); return rb.left >= rs.left - 1 && rb.right <= rs.right + 1""")
    assert inside
    U.shot(drv, "setup_preflight")


def test_p1_cif_offers_a_symmetrized_copy(fresh):
    """Plan T13: Setup on a P1 CIF of a symmetric structure says so in Problems,
    with Save Symmetrized Copy… (the Save As dialog, symmetrized CIF chosen)."""
    pytest.importorskip("ase")
    from ase.io import read, write
    drv, root = fresh[0], fresh[2]
    p1 = root / "structures" / "MgO_p1.cif"
    if not p1.exists():
        write(str(p1), read(str(root / "structures" / "MgO.cif")))
        drv.execute_script("window.MACE.loadFiles()")
        time.sleep(0.5)
    open_setup(drv, "structures/MgO_p1.cif", keys=False)
    H.wait(drv, "return window.__mace.problems().some((p) => p.action === 'Save Symmetrized Copy…')", 60)
    p = drv.execute_script("return window.__mace.problems().find((p) => p.action === 'Save Symmetrized Copy…')")
    assert p["severity"] == "info" and "spglib finds Fm-3m (225)" in p["message"]
    U.shot(drv, "setup_p1_note")
    drv.execute_script("document.querySelector('.bottom-tab[data-bottom=problems]').click()")
    H.wait(drv, "return [...document.querySelectorAll('.bottom-pane.problems button')].some((b) => b.textContent.includes('Save Symmetrized Copy'))", 10)
    drv.execute_script("[...document.querySelectorAll('.bottom-pane.problems button')].find((b) => b.textContent.includes('Save Symmetrized Copy')).click()")
    H.wait(drv, "const s = document.querySelector('dialog[open] select'); return !!s", 10)
    fmt = drv.execute_script("return [...document.querySelectorAll('dialog[open] select')].map((s) => s.value)")
    assert "cif:symmetrized" in fmt, fmt
    drv.execute_script("document.querySelectorAll('dialog[open]').forEach((d) => d.close())")
