"""New Workflow… (Phase 4) in headless Firefox and Chromium at 100 % and
110 % zoom, with the logging fake SLURM: a student builds full_electronic
for two CIFs from the Explorer's folder, changes the SP functional and the
BAND memory, reviews the exact commands, the first step's decks and job
scripts and the projected later steps, and submits; the jobs reach SLURM
(the fake) and the Workflows view shows the new workflow.

MACE_WF_SHOTS=<dir> saves screenshots at 110 % there."""
import os
import shutil
import time
from pathlib import Path

import pytest

import guiharness as H
import submit_util as SU

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")

TEST_DATA = H.REPO_ROOT / "test"
CASES = [("firefox", 1.0), ("firefox", 1.1), ("chromium", 1.0), ("chromium", 1.1)]


@pytest.fixture(params=CASES, ids=[f"{b}-{int(z * 100)}" for b, z in CASES])
def app(request, tmp_path):
    browser, zoom = request.param
    cif = TEST_DATA / "CIFs" / "1_dia_opt_rev1.cif"
    if not cif.exists():
        pytest.skip("test/ data corpus not present")
    from fake_slurm import FakeSlurm
    root = tmp_path / "proj"
    (root / "cifs").mkdir(parents=True)
    for n in ("A", "B"):
        shutil.copy(cif, root / "cifs" / f"{n}.cif")
    fake = FakeSlurm(tmp_path / "fs")
    env = SU.fake_env(fake, tmp_path)
    env.pop("XDG_CACHE_HOME")
    env.update(MACE_GUI_TEST_HOOKS="1", MACE_GUI_NO_WARMUP="1", MACE_SITE_CONFIG="none")
    proc, base, _ = H.start_server(root, tmp_path / "xdg", **env)
    drv = None
    try:
        if browser == "firefox":
            drv = H.firefox(1440, 900, prefs={"layout.css.devPixelsPerPx": str(zoom)})
        else:
            drv = H.chromium(1440, 900, args=(f"--force-device-scale-factor={zoom}",))
        H.load(drv, base)
        yield {"drv": drv, "base": base, "root": root, "fake": fake, "zoom": zoom, "browser": browser}
    finally:
        if drv is not None:
            drv.quit()
        proc.terminate()
        proc.wait(timeout=15)


def _shot(app, name):
    out = os.environ.get("MACE_WF_SHOTS")
    if out and app["zoom"] == 1.1:
        Path(out).mkdir(parents=True, exist_ok=True)
        time.sleep(0.3)
        app["drv"].save_screenshot(str(Path(out) / f"builder-{app['browser']}-110-{name}.png"))


def _click(drv, css, timeout=30):
    """A real click; the builder re-renders a section when a CLI answer
    arrives, so an element found a moment before may be replaced: find it again."""
    from selenium.common.exceptions import StaleElementReferenceException
    for _ in range(10):
        el = H.wait(drv, f"const e = document.querySelector({css!r}); return e && !e.disabled ? e : null", timeout)
        try:
            drv.execute_script("arguments[0].scrollIntoView({block: 'center'})", el)
            el.click()
            return el
        except StaleElementReferenceException:
            time.sleep(0.2)
    raise AssertionError(f"{css} kept being replaced")


def _st(drv, expr):
    return drv.execute_script(f"const s = window.__maceWfBuilder && window.__maceWfBuilder.state(); return s ? ({expr}) : null")


def _no_sideways(drv):
    return drv.execute_script("const s = document.querySelector('.wfb-sec'); return s.scrollWidth <= s.clientWidth + 1")


def test_build_preview_submit_full_electronic(app):
    drv, fake = app["drv"], app["fake"]
    # 1. structures: the Explorer's folder
    drv.execute_script("window.__mace.run('explorer.workflow', {dir: 'cifs'})")
    H.wait(drv, "return document.querySelector('#wfb-dialog[open]')")
    H.wait(drv, "const s = window.__maceWfBuilder.state(); return s.inputs && s.inputs.files.length === 2", 60)
    text = drv.execute_script("return document.querySelector('.wfb-sec').innerText")
    assert "named A in MACE" in text and "named B in MACE" in text and "2 CIF files picked" in text
    assert _no_sideways(drv)
    _shot(app, "1-structures")

    # 2. steps: the ready-made full_electronic chain
    _click(drv, "[data-act='next']")
    _click(drv, "[data-template='full_electronic']")
    H.wait(drv, "return document.querySelectorAll('.wfb-chain .wfb-node').length === 4")
    chain = drv.execute_script("return document.querySelector('.wfb-chain').innerText")
    assert "OPT" in chain and "BAND" in chain and "DOSS" in chain
    assert "Single point" in drv.execute_script("return document.querySelector('.wfb-glossary').innerText")
    _shot(app, "2-steps")

    # 3. settings: SP inherits OPT's settings; its functional becomes PBE0
    _click(drv, "[data-act='next']")
    _click(drv, "[data-form='SP_2']")
    sel = H.wait(drv, "return document.querySelector(\".wfb-form[data-step='SP_2'] select[data-field='functional']\")", 120)
    form = drv.execute_script("return document.querySelector('.wfb-form').innerText")
    assert "inherited from OPT (step 1)" in form
    locked = drv.execute_script("return [...document.querySelectorAll('.wfb-form .su-row.locked')].map((r) => r.dataset.row)")
    # nothing an SP step has is locked: the DFT grid, the SCF rows, spin polarization and the k-point mesh
    # (the last row that stayed locked) are each inherited and can be overridden
    assert locked == [], locked
    assert drv.execute_script("return document.querySelector(\".wfb-form .su-row[data-row='k_points_for_all_files'] input\").value") \
        == "12 24"
    from selenium.webdriver.support.ui import Select
    Select(sel).select_by_value("PBE0")
    H.wait(drv, "const s = window.__maceWfBuilder.state(); const f = s.forms.SP_2; return f && !s.formBusy.SP_2 && "
                "s.draft.steps.SP_2.method && s.draft.steps.SP_2.method.functional === 'PBE0'", 120)
    H.wait(drv, "return document.querySelector(\".wfb-form .su-row[data-row='functional'] .su-badge.changed\")", 30)
    _shot(app, "3-settings")

    # 4. resources: BAND's memory 48G, set here
    _click(drv, "[data-act='next']")
    H.wait(drv, "const s = window.__maceWfBuilder.state(); return s.defaults && !s.defaultsBusy", 120)
    time.sleep(1.0)                      # a check the SP change scheduled has started by now
    H.wait(drv, "const s = window.__maceWfBuilder.state(); return s.defaults && !s.defaultsBusy", 120)
    cell = H.wait(drv, "return document.querySelector(\"tr[data-step='BAND_3'] input[data-res='mem']\")", 120)
    cell.clear()
    cell.send_keys("48G")
    drv.execute_script("arguments[0].dispatchEvent(new Event('change', {bubbles: true}))", cell)
    H.wait(drv, "const s = window.__maceWfBuilder.state(); return s.defaults && !s.defaultsBusy && "
                "s.defaults.steps.BAND_3.origins.memory === 'spec'", 120)
    H.wait(drv, "return document.querySelector(\"tr[data-step='BAND_3'] [data-origin='you']\")", 30)
    assert drv.execute_script("return document.querySelector(\"tr[data-step='DOSS_4'] [data-origin='you']\")") is None
    _shot(app, "4-resources")

    # 5. review: the exact commands, the first step, the projected later steps
    _click(drv, "[data-act='next']")
    H.wait(drv, "const s = window.__maceWfBuilder.state(); return s.preview && !s.previewing", 300)
    p = _st(drv, "s.preview")
    assert p["code"] == 0, p.get("cli_log")
    assert not p["blocked"]
    review = drv.execute_script("return document.querySelector('.wfb-sec').innerText")
    assert "Submit sends 2 jobs now" in review
    assert "mace workflow --spec workflow_configs/gui_specs/" in review and "mace workflow --execute" in review
    proj = {s["step"]: s for s in p["projected"]}
    assert "PBE0" in proj["SP_2"]["deck_text"]
    assert "#SBATCH --mem=48G" in proj["BAND_3"]["script_text"]
    assert "projected" in review.lower()
    assert not fake.calls("sbatch")
    btn = drv.execute_script("return document.querySelector(\"[data-act='submit']\")")
    assert btn.text == "Submit 2 jobs" and btn.is_enabled()
    assert _no_sideways(drv)
    drv.execute_script("document.querySelector(\"details[data-projected='SP_2']\").open = true")
    _shot(app, "5-review")

    # submit: the jobs reach SLURM, the Workflows view shows the workflow
    btn.click()
    H.wait(drv, "return !document.querySelector('#wfb-dialog')", 300)
    assert len(fake.calls("sbatch")) == 2
    H.wait(drv, "return document.querySelector('#tab-workflows.active .wfv-item')", 60)
    H.wait(drv, "const t = document.querySelector('.wfv-title'); return t && t.innerText.includes('Workflow')"
                " || (document.querySelector('.wfv-item.on') && document.querySelector('.wfv-item.on').innerText.includes('OPT → SP → BAND → DOSS'))", 60)
    item = drv.execute_script("return document.querySelector('.wfv-item.on').innerText")
    assert "OPT → SP → BAND → DOSS" in item and "cifs-full_electronic" in item
    H.wait(drv, "return document.querySelector(\".wfv-node[data-mid='A'][data-step='OPT_1']\")", 60)
    _shot(app, "6-started")
    assert not H.errors(drv), H.errors(drv)


def _set(drv, find, value):
    """Set a control and send its change, in one script (the builder re-renders the
    section when a CLI answer arrives, so a handle kept from before can be stale)."""
    H.wait(drv, f"return {find}", 300)
    assert drv.execute_script(f"const e = {find}; if (!e) return false; e.value = arguments[0]; "
                              "e.dispatchEvent(new Event('change', {bubbles: true})); return true", value)


def _field(step, prefix):
    return (f"[...document.querySelectorAll(\".wfb-form[data-step='{step}'] [data-field]\")]"
            f".find((e) => e.dataset.field.startsWith({prefix!r}))")


def test_transport_and_charge_potential_forms(app):
    """TRANSPORT and CHARGE+POTENTIAL are set up with opt2d3's form like BAND
    and DOSS: the temperature range and the potential's kind (at the atoms,
    POTC) changed here reach the projected decks and the submitted spec."""
    drv, fake, root = app["drv"], app["fake"], app["root"]
    drv.execute_script("window.__mace.run('explorer.workflow', {dir: 'cifs'})")
    H.wait(drv, "return document.querySelector('#wfb-dialog[open]')")
    H.wait(drv, "const s = window.__maceWfBuilder.state(); return s.inputs && s.inputs.files.length === 2", 60)
    _click(drv, "[data-act='next']")
    _click(drv, "[data-template='transport_analysis']")
    H.wait(drv, "return document.querySelectorAll('.wfb-chain .wfb-node').length === 3")
    _set(drv, "document.querySelector('#wfb-add')", "CHARGE+POTENTIAL")
    H.wait(drv, "return document.querySelectorAll('.wfb-chain .wfb-node').length === 4")

    # TRANSPORT: 200 K to 600 K
    _click(drv, "[data-act='next']")
    _click(drv, "[data-form='TRANSPORT_3']")
    for prefix, value in (("q:Minimum temperature", "200"), ("q:Maximum temperature", "600")):
        _set(drv, _field("TRANSPORT_3", prefix), value)
        H.wait(drv, "const s = window.__maceWfBuilder.state(); return !s.formBusy.TRANSPORT_3 && s.forms.TRANSPORT_3 && "
                    "s.draft.steps.TRANSPORT_3.settings_text && "
                    f"Object.keys(s.draft.steps.TRANSPORT_3.answers).some((k) => k.startsWith({prefix!r}))", 300)
    text = _st(drv, "s.draft.steps.TRANSPORT_3.settings_text")
    assert '"temperature_range"' in text and "200.0" in text and "600.0" in text
    H.wait(drv, "return document.querySelector(\".wfb-steplist [data-form='TRANSPORT_3'] .su-badge.changed\")", 30)
    form = drv.execute_script("return document.querySelector('.wfb-form').innerText")
    assert "Minimum temperature" in form and "stand-in" in form
    assert _no_sideways(drv)
    _shot(app, "tc-3-transport")

    # CHARGE+POTENTIAL: the potential at the atoms (its second "Select type" row)
    _click(drv, "[data-form='CHARGE+POTENTIAL_4']")
    rows = H.wait(drv, "const r = window.__maceWfBuilder.state().forms['CHARGE+POTENTIAL_4']; "
                       "return r && r.form.filter((f) => f.id.startsWith('q:Select type')).map((f) => f.label)", 300)
    assert len(rows) == 2 and "potential" in rows[1].lower(), rows
    _set(drv, _field("CHARGE+POTENTIAL_4", "q:Select type (1-2) [1]:#2"), "POTC")
    H.wait(drv, "const s = window.__maceWfBuilder.state(), d = s.draft.steps['CHARGE+POTENTIAL_4']; "
                "return !s.formBusy['CHARGE+POTENTIAL_4'] && d.settings_text && d.settings_text.includes('POTC')", 300)
    _shot(app, "tc-3-charge")

    # review: the projected decks carry the changes; submit
    _click(drv, "[data-act='next']")
    _click(drv, "[data-act='next']")
    H.wait(drv, "const s = window.__maceWfBuilder.state(); return s.preview && !s.previewing", 300)
    p = _st(drv, "s.preview")
    assert p["code"] == 0 and not p["blocked"], p.get("cli_log")
    proj = {s["step"]: s for s in p["projected"]}
    tr = [ln.strip() for ln in proj["TRANSPORT_3"]["deck_text"].splitlines()]
    assert "200.0 600.0 50.0" in tr, proj["TRANSPORT_3"]["deck_text"]
    assert "POTC" in proj["CHARGE+POTENTIAL_4"]["deck_text"] and "ECH3" in proj["CHARGE+POTENTIAL_4"]["deck_text"]
    assert not fake.calls("sbatch")
    _shot(app, "tc-5-review")
    btn = drv.execute_script("return document.querySelector(\"[data-act='submit']\")")
    assert btn.text == "Submit 2 jobs" and btn.is_enabled()
    btn.click()
    H.wait(drv, "return !document.querySelector('#wfb-dialog')", 300)
    assert len(fake.calls("sbatch")) == 2
    (spec_dir,) = (root / "cifs" / "workflow_configs" / "gui_specs").iterdir()
    assert '"temperature_range"' in (spec_dir / "TRANSPORT_3_settings.json").read_text()
    assert "POTC" in (spec_dir / "CHARGE_POTENTIAL_4_settings.json").read_text()
    assert not H.errors(drv), H.errors(drv)


def test_removing_a_step_keeps_the_others_settings(app):
    """Removing a step keeps every other step's settings, SOC and resources; the steps after it move
    up one key (DOSS_4 -> DOSS_3). It used to wipe every step's settings without a word."""
    drv = app["drv"]
    if app["zoom"] != 1.0:
        pytest.skip("the chain logic does not depend on the zoom")
    drv.execute_script("window.__mace.run('explorer.workflow', {dir: 'cifs'})")
    H.wait(drv, "const s = window.__maceWfBuilder && window.__maceWfBuilder.state(); return s && s.inputs && s.inputs.files.length === 2", 60)
    _click(drv, "[data-act='next']")
    _click(drv, "[data-template='full_electronic']")
    H.wait(drv, "return document.querySelectorAll('.wfb-chain .wfb-node').length === 4")
    drv.execute_script("""const st = window.__maceWfBuilder.state().draft.steps;
        st.OPT_1 = {resources: {ntasks: 64}};
        st.SP_2 = {soc: {soc: true, soc_ecp: 'inpsoc'}, answers: {fmixing: '40'}};
        st.BAND_3 = {resources: {mem: '8G'}};
        st.DOSS_4 = {answers: {npoints: '2000'}, resources: {ntasks: 8}};""")
    _click(drv, "[data-remove='2']")
    H.wait(drv, "return document.querySelectorAll('.wfb-chain .wfb-node').length === 3")
    seq, steps = drv.execute_script("const s = window.__maceWfBuilder.state(); return [s.draft.sequence, s.draft.steps]")
    assert seq == ["OPT", "SP", "DOSS"]
    assert steps == {"OPT_1": {"resources": {"ntasks": 64}},
                     "SP_2": {"soc": {"soc": True, "soc_ecp": "inpsoc"}, "answers": {"fmixing": "40"}},
                     "DOSS_3": {"answers": {"npoints": "2000"}, "resources": {"ntasks": 8}}}, steps
