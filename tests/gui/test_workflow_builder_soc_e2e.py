"""Spin-orbit coupling in New Workflow… (1.1.6) in headless Firefox and
Chromium at 100 % and 110 % zoom: on a PbTe CIF, a chain OPT -> SP, SOC is
switched on for the SP step (its form previews the CLI's SOC deck), the OPT
step shows the group disabled with the CLI's reason, and Review's projected
SP deck is a two-component deck (TWOCOMPON, INPSOC).

MACE_WF_SHOTS=<dir> saves screenshots at 110 % there."""
import os
import time
from pathlib import Path

import pytest

import guiharness as H
import submit_util as SU
from test_setup_soc_api import PBTE_CIF, SOC_READY
from test_workflow_builder_e2e import _click, _no_sideways, _st

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")

CASES = [("firefox", 1.0), ("firefox", 1.1), ("chromium", 1.0), ("chromium", 1.1)]


@pytest.fixture(params=CASES, ids=[f"{b}-{int(z * 100)}" for b, z in CASES])
def app(request, tmp_path):
    browser, zoom = request.param
    from fake_slurm import FakeSlurm
    root = tmp_path / "proj"
    (root / "cifs").mkdir(parents=True)
    (root / "cifs" / "PbTe.cif").write_text(PBTE_CIF)
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
        yield {"drv": drv, "zoom": zoom, "browser": browser}
    finally:
        if drv is not None:
            drv.quit()
        proc.terminate()
        proc.wait(timeout=15)


def _shot(app, name, css=None):
    out = os.environ.get("MACE_WF_SHOTS")
    if not (out and app["zoom"] == 1.1):
        return
    Path(out).mkdir(parents=True, exist_ok=True)
    drv = app["drv"]
    if css:
        drv.execute_script(f"document.querySelector({css!r}).scrollIntoView({{block: 'center'}})")
    time.sleep(0.3)
    drv.save_screenshot(str(Path(out) / f"builder-soc-{app['browser']}-110-{name}.png"))


def _form_ready(drv, step, timeout=180):
    H.wait(drv, f"const s = window.__maceWfBuilder.state(); return s.forms[{step!r}] && !s.formBusy[{step!r}] "
                f"&& document.querySelector(\".wfb-form[data-step='{step}'] fieldset[data-part='soc']\")", timeout)


def test_soc_on_the_sp_step_of_a_workflow(app):
    drv = app["drv"]
    drv.execute_script("window.__mace.run('explorer.workflow', {dir: 'cifs'})")
    H.wait(drv, "return document.querySelector('#wfb-dialog[open]')")
    H.wait(drv, "const s = window.__maceWfBuilder.state(); return s.inputs && s.inputs.files.length === 1", 60)
    # steps: OPT -> SP
    _click(drv, "[data-act='next']")
    _click(drv, "[data-template='opt_sp']")
    H.wait(drv, "return document.querySelectorAll('.wfb-chain .wfb-node').length === 2")
    # the first step's conversion settings a SOC deck needs (an explicit basis, a functional the
    # 2c SCF runs); the first-step form itself is covered by the builder's own e2e test
    drv.execute_script("const s = window.__maceWfBuilder.state(); s.draft.steps.OPT_1 = {answers: arguments[0]};"
                       "s.forms = {};", dict(SOC_READY))
    _click(drv, "[data-act='next']")

    # the OPT step: the group is there, disabled, with the CLI's reason
    _click(drv, "[data-form='OPT_1']")
    _form_ready(drv, "OPT_1")
    opt = drv.execute_script("""const f = document.querySelector("fieldset[data-part='soc']");
      const sw = f.querySelector("[data-soc='soc']"); const why = document.getElementById(f.getAttribute('aria-describedby'));
      return {disabled: f.disabled, sw: sw.disabled, swDesc: sw.getAttribute('aria-describedby') === why.id,
              why: why ? why.innerText : '', legend: f.querySelector('legend').innerText}""")
    assert opt["disabled"] and opt["sw"] and opt["swDesc"], opt
    assert opt["legend"] == "Spin-orbit coupling"
    assert "OPTGEOM" in opt["why"] and "single points" in opt["why"] and "p. 166" in opt["why"], opt
    assert _no_sideways(drv)
    _shot(app, "opt-disabled", "fieldset[data-part='soc']")

    # the SP step: switch SOC on; the form previews the CLI's SOC deck
    _click(drv, "[data-form='SP_2']")
    _form_ready(drv, "SP_2")
    assert drv.execute_script("return document.querySelector(\"fieldset[data-part='soc']\").disabled") is False
    assert drv.execute_script("return document.querySelector(\"[data-soc='route']\").disabled") is True
    _click(drv, "[data-soc='soc']")
    H.wait(drv, "const s = window.__maceWfBuilder.state(); const f = s.forms.SP_2; return f && !s.formBusy.SP_2 && "
                "f.soc && f.soc.values && f.soc.values.soc === true && f.deck && f.deck.includes('TWOCOMPON')", 240)
    sp = drv.execute_script("""const f = document.querySelector("fieldset[data-part='soc']");
      const g = f.querySelector("[data-soc='guess']");
      return {on: f.querySelector("[data-soc='soc']").checked, route: f.querySelector("[data-soc='route']").value,
              routeOff: f.querySelector("[data-soc='route']").disabled,
              restarts: [...g.options].filter((o) => o.value).map((o) => o.disabled),
              guessDesc: (document.getElementById(g.getAttribute('aria-describedby')) || {}).innerText || '',
              badge: f.querySelector("[data-row='soc-on'] .su-badge").innerText,
              errors: f.querySelectorAll("[data-part='soc-error']").length}""")
    assert sp["on"] and sp["route"] == "inpsoc" and not sp["routeOff"], sp
    assert sp["restarts"] == [True, True] and "from scratch" in sp["guessDesc"], sp
    assert sp["badge"] == "set here" and sp["errors"] == 0, sp
    assert _st(drv, "JSON.stringify(window.__maceWfBuilder.draft().steps.SP_2.soc)") == '{"soc":true}'
    # a SMEAR width: the CLI writes SMEAR
    el = H.wait(drv, "return document.querySelector(\"[data-soc='smear']\")")
    el.clear()
    el.send_keys("0.005")
    drv.execute_script("arguments[0].dispatchEvent(new Event('change', {bubbles: true}))", el)
    H.wait(drv, "const s = window.__maceWfBuilder.state(); const f = s.forms.SP_2; return f && !s.formBusy.SP_2 && "
                "f.soc.values.smear === 0.005 && f.deck.includes('SMEAR')", 240)
    assert _no_sideways(drv)
    _shot(app, "sp-enabled", "fieldset[data-part='soc']")

    # review: the projected SP deck is the SOC deck
    _click(drv, "[data-sec='4']")
    H.wait(drv, "const s = window.__maceWfBuilder.state(); return s.preview && !s.previewing", 600)
    p = _st(drv, "s.preview")
    assert not p.get("code"), (p.get("refused"), (p.get("cli_log") or "")[-2000:])
    proj = {x["step"]: x for x in p["projected"]}
    assert "TWOCOMPON" in proj["SP_2"]["deck_text"] and "INPSOC" in proj["SP_2"]["deck_text"]
    assert '"soc": true' in p["spec"]["text"] and '"smear": 0.005' in p["spec"]["text"]
    text = drv.execute_script("return document.querySelector('.wfb-sec').innerText")
    assert "SP (step 2)" in text and "projected" in text
