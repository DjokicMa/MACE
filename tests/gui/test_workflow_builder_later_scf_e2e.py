"""A later step's DFT grid and SCF settings in the New Workflow dialog (1.1.6): they show the parent step's
values as inherited, each can be changed for this step only (and Reset goes back), and the projected deck in
the review has the changes."""
import pytest

import guiharness as H
from test_workflow_builder_step_options_e2e import _click, _set, app  # noqa: F401  (the fixture)

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")

FORM = ".wfb-form[data-step='SP_2']"


def _badge(drv, row):
    return drv.execute_script(f"const r = document.querySelector(\"{FORM} .su-row[data-row='{row}']\"); "
                              "return r && r.querySelector('.su-badge').textContent")


def test_sp_grid_and_maxcycle_override_the_parents(app):
    drv = app["drv"]
    drv.execute_script("window.__mace.run('explorer.workflow', {dir: 'cifs'})")
    H.wait(drv, "const s = window.__maceWfBuilder && window.__maceWfBuilder.state(); return s && s.inputs && s.inputs.files.length === 2", 60)
    _click(drv, "[data-act='next']")
    _click(drv, "[data-template='opt_sp']")
    H.wait(drv, "return document.querySelectorAll('.wfb-chain .wfb-node').length === 2")
    _click(drv, "[data-act='next']")
    _click(drv, "[data-form='SP_2']")
    H.wait(drv, f"return document.querySelector(\"{FORM} select[data-field='dft_grid']\")", 120)
    # inherited, with the parent's value, and not locked
    assert drv.execute_script(f"return !document.querySelector(\"{FORM} select[data-field='dft_grid']\").disabled")
    assert _badge(drv, "dft_grid") == "inherited from OPT (step 1) (XLGRID)", _badge(drv, "dft_grid")
    assert _badge(drv, "maxcycle") == "inherited from OPT (step 1) (800)", _badge(drv, "maxcycle")
    _set(drv, f"document.querySelector(\"{FORM} select[data-field='dft_grid']\")", "XXLGRID")
    H.wait(drv, "const s = window.__maceWfBuilder.state(); return !s.formBusy.SP_2 && s.draft.steps.SP_2 && s.draft.steps.SP_2.answers.dft_grid === 'XXLGRID'", 120)
    _set(drv, f"document.querySelector(\"{FORM} input[data-field='maxcycle']\")", "350")
    H.wait(drv, "const s = window.__maceWfBuilder.state(); return !s.formBusy.SP_2 && s.draft.steps.SP_2.answers.maxcycle === '350'", 120)
    assert _badge(drv, "maxcycle").startswith("changed from the step before (was 800)"), _badge(drv, "maxcycle")
    # Reset goes back to the inherited value
    _click(drv, f"{FORM} [data-reset='dft_grid']")
    H.wait(drv, "const s = window.__maceWfBuilder.state(); return !s.formBusy.SP_2 && !('dft_grid' in s.draft.steps.SP_2.answers)", 120)
    _set(drv, f"document.querySelector(\"{FORM} select[data-field='dft_grid']\")", "XXLGRID")
    H.wait(drv, "const s = window.__maceWfBuilder.state(); return !s.formBusy.SP_2 && s.draft.steps.SP_2.answers.dft_grid === 'XXLGRID'", 120)
    _click(drv, "[data-act='next']")
    H.wait(drv, "return window.__maceWfBuilder.state().sec === 3")
    H.wait(drv, "const s = window.__maceWfBuilder.state(); return s.defaults && !s.defaultsBusy", 120)
    _click(drv, "[data-act='next']")
    H.wait(drv, "const s = window.__maceWfBuilder.state(); return s.preview && !s.previewing", 300)
    p = drv.execute_script("return window.__maceWfBuilder.state().preview")
    assert p["code"] == 0, p.get("cli_log")
    sp = {s["step"]: s for s in p["projected"]}["SP_2"]["deck_text"]
    assert "XXLGRID" in sp and "MAXCYCLE\n350\n" in sp
    assert not H.errors(drv), H.errors(drv)
