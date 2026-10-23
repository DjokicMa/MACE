"""A later step's Spin polarized row in the New Workflow dialog (1.1.6): inherited from the step before with
its spin treatment shown, switched off for the SP step only (the row says it starts from a fresh guess), Reset
goes back, and the projected SP deck in the review has no SPIN and says so in its title."""
import pytest

import guiharness as H
from test_workflow_builder_step_options_e2e import _click, app  # noqa: F401  (the fixture)

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")

FORM = ".wfb-form[data-step='SP_2']"
SWITCH = f"{FORM} input[data-field='spin']"


def _row(drv, part):
    return drv.execute_script(f"const r = document.querySelector(\"{FORM} .su-row[data-row='spin']\"); "
                              f"const e = r && r.querySelector({part!r}); return e && e.textContent")


def test_sp_spin_overrides_the_parents(app):
    drv = app["drv"]
    drv.execute_script("window.__mace.run('explorer.workflow', {dir: 'cifs'})")
    H.wait(drv, "const s = window.__maceWfBuilder && window.__maceWfBuilder.state(); return s && s.inputs && s.inputs.files.length === 2", 60)
    _click(drv, "[data-act='next']")
    _click(drv, "[data-template='opt_sp']")
    H.wait(drv, "return document.querySelectorAll('.wfb-chain .wfb-node').length === 2")
    _click(drv, "[data-act='next']")
    _click(drv, "[data-form='SP_2']")
    H.wait(drv, f"return document.querySelector(\"{SWITCH}\")", 120)
    # inherited, shown as the spin treatment, and not locked
    assert drv.execute_script(f"const e = document.querySelector(\"{SWITCH}\"); return !e.disabled && e.checked")
    assert _row(drv, ".su-badge") == "inherited from OPT (step 1) (spin-polarized)", _row(drv, ".su-badge")
    _click(drv, SWITCH)
    H.wait(drv, "const s = window.__maceWfBuilder.state(); return !s.formBusy.SP_2 && s.draft.steps.SP_2 && s.draft.steps.SP_2.answers.spin === 'No'", 120)
    H.wait(drv, f"const r = document.querySelector(\"{FORM} .su-row[data-row='spin'] [data-part='field-note']\"); return r", 120)
    assert _row(drv, ".su-badge").startswith("changed from the step before (was spin-polarized)"), _row(drv, ".su-badge")
    assert _row(drv, "[data-part='field-note']") == "starts from a fresh guess: the parent's wavefunction is spin-polarized"
    # Reset goes back to the inherited value
    _click(drv, f"{FORM} [data-reset='spin']")
    H.wait(drv, "const s = window.__maceWfBuilder.state(); return !s.formBusy.SP_2 && !('spin' in s.draft.steps.SP_2.answers)", 120)
    H.wait(drv, f"const e = document.querySelector(\"{SWITCH}\"); return e && e.checked", 60)
    assert _row(drv, ".su-badge") == "inherited from OPT (step 1) (spin-polarized)"
    _click(drv, SWITCH)
    H.wait(drv, "const s = window.__maceWfBuilder.state(); return !s.formBusy.SP_2 && s.draft.steps.SP_2.answers.spin === 'No'", 120)
    _click(drv, "[data-act='next']")
    H.wait(drv, "return window.__maceWfBuilder.state().sec === 3")
    H.wait(drv, "const s = window.__maceWfBuilder.state(); return s.defaults && !s.defaultsBusy", 120)
    _click(drv, "[data-act='next']")
    H.wait(drv, "const s = window.__maceWfBuilder.state(); return s.preview && !s.previewing", 300)
    p = drv.execute_script("return window.__maceWfBuilder.state().preview")
    assert p["code"] == 0, p.get("cli_log")
    sp = {s["step"]: s for s in p["projected"]}["SP_2"]["deck_text"]
    assert "SPIN" not in [line.strip() for line in sp.splitlines()]
    assert sp.splitlines()[0].endswith(" - SPIN off (spin-polarized parent), fresh SCF guess")
    assert not H.errors(drv), H.errors(drv)
