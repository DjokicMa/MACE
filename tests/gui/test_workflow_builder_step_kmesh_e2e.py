"""A later step's k-point mesh row in the New Workflow dialog (1.1.6): inherited from the step before with its
SHRINK record shown, changed for the SP step only, Reset goes back, "auto" is offered beside the inherited
mesh, and the projected SP deck in the review has the step's own SHRINK."""
import pytest

import guiharness as H
from test_workflow_builder_step_options_e2e import _click, _set, app  # noqa: F401  (the fixture)

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")

FORM = ".wfb-form[data-step='SP_2']"
FIELD = f"{FORM} input[data-field='k_points_for_all_files']"


def _row(drv, part):
    return drv.execute_script(f"const r = document.querySelector(\"{FORM} .su-row[data-row='k_points_for_all_files']\"); "
                              f"const e = r && r.querySelector({part!r}); return e && e.textContent")


def _shrink(deck):
    lines = [line.strip() for line in deck.splitlines()]
    return lines[lines.index("SHRINK") + 1]


def test_sp_kmesh_overrides_the_parents(app):
    drv = app["drv"]
    drv.execute_script("window.__mace.run('explorer.workflow', {dir: 'cifs'})")
    H.wait(drv, "const s = window.__maceWfBuilder && window.__maceWfBuilder.state(); return s && s.inputs && s.inputs.files.length === 2", 60)
    _click(drv, "[data-act='next']")
    _click(drv, "[data-template='opt_sp']")
    H.wait(drv, "return document.querySelectorAll('.wfb-chain .wfb-node').length === 2")
    _click(drv, "[data-act='next']")
    _click(drv, "[data-form='SP_2']")
    H.wait(drv, f"return document.querySelector(\"{FIELD}\")", 120)
    # inherited, in the deck's own words, and not locked: the input shows the mesh, auto offered beside it
    assert drv.execute_script(f"const e = document.querySelector(\"{FIELD}\"); return !e.disabled && e.value") == "12 24"
    assert _row(drv, ".su-badge") == "inherited from OPT (step 1) (SHRINK 12 24)", _row(drv, ".su-badge")
    assert drv.execute_script(f"const e = document.querySelector(\"{FIELD}\"); "
                              "return [...e.list.options].map((o) => o.value)") == ["auto", "12 24"]
    assert not drv.execute_script(f"return document.querySelector(\"{FORM} .su-row.locked[data-row='k_points_for_all_files']\")")
    _set(drv, f"document.querySelector(\"{FIELD}\")", "9 18")
    H.wait(drv, "const s = window.__maceWfBuilder.state(); return !s.formBusy.SP_2 && s.draft.steps.SP_2 && "
                "s.draft.steps.SP_2.answers.k_points_for_all_files === '9 18'", 120)
    H.wait(drv, f"const r = document.querySelector(\"{FORM} .su-row[data-row='k_points_for_all_files'] .su-badge.changed\"); return r", 120)
    assert _row(drv, ".su-badge") == "changed from the step before (was SHRINK 12 24)", _row(drv, ".su-badge")
    deck = drv.execute_script("return window.__maceWfBuilder.state().forms.SP_2.deck")
    assert _shrink(deck) == "9 18"
    # Reset goes back to the inherited mesh; an emptied field does too
    _click(drv, f"{FORM} [data-reset='k_points_for_all_files']")
    H.wait(drv, "const s = window.__maceWfBuilder.state(); return !s.formBusy.SP_2 && !('k_points_for_all_files' in s.draft.steps.SP_2.answers)", 120)
    H.wait(drv, f"const e = document.querySelector(\"{FIELD}\"); return e && e.value === '12 24'", 60)
    assert _row(drv, ".su-badge") == "inherited from OPT (step 1) (SHRINK 12 24)"
    _set(drv, f"document.querySelector(\"{FIELD}\")", "auto")
    H.wait(drv, "const s = window.__maceWfBuilder.state(); return !s.formBusy.SP_2 && s.draft.steps.SP_2.answers.k_points_for_all_files === 'auto'", 120)
    H.wait(drv, f"return document.querySelector(\"{FORM} .su-row[data-row='k_points_for_all_files'] [data-part='field-note']\")", 120)
    assert _row(drv, "[data-part='field-note']").startswith("auto: SHRINK ")
    _set(drv, f"document.querySelector(\"{FIELD}\")", "")
    H.wait(drv, "const s = window.__maceWfBuilder.state(); return !s.formBusy.SP_2 && !('k_points_for_all_files' in s.draft.steps.SP_2.answers)", 120)
    _set(drv, f"document.querySelector(\"{FIELD}\")", "9 18")
    H.wait(drv, "const s = window.__maceWfBuilder.state(); return !s.formBusy.SP_2 && s.draft.steps.SP_2.answers.k_points_for_all_files === '9 18'", 120)
    _click(drv, "[data-act='next']")
    H.wait(drv, "return window.__maceWfBuilder.state().sec === 3")
    H.wait(drv, "const s = window.__maceWfBuilder.state(); return s.defaults && !s.defaultsBusy", 120)
    _click(drv, "[data-act='next']")
    H.wait(drv, "const s = window.__maceWfBuilder.state(); return s.preview && !s.previewing", 300)
    p = drv.execute_script("return window.__maceWfBuilder.state().preview")
    assert p["code"] == 0, p.get("cli_log")
    assert _shrink({s["step"]: s for s in p["projected"]}["SP_2"]["deck_text"]) == "9 18"
    assert _shrink(p["first_step"][0]["deck_text"]) == "12 24"
    assert not H.errors(drv), H.errors(drv)
