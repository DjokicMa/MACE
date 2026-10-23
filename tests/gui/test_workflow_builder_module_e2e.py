"""An ELASTIC step in the New Workflow dialog (1.1.6): "Add a step" offers ELASTIC, EOS, QHA and CPHF after
OPT, the step's form is the module block (Calculation Setup's module fields) with its projected deck, and the
review projects the deck the engine writes with it. PRINT is on by default, as in the CLI's questions, the planner
and Calculation Setup: the spec gets print: true, and print: false is kept only when it is unticked."""

import pytest

import guiharness as H
from test_workflow_builder_step_options_e2e import _click, _set, app  # noqa: F401 (the fixture)

pytestmark = pytest.mark.gui
pytest.importorskip("selenium.webdriver")


def test_elastic_step_from_add_a_step_to_the_review(app):  # noqa: F811
    drv = app["drv"]
    drv.execute_script("window.__mace.run('explorer.workflow', {dir: 'cifs'})")
    H.wait(drv, "const s = window.__maceWfBuilder && window.__maceWfBuilder.state(); return s && s.inputs && s.inputs.files.length === 2", 60)
    _click(drv, "[data-act='next']")
    _click(drv, "[data-template='basic_opt']")
    H.wait(drv, "return document.querySelectorAll('.wfb-chain .wfb-node').length === 1")
    offered = drv.execute_script("return [...document.querySelectorAll('#wfb-add option')].map((o) => o.value)")
    assert {"ELASTIC", "EOS", "QHA", "CPHF"} <= set(offered), offered
    _set(drv, "document.querySelector('#wfb-add')", "ELASTIC")
    H.wait(drv, "return document.querySelectorAll('.wfb-chain .wfb-node').length === 2")
    assert drv.execute_script("return window.__maceWfBuilder.state().draft.sequence") == ["OPT", "ELASTIC"]
    # nothing is built from a module step: it is not offered again
    offered = drv.execute_script("return [...document.querySelectorAll('#wfb-add option')].map((o) => o.value)")
    assert "ELASTIC" not in offered and "ELASTIC2" not in offered, offered
    _click(drv, "[data-act='next']")
    _click(drv, "[data-form='ELASTIC_2']")
    form = ".wfb-form[data-step='ELASTIC_2']"
    H.wait(drv, f"return document.querySelector(\"{form} select[data-modf='mode']\")", 180)
    modes = drv.execute_script(f"return [...document.querySelector(\"{form} select[data-modf='mode']\").options].map((o) => o.value)")
    assert modes == ["ELASTCON", "ELAPIEZO", "PIEZOCON", "PIEZOCP", "PHOTOELA"], modes
    assert "ELASTCON\nPRINT\nEND" in drv.execute_script(f"return document.querySelector(\"{form} .wfb-det pre\").textContent")
    assert drv.execute_script(f"return document.querySelector(\"{form} input[data-modf='print']\").checked")
    assert drv.execute_script("return window.__maceWfBuilder.draft().steps.ELASTIC_2") == {"module": {"print": True}}
    _set(drv, f"document.querySelector(\"{form} select[data-modf='mode']\")", "ELAPIEZO")
    H.wait(drv, "const s = window.__maceWfBuilder.state(); const r = s.forms.ELASTIC_2; "
                "return !s.formBusy.ELASTIC_2 && r && r.module === 'ELAPIEZO'", 180)
    _set(drv, f"document.querySelector(\"{form} input[data-modf='numderiv']\")", "5")
    H.wait(drv, "const s = window.__maceWfBuilder.state(); const r = s.forms.ELASTIC_2; "
                "return !s.formBusy.ELASTIC_2 && r && r.describe === 'ELAPIEZO: NUMDERIV 5 PRINT'", 180)
    assert drv.execute_script(f"return document.querySelector(\"{form} .su-row[data-row='mod-numderiv']\").classList.contains('changed')")
    assert "ELAPIEZO\nNUMDERIV\n5\nPRINT\nEND" in drv.execute_script(f"return document.querySelector(\"{form} .wfb-det pre\").textContent")
    sent = drv.execute_script("return window.__maceWfBuilder.draft().steps.ELASTIC_2")
    assert sent == {"module": {"mode": "ELAPIEZO", "numderiv": 5, "print": True}}, sent
    # PRINT unticked: kept as print: false in the draft, no PRINT in the spec or the deck; ticked again: the default
    drv.execute_script(f"document.querySelector(\"{form} input[data-modf='print']\").click()")
    H.wait(drv, "const s = window.__maceWfBuilder.state(); const r = s.forms.ELASTIC_2; "
                "return !s.formBusy.ELASTIC_2 && r && r.describe === 'ELAPIEZO: NUMDERIV 5'", 180)
    assert drv.execute_script("return window.__maceWfBuilder.state().draft.steps.ELASTIC_2.module.print") is False
    assert drv.execute_script("return window.__maceWfBuilder.draft().steps.ELASTIC_2") == {"module": {"mode": "ELAPIEZO", "numderiv": 5}}
    assert drv.execute_script(f"return document.querySelector(\"{form} .su-row[data-row='mod-print']\").classList.contains('changed')")
    drv.execute_script(f"document.querySelector(\"{form} input[data-modf='print']\").click()")
    H.wait(drv, "const s = window.__maceWfBuilder.state(); const r = s.forms.ELASTIC_2; "
                "return !s.formBusy.ELASTIC_2 && r && r.describe === 'ELAPIEZO: NUMDERIV 5 PRINT'", 180)
    assert "print" not in drv.execute_script("return window.__maceWfBuilder.state().draft.steps.ELASTIC_2.module")
    # review: the deck the engine writes with this block, from a stand-in of each structure's OPT
    _click(drv, "[data-act='next']")
    H.wait(drv, "return window.__maceWfBuilder.state().sec === 3")
    H.wait(drv, "const s = window.__maceWfBuilder.state(); return s.defaults && !s.defaultsBusy", 120)
    res = drv.execute_script("return window.__maceWfBuilder.state().defaults.steps.ELASTIC_2.resources")
    assert res["walltime"] == "7-00:00:00" and res["memory_per_cpu"] == "7G", res
    _click(drv, "[data-act='next']")
    H.wait(drv, "const s = window.__maceWfBuilder.state(); return s.preview && !s.previewing", 300)
    p = drv.execute_script("return window.__maceWfBuilder.state().preview")
    assert p["code"] == 0, p.get("cli_log")
    deck = {s["step"]: s for s in p["projected"]}["ELASTIC_2"]["deck_text"]
    assert "ELAPIEZO\nNUMDERIV\n5\nPRINT\nEND" in deck and "OPTGEOM" not in deck
    assert not H.errors(drv), H.errors(drv)
