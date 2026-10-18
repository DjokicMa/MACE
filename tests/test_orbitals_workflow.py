"""ORBITALS is a properties calculation type everywhere the others are listed,
and it is refused from a two-component (SOC) run everywhere one is known.

CRYSTAL23 cannot export two-component spinors: NEWK segfaults for a molecule
and for any k net other than the SCF's, and on the SCF net ORBITALS writes a
header and dies (HPCC, spec sec. 6). The refusal (kchoice.SOC_REFUSAL) is
applied by the deck writer (test_orbitals_deck.py), the workflow planner,
workflow specs (the GUI builder runs them), the GUI's Calculation Setup,
Batch Setup and builder (grayed out with the reason), and `mace preflight`.
"""
import json
import os
import shutil
from pathlib import Path

import pytest

from conftest import REPO_ROOT
from mace.orbitals.kchoice import SOC_REFUSAL

SCF = REPO_ROOT / "tests" / "data" / "orbitals" / "scf"
TWO_C_DECK = """HI two-component
MOLECULE
1
2
1 0.0 0.0 0.0
253 0.0 0.0 1.61
END
BASISSET
POB-TZVP-REV2
DFT
PBE0
END
TWOCOMPON
SOC
END
END
"""
ONE_C_DECK = TWO_C_DECK.replace("TWOCOMPON\nSOC\nEND\n", "").replace("two-component", "one")


# ------------------------------------------------------------------ registries
def test_registries_list_orbitals():
    from mace.workflow.common import constants as C
    assert "ORBITALS" in C.CALC_TYPES and C.CALC_TYPES["ORBITALS"]["depends_on"] == ["SP", "OPT"]
    assert "ORBITALS" in C.D3_CALC_TYPES and "ORBITALS" in C.OPTIONAL_CALC_TYPES
    assert C.SLURM_RESOURCE_DEFAULTS["ORBITALS"]["ntasks"] == C.PROPERTIES_NTASKS
    assert C.SCRATCH_DIR_DEFAULTS["ORBITALS"] == C.SCRATCH_DIR_DEFAULTS["DOSS"]
    from mace.workflow import engine, spec
    assert "ORBITALS" in engine.OPTIONAL_CALC_TYPES and "ORBITALS" in spec.D3_TYPES
    from mace.submission import resources_suggest, site
    assert "ORBITALS" in site.PROP_TYPES and site.type_kind("ORBITALS") == "prop"
    assert site.base_calc_type("orbitals2") == "ORBITALS"
    assert "ORBITALS" in resources_suggest.PROP_TYPES and "ORBITALS" in resources_suggest.PROP_WORK
    from mace.gui import batch, setup, workflow_build
    assert "ORBITALS" in setup.CALC_TYPES["opt2d3"] and setup.mode_for("x.out", "ORBITALS") == "opt2d3"
    assert "ORBITALS" in batch.FOLLOWUPS and batch.followup_mode("ORBITALS") == "opt2d3"
    assert batch.BUCKET_TYPE["completeorbitals"] == "ORBITALS"
    assert workflow_build.calc_kind("ORBITALS2") == "d3" and "ORBITALS" in workflow_build.CALC_HELP
    js = (REPO_ROOT / "mace" / "gui" / "static" / "js" / "batch" / "panel.js").read_text()
    assert '"POTENTIAL", "ORBITALS"]' in js
    assert 'ORBITALS: "ORB"' in (REPO_ROOT / "mace" / "gui" / "static" / "app.js").read_text()


def test_engine_and_planner_basic_configs_are_opt2d3s_default():
    from mace.workflow.engine import WorkflowEngine
    from mace.workflow.planner import WorkflowPlanner
    eng = WorkflowEngine.__new__(WorkflowEngine)
    default = {"calculation_type": "ORBITALS", "kmesh": "band_edges"}
    assert eng._get_default_d3_config("ORBITALS") == default
    assert eng._get_default_d3_config("ORBITALS2") == default
    planner = WorkflowPlanner.__new__(WorkflowPlanner)
    assert planner._get_basic_d3_config("ORBITALS") == default
    import d3_config
    assert d3_config.get_default_d3_config("ORBITALS") == default
    from mace.run_mace import _get_basic_d3_config_for_quick_start
    assert _get_basic_d3_config_for_quick_start("ORBITALS") == default


def test_planner_offers_orbitals_after_sp():
    from mace.workflow.planner import WorkflowPlanner
    planner = WorkflowPlanner.__new__(WorkflowPlanner)
    assert "ORBITALS" in planner._get_available_calc_types(["OPT", "SP"])
    assert planner._validate_numbered_calc_addition(["OPT", "SP"], "ORBITALS")
    assert not planner._validate_numbered_calc_addition(["FREQ"], "ORBITALS")
    assert planner.get_required_scripts("ORBITALS") == ["submit_prop.sh"]


# ------------------------------------------------------------------ SOC: the planner
def test_planner_refuses_orbitals_from_a_soc_sp():
    from mace.workflow import planner as P
    seq = ["OPT", "SP", "BAND", "ORBITALS"]
    cfgs = {"SP_2": {"calculation_type": "SP", "soc": True}}
    errors, warnings = P.soc_plan_problems(seq, cfgs)
    assert errors == [f"ORBITALS follows the SOC step SP: {SOC_REFUSAL}"]
    assert warnings == []
    # from a one-component SP: fine
    assert P.soc_plan_problems(seq, {"SP_2": {"calculation_type": "SP"}}) == ([], [])


# ------------------------------------------------------------------ SOC: workflow specs
def _spec(tmp_path, deck_text, sequence):
    from mace.workflow.spec import validate_spec
    d = tmp_path / "decks"
    d.mkdir(exist_ok=True)
    (d / "hi.d12").write_text(deck_text)
    raw = {"version": 1, "inputs": {"type": "d12", "files": ["decks/hi.d12"]}, "sequence": sequence}
    return validate_spec(raw, tmp_path)


def test_spec_refuses_orbitals_from_two_component_decks(tmp_path):
    from mace.workflow.spec import SpecError
    with pytest.raises(SpecError) as e:
        _spec(tmp_path, TWO_C_DECK, ["SP", "ORBITALS"])
    assert SOC_REFUSAL in str(e.value) and "hi.d12" in str(e.value)
    # BAND is not refused by the spec; one-component decks are fine
    assert _spec(tmp_path, TWO_C_DECK, ["SP", "BAND"])["sequence"] == ["SP", "BAND"]
    assert _spec(tmp_path, ONE_C_DECK, ["SP", "ORBITALS"])["sequence"] == ["SP", "ORBITALS"]


def test_spec_refuses_orbitals_after_a_soc_settings_file(tmp_path):
    from mace.workflow.spec import SpecError, validate_spec
    d = tmp_path / "decks"
    d.mkdir()
    (d / "x.d12").write_text(ONE_C_DECK.replace("END\nBASISSET", "OPTGEOM\nENDOPT\nEND\nBASISSET", 1))
    (tmp_path / "sp.json").write_text(json.dumps({"calculation_type": "SP", "soc": True}))
    raw = {"version": 1, "inputs": {"type": "d12", "files": ["decks/x.d12"]},
           "sequence": ["OPT", "SP", "ORBITALS"], "steps": {"SP_2": {"settings_file": "sp.json"}}}
    with pytest.raises(SpecError, match="SP_2 asks for SOC"):
        validate_spec(raw, tmp_path)
    (tmp_path / "sp.json").write_text(json.dumps({"calculation_type": "SP"}))
    assert validate_spec(raw, tmp_path)["sequence"] == ["OPT", "SP", "ORBITALS"]


# ------------------------------------------------------------------ SOC: the GUI
def test_setup_grays_out_orbitals_for_a_two_component_parent(tmp_path):
    from mace.gui import setup
    assert setup.parent_refusals(SCF / "hi2c.out") == {"ORBITALS": SOC_REFUSAL}
    assert setup.parent_refusals(SCF / "si.out") == {}
    info = setup.parent_info("opt2d3", "hi2c.out", [], SCF / "hi2c.out")
    assert info["refused"] == {"ORBITALS": SOC_REFUSAL}
    js = (REPO_ROOT / "mace" / "gui" / "static" / "js" / "setup" / "panel.js").read_text()
    assert 'aria-disabled="true" title="${esc(refused[t])}"' in js
    assert 'calc.getAttribute("aria-disabled") === "true"' in js


def test_builder_grays_out_orbitals_for_two_component_decks(tmp_path):
    from mace.gui import workflow_build as wb
    d = tmp_path / "d"
    d.mkdir()
    (d / "hi.d12").write_text(TWO_C_DECK)
    (d / "si.d12").write_text(ONE_C_DECK)
    inp = wb.Inputs("d12", d, [d / "hi.d12", d / "si.d12"], ["hi", "si"])
    refused = wb.input_refusals(inp)
    assert list(refused) == ["ORBITALS"] and refused["ORBITALS"].startswith("hi.d12: ")
    assert SOC_REFUSAL in refused["ORBITALS"]
    assert wb.input_refusals(wb.Inputs("d12", d, [d / "si.d12"], ["si"])) == {}
    js = (REPO_ROOT / "mace" / "gui" / "static" / "js" / "lazy" / "workflow_dialog.js").read_text()
    assert "disabled title=\"${esc(refused[baseOf(x)])}\"" in js


def test_batch_marks_two_component_parents_not_eligible(tmp_path):
    from mace.gui import batch
    root = tmp_path
    folder = root / "runs"
    folder.mkdir()
    for name in ("hi2c", "si"):
        shutil.copy(SCF / f"{name}.out", folder / f"{name}.out")
        (folder / f"{name}.f9").write_text("wavefunction placeholder\n")
    rows = {r["rel"]: r for r in batch.scan(root, folder, "followup", "ORBITALS", False)["rows"]}
    assert rows["runs/hi2c.out"]["eligible"] is False
    assert rows["runs/hi2c.out"]["why_not"] == SOC_REFUSAL
    assert rows["runs/si.out"]["eligible"] is True, rows["runs/si.out"]
    # BAND from the same 2c run is not refused here (the manual lists it for 2c)
    rows = {r["rel"]: r for r in batch.scan(root, folder, "followup", "BAND", False)["rows"]}
    assert rows["runs/hi2c.out"]["why_not"] != SOC_REFUSAL


# ------------------------------------------------------------------ SOC and records: preflight
def _orbitals_deck_dir(tmp_path, parent, deck_text=None):
    d = tmp_path / "pf"
    d.mkdir(parents=True)
    shutil.copy(SCF / f"{parent}.out", d / f"{parent}.out")
    deck = d / f"{parent}_orbitals.d3"
    deck.write_text(deck_text or f"NEWK\n1 0\nORBITALS\n{parent}_orbitals_orb\n0\n0\nEND\n")
    return d, deck


def test_preflight_fails_an_orbitals_deck_of_a_two_component_run(tmp_path):
    from mace.submission import preflight as pf
    d, deck = _orbitals_deck_dir(tmp_path, "hi2c")
    r = pf.preflight_deck(deck, runner=None)
    assert r.status == pf.FAIL and SOC_REFUSAL in r.reason and "hi2c.out" in r.reason
    # a directory picks the ORBITALS deck up, and only that .d3
    (d / "other_band.d3").write_text("BAND\nx\n1 4 100 1 10 1 0\n0 0 0 2 0 0\nEND\n")
    assert pf.collect_decks([str(d)]) == [deck]
    assert pf.main([str(d), "--static-only", "--no-banner"]) == 1


def test_preflight_fails_icar_and_iloc_on_one_line_and_iloc_1(tmp_path):
    from mace.submission import preflight as pf
    _, deck = _orbitals_deck_dir(tmp_path / "a", "si", "NEWK\n2 12\n1 0\nORBITALS\nsi_orbitals_orb\n1 0\nEND\n")
    r = pf.preflight_deck(deck, runner=None)
    assert r.status == pf.FAIL and "FORMAT ERROR" in r.reason
    _, deck = _orbitals_deck_dir(tmp_path / "b", "si", "NEWK\n2 12\n1 0\nORBITALS\nsi_orbitals_orb\n1\n1\nEND\n")
    r = pf.preflight_deck(deck, runner=None)
    assert r.status == pf.FAIL and "Wannier90" in r.reason


def test_preflight_passes_a_good_orbitals_deck_without_running_crystal(tmp_path):
    from mace.submission import preflight as pf
    _, deck = _orbitals_deck_dir(tmp_path, "si", "NEWK\n2 12\n1 0\nORBITALS\nsi_orbitals_orb\n1\n0\nEND\n")

    def runner(_):
        raise AssertionError("CRYSTAL is not run for a .d3")
    r = pf.preflight_deck(deck, runner=runner)
    assert r.status == pf.SKIPPED and "si.out is one-component" in r.reason
