"""Functional names in workflow plans saved by MACE 1.1.1.

The 1.1.1 planner stored its menus' prose spellings ("HSEsol-3C", "HF-3C",
"M06-2X", "HF", ...) and offered "-D3" on any non-3c choice. 1.1.1 never read
an SP step's custom_functional, so the step ran with the parent's method;
since 1.1.2 the engine writes the name into the deck verbatim and CRYSTAL23
rejects it. A plan is now normalised when it is loaded: legacy spellings
become the CRYSTAL23 keyword, a -D3 suffix CRYSTAL23 has no parameters for is
dropped with a warning, an unknown name stops that step before a deck is
written, and a valid name is never changed.
"""
import json

import pytest

import test_numbered_calc_keeps_parent_method as base
from mace.workflow.plan_functionals import normalize_plan_functionals
from mace.workflow.planner import WorkflowPlanner

PBE0 = base.DECK.format(ham="DFT\nSPIN\nPBE0\nENDDFT")


def _sp(mods):
    return {"SP_2": {"calculation_type": "SP", "inherit_settings": False,
                     "method_modifications": mods,
                     "basis_modifications": {"inherit_basis": True}}}


def _cfg(tmp_path, mods):
    eng = base._engine(tmp_path, PBE0, "dummy\n", _sp(mods))
    return base._numbered(eng, "SP")["config"]


@pytest.mark.parametrize("legacy,keyword", [
    ("HSEsol-3C", "HSESOL3C"), ("HF-3C", "HF3C"), ("PBEh-3C", "PBEH3C"),
    ("B97-3C", "B973C"), ("HSE-3C", "HSE3C"), ("PBEsol0-3C", "PBESOL03C"),
    ("M06-2X", "M062X"), ("M06-L", "M06L"), ("M06-HF", "M06HF"), ("HF", "RHF"),
])
def test_legacy_custom_functional_is_written_as_the_crystal23_keyword(tmp_path, legacy, keyword):
    assert _cfg(tmp_path, {"custom_functional": legacy})["method_modifications"] == {
        "functional": keyword}


@pytest.mark.parametrize("legacy,written", [
    ("M06-L-D3", "M06L"), ("CAM-B3LYP-D3", "CAM-B3LYP"), ("r2SCAN-D3", "r2SCAN"),
    ("PBESOL-D3", "PBESOL"), ("HSEsol-3C-D3", "HSESOL3C"),
])
def test_unparametrised_d3_suffix_is_dropped_with_a_warning(tmp_path, capsys, legacy, written):
    cfg = _cfg(tmp_path, {"custom_functional": legacy})
    assert cfg["method_modifications"] == {"functional": written}
    assert "-D3 suffix is dropped" in capsys.readouterr().out


@pytest.mark.parametrize("valid", ["PBE0", "PBE0-D3", "LC-wPBE-D3", "wB97X-D3", "UHF",
                                   "HSESOL3C", "mPW1PW91-D3", "VBH", "hse06"])
def test_valid_functional_is_never_changed(tmp_path, valid):
    assert _cfg(tmp_path, {"custom_functional": valid})["method_modifications"] == {
        "functional": valid}


def test_legacy_new_functional_is_normalised(tmp_path):
    # the 1.1.1 SP menu's option 6 stored new_functional "M06-2X"
    assert _cfg(tmp_path, {"new_functional": "M06-2X"})["functional"] == "M062X"


def test_unknown_functional_stops_the_step_before_a_deck_is_written(tmp_path, capsys):
    eng = base._engine(tmp_path, PBE0, "dummy\n", _sp({"custom_functional": "B2PLYP"}))
    seen = base._capture(eng)
    assert eng.generate_numbered_calculation("c1", "SP") is None
    assert "args" not in seen  # CRYSTALOptToD12 was never run
    assert "B2PLYP" in capsys.readouterr().out


def test_planner_saves_plans_with_crystal23_spellings(tmp_path):
    planner = WorkflowPlanner(work_dir=tmp_path)
    plan = {"workflow_id": "workflow_20260101_000000",
            "step_configurations": _sp({"custom_functional": "HSEsol-3C"})}
    saved = json.loads(planner.save_workflow_plan(plan).read_text())
    assert saved["step_configurations"]["SP_2"]["method_modifications"][
        "custom_functional"] == "HSESOL3C"


def test_normalise_plan_reports_unknown_and_leaves_it_as_written():
    plan = {"step_configurations": _sp({"custom_functional": "B2PLYP"})}
    messages, errors = normalize_plan_functionals(plan)
    assert errors and "B2PLYP" in errors[0]
    assert plan["step_configurations"]["SP_2"]["method_modifications"][
        "custom_functional"] == "B2PLYP"


# Real deck, real script (corpus-gated)
def test_real_legacy_3c_plan_writes_a_deck_crystal23_accepts(tmp_path):
    name, body = base_real(tmp_path, {"custom_functional": "HSEsol-3C"})
    assert "HSESOL3C" in body and "HSEsol-3C" not in body
    assert body[body.index("BASISSET") + 1] == "SOLDEF2MSVP"


def base_real(tmp_path, mods):
    base._need(base.DIA)
    d12 = base.DIA.with_suffix(".d12").read_text()
    out = base.DIA.with_suffix(".out").read_text()
    eng = base._engine(tmp_path, d12, out, _sp(mods))
    name, deck = base._only_deck(base._numbered(eng, "SP", True))
    return name, [line.strip() for line in deck.splitlines()]
