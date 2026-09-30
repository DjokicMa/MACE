"""Asking the workflow planner for a two-component SOC single point.

CRYSTAL23 runs spin-orbit coupling as a 2c-SCF single point only: no
optimisation or frequency run (manual p. 166), and from its wavefunction only
the properties of sec. 6.4 (p. 183) - no BOLTZTRA, no POT3/POTC. The planner
offers SOC on SP steps (a new level 4; levels 0-3 keep their numbers and
configs), refuses it on any other step of a plan, and warns about steps that
would start from a SOC run. The engine hands "soc" to the deck writer.
"""
import builtins
import json

import pytest

from mace.workflow import planner as P
from mace.workflow.engine import WorkflowEngine


def _sp_config(monkeypatch, answer):
    planner = P.WorkflowPlanner.__new__(P.WorkflowPlanner)
    answers = iter([answer])

    def one_answer(prompt=""):
        try:
            return next(answers)
        except StopIteration:   # asked again: the answer was not accepted
            raise EOFError(f"{answer!r} was not accepted")

    monkeypatch.setattr(builtins, "input", one_answer)
    return planner.configure_single_point_step("SP2", 3)


def test_level_4_asks_for_a_soc_single_point(monkeypatch):
    assert _sp_config(monkeypatch, "4") == {
        "calculation_type": "SP", "source": "CRYSTALOptToD12.py",
        "customization_level": 4, "inherit_settings": True, "soc": True}


def test_level_0_is_unchanged(monkeypatch):
    assert _sp_config(monkeypatch, "0") == {
        "calculation_type": "SP", "source": "CRYSTALOptToD12.py",
        "customization_level": 0, "inherit_settings": True}


def test_no_soc_plan_has_no_problems():
    seq = ["OPT", "SP", "BAND", "DOSS", "FREQ", "TRANSPORT"]
    cfgs = {"SP_2": {"calculation_type": "SP", "inherit_settings": True}}
    assert P.soc_plan_problems(seq, cfgs) == ([], [])


def test_soc_on_anything_but_sp_is_an_error():
    seq = ["OPT", "SP", "OPT2", "FREQ"]
    cfgs = {"OPT2_3": {"calculation_type": "OPT", "soc": True},
            "FREQ_4": {"calculation_type": "FREQ", "soc": True},
            "SP_2": {"calculation_type": "SP", "soc": True}}
    errors, _ = P.soc_plan_problems(seq, cfgs)
    assert len(errors) == 2
    assert any(e.startswith("OPT2_3:") and "p. 166" in e for e in errors)
    assert any(e.startswith("FREQ_4:") for e in errors)


def test_steps_that_cannot_follow_a_soc_sp_are_warned_about():
    seq = ["OPT", "SP", "TRANSPORT", "SP2", "BAND", "SP3", "FREQ"]
    cfgs = {"SP_2": {"calculation_type": "SP", "soc": True},
            "SP3_6": {"calculation_type": "SP", "soc": True}}
    errors, warnings = P.soc_plan_problems(seq, cfgs)
    assert errors == []
    assert len(warnings) == 2
    assert warnings[0].startswith("TRANSPORT follows the SOC step SP")
    assert warnings[1].startswith("FREQ follows the SOC step SP3")


def test_a_plan_with_soc_on_an_opt_step_is_not_executed(tmp_path, capsys):
    plan = {"workflow_sequence": ["OPT", "SP", "OPT2"],
            "step_configurations": {"OPT2_3": {"calculation_type": "OPT", "soc": True}},
            "input_files": {"cif": [], "d12": []}}
    plan_file = tmp_path / "workflow_plan_x.json"
    plan_file.write_text(json.dumps(plan))
    planner = P.WorkflowPlanner.__new__(P.WorkflowPlanner)
    planner.execute_workflow_plan(plan_file)
    err = capsys.readouterr().err
    assert "SOC is only for single-point (SP) steps" in err and "Not executing" in err


WF_ID = "workflow_20260930_120000"


@pytest.fixture
def engine(tmp_path):
    eng = WorkflowEngine.__new__(WorkflowEngine)
    eng.base_work_dir = tmp_path
    cfg_dir = tmp_path / "workflow_configs"
    cfg_dir.mkdir()
    plan = {"workflow_id": WF_ID, "workflow_sequence": ["OPT", "SP", "SP2", "BAND"],
            "step_configurations": {
                "SP_2": {"calculation_type": "SP", "inherit_settings": True, "soc": True},
                "SP2_3": {"calculation_type": "SP", "inherit_settings": True, "soc": False}}}
    (cfg_dir / f"workflow_plan_{WF_ID.replace('workflow_', '')}.json").write_text(json.dumps(plan))
    return eng


def test_the_engine_passes_soc_to_the_deck_writer(engine):
    assert engine._build_numbered_calc_config(WF_ID, "SP", "SP", None) == {
        "calculation_type": "SP", "soc": True}
    assert engine._build_numbered_calc_config(WF_ID, "SP", "SP", "PBE0") == {
        "calculation_type": "SP", "functional": "PBE0", "soc": True}
    # "soc": false is passed on too: a scalar step after a SOC parent
    assert engine._build_numbered_calc_config(WF_ID, "SP2", "SP", None) == {
        "calculation_type": "SP", "soc": False}
    # a step with no soc key: as before
    assert engine._build_numbered_calc_config(WF_ID, "BAND", "BAND", None) is None


def test_an_opt_step_offers_no_soc(monkeypatch, capsys):
    """SOC is a choice only where CRYSTAL allows it: not for an optimisation
    (manual p. 166). The OPT menu has no SOC level, and "4" is not taken."""
    planner = P.WorkflowPlanner.__new__(P.WorkflowPlanner)
    answers = iter(["4"])

    def one_answer(prompt=""):
        try:
            return next(answers)
        except StopIteration:
            raise EOFError("asked again")

    monkeypatch.setattr(builtins, "input", one_answer)
    with pytest.raises(EOFError):
        planner.configure_optimization_step("OPT2", 3)
    menu = capsys.readouterr().out
    assert "SOC" not in menu and "spin-orbit" not in menu.lower()


def test_the_sp_menu_offers_soc(monkeypatch, capsys):
    _sp_config(monkeypatch, "0")
    menu = capsys.readouterr().out
    assert "4: Spin-orbit coupling (two-component SOC single point)" in menu
