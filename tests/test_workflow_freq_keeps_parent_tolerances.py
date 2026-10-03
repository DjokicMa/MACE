"""A workflow FREQ step keeps a FREQ parent's SCF tolerances.

opt2d12 gives a FREQ deck Very tight SCF tolerances (9 9 9 11 38 / 11) when
nothing names any, except that a parent which is itself a FREQ calculation
keeps its own (d12_constants.freq_default_tolerances). The planner's Basic and
Advanced FREQ steps, its Expert fallback and the quick-start / --progress FREQ
step wrote Very tight into the plan as if the user had chosen it, so a FREQ
step after a FREQ parent replaced the parent's tolerances. Those defaults now
name no tolerances and opt2d12 applies its own rule: an OPT parent still gets
Very tight, as the manual opt2d12 FREQ route gives it.

The deck tests run the real engine route (WorkflowEngine ->
CRYSTALOptToD12.py --config-file) on the bundled PbTiO3 parent.
"""
import pytest

import test_numbered_calc_keeps_parent_method as base
from conftest import REPO_ROOT
from mace.workflow.planner import WorkflowPlanner

PARENT = REPO_ROOT / "tests" / "data" / "opt_restart" / "tqb_pto"
VERY_TIGHT = ("9 9 9 11 38", "11")
TIGHT = ("8 8 8 9 24", "9")


def _planner_step(route, tmp_path, monkeypatch):
    if route == "quick_start":
        from mace.run_mace import create_quick_workflow_plan

        class _Args:
            max_jobs = 10
        plan = create_quick_workflow_plan(tmp_path, [], "d12", ["OPT", "FREQ"], _Args())
        return plan["step_configurations"]["FREQ_2"]
    planner = WorkflowPlanner(work_dir=tmp_path / "plan")
    if route == "expert_fallback":
        monkeypatch.setattr(planner, "_copy_required_scripts_for_expert_mode", lambda: None)
        monkeypatch.setattr(planner, "_run_interactive_crystal_opt_config", lambda *a, **k: None)
        replies = iter(["3", "2"])                   # Expert, batch uniform
    else:
        replies = iter([{"basic": "1", "advanced": "2"}[route]])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(replies, ""))
    return planner.configure_frequency_step("FREQ", 3)


ROUTES = ["basic", "advanced", "expert_fallback", "quick_start"]


def _parent_deck(freq_parent):
    lines = PARENT.with_suffix(".d12").read_text().splitlines()
    if freq_parent:
        i, j = lines.index("OPTGEOM"), lines.index("ENDOPT")
        lines[i:j + 1] = ["FREQCALC", "ENDFREQ"]
        lines[lines.index("TOLINTEG") + 1] = TIGHT[0]
        lines[lines.index("TOLDEE") + 1] = TIGHT[1]
    return "\n".join(lines) + "\n"


def _scf(deck):
    body = deck.splitlines()
    return (body[body.index("TOLINTEG") + 1].strip(), body[body.index("TOLDEE") + 1].strip())


@pytest.mark.parametrize("route", ROUTES)
def test_default_freq_step_names_no_tolerances(tmp_path, monkeypatch, route):
    step = _planner_step(route, tmp_path, monkeypatch)
    assert "custom_tolerances" not in step.get("frequency_settings", {}), step
    assert "tolerance_modifications" not in step and "custom_tolerances" not in step


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("freq_parent,expected", [(True, TIGHT), (False, VERY_TIGHT)])
def test_real_freq_deck_tolerances(tmp_path, monkeypatch, route, freq_parent, expected):
    pytest.importorskip("numpy")
    step = _planner_step(route, tmp_path, monkeypatch)
    eng = base._engine(tmp_path, _parent_deck(freq_parent),
                       PARENT.with_suffix(".out").read_text(), {"FREQ_3": step})
    name, deck = base._only_deck(base._numbered(eng, "FREQ", True))
    assert "FREQCALC" in deck
    assert _scf(deck) == expected


def test_explicit_plan_tolerances_still_win_over_a_freq_parent(tmp_path):
    pytest.importorskip("numpy")
    step = {"calculation_type": "FREQ", "inherit_base_settings": True,
            "frequency_settings": {"mode": "GAMMA",
                                   "custom_tolerances": {"TOLINTEG": "10 10 10 12 40",
                                                         "TOLDEE": 12}}}
    eng = base._engine(tmp_path, _parent_deck(True),
                       PARENT.with_suffix(".out").read_text(), {"FREQ_3": step})
    name, deck = base._only_deck(base._numbered(eng, "FREQ", True))
    assert _scf(deck) == ("10 10 10 12 40", "12")
