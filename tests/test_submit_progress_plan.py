"""`mace submit --progress` must produce a REAL plan, not template guesswork.

The opt-in counterpart to test_manual_no_autoprogression.py. A flag that
merely re-enabled the engine's built-in defaults would regenerate the bug the
QA campaign fixed twice over (planless callbacks -> template defaults: wrong
SP functional, unrequested BAND/DOSS). So `--progress` writes a workflow plan
first and stamps every submission with its workflow id; progression then runs
the same plan-driven path a `mace workflow` run uses.
"""
import json
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT))

from mace.workflow.manual_progress import (  # noqa: E402
    available_templates, position_in_sequence)


def test_templates_include_the_documented_ones():
    names = available_templates()
    for expected in ("basic_opt", "opt_sp", "opt_sp_freq", "full_electronic",
                     "complete", "transport_analysis", "charge_analysis",
                     "combined_analysis", "double_opt"):
        assert expected in names, f"{expected} missing from --progress templates"


def test_position_in_sequence_handles_mid_sequence_start():
    """A hand-made deck can enter the plan anywhere — submitting an SP after a
    manual OPT is the normal case, and its step number must not be assumed 1."""
    seq = ["OPT", "SP", "BAND", "DOSS"]
    assert position_in_sequence("OPT", seq) == 1
    assert position_in_sequence("SP", seq) == 2
    assert position_in_sequence("DOSS", seq) == 4
    assert position_in_sequence("FREQ", seq) is None


def test_position_in_sequence_matches_numbered_types():
    assert position_in_sequence("OPT2", ["OPT", "OPT2", "SP"]) == 2
    # a bare OPT falls back to the first OPT-family step
    assert position_in_sequence("OPT3", ["OPT", "SP"]) == 1


def test_unknown_template_is_rejected(tmp_path):
    from mace.workflow.manual_progress import build_progress_plan

    with pytest.raises(ValueError, match="Unknown workflow template"):
        build_progress_plan(work_dir=tmp_path, db_path="materials.db",
                            input_files=[tmp_path / "mat.d12"],
                            progress="not_a_template")


def test_template_plan_is_written_and_findable(tmp_path, monkeypatch):
    """The plan must land where the completion callback looks for it:
    ``workflow_configs/workflow_plan_<id-minus-prefix>.json`` under the work
    dir, carrying the SAME workflow_id the submissions are stamped with."""
    from mace.workflow.manual_progress import build_progress_plan

    deck = tmp_path / "mat_opt.d12"
    deck.write_text("TESTGEOM\nEND\n")
    monkeypatch.chdir(tmp_path)

    plan_file, workflow_id, sequence = build_progress_plan(
        work_dir=tmp_path, db_path="materials.db", input_files=[deck],
        progress="full_electronic")

    assert sequence == ["OPT", "SP", "BAND", "DOSS"]
    assert plan_file.exists()
    assert plan_file.parent == tmp_path / "workflow_configs"
    # The engine derives the filename from the id; a mismatch means the
    # callback silently finds no plan and stops progressing.
    assert plan_file.name == f"workflow_plan_{workflow_id.replace('workflow_', '')}.json"

    plan = json.loads(plan_file.read_text())
    assert plan["workflow_id"] == workflow_id
    assert plan["workflow_sequence"] == sequence
    assert plan["step_configurations"], "plan must carry per-step settings"
    # Existing decks are the starting point: no CIF conversion in this path.
    assert plan.get("cif_conversion_config") is None
    assert plan["input_files"]["d12"] == [str(deck)]


def test_interactive_plan_uses_the_planners_own_prompts(tmp_path, monkeypatch):
    """`--progress interactive` must route through the SAME planner prompts
    `mace workflow --interactive` uses (that is the whole point of the mode:
    the follow-up steps get the user's settings, not template defaults), and
    must tell them the first step is the deck they already built."""
    from mace.workflow import manual_progress as mp
    from mace.workflow.planner import WorkflowPlanner

    deck = tmp_path / "mat_opt.d12"
    deck.write_text("TESTGEOM\nEND\n")
    monkeypatch.chdir(tmp_path)

    seen = {}

    def fake_steps(self, sequence, has_cifs):
        seen["sequence"] = list(sequence)
        seen["has_cifs"] = has_cifs
        return {f"{c}_{i + 1}": {"source": "existing_d12" if i == 0 else "planner"}
                for i, c in enumerate(sequence)}

    monkeypatch.setattr(WorkflowPlanner, "plan_workflow_sequence",
                        lambda self: ["OPT", "SP", "FREQ"])
    monkeypatch.setattr(WorkflowPlanner, "configure_workflow_steps", fake_steps)
    monkeypatch.setattr(WorkflowPlanner, "configure_queue_management",
                        lambda self: dict(mp.DEFAULT_QUEUE_CONFIG))
    monkeypatch.setattr(WorkflowPlanner, "copy_and_customize_scripts",
                        lambda self, sc, wid, qc: seen.setdefault("scripts", wid))

    plan_file, workflow_id, sequence = mp.build_progress_plan(
        work_dir=tmp_path, db_path="materials.db", input_files=[deck],
        progress="interactive")

    assert sequence == ["OPT", "SP", "FREQ"]
    assert seen["sequence"] == sequence
    # has_cifs=False is what makes step 1 "use the existing D12 as-is".
    assert seen["has_cifs"] is False
    assert seen["scripts"] == workflow_id, "SLURM scripts must be built for this id"

    plan = json.loads(plan_file.read_text())
    assert plan["mode"] == "manual_progress"
    assert plan["workflow_id"] == workflow_id
    assert plan["step_configurations"]["OPT_1"]["source"] == "existing_d12"
    assert plan["cif_conversion_config"] is None


def test_engine_loads_the_written_plan(tmp_path, monkeypatch):
    """End of the contract: the engine must resolve the plan we just wrote
    from a calculation stamped with its workflow_id."""
    from mace.workflow.manual_progress import build_progress_plan
    from mace.workflow import engine as engine_mod

    deck = tmp_path / "mat_sp.d12"
    deck.write_text("TESTGEOM\nEND\n")
    monkeypatch.chdir(tmp_path)

    plan_file, workflow_id, sequence = build_progress_plan(
        work_dir=tmp_path, db_path="materials.db", input_files=[deck],
        progress="full_electronic")

    eng = engine_mod.WorkflowEngine.__new__(engine_mod.WorkflowEngine)
    eng.base_work_dir = tmp_path
    assert eng.get_workflow_sequence(workflow_id) == sequence
    assert eng._get_plan_step_config(workflow_id, "BAND"), \
        "BAND step settings must be reachable for progression"
