"""One workflow = ONE workflow id.

Real incident (phase-3 QA, 2026-07-09): quick-start plans are saved without
a workflow_id. execute_workflow_plan and recreate_workflow_scripts each
minted their own timestamp id — one second apart (....001354 vs ....001355) —
so the generic workflow_scripts/ templates baked a shadow id into every job
script. Job callbacks then resolved .mace_context_<shadow>/materials.db:
records split across two context databases, and follow-up steps opened
fresh cwd-local DBs on top of that.

Contract: the id is resolved once, written back into the plan dict, and
every script_config receives that same id.
"""
from mace.workflow.executor import WorkflowExecutor


def test_recreate_scripts_single_sources_workflow_id(tmp_path):
    ex = WorkflowExecutor.__new__(WorkflowExecutor)
    ex.work_dir = tmp_path
    sc = {"step_specific_name": "x.sh",
          "source_script": str(tmp_path / "missing.sh")}
    plan = {"workflow_id": None,  # how quick-start plans actually arrive
            "step_configurations": {
                "OPT_1": {"slurm_config": {"scripts": {"nope.sh": sc}}}}}

    ex.recreate_workflow_scripts(plan)

    assert plan["workflow_id"], "resolved id must be written back to the plan"
    assert plan["workflow_id"].startswith("workflow_")
    assert sc["workflow_id"] == plan["workflow_id"], (
        "script templates must carry the SAME id as the plan/context")


def test_executed_plan_is_discoverable_by_callbacks(tmp_path):
    """Real incident (phase-4 QA): `mace workflow --execute <plan.json>` with
    a plan file living in ANOTHER directory never persisted the plan into the
    execution tree — every callback's plan lookup failed and progression
    silently fell back to template defaults (SP kept the parent functional
    instead of the plan's B3LYP-D3, BAND got 10000 points instead of 1000,
    DOSS got band indices instead of the energy window, and FREQ /
    CHARGE+POTENTIAL / OPT2 were never generated at all)."""
    from mace.workflow.engine import WorkflowEngine

    ex = WorkflowExecutor.__new__(WorkflowExecutor)
    ex.work_dir = tmp_path
    ex.configs_dir = tmp_path / "workflow_configs"
    plan = {"workflow_id": "workflow_20260702_193809",
            "workflow_sequence": ["OPT", "SP", "BAND", "DOSS", "FREQ",
                                  "CHARGE+POTENTIAL", "OPT2"],
            "step_configurations": {"SP_2": {"method_modifications":
                                             {"functional": "B3LYP-D3"}}}}

    ex._persist_plan_for_callbacks(plan, plan["workflow_id"])

    dest = ex.configs_dir / "workflow_plan_20260702_193809.json"
    assert dest.exists(), "plan must be persisted into the execution tree"

    eng = WorkflowEngine.__new__(WorkflowEngine)
    eng.base_work_dir = tmp_path
    loaded = eng._load_workflow_plan("workflow_20260702_193809")
    assert loaded is not None, "engine must find the persisted plan"
    assert loaded["workflow_sequence"][4] == "FREQ"
    assert loaded["step_configurations"]["SP_2"]["method_modifications"][
        "functional"] == "B3LYP-D3"
