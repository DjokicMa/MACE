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
