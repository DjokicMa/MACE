"""The CIF-conversion copy of a plan is never taken for the plan.

A workflow started from CIFs saves ``workflow_plan_workflow_<id>_updated.json``
(the plan with the generated decks) next to ``workflow_plan_<id>.json``. Both
match ``workflow_plan_*.json``: the progress report then found two plans and
showed no sequence, and the template store took the copy (it sorts last).
"""
import json
import sys

from conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT))

WID = "workflow_20991231_120000"
PLAN = {"workflow_id": WID, "created": "2099-12-31T12:00:00", "input_type": "cif",
        "workflow_sequence": ["OPT", "SP", "BAND"], "step_configurations": {},
        "input_files": {"cif": ["/x/a.cif"], "d12": []}}


def _configs(tmp_path):
    cfg = tmp_path / "workflow_configs"
    cfg.mkdir()
    (cfg / f"workflow_plan_{WID[len('workflow_'):]}.json").write_text(json.dumps(PLAN))
    updated = dict(PLAN, workflow_sequence=["UPDATED"], generated_d12s=["/x/a.d12"])
    (cfg / f"workflow_plan_{WID}_updated.json").write_text(json.dumps(updated))
    return cfg


def test_progress_report_reads_the_plan(tmp_path, monkeypatch):
    from mace.database.analysis.workflow_progress import WorkflowProgress
    for var in ("MACE_WORKFLOW_ID", "MACE_CONTEXT_DIR", "MACE_ISOLATION_MODE"):
        monkeypatch.delenv(var, raising=False)
    _configs(tmp_path)
    monkeypatch.chdir(tmp_path)
    analyzer = WorkflowProgress.__new__(WorkflowProgress)
    assert analyzer._load_planned_sequence() == (WID, ["OPT", "SP", "BAND"])


def test_template_store_takes_the_plan(tmp_path, monkeypatch, capsys):
    from mace.queue.manager import EnhancedCrystalQueueManager
    _configs(tmp_path)
    monkeypatch.chdir(tmp_path)
    mgr = EnhancedCrystalQueueManager.__new__(EnhancedCrystalQueueManager)
    mgr.enable_tracking = True
    mgr.db = None       # stops it after the file is chosen
    mgr.store_workflow_configuration_as_template()
    out = capsys.readouterr().out
    assert f"template: workflow_plan_{WID[len('workflow_'):]}.json" in out, out
