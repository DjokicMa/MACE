"""A workflow's context is archived when its last step finishes.

The planner's "Archive workflow context after completion" archived the
context right after the first wave was submitted: the running jobs'
MACE_CONTEXT_DIR then pointed at a folder that was gone, their callbacks
fell back to a database in the job folder, and the workflow stopped. The
archive (or export-and-delete) now happens in the job callback that finds
every step of the workflow finished.

Run through the real CLI and the real job callback, with the fake SLURM of
tests/fake_slurm.py and real CRYSTAL outputs (corpus).
"""
import re
import shutil

import pytest

from array_helpers import callback, rows
from conftest import find_data
from fake_slurm import FakeSlurm
from test_array_default_path_golden import _env, _mace
from test_workflow_isolation_scripts import _ok, _seed_plan, _write_plan

WID = "workflow_20991231_000010"


def _finish(fake, env, db, row, out, f9):
    work = shutil.os.path.dirname(row["input_file"])
    stem = shutil.os.path.splitext(shutil.os.path.basename(row["input_file"]))[0]
    shutil.copy(out, f"{work}/{stem}.out")
    shutil.copy(f9, f"{work}/{stem}.f9")
    script = open(f"{work}/{stem}.sh").read()
    exports = dict(re.findall(r'^export (MACE_WORKFLOW_ID|MACE_CONTEXT_DIR|MACE_ISOLATION_MODE)'
                              r'="([^"]*)"', script, re.M))
    fake.set_record(row["slurm_job_id"], state="COMPLETED")
    return _ok(callback(env, work, db, mode="completion",
                        extra_env=dict(exports, SLURM_JOB_ID=row["slurm_job_id"])))


@pytest.mark.parametrize("action", ["archive", "export_and_delete"])
def test_context_is_archived_only_after_the_last_step(tmp_path, action):
    opt_out = find_data("OPT/1_dia_opt_rev1.out")
    opt_f9 = find_data("OPT/1_dia_opt_rev1.f9")
    deck = find_data("OPT/1_dia_opt_rev1.d12")
    sp_out = find_data("SP/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized.out")
    sp_f9 = find_data("SP/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized.f9")
    fake = FakeSlurm(tmp_path)
    env = _env(tmp_path, fake)
    template = _seed_plan(tmp_path, env, deck)          # opt_sp

    root = tmp_path / "wf"
    _write_plan(root, template, deck, "one", WID, 16, post_completion_action=action)
    _ok(_mace(root, env, "workflow", "--execute", "plan_one.json"))
    context = root / f".mace_context_{WID}"
    assert context.is_dir(), "the context must stay while the workflow's jobs run"
    assert not (root / "archived_workflows").exists()
    db = context / "materials.db"

    # step 1 finishes: the workflow goes on, from the same context
    (opt,) = [r for r in rows(db).values() if r["calc_type"] == "OPT"]
    before = len(fake.calls("sbatch"))
    p = _finish(fake, env, db, opt, opt_out, opt_f9)
    assert context.is_dir(), p.stdout[-4000:]
    sp_rows = [r for r in rows(db).values() if r["calc_type"].startswith("SP")]
    assert len(sp_rows) == 1, p.stdout[-4000:]
    (sb,) = fake.calls("sbatch")[before:]
    assert f'export MACE_CONTEXT_DIR="{context}"' in sb["script_text"]

    # step 2 (the last) finishes: now the context is archived / exported
    p = _finish(fake, env, db, sp_rows[0], sp_out, sp_f9)
    assert not context.exists(), p.stdout[-4000:]
    if action == "archive":
        (archived,) = (root / "archived_workflows").iterdir()
        assert archived.name.startswith(WID)
        done = rows(archived / "materials.db").values()
    else:
        export = root / "workflow_results" / WID
        done = rows(export / f"{WID}_materials.db").values()
    assert sorted(r["status"] for r in done) == ["completed", "completed"]
    # no stray database was made in a job folder
    assert not [d for d in (root / "workflow_outputs").rglob("materials.db")]


def _context_with(tmp_path, statuses, action="archive"):
    import json
    from mace.database.materials import MaterialDatabase
    root = tmp_path / "wf"
    (root / "workflow_configs").mkdir(parents=True)
    (root / "workflow_configs" / f"workflow_plan_{WID[len('workflow_'):]}.json").write_text(
        json.dumps({"workflow_id": WID, "isolation_mode": "isolated",
                    "post_completion_action": action, "workflow_sequence": ["OPT", "SP"]}))
    context = root / f".mace_context_{WID}"
    context.mkdir()
    db = MaterialDatabase(str(context / "materials.db"))
    db.create_material("m1", "C2")
    for calc_type, status in statuses:
        cid = db.create_calculation("m1", calc_type,
                                    settings={"workflow_id": WID, "workflow_processed": True})
        db.update_calculation_status(cid, status)
    return root, context


@pytest.mark.parametrize("statuses, archived", [
    ([("OPT", "completed"), ("SP", "running")], False),
    ([("OPT", "completed"), ("SP", "submitted")], False),
    ([("OPT", "completed"), ("SP", "failed")], True),     # the end of this workflow
    ([("OPT", "resubmitted"), ("OPT", "failed")], True),
    ([("OPT", "resubmitted"), ("OPT", "pending")], False),
])
def test_archived_only_when_every_step_has_finished(tmp_path, monkeypatch, statuses, archived):
    from mace.workflow.engine import WorkflowEngine
    for k in ("MACE_WORKFLOW_ID", "MACE_CONTEXT_DIR", "MACE_ISOLATION_MODE"):
        monkeypatch.delenv(k, raising=False)
    root, context = _context_with(tmp_path, statuses)
    monkeypatch.chdir(root)
    engine = WorkflowEngine(str(context / "materials.db"), str(root))
    result = engine.finish_workflow_if_done(context / "materials.db")
    assert (result is not None) == archived
    assert context.exists() != archived


def test_completed_step_not_yet_progressed_keeps_the_context(tmp_path, monkeypatch):
    import json
    import sqlite3
    from mace.workflow.engine import WorkflowEngine
    for k in ("MACE_WORKFLOW_ID", "MACE_CONTEXT_DIR", "MACE_ISOLATION_MODE"):
        monkeypatch.delenv(k, raising=False)
    root, context = _context_with(tmp_path, [("OPT", "completed")])
    con = sqlite3.connect(str(context / "materials.db"))
    con.execute("update calculations set settings_json=?",
                (json.dumps({"workflow_id": WID}),))
    con.commit(); con.close()
    monkeypatch.chdir(root)
    engine = WorkflowEngine(str(context / "materials.db"), str(root))
    assert engine.finish_workflow_if_done(context / "materials.db") is None
    assert context.is_dir()
