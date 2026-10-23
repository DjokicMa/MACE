"""A workflow's chosen cap of 200 jobs counts (1.1.6).

Every plan up to 1.1.5 stored max_concurrent_jobs 200 whether the user chose it or not, so a cap of 200
reads as "unset" (the workflow then shares the user-wide queue budget). A plan since 1.1.6 stores the
cap only when somebody chose it and says so (execution_settings.max_jobs_explicit): a chosen 200 is kept,
and a 1.1.5 plan's 200 still reads as unset.
"""
import json
import sys

import pytest

from conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT))

from mace.queue import budget  # noqa: E402


def test_normalize_cap_keeps_a_chosen_200_only():
    assert budget.normalize_cap(200) is None
    assert budget.normalize_cap(200, explicit=True) == 200
    assert budget.normalize_cap(150) == 150
    assert budget.normalize_cap(None, explicit=True) is None
    assert budget.normalize_cap(0, explicit=True) is None


def test_plan_cap_reads_the_plans_own_flag():
    assert budget.plan_cap({"execution_settings": {"max_concurrent_jobs": 200}}) is None      # a 1.1.5 plan
    assert budget.plan_cap({"execution_settings": {"max_concurrent_jobs": 200, "max_jobs_explicit": True}}) == 200
    assert budget.plan_cap({"execution_settings": {"max_concurrent_jobs": 40}}) == 40
    assert budget.plan_cap({}) is None


def test_the_plans_a_spec_and_quick_start_write_carry_the_flag(tmp_path):
    from mace.run_mace import create_quick_workflow_plan
    from types import SimpleNamespace
    d12 = tmp_path / "a.d12"
    d12.write_text("x\n")
    for jobs, want in ((200, True), (None, False)):
        args = SimpleNamespace(max_jobs=jobs, work_dir=str(tmp_path), array=False, max_running=None)
        plan = create_quick_workflow_plan(tmp_path, [d12], "d12", ["OPT", "SP"], args)
        ex = plan["execution_settings"]
        assert ex["max_concurrent_jobs"] == jobs and bool(ex.get("max_jobs_explicit")) is want
        assert budget.plan_cap(plan) == (200 if jobs else None)


def test_a_chosen_200_survives_the_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("MACE_STATE_DIR", str(tmp_path / "state"))
    db = tmp_path / "materials.db"
    db.write_text("")
    wid = "workflow_20261021_000000_abcd"
    (tmp_path / "workflow_configs").mkdir()
    plan = {"execution_settings": {"max_concurrent_jobs": 200, "max_jobs_explicit": True}}
    (tmp_path / "workflow_configs" / f"workflow_plan_{wid[len('workflow_'):]}.json").write_text(json.dumps(plan))
    budget.register(db, wid, budget.plan_cap(plan))
    assert budget.workflow_cap(db, wid) == 200
    legacy = "workflow_20260101_000000_beef"
    old_plan = {"execution_settings": {"max_concurrent_jobs": 200}}
    (tmp_path / "workflow_configs" / f"workflow_plan_{legacy[len('workflow_'):]}.json").write_text(json.dumps(old_plan))
    budget.register(db, legacy, 200)           # what 1.1.5 registered
    assert budget.workflow_cap(db, legacy) is None
