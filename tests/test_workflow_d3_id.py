"""Regression test for the D3 workflow_id fallback (engine.py).

When a D3 (BAND/DOSS/...) step is generated, its workflow_id used to fall back to
the literal 'manual' whenever the parent calc had none recorded, which broke
workflow continuation (later steps look calculations up by workflow_id). It must
instead derive the real workflow id from the workflow output base, matching the
SP/numbered-step paths. Self-contained: uses a tmp DB, no test/ corpus.
"""
from pathlib import Path

import pytest

from mace.workflow.engine import WorkflowEngine


@pytest.fixture
def engine(tmp_path):
    return WorkflowEngine(db_path=str(tmp_path / "wf.db"),
                          base_work_dir=str(tmp_path), auto_submit=False)


def _make_parent(engine, tmp_path, settings):
    calc_id = engine.db.create_calculation(
        material_id="testmat", calc_type="SP",
        input_file=str(tmp_path / "testmat_sp.d12"),
        work_dir=str(tmp_path), settings=settings)
    return calc_id


def _capture_workflow_id(engine):
    captured = {}

    def fake_script(work_dir, material_id, calc_type, step_num, workflow_id):
        captured["workflow_id"] = workflow_id
        return None  # short-circuit before any submission

    engine._create_slurm_script_for_calculation = fake_script
    return captured


def test_d3_prefers_parent_stored_workflow_id(engine, tmp_path):
    parent = _make_parent(engine, tmp_path, {"workflow_id": "workflow_20260101_120000"})
    captured = _capture_workflow_id(engine)
    engine._create_and_submit_d3_calculation(
        "testmat", "BAND", tmp_path / "testmat_band.d3", tmp_path, parent)
    assert captured["workflow_id"] == "workflow_20260101_120000"


def test_d3_fallback_is_not_literal_manual(engine, tmp_path):
    """No workflow_id on the parent -> derive a real workflow id, never 'manual'."""
    parent = _make_parent(engine, tmp_path, {})  # no workflow_id recorded
    captured = _capture_workflow_id(engine)
    engine._create_and_submit_d3_calculation(
        "testmat", "BAND", tmp_path / "testmat_band.d3", tmp_path, parent)
    assert captured["workflow_id"] != "manual"
    assert captured["workflow_id"].startswith("workflow_")
