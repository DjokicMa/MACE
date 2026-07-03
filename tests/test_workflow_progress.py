"""Regression tests for mace/database/analysis/workflow_progress.py.

This module was written against a phantom schema and crashed on first real
use (testMACE1.1, 2026-07-02):
  - called db.get_calculations_for_material() (no such method -> AttributeError)
  - read 'calculation_type'/'calculation_id'/'job_id'/'notes' columns that
    don't exist (real schema: calc_type/calc_id/slurm_job_id, no notes)
  - max() over started_at crashed on NULL (dict.get returns None)
  - format_progress_report divided by zero on an empty DB
  - reported the hardcoded 4-step 'full_electronic' template instead of the
    workflow's actual saved plan (7 steps), hiding FREQ/C+P/OPT2 entirely

Driven through the REAL MaterialDatabase (sqlite in tmp) — no mocks on the
DB layer, matching how the CLI invokes it.
"""
import json

import pytest

from mace.database.materials import MaterialDatabase
from mace.database.analysis.workflow_progress import WorkflowProgress

SEQ = ["OPT", "SP", "BAND", "DOSS", "FREQ", "CHARGE+POTENTIAL", "OPT2"]


@pytest.fixture
def db(tmp_path):
    db = MaterialDatabase(db_path=str(tmp_path / "materials.db"))
    db.create_material(material_id="mat1", formula="C2",
                       source_type="d12", source_file="mat1.d12")
    return db


def _add_calc(db, calc_type, status, job=None):
    calc_id = db.create_calculation(material_id="mat1", calc_type=calc_type,
                                    input_file=f"mat1_{calc_type.lower()}.d12",
                                    work_dir=".")
    db.update_calculation_status(calc_id, status, slurm_job_id=job)
    return calc_id

def test_track_progress_real_schema(db):
    """The exact call that raised AttributeError, on real records: statuses,
    ids and job ids must come from the real columns."""
    _add_calc(db, "OPT", "completed", job="111")
    _add_calc(db, "SP", "completed", job="112")
    _add_calc(db, "DOSS", "failed", job="113")

    results = WorkflowProgress(db).track_progress(custom_sequence=SEQ)

    prog = results["materials"]["mat1"]
    by_type = {s["type"]: s for s in prog["workflow_steps"]}
    assert by_type["OPT"]["status"] == "completed"
    assert by_type["SP"]["status"] == "completed"
    assert by_type["DOSS"]["status"] == "failed"
    assert by_type["FREQ"]["status"] == "pending"
    assert by_type["SP"]["job_id"] == "112"           # slurm_job_id column
    assert by_type["SP"]["calculation_id"]            # calc_id column
    assert prog["completed_steps"] == 2
    assert results["summary"]["failed"] == 1


def test_pending_calc_with_null_started_at_does_not_crash(db):
    """created-but-never-started rows have NULL started_at; the most-recent
    selection must not TypeError on None comparisons."""
    _add_calc(db, "OPT", "pending")
    _add_calc(db, "OPT", "completed", job="9")  # second OPT record, started

    results = WorkflowProgress(db).track_progress(custom_sequence=["OPT"])
    assert results["materials"]["mat1"]["workflow_steps"][0]["status"] in (
        "completed", "running")


def test_format_report_empty_db(tmp_path):
    """Zero materials: the report must render (was: division by zero)."""
    db = MaterialDatabase(db_path=str(tmp_path / "empty.db"))
    wp = WorkflowProgress(db)
    report = wp.format_progress_report(wp.track_progress(custom_sequence=SEQ))
    assert "Total materials: 0" in report


def test_saved_plan_preferred_over_template(db, tmp_path, monkeypatch):
    """With workflow_configs/workflow_plan_*.json present, the report tracks
    the ACTUAL 7-step plan, not the 4-step 'full_electronic' template."""
    cfg = tmp_path / "workflow_configs"
    cfg.mkdir()
    (cfg / "workflow_plan_20260702_193809.json").write_text(
        json.dumps({"workflow_sequence": SEQ}))
    monkeypatch.chdir(tmp_path)

    _add_calc(db, "OPT", "completed", job="1")
    results = WorkflowProgress(db).track_progress()

    assert results["sequence"] == SEQ
    assert results["workflow"] == "workflow_20260702_193809"
    # 1 of 7 completed, not 1 of 4
    assert results["materials"]["mat1"]["total_steps"] == 7
