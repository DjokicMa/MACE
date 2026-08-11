"""A workflow-less calculation record must not orphan the follow-up steps.

Real incident (HPCC job 15338060, 2026-08-11, `mace submit --progress
full_electronic`): the OPT ran, progressed to SP, and the chain then stopped
dead. Cause chain:

  * the run used an isolated context, so the completion callback re-registered
    the finished OPT from its output file (_populate_completed_jobs_from_outputs).
    That scan records no workflow metadata, so settings_json held only
    {"output_file": ...};
  * execute_workflow_step still found the plan (it falls back to
    $MACE_WORKFLOW_ID) and correctly generated the SP;
  * but get_workflow_output_base read workflow_id from that same settings dict,
    found nothing, and fell through to its last-resort branch — minting a NEW
    workflow_<timestamp> directory with no plan file;
  * the SP was therefore stamped with the synthesized id, and when IT completed
    no plan could be found, so progression stopped instead of fanning out to
    BAND + DOSS.

Before the planless-progression gate this was masked: the no-plan branch emitted
default BAND+DOSS anyway, so the chain looked right while the workflow id was
already wrong.
"""
import json
import os
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT))


def _engine(tmp_path):
    from mace.workflow import engine as engine_mod
    eng = engine_mod.WorkflowEngine.__new__(engine_mod.WorkflowEngine)
    eng.base_work_dir = tmp_path
    return eng


def _scan_created_opt(tmp_path):
    """An OPT record exactly as the completion-callback scan writes it."""
    return {
        "calc_id": "1_dia_OPT_20260811_144247868",
        "material_id": "1_dia",
        "calc_type": "OPT",
        "status": "completed",
        "input_file": str(tmp_path / "1_dia_opt.d12"),
        "output_file": str(tmp_path / "1_dia_opt.out"),
        "work_dir": str(tmp_path),
        # note: no workflow_id — this is the whole point
        "settings_json": json.dumps({"output_file": str(tmp_path / "1_dia_opt.out")}),
    }


def test_output_dir_uses_env_workflow_id_when_record_has_none(tmp_path, monkeypatch):
    monkeypatch.setenv("MACE_WORKFLOW_ID", "workflow_20260811_144120")
    eng = _engine(tmp_path)

    out = eng.get_workflow_output_base(_scan_created_opt(tmp_path))

    assert out == tmp_path / "workflow_outputs" / "workflow_20260811_144120", out


def test_settings_workflow_id_still_wins_over_env(tmp_path, monkeypatch):
    """An explicitly recorded id is authoritative; the env is only a fallback."""
    monkeypatch.setenv("MACE_WORKFLOW_ID", "workflow_FROM_ENV")
    eng = _engine(tmp_path)
    calc = _scan_created_opt(tmp_path)
    calc["settings_json"] = json.dumps({"workflow_id": "workflow_FROM_RECORD"})

    out = eng.get_workflow_output_base(calc)

    assert out.name == "workflow_FROM_RECORD", out


def test_no_env_and_no_record_still_synthesizes(tmp_path, monkeypatch):
    """Outside any workflow the last-resort branch must still produce a dir."""
    monkeypatch.delenv("MACE_WORKFLOW_ID", raising=False)
    eng = _engine(tmp_path)

    out = eng.get_workflow_output_base(_scan_created_opt(tmp_path))

    assert out.parent == tmp_path / "workflow_outputs"
    assert out.name.startswith("workflow_")


def test_scan_stamps_workflow_id_into_settings(tmp_path, monkeypatch):
    """The scan-created record itself should carry the workflow it belongs to,
    so the DB is coherent and not only rescued by the env fallback."""
    from mace.database.materials import MaterialDatabase
    from mace.database.populate_completed_jobs import populate_database

    monkeypatch.setenv("MACE_WORKFLOW_ID", "workflow_20260811_144120")
    db = MaterialDatabase(str(tmp_path / "materials.db"))

    populate_database([{
        "material_id": "1_dia",
        "calc_type": "OPT",
        "output_file": str(tmp_path / "1_dia_opt.out"),
        "input_file": str(tmp_path / "1_dia_opt.d12"),
        "work_dir": str(tmp_path),
        "completed": True,
    }], db)

    calcs = db.get_calculations_by_status(material_id="1_dia")
    assert len(calcs) == 1, calcs
    settings = json.loads(calcs[0].get("settings_json") or "{}")
    assert settings.get("workflow_id") == "workflow_20260811_144120", settings


def test_scan_outside_workflow_records_no_workflow_id(tmp_path, monkeypatch):
    from mace.database.materials import MaterialDatabase
    from mace.database.populate_completed_jobs import populate_database

    monkeypatch.delenv("MACE_WORKFLOW_ID", raising=False)
    db = MaterialDatabase(str(tmp_path / "materials.db"))

    populate_database([{
        "material_id": "1_dia", "calc_type": "OPT",
        "output_file": str(tmp_path / "1_dia_opt.out"),
        "input_file": str(tmp_path / "1_dia_opt.d12"),
        "work_dir": str(tmp_path), "completed": True,
    }], db)

    calcs = db.get_calculations_by_status(material_id="1_dia")
    settings = json.loads(calcs[0].get("settings_json") or "{}")
    assert "workflow_id" not in settings, settings
