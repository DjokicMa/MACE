"""An OPT planned after an SP is generated from that SP, in the same workflow.

With no OPT completed, the engine used to re-convert the material's original
CIF with NewCifToD12 (generate_calculation_from_cif). That dropped every
setting of the SP deck and put the OPT in a freshly minted
workflow_<timestamp> directory, so the rest of the plan never found it. The
OPT now comes from the completed SP through CRYSTALOptToD12, like every other
derived step: same workflow id, next step number, the SP's method kept.

Runs the engine in a separate interpreter, as the completion callback does,
and CRYSTALOptToD12 for real on a corpus SP.
"""
import json
import os
import shutil
import sqlite3
import subprocess
import sys

import pytest

from conftest import REPO_ROOT, find_data
from mace.database.materials import MaterialDatabase

WF = "workflow_20260924_120000"


def _records(text):
    return [l.strip() for l in text.splitlines()]


def test_sp_then_opt_stays_in_the_workflow(tmp_path):
    src_d12 = find_data("SP/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized.d12")
    src_out = src_d12.with_suffix(".out")

    sp_dir = tmp_path / "workflow_outputs" / WF / "step_001_SP" / "dia"
    sp_dir.mkdir(parents=True)
    shutil.copy(src_d12, sp_dir / "dia.d12")
    shutil.copy(src_out, sp_dir / "dia.out")
    cfg = tmp_path / "workflow_configs"
    cfg.mkdir()
    (cfg / f"workflow_plan_{WF[len('workflow_'):]}.json").write_text(json.dumps(
        {"workflow_id": WF, "workflow_sequence": ["SP", "OPT"],
         "step_configurations": {}}))

    db_path = tmp_path / "materials.db"
    db = MaterialDatabase(str(db_path))
    db.create_material("dia", "C")
    sp_id = db.create_calculation("dia", "SP", input_file=str(sp_dir / "dia.d12"),
                                  work_dir=str(sp_dir),
                                  settings={"workflow_id": WF, "workflow_step": 1})
    db.update_calculation_status(sp_id, "completed", output_file=str(sp_dir / "dia.out"))

    code = ("import sys; from mace.workflow.engine import WorkflowEngine as E; "
            "print('NEW', E(sys.argv[1], sys.argv[2], auto_submit=False)"
            ".execute_workflow_step('dia', sys.argv[3]))")
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    env.pop("MACE_WORKFLOW_ID", None)
    r = subprocess.run([sys.executable, "-c", code, str(db_path), str(tmp_path), sp_id],
                       cwd=tmp_path, env=env, capture_output=True, text=True,
                       timeout=300, stdin=subprocess.DEVNULL)
    log = r.stdout + r.stderr
    assert r.returncode == 0, log
    assert "from CIF" not in log, log

    rows = sqlite3.connect(db_path).execute(
        "SELECT calc_type, status, input_file, settings_json FROM calculations "
        "WHERE calc_type = 'OPT'").fetchall()
    assert len(rows) == 1, log
    _, status, input_file, settings = rows[0]
    settings = json.loads(settings)
    assert status == "pending"
    assert settings["workflow_id"] == WF
    assert settings["parent_calc_id"] == sp_id

    # Same workflow folder, next step number; no second workflow minted.
    opt_d12 = tmp_path / "workflow_outputs" / WF / "step_002_OPT" / "dia_opt" / "dia_opt.d12"
    assert input_file == str(opt_d12)
    assert [p.name for p in (tmp_path / "workflow_outputs").iterdir()] == [WF]

    # The SP deck plus an OPTGEOM block: geometry, basis, method and SCF
    # settings all carried over.
    sp_recs = _records((sp_dir / "dia.d12").read_text())[1:]
    opt_recs = _records(opt_d12.read_text())[1:]
    assert "OPTGEOM" in opt_recs
    block = opt_recs[opt_recs.index("OPTGEOM"):opt_recs.index("ENDOPT") + 1]
    rest = opt_recs[:opt_recs.index("OPTGEOM")] + opt_recs[opt_recs.index("ENDOPT") + 1:]
    assert rest == sp_recs, "\n".join(opt_recs)
    assert "FULLOPTG" in block


@pytest.mark.parametrize("between", ["BAND", "DOSS", "TRANSPORT", "CHARGE+POTENTIAL"])
def test_opt_after_band_or_doss_comes_from_the_sp(tmp_path, between):
    """SP -> BAND/DOSS/TRANSPORT/CHARGE+POTENTIAL -> OPT: when that step completes and no OPT
    has run yet, the planned OPT is built from the SP. It used to print
    "No completed OPT found" and stall the workflow."""
    src_d12 = find_data("SP/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized.d12")
    wf_dir = tmp_path / "workflow_outputs" / WF
    sp_dir = wf_dir / "step_001_SP" / "dia"
    sp_dir.mkdir(parents=True)
    shutil.copy(src_d12, sp_dir / "dia.d12")
    shutil.copy(src_d12.with_suffix(".out"), sp_dir / "dia.out")
    mid_dir = wf_dir / f"step_002_{between}" / f"dia_{between.lower()}"
    mid_dir.mkdir(parents=True)
    (mid_dir / "dia.d3").write_text("x\n")
    (mid_dir / "dia.out").write_text("x\n")
    cfg = tmp_path / "workflow_configs"
    cfg.mkdir()
    (cfg / f"workflow_plan_{WF[len('workflow_'):]}.json").write_text(json.dumps(
        {"workflow_id": WF, "workflow_sequence": ["SP", between, "OPT"],
         "step_configurations": {}}))

    db_path = tmp_path / "materials.db"
    db = MaterialDatabase(str(db_path))
    db.create_material("dia", "C")
    sp_id = db.create_calculation("dia", "SP", input_file=str(sp_dir / "dia.d12"),
                                  work_dir=str(sp_dir),
                                  settings={"workflow_id": WF, "workflow_step": 1})
    db.update_calculation_status(sp_id, "completed", output_file=str(sp_dir / "dia.out"))
    mid_id = db.create_calculation("dia", between, input_file=str(mid_dir / "dia.d3"),
                                   work_dir=str(mid_dir),
                                   settings={"workflow_id": WF, "workflow_step": 2,
                                             "parent_calc_id": sp_id})
    db.update_calculation_status(mid_id, "completed", output_file=str(mid_dir / "dia.out"))

    code = ("import sys; from mace.workflow.engine import WorkflowEngine as E; "
            "print('NEW', E(sys.argv[1], sys.argv[2], auto_submit=False)"
            ".execute_workflow_step('dia', sys.argv[3]))")
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    env.pop("MACE_WORKFLOW_ID", None)
    r = subprocess.run([sys.executable, "-c", code, str(db_path), str(tmp_path), mid_id],
                       cwd=tmp_path, env=env, capture_output=True, text=True,
                       timeout=300, stdin=subprocess.DEVNULL)
    log = r.stdout + r.stderr
    assert r.returncode == 0, log

    rows = sqlite3.connect(db_path).execute(
        "SELECT input_file, settings_json FROM calculations WHERE calc_type = 'OPT'").fetchall()
    assert len(rows) == 1, log
    input_file, settings = rows[0]
    settings = json.loads(settings)
    assert settings["workflow_id"] == WF
    assert settings["parent_calc_id"] == sp_id
    assert input_file == str(wf_dir / "step_003_OPT" / "dia_opt" / "dia_opt.d12")
