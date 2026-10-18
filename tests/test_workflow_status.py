"""mace/workflow/status.py: a workflow's state derived from its plan, its
database (read-only), squeue and its control file — on synthetic folders."""
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from mace.workflow import control
from mace.workflow.status import list_workflows, parent_index, plan_files, workflow_status

NOW = datetime(2026, 10, 15, 12, 0, 0)
SEQ = ["OPT", "SP", "BAND", "DOSS"]


def _plan(folder, wid, seq=SEQ, mats=("A", "B"), **extra):
    (folder / "workflow_configs").mkdir(parents=True, exist_ok=True)
    plan = {"workflow_id": wid, "created": "2026-10-15T10:00:00", "mode": "manual_progress",
            "input_type": "d12", "workflow_sequence": list(seq),
            "input_files": {"d12": [str(folder / f"{m}.d12") for m in mats], "cif": []},
            "execution_settings": {"auto_progression": True}}
    plan.update(extra)
    (folder / "workflow_configs" / f"workflow_plan_{wid[len('workflow_'):]}.json").write_text(json.dumps(plan))
    return plan


def _db(folder, wid, rows):
    from mace.database.materials import MaterialDatabase
    path = folder / f".mace_context_{wid}" / "materials.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    MaterialDatabase(str(path))
    with sqlite3.connect(path) as conn:
        for i, r in enumerate(rows):
            settings = dict({"workflow_id": wid}, **r.pop("settings", {}))
            conn.execute(
                "INSERT INTO calculations (calc_id, material_id, calc_type, status, slurm_job_id, "
                "created_at, completed_at, settings_json, error_type) VALUES (?,?,?,?,?,?,?,?,?)",
                (r.get("calc_id", f"c{i}"), r["mat"], r["type"], r["status"], r.get("job"),
                 r.get("created", f"2026-10-15T10:{i:02d}:00"), r.get("completed"),
                 json.dumps(settings), r.get("error")))
    return path


def _states(st):
    return {m["material_id"]: [n["state"] for n in m["nodes"]] for m in st["materials"]}


def test_parent_index_follows_dependencies():
    assert parent_index(SEQ, 0) is None
    assert parent_index(SEQ, 1) == 0          # SP <- OPT
    assert parent_index(SEQ, 3) == 1          # DOSS <- SP, not BAND
    assert parent_index(["OPT", "OPT2", "SP"], 2) == 1
    assert parent_index(["OPT", "SP", "BAND", "DOSS", "FREQ"], 4) == 0


def test_states_from_db_squeue_and_rules(tmp_path):
    wid = "workflow_20261015_100000_ab12"
    _plan(tmp_path, wid)
    long_ago = (NOW - timedelta(hours=2)).isoformat()
    _db(tmp_path, wid, [
        {"mat": "A", "type": "OPT", "status": "completed", "completed": "2026-10-15T11:50:00",
         "settings": {"workflow_processed": True}},
        {"mat": "A", "type": "SP", "status": "submitted", "job": "4242_0"},
        {"mat": "B", "type": "OPT", "status": "failed", "job": "5001", "error": "SCF"},
        {"mat": "B", "type": "OPT", "status": "running", "job": "5002", "calc_id": "c-rec"},
    ])
    sq = [{"id": "4242_0", "state": "RUNNING"}]
    st = workflow_status(tmp_path, wid, squeue=sq, now=NOW)
    s = _states(st)
    assert s["A"] == ["completed", "running", "planned", "planned"]
    # B: a recovery attempt is the latest; squeue no longer lists it
    assert s["B"] == ["ending", "planned", "planned", "planned"]
    b_opt = st["materials"][1]["nodes"][0]
    assert b_opt["attempts"] == 2 and b_opt["calc_id"] == "c-rec"
    assert st["state"] == "running" and st["squeue"]["available"]
    json.dumps(st)                          # JSON-ready


def test_waiting_stale_blocked_held_paused(tmp_path):
    wid = "workflow_20261015_100000_cd34"
    _plan(tmp_path, wid, mats=("A", "B", "C"))
    _db(tmp_path, wid, [
        {"mat": "A", "type": "OPT", "status": "completed", "completed": (NOW - timedelta(minutes=5)).isoformat()},
        {"mat": "B", "type": "OPT", "status": "completed", "completed": (NOW - timedelta(hours=3)).isoformat()},
        {"mat": "C", "type": "OPT", "status": "failed"},
    ])
    s = _states(workflow_status(tmp_path, wid, squeue=None, use_squeue=False, now=NOW))
    assert s["A"][1] == "waiting" and s["B"][1] == "stale"
    assert s["C"] == ["failed", "blocked", "blocked", "blocked"]

    control.write_control(tmp_path, wid, "paused")
    st = workflow_status(tmp_path, wid, use_squeue=False, now=NOW)
    assert st["state"] == "paused" and _states(st)["A"][1] == "paused"
    control.write_control(tmp_path, wid, "active")
    plan = json.loads(plan_files(tmp_path)[wid].read_text())
    plan["execution_settings"]["auto_progression"] = False
    plan_files(tmp_path)[wid].write_text(json.dumps(plan))
    st = workflow_status(tmp_path, wid, use_squeue=False, now=NOW)
    assert _states(st)["A"][1] == "held" and not st["auto_progression"]


def test_completed_and_archived(tmp_path):
    wid = "workflow_20261015_100000_ef56"
    _plan(tmp_path, wid, seq=["OPT", "SP"], mats=("A",))
    db = _db(tmp_path, wid, [{"mat": "A", "type": "OPT", "status": "completed"},
                             {"mat": "A", "type": "SP", "status": "completed"}])
    arch = tmp_path / "archived_workflows" / f"{wid}_20261015_120000"
    arch.parent.mkdir()
    db.parent.rename(arch)
    st = workflow_status(tmp_path, wid, use_squeue=False)
    assert st["state"] == "completed" and st["archived"]


def test_list_includes_old_plans_and_skips_decoys(tmp_path):
    old = "workflow_20250618_145837"        # a 1.1.4 id (no suffix)
    new = "workflow_20261015_100000_ab12"
    _plan(tmp_path, old, mats=("A",))
    _plan(tmp_path, new, mats=("A",), label="mine")
    cfg = tmp_path / "workflow_configs"
    (cfg / "workflow_plan_20250618_145837_updated.json").write_text("{}")
    (cfg / "workflow_plan_garbage.json").write_text("{}")
    items = list_workflows(tmp_path)
    assert [i["workflow_id"] for i in items] == [new, old]
    assert items[0]["label"] == "mine" and items[1]["state"] == "planned"


def test_shared_database_rows_by_workflow_id(tmp_path):
    from mace.database.materials import MaterialDatabase
    wid = "workflow_20261015_100000_aa00"
    _plan(tmp_path, wid, seq=["OPT"], mats=("A",))
    MaterialDatabase(str(tmp_path / "materials.db"))
    with sqlite3.connect(tmp_path / "materials.db") as conn:
        for cid, w in (("x1", wid), ("x2", "workflow_other"), ("x3", None)):
            conn.execute("INSERT INTO calculations (calc_id, material_id, calc_type, status, created_at, "
                         "settings_json) VALUES (?,?,?,?,?,?)",
                         (cid, "A", "OPT", "completed" if cid == "x1" else "failed",
                          "2026-10-15T10:00:00", json.dumps({"workflow_id": w} if w else {})))
    st = workflow_status(tmp_path, wid, use_squeue=False)
    node = st["materials"][0]["nodes"][0]
    assert node["calc_id"] == "x1" and node["attempts"] == 1


def test_read_only(tmp_path):
    wid = "workflow_20261015_100000_bb11"
    _plan(tmp_path, wid, seq=["OPT"], mats=("A",))
    db = _db(tmp_path, wid, [{"mat": "A", "type": "OPT", "status": "completed"}])
    before = db.stat().st_mtime_ns
    db.chmod(0o444)
    try:
        workflow_status(tmp_path, wid, use_squeue=False)
    finally:
        db.chmod(0o644)
    assert db.stat().st_mtime_ns == before
    assert workflow_status(tmp_path, "workflow_20991231_000000", use_squeue=False) is None


def test_cancel_confirms_jobs_with_slurm(tmp_path):
    from mace.workflow.commands import workflow_jobs
    wid = "workflow_20261015_100000_cc22"
    _plan(tmp_path, wid, mats=("A", "B", "C"))
    _db(tmp_path, wid, [
        {"mat": "A", "type": "OPT", "status": "running", "job": "7001"},
        {"mat": "B", "type": "OPT", "status": "submitted", "job": "7002"},   # not in squeue
        {"mat": "C", "type": "OPT", "status": "completed", "job": "7003"},
    ])
    jobs = workflow_jobs(tmp_path, wid, squeue=[{"id": "7001", "state": "RUNNING"},
                                                {"id": "7003", "state": "RUNNING"},
                                                {"id": "9999", "state": "PENDING"}])
    assert [j["job_id"] for j in jobs["confirmed"]] == ["7001"]
    assert jobs["unconfirmed"] == ["7002"]


def test_held_state_lookup(tmp_path):
    wid = "workflow_20261015_100000_dd33"
    deep = tmp_path / "a" / "b"
    deep.mkdir(parents=True)
    assert control.held_state(wid, [deep]) is None
    control.write_control(tmp_path, wid, "paused")
    assert control.held_state(wid, [deep]) == "paused"
    control.write_control(tmp_path, wid, "active")
    assert control.held_state(wid, [deep]) is None
    control.control_path(tmp_path, wid).write_text("{not json")
    assert control.held_state(wid, [deep]) is None
    with pytest.raises(ValueError):
        control.write_control(tmp_path, wid, "stopped")


def test_a_job_submitted_moments_ago_is_queued_before_squeue_lists_it(tmp_path):
    """SLURM's own-jobs list is cached (a poll interval); a job MACE just
    submitted (Resume, Process now, Recover) is queued, not finishing."""
    wid = "workflow_20261015_100000_ab12"
    _plan(tmp_path, wid, seq=["OPT"], mats=("A", "B"))
    _db(tmp_path, wid, [
        {"mat": "A", "type": "OPT", "status": "submitted", "job": "6001",
         "created": (NOW - timedelta(seconds=40)).isoformat()},
        {"mat": "B", "type": "OPT", "status": "submitted", "job": "6002",
         "created": (NOW - timedelta(minutes=10)).isoformat()},
    ])
    st = workflow_status(tmp_path, wid, squeue=[], now=NOW)
    assert _states(st) == {"A": ["queued"], "B": ["ending"]}
