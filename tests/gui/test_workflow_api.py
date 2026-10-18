"""The Workflows view's server side (GUI_WORKFLOW_DESIGN.md section 5,
Phase 3): GET workflow/list and workflow/status on a real `mace gui` with
the logging fake SLURM, for workflows written the way the CLI writes them
(plan + the workflow's own database + control file); read only; the time
left of a running step; workflow.changed on the event stream after a poll
that changed a step."""
import json
import shutil
import sqlite3
import time
from pathlib import Path

import pytest

import guiharness as H
import p4util as U
import submit_util as SU

TEST_DATA = H.REPO_ROOT / "test"
WID = "workflow_20261015_100000_ab12"
OLD = "workflow_20250618_145837"          # a 1.1.4-style id, no suffix


def _plan(folder: Path, wid: str, seq, mats, label=None):
    (folder / "workflow_configs").mkdir(parents=True, exist_ok=True)
    plan = {"workflow_id": wid, "created": "2026-10-15T10:00:00", "mode": "manual_progress",
            "input_type": "d12", "workflow_sequence": list(seq), "label": label,
            "input_files": {"d12": [str(folder / f"{m}.d12") for m in mats], "cif": []},
            "execution_settings": {"auto_progression": True}}
    (folder / "workflow_configs" / f"workflow_plan_{wid[len('workflow_'):]}.json").write_text(json.dumps(plan))


def _db(folder: Path, wid: str, rows):
    from mace.database.materials import MaterialDatabase
    path = folder / f".mace_context_{wid}" / "materials.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    MaterialDatabase(str(path))
    conn = sqlite3.connect(path)
    with conn:
        for i, r in enumerate(rows):
            conn.execute(
                "INSERT INTO calculations (calc_id, material_id, calc_type, status, slurm_job_id, created_at, "
                "completed_at, settings_json, input_file) VALUES (?,?,?,?,?,?,?,?,?)",
                (r["id"], r["mat"], r["type"], r["status"], r.get("job"), f"2026-10-15T10:{i:02d}:00",
                 r.get("completed"), json.dumps({"workflow_id": wid}), r.get("input")))
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()
    return path


def _set_status(db: Path, calc_id: str, status: str):
    conn = sqlite3.connect(db)
    with conn:
        conn.execute("UPDATE calculations SET status=?, completed_at=? WHERE calc_id=?",
                     (status, "2026-10-15T11:00:00", calc_id))
    conn.close()


@pytest.fixture(scope="module")
def srv(tmp_path_factory):
    src = TEST_DATA / "OPT" / "1_dia_opt_rev1.out"
    if not src.exists():
        pytest.skip("test/ data corpus not present")
    s = SU.Served(tmp_path_factory.mktemp("wfapi"), {"sacct": {}}, MACE_SITE_CONFIG="none")
    root = s.root
    # a workflow in the served folder: A's OPT runs (half an optimization), B's OPT is done
    wf = root / "perov"
    run = wf / "workflow_outputs" / WID / "step_001_OPT" / "A"
    run.mkdir(parents=True)
    lines = src.read_text(errors="replace").splitlines(True)
    heads = [i for i, l in enumerate(lines) if "OPTIMIZATION - POINT" in l]
    (run / "A.out").write_text("".join(lines[:heads[-1] + 3]))
    shutil.copy(src.with_suffix(".d12"), run / "A.d12")
    (run / "A.sh").write_text("#!/bin/bash\n")
    _plan(wf, WID, ["OPT", "SP", "BAND"], ["A", "B"], label="perovskites")
    db = _db(wf, WID, [
        {"id": "a1", "mat": "A", "type": "OPT", "status": "running", "job": "5101", "input": str(run / "A.d12")},
        {"id": "b1", "mat": "B", "type": "OPT", "status": "completed", "completed": "2026-10-15T10:30:00"},
        {"id": "b2", "mat": "B", "type": "SP", "status": "submitted", "job": "5102"},
    ])
    s.fake.add_record(id="5101", name="A", state="RUNNING", user=SU.USER, workdir=str(run), elapsed="1:00:00",
                      limit="4:00:00")
    s.fake.add_record(id="5102", name="B", state="PENDING", user=SU.USER, workdir=str(wf), reason="Priority")
    # a 1.1.4 plan at the top, never launched
    _plan(root, OLD, ["OPT", "SP"], ["X"])
    s.wf, s.db = wf, db
    s.files = {p.relative_to(root).as_posix(): p.stat().st_mtime_ns for p in root.rglob("*")
               if p.is_file() and not p.name.endswith(("-wal", "-shm"))}
    yield s
    s.stop()


def _get(srv, route):
    st, _, r = srv.req(f"api/{route}")
    return st, r


def test_list_finds_cli_workflows_in_subfolders(srv):
    st, r = _get(srv, "workflow/list")
    assert st == 200, r
    ids = [(w["workflow_id"], w["folder_rel"]) for w in r["workflows"]]
    assert ids == [(WID, "perov"), (OLD, "")]
    mine = r["workflows"][0]
    assert mine["label"] == "perovskites" and mine["state"] == "running" and mine["state_label"] == "In progress"
    assert mine["plan_rel"] == f"perov/workflow_configs/workflow_plan_{WID[len('workflow_'):]}.json"
    assert r["workflows"][1]["state_label"] == "Not started"
    assert r["slurm"] is True and r["states"]["stale"][0] == "Not submitted"


def test_status_nodes_eta_and_links(srv):
    st, r = _get(srv, f"workflow/status?id={WID}&folder=perov")
    assert st == 200, r
    nodes = {m["material_id"]: m["nodes"] for m in r["materials"]}
    a, b = nodes["A"], nodes["B"]
    assert [n["state"] for n in a] == ["running", "planned", "planned"]
    assert [n["state"] for n in b] == ["completed", "queued", "planned"]
    assert a[0]["label"] == "Running" and b[1]["label"] == "Queued" and a[1]["label"] == "Not started"
    run_rel = f"perov/workflow_outputs/{WID}/step_001_OPT/A"
    assert a[0]["deck_rel"] == f"{run_rel}/A.d12" and a[0]["out_rel"] == f"{run_rel}/A.out"
    assert a[0]["script_rel"] == f"{run_rel}/A.sh" and a[0]["job_id"] == "5101"
    eta = a[0]["eta"]
    assert eta["output"] == f"{run_rel}/A.out" and eta["elapsed_s"] == 3600 and eta["limit_s"] == 4 * 3600
    assert eta["text"].endswith("left") and eta["summary"].startswith("Point")
    assert "eta" not in b[1]
    assert r["state_label"] == "In progress" and r["squeue"]["available"] is True
    assert r["database_rel"] == f"perov/.mace_context_{WID}/materials.db"
    # eta=0: nothing read from the outputs
    st, r = _get(srv, f"workflow/status?id={WID}&folder=perov&eta=0")
    assert "eta" not in r["materials"][0]["nodes"][0]


def test_bad_requests(srv):
    assert _get(srv, "workflow/status?id=nope&folder=perov")[0] == 400
    assert _get(srv, f"workflow/status?id={WID}&folder=../..")[0] == 403
    assert _get(srv, f"workflow/status?id={WID}&folder=missing")[0] == 404
    assert _get(srv, "workflow/status?id=workflow_20991231_000000&folder=perov")[0] == 404
    assert _get(srv, "workflow/nothing")[0] == 404


def test_read_only(srv):
    _get(srv, "workflow/list")
    _get(srv, f"workflow/status?id={WID}&folder=perov")
    # SQLite's own -wal/-shm reader files may appear (any reader of a WAL database makes them)
    after = {p.relative_to(srv.root).as_posix(): p.stat().st_mtime_ns for p in srv.root.rglob("*")
             if p.is_file() and not p.name.endswith(("-wal", "-shm"))}
    assert after == srv.files, set(after.items()) ^ set(srv.files.items())


def test_changed_event_after_a_poll(srv):
    """A step's state changes (the OPT job ends, its callback records it): the
    next poll of your jobs sends workflow.changed for that workflow only."""
    ev = H.SSE(srv.base)
    try:
        assert _get(srv, f"workflow/status?id={WID}&folder=perov")[0] == 200     # the view is open
        _get(srv, "jobs?scope=folder&refresh=1")
        time.sleep(1.0)
        assert not ev.of("workflow.changed")
        srv.fake.remove_record("5101")
        _set_status(srv.db, "a1", "completed")
        time.sleep(15.2)                                  # past the shared reader's refresh floor
        _get(srv, "jobs?scope=folder&refresh=1")
        e = ev.wait(lambda e: e[0] == "workflow.changed", timeout=30)
        assert [w["workflow_id"] for w in e[1]["workflows"]] == [WID]
        assert e[1]["workflows"][0]["folder_rel"] == "perov"
        st, r = _get(srv, f"workflow/status?id={WID}&folder=perov")
        assert r["materials"][0]["nodes"][0]["state"] == "completed"
        assert r["materials"][0]["nodes"][1]["state"] in ("waiting", "stale")
    finally:
        ev.close()
