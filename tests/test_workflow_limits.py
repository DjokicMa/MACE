"""Workflow limits closed for 1.1.6: resume in a shared database, and the
time left of running steps in ``mace workflow --status``."""
import json
import shutil
import sqlite3

from conftest import find_data
from mace.workflow import commands, control
from mace.workflow.status import format_status, workflow_status

WID = "workflow_20261015_100000_ee44"


def _plan(folder, seq=("OPT", "SP"), mats=("A",), **extra):
    (folder / "workflow_configs").mkdir(parents=True, exist_ok=True)
    plan = {"workflow_id": WID, "created": "2026-10-15T10:00:00", "input_type": "d12",
            "workflow_sequence": list(seq), "isolation_mode": "shared",
            "input_files": {"d12": [str(folder / f"{m}.d12") for m in mats], "cif": []},
            "execution_settings": {"auto_progression": True}}
    plan.update(extra)
    (folder / "workflow_configs" / f"workflow_plan_{WID[len('workflow_'):]}.json").write_text(json.dumps(plan))


def _shared_rows(folder, rows):
    from mace.database.materials import MaterialDatabase
    MaterialDatabase(str(folder / "materials.db"))
    with sqlite3.connect(folder / "materials.db") as conn:
        for cid, mat, ctype, status, settings, extra in rows:
            conn.execute("INSERT INTO calculations (calc_id, material_id, calc_type, status, created_at, "
                         "settings_json, slurm_job_id, input_file, output_file) VALUES (?,?,?,?,?,?,?,?,?)",
                         (cid, mat, ctype, status, "2026-10-15T10:00:00", json.dumps(settings),
                          extra.get("job"), extra.get("input"), extra.get("output")))


def _settings(folder, cid):
    with sqlite3.connect(folder / "materials.db") as conn:
        (s,) = conn.execute("SELECT settings_json FROM calculations WHERE calc_id=?", (cid,)).fetchone()
    return json.loads(s or "{}")


def test_resume_in_shared_database_starts_only_this_workflows_steps(tmp_path, monkeypatch, capsys):
    """A shared-mode workflow (the folder's materials.db) resumed: its own
    held completions are processed; other runs' rows (another workflow's,
    an unstamped hand submission's) are left alone."""
    from mace.workflow.engine import WorkflowEngine
    _plan(tmp_path, mats=("A", "B", "C"))
    _shared_rows(tmp_path, [
        ("mine", "A", "OPT", "completed", {"workflow_id": WID}, {}),
        ("other", "B", "OPT", "completed", {"workflow_id": "workflow_20261015_090000_aaaa"}, {}),
        ("hand", "C", "OPT", "completed", {}, {}),
    ])
    control.write_control(tmp_path, WID, "paused")
    seen = []
    monkeypatch.setattr(WorkflowEngine, "execute_workflow_step",
                        lambda self, mid, cid: seen.append(cid) or ["new"])
    monkeypatch.delenv("MACE_WORKFLOW_ID", raising=False)
    capsys.readouterr()
    assert commands.cmd_resume(tmp_path, WID, as_json=True) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["steps_started"] == 1 and out["shared_database"] is True
    assert seen == ["mine"]
    assert _settings(tmp_path, "mine").get("workflow_processed") is True
    assert not _settings(tmp_path, "other").get("workflow_processed")
    assert not _settings(tmp_path, "hand").get("workflow_processed")
    assert control.state_of(control.read_control(tmp_path, WID)) == "active"


def test_status_eta_for_running_steps(tmp_path):
    """--status gives each running step the Jobs view's estimate (read from
    its .out); other steps carry none."""
    src = find_data("OPT/1_dia_opt_rev1.out")
    lines = src.read_text(errors="replace").splitlines(keepends=True)
    cut = [i for i, l in enumerate(lines) if "OPTIMIZATION - POINT" in l]
    work = tmp_path / "A"
    work.mkdir()
    out = work / "A.out"
    out.write_text("".join(lines[:cut[-1] + 3]))       # point 3 of the optimization
    shutil.copy(find_data("OPT/1_dia_opt_rev1.d12"), work / "A.d12")
    _plan(tmp_path, mats=("A",))
    _shared_rows(tmp_path, [("run", "A", "OPT", "running", {"workflow_id": WID},
                             {"job": "4242", "input": str(work / "A.d12")})])
    sq = [{"id": "4242", "state": "RUNNING", "elapsed": "1:00:00", "time_limit": "4:00:00"}]
    st = workflow_status(tmp_path, WID, squeue=sq, eta=True)
    opt, sp = st["materials"][0]["nodes"]
    assert opt["state"] == "running" and sp["state"] == "planned" and "eta" not in sp
    eta = opt["eta"]
    assert eta["output"] == str(out) and eta["elapsed_s"] == 3600 and eta["limit_s"] == 4 * 3600
    assert eta["high_s"] and eta["text"].endswith("left") and eta["stage"] == "optimizing"
    assert eta["summary"].startswith("Point 3")
    assert "Running:" in format_status(st)
    # without eta=True nothing reads the .out
    assert "eta" not in workflow_status(tmp_path, WID, squeue=sq)["materials"][0]["nodes"][0]


def test_status_eta_without_output(tmp_path):
    _plan(tmp_path, mats=("A",))
    _shared_rows(tmp_path, [("run", "A", "OPT", "running", {"workflow_id": WID},
                             {"job": "1", "input": str(tmp_path / "missing" / "A.d12")})])
    st = workflow_status(tmp_path, WID, squeue=[{"id": "1", "state": "RUNNING"}], eta=True)
    assert st["materials"][0]["nodes"][0]["eta"] is None


def test_status_eta_too_early_says_why(tmp_path):
    src = find_data("OPT/1_dia_opt_rev1.out")
    lines = src.read_text(errors="replace").splitlines(keepends=True)
    cut = [i for i, l in enumerate(lines) if "OPTIMIZATION - POINT" in l]
    (tmp_path / "A.out").write_text("".join(lines[:cut[1]]))
    _plan(tmp_path, mats=("A",))
    _shared_rows(tmp_path, [("run", "A", "OPT", "running", {"workflow_id": WID},
                             {"job": "7", "output": str(tmp_path / "A.out")})])
    st = workflow_status(tmp_path, WID, squeue=[{"id": "7", "state": "RUNNING"}], eta=True)
    eta = st["materials"][0]["nodes"][0]["eta"]
    assert eta["low_s"] is None and "estimate needs" in eta["text"]
