"""A job that leaves the queue without writing its .out must not stay
'submitted' forever.

HPCC QA (2026-10-03): two follow-up jobs ended in seconds with no .out
(sacct COMPLETED). The completion callback and `mace manager --callback-mode
status_check` left both records 'submitted', so the workflow could neither
finish nor recover, and `mace monitor --status` counted them nowhere. The same
applies to a job canceled while pending or killed before CRYSTAL starts.
"""
import os
import sqlite3
import subprocess
import sys

import pytest

from conftest import REPO_ROOT

MANAGER = REPO_ROOT / "mace" / "queue" / "manager.py"
MACE_CLI = REPO_ROOT / "mace_cli"


def _env(tmp_path, sacct_state):
    bindir = tmp_path / "fakebin"
    bindir.mkdir()
    (bindir / "squeue").write_text("#!/bin/bash\necho 'JOBID,STATE,START'\n")
    (bindir / "sacct").write_text(f"#!/bin/bash\necho '{sacct_state}'\n")
    (bindir / "sbatch").write_text("#!/bin/bash\necho 'Submitted batch job 1'\n")
    for f in bindir.iterdir():
        f.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = f"{bindir}:{env['PATH']}"
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["MACE_NO_BANNER"] = "1"
    for var in ("MACE_CONTEXT_DIR", "MACE_WORKFLOW_ID", "MACE_ISOLATION_MODE",
                "SLURM_JOB_ID", "MACE_HOME"):
        env.pop(var, None)
    return env


def _make_db(tmp_path, with_output=False):
    sys.path.insert(0, str(REPO_ROOT))
    from mace.database.materials import MaterialDatabase
    work = tmp_path / "MgO_sp"
    work.mkdir()
    (work / "MgO_sp.d12").write_text("dummy\n")
    db = MaterialDatabase(str(tmp_path / "materials.db"))
    db.create_material("MgO", "MgO")
    calc_id = db.create_calculation("MgO", "SP", input_file=str(work / "MgO_sp.d12"),
                                    work_dir=str(work))
    db.update_calculation_status(calc_id, "submitted", slurm_job_id="999001")
    return calc_id


def _status(tmp_path, calc_id):
    with sqlite3.connect(tmp_path / "materials.db") as conn:
        return conn.execute(
            "SELECT status, error_message FROM calculations WHERE calc_id=?",
            (calc_id,)).fetchone()


@pytest.mark.parametrize("state", ["COMPLETED", "CANCELLED by 1234", "FAILED"])
def test_status_check_fails_job_that_left_no_output(tmp_path, state):
    calc_id = _make_db(tmp_path)
    proc = subprocess.run(
        [sys.executable, str(MANAGER), "--callback-mode", "status_check",
         "--db-path", str(tmp_path / "materials.db"), "--disable-error-recovery"],
        cwd=str(tmp_path), env=_env(tmp_path, state),
        capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    status, message = _status(tmp_path, calc_id)
    assert status == "failed", proc.stdout
    assert "MgO_sp.out" in (message or ""), message


def test_unknown_queue_state_keeps_the_record(tmp_path):
    """sacct unavailable: squeue alone is not trusted (it transiently
    returns an empty list), so the record is left as it is."""
    calc_id = _make_db(tmp_path)
    env = _env(tmp_path, "COMPLETED")
    (tmp_path / "fakebin" / "sacct").write_text("#!/bin/bash\nexit 1\n")
    proc = subprocess.run(
        [sys.executable, str(MANAGER), "--callback-mode", "status_check",
         "--db-path", str(tmp_path / "materials.db"), "--disable-error-recovery"],
        cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _status(tmp_path, calc_id)[0] == "submitted"


def test_monitor_status_counts_submitted(tmp_path):
    _make_db(tmp_path)
    proc = subprocess.run(
        [sys.executable, str(MACE_CLI), "--no-banner", "monitor", "--status"],
        cwd=str(tmp_path), env=_env(tmp_path, ""), capture_output=True,
        text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Submitted: 1" in proc.stdout, proc.stdout
