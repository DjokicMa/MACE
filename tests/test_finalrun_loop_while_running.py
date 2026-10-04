"""An optimization caught in a FINALRUN loop is recovered while it still runs.

HPCC re-test (1.1.5): the Si B3LYP-D3 TOLINTEG 7 loop showed its alternating
+/-2.2E-4 Ha jumps from about five minutes in, but nothing acted on it until
SLURM killed the job at its walltime (226 re-starts); the recovery ran only
once the job had left the queue. The queue manager's checks of running jobs
(`mace manager`, the early-failure and status checks, and the completion
callback of any other job) now cancel such a job and recover it as
`finalrun_loop` at once. Run through mace_cli / manager.py as subprocesses
with SLURM stubs; nothing reaches a real queue.
"""
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta

import pytest

from conftest import REPO_ROOT
from test_finalrun_loop import DECK, loop_output
from test_opt_timeout_restart import job_script

MACE_CLI = REPO_ROOT / "mace_cli"
MANAGER = REPO_ROOT / "mace" / "queue" / "manager.py"
JOB = "999001"


def _box(tmp_path, out_text, calc_type="OPT"):
    bindir = tmp_path / "fakebin"
    bindir.mkdir()
    log = tmp_path / "slurm.log"
    stubs = {
        "squeue": f"echo 'JOBID,STATE,START'; echo '{JOB},RUNNING,2026-01-01T00:00:00'",
        "sacct": "echo RUNNING",
        "scancel": f'echo "scancel $*" >> {log}',
        "sbatch": f'echo "sbatch $*" >> {log}; echo "Submitted batch job 999002"',
    }
    for name, body in stubs.items():
        (bindir / name).write_text("#!/bin/bash\n" + body + "\n")
        (bindir / name).chmod(0o755)
    work = tmp_path / "work"
    work.mkdir()
    (work / "si.d12").write_text(DECK)
    job_script(work, "1-00:00:00", name="si")
    (work / "si.out").write_text(out_text)

    sys.path.insert(0, str(REPO_ROOT))
    from mace.database.materials import MaterialDatabase
    db = MaterialDatabase(str(work / "materials.db"))
    db.create_material("si", "Si")
    calc_id = db.create_calculation("si", calc_type, input_file=str(work / "si.d12"),
                                    work_dir=str(work))
    db.update_calculation_status(calc_id, "running", slurm_job_id=JOB)
    started = (datetime.now() - timedelta(minutes=30)).isoformat()
    with sqlite3.connect(work / "materials.db") as con:
        con.execute("UPDATE calculations SET started_at=? WHERE calc_id=?", (started, calc_id))

    env = dict(os.environ)
    env["PATH"] = f"{bindir}:{env['PATH']}"
    env["MACE_NO_BANNER"] = "1"
    for var in ("MACE_CONTEXT_DIR", "MACE_WORKFLOW_ID", "MACE_ISOLATION_MODE", "SLURM_JOB_ID",
                "MACE_HOME", "PYTHONPATH", "COLORTERM", "TERM", "NO_COLOR"):
        env.pop(var, None)
    return work, calc_id, env, log


def _row(work, calc_id):
    with sqlite3.connect(work / "materials.db") as con:
        con.row_factory = sqlite3.Row
        return dict(con.execute("SELECT * FROM calculations WHERE calc_id=?",
                                (calc_id,)).fetchone())


def _manager(work, env, *args):
    env = dict(env, PYTHONPATH=str(REPO_ROOT))      # as the job scripts' callback has it
    return subprocess.run([sys.executable, str(MANAGER), "--db-path",
                           str(work / "materials.db"), *args],
                          cwd=str(work), env=env, capture_output=True, text=True, timeout=180)


def _assert_recovered(work, calc_id, log, r):
    text = r.stdout + r.stderr
    assert r.returncode == 0, text
    slurm = log.read_text() if log.exists() else ""
    assert f"scancel {JOB}" in slurm, text
    row = _row(work, calc_id)
    assert row["error_type"] == "finalrun_loop", (row, text)
    assert row["status"] == "resubmitted", (row, text)
    assert "sbatch" in slurm
    deck = (work / "si.d12").read_text()
    assert "\nTOLINTEG\n8 8 8 9 24\n" in deck and "\nSPIN\n" in deck


def test_mace_manager_recovers_a_running_loop(tmp_path):
    work, calc_id, env, log = _box(tmp_path, loop_output(end="running"))
    r = subprocess.run([sys.executable, str(MACE_CLI), "manager"], cwd=str(work), env=env,
                       capture_output=True, text=True, timeout=180)
    _assert_recovered(work, calc_id, log, r)


@pytest.mark.parametrize("mode", ["completion", "status_check", "early_failure"])
def test_callbacks_recover_a_running_loop(tmp_path, mode):
    work, calc_id, env, log = _box(tmp_path, loop_output(end="running"))
    env["SLURM_JOB_ID"] = "999777"            # another job's callback
    r = _manager(work, env, "--callback-mode", mode)
    _assert_recovered(work, calc_id, log, r)


def test_a_running_optimization_that_is_not_looping_is_left_alone(tmp_path):
    work, calc_id, env, log = _box(tmp_path, loop_output(restarts=2, end="running"))
    r = _manager(work, env, "--callback-mode", "completion")
    assert r.returncode == 0, r.stdout + r.stderr
    assert not log.exists() or "scancel" not in log.read_text()
    assert _row(work, calc_id)["status"] == "running"
