"""Shared pieces of the job-array tests: decks, finished outputs, task rows
and the job callback run the way a job script runs it."""
import json
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from conftest import REPO_ROOT

MACE_CLI = REPO_ROOT / "mace_cli"
MANAGER = REPO_ROOT / "mace" / "queue" / "manager.py"

SP_D12 = """TESTGEOM
CRYSTAL
0 0 0
227
3.5
1
6 0.125 0.125 0.125
END
6 2
0 0 2 2. 1.
 8.0 1.0
 2.0 1.0
0 1 1 4. 1.
 1.0 1.0 1.0
99 0
END
DFT
B3LYP
END
SHRINK
8 8
END
"""
OPT_D12 = SP_D12.replace("6 0.125 0.125 0.125\nEND\n",
                         "6 0.125 0.125 0.125\nOPTGEOM\nFULLOPTG\nENDOPT\nEND\n", 1)

# The end of a finished CRYSTAL SP output (the lines the completion check reads).
SP_DONE = (" == SCF ENDED - CONVERGENCE ON ENERGY      E(AU) -7.5848990805E+01 CYCLES  12\n"
           " EEEEEEEEEE TERMINATION  DATE 04 10 2026 TIME 12:53:46.2\n"
           "    TOTAL CPU TIME =     12.401\n")


def task_env(array_id, task, raw, manifest=None, calc_id=None):
    env = {"SLURM_JOB_ID": str(raw), "SLURM_ARRAY_JOB_ID": str(array_id),
           "SLURM_ARRAY_TASK_ID": str(task), "SLURM_ARRAY_TASK_COUNT": "4",
           "MACE_ARRAY_RAW_ID": str(raw), "MACE_ARRAY_TASK": f"{array_id}_{task}"}
    if manifest:
        env["MACE_ARRAY_MANIFEST"] = str(manifest)
    if calc_id:
        env["MACE_ARRAY_CALC_ID"] = calc_id
    return env


def callback(fake_env, cwd, db, mode="status_check", extra_env=None, timeout=600,
             background=False):
    """manager.py as the job script's last line runs it (install pinned)."""
    env = dict(fake_env, PYTHONPATH=str(REPO_ROOT), MACE_HOME=str(REPO_ROOT))
    env.update(extra_env or {})
    cmd = [sys.executable, str(MANAGER), "--callback-mode", mode, "--db-path", str(db)]
    if background:
        return subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
    return subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True,
                          timeout=timeout)


def make_task_rows(db_path, work_dirs, array_id="4242", calc_type="SP",
                   names=None, settings=None):
    """One submitted row per task, as `mace submit --array --track` records them."""
    sys.path.insert(0, str(REPO_ROOT))
    from mace.database.materials import MaterialDatabase
    db = MaterialDatabase(str(db_path))
    ids = []
    for t, wd in enumerate(work_dirs):
        name = (names or [f"mat{i}" for i in range(len(work_dirs))])[t]
        wd = Path(wd)
        wd.mkdir(parents=True, exist_ok=True)
        (wd / f"{name}.d12").write_text(SP_D12)
        calc_id = db.create_calculation(material_id=name, calc_type=calc_type,
                                        input_file=str(wd / f"{name}.d12"), work_dir=str(wd),
                                        settings=dict(settings or {}))
        db.update_calculation_status(calc_id, "submitted", slurm_job_id=f"{array_id}_{t}")
        db.update_calculation_settings(calc_id, {"slurm_array": {
            "array_job_id": array_id, "task": t, "tag": "arr_test_0",
            "array_dir": str(Path(db_path).parent / ".mace_arrays" / "arr_test_0")}},
            merge=True)
        ids.append(calc_id)
    return db, ids


def rows(db_path):
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        return {r["calc_id"]: dict(r) for r in conn.execute("SELECT * FROM calculations")}


def age(path, seconds):
    """Make a file look ``seconds`` old."""
    t = time.time() - seconds
    os.utime(path, (t, t))


def shift_absent_since(db_path, calc_id, minutes):
    with sqlite3.connect(db_path) as conn:
        (settings,) = conn.execute("SELECT settings_json FROM calculations WHERE calc_id=?",
                                   (calc_id,)).fetchone()
        s = json.loads(settings)
        s["slurm_array"]["absent_since"] = (
            datetime.fromisoformat(s["slurm_array"]["absent_since"])
            - timedelta(minutes=minutes)).isoformat()
        conn.execute("UPDATE calculations SET settings_json=? WHERE calc_id=?",
                     (json.dumps(s), calc_id))
