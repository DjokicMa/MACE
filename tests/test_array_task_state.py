"""How the job callback judges job array tasks (rows A_T), run as a subprocess
the way a job script runs it, against the fake SLURM.

A task is classified only on positive evidence: its own output written after
its submission, a queue or accounting record that names it, or - for a task
canceled while pending, of which SLURM keeps no record at all - two sweeps at
least ten minutes apart that both find nothing of it. Never from a sibling's
output in a shared directory, never from an older output of the same deck,
and never from a task list SLURM printed cut short.
"""
import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

from array_helpers import (SP_D12, SP_DONE, age, callback, make_task_rows, rows,
                           shift_absent_since, task_env)
from conftest import REPO_ROOT
from fake_slurm import FakeSlurm


@pytest.fixture
def shared(tmp_path):
    """Four tasks of array 4242 sharing one directory (mace submit DIR --array)."""
    d = tmp_path / "decks"
    fake = FakeSlurm(tmp_path / "slurm")
    db, ids = make_task_rows(tmp_path / "materials.db", [d] * 4)
    return tmp_path, d, fake, db, ids


def _status(tmp_path, ids):
    r = rows(tmp_path / "materials.db")
    return [r[i]["status"] for i in ids]


def test_pending_bracket_keeps_rows_and_never_asks_sacct(shared):
    tmp_path, d, fake, db, ids = shared
    fake.update(records=[
        {"id": "4242_0", "array": "4242", "task": 0, "state": "RUNNING"},
        *({"id": f"4242_{t}", "array": "4242", "task": t, "state": "PENDING"} for t in (1, 2, 3))],
        throttle={"4242": 2})
    # an OLD finished output of task 1's deck sits in the directory
    (d / "mat1.out").write_text(SP_DONE)
    age(d / "mat1.out", 3600)
    p = callback(fake.env, d, tmp_path / "materials.db")
    assert p.returncode == 0, p.stdout + p.stderr
    assert _status(tmp_path, ids) == ["running", "submitted", "submitted", "submitted"]
    assert not fake.calls("sacct")


def test_old_output_of_the_same_deck_is_not_this_task(shared):
    """The critical guard: the array has left the queue, sacct knows
    nothing, and the deck's output is from an earlier run."""
    tmp_path, d, fake, db, ids = shared
    (d / "mat1.out").write_text(SP_DONE)
    age(d / "mat1.out", 3600)
    p = callback(fake.env, d, tmp_path / "materials.db")
    assert p.returncode == 0, p.stdout + p.stderr
    r = rows(tmp_path / "materials.db")
    assert r[ids[1]]["status"] == "submitted"
    # sacct knows nothing of the array either: no evidence of any kind,
    # so not even the absence clock starts
    assert not json.loads(r[ids[1]]["settings_json"])["slurm_array"].get("absent_since")


def test_canceled_while_pending_in_a_shared_dir(shared):
    """Siblings 0-2 finished in the same directory, task 3 was canceled
    while pending: SLURM has no record of it anywhere."""
    tmp_path, d, fake, db, ids = shared
    for t in range(3):
        (d / f"mat{t}.out").write_text(SP_DONE)
    fake.update(sacct={f"4242_{t}": "COMPLETED" for t in range(3)},
                sacct_array={"4242": [[f"4242_{t}", "COMPLETED"] for t in range(3)]})
    p = callback(fake.env, d, tmp_path / "materials.db")
    assert p.returncode == 0, p.stdout + p.stderr
    assert _status(tmp_path, ids) == ["completed", "completed", "completed", "submitted"]
    r = rows(tmp_path / "materials.db")
    assert json.loads(r[ids[3]]["settings_json"])["slurm_array"]["absent_since"]
    # second sweep, still inside the window: kept
    p = callback(fake.env, d, tmp_path / "materials.db")
    assert _status(tmp_path, ids)[3] == "submitted"
    # second sweep after the window
    shift_absent_since(tmp_path / "materials.db", ids[3], 11)
    p = callback(fake.env, d, tmp_path / "materials.db")
    assert p.returncode == 0, p.stdout + p.stderr
    r = rows(tmp_path / "materials.db")[ids[3]]
    assert r["status"] == "failed"
    assert r["error_type"] == "no_output"
    assert "mat3.out" in r["error_message"]
    assert r["output_file"] is None, "never a sibling's output"


def test_task_seen_again_clears_the_absence(shared):
    tmp_path, d, fake, db, ids = shared
    fake.update(sacct_array={"4242": [["4242_0", "COMPLETED"]]})
    callback(fake.env, d, tmp_path / "materials.db")
    r = rows(tmp_path / "materials.db")[ids[2]]
    assert json.loads(r["settings_json"])["slurm_array"].get("absent_since")
    fake.update(sacct_array={"4242": [["4242_0", "COMPLETED"], ["4242_2", "RUNNING"]]})
    callback(fake.env, d, tmp_path / "materials.db")
    r = rows(tmp_path / "materials.db")[ids[2]]
    assert r["status"] == "running"
    assert "absent_since" not in json.loads(r["settings_json"])["slurm_array"]


def test_dbd_lag_own_output_completes_the_row(shared):
    """The task ended moments ago: no queue entry, no accounting record yet,
    but its own output is there and newer than its submission."""
    tmp_path, d, fake, db, ids = shared
    (d / "mat2.out").write_text(SP_DONE)
    p = callback(fake.env, d, tmp_path / "materials.db")
    assert p.returncode == 0, p.stdout + p.stderr
    r = rows(tmp_path / "materials.db")
    assert r[ids[2]]["status"] == "completed"
    assert r[ids[2]]["output_file"] == str(d / "mat2.out")
    assert [r[i]["status"] for i in (ids[0], ids[1], ids[3])] == ["submitted"] * 3


def test_comma_bracket_rows_stay_queued(tmp_path):
    d = tmp_path / "decks"
    fake = FakeSlurm(tmp_path / "slurm", {"throttle": {"4242": 3}, "records": [
        {"id": f"4242_{t}", "array": "4242", "task": t, "state": "PENDING"} for t in (1, 4, 5, 6)]})
    db, ids = make_task_rows(tmp_path / "materials.db", [d] * 7)
    p = callback(fake.env, d, tmp_path / "materials.db")
    assert p.returncode == 0, p.stdout + p.stderr
    out = subprocess.run(["squeue", "-o", "%i,%T,%S"], env=dict(fake.env, SLURM_BITSTR_LEN="0"),
                         capture_output=True, text=True).stdout
    assert "4242_[1,4-6%3],PENDING" in out
    st = dict(zip(range(7), _status(tmp_path, ids)))
    assert [st[t] for t in (1, 4, 5, 6)] == ["submitted"] * 4
    asked = {c["argv"][2] for c in fake.calls("sacct")}
    assert not asked & {"4242_1", "4242_4", "4242_5", "4242_6"}


def test_truncated_task_list_changes_nothing(shared):
    tmp_path, d, fake, db, ids = shared
    fake.update(records=[{"id": f"4242_{t}", "array": "4242", "task": t, "state": "PENDING"}
                         for t in (1, 3)],
                bracket_force={"4242": "4242_[1,3,5,..."},
                sacct={f"4242_{t}": "CANCELLED" for t in range(4)})
    for t in range(4):
        (d / f"mat{t}.out").write_text(SP_DONE)
    before = rows(tmp_path / "materials.db")
    p = callback(fake.env, d, tmp_path / "materials.db")
    assert p.returncode == 0, p.stdout + p.stderr
    assert rows(tmp_path / "materials.db") == before, p.stdout
    assert not fake.calls("sacct")


def test_task_callback_completes_its_own_row(shared):
    tmp_path, d, fake, db, ids = shared
    fake.update(records=[{"id": "4242_1", "array": "4242", "task": 1, "state": "RUNNING"},
                         {"id": "4242_0", "array": "4242", "task": 0, "state": "RUNNING"}])
    (d / "mat1.out").write_text(SP_DONE)
    (d / "mat0.out").write_text(SP_DONE)  # a sibling finished too; it is still RUNNING
    p = callback(fake.env, d, tmp_path / "materials.db", mode="completion",
                 extra_env=task_env("4242", 1, 9999))
    assert p.returncode == 0, p.stdout + p.stderr
    st = _status(tmp_path, ids)
    assert st[1] == "completed"
    assert st[0] == "running"


def test_last_task_raw_id_equal_to_the_array_id_is_never_stored(shared):
    """The last task to start runs with SLURM_JOB_ID = A (measured)."""
    tmp_path, d, fake, db, ids = shared
    fake.update(records=[{"id": "4242_3", "array": "4242", "task": 3, "state": "RUNNING",
                          "raw": "4242"}])
    (d / "mat3.out").write_text(SP_DONE)
    p = callback(fake.env, d, tmp_path / "materials.db", mode="completion",
                 extra_env=task_env("4242", 3, 4242))
    assert p.returncode == 0, p.stdout + p.stderr
    r = rows(tmp_path / "materials.db")
    assert r[ids[3]]["status"] == "completed"
    assert "4242" not in {x["slurm_job_id"] for x in r.values()}


def test_child_job_of_a_task_does_not_take_the_task_row(shared):
    """A follow-up submitted from inside task 1 inherits its MACE_ARRAY_* and
    SLURM_ARRAY_* variables (measured); it must not classify 4242_1."""
    tmp_path, d, fake, db, ids = shared
    fake.update(records=[{"id": "4242_1", "array": "4242", "task": 1, "state": "RUNNING"},
                         {"id": "6000", "state": "RUNNING"}])
    (d / "mat1.out").write_text(SP_DONE)
    env = dict(task_env("4242", 1, 9999), SLURM_JOB_ID="6000")
    p = callback(fake.env, d, tmp_path / "materials.db", mode="completion", extra_env=env)
    assert p.returncode == 0, p.stdout + p.stderr
    assert _status(tmp_path, ids)[1] == "running"


def test_task_finds_a_row_not_yet_recorded_through_the_manifest(tmp_path):
    """Without --hold a task can end before its row carries A_T: the
    manifest names its calc_id."""
    d = tmp_path / "decks"
    d.mkdir()
    fake = FakeSlurm(tmp_path / "slurm")
    sys.path.insert(0, str(REPO_ROOT))
    from mace.database.materials import MaterialDatabase
    db = MaterialDatabase(str(tmp_path / "materials.db"))
    (d / "mat0.d12").write_text(SP_D12)
    calc_id = db.create_calculation(material_id="mat0", calc_type="SP",
                                    input_file=str(d / "mat0.d12"), work_dir=str(d))
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("# MACE job array manifest v1: task\tdir\tjob\tscript\tcalc_id\n"
                        f"0\t{d}\tmat0\t{d}/mat0.sh\t{calc_id}\n")
    (d / "mat0.out").write_text(SP_DONE)
    p = callback(fake.env, d, tmp_path / "materials.db", mode="completion",
                 extra_env=task_env("4242", 0, 9999, manifest=manifest, calc_id=calc_id))
    assert p.returncode == 0, p.stdout + p.stderr
    assert rows(tmp_path / "materials.db")[calc_id]["status"] == "completed"


def _workflow_root(tmp_path, n):
    root = tmp_path / "wf"
    for sub in ("workflow_configs", "workflow_scripts", "workflow_inputs"):
        (root / sub).mkdir(parents=True)
    dirs = [root / "workflow_outputs" / "workflow_x" / "step_001_SP" / f"mat{i}" for i in range(n)]
    return root, dirs


@pytest.mark.parametrize("hold_seconds,scale,expect_fallback",
                         [(3, "0.1", False), (25, "0.02", True)])
def test_many_task_callbacks_at_once(tmp_path, hold_seconds, scale, expect_fallback):
    """20 tasks of one array end together. Their callbacks contend for the
    lock while another holder keeps it: they wait with backoff (timings
    scaled down here) and, when it never comes free, record their own row
    without it. No row is left 'submitted' or 'running', and a later sweep
    chains every completed row."""
    n = 20
    root, dirs = _workflow_root(tmp_path, n)
    fake = FakeSlurm(tmp_path / "slurm")
    db_path = root / "materials.db"
    db, ids = make_task_rows(db_path, dirs)
    fake.update(records=[{"id": f"4242_{t}", "array": "4242", "task": t, "state": "RUNNING"}
                         for t in range(n)])
    for t, wd in enumerate(dirs):
        (wd / f"mat{t}.out").write_text(SP_DONE)
    sys.path.insert(0, str(REPO_ROOT))
    from mace.queue.queue_lock_manager import QueueLockManager
    holder = QueueLockManager(lock_dir=root / ".queue_locks", lock_timeout=300)
    assert holder.acquire_lock("queue_manager_completion", timeout=5)
    procs = []
    try:
        for t, wd in enumerate(dirs):
            procs.append(callback(fake.env, wd, db_path, mode="completion", background=True,
                                  extra_env=dict(task_env("4242", t, 9000 + t),
                                                 MACE_ARRAY_LOCK_SCALE=scale)))
            time.sleep(0.05)
        time.sleep(hold_seconds)
    finally:
        holder.release_lock("queue_manager_completion")
    outs = [p.communicate(timeout=600)[0] for p in procs]
    assert all(p.returncode == 0 for p in procs), outs
    fell_back = sum("recording this task's own result without it" in o for o in outs)
    assert (fell_back > 0) == expect_fallback, fell_back
    assert not any("Could not acquire lock" in o for o in outs)
    # nothing is left waiting
    r = rows(db_path)
    assert all(r[i]["status"] == "completed" for i in ids), [r[i]["status"] for i in ids]
    # a later sweep (any callback) chains whatever is not processed yet
    p = callback(fake.env, root, db_path, mode="completion")
    assert p.returncode == 0, p.stdout + p.stderr
    r = rows(db_path)
    assert all(json.loads(r[i]["settings_json"]).get("workflow_processed") for i in ids)
