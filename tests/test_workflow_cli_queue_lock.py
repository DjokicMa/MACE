"""`mace workflow --process` and `--resume` start steps under the queue lock
the job callbacks hold (QUEUE_STATE_LOCK in the database's .queue_locks), so
a callback that is generating the same step cannot submit it a second time
(1.1.6 checkpoint review).

A background process holds the lock, standing in for a callback in flight;
real `mace_cli` subprocesses, tests/fake_slurm.py and a real OPT output."""
import json
import subprocess
import sys
import time

from array_helpers import rows
from conftest import REPO_ROOT
from test_array_default_path_golden import MACE_CLI
from test_workflow_spec_cli_e2e import _complete_without_callback, _json, _ok, _setup, _mace

HOLDER = """
import sys
from pathlib import Path
sys.path.insert(0, {repo!r})
from mace.queue.queue_lock_manager import QueueLockManager, QUEUE_STATE_LOCK
m = QueueLockManager(lock_dir=Path({lock_dir!r}), lock_timeout=300)
assert m.acquire_lock(QUEUE_STATE_LOCK, timeout=30)
print("held", flush=True)
sys.stdin.readline()
m.release_lock(QUEUE_STATE_LOCK)
print("released", flush=True)
"""


def _launched(tmp_path):
    out, deck, fake, env, root = _setup(tmp_path)
    (root / "spec.json").write_text(json.dumps({
        "version": 1, "inputs": {"type": "d12", "files": [deck.name]}, "template": "opt_sp"}))
    w = _json(_ok(_mace(root, env, "workflow", "--spec", "spec.json", "--write-plan", "--json")))
    wid = w["workflow_id"]
    _ok(_mace(root, env, "submit", "--progress-plan", w["plan_file"], deck.name))
    db = root / f".mace_context_{wid}" / "materials.db"
    opt = _complete_without_callback(db, "OPT", out)
    fake.remove_record(opt["slurm_job_id"])
    return fake, env, root, wid, db


def _waits_for_the_lock(fake, env, root, db, *args):
    holder = subprocess.Popen([sys.executable, "-c", HOLDER.format(repo=str(REPO_ROOT),
                                                                    lock_dir=str(db.parent / ".queue_locks"))],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        cmd = subprocess.Popen([sys.executable, str(MACE_CLI), "--no-banner", *args], cwd=str(root), env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        time.sleep(6)
        # the callback in flight holds the lock: nothing is submitted meanwhile
        assert cmd.poll() is None, cmd.communicate()
        assert len(fake.calls("sbatch")) == 1
        holder.stdin.write("\n")
        holder.stdin.flush()
        assert holder.stdout.readline().strip() == "released"
        out, err = cmd.communicate(timeout=600)
    finally:
        if holder.poll() is None:
            holder.kill()
    assert cmd.returncode == 0, out + err
    return json.loads(out)


def test_process_waits_for_the_callbacks_queue_lock(tmp_path):
    fake, env, root, wid, db = _launched(tmp_path)
    r = _waits_for_the_lock(fake, env, root, db, "workflow", "--process", wid, "--json")
    assert r["steps_started"] == 1
    assert len(fake.calls("sbatch")) == 2
    assert len([x for x in rows(db).values() if x["calc_type"] == "SP"]) == 1


def test_resume_waits_for_the_callbacks_queue_lock(tmp_path):
    fake, env, root, wid, db = _launched(tmp_path)
    _ok(_mace(root, env, "workflow", "--pause", wid, "--json"))
    r = _waits_for_the_lock(fake, env, root, db, "workflow", "--resume", wid, "--json")
    assert r["steps_started"] == 1 and r["state"] == "active"
    assert len(fake.calls("sbatch")) == 2
