"""The completion callback never loses its update and never submits twice
when the database is busy (PLAN_analysis C15).

The real job callback (manager.py, as the job script's last line runs it)
in a real `mace submit --progress` folder, with the shared FakeSlurm
(tests/fake_slurm.py, which logs every sbatch) and a real CRYSTAL output
copied from the corpus. The duplicate case uses the test hook
MACE_TEST_HOOKS=1 + MACE_TEST_LOCKED_AFTER_SBATCH=1: a 'database is
locked' error (SQLITE_BUSY) right after the first sbatch."""
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from array_helpers import callback, rows, task_env
from conftest import REPO_ROOT, find_data
from fake_slurm import FakeSlurm
from test_array_default_path_golden import _env, _mace



def _ok(p):
    assert p.returncode == 0, p.stdout[-5000:] + p.stderr[-3000:]
    return p


def _context_db(root):
    (db,) = root.glob(".mace_context_*/materials.db")
    return db


def _submitted_opt(tmp_path, array=False):
    """`mace submit --progress opt_sp` on one copied real OPT deck; its
    finished real output put in place. Returns (fake, env, decks, db, row, exports)."""
    out = find_data("OPT/1_dia_opt_rev1.out")
    deck = find_data("OPT/1_dia_opt_rev1.d12")
    fake = FakeSlurm(tmp_path)
    env = _env(tmp_path, fake)
    d = tmp_path / "decks"
    d.mkdir()
    names = ("dia0", "dia1") if array else ("dia0",)
    for n in names:
        shutil.copy(deck, d / f"{n}.d12")
    _ok(_mace(tmp_path, env, "submit", "--progress", "opt_sp", *(["--array"] if array else []), "decks"))
    db = _context_db(d)
    r = list(rows(db).values())
    row = sorted(r, key=lambda x: x["slurm_job_id"])[0]
    stem = Path(row["input_file"]).stem
    shutil.copy(out, Path(row["work_dir"] or d) / f"{stem}.out")
    script = (Path(row["work_dir"] or d) / f"{stem}.sh").read_text()
    exports = dict(re.findall(r'^export (MACE_WORKFLOW_ID|MACE_CONTEXT_DIR|MACE_ISOLATION_MODE)="([^"]*)"', script, re.M))
    for x in r:
        fake.set_record(x["slurm_job_id"], state="RUNNING")
    return fake, env, d, db, row, exports


def test_locked_after_sbatch_submits_once_plain_job(tmp_path):
    fake, env, d, db, row, exports = _submitted_opt(tmp_path)
    before = len(fake.calls("sbatch"))
    p = _ok(callback(env, d, db, mode="completion",
                     extra_env=dict(exports, SLURM_JOB_ID=row["slurm_job_id"], MACE_TEST_HOOKS="1",
                                    MACE_TEST_LOCKED_AFTER_SBATCH="1")))
    out = p.stdout + p.stderr
    sb = fake.calls("sbatch")[before:]
    assert len(sb) == 1, out[-4000:]
    assert rows(db)[row["calc_id"]]["status"] == "completed", out[-4000:]
    assert "A follow-up was already submitted in this callback" in out, out[-4000:]
    assert "Fall back" not in out


def test_locked_after_sbatch_submits_once_array_task(tmp_path):
    fake, env, d, db, row, exports = _submitted_opt(tmp_path, array=True)
    before = len(fake.calls("sbatch"))
    array_id, task = row["slurm_job_id"].split("_")
    p = _ok(callback(env, d, db, mode="completion",
                     extra_env=dict(task_env(array_id, task, 9100), **exports, MACE_ARRAY_LOCK_SCALE="0.01",
                                    MACE_TEST_HOOKS="1", MACE_TEST_LOCKED_AFTER_SBATCH="1")))
    out = p.stdout + p.stderr
    sb = fake.calls("sbatch")[before:]
    assert len(sb) == 1, out[-4000:]
    assert rows(db)[row["calc_id"]]["status"] == "completed", out[-4000:]
    assert "A follow-up was already submitted in this callback" in out, out[-4000:]


HOLD = textwrap.dedent("""
    import sqlite3, sys, time
    c = sqlite3.connect(sys.argv[1], isolation_level=None, timeout=60)
    c.execute('BEGIN EXCLUSIVE')
    print('held', flush=True)
    time.sleep(float(sys.argv[2]))
    c.execute('ROLLBACK')
    """)


def test_exclusive_lock_for_forty_seconds_then_the_callback_completes(tmp_path):
    fake, env, d, db, row, exports = _submitted_opt(tmp_path)
    holder = subprocess.Popen([sys.executable, "-c", HOLD, str(db), "40"], stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        t0 = time.time()
        p = _ok(callback(env, d, db, mode="completion",
                         extra_env=dict(exports, SLURM_JOB_ID=row["slurm_job_id"])))
    finally:
        holder.wait(timeout=120)
    out = p.stdout + p.stderr
    assert rows(db)[row["calc_id"]]["status"] == "completed", out[-4000:]
    # never an unlocked second run: every callback run was one of the locked attempts
    assert "Fall back" not in out and "Error in callback with locking" not in out, out[-4000:]
    assert out.count("=== Queue Manager Callback") <= 6
    assert len(fake.calls("sbatch")) >= 2           # the OPT, then exactly one SP follow-up
    sps = [x for x in rows(db).values() if x["calc_type"].startswith("SP")]
    assert len(sps) == 1, out[-4000:]
    print(f"callback finished {time.time() - t0:.0f} s after a 40 s exclusive lock")


QUEUE_LOCK = textwrap.dedent("""
    import sys, time
    sys.path.insert(0, sys.argv[3])
    from mace.queue.queue_lock_manager import QueueLockManager
    from pathlib import Path
    m = QueueLockManager(lock_dir=Path(sys.argv[1]))
    assert m.acquire_lock('queue_manager_completion', 10)
    print('held', flush=True)
    time.sleep(float(sys.argv[2]))
    m.release_lock('queue_manager_completion')
    """)


def test_queue_lock_held_records_own_row_and_submits_nothing(tmp_path):
    fake, env, d, db, row, exports = _submitted_opt(tmp_path)
    lock_dir = Path(db).parent / ".queue_locks"
    holder = subprocess.Popen([sys.executable, "-c", QUEUE_LOCK, str(lock_dir), "30", str(REPO_ROOT)],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                              env=dict(os.environ, PYTHONPATH=str(REPO_ROOT)))
    try:
        line = holder.stdout.readline().strip()
        assert line == "held", holder.stderr.read()
        before = len(fake.calls("sbatch"))
        p = _ok(callback(env, d, db, mode="completion",
                         extra_env=dict(exports, SLURM_JOB_ID=row["slurm_job_id"], MACE_ARRAY_LOCK_SCALE="0.05")))
    finally:
        holder.kill()
        holder.wait()
    out = p.stdout + p.stderr
    assert rows(db)[row["calc_id"]]["status"] == "completed", out[-4000:]
    assert "next sweep" in out
    assert len(fake.calls("sbatch")) == before, out[-4000:]


def test_every_sbatch_site_counts_its_submission():
    """A new sbatch argv under mace/ must call note_submission() after it
    succeeds, or the callback could re-run after submitting (FR9)."""
    root = REPO_ROOT / "mace"
    argv = re.compile(r"""\[\s*['"]sbatch['"]\s*,""")
    # a dry run, not a submission: SLURM answers sbatch --test-only without queuing anything
    exempt = {("recovery/slurm_limits.py", "--test-only"), ("cluster/fit.py", "--test-only"),
              # the MACE GUI's own job (mace gui --slurm): no calculation, never run by a callback
              ("gui/slurmsession.py", "--job-name={JOB_NAME}")}
    missing = []
    for f in sorted(root.rglob("*.py")):
        lines = f.read_text(errors="replace").splitlines()
        rel = f.relative_to(root).as_posix()
        for i, line in enumerate(lines):
            if not argv.search(line) or line.strip().startswith("#"):
                continue
            if any(rel == r and tag in line for r, tag in exempt):
                continue
            if "_write_submitted(" in line:          # the record of an argv, not a call
                continue
            window = "\n".join(lines[i:i + 22])
            if "note_submission(" not in window:
                missing.append(f"{rel}:{i + 1}")
    assert not missing, missing


def test_an_error_after_a_submission_never_reruns_the_callback(monkeypatch, capsys):
    """Any error (not only a busy database) raised after the callback
    submitted a follow-up: the old fallback re-ran the whole callback
    without the lock, which would submit that follow-up again."""
    from mace.queue import manager as M
    from mace.queue import slurm_ids

    class Locks:
        def with_lock(self, name, fn, *a, **k):
            return fn(*a)

    runs, own = [], []
    mgr = M.EnhancedCrystalQueueManager.__new__(M.EnhancedCrystalQueueManager)
    mgr.throttler, mgr.lock_manager = None, Locks()
    monkeypatch.setattr(M, "verified_array_task", lambda: None)
    monkeypatch.setattr(mgr, "classify_own_row", lambda: own.append(1), raising=False)

    def callback_that_submits_then_fails(mode):
        runs.append(mode)
        slurm_ids.note_submission()
        raise RuntimeError("something unrelated broke")
    monkeypatch.setattr(mgr, "_run_callback_check_locked", callback_that_submits_then_fails, raising=False)
    mgr.run_callback_check("completion")
    assert runs == ["completion"] and own == [1]
    assert "A follow-up was already submitted" in capsys.readouterr().out

    # an error before any submission keeps the old fallback (one unlocked run)
    runs.clear()

    def callback_that_fails(mode):
        runs.append(mode)
        if len(runs) == 1:
            raise RuntimeError("lock file problem")
    monkeypatch.setattr(mgr, "_run_callback_check_locked", callback_that_fails, raising=False)
    mgr.run_callback_check("completion")
    assert runs == ["completion", "completion"]
