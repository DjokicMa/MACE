"""A job that left the queue is classified from its full sacct state.

sacct without -P cuts the State column at 10 characters: OUT_OF_MEMORY
came back as OUT_OF_ME+ (read as OUT_OF_ME), not an end state, so a job
killed for memory before writing its .out stayed 'submitted' forever. The
real callback (manager.py --callback-mode status_check) with
tests/fake_slurm.py, whose sacct cuts the column as SLURM's does."""
import sqlite3
import sys

import pytest

from array_helpers import callback
from conftest import REPO_ROOT
from fake_slurm import FakeSlurm


@pytest.mark.parametrize("state, stored", [
    ("OUT_OF_MEMORY", "OUT_OF_MEMORY"), ("CANCELLED by 1234", "CANCELLED"),
    ("NODE_FAIL", "NODE_FAIL"), ("PREEMPTED", "PREEMPTED"), ("BOOT_FAIL", "BOOT_FAIL"),
    ("DEADLINE", "DEADLINE"), ("TIMEOUT", "TIMEOUT"), ("FAILED", "FAILED"),
])
def test_a_job_gone_with_no_output_is_failed_with_its_end_state(tmp_path, state, stored):
    fake = FakeSlurm(tmp_path / "slurm")
    fake.update(records=[], sacct={"5000": [["5000", state]]})
    sys.path.insert(0, str(REPO_ROOT))
    from mace.database.materials import MaterialDatabase
    work = tmp_path / "m1"
    work.mkdir()
    (work / "m1.d12").write_text("dummy\n")
    db_path = tmp_path / "materials.db"
    db = MaterialDatabase(str(db_path))
    db.create_material("m1", "m1")
    calc_id = db.create_calculation("m1", "SP", input_file=str(work / "m1.d12"), work_dir=str(work))
    db.update_calculation_status(calc_id, "submitted", slurm_job_id="5000")
    p = callback(fake.env, tmp_path, db_path, mode="status_check",
                 extra_env={"HOME": str(tmp_path), "USER": "tester"})
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    with sqlite3.connect(db_path) as conn:
        status, slurm_state, message = conn.execute(
            "SELECT status, slurm_state, error_message FROM calculations WHERE calc_id=?",
            (calc_id,)).fetchone()
    assert status == "failed", p.stdout[-3000:]
    assert slurm_state == stored
    assert stored in message, message


def _ended(tmp_path, state, ended_ago_days, extra_env=None):
    from datetime import datetime, timedelta
    fake = FakeSlurm(tmp_path / "slurm")
    end = (datetime.now() - timedelta(days=ended_ago_days)).strftime("%Y-%m-%dT%H:%M:%S")
    fake.update(records=[], sacct={"5000": [["5000", state, end]]})
    sys.path.insert(0, str(REPO_ROOT))
    from mace.database.materials import MaterialDatabase
    work = tmp_path / "m1"
    work.mkdir()
    (work / "m1.d12").write_text("dummy\n")
    db_path = tmp_path / "materials.db"
    db = MaterialDatabase(str(db_path))
    db.create_material("m1", "m1")
    calc_id = db.create_calculation("m1", "SP", input_file=str(work / "m1.d12"), work_dir=str(work))
    db.update_calculation_status(calc_id, "submitted", slurm_job_id="5000")
    p = callback(fake.env, tmp_path, db_path, mode="status_check",
                 extra_env=dict({"HOME": str(tmp_path), "USER": "tester"}, **(extra_env or {})))
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    with sqlite3.connect(db_path) as conn:
        status, slurm_state = conn.execute(
            "SELECT status, slurm_state FROM calculations WHERE calc_id=?", (calc_id,)).fetchone()
    return p.stdout + p.stderr, status, slurm_state, fake


def test_a_job_that_ended_days_ago_is_failed_but_not_recovered(tmp_path):
    """Upgrading: a row stuck 'submitted' since an old OUT_OF_MEMORY kill is
    recorded failed with its real state, and never resubmitted on its own."""
    out, status, slurm_state, fake = _ended(tmp_path, "OUT_OF_MEMORY", 10)
    assert (status, slurm_state) == ("failed", "OUT_OF_MEMORY"), out[-3000:]
    assert "Handling failed calculation" not in out, out[-3000:]
    assert fake.calls("sbatch") == []
    assert "job 5000 (m1) ended OUT_OF_MEMORY on" in out, out[-3000:]
    assert "not resubmitted automatically" in out and "mace recover" in out


def test_a_job_that_ended_recently_is_recovered_as_before(tmp_path):
    out, status, slurm_state, _ = _ended(tmp_path, "OUT_OF_MEMORY", 0.05)
    assert slurm_state == "OUT_OF_MEMORY", out[-3000:]
    assert "Handling failed calculation" in out, out[-3000:]
    assert "not resubmitted automatically" not in out


def test_the_age_limit_is_configurable(tmp_path):
    out, _, _, _ = _ended(tmp_path, "OUT_OF_MEMORY", 10, {"MACE_RECOVER_MAX_AGE_DAYS": "30"})
    assert "Handling failed calculation" in out, out[-3000:]


# A stale job is not resubmitted, but its failure is still classified so the
# `mace recover --db` the line points to has something to prepare (it used to
# be recorded 'failed' with no error type: "Prepared recovery for 0").


def _stale_with_deck(tmp_path, state, with_output):
    import shutil
    from datetime import datetime, timedelta
    from conftest import TEST_DATA
    oom_out = TEST_DATA / "FAILED_QA" / "3_dia3_opt_killed_signal9_oom.out"
    deck = TEST_DATA / "OPT" / "3_dia3_opt_rev1.d12"
    script = TEST_DATA / "OPT" / "3_dia3_opt_rev1.sh"
    if not (oom_out.exists() and deck.exists() and script.exists()):
        pytest.skip("needs the test/ corpus")
    fake = FakeSlurm(tmp_path / "slurm")
    end = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%dT%H:%M:%S")
    fake.update(records=[], sacct={"5000": [["5000", state, end]]})
    sys.path.insert(0, str(REPO_ROOT))
    from mace.database.materials import MaterialDatabase
    work = tmp_path / "m1"
    work.mkdir()
    shutil.copy(deck, work / deck.name)
    shutil.copy(script, work / script.name)
    if with_output:
        shutil.copy(oom_out, work / "3_dia3_opt_rev1.out")
    db_path = tmp_path / "materials.db"
    db = MaterialDatabase(str(db_path))
    db.create_material("3_dia3", "C")
    calc_id = db.create_calculation("3_dia3", "OPT", input_file=str(work / deck.name), work_dir=str(work))
    db.update_calculation_status(calc_id, "submitted", slurm_job_id="5000")
    env = {"HOME": str(tmp_path), "USER": "tester"}
    p = callback(fake.env, tmp_path, db_path, mode="status_check", extra_env=env)
    out = p.stdout + p.stderr
    assert p.returncode == 0, out[-3000:]
    assert "not resubmitted automatically" in out, out[-3000:]
    assert fake.calls("sbatch") == []
    with sqlite3.connect(db_path) as conn:
        status, error_type, slurm_state = conn.execute(
            "SELECT status, error_type, slurm_state FROM calculations WHERE calc_id=?",
            (calc_id,)).fetchone()
    import os
    import subprocess
    renv = dict(os.environ)
    renv.update(fake.env)
    renv.update(env)
    renv.pop("PYTHONPATH", None)
    # --prepare-only: 1.1.5's mace recover (1.1.6's plain one resubmits after a confirmation)
    r = subprocess.run([sys.executable, str(REPO_ROOT / "mace_cli"), "recover", "--action", "recover",
                        "--prepare-only", "--db", str(db_path)], cwd=work, stdin=subprocess.DEVNULL,
                       capture_output=True, text=True, timeout=300, env=renv)
    return status, error_type, slurm_state, r.stdout + r.stderr, fake


@pytest.mark.parametrize("with_output", [True, False])
def test_a_stale_job_is_classified_so_mace_recover_prepares_it(tmp_path, with_output):
    status, error_type, slurm_state, rec, fake = _stale_with_deck(tmp_path, "OUT_OF_MEMORY", with_output)
    assert (status, slurm_state) == ("failed", "OUT_OF_MEMORY")
    assert error_type == "memory_error", error_type
    assert "Prepared recovery for 1 calculation(s)" in rec, rec[-3000:]
    assert fake.calls("sbatch") == []


def test_a_stale_timeout_without_output_is_recoverable(tmp_path):
    status, error_type, slurm_state, rec, _ = _stale_with_deck(tmp_path, "TIMEOUT", False)
    assert (status, slurm_state, error_type) == ("failed", "TIMEOUT", "timeout_error")
    assert "Prepared recovery for 1 calculation(s)" in rec, rec[-3000:]
