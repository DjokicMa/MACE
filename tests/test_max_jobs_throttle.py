"""The max-jobs throttle counts the jobs really in the queue.

It counted the entries of crystal_job_status.json, which are never pruned
and were also read from ~ and the temp folder: 220 stale entries from an
old run blocked every new submission ("Queue nearly full (220/250)") with
an empty queue, while a really full queue, unknown to that file, did not
stop anything. The real callback (manager.py --callback-mode submit_new)
with tests/fake_slurm.py."""
import json

from array_helpers import SP_D12, callback
from fake_slurm import FakeSlurm


def _setup(tmp_path, stale=0, queued=0):
    fake = FakeSlurm(tmp_path / "slurm")
    home = tmp_path / "home"
    home.mkdir()
    if stale:
        (home / "crystal_job_status.json").write_text(json.dumps(
            {"submitted": {str(1000 + i): {"calc_id": "x"} for i in range(stale)},
             "pending": [], "completed": []}))
    fake.update(records=[{"id": str(5000 + i), "state": "RUNNING" if i % 2 else "PENDING",
                          "user": "tester"} for i in range(queued)])
    w = tmp_path / "w"
    w.mkdir()
    (w / "newmat.d12").write_text(SP_D12)
    # the manager's own limit when no --max-jobs is given is the queue
    # budget (mace/queue/budget.py): 225 queued of max_jobs 250
    (tmp_path / "site.toml").write_text("[queue]\nmax_jobs = 250\nreserve = 25\n")
    env = {"HOME": str(home), "USER": "tester", "TMPDIR": str(tmp_path / "tmp"),
           "MACE_SITE_CONFIG": str(tmp_path / "site.toml")}
    (tmp_path / "tmp").mkdir()
    return fake, w, env


def _submit_new(fake, w, env):
    p = callback(fake.env, w, w / "materials.db", mode="submit_new", extra_env=env)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    return p.stdout + p.stderr


def test_stale_status_entries_do_not_block_an_empty_queue(tmp_path):
    fake, w, env = _setup(tmp_path, stale=220)
    out = _submit_new(fake, w, env)
    assert "Queue nearly full" not in out, out[-3000:]
    assert len(fake.calls("sbatch")) == 1, out[-3000:]


def test_a_full_queue_stops_submission(tmp_path):
    fake, w, env = _setup(tmp_path, queued=225)
    out = _submit_new(fake, w, env)
    assert "Queue nearly full (225/250)" in out, out[-3000:]
    assert fake.calls("sbatch") == [], out[-3000:]


def test_without_squeue_the_databases_active_jobs_count(tmp_path):
    """squeue fails: the database's submitted/running rows are counted (here
    225 of them), not the status file."""
    import sys
    from conftest import REPO_ROOT
    sys.path.insert(0, str(REPO_ROOT))
    from mace.database.materials import MaterialDatabase
    fake, w, env = _setup(tmp_path, stale=0)
    db = MaterialDatabase(str(w / "materials.db"))
    for i in range(225):
        cid = db.create_calculation(material_id=f"m{i}", calc_type="SP",
                                    input_file=str(w / f"m{i}.d12"), work_dir=str(w))
        db.update_calculation_status(cid, "running" if i % 2 else "submitted",
                                     slurm_job_id=str(7000 + i))
    fake.update(fail={"squeue": "slurm_load_jobs error: Socket timed out"})
    out = _submit_new(fake, w, env)
    assert "counting this database's active jobs" in out, out[-3000:]
    assert "Queue nearly full (225/250)" in out, out[-3000:]
    assert fake.calls("sbatch") == [], out[-3000:]
