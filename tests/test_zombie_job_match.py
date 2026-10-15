"""`mace completion --remove-zombie-jobs` cancels only the job of the
calculation it looked at.

The job id came from a glob of *<name>*.o, so the finished `si` took
`si_sp-19000002.o` (a different, healthy job still running) as its own and
scancel killed it. Now only the calculation's own logs count
(<name>-<id>.o, <name>_<id>.o, <name>.o<id>, <name>-<A>_<T>.o), the newest
wins, and SLURM must confirm the job's name and working directory before
anything is canceled. The real CLI with tests/fake_slurm.py and a real
CRYSTAL output from the corpus."""
import shutil
from pathlib import Path

from conftest import find_data
from fake_slurm import FakeSlurm
from test_array_default_path_golden import _env, _mace


def _setup(tmp_path):
    fake = FakeSlurm(tmp_path / "slurm")
    env = _env(tmp_path, fake)
    w = tmp_path / "w"
    w.mkdir()
    done = find_data("SP/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized.out")
    shutil.copy(done, w / "si.out")                                # finished
    (w / "si_sp.out").write_text(Path(done).read_text()[:3000])    # still running
    return fake, env, w


def _zombies(tmp_path, env, w):
    p = _mace(tmp_path, env, "completion", "--remove-zombie-jobs", cwd=w, stdin="y\nn\n")
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    return p.stdout + p.stderr


def _canceled(fake):
    return [c["argv"][1:] for c in fake.calls("scancel")]


def test_another_jobs_log_is_not_this_calculations_job(tmp_path):
    """si_sp has just started (no si_sp.out yet); si finished long ago."""
    fake, env, w = _setup(tmp_path)
    (w / "si_sp.out").unlink()
    (w / "si_sp-19000002.o").write_text("")
    fake.update(records=[{"id": "19000002", "state": "RUNNING", "name": "si_sp",
                          "workdir": str(w)}])
    out = _zombies(tmp_path, env, w)
    assert _canceled(fake) == [], out
    assert "No zombie jobs detected" in out, out


def test_the_calculations_own_running_job_is_canceled(tmp_path):
    fake, env, w = _setup(tmp_path)
    (w / "si-19000001.o").write_text("")
    (w / "si_sp-19000002.o").write_text("")
    fake.update(records=[{"id": "19000001", "state": "RUNNING", "name": "si", "workdir": str(w)},
                         {"id": "19000002", "state": "RUNNING", "name": "si_sp",
                          "workdir": str(w)}])
    out = _zombies(tmp_path, env, w)
    assert ["19000001"] in _canceled(fake), out
    assert all("19000002" not in c for c in _canceled(fake)), out


def test_the_newest_of_its_own_logs_wins(tmp_path):
    """A restart leaves si-19000001.o and si-19000005.o: the running job is
    the newer one, whatever order the folder lists them in."""
    fake, env, w = _setup(tmp_path)
    for jid in ("19000005", "19000001"):
        (w / f"si-{jid}.o").write_text("")
    fake.update(records=[{"id": "19000005", "state": "RUNNING", "name": "si", "workdir": str(w)}])
    out = _zombies(tmp_path, env, w)
    assert _canceled(fake) == [["19000005"]], out


def test_a_job_slurm_does_not_confirm_is_never_canceled(tmp_path):
    """The id in the log name is running, but SLURM says it is another job
    (another name, or another folder): left alone, with the reason."""
    fake, env, w = _setup(tmp_path)
    (w / "si-19000001.o").write_text("")
    other = tmp_path / "elsewhere"
    other.mkdir()
    fake.update(records=[{"id": "19000001", "state": "RUNNING", "name": "si", "workdir": str(other)}])
    out = _zombies(tmp_path, env, w)
    assert _canceled(fake) == [], out
    assert "Not canceled 19000001" in out, out
