"""`mace monitor --status` and `mace manager --status` with job array tasks:
task rows show their array and task, plain rows print as before, and the
queue counts every pending task of an array (as SLURM counts it)."""
import re

from array_helpers import SP_D12
from fake_slurm import FakeSlurm
from test_array_default_path_golden import _env, _mace


def _ok(p):
    assert p.returncode == 0, p.stdout[-4000:] + p.stderr[-2000:]
    return p


def test_monitor_status_shows_array_and_task(tmp_path):
    fake = FakeSlurm(tmp_path)
    env = _env(tmp_path, fake)
    d = tmp_path / "decks"
    d.mkdir()
    for n in ("m0", "m1"):
        (d / f"{n}.d12").write_text(SP_D12)
    _ok(_mace(tmp_path, env, "submit", "--array", "--track", "decks"))
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "p0.d12").write_text(SP_D12)
    _ok(_mace(tmp_path, env, "submit", "--track", "plain/p0.d12"))
    out = _ok(_mace(tmp_path, env, "monitor", "--status")).stdout
    assert re.search(r"m0_SP_\S+ \(m0\): submitted  \[array 4242, task \d\]", out), out
    assert re.search(r"\s p0_SP_\S+ \(p0\): submitted\n", out), out
    detailed = _ok(_mace(tmp_path, env, "monitor", "--status", "--detailed")).stdout
    assert "Job arrays: 1 arrays, 2 tasks" in detailed
    assert "[array 4242, task" in detailed


def test_monitor_status_without_arrays_says_nothing_of_them(tmp_path):
    fake = FakeSlurm(tmp_path)
    env = _env(tmp_path, fake)
    (tmp_path / "p0.d12").write_text(SP_D12)
    _ok(_mace(tmp_path, env, "submit", "--track", "p0.d12"))
    out = _ok(_mace(tmp_path, env, "monitor", "--status", "--detailed")).stdout
    assert "Job arrays" not in out and "[array" not in out


def test_manager_status_counts_every_task(tmp_path):
    fake = FakeSlurm(tmp_path, {"throttle": {"4242": 2}, "records": [
        {"id": "4242_0", "array": "4242", "task": 0, "state": "RUNNING"},
        {"id": "4242_1", "array": "4242", "task": 1, "state": "PENDING"},
        {"id": "4242_3", "array": "4242", "task": 3, "state": "PENDING"},
        {"id": "4242_4", "array": "4242", "task": 4, "state": "PENDING"},
        {"id": "777", "state": "RUNNING"}]})
    env = _env(tmp_path, fake)
    out = _ok(_mace(tmp_path, env, "manager", "--status")).stdout
    assert "Running Jobs: 2" in out and "Pending Jobs: 3" in out
