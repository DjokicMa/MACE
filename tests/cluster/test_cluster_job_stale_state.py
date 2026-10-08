"""`mace cluster job` never shows a running job as "PENDING (Priority),
Elapsed 0 s" next to its live SCF progress.

scontrol and squeue answers are cached for a minute (15 s refresh floor). A
job that started after the last answer kept the cached PENDING until the
next fetch, while its .out already showed CRYSTAL running. When the output
was written after SLURM was asked, MACE now asks again; when the refresh
floor keeps the old answer, the output is the newer evidence and the job is
shown as running (with a note, and no made-up elapsed time).

The fake SLURM answers; the cluster test clock (MACE_CLUSTER_TEST_CLOCK)
moves time, and the .out's mtime is set explicitly.
"""
import os
import subprocess
import sys
import time

import pytest

from cluster_fixtures import own_cache  # noqa: F401  (autouse)
from cluster_fixtures import MACE_CLI, fake, use_env
from mace.cluster import jobinfo


def _text(d, state, reason, runtime):
    return (f"JobId=4001 JobName=mat1_opt UserId=user1(1001) GroupId=g(1) JobState={state} "
            f"Reason={reason} RunTime={runtime} TimeLimit=01:00:00 StartTime=2030-01-01T00:00:00 "
            f"Partition=general-long NodeList=cpu-001 NumNodes=1 NumCPUs=32 "
            f"WorkDir={d} StdOut={d}/mat1_opt-4001.o StdErr={d}/mat1_opt-4001.o "
            f"Command={d}/mat1_opt.sh Account=acct1 QOS=normal")


@pytest.fixture
def world(tmp_path, monkeypatch):
    d = tmp_path / "proj"
    d.mkdir()
    fs = fake(tmp_path)
    use_env(monkeypatch, fs)
    clock = tmp_path / "clock"
    t0 = time.time() - 600
    clock.write_text(str(t0))
    monkeypatch.setenv("MACE_GUI_TEST_HOOKS", "1")
    monkeypatch.setenv("MACE_CLUSTER_TEST_CLOCK", str(clock))
    fs.add_record(id="4001", state="PENDING", name="mat1_opt", workdir=str(d), time_limit="1:00:00",
                  elapsed="0:00", cpus=32, partition="general-long", reason="Priority")
    fs.update(job_text={"4001": _text(d, "PENDING", "Priority", "00:00:00")})
    return d, fs, clock, t0


def _start(d, fs, written_at):
    """The job starts: SLURM says RUNNING, CRYSTAL writes its output."""
    fs.set_record("4001", state="RUNNING", elapsed="0:40", reason="None")
    fs.update(job_text={"4001": _text(d, "RUNNING", "None", "00:00:40")})
    out = d / "mat1_opt.out"
    out.write_text(" CYC   1 ETOT(AU) -1.0\n")
    os.utime(out, (written_at, written_at))


def test_a_pending_job_still_pending_stays_pending(world):
    d, fs, clock, t0 = world
    r = jobinfo.job("4001")
    assert r["state"] == "PENDING" and r["reason"] == "Priority"
    assert "state_note" not in r


def test_the_output_written_after_the_answer_asks_slurm_again(world):
    d, fs, clock, t0 = world
    assert jobinfo.job("4001")["state"] == "PENDING"            # cached at t0
    _start(d, fs, t0 + 10)
    clock.write_text(str(t0 + 30))                               # past the 15 s floor, within the 60 s TTL
    r = jobinfo.job("4001")
    assert r["state"] == "RUNNING" and r["reason"] == "None"
    assert r["elapsed_s"] == 40
    assert "state_note" not in r


def test_within_the_refresh_floor_the_output_wins(world):
    d, fs, clock, t0 = world
    assert jobinfo.job("4001")["state"] == "PENDING"
    _start(d, fs, t0 + 3)
    clock.write_text(str(t0 + 5))                                # SLURM may not be asked again yet
    r = jobinfo.job("4001")
    assert r["state"] == "RUNNING"
    assert r["reason"] in (None, "None") and r["elapsed_s"] is None
    assert "predates the output CRYSTAL is writing" in r["state_note"]


def test_an_older_output_of_an_earlier_run_does_not_count(world):
    d, fs, clock, t0 = world
    out = d / "mat1_opt.out"
    out.write_text(" CYC   1 ETOT(AU) -1.0\n")
    os.utime(out, (t0 - 300, t0 - 300))
    assert jobinfo.job("4001")["state"] == "PENDING"
    clock.write_text(str(t0 + 30))
    r = jobinfo.job("4001")
    assert r["state"] == "PENDING" and r["reason"] == "Priority"


def test_the_cli_line(world):
    d, fs, clock, t0 = world
    env = dict(os.environ)
    run = lambda: subprocess.run([sys.executable, str(MACE_CLI), "--no-banner", "cluster", "job", "4001"],  # noqa: E731
                                 env=env, capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL)
    first = run()
    assert "PENDING (Priority)" in first.stdout, first.stdout + first.stderr
    _start(d, fs, t0 + 3)
    clock.write_text(str(t0 + 5))
    second = run()
    assert "Job 4001 · mat1_opt · RUNNING" in second.stdout, second.stdout + second.stderr
    assert "PENDING" not in second.stdout.split("Note:")[0] and "Elapsed 0 s" not in second.stdout
    assert "Note: SLURM's last answer (PENDING" in second.stdout
