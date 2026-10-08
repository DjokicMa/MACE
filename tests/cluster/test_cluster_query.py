"""F1: the one cached, rate-limited runner (query.run), the shared reader and
the one poller, against FakeSlurm."""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from cluster_fixtures import own_cache  # noqa: F401  (autouse)
from cluster_fixtures import REPO_ROOT, fake, use_env

from mace.cluster import poller, query, reader


@pytest.fixture
def fs(tmp_path, monkeypatch):
    f = fake(tmp_path)
    use_env(monkeypatch, f)
    return f


@pytest.fixture
def clock(tmp_path, monkeypatch):
    path = tmp_path / "clock"
    path.write_text("1000000")
    monkeypatch.setenv("MACE_GUI_TEST_HOOKS", "1")
    monkeypatch.setenv("MACE_CLUSTER_TEST_CLOCK", str(path))

    class C:
        def set(self, t):
            path.write_text(str(t))

        def add(self, dt):
            path.write_text(str(float(path.read_text()) + dt))
    return C()


def _calls(fs, tool, *words):
    return [c for c in fs.calls(tool) if all(w in c["argv"] for w in words)]


def test_single_flight(fs, tmp_path):
    """F1-a: 50 threads and 3 processes asking for nodes at once: one call."""
    fs.update(delay={"scontrol": 1.0})
    child = ("import sys; sys.path.insert(0, %r); from mace.cluster import reader; "
             "r = reader.nodes(); print(len(r.value or []))" % str(REPO_ROOT))
    procs = [subprocess.Popen([sys.executable, "-c", child], env=fs.env, stdout=subprocess.PIPE, text=True)
             for _ in range(3)]
    results = []
    threads = [threading.Thread(target=lambda: results.append(reader.nodes())) for _ in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    outs = [p.communicate(timeout=60)[0].strip() for p in procs]
    assert len(_calls(fs, "scontrol", "nodes")) == 1, fs.calls("scontrol")
    assert {len(r.value) for r in results} == {11}
    assert outs == ["11"] * 3


def test_ttl_and_refresh_floor(fs, clock):
    """F1-b: no call within the TTL; one after; a refresh inside the floor
    makes no call and says when the next is allowed."""
    reader.sinfo()
    reader.sinfo()
    assert len(fs.calls("sinfo")) == 1
    clock.add(59)
    reader.sinfo()
    assert len(fs.calls("sinfo")) == 1
    clock.add(2)
    reader.sinfo()
    assert len(fs.calls("sinfo")) == 2
    clock.add(5)
    r = reader.sinfo(refresh=True)
    assert len(fs.calls("sinfo")) == 2
    assert 9.9 <= r.next_allowed_in <= 10.1
    clock.add(11)
    reader.sinfo(refresh=True)
    assert len(fs.calls("sinfo")) == 3


def test_backoff_and_stale(fs, clock):
    """F1-c: a failing tool backs off 30, 60, 120 ... 300 s; the last good
    value comes back marked stale with the error."""
    good = reader.nodes()
    assert good.value and not good.stale
    fs.update(fail={"scontrol": "slurm_load_node error: Socket timed out"})
    waits = []
    for _ in range(6):
        clock.add(61)
        before = len(fs.calls("scontrol"))
        r = reader.nodes()
        assert len(fs.calls("scontrol")) == before + 1
        assert r.stale and r.value == good.value
        assert "Socket timed out" in r.error
        rec = query.read_cache("nodes")
        waits.append(round(rec["next_try"] - query.now()))
        # inside the backoff: no call
        r2 = reader.nodes(refresh=True)
        assert len(fs.calls("scontrol")) == before + 1 and r2.stale
        clock.add(rec["next_try"] - query.now() - 61)
    assert waits == [30, 60, 120, 240, 300, 300]


def test_bounded_cache_files(fs, clock):
    """F1-d: one .json and one .lock per key after 1000 calls."""
    for i in range(1000):
        if i % 100 == 0:
            clock.add(61)
        reader.sinfo()
    names = sorted(p.name for p in query.cache_dir().iterdir())
    assert names == ["sinfo.json", "sinfo.lock"], names
    assert len(fs.calls("sinfo")) == 10
    assert oct(query.cache_dir().stat().st_mode & 0o777) == "0o700"
    assert oct((query.cache_dir() / "sinfo.json").stat().st_mode & 0o777) == "0o600"


def test_x7_ttls_fresh_and_fetch_log(fs, clock):
    """F1-e: squeue_own refreshes once past 60 s; two refreshes within 15 s
    give one call; fresh always calls and is tagged; the log keeps 50."""
    reader.squeue_own()
    clock.add(61)
    reader.squeue_own()
    reader.squeue_own()
    assert len(fs.calls("squeue")) == 2
    clock.add(16)
    reader.squeue_own(refresh=True)
    clock.add(5)
    reader.squeue_own(refresh=True)
    assert len(fs.calls("squeue")) == 3
    for _ in range(3):
        reader.squeue_own(fresh=True)
    assert len(fs.calls("squeue")) == 6
    tags = [t for _, t in query.fetch_log("ownjobs")]
    assert tags == ["cli", "cli", "refresh", "fresh", "fresh", "fresh"]
    for _ in range(60):
        reader.squeue_own(fresh=True)
    assert len(query.fetch_log("ownjobs")) == 50
    # the TTLs of the table
    for fn, key, ttl in ((reader.nodes, "nodes", 60), (reader.partitions, "partitions", 3600),
                         (reader.reservations, "reservations", 300), (reader.fairshare, "sshare", 600)):
        fn()
        n = len(query.fetch_log(key))
        clock.add(ttl - 1)
        fn()
        assert len(query.fetch_log(key)) == n, key
        clock.add(2)
        fn()
        assert len(query.fetch_log(key)) == n + 1, key


def test_lock_wait_is_bounded(fs, monkeypatch):
    """F1-f: a lock another process holds: stale value after the bounded
    wait with the note, never a hang."""
    reader.sinfo()
    lock = query.cache_dir() / "sinfo.lock"
    holder = subprocess.Popen([sys.executable, "-c",
                               "import fcntl, sys, time; f = open(sys.argv[1], 'a'); "
                               "fcntl.flock(f, fcntl.LOCK_EX); print('held', flush=True); time.sleep(30)",
                               str(lock)], stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        monkeypatch.setattr(query, "LOCK_EXTRA", 0.5)
        t0 = time.monotonic()
        r = query.run("sinfo", ["sinfo", "-h"], ttl=0, refresh_floor=0, timeout=1.0)
        took = time.monotonic() - t0
        assert took < 5, took
        assert r.stale and "Another MACE process is reading sinfo" in r.note
        assert r.value is not None
        assert len(fs.calls("sinfo")) == 1
    finally:
        holder.kill()
        holder.wait()


def test_allocation_env_is_stripped(fs, monkeypatch, clock):
    """F1-g: inside an allocation only SLURM_CONF reaches squeue and
    sbatch --test-only (plus MACE's own SLURM_BITSTR_LEN); argv and env equal
    the run outside."""
    from mace.cluster import fit
    req = fit.request_from(cpus=4, time_text="1:00:00")
    fit.predict(req)
    reader.squeue_own(fresh=True)
    outside = [(c["argv"], c["env"]) for c in fs.calls() if c["tool"] in ("sbatch", "squeue")]
    for k, v in {"SLURM_JOB_ID": "77", "SLURM_MEM_PER_NODE": "4000", "SLURM_CPUS_ON_NODE": "8",
                 "SLURM_CONF": "/etc/slurm/slurm.conf"}.items():
        monkeypatch.setenv(k, v)
    clock.add(200)
    fit.predict(req)
    reader.squeue_own(fresh=True)
    inside = [(c["argv"], c["env"]) for c in fs.calls() if c["tool"] in ("sbatch", "squeue")][len(outside):]
    assert [a for a, _ in inside] == [a for a, _ in outside]
    for _, e in inside:
        assert set(e) == {"SLURM_CONF", "SLURM_BITSTR_LEN"}, e
    for (_, eo), (_, ei) in zip(outside, inside):
        assert set(ei) - {"SLURM_CONF"} == set(eo)


def test_poller_finished_events(fs):
    """F1-h: RUNNING then gone: one batched sacct for all gone ids and one
    jobs.finished per id; nothing ticks when nothing is active; two sinks
    each receive it once."""
    fs.add_record(id="4001", state="RUNNING", name="mat1_sp", workdir="/scratch/user1/a")
    fs.add_record(id="4002", state="PENDING", name="mat2_sp", workdir="/scratch/user1/b")
    fs.update(sacct_rows={"4001": ["4001|mat1_sp|COMPLETED|00:10:00|01:00:00|4|4G||00:00:00|cpu=4|cpu-001|0:0|"
                                   "2030-01-01T00:00:00|2030-01-01T00:10:00|/scratch/user1/a"],
                          "4002": ["4002|mat2_sp|TIMEOUT|01:00:02|01:00:00|4|4G||00:00:00|cpu=4|cpu-002|0:0|"
                                   "2030-01-01T00:00:00|2030-01-01T01:00:02|/scratch/user1/b"]})
    sink_a, sink_b = [], []

    def emit(name, data):
        sink_a.append((name, data))
        sink_b.append((name, data))
    active = [True]
    p = poller.Poller(emit, interval=lambda: 0.0, active=lambda: active[0])
    p.tick()
    assert sink_a == []
    fs.remove_record("4001")
    fs.remove_record("4002")
    query._which_cache.clear()
    p.tick()
    assert [e[1]["id"] for e in sink_a] == ["4001", "4002"]
    assert [e[1]["state"] for e in sink_a] == ["COMPLETED", "TIMEOUT"]
    sacct = fs.calls("sacct")
    assert len(sacct) == 1 and "4001,4002" in sacct[0]["argv"]
    p.tick()
    assert len(sink_a) == 2 and len(sink_b) == 2
    # inactive: the loop does not call squeue
    active[0] = False
    n = len(fs.calls("squeue"))
    p.interval = lambda: 0.05
    p.start()
    time.sleep(0.4)
    p.stop()
    assert len(fs.calls("squeue")) == n


def test_poll_seconds_range(monkeypatch):
    monkeypatch.delenv("MACE_GUI_TEST_HOOKS", raising=False)
    assert poller.poll_seconds(None) == 60
    assert poller.poll_seconds(5) == 30
    assert poller.poll_seconds(1000) == 300
    monkeypatch.setenv("MACE_CLUSTER_TEST_POLL_SECONDS", "2")
    assert poller.poll_seconds(60) == 60          # the hook needs MACE_GUI_TEST_HOOKS=1
    monkeypatch.setenv("MACE_GUI_TEST_HOOKS", "1")
    assert poller.poll_seconds(60) == 2


def test_missing_tool_is_an_error_not_a_crash(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "c"))
    query._which_cache.clear()
    query._name_cache.clear()
    r = query.run("x", ["no-such-tool-here"], ttl=60)
    assert r.value is None and "not found" in r.error


def _sacct_row(jid, state="COMPLETED"):
    return [f"{jid}|mat_sp|{state}|00:10:00|01:00:00|4|4G||00:00:00|cpu=4|cpu-001|0:0|"
            "2030-01-01T00:00:00|2030-01-01T00:10:00|/scratch/user1/a"]


def test_sacct_new_id_within_refresh_floor(fs):
    """A second finished job opened within 15 s of the first still gets its
    own accounting rows (the cached batch covers another id set), so it is
    never refused as "Not one of your jobs"."""
    fs.update(sacct_rows={"5001": _sacct_row("5001"), "5002": _sacct_row("5002", "TIMEOUT")})
    assert "5001" in reader.sacct_for(["5001"])
    rows = reader.sacct_for(["5002"])
    assert "5002" in rows and rows["5002"]["State"] == "TIMEOUT"
    # the covered set is answered from the cache: no third call
    n = len(fs.calls("sacct"))
    assert "5001" in reader.sacct_for(["5001"])
    assert len(fs.calls("sacct")) == n


def test_poller_gone_job_after_recent_batch(fs):
    """The poller's batch for a just-finished job is not swallowed by a Job
    document's sacct of another id a few seconds earlier."""
    fs.add_record(id="6001", state="RUNNING", name="mat1_sp", workdir="/scratch/user1/a")
    fs.update(sacct_rows={"6001": _sacct_row("6001"), "6009": _sacct_row("6009")})
    sink = []
    p = poller.Poller(lambda n, d: sink.append(d), interval=lambda: 0.0, active=lambda: True)
    p.tick()
    reader.sacct_for(["6009"])                      # a Job document of an old job
    fs.remove_record("6001")
    query._which_cache.clear()
    p.tick()
    assert [e["state"] for e in sink] == ["COMPLETED"], sink
