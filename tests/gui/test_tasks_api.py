"""The process pool and background tasks through the real `mace gui` (plan
B3/B4, spec sections 8.4.1, 8.4.8 and 9.1): created with the first job,
capped, niced, shut down when idle, sharing one CPU budget with the request
threads; progress over /api/events at most 4 times a second; cancel in
under 2.5 s; a quick job answered inline and a slow one detached (202).
The workers start without re-running mace_cli (spike S1)."""
import os
import sys
import threading
import time

import pytest

import guiharness as H
import p4util as U

pytest.importorskip("numpy")


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("tasks")
    root, xdg = H.examples_copy(tmp), tmp / "xdg"
    proc, base, _ = H.start_server(root, xdg, MACE_GUI_NO_WARMUP="1", MACE_GUI_TEST_HOOKS="1",
                                   MACE_GUI_MAX_WORKERS="2")
    yield base, root, proc
    proc.terminate()
    proc.wait(timeout=15)


def task(base, **body):
    return U.post(base, "api/test/task", body)


def stat_nice(pid):
    with open(f"/proc/{pid}/stat") as fh:
        return int(fh.read().rsplit(")", 1)[1].split()[16])       # field 19 of the whole line


def test_pool_lazily_created_and_capped(served):
    """No worker before the first job; eight jobs never run more than 2 at once."""
    base, _, proc = served
    st, _, r = U.req(base, "api/test/pool")
    assert r["pool"] is None and not H.descendants(proc.pid), (r, H.descendants(proc.pid))
    res, threads = [], []

    def one():
        st, _, r = task(base, seconds=0.6, steps=3, wait=0)
        tid = r["task"]["id"]
        end = time.time() + 30
        while time.time() < end:
            st2, _, r2 = U.req(base, f"api/tasks/{tid}/result")
            if st2 == 200:
                res.append(r2)
                return
            time.sleep(0.1)
    for _ in range(8):
        t = threading.Thread(target=one)
        t.start()
        threads.append(t)
    for t in threads:
        t.join(60)
    assert len(res) == 8
    events = sorted([(r["start"], 1) for r in res] + [(r["end"], -1) for r in res])
    run = peak = 0
    for _, d in events:
        run += d
        peak = max(peak, run)
    assert peak <= 2 and len({r["pid"] for r in res}) <= 2, (peak, {r["pid"] for r in res})
    pool = U.req(base, "api/resources")[2]["pool"]
    assert pool["cap"] == 2 and 1 <= pool["workers_used"] <= 2, pool


def test_pool_workers_are_niced(served):
    base = served[0]
    task(base, seconds=0.2, steps=1, wait=5)
    pids = U.req(base, "api/test/pool")[2]["pool"]["pids"]
    assert pids
    for pid in pids:
        assert stat_nice(pid) == 10, (pid, stat_nice(pid))


def test_pool_worker_clean_imports(served):
    """From the real `mace_cli gui` subprocess: the worker did not run mace_cli
    again (no __mp_main__, no ui facade, no pandas/pyarrow/plotly), its
    `queue` is the standard library's, mace/ (or a folder below it) is not on
    its sys.path, it does not run inside the
    checkout, and its environment is the server's (pin_install_home ran once,
    in the server). Only booleans and a one-way fingerprint come back: the
    test hooks never return the environment itself."""
    base = served[0]
    st, _, w = task(base, job="probe", wait=30)
    assert st == 200, w
    par = U.req(base, "api/test/pool")[2]
    assert w["mp_main"] is None and w["main"] is None, w
    assert not w["loaded"], w["loaded"]
    stdlib = os.path.dirname(os.__file__)
    assert os.path.dirname(w["queue_file"]) == stdlib or "/lib/python3" in w["queue_file"], w["queue_file"]
    assert "/mace/queue" not in w["queue_file"]
    assert par["mace_dir_on_sys_path"], "the server has mace/ on its sys.path (mace_cli puts it there)"
    assert not w["mace_dir_on_sys_path"] and not w["mace_subdir_on_sys_path"], w
    assert not w["cwd_inside_repo"], w
    assert w["env_fingerprint"] == par["env_fingerprint"]
    assert "env" not in w and "sys_path" not in w and "sys_path" not in par
    assert set(par["env"]) <= {"OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MACE_GUI_MAX_WORKERS",
                               "SLURM_CPUS_ON_NODE", "SLURM_CPUS_PER_TASK", "CUDA_VISIBLE_DEVICES"}, par["env"]
    assert w["threads"]["OMP_NUM_THREADS"] == "1" and w["nice"] == 10


def test_wait_then_detach(served):
    base = served[0]
    t0 = time.time()
    st, _, r = task(base, seconds=0.05, steps=1)
    assert st == 200 and "pid" in r and r["task"]["state"] == "done", r
    assert time.time() - t0 < 2.5
    st, _, r = task(base, seconds=1.0, steps=4)
    assert st == 202 and r["detached"] and r["task"]["state"] in ("queued", "running"), r
    tid = r["task"]["id"]
    assert U.req(base, f"api/tasks/{tid}/result")[0] == 409            # not done yet
    end = time.time() + 20
    while U.req(base, f"api/tasks/{tid}/result")[0] == 409 and time.time() < end:
        time.sleep(0.1)
    st, _, res = U.req(base, f"api/tasks/{tid}/result")
    assert st == 410 or "pid" in res
    # fetched once: the result is let go of
    assert U.req(base, f"api/tasks/{tid}/result")[0] == 410


def test_task_progress_over_sse(served):
    base = served[0]
    sse = H.sse_events(base)
    try:
        sse.wait(lambda e: e[0] == "hello", 10)
        st, _, r = task(base, seconds=2.0, steps=20, wait=0)
        tid = r["task"]["id"]
        done = sse.wait(lambda e: e[0] == "task" and e[1]["id"] == tid and e[1]["state"] == "done", 30)
        evs = sse.of("task", lambda d: d["id"] == tid)
        prog = [e for e in evs if e[1]["state"] == "running" and 0 < e[1]["progress"] < 1]
        assert len(prog) >= 2, [(e[1]["state"], e[1]["progress"]) for e in evs]
        running = [e for e in evs if e[1]["state"] == "running"]
        span = max(0.5, done[2] - running[0][2])
        assert len(prog) / span <= 4.5, (len(prog), span)
        assert [e[1]["state"] for e in evs].count("done") == 1
    finally:
        sse.close()


def test_task_cancel_latency(served):
    base = served[0]
    sse = H.sse_events(base)
    try:
        sse.wait(lambda e: e[0] == "hello", 10)
        st, _, r = task(base, seconds=10, steps=100, wait=0)
        tid = r["task"]["id"]
        sse.wait(lambda e: e[0] == "task" and e[1]["id"] == tid and e[1]["state"] == "running", 20)
        time.sleep(0.5)
        t0 = time.time()
        assert U.post(base, f"api/tasks/{tid}/cancel", {})[0] == 200
        ev = sse.wait(lambda e: e[0] == "task" and e[1]["id"] == tid and e[1]["state"] == "canceled", 10)
        assert ev[2] - t0 <= 2.5, ev[2] - t0
        assert ev[1]["log"].startswith("Canceled after")
    finally:
        sse.close()


def test_task_failed_then_retry(served):
    base = served[0]
    st, _, r = task(base, seconds=0.4, steps=4, fail=True, wait=10)
    assert st == 500 and "on purpose" in r["error"] and "RuntimeError" in (r["traceback"] or ""), r
    tid = r["task"]["id"]
    rec = [t for t in U.req(base, "api/tasks")[2]["tasks"] if t["id"] == tid][0]
    assert rec["state"] == "failed" and rec["error"] == "the test job failed on purpose"
    st, _, r2 = U.post(base, f"api/tasks/{tid}/retry", {})
    assert r2["task"]["retry_of"] == tid
    assert U.post(base, f"api/tasks/{tid}/remove", {})[2] == {"removed": True}


def test_cpu_budget_shared_by_gate_and_pool(served):
    """Two slow pool jobs and a burst of held inline symmetry calls: never more
    than the cap (2) running at once, counted together."""
    pytest.importorskip("ase")                    # reading the structure and spglib's symmetry
    pytest.importorskip("spglib")
    base = served[0]
    mgo = U.req(base, "api/structure?path=structures/MgO.cif")[2]
    body = {"cell": mgo["cell"], "numbers": mgo["numbers"], "frac": mgo["frac"]}
    U.post(base, "api/test/gate/reset", {})
    samples, stop = [], threading.Event()

    def sample():
        while not stop.is_set():
            b = U.req(base, "api/resources")[2]["budget"]
            samples.append(b["inline_running"] + b["pool_running"])
            time.sleep(0.02)
    s = threading.Thread(target=sample)
    s.start()
    jobs = [task(base, seconds=2.0, steps=4, wait=0)[2]["task"]["id"] for _ in range(2)]
    time.sleep(0.3)
    calls = [threading.Thread(target=lambda k=k: U.post(base, "api/symmetry", {**body, "key": f"k{k}"},
                                                        headers={"X-Mace-Test-Hold-Ms": "300"})) for k in range(8)]
    for c in calls:
        c.start()
    for c in calls:
        c.join(60)
    for tid in jobs:
        end = time.time() + 30
        while [t for t in U.req(base, "api/tasks")[2]["tasks"] if t["id"] == tid][0]["state"] in ("queued", "running"):
            assert time.time() < end
            time.sleep(0.1)
    stop.set()
    s.join(10)
    gate = U.req(base, "api/test/gate")[2]
    assert max(samples) <= 2 and gate["max_running"] <= 2, (max(samples), gate)
    assert max(samples) >= 1


def test_pool_idle_shutdown(served):
    base = served[0]
    U.post(base, "api/test/pool/config", {"idle_s": 2})
    task(base, seconds=0.1, steps=1, wait=5)
    pids = U.req(base, "api/test/pool")[2]["pool"]["pids"]
    assert pids
    end = time.time() + 15
    while time.time() < end and any(os.path.exists(f"/proc/{p}") for p in pids):
        time.sleep(0.2)
    assert not any(os.path.exists(f"/proc/{p}") for p in pids), pids
    pool = U.req(base, "api/resources")[2]["pool"]
    assert pool["alive"] == 0 and not pool["started"], pool
    # the next job starts it again
    st, _, r = task(base, seconds=0.05, steps=1, wait=10)
    assert st == 200 and r["pid"] not in pids
    U.post(base, "api/test/pool/config", {"idle_s": 60})


@pytest.mark.parametrize("path", ["api/tasks/x/cancel", "api/tasks/x/retry", "api/tasks/x/remove"])
def test_task_routes_require_token_host_and_xrw(served, path):
    base = served[0]
    assert U.post(base, path, {}, token=False)[0] == 403
    port = base.split("/")[2].rsplit(":", 1)[1]
    assert U.post(base, path, {}, headers={"Host": f"evil.example:{port}"})[0] == 421
    assert U.post(base, path, {}, headers={"X-Requested-With": "x"})[0] == 403
    assert U.req(base, "api/tasks", token=False)[0] == 403
    assert U.req(base, "api/tasks/x/result", token=False)[0] == 403
