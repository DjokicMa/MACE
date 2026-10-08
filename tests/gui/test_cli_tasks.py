"""CLI children as cancelable tasks (plan T1, spec sections 8.4.8 and 9):
deck previews, plots and preflight run the real mace_cli at nice 10 in its
own process group, holding one slot of the CPU budget; a cancel stops the
whole group (SIGTERM, then SIGKILL after 2 s) in under 2.5 s; a newer run
for the same document supersedes the older one; what they write in the
server's cache folder is bounded by the one memory budget."""
import os
import time

import pytest

import guiharness as H
import p4util as U


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("clitasks")
    root, xdg = H.examples_copy(tmp), tmp / "xdg"
    proc, base, _ = H.start_server(root, xdg, MACE_GUI_NO_WARMUP="1", MACE_GUI_TEST_HOOKS="1",
                                   MACE_GUI_MAX_WORKERS="2")
    yield base, root, proc
    proc.terminate()
    proc.wait(timeout=15)


def _stat(pid):
    try:
        with open(f"/proc/{pid}/stat") as fh:
            rest = fh.read().rsplit(")", 1)[1].split()
    except OSError:
        return None
    return {"state": rest[0], "pgrp": int(rest[2]), "nice": int(rest[16])}


def _group(pgid):
    out = []
    for d in os.listdir("/proc"):
        if d.isdigit():
            s = _stat(int(d))
            if s and s["pgrp"] == pgid and s["state"] != "Z":
                out.append(int(d))
    return out


def _wait_pid(base, tid, timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        t = next((x for x in U.req(base, "api/tasks")[2]["tasks"] if x["id"] == tid), None)
        if t and t.get("pid"):
            return t
        time.sleep(0.05)
    raise AssertionError("the CLI child never started")


def _wait_state(base, tid, states, timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        t = next((x for x in U.req(base, "api/tasks")[2]["tasks"] if x["id"] == tid), None)
        if t and t["state"] in states:
            return t
        time.sleep(0.05)
    raise AssertionError(f"task {tid} never reached {states}")


def test_cli_child_runs_niced_in_its_own_group_and_cancels_fast(served):
    base = served[0]
    st, _, r = U.post(base, "api/test/cli", {"seconds": 30, "doc": "cancel-me"})
    assert st == 202, r
    t = _wait_pid(base, r["task"]["id"])
    pid = t["pid"]
    s = _stat(pid)
    # nice(1) execs the CLI in place: the child itself is the group leader, at nice 10
    assert s and s["nice"] == 10 and s["pgrp"] == pid, s
    time.sleep(0.4)                                  # its own child (the grandchild) is up too
    assert len(_group(pid)) >= 2, _group(pid)
    t0 = time.time()
    assert U.post(base, f"api/tasks/{t['id']}/cancel", {})[0] == 200
    done = _wait_state(base, t["id"], ("canceled",))
    took = time.time() - t0
    assert took <= 2.5, took
    assert not _group(pid), _group(pid)              # no orphan left running, no zombie kept
    assert done["cancel_s"] is not None and done["cancel_s"] <= 2.5


def test_newer_run_for_the_same_document_supersedes(served):
    base = served[0]
    a = U.post(base, "api/test/cli", {"seconds": 30, "doc": "doc-A"})[2]["task"]
    _wait_pid(base, a["id"])
    b = U.post(base, "api/test/cli", {"seconds": 0.2, "doc": "doc-A"})[2]["task"]
    old = _wait_state(base, a["id"], ("canceled",))
    assert old["message"] == "superseded" and old["superseded"], old
    new = _wait_state(base, b["id"], ("done", "failed"))
    assert new["state"] == "done", new
    # a different document is not touched
    c = U.post(base, "api/test/cli", {"seconds": 0.3, "doc": "doc-B"})[2]["task"]
    d = U.post(base, "api/test/cli", {"seconds": 0.3, "doc": "doc-C"})[2]["task"]
    assert _wait_state(base, c["id"], ("done",))["state"] == "done"
    assert _wait_state(base, d["id"], ("done",))["state"] == "done"


def test_cli_children_share_the_one_cpu_budget(served):
    base = served[0]
    U.post(base, "api/test/gate/reset", {})
    ids = [U.post(base, "api/test/cli", {"seconds": 0.6, "doc": f"burst-{k}"})[2]["task"]["id"] for k in range(6)]
    for tid in ids:
        _wait_state(base, tid, ("done",), 60)
    gate = U.req(base, "api/test/gate")[2]
    assert gate["max_running"] <= 2, gate          # MACE_GUI_MAX_WORKERS=2


def test_disk_cache_stays_under_its_share_of_the_budget(tmp_path):
    """The plot and Setup folders (on /tmp) go oldest first when over a quarter of cache_mb; one in use stays."""
    import sys
    sys.path.insert(0, str(H.REPO_ROOT))
    from mace.gui.cache import DiskCache
    from mace.gui.memory import MemoryLedger
    led = MemoryLedger(4 * 1024 * 1024)               # 1 MB for the folders
    disk = DiskCache(tmp_path / "runs")
    led.attach_disk(disk)
    held = disk.acquire("busy")
    (held / "x.bin").write_bytes(os.urandom(300_000))
    for k in range(20):
        d = disk.acquire(f"r{k}")
        (d / "plot.svg").write_bytes(os.urandom(100_000))
        disk.release(f"r{k}")
        led.enforce()
        assert disk.used() <= led.disk_cap(), (k, disk.used())
    assert held.is_dir() and (held / "x.bin").exists()   # in use: never dropped
    assert not (tmp_path / "runs" / "r0").exists()        # the oldest went
    assert disk.report()["dropped"] >= 10
    disk.release("busy", remove=True)
    assert not held.exists()
