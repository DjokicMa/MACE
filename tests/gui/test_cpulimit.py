"""Per-process CPU-time limits (plan D18/D19): the limit and per-pid use, the
banner, clirun's classification of a kernel kill, the pool (a worker killed
by its limit fails only its own task; idle recycling), the ML worker restart
rule, the server's own clean stop, and the database build's relaunch rule."""
import http.client
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

import guiharness as H
from mace.gui import buildresume, clirun, cpulimit

linux = pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="needs /proc")


def test_limit_from_getrlimit(monkeypatch):
    import resource
    monkeypatch.delenv("MACE_GUI_TEST_CPU_LIMIT_S", raising=False)
    monkeypatch.setattr(resource, "getrlimit", lambda k: (7200, 7200))
    assert cpulimit.limit() == 7200.0
    monkeypatch.setattr(resource, "getrlimit", lambda k: (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    assert cpulimit.limit() is None
    monkeypatch.setenv("MACE_GUI_TEST_HOOKS", "1")
    monkeypatch.setenv("MACE_GUI_TEST_CPU_LIMIT_S", "120")
    assert cpulimit.limit() == 120.0


@linux
def test_checks_land_between_the_stop_share_and_the_kill():
    """Once a minute under a long limit; under a short one at least three checks fit in the last 3 %."""
    assert cpulimit.check_every(7200) == 60.0
    assert cpulimit.check_every(600) == pytest.approx(6.0)
    assert cpulimit.check_every(60) == 2.0
    for lim in (600, 1800, 7200):
        assert 3 * cpulimit.check_every(lim) <= (1 - cpulimit.STOP_AT) * lim + 1e-9


def test_used_reads_proc():
    before = cpulimit.used(os.getpid())
    end = time.process_time() + 0.3
    while time.process_time() < end:
        pass
    after = cpulimit.used(os.getpid())
    assert after - before >= 0.2
    assert cpulimit.used(999999999) is None


def _banner(tmp_path, wrap=None, **env):
    e = H.env(tmp_path / "xdg", MACE_GUI_NO_WARMUP="1", **env)
    cmd = [sys.executable, str(H.MACE_CLI), "--no-banner", "gui", "--dir", str(tmp_path), "--port", "0"]
    if wrap:
        cmd = ["bash", "-c", wrap + ' "$@"', "x"] + cmd
    p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         env=e, start_new_session=True)
    out = []
    try:
        for _ in range(30):
            line = p.stdout.readline()
            out.append(line)
            if not line or line.startswith("Press Ctrl+C"):
                break
    finally:
        p.terminate()
        p.wait(15)
    return "".join(out)


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_banner_only_with_a_finite_limit_and_not_in_a_job(tmp_path):
    out = _banner(tmp_path, "ulimit -S -t 3600; exec")
    assert "This node limits each process to 60 CPU minutes." in out
    out = _banner(tmp_path, "ulimit -S -t 3600; exec", SLURM_JOB_ID="123")
    assert "CPU minutes" not in out
    out = _banner(tmp_path)
    if cpulimit.limit() is None:
        assert "CPU minutes" not in out


@linux
@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_clirun_classifies_a_kernel_kill(tmp_path):
    burn = [ "bash", "-c", f'ulimit -t 2; exec {sys.executable} -c "while 1: pass"']
    r = clirun.run(burn, tmp_path, timeout=60)
    assert r.returncode in (-9, -24) and r.cpu_limit is True
    # a plain SIGKILL at low CPU is not the limit
    p_argv = [sys.executable, "-c", "import os, signal, time; time.sleep(0.6); os.kill(os.getpid(), signal.SIGKILL)"]
    r = clirun.run(p_argv, tmp_path, timeout=60)
    assert r.returncode == -9 and r.cpu_limit is False


class Srv:
    def __init__(self, tmp, **env):
        root = tmp / "served"
        root.mkdir(exist_ok=True)
        (tmp / "tmp").mkdir(exist_ok=True)
        e = H.env(tmp / "xdg", MACE_GUI_NO_WARMUP="1", MACE_GUI_TEST_HOOKS="1", TMPDIR=str(tmp / "tmp"),
                  XDG_RUNTIME_DIR=str(tmp / "run"), **env)
        (tmp / "run").mkdir(exist_ok=True)
        self.p = H.ServerProcess([sys.executable, str(H.MACE_CLI), "--no-banner", "gui", "--dir", str(root),
                                  "--port", "0", "--token", H.TOKEN], stdin=subprocess.DEVNULL,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=e,
                                 start_new_session=True, preexec_fn=H.die_with_pytest)
        self.lines = []
        for _ in range(40):
            line = self.p.stdout.readline()
            self.lines.append(line)
            m = re.search(r"http://127\.0\.0\.1:(\d+)(/s/[^/]+/)", line)
            if m:
                self.port, self.prefix = int(m.group(1)), m.group(2)
                break
        threading.Thread(target=lambda: [self.lines.append(x) for x in self.p.stdout], daemon=True).start()
        self.tmp = tmp

    def req(self, route, body=None, timeout=120):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        h = {"X-Mace-Token": H.TOKEN, "X-Requested-With": "mace-gui", "Content-Type": "application/json"}
        c.request("POST" if body is not None else "GET", self.prefix + "api/" + route,
                  body=json.dumps(body) if body is not None else None, headers=h)
        r = c.getresponse()
        data = json.loads(r.read() or b"{}")
        c.close()
        return r.status, data

    def wait_task(self, tid, timeout=60):
        end = time.time() + timeout
        while time.time() < end:
            st, d = self.req("tasks")
            t = next((x for x in d["tasks"] if x["id"] == tid), None)
            if t and t["state"] in ("done", "failed", "canceled"):
                return t
            time.sleep(0.3)
        raise AssertionError(f"task {tid} did not end")

    def stop(self):
        if self.p.poll() is None:
            self.p.terminate()
            self.p.wait(15)


@linux
def test_a_worker_killed_by_its_limit_fails_only_its_task(tmp_path):
    s = Srv(tmp_path, MACE_GUI_MAX_WORKERS="2", MACE_GUI_TEST_WORKER_RLIMIT_CPU="3")
    try:
        st, a = s.req("test/task", {"job": "test.burn", "seconds": 10, "wait": 0, "title": "burn"})
        st, b = s.req("test/task", {"job": "sleep", "seconds": 4, "wait": 0, "title": "sleep"})
        ta, tb = s.wait_task(a["task"]["id"]), s.wait_task(b["task"]["id"])
        assert ta["state"] == "failed" and "CPU-time limit" in ta["error"] and "stopped the task" in ta["error"], ta
        assert tb["state"] == "done", tb
        before = s.req("test/pool")[1] if s.req("test/pool")[0] == 200 else None
        st, c = s.req("test/task", {"job": "sleep", "seconds": 0.5, "wait": 0, "title": "after"})
        assert s.wait_task(c["task"]["id"])["state"] == "done"
        rep = s.req("resources")[1]["pool"]
        assert rep["broken"] >= 1 and rep["executors_started"] >= 2
    finally:
        s.stop()


@linux
def test_an_idle_pool_past_60_percent_starts_fresh_workers(tmp_path):
    s = Srv(tmp_path, MACE_GUI_MAX_WORKERS="1", MACE_GUI_TEST_CPU_LIMIT_S="6")
    try:
        st, a = s.req("test/task", {"job": "test.burn", "seconds": 4, "wait": 0})
        t = s.wait_task(a["task"]["id"])
        assert t["state"] == "done"
        pids1 = set(s.req("resources")[1]["pool"]["pids"])
        st, b = s.req("test/task", {"job": "sleep", "seconds": 0.3, "wait": 0})
        assert s.wait_task(b["task"]["id"])["state"] == "done"
        rep = s.req("resources")[1]["pool"]
        assert rep["recycled"] == 1 and not (set(rep["pids"]) & pids1), rep
        assert rep["alive"] <= 1
    finally:
        s.stop()


@linux
def test_the_server_stops_cleanly_near_its_own_limit(tmp_path):
    s = Srv(tmp_path, MACE_GUI_TEST_CPU_LIMIT_S="12")
    try:
        cache = list((tmp_path / "tmp").glob("mace_gui_cache_*"))
        state = list((tmp_path / "run").glob("mace-gui-*.json"))
        assert cache and state
        used = s.req("test/burn-server", {"seconds": 0.1})[1]["cpu"]
        s.req("test/burn-server", {"seconds": max(0.1, 6.5 - used)})                  # past 50 %
        time.sleep(2.5)
        st, info = s.req("info")
        assert info["cpu_limit_s"] == 12 and info["cpu"]["level"] in ("chip", "warn"), info["cpu"]
        s.req("test/burn-server", {"seconds": 12 * 0.82 - info["cpu"]["used_s"] + 0.2})  # past 80 %
        time.sleep(2.5)
        assert s.req("info")[1]["cpu"]["level"] == "warn"
        try:
            s.req("test/burn-server", {"seconds": 12 * 0.98 - s.req("info")[1]["cpu"]["used_s"] + 0.3}, timeout=30)
        except OSError:
            pass
        s.p.wait(15)
        assert s.p.returncode == 0
        time.sleep(0.3)
        assert any("stopped before this node's CPU-time limit (0 min per process)" in ln for ln in s.lines), s.lines[-4:]
        assert not list((tmp_path / "tmp").glob("mace_gui_cache_*")) and not list((tmp_path / "run").glob("*.json"))
    finally:
        s.stop()


class _FakeWorker:
    def __init__(self, pid):
        self.pid, self.socket_path, self.hello, self.killed = pid, "/nonexistent", {"op": "hello"}, False

    def alive(self):
        return not self.killed

    def kill(self):
        self.killed = True

    def close(self):
        pass


@linux
def test_ml_worker_restarts_before_the_limit(monkeypatch):
    """A burning stand-in process plays the ML worker: past 60 % of the
    (fake) limit, or when the next Clean could cross 90 %, acquire() starts a
    new worker first (cpu_restarts)."""
    from mace.gui import mlip_host
    monkeypatch.setenv("MACE_GUI_TEST_HOOKS", "1")
    monkeypatch.setenv("MACE_GUI_TEST_CPU_LIMIT_S", "4")
    burner = subprocess.Popen([sys.executable, "-c", "import time\nend=time.process_time()+2.8\n"
                                                     "while time.process_time()<end: pass\ntime.sleep(60)"])
    try:
        time.sleep(3.5)
        host = mlip_host.MlipHost()
        host.worker, host.model_id = _FakeWorker(burner.pid), "m"
        monkeypatch.setattr(host, "_ping", staticmethod(lambda w, timeout=5.0: True))
        new = []

        class FakeWP:
            def __init__(self, *a, **k):
                self.pid, self.socket_path, self.hello = os.getpid(), "/nonexistent", {"op": "hello"}
                new.append(self)

            def alive(self):
                return True

            def kill(self):
                pass

            def close(self):
                pass
        from mace.preopt import detect, mlip_client, proc, registry
        monkeypatch.setattr(mlip_client, "WorkerProcess", FakeWP)
        monkeypatch.setattr(registry, "get", lambda mid: {"id": mid})
        monkeypatch.setattr(detect, "configured_python", lambda: sys.executable)
        monkeypatch.setattr(detect, "weights_status", lambda m: {"status": "bundled"})
        monkeypatch.setattr(proc, "mem_available_gb", lambda: 16.0)

        class Task:
            id, cancel_requested, message, log = "t1", False, "", ""
            payload = {"time_limit": 0}
        t = Task()
        release = host.acquire(t, "m", 1)
        assert host.cpu_restarts == 1 and len(new) == 1 and "CPU-time limit" in t.message
        assert t.payload["mlip_lease"]["pid"] == os.getpid()
        release()
        # the fresh worker (this test process, a little CPU) is reused, unless a run could cross 90 %
        t2 = Task()
        t2.payload = {"time_limit": 0}
        host.acquire(t2, "m", 1)()
        assert host.cpu_restarts == (2 if cpulimit.used(os.getpid()) >= 2.4 else 1)
        assert host.report()["cpu_restarts"] == host.cpu_restarts
    finally:
        burner.kill()
        burner.wait(5)


def test_build_relaunch_rule():
    argv = ["analyze", "--extract-properties", "--skip-unchanged", "--backup", "--keep-backups", "3", "--register"]
    assert buildresume.resume_argv(argv) == ["analyze", "--extract-properties", "--skip-unchanged", "--register"]
    assert buildresume.resume_argv(["x", "--keep-backups=2", "--backup"]) == ["x"]

    class R:
        def __init__(self, cpu):
            self.cpu_limit = cpu
    seen, notes = [], []
    results = iter([R(True), R(True), R(False)])
    out = buildresume.run_with_resume(lambda a: seen.append(a) or next(results), argv, note=notes.append,
                                      limit_minutes="120")
    assert not out.cpu_limit and len(seen) == 3
    assert seen[0] == argv and seen[1] == seen[2] == buildresume.resume_argv(argv)
    assert "attempt 2 of 4" in notes[0] and "attempt 3 of 4" in notes[1] and "Files already written are kept" in notes[0]
    with pytest.raises(buildresume.BuildStopped, match="Turn on 'Only new and changed files'"):
        buildresume.run_with_resume(lambda a: R(True), ["analyze", "--backup"])
    calls = []
    with pytest.raises(buildresume.BuildStopped, match="4 times"):
        buildresume.run_with_resume(lambda a: calls.append(a) or R(True), argv)
    assert len(calls) == 4

    class Ctx:
        canceled = True
    calls = []
    res = buildresume.run_with_resume(lambda a: calls.append(a) or R(True), argv, ctx=Ctx())
    assert len(calls) == 1 and res.cpu_limit


@linux
@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_build_relaunch_after_a_real_kernel_kill(tmp_path):
    """The first attempt (with --backup) runs under a 1 s RLIMIT_CPU and is
    killed by the kernel; the relaunch (no --backup) finishes."""
    marker = tmp_path / "attempts.txt"
    script = tmp_path / "build.py"
    script.write_text("import sys, time\nopen(sys.argv[1], 'a').write(' '.join(sys.argv[2:]) + '\\n')\n"
                      "if '--backup' in sys.argv:\n    end = time.process_time() + 30\n"
                      "    while time.process_time() < end: pass\n")

    def attempt(argv):
        wrap = "ulimit -t 1; exec \"$@\"" if "--backup" in argv else "exec \"$@\""
        return clirun.run(["bash", "-c", wrap, "x", sys.executable, str(script), str(marker), *argv], tmp_path,
                          timeout=60)
    res = buildresume.run_with_resume(attempt, ["--skip-unchanged", "--backup"])
    assert res.returncode == 0 and not res.cpu_limit
    assert marker.read_text().splitlines() == ["--skip-unchanged --backup", "--skip-unchanged"]
