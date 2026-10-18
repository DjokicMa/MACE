"""MACE GUI's update notice (update_gui.py) through the real `mace_cli gui`
subprocess and HTTP, on the bare-origin fixture of tests/test_update.py: the
check (on demand and in the background), the dialog's data, Not Now and Skip
This Version, the refusals (local changes, network mode, running tasks), Update
Now and Restart (the same process, port, prefix and signed-in session; the
connect clients see no second READY line and no STOPPED line), and Update When
I Sign Out (the helper runs only after the server process is gone; the result
shows at the next start)."""
import http.client
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

import guiharness as H
from test_shutdown_api import Srv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_update import clone, git, origin  # noqa: E402,F401 - the fixture

pytestmark = pytest.mark.skipif(not __import__("shutil").which("git"), reason="needs git")


def _env(home, **extra):
    e = {"MACE_GUI_TEST_HOOKS": "1", "MACE_GUI_TEST_UPDATE_HOME": str(home), "MACE_GUI_UPDATE_CHECK": "0",
         "MACE_GUI_UPDATE_DELAY": "0"}
    e.update(extra)
    return e


def _wait(fn, timeout=60, step=0.2):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        v = fn()
        if v:
            return v
        time.sleep(step)
    raise AssertionError("timed out")


def _status(s):
    st, d, _ = s.req("api/update")
    assert st == 200, d
    return d


def _check(s):
    st, d, _ = s.req("api/update/check", {})
    assert st == 200, d
    return d


def test_check_on_demand_gives_the_dialog_data(origin, tmp_path):
    home = clone(origin, tmp_path / "me")
    s = Srv(tmp_path, **_env(home))
    try:
        d = _status(s)
        assert d["report"] is None and d["checked_at"] is None and d["channel"] == "release"
        d = _check(s)
        r = d["report"]
        assert d["notice"] is True and r["available"] is True and r["refusal"] is None
        assert r["current"]["version"] == "1.1.4" and r["target"]["version"] == "1.1.5"
        assert any(ln.startswith("## [1.1.5]") for ln in r["changelog"])
        assert r["requirements"]["files"] == ["requirements.txt"]
        assert d["tasks"] == [] and d["blocked"] is None
        assert d["restart"]["mode"] == "local" and d["restart"]["in_place"] is True
        assert git(home, "rev-parse", "HEAD") == git(home, "rev-parse", "v1.1.4")
        # another channel, asked for once (Help > Check for Updates in the dialog's channel menu)
        st, d, _ = s.req("api/update/check", {"channel": "preview"})
        assert st == 200 and d["report"]["target"]["ref"] == "v1.1.6-preview"
    finally:
        s.kill()


def test_background_check_at_start_and_off_switch(origin, tmp_path):
    home = clone(origin, tmp_path / "me")
    (tmp_path / "a").mkdir()
    s = Srv(tmp_path / "a", **_env(home, MACE_GUI_UPDATE_CHECK="1"))
    try:
        d = _wait(lambda: (lambda x: x if x["checked_at"] else None)(_status(s)))
        assert d["notice"] is True and d["enabled"] is True and d["report"]["target"]["ref"] == "v1.1.5"
    finally:
        s.kill()
    # Settings > Server: update.check false (the env switch is the tests' and CI's)
    xdg = tmp_path / "b" / "xdg"
    (xdg / "mace").mkdir(parents=True)
    (xdg / "mace" / "gui.json").write_text(json.dumps({"version": 1, "settings": {"update.check": False}}))
    s = Srv(tmp_path / "b", **_env(home, MACE_GUI_UPDATE_CHECK="1"))
    try:
        time.sleep(3)
        d = _status(s)
        assert d["enabled"] is False and d["checked_at"] is None
    finally:
        s.kill()


def test_check_offline_fails_quietly(origin, tmp_path):
    home = clone(origin, tmp_path / "me")
    git(home, "remote", "set-url", "origin", str(tmp_path / "gone.git"))
    s = Srv(tmp_path, **_env(home))
    try:
        d = _check(s)
        assert d["notice"] is False and d["error"] and "origin" in d["error"]
    finally:
        s.kill()


def test_not_now_and_skip_this_version(origin, tmp_path):
    home = clone(origin, tmp_path / "me")
    s = Srv(tmp_path, **_env(home))
    try:
        _check(s)
        st, d, _ = s.req("api/update/choose", {"choice": "later"})
        assert st == 200 and d["notice"] is False
        st, d, _ = s.req("api/update/choose", {"choice": "skip"})
        assert st == 200 and d["skipped"]["ref"] == "v1.1.5"
    finally:
        s.kill()
    s = Srv(tmp_path, **_env(home))          # the next session: Not Now reminds again, Skip does not
    try:
        d = _check(s)
        assert d["notice"] is False and d["skipped"]["ref"] == "v1.1.5" and d["report"]["available"] is True
        st, d, _ = s.req("api/update/check", {"channel": "preview"})
        assert d["notice"] is True                       # a newer target than the skipped one
    finally:
        s.kill()


def test_local_changes_are_refused_with_the_exact_text(origin, tmp_path):
    home = clone(origin, tmp_path / "me")
    (home / "mace_cli").write_text("edited\n")
    s = Srv(tmp_path, **_env(home))
    try:
        d = _check(s)
        r = d["report"]
        assert r["refusal_kind"] == "local_changes" and r["stash"] == f"git -C {home} stash"
        for choice in ("now", "signout"):
            st, e, _ = s.req("api/update/choose", {"choice": choice})
            assert st == 409 and e["error"] == r["refusal"]
        assert (home / "mace_cli").read_text() == "edited\n"
    finally:
        s.kill()


def test_running_tasks_block_update_now(origin, tmp_path):
    home = clone(origin, tmp_path / "me")
    s = Srv(tmp_path, **_env(home))
    try:
        _check(s)
        st, t, _ = s.req("api/test/task", {"seconds": 30, "job": "sleep", "wait": 0})
        assert st in (200, 202), t
        _wait(lambda: _status(s)["tasks"])
        st, e, _ = s.req("api/update/choose", {"choice": "now"})
        assert st == 409 and e["tasks"] and "running" in e["error"]
        assert git(home, "rev-parse", "HEAD") == git(home, "rev-parse", "v1.1.4")
    finally:
        s.kill()


def test_network_mode_refuses(tmp_path):
    """A server on another address (--insecure-network): anyone with the
    token could replace the code, so the update is refused (127.0.0.2 is
    "network" to MACE GUI and still stays on this computer)."""
    root = tmp_path / "served"
    root.mkdir()
    e = H.env(tmp_path / "xdg", MACE_GUI_NO_WARMUP="1", MACE_GUI_UPDATE_CHECK="0")
    p = H.ServerProcess([sys.executable, str(H.MACE_CLI), "--no-banner", "gui", "--dir", str(root), "--port", "0",
                         "--host", "127.0.0.2", "--insecure-network", "--token", H.TOKEN],
                        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=e,
                        start_new_session=True, preexec_fn=H.die_with_pytest)
    try:
        for _ in range(40):
            m = re.search(r"http://127\.0\.0\.2:(\d+)(/s/[^/]+/)", p.stdout.readline())
            if m:
                break
        port, prefix = int(m.group(1)), m.group(2)

        def req(route, body=None):
            c = http.client.HTTPConnection("127.0.0.2", port, timeout=30)
            c.request("POST" if body is not None else "GET", prefix + route,
                      body=json.dumps(body) if body is not None else None,
                      headers={"X-Mace-Token": H.TOKEN, "X-Requested-With": "mace-gui",
                               "Content-Type": "application/json"})
            r = c.getresponse()
            return r.status, json.loads(r.read())
        st, d = req("api/update")
        assert st == 200 and d["blocked"] and "network" in d["blocked"]
        for choice in ("now", "signout"):
            st, d = req("api/update/choose", {"choice": choice})
            assert st == 403 and "mace update" in d["error"]
    finally:
        p.kill()
        p.wait(5)


def _sign_in(s):
    st, d, _ = s.req("api/login-code", {})
    c = http.client.HTTPConnection("127.0.0.1", s.port, timeout=30)
    c.request("GET", f"{s.prefix}?login={d['code']}")
    r = c.getresponse()
    r.read()
    cookie = r.getheader("Set-Cookie").split(";")[0]
    return cookie


def _get_with_cookie(s, route, cookie):
    c = http.client.HTTPConnection("127.0.0.1", s.port, timeout=10)
    c.request("GET", s.prefix + route, headers={"Cookie": cookie, "X-Requested-With": "mace-gui"})
    r = c.getresponse()
    return r.status, r.read()


def _restarted(s, old_started):
    try:
        st, d, _ = s.req("api/update")
    except OSError:
        return None
    return d if st == 200 and d.get("started") != old_started else None


def test_update_now_restarts_in_place_and_keeps_the_session(origin, tmp_path):
    home = clone(origin, tmp_path / "me")
    s = Srv(tmp_path, **_env(home))
    try:
        cookie = _sign_in(s)
        assert _get_with_cookie(s, "api/info", cookie)[0] == 200
        before = _check(s)
        pid = s.p.pid
        st, d, _ = s.req("api/update/choose", {"choice": "now"})
        assert st == 202, d
        d = _wait(lambda: _restarted(s, before["started"]), timeout=120, step=0.5)
        assert s.p.poll() is None and s.p.pid == pid                         # the same process (exec)
        assert d["result"]["ok"] is True and d["result"]["to"] == "1.1.5" and d["result"]["restart"] == "in-place"
        assert "requirements.txt changed" in d["result"]["output"]                  # --no-pip: the line only
        assert git(home, "describe", "--tags") == "v1.1.5"
        assert _get_with_cookie(s, "api/info", cookie)[0] == 200                  # still signed in
        assert d["notice"] is False
        out = "".join(s.lines)
        assert out.count("MACE GUI serving") == 2 and "MACE was updated; restarting" in out
    finally:
        s.kill()


def test_update_now_in_managed_mode_sends_no_second_ready_or_stopped(origin, tmp_path):
    home = clone(origin, tmp_path / "me")
    s = Srv(tmp_path, managed=True, **_env(home))
    try:
        before = _check(s)
        st, d, _ = s.req("api/update/choose", {"choice": "now"})
        assert st == 202, d
        d = _wait(lambda: _restarted(s, before["started"]), timeout=120, step=0.5)
        assert d["result"]["ok"] is True and d["restart"]["mode"] == "managed"
        out = "".join(s.lines)
        assert out.count("MACE_GUI_READY ") == 1 and "MACE_GUI_STOPPED" not in out
        s.p.stdin.write("ping\n")             # the controller of the new process reads the same stdin
        s.p.stdin.flush()
        assert s.p.poll() is None
    finally:
        s.kill()


FAKE_UPDATE = r"""
import json, os, sys, time
pid = int(os.environ["FAKE_SERVER_PID_FILE"] and open(os.environ["FAKE_SERVER_PID_FILE"]).read())
status = "/proc/%d/status" % pid
alive = os.path.exists(status) and "State:\tZ" not in open(status).read()
json.dump({"argv": sys.argv[1:], "server_alive": alive, "t": time.time()}, open(os.environ["FAKE_UPDATE_LOG"], "w"))
print("Updated (fake).")
"""


def test_update_when_i_sign_out_runs_after_the_server_stopped(origin, tmp_path):
    home = clone(origin, tmp_path / "me")
    fake = tmp_path / "fake_update.py"
    fake.write_text(FAKE_UPDATE)
    log = tmp_path / "fake.json"
    pidf = tmp_path / "pid"
    env = _env(home, MACE_GUI_TEST_UPDATE_ARGV=json.dumps([sys.executable, str(fake)]),
               FAKE_UPDATE_LOG=str(log), FAKE_SERVER_PID_FILE=str(pidf))
    s = Srv(tmp_path, **env)
    try:
        pidf.write_text(str(s.p.pid))
        _check(s)
        st, d, _ = s.req("api/update/choose", {"choice": "signout"})
        assert st == 200 and d["deferred"]["version"] == "1.1.5"
        time.sleep(1.0)
        assert not log.exists()                                  # nothing before the sign-out
        st, _, _ = s.req("api/shutdown", {"mode": "now"})
        assert st == 202
        assert s.gone(30)
        t_gone = time.time()
        rec = json.loads(_wait(lambda: log.exists() and log.read_text(), timeout=60))
        assert rec["server_alive"] is False
        assert rec["argv"][:2] == ["--channel", "release"] and "--yes" in rec["argv"] and "--no-pip" in rec["argv"]
        assert rec["t"] >= t_gone - 1.0
    finally:
        s.kill()
    # the result shows at the next start
    state = tmp_path / "xdg" / "cache" / "mace" / "gui" / "update.json"
    _wait(lambda: state.exists() and json.loads(state.read_text()).get("result"), timeout=30)
    s = Srv(tmp_path, **env)
    try:
        d = _status(s)
        assert d["result"]["ok"] is True and d["result"]["how"] == "signout"
        assert "Updated (fake)." in d["result"]["output"]
        st, d, _ = s.req("api/update/choose", {"choice": "ack"})
        assert st == 200 and d["result"] is None
    finally:
        s.kill()


def test_signout_without_a_deferred_update_runs_nothing(origin, tmp_path):
    home = clone(origin, tmp_path / "me")
    fake = tmp_path / "fake_update.py"
    fake.write_text(FAKE_UPDATE)
    log = tmp_path / "fake.json"
    env = _env(home, MACE_GUI_TEST_UPDATE_ARGV=json.dumps([sys.executable, str(fake)]), FAKE_UPDATE_LOG=str(log),
               FAKE_SERVER_PID_FILE=str(tmp_path / "pid"))
    s = Srv(tmp_path, **env)
    try:
        _check(s)
        s.req("api/update/choose", {"choice": "signout"})
        s.req("api/update/choose", {"choice": "cancel-signout"})
        s.req("api/shutdown", {"mode": "now"})
        assert s.gone(30)
        time.sleep(2)
        assert not log.exists()
    finally:
        s.kill()


def test_update_now_under_http_control_keeps_the_ready_file(origin, tmp_path):
    """mace gui --slurm's server (--managed --http-control --ready-file): the
    restart keeps the ready file (same port and token) and writes no
    stopped.json, so the login node's mace gui --slurm stays attached."""
    home = clone(origin, tmp_path / "me")
    root = tmp_path / "served"
    root.mkdir()
    ready = tmp_path / "job" / "ready.json"
    e = H.env(tmp_path / "xdg", MACE_GUI_NO_WARMUP="1", **_env(home))
    p = H.ServerProcess([sys.executable, str(H.MACE_CLI), "--no-banner", "gui", "--dir", str(root), "--port", "0",
                         "--managed", "--http-control", "--ready-file", str(ready)],
                        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=e,
                        start_new_session=True, preexec_fn=H.die_with_pytest)
    try:
        r = json.loads(_wait(lambda: ready.exists() and ready.read_text(), timeout=60))

        def req(route, body=None):
            c = http.client.HTTPConnection("127.0.0.1", r["port"], timeout=30)
            c.request("POST" if body is not None else "GET", r["prefix"] + route,
                      body=json.dumps(body) if body is not None else None,
                      headers={"X-Mace-Token": r["token"], "X-Requested-With": "mace-gui",
                               "Content-Type": "application/json"})
            x = c.getresponse()
            return x.status, json.loads(x.read())
        req("api/managed/attach", {"watcher": "w1"})
        st, before = req("api/update/check", {})
        assert st == 200 and before["restart"]["mode"] == "slurm"
        st, d = req("api/update/choose", {"choice": "now"})
        assert st == 202, d

        def back():
            try:
                st, d = req("api/update")
            except OSError:
                return None
            return d if st == 200 and d["started"] != before["started"] else None
        d = _wait(back, timeout=120, step=0.5)
        assert d["result"]["ok"] is True and p.poll() is None
        r2 = json.loads(ready.read_text())
        assert r2["token"] == r["token"] and r2["port"] == r["port"]
        assert not (ready.parent / "stopped.json").exists()
    finally:
        p.kill()
        p.wait(5)


def test_update_now_under_http_control_survives_past_the_managed_timeout(origin, tmp_path):
    """The restarted --slurm server keeps the watcher list: the login node's
    client only pings as the watcher it attached as before the restart, and the
    server must not stop at MACE_GUI_MANAGED_TIMEOUT (it did, after 300 s)."""
    home = clone(origin, tmp_path / "me")
    root = tmp_path / "served"
    root.mkdir()
    ready = tmp_path / "job" / "ready.json"
    e = H.env(tmp_path / "xdg", MACE_GUI_NO_WARMUP="1", MACE_GUI_MANAGED_TIMEOUT="4", MACE_GUI_MANAGED_GRACE="3",
              **_env(home))
    p = H.ServerProcess([sys.executable, str(H.MACE_CLI), "--no-banner", "gui", "--dir", str(root), "--port", "0",
                         "--managed", "--http-control", "--ready-file", str(ready)],
                        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=e,
                        start_new_session=True, preexec_fn=H.die_with_pytest)
    try:
        r = json.loads(_wait(lambda: ready.exists() and ready.read_text(), timeout=60))

        def req(route, body=None):
            c = http.client.HTTPConnection("127.0.0.1", r["port"], timeout=30)
            c.request("POST" if body is not None else "GET", r["prefix"] + route,
                      body=json.dumps(body) if body is not None else None,
                      headers={"X-Mace-Token": r["token"], "X-Requested-With": "mace-gui",
                               "Content-Type": "application/json"})
            x = c.getresponse()
            return x.status, json.loads(x.read())
        req("api/managed/attach", {"watcher": "w1"})
        st, before = req("api/update/check", {})
        st, d = req("api/update/choose", {"choice": "now"})
        assert st == 202, d

        def back():
            try:
                req("api/managed/ping", {"watcher": "w1"})
                st, d = req("api/update")
            except OSError:
                return None
            return d if st == 200 and d["started"] != before["started"] else None
        _wait(back, timeout=120, step=0.5)
        end = time.monotonic() + 10                  # 2.5x the managed timeout
        while time.monotonic() < end:
            st, _ = req("api/managed/ping", {"watcher": "w1"})
            assert st == 200 and p.poll() is None
            time.sleep(1.0)
        assert not (ready.parent / "stopped.json").exists()
    finally:
        p.kill()
        p.wait(5)


def test_a_lingering_managed_server_survives_update_now(origin, tmp_path):
    """The first ssh session is gone (stdin at EOF) and a reattached connect keeps
    the server: after Update Now the new process must not read the dead stdin
    and stop, and the reattached watcher stays attached."""
    home = clone(origin, tmp_path / "me")
    s = Srv(tmp_path, managed=True, MACE_GUI_MANAGED_TIMEOUT="4", MACE_GUI_MANAGED_GRACE="3", **_env(home))
    try:
        st, d, _ = s.req("api/managed/attach", {"watcher": "w1"})
        assert st == 200, d
        s.p.stdin.close()                             # the first connection is gone: it lingers for w1
        time.sleep(1.5)
        assert s.p.poll() is None
        before = _check(s)
        st, d, _ = s.req("api/update/choose", {"choice": "now"})
        assert st == 202, d

        def back():
            try:
                s.req("api/managed/ping", {"watcher": "w1"})
            except OSError:
                return "dead" if s.p.poll() is not None else None
            return _restarted(s, before["started"])
        got = _wait(back, timeout=120, step=0.5)
        assert got != "dead", "".join(s.lines)[-2000:]
        end = time.monotonic() + 8
        while time.monotonic() < end:
            st, _, _ = s.req("api/managed/ping", {"watcher": "w1"})
            assert st == 200 and s.p.poll() is None
            time.sleep(1.0)
        # a later connect still finds it (the lingering state file is back)
        files = list((tmp_path / "xdg" / "cache" / "mace" / "gui").glob("managed-*.json"))
        assert files, "the lingering state file was not rewritten after the restart"
    finally:
        s.kill()


def _checkout_copy(dst):
    """A copy of this checkout's mace/ package and requirements.txt."""
    import shutil
    src = Path(__file__).resolve().parents[2]
    shutil.copytree(src / "mace", dst / "mace", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(src / "requirements.txt", dst / "requirements.txt")
    return dst


def test_the_restart_check_loads_the_gui_modules(tmp_path):
    """The pre-restart check imports what the restarted server loads (it used to
    run mace gui --help, which imports no GUI code)."""
    from mace.gui import update_gui
    good = _checkout_copy(tmp_path / "good")
    code, out = update_gui.smoke(good)
    assert code == 0, out
    syn = _checkout_copy(tmp_path / "syntax")
    with open(syn / "mace" / "gui" / "server.py", "a") as fh:
        fh.write("\ndef broken(:\n")
    code, out = update_gui.smoke(syn)
    assert code == 1 and "server.py" in out
    imp = _checkout_copy(tmp_path / "import")
    p = imp / "mace" / "gui" / "jobs.py"
    p.write_text(p.read_text() + "\nfrom mace.gui import no_such_module_xyz\n")
    code, out = update_gui.smoke(imp)
    assert code == 1 and "mace.gui.jobs" in out
    req = _checkout_copy(tmp_path / "req")
    p = req / "mace" / "gui" / "jobs.py"
    p.write_text(p.read_text() + "\nimport no_such_requirement_xyz\n")
    with open(req / "requirements.txt", "a") as fh:
        fh.write("no-such-requirement-xyz>=1.0\n")
    code, out = update_gui.smoke(req)
    assert code == 1 and "no_such_requirement_xyz" in out
    opt = _checkout_copy(tmp_path / "optional")       # a missing optional module is not an error
    p = opt / "mace" / "gui" / "jobs.py"
    p.write_text(p.read_text() + "\ntry:\n    import no_such_optional_xyz\nexcept ImportError:\n    pass\n")
    assert update_gui.smoke(opt)[0] == 0


def test_update_now_keeps_the_old_server_when_the_new_code_would_not_start(origin, tmp_path):
    home = clone(origin, tmp_path / "me")
    broken = _checkout_copy(tmp_path / "broken")
    with open(broken / "mace" / "gui" / "server.py", "a") as fh:
        fh.write("\ndef broken(:\n")
    s = Srv(tmp_path, **_env(home, MACE_GUI_TEST_SMOKE_HOME=str(broken)))
    try:
        before = _check(s)
        st, d, _ = s.req("api/update/choose", {"choice": "now"})
        assert st == 202, d
        d = _wait(lambda: (_status(s)["phase"] == "done") and _status(s), timeout=120, step=0.5)
        assert d["result"]["ok"] is True and d["result"]["restart"] == "manual"
        assert "would not start" in d["result"]["restart_why"] and "server.py" in d["result"]["restart_why"]
        assert d["started"] == before["started"] and s.p.poll() is None
    finally:
        s.kill()


def test_two_update_now_clicks_start_one_update(tmp_path, monkeypatch):
    import threading
    import types
    from mace.gui import update_gui
    monkeypatch.setenv("MACE_GUI_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    started = []
    gate = threading.Barrier(8)
    seen = threading.local()

    class Tasks:
        def running(self):
            if not getattr(seen, "once", False):
                seen.once = True
                gate.wait(5)                  # every request passes the early checks together
            return []

        def running_count(self):
            return 0

        def broadcast(self, *a, **k):
            pass
    st = types.SimpleNamespace(test_hooks=False, tasks=Tasks(), stopping=threading.Event(), bind_host="127.0.0.1",
                               request_stop=lambda r: None, managed=None)
    u = update_gui.Updater(st)
    monkeypatch.setattr(update_gui.Updater, "_now", lambda self, r, pip: started.append(1))
    u.report = {"available": True, "channel": "release", "target": {"ref": "v9", "version": "9"},
                "current": {"version": "1"}}
    codes = []
    ts = [threading.Thread(target=lambda: codes.append(u.choose({"choice": "now"})[0])) for _ in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(10)
    assert sorted(codes) == [202] + [409] * 7 and len(started) == 1
