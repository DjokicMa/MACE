"""`mace gui connect` end to end, no network: a fake ssh (fake_ssh.py) runs the
remote command under env -i with an empty HOME, for the jump and the nested
hop (stdin and its EOF cross two fake ssh processes), real mace_cli
subprocesses on both ends, and a fresh headless Firefox that opens the URL
connect prints and clicks the Explorer: the copied real test/ files are
listed. Then Ctrl+C, kill -9 of the ssh (no task: the server stops; with a
task: it lingers, a second connect reattaches and the Tasks panel shows the
task finishing), and the standalone bash script through the same fake ssh."""
import json
import os
import pty
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

import guiharness as H

pytestmark = pytest.mark.gui
REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "test"
FILES = ["OPT/1_dia_opt_rev1.out", "OPT/1_dia_opt_rev1.d12", "CIFs/1_dia_opt_rev1.cif"]
FAKE = Path(__file__).with_name("fake_ssh.py")


@pytest.fixture
def world(tmp_path):
    if not all((CORPUS / f).is_file() for f in FILES):
        pytest.skip("test/ corpus not present")
    if shutil.which("bash") is None:
        pytest.skip("needs bash")
    data = tmp_path / "data"
    data.mkdir()
    for f in FILES:
        shutil.copy(CORPUS / f, data / Path(f).name)
    fakebin = tmp_path / "fakebin"
    fakebin.mkdir()
    (fakebin / "ssh").symlink_to(FAKE)
    os.chmod(FAKE, 0o755)
    node_home, log = tmp_path / "nodehome", tmp_path / "sshlog"
    node_home.mkdir()
    log.mkdir()
    node_tmp = tmp_path / "nodetmp"
    node_tmp.mkdir()
    node_env = {"XDG_CACHE_HOME": str(tmp_path / "nodecache"), "XDG_CONFIG_HOME": str(tmp_path / "nodecfg"),
                "TMPDIR": str(node_tmp), "MACE_GUI_NO_WARMUP": "1", "MACE_GUI_TEST_HOOKS": "1",
                "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1"}
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONPATH", "MACE_HOME", "COLORTERM", "TERM", "NO_COLOR", "BROWSER", "SSH_CONNECTION")}
    env.update(PATH=f"{fakebin}:{env.get('PATH', '/usr/bin:/bin')}", FAKE_SSH_HOME=str(node_home),
               FAKE_SSH_LOG=str(log), FAKE_SSH_BIN=str(fakebin), FAKE_SSH_ENV=json.dumps(node_env),
               XDG_CONFIG_HOME=str(tmp_path / "localcfg"), PYTHONDONTWRITEBYTECODE="1")
    procs = []
    yield {"tmp": tmp_path, "data": data, "env": env, "log": log, "node_tmp": node_tmp, "procs": procs,
           "mace": f"{sys.executable} {REPO / 'mace_cli'}"}
    for p in procs:
        if p.poll() is None:
            p.kill()
            p.wait(5)
    # nothing left on the "node"
    for f in log.glob("*.json"):
        pid = json.loads(f.read_text())["child"]
        if not pid:
            continue
        try:
            os.killpg(pid, signal.SIGKILL)
        except OSError:
            pass


class Connect:
    """A `mace gui connect` subprocess; its output collected in a thread."""

    def __init__(self, w, *extra, hop="jump", tty=False, script=False):
        lp = H.free_port()
        if script:
            argv = ["bash", str(REPO / "scripts" / "mace-gui-connect.sh")]
        else:
            argv = [sys.executable, str(REPO / "mace_cli"), "--no-banner", "gui", "connect"]
        argv += ["--gateway", "fakegw", "--node", "n1", "--dir", str(w["data"]), "--no-open", "--hop", hop,
                 "--mace", w["mace"], "--local-port", str(lp), *extra]
        self.master = None
        if tty:
            self.master, slave = pty.openpty()
            stdin = slave
        else:
            stdin = subprocess.PIPE
        self.p = subprocess.Popen(argv, env=w["env"], stdin=stdin, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, start_new_session=True)
        if tty:
            os.close(slave)
        w["procs"].append(self.p)
        self.lp = lp
        self.out = []
        threading.Thread(target=lambda: [self.out.append(x) for x in self.p.stdout], daemon=True).start()

    def text(self):
        return "".join(self.out)

    def wait_for(self, pattern, timeout=60):
        end = time.time() + timeout
        while time.time() < end:
            m = re.search(pattern, self.text())
            if m:
                return m
            if self.p.poll() is not None:
                time.sleep(0.3)
                m = re.search(pattern, self.text())
                if m:
                    return m
                raise AssertionError(f"connect ended ({self.p.returncode}):\n{self.text()}")
            time.sleep(0.1)
        raise AssertionError(f"no {pattern!r} in:\n{self.text()}")

    def url(self):
        return self.wait_for(r"MACE GUI on n1: (\S+)").group(1)

    def sigint(self):
        os.kill(self.p.pid, signal.SIGINT)


def outer_ssh(w):
    """The fake ssh that connect started (its argv names the gateway or the node)."""
    recs = [json.loads(f.read_text()) for f in w["log"].glob("*.json")]
    outer = [r for r in recs if "fakegw" in " ".join(r["argv"])]
    return outer[-1]


def server_pid(w):
    """The node's server: the remote child whose command runs mace gui --managed."""
    recs = [json.loads(f.read_text()) for f in w["log"].glob("*.json")]
    inner = [r for r in recs if r["target"].endswith("n1")]
    return inner[-1]["child"]


def alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        return "Z" not in open(f"/proc/{pid}/stat").read().split(")")[1].split()[0]
    except OSError:
        return False


def wait_gone(pid, seconds):
    end = time.time() + seconds
    while time.time() < end:
        if not alive(pid):
            return True
        time.sleep(0.1)
    return False


def port_free(port):
    """Nothing listens there any more (TIME_WAIT leftovers do not count)."""
    with socket.socket() as s:
        s.settimeout(2)
        try:
            s.connect(("127.0.0.1", port))
            return False
        except OSError:
            return True


def open_and_check_explorer(drv, url):
    drv.get(url)
    H.wait(drv, "return document.documentElement.dataset.appReady === '1'", 60)
    btn = drv.find_element("css selector", '#activity [data-cmd="view.explorer"]')
    if btn.get_attribute("aria-pressed") != "true":
        btn.click()
    names = H.wait(drv, """const r = [...document.querySelectorAll('#file-list .file')].map(e => e.dataset.path);
        return r.length >= 3 ? r : null""", 30)
    assert {"1_dia_opt_rev1.out", "1_dia_opt_rev1.d12", "1_dia_opt_rev1.cif"} <= set(names), names


@pytest.mark.parametrize("hop", ["jump", "nested"])
def test_connect_opens_the_real_files_and_ctrl_c_stops_the_server(world, hop):
    c = Connect(world, hop=hop)
    url = c.url()
    assert url.startswith(f"http://127.0.0.1:{c.lp}/s/")
    rec = outer_ssh(world)
    if hop == "jump":
        assert rec["jumps"] == ["fakegw"] and rec["target"].endswith("n1")
    else:
        assert rec["target"] == "fakegw" and "BatchMode=yes" in " ".join(rec["argv"])
    pid = server_pid(world)
    drv = H.firefox()
    try:
        open_and_check_explorer(drv, url)
    finally:
        drv.quit()
    assert list(world["node_tmp"].glob("mace_gui_cache_*"))
    c.sigint()
    c.wait_for(r"Stopped the server on n1\.", 15)
    assert wait_gone(pid, 5), "the node's server is still running"
    assert c.p.wait(10) == 0
    assert port_free(c.lp)
    assert not list(world["node_tmp"].glob("mace_gui_cache_*"))


@pytest.mark.parametrize("hop", ["jump", "nested"])
def test_kill_9_of_ssh_stops_an_idle_server(world, hop):
    c = Connect(world, hop=hop)
    c.url()
    pid = server_pid(world)
    os.kill(outer_ssh(world)["pid"], signal.SIGKILL)
    assert wait_gone(pid, 10), "the server outlived the connection with no task running"
    assert not list(world["node_tmp"].glob("mace_gui_cache_*"))


def test_kill_9_during_a_task_lingers_and_a_second_connect_reattaches(world):
    import http.client
    tok = "e2e-node-token-0001"
    world["env"]["FAKE_SSH_ENV"] = json.dumps({**json.loads(world["env"]["FAKE_SSH_ENV"]), "MACE_GUI_TOKEN": tok})
    c = Connect(world)
    url = c.url()
    pid = server_pid(world)
    prefix = re.search(r"(/s/[^/]+/)", url).group(1)
    conn = http.client.HTTPConnection("127.0.0.1", c.lp, timeout=10)
    conn.request("POST", prefix + "api/test/task", body=json.dumps({"seconds": 30, "job": "sleep", "wait": 0,
                                                                    "title": "Long test task"}),
                 headers={"X-Mace-Token": tok, "X-Requested-With": "mace-gui", "Content-Type": "application/json"})
    assert conn.getresponse().status == 202
    os.kill(outer_ssh(world)["pid"], signal.SIGKILL)
    time.sleep(2)
    assert alive(pid), "the server must linger while its task runs"
    c2 = Connect(world, tty=True)
    m = c2.wait_for(r"Reattached to the server still running on n1 \((\d) tasks? running\)")
    if m.group(1) != "1":
        conn = http.client.HTTPConnection("127.0.0.1", c2.lp, timeout=10)
        conn.request("GET", prefix + "api/tasks", headers={"X-Mace-Token": tok})
        raise AssertionError(conn.getresponse().read().decode()[:2000])
    url2 = c2.url()
    drv = H.firefox()
    try:
        drv.get(url2)
        H.wait(drv, "return document.documentElement.dataset.appReady === '1'", 60)
        drv.find_element("css selector", '#activity [data-cmd="view.tasks"]').click()
        H.wait(drv, """const r = [...document.querySelectorAll('.task-row')].find(
            e => e.textContent.includes("Long test task")); return r && r.classList.contains("s-done")""", 60)
    finally:
        drv.quit()
    assert alive(pid)
    c2.sigint()
    time.sleep(1.0)
    if c2.master is not None:
        os.write(c2.master, b"y\n")
    c2.wait_for(r"Stopped the server on n1", 20)
    assert wait_gone(pid, 10)


def test_bash_script_through_the_fake_ssh(world):
    tmp = world["tmp"]
    marker = tmp / "opened.txt"
    stub = tmp / "browser-stub"
    stub.write_text(f"#!/bin/sh\necho \"$1\" >> {marker}\n")
    stub.chmod(0o755)
    world["env"]["BROWSER"] = str(stub)
    c = Connect(world, script=True)
    # --no-open is passed: the URL is printed only
    url = c.url()
    pid = server_pid(world)
    drv = H.firefox()
    try:
        open_and_check_explorer(drv, url)
    finally:
        drv.quit()
    os.kill(c.p.pid, signal.SIGINT)
    assert wait_gone(pid, 15)
    assert c.p.wait(15) in (0, 130)
    assert port_free(c.lp)
    left = [p for p in Path(os.environ.get("TMPDIR", "/tmp")).glob("mace-gui-connect.*")
            if p.stat().st_mtime > time.time() - 120]
    assert not left
