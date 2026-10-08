"""`mace gui login-url` for a server started by `mace gui connect`, no network
and no browser: the fake ssh (fake_ssh.py) runs the node's `mace gui --managed`
under env -i, real mace_cli subprocesses on both ends. On the connect computer
login-url prints a fresh link on the forwarded port that signs in through the
tunnel; on the node it says to run it on the connect computer (the node's
own port is not the one the browser reaches)."""
import http.client
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

import pytest

import guiharness as H
from test_connect_e2e import FAKE, REPO, Connect, server_pid, wait_gone


@pytest.fixture
def world(tmp_path):
    if shutil.which("bash") is None:
        pytest.skip("needs bash")
    data = tmp_path / "data"
    data.mkdir()
    (data / "note.txt").write_text("served\n")
    fakebin = tmp_path / "fakebin"
    fakebin.mkdir()
    (fakebin / "ssh").symlink_to(FAKE)
    os.chmod(FAKE, 0o755)
    node_home, log, node_tmp, local_run = (tmp_path / n for n in ("nodehome", "sshlog", "nodetmp", "localrun"))
    for d in (node_home, log, node_tmp, local_run):
        d.mkdir()
    node_env = {"XDG_CACHE_HOME": str(tmp_path / "nodecache"), "XDG_CONFIG_HOME": str(tmp_path / "nodecfg"),
                "TMPDIR": str(node_tmp), "MACE_GUI_NO_WARMUP": "1", "PYTHONDONTWRITEBYTECODE": "1",
                "OMP_NUM_THREADS": "1"}
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONPATH", "MACE_HOME", "COLORTERM", "TERM", "NO_COLOR", "BROWSER", "SSH_CONNECTION",
                        "MACE_GUI_TOKEN", "MACE_GUI_PORT")}
    env.update(PATH=f"{fakebin}:{env.get('PATH', '/usr/bin:/bin')}", FAKE_SSH_HOME=str(node_home),
               FAKE_SSH_LOG=str(log), FAKE_SSH_BIN=str(fakebin), FAKE_SSH_ENV=json.dumps(node_env),
               XDG_CONFIG_HOME=str(tmp_path / "localcfg"), XDG_CACHE_HOME=str(tmp_path / "localcache"),
               XDG_RUNTIME_DIR=str(local_run), PYTHONDONTWRITEBYTECODE="1")
    procs = []
    yield {"tmp": tmp_path, "data": data, "env": env, "node_env": node_env, "log": log, "local_run": local_run,
           "procs": procs, "mace": f"{sys.executable} {REPO / 'mace_cli'}"}
    for p in procs:
        if p.poll() is None:
            p.kill()
            p.wait(5)
    for f in log.glob("*.json"):
        pid = json.loads(f.read_text())["child"]
        if pid:
            try:
                os.killpg(pid, 9)
            except OSError:
                pass


def login_url(env, *extra):
    return subprocess.run([sys.executable, str(REPO / "mace_cli"), "--no-banner", "gui", "login-url", *extra],
                          env=env, capture_output=True, text=True, timeout=60)


def signs_in(link):
    u = urlsplit(link)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=10)
    conn.request("GET", f"{u.path}?{u.query}")
    r = conn.getresponse()
    r.read()
    return r.status == 303 and "set-cookie" in {k.lower() for k in r.headers.keys()}


def test_login_url_works_on_the_connect_computer_and_points_there_from_the_node(world):
    c = Connect(world)
    url = c.url()
    pid = server_pid(world)
    # the connect computer: a fresh link on the forwarded port, through the tunnel
    r = login_url(world["env"])
    assert r.returncode == 0, r.stderr
    link = r.stdout.strip()
    assert link.startswith(f"http://127.0.0.1:{c.lp}/s/") and "?login=" in link and link != url
    assert signs_in(link)
    assert not signs_in(link), "a sign-in link works once"
    st = json.loads((world["local_run"] / f"mace-gui-{c.lp}.json").read_text())
    assert st["via"] == "connect" and st["port"] == c.lp and oct(
        (world["local_run"] / f"mace-gui-{c.lp}.json").stat().st_mode & 0o777) == "0o600"
    # the node: the server is found, but its links come from the connect computer
    node = {k: v for k, v in world["env"].items() if k not in ("XDG_RUNTIME_DIR",)}
    node.update(world["node_env"])
    r = login_url(node)
    assert r.returncode == 1 and "no MACE GUI" not in r.stderr, r.stderr
    assert "started by mace gui connect" in r.stderr and "computer where" in r.stderr
    markers = list(Path(world["node_env"]["XDG_CACHE_HOME"], "mace", "gui").glob("url-*.json"))
    assert markers and all("token" not in json.loads(m.read_text()) for m in markers)
    # both files go when the session ends
    c.sigint()
    c.wait_for(r"Stopped the server on n1\.", 15)
    assert c.p.wait(10) == 0
    assert wait_gone(pid, 5)
    assert not (world["local_run"] / f"mace-gui-{c.lp}.json").exists()
    assert not list(Path(world["node_env"]["XDG_CACHE_HOME"], "mace", "gui").glob("url-*.json"))
    r = login_url(world["env"])
    assert r.returncode == 1 and "no MACE GUI" in r.stderr
