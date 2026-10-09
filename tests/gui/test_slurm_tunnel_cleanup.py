"""`mace gui --slurm`: the ssh tunnel it starts never outlives it. A fake
`ssh` (a bash wrapper that writes its PID, and optionally a child's, to a
file, then forwards like ssh -N until it is killed) stands in for ssh. After
each way the session can end, every PID it wrote is gone: Sign Out and Stop
Server (the shutdown API, through the tunnel), the `stop` control line,
Ctrl+C, SIGTERM, SIGHUP, SIGKILL of the session (Linux: the tunnel gets
SIGTERM when its parent dies), an error after the tunnel opened, Ctrl+C
while the tunnel comes up, and an interpreter exit that skipped close()."""
import json
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
import urllib.request
from pathlib import Path

import pytest

from test_slurm_session_e2e import FAKE_SSH, REPO, Slurm, _sigint_default, _wait_gone, cluster  # noqa: F401

from mace.gui import slurmsession as ss


def _fake_tunnel(tmp, child=False):
    """The fake ssh: PIDs go to <tmp>/tunnel.pids, one per line."""
    pids = tmp / "tunnel.pids"
    f = tmp / "fake_tunnel_ssh.sh"
    f.write_text("#!/bin/bash\n"
                 f"echo $$ >> '{pids}'\n"
                 + (f"sleep 1000000 & echo $! >> '{pids}'\n" if child else "")
                 + f"exec '{sys.executable}' '{FAKE_SSH}' \"$@\"\n")
    f.chmod(0o755)
    return f, pids


def _running(pid):
    try:
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().rsplit(")", 1)[1].split()[0] != "Z"     # a zombie has ended
    except FileNotFoundError:
        return False
    except OSError:
        pass
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def _pids(path, n, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        got = [int(x) for x in path.read_text().split()] if path.exists() else []
        if len(got) >= n:
            return got
        time.sleep(0.1)
    raise AssertionError(f"the fake ssh wrote {path.read_text() if path.exists() else 'nothing'}")


def _all_gone(pids, timeout=15):
    end = time.time() + timeout
    while time.time() < end:
        if not any(_running(p) for p in pids):
            return True
        time.sleep(0.1)
    return False


def _kill(pids):
    for p in pids:
        try:
            os.kill(p, signal.SIGKILL)
        except OSError:
            pass


class Run(Slurm):
    """Slurm (as connect starts it), with the fake tunnel and Ctrl+C as in a terminal."""

    def __init__(self, c, child=True):
        ssh, self.pidfile = _fake_tunnel(c["tmp"], child)
        self.port = ss.free_port()
        self.p = subprocess.Popen([sys.executable, str(REPO / "mace_cli"), "--no-banner", "gui", "--slurm",
                                   "--managed", "--port", str(self.port), "--dir", str(c["root"])],
                                  env={**c["env"], "MACE_GUI_SLURM_SSH": str(ssh)}, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                  preexec_fn=_sigint_default)
        c["procs"].append(self.p)
        self.events, self.err = [], []
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=lambda: [self.err.append(x) for x in self.p.stderr], daemon=True).start()
        self.ready = self.wait("MACE_GUI_READY")
        self.pids = _pids(self.pidfile, 2 if child else 1)
        assert all(_running(p) for p in self.pids)


@pytest.mark.parametrize("how", ["shutdown-api", "stop-line", "sigint"])
def test_the_tunnel_ends_when_the_server_stops(cluster, how):
    s = Run(cluster)
    try:
        r = s.ready
        if how == "shutdown-api":           # Sign Out and Stop Server, through the tunnel
            req = urllib.request.Request(f"http://127.0.0.1:{s.port}{r['prefix']}api/shutdown",
                                         data=json.dumps({"mode": "now"}).encode(), method="POST",
                                         headers={"X-Mace-Token": r["token"], "X-Requested-With": "mace-gui",
                                                  "Content-Type": "application/json"})
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(req, timeout=10) as resp:
                assert resp.status == 202
        elif how == "stop-line":
            s.send("stop")
        else:
            s.p.send_signal(signal.SIGINT)
        assert s.p.wait(60) == 0, "".join(s.err)
        _wait_gone(cluster["fake"], r["job"])
        assert _all_gone(s.pids), [p for p in s.pids if _running(p)]
    finally:
        _kill(s.pids)


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGHUP])
def test_the_tunnel_ends_on_a_termination_signal(cluster, sig):
    s = Run(cluster)
    try:
        s.p.send_signal(sig)
        assert s.p.wait(30) == 128 + sig, "".join(s.err)
        assert _all_gone(s.pids), [p for p in s.pids if _running(p)]
    finally:
        _kill(s.pids)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="PR_SET_PDEATHSIG is Linux only")
def test_the_tunnel_ends_when_the_session_is_killed(cluster):
    """SIGKILL: nothing runs in the session; the kernel sends the tunnel
    SIGTERM (ssh then ends its own children; this fake has none)."""
    s = Run(cluster, child=False)
    try:
        s.p.kill()
        s.p.wait(10)
        assert _all_gone(s.pids), [p for p in s.pids if _running(p)]
    finally:
        _kill(s.pids)


# ------------------------------------------------------------------ in process

@pytest.fixture
def tunnel_ssh(tmp_path, monkeypatch):
    ssh, pids = _fake_tunnel(tmp_path, child=True)
    monkeypatch.setenv("MACE_GUI_SLURM_SSH", str(ssh))
    for k in ("MACE_HOME", "XDG_CONFIG_HOME"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("MACE_SITE_CONFIG", str(tmp_path / "nosite.toml"))   # no site settings
    monkeypatch.setenv("MACE_SITE_GROUP_CONFIG", "")
    monkeypatch.setenv("XDG_CONFIG_DIRS", str(tmp_path / "noxdg"))
    yield pids
    if pids.exists():
        _kill(int(x) for x in pids.read_text().split())


def test_an_error_after_the_tunnel_opened(tmp_path, monkeypatch, tunnel_ssh):
    seen = []

    def start(self):
        self.tunnel = ss.Tunnel("node01", ss.free_port(), 5555)
        seen.extend(_pids(tunnel_ssh, 2))
        raise RuntimeError("something broke")
    monkeypatch.setattr(ss.Session, "start", start)
    with pytest.raises(RuntimeError, match="something broke"):
        ss.main(["--managed", "--dir", str(tmp_path)])
    assert seen and _all_gone(seen, 5), [p for p in seen if _running(p)]
    assert not ss._TUNNELS


def test_ctrl_c_while_the_tunnel_comes_up(monkeypatch, tunnel_ssh):
    def api(*a, **k):
        _pids(tunnel_ssh, 2)
        raise KeyboardInterrupt
    monkeypatch.setattr(ss, "_api", api)
    with pytest.raises(KeyboardInterrupt):
        ss.open_tunnel("node01", ss.free_port(), {"port": 5555, "prefix": "/p/", "token": "t"}, wait_s=20)
    pids = _pids(tunnel_ssh, 2)
    assert _all_gone(pids, 5), [p for p in pids if _running(p)]
    assert not ss._TUNNELS


def test_a_tunnel_never_closed_ends_at_exit(tmp_path, tunnel_ssh):
    code = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(REPO)!r})
        from mace.gui import slurmsession as ss
        ss.Tunnel("node01", ss.free_port(), 5555)
        import time
        end = time.time() + 30
        while time.time() < end and len(open({str(tunnel_ssh)!r}).read().split() if
                                          __import__("os").path.exists({str(tunnel_ssh)!r}) else []) < 2:
            time.sleep(0.05)
        sys.exit(3)
    """)
    p = subprocess.run([sys.executable, "-c", code], env=os.environ.copy(), timeout=60)
    assert p.returncode == 3
    pids = _pids(tunnel_ssh, 2)
    assert _all_gone(pids, 5), [p for p in pids if _running(p)]


def test_a_dead_tunnel_takes_its_group_with_it(tunnel_ssh):
    """ssh ending on its own (a dropped connection): what is left of its
    process group goes at once, not just at close()."""
    t = ss.Tunnel("node01", ss.free_port(), 5555)
    try:
        leader, child = _pids(tunnel_ssh, 2)
        os.kill(leader, signal.SIGKILL)
        end = time.time() + 10
        while t.alive() and time.time() < end:
            time.sleep(0.05)
        assert not t.alive()
        assert _all_gone([child], 5)
    finally:
        t.close()
