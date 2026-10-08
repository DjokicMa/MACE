"""`mace gui` start-up hints and `--managed` (T3), all through the real
mace_cli subprocess with a temp HOME: the exact-node hint with the connect
one-liner, the VS Code hint, --open with $BROWSER and the headless-node
guard, the managed READY line, the watchdog (stdin EOF, timeout, pings,
stop), lingering for a running task, reattach, a busy port, and the deny
page on a remapped port."""
import http.client
import json
import os
import re
import socket
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
MACE_CLI = REPO / "mace_cli"


def _env(tmp_path, **extra):
    e = {k: v for k, v in os.environ.items()
         if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH", "MACE_GUI_TOKEN", "MACE_GUI_PORT",
                      "SSH_CONNECTION", "SLURM_JOB_ID", "TERM_PROGRAM", "VSCODE_IPC_HOOK_CLI", "BROWSER")}
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    e.update(HOME=str(home), XDG_CONFIG_HOME=str(tmp_path / "cfg"), XDG_CACHE_HOME=str(tmp_path / "cache"),
             PYTHONDONTWRITEBYTECODE="1", MACE_GUI_NO_WARMUP="1", TMPDIR=str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir(exist_ok=True)
    e.update(extra)
    return e


def _banner(tmp_path, *args, env=None, n=30, linger=0.0):
    p = subprocess.Popen([sys.executable, str(MACE_CLI), "--no-banner", "gui", "--dir", str(tmp_path), "--port", "0",
                          *args], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, env=env)
    lines = []
    try:
        for _ in range(n):
            line = p.stdout.readline()
            if not line:
                break
            lines.append(line)
            if line.startswith("Press Ctrl+C"):
                break
        time.sleep(linger)           # what comes after the banner (--open)
    finally:
        p.terminate()
        p.wait(timeout=15)
    lines.append(p.stdout.read())
    return "".join(lines)


def test_ssh_session_hint_names_the_exact_node(tmp_path):
    out = _banner(tmp_path, "--gateway", "gw.example.edu",
                  env=_env(tmp_path, SSH_CONNECTION="10.0.0.1 5555 10.0.0.2 22", MACE_GUI_PRINT_TOKEN="1"))
    node = socket.gethostname()
    short = node.split(".")[0]
    assert re.search(rf"This server runs on ({re.escape(node)}|{re.escape(short)})\. The tunnel must go to this "
                     "exact node", out), out
    assert "From your computer, with MACE installed there:" in out
    assert re.search(rf"mace gui connect --gateway \S+@gw\.example\.edu --node ({re.escape(node)}|{re.escape(short)})"
                     r" --dir \S+", out), out
    port = re.search(r"127\.0\.0\.1:(\d+)/", out).group(1)
    assert re.search(rf"ssh -N -J \S+@gw\.example\.edu -L {port}:127\.0\.0\.1:{port} \S+@\S+", out), out
    assert "add --hop nested (see docs/gui.md)" in out
    assert "laptop" not in out


def test_vscode_terminal_gets_the_ctrl_click_hint_instead(tmp_path):
    out = _banner(tmp_path, env=_env(tmp_path, SSH_CONNECTION="10.0.0.1 5555 10.0.0.2 22", TERM_PROGRAM="vscode",
                                     MACE_GUI_PRINT_TOKEN="1"))
    assert "VS Code: this terminal forwards the port for you. Ctrl+click the sign-in link above" in out
    assert "The Ports tab opens\nthe bare address, which needs the token." in out
    assert "ssh -N -J" not in out
    assert re.search(r"http://127\.0\.0\.1:\d+/s/", out)


def _stub(path: Path, marker: Path):
    path.write_text(f"#!{sys.executable}\nimport sys\nopen({str(marker)!r}, 'a').write(' '.join(sys.argv) + '\\n')\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_open_uses_browser_env(tmp_path):
    marker = tmp_path / "opened.txt"
    stub = tmp_path / "mybrowser"
    _stub(stub, marker)
    _banner(tmp_path, "--open", env=_env(tmp_path, BROWSER=str(stub), MACE_GUI_PRINT_TOKEN="1"), linger=3.0)
    for _ in range(50):
        if marker.exists():
            break
        time.sleep(0.1)
    assert re.search(r"http://127\.0\.0\.1:\d+/s/\S+", marker.read_text())


def test_headless_ssh_node_never_starts_a_browser(tmp_path):
    if not sys.platform.startswith("linux"):
        pytest.skip("the guard is for Linux ssh nodes")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    marker = tmp_path / "launched.txt"
    os.symlink(sys.executable, bindir / "python3")
    for name in ("www-browser", "x-www-browser", "lynx", "w3m", "links", "elinks", "xdg-open", "gio", "firefox",
                 "chromium", "google-chrome", "sensible-browser"):
        _stub(bindir / name, marker)
    env = _env(tmp_path, SSH_CONNECTION="10.0.0.1 5555 10.0.0.2 22", PATH=str(bindir), MACE_GUI_PRINT_TOKEN="1")
    for k in ("DISPLAY", "WAYLAND_DISPLAY", "BROWSER"):
        env.pop(k, None)
    out = _banner(tmp_path, "--open", env=env, linger=2.0)
    assert "No browser on this node: open the URL on your computer through a tunnel" in out
    assert not marker.exists()


# ---------------------------------------------------------------- managed mode

class Managed:
    def __init__(self, root, env, port=0):
        self.p = subprocess.Popen([sys.executable, str(MACE_CLI), "--no-banner", "gui", "--managed", "--dir",
                                   str(root), "--port", str(port)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True, env=env, start_new_session=True)
        self.lines = []
        self.err = []
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=lambda: [self.err.append(x) for x in self.p.stderr], daemon=True).start()

    def _read(self):
        for line in self.p.stdout:
            self.lines.append(line.rstrip("\n"))

    def tagged(self, tag, timeout=30):
        end = time.time() + timeout
        while time.time() < end:
            for line in self.lines:
                if line.startswith(tag + " ") or line == tag:
                    rest = line[len(tag):].strip()
                    return json.loads(rest) if rest.startswith("{") else rest
            if self.p.poll() is not None and not any(x.startswith(tag) for x in self.lines):
                time.sleep(0.2)
                if not any(x.startswith(tag) for x in self.lines):
                    return None
            time.sleep(0.05)
        return None

    def send(self, line):
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()

    def close_stdin(self):
        self.p.stdin.close()

    def gone_within(self, seconds):
        try:
            self.p.wait(seconds)
            return True
        except subprocess.TimeoutExpired:
            return False

    def kill(self):
        if self.p.poll() is None:
            try:
                os.killpg(self.p.pid, 9)
            except OSError:
                pass
            self.p.wait(5)


def api(ready, route, body=None, port=None):
    c = http.client.HTTPConnection("127.0.0.1", port or ready["port"], timeout=10)
    h = {"X-Mace-Token": ready["token"], "X-Requested-With": "mace-gui", "Content-Type": "application/json"}
    c.request("POST" if body is not None else "GET", ready["prefix"] + "api/" + route,
              body=json.dumps(body) if body is not None else None, headers=h)
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, (json.loads(data) if data.startswith(b"{") else data)


@pytest.fixture
def mroot(tmp_path):
    d = tmp_path / "served"
    d.mkdir()
    return d


def test_ready_line_and_stdin_eof_stops_and_cleans_up(tmp_path, mroot):
    m = Managed(mroot, _env(tmp_path))
    try:
        ready = m.tagged("MACE_GUI_READY")
        assert ready is not None, m.err
        for k in ("url", "port", "prefix", "token", "login", "host", "fqdn", "root", "version", "reattached", "tasks"):
            assert k in ready, k
        assert ready["reattached"] is False and ready["root"] == str(mroot.resolve())
        assert api(ready, "info")[0] == 200
        caches = list((tmp_path / "tmp").glob("mace_gui_cache_*"))
        assert caches
        t0 = time.time()
        m.close_stdin()
        assert m.gone_within(3), "no stop after stdin EOF"
        assert time.time() - t0 < 3 and m.p.returncode == 0
        assert m.tagged("MACE_GUI_STOPPED", 2) == {"reason": "managed-disconnect"}
        assert not list((tmp_path / "tmp").glob("mace_gui_cache_*"))
        assert not list(Path(tmp_path / "cache").rglob("managed-*.json"))
    finally:
        m.kill()


def test_no_pings_stops_after_the_timeout_and_pings_keep_it_up(tmp_path, mroot):
    m = Managed(mroot, _env(tmp_path, MACE_GUI_MANAGED_TIMEOUT="3"))
    k = Managed(tmp_path / "home", _env(tmp_path, MACE_GUI_MANAGED_TIMEOUT="3"))
    try:
        assert m.tagged("MACE_GUI_READY") and k.tagged("MACE_GUI_READY")
        t0 = time.time()
        for _ in range(8):
            k.send("ping")
            time.sleep(1)
        assert m.gone_within(1) and time.time() - t0 < 9
        assert k.p.poll() is None, "pings every second kept it up"
        k.send("stop")
        assert k.gone_within(3) and k.tagged("MACE_GUI_STOPPED", 2) == {"reason": "managed-stop"}
    finally:
        m.kill()
        k.kill()


def test_busy_port_is_exit_98(tmp_path, mroot):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        m = Managed(mroot, _env(tmp_path), port=s.getsockname()[1])
        try:
            assert m.gone_within(30)
            assert m.p.returncode == 98 and "MACE_GUI_ERROR port-busy" in m.lines
        finally:
            m.kill()


def test_linger_for_a_running_task_then_reattach_then_stop_after_grace(tmp_path, mroot):
    env = _env(tmp_path, MACE_GUI_TEST_HOOKS="1", MACE_GUI_MANAGED_GRACE="2")
    m = Managed(mroot, env)
    r2 = None
    try:
        ready = m.tagged("MACE_GUI_READY")
        st, body = api(ready, "test/task", {"seconds": 10, "job": "sleep", "wait": 0})
        assert st == 202, body
        m.close_stdin()
        time.sleep(2.0)
        assert m.p.poll() is None, "the server must linger while a task runs"
        state = list((tmp_path / "cache").rglob("managed-*.json"))
        assert len(state) == 1 and stat.S_IMODE(state[0].stat().st_mode) == 0o600
        assert stat.S_IMODE(state[0].parent.stat().st_mode) == 0o700
        assert any("connection lost; keeping the server for 1 running task(s)" in e for e in m.err)
        # reattach: a second managed start for the same folder relays to the first
        r2 = Managed(mroot, env)
        ready2 = r2.tagged("MACE_GUI_READY")
        assert ready2["reattached"] is True and ready2["prefix"] == ready["prefix"] and ready2["tasks"] == 1
        assert ready2["port"] != ready["port"]
        st, info = api(ready2, "info")
        assert st == 200 and info["root"] == str(mroot.resolve())
        # closing the proxy's stdin detaches only
        r2.close_stdin()
        assert r2.gone_within(10)
        assert m.p.poll() is None
        # the task ends (10 s), then the 2 s grace (+ the watchdog's poll), then it stops
        assert m.gone_within(10 + 2 + 7), "no stop after the task and the grace"
        # nobody reads its stdout any more (it went to /dev/null when it started to linger)
        assert m.p.returncode == 0
        assert not list((tmp_path / "cache").rglob("managed-*.json"))
        assert not list((tmp_path / "tmp").glob("mace_gui_cache_*"))
    finally:
        m.kill()
        if r2 is not None:
            r2.kill()


def test_stop_line_stops_at_once_even_with_a_task(tmp_path, mroot):
    m = Managed(mroot, _env(tmp_path, MACE_GUI_TEST_HOOKS="1"))
    try:
        ready = m.tagged("MACE_GUI_READY")
        api(ready, "test/task", {"seconds": 30, "job": "sleep", "wait": 0})
        m.send("stop")
        assert m.gone_within(5)
    finally:
        m.kill()


def test_deny_page_on_a_remapped_port_mentions_vscode(tmp_path, mroot):
    """VS Code's Ports tab opens http://localhost:<other port>/: the token page,
    with the Ctrl+click line (through a stdlib forwarder)."""
    m = Managed(mroot, _env(tmp_path))
    try:
        ready = m.tagged("MACE_GUI_READY")
        lsock = socket.socket()
        lsock.bind(("127.0.0.1", 0))
        lsock.listen(4)
        lp = lsock.getsockname()[1]

        def fwd():
            while True:
                try:
                    c, _ = lsock.accept()
                except OSError:
                    return
                u = socket.create_connection(("127.0.0.1", ready["port"]))
                for a, b in ((c, u), (u, c)):
                    threading.Thread(target=lambda a=a, b=b: _pump(a, b), daemon=True).start()
        threading.Thread(target=fwd, daemon=True).start()
        c = http.client.HTTPConnection("127.0.0.1", lp, timeout=10)
        c.request("GET", "/", headers={"Host": f"localhost:{lp}"})
        r = c.getresponse()
        body = r.read().decode()
        assert r.status == 404 and "Ctrl+click" in body and "VS Code" in body
        lsock.close()
    finally:
        m.kill()


def _pump(a, b):
    try:
        while True:
            d = a.recv(65536)
            if not d:
                break
            b.sendall(d)
    except OSError:
        pass
    finally:
        for s in (a, b):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
