"""`mace gui` end to end through the real entry point: `mace_cli gui` as a
subprocess, spoken to with raw HTTP. The release gate for the server's
access rules (token, Host, session path, cookie scope, POST header, test
hooks, connection limits), its start-up rules (--host, --token,
MACE_GUI_TOKEN, MACE_GUI_PORT, the URL file when stdout is not a terminal)
and its shutdown. Nothing here needs ase, plotly or a browser.
"""
import http.client
import json
import os
import re
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from conftest import REPO_ROOT

MACE_CLI = REPO_ROOT / "mace_cli"
EXAMPLES = REPO_ROOT / "examples" / "gui"


def _env(tmp: Path, **extra):
    env = {k: v for k, v in os.environ.items()
           if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH", "MACE_GUI_TOKEN",
                        "MACE_GUI_PORT", "MACE_GUI_PRINT_TOKEN", "MACE_GUI_TEST_HOOKS", "SSH_CONNECTION",
                        "SLURM_JOB_ID")}
    run = tmp / "run"
    run.mkdir(mode=0o700, exist_ok=True)
    env.update(PYTHONDONTWRITEBYTECODE="1", XDG_RUNTIME_DIR=str(run), XDG_CACHE_HOME=str(tmp / "cache"),
               XDG_CONFIG_HOME=str(tmp / "config"), MACE_GUI_NO_WARMUP="1")
    env.update(extra)
    return env


def _foreground_sigint():
    """Ctrl+C at its default, as in a terminal: a pytest started in the
    background of a script has SIGINT ignored, which a child would inherit."""
    signal.signal(signal.SIGINT, signal.SIG_DFL)


def _launch(root: Path, tmp: Path, *args, env=None):
    return subprocess.Popen([sys.executable, str(MACE_CLI), "--no-banner", "gui", "--dir", str(root), *args],
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, env=env or _env(tmp), cwd=str(tmp), start_new_session=True,
                            preexec_fn=_foreground_sigint)


def _banner(proc, until="Press Ctrl+C", limit=40):
    lines = []
    for _ in range(limit):
        line = proc.stdout.readline()
        if not line:
            break
        lines.append(line)
        if line.startswith(until):
            break
    return "".join(lines)


def _stop(proc, timeout=10):
    """terminate, wait, kill, wait: a test never leaves a server behind."""
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    for f in (proc.stdout, proc.stderr):
        try:
            f.close()
        except Exception:
            pass


def _request(base, path, method="GET", headers=None, body=None, raw_path=False):
    u = urlsplit(base)
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=20)
    try:
        c.request(method, path if raw_path else u.path + path, body=body, headers=headers or {})
        r = c.getresponse()
        return r.status, {k.lower(): v for k, v in r.getheaders()}, r.read()
    finally:
        c.close()


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("guismoke")
    root = tmp / "examples"
    shutil.copytree(EXAMPLES, root)
    proc = _launch(root, tmp, "--port", "0")
    try:
        out = _banner(proc)
        m = re.search(r"(http://127\.0\.0\.1:(\d+)/s/[A-Za-z0-9_-]+/)\s", out)
        f = re.search(r"the token is in (\S+),", out)
        assert m and f, out
        url_file = Path(f.group(1))
        state = json.loads(url_file.read_text())
        token = state["token"]
        yield {"proc": proc, "base": m.group(1), "port": int(m.group(2)), "token": token, "out": out,
               "url_file": url_file, "state": state, "root": root, "tmp": tmp}
    finally:
        _stop(proc)


def _auth(srv):
    return {"X-Mace-Token": srv["token"]}


# --- the printed URL and the URL file -----------------------------------------------

def test_piped_stdout_prints_no_token_and_a_private_state_file(server):
    out = server["out"]
    assert server["token"] not in out
    assert "?token=" not in out and "?login=" not in out
    assert "mace gui login-url --port" in out
    st = server["url_file"].stat()
    assert stat.S_IMODE(st.st_mode) == 0o600
    assert server["state"]["base"] == server["base"] and server["state"]["pid"] == server["proc"].pid
    assert server["url_file"].parent == server["tmp"] / "run" and server["url_file"].suffix == ".json"


# --- access rules ---------------------------------------------------------------------

def test_no_token_is_403_and_the_page_says_what_to_do(server):
    status, _, body = _request(server["base"], "")
    assert status == 403
    assert b"<title>MACE</title>" in body and b"sign-in link" in body
    assert _request(server["base"], "api/info")[0] == 403
    assert _request(server["base"], "static/app.js")[0] == 403


def test_foreign_host_is_421(server):
    h = {**_auth(server), "Host": "attacker.example"}
    assert _request(server["base"], "api/info", headers=h)[0] == 421


def test_token_gives_the_page_with_its_csp(server):
    status, headers, body = _request(server["base"], "", headers=_auth(server))
    assert status == 200 and b"<title>MACE</title>" in body
    csp = headers["content-security-policy"]
    assert "frame-ancestors 'none'" in csp and "object-src 'none'" in csp
    assert "unsafe-eval" not in csp
    assert headers["x-content-type-options"] == "nosniff"


def test_a_login_link_becomes_a_cookie_scoped_to_the_session_path(server):
    st, _, body = _request(server["base"], "api/login-code", method="POST", headers={**_auth(server), "X-Requested-With": "mace-gui",
                                                                      "Content-Type": "application/json"}, body=b"{}")
    code = json.loads(body)["code"]
    status, headers, body = _request(server["base"], f"?login={code}")
    assert status == 303 and headers["location"] == "./"
    assert server["token"].encode() not in body
    cookie = headers["set-cookie"]
    assert server["token"] not in cookie
    path = re.search(r"Path=([^;]+)", cookie).group(1)
    assert path != "/" and path == urlsplit(server["base"]).path
    assert "HttpOnly" in cookie and "SameSite=Strict" in cookie


def test_outside_the_session_path_is_404_even_with_the_token(server):
    h = _auth(server)
    h_tok = {"X-Mace-Token": server["token"]}
    for path in ("/", "/index.html", "/api/info", "/static/app.js", "/x/api/info"):
        assert _request(server["base"], path, headers=h, raw_path=True)[0] == 404, path
        assert _request(server["base"], path, headers=h_tok, raw_path=True)[0] == 404, path
    status, _, body = _request(server["base"], "/", raw_path=True)
    assert status == 404 and b"<title>MACE</title>" in body
    # another session's path (a page from before a restart): 403, so the page
    # tells the user the server restarted; never served, even with the token
    for path in ("/s/wrong/api/info", "/s/wrong/static/app.js"):
        assert _request(server["base"], path, headers=h_tok, raw_path=True)[0] == 403, path
    status, _, body = _request(server["base"], "/s/wrong/", headers=h_tok, raw_path=True)
    assert status == 403 and b"<title>MACE</title>" in body


def test_session_path_without_its_slash_redirects(server):
    path = urlsplit(server["base"]).path
    status, headers, _ = _request(server["base"], path[:-1], raw_path=True)
    assert status == 301 and headers["location"] == path


@pytest.mark.parametrize("path", ["../../etc/passwd", "/etc/passwd", "outputs/../../x"])
def test_paths_outside_the_folder_are_refused(server, path):
    from urllib.parse import quote
    status, _, _ = _request(server["base"], "api/text?path=" + quote(path), headers=_auth(server))
    assert status in (403, 404)
    status, _, _ = _request(server["base"], "static/" + path.lstrip("/"), headers=_auth(server))
    assert status in (403, 404)


def test_post_needs_the_gui_header(server):
    h = {**_auth(server), "Content-Type": "application/json"}
    assert _request(server["base"], "api/deck/save", "POST", h, b"{}")[0] == 403
    assert _request(server["base"], "api/deck/save", "POST", {**h, "X-Requested-With": "other"}, b"{}")[0] == 403


TEST_ROUTES_GET = ["test/gate", "test/log", "test/dblog", "test/sessions", "test/parse_tmp", "test/held",
                   "test/mlip", "test/pool"]
TEST_ROUTES_POST = ["test/evict", "test/gate/reset", "test/parse_cache/clear", "test/task", "test/pool/config",
                    "test/held/config", "test/held/put", "test/cli", "test/sse/drop", "test/profile/fail", "test/root_gen"]


def test_every_test_route_is_404_without_the_hooks(server):
    for r in TEST_ROUTES_GET:
        assert _request(server["base"], "api/" + r, headers=_auth(server))[0] == 404, r
    h = {**_auth(server), "X-Requested-With": "mace-gui", "Content-Type": "application/json"}
    for r in TEST_ROUTES_POST:
        assert _request(server["base"], "api/" + r, "POST", h, b"{}")[0] == 404, r
    # the test-only builder, too
    st, _, _ = _request(server["base"], "api/build/identity", "POST", h, b'{"source": {}}')
    assert st == 404, st


def test_test_route_list_covers_the_server():
    """Every test/* route the server has is in the lists above."""
    src = (REPO_ROOT / "mace" / "gui" / "server.py").read_text()
    routes = set(re.findall(r'route == "(test/[a-z_/]+)"', src))
    assert routes and routes <= set(TEST_ROUTES_GET) | set(TEST_ROUTES_POST), routes - set(TEST_ROUTES_GET + TEST_ROUTES_POST)


# --- connection limits ----------------------------------------------------------------

def test_100_idle_sockets_do_not_block_an_authenticated_request(server):
    idle = []
    try:
        for _ in range(100):
            idle.append(socket.create_connection(("127.0.0.1", server["port"]), timeout=5))
        time.sleep(0.3)
        t0 = time.time()
        status, _, _ = _request(server["base"], "api/info", headers=_auth(server))
        assert status == 200 and time.time() - t0 < 10
    finally:
        for s in idle:
            s.close()


def test_an_idle_socket_is_dropped_after_the_handler_timeout(server):
    from mace.gui.server import HANDLER_TIMEOUT_S
    s = socket.create_connection(("127.0.0.1", server["port"]), timeout=HANDLER_TIMEOUT_S + 10)
    t0 = time.time()
    try:
        data = s.recv(1024)            # the server closes it: b"" (or a reset)
    except (ConnectionResetError, socket.timeout):
        data = None
    finally:
        s.close()
    took = time.time() - t0
    assert data == b"", data
    assert took <= HANDLER_TIMEOUT_S + 5, took


# --- shutdown (last: it stops the module's server) -------------------------------------

def test_sigterm_stops_it_and_removes_the_url_file(server):
    proc = server["proc"]
    t0 = time.time()
    os.kill(proc.pid, signal.SIGTERM)
    proc.wait(timeout=5)
    assert time.time() - t0 < 5
    assert proc.returncode == 0
    assert not server["url_file"].exists()


@pytest.mark.parametrize("sig", [signal.SIGHUP, signal.SIGINT], ids=["sighup", "sigint"])
def test_a_hangup_cleans_up_like_sigterm(tmp_path, sig):
    """A dropped ssh session or a closed terminal sends SIGHUP: the server
    stops within 5 s and removes its URL file (it holds the token) and its
    temporary cache folder, as on SIGTERM and Ctrl+C."""
    root = tmp_path / "srv"
    root.mkdir()
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    proc = _launch(root, tmp_path, "--port", "0", env=_env(tmp_path, TMPDIR=str(scratch)))
    try:
        out = _banner(proc)
        f = re.search(r"the token is in (\S+),", out)
        assert f, out
        url_file = Path(f.group(1))
        assert url_file.exists() and list(scratch.glob("mace_gui_cache_*"))
        t0 = time.time()
        os.kill(proc.pid, sig)
        proc.wait(timeout=5)
        assert time.time() - t0 < 5
        assert proc.returncode == 0
        assert not url_file.exists()
        assert not list(scratch.glob("mace_gui_cache_*"))
        assert "Traceback" not in proc.stderr.read()
    finally:
        _stop(proc)



@pytest.mark.skipif(not hasattr(os, "openpty"), reason="needs a pseudo-terminal")
def test_closing_its_terminal_cleans_up(tmp_path):
    """The real hangup: the server runs on a terminal that closes right
    after the banner (the kernel sends SIGHUP). It exits 0, at once, and
    leaves no temporary cache folder, however soon the terminal goes."""
    import pty
    root = tmp_path / "srv"
    root.mkdir()
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    env = _env(tmp_path, TMPDIR=str(scratch))
    argv = [sys.executable, str(MACE_CLI), "--no-banner", "gui", "--dir", str(root), "--port", "0"]
    for _ in range(3):
        pid, fd = pty.fork()
        if pid == 0:                                  # pragma: no cover (the child)
            os.chdir(str(tmp_path))
            os.execve(argv[0], argv, env)
        buf, end = b"", time.time() + 60
        while b"Press Ctrl+C" not in buf and time.time() < end:
            try:
                buf += os.read(fd, 4096)
            except OSError:
                break
        os.close(fd)
        end = time.time() + 10
        done = 0
        while time.time() < end and not done:
            done, status = os.waitpid(pid, os.WNOHANG)
            time.sleep(0.05)
        if not done:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
            pytest.fail("the server outlived its terminal:\n" + buf.decode(errors="replace"))
        assert os.waitstatus_to_exitcode(status) == 0, (os.waitstatus_to_exitcode(status), buf.decode(errors="replace"))
        assert not list(scratch.glob("mace_gui_cache_*"))

def test_a_hangup_under_nohup_is_still_ignored(tmp_path):
    """nohup sets SIGHUP to ignored before exec: the server keeps running."""
    root = tmp_path / "srv"
    root.mkdir()
    proc = subprocess.Popen(["nohup", sys.executable, str(MACE_CLI), "--no-banner", "gui", "--dir", str(root),
                             "--port", "0"],
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            env=_env(tmp_path), cwd=str(tmp_path), start_new_session=True)
    try:
        assert "MACE GUI serving" in _banner(proc)
        os.kill(proc.pid, signal.SIGHUP)
        time.sleep(1.0)
        assert proc.poll() is None
    finally:
        _stop(proc)


# --- start-up rules (each its own short-lived server) ---------------------------------

def _start_and_read(root, tmp, *args, env=None):
    proc = _launch(root, tmp, *args, env=env)
    try:
        out = _banner(proc)
    finally:
        _stop(proc)
    err = proc.stderr.read() if not proc.stderr.closed else ""
    return proc, out, err


def _start_capture(tmp, *args, env=None):
    root = tmp / "srv"
    root.mkdir(exist_ok=True)
    proc = _launch(root, tmp, *args, env=env)
    try:
        out = _banner(proc)
        time.sleep(0.2)
    finally:
        if proc.poll() is None:
            proc.terminate()
        out_rest, err = proc.communicate(timeout=15)
    return proc, out + out_rest, err


def _token_of(out):
    m = re.search(r"Token \(for the sign-in form\): (\S+)", out)
    return m.group(1) if m else None


def test_non_loopback_host_needs_insecure_network(tmp_path):
    proc = _launch(tmp_path, tmp_path, "--port", "0", "--host", "0.0.0.0")
    out, err = proc.communicate(timeout=30)
    assert proc.returncode == 2
    assert "--insecure-network" in err and "unencrypted" in err
    assert "http://" not in out


def test_test_hooks_are_refused_on_a_non_loopback_host(tmp_path):
    env = _env(tmp_path, MACE_GUI_TEST_HOOKS="1", MACE_GUI_PRINT_TOKEN="1")
    proc, out, err = _start_capture(tmp_path, "--port", "0", "--host", "0.0.0.0", "--insecure-network", env=env)
    assert "MACE_GUI_TEST_HOOKS is ignored" in err, err
    assert "test hooks enabled" not in err


def test_test_hooks_say_so_on_stderr(tmp_path):
    env = _env(tmp_path, MACE_GUI_TEST_HOOKS="1", MACE_GUI_PRINT_TOKEN="1")
    proc, out, err = _start_capture(tmp_path, "--port", "0", env=env)
    assert "mace gui: test hooks enabled (MACE_GUI_TEST_HOOKS=1)" in err, err


def test_random_token_differs_per_run(tmp_path):
    env = _env(tmp_path, MACE_GUI_PRINT_TOKEN="1")
    a = _token_of(_start_capture(tmp_path, "--port", "0", env=env)[1])
    b = _token_of(_start_capture(tmp_path, "--port", "0", env=env)[1])
    assert a and b and a != b and len(a) >= 32


def test_token_flag_env_and_precedence(tmp_path):
    env = _env(tmp_path, MACE_GUI_PRINT_TOKEN="1")
    proc, out, err = _start_capture(tmp_path, "--port", "0", "--token", "flag-token-1234567", env=env)
    assert _token_of(out) == "flag-token-1234567"
    assert "can see --token in ps" in err and "MACE_GUI_TOKEN" in err
    env2 = _env(tmp_path, MACE_GUI_PRINT_TOKEN="1", MACE_GUI_TOKEN="env-token-4567890")
    proc, out, err = _start_capture(tmp_path, "--port", "0", env=env2)
    assert _token_of(out) == "env-token-4567890"
    assert "in ps" not in err
    proc, out, err = _start_capture(tmp_path, "--port", "0", "--token", "flag-wins-78901234", env=env2)
    assert _token_of(out) == "flag-wins-78901234"


@pytest.mark.parametrize("bad", ["a", "short-token-15ch", "my secret phrase!!", "a;b-0123456789abcdef",
                                 'p@ss"w,rd-0123456789', "\u00fcn\u00ef-0123456789abcdef"])
@pytest.mark.parametrize("via", ["flag", "env"])
def test_a_chosen_token_must_be_long_and_url_safe(tmp_path, bad, via):
    """A chosen token goes into the URL and the cookie as it is, and the
    session path is derived from it: 16+ URL-safe characters or exit 2."""
    if bad == "short-token-15ch":
        bad = bad[:15]
    env = _env(tmp_path, MACE_GUI_PRINT_TOKEN="1", **({"MACE_GUI_TOKEN": bad} if via == "env" else {}))
    args = ["--port", "0"] + (["--token", bad] if via == "flag" else [])
    proc = _launch(tmp_path, tmp_path, *args, env=env)
    out, err = proc.communicate(timeout=30)
    assert proc.returncode == 2, (out, err)
    assert ("--token" if via == "flag" else "MACE_GUI_TOKEN") in err and "at least 16 characters" in err
    assert "secrets.token_urlsafe" in err
    assert "http://" not in out


def test_a_16_character_url_safe_token_works(tmp_path):
    token = "Ab0._~-xyzXYZ789"
    env = _env(tmp_path, MACE_GUI_PRINT_TOKEN="1", MACE_GUI_TOKEN=token)
    proc = _launch(tmp_path, tmp_path, "--port", "0", env=env)
    try:
        out = _banner(proc)
        assert _token_of(out) == token, out
        base = re.search(r"(http://127\.0\.0\.1:(\d+)/s/[A-Za-z0-9_-]+/)\?login=(\S+)", out)
        st, headers, _ = _request(base.group(1), f"?login={base.group(3)}")
        assert st == 303
        cookie = headers["set-cookie"].split(";", 1)[0]
        assert _request(base.group(1), "api/info", headers={"Cookie": cookie})[0] == 200
        assert _request(base.group(1), "api/info", headers={"X-Mace-Token": token})[0] == 200
    finally:
        _stop(proc)


def test_the_tunnel_hint_names_no_site_without_a_gateway(tmp_path):
    """Without MACE_GUI_GATEWAY or --gateway the hint says LOGIN_HOST (paste-safe) and how to fill it in."""
    env = _env(tmp_path, MACE_GUI_PRINT_TOKEN="1", SSH_CONNECTION="10.0.0.1 5555 10.0.0.2 22")
    env.pop("MACE_GUI_GATEWAY", None)
    proc, out, err = _start_capture(tmp_path, "--port", "0", env=env)
    assert re.search(r"ssh -N -J \S+@LOGIN_HOST -L ", out), out
    assert "MACE_GUI_GATEWAY" in out and "msu.edu" not in out
    env["MACE_GUI_GATEWAY"] = "login.example.org"
    proc, out, err = _start_capture(tmp_path, "--port", "0", env=env)
    assert re.search(r"ssh -N -J \S+@login\.example\.org -L ", out), out
    assert "LOGIN_HOST" not in out and "<gateway>" not in out


def test_fixed_token_keeps_the_same_session_path(tmp_path):
    env = _env(tmp_path, MACE_GUI_PRINT_TOKEN="1", MACE_GUI_TOKEN="same-token-for-both-runs")
    a = re.search(r"/s/([A-Za-z0-9_-]+)/", _start_capture(tmp_path, "--port", "0", env=env)[1]).group(1)
    b = re.search(r"/s/([A-Za-z0-9_-]+)/", _start_capture(tmp_path, "--port", "0", env=env)[1]).group(1)
    assert a == b and "same-token" not in a


def test_mace_gui_port_is_the_default_port(tmp_path):
    with socket.socket() as so:
        so.bind(("127.0.0.1", 0))
        port = so.getsockname()[1]
    env = _env(tmp_path, MACE_GUI_PRINT_TOKEN="1", MACE_GUI_PORT=str(port))
    proc, out, err = _start_capture(tmp_path, env=env)
    assert f"http://127.0.0.1:{port}/s/" in out, (out, err)
    # the port in use: the error recommends a personal port
    with socket.socket() as so:
        so.bind(("127.0.0.1", 0))
        so.listen(1)
        busy = so.getsockname()[1]
        p = _launch(tmp_path, tmp_path, "--port", str(busy), env=_env(tmp_path))
        out, err = p.communicate(timeout=30)
    assert p.returncode == 2 and "MACE_GUI_PORT" in err and "--port 0" in err


def test_help_has_no_one_time_or_stdlib_only_claims():
    out = subprocess.run([sys.executable, str(MACE_CLI), "gui", "--help"], capture_output=True, text=True,
                         env={k: v for k, v in os.environ.items() if k not in ("MACE_HOME", "PYTHONPATH")},
                         timeout=60).stdout
    assert "--token" in out and "MACE_GUI_TOKEN" in out and "--insecure-network" in out
    # the one-time sign-in links are real now; the server is not standard-library only
    assert "one-time sign-in link" in out and "standard library only" not in out


def test_a_checkout_without_the_gui_exits_1_with_a_message(tmp_path):
    """A mace_cli whose mace package has no gui/ (a stale MACE_HOME, a partial
    copy): exit 1 and a message, not a traceback."""
    tree = tmp_path / "old"
    tree.mkdir()
    shutil.copy2(MACE_CLI, tree / "mace_cli")
    for p in REPO_ROOT.iterdir():
        if p.name not in ("mace_cli", "mace", "test", ".git") and not p.name.startswith("."):
            (tree / p.name).symlink_to(p)
    (tree / "mace").mkdir()
    for p in (REPO_ROOT / "mace").iterdir():
        if p.name != "gui":
            (tree / "mace" / p.name).symlink_to(p)
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "COLORTERM", "TERM")}
    env.update(MACE_HOME=str(tree), PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, str(tree / "mace_cli"), "--no-banner", "gui", "--dir", str(tmp_path)],
                       capture_output=True, text=True, env=env, cwd=str(tmp_path), timeout=60)
    text = r.stdout + r.stderr
    assert r.returncode == 1, text
    assert "could not import the GUI" in text and "MACE_HOME" in text
    assert "Traceback" not in text
