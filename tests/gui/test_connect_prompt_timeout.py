"""mace gui connect through a jump host while ssh asks for a passphrase (1.1.6):
"Connection to UNKNOWN port 65535 timed out" while the gateway's passphrase
was still being typed.

With -J, ssh reaches the node over ``ssh -W`` to the gateway (an implicit
ProxyCommand that gets none of the command's -o options), and the node's
ConnectTimeout also covers the wait for the node's banner, which includes the
gateway's login. The fix: 120 s instead of 20 when a jump host may prompt, a
TCP probe that keeps an unreachable gateway a fast failure, a message that
names the node behind the gateway, and an ssh-add hint when no agent holds
the key. connect.py, connect_check.py and both scripts alike."""
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from mace.gui import connect as C
from mace.gui import connect_check as K

REPO = Path(__file__).resolve().parents[2]
SH = REPO / "scripts" / "mace-gui-connect.sh"
PS1 = REPO / "scripts" / "mace-gui-connect.ps1"
PROF = {"gateway": "me@gw.example.edu", "node": "dev-node-01", "dir": "~/runs"}
PWSH = shutil.which("pwsh")
SSH = shutil.which("ssh")
UNKNOWN = ["Connection timed out during banner exchange", "Connection to UNKNOWN port 65535 timed out"]


class _G:
    """subprocess.run for ``ssh -G``: per-host settings ({host: text}), "" for the others."""

    def __init__(self, hosts=None):
        self.hosts = hosts or {}
        self.calls = []

    def __call__(self, argv, **kw):
        self.calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, self.hosts.get(argv[-1], ""), "")


# ---- the cause, with the real ssh: no sshd needed

@pytest.mark.skipif(SSH is None, reason="needs ssh")
def test_real_ssh_times_out_the_banner_while_the_proxy_still_logs_in():
    """A ProxyCommand that has not connected yet (the jump host's ssh waiting
    for a passphrase) ends at the TARGET's ConnectTimeout, with the exact lines
    Marcus saw: a proxied connection has no address (UNKNOWN, port 65535)."""
    t0 = time.monotonic()
    r = subprocess.run([SSH, "-F", "/dev/null", "-o", "ConnectTimeout=2", "-o", "ProxyCommand=sleep 8",
                        "target.invalid", "true"], capture_output=True, text=True, timeout=30,
                       stdin=subprocess.DEVNULL)
    assert r.returncode == 255 and time.monotonic() - t0 < 7
    assert all(ln in r.stderr for ln in UNKNOWN), r.stderr


@pytest.mark.skipif(SSH is None, reason="needs ssh")
def test_real_ssh_gives_the_jump_hosts_ssh_none_of_the_command_options():
    """-J is an implicit ProxyCommand ``ssh -W [%h]:%p <jump>``: our -o
    ConnectTimeout does not reach the gateway's own connection."""
    r = subprocess.run([SSH, "-v", "-F", "/dev/null", "-o", "ConnectTimeout=3", "-J", "gw.invalid",
                        "target.invalid", "true"], capture_output=True, text=True, timeout=60,
                       stdin=subprocess.DEVNULL)
    line = next(ln for ln in r.stderr.splitlines() if "implicit ProxyCommand" in ln)
    assert "-W" in line and "gw.invalid" in line and "ConnectTimeout" not in line


# ---- the timeout: 120 s when a jump host may prompt

def test_jump_hop_gets_the_prompt_timeout():
    chain = C.jump_chain(PROF, "jump", runner=_G())
    assert chain == ["me@gw.example.edu"]
    assert C.prompt_wait(chain, _G()) is True
    argv = C.build_ssh_argv(PROF, 8765, 40000, 40001, "jump", prompt=True)
    assert argv[:3] == ["ssh", "-J", "me@gw.example.edu"]
    assert "ConnectTimeout=120" in argv and "ConnectTimeout=20" not in argv


def test_batchmode_jump_hosts_keep_the_short_timeout():
    g = _G({"me@gw.example.edu": "batchmode yes\n", "bastion": "batchmode yes\n"})
    assert C.prompt_wait(["bastion", "me@gw.example.edu"], g) is False
    assert C.prompt_wait(["bastion", "me@gw.example.edu"], _G({"bastion": "batchmode yes\n"})) is True
    argv = C.build_ssh_argv(PROF, 8765, 40000, 40001, "jump", prompt=False)
    assert "ConnectTimeout=20" in argv


def test_nested_hop_without_jumps_connects_directly():
    assert C.jump_chain(PROF, "nested", runner=_G()) == []
    assert C.prompt_wait([], _G()) is False
    argv = C.build_ssh_argv(PROF, 8765, 40000, 40001, "nested", prompt=False)
    assert argv.count("ConnectTimeout=20") == 1
    assert "ConnectTimeout=20" in argv[-1]                     # the inner hop keeps its own (BatchMode) value


def test_bastions_and_a_config_proxyjump_count_too():
    prof = {**PROF, "jump": "bastion:2222, b2"}
    assert C.jump_chain(prof, "nested", runner=_G()) == ["bastion:2222", "b2"]
    assert C.jump_chain(prof, "jump", runner=_G()) == ["bastion:2222", "b2", "me@gw.example.edu"]
    g = _G({"dev-node-01": "proxyjump hop1,hop2\n"})
    assert C.jump_chain({"node": "dev-node-01"}, "jump", runner=g) == ["hop1", "hop2"]
    g = _G({"dev-node-01": "proxycommand ssh -W %h:%p hop1\n"})
    chain = C.jump_chain({"node": "dev-node-01"}, "jump", runner=g)
    assert chain == ["proxycommand"] and C.prompt_wait(chain, g) is True
    assert C._jump_spec("bastion:2222") == ("bastion", 2222) and C._jump_spec("me@b") == ("me@b", None)


def test_a_timeout_of_yours_still_wins():
    argv = C.build_ssh_argv({**PROF, "ssh_options": ["-oConnectTimeout=45"]}, 8765, 1, 2, "jump", prompt=True)
    assert "ConnectTimeout=120" not in argv and "-oConnectTimeout=45" in argv
    argv = C.build_ssh_argv(PROF, 8765, 1, 2, "jump", timeout="90", prompt=True)
    assert not any("ConnectTimeout" in a for a in argv)


# ---- an unreachable gateway still fails fast

def test_first_hop_probe_names_a_gateway_that_does_not_answer():
    seen = []

    def slow(addr, timeout):
        seen.append((addr, timeout))
        raise socket.timeout("timed out")
    g = _G({"me@gw.example.edu": "hostname login.example.edu\nport 2200\n"})
    assert C.first_hop_timed_out(["me@gw.example.edu"], g, slow) == "login.example.edu"
    assert seen == [(("login.example.edu", 2200), C.CONNECT_TIMEOUT_S)]
    seen.clear()
    g = _G({"bastion": "hostname b.example.edu\nconnecttimeout 7\n"})
    assert C.first_hop_timed_out(["bastion:2222", "gw"], g, slow) == "b.example.edu"
    assert seen == [(("b.example.edu", 2222), 7)]


@pytest.mark.parametrize("err", [ConnectionRefusedError, socket.gaierror, OSError])
def test_first_hop_probe_leaves_fast_errors_to_ssh(err):
    def fail(addr, timeout):
        raise err("no")
    assert C.first_hop_timed_out(["gw"], _G({"gw": "hostname gw\n"}), fail) is None


def test_first_hop_probe_skips_a_proxied_gateway():
    def never(addr, timeout):
        raise AssertionError("no probe through a proxy")
    assert C.first_hop_timed_out(["gw"], _G({"gw": "hostname gw\nproxyjump b\n"}), never) is None
    assert C.first_hop_timed_out(["proxycommand"], _G(), never) is None
    assert C.first_hop_timed_out([], _G(), never) is None


@pytest.fixture
def black_hole():
    """A loopback port whose accept queue is full: a connect to it hangs (its SYNs are dropped), as one to a
    gateway behind a VPN that is down does."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(0)
    port = srv.getsockname()[1]
    filler = socket.create_connection(("127.0.0.1", port), timeout=2)
    probe = socket.socket()
    probe.settimeout(1)
    try:
        probe.connect(("127.0.0.1", port))
        pytest.skip("this system accepts past a full backlog")
    except socket.timeout:
        pass
    finally:
        probe.close()
    yield port
    filler.close()
    srv.close()


def test_first_hop_probe_on_a_real_socket(black_hole):
    g = _G({"gw": f"hostname 127.0.0.1\nport {black_hole}\nconnecttimeout 1\n"})
    t0 = time.monotonic()
    assert C.first_hop_timed_out(["gw"], g) == "127.0.0.1"
    assert time.monotonic() - t0 < 10


def _stub_ssh(tmp_path, g_text, body="echo \"$@\" >> \"$FAKE_SSH_RAN\"; exit 255"):
    fb = tmp_path / "fakebin"
    fb.mkdir(exist_ok=True)
    (fb / "ssh").write_text("#!/bin/sh\n[ \"$1\" = -V ] && { echo 'OpenSSH_9.6p1' >&2; exit 0; }\n"
                            f"[ \"$1\" = -G ] && {{ printf '%s' {shlex.quote(g_text)}; exit 0; }}\n{body}\n")
    (fb / "ssh").chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON") and k not in ("MACE_HOME", "BROWSER")}
    env.update(PATH=f"{fb}:{env.get('PATH', '/usr/bin:/bin')}", HOME=str(tmp_path / "home"),
               XDG_CONFIG_HOME=str(tmp_path / "cfg"), FAKE_SSH_RAN=str(tmp_path / "ran"))
    (tmp_path / "home").mkdir(exist_ok=True)
    return env


def test_connect_and_bash_script_fail_fast_on_a_gateway_that_does_not_answer(tmp_path, black_hole):
    env = _stub_ssh(tmp_path, f"hostname 127.0.0.1\nport {black_hole}\nconnecttimeout 1\n")
    flags = ["--gateway", "me@gw", "--node", "n1", "--hop", "jump", "--no-open"]
    for cmd in ([sys.executable, str(REPO / "mace_cli"), "--no-banner", "gui", "connect", *flags],
                ["bash", str(SH), *flags]):
        t0 = time.monotonic()
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
        out = r.stdout + r.stderr
        assert r.returncode == 1, out
        assert "cannot reach host 127.0.0.1 (timeout): check the name, and your network or VPN." in out
        assert time.monotonic() - t0 < 30
        assert not (tmp_path / "ran").exists()                 # no ssh session was started


# ---- the message: first a node that is down, then a login that took too long; never "host UNKNOWN"

def test_unknown_timeout_names_the_node_behind_the_gateway():
    """Marcus's real case: dev-amd20-v100 was down (dev-amd20-h200 worked); a slow passphrase reads the same."""
    kind, msg = C.classify_failure(255, UNKNOWN, "dev-amd20-v100", "djokicma@hpcc.msu.edu", hop="jump",
                                   via=["djokicma@hpcc.msu.edu"], wait_s=120, add_key="~/.ssh/id_ed25519_hpcc")
    assert kind == "unreachable"
    assert msg.startswith("the node dev-amd20-v100 did not answer through the gateway djokicma@hpcc.msu.edu: it may "
                          "be down or not accept logins (try another node, or mace gui connect --check). Or the "
                          "login took too long: ssh waited 120 s")
    assert "UNKNOWN port 65535" in msg and "passphrase or approving 2FA took longer" in msg
    assert "ssh-add ~/.ssh/id_ed25519_hpcc" in msg and "--ssh-option=-oConnectTimeout=300" in msg
    assert "cannot reach host UNKNOWN" not in msg


def test_unknown_timeout_through_a_bastion():
    _, msg = C.classify_failure(255, UNKNOWN, "", "gw", hop="jump", via=["bastion"], wait_s=20)
    assert msg.startswith("gw did not answer through bastion: it may be down") and "ssh-add <your key>" in msg
    _, msg = C.classify_failure(255, UNKNOWN, "n1", "", via=["proxycommand"])
    assert msg.startswith("n1 did not answer through its proxy:")


def test_a_named_host_that_times_out_is_unchanged():
    kind, msg = C.classify_failure(255, ["ssh: connect to host gw.example.edu port 22: Connection timed out"],
                                   "n1", "gw.example.edu", hop="jump", via=["gw.example.edu"], wait_s=120)
    assert msg == C.unreachable_message("gw.example.edu", "timeout")


def _unknown_body():
    return "; ".join(f"echo '{ln}' >&2" for ln in UNKNOWN) + "; exit 255"


def test_bash_script_names_the_node_behind_the_gateway(tmp_path):
    env = _stub_ssh(tmp_path, "user me\n", _unknown_body())
    r = subprocess.run(["bash", str(SH), "--gateway", "me@gw.invalid", "--node", "n1", "--no-open"], env=env,
                       capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
    assert r.returncode == 1
    want = C.classify_failure(255, UNKNOWN, "n1", "me@gw.invalid", hop="jump", via=["me@gw.invalid"], wait_s=120)[1]
    want = want.replace("mace gui connect --check", f"{SH} --check")
    assert f"mace-gui-connect: {want}" in r.stderr, r.stderr
    assert "cannot reach host UNKNOWN" not in r.stderr


def test_mace_gui_connect_names_the_node_behind_the_gateway(tmp_path):
    env = _stub_ssh(tmp_path, "user me\n", _unknown_body())
    r = subprocess.run([sys.executable, str(REPO / "mace_cli"), "--no-banner", "gui", "connect", "--gateway",
                        "me@gw.invalid", "--node", "n1", "--hop", "jump", "--no-open"], env=env,
                       capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
    assert r.returncode == 1
    assert ("the node n1 did not answer through the gateway me@gw.invalid: it may be down or not accept logins"
            in r.stdout), r.stdout
    assert "ssh waited 120 s" in r.stdout
    assert ("(waiting for n1 through the gateway me@gw.invalid, up to 120 s; if n1 is down, press Ctrl+C and pick "
            "another node)") in r.stdout


@pytest.mark.parametrize("hop,want", [("jump", "ConnectTimeout=120"), ("nested", "ConnectTimeout=20")])
def test_dry_runs_agree_on_the_timeout(tmp_path, hop, want):
    env = _stub_ssh(tmp_path, "user me\n")
    flags = ["--gateway", "me@gw.example.edu", "--node", "n1", "--hop", hop, "--dry-run"]
    r = subprocess.run([sys.executable, str(REPO / "mace_cli"), "--no-banner", "gui", "connect", *flags], env=env,
                       capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
    py = shlex.split(next(ln for ln in r.stdout.splitlines() if ln.startswith("ssh ")))
    sh = [ln for ln in subprocess.run(["bash", str(SH), *flags], env=env, capture_output=True, text=True,
                                      timeout=60).stdout.splitlines() if not ln.startswith("# ")]
    assert want in py and sh == py
    env = _stub_ssh(tmp_path, "user me\nbatchmode yes\n")
    r = subprocess.run([sys.executable, str(REPO / "mace_cli"), "--no-banner", "gui", "connect", *flags], env=env,
                       capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
    assert "ConnectTimeout=20" in r.stdout and "ConnectTimeout=120" not in r.stdout


# ---- ssh-add: asked once instead of on each hop

@pytest.fixture
def keyhome(tmp_path):
    if shutil.which("ssh-keygen") is None:
        pytest.skip("needs ssh-keygen")
    home = tmp_path / "home"
    (home / ".ssh").mkdir(parents=True)
    for name, pw in (("id_locked", "a passphrase"), ("id_open", "")):
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", pw, "-f", str(home / ".ssh" / name)], check=True,
                       stdin=subprocess.DEVNULL, capture_output=True)
    return home


def _runner(g_text):
    def run(argv, **kw):
        if argv[:2] == ["ssh", "-G"]:
            return subprocess.CompletedProcess(argv, 0, g_text, "")
        return subprocess.run(argv, **kw)
    return run


def test_unloaded_key_finds_a_passphrase_key_no_agent_holds(keyhome):
    env = {k: v for k, v in os.environ.items() if k != "SSH_AUTH_SOCK"}
    got = K.unloaded_key(PROF, runner=_runner("identityfile ~/.ssh/id_locked\n"), home=keyhome, env=env)
    assert got == ("~/.ssh/id_locked", False)
    assert K.unloaded_key(PROF, runner=_runner("identityfile ~/.ssh/id_open\n"), home=keyhome, env=env) is None
    assert K.unloaded_key(PROF, runner=_runner("identityfile ~/.ssh/id_none\n"), home=keyhome, env=env) is None


def test_unloaded_key_is_none_when_the_agent_holds_it(keyhome, tmp_path):
    if shutil.which("ssh-agent") is None or shutil.which("ssh-add") is None:
        pytest.skip("needs ssh-agent")
    sock = tmp_path / "a.sock"
    agent = subprocess.Popen(["ssh-agent", "-D", "-a", str(sock)], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):
            if sock.exists():
                break
            time.sleep(0.1)
        env = dict(os.environ, SSH_AUTH_SOCK=str(sock))
        g = "identityfile ~/.ssh/id_locked\n"
        assert K.unloaded_key(PROF, runner=_runner(g), home=keyhome, env=env) == ("~/.ssh/id_locked", True)
        # load it (the private half without its passphrase stands in for typing it once)
        plain = tmp_path / "plain"
        shutil.copy(keyhome / ".ssh" / "id_locked", plain)
        plain.chmod(0o600)
        subprocess.run(["ssh-keygen", "-q", "-p", "-P", "a passphrase", "-N", "", "-f", str(plain)], check=True,
                       stdin=subprocess.DEVNULL, capture_output=True)
        subprocess.run(["ssh-add", "-q", str(plain)], env=env, check=True, stdin=subprocess.DEVNULL,
                       capture_output=True)
        assert K.unloaded_key(PROF, runner=_runner(g), home=keyhome, env=env) is None
    finally:
        agent.terminate()
        agent.wait(10)


def test_passphrase_hint_text():
    hint = K.passphrase_hint("~/.ssh/k", True, twice=True, windows=False)
    assert hint == ("~/.ssh/k has a passphrase and no ssh agent holds it, so ssh asks for it for the gateway and "
                    "again for the node. To type it once per session, first run: ssh-add ~/.ssh/k")
    assert K.passphrase_hint("~/.ssh/k", False, windows=False).endswith('eval "$(ssh-agent)" && ssh-add ~/.ssh/k')
    assert K.passphrase_hint("~/.ssh/k", False, windows=True).endswith("Start-Service ssh-agent, then ssh-add ~/.ssh/k")
    assert "on each connection" in K.passphrase_hint("~/.ssh/k", True)
    assert "ssh-add ~/.ssh/k before connecting makes ssh ask only once" in C.waiting_note("n1", "~/.ssh/k")
    # a key path with a space (a Windows user folder) or a quote: one argument of the shell shown
    win = r"C:\Users\Jane Doe\.ssh\id_ed25519"
    assert K.passphrase_hint(win, True, windows=True).endswith(f"ssh-add '{win}'")
    assert K.passphrase_hint(win, False, windows=True).endswith(f"Start-Service ssh-agent, then ssh-add '{win}'")
    assert K.passphrase_hint("~/my keys/id", False, windows=False).endswith("""eval "$(ssh-agent)" && ssh-add ~/'my keys/id'""")
    assert K.shell_path("/home/o'brien/k", False) == "'/home/o'\"'\"'brien/k'" and K.shell_path("C:/Users/o'brien/k", True) == "'C:/Users/o''brien/k'"
    assert K.shell_path(r"C:\Users\jd\.ssh\id", True) == r"C:\Users\jd\.ssh\id"         # nothing to quote


def test_connect_and_bash_script_suggest_ssh_add(tmp_path, keyhome):
    """The real invocation paths, before ssh starts: the key from ssh -G has a passphrase and no agent answers."""
    env = _stub_ssh(tmp_path, "user me\nidentityfile ~/.ssh/id_locked\n", "exit 255")
    env.pop("SSH_AUTH_SOCK", None)
    want = 'so ssh asks for it for the gateway and again for the node. To type it once per session, first run: ' \
           'eval "$(ssh-agent)" && ssh-add ~/.ssh/id_locked'
    for cmd in ([sys.executable, str(REPO / "mace_cli"), "--no-banner", "gui", "connect"],
                ["bash", str(SH)]):
        r = subprocess.run(cmd + ["--gateway", "me@gw.invalid", "--node", "n1", "--hop", "jump", "--no-open"],
                           env=env, capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
        assert "~/.ssh/id_locked has a passphrase and no ssh agent holds it, " + want in r.stdout, r.stdout


# ---- PowerShell

def test_powershell_has_the_same_rules():
    ps = PS1.read_text()
    for word in ('"ConnectTimeout=$wait"', '"120"', "First-Hop-TimedOut", "Unloaded-Key", "ConnectAsync",
                 "for the gateway and again for the node", "-SshOption '-oConnectTimeout=300'",
                 '"batchmode"', '"proxyjump"'):
        assert word in ps, word


def _ps_env(tmp_path, g_text, body):
    env = _stub_ssh(tmp_path, g_text, body)
    (Path(env["PATH"].split(":")[0]) / "ssh.exe").write_text("#!/bin/sh\nexit 0\n")
    (Path(env["PATH"].split(":")[0]) / "ssh.exe").chmod(0o755)
    env["APPDATA"] = str(tmp_path / "appdata")
    env.pop("USERNAME", None)
    return env


@pytest.mark.skipif(PWSH is None, reason="needs pwsh (PowerShell 7) to run the .ps1")
def test_powershell_dry_run_and_unknown_warning(tmp_path):
    env = _ps_env(tmp_path, "user me\n", _unknown_body())
    r = subprocess.run([PWSH, "-NoProfile", "-NonInteractive", "-File", str(PS1), "-Gateway", "me@gw",
                        "-Node", "n1", "-DryRun"], env=env, capture_output=True, timeout=120, stdin=subprocess.DEVNULL)
    assert b"ConnectTimeout=120" in r.stdout, r.stdout
    r = subprocess.run([PWSH, "-NoProfile", "-NonInteractive", "-File", str(PS1), "-Gateway", "me@gw",
                        "-Node", "n1", "-NoOpen", "-LocalPort", "18766"], env=env, capture_output=True, timeout=120,
                       stdin=subprocess.DEVNULL)
    out = re.sub(r"\s*\n\s*", " ", (r.stdout + r.stderr).decode().replace("\r\n", "\n"))
    assert r.returncode == 1
    assert ("the node n1 did not answer through the gateway me@gw: it may be down or not accept logins (try "
            "another node, or -Check). Or the login took too long: ssh waited 120 s") in out, out
    assert "(waiting for n1 through the gateway me@gw, up to 120 s; if n1 is down, press Ctrl+C" in out


@pytest.mark.skipif(PWSH is None, reason="needs pwsh (PowerShell 7) to run the .ps1")
def test_powershell_fails_fast_on_a_gateway_that_does_not_answer(tmp_path, black_hole):
    env = _ps_env(tmp_path, f"hostname 127.0.0.1\nport {black_hole}\nconnecttimeout 1\n",
                  "echo \"$@\" >> \"$FAKE_SSH_RAN\"; exit 255")
    r = subprocess.run([PWSH, "-NoProfile", "-NonInteractive", "-File", str(PS1), "-Gateway", "me@gw",
                        "-Node", "n1", "-NoOpen", "-LocalPort", "18767"], env=env, capture_output=True, timeout=120,
                       stdin=subprocess.DEVNULL)
    out = re.sub(r"\s*\n\s*", " ", (r.stdout + r.stderr).decode())
    assert r.returncode == 1 and "cannot reach host 127.0.0.1 (timeout)" in out, out
    assert not (tmp_path / "ran").exists()


@pytest.mark.skipif(PWSH is None, reason="needs pwsh (PowerShell 7) to run the .ps1")
def test_powershell_key_arg_quotes_like_shell_path(tmp_path):
    keys = ["~/.ssh/id", "~/my keys/id", r"C:\Users\Jane Doe\.ssh\id", "/o'neil/k", r"C:\Users\jd\.ssh\id"]
    for windows in (False, True):
        probe = tmp_path / f"key_arg_{windows}.ps1"
        probe.write_text("$OnWindows = $" + str(windows).lower() + "; $src = Get-Content -Raw $args[0]\n"
                         "foreach ($x in [regex]::Matches($src, '(?ms)^function (ShQuote|Key-Arg)\\b.*?^\\}')) "
                         "{ Invoke-Expression $x.Value }\nforeach ($k in $args[1..($args.Count - 1)]) { Key-Arg $k }\n")
        r = subprocess.run([PWSH, "-NoProfile", "-NonInteractive", "-File", str(probe), str(PS1), *keys],
                           capture_output=True, timeout=120, stdin=subprocess.DEVNULL, text=True)
        assert r.stdout.splitlines() == [K.shell_path(k, windows) for k in keys], r.stdout + r.stderr


# ---- a node that is down fails in seconds: node_probe, a second login to the gateway that cannot prompt

def test_node_probe_command():
    g = _G({"me@dev-amd20-v100": "hostname dev-amd20-v100.example.edu\nport 2222\n",
            "me@gw.example.edu": "connecttimeout none\n"})
    argv, script = C.node_probe({**PROF, "node": "dev-amd20-v100", "jump": "b1"}, "me", g)
    assert argv == ["ssh", "-J", "b1", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "me@gw.example.edu",
                    "bash -s"]
    assert script.startswith(f"timeout {C.NODE_PROBE_S} bash -c 'exec 3<>/dev/tcp/\"$0\"/\"$1\" && head -c 4 <&3' "
                             "dev-amd20-v100.example.edu 2222\n")
    assert script.endswith('echo "MACE_NODE_PROBE $c"\n')
    # your ssh config's ConnectTimeout for the gateway is kept
    argv, _ = C.node_probe(PROF, None, _G({"me@gw.example.edu": "connecttimeout 90\n"}))
    assert not any("ConnectTimeout" in a for a in argv)


@pytest.mark.parametrize("out,rc,down", [
    ("\nMACE_NODE_PROBE 124\n", 0, True),         # nothing from the node's ssh in time
    ("SSH-\nMACE_NODE_PROBE 0\n", 0, False),      # its banner came
    ("bash: connect: Connection refused\n\nMACE_NODE_PROBE 1\n", 0, False),   # ssh -J says that at once itself
    ("\nMACE_NODE_PROBE 127\n", 0, False),        # no timeout or bash on the gateway
    ("", 255, False),                               # no login without a prompt (BatchMode)
])
def test_node_down_reads_only_a_timeout(out, rc, down):
    seen = {}

    def run(argv, **kw):
        seen.update(kw)
        return subprocess.CompletedProcess(argv, rc, out, "")
    assert C.node_down(["ssh", "gw", "bash -s"], "script", run) is down
    assert seen["input"] == "script" and seen["timeout"] > C.NODE_PROBE_S


def _probe_stub(tmp_path, node_port, closed_port):
    """A gateway stub: ssh -G names the node's port, the probe's "bash -s" runs here as on a gateway, and the
    session waits for a banner that never comes (the node behind the gateway is down)."""
    fb = tmp_path / "fakebin"
    fb.mkdir(exist_ok=True)
    (fb / "ssh").write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = -G ]; then case \"$2\" in\n"
        f"  *gw*) printf 'hostname 127.0.0.1\\nport {closed_port}\\n' ;;\n"
        f"  *) printf 'hostname 127.0.0.1\\nport {node_port}\\n' ;;\n"
        "esac; exit 0; fi\n"
        "for a in \"$@\"; do last=$a; done\n"
        "if [ \"$last\" = 'bash -s' ]; then echo probe >> \"$FAKE_SSH_RAN\"; exec bash -s; fi\n"
        "echo session >> \"$FAKE_SSH_RAN\"\n"
        "exec sleep 300\n")
    (fb / "ssh").chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON") and k not in ("MACE_HOME", "BROWSER")}
    env.update(PATH=f"{fb}:/usr/bin:/bin", HOME=str(tmp_path / "home"), XDG_CONFIG_HOME=str(tmp_path / "cfg"),
               FAKE_SSH_RAN=str(tmp_path / "ran"))
    (tmp_path / "home").mkdir(exist_ok=True)
    return env


@pytest.fixture
def closed_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.mark.skipif(shutil.which("timeout") is None or os.name == "nt", reason="needs bash and timeout")
@pytest.mark.parametrize("tool", ["connect", "bash"])
def test_a_node_that_does_not_answer_fails_in_seconds(tmp_path, black_hole, closed_port, tool):
    env = _stub_ssh_env = _probe_stub(tmp_path, black_hole, closed_port)
    flags = ["--gateway", "me@gw", "--node", "n1", "--no-open"]
    cmd = ([sys.executable, str(REPO / "mace_cli"), "--no-banner", "gui", "connect", "--hop", "jump", *flags]
           if tool == "connect" else ["bash", str(SH), *flags])
    t0 = time.monotonic()
    r = subprocess.run(cmd, env=_stub_ssh_env, capture_output=True, text=True, timeout=90, stdin=subprocess.DEVNULL)
    took = time.monotonic() - t0
    out = r.stdout + r.stderr
    assert r.returncode == 1, out
    assert ("the node n1 did not answer through the gateway me@gw: it may be down or not accept logins (its ssh "
            f"sent nothing within {C.NODE_PROBE_S} s when the gateway tried it). Try another node, or ") in out, out
    assert "(waiting for n1 through the gateway me@gw, up to 120 s; if n1 is down, press Ctrl+C" in out
    assert took < 40, took                                       # not the 120 s ssh would wait
    assert sorted((tmp_path / "ran").read_text().split()) == ["probe", "session"]


# ---- the real thing: OpenSSH's sshd as an unprivileged gateway on loopback, nodes that are down or hang

@pytest.fixture
def lab(tmp_path, black_hole):
    sshd = shutil.which("sshd") or ("/usr/sbin/sshd" if os.path.exists("/usr/sbin/sshd") else None)
    if os.name == "nt" or not sshd or not shutil.which("ssh-keygen") or not shutil.which("timeout"):
        pytest.skip("needs OpenSSH's sshd, ssh-keygen and timeout")
    d = tmp_path / "lab"
    d.mkdir()
    for k in ("hostkey", "userkey"):
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(d / k)], check=True,
                       stdin=subprocess.DEVNULL, capture_output=True)
    (d / "authorized_keys").write_text((d / "userkey.pub").read_text())
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    (d / "sshd_config").write_text(
        f"Port {port}\nListenAddress 127.0.0.1\nHostKey {d / 'hostkey'}\nPidFile {d / 'sshd.pid'}\n"
        f"AuthorizedKeysFile {d / 'authorized_keys'}\nStrictModes no\nPasswordAuthentication no\n"
        "KbdInteractiveAuthentication no\nUsePAM no\nAllowTcpForwarding yes\n")
    hung = socket.socket()                     # accepts, never sends a banner (a node whose sshd hangs)
    hung.bind(("127.0.0.1", 0))
    hung.listen(8)
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or "me"
    blocks = []
    for name, p in (("labgw", port), ("labdown", black_hole), ("labhung", hung.getsockname()[1]), ("labup", port)):
        blocks.append(f"Host {name}\n  HostName 127.0.0.1\n  Port {p}\n  User {user}\n  IdentityFile {d / 'userkey'}\n"
                      f"  IdentitiesOnly yes\n  UserKnownHostsFile {d / 'known_hosts'}\n  StrictHostKeyChecking no\n"
                      "  LogLevel ERROR\n")
    (d / "client_config").write_text("".join(blocks))
    fb = d / "bin"
    fb.mkdir()
    (fb / "ssh").write_text(f"#!/bin/sh\nexec {shutil.which('ssh')} -F {d / 'client_config'} \"$@\"\n")
    (fb / "ssh").chmod(0o755)
    p = subprocess.Popen([sshd, "-D", "-f", str(d / "sshd_config"), "-E", str(d / "sshd.log")],
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON") and k not in ("MACE_HOME", "BROWSER",
                                                                                          "SSH_AUTH_SOCK")}
    env.update(PATH=f"{fb}:{os.environ.get('PATH', '/usr/bin:/bin')}", HOME=str(d), XDG_CONFIG_HOME=str(d / "cfg"))
    try:
        for _ in range(50):
            ok = subprocess.run(["ssh", "-o", "BatchMode=yes", "labgw", "true"], env=env, capture_output=True,
                                stdin=subprocess.DEVNULL, timeout=20).returncode == 0
            if ok:
                break
            time.sleep(0.2)
        else:
            pytest.skip("an unprivileged sshd does not start here: " + (d / "sshd.log").read_text()[-300:])
        yield env
    finally:
        p.terminate()
        p.wait(10)
        hung.close()


def _helpers_left(lab_env):
    r = subprocess.run(["pgrep", "-f", f"{lab_env['HOME']}/client_config .*-W"], capture_output=True, text=True)
    return r.stdout.split()


@pytest.mark.parametrize("tool", ["connect", "bash"])
@pytest.mark.parametrize("node", ["labdown", "labhung"])
def test_real_sshd_gateway_with_a_node_that_is_down(lab, tool, node):
    """Through a real OpenSSH gateway: plain -J would wait 120 s for the node; the probe ends it in seconds,
    and the session's -J helper (ssh -W, which our ConnectTimeout never reaches) does not stay behind."""
    flags = ["--gateway", "labgw", "--node", node, "--no-open"]
    cmd = ([sys.executable, str(REPO / "mace_cli"), "--no-banner", "gui", "connect", "--hop", "jump", *flags]
           if tool == "connect" else ["bash", str(SH), *flags])
    t0 = time.monotonic()
    r = subprocess.run(cmd, env=lab, capture_output=True, text=True, timeout=200, stdin=subprocess.DEVNULL)
    out = r.stdout + r.stderr
    assert r.returncode == 1 and f"the node {node} did not answer through the gateway labgw: it may be down" in out, out
    assert time.monotonic() - t0 < 40
    time.sleep(0.5)
    assert not _helpers_left(lab)


@pytest.mark.parametrize("tool", ["connect", "bash"])
def test_real_sshd_gateway_with_a_node_that_is_up(lab, tool):
    flags = ["--gateway", "labgw", "--node", "labup", "--no-open", "--mace", "/nonexistent/mace_cli"]
    cmd = ([sys.executable, str(REPO / "mace_cli"), "--no-banner", "gui", "connect", "--hop", "jump", *flags]
           if tool == "connect" else ["bash", str(SH), *flags])
    r = subprocess.run(cmd, env=lab, capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL)
    out = r.stdout + r.stderr
    assert r.returncode == 127 and "did not find" in out and "did not answer" not in out, out


def test_explicit_proxy_timeout_does_not_bound_the_gateways_connect(lab, black_hole):
    """Why the probe, and not ProxyCommand=ssh -o ConnectTimeout=N -W: that timeout covers only the login to
    the gateway; the gateway's own connect to a node that does not answer runs on (the kernel's ~2 min)."""
    p = subprocess.Popen(["ssh", "-o", "ConnectTimeout=2", "-W", f"127.0.0.1:{black_hole}", "labgw"], env=lab,
                         stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            p.wait(8)                                           # still waiting, long past its ConnectTimeout
    finally:
        p.kill()
        p.wait(5)


def test_check_names_a_silent_node():
    ck = K.Checker({"gateway": "gw", "node": "n1"}, no_net=True, out=lambda _t: None)
    level, text, fix = ck.classify(subprocess.CompletedProcess([], 255, "", "\n".join(UNKNOWN)), "n1", "jump")
    assert level == "FAIL" and text.startswith("n1 did not answer through the gateway gw in time") and "down" in text
    assert fix == "pick another node (--node), then check again"


@pytest.mark.parametrize("tool", ["connect", "bash"])
def test_real_sshd_check_names_the_node_that_is_down(lab, tool):
    flags = ["--gateway", "labgw", "--node", "labdown", "--hop", "jump", "--check"]
    cmd = ([sys.executable, str(REPO / "mace_cli"), "--no-banner", "gui", "connect", *flags]
           if tool == "connect" else ["bash", str(SH), *flags])
    r = subprocess.run(cmd, env=lab, capture_output=True, text=True, timeout=200, stdin=subprocess.DEVNULL)
    assert ("FAIL  labdown did not answer through the gateway labgw in time (ssh calls it UNKNOWN port 65535): it "
            "may be down or not accept logins") in r.stdout, r.stdout
    assert "pick another node (--node), then check again" in r.stdout


def test_a_banner_timeout_without_its_host_line_is_still_the_node():
    """With LogLevel ERROR ssh prints only "Connection timed out during banner exchange"."""
    _, msg = C.classify_failure(255, UNKNOWN[:1], "n1", "gw", hop="jump", via=["gw"], wait_s=120)
    assert msg.startswith("the node n1 did not answer through the gateway gw: it may be down")
    _, msg = C.classify_failure(255, UNKNOWN[:1], "", "gw", via=[])          # a direct connection: as before
    assert msg == C.unreachable_message("the host", "timeout")
