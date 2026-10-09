"""mace gui connect, unit level (connect.py): ssh argv for both hops, remote
command quoting (injection-safe through two shells), profiles (TOML round
trip, 0600, unknown keys kept), first-run prompts, failure classification,
the auto-hop decision, the ssh -G user mapping and port fallback."""
import os
import shlex
import socket
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from mace.gui import connect as C

PROF = {"gateway": "me@gw.example.edu", "node": "dev-node-01", "dir": "~/runs"}


def test_jump_argv_exact():
    argv = C.build_ssh_argv({**PROF, "jump": "bastion.example.edu"}, 8765, 40000, 40001, "jump")
    assert argv[:3] == ["ssh", "-J", "bastion.example.edu,me@gw.example.edu"]
    assert argv[3:9] == ["-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=30",
                         "-o", "ServerAliveCountMax=4"]
    assert argv[9:12] == ["-L", "127.0.0.1:8765:127.0.0.1:40000", "dev-node-01"]
    assert argv[12].startswith("exec bash -lc ") and len(argv) == 13
    assert "-n" not in argv and "-t" not in argv and "-N" not in argv


def test_nested_argv_exact():
    argv = C.build_ssh_argv(PROF, 8765, 40000, 40001, "nested")
    assert argv[:7] == ["ssh", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=30",
                        "-o", "ServerAliveCountMax=4"]
    assert argv[7:10] == ["-L", "127.0.0.1:8765:127.0.0.1:40001", "me@gw.example.edu"]
    inner = shlex.split(argv[10])
    assert inner[:7] == ["ssh", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes", "-L",
                         "127.0.0.1:40001:127.0.0.1:40000"]
    assert inner[7] == "dev-node-01" and inner[8] == C.remote_command(PROF, 40000)
    assert "-n" not in inner and "-t" not in inner


def test_no_node_runs_on_the_gateway_with_two_workers():
    argv = C.build_ssh_argv({"gateway": "gw"}, 8765, 40000, 40001, "jump")
    assert argv[-2] == "gw" and "-J" not in argv
    assert "env MACE_GUI_MAX_WORKERS=2" in argv[-1]
    assert "MACE_GUI_MAX_WORKERS" not in C.remote_command(PROF, 40000)


def test_node_user_from_the_gateway_alias():
    argv = C.build_ssh_argv({"gateway": "hpcc", "node": "dev-node-01"}, 8765, 1, 2, "jump", user_for_node="alice")
    assert "alice@dev-node-01" in argv
    argv = C.build_ssh_argv({"gateway": "hpcc", "node": "bob@dev-node-01"}, 8765, 1, 2, "jump", user_for_node="alice")
    assert "bob@dev-node-01" in argv


def test_effective_node_user_via_ssh_g(tmp_path, monkeypatch):
    fake = tmp_path / "ssh"
    fake.write_text("#!/bin/sh\necho 'hostname hpcc.example.edu'\necho 'user zqxalias'\necho 'port 22'\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:/usr/bin:/bin")
    assert C.effective_node_user("hpcc") == "zqxalias"
    assert C.effective_node_user(f"{C.local_user()}@x") is None
    assert C.effective_node_user("someone@x") == "someone"


NASTY = ["a b", "$HOME/x", 'q"uote', "it's", "a'; rm -rf ~; '", "x`id`", "$(id)", "semi;colon"]


@pytest.mark.parametrize("value", NASTY)
def test_dir_reaches_cd_as_one_literal_argument_through_two_shells(tmp_path, value):
    """The folder (and mace) values arrive as single literal arguments, also
    through the nested hop's two levels of shell parsing."""
    home = tmp_path / "home"
    target = tmp_path / "srv" / value
    target.mkdir(parents=True)
    rec = tmp_path / "rec"
    mace = tmp_path / "fake mace"
    mace.write_text(f"#!/bin/sh\npwd > {shlex.quote(str(rec))}\nfor a in \"$@\"; do printf '%s\\n' \"$a\" >> "
                    f"{shlex.quote(str(rec))}; done\n")
    mace.chmod(0o755)
    prof = {"gateway": "gw", "node": "n1", "dir": str(target), "mace": shlex.quote(str(mace))}
    rcmd = C.remote_command(prof, 41234)
    # the nested hop: the gateway's shell parses the inner ssh line; take its last argument as the node's command
    inner = C.build_ssh_argv(prof, 8765, 41234, 41235, "nested")[-1]
    node_cmd = subprocess.run(["sh", "-c", 'eval "set -- $0"; for a; do last=$a; done; printf "%s" "$last"', inner],
                              capture_output=True, text=True, check=True).stdout
    assert node_cmd == rcmd
    # the node's login shell runs it (bash -lc with a clean HOME)
    env = {"HOME": str(home), "PATH": "/usr/bin:/bin"}
    home.mkdir(exist_ok=True)
    r = subprocess.run(["sh", "-c", node_cmd], env=env, capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    lines = rec.read_text().splitlines()
    assert lines[0] == str(target)
    assert lines[1:] == ["--no-banner", "gui", "--managed", "--port", "41234", "--dir", "."]
    assert not (Path.home() / "rm").exists()


def test_tilde_dir_is_home_relative(tmp_path):
    home = tmp_path / "home"
    (home / "my runs").mkdir(parents=True)
    rec = tmp_path / "rec"
    mace = tmp_path / "m"
    mace.write_text(f"#!/bin/sh\npwd > {rec}\n")
    mace.chmod(0o755)
    cmd = C.remote_command({"gateway": "g", "node": "n", "dir": "~/my runs", "mace": str(mace)}, 1)
    subprocess.run(["sh", "-c", cmd], env={"HOME": str(home), "PATH": "/usr/bin:/bin"}, check=True, timeout=30)
    assert rec.read_text().strip() == str(home / "my runs")


def test_profiles_round_trip_0600_unknown_keys_kept(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    data = {"defaults": {"profile": "hpcc"},
            "profiles": {"hpcc": {"gateway": "me@gw.example.edu", "node": "dev-node-01", "local_port": 8800,
                                  "open": False, "ssh_options": ["-o", 'IdentityFile="~/.ssh/k 1"'],
                                  "setup": 'module load Python; echo "hi\\there"', "future_key": "kept"}}}
    path = C.save_profiles(data)
    assert path == tmp_path / "cfg" / "mace" / "connect.toml"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    back = C.load_profiles(path)
    assert back == data
    flat, bad = C.parse_flat(path.read_text())
    assert flat == data and bad == []
    assert path.read_text().startswith("# mace gui connect profiles")


def test_save_keeps_unknown_keys_of_a_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    C.save_profiles({"profiles": {"p": {"gateway": "g", "mystery": 3}}})
    assert C.main(["p", "--node", "n2", "--save", "--dry-run"]) == 0
    back = C.load_profiles()
    assert back["profiles"]["p"] == {"gateway": "g", "mystery": 3, "node": "n2"}


def test_list_and_forget(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    C.save_profiles({"defaults": {"profile": "a"}, "profiles": {"a": {"gateway": "g1"}, "b": {"gateway": "g2"}}})
    assert C.main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "a (default): g1" in out and "b: g2" in out
    assert C.main(["--forget", "a"]) == 0
    assert "a" not in C.load_profiles()["profiles"]
    assert C.main(["--forget", "zz"]) == 2


def test_first_run_prompts_and_login_node_warning():
    answers = iter(["me@gw.example.edu", "", "", "~/runs", "module load Python", "", "hpcc"])
    said = []
    prof = C.first_run_prompts(lambda q: next(answers), said.append)
    assert prof == {"gateway": "me@gw.example.edu", "dir": "~/runs", "setup": "module load Python", "_name": "hpcc"}
    assert any("Login nodes are shared; prefer a dev node" in s for s in said)


def test_no_tty_and_no_gateway_exits_2(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env["XDG_CONFIG_HOME"] = str(tmp_path)
    r = subprocess.run([sys.executable, str(Path(C.__file__).resolve().parents[2] / "mace_cli"), "--no-banner",
                        "gui", "connect"], env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 2 and "Pass --gateway" in r.stderr


@pytest.mark.parametrize("code,err,out,kind", [
    (2, ["mace gui: error: unrecognized arguments: --managed"], [], "old-remote"),
    (2, ["mace: error: argument command: invalid choice: 'gui' (choose from 'workflow', ...)"], [], "old-remote"),
    (127, ["bash: line 1: mace: command not found"], [], "mace-not-found"),
    (1, ["bash: line 1: cd: /nope: No such file or directory"], [], "mace-not-found"),
    (2, ["mace gui: /x is not a directory"], [], "dir"),
    (98, [], ["MACE_GUI_ERROR port-busy"], "port-busy"),
    (255, ["channel 2: open failed: administratively prohibited: open failed"], [], "forwarding"),
    (255, ["me@dev-node-01: Permission denied (publickey,gssapi-with-mic)."], [], "node-auth"),
    (255, ["me@gw.example.edu: Permission denied (publickey)."], [], "key"),
    (255, ["me@gw.example.edu: Permission denied (publickey,password)."], [], "key"),
    (255, ['Load key "/home/me/.ssh/id_ed25519": incorrect passphrase supplied to decrypt private key',
           "me@gw.example.edu: Permission denied (publickey,keyboard-interactive)."], [], "key"),
    (255, ["read_passphrase: can't open /dev/tty: No such device or address",
           "me@gw.example.edu: Permission denied (publickey)."], [], "key"),
    (255, ["me@gw.example.edu: Permission denied (password)."], [], "ssh"),
    (255, ["Host key verification failed."], [], "host-key"),
    (255, ["ssh: Could not resolve hostname gw: Name or service not known"], [], "ssh"),
])
def test_classify_failure(code, err, out, kind):
    got, msg = C.classify_failure(code, err, "dev-node-01", "me@gw.example.edu", out)
    assert got == kind, msg
    if kind == "old-remote":
        assert "older than this connect" in msg and "run-it-on-a-cluster" in msg
    if kind == "key":
        assert "ssh-add" in msg and 'eval "$(ssh-agent)"' in msg and "OpenSSH Authentication Agent" in msg
        assert "Host alias from ~/.ssh/config" in msg
    if kind == "forwarding":
        assert "administratively prohibited" in msg and "ask your administrators" in msg
        assert "OnDemand" not in msg


def test_auto_hop_retries_only_on_node_refusal():
    assert C.node_auth_refused(["me@dev-node-01: Permission denied (publickey)."], "dev-node-01", "me@gw")
    assert not C.node_auth_refused(["me@gw: Permission denied (publickey)."], "dev-node-01", "me@gw")
    assert not C.node_auth_refused(["Host key verification failed.", "me@dev-node-01: Permission denied (x)."],
                                   "dev-node-01", "me@gw")
    assert not C.node_auth_refused([], "dev-node-01", "me@gw")


@pytest.mark.parametrize("text,old", [("OpenSSH_for_Windows_7.7p1, LibreSSL 2.6.5", True),
                                      ("OpenSSH_for_Windows_8.1p1", False), ("OpenSSH_for_Windows_9.5p1", False),
                                      ("OpenSSH_10.0p2, OpenSSL 3.5", False)])
def test_windows_ssh_version(text, old):
    assert C.windows_ssh_too_old(text) is old


def test_local_port_fallback_warns(capsys):
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen(1)
        p = busy.getsockname()[1]
        said = []
        got = C.pick_local_port(p, warn=said.append)
    assert got != p and said and f"Drafts saved under port {p} won't be offered on {got}" in said[0]


def test_local_url_rewrites_only_the_port():
    assert C._local_url("http://127.0.0.1:41234/s/abc/?login=xyz", 8765) == "http://127.0.0.1:8765/s/abc/?login=xyz"


def test_dry_run_prints_both_hops_for_auto(capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert C.main(["--dry-run", "--gateway", "me@gw", "--node", "n1", "--dir", "~/x"]) == 0
    out = capsys.readouterr().out
    assert "# hop jump" in out and "# hop nested" in out and "token" not in out.lower()


def test_stopped_page_command_quotes_the_folder(tmp_path):
    """The Stopped page's "mace gui --dir <folder>" is one shell word, also with spaces or quotes."""
    import re
    import shutil
    import subprocess
    from pathlib import Path
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    js = (Path(C.__file__).parent / "static" / "js" / "lazy" / "connect_ui.js").read_text()
    fn = re.search(r"export (function shellWord\(text\) \{.*?\n\})", js, re.S).group(1)
    script = tmp_path / "t.js"
    script.write_text(fn + '\nfor (const p of ["/home/u/runs", "/home/u/my runs", "/x/it\'s"]) console.log(shellWord(p));\n')
    out = subprocess.run([node, str(script)], capture_output=True, text=True, timeout=30).stdout.splitlines()
    assert out == ["/home/u/runs", "'/home/u/my runs'", "'/x/it'\"'\"'s'"]
    import shlex
    assert [shlex.split(w)[0] for w in out] == ["/home/u/runs", "/home/u/my runs", "/x/it's"]


def test_key_failure_needs_ssh_exit_255():
    denied = ["me@gw.example.edu: Permission denied (publickey)."]
    assert C.key_auth_failed(255, denied)
    assert C.key_auth_failed(255, ["Enter passphrase for key '/home/me/.ssh/id_rsa': "])
    assert C.key_auth_failed(255, ["sign_and_send_pubkey: signing failed for ED25519 \"/home/me/.ssh/id_ed25519\" "
                                   "from agent: agent refused operation", *denied])
    assert not C.key_auth_failed(1, denied)          # the remote command failed, not ssh
    assert not C.key_auth_failed(255, ["ssh: Could not resolve hostname gw: Name or service not known"])


@pytest.mark.parametrize("gateway,hosts,flagged", [
    ("netid@university.edu", [], True),
    ("NetID@University.EDU", [], True),
    ("me@cluster.org", [], True),                   # a real two-label login host: warned, an alias silences it
    ("netid@login.university.edu", [], False),
    ("me@hpcc.example.edu", [], False),
    ("hpcc", [], False),
    ("me@gw", [], False),
    ("me@10.0.0.5", [], False),
    ("netid@university.edu", ["university.edu"], False),
    ("netid@university.edu", ["*.edu"], False),
    ("netid@university.edu", ["other", "university.*"], False),
    ("netid@university.edu", ["hpcc", "*.example.org"], True),
])
def test_gateway_looks_like_email(gateway, hosts, flagged):
    assert C.gateway_looks_like_email(gateway, hosts) is flagged


def test_ssh_config_host_patterns(tmp_path):
    cfg = tmp_path / "config"
    cfg.write_text("Host hpcc university.edu\n  HostName login.university.edu\n  User netid\n"
                   "host=*.example.org !bad.example.org\nHostKeyAlgorithms ssh-ed25519\nMatch host x\n")
    assert C.ssh_config_hosts(cfg) == ["hpcc", "university.edu", "*.example.org"]
    assert C.ssh_config_hosts(tmp_path / "missing") == []


def _stub_ssh(tmp_path, stderr_text):
    """An "ssh" that logs that it ran, prints ssh's message and exits 255."""
    fakebin = tmp_path / "fakebin"
    fakebin.mkdir()
    ran = tmp_path / "ran"
    ssh = fakebin / "ssh"
    ssh.write_text("#!/bin/sh\necho \"$@\" >> %s\nprintf '%%s\\n' %s >&2\nsleep 0.2\nexit 255\n"
                   % (shlex.quote(str(ran)), " ".join(shlex.quote(x) for x in stderr_text)))
    ssh.chmod(0o755)
    return fakebin, ran


def _connect(tmp_path, fakebin, *args):
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON") and k not in ("BROWSER",)}
    env.update(PATH=f"{fakebin}:/usr/bin:/bin", XDG_CONFIG_HOME=str(tmp_path / "cfg"), HOME=str(tmp_path / "home"))
    (tmp_path / "home").mkdir(exist_ok=True)
    return subprocess.run([sys.executable, str(Path(C.__file__).resolve().parents[2] / "mace_cli"), "--no-banner",
                           "gui", "connect", "--no-open", *args], env=env, stdin=subprocess.DEVNULL,
                          capture_output=True, text=True, timeout=60)


@pytest.mark.parametrize("stderr_text", [
    ["netid@login.example.edu: Permission denied (publickey)."],
    ["read_passphrase: can't open /dev/tty: No such device or address",
     "netid@login.example.edu: Permission denied (publickey,password)."],
    ['Load key "/home/netid/.ssh/id_ed25519": incorrect passphrase supplied to decrypt private key',
     "netid@login.example.edu: Permission denied (publickey,gssapi-keyex,gssapi-with-mic)."],
])
def test_key_failure_prints_the_agent_hint(tmp_path, stderr_text):
    fakebin, ran = _stub_ssh(tmp_path, stderr_text)
    r = _connect(tmp_path, fakebin, "--gateway", "netid@login.example.edu")
    assert r.returncode == 1 and ran.exists()
    assert "mace gui connect: " + C.KEY_HINT in r.stdout
    assert "looks like an email address" not in r.stderr
    assert "ssh could not connect: check the gateway" not in r.stdout


def test_email_like_gateway_warns_and_still_connects(tmp_path):
    fakebin, ran = _stub_ssh(tmp_path, ["ssh: connect to host university.edu port 22: Connection timed out"])
    C.save_profiles({"defaults": {"profile": "hpcc"}, "profiles": {"hpcc": {"gateway": "netid@university.edu"}}},
                    tmp_path / "cfg" / "mace" / "connect.toml")
    r = _connect(tmp_path, fakebin)
    want = C.email_gateway_warning("netid@university.edu", tmp_path / "cfg" / "mace" / "connect.toml", "hpcc")
    assert want in r.stderr
    assert "user@login-host" in want and "mace gui connect hpcc --gateway <user@login-host> --save" in want
    # a warning: ssh still ran, to the saved gateway
    assert "Connecting to netid@university.edu..." in r.stdout
    assert "netid@university.edu" in ran.read_text()
    assert C.KEY_HINT not in r.stdout
    # an ~/.ssh/config Host for it silences the warning
    (tmp_path / "home" / ".ssh").mkdir(parents=True)
    (tmp_path / "home" / ".ssh" / "config").write_text("Host university.edu\n  HostName login.university.edu\n")
    r = _connect(tmp_path, fakebin)
    assert "looks like an email address" not in r.stderr
