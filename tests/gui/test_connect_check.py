"""`mace gui connect --check` / `--fix-config` (connect_check.py) with a fake
runner for ssh -G, ssh-add -l, ssh-keygen -lf and the probes' exit codes; one
real `mace_cli gui connect --check --no-net` subprocess; and the connect UX
around it (ConnectTimeout, "cannot reach host", an unresolvable gateway, the
remote MACE version, the mace_cli fallback)."""
import os
import shlex
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from mace.gui import connect as C
from mace.gui import connect_check as K

REPO = Path(__file__).resolve().parents[2]
CP = subprocess.CompletedProcess

GW_G = ("user netid\nhostname login.example.edu\nport 22\nidentityfile ~/.ssh/id_cluster\nidentitiesonly yes\n"
        "controlmaster false\nproxyjump none\n")
NODE_G = ("user netid\nhostname dev-node-01\nport 22\nidentityfile ~/.ssh/id_rsa\nidentityfile ~/.ssh/id_ed25519\n"
          "controlmaster false\nproxyjump none\n")
FP = "SHA256:Zm9vYmFyYmF6cXV4cXV1eGZvb2Jhcg"


class Fake:
    """subprocess.run for the checks: tables for ssh -G, the agent, ssh-keygen
    and the exit code and stderr of each probe (gateway, jump, nested)."""

    def __init__(self, g=None, agent=(2, "", "Could not open a connection to your authentication agent."),
                 probes=None, cm=255, ps="Running Manual", version="OpenSSH_9.6p1, OpenSSL 3.0.13"):
        self.g = {"hpcc": GW_G, "netid@dev-node-01": NODE_G, "dev-node-01": NODE_G, **(g or {})}
        self.agent, self.probes, self.cm, self.ps, self.version = agent, probes or {}, cm, ps, version
        self.calls = []

    def __call__(self, argv, **kw):
        self.calls.append(list(argv))
        if argv[:2] == ["ssh", "-V"]:
            return CP(argv, 0, "", self.version + "\n")
        if argv[:2] == ["ssh", "-G"]:
            return CP(argv, 0, self.g.get(argv[2], f"user netid\nhostname {argv[2].split('@')[-1]}\nport 22\n"), "")
        if argv[:2] == ["ssh-add", "-l"]:
            return CP(argv, *self.agent)
        if argv[0] == "ssh-keygen":
            return CP(argv, 0, f"256 {FP} netid@laptop (ED25519)\n", "")
        if argv[0] == "powershell":
            return CP(argv, 0, self.ps + "\n", "")
        if argv[:3] == ["ssh", "-O", "check"]:
            return CP(argv, self.cm, "", "")
        if argv[0] == "ssh":
            kind = "nested" if argv[-1].startswith("ssh ") else "jump" if argv[-2] != "hpcc" else "gateway"
            rc, err = self.probes.get(kind, (0, ""))
            return CP(argv, rc, "", err)
        raise AssertionError(argv)

    def ran(self, kind):
        return [c for c in self.calls if c[0] == "ssh" and c[1] not in ("-G", "-V", "-O") and
                (("nested" == kind) == c[-1].startswith("ssh "))]


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    monkeypatch.setenv("HOME", str(h))          # node_key_options looks for the key files under it
    (h / ".ssh").mkdir(parents=True)
    os.chmod(h / ".ssh", 0o700)
    (h / ".ssh" / "config").write_text("Host hpcc\n  HostName login.example.edu\n  User netid\n"
                                       "  IdentityFile ~/.ssh/id_cluster\n")
    (h / ".ssh" / "id_cluster").write_text("private half: never read by the check\n")
    (h / ".ssh" / "id_cluster.pub").write_text("ssh-ed25519 AAAA netid@laptop\n")
    for f in ("config", "id_cluster"):
        os.chmod(h / ".ssh" / f, 0o600)
    return h


PROF = {"gateway": "hpcc", "node": "dev-node-01"}


def _check(home, fake, prof=PROF, name="p", no_net=False, system="linux", env=None):
    out = []
    ck = K.Checker(dict(prof), name, no_net=no_net, runner=fake, which=lambda _: "/usr/bin/ssh",
                   resolver=lambda *_: [("ok",)], system=system, home=home,
                   env={"SSH_AUTH_SOCK": "/run/agent.sock"} if env is None else env, out=out.append)
    code = ck.run_all()
    return code, "\n".join(out), ck


def test_node_password_problem_is_predicted(home):
    """The maintainer's case: Host hpcc names ~/.ssh/id_cluster, the node gets only ssh's default key files, and
    no agent holds the key: a plain ssh -J asks for the node's password; connect adds the key."""
    code, out, _ = _check(home, Fake(), no_net=True)
    assert "INFO  for the jump hop, the connection also gives dev-node-01: -o IdentityFile=~/.ssh/id_cluster " \
           "-o IdentitiesOnly=yes" in out
    assert ("WARN  dev-node-01 gets none of ~/.ssh/id_cluster from your ssh config, and the agent does not hold "
            "them, so a plain ssh -J hpcc dev-node-01 would ask for the "
            "password of dev-node-01") in out
    assert "fix: the connection adds them for you (above); for plain ssh too, add a Host block for the node " \
           "(--fix-config prints one) or run ssh-add ~/.ssh/id_cluster" in out
    assert "WARN  cannot reach an ssh agent (Could not open a connection to your authentication agent.)" in out
    assert 'fix: eval "$(ssh-agent)" && ssh-add ~/.ssh/id_cluster' in out
    assert "SKIP  live probes skipped: no ssh login was tried" in out
    assert out.splitlines()[-1] == "Verdict (without the probes): no problem found. Connect with: mace gui connect p"
    assert code == 0


def test_agent_holding_the_key_covers_the_node(home):
    fake = Fake(agent=(0, f"256 {FP} netid@laptop (ED25519)\n3072 SHA256:other /home/x/.ssh/id_rsa (RSA)\n", ""))
    code, out, _ = _check(home, fake, no_net=True)
    assert "OK    the jump hop offers dev-node-01 your gateway key (the agent holds ~/.ssh/id_cluster)" in out
    # fingerprints only: no comments (email addresses, paths)
    assert f"OK    agent: 2 key(s): {FP} (ED25519), SHA256:other (RSA)" in out
    assert "netid@laptop" not in out and "/home/x" not in out
    assert f"OK    ~/.ssh/id_cluster ({FP}) is in the agent" in out
    # only the public half went to ssh-keygen
    kg = [c for c in fake.calls if c[0] == "ssh-keygen"]
    assert kg and all(c[-1].endswith(".pub") for c in kg)
    assert code == 0


def test_empty_agent_and_key_not_in_it(home):
    code, out, _ = _check(home, Fake(agent=(1, "The agent has no identities.\n", "")), no_net=True)
    assert "WARN  the ssh agent holds no keys\n      fix: ssh-add ~/.ssh/id_cluster" in out
    assert f"WARN  ~/.ssh/id_cluster ({FP}) is not in the agent" in out


def test_probes_jump_refused_nested_works(home):
    fake = Fake(probes={"jump": (255, "netid@dev-node-01: Permission denied (publickey,password).\n")})
    code, out, _ = _check(home, fake)
    assert "OK    ssh hpcc true: logged in without a prompt" in out
    assert "FAIL  jump hop: dev-node-01 refused your key from this computer" in out
    assert "OK    nested hop: the gateway logs in to dev-node-01 by itself" in out
    assert out.splitlines()[-1] == ("Verdict: only the nested hop works. Connect with: mace gui connect p --hop "
                                    "nested --save")
    assert code == 0
    gw = fake.ran("gateway")[0]
    assert gw[:5] == ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"] and gw[-2:] == ["hpcc", "true"]
    jump = [c for c in fake.ran("jump") if c[-2] != "hpcc"][0]
    assert jump[1:3] == ["-J", "hpcc"] and "IdentityFile=~/.ssh/id_cluster" in jump and jump[-1] == "true"
    nested = fake.ran("nested")[0]
    assert shlex.split(nested[-1]) == ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "dev-node-01",
                                       "true"]


def test_probes_both_hops_work(home):
    code, out, _ = _check(home, Fake())
    assert "OK    jump hop: ssh -J hpcc dev-node-01 true works" in out
    assert out.splitlines()[-1] == "Verdict: the jump hop works. Connect with: mace gui connect p"
    code, out, _ = _check(home, Fake(), prof={**PROF, "hop": "nested"})
    assert out.splitlines()[-1] == "Verdict: the nested hop works. Connect with: mace gui connect p"


def test_two_factor_gateway_stops_the_probes(home):
    fake = Fake(probes={"gateway": (255, "netid@login.example.edu: Permission denied (keyboard-interactive).\n")})
    code, out, _ = _check(home, fake)
    assert ("WARN  hpcc asks for a 2FA code or a password, which the check cannot type; the hop probes stop "
            "here" in out)
    assert "fix: turn on connection sharing (ControlMaster, see --fix-config), log in once with ssh hpcc" in out
    assert not [c for c in fake.ran("jump") if c[-2] != "hpcc"] and not fake.ran("nested")
    code, out, _ = _check(home, fake, system="windows")
    assert "fix: skip the probes with -NoNet (Windows ssh has no connection sharing)" in out


def test_refused_key_and_unreachable_and_host_key(home):
    _, out, _ = _check(home, Fake(probes={"gateway": (255, "netid@login.example.edu: Permission denied "
                                                           "(publickey,keyboard-interactive).\n")}))
    assert "FAIL  hpcc refused your keys (Permission denied: publickey,keyboard-interactive)" in out
    code, out, _ = _check(home, Fake(probes={"gateway": (255, "ssh: connect to host login.example.edu port 22: "
                                                              "Connection timed out\n")}))
    assert "FAIL  cannot reach host login.example.edu (timeout)" in out
    assert "fix: check the name, and your network or VPN (ssh -v login.example.edu true shows where it stops)" in out
    assert out.splitlines()[-1] == "Verdict: fix the FAIL lines above, then check again." and code == 1
    _, out, _ = _check(home, Fake(probes={"gateway": (255, "Host key verification failed.\n")}))
    assert "FAIL  ssh does not know the host key of hpcc yet, or it changed" in out
    _, out, _ = _check(home, Fake(probes={"jump": (255, "channel 0: open failed: connect failed: Name or service "
                                                        "not known\nstdio forwarding failed\n")}))
    assert "FAIL  cannot reach host dev-node-01 (DNS on the gateway: the name does not resolve)" in out


def test_live_control_master_is_reused(home):
    fake = Fake(g={"hpcc": GW_G.replace("controlmaster false", "controlmaster auto")}, cm=0)
    _, out, _ = _check(home, fake)
    assert "OK    connection sharing is on for hpcc (ControlMaster auto)" in out
    assert "OK    a shared connection to hpcc is open: the probes reuse it" in out
    assert ["ssh", "-O", "check", "hpcc"] in fake.calls


def test_permissions(home):
    os.chmod(home / ".ssh", 0o755)
    os.chmod(home / ".ssh" / "id_cluster", 0o644)
    code, out, _ = _check(home, Fake(), no_net=True)
    assert "WARN  ~/.ssh is mode 755; ssh wants 700\n      fix: chmod 700 ~/.ssh" in out
    assert "FAIL  ~/.ssh/id_cluster is mode 644; ssh wants 600" in out and "fix: chmod 600 ~/.ssh/id_cluster" in out
    assert code == 1


def test_windows_checks(home):
    fake = FakeKeys(False, g={"hpcc": GW_G.replace("controlmaster false", "controlmaster auto")}, ps="Stopped Disabled",
                version="OpenSSH_for_Windows_7.7p1, LibreSSL 2.6.5")
    out = []
    ck = K.Checker(dict(PROF), "p", no_net=True, runner=fake, which=lambda _: r"C:\Program Files\Git\usr\bin\ssh.exe",
                   system="windows", home=home, env={}, out=out.append)
    code = ck.run_all()
    text = "\n".join(out)
    assert "WARN  Windows OpenSSH OpenSSH_for_Windows_7.7p1, LibreSSL 2.6.5 is older than 8.1" in text
    assert r"WARN  this ssh is C:\Program Files\Git\usr\bin\ssh.exe, not Windows' own OpenSSH" in text
    assert "WARN  ControlMaster is set for hpcc, which Windows OpenSSH does not support" in text
    assert ("FAIL  the OpenSSH Authentication Agent service (ssh-agent) is disabled\n      fix: in an administrator "
            "PowerShell: Get-Service ssh-agent | Set-Service -StartupType Automatic; Start-Service ssh-agent") in text
    assert "icacls <key file> /inheritance:r" in text and "chmod" not in text
    assert code == 1


def test_email_and_unresolvable_gateway(home):
    out = []
    ck = K.Checker({"gateway": "netid@university.edu"}, None, runner=Fake(), which=lambda _: "/usr/bin/ssh",
                   resolver=lambda *_: (_ for _ in ()).throw(socket.gaierror("no")), system="linux", home=home,
                   env={}, out=out.append)
    ck.run_all()
    text = "\n".join(out)
    assert "WARN  the gateway netid@university.edu looks like an email address" in text
    assert "fix: mace gui connect --gateway <user@login-host> --save" in text
    assert "WARN  university.edu matches no Host in ~/.ssh/config and does not resolve (DNS)" in text


def test_fix_config_prints_a_block_per_platform(home):
    for system, keychain, cm in (("linux", False, True), ("darwin", True, True), ("windows", False, False)):
        ck = K.Checker({"gateway": "netid@hpcc.example.edu", "node": "dev-node-01"}, "p", runner=Fake(
            g={"netid@hpcc.example.edu": GW_G.replace("login.example.edu", "hpcc.example.edu")}),
            system=system, home=home, env={}, out=lambda _: None)
        block = "\n".join(ck.fix_config())
        assert "Host hpcc\n    HostName hpcc.example.edu\n    User netid\n    IdentityFile ~/.ssh/id_cluster" in block
        assert ("UseKeychain yes" in block) is keychain and ("ControlMaster auto" in block) is cm
        assert "Host dev-node-01\n    User netid\n    IdentityFile ~/.ssh/id_cluster\n    IdentitiesOnly yes\n" \
               "    ProxyJump hpcc" in block
        assert block.splitlines()[-1] == ("# Then connect with: mace gui connect p --gateway hpcc --node dev-node-01 "
                                          "--save")
    assert K.suggest_alias("login.example.edu") == "example" and K.suggest_alias("hpcc.example.edu") == "hpcc"


def test_check_never_writes_ssh_files(home):
    before = {p: p.read_bytes() for p in (home / ".ssh").iterdir()}
    _check(home, Fake())
    K.Checker(dict(PROF), "p", runner=Fake(), home=home, out=lambda _: None).fix_config()
    assert {p: p.read_bytes() for p in (home / ".ssh").iterdir()} == before


def test_messages_have_no_quotes_or_dollars_except_shell_fixes():
    for key, (plat, text) in K.MESSAGES.items():
        assert plat in ("all", "posix", "win"), key
        if key not in ("agent-none-fix", "perm-win-fix"):
            assert '"' not in text and "$" not in text and "`" not in text, key


# ---- the real invocation path


def _fake_ssh_bin(tmp_path):
    fb = tmp_path / "fakebin"
    fb.mkdir()
    (fb / "ssh").write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = -V ]; then echo 'OpenSSH_9.6p1, OpenSSL 3.0' >&2; exit 0; fi\n"
        "if [ \"$1\" = -G ]; then\n"
        "  case \"$2\" in\n"
        "    hpcc) printf 'user netid\\nhostname login.example.edu\\nport 22\\nidentityfile ~/.ssh/id_cluster\\n'"
        " ;;\n"
        "    *) printf 'user netid\\nhostname %s\\nport 22\\nidentityfile ~/.ssh/id_ed25519\\n' \"$2\" ;;\n"
        "  esac; exit 0\nfi\n"
        "echo \"$@\" >> \"$FAKE_SSH_RAN\"\nexit 255\n")
    (fb / "ssh").chmod(0o755)
    return fb


def test_mace_cli_check_no_net(tmp_path, home):
    fb = _fake_ssh_bin(tmp_path)
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON") and k != "SSH_AUTH_SOCK"}
    env.update(PATH=f"{fb}:{env.get('PATH', '/usr/bin:/bin')}", HOME=str(home), XDG_CONFIG_HOME=str(tmp_path / "c"),
               FAKE_SSH_RAN=str(tmp_path / "ran"))
    r = subprocess.run([sys.executable, str(REPO / "mace_cli"), "--no-banner", "gui", "connect", "--gateway", "hpcc",
                        "--node", "dev-node-01", "--check", "--no-net", "--fix-config"], env=env,
                       capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
    assert r.returncode == 0, r.stdout + r.stderr
    out = r.stdout
    assert out.startswith("Checking the ssh setup for hpcc -> dev-node-01 (read-only; nothing is changed)\n")
    assert "so a plain ssh -J hpcc dev-node-01 would ask for the password of dev-node-01" in out
    assert "SKIP  live probes skipped" in out
    assert ("Verdict (without the probes): no problem found. Connect with: mace gui connect --gateway hpcc --node "
            "dev-node-01") in out
    assert "# A suggested ~/.ssh/config block" in out and "Host dev-node-01" in out
    assert not (tmp_path / "ran").exists()           # --no-net: no ssh login at all
    assert not (tmp_path / "c" / "mace" / "connect.toml").exists()


# ---- connect UX (part B)


def test_connect_timeout_is_the_default_unless_given():
    prof = {"gateway": "gw", "node": "n"}
    for hop in ("jump", "nested"):
        argv = C.build_ssh_argv(prof, 1, 2, 3, hop)
        assert argv.count("ConnectTimeout=20") == 1
    argv = C.build_ssh_argv({**prof, "ssh_options": ["-oConnectTimeout=60"]}, 1, 2, 3, "jump")
    assert "ConnectTimeout=20" not in argv and "-oConnectTimeout=60" in argv
    assert "ConnectTimeout=20" in C.build_ssh_argv(prof, 1, 2, 3, "nested")[-1]


def test_unreachable_message():
    kind, m = C.classify_failure(255, ["ssh: connect to host gw.example.edu port 22: Connection refused"], "n", "gw")
    assert kind == "unreachable"
    assert m == "cannot reach host gw.example.edu (connection refused): check the name, and your network or VPN."


def test_gateway_unresolvable():
    bad = lambda *_: (_ for _ in ()).throw(socket.gaierror("no"))
    good = lambda *_: [("ok",)]
    run = lambda argv, **kw: CP(argv, 0, f"hostname {argv[2].split('@')[-1]}\nproxyjump none\n", "")
    assert C.gateway_unresolvable("me@typo.example.edu", "", run, bad, []) == "typo.example.edu"
    assert C.gateway_unresolvable("me@login.example.edu", "", run, good, []) is None
    assert C.gateway_unresolvable("hpcc", "", run, bad, ["hpcc"]) is None            # a Host alias
    assert C.gateway_unresolvable("me@x.example.edu", "bastion", run, bad, []) is None   # resolved over there
    pj = lambda argv, **kw: CP(argv, 0, "hostname inner\nproxyjump bastion\n", "")
    assert C.gateway_unresolvable("me@inner", "", pj, bad, []) is None
    assert C.unresolvable_warning("typo.example.edu").startswith(
        "Warning: typo.example.edu matches no Host in ~/.ssh/config and does not resolve (DNS)")


def test_version_note():
    assert C.version_note("1.1.6", "1.1.6", "n") is None
    w = C.version_note("1.1.6", "1.1.5", "dev-node-01", "/home/me/MACE")
    assert w.startswith("Warning: MACE on dev-node-01 is 1.1.5, older than 1.1.6 here.")
    assert "git -C /home/me/MACE pull" in w
    assert "git pull in its MACE_HOME folder" in C.version_note("1.1.6", "1.1.4", "n")
    assert C.version_note("1.1.5", "1.1.6", "n").startswith("Note: MACE here is 1.1.5, on n 1.1.6.")


def _run_remote(cmd, home, extra_env=None):
    env = {"HOME": str(home), "PATH": "/usr/bin:/bin", **(extra_env or {})}
    return subprocess.run(["sh", "-c", cmd], env=env, capture_output=True, text=True, timeout=30)


@pytest.mark.parametrize("where", ["mace", "MACE", "MACE_HOME"])
def test_mace_cli_fallback_on_the_node(tmp_path, where):
    home = tmp_path / "nodehome"
    (home / "runs").mkdir(parents=True)
    inst = tmp_path / "inst" if where == "MACE_HOME" else home / where
    inst.mkdir(parents=True)
    rec = tmp_path / "rec"
    (inst / "mace_cli").write_text(f"#!/bin/sh\nprintf '%s\\n' \"$@\" > {shlex.quote(str(rec))}\n")
    (inst / "mace_cli").chmod(0o755)
    cmd = C.remote_command({"gateway": "g", "node": "n", "dir": "~/runs"}, 41234)
    r = _run_remote(cmd, home, {"MACE_HOME": str(inst)} if where == "MACE_HOME" else None)
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == [f"MACE_GUI_USING {inst / 'mace_cli'}"]
    assert rec.read_text().split() == ["--no-banner", "gui", "--managed", "--port", "41234", "--dir", "."]


def test_mace_fallback_from_a_missing_path_and_none_found(tmp_path):
    home = tmp_path / "h"
    (home / "MACE").mkdir(parents=True)
    (home / "MACE" / "mace_cli").write_text("#!/bin/sh\necho ran\n")
    (home / "MACE" / "mace_cli").chmod(0o755)
    r = _run_remote(C.remote_command({"gateway": "g", "node": "n", "mace": "~/MACE/mace"}, 1), home)
    assert r.stdout.splitlines() == [f"MACE_GUI_USING {home}/MACE/mace_cli", "ran"]
    # nothing found: the usual "not found" (exit 127), which connect explains
    r = _run_remote(C.remote_command({"gateway": "g", "node": "n"}, 1), tmp_path)
    assert r.returncode == 127 and C.classify_failure(127, r.stderr.splitlines())[0] == "mace-not-found"
    # a full mace_cli path or a multi-word command runs as given
    assert "MACE_GUI_USING" not in C.remote_command({"node": "n", "mace": "~/m/mace_cli"}, 1)
    assert "MACE_GUI_USING" not in C.remote_command({"node": "n", "mace": "python3 ~/m/mace_cli"}, 1)


def test_connect_says_which_mace_it_used(tmp_path):
    fb = tmp_path / "fakebin"
    fb.mkdir()
    (fb / "ssh").write_text("#!/bin/sh\n[ \"$1\" = -G ] && exit 0\necho 'MACE_GUI_USING /home/me/MACE/mace_cli'\n"
                            "echo 'bash: line 1: cd: /x: No such file or directory' >&2\nexit 1\n")
    (fb / "ssh").chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON") and k != "BROWSER"}
    env.update(XDG_CONFIG_HOME=str(tmp_path / "cfg"), HOME=str(tmp_path / "home"),
               PATH=f"{fb}:/usr/bin:/bin")
    (tmp_path / "home").mkdir()
    r = subprocess.run([sys.executable, str(REPO / "mace_cli"), "--no-banner", "gui", "connect", "--no-open",
                        "--gateway", "me@gw", "--node", "n1"], env=env, capture_output=True, text=True,
                       timeout=60, stdin=subprocess.DEVNULL)
    assert "mace is not on n1's PATH there; using /home/me/MACE/mace_cli." in r.stdout, r.stdout
    assert "mace gui connect --mace /home/me/MACE/mace_cli --save" in r.stdout


# ---- the standalone scripts: the same checks and words (bash --check, PowerShell -Check)

SH = REPO / "scripts" / "mace-gui-connect.sh"
PS1 = REPO / "scripts" / "mace-gui-connect.ps1"
FAKE_SSH = r'''#!/usr/bin/env bash
# a fake ssh for --check: -V, -G (FAKE_GW_G for hpcc), -O check, and probes whose exit code and
# stderr come from FAKE_<GATEWAY|JUMP|NESTED>_RC and _ERR
case "$1" in
  -V) echo "OpenSSH_9.6p1, OpenSSL 3.0.13" >&2; exit 0 ;;
  -G) case "$2" in
        hpcc) printf '%s\n' "$FAKE_GW_G" ;;
        *) printf 'user netid\nhostname %s\nport 22\nidentityfile ~/.ssh/id_rsa\n' "${2#*@}"
           printf 'identityfile ~/.ssh/id_ed25519\n' ;;
      esac; exit 0 ;;
  -O) exit 255 ;;
esac
last="${*: -1}"; prev="${*: -2:1}"
if [[ "$last" == "ssh "* ]]; then kind=NESTED
elif [ "$prev" = hpcc ]; then kind=GATEWAY
else kind=JUMP; fi
rc="FAKE_${kind}_RC"; err="FAKE_${kind}_ERR"
echo "$kind $*" >> "${FAKE_SSH_RAN:-/dev/null}"
[ -n "${!err:-}" ] && printf '%s\n' "${!err}" >&2
exit "${!rc:-0}"
'''


def _script_env(tmp_path, home, **extra):
    fb = tmp_path / "fakebin"
    fb.mkdir(exist_ok=True)
    (fb / "ssh").write_text(FAKE_SSH)
    (fb / "ssh").chmod(0o755)
    drop = ("SSH_AUTH_SOCK", "BROWSER")
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON") and k not in drop}
    env.update(PATH=f"{fb}:{env.get('PATH', '/usr/bin:/bin')}", HOME=str(home), USERPROFILE=str(home),
               XDG_CONFIG_HOME=str(tmp_path / "cfg"), APPDATA=str(tmp_path / "cfg"), FAKE_GW_G=GW_G.strip(),
               FAKE_SSH_RAN=str(tmp_path / "ran"), **extra)
    return env


@pytest.fixture
def realkey(home):
    """A real key pair (no passphrase, made for the test), so ssh-keygen -lf works on its .pub."""
    import shutil
    if shutil.which("ssh-keygen") is None or shutil.which("ssh-add") is None:
        pytest.skip("needs ssh-keygen and ssh-add")
    for f in ("id_cluster", "id_cluster.pub"):
        (home / ".ssh" / f).unlink()
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "test", "-f",
                    str(home / ".ssh" / "id_cluster")], check=True, timeout=30)
    return home


def _norm(text, script):
    out = []
    for ln in text.replace(str(script), "mace gui connect").splitlines():
        out.append(ln[:-len(" --save")] if ln.endswith(" --save") else ln)
    return out


SCENARIOS = {
    "no-net": (["--no-net", "--fix-config"], {}),
    "jump-refused": ([], {"FAKE_JUMP_RC": "255",
                          "FAKE_JUMP_ERR": "netid@dev-node-01: Permission denied (publickey,password)."}),
    "both-hops": ([], {}),
    "two-factor": ([], {"FAKE_GATEWAY_RC": "255",
                        "FAKE_GATEWAY_ERR": "netid@login.example.edu: Permission denied (keyboard-interactive)."}),
    "timeout": ([], {"FAKE_GATEWAY_RC": "255",
                     "FAKE_GATEWAY_ERR": "ssh: connect to host login.example.edu port 22: Connection timed out"}),
    "node-dns": ([], {"FAKE_JUMP_RC": "255",
                      "FAKE_JUMP_ERR": "channel 0: open failed: connect failed: Name or service not known\n"
                                       "stdio forwarding failed",
                      "FAKE_NESTED_RC": "255", "FAKE_NESTED_ERR": "ssh: Could not resolve hostname dev-node-01: "
                                                                  "Name or service not known"}),
    "nested-profile": (["--hop", "nested"], {}),
}


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_bash_check_prints_what_mace_gui_connect_prints(tmp_path, realkey, scenario):
    flags, extra = SCENARIOS[scenario]
    env = _script_env(tmp_path, realkey, **extra)
    argv = ["--gateway", "hpcc", "--node", "dev-node-01", "--check", *flags]
    py = subprocess.run([sys.executable, str(REPO / "mace_cli"), "--no-banner", "gui", "connect", *argv], env=env,
                        capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL)
    sh = subprocess.run(["bash", str(SH), *argv], env=env, capture_output=True, text=True, timeout=120,
                        stdin=subprocess.DEVNULL)
    assert py.returncode == sh.returncode, (py.stdout, sh.stdout, sh.stderr)
    assert _norm(sh.stdout, SH) == _norm(py.stdout, SH), sh.stderr
    assert "Verdict" in py.stdout


def test_bash_check_permissions_and_agent_match(tmp_path, realkey):
    os.chmod(realkey / ".ssh", 0o755)
    os.chmod(realkey / ".ssh" / "id_cluster", 0o640)
    env = _script_env(tmp_path, realkey)
    # a short folder for the socket: a Unix socket path holds at most 107
    # bytes, which a long pytest temp path can pass ("path too long")
    import shutil
    import tempfile
    sockdir = tempfile.mkdtemp(prefix="mace-ag-", dir="/tmp" if os.path.isdir("/tmp") else None)
    sock = os.path.join(sockdir, "a.sock")
    agent = subprocess.run(["ssh-agent", "-a", sock], capture_output=True, text=True, timeout=30, env=env)
    if "SSH_AGENT_PID=" not in agent.stdout:
        shutil.rmtree(sockdir, ignore_errors=True)
        pytest.skip(f"ssh-agent did not start: {(agent.stderr or agent.stdout).strip()[:200]}")
    pid = int(agent.stdout.split("SSH_AGENT_PID=")[1].split(";")[0])
    try:
        env["SSH_AUTH_SOCK"] = sock
        os.chmod(realkey / ".ssh" / "id_cluster", 0o600)
        subprocess.run(["ssh-add", str(realkey / ".ssh" / "id_cluster")], env=env, capture_output=True,
                       check=True, timeout=30)
        os.chmod(realkey / ".ssh" / "id_cluster", 0o640)
        argv = ["--gateway", "hpcc", "--node", "dev-node-01", "--check", "--no-net"]
        py = subprocess.run([sys.executable, str(REPO / "mace_cli"), "--no-banner", "gui", "connect", *argv],
                            env=env, capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
        sh = subprocess.run(["bash", str(SH), *argv], env=env, capture_output=True, text=True, timeout=60)
    finally:
        os.kill(pid, 15)
        shutil.rmtree(sockdir, ignore_errors=True)
    assert _norm(sh.stdout, SH) == _norm(py.stdout, SH)
    assert "OK    the jump hop offers dev-node-01 your gateway key (the agent holds ~/.ssh/id_cluster)" in py.stdout
    assert "OK    agent: 1 key(s): SHA256:" in py.stdout and "(ED25519)" in py.stdout and " test " not in py.stdout
    assert "WARN  ~/.ssh is mode 755; ssh wants 700" in py.stdout
    assert "FAIL  ~/.ssh/id_cluster is mode 640" in py.stdout and py.returncode == sh.returncode == 1


def test_messages_are_in_the_scripts():
    """Every check message (its fixed words between the {placeholders}) is in the bash script (all, posix)
    and the PowerShell script (all, win)."""
    import re
    sh = SH.read_text().replace('\\"', '"').replace("\\$", "$")
    ps = PS1.read_text().replace("''", "'").replace("`$", "$")
    for key, (plat, text) in K.MESSAGES.items():
        for part in re.split(r"\{\w+\}", text):
            if len(part.strip()) < 6:
                continue
            if plat in ("all", "posix"):
                assert part.strip() in sh, (key, part)
            if plat in ("all", "win"):
                want = part.strip().replace("--fix-config", "-FixConfig").replace("--no-net", "-NoNet")
                assert want in ps, (key, part)


def test_scripts_have_the_check_flags():
    import re
    sh, ps = SH.read_text(), PS1.read_text()
    for f in ("check", "no-net", "fix-config"):
        assert re.search(rf"^\s+--{f}\)", sh, re.M), f
    for f in ("Check", "NoNet", "FixConfig"):
        assert f"[switch]${f}" in ps, f
    # the runtime additions: ConnectTimeout, the mace_cli fallback, MACE_GUI_USING, an unresolvable gateway
    for word in ("ConnectTimeout=20", "MACE_GUI_USING", "does not resolve (DNS)", "mace is not on"):
        assert word in sh and word in ps, word


def _pwsh():
    import shutil
    return shutil.which("pwsh")


@pytest.mark.skipif(_pwsh() is None, reason="needs pwsh")
@pytest.mark.parametrize("hop", ["jump", "nested"])
def test_powershell_dry_run_parity(tmp_path, realkey, hop):
    env = _script_env(tmp_path, realkey)
    flags = ["--gateway", "hpcc", "--node", "dev-node-01", "--hop", hop, "--mace", "~/MACE/mace", "--dry-run"]
    r = subprocess.run([sys.executable, str(REPO / "mace_cli"), "--no-banner", "gui", "connect", *flags], env=env,
                       capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
    want = shlex.split([ln for ln in r.stdout.splitlines() if ln.startswith("ssh ")][0])
    ps = subprocess.run([_pwsh(), "-NoProfile", "-File", str(PS1), "-Gateway", "hpcc", "-Node", "dev-node-01", "-Hop",
                         hop, "-Mace", "~/MACE/mace", "-DryRun"], env=env, capture_output=True, text=True, timeout=120)
    assert ps.returncode == 0, ps.stderr
    # the PowerShell list leaves out the ssh command itself; its "# hop ..." label and the one-line
    # "ssh --% ..." form to paste into a console (1.1.5) are not arguments
    assert [ln for ln in ps.stdout.splitlines() if not ln.startswith(("# ", "ssh --% "))] == want[1:]


@pytest.mark.skipif(_pwsh() is None, reason="needs pwsh")
@pytest.mark.parametrize("scenario", ["no-net", "jump-refused", "two-factor", "timeout"])
def test_powershell_check_matches(tmp_path, realkey, scenario):
    """pwsh off Windows: the platform-neutral lines are the same as mace gui connect's."""
    flags, extra = SCENARIOS[scenario]
    env = _script_env(tmp_path, realkey, **extra)
    argv = ["--gateway", "hpcc", "--node", "dev-node-01", "--check", *flags]
    py = subprocess.run([sys.executable, str(REPO / "mace_cli"), "--no-banner", "gui", "connect", *argv], env=env,
                        capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL)
    psflags = {"--no-net": "-NoNet", "--fix-config": "-FixConfig", "--hop": "-Hop"}
    ps = subprocess.run([_pwsh(), "-NoProfile", "-File", str(PS1), "-Gateway", "hpcc", "-Node", "dev-node-01",
                         "-Check", *[psflags.get(f, f) for f in flags]], env=env, capture_output=True, text=True,
                        timeout=120)
    assert ps.returncode == py.returncode, ps.stdout + ps.stderr
    shared = [ln for ln in py.stdout.splitlines() if ln[:5] in ("OK   ", "WARN ", "FAIL ", "SKIP ", "INFO ")
              and "agent" not in ln and "permissions" not in ln and "connection sharing" not in ln]
    got = ps.stdout.splitlines()
    for ln in shared:
        assert ln.replace("--fix-config", "-FixConfig") in got, ln
    verdict = [ln for ln in py.stdout.splitlines() if ln.startswith("Verdict")][0].split(" Connect with:")[0]
    assert any(ln.startswith(verdict) for ln in got), ps.stdout


# ---- the scripts' connect UX: the same messages as mace gui connect

def _stub(tmp_path, body):
    fb = tmp_path / "stubbin"
    fb.mkdir()
    (fb / "ssh").write_text("#!/bin/sh\n[ \"$1\" = -G ] && exit 0\n" + body)
    (fb / "ssh").chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON") and k != "BROWSER"}
    env.update(PATH=f"{fb}:/usr/bin:/bin", HOME=str(tmp_path / "h"), XDG_CONFIG_HOME=str(tmp_path / "cfg"))
    (tmp_path / "h").mkdir(exist_ok=True)
    return env


def _both(env, *flags):
    py = subprocess.run([sys.executable, str(REPO / "mace_cli"), "--no-banner", "gui", "connect", "--no-open", *flags],
                        env=env, capture_output=True, text=True, timeout=90, stdin=subprocess.DEVNULL)
    sh = subprocess.run(["bash", str(SH), "--no-open", *flags], env=env, capture_output=True, text=True, timeout=90,
                        stdin=subprocess.DEVNULL)
    return py, sh


def test_scripts_and_connect_say_cannot_reach_host(tmp_path):
    env = _stub(tmp_path, "echo 'ssh: connect to host gw.example.edu port 22: Connection timed out' >&2\nexit 255\n")
    py, sh = _both(env, "--gateway", "me@gw.example.edu", "--node", "n1")
    want = "cannot reach host gw.example.edu (timeout): check the name, and your network or VPN."
    assert f"mace gui connect: {want}" in py.stdout, py.stdout
    assert "mace gui connect --check (with this profile or these flags) tests each step" in py.stdout
    assert f"mace-gui-connect: {want}" in sh.stderr and "--check (with the same profile or flags)" in sh.stderr
    assert py.returncode == sh.returncode == 1


def test_scripts_and_connect_say_which_mace_they_used(tmp_path):
    env = _stub(tmp_path, "echo 'MACE_GUI_USING /home/me/MACE/mace_cli'\nexit 1\n")
    py, sh = _both(env, "--gateway", "me@gw", "--node", "n1")
    assert "mace is not on n1's PATH there; using /home/me/MACE/mace_cli." in py.stdout
    assert "mace is not on n1's PATH there; using /home/me/MACE/mace_cli." in sh.stdout


def test_unresolvable_gateway_warns_and_still_connects(tmp_path):
    env = _stub(tmp_path, "echo \"$@\" >> \"$HOME/ran\"\nexit 255\n")
    py, sh = _both(env, "--gateway", "me@no-such-host.invalid", "--node", "n1")
    want = ("no-such-host.invalid matches no Host in ~/.ssh/config and does not resolve (DNS): check the gateway's "
            "name (your cluster's login host, e.g. login.example.edu). Connecting anyway.")
    assert want in py.stdout and want in sh.stderr
    assert (tmp_path / "h" / "ran").read_text().count("no-such-host.invalid") >= 2      # both still tried


# ---- the probes pass the profile's ssh_options where the connection does (1.1.6 review)
OPT = "-oIdentityFile=~/.ssh/k"


def _gateway_probe(fake):
    return [c for c in fake.ran("gateway") if c[-2:] == ["hpcc", "true"]][0]


def test_nested_profile_probes_carry_the_ssh_options(home):
    fake = Fake()
    code, out, _ = _check(home, fake, prof={**PROF, "hop": "nested", "ssh_options": [OPT]})
    assert OPT in _gateway_probe(fake)                 # connect's nested hop gives them to the gateway's ssh
    assert OPT in fake.ran("nested")[0]
    jump = [c for c in fake.ran("jump") if c[-2] != "hpcc"][0]
    assert OPT in jump
    assert code == 0


def test_jump_profile_probes_the_gateway_as_the_jump_hop_reaches_it(home):
    """-J's gateway leg gets no command-line options; the nested probe, as the nested hop, does."""
    fake = Fake()
    _check(home, fake, prof={**PROF, "hop": "jump", "ssh_options": [OPT]})
    assert OPT not in _gateway_probe(fake)
    assert OPT in fake.ran("nested")[0]


def _ran(tmp_path):
    return [ln for ln in (tmp_path / "ran").read_text().splitlines()] if (tmp_path / "ran").exists() else []


def test_bash_check_probes_carry_the_ssh_options(tmp_path, realkey):
    env = _script_env(tmp_path, realkey)
    r = subprocess.run(["bash", str(SH), "--gateway", "hpcc", "--node", "dev-node-01", "--hop", "nested",
                        f"--ssh-option={OPT}", "--check"], env=env, capture_output=True, text=True, timeout=120,
                       stdin=subprocess.DEVNULL)
    assert "Verdict" in r.stdout, r.stdout + r.stderr
    ran = _ran(tmp_path)
    for kind in ("GATEWAY", "JUMP", "NESTED"):
        assert any(ln.startswith(kind + " ") and OPT in ln for ln in ran), (kind, ran)


@pytest.mark.skipif(_pwsh() is None, reason="needs pwsh")
def test_powershell_check_probes_carry_the_ssh_options(tmp_path, realkey):
    env = _script_env(tmp_path, realkey)
    r = subprocess.run([_pwsh(), "-NoProfile", "-File", str(PS1), "-Gateway", "hpcc", "-Node", "dev-node-01", "-Hop",
                        "nested", "-SshOption", OPT, "-Check"], env=env, capture_output=True, text=True, timeout=120)
    assert "Verdict" in r.stdout, r.stdout + r.stderr
    ran = _ran(tmp_path)
    for kind in ("GATEWAY", "JUMP", "NESTED"):
        assert any(ln.startswith(kind + " ") and OPT in ln for ln in ran), (kind, ran)


def test_agent_fix_names_the_same_key_as_the_other_fixes(home):
    """With no key file named for the gateway, every fix line names ~/.ssh/id_ed25519_cluster."""
    fake = Fake(g={"hpcc": "user netid\nhostname login.example.edu\nport 22\ncontrolmaster false\n"},
                agent=(1, "The agent has no identities.\n", ""))
    code, out, _ = _check(home, fake, no_net=True)
    assert "fix: ssh-add ~/.ssh/id_ed25519_cluster" in out, out
    assert "ssh-add ~/.ssh/id_ed25519\n" not in out + "\n"


def test_scripts_name_the_same_fallback_key():
    for text in (SH.read_text(), PS1.read_text()):
        assert '"~/.ssh/id_ed25519" }' not in text and "key:-~/.ssh/id_ed25519}" not in text


# ---- Windows: a disabled ssh-agent service is a WARN when ssh does not need it; no ssh client exits 2

class FakeKeys(Fake):
    """ssh-keygen -y succeeds only for a key without a passphrase."""

    def __init__(self, plain, **kw):
        super().__init__(**kw)
        self.plain = plain

    def __call__(self, argv, **kw):
        if argv[:2] == ["ssh-keygen", "-y"]:
            self.calls.append(list(argv))
            return CP(argv, 0 if self.plain else 1, "ssh-ed25519 AAAA\n" if self.plain else "", "" if self.plain
                      else "Load key: incorrect passphrase supplied to decrypt private key\n")
        return super().__call__(argv, **kw)


def _win(home, fake, **kw):
    return _check(home, fake, system="windows", env={}, **kw)


def test_disabled_agent_is_a_warning_for_a_key_without_a_passphrase(home):
    code, out, _ = _win(home, FakeKeys(True, ps="Stopped Disabled"))
    assert "WARN  the OpenSSH Authentication Agent service (ssh-agent) is disabled" in out
    assert "FAIL" not in out and code == 0


def test_disabled_agent_with_a_passphrase_key_is_a_warning_when_the_logins_work(home):
    code, out, _ = _win(home, FakeKeys(False, ps="Stopped Disabled"))
    assert "WARN  the OpenSSH Authentication Agent service (ssh-agent) is disabled" in out
    assert "FAIL" not in out and code == 0


def test_disabled_agent_with_a_passphrase_key_fails_when_the_logins_fail(home):
    code, out, _ = _win(home, FakeKeys(False, ps="Stopped Disabled",
                                      probes={"gateway": (255, "netid@login.example.edu: Permission denied "
                                                              "(publickey).\n")}))
    assert "FAIL  the OpenSSH Authentication Agent service (ssh-agent) is disabled" in out and code == 1


def test_disabled_agent_with_a_passphrase_key_and_no_probes_stays_a_failure(home):
    code, out, _ = _win(home, FakeKeys(False, ps="Stopped Disabled"), no_net=True)
    assert "FAIL  the OpenSSH Authentication Agent service (ssh-agent) is disabled" in out and code == 1


def test_no_ssh_client_exits_2_as_a_real_connect_does():
    out = []
    ck = K.Checker(dict(PROF), "p", no_net=True, runner=Fake(), which=lambda _: None, system="linux",
                   out=out.append)
    assert ck.run_all() == 2 and any("no ssh command on PATH" in ln for ln in out)
