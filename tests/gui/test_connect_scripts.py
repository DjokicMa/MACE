"""The standalone connect scripts (scripts/mace-gui-connect.sh, .ps1): syntax,
shellcheck, profile parity with `mace gui connect` (same connect.toml, same
ssh argv for both hops), the unreadable-form note, --export-script, and the
PowerShell script's static parity with the bash one."""
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from mace.gui import connect as C
from mace.gui import connect_scripts as CS

REPO = Path(__file__).resolve().parents[2]
SH = REPO / "scripts" / "mace-gui-connect.sh"
PS1 = REPO / "scripts" / "mace-gui-connect.ps1"
pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _env(tmp_path):
    e = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    e.update(XDG_CONFIG_HOME=str(tmp_path / "cfg"), HOME=str(tmp_path / "home"))
    (tmp_path / "home").mkdir(exist_ok=True)
    return e


def test_bash_syntax_and_shellcheck():
    subprocess.run(["bash", "-n", str(SH)], check=True)
    if shutil.which("shellcheck") is None:
        pytest.skip("shellcheck is not installed")
    r = subprocess.run(["shellcheck", "-S", "warning", str(SH)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout


def test_bash_script_is_executable():
    assert os.access(SH, os.X_OK)


PROFILES = {
    "jump": {"gateway": "me@gw.example.edu", "node": "dev-node-01", "dir": "~/my runs", "hop": "jump",
             "setup": "module load Python; export X='a b'", "mace": "~/MACE/mace_cli", "local_port": 8800,
             "ssh_options": ["-o", "IdentityFile=~/.ssh/k"], "roots": ["~/proj", "/mnt/scratch/x"]},
    "nested": {"gateway": "gw", "jump": "bastion", "node": "dev-node-01", "dir": "/data/it's here",
               "hop": "nested", "conda": "mace env"},
    "nonode": {"gateway": "me@gw", "dir": "~"},
}


@pytest.mark.parametrize("name", sorted(PROFILES))
def test_profile_parity_with_mace_gui_connect(tmp_path, name):
    """A profile saved by mace gui connect is read by the bash script, whose
    ssh argv equals mace gui connect's for the same profile."""
    env = _env(tmp_path)
    data = {"defaults": {"profile": name}, "profiles": {name: PROFILES[name]}}
    path = Path(env["XDG_CONFIG_HOME"]) / "mace" / "connect.toml"
    C.save_profiles(data, path)
    # mace gui connect --dry-run, through mace_cli
    r = subprocess.run([sys.executable, str(REPO / "mace_cli"), "--no-banner", "gui", "connect", "--dry-run"],
                       env=env, capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
    assert r.returncode == 0, r.stderr
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("ssh ")]
    want = shlex.split(lines[0])
    got = subprocess.run(["bash", str(SH), "--dry-run"], env=env, capture_output=True, text=True,
                         timeout=60).stdout.splitlines()
    # the script prints one argument per line; arguments holding newlines do not occur here
    assert got == want


def test_flags_override_the_profile(tmp_path):
    env = _env(tmp_path)
    C.save_profiles({"defaults": {"profile": "p"}, "profiles": {"p": PROFILES["jump"]}},
                    Path(env["XDG_CONFIG_HOME"]) / "mace" / "connect.toml")
    got = subprocess.run(["bash", str(SH), "--dry-run", "--node", "other-node", "--local-port", "9000"], env=env,
                         capture_output=True, text=True, timeout=60).stdout.splitlines()
    # the gateway's user (me@...) goes to the node too, as mace gui connect does
    assert "me@other-node" in got and "127.0.0.1:9000:127.0.0.1:40000" in got


def test_unreadable_form_is_named_with_its_line(tmp_path):
    env = _env(tmp_path)
    conf = Path(env["XDG_CONFIG_HOME"]) / "mace" / "connect.toml"
    conf.parent.mkdir(parents=True)
    conf.write_text('[defaults]\nprofile = "p"\n\n[profiles.p]\ngateway = "gw"\nssh_options = [\n  "-v",\n]\n'
                    "node = 'literal'\n")
    r = subprocess.run(["bash", str(SH), "--dry-run"], env=env, capture_output=True, text=True, timeout=60)
    assert "connect.toml uses a form this script cannot read (line 6)" in r.stderr
    assert "(line 9)" in r.stderr
    assert r.stdout.splitlines()[-2] == "gw"


def test_export_fills_the_defaults_and_parses_back(tmp_path):
    prof = PROFILES["jump"]
    out = tmp_path / "c.sh"
    env = _env(tmp_path)
    r = subprocess.run([sys.executable, str(REPO / "mace_cli"), "--no-banner", "gui", "connect", "--export-script",
                        "bash", "-o", str(out), "--gateway", prof["gateway"], "--node", prof["node"], "--dir",
                        prof["dir"], "--setup", prof["setup"], "--mace", prof["mace"], "--local-port", "8800",
                        "--ssh-option=-o", "--ssh-option=IdentityFile=~/.ssh/k", "--root", "~/proj", "--root",
                        "/mnt/scratch/x", "--hop", "jump"],
                       env=env, capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
    assert r.returncode == 0, r.stderr
    assert os.access(out, os.X_OK)
    # no profile file at all: the exported defaults alone give the same argv
    got = subprocess.run(["bash", str(out), "--dry-run"], env=env, capture_output=True, text=True,
                         timeout=60).stdout.splitlines()
    want = C.build_ssh_argv({**prof, "local_port": 8800}, 8800, 40000, 40001, "jump", user_for_node="me")
    assert got == want
    ps = CS.export_script("powershell", prof)
    assert "$D_GATEWAY = 'me@gw.example.edu'" in ps and "$D_ROOTS = @('~/proj', '/mnt/scratch/x')" in ps
    assert "$D_LOCAL_PORT = 8800" in ps


def test_powershell_static_parity():
    ps, sh = PS1.read_text(), SH.read_text()
    params = set(re.findall(r"\[(?:string|int|string\[\]|switch)\]\$(\w+)", ps))
    assert {"Gateway", "Jump", "Node", "Hop", "Dir", "Mace", "Setup", "Conda", "LocalPort", "SshOption", "Root",
            "NoOpen", "DryRun", "Manual"} <= params
    sh_flags = set(re.findall(r"^\s+--([a-z-]+)\)", sh, re.M))
    for f in ("gateway", "jump", "node", "hop", "dir", "mace", "setup", "conda", "local-port", "ssh-option", "root",
              "no-open", "dry-run"):
        assert f in sh_flags, f
        assert f.replace("-", "") in {p.lower() for p in params}, f
    keys = lambda t: re.findall(r"\$?D_([A-Z_]+)\s*=", t.split(CS.BEGIN)[1].split(CS.END)[0])
    assert keys(ps) == keys(sh) == [k.upper() for k in CS.FIELDS]
    for word in ("MACE_GUI_READY", "MACE_GUI_LOGIN", "MACE_GUI_STOPPED", "MACE_GUI_ERROR", '"ping"', '"detach"',
                 "--managed", "ExitOnForwardFailure=yes", "BatchMode=yes", "Unblock-File"):
        assert word.strip('"') in ps, word
    for key in ("gateway", "jump", "node", "hop", "dir", "mace", "setup", "conda", "local_port", "ssh_options",
                "roots", "open"):
        assert f'"{key}"' in ps or f"{key})" in sh, key
    if shutil.which("pwsh"):
        r = subprocess.run(["pwsh", "-NoProfile", "-Command",
                            f"$null = [System.Management.Automation.Language.Parser]::ParseFile('{PS1}', [ref]$null, "
                            "[ref]$e); if ($e) { $e; exit 1 }"], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout


# the key-failure hint and the email-like gateway warning: the same words as mace gui connect
HINT_CORE = ('If the key has a passphrase, ssh cannot ask for it here: load it into an agent first, with '
             'eval "$(ssh-agent)" and ssh-add (on Windows: start the OpenSSH Authentication Agent service, then '
             'run ssh-add)')
HINT_TAIL = "A Host alias from ~/.ssh/config also works as the gateway."
WARN_PARTS = ("looks like an email address. It should be user@login-host (your cluster's login node, e.g. "
              "netid@login.example.edu) or a Host alias from ~/.ssh/config. Connecting anyway; to fix the profile in ",
              "--gateway <user@login-host> --save")


def test_key_hint_and_email_warning_parity():
    ps, sh = PS1.read_text(), SH.read_text()
    sh_hint = sh.replace('\\"\\$(ssh-agent)\\"', '"$(ssh-agent)"')    # the bash string's escapes
    for text in (C.KEY_HINT, sh_hint, ps):
        assert HINT_CORE in text and HINT_TAIL in text
    py_warn = C.email_gateway_warning("netid@university.edu", Path("connect.toml"), "p")
    for text in (py_warn, sh, ps):
        for part in WARN_PARTS:
            assert part in text, part
    # the same ssh messages and the same email heuristic
    for pat in C.KEY_FAILURE_RE.pattern.split("|"):
        assert pat in sh.replace("\\\\(", "\\("), pat
    assert "'^[a-z0-9-]+\\.[a-z]{2,}$'" in ps and "'^[a-z0-9-]+[.][a-z]{2,}$'" in sh
    assert r"[a-z0-9-]+\.[a-z]{2,}" in Path(C.__file__).read_text()
    assert "if ($code -eq 255)" in ps            # PowerShell leaves ssh's stderr to the console: the exit code only


def _stub_ssh(tmp_path, stderr_lines):
    """An "ssh" that logs its arguments, prints ssh's message and exits 255."""
    fakebin = tmp_path / "fakebin"
    fakebin.mkdir()
    ran = tmp_path / "ran"
    ssh = fakebin / "ssh"
    ssh.write_text("#!/bin/sh\n[ \"$1\" = -G ] && exit 0\necho \"$@\" >> %s\nprintf '%%s\\n' %s >&2\nsleep 0.2\n"
                   "exit 255\n" % (shlex.quote(str(ran)), " ".join(shlex.quote(x) for x in stderr_lines)))
    ssh.chmod(0o755)
    return fakebin, ran


@pytest.mark.parametrize("gateway,stderr_lines,hint,warn", [
    ("netid@login.example.edu", ["netid@login.example.edu: Permission denied (publickey)."], True, False),
    ("netid@login.example.edu", ["Enter passphrase for key '/home/netid/.ssh/id_ed25519': ",
                                 "netid@login.example.edu: Permission denied (publickey,password)."], True, False),
    ("netid@university.edu", ["ssh: connect to host university.edu port 22: Connection timed out"], False, True),
    ("netid@university.edu", ["netid@university.edu: Permission denied (publickey)."], True, True),
])
def test_bash_script_key_hint_and_email_warning(tmp_path, gateway, stderr_lines, hint, warn):
    if shutil.which("mkfifo") is None:
        pytest.skip("needs mkfifo")
    fakebin, ran = _stub_ssh(tmp_path, stderr_lines)
    env = _env(tmp_path)
    env["PATH"] = f"{fakebin}:{env.get('PATH', '/usr/bin:/bin')}"
    env.pop("BROWSER", None)
    r = subprocess.run(["bash", str(SH), "--gateway", gateway, "--no-open"], env=env, capture_output=True,
                       text=True, timeout=60, stdin=subprocess.DEVNULL)
    assert r.returncode == 1
    assert gateway in ran.read_text()            # the warning does not stop the connection
    assert (HINT_CORE in r.stderr.replace("\n", " ")) is hint, r.stderr
    assert (WARN_PARTS[0] in r.stderr) is warn, r.stderr
    if warn:
        assert f"to fix the profile in {env['XDG_CONFIG_HOME']}/mace/connect.toml" in r.stderr


@pytest.mark.parametrize("config,warn", [("Host hpcc *.EDU\n  HostName login.university.edu\n", False),
                                         ("Host=university.edu\n", False),
                                         ("Host hpcc !university.edu\nHostName university.edu\n", True)])
def test_bash_script_email_warning_respects_ssh_config(tmp_path, config, warn):
    env = _env(tmp_path)
    (tmp_path / "home" / ".ssh").mkdir()
    (tmp_path / "home" / ".ssh" / "config").write_text(config)
    r = subprocess.run(["bash", str(SH), "p", "--gateway", "NetID@University.edu", "--dry-run"], env=env,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0
    assert (WARN_PARTS[0] in r.stderr) is warn, r.stderr
    assert C.gateway_looks_like_email("NetID@University.edu", C.ssh_config_hosts(
        tmp_path / "home" / ".ssh" / "config")) is warn
    if warn:
        assert "mace gui connect p --gateway <user@login-host> --save" in r.stderr
