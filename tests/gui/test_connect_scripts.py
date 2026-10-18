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


def _argv_lines(stdout):
    """The scripts' --dry-run: one argument per line, after comment lines (# hop ...)."""
    lines = stdout.split("\n")        # not splitlines(): a value may hold \r or \f
    if lines and lines[-1] == "":
        lines.pop()
    return [ln for ln in lines if not ln.startswith("# ")]


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
    got = _argv_lines(subprocess.run(["bash", str(SH), "--dry-run"], env=env, capture_output=True, text=True,
                         timeout=60).stdout)
    # the script prints one argument per line; arguments holding newlines do not occur here
    assert got == want


def test_flags_override_the_profile(tmp_path):
    env = _env(tmp_path)
    C.save_profiles({"defaults": {"profile": "p"}, "profiles": {"p": PROFILES["jump"]}},
                    Path(env["XDG_CONFIG_HOME"]) / "mace" / "connect.toml")
    got = _argv_lines(subprocess.run(["bash", str(SH), "--dry-run", "--node", "other-node", "--local-port", "9000"], env=env,
                         capture_output=True, text=True, timeout=60).stdout)
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
    got = _argv_lines(subprocess.run(["bash", str(out), "--dry-run"], env=env, capture_output=True, text=True,
                         timeout=60).stdout)
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
                            "$t = $null; $e = $null; "      # [ref] needs variables that exist
                            f"$null = [System.Management.Automation.Language.Parser]::ParseFile('{PS1}', [ref]$t, "
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


# ---- a ~ that the local shell expanded (--mace ~/x arrives as /home/you/x): the same rewrite and note

@pytest.mark.parametrize("hop", ["jump", "nested"])
def test_local_home_paths_rewrite_parity(tmp_path, hop):
    env = _env(tmp_path)
    home = env["HOME"]
    flags = ["--gateway", "me@gw.example.edu", "--node", "dev-node-01", "--hop", hop, "--mace",
             f"python3 {home}/mace-1.1.5-test/mace/mace_cli", "--dir", f"{home}/my runs", "--root", home,
             "--root", "/scratch/x", "--conda", f"{home}/envs/m", "--dry-run"]
    r = subprocess.run([sys.executable, str(REPO / "mace_cli"), "--no-banner", "gui", "connect", *flags],
                       env=env, capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
    assert r.returncode == 0, r.stderr
    want = shlex.split([ln for ln in r.stdout.splitlines() if ln.startswith("ssh ")][0])
    s = subprocess.run(["bash", str(SH), *flags], env=env, capture_output=True, text=True, timeout=60)
    assert s.returncode == 0, s.stderr
    assert _argv_lines(s.stdout) == want
    assert home not in "\n".join(want)
    note = lambda t: [ln for ln in t.splitlines() if ln.startswith("Note: treated")]
    assert note(s.stderr) == note(r.stderr) and len(note(r.stderr)) == 1
    assert f"{home}/mace-1.1.5-test/mace/mace_cli as ~/mace-1.1.5-test/mace/mace_cli" in note(r.stderr)[0]


@pytest.mark.parametrize("hop", ["jump", "nested"])
def test_double_slash_literal_path_parity(tmp_path, hop):
    """A leading // keeps a path under this computer's home literal (it exists at
    the same absolute path on the node): the bash script sends what mace gui
    connect sends, neither rewrites it to ~/... nor prints the note."""
    env = _env(tmp_path)
    home = env["HOME"]
    flags = ["--gateway", "me@gw.example.edu", "--node", "dev-node-01", "--hop", hop, "--mace",
             f"python3 //{home}/m/mace_cli", "--dir", f"//{home}/my runs", "--root", f"//{home}",
             "--root", f"{home}/r", "--conda", f"//{home}/envs/m", "--dry-run"]
    r = subprocess.run([sys.executable, str(REPO / "mace_cli"), "--no-banner", "gui", "connect", *flags],
                       env=env, capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
    assert r.returncode == 0, r.stderr
    want = shlex.split([ln for ln in r.stdout.splitlines() if ln.startswith("ssh ")][0])
    s = subprocess.run(["bash", str(SH), *flags], env=env, capture_output=True, text=True, timeout=60)
    assert s.returncode == 0, s.stderr
    assert _argv_lines(s.stdout) == want
    text = "\n".join(want)
    assert f"{home}/m/mace_cli" in text and f"{home}/my runs" in text and f"{home}/envs/m" in text
    assert "//" + home not in text and '"$HOME"/r' in text
    note = lambda t: [ln for ln in t.splitlines() if ln.startswith("Note: treated")]
    assert note(s.stderr) == note(r.stderr) == [f"Note: treated {home}/r as ~/r on the remote (your shell turned ~ "
                                                "into this computer's home folder; quote the ~, as '~/r', to avoid "
                                                "this)."]


def test_powershell_keeps_double_slash_paths_literal_like_bash():
    ps, sh = PS1.read_text(), SH.read_text()
    assert 'if ($s.StartsWith("//")) { return $s }' in ps                       # Localize-Home
    assert 'if ($s.StartsWith("//")) { return ShQuote $s.Substring(1) }' in ps  # HomePath
    assert '"//"*) return 0 ;;' in sh and '"//"*) shq "${1#/}" ;;' in sh


def test_bash_script_mace_not_found_names_the_rewritten_path(tmp_path):
    if shutil.which("mkfifo") is None:
        pytest.skip("needs mkfifo")
    fakebin = tmp_path / "fakebin"
    fakebin.mkdir()
    (fakebin / "ssh").write_text("#!/bin/sh\n[ \"$1\" = -G ] && exit 0\necho 'mace_cli: not found' >&2\nexit 127\n")
    (fakebin / "ssh").chmod(0o755)
    env = _env(tmp_path)
    env["PATH"] = f"{fakebin}:{env.get('PATH', '/usr/bin:/bin')}"
    home = env["HOME"]
    r = subprocess.run(["bash", str(SH), "--gateway", "me@gw", "--node", "n1", "--no-open", "--mace",
                        f"{home}/m/mace_cli"], env=env, capture_output=True, text=True, timeout=60,
                       stdin=subprocess.DEVNULL)
    assert r.returncode == 127
    assert (f"{home}/m/mace_cli as ~/m/mace_cli came from this computer's home folder (your shell expands an "
            "unquoted ~ here): check that it exists on n1. Quote the ~ ('~/...')") in r.stderr


def test_powershell_rewrites_local_home_paths_like_bash():
    ps, sh = PS1.read_text(), SH.read_text()
    py = C.localized_note([("/home/you/a", "~/a")])
    core = "on the remote (your shell turned ~ into this computer's home folder; quote the ~, as '"
    for text in (ps, sh, py):
        assert core in text and "to avoid this)." in text
    for name in ("$DirV = Localize-Home $DirV", "$CondaV = Localize-Home $CondaV", "Localize-Home $_ })",
                 '"conda activate " + (HomePath $CondaV)', "if ($code -eq 127)"):
        assert name in ps, name
    assert 'conda activate $(home_path "$CONDA")' in sh
    tail = "came from this computer's home folder (your shell expands an unquoted ~ here): check that it exists on"
    assert tail in ps and tail in sh


def _stub_ssh_g(tmp_path):
    """An "ssh" whose -G gives the gateway (gw...) its own key file, as a Host block would."""
    fakebin = tmp_path / "fakebin"
    fakebin.mkdir()
    (fakebin / "ssh").write_text('#!/bin/sh\n[ "$1" = -G ] || exit 255\ncase "$2" in\n'
                                 '  *gw*) printf "user netid\\nidentityfile ~/.ssh/id_cluster\\n" ;;\n'
                                 '  *) printf "user netid\\nidentityfile ~/.ssh/id_ed25519\\n" ;;\nesac\n')
    (fakebin / "ssh").chmod(0o755)
    return fakebin


@pytest.mark.parametrize("hop", ["jump", "nested"])
def test_node_key_options_parity(tmp_path, hop):
    """The maintainer's case: gateway netid@gw with an IdentityFile in its Host block; the node hop of ssh -J
    gets it too, in mace gui connect and the bash script alike."""
    env = _env(tmp_path)
    (Path(env["HOME"]) / ".ssh").mkdir()
    (Path(env["HOME"]) / ".ssh" / "id_cluster").write_text("k")
    env["PATH"] = f"{_stub_ssh_g(tmp_path)}:{env.get('PATH', '/usr/bin:/bin')}"
    flags = ["--gateway", "netid@gw.example.edu", "--node", "dev-node", "--hop", hop, "--dry-run"]
    r = subprocess.run([sys.executable, str(REPO / "mace_cli"), "--no-banner", "gui", "connect", *flags],
                       env=env, capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
    assert r.returncode == 0, r.stderr
    want = shlex.split([ln for ln in r.stdout.splitlines() if ln.startswith("ssh ")][0])
    got = _argv_lines(subprocess.run(["bash", str(SH), *flags], env=env, capture_output=True, text=True,
                         timeout=60).stdout)
    assert got == want
    assert ("IdentityFile=~/.ssh/id_cluster" in want) is (hop == "jump")
    if hop == "jump":
        i = want.index("IdentityFile=~/.ssh/id_cluster")
        assert want[i - 1] == "-o" and want[i + 1:i + 3] == ["-L", "127.0.0.1:8765:127.0.0.1:40000"]
        assert want[-2] == "netid@dev-node"


def test_powershell_node_keys_like_bash():
    ps, sh = PS1.read_text(), SH.read_text()
    for word in ("IdentityFile=", "IdentityAgent=", "IdentitiesOnly=yes",
                 "mace-gui-connect-the-node-asks-for-a-password"):
        assert word in ps and word in sh, word
    assert "$keys = @(Node-Keys $target)" in ps and 'node_keys "$target"' in sh


# ---- the Windows 11 run (1.1.5 final): TOML escapes, PowerShell 5.1 and its
# stderr, the profile file under MobaXterm / Git Bash, missing folders, host keys

PWSH = shutil.which("pwsh")
needs_pwsh = pytest.mark.skipif(PWSH is None, reason="needs pwsh (PowerShell 7) to run the .ps1")

# saved by mace gui connect (its own escapes), and written by hand (every basic-string escape tomllib reads)
TRICKY = {"gateway": "me@gw.example.edu", "node": "dev-node-01", "hop": "jump",
          "dir": "C:\\Users\\me\\tmp\\k \"q\" \t tab caf\u00e9 \U0001F600 \u0001",
          "roots": ["a\\tb", "x\\\\y", "\\n"], "setup": "echo '\\t' \"$X\""}
HAND = ('[defaults]\nprofile = "h"\n\n[profiles.h]\ngateway = "me@gw.example.edu"\nnode = "dev-node-01"\n'
        'hop = "nested"\n'
        'dir = "C:\\\\Users\\\\me\\\\tmp\\\\k\\t\\"\\b\\f\\r|\\u00e9|\\U0001F600|\\u0041\\\\u0041"\n'
        'roots = ["\\\\\\\\server\\\\share", "a\\"b", "\\u00e9\\t"]\n')


def _want(tmp_path, conf_text):
    import tomllib
    data = tomllib.loads(conf_text)
    prof = data["profiles"][data["defaults"]["profile"]]
    unode = C.effective_node_user(prof["gateway"])
    return C.build_ssh_argv(prof, int(prof.get("local_port") or 8765), 40000, 40001, prof.get("hop", "jump"), unode,
                            node_keys=C.node_key_options(prof["gateway"], C.node_target(prof, unode))
                            if prof.get("hop", "jump") == "jump" else [])


def _toml_cases(tmp_path):
    saved = C.dump_profiles({"defaults": {"profile": "t"}, "profiles": {"t": TRICKY}})
    return [("saved", saved), ("hand", HAND)]


@pytest.mark.parametrize("case", ["saved", "hand"])
def test_toml_escapes_round_trip_in_the_bash_script(tmp_path, case):
    """\\\\ then t is a backslash and a t (C:\\\\Users\\\\me\\\\tmp stays C:\\Users\\me\\tmp), as tomllib reads it."""
    text = dict(_toml_cases(tmp_path))[case]
    env = _env(tmp_path)
    conf = Path(env["XDG_CONFIG_HOME"]) / "mace" / "connect.toml"
    conf.parent.mkdir(parents=True)
    conf.write_text(text, encoding="utf-8")
    want = _want(tmp_path, text)
    r = subprocess.run(["bash", str(SH), "--dry-run"], env=env, capture_output=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert "".join(a + "\n" for a in _argv_lines(r.stdout.decode("utf-8"))) == "".join(a + "\n" for a in want)
    assert "\t" not in want[-2] and ("C:\\Users\\me\\tmp\\k" in want[-1])


def _ps_env(tmp_path):
    env = _env(tmp_path)
    env["APPDATA"] = str(tmp_path / "appdata")
    env.pop("USERNAME", None)
    return env


def _run_ps(env, *args):
    return subprocess.run([PWSH, "-NoProfile", "-NonInteractive", "-File", str(PS1), *args], env=env,
                          capture_output=True, timeout=120, stdin=subprocess.DEVNULL)


def _ps_dry(stdout):
    """(one argument per line, the PowerShell line) of the .ps1's -DryRun output."""
    text = stdout.decode("utf-8").replace("\r\n", "\n")
    args, marker, rest = text.partition("# PowerShell (paste as is; not for cmd.exe):\n")
    assert marker, text
    args = "".join(ln + "\n" for ln in _argv_lines(args))       # without the "# hop ..." label
    return args, rest.rstrip("\n")


@needs_pwsh
@pytest.mark.parametrize("case", ["saved", "hand"])
def test_toml_escapes_round_trip_in_the_powershell_script(tmp_path, case):
    from test_connect_windows import _ms_argv
    text = dict(_toml_cases(tmp_path))[case]
    env = _ps_env(tmp_path)
    conf = Path(env["APPDATA"]) / "mace" / "connect.toml"
    conf.parent.mkdir(parents=True)
    conf.write_text(text, encoding="utf-8")
    want = _want(tmp_path, text)
    r = _run_ps(env, "-DryRun")
    assert r.returncode == 0, r.stderr.decode()
    args, ps_line = _ps_dry(r.stdout)
    assert args == "".join(a + "\n" for a in want[1:])
    assert ps_line.startswith("ssh --% ") and _ms_argv(ps_line[len("ssh --% "):]) == want[1:]


@needs_pwsh
@pytest.mark.parametrize("name", sorted(PROFILES))
def test_powershell_profile_parity_with_mace_gui_connect(tmp_path, name):
    """The .ps1 reads the profile and builds mace gui connect's argv (run by pwsh)."""
    env = _ps_env(tmp_path)
    data = {"defaults": {"profile": name}, "profiles": {name: PROFILES[name]}}
    C.save_profiles(data, Path(env["XDG_CONFIG_HOME"]) / "mace" / "connect.toml")
    C.save_profiles(data, Path(env["APPDATA"]) / "mace" / "connect.toml")
    r = subprocess.run([sys.executable, str(REPO / "mace_cli"), "--no-banner", "gui", "connect", "--dry-run"],
                       env=env, capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
    want = shlex.split([ln for ln in r.stdout.splitlines() if ln.startswith("ssh ")][0])
    p = _run_ps(env, "-DryRun")
    assert p.returncode == 0, p.stderr.decode()
    assert _ps_dry(p.stdout)[0].splitlines() == want[1:]


def _ps_code_lines():
    for n, line in enumerate(PS1.read_text().splitlines(), 1):
        code = line.split(" #", 1)[0] if not line.lstrip().startswith("#") else ""
        yield n, code


def test_powershell_never_redirects_a_native_commands_stderr():
    """Windows PowerShell 5.1: with $ErrorActionPreference = "Stop", a native
    command's stderr line redirected by 2>&1 (or 2>$null) becomes a
    NativeCommandError that ends the script (about_Redirection; only
    PowerShell 7.2+ changed that). ssh -V writes its version to stderr, so
    every real run failed. Native programs go through Invoke-Native (a
    Process that reads both pipes itself) instead."""
    text = PS1.read_text()
    assert '$ErrorActionPreference = "Stop"' in text
    for n, code in _ps_code_lines():
        assert not re.search(r"\d?>&\d|\b2>|\*>", code), (n, code)        # no stream redirection at all
        assert not re.search(r"&\s*['\"]?ssh", code), (n, code)           # no "& ssh ..." call
        assert "Write-Error" not in code, (n, code)                        # ends the script under Stop
    assert 'Invoke-Native "ssh" @("-V")' in text and text.count('Invoke-Native "ssh" @("-G"') == 2
    assert "$psi.RedirectStandardError = $true" in text.split("function Invoke-Native")[1].split("\n}\n")[0]
    # the session itself leaves ssh's stderr on the console (its prompts must show), outside PowerShell's streams
    assert "$psi.RedirectStandardError = $false" in text
    # Start-Process (the browser) can throw too: never unguarded under Stop
    for n, code in _ps_code_lines():
        if "Start-Process" in code:
            assert "try {" in code, (n, code)


def test_powershell_unquote_is_one_pass():
    text = PS1.read_text()
    body = text.split("function Unquote-Basic")[1].split("\nfunction ")[0]
    assert "[regex]::Replace(" in body and "-replace" not in body
    sh_body = SH.read_text().split("unbasic() {")[1].split("\n}\n")[0]
    assert "sed" not in sh_body


@needs_pwsh
def test_powershell_parses():
    r = subprocess.run([PWSH, "-NoProfile", "-Command",
                        "$t = $null; $e = $null; "
                        f"$null = [System.Management.Automation.Language.Parser]::ParseFile('{PS1}', [ref]$t, "
                        "[ref]$e); if ($e) { $e; exit 1 }"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout


# PowerShell ends a single-quoted string at ' and also at the curly quotes U+2018-U+201B; Windows
# PowerShell 5.1 reads a BOM-less .ps1 as ANSI, where the UTF-8 byte 0x91 of Ñ (C3 91) is ‘
QUOTES = ["it's", "O’Brien", "‘a’ ‚b‛", "x’; Remove-Item -Recurse C:\\ #", "/home/PEÑA/proj", "tab\there",
          "line\nnext", "€ and \U0001F600", "’"]


@pytest.mark.parametrize("value", QUOTES)
def test_powershell_export_quotes_every_single_quote_character(value):
    lit = CS._ps_str(value)
    assert lit.isascii(), lit
    # outside the [char] codes only '...' runs with every quote doubled: nothing can end a string early
    for run in re.findall(r"'((?:[^']|'')*)'", lit):
        assert not set(run.replace("''", "")) & set("'\u2018\u2019\u201a\u201b"), lit
    block = "\n".join(CS.defaults_block("powershell", {"dir": value, "roots": [value, "b"]}))
    assert block.isascii()


@needs_pwsh
def test_powershell_export_values_read_back_exactly(tmp_path):
    block = "\n".join(CS.defaults_block("powershell", {"dir": QUOTES[3], "setup": QUOTES[4], "roots": QUOTES}))
    script = tmp_path / "v.ps1"
    # CRLF and no BOM, as mace gui connect --export-script powershell writes it
    script.write_text(block + "\n[Console]::OutputEncoding = [Text.Encoding]::UTF8\n"
                      "ConvertTo-Json -Compress @($D_DIR, $D_SETUP, $D_ROOTS)\n", encoding="utf-8", newline="\r\n")
    r = subprocess.run([PWSH, "-NoProfile", "-NonInteractive", "-File", str(script)], capture_output=True,
                       timeout=120, stdin=subprocess.DEVNULL)
    assert r.returncode == 0, r.stderr.decode()
    import json
    assert json.loads(r.stdout.decode("utf-8")) == [QUOTES[3], QUOTES[4], QUOTES]


def _fake_windows_bash(tmp_path, uname, with_cygpath):
    fakebin = tmp_path / "fakebin"
    fakebin.mkdir(exist_ok=True)
    (fakebin / "uname").write_text(f"#!/bin/sh\necho {uname}\n")
    (fakebin / "uname").chmod(0o755)
    if with_cygpath:
        (fakebin / "cygpath").write_text("#!/bin/sh\n[ \"$1\" = -u ] && echo %s\n" % shlex.quote(str(tmp_path / "win")))
        (fakebin / "cygpath").chmod(0o755)
    env = _env(tmp_path)
    env["PATH"] = f"{fakebin}:{env.get('PATH', '/usr/bin:/bin')}"
    env["APPDATA"] = "C:\\Users\\me\\AppData\\Roaming"
    return env


@pytest.mark.parametrize("uname", ["CYGWIN_NT-10.0-22631", "MSYS_NT-10.0-22631", "MINGW64_NT-10.0-22631"])
def test_bash_script_reads_the_windows_profile_under_mobaxterm_and_git_bash(tmp_path, uname):
    env = _fake_windows_bash(tmp_path, uname, with_cygpath=True)
    win = tmp_path / "win" / "mace" / "connect.toml"
    C.save_profiles({"defaults": {"profile": "w"}, "profiles": {"w": {"gateway": "me@gw", "node": "n1"}}}, win)
    r = subprocess.run(["bash", str(SH), "--dry-run"], env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines()[-2] == "me@n1"
    assert f"reading the profiles of Windows' mace gui connect in {win}" in r.stderr
    # the XDG file wins when it exists
    C.save_profiles({"defaults": {"profile": "x"}, "profiles": {"x": {"gateway": "me@gw", "node": "n2"}}},
                    Path(env["XDG_CONFIG_HOME"]) / "mace" / "connect.toml")
    r = subprocess.run(["bash", str(SH), "--dry-run"], env=env, capture_output=True, text=True, timeout=60)
    assert r.stdout.splitlines()[-2] == "me@n2" and "Windows' mace gui connect" not in r.stderr


def test_bash_script_ignores_appdata_on_linux(tmp_path):
    env = _fake_windows_bash(tmp_path, "Linux", with_cygpath=True)
    C.save_profiles({"defaults": {"profile": "w"}, "profiles": {"w": {"gateway": "me@gw", "node": "n1"}}},
                    tmp_path / "win" / "mace" / "connect.toml")
    r = subprocess.run(["bash", str(SH), "--dry-run"], env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and "Windows' mace gui connect" not in r.stderr


def test_bash_winpath_without_cygpath(tmp_path):
    body = SH.read_text().split("winpath() {")[1].split("\n}\n")[0]
    r = subprocess.run(["bash", "-c", "winpath() {" + body + "\n}\nwinpath 'C:\\Users\\me\\AppData\\Roaming'"],
                       capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"})
    assert r.stdout in ("/cygdrive/c/Users/me/AppData/Roaming", "/c/Users/me/AppData/Roaming")


@pytest.mark.parametrize("hop", ["jump", "nested"])
def test_msys_paths_pass_unchanged_and_conversion_is_off(tmp_path, hop):
    """Git Bash: // and /c/ paths reach ssh as written (MSYS2_ARG_CONV_EXCL), with mace gui connect's argv."""
    env = _fake_windows_bash(tmp_path, "MINGW64_NT-10.0-22631", with_cygpath=False)
    flags = ["--gateway", "me@gw.example.edu", "--node", "dev-node-01", "--hop", hop, "--dir", "//home/me/x",
             "--root", "/c/Users/me/proj", "--dry-run"]
    r = subprocess.run([sys.executable, str(REPO / "mace_cli"), "--no-banner", "gui", "connect", *flags],
                       env=env, capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
    want = shlex.split([ln for ln in r.stdout.splitlines() if ln.startswith("ssh ")][0])
    got = subprocess.run(["bash", str(SH), *flags], env=env, capture_output=True, text=True, timeout=60)
    assert _argv_lines(got.stdout) == want
    assert "/c/Users/me/proj" in want[-1] and "/home/me/x" in want[-1]
    fakebin = tmp_path / "fakebin"
    (fakebin / "ssh").write_text("#!/bin/sh\n[ \"$1\" = -G ] && exit 0\necho \"conv=$MSYS2_ARG_CONV_EXCL\" >&2\n"
                                 "exit 255\n")
    (fakebin / "ssh").chmod(0o755)
    run = subprocess.run(["bash", str(SH), *flags[:-1], "--no-open"], env=env, capture_output=True, text=True,
                         timeout=60, stdin=subprocess.DEVNULL)
    assert "conv=*" in run.stderr


def _stub_ssh_body(tmp_path, body):
    fakebin = tmp_path / "fakebin"
    fakebin.mkdir(exist_ok=True)
    (fakebin / "ssh").write_text("#!/bin/sh\n[ \"$1\" = -G ] && exit 0\n" + body + "\n")
    (fakebin / "ssh").chmod(0o755)
    env = _env(tmp_path)
    env["PATH"] = f"{fakebin}:{env.get('PATH', '/usr/bin:/bin')}"
    env.pop("BROWSER", None)
    return env


def test_bash_script_names_a_missing_folder(tmp_path):
    env = _stub_ssh_body(tmp_path, "echo 'bash: line 1: cd: /mnt/home/me/nope: No such file or directory' >&2\n"
                                   "exit 1")
    r = subprocess.run(["bash", str(SH), "--gateway", "me@gw", "--node", "n1", "--dir", "/mnt/home/me/nope",
                        "--no-open"], env=env, capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
    assert r.returncode == 2
    assert "the folder /mnt/home/me/nope does not exist on n1" in r.stderr


@pytest.mark.parametrize("hop,cmd", [("jump", "ssh -J me@gw.example.edu dev-node-01 true"),
                                     ("nested", "ssh me@gw.example.edu true")])
def test_bash_script_host_key_failure_gives_the_trust_command(tmp_path, hop, cmd):
    env = _stub_ssh_body(tmp_path, "echo 'Host key verification failed.' >&2\nexit 255")
    r = subprocess.run(["bash", str(SH), "--gateway", "me@gw.example.edu", "--node", "me@dev-node-01", "--hop", hop,
                        "--no-open"], env=env, capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
    cmd = cmd.replace(" dev-node-01", " me@dev-node-01")
    assert r.returncode == 1 and f"    {cmd}\n" in r.stderr
    assert ("--hop nested" in r.stderr) is (hop == "jump")
    py = C.host_key_trust_command({"gateway": "me@gw.example.edu", "node": "me@dev-node-01"}, hop, None)
    assert py == cmd


def test_powershell_has_the_same_hints():
    ps = PS1.read_text()
    assert 'if ($code -eq 1)' in ps and "does not exist on" in ps
    assert "Host key verification failed" in ps and "$script:TrustCmd" in ps and "-Hop nested" in ps
    assert '[ValidateSet("", "auto", "jump", "nested")]' in ps and "$HopAuto" in ps
    assert "-Encoding UTF8" in ps


@needs_pwsh
@pytest.mark.parametrize("stderr_line,code,want", [
    ("Host key verification failed.", 255, "ssh -J me@gw.example.edu me@dev-node-01 true"),
    ("bash: line 1: cd: /mnt/home/me/nope: No such file or directory", 1, "the folder /mnt/home/me/nope does not exist"),
])
def test_powershell_real_run_reaches_its_hints(tmp_path, stderr_line, code, want):
    """The run path past -DryRun (ssh -V through Invoke-Native, the session,
    the exit-code hints), with stub ssh programs; ssh -V writes to stderr as
    the real one does."""
    fakebin = tmp_path / "fakebin"
    fakebin.mkdir()
    (fakebin / "ssh").write_text("#!/bin/sh\n[ \"$1\" = -V ] && { echo 'OpenSSH_for_Windows_9.5p2' >&2; exit 0; }\n"
                                 "exit 0\n")
    (fakebin / "ssh.exe").write_text("#!/bin/sh\necho %s >&2\nexit %d\n" % (shlex.quote(stderr_line), code))
    for f in ("ssh", "ssh.exe"):
        (fakebin / f).chmod(0o755)
    env = _ps_env(tmp_path)
    env["PATH"] = f"{fakebin}:{env.get('PATH', '/usr/bin:/bin')}"
    r = _run_ps(env, "-Gateway", "me@gw.example.edu", "-Node", "me@dev-node-01", "-Dir", "/mnt/home/me/nope",
                "-NoOpen", "-LocalPort", "18765")
    out = (r.stdout + r.stderr).decode().replace("\r\n", "\n")
    out = re.sub(r"\s*\n\s*", " ", out)                  # warnings wrap at the console width
    assert r.returncode == 1, out
    assert want in out, out
    assert "NativeCommandError" not in out


@pytest.mark.parametrize("first,leg", [
    ("The authenticity of host 'gw.example.edu (192.0.2.7)' can't be established.", "gateway"),
    ("No ED25519 host key is known for gw.example.edu and you have requested strict checking.", "gateway"),
    ("The authenticity of host 'dev-node-01 (<no hostip for proxy command>)' can't be established.", "node"),
    ("", "both"),
])
def test_bash_script_names_the_leg_whose_host_key_failed(tmp_path, first, leg):
    """With -J the gateway's key is checked here too (its own ssh process, the
    real known_hosts): the message names the host, as mace gui connect's does."""
    lines = ([first] if first else []) + ["Host key verification failed."]
    env = _stub_ssh_body(tmp_path, "printf '%s\\n' " + " ".join(shlex.quote(x) for x in lines) + " >&2\nexit 255")
    r = subprocess.run(["bash", str(SH), "--gateway", "me@gw.example.edu", "--node", "dev-node-01", "--no-open"],
                       env=env, capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
    gw, node = "    ssh me@gw.example.edu true\n", "    ssh -J me@gw.example.edu me@dev-node-01 true\n"  # the gateway's user
    assert r.returncode == 1
    assert (gw in r.stderr) is (leg in ("gateway", "both")), r.stderr
    assert (node in r.stderr) is (leg in ("node", "both")), r.stderr
    if leg == "both":
        assert r.stderr.index(gw) < r.stderr.index(node)
    kind, msg = C.classify_failure(255, lines, "dev-node-01", "me@gw.example.edu", hop="jump",
                                   trust_cmd=node.strip(), gateway_cmd=gw.strip())
    assert (gw.strip() in msg) is (gw in r.stderr) and (node.strip() in msg) is (node in r.stderr)


def test_powershell_gives_both_host_key_steps_for_the_jump_hop():
    ps = PS1.read_text()
    assert "$script:GatewayTrustCmd" in ps and "first" in ps.split("$script:GatewayTrustCmd")[-1]


# ---- connect.toml with Windows line ends (Python on Windows wrote CRLF before 1.1.5 final)

@pytest.mark.parametrize("eol", ["\n", "\r\n"])
def test_crlf_and_lf_profiles_read_alike_in_python_and_bash(tmp_path, eol):
    env = _env(tmp_path)
    conf = Path(env["XDG_CONFIG_HOME"]) / "mace" / "connect.toml"
    conf.parent.mkdir(parents=True)
    text = C.dump_profiles({"defaults": {"profile": "j"}, "profiles": {"j": PROFILES["jump"]}})
    conf.write_bytes(text.replace("\n", eol).encode())
    r = subprocess.run([sys.executable, str(REPO / "mace_cli"), "--no-banner", "gui", "connect", "--dry-run"],
                       env=env, capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
    assert r.returncode == 0, r.stderr
    want = shlex.split([ln for ln in r.stdout.splitlines() if ln.startswith("ssh ")][0])
    assert "dev-node-01" in want[-2]
    got = subprocess.run(["bash", str(SH), "--dry-run"], env=env, capture_output=True, text=True, timeout=60)
    assert got.returncode == 0, got.stderr
    assert _argv_lines(got.stdout) == want
    assert "\r" not in got.stdout


@needs_pwsh
@pytest.mark.parametrize("eol", ["\n", "\r\n"])
def test_crlf_and_lf_profiles_read_alike_in_powershell(tmp_path, eol):
    env = _ps_env(tmp_path)
    text = C.dump_profiles({"defaults": {"profile": "j"}, "profiles": {"j": PROFILES["jump"]}})
    C.save_profiles({"defaults": {"profile": "j"}, "profiles": {"j": PROFILES["jump"]}},
                    Path(env["XDG_CONFIG_HOME"]) / "mace" / "connect.toml")     # what mace gui connect reads
    conf = Path(env["APPDATA"]) / "mace" / "connect.toml"
    conf.parent.mkdir(parents=True)
    conf.write_bytes(text.replace("\n", eol).encode())
    r = subprocess.run([sys.executable, str(REPO / "mace_cli"), "--no-banner", "gui", "connect", "--dry-run"],
                       env=env, capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
    want = shlex.split([ln for ln in r.stdout.splitlines() if ln.startswith("ssh ")][0])
    p = _run_ps(env, "-DryRun")
    assert p.returncode == 0, p.stderr.decode()
    assert _ps_dry(p.stdout)[0].splitlines() == want[1:]


def test_save_profiles_writes_lf_line_ends(tmp_path, monkeypatch):
    """Even on Windows: text mode there would write CRLF, which older connect scripts misread."""
    import tomllib
    real_fdopen = C.os.fdopen
    seen = {}

    def fdopen(fd, mode="r", *a, **kw):
        seen.update(kw)
        return real_fdopen(fd, mode, *a, **kw)
    monkeypatch.setattr(C.os, "fdopen", fdopen)
    p = C.save_profiles({"defaults": {"profile": "j"}, "profiles": {"j": PROFILES["jump"]}}, tmp_path / "c.toml")
    assert seen.get("newline") == "\n"
    raw = p.read_bytes()
    assert b"\r" not in raw
    assert tomllib.loads(raw.decode())["profiles"]["j"]["node"] == "dev-node-01"


# ---- MobaXterm hand test (1.1.5 final): READY seen at once, the stty noise, // paths, dry-run labels

FAKE_FORWARD = r'''#!/bin/bash
[ "$1" = -G ] && exit 0
LP=""; prev=""
for a in "$@"; do [ "$prev" = -L ] && LP="$(printf '%s' "$a" | cut -d: -f2)"; prev="$a"; done
# the local forward: its first connection stalls (a forward that is not ready), later ones answer 200
PY_SERVER
SRV=$!
# a node's login profile that runs stty without a terminal
echo "stty: 'standard input': Inappropriate ioctl for device" >&2
sleep 0.5
echo 'MACE_GUI_READY {"prefix": "/s/x/", "token": "t", "login": "http://127.0.0.1:1/s/x/?login=abc"}'
while IFS= read -r l; do [ "$l" = detach ] && break; [ "$l" = stop ] && break; done
kill $SRV 2>/dev/null
exit 0
'''
PY_SERVER = ('"%s" -c \'import socket, sys, time\n'
             's = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n'
             's.bind(("127.0.0.1", int(sys.argv[1]))); s.listen(5)\n'
             'stalled = s.accept()[0]\n'
             'while True:\n'
             '    c = s.accept()[0]; c.recv(4096); c.sendall(b"HTTP/1.0 200 OK\\r\\n\\r\\n{}"); c.close()\n'
             '\' "$LP" &')


def _pty_run(argv, env, until, limit=30.0):
    import pty
    import select
    import time
    pid, fd = pty.fork()
    if pid == 0:                                    # pragma: no cover - the child
        os.execvpe(argv[0], argv, env)
    buf, t0 = b"", time.monotonic()
    try:
        while time.monotonic() - t0 < limit:
            r, _, _ = select.select([fd], [], [], 0.2)
            if r:
                try:
                    data = os.read(fd, 4096)
                except OSError:
                    break
                if not data:
                    break
                buf += data
                if until in buf:
                    return buf.decode(errors="replace"), time.monotonic() - t0
        return buf.decode(errors="replace"), None
    finally:
        # the child leads its own session (pty.fork): end the script, its ssh and the fake forward together
        import signal
        import time as _t
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(pid, sig)
            except (ProcessLookupError, PermissionError):
                break
            end = _t.monotonic() + 3
            while _t.monotonic() < end:
                try:
                    if os.waitpid(pid, os.WNOHANG)[0]:
                        break
                except ChildProcessError:
                    break
                _t.sleep(0.1)
            else:
                continue
            break
        try:
            os.killpg(pid, signal.SIGKILL)          # what is left of the group (the fake forward)
        except (ProcessLookupError, PermissionError):
            pass
        os.close(fd)


def _forward_env(tmp_path):
    fakebin = tmp_path / "fakebin"
    fakebin.mkdir()
    (fakebin / "ssh").write_text(FAKE_FORWARD.replace("PY_SERVER", PY_SERVER % sys.executable))
    (fakebin / "ssh").chmod(0o755)
    env = _env(tmp_path)
    env["PATH"] = f"{fakebin}:{env.get('PATH', '/usr/bin:/bin')}"
    env.pop("BROWSER", None)
    return env


@pytest.mark.parametrize("hop", ["nested", "jump"])
def test_bash_script_shows_the_url_without_input_on_a_terminal(tmp_path, hop):
    """READY must lead to the URL at once, with nothing typed: a forward whose
    first connection stalls used to block the script on it for good (MobaXterm
    showed the URL only after Enter)."""
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = _forward_env(tmp_path)
    out, took = _pty_run(["bash", str(SH), "--gateway", "me@gw", "--node", "n1", "--hop", hop, "--no-open",
                          "--local-port", str(port)], env, b"MACE GUI on n1:")
    assert took is not None and took < 15, out
    assert "http://127.0.0.1:%d/s/x/?login=abc" % port in out


def test_bash_script_drops_the_stty_noise_before_ready(tmp_path):
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = _forward_env(tmp_path)
    out, took = _pty_run(["bash", str(SH), "--gateway", "me@gw", "--node", "n1", "--hop", "nested", "--no-open",
                          "--local-port", str(port)], env, b"MACE GUI on n1:")
    assert took is not None, out
    assert "Inappropriate ioctl" not in out


def test_bash_script_reads_ssh_output_from_a_file_not_a_fifo():
    """ssh's stdout is a regular file the script polls (MobaXterm's ssh wrote
    READY to the old FIFO only once something else woke it up)."""
    sh = SH.read_text()
    assert 'mkfifo "$TMPD/in"\n' in sh and '"$TMPD/out"' in sh and 'mkfifo "$TMPD/in" "$TMPD/out"' not in sh
    assert "read -t" in sh.split("http_ok() {")[1].split("\n}\n")[0]


def test_double_slash_profile_path_parity(tmp_path):
    """A profile with dir = "//mnt/research/x" (saved by mace gui connect): the
    script's argv equals mace gui connect's (both send the literal /mnt/...)."""
    env = _env(tmp_path)
    prof = {"gateway": "me@gw.example.edu", "node": "dev-node-01", "hop": "nested", "dir": "//mnt/research/PFGP",
            "roots": ["//mnt/research/PFGP/a b"]}
    C.save_profiles({"defaults": {"profile": "d"}, "profiles": {"d": prof}},
                    Path(env["XDG_CONFIG_HOME"]) / "mace" / "connect.toml")
    r = subprocess.run([sys.executable, str(REPO / "mace_cli"), "--no-banner", "gui", "connect", "--dry-run"],
                       env=env, capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
    want = shlex.split([ln for ln in r.stdout.splitlines() if ln.startswith("ssh ")][0])
    got = _argv_lines(subprocess.run(["bash", str(SH), "--dry-run"], env=env, capture_output=True, text=True,
                                     timeout=60).stdout)
    assert got == want
    assert "cd -- /mnt/research/PFGP" in want[-1]


@pytest.mark.parametrize("flags,label", [(["--gateway", "g", "--node", "n", "--hop", "nested"], "# hop nested"),
                                         (["--gateway", "g", "--node", "n", "--hop", "jump"], "# hop jump"),
                                         (["--gateway", "g", "--node", "n", "--hop", "auto"], "# hop jump"),
                                         (["--gateway", "g"], "# no hop")])
def test_bash_dry_run_labels_the_hop(tmp_path, flags, label):
    out = subprocess.run(["bash", str(SH), "--dry-run", *flags], env=_env(tmp_path), capture_output=True,
                         text=True, timeout=60).stdout.splitlines()
    assert out[0].startswith(label) and out[1] == "ssh"


@needs_pwsh
def test_powershell_dry_run_labels_the_hop(tmp_path):
    p = _run_ps(_ps_env(tmp_path), "-Gateway", "g", "-Node", "n", "-Hop", "nested", "-DryRun")
    assert p.stdout.decode().splitlines()[0].startswith("# hop nested")


def test_bash_script_drops_refused_channels_until_the_tunnel_answers(tmp_path):
    """The jump hop's local forward listens from the moment ssh connects: a
    connection that comes before the server listens (an old GUI tab retrying)
    makes ssh print 'channel N: open failed: connect failed: Connection
    refused'. Noise until the tunnel answers; kept in the error file."""
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = _forward_env(tmp_path)
    fake = tmp_path / "fakebin" / "ssh"
    fake.write_text(fake.read_text().replace(
        "sleep 0.5\n", "echo 'channel 2: open failed: connect failed: Connection refused' >&2\n"
                       "echo 'channel 2: open failed: connect failed: Connection refused' >&2\nsleep 0.5\n"))
    out, took = _pty_run(["bash", str(SH), "--gateway", "me@gw", "--node", "n1", "--hop", "jump", "--no-open",
                          "--local-port", str(port)], env, b"MACE GUI on n1:")
    assert took is not None, out
    assert "open failed" not in out


def test_bash_script_keeps_refused_channels_for_the_diagnosis(tmp_path):
    env = _stub_ssh_body(tmp_path, "echo 'channel 2: open failed: connect failed: Connection refused' >&2\n"
                                   "echo 'me@n1: Permission denied (publickey).' >&2\nexit 255")
    r = subprocess.run(["bash", str(SH), "--gateway", "me@gw", "--node", "n1", "--no-open"], env=env,
                       capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
    assert r.returncode == 1 and "Permission denied" in r.stderr
    assert "open failed" not in r.stderr                 # never confirmed: not shown, but the file has it
    sh = SH.read_text()
    assert 'printf \'%s\\n\' "$l" >> "$TMPD/err"' in sh


# ---- a ConnectTimeout from your ssh config is kept (1.1.6 review): connect.py and both scripts alike

def _stub_ssh_g_timeout(tmp_path, value):
    fakebin = tmp_path / "fakebin-ct"
    fakebin.mkdir()
    (fakebin / "ssh").write_text('#!/bin/sh\n[ "$1" = -G ] || exit 255\n'
                                 f'printf "user netid\\nconnecttimeout {value}\\n"\n')
    (fakebin / "ssh").chmod(0o755)
    return fakebin


@pytest.mark.parametrize("hop", ["jump", "nested"])
@pytest.mark.parametrize("value", ["90", "none"])
def test_config_connect_timeout_parity(tmp_path, hop, value):
    env = _env(tmp_path)
    env["PATH"] = f"{_stub_ssh_g_timeout(tmp_path, value)}:{env.get('PATH', '/usr/bin:/bin')}"
    flags = ["--gateway", "netid@gw.example.edu", "--node", "dev-node", "--hop", hop, "--dry-run"]
    r = subprocess.run([sys.executable, str(REPO / "mace_cli"), "--no-banner", "gui", "connect", *flags],
                       env=env, capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
    assert r.returncode == 0, r.stderr
    want = shlex.split([ln for ln in r.stdout.splitlines() if ln.startswith("ssh ")][0])
    if value == "90":
        assert "ConnectTimeout=20" not in want
        assert ("ConnectTimeout=90" in want[-1]) is (hop == "nested")
    else:
        assert "ConnectTimeout=20" in want
    got = _argv_lines(subprocess.run(["bash", str(SH), *flags], env=env, capture_output=True, text=True,
                                     timeout=60).stdout)
    assert got == want
    if PWSH:
        pe = dict(env, APPDATA=str(tmp_path / "appdata"))
        pe.pop("USERNAME", None)
        ps = _run_ps(pe, "-Gateway", "netid@gw.example.edu", "-Node", "dev-node", "-Hop", hop, "-DryRun")
        assert ps.returncode == 0, ps.stderr.decode()
        args, _ = _ps_dry(ps.stdout)
        assert args == "".join(a + "\n" for a in want[1:])
