"""mace gui connect: the findings of the Windows 11 run (1.1.5 final).

A missing --dir named as a missing folder (not a missing mace), the host-key
prompt and failure on the first ssh -J to a node (never trusted silently,
never a reason for the auto hop), the stderr reader that wrote during
interpreter shutdown, ssh killed with this process on Windows (a Job
Object), and the PowerShell form of --dry-run."""
import ctypes
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

import pytest

from mace.gui import connect as C

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = [REPO / "scripts" / "mace-gui-connect.sh", REPO / "scripts" / "mace-gui-connect.ps1"]
GW, NODE = "me@gw.example.edu", "dev-node-01"


# ---- C: a --dir that does not exist on the node

@pytest.mark.parametrize("line", [
    "bash: line 1: cd: /mnt/home/me/does-not-exist: No such file or directory",
    "bash: cd: /mnt/home/me/does-not-exist: No such file or directory",
    "sh: 1: cd: can't cd to /mnt/home/me/does-not-exist",
])
def test_failed_cd_names_the_folder_not_mace(line):
    kind, msg = C.classify_failure(1, [line], NODE, GW)
    assert kind == "dir", msg
    assert "/mnt/home/me/does-not-exist" in msg and NODE in msg and "--dir" in msg
    assert "did not find the mace command" not in msg


def test_missing_mace_path_is_still_mace_not_found():
    kind, _ = C.classify_failure(127, ["bash: line 1: /home/me/m/mace_cli: No such file or directory"], NODE, GW)
    assert kind == "mace-not-found"


# ---- D: the host key of a node never seen from this computer

def test_waiting_note_mentions_the_host_key_question():
    for jump_node in ("", NODE):
        note = C.waiting_note(jump_node)
        assert "2FA" in note and "host-key" in note and "yes/no" in note
    assert "password" in C.waiting_note(NODE)


def test_host_key_failure_gives_the_one_time_trust_command():
    err = ["The authenticity of host 'dev-node-01 (<no hostip for proxy command>)' can't be established.",
           "Host key verification failed."]
    kind, msg = C.classify_failure(255, err, NODE, GW, hop="jump", trust_cmd=f"ssh -J {GW} {NODE} true")
    assert kind == "host-key"
    assert f"ssh -J {GW} {NODE} true" in msg and "--hop nested" in msg
    assert "StrictHostKeyChecking" not in msg


def test_trust_command_matches_the_session():
    prof = {"gateway": GW, "node": NODE, "jump": "bastion"}
    assert C.host_key_trust_command(prof, "jump", None) == f"ssh -J bastion,{GW} {NODE} true"
    assert C.host_key_trust_command(prof, "jump", "netid") == f"ssh -J bastion,{GW} netid@{NODE} true"
    assert C.host_key_trust_command(prof, "nested", None) == f"ssh -J bastion {GW} true"
    assert C.host_key_trust_command({"gateway": "gw"}, "jump", None) == "ssh gw true"


def test_changed_host_key_is_not_offered_for_trust():
    err = ["@    WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!     @", "Host key verification failed."]
    kind, msg = C.classify_failure(255, err, NODE, GW, hop="jump", trust_cmd=f"ssh -J {GW} {NODE} true")
    assert kind == "host-key"
    assert "ssh -J" not in msg and " true" not in msg
    assert "changed" in msg.lower() and "administrators" in msg


def test_nothing_trusts_host_keys_silently():
    for path in [Path(C.__file__), *SCRIPTS]:
        text = path.read_text()
        assert "StrictHostKeyChecking" not in text, path
        assert "accept-new" not in text, path
        assert "UserKnownHostsFile" not in text, path


def _stub(tmp_path, body):
    fakebin = tmp_path / "fakebin"
    fakebin.mkdir(exist_ok=True)
    ssh = fakebin / "ssh"
    ssh.write_text("#!/bin/sh\n[ \"$1\" = -G ] && exit 0\necho \"$@\" >> %s\n%s\n"
                   % (shlex.quote(str(tmp_path / "ran")), body))
    ssh.chmod(0o755)
    return fakebin


def _connect(tmp_path, fakebin, *args):
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON") and k != "BROWSER"}
    env.update(PATH=f"{fakebin}:/usr/bin:/bin", XDG_CONFIG_HOME=str(tmp_path / "cfg"), HOME=str(tmp_path / "home"))
    (tmp_path / "home").mkdir(exist_ok=True)
    return subprocess.run([sys.executable, str(REPO / "mace_cli"), "--no-banner", "gui", "connect", "--no-open",
                           *args], env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60)


def test_auto_hop_does_not_fall_back_on_a_host_key_failure(tmp_path):
    """A host-key refusal is the user's trust decision (or an attack): auto
    stays on jump and says how to trust the node once."""
    fakebin = _stub(tmp_path, "echo 'Host key verification failed.' >&2\nexit 255")
    r = _connect(tmp_path, fakebin, "--gateway", GW, "--node", NODE)
    assert r.returncode == 1
    assert len((tmp_path / "ran").read_text().splitlines()) == 1          # no nested retry
    assert f"ssh -J {GW} me@{NODE} true" in r.stdout and "--hop nested" in r.stdout   # the gateway's user


def test_cli_names_the_missing_folder(tmp_path):
    fakebin = _stub(tmp_path, "echo 'bash: line 1: cd: /mnt/home/me/does-not-exist: No such file or directory' >&2\n"
                              "exit 1")
    r = _connect(tmp_path, fakebin, "--gateway", GW, "--node", NODE, "--dir", "/mnt/home/me/does-not-exist")
    assert r.returncode == 2
    assert "/mnt/home/me/does-not-exist" in r.stdout and "did not find the mace command" not in r.stdout
    assert "Fatal Python error" not in r.stderr


# ---- C: the stderr reader thread and interpreter shutdown

def test_stderr_reader_is_quiet_after_finish(capfd):
    """ssh's proxy child can hold the stderr pipe after ssh exits: the
    daemon reader stops writing once the session is finished, so it never
    holds sys.stderr's lock while the interpreter shuts down."""
    argv = [sys.executable, "-c", "import subprocess, sys; sys.stderr.write('first\\n'); sys.stderr.flush(); "
            "subprocess.Popen([sys.executable, '-c', 'import time, sys; time.sleep(1.0); "
            "sys.stderr.write(\"late\\\\n\")'])"]
    sess = C.Session(argv, say=lambda s: None)
    sess.proc.wait(10)
    sess.finish_readers(0.5)
    assert "first" in sess.stderr_lines
    time.sleep(1.5)
    err = capfd.readouterr().err
    assert "first" in err and "late" not in err


def test_stderr_prompt_without_newline_is_shown_at_once(capfd):
    """A prompt ssh writes to stderr without a newline is shown before the
    answer, not after (line buffering would hide it)."""
    argv = [sys.executable, "-c", "import sys, time; sys.stderr.write('Are you sure (yes/no)? '); "
            "sys.stderr.flush(); time.sleep(1.0)"]
    sess = C.Session(argv, say=lambda s: None)
    time.sleep(0.6)
    assert "Are you sure (yes/no)? " in capfd.readouterr().err
    sess.proc.wait(10)
    sess.finish_readers(2)
    assert sess.stderr_lines == ["Are you sure (yes/no)? "]


# ---- E: ssh dies with this process on Windows (a Job Object)

class _K32:
    def __init__(self, ok=True):
        self.calls, self.ok = [], ok

    def CreateJobObjectW(self, a, b):
        self.calls.append(("create",))
        return 1234

    def SetInformationJobObject(self, job, cls, info, size):
        flags = ctypes.cast(info, ctypes.POINTER(C._JOBOBJECT_EXTENDED_LIMIT_INFORMATION)).contents
        self.calls.append(("set", job, cls, flags.BasicLimitInformation.LimitFlags, size))
        return 1

    def AssignProcessToJobObject(self, job, handle):
        self.calls.append(("assign", job, handle))
        return 1 if self.ok else 0

    def CloseHandle(self, h):
        self.calls.append(("close", h))
        return 1


def test_job_object_kills_ssh_with_this_process():
    k = _K32()
    assert C.kill_with_this_process(4321, os_name="nt", kernel32=k) == 1234
    assert k.calls[0] == ("create",)
    _, job, cls, flags, size = k.calls[1]
    assert (job, cls, flags & 0x2000) == (1234, 9, 0x2000)
    assert size == ctypes.sizeof(C._JOBOBJECT_EXTENDED_LIMIT_INFORMATION)
    assert k.calls[2] == ("assign", 1234, 4321) and len(k.calls) == 3      # the handle stays open


def test_job_object_failure_is_harmless():
    k = _K32(ok=False)
    assert C.kill_with_this_process(4321, os_name="nt", kernel32=k) is None
    assert k.calls[-1] == ("close", 1234)


def test_job_object_is_a_no_op_elsewhere():
    class Boom:
        def __getattr__(self, name):
            raise AssertionError("kernel32 used off Windows")
    assert C.kill_with_this_process(4321, os_name="posix", kernel32=Boom()) is None


@pytest.mark.skipif(ctypes.sizeof(ctypes.c_void_p) != 8, reason="64-bit layout")
def test_job_limit_struct_matches_the_windows_x64_layout():
    assert ctypes.sizeof(C._JOBOBJECT_EXTENDED_LIMIT_INFORMATION) == 144


# ---- F: --dry-run on Windows prints a PowerShell form too

def _ms_argv(cmdline):
    """The Microsoft C runtime's command-line rules (what ssh.exe parses)."""
    out, cur, i, inq, have = [], [], 0, False, False
    while i < len(cmdline):
        c = cmdline[i]
        if c == "\\":
            j = i
            while j < len(cmdline) and cmdline[j] == "\\":
                j += 1
            n = j - i
            if j < len(cmdline) and cmdline[j] == '"':
                cur.append("\\" * (n // 2))
                if n % 2:
                    cur.append('"')
                    i = j + 1
                else:
                    i = j
                have = True
                continue
            cur.append("\\" * n)
            i, have = j, True
            continue
        if c == '"':
            inq, have = not inq, True
        elif c in " \t" and not inq:
            if have:
                out.append("".join(cur))
            cur, have = [], False
        else:
            cur.append(c)
            have = True
        i += 1
    if have:
        out.append("".join(cur))
    return out


def test_dry_run_on_windows_adds_a_powershell_form(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr(C, "on_windows", lambda: True)
    assert C.main(["--dry-run", "--gateway", GW, "--node", NODE, "--dir", "~/my runs", "--setup",
                   "module load Python; export X='a \"b\"'"]) == 0
    lines = capsys.readouterr().out.splitlines()
    posix = [ln for ln in lines if ln.startswith("ssh ") and not ln.startswith("ssh --%")]
    ps = [ln for ln in lines if ln.startswith("ssh --% ")]
    assert len(posix) == len(ps) == 2                                  # hop auto: jump and nested
    assert any("PowerShell" in ln and ln.startswith("#") for ln in lines)
    assert any("bash" in ln and ln.startswith("#") for ln in lines)
    for a, b in zip(posix, ps):
        assert _ms_argv(b[len("ssh --% "):]) == shlex.split(a)[1:]
    assert lines.index(posix[0]) < lines.index(ps[0])                    # the POSIX line stays first


def test_dry_run_elsewhere_prints_the_posix_form_only(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr(C, "on_windows", lambda: False)
    assert C.main(["--dry-run", "--gateway", GW, "--node", NODE]) == 0
    assert "--%" not in capsys.readouterr().out


# ---- B: the flat TOML reader (Python < 3.11) decodes every basic-string escape as tomllib does

def test_parse_flat_matches_tomllib():
    import tomllib
    from test_connect_scripts import HAND, TRICKY
    saved = C.dump_profiles({"defaults": {"profile": "t"}, "profiles": {"t": TRICKY}})
    for text in (HAND, saved):
        data, bad = C.parse_flat(text)
        assert bad == [] and data == tomllib.loads(text)


# ---- the gateway's host key with ssh -J (the jump leg checks the real known_hosts too)

GW_CMD, NODE_CMD = f"ssh {GW} true", f"ssh -J {GW} {NODE} true"


def _hk(err):
    return C.classify_failure(255, err, NODE, GW, hop="jump", trust_cmd=NODE_CMD, gateway_cmd=GW_CMD)


@pytest.mark.parametrize("first", [
    "The authenticity of host 'gw.example.edu (192.0.2.7)' can't be established.",
    "No ED25519 host key is known for gw.example.edu and you have requested strict checking.",
])
def test_host_key_failure_of_the_gateway_names_the_gateway(first):
    kind, msg = _hk([first, "Host key verification failed.", "Connection closed by UNKNOWN port 65535"])
    assert kind == "host-key"
    assert "gateway" in msg and GW_CMD in msg and NODE_CMD not in msg and "--hop nested" not in msg


@pytest.mark.parametrize("first", [
    "The authenticity of host 'dev-node-01 (<no hostip for proxy command>)' can't be established.",
    "No ED25519 host key is known for dev-node-01 and you have requested strict checking.",
])
def test_host_key_failure_of_the_node_names_the_node(first):
    kind, msg = _hk([first, "Host key verification failed."])
    assert "dev-node-01's host key" in msg and NODE_CMD in msg and "--hop nested" in msg
    assert GW_CMD + "\n" not in msg


def test_host_key_failure_of_an_unknown_host_gives_both_steps_in_order():
    kind, msg = _hk(["Host key verification failed."])
    assert GW_CMD in msg and NODE_CMD in msg and msg.index(GW_CMD) < msg.index(NODE_CMD)


def test_host_key_named_by_an_alias_hostname_is_matched_through_ssh_g(monkeypatch):
    """A gateway alias (hpcc -> login.example.edu): ssh names the HostName."""
    kind, msg = C.classify_failure(255, ["The authenticity of host 'login.example.edu (192.0.2.7)' can't be "
                                         "established.", "Host key verification failed."], NODE, "hpcc",
                                   hop="jump", trust_cmd="ssh -J hpcc dev-node-01 true", gateway_cmd="ssh hpcc true",
                                   gateway_hostname="login.example.edu")
    assert "ssh hpcc true" in msg and "ssh -J hpcc dev-node-01 true" not in msg


# ---- the stty line a node's login profile prints without a terminal (MobaXterm hand test)

def test_stty_noise_before_ready_is_not_echoed_but_kept(capfd):
    argv = [sys.executable, "-c",
            "import sys, time; sys.stderr.write(\"stty: 'standard input': Inappropriate ioctl for device\\n\");"
            "sys.stderr.write('real message\\n'); sys.stderr.flush(); time.sleep(0.3)"]
    sess = C.Session(argv, say=lambda s: None)
    sess.proc.wait(10)
    sess.finish_readers(2)
    err = capfd.readouterr().err
    assert "Inappropriate ioctl" not in err and "real message" in err
    assert any("Inappropriate ioctl" in ln for ln in sess.stderr_lines)        # still there for the diagnosis


def test_stty_noise_after_ready_is_echoed(capfd):
    argv = [sys.executable, "-c",
            "import sys, time; print('MACE_GUI_READY {}', flush=True); time.sleep(0.5);"
            "sys.stderr.write('stty: standard input: Inappropriate ioctl for device\\n'); sys.stderr.flush()"]
    sess = C.Session(argv, say=lambda s: None)
    sess.proc.wait(10)
    sess.finish_readers(2)
    assert "Inappropriate ioctl" in capfd.readouterr().err


# ---- refused forward channels before the tunnel answers (jump hop, MobaXterm hand test)

REFUSED = "channel 2: open failed: connect failed: Connection refused"


def test_refused_channels_are_quiet_until_the_tunnel_is_confirmed(capfd):
    argv = [sys.executable, "-c",
            "import sys, time; print('MACE_GUI_READY {}', flush=True);"
            f"sys.stderr.write({REFUSED!r} + '\\n'); sys.stderr.flush(); time.sleep(0.6);"
            f"sys.stderr.write({REFUSED.replace('2', '3', 1)!r} + '\\n'); sys.stderr.flush(); time.sleep(0.3)"]
    sess = C.Session(argv, say=lambda s: None)
    time.sleep(0.3)
    sess.tunnel_confirmed()
    sess.proc.wait(10)
    sess.finish_readers(2)
    err = capfd.readouterr().err
    assert "channel 2: open failed" not in err                 # before the tunnel answered
    assert "channel 3: open failed" in err                     # after: a real problem, shown
    assert any("channel 2: open failed" in ln for ln in sess.stderr_lines)


def test_drive_confirms_the_tunnel_after_wait_ready(monkeypatch):
    src = Path(C.__file__).read_text()
    i = src.index("ok = wait_ready(base, token)")
    assert "sess.tunnel_confirmed()" in src[i:i + 200]
    assert "REFUSED_NOISE_RE.match(ln)" in src[i:i + 600]          # shown when the tunnel never answers
