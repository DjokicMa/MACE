"""Two people on one development node: per-user ports, the session registry,
the ::1 listener, and finding ssh when Windows' PATH lost it."""
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from conftest import REPO_ROOT

from mace.gui import connect_check as K
from mace.gui import ports, registry, sshfind

MACE_CLI = REPO_ROOT / "mace_cli"


# ---------------------------------------------------------------- ports
def test_user_port_comes_from_the_user_id():
    assert ports.user_port(uid=846118) == 26118
    assert ports.user_port(uid=1000) == 21000
    assert ports.user_port(uid=10000) == 20000
    assert ports.USER_PORT_LOW <= ports.user_port(uid=2 ** 31 - 1) < ports.USER_PORT_LOW + ports.USER_PORT_SPAN


def test_two_users_of_one_group_get_different_ports():
    assert ports.user_port(uid=846118) != ports.user_port(uid=846119)


def test_user_port_without_ids_hashes_the_name(monkeypatch):
    import zlib
    monkeypatch.delattr(os, "getuid", raising=False)
    assert ports.user_slot("alice") == zlib.crc32(b"alice") % ports.USER_PORT_SPAN
    assert ports.user_port("alice") == ports.user_port("alice") != ports.user_port("bob")


def test_shared_node_detection():
    assert ports.shared_node({"SSH_CONNECTION": "1 2 3 4"})
    assert ports.shared_node({"SLURM_JOB_ID": "12"})
    assert not ports.shared_node({})
    assert not ports.shared_node({"SSH_CONNECTION": "1 2 3 4", "MACE_GUI_SHARED": "0"})
    assert ports.shared_node({"MACE_GUI_SHARED": "1"})


def test_default_port_local_and_shared():
    assert ports.default_port({}, uid=846118) == (8765, "local")
    assert ports.default_port({"SSH_CONNECTION": "x"}, uid=846118) == (26118, "user")


def test_pick_goes_past_a_busy_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        busy = s.getsockname()[1]
        got = ports.pick(busy)
        assert got is not None and got > busy
        assert not ports.is_free(busy)


def test_pick_gives_up_when_every_port_is_busy():
    assert ports.pick(30000, tries=3, free=lambda p, h: False) is None


@pytest.mark.skipif(not socket.has_ipv6, reason="no IPv6")
def test_a_port_busy_on_ipv6_loopback_is_not_picked():
    try:
        s = socket.socket(socket.AF_INET6)
        s.bind(("::1", 0))
    except OSError:
        pytest.skip("no ::1 here")
    with s:
        s.listen(1)
        port = s.getsockname()[1]
        assert ports.is_free(port, "127.0.0.1")           # the IPv4 side alone would say yes
        assert not ports.is_free_loopback(port)


def test_forward_lines_name_node_user_and_port():
    lines = ports.forward_lines(26118, "dev-amd24", "alice", "hpcc.example.edu")
    assert lines[0] == "ssh -N -J alice@hpcc.example.edu -L 26118:127.0.0.1:26118 alice@dev-amd24"
    assert "<" not in lines[0]


# ---------------------------------------------------------------- registry
def test_registration_lifecycle(tmp_path):
    reg = registry.Registration(26118, "/runs/a", directory=tmp_path, beat_s=0.05).start()
    f = reg.path
    assert f.is_file() and (f.stat().st_mode & 0o777) == 0o600
    got = registry.live(tmp_path, port_open=lambda p: True)
    assert [d["port"] for d in got] == [26118] and got[0]["here"] and "token" not in got[0]
    assert registry.serving("/runs/a", got) is got[0]
    assert registry.serving("/runs/b", got) is None
    reg.close()
    assert not f.exists() and registry.live(tmp_path) == []


def test_heartbeat_touches_the_record_and_follows_open_folder(tmp_path):
    folder = ["/runs/a"]
    reg = registry.Registration(26118, "/runs/a", directory=tmp_path, beat_s=0.05, root_fn=lambda: folder[0]).start()
    try:
        os.utime(reg.path, (1, 1))
        time.sleep(0.4)
        assert reg.path.stat().st_mtime > 1000
        folder[0] = "/runs/b"
        time.sleep(0.4)
        assert json.loads(reg.path.read_text())["root"] == "/runs/b"
    finally:
        reg.close()


def test_dead_process_records_are_pruned(tmp_path):
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    (tmp_path / f"{registry.short_host()}-26118.json").write_text(
        json.dumps({"host": registry.short_host(), "port": 26118, "pid": p.pid, "root": "/x"}))
    assert registry.live(tmp_path) == []
    assert not list(tmp_path.glob("*.json"))


def test_a_record_from_another_node_counts_only_while_it_is_touched(tmp_path):
    f = tmp_path / "dev-intel18-26118.json"
    f.write_text(json.dumps({"host": "dev-intel18", "port": 26118, "pid": 1, "root": "/x"}))
    got = registry.live(tmp_path)
    assert len(got) == 1 and not got[0]["here"]
    old = time.time() - registry.FRESH_S - 60
    os.utime(f, (old, old))
    assert registry.live(tmp_path) == []
    assert f.exists()                                      # not ours to delete


def test_pid_alive():
    assert registry.pid_alive(os.getpid())
    assert not registry.pid_alive(0) and not registry.pid_alive(-3)


def _windows(monkeypatch, handle, last_error, exit_code=259):
    """Windows' kernel32 as pid_alive asks it (ctypes.WinDLL), on this host."""
    import ctypes

    class K32:
        def OpenProcess(self, access, inherit, pid):
            return handle

        def GetExitCodeProcess(self, h, ref):
            ref._obj.value = exit_code
            return 1

        def CloseHandle(self, h):
            return 1

    monkeypatch.setattr(ctypes, "WinDLL", lambda name, use_last_error=False: K32(), raising=False)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: last_error, raising=False)


@pytest.mark.parametrize("handle, last_error, exit_code, alive", [
    (0, 5, None, True),        # ERROR_ACCESS_DENIED: another user's process, alive
    (0, 87, None, False),      # ERROR_INVALID_PARAMETER: no such process
    (42, 0, 259, True),        # STILL_ACTIVE
    (42, 0, 0, False),         # exited
])
def test_pid_alive_on_windows(monkeypatch, handle, last_error, exit_code, alive):
    """A lock or record of another user's live process is not broken: on
    Windows OpenProcess is refused with ERROR_ACCESS_DENIED for it."""
    _windows(monkeypatch, handle, last_error, exit_code)
    with monkeypatch.context() as m:
        m.setattr(os, "name", "nt")
        m.setattr(os, "kill", lambda *a: pytest.fail("os.kill ends the process on Windows"))
        got = registry.pid_alive(4321)
    assert got is alive


def test_pid_probes_ask_pid_alive_on_windows(monkeypatch, tmp_path):
    """mace run's stale-run and claim checks and the pre-optimization sweep
    never call os.kill(pid, 0) on Windows (it ends the process there)."""
    from mace.preopt import proc
    from mace.submission import local_run
    asked = []
    monkeypatch.setattr(registry, "pid_alive", lambda pid: asked.append(pid) or True)
    claims = local_run.Claims("winhost")
    claim = tmp_path / "opt-x.claim"
    claim.write_text("pid 4322 on winhost: x.d12\n")
    with monkeypatch.context() as m:
        m.setattr(os, "name", "nt")
        m.setattr(os, "kill", lambda *a: pytest.fail("os.kill ends the process on Windows"))
        got = (proc._alive(4321), claims._live(claim, "pid 4322 on winhost: x.d12"))
    assert got == (True, True) and asked == [4321, 4322]


def test_describe_is_plain_text():
    text = registry.describe({"port": 26118, "host": "dev-amd24", "root": "/runs", "pid": 5, "here": True,
                              "started": time.time() - 7200})
    assert "26118" in text and "dev-amd24" in text and "2 h" in text and "process 5" in text


# ---------------------------------------------------------------- the real command, as a subprocess
def _env(tmp_path, **extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("MACE_GUI", "SLURM_", "SSH_"))}
    run = tmp_path / "run"
    run.mkdir(exist_ok=True)
    run.chmod(0o700)
    env.update(MACE_GUI_STATE_DIR=str(tmp_path / "state"), XDG_RUNTIME_DIR=str(run), MPLBACKEND="Agg",
               MACE_GUI_SHARED="1", HOME=str(tmp_path), XDG_CONFIG_HOME=str(tmp_path / "cfg"),
               XDG_CACHE_HOME=str(tmp_path / "cache"), COLUMNS="200")
    env.update(extra)
    return env


def _wait_record(tmp_path, n=1, timeout=40):
    end = time.time() + timeout
    while time.time() < end:
        got = list((tmp_path / "state" / "sessions").glob("*.json")) if (tmp_path / "state" / "sessions").is_dir() else []
        if len(got) >= n:
            return [json.loads(f.read_text()) for f in got]
        time.sleep(0.2)
    raise AssertionError("the server did not register")


def test_real_command_user_port_registry_and_ipv6(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    env = _env(tmp_path)
    cmd = [sys.executable, str(MACE_CLI), "--no-banner", "gui", "--dir", str(work)]
    first = subprocess.Popen(cmd, env=env, cwd=tmp_path, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        rec = _wait_record(tmp_path)[0]
        want = ports.user_port()
        assert rec["port"] >= want and rec["pid"] == first.pid and rec["root"] == str(work)
        # IPv4, IPv6 loopback and the name localhost all answer (any HTTP status: no sign-in yet)
        for host in ("127.0.0.1", "::1", "localhost"):
            if host == "::1" and not socket.has_ipv6:
                continue
            try:
                with socket.create_connection((host, rec["port"]), timeout=5) as s:
                    s.sendall(f"GET / HTTP/1.1\r\nHost: localhost:{rec['port']}\r\nConnection: close\r\n\r\n".encode())
                    assert s.recv(64).startswith(b"HTTP/1.")
            except OSError as exc:
                if host == "::1":
                    pytest.skip(f"no IPv6 loopback here ({exc})")
                raise
        # the same folder again: reported, not started, exit 0
        again = subprocess.run(cmd, env=env, cwd=tmp_path, capture_output=True, text=True, timeout=60)
        assert again.returncode == 0 and "already running for you" in again.stdout, again.stdout + again.stderr
        assert f"port {rec['port']}" in again.stdout and "mace gui --new-session" in again.stdout
        # another folder: a second server on the next port, naming the first
        other = tmp_path / "other"
        other.mkdir()
        second = subprocess.Popen(cmd[:-1] + [str(other)], env=env, cwd=tmp_path, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True)
        try:
            recs = _wait_record(tmp_path, 2)
            assert len({r["port"] for r in recs}) == 2
        finally:
            second.terminate()
            second.wait(timeout=30)
        listing = subprocess.run([sys.executable, str(MACE_CLI), "--no-banner", "gui", "--sessions"], env=env,
                                 cwd=tmp_path, capture_output=True, text=True, timeout=60)
        assert listing.returncode == 0 and f"port {rec['port']}" in listing.stdout
        assert f"-L {rec['port']}:127.0.0.1:{rec['port']}" in listing.stdout
    finally:
        first.terminate()
        first.wait(timeout=30)
    assert not list((tmp_path / "state" / "sessions").glob("*.json"))     # records go with the servers


def test_named_port_is_used_as_is_and_busy_is_an_error(tmp_path):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        busy = s.getsockname()[1]
        r = subprocess.run([sys.executable, str(MACE_CLI), "--no-banner", "gui", "--dir", str(tmp_path), "--port",
                            str(busy)], env=_env(tmp_path), cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and "cannot listen" in r.stderr and "--sessions" in r.stderr


# ---------------------------------------------------------------- ssh discovery (Windows)
SYS32 = r"C:\Windows\System32\OpenSSH\ssh.exe"
GIT = r"C:\Program Files\Git\usr\bin\ssh.exe"
WINENV = {"SystemRoot": r"C:\Windows", "ProgramFiles": r"C:\Program Files", "PATH": r"C:\nowhere"}


def test_path_wins():
    f = sshfind.find_ssh(WINENV, which=lambda n: r"C:\x\ssh.exe", windows=True)
    assert f.on_path and f.path == r"C:\x\ssh.exe"


def test_system32_is_found_when_path_lost_it():
    f = sshfind.find_ssh(WINENV, which=lambda n: None, isfile=lambda p: p == SYS32, windows=True)
    assert f.path == SYS32 and f.source == "Windows OpenSSH" and not f.on_path
    assert f.folder == r"C:\Windows\System32\OpenSSH"


def test_git_for_windows_is_the_second_choice():
    f = sshfind.find_ssh(WINENV, which=lambda n: None, isfile=lambda p: p in (GIT, SYS32), windows=True)
    assert f.source == "Windows OpenSSH"
    f = sshfind.find_ssh(WINENV, which=lambda n: None, isfile=lambda p: p == GIT, windows=True)
    assert f.source == "Git for Windows" and "own agent" in sshfind.found_note(f)


def test_per_user_git_install_and_x86():
    env = {**WINENV, "LOCALAPPDATA": r"C:\Users\me\AppData\Local", "ProgramFiles(x86)": r"C:\Program Files (x86)"}
    want = r"C:\Users\me\AppData\Local\Programs\Git\usr\bin\ssh.exe"
    assert sshfind.find_ssh(env, which=lambda n: None, isfile=lambda p: p == want, windows=True).path == want
    x86 = r"C:\Program Files (x86)\Git\usr\bin\ssh.exe"
    assert sshfind.find_ssh(env, which=lambda n: None, isfile=lambda p: p == x86, windows=True).path == x86


def test_nothing_found_says_how_to_install():
    assert sshfind.find_ssh(WINENV, which=lambda n: None, isfile=lambda p: False, windows=True) is None
    got, why = sshfind.ensure_ssh(lambda t: None, dict(WINENV), which=lambda n: None, isfile=lambda p: False,
                                  windows=True)
    assert got is None and "Add-WindowsCapability -Online -Name OpenSSH.Client" in why and "ssh -V" in why


def test_ensure_ssh_puts_its_folder_first_in_path_and_says_so():
    env = dict(WINENV)
    said = []
    got, why = sshfind.ensure_ssh(said.append, env, which=lambda n: None, isfile=lambda p: p == SYS32, windows=True)
    assert why is None and env["PATH"].startswith(r"C:\Windows\System32\OpenSSH;")
    assert len(said) == 1 and "not on your PATH" in said[0] and SYS32 in said[0]
    said.clear()
    sshfind.ensure_ssh(said.append, env, which=lambda n: None, isfile=lambda p: p == SYS32, windows=True)
    assert env["PATH"].count("OpenSSH") == 1                  # not added twice


def test_posix_without_ssh_is_not_searched_in_windows_places():
    assert sshfind.find_ssh({"PATH": "/nope"}, which=lambda n: None, isfile=lambda p: True, windows=False) is None
    assert "openssh" in sshfind.missing_message(False)


def test_vscode_setting_with_comments_and_trailing_commas(tmp_path):
    settings = tmp_path / "Code" / "User" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text('{\n  // the ssh Remote-SSH runs\n  "remote.SSH.path": "D:\\\\tools\\\\ssh.exe",\n'
                        '  "editor.fontSize": 14,\n}\n')
    assert sshfind.vscode_ssh_path({"APPDATA": str(tmp_path)}) == r"D:\tools\ssh.exe"
    assert sshfind.vscode_ssh_path({}) is None
    f = sshfind.find_ssh({**WINENV, "APPDATA": str(tmp_path)}, which=lambda n: None,
                         isfile=lambda p: p == r"D:\tools\ssh.exe", windows=True)
    assert f.source == "VS Code setting"


def test_check_reports_an_ssh_that_is_off_path(monkeypatch):
    class Fake:
        def __call__(self, argv, **kw):
            return subprocess.CompletedProcess(argv, 0, "", "OpenSSH_for_Windows_9.5p1, LibreSSL 3.8.2")

    monkeypatch.setattr(sshfind, "find_ssh", lambda *a, **k: sshfind.SshFound(SYS32, "Windows OpenSSH", False))
    monkeypatch.setattr(sshfind, "prepend_to_path", lambda folder, env=None: None)
    lines = []
    c = K.Checker({"gateway": "gw"}, "p", runner=Fake(), which=lambda n: None, system="windows",
                  out=lines.append, no_net=True)
    assert c.check_ssh()
    text = "\n".join(lines)
    assert "WARN  ssh is not on your PATH, but " + SYS32 in text and "add C:\\Windows\\System32\\OpenSSH to your user PATH" in text


def test_check_missing_ssh_lists_where_it_looked(monkeypatch):
    monkeypatch.setattr(sshfind, "find_ssh", lambda *a, **k: None)
    lines = []
    c = K.Checker({"gateway": "gw"}, "p", which=lambda n: None, system="windows", out=lines.append, no_net=True)
    assert not c.check_ssh()
    text = "\n".join(lines)
    assert "FAIL  no ssh command on PATH" in text and "Add-WindowsCapability" in text and "Git for Windows" in text


def test_the_powershell_script_looks_in_the_same_places():
    ps = (REPO_ROOT / "scripts" / "mace-gui-connect.ps1").read_text()
    for needle in (r"System32\OpenSSH", r"$env:ProgramFiles\OpenSSH",
                   r"Git\usr\bin", "remote.SSH.path", "Add-WindowsCapability", "Find-Native", "Native-Path"):
        assert needle in ps, needle
    assert 'FileName = "ssh.exe"' not in ps                  # every launch goes through the finder


# ---------------------------------------------------------------- PowerShell 5.1 and 7 pass the same argv to ssh
def _pwsh():
    import shutil
    return shutil.which("pwsh")


TRICKY = ["", 'say "hi"', "two words", "back\\slash\\", 'q"uote\\', "-o", "ConnectTimeout=20", "a'b", "$HOME", "~/x y"]


@pytest.mark.skipif(_pwsh() is None, reason="needs pwsh")
@pytest.mark.parametrize("mode", ["Legacy", "Standard", "Windows"])
def test_native_argument_passing_is_the_same_in_every_mode(tmp_path, mode):
    """The script builds ssh's command line itself (ProcessStartInfo.Arguments through Win-Quote), so PowerShell 5.1
    (drops empty strings, mangles quotes) and 7.3+ ($PSNativeCommandArgumentPassing) hand ssh the same argv."""
    import re
    text = (REPO_ROOT / "scripts" / "mace-gui-connect.ps1").read_text()
    start = text.index("function Win-Quote")
    end = text.index("# ------------------------------------------------------------------ profile")
    funcs = tmp_path / "funcs.ps1"
    funcs.write_text(text[start:end], encoding="utf-8")
    probe = tmp_path / "probe.py"
    probe.write_text("import json, sys\nprint(json.dumps(sys.argv[1:]))\n")
    args = ", ".join("'" + a.replace("'", "''") + "'" for a in TRICKY)
    script = (f"$PSNativeCommandArgumentPassing = '{mode}'\n. '{funcs}'\n"
              f"$r = Invoke-Native '{sys.executable}' @('{probe}', {args})\n"
              "Write-Output $r.Out; exit $r.Code")
    run = tmp_path / "run.ps1"
    run.write_text(script, encoding="utf-8")
    r = subprocess.run([_pwsh(), "-NoProfile", "-File", str(run)], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout.strip().splitlines()[-1]) == TRICKY


def test_the_script_uses_no_powershell_7_syntax():
    import re
    text = (REPO_ROOT / "scripts" / "mace-gui-connect.ps1").read_text()
    code = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
    assert not re.search(r"\?\?|\|\||(?<![&\"'])&&(?![&\"'])\s*\$|\s\?\s+\S+\s+:\s", re.sub(r'"[^"\n]*"|\'[^\'\n]*\'', "", code))
    assert "Get-WmiObject" not in code and "Out-File" not in code and "Set-Content" not in code
    assert "Invoke-WebRequest" not in code or "-UseBasicParsing" in code
    assert "Get-Content" not in code or "-Encoding UTF8" in code        # a BOM is fine; 5.1 reads ANSI otherwise


# ---------------------------------------------------------------- held jobs: badge data and Release
def test_held_by_reasons():
    from mace.gui import jobs
    assert jobs.held_by("PENDING", "JobHeldUser") == "user"
    assert jobs.held_by("PENDING", "JobHeldAdmin") == "admin"
    assert jobs.held_by("PENDING", "Priority") is None
    assert jobs.held_by("RUNNING", "JobHeldUser") is None


def _fake_squeue(monkeypatch, listing, calls):
    from mace.gui import jobs
    monkeypatch.setattr(jobs, "_env", lambda: {"PATH": "/x"})
    monkeypatch.setattr(jobs, "_which", lambda tool, env: f"/usr/bin/{tool}")
    monkeypatch.setattr(jobs, "user", lambda: "me")
    monkeypatch.setattr("mace.queue.slurm_ids.own_job_ids", lambda env: {"999"})

    def run(argv, env, timeout=10.0):
        calls.append(argv)
        if argv[0].endswith("squeue"):
            jid = argv[argv.index("-j") + 1]
            out = "".join(f"{i}|{s}|{r}\n" for i, s, r in listing if i == jid or i.startswith(jid + "_"))
            return subprocess.CompletedProcess(argv, 0, out, "")
        return subprocess.CompletedProcess(argv, 0, "", "")
    monkeypatch.setattr(jobs, "_run", run)
    return jobs


def test_release_runs_scontrol_only_for_jobs_held_by_the_user(monkeypatch, tmp_path):
    calls = []
    jobs = _fake_squeue(monkeypatch, [("100", "PENDING", "JobHeldUser"), ("200", "PENDING", "JobHeldAdmin"),
                                      ("300", "PENDING", "Priority"), ("400", "RUNNING", "dev-amd24"),
                                      ("999", "PENDING", "JobHeldUser")], calls)
    res = jobs.release(tmp_path, ["100", "200", "300", "400", "500", "x; rm -rf", "999"])
    assert res["released"] == ["100"]
    why = {r["id"]: r["reason"] for r in res["refused"]}
    assert "administrator" in why["200"] and "not held" in why["300"] and "not held" in why["400"]
    assert "not one of your" in why["500"] and "not a job id" in why["x; rm -rf"] and "MACE GUI runs in" in why["999"]
    assert [c for c in calls if c[0].endswith("scontrol")] == [["/usr/bin/scontrol", "release", "100"]]


def test_describe_rows_carry_the_held_flag(tmp_path):
    from mace.gui import jobs
    rows = [{"id": "100", "name": "a", "state": "PENDING", "elapsed": "0:00", "limit": "1:00", "nodes": "1",
             "cpus": "4", "reason": "JobHeldUser", "workdir": str(tmp_path)},
            {"id": "101", "name": "b", "state": "PENDING", "elapsed": "0:00", "limit": "1:00", "nodes": "1",
             "cpus": "4", "reason": "Priority", "workdir": str(tmp_path)}]
    out = jobs.describe(tmp_path, rows, "all", [], manifests={})
    assert [r["held"] for r in out] == ["user", None]


# ---------------------------------------------------------------- a server whose session went away stops
import argparse
import signal as _signal


def _args(**kw):
    return argparse.Namespace(**{"keep": False, "managed": False, "http_control": False, **kw})


def test_session_watch_decisions(monkeypatch):
    from mace.gui import cli
    monkeypatch.delenv("MACE_GUI_KEEP", raising=False)
    r, w = os.pipe()
    try:
        assert cli.session_watch_wanted(_args(), r)                        # a pipe from sshd without a terminal
        assert not cli.session_watch_wanted(_args(keep=True), r)
        assert not cli.session_watch_wanted(_args(managed=True), r)        # mace gui connect has its own protocol
        assert not cli.session_watch_wanted(_args(http_control=True), r)
        monkeypatch.setenv("MACE_GUI_KEEP", "1")
        assert not cli.session_watch_wanted(_args(), r)
        monkeypatch.delenv("MACE_GUI_KEEP")
        old = _signal.signal(_signal.SIGHUP, _signal.SIG_IGN)               # nohup
        try:
            assert not cli.session_watch_wanted(_args(), r)
        finally:
            _signal.signal(_signal.SIGHUP, old)
        fd = os.open(os.devnull, os.O_RDONLY)                               # cron, systemd, `< /dev/null &`
        try:
            assert not cli.session_watch_wanted(_args(), fd)
        finally:
            os.close(fd)
        os.close(w)
        w = None
        assert not cli.session_watch_wanted(_args(), r)                     # already closed at launch
    finally:
        os.close(r)
        if w is not None:
            os.close(w)
    a, b = socket.socketpair()                                              # sshd's socket stdin
    try:
        assert cli.session_watch_wanted(_args(), a.fileno())
        b.close()
        assert not cli.session_watch_wanted(_args(), a.fileno())
    finally:
        a.close()


def _gui(tmp_path, name, stdin, extra=(), preexec=None, pass_fds=()):
    work = tmp_path / name
    work.mkdir()
    cmd = [sys.executable, str(MACE_CLI), "--no-banner", "gui", "--dir", str(work), "--port", "0", *extra]
    env = _env(tmp_path)
    env["MACE_GUI_STATE_DIR"] = str(tmp_path / f"state-{name}")
    p = subprocess.Popen(cmd, env=env, cwd=tmp_path, stdin=stdin, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, preexec_fn=preexec)
    sess = tmp_path / f"state-{name}" / "sessions"
    end = time.time() + 60
    while time.time() < end:
        got = list(sess.glob("*.json")) if sess.is_dir() else []
        if got:
            return p, json.loads(got[0].read_text()), sess
        time.sleep(0.2)
    p.kill()
    raise AssertionError("the server did not register")


def _port_closed(port):
    try:
        socket.create_connection(("127.0.0.1", port), timeout=1).close()
        return False
    except OSError:
        return True


def test_server_stops_when_its_stdin_pipe_closes(tmp_path):
    """`ssh node "mace gui"` without a terminal: no hangup, but the connection's end closes stdin."""
    p, rec, sess = _gui(tmp_path, "pipe", subprocess.PIPE)
    try:
        assert p.poll() is None and not _port_closed(rec["port"])
        p.stdin.close()
        assert p.wait(timeout=30) == 0
        assert _port_closed(rec["port"]) and not list(sess.glob("*.json"))
        assert "session that started it closed" in p.stdout.read()
    finally:
        if p.poll() is None:
            p.kill()


def test_server_stops_when_its_stdin_socket_closes(tmp_path):
    a, b = socket.socketpair()
    try:
        p, rec, sess = _gui(tmp_path, "sock", b.fileno())
        b.close()                                  # the child has its own copy; ours is gone
        assert p.poll() is None
        a.close()                                  # the "client" went away
        assert p.wait(timeout=30) == 0 and _port_closed(rec["port"])
    finally:
        for s in (a, b):
            s.close()
        if p.poll() is None:
            p.kill()


def test_keep_and_nohup_servers_outlive_their_session(tmp_path):
    kept, rec1, _ = _gui(tmp_path, "keep", subprocess.PIPE, extra=("--keep",))
    nohup, rec2, _ = _gui(tmp_path, "nohup", subprocess.PIPE,
                          preexec=lambda: _signal.signal(_signal.SIGHUP, _signal.SIG_IGN))
    try:
        kept.stdin.close()
        nohup.stdin.close()
        time.sleep(4)
        assert kept.poll() is None and nohup.poll() is None
        assert not _port_closed(rec1["port"]) and not _port_closed(rec2["port"])
    finally:
        for q in (kept, nohup):
            q.terminate()
            q.wait(timeout=30)


@pytest.mark.skipif(_pwsh() is None, reason="needs pwsh")
def test_ssh_options_survive_powershell_file_invocation(tmp_path):
    """powershell -File passes -SshOption "-o","A=B" as ONE string "-o,A=B"; the script splits it back."""
    env = {**os.environ, "APPDATA": str(tmp_path / "appdata"), "USERPROFILE": str(tmp_path), "HOME": str(tmp_path)}
    ps1 = str(REPO_ROOT / "scripts" / "mace-gui-connect.ps1")
    base = ["-Gateway", "hpcc", "-Node", "dev-node-01", "-DryRun"]

    def argv(*cmd):
        r = subprocess.run([_pwsh(), "-NoProfile", *cmd], env=env, capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, r.stderr
        return [ln for ln in r.stdout.splitlines() if not ln.startswith(("# ", "ssh --% "))]

    joined = argv("-File", ps1, *base, "-SshOption", "-o,SetEnv=A=B,-o,HostKeyAlgorithms=a,b")
    spaced = argv("-Command", f"& '{ps1}' -Gateway hpcc -Node dev-node-01 -DryRun "
                              "-SshOption '-o','SetEnv=A=B','-o','HostKeyAlgorithms=a,b'")
    assert joined == spaced
    i = joined.index("SetEnv=A=B")
    assert joined[i - 1] == "-o" and joined[joined.index("HostKeyAlgorithms=a,b") - 1] == "-o"


def test_gateway_alias_from_ssh_config_in_the_nodes_domain(tmp_path):
    from mace.gui import cli
    cfg = tmp_path / "config"
    cfg.write_text("Host *\n  ServerAliveInterval 60\nHost hpc\n  HostName login.cluster.example.edu\n  User me\n"
                   "Host elsewhere\n  HostName ssh.other.org\n")
    assert cli.config_alias_for_domain("cluster.example.edu", cfg) == "hpc"
    assert cli.config_alias_for_domain("nowhere.net", cfg) is None
    assert cli.config_alias_for_domain("", cfg) is None
    cfg.write_text(cfg.read_text() + "Host hpc2\n  HostName login2.cluster.example.edu\n")
    assert cli.config_alias_for_domain("cluster.example.edu", cfg) is None          # two candidates: no guess


def test_placeholder_names_the_nodes_domain(monkeypatch, tmp_path):
    from mace.gui import cli
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("MACE_GUI_GATEWAY", raising=False)
    monkeypatch.setenv("MACE_SITE_GROUP_CONFIG", str(tmp_path / "none.toml"))
    monkeypatch.setattr(cli, "node_domain", lambda: "cluster.example.edu")
    text = "\n".join(cli.ssh_hint(26118, None))
    assert "LOGIN_HOST" in text and "this node is in cluster.example.edu" in text
