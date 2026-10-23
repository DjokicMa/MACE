"""`mace update --remote` through a jump host (-J, or a ProxyJump in the ssh
config) gets mace gui connect's PROMPT_TIMEOUT_S: the gateway's ConnectTimeout
also covers the login to the jump host, its passphrase prompt or 2FA, which a
20 s limit cut off. A first jump host that does not answer still fails fast,
from connect.py's TCP precheck. Through mace_cli with a fake ssh whose `ssh -G`
points the jump host at a local socket."""
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from mace import update as U
from mace.gui import connect as C

REPO = Path(__file__).resolve().parents[1]

FAKE_SSH = """#!/usr/bin/env bash
if [ "$1" = "-G" ]; then
  case "$2" in
    me@bastion) printf 'hostname 127.0.0.1\\nport %s\\nconnecttimeout 1\\nbatchmode %s\\n' "$FAKE_PORT" "$FAKE_BATCH" ;;
    me@gw.example.edu) [ -n "$FAKE_GW_PJ" ] && printf 'proxyjump %s\\n' "$FAKE_GW_PJ" ;;
  esac
  exit 0
fi
echo "$*" >> "$FAKE_SSH_LOG"
echo "remote update ran"
"""


@pytest.fixture
def listener():
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(0)
    held = []
    yield srv, held
    for s in held:
        s.close()
    srv.close()


def run(tmp_path, port, profile, batch="no", gw_pj=""):
    fb = tmp_path / "fakebin"
    fb.mkdir(exist_ok=True)
    (fb / "ssh").write_text(FAKE_SSH)
    (fb / "ssh").chmod(0o755)
    cfg = tmp_path / "cfg"
    C.save_profiles({"defaults": {"profile": "hpcc"}, "profiles": {"hpcc": profile}}, cfg / "mace" / "connect.toml")
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON") and k != "MACE_HOME"}
    env.update(PATH=f"{fb}:{env['PATH']}", XDG_CONFIG_HOME=str(cfg), FAKE_PORT=str(port), FAKE_BATCH=batch,
               FAKE_GW_PJ=gw_pj, FAKE_SSH_LOG=str(tmp_path / "ssh.log"))
    r = subprocess.run([sys.executable, str(REPO / "mace_cli"), "--no-banner", "update", "--remote", "--remote-only",
                        "--dry-run"], env=env, capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL)
    log = (tmp_path / "ssh.log").read_text() if (tmp_path / "ssh.log").exists() else ""
    return r, log


JUMP = {"gateway": "me@gw.example.edu", "jump": "me@bastion"}


def test_a_jump_host_that_may_prompt_gets_the_long_timeout(tmp_path, listener):
    srv, _ = listener
    r, log = run(tmp_path, srv.getsockname()[1], JUMP)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "remote update ran" in r.stdout
    assert f"ConnectTimeout={C.PROMPT_TIMEOUT_S}" in log and "-J me@bastion" in log
    assert f"ConnectTimeout={C.CONNECT_TIMEOUT_S}" not in log


def test_a_proxyjump_in_the_ssh_config_counts_too(tmp_path, listener):
    srv, _ = listener
    r, log = run(tmp_path, srv.getsockname()[1], {"gateway": "me@gw.example.edu"}, gw_pj="me@bastion")
    assert r.returncode == 0, r.stdout + r.stderr
    assert f"ConnectTimeout={C.PROMPT_TIMEOUT_S}" in log and "-J" not in log


def test_a_batchmode_jump_host_and_no_jump_keep_the_short_timeout(tmp_path, listener):
    srv, _ = listener
    r, log = run(tmp_path, srv.getsockname()[1], JUMP, batch="yes")
    assert r.returncode == 0, r.stdout + r.stderr
    assert f"ConnectTimeout={C.CONNECT_TIMEOUT_S}" in log
    (tmp_path / "ssh.log").unlink()
    r, log = run(tmp_path, srv.getsockname()[1], {"gateway": "me@gw.example.edu"})
    assert r.returncode == 0, r.stdout + r.stderr
    assert f"ConnectTimeout={C.CONNECT_TIMEOUT_S}" in log


def test_an_unreachable_first_hop_fails_fast(tmp_path, listener):
    srv, held = listener
    port = srv.getsockname()[1]
    # a listen backlog that is full drops the next SYN: the TCP connection times out
    c = socket.create_connection(("127.0.0.1", port), timeout=2)
    held.append(c)
    r, log = run(tmp_path, port, JUMP)
    assert r.returncode == 255, r.stdout + r.stderr
    assert "cannot reach host 127.0.0.1 (timeout)" in r.stdout
    assert "mace gui connect hpcc --check" in r.stdout
    assert log == ""                                 # the update's ssh never started


def test_remote_argv_uses_connects_helpers():
    """No timeout logic of its own: the ssh -G answers decide, through connect.py."""
    def runner(argv, **kw):
        out = "batchmode no\n" if argv[:2] == ["ssh", "-G"] and argv[2] == "me@bastion" else ""
        return subprocess.CompletedProcess(argv, 0, out, "")
    argv = U.remote_argv(JUMP, [], False, runner=runner)
    assert f"ConnectTimeout={C.PROMPT_TIMEOUT_S}" in argv
    argv = U.remote_argv({**JUMP, "ssh_options": ["-o", "ConnectTimeout=300"]}, [], False, runner=runner)
    assert f"ConnectTimeout={C.PROMPT_TIMEOUT_S}" not in argv and "ConnectTimeout=300" in argv


# ---- the standalone scripts' --update-remote / -UpdateRemote: the same wait and precheck

SCRIPT_SSH = """#!/bin/sh
if [ "$1" = -G ]; then
  case "$2" in me@bastion) printf 'hostname 127.0.0.1\\nport %s\\nconnecttimeout 1\\n' "$FAKE_PORT" ;; esac
  exit 0
fi
printf '%s\\n' "$*" >> "$LOG"
exit 0
"""
SH = REPO / "scripts" / "mace-gui-connect.sh"
PS1 = REPO / "scripts" / "mace-gui-connect.ps1"


def _accepting(srv):
    import threading

    def loop():
        while True:
            try:
                c, _ = srv.accept()
            except OSError:
                return
            c.close()
    threading.Thread(target=loop, daemon=True).start()


def run_script(tmp_path, port, kind):
    fb = tmp_path / "fb"
    fb.mkdir(exist_ok=True)
    (fb / "ssh").write_text(SCRIPT_SSH)
    (fb / "ssh").chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k not in ("BROWSER", "APPDATA", "USERPROFILE")}
    env.update(PATH=f"{fb}:/usr/bin:/bin", HOME=str(tmp_path), XDG_CONFIG_HOME=str(tmp_path / "c"),
               LOG=str(tmp_path / "log"), FAKE_PORT=str(port))
    if kind == "sh":
        cmd = ["bash", str(SH), "--gateway", "me@gw", "--node", "n1", "--jump", "me@bastion", "--no-open",
               "--update-remote", "preview"]
    else:
        cmd = [__import__("shutil").which("pwsh"), "-NoProfile", "-File", str(PS1), "-Gateway", "me@gw", "-Node",
               "n1", "-Jump", "me@bastion", "-Manual", "-UpdateRemote", "preview"]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL)
    log = (tmp_path / "log").read_text().splitlines() if (tmp_path / "log").exists() else []
    return r, log


KINDS = ["sh", pytest.param("ps1", marks=pytest.mark.skipif(not __import__("shutil").which("pwsh"),
                                                            reason="needs pwsh"))]


@pytest.mark.parametrize("kind", KINDS)
def test_script_update_through_a_jump_host_waits_for_its_prompt(tmp_path, listener, kind):
    srv, _ = listener
    _accepting(srv)
    r, log = run_script(tmp_path, srv.getsockname()[1], kind)
    assert log and "-J me@bastion" in log[0] and " me@gw exec bash -lc " in log[0], r.stdout + r.stderr
    assert f"ConnectTimeout={C.PROMPT_TIMEOUT_S}" in log[0]


@pytest.mark.parametrize("kind", KINDS)
def test_script_update_with_an_unreachable_first_hop_fails_fast(tmp_path, listener, kind):
    srv, held = listener
    held.append(socket.create_connection(srv.getsockname(), timeout=2))
    r, log = run_script(tmp_path, srv.getsockname()[1], kind)
    out = r.stdout + r.stderr
    assert "cannot reach host 127.0.0.1 (timeout)" in out and "mace update did not run" in out, out
    assert not any("update --help" in ln for ln in log)
