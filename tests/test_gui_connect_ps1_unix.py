"""scripts/mace-gui-connect.ps1 under pwsh on Linux/macOS: no %USERPROFILE% or
%APPDATA% there, so the profile is read from $XDG_CONFIG_HOME or ~/.config
(as mace gui connect keeps it) and the email-like gateway check reads
~/.ssh/config from $HOME. Both used to stop the script with "Cannot bind
argument to parameter 'Path' because it is null"."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PS1 = REPO / "scripts" / "mace-gui-connect.ps1"


def test_no_windows_only_folders_outside_the_windows_branch():
    text = PS1.read_text()
    assert "$env:USERPROFILE" not in text
    body = text[text.index("function Connect-Toml"):text.index("function Read-Profile")]
    assert text.count("$env:APPDATA \"mace") == body.count("$env:APPDATA \"mace") == 1


def _run(home, gateway=None, extra=()):
    env = {k: v for k, v in os.environ.items() if k not in ("USERPROFILE", "APPDATA", "XDG_CONFIG_HOME", "BROWSER")}
    env["HOME"] = str(home)
    args = [shutil.which("pwsh"), "-NoProfile", "-File", str(PS1), "-DryRun", *extra]
    if gateway:
        args += ["-Gateway", gateway, "-Node", "n1"]
    return subprocess.run(args, env=env, capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL)


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="needs pwsh")
def test_email_like_gateway_from_the_xdg_profile_warns_and_runs(tmp_path):
    conf = tmp_path / ".config" / "mace" / "connect.toml"
    conf.parent.mkdir(parents=True)
    conf.write_text('[defaults]\nprofile = "p"\n\n[profiles.p]\ngateway = "me@example.edu"\nnode = "n1"\n')
    r = _run(tmp_path)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert "is null" not in out
    assert "me@example.edu looks like an email address" in out
    assert str(conf) in out.replace("\n", "")
    assert "127.0.0.1:8765:127.0.0.1:40000" in out


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="needs pwsh")
def test_ssh_config_host_pattern_silences_the_email_warning(tmp_path):
    r = _run(tmp_path, "me@gw")
    assert r.returncode == 0, r.stdout + r.stderr
    (tmp_path / ".ssh").mkdir()
    (tmp_path / ".ssh" / "config").write_text("Host *.edu\n  User me\n")
    r = _run(tmp_path, "me@example.edu")
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert "is null" not in out and "looks like an email" not in out
