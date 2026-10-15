"""The "From your computer" hint `mace gui` prints on a cluster node (1.1.5
final, MobaXterm hand test): every command line in it can be pasted into a
shell as is (no <placeholder>, which bash reads as a redirect), the gateway
comes from MACE_GUI_GATEWAY or site.toml's [gui] gateway when one is set,
and a computer without MACE gets the standalone script."""
import re
import shlex
import subprocess
import sys

import pytest

from mace.gui import cli
from mace.gui import connect as C

COMMAND = re.compile(r"^  (mace|ssh|bash|python|\.\\|powershell) ")


@pytest.fixture
def no_site(tmp_path, monkeypatch):
    monkeypatch.delenv("MACE_GUI_GATEWAY", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("MACE_SITE_GROUP_CONFIG", str(tmp_path / "none.toml"))
    monkeypatch.delenv("MACE_SITE_CONFIG", raising=False)
    return tmp_path


def _commands(lines):
    return [ln.strip() for ln in lines if COMMAND.match(ln)]


def test_hint_without_a_gateway_is_paste_safe(no_site):
    lines = cli.ssh_hint(8790, None, "/mnt/home/me/runs")
    cmds = _commands(lines)
    assert len(cmds) >= 3, lines
    for c in cmds:
        assert "<" not in c and ">" not in c, c
        shlex.split(c)
    assert any("LOGIN_HOST" in c for c in cmds)
    assert any("replace LOGIN_HOST" in ln for ln in lines)
    assert "<gateway>" not in "\n".join(lines)


def test_the_placeholder_line_runs_in_bash_without_a_redirect_error(no_site, tmp_path):
    ssh_line = next(c for c in _commands(cli.ssh_hint(8790, None)) if c.startswith("ssh "))
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "ssh").write_text("#!/bin/sh\nexit 0\n")
    (fake / "ssh").chmod(0o755)
    r = subprocess.run(["bash", "-c", ssh_line], capture_output=True, text=True, cwd=tmp_path,
                       env={"PATH": f"{fake}:/usr/bin:/bin"})
    assert r.returncode == 0 and "No such file or directory" not in r.stderr, r.stderr


def test_site_toml_names_the_gateway(no_site):
    site = no_site / "xdg" / "mace" / "site.toml"
    site.parent.mkdir(parents=True)
    site.write_text('[gui]\ngateway = "login.example.org"\n')
    lines = cli.ssh_hint(8790, None)
    assert any("@login.example.org" in c for c in _commands(lines)), lines
    assert "LOGIN_HOST" not in "\n".join(lines)
    assert "@login.example.org" in cli.connect_line(None)


def test_environment_and_flag_beat_site_toml(no_site, monkeypatch):
    site = no_site / "xdg" / "mace" / "site.toml"
    site.parent.mkdir(parents=True)
    site.write_text('[gui]\ngateway = "login.example.org"\n')
    monkeypatch.setenv("MACE_GUI_GATEWAY", "env.example.org")
    assert "@env.example.org" in cli.connect_line(None)
    assert "@flag.example.org" in cli.connect_line("flag.example.org")


@pytest.mark.parametrize("bad", ['"login.example.org; rm -rf ~"', '"a b"', '"<gw>"', "3"])
def test_an_unusable_site_gateway_is_ignored(no_site, bad):
    site = no_site / "xdg" / "mace" / "site.toml"
    site.parent.mkdir(parents=True)
    site.write_text(f"[gui]\ngateway = {bad}\n")
    assert "LOGIN_HOST" in cli.connect_line(None)


def test_gui_table_gives_no_unknown_table_warning(no_site):
    from mace.submission import site
    p = no_site / "xdg" / "mace" / "site.toml"
    p.parent.mkdir(parents=True)
    p.write_text('[gui]\ngateway = "login.example.org"\n')
    s = site.load()
    assert all(w.startswith("MACE_SITE_GROUP_CONFIG names") for w in s.warnings), s.warnings


def test_hint_offers_the_standalone_script_and_it_exports(no_site, tmp_path):
    lines = cli.ssh_hint(8790, "me@login.example.org", "/data/runs")
    text = "\n".join(lines)
    assert "MobaXterm" in text and "PowerShell" in text
    export = next(c for c in _commands(lines) if "--export-script" in c)
    argv = shlex.split(export)
    assert argv[:3] == ["mace", "gui", "connect"]
    out = tmp_path / "c.sh"
    r = C.main(argv[3:argv.index("-o")] + ["-o", str(out)])
    assert r == 0 and "D_GATEWAY=me@login.example.org" in out.read_text()


def test_gui_dialog_fallbacks_are_paste_safe():
    from pathlib import Path
    js = (Path(cli.__file__).parent / "static" / "js" / "lazy" / "connect_ui.js").read_text()
    for m in re.finditer(r"line\(\"cn-\w+\", [^\n]*", js):
        assert "<user>" not in m.group(0) and "<gateway>" not in m.group(0) and "<folder>" not in m.group(0), m.group(0)
    assert "--export-script" in js


def test_hint_says_the_tunnel_must_end_at_127_0_0_1(no_site):
    text = "\n".join(cli.ssh_hint(8790, "me@login.example.org"))
    assert "127.0.0.1:8790" in text and "listens only on" in text
