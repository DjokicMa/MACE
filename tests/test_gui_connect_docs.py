"""docs/gui.md's connection and session sections (PLAN_connect T8, T16): the
agreed headings, the which-method table, placeholders only, every `mace gui
connect` line in the docs parses (--dry-run through mace_cli), every script
flag the docs use exists in the scripts, and the messages quoted are the
ones the code prints."""
import os
import re
import shlex
import subprocess
import sys

from conftest import REPO_ROOT

DOC = (REPO_ROOT / "docs" / "gui.md").read_text()
SH = (REPO_ROOT / "scripts" / "mace-gui-connect.sh").read_text()
PS1 = (REPO_ROOT / "scripts" / "mace-gui-connect.ps1").read_text()
HEADS = re.findall(r"^#+ (.+)$", DOC, re.M)


def test_headings():
    for h in ("First run from your computer", "mace gui connect (Linux, macOS, WSL, Windows)", "Dry-run quoting",
              "First use of a node with --hop jump", "Windows (PowerShell)",
              "MobaXterm", "VS Code Remote-SSH", "Jump hosts and nodes that accept logins only from the login node",
              "Run it on a cluster and view it on your computer", "Open Folder", "Sign out and stop the server",
              "Long sessions on shared nodes", "Sign-in links, sessions and expiry", "Reporting a problem"):
        assert h in HEADS, h
    assert "First run from a laptop" not in HEADS and not any("HPCC (or any cluster)" in h for h in HEADS)


def test_which_method_table():
    for row in ("| MACE on your Linux or macOS computer (or WSL) | `mace gui connect` |",
                "| Linux or macOS without MACE | `mace-gui-connect.sh` |", "| Windows | `mace-gui-connect.ps1`",
                "| MobaXterm |", "| VS Code Remote-SSH |", "| Nothing but ssh |"):
        assert row in DOC, row


def test_placeholders_only():
    assert "hpcc.example.edu" in DOC and "dev-node-01" in DOC
    assert "msu.edu" not in DOC and "OnDemand" not in DOC and "dev-amd20" not in DOC.split("## Run it on a cluster")[1]
    assert "Unblock-File" in DOC and "ssh -V" in DOC


def _connect_lines():
    out = []
    for line in DOC.splitlines():
        m = re.search(r"(?:^|`)\s*(mace gui connect(?: [^`]*)?)", line.strip())
        if m and "connect.toml" not in m.group(1):
            cmd = m.group(1).strip().rstrip("`").strip()
            if cmd != "mace gui connect" and not cmd.endswith("(the default"):
                out.append(cmd)
    return sorted(set(out))


def test_every_connect_line_in_the_docs_parses(tmp_path):
    lines = _connect_lines()
    assert any("--gateway me@hpcc.example.edu --node dev-node-01" in ln for ln in lines), lines
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON") and k != "MACE_HOME"}
    env.update(XDG_CONFIG_HOME=str(tmp_path / "cfg"), HOME=str(tmp_path))
    for line in lines:
        argv = shlex.split(line.replace("<", "").replace(">", "").replace("…", ""))[3:]
        if any(a.startswith("--export-script") for a in argv):
            argv += ["--gateway", "g"]
            cwd = tmp_path
        else:
            argv += ["--dry-run"] + ([] if "--gateway" in argv or (argv and not argv[0].startswith("-"))
                                     else ["--gateway", "g"])
            cwd = tmp_path
        if argv and not argv[0].startswith("-") and argv[0] not in ("--dry-run",):
            argv = argv[1:]                                     # a profile name: none saved here
            if "--gateway" not in argv:
                argv += ["--gateway", "g"]
        r = subprocess.run([sys.executable, str(REPO_ROOT / "mace_cli"), "--no-banner", "gui", "connect", *argv],
                           env=env, cwd=str(cwd), capture_output=True, text=True, timeout=60,
                           stdin=subprocess.DEVNULL)
        assert r.returncode == 0, (line, r.stderr)


def test_script_flags_in_the_docs_exist():
    ps_params = set(re.findall(r"\[(?:string|int|string\[\]|switch)\]\$(\w+)", PS1))
    for flag in set(re.findall(r"(?<![\w-])-([A-Z][A-Za-z]+)\b", DOC)):
        if flag in ("ExecutionPolicy", "File", "Command", "NoProfile", "J", "L", "N", "Manual") or flag.isupper():
            continue
        assert flag in ps_params, flag
    assert "Manual" in ps_params
    sh_flags = set(re.findall(r"^\s+--([a-z-]+)[)=]", SH, re.M))
    for m in re.finditer(r"mace-gui-connect\.sh([^`\n]*)", DOC):
        for f in re.findall(r"--([a-z-]+)", m.group(1)):
            assert f in sh_flags, f


def test_quoted_messages_are_the_codes():
    code = "".join((REPO_ROOT / "mace" / "gui" / f).read_text() for f in
                   ("cli.py", "connect.py", "server.py", "managed.py", "tasks.py", "cpulimit.py", "redact.py"))
    code += (REPO_ROOT / "mace" / "gui" / "static" / "js" / "lazy" / "connect_ui.js").read_text()
    for msg in ("This sign-in link was already used or has expired.", "Hidden folders cannot be opened here",
                "Open a project folder inside your home, not your home folder itself.",
                "(waiting: approve the 2FA prompt if there is one. If ssh asks whether to continue connecting", "(the host-key yes/no question), check the fingerprint and answer yes", "is older than this connect",
                "did not accept a login through the gateway", "MACE GUI has stopped",
                "Reattached to the server still running on", "This node limits each process to",
                "stopped before this node's CPU-time limit", "The server on {node} was stopped from the browser."):
        assert msg in code, msg
    assert "The server on dev-node-01 was stopped from" in DOC
