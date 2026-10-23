"""`mace --help` lists every command once, in groups; `mace help GROUP`.

The registry (COMMANDS, COMMAND_GROUPS in mace_cli) is the one place a
command is named: the parser's choices, the grouped list and `mace help`
all come from it. Every check runs the real `mace_cli` in a subprocess.
"""
import json
import os
import re
import subprocess
import sys

import pytest

from conftest import REPO_ROOT

MACE_CLI = REPO_ROOT / "mace_cli"
DOCS_CLI = REPO_ROOT / "docs" / "cli.md"


def _env():
    return {k: v for k, v in os.environ.items()
            if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH")}


def _mace(*args):
    r = subprocess.run([sys.executable, str(MACE_CLI), *args], env=_env(), cwd=str(REPO_ROOT),
                       stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120)
    return r.returncode, r.stdout, r.stderr


_REG = None


def _registry():
    global _REG
    if _REG is None:
        code = ("import importlib.machinery, importlib.util, json\n"
                f"loader = importlib.machinery.SourceFileLoader('mace_cli_mod', {str(MACE_CLI)!r})\n"
                "spec = importlib.util.spec_from_loader('mace_cli_mod', loader)\n"
                "m = importlib.util.module_from_spec(spec)\n"
                "loader.exec_module(m)\n"
                "print(json.dumps({'commands': list(m.COMMANDS), 'groups': [[k, t, p, [e[0] for e in es]]"
                " for k, t, p, es in m.COMMAND_GROUPS], 'epilog': m.GENERAL_HELP_EPILOG}))\n")
        r = subprocess.run([sys.executable, "-c", code], env=_env(), cwd=str(REPO_ROOT),
                           stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, r.stderr
        _REG = json.loads(r.stdout.strip().splitlines()[-1])
    return _REG


def _line(name):
    return re.compile(rf"^  {re.escape(name):<12}\S", re.M)


def test_every_command_is_in_exactly_one_group():
    reg = _registry()
    grouped = [n for _, _, _, names in reg["groups"] for n in names]
    dup = sorted({n for n in grouped if grouped.count(n) > 1})
    assert not dup, f"in more than one group: {dup}"
    assert set(grouped) == set(reg["commands"]), (
        f"add it to COMMAND_GROUPS in mace_cli (see the comment above COMMANDS): "
        f"ungrouped {sorted(set(reg['commands']) - set(grouped))}, unknown {sorted(set(grouped) - set(reg['commands']))}")
    assert len(reg["commands"]) == len(set(reg["commands"]))


def test_general_help_lists_every_command_once_in_group_order():
    rc, out, err = _mace("--no-banner", "--help")
    assert rc == 0, err
    reg = _registry()
    pos = []
    for key, title, purpose, names in reg["groups"]:
        heads = [ln for ln in out.splitlines() if ln.startswith(title)]
        assert len(heads) == 1, (title, heads)
        pos.append(out.index(heads[0] + "\n"))
    assert pos == sorted(pos)
    order = []
    for _, _, _, names in reg["groups"]:
        for n in names:
            hits = _line(n).findall(out)
            assert len(hits) == 1, (n, hits)
            order.append(_line(n).search(out).start())
    assert order == sorted(order)
    # the site line names every action `mace site` accepts
    site_line = next(ln for ln in out.splitlines() if ln.startswith("  site        "))
    assert site_line.startswith("  site        Show job-script site settings")
    from mace.submission import site_cli
    actions = next(a for a in site_cli.build_parser()._actions if a.dest == "action").choices
    assert "--init" in site_line
    for word in actions:
        if word != "show":                 # the default action is the command itself
            assert word in site_line, (word, site_line)


def test_general_help_epilog_fits_80_columns():
    long = [ln for ln in _registry()["epilog"].splitlines() if len(ln) > 79]
    assert not long, long


def test_usage_does_not_repeat_the_command_list():
    rc, out, err = _mace("--no-banner", "--help")
    assert "{workflow," not in out and "[COMMAND]" in out and "Command to run" in out
    rc, out, err = _mace("--no-banner", "nosuch")
    assert rc == 2 and "invalid choice: 'nosuch'" in err


def test_quick_start_order():
    rc, out, err = _mace("--no-banner", "--help")
    block = out[out.index("Quick start:"):out.index("More help:")]
    cmds = [ln.split()[1] for ln in block.splitlines()[1:] if ln.strip().startswith("mace ")]
    assert cmds.index("gui") < cmds.index("workflow")
    assert cmds.index("site") < cmds.index("submit")
    assert set(cmds) <= set(_registry()["commands"]), cmds


@pytest.mark.parametrize("key", ["workflows", "monitoring", "manual", "results", "special", "gui", "setup",
                                 "aliases"])
def test_help_group_prints_only_that_group(key):
    reg = _registry()
    groups = reg["groups"]
    i = [g[0] for g in groups].index(key)
    rc, out, err = _mace("help", key)
    assert rc == 0, err
    assert out.startswith(groups[i][1])
    for n in groups[i][3]:
        assert _line(n).search(out), (n, out)
    other = groups[(i + 1) % len(groups)][3][0]
    assert not _line(other).search(out), (other, out)
    assert "___" not in out and "MACE - Mendoza" not in out           # no banner


def test_help_with_no_word_is_the_general_help():
    a, b = _mace("help"), _mace("--no-banner", "--help")
    assert a[0] == b[0] == 0
    assert a[1] == b[1]


def test_help_gui_is_the_group():
    rc, out, err = _mace("help", "gui")
    assert rc == 0
    for w in ("GUI", "--slurm", "connect", "login-url", "diagnostics", "mace gui --help for its options."):
        assert w in out, (w, out)
    assert out != _mace("gui", "--help")[1]


@pytest.mark.parametrize("command", ["submit", "cluster", "convert", "site", "status", "defects", "opt2cif"])
def test_help_command_is_the_command_help(command):
    a, b = _mace("help", command), _mace(command, "--help")
    assert a[0] == b[0] == 0, (a[2], b[2])
    assert a[1] == b[1]


def test_help_unknown_word_exits_2_with_groups():
    rc, out, err = _mace("help", "nosuch")
    assert rc == 2 and "Unknown group or command 'nosuch'" in err
    assert "workflows, monitoring, manual, results, special, gui, setup, aliases" in err


@pytest.mark.parametrize("argv", [("help", "-h"), ("help", "submit", "--help"), ("help", "monitoring", "-h")])
def test_help_dash_h_prints_the_help_command_text(argv):
    want = _mace("help", "--help")
    got = _mace(*argv)
    assert got[0] == want[0] == 0
    assert got[1] == want[1]
    assert "-h/--help anywhere after `help` prints this text" in " ".join(got[1].split()).replace("\n", " ") or \
        "-h/--help anywhere after `help`" in got[1]


@pytest.mark.parametrize("argv, word", [
    (("status", "--help"), "monitor --status"),
    (("queue", "--help"), "manager"),
    (("--no-banner", "defects", "-h"), "mace defects"),
    (("cluster", "-h"), "mace cluster"),
    (("gui", "connect", "--help"), "connect"),
    (("analyze", "phonopy", "--help"), "phonopy"),
])
def test_old_invocations_still_answer(argv, word):
    rc, out, err = _mace(*argv)
    assert rc == 0, err
    assert word in out and "Commands, grouped from automated workflows" not in out


def test_docs_cli_follows_the_groups():
    text = DOCS_CLI.read_text()
    start = text.index("## Commands by group")
    end = text.index("\n## ", start + 5)
    part = text[start:end]
    heads = re.findall(r"^### (.+)$", part, re.M)
    groups = _registry()["groups"]
    assert heads == [g[1] for g in groups]
    blocks = re.split(r"^### .+$", part, flags=re.M)[1:]
    for (key, title, purpose, names), block in zip(groups, blocks):
        rows = re.findall(r"^\| `mace ([a-z0-9]+)` \|", block, re.M)
        assert rows == names, (title, rows, names)
    assert text.count("\n## Site settings (`mace site`, site.toml)\n") == 1
    assert text.count("\n## mace cluster\n") == 1
    assert text.index("## Commands by group") < text.index("## Site settings")
