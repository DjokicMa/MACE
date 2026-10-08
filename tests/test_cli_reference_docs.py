"""docs/cli.md's Command reference has one section per command, and each
section names exactly the options its `mace <command> --help` names.

Every command of the registry (COMMANDS in mace_cli) gets a section headed
`### mace <command>`, in the order of `mace --help`. The options are compared
as sets of `--long-flags`: a flag the help shows but the section leaves out,
or a flag the section names but no help shows any more, fails the test. The
help is the real `mace_cli` run in a subprocess, with a wide COLUMNS so
argparse never wraps a flag across two lines.
"""
import os
import re
import subprocess
import sys

import pytest

from conftest import REPO_ROOT

MACE_CLI = REPO_ROOT / "mace_cli"
DOC = (REPO_ROOT / "docs" / "cli.md").read_text()

# The help screens of a command: its own, then those of its subcommands.
SUBHELPS = {
    "defects": [("build",), ("potc",), ("analyze",)],
    "phonopy": [("generate",), ("collect",)],
    "gui": [("--slurm",), ("connect",), ("login-url",), ("diagnostics",)],
}

FLAG = re.compile(r"(?<![\w-])--[a-z][a-z0-9_]*(?:-[a-z0-9_]+)*(?![\w-])")


def _env():
    env = {k: v for k, v in os.environ.items()
           if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH")}
    env["COLUMNS"] = "1000"
    return env


def _help(*argv):
    r = subprocess.run([sys.executable, str(MACE_CLI), "--no-banner", *argv, "--help"], env=_env(),
                       cwd=str(REPO_ROOT), stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, (argv, r.stderr)
    return r.stdout


def _commands():
    code = ("import importlib.machinery, importlib.util\n"
            f"loader = importlib.machinery.SourceFileLoader('mace_cli_mod', {str(MACE_CLI)!r})\n"
            "spec = importlib.util.spec_from_loader('mace_cli_mod', loader)\n"
            "m = importlib.util.module_from_spec(spec)\n"
            "loader.exec_module(m)\n"
            "print(' '.join(e[0] for k, t, p, es in m.COMMAND_GROUPS for e in es))\n")
    r = subprocess.run([sys.executable, "-c", code], env=_env(), cwd=str(REPO_ROOT),
                       stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    return r.stdout.split()


COMMANDS = _commands()


def _reference():
    start = DOC.index("\n## Command reference\n")
    end = DOC.index("\n## ", start + 5)
    return DOC[start:end]


def _sections():
    ref = _reference()
    heads = list(re.finditer(r"^### mace ([a-z0-9]+)(?: \(summary\))?$", ref, re.M))
    out = {}
    for i, m in enumerate(heads):
        stop = heads[i + 1].start() if i + 1 < len(heads) else len(ref)
        out[m.group(1)] = ref[m.end():stop]
    return [m.group(1) for m in heads], out


def test_one_section_per_command_in_help_order():
    order, _ = _sections()
    assert order == COMMANDS


@pytest.mark.parametrize("command", COMMANDS)
def test_section_names_exactly_the_help_flags(command):
    text = _help(command)
    for sub in SUBHELPS.get(command, []):
        text += _help(command, *sub)
    want = set(FLAG.findall(text)) - {"--help"}
    _, sections = _sections()
    have = set(FLAG.findall(sections[command])) - {"--help"}
    assert not want - have, f"mace {command}: document {sorted(want - have)} in docs/cli.md"
    assert not have - want, f"mace {command}: docs/cli.md names {sorted(have - want)}, which its help does not"


def test_group_table_links_and_reference_links_resolve():
    anchors = set()
    for h in re.findall(r"^#+ (.+)$", DOC, re.M):
        s = re.sub(r"[^\w\- ]", "", re.sub(r"[`*_]", "", h.strip().lower())).replace(" ", "-")
        anchors.add(s)
    missing = [a for a in re.findall(r"\]\(#([^)]+)\)", DOC) if a not in anchors]
    assert not missing, missing
