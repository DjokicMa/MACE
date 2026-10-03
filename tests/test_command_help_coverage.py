"""Every `mace <command> --help` answers, and from one place only.

mace_cli kept hand-written help for convert, opt2d12, opt2d3, opt2cif and
completion that was never shown - `--help` on those goes straight to the
script's own argparse - so the two drifted apart (the hand-written text
held the examples). The examples now live in the scripts' argparse help,
and a command has either curated text in mace_cli or a forwarding path,
never both.
"""
import json
import os
import re
import subprocess
import sys

import pytest

from conftest import REPO_ROOT

MACE_CLI = REPO_ROOT / "mace_cli"


def _env():
    return {k: v for k, v in os.environ.items()
            if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH")}


def _needs(command):
    if command == "convert":
        pytest.importorskip("ase", reason="NewCifToD12 imports ase (absent in CI)")


def _mace(*args):
    r = subprocess.run([sys.executable, str(MACE_CLI), *args], env=_env(),
                       cwd=str(REPO_ROOT), stdin=subprocess.DEVNULL,
                       capture_output=True, text=True, timeout=120)
    return r.returncode, r.stdout + r.stderr


def _commands():
    rc, text = _mace("--no-banner", "--help")
    assert rc == 0, text
    m = re.search(r"\{(workflow,[^}]*)\}", text)
    assert m, text
    return m.group(1).split(",")


COMMANDS = _commands()


def test_command_list_is_complete():
    for name in ("convert", "opt2d12", "opt2d3", "opt2cif", "completion", "ions",
                 "defects", "phonopy", "workflow", "version"):
        assert name in COMMANDS, (name, COMMANDS)


@pytest.mark.parametrize("command", COMMANDS)
def test_every_command_help_answers_and_names_itself(command):
    _needs(command)
    rc, text = _mace(command, "--help")
    assert rc == 0, text
    assert f"mace {command}" in text, text
    assert "Command to run" not in text, text          # not the general help


def test_curated_and_forwarded_commands_do_not_overlap():
    code = ("import importlib.machinery, importlib.util, json, sys\n"
            f"loader = importlib.machinery.SourceFileLoader('mace_cli_mod', {str(MACE_CLI)!r})\n"
            "spec = importlib.util.spec_from_loader('mace_cli_mod', loader)\n"
            "m = importlib.util.module_from_spec(spec)\n"
            "loader.exec_module(m)\n"
            "print(json.dumps({'curated': sorted(m.COMMAND_HELP),"
            " 'forwarded': sorted(m.PASSTHROUGH_COMMANDS)}))\n")
    r = subprocess.run([sys.executable, "-c", code], env=_env(), cwd=str(REPO_ROOT),
                       stdin=subprocess.DEVNULL, capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0, r.stderr
    data = json.loads(r.stdout.strip().splitlines()[-1])
    curated, forwarded = set(data["curated"]), set(data["forwarded"])
    assert not curated & forwarded, curated & forwarded
    assert set(COMMANDS) == curated | forwarded, set(COMMANDS) ^ (curated | forwarded)


@pytest.mark.parametrize("command, example", [
    ("convert", "mace convert --cif_dir . --batch --options_file myopts.json"),
    ("opt2d12", "mace opt2d12 --out-file optimized.out --calc-type SP --non-interactive"),
    ("opt2d12", "mace opt2d12 --directory opts --config-file sp_template.json --yes"),
    ("opt2d3", "mace opt2d3 --input optimized.out --calc-type BAND"),
    ("opt2d3", "mace opt2d3 --batch --calc-type BAND"),
    ("opt2cif", "mace opt2cif . --output-dir cifs/"),
    ("completion", "mace completion --organize --output-dir results"),
])
def test_forwarded_help_carries_the_examples(command, example):
    _needs(command)
    rc, text = _mace(command, "--help")
    assert rc == 0, text
    assert example in text, text


@pytest.mark.parametrize("command, note", [
    ("convert", "not individual file arguments"),
    ("opt2d12", "There is no positional argument"),
    ("opt2d3", "There is no positional argument"),
    ("opt2cif", "3D, 2D, 1D and 0D"),
    ("completion", "Status categories"),
])
def test_forwarded_help_carries_the_usage_notes(command, note):
    _needs(command)
    rc, text = _mace(command, "--help")
    assert rc == 0, text
    assert note in " ".join(text.split()), text


def test_general_help_points_at_command_help():
    rc, text = _mace("--no-banner", "--help")
    assert rc == 0
    assert "mace <command> --help" in text
