"""`mace opt2d3 --help` says `usage: mace opt2d3`, not the script's file name.

The commands mace_cli hands to a script's own argparse (convert, opt2d12,
opt2d3, opt2cif, completion) set argv[0] to the script file, so their help
read `usage: CRYSTALOptToD3.py ...` - a name the user never typed. Run
directly, a script keeps its own name.
"""
import os
import subprocess
import sys

import pytest

from conftest import REPO_ROOT

MACE_CLI = REPO_ROOT / "mace_cli"

FORWARDED = {
    "convert": "Crystal_d12/NewCifToD12.py",
    "opt2d12": "Crystal_d12/CRYSTALOptToD12.py",
    "opt2d3": "Crystal_d3/CRYSTALOptToD3.py",
    "opt2cif": "Crystal_d12/CrystalOutToCif.py",
    "completion": "mace/completion_checker.py",
}


def _env():
    return {k: v for k, v in os.environ.items()
            if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH")}


def _needs(command):
    if command == "convert":
        pytest.importorskip("ase", reason="NewCifToD12 imports ase (absent in CI)")


def _run(argv):
    r = subprocess.run([sys.executable, *argv], env=_env(), cwd=str(REPO_ROOT),
                       stdin=subprocess.DEVNULL, capture_output=True, text=True,
                       timeout=120)
    return r.returncode, r.stdout + r.stderr


@pytest.mark.parametrize("command, script", sorted(FORWARDED.items()))
def test_mace_help_names_the_mace_command(command, script):
    _needs(command)
    rc, text = _run([str(MACE_CLI), command, "--help"])
    assert rc == 0, text
    assert text.lstrip().startswith(f"usage: mace {command} "), text[:300]
    assert os.path.basename(script) not in text, text


@pytest.mark.parametrize("command, script", sorted(FORWARDED.items()))
def test_script_run_directly_keeps_its_own_name(command, script):
    _needs(command)
    rc, text = _run([str(REPO_ROOT / script), "--help"])
    assert rc == 0, text
    assert text.lstrip().startswith(f"usage: {os.path.basename(script)} "), text[:300]
