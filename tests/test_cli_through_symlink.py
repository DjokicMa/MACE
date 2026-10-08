"""`mace` works when it is a symbolic link to mace_cli on your PATH.

Putting a link such as ~/bin/mace -> <checkout>/mace_cli on PATH is the
simplest install that needs no shell startup file and no MACE_HOME. The
script found its Crystal_d12/Crystal_d3 tools next to the link instead of
next to itself, so every command that loads them stopped with
"No module named ...". Each command runs as a subprocess through a real
link, from another folder, with MACE_HOME and PYTHONPATH unset.
"""
import os
import subprocess
import sys

import pytest

from conftest import REPO_ROOT


@pytest.fixture(scope="module")
def link(tmp_path_factory):
    d = tmp_path_factory.mktemp("bin")
    (d / "mace").symlink_to(REPO_ROOT / "mace_cli")
    return d / "mace"


@pytest.mark.parametrize("argv", [
    ("version",), ("convert", "--help"), ("opt2d12", "--help"), ("opt2d3", "--help"),
    ("opt2cif", "--help"), ("ions", "--help"), ("monitor", "--help"),
])
def test_command_runs_through_a_link(link, argv, tmp_path):
    env = {k: v for k, v in os.environ.items()
           if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH")}
    r = subprocess.run([sys.executable, str(link), "--no-banner", *argv], env=env, cwd=str(tmp_path),
                       stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "No module named" not in r.stderr and "Traceback" not in r.stderr
