"""`mace <command> -h` shows that command's help, never the general help.

Only a literal `--help` straight after the command was routed to the
command's help; `-h`, or a help flag after other arguments
(`mace preflight x -h`), fell to the top-level parser, which answered with
the general `mace` usage. A leading `--no-banner` also hid `defects -h`.
"""
import os
import subprocess
import sys

import pytest

from conftest import REPO_ROOT

MACE_CLI = REPO_ROOT / "mace_cli"


def _mace(*args):
    env = {k: v for k, v in os.environ.items()
           if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME")}
    r = subprocess.run([sys.executable, str(MACE_CLI), *args], env=env,
                       stdin=subprocess.DEVNULL, capture_output=True, text=True,
                       timeout=120)
    return r.returncode, r.stdout + r.stderr


@pytest.mark.parametrize("argv, usage", [
    (("preflight", "x", "-h"), "Usage: mace preflight"),
    (("preflight", "-h"), "Usage: mace preflight"),
    (("preopt", "-h"), "Usage: mace preopt"),
    (("preopt", "x.cif", "-h"), "Usage: mace preopt"),
    (("mlip", "fetch", "-h"), "usage: mace mlip fetch"),       # a subcommand's or tool's own help
    (("mlip", "-h"), "Usage: mace mlip"),
    (("calc", "opt", "--help"), "usage: mace calc opt"),
    (("--no-banner", "calc", "sp", "x.cif", "-h"), "usage: mace calc sp"),
    (("calc", "--help"), "Usage: mace calc"),
    (("analyze", "porosity", "--help"), "usage: mace analyze porosity"),
    (("analyze", "x.out", "--help"), "Usage: mace analyze"),
    (("analyze", "phonopy", "-h"), "usage: mace analyze phonopy"),
    (("--no-banner", "analyze", "elastic", "--help"), "usage: mace analyze elastic"),
    (("submit", "-h"), "Usage: mace submit"),
    (("submit", "somedir", "--help"), "Usage: mace submit"),
    (("workflow", "-h"), "Usage: mace workflow"),
    (("analyze", "-h"), "Usage: mace analyze"),
    (("--no-banner", "defects", "-h"), "mace defects {build,potc,analyze}"),
    (("--no-banner", "submit", "-h"), "Usage: mace submit"),
])
def test_command_help_flag(argv, usage):
    rc, text = _mace(*argv)
    assert rc == 0, text
    assert usage in text, text
    assert "Command to run" not in text, text      # not the general help


def test_general_help_still_general():
    rc, text = _mace("-h")
    assert rc == 0 and "Command to run" in text
