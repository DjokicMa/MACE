"""`mace --help`: the usage line's command list names every command.

`mace ions` dispatches before the top-level parser, like opt2d12, and the
help text lists it, but the parser's choices left it out, so the usage line
`{workflow,submit,...}` did not name it.
"""
import os
import re
import subprocess
import sys

from conftest import REPO_ROOT

MACE_CLI = REPO_ROOT / "mace_cli"


def test_usage_choices_include_ions_and_other_new_commands():
    env = dict(os.environ, MACE_NO_BANNER="1")
    proc = subprocess.run([sys.executable, str(MACE_CLI), "--no-banner", "--help"],
                          env=env, capture_output=True, text=True, timeout=120)
    out = proc.stdout + proc.stderr
    m = re.search(r"\{(workflow,[^}]*)\}", out)
    assert m, out
    choices = set(m.group(1).split(","))
    for name in ("ions", "defects", "phonopy", "opt2d12"):
        assert name in choices, (name, sorted(choices))
