"""Plain `mace` commands never import the GUI: `python -X importtime mace_cli`
as a subprocess (the real invocation path) lists no mace.gui module."""
import os
import subprocess
import sys

import pytest

from conftest import REPO_ROOT

MACE_CLI = REPO_ROOT / "mace_cli"


@pytest.mark.parametrize("args", [["--help"], ["version"], ["monitor", "--help"], ["submit", "--help"],
                                  ["gui", "-h"], ["gui", "--help"]],
                         ids=lambda a: " ".join(a))
def test_command_imports_no_gui_module(args, tmp_path):
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "MACE_HOME", "COLORTERM", "TERM")}
    env.update(PYTHONDONTWRITEBYTECODE="1", NO_COLOR="1")
    r = subprocess.run([sys.executable, "-X", "importtime", str(MACE_CLI), "--no-banner", *args],
                       capture_output=True, text=True, env=env, cwd=str(tmp_path), timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr[-2000:]
    imported = [ln.split("|")[-1].strip() for ln in r.stderr.splitlines() if ln.startswith("import time:")]
    assert len(imported) > 20, r.stderr[-2000:]                    # importtime ran
    gui = [m for m in imported if m == "mace.gui" or m.startswith("mace.gui.")]
    assert not gui, gui
