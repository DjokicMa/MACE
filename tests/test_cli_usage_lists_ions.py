"""`mace --help`: the grouped command list names every command.

`mace ions` dispatches before the top-level parser, like opt2d12, and the
help text lists it, but the parser's choices once left it out. The usage
line now reads COMMAND; the grouped list below it names each command once,
and the parser's choices are the same registry (COMMANDS).
"""
import os
import re
import subprocess
import sys

from conftest import REPO_ROOT

MACE_CLI = REPO_ROOT / "mace_cli"


def _registry_commands():
    code = ("import importlib.machinery, importlib.util, json\n"
            f"loader = importlib.machinery.SourceFileLoader('mace_cli_mod', {str(MACE_CLI)!r})\n"
            "spec = importlib.util.spec_from_loader('mace_cli_mod', loader)\n"
            "m = importlib.util.module_from_spec(spec)\n"
            "loader.exec_module(m)\n"
            "print(json.dumps(list(m.COMMANDS)))\n")
    env = {k: v for k, v in os.environ.items() if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH")}
    r = subprocess.run([sys.executable, "-c", code], env=env, cwd=str(REPO_ROOT), stdin=subprocess.DEVNULL,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    import json
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_usage_choices_include_ions_and_other_new_commands():
    env = dict(os.environ, MACE_NO_BANNER="1")
    proc = subprocess.run([sys.executable, str(MACE_CLI), "--no-banner", "--help"],
                          env=env, capture_output=True, text=True, timeout=120)
    out = proc.stdout + proc.stderr
    commands = set(_registry_commands())
    for name in ("ions", "defects", "phonopy", "opt2d12", "preopt", "mlip"):
        assert re.search(rf"^  {name:<12}\S", out, re.M), (name, out)
        assert name in commands, (name, sorted(commands))
