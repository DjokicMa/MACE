"""--array and --max-running read the same everywhere: `mace submit -h` and
`mace workflow -h` (curated text), the argparse help strings behind them, and
`--max-running`'s wording on both commands."""
import ast
import importlib.machinery
import importlib.util
import os
import re
import subprocess
import sys

import pytest

from conftest import REPO_ROOT

MACE_CLI = REPO_ROOT / "mace_cli"
RUN_MACE = REPO_ROOT / "mace" / "run_mace.py"


def _env():
    return {k: v for k, v in os.environ.items()
            if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH")}


def _mace(*args):
    r = subprocess.run([sys.executable, str(MACE_CLI), *args], env=_env(), cwd=str(REPO_ROOT),
                       stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120)
    return r.returncode, r.stdout + r.stderr


def _command_help():
    loader = importlib.machinery.SourceFileLoader("mace_cli_help_mod", str(MACE_CLI))
    spec = importlib.util.spec_from_loader("mace_cli_help_mod", loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod.COMMAND_HELP


def _argparse_help(path, flag):
    """The help= string of add_argument(flag, ...) in a source file."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument"
                and node.args and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == flag):
            for kw in node.keywords:
                if kw.arg == "help":
                    return kw.value.value.replace("%%", "%")
    raise AssertionError(f"no add_argument({flag!r}) in {path}")


def _curated(text, flag):
    """The description of ``flag`` in a curated help block (to the next option)."""
    m = re.search(rf"^  {re.escape(flag)}(?: N)?\s+(.*?)(?=^\s*$|^  -)", text, re.M | re.S)
    assert m, (flag, text)
    return m.group(1)


def _words(text):
    return " ".join(text.split())


@pytest.mark.parametrize("command,source", [("submit", MACE_CLI), ("workflow", RUN_MACE)])
@pytest.mark.parametrize("flag", ["--array", "--max-running"])
def test_curated_help_matches_argparse_help(command, source, flag):
    curated = _curated(_command_help()[command], flag)
    assert _words(curated) == _words(_argparse_help(source, flag))


def test_max_running_reads_the_same_on_both_commands():
    help_ = _command_help()
    assert _words(_curated(help_["submit"], "--max-running")) == \
        _words(_curated(help_["workflow"], "--max-running"))


@pytest.mark.parametrize("argv", [("submit", "-h"), ("submit", "--help"),
                                  ("workflow", "-h"), ("workflow", "--help")])
def test_flags_are_in_the_shown_help(argv):
    rc, text = _mace("--no-banner", *argv)
    assert rc == 0, text
    assert "--array" in text and "--max-running N" in text


def test_run_mace_argparse_help_lists_them():
    r = subprocess.run([sys.executable, str(RUN_MACE), "--help"],
                       env=dict(_env(), PYTHONPATH=str(REPO_ROOT)), cwd=str(REPO_ROOT),
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "--array" in r.stdout and "--max-running N" in r.stdout
