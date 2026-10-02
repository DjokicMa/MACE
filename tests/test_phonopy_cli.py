"""`mace phonopy` through the real mace_cli entry point (no phonopy needed).

phonopy is optional, so the command must be listed and documented without it,
and asking it to do work without phonopy must end in an install hint and exit
code 2 - not a traceback.
"""
import os
import subprocess
import sys

from conftest import REPO_ROOT

MACE_CLI = REPO_ROOT / "mace_cli"


def _run(*args, env=None, cwd=None):
    e = dict(os.environ if env is None else env)
    e["MACE_NO_BANNER"] = "1"
    return subprocess.run([sys.executable, str(MACE_CLI), *args], capture_output=True,
                          text=True, timeout=300, env=e, cwd=cwd)


def test_phonopy_help_exits_0():
    p = _run("phonopy", "--help")
    assert p.returncode == 0, p.stderr
    assert "mace phonopy generate" in p.stdout and "mace phonopy collect" in p.stdout
    assert "--track" in p.stdout                      # the "never --track" warning


def test_top_level_help_lists_phonopy():
    p = _run("--no-banner", "--help")
    assert p.returncode == 0
    assert "phonopy" in p.stdout


def test_subcommand_help_reaches_the_subcommand_parser():
    p = _run("phonopy", "generate", "--help")
    assert p.returncode == 0, p.stderr
    assert "--scale-shrink" in p.stdout and "--amplitude" in p.stdout


def test_generate_without_phonopy_exits_2_with_hint(tmp_path):
    """A shadow package that fails to import stands in for "not installed", so
    this runs whether or not phonopy is present in the test environment."""
    shim = tmp_path / "shim" / "phonopy"
    shim.mkdir(parents=True)
    (shim / "__init__.py").write_text("raise ImportError('phonopy not installed (test shim)')\n")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tmp_path / "shim") + os.pathsep + env.get("PYTHONPATH", "")
    (tmp_path / "x.out").write_text("")
    (tmp_path / "x.d12").write_text("")
    p = _run("phonopy", "generate", "x.out", "--dim", "2", "2", "2", env=env, cwd=tmp_path)
    assert p.returncode == 2, p.stdout + p.stderr
    assert "pip install 'phonopy<5'" in p.stderr
    assert "Traceback" not in p.stdout + p.stderr
