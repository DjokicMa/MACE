"""phonopy on the command line, through the real mace_cli entry point (no phonopy needed).

`mace phonopy` (1.1.5, 1.1.6-preview) was folded into `mace opt2d12` (choice 8,
--calc-type PHONOPY) and `mace analyze phonopy`: it says so and exits 2, running
nothing. phonopy is optional, so the new forms are listed and documented without
it, and asking them to do work without phonopy ends in an install hint - not a
traceback.
"""
import os
import subprocess
import sys

from conftest import REPO_ROOT

MACE_CLI = REPO_ROOT / "mace_cli"
FOLDED = ("mace phonopy was folded into mace opt2d12 (choice 8 / --calc-type PHONOPY) "
          "and mace analyze phonopy")


def _run(*args, env=None, cwd=None):
    e = dict(os.environ if env is None else env)
    e["MACE_NO_BANNER"] = "1"
    e.pop("PYTHONPATH", None) if env is None else None
    return subprocess.run([sys.executable, str(MACE_CLI), *args], capture_output=True,
                          text=True, timeout=300, env=e, cwd=cwd)


def test_mace_phonopy_says_where_it_went_and_exits_2(tmp_path):
    for argv in (("phonopy",), ("phonopy", "--help"), ("phonopy", "generate", "x.out", "--dim", "2", "2", "2"),
                 ("--no-banner", "phonopy", "collect", "d")):
        p = _run(*argv, cwd=tmp_path)
        assert p.returncode == 2, (argv, p.stdout, p.stderr)
        assert p.stderr.strip() == FOLDED, (argv, p.stderr)
        assert p.stdout == ""
    assert list(tmp_path.iterdir()) == []


def test_top_level_help_lists_the_new_forms():
    p = _run("--no-banner", "--help")
    assert p.returncode == 0
    import re
    assert not re.search(r"^  (phonopy|elastic) {2,}\S", p.stdout, re.M), p.stdout
    p = _run("analyze", "--help")
    assert p.returncode == 0
    assert "mace analyze phonopy DIR" in p.stdout and "mace analyze elastic RUN.out" in p.stdout


def test_analyze_phonopy_help_reaches_its_parser():
    p = _run("analyze", "phonopy", "--help")
    assert p.returncode == 0, p.stderr
    assert "--nac-from" in p.stdout and "--band" in p.stdout and "FORCE_SETS" in p.stdout


def test_opt2d12_help_offers_phonopy():
    p = _run("opt2d12", "--help")
    assert p.returncode == 0, p.stderr
    text = " ".join(p.stdout.split())
    assert "--calc-type {SP,OPT,FREQ,ELASTIC,EOS,QHA,CPHF,PHONOPY}" in text
    assert "--dim" in text and "--keep-shrink" in text and "--amplitude" in text


def test_without_phonopy_exits_with_the_hint(tmp_path):
    """A shadow package that fails to import stands in for "not installed", so
    this runs whether or not phonopy is present in the test environment."""
    shim = tmp_path / "shim" / "phonopy"
    shim.mkdir(parents=True)
    (shim / "__init__.py").write_text("raise ImportError('phonopy not installed (test shim)')\n")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = str(tmp_path / "shim")
    import shutil
    fix = REPO_ROOT / "tests" / "fixtures" / "phonopy_crystal" / "Si"
    shutil.copy(fix / "parent.out", tmp_path / "x.out")
    shutil.copy(fix / "parent.d12", tmp_path / "x.d12")
    p = _run("opt2d12", "--out-file", "x.out", "--calc-type", "PHONOPY", "--non-interactive",
             "--dim", "2", "2", "2", env=env, cwd=tmp_path)
    assert p.returncode == 1, p.stdout + p.stderr
    assert "pip install 'phonopy<5'" in p.stdout + p.stderr
    assert "Traceback" not in p.stdout + p.stderr
    assert not [f for f in tmp_path.rglob("*_disp-*")]
    (tmp_path / "set").mkdir()
    (tmp_path / "set" / "mace_phonopy.json").write_text('{"jobs": []}')
    p = _run("analyze", "phonopy", "set", env=env, cwd=tmp_path)
    assert p.returncode == 2, p.stdout + p.stderr
    assert "pip install 'phonopy<5'" in p.stderr
    assert "Traceback" not in p.stdout + p.stderr
