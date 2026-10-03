"""`mace convert --help` works on a python without ase.

NewCifToD12 imported ase at the top, so without ase (the CI interpreters)
`mace convert --help` died with ModuleNotFoundError before argparse ran. ase
is only needed to read a CIF. Here ase is hidden by a package of the same
name that fails to import, so the test means the same on a python that has it.
"""
import os
import subprocess
import sys

from conftest import REPO_ROOT

MACE_CLI = REPO_ROOT / "mace_cli"


def _mace(tmp_path, *args):
    shadow = tmp_path / "shadow" / "ase"
    shadow.mkdir(parents=True, exist_ok=True)
    (shadow / "__init__.py").write_text("raise ImportError(\"No module named 'ase'\")\n")
    env = {k: v for k, v in os.environ.items()
           if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH")}
    env.update(PYTHONPATH=str(tmp_path / "shadow"), MACE_NO_BANNER="1")
    r = subprocess.run([sys.executable, str(MACE_CLI), "convert", *args], cwd=str(tmp_path),
                       env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                       timeout=120)
    return r.returncode, r.stdout + r.stderr


def test_convert_help_needs_no_ase(tmp_path):
    rc, text = _mace(tmp_path, "--help")
    assert rc == 0, text
    assert "--cif_dir" in text and "--batch" in text
    assert "Traceback" not in text


def test_convert_without_ase_says_what_is_missing(tmp_path):
    (tmp_path / "x.cif").write_text("data_x\n")
    rc, text = _mace(tmp_path, "--cif_dir", str(tmp_path))
    assert rc == 1, text
    assert "needs ase" in text and "pip install ase" in text
    assert "Traceback" not in text
