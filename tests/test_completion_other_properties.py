"""`mace completion` does not call every properties run it cannot type a BAND.

A finished properties run whose .d3 drives none of BAND, DOSS, TRANSPORT or
CHARGE+POTENTIAL - a matrix dump (BASISSET with print codes 60 and 64, the
H(R)/S(R) dump), for one - was listed as "Band structure (D3 BAND)" and moved
to completeband/. MATDUMP is not a calculation type in this release, so such
a run is an "other properties" run.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

from conftest import REPO_ROOT
from mace import completion_checker as cc

DATA = Path(__file__).resolve().parent / "data"
MACE_CLI = REPO_ROOT / "mace_cli"
MATDUMP_D3 = "NEWK\n8 8\n1 0\nBASISSET\n2\n60 1\n64 1\nEND\n"
BAND_D3 = "BAND\nx_band\n1 12 1000 1 52 1 0\n0 0 0 6 6 6\nEND\n"


def _stage(d, name, deck):
    # a finished run's output (CRYSTAL's closing lines) beside its deck
    shutil.copy(DATA / "low_dim_groups" / "polyyne_rg28.out", d / f"{name}.out")
    (d / f"{name}.d3").write_text(deck)


def _mace(cwd, *args):
    env = {k: v for k, v in os.environ.items()
           if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH")}
    env["MACE_NO_BANNER"] = "1"
    r = subprocess.run([sys.executable, str(MACE_CLI), "completion", *args], cwd=str(cwd),
                       env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                       timeout=120)
    return r.returncode, r.stdout + r.stderr


def test_matrix_dump_run_is_listed_as_other_properties(tmp_path):
    _stage(tmp_path, "x_matdump", MATDUMP_D3)
    rc, text = _mace(tmp_path)
    assert rc == 0, text
    assert "Band structure" not in text, text
    assert "Other properties (D3)" in text, text
    assert cc.categorize_output_file(tmp_path / "x_matdump.out")[0] == "completeproperties"


def test_band_run_is_still_a_band(tmp_path):
    _stage(tmp_path, "x_band", BAND_D3)
    assert cc.categorize_output_file(tmp_path / "x_band.out")[0] == "completeband"


def test_organize_moves_it_to_its_own_folder(tmp_path):
    _stage(tmp_path, "x_matdump", MATDUMP_D3)
    rc, text = _mace(tmp_path, "--organize")
    assert rc == 0, text
    assert not (tmp_path / "sorted" / "completeband").exists(), text
    assert (tmp_path / "sorted" / "completeproperties" / "x_matdump.out").exists(), text
