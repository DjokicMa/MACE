"""opt2d3 names its files after the output without _sp/_opt - unless that collides.

anatase_sp.out and anatase_opt.out in one directory both became
anatase_doss.d3 (and anatase_doss.f9), the second overwriting the first.
With both present each keeps its whole name; alone, the name is as before
(the workflow expects material_band.d3). Real output: the AgBr OPT in
tests/data, under the two names.
"""
import shutil
import subprocess
import sys

from conftest import REPO_ROOT

SRC = REPO_ROOT / "tests" / "data" / "opt_restart" / "tqc_agbr.out"
CONFIG = REPO_ROOT / "Crystal_d3" / "example_configs" / "doss_total_only.json"


def _doss(cwd, name):
    run = subprocess.run(
        [sys.executable, str(REPO_ROOT / "mace_cli"), "opt2d3", "--input", name,
         "--calc-type", "DOSS", "--config-file", str(CONFIG), "--output-dir", str(cwd)],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, cwd=cwd, timeout=300)
    assert run.returncode == 0, run.stdout + run.stderr
    return run.stdout + run.stderr


def _stage(cwd, stems):
    for stem in stems:
        shutil.copy(SRC, cwd / f"{stem}.out")
        (cwd / f"{stem}.f9").write_bytes(stem.encode())


def test_sp_and_opt_side_by_side_keep_their_own_decks(tmp_path):
    _stage(tmp_path, ["agbr_sp", "agbr_opt"])
    _doss(tmp_path, "agbr_sp.out")
    _doss(tmp_path, "agbr_opt.out")
    assert (tmp_path / "agbr_sp_doss.d3").exists() and (tmp_path / "agbr_opt_doss.d3").exists()
    assert (tmp_path / "agbr_sp_doss.f9").read_bytes() == b"agbr_sp"
    assert (tmp_path / "agbr_opt_doss.f9").read_bytes() == b"agbr_opt"
    assert not (tmp_path / "agbr_doss.d3").exists()


def test_a_lone_output_keeps_the_short_name(tmp_path):
    _stage(tmp_path, ["agbr_sp"])
    _doss(tmp_path, "agbr_sp.out")
    assert (tmp_path / "agbr_doss.d3").exists()
    assert (tmp_path / "agbr_doss.f9").read_bytes() == b"agbr_sp"


def test_a_plain_output_beside_its_sp_keeps_its_own_deck(tmp_path):
    """agbr.out and agbr_sp.out both made agbr_doss.d3: the second run wrote
    over the first deck and its wavefunction."""
    _stage(tmp_path, ["agbr", "agbr_sp"])
    _doss(tmp_path, "agbr.out")
    _doss(tmp_path, "agbr_sp.out")
    assert (tmp_path / "agbr_doss.f9").read_bytes() == b"agbr"
    assert (tmp_path / "agbr_sp_doss.f9").read_bytes() == b"agbr_sp"


def test_an_sp_of_an_opt_takes_its_own_wavefunction(tmp_path):
    """agbr_opt_sp.out took the OPT's agbr_opt.f9 beside it before its own
    agbr_opt_sp.f9, and its agbr_opt_* name is the one agbr_opt.out keeps when
    an agbr_sp.out is there too: beside agbr_opt.out it keeps its whole name."""
    _stage(tmp_path, ["agbr_opt", "agbr_sp", "agbr_opt_sp"])
    _doss(tmp_path, "agbr_opt_sp.out")
    _doss(tmp_path, "agbr_opt.out")
    _doss(tmp_path, "agbr_sp.out")
    assert (tmp_path / "agbr_opt_sp_doss.f9").read_bytes() == b"agbr_opt_sp"
    assert (tmp_path / "agbr_opt_doss.f9").read_bytes() == b"agbr_opt"
    assert (tmp_path / "agbr_sp_doss.f9").read_bytes() == b"agbr_sp"


def test_an_sp_of_an_opt_alone_takes_its_own_wavefunction(tmp_path):
    _stage(tmp_path, ["agbr_opt_sp"])
    (tmp_path / "agbr_opt.f9").write_bytes(b"the OPT's")
    _doss(tmp_path, "agbr_opt_sp.out")
    assert (tmp_path / "agbr_opt_doss.f9").read_bytes() == b"agbr_opt_sp"
