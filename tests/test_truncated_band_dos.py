"""A short BAND.DAT/DOSS.DAT (a copy still in progress, a full quota) is refused.

The readers sized their arrays from the header: the missing k points were drawn
at k = 0, E = 0, a missing beta block made a "closed shell" run, and a DOSS file
cut inside spin up drew half of it as spin down. The CLI (both backends) exits 1
with "truncated", and the GUI's readers raise it (the page shows the message).
"""
import os
import shutil
import subprocess
import sys

import pytest

from conftest import REPO_ROOT

FE = REPO_ROOT / "tests" / "data" / "spin_fe"


@pytest.fixture
def cut(tmp_path):
    for f in FE.iterdir():
        shutil.copy(f, tmp_path)
    for name, keep in (("fe_band.BAND.DAT", 120), ("fe_doss.DOSS.DAT", 300)):
        lines = (tmp_path / name).read_text().splitlines(True)
        (tmp_path / name).write_text("".join(lines[:keep]))
    return tmp_path


def test_gui_readers_refuse_a_short_file(cut):
    from mace.gui import results
    with pytest.raises(ValueError, match=r"truncated: \d+ of 400 k points"):
        results.band_data(cut / "fe_band.BAND.DAT")
    with pytest.raises(ValueError, match=r"truncated: \d+ of 1336 energy points"):
        results.dos_data(cut / "fe_doss.DOSS.DAT")


def test_gui_readers_take_the_whole_files():
    from mace.gui import results
    assert results.band_data(FE / "fe_band.BAND.DAT")["n_kpoints"] == 200
    assert results.dos_data(FE / "fe_doss.DOSS.DAT")


def test_a_beta_block_cut_short_is_refused(tmp_path):
    from mace.gui import results
    for f in FE.iterdir():
        shutil.copy(f, tmp_path)
    lines = (tmp_path / "fe_band.BAND.DAT").read_text().splitlines(True)
    (tmp_path / "fe_band.BAND.DAT").write_text("".join(lines[:-5]))
    with pytest.raises(ValueError, match="truncated"):
        results.band_data(tmp_path / "fe_band.BAND.DAT")


@pytest.mark.parametrize("backend", ["plotly", "matplotlib"])
@pytest.mark.parametrize("flag,name", [("--band", "fe_band.BAND.DAT"), ("--dos", "fe_doss.DOSS.DAT")])
def test_cli_exits_1_on_a_short_file(cut, tmp_path, backend, flag, name):
    env = {**os.environ, "MPLBACKEND": "Agg", "MACE_FOLDER_SETTINGS": "none", "MACE_PLOT_BACKEND": backend}
    out_dir = tmp_path / "o"
    p = subprocess.run([sys.executable, str(REPO_ROOT / "mace_cli"), "--no-banner", "plotting", flag, name,
                        "-o", str(out_dir), "--format", "png"], cwd=str(cut), capture_output=True, text=True,
                       timeout=300, env=env)
    out = p.stdout + p.stderr
    assert p.returncode == 1, out[-2000:]
    assert "truncated" in out
    assert not out_dir.exists() or not list(out_dir.glob("*.png"))
