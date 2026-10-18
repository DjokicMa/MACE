"""`mace plotting` exits non-zero when a requested figure was not drawn (1.1.6 review F4-3).

Each plotter goes on to the next file after an error; the run used to exit 0 all the same, so a
script could not tell (e.g. --energy-reference vacuum without the run's POTC files). The real CLI
runs as a subprocess, with both backends.
"""
import os
import shutil
import subprocess
import sys

import pytest

from conftest import REPO_ROOT

MACE_CLI = REPO_ROOT / "mace_cli"
FE = REPO_ROOT / "tests" / "data" / "spin_fe"


def _plotting(args, cwd, backend, traceback_ok=False):
    env = {**os.environ, "MPLBACKEND": "Agg", "MACE_FOLDER_SETTINGS": "none", "MACE_PLOT_BACKEND": backend}
    p = subprocess.run([sys.executable, str(MACE_CLI), "--no-banner", "plotting", *args], cwd=str(cwd),
                       capture_output=True, text=True, timeout=300, env=env)
    out = p.stdout + p.stderr
    assert traceback_ok or "Traceback (most recent call last)" not in out, out[-3000:]
    return p, out


@pytest.fixture
def fe(tmp_path):
    for f in FE.iterdir():
        shutil.copy(f, tmp_path)
    return tmp_path


@pytest.mark.parametrize("backend", ["plotly", "matplotlib"])
def test_vacuum_reference_without_potc_exits_non_zero(fe, tmp_path, backend):
    p, out = _plotting(["--band", "fe_band.BAND.DAT", "--energy-reference", "vacuum", "-o", str(tmp_path / "o")],
                       fe, backend)
    assert p.returncode == 1, out[-2000:]
    assert "POTC" in out and "1 figure(s) not drawn" in p.stderr and "fe_band.BAND.DAT" in p.stderr


@pytest.mark.parametrize("backend", ["plotly", "matplotlib"])
def test_band_dos_without_a_partner_exits_non_zero(fe, tmp_path, backend):
    (fe / "fe_doss.DOSS.DAT").unlink()
    p, out = _plotting(["--band-dos", "fe_band.BAND.DAT", "-o", str(tmp_path / "o")], fe, backend)
    assert p.returncode == 1, out[-2000:]
    assert "skipped" in out and "not drawn" in p.stderr


def test_one_bad_file_fails_the_run_but_the_good_one_is_drawn(fe, tmp_path):
    bad = fe / "broken_band.BAND.DAT"
    bad.write_text("# not a BAND.DAT\n")
    out_dir = tmp_path / "o"
    p, out = _plotting(["--band", "fe_band.BAND.DAT", str(bad), "-o", str(out_dir), "--format", "svg"], fe, "matplotlib",
                       traceback_ok=True)     # the matplotlib band plotter prints its own traceback
    assert p.returncode == 1, out[-2000:]
    assert (out_dir / "fe.BANDS.svg").is_file()
    assert "broken_band.BAND.DAT" in p.stderr and "fe_band" not in p.stderr.split("not drawn:")[1]


def test_a_good_run_exits_zero(fe, tmp_path):
    p, out = _plotting(["--dos", "fe_doss.DOSS.DAT", "-o", str(tmp_path / "o"), "--format", "svg"], fe, "matplotlib")
    assert p.returncode == 0, out[-2000:]
    assert "not drawn" not in out


def test_matplotlib_band_dos_writes_no_error_pane_as_the_figure(fe, tmp_path):
    """The matplotlib band + DOS figure drew a reader failure as the figure (an
    image of the message) and exited 0; it now fails the file and writes nothing."""
    out_dir = tmp_path / "o"
    p, out = _plotting(["--band-dos", "fe_band.BAND.DAT", "--energy-reference", "vacuum", "-o", str(out_dir),
                        "--format", "png"], fe, "matplotlib")
    assert p.returncode == 1, out[-2000:]
    assert "POTC" in out and "not drawn" in p.stderr
    assert not out_dir.exists() or not list(out_dir.glob("*.png"))
    (fe / "fe_doss.DOSS.DAT").write_text("")
    p, out = _plotting(["--band-dos", "fe_band.BAND.DAT", "-o", str(out_dir), "--format", "png"], fe, "matplotlib")
    assert p.returncode == 1 and "Generated" not in out, out[-2000:]


@pytest.mark.parametrize("flag", [["--no-dos-total"], ["--dos-spin", "overlaid"], ["--units", "meV"]])
def test_matplotlib_band_dos_refuses_what_it_cannot_draw(fe, tmp_path, flag):
    out_dir = tmp_path / "o"
    p, out = _plotting(["--band-dos", "fe_band.BAND.DAT", *flag, "-o", str(out_dir), "--format", "png"],
                       fe, "matplotlib")
    assert p.returncode == 1, out[-2000:]
    assert "cannot draw" in out and flag[0] in out and "--backend plotly" in out
    assert not out_dir.exists() or not list(out_dir.iterdir())
    # the defaults named explicitly (what the GUI passes under a folder file) still draw
    p, out = _plotting(["--band-dos", "fe_band.BAND.DAT", "--dos-total", "--dos-spin", "auto", "--units", "eV",
                        "-o", str(out_dir), "--format", "png"], fe, "matplotlib")
    assert p.returncode == 0 and (out_dir / "fe.BAND_DOS.png").is_file(), out[-2000:]


@pytest.mark.parametrize("limits", [["--e-lower", "1", "--e-upper", "1"], ["--e-lower", "2", "--e-upper", "-2"]])
def test_equal_or_inverted_energy_limits_are_refused(fe, tmp_path, limits):
    """As the GUI refuses them: equal limits drew a zero-height plot, inverted ones were swapped."""
    p, out = _plotting(["--band", "fe_band.BAND.DAT", *limits, "-o", str(tmp_path / "o")], fe, "plotly")
    assert p.returncode == 2 and "must be below --e-upper" in out, out[-2000:]
    assert not (tmp_path / "o").exists()


def test_output_with_a_figure_name_is_refused(fe, tmp_path):
    """-o fig.svg made a folder named fig.svg for --band/--dos/--band-dos."""
    p, out = _plotting(["--dos", "fe_doss.DOSS.DAT", "-o", str(tmp_path / "fig.svg")], fe, "plotly")
    assert p.returncode == 2 and "names the folder" in out, out[-2000:]
    assert not (tmp_path / "fig.svg").exists()
