"""`mace plotting --orbitals` run as the real `mace_cli` subprocess on real ORBITALS fixtures."""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
MACE_CLI = REPO / "mace_cli"
DATA = REPO / "tests" / "data" / "orbitals"

pytestmark = pytest.mark.skipif(not MACE_CLI.is_file(), reason="mace_cli not found")


def _run(args, cwd, timeout=300):
    proc = subprocess.run([sys.executable, str(MACE_CLI), "plotting", *args], cwd=str(cwd),
                          capture_output=True, text=True, timeout=timeout)
    out = proc.stdout + proc.stderr
    assert "Traceback (most recent call last)" not in out, out[-3000:]
    return proc, out


def _copy(folder, dst):
    for f in (DATA / folder).iterdir():
        shutil.copy(f, dst / f.name)


def test_list_orbitals_prints_levels_and_check(tmp_path):
    _copy("si_g", tmp_path)
    proc, out = _run(["--orbitals", "--list-orbitals"], tmp_path)
    assert proc.returncode == 0, out[-2000:]
    assert "Self-check passed" in out
    assert "HOMO" in out and "LUMO" in out and "degenerate 12-14" in out
    assert not (tmp_path / "si_g_orbitals").exists()          # listing writes nothing


def test_molecule_homo_cube_and_html(tmp_path):
    pytest.importorskip("plotly")
    _copy("h2o", tmp_path)
    proc, out = _run(["--orbitals", "--mo", "HOMO", "-o", "plots"], tmp_path)
    assert proc.returncode == 0, out[-2000:]
    cube = tmp_path / "h2o_c_orbitals" / "h2o_c_ab_MO5_re.cube"
    assert cube.is_file()
    assert list((tmp_path / "plots").glob("h2o_c_ab_MO5_re.iso.html"))
    # the Molden file stays exactly as CRYSTAL wrote it
    assert (tmp_path / "h2o_c.molden").read_bytes() == (DATA / "h2o" / "h2o_c.molden").read_bytes()


def test_periodic_degenerate_sum_from_explicit_file(tmp_path):
    pytest.importorskip("plotly")
    _copy("si_g", tmp_path)
    proc, out = _run(["--orbitals", "si_g_00001_K000_real.molden", "--mo", "HOMO", "--degenerate-sum",
                      "--quality", "preview", "-o", "plots"], tmp_path)
    assert proc.returncode == 0, out[-2000:]
    assert (tmp_path / "si_g_orbitals" / "si_g_k1_ab_MO12-13-14_abs2_preview.cube").is_file()


def test_complex_k_in_3d_is_refused(tmp_path):
    _copy("si_f", tmp_path)
    proc, out = _run(["--orbitals", "--k", "4", "--mo", "HOMO"], tmp_path)
    assert "not available" in out and "complex k point" in out
    assert not list(tmp_path.glob("si_f_orbitals/*.cube"))


def test_polymer_complex_k_im_part(tmp_path):
    pytest.importorskip("plotly")
    _copy("polyyne", tmp_path)
    proc, out = _run(["--orbitals", "--k", "2", "--mo", "HOMO", "--part", "im", "--cells", "4",
                      "--quality", "preview", "-o", "plots"], tmp_path)
    assert proc.returncode == 0, out[-2000:]
    assert (tmp_path / "py_f_orbitals" / "py_f_k2_ab_MO6_im_4x1x1_preview.cube").is_file()


def test_open_shell_beta_channel(tmp_path):
    pytest.importorskip("plotly")
    _copy("o2", tmp_path)
    proc, out = _run(["--orbitals", "--mo", "HOMO", "--orb-spin", "beta", "--quality", "preview",
                      "-o", "plots"], tmp_path)
    assert proc.returncode == 0, out[-2000:]
    assert list((tmp_path / "o2_c_orbitals").glob("o2_c_beta_MO*_re_preview.cube"))
