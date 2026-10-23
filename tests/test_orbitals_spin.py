"""Spin-polarized ORBITALS runs (1.1.6): both spin channels pass the self-check and draw.

CRYSTAL23's ORBITALS writes the BETA block of a spin-polarized run whose alpha and beta are the
same (a closed shell run spin-polarized, MACE's default, SPINLOCK) with coefficients that are not
orbitals: zeros for the EC molecule, a rank-3 matrix for diamond, values up to 1e42 for h-BN. A run
whose spins differ (O2, bcc Fe) has valid BETA coefficients. Where every BETA level has the energy and
occupation of its ALPHA partner and the .out reports no net spin, BETA draws the ALPHA orbitals (checked
again); the check of the file's own BETA coefficients stays what it was. The example suite's real files
(examples/suite, .molden.xz) and the HPCC fixtures in tests/data/orbitals.
"""
import lzma
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mace.orbitals import api
from mace.orbitals.dataset import CHECK_FAIL, OrbitalRun, OrbitalsError

REPO = Path(__file__).resolve().parent.parent
SUITE = REPO / "examples" / "suite"
DATA = REPO / "tests" / "data" / "orbitals"
RUNS = {"diamond": "2-dense-3d/diamond", "hBN": "3-slab-2d/hBN", "polyacetylene": "4-polymer-1d/polyacetylene",
        "EC": "5-molecule-0d/EC"}


def suite_run(name, dst: Path) -> Path:
    """The suite's ORBITALS folder of ``name`` in ``dst``, its .molden files decompressed as CRYSTAL wrote them;
    returns the .out."""
    src = SUITE / RUNS[name] / "ORBITALS"
    for f in src.iterdir():
        if f.name.endswith(".xz"):
            (dst / f.name[:-3]).write_bytes(lzma.decompress(f.read_bytes()))
        else:
            shutil.copy(f, dst / f.name)
    return next(dst.glob("*.out"))


@pytest.mark.parametrize("name", list(RUNS))
def test_both_spins_pass_and_draw(name, tmp_path):
    run = api.open_run(suite_run(name, tmp_path))
    for e in run.kpoints():
        if not e.usable:
            continue
        kd = run.load(e.index)
        assert kd.spins == ["ALPHA", "BETA"]
        # the file's own BETA coefficients are not orbitals; its ALPHA ones pass
        assert kd.file_check["BETA"] >= CHECK_FAIL and kd.file_check["ALPHA"] < 1e-3, kd.file_check
        # alpha = beta: BETA draws the ALPHA orbitals, checked again (the check itself is unchanged)
        assert kd.beta_from_alpha and kd.failed_spins == [] and kd.status == "ok"
        assert kd.check["BETA"] == pytest.approx(kd.check["ALPHA"]) and kd.check["BETA"] < 1e-3
        assert kd.spin_note().startswith("β = α: CRYSTAL wrote no valid β coefficients")
        s = kd.summary()
        assert s["beta_from_alpha"] and s["spin_note"] == kd.spin_note() and s["file_check"]["BETA"] >= CHECK_FAIL
        _, b, _ = run.grid(e.index, "BETA", ["HOMO"], quality="preview")
        _, a, _ = run.grid(e.index, "ALPHA", ["HOMO"], quality="preview")
        assert np.abs(b).max() > 0 and np.allclose(a, b)


@pytest.mark.parametrize("folder, out", [("o2", "o2_c.out"), ("fe", "fe_f.out")])
def test_a_run_whose_spins_differ_draws_its_own_beta(folder, out):
    """O2 (a triplet) and bcc Fe: CRYSTAL's BETA coefficients are orbitals; nothing is substituted."""
    run = api.open_run(DATA / folder / out)
    kd = run.load(run.kpoints()[0].index)
    assert not kd.beta_from_alpha and kd.failed_spins == [] and kd.spin_note() == ""
    assert kd.check["BETA"] == kd.file_check["BETA"] < 1e-3
    assert abs(run.out.spin_density) > 1.9               # 2 unpaired electrons per cell
    a, b = kd.channel("ALPHA"), kd.channel("BETA")
    assert not np.allclose(kd.C[:, a], kd.C[:, b])


@pytest.mark.parametrize("folder, out", [("h2o", "h2o_c.out"), ("si_g", None), ("polyyne", None)])
def test_closed_shell_files_are_unchanged(folder, out):
    """ALPHA+BETA (a closed-shell run): one channel, nothing about spin to say."""
    run = api.open_run(DATA / folder / out) if out else OrbitalRun.from_path(sorted((DATA / folder).glob("*_real.molden"))[0])
    kd = run.load(run.kpoints()[0].index)
    assert kd.spins == ["ALPHA+BETA"] and not kd.beta_from_alpha and kd.spin_note() == "" and kd.status == "ok"


def test_beta_stays_refused_where_the_run_has_net_spin(tmp_path):
    """The same broken BETA block in a run whose .out reports a net spin is not alpha = beta: it is refused."""
    out = suite_run("diamond", tmp_path)
    text = out.read_text()
    assert "SUMMED SPIN DENSITY   -0.00000000" in text
    out.write_text(text.replace("SUMMED SPIN DENSITY   -0.00000000", "SUMMED SPIN DENSITY    2.00000000"))
    run = api.open_run(out)
    kd = run.load(1)
    assert run.out.spin_density == 2.0
    assert not kd.beta_from_alpha and kd.failed_spins == ["BETA"] and kd.check["BETA"] >= CHECK_FAIL
    assert "β: the coefficients fail the orthonormality self-check" in kd.spin_note()
    with pytest.raises(OrbitalsError, match="BETA orbitals of this file fail the orthonormality"):
        run.grid(1, "BETA", ["HOMO"], quality="preview")
    run.grid(1, "ALPHA", ["HOMO"], quality="preview")          # the other channel still draws


def test_mace_plotting_orbitals_draws_beta(tmp_path):
    """`mace plotting --orbitals --orb-spin beta` (the real mace_cli) on the suite's diamond: the β cube is
    written, and the self-check line and the β = α note are printed."""
    pytest.importorskip("plotly")                         # the figure drawn from the cube
    suite_run("diamond", tmp_path)
    proc = subprocess.run([sys.executable, str(REPO / "mace_cli"), "plotting", "--orbitals", "--orb-spin", "beta",
                           "--mo", "HOMO", "--quality", "preview", "-o", "plots"], cwd=str(tmp_path),
                          capture_output=True, text=True, timeout=300)
    out = proc.stdout + proc.stderr
    assert "Traceback (most recent call last)" not in out, out[-3000:]
    assert "Self-check passed (max |C^H S C - 1| = 1.5e-05)." in out, out[-3000:]
    assert "β = α: CRYSTAL wrote no valid β coefficients" in out and "refusing" not in out, out[-3000:]
    cubes = list((tmp_path / "diamond_orbitals").glob("*_beta_MO6_re_preview.cube"))
    assert cubes, out[-3000:]


def test_mace_plotting_list_orbitals_says_beta_is_alpha(tmp_path):
    """`mace plotting --orbitals --list-orbitals` (no plotly needed): the self-check over the drawn spins and
    the β = α line, as the GUI's picker shows them."""
    suite_run("diamond", tmp_path)
    proc = subprocess.run([sys.executable, str(REPO / "mace_cli"), "plotting", "--orbitals", "--list-orbitals",
                           "--orb-spin", "beta"], cwd=str(tmp_path), capture_output=True, text=True, timeout=300)
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0 and "Traceback (most recent call last)" not in out, out[-3000:]
    assert "Self-check passed (max |C^H S C - 1| = 1.5e-05)." in out, out[-3000:]
    assert "β = α: CRYSTAL wrote no valid β coefficients (max |C^H S C − 1| = 2.6)" in out, out[-3000:]
    assert "Levels, spin BETA" in out, out[-3000:]
