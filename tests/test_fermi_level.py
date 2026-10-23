"""CRYSTAL's "no Fermi level" placeholder (mace/utils/fermi_level.py).

The example suite's diamond SP is spin-polarized (MACE's SPIN default) with
SPINLOCK 0 50: its SCF ends in 6 cycles with the spins still locked, so it
computes no Fermi level ("SPIN LOCKING: NO ENERGY GAP COMPUTED") and the
wavefunction keeps -1.0 Hartree. Its BAND run (no NEWK) prints FERMI ENERGY
-0.100E+01 and writes BAND.DAT / fort.25 as E - (-1.0): a "direct 249 eV
gap", no hole masses, the band plot 27 eV off. The Fermi level is now the
top of the occupied bands (12 electrons, alpha = beta = 6: bands 1-6), the
same -0.07797 Ha the DOSS run's NEWK computes, for every reader: the
extractor, the band-edge and mass analysis, the GUI's band plot and mace
plotting, the fort.25 conversion; the property output and the plot say so.
A run whose bands overlap (a metal) gets no number."""
import contextlib
import io
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import REPO_ROOT
from mace.utils import fermi_level as FL

HA = 27.211386245988
DOSS_EF = -0.07797                  # the diamond DOSS run's EFERMI (its NEWK), in its DOSS.DAT


@pytest.fixture(scope="module")
def diamond(tmp_path_factory):
    from mace import examples_suite as ES
    return ES.copy_suite(tmp_path_factory.mktemp("fermi") / "s") / "2-dense-3d" / "diamond"


def _extract(path):
    from mace.utils.property_extractor import CrystalPropertyExtractor
    x = CrystalPropertyExtractor(db_path=None, enable_tracking=False)
    with contextlib.redirect_stdout(io.StringIO()):
        return x.extract_all_properties(Path(path))


def test_the_placeholder_is_a_spin_locked_scf(diamond):
    sp = (diamond / "SP" / "diamond_sp.out").read_text()
    assert "ALPHA-BETA ELECTRONS LOCKED TO   0 FOR  50 SCF CYCLES" in sp and "SPIN LOCKING: NO ENERGY GAP COMPUTED" in sp
    band = (diamond / "BAND" / "diamond_band.out").read_text()
    assert "FERMI ENERGY              -0.100E+01" in band
    assert "# EFERMI (HARTREE)   -1.00000" in (diamond / "BAND" / "diamond_band.BAND.DAT").read_text()
    assert FL.is_placeholder(-1.0) and FL.is_placeholder("-0.100E+01") and not FL.is_placeholder(-0.07797)


def test_the_band_files_fermi_level_is_derived(diamond):
    for name in ("diamond_band.BAND.DAT", "diamond_band.f25"):
        fl = FL.band_fermi(diamond / "BAND" / name)
        assert fl.derived and fl.file_au == -1.0, name
        assert fl.value_au == pytest.approx(DOSS_EF, abs=2e-5), name
        assert "SPINLOCK" in fl.note and "6 alpha, 6 beta" in fl.note
    assert FL.spin_occupations((diamond / "BAND" / "diamond_band.out").read_text()) == (6, 6)
    # a file with CRYSTAL's own Fermi level is left as it is
    fl = FL.band_fermi(REPO_ROOT / "tests" / "data" / "spin_fe" / "fe_band.BAND.DAT")
    assert not fl.derived and fl.shift_au == 0.0


def test_overlapping_bands_give_no_number():
    k = np.linspace(0, 1, 20)[:, None]
    metal = np.hstack([k - 0.6, k * 0.1 - 0.05, k + 0.3])         # band 2 (occupied) crosses band 3's bottom
    value, why = FL.derive([metal, metal], (2, 2))
    assert value is None and "metal" in why
    insulator = np.hstack([k * 0.1 - 0.6, k * 0.1 - 0.2, k + 0.3])
    assert FL.derive([insulator], (2, 2))[0] == pytest.approx(-0.1)
    assert FL.derive([insulator], (2, 1))[0] is None            # one spin block for unequal spins


def test_the_extraction_uses_it(diamond):
    p = _extract(diamond / "BAND" / "diamond_band.out")
    assert p["fermi_energy_au"] == pytest.approx(DOSS_EF, abs=2e-5) and "derived" in p["fermi_energy_note"]
    assert p["band_path_gap_type"] == "indirect" and p["band_fundamental_gap_ev"] == pytest.approx(5.9911, abs=1e-3)
    assert p["band_path_vbm_k"] == "Γ" and p["band_dat_vbm_ev"] == pytest.approx(0.0, abs=1e-3)
    assert p["real_hole_effective_mass"] > 0
    assert any("E_F derived" in n for n in p["band_edges"]["notes"])
    d = _extract(diamond / "DOSS" / "diamond_doss.out")
    assert d["fermi_energy_au"] == pytest.approx(DOSS_EF) and "DOSS.DAT" in d["fermi_energy_note"]
    # a run with no band data of its own: no Fermi energy rather than -1.0
    c = _extract(diamond / "CHARGE+POTENTIAL" / "diamond_charge+potential.out")
    assert "fermi_energy_au" not in c and "fermi_energy_au" not in (c.get("charge_potential") or {})


def test_the_band_plot_gui_and_cli(diamond, tmp_path):
    from mace.gui import results
    from mace.plotting import f25
    for name in ("diamond_band.BAND.DAT", "diamond_band.f25"):
        d = results.band_data(diamond / "BAND" / name)
        assert "E_F derived" in d["fermi_note"], name
        # E - E_F: the valence band top at 0, the gap above it
        top = max(v for col in d["up"] for v in col if v is not None and v <= 0.01)
        assert top == pytest.approx(0.0, abs=2e-3), name
    text = f25.band_dat_text(diamond / "BAND" / "diamond_band.f25")
    assert "# EFERMI (HARTREE)   -0.07797" in text
    env = {**{k: v for k, v in os.environ.items() if k != "PYTHONPATH"}, "MPLBACKEND": "Agg"}
    r = subprocess.run([sys.executable, str(REPO_ROOT / "mace_cli"), "--no-banner", "plotting", "--band",
                        str(diamond / "BAND" / "diamond_band.BAND.DAT"), "--format", "html"],
                       capture_output=True, text=True, env=env, timeout=300, cwd=tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "E_F derived" in r.stdout
