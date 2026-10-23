"""FREQ derived properties (mace/properties/defs/vibrations.py): Debye
temperatures from the phonon DOS moments, the acoustic velocities at Γ of
the phonon bands and their Debye temperature (a real diamond SCELPHONO
fort.25, tests/data/f25), and the mean-square displacements of a
molecule from its classical-amplitude normal modes (the example suite's
EC FREQ run)."""
import contextlib
import io
import math
import shutil
from pathlib import Path

import numpy as np
import pytest

from conftest import REPO_ROOT
from mace.properties import registry
from mace.properties.defs import vibrations as VB
from mace.utils.property_extractor import CrystalPropertyExtractor

F25 = REPO_ROOT / "tests" / "data" / "f25" / "dia_phonon_f25.fort.25"


@pytest.fixture(scope="module")
def suite(tmp_path_factory):
    from mace import examples_suite as ES
    root = tmp_path_factory.mktemp("props_freq") / "suite"
    ES.copy_suite(root)
    return root


def derived(out: Path, only):
    x = CrystalPropertyExtractor(db_path=None, enable_tracking=False)
    x.run_derived = False
    with contextlib.redirect_stdout(io.StringIO()):
        props = x.extract_all_properties(Path(out))
    text = Path(out).read_text(errors="replace")
    return registry.run_derived(Path(out), text, {k: v for k, v in props.items() if not k.startswith("_")},
                                calc_type=props.get("calculation_type"), only=only,
                                dimensionality=CrystalPropertyExtractor._dimensionality(text))


def test_debye_moments_of_a_debye_spectrum_are_its_debye_temperature():
    """g(nu) = nu^2 up to nu_D: every Theta(n) is h c nu_D / k_B (1500 cm-1 = 2158.17 K)."""
    nu = np.linspace(0, 2000, 8001)
    g = np.where(nu <= 1500, nu ** 2, 0.0)
    th = VB.debye_moments(nu, g)
    for n in (-1, 0, 1, 2):
        assert th[n] == pytest.approx(1500 * 1.438776877, rel=2e-4)


def test_diamond_phonon_dos_debye_temperatures():
    nu, g = VB.pdos(F25)
    assert len(nu) == 160 and nu[1] - nu[0] == pytest.approx(10.0629)
    th = VB.debye_moments(nu, g)
    # Theta(n) rises with n; diamond's high-temperature Theta(2) from this 2x2x2 supercell
    assert th[-2] < th[-1] < th[0] < th[1] < th[2]
    assert th[2] == pytest.approx(1774.9, abs=1.0)


def test_acoustic_slopes_by_hand():
    """Γ→X of diamond: the header's DK is the segment's length (2 pi / a = 0.937968 bohr^-1 for
    a = 3.5448 Å); the slowest TA branch's first point, 26.157 cm-1 one step (DK/49) from Γ, gives
    v = 2 pi c nu / k = 13.62 km/s; the least-squares slope over the first 5 points agrees."""
    a_bohr = 3.54481120 / 0.529177210903
    recs = VB.acoustic_velocities(F25)
    gx = next(r for r in recs if r["to"] == "(4,0,4)")
    assert gx["segment_length_bohr_inv"] == pytest.approx(2 * math.pi / a_bohr, rel=1e-5)
    k1 = gx["segment_length_bohr_inv"] / 49 / 0.529177210903e-10
    hand = 2 * math.pi * 2.99792458e10 * 26.1566 / k1 / 1e3
    assert gx["velocities_km_s"][0] == pytest.approx(hand, rel=5e-3)
    # the path's L-Γ segment ends at Γ: walked backward
    assert any(r["to"] == "(4,4,4)" for r in recs)
    vm = VB.debye_average(recs)
    assert min(v for r in recs for v in r["velocities_km_s"]) < vm < max(v for r in recs for v in r["velocities_km_s"])
    # Theta = (h v_m / k_B)(3 n / 4 pi)^(1/3) for n = 8 atoms / a^3
    n = 8 / 3.54481120 ** 3
    assert VB.debye_from_sound(vm, n) == pytest.approx(
        6.62607015e-34 * vm * 1e3 / 1.380649e-23 * (3 * n * 1e30 / (4 * math.pi)) ** (1 / 3))


def test_phonon_properties_through_the_registry(suite, tmp_path):
    """The FREQ run's fort.25 beside it (<stem>.fort.25): both phonon properties stored."""
    shutil.copy(F25, tmp_path / F25.name)
    out = tmp_path / "dia_phonon_f25.out"
    shutil.copy(suite / "2-dense-3d" / "diamond" / "FREQ" / "diamond_freq.out", out)
    vals, rep = derived(out, ["phonon_debye_temperature", "phonon_sound_velocities"])
    assert rep["phonon_debye_temperature"]["status"] == "stored", rep
    assert rep["phonon_sound_velocities"]["status"] == "stored", rep
    assert vals["debye_temperature_phonon_k"] == pytest.approx(1774.9, abs=1.0)
    assert vals["phonon_debye_moments"]["theta_max_k"] > vals["debye_temperature_phonon_k"]
    assert vals["sound_velocity_phonon_ta_min_km_s"] < vals["sound_velocity_phonon_la_max_km_s"]
    # by hand: the run is a 2x2x2 SCELPHONO supercell (16 atoms in 89.086 Å³), the same density as the
    # conventional cell's 8 atoms in a^3; atoms_in_unit_cell (16) over primitive_cell_volume (11.136 Å³,
    # the primitive cell's) was 8x the density and 2x the temperature (6300 K)
    vm, a = vals["sound_velocity_phonon_mean_km_s"], 3.54481120
    hand = 6.62607015e-34 * vm * 1e3 / 1.380649e-23 * (3 * (8 / a ** 3) * 1e30 / (4 * math.pi)) ** (1 / 3)
    assert vals["debye_temperature_sound_k"] == pytest.approx(hand, rel=1e-5)
    assert vals["debye_temperature_sound_k"] == pytest.approx(3150, abs=5)
    assert VB.atom_density(out.read_text()) == pytest.approx(16 / 89.085972)
    # no fort.25: skipped, with the reason
    _, rep = derived(suite / "2-dense-3d" / "diamond" / "FREQ" / "diamond_freq.out", ["phonon_debye_temperature"])
    assert rep["phonon_debye_temperature"]["status"] == "skipped"


def test_molecule_mean_square_displacements(suite):
    """EC: CRYSTAL's classical amplitudes satisfy sum m x^2 omega = hbar to its 4 printed decimals;
    sum_i m_i <u_i^2>(0 K) = sum_nu hbar / (2 omega_nu) (atomic units) exactly; <u^2> grows with T and
    H moves most."""
    out = suite / "5-molecule-0d" / "EC" / "FREQ" / "EC_freq.out"
    vals, rep = derived(out, ["mean_square_displacement"])
    assert rep["mean_square_displacement"]["status"] == "stored", rep
    d = vals["mean_square_displacements"]
    assert d["modes_used"] == 24 and d["modes_left_out_zero"] == 6 and d["normalization_deviation"] < 3e-3
    text = out.read_text(errors="replace")
    m = np.asarray(VB.masses(text)) * VB.AMU_ME
    u0 = np.array([a["u2_0K_A2"] for a in d["atoms"]]) / 0.529177210903 ** 2
    nm = VB.normal_modes(out)
    w = [md["freq"] * VB.HARTREE_PER_CM for md in nm["modes"] if md["freq"] > 1.0]
    assert float((m * u0).sum()) == pytest.approx(sum(0.5 / x for x in w), rel=1e-4)
    assert vals["msd_mean_300k_ang2"] > vals["msd_mean_0k_ang2"]
    heavy = max(a["u2_300K_A2"] for a in d["atoms"] if a["element"] != "H")
    assert vals["msd_max_300k_ang2"] == pytest.approx(max(a["u2_300K_A2"] for a in d["atoms"]), abs=1e-6)
    assert max(a["u2_300K_A2"] for a in d["atoms"] if a["element"] == "H") > heavy
    a = d["atoms"][0]
    assert a["B_300K_A2"] == pytest.approx(8 * math.pi ** 2 * a["u2_300K_A2"] / 3, rel=1e-4)


def test_periodic_mean_square_displacement_is_not_applicable(suite):
    _, rep = derived(suite / "2-dense-3d" / "diamond" / "FREQ" / "diamond_freq.out", ["mean_square_displacement"])
    assert rep["mean_square_displacement"]["status"] == "skipped"
    assert "Γ" in rep["mean_square_displacement"]["reason"]
