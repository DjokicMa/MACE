"""The final energies of a multi-point geometry optimization.

From the second point of a CRYSTAL OPT on (and from the first point of a
restarted one), the ``TOTAL ENERGY(DFT)(AU)`` line already includes the D3
dispersion and, for a 3c method, the gCP correction: point 2's ``DE`` is
measured from point 1's ``TOTAL ENERGY + DISP (+ GCP)`` value, and the
``OPT END - CONVERGED * E(AU)`` energy equals the last ``TOTAL ENERGY(DFT)``
line. The extractor took that line as the plain DFT energy and added the
D3 and gCP printed after OPT END once more, so ``total_energy_corrected_au``
counted them twice (Ti11O14 by -0.63 AU, 4LG by -0.094 AU),
``total_energy_au`` held the corrected value, and
``total_energy_plus_d3_au`` came from point 1's printed line. A
one-point OPT (whose only DFT line is the plain energy) and an SP are
unchanged.
"""
import pytest

from conftest import find_data, energy_props
from mace.constants import HARTREE_TO_EV


def _check(p, end, d3, gcp=0.0):
    assert p["total_energy_corrected_au"] == pytest.approx(end, abs=1e-8)
    assert p["total_energy_au"] == pytest.approx(end - d3 - gcp, abs=1e-8)
    assert p["total_energy_plus_d3_au"] == pytest.approx(end - gcp, abs=1e-8)
    assert p["d3_dispersion_energy_au"] == pytest.approx(d3, abs=1e-10)
    for au in ("total_energy", "total_energy_corrected", "total_energy_plus_d3"):
        assert p[f"{au}_ev"] == pytest.approx(p[f"{au}_au"] * HARTREE_TO_EV, rel=1e-12)


def test_multi_point_d3_opt_counts_dispersion_once(extractor):
    out = find_data("OPT/Ti11O14_mp-759754_sg1_sym_CRYSTAL_OPT_symm_PBE-D3_full.basis."
                    "triplezeta_opt_B3LYP-D3-D3_optimized.out")
    _check(energy_props(extractor, out), end=-1.039900366138E+04, d3=-6.275276152109E-01)


def test_multi_point_3c_opt_counts_d3_and_gcp_once(extractor):
    out = find_data("SP/4LG_2x2_AA_opt_HSESOL3C_optimized.out")
    _check(energy_props(extractor, out), end=-3.644677628625E+03,
           d3=-2.077808904511E-01, gcp=1.134282559455E-01)


def test_restarted_opt_first_point_already_includes_corrections(extractor):
    out = find_data("OPT/4LG_2x2_ABAB_opt_HSESOL3C_optimized.out")
    _check(energy_props(extractor, out), end=-3.644677406813E+03,
           d3=-2.077938811273E-01, gcp=1.134282660767E-01)


def test_one_point_opt_is_unchanged(extractor):
    out = find_data("OPT/1LiFSI-1DEC-conf1_MOLECULE_OPT_symm_HSESOL3C_SOLDEF2MSVP_opt_"
                    "HSESOL3C_optimized.out")
    p = energy_props(extractor, out)
    # its DFT line is the plain energy: the components add up to OPT END
    assert p["total_energy_au"] == -1.7765192784513E+03
    assert p["total_energy_corrected_au"] == pytest.approx(-1.776438618139E+03, abs=1e-8)
    assert p["total_energy_corrected_au"] == pytest.approx(
        p["total_energy_au"] + p["d3_dispersion_energy_au"] + p["gcp_energy_au"], abs=1e-12)
