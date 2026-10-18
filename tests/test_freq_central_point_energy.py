"""A FREQ restart run's energy comes from its CENTRAL POINT line.

A FREQ run restarted from a wavefunction prints no SCF energy, only the
central point's TOTAL ENERGY (AU) in the displacement table and the EL
line of the thermodynamics, so most FREQ outputs stored no total energy.
That energy is the corrected total (it equals total_energy_corrected_au,
D3 and gCP included, wherever both are printed); the plain DFT energy and
the D3/gCP terms are not printed and are not stored.
"""
import pytest

from conftest import find_data, energy_props
from mace.constants import HARTREE_TO_EV

FREQ = "FREQ/1LiFSI-1DEC-conf1_*_freq_HSESOL3C_optimized.out"


def test_restart_freq_corrected_energy_from_central_point(extractor):
    p = energy_props(extractor, find_data(FREQ))
    assert p["total_energy_corrected_au"] == pytest.approx(-1.776438729722E+03, abs=1e-9)
    assert p["total_energy_corrected_ev"] == pytest.approx(-1.776438729722E+03 * HARTREE_TO_EV, rel=1e-12)
    assert p["electronic_energy_au"] == pytest.approx(p["total_energy_corrected_au"], abs=1e-8)
    for k in ("total_energy_au", "d3_dispersion_energy_au", "gcp_energy_au"):
        assert k not in p
