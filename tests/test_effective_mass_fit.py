"""Effective masses come from a least-squares fit, not a 3-point difference.

The band-curvature mass was a 3-point second difference over BAND.DAT
energies printed to 1e-5 Ha with a k step of about 0.004 bohr^-1, so m*
could only take values near 1.76/n (diamond's CBM electron came out 0.885).
The window was also clipped at the path end (its center then sat off the
extremum), took the first of degenerate bands, and ran over both spin
blocks as one array. The mass now comes from a quadratic fit over about
+-0.05 bohr^-1 around the extremum, inside one path segment and one spin
block (one-sided at a segment end, with k0 fixed at the extremum).
"""
import pytest

from conftest import find_data

DIAMOND = "BAND/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized_band.BAND.DAT"


def _analyze(path):
    from mace.utils.advanced_electronic_analyzer import AdvancedElectronicAnalyzer
    return AdvancedElectronicAnalyzer().analyze_material(path, None)


def test_diamond_cbm_electron_mass_from_the_fit():
    r = _analyze(find_data(DIAMOND))
    # CBM along Gamma-X, longitudinal mass: an LSQ fit gives 1.52-1.61
    assert 1.4 <= r["electron_effective_mass"] <= 1.7


def test_diamond_vbm_hole_mass_is_the_band_edge_one():
    r = _analyze(find_data(DIAMOND))
    # VBM at Gamma (path start, one-sided window), triply degenerate: the
    # light-hole pair curves to 0.30 m_e, the band that stays at the edge
    # (the heavy hole) to 0.48 m_e along Gamma-X
    assert r["hole_effective_mass"] == pytest.approx(0.476, abs=0.02)

