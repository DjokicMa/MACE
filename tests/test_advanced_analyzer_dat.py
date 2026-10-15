"""AdvancedElectronicAnalyzer reads wrapped DOSS.DAT rows and BAND-only gaps.

read_dos_data took every physical line of a DOSS.DAT as an energy point.
CRYSTAL wraps a record with many projections over several lines (NPROJ 26:
17 + 10 fields), so a 1002-point spin-polarized file read as 4008 points,
continuation lines as energies, and a molecule's gap came out as 68.76 eV.
It now uses DatFileProcessor's DOSS.DAT parser (the extractor's own), one
energy grid with the spin channels summed. A gap is no longer read off the
DOS: CRYSTAL puts E_F on a molecule's broadened HOMO peak and a DOSS band
range leaves states out, so it was not the material's gap; with a DOSS.DAT
alone neither a gap nor a classification is reported. A BAND.DAT alone
returned gap 0.0 and 'unknown'; its gap now comes from the band edges.
"""
import pytest

from conftest import find_data

STEM = "_MOLECULE_OPT_symm_HSESOL3C_SOLDEF2MSVP_opt_HSESOL3C_optimized_CRYSTAL_SP_symm_HSESOL3C_SOLDEF2MSVP"


def _analyzer():
    from mace.utils.advanced_electronic_analyzer import AdvancedElectronicAnalyzer
    return AdvancedElectronicAnalyzer()


def test_wrapped_doss_rows_are_one_energy_grid():
    doss = find_data("DOSS/1LiFSI-1DEC-conf1" + STEM + "_doss.DOSS.DAT")
    E, g = _analyzer().read_dos_data(doss)
    assert len(E) == len(g) == 1002                       # NEPTS 1002, two spin blocks
    assert E[0] == -7.8972E-02 and (E[1:] > E[:-1]).all()  # one ascending grid
    assert (g >= 0).all()                                 # beta (negative) summed as states


def test_doss_alone_reports_no_gap_or_classification():
    doss = find_data("DOSS/1LiFSI-1DEC-conf1" + STEM + "_doss.DOSS.DAT")
    r = _analyzer().analyze_material(None, doss)
    assert r["dos_points"] == 1002
    assert r["gap_ev"] is None and r["gap_hartree"] is None
    assert r["electronic_classification"] is None


def test_band_alone_gap_from_the_band_edges():
    band = find_data("BAND/1LiFSI-1DEC-conf4" + STEM + "_band.BAND.DAT")
    r = _analyzer().analyze_material(band, None)
    # the extractor's own BAND.DAT gap of this output (band_dat_band_gap_ev)
    assert r["gap_ev"] == pytest.approx(7.8377, abs=1e-3)
    assert r["gap_source"] == "band_structure"
    assert r["electronic_classification"] == "insulator"


def test_extractor_stores_no_dos_only_gap(extractor):
    out = find_data("DOSS/1LiFSI-1DEC-conf1" + STEM + "_doss.out")
    props = extractor._extract_advanced_band_dos_analysis(out, {})
    assert props["advanced_analysis_available"] is True
    assert "band_gap_advanced_ev" not in props and "electronic_classification_advanced" not in props


def test_hole_mass_without_electron_mass():
    # the scattering constants were set only with an electron mass
    t = _analyzer().calculate_transport_properties(
        {"electronic_classification": "semiconductor", "gap_ev": 1.0},
        {"electron_effective_mass": None, "hole_effective_mass": 0.5})
    assert t["hole_mobility_estimate"] > 0
