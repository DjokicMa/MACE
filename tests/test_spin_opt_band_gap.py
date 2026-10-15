"""A spin-polarized OPT stores its band_gap.

After OPT END, CRYSTAL prints ``ALPHA BAND GAP`` and ``BETA BAND GAP`` for
a spin-polarized run. With that summary the extractor stored only
alpha_band_gap and beta_band_gap, never band_gap, and the classification
took the alpha channel. band_gap is now the material's gap as
_final_gap_value reports it: the fundamental gap of the last SCF when it
printed the band edges of the final geometry (a one-point OPT), else the
smaller channel gap of the summary; the classification uses the same value.
"""
from conftest import find_data

TI11O14 = ("OPT/Ti11O14_mp-759754_sg1_sym_CRYSTAL_OPT_symm_PBE-D3_full.basis.triplezeta_opt_"
           "B3LYP-D3-D3_optimized.out")


def _all(extractor, pattern):
    path = find_data(pattern)
    content = path.read_text(errors="ignore")
    props = extractor.extract_all_properties(path, material_id="x", calc_id="y")
    return props, extractor._final_gap_value(content)


def test_one_point_spin_opt(extractor):
    p, final = _all(extractor, "OPT/1LiFSI-1DEC-conf1_MOLECULE_OPT_symm_HSESOL3C_SOLDEF2MSVP_opt_"
                               "HSESOL3C_optimized.out")
    assert p["band_gap"] == 7.8402 == final
    assert p["classification_band_gap"] == 7.8402


def test_multi_point_spin_opt_uses_the_final_summary(extractor):
    p, final = _all(extractor, "OPT/3.4^2T137_BULK_OPTGEOM_TZ.out")
    assert p["band_gap"] == 2.5678 == final


def test_spin_channels_differ_smaller_one_everywhere(extractor):
    # ALPHA 0.1826, BETA 0.1403 eV after OPT END
    p, final = _all(extractor, TI11O14)
    assert p["band_gap"] == 0.1403 == final
    assert p["classification_band_gap"] == 0.1403


def test_classifier_spin_channels_without_band_gap(extractor):
    c = extractor._classify_electronic_properties({"alpha_band_gap": 0.1826, "beta_band_gap": 0.1403})
    assert c["classification_band_gap"] == 0.1403
