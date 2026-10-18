"""An SP conducting in every SCF cycle stores a band gap of 0.

CRYSTAL prints POSSIBLY CONDUCTING STATE in each cycle and no BAND GAP
line, so no band_gap was stored and neither electronic_classification nor
conductivity_type, while a multi-point OPT ending conducting stores 0.0
and is classified a conductor.
"""
from conftest import find_data


def test_conducting_throughout_sp_has_gap_zero(extractor):
    p = extractor.extract_all_properties(find_data("SP/3,4^2T1-CA_rev1_sp_B3LYP-D3-D3_optimized.out"))
    assert p["band_gap"] == 0.0
    assert p["electronic_classification"] == "metal"   # 'conductor' before 1.1.6
    assert "conductivity_type" not in p                  # merged into electronic_classification
    # an unrestricted run (one conducting record for both spin channels)
    assert p["spin_polarized"] is True
    assert "band_gap_type" not in p
