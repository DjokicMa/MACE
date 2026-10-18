"""scf_cycles and scf_fmixing_percentage read the lines CRYSTAL23 prints.

Their patterns ('SCF FIELD CONVERGENCE IN n CYCLES', 'FMIXING: n') never
matched a CRYSTAL23 output, so neither key was stored. The cycles are the
last SCF's: '== SCF ENDED ... CYCLES n' of an SP (and an OPT's first
point), the '(n)' of the last 'TOTAL ENERGY(DFT)(AU)( n)' line of an OPT's
later points; the Fock/KS mixing is 'FOCK/KS MATRIX MIXING SET TO n %'.
"""
from conftest import find_data


def _props(extractor, path):
    content = find_data(path).read_text(errors="replace")
    return {**extractor._extract_computational_properties(content),
            **extractor._extract_scf_settings(content)}


def test_sp_cycles_and_mixing(extractor):
    p = _props(extractor, "SP/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized.out")
    assert p["scf_cycles"] == 8
    assert p["scf_fmixing_percentage"] == 30


def test_multi_point_opt_cycles_are_the_last_scf(extractor):
    # point 1: SCF ENDED ... CYCLES 140; the last point (20): TOTAL ENERGY(DFT)(AU)( n)
    p = _props(extractor, "OPT/4LG_2x2_AA_opt_HSESOL3C_optimized.out")
    content = find_data("OPT/4LG_2x2_AA_opt_HSESOL3C_optimized.out").read_text(errors="replace")
    last = [l for l in content.splitlines() if "TOTAL ENERGY(DFT)(AU)(" in l][-1]
    assert p["scf_cycles"] == int(last.split("(AU)(")[1].split(")")[0]) != 140
