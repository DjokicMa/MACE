"""Mulliken A.O. populations keep their continuation lines.

CRYSTAL prints an atom's A.O. populations 8 per line and continues on
indented lines. The row lookahead took any line starting with spaces and a
number as the next atom, so the continuation was cut off (an O atom of a
pob-TZVP-rev2 molecule kept 8 of its 15 populations).
"""
from conftest import find_data

SP = "SP/1LiFSI-1DEC-conf1_*_sp_HSESOL3C_optimized.out"


def test_alpha_beta_rows_keep_continuation_lines(extractor):
    content = find_data(SP).read_text(errors="replace")
    ab = extractor._extract_mulliken_section(content, "ALPHA+BETA ELECTRONS")
    amb = extractor._extract_mulliken_section(content, "ALPHA-BETA ELECTRONS")
    o1 = ab["atoms"][0]
    assert (o1["atom_number"], o1["element"]) == (1, "O")
    assert o1["orbital_populations"] == [1.242, 0.711, 0.756, 1.154, 0.974, 0.835, 0.990, 0.693,
                                         0.532, 0.702, 0.005, 0.001, 0.011, 0.014, 0.012]
    s4 = ab["atoms"][3]
    assert s4["element"] == "S" and len(s4["orbital_populations"]) == 18
    assert len(ab["atoms"]) == len(amb["atoms"])
    for a, b in zip(ab["atoms"], amb["atoms"]):
        assert a["atom_number"] == b["atom_number"]
        assert len(a["orbital_populations"]) == len(b["orbital_populations"])
