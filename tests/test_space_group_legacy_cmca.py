"""The legacy symbol "C M C A" of space group 64 (Cmce since 2002).

CRYSTAL23 prints "SPACE GROUP (CENTROSYMMETRIC) : C M C A" for the Ti13O22
outputs of the corpus; the symbol table held only CMCE, so they stored no
space group number.
"""
import pytest

from conftest import find_data


@pytest.mark.parametrize("path", [
    "OPT/Ti13O22_mp-656852_sg64_*_optimized.out",
    "SP/Ti13O22_mp-656852_sg64_*_sp_B3LYP-D3-D3_optimized.out",
])
def test_cmca_is_64(path):
    from mace.utils.formula_extractor import extract_space_group_from_output
    out = find_data(path, must_contain="C M C A")
    assert extract_space_group_from_output(out) == 64


def test_cmca_through_the_extractor(extractor):
    out = find_data("SP/Ti13O22_mp-656852_sg64_*_sp_B3LYP-D3-D3_optimized.out", must_contain="C M C A")
    assert extractor.extract_all_properties(out)["space_group_number"] == 64
