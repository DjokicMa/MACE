"""Simple cubic groups get SeeK-path's cP1/cP2 split.

SeeK-path (HPKOT, Hinuma et al. 2017; seekpath/hpkot/__init__.py) uses cP1
for space groups 195-206 and cP2 for 207-230. cP1's path ends with an extra
M-X_1 segment, which cP2 groups (P432, P-43m, Pm-3m, ...) do not need, since
X and X_1 are equivalent there.
"""
import pytest

from d3_kpoints import get_extended_bravais, get_seekpath_full_kpath, seekpath_data

P_CUBIC = [195, 198, 200, 201, 205, 207, 208, 212, 213, 215, 218, 221, 222, 223, 224]


@pytest.mark.parametrize("sg", P_CUBIC)
def test_simple_cubic_variant_follows_hpkot(sg):
    assert get_extended_bravais(sg, "P") == ("cP1" if sg <= 206 else "cP2")


def test_pm3m_static_path_is_cp2():
    segments, info = get_seekpath_full_kpath(221, "P")
    assert info["lookup_key"] == "cP2"
    assert segments == seekpath_data["cP2"]["segments"]


def test_p432_static_path_is_cp2_noinv():
    segments, info = get_seekpath_full_kpath(207, "P")
    assert info["lookup_key"] == "cP2_noinv"
    assert segments == seekpath_data["cP2_noinv"]["segments"]
