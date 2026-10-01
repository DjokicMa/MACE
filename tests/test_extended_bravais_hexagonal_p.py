"""Primitive hexagonal and trigonal groups get SeeK-path's hP1/hP2 split.

SeeK-path (HPKOT, Hinuma et al. 2017; seekpath 2.2.2) uses hP2 for space
groups 150, 152, 154, 156, 158, 164, 165 and 168-194, and hP1 for the other
primitive trigonal groups (143-145, 147, 149, 151, 153, 157, 159, 162, 163).
hP1's path ends with an extra K-H_2 segment, which hP2 groups do not need
since H and H_2 are equivalent there. Without the seekpath library every
primitive hexagonal group got hP1. The group lists were read from seekpath
2.2.2 on a structure of each group (test_hp_variant_matches_seekpath does
the same where seekpath is installed).
"""
import pytest

from d3_kpoints import get_extended_bravais, get_seekpath_full_kpath, seekpath_data

HP2 = [150, 152, 154, 156, 158, 164, 165] + list(range(168, 195))
HP1 = [143, 144, 145, 147, 149, 151, 153, 157, 159, 162, 163]


@pytest.mark.parametrize("sg", HP1 + HP2)
def test_hexagonal_variant_follows_hpkot(sg):
    expected = "hP2" if sg in HP2 else "hP1"
    assert get_extended_bravais(sg, "P") == expected
    assert get_extended_bravais(sg, "P", 3.1, 3.1, 5.0, 90, 90, 120) == expected


def test_p6_mmm_static_path_is_hp2():
    segments, info = get_seekpath_full_kpath(191, "P")
    assert info["lookup_key"] == "hP2"
    assert segments == seekpath_data["hP2"]["segments"]


def test_p321_static_path_is_hp2_noinv():
    segments, info = get_seekpath_full_kpath(150, "P")
    assert info["lookup_key"] == "hP2_noinv"
    assert segments == seekpath_data["hP2_noinv"]["segments"]


def test_p_31m_keeps_hp1():
    segments, info = get_seekpath_full_kpath(162, "P")
    assert info["lookup_key"] == "hP1"


@pytest.mark.parametrize("sg", HP1 + HP2)
def test_hp_variant_matches_seekpath(sg, seekpath_path):
    result = seekpath_path(sg, (3.1, 3.1, 5.0, 90, 90, 120))
    assert get_extended_bravais(sg, "P") == result["bravais_lattice_extended"]
