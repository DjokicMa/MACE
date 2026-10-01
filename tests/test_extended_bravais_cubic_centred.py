"""Face- and body-centred cubic groups get SeeK-path's variants.

SeeK-path (HPKOT, Hinuma et al. 2017; seekpath 2.2.2) splits the
face-centred cubic groups like the primitive ones: cF1 for 196, 202 and 203
(groups below 207), cF2 for 209, 210, 216, 219 and 225-228. The static
lookup had the list the other way round for 202, 203 and 225-228, so
Fm-3m, Fd-3m (diamond, Si) and the rest got the cF1 path with its extra
X-W_2 segment, and F23 got cF2.
"""
import pytest

from d3_kpoints import get_extended_bravais, get_seekpath_full_kpath, seekpath_data

CF1 = [196, 202, 203]
CF2 = [209, 210, 216, 219, 225, 226, 227, 228]


@pytest.mark.parametrize("sg", CF1 + CF2)
def test_face_centred_variant_follows_hpkot(sg):
    assert get_extended_bravais(sg, "F") == ("cF1" if sg in CF1 else "cF2")


def test_fd3m_static_path_is_cf2():
    segments, info = get_seekpath_full_kpath(227, "F")
    assert info["lookup_key"] == "cF2"
    assert segments == seekpath_data["cF2"]["segments"]


def test_f23_static_path_is_cf1_noinv():
    segments, info = get_seekpath_full_kpath(196, "F")
    assert info["lookup_key"] == "cF1_noinv"
    assert segments == seekpath_data["cF1_noinv"]["segments"]


@pytest.mark.parametrize("sg", CF1 + CF2)
def test_cf_variant_matches_seekpath(sg, seekpath_path):
    result = seekpath_path(sg, (5.1, 5.1, 5.1, 90, 90, 90))
    assert get_extended_bravais(sg, "F") == result["bravais_lattice_extended"]
