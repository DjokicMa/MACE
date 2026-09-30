"""Primitive monoclinic groups keep the mP1 SeeK-path when cell parameters are known.

get_extended_bravais hands a P monoclinic group with cell parameters to
determine_monoclinic_variant, which split P from C at space group 11. P2/c
(13) and P2_1/c (14) are primitive, but got the C-centred mS1 path - points
of another Brillouin zone - whenever the band or phonon path was built from
a CRYSTAL output without the seekpath library. SeeK-path (HPKOT) gives every
primitive monoclinic lattice mP1; the C-centred groups are 5, 8, 9, 12, 15.
"""
import pytest

import d3_kpoints
from d3_kpoints import get_extended_bravais, get_seekpath_full_kpath, seekpath_data

CELL = (5.1, 6.3, 7.4, 90.0, 104.0, 90.0)

OUTPUT = """\
 SPACE GROUP (CENTROSYMMETRIC)        :  P 1 21/C 1

 LATTICE PARAMETERS  (ANGSTROMS AND DEGREES) - CONVENTIONAL CELL
        A           B           C        ALPHA        BETA       GAMMA
     5.10000     6.30000     7.40000    90.00000   104.00000    90.00000

"""


@pytest.mark.parametrize("sg", [3, 4, 6, 7, 10, 11, 13, 14])
def test_primitive_monoclinic_is_mp1(sg):
    assert get_extended_bravais(sg, "P", *CELL) == "mP1"
    assert get_extended_bravais(sg, "P") == "mP1"


@pytest.mark.parametrize("sg", [5, 8, 9, 12, 15])
def test_c_centred_monoclinic_is_unchanged(sg):
    assert get_extended_bravais(sg, "C", *CELL) == "mS1"
    assert get_extended_bravais(sg, "C") == "mS1"


def test_p21c_path_from_an_output_is_mp1(tmp_path, monkeypatch):
    # the static table's route, as without the seekpath library
    monkeypatch.setattr(d3_kpoints, "SEEKPATH_LIBRARY_AVAILABLE", False)
    out = tmp_path / "p21c.out"
    out.write_text(OUTPUT)
    segments, info = get_seekpath_full_kpath(14, "P", str(out))
    assert info["extended_bravais"] == "mP1"
    assert segments == seekpath_data["mP1"]["segments"]
