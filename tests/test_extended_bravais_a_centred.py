"""A-centred orthorhombic groups get SeeK-path's oA1/oA2 paths.

Amm2, Aem2, Ama2 and Aea2 (38-41) are A-centred; SeeK-path (HPKOT) gives
them oA1 when b < c and oA2 when b > c (conventional axes, standard
setting). The static lookup had no branch for an "A" lattice and fell
through to the primitive oP1 path, whose X, Y, S, U, ... are not the
special points of the A-centred zone.
"""
import itertools

import pytest

import d3_kpoints
from d3_kpoints import get_extended_bravais, get_seekpath_full_kpath, seekpath_data

A_GROUPS = [38, 39, 40, 41]
CELLS = [(5.1, 6.3, 7.4), (5.1, 7.4, 6.3), (7.4, 3.0, 6.3), (3.0, 12.0, 5.1)]

TEMPLATE = """ LATTICE PARAMETERS  (ANGSTROMS AND DEGREES) - CONVENTIONAL CELL
        A           B           C        ALPHA        BETA       GAMMA
 {:11.5f} {:11.5f} {:11.5f} {:11.5f} {:11.5f} {:11.5f}

"""


@pytest.mark.parametrize("sg", A_GROUPS)
@pytest.mark.parametrize("a,b,c", CELLS)
def test_a_centred_variant(sg, a, b, c):
    assert get_extended_bravais(sg, "A", a, b, c, 90, 90, 90) == ("oA1" if b < c else "oA2")


@pytest.mark.parametrize("sg", A_GROUPS)
def test_a_centred_without_cell_is_oa(sg):
    assert get_extended_bravais(sg, "A") == "oA1"


@pytest.mark.parametrize("a,b,c", CELLS)
def test_amm2_static_path_from_output(tmp_path, monkeypatch, a, b, c):
    monkeypatch.setattr(d3_kpoints, "SEEKPATH_LIBRARY_AVAILABLE", False)
    out = tmp_path / "amm2.out"
    out.write_text(TEMPLATE.format(a, b, c, 90, 90, 90))
    segments, info = get_seekpath_full_kpath(38, "A", str(out))
    key = "oA1_noinv" if b < c else "oA2_noinv"
    assert info["lookup_key"] == key
    assert segments == seekpath_data[key]["segments"]


@pytest.mark.parametrize("sg", A_GROUPS)
def test_oa_variant_matches_seekpath(sg, seekpath_path):
    for a, b, c in itertools.permutations((3.0, 5.1, 7.4)):
        result = seekpath_path(sg, (a, b, c, 90, 90, 90))
        assert get_extended_bravais(sg, "A", a, b, c, 90, 90, 90) == \
            result["bravais_lattice_extended"], (a, b, c)
