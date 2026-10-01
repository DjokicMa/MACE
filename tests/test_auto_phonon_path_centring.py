"""get_auto_phonon_path's coordinate paths follow the lattice centring.

A coordinate format other than "vectors" (e.g. "coordinates") fell through to
d12_constants.SPACEGROUP_TO_PATH / HIGH_SYMMETRY_PATHS, which map a space-group
number to one path per crystal system: C2/m got the primitive monoclinic
path, Fmmm and Immm the primitive orthorhombic one, I4/mmm the primitive
tetragonal one and R-3m the hexagonal one. That was the tables' only reader
(get_auto_phonon_path itself has no caller in MACE), so they are removed and
every coordinate format takes the centring-aware path the "vectors" format
already used (d3_kpoints.get_band_path_from_symmetry).
"""
import pytest

import d12_calc_freq
import d12_constants


@pytest.mark.parametrize("space_group, lattice", [(12, "C"), (38, "A"), (69, "F"), (71, "I"),
                                                  (139, "I"), (166, "R"), (225, "F")])
def test_coordinate_path_is_the_centring_aware_path(space_group, lattice):
    vectors = d12_calc_freq.get_auto_phonon_path(
        None, space_group, shrink=16, format_type="vectors", lattice_type=lattice)
    other = d12_calc_freq.get_auto_phonon_path(
        None, space_group, shrink=16, format_type="coordinates", lattice_type=lattice)
    assert other == vectors


def test_the_centring_blind_tables_are_gone():
    assert not hasattr(d12_constants, "SPACEGROUP_TO_PATH")
    assert not hasattr(d12_constants, "HIGH_SYMMETRY_PATHS")
