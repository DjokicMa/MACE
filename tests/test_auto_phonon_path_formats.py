"""get_auto_phonon_path gives a path in the "labels" and "vectors" formats.

Both formats go through get_band_path_from_symmetry, guarded by a check on
get_crystal_system_from_space_group - which d12_calc_freq never imported, so
either format stopped with NameError before returning anything.
"""
import pytest

import d12_calc_freq
from d3_kpoints import get_band_path_from_symmetry, get_kpoint_coordinates_from_labels


@pytest.mark.parametrize("space_group, lattice", [(221, "P"), (225, "F"), (194, "P")])
def test_labels_format(space_group, lattice):
    labels = get_band_path_from_symmetry(space_group, lattice)
    segments = d12_calc_freq.get_auto_phonon_path(
        None, space_group, shrink=0, format_type="labels", lattice_type=lattice)
    assert segments == [f"{a} {b}" for a, b in zip(labels[:-1], labels[1:])]


@pytest.mark.parametrize("space_group, lattice", [(221, "P"), (225, "F"), (194, "P")])
def test_vectors_format(space_group, lattice):
    labels = get_band_path_from_symmetry(space_group, lattice)
    frac = get_kpoint_coordinates_from_labels(labels, space_group, lattice)
    segments = d12_calc_freq.get_auto_phonon_path(
        None, space_group, shrink=12, format_type="vectors", lattice_type=lattice)
    assert segments == [[int(round(v * 12)) for v in seg] for seg in frac]
    assert segments and all(len(s) == 6 for s in segments)
