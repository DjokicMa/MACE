"""generate_k_points follows its k*a density window for long lattice vectors.

The rule picks the smallest shrinking factor k with 40 < k*a < 80 Angstrom.
A vector of 40 Angstrom or more already reaches that density with one
k-point, so it gets k = 1. Before the fix the search started at k = 2, never
matched for such vectors, and the "unset" fallback gave them 12 - a 44 x 44 x
48 Angstrom molecule box was written with SHRINK 0 24 / 12 12 12.
"""

import pytest

from Crystal_d12.d12_constants import generate_k_points


@pytest.mark.parametrize("length", [40.0, 44.0, 48.0, 79.9, 80.0, 120.0, 500.0])
def test_a_long_vector_gets_one_k_point(length):
    assert generate_k_points(length, length, length, "CRYSTAL", 1) == (1, 1, 1)


def test_the_molecule_box_from_the_corpus_gets_a_gamma_mesh():
    # 1LiFSI-1EC-conf1: an isolated cluster in a 44 x 44 x 48 Angstrom box.
    assert generate_k_points(44.0, 44.0, 48.0, "CRYSTAL", 1) == (1, 1, 1)


def test_k_never_grows_with_the_length():
    lengths = [x / 4 for x in range(8, 800)]  # 2 .. 200 Angstrom
    ks = [generate_k_points(a, a, a, "CRYSTAL", 1)[0] for a in lengths]
    assert all(k1 >= k2 for k1, k2 in zip(ks, ks[1:]))


def test_mixed_long_and_short_vectors():
    # A slab-like cell kept as 3D: short in-plane, long stacking axis.
    assert generate_k_points(5.0, 5.0, 60.0, "CRYSTAL", 1) == (9, 9, 1)
    assert generate_k_points(5.0, 5.0, 60.0, "SLAB", 1) == (9, 9, 1)
    assert generate_k_points(50.0, 1.0, 1.0, "POLYMER", 1) == (1, 1, 1)


@pytest.mark.parametrize("cell,dim,sg,expected", [
    ((5.43, 5.43, 5.43), "CRYSTAL", 227, (8, 8, 8)),
    ((3.0, 3.0, 3.0), "CRYSTAL", 1, (15, 15, 15)),
    ((10.0, 12.0, 30.0), "CRYSTAL", 1, (5, 4, 2)),
    ((2.46, 2.46, 20.0), "SLAB", 1, (18, 18, 1)),
    ((39.9, 39.9, 39.9), "CRYSTAL", 1, (2, 2, 2)),
    ((0.5, 0.5, 0.5), "CRYSTAL", 1, (12, 12, 12)),  # below the window: unchanged fallback
])
def test_short_cells_are_unchanged(cell, dim, sg, expected):
    assert generate_k_points(*cell, dim, sg) == expected
