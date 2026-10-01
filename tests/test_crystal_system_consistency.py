"""Drift guard for the space-group -> crystal-system logic of the k-paths.

Crystal_d3/d3_kpoints.py get_crystal_system_from_space_group(sg, lattice)
maps a number and a lattice-centering letter to a centering-aware k-path
table key. This pins the property that matters: it follows the immutable
International Tables ranges, so a future edit can't silently send a space
group to the wrong table.

It used to be checked against a second, independent implementation,
Crystal_d12/d12_constants.py SPACEGROUP_TO_PATH. That table ignored the
lattice centring, its only reader was a fallback in
d12_calc_freq.get_auto_phonon_path, and it has been removed
(tests/test_auto_phonon_path_centring.py).
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
for sub in ("Crystal_d12", "Crystal_d3"):
    p = str(REPO_ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from d3_kpoints import get_crystal_system_from_space_group


def _base(label: str) -> str:
    """Reduce a path key ('cubic_fc', 'monoclinic_simple') to its base system."""
    return label.split("_")[0]


# Canonical International Tables for Crystallography space-group ranges.
# (Trigonal 143-167 is folded into 'hexagonal' by the k-path tables.)
def _canonical_base(sg: int) -> str:
    if sg <= 2:
        return "triclinic"
    if sg <= 15:
        return "monoclinic"
    if sg <= 74:
        return "orthorhombic"
    if sg <= 142:
        return "tetragonal"
    if sg <= 194:
        return "hexagonal"
    return "cubic"


@pytest.mark.parametrize("sg", range(1, 231))
def test_base_system_follows_the_international_tables(sg):
    d3 = _base(get_crystal_system_from_space_group(sg, "P"))
    assert d3 == _canonical_base(sg), (
        f"space group {sg}: d3={d3!r} canonical={_canonical_base(sg)!r}")
