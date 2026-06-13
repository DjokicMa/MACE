"""Drift guard for the duplicated space-group -> crystal-system logic.

Two independent implementations exist (intentionally — they have different
inputs and output vocabularies):

  * Crystal_d12/d12_constants.py  SPACEGROUP_TO_PATH  (number -> path key)
  * Crystal_d3/d3_kpoints.py      get_crystal_system_from_space_group(sg, lattice)
    (number + lattice-centering letter -> centering-aware k-path table key)

They are NOT merged (consolidation is the item-6 shared-module work, and the
k-path selection is validated logic we don't want to churn). Instead these
tests pin the property that matters: both must agree on the *base* crystal
system, and both must follow the immutable International Tables ranges — so a
future edit to either can't silently send a space group to the wrong table.
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
for sub in ("Crystal_d12", "Crystal_d3"):
    p = str(REPO_ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from d12_constants import SPACEGROUP_TO_PATH
from d3_kpoints import get_crystal_system_from_space_group


def _base(label: str) -> str:
    """Reduce a path key ('cubic_fc', 'monoclinic_simple') to its base system."""
    return label.split("_")[0]


# Canonical International Tables for Crystallography space-group ranges.
# (Trigonal 143-167 is folded into 'hexagonal' by BOTH implementations.)
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
def test_two_implementations_agree_on_base_system(sg):
    d12 = _base(SPACEGROUP_TO_PATH[sg])
    d3 = _base(get_crystal_system_from_space_group(sg, "P"))
    assert d12 == d3 == _canonical_base(sg), (
        f"space group {sg}: d12={d12!r} d3={d3!r} canonical={_canonical_base(sg)!r}")
