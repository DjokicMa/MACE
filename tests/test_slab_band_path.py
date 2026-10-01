"""A slab's BAND deck explores the plane of the slab only.

CRYSTAL23 manual p.310, BAND note 3: "in two-(one-) dimensional cases I3, J3
(I2,I3,J2,J3) are formally input as zero". A SLAB output says
"SLAB CALCULATION" and names its layer group and the corresponding space
group ("TWO-SIDED PLANE GROUP N. 80 : P 6/M M M", "CORRESPONDING SPACE GROUP
N. 191"; App. A.2 p.421: layer groups are identified by the corresponding
space group). The outputs used are real CRYSTAL23 slab outputs of graphene in
layer groups 37 (pmmm), 47 (cmmm) and 80 (p6/mmm).
"""
from pathlib import Path

import pytest

import CRYSTALOptToD3 as d3gen

DATA = Path(__file__).parent / "data" / "low_dim_groups"


@pytest.mark.parametrize("name,space_group,lattice", [
    ("graphene_lg37", 47, "P"),
    ("graphene_lg47", 65, "C"),
    ("graphene_lg80", 191, "P"),
])
def test_slab_symmetry_is_read(name, space_group, lattice):
    info = d3gen.D3Generator(str(DATA / f"{name}.out"), "BAND").structure_info
    assert info["dimensionality"] == 2
    assert info["space_group"] == space_group
    assert info["lattice_type"] == lattice
