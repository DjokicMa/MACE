"""SLAB and POLYMER outputs have computed geometry blocks too.

A 3D output heads each computed block with "PRIMITIVE CELL - CENTRING CODE";
a SLAB or POLYMER output prints a plain "PRIMITIVE CELL" there (measured:
the graphene and polyyne CRYSTAL23 outputs in tests/data). Only the 3D form
was recognized, so a 2D or 1D caller got "no computed geometry block" and
the legacy reader.
"""
import contextlib
import io

import pytest

from conftest import REPO_ROOT

si = pytest.importorskip("seekpath_interface")
DATA = REPO_ROOT / "tests" / "data" / "low_dim_groups"


@pytest.mark.parametrize("name,a,n", [("graphene_lg80", 2.4612, 2), ("polyyne_rg28", None, None)])
def test_low_dim_outputs_have_a_computed_block(name, a, n):
    lines = (DATA / f"{name}.out").read_text().split("\n")
    blocks = si._geometry_blocks(lines)
    assert blocks, name
    block, which = si.select_geometry_block(blocks, lines)
    assert which == "first"
    assert block["params"] is not None and block["n_atoms"] == len(block["numbers"])
    if a is not None:
        assert block["params"]["a"] == a and block["n_atoms"] == n


def test_slab_structure_is_read_from_its_block():
    with contextlib.redirect_stdout(io.StringIO()) as printed:
        structure = si.parse_crystal_structure(str(DATA / "graphene_lg80.out"))
    assert structure["geometry"] == "first"
    assert "no computed geometry block" not in printed.getvalue()
