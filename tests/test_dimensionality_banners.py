"""CRYSTAL's dimensionality banners, as CRYSTAL23 prints them near the top of
an output: "CRYSTAL CALCULATION", "SLAB CALCULATION", "POLYMER CALCULATION"
and "MOLECULAR CALCULATION" (331 corpus molecule outputs; never "MOLECULE
CALCULATION"). The GUI looked for "MOLECULE CALCULATION", so a molecule was
told only by the later "DIMENSIONALITY OF THE SYSTEM 0" line: an output still
being written, cut before that line, read as a 3D crystal. The workflow
planner's stand-in outputs wrote the wrong banner too, so opt2d3 read a
planned molecule as 3D."""
from pathlib import Path

import pytest

from conftest import REPO_ROOT, TEST_DATA

EC = TEST_DATA / "OPT" / "EC_MOLECULE_OPT_symm_HSESOL3C_SOLDEF2MSVP_opt_HSESOL3C_optimized.out"
SLAB = TEST_DATA / "SP" / "4LG_FSI_2x2_AA_graphene_opt_sp.out"
POLY = REPO_ROOT / "tests" / "data" / "low_dim_groups" / "polyyne_rg4.out"
CRYS = TEST_DATA / "OPT" / "1_dia_opt_rev1.out"


def _head_before_dim_line(path):
    """The output as written up to its banner, before any DIMENSIONALITY line (a live run)."""
    lines = Path(path).read_text(errors="replace").splitlines()
    k = next(i for i, x in enumerate(lines) if "DIMENSIONALITY OF THE SYSTEM" in x)
    return lines[:k]


@pytest.mark.parametrize("path,dim", [(EC, 0), (POLY, 1), (SLAB, 2), (CRYS, 3)])
def test_banner_alone_tells_the_dimensionality(path, dim):
    if not Path(path).is_file():
        pytest.skip("test/ corpus not present")
    from mace.gui import structures, filekind
    head = _head_before_dim_line(path)
    assert any("CALCULATION" in x for x in head)
    assert structures._out_dimensionality(head) == dim
    assert filekind.dim_of_head(head) == dim


def test_molecule_banner_is_crystals_own():
    from mace.gui import structures
    banners = dict(structures._DIM_BANNERS)
    assert banners.get("MOLECULAR CALCULATION") == 0 and "MOLECULE CALCULATION" not in banners


def test_planner_stand_in_output_names_a_molecule_as_crystal_does(tmp_path):
    from mace.workflow.dummy_file_creator import DummyFileCreator
    out = tmp_path / "mol.out"
    DummyFileCreator().create_dummy_out(out, {"dimensionality": "MOLECULE", "title": "mol"})
    text = out.read_text()
    assert " MOLECULAR CALCULATION\n" in text and "MOLECULE CALCULATION" not in text
