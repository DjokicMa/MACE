"""A SLAB child's k-point mesh keeps its layer group's symmetry.

CRYSTAL applies SHRINK along the primitive reciprocal vectors and stops with
"ERROR **** CAPPA **** SHRINK BREAKS SYMMETRY" when two directions the layer
group relates get different factors. opt2d12 (from a .out alone) and cif2d12
generate the mesh from the conventional a and b, and only equalised it for 3D
decks, so graphene in the centred-rectangular layer group 47 (a 2.46, b 4.26)
got SHRINK 18 10 1 and CRYSTAL refused it.

Measured with CRYSTAL/23-intel-2023a on HPCC (SCF runs, not preflight):
    layer group 47, 18 10 1            -> SHRINK BREAKS SYMMETRY
    layer group 47, 18 18 1 (opt2d12)  -> SCF converged
    layer group 47, 18 9 1 (cif2d12)   -> SHRINK BREAKS SYMMETRY
    layer group 47, 18 18 1 (cif2d12)  -> SCF converged
    layer group 37, 18 10 1            -> SCF converged (primitive rectangular)
    layer group 80, 18 18 1            -> SCF converged
    rod groups 28 and 51, 16 1 1       -> SCF converged
A child made with the parent .d12 present takes the parent's own SHRINK and is
unchanged.
"""
import shutil
import subprocess
import sys

import pytest

from conftest import REPO_ROOT
from d12_constants import (
    CENTRED_RECTANGULAR_LAYER_GROUPS,
    LAYER_GROUP_ROWS,
    slab_k_points,
)

MACE_CLI = REPO_ROOT / "mace_cli"
LOW_DIM_DATA = REPO_ROOT / "tests" / "data" / "low_dim_groups"


def test_centred_rectangular_groups_are_the_appendix_c_rows():
    # CRYSTAL23's own numbering, measured: its output names these nine groups
    # (and no others of the 80) with a C lattice symbol - 16 is "C 2/M 1 1",
    # 18 is "P 21/B 1 1", unlike the International Tables order.
    assert CENTRED_RECTANGULAR_LAYER_GROUPS == {10, 13, 16, 22, 26, 35, 36, 47, 48}
    assert all(LAYER_GROUP_ROWS[g - 1][1].startswith("C")
               for g in CENTRED_RECTANGULAR_LAYER_GROUPS)


@pytest.mark.parametrize("group,mesh,cell,expected", [
    # centred rectangular: one factor for both in-plane directions
    (47, (18, 10, 1), [2.4612, 4.26292264], (18, 18, 1)),
    (10, (10, 18, 1), None, (18, 18, 1)),
    # ...raised to what the (shorter) primitive vector needs: a 3, b 4 gives
    # a primitive length of 2.5, which needs 18 where max(15, 12) is 15.
    (26, (15, 12, 1), [3.0, 4.0], (18, 18, 1)),
    # square and hexagonal lattices
    (61, (12, 10, 1), None, (12, 12, 1)),
    (80, (18, 16, 1), None, (18, 18, 1)),
    # nothing relates a and b: unchanged
    (1, (18, 10, 1), None, (18, 10, 1)),
    (37, (18, 10, 1), [2.4612, 4.26292], (18, 10, 1)),
    (47, (18, 18, 1), [2.4612, 4.26292264], (18, 18, 1)),
    # not a layer group
    (None, (18, 10, 1), None, (18, 10, 1)),
    (191, (18, 10, 1), None, (18, 10, 1)),
    (True, (18, 10, 1), None, (18, 10, 1)),
])
def test_slab_k_points(group, mesh, cell, expected):
    assert slab_k_points(mesh, group, cell) == expected


def _shrink(deck_text):
    lines = deck_text.splitlines()
    i = lines.index("SHRINK")
    return lines[i + 1:i + 3]


def _child(tmp_path, name, with_deck):
    shutil.copy(LOW_DIM_DATA / f"{name}.out", tmp_path)
    args = [sys.executable, str(MACE_CLI), "opt2d12", "--out-file", f"{name}.out",
            "--non-interactive", "--calc-type", "SP"]
    if with_deck:
        shutil.copy(LOW_DIM_DATA / f"{name}.d12", tmp_path)
        args += ["--d12-file", f"{name}.d12"]
    result = subprocess.run(args, cwd=tmp_path, input="", capture_output=True,
                            text=True, timeout=300)
    assert result.returncode == 0, (result.stdout + result.stderr)[-1500:]
    kids = [p for p in tmp_path.glob("*.d12") if p.name != f"{name}.d12"]
    assert len(kids) == 1, sorted(p.name for p in tmp_path.iterdir())
    return kids[0].read_text()


@pytest.mark.parametrize("name,shrink", [
    ("graphene_lg47", ["0 36", "18 18 1"]),   # was 18 10 1: refused by CRYSTAL
    ("graphene_lg37", ["0 36", "18 10 1"]),   # primitive rectangular: kept
    ("graphene_lg80", ["0 36", "18 18 1"]),
    ("polyyne_rg28", ["0 32", "16 1 1"]),
    ("polyyne_rg51", ["0 32", "16 1 1"]),
])
def test_out_only_child_mesh_respects_the_group(tmp_path, name, shrink):
    assert _shrink(_child(tmp_path, name, with_deck=False)) == shrink


@pytest.mark.parametrize("name", ["graphene_lg47", "graphene_lg37", "graphene_lg80"])
def test_child_with_the_parent_deck_keeps_the_parents_shrink(tmp_path, name):
    parent = (LOW_DIM_DATA / f"{name}.d12").read_text()
    assert _shrink(_child(tmp_path, name, with_deck=True)) == _shrink(parent)


_CMMM_CIF = """data_t
_cell_length_a 2.4612
_cell_length_b 5.0
_cell_length_c 20.0000
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
_space_group_name_H-M_alt 'C m m m'
_space_group_IT_number 65
loop_
_space_group_symop_operation_xyz
'x, y, z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
C1 C 0.0 0.333333 0.0 1.0
C2 C 0.0 0.666667 0.0 1.0
C3 C 0.5 0.833333 0.0 1.0
C4 C 0.5 0.166667 0.0 1.0
"""


def test_cif2d12_centred_rectangular_slab_mesh(tmp_path):
    """cif2d12 wrote 18 9 1 for this layer group 47 slab (refused by CRYSTAL)."""
    pytest.importorskip("ase", reason="NewCifToD12 imports ase.io")
    import NewCifToD12 as writer

    cif_dir = tmp_path / "cif"
    cif_dir.mkdir()
    (cif_dir / "t.cif").write_text(_CMMM_CIF)
    opts = {
        "dimensionality": "SLAB", "calculation_type": "SP", "method": "DFT",
        "functional": "PBE", "dft_functional": "PBE",
        "basis_set": "POB-TZVP-REV2", "basis_set_type": "INTERNAL",
        "dft_grid": "XLGRID", "use_dispersion": False, "dispersion": False,
        "is_spin_polarized": False, "spin_polarized": False,
        "tolerances": {"TOLINTEG": "7 7 7 7 14", "TOLDEE": 7},
        "shrink": [4, 4], "k_points": 4, "scf_maxcycle": 100, "fmixing": 30,
        "scf_method": "DIIS", "symmetry_handling": "CIF",
        "optimization_settings": {}, "freq_settings": {}, "layer_group": 47,
    }
    cif = writer.parse_cif(str(cif_dir / "t.cif"))
    cif = writer.verify_and_reduce_to_asymmetric_unit(cif, 1e-5, False)
    out = tmp_path / "t.d12"
    assert writer.create_d12_file(cif, str(out), opts)
    text = out.read_text()
    assert text.splitlines()[1:5] == ["SLAB", "47", "2.46120000 5.00000000", "1"]
    assert _shrink(text) == ["0 36", "18 18 1"]
