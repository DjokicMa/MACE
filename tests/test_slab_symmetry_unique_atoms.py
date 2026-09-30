"""A SLAB deck lists only the atoms that are symmetry-unique in its layer group.

A SLAB atom record is "z in Angstrom, x, y in fractional units" and the count
before it is "NATR number of non-equivalent atoms in the asymmetric unit"
(CRYSTAL23 manual, page 21). CRYSTAL builds the rest of the slab from them
with the layer group's operators, whose z-reversing elements sit at z = 0.

The converter wrote every atom it was handed, with z = c * fractional z. For a
Bi2 bilayer (P -3 m 1, one Bi at (1/3, 2/3, 0.0449), layer group 72) handed
the whole cell - what the batch path keeps when spglib's group disagrees with
the CIF's, or what ``write_only_unique = false`` keeps - that gave a second Bi
at (2/3, 1/3, 19.101645): the inversion image of the first, but 19.1 A above
the layer instead of 0.9 A below it, so CRYSTAL built a four-atom structure
38 A thick.
"""
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "Crystal_d12"))

from d12_constants import DEFAULT_TOLERANCES  # noqa: E402

BASIS_TZ = str(REPO_ROOT / "Crystal_d12" / "basis_sets" / "full.basis.triplezeta") + "/"

BI2_CIF = """data_Bi2
_symmetry_space_group_name_H-M   'P -3 m 1'
_cell_length_a   4.29363
_cell_length_b   4.29363
_cell_length_c   20.0
_cell_angle_alpha   90.0
_cell_angle_beta   90.0
_cell_angle_gamma   120.0
_symmetry_Int_Tables_number   164
loop_
 _symmetry_equiv_pos_site_id
 _symmetry_equiv_pos_as_xyz
  1  'x, y, z'
  2  '-y, x-y, z'
  3  '-x+y, -x, z'
  4  'y, x, -z'
  5  'x-y, -y, -z'
  6  '-x, -x+y, -z'
  7  '-x, -y, -z'
  8  'y, -x+y, -z'
  9  'x-y, x, -z'
  10  '-y, -x, z'
  11  '-x+y, y, z'
  12  'x, x-y, z'
loop_
 _atom_site_type_symbol
 _atom_site_label
 _atom_site_symmetry_multiplicity
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
  Bi  Bi0  2  0.333333333333  0.666666666667  0.044917764  1
"""

# Non-centrosymmetric and polar in z: a buckled honeycomb with its two atoms
# on different elements (P 3 m 1, layer group 69). No operation reverses z,
# so every z is kept as written.
BN_CIF = """data_BN
_symmetry_space_group_name_H-M   'P 3 m 1'
_cell_length_a   2.5
_cell_length_b   2.5
_cell_length_c   20.0
_cell_angle_alpha   90.0
_cell_angle_beta   90.0
_cell_angle_gamma   120.0
_symmetry_Int_Tables_number   156
loop_
 _symmetry_equiv_pos_site_id
 _symmetry_equiv_pos_as_xyz
  1  'x, y, z'
  2  '-y, x-y, z'
  3  '-x+y, -x, z'
  4  '-y, -x, z'
  5  '-x+y, y, z'
  6  'x, x-y, z'
loop_
 _atom_site_type_symbol
 _atom_site_label
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
  B  B1  0.333333333333  0.666666666667  0.0  1
  N  N1  0.666666666667  0.333333333333  0.01  1
"""


@pytest.fixture
def writer():
    pytest.importorskip("ase", reason="NewCifToD12 imports ase.io")
    import NewCifToD12

    return NewCifToD12


def slab_options(**overrides):
    opts = dict(
        dimensionality="SLAB",
        calculation_type="SP",
        basis_set_type="EXTERNAL",
        basis_set=BASIS_TZ,
        method="DFT",
        dft_functional="PBE",
        is_spin_polarized=False,
        tolerances=DEFAULT_TOLERANCES,
        scf_method="DIIS",
        symmetry_handling="CIF",
    )
    opts.update(overrides)
    return opts


def atom_records(deck_text):
    lines = deck_text.splitlines()
    assert lines[1] == "SLAB"
    natoms = int(lines[4])
    return [lines[5 + i].split() for i in range(natoms)]


def write(writer, tmp_path, cif_text, name, **options):
    cif = tmp_path / f"{name}.cif"
    cif.write_text(cif_text)
    data = writer.parse_cif(str(cif), interactive=False)
    out = tmp_path / f"{name}.d12"
    assert writer.create_d12_file(
        data, str(out), slab_options(**options), interactive=False
    )
    return data, atom_records(out.read_text())


def test_bi2_full_cell_writes_one_bi_at_its_own_z(writer, tmp_path):
    data, rows = write(writer, tmp_path, BI2_CIF, "bi2", layer_group=72)
    assert len(data["symbols"]) == 2, "ASE hands the converter the whole cell"
    assert len(rows) == 1, rows
    nat, x, y, z = rows[0][:4]
    assert nat == "283"  # Bi with the external basis set's ECP
    assert float(x) == pytest.approx(1 / 3, abs=1e-9)
    assert float(y) == pytest.approx(2 / 3, abs=1e-9)
    assert float(z) == pytest.approx(20.0 * 0.044917764, abs=1e-6)


def test_bi2_inversion_image_first_is_written_below_the_layer(writer, tmp_path):
    """The image atom alone is a valid asymmetric unit - but its z is
    -0.898 A, not 19.10 A: the layer's inversion centre is at z = 0."""
    data, rows = write(writer, tmp_path, BI2_CIF, "bi2img", layer_group=72)
    image = dict(
        data,
        symbols=["Bi"],
        atomic_numbers=[83],
        positions=[[2 / 3, 1 / 3, 1 - 0.044917764]],
    )
    out = tmp_path / "image.d12"
    assert writer.create_d12_file(
        image, str(out), slab_options(layer_group=72), interactive=False
    )
    (row,) = atom_records(out.read_text())
    assert float(row[3]) == pytest.approx(-20.0 * 0.044917764, abs=1e-6)


def test_bi2_after_batch_reduction_is_unchanged(writer, tmp_path):
    """The asymmetric unit the batch path normally hands over already has one
    Bi on the +z side; its record does not change."""
    cif = tmp_path / "bi2.cif"
    cif.write_text(BI2_CIF)
    data = writer.parse_cif(str(cif), interactive=False)
    data = writer.verify_and_reduce_to_asymmetric_unit(data, interactive=False)
    out = tmp_path / "bi2red.d12"
    assert writer.create_d12_file(
        data, str(out), slab_options(layer_group=72), interactive=False
    )
    (row,) = atom_records(out.read_text())
    assert row[:4] == ["283", "0.3333333333", "0.6666666667", "0.898355"]


def test_polar_slab_keeps_both_inequivalent_atoms(writer, tmp_path):
    data, rows = write(writer, tmp_path, BN_CIF, "bn", layer_group=69)
    assert len(rows) == 2
    assert [r[-1] for r in rows] == ["B", "N"]
    assert float(rows[1][3]) == pytest.approx(0.2, abs=1e-6)


def test_lower_named_group_keeps_every_atom(writer, tmp_path):
    """Naming P1 (layer group 1) for the Bi2 cell: nothing is equivalent
    under it, so both atoms stay."""
    _, rows = write(writer, tmp_path, BI2_CIF, "bi2p1", layer_group=1)
    assert len(rows) == 2


def test_named_group_of_a_different_order_is_not_reduced(writer, tmp_path, capsys):
    """Layer group 66 (P-3, order 6) named for a P -3 m 1 (order 12) cell:
    the CIF's operators are not that group's, so none of them is used to drop
    an atom, and the converter says so."""
    _, rows = write(writer, tmp_path, BI2_CIF, "bi2p3", layer_group=66)
    out = capsys.readouterr()
    assert "layer group 66" in (out.out + out.err)
    assert len(rows) == 2


# The layer's z records move by ONE offset for the whole layer. Rounding each
# fractional z to its own nearest integer cut a layer centred on z = 1/2 in
# two - atoms at 0.45 and 0.55 went to -0.05 c and +0.45 c... and to +0.45 c
# and -0.45 c when the plane was taken at 0 - putting half the layer on the far
# side of the vacuum.


def min_image_dz(z1, z2, c):
    d = (z2 - z1) % 1.0
    return (d - round(d)) * c


@pytest.mark.parametrize(
    "zs, expected",
    [
        ([0.45, 0.55], [-0.05, 0.05]),        # centred on 1/2
        ([0.55, 0.45], [0.05, -0.05]),
        ([0.97, 0.03], [-0.03, 0.03]),        # straddling 0/1
        ([0.03, 0.97, 0.0], [0.03, -0.03, 0.0]),
        ([0.044917764], [0.044917764]),       # Bi2 asymmetric unit
        ([1 - 0.044917764], [-0.044917764]),  # its inversion image
        ([0.40], [-0.10]),                    # half a layer centred on 1/2
        ([0.10, 0.12], [0.10, 0.12]),         # half a layer centred on 0
    ],
)
def test_z_reversing_layer_is_measured_from_its_symmetry_plane(writer, zs, expected):
    got = writer.slab_cartesian_z(zs, 20.0, z_reversing=True)
    assert got == pytest.approx([20.0 * e for e in expected], abs=1e-9)
    for i in range(len(zs)):
        for j in range(len(zs)):
            assert got[j] - got[i] == pytest.approx(
                min_image_dz(zs[i], zs[j], 20.0), abs=1e-9
            )


@pytest.mark.parametrize(
    "zs, expected",
    [
        ([0.0, 0.01], [0.0, 0.01]),           # in the cell: c * z, as before
        ([0.45, 0.55], [0.45, 0.55]),
        ([0.97, 0.03], [-0.03, 0.03]),        # straddling: kept in one piece
    ],
)
def test_polar_layer_keeps_its_height_and_stays_in_one_piece(writer, zs, expected):
    got = writer.slab_cartesian_z(zs, 20.0, z_reversing=False)
    assert got == pytest.approx([20.0 * e for e in expected], abs=1e-9)


def two_element_layer(data, z_bi, z_sb):
    return dict(
        data,
        symbols=["Bi", "Sb"],
        atomic_numbers=[83, 51],
        positions=[[1 / 3, 2 / 3, z_bi], [0.0, 0.0, z_sb]],
    )


@pytest.mark.parametrize(
    "z_bi, z_sb, want_bi, want_sb",
    [
        (0.45, 0.50, -0.05, 0.0),    # layer centred on z = 1/2
        (0.55, 0.47, 0.05, -0.03),
        (0.97, 0.02, -0.03, 0.02),   # layer straddling z = 0/1
    ],
)
def test_slab_deck_keeps_the_layer_in_one_piece(
    writer, tmp_path, z_bi, z_sb, want_bi, want_sb
):
    data, _ = write(writer, tmp_path, BI2_CIF, "bi2base", layer_group=72)
    out = tmp_path / "layer.d12"
    assert writer.create_d12_file(
        two_element_layer(data, z_bi, z_sb), str(out),
        slab_options(layer_group=72), interactive=False,
    )
    rows = atom_records(out.read_text())
    assert [r[-1] for r in rows] == ["Bi", "Sb"]
    z = [float(r[3]) for r in rows]
    assert z == pytest.approx([20.0 * want_bi, 20.0 * want_sb], abs=1e-6)
    assert z[1] - z[0] == pytest.approx(min_image_dz(z_bi, z_sb, 20.0), abs=1e-6)


def test_named_group_of_the_same_order_but_other_operations_is_not_reduced(
    writer, tmp_path, capsys
):
    """Layer group 75 (p6/m) has twelve operations, as P -3 m 1 does, but
    not the same ones: the CIF's operators are not that group's, so none of
    them is used to drop an atom."""
    _, rows = write(writer, tmp_path, BI2_CIF, "bi2p6m", layer_group=75)
    out = capsys.readouterr()
    assert "layer group 75" in (out.out + out.err)
    assert len(rows) == 2
