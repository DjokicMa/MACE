"""Space-group tables in d12_constants, checked against their sources.

- MULTI_ORIGIN_SPACEGROUPS must be exactly the space groups the International
  Tables give two origin choices.
- SPACEGROUP_ALTERNATIVES spells group 90 the way CRYSTAL does, "P 4 21 2",
  maps P-42c to 112 and has no duplicate keys.
"""
import ast
import sys

import pytest

from conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "Crystal_d12"))

import d12_constants as C  # noqa: E402

# International Tables for Crystallography, Vol. A: the space groups listed
# with two origin choices.
ITA_TWO_ORIGINS = {48, 50, 59, 68, 70, 85, 86, 88, 125, 126, 129, 130, 133, 134,
                   137, 138, 141, 142, 201, 203, 222, 224, 227, 228}


def test_multi_origin_groups_are_the_ita_two_origin_groups():
    assert set(C.MULTI_ORIGIN_SPACEGROUPS) == ITA_TWO_ORIGINS


def test_pbcn_has_one_origin():
    assert 60 not in C.MULTI_ORIGIN_SPACEGROUPS


def test_group_90_is_spelled_as_crystal_spells_it():
    assert C.SPACEGROUP_ALTERNATIVES["P 4 21 2"] == 90
    assert "P 42 1 2" not in C.SPACEGROUP_ALTERNATIVES
    assert C.spacegroup_number_from_symbol("P 4 21 2") == 90
    assert C.spacegroup_number_from_symbol("P 42 1 2") is None


def test_p42c_is_112():
    assert C.SPACEGROUP_ALTERNATIVES["P-42c"] == 112
    assert C.SPACEGROUP_ALTERNATIVES["P-421c"] == 114


def test_alternatives_have_no_duplicate_keys():
    """A repeated key in a dict literal silently keeps only its last value."""
    tree = ast.parse((REPO_ROOT / "Crystal_d12" / "d12_constants.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == "SPACEGROUP_ALTERNATIVES" for t in node.targets):
            keys = [k.value for k in node.value.keys]
            assert len(keys) == len(set(keys)), sorted(k for k in keys if keys.count(k) > 1)
            assert C.SPACEGROUP_ALTERNATIVES["P1211"] == 4
            assert C.SPACEGROUP_ALTERNATIVES["P121"] == 3
            return
    pytest.fail("SPACEGROUP_ALTERNATIVES assignment not found")


@pytest.mark.parametrize("symbol,number", [("P 4 21 2", 90), ("P -4 2 C", 112),
                                           ("P -4 21 C", 114)])
def test_crystal_output_symbols_parse_to_their_group(tmp_path, symbol, number):
    from d12_parsers import CrystalOutputParser
    out = tmp_path / "x.out"
    out.write_text("")
    parser = CrystalOutputParser(str(out))
    parser._extract_spacegroup([f" SPACE GROUP (CENTROSYMMETRIC) : {symbol}"])
    assert parser.data["spacegroup"] == number


PBCN_CIF = """data_x
_symmetry_space_group_name_H-M 'P b c n'
_symmetry_Int_Tables_number 60
_cell_length_a 5.0
_cell_length_b 6.0
_cell_length_c 7.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Si1 Si 0 0.1 0.25
"""


@pytest.mark.parametrize("origin", ["ALTERNATE", "STANDARD", "AUTO"])
def test_pbcn_deck_writes_the_standard_setting_record(tmp_path, origin):
    """Pbcn was listed with an "alternate origin" whose code, 0 1 0, is the
    rhombohedral-axes flag; asking for it wrote that into the deck."""
    pytest.importorskip("ase", reason="CIF parsing needs ase (absent in CI)")
    import NewCifToD12
    (tmp_path / "x.cif").write_text(PBCN_CIF)
    cif = NewCifToD12.parse_cif(str(tmp_path / "x.cif"))
    out = tmp_path / "x.d12"
    ok = NewCifToD12.create_d12_file(cif, str(out), dict(
        dimensionality="CRYSTAL", calculation_type="SP", basis_set_type="INTERNAL",
        basis_set="POB-TZVP-REV2", method="DFT", dft_functional="PBE",
        is_spin_polarized=False, tolerances=C.DEFAULT_TOLERANCES, scf_method="DIIS",
        symmetry_handling="CIF", origin_setting=origin))
    assert ok is True
    lines = out.read_text().splitlines()
    assert lines[1:4] == ["CRYSTAL", "0 0 0", "60"]
