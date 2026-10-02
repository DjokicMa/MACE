"""A basis-library file CRYSTAL cannot read is refused on every deck path.

stuttgart/204, 219, 220, 227 and 294 (and 294 in both full.basis libraries)
break the basis input (manual pp. 25-27): a placeholder where NSHL belongs,
"X" for a shell charge, fewer shells than announced. Round 5 refused them only
when a SOC deck read stuttgart/ (soc_ecp.read_stuttgart); every reader that
copies a library file into a deck now refuses them too, naming the file, and
leaves no deck.

204/219/220/227 are Z < 37, so cif2d12 reads the bare-Z file for those
elements; only an opt2d12 parent whose geometry carries those atom numbers
makes MACE read them. 294 (Pu) is reached from a CIF as well.
"""
import copy
import io
import json
import os
import sys

import pytest

from conftest import REPO_ROOT

import CRYSTALOptToD12 as M
import d12_constants
from basis_file_check import BasisFileError
from d12_writer import write_basis_block

LIB = REPO_ROOT / "Crystal_d12" / "basis_sets"
STUTTGART = str(LIB / "stuttgart") + "/"


def _out(atoms):
    return {
        "dimensionality": "CRYSTAL", "spacegroup": 225, "origin_setting": "0 0 0",
        "conventional_cell": [5.6, 5.6, 5.6, 90.0, 90.0, 90.0],
        "coordinates": [{"atom_number": str(z), "x": f"{i / 2:.6f}", "y": f"{i / 2:.6f}",
                         "z": f"{i / 2:.6f}", "is_unique": True} for i, z in enumerate(atoms)],
        "functional": "PBE0", "calculation_type": "SP",
    }


DECK = {"functional": "PBE0", "dispersion": False, "dft_grid": "XLGRID",
        "calculation_type": "SP", "k_points": "8 16",
        "tolerances": {"TOLINTEG": "7 7 7 7 14", "TOLDEE": 7}}


class _NoTTY:
    def isatty(self):
        return False


@pytest.fixture
def opt2d12(tmp_path, monkeypatch):
    """opt2d12 on a parent whose geometry carries ``atoms``, with the
    stuttgart library as the template's basis directory."""
    monkeypatch.setattr(sys, "stdin", _NoTTY())
    monkeypatch.chdir(tmp_path)

    def run(atoms, basis_dir=STUTTGART):
        class Parser:
            def __init__(self, path, kind):
                self.kind = kind

            def parse(self):
                return copy.deepcopy(_out(atoms) if self.kind == "out" else DECK)

        monkeypatch.setattr(M, "CrystalOutputParser", lambda p: Parser(p, "out"))
        monkeypatch.setattr(M, "CrystalInputParser", lambda p: Parser(p, "d12"))
        (tmp_path / "x.out").write_text("")
        (tmp_path / "x.d12").write_text("")
        (tmp_path / "t.json").write_text(json.dumps({
            "calculation_type": "SP", "functional": "PBE0", "dispersion": False,
            "dft_grid": "XLGRID", "basis_set_type": "EXTERNAL", "basis_set": basis_dir,
            "basis_set_path": basis_dir}))
        monkeypatch.setattr(sys, "argv", ["CRYSTALOptToD12.py", "--out-file", "x.out",
                                          "--config-file", "t.json", "--output-dir", "sp",
                                          "--yes"])
        try:
            M.main()
            status = 0
        except SystemExit as e:
            status = e.code
        out = tmp_path / "sp"
        return status, sorted(os.listdir(out)) if out.exists() else []
    return run


@pytest.mark.parametrize("nat,why", [
    (204, "it announces 2 shells but holds 1"),
    (219, "no shell count in its first line"),
    (220, "no shell count in its first line"),
    (227, "shell charge 'X' is not a number"),
])
def test_opt2d12_refuses_an_unreadable_stuttgart_file(opt2d12, capsys, nat, why):
    status, files = opt2d12([nat, 8])
    err = capsys.readouterr().err
    assert status == 1
    assert files == []
    assert f"stuttgart/{nat} (Z={nat - 200}) cannot be used: " in err and why in err
    assert M.LAST_RESULT["reason"] and f"stuttgart/{nat}" in M.LAST_RESULT["reason"]


def test_opt2d12_still_writes_a_readable_stuttgart_deck(opt2d12):
    status, files = opt2d12([282, 8])
    assert status == 0 and files == ["x_sp_PBE0_optimized.d12"]


def test_read_basis_file_refuses_the_pu_file():
    with pytest.raises(BasisFileError, match=r"stuttgart/294 \(Z=94\) cannot be used: "
                                             r"it announces 11 shells but holds 9"):
        d12_constants.read_basis_file(STUTTGART, 94)
    # the default library's Pu file has the same fault
    with pytest.raises(BasisFileError, match="full.basis.triplezeta/294"):
        d12_constants.read_basis_file(str(LIB / "full.basis.triplezeta"), 94)


def test_read_basis_file_checks_the_bare_z_fallback(tmp_path):
    (tmp_path / "82").write_text("82 3\n0 0 1 2.0 1.0\n1.0 1.0\n")
    with pytest.raises(BasisFileError, match="it announces 3 shells but holds 1"):
        d12_constants.read_basis_file(str(tmp_path), 82)


def test_read_basis_file_of_a_readable_file_is_unchanged():
    text = d12_constants.read_basis_file(STUTTGART, 82)
    assert text == (LIB / "stuttgart" / "282").read_text()
    assert d12_constants.read_basis_file(STUTTGART, 86) == ""     # no file: as before


def test_the_coverage_check_refuses_an_unreadable_file_before_any_question():
    with pytest.raises(BasisFileError, match="stuttgart/294"):
        d12_constants.check_basis_set_compatibility(STUTTGART, [94, 8], "EXTERNAL")
    # a readable or missing file is reported as before
    assert d12_constants.check_basis_set_compatibility(STUTTGART, [82, 8], "EXTERNAL") == (True, [])


def test_d12_writer_basis_block_refuses_an_unreadable_file():
    buf = io.StringIO()
    with pytest.raises(BasisFileError, match="stuttgart/294"):
        write_basis_block(buf, {"basis_set_type": "EXTERNAL", "basis_set": STUTTGART},
                          {"elements": [94]})


PU_CIF = """data_Pu
_symmetry_space_group_name_H-M 'F m -3 m'
_symmetry_Int_Tables_number 225
_cell_length_a 4.64
_cell_length_b 4.64
_cell_length_c 4.64
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Pu1 Pu 0 0 0
"""


@pytest.mark.parametrize("bypass_precheck", [False, True], ids=["pre-check", "read site"])
def test_cif2d12_refuses_pu_with_the_stuttgart_library(tmp_path, capsys, monkeypatch,
                                                        bypass_precheck):
    pytest.importorskip("ase", reason="CIF parsing needs ase (absent in CI)")
    import NewCifToD12
    from d12_constants import DEFAULT_TOLERANCES
    if bypass_precheck:
        monkeypatch.setattr(NewCifToD12, "check_basis_set_compatibility",
                            lambda *a, **k: (True, []))
    (tmp_path / "Pu.cif").write_text(PU_CIF)
    cif = NewCifToD12.parse_cif(str(tmp_path / "Pu.cif"))
    out = tmp_path / "Pu.d12"
    ok = NewCifToD12.create_d12_file(cif, str(out), dict(
        dimensionality="CRYSTAL", calculation_type="SP", method="DFT",
        dft_functional="PBE0", is_spin_polarized=False, tolerances=DEFAULT_TOLERANCES,
        scf_method="DIIS", symmetry_handling="CIF", basis_set_type="EXTERNAL",
        basis_set=STUTTGART), interactive=False)
    err = capsys.readouterr().err
    assert ok is False and not out.exists()
    assert not [n for n in os.listdir(tmp_path) if n.startswith(".")]
    assert "Not writing Pu.d12: " in err and "stuttgart/294 (Z=94) cannot be used" in err
