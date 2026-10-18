"""From the default settings to a SOC deck in one step (E-M6).

Measured with `mace convert --batch` on AgBr, starting from the saved
default options plus "soc": true: three runs, three refusals, one at a time
and none naming its remedy - the internal basis (BASISSET), then the basis
key (which wanted this checkout's absolute path, so the options file broke
on another machine), then HSE06. Now one refusal lists every change with
the keys to set, "basis_set" takes a bundled library by name, and saved
settings store a bundled library by name.
"""
import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("ase", reason="CIF parsing needs ase")
import basis_library  # noqa: E402
import NewCifToD12  # noqa: E402
import soc_ecp  # noqa: E402
from d12_constants import DEFAULT_TOLERANCES  # noqa: E402

AGBR = """data_AgBr
_symmetry_space_group_name_H-M 'F m -3 m'
_symmetry_Int_Tables_number 225
_cell_length_a 5.774
_cell_length_b 5.774
_cell_length_c 5.774
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Ag1 Ag 0 0 0
Br1 Br 0.5 0.5 0.5
"""

TZ = Path(basis_library.BASIS_SETS_DIR) / "full.basis.triplezeta"


def _convert(tmp_path, **opts):
    src = tmp_path / "cifs"
    src.mkdir(exist_ok=True)
    (src / "AgBr.cif").write_text(AGBR)
    base = dict(dimensionality="CRYSTAL", calculation_type="SP", method="DFT",
                dft_functional="HSE06", use_dispersion=True, is_spin_polarized=True,
                tolerances=DEFAULT_TOLERANCES, scf_method="DIIS", symmetry_handling="CIF",
                basis_set_type="INTERNAL", basis_set="POB-TZVP-REV2", soc=True)
    base.update(opts)
    out = tmp_path / "out"
    written, _ = NewCifToD12.process_cifs(str(src), base, str(out), interactive=False)
    return written, sorted(out.glob("*.d12")) if out.exists() else []


def test_default_settings_get_one_refusal_with_every_remedy(tmp_path, capsys):
    written, decks = _convert(tmp_path)
    assert written == 0 and not decks
    err = capsys.readouterr().err
    assert "a SOC deck needs 2 changes" in err
    assert '"basis_set_type": "EXTERNAL"' in err and '"basis_set": "full.basis.triplezeta"' in err
    assert "HSE06-D3" in err and "PBE0" in err and '"dft_functional"' in err


def test_following_the_remedy_writes_the_deck(tmp_path):
    written, decks = _convert(tmp_path, basis_set_type="EXTERNAL",
                              basis_set="full.basis.triplezeta", dft_functional="PBE0")
    assert written == 1
    text = decks[0].read_text()
    assert "TWOCOMPON\nSOC\nEND" in text and "INPSOC" in text and "PBE0-D3" in text


def test_a_bare_library_name_is_the_bundled_library():
    assert basis_library.bundled_family_for("full.basis.triplezeta") == "full.basis.triplezeta"
    assert basis_library.bundled_family_for("stuttgart/") == "stuttgart"
    assert basis_library.read_record("full.basis.triplezeta", 6) == \
        basis_library.read_record(str(TZ), 6)


def test_an_existing_directory_of_that_name_wins(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "stuttgart").mkdir()
    assert basis_library.bundled_family_for("stuttgart") is None


def test_saved_settings_store_a_bundled_library_by_name():
    saved = basis_library.portable_options(
        {"basis_set": str(TZ) + "/", "basis_set_path": str(TZ), "functional": "PBE0"})
    assert saved == {"basis_set": "full.basis.triplezeta",
                     "basis_set_path": "full.basis.triplezeta", "functional": "PBE0"}
    assert basis_library.portable_path("/somewhere/else/mybasis") == "/somewhere/else/mybasis"


def test_opt2d12_template_saves_the_library_by_name():
    import CRYSTALOptToD12 as M
    assert M.options_for_template({"basis_set": str(TZ) + "/"})["basis_set"] == \
        "full.basis.triplezeta"


def test_single_refusal_keeps_its_own_words():
    """One problem reads as before (no list), with its remedy appended."""
    deck = "t\nCRYSTAL\n0 0 0\n225\n5.0\n1\n6 0 0 0\nEND\nBASISSET\nPOB-TZVP-REV2\nDFT\nPBE0\nENDDFT\nSHRINK\n4 4\nSCFDIR\nEND\n"
    with pytest.raises(soc_ecp.SocError) as exc:
        soc_ecp.soc_deck(deck, log=lambda m: None)
    msg = str(exc.value)
    assert msg.startswith("the deck uses an internal basis-set library (BASISSET)")
    assert "changes" not in msg
