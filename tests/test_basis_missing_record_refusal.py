"""A deck is never written without a basis for one of its elements.

Two routes wrote one (both before 1.1.6):

* opt2d12 moving a parent with an internal basis (STO-3G has Kr) to an
  EXTERNAL library family that lacks an element (full.basis.triplezeta has no
  Kr) skipped the coverage check for every EXTERNAL deck, and its writers only
  warned on a missing record and wrote ``99 0 / END`` anyway: rc 0,
  "Successfully created", a deck CRYSTAL rejects.
* cif2d12 with an EXTERNAL basis directory that does not exist (an options
  file saved on another machine) passed the coverage check, and
  read_basis_file returned "" for every element: a deck with no basis at all.

Both are refused now, naming the element and the family or the directory, and
the "Continue anyway?" question that let a user knowingly write such a deck is
gone: CRYSTAL always rejects it.
"""
import json
import shutil
import subprocess
import sys

import pytest

from conftest import REPO_ROOT, find_data

MACE_CLI = REPO_ROOT / "mace_cli"
TZ = REPO_ROOT / "Crystal_d12" / "basis_sets" / "full.basis.triplezeta"

KRF = """KrF
CRYSTAL
0 0 0
225
5.80
2
36 0.0 0.0 0.0
9 0.5 0.5 0.5
BASISSET
STO-3G
DFT
PBE
END
SHRINK
8 8
TOLDEE
7
END
"""


def _mace(cwd, *args):
    r = subprocess.run([sys.executable, str(MACE_CLI), "--no-banner", *args], cwd=cwd,
                       stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=300)
    return r.returncode, r.stdout + r.stderr


@pytest.mark.parametrize("key", ["basis_set_path", "basis_set"],
                         ids=["basis_set_path", "legacy basis_set path"])
def test_opt2d12_refuses_an_external_family_without_the_element(tmp_path, key):
    """The reviewer's recipe: KrF (STO-3G) to full.basis.triplezeta, which has no Kr."""
    (tmp_path / "KrF.d12").write_text(KRF)
    config = {"calculation_type": "SP", "basis_set_type": "EXTERNAL", "basis_set": str(TZ) + "/"}
    if key == "basis_set_path":
        config["basis_set_path"] = str(TZ) + "/"
    (tmp_path / "c.json").write_text(json.dumps(config))
    code, log = _mace(tmp_path, "opt2d12", "--d12-file", "KrF.d12", "--config-file", "c.json",
                      "--yes")
    assert not list(tmp_path.glob("KrF_sp*.d12")), log[-3000:]
    assert "Successfully created" not in log
    assert "Kr" in log and "full.basis.triplezeta" in log, log[-3000:]
    assert code != 0


def test_opt2d12_writer_refuses_a_missing_record_even_past_the_check(tmp_path, monkeypatch):
    """The writer itself refuses a record it cannot find (no '99 0 / END' without it)."""
    import CRYSTALOptToD12 as M
    from d12_parsers import CrystalOutputParser

    geo = CrystalOutputParser(str(find_data("OPT/1_dia_opt_rev1.out"))).parse()
    monkeypatch.setattr(M, "check_basis_set_compatibility", lambda *a, **k: (True, []))
    empty = tmp_path / "lib"
    empty.mkdir()
    (empty / "8").write_text("8 1\n0 0 1 2.0 1.0\n 1.0 1.0\n")
    for functional, method in (("PBE", "DFT"), ("RHF", "HF")):
        for key in ("basis_set_path", "basis_set"):
            settings = dict(geo)
            settings.update(basis_set_type="EXTERNAL", functional=functional, method=method,
                            basis_set=str(empty) + "/")
            settings.pop("basis_set_path", None)
            if key == "basis_set_path":
                settings["basis_set_path"] = str(empty) + "/"
            out = tmp_path / f"x_{functional}_{key}.d12"
            M.LAST_RESULT["reason"] = None
            assert M.write_d12_file(str(out), geo, settings, ask=False) is False, (functional, key)
            assert not out.exists()
            assert "C (Z=6)" in M.LAST_RESULT["reason"], M.LAST_RESULT["reason"]


def test_opt2d12_never_asks_to_continue_without_a_basis(tmp_path, monkeypatch):
    """At a terminal too: no 'continue anyway?', the deck is refused."""
    import CRYSTALOptToD12 as M
    from d12_parsers import CrystalOutputParser

    geo = CrystalOutputParser(str(find_data("OPT/1_dia_opt_rev1.out"))).parse()
    settings = dict(geo)
    settings["basis_set_type"] = "INTERNAL"
    monkeypatch.setattr(M, "check_basis_set_compatibility", lambda *a, **k: (False, [6]))
    monkeypatch.setattr(M, "yes_no_prompt", lambda *a, **k: pytest.fail("asked to continue"))
    out = tmp_path / "x.d12"
    assert M.write_d12_file(str(out), geo, settings, ask=True) is False
    assert not out.exists()
    assert "C" in M.LAST_RESULT["reason"]


def test_read_basis_file_raises_for_a_missing_record(tmp_path):
    from basis_file_check import BasisFileError
    from d12_constants import read_basis_file

    (tmp_path / "6").write_text("6 1\n0 0 1 2.0 1.0\n 1.0 1.0\n")
    assert read_basis_file(str(tmp_path), 6).startswith("6 1")
    with pytest.raises(BasisFileError, match=r"O \(Z=8\)"):
        read_basis_file(str(tmp_path), 8)
    with pytest.raises(BasisFileError, match="Kr"):
        read_basis_file(str(TZ) + "/", 36)


def test_a_missing_external_directory_is_refused_by_the_coverage_check(tmp_path):
    from basis_file_check import BasisFileError
    from d12_constants import check_basis_set_compatibility

    missing = str(tmp_path / "nonexistent" / "full.basis.triplezeta") + "/"
    with pytest.raises(BasisFileError, match="does not exist"):
        check_basis_set_compatibility(missing, [6], "EXTERNAL")


def test_cif2d12_refuses_a_missing_external_directory(tmp_path):
    """The reviewer's recipe: an options file whose EXTERNAL directory is not on this machine."""
    pytest.importorskip("ase", reason="NewCifToD12 imports ase.io")
    cif = find_data("CIFs/1_dia_opt_rev1.cif")
    (tmp_path / "cifs").mkdir()
    shutil.copy2(cif, tmp_path / "cifs" / cif.name)
    opts = tmp_path / "o.json"
    opts.write_text(json.dumps({
        "calculation_type": "SP", "method": "DFT", "dft_functional": "PBE",
        "basis_set_type": "EXTERNAL", "basis_set": "/nonexistent/hpcc/full.basis.triplezeta/",
        "symmetry_handling": "CIF"}))
    code, log = _mace(tmp_path, "convert", "--batch", "--options_file", str(opts),
                      "--cif_dir", str(tmp_path / "cifs"), "--output_dir", str(tmp_path / "out"))
    assert not list((tmp_path / "out").glob("*.d12")) if (tmp_path / "out").exists() else True, log[-3000:]
    assert "Created" not in log.replace("Not created", "")
    assert "/nonexistent/hpcc/full.basis.triplezeta" in log and "does not exist" in log, log[-3000:]


def test_cif2d12_never_asks_to_continue_without_a_basis(tmp_path, monkeypatch):
    pytest.importorskip("ase", reason="NewCifToD12 imports ase.io")
    import NewCifToD12
    from d12_constants import DEFAULT_TOLERANCES

    cif = NewCifToD12.parse_cif(str(find_data("CIFs/1_dia_opt_rev1.cif")))
    monkeypatch.setattr(NewCifToD12, "check_basis_set_compatibility", lambda *a, **k: (False, [6]))
    monkeypatch.setattr(NewCifToD12, "yes_no_prompt", lambda *a, **k: pytest.fail("asked to continue"))
    out = tmp_path / "x.d12"
    ok = NewCifToD12.create_d12_file(cif, str(out), dict(
        dimensionality="CRYSTAL", calculation_type="SP", method="DFT",
        dft_functional="PBE", is_spin_polarized=False, tolerances=DEFAULT_TOLERANCES,
        scf_method="DIIS", symmetry_handling="CIF", basis_set_type="INTERNAL",
        basis_set="POB-TZVP-REV2"), interactive=True)
    assert ok is False and not out.exists()
