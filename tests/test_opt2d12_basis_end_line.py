"""Every opt2d12 deck with an EXTERNAL basis ends the basis input with one "99 0".

CRYSTAL reads the basis input up to "99 0". Commit 7b97cd44 moved the library
records into _write_library_records and lost the "99 0" after them on the
RHF/UHF paths and the DFT legacy-path one, so those decks stopped at input.
A library record that already ends with "99 0" (a hand-made file) must not
get a second one.
"""
import pytest

from conftest import find_data

TZ_DZ = "full.basis.doublezeta"


def _basis_block(text):
    """The lines between the geometry's END and the next END."""
    lines = text.splitlines()
    first = lines.index("END")
    rest = lines[first + 1:]
    return rest[:rest.index("END")]


def _geo():
    from d12_parsers import CrystalOutputParser

    return CrystalOutputParser(str(find_data("OPT/1_dia_opt_rev1.out"))).parse()


def _write(tmp_path, functional, method, key, libdir, name):
    import CRYSTALOptToD12 as M

    geo = _geo()
    settings = dict(geo)
    settings.update(basis_set_type="EXTERNAL", functional=functional, method=method,
                    basis_set=str(libdir) + "/", calculation_type="SP")
    settings.pop("basis_set_path", None)
    settings.pop("use_original_external_basis", None)
    if key == "basis_set_path":
        settings["basis_set_path"] = str(libdir) + "/"
    out = tmp_path / name
    assert M.write_d12_file(str(out), geo, settings, ask=False) is not False, M.LAST_RESULT
    return out.read_text()


PATHS = [(f, m, k) for f, m in (("RHF", "HF"), ("UHF", "HF"), ("PBE", "DFT"))
         for k in ("basis_set_path", "basis_set")]


@pytest.mark.parametrize("functional,method,key", PATHS,
                         ids=[f"{f}-{k}" for f, _, k in PATHS])
def test_library_basis_ends_with_one_99_0(tmp_path, functional, method, key):
    from conftest import REPO_ROOT

    lib = REPO_ROOT / "Crystal_d12" / "basis_sets" / TZ_DZ
    text = _write(tmp_path, functional, method, key, lib, "x.d12")
    block = _basis_block(text)
    assert block[-1].split() == ["99", "0"], block[-3:]
    assert sum(1 for line in block if line.split() == ["99", "0"]) == 1


def test_a_record_that_already_ends_with_99_0_gets_no_second(monkeypatch):
    """basis_file_check refuses such a record today; the writer still adds no second one."""
    import io

    import CRYSTALOptToD12 as M

    monkeypatch.setattr(M.basis_library, "record_exists", lambda d, k: True)
    monkeypatch.setattr(M.basis_library, "read_record",
                        lambda d, k: f"{k} 1\n0 0 1 2.0 1.0\n 1.0 1.0\n" + ("99 0\n" if k == 8 else ""))
    for atoms in ({6, 8}, {6}):
        f = io.StringIO()
        M._write_library_records(f, "lib", atoms)
        lines = f.getvalue().splitlines()
        assert lines[-1] == "99 0"
        assert lines.count("99 0") == 1, (atoms, lines)


def test_parent_external_basis_ends_with_one_99_0(tmp_path):
    """use_original_external_basis: the parent's records, then one "99 0"."""
    import CRYSTALOptToD12 as M

    geo = _geo()
    records = ["6 1", "0 0 1 2.0 1.0", " 1.0 1.0"]
    for functional, method in (("RHF", "HF"), ("PBE", "DFT")):
        settings = dict(geo)
        settings.update(basis_set_type="EXTERNAL", functional=functional, method=method,
                        use_original_external_basis=True, calculation_type="SP")
        out = tmp_path / f"p_{functional}.d12"
        assert M.write_d12_file(str(out), geo, settings, records, ask=False) is not False
        block = _basis_block(out.read_text())
        assert block[-1].split() == ["99", "0"]
        assert sum(1 for line in block if line.split() == ["99", "0"]) == 1
