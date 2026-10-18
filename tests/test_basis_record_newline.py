"""Every basis record a deck writer reads ends with a newline.

92 records of the bundled stuttgart family (H-Zn, Ga-As, Tc and most of its
ECP records) are stored without a final newline, so the next record - or the
'99 0' that ends the basis input - was glued onto the last primitive line:
"0.4743762600 1.0000000000000099 0" in a GaAs stuttgart deck, which has no
'99 0' record at all. basis_library.read_record, which every deck writer
reads through, now ends each record with a newline.
"""
import io

import pytest

import basis_library as BL
from conftest import REPO_ROOT

BASIS_SETS = REPO_ROOT / "Crystal_d12" / "basis_sets"


@pytest.mark.parametrize("family", BL.FAMILIES)
def test_every_record_read_for_a_deck_ends_with_a_newline(family):
    lib = BL.load_family(family)
    for key in lib.keys():
        try:
            text = BL.read_record(str(BASIS_SETS / family), key)
        except ValueError:          # records CRYSTAL could not read are refused anyway
            continue
        assert text.endswith("\n"), (family, key)
        assert not text.endswith("\n\n") or lib.record(key).endswith("\n\n"), (family, key)


def test_a_stuttgart_deck_has_its_99_0_record():
    from d12_writer import write_basis_block
    f = io.StringIO()
    write_basis_block(f, {"basis_set_type": "EXTERNAL", "basis_set": str(BASIS_SETS / "stuttgart")},
                      {"elements": [31, 33]})
    lines = f.getvalue().split("\n")
    assert "99 0" in lines
    assert all(not ln.rstrip().endswith("99 0") or ln.strip() == "99 0" for ln in lines)


def test_records_that_had_their_newline_are_unchanged():
    for family in BL.FAMILIES:
        lib = BL.load_family(family)
        for key in lib.keys():
            raw = lib.record(key)
            if raw.endswith("\n"):
                try:
                    assert BL.read_record(str(BASIS_SETS / family), key) == raw
                except ValueError:
                    pass


# The HPCC SOC runs found stuttgart records 1, 6, 8, 33 and 249 without a
# final newline in the 1.1.5 libraries. Every family and every record is
# covered: the bundled families through read_record (above and here), the
# data-only SOC family, and per-element files on disk (the archived stuttgart
# directory), which read_record serves from disk.

HPCC_RECORDS = ("1", "6", "8", "33", "249")


@pytest.mark.parametrize("key", HPCC_RECORDS)
def test_the_records_hpcc_saw_are_stored_without_and_read_with_a_newline(key):
    raw = BL.load_family("stuttgart").record(key)
    assert not raw.endswith("\n")
    text = BL.read_record(str(BASIS_SETS / "stuttgart"), key)
    assert text == raw + "\n"


def test_a_deck_with_the_hpcc_records_keeps_every_header_and_99_0_on_its_own_line():
    from d12_writer import write_basis_block
    f = io.StringIO()
    write_basis_block(f, {"basis_set_type": "EXTERNAL", "basis_set": str(BASIS_SETS / "stuttgart")},
                      {"elements": [1, 6, 8, 33, 249]})
    lines = f.getvalue().split("\n")
    for key in HPCC_RECORDS:
        header = BL.load_family("stuttgart").record(key).split("\n", 1)[0].strip()
        assert header in [ln.strip() for ln in lines], key
    assert lines.count("99 0") == 1


def test_every_family_file_is_covered():
    """Every .lib in basis_sets/ is a bundled or data-only family."""
    libs = {p.stem for p in BASIS_SETS.glob("*.lib")}
    assert libs == set(BL.FAMILIES) | set(BL.DATA_ONLY_FAMILIES)


def test_every_official_soc_record_ends_with_a_newline():
    for z in BL.soc_basis_elements():
        assert BL.soc_basis_record(z).endswith("\n"), z


def test_records_read_from_disk_end_with_a_newline():
    directory = REPO_ROOT / "Crystal_d3" / "Archived" / "basis" / "stuttgart"
    read = 0
    for path in sorted(directory.iterdir()):
        if not path.is_file() or not path.name.isdigit():
            continue
        try:
            text = BL.read_record(str(directory), path.name)
        except ValueError:
            continue
        read += 1
        assert text.endswith("\n"), path.name
    assert read > 100
