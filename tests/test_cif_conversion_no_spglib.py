"""CIF -> D12 conversion must stay structurally correct without spglib.

Real-world failure (phase-3 QA, 2026-07-09, HPCC job 12085937): quick-start
converted 3_dia3.cif on a python without spglib. ASE expands the CIF's 2
asymmetric-unit atoms into the full 40-atom cell, and the no-spglib path
wrote ALL of them while still declaring space group 227 — CRYSTAL re-applied
the operators to the expanded set and died with
"DISTANCE BETWEEN ATOMS 13 54 TOO SMALL: 0.6850 ANGSTROM / ERROR NEIGHB".
The other 7 materials survived only because all their atoms sit on special
positions whose orbits fold exactly.

Fallback contract: per the CIF spec the raw _atom_site_ records ARE the
asymmetric unit, so without spglib the converter must write those (they are
byte-for-byte what the user's known-good manual decks contain).
"""
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "Crystal_d12"))

import NewCifToD12  # noqa: E402

DATA = Path(__file__).parent / "data"
DIA3_CIF = DATA / "3_dia3_opt_BULK_OPTGEOM_symm.cif"
DIA_CIF = DATA / "1_dia_opt_BULK_OPTGEOM_symm.cif"


def test_parse_cif_keeps_raw_asymmetric_records():
    d = NewCifToD12.parse_cif(str(DIA3_CIF))
    assert len(d["symbols"]) == 40, "ASE-expanded cell expected"
    assert d["cif_atom_symbols"] == ["C", "C"]
    assert len(d["cif_atom_positions"]) == 2
    assert d["spacegroup"] == 227


def test_no_spglib_reduction_uses_cif_records(monkeypatch):
    monkeypatch.setattr(NewCifToD12, "SPGLIB_AVAILABLE", False)
    d = NewCifToD12.parse_cif(str(DIA3_CIF))

    reduced = NewCifToD12.verify_and_reduce_to_asymmetric_unit(d)

    assert len(reduced["symbols"]) == 2, (
        "no-spglib fallback must write the CIF's asymmetric unit, not the "
        "expanded cell (CRYSTAL re-applies the space group -> ERROR NEIGHB)")
    assert reduced["spacegroup"] == 227
    # normalized coordinates must match the user's known-good manual deck
    got = {tuple(round(x, 5) for x in p) for p in reduced["positions"]}
    assert got == {(0.75, 0.75, 0.75), (0.83912, 0.83912, 0.66088)}, got
    assert reduced["atomic_numbers"] == [6, 6]


def test_no_spglib_reduction_1_dia(monkeypatch):
    monkeypatch.setattr(NewCifToD12, "SPGLIB_AVAILABLE", False)
    d = NewCifToD12.parse_cif(str(DIA_CIF))

    reduced = NewCifToD12.verify_and_reduce_to_asymmetric_unit(d)

    assert len(reduced["symbols"]) == 1
    assert reduced["positions"][0] == pytest.approx([0.0, 0.0, 0.0])


def test_spglib_reduction_still_works():
    """Regression guard for the normal spglib path (available locally)."""
    if not NewCifToD12.SPGLIB_AVAILABLE:
        pytest.skip("spglib not installed in this environment")
    d = NewCifToD12.parse_cif(str(DIA3_CIF))

    reduced = NewCifToD12.verify_and_reduce_to_asymmetric_unit(d)

    assert len(reduced["symbols"]) == 2
    assert reduced["spacegroup"] == 227
