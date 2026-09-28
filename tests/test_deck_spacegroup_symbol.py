"""A deck that gives its space group by symbol (IFLAG = 1) is read.

CrystalInputParser looked the symbol up in the compact table ("Fm-3m"), which
never matches CRYSTAL's spaced input form ("F M 3 M"), so the cell and atoms of
every IFLAG = 1 deck were silently left out of the parse.

The accepted spellings are the ones CRYSTAL/23-intel-2023a accepts, measured on
HPCC with TESTGEOM over every spaced symbol in SPACEGROUP_ALTERNATIVES: 230
accepted (each echoed back with the point-group order and centrosymmetry of
its number here); "A E M 2", "A E A 2", "C M C E", "C M M E", "C C C E"
(invalid character) and "P 42 1 2" (group 90 is "P 4 21 2") refused. CRYSTAL
also refused "FM3M" and "P -4 2 1 M", read "F M -3 M" as a different,
non-centrosymmetric group, and accepted lower case and extra blanks.
"""
import textwrap

import pytest

from d12_constants import (
    CRYSTAL_INPUT_SPACEGROUP_SYMBOLS,
    SPACEGROUP_SYMBOLS,
    spacegroup_number_from_symbol,
)
from d12_parsers import CrystalInputParser


def test_one_accepted_spelling_per_space_group():
    assert sorted(CRYSTAL_INPUT_SPACEGROUP_SYMBOLS.values()) == list(range(1, 231))


@pytest.mark.parametrize("symbol,number", [
    ("F M 3 M", 225), ("f m 3 m", 225), ("F  M  3  M", 225), ("  P 21/C  ", 14),
    ("F D 3 M", 227), ("P 21/C", 14), ("P -4 21 M", 113), ("R -3 C", 167),
    ("P 63/M M C", 194), ("P 4 21 2", 90), ("C M C A", 64), ("A B M 2", 39),
    # refused or misread by CRYSTAL23
    ("FM3M", None), ("F M -3 M", None), ("P -4 2 1 M", None), ("P 42 1 2", None),
    ("C M C E", None), ("A E M 2", None), ("", None),
])
def test_symbol_lookup(symbol, number):
    assert spacegroup_number_from_symbol(symbol) == number


def _parse(tmp_path, symbol, cell, atoms):
    deck = tmp_path / "t.d12"
    deck.write_text(textwrap.dedent(f"""\
        t
        CRYSTAL
        1 0 0
        {symbol}
        {cell}
        {len(atoms)}
        """) + "\n".join(atoms) + "\nEND\nBASISSET\nSTO-3G\nSHRINK\n4 8\nEND\n")
    return CrystalInputParser(str(deck)).parse()


@pytest.mark.parametrize("symbol,cell,expected", [
    ("F M 3 M", "5.64", (5.64, 5.64, 5.64, 90.0, 90.0, 90.0)),
    ("F D 3 M", "5.43", (5.43, 5.43, 5.43, 90.0, 90.0, 90.0)),
    ("P 21/C", "5.0 6.0 7.0 100.0", (5.0, 6.0, 7.0, 90.0, 100.0, 90.0)),
    ("P -4 21 M", "5.0 6.0", (5.0, 5.0, 6.0, 90.0, 90.0, 90.0)),
    ("R -3 C", "4.76 13.0", (4.76, 4.76, 13.0, 90.0, 90.0, 120.0)),
])
def test_iflag_1_deck_geometry_is_read(tmp_path, symbol, cell, expected):
    p = _parse(tmp_path, symbol, cell, ["11 0.0 0.0 0.0", "17 0.5 0.5 0.5"])
    cp = p["cell_parameters"]
    assert tuple(cp[k] for k in ("a", "b", "c", "alpha", "beta", "gamma")) == expected
    assert p["n_atoms"] == 2 and "geometry_unparsed" not in p


@pytest.mark.parametrize("symbol", ["F M -3 M", "P -4 2 1 M", "FM3M"])
def test_unreadable_symbol_is_reported_not_silently_dropped(tmp_path, symbol):
    p = _parse(tmp_path, symbol, "5.64", ["11 0.0 0.0 0.0"])
    assert "space group symbol" in p["geometry_unparsed"]
    assert symbol in p["geometry_unparsed"]
    assert "cell_parameters" not in p and "atoms" not in p


def test_numeric_decks_are_unaffected(tmp_path):
    deck = tmp_path / "t.d12"
    deck.write_text("t\nCRYSTAL\n0 0 0\n225\n5.64\n1\n11 0.0 0.0 0.0\nEND\n")
    p = CrystalInputParser(str(deck)).parse()
    assert p["spacegroup"] == 225 and p["cell_record"] == [5.64]
    assert "geometry_unparsed" not in p
    assert SPACEGROUP_SYMBOLS[225] == "Fm-3m"
