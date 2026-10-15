"""IFSO > 1 decks in the GUI reader: the origin shift CRYSTAL applies.

Before, ``structures.d12_atoms`` ignored the IX IY IZ record, so a deck with a
non-standard origin was drawn as if IFSO were 0 (a different structure).
"""
import pytest

from test_external_geometry_deck import BASIS, _deck
from test_gui_examples import needs_ase


# --- IFSO > 1: a non-standard origin shift ------------------------------------
# CRYSTAL23 manual p.19: "IFSO >1 non-standard shift of the origin given as
# input", record IX IY IZ "in fractions of the crystallographic cell lattice
# vectors times 24". The manual gives no direction; CRYSTAL23 itself does
# (TESTGEOM on MSU HPCC, 2026-10-14, P-1 with C 0.1 0.2 0.3 and O 0.3 0.1 0.05):
# the cell CRYSTAL prints, atoms in the deck's own frame.
IFSO_CELL = "5.0 6.0 7.0 80 85 95\n2\n6 0.1 0.2 0.3\n8 0.3 0.1 0.05\nEND\n"
IFSO_CRYSTAL_OUT = {
    "": [(6, 0.1, 0.2, 0.3), (6, -0.1, -0.2, -0.3), (8, 0.3, 0.1, 0.05), (8, -0.3, -0.1, -0.05)],
    "3 0 0": [(6, 0.1, 0.2, 0.3), (6, -0.35, -0.2, -0.3), (8, 0.3, 0.1, 0.05), (8, 0.45, -0.1, -0.05)],
    "0 3 6": [(6, 0.1, 0.2, 0.3), (6, -0.1, -0.45, 0.2), (8, 0.3, 0.1, 0.05), (8, -0.3, -0.35, 0.45)],
}


def _same_sites(frac, numbers, expected):
    import numpy as np
    got = sorted((int(n), *np.round(np.mod(f, 1.0), 6) % 1.0) for n, f in zip(numbers, frac))
    want = sorted((n, *np.round(np.mod((x, y, z), 1.0), 6) % 1.0) for n, x, y, z in expected)
    assert len(got) == len(want)
    for g, w in zip(got, want):
        assert g[0] == w[0] and max(abs(a - b) for a, b in zip(g[1:], w[1:])) < 1e-6, (got, want)


@needs_ase
@pytest.mark.parametrize("shift", ["", "3 0 0", "0 3 6"])
def test_origin_shift_expands_as_crystal_does(tmp_path, shift):
    from mace.gui import structures
    flags = "0 0 2\n2\n" + shift + "\n" if shift else "0 0 0\n2\n"
    s = structures.read_structure(_deck(tmp_path, "t\nCRYSTAL\n" + flags + IFSO_CELL + BASIS, "p1bar.d12"))
    _same_sites(s["frac"], s["numbers"], IFSO_CRYSTAL_OUT[shift])


def test_parser_and_opt2d12_writer_keep_the_record_in_24ths(tmp_path):
    """The parser keeps IX IY IZ as written (24ths) and opt2d12's writer
    writes them back unchanged, so a derived deck keeps the parent's origin;
    only the GUI reader divides by 24."""
    from d12_parsers import CrystalInputParser
    from charge_records import origin_shift_record
    deck = _deck(tmp_path, "t\nCRYSTAL\n0 0 2\n2\n0 3 6\n" + IFSO_CELL + BASIS, "p1bar.d12")
    parsed = CrystalInputParser(str(deck)).parse()
    assert parsed["origin_setting"] == "0 0 2"
    assert parsed["origin_shift"] == [0.0, 3.0, 6.0]
    assert origin_shift_record(parsed["origin_shift"]) == "0 3 6"
