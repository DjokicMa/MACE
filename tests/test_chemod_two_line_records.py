"""A CHEMOD record is read as CRYSTAL reads it, whatever its line layout.

CRYSTAL reads CHEMOD (basis-set input, manual p.80) free format: NC, then per
atom its label and one shell charge per shell of its basis. The label may sit
on the charges' line or on a line of its own; both layouts occur. Read one
line per atom, a two-line record whose charges are bare integers ("2 8 1 0")
gave the charges line as a label, so chemod_labels named atoms the record
does not touch, and explicit_chemod_deficit (an explicit basis: the charge
the record sets, carried as DOPING when the CHEMOD cannot be) came out wrong.
"""
from decimal import Decimal

from charge_records import chemod_entries, chemod_labels, explicit_chemod_deficit

# Two atoms (Li, F) with explicit 3-shell bases; CHEMOD makes Li+ (2 0 0) and
# F- (2 8 0 vs a neutral 2 7 0): no net charge.
DECK = """LiF pair
MOLECULE
1
2
3 0.0 0.0 0.0
9 0.0 0.0 1.6
END
3 3
0 0 1 2.0 1.0
10.0 1.0
0 1 1 1.0 1.0
1.0 1.0
0 1 1 0.0 1.0
0.1 1.0
9 3
0 0 1 2.0 1.0
100.0 1.0
0 1 1 7.0 1.0
10.0 1.0
0 1 1 0.0 1.0
1.0 1.0
99 0
{chemod}END
DFT
PBE0
END
END
"""

ONE_LINE = "CHEMOD\n2\n1 2 0 0\n2 2 8 0\n"
TWO_LINE = "CHEMOD\n2\n1\n2 0 0\n2\n2 8 0\n"
TWO_LINE_REMOVING = "CHEMOD\n1\n1\n2 0 0\n"     # Li+ alone: one electron removed


def test_labels_of_a_two_line_record_with_integer_charges():
    rec = TWO_LINE.splitlines()
    assert chemod_labels(rec) == [1, 2]
    assert chemod_entries(rec) == [(1, ["2", "0", "0"]), (2, ["2", "8", "0"])]
    assert chemod_labels(ONE_LINE.splitlines()) == [1, 2]


def test_shell_counts_read_the_record_as_a_token_stream():
    """Any layout, even the charges split over lines."""
    rec = "CHEMOD\n2\n1 2\n0 0 2\n2 8 0\n".splitlines()
    assert chemod_entries(rec, lambda label: 3) == [(1, ["2", "0", "0"]), (2, ["2", "8", "0"])]


def test_explicit_deficit_is_the_same_for_both_layouts():
    for chemod, deficit in ((ONE_LINE, 0), (TWO_LINE, 0), (TWO_LINE_REMOVING, 1)):
        deck = DECK.format(chemod=chemod)
        assert explicit_chemod_deficit(deck, chemod.splitlines()) == Decimal(deficit), chemod
