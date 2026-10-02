"""Every record of the external basis families MACE writes into decks is neutral.

CRYSTAL23 checks that the formal shell charges of the cell's atoms add up to
its nuclear charge, and stops with "INPBAS UNIT CELL NOT NEUTRAL" otherwise
(measured for LaN and CeO2 with the default TZ family). The La-Lu records
(ERD small-core 2017) carried the site's trivalent-ion charges, ZNUC - 3, and
the Ac-Es records an f shell 3 lower than the site's; every La-Lu or Ac-Es deck
on full.basis.triplezeta / doublezeta, and every Ac-Es deck on
full.basis.triplezeta.rev2022, failed. A record's shell charges must add up
to its ZNUC (the INPUT pseudopotential's) or Z (all-electron).
"""
import pytest

import basis_library

DECK_FAMILIES = ("full.basis.triplezeta", "full.basis.doublezeta",
                 "full.basis.triplezeta.rev2022")


def charges(body):
    """(ZNUC, sum of the formal shell charges) of a record."""
    lines = body.splitlines()
    label, nshell = (int(t) for t in lines[0].split()[:2])
    znuc, i = float(label % 100), 1
    if lines[1].strip().upper() == "INPUT":
        head = lines[2].split()
        znuc, i = float(head[0]), 3 + sum(int(x) for x in head[1:7])
    total = 0.0
    for _ in range(nshell):
        rec = lines[i].split()
        total += float(rec[3])
        i += 1 + (int(rec[2]) if int(rec[0]) == 0 else 0)
    return znuc, total


@pytest.mark.parametrize("family", DECK_FAMILIES)
def test_shell_charges_add_up_to_the_nuclear_charge(family):
    lib = basis_library.load_family(family)
    off = {}
    for key in lib.keys():
        znuc, total = charges(lib.record_bytes(key).decode("ascii"))
        if abs(znuc - total) > 1e-9:
            off[key] = (znuc, total)
    assert off == {}


def test_ln_records_keep_the_published_4f_count():
    """Only the empty 6s and 5d shells got the three electrons: Eu keeps 4f6."""
    body = basis_library.load_family("full.basis.triplezeta").record_bytes("263").decode()
    f_shells = [l.split() for l in body.splitlines() if l.startswith("0 4 ")]
    assert float(f_shells[0][3]) == 6.0
