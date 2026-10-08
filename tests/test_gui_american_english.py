"""The text sqlite stores in the shipped example database is American English.

The spelling scan of the GUI's sources, its docs, the example folder and the
logo files is tests/test_american_english.py (one regular expression for all
of MACE); a binary file is not text to it, so this dumps the rows of each
example database and checks them with the same expression.
"""
import sqlite3

import pytest

from conftest import REPO_ROOT
from test_american_english import BRITISH


@pytest.mark.parametrize("path", sorted((REPO_ROOT / "examples" / "gui").glob("*.db")),
                         ids=lambda p: p.name)
def test_example_database_text_is_american_english(path):
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        # the rows the example build wrote; the schema's own status names (a
        # CHECK constraint lists the stored value) are identifiers, not text
        dump = "\n".join(line for line in conn.iterdump() if line.startswith("INSERT"))
    finally:
        conn.close()
    assert not [m.group(0) for m in BRITISH.finditer(dump)]


def test_the_example_database_is_scanned():
    assert sorted((REPO_ROOT / "examples" / "gui").glob("*.db")), "examples/gui has no database to check"
