"""Run Preflight's result, parsed into Problems (plan T6): a CRYSTAL that
rejects the deck under TESTPDIM, and the structure check failing when no
CRYSTAL ran. The CRYSTAL here is a stand-in script written by the test."""
import os
import stat
from pathlib import Path

import pytest

from mace.gui import clirun, preflight

DECK = """MgO test deck
CRYSTAL
0 0 0
225
4.212
2
12 0.0 0.0 0.0
8 0.5 0.5 0.5
OPTGEOM
END
BASISSET
POB-TZVP-REV2
END
"""


@pytest.fixture
def fake_crystal(tmp_path):
    p = tmp_path / "bin" / "crystal"
    p.parent.mkdir()
    p.write_text("#!/bin/sh\ncat > /dev/null\necho ' ERROR **** LoadBa **** UNIT CELL NOT NEUTRAL'\nexit 1\n")
    p.chmod(p.stat().st_mode | stat.S_IXUSR)
    return p


def test_a_testpdim_rejection_becomes_an_error_problem(tmp_path, fake_crystal):
    work = tmp_path / "work"
    work.mkdir()
    r = preflight.run("MgO_test.d12", DECK, work, None, str(fake_crystal))
    assert r["status"] == "rejected" and r["crystal"] == str(fake_crystal)
    assert r["reason"] == "ERROR **** LoadBa **** UNIT CELL NOT NEUTRAL"
    assert [p["severity"] for p in r["problems"]] == ["error"]
    assert r["problems"][0]["message"] == "REJECTED: ERROR **** LoadBa **** UNIT CELL NOT NEUTRAL"


def test_a_failed_structure_check_does_not_blame_crystal(tmp_path, monkeypatch):
    """Without CRYSTAL only MACE's structure check ran: a failure says so, not "REJECTED"."""
    monkeypatch.setattr(preflight, "fast_crystal", lambda explicit=None: None)
    monkeypatch.setattr(clirun, "run", lambda *a, **k: clirun.Result(1, "MgO_test.d12  atoms overlap\n", "", 0.1, 1))
    work = tmp_path / "work"
    work.mkdir()
    r = preflight.run("MgO_test.d12", DECK, work)
    assert r["status"] == "rejected"
    msg = r["problems"][0]["message"]
    assert msg.startswith("Structure check failed (CRYSTAL not run)") and "atoms overlap" in msg, msg
    assert "REJECTED" not in msg
