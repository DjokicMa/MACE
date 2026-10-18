"""`mace opt2d12 --d12-file DECK` (1.1.6: a deck is its own parent, no .out)
with a SOC template.

The SOC code read the parent .out for metallic evidence and the parent .f9 for
a 2c restart; with a deck as the parent there is neither. A SOC single point
is written from the deck (no metal evidence to read), and a 2c restart is
refused with the reason: a deck has no density matrix.
"""
import json
import shutil
import subprocess
import sys

from conftest import REPO_ROOT

MACE_CLI = REPO_ROOT / "mace_cli"
AGBR = next((REPO_ROOT / "tests" / "data" / "ecp_decks").glob("Ag1Br1*.d12"))


def _run(tmp_path, template):
    shutil.copy2(AGBR, tmp_path / AGBR.name)
    (tmp_path / "t.json").write_text(json.dumps(template))
    r = subprocess.run([sys.executable, str(MACE_CLI), "--no-banner", "opt2d12",
                        "--d12-file", AGBR.name, "--config-file", "t.json", "--yes"],
                       cwd=tmp_path, input="\n" * 40, capture_output=True, text=True, timeout=300)
    kids = sorted(p for p in tmp_path.glob("*.d12") if p.name != AGBR.name)
    return r.returncode, r.stdout + r.stderr, kids


def test_a_soc_single_point_from_a_deck(tmp_path):
    code, log, kids = _run(tmp_path, {"calculation_type": "SP", "soc": True})
    assert code == 0, log[-3000:]
    assert "Traceback" not in log
    (kid,) = kids
    text = kid.read_text()
    assert "TWOCOMPON\nSOC\nEND\nSCFDIR\n" in text and "INPSOC\nINTERNAL 1.0\n" in text
    assert "Warning: SOC deck: Br (atomic number 35) is all-electron" in log


def test_a_2c_restart_from_a_deck_is_refused_with_the_reason(tmp_path):
    code, log, kids = _run(tmp_path, {"calculation_type": "SP", "soc": True,
                                      "soc_guess": "guesspso"})
    assert "Traceback" not in log
    assert code != 0 and kids == []
    assert "GUESSPSO" in log and "a deck does not have" in log
