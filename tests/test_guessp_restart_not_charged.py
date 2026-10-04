"""A neutral run restarted from another density matrix is not a charged run.

After "RESTART FROM A PREVIOUS DENSITY MATRIX" (GUESSP) CRYSTAL renormalizes
the projected density every cycle (e.g. CHARGE NORMALIZATION FACTOR
0.99987300 in a 216-electron carbon reoptimization). 1.1.5 took any factor
other than 1 as a charged-run signal, so such runs got electron_count /
fragment_charges rows and a "Fragments" section. CHEMOD is not read under
GUESSP, so after a restart only the net charge and "UNIT CELL NOT NEUTRAL"
count. The output is the synthetic neutral Li+ FSI- EC excerpt with the
restart lines added.
"""
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

from conftest import REPO_ROOT
from mace.utils.ion_fragments import fragment_charges

OUT = Path(__file__).resolve().parent / "data" / "charged_runs" / "li_fsi_ec_neutral.out"
FACTOR = " CHARGE NORMALIZATION FACTOR   1.00000000"


def _restarted(factor="0.99987300", extra=""):
    text = OUT.read_text()
    assert FACTOR in text
    return text.replace(FACTOR, " RESTART FROM A PREVIOUS DENSITY MATRIX - DEP ACTIVE\n"
                        f" CHARGE NORMALIZATION FACTOR   {factor}{extra}", 1)


def _analyze(tmp_path, text):
    out = tmp_path / "restarted.out"
    out.write_text(text)
    env = {k: v for k, v in os.environ.items() if k not in ("COLORTERM", "TERM", "NO_COLOR")}
    run = subprocess.run([sys.executable, str(REPO_ROOT / "mace_cli"), "analyze",
                          "--extract-properties", "--output-file", out.name,
                          "--db-path", str(tmp_path / "m.db")],
                         cwd=tmp_path, stdin=subprocess.DEVNULL, capture_output=True,
                         text=True, timeout=300, env=env)
    assert run.returncode == 0, run.stdout + run.stderr
    with sqlite3.connect(tmp_path / "m.db") as conn:
        names = {r[0] for r in conn.execute("SELECT property_name FROM properties")}
    return run.stdout, names


def test_restarted_neutral_run_gets_no_fragment_rows_or_report(tmp_path):
    stdout, names = _analyze(tmp_path, _restarted())
    assert "Fragments" not in stdout
    assert "electron_count" not in names and "fragment_charges" not in names


def test_restart_does_not_hide_a_cell_crystal_called_not_neutral():
    text = _restarted(extra="\n WARNING **** INPBAS **** UNIT CELL NOT NEUTRAL")
    assert "fragment_charges" in fragment_charges(text)


def test_renormalized_chemod_guess_without_restart_still_counts():
    text = OUT.read_text().replace(FACTOR, " CHARGE NORMALIZATION FACTOR   1.00317460", 1)
    assert "fragment_charges" in fragment_charges(text)
