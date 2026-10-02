"""opt2d12 with only a parent .out and no Mulliken print: warn only on a sign of charge.

The E4b note ("a charged parent's state cannot be recovered") was a WARNING
for every parent .out without a Mulliken print - every neutral FREQ parent
among them. Most such parents are neutral. It is now a WARNING only when the
.out shows a sign of a charged or ionic run: CHEMOD ("ELECTRONIC
CONFIGURATION MODIFIED IN n ATOMS"), CHARGED ("CHARGED CELL"), "UNIT CELL
NOT NEUTRAL", or a renormalised guess (CHARGE NORMALIZATION FACTOR other
than 1). Otherwise it is an informational note: DOPING prints nothing
(measured on CRYSTAL23), so the note still says to add it by hand if the
parent had one.
"""
import os
import subprocess
import sys

import pytest

from charge_records import charge_from_parent_out
from conftest import REPO_ROOT
from ion_fixtures import molecule_deck, synth_out

LI_WATER = [("Li", (0.0, 0.0, 0.0)), ("O", (1.95, 0.0, 0.0)), ("H", (2.54, 0.76, 0.0)),
            ("H", (2.54, -0.76, 0.0))]
CHILD = molecule_deck(LI_WATER, title="child")

SIGNS = {
    "chemod": " *******************************************************************************\n"
              " * ELECTRONIC CONFIGURATION MODIFIED IN    1 ATOMS\n"
              " *******************************************************************************\n"
              "   1(  3)   2.0   0.0   0.0   0.0   0.0\n",
    "charged": " INFORMATION **** CHARGED ****  CHARGED CELL - NO TOTAL ENERGY\n",
    "not_neutral": " WARNING **** INPBAS **** UNIT CELL NOT NEUTRAL\n",
    "renormalised": " CHARGE NORMALIZATION FACTOR   0.91666667\n",
}


def test_a_neutral_looking_out_gets_an_informational_note():
    out, notes = charge_from_parent_out(CHILD, synth_out(LI_WATER, title="parent"))
    assert out == CHILD
    assert notes and not any(n.startswith("WARNING") for n in notes), notes
    assert "DOPING" in notes[0]  # still says what cannot be seen


def test_a_normalisation_factor_of_one_is_no_sign():
    extra = " CHARGE NORMALIZATION FACTOR   1.00000000\n"
    _, notes = charge_from_parent_out(CHILD, synth_out(LI_WATER, title="parent", extra=extra))
    assert not any(n.startswith("WARNING") for n in notes), notes


@pytest.mark.parametrize("sign", sorted(SIGNS))
def test_a_sign_of_charge_keeps_the_warning(sign):
    out, notes = charge_from_parent_out(CHILD, synth_out(LI_WATER, title="parent", extra=SIGNS[sign]))
    assert out == CHILD
    assert any(n.startswith("WARNING") and "cannot be recovered" in n for n in notes), notes


def _run(tmp_path):
    env = {k: v for k, v in os.environ.items() if k not in ("COLORTERM", "TERM", "NO_COLOR")}
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "mace_cli"), "opt2d12", "--out-file", "parent.out",
         "--calc-type", "SP", "--non-interactive", "--output-dir", "o"],
        cwd=tmp_path, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=300, env=env)


def test_opt2d12_on_a_neutral_parent_out_does_not_warn(tmp_path):
    (tmp_path / "parent.out").write_text(synth_out(LI_WATER, title="parent"))
    run = _run(tmp_path)
    said = run.stdout + run.stderr
    assert run.returncode == 0, said
    assert "cannot be recovered" not in said
    assert "[WARN]" not in " ".join(l for l in said.splitlines() if "Mulliken" in l)


def test_opt2d12_on_a_chemod_parent_out_warns(tmp_path):
    (tmp_path / "parent.out").write_text(synth_out(LI_WATER, title="parent", extra=SIGNS["chemod"]))
    run = _run(tmp_path)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "cannot be recovered" in run.stdout + run.stderr
