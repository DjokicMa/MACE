"""A FREQ deck's default SCF tolerances never loosen a parent's value.

FREQ defaults to Very tight (TOLINTEG 9 9 9 11 38, TOLDEE 11), which used to
replace a parent's tolerances whatever they were: a parent at 10 10 10 12 40 /
12 got a looser FREQ deck. Now each TOLINTEG entry and TOLDEE is the larger of
the parent's and Very tight's, on every route (the menu's default,
--non-interactive, --config-file as the workflow engine runs it, Setup). A
parent tighter in every value keeps its own ("5. Keep the parent's"); one
tighter in some values only gets the merged set as choice 6. The decks run the
real ``mace_cli`` with the Si parent of tests/fixtures/phonopy_crystal/Si.
"""
import json
import os
import shutil
import subprocess
import sys

import pytest

from conftest import REPO_ROOT
from d12_constants import FREQ_TIGHTER_OPTION, freq_default_tolerances, scf_tolerances

MACE_CLI = REPO_ROOT / "mace_cli"
SI = REPO_ROOT / "tests" / "fixtures" / "phonopy_crystal" / "Si"
TIGHTER = ("10 10 10 12 40", "12")      # tighter than Very tight in every value
MIXED = ("10 8 10 8 50", "9")           # tighter in three TOLINTEG entries, looser elsewhere
MERGED = ("10 9 10 11 50", "11")


def _parent(tmp_path, tolinteg, toldee):
    text = (SI / "parent.d12").read_text()
    text = text.replace("TOLINTEG\n8 8 8 8 16", f"TOLINTEG\n{tolinteg}").replace("TOLDEE\n9", f"TOLDEE\n{toldee}")
    (tmp_path / "si.d12").write_text(text)
    return tmp_path / "si.d12"


def _run(tmp_path, args, stdin=""):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, str(MACE_CLI), "opt2d12", "--d12-file", "si.d12", *args],
                       cwd=tmp_path, input=stdin, capture_output=True, text=True, timeout=300, env=env)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    decks = [p for p in tmp_path.rglob("*.d12") if p.name != "si.d12"]
    assert len(decks) == 1, decks
    return r.stdout, decks[0].read_text()


def _scf(deck):
    lines = [" ".join(x.split()) for x in deck.splitlines()]
    i = lines.index("TOLINTEG")
    return lines[i + 1], lines[lines.index("TOLDEE", i) + 1]


def test_the_rule():
    very = scf_tolerances("3")
    assert freq_default_tolerances("SP", None) == very
    assert freq_default_tolerances("OPT", scf_tolerances("1")) == very
    assert freq_default_tolerances("OPT", scf_tolerances("3")) == very
    assert freq_default_tolerances("OPT", {"TOLINTEG": TIGHTER[0], "TOLDEE": 12}) == {
        "TOLINTEG": TIGHTER[0], "TOLDEE": 12}
    assert freq_default_tolerances("SP", {"TOLINTEG": "10  8 10 8 50", "TOLDEE": "9"}) == {
        "TOLINTEG": MERGED[0], "TOLDEE": 11}
    assert freq_default_tolerances("SP", {"TOLDEE": 13}) == {"TOLINTEG": "9 9 9 11 38", "TOLDEE": 13}
    # a FREQ parent keeps its own, looser or not
    assert freq_default_tolerances("FREQ", scf_tolerances("2")) == scf_tolerances("2")


def test_enter_keeps_a_tighter_parent(tmp_path):
    _parent(tmp_path, *TIGHTER)
    out, deck = _run(tmp_path, ["--calc-type", "FREQ"], "\n" * 200)
    assert "5. Keep the parent's - TOLINTEG: 10 10 10 12 40, TOLDEE: 12" in out
    assert "Select SCF convergence level [1-5, default=5]" in out
    assert "FREQCALC" in deck and _scf(deck) == TIGHTER


def test_enter_takes_the_tighter_of_each_value(tmp_path):
    _parent(tmp_path, *MIXED)
    out, deck = _run(tmp_path, ["--calc-type", "FREQ"], "\n" * 200)
    assert f"6. {FREQ_TIGHTER_OPTION} - TOLINTEG: 10 9 10 11 50, TOLDEE: 11" in out
    assert "Select SCF convergence level [1-6, default=6]" in out
    assert _scf(deck) == MERGED


def test_a_looser_parent_still_gets_very_tight(tmp_path):
    _parent(tmp_path, "8 8 8 8 16", "9")
    out, deck = _run(tmp_path, ["--calc-type", "FREQ"], "\n" * 200)
    assert "Select SCF convergence level [1-5, default=3]" in out
    assert _scf(deck) == ("9 9 9 11 38", "11")


@pytest.mark.parametrize("tol, want", [(TIGHTER, TIGHTER), (MIXED, MERGED)])
def test_non_interactive(tmp_path, tol, want):
    _parent(tmp_path, *tol)
    _, deck = _run(tmp_path, ["--calc-type", "FREQ", "--non-interactive", "--output-dir", "out"])
    assert _scf(deck) == want


@pytest.mark.parametrize("tol, want", [(TIGHTER, TIGHTER), (MIXED, MERGED)])
def test_config_file_as_the_workflow_engine_runs_it(tmp_path, tol, want):
    """The planner's FREQ steps name no tolerances: opt2d12's rule applies to them."""
    _parent(tmp_path, *tol)
    (tmp_path / "cfg.json").write_text(json.dumps({"calculation_type": "FREQ"}))
    _, deck = _run(tmp_path, ["--config-file", "cfg.json", "--yes", "--output-dir", "out"])
    assert _scf(deck) == want
