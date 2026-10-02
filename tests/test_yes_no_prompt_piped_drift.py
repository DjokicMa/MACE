"""A yes/no question fed wrong answers from a pipe stops the run.

opt2d12 and cif2d12 gained questions since 1.1.1, so an answer file written
for 1.1.1 can land a menu number on a yes/no question. The question rejected
it, took the next answer, and on a blank fell through to its default: the
run went on and wrote a deck from answers out of step with the questions.
With no terminal, a few invalid answers in a row now stop the run with an
error. At a terminal the question still asks until it gets an answer.
"""
import subprocess
import sys
from pathlib import Path

import pytest

import d12_constants
from conftest import REPO_ROOT

MACE_CLI = REPO_ROOT / "mace_cli"
CIF = Path(__file__).parent / "data" / "1_dia_opt_BULK_OPTGEOM_symm.cif"


def test_cif2d12_stops_on_answers_out_of_step(tmp_path):
    pytest.importorskip("ase", reason="CIF parsing needs ase")
    cif_dir = tmp_path / "cif"
    cif_dir.mkdir()
    (cif_dir / CIF.name).write_bytes(CIF.read_bytes())
    # Menu numbers at "Do you want to use these default settings? [Y/n]"
    r = subprocess.run(
        [sys.executable, str(MACE_CLI), "convert", "--no-banner",
         "--cif_dir", str(cif_dir), "--output_dir", str(tmp_path / "out")],
        cwd=tmp_path, input="1\n1\n3\n" + "\n" * 60,
        capture_output=True, text=True, timeout=300)
    log = r.stdout + r.stderr
    assert r.returncode != 0, log[-3000:]
    assert "not a terminal" in log
    assert not list((tmp_path / "out").glob("*.d12"))


def _answers(monkeypatch, replies, tty):
    replies = iter(replies)
    monkeypatch.setattr(d12_constants, "_nav_read", lambda prompt="", valid_set=None: next(replies))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: tty, raising=False)


def test_one_or_two_bad_answers_are_still_asked_again(monkeypatch):
    _answers(monkeypatch, ["1", "maybe", "n"], tty=False)
    assert d12_constants.yes_no_prompt("Go?", "yes") is False


def test_a_terminal_is_asked_until_it_answers(monkeypatch):
    _answers(monkeypatch, ["1", "2", "3", "4", "5", "y"], tty=True)
    assert d12_constants.yes_no_prompt("Go?", "no") is True


def test_no_terminal_stops_after_three_bad_answers(monkeypatch):
    _answers(monkeypatch, ["1", "1", "3", ""], tty=False)
    with pytest.raises(SystemExit) as exc:
        d12_constants.yes_no_prompt("Keep the basis set?", "yes")
    assert "Keep the basis set?" in str(exc.value)
