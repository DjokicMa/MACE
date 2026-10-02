"""opt2d12 answers piped in an older order no longer change a deck silently.

1.1.5 added questions to opt2d12 (the grid, one LEVSHIFT "ISHIFT ILOCK"
answer, the HISTDIIS size), so an answer file written for 1.1.1 puts its
answers on other questions. Eleven blanks then "3" on 1_dia_opt_BULK_OPTGEOM
meant "very tight SCF" in 1.1.1; it now lands on the grid menu and writes
LGRID, exit 0, with nothing said. A valid answer on the wrong question cannot
be told from a deliberate one, so:

* a piped run lists every drift-prone record that differs from the parent;
* the free-text LEVSHIFT and HISTDIIS answers are checked, asked again when
  invalid, and stop a piped run after three (they fell back to the default,
  which shifted the rest of the file by one more answer).
"""
import builtins
import os
import shutil
import subprocess
import sys

import pytest

import d12_interactive
from conftest import REPO_ROOT, TEST_DATA

MACE_CLI = REPO_ROOT / "mace_cli"


def _opt2d12(tmp_path, answers):
    parent = TEST_DATA / "OPT" / "1_dia_opt_BULK_OPTGEOM"
    if not parent.with_suffix(".out").exists():
        pytest.skip("test/ corpus not present (gitignored)")
    for ext in (".out", ".d12"):
        shutil.copy(parent.with_suffix(ext), tmp_path)
    env = {k: v for k, v in os.environ.items()
           if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME")}
    r = subprocess.run([sys.executable, str(MACE_CLI), "opt2d12", "--no-banner",
                        "--out-file", "1_dia_opt_BULK_OPTGEOM.out"], cwd=tmp_path,
                       input=answers, capture_output=True, text=True, timeout=300, env=env)
    return r, r.stdout + r.stderr


def test_a_1_1_1_answer_file_says_what_it_changed(tmp_path):
    r, log = _opt2d12(tmp_path, "\n" * 11 + "3\n" + "\n" * 80)
    assert r.returncode == 0, log[-3000:]
    deck = next(tmp_path.glob("*_optimized.d12")).read_text()
    assert "LGRID" in deck.split()
    assert "differ from the parent deck" in log, log[-3000:]
    assert "DFT grid: DEFAULT -> LGRID" in log


def test_all_defaults_keep_the_parent_and_say_nothing(tmp_path):
    r, log = _opt2d12(tmp_path, "\n" * 92)
    assert r.returncode == 0, log[-3000:]
    assert "differ from the parent deck" not in log


def _feed(monkeypatch, replies, tty=False):
    it = iter(replies)
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(it))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: tty, raising=False)


def test_levshift_takes_two_numbers_or_blank(monkeypatch):
    _feed(monkeypatch, ["3 0"])
    assert d12_interactive._ask_levshift((5, 1)) == (3, 0)
    _feed(monkeypatch, [""])
    assert d12_interactive._ask_levshift((5, 1)) == (5, 1)


def test_levshift_asks_again_for_a_1_1_1_single_value(monkeypatch):
    # 1.1.1 asked the shift, then the cycles, on two lines
    _feed(monkeypatch, ["5", "4 1"])
    assert d12_interactive._ask_levshift((5, 1)) == (4, 1)


def test_levshift_stops_a_piped_run_out_of_step(monkeypatch):
    _feed(monkeypatch, ["5", "10", "y", ""])
    with pytest.raises(SystemExit) as exc:
        d12_interactive._ask_levshift((5, 1))
    assert "not a terminal" in str(exc.value)


def test_histdiis_size_is_checked(monkeypatch):
    _feed(monkeypatch, ["y", "60"])
    assert d12_interactive._ask_positive_int("HISTDIIS size [100]: ", 100) == 60
    _feed(monkeypatch, [""])
    assert d12_interactive._ask_positive_int("HISTDIIS size [100]: ", 100) == 100
    _feed(monkeypatch, ["y", "n", "0", ""])
    with pytest.raises(SystemExit):
        d12_interactive._ask_positive_int("HISTDIIS size [100]: ", 100)
