"""opt2d12 never drops a requested spin state, and never keeps one that no
longer fits the electron count.

Measured on the corpus diamond (test/OPT/1_dia_opt_rev1: a symmetric deck
with SPINLOCK 0 50, 12 electrons):

* ``--net-charge 1 --spinlock 1`` wrote DOPING -1 with the parent's SPINLOCK
  0 50 (the requested value was dropped because the deck already had a
  SPINLOCK). 11 electrons cannot hold nalpha-nbeta = 0, so CRYSTAL stops
  ("NALPHA-NBETA NOT COMPATIBLE").
* ``--net-charge 1`` alone wrote the same impossible deck with rc 0.
* ``--spinlock 2`` (or ``--chemod yes``) without a charge change was ignored
  silently, rc 0, no SPINLOCK; the 1.1.6 GUI offers these switches alone.
"""
import os
import shutil
import subprocess
import sys

import pytest

from conftest import REPO_ROOT, find_data


def _env():
    env = dict(os.environ)
    for k in ("COLORTERM", "TERM", "NO_COLOR"):
        env.pop(k, None)
    return env


@pytest.fixture
def dia(tmp_path):
    out = find_data("OPT/1_dia_opt_rev1.out")
    shutil.copy(out, tmp_path / "dia.out")
    shutil.copy(out.with_suffix(".d12"), tmp_path / "dia.d12")
    assert "SPINLOCK" in (tmp_path / "dia.d12").read_text()
    return tmp_path


def _run(cwd, *extra):
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "mace_cli"), "opt2d12", "--out-file", "dia.out",
         "--d12-file", "dia.d12", "--calc-type", "SP", "--non-interactive",
         "--output-dir", "o", *extra],
        cwd=cwd, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=300,
        env=_env())
    decks = list((cwd / "o").glob("*.d12")) if (cwd / "o").exists() else []
    lines = decks[0].read_text().splitlines() if decks else None
    return proc, lines


def _record(lines, key):
    i = lines.index(key)
    return lines[i + 1].split()


def test_requested_spinlock_replaces_the_parents(dia):
    proc, lines = _run(dia, "--net-charge", "1", "--spinlock", "1", "--spinlock-cycles", "20")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _record(lines, "DOPING") == ["-1"]
    assert _record(lines, "SPINLOCK") == ["1", "20"]
    assert lines.count("SPINLOCK") == 1


def test_odd_charge_with_the_parents_even_spinlock_is_refused(dia):
    proc, lines = _run(dia, "--net-charge", "1")
    assert proc.returncode == 1
    assert lines is None
    text = proc.stdout + proc.stderr
    assert "SPINLOCK 0 50" in text and "--spinlock 1" in text


def test_wrong_parity_spinlock_is_refused(dia):
    proc, lines = _run(dia, "--net-charge", "1", "--spinlock", "2")
    assert proc.returncode == 1 and lines is None
    assert "odd" in proc.stdout + proc.stderr


def test_even_charge_keeps_the_parents_spinlock(dia):
    proc, lines = _run(dia, "--net-charge", "2")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _record(lines, "SPINLOCK") == ["0", "50"]
    assert _record(lines, "DOPING") == ["-2"]


def test_spinlock_alone_is_written(dia):
    proc, lines = _run(dia, "--spinlock", "2")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _record(lines, "SPINLOCK") == ["2", "50"]
    assert lines.count("SPINLOCK") == 1


def test_spinlock_alone_checks_the_parity_from_the_parent_out(dia):
    """The deck has symmetry, so the parent .out's 12 electrons decide."""
    proc, lines = _run(dia, "--spinlock", "1")
    assert proc.returncode == 1 and lines is None
    assert "12 electrons" in proc.stdout + proc.stderr


def test_spinlock_cycles_alone_changes_the_cycles(dia):
    proc, lines = _run(dia, "--spinlock-cycles", "9")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _record(lines, "SPINLOCK") == ["0", "9"]


def test_chemod_alone_is_refused(dia):
    proc, lines = _run(dia, "--chemod", "yes")
    assert proc.returncode == 1 and lines is None
    assert "--net-charge" in proc.stdout + proc.stderr
