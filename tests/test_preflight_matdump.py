"""`mace preflight` on a .d3 deck: a MATDUMP deck is checked without CRYSTAL.

Pre-flight runs the serial `crystal` with TESTPDIM, which only means anything
for a .d12. A .d3 is a `properties` deck: inserting TESTPDIM into it and
handing it to `crystal` says nothing about it. A MATDUMP deck can still be
checked without running anything, because everything that can be wrong with it
is visible on disk:

* the record grammar, manual p.310 (BASISSET: NPR, then NPR prtrec records)
  and p.71 (prtrec: position and value), with print options 60 (overlap) and
  64 (Fock/KS);
* N against the bounds MACE derives from the parent SCF output (the vector
  pool is a hard ceiling; below the derived count is a truncation);
* the wavefunction submit_prop.sh stages (`cp $DIR/$JOB.f9 .../fort.9`).

Every other .d3 kind is reported as not checked, never as passed.
"""
import subprocess
import sys

import pytest

from conftest import REPO_ROOT
from mace.submission import preflight as pf

MACE_CLI = REPO_ROOT / "mace_cli"

_PARENT = "\n".join([
    " TYPE OF CALCULATION :  RESTRICTED CLOSED SHELL",
    " NO.OF VECTORS CREATED 6999 STARS 1021 RMAX   101.51010 BOHR",
    " MAX G-VECTOR INDEX FOR 1- AND 2-ELECTRON INTEGRALS 321",
]) + "\n"


def _matdump(tmp_path, n=321, body=None, parent=_PARENT, f9=b"\x00" * 64,
             stem="mat_matdump"):
    deck = tmp_path / f"{stem}.d3"
    deck.write_text(body if body is not None
                    else f"BASISSET\n2\n60 {n}\n64 {n}\nEND\n")
    if parent is not None:
        (tmp_path / "mat_sp.out").write_text(parent)
    if f9 is not None:
        (tmp_path / f"{stem}.f9").write_bytes(f9)
    return deck


def _check(deck):
    # runner=None would be "structure only" for a .d12; a MATDUMP .d3 must be
    # checked the same way whether or not a CRYSTAL binary exists.
    return pf.preflight_deck(deck, runner=None)


def test_a_good_matdump_deck_is_checked_and_never_reported_as_passed(tmp_path):
    result = _check(_matdump(tmp_path))
    assert result.status == pf.SKIPPED      # CRYSTAL was not run: never PASS
    assert "MATDUMP" in result.reason
    assert "N = 321" in result.reason
    assert "mat_matdump.f9" in result.reason


def test_crystal_is_never_run_on_a_d3(tmp_path):
    def runner(workdir):
        raise AssertionError("pre-flight ran crystal on a properties deck")

    result = pf.preflight_deck(_matdump(tmp_path), runner=runner)
    assert result.status == pf.SKIPPED


@pytest.mark.parametrize("body, why", [
    ("BASISSET\n1\n60 321\n64 321\nEND\n", "NPR"),
    ("BASISSET\n2\n60 321\nEND\n", "NPR"),
    ("BASISSET\n2\n60 321\n64\nEND\n", "pair"),
    ("BASISSET\n2\n60 321\n59 321\nEND\n", "64"),
    ("BASISSET\n2\n60 321\n64 99\nEND\n", "same N"),
    ("BASISSET\nTWO\n60 321\n64 321\nEND\n", "NPR"),
    ("BASISSET\n2\n60 321\n64 321\n", "END"),
])
def test_a_malformed_matdump_record_is_refused(tmp_path, body, why):
    result = _check(_matdump(tmp_path, body=body))
    assert result.status == pf.REFUSED, result
    assert why in result.reason


def test_n_above_the_vector_pool_is_refused(tmp_path):
    result = _check(_matdump(tmp_path, n=7001))
    assert result.status == pf.REFUSED
    assert "6999" in result.reason


def test_n_below_the_derived_count_is_accepted_but_said(tmp_path):
    result = _check(_matdump(tmp_path, n=61))
    assert result.status == pf.SKIPPED
    assert "TRUNCATED" in result.reason
    assert "321" in result.reason


def test_a_molecule_parent_is_refused(tmp_path):
    parent = _PARENT.replace("INTEGRALS 321", "INTEGRALS   1")
    result = _check(_matdump(tmp_path, n=1, parent=parent))
    assert result.status == pf.REFUSED
    assert "MOLECULE" in result.reason


def test_a_missing_wavefunction_is_refused(tmp_path):
    result = _check(_matdump(tmp_path, f9=None))
    assert result.status == pf.REFUSED
    assert "mat_matdump.f9" in result.reason


def test_an_empty_wavefunction_is_refused(tmp_path):
    result = _check(_matdump(tmp_path, f9=b""))
    assert result.status == pf.REFUSED
    assert "empty" in result.reason


def test_a_bare_fort9_is_named_but_not_what_the_job_script_stages(tmp_path):
    deck = _matdump(tmp_path, f9=None)
    (tmp_path / "fort.9").write_bytes(b"\x00" * 64)
    result = _check(deck)
    assert result.status == pf.REFUSED
    assert "fort.9" in result.reason and "mat_matdump.f9" in result.reason


def test_no_parent_output_means_n_could_not_be_bounded(tmp_path):
    result = _check(_matdump(tmp_path, parent=None))
    assert result.status == pf.ERROR
    assert "parent" in result.reason


def test_two_candidate_parents_are_not_guessed_between(tmp_path):
    deck = _matdump(tmp_path)
    (tmp_path / "mat_opt.out").write_text(_PARENT)
    result = _check(deck)
    assert result.status == pf.ERROR
    assert "mat_sp.out" in result.reason and "mat_opt.out" in result.reason


@pytest.mark.parametrize("body, kind", [
    ("NEWK\n12 12\n1 0\nBAND\nX\n1 4 100 1 10 1 0\n0 0 0 2 0 2\nEND\n", "BAND"),
    ("NEWK\n12 12\n1 0\nDOSS\n0 400 1 10 1 14 0\nEND\n", "DOSS"),
])
def test_other_d3_kinds_are_plainly_not_checked(tmp_path, body, kind):
    deck = tmp_path / "mat_band.d3"
    deck.write_text(body)
    result = _check(deck)
    assert result.status == pf.ERROR
    assert "not checked" in result.reason
    assert kind in result.reason
    assert "MATDUMP" in result.reason


def test_refused_counts_as_a_bad_deck_for_the_exit_code(tmp_path):
    bad = pf.PreflightResult(tmp_path / "x.d3", pf.REFUSED, "r")
    assert pf.exit_code([bad]) == 1


@pytest.mark.skipif(not MACE_CLI.is_file(), reason="mace_cli not found")
def test_the_cli_checks_a_d3_without_any_crystal_binary(tmp_path):
    import os

    deck = _matdump(tmp_path)
    env = {k: v for k, v in os.environ.items()
           if k not in ("MACE_CRYSTAL_BIN", "EBROOTCRYSTAL")}
    env["PATH"] = str(tmp_path / "empty")
    proc = subprocess.run(
        [sys.executable, str(MACE_CLI), "preflight", str(deck)],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=300,
        env=env)
    combined = proc.stdout + proc.stderr
    assert "Traceback" not in combined, combined
    assert proc.returncode == 0, combined
    assert "No CRYSTAL binary found" not in combined
    assert "MATDUMP" in combined


@pytest.mark.skipif(not MACE_CLI.is_file(), reason="mace_cli not found")
def test_the_cli_fails_a_bad_matdump_deck(tmp_path):
    deck = _matdump(tmp_path, n=7001)
    proc = subprocess.run(
        [sys.executable, str(MACE_CLI), "preflight", str(deck), "--static-only"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=300)
    assert proc.returncode == 1, proc.stdout + proc.stderr
