"""A density matrix staged by hand as fort.20 in the job's scratch directory.

Under 1.1.1 that was the only way to restart from another run's density:
copy it to $SCRATCH/crys23/<job>/fort.20 and put GUESSP in the deck. The
1.1.2 GUESSP staging ignored it - GUESSP was stripped when there was no
<job>.f20 / <job>.f9 next to the deck, and the job's own old <job>.f9 was
copied over it otherwise. A <job>.f20 next to the deck is still the explicit
choice and wins; without one, a non-empty fort.20 already in scratch is kept.

Run through the generated job script, with `mpirun` replaced by a stub that
records the INPUT and fort.20 CRYSTAL would read.
"""
import os
import stat
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GENERATOR = REPO / "mace" / "submission" / "submitcrystal23.sh"
NO_HPCC_USER = "mace_no_such_user_zz"

DECK = ("mat\nCRYSTAL\n0 0 0\n227\n3.567\n1\n6 0.125 0.125 0.125\nEND\n"
        "6 0\nEND\nDFT\nPBE\nEND\nSHRINK\n4 4\nGUESSP\nEND\n")
DECK_2C = ("mat\nCRYSTAL\n0 0 0\n227\n3.567\n1\n6 0.125 0.125 0.125\nEND\n"
           "6 0\nEND\nDFT\nPBE0\nENDDFT\nTWOCOMPON\nSOC\n{guess}\nEND\nEND\n")
HAND = "HAND-STAGED DENSITY\n"

MPIRUN_STUB = """#!/bin/bash
cp INPUT "$DIR/$JOB.ran_input"
if [ -f fort.20 ]; then cp fort.20 "$DIR/$JOB.ran_fort20"; fi
echo "stub crystal run" > fort.9
"""


def _run(tmp_path: Path, deck: str, files=(), hand=HAND, f9_newer=False, empty=()):
    submit = tmp_path / "submit"
    submit.mkdir()
    (submit / "mat.d12").write_text(deck)
    for name in files:
        (submit / name).write_text("MATRIX-" + name)
    for name in empty:
        (submit / name).write_text("")
    scratch = tmp_path / "scr"
    job_dir = scratch / "crys23" / "mat"
    job_dir.mkdir(parents=True)
    if hand is not None:
        (job_dir / "fort.20").write_text(hand)
        # The hand-staged matrix is copied in after the job last ran
        # (f9_newer: the other way round - a fort.20 an earlier run left).
        old, new = time.time() - 7200, time.time() - 60
        f20_t, f9_t = (old, new) if f9_newer else (new, old)
        os.utime(job_dir / "fort.20", (f20_t, f20_t))
        if (submit / "mat.f9").exists():
            os.utime(submit / "mat.f9", (f9_t, f9_t))
    subprocess.run(["bash", str(GENERATOR), "mat"], cwd=submit, check=True,
                   capture_output=True, text=True)
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    mpirun = stubs / "mpirun"
    mpirun.write_text(MPIRUN_STUB)
    mpirun.chmod(mpirun.stat().st_mode | stat.S_IEXEC)
    # The job pins this checkout's queue manager for its completion callback;
    # a no-op `python` keeps the real one from running.
    py = stubs / "python"
    py.write_text("#!/bin/bash\nexit 0\n")
    py.chmod(py.stat().st_mode | stat.S_IEXEC)
    env = {"PATH": f"{stubs}:/usr/bin:/bin", "HOME": str(tmp_path),
           "SLURM_SUBMIT_DIR": str(submit), "SLURM_NTASKS": "1",
           "EBROOTCRYSTAL": str(tmp_path), "SCRATCH": str(scratch), "USER": NO_HPCC_USER}
    r = subprocess.run(["bash", str(submit / "mat.sh")], cwd=submit, env=env,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    ran_input = (submit / "mat.ran_input").read_text()
    f20 = submit / "mat.ran_fort20"
    return ran_input, (f20.read_text() if f20.exists() else None), r.stdout


def _has_guessp(deck_text: str) -> bool:
    return any(l.strip().upper() == "GUESSP" for l in deck_text.splitlines())


def test_hand_staged_fort20_is_kept_when_nothing_is_next_to_the_deck(tmp_path):
    ran, f20, log = _run(tmp_path, DECK)
    assert _has_guessp(ran)                         # not a cold start
    assert f20 == HAND
    assert "kept the fort.20 already in" in log
    assert "dropped it" not in log


def test_hand_staged_fort20_wins_over_the_jobs_own_old_f9(tmp_path):
    ran, f20, log = _run(tmp_path, DECK, files=["mat.f9"])
    assert _has_guessp(ran)
    assert f20 == HAND
    assert "kept the fort.20 already in" in log


def test_a_fort20_older_than_the_jobs_own_f9_is_replaced_by_it(tmp_path):
    """The walltime-killed restart of the same job: the fort.20 an earlier run
    of it left in scratch is older than the .f9 that run ended with."""
    ran, f20, log = _run(tmp_path, DECK, files=["mat.f9"], f9_newer=True)
    assert _has_guessp(ran)
    assert f20 == "MATRIX-mat.f9"
    assert "this job own mat.f9" in log


def test_a_newer_but_empty_f9_does_not_discard_the_fort20(tmp_path):
    """An aborted run copies back a 0-byte fort.9 - no reason to drop the
    hand-staged matrix (and with it GUESSP)."""
    ran, f20, log = _run(tmp_path, DECK, empty=["mat.f9"], f9_newer=True)
    assert f20 == HAND and _has_guessp(ran)


def test_an_f20_next_to_the_deck_still_wins(tmp_path):
    ran, f20, log = _run(tmp_path, DECK, files=["mat.f20", "mat.f9"])
    assert _has_guessp(ran)
    assert f20 == "MATRIX-mat.f20"
    assert "staged mat.f20 as fort.20" in log


def test_an_empty_fort20_in_scratch_is_no_guess(tmp_path):
    ran, f20, log = _run(tmp_path, DECK, files=["mat.f9"], hand="")
    assert f20 == "MATRIX-mat.f9"
    assert "this job own mat.f9" in log


def test_with_nothing_anywhere_guessp_is_dropped_and_the_message_says_where_it_looked(tmp_path):
    ran, f20, log = _run(tmp_path, DECK, hand=None)
    assert not _has_guessp(ran)
    assert f20 is None
    assert "and no fort.20" in log and "mat.f20" in log and "mat.f9" in log
    assert "on disk" not in log


def test_twocompon_guessp_keeps_a_hand_staged_fort20(tmp_path):
    ran, f20, log = _run(tmp_path, DECK_2C.format(guess="GUESSP"), files=["mat.f9"])
    assert _has_guessp(ran)
    assert f20 == HAND
    assert "kept the fort.20 already in" in log
    assert "dropped it" not in log


def test_twocompon_guesspnoso_keeps_a_hand_staged_fort20_without_a_warning(tmp_path):
    ran, f20, log = _run(tmp_path, DECK_2C.format(guess="GUESSPNOSO"))
    assert "GUESSPNOSO" in ran.splitlines()
    assert f20 == HAND
    assert "kept the fort.20 already in" in log
    assert "WARNING" not in log


def test_twocompon_without_any_matrix_still_drops_guessp(tmp_path):
    ran, f20, log = _run(tmp_path, DECK_2C.format(guess="GUESSP"), files=["mat.f9"], hand=None)
    assert not _has_guessp(ran)
    assert f20 is None
    assert "on disk" not in log
