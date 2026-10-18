"""GUESSP: start the SCF from a previous run's density matrix.

Every MACE run already saves its converged density matrix as ``$JOB.f9``, but
nothing ever read one back, so each step in a chain re-converged from a
superposition of atomic densities. CRYSTAL reads the guess from fort.20 - "copy
file fort.9 to fort.20" - so the job script stages ``$JOB.f20`` and the deck
carries the GUESSP keyword.

Measured on real CRYSTAL23, same MgO cell, same functional and basis:
    cold start                    -275.14904353964 AU, 11 cycles
    GUESSP from the run-1 matrix  -275.14904354956 AU,  5 cycles
with CRYSTAL reporting "SCF GUESS FROM A PREVIOUS RUN DENSITY MATRIX". Same
answer, 6 fewer cycles.
"""
import io
import sys

import pytest

from conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "Crystal_d12"))

from d12_writer import write_scf_section  # noqa: E402

SUBMIT_SH = REPO_ROOT / "mace" / "submission" / "submitcrystal23.sh"


def _scf(**kw):
    buf = io.StringIO()
    args = dict(tolerances={"TOLINTEG": "7 7 7 7 14", "TOLDEE": 7}, k_points=(8, 8, 8),
                dimensionality="CRYSTAL", use_smearing=False, smearing_width=0.0,
                scf_method="DIIS", scf_maxcycle=800, fmixing=30, num_atoms=2,
                spacegroup=1)
    args.update(kw)
    write_scf_section(buf, **args)
    return [ln.strip() for ln in buf.getvalue().splitlines() if ln.strip()]


def test_guessp_is_emitted_when_requested():
    assert "GUESSP" in _scf(guessp=True)


def test_guessp_is_absent_by_default():
    """Opt-in only: a stale fort.20 from a different geometry or basis would be
    a wrong starting guess, and the writer cannot know which it is."""
    assert "GUESSP" not in _scf()
    assert "GUESSP" not in _scf(guessp=False)


def test_guessp_precedes_scfdir_and_the_deck_still_terminates():
    """It belongs in input block 3 with the other SCF keywords, not appended
    after the block has been closed."""
    lines = _scf(guessp=True)
    assert lines.index("GUESSP") < lines.index("SCFDIR")
    assert lines[-1] == "END"


def test_job_script_stages_f20_only_when_present():
    """The restore must be conditional. Every existing deck runs through this
    same script, and an unconditional cp would fail the job for all of them."""
    text = SUBMIT_SH.read_text()
    assert 'if [ -s "$DIR/$JOB.f20" ]' in text, "restore must be guarded"
    assert 'cp "$DIR/$JOB.f20" "$scratch/$JOB/fort.20"' in text
    # ...and the save side, which makes a predecessor's matrix available at all.
    assert 'cp fort.9 "$DIR/$JOB.f9"' in text


def test_job_script_restore_happens_before_crystal_runs():
    """Staged after the binary starts, fort.20 would simply be ignored."""
    text = SUBMIT_SH.read_text()
    assert text.index("$JOB.f20") < text.index("Pcrystal")


# ---------------------------------------------------------------------------
# The staging logic, exercised on the GENERATED script rather than the template
# ---------------------------------------------------------------------------

def _staging_block(tmp_path):
    """Generate a real job script and cut the GUESSP staging block out of it.

    submitcrystal23.sh is a GENERATOR - it echoes the job script line by line -
    so asserting on the template proves nothing about what actually runs. A
    quoting slip inside those echoes (the block uses a bracket expression, and
    the surrounding echo is single-quoted) would emit a broken script while the
    template still read correctly.
    """
    import shutil
    import subprocess

    gen = tmp_path / "submitcrystal23.sh"
    shutil.copy(SUBMIT_SH, gen)
    # Do not actually queue anything.
    gen.write_text(gen.read_text().replace("\nsbatch ", "\n#sbatch "))
    subprocess.run(["bash", str(gen), "testmat"], cwd=tmp_path,
                   capture_output=True, text=True)
    script = (tmp_path / "testmat.sh").read_text()
    assert subprocess.run(["bash", "-n", str(tmp_path / "testmat.sh")]).returncode == 0, \
        "generated job script is not valid shell"
    start = script.index("# GUESSP restart")
    return script[start:script.index("\nfi\n", start) + 4]


def _stage(tmp_path, block, has_guessp, files, tag, empty=()):
    import subprocess

    d = tmp_path / tag
    scratch = d / "scratch" / "testmat"
    scratch.mkdir(parents=True)
    (d / "testmat.d12").write_text(
        "title\nCRYSTAL\n" + ("GUESSP\n" if has_guessp else "") + "END\n")
    for name in files:
        (d / name).write_text("MATRIX-" + name)
    for name in empty:
        (d / name).write_text("")
    # The real script copies the deck to INPUT before the GUESSP block runs,
    # and the block edits that copy - mirror it or the block sees no deck.
    (scratch / "INPUT").write_text((d / "testmat.d12").read_text())
    prelude = f'DIR="{d}"\nJOB=testmat\nscratch="{d}/scratch"\n'
    subprocess.run(["bash", "-c", prelude + block], capture_output=True, text=True)
    staged = scratch / "fort.20"
    return staged.read_text() if staged.exists() else None


def test_explicitly_staged_f20_wins_over_the_jobs_own_f9(tmp_path):
    """A chained step stages the PREDECESSOR's matrix as $JOB.f20; that is more
    specific than whatever this job produced on an earlier attempt."""
    block = _staging_block(tmp_path)
    got = _stage(tmp_path, block, True, ["testmat.f20", "testmat.f9"], "both")
    assert got == "MATRIX-testmat.f20"


def test_falls_back_to_the_materials_own_f9(tmp_path):
    """The walltime-killed restart: re-submitting the same job should pick up
    the matrix it already produced, with no manual staging."""
    block = _staging_block(tmp_path)
    got = _stage(tmp_path, block, True, ["testmat.f9"], "ownf9")
    assert got == "MATRIX-testmat.f9"


def test_guessp_is_stripped_when_there_is_no_matrix_to_restart_from(tmp_path):
    """CRYSTAL aborts on GUESSP with no fort.20 rather than cold starting, so
    leaving the record in would burn the job - and the first step of any chain
    is exactly that case. The scratch copy loses it; the user's deck does not.
    """
    block = _staging_block(tmp_path)
    d = tmp_path / "nomatrix"
    assert _stage(tmp_path, block, True, [], "nomatrix") is None
    assert "GUESSP" not in (d / "scratch" / "testmat" / "INPUT").read_text()
    assert "GUESSP" in (d / "testmat.d12").read_text(), "user deck must be untouched"


def test_nothing_is_staged_when_the_deck_does_not_ask_for_it(tmp_path):
    """CRYSTAL ignores fort.20 without GUESSP, and a .f9 can be large - copying
    it for every job in a sweep would be pure I/O."""
    block = _staging_block(tmp_path)
    assert _stage(tmp_path, block, False, ["testmat.f9"], "noguessp") is None


def test_an_empty_f20_is_not_staged(tmp_path):
    """A 0-byte $JOB.f20 (hand-placed, or a copy of the empty fort.9 an
    aborted run leaves) is no guess: CRYSTAL would read an empty fort.20.
    Fall back to the job's own non-empty .f9, or cold start without one."""
    block = _staging_block(tmp_path)
    got = _stage(tmp_path, block, True, ["testmat.f9"], "emptyf20", empty=["testmat.f20"])
    assert got == "MATRIX-testmat.f9"

    d = tmp_path / "onlyempty"
    assert _stage(tmp_path, block, True, [], "onlyempty", empty=["testmat.f20"]) is None
    assert "GUESSP" not in (d / "scratch" / "testmat" / "INPUT").read_text()


# --- GUESSP inside TWOCOMPON (= GUESSPNOSO, manual p.173) ------------------
#
# Inside a TWOCOMPON block GUESSP / GUESSPNOSO start the 2c-SCF from a
# ONE-component matrix of an unrestricted run without symmetry (p.173), and
# 2NDVARIAT is GUESSPNOSO + PRTENESOC + FMIXING 0 + MAXCYCLE 1 (pp.173-174).
# The job's own $JOB.f9 from an earlier run of a 2c deck is a TWO-component
# matrix, so only a deliberately staged $JOB.f20 may become fort.20.

def _deck_2c(guess, top_level=""):
    return ("title\nCRYSTAL\nEND\n" + top_level + "DFT\nPBE0\nENDDFT\nTWOCOMPON\nSOC\n"
            + (guess + "\n" if guess else "") + "END\nEND\n")


def _stage_deck(tmp_path, block, deck, files, tag):
    import subprocess

    d = tmp_path / tag
    scratch = d / "scratch" / "testmat"
    scratch.mkdir(parents=True)
    (d / "testmat.d12").write_text(deck)
    for name in files:
        (d / name).write_text("MATRIX-" + name)
    (scratch / "INPUT").write_text(deck)
    prelude = f'DIR="{d}"\nJOB=testmat\nscratch="{d}/scratch"\n'
    run = subprocess.run(["bash", "-c", prelude + block], capture_output=True, text=True)
    staged = scratch / "fort.20"
    return (staged.read_text() if staged.exists() else None,
            (scratch / "INPUT").read_text(), run.stdout + run.stderr)


def test_guessp_in_twocompon_never_stages_the_jobs_own_2c_f9(tmp_path):
    block = _staging_block(tmp_path)
    staged, deck, log = _stage_deck(tmp_path, block, _deck_2c("GUESSP"), ["testmat.f9"], "f9")
    assert staged is None
    assert "GUESSP" not in deck.splitlines()          # cold start instead of an abort
    assert "TWOCOMPON" in deck and "SOC" in deck.splitlines()
    assert "TWOCOMPON" in log


def test_guessp_in_twocompon_stages_an_explicit_f20(tmp_path):
    block = _staging_block(tmp_path)
    staged, deck, log = _stage_deck(tmp_path, block, _deck_2c("GUESSP"),
                                    ["testmat.f20", "testmat.f9"], "f20")
    assert staged == "MATRIX-testmat.f20"
    assert "GUESSP" in deck.splitlines()
    assert "SYMMREMO" in log and "p.173" in log


@pytest.mark.parametrize("guess", ["GUESSPNOSO", "guesspnoso", "2NDVARIAT\n0"])
def test_guesspnoso_in_twocompon_is_left_alone_without_an_f20(tmp_path, guess):
    """Explicit GUESSPNOSO matched main's 1c prefix pattern and got the job's
    own 2c .f9 as fort.20."""
    block = _staging_block(tmp_path)
    staged, deck, log = _stage_deck(tmp_path, block, _deck_2c(guess), ["testmat.f9"], "noso")
    assert staged is None
    assert deck == _deck_2c(guess)                    # left as written
    assert "WARNING" in log


def test_guesspnoso_in_twocompon_stages_an_explicit_f20(tmp_path):
    block = _staging_block(tmp_path)
    staged, deck, _ = _stage_deck(tmp_path, block, _deck_2c("GUESSPNOSO"),
                                  ["testmat.f20", "testmat.f9"], "nosof20")
    assert staged == "MATRIX-testmat.f20"
    assert deck == _deck_2c("GUESSPNOSO")


def test_endtwo_closes_the_twocompon_block(tmp_path):
    """The manual's examples close TWOCOMPON with END or ENDTWO (p.172)."""
    block = _staging_block(tmp_path)
    deck = "title\nCRYSTAL\nEND\nTWOCOMPON\nGUESSP\nENDTWO\nEND\n"
    staged, got, _ = _stage_deck(tmp_path, block, deck, ["testmat.f9"], "endtwo")
    assert staged is None and "GUESSP" not in got.splitlines()


# A GUESSP at the top level of a deck that has a TWOCOMPON block is read by
# CRYSTAL like GUESSPNOSO too (measured on HPCC, Bi2 slab, PBE0+SOC): with the
# fort.20 of a 1c UKS SYMMREMO run cycle 0 is exactly the 1c energy; with the
# deck own 2c fort.9 as fort.20 it stops with "io error Read_real_1d"; with a
# 1c matrix computed with symmetry it stops with "READFG RESTART NOT POSSIBLE -
# SYMMETRY INCOMPATIBILITY". So it gets the TWOCOMPON treatment.


def test_top_level_guessp_in_a_twocompon_deck_never_stages_the_jobs_own_2c_f9(tmp_path):
    block = _staging_block(tmp_path)
    deck = _deck_2c(None, top_level="GUESSP\n")
    staged, got, log = _stage_deck(tmp_path, block, deck, ["testmat.f9"], "toplevel")
    assert staged is None                             # was the 2c .f9 -> Read_real_1d crash
    assert "GUESSP" not in got.splitlines()           # cold start instead
    assert "TWOCOMPON" in got and "SOC" in got.splitlines()
    assert "TWOCOMPON" in log


def test_top_level_guessp_in_a_twocompon_deck_stages_an_explicit_f20(tmp_path):
    block = _staging_block(tmp_path)
    deck = _deck_2c(None, top_level="GUESSP\n")
    staged, got, log = _stage_deck(tmp_path, block, deck,
                                   ["testmat.f20", "testmat.f9"], "toplevelf20")
    assert staged == "MATRIX-testmat.f20"
    assert got == deck
    assert "SYMMREMO" in log and "p.173" in log
    # Measured: that guess was read but the SCF then diverged where the cold
    # start converged, so the user is told.
    assert "WARNING" in log and "diverged" in log


def test_top_level_guessp_without_twocompon_still_restarts_from_the_jobs_own_f9(tmp_path):
    block = _staging_block(tmp_path)
    deck = "title\nCRYSTAL\nEND\nGUESSP\nDFT\nPBE0\nENDDFT\nEND\n"
    staged, got, _ = _stage_deck(tmp_path, block, deck, ["testmat.f9"], "onecomp")
    assert staged == "MATRIX-testmat.f9"
    assert got == deck


# ---------------------------------------------------------------------------
# Keywords that only START with "GUESSP" are not GUESSP
# ---------------------------------------------------------------------------
# Manual pp. 170-173: inside TWOCOMPON, GUESSPSO is the guess from a previous
# 2c-SCF density matrix, GUESSP[NOSO] the guess from a previous 1c-SCF one;
# GUESSPAT (p. 115) and GUESSPATNC (p. 171) are atomic guesses that read no
# matrix. The manual says GUESSP reads fort.20 (pp. 114-115) but does not say
# which file GUESSPSO or GUESSPNOSO reads.

def _stage_records(tmp_path, block, records, files, tag):
    """(fort.20 staged, job output, scratch INPUT, deck) for a deck holding
    ``records`` (one per line) between its geometry and its final END."""
    deck = "title\nCRYSTAL\nEND\n" + "".join(r + "\n" for r in records) + "END\n"
    staged, scratch_input, log = _stage_deck(tmp_path, block, deck, files, tag)
    return staged, log, scratch_input, deck


TWOC_SO = ["TWOCOMPON", "SOC", "GUESSPSO", "END"]
TWOC_NOSO = ["TWOCOMPON", "SOC", "GUESSPNOSO", "END"]


def test_guesspso_is_not_treated_as_guessp_and_never_stripped(tmp_path):
    block = _staging_block(tmp_path)
    staged, out, scratch_input, deck = _stage_records(tmp_path, block, TWOC_SO, [], "so_none")
    assert staged is None
    assert scratch_input == deck                         # the record stays as written
    assert "dropped" not in out and "GUESSP requested" not in out
    assert "GUESSPSO" in out and "left in the deck" in out


def test_guesspso_gets_a_staged_f20_or_the_jobs_own_2c_f9_as_fort20(tmp_path):
    block = _staging_block(tmp_path)
    staged, out, scratch_input, deck = _stage_records(
        tmp_path, block, TWOC_SO, ["testmat.f20", "testmat.f9"], "so_both")
    assert staged == "MATRIX-testmat.f20" and scratch_input == deck
    assert out.startswith("GUESSPSO: staged testmat.f20 as fort.20")
    staged, out, scratch_input, deck = _stage_records(tmp_path, block, TWOC_SO, ["testmat.f9"], "so_f9")
    assert staged == "MATRIX-testmat.f9" and scratch_input == deck
    assert out.startswith("GUESSPSO: staged this job own 2c testmat.f9 as fort.20")


def test_guesspnoso_never_gets_the_jobs_own_2c_f9(tmp_path):
    """GUESSPNOSO wants a 1c matrix; this 2c deck's own .f9 is a 2c one."""
    block = _staging_block(tmp_path)
    staged, out, scratch_input, deck = _stage_records(tmp_path, block, TWOC_NOSO, ["testmat.f9"], "noso_f9")
    assert staged is None and scratch_input == deck and "dropped" not in out
    staged, out, scratch_input, deck = _stage_records(tmp_path, block, TWOC_NOSO, ["testmat.f20"], "noso_f20")
    assert staged == "MATRIX-testmat.f20" and scratch_input == deck


def test_the_no_matrix_echo_is_worded_per_keyword(tmp_path):
    """Only GUESSPSO was measured to stop CRYSTAL with nothing in fort.20; a
    GUESSPNOSO job must not be told that about GUESSPSO."""
    block = _staging_block(tmp_path)
    _, out, _, _ = _stage_records(tmp_path, block, TWOC_SO, [], "so_echo")
    assert "CRYSTAL stops on GUESSPSO without fort.20 (measured)" in out
    # GUESSPNOSO inside TWOCOMPON takes 1.1.5's one-component branch, whose
    # warning names the one-component $JOB.f20 it needs
    _, out, _, _ = _stage_records(tmp_path, block, TWOC_NOSO, ["testmat.f9"], "noso_echo")
    assert out.startswith("WARNING: GUESSPNOSO in a TWOCOMPON deck needs a one-component")
    assert "GUESSPSO" not in out


def test_an_atomic_guess_stages_nothing(tmp_path):
    block = _staging_block(tmp_path)
    for i, records in enumerate((["GUESSPAT"], ["TWOCOMPON", "SOC", "GUESSPATNC", "1",
                                                 "1 90.0 90.0 1.0", "ENDTWO"])):
        staged, out, scratch_input, deck = _stage_records(tmp_path, block, records,
                                                       ["testmat.f9"], f"pat{i}")
        assert staged is None and scratch_input == deck and out == ""
