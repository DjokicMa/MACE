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
    assert 'if [ -f "$DIR/$JOB.f20" ]' in text, "restore must be guarded"
    assert 'cp "$DIR/$JOB.f20" "$scratch/$JOB/fort.20"' in text
    # ...and the save side, which makes a predecessor's matrix available at all.
    assert "cp fort.9 ${DIR}/${JOB}.f9" in text


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


def _stage(tmp_path, block, has_guessp, files, tag):
    import subprocess

    d = tmp_path / tag
    scratch = d / "scratch" / "testmat"
    scratch.mkdir(parents=True)
    (d / "testmat.d12").write_text(
        "title\nCRYSTAL\n" + ("GUESSP\n" if has_guessp else "") + "END\n")
    for name in files:
        (d / name).write_text("MATRIX-" + name)
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
