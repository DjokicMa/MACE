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
