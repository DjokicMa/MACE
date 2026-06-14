"""Regression tests for the error-recovery chain (Wave B1, commit acf632c9).

These are self-contained (no test/ corpus needed): a .d12 is an INPUT file, so a
minimal representative deck built in tmp exercises the recovery input-editing and
resubmission logic directly. SLURM submission is mocked.
"""
import pytest

# A minimal, structurally representative CRYSTAL d12 with BOTH an OPTGEOM
# MAXCYCLE (geometry steps) and an SCF MAXCYCLE — the case that exposed the
# block-awareness bug.
D12 = """\
TITLE
CRYSTAL
0 0 0
1
2.0
1
6 0.0 0.0 0.0
OPTGEOM
MAXCYCLE
50
ENDOPT
END
BASISSET
POB-TZVP-REV2
DFT
PBE
ENDDFT
TOLDEE
9
FMIXING
30
MAXCYCLE
100
END
"""


def _scf_maxcycle(path):
    """The SCF (non-OPTGEOM) MAXCYCLE value in a d12, or None."""
    lines = path.read_text().splitlines()
    in_opt = False
    for i, line in enumerate(lines):
        k = line.strip().upper()
        if k == "OPTGEOM":
            in_opt = True
        elif k in ("ENDOPT", "ENDOPTGEOM") or (in_opt and k == "END"):
            in_opt = False
        if k == "MAXCYCLE" and not in_opt and i + 1 < len(lines):
            return int(lines[i + 1].strip())
    return None


def _optgeom_maxcycle(path):
    lines = path.read_text().splitlines()
    in_opt = False
    for i, line in enumerate(lines):
        k = line.strip().upper()
        if k == "OPTGEOM":
            in_opt = True
        elif k in ("ENDOPT", "ENDOPTGEOM") or (in_opt and k == "END"):
            in_opt = False
        if k == "MAXCYCLE" and in_opt and i + 1 < len(lines):
            return int(lines[i + 1].strip())
    return None


@pytest.fixture
def engine(tmp_path):
    from mace.recovery.recovery import ErrorRecoveryEngine
    return ErrorRecoveryEngine(db_path=str(tmp_path / "rec.db"))


@pytest.fixture
def original_d12(tmp_path):
    p = tmp_path / "job.d12"
    p.write_text(D12)
    return p


def test_convergence_handler_bumps_only_scf_maxcycle(engine, original_d12, tmp_path):
    calc = {"calc_id": "C1", "material_id": "M1", "calc_type": "OPT",
            "input_file": str(original_d12), "work_dir": str(tmp_path),
            "error_type": "convergence_error"}
    res = engine.attempt_recovery(calc, create_record=False)
    assert res is not None
    fixed = res["fixed_input_file"]
    assert "_recovery_" in fixed.name and fixed.exists()
    # SCF MAXCYCLE bumped by the default 1000; OPTGEOM MAXCYCLE untouched.
    assert _scf_maxcycle(fixed) == 1100
    assert _optgeom_maxcycle(fixed) == 50


def test_attempt_recovery_returns_artifacts_dict(engine, original_d12, tmp_path):
    calc = {"calc_id": "C1", "material_id": "M1", "calc_type": "OPT",
            "input_file": str(original_d12), "work_dir": str(tmp_path),
            "error_type": "convergence_error"}
    res = engine.attempt_recovery(calc, create_record=False)
    assert set(res) >= {"fixed_input_file", "fixed_job_script", "recovery_calc_id"}
    assert res["fixed_job_script"] is None          # input fix, no script change
    assert res["recovery_calc_id"] is None          # create_record=False -> no row


def test_memory_handler_returns_bumped_script(engine, original_d12, tmp_path):
    script = tmp_path / "job.sh"
    script.write_text("#!/bin/bash\n#SBATCH --mem=80GB\n#SBATCH --time=24:00:00\n")
    calc = {"calc_id": "C2", "material_id": "M1", "calc_type": "OPT",
            "input_file": str(original_d12), "work_dir": str(tmp_path),
            "job_script": str(script), "error_type": "memory_error"}
    res = engine.attempt_recovery(calc, create_record=False)
    assert res["fixed_input_file"] == original_d12      # input unchanged
    bumped = res["fixed_job_script"]
    assert bumped is not None and bumped.exists()
    assert "--mem=120GB" in bumped.read_text()          # 80 * 1.5


def test_memory_handler_preserves_mem_per_cpu_form(engine, original_d12, tmp_path):
    """B1 regression: the standard CRYSTAL template uses --mem-per-cpu=5G with
    ntasks=32 (=160GB total). The bump must PRESERVE the per-cpu form (5->7G per
    cpu); rewriting it to a bare total --mem=7GB silently cuts the allocation
    ~23x and guarantees a worse OOM. Drives the real memory_handler (not mocked).
    """
    import re
    script = tmp_path / "job.sh"
    script.write_text(
        "#!/bin/bash\n#SBATCH --ntasks=32\n#SBATCH --mem-per-cpu=5G\n#SBATCH -t 7-00:00:00\n")
    calc = {"calc_id": "C3", "material_id": "M1", "calc_type": "OPT",
            "input_file": str(original_d12), "work_dir": str(tmp_path),
            "job_script": str(script), "error_type": "memory_error"}
    res = engine.attempt_recovery(calc, create_record=False)
    bumped = res["fixed_job_script"]
    assert bumped is not None and bumped.exists()
    text = bumped.read_text()
    assert "--mem-per-cpu=7GB" in text                      # 5 * 1.5, form preserved
    # Must NOT have been collapsed into a bare total --mem= (the 23x-cut bug).
    assert not re.search(r"#SBATCH\s+--mem=", text)


def test_submit_to_slurm_sbatches_override_batch_file(tmp_path, monkeypatch):
    """B2 regression: a recovery-bumped, ready-made SLURM batch file (literal
    #SBATCH directives, mode 0o644 / NOT executable) passed via
    submit_script_override must be submitted with `sbatch` and its job id parsed
    -- NOT exec'd directly. Direct execution raised a PermissionError that was
    swallowed upstream, so recovery silently never resubmitted. Drives the REAL
    submit_to_slurm (the method the old test mocked away) against a fake sbatch.
    """
    import os
    import stat
    from mace.queue.manager import EnhancedCrystalQueueManager

    work = tmp_path / "work"
    work.mkdir()
    inp = work / "job.d12"
    inp.write_text("dummy\n")

    # Exactly what the recovery handlers write: a self-contained batch file with
    # literal #SBATCH lines and NO generator-echo markers, left non-executable.
    bumped = work / "job_recovery_x.sh"
    bumped.write_text(
        "#!/bin/bash --login\n#SBATCH -J job\n#SBATCH --mem-per-cpu=7GB\nPcrystal\n")
    assert not (bumped.stat().st_mode & stat.S_IXUSR)       # proves +x is not required

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "sbatch"
    fake.write_text('#!/bin/bash\necho "Submitted batch job 778899"\n')
    fake.chmod(fake.stat().st_mode | stat.S_IRWXU)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    mgr = EnhancedCrystalQueueManager(
        d12_dir=str(work), db_path=str(tmp_path / "mgr.db"),
        enable_tracking=True, enable_error_recovery=False, organize_outputs=False)

    job_id = mgr.submit_to_slurm(inp, work, "OPT", submit_script_override=bumped)
    assert job_id == "778899"


def test_resubmit_submits_the_fix_not_the_original(engine, original_d12, tmp_path):
    """The core B1 fix: the resubmission must carry the FIXED input."""
    from mace.queue.manager import EnhancedCrystalQueueManager
    mgr = EnhancedCrystalQueueManager(
        d12_dir=str(tmp_path), db_path=str(tmp_path / "mgr.db"),
        enable_tracking=True, enable_error_recovery=False, organize_outputs=False)
    mgr.is_workflow_context = False  # force in-place (no copy into a workflow dir)

    captured = {}

    def fake_submit(input_file, work_dir, calc_type, submit_script_override=None):
        from pathlib import Path
        captured["scf"] = _scf_maxcycle(Path(input_file))
        captured["override"] = submit_script_override
        return "JOB1"

    mgr.submit_to_slurm = fake_submit

    calc = {"calc_id": "C1", "material_id": "M1", "calc_type": "OPT",
            "input_file": str(original_d12), "work_dir": str(tmp_path),
            "error_type": "convergence_error"}
    rec = engine.attempt_recovery(calc, create_record=False)
    assert mgr.resubmit_fixed_calculation(
        calc, fixed_input=rec["fixed_input_file"], fixed_job_script=rec["fixed_job_script"])
    assert captured["scf"] == 1100          # the fix (was 100)


def test_resubmit_without_fix_falls_back_to_recorded_input(tmp_path, original_d12):
    """Regression: with no fix supplied, the recorded original is resubmitted."""
    from pathlib import Path
    from mace.queue.manager import EnhancedCrystalQueueManager
    mgr = EnhancedCrystalQueueManager(
        d12_dir=str(tmp_path), db_path=str(tmp_path / "mgr.db"),
        enable_tracking=True, enable_error_recovery=False, organize_outputs=False)
    mgr.is_workflow_context = False
    captured = {}

    def fake_submit(input_file, work_dir, calc_type, submit_script_override=None):
        captured["scf"] = _scf_maxcycle(Path(input_file))
        return "JOB2"

    mgr.submit_to_slurm = fake_submit
    calc = {"calc_id": "C1", "material_id": "M1", "calc_type": "OPT",
            "input_file": str(original_d12), "work_dir": str(tmp_path)}
    assert mgr.resubmit_fixed_calculation(calc)
    assert captured["scf"] == 100           # original, unbumped
