"""Error classification against REAL failed outputs (phase-3 QA, 2026-07-09).

Both failures were classified 'unknown_error' -> "not recoverable":
- 1_dia B3LYP-D3 SCF blow-up ends with "ERROR **** ZERO **** FERMI ENERGY
  NOT IN INTERVAL" — an SCF convergence failure the recovery engine has a
  handler for (FMIXING/MAXCYCLE adjustments), but no pattern matched it.
- 3_dia3 pre-fix conversion died with "DISTANCE BETWEEN ATOMS 13 54 TOO
  SMALL" + "ERROR **** NEIGHB ****" — a geometry error (correctly
  unrecoverable), but the patterns only knew "ATOMS TOO CLOSE" phrasing.
"""
import pytest

from conftest import find_data


@pytest.fixture
def mgr(tmp_path):
    from mace.queue.manager import EnhancedCrystalQueueManager
    return EnhancedCrystalQueueManager(
        d12_dir=str(tmp_path), db_path=str(tmp_path / "materials.db"),
        enable_tracking=True, organize_outputs=False)


def _classify(mgr, out_path):
    return mgr.analyze_calculation_error({"output_file": str(out_path)})


def test_fermi_not_in_interval_is_convergence_error(mgr):
    out = find_data("FAILED_QA/1_dia_b3lyp_fermi_not_in_interval.out")
    error_type, msg = _classify(mgr, out)
    assert error_type == "convergence_error", (error_type, msg)


def test_neighb_too_close_is_geometry_error(mgr):
    out = find_data("FAILED_QA/3_dia3_neighb_atoms_too_close.out")
    error_type, msg = _classify(mgr, out)
    assert error_type == "geometry_error", (error_type, msg)


def test_geometry_error_correctly_not_auto_recovered(mgr):
    """A broken input deck must not be resubmitted with bumped resources."""
    ok = mgr.attempt_error_recovery(
        {"calc_id": "x"}, "geometry_error", "Detected: **** NEIGHB ****")
    assert ok is False


def test_convergence_error_passes_recoverable_gate(mgr, monkeypatch):
    """convergence_error must reach the recovery engine (not be rejected at
    the recoverable-set gate like unknown_error was)."""
    reached = {}

    class _Engine:
        def attempt_recovery(self, calc, create_record=False):
            reached["error_type"] = calc.get("error_type")
            return None  # recovery itself declines; the gate is the test

    monkeypatch.setattr(mgr, "get_recovery_attempt_count", lambda cid: 0)
    mgr.error_recovery_engine = _Engine()
    mgr.attempt_error_recovery(
        {"calc_id": "x", "material_id": "m"}, "convergence_error",
        "Detected: FERMI ENERGY NOT IN INTERVAL")
    assert reached.get("error_type") == "convergence_error"


def test_memory_handler_survives_none_job_script(tmp_path):
    """Real incident (phase-3 QA, 3,4^2T7_CA OPT OOM): workflow-created calc
    records carry job_script=None (key PRESENT, so dict.get's default never
    applies) — Path(None) raised TypeError inside memory_handler and the
    recovery chain died before ever bumping resources. The submit generators
    always write <input stem>.sh beside the input; the handler must fall
    back to it and produce the bumped recovery script."""
    from pathlib import Path
    from mace.recovery.recovery import ErrorRecoveryEngine

    eng = ErrorRecoveryEngine.__new__(ErrorRecoveryEngine)
    d12 = tmp_path / "mat.d12"
    d12.write_text("x\n")
    (tmp_path / "mat.sh").write_text(
        "#!/bin/bash\n#SBATCH --mem-per-cpu=5G\n#SBATCH -t 1-00:00:00\n")
    calc = {"calc_id": "c1", "job_script": None, "input_file": str(d12)}

    result = eng.memory_handler(calc, {"memory_factor": 1.5, "max_memory": "200GB"})

    assert result is not None, "memory fix must succeed via the <input>.sh fallback"
    bumped = list(tmp_path.glob("mat_recovery_*.sh"))
    assert bumped, "no bumped recovery script written"
    assert "--mem-per-cpu=7GB" in bumped[0].read_text()


def test_sigkill_oom_is_memory_error(mgr):
    """Real incident (phase-3 QA, 3_dia3 OPT job 12091685): the OOM killer
    SIGKILLs MPI ranks ("KILLED BY SIGNAL: 9") while SLURM records the job
    COMPLETED — classified unknown_error and left unrecovered even though a
    memory bump is exactly the fix."""
    out = find_data("FAILED_QA/3_dia3_opt_killed_signal9_oom.out")
    error_type, msg = _classify(mgr, out)
    assert error_type == "memory_error", (error_type, msg)
