"""The HSE-type odd-electron crash is not routed to the memory recovery.

Measured on HPCC (CRYSTAL/23, 2026-10-01/02): HSEsol-3c and HSE06 on an
odd-electron open-shell Li+ molecule (0D) and a neutral odd-electron 3D box.
Each SCF blows up ("DIIS TEST: NaN" at cycle 1-3) and every rank dies:
"KILLED BY SIGNAL: 11 (Segmentation fault)", 6 or 9. The fixtures
(tests/data/charged_runs/odd_*) are synthetic CRYSTAL23-format excerpts of
those runs: the hand-built Li+ FSI- EC molecule of the other charged-run
fixtures (137 electrons with CHEMOD on Li; 141 in the neutral 2-Li box),
ending as the measured runs ended. The queue manager read those lines as
memory_error and resubmitted with more memory, which cannot help (the same
decks with PBE0/PBEsol0 converge). The run is now reported as its own,
non-recoverable error; the functional is never changed.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

from conftest import find_data

DATA = Path(__file__).resolve().parent / "data" / "charged_runs"
CRASHES = ["odd_hsesol3c_0d_crash.out", "odd_hse06_0d_crash.out", "odd_hsesol3c_3d_crash.out"]


@pytest.fixture
def mgr(tmp_path):
    from mace.queue.manager import EnhancedCrystalQueueManager
    return EnhancedCrystalQueueManager(
        d12_dir=str(tmp_path), db_path=str(tmp_path / "materials.db"),
        enable_tracking=True, organize_outputs=False)


@pytest.mark.parametrize("name", CRASHES)
def test_the_fixtures_are_the_measured_crash(name):
    text = (DATA / name).read_text()
    assert "DIIS TEST:         NaN" in text and "KILLED BY SIGNAL" in text
    assert "UNRESTRICTED OPEN SHELL" in text and "TOTAL CPU TIME" not in text


@pytest.mark.parametrize("name", CRASHES)
def test_queue_manager_does_not_call_it_a_memory_error(mgr, name):
    error_type, msg = mgr.analyze_calculation_error({"output_file": str(DATA / name)})
    assert error_type == "screened_hybrid_odd_error", (error_type, msg)
    assert "PBE0" in msg and "keeps the functional" in msg


def test_it_is_not_auto_recovered(mgr):
    assert mgr.attempt_error_recovery({"calc_id": "x"}, "screened_hybrid_odd_error", "m") is False


@pytest.mark.parametrize("name", CRASHES)
def test_error_detector_does_not_suggest_more_memory(tmp_path, name):
    from mace.recovery.detector import CrystalErrorDetector
    det = CrystalErrorDetector(str(tmp_path), str(tmp_path / "d.db"), enable_tracking=False)
    res = det.analyze_output_file(DATA / name)
    assert res["error_type"] == "screened_hybrid_odd_error", res["error_type"]
    assert res["recoverable"] is False
    assert not any(a["action"] == "increase_memory" for a in det.suggest_recovery_actions(res))


def test_a_real_oom_kill_is_still_a_memory_error(mgr):
    """Closed-shell B3LYP OPT killed by the OOM killer (job 12091685)."""
    out = find_data("FAILED_QA/3_dia3_opt_killed_signal9_oom.out")
    assert mgr.analyze_calculation_error({"output_file": str(out)})[0] == "memory_error"


def test_status_sweep_does_not_resubmit_it(monkeypatch, tmp_path):
    """End to end through the status sweep: the job left the queue, sacct
    says FAILED, the .out is the crash - no resubmission, error recorded."""
    import mace.queue.manager as qm
    from mace.queue.manager import EnhancedCrystalQueueManager

    shutil.copy(DATA / "odd_hsesol3c_0d_crash.d12", tmp_path / "li_odd.d12")
    deck = (tmp_path / "li_odd.d12").read_text()
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    mgr = EnhancedCrystalQueueManager(d12_dir=str(tmp_path), db_path=str(tmp_path / "m.db"),
                                      enable_tracking=True, organize_outputs=False)
    mgr.is_workflow_context = False
    submitted = []
    mgr.submit_to_slurm = (lambda input_file, work_dir, calc_type, submit_script_override=None:
                           submitted.append(input_file) or str(777001 + len(submitted) - 1))
    calc_id = mgr.submit_calculation(tmp_path / "li_odd.d12")
    (tmp_path / "li_odd.sh").write_text("#!/bin/bash\n#SBATCH --mem-per-cpu=5G\n#SBATCH -t 1:00:00\n")
    shutil.copy(DATA / "odd_hsesol3c_0d_crash.out", tmp_path / "li_odd.out")

    def fake_run(cmd, capture_output=True, text=True, **kw):
        out = "JOBID,STATE,START\n" if cmd[0] == "squeue" else "FAILED \n"
        return subprocess.CompletedProcess(cmd, 0, out, "")
    monkeypatch.setattr(qm.subprocess, "run", fake_run)

    mgr.check_queue_status()

    calc = mgr.db.get_calculation(calc_id)
    assert calc["status"] == "failed", calc
    assert calc["error_type"] == "screened_hybrid_odd_error"
    assert len(submitted) == 1
    assert (tmp_path / "li_odd.d12").read_text() == deck
    assert not list(tmp_path.glob("*recovery*.sh"))


# ------------------------------------------- squeue COMPLETED is not enough

def _sweep_with_squeue_completed(monkeypatch, tmp_path, out_file):
    """squeue still lists the job as COMPLETED; the .out says what happened."""
    import mace.queue.manager as qm
    from mace.queue.manager import EnhancedCrystalQueueManager

    shutil.copy(DATA / "odd_hsesol3c_0d_crash.d12", tmp_path / "li_odd.d12")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    mgr = EnhancedCrystalQueueManager(d12_dir=str(tmp_path), db_path=str(tmp_path / "m.db"),
                                      enable_tracking=True, organize_outputs=False)
    mgr.is_workflow_context = False
    submitted, completed = [], []
    mgr.submit_to_slurm = (lambda input_file, work_dir, calc_type, submit_script_override=None:
                           submitted.append(input_file) or str(777001 + len(submitted) - 1))
    mgr.handle_completed_calculation = completed.append
    calc_id = mgr.submit_calculation(tmp_path / "li_odd.d12")
    (tmp_path / "li_odd.sh").write_text("#!/bin/bash\n#SBATCH --mem-per-cpu=5G\n#SBATCH -t 1:00:00\n")
    shutil.copy(out_file, tmp_path / "li_odd.out")

    def fake_run(cmd, capture_output=True, text=True, **kw):
        if cmd[0] == "squeue":
            return subprocess.CompletedProcess(cmd, 0, "JOBID,STATE,START\n777001,COMPLETED,N/A\n", "")
        return subprocess.CompletedProcess(cmd, 0, "COMPLETED \n", "")
    monkeypatch.setattr(qm.subprocess, "run", fake_run)
    mgr.check_queue_status()
    return mgr.db.get_calculation(calc_id), completed, submitted


def test_squeue_completed_with_a_crashed_out_is_failed(monkeypatch, tmp_path):
    """SLURM records a job whose ranks were killed as COMPLETED (measured:
    job 12091685); while squeue still lists it, the record was marked
    completed and its properties extracted without the .out being read."""
    calc, completed, submitted = _sweep_with_squeue_completed(
        monkeypatch, tmp_path, DATA / "odd_hsesol3c_0d_crash.out")
    assert calc["status"] == "failed", calc
    assert calc["error_type"] == "screened_hybrid_odd_error"
    assert completed == [] and len(submitted) == 1


def test_squeue_completed_with_a_finished_out_is_completed(monkeypatch, tmp_path):
    calc, completed, _ = _sweep_with_squeue_completed(
        monkeypatch, tmp_path, DATA / "li_fsi_ec_neutral.out")
    assert calc["status"] == "completed", calc
    assert completed == [calc["calc_id"]]
