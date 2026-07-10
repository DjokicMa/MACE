"""Regression tests for `mace submit --track` (opt-in DB tracking).

CLI-level tests run mace_cli as a subprocess (no test/ corpus needed). The
mechanism-level test drives the tracking manager the flag wires up, with SLURM
mocked, against a real d12 (skips if the corpus is absent).
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT, find_data

MACE_CLI = REPO_ROOT / "mace_cli"
ENV = {**os.environ, "MACE_NO_BANNER": "1"}


def _run(args, cwd):
    return subprocess.run([sys.executable, str(MACE_CLI), *args],
                          cwd=str(cwd), env=ENV, capture_output=True, text=True, timeout=120)


def test_submit_help_lists_track():
    r = _run(["submit", "--help"], REPO_ROOT)
    assert "--track" in r.stdout


def test_track_with_nosubmit_is_rejected(tmp_path):
    (tmp_path / "x.d12").write_text("dummy\n")
    r = _run(["submit", "--track", "--nosubmit", str(tmp_path / "x.d12")], tmp_path)
    assert r.returncode != 0
    assert "nosubmit" in (r.stdout + r.stderr).lower()
    # The guard must fire BEFORE any DB is opened.
    assert not (tmp_path / "materials.db").exists()


def test_bare_submit_creates_no_database(tmp_path):
    """Default (no --track) submit must not open/create materials.db."""
    r = _run(["submit", "."], tmp_path)  # empty dir -> "no files", exits
    assert not (tmp_path / "materials.db").exists()


def test_tracked_submission_records_calc_in_place(monkeypatch, tmp_path):
    """The manager path the flag uses records a calc, in-place, and submits the
    original file (SLURM mocked)."""
    import shutil
    from mace.queue.manager import EnhancedCrystalQueueManager

    real = find_data("OPT/1LiFSI-3EMS-conf4*opt_HSESOL3C_optimized.d12")
    d12 = tmp_path / "job.d12"
    shutil.copy2(real, d12)
    monkeypatch.chdir(tmp_path)

    mgr = EnhancedCrystalQueueManager(
        d12_dir=str(tmp_path), db_path=str(tmp_path / "materials.db"),
        enable_tracking=True, organize_outputs=False)
    mgr.is_workflow_context = False
    captured = {}

    def fake_submit(input_file, work_dir, calc_type, submit_script_override=None):
        captured["input"] = Path(input_file)
        return "JOB1"

    mgr.submit_to_slurm = fake_submit

    calc_id = mgr.submit_calculation(Path(d12))
    assert calc_id
    calc = mgr.db.get_calculation(calc_id)
    assert calc["status"] == "submitted"
    # in-place: the recorded + submitted input is the original file, not a copy
    assert Path(calc["input_file"]).name == "job.d12"
    assert captured["input"].name == "job.d12"


def test_recovered_resubmission_records_override_script(monkeypatch, tmp_path):
    """When a recovery handler resubmits with a bumped *_recovery_N.sh via
    job_script_override, the DB must record THAT script as job_script — not the
    original generated one. Recording the original made the next memory/timeout
    bump restart from the original resources, so escalation plateaued after a
    single bump instead of compounding."""
    import shutil
    from mace.queue.manager import EnhancedCrystalQueueManager

    real = find_data("OPT/1LiFSI-3EMS-conf4*opt_HSESOL3C_optimized.d12")
    d12 = tmp_path / "job.d12"
    shutil.copy2(real, d12)
    monkeypatch.chdir(tmp_path)

    bumped = tmp_path / "job_recovery_1.sh"
    bumped.write_text("#!/bin/bash\n#SBATCH --mem=200G\n#SBATCH -t 7-00:00:00\n")

    mgr = EnhancedCrystalQueueManager(
        d12_dir=str(tmp_path), db_path=str(tmp_path / "materials.db"),
        enable_tracking=True, organize_outputs=False)
    mgr.is_workflow_context = False
    mgr.submit_to_slurm = (
        lambda input_file, work_dir, calc_type, submit_script_override=None: "JOB2")

    calc_id = mgr.submit_calculation(Path(d12), job_script_override=bumped)
    assert calc_id
    calc = mgr.db.get_calculation(calc_id)
    assert calc["job_script"] == str(bumped)


def test_lock_dir_anchored_to_database(tmp_path):
    """Completion callbacks run in each job's own cwd but share one
    materials.db — the queue lock must live beside the DB, not the cwd:
    per-cwd lock dirs gave concurrent callbacks no mutual exclusion and
    near-simultaneous completions duplicated follow-up calculations."""
    from mace.queue.manager import EnhancedCrystalQueueManager
    job_dir = tmp_path / "wf" / "step_002_SP" / "mat_sp"
    job_dir.mkdir(parents=True)
    db = tmp_path / "materials.db"
    mgr = EnhancedCrystalQueueManager(
        d12_dir=str(job_dir), db_path=str(db), enable_tracking=True)
    if mgr.lock_manager is None:
        import pytest
        pytest.skip("queue locking unavailable in this environment")
    assert Path(str(mgr.lock_manager.lock_dir)).resolve() == (tmp_path / ".queue_locks").resolve()


def test_completion_callback_classifies_its_own_job(monkeypatch, tmp_path):
    """The completion callback is the LAST line of the job script itself, so
    squeue still lists the invoking job as RUNNING when the callback checks it.
    Real-world (phase-1 smoke, job 12082006): a lone tracked manual job
    finished, its own callback saw itself RUNNING, left the record 'running'
    and never generated the SP — with no later job's callback to sweep it up,
    the chain stalled forever. The calc belonging to $SLURM_JOB_ID must be
    classified from its output file, not from queue state."""
    import shutil
    import mace.queue.manager as qm
    from mace.queue.manager import EnhancedCrystalQueueManager

    real_d12 = find_data("OPT/1LiFSI-3EMS-conf4*opt_HSESOL3C_optimized.d12")
    real_out = find_data("OPT/*.out", must_contain="OPT END")
    shutil.copy2(real_d12, tmp_path / "job.d12")
    shutil.copy2(real_out, tmp_path / "job.out")
    monkeypatch.chdir(tmp_path)

    mgr = EnhancedCrystalQueueManager(
        d12_dir=str(tmp_path), db_path=str(tmp_path / "materials.db"),
        enable_tracking=True, organize_outputs=False)
    mgr.is_workflow_context = False
    mgr.submit_to_slurm = (
        lambda input_file, work_dir, calc_type, submit_script_override=None: "777001")
    calc_id = mgr.submit_calculation(tmp_path / "job.d12")
    assert mgr.db.get_calculation(calc_id)["slurm_job_id"] == "777001"

    # squeue reports the job as still RUNNING (it is: we're inside it)
    def fake_squeue(cmd, capture_output=True, text=True, **kw):
        class R:
            returncode = 0
            stdout = "JOBID,STATE,START\n777001,RUNNING,2026-07-08T22:44:00\n"
            stderr = ""
        return R()

    monkeypatch.setattr(qm.subprocess, "run", fake_squeue)
    monkeypatch.setenv("SLURM_JOB_ID", "777001")
    handled = []
    mgr.handle_completed_calculation = handled.append

    mgr.check_queue_status()

    calc = mgr.db.get_calculation(calc_id)
    assert calc["status"] == "completed", (
        f"own job left '{calc['status']}': callback trusted squeue over output")
    assert handled == [calc_id]


def test_plan_next_does_not_resubmit_engine_submitted_calcs(monkeypatch, tmp_path):
    """execute_workflow_step submits what it generates; the manager's
    auto-submit loop then submitted the SAME calc again (phase-1 smoke: SP
    jobs 12083826 + 12083827 from one OPT), re-running the raw script
    generator in place and wiping the workflow context exports — the
    duplicate's callbacks then opened a fresh cwd-local DB and fanned out
    duplicate BAND/DOSS. Calcs that already carry a slurm_job_id must be
    left alone; ones the engine failed to submit are still picked up."""
    import mace.queue.manager as qm
    from mace.queue.manager import EnhancedCrystalQueueManager

    mgr = EnhancedCrystalQueueManager(
        d12_dir=str(tmp_path), db_path=str(tmp_path / "materials.db"),
        enable_tracking=True, organize_outputs=False)
    mgr.is_workflow_context = False
    mgr.auto_submit_followups = True

    mgr.db.create_material(material_id="1_dia", formula="C")
    d12 = tmp_path / "1_dia_sp.d12"
    d12.write_text("dummy\n")
    submitted_id = mgr.db.create_calculation(
        material_id="1_dia", calc_type="SP",
        input_file=str(d12), work_dir=str(tmp_path))
    mgr.db.update_calculation_status(submitted_id, "submitted",
                                     slurm_job_id="12083826")
    unsubmitted_id = mgr.db.create_calculation(
        material_id="1_dia", calc_type="BAND",
        input_file=str(d12), work_dir=str(tmp_path))

    class _FakeEngine:
        def __init__(self, *a, **k):
            pass

        def execute_workflow_step(self, material_id, calc_id):
            return [submitted_id, unsubmitted_id]

    import mace.workflow.engine as eng_mod
    monkeypatch.setattr(eng_mod, "WorkflowEngine", _FakeEngine)

    resubmitted = []
    mgr.submit_to_slurm = (
        lambda input_file, work_dir, calc_type, submit_script_override=None:
        resubmitted.append(calc_type) or "999999")

    mgr.plan_next_calculation("1_dia", "opt_1")

    assert "SP" not in resubmitted, "already-submitted calc was re-submitted"
    assert resubmitted == ["BAND"], "engine-unsubmitted calc must still be picked up"
    calc = mgr.db.get_calculation(submitted_id)
    assert calc["slurm_job_id"] == "12083826", "duplicate submission overwrote the real job id"


def _mgr_with_tracked_job(monkeypatch, tmp_path, job_id="777001"):
    import shutil
    from mace.queue.manager import EnhancedCrystalQueueManager

    real = find_data("OPT/1LiFSI-3EMS-conf4*opt_HSESOL3C_optimized.d12")
    shutil.copy2(real, tmp_path / "job.d12")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    mgr = EnhancedCrystalQueueManager(
        d12_dir=str(tmp_path), db_path=str(tmp_path / "materials.db"),
        enable_tracking=True, organize_outputs=False)
    mgr.is_workflow_context = False
    mgr.submit_to_slurm = (
        lambda input_file, work_dir, calc_type, submit_script_override=None: job_id)
    calc_id = mgr.submit_calculation(tmp_path / "job.d12")
    return mgr, calc_id


def test_empty_squeue_blip_does_not_fail_running_jobs(monkeypatch, tmp_path):
    """Real incident (phase-3 QA, 2026-07-09): gateway squeue transiently
    returned an EMPTY list while 4 OPTs were RUNNING (sacct proved it); a
    callback in that window classified every in-flight job from its
    half-written output and marked them all failed. A job missing from
    squeue must be confirmed terminal via sacct before classification."""
    import mace.queue.manager as qm
    mgr, calc_id = _mgr_with_tracked_job(monkeypatch, tmp_path)

    def fake_run(cmd, capture_output=True, text=True, **kw):
        class R:
            returncode = 0
            stderr = ""
        r = R()
        r.stdout = "JOBID,STATE,START\n" if cmd[0] == "squeue" else "   RUNNING \n"
        return r

    monkeypatch.setattr(qm.subprocess, "run", fake_run)
    classified = []
    mgr.check_completed_or_failed_job = classified.append

    mgr.check_queue_status()

    calc = mgr.db.get_calculation(calc_id)
    assert calc["status"] == "running", (
        f"squeue blip marked job '{calc['status']}' though sacct says RUNNING")
    assert classified == []


def test_sacct_terminal_state_still_classified_by_output(monkeypatch, tmp_path):
    """When sacct confirms the job left the queue, output-based
    classification must still run (previous behavior preserved)."""
    import mace.queue.manager as qm
    mgr, calc_id = _mgr_with_tracked_job(monkeypatch, tmp_path)

    def fake_run(cmd, capture_output=True, text=True, **kw):
        class R:
            returncode = 0
            stderr = ""
        r = R()
        r.stdout = "JOBID,STATE,START\n" if cmd[0] == "squeue" else " CANCELLED by 12345 \n"
        return r

    monkeypatch.setattr(qm.subprocess, "run", fake_run)
    classified = []
    mgr.check_completed_or_failed_job = classified.append

    mgr.check_queue_status()

    assert len(classified) == 1 and classified[0]["calc_id"] == calc_id


def test_manual_d3_submissions_get_correct_calc_type(tmp_path):
    """Real incident (phase-5 QA): `mace submit 1_dia_transport.d3` recorded
    the calc as SP — no filename token matched and the content check knew
    only OPTGEOM/FREQCALC — so its completion callback would have fanned out
    BAND/DOSS from a BOLTZTRA run. Deck contents below mirror the real
    generated files."""
    from mace.queue.manager import EnhancedCrystalQueueManager

    mgr = EnhancedCrystalQueueManager(
        d12_dir=str(tmp_path), db_path=str(tmp_path / "materials.db"),
        enable_tracking=False, organize_outputs=False)

    transport = tmp_path / "1_dia_transport.d3"
    transport.write_text(
        "BOLTZTRA\nTRANGE\n100 800 50\nMURANGE\n-5.35 -1.35 0.01\n"
        "TDFRANGE\n-5.0 5.0 0.01\nEND\n")
    cp = tmp_path / "1_dia_charge+potential.d3"
    cp.write_text("ECH3\n100\nPOT3\n100\n5\nEND\n")
    # content-only fallbacks (no type token in the name)
    anon_transport = tmp_path / "mystery.d3"
    anon_transport.write_text("BOLTZTRA\nTRANGE\n100 800 50\nEND\n")

    assert mgr.determine_calc_type_from_file(transport) == "TRANSPORT"
    assert mgr.determine_calc_type_from_file(cp) == "CHARGE+POTENTIAL"
    assert mgr.determine_calc_type_from_file(anon_transport) == "TRANSPORT"
