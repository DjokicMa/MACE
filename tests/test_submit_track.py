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
