"""Regression tests for SLURM job-id capture in the untracked submit path.

submission/crystal.py and submission/properties.py used os.system("sbatch ...")
which discards sbatch's stdout, so the job id of an untracked `mace submit` was
lost (not trackable/recoverable). They now use subprocess.run and parse
"Submitted batch job N" (mirroring EnhancedCrystalQueueManager.submit_to_slurm).
"""
import os
import sys
import stat

import pytest

from conftest import REPO_ROOT
from mace.submission.crystal import extract_slurm_job_id as crystal_parse
from mace.submission.properties import extract_slurm_job_id as props_parse


@pytest.mark.parametrize("parse", [crystal_parse, props_parse])
def test_extract_slurm_job_id(parse):
    assert parse("Submitted batch job 1234567") == "1234567"
    assert parse("Submitted batch job 42\n") == "42"
    assert parse("sbatch: error: invalid partition") is None
    assert parse("") is None
    assert parse(None) is None


def test_no_os_system_in_submit_paths():
    """Guard against reintroducing os.system on the submission paths."""
    for rel in ("mace/submission/crystal.py", "mace/submission/properties.py"):
        src = (REPO_ROOT / rel).read_text()
        assert "os.system" not in src, f"{rel} reintroduced os.system"
        assert "subprocess.run" in src, f"{rel} should use subprocess.run"


def _make_fake_sbatch(bin_dir, job_id="999"):
    """Create a fake `sbatch` on PATH that mimics SLURM's success output."""
    fake = bin_dir / "sbatch"
    fake.write_text("#!/bin/bash\necho \"Submitted batch job %s\"\n" % job_id)
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC | stat.S_IRWXU)
    return fake


def test_submit_main_captures_and_summarizes_job_id(tmp_path, monkeypatch, capsys):
    """End-to-end: run crystal.main() against a fake sbatch and assert the job id
    is captured per-file and listed in the final summary."""
    from mace.submission import crystal

    # A .d12 whose name drives the generated .sh (generator only uses the name).
    work = tmp_path / "work"
    work.mkdir()
    (work / "job1.d12").write_text("dummy\n")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _make_fake_sbatch(bin_dir, job_id="555111")
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    monkeypatch.setattr(sys, "argv", ["submitcrystal23.py", str(work)])
    cwd = os.getcwd()
    try:
        crystal.main()
    finally:
        os.chdir(cwd)

    out = capsys.readouterr().out
    # The generator must have produced the .sh and the fake sbatch returned the id.
    assert (work / "job1.sh").exists()
    assert "job 555111" in out
    assert "Submitted 1 job(s): 555111" in out
