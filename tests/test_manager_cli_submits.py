"""`mace manager --d12-dir DIR` keeps the queue fed.

HPCC QA (2026-10-03): with three decks, `mace manager --d12-dir . --organize
--max-jobs 5 --reserve 0` ran a completion callback ('Manual submission (no
workflow plan) ... no follow-up jobs submitted') and submitted nothing; only
`--callback-mode submit_new` submitted. The CLI forwarded no mode and the
manager's own default is the job-script callback mode. The manager was also
built twice ('Error recovery enabled' printed twice).
"""
import os
import sqlite3
import subprocess
import sys

import pytest
from pathlib import Path

from conftest import REPO_ROOT
from test_submit_progress_cli_e2e import OPT_D12

MACE_CLI = REPO_ROOT / "mace_cli"


@pytest.fixture
def env(tmp_path):
    bindir = tmp_path / "fakebin"
    bindir.mkdir()
    (bindir / "sbatch").write_text(
        "#!/bin/bash\nn=$(cat \"$0.count\" 2>/dev/null || echo 0); n=$((n+1)); "
        "echo $n > \"$0.count\"; echo \"Submitted batch job 77$n\"\n")
    (bindir / "squeue").write_text("#!/bin/bash\nexit 0\n")
    (bindir / "sacct").write_text("#!/bin/bash\nexit 0\n")
    for f in bindir.iterdir():
        f.chmod(0o755)
    e = dict(os.environ)
    e["PATH"] = f"{bindir}:{e['PATH']}"
    e["MACE_NO_BANNER"] = "1"
    for var in ("MACE_CONTEXT_DIR", "MACE_WORKFLOW_ID", "MACE_ISOLATION_MODE",
                "SLURM_JOB_ID", "MACE_HOME", "PYTHONPATH"):
        e.pop(var, None)
    return e


@pytest.mark.parametrize("organize", [False, True])
def test_manager_submits_the_decks_in_d12_dir(tmp_path, env, organize):
    for name in ("MgO", "CaO", "SrO"):
        (tmp_path / f"{name}_opt.d12").write_text(OPT_D12)
    args = ["--d12-dir", ".", "--max-jobs", "5", "--reserve", "0"]
    if organize:
        args.append("--organize")
    proc = subprocess.run(
        [sys.executable, str(MACE_CLI), "--no-banner", "manager", *args],
        cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=300)
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out

    with sqlite3.connect(tmp_path / "materials.db") as conn:
        rows = conn.execute(
            "SELECT status, slurm_job_id FROM calculations").fetchall()
    assert len(rows) == 3, out
    assert all(status == "submitted" and job for status, job in rows), rows
    assert out.count("Error recovery enabled") == 1, out


def test_organized_job_callback_uses_the_managers_database(tmp_path, env):
    """HPCC QA (2026-10-03): the --organize jobs' callbacks ran in
    opt/<material>/ and opened a new materials.db there, so the records in
    the manager's database stayed 'submitted' with no properties."""
    (tmp_path / "MgO_opt.d12").write_text(OPT_D12)
    proc = subprocess.run(
        [sys.executable, str(MACE_CLI), "--no-banner", "manager", "--d12-dir", ".",
         "--organize", "--max-jobs", "5", "--reserve", "0"],
        cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    with sqlite3.connect(tmp_path / "materials.db") as conn:
        (work_dir,) = conn.execute("SELECT work_dir FROM calculations").fetchone()
    work_dir = Path(work_dir)
    assert work_dir != tmp_path
    (script,) = work_dir.glob("*.sh")
    text = script.read_text()

    # Run the script's callback section as the job would, with python
    # replaced by a stub that prints the arguments it is given.
    callback = text[text.index("# First check if MACE_HOME is set"):]
    shim = tmp_path / "shim"
    shim.mkdir()
    (shim / "python").write_text("#!/bin/bash\necho \"ARGS: $*\"\n")
    (shim / "python").chmod(0o755)
    exports = "\n".join(l for l in text.splitlines()
                        if l.startswith(("export JOB=", "export MACE_DB_PATH=")))
    run_env = dict(env)
    run_env["PATH"] = f"{shim}:{env['PATH']}"
    run_env["MACE_HOME"] = str(REPO_ROOT)
    out = subprocess.run(["bash", "-c", f"DIR={work_dir}\n{exports}\n{callback}"],
                         cwd=str(work_dir), env=run_env, capture_output=True,
                         text=True, timeout=60).stdout
    assert f"--db-path {tmp_path / 'materials.db'}" in out, out
