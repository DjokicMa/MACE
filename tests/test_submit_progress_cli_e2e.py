"""`mace submit` through the real CLI, with SLURM stubbed out.

Isolated unit calls miss the wiring these tests cover: that `--progress`
reaches the plan builder, that the submitted calculation is stamped with the
plan's workflow_id (without which the completion callback finds no plan and
progression stops dead), and that a bare `mace submit --track` leaves no
workflow behind at all.
"""
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT

MACE_CLI = REPO_ROOT / "mace_cli"

D12 = """TESTGEOM
CRYSTAL
0 0 0
227
3.5
1
6 0.125 0.125 0.125
END
6 2
0 0 2 2. 1.
 8.0 1.0
 2.0 1.0
0 1 1 4. 1.
 1.0 1.0 1.0
99 0
END
DFT
B3LYP
END
SHRINK
8 8
END
"""


@pytest.fixture
def fake_slurm(tmp_path):
    """A PATH with sbatch/squeue/sacct stubs so nothing reaches a real queue."""
    bindir = tmp_path / "fakebin"
    bindir.mkdir()
    (bindir / "sbatch").write_text(
        "#!/bin/bash\necho 'Submitted batch job 999001'\n")
    (bindir / "squeue").write_text("#!/bin/bash\nexit 0\n")
    (bindir / "sacct").write_text("#!/bin/bash\nexit 0\n")
    (bindir / "scancel").write_text("#!/bin/bash\nexit 0\n")
    for f in bindir.iterdir():
        f.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = f"{bindir}:{env['PATH']}"
    env["MACE_NO_BANNER"] = "1"
    env.pop("MACE_WORKFLOW_ID", None)
    env.pop("MACE_PLANLESS_PROGRESSION", None)
    return env


def _run_submit(workdir, env, *args):
    return subprocess.run(
        [sys.executable, str(MACE_CLI), "--no-banner", "submit", *args],
        cwd=str(workdir), env=env, capture_output=True, text=True, timeout=300)


def _calc_rows(db_path):
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(
            "SELECT calc_type, status, settings_json FROM calculations")]


def test_bare_track_submit_creates_no_workflow(tmp_path, fake_slurm):
    """The reported bug, from the outside: a hand-run submission must leave a
    single calculation and no workflow directory behind."""
    deck = tmp_path / "mat_opt.d12"
    deck.write_text(D12)

    proc = _run_submit(tmp_path, fake_slurm, "--track", str(deck))
    assert proc.returncode == 0, proc.stdout + proc.stderr

    assert not (tmp_path / "workflow_configs").exists()
    assert not (tmp_path / "workflow_outputs").exists()

    rows = _calc_rows(tmp_path / "materials.db")
    assert len(rows) == 1, rows
    assert rows[0]["calc_type"] == "OPT"
    assert not json.loads(rows[0]["settings_json"] or "{}").get("workflow_id")


def test_progress_writes_plan_and_stamps_submission(tmp_path, fake_slurm):
    deck = tmp_path / "mat_opt.d12"
    deck.write_text(D12)

    proc = _run_submit(tmp_path, fake_slurm, "--progress", "full_electronic", str(deck))
    assert proc.returncode == 0, proc.stdout + proc.stderr

    plans = list((tmp_path / "workflow_configs").glob("workflow_plan_*.json"))
    assert len(plans) == 1, [p.name for p in plans]
    plan = json.loads(plans[0].read_text())
    workflow_id = plan["workflow_id"]
    assert plan["workflow_sequence"] == ["OPT", "SP", "BAND", "DOSS"]

    rows = _calc_rows(tmp_path / "materials.db")
    assert len(rows) == 1, rows
    settings = json.loads(rows[0]["settings_json"] or "{}")
    assert settings.get("workflow_id") == workflow_id, (
        "submission must carry the plan's id or the callback finds no plan")
    assert settings.get("workflow_step") == 1
    assert settings.get("workflow_calc_type") == "OPT"


def test_progress_records_mid_sequence_step_number(tmp_path, fake_slurm):
    """Submitting an SP built by hand from a finished OPT: the plan still
    describes the whole chain, but this deck enters it at step 2."""
    deck = tmp_path / "mat_sp.d12"
    deck.write_text(D12)

    proc = _run_submit(tmp_path, fake_slurm, "--progress", "full_electronic", str(deck))
    assert proc.returncode == 0, proc.stdout + proc.stderr

    rows = _calc_rows(tmp_path / "materials.db")
    settings = json.loads(rows[0]["settings_json"] or "{}")
    assert rows[0]["calc_type"] == "SP"
    assert settings.get("workflow_step") == 2, settings


def test_unknown_template_fails_before_submitting(tmp_path, fake_slurm):
    deck = tmp_path / "mat_opt.d12"
    deck.write_text(D12)

    proc = _run_submit(tmp_path, fake_slurm, "--progress", "not_a_template", str(deck))
    assert proc.returncode != 0
    assert "Unknown workflow template" in (proc.stdout + proc.stderr)
    assert not (tmp_path / "materials.db").exists(), "nothing may be submitted"
