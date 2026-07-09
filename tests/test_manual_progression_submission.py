"""Manual (bare-directory) tracked progression must be able to SUBMIT.

Real-world (phase-1 smoke, 2026-07-08): a lone `mace submit --track` OPT
completed and generated its SP deck, but submission died with
"No SLURM template found for SP" — _create_slurm_script_for_calculation only
knew workflow_scripts/ templates and a base-dir submitcrystal23.sh copy,
neither of which exists in a bare manual directory. It must fall back to the
same generators `mace submit` itself uses (mace/submission/).

Companion hazard: the injected MACE_CONTEXT_DIR pointed at a nonexistent
.mace_context_<wfid>, so the .sh runtime check failed and nested follow-up
callbacks fell back to a cwd-relative DB — a fresh empty database in the
follow-up's own directory, which then saw its own d12 as never-submitted and
resubmitted itself (the manual-run resubmission storm). The exports must
point at the directory holding the engine's actual database.
"""
from pathlib import Path

import pytest

from mace.workflow.engine import WorkflowEngine


class _DB:
    def __init__(self, db_path):
        self.db_path = db_path


@pytest.fixture
def engine(tmp_path, monkeypatch):
    for var in ("MACE_CONTEXT_DIR", "MACE_WORKFLOW_ID", "MACE_ISOLATION_MODE"):
        monkeypatch.delenv(var, raising=False)
    eng = WorkflowEngine.__new__(WorkflowEngine)
    eng.base_work_dir = tmp_path
    db_file = tmp_path / "materials.db"
    db_file.touch()
    eng.db = _DB(db_file)
    return eng


def _calc_dir(tmp_path, step, name):
    d = tmp_path / "workflow_outputs" / "workflow_test" / step / name
    d.mkdir(parents=True)
    return d


def test_bare_dir_sp_script_falls_back_to_mace_submit_generator(engine, tmp_path):
    calc_dir = _calc_dir(tmp_path, "step_002_SP", "1_dia_sp")
    (calc_dir / "1_dia_sp.d12").write_text("dummy\n")

    script = engine._create_slurm_script_for_calculation(
        calc_dir, "1_dia_sp", "SP", 2, "workflow_test")

    assert script == calc_dir / "1_dia_sp.sh" and script.exists()
    content = script.read_text()
    assert "export JOB=1_dia_sp" in content
    # the follow-up must carry the completion callback or the chain dies here
    assert "--callback-mode completion" in content
    # d12 job: crystal generator, not the d3/property one
    assert "$JOB.d12" in content


def test_bare_dir_band_script_uses_property_generator(engine, tmp_path):
    calc_dir = _calc_dir(tmp_path, "step_003_BAND", "1_dia_band")
    (calc_dir / "1_dia_band.d3").write_text("dummy\n")

    script = engine._create_slurm_script_for_calculation(
        calc_dir, "1_dia_band", "BAND", 3, "workflow_test")

    assert script.exists()
    content = script.read_text()
    assert "export JOB=1_dia_band" in content
    assert "$JOB.d3" in content


def test_context_dir_export_points_at_real_database_dir(engine, tmp_path):
    """No .mace_context_<wfid> exists in manual runs: the exports must point
    at the engine DB's directory so nested callbacks hit the shared DB
    instead of creating a fresh one in their own cwd."""
    calc_dir = _calc_dir(tmp_path, "step_002_SP", "1_dia_sp")
    (calc_dir / "1_dia_sp.d12").write_text("dummy\n")

    script = engine._create_slurm_script_for_calculation(
        calc_dir, "1_dia_sp", "SP", 2, "workflow_test")

    content = script.read_text()
    expected = str(tmp_path.resolve())
    assert f'export MACE_CONTEXT_DIR="{expected}"' in content, (
        "MACE_CONTEXT_DIR must be the directory holding materials.db")
    assert f'{expected}/.mace_context_' not in content


def test_existing_mace_context_dir_still_preferred(engine, tmp_path):
    """When the isolated-context dir DOES exist (real workflow runs), keep
    pointing at it — don't-break-what-works."""
    ctx = tmp_path / ".mace_context_workflow_test"
    ctx.mkdir()
    calc_dir = _calc_dir(tmp_path, "step_002_SP", "2_dia_sp")
    (calc_dir / "2_dia_sp.d12").write_text("dummy\n")

    script = engine._create_slurm_script_for_calculation(
        calc_dir, "2_dia_sp", "SP", 2, "workflow_test")

    assert f'export MACE_CONTEXT_DIR="{ctx.resolve()}"' in script.read_text() or \
           f'export MACE_CONTEXT_DIR="{ctx}"' in script.read_text()
