"""An incomplete own template copy never costs a step its planned resources.

An isolated workflow builds its steps from its own copy of the job-script
templates (``.mace_context_<id>/workflow_scripts/``). That copy could be left
incomplete (a copy that failed part way, e.g. over quota), and once the
folder existed the engine looked nowhere else: a step whose template was
missing got the generic script, losing the plan's cores, memory and walltime.
"""
import re
import shutil

import pytest

from mace.workflow import context as context_mod
from mace.workflow.context import freeze_shared_scripts, workflow_scripts_dir
from mace.workflow.engine import WorkflowEngine

WID = "workflow_20991231_000020"


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


def _sp_template(engine, tmp_path, ntasks):
    """A real SP job script, as a template with the plan's core count."""
    d = tmp_path / "gen" / "x_sp"
    d.mkdir(parents=True)
    (d / "x_sp.d12").write_text("dummy\n")
    text = engine._create_slurm_script_for_calculation(d, "x_sp", "SP", 2, None).read_text()
    text = re.sub(r"--ntasks=\d+", f"--ntasks={ntasks}", text)
    assert f"--ntasks={ntasks}" in text
    shutil.rmtree(tmp_path / "gen")
    return text.replace("export JOB=x_sp", "export JOB=$1")


def test_step_missing_from_own_copy_uses_the_shared_template(engine, tmp_path, capsys):
    shared = tmp_path / "workflow_scripts"
    shared.mkdir()
    (shared / "submitcrystal23_sp_2.sh").write_text(_sp_template(engine, tmp_path, 17))
    own = workflow_scripts_dir(tmp_path, WID)
    own.mkdir(parents=True)
    (own / "submitcrystal23_opt_1.sh").write_text("#!/bin/bash\n")   # the SP one is missing

    calc_dir = tmp_path / "workflow_outputs" / WID / "step_002_SP" / "m_sp"
    calc_dir.mkdir(parents=True)
    (calc_dir / "m_sp.d12").write_text("dummy\n")
    script = engine._create_slurm_script_for_calculation(calc_dir, "m_sp", "SP", 2, WID)
    assert "--ntasks=17" in script.read_text()
    assert "not in this workflow's own templates" in capsys.readouterr().out


def test_a_failed_copy_leaves_no_partial_own_folder(tmp_path, monkeypatch):
    shared = tmp_path / "workflow_scripts"
    shared.mkdir()
    for name in ("submitcrystal23_opt_1.sh", "submitcrystal23_sp_2.sh"):
        (shared / name).write_text("#!/bin/bash\n")
    old = tmp_path / ".mace_context_workflow_20991231_000001"
    old.mkdir()
    real_copy = shutil.copy2
    calls = []

    def copy_fails_on_second(src, dst, *a, **k):
        calls.append(src)
        if len(calls) == 2:
            raise OSError(122, "Disk quota exceeded")
        return real_copy(src, dst, *a, **k)

    monkeypatch.setattr(context_mod.shutil, "copy2", copy_fails_on_second)
    freeze_shared_scripts(tmp_path, WID)
    # no own folder at all: the workflow keeps using the shared templates
    assert not (old / "workflow_scripts").exists()
    assert [p.name for p in old.iterdir()] == []

    monkeypatch.setattr(context_mod.shutil, "copy2", real_copy)
    freeze_shared_scripts(tmp_path, WID)
    assert sorted(p.name for p in (old / "workflow_scripts").iterdir()) == [
        "submitcrystal23_opt_1.sh", "submitcrystal23_sp_2.sh"]
