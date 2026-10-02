"""A workflow_scripts/ template copied before 1.1.3 is rebuilt from the plan.

A workflow executed by 1.1.1/1.1.2 keeps the generator copies taken then
(tests/data/job_scripts/*_v1.1.1.sh are the 1.1.1 generators): no scratch
guard (an empty $SCRATCH gives "END OF DATA IN INPUT DECK"), no RESTART and
GUESSP staging. Every later step of that workflow was built from them, with
only a warning. The engine now rebuilds such a template the way execution
builds it - the current generator with the plan's resources, node exclusion
and custom directives - so the step gets what a fresh execution of the same
plan gives it. The old copy is kept as <name>.pre-1.1.3. Without a plan
entry for the template it is left alone and the warning stays.
"""
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from mace.workflow.engine import WorkflowEngine
from mace.workflow.executor import WorkflowExecutor

DATA = Path(__file__).resolve().parent / "data" / "job_scripts"
WF = "workflow_20260801_120000"


class _DB:
    def __init__(self, db_path):
        self.db_path = db_path


def _plan(tmp_path, script, name, calc_type):
    cfg = {"source_script": script, "step_specific_name": name,
           "resources": {"ntasks": 16, "nodes": 1, "walltime": "2-00:00:00",
                         "memory_per_cpu": "7G", "memory": "80G", "account": "general"},
           "customizations": [{"find": "#SBATCH --cpus-per-task=1", "replace": "#SBATCH --cpus-per-task=1\n#SBATCH --constraint=amr"}],
           "node_exclusion": "agg-[011-012]", "copy_to_workdir": True}
    plan = {"workflow_id": WF, "workflow_sequence": ["OPT", calc_type],
            "step_configurations": {f"{calc_type}_2": {"slurm_config": {"scripts": {script: cfg}}}}}
    (tmp_path / "workflow_configs").mkdir(exist_ok=True)
    (tmp_path / "workflow_configs" / f"workflow_plan_{WF.replace('workflow_', '')}.json").write_text(
        json.dumps(plan))
    return plan


def _executor(tmp_path):
    ex = WorkflowExecutor.__new__(WorkflowExecutor)
    ex.work_dir = tmp_path
    return ex


@pytest.fixture
def engine(tmp_path, monkeypatch):
    for var in ("MACE_CONTEXT_DIR", "MACE_WORKFLOW_ID", "MACE_ISOLATION_MODE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)
    eng = WorkflowEngine.__new__(WorkflowEngine)
    eng.base_work_dir = tmp_path
    (tmp_path / "materials.db").touch()
    eng.db = _DB(tmp_path / "materials.db")
    return eng


CASES = [("submitcrystal23.sh", "submitcrystal23_sp_2.sh", "SP", "d12"),
         ("submit_prop.sh", "submit_prop_band_2.sh", "BAND", "d3")]


@pytest.mark.parametrize("script,name,calc_type,ext", CASES)
def test_old_template_is_rebuilt_as_a_fresh_execution_writes_it(engine, tmp_path, capsys,
                                                                 script, name, calc_type, ext):
    plan = _plan(tmp_path, script, name, calc_type)
    # what executing this plan writes today ...
    _executor(tmp_path).recreate_workflow_scripts(json.loads(json.dumps(plan)))
    fresh = (tmp_path / "workflow_scripts" / name).read_text()
    assert "SCRATCH GUARD" in fresh
    # ... and what 1.1.1 wrote for it
    stub = SimpleNamespace(work_dir=tmp_path, isolation_mode="isolated")
    cfg = dict(plan["step_configurations"][f"{calc_type}_2"]["slurm_config"]["scripts"][script],
               workflow_id=WF)
    old = WorkflowExecutor.apply_script_customizations(
        stub, (DATA / f"{Path(script).stem}_v1.1.1.sh").read_text(), cfg)
    assert "SCRATCH GUARD" not in old
    (tmp_path / "workflow_scripts" / name).write_text(old)

    d = tmp_path / "workflow_outputs" / WF / f"step_002_{calc_type}" / "mat"
    d.mkdir(parents=True)
    (d / f"mat.{ext}").write_text("dummy\n")
    job = engine._create_slurm_script_for_calculation(d, "mat", calc_type, 2, WF)

    out = capsys.readouterr().out
    assert "older MACE" in out and "Rebuilt" in out
    assert (tmp_path / "workflow_scripts" / name).read_text() == fresh
    assert (tmp_path / "workflow_scripts" / f"{name}.pre-1.1.3").read_text() == old
    text = Path(job).read_text()
    assert "SCRATCH GUARD" in text
    for directive in ("--ntasks=16", "2-00:00:00", "--exclude=agg-[011-012]", "--constraint=amr",
                      f'MACE_WORKFLOW_ID="{WF}"'):
        assert directive in text, directive
    assert subprocess.run(["bash", "-n", str(job)]).returncode == 0


def test_without_a_plan_entry_the_template_is_kept_and_warned_about(engine, tmp_path, capsys):
    scripts = tmp_path / "workflow_scripts"
    scripts.mkdir()
    old = (DATA / "submitcrystal23_v1.1.1.sh").read_text()
    (scripts / "submitcrystal23_sp_2.sh").write_text(old)
    d = tmp_path / "workflow_outputs" / WF / "step_002_SP" / "mat"
    d.mkdir(parents=True)
    (d / "mat.d12").write_text("dummy\n")
    engine._create_slurm_script_for_calculation(d, "mat", "SP", 2, WF)
    assert "was copied from an older MACE" in capsys.readouterr().out
    assert (scripts / "submitcrystal23_sp_2.sh").read_text() == old
    assert not (scripts / "submitcrystal23_sp_2.sh.pre-1.1.3").exists()
