"""Workflow follow-up steps must submit the whole job script.

HPCC QA (2026-10-03): every automatic follow-up (SP, FREQ, BAND, DOSS) ran
for about 5 seconds and wrote no .out. The engine wrote the customized
script generator to <material>.sh and then ran `bash <material>.sh
<material>`. The generator's first line truncates <material>.sh, the file
bash is reading, and rewrites it, so bash read the rest of a different file.
On HPCC the submitted script stopped after the module loads: no Pcrystal and
no completion callback. The executor (first workflow step) did the same.

These tests run the real submission methods against the shipped generators,
with sbatch replaced by a stub that keeps a copy of the script it is given.
"""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from conftest import REPO_ROOT

GENERATORS = REPO_ROOT / "mace" / "submission"


@pytest.fixture
def capture_sbatch(tmp_path, monkeypatch):
    bindir = tmp_path / "fakebin"
    bindir.mkdir()
    captured = tmp_path / "captured"
    captured.mkdir()
    sbatch = bindir / "sbatch"
    sbatch.write_text(
        "#!/bin/bash\n"
        f"cp \"$1\" '{captured}'/\n"
        "echo 'Submitted batch job 4242'\n")
    sbatch.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    for var in ("MACE_CONTEXT_DIR", "MACE_WORKFLOW_ID", "MACE_ISOLATION_MODE"):
        monkeypatch.delenv(var, raising=False)
    return captured


def _assert_complete(script: Path, program: str):
    text = script.read_text()
    # HPCC rerun: with the whole script submitted, the SP and FREQ ran
    # CRYSTAL and then stopped at 'syntax error: unexpected end of file'
    # before the callback, because the engine had replaced the generator's
    # callback section from '# ADDED' up to the 'fi' in the word 'file'.
    check = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert check.returncode == 0, check.stderr
    assert "# END SCRATCH FALLBACK CLEANUP" in text
    assert 'MACE_DB_PATH' in text, "the generator's own callback was replaced"
    assert text.startswith("#!/bin/bash --login"), text[:200]
    assert f"$EBROOTCRYSTAL/bin/{program}" in text, (
        f"submitted script has no {program} line ({len(text.splitlines())} lines)")
    assert "--callback-mode completion" in text
    # the script handed to sbatch is the generated batch file, not a generator
    assert "echo '#!/bin/bash --login' >" not in text


@pytest.mark.parametrize("generator,program,calc_type", [
    ("submitcrystal23.sh", "Pcrystal", "SP"),
    ("submit_prop.sh", "Pproperties", "BAND"),
])
def test_engine_follow_up_submits_complete_script(tmp_path, capture_sbatch,
                                                  generator, program, calc_type):
    from mace.workflow.engine import WorkflowEngine

    class _DB:
        db_path = tmp_path / "materials.db"

    eng = WorkflowEngine.__new__(WorkflowEngine)
    eng.base_work_dir = tmp_path
    eng.db = _DB()
    # the planner copies the generators into workflow_scripts/
    scripts = tmp_path / "workflow_scripts"
    scripts.mkdir()
    name = ("submitcrystal23_sp_2.sh" if calc_type == "SP"
            else "submit_prop_band_3.sh")
    shutil.copy(GENERATORS / generator, scripts / name)

    calc_dir = tmp_path / "workflow_outputs" / "workflow_t" / "step_002" / "MgO"
    calc_dir.mkdir(parents=True)
    script = eng._create_slurm_script_for_calculation(
        calc_dir, "MgO", calc_type, 2, "workflow_t")

    job_id = eng._submit_calculation_to_slurm(script, calc_dir)

    assert job_id == "4242"
    _assert_complete(capture_sbatch / "MgO.sh", program)
    _assert_complete(calc_dir / "MgO.sh", program)
    assert [p.name for p in calc_dir.glob("*.sh")] == ["MgO.sh"]


def test_executor_first_step_submits_complete_script(tmp_path, capture_sbatch):
    from mace.workflow.executor import WorkflowExecutor

    calc_dir = tmp_path / "step_001_OPT" / "Si"
    calc_dir.mkdir(parents=True)
    # generate_individual_slurm_script writes the customized generator here
    script = calc_dir / "Si.sh"
    shutil.copy(GENERATORS / "submitcrystal23.sh", script)

    executor = WorkflowExecutor.__new__(WorkflowExecutor)
    job_id = executor.submit_slurm_job(script, calc_dir)

    assert job_id == "4242"
    _assert_complete(capture_sbatch / "Si.sh", "Pcrystal")
