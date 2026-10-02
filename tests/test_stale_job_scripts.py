"""Job scripts written by a MACE older than 1.1.3 are reused, never rewritten.

`mace submit` keeps an existing <job>.sh unless --overwrite-sh is given, and a
workflow builds every step's script from its workflow_scripts/ copy taken when
it was planned. Scripts from 1.1.1/1.1.2 have no scratch guard (an empty
$SCRATCH on a node gives "END OF DATA IN INPUT DECK") and no GUESSP staging,
so reusing one must say so loudly - without touching the script, which may
have been edited by hand.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mace.workflow.engine import WorkflowEngine

REPO = Path(__file__).resolve().parent.parent
MACE_CLI = REPO / "mace_cli"

# What the 1.1.1 generators wrote (the parts that matter here).
OLD_BODY = """export DIR=$SLURM_SUBMIT_DIR
export scratch=$SCRATCH/crys23{prop}

echo "submit directory: "
echo $SLURM_SUBMIT_DIR

module purge
module load CRYSTAL/23-intel-2023a

mkdir  -p $scratch/$JOB
cp $DIR/$JOB.{ext}  $scratch/$JOB/INPUT
cd $scratch/$JOB

I_MPI_HYDRA_BOOTSTRAP="ssh" mpirun -n $SLURM_NTASKS $EBROOTCRYSTAL/bin/{exe} 2>&1 >& $DIR/${{JOB}}.out
cp fort.9 ${{DIR}}/${{JOB}}.f9
"""


def old_script(job, ext="d12"):
    prop = "/prop" if ext == "d3" else ""
    exe = "Pproperties" if ext == "d3" else "Pcrystal"
    return (f"#!/bin/bash --login\n#SBATCH -J {job}\n#SBATCH -o {job}-%J.o\n"
            f"export JOB={job}\n" + OLD_BODY.format(prop=prop, ext=ext, exe=exe))


def _env():
    env = dict(os.environ)
    for k in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME"):
        env.pop(k, None)
    return env


def _submit(target, cwd, *flags, stdin=""):
    return subprocess.run([sys.executable, str(MACE_CLI), "submit", str(target), "--nosubmit", *flags],
                          cwd=cwd, input=stdin, capture_output=True, text=True, timeout=120,
                          env=_env())


def test_submit_warns_when_reusing_a_pre_guard_crystal_script(tmp_path):
    (tmp_path / "mat_sp.d12").write_text("mat\nCRYSTAL\nEND\n")
    old = old_script("mat_sp")
    (tmp_path / "mat_sp.sh").write_text(old)
    r = _submit(tmp_path / "mat_sp.d12", tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    log = r.stdout + r.stderr
    assert "Using existing script: mat_sp.sh" in log
    assert "WARNING: mat_sp.sh was written by an older MACE" in log
    assert "--overwrite-sh" in log
    assert (tmp_path / "mat_sp.sh").read_text() == old         # not rewritten


def test_submit_warns_for_a_pre_guard_properties_script_in_a_directory(tmp_path):
    (tmp_path / "mat_band.d3").write_text("BAND\nEND\n")
    old = old_script("mat_band", "d3")
    (tmp_path / "mat_band.sh").write_text(old)
    r = _submit(tmp_path, tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "WARNING: mat_band.sh was written by an older MACE" in r.stdout + r.stderr
    assert (tmp_path / "mat_band.sh").read_text() == old


def test_submit_is_quiet_for_a_current_script_and_for_a_users_own(tmp_path):
    (tmp_path / "mat_sp.d12").write_text("mat\nCRYSTAL\nEND\n")
    r = _submit(tmp_path / "mat_sp.d12", tmp_path)          # generates a current one
    assert r.returncode == 0, r.stdout + r.stderr
    assert "SCRATCH GUARD" in (tmp_path / "mat_sp.sh").read_text()
    r = _submit(tmp_path / "mat_sp.d12", tmp_path)          # and reuses it
    assert "Using existing script" in r.stdout and "older MACE" not in r.stdout + r.stderr

    (tmp_path / "mat_sp.sh").write_text("#!/bin/bash\nsrun my_own_crystal_wrapper mat_sp\n")
    r = _submit(tmp_path / "mat_sp.d12", tmp_path)
    assert "older MACE" not in r.stdout + r.stderr


def test_overwrite_sh_replaces_the_old_script_without_a_warning(tmp_path):
    (tmp_path / "mat_sp.d12").write_text("mat\nCRYSTAL\nEND\n")
    (tmp_path / "mat_sp.sh").write_text(old_script("mat_sp"))
    r = _submit(tmp_path / "mat_sp.d12", tmp_path, "--overwrite-sh")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "older MACE" not in r.stdout + r.stderr
    assert "SCRATCH GUARD" in (tmp_path / "mat_sp.sh").read_text()


# --- workflow_scripts/ snapshots -------------------------------------------

class _DB:
    def __init__(self, db_path):
        self.db_path = db_path


@pytest.fixture
def engine(tmp_path, monkeypatch):
    for var in ("MACE_CONTEXT_DIR", "MACE_WORKFLOW_ID", "MACE_ISOLATION_MODE"):
        monkeypatch.delenv(var, raising=False)
    eng = WorkflowEngine.__new__(WorkflowEngine)
    eng.base_work_dir = tmp_path
    (tmp_path / "materials.db").touch()
    eng.db = _DB(tmp_path / "materials.db")
    return eng


def _calc_dir(tmp_path, step, name):
    d = tmp_path / "workflow_outputs" / "workflow_test" / step / name
    d.mkdir(parents=True)
    return d


def test_workflow_warns_when_its_copied_script_predates_the_guard(engine, tmp_path, capsys):
    scripts = tmp_path / "workflow_scripts"
    scripts.mkdir()
    old = old_script("$1")
    (scripts / "submitcrystal23_sp_2.sh").write_text(old)
    calc_dir = _calc_dir(tmp_path, "step_002_SP", "1_dia_sp")
    (calc_dir / "1_dia_sp.d12").write_text("dummy\n")
    engine._create_slurm_script_for_calculation(calc_dir, "1_dia_sp", "SP", 2, "workflow_test")
    out = capsys.readouterr().out
    assert "WARNING: workflow_scripts/submitcrystal23_sp_2.sh was copied from an older MACE" in out
    assert "re-plan" in out
    assert (scripts / "submitcrystal23_sp_2.sh").read_text() == old


def test_workflow_is_quiet_with_a_current_template(engine, tmp_path, capsys):
    scripts = tmp_path / "workflow_scripts"
    scripts.mkdir()
    current = (REPO / "mace" / "submission" / "submitcrystal23.sh").read_text()
    (scripts / "submitcrystal23_sp_2.sh").write_text(current)
    calc_dir = _calc_dir(tmp_path, "step_002_SP", "1_dia_sp")
    (calc_dir / "1_dia_sp.d12").write_text("dummy\n")
    engine._create_slurm_script_for_calculation(calc_dir, "1_dia_sp", "SP", 2, "workflow_test")
    assert "older MACE" not in capsys.readouterr().out
