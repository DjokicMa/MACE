"""Characterization tests for the pure logic kernels in
mace/workflow/planner.py (WorkflowPlanner).

These pin CURRENT, validated behavior of resource scaling, the dependency
gates, calc-instance numbering, material-ID derivation, submit-script dispatch,
and the SLURM-script templating engine — the decisions that become the actual
job scripts and workflow DAG. Expected values were captured from the real code
(no SLURM, no DB, no 12GB corpus).

WorkflowPlanner.__init__ only mkdir's four work subdirs (and instantiates the
node manager); after construction the tested methods do no further I/O, so a
single tmp-backed planner is shared across the module.
"""
import pytest

from mace.workflow.planner import WorkflowPlanner


@pytest.fixture(scope="module")
def planner(tmp_path_factory):
    return WorkflowPlanner(work_dir=tmp_path_factory.mktemp("planner_work"))


# --- apply_calc_type_scaling: walltime is D-HH:MM:SS; mutates dict in place ---
@pytest.mark.parametrize("resources,calc,expected", [
    ({"walltime": "7-00:00:00", "memory_per_cpu": "5G"}, "OPT",
     {"walltime": "7-00:00:00", "memory_per_cpu": "5G"}),
    ({"walltime": "7-00:00:00", "memory_per_cpu": "5G"}, "SP",
     {"walltime": "3-00:00:00", "memory_per_cpu": "4G"}),        # 5G * 0.8
    ({"walltime": "7-00:00:00", "memory_per_cpu": "5G"}, "FREQ",
     {"walltime": "7-00:00:00", "memory_per_cpu": "7G"}),        # int(5*1.5)
    ({"walltime": "2:00:00", "memory": "80G"}, "BAND",
     {"walltime": "2:00:00", "memory": "48G"}),                  # 80G * 0.6
    ({"walltime": "2:00:00", "memory": "80G"}, "BAND2",
     {"walltime": "2:00:00", "memory": "48G"}),                  # BAND2 scales like BAND
    ({"walltime": "2:00:00", "memory": "80G"}, "TRANSPORT",
     {"walltime": "2:00:00", "memory": "80G"}),                  # unknown -> unchanged
])
def test_apply_calc_type_scaling(planner, resources, calc, expected):
    assert planner.apply_calc_type_scaling(resources, calc) == expected


def test_get_default_resources_memory_key_divergence(planner):
    opt = planner.get_default_resources("submitcrystal23.sh", "OPT")
    assert opt["ntasks"] == 32 and opt["walltime"] == "7-00:00:00"
    assert opt["memory_per_cpu"] == "5G" and "memory" not in opt

    sp = planner.get_default_resources("submitcrystal23.sh", "SP")
    assert sp["walltime"] == "3-00:00:00" and sp["memory_per_cpu"] == "4G"

    band = planner.get_default_resources("submit_prop.sh", "BAND")
    assert band["ntasks"] == 28 and band["memory"] == "48G"
    assert "memory_per_cpu" not in band              # submit_prop uses total 'memory'

    fb = planner.get_default_resources("unknown.sh", "OPT")
    assert fb["ntasks"] == 16 and fb["memory_per_cpu"] == "4G"


@pytest.mark.parametrize("seq,base,expected", [
    ([], "OPT", "OPT"),
    (["OPT"], "OPT", "OPT2"),
    (["OPT", "OPT2"], "OPT", "OPT3"),
    (["OPT", "SP"], "SP", "SP2"),
    (["OPT", "SP", "BAND"], "DOSS", "DOSS"),
])
def test_get_next_numbered_calc(planner, seq, base, expected):
    assert planner._get_next_numbered_calc(seq, base) == expected


@pytest.mark.parametrize("seq,new,expected", [
    (["OPT"], "SP", True),
    (["OPT"], "BAND", True),
    ([], "BAND", False),                       # BAND needs SP or OPT
    ([], "FREQ", False),                       # FREQ needs OPT
    (["OPT"], "FREQ", True),
    (["OPT", "SP"], "CHARGE+POTENTIAL", True),  # '+' survives the regex char class
    ([], "CHARGE+POTENTIAL", False),
    ([], "OPT", True),
    ([], "garbage!", False),                   # regex non-match
    (["SP"], "DOSS", True),                    # DOSS needs SP-or-OPT (any)
])
def test_validate_numbered_calc_addition(planner, seq, new, expected):
    assert planner._validate_numbered_calc_addition(seq, new) is expected


def test_validate_calc_addition_strict_and_keyerror(planner):
    # Strict: requires ALL declared deps present (BAND depends on SP AND OPT).
    assert planner.validate_calc_addition(["SP", "OPT"], "BAND") is True
    assert planner.validate_calc_addition(["SP"], "BAND") is False
    assert planner.validate_calc_addition([], "OPT") is True
    assert planner.validate_calc_addition([], "SP") is True
    # Numbered names are absent from calc_types -> KeyError (callers pass base types).
    with pytest.raises(KeyError):
        planner.validate_calc_addition([], "BAND2")


@pytest.mark.parametrize("stem,expected", [
    ("test1_opt_BULK_PBE", "test1"),
    ("mat_name_BULK_OPTGEOM", "mat_name"),
    ("quartz_SP_HSE06", "quartz"),
    ("foo_PBE0_POB-TZVP", "foo"),
    ("bar_D3", "bar"),
    ("plain", "plain"),
])
def test_create_clean_material_id(planner, stem, expected):
    from pathlib import Path
    assert planner.create_clean_material_id(Path(stem + ".d12")) == expected


@pytest.mark.parametrize("calc,expected", [
    ("OPT", ["submitcrystal23.sh"]),
    ("FREQ", ["submitcrystal23.sh"]),
    ("SP3", ["submitcrystal23.sh"]),
    ("BAND", ["submit_prop.sh"]),
    ("DOSS2", ["submit_prop.sh"]),
    ("CHARGE+POTENTIAL", ["submit_prop.sh"]),
    ("TRANSPORT", ["submit_prop.sh"]),
    ("WEIRD", ["submitcrystal23.sh"]),          # default
])
def test_get_required_scripts(planner, calc, expected):
    assert planner.get_required_scripts(calc) == expected


def test_apply_script_customizations_rewrites_sbatch_and_callback(planner):
    template = "\n".join([
        "echo '#SBATCH --ntasks=32' >> $1.sh",
        "echo '#SBATCH -t 7-00:00:00' >> $1.sh",
        "echo '#SBATCH --mem-per-cpu=5G' >> $1.sh",
        "echo '#SBATCH -A old' >> $1.sh",
        "echo 'export JOB=test' >> $1.sh",
        "python enhanced_queue_manager.py --max-jobs 250 --reserve 30 "
        "--max-submit 5 --max-recovery-attempts 9",
    ])
    script_config = {
        "resources": {"ntasks": 16, "walltime": "3-00:00:00",
                      "memory_per_cpu": "4G", "account": "foo"},
        "customizations": [{"directive": "--gres=gpu:1"}],
        "node_exclusion": "amr-[001-005]",
        "workflow_id": "wf_test",
    }
    queue_config = {"max_jobs": 950, "reserve_slots": 50,
                    "max_submit_batch": 10, "max_recovery_attempts": 3}

    out = planner.apply_script_customizations(template, script_config, queue_config)

    for needle in ("--ntasks=16", "-t 3-00:00:00", "--mem-per-cpu=4G", "-A foo",
                   "--exclude=amr-[001-005]", "--gres=gpu:1",
                   'MACE_WORKFLOW_ID="wf_test"',
                   "--max-jobs 950 --reserve 50 --max-submit 10",
                   "--max-recovery-attempts 3"):
        assert needle in out, f"missing {needle!r}"
    assert "250" not in out  # old hardcoded callback values replaced
