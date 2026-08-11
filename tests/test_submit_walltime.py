"""`mace submit --walltime` must reach the SLURM directive on every path.

The per-calc-type defaults are production-sized (OPT 7 days, SP 3 days). A test
or QA run does not need them and queues far sooner without them, so one flag
overrides walltime for the manual submission and for every step of a
`--progress` plan.

The plan path is the subtle one: resources are baked into workflow_scripts/*.sh
by copy_and_customize_scripts, so an override applied after that call would be
written to the plan JSON and silently ignored by the jobs actually submitted.
"""
import json
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT))

from mace.workflow import manual_progress as mp  # noqa: E402


def test_apply_walltime_rewrites_both_resource_shapes():
    """slurm_config carries resources twice — top level and per script entry.
    Both feed the generated directives, so both must be rewritten."""
    step_configs = {
        "OPT_1": {"slurm_config": {
            "resources": {"walltime": "7-00:00:00", "ntasks": 32},
            "scripts": {"submitcrystal23.sh": {
                "resources": {"walltime": "7-00:00:00", "ntasks": 32}}}}},
        "BAND_2": {"slurm_config": {
            "resources": {"walltime": "2:00:00"},
            "scripts": {"submit_prop.sh": {"resources": {"walltime": "2:00:00"}}}}},
    }
    mp._apply_walltime(step_configs, "0:30:00")

    for key, cfg in step_configs.items():
        slurm = cfg["slurm_config"]
        assert slurm["resources"]["walltime"] == "0:30:00", key
        for s in slurm["scripts"].values():
            assert s["resources"]["walltime"] == "0:30:00", key
    # unrelated fields untouched
    assert step_configs["OPT_1"]["slurm_config"]["resources"]["ntasks"] == 32


def test_apply_walltime_tolerates_missing_slurm_config():
    step_configs = {"OPT_1": {"source": "existing_d12"}, "SP_2": {"slurm_config": {}}}
    mp._apply_walltime(step_configs, "1:00:00")  # must not raise


def test_progress_plan_walltime_reaches_generated_scripts(tmp_path, monkeypatch):
    """End of the contract: the override must be applied BEFORE the scripts are
    written, so what SLURM receives matches the plan."""
    from mace.workflow.planner import WorkflowPlanner

    deck = tmp_path / "mat_opt.d12"
    deck.write_text("TESTGEOM\nEND\n")
    monkeypatch.chdir(tmp_path)

    seen = {}

    def spy_scripts(self, step_configs, workflow_id, queue_config):
        # snapshot walltimes AT THE MOMENT the scripts would be generated
        seen["at_generation"] = sorted(
            cfg["slurm_config"]["resources"]["walltime"]
            for cfg in step_configs.values()
            if isinstance(cfg.get("slurm_config"), dict)
        )

    monkeypatch.setattr(WorkflowPlanner, "copy_and_customize_scripts", spy_scripts)

    plan_file, workflow_id, sequence = mp.build_progress_plan(
        work_dir=tmp_path, db_path="materials.db", input_files=[deck],
        progress="full_electronic", walltime="2:00:00")

    assert seen["at_generation"] == ["2:00:00"] * len(sequence), seen
    plan = json.loads(plan_file.read_text())
    for cfg in plan["step_configurations"].values():
        assert cfg["slurm_config"]["resources"]["walltime"] == "2:00:00"


def test_progress_plan_without_walltime_keeps_defaults(tmp_path, monkeypatch):
    from mace.workflow.planner import WorkflowPlanner

    deck = tmp_path / "mat_opt.d12"
    deck.write_text("TESTGEOM\nEND\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(WorkflowPlanner, "copy_and_customize_scripts",
                        lambda self, sc, wid, qc: None)

    plan_file, _, _ = mp.build_progress_plan(
        work_dir=tmp_path, db_path="materials.db", input_files=[deck],
        progress="opt_sp")

    plan = json.loads(plan_file.read_text())
    walltimes = {k: c["slurm_config"]["resources"]["walltime"]
                 for k, c in plan["step_configurations"].items()}
    assert walltimes["OPT_1"] == "7-00:00:00", walltimes
    assert walltimes["SP_2"] == "3-00:00:00", walltimes


def test_queue_manager_rewrites_template_walltime(tmp_path):
    """The --track path submits through the queue manager, which runs the
    GENERATOR template (the template emits the directives and then submits, so
    rewriting the generated file afterwards is too late). Regression: the first
    HPCC run of --track --walltime produced -t 7-00:00:00 because this path was
    not wired."""
    from mace.queue.manager import EnhancedCrystalQueueManager

    template = tmp_path / "submitcrystal23.sh"
    template.write_text(
        "#!/bin/bash\n"
        "echo '#SBATCH --ntasks=32' >> $1.sh\n"
        "echo '#SBATCH -t 7-00:00:00' >> $1.sh\n"
        "echo '#SBATCH --mem-per-cpu=5G' >> $1.sh\n")

    mgr = EnhancedCrystalQueueManager.__new__(EnhancedCrystalQueueManager)
    mgr.walltime_override = "2:00:00"
    out = mgr._template_with_walltime(template, tmp_path)

    assert out is not None
    text = Path(out).read_text()
    assert "echo '#SBATCH -t 2:00:00' >> $1.sh" in text, text
    assert "7-00:00:00" not in text
    # untouched directives survive
    assert "#SBATCH --ntasks=32" in text and "#SBATCH --mem-per-cpu=5G" in text


def test_queue_manager_without_override_uses_original_template(tmp_path):
    from mace.queue.manager import EnhancedCrystalQueueManager

    template = tmp_path / "submitcrystal23.sh"
    template.write_text("echo '#SBATCH -t 7-00:00:00' >> $1.sh\n")
    mgr = EnhancedCrystalQueueManager.__new__(EnhancedCrystalQueueManager)
    mgr.walltime_override = None
    assert mgr._template_with_walltime(template, tmp_path) is None


@pytest.mark.parametrize("mod_name,script,default,override", [
    ("mace.submission.crystal", "submitcrystal23.sh", "7-00:00:00", "2:00:00"),
    # the D3 default is already 2:00:00, so override with a distinct value or
    # the "default is gone" assertion cannot distinguish them
    ("mace.submission.properties", "submit_prop.sh", "2:00:00", "0:30:00"),
])
def test_manual_path_writes_walltime_directive(tmp_path, monkeypatch, mod_name,
                                               script, default, override):
    """The non-tracking path rewrites the generator template; the customized
    copy must carry the requested -t."""
    import importlib
    mod = importlib.import_module(mod_name)

    template = tmp_path / script
    template.write_text(
        "#!/bin/bash\n"
        "echo '#SBATCH --ntasks=32' >> $1.sh\n"
        f"echo '#SBATCH -t {default}' >> $1.sh\n"
        "echo '#SBATCH --mem-per-cpu=5G' >> $1.sh\n"
        "echo '#SBATCH -A mendoza_q' >> $1.sh\n")

    get_defaults = (mod.get_default_d12_resources
                    if hasattr(mod, "get_default_d12_resources")
                    else mod.get_default_d3_resources)
    monkeypatch.setattr(mod, "get_default_amd20_exclusion", lambda: None, raising=False)
    resources = dict(get_defaults())
    resources["walltime"] = override
    resources["node_exclusion"] = None

    custom = mod.create_custom_slurm_script(template, resources)
    text = Path(custom).read_text()
    assert f"#SBATCH -t {override}" in text, text
    assert f"#SBATCH -t {default}" not in text
    Path(custom).unlink()
