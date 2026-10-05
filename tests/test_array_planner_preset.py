"""The planner's old "--array=1-10" custom-directive preset (option 5).

It made every job built from the step's template a 10-task array of the SAME
material script: ten CRYSTAL runs sharing one scratch directory and one
$JOB.out, and ten completion callbacks. On the `mace submit --progress
interactive` route it survived into every follow-up job; on `mace workflow
--execute` it was dropped silently. Job arrays of many materials are now
`--array` of mace submit / mace workflow: the preset is gone (the other keys
keep their numbers), an --array directive is refused when typed, and one in
an old plan is dropped with a warning on both template-building paths.
"""
import json

import pytest

from mace.submission.job_array import drop_array_directives, is_array_directive
from test_planner_free_text_prompts_stop import _feed, _planner


@pytest.mark.parametrize("d,yes", [("--array=1-10", True), ("--array 0-3", True),
                                   ("-a 1-10", True), ("-a1-10", True), ("--array", True),
                                   ("--account=general", False), ("-A general", False),
                                   ("--constraint=amd20", False)])
def test_array_directive_recognized(d, yes):
    assert is_array_directive(d) is yes


def test_option_5_is_gone_and_refused(monkeypatch, capsys):
    _feed(monkeypatch, ["5", "4", ""])
    got = [c["directive"] for c in _planner().get_additional_customizations("OPT")]
    assert got == ["--gres=gpu:1"]
    cap = capsys.readouterr()
    out = cap.out + cap.err
    assert "--array=1-10" not in out
    assert "Job arrays run one material per task" in out
    assert "6: Custom directive" in out


def test_piped_5_stops_after_repeats(monkeypatch):
    _feed(monkeypatch, ["5", "5", "5"])
    with pytest.raises(SystemExit) as exc:
        _planner().get_additional_customizations("OPT")
    assert "not a terminal" in str(exc.value)


def test_typed_array_directives_are_refused(monkeypatch, capsys):
    _feed(monkeypatch, ["--array=0-9", "6", "-a 1-10", "--constraint=amd20", ""])
    got = [c["directive"] for c in _planner().get_additional_customizations("OPT")]
    assert got == ["--constraint=amd20"]
    cap = capsys.readouterr()
    assert (cap.out + cap.err).count("Job arrays run one material per task") == 2


def test_drop_array_directives_keeps_the_others(capsys):
    kept = drop_array_directives([{"directive": "--array=1-10"},
                                  {"directive": "--constraint=amd20"},
                                  {"find": "a", "replace": "b"}], "submitcrystal23_opt_1.sh")
    assert kept == [{"directive": "--constraint=amd20"}, {"find": "a", "replace": "b"}]
    assert "dropped the custom SLURM directive '--array=1-10' of submitcrystal23_opt_1.sh" in \
        capsys.readouterr().out


def test_old_plan_directive_dropped_by_the_planner(tmp_path, capsys):
    from mace.workflow.planner import WorkflowPlanner
    planner = WorkflowPlanner(work_dir=tmp_path)
    template = "\n".join(["echo '#SBATCH --ntasks=32' >> $1.sh",
                          "echo '#SBATCH -t 7-00:00:00' >> $1.sh",
                          "echo 'export JOB=test' >> $1.sh"])
    cfg = {"resources": {"ntasks": 16, "walltime": "1-00:00:00"},
           "customizations": [{"directive": "--array=1-10", "description": "Common: --array=1-10"},
                              {"directive": "--constraint=amd20"}],
           "workflow_id": "wf_test"}
    out = planner.apply_script_customizations(template, cfg, {
        "max_jobs": 950, "reserve_slots": 50, "max_submit_batch": 10,
        "max_recovery_attempts": 3})
    assert "--array" not in out and "--constraint=amd20" in out
    assert "dropped the custom SLURM directive '--array=1-10'" in capsys.readouterr().out


def test_old_plan_directive_dropped_by_the_engine_rebuild(tmp_path, capsys, monkeypatch):
    from test_workflow_template_rebuild import DATA, WF, _plan
    from mace.workflow.engine import WorkflowEngine
    for var in ("MACE_CONTEXT_DIR", "MACE_WORKFLOW_ID", "MACE_ISOLATION_MODE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)
    plan = _plan(tmp_path, "submitcrystal23.sh", "submitcrystal23_sp_2.sh", "SP")
    cfg = plan["step_configurations"]["SP_2"]["slurm_config"]["scripts"]["submitcrystal23.sh"]
    cfg["customizations"].append({"directive": "--array=1-10"})
    (tmp_path / "workflow_configs" / f"workflow_plan_{WF.replace('workflow_', '')}.json") \
        .write_text(json.dumps(plan))
    eng = WorkflowEngine.__new__(WorkflowEngine)
    eng.base_work_dir = tmp_path
    tpl = tmp_path / "workflow_scripts" / "submitcrystal23_sp_2.sh"
    tpl.parent.mkdir()
    old = (DATA / "submitcrystal23_v1.1.1.sh").read_text()
    tpl.write_text(old)
    content = eng._rebuild_pre_guard_template(tpl, old, WF)
    assert content and "--array" not in content
    assert "dropped the custom SLURM directive '--array=1-10'" in capsys.readouterr().out
