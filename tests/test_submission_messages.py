"""Messages the HPCC QA (2026-10-03) found misleading on every run.

(a) the engine warned that every shipped template 'will cause submission
    issues' (all of them are generators, which it runs correctly);
(c) workflow execution said 'workflow completed synchronously' when only
    its first step had been submitted;
(d) `mace database --action stats` suggested `--export-csv FILE`, where the
    documented form is --format/--output;
(e) `mace submit --help` (MACE's own text) did not list --walltime.
((b), the manager's 'Could not extract job ID from template output', is
covered by test_manager_cli_submits.)
"""
import os
import shutil
import subprocess
import sys

from conftest import REPO_ROOT

MACE_CLI = REPO_ROOT / "mace_cli"


def _env():
    env = dict(os.environ)
    env["MACE_NO_BANNER"] = "1"
    for var in ("MACE_CONTEXT_DIR", "MACE_WORKFLOW_ID", "MACE_ISOLATION_MODE",
                "MACE_HOME", "PYTHONPATH"):
        env.pop(var, None)
    return env


def _cli(cwd, *args):
    proc = subprocess.run([sys.executable, str(MACE_CLI), "--no-banner", *args],
                          cwd=str(cwd), env=_env(), capture_output=True, text=True,
                          timeout=120)
    return proc.returncode, proc.stdout + proc.stderr


def test_engine_does_not_call_the_shipped_generators_a_problem(tmp_path, capsys, monkeypatch):
    from mace.workflow.engine import WorkflowEngine
    for var in ("MACE_CONTEXT_DIR", "MACE_WORKFLOW_ID", "MACE_ISOLATION_MODE"):
        monkeypatch.delenv(var, raising=False)

    class _DB:
        db_path = tmp_path / "materials.db"

    eng = WorkflowEngine.__new__(WorkflowEngine)
    eng.base_work_dir = tmp_path
    eng.db = _DB()
    (tmp_path / "workflow_scripts").mkdir()
    shutil.copy(REPO_ROOT / "mace" / "submission" / "submitcrystal23.sh",
                tmp_path / "workflow_scripts" / "submitcrystal23_sp_2.sh")
    calc_dir = tmp_path / "step_002_SP" / "MgO"
    calc_dir.mkdir(parents=True)
    eng._create_slurm_script_for_calculation(calc_dir, "MgO", "SP", 2, "workflow_t")
    out = capsys.readouterr().out
    assert "will cause submission issues" not in out
    assert "appears to be a script generator" not in out


def test_execution_does_not_claim_the_workflow_completed(tmp_path, capsys, monkeypatch):
    from mace.workflow.executor import WorkflowExecutor
    ex = WorkflowExecutor.__new__(WorkflowExecutor)
    ex.active_workflows = {}
    for name in ("_configure_queue_manager_from_plan", "_apply_execution_settings",
                 "recreate_workflow_scripts",
                 "prepare_input_files", "execute_workflow_steps"):
        monkeypatch.setattr(ex, name, lambda *a, **k: None, raising=False)
    monkeypatch.setattr(ex, "copy_config_files", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(ex, "_validate_workflow_plan", lambda *a, **k: True, raising=False)
    ex._execute_workflow_plan_impl({"workflow_sequence": ["OPT", "SP"]}, "workflow_t")
    out = capsys.readouterr().out
    assert "completed synchronously" not in out
    assert "mace monitor" in out


def test_stats_hint_names_the_export_options(tmp_path):
    rc, out = _cli(tmp_path, "database", "--action", "stats")
    assert rc == 0, out
    assert "--export-csv" not in out
    assert "--action export --format csv --output FILE" in out


def test_submit_help_lists_walltime(tmp_path):
    rc, out = _cli(tmp_path, "submit", "--help")
    assert rc == 0, out
    assert "--walltime" in out
