"""The workflow plan JSON: what the planner saves, what the executor loads and
checks, and what the engine reads back from a job's callback.

The file is the only link between planning (`mace workflow`), execution
(`mace workflow --execute <plan.json>`) and progression (each job's completion
callback, which finds workflow_configs/workflow_plan_<id>.json by searching
upward from its job directory). These tests follow one plan through all three
without SLURM or a database.
"""
import json
from pathlib import Path

import pytest

from mace.workflow.engine import WorkflowEngine
from mace.workflow.executor import WorkflowExecutor
from mace.workflow.planner import WorkflowPlanner

WF_ID = "workflow_20260702_193809"


def make_plan(input_dir, **over):
    """A plan with the keys and layout run_workflow_execution writes."""
    plan = {
        "created": "2026-07-02T19:38:09",
        "input_type": "d12",
        "input_directory": input_dir,
        "input_files": {"cif": [], "d12": [str(Path(input_dir) / "dia.d12")]},
        "workflow_sequence": ["OPT", "SP", "BAND", "DOSS", "FREQ", "OPT2"],
        "step_configurations": {
            "OPT_1": {"calculation_type": "OPT", "optimization_type": "FULLOPTG"},
            "SP_2": {"calculation_type": "SP",
                     "method_modifications": {"new_functional": "B3LYP"}},
            "OPT2_6": {"calculation_type": "OPT",
                       "optimization_settings": {"TOLDEE": 9, "MAXCYCLE": 800}},
        },
        "cif_conversion_config": None,
        "isolation_mode": "shared",
        "post_completion_action": "keep",
        "queue_management": {"max_jobs": 150, "reserve_slots": 20,
                             "max_submit_batch": 4, "max_recovery_attempts": 2},
        "execution_settings": {"max_concurrent_jobs": 200,
                               "enable_material_tracking": True,
                               "auto_progression": True},
        "workflow_id": WF_ID,
    }
    plan.update(over)
    return plan


@pytest.fixture
def input_dir(tmp_path):
    d = tmp_path / "inputs"
    d.mkdir()
    (d / "dia.d12").write_text("dia\nCRYSTAL\n0 0 0\n227\n3.56\n1\n6 0.125 0.125 0.125\nEND\n")
    return d


def bare_executor(work_dir):
    """An executor with its directories but no database or queue manager."""
    ex = WorkflowExecutor.__new__(WorkflowExecutor)
    ex.work_dir = Path(work_dir)
    ex.configs_dir = ex.work_dir / "workflow_configs"
    ex.outputs_dir = ex.work_dir / "workflow_outputs"
    ex.temp_dir = ex.work_dir / "workflow_temp"
    ex.db_path = str(ex.work_dir / "materials.db")
    ex.db = None
    ex.queue_manager = None
    ex.active_workflows = {}
    return ex


def bare_engine(base):
    eng = WorkflowEngine.__new__(WorkflowEngine)
    eng.base_work_dir = Path(base)
    return eng


# ------------------------------------------------------------ planner: save

def test_planner_saves_the_plan_under_its_workflow_id(tmp_path, input_dir):
    planner = WorkflowPlanner(str(tmp_path / "plan_here"))
    plan = make_plan(str(input_dir))
    path = planner.save_workflow_plan(plan)
    assert path == planner.configs_dir / "workflow_plan_20260702_193809.json"
    assert json.loads(path.read_text()) == plan


def test_planner_writes_paths_as_strings(tmp_path, input_dir):
    """The planner holds Paths while planning; the file must still be JSON."""
    planner = WorkflowPlanner(str(tmp_path))
    plan = make_plan(input_dir, input_files={"cif": [], "d12": [input_dir / "dia.d12"]})
    loaded = json.loads(planner.save_workflow_plan(plan).read_text())
    assert loaded["input_directory"] == str(input_dir)
    assert loaded["input_files"]["d12"] == [str(input_dir / "dia.d12")]


def test_planner_names_a_plan_without_id_by_timestamp(tmp_path, input_dir):
    planner = WorkflowPlanner(str(tmp_path))
    plan = make_plan(str(input_dir))
    del plan["workflow_id"]
    path = planner.save_workflow_plan(plan)
    stamp = path.name[len("workflow_plan_"):-len(".json")]
    assert path.parent == planner.configs_dir
    assert len(stamp) == 15 and stamp[8] == "_" and stamp.replace("_", "").isdigit()


# ---------------------------------------------------- planner: --execute FILE

def test_planner_execute_asks_first_and_can_be_declined(tmp_path, input_dir, monkeypatch):
    planner = WorkflowPlanner(str(tmp_path))
    path = planner.save_workflow_plan(make_plan(str(input_dir)))
    built = []
    monkeypatch.setattr("mace.workflow.executor.WorkflowExecutor",
                        lambda *a, **k: built.append(a) or pytest.fail("executor built"))
    prompts = []
    monkeypatch.setattr("builtins.input", lambda p="": prompts.append(p) or "n")
    planner.execute_workflow_plan(path)
    assert len(prompts) == 1 and "Proceed with workflow execution?" in prompts[0]
    assert built == []


def test_planner_execute_hands_the_file_to_the_executor(tmp_path, input_dir, monkeypatch):
    planner = WorkflowPlanner(str(tmp_path))
    path = planner.save_workflow_plan(make_plan(str(input_dir)))
    calls = []

    class FakeExecutor:
        def __init__(self, work_dir, db_path):
            calls.append(("init", work_dir, db_path))

        def execute_workflow_plan(self, plan_file):
            calls.append(("execute", plan_file))

    monkeypatch.setattr("mace.workflow.executor.WorkflowExecutor", FakeExecutor)
    monkeypatch.setattr("builtins.input", lambda p="": "")        # default: yes
    planner.execute_workflow_plan(path)
    assert calls == [("init", str(planner.work_dir), planner.db_path), ("execute", path)]


# ---------------------------------------------------- executor: load + persist

def test_executor_loads_the_file_and_persists_it_for_callbacks(tmp_path, input_dir):
    planner = WorkflowPlanner(str(tmp_path / "planned_elsewhere"))
    plan_file = planner.save_workflow_plan(make_plan(str(input_dir)))

    ex = bare_executor(tmp_path / "run")
    seen = []
    ex._execute_workflow_plan_impl = lambda plan, wid, mode: seen.append((plan, wid, mode))
    ex.execute_workflow_plan(plan_file)

    (plan, wid, mode), = seen
    assert (wid, mode) == (WF_ID, "shared")
    assert plan == json.loads(plan_file.read_text())
    persisted = tmp_path / "run" / "workflow_configs" / "workflow_plan_20260702_193809.json"
    assert json.loads(persisted.read_text()) == plan


def test_executor_mints_one_id_for_a_plan_without_one(tmp_path, input_dir):
    plan_file = tmp_path / "p.json"
    plan_file.write_text(json.dumps(make_plan(str(input_dir), workflow_id=None)))
    ex = bare_executor(tmp_path)
    seen = []
    ex._execute_workflow_plan_impl = lambda plan, wid, mode: seen.append((plan, wid))
    ex.execute_workflow_plan(plan_file)
    (plan, wid), = seen
    assert wid.startswith("workflow_") and plan["workflow_id"] == wid
    persisted = ex.configs_dir / f"workflow_plan_{wid[len('workflow_'):]}.json"
    assert json.loads(persisted.read_text())["workflow_id"] == wid


def test_persisting_never_overwrites_an_existing_plan(tmp_path, input_dir):
    ex = bare_executor(tmp_path)
    ex.configs_dir.mkdir(parents=True)
    dest = ex.configs_dir / "workflow_plan_20260702_193809.json"
    dest.write_text('{"workflow_sequence": ["OPT"]}')
    ex._persist_plan_for_callbacks(make_plan(str(input_dir)), WF_ID)
    assert json.loads(dest.read_text()) == {"workflow_sequence": ["OPT"]}


def test_cif_conversion_config_is_written_beside_the_plan_once(tmp_path, input_dir):
    ex = bare_executor(tmp_path)
    for d in (ex.configs_dir, ex.temp_dir):
        d.mkdir(parents=True)
    cfg = {"symmetry_handling": "CIF", "dimensionality": "CRYSTAL"}
    ex.copy_config_files(make_plan(str(input_dir), cif_conversion_config=cfg), WF_ID)
    written = ex.configs_dir / "cif_conversion_config.json"
    assert json.loads(written.read_text()) == cfg
    ex.copy_config_files(make_plan(str(input_dir), cif_conversion_config={"other": 1}), WF_ID)
    assert json.loads(written.read_text()) == cfg


# ------------------------------------------------------------ queue settings

def test_queue_settings_are_read_from_the_top_level_block(tmp_path, input_dir):
    ex = bare_executor(tmp_path)
    ex._configure_queue_manager_from_plan(make_plan(str(input_dir)))
    assert ex._pending_queue_config == {"max_jobs": 150, "reserve_slots": 20,
                                        "max_submit_batch": 4, "max_recovery_attempts": 2}


def test_queue_settings_fall_back_to_the_old_nested_block_then_defaults(tmp_path, input_dir):
    ex = bare_executor(tmp_path)
    plan = make_plan(str(input_dir))
    nested = plan.pop("queue_management")
    plan["execution_settings"]["queue_management"] = nested
    ex._configure_queue_manager_from_plan(plan)
    assert ex._pending_queue_config["max_jobs"] == 150
    del plan["execution_settings"]["queue_management"]
    ex._configure_queue_manager_from_plan(plan)
    assert ex._pending_queue_config == {"max_jobs": 200, "reserve_slots": 30,
                                        "max_submit_batch": 5, "max_recovery_attempts": 3}


def test_queue_settings_update_an_existing_manager(tmp_path, input_dir):
    class QM:
        max_jobs = reserve_slots = max_submit_per_callback = max_recovery_attempts = None
    ex = bare_executor(tmp_path)
    ex.queue_manager = QM()
    ex._configure_queue_manager_from_plan(make_plan(str(input_dir)))
    qm = ex.queue_manager
    assert (qm.max_jobs, qm.reserve_slots, qm.max_submit_per_callback,
            qm.max_recovery_attempts) == (150, 20, 4, 2)


# ------------------------------------------------------------ validation

def test_a_complete_plan_validates(tmp_path, input_dir):
    assert bare_executor(tmp_path)._validate_workflow_plan(make_plan(str(input_dir)))


@pytest.mark.parametrize("sequence", [["OPT", "SP2", "BAND3", "DOSS", "CHARGE+POTENTIAL",
                                       "TRANSPORT", "FREQ", "OPT2"]])
def test_numbered_steps_are_valid_types(tmp_path, input_dir, sequence):
    plan = make_plan(str(input_dir), workflow_sequence=sequence)
    assert bare_executor(tmp_path)._validate_workflow_plan(plan)


def _errors(ex, plan, capsys):
    capsys.readouterr()
    ok = ex._validate_workflow_plan(plan)
    captured = capsys.readouterr()
    return ok, captured.out + captured.err


@pytest.mark.parametrize("change, message", [
    ({"workflow_sequence": []}, "Empty workflow sequence"),
    ({"workflow_sequence": ["OPT", "PHONON"]}, "Invalid calculation type: PHONON"),
    ({"step_configurations": {"OPT_1": "FULLOPTG"}}, "Invalid step configuration for OPT_1"),
    ({"execution_settings": {"max_concurrent_jobs": 0}}, "Invalid max_concurrent_jobs: 0"),
    ({"execution_settings": {"max_concurrent_jobs": "200"}}, "Invalid max_concurrent_jobs: 200"),
    ({"queue_management": {"max_jobs": 1001}}, "Invalid queue max_jobs: 1001"),
])
def test_invalid_plans_are_refused_with_the_reason(tmp_path, input_dir, capsys, change, message):
    ok, out = _errors(bare_executor(tmp_path), make_plan(str(input_dir), **change), capsys)
    assert not ok
    assert message in out


@pytest.mark.parametrize("field", ["workflow_sequence", "input_directory", "input_type"])
def test_missing_required_fields_are_refused(tmp_path, input_dir, capsys, field):
    plan = make_plan(str(input_dir))
    del plan[field]
    ok, out = _errors(bare_executor(tmp_path), plan, capsys)
    assert not ok and f"Missing required field: {field}" in out


def test_input_directory_must_exist_and_hold_inputs(tmp_path, input_dir, capsys):
    ex = bare_executor(tmp_path)
    ok, out = _errors(ex, make_plan(str(tmp_path / "gone")), capsys)
    assert not ok and "Input directory does not exist" in out
    empty = tmp_path / "empty"
    empty.mkdir()
    ok, out = _errors(ex, make_plan(str(empty), input_files={"cif": [], "d12": []}), capsys)
    assert not ok and "No D12 files found" in out
    ok, out = _errors(ex, make_plan(str(empty), input_type="cif",
                                    input_files={"cif": [], "d12": []}), capsys)
    assert not ok and "No CIF files found" in out
    # A d12 plan whose file list is empty still runs when the directory has decks.
    assert ex._validate_workflow_plan(make_plan(str(input_dir),
                                                input_files={"cif": [], "d12": []}))


def test_an_invalid_plan_is_not_executed(tmp_path, input_dir):
    ex = bare_executor(tmp_path)
    ex.recreate_workflow_scripts = lambda plan: pytest.fail("ran an invalid plan")
    ex._execute_workflow_plan_impl(make_plan(str(input_dir), workflow_sequence=[]), WF_ID,
                                   "shared")


# ------------------------------------------------------ engine: reading back

def test_callback_finds_the_plan_from_a_nested_job_directory(tmp_path, input_dir, monkeypatch):
    ex = bare_executor(tmp_path)
    ex._persist_plan_for_callbacks(make_plan(str(input_dir)), WF_ID)
    job_dir = tmp_path / "workflow_outputs" / WF_ID / "step_002_SP" / "dia"
    job_dir.mkdir(parents=True)
    monkeypatch.chdir(job_dir)
    eng = bare_engine(tmp_path / "somewhere_else")
    assert eng.get_workflow_sequence(WF_ID) == ["OPT", "SP", "BAND", "DOSS", "FREQ", "OPT2"]
    assert eng._get_plan_step_config(WF_ID, "SP")["method_modifications"] == \
        {"new_functional": "B3LYP"}
    # OPT2's settings are its own, not OPT's.
    assert eng._get_plan_step_config(WF_ID, "OPT2")["optimization_settings"]["TOLDEE"] == 9
    assert "optimization_settings" not in eng._get_plan_step_config(WF_ID, "OPT")
    assert eng._get_plan_step_config(WF_ID, "TRANSPORT") == {}


def test_unreadable_or_missing_plans_read_as_none(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    eng = bare_engine(tmp_path)
    assert eng._load_workflow_plan("") is None
    assert eng._load_workflow_plan(WF_ID) is None
    assert eng.get_workflow_sequence(WF_ID) is None
    assert eng._get_plan_step_config(WF_ID, "SP") == {}
    (tmp_path / "workflow_configs").mkdir()
    (tmp_path / "workflow_configs" / "workflow_plan_20260702_193809.json").write_text("{not json")
    assert eng._load_workflow_plan(WF_ID) is None
