"""OPT -> SP -> MATDUMP -> WANNIER as a planned workflow.

MATDUMP is a properties (.d3) step like BAND/DOSS: the engine generates it
from the SP wavefunction with CRYSTALOptToD3 and submits it with submit_prop.sh.
WANNIER is not a SLURM job: once MATDUMP has completed, the engine runs the
bundled lcao2wannier (William Comaskey's) on the dump locally, as a post-step,
and records the outcome. When the conversion cannot run it is skipped with a
message saying why and how to run it by hand; it never blocks the workflow.

Everything here runs on fake outputs: no CRYSTAL, no SLURM, no real dump.
"""
import json
from pathlib import Path

import pytest

import mace.wannier.driver as driver
from mace.workflow.engine import WorkflowEngine
from mace.workflow.planner import WorkflowPlanner

SEQ = ["OPT", "SP", "MATDUMP", "WANNIER"]


# --------------------------------------------------------------------------
# Planner
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def planner(tmp_path_factory):
    return WorkflowPlanner(work_dir=tmp_path_factory.mktemp("planner_work"))


def test_matdump_runs_on_the_properties_script_and_wannier_on_none(planner):
    assert planner.get_required_scripts("MATDUMP") == ["submit_prop.sh"]
    assert planner.get_required_scripts("WANNIER") == []


@pytest.mark.parametrize("seq,new,expected", [
    (["OPT", "SP"], "MATDUMP", True),
    (["SP"], "MATDUMP", True),
    ([], "MATDUMP", False),                     # needs a wavefunction
    (["OPT", "SP"], "WANNIER", False),          # needs a MATDUMP before it
    (["OPT", "SP", "MATDUMP"], "WANNIER", True),
])
def test_matdump_and_wannier_dependencies(planner, seq, new, expected):
    assert planner._validate_numbered_calc_addition(seq, new) is expected


def test_both_steps_can_be_added_to_a_custom_workflow(planner):
    available = planner._get_available_calc_types(["OPT", "SP"])
    assert "MATDUMP" in available and "WANNIER" in available


def test_a_template_chains_the_whole_hand_off(planner):
    assert SEQ in [s for s in planner.workflow_templates.values()
                   if isinstance(s, list)]
    # The existing menu numbers do not move: the new template comes last.
    keys = list(planner.workflow_templates)
    assert keys.index("custom") == 8
    assert planner.workflow_templates[keys[-1]] == SEQ


def test_configuring_the_steps_asks_nothing_and_gives_wannier_no_job_script(
        planner, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail(
        "MATDUMP/WANNIER configuration must not prompt"))
    slurm_for = []
    monkeypatch.setattr(planner, "configure_slurm_scripts",
                        lambda calc, step: slurm_for.append(calc) or {"scripts": {}})
    monkeypatch.setattr(planner, "configure_single_point_step",
                        lambda calc, step: {"calculation_type": "SP"})

    configs = planner.configure_workflow_steps(SEQ, has_cifs=False)

    matdump = configs["MATDUMP_3"]
    assert matdump["d3_config"] == {"calculation_type": "MATDUMP",
                                    "n_rvectors": "auto"}
    assert "slurm_config" in matdump
    wannier = configs["WANNIER_4"]
    assert wannier["local_post_step"] is True
    assert "slurm_config" not in wannier
    assert "WANNIER" not in slurm_for


# --------------------------------------------------------------------------
# Engine: dependencies and triggering
# --------------------------------------------------------------------------


class _FakeDB:
    def __init__(self, calcs):
        self.calcs = list(calcs)
        self.created = []
        self.updates = []

    def get_calculations_by_status(self, material_id=None, **kw):
        return self.calcs

    def get_calculation(self, cid):
        return next((c for c in self.calcs if c["calc_id"] == cid), None)

    def create_calculation(self, **kwargs):
        calc_id = f"{kwargs['calc_type'].lower()}_new"
        self.created.append(kwargs)
        self.calcs.append({"calc_id": calc_id, "material_id": kwargs["material_id"],
                           "calc_type": kwargs["calc_type"], "status": "pending",
                           "settings_json": json.dumps(kwargs.get("settings") or {})})
        return calc_id

    def update_calculation_status(self, calc_id, status, **kwargs):
        self.updates.append((calc_id, status, kwargs))
        for calc in self.calcs:
            if calc["calc_id"] == calc_id:
                calc["status"] = status

    def update_workflow_state(self, *a, **k):
        pass


def _calc(calc_type, status="completed", cid=None, output_file=None):
    return {"calc_id": cid or f"{calc_type.lower()}_1", "material_id": "mat",
            "calc_type": calc_type, "status": status,
            "output_file": output_file,
            "settings_json": json.dumps({"workflow_id": "wf_test"})}


@pytest.fixture
def engine():
    eng = WorkflowEngine.__new__(WorkflowEngine)
    eng.triggered = []
    eng._use_new_d3_generation = lambda: True
    eng.generate_d3_calculation_new = lambda src, t: eng.triggered.append((t, src)) or f"id_{t}"
    eng.generate_property_calculation = lambda src, t: eng.triggered.append(("PROP_" + t, src)) or f"id_{t}"
    eng.generate_numbered_calculation = lambda src, t: eng.triggered.append((t, src)) or f"id_{t}"
    eng.run_wannier_post_step = lambda src, t="WANNIER": eng.triggered.append((t, src)) or f"id_{t}"
    return eng


def test_matdump_depends_on_the_wavefunction_and_wannier_on_the_dump(engine):
    seq = ["OPT", "SP", "BAND", "MATDUMP", "FREQ", "WANNIER"]
    assert engine._find_dependency_in_sequence("MATDUMP", seq) == "SP"
    assert engine._find_dependency_in_sequence("WANNIER", seq) == "MATDUMP"


def test_sweep_after_sp_fires_matdump_but_not_the_conversion(engine):
    engine.db = _FakeDB([_calc("OPT"), _calc("SP")])
    engine._calculation_already_exists = lambda mid, t: t in {"OPT", "SP"}
    engine._check_and_trigger_pending_calculations("mat", SEQ, skip_geometry_steps=True)
    assert engine.triggered == [("MATDUMP", "sp_1")]


def test_sweep_after_matdump_runs_the_conversion_from_that_dump(engine):
    engine.db = _FakeDB([_calc("OPT"), _calc("SP"), _calc("MATDUMP")])
    engine._calculation_already_exists = lambda mid, t: t in {"OPT", "SP", "MATDUMP"}
    engine._check_and_trigger_pending_calculations("mat", SEQ, skip_geometry_steps=True)
    assert engine.triggered == [("WANNIER", "matdump_1")]


def test_sp_completion_generates_the_planned_matdump(engine):
    sp = _calc("SP", cid="sp_1")
    engine.db = _FakeDB([_calc("OPT"), sp])
    engine._cleanup_failed_workflow_dirs = lambda: None
    engine.get_workflow_sequence = lambda wid: SEQ
    engine._calculation_already_exists = lambda mid, t: t in {"OPT", "SP"}
    engine._check_and_trigger_pending_calculations = lambda *a, **k: []

    engine.execute_workflow_step("mat", "sp_1")

    assert ("PROP_MATDUMP", "sp_1") in engine.triggered


def test_matdump_is_a_supported_property_calculation(engine, tmp_path):
    out = tmp_path / "mat_sp.out"
    out.write_text("x")
    (tmp_path / "mat_sp.f9").write_bytes(b"\x00" * 8)
    engine.db = _FakeDB([_calc("SP", output_file=str(out))])
    engine.db.calcs[0]["input_file"] = str(tmp_path / "mat_sp.d12")
    engine._find_most_recent_wavefunction_calc = lambda mid: "sp_1"
    script = tmp_path / "CRYSTALOptToD3.py"
    script.write_text("# stub\n")
    engine.script_paths = {"crystal_to_d3": script}
    del engine.generate_property_calculation      # the real one
    assert engine.generate_property_calculation("sp_1", "MATDUMP") == "id_MATDUMP"
    assert engine.triggered == [("MATDUMP", "sp_1")]


def test_the_default_matdump_config_loads_and_derives_n(engine):
    """An empty configuration block makes load_d3_config fail ("Failed to load
    configuration"), so MATDUMP needs a real default."""
    assert engine._get_default_d3_config("MATDUMP") == {
        "calculation_type": "MATDUMP", "n_rvectors": "auto"}


# --------------------------------------------------------------------------
# Engine: the local conversion post-step
# --------------------------------------------------------------------------


@pytest.fixture
def real_post_step(tmp_path):
    eng = WorkflowEngine.__new__(WorkflowEngine)
    dump = tmp_path / "mat_matdump.out"
    eng.db = _FakeDB([_calc("MATDUMP", output_file=str(dump))])
    return eng, dump


def _result(dump, audit="pass", produced=True):
    return driver.ConversionResult(
        returncode=0, stdout="", stderr="", command=[],
        output_dir=dump.parent / "mat_matdump.wannier", seed="mat_matdump",
        produced={"mat_matdump": list(driver.HANDOFF_SUFFIXES)} if produced else {},
        audit=audit, streamed=True)


def test_the_post_step_converts_the_dump_and_records_success(
        real_post_step, monkeypatch, capsys):
    engine, dump = real_post_step
    dump.write_text("OVERLAP MATRIX - CELL N.   1(  0  0  0)\n")
    seen = {}

    def fake_convert(parent, **kwargs):
        seen["parent"] = parent
        seen["kwargs"] = kwargs
        return _result(dump)

    monkeypatch.setattr(driver, "convert", fake_convert)
    calc_id = engine.run_wannier_post_step("matdump_1", "WANNIER")

    assert calc_id == "wannier_new"
    assert seen["parent"] == dump
    assert seen["kwargs"].get("echo") is not None        # progress is shown
    created = engine.db.created[0]
    assert created["calc_type"] == "WANNIER"
    assert created["prerequisite_calc_id"] == "matdump_1"
    assert engine.db.updates[-1][:2] == ("wannier_new", "completed")
    assert "William Comaskey" in capsys.readouterr().out


def test_a_refused_model_is_recorded_as_failed(real_post_step, monkeypatch, capsys):
    engine, dump = real_post_step
    dump.write_text("x")
    monkeypatch.setattr(driver, "convert", lambda parent, **kw: _result(dump, audit="fail"))
    assert engine.run_wannier_post_step("matdump_1", "WANNIER") is None
    assert engine.db.updates[-1][:2] == ("wannier_new", "failed")
    assert "mace wannier --input" in capsys.readouterr().out


def test_a_missing_dump_is_skipped_with_a_clear_message(real_post_step, monkeypatch,
                                                        capsys):
    engine, dump = real_post_step
    monkeypatch.setattr(driver, "convert",
                        lambda *a, **k: pytest.fail("must not convert"))
    assert engine.run_wannier_post_step("matdump_1", "WANNIER") is None
    text = capsys.readouterr().out
    assert "skipped" in text and str(dump.name) in text
    assert "mace wannier --input" in text
    assert engine.db.updates[-1][1] == "failed"
    assert "skipped" in engine.db.updates[-1][2]["error_message"]


def test_an_unrunnable_conversion_is_skipped_with_its_reason(real_post_step,
                                                             monkeypatch, capsys):
    engine, dump = real_post_step
    dump.write_text("x")

    def unavailable(*a, **k):
        raise driver.Lcao2WannierUnavailable("numpy/scipy missing")

    monkeypatch.setattr(driver, "convert", unavailable)
    assert engine.run_wannier_post_step("matdump_1", "WANNIER") is None
    text = capsys.readouterr().out
    assert "skipped" in text and "numpy/scipy missing" in text


def test_the_executor_accepts_a_plan_with_the_hand_off(tmp_path):
    """The executor validates every planned type before running a plan; a
    WANNIER step would make it reject the whole plan."""
    from mace.workflow.executor import WorkflowExecutor

    ex = WorkflowExecutor.__new__(WorkflowExecutor)
    (tmp_path / "in.d12").write_text("x\n")
    plan = {"workflow_sequence": SEQ, "input_directory": str(tmp_path),
            "input_type": "d12", "step_configurations": {}}
    assert ex._validate_workflow_plan(plan) is True
