"""Plan fidelity: progression-generated calculations must follow the workflow
plan's step_configurations.

Real-world failure (testMACE1.1 run, 2026-07-02..06): the callback-driven
engine consulted only per-material expert config FILES (which the planner
writes for FREQ) and fell back to engine defaults / pure inheritance for
everything else. Observed deviations from the user's saved plan:
  - SP planned with new_functional B3LYP (+dispersion) silently re-ran PBE-D3
  - BAND planned with 1000-point seekpath paths ran the 10000-point default
    table path
  - DOSS energy window was dropped (band-index mode used instead)
  - OPT2 optimization_settings came from inheritance, not the plan
    (its TOLDEE 9 vs the plan's 7 proved it)

Now generate_d3_calculation_new and the numbered-d12 path resolve settings
with plan-aware precedence: expert_config_file -> inline d3_settings/d3_config
/ method_modifications / optimization_settings -> engine defaults.

The plan JSON here mirrors the user's real workflow_plan_20260702_193809.json.
"""
import json
from pathlib import Path

import pytest

from mace.workflow.engine import WorkflowEngine

WF_ID = "workflow_20260702_193809"

PLAN = {
    "workflow_id": WF_ID,
    "workflow_sequence": ["OPT", "SP", "BAND", "DOSS", "FREQ",
                          "CHARGE+POTENTIAL", "OPT2"],
    "step_configurations": {
        "SP_2": {
            "calculation_type": "SP",
            "inherit_settings": False,
            "method_modifications": {"new_functional": "B3LYP",
                                     "use_dispersion": True},
            "basis_modifications": {"inherit_basis": True},
        },
        "BAND_3": {
            "calculation_type": "BAND",
            "d3_config_mode": "expert",
            "expert_config_file": "/mnt/OTHER-MACHINE/expert_d3_configs/d3_band_step_3_config.json",
            "d3_settings": {
                "version": "1.0", "type": "d3_configuration",
                "calculation_type": "BAND",
                "configuration": {"path_method": "coordinates", "n_points": 1000,
                                  "kpath_source": "seekpath_inv", "shrink": "auto",
                                  "labels": "auto", "auto_path": True},
            },
        },
        "DOSS_4": {
            "calculation_type": "DOSS",
            "d3_settings": {
                "version": "1.0", "type": "d3_configuration",
                "calculation_type": "DOSS",
                "configuration": {"projection_type": 3, "n_points": 10000,
                                  "npol": 14,
                                  "energy_window": [-0.3675, 0.7350]},
            },
        },
        "CHARGE+POTENTIAL_6": {
            "calculation_type": "CHARGE+POTENTIAL",
            "d3_config": {
                "calculation_type": "CHARGE+POTENTIAL",
                "charge_config": {"type": "ECH3", "n_points": 150},
                "potential_config": {"type": "POT3", "n_points": 150},
            },
        },
        "OPT2_7": {
            "calculation_type": "OPT",
            "optimization_type": "FULLOPTG",
            "optimization_settings": {"TOLDEG": 3e-05, "TOLDEX": 0.00012,
                                      "TOLDEE": 7, "MAXCYCLE": 800},
            "inherit_settings": True,
        },
    },
}


@pytest.fixture
def engine(tmp_path):
    eng = WorkflowEngine.__new__(WorkflowEngine)
    eng.base_work_dir = tmp_path
    cfg_dir = tmp_path / "workflow_configs"
    cfg_dir.mkdir()
    (cfg_dir / f"workflow_plan_{WF_ID.replace('workflow_', '')}.json").write_text(
        json.dumps(PLAN))
    return eng


# ---------------------------------------------------------------------------
# Step-config resolution
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("calc_type,expect_key", [
    ("SP", "method_modifications"),
    ("BAND", "d3_settings"),
    ("DOSS", "d3_settings"),
    ("CHARGE+POTENTIAL", "d3_config"),
    ("OPT2", "optimization_settings"),
])
def test_plan_step_config_matched_by_type(engine, calc_type, expect_key):
    cfg = engine._get_plan_step_config(WF_ID, calc_type)
    assert expect_key in cfg, f"{calc_type} did not resolve its plan step config"


def test_plan_step_config_absent_type_and_absent_plan(engine):
    assert engine._get_plan_step_config(WF_ID, "TRANSPORT") == {}
    assert engine._get_plan_step_config("workflow_nonexistent", "BAND") == {}
    assert engine._get_plan_step_config(None, "BAND") == {}


# ---------------------------------------------------------------------------
# Numbered-d12 config (SP method mods, OPT2 optimization settings)
# ---------------------------------------------------------------------------
def test_sp_config_applies_plan_functional_switch(engine):
    """The exact deviation from the real run: parent deck says PBE-D3, the
    plan says B3LYP + dispersion — the plan must win."""
    cfg = engine._build_numbered_calc_config(WF_ID, "SP", "SP", "PBE-D3")
    assert cfg["functional"] == "B3LYP-D3"
    assert cfg["dispersion"] is True
    assert cfg["calculation_type"] == "SP"


def test_opt2_config_applies_plan_optimization_settings(engine):
    cfg = engine._build_numbered_calc_config(WF_ID, "OPT2", "OPT", "PBE-D3")
    assert cfg["optimization_settings"]["TOLDEE"] == 7      # plan, not inherited 9
    assert cfg["optimization_settings"]["TOLDEG"] == 3e-05
    assert cfg["optimization_type"] == "FULLOPTG"
    assert cfg["functional"] == "PBE-D3"                    # inheritance kept


def test_no_plan_falls_back_to_extracted_functional(engine):
    cfg = engine._build_numbered_calc_config("workflow_nonexistent", "SP", "SP", "PBE-D3")
    assert cfg == {"calculation_type": "SP", "functional": "PBE-D3"}
    assert engine._build_numbered_calc_config("workflow_nonexistent", "SP", "SP", None) is None


# ---------------------------------------------------------------------------
# D3 generation config (BAND/DOSS/C+P) through the real generation path
# ---------------------------------------------------------------------------
def _run_d3_generation(engine, tmp_path, monkeypatch, calc_type, plan_wf_id=WF_ID):
    """Drive generate_d3_calculation_new with a faked CRYSTALOptToD3 run and
    capture the --config-file content it was handed."""
    import mace.workflow.engine as eng_mod

    wf_out = tmp_path / "workflow_outputs" / plan_wf_id
    wf_out.mkdir(parents=True, exist_ok=True)

    class _DB:
        def get_calculation(self, cid):
            return {"calc_id": cid, "material_id": "1_dia", "status": "completed",
                    "output_file": str(tmp_path / "1_dia_sp.out"),
                    "settings_json": json.dumps({"workflow_id": plan_wf_id})}

    engine.db = _DB()
    engine._find_most_recent_wavefunction_calc = lambda mid: "wf_calc_1"
    engine.get_workflow_output_base = lambda calc: wf_out
    fake_script = tmp_path / "CRYSTALOptToD3.py"
    fake_script.write_text("# stub\n")
    engine.script_paths = {"crystal_to_d3": str(fake_script)}
    engine._create_and_submit_d3_calculation = lambda *a: "calc_ok"

    captured = {}

    def fake_run(cmd, **kwargs):
        cfg_path = Path(cmd[cmd.index("--config-file") + 1])
        captured["config"] = json.loads(cfg_path.read_text())
        captured["config_path"] = cfg_path
        out_dir = Path(cmd[cmd.index("--output-dir") + 1])
        stem = f"1_dia_sp_{calc_type.lower()}"
        (out_dir / f"{stem}.d3").write_text("BAND\nEND\n")
        (out_dir / f"{stem}.f9").write_bytes(b"\x00")

        class R:
            returncode = 0
            stdout = stderr = ""
        return R()

    monkeypatch.setattr(eng_mod.subprocess, "run", fake_run)
    assert engine.generate_d3_calculation_new("src_sp_1", calc_type) == "calc_ok"
    return captured


def test_band_generation_uses_plan_d3_settings(engine, tmp_path, monkeypatch):
    """The real deviation: BAND ran 10000-point default paths; the plan said
    1000-point seekpath. (The plan's expert_config_file points at another
    machine and doesn't exist locally -> inline d3_settings are used.)"""
    cap = _run_d3_generation(engine, tmp_path, monkeypatch, "BAND")
    conf = cap["config"]["configuration"]
    assert conf["n_points"] == 1000
    assert conf["kpath_source"] == "seekpath_inv"


def test_band_expert_file_local_fallback(engine, tmp_path, monkeypatch):
    """When the plan's (foreign-path) expert_config_file exists under this
    run's workflow_configs/expert_d3_configs, that file is used directly."""
    exp_dir = tmp_path / "workflow_configs" / "expert_d3_configs"
    exp_dir.mkdir(parents=True)
    exp = exp_dir / "d3_band_step_3_config.json"   # basename from PLAN's pointer
    exp.write_text(json.dumps({
        "version": "1.0", "type": "d3_configuration", "calculation_type": "BAND",
        "configuration": {"n_points": 777, "path_method": "coordinates"}}))

    cap = _run_d3_generation(engine, tmp_path, monkeypatch, "BAND")
    assert cap["config_path"] == exp
    assert cap["config"]["configuration"]["n_points"] == 777


def test_cp_generation_wraps_plan_d3_config(engine, tmp_path, monkeypatch):
    cap = _run_d3_generation(engine, tmp_path, monkeypatch, "CHARGE+POTENTIAL")
    conf = cap["config"]["configuration"]
    assert conf["charge_config"]["n_points"] == 150       # plan, not default 100
    assert cap["config"]["type"] == "d3_configuration"


def test_no_plan_uses_engine_defaults(engine, tmp_path, monkeypatch):
    """Without a matching plan the engine defaults still apply (BAND 10000)."""
    cap = _run_d3_generation(engine, tmp_path, monkeypatch, "BAND",
                             plan_wf_id="workflow_no_plan_here")
    assert cap["config"]["configuration"]["n_points"] == 10000
