"""A d12 step's own expert_config_file in the plan is what its deck is built from.

BAND/DOSS/TRANSPORT steps honored the plan step's ``expert_config_file``
first; the d12 steps (SP, including the first one after the OPT, OPTn, SPn
and FREQ) ignored it. The first SP looked for no expert configuration at all
and the others only in the conventional expert_<type>_configs/ folders, so a
plan whose SP step named a configuration with another functional ran the
parent's method. The d12 steps now take the same precedence as the D3 ones:
the step's expert_config_file, then what they used before.
"""
import json

import pytest

import test_numbered_calc_keeps_parent_method as base

PBE0_SP = {"calculation_type": "SP", "functional": "PBE0"}


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return path


@pytest.mark.parametrize("target,key,base_type,stdin", [
    ("SP", "SP_2", "SP", "y\n"),
    ("SP2", "SP2_4", "SP", "y\n"),
    ("OPT2", "OPT2_4", "OPT", "y\n"),
    ("FREQ", "FREQ_3", "FREQ", "y\n" + "\n" * 17)])
def test_the_steps_expert_config_file_is_used(tmp_path, target, key, base_type, stdin):
    cfg = _write(tmp_path / "mine" / f"{target.lower()}.json",
                 {"calculation_type": base_type, "functional": "PBE0"})
    step = {"calculation_type": base_type, "expert_mode": True,
            "expert_config_file": str(cfg),
            # inline settings the expert file takes precedence over
            "method_modifications": {"new_functional": "B3LYP"}}
    eng = base._engine(tmp_path, base.DECK.format(ham="DFT\nSPIN\nHSE06\nENDDFT"),
                       "dummy\n", {key: step})
    seen = base._numbered(eng, target)
    assert seen["config"] == {"calculation_type": base_type, "functional": "PBE0"}
    assert seen["args"][seen["args"].index("--config-file") + 1] == str(cfg)
    assert seen["stdin"] == stdin


def test_a_path_from_another_machine_is_found_in_workflow_configs(tmp_path):
    step = {"calculation_type": "SP",
            "expert_config_file": "/elsewhere/workflow_configs/expert_sp_configs/custom_sp.json"}
    eng = base._engine(tmp_path, base.DECK.format(ham="DFT\nSPIN\nHSE06\nENDDFT"),
                       "dummy\n", {"SP_2": step})
    _write(tmp_path / "workflow_configs" / "expert_sp_configs" / "custom_sp.json", PBE0_SP)
    assert base._numbered(eng, "SP")["config"] == PBE0_SP


def test_a_missing_expert_file_falls_back_to_the_steps_settings(tmp_path):
    step = {"calculation_type": "SP", "expert_config_file": str(tmp_path / "gone.json"),
            "method_modifications": {"new_functional": "PBE0"}}
    eng = base._engine(tmp_path, base.DECK.format(ham="DFT\nSPIN\nHSE06\nENDDFT"),
                       "dummy\n", {"SP_2": step})
    cfg = base._numbered(eng, "SP")["config"]
    assert cfg["functional"] == "PBE0" and cfg["calculation_type"] == "SP"


def test_without_one_the_parent_method_is_kept(tmp_path):
    """Follow-up steps keep the parent's settings when the plan changes none."""
    eng = base._engine(tmp_path, base.DECK.format(ham="DFT\nSPIN\nPBE0\nENDDFT"), "dummy\n",
                       {"SP_2": {"calculation_type": "SP", "inherit_settings": True}})
    cfg = base._numbered(eng, "SP")["config"]
    assert cfg == {"calculation_type": "SP", "inherit_settings": True}


def test_real_sp_deck_has_the_expert_files_functional(tmp_path):
    """Corpus: B3LYP-D3 diamond OPT; the plan's SP configuration says PBE0."""
    base._need(base.DIA)
    cfg = _write(tmp_path / "mine" / "sp.json", {"calculation_type": "SP", "functional": "PBE0"})
    eng = base._engine(tmp_path, base.DIA.with_suffix(".d12").read_text(),
                       base.DIA.with_suffix(".out").read_text(),
                       {"SP_2": {"calculation_type": "SP", "expert_config_file": str(cfg)}})
    name, deck = base._only_deck(base._numbered(eng, "SP", True))
    body = deck.splitlines()
    assert "PBE0" in body and "B3LYP" not in body
    assert "OPTGEOM" not in body


def test_callback_submits_the_sp_deck_the_expert_file_describes(tmp_path):
    """Through the CLI and the real job callback: the OPT of a workflow whose
    SP step names a PBE0 configuration completes, and the SP submitted next
    is a PBE0 deck."""
    import re
    import shutil
    from pathlib import Path

    from array_helpers import callback, rows
    from conftest import find_data
    from fake_slurm import FakeSlurm
    from test_array_default_path_golden import _env, _mace
    from test_workflow_isolation_scripts import _ok, _seed_plan, _write_plan

    out = find_data("OPT/1_dia_opt_rev1.out")
    deck = find_data("OPT/1_dia_opt_rev1.d12")
    fake = FakeSlurm(tmp_path)
    env = _env(tmp_path, fake)
    template = _seed_plan(tmp_path, env, deck)
    root = tmp_path / "wf"
    cfg = _write(tmp_path / "mine" / "sp.json", {"calculation_type": "SP", "functional": "PBE0"})
    template["step_configurations"]["SP_2"]["expert_config_file"] = str(cfg)
    _write_plan(root, template, deck, "one", "workflow_20991231_000003", 16)
    _ok(_mace(root, env, "workflow", "--execute", "plan_one.json"))

    db = root / ".mace_context_workflow_20991231_000003" / "materials.db"
    (opt,) = [r for r in rows(db).values() if r["calc_type"] == "OPT"]
    work = Path(opt["input_file"]).parent
    stem = Path(opt["input_file"]).stem
    shutil.copy(out, work / f"{stem}.out")
    exports = dict(re.findall(r'^export (MACE_WORKFLOW_ID|MACE_CONTEXT_DIR|MACE_ISOLATION_MODE)'
                              r'="([^"]*)"', (work / f"{stem}.sh").read_text(), re.M))
    fake.set_record(opt["slurm_job_id"], state="COMPLETED")
    p = _ok(callback(env, work, db, mode="completion",
                     extra_env=dict(exports, SLURM_JOB_ID=opt["slurm_job_id"])))
    (sp,) = [r for r in rows(db).values() if r["calc_type"] == "SP"]
    assert sp["slurm_job_id"], p.stdout[-4000:]
    body = Path(sp["input_file"]).read_text().splitlines()
    assert "PBE0" in body and "B3LYP" not in body
