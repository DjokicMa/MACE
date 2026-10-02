"""A planned 2c restart ("soc_guess") through the workflow engine.

The engine runs opt2d12 in a throwaway directory and moves the deck it wrote
into the step directory under the job's name. A GUESSPSO/GUESSPNOSO deck has
its parent's .f9 staged beside it as <deck stem>.f20 - the file the job script
copies to fort.20 - and CRYSTAL stops on GUESSPSO without it (measured). So
the engine copies the parent's .f9 into the throwaway directory (opt2d12
refuses the deck without it) and moves the .f20 with the deck, renamed to the
job; a 2c restart deck without one fails the step.
"""
import json
from pathlib import Path

import pytest

from mace.workflow.engine import WorkflowEngine
from test_guessp_engine_chain import EXT_BASIS, F9, HAM, SCF, _FakeDB, _geom

WF_ID = "workflow_20260930_120000"
SOC_SCF = SCF.replace("SCFDIR\n", "TWOCOMPON\nSOC\nGUESSPSO\nEND\nSCFDIR\n")
PARENT_DECK = "pto_sp\n" + _geom() + EXT_BASIS + HAM + SCF.replace("SCFDIR\n", "TWOCOMPON\nSOC\nEND\nSCFDIR\n")
CHILD_DECK = "pto_sp2\n" + _geom(a="3.95012345") + EXT_BASIS + HAM + SOC_SCF


def _engine(tmp_path, step_cfg, f9=F9):
    src = tmp_path / "parent"
    src.mkdir()
    (src / "parent.d12").write_text(PARENT_DECK)
    (src / "parent.out").write_text("fake CRYSTAL output\n")
    if f9 is not None:
        (src / "parent.f9").write_bytes(f9)
    cfg = tmp_path / "workflow_configs"
    cfg.mkdir()
    (cfg / f"workflow_plan_{WF_ID.replace('workflow_', '')}.json").write_text(json.dumps(
        {"workflow_id": WF_ID, "workflow_sequence": ["SP", "SP2"],
         "step_configurations": {"SP2_2": step_cfg}}))
    calc = {"calc_id": "c1", "material_id": "pto", "status": "completed", "calc_type": "SP",
            "output_file": str(src / "parent.out"), "input_file": str(src / "parent.d12"),
            "work_dir": str(src),
            "settings_json": json.dumps({"workflow_id": WF_ID, "workflow_step": 1})}
    eng = WorkflowEngine(db_path=str(tmp_path / "m.db"), base_work_dir=str(tmp_path),
                         auto_submit=False)
    eng.db = _FakeDB(calc)
    return eng


def _fake_opt2d12(eng, seen, stage=True, deck_text=CHILD_DECK):
    """Stand-in for CRYSTALOptToD12: what it does with soc_guess - read the
    config, find <parent stem>.f9 beside the .out, write the deck and stage
    the .f9 as <deck stem>.f20."""
    def run(script_path, work_dir, args=None, input_data=None):
        work_dir = Path(work_dir)
        config = json.loads(Path(next(a for a in args if a.endswith("_temp_config.json")))
                            .read_text())
        seen["config"] = config
        seen["f9"] = (work_dir / "parent.f9").read_bytes() if (work_dir / "parent.f9").exists() else None
        deck = work_dir / "pto_sp_generated.d12"
        deck.write_text(deck_text)
        if stage and seen["f9"]:
            deck.with_suffix(".f20").write_bytes(seen["f9"])
        return True, "", ""
    eng.run_script_in_isolated_directory = run


def test_the_parent_matrix_reaches_opt2d12_and_moves_with_the_deck(tmp_path, capsys):
    eng = _engine(tmp_path, {"calculation_type": "SP", "soc": True, "soc_guess": "guesspso"})
    seen = {}
    _fake_opt2d12(eng, seen)
    assert eng.generate_numbered_calculation("c1", "SP2")
    assert seen["config"]["soc_guess"] == "guesspso" and seen["config"]["soc"] is True
    assert seen["f9"] == F9
    deck = Path(eng.db.created[-1]["input_file"])
    assert deck.read_text() == CHILD_DECK
    f20 = deck.with_suffix(".f20")
    assert f20.read_bytes() == F9 and f20.stem == deck.stem
    assert "GUESSPSO: the parent's density matrix moved with the deck" in capsys.readouterr().out


def test_a_2c_restart_deck_without_its_matrix_fails_the_step(tmp_path, capsys):
    eng = _engine(tmp_path, {"calculation_type": "SP", "soc": True, "soc_guess": "guesspso"})
    _fake_opt2d12(eng, {}, stage=False)
    assert eng.generate_numbered_calculation("c1", "SP2") is None
    assert eng.db.created == []
    assert not list(tmp_path.rglob("step_*/**/*.d12"))
    assert "CRYSTAL stops on GUESSPSO with nothing in fort.20 (measured)" in capsys.readouterr().out


def test_without_soc_guess_no_matrix_is_copied(tmp_path):
    eng = _engine(tmp_path, {"calculation_type": "SP", "soc": True})
    seen = {}
    # a plain SOC deck: no GUESSPSO record, nothing staged
    _fake_opt2d12(eng, seen, deck_text=CHILD_DECK.replace("GUESSPSO\n", ""))
    assert eng.generate_numbered_calculation("c1", "SP2")
    assert seen["f9"] is None and "soc_guess" not in seen["config"]
    deck = Path(eng.db.created[-1]["input_file"])
    assert not deck.with_suffix(".f20").exists()


def test_the_plan_hands_the_soc_options_to_the_deck_writer(tmp_path):
    eng = _engine(tmp_path, {"calculation_type": "SP", "soc": True, "soc_guess": "guesspso",
                             "soc_ecp": "stutsc", "soc_basis": "pob-tzvp-rev2-soc"})
    assert eng._build_numbered_calc_config(WF_ID, "SP2", "SP", None) == {
        "calculation_type": "SP", "soc": True, "soc_guess": "guesspso", "soc_ecp": "stutsc",
        "soc_basis": "pob-tzvp-rev2-soc"}
