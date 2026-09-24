"""NUMDERIV is written only when the user, plan or config asks for it.

opt2d12's FREQ prompt already defaults to no NUMDERIV (CRYSTAL's default
difference formula). The planner's FREQ presets, its advanced prompt
(default 2), its expert fallback, the quick-start FREQ step and the FREQ
templates all wrote NUMDERIV 2 on their own.
"""
import argparse

import pytest

import d12_calc_freq
import d12_constants
import test_opt2d12_prompt_defaults as prompts
from mace.workflow.planner import WorkflowPlanner


@pytest.mark.parametrize("module", [d12_constants, d12_calc_freq])
def test_templates_and_defaults_do_not_force_numderiv(module):
    for name, template in module.FREQ_TEMPLATES.items():
        assert "numderiv" not in template, name
    assert "NUMDERIV" not in module.DEFAULT_FREQ_SETTINGS


def _freq_step(tmp_path, monkeypatch, level, numderiv_answer=""):
    planner = WorkflowPlanner(work_dir=tmp_path)
    monkeypatch.setattr(planner, "_copy_required_scripts_for_expert_mode", lambda: None)
    monkeypatch.setattr(planner, "_run_interactive_crystal_opt_config", lambda *a, **k: None)

    def answer(prompt=""):
        if "customization level" in prompt:
            return level
        if "Choose configuration mode" in prompt:
            return "2"
        if "Select method (0-2)" in prompt:
            return numderiv_answer
        return ""

    monkeypatch.setattr("builtins.input", answer)
    return planner.configure_frequency_step("FREQ", 3)["frequency_settings"]


@pytest.mark.parametrize("level", ["1", "2", "3"])
def test_planner_freq_steps_leave_numderiv_out(tmp_path, monkeypatch, level):
    assert "numderiv" not in _freq_step(tmp_path, monkeypatch, level)


@pytest.mark.parametrize("answer", ["1", "2"])
def test_planner_advanced_numderiv_answer_is_kept(tmp_path, monkeypatch, answer):
    assert _freq_step(tmp_path, monkeypatch, "2", answer)["numderiv"] == int(answer)


def test_quick_start_freq_leaves_numderiv_out(tmp_path):
    from mace import run_mace
    plan = run_mace.create_quick_workflow_plan(tmp_path, [tmp_path / "a.d12"], "d12",
                                               ["OPT", "FREQ"], argparse.Namespace(max_jobs=1))
    assert "numderiv" not in plan["step_configurations"]["FREQ_2"]["frequency_settings"]


def test_real_freq_deck_without_a_request_has_no_numderiv(tmp_path):
    """The engine's no-plan FREQ step: config {"calculation_type": "FREQ"}."""
    name = prompts._copy_parent("OPT/1_dia_opt_rev1", tmp_path)
    (tmp_path / "cfg.json").write_text('{"calculation_type": "FREQ"}')
    child = prompts._opt2d12(tmp_path, name, stdin="y\n" + "\n" * 17,
                             extra=["--config-file", "cfg.json"])
    assert "FREQCALC" in child and "NUMDERIV" not in child


# ------------------------------------------------ a FREQ parent's NUMDERIV

FREQ_PARENT = "FREQ/1_dia_opt_rev1_freq_B3LYP-D3-D3_optimized_supercel222"


def _freq_parent_with_numderiv(tmp_path, value=1):
    """The corpus FREQ deck with 'NUMDERIV / <value>' added to its FREQCALC block."""
    name = prompts._copy_parent(FREQ_PARENT, tmp_path)
    deck = tmp_path / f"{name}.d12"
    text = deck.read_text()
    assert "FREQCALC\n" in text
    deck.write_text(text.replace("FREQCALC\n", f"FREQCALC\nNUMDERIV\n{value}\n", 1))
    return name


def test_parser_reads_numderiv_under_the_writers_key(tmp_path):
    from d12_parsers import CrystalInputParser
    name = _freq_parent_with_numderiv(tmp_path, 1)
    freq = CrystalInputParser(str(tmp_path / f"{name}.d12")).parse()["freq_settings"]
    assert freq == {"numderiv": 1}


def test_freq_parent_numderiv_kept_non_interactive(tmp_path):
    name = _freq_parent_with_numderiv(tmp_path, 1)
    child = prompts._opt2d12(tmp_path, name, stdin="",
                             extra=["--non-interactive", "--calc-type", "FREQ"])
    assert prompts._after(child, "NUMDERIV") == "1"


def test_freq_parent_numderiv_is_the_prompt_default(tmp_path):
    name = _freq_parent_with_numderiv(tmp_path, 1)
    child = prompts._opt2d12(tmp_path, name, answers={"exact settings": "n",
                                                      "calculation type": "3"})
    assert prompts._after(child, "NUMDERIV") == "1"


def test_answer_still_overrides_the_parents_numderiv(tmp_path):
    name = _freq_parent_with_numderiv(tmp_path, 1)
    child = prompts._opt2d12(tmp_path, name, answers={"exact settings": "n",
                                                      "calculation type": "3",
                                                      "Select method (0-2)": "0"})
    assert "NUMDERIV" not in child
