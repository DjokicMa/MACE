"""FREQ decks default to Very tight SCF tolerances (9 9 9 11 38 / TOLDEE 11).

This is the one deliberate exception to "a derived deck reproduces its
parent": frequencies need tighter SCF convergence than the optimization they
follow. It holds on every path that writes a FREQ deck with no tolerances
named - the opt2d12 menu, --config-file (the engine's no-plan FREQ step),
--non-interactive, "Use these exact settings", and the planner's FREQ presets
(including the expert fallback, which used 12 12 12 12 24 / 12). Explicit
tolerances from the user, plan or config still win, and a parent that is
itself a FREQ calculation keeps its own.

The deck tests run the real ``mace_cli opt2d12`` on the corpus and skip when
``test/`` is absent.
"""
import json
import shutil
import subprocess
import sys

import pytest

import d12_constants
import test_opt2d12_prompt_defaults as prompts
from conftest import TEST_DATA
from d12_constants import freq_default_tolerances, scf_tolerances
from mace.workflow.planner import WorkflowPlanner

OPT_PARENT = "OPT/1_dia_opt_rev1"                  # TOLINTEG 7 7 7 7 14, TOLDEE 7
FREQ_PARENT = "FREQ/1_dia_opt_rev1_freq_B3LYP-D3-D3_optimized_supercel222"
VERY_TIGHT = ["9", "9", "9", "11", "38"]


def _scf(lines):
    return prompts._scf_after(lines, "TOLINTEG").split(), prompts._scf_after(lines, "TOLDEE")


def _config(tmp_path, cfg):
    (tmp_path / "cfg.json").write_text(json.dumps(cfg))
    return ["--config-file", "cfg.json"]


def test_helper():
    assert freq_default_tolerances() == scf_tolerances("3")
    assert freq_default_tolerances("OPT", scf_tolerances("1")) == scf_tolerances("3")
    assert freq_default_tolerances("FREQ", scf_tolerances("2")) == scf_tolerances("2")
    assert freq_default_tolerances("FREQ", None) == scf_tolerances("3")


def test_config_without_tolerances(tmp_path):
    """The engine's no-plan FREQ step: config {"calculation_type": "FREQ"}."""
    name = prompts._copy_parent(OPT_PARENT, tmp_path)
    child = prompts._opt2d12(tmp_path, name, stdin="y\n" + "\n" * 17,
                             extra=_config(tmp_path, {"calculation_type": "FREQ"}))
    assert "FREQCALC" in child
    assert _scf(child) == (VERY_TIGHT, "11")


def test_config_tolerances_win(tmp_path):
    name = prompts._copy_parent(OPT_PARENT, tmp_path)
    cfg = {"calculation_type": "FREQ",
           "frequency_settings": {"mode": "GAMMA",
                                  "custom_tolerances": {"TOLINTEG": "8 8 8 9 24", "TOLDEE": 9}}}
    child = prompts._opt2d12(tmp_path, name, stdin="y\n" + "\n" * 17, extra=_config(tmp_path, cfg))
    assert _scf(child) == ("8 8 8 9 24".split(), "9")



@pytest.mark.parametrize("cfg", [
    {"calculation_type": "FREQ", "tolerances": {"TOLDEE": 12}},
    {"calculation_type": "FREQ", "tolerance_modifications": {"custom_tolerances": {"TOLDEE": 12}}},
    {"calculation_type": "FREQ", "frequency_settings": {"mode": "GAMMA",
                                                        "custom_tolerances": {"TOLDEE": 12}}},
])
def test_config_naming_only_toldee_keeps_very_tight_tolinteg(tmp_path, cfg):
    """A partial FREQ override is laid over the FREQ default, not over the
    optimization's tolerances (which gave TOLINTEG 7 7 7 7 14 / TOLDEE 12)."""
    name = prompts._copy_parent(OPT_PARENT, tmp_path)
    child = prompts._opt2d12(tmp_path, name, stdin="y\n" + "\n" * 17, extra=_config(tmp_path, cfg))
    assert _scf(child) == (VERY_TIGHT, "12")

def test_non_interactive_with_nothing_on_stdin(tmp_path):
    name = prompts._copy_parent(OPT_PARENT, tmp_path)
    child = prompts._opt2d12(tmp_path, name, stdin="",
                             extra=["--calc-type", "FREQ", "--non-interactive"])
    assert "FREQCALC" in child
    assert _scf(child) == (VERY_TIGHT, "11")


def test_exact_settings_with_calc_type_freq(tmp_path):
    """"Use these exact settings" with --calc-type FREQ wrote an OPT deck."""
    name = prompts._copy_parent(OPT_PARENT, tmp_path)
    child = prompts._opt2d12(tmp_path, name, {"exact settings": "y"}, extra=["--calc-type", "FREQ"])
    assert "FREQCALC" in child and "OPTGEOM" not in child
    assert _scf(child) == (VERY_TIGHT, "11")
    assert "B3LYP-D3" in child


def test_menu_answer_still_wins(tmp_path):
    name = prompts._copy_parent(OPT_PARENT, tmp_path)
    child = prompts._opt2d12(tmp_path, name, {"exact settings": "n", "calculation type": "3",
                                              "SCF convergence level": "1"})
    assert _scf(child) == ("7 7 7 7 14".split(), "7")


def _tight_freq_parent(tmp_path):
    """A real FREQ parent, its SCF tolerances set to Tight."""
    src = TEST_DATA / f"{FREQ_PARENT}.d12"
    if not src.exists():
        pytest.skip("test/ corpus not present (gitignored, ~12GB)")
    lines = src.read_text().splitlines()
    i = len(lines) - 1 - lines[::-1].index("TOLINTEG")
    lines[i + 1] = "8 8 8 9 24"
    j = len(lines) - 1 - lines[::-1].index("TOLDEE")
    lines[j + 1] = "9"
    (tmp_path / "freqparent.d12").write_text("\n".join(lines) + "\n")
    shutil.copy(src.with_suffix(".out"), tmp_path / "freqparent.out")
    return "freqparent"


def test_freq_parent_keeps_its_own_tolerances_with_a_config(tmp_path):
    name = _tight_freq_parent(tmp_path)
    child = prompts._opt2d12(tmp_path, name, stdin="y\n" + "\n" * 17,
                             extra=_config(tmp_path, {"calculation_type": "FREQ"}))
    assert _scf(child) == ("8 8 8 9 24".split(), "9")


def test_freq_parent_keeps_its_own_tolerances_on_blank_answers(tmp_path):
    name = _tight_freq_parent(tmp_path)
    child = prompts._opt2d12(tmp_path, name, {"exact settings": "n", "calculation type": "3"})
    assert _scf(child) == ("8 8 8 9 24".split(), "9")


def test_planner_expert_fallback_is_very_tight(tmp_path, monkeypatch):
    planner = WorkflowPlanner(work_dir=tmp_path)
    monkeypatch.setattr(planner, "_copy_required_scripts_for_expert_mode", lambda: None)
    monkeypatch.setattr(planner, "_run_interactive_crystal_opt_config", lambda *a, **k: None)
    replies = iter(["3", "2"])                       # Expert, batch uniform
    monkeypatch.setattr("builtins.input", lambda prompt="": next(replies, ""))
    cfg = planner.configure_frequency_step("FREQ", 3)
    assert cfg["frequency_settings"]["custom_tolerances"] == scf_tolerances("3")


def test_planner_basic_freq_is_very_tight(tmp_path, monkeypatch):
    planner = WorkflowPlanner(work_dir=tmp_path)
    monkeypatch.setattr("builtins.input", lambda prompt="": "1")
    cfg = planner.configure_frequency_step("FREQ", 3)
    assert cfg["frequency_settings"]["custom_tolerances"] == scf_tolerances("3")
