"""The planner's Expert FREQ fallback writes what its message says.

When the interactive opt2d12 run behind Expert FREQ fails, the planner said
"using advanced settings" but wrote IR intensities and Raman activities
through CPHF, which neither Advanced's defaults nor the manual route give:
Expert is opt2d12's interactive FREQ route, whose Enter-through answer is no
intensities, or a FREQ parent's own. The fallback now writes that default and
says so.
"""
import pytest

import test_freq_child_keeps_parent_intensities as intens
from mace.workflow.planner import WorkflowPlanner


def _fallback(tmp_path, monkeypatch):
    planner = WorkflowPlanner(work_dir=tmp_path / "plan")
    monkeypatch.setattr(planner, "_copy_required_scripts_for_expert_mode", lambda: None)
    monkeypatch.setattr(planner, "_run_interactive_crystal_opt_config", lambda *a, **k: None)
    replies = iter(["3", "2"])                       # Expert, batch uniform
    monkeypatch.setattr("builtins.input", lambda prompt="": next(replies, ""))
    return planner.configure_frequency_step("FREQ", 3)


def test_fallback_message_names_what_it_writes(tmp_path, monkeypatch, capsys):
    step = _fallback(tmp_path, monkeypatch)
    said = capsys.readouterr()
    said = said.out + said.err
    assert "using advanced settings" not in said
    assert "opt2d12's FREQ defaults" in said
    assert not {"intensities", "raman", "ir_method", "cphf_max_iter"} & set(
        step["frequency_settings"]), step


def test_fallback_after_an_opt_parent_is_the_manual_default(tmp_path, monkeypatch):
    """Before: INTENS / INTRAMAN / INTCPHF from a parent that asked for none."""
    pytest.importorskip("numpy")
    step = _fallback(tmp_path, monkeypatch)
    assert intens._workflow_child(tmp_path, "FREQCALC\nNOINTENS\nENDFREQ", step) == [
        "FREQCALC", "NOINTENS", "ENDFREQ"]


def test_fallback_after_a_freq_parent_keeps_its_intensities(tmp_path, monkeypatch):
    pytest.importorskip("numpy")
    step = _fallback(tmp_path, monkeypatch)
    assert intens._workflow_child(tmp_path, intens.IR_RAMAN_PARENT, step) == intens.IR_RAMAN_CHILD
