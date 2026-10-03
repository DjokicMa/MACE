"""A FREQ child of a FREQ parent that computed IR/Raman intensities keeps them.

opt2d12 --non-interactive already repeats the parent's FREQCALC block, but two
routes dropped its intensity records (INTENS, INTCPHF, INTRAMAN, IRSPEC, ...):

- the interactive opt2d12 FREQ route, whose template menu defaulted to
  "1: Basic frequencies only", so Enter-through wrote NOINTENS;
- the workflow, whose Basic / quick-start FREQ steps wrote
  ``intensities: False`` into the plan as if the user had chosen it, and whose
  frequency_settings replaced the parent's block.

Following the parent unless the user changes it: the template menu now offers
"0: Keep the parent's FREQCALC settings" as its default when the parent has
some, the default plan steps name no intensity setting, and a config that
names none keeps the parent's. A config or answer that names one still wins.
"""
import shutil
import subprocess
import sys

import pytest

import test_freqcalc_parent_settings as agbr
import test_numbered_calc_keeps_parent_method as base
from conftest import REPO_ROOT
from mace.workflow.planner import WorkflowPlanner

IR_PARENT = "FREQCALC\nINTENS\nINTCPHF\nENDCPHF\nENDFREQ"
IR_RAMAN_PARENT = ("FREQCALC\nINTENS\nINTRAMAN\nINTCPHF\nENDCPHF\n"
                   "IRSPEC\nRANGE\n0.0 4000.0\nEND\nENDFREQ")
IR_RAMAN_CHILD = ["FREQCALC", "INTENS", "INTRAMAN", "INTCPHF", "ENDCPHF",
                  "IRSPEC", "RANGE", "0.0 4000.0", "END", "ENDFREQ"]


def _block(text):
    lines = text.split("\n")
    return lines[lines.index("FREQCALC"):lines.index("ENDFREQ") + 1]


# ------------------------------------------------- interactive opt2d12 route

def _interactive_child(tmp_path, freqcalc, answers):
    lines = agbr.AGBR.with_suffix(".d12").read_text().split("\n")
    start, end = lines.index("OPTGEOM"), lines.index("ENDOPT")
    (tmp_path / "agbr_freq.d12").write_text(
        "\n".join(lines[:start] + freqcalc.split("\n") + lines[end + 1:]))
    shutil.copy(agbr.AGBR.with_suffix(".out"), tmp_path / "agbr_freq.out")
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "mace_cli"), "opt2d12", "--out-file", "agbr_freq.out",
         "--d12-file", "agbr_freq.d12"],
        cwd=tmp_path, input=answers + "\n" * 300, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, (result.stdout + result.stderr)[-1500:]
    kids = [p for p in tmp_path.glob("*.d12") if p.name != "agbr_freq.d12"]
    assert len(kids) == 1, sorted(p.name for p in tmp_path.iterdir())
    return _block(kids[0].read_text()), result.stdout


@pytest.mark.parametrize("parent,child", [
    (IR_PARENT, ["FREQCALC", "INTENS", "INTCPHF", "ENDCPHF", "ENDFREQ"]),
    (IR_RAMAN_PARENT, IR_RAMAN_CHILD),
])
def test_interactive_freq_child_keeps_the_parents_intensities(tmp_path, parent, child):
    """Modify settings (n), FREQ (3), then Enter-through. Before: NOINTENS."""
    got, out = _interactive_child(tmp_path, parent, "n\n3\n")
    assert "0: Keep the parent's FREQCALC settings" in out
    assert got == child


def test_interactive_basic_template_still_drops_them(tmp_path):
    """Choosing template 1 is the user changing the parent's settings."""
    got, _ = _interactive_child(tmp_path, IR_PARENT, "n\n3\n1\n1\n")
    assert got == ["FREQCALC", "NOINTENS", "ENDFREQ"]


def test_interactive_parent_without_freq_settings_offers_no_keep(tmp_path):
    got, out = _interactive_child(tmp_path, "FREQCALC\nNOINTENS\nENDFREQ", "n\n3\n")
    assert "Keep the parent's FREQCALC" not in out
    assert got == ["FREQCALC", "NOINTENS", "ENDFREQ"]


# ------------------------------------------------------------ workflow route

def _planner_step(route, tmp_path, monkeypatch):
    if route == "quick_start":
        from mace.run_mace import create_quick_workflow_plan

        class _Args:
            max_jobs = 10
        plan = create_quick_workflow_plan(tmp_path, [], "d12", ["OPT", "FREQ"], _Args())
        return plan["step_configurations"]["FREQ_2"]
    planner = WorkflowPlanner(work_dir=tmp_path / "plan")
    replies = iter([{"basic": "1", "advanced": "2"}[route]])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(replies, ""))
    return planner.configure_frequency_step("FREQ", 3)


def _workflow_child(tmp_path, parent_block, step):
    d12, out = __import__("test_workflow_freq_keeps_parent_numderiv")._agbr_freq_parent(parent_block)
    eng = base._engine(tmp_path, d12, out, {"FREQ_3": step})
    seen = base._numbered(eng, "FREQ", run_script=True)
    assert seen["ok"], seen["log"][-1500:]
    (deck,) = seen["decks"].values()
    return _block(deck)


ROUTES = ["basic", "advanced", "quick_start"]


@pytest.mark.parametrize("route", ROUTES)
def test_default_plan_step_names_no_intensity_setting(tmp_path, monkeypatch, route):
    fs = _planner_step(route, tmp_path, monkeypatch)["frequency_settings"]
    assert not {"intensities", "raman", "ir_method"} & set(fs), fs


@pytest.mark.parametrize("route", ROUTES)
def test_workflow_freq_child_keeps_the_parents_intensities(tmp_path, monkeypatch, route):
    pytest.importorskip("numpy")
    step = _planner_step(route, tmp_path, monkeypatch)
    assert _workflow_child(tmp_path, IR_RAMAN_PARENT, step) == IR_RAMAN_CHILD


@pytest.mark.parametrize("route", ROUTES)
def test_workflow_opt_parent_still_gets_no_intensities(tmp_path, monkeypatch, route):
    pytest.importorskip("numpy")
    step = _planner_step(route, tmp_path, monkeypatch)
    assert _workflow_child(tmp_path, "FREQCALC\nNOINTENS\nENDFREQ", step) == [
        "FREQCALC", "NOINTENS", "ENDFREQ"]


def test_advanced_pure_frequencies_is_an_explicit_no(tmp_path, monkeypatch):
    pytest.importorskip("numpy")
    planner = WorkflowPlanner(work_dir=tmp_path / "plan")

    def reply(prompt=""):
        if "customization level" in prompt:
            return "2"
        if "Select calculation type" in prompt:
            return "1"
        return ""

    monkeypatch.setattr("builtins.input", reply)
    step = planner.configure_frequency_step("FREQ", 3)
    assert step["frequency_settings"]["intensities"] is False
    assert _workflow_child(tmp_path, IR_PARENT, step) == ["FREQCALC", "NOINTENS", "ENDFREQ"]
