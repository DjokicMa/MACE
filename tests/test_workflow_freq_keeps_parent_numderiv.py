"""A workflow FREQ step keeps a FREQ parent's NUMDERIV unless the plan sets one.

opt2d12 copies a FREQ parent's FREQCALC block into a FREQ child, NUMDERIV
included. A workflow FREQ step hands CRYSTALOptToD12 the plan's
frequency_settings, which replace the parent's; the planner leaves NUMDERIV
out unless the user picks 1 or 2, so a parent's NUMDERIV 2 (central
differences, two displacements per atom and direction; CRYSTAL23 manual
p.218-219) became CRYSTAL's default in the workflow child only.

NUMDERIV is now kept from the parent when the plan does not name it. A plan
that names it wins, and the planner records an explicit "0" as "no keyword".
"""
import pytest

import test_freqcalc_parent_settings as agbr
import test_numbered_calc_keeps_parent_method as base
from mace.workflow.planner import WorkflowPlanner

BASIC = {"mode": "GAMMA", "intensities": False, "temperatures": [298.15]}


def _agbr_freq_parent(freqcalc):
    lines = agbr.AGBR.with_suffix(".d12").read_text().split("\n")
    start, end = lines.index("OPTGEOM"), lines.index("ENDOPT")
    d12 = "\n".join(lines[:start] + freqcalc.split("\n") + lines[end + 1:])
    return d12, agbr.AGBR.with_suffix(".out").read_text()


def _child_freqcalc(tmp_path, parent_block, freq_settings):
    d12, out = _agbr_freq_parent(parent_block)
    step = {"calculation_type": "FREQ", "inherit_base_settings": True,
            "frequency_settings": freq_settings}
    eng = base._engine(tmp_path, d12, out, {"FREQ_3": step})
    seen = base._numbered(eng, "FREQ", run_script=True)
    assert seen["ok"], seen["log"][-1500:]
    (deck,) = seen["decks"].values()
    lines = deck.split("\n")
    return lines[lines.index("FREQCALC"):lines.index("ENDFREQ") + 1]


def test_plan_without_numderiv_keeps_the_parents(tmp_path):
    child = _child_freqcalc(tmp_path, "FREQCALC\nNUMDERIV\n2\nNOINTENS\nENDFREQ", dict(BASIC))
    assert child[:3] == ["FREQCALC", "NUMDERIV", "2"]


def test_plan_numderiv_wins_over_the_parents(tmp_path):
    child = _child_freqcalc(tmp_path, "FREQCALC\nNUMDERIV\n2\nNOINTENS\nENDFREQ",
                            dict(BASIC, numderiv=1))
    assert child[:3] == ["FREQCALC", "NUMDERIV", "1"]


def test_plan_explicit_no_numderiv_drops_the_parents(tmp_path):
    child = _child_freqcalc(tmp_path, "FREQCALC\nNUMDERIV\n2\nNOINTENS\nENDFREQ",
                            dict(BASIC, numderiv=None))
    assert "NUMDERIV" not in child


def test_parent_without_numderiv_gets_none(tmp_path):
    child = _child_freqcalc(tmp_path, "FREQCALC\nNOINTENS\nENDFREQ", dict(BASIC))
    assert "NUMDERIV" not in child


def _advanced_numderiv(tmp_path, monkeypatch, answer):
    planner = WorkflowPlanner(work_dir=tmp_path)

    def reply(prompt=""):
        if "customization level" in prompt:
            return "2"
        if "Select method (0-2)" in prompt:
            return answer
        return ""

    monkeypatch.setattr("builtins.input", reply)
    return planner.configure_frequency_step("FREQ", 3)["frequency_settings"]


def test_planner_blank_leaves_numderiv_to_the_parent(tmp_path, monkeypatch):
    assert "numderiv" not in _advanced_numderiv(tmp_path, monkeypatch, "")


def test_planner_zero_is_an_explicit_no_numderiv(tmp_path, monkeypatch):
    settings = _advanced_numderiv(tmp_path, monkeypatch, "0")
    assert "numderiv" in settings and settings["numderiv"] is None
