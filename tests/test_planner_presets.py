"""The workflow planner offers opt2d12's convergence levels, and nothing else.

Before, the planner kept its own numbers: its "standard" OPT was
0.00003/0.00012 (opt2d12's Very Tight), its "tighter" OPT was
0.000015/0.00006/TOLDEE 8/MAXCYCLE 1000 (no opt2d12 level), its SP "tight"
was really Very tight, the TOLINTEG hints named values that are no level
(8 8 8 8 16, 9 9 9 9 18), and the basic OPT menu wrote ATOMSONLY, which is
not a CRYSTAL keyword (ATOMONLY is). Level 0 of a derived OPT step now
inherits the previous OPT's OPTGEOM instead of imposing 3e-5.
"""
import argparse
import builtins

import pytest

import test_numbered_calc_keeps_parent_method as base
from d12_constants import opt_convergence, scf_tolerances
from mace.workflow.planner import WorkflowPlanner


@pytest.fixture
def planner(tmp_path):
    return WorkflowPlanner(work_dir=tmp_path)


def _answers(monkeypatch, replies):
    """Reply to input() prompts in order; blank once they run out."""
    replies = iter(replies)
    prompts = []

    def fake_input(prompt=""):
        prompts.append(prompt)
        return next(replies, "")

    monkeypatch.setattr(builtins, "input", fake_input)
    return prompts


def test_default_cif_config_is_standard(planner):
    cfg = planner.get_default_cif_config("OPT")
    assert cfg["optimization_settings"] == opt_convergence("1", upper=True)
    assert cfg["tolerances"] == scf_tolerances("1")


def test_quick_start_cif_config_is_standard(tmp_path):
    from mace import run_mace
    args = argparse.Namespace(max_jobs=1)
    plan = run_mace.create_quick_workflow_plan(tmp_path, [tmp_path / "a.cif"], "cif",
                                               ["OPT", "SP", "FREQ"], args)
    assert plan["cif_conversion_config"]["optimization_settings"] == opt_convergence("1", upper=True)
    freq = plan["step_configurations"]["FREQ_3"]["frequency_settings"]
    assert freq["custom_tolerances"] == scf_tolerances("3")


def test_level0_opt_step_inherits_the_previous_opt(planner, monkeypatch):
    _answers(monkeypatch, ["0"])
    cfg = planner.configure_optimization_step("OPT2", 4)
    assert "optimization_settings" not in cfg and "optimization_type" not in cfg
    assert cfg["inherit_settings"] is True and cfg["calculation_type"] == "OPT"


def test_basic_opt_blank_answers_keep_the_previous_opt(planner, monkeypatch):
    _answers(monkeypatch, [])
    cfg = planner._get_basic_opt_config()
    assert cfg == {"inherit_base_settings": True}


@pytest.mark.parametrize("level", ["1", "2", "3"])
def test_basic_opt_offers_the_three_levels(planner, monkeypatch, level):
    _answers(monkeypatch, ["", level])
    cfg = planner._get_basic_opt_config()
    assert cfg["optimization_settings"] == opt_convergence(level, upper=True)
    assert "optimization_type" not in cfg


def test_basic_opt_atomonly_is_the_crystal_keyword(planner, monkeypatch):
    _answers(monkeypatch, ["2", ""])
    assert planner._get_basic_opt_config()["optimization_type"] == "ATOMONLY"


def _sp_basic(planner, monkeypatch, level):
    monkeypatch.setattr(planner, "_get_method_modifications", lambda: {"inherit_functional": True})
    monkeypatch.setattr(planner, "_get_basis_modifications", lambda: {"inherit_basis": True})
    _answers(monkeypatch, [level])
    return planner._get_basic_sp_config()


def test_basic_sp_blank_keeps_the_parent_tolerances(planner, monkeypatch):
    assert "tolerance_modifications" not in _sp_basic(planner, monkeypatch, "")


@pytest.mark.parametrize("level", ["1", "2", "3"])
def test_basic_sp_offers_the_three_scf_levels(planner, monkeypatch, level):
    cfg = _sp_basic(planner, monkeypatch, level)
    assert cfg["tolerance_modifications"] == {"custom_tolerances": scf_tolerances(level)}


def test_custom_tolerance_hints_name_the_presets(planner, monkeypatch, capsys):
    _answers(monkeypatch, [])
    assert planner._get_custom_tolerances() == {}
    out = capsys.readouterr().out
    assert "8 8 8 9 24" in out and "9 9 9 11 38" in out
    assert "8 8 8 8 16" not in out and "9 9 9 9 18" not in out


# ---------------------------------------------------------------------------
# Engine: the plan step reaches the deck
# ---------------------------------------------------------------------------
def test_plan_opt_settings_without_a_type_keep_the_parent_type(tmp_path):
    step = {"calculation_type": "OPT", "inherit_base_settings": True,
            "optimization_settings": opt_convergence("2", upper=True)}
    eng = base._engine(tmp_path, base.DECK.format(ham="UHF"), "dummy\n", {"OPT2_4": step})
    cfg = base._numbered(eng, "OPT2")["config"]
    assert cfg["optimization_settings"] == opt_convergence("2", upper=True)
    assert "optimization_type" not in cfg


def _atomonly_tight_parent():
    base._need(base.DIA)
    d12 = base.DIA.with_suffix(".d12").read_text()
    # (the title names OPTGEOM too, so match the whole record)
    optgeom = d12[d12.index("\nOPTGEOM\n"):d12.index("\nENDOPT\n")]
    new = ("\nOPTGEOM\nATOMONLY\nMAXCYCLE\n600\nTOLDEG\n0.0001\nTOLDEX\n0.0004\n"
           "TOLDEE\n8\nMAXTRADIUS\n0.25")
    return d12.replace(optgeom, new), base.DIA.with_suffix(".out").read_text()


def _optgeom(deck):
    lines = deck.splitlines()
    return lines[lines.index("OPTGEOM"):lines.index("ENDOPT")]


def test_real_level0_opt2_reproduces_the_parent_optgeom(tmp_path):
    """Planner level-0 OPT2 step, run through the engine and the real script."""
    d12, out = _atomonly_tight_parent()
    step = {"calculation_type": "OPT", "source": "CRYSTALOptToD12.py",
            "inherit_settings": True, "customization_level": 0}
    eng = base._engine(tmp_path, d12, out, {"OPT2_4": step})
    name, deck = base._only_deck(base._numbered(eng, "OPT2", True))
    assert _optgeom(deck) == _optgeom(d12)


def test_real_basic_level_keeps_the_parent_type(tmp_path):
    d12, out = _atomonly_tight_parent()
    step = {"calculation_type": "OPT", "inherit_base_settings": True,
            "optimization_settings": opt_convergence("3", upper=True)}
    eng = base._engine(tmp_path, d12, out, {"OPT2_4": step})
    name, deck = base._only_deck(base._numbered(eng, "OPT2", True))
    body = _optgeom(deck)
    assert body[1] == "ATOMONLY"
    assert body[body.index("TOLDEG") + 1] == "0.00003"
    assert body[body.index("TOLDEE") + 1] == "9"
    assert body[body.index("MAXTRADIUS") + 1] == "0.25"
