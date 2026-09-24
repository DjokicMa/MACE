"""One table of convergence presets, used everywhere.

opt2d12's menus define the levels: OPT convergence Standard / Tight / Very
Tight and SCF tolerances Standard / Tight / Very tight. They live in
d12_constants, and the other tools read them from there instead of keeping
their own copies (which had drifted: the planner's "standard" OPT was the
Very Tight 3e-5/1.2e-4, and the writer's fallback for a missing TOLDEG was
also Very Tight while its default settings were Standard).
"""
import io

import d12_calc_basic
import d12_constants
import d12_interactive
from d12_constants import (OPT_CONVERGENCE_PRESETS, SCF_TOLERANCE_PRESETS,
                           opt_convergence, scf_tolerances)


def test_preset_values():
    assert opt_convergence("1") == {"toldeg": 0.0003, "toldex": 0.0012, "toldee": 7, "maxcycle": 800}
    assert opt_convergence("2") == {"toldeg": 0.0001, "toldex": 0.0004, "toldee": 8, "maxcycle": 800}
    assert opt_convergence("3") == {"toldeg": 0.00003, "toldex": 0.00012, "toldee": 9, "maxcycle": 800}
    assert opt_convergence("1", upper=True) == {"TOLDEG": 0.0003, "TOLDEX": 0.0012,
                                                "TOLDEE": 7, "MAXCYCLE": 800}
    assert scf_tolerances("1") == {"TOLINTEG": "7 7 7 7 14", "TOLDEE": 7}
    assert scf_tolerances("2") == {"TOLINTEG": "8 8 8 9 24", "TOLDEE": 9}
    assert scf_tolerances("3") == {"TOLINTEG": "9 9 9 11 38", "TOLDEE": 11}
    assert [p["name"] for p in OPT_CONVERGENCE_PRESETS.values()] == ["Standard", "Tight", "Very Tight"]
    assert [p["name"] for p in SCF_TOLERANCE_PRESETS.values()] == ["Standard", "Tight", "Very tight"]


def test_defaults_are_the_standard_presets():
    for defaults in (d12_constants.DEFAULT_OPT_SETTINGS, d12_calc_basic.DEFAULT_OPT_SETTINGS):
        assert {k: defaults[k] for k in ("toldeg", "toldex", "toldee", "maxcycle")} == opt_convergence("1")
    assert d12_constants.DEFAULT_TOLERANCES == scf_tolerances("1")
    assert {k: d12_constants.DEFAULT_FREQ_SETTINGS[k] for k in ("TOLINTEG", "TOLDEE")} == scf_tolerances("3")


def test_menu_matchers_use_the_table():
    for level, preset in OPT_CONVERGENCE_PRESETS.items():
        assert d12_calc_basic._opt_preset_for(opt_convergence(level)) == level
    for level in SCF_TOLERANCE_PRESETS:
        assert d12_interactive._tolerance_preset_for(scf_tolerances(level)) == level


def _optgeom(settings):
    f = io.StringIO()
    d12_calc_basic.write_optimization_section(f, "FULLOPTG", settings)
    return f.getvalue().split()


def test_writer_fills_missing_tolerances_with_standard():
    """A settings dict without tolerances used to get 0.00003/0.00012 (Very Tight)."""
    body = _optgeom({"type": "FULLOPTG"})
    assert body == ["OPTGEOM", "FULLOPTG", "MAXCYCLE", "800", "TOLDEG", "0.0003",
                    "TOLDEX", "0.0012", "TOLDEE", "7", "ENDOPT"]


def test_custom_convergence_defaults_are_standard(monkeypatch):
    """Custom level with no parent values: blank answers give the Standard values."""
    answers = iter(["4"])                 # Custom

    def fake_read(prompt="", valid_set=None):
        return next(answers, "")

    monkeypatch.setattr(d12_calc_basic, "_nav_read", fake_read)
    monkeypatch.setattr(d12_calc_basic, "_nav_float", lambda prompt="", default=None: float(default))
    monkeypatch.setattr(d12_calc_basic, "_nav_int", lambda prompt="", default=None, choices=None: int(default))
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    monkeypatch.setattr(d12_constants, "get_user_input", lambda prompt, options, default=None: "1")
    monkeypatch.setattr(d12_constants, "yes_no_prompt", lambda prompt, default="yes": False)
    cfg = d12_calc_basic._configure_optimization_impl({})
    assert {k: cfg[k] for k in ("toldeg", "toldex", "toldee", "maxcycle")} == opt_convergence("1")


def test_choosing_a_level_writes_its_values(monkeypatch):
    for level in ("1", "2", "3"):
        answers = iter([level])
        monkeypatch.setattr(d12_calc_basic, "_nav_read", lambda prompt="", valid_set=None: next(answers, ""))
        monkeypatch.setattr("builtins.input", lambda prompt="": "")
        monkeypatch.setattr(d12_constants, "get_user_input", lambda prompt, options, default=None: "1")
        monkeypatch.setattr(d12_constants, "yes_no_prompt", lambda prompt, default="yes": False)
        cfg = d12_calc_basic._configure_optimization_impl({})
        assert {k: cfg[k] for k in ("toldeg", "toldex", "toldee", "maxcycle")} == opt_convergence(level)
        assert cfg["convergence"] == OPT_CONVERGENCE_PRESETS[level]["name"]


def test_scf_menus_use_the_table(monkeypatch):
    for level in ("1", "2", "3"):
        monkeypatch.setattr("builtins.input", lambda prompt="": level)
        assert d12_constants.configure_tolerances(calculation_type="SP") == scf_tolerances(level)
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    assert d12_constants.configure_tolerances(calculation_type="FREQ") == scf_tolerances("3")
    assert d12_constants.configure_tolerances(calculation_type="SP") == scf_tolerances("1")
