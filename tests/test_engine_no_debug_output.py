"""The workflow engine prints no raw DEBUG lines while it generates steps.

A FREQ step printed "DEBUG: Found frequency settings: {...}", and plan
lookup, sequence position and progression printed further DEBUG dumps of
dicts into every workflow log.
"""
import re

import test_numbered_calc_keeps_parent_method as base
from conftest import REPO_ROOT

STEP = {"calculation_type": "FREQ", "inherit_base_settings": True,
        "frequency_settings": {"mode": "GAMMA", "intensities": False}}


def test_freq_step_generation_prints_no_debug(tmp_path, capsys):
    eng = base._engine(tmp_path, base.DECK.format(ham="DFT\nPBE0\nENDDFT"), "dummy\n",
                       {"FREQ_3": STEP})
    base._numbered(eng, "FREQ")
    assert "DEBUG" not in capsys.readouterr().out


def test_engine_has_no_debug_prints():
    source = (REPO_ROOT / "mace" / "workflow" / "engine.py").read_text()
    assert not re.findall(r'print\(f?"\s*DEBUG', source)
