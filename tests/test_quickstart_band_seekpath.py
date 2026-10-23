"""Quick-start / --progress BAND steps write opt2d3's Enter-through deck.

SeeK-path is the default k-path of a BAND step (opt2d3's automatic path,
format 4), by every route. v1.1.1 made the quick-start BAND ask for the
SeeK-path vectors (``seekpath_full``); 1.1.2 moved it to the high-symmetry
labels so it matched opt2d3's Enter-through answer; 1.1.6 made SeeK-path
that answer, so the quick-start configuration is again SeeK-path, and the
engine's and the planner's Basic default are the same dict. The decks
themselves are compared in tests/test_workflow_band_matches_opt2d3.py.
"""
import sys

import pytest

from conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT))

from mace.run_mace import _get_basic_d3_config_for_quick_start  # noqa: E402


def test_band_is_the_opt2d3_default():
    from mace.workflow.common.constants import opt2d3_default_band_config
    from mace.workflow.engine import WorkflowEngine
    cfg = _get_basic_d3_config_for_quick_start("BAND")
    assert cfg == opt2d3_default_band_config()
    assert cfg == WorkflowEngine.__new__(WorkflowEngine)._get_default_d3_config("BAND")
    assert cfg["seekpath_full"] is True and cfg["path_method"] == "coordinates"


def test_band_config_still_valid():
    sys.path.insert(0, str(REPO_ROOT / "Crystal_d3"))
    from d3_config import validate_d3_config

    ok, errors = validate_d3_config(_get_basic_d3_config_for_quick_start("BAND"))
    assert ok, errors


@pytest.mark.parametrize("calc_type", ["DOSS", "TRANSPORT", "CHARGE+POTENTIAL"])
def test_non_band_configs_have_no_kpath(calc_type):
    cfg = _get_basic_d3_config_for_quick_start(calc_type)
    assert "seekpath_full" not in cfg and "path_method" not in cfg, cfg
