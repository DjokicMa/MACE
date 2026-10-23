"""Every default BAND NSUB is opt2d3's 1000; every default DOSS has the Standard energy resolution.

NSUB is the total number of k points along the whole path (CRYSTAL23 manual
p.309; note 4 p.310: CRYSTAL spreads them over the lines). opt2d3 offers
1000 (its prompt and its writer's default), while a workflow BAND step
without a step configuration (engine), the planner's basic and advanced
BAND configurations and the quick-start configuration used 10000 - ten times
the k points and the cost for the same path. DOSS's n_points is the number
of energy points (the second number of the DOSS record); its default was
10000 in opt2d3's prompt and writer and in every workflow route, 1000 in the
1.1.6 preview, and follows from the energy step since 1.1.6 (the Standard
step, Crystal_d3/d3_config.doss_npts).
"""
import builtins
import inspect
import re

import pytest

pytest.importorskip("numpy")

OPT2D3_BAND_NSUB = 1000


def _opt2d3_defaults():
    import CRYSTALOptToD3
    import d3_interactive
    # the computed path's deck (an edited path, band_path, is written from it)
    writer = inspect.getsource(CRYSTALOptToD3.D3Generator._write_computed_band_d3)
    prompt = inspect.getsource(d3_interactive)
    return ({int(v) for v in re.findall(r'config\.get\("n_points", (\d+)\)', writer)},
            re.search(r'Total number of k-points along path \[(\d+)\]: ", default=(\d+)', prompt)
            .groups())


def test_opt2d3_default_is_1000():
    writer, (shown, used) = _opt2d3_defaults()
    assert writer == {OPT2D3_BAND_NSUB}
    assert int(shown) == int(used) == OPT2D3_BAND_NSUB


def test_workflow_band_defaults_match_opt2d3(tmp_path, monkeypatch):
    from mace.workflow.engine import WorkflowEngine
    from mace.workflow.planner import WorkflowPlanner
    from mace.run_mace import _get_basic_d3_config_for_quick_start

    engine = WorkflowEngine.__new__(WorkflowEngine)
    planner = WorkflowPlanner(work_dir=tmp_path)
    monkeypatch.setattr(builtins, "input", lambda prompt="": "")
    found = {
        "engine": engine._get_default_d3_config("BAND")["n_points"],
        "planner basic": planner._get_basic_d3_config("BAND")["n_points"],
        "planner advanced": planner._get_advanced_d3_config("BAND")["n_points"],
        "quick start": _get_basic_d3_config_for_quick_start("BAND")["n_points"],
    }
    assert found == dict.fromkeys(found, OPT2D3_BAND_NSUB)


def test_opt2d3_doss_default_is_the_standard_resolution():
    import CRYSTALOptToD3
    import d3_interactive
    writer = inspect.getsource(CRYSTALOptToD3.D3Generator._write_doss_d3)
    prompt = inspect.getsource(d3_interactive._doss_resolution_menu)
    assert 'config.get("n_points"' not in writer and "doss_npts(" in writer
    assert 'C.DOSS_RESOLUTIONS.index("standard")' in prompt


def test_doss_defaults_to_the_standard_resolution(tmp_path, monkeypatch):
    from mace.workflow.engine import WorkflowEngine
    from mace.workflow.planner import WorkflowPlanner
    from mace.run_mace import _get_basic_d3_config_for_quick_start

    engine = WorkflowEngine.__new__(WorkflowEngine)
    planner = WorkflowPlanner(work_dir=tmp_path)
    monkeypatch.setattr(builtins, "input", lambda prompt="": "")
    found = {
        "engine": engine._get_default_d3_config("DOSS"),
        "planner basic": planner._get_basic_d3_config("DOSS"),
        "planner advanced": planner._get_advanced_d3_config("DOSS"),
        "quick start": _get_basic_d3_config_for_quick_start("DOSS"),
    }
    assert {k: (c["energy_resolution"], c["n_points"]) for k, c in found.items()} == \
        dict.fromkeys(found, ("standard", "auto"))
