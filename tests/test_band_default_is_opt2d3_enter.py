"""A BAND step nobody configured writes opt2d3's Enter-through deck.

Pressing Enter at every opt2d3 BAND question gives the automatic path in
high-symmetry LABEL format (shrink 0), NSUB 1000, all bands, no eigenvalue
print. The routes that choose a BAND configuration for the user -- the
engine default, the planner's Basic answer and the quick-start / --progress
plan -- wrote something else: k-point vectors (format 2) or the SeeK-path
vectors (format 4), so the same parent got a different path, title and
NLINE record depending on whether it went through a workflow or opt2d3.

Each test runs both real routes as subprocesses: opt2d3 interactively with
only Enter on stdin, and CRYSTALOptToD3.py --config-file with the route's
configuration wrapped the way the engine wraps a plan d3_config.
"""
import builtins
import shutil

import pytest

from conftest import find_data
from test_workflow_band_matches_opt2d3 import _manual, _workflow, DATA

pytest.importorskip("numpy")

PARENTS = {
    "tqb_pto": DATA / "opt_restart" / "tqb_pto.out",              # 3D, Pm-3m
    "tqc_agbr": DATA / "opt_restart" / "tqc_agbr.out",            # 3D
    "graphene_lg80": DATA / "low_dim_groups" / "graphene_lg80.out",  # slab
    "graphene_lg47": DATA / "low_dim_groups" / "graphene_lg47.out",  # slab
}

# Corpus SP parents: several lattices and a slab (skipped without test/)
CORPUS = [
    "SP/1_dia_opt_rev1_sp_*.out",                 # F d -3 m
    "SP/2_dia2_opt_rev1_sp_*.out",                # R -3 m
    "SP/3.4^2T137_rev1_sp_*.out",                 # F m -3
    "SP/3,4^2T1-CA_rev1_sp_*.out",                # P -4 m 2
    "SP/3,4^2T13-CA_rev1_sp_*.out",               # P n n m
    "SP/3.4^9T2_BULK_*_sp_*.out",                 # P -1
    "SP/Ag1Cl1_sym_*_sp_*.out",                   # P m -3 m
    "SP/Ag1Cl2_sym_*_sp_*.out",                   # I 4/m m m
    "SP/Ag1Cl3_sym_*_sp_*.out",                   # F m m m
    "SP/Ag1Cl5_sym_*_sp_*.out",                   # C m
    "SP/Ag2Br2_sym_*_sp_*.out",                   # P 63 m c
    "SP/Ag2Br3_mp-862982_*_sp_*.out",             # R -3 c
    "SP/Ag2S_mp-2025704_*_sp_*.out",              # P 21/m
    "SP/4LG_2x2_AA_opt_HSESOL3C_optimized.out",   # slab
]


def _stage(tmp_path, src):
    d = tmp_path / "parent"
    d.mkdir()
    shutil.copy(src, d / src.name)
    if src.with_suffix(".d12").exists():
        shutil.copy(src.with_suffix(".d12"), d / src.with_suffix(".d12").name)
    (d / src.with_suffix(".f9").name).write_text("wavefunction placeholder\n")
    return d / src.name


def _route_config(route, tmp_path, monkeypatch):
    if route == "engine":
        from mace.workflow.engine import WorkflowEngine
        return WorkflowEngine.__new__(WorkflowEngine)._get_default_d3_config("BAND")
    if route == "planner_basic":
        from mace.workflow.planner import WorkflowPlanner
        monkeypatch.setattr(builtins, "input", lambda prompt="": "")
        return WorkflowPlanner(work_dir=tmp_path / "plan")._get_basic_d3_config("BAND")
    from mace.run_mace import _get_basic_d3_config_for_quick_start
    return _get_basic_d3_config_for_quick_start("BAND")


ROUTES = ["engine", "planner_basic", "quick_start"]


@pytest.mark.parametrize("parent", sorted(PARENTS))
@pytest.mark.parametrize("route", ROUTES)
def test_default_band_deck_is_opt2d3_enter_deck(tmp_path, monkeypatch, route, parent):
    out = _stage(tmp_path, PARENTS[parent])
    manual = _manual(out, tmp_path / "manual", [])
    config = _route_config(route, tmp_path, monkeypatch)
    assert _workflow(out, tmp_path / "workflow", config) == manual


@pytest.mark.parametrize("pattern", CORPUS)
def test_default_band_deck_is_opt2d3_enter_deck_corpus(tmp_path, monkeypatch, pattern):
    out = _stage(tmp_path, find_data(pattern))
    manual = _manual(out, tmp_path / "manual", [])
    for route in ROUTES:
        config = _route_config(route, tmp_path, monkeypatch)
        assert _workflow(out, tmp_path / f"wf_{route}", config) == manual, route
