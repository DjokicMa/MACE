"""A DOSS step nobody configured: its plan entry describes the deck it gets.

HPCC QA (2026-10-03): the quick-start plan's DOSS step read
{projection_type: 0 '# Total DOS only', energy_range: [-20, 20]}, but the
deck had 8 orbital projections over all bands. The D3 writer honours
projection_type 1-4 and the key energy_window, so 0 fell through to its
default (by element and orbital type, with totals) and energy_range was
never read. The deck was opt2d3's Enter-through deck, which is right; the
configuration claimed something else. The engine default and the planner's
Basic answer carried the same configuration.

The routes now carry opt2d3's Enter-through answers. Each test checks the
deck is still opt2d3's (run as subprocesses as in the BAND tests) and that
the configuration says what the writer does with it.
"""
import builtins
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT

pytest.importorskip("numpy")

SCRIPT = REPO_ROOT / "Crystal_d3" / "CRYSTALOptToD3.py"
DATA = Path(__file__).parent / "data"
PARENTS = {
    "tqb_pto": DATA / "opt_restart" / "tqb_pto.out",
    "graphene_lg80": DATA / "low_dim_groups" / "graphene_lg80.out",
}
ROUTES = ["engine", "planner_basic", "quick_start"]


def _env():
    env = dict(os.environ)
    for k in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME"):
        env.pop(k, None)
    return env


def _stage(tmp_path, src):
    d = tmp_path / "parent"
    d.mkdir()
    shutil.copy(src, d / src.name)
    if src.with_suffix(".d12").exists():
        shutil.copy(src.with_suffix(".d12"), d / src.with_suffix(".d12").name)
    (d / src.with_suffix(".f9").name).write_text("wavefunction placeholder\n")
    return d / src.name


def _deck(out_dir):
    (deck,) = Path(out_dir).glob("*_doss.d3")
    return deck.read_text()


def _manual(parent, out_dir):
    r = subprocess.run([sys.executable, str(SCRIPT), "--input", str(parent),
                        "--calc-type", "DOSS", "--output-dir", str(out_dir)],
                       input="\n" * 20, capture_output=True, text=True,
                       env=_env(), timeout=120)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return _deck(out_dir)


def _workflow(parent, out_dir, config):
    Path(out_dir).mkdir(parents=True)
    cfg = Path(out_dir) / "doss_config.json"
    cfg.write_text(json.dumps({"version": "1.0", "type": "d3_configuration",
                               "calculation_type": "DOSS", "configuration": config}))
    r = subprocess.run([sys.executable, str(SCRIPT), "--input", str(parent),
                        "--calc-type", "DOSS", "--output-dir", str(out_dir),
                        "--config-file", str(cfg)],
                       capture_output=True, text=True, env=_env(), timeout=120)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return _deck(out_dir)


def _route_config(route, tmp_path, monkeypatch):
    if route == "engine":
        from mace.workflow.engine import WorkflowEngine
        return WorkflowEngine.__new__(WorkflowEngine)._get_default_d3_config("DOSS")
    if route == "planner_basic":
        from mace.workflow.planner import WorkflowPlanner
        monkeypatch.setattr(builtins, "input", lambda prompt="": "")
        return WorkflowPlanner(work_dir=tmp_path / "plan")._get_basic_d3_config("DOSS")
    from mace.run_mace import _get_basic_d3_config_for_quick_start
    return _get_basic_d3_config_for_quick_start("DOSS")


@pytest.mark.parametrize("route", ROUTES)
def test_config_says_what_the_writer_does(tmp_path, monkeypatch, route):
    config = _route_config(route, tmp_path, monkeypatch)
    # opt2d3's default answer: by element and orbital type, with totals
    assert config["projection_type"] == 3
    assert config["project_orbital_types"] is True
    assert config["element_only"] is False
    assert config["include_element_totals"] is True
    # all bands: no window or band range the writer would not read
    assert "energy_range" not in config and "energy_window" not in config
    assert config["n_points"] == 10000


@pytest.mark.parametrize("parent", sorted(PARENTS))
@pytest.mark.parametrize("route", ROUTES)
def test_default_doss_deck_is_opt2d3_enter_deck(tmp_path, monkeypatch, route, parent):
    out = _stage(tmp_path, PARENTS[parent])
    manual = _manual(out, tmp_path / "manual")
    config = _route_config(route, tmp_path, monkeypatch)
    assert _workflow(out, tmp_path / "workflow", config) == manual
