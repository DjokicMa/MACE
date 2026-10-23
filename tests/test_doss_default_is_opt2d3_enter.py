"""A DOSS step nobody configured: its plan entry describes the deck it gets.

HPCC QA (2026-10-03): the quick-start plan's DOSS step read
{projection_type: 0 '# Total DOS only', energy_range: [-20, 20]}, but the
deck had 8 orbital projections over all bands. The D3 writer honors
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
    # the Standard energy resolution: the points follow from the step and the span
    assert config["energy_resolution"] == "standard" and config["n_points"] == "auto"


@pytest.mark.parametrize("parent", sorted(PARENTS))
@pytest.mark.parametrize("route", ROUTES)
def test_default_doss_deck_is_opt2d3_enter_deck(tmp_path, monkeypatch, route, parent):
    out = _stage(tmp_path, PARENTS[parent])
    manual = _manual(out, tmp_path / "manual")
    config = _route_config(route, tmp_path, monkeypatch)
    assert _workflow(out, tmp_path / "workflow", config) == manual


@pytest.mark.parametrize("parent", sorted(PARENTS))
def test_mace_opt2d3_enter_writes_the_standard_steps_points(tmp_path, parent):
    """`mace opt2d3 -t DOSS` with every answer Enter: the DOSS record's NPT
    (its second number) is the Standard step's: over every band (the parent's
    estimated span) when that needs no more than 5000 points or the parent has no
    trustworthy E_F, else over the default window around E_F (1000 in the 1.1.6
    preview, 10000 before)."""
    out = _stage(tmp_path, PARENTS[parent])
    env = _env()
    env.pop("PYTHONPATH", None)
    env["MACE_NO_BANNER"] = "1"
    r = subprocess.run([sys.executable, str(REPO_ROOT / "mace_cli"), "opt2d3",
                        "--input", str(out), "--calc-type", "DOSS"],
                       cwd=str(out.parent), input="\n" * 40, capture_output=True,
                       text=True, env=env, timeout=180)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    lines = _deck(out.parent).splitlines()
    record = lines[lines.index("DOSS") + 1].split()
    sys.path.insert(0, str(REPO_ROOT / "Crystal_d3"))
    import d3_config as C
    import d3_interactive
    span = C.doss_span_estimate_ev(C.doss_atomic_numbers(out.read_text()))
    assert C.doss_npts(span) > C.DOSS_WINDOW_OFFER_NPTS
    if d3_interactive.trusted_fermi_au(str(out))[0] is None:
        assert record[1:4] == [str(C.doss_npts(span)), "1", record[3]] and record[3] != "-1", record
    else:
        lo, hi = C.DOSS_DEFAULT_WINDOW_EV
        assert record[1:4] == [str(C.doss_npts(hi - lo)), "-1", "-1"], record
