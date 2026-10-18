"""A workflow TRANSPORT deck is byte for byte the deck opt2d3 writes when
every question is answered with Enter (TRANGE "100.0 800.0 50.0" on both
routes): the engine default, run through CRYSTALOptToD3.py --config-file as
the engine wraps it, against opt2d3 run interactively on the same parent."""
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
PARENT = Path(__file__).parent / "data" / "opt_restart" / "tqb_pto.out"


def _env():
    env = dict(os.environ)
    for k in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH"):
        env.pop(k, None)
    return env


def _stage(d):
    d.mkdir(parents=True)
    shutil.copy(PARENT, d / PARENT.name)
    if PARENT.with_suffix(".d12").exists():
        shutil.copy(PARENT.with_suffix(".d12"), d / PARENT.with_suffix(".d12").name)
    (d / PARENT.with_suffix(".f9").name).write_text("wavefunction placeholder\n")
    return d / PARENT.name


def _deck(out_dir):
    decks = list(Path(out_dir).glob("*_transport.d3"))
    assert len(decks) == 1, list(Path(out_dir).iterdir())
    return decks[0].read_text()


def _run(argv):
    r = subprocess.run([sys.executable, str(SCRIPT), *argv], input="\n" * 40, capture_output=True,
                       text=True, env=_env(), timeout=120)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]


@pytest.mark.parametrize("route", ["engine", "planner", "quick_start"])
def test_default_transport_deck_is_opt2d3s(tmp_path, route):
    from mace.workflow.engine import WorkflowEngine
    from mace.workflow.planner import WorkflowPlanner
    from mace.run_mace import _get_basic_d3_config_for_quick_start
    if route == "engine":
        config = WorkflowEngine.__new__(WorkflowEngine)._get_default_d3_config("TRANSPORT")
    elif route == "planner":
        config = WorkflowPlanner.__new__(WorkflowPlanner)._get_basic_d3_config("TRANSPORT")
    else:
        config = _get_basic_d3_config_for_quick_start("TRANSPORT")
    parent = _stage(tmp_path / "manual")
    _run(["--input", str(parent), "--calc-type", "TRANSPORT", "--output-dir", str(parent.parent)])
    manual = _deck(parent.parent)
    parent = _stage(tmp_path / "workflow")
    cfg = tmp_path / "transport_config.json"
    cfg.write_text(json.dumps({"version": "1.0", "type": "d3_configuration",
                               "calculation_type": "TRANSPORT", "configuration": config}))
    _run(["--input", str(parent), "--calc-type", "TRANSPORT", "--output-dir", str(parent.parent),
          "--config-file", str(cfg)])
    workflow = _deck(parent.parent)
    assert "100.0 800.0 50.0" in manual
    assert workflow == manual
