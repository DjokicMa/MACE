"""A workflow BAND deck is the deck opt2d3 writes for the same choices.

The workflow engine hands a plan's BAND configuration (the engine default,
the planner's Basic or Advanced answers, or opt2d3's own saved configuration
from an Expert step) to CRYSTALOptToD3.py --config-file. Each test runs both
real routes as subprocesses on the same parent: opt2d3 interactively with
piped answers, and the generator with the configuration wrapped the way the
engine wraps it.
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
    "graphene_lg80": DATA / "low_dim_groups" / "graphene_lg80.out",  # slab, p6/mmm
    "graphene_lg47": DATA / "low_dim_groups" / "graphene_lg47.out",  # slab, cmmm
    "tqb_pto": DATA / "opt_restart" / "tqb_pto.out",  # 3D, Pm-3m
}


# A plan's k-point vector (opt2d3 format 2) BAND configuration, as the
# engine default was before it became opt2d3's Enter-through answers
VECTORS_CONFIG = {"calculation_type": "BAND", "path": "auto", "bands": "auto",
                  "shrink": "auto", "labels": "auto", "auto_path": True,
                  "n_points": 1000, "path_method": "coordinates"}


def _env():
    env = dict(os.environ)
    for k in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME"):
        env.pop(k, None)
    return env


def _stage(tmp_path, name):
    src = PARENTS[name]
    d = tmp_path / "parent"
    d.mkdir(exist_ok=True)
    shutil.copy(src, d / src.name)
    d12 = src.with_suffix(".d12")
    if d12.exists():
        shutil.copy(d12, d / d12.name)
    (d / src.with_suffix(".f9").name).write_text("wavefunction placeholder\n")
    return d / src.name


def _deck(out_dir):
    decks = list(Path(out_dir).glob("*_band.d3"))
    assert len(decks) == 1, list(Path(out_dir).iterdir())
    return decks[0].read_text()


def _manual(parent, out_dir, answers):
    r = subprocess.run([sys.executable, str(SCRIPT), "--input", str(parent),
                        "--calc-type", "BAND", "--output-dir", str(out_dir)],
                       input="\n".join(answers) + "\n" * 10, capture_output=True,
                       text=True, env=_env(), timeout=120)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return _deck(out_dir)


def _workflow(parent, out_dir, config):
    # As WorkflowEngine.generate_d3_calculation_new wraps a plan d3_config
    Path(out_dir).mkdir(parents=True)
    cfg = Path(out_dir) / "band_config.json"
    cfg.write_text(json.dumps({"version": "1.0", "type": "d3_configuration",
                               "calculation_type": "BAND", "configuration": config}))
    r = subprocess.run([sys.executable, str(SCRIPT), "--input", str(parent),
                        "--calc-type", "BAND", "--output-dir", str(out_dir),
                        "--config-file", str(cfg)],
                       capture_output=True, text=True, env=_env(), timeout=120)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return _deck(out_dir)


def _advanced(tmp_path, monkeypatch, answers):
    from mace.workflow.planner import WorkflowPlanner
    it = iter(answers)
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(it, ""))
    return WorkflowPlanner(work_dir=tmp_path / "plan")._get_advanced_d3_config("BAND")


def _records(deck):
    lines = deck.splitlines()
    nline = int(lines[2].split()[0])
    assert nline == len(lines) - 4 and nline > 0, deck
    return lines


@pytest.mark.parametrize("parent", ["graphene_lg80", "graphene_lg47"])
def test_manual_slab_path_uses_corresponding_space_group(tmp_path, parent):
    """opt2d3 read a slab as P1 (its "CORRESPONDING SPACE GROUP N. 191" line
    has no symbol) and wrote the triclinic path; the config route reads the
    space group the way CRYSTALOptToD3's structure parser does."""
    out = _stage(tmp_path, parent)
    manual = _manual(out, tmp_path / "manual", ["1", "2", "", "1", "n"])
    workflow = _workflow(out, tmp_path / "workflow", dict(VECTORS_CONFIG))
    assert _records(manual)[2:] == _records(workflow)[2:]


@pytest.mark.parametrize("parent", sorted(PARENTS))
def test_vectors_deck_matches_opt2d3(tmp_path, parent):
    """A plan's k-point vector BAND configuration is opt2d3's format 2; the
    decks agree to the title, which on a slab said "in-plane path" on the
    manual route only. (The default BAND step is opt2d3's Enter-through
    deck: test_band_default_is_opt2d3_enter.)"""
    out = _stage(tmp_path, parent)
    manual = _manual(out, tmp_path / "manual", ["1", "2", "", "1", "n"])
    assert _workflow(out, tmp_path / "workflow", dict(VECTORS_CONFIG)) == manual


@pytest.mark.parametrize("parent", sorted(PARENTS))
@pytest.mark.parametrize("fmt", ["1", "2", "3", "4"])
def test_expert_saved_config_replays_to_the_same_deck(tmp_path, parent, fmt):
    """A planner Expert step saves opt2d3's own configuration and the engine
    replays it through --config-file: the deck must be the one opt2d3 wrote
    (the literature title lost its "Literature" source on the manual route)."""
    out = _stage(tmp_path, parent)
    saved = tmp_path / "saved.json"
    r = subprocess.run([sys.executable, str(SCRIPT), "--input", str(out),
                        "--calc-type", "BAND", "--output-dir", str(tmp_path / "manual"),
                        "--save-config", "--options-file", str(saved)],
                       input="\n".join(["1", fmt, "", "1", "n"]) + "\n" * 10,
                       capture_output=True, text=True, env=_env(), timeout=120)
    assert r.returncode == 0 and saved.exists(), r.stdout[-2000:] + r.stderr[-2000:]
    manual = _deck(tmp_path / "manual")
    config = json.loads(saved.read_text())["configuration"]
    assert _workflow(out, tmp_path / "workflow", config) == manual


@pytest.mark.parametrize("parent", sorted(PARENTS))
@pytest.mark.parametrize("fmt", ["1", "2", "3", "4"])
def test_advanced_path_format_matches_opt2d3(tmp_path, monkeypatch, parent, fmt):
    out = _stage(tmp_path, parent)
    config = _advanced(tmp_path, monkeypatch, ["1", fmt, "", ""])
    manual = _manual(out, tmp_path / "manual", ["1", fmt, "", "1", "n"])
    workflow = _workflow(out, tmp_path / "workflow", config)
    assert _records(workflow) == _records(manual)


def test_advanced_custom_labels_and_band_range(tmp_path, monkeypatch):
    out = _stage(tmp_path, "tqb_pto")
    config = _advanced(tmp_path, monkeypatch,
                       ["2", "G X M G", "n", "3", "40", ""])
    manual = _manual(out, tmp_path / "manual",
                     ["3", "G X M G", "", "3", "3", "40", "n"])
    workflow = _workflow(out, tmp_path / "workflow", config)
    assert _records(workflow) == _records(manual)
    assert workflow.splitlines()[2].split()[3:5] == ["3", "40"]
