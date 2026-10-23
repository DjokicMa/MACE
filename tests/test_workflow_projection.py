"""Projected later steps of a workflow preview (mace/workflow/projection.py):
in a preview copy of a folder, after the first step was "submitted" there,
the real engine writes each later step's deck and job script for one
material from stand-in parent outputs, so the GUI's builder can show them.
It refuses to run anywhere but in a preview copy (it marks steps completed).

Real CLI + fake SLURM, a corpus CIF."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from array_helpers import rows
from conftest import find_data
from fake_slurm import FakeSlurm
from test_array_default_path_golden import _mace

REPO = Path(__file__).resolve().parents[1]


def _ok(p):
    assert p.returncode == 0, p.stdout[-4000:] + p.stderr[-4000:]
    return p


def _env(tmp_path, fake):
    env = dict(fake.env)
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env.update(HOME=str(home), USER="tester", XDG_CACHE_HOME=str(tmp_path / "cache"), MACE_SITE_CONFIG="none")
    env.pop("MACE_HOME", None)
    env.pop("PYTHONPATH", None)
    return env


def _project(folder, env, *args):
    return subprocess.run([sys.executable, "-m", "mace.workflow.projection", "--folder", str(folder), *args],
                          cwd=str(folder), env=dict(env, PYTHONPATH=str(REPO)), capture_output=True, text=True,
                          timeout=600)


def _launched(tmp_path, marker=True):
    cif = find_data("CIFs/1_dia_opt_rev1.cif")
    fake = FakeSlurm(tmp_path / "fs")
    env = _env(tmp_path, fake)
    root = tmp_path / "copy"
    folder = root / "cifs"
    folder.mkdir(parents=True)
    if marker:
        (root / ".mace_preview_mirror").write_text("preview\n")
    for name in ("A", "B"):
        shutil.copy(cif, folder / f"{name}.cif")
    (folder / "spec.json").write_text(json.dumps({
        "version": 1, "inputs": {"type": "cif", "files": ["A.cif", "B.cif"]}, "template": "full_electronic",
        "steps": {"SP_2": {"overrides": {"method_modifications": {"new_functional": "PBE0"}}},
                  "BAND_3": {"resources": {"memory": "48G"}}}}))
    w = json.loads(_ok(_mace(folder, env, "workflow", "--spec", "spec.json", "--write-plan", "--json")).stdout)
    _ok(_mace(folder, env, *w["launch"][1:]))
    return folder, env, fake, w["workflow_id"]


def test_projection_writes_later_steps_for_one_material(tmp_path):
    pytest.importorskip("ase", reason="the workflow starts from a CIF, which mace convert reads with ase")
    folder, env, fake, wid = _launched(tmp_path)
    fake.update(next_id=7000)
    p = _ok(_project(folder, env, "--workflow-id", wid, "--json"))
    out = json.loads(p.stdout)
    assert out["material_id"] == "A"
    steps = {s["step"]: s for s in out["steps"]}
    assert sorted(steps) == ["BAND_3", "DOSS_4", "SP_2"]
    sp = Path(steps["SP_2"]["input_file"])
    assert sp.is_file() and "PBE0" in sp.read_text()
    # the stand-in parent's atoms reach the projected deck (a stand-in .out whose atom tables the
    # parser could not read gave an SP deck with "0" atoms in the 1.1.6 preview's workflow review)
    lines = sp.read_text().splitlines()
    i = lines.index("227")
    assert lines[i + 2] == "1" and lines[i + 3].split()[0] == "6", lines[:i + 4]
    assert steps["SP_2"]["parent"] == "OPT_1"
    # a stand-in holds the deck's unique atom in the cubic cell, not the structure: its BAND k-path is the
    # static table path of Fd-3m (cF2), not SeeK-path's for the stand-in read as simple cubic (GAMMA-X-M-...)
    title = Path(steps["BAND_3"]["input_file"]).read_text().splitlines()[1]
    assert "GAMMA-X-U" in title and "GAMMA-X-M" not in title, title
    band_sh = Path(steps["BAND_3"]["script"]).read_text()
    assert "#SBATCH --mem=48G" in band_sh
    assert Path(steps["DOSS_4"]["input_file"]).suffix == ".d3"
    # B was left as the launch left it: only A's chain was projected
    db = folder / f".mace_context_{wid}" / "materials.db"
    b = [r for r in rows(db).values() if r["material_id"] == "B"]
    assert [(r["calc_type"], r["status"]) for r in b] == [("OPT", "submitted")]


def test_projection_refuses_outside_a_preview_copy(tmp_path):
    pytest.importorskip("ase", reason="the workflow starts from a CIF, which mace convert reads with ase")
    folder, env, fake, wid = _launched(tmp_path, marker=False)
    db = folder / f".mace_context_{wid}" / "materials.db"
    before = {k: dict(v) for k, v in rows(db).items()}
    p = _project(folder, env, "--workflow-id", wid, "--json")
    assert p.returncode == 2
    assert "preview" in json.loads(p.stdout)["error"]
    assert {k: dict(v) for k, v in rows(db).items()} == before
    assert not list(folder.rglob("*.out"))
