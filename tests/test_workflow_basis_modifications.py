"""A workflow step's basis_modifications take an external library too (1.1.6).

{"new_basis": <internal set>} writes BASISSET; {"new_basis": <library>, "basis_set_type": "EXTERNAL"} the
library's records, whatever the parent had, so a chain can go external -> internal -> external. The CLI
planner's basis menu offers MACE's bundled external libraries (8, 9) next to the internal sets.
"""
import json
import os
import shutil
import subprocess
import sys

import pytest

from conftest import REPO_ROOT
from mace.workflow.spec import SpecError, build_plan, load_spec

MACE_CLI = REPO_ROOT / "mace_cli"
PARENT = REPO_ROOT / "tests" / "data" / "opt_restart" / "tqc_agbr"


@pytest.fixture(autouse=True)
def _plain_site(monkeypatch, tmp_path):
    monkeypatch.setenv("MACE_SITE_CONFIG", "none")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


def _spec(tmp_path, steps, seq=("OPT", "SP", "SP2")):
    deck = tmp_path / "mat_opt.d12"
    shutil.copy(str(PARENT) + ".d12", deck)
    p = tmp_path / "spec.json"
    p.write_text(json.dumps({"version": 1, "inputs": {"type": "d12", "files": [deck.name]},
                             "sequence": list(seq), "steps": steps}))
    return p


def test_external_and_internal_switches_reach_the_plan(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ext = {"new_basis": "full.basis.doublezeta", "basis_set_type": "EXTERNAL"}
    spec = load_spec(_spec(tmp_path, {"SP_2": {"overrides": {"basis_modifications": {"new_basis": "POB-TZVP-REV2"}}},
                                      "SP2_3": {"overrides": {"basis_modifications": ext}}}))
    _, _, plan = build_plan(spec, tmp_path)
    assert plan["step_configurations"]["SP2_3"]["basis_modifications"] == ext
    assert plan["step_configurations"]["SP_2"]["basis_modifications"] == {"new_basis": "POB-TZVP-REV2"}


@pytest.mark.parametrize("bm, msg", [
    ({"new_basis": "full.basis.nowhere", "basis_set_type": "EXTERNAL"}, "is not a basis library"),
    ({"new_basis": "POB-TZVP-REV2", "basis_set_type": "LIBRARY"}, "basis_set_type"),
    ({"new_basis": "full.basis.triplezeta"}, "is not an internal basis set"),
])
def test_bad_basis_modifications_say_why(tmp_path, bm, msg):
    with pytest.raises(SpecError, match=msg):
        load_spec(_spec(tmp_path, {"SP_2": {"overrides": {"basis_modifications": bm}}}))


def test_mace_workflow_spec_json_takes_an_external_library(tmp_path):
    p = _spec(tmp_path, {"SP_2": {"overrides": {"basis_modifications": {
        "new_basis": "full.basis.triplezeta", "basis_set_type": "EXTERNAL"}}}})
    env = {k: v for k, v in os.environ.items() if k not in ("COLORTERM", "TERM", "NO_COLOR", "PYTHONPATH")}
    env.update(MACE_SITE_CONFIG="none", HOME=str(tmp_path / "home"))
    r = subprocess.run([sys.executable, str(MACE_CLI), "--no-banner", "workflow", "--spec", str(p), "--work-dir",
                        str(tmp_path / "wd"), "--json"], cwd=tmp_path, capture_output=True, text=True, timeout=300,
                       env=env, stdin=subprocess.DEVNULL)
    out = json.loads(r.stdout[r.stdout.index("{"):])
    assert out["ok"] is True, out
    assert out["steps"]["SP_2"]["settings"]["basis_modifications"]["basis_set_type"] == "EXTERNAL"


@pytest.mark.parametrize("answer, want", [("8", {"new_basis": "full.basis.triplezeta", "basis_set_type": "EXTERNAL"}),
                                          ("2", {"new_basis": "POB-TZVP-REV2"}),
                                          ("", {"inherit_basis": True})])
def test_the_planners_basis_menu_offers_the_bundled_libraries(tmp_path, monkeypatch, answer, want):
    from mace.workflow.planner import WorkflowPlanner
    p = WorkflowPlanner(work_dir=tmp_path / "plan")
    monkeypatch.setattr("builtins.input", lambda prompt="": answer)
    assert p._get_basis_modifications() == want
