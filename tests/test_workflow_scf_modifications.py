"""A later workflow step's SCF block and DFT grid over its parent's (1.1.6).

The spec's and plan's ``scf_modifications`` (MAXCYCLE, FMIXING, the mixing METHOD, LEVSHIFT, SMEAR,
SPINLOCK, TOLDEE) and ``grid_modifications`` ({"new_grid": ...}) are checked with the deck writer's own
check (d12_constants.check_scf_modifications), handed to CRYSTALOptToD12 by the engine, and written by
``mace opt2d12 --config-file`` as its interactive "modify settings" answers would be. The CLI planner wrote
both keys before, and the engine never read them. A key left out keeps the parent's record. An explicit
MAXCYCLE on a FREQ deck is written as given (with a warning below 200); an inherited one keeps the floor.
"""
import json
import os
import re
import shutil
import subprocess
import sys

import pytest

from conftest import REPO_ROOT
from mace.workflow.spec import SpecError, build_plan, load_spec, resolve_defaults

MACE_CLI = REPO_ROOT / "mace_cli"
PARENT = REPO_ROOT / "tests" / "data" / "opt_restart" / "tqc_agbr"
sys.path.insert(0, str(REPO_ROOT / "Crystal_d12"))
import d12_constants as C  # noqa: E402


@pytest.fixture(autouse=True)
def _plain_site(monkeypatch, tmp_path):
    monkeypatch.setenv("MACE_SITE_CONFIG", "none")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


def _env():
    return {k: v for k, v in os.environ.items()
            if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH")}


def _opt2d12(tmp_path, config, *extra):
    for ext in (".d12", ".out"):
        shutil.copy(str(PARENT) + ext, tmp_path / f"p{ext}")
    (tmp_path / "c.json").write_text(json.dumps(config))
    r = subprocess.run([sys.executable, str(MACE_CLI), "opt2d12", "--out-file", "p.out", "--d12-file", "p.d12",
                        "--config-file", "c.json", "--non-interactive", "--output-dir", "o", *extra],
                       cwd=tmp_path, capture_output=True, text=True, timeout=300, env=_env(),
                       stdin=subprocess.DEVNULL)
    decks = list((tmp_path / "o").glob("*.d12")) if (tmp_path / "o").is_dir() else []
    return r.returncode, (decks[0].read_text() if decks else None), r.stdout + r.stderr


def _record(deck, word):
    m = re.search(rf"^{word}\n(.+)$", deck, re.M)
    return m.group(1).strip() if m else None


def test_the_check_takes_the_interactive_answers_values():
    assert C.check_scf_modifications({"MAXCYCLE": "300", "METHOD": "broyden", "LEVSHIFT": [5, 1], "SMEAR": None,
                                      "SPINLOCK": 0, "FMIXING": 40, "TOLDEE": 9}) == \
        {"MAXCYCLE": 300, "METHOD": "BROYDEN", "LEVSHIFT": [5, 1], "SMEAR": None, "SPINLOCK": 0, "FMIXING": 40,
         "TOLDEE": 9}
    for bad, msg in (({"MAXCYCLE": 0}, "MAXCYCLE"), ({"FMIXING": 120}, "FMIXING"), ({"METHOD": "PULAY"}, "METHOD"),
                     ({"LEVSHIFT": [5, 3]}, "ILOCK"), ({"SMEAR": -1}, "SMEAR"), ({"SMEAR": True}, "SMEAR"),
                     ({"GRID": "XLGRID"}, "not known")):
        with pytest.raises(ValueError, match=msg):
            C.check_scf_modifications(bad)


def test_opt2d12_writes_a_configs_scf_modifications(tmp_path):
    code, deck, log = _opt2d12(tmp_path, {"calculation_type": "SP", "dft_grid": "XXLGRID", "scf_modifications": {
        "MAXCYCLE": 333, "FMIXING": 45, "METHOD": "BROYDEN", "LEVSHIFT": [3, 1], "SMEAR": 0.005}})
    assert code == 0 and deck, log[-3000:]
    assert _record(deck, "MAXCYCLE") == "333" and _record(deck, "FMIXING") == "45"
    assert re.search(r"^BROYDEN\n", deck, re.M) and _record(deck, "LEVSHIFT").split() == ["3", "1"]
    assert float(_record(deck, "SMEAR")) == pytest.approx(0.005) and "XXLGRID" in deck
    assert "SCF settings changed: MAXCYCLE 333, FMIXING 45, BROYDEN, LEVSHIFT 3 1, SMEAR 0.005 Ha" in log


def test_a_key_left_out_keeps_the_parents_record(tmp_path):
    parent = (str(PARENT) + ".d12")
    code, deck, log = _opt2d12(tmp_path, {"calculation_type": "SP", "scf_modifications": {"FMIXING": 45}})
    assert code == 0 and deck, log[-3000:]
    text = open(parent).read()
    assert _record(deck, "MAXCYCLE") == _record(text, "MAXCYCLE") and _record(deck, "TOLDEE") == _record(text, "TOLDEE")


@pytest.mark.parametrize("mods, want, warned", [({"MAXCYCLE": 150}, 150, True), ({}, None, False)])
def test_an_explicit_freq_maxcycle_wins_over_the_floor(tmp_path, mods, want, warned):
    cfg = {"calculation_type": "FREQ", **({"scf_modifications": mods} if mods else {})}
    code, deck, log = _opt2d12(tmp_path, cfg)
    assert code == 0 and deck, log[-3000:]
    n = int(_record(deck, "MAXCYCLE"))
    assert n == want if want else n >= C.TIGHT_SCF_MIN_MAXCYCLE
    assert ("below the 200" in log) is warned


def test_a_bad_value_writes_nothing_and_says_why(tmp_path):
    code, deck, log = _opt2d12(tmp_path, {"calculation_type": "SP", "scf_modifications": {"FMIXING": 120}})
    assert code != 0 and deck is None
    assert "FMIXING is a percentage" in log


def _spec(tmp_path, steps, seq=("OPT", "SP", "FREQ")):
    deck = tmp_path / "mat_opt.d12"
    shutil.copy(str(PARENT) + ".d12", deck)
    p = tmp_path / "spec.json"
    p.write_text(json.dumps({"version": 1, "inputs": {"type": "d12", "files": [deck.name]},
                             "sequence": list(seq), "steps": steps}))
    return p


def test_spec_overrides_reach_the_plan_and_the_engine_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = load_spec(_spec(tmp_path, {"SP_2": {"overrides": {
        "scf_modifications": {"MAXCYCLE": "400", "METHOD": "anderson"},
        "grid_modifications": {"new_grid": "XXLGRID"}}}}))
    assert spec["steps"]["SP_2"]["overrides"]["scf_modifications"] == {"MAXCYCLE": 400, "METHOD": "ANDERSON"}
    _, wid, plan = build_plan(spec, tmp_path)
    sp = plan["step_configurations"]["SP_2"]
    assert sp["scf_modifications"] == {"MAXCYCLE": 400, "METHOD": "ANDERSON"}
    assert sp["grid_modifications"] == {"new_grid": "XXLGRID"}


@pytest.mark.parametrize("ov, msg", [
    ({"scf_modifications": {"FMIXING": "lots"}}, "FMIXING is a percentage"),
    ({"grid_modifications": {"new_grid": "BIGGRID"}}, "grid_modifications"),
])
def test_spec_checks_say_what_is_wrong(tmp_path, ov, msg):
    with pytest.raises(SpecError, match=msg):
        load_spec(_spec(tmp_path, {"SP_2": {"overrides": ov}}))


def test_a_freq_maxcycle_below_the_floor_is_a_warning(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = load_spec(_spec(tmp_path, {"FREQ_3": {"overrides": {"scf_modifications": {"MAXCYCLE": 120}}}}))
    w = resolve_defaults(spec, tmp_path)["FREQ_3"]["warnings"]
    assert any("MAXCYCLE 120 is below the 200" in x for x in w)


def test_mace_workflow_spec_json_checks_them(tmp_path):
    """The CLI path the GUI runs: mace workflow --spec SPEC --json."""
    bad = _spec(tmp_path, {"SP_2": {"overrides": {"scf_modifications": {"LEVSHIFT": [5]}}}})
    env = {**_env(), "MACE_SITE_CONFIG": "none", "HOME": str(tmp_path / "home")}
    r = subprocess.run([sys.executable, str(MACE_CLI), "--no-banner", "workflow", "--spec", str(bad), "--work-dir",
                        str(tmp_path / "wd"), "--json"], cwd=tmp_path, capture_output=True, text=True, timeout=300,
                       env=env, stdin=subprocess.DEVNULL)
    out = json.loads(r.stdout[r.stdout.index("{"):])
    assert out["ok"] is False and "LEVSHIFT is [ISHIFT, ILOCK]" in out["error"]
    good = _spec(tmp_path, {"FREQ_3": {"overrides": {"scf_modifications": {"MAXCYCLE": 150, "FMIXING": 40}}}})
    r = subprocess.run([sys.executable, str(MACE_CLI), "--no-banner", "workflow", "--spec", str(good), "--work-dir",
                        str(tmp_path / "wd"), "--json"], cwd=tmp_path, capture_output=True, text=True, timeout=300,
                       env=env, stdin=subprocess.DEVNULL)
    out = json.loads(r.stdout[r.stdout.index("{"):])
    assert out["ok"] is True, out
    assert out["steps"]["FREQ_3"]["settings"]["scf_modifications"] == {"MAXCYCLE": 150, "FMIXING": 40}
    assert any("below the 200" in w for w in out["steps"]["FREQ_3"]["warnings"])


def test_the_planners_scf_answers_are_checked_when_asked(tmp_path, monkeypatch):
    """Review 1.1.6 E5: the CLI planner took FMIXING 150 or TOLDEE 0 into the plan, and the step's deck then
    failed when the workflow reached it. Each answer is checked as it is given and asked again."""
    from mace.workflow.planner import WorkflowPlanner
    planner = WorkflowPlanner(work_dir=tmp_path)
    answers = iter(["0", "8", "", "150", "abc", "40", "broyden"])
    prompts = []

    def fake_input(prompt=""):
        prompts.append(prompt)
        return next(answers)

    monkeypatch.setattr("builtins.input", fake_input)
    mods = planner._get_scf_modifications()
    assert mods == {"TOLDEE": 8, "FMIXING": 40, "METHOD": "BROYDEN"}
    assert C.check_scf_modifications(mods) == mods
    assert sum("TOLDEE" in p for p in prompts) == 2 and sum("FMIXING" in p for p in prompts) == 3
