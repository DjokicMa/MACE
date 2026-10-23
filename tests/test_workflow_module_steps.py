"""ELASTIC, EOS, QHA and CPHF as workflow steps (opt2d12's modules after OPT/SP).

A spec names them (with their module block as the step's "elastic_settings"),
the planner offers them, and the engine writes their decks with
CRYSTALOptToD12 from the OPT before them, as `mace opt2d12 --calc-type
ELASTIC --elastic-settings ...` does. The end-to-end tests run the real CLI
and job callback with fake SLURM on the diamond OPT of the test/ corpus, and
complete the ELASTIC step with the diamond ELASTCON run of HPCC.
"""
import json
import re
import shutil
from pathlib import Path

import pytest

from array_helpers import rows
from conftest import REPO_ROOT
from test_workflow_spec_cli_e2e import _complete, _json, _ok, _setup
from test_array_default_path_golden import _mace

from mace.workflow import spec as S
from mace.workflow.common.constants import CALC_TYPES, MODULE_STEP_TYPES, SLURM_RESOURCE_DEFAULTS
from mace.workflow.planner import WorkflowPlanner, soc_plan_problems
from mace.workflow.status import parent_index

HPCC = REPO_ROOT / "tests" / "data" / "hpcc_116" / "mechanics"
SEQ = ["OPT", "SP", "ELASTIC", "EOS", "QHA", "CPHF"]


def _block(lines, key):
    i = lines.index(key)
    return lines[i:lines.index("END", i) + 1]


@pytest.fixture
def deck_dir(tmp_path):
    d = tmp_path / "decks"
    d.mkdir()
    (d / "dia.d12").write_text((HPCC / "dia_elastcon.d12").read_text())
    return d


def _spec(d, steps=None, sequence=SEQ):
    return S.validate_spec({"version": 1, "inputs": {"type": "d12", "files": ["dia.d12"]},
                            "sequence": list(sequence), "steps": steps or {}}, d)


# ------------------------------------------------------------------ spec
def test_module_steps_are_step_types_after_opt_or_sp(deck_dir):
    assert set(MODULE_STEP_TYPES) <= set(CALC_TYPES)
    assert _spec(deck_dir)["sequence"] == SEQ
    assert _spec(deck_dir, sequence=["SP", "ELASTIC"])["sequence"] == ["SP", "ELASTIC"]
    for t in MODULE_STEP_TYPES:
        assert CALC_TYPES[t]["depends_on"] == ["OPT", "SP"]
        with pytest.raises(S.SpecError, match=f"{t} needs one of OPT, SP before it"):
            _spec(deck_dir, sequence=["BAND", t])


def test_module_settings_are_the_steps_overrides(deck_dir):
    sp = _spec(deck_dir, {"ELASTIC_3": {"overrides": {"elastic_settings": {"mode": "ELAPIEZO", "numderiv": 5}}},
                          "QHA_5": {"overrides": {"elastic_settings": {"temperat": "60 10 600"}}}})
    plan = S.resolve_defaults(sp, deck_dir)
    assert plan["ELASTIC_3"]["settings"]["elastic_settings"] == {"mode": "ELAPIEZO", "numderiv": 5}
    assert plan["ELASTIC_3"]["settings_origin"] == "spec overrides"
    assert plan["EOS_4"]["settings"]["calculation_type"] == "EOS"
    assert plan["EOS_4"]["settings_origin"] == "parent"


@pytest.mark.parametrize("key, ov, words", [
    ("ELASTIC_3", {"elastic_settings": {"mode": "EOS"}}, "an EOS block is an EOS step's"),
    ("EOS_4", {"elastic_settings": {"mode": "ELASTCON"}}, "an ELASTCON block is not an EOS step's"),
    ("ELASTIC_3", {"elastic_settings": {"numderiv": "x"}}, "NUMDERIV must be an integer"),
    ("ELASTIC_3", {"elastic_settings": {"mode": "BOGUS"}}, "unknown module BOGUS"),
    ("ELASTIC_3", {"method_modifications": {"new_functional": "PBE0"}}, "it reads: elastic_settings"),
    ("SP_2", {"elastic_settings": {"mode": "ELASTCON"}}, "elastic_settings are not read for an SP step"),
    # the article follows the sound: an ELASTIC, an EOS; a QHA, a CPHF
    ("ELASTIC_3", {"d3_config": {}}, "d3_config are not read for an ELASTIC step"),
    ("EOS_4", {"d3_config": {}}, "d3_config are not read for an EOS step"),
    ("QHA_5", {"d3_config": {}}, "d3_config are not read for a QHA step"),
    ("CPHF_6", {"d3_config": {}}, "d3_config are not read for a CPHF step"),
    ("CPHF_6", {"soc": True}, "CPHF/CPKS is not in the keywords a two-component SCF accepts"),
    ("ELASTIC_3", {"soc": True}, "TWO-COMPONENT SCF IS NOT COMPATIBLE WITH OPTGEOM"),
])
def test_refusals(deck_dir, key, ov, words):
    with pytest.raises(S.SpecError) as err:
        _spec(deck_dir, {key: {"overrides": ov}})
    assert words in str(err.value)


def test_calc_type_articles():
    words = {t: S._a(t).split()[0] for t in ("ELASTIC", "EOS", "OPT", "ORBITALS", "SP", "SP2", "ELAPIEZO",
                                              "QHA", "CPHF", "FREQ", "BAND", "DOSS", "TRANSPORT", "SOC", "PIEZOCON")}
    assert {t for t, a in words.items() if a == "an"} == {"ELASTIC", "EOS", "OPT", "ORBITALS", "SP", "SP2",
                                                         "ELAPIEZO"}
    assert S._a("EOS") == "an EOS" and S._a("QHA") == "a QHA"


def test_a_settings_file_of_another_type_is_refused(deck_dir):
    (deck_dir / "eos.json").write_text(json.dumps({"calculation_type": "ELASTIC"}))
    with pytest.raises(S.SpecError, match="is for an ELASTIC deck, but this step is EOS"):
        _spec(deck_dir, {"EOS_4": {"settings_file": "eos.json"}})
    (deck_dir / "eos.json").write_text(json.dumps({"calculation_type": "EOS",
                                                   "elastic_settings": {"range": [1.1, 0.9, 6]}}))
    _spec(deck_dir, {"EOS_4": {"settings_file": "eos.json"}})     # the order of RANGE is the deck writer's check


# ------------------------------------------------------------------ the step they are built from
def test_built_from_the_nearest_opt_else_the_sp():
    assert parent_index(["OPT", "SP", "ELASTIC"], 2) == 0
    assert parent_index(["OPT", "SP", "OPT2", "SP2", "EOS"], 4) == 2
    assert parent_index(["SP", "QHA"], 1) == 0
    assert parent_index(["OPT", "ELASTIC", "EOS", "OPT2"], 3) == 0     # nothing is built from a module step
    from mace.workflow.engine import WorkflowEngine
    dep = WorkflowEngine._find_dependency_in_sequence
    eng = WorkflowEngine.__new__(WorkflowEngine)
    assert dep(eng, "ELASTIC", ["OPT", "SP", "BAND", "ELASTIC"]) == "OPT"
    assert dep(eng, "CPHF", ["SP", "BAND", "CPHF"]) == "SP"


def test_a_soc_sp_before_a_module_step_is_named_only_without_an_opt():
    cfg = {"SP_2": {"soc": True}}
    assert soc_plan_problems(["OPT", "SP", "ELASTIC"], cfg) == ([], [])
    errors, warnings = soc_plan_problems(["SP", "ELASTIC"], {"SP_1": {"soc": True}})
    assert not errors and warnings and "ELASTIC follows the SOC step SP" in warnings[0]


# ------------------------------------------------------------------ planner
def test_planner_offers_them_after_opt(tmp_path):
    p = WorkflowPlanner(work_dir=tmp_path)
    for t in MODULE_STEP_TYPES:
        assert t in p.calc_types
        assert t in p._get_available_calc_types(["OPT"])
        assert p._validate_numbered_calc_addition(["OPT"], t)
        assert p._validate_numbered_calc_addition(["SP"], t)
        assert not p._validate_numbered_calc_addition(["BAND"], t)


def test_planner_custom_sequence_takes_them(tmp_path, monkeypatch):
    answers = iter(["OPT", "SP", "ELASTIC", "QHA", "done"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    assert WorkflowPlanner(work_dir=tmp_path).design_custom_workflow() == ["OPT", "SP", "ELASTIC", "QHA"]


@pytest.mark.parametrize("calc, answers, settings", [
    ("ELASTIC", ["2", "5", "", "2", "y", "n", "y"],
     {"mode": "ELAPIEZO", "numderiv": 5, "stepsize": 0.015, "nuchess": True, "hessnum2": True, "print": True}),
    ("EOS", ["0.94 1.06 6", "", "n", "y"], {"mode": "EOS", "range": [0.94, 1.06, 6], "print": True}),
    ("QHA", ["", "", "", "2 2 2", "n"], None),
    ("CPHF", ["y", "n", "", "n"], {"mode": "CPHF", "third": True}),
])
def test_planner_asks_opt2d12s_module_questions(tmp_path, monkeypatch, calc, answers, settings):
    it = iter(answers)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(it))
    cfg = WorkflowPlanner(work_dir=tmp_path).configure_module_step(calc, 3)
    assert cfg["calculation_type"] == calc and cfg["source"] == "CRYSTALOptToD12.py"
    es = json.loads(json.dumps(cfg["elastic_settings"]))
    if settings is None:
        assert es["mode"] == "QHA" and es["scelphono"] == [2, 0, 0, 0, 2, 0, 0, 0, 2]
    else:
        assert es == settings
    assert S.module_settings_problem(calc, es) is None


@pytest.mark.parametrize("calc, mem", [("ELASTIC", "7G"), ("EOS", "5G"), ("QHA", "7G"), ("CPHF", "5G")])
def test_planner_resources_are_a_crystal_job_of_a_week(tmp_path, monkeypatch, calc, mem):
    monkeypatch.setenv("MACE_SITE_CONFIG", "none")
    p = WorkflowPlanner(work_dir=tmp_path)
    assert p.get_required_scripts(calc) == ["submitcrystal23.sh"]
    res = p.get_default_resources("submitcrystal23.sh", calc)
    assert res["walltime"] == "7-00:00:00" and res["memory_per_cpu"] == mem
    assert SLURM_RESOURCE_DEFAULTS[calc]["memory_per_cpu"] == mem


# ------------------------------------------------------------------ CLI: spec -> plan -> OPT done -> module decks
def _run_to_module_decks(tmp_path, steps):
    out, deck, fake, env, root = _setup(tmp_path)
    (root / "spec.json").write_text(json.dumps({
        "version": 1, "label": "mech", "inputs": {"type": "d12", "files": [deck.name]},
        "sequence": ["OPT", "ELASTIC", "EOS", "QHA", "CPHF"], "steps": steps}))
    check = _json(_ok(_mace(root, env, "workflow", "--spec", "spec.json", "--json")))
    w = _json(_ok(_mace(root, env, "workflow", "--spec", "spec.json", "--write-plan", "--json")))
    wid = w["workflow_id"]
    _ok(_mace(root, env, "submit", "--progress-plan", w["plan_file"], deck.name))
    db = root / f".mace_context_{wid}" / "materials.db"
    _, p = _complete(root, env, fake, db, "OPT", out)
    return check, w, p, fake, env, root, db


def test_cli_spec_writes_each_module_deck_from_the_opt(tmp_path):
    steps = {"ELASTIC_2": {"overrides": {"elastic_settings": {"mode": "ELASTCON", "nuchess": True,
                                                               "hessnum2": True, "print": True}},
                           "resources": {"walltime": "2-00:00:00"}},
             "QHA_4": {"overrides": {"elastic_settings": {"temperat": "60 10 600", "scelphono": "2 2 2"}}}}
    check, w, p, fake, env, root, db = _run_to_module_decks(tmp_path, steps)
    # the check: each step's resources and settings, with where they come from
    res = {k: v for k, v in check.items() if isinstance(v, dict) and "resources" in v} or check.get("steps")
    assert res["ELASTIC_2"]["resources"]["walltime"] == "2-00:00:00"
    assert res["EOS_3"]["resources"]["walltime"] == "7-00:00:00"
    assert res["QHA_4"]["resources"]["memory_per_cpu"] == "7G"
    by_type = {r["calc_type"]: r for r in rows(db).values()}
    assert set(by_type) == {"OPT", "ELASTIC", "EOS", "QHA", "CPHF"}, p.stdout[-6000:]
    assert all(by_type[t]["slurm_job_id"] for t in MODULE_STEP_TYPES)
    assert len(fake.calls("sbatch")) == 5
    expect = {"ELASTIC": ("ELASTCON", ["ELASTCON", "NUCHESS", "HESSNUM2", "PRINT", "END"], 8),
              "EOS": ("EOS", ["EOS", "END"], 8),
              "QHA": ("QHA", ["QHA", "TEMPERAT", "60 10.0 600.0", "END"], 10),
              "CPHF": ("CPHF", ["CPHF", "END"], 8)}
    for t, (key, block, toldee) in expect.items():
        deck = Path(by_type[t]["input_file"])
        assert deck.parent.parent.name.endswith(f"_{t}"), deck
        lines = deck.read_text().splitlines()
        assert _block(lines, key) == block, (t, lines)
        assert "OPTGEOM" not in lines and "B3LYP-D3" in lines
        start = lines.index("ENDDFT")
        assert int(lines[lines.index("TOLDEE", start) + 1]) >= toldee
        assert int(lines[lines.index("MAXCYCLE", start) + 1]) >= 200
        # a CRYSTAL job (Pcrystal), not a properties job
        script = deck.with_suffix(".sh").read_text()
        assert "submit_prop" not in script and re.search(r"crystal", script, re.I)
        assert json.loads(by_type[t]["settings_json"])["workflow_id"] == w["workflow_id"]
    assert "-t 2-00:00:00" in Path(by_type["ELASTIC"]["input_file"]).with_suffix(".sh").read_text()
    lines = Path(by_type["QHA"]["input_file"]).read_text().splitlines()
    assert lines[lines.index("SCELPHONO"):lines.index("SCELPHONO") + 4] == ["SCELPHONO", "2 0 0", "0 2 0", "0 0 2"]


def test_cli_a_finished_module_step_completes_and_starts_nothing(tmp_path):
    check, w, p, fake, env, root, db = _run_to_module_decks(tmp_path, {})
    n = len(fake.calls("sbatch"))
    el, p = _complete(root, env, fake, db, "ELASTIC", HPCC / "dia_elastcon.out")
    assert rows(db)[el["calc_id"]]["status"] == "completed", p.stdout[-4000:]
    assert len(fake.calls("sbatch")) == n
    st = _json(_ok(_mace(root, env, "workflow", "--status", w["workflow_id"], "--json")))
    states = {nd["step"]: nd["state"] for nd in st["materials"][0]["nodes"]}
    assert states["ELASTIC_2"] == "completed" and states["OPT_1"] == "completed", states
