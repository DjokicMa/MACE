"""The TAU workflow step (mace/workflow/tau_step.py) without SLURM.

* options, TRANSPORT's relaxation_time and the walltime scaling;
* the axes from the deck's symmetry, and the strained decks of each
  symmetry class (one unstrained run shared by the axes);
* the job group finishing in any order, a recovered member, and the fit
  (masses given, the HPCC diamond strain series as the members' outputs);
* every fallback: a metal, too few finished runs, no m*, tau out of range,
  a deck that cannot be strained, a missing group list; strict holds;
* the workflow spec: relaxation_time auto puts a TAU step before TRANSPORT,
  its keys move, conflicts are refused, the plan gets the options and the
  SP's resources with 1.5x the walltime;
* the output scan never records a strained run as a single point.
"""
import itertools
import json
import random
import shutil
from pathlib import Path

import pytest

from conftest import REPO_ROOT
from mace.workflow import tau_step as T

STRAIN = REPO_ROOT / "tests" / "data" / "hpcc_116" / "strain"
TAIL = """END
6 2
0 0 3 2.0 1.0
13575.349682 0.00022245814352
2035.2333680 0.00172327382520
463.22562359 0.00892557153140
0 1 1 4.0 1.0
0.5 1.0 1.0
99 0
END
DFT
B3LYP
ENDDFT
TOLDEE
7
GUESSP
SHRINK
8 8
END
"""


def deck(geom="CRYSTAL", group="227", params="3.5448", atoms=("6 0.125 0.125 0.125",), head="0 0 0"):
    lines = ["test", geom] + ([head] if geom == "CRYSTAL" else []) + [group, params, str(len(atoms)), *atoms]
    return "\n".join(lines) + "\n" + TAIL


# ------------------------------------------------------------------ options
def test_options_defaults_and_checks():
    o = T.check_options({})
    assert o == {"axes": "auto", "strains": [-0.01, -0.005, 0.005, 0.01], "carrier": "auto", "m_star": "auto",
                 "strict": False, "fallback_fs": 10.0, "temperature": 300.0}
    assert T.check_options({"axes": "x, z", "strains": "0.01 -0.01", "m_star": "0.3"})["axes"] == ["x", "z"]
    assert T.check_options({"m_star": 0.3})["m_star"] == 0.3
    for bad in ({"axes": ["x", "x"]}, {"axes": ["w"]}, {"strains": [0.0, 0.01]}, {"strains": [0.1, -0.1]},
                {"strains": [0.01]}, {"carrier": "both"}, {"m_star": -1}, {"strict": "yes"},
                {"fallback_fs": 0.1}, {"wavelength": 1}):
        with pytest.raises(T.TauOptionError):
            T.check_options(bad)
    assert T.check_relaxation_time("AUTO") == "auto" and T.check_relaxation_time("25") == 25.0
    for bad in ("soon", 0.5, 1e6):
        with pytest.raises(T.TauOptionError):
            T.check_relaxation_time(bad)


@pytest.mark.parametrize("wt,scaled", [("2:00:00", "3:00:00"), ("3-00:00:00", "4-12:00:00"), ("30", "0:45:00"),
                                       ("1-12", "2-06:00:00"), ("45:00", "1:08:00"), ("soon", "soon")])
def test_walltime_scaled_by_one_and_a_half(wt, scaled):
    assert T.scale_walltime(wt) == scaled


def test_tau_before_and_insert():
    assert T.tau_before(["SP", "BAND", "TAU", "TRANSPORT"], "TRANSPORT") == "TAU"
    assert T.tau_before(["SP", "TAU", "SP2", "TRANSPORT"], "TRANSPORT") is None      # another SP between
    assert T.tau_before(["SP", "TRANSPORT"], "TRANSPORT") is None
    assert T.insert_tau(["SP", "TRANSPORT"], 1) == ["SP", "TAU", "TRANSPORT"]
    assert T.insert_tau(["SP", "TAU", "TRANSPORT", "SP2", "TRANSPORT2"], 4)[4] == "TAU2"


# ------------------------------------------------------------------ symmetry
@pytest.mark.parametrize("text,dim,axes", [
    (deck(group="227"), 3, ["x"]),
    (deck(group="136", params="4.59 2.96"), 3, ["x", "z"]),
    (deck(group="194", params="2.46 6.7"), 3, ["x", "z"]),
    (deck(group="62", params="5.0 6.0 7.0"), 3, ["x", "y", "z"]),
    (deck(group="P 4/M M M", params="4.0 5.0", head="0 1 0"), 3, ["x", "z"]),       # a symbol: by the parameters
    (deck(geom="SLAB", group="78", params="2.46"), 2, ["x"]),
    (deck(geom="SLAB", group="1", params="3.0 4.0 80.0"), 2, ["x", "y"]),
    (deck(geom="POLYMER", group="1", params="2.5"), 1, ["x"]),
])
def test_axes_from_the_symmetry(text, dim, axes):
    d, a, why = T.symmetry_axes(text)
    assert (d, a) == (dim, axes) and why


def test_a_molecule_has_no_axis():
    from mace.transport.strain import StrainError
    with pytest.raises(StrainError):
        T.symmetry_axes("m\nMOLECULE\n1\n1\n6 0 0 0\n" + TAIL)


# ------------------------------------------------------------------ an engine without SLURM
@pytest.fixture
def engine(tmp_path, monkeypatch):
    monkeypatch.delenv("MACE_WORKFLOW_ID", raising=False)
    monkeypatch.chdir(tmp_path)
    from mace.workflow.engine import WorkflowEngine
    eng = WorkflowEngine(str(tmp_path / "materials.db"), str(tmp_path))
    jobs = iter(range(9000, 9999))
    eng.submitted = []

    def script(calc_dir, name, calc_type, step_num, wid):
        p = Path(calc_dir) / f"{name}.sh"
        p.write_text("#!/bin/bash\n")
        return p

    def submit(script_path, work_dir, calc_id=None):
        eng.submitted.append(Path(script_path).name)
        return str(next(jobs))
    monkeypatch.setattr(eng, "_create_slurm_script_for_calculation", script)
    monkeypatch.setattr(eng, "_submit_calculation_to_slurm", submit)
    return eng


WID = "workflow_20261022_120000_abcd"


def _plan(eng, sequence, tau=None):
    d = Path(eng.base_work_dir) / "workflow_configs"
    d.mkdir(exist_ok=True)
    steps = {f"{t}_{i + 1}": {} for i, t in enumerate(sequence)}
    for k in steps:
        if k.startswith("TAU"):
            steps[k] = {"calculation_type": "TAU", "tau": tau or {}}
    (d / f"workflow_plan_{WID[len('workflow_'):]}.json").write_text(json.dumps(
        {"workflow_id": WID, "workflow_sequence": sequence, "step_configurations": steps}))


def _sp(eng, text, material="dia"):
    d = Path(eng.base_work_dir) / "workflow_outputs" / WID / "step_001_SP" / material
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{material}_sp.d12").write_text(text)
    eng.db.create_material(material_id=material, formula="C")
    cid = eng.db.create_calculation(material_id=material, calc_type="SP", input_file=str(d / f"{material}_sp.d12"),
                                    work_dir=str(d), settings={"workflow_id": WID})
    eng.db.update_calculation_status(cid, "completed", output_file=str(d / f"{material}_sp.out"))
    return cid


@pytest.mark.parametrize("text,n,axes", [
    (deck(group="227"), 5, ["x"]),
    (deck(group="136", params="4.59 2.96"), 9, ["x", "z"]),
    (deck(group="62", params="5.0 6.0 7.0"), 13, ["x", "y", "z"]),
    (deck(geom="SLAB", group="1", params="3.0 4.0 80.0"), 9, ["x", "y"]),
    (deck(geom="POLYMER", group="1", params="2.5"), 5, ["x"]),
])
def test_strained_decks_per_symmetry_class(engine, text, n, axes):
    seq = ["SP", "TAU", "TRANSPORT"]
    _plan(engine, seq)
    ids = T.generate(engine, _sp(engine, text), "TAU", WID)
    assert len(ids) == n == T.member_count(axes, T.DEFAULTS["strains"]) and len(engine.submitted) == n
    folder = Path(engine.base_work_dir) / "workflow_outputs" / WID / "step_002_TAU" / "dia"
    manifest = json.loads((folder / T.MANIFEST).read_text())
    assert manifest["axes"] == axes
    names = sorted(m["member"] for m in manifest["members"])
    assert [m for m in names if m.endswith("_p0pct")] == ["x_p0pct"]           # one unstrained run
    for ax in axes:
        assert sorted(m for m in names if m.startswith(ax) and not m.endswith("p0pct")) == \
            sorted(f"{ax}_{t}" for t in ("m1pct", "m0_5pct", "p0_5pct", "p1pct"))
    for m in manifest["members"]:
        text_m = (folder / m["deck"]).read_text()
        assert "GUESSP" not in text_m and "TOLDEE\n8" in text_m
    rows = T.group_rows(engine.db, "dia", "TAU")
    assert {json.loads(r["settings_json"])["tau_member"] for r in rows} == set(names)
    assert all(r["status"] == "submitted" and r["work_dir"] == str(folder) for r in rows)
    assert T.started(engine, "dia", "TAU", WID)
    # nothing finished: TRANSPORT waits
    assert T.transport_gate(engine, "dia", WID, seq, "TRANSPORT") == ("wait", None)


# ------------------------------------------------------------------ the group and the fit
def _group(engine, tau=None, seq=("SP", "TAU", "TRANSPORT")):
    _plan(engine, list(seq), tau)
    T.generate(engine, _sp(engine, deck()), "TAU", WID)
    rows = {json.loads(r["settings_json"])["tau_member"]: r for r in T.group_rows(engine.db, "dia", "TAU")}
    return rows, list(seq)


def _done(engine, row, member, edit=None):
    out = Path(row["input_file"]).with_suffix(".out")
    text = (STRAIN / f"dia_sp_dp_{member}.out").read_text()
    out.write_text(edit(text) if edit else text)
    engine.db.update_calculation_status(row["calc_id"], "completed", output_file=str(out))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_group_fits_once_after_its_last_member_in_any_order(engine, seed):
    rows, seq = _group(engine, {"m_star": 0.5, "carrier": "hole"})
    order = list(rows)
    random.Random(seed).shuffle(order)
    for i, m in enumerate(order):
        assert T.try_fit(engine, "dia", WID, "TAU", seq) is None
        _done(engine, rows[m], m)
    s = T.try_fit(engine, "dia", WID, "TAU", seq)
    assert s["fallback"] is False and s["carrier"] == "hole" and s["carrier_reason"] == "the step's carrier"
    hole = s["results"]["hole"]["x"]
    assert hole["m_c"] == 0.5 and hole["m_c_source"] == "the TAU step's m_star"
    assert s["low_confidence"] is (hole["E1_r2"] < 0.9) and s["used_fs"] == float(f"{s['computed_fs']:.4g}")
    folder = Path(rows[order[0]]["work_dir"])
    saved = json.loads((folder / "tau_hole_x.json").read_text())
    assert saved["tau_fs"] == s["computed_fs"] and saved["fit"]["axis"] == "x"
    assert T.transport_gate(engine, "dia", WID, seq, "TRANSPORT") == ("go", s)
    # final: a later call reads it
    assert T.try_fit(engine, "dia", WID, "TAU", seq) == s


def test_a_recovered_member_replaces_the_failed_one(engine):
    rows, seq = _group(engine, {"m_star": 0.5})
    for m in ("x_p0pct", "x_m1pct", "x_p1pct", "x_p0_5pct"):
        _done(engine, rows[m], m)
    failed = rows["x_m0_5pct"]
    engine.db.update_calculation_status(failed["calc_id"], "resubmitted")
    # the recovery run: its own deck, no tau_member (lineage not carried): found by its deck's name
    rdeck = Path(failed["input_file"]).with_name(Path(failed["input_file"]).stem + "_recovery_20261022.d12")
    shutil.copy(failed["input_file"], rdeck)
    rid = engine.db.create_calculation(material_id="dia", calc_type="TAU", input_file=str(rdeck),
                                       work_dir=failed["work_dir"], settings={"workflow_id": WID})
    engine.db.update_calculation_status(rid, "submitted", slurm_job_id="77")
    manifest = json.loads((Path(failed["work_dir"]) / T.MANIFEST).read_text())
    st = T.group_state(engine.db, "dia", "TAU", manifest)
    assert st["active"] == ["x_m0_5pct"] and not st["finished"]
    assert T.try_fit(engine, "dia", WID, "TAU", seq) is None
    _done(engine, engine.db.get_calculation(rid), "x_m0_5pct")
    s = T.try_fit(engine, "dia", WID, "TAU", seq)
    assert s["members"] == {"total": 5, "completed": 5, "failed": []} and not s["fallback"]


def test_fallback_too_few_runs_and_strict_holds(engine):
    rows, seq = _group(engine, {"m_star": 0.5, "strict": True})
    _done(engine, rows["x_p0pct"], "x_p0pct")
    _done(engine, rows["x_p1pct"], "x_p1pct")
    for m in ("x_m1pct", "x_m0_5pct", "x_p0_5pct"):
        engine.db.update_calculation_status(rows[m]["calc_id"], "failed")
    s = T.try_fit(engine, "dia", WID, "TAU", seq)
    assert s["fallback"] and s["used_fs"] == 10.0 and "too few strained runs" in s["reason"]
    assert s["held"] and T.transport_gate(engine, "dia", WID, seq, "TRANSPORT")[0] == "held"
    assert "fallback" in T.line(s)


def test_fallback_metal(engine):
    rows, seq = _group(engine, {"m_star": 0.5})
    no_edges = lambda t: "\n".join(ln for ln in t.splitlines()
                                   if "TOP OF VALENCE" not in ln and "BOTTOM OF VIRTUAL" not in ln)
    for m, r in rows.items():
        _done(engine, r, m, no_edges)
    s = T.try_fit(engine, "dia", WID, "TAU", seq)
    assert s["fallback"] and s["used_fs"] == 10.0 and s["reason"].startswith("a metal or no gap")
    assert not s["held"] and T.transport_gate(engine, "dia", WID, seq, "TRANSPORT")[0] == "go"


def test_fallback_no_effective_mass(engine):
    rows, seq = _group(engine)                               # m* auto, and no BAND or DOSS in the plan
    for m, r in rows.items():
        _done(engine, r, m)
    s = T.try_fit(engine, "dia", WID, "TAU", seq)
    assert s["fallback"] and s["reason"].startswith("no effective mass")
    assert "add a BAND or DOSS step" in s["reason"]


def test_fallback_tau_out_of_range(engine):
    rows, seq = _group(engine, {"m_star": 1e-4, "carrier": "hole"})
    for m, r in rows.items():
        _done(engine, r, m)
    s = T.try_fit(engine, "dia", WID, "TAU", seq)
    assert s["fallback"] and "outside 1-100000 fs" in s["reason"] and s["computed_fs"] > 1e5


def test_low_confidence_is_used_and_flagged(engine):
    rows, seq = _group(engine, {"m_star": 0.5})              # electrons: diamond's CBM along x, R^2 0.54
    for m, r in rows.items():
        _done(engine, r, m)
    s = T.try_fit(engine, "dia", WID, "TAU", seq)
    assert not s["fallback"] and s["low_confidence"] and not s["held"]
    assert "MACE has no doping" in s["carrier_reason"] and "low confidence" in T.line(s)


def test_the_fit_waits_for_running_band_and_doss(engine):
    rows, seq = _group(engine, seq=("SP", "BAND", "TAU", "TRANSPORT"))
    band = engine.db.create_calculation(material_id="dia", calc_type="BAND", input_file="x_band.d3",
                                        work_dir=str(engine.base_work_dir), settings={"workflow_id": WID})
    engine.db.update_calculation_status(band, "running", slurm_job_id="5")
    for m, r in rows.items():
        _done(engine, r, m)
    assert T.try_fit(engine, "dia", WID, "TAU", seq) is None
    engine.db.update_calculation_status(band, "failed")         # no mass from it: the fit runs, falls back
    s = T.try_fit(engine, "dia", WID, "TAU", seq)
    assert s["fallback"] and s["reason"].startswith("no effective mass")


def test_a_deck_that_cannot_be_strained_does_not_hold_transport(engine):
    _plan(engine, ["SP", "TAU", "TRANSPORT"])
    sp = _sp(engine, "m\nMOLECULE\n1\n1\n6 0 0 0\n" + TAIL)
    assert T.generate(engine, sp, "TAU", WID) == []
    assert T.started(engine, "dia", "TAU", WID)
    gate, s = T.transport_gate(engine, "dia", WID, ["SP", "TAU", "TRANSPORT"], "TRANSPORT")
    assert gate == "go" and s["fallback"] and "strained decks could not be written" in s["reason"]


def test_a_group_without_its_list_falls_back(engine):
    rows, seq = _group(engine, {"m_star": 0.5})
    (Path(next(iter(rows.values()))["work_dir"]) / T.MANIFEST).unlink()
    s = T.try_fit(engine, "dia", WID, "TAU", seq)
    assert s["fallback"] and "tau_group.json is missing" in s["reason"]


def test_relaxtim_goes_into_the_opt2d3_configuration():
    s = {"used_fs": 2198.0, "fallback": False, "low_confidence": True, "carrier": "electron", "axis": "x",
         "step": "TAU", "json": "tau_electron_x.json"}
    env = T.apply_to_config({"version": "1.0", "configuration": {"relaxation_time": 10}}, s)
    assert env["configuration"]["relaxation_time"] == 2198.0
    assert T.apply_to_config({"relaxation_time": 10}, s)["relaxation_time"] == 2198.0
    p = T.provenance(s, Path("/w/step_004_TAU/dia"))
    assert p["relaxation_time_fs"] == 2198.0 and p["json"].endswith("step_004_TAU/dia/tau_electron_x.json")
    # the writer: RELAXTIM is written when it is not CRYSTAL's 10 fs
    import sys
    sys.path.insert(0, str(REPO_ROOT / "Crystal_d3"))
    from CRYSTALOptToD3 import D3Generator as W
    w = W.__new__(W)
    w.input_file, w.input_dir, w.base_name = Path("/nonexistent/x.out"), Path("/nonexistent"), "x"
    w._newk_mesh = lambda d12: ["8 8", "16"]
    assert "RELAXTIM\n2198.0" in w._write_transport_d3({"relaxation_time": 2198.0})
    assert "RELAXTIM" not in w._write_transport_d3({"relaxation_time": 10.0})


# ------------------------------------------------------------------ the spec
def _spec(tmp_path, **raw):
    from mace.workflow import spec as S
    (tmp_path / "a.d12").write_text(deck())
    base = {"version": 1, "inputs": {"type": "d12", "files": ["a.d12"]}}
    return S, S.validate_spec({**base, **raw}, tmp_path)


def test_spec_auto_inserts_the_tau_step_and_moves_the_keys(tmp_path):
    S, sp = _spec(tmp_path, sequence=["SP", "TRANSPORT", "CHARGE+POTENTIAL"],
                  steps={"TRANSPORT_2": {"overrides": {"relaxation_time": "auto"}},
                         "CHARGE+POTENTIAL_3": {"resources": {"walltime": "1:00:00"}}})
    assert sp["sequence"] == ["SP", "TAU", "TRANSPORT", "CHARGE+POTENTIAL"]
    assert sorted(sp["steps"]) == ["CHARGE+POTENTIAL_4", "TRANSPORT_3"]
    # a template whose chain changes becomes a custom sequence
    S, sp = _spec(tmp_path, template="transport_analysis",
                  steps={"TRANSPORT_3": {"overrides": {"relaxation_time": "auto"}}})
    assert sp["sequence"] == ["OPT", "SP", "TAU", "TRANSPORT"] and sp["template"] is None
    # a TAU step already there is used, not doubled
    S, sp = _spec(tmp_path, sequence=["SP", "TAU", "TRANSPORT"],
                  steps={"TAU_2": {"overrides": {"carrier": "hole", "axes": ["x"]}},
                         "TRANSPORT_3": {"overrides": {"relaxation_time": "auto"}}})
    assert sp["sequence"] == ["SP", "TAU", "TRANSPORT"]


@pytest.mark.parametrize("raw,msg", [
    ({"sequence": ["SP", "TAU", "TRANSPORT"], "steps": {"TRANSPORT_3": {"overrides": {"relaxation_time": 20}}}},
     "whose tau it would not use"),
    ({"sequence": ["SP", "BAND"], "steps": {"BAND_2": {"overrides": {"relaxation_time": "auto"}}}},
     "a TRANSPORT step's setting"),
    ({"sequence": ["SP", "TAU"], "steps": {"TAU_2": {"overrides": {"carrier": "both"}}}}, "carrier"),
    ({"sequence": ["SP", "TRANSPORT"], "steps": {"TRANSPORT_2": {"overrides": {"relaxation_time": "later"}}}},
     "default, auto or a time"),
    ({"sequence": ["OPT", "TAU"]}, "TAU needs one of SP before it"),
])
def test_spec_refusals(tmp_path, raw, msg):
    from mace.workflow.spec import SpecError
    with pytest.raises(SpecError, match=msg):
        _spec(tmp_path, **raw)


def test_spec_relaxation_time_beside_a_settings_file(tmp_path):
    from mace.workflow.spec import SpecError
    (tmp_path / "t.json").write_text(json.dumps({"calculation_type": "TRANSPORT", "relaxation_time": 30}))
    S, sp = _spec(tmp_path, sequence=["SP", "TRANSPORT"],
                  steps={"TRANSPORT_2": {"settings_file": "t.json", "overrides": {"relaxation_time": "auto"}}})
    assert sp["sequence"] == ["SP", "TAU", "TRANSPORT"]
    with pytest.raises(SpecError, match="its RELAXTIM is the relaxation time"):
        _spec(tmp_path, sequence=["SP", "TRANSPORT"],
              steps={"TRANSPORT_2": {"settings_file": "t.json", "overrides": {"relaxation_time": 30}}})


def test_spec_to_plan_options_resources_and_fixed_time(tmp_path):
    S, sp = _spec(tmp_path, sequence=["SP", "TRANSPORT"],
                  steps={"SP_1": {"resources": {"walltime": "1-00:00:00", "ntasks": 16}},
                         "TRANSPORT_2": {"overrides": {"relaxation_time": "auto"}}})
    assert sp["sequence"] == ["SP", "TAU", "TRANSPORT"]
    sp["steps"]["TAU_2"] = {"overrides": {"carrier": "hole", "strict": True}}
    d = S.resolve_defaults(sp, tmp_path)
    tau = d["TAU_2"]
    assert tau["resources"]["walltime"] == "1-12:00:00" and tau["resources"]["ntasks"] == 16
    assert tau["origins"]["walltime"] == "spec (SP_1, 1.5x)"
    assert tau["settings"]["tau"]["carrier"] == "hole" and tau["settings"]["tau"]["strict"] is True
    assert d["TRANSPORT_3"]["settings"]["d3_config"]["relaxation_time"] == 10      # the TAU step sets it later
    S, sp = _spec(tmp_path, sequence=["SP", "TRANSPORT"],
                  steps={"TRANSPORT_2": {"overrides": {"relaxation_time": 35}}})
    assert sp["sequence"] == ["SP", "TRANSPORT"]
    assert S.resolve_defaults(sp, tmp_path)["TRANSPORT_2"]["settings"]["d3_config"]["relaxation_time"] == 35
    # without a TAU step the strained runs' resources are the built-in SP's, 1.5x the walltime
    S, sp = _spec(tmp_path, sequence=["SP", "TAU"])
    d = S.resolve_defaults(sp, tmp_path)
    assert d["TAU_2"]["resources"]["walltime"] == T.scale_walltime(d["SP_1"]["resources"]["walltime"])


def test_the_output_scan_skips_strained_runs(tmp_path):
    from mace.database.populate_completed_jobs import scan_for_completed_calculations
    for step in ("step_004_TAU", "step_002_SP"):
        d = tmp_path / step / "dia"
        d.mkdir(parents=True)
        shutil.copy(STRAIN / "dia_sp_dp_x_p1pct.out", d / "dia_sp_dp_x_p1pct.out")
    found = scan_for_completed_calculations(tmp_path)
    assert [Path(c["output_file"]).parent.parent.name for c in found] == ["step_002_SP"]
