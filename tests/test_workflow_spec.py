"""Workflow specs (mace/workflow/spec.py): validation, per-step settings and
resources with their origins, and the golden rule — a spec that changes
nothing writes exactly today's template plan, job scripts included."""
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from mace.workflow.common.constants import WORKFLOW_TEMPLATES
from mace.workflow.spec import SpecError, apply_spec, build_plan, load_spec, resolve_defaults

DECK = """TESTGEOM
CRYSTAL
0 0 0
227
3.5
1
6 0.125 0.125 0.125
OPTGEOM
FULLOPTG
ENDOPT
END
6 2
0 0 2 2. 1.
 8.0 1.0
 2.0 1.0
99 0
END
DFT
B3LYP
END
SHRINK
8 8
END
"""


@pytest.fixture(autouse=True)
def _plain_site(monkeypatch, tmp_path):
    monkeypatch.setenv("MACE_SITE_CONFIG", "none")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for k in list(os.environ):
        if k.startswith("MACE_SLURM_") or k in ("MACE_WORKFLOW_ID", "MACE_CRYSTAL_MODULE"):
            monkeypatch.delenv(k, raising=False)


def _spec(tmp_path, body, name="spec.json"):
    p = tmp_path / name
    p.write_text(json.dumps(body))
    return p


def _deck(tmp_path, name="mat_opt.d12"):
    p = tmp_path / name
    p.write_text(DECK)
    return p


def _strip(plan, *ids):
    plan = json.loads(json.dumps(plan))
    for k in ("created", "workflow_id", "spec", "label"):
        plan.pop(k, None)
    text = json.dumps(plan, sort_keys=True)
    for i in ids:
        text = text.replace(i, "WID").replace(i[len("workflow_"):], "WID")
    return json.loads(text)


def _own_scripts(folder, wid):
    d = folder / f".mace_context_{wid}" / "workflow_scripts"
    return {p.name: p.read_text().replace(wid, "WID") for p in sorted(d.glob("*.sh"))}


@pytest.mark.parametrize("template", sorted(WORKFLOW_TEMPLATES))
def test_golden_deck_spec_equals_progress_template(tmp_path, template, monkeypatch):
    """`mace workflow --spec` with no settings == `mace submit --progress TEMPLATE`:
    the same plan, the same job scripts byte for byte."""
    from mace.workflow.manual_progress import build_progress_plan
    monkeypatch.chdir(tmp_path)
    deck = _deck(tmp_path)
    plan_a, wid_a, _ = build_progress_plan(tmp_path, "materials.db", [deck], template)
    spec = load_spec(_spec(tmp_path, {"version": 1, "inputs": {"type": "d12", "files": [deck.name]},
                                      "template": template}))
    plan_b, wid_b, plan = build_plan(spec, tmp_path)
    a = json.loads(plan_a.read_text())
    assert _strip(a, wid_a) == _strip(plan, wid_b)
    assert plan["spec"]["template"] == template
    assert _own_scripts(tmp_path, wid_a) == _own_scripts(tmp_path, wid_b)
    assert _own_scripts(tmp_path, wid_b), "the workflow keeps its own templates"


@pytest.mark.parametrize("template", ["full_electronic", "opt_sp_freq", "double_opt"])
def test_golden_cif_spec_equals_quick_start(tmp_path, template, monkeypatch):
    from mace.run_mace import create_quick_workflow_plan
    monkeypatch.chdir(tmp_path)
    cif = tmp_path / "A.cif"
    cif.write_text("data_A\n")
    ref = create_quick_workflow_plan(tmp_path, [cif], "cif", list(WORKFLOW_TEMPLATES[template]),
                                     SimpleNamespace(max_jobs=None, work_dir=str(tmp_path)))
    spec = load_spec(_spec(tmp_path, {"version": 1, "inputs": {"type": "cif", "directory": "."},
                                      "template": template}))
    _, wid, plan = build_plan(spec, tmp_path)
    assert _strip(ref) == _strip(plan, wid)
    assert plan["workflow_id"] == wid


def test_step_settings_and_resources_reach_plan_and_scripts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    deck = _deck(tmp_path)
    band = tmp_path / "band.json"
    band.write_text(json.dumps({"version": "1.0", "type": "d3_configuration",
                                "calculation_type": "BAND", "configuration": {"n_points": 500}}))
    freq = tmp_path / "freq.json"
    freq.write_text(json.dumps({"calculation_type": "FREQ"}))
    spec = load_spec(_spec(tmp_path, {
        "version": 1, "label": "test-run", "inputs": {"type": "d12", "files": [deck.name]},
        "template": "complete",
        "steps": {"OPT_1": {"resources": {"ntasks": 16, "partition": "short"}},
                  "SP_2": {"overrides": {"method_modifications": {"new_functional": "HSE06"}}},
                  "BAND_3": {"settings_file": "band.json", "resources": {"memory": "48G",
                                                                          "exclude": "node-[1-2]"}},
                  "FREQ_5": {"settings_file": "freq.json"}},
        "execution": {"walltime": "4:00:00", "max_jobs": 50}}))
    plan_file, wid, plan = build_plan(spec, tmp_path)
    steps = plan["step_configurations"]
    assert steps["SP_2"]["method_modifications"] == {"new_functional": "HSE06"}
    copy_dir = tmp_path / "workflow_configs" / f"spec_{wid[len('workflow_'):]}"
    assert steps["BAND_3"]["expert_config_file"] == str(copy_dir / "BAND_3_config.json")
    assert json.loads((copy_dir / "BAND_3_config.json").read_text())["configuration"] == {"n_points": 500}
    assert steps["FREQ_5"]["expert_config_file"] == str(copy_dir / "FREQ_5_config.json")
    assert (copy_dir / "spec.json").is_file()
    assert plan["label"] == "test-run"
    # unchanged steps are the template's
    assert "expert_config_file" not in steps["DOSS_4"] and "method_modifications" not in steps["DOSS_4"]
    scripts = tmp_path / f".mace_context_{wid}" / "workflow_scripts"
    band_sh = (scripts / "submit_prop_band_3.sh").read_text()
    assert "--mem=48G" in band_sh and "--exclude=node-[1-2]" in band_sh and "-t 4:00:00" in band_sh
    opt_sh = (scripts / "submitcrystal23_opt_1.sh").read_text()
    assert "--ntasks=16" in opt_sh and "short" in opt_sh
    doss_sh = (scripts / "submit_prop_doss_4.sh").read_text()
    assert "--mem=48G" not in doss_sh and "-t 4:00:00" in doss_sh


@pytest.mark.parametrize("calc", ["TRANSPORT", "CHARGE+POTENTIAL"])
def test_properties_settings_file_names_its_own_step(tmp_path, calc):
    """An opt2d3 settings file for another properties run is refused (the
    engine would hand it to CRYSTALOptToD3 as this step's); one without a
    calculation type, or for this step, is copied for the engine."""
    _deck(tmp_path)
    key = f"{calc}_3"
    for name, ctype, ok in (("other.json", "BAND", False), ("bare.json", None, True), ("own.json", calc, True)):
        cfg = {"temperature_range": [200, 600, 50]}
        if ctype:
            cfg["calculation_type"] = ctype
        (tmp_path / name).write_text(json.dumps({"version": "1.0", "type": "d3_configuration",
                                                 "configuration": cfg}))
        body = {"version": 1, "inputs": {"type": "d12", "files": ["mat_opt.d12"]},
                "sequence": ["OPT", "SP", calc], "steps": {key: {"settings_file": name}}}
        if ok:
            spec = load_spec(_spec(tmp_path, body))
            plan = {"step_configurations": {"OPT_1": {}, "SP_2": {}, key: {}}}
            apply_spec(plan, spec)
            assert plan["step_configurations"][key]["expert_config_file"] == str(tmp_path / name)
        else:
            with pytest.raises(SpecError, match=f"is for a BAND step, but this step is {re.escape(calc)}"):
                load_spec(_spec(tmp_path, body))


def test_cif_first_step_settings_file_is_the_conversion_config(tmp_path):
    cif = tmp_path / "A.cif"
    cif.write_text("data_A\n")
    opts = tmp_path / "cif2d12.json"
    opts.write_text(json.dumps({"calculation_type": "OPT", "dft_functional": "PBE0"}))
    spec = load_spec(_spec(tmp_path, {"version": 1, "inputs": {"type": "cif", "files": ["A.cif"]},
                                      "sequence": ["OPT", "SP"],
                                      "steps": {"OPT_1": {"settings_file": "cif2d12.json",
                                                          "overrides": {"basis_set": "DEF2-TZVP"}}}}))
    plan = {"step_configurations": {"OPT_1": {"source": "cif_conversion"}, "SP_2": {}},
            "cif_conversion_config": {}}
    origins = apply_spec(plan, spec)
    assert plan["cif_conversion_config"] == {"calculation_type": "OPT", "dft_functional": "PBE0",
                                             "basis_set": "DEF2-TZVP"}
    assert origins["OPT_1"]["settings"].startswith("spec settings_file")


@pytest.mark.parametrize("body,msg", [
    ({"version": 2}, "version"),
    ({"inputs": {"type": "d12", "files": ["mat_opt.d12"]}, "template": "nope"}, "unknown template"),
    ({"inputs": {"type": "d12", "files": ["missing.d12"]}, "template": "opt_sp"}, "not found"),
    ({"inputs": {"type": "d12", "files": ["mat_opt.d12"]}, "sequence": ["OPT", "BOGUS"]}, "unknown calculation"),
    ({"inputs": {"type": "d12", "files": ["mat_opt.d12"]}, "sequence": ["OPT", "SP", "SP"]}, "twice"),
    ({"inputs": {"type": "d12", "files": ["mat_opt.d12"]}, "template": "opt_sp",
      "steps": {"BAND_3": {}}}, "not in the sequence"),
    ({"inputs": {"type": "d12", "files": ["mat_opt.d12"]}, "template": "opt_sp",
      "steps": {"SP_2": {"overrides": {"d3_config": {}}}}}, "not read for a SP step"),
    ({"inputs": {"type": "d12", "files": ["mat_opt.d12"]}, "template": "opt_sp",
      "steps": {"OPT_1": {"overrides": {"optimization_settings": {}}}}}, "runs the decks as they are"),
    ({"inputs": {"type": "d12", "files": ["mat_opt.d12"]}, "template": "opt_sp",
      "steps": {"SP_2": {"resources": {"memory": "1G", "memory_per_cpu": "1G"}}}}, "not both"),
    ({"inputs": {"type": "d12", "files": ["mat_opt.d12"]}, "template": "opt_sp",
      "steps": {"SP_2": {"resources": {"walltime": "forever"}}}}, "SLURM time"),
    ({"inputs": {"type": "d12", "files": ["mat_opt.d12"]}, "template": "opt_sp",
      "execution": {"job_arrays": {"max_running": 0}}}, "max_running"),
    ({"inputs": {"type": "cif", "files": ["mat_opt.d12"]}, "template": "opt_sp"}, "not .cif"),
    ({"inputs": {"type": "d12", "files": ["mat_opt.d12"]}, "template": "opt_sp", "extra": 1}, "unknown spec key"),
    # never applied, so not accepted (a d12 follow-up always inherits its parent's settings)
    ({"inputs": {"type": "d12", "files": ["mat_opt.d12"]}, "template": "opt_sp",
      "steps": {"SP_2": {"inherit": False, "overrides": {"method_modifications": {"new_functional": "PBE0"}}}}},
     r"unknown key\(s\) inherit"),
])
def test_validation_says_what_is_wrong(tmp_path, body, msg):
    _deck(tmp_path)
    body = dict({"version": 1}, **body)
    with pytest.raises(SpecError, match=msg):
        load_spec(_spec(tmp_path, body))


def test_resolve_defaults_reports_origins(tmp_path, monkeypatch):
    deck = _deck(tmp_path)
    site_toml = tmp_path / "site.toml"
    site_toml.write_text("[resources.properties]\nntasks = 4\n")
    monkeypatch.setenv("MACE_SITE_CONFIG", str(site_toml))
    monkeypatch.setenv("MACE_SLURM_MEM_PER_CPU", "3G")
    spec = load_spec(_spec(tmp_path, {
        "version": 1, "inputs": {"type": "d12", "files": [deck.name]}, "template": "full_electronic",
        "steps": {"DOSS_4": {"resources": {"walltime": "1:00:00"}},
                  "SP_2": {"overrides": {"method_modifications": {"new_functional": "PBE0"}}}}}))
    steps = resolve_defaults(spec, tmp_path)
    assert steps["OPT_1"]["settings_origin"] == "deck"
    assert steps["SP_2"]["settings_origin"] == "spec overrides"
    assert steps["BAND_3"]["settings_origin"] == "builtin (opt2d3 defaults)"
    assert steps["OPT_1"]["origins"]["ntasks"] == "builtin"
    assert steps["OPT_1"]["resources"]["memory_per_cpu"] == "3G"
    assert steps["OPT_1"]["origins"]["memory_per_cpu"] == "env"
    assert steps["BAND_3"]["resources"]["ntasks"] == 4
    assert steps["BAND_3"]["origins"]["ntasks"] == "site [resources.properties]"
    assert steps["DOSS_4"]["origins"]["walltime"] == "spec"
    assert not list(tmp_path.glob("workflow_configs")), "resolve_defaults writes nothing"


def test_progress_report_uses_the_one_template_list():
    from mace.database.analysis.workflow_progress import WorkflowProgress
    assert WorkflowProgress.WORKFLOWS == {k: list(v) for k, v in WORKFLOW_TEMPLATES.items()}


def test_planner_menu_order(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from mace.workflow.planner import WorkflowPlanner
    p = WorkflowPlanner(str(tmp_path), "materials.db")
    keys = list(p.workflow_templates)
    assert keys[:9] == ["basic_opt", "opt_sp", "full_electronic", "double_opt", "complete",
                        "transport_analysis", "charge_analysis", "combined_analysis", "custom"]
    assert keys[9:] == ["opt_sp_freq"]
    assert p.workflow_templates["opt_sp_freq"] == WORKFLOW_TEMPLATES["opt_sp_freq"]


def _account_lines(folder, wid):
    import re
    out = {}
    for p in sorted((folder / f".mace_context_{wid}" / "workflow_scripts").glob("*.sh")):
        m = re.search(r"#SBATCH -A ([^\s']+)", p.read_text())
        out[p.name] = m.group(1) if m else None
    return out


@pytest.mark.parametrize("kind", ["d12", "cif"])
@pytest.mark.parametrize("case", ["member", "not_member", "env", "site"])
def test_resolve_defaults_names_the_account_the_job_scripts_get(tmp_path, monkeypatch, kind, case):
    """Every step names the account its job script will get, None when the
    script has no -A line (SLURM's default account): the builder shows this."""
    from conftest import find_data
    from fake_slurm import FakeSlurm
    from mace.submission import site
    accounts = {"member": ["mendoza_q", "general"], "not_member": ["physics"], "env": ["physics"],
                "site": ["chem"]}[case]
    fake = FakeSlurm(tmp_path / "fs", {"assoc": accounts, "default_account": accounts[0]})
    monkeypatch.setenv("PATH", fake.env["PATH"])
    monkeypatch.setenv("FAKE_SLURM_STATE", fake.env["FAKE_SLURM_STATE"])
    monkeypatch.setenv("FAKE_SLURM_LOG", fake.env["FAKE_SLURM_LOG"])
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("MACE_HOSTNAME", "node1.cluster.example.org")
    monkeypatch.setenv("USER", "student")
    if case == "env":
        monkeypatch.setenv("MACE_SLURM_ACCOUNT", "physics")
    if case == "site":
        toml = tmp_path / "site.toml"
        toml.write_text('[slurm]\naccount = "chem"\n')
        monkeypatch.setenv("MACE_SITE_CONFIG", str(toml))
    site.forget_slurm_cache()
    monkeypatch.setattr(site, "_SLURM_MEMO", {})
    if kind == "d12":
        files = [_deck(tmp_path).name]
    else:
        import shutil
        shutil.copy(find_data("CIFs/1_dia_opt_rev1.cif"), tmp_path / "A.cif")
        files = ["A.cif"]
    spec = load_spec(_spec(tmp_path, {"version": 1, "inputs": {"type": kind, "files": files},
                                      "template": "opt_sp"}))
    want = {"member": "mendoza_q", "not_member": None, "env": "physics", "site": "chem"}[case]
    steps = resolve_defaults(spec, tmp_path)
    assert {k: v["resources"].get("account", "<absent>") for k, v in steps.items()} == \
        {"OPT_1": want, "SP_2": want}
    origin = {"member": "builtin", "not_member": "builtin", "env": "env", "site": "site"}[case]
    assert {v["origins"]["account"] for v in steps.values()} == {origin}
    _plan, wid, plan = build_plan(spec, tmp_path)
    if kind == "d12":                  # a deck plan writes its job scripts now
        assert set(_account_lines(tmp_path, wid).values()) == {want}
    else:                              # a CIF plan's scripts come from these resources at --execute
        from mace.submission.site import DEFAULTS
        got = {k: c["slurm_config"]["resources"].get("account", DEFAULTS["account"])
               for k, c in plan["step_configurations"].items()}
        assert set(got.values()) == {want}


# ------------------------------------------------------------------ spin-orbit coupling (1.1.6)
# SOC is a per-step setting of SP steps: the overrides "soc", "soc_ecp",
# "soc_basis", "soc_guess" and "smear" go into the plan step, the engine hands
# them to opt2d12 (_build_numbered_calc_config), and opt2d12 writes the SOC deck.
import sys as _sys  # noqa: E402
_sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Crystal_d12"))
import soc_ecp as _E  # noqa: E402

SOC_ON = {"soc": True, "soc_ecp": "stutsc", "soc_basis": "pob-tzvp-rev2-soc", "smear": 0.005}


def _soc_spec(tmp_path, sequence, steps, kind="d12"):
    if kind == "d12":
        files = [_deck(tmp_path).name]
    else:
        (tmp_path / "A.cif").write_text("data_A\n")
        files = ["A.cif"]
    return {"version": 1, "inputs": {"type": kind, "files": files}, "sequence": sequence, "steps": steps}


def test_soc_overrides_on_an_sp_step_reach_the_plan_step(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = load_spec(_spec(tmp_path, _soc_spec(tmp_path, ["OPT", "SP", "BAND"], {
        "SP_2": {"overrides": {**SOC_ON, "method_modifications": {"new_functional": "PBE0"}}}})))
    assert spec["steps"]["SP_2"]["overrides"]["smear"] == 0.005
    _, wid, plan = build_plan(spec, tmp_path)
    sp = plan["step_configurations"]["SP_2"]
    assert {k: sp[k] for k in SOC_ON} == SOC_ON
    assert sp["method_modifications"] == {"new_functional": "PBE0"}
    # the engine reads them back from the plan and gives opt2d12 all of them
    from mace.workflow.engine import WorkflowEngine
    eng = WorkflowEngine.__new__(WorkflowEngine)
    eng.base_work_dir = tmp_path
    monkeypatch.setattr(eng, "_load_workflow_plan", lambda w: plan if w == wid else None, raising=False)
    cfg = eng._build_numbered_calc_config(wid, "SP", "SP", None)
    assert cfg == {"calculation_type": "SP", "soc": True, "soc_ecp": "stutsc",
                   "soc_basis": "pob-tzvp-rev2-soc", "smear": 0.005, "functional": "PBE0",
                   # no use_dispersion: opt2d12 keeps the parent's D3 (follow-ups preserve the parent)
                   "method_modifications": {"new_functional": "PBE0"}}
    # the steps without SOC keys are as before
    assert eng._build_numbered_calc_config(wid, "BAND", "BAND", None) is None


def test_the_engine_hands_smear_to_opt2d12(tmp_path):
    from mace.workflow.engine import WorkflowEngine
    wf = "workflow_20261016_120000"
    cfg_dir = tmp_path / "workflow_configs"
    cfg_dir.mkdir()
    (cfg_dir / f"workflow_plan_{wf[len('workflow_'):]}.json").write_text(json.dumps(
        {"workflow_id": wf, "workflow_sequence": ["OPT", "SP", "SP2"],
         "step_configurations": {"SP_2": {"calculation_type": "SP", "soc": True, "smear": 0.01},
                                 "SP2_3": {"calculation_type": "SP", "inherit_settings": True}}}))
    eng = WorkflowEngine.__new__(WorkflowEngine)
    eng.base_work_dir = tmp_path
    assert eng._build_numbered_calc_config(wf, "SP", "SP", None) == {
        "calculation_type": "SP", "soc": True, "smear": 0.01}
    # no SOC key: no config, exactly as before
    assert eng._build_numbered_calc_config(wf, "SP2", "SP", None) is None


@pytest.mark.parametrize("overrides,msg", [
    ({"soc": "yes"}, "soc must be true or false"),
    ({"soc": True, "soc_ecp": "ecp60"}, "soc_ecp must be"),
    ({"soc": True, "soc_basis": "def2-tzvp"}, "soc_basis must be one of"),
    ({"soc": True, "soc_guess": "guessp"}, "soc_guess must be"),
    ({"soc": True, "smear": 0}, "smear must be a width in hartree"),
    ({"soc": True, "smear": True}, "smear must be a width in hartree"),
    ({"soc": True, "smear": "wide"}, "smear must be a width in hartree"),
    ({"soc_ecp": "inpsoc"}, 'only with "soc": true'),
    ({"soc": False, "smear": 0.005}, 'only with "soc": true'),
])
def test_soc_values_are_checked_with_the_clis_checks(tmp_path, overrides, msg):
    with pytest.raises(SpecError, match=msg):
        load_spec(_spec(tmp_path, _soc_spec(tmp_path, ["OPT", "SP"], {"SP_2": {"overrides": overrides}})))


def test_soc_false_and_a_restart_guess_are_accepted_on_an_sp_step(tmp_path):
    spec = load_spec(_spec(tmp_path, _soc_spec(tmp_path, ["OPT", "SP", "SP2"], {
        "SP_2": {"overrides": {"soc": True, "soc_guess": "guesspnoso"}},
        "SP2_3": {"overrides": {"soc": False}}})))
    assert spec["steps"]["SP_2"]["overrides"] == {"soc": True, "soc_guess": "guesspnoso"}
    assert spec["steps"]["SP2_3"]["overrides"] == {"soc": False}


@pytest.mark.parametrize("sequence,step,reason", [
    (["OPT", "SP", "OPT2"], "OPT2_3", _E.unsupported_message("OPTGEOM")),
    (["OPT", "SP", "FREQ"], "FREQ_3", _E.unsupported_message("FREQCALC")),
    (["OPT", "SP", "BAND"], "BAND_3", "A properties step has no SOC setting"),
    (["OPT", "SP", "CHARGE+POTENTIAL"], "CHARGE+POTENTIAL_3", "A properties step has no SOC setting"),
])
def test_soc_on_a_step_that_is_not_a_single_point_is_refused_with_the_reason(tmp_path, sequence, step, reason):
    for ov in ({"soc": True}, {"smear": 0.005}, {"soc": False}):
        with pytest.raises(SpecError) as e:
            load_spec(_spec(tmp_path, _soc_spec(tmp_path, sequence, {step: {"overrides": ov}})))
        assert f"steps.{step}.overrides" in str(e.value) and reason in str(e.value)


def test_soc_with_a_settings_file_on_the_same_step_is_refused(tmp_path):
    (tmp_path / "sp.json").write_text(json.dumps({"calculation_type": "SP"}))
    with pytest.raises(SpecError, match="settings file"):
        load_spec(_spec(tmp_path, _soc_spec(tmp_path, ["OPT", "SP"], {
            "SP_2": {"settings_file": "sp.json", "overrides": {"soc": True}}})))


def test_soc_on_a_cif_start_sp_is_the_conversion_config(tmp_path):
    """A CIF start converts with the plan's cif_conversion_config (mace
    convert's options file, which writes SOC decks): its first SP step's SOC
    overrides go there."""
    spec = load_spec(_spec(tmp_path, _soc_spec(tmp_path, ["SP", "BAND"], {
        "SP_1": {"overrides": {"functional": "PBE0", **SOC_ON}}}, kind="cif")))
    plan = {"step_configurations": {"SP_1": {"source": "cif_conversion"}, "BAND_2": {}},
            "cif_conversion_config": {"calculation_type": "SP", "basis_set_type": "EXTERNAL"}}
    apply_spec(plan, spec)
    assert plan["cif_conversion_config"] == {"calculation_type": "SP", "basis_set_type": "EXTERNAL",
                                             "functional": "PBE0", **SOC_ON}
    assert plan["step_configurations"]["SP_1"] == {"source": "cif_conversion"}


@pytest.mark.parametrize("sequence,overrides,msg", [
    (["OPT", "SP"], {"soc": True}, _E.unsupported_message("OPTGEOM")),
    (["SP"], {"soc": True, "soc_guess": "guesspso"}, _E.NO_PARENT_FOR_GUESS),
    (["SP"], {"soc": True, "soc_ecp": "nope"}, "soc_ecp must be"),
])
def test_soc_on_a_cif_start_is_checked(tmp_path, sequence, overrides, msg):
    key = f"{sequence[0]}_1"
    with pytest.raises(SpecError) as e:
        load_spec(_spec(tmp_path, _soc_spec(tmp_path, sequence, {key: {"overrides": overrides}}, kind="cif")))
    assert msg in str(e.value)


def test_steps_that_cannot_start_from_a_soc_run_are_named(tmp_path, monkeypatch, capsys):
    """The planner's own SOC check (soc_plan_problems): a FREQ straight after a
    SOC single point would be refused when it is generated; the check says
    so on the step (mace workflow --spec --json) and --write-plan warns."""
    monkeypatch.chdir(tmp_path)
    spec = load_spec(_spec(tmp_path, _soc_spec(tmp_path, ["OPT", "SP", "FREQ", "BAND"], {
        "SP_2": {"overrides": {"soc": True}}})))
    steps = resolve_defaults(spec, tmp_path)
    assert steps["FREQ_3"]["warnings"] and "follows the SOC step SP" in steps["FREQ_3"]["warnings"][0]
    assert steps["BAND_4"]["warnings"] == [] and steps["SP_2"]["warnings"] == []
    build_plan(spec, tmp_path)
    out = capsys.readouterr().out
    assert "Warning: FREQ follows the SOC step SP" in out
    # without SOC: no warnings anywhere
    plain = load_spec(_spec(tmp_path, _soc_spec(tmp_path, ["OPT", "SP", "FREQ"], {}), name="plain.json"))
    assert all(v["warnings"] == [] for v in resolve_defaults(plain, tmp_path).values())
