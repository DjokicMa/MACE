"""Workflow fixes from the 1.1.6 pre-release review, through the real CLI
(mace_cli as a subprocess) with the fake SLURM stubs."""
import json
import shutil
from pathlib import Path

import pytest

from conftest import find_data
from fake_slurm import FakeSlurm
from test_array_default_path_golden import _env, _mace


def _ok(p):
    assert p.returncode == 0, p.stdout[-4000:] + p.stderr[-4000:]
    return p


def _json(p):
    return json.loads(p.stdout)


def _root(tmp_path, kind):
    fake = FakeSlurm(tmp_path)
    env = _env(tmp_path, fake)
    env["XDG_CACHE_HOME"] = str(tmp_path / "cache")
    env["MACE_SITE_CONFIG"] = "none"
    root = tmp_path / "wf"
    (root / "inputs").mkdir(parents=True)
    if kind == "cif":
        src = find_data("CIFs/1_dia_opt_rev1.cif")
        shutil.copy(src, root / "inputs" / src.name)
        inputs = {"type": "cif", "directory": "inputs"}
    else:
        src = find_data("OPT/1_dia_opt_rev1.d12")
        shutil.copy(src, root / src.name)
        inputs = {"type": "d12", "files": [src.name]}
    return fake, env, root, inputs


# ---------------------------------------------------------------- W1
@pytest.mark.parametrize("kind", ["cif", "d12"])
@pytest.mark.parametrize("sequence,steps", [
    (["OPT", "SP", "SP2"], {"SP2_3": {"overrides": {"soc": True}}}),
    (["OPT", "OPT2", "OPT3"], {}),
    (["OPT", "SP", "BAND", "SP2", "DOSS"], {}),
])
def test_numbered_steps_build_in_the_check_and_the_plan(tmp_path, kind, sequence, steps):
    """SP2, OPT3 (any numbered repeat) in a spec: the check answers
    and --write-plan writes the plan, configured as the base type."""
    fake, env, root, inputs = _root(tmp_path, kind)
    (root / "spec.json").write_text(json.dumps({"version": 1, "inputs": inputs,
                                                "sequence": sequence, "steps": steps}))
    chk = _json(_ok(_mace(root, env, "workflow", "--spec", "spec.json", "--json")))
    assert chk["ok"], chk
    w = _json(_ok(_mace(root, env, "workflow", "--spec", "spec.json", "--write-plan", "--json")))
    plan = json.loads(Path(w["plan_file"]).read_text())
    cfgs = plan["step_configurations"]
    keys = [f"{t}_{i + 1}" for i, t in enumerate(sequence)]
    assert list(cfgs) == keys
    for t, k in zip(sequence, keys):
        if k.endswith("_1"):
            continue
        base = t.rstrip("0123456789")
        assert cfgs[k]["calculation_type"] == base, (k, cfgs[k])
        assert cfgs[k]["slurm_config"]
    if "SP2_3" in steps:
        assert cfgs["SP2_3"]["soc"] is True


# ---------------------------------------------------------------- W3
@pytest.mark.parametrize("kind,sequence,steps", [
    ("cif", ["OPT", "SP", "ORBITALS"], {"SP_2": {"overrides": {"soc": True}}}),
    ("cif", ["SP", "ORBITALS"], {"SP_1": {"overrides": {"soc": True}}}),
    ("d12", ["OPT", "SP", "SP2", "ORBITALS"], {"SP2_3": {"overrides": {"soc": True}}}),
])
def test_orbitals_after_an_overrides_soc_sp_is_refused_by_the_check(tmp_path, kind, sequence, steps):
    fake, env, root, inputs = _root(tmp_path, kind)
    (root / "spec.json").write_text(json.dumps({"version": 1, "inputs": inputs,
                                                "sequence": sequence, "steps": steps}))
    for extra in ((), ("--write-plan",)):
        p = _mace(root, env, "workflow", "--spec", "spec.json", *extra, "--json")
        assert p.returncode != 0, p.stdout
        assert not _json(p)["ok"] and "ORBITALS" in _json(p)["error"]
    assert not (root / "workflow_configs").exists()


def test_a_scalar_sp2_after_a_soc_sp_lets_orbitals_run(tmp_path):
    """"soc": false on SP2 is a scalar run: ORBITALS after it is fine."""
    fake, env, root, inputs = _root(tmp_path, "d12")
    (root / "spec.json").write_text(json.dumps({
        "version": 1, "inputs": inputs, "sequence": ["OPT", "SP", "SP2", "ORBITALS"],
        "steps": {"SP_2": {"overrides": {"soc": True}}, "SP2_3": {"overrides": {"soc": False}}}}))
    assert _json(_ok(_mace(root, env, "workflow", "--spec", "spec.json", "--json")))["ok"]


# ---------------------------------------------------------------- W2
def test_overrides_next_to_a_settings_file_are_refused(tmp_path):
    """The engine generates a step with a settings file from that file
    alone: overrides beside it are refused by the check (which showed them
    before), not silently dropped at run time."""
    fake, env, root, inputs = _root(tmp_path, "d12")
    (root / "sp.json").write_text(json.dumps({"functional": "PBE0"}))
    (root / "band.json").write_text(json.dumps({"calculation_type": "BAND",
                                                "configuration": {"n_points": 1000}}))
    for steps in ({"SP_2": {"settings_file": "sp.json",
                            "overrides": {"method_modifications": {"new_functional": "HSE06"}}}},
                  {"BAND_3": {"settings_file": "band.json", "overrides": {"d3_config": {"n_points": 200}}}}):
        (root / "spec.json").write_text(json.dumps({"version": 1, "inputs": inputs,
                                                    "sequence": ["OPT", "SP", "BAND"], "steps": steps}))
        p = _mace(root, env, "workflow", "--spec", "spec.json", "--json")
        assert p.returncode != 0 and "settings file" in _json(p)["error"], p.stdout
    # a CIF start's first step: the overrides are merged into the conversion settings
    fake, env, root, inputs = _root(tmp_path / "c", "cif")
    (root / "opt.json").write_text(json.dumps({"calculation_type": "OPT", "dft_functional": "PBE"}))
    (root / "spec.json").write_text(json.dumps({
        "version": 1, "inputs": inputs, "sequence": ["OPT", "SP"],
        "steps": {"OPT_1": {"settings_file": "opt.json", "overrides": {"dft_functional": "PBE0"}}}}))
    r = _json(_ok(_mace(root, env, "workflow", "--spec", "spec.json", "--json")))
    assert r["steps"]["OPT_1"]["settings"]["dft_functional"] == "PBE0"


# ---------------------------------------------------------------- W4
def _submitted_opt_sp(tmp_path):
    fake, env, root, inputs = _root(tmp_path, "d12")
    (root / "spec.json").write_text(json.dumps({"version": 1, "inputs": inputs, "template": "opt_sp"}))
    w = _json(_ok(_mace(root, env, "workflow", "--spec", "spec.json", "--write-plan", "--json")))
    _ok(_mace(root, env, *w["launch"][1:]))
    return fake, env, root, w["workflow_id"]


def _complete_unstamped(db, out):
    import sqlite3
    con = sqlite3.connect(db)
    (cid, inp, sj) = con.execute("SELECT calc_id, input_file, settings_json FROM calculations").fetchone()
    s = json.loads(sj or "{}")
    s.pop("workflow_id", None)
    o = Path(inp).with_suffix(".out")
    shutil.copy(out, o)
    with con:
        con.execute("UPDATE calculations SET status='completed', output_file=?, settings_json=? "
                    "WHERE calc_id=?", (str(o), json.dumps(s), cid))
    con.close()
    return cid


def _settings(db, cid):
    import sqlite3
    con = sqlite3.connect(db)
    (sj,) = con.execute("SELECT settings_json FROM calculations WHERE calc_id=?", (cid,)).fetchone()
    con.close()
    return json.loads(sj or "{}")


@pytest.mark.parametrize("command", ["--process", "--resume"])
def test_process_starts_the_next_step_of_an_unstamped_row(tmp_path, command):
    """A completed row of the workflow's own database without the
    workflow_id stamp (as the output scan records it): --process/--resume
    start its next step instead of marking it processed with none."""
    fake, env, root, wid = _submitted_opt_sp(tmp_path)
    db = root / f".mace_context_{wid}" / "materials.db"
    cid = _complete_unstamped(db, find_data("OPT/1_dia_opt_rev1.out"))
    n = len(fake.calls("sbatch"))
    r = _json(_ok(_mace(root, env, "workflow", command, wid, "--json")))
    assert r["steps_started"] == 1, r
    assert len(fake.calls("sbatch")) == n + 1
    assert _settings(db, cid).get("workflow_processed") is True


def test_process_leaves_a_row_unprocessed_when_the_plan_is_gone(tmp_path):
    """No plan to read: nothing is started and the row is not marked
    processed, so a later --process (with the plan back) can start it."""
    from mace.workflow.engine import WorkflowEngine
    fake, env, root, wid = _submitted_opt_sp(tmp_path)
    db = root / f".mace_context_{wid}" / "materials.db"
    cid = _complete_unstamped(db, find_data("OPT/1_dia_opt_rev1.out"))
    plan = root / "workflow_configs" / f"workflow_plan_{wid[len('workflow_'):]}.json"
    assert plan.is_file()
    eng = WorkflowEngine(str(db), str(root))
    eng.get_workflow_sequence = lambda w: None
    assert eng.process_completed_calculations(workflow_id=wid) == 0
    assert not _settings(db, cid).get("workflow_processed")


# ---------------------------------------------------------------- W5
def test_cancel_without_squeue_cancels_nothing_and_says_so(tmp_path):
    fake, env, root, wid = _submitted_opt_sp(tmp_path)
    blind = dict(env, PATH="/usr/bin:/bin")
    c = _mace(root, blind, "workflow", "--cancel", wid, "--yes", "--json")
    r = _json(c)
    assert c.returncode != 0 and r["ok"] is False and r["unverified"], c.stdout + c.stderr
    assert "SLURM could not be asked" in r["error"] and r["state"] == "active"
    from mace.workflow import control
    assert control.state_of(control.read_control(root, wid)) == "active"
    assert not fake.calls("scancel")
    t = _mace(root, blind, "workflow", "--cancel", wid, "--yes")
    assert t.returncode != 0 and "no job was canceled" in t.stderr
    # with squeue back, the cancel goes through
    ok = _json(_ok(_mace(root, env, "workflow", "--cancel", wid, "--yes", "--json")))
    assert ok["state"] == "canceled" and ok["canceled"]


# ---------------------------------------------------------------- W6
def test_status_reads_the_exported_database_of_export_and_delete(tmp_path):
    fake, env, root, wid = _submitted_opt_sp(tmp_path)
    ctx = root / f".mace_context_{wid}"
    _complete_unstamped(ctx / "materials.db", find_data("OPT/1_dia_opt_rev1.out"))
    dest = root / "workflow_results" / wid
    dest.mkdir(parents=True)
    shutil.copy(ctx / "materials.db", dest / f"{wid}_materials.db")
    shutil.rmtree(ctx)
    st = _json(_ok(_mace(root, env, "workflow", "--status", wid, "--json")))
    assert st["database"] == str(dest / f"{wid}_materials.db") and st["archived"] is True, st
    assert st["state"] != "planned"
    assert st["materials"][0]["nodes"][0]["state"] == "completed"
    # nothing is processed in an exported database
    r = _json(_ok(_mace(root, env, "workflow", "--process", wid, "--json")))
    assert r["steps_started"] == 0


# ---------------------------------------------------------------- W7
@pytest.mark.parametrize("argv", [("workflow", "--list", "--json"), ("recover", "--json", "--dry-run")])
def test_json_output_is_pure_json_without_no_banner(tmp_path, argv):
    import subprocess
    import sys
    from test_array_default_path_golden import MACE_CLI
    fake, env, root, inputs = _root(tmp_path, "d12")
    p = subprocess.run([sys.executable, str(MACE_CLI), *argv], cwd=str(root), env=env,
                       capture_output=True, text=True, timeout=300)
    json.loads(p.stdout)


# ---------------------------------------------------------------- E M4
def test_bare_workflow_status_lists_the_workflows_and_writes_nothing(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    fake, env, root, wid = _submitted_opt_sp(tmp_path)
    p = _ok(_mace(empty, env, "workflow", "--status"))
    assert not list(empty.iterdir()), list(empty.iterdir())
    assert "mace.py" not in p.stdout and "No workflow plans" in p.stdout
    before = sorted(x.name for x in root.iterdir())
    p = _ok(_mace(root, env, "workflow", "--status"))
    assert wid in p.stdout and "running" in p.stdout and "mace workflow --status <ID>" in p.stdout, p.stdout
    assert "No recent calculations" not in p.stdout and "mace.py" not in p.stdout
    assert sorted(x.name for x in root.iterdir()) == before
    r = _json(_ok(_mace(root, env, "workflow", "--status", "--json")))
    assert [w["workflow_id"] for w in r["workflows"]] == [wid]


# ---------------------------------------------------------------- E M10
def test_follow_up_steps_honor_the_workflows_max_jobs(tmp_path):
    """execution.max_jobs 1, two decks: an explicit max_jobs is a hard cap on
    every job of the workflow, its first step included (mace/queue/budget.py:
    per-job admission). The second OPT waits held; the first OPT's completion
    starts its SP (next steps first); once that SP ended, the drain of
    `mace workflow --process` submits the held OPT."""
    from array_helpers import rows
    fake, env, root, inputs = _root(tmp_path, "d12")
    deck = Path(inputs["files"][0])
    shutil.copy(root / deck.name, root / deck.name.replace("1_dia", "2_dia"))
    (root / "spec.json").write_text(json.dumps({
        "version": 1, "inputs": {"type": "d12", "directory": "."}, "template": "opt_sp",
        "execution": {"max_jobs": 1}}))
    w = _json(_ok(_mace(root, env, "workflow", "--spec", "spec.json", "--write-plan", "--json")))
    _ok(_mace(root, env, *w["launch"][1:]))
    db = root / f".mace_context_{w['workflow_id']}" / "materials.db"
    out = find_data("OPT/1_dia_opt_rev1.out")
    opts = sorted((r for r in rows(db).values() if r["calc_type"] == "OPT"), key=lambda r: r["material_id"])
    assert len(opts) == 2
    assert sorted(r["status"] for r in opts) == ["held", "submitted"]
    first = next(r for r in opts if r["status"] == "submitted")
    second = next(r for r in opts if r["status"] == "held")
    assert json.loads(second["settings_json"])["held_reason"] == "workflow_cap"

    def complete(row):
        import re
        from array_helpers import callback
        work = Path(row["input_file"]).parent
        stem = Path(row["input_file"]).stem
        shutil.copy(out, work / f"{stem}.out")
        exports = dict(re.findall(r'^export (MACE_WORKFLOW_ID|MACE_CONTEXT_DIR|MACE_ISOLATION_MODE)'
                                  r'="([^"]*)"', (work / f"{stem}.sh").read_text(), re.M))
        fake.set_record(row["slurm_job_id"], state="COMPLETED")
        return _ok(callback(env, work, db, mode="completion",
                            extra_env=dict(exports, SLURM_JOB_ID=row["slurm_job_id"])))

    n = len(fake.calls("sbatch"))
    p = complete(first)
    sps = [r for r in rows(db).values() if r["calc_type"] == "SP"]
    assert [r["status"] for r in sps] == ["submitted"], p.stdout[-3000:]
    assert rows(db)[second["calc_id"]]["status"] == "held"
    assert len(fake.calls("sbatch")) == n + 1
    # that SP ends: the held OPT starts (here by --process, as the next
    # completion step would)
    import sqlite3
    con = sqlite3.connect(db)
    with con:
        con.execute("UPDATE calculations SET status='completed' WHERE calc_id=?", (sps[0]["calc_id"],))
    con.close()
    fake.remove_record(sps[0]["slurm_job_id"])
    r = _json(_ok(_mace(root, env, "workflow", "--process", w["workflow_id"], "--json")))
    assert r["released"] == 1 and len(fake.calls("sbatch")) == n + 2
    assert rows(db)[second["calc_id"]]["status"] == "submitted"


# ---------------------------------------------------------------- soc_fmixing
def test_soc_fmixing_reaches_the_plan_and_opt2d12(tmp_path, monkeypatch):
    from mace.workflow.engine import WorkflowEngine
    from mace.workflow.spec import SpecError, build_plan, load_spec
    monkeypatch.setenv("MACE_SITE_CONFIG", "none")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("MACE_WORKFLOW_ID", raising=False)
    monkeypatch.chdir(tmp_path)
    shutil.copy(find_data("OPT/1_dia_opt_rev1.d12"), tmp_path / "a.d12")

    def spec(ov, name="s.json"):
        (tmp_path / name).write_text(json.dumps({"version": 1, "inputs": {"type": "d12", "files": ["a.d12"]},
                                                 "sequence": ["OPT", "SP"], "steps": {"SP_2": {"overrides": ov}}}))
        return load_spec(tmp_path / name)

    for bad, msg in (({"soc": True, "soc_fmixing": 101}, "0 to 100"),
                     ({"soc": True, "soc_fmixing": True}, "whole number"),
                     ({"soc_fmixing": 50}, 'only with "soc": true')):
        with pytest.raises(SpecError, match=msg):
            spec(bad, "bad.json")
    _, wid, plan = build_plan(spec({"soc": True, "soc_fmixing": 50}), tmp_path)
    assert plan["step_configurations"]["SP_2"]["soc_fmixing"] == 50
    eng = WorkflowEngine.__new__(WorkflowEngine)
    eng.base_work_dir = tmp_path
    cfg = eng._build_numbered_calc_config(wid, "SP", "SP", None)
    assert cfg["soc"] is True and cfg["soc_fmixing"] == 50


def test_the_builder_writes_soc_fmixing_as_the_steps_override(tmp_path):
    from mace.gui import workflow_build as wb
    d = tmp_path / "in"
    d.mkdir()
    f = d / "A.cif"
    f.write_text("data_A\n")
    inp = wb.Inputs("cif", d, [f], ["A"])
    files = wb.spec_files(inp, {"sequence": ["OPT", "SP"], "steps": {
        "SP_2": {"soc": {"soc": True, "soc_fmixing": 60}}}})
    assert json.loads(files["spec.json"])["steps"]["SP_2"]["overrides"] == {"soc": True, "soc_fmixing": 60}
    with pytest.raises(wb.BuildError, match="0 to 100"):
        wb.spec_files(inp, {"sequence": ["OPT", "SP"], "steps": {"SP_2": {"soc": {"soc": True, "soc_fmixing": -1}}}})


# ---------------------------------------------------------------- B-gui V1
def test_a_cif_spec_converts_and_submits_only_the_picked_cifs(tmp_path):
    """Three CIFs in the folder, the spec picks two: exactly those two are
    converted and submitted (the preview's count is the run's)."""
    from array_helpers import rows
    fake, env, root, inputs = _root(tmp_path, "cif")
    src = next((root / "inputs").glob("*.cif"))
    for n in ("a", "b", "c"):
        shutil.copy(src, root / "inputs" / f"{n}.cif")
    src.unlink()
    (root / "spec.json").write_text(json.dumps({
        "version": 1, "inputs": {"type": "cif", "files": ["inputs/a.cif", "inputs/c.cif"]},
        "template": "opt_sp"}))
    w = _json(_ok(_mace(root, env, "workflow", "--spec", "spec.json", "--write-plan", "--json")))
    p = _ok(_mace(root, env, *w["launch"][1:], stdin="y\n"))
    db = root / f".mace_context_{w['workflow_id']}" / "materials.db"
    mats = sorted(r["material_id"] for r in rows(db).values())
    assert len(fake.calls("sbatch")) == 2, p.stdout[-3000:]
    assert len(mats) == 2 and not any(m.startswith("b") for m in mats), mats
    decks = sorted(x.name for x in (root / "workflow_outputs").rglob("*.d12"))
    assert not any(d.startswith("b") for d in decks), decks


# ---------------------------------------------------------------- B-gui V2
@pytest.mark.parametrize("kind", ["d12", "cif"])
def test_a_plan_with_orbitals_after_a_soc_sp_is_not_launched(tmp_path, kind):
    """A plan edited (or written by another route) to read ORBITALS from a
    SOC SP: --execute (CIF; SOC on the first step's conversion settings) and
    submit --progress-plan (decks; SOC on the SP step) refuse it before any
    job is submitted."""
    fake, env, root, inputs = _root(tmp_path, kind)
    seq = ["OPT", "SP", "ORBITALS"] if kind == "d12" else ["SP", "ORBITALS"]
    (root / "spec.json").write_text(json.dumps({"version": 1, "inputs": inputs, "sequence": seq}))
    w = _json(_ok(_mace(root, env, "workflow", "--spec", "spec.json", "--write-plan", "--json")))
    plan_file = Path(w["plan_file"])
    plan = json.loads(plan_file.read_text())
    if kind == "d12":
        plan["step_configurations"]["SP_2"]["soc"] = True
    else:
        plan["cif_conversion_config"]["soc"] = True
    plan_file.write_text(json.dumps(plan))
    p = _mace(root, env, *w["launch"][1:], stdin="y\n")
    text = p.stdout + p.stderr
    assert "ORBITALS follows the SOC step" in text, text[-3000:]
    assert ("Not executing" if kind == "cif" else "Error") in text
    assert not fake.calls("sbatch")


# ---------------------------------------------------------------- W S1
def test_a_moved_folders_spec_settings_copy_is_found(tmp_path, capsys):
    """A plan's settings file path from before the folder moved: the engine
    finds the spec's copy in workflow_configs/spec_<id>/ (d12 and d3 steps)
    and says so when it finds none."""
    from mace.workflow.engine import WorkflowEngine
    from mace.workflow.spec import spec_dir
    wid = "workflow_20261016_120000_abcd"
    copy = spec_dir(tmp_path, wid) / "BAND_3_config.json"
    copy.parent.mkdir(parents=True)
    copy.write_text("{}")
    eng = WorkflowEngine.__new__(WorkflowEngine)
    eng.base_work_dir = tmp_path
    gone = "/old/place/workflow_configs/spec_20261016_120000_abcd/BAND_3_config.json"
    assert eng._find_plan_config(gone, wid, "expert_d3_configs") == copy
    assert eng._find_plan_config("/old/none.json", wid, "expert_d3_configs") is None
    eng._get_plan_step_config = lambda w, t: {"expert_config_file": gone.replace("BAND_3", "SP_2")}
    (spec_dir(tmp_path, wid) / "SP_2_config.json").write_text("{}")
    assert eng._plan_expert_config_file(wid, "SP") == spec_dir(tmp_path, wid) / "SP_2_config.json"


# ---------------------------------------------------------------- W S3
def test_the_gui_list_keeps_the_newest_workflows(tmp_path, monkeypatch):
    from mace.gui import workflow as gw
    folders = [tmp_path / "a", tmp_path / "b", tmp_path / "c"]
    ids = {folders[0]: ["workflow_20260101_000000_aaaa", "workflow_20261001_000000_aaab"],
           folders[1]: ["workflow_20260501_000000_bbbb"],
           folders[2]: ["workflow_20261010_000000_cccc"]}
    monkeypatch.setattr(gw, "MAX_WORKFLOWS", 2)
    monkeypatch.setattr(gw, "workflow_folders", lambda root: folders)
    monkeypatch.setattr(gw.wf_status, "list_workflows", lambda f, squeue=None, use_squeue=False: [
        {"workflow_id": w, "state": "running", "plan_file": None} for w in ids[f]])
    r = gw.list_view(tmp_path, None)
    assert [w["workflow_id"] for w in r["workflows"]] == ["workflow_20261010_000000_cccc",
                                                          "workflow_20261001_000000_aaab"]
    assert r["more"] == 2
