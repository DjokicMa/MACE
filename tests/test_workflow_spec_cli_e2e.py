"""Spec -> plan -> submit -> callback -> next step, and pause / resume /
cancel, through the real CLI and job callback with fake SLURM and a real
OPT output (corpus)."""
import json
import re
import shutil
from pathlib import Path

import pytest

from array_helpers import callback, rows
from conftest import find_data
from fake_slurm import FakeSlurm
from test_array_default_path_golden import _env, _mace


def _ok(p):
    assert p.returncode == 0, p.stdout[-4000:] + p.stderr[-4000:]
    return p


def _json(p):
    return json.loads(p.stdout)


def _setup(tmp_path):
    out = find_data("OPT/1_dia_opt_rev1.out")
    deck = find_data("OPT/1_dia_opt_rev1.d12")
    fake = FakeSlurm(tmp_path)
    env = _env(tmp_path, fake)
    env["XDG_CACHE_HOME"] = str(tmp_path / "cache")
    env["MACE_SITE_CONFIG"] = "none"
    root = tmp_path / "wf"
    root.mkdir()
    shutil.copy(deck, root / deck.name)
    return out, deck, fake, env, root


def _complete(root, env, fake, db, calc_type, out):
    (row,) = [r for r in rows(db).values() if r["calc_type"] == calc_type]
    work = Path(row["input_file"]).parent
    stem = Path(row["input_file"]).stem
    shutil.copy(out, work / f"{stem}.out")
    exports = dict(re.findall(r'^export (MACE_WORKFLOW_ID|MACE_CONTEXT_DIR|MACE_ISOLATION_MODE)'
                              r'="([^"]*)"', (work / f"{stem}.sh").read_text(), re.M))
    fake.set_record(row["slurm_job_id"], state="COMPLETED")
    return row, _ok(callback(env, work, db, mode="completion",
                             extra_env=dict(exports, SLURM_JOB_ID=row["slurm_job_id"])))


def test_spec_pause_resume_cancel(tmp_path):
    out, deck, fake, env, root = _setup(tmp_path)
    (root / "spec.json").write_text(json.dumps({
        "version": 1, "label": "e2e", "inputs": {"type": "d12", "files": [deck.name]},
        "template": "full_electronic",
        "steps": {"SP_2": {"overrides": {"method_modifications": {"new_functional": "PBE0"}},
                           "resources": {"walltime": "1:00:00"}},
                  "BAND_3": {"resources": {"memory": "48G"}}}}))
    w = _json(_ok(_mace(root, env, "workflow", "--spec", "spec.json", "--write-plan", "--json")))
    wid, plan_file = w["workflow_id"], w["plan_file"]
    assert w["launch"][:3] == ["mace", "submit", "--progress-plan"]

    _ok(_mace(root, env, "submit", "--progress-plan", plan_file, deck.name))
    assert len(fake.calls("sbatch")) == 1
    db = root / f".mace_context_{wid}" / "materials.db"
    (opt,) = rows(db).values()
    assert json.loads(opt["settings_json"])["workflow_id"] == wid
    # a launched plan is not launched twice
    again = _mace(root, env, "submit", "--progress-plan", plan_file, deck.name)
    assert again.returncode != 0 and "launched already" in again.stdout + again.stderr

    st = _json(_ok(_mace(root, env, "workflow", "--status", wid, "--json")))
    assert st["label"] == "e2e" and [n["state"] for n in st["materials"][0]["nodes"]][1:] == \
        ["planned"] * 3
    lst = _json(_ok(_mace(root, env, "workflow", "--list", "--json")))
    assert [x["workflow_id"] for x in lst["workflows"]] == [wid]

    # paused: the completion is recorded, nothing is submitted
    _ok(_mace(root, env, "workflow", "--pause", wid, "--json"))
    opt, p = _complete(root, env, fake, db, "OPT", out)
    assert rows(db)[opt["calc_id"]]["status"] == "completed", p.stdout[-4000:]
    assert not [r for r in rows(db).values() if r["calc_type"] == "SP"], p.stdout[-4000:]
    assert len(fake.calls("sbatch")) == 1 and "paused" in p.stdout
    st = _json(_ok(_mace(root, env, "workflow", "--status", wid, "--json")))
    assert st["state"] == "paused" and st["materials"][0]["nodes"][1]["state"] == "paused"

    # resume: the held SP goes out, with the spec's functional and walltime
    r = _json(_ok(_mace(root, env, "workflow", "--resume", wid, "--json")))
    assert r["steps_started"] == 1
    (sp,) = [x for x in rows(db).values() if x["calc_type"] == "SP"]
    assert sp["slurm_job_id"] and len(fake.calls("sbatch")) == 2
    assert "PBE0" in Path(sp["input_file"]).read_text()
    assert "-t 1:00:00" in Path(sp["input_file"]).with_suffix(".sh").read_text()

    # cancel: lists this workflow's own jobs first; --yes cancels them
    c = _mace(root, env, "workflow", "--cancel", wid, "--json")
    assert c.returncode == 1
    listed = _json(c)
    assert [j["job_id"] for j in listed["jobs"]] == [sp["slurm_job_id"]]
    assert not fake.calls("scancel")
    c = _json(_ok(_mace(root, env, "workflow", "--cancel", wid, "--json", "--yes")))
    assert c["canceled"] == [sp["slurm_job_id"]] and c["state"] == "canceled"
    (call,) = fake.calls("scancel")
    assert sp["slurm_job_id"] in json.dumps(call)
    ctl = json.loads((root / "workflow_configs" /
                      f"workflow_control_{wid[len('workflow_'):]}.json").read_text())
    assert ctl["state"] == "canceled"


def test_115_progress_plan_status_and_continue(tmp_path):
    """A plan written the 1.1.5 way (`mace submit --progress`) shows in
    --list/--status and still progresses (no control file at all)."""
    out, deck, fake, env, root = _setup(tmp_path)
    _ok(_mace(root, env, "submit", "--progress", "opt_sp", deck.name))
    (plan,) = (root / "workflow_configs").glob("workflow_plan_*.json")
    wid = json.loads(plan.read_text())["workflow_id"]
    db = root / f".mace_context_{wid}" / "materials.db"
    _complete(root, env, fake, db, "OPT", out)
    assert [r for r in rows(db).values() if r["calc_type"] == "SP"]
    st = _json(_ok(_mace(root, env, "workflow", "--status", wid, "--json")))
    assert [n["state"] for n in st["materials"][0]["nodes"]][0] == "completed"
    assert not list((root / "workflow_configs").glob("workflow_control_*"))


def test_spec_job_arrays_first_step(tmp_path):
    out, deck, fake, env, root = _setup(tmp_path)
    second = root / deck.name.replace("1_dia", "2_dia")
    shutil.copy(deck, second)
    (root / "spec.json").write_text(json.dumps({
        "version": 1, "inputs": {"type": "d12", "directory": "."}, "template": "opt_sp",
        "execution": {"job_arrays": {"first_step": True, "max_running": 5}}}))
    w = _json(_ok(_mace(root, env, "workflow", "--spec", "spec.json", "--write-plan", "--json")))
    assert json.loads(Path(w["plan_file"]).read_text())["job_arrays"] == \
        {"first_step": True, "max_running": 5}
    _ok(_mace(root, env, *w["launch"][1:]))
    (call,) = fake.calls("sbatch")
    assert "%5" in json.dumps(call)
    db = root / f".mace_context_{w['workflow_id']}" / "materials.db"
    ids = sorted(r["slurm_job_id"] for r in rows(db).values())
    assert len(ids) == 2 and all("_" in i for i in ids)


def test_cif_spec_executes_with_its_conversion_settings(tmp_path):
    """CIF start: --spec --write-plan, then the existing `mace workflow
    --execute PLAN`; the first deck is converted with the spec's settings."""
    pytest.importorskip("ase", reason="mace convert reads CIF files with ase")
    cif = find_data("CIFs/1_dia_opt_rev1.cif")
    fake = FakeSlurm(tmp_path)
    env = _env(tmp_path, fake)
    env["XDG_CACHE_HOME"] = str(tmp_path / "cache")
    env["MACE_SITE_CONFIG"] = "none"
    root = tmp_path / "wf"
    (root / "inputs").mkdir(parents=True)
    shutil.copy(cif, root / "inputs" / cif.name)
    (root / "spec.json").write_text(json.dumps({
        "version": 1, "inputs": {"type": "cif", "directory": "inputs"}, "template": "opt_sp",
        "steps": {"OPT_1": {"overrides": {"dft_functional": "PBE0"},
                            "resources": {"ntasks": 12}}}}))
    w = _json(_ok(_mace(root, env, "workflow", "--spec", "spec.json", "--write-plan", "--json")))
    assert w["launch"] == ["mace", "workflow", "--execute", w["plan_file"]]
    p = _ok(_mace(root, env, *w["launch"][1:], stdin="y\n"))
    db = root / f".mace_context_{w['workflow_id']}" / "materials.db"
    (opt,) = rows(db).values()
    deck = Path(opt["input_file"]).read_text()
    assert "PBE0" in deck, p.stdout[-3000:]
    assert "--ntasks=12" in Path(opt["input_file"]).with_suffix(".sh").read_text()
    assert len(fake.calls("sbatch")) == 1


def test_two_plans_before_launch_first_uses_its_own_templates(tmp_path):
    """Two plans written in one folder before either is launched: the
    folder's workflow_scripts/ holds the second plan's templates; the first
    plan's first submission uses the copy kept in its own context."""
    out, deck, fake, env, root = _setup(tmp_path)
    for name, ntasks in (("a.json", 12), ("b.json", 20)):
        (root / name).write_text(json.dumps({
            "version": 1, "inputs": {"type": "d12", "files": [deck.name]}, "template": "opt_sp",
            "steps": {"OPT_1": {"resources": {"ntasks": ntasks}}}}))
    a = _json(_ok(_mace(root, env, "workflow", "--spec", "a.json", "--write-plan", "--json")))
    b = _json(_ok(_mace(root, env, "workflow", "--spec", "b.json", "--write-plan", "--json")))
    assert a["workflow_id"] != b["workflow_id"]
    assert "--ntasks=20" in (root / "workflow_scripts" / "submitcrystal23_opt_1.sh").read_text()
    _ok(_mace(root, env, *a["launch"][1:]))
    db = root / f".mace_context_{a['workflow_id']}" / "materials.db"
    (opt,) = rows(db).values()
    script = Path(opt["input_file"]).with_suffix(".sh").read_text()
    assert "--ntasks=12" in script and "--ntasks=20" not in script


def _complete_without_callback(db, calc_type, out):
    """The job finished but its completion step never ran (a node crash, a
    callback that failed): the database says completed, nothing processed it."""
    import sqlite3
    (row,) = [r for r in rows(db).values() if r["calc_type"] == calc_type]
    work = Path(row["input_file"]).parent
    shutil.copy(out, work / f"{Path(row['input_file']).stem}.out")
    con = sqlite3.connect(db)
    with con:
        con.execute("UPDATE calculations SET status='completed', output_file=? WHERE calc_id=?",
                    (str(work / f"{Path(row['input_file']).stem}.out"), row["calc_id"]))
    con.close()
    return row


def test_process_starts_a_step_whose_completion_step_never_ran(tmp_path):
    """`mace workflow --process ID`: what the job callback would have done,
    for this workflow only; refused while it is paused or canceled."""
    out, deck, fake, env, root = _setup(tmp_path)
    (root / "spec.json").write_text(json.dumps({
        "version": 1, "inputs": {"type": "d12", "files": [deck.name]}, "template": "opt_sp"}))
    w = _json(_ok(_mace(root, env, "workflow", "--spec", "spec.json", "--write-plan", "--json")))
    wid = w["workflow_id"]
    _ok(_mace(root, env, "submit", "--progress-plan", w["plan_file"], deck.name))
    db = root / f".mace_context_{wid}" / "materials.db"
    opt = _complete_without_callback(db, "OPT", out)
    fake.remove_record(opt["slurm_job_id"])

    _ok(_mace(root, env, "workflow", "--pause", wid, "--json"))
    p = _mace(root, env, "workflow", "--process", wid, "--json")
    assert p.returncode == 1 and "paused" in _json(p)["error"], p.stdout + p.stderr
    assert len(fake.calls("sbatch")) == 1
    _ok(_mace(root, env, "workflow", "--resume", wid, "--json"))       # starts the SP itself
    assert len(fake.calls("sbatch")) == 2

    # a second workflow in the same folder: --process starts its step and leaves the first alone
    other = root / "other"
    other.mkdir()
    shutil.copy(deck, other / deck.name)
    (other / "spec.json").write_text(json.dumps({
        "version": 1, "inputs": {"type": "d12", "files": [deck.name]}, "template": "opt_sp"}))
    w2 = _json(_ok(_mace(other, env, "workflow", "--spec", "spec.json", "--write-plan", "--json")))
    _ok(_mace(other, env, "submit", "--progress-plan", w2["plan_file"], deck.name))
    db2 = other / f".mace_context_{w2['workflow_id']}" / "materials.db"
    _complete_without_callback(db2, "OPT", out)
    n = len(fake.calls("sbatch"))
    r = _json(_ok(_mace(other, env, "workflow", "--process", w2["workflow_id"], "--json")))
    assert r["ok"] and r["steps_started"] == 1 and r["workflow_id"] == w2["workflow_id"]
    assert len(fake.calls("sbatch")) == n + 1
    sp = [x for x in rows(db2).values() if x["calc_type"] == "SP"]
    assert len(sp) == 1 and json.loads(sp[0]["settings_json"])["workflow_id"] == w2["workflow_id"]
    # nothing left to start: 0, still ok
    r = _json(_ok(_mace(other, env, "workflow", "--process", w2["workflow_id"], "--json")))
    assert r["steps_started"] == 0
    bad = _mace(other, env, "workflow", "--process", "workflow_20990101_000000_ffff", "--json")
    assert bad.returncode == 1 and "no plan" in _json(bad)["error"]


def test_d12_spec_in_a_subfolder_writes_its_plan_with_the_decks(tmp_path):
    """--spec decks/spec.json --write-plan run from the parent folder: the
    decks are submitted from their own folder, so the plan goes there and the
    printed launch works from where it was printed; an explicit --work-dir
    elsewhere is refused before anything is written."""
    out, deck, fake, env, root = _setup(tmp_path)
    decks = root / "decks"
    decks.mkdir()
    (root / deck.name).rename(decks / deck.name)
    (decks / "spec.json").write_text(json.dumps({
        "version": 1, "inputs": {"type": "d12", "files": [deck.name]}, "template": "opt_sp"}))
    bad = _mace(root, env, "workflow", "--spec", "decks/spec.json", "--write-plan", "--work-dir", ".",
                "--json")
    assert bad.returncode == 2 and str(decks) in _json(bad)["error"], bad.stdout + bad.stderr
    assert not (root / "workflow_configs").exists() and not (decks / "workflow_configs").exists()
    assert not list(root.glob(".mace_context_*")) and not (root / "workflow_scripts").exists()

    w = _json(_ok(_mace(root, env, "workflow", "--spec", "decks/spec.json", "--write-plan", "--json")))
    assert Path(w["plan_file"]).parent == decks / "workflow_configs"
    assert not (root / "workflow_configs").exists()
    _ok(_mace(root, env, *w["launch"][1:]))
    assert len(fake.calls("sbatch")) == 1
    st = _json(_ok(_mace(decks, env, "workflow", "--list", "--json")))
    assert [x["workflow_id"] for x in st["workflows"]] == [w["workflow_id"]]


def test_cancel_reports_each_job_when_scancel_fails_for_one(tmp_path):
    """One scancel for all ids that fails for one of them: the others were
    canceled, and --cancel says so, id by id."""
    out, deck, fake, env, root = _setup(tmp_path)
    shutil.copy(deck, root / deck.name.replace("1_dia", "2_dia"))
    (root / "spec.json").write_text(json.dumps({
        "version": 1, "inputs": {"type": "d12", "directory": "."}, "template": "opt_sp"}))
    w = _json(_ok(_mace(root, env, "workflow", "--spec", "spec.json", "--write-plan", "--json")))
    _ok(_mace(root, env, *w["launch"][1:]))
    db = root / f".mace_context_{w['workflow_id']}" / "materials.db"
    a, b = sorted(r["slurm_job_id"] for r in rows(db).values())
    fake.update(scancel_refuse={a: "Job/step already completing or completed"})
    c = _mace(root, env, "workflow", "--cancel", w["workflow_id"], "--yes", "--json")
    r = _json(c)
    assert c.returncode == 1 and r["state"] == "canceled", c.stdout + c.stderr
    assert r["canceled"] == [b], r
    assert [x["job_id"] for x in r["refused"]] == [a] and "already completing" in r["refused"][0]["reason"]
    text = _mace(root, env, "workflow", "--cancel", w["workflow_id"], "--yes").stdout
    assert f"not canceled {a}" in text and b not in text.split("scancel:")[1].split("\n")[0]
