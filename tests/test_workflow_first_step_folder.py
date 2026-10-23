"""A workflow started from a folder of existing calculations (1.1.6).

`mace workflow --spec SPEC --write-plan` then `mace submit --progress-plan
PLAN` (what the GUI's New Workflow runs) used to submit each deck in place: a
deck whose job script had been edited stopped the launch ("would not reach N
job script(s)"), a finished output had to be overwritten, and a deck whose
name was not <material>_<type> got a second copy beside it. A workflow now
leaves that folder as it is: every deck runs in its own step folder,
workflow_outputs/<id>/step_NNN_<TYPE>/<material>/, with a job script written
there, and a deck whose own output already finished is not run again (that
output, read where it is, starts the next steps). `mace submit --progress
TEMPLATE` (no plan file) does the same since; plain `mace submit` still runs
in place and keeps an edited job script.

The folder (tests/data/existing_calcs, real CRYSTAL23 diamond OPT):
  dia_opt      finished OPT (.out), a hand-edited .sh and a .f9
  fresh_opt    never run, a hand-edited .sh
  guess_opt    never run, GUESSP from its own .f9
  stopped_opt  an output cut off mid-run
"""
import hashlib
import json
import re
import shutil
import sqlite3
from pathlib import Path

import pytest

from array_helpers import callback, rows
from conftest import REPO_ROOT
from fake_slurm import FakeSlurm
from test_array_default_path_golden import _env, _mace

DATA = Path(__file__).resolve().parent / "data" / "existing_calcs"
F9 = b"not a real density matrix: the workflow copies it, never reads it\n"


def _ok(p):
    assert p.returncode == 0, p.stdout[-4000:] + p.stderr[-4000:]
    return p


def _sums(folder):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.is_file()}


def _folder(tmp_path):
    fake = FakeSlurm(tmp_path)
    env = _env(tmp_path, fake)
    env["XDG_CACHE_HOME"] = str(tmp_path / "cache")
    env["MACE_SITE_CONFIG"] = "none"
    root = tmp_path / "calcs"
    shutil.copytree(DATA, root)
    # .f9 files are not committed (*.f9 is ignored): a few bytes stand in
    for name in ("dia_opt.f9", "guess_opt.f9"):
        (root / name).write_bytes(F9)
    return fake, env, root


def _launch(root, env, template="opt_sp", files=None):
    (root / "spec.json").write_text(json.dumps({
        "version": 1, "template": template,
        "inputs": {"type": "d12", **({"files": files} if files else {"directory": "."})}}))
    w = json.loads(_ok(_mace(root, env, "workflow", "--spec", "spec.json", "--write-plan", "--json")).stdout)
    assert w["launch"][:3] == ["mace", "submit", "--progress-plan"]
    return w, _mace(root, env, *w["launch"][1:])


def _by(db, calc_type):
    return {r["material_id"]: r for r in rows(db).values() if r["calc_type"] == calc_type}


def test_a_folder_of_existing_calculations_starts_without_touching_it(tmp_path):
    fake, env, root = _folder(tmp_path)
    before = _sums(root)
    w, p = _launch(root, env)
    _ok(p)
    log = p.stdout + p.stderr
    assert "would not reach" not in log and "--overwrite-sh" not in log, log[-3000:]
    wid = w["workflow_id"]
    # the user's files: byte for byte as they were, and nothing new beside them
    after = _sums(root)
    assert {k: after[k] for k in before} == before
    assert sorted(set(after) - set(before)) == ["spec.json"], sorted(set(after) - set(before))

    db = root / f".mace_context_{wid}" / "materials.db"
    opt = _by(db, "OPT")
    assert sorted(opt) == ["dia", "fresh", "guess", "stopped"]
    # the finished OPT is not run again: its row is its own deck and output, in place
    assert opt["dia"]["status"] == "completed" and not opt["dia"]["slurm_job_id"]
    assert Path(opt["dia"]["input_file"]) == root / "dia_opt.d12"
    assert Path(opt["dia"]["output_file"]) == root / "dia_opt.out"
    # every other deck runs in its step folder, with a job script written there
    step1 = root / "workflow_outputs" / wid / "step_001_OPT"
    for mid in ("fresh", "guess", "stopped"):
        row = opt[mid]
        deck = Path(row["input_file"])
        assert deck == step1 / mid / f"{mid}_opt.d12" and Path(row["work_dir"]) == step1 / mid
        assert deck.read_bytes() == (root / f"{mid}_opt.d12").read_bytes()
        script = deck.with_suffix(".sh").read_text()
        assert "--ntasks=8" not in script and "edited by hand" not in script
        assert f'MACE_WORKFLOW_ID="{wid}"' in script or wid in script
        assert row["slurm_job_id"] and row["status"] == "submitted"
    # a GUESSP deck takes its own matrix along; the others do not need one
    assert (step1 / "guess" / "guess_opt.f9").read_bytes() == F9
    assert not list((step1 / "fresh").glob("*.f9"))
    # the finished OPT's next step went out with the three first steps
    (sp,) = _by(db, "SP").values()
    assert sp["material_id"] == "dia" and sp["slurm_job_id"]
    assert Path(sp["input_file"]).parent.parent == root / "workflow_outputs" / wid / "step_002_SP"
    assert len(fake.calls("sbatch")) == 4
    for c in fake.calls("sbatch"):
        assert str(root / "workflow_outputs" / wid) in json.dumps(c)

    st = json.loads(_ok(_mace(root, env, "workflow", "--status", wid, "--json")).stdout)
    states = {m["material_id"]: [n["state"] for n in m["nodes"]] for m in st["materials"]}
    assert states["dia"][0] == "completed" and states["dia"][1] != "planned", states
    assert all(states[m][1] == "planned" for m in ("fresh", "guess", "stopped")), states


def test_a_first_step_in_its_step_folder_progresses(tmp_path):
    """The completion callback of a first step run in its step folder
    starts the next step, as for every later step."""
    fake, env, root = _folder(tmp_path)
    w, p = _launch(root, env, files=["fresh_opt.d12"])
    _ok(p)
    wid = w["workflow_id"]
    db = root / f".mace_context_{wid}" / "materials.db"
    (row,) = rows(db).values()
    work = Path(row["input_file"]).parent
    assert work == root / "workflow_outputs" / wid / "step_001_OPT" / "fresh"
    shutil.copy(DATA / "dia_opt.out", work / f"{Path(row['input_file']).stem}.out")
    exports = dict(re.findall(r'^export (MACE_WORKFLOW_ID|MACE_CONTEXT_DIR|MACE_ISOLATION_MODE)'
                              r'="([^"]*)"', (work / "fresh_opt.sh").read_text(), re.M))
    assert exports.get("MACE_CONTEXT_DIR") == str(root / f".mace_context_{wid}")
    fake.set_record(row["slurm_job_id"], state="COMPLETED")
    cb = _ok(callback(env, work, db, mode="completion", extra_env=dict(exports, SLURM_JOB_ID=row["slurm_job_id"])))
    assert rows(db)[row["calc_id"]]["status"] == "completed", cb.stdout[-3000:]
    (sp,) = _by(db, "SP").values()
    assert Path(sp["input_file"]).parent.parent == root / "workflow_outputs" / wid / "step_002_SP"
    assert len(fake.calls("sbatch")) == 2
    assert sorted(p.name for p in root.iterdir() if p.is_file()) == sorted(
        [p.name for p in DATA.iterdir()] + ["dia_opt.f9", "guess_opt.f9", "spec.json"])


def test_a_plan_launched_in_place_by_an_older_mace_still_progresses(tmp_path):
    """Before 1.1.6 the first step ran in the decks' folder: such a row,
    finished, still starts its next step in the workflow's step folders."""
    fake, env, root = _folder(tmp_path)
    (root / "spec.json").write_text(json.dumps({
        "version": 1, "template": "opt_sp", "inputs": {"type": "d12", "files": ["fresh_opt.d12"]}}))
    w = json.loads(_ok(_mace(root, env, "workflow", "--spec", "spec.json", "--write-plan", "--json")).stdout)
    wid = w["workflow_id"]
    ctx = root / f".mace_context_{wid}"
    ctx.mkdir(exist_ok=True)
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from mace.database.materials import MaterialDatabase
    db = MaterialDatabase(str(ctx / "materials.db"))
    db.create_material(material_id="fresh", formula="C2", source_type="d12",
                       source_file=str(root / "fresh_opt.d12"))
    # as an older `mace submit --progress-plan` recorded it: the deck where it is
    calc_id = db.create_calculation(material_id="fresh", calc_type="OPT",
                                    input_file=str(root / "fresh_opt.d12"), work_dir=str(root))
    db.update_calculation_settings(calc_id, {"workflow_id": wid, "workflow_step": 1,
                                             "workflow_calc_type": "OPT"}, merge=True)
    shutil.copy(DATA / "dia_opt.out", root / "fresh_opt.out")
    with sqlite3.connect(ctx / "materials.db") as con:
        con.execute("UPDATE calculations SET status='completed', output_file=? WHERE calc_id=?",
                    (str(root / "fresh_opt.out"), calc_id))
    r = json.loads(_ok(_mace(root, env, "workflow", "--process", wid, "--json")).stdout)
    assert r["steps_started"] == 1
    (sp,) = _by(ctx / "materials.db", "SP").values()
    assert Path(sp["input_file"]).parent.parent == root / "workflow_outputs" / wid / "step_002_SP"
    assert sp["slurm_job_id"] and len(fake.calls("sbatch")) == 1


def test_two_decks_of_one_material_are_refused_before_anything_runs(tmp_path):
    fake, env, root = _folder(tmp_path)
    shutil.copy(root / "fresh_opt.d12", root / "fresh_sp.d12")
    w, p = _launch(root, env, files=["fresh_opt.d12", "fresh_sp.d12"])
    assert p.returncode != 0
    assert "same material name fresh" in p.stdout + p.stderr, (p.stdout + p.stderr)[-2000:]
    assert not fake.calls("sbatch")
    assert not list((root / "workflow_outputs").rglob("*.d12"))


def _progress(root, env, *args):
    """`mace submit --progress` (no plan file): its workflow id and run."""
    p = _mace(root, env, "submit", "--progress", *args)
    m = re.search(r"Workflow ID:\s*(workflow_\S+)", p.stdout + p.stderr)
    return (m.group(1) if m else None), p


def _top(root):
    return sorted(p.name for p in root.iterdir())


def test_manual_progress_starts_in_step_folders_without_touching_the_decks(tmp_path):
    """`mace submit --progress TEMPLATE` (a manual submission that
    progresses) lays its first step out as --progress-plan does: the edited
    job scripts and finished outputs beside the decks no longer stop it."""
    fake, env, root = _folder(tmp_path)
    before, names = _sums(root), _top(root)
    wid, p = _progress(root, env, "opt_sp", ".")
    _ok(p)
    log = p.stdout + p.stderr
    assert wid and "would not reach" not in log and "--overwrite-sh" not in log, log[-3000:]
    # the user's files: byte for byte as they were, and no file (or .mace_arrays) beside them
    assert _sums(root) == before
    new = set(_top(root)) - set(names)
    assert f".mace_context_{wid}" in new and "workflow_outputs" in new and ".mace_arrays" not in new, new
    assert all(n.startswith(("workflow_", ".mace_context_")) for n in new), new

    db = root / f".mace_context_{wid}" / "materials.db"
    opt = _by(db, "OPT")
    assert sorted(opt) == ["dia", "fresh", "guess", "stopped"]
    assert opt["dia"]["status"] == "completed" and not opt["dia"]["slurm_job_id"]
    assert Path(opt["dia"]["output_file"]) == root / "dia_opt.out"
    step1 = root / "workflow_outputs" / wid / "step_001_OPT"
    for mid in ("fresh", "guess", "stopped"):
        deck = Path(opt[mid]["input_file"])
        assert deck == step1 / mid / f"{mid}_opt.d12" and Path(opt[mid]["work_dir"]) == step1 / mid
        assert deck.read_bytes() == (root / f"{mid}_opt.d12").read_bytes()
        script = deck.with_suffix(".sh").read_text()
        assert "edited by hand" not in script and wid in script
        assert opt[mid]["slurm_job_id"] and opt[mid]["status"] == "submitted"
    assert (step1 / "guess" / "guess_opt.f9").read_bytes() == F9
    # the finished OPT is not run again; its SP went out at launch
    (sp,) = _by(db, "SP").values()
    assert sp["material_id"] == "dia" and sp["slurm_job_id"]
    assert Path(sp["input_file"]).parent.parent == root / "workflow_outputs" / wid / "step_002_SP"
    assert len(fake.calls("sbatch")) == 4
    for c in fake.calls("sbatch"):
        assert str(root / "workflow_outputs" / wid) in json.dumps(c)


def test_manual_progress_first_step_progresses_from_its_step_folder(tmp_path):
    fake, env, root = _folder(tmp_path)
    wid, p = _progress(root, env, "opt_sp", "fresh_opt.d12")
    _ok(p)
    db = root / f".mace_context_{wid}" / "materials.db"
    (row,) = rows(db).values()
    work = Path(row["input_file"]).parent
    assert work == root / "workflow_outputs" / wid / "step_001_OPT" / "fresh"
    shutil.copy(DATA / "dia_opt.out", work / "fresh_opt.out")
    exports = dict(re.findall(r'^export (MACE_WORKFLOW_ID|MACE_CONTEXT_DIR|MACE_ISOLATION_MODE)'
                              r'="([^"]*)"', (work / "fresh_opt.sh").read_text(), re.M))
    assert exports.get("MACE_WORKFLOW_ID") == wid
    fake.set_record(row["slurm_job_id"], state="COMPLETED")
    cb = _ok(callback(env, work, db, mode="completion", extra_env=dict(exports, SLURM_JOB_ID=row["slurm_job_id"])))
    assert rows(db)[row["calc_id"]]["status"] == "completed", cb.stdout[-3000:]
    (sp,) = _by(db, "SP").values()
    assert Path(sp["input_file"]).parent.parent == root / "workflow_outputs" / wid / "step_002_SP"
    assert sp["slurm_job_id"] and len(fake.calls("sbatch")) == 2
    assert (root / "fresh_opt.sh").read_bytes() == (DATA / "fresh_opt.sh").read_bytes()
    assert not (root / "fresh_opt.out").exists()


def test_a_manual_progression_running_in_place_still_progresses(tmp_path):
    """Before 1.1.6 `mace submit --progress` ran the decks in their folder:
    the completion callback of such a job, run there, still finds the row
    and starts the next step in the workflow's step folders. (Run in place,
    the callback also takes in any other finished output of the folder: a
    folder of one deck here.)"""
    fake, env, root = _folder(tmp_path)
    root = root / "solo"
    root.mkdir()
    shutil.copy(DATA / "fresh_opt.d12", root)
    wid, p = _progress(root, env, "opt_sp", "fresh_opt.d12")
    _ok(p)
    db = root / f".mace_context_{wid}" / "materials.db"
    (row,) = rows(db).values()
    staged = Path(row["input_file"])
    # as an older MACE recorded and submitted it: the deck where it is, its job script beside it
    script = staged.with_suffix(".sh").read_text().replace(str(staged.parent), str(root))
    shutil.rmtree(root / "workflow_outputs" / wid)
    (root / "fresh_opt.sh").write_text(script)
    with sqlite3.connect(db) as con:
        con.execute("UPDATE calculations SET input_file=?, work_dir=?, job_script=? WHERE calc_id=?",
                    (str(root / "fresh_opt.d12"), str(root), str(root / "fresh_opt.sh"), row["calc_id"]))
    shutil.copy(DATA / "dia_opt.out", root / "fresh_opt.out")
    exports = dict(re.findall(r'^export (MACE_WORKFLOW_ID|MACE_CONTEXT_DIR|MACE_ISOLATION_MODE)'
                              r'="([^"]*)"', script, re.M))
    assert exports.get("MACE_WORKFLOW_ID") == wid
    fake.set_record(row["slurm_job_id"], state="COMPLETED")
    cb = _ok(callback(env, root, db, mode="completion", extra_env=dict(exports, SLURM_JOB_ID=row["slurm_job_id"])))
    done = rows(db)[row["calc_id"]]
    assert done["status"] == "completed", cb.stdout[-3000:]
    assert Path(done["output_file"]) == root / "fresh_opt.out"
    (sp,) = _by(db, "SP").values()
    assert Path(sp["input_file"]).parent.parent == root / "workflow_outputs" / wid / "step_002_SP"
    assert sp["slurm_job_id"] and len(fake.calls("sbatch")) == 2


def test_manual_progress_refuses_two_decks_of_one_material_before_writing_a_plan(tmp_path):
    fake, env, root = _folder(tmp_path)
    (root / "two").mkdir()
    shutil.copy(root / "fresh_opt.d12", root / "two" / "fresh_opt.d12")
    shutil.copy(root / "fresh_opt.d12", root / "two" / "fresh_sp.d12")
    root = root / "two"
    _wid, p = _progress(root, env, "opt_sp", ".")
    assert p.returncode != 0
    assert "same material name fresh" in p.stdout + p.stderr, (p.stdout + p.stderr)[-2000:]
    assert not fake.calls("sbatch")
    assert not list(root.glob("workflow_configs/workflow_plan_*.json"))


@pytest.mark.parametrize("how", ["progress", "progress-plan"])
def test_first_step_job_arrays_live_in_the_step_folder(tmp_path, how):
    """The first step's job arrays are written in its step folder, as every
    later step's: the decks' folder gets no .mace_arrays."""
    fake, env, root = _folder(tmp_path)
    (root / "dia_opt.out").unlink()          # three decks to run, as one array
    if how == "progress":
        wid, p = _progress(root, env, "opt_sp", "--array", ".")
    else:
        (root / "spec.json").write_text(json.dumps({
            "version": 1, "template": "opt_sp", "inputs": {"type": "d12", "directory": "."}}))
        w = json.loads(_ok(_mace(root, env, "workflow", "--spec", "spec.json", "--write-plan", "--json")).stdout)
        wid = w["workflow_id"]
        p = _mace(root, env, *w["launch"][1:], "--array")
    _ok(p)
    assert not (root / ".mace_arrays").exists()
    step1 = root / "workflow_outputs" / wid / "step_001_OPT"
    (sub,) = step1.glob(".mace_arrays/*/submitted.json")
    data = json.loads(sub.read_text())
    assert sorted(m["job"] for m in data["members"]) == ["dia_opt", "fresh_opt", "guess_opt", "stopped_opt"]
    db = root / f".mace_context_{wid}" / "materials.db"
    assert all("_" in r["slurm_job_id"] for r in rows(db).values()), rows(db)


def test_plain_submit_keeps_its_script_protection(tmp_path):
    """`mace submit` without --progress runs the decks in place: it still
    refuses to replace an edited job script."""
    fake, env, root = _folder(tmp_path)
    for args in (["--track", "--walltime", "1:00:00"], ["--walltime", "1:00:00"]):
        p = _mace(root, env, "submit", *args, "fresh_opt.d12")
        assert p.returncode != 0 and "would not reach" in p.stdout + p.stderr, (args, (p.stdout + p.stderr)[-2000:])
        assert not fake.calls("sbatch")
        assert (root / "fresh_opt.sh").read_bytes() == (DATA / "fresh_opt.sh").read_bytes()


def test_a_properties_deck_takes_its_wavefunction_to_its_step_folder(tmp_path):
    """A .d3 runs from its SCF run's wavefunction ($JOB.f9, submit_prop.sh):
    staged for --progress, the .f9 goes with it, renamed like the deck."""
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from mace.workflow.manual_progress import stage_deck
    deck = tmp_path / "dia_band.d3"
    deck.write_text("NEWK\n12 12\n1 0\nBAND\nDIA\n4 12 100 1 8 1 0\nEND\n")
    (tmp_path / "dia_band.f9").write_bytes(F9)
    staged = stage_deck(deck, tmp_path / "step_003_BAND" / "dia", "dia_band")
    assert staged.read_text() == deck.read_text()
    assert staged.with_suffix(".f9").read_bytes() == F9
    assert (tmp_path / "dia_band.f9").read_bytes() == F9


def test_a_first_step_waiting_over_a_day_keeps_its_step_folder(tmp_path):
    """The engine's clean-up of failed workflow folders removed any
    workflow_outputs/workflow_* folder over a day old with no output in it:
    a first step staged there and still pending (or held) lost its deck and
    job script. Only a folder with no file at all, of a workflow with no
    step waiting, is removed now."""
    import os
    import time
    fake, env, root = _folder(tmp_path)
    wid, p = _progress(root, env, "opt_sp", "fresh_opt.d12")
    _ok(p)
    wdir = root / "workflow_outputs" / wid
    deck = wdir / "step_001_OPT" / "fresh" / "fresh_opt.d12"
    assert deck.is_file() and deck.with_suffix(".sh").is_file()
    # an abandoned folder of a generation that failed: empty folders only
    empty = root / "workflow_outputs" / "workflow_20200101_000000_0001"
    (empty / "step_001_OPT" / "x").mkdir(parents=True)
    old = time.time() - 25 * 3600
    for d in (wdir, empty):
        os.utime(d, (old, old))
    for args in (["workflow", "--status", wid], ["workflow", "--process", wid]):
        r = _ok(_mace(root, env, *args))
        assert deck.is_file() and deck.with_suffix(".sh").is_file(), (r.stdout + r.stderr)[-2000:]
    assert not empty.exists()
    # held, with its files gone: still kept while its database says it waits
    (row,) = rows(root / f".mace_context_{wid}" / "materials.db").values()
    with sqlite3.connect(root / f".mace_context_{wid}" / "materials.db") as con:
        con.execute("UPDATE calculations SET status='held' WHERE calc_id=?", (row["calc_id"],))
    for f in deck.parent.iterdir():
        f.unlink()
    os.utime(wdir, (old, old))
    _ok(_mace(root, env, "workflow", "--process", wid))
    assert deck.parent.is_dir()


def _one_opt(root, wid):
    (row,) = rows(root / f".mace_context_{wid}" / "materials.db").values()
    return row


@pytest.mark.parametrize("how", ["edited", "symlink", "rerun"])
def test_a_finished_output_that_is_not_the_decks_run_runs_the_deck_again(tmp_path, how):
    """A finished .out counts only as the deck's own run: a deck changed
    after its output was written, or whose .out is a symlink, runs again in
    its step folder, and --rerun-finished runs every deck."""
    import os
    fake, env, root = _folder(tmp_path)
    args = []
    if how == "edited":
        later = (root / "dia_opt.out").stat().st_mtime + 3600
        os.utime(root / "dia_opt.d12", (later, later))
    elif how == "symlink":
        (tmp_path / "elsewhere").mkdir()
        (root / "dia_opt.out").rename(tmp_path / "elsewhere" / "dia_opt.out")
        (root / "dia_opt.out").symlink_to(tmp_path / "elsewhere" / "dia_opt.out")
    else:
        args = ["--rerun-finished"]
    wid, p = _progress(root, env, "opt_sp", *args, "dia_opt.d12")
    _ok(p)
    log = p.stdout + p.stderr
    assert "Finished already" not in log, log[-2000:]
    assert ("changed after dia_opt.out was written" in log) == (how == "edited"), log[-2000:]
    row = _one_opt(root, wid)
    assert row["status"] == "submitted" and row["slurm_job_id"]
    assert Path(row["input_file"]) == root / "workflow_outputs" / wid / "step_001_OPT" / "dia" / "dia_opt.d12"
    assert len(fake.calls("sbatch")) == 1


def test_a_finished_output_older_than_a_copy_of_its_folder_still_counts(tmp_path):
    """A folder copied without its file times can put the deck a moment
    after its output: that output still counts as the deck's run."""
    import os
    fake, env, root = _folder(tmp_path)
    t = (root / "dia_opt.out").stat().st_mtime + 30
    os.utime(root / "dia_opt.d12", (t, t))
    wid, p = _progress(root, env, "opt_sp", "dia_opt.d12")
    _ok(p)
    assert "Finished already, not run again: dia_opt.d12" in p.stdout + p.stderr
    assert _by(root / f".mace_context_{wid}" / "materials.db", "OPT")["dia"]["status"] == "completed"


def test_rerun_finished_needs_a_progression(tmp_path):
    fake, env, root = _folder(tmp_path)
    p = _mace(root, env, "submit", "--rerun-finished", "--track", "dia_opt.d12")
    assert p.returncode != 0 and "--rerun-finished applies to --progress" in p.stdout + p.stderr
    assert not fake.calls("sbatch")


def _external(root):
    """fresh_opt.d12 alone, its geometry record EXTERNAL (CRYSTAL reads fort.34)."""
    for n in ("dia_opt", "guess_opt", "stopped_opt"):
        for s in (".d12", ".out", ".f9", ".sh"):
            (root / (n + s)).unlink(missing_ok=True)
    d = root / "fresh_opt.d12"
    lines = d.read_text().splitlines()
    d.write_text("\n".join([lines[0], "EXTERNAL"] + lines[6:]) + "\n")
    return d


def test_an_external_deck_takes_the_geometry_its_job_script_stages(tmp_path):
    """An EXTERNAL deck whose edited job script copies $JOB.gui to fort.34:
    staged into its step folder, that file goes with it as <stem>.ext (what
    the step's job script copies to fort.34)."""
    fake, env, root = _folder(tmp_path)
    _external(root)
    (root / "fresh_opt.gui").write_text("3 1 1\n")
    sh = root / "fresh_opt.sh"
    sh.write_text(sh.read_text().replace("cp $DIR/$JOB.d12 $SCRATCH/$JOB/INPUT",
                  "cp $DIR/$JOB.d12 $SCRATCH/$JOB/INPUT && cp $DIR/$JOB.gui $SCRATCH/$JOB/fort.34"))
    assert "fort.34" in sh.read_text()
    wid, p = _progress(root, env, "opt_sp", ".")
    _ok(p)
    staged = Path(_one_opt(root, wid)["input_file"])
    assert staged.with_suffix(".ext").read_text() == "3 1 1\n"
    assert "$JOB.ext" in staged.with_suffix(".sh").read_text()
    assert len(fake.calls("sbatch")) == 1


def test_an_external_deck_without_its_geometry_is_refused_before_anything_runs(tmp_path):
    fake, env, root = _folder(tmp_path)
    _external(root)
    wid, p = _progress(root, env, "opt_sp", ".")
    assert p.returncode != 0
    log = p.stdout + p.stderr
    assert "reads its geometry from an external file" in log and "fresh_opt.ext" in log, log[-2000:]
    assert not fake.calls("sbatch")
    assert not rows(root / f".mace_context_{wid}" / "materials.db")


def test_an_optgeom_restart_deck_staged_into_its_step_folder_is_said_to_start_over(tmp_path):
    """The job script continues an OPTGEOM RESTART from the OPTINFO.DAT of
    the scratch directory the deck's own folder recorded; from its step
    folder the deck runs in a new one and starts over. The launch says so."""
    fake, env, root = _folder(tmp_path)
    d = root / "fresh_opt.d12"
    d.write_text(d.read_text().replace("FULLOPTG\n", "FULLOPTG\nRESTART\n", 1))
    assert "RESTART" in d.read_text()
    for n in ("dia_opt", "stopped_opt"):
        for suffix in (".d12", ".out", ".f9", ".sh"):
            (root / (n + suffix)).unlink(missing_ok=True)
    wid, p = _progress(root, env, "opt_sp", ".")
    _ok(p)
    log = " ".join((p.stdout + p.stderr).split())
    assert "fresh_opt.d12 continues an optimization (OPTGEOM RESTART)" in log, log[-2000:]
    assert "starts the optimization over" in log and "guess_opt.d12 continues" not in log
    assert len(fake.calls("sbatch")) == 2


def test_a_busy_queue_at_launch_still_submits_the_decks_to_run(tmp_path):
    """Starting the next steps of the finished decks waits for the queue
    lock; when another MACE process holds it, the decks to run still go
    out (they were dropped: recorded nowhere, and the launch refused again),
    and `mace workflow --process` starts the finished decks' next steps."""
    import subprocess
    import sys
    fake, env, root = _folder(tmp_path)
    (root / "spec.json").write_text(json.dumps({
        "version": 1, "template": "opt_sp",
        "inputs": {"type": "d12", "files": ["dia_opt.d12", "fresh_opt.d12"]}}))
    w = json.loads(_ok(_mace(root, env, "workflow", "--spec", "spec.json", "--write-plan", "--json")).stdout)
    wid = w["workflow_id"]
    ctx = root / f".mace_context_{wid}"
    (ctx / ".queue_locks").mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(REPO_ROOT))
    from mace.queue.queue_lock_manager import QueueLockManager, QUEUE_STATE_LOCK
    lm = QueueLockManager(lock_dir=ctx / ".queue_locks", lock_timeout=300)
    assert lm.acquire_lock(QUEUE_STATE_LOCK, timeout=5)
    try:
        # mace_cli as `mace` runs it, with the queue-lock wait cut from 120 s
        code = ("import runpy, sys; sys.path.insert(0, sys.argv[1]); "
                "import mace.workflow.commands as c; c.LOCK_WAIT_S = 2; "
                "sys.argv = [sys.argv[1] + '/mace_cli', '--no-banner'] + sys.argv[2:]; "
                "runpy.run_path(sys.argv[0], run_name='__main__')")
        p = subprocess.run([sys.executable, "-c", code, str(REPO_ROOT), *w["launch"][1:]], cwd=str(root),
                           env=env, capture_output=True, text=True, timeout=600)
    finally:
        lm.release_lock(QUEUE_STATE_LOCK)
    _ok(p)
    assert "were not started" in p.stdout + p.stderr and f"--process {wid}" in p.stdout + p.stderr
    db = ctx / "materials.db"
    opt = _by(db, "OPT")
    assert opt["fresh"]["status"] == "submitted" and opt["fresh"]["slurm_job_id"]
    assert opt["dia"]["status"] == "completed" and not _by(db, "SP")
    _ok(_mace(root, env, "workflow", "--process", wid))
    (sp,) = _by(db, "SP").values()
    assert sp["material_id"] == "dia" and sp["slurm_job_id"]
    assert len(fake.calls("sbatch")) == 2
