"""The plan's job_arrays setting: the first wave of a workflow as SLURM job
arrays, through `mace workflow` and `mace submit --progress` alike (the
workflow route must equal the manual one), with the steps after it chained
per material by each task's own callback."""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from array_helpers import MANAGER, OPT_D12, callback, rows, task_env
from conftest import REPO_ROOT, find_data
from fake_slurm import FakeSlurm
from test_array_default_path_golden import _env, _mace


def _setup(tmp_path, names, deck=OPT_D12, state=None):
    fake = FakeSlurm(tmp_path, state)
    env = _env(tmp_path, fake)
    d = tmp_path / "decks"
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        (d / f"{n}.d12").write_text(deck)
    return fake, env, d


def _ok(p):
    assert p.returncode == 0, p.stdout[-5000:] + p.stderr[-3000:]
    return p


def _plan(root):
    (plan,) = (root / "workflow_configs").glob("workflow_plan_*.json")
    return json.loads(plan.read_text())


def _context_db(root):
    (db,) = root.glob(".mace_context_*/materials.db")
    return db


def test_quick_start_first_wave_is_an_array(tmp_path):
    fake, env, d = _setup(tmp_path, ["m0", "m1", "m2"])
    p = _ok(_mace(tmp_path, env, "workflow", "--quick-start", "--d12-dir", "decks",
                  "--workflow", "opt_sp", "--array", "--max-running", "2"))
    plan = _plan(tmp_path)
    assert plan["job_arrays"] == {"first_step": True, "max_running": 2}
    (w,) = tmp_path.glob("workflow_outputs/*/step_001_OPT/.mace_arrays/arr_*/arr_*.sh")
    text = w.read_text()
    assert "#SBATCH --array=0-2%2" in text and "#SBATCH -J mace_arr_opt_0" in text
    assert [c["argv"][:2] for c in fake.calls("sbatch")] == [["sbatch", "--hold"]]
    by_job = {x["slurm_job_id"]: x for x in rows(_context_db(tmp_path)).values()}
    assert sorted(by_job) == ["4242_0", "4242_1", "4242_2"]
    for x in by_job.values():
        s = json.loads(x["settings_json"])
        assert s["workflow_step"] == 1 and s["slurm_array"]["array_job_id"] == "4242"
        assert x["job_script"].endswith(f"{x['material_id']}.sh")
        assert "export MACE_WORKFLOW_ID=" in Path(x["job_script"]).read_text()
        assert x["output_file"].endswith(f"{x['material_id']}.out")
    # the per-material job scripts are what the default route writes
    fake2, env2, d2 = _setup(tmp_path / "plain", ["m0", "m1", "m2"])
    _ok(_mace(tmp_path / "plain", env2, "workflow", "--quick-start", "--d12-dir", "decks",
              "--workflow", "opt_sp"))
    for name in ("m0", "m1", "m2"):
        (a,) = tmp_path.glob(f"workflow_outputs/*/step_001_OPT/{name}/{name}.sh")
        (b,) = (tmp_path / "plain").glob(f"workflow_outputs/*/step_001_OPT/{name}/{name}.sh")
        norm = lambda t, root: re.sub(r"(cksum of the lines above: )\d+ \d+", r"\1N", re.sub(
            r"workflow_\d{8}_\d{6}(?:_[0-9a-f]{4})?", "WF", t.replace(str(root), "<R>")))
        assert norm(a.read_text(), tmp_path) == norm(b.read_text(), tmp_path / "plain")


def test_execute_flag_overrides_and_persists(tmp_path):
    fake, env, d = _setup(tmp_path, ["m0", "m1"])
    _ok(_mace(tmp_path, env, "workflow", "--quick-start", "--d12-dir", "decks",
              "--workflow", "basic_opt"))
    plan_file = next((tmp_path / "workflow_configs").glob("workflow_plan_*.json"))
    plan = json.loads(plan_file.read_text())
    assert "job_arrays" not in plan
    other = tmp_path / "again"
    other.mkdir()
    plan.pop("workflow_id", None)
    (other / "plan.json").write_text(json.dumps(plan))
    shutil.copytree(d, other / "decks")
    before = len(fake.calls("sbatch"))
    _ok(_mace(other, env, "workflow", "--execute", "plan.json", "--array"))
    assert [c["argv"][:2] for c in fake.calls("sbatch")[before:]] == [["sbatch", "--hold"]]
    assert _plan(other)["job_arrays"] == {"first_step": True, "max_running": None}


@pytest.mark.parametrize("args,msg", [(["--max-running", "2"], "use it with --array"),
                                      (["--status", "--array"], "--array goes with")])
def test_workflow_flag_errors(tmp_path, args, msg):
    fake, env, d = _setup(tmp_path, ["m0"])
    p = _mace(tmp_path, env, "workflow", *(args if "--status" in args else
                                          ["--quick-start", "--d12-dir", "decks", *args]))
    assert p.returncode != 0 and msg in p.stdout + p.stderr


def _wrapper_body(root):
    (w,) = list(root.rglob(".mace_arrays/arr_*/arr_*.sh"))
    text = w.read_text().replace(str(w.parent.resolve()), "<ARRAY_DIR>")
    return re.sub(r"arr_\d{8}_\d{6}_\d+", "<TAG>", text)


def test_workflow_route_equals_the_manual_route(tmp_path):
    fa, ea, da = _setup(tmp_path / "wf", ["m0", "m1"])
    _ok(_mace(tmp_path / "wf", ea, "workflow", "--quick-start", "--d12-dir", "decks",
              "--workflow", "opt_sp", "--array", "--max-running", "2"))
    fb, eb, db_ = _setup(tmp_path / "man", ["m0", "m1"])
    _ok(_mace(tmp_path / "man", eb, "submit", "--progress", "opt_sp", "--array",
              "--max-running", "2", "decks"))
    plan_a, plan_b = _plan(tmp_path / "wf"), _plan(db_)
    assert plan_a["job_arrays"] == plan_b["job_arrays"] == {"first_step": True,
                                                            "max_running": 2}
    assert plan_a["step_configurations"].keys() == plan_b["step_configurations"].keys()
    # The two routes' job scripts spell some directives differently (the
    # executor rebuilds its templates: "-N 1" there is "--nodes=1", the
    # "# or general" comment is gone), and the wrapper copies the request
    # verbatim: the request is the same, and so is everything else.
    from mace.submission.job_array import resource_key
    wa, wb = _wrapper_body(tmp_path / "wf"), _wrapper_body(tmp_path / "man")
    assert resource_key(wa) == resource_key(wb)
    strip = lambda t: [l for l in t.splitlines() if not l.startswith("#SBATCH")]
    assert strip(wa) == strip(wb)
    ids_a = sorted(x["slurm_job_id"] for x in rows(_context_db(tmp_path / "wf")).values())
    ids_b = sorted(x["slurm_job_id"] for x in rows(_context_db(db_)).values())
    assert ids_a == ids_b == ["4242_0", "4242_1"]


def test_each_task_chains_its_own_next_step(tmp_path):
    """Task 0's callback, in the workflow's context: its OPT row completes
    from its own output and its SP goes in as an ordinary job; task 1, still
    running, is left alone. Real CRYSTAL output (corpus)."""
    out = find_data("OPT/1_dia_opt_rev1.out")
    deck = find_data("OPT/1_dia_opt_rev1.d12")
    fake, env, d = _setup(tmp_path, [])
    for n in ("dia0", "dia1"):
        shutil.copy(deck, d / f"{n}.d12")
    _ok(_mace(tmp_path, env, "submit", "--progress", "opt_sp", "--array", "decks"))
    db = _context_db(d)
    by_job = {x["slurm_job_id"]: x for x in rows(db).values()}
    row0 = by_job["4242_0"]
    stem, work = Path(row0["input_file"]).stem, Path(row0["input_file"]).parent
    # the first step runs in its step folder, its job array beside it
    assert work.parent.name == "step_001_OPT" and (work.parent / ".mace_arrays").is_dir()
    assert not (d / ".mace_arrays").exists()
    shutil.copy(out, work / f"{stem}.out")
    script = (work / f"{stem}.sh").read_text()
    exports = dict(re.findall(r'^export (MACE_WORKFLOW_ID|MACE_CONTEXT_DIR|MACE_ISOLATION_MODE)'
                              r'="([^"]*)"', script, re.M))
    for t in (0, 1):
        fake.set_record(f"4242_{t}", state="RUNNING")
    before = len(fake.calls("sbatch"))
    p = _ok(callback(env, work, db, mode="completion",
                     extra_env=dict(task_env("4242", 0, 9100), **exports)))
    r = {x["calc_id"]: x for x in rows(db).values()}
    assert r[row0["calc_id"]]["status"] == "completed", p.stdout[-3000:]
    new = [x for x in r.values() if x["calc_type"].startswith("SP")]
    assert len(new) == 1 and new[0]["slurm_job_id"] == "5000", p.stdout[-3000:]
    sb = fake.calls("sbatch")[before:]
    assert [c["argv"][1] for c in sb] != ["--hold"] and len(sb) == 1
    assert "--array" not in sb[0]["script_text"]
    assert not fake.calls("scancel")
    assert by_job["4242_1"]["calc_id"] in r and r[by_job["4242_1"]["calc_id"]]["status"] in (
        "submitted", "running")


READY_MADE = """#!/bin/bash --login
#SBATCH -J job
#SBATCH -o job-%J.o
#SBATCH --ntasks=4
#SBATCH -t 1:00:00
touch "{marker}"
echo ran-the-job-body
"""


@pytest.mark.parametrize("array", [False, True])
def test_ready_made_source_script_is_never_run_here(tmp_path, array):
    """A plan's source_script can be a finished SLURM script rather than a
    generator: the default route hands it to sbatch as it is, and so must
    the array route (it is the task's script itself, never run locally)."""
    fake, env, d = _setup(tmp_path, ["m0", "m1"])
    _ok(_mace(tmp_path, env, "workflow", "--quick-start", "--d12-dir", "decks",
              "--workflow", "basic_opt"))
    plan = _plan(tmp_path)
    plan.pop("workflow_id", None)
    other = tmp_path / "again"
    other.mkdir()
    shutil.copytree(d, other / "decks")
    marker = other / "JOB_BODY_RAN_HERE"
    ready = other / "ready.sh"
    ready.write_text(READY_MADE.format(marker=marker))
    scripts = plan["step_configurations"]["OPT_1"]["slurm_config"]["scripts"]
    (name,) = [k for k in scripts if k.startswith("submitcrystal23")]
    scripts[name]["source_script"] = str(ready)
    (other / "plan.json").write_text(json.dumps(plan))
    before = len(fake.calls("sbatch"))
    _ok(_mace(other, env, "workflow", "--execute", "plan.json", *(["--array"] if array else [])))
    assert not marker.exists()
    calls = fake.calls("sbatch")[before:]
    if not array:
        assert sorted(c["argv"] for c in calls) == [["sbatch", "m0.sh"], ["sbatch", "m1.sh"]]
        return
    (c,) = calls
    assert c["argv"][:2] == ["sbatch", "--hold"]
    (w,) = other.rglob(".mace_arrays/arr_*/arr_*.sh")
    by_job = {x["slurm_job_id"]: x for x in rows(_context_db(other)).values()}
    assert sorted(by_job) == ["4242_0", "4242_1"]
    from mace.submission.job_array import resource_key
    for x in by_job.values():
        task_script = Path(x["job_script"]).read_text()
        assert "ran-the-job-body" in task_script
        assert resource_key(w.read_text().replace("#SBATCH --array=0-1\n", "")) == \
            resource_key(task_script)
