"""`mace submit --array` through the real CLI, with the fake SLURM on PATH.

The arrays reuse the job scripts the default path writes, so those must be
byte-identical; what changes is only how they reach SLURM (one wrapper per
resource request, submitted held and released after the rows are written)
and the ids (A_T). Every per-job action afterwards - cancel, FINALRUN-loop
recovery, timeout restart, zombie removal - addresses the one task.
"""
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from array_helpers import MANAGER, OPT_D12, SP_D12, SP_DONE, callback, rows
from conftest import REPO_ROOT
from fake_slurm import FakeSlurm
from test_array_default_path_golden import _env, _mace, _norm, _snapshot, compare

TEMPLATE = REPO_ROOT / "mace" / "submission" / "submitcrystal23.sh"


def _setup(tmp_path, names, deck=SP_D12, state=None):
    fake = FakeSlurm(tmp_path, state)
    env = _env(tmp_path, fake)
    d = tmp_path / "decks"
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        (d / f"{n}.d12").write_text(deck)
    return fake, env, d


def _wrappers(d):
    return sorted(d.glob(".mace_arrays/arr_*/arr_*.sh"))


def _ok(p):
    assert p.returncode == 0, p.stdout[-5000:] + p.stderr[-3000:]
    return p


def test_untracked_array_submits_one_wrapper_held_then_releases(tmp_path):
    fake, env, d = _setup(tmp_path / "arr", ["m0", "m1", "m2", "m3"])
    p = _ok(_mace(tmp_path / "arr", env, "submit", "--array", "decks"))
    (w,) = _wrappers(d)
    calls = [c["argv"] for c in fake.calls() if c["tool"] in ("sbatch", "scontrol")]
    assert ["sbatch", "--hold", w.name] in calls
    assert calls[-1] == ["scontrol", "release", "4242"]
    assert not [c for c in fake.calls("sbatch") if c["argv"][1] != "--hold"]
    (sb,) = fake.calls("sbatch")
    assert sb["cwd"] == str(w.parent)
    for t in range(4):
        assert f"4242_{t}" in p.stdout
    assert "#SBATCH --array=0-3\n" in w.read_text()
    # the job scripts are the ones the default path writes
    fake2, env2, d2 = _setup(tmp_path / "plain", ["m0", "m1", "m2", "m3"])
    _ok(_mace(tmp_path / "plain", env2, "submit", "decks"))
    for n in ("m0", "m1", "m2", "m3"):
        assert _norm((d / f"{n}.sh").read_text(), tmp_path / "arr") == \
            _norm((d2 / f"{n}.sh").read_text(), tmp_path / "plain")
    data = json.loads((w.parent / "submitted.json").read_text())
    assert data["array_job_id"] == "4242" and data["released"]
    assert sorted(m["job"] for m in data["members"]) == ["m0", "m1", "m2", "m3"]


def test_max_running_throttles_each_array(tmp_path):
    fake, env, d = _setup(tmp_path, ["m0", "m1", "m2", "m3"])
    p = _ok(_mace(tmp_path, env, "submit", "--array", "--max-running", "2", "decks"))
    (w,) = _wrappers(d)
    assert "#SBATCH --array=0-3%2\n" in w.read_text()
    assert "At most 2 task(s) running at once" in p.stdout


@pytest.mark.parametrize("args,msg", [(["--max-running", "2"], "use it with --array"),
                                      (["--array", "--max-running", "0"], "1 or more")])
def test_max_running_needs_array(tmp_path, args, msg):
    fake, env, d = _setup(tmp_path, ["m0", "m1"])
    p = _mace(tmp_path, env, "submit", *args, "decks")
    assert p.returncode == 1 and msg in p.stdout + p.stderr
    assert not fake.calls("sbatch")


def _script_with(d, name, mem):
    subprocess.run(["bash", str(TEMPLATE), name], cwd=d, check=True, capture_output=True)
    sh = d / f"{name}.sh"
    sh.write_text(sh.read_text().replace("--mem-per-cpu=5G", f"--mem-per-cpu={mem}"))


def test_different_requests_go_to_different_arrays(tmp_path):
    fake, env, d = _setup(tmp_path, ["a", "b", "c", "d", "e"])
    _script_with(d, "c", "10G")
    _script_with(d, "d", "10G")
    _script_with(d, "e", "20G")
    p = _ok(_mace(tmp_path, env, "submit", "--array", "decks"))
    ws = _wrappers(d)
    assert len(ws) == 2
    mems = sorted(re.search(r"--mem-per-cpu=(\S+)", w.read_text()).group(1) for w in ws)
    assert mems == ["10G", "5G"]
    for w in ws:
        assert "#SBATCH --array=0-1\n" in w.read_text()
    plain = [c["argv"] for c in fake.calls("sbatch") if c["argv"][1] != "--hold"]
    assert plain == [["sbatch", "e.sh"]]
    assert "e: the only deck with this request" in p.stdout
    assert "--mem-per-cpu=10G" in p.stdout and "--mem-per-cpu=5G" in p.stdout


def _snap(tmp_path, args, names, scripts=()):
    fake, env, d = _setup(tmp_path, names)
    for n, mem in scripts:
        _script_with(d, n, mem)
    _ok(_mace(tmp_path, env, "submit", *args, "decks"))
    snap = _snapshot(tmp_path, fake)
    snap["files"] = {k: v for k, v in snap["files"].items() if ".mace_arrays" not in k}
    return snap


@pytest.mark.parametrize("names,scripts", [
    (["only"], ()),
    (["p", "q"], (("p", "7G"), ("q", "9G"))),
])
def test_groups_of_one_are_the_default_path(tmp_path, names, scripts):
    """A deck with a request of its own goes in exactly as without --array:
    same job script, same sbatch call, nothing else."""
    a = _snap(tmp_path / "a", ["--array"], names, scripts)
    b = _snap(tmp_path / "b", [], names, scripts)
    a["calls"] = [c for c in a["calls"] if c["argv"][:3] != ["scontrol", "show", "config"]]
    compare(json.loads(_norm(json.dumps(a), tmp_path / "a")),
            json.loads(_norm(json.dumps(b), tmp_path / "b")))
    assert not list((tmp_path / "a" / "decks").glob(".mace_arrays/*/submitted.json"))


def test_nosubmit_writes_the_array_and_submits_nothing(tmp_path):
    fake, env, d = _setup(tmp_path, ["m0", "m1"])
    p = _ok(_mace(tmp_path, env, "submit", "--array", "--nosubmit", "decks"))
    (w,) = _wrappers(d)
    assert f"cd {w.parent} && sbatch {w.name}" in p.stdout
    assert not fake.calls("sbatch") and not (tmp_path / "materials.db").exists()


def test_late_sbatch_line_is_not_part_of_the_request(tmp_path):
    fake, env, d = _setup(tmp_path, ["a", "b"])
    subprocess.run(["bash", str(TEMPLATE), "b"], cwd=d, check=True, capture_output=True)
    sh = d / "b.sh"
    sh.write_text(sh.read_text().replace("export JOB=b\n", "export JOB=b\n#SBATCH --mem=99G\n"))
    _ok(_mace(tmp_path, env, "submit", "--array", "decks"))
    (w,) = _wrappers(d)
    assert "99G" not in w.read_text() and "--array=0-1" in w.read_text()


def test_mixed_directory_gives_one_array_per_kind(tmp_path):
    fake, env, d = _setup(tmp_path, ["a", "b"])
    for n in ("c", "e"):
        (d / f"{n}.d3").write_text("BAND\nX\n2 8 50 1 8 1 0\n 0 0 0  4 0 4\n 4 0 4  4 2 4\nEND\n")
    _ok(_mace(tmp_path, env, "submit", "--array", "decks", stdin="3\n"))
    ws = _wrappers(d)
    assert len(ws) == 2
    assert {re.search(r"#SBATCH -J (\S+)", w.read_text()).group(1).rsplit("_", 1)[0]
            for w in ws} == {"mace_arr_d12", "mace_arr_d3"}
    assert [c["argv"][-1] for c in fake.calls("scontrol")
            if c["argv"][1] == "release"] == ["4242", "4243"]


# --- tracked ---------------------------------------------------------------

def test_tracked_rows_are_tasks(tmp_path):
    fake, env, d = _setup(tmp_path, ["m0", "m1", "m2"])
    p = _ok(_mace(tmp_path, env, "submit", "--array", "--track", "decks"))
    r = list(rows(tmp_path / "materials.db").values())
    by_job = {x["slurm_job_id"]: x for x in r}
    assert sorted(by_job) == ["4242_0", "4242_1", "4242_2"]
    for jid, x in by_job.items():
        stem = Path(x["input_file"]).stem
        assert x["job_script"] == str(d.resolve() / f"{stem}.sh")
        assert x["status"] == "submitted" and x["submitted_at"]
        sa = json.loads(x["settings_json"])["slurm_array"]
        assert sa["array_job_id"] == "4242" and jid == f"4242_{sa['task']}"
    status = json.loads((d / "crystal_job_status.json").read_text()
                        if (d / "crystal_job_status.json").exists()
                        else (tmp_path / "decks" / "crystal_job_status.json").read_text())
    assert sorted(status["submitted"]) == ["4242_0", "4242_1", "4242_2"]
    assert "Job ID(s): 4242_0, 4242_1, 4242_2" in p.stdout
    # the job scripts carry the database the callback must use
    for n in ("m0", "m1", "m2"):
        assert "export MACE_DB_PATH=" in (d / f"{n}.sh").read_text()


def test_tracked_job_id_line_names_only_this_runs_jobs(tmp_path):
    """The copy-paste "Job ID(s):" line of mace submit --track lists the jobs
    this run submitted, not every job crystal_job_status.json remembers from
    earlier submissions in the folder (seen on HPCC: an ended job's id next to
    the new one, and the GUI's toast then claimed two jobs)."""
    fake, env, d = _setup(tmp_path, ["first"])
    p = _ok(_mace(tmp_path, env, "submit", "--track", "decks/first.d12"))
    first = re.search(r"Job ID\(s\): (.+)", p.stdout).group(1).strip()
    (d / "second.d12").write_text(SP_D12)
    p = _ok(_mace(tmp_path, env, "submit", "--track", "decks/second.d12"))
    line = re.search(r"Job ID\(s\): (.+)", p.stdout).group(1).strip()
    assert first not in line.split(", ") and len(line.split(", ")) == 1, (first, line)


def test_progress_rows_carry_the_workflow_and_the_task(tmp_path):
    fake, env, d = _setup(tmp_path, ["m0", "m1"], deck=OPT_D12)
    _ok(_mace(tmp_path, env, "submit", "--progress", "opt_sp", "--array", "decks"))
    (plan,) = (d / "workflow_configs").glob("workflow_plan_*.json")
    wf = json.loads(plan.read_text())["workflow_id"]
    (db,) = d.glob(".mace_context_*/materials.db")
    assert len(rows(db)) == 2
    for x in rows(db).values():
        s = json.loads(x["settings_json"])
        assert s["workflow_id"] == wf and s["workflow_step"] == 1
        assert s["slurm_array"]["array_job_id"] == "4242"


def test_partial_wave_leaves_nothing_in_the_queue(tmp_path):
    fake, env, d = _setup(tmp_path, ["m0", "m1", "m2", "m3"],
                          state={"max_array_tasks": 2, "fail_sbatch_call": 2})
    p = _mace(tmp_path, env, "submit", "--array", "--track", "decks")
    assert p.returncode == 1
    assert "QOSMaxSubmitJobPerUserLimit" in p.stdout
    assert [c["argv"] for c in fake.calls("scancel")] == [["scancel", "4242"]]
    assert ["squeue", "-h", "-r", "-j", "4242", "-o", "%T,%r"] in [
        c["argv"] for c in fake.calls("squeue")]
    assert not fake.records()
    assert all(not x["slurm_job_id"] for x in rows(tmp_path / "materials.db").values())
    assert not [c for c in fake.calls("scontrol") if c["argv"][1] == "release"]


def test_wave_larger_than_the_free_submit_slots_is_refused_up_front(tmp_path):
    fake, env, d = _setup(tmp_path, ["m0", "m1", "m2"], state={
        "assoc_mgr": "Account=mendoza_q UserName=tester(1)\n MaxSubmitJobs=1000(999)"})
    p = _mace(tmp_path, env, "submit", "--array", "decks")
    assert p.returncode == 1 and "MaxSubmitJobs is 1000 and 999" in p.stdout
    assert not fake.calls("sbatch")


# --- the submit / record race ------------------------------------------------

def _hook(tmp_path, d, when):
    """Task 0 fails at once (a bad deck) and its callback runs right away:
    at release (held submission) or at submission (unheld)."""
    probe = tmp_path / "row_at_callback.json"
    code = (
        "import json,sqlite3,sys;c=sqlite3.connect(sys.argv[1]);"
        "r=c.execute('SELECT calc_id,slurm_job_id,status FROM calculations WHERE calc_id=?',"
        "(sys.argv[2],)).fetchone();json.dump(r,open(sys.argv[3],'w'))")
    script = (
        f"M=$(ls {d}/.mace_arrays/*/manifest.tsv | head -n 1); "
        "CALC=$(awk -F'\\t' '$1==0 {print $5}' \"$M\"); JOB=$(awk -F'\\t' '$1==0 {print $3}' \"$M\"); "
        f"printf ' ERROR **** INPUT **** bad keyword\\n' > {d}/$JOB.out; "
        f"{sys.executable} -c \"{code}\" {tmp_path}/materials.db \"$CALC\" {probe}; "
        "export MACE_ARRAY_MANIFEST=\"$M\" MACE_ARRAY_CALC_ID=\"$CALC\"; "
        f"cd {d} && {sys.executable} {MANAGER} --callback-mode completion "
        f"--db-path {tmp_path}/materials.db")
    return {"when": when, "cmd": ["bash", "-c", script],
            "env": {"SLURM_JOB_ID": "9100", "SLURM_ARRAY_JOB_ID": "{A}",
                    "SLURM_ARRAY_TASK_ID": "0", "SLURM_ARRAY_TASK_COUNT": "2",
                    "MACE_ARRAY_RAW_ID": "9100", "MACE_ARRAY_TASK": "{A}_0",
                    "PYTHONPATH": str(REPO_ROOT), "MACE_HOME": str(REPO_ROOT)}}, probe


@pytest.mark.parametrize("held", [True, False])
def test_a_task_that_ends_at_once_finds_its_row(tmp_path, held):
    fake, env, d = _setup(tmp_path, ["m0", "m1"])
    hook, probe = _hook(tmp_path, d.resolve(), "release" if held else "submit")
    fake.update(hooks=[hook], no_array_hold=not held)
    _ok(_mace(tmp_path, env, "submit", "--array", "--track", "decks"))
    seen = json.loads(probe.read_text())
    tools = [c["argv"][:2] for c in fake.calls()]
    if held:
        assert seen[1] == "4242_0" and seen[2] == "submitted", seen
        assert tools.index(["sbatch", "--hold"]) < tools.index(["scontrol", "release"])
    else:
        assert seen[1] is None, "the callback ran before the row was recorded"
    r = {x["calc_id"]: x for x in rows(tmp_path / "materials.db").values()}
    row = r[seen[0]]
    assert row["status"] == "failed", (row, fake.hook_log.read_text()[-3000:])
    assert row["slurm_job_id"] == "4242_0"
    other = [x for x in r.values() if x["calc_id"] != seen[0]]
    assert [x["status"] for x in other] == ["submitted"]


# --- no raw ids ------------------------------------------------------------------

def test_no_code_stores_a_raw_job_id():
    """A task's SLURM_JOB_ID (the last task's equals the array id) and the id
    in "JOB <raw> ON <node> CANCELLED" are never written as a row's job id."""
    offenders = []
    for path in (REPO_ROOT / "mace").rglob("*.py"):
        text = path.read_text(errors="ignore")
        for i, line in enumerate(text.splitlines(), 1):
            if "SLURM_JOB_ID" in line and re.search(r"slurm_job_id\s*=|update_calculation_status",
                                                     line):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{i}: {line.strip()}")
            if re.search(r"JOB \(\\d\+\) ON", line):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{i}: {line.strip()}")
    assert not offenders, offenders


# --- per-task actions after a tracked array submission -----------------------

def _tracked(tmp_path, names, deck=SP_D12, args=()):
    fake, env, d = _setup(tmp_path, names, deck=deck)
    _ok(_mace(tmp_path, env, "submit", "--array", "--track", *args, "decks"))
    db = tmp_path / "materials.db"
    by_job = {x["slurm_job_id"]: x for x in rows(db).values()}
    return fake, env, d, db, by_job


def _log_link(d, tag_dir, job, task_id, text):
    logs = tag_dir / "logs"
    log = logs / f"{tag_dir.name}-{task_id}.o"
    log.write_text(text)
    os.symlink(log, d / f"{job}-{task_id}.o")


def test_finalrun_loop_in_one_running_task(tmp_path):
    from test_finalrun_loop import DECK, loop_output
    fake, env, d, db, by_job = _tracked(tmp_path, ["si0", "si1", "si2"], deck=DECK)
    for t in range(3):
        fake.set_record(f"4242_{t}", state="RUNNING", start="2030-01-01T00:00:00")
    row = by_job["4242_2"]
    stem = Path(row["input_file"]).stem
    _ok(callback(env, d, db))                     # marks the three running
    (d / f"{stem}.out").write_text(loop_output(end="running"))
    before = len(fake.calls("sbatch"))
    _ok(callback(env, d, db))
    assert [c["argv"] for c in fake.calls("scancel")] == [["scancel", "4242_2"]]
    resub = fake.calls("sbatch")[before:]
    assert [Path(c["argv"][-1]) for c in resub] == [d.resolve() / f"{stem}.sh"]
    assert all("--array" not in c["script_text"] for c in resub)
    r = {x["slurm_job_id"]: x for x in rows(db).values()}
    assert r["4242_2"]["status"] == "resubmitted" and r["4242_2"]["error_type"] == "finalrun_loop"
    assert r["5000"]["status"] == "submitted"
    assert [r[f"4242_{t}"]["status"] for t in (0, 1)] == ["running", "running"]


def test_timeout_of_one_task_restarts_that_task_alone(tmp_path):
    fake, env, d, db, by_job = _tracked(tmp_path, ["m0", "m1", "m2"],
                                        args=("--walltime", "1:00:00"))
    (tag_dir,) = [w.parent for w in _wrappers(d)]
    fake.remove_record("4242_1")
    fake.set_record("4242_0", state="RUNNING")
    fake.set_record("4242_2", state="RUNNING")
    fake.update(sacct={"4242_1": "TIMEOUT"})
    row = by_job["4242_1"]
    stem = Path(row["input_file"]).stem
    (d / f"{stem}.out").write_text(" CYCLE   12 ETOT(AU) -7.5848990805E+01\n")
    _log_link(d, tag_dir, stem, "4242_1",
              "slurmstepd: error: *** JOB 9101 ON n001 CANCELLED AT 2030-01-01T01:00:00 "
              "DUE TO TIME LIMIT ***\n")
    before = len(fake.calls("sbatch"))
    _ok(callback(env, d, db))
    r = {x["slurm_job_id"]: x for x in rows(db).values()}
    assert r["4242_1"]["status"] == "resubmitted" and r["4242_1"]["error_type"] == "timeout_error"
    new = fake.calls("sbatch")[before:]
    test_only = [c for c in new if "--test-only" in c["argv"]]
    real = [c for c in new if "--test-only" not in c["argv"]]
    assert test_only and len(real) == 1
    bumped = Path(real[0]["argv"][-1])
    assert bumped.parent == d.resolve() and bumped != d.resolve() / f"{stem}.sh"
    assert "--array" not in real[0]["script_text"]
    assert "#SBATCH -t 0-02:00:00\n" in real[0]["script_text"], \
        real[0]["script_text"][:400]
    assert not fake.calls("scancel")
    assert [r[f"4242_{t}"]["status"] for t in (0, 2)] == ["running", "running"]
    # the restart is an ordinary job: it carries no array settings
    assert "slurm_array" not in json.loads(r["5000"]["settings_json"] or "{}")


def test_early_failure_cancel_and_bare_array_id(tmp_path, monkeypatch):
    fake, env, d, db, by_job = _tracked(tmp_path, ["m0", "m1"])
    for k in ("PATH", "FAKE_SLURM_STATE", "FAKE_SLURM_LOG"):
        monkeypatch.setenv(k, fake.env[k])
    monkeypatch.chdir(d)
    from mace.queue.manager import EnhancedCrystalQueueManager
    mgr = EnhancedCrystalQueueManager(d12_dir=str(d), db_path=str(db), enable_tracking=True)
    mgr.cancel_job("4242", by_job["4242_1"]["calc_id"])
    assert not fake.calls("scancel"), "scancel 4242 would cancel every task"
    mgr.cancel_job("4242_1", by_job["4242_1"]["calc_id"])
    assert [c["argv"] for c in fake.calls("scancel")] == [["scancel", "4242_1"]]


def test_untracked_array_id_is_protected_too(tmp_path, monkeypatch):
    fake, env, d = _setup(tmp_path, ["m0", "m1"])
    _ok(_mace(tmp_path, env, "submit", "--array", "decks"))
    for k in ("PATH", "FAKE_SLURM_STATE", "FAKE_SLURM_LOG"):
        monkeypatch.setenv(k, fake.env[k])
    from mace.queue.slurm_ids import safe_scancel
    assert safe_scancel("4242", search_dirs=[d]).returncode == 1
    assert not fake.calls("scancel")


def test_zombie_removal_cancels_the_task(tmp_path):
    fake, env, d = _setup(tmp_path, ["X"])
    (d / "X.out").write_text(SP_DONE)
    (d / "X-4242_3.o").write_text("")
    fake.update(records=[{"id": "4242_3", "array": "4242", "task": 3, "state": "RUNNING"},
                         {"id": "4242_4", "array": "4242", "task": 4, "state": "RUNNING"}])
    p = subprocess.run([sys.executable, str(REPO_ROOT / "mace" / "completion_checker.py"),
                        "--directory", str(d), "--remove-zombie-jobs"], input="y\nn\n",
                       env=dict(env, PYTHONPATH=str(REPO_ROOT)), capture_output=True, text=True,
                       timeout=300)
    _ok(p)
    assert [c["argv"] for c in fake.calls("scancel")] == [["scancel", "4242_3"]]


# --- generators that submit their own script -----------------------------------

SELF_SUBMITTING = TEMPLATE.read_text().replace("# sbatch $1.sh", "sbatch $1.sh")
ABSOLUTE_SBATCH = TEMPLATE.read_text().replace("# sbatch $1.sh", '"$FAKE_SBATCH" $1.sh')


@pytest.mark.parametrize("template,how", [(SELF_SUBMITTING, "path"), (ABSOLUTE_SBATCH, "absolute")])
def test_self_submitting_generator_is_submitted_once(tmp_path, monkeypatch, template, how):
    fake, env, d = _setup(tmp_path, ["s0", "s1", "s2"])
    (d / "o0.d12").write_text(OPT_D12)
    gen = tmp_path / "self_submitting.sh"
    gen.write_text(template)
    for k in ("PATH", "FAKE_SLURM_STATE", "FAKE_SLURM_LOG", "HOME", "USER"):
        monkeypatch.setenv(k, env[k])
    monkeypatch.setenv("FAKE_SBATCH", str(fake.bindir / "sbatch"))
    monkeypatch.chdir(d)
    from mace.queue.manager import EnhancedCrystalQueueManager
    mgr = EnhancedCrystalQueueManager(d12_dir=str(d), db_path=str(tmp_path / "materials.db"),
                                      enable_tracking=True, organize_outputs=False)
    real = mgr._get_submit_script_for_calc_type
    monkeypatch.setattr(mgr, "_get_submit_script_for_calc_type",
                        lambda ct: str(gen) if ct == "OPT" else real(ct))
    results = mgr.submit_calculations_as_arrays(sorted(d.glob("*.d12")))
    assert all(job for _, job in results), results
    opt = [c for c in fake.calls("sbatch") if "o0" in c["argv"][-1] and "--test-only" not in c["argv"]]
    if how == "absolute":
        assert len(opt) == 1, [c["argv"] for c in opt]
    else:
        # the default path's behavior for such a template, unchanged
        assert all(c["argv"][1] != "--hold" for c in opt)
    (w,) = _wrappers(d)
    assert "o0" not in (w.parent / "manifest.tsv").read_text()
    assert "--array=0-2" in w.read_text()
    by_job = {x["slurm_job_id"]: x for x in rows(tmp_path / "materials.db").values()}
    assert sorted(j for j in by_job if j.startswith("4242_")) == ["4242_0", "4242_1", "4242_2"]
    assert any(not j.startswith("4242") for j in by_job)


@pytest.mark.parametrize("track", [False, True])
def test_a_space_in_the_path_keeps_the_wrapper_directives_whole(tmp_path, track):
    """sbatch splits #SBATCH lines on whitespace, so no directive of the
    wrapper may carry the (absolute) path of a folder with a space in it."""
    root = tmp_path / "probe space root"
    fake, env, _ = _setup(root, [])
    d = root / "my decks"
    d.mkdir()
    for n in ("m0", "m1"):
        (d / f"{n}.d12").write_text(SP_D12)
    p = _ok(_mace(root, env, "submit", "--array", *(["--track"] if track else []), "my decks"))
    (w,) = _wrappers(d)
    (sb,) = fake.calls("sbatch")
    assert sb["argv"] == ["sbatch", "--hold", w.name] and sb["cwd"] == str(w.parent)
    for line in w.read_text().splitlines():
        if line.startswith("#SBATCH"):
            assert "space" not in line and "my decks" not in line, line
    (o,) = [l for l in w.read_text().splitlines() if l.startswith("#SBATCH -o ")]
    assert (Path(sb["cwd"]) / o.split()[2]).parent == w.parent / "logs"
    assert (w.parent / "logs").is_dir()
    assert "4242_0" in p.stdout and "4242_1" in p.stdout
    if track:
        assert sorted(x["slurm_job_id"] for x in rows(root / "materials.db").values()) == [
            "4242_0", "4242_1"]
