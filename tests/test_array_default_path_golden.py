"""The default (one job per calculation) submission path, pinned byte for byte.

SLURM job arrays are opt-in (`mace submit --array`, the plan's `job_arrays`
key). Without them every job script, every SLURM command and every database
row must stay exactly what MACE wrote before arrays existed. These goldens
were captured from the real CLI, run as a subprocess with the logging fake
SLURM of tests/fake_slurm.py, BEFORE any array code was written; every later
change must keep them green unchanged.

What is compared, per scenario:
- every sbatch/squeue/sacct/scancel/scontrol/sinfo/sacctmgr call: argv, working
  directory, the SLURM_* variables in its environment and, for sbatch, the
  submitted script. The calls are compared as a whole and the sequence of
  tool names in order. (The order of the decks themselves comes from
  os.listdir/glob, which is the file system's order, so the calls are
  compared sorted.) The one intended change is that MACE's own squeue and
  sacct calls carry SLURM_BITSTR_LEN=0 (full array task expressions); it shows
  in the "env" field of those calls and nowhere else;
- every job script and workflow_scripts/ template written;
- every database row (paths relative, timestamps removed);
- the workflow plan JSON (which has no job_arrays key on this path).

Regenerate (only for an intended change, reviewed by hand):
    MACE_UPDATE_ARRAY_GOLDEN=1 pytest tests/test_array_default_path_golden.py
"""
import json
import os
import re
import site
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT
from fake_slurm import FakeSlurm

MACE_CLI = REPO_ROOT / "mace_cli"
GOLDEN_DIR = Path(__file__).resolve().parent / "data" / "array_default_golden"
UPDATE = os.environ.get("MACE_UPDATE_ARRAY_GOLDEN") == "1"

SP_D12 = """TESTGEOM
CRYSTAL
0 0 0
227
3.5
1
6 0.125 0.125 0.125
END
6 2
0 0 2 2. 1.
 8.0 1.0
 2.0 1.0
0 1 1 4. 1.
 1.0 1.0 1.0
99 0
END
DFT
B3LYP
END
SHRINK
8 8
END
"""
OPT_D12 = SP_D12.replace("6 0.125 0.125 0.125\nEND\n",
                         "6 0.125 0.125 0.125\nOPTGEOM\nFULLOPTG\nENDOPT\nEND\n", 1)
BAND_D3 = """BAND
TESTBAND
2 8 50 1 8 1 0
 0 0 0  4 0 4
 4 0 4  4 2 4
END
"""


def _env(tmp_path, fake):
    env = dict(fake.env)
    # A private HOME: the legacy status file is also looked for in ~, and a
    # user's own crystal_job_status.json there must not leak into the run.
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env["HOME"] = str(home)
    env["PYTHONUSERBASE"] = site.getuserbase()
    env["USER"] = "tester"
    env.pop("MACE_HOME", None)
    env.pop("PYTHONPATH", None)
    return env


def _mace(tmp_path, env, *args, cwd=None, stdin=""):
    return subprocess.run([sys.executable, str(MACE_CLI), "--no-banner", *args],
                          cwd=str(cwd or tmp_path), env=env, input=stdin,
                          capture_output=True, text=True, timeout=600)


def _callback(env, workdir, db):
    # As the job script runs it: the submitting install pinned on PYTHONPATH.
    env = dict(env, PYTHONPATH=str(REPO_ROOT), MACE_HOME=str(REPO_ROOT))
    return subprocess.run([sys.executable, str(REPO_ROOT / "mace" / "queue" / "manager.py"),
                           "--callback-mode", "status_check", "--db-path", str(db)],
                          cwd=str(workdir), env=env, capture_output=True, text=True,
                          timeout=600)


def _norm(text, tmp_path):
    if text is None:
        return None
    text = str(text)
    for root, tag in ((str(tmp_path.resolve()), "<TMP>"), (str(tmp_path), "<TMP>"),
                      (str(REPO_ROOT.resolve()), "<REPO>"), (str(REPO_ROOT), "<REPO>")):
        text = text.replace(root, tag)
    text = re.sub(r"workflow_\d{8}_\d{6}", "workflow_<ID>", text)
    text = re.sub(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?", "<ISO>", text)
    text = re.sub(r"\d{8}_\d{6}\d*", "<TS>", text)
    text = re.sub(r"_custom_\d+\.sh", "_custom_<PID>.sh", text)
    return text


def _snapshot(tmp_path, fake):
    calls = []
    for c in fake.calls():
        entry = {"tool": c["tool"],
                 "argv": [_norm(a, tmp_path) for a in c["argv"]],
                 "cwd": _norm(c["cwd"], tmp_path),
                 "env": {k: _norm(v, tmp_path) for k, v in c["env"].items()}}
        if c["tool"] == "sbatch":
            entry["script"] = _norm(c.get("script"), tmp_path)
            entry["script_text"] = _norm(c.get("script_text"), tmp_path)
        calls.append(entry)
    files = {}
    for p in sorted(tmp_path.rglob("*.sh")):
        rel = p.relative_to(tmp_path)
        if rel.parts[0] in ("fakebin", "home") or ".mace_context_" in str(rel):
            continue
        files[_norm(str(rel), tmp_path)] = _norm(p.read_text(errors="replace"), tmp_path)
    plans = {}
    for p in sorted(tmp_path.rglob("workflow_plan_*.json")):
        plans[_norm(str(p.relative_to(tmp_path)), tmp_path)] = json.loads(
            _norm(p.read_text(), tmp_path))
    rows = []
    for db in sorted(tmp_path.rglob("materials.db")):
        with sqlite3.connect(db) as conn:
            conn.row_factory = sqlite3.Row
            for r in conn.execute("SELECT * FROM calculations"):
                r = dict(r)
                settings = json.loads(r.get("settings_json") or "{}")
                settings = {k: v for k, v in settings.items()
                            if "time" not in k and not k.endswith("_at")}
                rows.append({
                    "db": _norm(str(db.relative_to(tmp_path)), tmp_path),
                    "material_id": r["material_id"], "calc_type": r["calc_type"],
                    "status": r["status"], "slurm_job_id": r["slurm_job_id"],
                    "slurm_state": r.get("slurm_state"),
                    "input_file": _norm(r["input_file"], tmp_path),
                    "work_dir": _norm(r["work_dir"], tmp_path),
                    "job_script": _norm(r.get("job_script"), tmp_path),
                    "output_file": _norm(r.get("output_file"), tmp_path),
                    "error_type": r.get("error_type"),
                    "error_message": _norm(r.get("error_message"), tmp_path),
                    "settings": json.loads(_norm(json.dumps(settings, sort_keys=True),
                                                 tmp_path)),
                })
    rows.sort(key=lambda r: json.dumps(r, sort_keys=True))
    return {"calls": calls, "files": files, "plans": plans, "rows": rows}


def _decks(d, names):
    d.mkdir(parents=True, exist_ok=True)
    for name in names:
        if name.endswith(".d3"):
            (d / name).write_text(BAND_D3)
        elif "_opt" in name:
            (d / name).write_text(OPT_D12)
        else:
            (d / name).write_text(SP_D12)


def scenario_submit_dir_mixed(tmp_path, fake, env):
    _decks(tmp_path / "decks", ["a_opt.d12", "b_sp.d12", "c_band.d3"])
    return [_mace(tmp_path, env, "submit", "decks", stdin="3\n")]


def scenario_submit_track_dir(tmp_path, fake, env):
    _decks(tmp_path / "decks", ["a_opt.d12", "b_sp.d12", "c_band.d3"])
    return [_mace(tmp_path, env, "submit", "--track", "decks", stdin="3\n")]


def scenario_submit_progress(tmp_path, fake, env):
    _decks(tmp_path / "decks", ["a_opt.d12"])
    return [_mace(tmp_path, env, "submit", "--progress", "opt_sp", "decks/a_opt.d12")]


def scenario_submit_walltime_exclude(tmp_path, fake, env):
    _decks(tmp_path / "decks", ["a_opt.d12", "b_sp.d12"])
    return [_mace(tmp_path, env, "submit", "--walltime", "0:30:00", "--exclude-nodes", "n1",
                  "decks")]


def scenario_workflow_quick_start(tmp_path, fake, env):
    _decks(tmp_path / "decks", ["a_opt.d12", "b_opt.d12"])
    return [_mace(tmp_path, env, "workflow", "--quick-start", "--d12-dir", "decks",
                  "--workflow", "opt_sp")]


def scenario_status_check_callback(tmp_path, fake, env):
    """The queue sweep a job callback runs, on plain job ids: one running, one
    left the queue with no record (kept), one FAILED without output."""
    _decks(tmp_path / "decks", ["a_sp.d12", "b_sp.d12", "c_sp.d12"])
    procs = [_mace(tmp_path, env, "submit", "--track", "decks")]
    fake.set_record("5000", state="RUNNING", start="2030-01-01T00:00:00")
    fake.remove_record("5001")
    fake.remove_record("5002")
    fake.update(sacct={"5002": "FAILED"})
    procs.append(_callback(env, tmp_path / "decks", tmp_path / "materials.db"))
    return procs


SCENARIOS = {
    "submit_dir_mixed": scenario_submit_dir_mixed,
    "submit_track_dir": scenario_submit_track_dir,
    "submit_progress": scenario_submit_progress,
    "submit_walltime_exclude": scenario_submit_walltime_exclude,
    "workflow_quick_start": scenario_workflow_quick_start,
    "status_check_callback": scenario_status_check_callback,
}


def run_scenario(name, tmp_path):
    fake = FakeSlurm(tmp_path)
    env = _env(tmp_path, fake)
    procs = SCENARIOS[name](tmp_path, fake, env)
    for p in procs:
        assert p.returncode == 0, p.stdout[-4000:] + p.stderr[-4000:]
    return _snapshot(tmp_path, fake)


def _canonical_calls(calls):
    return sorted(json.dumps(c, sort_keys=True) for c in calls)


def _canonical_plans(plans):
    # A plan lists its input decks in directory order, which is the
    # filesystem's (tmpfs and ext4 differ); the order itself is not pinned.
    plans = json.loads(json.dumps(plans))
    for plan in plans.values():
        for kind, names in (plan.get("input_files") or {}).items():
            plan["input_files"][kind] = sorted(names)
    return plans


def _canonical_rows(rows):
    # Decks go in in directory order, so which deck gets which (fake) job id
    # depends on the filesystem; every other field and the set of ids are pinned.
    masked = sorted(json.dumps(dict(r, slurm_job_id="<JOB>"), sort_keys=True)
                    for r in rows)
    return masked, sorted(str(r.get("slurm_job_id")) for r in rows)


def compare(snapshot, golden):
    assert [c["tool"] for c in snapshot["calls"]] == [c["tool"] for c in golden["calls"]]
    assert _canonical_calls(snapshot["calls"]) == _canonical_calls(golden["calls"])
    assert snapshot["files"] == golden["files"]
    assert _canonical_plans(snapshot["plans"]) == _canonical_plans(golden["plans"])
    assert _canonical_rows(snapshot["rows"]) == _canonical_rows(golden["rows"])


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_default_path_is_unchanged(name, tmp_path):
    snapshot = run_scenario(name, tmp_path)
    golden_file = GOLDEN_DIR / f"{name}.json"
    if UPDATE:
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        golden_file.write_text(json.dumps(snapshot, indent=1, sort_keys=True) + "\n")
    golden = json.loads(golden_file.read_text())
    compare(snapshot, golden)
    for plan in snapshot["plans"].values():
        assert "job_arrays" not in plan
