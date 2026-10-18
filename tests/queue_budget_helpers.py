"""Shared pieces of the queue budget tests (mace/queue/budget.py): a fake
SLURM with a per-user MaxSubmitJobs, site settings, held rows made by hand,
and the real mace_cli / job callback run as subprocesses."""
import json
import re
import shutil
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

from array_helpers import callback, rows
from conftest import REPO_ROOT
from fake_slurm import FakeSlurm
from test_array_default_path_golden import OPT_D12, SP_D12, _env, _mace

__all__ = ["callback", "rows", "OPT_D12", "SP_D12", "_mace", "setup", "site", "decks", "live",
           "held_rows", "make_held", "make_db", "drain", "plan_spec", "launch_spec", "complete",
           "by_type", "free_others", "add_others", "sbatch_calls"]


def setup(tmp_path, max_submit=None, others=0, site_text=None, assoc=None, state=None):
    """(fake, env): the fake SLURM (``max_submit``: SLURM's own per-user
    limit; ``assoc``: the limit scontrol show assoc_mgr reports), ``others``
    running jobs of the user that are not MACE's, and site settings."""
    st = dict(state or {})
    if max_submit:
        st["max_submit"] = max_submit
    if assoc:
        st["assoc_mgr"] = (f"ClusterName=c Account=mendoza_q UserName=tester(1000) Partition= "
                           f"DefAssoc=Yes MaxSubmitJobs={assoc}({others})")
    fake = FakeSlurm(tmp_path, st)
    for i in range(others):
        fake.add_record(id=str(900000 + i), state="RUNNING", user="tester", name=f"other{i}")
    env = _env(tmp_path, fake)
    env["MACE_QUEUE_STATE_DIR"] = str(tmp_path / "queue_state")
    env["XDG_CACHE_HOME"] = str(tmp_path / "cache")
    if site_text is not None:
        site(env, tmp_path, site_text)
    else:
        env["MACE_SITE_CONFIG"] = "none"
    return fake, env


def site(env, tmp_path, text):
    p = Path(tmp_path) / "site.toml"
    p.write_text(text)
    env["MACE_SITE_CONFIG"] = str(p)


def decks(folder, n, prefix="m", text=OPT_D12):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (folder / f"{prefix}{i:03d}_dia_opt_rev1.d12").write_text(text)
    return folder


def live(fake):
    return sum(1 for r in fake.records() if r["state"] in ("PENDING", "RUNNING", "CONFIGURING", "COMPLETING"))


def sbatch_calls(fake, sentinel=False):
    """The real sbatch calls (the mace-drain sentinel's only with sentinel=True)."""
    return [c for c in fake.calls("sbatch") if "--test-only" not in c["argv"]
            and (sentinel or c["argv"][-1] != "mace-drain.sh")]


def held_rows(db):
    return [r for r in rows(db).values() if r["status"] == "held"]


def make_db(path):
    """An empty MACE database (the real schema)."""
    sys.path.insert(0, str(REPO_ROOT))
    from mace.database.materials import MaterialDatabase
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    MaterialDatabase(str(path))
    return Path(path)


def make_held(db, work, name, step=1, since=None, reason="budget", workflow_id=None, calc_type=None,
              status="held", job_id=None, error=None, created=None):
    """A held row (or, with ``status``, a row of another state) whose job
    script is a plain batch file."""
    work = Path(work) / name
    work.mkdir(parents=True, exist_ok=True)
    (work / f"{name}.d12").write_text(SP_D12)
    (work / f"{name}.sh").write_text(f"#!/bin/bash\n#SBATCH -J {name}\n#SBATCH -t 1:00:00\necho {name}\n")
    since = since or datetime.now().isoformat(timespec="seconds")
    st = {"workflow_step": step}
    if workflow_id:
        st["workflow_id"] = workflow_id
    if status == "held":
        st.update(held_reason=reason, held_since=since,
                  held_submit={"argv": ["sbatch", f"{name}.sh"], "cwd": str(work), "n": 1, "on_release": {}})
    calc_id = f"{name}_{calc_type or ('OPT' if step == 1 else 'SP')}"
    with sqlite3.connect(db) as conn:
        conn.execute("INSERT INTO calculations (calc_id, material_id, calc_type, status, created_at, input_file, "
                     "work_dir, settings_json, slurm_job_id, error_message) VALUES (?,?,?,?,?,?,?,?,?,?)",
                     (calc_id, name, calc_type or ("OPT" if step == 1 else "SP"), status,
                      created or since, str(work / f"{name}.d12"), str(work), json.dumps(st), job_id, error))
    return calc_id


def register(env, db):
    """Put a database in the registry of active workflows, as a hold does."""
    import os
    saved = os.environ.get("MACE_QUEUE_STATE_DIR")
    os.environ["MACE_QUEUE_STATE_DIR"] = env["MACE_QUEUE_STATE_DIR"]
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from mace.queue import budget
        budget.register(db)
    finally:
        if saved is None:
            os.environ.pop("MACE_QUEUE_STATE_DIR", None)
        else:
            os.environ["MACE_QUEUE_STATE_DIR"] = saved


def drain(tmp_path, env, *args, cwd=None):
    p = _mace(tmp_path, env, "queue", "drain", *args, cwd=cwd)
    return p


def plan_spec(root, env, sequence, inputs="decks", kind="d12", execution=None):
    spec = {"version": 1, "inputs": {"type": kind, "directory": inputs}, "sequence": sequence}
    if execution:
        spec["execution"] = execution
    (Path(root) / "spec.json").write_text(json.dumps(spec))
    p = _mace(root, env, "workflow", "--spec", "spec.json", "--write-plan", "--json")
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    return json.loads(p.stdout)


def launch_spec(root, env, w):
    p = _mace(root, env, *w["launch"][1:])
    db = next(Path(root).rglob(f".mace_context_{w['workflow_id']}/materials.db"))
    return p, db


def by_type(db, t, status=None):
    return sorted((r for r in rows(db).values() if r["calc_type"] == t
                   and (status is None or r["status"] == status)), key=lambda r: r["material_id"])


def complete(fake, env, db, row, out, f9=None, remove=True):
    """The job of ``row`` ends: its output appears, the job leaves the queue
    and its completion callback runs (with the job script's exports)."""
    work = Path(row["input_file"]).parent
    stem = Path(row["input_file"]).stem
    shutil.copy(out, work / f"{stem}.out")
    if f9:
        shutil.copy(f9, work / f"{stem}.f9")
    sh = work / f"{stem}.sh"
    exports = dict(re.findall(r'^export (MACE_WORKFLOW_ID|MACE_CONTEXT_DIR|MACE_ISOLATION_MODE)'
                              r'="([^"]*)"', sh.read_text(), re.M)) if sh.is_file() else {}
    fake.set_record(row["slurm_job_id"], state="COMPLETED")
    p = callback(env, work, db, mode="completion", extra_env=dict(exports, SLURM_JOB_ID=row["slurm_job_id"]))
    if remove:
        fake.remove_record(row["slurm_job_id"])
    return p


def free_others(fake, n):
    others = [r for r in fake.records() if r.get("name", "").startswith("other")][:n]
    for r in others:
        fake.remove_record(r["id"])


def add_others(fake, n, start=950000):
    for i in range(n):
        fake.add_record(id=str(start + i), state="PENDING", user="tester", name=f"other_x{i}")


def old(minutes):
    return (datetime.now() - timedelta(minutes=minutes)).isoformat(timespec="seconds")
