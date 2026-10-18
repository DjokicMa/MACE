"""MACE's user-wide SLURM job budget (mace/queue/budget.py), through the real
mace_cli and job callback run as subprocesses with the logging fake SLURM
(tests/fake_slurm.py: ``max_submit`` refuses past a per-user MaxSubmitJobs
with SLURM's own text; ``peak_live`` is the most jobs ever queued).

The budget is the user's submit limit minus a reserve ([queue] in
site.toml). Every MACE sbatch is admitted against it; what does not fit is
held (a database row with status 'held' and its sbatch), never lost, and a
drain submits it later in the [queue] order."""
import json
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

from conftest import REPO_ROOT
from queue_budget_helpers import (OPT_D12, SP_D12, _mace, by_type, callback, decks, drain, held_rows, live,
                                  make_db, make_held, old, plan_spec, launch_spec, register, rows,
                                  sbatch_calls, setup, site)

MANAGER = REPO_ROOT / "mace" / "queue" / "manager.py"


def status_json(tmp_path, env):
    p = _mace(tmp_path, env, "queue", "status", "--json")
    assert p.returncode == 0, p.stdout + p.stderr
    return json.loads(p.stdout)


# ------------------------------------------------------------------ 1. budget resolution
ASSOC_FMT = "account,partition,qos,maxjobs,maxsubmit,maxwall,grptres,maxtres,defaultqos"


@pytest.mark.parametrize("case", ["site_budget", "assoc_only", "sacctmgr_qos_only", "both_min", "nothing",
                                  "legacy_max_jobs", "reserve_once"])
def test_budget_resolution(tmp_path, case):
    state = {}
    site_text = None
    if case == "site_budget":
        site_text = "[queue]\nbudget = 37\nreserve = 3\n"
    if case in ("assoc_only", "both_min", "reserve_once"):
        state["assoc_mgr"] = ("ClusterName=c Account=mendoza_q UserName=tester(1000) Partition= DefAssoc=Yes "
                              "MaxSubmitJobs=1040(12)")
    if case in ("sacctmgr_qos_only", "both_min"):
        state["assoc_text"] = "mendoza_q|||||||||\n".replace("|||||||||", "||normal||||||")
        state["qos_text"] = "normal|800|||\n"
    if case == "legacy_max_jobs":
        site_text = "[queue]\nmax_jobs = 950\nreserve = 50\n"
    fake, env = setup(tmp_path, state=state, site_text=site_text)
    st = status_json(tmp_path, env)
    want = {"site_budget": (37, None, 3), "assoc_only": (940, 1040, 100), "sacctmgr_qos_only": (720, 800, 80),
            "both_min": (720, 800, 80), "nothing": (900, None, 50), "legacy_max_jobs": (900, None, 50),
            "reserve_once": (940, 1040, 100)}[case]
    assert (st["budget"], st["limit"], st["reserve"]) == want, st
    if case == "nothing":
        assert "could not be read" in st["source"]


def test_callback_flags_carry_the_budget_once(tmp_path):
    """site.resolve_queue / clamp_queue: the callback keeps max_jobs - reserve
    queued, so the reserve is not added a second time (1040 - 100 = 940,
    not 1040 - 52 - 52 = 936 or 888)."""
    sys.path.insert(0, str(REPO_ROOT))
    from mace.submission import site as S
    assert S.queue_from_limit(1040) == (1040, 100)
    assert S.queue_from_limit(1000) == (950, 50)          # the same budget, 900: the templates' bytes
    assert S.queue_from_limit(200) == (200, 20)
    assert S.default_reserve(1040) == 100 and S.default_reserve(30) == 3


# ------------------------------------------------------------------ 2. first step never exceeds B
def test_first_step_deck_path_holds_what_does_not_fit(tmp_path):
    fake, env = setup(tmp_path, max_submit=30, others=5, site_text="[queue]\nbudget = 20\n")
    root = tmp_path / "wf"
    decks(root / "decks", 30)
    w = plan_spec(root, env, ["OPT", "SP"])
    p, db = launch_spec(root, env, w)
    out = p.stdout + p.stderr
    assert "held 15" in out, out[-3000:]
    r = rows(db)
    assert sum(1 for x in r.values() if x["status"] == "submitted") == 15
    held = held_rows(db)
    assert len(held) == 15
    assert all(json.loads(x["settings_json"])["held_reason"] == "budget" for x in held)
    assert fake.load().get("peak_live", 0) <= 20
    assert fake.load().get("refused", 0) == 0
    assert len(sbatch_calls(fake)) == 15
    # mace workflow --status shows them as held (not queued) and --json counts them
    home = Path(w["plan_file"]).parent.parent
    p = _mace(home, env, "workflow", "--status", w["workflow_id"], "--json")
    st = json.loads(p.stdout)
    assert st["held"] == {"budget": 15}, p.stdout[-2000:]
    assert st["counts"].get("held") == 15
    p = _mace(home, env, "workflow", "--status", w["workflow_id"])
    assert "held" in p.stdout and "queue budget full" in p.stdout


def test_first_step_cif_path_holds_what_does_not_fit(tmp_path):
    pytest.importorskip("ase", reason="the CIF conversion reads CIFs with ase")
    fake, env = setup(tmp_path, max_submit=10, site_text="[queue]\nbudget = 3\n")
    root = tmp_path / "wf"
    (root / "inputs").mkdir(parents=True)
    src = REPO_ROOT / "tests" / "data" / "1_dia_opt_BULK_OPTGEOM_symm.cif"
    for i in range(5):
        (root / "inputs" / f"c{i}_dia_opt_rev1.cif").write_text(src.read_text())
    w = plan_spec(root, env, ["OPT", "SP"], inputs="inputs", kind="cif")
    p, db = launch_spec(root, env, w)
    out = p.stdout + p.stderr
    r = rows(db)
    assert sum(1 for x in r.values() if x["status"] == "submitted") == 3, out[-4000:]
    assert len(held_rows(db)) == 2
    assert not [x for x in r.values() if x["status"] == "failed"]
    assert fake.load().get("peak_live", 0) <= 3
    assert "held 2" in out


def test_first_step_array_submits_the_tasks_that_fit(tmp_path):
    fake, env = setup(tmp_path, max_submit=40, others=5, site_text="[queue]\nbudget = 20\n")
    root = decks(tmp_path / "decks", 30)
    p = _mace(tmp_path, env, "submit", "--array", "--progress", "opt_sp", str(root))
    out = p.stdout + p.stderr
    assert p.returncode == 0, out[-3000:]
    db = next(tmp_path.rglob(".mace_context_*/materials.db"))
    r = rows(db)
    tasks = [x for x in r.values() if x["status"] == "submitted"]
    assert len(tasks) == 15 and all("_" in x["slurm_job_id"] for x in tasks)
    assert len(held_rows(db)) == 15
    assert fake.load().get("peak_live", 0) <= 20
    assert "Nothing was submitted" not in out
    # the held tasks are released later as single jobs, still within the budget
    for x in tasks[:4]:
        fake.remove_record(x["slurm_job_id"])
    p = drain(tmp_path, env, "--db", str(db))
    assert p.returncode == 0, p.stdout + p.stderr
    assert len(held_rows(db)) == 11
    assert fake.load().get("peak_live", 0) <= 20


# ------------------------------------------------------------------ 4. refusal = held
def test_slurm_refusal_holds_the_job_and_stops_the_pass(tmp_path):
    """The budget says 20 but SLURM refuses past 8 (a limit MACE could not
    read): the 9th job is held (slurm_limit), nothing more is tried in that
    pass, SLURM's text is shown and the limit is remembered."""
    fake, env = setup(tmp_path, max_submit=8, site_text="[queue]\nbudget = 20\n")
    root = decks(tmp_path / "decks", 12)
    p = _mace(tmp_path, env, "submit", "--track", str(root))
    out = p.stdout + p.stderr
    db = tmp_path / "materials.db"
    r = rows(db)
    assert sum(1 for x in r.values() if x["status"] == "submitted") == 8, out[-3000:]
    held = held_rows(db)
    assert len(held) == 4
    reasons = sorted(json.loads(x["settings_json"])["held_reason"] for x in held)
    assert reasons == ["slurm_limit"] * 4
    assert not [x for x in r.values() if x["status"] == "failed"]
    assert len(sbatch_calls(fake)) == 9          # no sbatch after SLURM said no
    assert "QOSMaxSubmitJobPerUserLimit" in out or "job submit limit" in out
    obs = json.loads((Path(env["MACE_QUEUE_STATE_DIR"]) / "observed_limit.json").read_text())
    assert obs["limit"] == 8
    # the learned limit lowers the budget for the next hour
    st = status_json(tmp_path, env)
    assert st["budget"] < 8


def test_other_sbatch_errors_stay_failed_with_slurms_message(tmp_path):
    fake, env = setup(tmp_path, state={"reject_account": True}, site_text="[queue]\nbudget = 20\n")
    root = decks(tmp_path / "decks", 2)
    p = _mace(tmp_path, env, "submit", "--track", str(root))
    out = p.stdout + p.stderr
    assert "Invalid account" in out
    assert not held_rows(tmp_path / "materials.db")


# ------------------------------------------------------------------ 5. resume order
def _order_setup(tmp_path, free, order, share=None):
    fake, env = setup(tmp_path, others=10, site_text=f"[queue]\nbudget = {10 + free}\norder = \"{order}\"\n"
                      + (f"followup_share = {share}\n" if share is not None else ""))
    db = make_db(tmp_path / "materials.db")
    work = tmp_path / "work"
    # first steps F0 (oldest) .. F2, next steps S0 (oldest) .. S2 and a step-3 row
    for i in range(3):
        make_held(db, work, f"F{i}", step=1, since=old(60 - i))
        make_held(db, work, f"S{i}", step=2, since=old(50 - i))
    make_held(db, work, "B0", step=3, since=old(40), calc_type="BAND")
    return fake, env, db


def _released(db):
    return sorted(x["material_id"] for x in rows(db).values() if x["status"] == "submitted")


def test_depth_order_releases_next_steps_first_oldest_first(tmp_path):
    fake, env, db = _order_setup(tmp_path, 3, "depth")
    p = drain(tmp_path, env, "--db", str(db))
    assert p.returncode == 0, p.stdout + p.stderr
    assert _released(db) == ["B0", "S0", "S1"]
    seq = [c["argv"][-1] for c in sbatch_calls(fake)]
    assert seq == ["B0.sh", "S0.sh", "S1.sh"]          # deepest step, then oldest first (no LIFO)


def test_breadth_order_releases_first_steps_first(tmp_path):
    fake, env, db = _order_setup(tmp_path, 4, "breadth")
    p = drain(tmp_path, env, "--db", str(db))
    assert p.returncode == 0, p.stdout + p.stderr
    assert _released(db) == ["F0", "F1", "F2", "S0"]


def test_mixed_order_shares_the_free_slots(tmp_path):
    fake, env, db = _order_setup(tmp_path, 4, "mixed", share=0.5)
    p = drain(tmp_path, env, "--db", str(db))
    assert p.returncode == 0, p.stdout + p.stderr
    assert _released(db) == ["B0", "F0", "F1", "S0"]


def test_dry_run_and_json(tmp_path):
    fake, env, db = _order_setup(tmp_path, 2, "depth")
    p = drain(tmp_path, env, "--db", str(db), "--dry-run", "--json")
    rep = json.loads(p.stdout)
    assert [r["material_id"] for r in rep["released"]] == ["B0", "S0"]
    assert rep["dry_run"] and not sbatch_calls(fake)
    assert len(held_rows(db)) == 7
    p = drain(tmp_path, env, "--db", str(db), "--json")
    rep = json.loads(p.stdout)
    assert len(rep["released"]) == 2 and rep["held"] == {"budget": 5}


# ------------------------------------------------------------------ 6. caps and old 1.1.5 plans
def test_workflow_cap_counts_only_real_live_rows(tmp_path):
    """An explicit max_jobs is a hard cap per job; held rows never count as
    live, so they cannot deadlock it (the S6 phantom rows)."""
    fake, env = setup(tmp_path, site_text="[queue]\nbudget = 50\n")
    root = tmp_path / "wf"
    decks(root / "decks", 10)
    w = plan_spec(root, env, ["OPT", "SP"], execution={"max_jobs": 4})
    p, db = launch_spec(root, env, w)
    r = rows(db)
    assert sum(1 for x in r.values() if x["status"] == "submitted") == 4, (p.stdout + p.stderr)[-3000:]
    held = held_rows(db)
    assert len(held) == 6
    assert {json.loads(x["settings_json"])["held_reason"] for x in held} == {"workflow_cap"}
    # all four jobs leave the queue: the drain releases four more, never above the cap
    for x in by_type(db, "OPT", "submitted"):
        fake.remove_record(x["slurm_job_id"])
        with sqlite3.connect(db) as conn:
            conn.execute("UPDATE calculations SET status='completed' WHERE calc_id=?", (x["calc_id"],))
    p = drain(tmp_path, env, "--db", str(db))
    assert sum(1 for x in rows(db).values() if x["status"] == "submitted") == 4
    assert len(held_rows(db)) == 2


def test_a_115_plan_with_200_is_uncapped(tmp_path):
    sys.path.insert(0, str(REPO_ROOT))
    from mace.queue import budget
    assert budget.normalize_cap(200) is None and budget.normalize_cap(5) == 5 and budget.normalize_cap(None) is None
    fake, env = setup(tmp_path, site_text="[queue]\nbudget = 50\n")
    root = tmp_path / "wf"
    decks(root / "decks", 3)
    w = plan_spec(root, env, ["OPT", "SP"])
    plan = json.loads(Path(w["plan_file"]).read_text())
    assert plan["execution_settings"]["max_concurrent_jobs"] is None
    plan["execution_settings"]["max_concurrent_jobs"] = 200          # as 1.1.5 wrote it
    Path(w["plan_file"]).write_text(json.dumps(plan))
    from mace.workflow.engine import WorkflowEngine
    eng = WorkflowEngine.__new__(WorkflowEngine)
    eng._load_workflow_plan = lambda wid: plan
    assert eng._job_cap(w["workflow_id"]) is None


def test_old_callback_flags_use_the_smaller_budget(tmp_path):
    """A 1.1.5/1.1.6 job script's --max-jobs 950 --reserve 50 with a site
    budget of 940: the callback keeps at most 900 queued."""
    fake, env = setup(tmp_path, others=897, site_text="[queue]\nlimit = 1040\n")
    db = make_db(tmp_path / "materials.db")
    for i in range(5):
        make_held(db, tmp_path / "work", f"F{i}", since=old(30 - i))
    register(env, db)
    p = subprocess.run([sys.executable, str(MANAGER), "--max-jobs", "950", "--reserve", "50",
                        "--callback-mode", "status_check", "--db-path", str(db)],
                       cwd=str(tmp_path), env=dict(env, PYTHONPATH=str(REPO_ROOT), MACE_HOME=str(REPO_ROOT)),
                       capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    assert len(sbatch_calls(fake)) == 3, p.stdout[-3000:]
    assert len(held_rows(db)) == 2
    # without the old flags the site budget (940) applies
    p = drain(tmp_path, env, "--db", str(db))
    assert not held_rows(db)


# ------------------------------------------------------------------ 7. cross-workflow budget
def test_two_workflows_and_a_manual_submit_share_the_budget(tmp_path):
    fake, env = setup(tmp_path, max_submit=60, others=4, site_text="[queue]\nbudget = 20\n")
    procs = []
    for k in ("a", "b"):
        root = tmp_path / k
        decks(root / "decks", 12, prefix=k)
        w = plan_spec(root, env, ["OPT", "SP"])
        procs.append(subprocess.Popen([sys.executable, str(REPO_ROOT / "mace_cli"), "--no-banner",
                                       *w["launch"][1:]], cwd=str(root), env=env,
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
    m = decks(tmp_path / "manual", 10, prefix="x")
    procs.append(subprocess.Popen([sys.executable, str(REPO_ROOT / "mace_cli"), "--no-banner", "submit",
                                   "--track", str(m)], cwd=str(tmp_path), env=env,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
    outs = [p.communicate(timeout=600)[0] for p in procs]
    assert fake.load().get("peak_live", 0) <= 20, outs
    assert live(fake) == 20
    assert fake.load().get("refused", 0) == 0
    dbs = list(tmp_path.rglob(".mace_context_*/materials.db")) + [tmp_path / "materials.db"]
    held = sum(len(held_rows(d)) for d in dbs)
    assert held == 34 - 16
    # parallel drains after 10 jobs left: still never above the budget
    for r in [r for r in fake.records() if not r.get("name", "").startswith("other")][:10]:
        fake.remove_record(r["id"])
    ps = [subprocess.Popen([sys.executable, str(REPO_ROOT / "mace_cli"), "--no-banner", "queue", "drain"]
                           + sum((["--db", str(d)] for d in dbs), []), cwd=str(tmp_path), env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) for _ in range(3)]
    [p.communicate(timeout=600) for p in ps]
    assert fake.load().get("peak_live", 0) <= 20
    assert live(fake) == 20
    assert sum(len(held_rows(d)) for d in dbs) == held - 10


# ------------------------------------------------------------------ 8. drain triggers and the sentinel
def test_a_completion_drains_other_workflows_held_work(tmp_path):
    fake, env = setup(tmp_path, site_text="[queue]\nbudget = 3\n")
    other = make_db(tmp_path / "other" / "materials.db")
    for i in range(3):
        make_held(other, tmp_path / "other", f"H{i}", since=old(30 - i))
    register(env, other)
    # a manual tracked job (no plan) ends: its callback records it and drains
    folder = tmp_path / "manual"
    folder.mkdir()
    (folder / "solo.d12").write_text(SP_D12)
    add_others = [fake.add_record(id=str(800000 + i), state="RUNNING", user="tester", name=f"other{i}")
                  for i in range(2)]
    p = _mace(tmp_path, env, "submit", "--track", str(folder / "solo.d12"))
    assert p.returncode == 0, p.stdout + p.stderr
    db = tmp_path / "materials.db"
    row = next(iter(rows(db).values()))
    from array_helpers import SP_DONE
    (folder / "solo.out").write_text(SP_DONE)
    fake.set_record(row["slurm_job_id"], state="COMPLETED")
    for r in [r for r in fake.records() if r.get("name", "").startswith("other")]:
        fake.remove_record(r["id"])
    p = callback(env, folder, db, mode="completion",
                 extra_env={"SLURM_JOB_ID": row["slurm_job_id"]})
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    assert "Manual submission" in p.stdout
    assert not held_rows(other), p.stdout[-3000:]
    assert live(fake) == 3
    assert not [r for r in fake.records() if r.get("name") == "mace-drain"]     # MACE jobs are queued


def _sentinel_jobs(fake):
    return [r for r in fake.records() if r.get("name") == "mace-drain"]


def test_sentinel_appears_reschedules_and_stops(tmp_path):
    # the queue is full of other jobs: held work, no MACE job queued
    fake, env = setup(tmp_path, others=5, site_text="[queue]\nbudget = 5\nlimit = 10\n")
    db = make_db(tmp_path / "materials.db")
    for i in range(2):
        make_held(db, tmp_path / "work", f"H{i}", since=old(30 - i))
    register(env, db)
    p = drain(tmp_path, env)
    assert p.returncode == 0, p.stdout + p.stderr
    sent = _sentinel_jobs(fake)
    assert len(sent) == 1, p.stdout
    script = sbatch_calls(fake, sentinel=True)[-1]["script_text"]
    assert "--begin=now+15minutes" in script and "queue drain --sentinel" in script
    assert "#SBATCH --time=0:05:00" in script and "#SBATCH --cpus-per-task=1" in script
    # never two
    drain(tmp_path, env)
    assert len(_sentinel_jobs(fake)) == 1
    # mace queue status names it
    assert status_json(tmp_path, env)["sentinel_job"] == sent[0]["id"]
    # the sentinel runs while nothing frees up: it submits its successor
    fake.set_record(sent[0]["id"], state="RUNNING")
    p = _mace(tmp_path, dict(env, SLURM_JOB_ID=sent[0]["id"], SLURM_JOB_NAME="mace-drain"),
              "queue", "drain", "--sentinel")
    fake.remove_record(sent[0]["id"])
    nxt = _sentinel_jobs(fake)
    assert len(nxt) == 1 and nxt[0]["id"] != sent[0]["id"], p.stdout
    # the other jobs end: the next sentinel run releases the work and does not come back
    for r in [r for r in fake.records() if r.get("name", "").startswith("other")]:
        fake.remove_record(r["id"])
    fake.set_record(nxt[0]["id"], state="RUNNING")
    p = _mace(tmp_path, dict(env, SLURM_JOB_ID=nxt[0]["id"], SLURM_JOB_NAME="mace-drain"),
              "queue", "drain", "--sentinel")
    fake.remove_record(nxt[0]["id"])
    assert not held_rows(db)
    assert not _sentinel_jobs(fake), p.stdout


def test_no_sentinel_while_a_mace_job_is_queued_or_when_off(tmp_path):
    fake, env = setup(tmp_path, site_text="[queue]\nbudget = 1\nlimit = 10\n")
    db = make_db(tmp_path / "materials.db")
    make_held(db, tmp_path / "work", "Q0", status="submitted", job_id="5555", since=old(5))
    fake.add_record(id="5555", state="PENDING", user="tester", name="Q0")
    make_held(db, tmp_path / "work", "H0", since=old(4))
    register(env, db)
    drain(tmp_path, env)
    assert not _sentinel_jobs(fake)                  # Q0's completion will release H0
    fake.remove_record("5555")
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE calculations SET status='completed' WHERE calc_id='Q0_OPT'")
    fake.add_record(id="900001", state="RUNNING", user="tester", name="other")
    site(env, tmp_path, "[queue]\nbudget = 1\nlimit = 10\nsentinel = false\n")
    p = drain(tmp_path, env, "--json")
    assert not _sentinel_jobs(fake)
    assert json.loads(p.stdout)["sentinel"]["state"] == "off"


def test_sentinel_is_submitted_when_the_queue_cannot_be_read(tmp_path):
    """squeue and scontrol both fail (a slurmctld timeout) during what may be
    the last MACE job's drain: nothing is released, but the sentinel is
    submitted so the held work does not stall forever; only one per hour."""
    fake, env = setup(tmp_path, site_text="[queue]\nbudget = 5\nlimit = 10\n",
                      state={"fail": {"squeue": "slurm_load_jobs error: Socket timed out",
                                      "scontrol": "slurm_load_assoc_mgr error: Socket timed out"}})
    db = make_db(tmp_path / "materials.db")
    make_held(db, tmp_path / "work", "H0", since=old(5))
    register(env, db)
    rep = json.loads(drain(tmp_path, env, "--json").stdout)
    assert rep["released"] == [] and rep["sentinel"]["state"] == "submitted", rep
    assert len(_sentinel_jobs(fake)) == 1 and not sbatch_calls(fake)
    assert len(held_rows(db)) == 1
    rep = json.loads(drain(tmp_path, env, "--json").stdout)
    assert rep["sentinel"]["state"] == "queued" and len(_sentinel_jobs(fake)) == 1, rep


def test_gui_watcher_tick_drains_at_most_every_two_minutes(monkeypatch, tmp_path):
    sys.path.insert(0, str(REPO_ROOT))
    from mace.gui import workflow as gw
    from mace.queue import budget
    calls = []
    monkeypatch.setattr(budget, "registered_dbs", lambda env=None: ["x.db"])
    monkeypatch.setattr(budget, "drain", lambda **k: calls.append(k) or {"released": []})
    w = gw.WorkflowWatcher(lambda: tmp_path, lambda *a: None, lambda: [])
    w.maybe_drain(now=1000.0)
    w.maybe_drain(now=1050.0)
    w.maybe_drain(now=1121.0)
    assert len(calls) == 2 and calls[0]["timeout"] == 2


# ------------------------------------------------------------------ 9. old rows become held
def test_rows_lost_by_earlier_versions_are_converted_and_released(tmp_path):
    fake, env = setup(tmp_path, site_text="[queue]\nbudget = 10\n")
    db = make_db(tmp_path / "materials.db")
    w = tmp_path / "work"
    make_held(db, w, "P_old", status="pending", created=old(30))                      # phantom: converted
    make_held(db, w, "F_lim", status="failed", created=old(30), error="SLURM submission failed")
    make_held(db, w, "F_empty", status="failed", created=old(30), calc_type="DOSS", step=3)
    make_held(db, w, "P_new", status="pending", created=old(2))                       # too fresh: left
    make_held(db, w, "P_twin", status="pending", created=old(30))                     # queued twin: left
    fake.add_record(id="7777", state="PENDING", user="tester", name="P_twin")
    make_held(db, w, "F_acct", status="failed", created=old(30),
              error="sbatch: error: Batch job submission failed: Invalid account")        # other error: left
    p = drain(tmp_path, env, "--db", str(db), "--json")
    rep = json.loads(p.stdout)
    assert rep["converted"] == 3, p.stdout + p.stderr
    st = {x["material_id"]: x["status"] for x in rows(db).values()}
    assert st == {"P_old": "submitted", "F_lim": "submitted", "F_empty": "submitted", "P_new": "pending",
                  "P_twin": "pending", "F_acct": "failed"}


def test_rows_are_not_converted_when_the_queue_cannot_be_read(tmp_path):
    fake, env = setup(tmp_path, site_text="[queue]\nbudget = 10\n",
                      state={"fail": {"squeue": "Socket timed out", "scontrol": "Socket timed out"}})
    db = make_db(tmp_path / "materials.db")
    make_held(db, tmp_path / "work", "P_old", status="pending", created=old(30))   # its job may be queued
    rep = json.loads(drain(tmp_path, env, "--db", str(db), "--json").stdout)
    assert rep["converted"] == 0 and rows(db)["P_old_OPT"]["status"] == "pending", rep
    assert not sbatch_calls(fake)


def test_drain_exits_nonzero_when_a_release_failed(tmp_path):
    fake, env = setup(tmp_path, site_text="[queue]\nbudget = 10\n")
    db = make_db(tmp_path / "materials.db")
    make_held(db, tmp_path / "work", "Gone", since=old(5))
    shutil.rmtree(tmp_path / "work" / "Gone")                      # its job folder was deleted
    p = drain(tmp_path, env, "--db", str(db))
    assert p.returncode == 1 and "could not be submitted" in p.stdout + p.stderr, p.stdout + p.stderr
    p = drain(tmp_path, env, "--db", str(db), "--json")
    assert p.returncode == 1 and json.loads(p.stdout)["failed"]


# ------------------------------------------------------------------ 10. untracked mace submit
def test_untracked_submit_lists_what_did_not_fit_and_asks_for_the_reserve(tmp_path):
    fake, env = setup(tmp_path, site_text="[queue]\nbudget = 3\nlimit = 10\n")
    root = decks(tmp_path / "decks", 5, text=SP_D12)
    p = _mace(tmp_path, env, "submit", str(root))
    out = p.stdout + p.stderr
    assert len(sbatch_calls(fake)) == 3, out[-3000:]
    assert "Not submitted" in out and "--yes" in out
    assert p.returncode == 1
    for r in fake.records():
        fake.remove_record(r["id"])
    fake2_calls = len(sbatch_calls(fake))
    add = [fake.add_record(id=str(600000 + i), state="RUNNING", user="tester", name="other") for i in range(3)]
    p = _mace(tmp_path, env, "submit", "--yes", str(root))
    out = p.stdout + p.stderr
    assert len(sbatch_calls(fake)) - fake2_calls == 5, out[-3000:]
    assert "from the reserve" in out
    assert live(fake) == 8                   # never past the limit (10)


def test_executor_accepts_queue_limits_above_1000(tmp_path):
    sys.path.insert(0, str(REPO_ROOT))
    from mace.workflow.executor import WorkflowExecutor
    ex = WorkflowExecutor.__new__(WorkflowExecutor)
    plan = {"workflow_sequence": ["OPT"], "input_type": "d12", "input_files": {"d12": []},
            "step_configurations": {}, "execution_settings": {"max_concurrent_jobs": None},
            "queue_management": {"max_jobs": 1040, "reserve_slots": 100}}
    src = Path(REPO_ROOT / "mace" / "workflow" / "executor.py").read_text()
    assert "must be 1-1000" not in src
    assert "max_jobs > 1000" not in src


def test_every_sbatch_site_goes_through_the_budget():
    """No MACE code calls sbatch for a calculation except through
    mace/queue/budget.py (sbatch, run_generator) or the array submission
    (job_array.submit_prepared, which takes its slots from a budget session)."""
    import re
    root = REPO_ROOT / "mace"
    direct = re.compile(r"""subprocess\.run\(\s*\[\s*['"]sbatch['"]""")
    allowed = {"queue/budget.py", "gui/slurmsession.py", "cluster/fit.py", "recovery/slurm_limits.py"}
    bad = []
    for f in sorted(root.rglob("*.py")):
        rel = f.relative_to(root).as_posix()
        if rel in allowed:
            continue
        text = f.read_text(errors="replace")
        for m in direct.finditer(text):
            bad.append(f"{rel}:{text[:m.start()].count(chr(10)) + 1}")
    assert not bad, bad


def test_the_ending_job_does_not_count(tmp_path):
    """A completion callback (or the sentinel) runs as its job's last step:
    that job's slot counts as free (on HPCC the first callback of a full
    budget released nothing until the next one)."""
    fake, env = setup(tmp_path, site_text="[queue]\nbudget = 1\nlimit = 10\n")
    db = make_db(tmp_path / "materials.db")
    make_held(db, tmp_path / "work", "H0", since=old(5))
    register(env, db)
    fake.add_record(id="777", state="RUNNING", user="tester", name="ending")
    drain(tmp_path, env, "--json")
    assert len(held_rows(db)) == 1
    p = _mace(tmp_path, dict(env, SLURM_JOB_ID="777"), "queue", "drain", "--sentinel", "--json")
    assert len(json.loads(p.stdout)["released"]) == 1, p.stdout + p.stderr
    assert not held_rows(db)


def test_jobs_in_hidden_partitions_count():
    src = (REPO_ROOT / "mace" / "queue" / "budget.py").read_text()
    assert '["squeue", "-a", "-u", user, "-h", "-r"' in src


@pytest.mark.parametrize("args", [(), ("--status",), ("--max-jobs", "5")])
def test_mace_queue_other_than_drain_or_status_prints_help(tmp_path, args):
    """1.1.5's `mace queue` alias of `mace manager` is gone: a bare `mace
    queue` ran the manager's full cycle, which submits the decks under the
    current folder and creates materials.db there."""
    fake, env = setup(tmp_path, site_text="[queue]\nbudget = 5\n")
    work = decks(tmp_path / "q", 2)
    p = _mace(tmp_path, env, "queue", *args, cwd=work)
    assert p.returncode == 2, p.stdout + p.stderr
    assert "mace queue drain" in p.stdout and "mace manager" in p.stdout
    assert not fake.calls("sbatch")
    assert not (work / "materials.db").exists() and not (work / ".queue_locks").exists()


def test_a_row_canceled_during_a_drain_is_not_submitted_or_held_again(tmp_path, monkeypatch):
    """mace workflow --cancel while a drain runs: the drain read its
    candidates before the cancel; release() reads the row again under the
    lock (which cancel_held takes too) and skips it, and hold_row never turns
    a canceled or completed row back into a held one."""
    fake, env = setup(tmp_path, site_text="[queue]\nbudget = 5\nlimit = 10\n")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    sys.path.insert(0, str(REPO_ROOT))
    from mace.queue import budget
    db = make_db(tmp_path / "materials.db")
    a = make_held(db, tmp_path / "work", "A0", since=old(5), workflow_id="W1")
    make_held(db, tmp_path / "work", "B0", since=old(4), workflow_id="W2")
    real = budget.held_rows

    def read_then_cancel(p):
        rows_ = real(p)
        budget.cancel_held(p, "W1")              # re-entrant here; another process waits for the lock
        return rows_
    monkeypatch.setattr(budget, "held_rows", read_then_cancel)
    rep = budget.drain(env=dict(env), dbs=[str(db)], quiet=True, sentinel=False, migrate=False)
    monkeypatch.setattr(budget, "held_rows", real)
    assert [r["material_id"] for r in rep["released"]] == ["B0"] and not rep["failed"], rep
    assert [c["argv"][-1] for c in sbatch_calls(fake)] == ["B0.sh"]
    assert rows(db)[a]["status"] == "cancelled"  # british-ok: stored database value
    # a SLURM limit refusal arriving now must not revive it
    budget.hold_row(budget.RowRef(str(db), a), "slurm_limit", ["sbatch", "A0.sh"], tmp_path)
    assert rows(db)[a]["status"] == "cancelled"  # british-ok: stored database value
    # cancel_held takes the budget lock
    seen = []
    monkeypatch.setattr(budget._LOCK, "acquire", lambda t, env=None: seen.append(t) or False)
    budget.cancel_held(str(db), "W2")
    assert seen


def test_manager_reserve_without_max_jobs_warns_that_it_is_ignored(tmp_path):
    fake, env = setup(tmp_path, site_text="[queue]\nbudget = 5\n")
    p = _mace(tmp_path, env, "manager", "--status", "--reserve", "20")
    assert "--reserve is ignored without --max-jobs" in p.stdout + p.stderr and "[queue] reserve" in p.stdout + p.stderr
    p = subprocess.run([sys.executable, str(REPO_ROOT / "mace" / "queue" / "manager.py"), "--status",
                        "--reserve", "20"], cwd=str(tmp_path), env=dict(env, PYTHONPATH=str(REPO_ROOT)),
                       capture_output=True, text=True, timeout=300)
    assert "--reserve is ignored without --max-jobs" in p.stderr, p.stdout + p.stderr


def test_run_generator_replays_sbatch_calls_in_order(tmp_path, monkeypatch):
    """The shim's files were call.<pid>.<ns>: sorted by pid text first, so
    two submissions could be replayed in reverse order."""
    sys.path.insert(0, str(REPO_ROOT))
    from mace.queue import budget
    seen = []
    monkeypatch.setattr(budget, "sbatch", lambda argv, cwd, **k: seen.append(argv[-1]) or
                        budget.Outcome("submitted", job_id=str(len(seen))))
    script = "; ".join(f"sbatch {n}.sh" for n in "abcde")
    p = budget.run_generator(lambda env: subprocess.run(["sh", "-c", script], env=env, cwd=str(tmp_path),
                                                        capture_output=True, text=True), tmp_path)
    assert seen == [f"{n}.sh" for n in "abcde"], seen
    assert p.stdout.count("Submitted batch job") == 5


def test_an_unreadable_squeue_is_an_unreadable_queue_even_with_assoc_mgr(tmp_path):
    """squeue fails but scontrol's assoc_mgr answers: assoc_mgr gives a count,
    not the jobs, so a 1.1.5 pending row whose job is queued was converted and
    submitted a second time. The queue counts as unreadable: nothing converted,
    nothing released."""
    fake, env = setup(tmp_path, site_text="[queue]\nbudget = 10\nsentinel = false\n", assoc=50,
                      state={"fail": {"squeue": "slurm_load_jobs error: Socket timed out"}})
    db = make_db(tmp_path / "materials.db")
    make_held(db, tmp_path / "work", "P_twin", status="pending", created=old(30))
    make_held(db, tmp_path / "work", "H1", since=old(5))
    fake.add_record(id="7777", state="PENDING", user="tester", name="P_twin")
    rep = json.loads(drain(tmp_path, env, "--db", str(db), "--json").stdout)
    assert rep["converted"] == 0 and rep["released"] == [], rep
    assert not sbatch_calls(fake)
    assert rows(db)["P_twin_OPT"]["status"] == "pending"
    assert rows(db)["H1_OPT"]["status"] == "held"


def test_an_unreadable_queue_does_not_trust_a_sentinel_marker_that_has_run(tmp_path):
    """The marker names a sentinel from 50 minutes ago: it began after 15 and
    ran at most 5, so it is gone. With the queue unreadable the last MACE
    job's drain arms a new sentinel instead of waiting on the old one."""
    fake, env = setup(tmp_path, site_text="[queue]\nbudget = 5\nlimit = 10\n",
                      state={"fail": {"squeue": "slurm_load_jobs error: Socket timed out",
                                      "scontrol": "slurm_load_assoc_mgr error: Socket timed out"}})
    db = make_db(tmp_path / "materials.db")
    make_held(db, tmp_path / "work", "H0", since=old(60))
    register(env, db)
    qs = Path(env["MACE_QUEUE_STATE_DIR"])
    qs.mkdir(exist_ok=True)
    (qs / "sentinel.json").write_text(json.dumps({"job_id": "4444", "t": time.time() - 3000}))
    rep = json.loads(drain(tmp_path, env, "--json").stdout)
    assert rep["sentinel"]["state"] == "submitted", rep
    (qs / "sentinel.json").write_text(json.dumps({"job_id": "4445", "t": time.time() - 300}))
    rep = json.loads(drain(tmp_path, env, "--json").stdout)
    assert rep["sentinel"] == {"state": "queued", "job_id": "4445",
                               "note": "the queue could not be read"}, rep


def test_a_refused_array_teaches_no_limit(tmp_path):
    """SLURM refuses an array of 15 at 5 queued (MaxSubmit 12): the limit is
    anywhere up to 19, so nothing is learned from it (it was learned as 5, and
    the drain then released 4 where 12 fit); a refused single job teaches it."""
    fake, env = setup(tmp_path, max_submit=12, others=5, site_text="[queue]\nbudget = 20\n")
    root = decks(tmp_path / "decks", 30)
    p = _mace(tmp_path, env, "submit", "--array", "--progress", "opt_sp", str(root))
    assert p.returncode == 0, p.stdout + p.stderr
    obs = Path(env["MACE_QUEUE_STATE_DIR"]) / "observed_limit.json"
    assert not obs.exists(), obs.read_text()
    from queue_budget_helpers import free_others
    free_others(fake, 5)
    db = next(tmp_path.rglob(".mace_context_*/materials.db"))
    rep = json.loads(drain(tmp_path, env, "--db", str(db), "--json").stdout)
    assert len(rep["released"]) == 12 and fake.load().get("peak_live") == 12, rep
    assert json.loads(obs.read_text())["limit"] == 12


def test_a_release_interrupted_after_sbatch_is_not_submitted_twice(tmp_path):
    """A drain killed between sbatch accepting a held row's job and recording
    it (a callback at its walltime) left the row held, and the next drain
    submitted it again. The row is marked before its sbatch; the next drain
    finds its job queued and records that job instead."""
    fake, env = setup(tmp_path, site_text="[queue]\nbudget = 10\nlimit = 20\n")
    db = make_db(tmp_path / "materials.db")
    make_held(db, tmp_path / "work", "K1", since=old(30))
    make_held(db, tmp_path / "work", "K2", since=old(29))
    register(env, db)
    with sqlite3.connect(db) as conn:
        st = json.loads(conn.execute("SELECT settings_json FROM calculations WHERE calc_id='K1_OPT'").fetchone()[0])
        st["held_releasing"] = "2026-10-17T12:00:00"
        conn.execute("UPDATE calculations SET settings_json=? WHERE calc_id='K1_OPT'", (json.dumps(st),))
    fake.add_record(id="7001", state="PENDING", user="tester", name="K1")   # K1's job, accepted
    fake.add_record(id="7002", state="PENDING", user="tester", name="K2")   # another K2: not ours
    rep = json.loads(drain(tmp_path, env, "--json").stdout)
    assert [c["argv"][-1] for c in sbatch_calls(fake)] == ["K2.sh"], rep
    r = rows(db)
    assert r["K1_OPT"]["status"] == "submitted" and r["K1_OPT"]["slurm_job_id"] == "7001"
    assert r["K2_OPT"]["status"] == "submitted" and r["K2_OPT"]["slurm_job_id"] != "7002"
    assert "held_releasing" not in json.loads(r["K2_OPT"]["settings_json"])
