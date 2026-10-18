"""The queue budget at full scale (opt-in: MACE_QUEUE_FULLSCALE=1; about ten
minutes): 1000 materials OPT -> SP -> BAND + DOSS against SLURM's per-user
limit of 1040 (the fake refuses past it), 40 jobs of the user that are not
MACE's, budget 1040 - 100 = 940, at most 300 of the user's jobs running at
once (SLURM starts pending jobs oldest first).

The real mace.queue.budget code does every submission, with the fake SLURM's
own squeue and sbatch: each completion holds the job's next steps (as the
completion callback does) and drains, exactly the callback's sequence. The
callback itself runs as a subprocess in tests/test_queue_budget_workflow.py;
here it is driven in process so 4000 jobs finish in minutes. It must end
with every material finished, the queue never above the budget, and no job
refused or lost."""
import json
import os
import sqlite3
import time

import pytest

from queue_budget_helpers import make_db, make_held, setup

pytestmark = pytest.mark.skipif(os.environ.get("MACE_QUEUE_FULLSCALE") != "1",
                                reason="full-scale queue simulation: set MACE_QUEUE_FULLSCALE=1")

N, LIMIT, OTHERS, RUNNING = 1000, 1040, 40, 300
NEXT = {"OPT": ["SP"], "SP": ["BAND", "DOSS"], "BAND": [], "DOSS": []}
STEP = {"OPT": 1, "SP": 2, "BAND": 3, "DOSS": 4}


def test_thousand_materials_against_1040(tmp_path, monkeypatch):
    fake, env = setup(tmp_path, max_submit=LIMIT, others=OTHERS, site_text=f"[queue]\nlimit = {LIMIT}\n")
    for k in ("PATH", "FAKE_SLURM_STATE", "FAKE_SLURM_LOG", "MACE_QUEUE_STATE_DIR", "MACE_SITE_CONFIG",
              "HOME", "USER", "XDG_CACHE_HOME"):
        monkeypatch.setenv(k, env[k])
    import sys
    from conftest import REPO_ROOT
    sys.path.insert(0, str(REPO_ROOT))
    from mace.queue import budget
    budget.forget_resolved()
    fake.update(sbatch_delay=0)
    db = make_db(tmp_path / "materials.db")
    work = tmp_path / "work"
    wid = "workflow_fullscale"

    def new_row(mid, ctype):
        cid = make_held(db, work, f"{mid}_{ctype}", step=STEP[ctype], workflow_id=wid, calc_type=ctype)
        with sqlite3.connect(db) as conn:
            conn.execute("UPDATE calculations SET material_id=? WHERE calc_id=?", (mid, cid))
        return cid

    # the first step: every OPT is offered; the budget admits what fits
    t0 = time.time()
    with budget.deferred():
        for i in range(N):
            new_row(f"m{i:04d}", "OPT")
    budget.register(db)
    rep = budget.drain(dbs=[str(db)], out=lambda *_: None, quiet=True)
    assert len(rep["released"]) == LIMIT - 100 - OTHERS

    def jobs():
        with sqlite3.connect(db) as conn:
            return {jid: (cid, mid, t) for cid, mid, t, jid in conn.execute(
                "SELECT calc_id, material_id, calc_type, slurm_job_id FROM calculations WHERE status='submitted'")}

    events = 0
    while True:
        st = fake.load()
        recs = st.get("records", [])
        mace = [r for r in recs if not r.get("name", "").startswith("other")]
        if not mace:
            break
        # SLURM starts pending jobs oldest first, at most RUNNING of the user's at once
        running = sum(1 for r in recs if r["state"] == "RUNNING")
        for r in sorted((r for r in mace if r["state"] == "PENDING"), key=lambda r: int(r["id"])):
            if running >= RUNNING:
                break
            r["state"] = "RUNNING"
            running += 1
        fake.save(st)
        # the oldest running MACE job ends; its callback holds the next steps, then drains
        done = min((r for r in fake.records() if r["state"] == "RUNNING"
                    and not r.get("name", "").startswith("other")), key=lambda r: int(r["id"]))
        cid, mid, ctype = jobs()[done["id"]]
        fake.remove_record(done["id"])
        with sqlite3.connect(db) as conn:
            conn.execute("UPDATE calculations SET status='completed' WHERE calc_id=?", (cid,))
        with budget.deferred():
            for nxt in NEXT[ctype]:
                nid = new_row(mid, nxt)
                with sqlite3.connect(db) as conn:
                    conn.execute("UPDATE calculations SET status='pending' WHERE calc_id=?", (nid,))
                with budget.row(db, nid):
                    o = budget.sbatch(["sbatch", f"{mid}_{nxt}.sh"], work / f"{mid}_{nxt}")
                assert o.held
        budget.drain(dbs=[str(db)], out=lambda *_: None, quiet=True, sentinel=False)
        events += 1
        assert fake.load().get("peak_live", 0) <= LIMIT - 100, events

    with sqlite3.connect(db) as conn:
        final = dict(conn.execute("SELECT status, COUNT(*) FROM calculations GROUP BY status").fetchall())
        per = conn.execute("SELECT COUNT(DISTINCT material_id) FROM calculations WHERE calc_type IN "
                           "('BAND','DOSS') AND status='completed'").fetchone()[0]
    st = fake.load()
    report = {"events": events, "final": final, "materials_finished": per, "peak_live": st.get("peak_live"),
              "refused": st.get("refused", 0), "seconds": round(time.time() - t0)}
    print(json.dumps(report))
    (tmp_path.parent / "fullscale_report.json").write_text(json.dumps(report))
    assert final == {"completed": 4 * N}, report
    assert per == N
    assert st.get("refused", 0) == 0
    assert st.get("peak_live", 0) <= LIMIT - 100
