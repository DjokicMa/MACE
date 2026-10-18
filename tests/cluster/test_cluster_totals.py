"""Totals beyond one job (mace/cluster/totals.py, 1.1.6): SLURM's queue-wait
estimate, the rest of a job array, the rest of a workflow. No array or
workflow history exists in the corpus, so these are checked on constructed
cases; every total is labeled rough and says what it is based on."""
import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path

from mace.cluster import totals as T

NOW = 1_800_000_000.0


def _iso(t):
    return datetime.fromtimestamp(t).strftime("%Y-%m-%dT%H:%M:%S")


def test_queue_wait_is_slurms_estimate():
    q = T.queue_wait({"state": "PENDING", "start": _iso(NOW + 7200), "reason": "Priority"}, NOW)
    assert q["low_s"] == 7200 and q["high_s"] is None
    assert "queue wait estimate from SLURM" in q["text"] and q["text"].startswith("Queue wait: about 2 h 0 min ")
    q = T.queue_wait({"state": "PENDING", "start": "N/A", "reason": "Priority"}, NOW)
    assert q["low_s"] is None and "no start estimate yet (reason: Priority)" in q["text"]
    assert T.queue_wait({"state": "RUNNING", "start": _iso(NOW)}, NOW) is None


def _rows(spec):
    return [{"id": f"77_{t}", "array_job": "77", "array_task": str(t), "state": st, "elapsed": el,
             "start": _iso(NOW + s) if s is not None else "N/A"} for t, st, el, s in spec]


def test_array_total_from_finished_tasks():
    acct = {"77_1": {"State": "COMPLETED", "Elapsed": "01:00:00"}, "77_1.batch": {"State": "COMPLETED", "Elapsed": "01:00:00"},
            "77_2": {"State": "COMPLETED", "Elapsed": "02:00:00"}, "88_1": {"State": "COMPLETED", "Elapsed": "09:00:00"}}
    rows = _rows([(3, "RUNNING", "0:30:00", None), (4, "PENDING", "0:00", 3600)])
    a = T.array_total("77", rows, acct, NOW)
    # the pending task starts in 1 h and takes 1-2 h; the running one has 0.5-1.5 h left
    assert (a["low_s"], a["high_s"]) == (2 * 3600, 3 * 3600)
    assert a["text"] == "Array 77: 1 running, 1 pending, 2 finished; about 2 h 0 min - 3 h 0 min until the last task ends (rough)"
    assert "the 2 finished" in a["basis"] and "queue wait estimate from SLURM" in a["basis"]
    # a pending task without a SLURM estimate: "at least", and why
    a = T.array_total("77", _rows([(3, "RUNNING", "0:30:00", None), (4, "PENDING", "0:00", None)]), acct, NOW)
    assert a["high_s"] is None and "1 pending task(s) have no SLURM start estimate yet" in a["text"]


def test_array_total_before_any_task_finished_uses_this_tasks_estimate():
    rows = _rows([(1, "RUNNING", "1:00:00", None), (2, "RUNNING", "1:00:00", None)])
    a = T.array_total("77", rows, {}, NOW, own_total={"low_s": 1800, "high_s": 3600, "elapsed_s": 3600})
    assert (a["low_s"], a["high_s"]) == (1800, 3600) and "this task's own estimate" in a["basis"]
    a = T.array_total("77", rows, {}, NOW)
    assert a["low_s"] is None and a["confidence"] == "unknown" and "unknown until one finishes" in a["text"]


def _workflow_dir(tmp_path, done=("OPT",), me="SP", seq=("OPT", "SP", "BAND", "DOSS")):
    d = tmp_path / "wf"
    (d / "workflow_configs").mkdir(parents=True)
    (d / "workflow_configs" / "workflow_plan_20261014_1.json").write_text(json.dumps({"workflow_sequence": list(seq)}))
    con = sqlite3.connect(d / "materials.db")
    con.execute("CREATE TABLE calculations (calc_id TEXT, material_id TEXT, calc_type TEXT, status TEXT, "
                "slurm_job_id TEXT, settings_json TEXT)")
    s = json.dumps({"workflow_id": "workflow_20261014_1"})
    for i, t in enumerate(done):
        con.execute("INSERT INTO calculations VALUES (?,?,?,?,?,?)", (f"c{i}", "m1", t, "completed", f"90{i}", s))
    con.execute("INSERT INTO calculations VALUES (?,?,?,?,?,?)", ("cme", "m1", me, "running", "4242", s))
    con.commit()
    con.close()
    return d


def _records():
    # finished runs the history model scales from (same family, sizes near this one)
    out = []
    for ct, wall in (("BAND", 600.0), ("DOSS", 900.0)):
        for k in range(4):
            out.append({"file": f"/x/{ct}{k}.out", "stem": f"{ct}{k}", "calc_type": ct, "family": "hybrid",
                        "nao": 100, "nk": 10, "atoms": 4, "cores": 32, "cycles": 10, "wall_s": wall * (1 + 0.1 * k),
                        "group": "unknown"})
    return out


def test_workflow_total_steps_after_this_job(tmp_path):
    d = _workflow_dir(tmp_path)
    out = d / "m1_sp.out"
    out.write_text(" N. OF ATOMS PER CELL        4\n NUMBER OF AO  100\n NUMBER OF K POINTS IN THE IBZ   10\n")
    (d / "m1_sp.d12").write_text("x\nCRYSTAL\nEND\nDFT\nB3LYP\nENDDFT\n")
    w = T.workflow_total(d, "4242", out, 32, [], NOW, own_total={"low_s": 600, "high_s": 1200}, records=_records())
    assert [s["step"] for s in w["steps"]] == ["BAND", "DOSS"]
    assert w["low_s"] > 600 and w["high_s"] > w["low_s"]
    assert w["text"].startswith("Workflow workflow_20261014_1: 2 step(s) after this one (BAND, DOSS); about ")
    assert "of run time in total (rough); plus each later step's queue wait (not estimated" in w["text"]
    assert "history model" in w["basis"]
    # no similar finished runs: the steps are named and the time is unknown
    w = T.workflow_total(d, "4242", out, 32, [], NOW, own_total={"low_s": 600, "high_s": 1200}, records=[])
    assert "(BAND, DOSS)" in w["text"] and w["high_s"] is None
    assert "at least 10 min of run time (BAND, DOSS without an estimate)" in w["text"]
    # an untracked job, or one without a plan: no workflow total
    assert T.workflow_total(d, "9999", out, 32, [], NOW, records=[]) is None


def test_workflow_last_step(tmp_path):
    d = _workflow_dir(tmp_path, done=("OPT", "SP", "BAND"), me="DOSS")
    w = T.workflow_total(d, "4242", None, 32, [], NOW, own_total={"low_s": 60, "high_s": 120}, records=[])
    assert w["steps"] == [] and "this is its last step (OPT > SP > BAND > DOSS)" in w["text"]
