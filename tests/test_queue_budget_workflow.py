"""The queue budget through whole workflows: real CRYSTAL outputs from the
test/ corpus (skipped without it), the real mace_cli launch and the real job
callback (mace/queue/manager.py --callback-mode completion) as subprocesses,
the fake SLURM refusing past its MaxSubmitJobs."""
import json
from pathlib import Path

import pytest

from conftest import find_data
from queue_budget_helpers import (_mace, add_others, by_type, complete, decks, drain, held_rows, live,
                                  launch_spec, plan_spec, rows, sbatch_calls, setup)


def outputs():
    opt = find_data("OPT/1_dia_opt_rev1.out")
    d12 = find_data("OPT/1_dia_opt_rev1.d12")
    sp = Path(str(find_data("SP/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized.out")))
    return opt, d12, sp, sp.with_suffix(".f9")


def wf(tmp_path, n, sequence, budget, max_submit=None, others=0, execution=None, extra_site=""):
    opt, d12, sp, f9 = outputs()
    fake, env = setup(tmp_path, max_submit=max_submit, others=others,
                      site_text=f"[queue]\nbudget = {budget}\n{extra_site}")
    root = tmp_path / "wf"
    decks(root / "decks", n, text=d12.read_text())
    w = plan_spec(root, env, sequence, execution=execution)
    p, db = launch_spec(root, env, w)
    assert p.returncode in (0, 1), p.stdout[-3000:] + p.stderr[-3000:]
    return fake, env, root, w, db


def test_fan_out_admits_band_and_doss_one_by_one(tmp_path):
    """SP completion with one free slot: BAND is submitted, DOSS held
    (reason budget), never failed; the next free slot releases DOSS."""
    opt, d12, sp, f9 = outputs()
    fake, env, root, w, db = wf(tmp_path, 1, ["OPT", "SP", "BAND", "DOSS"], budget=4, max_submit=4)
    o = by_type(db, "OPT", "submitted")[0]
    p = complete(fake, env, db, o, opt)
    assert p.returncode == 0, p.stdout[-3000:]
    s = by_type(db, "SP", "submitted")[0]
    add_others(fake, 3)                 # 3 others: one slot left once the SP has ended
    p = complete(fake, env, db, s, sp, f9=f9, remove=False)
    assert p.returncode == 0, p.stdout[-4000:]
    fan = by_type(db, "BAND") + by_type(db, "DOSS")
    assert sorted(r["status"] for r in fan) == ["held", "submitted"], p.stdout[-4000:]
    held = [r for r in fan if r["status"] == "held"][0]
    assert json.loads(held["settings_json"])["held_reason"] == "budget"
    assert not [r for r in rows(db).values() if r["status"] == "failed"]
    assert fake.load().get("peak_live", 0) <= 4 and fake.load().get("refused", 0) == 0
    # one of the other jobs ends: the next drain submits the held one
    fake.remove_record(next(r["id"] for r in fake.records() if r.get("name", "").startswith("other_x")))
    drain(tmp_path, env)
    assert sorted(r["status"] for r in by_type(db, "BAND") + by_type(db, "DOSS")) == ["submitted", "submitted"]


def test_hard_workflow_cap_has_no_fan_out_overshoot(tmp_path):
    """The S4 setup (cap 3, SP completion at 2 live): BAND+DOSS no longer
    overshoot the cap; the second one waits held (workflow_cap)."""
    opt, d12, sp, f9 = outputs()
    fake, env, root, w, db = wf(tmp_path, 3, ["OPT", "SP", "BAND", "DOSS"], budget=50,
                                execution={"max_jobs": 3})
    opts = by_type(db, "OPT", "submitted")
    assert len(opts) == 3
    complete(fake, env, db, opts[0], opt)
    s = by_type(db, "SP", "submitted")[0]
    p = complete(fake, env, db, s, sp, f9=f9)
    live_rows = [r for r in rows(db).values() if r["status"] in ("submitted", "running", "pending", "queued")]
    assert len(live_rows) <= 3, p.stdout[-3000:]
    held = held_rows(db)
    assert held and {json.loads(h["settings_json"])["held_reason"] for h in held} == {"workflow_cap"}


def test_s6_phantom_deadlock_now_completes(tmp_path):
    """10 decks, max_jobs 4, SLURM's limit 4: six first steps are held (not
    phantom 'pending'), the cap counts only real jobs, and every material
    finishes OPT and SP."""
    opt, d12, sp, f9 = outputs()
    fake, env, root, w, db = wf(tmp_path, 10, ["OPT", "SP"], budget=50, max_submit=4,
                                execution={"max_jobs": 4})
    assert len(by_type(db, "OPT", "submitted")) == 4
    assert len(held_rows(db)) == 6
    for _ in range(40):
        todo = [r for r in rows(db).values() if r["status"] == "submitted"]
        if not todo:
            break
        r = todo[0]
        complete(fake, env, db, r, opt if r["calc_type"] == "OPT" else sp,
                 f9=None if r["calc_type"] == "OPT" else f9)
        assert fake.load().get("peak_live", 0) <= 4
    st = {(r["calc_type"], r["status"]) for r in rows(db).values()}
    assert st == {("OPT", "completed"), ("SP", "completed")}, rows(db)
    assert len(by_type(db, "SP", "completed")) == 10
    assert fake.load().get("refused", 0) == 0


def test_freed_slot_goes_to_the_oldest_finished_material(tmp_path):
    """Completions are progressed oldest first (the old ORDER BY created_at
    DESC gave the newest the slot): m000 and m001 finish while the budget is
    full, then one slot frees: m000's SP is submitted first."""
    opt, d12, sp, f9 = outputs()
    fake, env, root, w, db = wf(tmp_path, 3, ["OPT", "SP"], budget=3)
    opts = by_type(db, "OPT", "submitted")
    add_others(fake, 2)                     # every finished OPT's slot is taken at once
    complete(fake, env, db, opts[0], opt)
    complete(fake, env, db, opts[1], opt)
    assert [r["status"] for r in by_type(db, "SP")] == ["held", "held"]
    for r in [r for r in fake.records() if r.get("name", "").startswith("other_x")][:1]:
        fake.remove_record(r["id"])
    drain(tmp_path, env)
    sps = {r["material_id"]: r["status"] for r in by_type(db, "SP")}
    assert sps == {"m000_dia": "submitted", "m001_dia": "held"}, sps


def test_small_end_to_end_never_exceeds_the_budget(tmp_path):
    """12 materials OPT -> SP -> BAND + DOSS, budget 6 under SLURM's 8, two
    other jobs: everything finishes, nothing is lost or refused, the queue
    never holds more than 6 + 2... of which MACE never more than the budget."""
    opt, d12, sp, f9 = outputs()
    fake, env, root, w, db = wf(tmp_path, 12, ["OPT", "SP", "BAND", "DOSS"], budget=6, max_submit=8, others=2)
    for _ in range(200):
        todo = sorted((r for r in rows(db).values() if r["status"] == "submitted"),
                      key=lambda r: r["submitted_at"] or "")
        if not todo:
            break
        r = todo[0]
        out, f = {"OPT": (opt, None), "SP": (sp, f9)}.get(r["calc_type"], (sp, None))
        if r["calc_type"] in ("BAND", "DOSS"):
            # a properties job: the callback only needs it gone with an output
            Path(r["input_file"]).with_suffix(".out").write_text(sp.read_text())
        complete(fake, env, db, r, out, f9=f)
        assert fake.load().get("peak_live", 0) <= 6
    final = {}
    for r in rows(db).values():
        final[(r["calc_type"], r["status"])] = final.get((r["calc_type"], r["status"]), 0) + 1
    assert final.get(("OPT", "completed")) == 12 and final.get(("SP", "completed")) == 12, final
    assert not held_rows(db)
    assert fake.load().get("refused", 0) == 0
