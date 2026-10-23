"""Time left beyond one job: its queue wait, the rest of its job array and
the rest of its workflow (1.1.6). ``jobinfo.job()`` adds them to the Job
document and ``mace cluster job``; ``mace monitor --status`` reads the same.

* **Queue wait**: SLURM's own start estimate for a pending job (squeue's
  ``%S``, what ``squeue --start`` prints), labeled "queue wait estimate from
  SLURM". SLURM gives one only when its backfill scheduler has placed the job;
  otherwise the text says so. MACE does not model the queue itself.
* **Array**: the tasks not finished yet. Time per task = [min, max] of the
  array's finished tasks (sacct Elapsed); before any finished, the running
  task's own progress total. A pending task ends at its SLURM start estimate
  plus a task's time; the array ends when its last task does.
* **Workflow**: the steps of the job's workflow plan
  (``workflow_configs/workflow_plan_<id>.json``) not done yet, each from the
  size-aware history model (``history.estimate_from``: the nearest finished runs
  of that calc type by AOs, k points and atoms, scaled to this system); queue
  waits of steps not submitted yet are not estimated and the text says so.

Every total says what it is based on; anything missing makes it "at least"
or "unknown", never a guess. All of it is "rough": the corpus holds no array
or workflow histories to backtest against.
"""
from __future__ import annotations

import collections
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from mace.cluster import progress as progress_mod
from mace.recovery.slurm_limits import parse_walltime

_fmt = progress_mod._fmt_dur
QUEUE_LABEL = "queue wait estimate from SLURM"


def _about(lo: Optional[float], hi: Optional[float]) -> str:
    if lo is None or hi is None:
        return f"about {_span(lo, hi)}"
    return progress_mod._about(lo, hi)


def _span(lo: Optional[float], hi: Optional[float]) -> str:
    if lo is None:
        return "unknown"
    if hi is None:
        return f"at least {_fmt(lo)}"
    return progress_mod._span(lo, hi)


def slurm_start(value: Optional[str]) -> Optional[float]:
    """squeue %S (ISO local time) as an epoch; None for N/A, Unknown or junk."""
    if not value or value in ("N/A", "Unknown", "None", "(null)"):
        return None
    try:
        return datetime.fromisoformat(value.strip()).timestamp()
    except ValueError:
        return None


def queue_wait(row: Optional[Dict[str, Any]], now: float) -> Optional[Dict[str, Any]]:
    """For a PENDING job: SLURM's start estimate as a wait from now."""
    if not row or (row.get("state") or "").upper() != "PENDING":
        return None
    t = slurm_start(row.get("start"))
    if t is None:
        return {"id": "queue", "label": "Queue wait", "low_s": None, "high_s": None, "confidence": "unknown",
                "text": f"Queue wait: SLURM has no start estimate yet (reason: {row.get('reason') or 'none given'})",
                "basis": f"{QUEUE_LABEL}: none yet (squeue --start shows N/A until the scheduler places the job)"}
    wait = max(0.0, t - now)
    when = datetime.fromtimestamp(t).strftime("%a %H:%M")
    return {"id": "queue", "label": "Queue wait", "low_s": wait, "high_s": None, "start_epoch": t,
            "confidence": "rough",
            "text": f"Queue wait: about {_fmt(wait)} (starts around {when}, {QUEUE_LABEL})",
            "basis": f"{QUEUE_LABEL} (squeue --start); it moves as other jobs end early or late"}


def _elapsed(row: Dict[str, Any]) -> Optional[float]:
    v = row.get("Elapsed") or row.get("elapsed")
    return float(parse_walltime(v)) if v else None


def array_total(array_job: str, squeue_rows: Sequence[Dict[str, Any]], acct_rows: Dict[str, Dict[str, Any]],
                now: float, own_total: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Time until the array's last task ends (see the module doc)."""
    tasks = [r for r in squeue_rows if str(r.get("array_job") or "") == array_job
             and (r.get("array_task") or "N/A") not in ("N/A", "")]
    done = [_elapsed(r) for k, r in acct_rows.items()
            if k.startswith(array_job + "_") and "." not in k and (r.get("State") or "").startswith("COMPLETED")]
    done = [d for d in done if d]
    pend = [r for r in tasks if (r.get("state") or "").upper() == "PENDING"]
    run = [r for r in tasks if (r.get("state") or "").upper() == "RUNNING"]
    if not tasks:
        return None
    if done:
        lo_t, hi_t = min(done), max(done)
        per = f"{_span(lo_t, hi_t)} per task (the {len(done)} finished)"
    elif own_total and own_total.get("elapsed_s") is not None and own_total.get("low_s") is not None:
        lo_t = own_total["elapsed_s"] + own_total["low_s"]
        hi_t = own_total["elapsed_s"] + own_total["high_s"] if own_total.get("high_s") is not None else None
        per = f"{_span(lo_t, hi_t)} per task (this task's own estimate; no task has finished yet)"
    else:
        lo_t = hi_t = None
        per = "time per task unknown until one finishes"
    ends_lo: List[float] = []
    ends_hi: List[Optional[float]] = []
    no_start = 0
    for r in run:
        e = _elapsed(r) or 0.0
        if lo_t is not None:
            ends_lo.append(max(0.0, lo_t - e))
            ends_hi.append(max(0.0, hi_t - e) if hi_t is not None and hi_t > e else None)
    for r in pend:
        t = slurm_start(r.get("start"))
        if t is None:
            no_start += 1
            continue
        if lo_t is not None:
            ends_lo.append(max(0.0, t - now) + lo_t)
            ends_hi.append(max(0.0, t - now) + hi_t if hi_t is not None else None)
    lo = max(ends_lo) if ends_lo else None
    hi = max(ends_hi) if ends_hi and all(h is not None for h in ends_hi) and not no_start else None
    counts = f"{len(run)} running, {len(pend)} pending, {len(done)} finished"
    if lo is None:
        text = f"Array {array_job}: {counts}; time left unknown ({per})"
    elif hi is None:
        why = (f"{no_start} pending task(s) have no SLURM start estimate yet" if no_start
               else "a running task is past the longest finished one")
        text = f"Array {array_job}: {counts}; at least {_fmt(lo)} until the last task ends ({why})"
    else:
        text = f"Array {array_job}: {counts}; {_about(lo, hi)} until the last task ends (rough)"
    return {"id": "array", "label": f"Array {array_job}", "low_s": lo, "high_s": hi,
            "confidence": "rough" if lo is not None else "unknown", "text": text,
            "basis": (f"{per}; a pending task starts at its {QUEUE_LABEL}" if pend else per)}


# ------------------------------------------------------------------ workflow
def _plan_sequence(job_dir: Path, workflow_id: str) -> Optional[List[str]]:
    name = f"workflow_plan_{workflow_id.replace('workflow_', '')}.json"
    d = job_dir
    for _ in range(6):
        cand = d / "workflow_configs" / name
        if cand.is_file():
            try:
                seq = json.loads(cand.read_text()).get("workflow_sequence")
                return [str(x) for x in seq] if seq else None
            except (OSError, ValueError):
                return None
        if d.parent == d:
            break
        d = d.parent
    return None


def _db_rows(db: Path, job_id: str) -> Optional[Dict[str, Any]]:
    try:
        conn = sqlite3.connect(progress_mod.ro_uri(db), uri=True, timeout=5)
    except sqlite3.Error:
        return None
    try:
        conn.row_factory = sqlite3.Row
        me = conn.execute("SELECT * FROM calculations WHERE slurm_job_id = ? ORDER BY rowid DESC LIMIT 1",
                          (str(job_id),)).fetchone()
        if me is None:
            return None
        rows = [dict(r) for r in conn.execute("SELECT calc_id, calc_type, status, slurm_job_id, settings_json "
                                              "FROM calculations WHERE material_id = ?", (me["material_id"],))]
        return {"me": dict(me), "rows": rows}
    except sqlite3.Error:
        return None
    finally:
        conn.close()


def _base_type(t: str) -> str:
    t = (t or "").upper()
    return t.rstrip("0123456789") or t


def workflow_total(job_dir: Path, job_id: str, out: Optional[Path], cores: Optional[int],
                   squeue_rows: Sequence[Dict[str, Any]], now: float,
                   own_total: Optional[Dict[str, Any]] = None,
                   records: Optional[Sequence[Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
    """The steps of this job's workflow plan after this job (see the module doc); None when the job is
    not tracked in a materials.db with a workflow plan."""
    db = job_dir / "materials.db"
    if not db.is_file():
        return None
    info = _db_rows(db, job_id)
    if not info:
        return None
    try:
        wf = json.loads(info["me"].get("settings_json") or "{}").get("workflow_id")
    except ValueError:
        wf = None
    seq = _plan_sequence(job_dir, wf) if wf else None
    if not seq:
        return None
    me_type = info["me"]["calc_type"]
    have = collections.Counter(_base_type(r["calc_type"]) for r in info["rows"]
                               if r["status"] == "completed" or r["calc_id"] == info["me"]["calc_id"])
    remaining: List[str] = []
    for step in seq:
        b = _base_type(step)
        if have[b] > 0:
            have[b] -= 1
        else:
            remaining.append(step)
    from mace.cluster import history
    feats = None
    if out is not None and out.is_file():
        try:
            text = out.read_text(errors="replace")
            deck = out.with_suffix(".d12")
            dtext = deck.read_text(errors="replace") if deck.is_file() else ""
            m_at, m_ao, m_nk = history._ATOMS.search(text), history._AO.search(text), history._NK.search(text)
            feats = {"family": history.functional_family(dtext) if dtext else "unknown",
                     "spin": "SPIN" in dtext.upper().split(), "atoms": int(m_at.group(1)) if m_at else None,
                     "nao": int(m_ao.group(1)) if m_ao else None, "nk": int(m_nk.group(1)) if m_nk else None}
        except OSError:
            feats = None
    if records is None:
        try:
            records = history.index(job_dir, use_sacct=False)
        except Exception:
            records = []
    steps = []
    lo_sum, hi_sum, unknown = 0.0, 0.0, []
    for step in remaining:
        est = history.estimate_from({**(feats or {}), "calc_type": _base_type(step)}, records, cores=cores) \
            if feats else {"wall": None, "message": "this run's size is not known yet"}
        w = est.get("wall")
        queued = next((r for r in info["rows"] if _base_type(r["calc_type"]) == _base_type(step)
                       and r["status"] not in ("completed",) and r["calc_id"] != info["me"]["calc_id"]
                       and r.get("slurm_job_id")), None)
        qrow = next((r for r in squeue_rows if queued and r.get("id") == str(queued["slurm_job_id"])), None)
        qw = queue_wait(qrow, now) if qrow else None
        if w:
            lo_sum += w["p10"]
            hi_sum += w["p90"]
        else:
            unknown.append(step)
        if qw and qw.get("low_s") is not None:
            lo_sum += qw["low_s"]
        steps.append({"step": step, "wall": w, "n": est.get("n"), "message": est.get("message"),
                      "queue": qw})
    own_lo = (own_total or {}).get("low_s")
    own_hi = (own_total or {}).get("high_s")
    lo = lo_sum + (own_lo or 0.0)
    hi = None if unknown or own_hi is None and own_total is not None else hi_sum + (own_hi or 0.0)
    names = ", ".join(remaining) if remaining else "none"
    if not remaining:
        text = f"Workflow {wf}: this is its last step ({' > '.join(seq)})"
        return {"id": "workflow", "label": "Workflow", "low_s": own_lo, "high_s": own_hi, "steps": [],
                "confidence": "rough", "text": text, "basis": "no planned step after this job"}
    queue_note = "; plus each later step's queue wait (not estimated: SLURM gives one only once a job is queued)"
    if unknown and not lo:
        text = (f"Workflow {wf}: {len(remaining)} step(s) after this one ({names}); time unknown "
                f"({unknown[0]}: {next(s['message'] for s in steps if s['step'] == unknown[0]) or 'no similar runs'})")
    elif hi is None:
        text = (f"Workflow {wf}: {len(remaining)} step(s) after this one ({names}); at least {_fmt(lo)} of run time"
                + (f" ({', '.join(unknown)} without an estimate)" if unknown else "") + queue_note)
    else:
        text = (f"Workflow {wf}: {len(remaining)} step(s) after this one ({names}); {_about(lo, hi)} of run "
                f"time in total (rough){queue_note}")
    basis = ("each step from the history model (the nearest finished runs of that calc type, scaled to this "
             "system's AOs and k points; 10-90 % of their predictions) plus this job's own estimate")
    return {"id": "workflow", "label": "Workflow", "low_s": lo, "high_s": hi, "steps": steps,
            "confidence": "rough" if lo else "unknown", "text": text, "basis": basis, "sequence": seq}


def totals(res: Dict[str, Any], row: Optional[Dict[str, Any]], squeue_rows: Sequence[Dict[str, Any]],
           acct_rows: Dict[str, Dict[str, Any]], now: float, job_dir: Optional[Path] = None,
           out: Optional[Path] = None) -> List[Dict[str, Any]]:
    """The queue, array and workflow totals for jobinfo.job()'s result ``res``."""
    out_list: List[Dict[str, Any]] = []
    prog = res.get("progress") or {}
    tot = next((lv for lv in prog.get("levels") or [] if lv.get("id") == "total"), None)
    own = None
    if tot is not None:
        own = {"low_s": tot["remaining_s"]["low"], "high_s": tot["remaining_s"]["high"],
               "elapsed_s": res.get("elapsed_s")}
    q = queue_wait(row, now)
    if q:
        out_list.append(q)
    aid = res.get("array_job") or (row or {}).get("array_job")
    if aid and (res.get("array_tasks") or {}).get("total_in_queue"):
        a = array_total(str(aid), squeue_rows, acct_rows, now, own_total=own)
        if a:
            out_list.append(a)
    if job_dir is not None:
        try:
            cores = int(str(res.get("ncpus") or "").split()[0]) if res.get("ncpus") else None
        except ValueError:
            cores = None
        try:
            w = workflow_total(job_dir, res["id"], out, cores, squeue_rows, now, own_total=own)
        except Exception as exc:        # a broken plan or DB must not take the job view down
            w = {"id": "workflow", "label": "Workflow", "low_s": None, "high_s": None, "confidence": "unknown",
                 "text": f"Workflow: no estimate ({type(exc).__name__})", "basis": str(exc)}
        if w:
            out_list.append(w)
    return out_list
