"""The ETA column and totals footer of `mace monitor --status` / `mace status`
(1.1.6), from the same functions as `mace cluster job` and the GUI's Job
document: ``progress.progress()`` per output, ``totals`` for the queue wait,
arrays and workflows.

SLURM is asked once for the whole table: the user's cached ``squeue`` rows
(``reader.squeue_own``, shared with the Cluster view, 60 s cache) and, only
when an array is listed, one batched ``sacct`` for its tasks. Each job's
progress is read from its output (memoized per file stamp). A job with
nothing to say gets a blank cell, never an error.
"""
from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from mace.cluster import progress as progress_mod
from mace.cluster import totals as totals_mod
from mace.recovery.slurm_limits import parse_walltime

# not 'resubmitted': that row is a recovered run's superseded parent, whose job
# ended; its successor is its own row (pending/submitted/running)
ACTIVE = ("pending", "submitted", "running")


def _short(s: float) -> str:
    """2.1 h, 25 min, 40 s: one unit, short."""
    if s >= 3600:
        return f"{s / 3600:.1f} h"
    if s >= 60:
        return f"{round(s / 60)} min"
    return f"{round(s)} s"


def _pair(lo: float, hi: float) -> str:
    """"1.4-3.2 h" when both ends share a unit, else "40 min-1.5 h"."""
    a, b = _short(lo), _short(hi)
    ua, ub = a.split()[-1], b.split()[-1]
    return f"{a.rsplit(' ', 1)[0]}-{b}" if ua == ub else f"{a}-{b}"


def eta_cell(total: Optional[Dict[str, Any]], stalled: bool = False) -> str:
    """The compact ETA of one job from its total level: "~2.1 h (1.4-3.2 h)", ">= 1.4 h",
    "finished"; "rough" when the level is; "possibly stalled" when flagged; "" when unknown."""
    if not total:
        return ""
    if (total.get("info") or {}).get("ended"):
        return "finished"
    lo, hi = total["remaining_s"]["low"], total["remaining_s"]["high"]
    if lo is None:
        txt = ""
    elif hi is None:
        txt = f">= {_short(lo)}"
    else:
        mid = math.sqrt(max(lo, 1.0) * max(hi, 1.0))
        txt = f"~{_short(mid)} ({_pair(lo, hi)})" if hi > lo * 1.05 else f"~{_short(hi)}"
    if txt and total.get("confidence") == "rough":
        txt += " rough"
    if stalled:
        txt = (txt + ", " if txt else "") + "possibly stalled"
    return txt


def _out_of(calc: Dict[str, Any]) -> Optional[Path]:
    for key in ("output_file", "input_file"):
        v = calc.get(key)
        if not v:
            continue
        p = Path(v)
        if not p.is_absolute() and calc.get("work_dir"):
            p = Path(calc["work_dir"]) / p
        p = p.with_suffix(".out")
        if p.is_file():
            return p
    return None


def job_rows(calcs: Sequence[Dict[str, Any]], squeue_rows: Sequence[Dict[str, Any]],
             now: Optional[float] = None) -> List[Dict[str, Any]]:
    """One row per active tracked calculation that has a SLURM id:
    {calc_id, calc_type, job_id, state, eta, stalled, total, queue}."""
    now = now or time.time()
    by_id = {str(r.get("id")): r for r in squeue_rows}
    out = []
    for c in calcs:
        jid = str(c.get("slurm_job_id") or "")
        if not jid or (c.get("status") or "") not in ACTIVE:
            continue
        row = by_id.get(jid)
        state = (row or {}).get("state") or c.get("slurm_state") or (c.get("status") or "").upper()
        rec = {"calc_id": c.get("calc_id"), "calc_type": c.get("calc_type"), "job_id": jid, "state": state,
               "eta": "", "stalled": False, "total": None, "queue": None, "out": None}
        if row is not None and (row.get("state") or "").upper() == "PENDING":
            q = totals_mod.queue_wait(row, now)
            rec["queue"] = q
            if q and q.get("low_s") is not None:
                rec["eta"] = f"starts in ~{_short(q['low_s'])} (SLURM)"
            out.append(rec)
            continue
        out_p = _out_of(c)
        if out_p is not None:
            rec["out"] = str(out_p)
            elapsed = parse_walltime(row.get("elapsed")) if row and row.get("elapsed") else None
            limit = parse_walltime(row.get("time_limit")) if row and row.get("time_limit") else None
            try:
                pr = progress_mod.progress(out_p, job={"elapsed_s": elapsed, "limit_s": limit}
                                           if elapsed is not None and (state or "").upper() == "RUNNING" else None)
                tot = next((lv for lv in pr.get("levels") or [] if lv.get("id") == "total"), None)
                rec["total"] = {"low_s": tot["remaining_s"]["low"], "high_s": tot["remaining_s"]["high"],
                                "confidence": tot["confidence"], "text": tot["text"]} if tot else None
                rec["stalled"] = bool(pr.get("stalled"))
                rec["eta"] = eta_cell(tot, rec["stalled"])
            except Exception:
                pass                    # a broken output: a blank cell, as before
        out.append(rec)
    return out


def footer(calcs: Sequence[Dict[str, Any]], rows: Sequence[Dict[str, Any]], squeue_rows: Sequence[Dict[str, Any]],
           now: Optional[float] = None, acct_for=None) -> List[str]:
    """The totals footer: one line per array listed and per workflow (one per workflow plan)."""
    now = now or time.time()
    lines: List[str] = []
    by_id = {str(r.get("id")): r for r in squeue_rows}
    arrays = sorted({str(by_id[r["job_id"]].get("array_job")) for r in rows
                     if r["job_id"] in by_id and by_id[r["job_id"]].get("array_job")
                     and (by_id[r["job_id"]].get("array_task") or "N/A") not in ("N/A", "")})
    acct = {}
    if arrays and acct_for is not None:
        try:
            acct = acct_for(arrays)              # one batched sacct for every array listed
        except Exception:
            acct = {}
    for aid in arrays:
        own = next((r["total"] for r in rows if r["total"] and by_id.get(r["job_id"], {}).get("array_job") == aid), None)
        a = totals_mod.array_total(aid, squeue_rows, acct, now, own_total=own and {**own, "elapsed_s": None})
        if a:
            lines.append(a["text"])
    seen = set()
    calc_by = {str(c.get("slurm_job_id")): c for c in calcs if c.get("slurm_job_id")}
    for r in rows:
        if (r["state"] or "").upper() != "RUNNING" or not r.get("out"):
            continue
        c = calc_by.get(r["job_id"]) or {}
        job_dir = Path(r["out"]).parent
        key = (str(job_dir), c.get("material_id"))
        if key in seen:
            continue
        seen.add(key)
        try:
            w = totals_mod.workflow_total(job_dir, r["job_id"], Path(r["out"]), None, squeue_rows, now,
                                          own_total=r["total"])
        except Exception:
            w = None
        if w and w.get("steps"):
            lines.append(w["text"])
    return lines


def table_lines(rows: Sequence[Dict[str, Any]]) -> List[str]:
    """The jobs table: id, type, SLURM job, state, ETA (blank when unknown)."""
    if not rows:
        return []
    w_id = max(len(str(r["calc_id"])) for r in rows)
    out = ["Jobs (ETA from the output; 'rough' = wide or thin-data estimate):"]
    for r in rows:
        cell = r["eta"]
        out.append(f"  {str(r['calc_id']):<{w_id}}  {str(r['calc_type'] or ''):<5} {r['job_id']:<10} "
                   f"{str(r['state'] or ''):<9}{(' ' + cell) if cell else ''}".rstrip())
    return out


def status_section(calcs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Everything the status view prints, from one cached squeue (and one sacct for arrays)."""
    active = [c for c in calcs if c.get("slurm_job_id") and (c.get("status") or "") in ACTIVE]
    if not active:
        return {"rows": [], "lines": [], "footer": []}
    from mace.cluster import reader
    try:
        squeue_rows = reader.squeue_own().value or [] if reader.slurm_present() else []
    except Exception:
        squeue_rows = []
    now = time.time()
    rows = job_rows(active, squeue_rows, now)
    foot = footer(active, rows, squeue_rows, now, acct_for=lambda ids: reader.sacct_for(ids))
    return {"rows": rows, "lines": table_lines(rows), "footer": foot}
