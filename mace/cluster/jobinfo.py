"""One of the user's own jobs in detail: state, resources, efficiency,
files, progress, ETA and walltime risk.

Only the user's own jobs: an id must look like ``N`` or ``N_T`` and be in
the cached ``squeue -u`` rows or the ``sacct -u`` rows, else it is "Not one
of your jobs". ``scontrol show job`` runs when a Job document opens and on
its Refresh (and once per ``mace cluster job``); accounting comes from the
shared batched ``sacct``; ``sstat`` only on the explicit "Show Live Use".

Efficiency (CPU = TotalCPU / (Elapsed x NCPUS), memory = MaxRSS / request)
is shown only when plausible (TotalCPU > 0 and <= 1.05 x cores x elapsed;
MaxRSS > 0): MACE's job script starts the MPI ranks through mpirun's ssh
bootstrap, outside SLURM's steps, so SLURM often measures nothing for them
("Not measured by SLURM for this job").
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from mace.cluster import progress as progress_mod
from mace.cluster import reader, recoveryview
from mace.recovery.slurm_limits import format_walltime, parse_walltime

NOT_MEASURED = "Not measured by SLURM for this job"
# sacct states of a job that has not ended yet
_QUEUED = {"PENDING", "RUNNING", "REQUEUED", "RESIZING", "SUSPENDED", "CONFIGURING", "COMPLETING"}


class NotYours(LookupError):
    pass


def _mem_mb(text: Optional[str]) -> Optional[float]:
    m = re.match(r"^([\d.]+)([KMGT]?)", (text or "").strip())
    if not m:
        return None
    return float(m.group(1)) * {"": 1 / 1024 / 1024, "K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 ** 2}[m.group(2)]


def efficiency(acct: Dict[str, Dict[str, Any]], job_id: str) -> Dict[str, Any]:
    """CPU and memory efficiency from sacct rows (allocation + steps), each
    only when plausible."""
    main = acct.get(job_id) or {}
    steps = [v for k, v in acct.items() if k.startswith(job_id + ".")]
    elapsed = parse_walltime(main.get("Elapsed"))
    ncpus = int(main.get("NCPUS") or 0) if str(main.get("NCPUS") or "").isdigit() else 0
    tcpu = [parse_walltime((r.get("TotalCPU") or "").split(".")[0]) for r in [main] + steps]
    tcpu = max([t for t in tcpu if t] or [0])
    rss = max([_mem_mb(r.get("MaxRSS")) or 0 for r in [main] + steps] or [0])
    req = main.get("ReqMem") or ""
    req_mb = _mem_mb(req.rstrip("nc"))
    if req.endswith("c") and req_mb and ncpus:
        req_mb *= ncpus
    out = {"cpu": None, "memory": None, "cpu_note": NOT_MEASURED, "memory_note": NOT_MEASURED,
           "max_rss_mb": rss or None, "req_mem_mb": req_mb, "total_cpu_s": tcpu or None,
           "elapsed_s": elapsed, "ncpus": ncpus or None}
    if elapsed and ncpus and tcpu and 0 < tcpu <= 1.05 * ncpus * elapsed:
        out["cpu"] = round(100.0 * tcpu / (ncpus * elapsed), 1)
        out["cpu_note"] = None
    if rss and req_mb:
        out["memory"] = round(100.0 * rss / req_mb, 1)
        out["memory_note"] = None
    return out


_NOT_STARTED = ("PENDING", "CONFIGURING")


def _mtime(path: Optional[str]) -> Optional[float]:
    try:
        return os.path.getmtime(path) if path else None
    except OSError:
        return None


def own_row(job_id: str, refresh: bool = False, tag: Optional[str] = None) -> Optional[Dict[str, Any]]:
    res = reader.squeue_own(refresh=refresh, tag=tag)
    for r in res.value or []:
        if r.get("id") == job_id:
            return r
    return None


ARRAY_SHOWN = 200


def array_tasks(job_id: str, row: Optional[Dict[str, Any]], detail: Dict[str, str],
                rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The queued and running tasks of this job's array, from the cached
    ``squeue -r`` rows (no query of its own); None for a job that is not an
    array task."""
    task = (row or {}).get("array_task") or detail.get("ArrayTaskId") or ""
    if "_" not in job_id and task in ("", "N/A", "(null)"):
        return None
    aid = job_id.split("_", 1)[0] if "_" in job_id else ((row or {}).get("array_job") or detail.get("ArrayJobId") or "")
    if not aid:
        return None
    mine = [r for r in rows if str(r.get("array_job") or "") == aid and (r.get("array_task") or "N/A") != "N/A"]
    counts: Dict[str, int] = {}
    for r in mine:
        st = (r.get("state") or "UNKNOWN").upper()
        counts[st] = counts.get(st, 0) + 1
    mine.sort(key=lambda r: int(str(r.get("array_task")).split("%")[0]) if str(r.get("array_task")).isdigit() else 10 ** 9)
    return {"array_job": aid, "total_in_queue": len(mine), "counts": counts,
            "tasks": [{"id": r.get("id"), "task": r.get("array_task"), "state": r.get("state"),
                       "elapsed": r.get("elapsed"), "reason": r.get("reason"), "name": r.get("name")}
                      for r in mine[:ARRAY_SHOWN]]}


def _files(detail: Dict[str, str], row: Optional[Dict[str, Any]], job_id: str,
           may_read: Optional[Callable[[str], bool]] = None) -> Dict[str, Any]:
    """The job's files; nothing is opened in a folder ``may_read`` refuses."""
    may_read = may_read or (lambda p: True)
    workdir = detail.get("WorkDir") or (row or {}).get("workdir") or ""
    name = detail.get("JobName") or (row or {}).get("name") or ""
    files: Dict[str, Any] = {"workdir": workdir or None, "stdout": detail.get("StdOut") or None,
                             "stderr": detail.get("StdErr") or None}
    deck_dir, deck = workdir, name
    if "_" in job_id:
        # a MACE array task: the deck comes from the array's manifest
        cmd = detail.get("Command") or ""
        man = Path(cmd).parent / "manifest.tsv" if cmd else None
        if man and may_read(str(man.parent)) and man.is_file():
            try:
                from mace.submission.job_array import read_manifest
                task = job_id.split("_", 1)[1]
                for m in read_manifest(man):
                    if m["task"] == task:
                        deck_dir, deck = m["dir"], m["job"]
                        files["manifest"] = str(man)
                        break
            except Exception:
                pass
    if deck_dir and deck:
        out = Path(deck_dir) / f"{deck}.out"
        files["out"] = str(out)
        files["deck"] = str(Path(deck_dir) / f"{deck}.d12")
        ptr = Path(deck_dir) / f".{deck}.scratch"
        files["scratch"] = None
        if may_read(deck_dir):
            try:
                files["scratch"] = ptr.read_text().strip() or None
            except OSError:
                files["scratch"] = None
        files["job_dir"] = deck_dir
        files["deck_stem"] = deck
    return files


def owned(path: str) -> bool:
    try:
        return os.stat(path).st_uid == os.getuid()
    except (OSError, AttributeError):
        return False


def stopped_progress(prog: Dict[str, Any], state: str) -> Dict[str, Any]:
    """The progress of a job SLURM ended (TIMEOUT, CANCELLED, ...) while its
    output had not finished: a killed CRYSTAL writes no last line, so the
    output alone reads as still running. The stage it was in reads "stopped
    here", nothing is left to estimate. A copy: progress() keeps its result
    in a memo."""
    import copy
    prog = copy.deepcopy(prog)
    why = f"the job ended ({state}) before CRYSTAL finished; nothing is left to estimate"
    prog["stopped"] = state
    prog["levels"] = [lv for lv in prog["levels"] if lv["depth"] < 2]      # no loop runs inside a stage now
    at = next((lv["label"] for lv in prog["levels"] if lv["depth"] == 1 and lv["status"] == "running"), None)
    prog["summary"] = f"Stopped ({state})" + (f" in {at}" if at else f" - {prog.get('summary') or 'output unfinished'}")
    for lv in prog["levels"]:
        if lv["depth"] == 0:
            lv.update(status="done", confidence="fair", unknown_reason=None, basis=why)
            lv["remaining_s"] = {"low": 0.0, "high": 0.0}
            lv["info"]["basis_head"] = why
            lv["text"] = f"Total: the job ended ({state}) before CRYSTAL finished"
            continue
        lv["basis"] = ""                    # the reasons behind estimates that no longer apply
        if lv["status"] == "running":
            lv["status"] = "stopped"
            lv["text"] = f"{lv['label']}: stopped here ({state})"
        elif lv["status"] == "pending":
            lv["status"] = "not reached"
            lv["text"] = f"{lv['label']}: not reached"
            lv["remaining_s"] = {"low": None, "high": None}
    eta = prog.setdefault("eta", {})
    eta.update(low_s=None, high_s=None, confidence="unknown", basis=why)
    prog["risk"] = {"level": None, "sentence": None, "note": None}
    return prog


def job(job_id: str, refresh: bool = False, live: bool = False, use_engine: bool = False,
        tag: Optional[str] = None, path_policy: Optional[Callable[[str], str]] = None,
        scratch_ok: Optional[Callable[[str], bool]] = None, batch_sacct_now: bool = True) -> Dict[str, Any]:
    """Everything the Job document and `mace cluster job` show.

    ``path_policy(path) -> "in_root" | "other_root" | "outside"`` (the GUI's
    served-root rule); without it (the CLI) a file is read when the user
    owns it. ``scratch_ok(path)`` decides whether a scratch dir may be read
    (the GUI: inside the scratch roots). ``batch_sacct_now=False`` (the
    GUI) leaves a running job's accounting to the poller's next batch."""
    job_id = str(job_id).strip()
    if not reader.valid_job_id(job_id):
        raise ValueError(f"not a job id: {job_id!r}")
    row = own_row(job_id, tag=tag)
    if row is not None:
        # a queued or running job: accounting comes from the shared batch the
        # poller runs for every open Job document (no sacct of its own)
        cached = (reader.query.read_cache("sacct:watched").get("value") or {}).get("rows") or {}
        acct = {k: v for k, v in cached.items() if k == job_id or k.startswith(job_id + ".")}
        if not acct and batch_sacct_now:
            acct = reader.sacct_for([job_id], tag=tag)
    else:
        acct = reader.sacct_for([job_id], tag=tag)
    if row is None and job_id not in acct:
        raise NotYours("Not one of your jobs")
    d = reader.job_detail(job_id, refresh=refresh, tag=tag)
    detail = d.value or {}
    if detail and detail.get("UserId") and not detail["UserId"].startswith(reader.user() + "("):
        raise NotYours("Not one of your jobs")
    a = acct.get(job_id) or {}
    acct_state = (a.get("State") or "").split(" ")[0]
    if detail and row is None and acct_state and acct_state not in _QUEUED:
        # the queue no longer lists it and accounting says it ended: scontrol forgets an ended job
        # within minutes on many clusters, its refresh then fails and the cache keeps the last
        # record ("RUNNING") for good. Its state, run time and reason come from sacct instead.
        detail = {k: v for k, v in detail.items() if k not in ("JobState", "RunTime", "Reason")}
    if not detail and row is None and a:
        # an ended job scontrol has forgotten: its name, size, node, start and
        # folder (so its output) from sacct, as the Job document showed them
        detail = {k: a[f] for k, f in (("JobName", "JobName"), ("NumCPUs", "NCPUS"), ("NodeList", "NodeList"),
                                       ("StartTime", "Start"), ("WorkDir", "WorkDir")) if a.get(f)}
    if path_policy is not None:
        may_read = lambda p: path_policy(os.path.realpath(p)) in ("in_root", "other_root")
    else:
        may_read = lambda p: owned(os.path.realpath(p))
    state_note = None
    if (detail.get("JobState") or (row or {}).get("state")) in _NOT_STARTED:
        # A cached "not started yet" answer (scontrol and squeue are cached for a
        # minute) that the job's output contradicts: CRYSTAL wrote the .out after
        # SLURM was asked. Ask again; when the refresh floor keeps the old answer,
        # the output is the newer evidence.
        asked = d.fetched_at
        written = _mtime(_files(detail, row, job_id, may_read).get("out"))
        if written is not None and asked is not None and written > asked:
            d = reader.job_detail(job_id, refresh=True, tag=tag)
            detail = d.value or detail
            row = own_row(job_id, refresh=True, tag=tag) or row
            if (detail.get("JobState") or (row or {}).get("state")) in _NOT_STARTED and \
                    (d.fetched_at is None or written > d.fetched_at):
                age = int(d.age or 0)
                state_note = (f"SLURM's last answer ({detail.get('JobState') or (row or {}).get('state')}, "
                              f"{age} s old) predates the output CRYSTAL is writing: shown as running")
                detail = {k: v for k, v in detail.items() if k not in ("Reason", "RunTime")}
                detail["JobState"] = "RUNNING"
                if row is not None:
                    row = {k: v for k, v in row.items() if k not in ("reason", "elapsed")}
                    row["state"] = "RUNNING"
    state = (detail.get("JobState") or (row or {}).get("state") or
             ((acct.get(job_id) or {}).get("State") or "").split(" ")[0] or "UNKNOWN")
    limit_s = parse_walltime(detail.get("TimeLimit") or (row or {}).get("time_limit") or
                             (acct.get(job_id) or {}).get("Timelimit"))
    elapsed_s = parse_walltime(detail.get("RunTime") or (row or {}).get("elapsed") or
                               (acct.get(job_id) or {}).get("Elapsed"))
    if state == "RUNNING" and row is not None and elapsed_s is not None and detail.get("RunTime") and d.age:
        # scontrol's RunTime comes from the cached query, which can be a minute or more
        # old: a running job's clock has moved on by the cache's age since
        elapsed_s += d.age
        if limit_s:
            elapsed_s = min(elapsed_s, limit_s)
    res: Dict[str, Any] = {
        "id": job_id, "name": detail.get("JobName") or (row or {}).get("name"), "state": state,
        "reason": detail.get("Reason") or (row or {}).get("reason"),
        "start": detail.get("StartTime") or (row or {}).get("start"),
        "elapsed_s": elapsed_s, "limit_s": limit_s, "time_limit": detail.get("TimeLimit") or (row or {}).get("time_limit"),
        "ncpus": detail.get("NumCPUs") or (row or {}).get("cpus"), "nnodes": detail.get("NumNodes") or (row or {}).get("nodes"),
        "nodelist": detail.get("NodeList") or (row or {}).get("nodelist"),
        "partition": detail.get("Partition") or (row or {}).get("partition"),
        "account": detail.get("Account") or (row or {}).get("account"), "qos": detail.get("QOS"),
        "tres": detail.get("ReqTRES") or detail.get("TRES"), "dependency": detail.get("Dependency"),
        "array_job": detail.get("ArrayJobId"), "array_task": detail.get("ArrayTaskId"),
        "detail_cache": d.as_dict(), "accounting": acct.get(job_id), "efficiency": efficiency(acct, job_id),
        "scontrol_command": f"scontrol show job {job_id}",
    }
    if state_note:
        res["state_note"] = state_note
    res["array_tasks"] = array_tasks(job_id, row, detail, reader.squeue_own(tag=tag).value or [])
    files = _files(detail, row, job_id, may_read)
    access = {}
    for k in ("workdir", "stdout", "stderr", "out", "deck", "scratch"):
        p = files.get(k)
        if not p:
            continue
        rp = os.path.realpath(p)
        if path_policy is not None:
            access[k] = path_policy(rp)
        else:
            access[k] = "readable" if owned(rp) else "outside"
    res["files"] = files
    res["access"] = access
    if live:
        s = reader.sstat(job_id, tag=tag)
        res["live"] = {"rows": s.value, "cache": s.as_dict()}
    out = files.get("out")
    readable = out and access.get("out") in ("in_root", "other_root", "readable") and os.path.isfile(out)
    scratch = files.get("scratch")
    if scratch and not (scratch_ok(scratch) if scratch_ok else owned(scratch)):
        scratch = None
    if readable:
        try:
            prog = progress_mod.progress(out, scratch_dir=scratch,
                                         job={"elapsed_s": elapsed_s if state == "RUNNING" else None,
                                              "limit_s": limit_s})
        except Exception as exc:
            prog = {"mode": "unknown", "error": f"{type(exc).__name__}: {exc}"}
        if state not in _QUEUED and state != "UNKNOWN" and prog.get("levels") and not prog.get("ended"):
            prog = stopped_progress(prog, state)
        res["progress"] = prog
        lvl = (prog.get("risk") or {}).get("level")
        if lvl and lvl != "ok" and state == "RUNNING":
            job_dir = Path(files.get("job_dir") or os.path.dirname(out))
            cfg = recoveryview.timeout_config(job_dir, use_engine=use_engine)
            db = job_dir / "materials.db"
            tracked, spent, ctype = progress_mod.retries_spent(db, job_id, files.get("deck_stem"))
            other = progress_mod.other_tracked_active(db, reader.squeue_own(tag=tag).value or [], job_id)
            qlim = _queue_limit_text(res.get("partition"))
            prog["risk"]["sentence"] = progress_mod.risk_sentence(
                lvl, tracked, cfg, spent, other, ctype or (prog.get("mode") or "").upper(), str(job_dir), qlim)
            prog["risk"]["config"] = {k: cfg.get(k) for k in ("path", "source", "note", "via")}
            remaining = (limit_s - elapsed_s) if (limit_s and elapsed_s is not None) else None
            eta = prog.get("eta") or {}
            lo_t = progress_mod._fmt_dur(eta.get("low_s"))
            left = (f"about {lo_t} - {progress_mod._fmt_dur(eta.get('high_s'))} left in total"
                    if eta.get("high_s") is not None else f"at least {lo_t} left in total")
            if lvl == "likely to exceed" and remaining is not None:
                prog["risk"]["headline"] = (f"Job {job_id} will likely run out of time: {left}, "
                                            f"{progress_mod._fmt_dur(remaining)} of walltime remaining.")
            elif lvl == "tight" and remaining is not None:
                prog["risk"]["headline"] = (f"Job {job_id} may run out of time: {left}, "
                                            f"{progress_mod._fmt_dur(remaining)} of walltime remaining.")
    elif out:
        res["progress"] = None
        res["progress_note"] = ("The output is not readable here" if os.path.exists(out)
                                else "No output yet" if state in ("PENDING", "CONFIGURING") else "No output found")
    return res


def _queue_limit_text(partition: Optional[str]) -> Optional[str]:
    if not partition:
        return None
    try:
        rec = reader.query.read_cache("partitions")
        for p in rec.get("value") or []:
            if p.get("name") == partition.split(",")[0]:
                t = parse_walltime(p.get("max_time"))
                if t:
                    d = t // 86400
                    return f"{d} days" if t % 86400 == 0 and d else format_walltime(t)
    except Exception:
        return None
    return None
