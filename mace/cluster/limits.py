"""Association and QOS limits, the user's live use of them, fairshare, and
the headroom warnings a batch submission gets before SLURM refuses it.

Sources (all cached through query.run): ``sacctmgr -nP show assoc user=U
format=account,partition,qos,maxjobs,maxsubmit,maxwall,grptres,maxtres,
defaultqos``; ``sacctmgr -nP show qos format=name,maxsubmitpu,maxjobspu,
maxtrespu,maxwall``; ``scontrol show assoc_mgr users=U flags=assoc`` (live
MaxSubmitJobs use, read by ``job_array.parse_submit_slots``); ``sshare -U -l
-P``; and the user's own queue (``squeue_own``) for running/submitted jobs
and running cores. Array tasks count one by one.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from mace.queue.slurm_ids import Truncated, expand_queue_id

ASSOC_FORMAT = "account,partition,qos,maxjobs,maxsubmit,maxwall,grptres,maxtres,defaultqos"
QOS_FORMAT = "name,maxsubmitpu,maxjobspu,maxtrespu,maxwall"


def _int(v) -> Optional[int]:
    v = (v or "").strip()
    return int(v) if v.isdigit() else None


def _tres_cpu(text: str) -> Optional[int]:
    m = re.search(r"(?:^|,)cpu=(\d+)", text or "")
    return int(m.group(1)) if m else None


def parse_assoc(text: str) -> List[Dict[str, Any]]:
    out = []
    for line in (text or "").splitlines():
        f = line.strip().split("|")
        if len(f) < 9 or not f[0]:
            continue
        out.append({"account": f[0], "partition": f[1] or None, "qos": [q for q in f[2].split(",") if q],
                    "max_jobs": _int(f[3]), "max_submit": _int(f[4]), "max_wall": f[5] or None,
                    "grp_tres": f[6] or None, "max_tres": f[7] or None,
                    "max_cpus": _tres_cpu(f[7]), "default_qos": f[8] or None})
    return out


def parse_qos(text: str) -> Dict[str, Dict[str, Any]]:
    out = {}
    for line in (text or "").splitlines():
        f = line.strip().split("|")
        if len(f) < 5 or not f[0]:
            continue
        out[f[0]] = {"name": f[0], "max_submit_pu": _int(f[1]), "max_jobs_pu": _int(f[2]),
                     "max_tres_pu": f[3] or None, "max_cpus_pu": _tres_cpu(f[3]), "max_wall": f[4] or None}
    return out


def parse_sshare(text: str) -> List[Dict[str, Any]]:
    lines = [l for l in (text or "").splitlines() if l.strip()]
    if not lines:
        return []
    head = [h.strip() for h in lines[0].split("|")]
    out = []
    for line in lines[1:]:
        f = line.split("|")
        row = dict(zip(head, f))
        if not row.get("User"):
            continue

        def num(k):
            try:
                return float(row.get(k) or "")
            except ValueError:
                return None
        out.append({"account": row.get("Account", "").strip(), "user": row.get("User", "").strip(),
                    "fairshare": num("FairShare"), "effective_usage": num("EffectvUsage"),
                    "norm_shares": num("NormShares"), "raw_usage": num("RawUsage")})
    return out


def own_usage(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Counts from the user's squeue rows (``squeue -r``; a folded array token
    still expands): running jobs, submitted jobs (every array task counts),
    running cores, per account."""
    out = {"running": 0, "pending": 0, "submitted": 0, "running_cpus": 0, "unknown_tasks": False,
           "by_account": {}}
    for r in rows:
        ids = expand_queue_id(r.get("id", ""))
        n = 1 if isinstance(ids, Truncated) else max(1, len(ids))
        if isinstance(ids, Truncated):
            out["unknown_tasks"] = True
        acct = r.get("account") or ""
        a = out["by_account"].setdefault(acct, {"running": 0, "pending": 0, "submitted": 0, "running_cpus": 0})
        st = (r.get("state") or "").upper()
        out["submitted"] += n
        a["submitted"] += n
        if st == "RUNNING":
            out["running"] += n
            a["running"] += n
            try:
                c = int(r.get("cpus") or 0)
            except ValueError:
                c = 0
            out["running_cpus"] += c * n
            a["running_cpus"] += c * n
        elif st in ("PENDING", "CONFIGURING", "REQUEUED", "SUSPENDED"):
            out["pending"] += n
            a["pending"] += n
    return out


def effective_limits(assoc: Sequence[Dict[str, Any]], qos: Dict[str, Dict[str, Any]],
                     account: Optional[str] = None) -> Dict[str, Optional[int]]:
    """The tightest MaxJobs / MaxSubmit / cores-per-user for the account (or
    over the user's associations), from the association and its QOS."""
    rows = [a for a in assoc if account is None or a["account"] == account]
    jobs, submit, cpus = [], [], []
    for a in rows:
        if a["max_jobs"]:
            jobs.append(a["max_jobs"])
        if a["max_submit"]:
            submit.append(a["max_submit"])
        if a["max_cpus"]:
            cpus.append(a["max_cpus"])
        for qn in a["qos"] or ([a["default_qos"]] if a.get("default_qos") else []):
            q = qos.get(qn)
            if not q:
                continue
            if q["max_jobs_pu"]:
                jobs.append(q["max_jobs_pu"])
            if q["max_submit_pu"]:
                submit.append(q["max_submit_pu"])
            if q["max_cpus_pu"]:
                cpus.append(q["max_cpus_pu"])
    pick = (min if account else max)       # one account: its tightest; overall: the loosest account
    return {"max_jobs": pick(jobs) if jobs else None, "max_submit": pick(submit) if submit else None,
            "max_cpus": pick(cpus) if cpus else None}


def headroom(n_jobs: int, cpus_per_job: int, usage: Dict[str, Any], limits: Dict[str, Optional[int]],
             submit_slots: Optional[tuple] = None) -> List[str]:
    """Warnings for adding ``n_jobs`` jobs of ``cpus_per_job`` cores each.
    ``submit_slots`` = (limit, used) from assoc_mgr when known (it counts what
    SLURM counts); otherwise the own queue count is used."""
    warns: List[str] = []
    if n_jobs <= 0:
        return warns
    if submit_slots:
        limit, used = submit_slots
    else:
        limit, used = limits.get("max_submit"), usage.get("submitted", 0)
    if limit:
        free = max(0, limit - used)
        if n_jobs > free:
            refused = n_jobs - free
            warns.append(f"This adds {n_jobs} job{'s' if n_jobs != 1 else ''}; {used} of {limit} submit slots "
                         f"are in use, so {refused} more {'is' if refused == 1 else 'are'} refused by SLURM.")
    mj = limits.get("max_jobs")
    if mj:
        running = usage.get("running", 0)
        could = max(0, mj - running)
        if n_jobs > could:
            wait = n_jobs - could
            warns.append(f"At most {mj} of your jobs run at once ({running} running now); {wait} of the new "
                         f"jobs will wait with reason AssocMaxJobsLimit.")
    mc = limits.get("max_cpus")
    if mc and cpus_per_job:
        if usage.get("running_cpus", 0) + n_jobs * cpus_per_job > mc:
            warns.append(f"Running cores would exceed your QOS limit ({mc}): extra jobs wait with reason "
                         f"QOSMaxCpuPerUserLimit.")
    return warns


def slots_sentence(limit: Optional[int], used: Optional[int]) -> Optional[str]:
    if not limit or used is None:
        return None
    if used >= 0.9 * limit:
        return f"Your submissions: {used} of {limit} slots in use. Array tasks count one by one."
    return None
