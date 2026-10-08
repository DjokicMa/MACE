"""The shared SLURM reader: thin wrappers over ``query.run``, one per key.

| function | key | TTL / refresh floor |
|---|---|---|
| ``squeue_own(max_age=60)`` | ownjobs | 60 s / 15 s |
| ``sinfo()`` | sinfo | 60 s / 15 s |
| ``assoc_limits()`` | assoc, qos (10 min / 2 min), usage (2 min / 1 min) | |
| ``quotas()`` | quota | 10 min / 2 min |
| ``nodes()`` | nodes | 60 s / 60 s |
| ``partitions()`` | partitions | 1 h / 5 min |
| ``reservations()`` | reservations | 5 min / 1 min |
| ``fairshare()`` | sshare | 10 min / 2 min |
| ``job_detail(id)`` | job:<id> | 60 s / 15 s |
| ``sacct_batch(ids)`` | sacct:watched | 60 s / 15 s |
| ``sstat(id)`` | sstat:<id> | 60 s / 30 s |

Every function returns a ``query.Result``; its ``value`` is already parsed
(lists/dicts), so the cache holds JSON. Only the user's own jobs are asked
about in detail (``squeue -u``, ``sacct -u``).
"""
from __future__ import annotations

import getpass
import os
import re
from typing import Any, Dict, List, Optional, Sequence

from mace.cluster import limits as limits_mod
from mace.cluster import nodes as nodes_mod
from mace.cluster import query
from mace.cluster import reservations as res_mod
from mace.queue.slurm_ids import split_squeue_line

# One squeue format for the Jobs tab and the Job document (spike S9): the
# name goes last, so a "|" in a job name cannot shift the other fields.
SQUEUE_FIELDS = (("id", "%i"), ("state", "%T"), ("reason", "%r"), ("start", "%S"), ("time_left", "%L"),
                 ("time_limit", "%l"), ("elapsed", "%M"), ("nodes", "%D"), ("cpus", "%C"),
                 ("partition", "%P"), ("account", "%a"), ("nodelist", "%N"), ("workdir", "%Z"),
                 ("array_job", "%F"), ("array_task", "%K"), ("name", "%j"))
SQUEUE_FORMAT = "|".join(code for _, code in SQUEUE_FIELDS)
SACCT_FIELDS = ("JobID", "JobName", "State", "Elapsed", "Timelimit", "NCPUS", "ReqMem", "MaxRSS", "TotalCPU",
                "AllocTRES", "NodeList", "ExitCode", "Start", "End", "WorkDir")
JOB_ID_RE = re.compile(r"^\d+(_\d+)?$")


def user() -> str:
    return os.environ.get("USER") or os.environ.get("LOGNAME") or getpass.getuser()


def slurm_present() -> bool:
    return bool(query.available("sinfo") or query.available("squeue"))


# ------------------------------------------------------------------ parsers
def parse_squeue(text: str) -> List[Dict[str, Any]]:
    names = [n for n, _ in SQUEUE_FIELDS]
    rows = []
    for line in (text or "").splitlines():
        if not line.strip():
            continue
        f = line.rstrip("\n").split("|", len(names) - 1)
        if len(f) < len(names):
            continue
        r = dict(zip(names, (x.strip() for x in f)))
        rows.append(r)
    return rows


def parse_sinfo(text: str) -> List[Dict[str, Any]]:
    out = []
    for line in (text or "").splitlines():
        f = line.strip().split("|")
        if len(f) < 5:
            continue
        a, i, o, t = (f[4].split("/") + ["0"] * 4)[:4]
        out.append({"partition": f[0].rstrip("*"), "default": f[0].endswith("*"), "avail": f[1],
                    "time_limit": f[2], "nodes": int(f[3]) if f[3].isdigit() else 0,
                    "alloc": int(a) if a.isdigit() else 0, "idle": int(i) if i.isdigit() else 0,
                    "other": int(o) if o.isdigit() else 0, "total": int(t) if t.isdigit() else 0})
    return out


def parse_sacct(text: str) -> Dict[str, Dict[str, Any]]:
    """{JobID: fields} for ``sacct -P -n -o <SACCT_FIELDS>``; step lines
    (``A.batch``, ``A.extern``, ``A.0``) are kept under their own ids."""
    out: Dict[str, Dict[str, Any]] = {}
    for line in (text or "").splitlines():
        f = line.split("|")
        if len(f) < len(SACCT_FIELDS):
            continue
        rec = dict(zip(SACCT_FIELDS, (x.strip() for x in f)))
        out[rec["JobID"]] = rec
    return out


def parse_job_detail(text: str) -> Optional[Dict[str, str]]:
    for line in (text or "").splitlines():
        if line.startswith("JobId="):
            return nodes_mod.kv_record(line)
    return None


# ------------------------------------------------------------------ readers
def squeue_own(max_age: float = 60, refresh: bool = False, background: bool = False,
               tag: Optional[str] = None, fresh: bool = False) -> query.Result:
    argv = ["squeue", "-h", "-r", "-u", user(), "-o", SQUEUE_FORMAT]
    return query.run("ownjobs", argv, ttl=max_age, refresh_floor=15, refresh=refresh, fresh=fresh,
                     background=background, tag=tag, parse=lambda p: parse_squeue(p.stdout))


def sinfo(refresh=False, background=False, tag=None) -> query.Result:
    return query.run("sinfo", ["sinfo", "-h", "-o", "%P|%a|%l|%D|%F"], ttl=60, refresh_floor=15,
                     refresh=refresh, background=background, tag=tag, parse=lambda p: parse_sinfo(p.stdout))


def nodes(refresh=False, background=False, tag=None, method: str = "scontrol") -> query.Result:
    if method == "sinfo":
        argv = ["sinfo", "-h", "-N", "-o", "%n|%T|%c|%C|%m|%e|%f|%G|%P"]
        parse = lambda p: nodes_mod.parse_sinfo_nodes(p.stdout)
    else:
        argv = ["scontrol", "show", "nodes", "-o"]
        parse = lambda p: nodes_mod.parse_nodes(p.stdout)
    return query.run("nodes", argv, ttl=60, refresh_floor=60, refresh=refresh, background=background,
                     tag=tag, parse=parse)


def partitions(refresh=False, background=False, tag=None) -> query.Result:
    return query.run("partitions", ["scontrol", "show", "partition", "-a", "-o"], ttl=3600, refresh_floor=300,
                     refresh=refresh, background=background, tag=tag,
                     parse=lambda p: nodes_mod.parse_partitions(p.stdout))


def reservations(refresh=False, background=False, tag=None) -> query.Result:
    return query.run("reservations", ["scontrol", "show", "res", "-o"], ttl=300, refresh_floor=60,
                     refresh=refresh, background=background, tag=tag, parse=lambda p: res_mod.parse(p.stdout))


def fairshare(refresh=False, background=False, tag=None) -> query.Result:
    return query.run("sshare", ["sshare", "-U", "-l", "-P"], ttl=600, refresh_floor=120, refresh=refresh,
                     background=background, tag=tag, parse=lambda p: limits_mod.parse_sshare(p.stdout))


def assoc_limits(refresh=False, background=False, tag=None) -> Dict[str, query.Result]:
    u = user()
    return {
        "assoc": query.run("assoc", ["sacctmgr", "-nP", "show", "assoc", f"user={u}",
                                     f"format={limits_mod.ASSOC_FORMAT}"], ttl=600, refresh_floor=120,
                           refresh=refresh, background=background, tag=tag,
                           parse=lambda p: limits_mod.parse_assoc(p.stdout)),
        "qos": query.run("qos", ["sacctmgr", "-nP", "show", "qos", f"format={limits_mod.QOS_FORMAT}"],
                         ttl=600, refresh_floor=120, refresh=refresh, background=background, tag=tag,
                         parse=lambda p: limits_mod.parse_qos(p.stdout)),
        "usage": query.run("usage", ["scontrol", "show", "assoc_mgr", f"users={u}", "flags=assoc"], ttl=120,
                           refresh_floor=60, refresh=refresh, background=background, tag=tag),
    }


def quotas(refresh=False, tag=None, rediscover=False):
    from mace.cluster import quotas as quotas_mod
    return quotas_mod.read(refresh=refresh, tag=tag, rediscover=rediscover)


def valid_job_id(job_id: str) -> bool:
    return bool(JOB_ID_RE.match(str(job_id or "")))


def job_detail(job_id: str, refresh=False, tag=None) -> query.Result:
    if not valid_job_id(job_id):
        raise ValueError(f"not a job id: {job_id!r}")
    return query.run(f"job:{job_id}", ["scontrol", "show", "job", "-o", str(job_id)], ttl=60,
                     refresh_floor=15, refresh=refresh, tag=tag, parse=lambda p: parse_job_detail(p.stdout))


def sacct_batch(ids: Sequence[str], refresh=False, tag=None, max_age: float = 60) -> query.Result:
    """One ``sacct -j a,b,c`` for every watched id (own jobs only: ``-u``)."""
    ids = sorted({str(i) for i in ids if valid_job_id(str(i))})
    if not ids:
        return query.Result(value={}, source="none")
    argv = ["sacct", "-u", user(), "-j", ",".join(ids), "-P", "-n", "-o", ",".join(SACCT_FIELDS)]
    # One cache record for the batch, which remembers the id set it answered.
    # A set it does not cover is a new question: it is asked now (one call per
    # new id set, i.e. per user action or finished job), so another set's rows
    # never stand in for it; the TTL and the 15 s floor apply to a covered set.
    cached = set((query.read_cache("sacct:watched").get("value") or {}).get("ids") or [])
    covered = set(ids) <= cached
    return query.run("sacct:watched", argv, ttl=max_age, refresh_floor=15 if covered else 0,
                     refresh=refresh or not covered, tag=tag,
                     parse=lambda p: {"ids": ids, "rows": parse_sacct(p.stdout)})


def sacct_for(ids: Sequence[str], refresh=False, tag=None, max_age: float = 60) -> Dict[str, Dict[str, Any]]:
    """The sacct rows for these ids from the shared batch: when the cached
    batch covers them it is used as is, else one new batch call runs."""
    want = {str(i) for i in ids if valid_job_id(str(i))}
    if not want:
        return {}
    rec = query.read_cache("sacct:watched")
    val = rec.get("value") or {}
    covered = want <= set(val.get("ids") or [])
    if covered and not refresh and rec.get("fetched_at") and query.now() - float(rec["fetched_at"]) < max_age:
        return val.get("rows") or {}
    known = set(val.get("ids") or []) if rec.get("fetched_at") and \
        query.now() - float(rec["fetched_at"]) < 600 else set()
    res = sacct_batch(sorted(want | known), refresh=refresh or not covered, tag=tag, max_age=max_age)
    return (res.value or {}).get("rows") or {}


def sstat(job_id: str, refresh=False, tag=None) -> query.Result:
    if not valid_job_id(job_id):
        raise ValueError(f"not a job id: {job_id!r}")
    return query.run(f"sstat:{job_id}", ["sstat", "-j", f"{job_id}.batch", "-P", "-n", "-o", "MaxRSS,AveCPU"],
                     ttl=60, refresh_floor=30, refresh=refresh, tag=tag,
                     parse=lambda p: [dict(zip(("MaxRSS", "AveCPU"), l.split("|")))
                                      for l in p.stdout.splitlines() if "|" in l])
