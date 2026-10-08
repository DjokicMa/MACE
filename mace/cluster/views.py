"""The data behind each `mace cluster` subcommand and each Cluster view tab,
as JSON-able dicts. The CLI renders them as text; the GUI serves them.
All SLURM reads go through ``reader`` (cached, rate-limited)."""
from __future__ import annotations

import os
import socket
import time
from typing import Any, Dict, List, Optional

from mace.cluster import config as config_mod
from mace.cluster import fit as fit_mod
from mace.cluster import limits as limits_mod
from mace.cluster import nodes as nodes_mod
from mace.cluster import query, reader
from mace.cluster import reservations as res_mod

NO_SLURM = ("No SLURM on this machine (sinfo not found). mace cluster reads the cluster you are logged in to; "
            "run it on a login or development node.")


def _meta(*results) -> Dict[str, Any]:
    rs = [r for r in results if r is not None]
    fetched = [r.fetched_at for r in rs if r.fetched_at]
    nxt = max([r.next_allowed_in for r in rs] or [0])
    return {"updated_at": min(fetched) if fetched else None,
            "age": round(query.now() - min(fetched), 1) if fetched else None,
            "stale": any(r.stale for r in rs), "next_allowed_in": round(nxt, 1),
            "errors": [f"Could not read {r.key or 'data'}: {r.error}" for r in rs if r.error],
            "notes": [r.note for r in rs if r.note]}


def in_allocation() -> Optional[Dict[str, str]]:
    j = (os.environ.get("SLURM_JOB_ID") or "").strip()
    if not j:
        return None
    return {"job": j, "node": os.environ.get("SLURMD_NODENAME") or socket.gethostname().split(".")[0]}


def capabilities(served_root: Optional[str] = None) -> Dict[str, Any]:
    from mace.cluster import quotas
    srcs = []
    if query.available("quota"):
        srcs.append("quota")
    if query.available("lfs"):
        srcs.append("lfs")
    if query.available("mmlsquota"):
        srcs.append("mmlsquota")
    cfg = config_mod.load()
    if cfg.get("quota", "command"):
        srcs.insert(0, "site command")
    slurm = reader.slurm_present()
    on_login = None
    try:
        from mace.submission.local_run import gateway_reason
        on_login = bool(gateway_reason(os.environ, socket.gethostname()))
    except Exception:
        on_login = None
    return {"slurm": slurm, "sinfo": bool(query.available("sinfo")), "sacct": bool(query.available("sacct")),
            "sshare": bool(query.available("sshare")), "squeue": bool(query.available("squeue")),
            "quota_sources": srcs, "df": bool(query.available("df")),
            "cluster_name": query.cluster_name() if slurm else None, "on_login_node": on_login,
            "in_allocation": in_allocation(), "site_config_path": str(config_mod.site_path() or ""),
            "config_errors": cfg.errors, "config_warnings": cfg.warnings}


def _accounts_qos(tag=None, background=False):
    a = reader.assoc_limits(tag=tag, background=background)
    assoc = a["assoc"].value or []
    accounts = sorted({x["account"] for x in assoc})
    qos = sorted({q for x in assoc for q in (x["qos"] or [])} | {x["default_qos"] for x in assoc if x["default_qos"]})
    return a, assoc, accounts, qos


def partitions_annotated(tag=None, background=False, refresh=False):
    p = reader.partitions(tag=tag, background=background, refresh=refresh)
    a, assoc, accounts, qos = _accounts_qos(tag, background)
    parts = nodes_mod.usable_partitions(p.value or [], accounts, nodes_mod.user_groups(), qos)
    if not accounts:                       # no association data: do not hide anything
        for x in parts:
            x["usable"], x["why_not"] = (not x.get("hidden")), None
    return p, a, parts


def nodes_view(group_by: str = "feature", all_partitions: bool = False, partition: Optional[str] = None,
               feature: Optional[str] = None, free: bool = False, refresh: bool = False, background: bool = False,
               tag: Optional[str] = None, page: int = 0, per_page: int = 500) -> Dict[str, Any]:
    cfg = config_mod.load()
    method = "sinfo" if cfg.get("cluster", "node_query") == "sinfo" else "scontrol"
    n = reader.nodes(refresh=refresh, background=background, tag=tag, method=method)
    p, a, parts = partitions_annotated(tag, background)
    nodes = n.value or []
    usable = {x["name"] for x in parts if x["usable"]}
    only = None if all_partitions else (usable or None)
    if partition:
        only = {partition}
    if feature:
        nodes = [x for x in nodes if feature in (x.get("features") or [])]
    if free:
        nodes = [x for x in nodes if nodes_mod.is_up(x) and x["cpus_effective"] > x["cpus_alloc"]]
    rows = nodes_mod.summary(nodes, parts, group_by=group_by, regex=cfg.get("cluster", "group_feature_regex"),
                             only_partitions=only)
    shown_parts = [x for x in parts if all_partitions or x["usable"]]
    per = [{k: x[k] for k in ("name", "state", "flags", "partitions", "cpus_total", "cpus_alloc", "cpus_effective",
                              "mem_total_mb", "mem_alloc_mb", "features", "gres_total", "gres_used")}
           for x in nodes if only is None or set(x.get("partitions") or []) & only]
    return {"cluster": query.cluster_name(), "group_by": group_by, "rows": rows, "totals": nodes_mod.totals(rows),
            "partitions": [{"name": x["name"], "max_time": x["max_time"], "usable": x["usable"],
                            "why_not": x["why_not"], "hidden": x["hidden"], "state": x["state"]} for x in shown_parts],
            "usable_partitions": sorted(usable), "all_partitions": all_partitions,
            "features": sorted({f for x in nodes for f in (x.get("features") or [])}),
            "nodes": per[page * per_page:(page + 1) * per_page], "nodes_total": len(per), "page": page,
            "note": fit_mod.SNAPSHOT_NOTE, **_meta(n, p, a["assoc"])}


def fit_view(req: Dict[str, Any], predict: bool = False, script_path=None, refresh: bool = False,
             tag: Optional[str] = None, server_gap: bool = False, background: bool = False) -> Dict[str, Any]:
    n = reader.nodes(tag=tag, background=background)
    p, a, parts = partitions_annotated(tag, background)
    out = fit_mod.fit(req, n.value or [], parts) if req else {"partitions": [], "unmet_reasons": {}}
    r = reader.reservations(tag=tag, background=background)
    if req and req.get("time_s"):
        wanted = [x for x in parts if x["usable"] and (not req.get("partition") or x["name"] == req["partition"])]
        out["reservations"] = res_mod.conflicts(req["time_s"], time.time(), wanted, r.value or [], n.value or [])
    else:
        out["reservations"] = []
    if predict:
        out["predict"] = fit_mod.predict(req if script_path is None else None, script_path=script_path,
                                         refresh=refresh, server_gap=server_gap, tag=tag)
    out.update(_meta(n, p, r))
    return out


def reservations_view(refresh=False, background=False, tag=None) -> Dict[str, Any]:
    r = reader.reservations(refresh=refresh, background=background, tag=tag)
    now = time.time()
    rows = []
    for x in r.value or []:
        if x.get("end_ts") and x["end_ts"] < now:
            continue
        rows.append({**x, "starts_in_s": (x["start_ts"] - now) if x.get("start_ts") else None})
    rows.sort(key=lambda x: x.get("start_ts") or 0)
    return {"reservations": rows, "message": None if rows else "No reservations scheduled.", **_meta(r)}


def default_account(assoc_mgr_text: str, assoc) -> Optional[str]:
    """The user's default account: the association SLURM marks DefAssoc=Yes,
    else the first association."""
    import re
    u = reader.user()
    blocks = re.split(r"(?=ClusterName=)", assoc_mgr_text or "")
    for b in blocks:
        if "DefAssoc=Yes" in b and re.search(rf"\bUserName={re.escape(u)}\b", b):
            m = re.search(r"\bAccount=(\S+)", b)
            if m:
                return m.group(1)
    return assoc[0]["account"] if assoc else None


def limits_view(check_submit: Optional[int] = None, cpus: Optional[int] = None, account: Optional[str] = None,
                refresh=False, background=False, tag=None) -> Dict[str, Any]:
    a = reader.assoc_limits(refresh=refresh, background=background, tag=tag)
    assoc = a["assoc"].value or []
    qos = a["qos"].value or {}
    rows_r = reader.squeue_own(refresh=refresh, background=background, tag=tag)
    usage = limits_mod.own_usage(rows_r.value or [])
    fs = reader.fairshare(refresh=refresh, background=background, tag=tag)
    from mace.submission.job_array import parse_submit_slots
    if account is None:
        account = default_account(a["usage"].value or "", assoc)
    slots = parse_submit_slots(a["usage"].value or "", reader.user(), account) if a["usage"].value else None
    table = []
    for x in assoc:
        eff = limits_mod.effective_limits([x], qos, x["account"])
        u = usage["by_account"].get(x["account"], {})
        table.append({"account": x["account"], "partition": x["partition"], "qos": ",".join(x["qos"]),
                      "max_jobs": eff["max_jobs"], "max_submit": eff["max_submit"], "max_wall": x["max_wall"],
                      "max_cpus": eff["max_cpus"], "running": u.get("running", 0), "submitted": u.get("submitted", 0),
                      "running_cpus": u.get("running_cpus", 0)})
    lim = limits_mod.effective_limits(assoc, qos, account)
    out = {"associations": table, "fairshare": fs.value or [], "usage": usage, "limits": lim, "account": account,
           "submit_slots": {"limit": slots[0], "used": slots[1]} if slots else None, **_meta(a["assoc"], rows_r, fs)}
    if check_submit:
        out["headroom"] = limits_mod.headroom(int(check_submit), int(cpus or 1), usage, lim, slots)
    note = limits_mod.slots_sentence(*(slots or (None, None)))
    out["slots_note"] = note
    return out


def quotas_view(refresh=False, rediscover=False, tag=None, served_root=None, background=False) -> Dict[str, Any]:
    from mace.cluster import quotas
    r = quotas.read(refresh=refresh, rediscover=rediscover, tag=tag, served_root=served_root, background=background)
    data = r.value or {"rows": [], "sources": [], "notes": [], "warn_percent": 80, "critical_percent": 90}
    meta = _meta(r)
    meta["notes"] = list(data.get("notes") or []) + meta["notes"]
    return {**data, "warnings": quotas.warnings(data), **meta}


def overview(refresh=False, background=False, tag=None, served_root=None) -> Dict[str, Any]:
    nv = nodes_view(refresh=refresh, background=background, tag=tag)
    lv = limits_view(refresh=refresh, background=background, tag=tag)
    qv = quotas_view(refresh=refresh, tag=tag, served_root=served_root, background=background)
    rv = reservations_view(refresh=refresh, background=background, tag=tag)
    soon = [x for x in rv["reservations"] if x.get("starts_in_s") is not None and x["starts_in_s"] < 7 * 86400]
    fsr = lv["fairshare"]
    worst_q = None
    for row in qv.get("rows", []):
        for k in ("files_pct", "space_pct"):
            v = row.get(k)
            if v is not None and row.get("level") in ("warning", "critical") and (worst_q is None or v > worst_q[1]):
                worst_q = (f"{row['label']} {'files' if k == 'files_pct' else 'space'}", v, row["level"])
    return {"cluster": query.cluster_name(), "in_allocation": in_allocation(),
            "usable_partitions": nv["usable_partitions"], "totals": nv["totals"],
            "usage": lv["usage"], "limits": lv["limits"], "submit_slots": lv["submit_slots"],
            "fairshare": fsr, "quota_warnings": qv["warnings"],
            "storage": {"label": worst_q[0], "percent": worst_q[1], "level": worst_q[2]} if worst_q else None,
            "reservations_soon": soon, "updated_at": nv["updated_at"], "age": nv["age"],
            "next_allowed_in": max(nv["next_allowed_in"], lv["next_allowed_in"]),
            "errors": nv["errors"] + lv["errors"] + qv["errors"] + rv["errors"],
            "notes": nv["notes"] + lv["notes"]}
