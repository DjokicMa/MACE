"""Nodes and partitions: parse SLURM's one-line records, keep the partitions
the user may submit to, and sum free resources by CPU type or partition.

Node query (spike S1, measured on a 1000-node cluster): ``scontrol show
nodes -o`` took 0.2 s for 420 KB and carries everything needed (CPUs,
memory, features, GPUs in use from ``AllocTRES``); ``sinfo -N`` took the same
time without GPU use. So ``scontrol show nodes -o`` is the source.

A node record is ``{name, state, flags, partitions, cpus_total, cpus_alloc,
cpus_effective, mem_total_mb, mem_alloc_mb, mem_free_mb, features[],
gres_total{type:n}, gres_used{type:n}}``. "CPU type" is one node feature
chosen by ``[cluster] group_feature_regex`` (site.toml); nodes without such a
feature are grouped by their whole feature set.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

_KEY = re.compile(r"(?:^|\s)([A-Z][A-Za-z0-9_]*(?::[A-Za-z0-9_]+)?)=")

# state flags that make a node unusable for new work
BAD_FLAGS = {"DRAIN", "DRAINED", "DRAINING", "DOWN", "MAINT", "RESERVED", "FAIL", "FAILING",
             "NOT_RESPONDING", "POWERED_DOWN", "POWERING_DOWN", "POWER_DOWN", "REBOOT_REQUESTED",
             "REBOOT_ISSUED", "INVALID_REG", "FUTURE", "UNKNOWN", "INVAL", "BLOCKED"}
# sinfo's one-character suffixes
_SUFFIX = {"*": "NOT_RESPONDING", "~": "POWERED_DOWN", "#": "POWERING_UP", "!": "POWERING_DOWN",
           "%": "POWERING_DOWN", "$": "MAINT", "@": "REBOOT_REQUESTED", "^": "REBOOT_ISSUED",
           "-": "PLANNED"}


def kv_record(line: str) -> Dict[str, str]:
    """``Key=value Key2=value with spaces`` -> dict (a value runs to the next
    `` Key=``)."""
    out: Dict[str, str] = {}
    marks = list(_KEY.finditer(line))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(line)
        out[m.group(1)] = line[m.end():end].strip()
    return out


def split_state(text: str):
    """``MIXED+DRAIN`` -> ("mixed", ["DRAIN"]); ``mixed-`` -> ("mixed",
    ["PLANNED"]); ``drained*`` -> ("drained", ["NOT_RESPONDING"]). Unknown
    flags are kept as printed."""
    t = (text or "").strip()
    flags: List[str] = []
    while t and t[-1] in _SUFFIX:
        flags.append(_SUFFIX[t[-1]])
        t = t[:-1]
    parts = t.split("+")
    base = parts[0].lower()
    flags = [p.upper() for p in parts[1:] if p] + flags
    if base in ("drained", "draining", "drain"):
        flags.append("DRAIN")
    if base in ("alloc", "allocated"):
        base = "allocated"
    if base == "mix":
        base = "mixed"
    return base, flags


def _int(v, default=0) -> int:
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return default


def _mem_mb(text: str) -> Optional[float]:
    m = re.match(r"^([\d.]+)([KMGTP]?)$", (text or "").strip())
    if not m:
        return None
    mult = {"": 1, "K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 ** 2, "P": 1024 ** 3}[m.group(2)]
    return float(m.group(1)) * mult


def _tres(text: str) -> Dict[str, str]:
    out = {}
    for part in (text or "").split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def gres_of(tres: Dict[str, str]) -> Dict[str, int]:
    """{gpu type: n} from a TRES map (``gres/gpu:a100=4``); a plain
    ``gres/gpu=4`` without a typed entry counts as type "gpu"."""
    typed: Dict[str, int] = {}
    plain = 0
    for k, v in tres.items():
        if k.startswith("gres/gpu:"):
            t = k.split(":", 1)[1]
            if t in ("gpumem", "gpuutil"):
                continue
            typed[t] = typed.get(t, 0) + _int(v)
        elif k == "gres/gpu":
            plain = _int(v)
    if not typed and plain:
        typed["gpu"] = plain
    return typed


def _gres_config(text: str) -> Dict[str, int]:
    """``gpu:a100:4(S:1,3,5,7)`` -> {"a100": 4}; ``(null)`` -> {}."""
    out: Dict[str, int] = {}
    for part in re.split(r",(?![^()]*\))", text or ""):
        part = re.sub(r"\(.*?\)", "", part).strip()
        if not part.startswith("gpu"):
            continue
        bits = part.split(":")
        if len(bits) == 3:
            out[bits[1]] = out.get(bits[1], 0) + _int(bits[2])
        elif len(bits) == 2:
            out["gpu"] = out.get("gpu", 0) + _int(bits[1])
    return out


def parse_nodes(text: str) -> List[Dict[str, Any]]:
    """Records of ``scontrol show nodes -o`` (one line per node)."""
    nodes = []
    for line in (text or "").splitlines():
        if not line.startswith("NodeName="):
            continue
        r = kv_record(line)
        base, flags = split_state(r.get("State", ""))
        cfg, alloc = _tres(r.get("CfgTRES", "")), _tres(r.get("AllocTRES", ""))
        total_gres = gres_of(cfg) or _gres_config(r.get("Gres", ""))
        mem_total = _int(r.get("RealMemory"))
        spec = _int(r.get("MemSpecLimit"), 0)
        feats = [f for f in (r.get("AvailableFeatures") or "").split(",") if f and f != "(null)"]
        nodes.append({
            "name": r.get("NodeName", ""), "state": base, "flags": flags,
            "partitions": [p for p in (r.get("Partitions") or "").split(",") if p],
            "cpus_total": _int(r.get("CPUTot")), "cpus_alloc": _int(r.get("CPUAlloc")),
            "cpus_effective": _int(r.get("CPUEfctv"), _int(r.get("CPUTot"))),
            "mem_total_mb": max(0, mem_total - spec), "mem_alloc_mb": _int(r.get("AllocMem")),
            "mem_free_mb": _int(r.get("FreeMem"), 0) if r.get("FreeMem") not in (None, "N/A") else None,
            "features": feats, "gres_total": total_gres, "gres_used": gres_of(alloc),
        })
    return nodes


def parse_sinfo_nodes(text: str) -> List[Dict[str, Any]]:
    """Fallback: ``sinfo -h -N -o "%n|%T|%c|%C|%m|%e|%f|%G|%P"`` (deduplicated by
    node; no GPU use)."""
    by: Dict[str, Dict[str, Any]] = {}
    for line in (text or "").splitlines():
        f = line.split("|")
        if len(f) < 9:
            continue
        name, state, cpus, aiot, mem, free, feats, gres, part = f[:9]
        a, i, o, t = (aiot.split("/") + ["0"] * 4)[:4]
        base, flags = split_state(state)
        rec = by.get(name)
        if rec is None:
            rec = by[name] = {"name": name, "state": base, "flags": flags, "partitions": [],
                              "cpus_total": _int(t), "cpus_alloc": _int(a), "cpus_effective": _int(t) - _int(o),
                              "mem_total_mb": _int(mem), "mem_alloc_mb": 0,
                              "mem_free_mb": _int(free) if free.strip().isdigit() else None,
                              "features": [x for x in feats.split(",") if x and x != "(null)"],
                              "gres_total": _gres_config(gres), "gres_used": {}}
        p = part.rstrip("*")
        if p and p not in rec["partitions"]:
            rec["partitions"].append(p)
    return list(by.values())


def parse_partitions(text: str) -> List[Dict[str, Any]]:
    """Records of ``scontrol show partition -a -o``."""
    out = []
    for line in (text or "").splitlines():
        if not line.startswith("PartitionName="):
            continue
        r = kv_record(line)
        out.append({
            "name": r.get("PartitionName", ""), "state": r.get("State", ""),
            "max_time": r.get("MaxTime", ""), "default_time": r.get("DefaultTime", ""),
            "max_nodes": r.get("MaxNodes", ""), "def_mem_per_cpu": r.get("DefMemPerCPU"),
            "max_mem_per_cpu": r.get("MaxMemPerCPU"),
            "allow_accounts": r.get("AllowAccounts", "ALL"), "allow_groups": r.get("AllowGroups", "ALL"),
            "allow_qos": r.get("AllowQos", "ALL"), "deny_accounts": r.get("DenyAccounts", ""),
            "deny_qos": r.get("DenyQos", ""), "nodes": r.get("Nodes", ""),
            "hidden": r.get("Hidden", "NO").upper() == "YES", "default": r.get("Default", "NO").upper() == "YES",
            "total_nodes": _int(r.get("TotalNodes")), "total_cpus": _int(r.get("TotalCPUs")),
        })
    return out


def user_groups() -> List[str]:
    names = []
    try:
        import grp
        for g in os.getgroups():
            try:
                names.append(grp.getgrgid(g).gr_name)
            except KeyError:
                continue
    except (OSError, AttributeError, ImportError):
        pass
    return names


def _listed(value: Optional[str]) -> List[str]:
    return [x.strip() for x in (value or "").split(",") if x.strip() and x.strip() != "(null)"]


def partition_usable(p: Dict[str, Any], accounts: Iterable[str], groups: Iterable[str],
                     qos: Iterable[str]) -> Optional[str]:
    """None when the user may submit to the partition, else why not."""
    accounts, groups, qos = set(accounts), set(groups), set(qos)
    deny_a, deny_q = set(_listed(p.get("deny_accounts"))), set(_listed(p.get("deny_qos")))
    allow_a, allow_g, allow_q = _listed(p.get("allow_accounts")), _listed(p.get("allow_groups")), _listed(p.get("allow_qos"))
    if accounts and accounts <= deny_a:
        return "your accounts are denied"
    usable_qos = qos - deny_q
    if qos and not usable_qos:
        return "your QOS are denied"
    if allow_a and "ALL" not in allow_a and not (accounts & set(allow_a)):
        return "not usable by your accounts"
    if allow_g and "ALL" not in allow_g and not (groups & set(allow_g)):
        return "not usable by your groups"
    if allow_q and "ALL" not in allow_q and not (usable_qos & set(allow_q)):
        return "not usable by your QOS"
    return None


def usable_partitions(parts: Sequence[Dict[str, Any]], accounts, groups, qos) -> List[Dict[str, Any]]:
    """Every partition (visible or hidden) annotated with ``usable`` and
    ``why_not``; the caller filters."""
    out = []
    for p in parts:
        why = partition_usable(p, accounts, groups, qos)
        out.append({**p, "usable": why is None, "why_not": why})
    return out


def is_up(node: Dict[str, Any]) -> bool:
    return node["state"] not in ("down", "drained", "draining", "drain", "fail", "unknown", "future") and \
        not (set(node.get("flags") or []) & BAD_FLAGS)


def group_of(node: Dict[str, Any], regex: Optional[str]) -> str:
    if regex:
        rx = re.compile(regex, re.I)
        for f in node.get("features") or []:
            if rx.match(f):
                return f
    feats = node.get("features") or []
    return ",".join(feats) if feats else "(no features)"


def summary(nodes: Sequence[Dict[str, Any]], parts: Optional[Sequence[Dict[str, Any]]] = None,
            group_by: str = "feature", regex: Optional[str] = None,
            only_partitions: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
    """Rows ``{group, nodes, idle, mixed, allocated, down_drain, cpus_free,
    cpus_total, mem_free_gb, gpus_free{type}, gpus_total{type}}``.

    ``only_partitions``: count only nodes in one of these partitions. A node
    in several partitions counts once per partition row (group_by
    "partition") and once overall (group_by "feature")."""
    rows: Dict[str, Dict[str, Any]] = {}

    def row(name):
        return rows.setdefault(name, {"group": name, "nodes": 0, "idle": 0, "mixed": 0, "allocated": 0,
                                      "down_drain": 0, "cpus_free": 0, "cpus_total": 0,
                                      "mem_free_gb": 0.0, "mem_total_gb": 0.0,
                                      "gpus_free": {}, "gpus_total": {}})

    for n in nodes:
        nparts = set(n.get("partitions") or [])
        if only_partitions is not None and not (nparts & only_partitions):
            continue
        if group_by == "partition":
            keys = sorted(nparts & only_partitions) if only_partitions is not None else sorted(nparts) or ["(none)"]
        else:
            keys = [group_of(n, regex)]
        for k in keys:
            r = row(k)
            r["nodes"] += 1
            r["cpus_total"] += n["cpus_effective"]
            r["mem_total_gb"] += n["mem_total_mb"] / 1024.0
            for t, c in (n.get("gres_total") or {}).items():
                r["gpus_total"][t] = r["gpus_total"].get(t, 0) + c
            if not is_up(n):
                r["down_drain"] += 1
                continue
            if n["state"] == "idle":
                r["idle"] += 1
            elif n["state"] == "mixed":
                r["mixed"] += 1
            elif n["state"] == "allocated":
                r["allocated"] += 1
            r["cpus_free"] += max(0, n["cpus_effective"] - n["cpus_alloc"])
            r["mem_free_gb"] += max(0, n["mem_total_mb"] - n["mem_alloc_mb"]) / 1024.0
            for t, c in (n.get("gres_total") or {}).items():
                used = (n.get("gres_used") or {}).get(t, 0)
                r["gpus_free"][t] = r["gpus_free"].get(t, 0) + max(0, c - used)
    out = sorted(rows.values(), key=lambda r: (-r["cpus_total"], r["group"]))
    for r in out:
        r["mem_free_gb"] = round(r["mem_free_gb"], 1)
        r["mem_total_gb"] = round(r["mem_total_gb"], 1)
    return out


def totals(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    t = {"nodes": 0, "idle": 0, "mixed": 0, "allocated": 0, "down_drain": 0, "cpus_free": 0, "cpus_total": 0}
    for r in rows:
        for k in t:
            t[k] += r[k]
    return t
