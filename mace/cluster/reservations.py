"""Reservations and maintenance windows (``scontrol show res -o``), and which
of them a job of a given walltime would run into."""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from mace.cluster.nodes import kv_record


def _ts(text: str) -> Optional[float]:
    try:
        return datetime.strptime(text.strip(), "%Y-%m-%dT%H:%M:%S").timestamp()
    except (ValueError, AttributeError):
        return None


def parse(text: str) -> List[Dict[str, Any]]:
    """Records; "No reservations in the system" gives []."""
    out = []
    for line in (text or "").splitlines():
        if not line.startswith("ReservationName="):
            continue
        r = kv_record(line)
        flags = [f for f in (r.get("Flags") or "").split(",") if f]
        out.append({"name": r.get("ReservationName", ""), "start": r.get("StartTime", ""),
                    "end": r.get("EndTime", ""), "start_ts": _ts(r.get("StartTime", "")),
                    "end_ts": _ts(r.get("EndTime", "")), "nodes": r.get("Nodes", ""),
                    "node_count": int(r.get("NodeCnt") or 0) if (r.get("NodeCnt") or "").isdigit() else 0,
                    "partition": r.get("PartitionName", "") if r.get("PartitionName") not in ("(null)", None) else "",
                    "flags": flags, "maintenance": "MAINT" in flags,
                    "users": r.get("Users", ""), "accounts": r.get("Accounts", ""),
                    "state": r.get("State", "")})
    return out


def expand_hostlist(expr: str, limit: int = 100000) -> List[str]:
    """``cpu-[001-003,007],gpu-01`` -> names (bounded)."""
    out: List[str] = []
    for part in re.split(r",(?![^\[]*\])", expr or ""):
        part = part.strip()
        if not part or part == "(null)":
            continue
        m = re.match(r"^(.*?)\[([^\]]+)\](.*)$", part)
        if not m:
            out.append(part)
            continue
        pre, body, post = m.groups()
        for rng in body.split(","):
            if "-" in rng:
                lo, hi = rng.split("-", 1)
                width = len(lo)
                try:
                    for i in range(int(lo), int(hi) + 1):
                        out.append(f"{pre}{str(i).zfill(width)}{post}")
                        if len(out) >= limit:
                            return out
                except ValueError:
                    out.append(part)
            else:
                out.append(f"{pre}{rng}{post}")
    return out


def conflicts(walltime_s: float, now: float, partitions: Sequence[Dict[str, Any]],
              reservations: Sequence[Dict[str, Any]],
              nodes: Optional[Sequence[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Reservations that start before ``now + walltime`` and share nodes with
    the request's partitions (or name one of them). Each gets
    ``starts_in_s`` and ``maintenance``."""
    end = now + walltime_s
    pnames = {p["name"] for p in partitions}
    pnodes = set()
    if nodes is not None:
        for n in nodes:
            if set(n.get("partitions") or []) & pnames:
                pnodes.add(n["name"])
    else:
        for p in partitions:
            pnodes.update(expand_hostlist(p.get("nodes", "")))
    out = []
    for r in reservations:
        if r.get("start_ts") is None:
            continue
        if r.get("end_ts") is not None and r["end_ts"] <= now:
            continue
        if r["start_ts"] >= end:
            continue
        shared = (r.get("partition") and r["partition"] in pnames) or \
            bool(pnodes & set(expand_hostlist(r.get("nodes", ""))))
        if r.get("nodes", "").upper() == "ALL":
            shared = True
        if shared:
            out.append({**r, "starts_in_s": max(0.0, r["start_ts"] - now)})
    return out


def human_duration(seconds: float) -> str:
    s = int(max(0, seconds))
    d, rem = divmod(s, 86400)
    h, rem = divmod(rem, 3600)
    m = rem // 60
    if d:
        return f"{d} d {h} h"
    if h:
        return f"{h} h {m} min"
    return f"{m} min"
