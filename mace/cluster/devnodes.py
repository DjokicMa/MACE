"""The least-loaded interactive (development) node.

Hosts come only from ``[devnodes] hosts`` in site.toml; nothing is built
in. Each is probed over ssh (BatchMode, 5 s connect timeout, at most 4 at
once) for ``/proc/loadavg``, ``nproc`` and ``MemAvailable``, cached 2 minutes
(key ``devnodes``). Ranked by load per core (``pick_by = "load"``) or by
available memory (``"memory"``); an unreachable host is listed, never
picked.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from mace.cluster import config as config_mod
from mace.cluster import query

NO_HOSTS = ("No interactive nodes are configured; add [devnodes] hosts = [...] to ~/.config/mace/site.toml "
            "(see docs/cli.md#mace-cluster).")
PROBE = "cat /proc/loadavg; nproc; grep MemAvailable /proc/meminfo"


def parse_probe(text: str) -> Optional[Dict[str, Any]]:
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    if len(lines) < 3:
        return None
    try:
        load1 = float(lines[0].split()[0])
        cores = int(lines[1])
        m = re.search(r"MemAvailable:\s+(\d+)\s*kB", lines[2])
        mem = int(m.group(1)) / 1024 / 1024 if m else None
    except (ValueError, IndexError):
        return None
    return {"load1": load1, "cores": cores, "load_per_core": round(load1 / max(cores, 1), 3),
            "mem_available_gb": round(mem, 1) if mem is not None else None}


def _probe(host: str) -> Dict[str, Any]:
    p = query.execute(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host, PROBE], timeout=15)
    if p.returncode != 0:
        return {"host": host, "reachable": False, "error": query._first_line(p.stderr or p.stdout)}
    info = parse_probe(p.stdout)
    if info is None:
        return {"host": host, "reachable": False, "error": "unreadable answer"}
    return {"host": host, "reachable": True, **info}


def rank(rows: List[Dict[str, Any]], pick_by: str = "load") -> List[Dict[str, Any]]:
    ok = [r for r in rows if r.get("reachable")]
    bad = [r for r in rows if not r.get("reachable")]
    if pick_by == "memory":
        ok.sort(key=lambda r: (-(r.get("mem_available_gb") or 0), r["load_per_core"]))
    else:
        ok.sort(key=lambda r: (r["load_per_core"], -(r.get("mem_available_gb") or 0)))
    return ok + bad


def probe_all(refresh: bool = False, cfg=None) -> Dict[str, Any]:
    cfg = cfg or config_mod.load()
    hosts = list(cfg.get("devnodes", "hosts") or [])
    if not hosts:
        return {"hosts": [], "message": NO_HOSTS, "rows": [], "pick": None}

    def fetch():
        with ThreadPoolExecutor(max_workers=4) as ex:
            rows = list(ex.map(_probe, hosts))
        return query.Proc(0, json.dumps(rows), "")
    res = query.run("devnodes-" + "-".join(hosts)[:60], fetch, ttl=120, refresh_floor=60, refresh=refresh,
                    parse=lambda p: json.loads(p.stdout))
    rows = rank(res.value or [], cfg.get("devnodes", "pick_by"))
    pick = rows[0]["host"] if rows and rows[0].get("reachable") else None
    return {"hosts": hosts, "rows": rows, "pick": pick, "cache": res.as_dict(),
            "message": None if pick else "No configured interactive node answered."}
