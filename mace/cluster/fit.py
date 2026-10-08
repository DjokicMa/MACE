""""Could it start now?" (a filter over the nodes, not a promise) and
"Predict start" (``sbatch --test-only``: SLURM itself, nothing submitted).

A node fits a request when it has no DRAIN/DOWN/MAINT/RESERVED/FAIL/
NOT_RESPONDING flag, enough free CPUs and memory per node, enough free GPUs
of the requested type, the features of the ``--constraint`` expression
(``&``, ``|`` and plain names; brackets and counts are reported as "not
evaluated"), and sits in a usable partition whose MaxTime covers the
walltime. Priority, limits and backfill decide the real start.
"""
from __future__ import annotations

import hashlib
import math
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from mace.cluster import nodes as nodes_mod
from mace.recovery.slurm_limits import parse_test_only, parse_walltime, format_walltime

SNAPSHOT_NOTE = "Free now is a snapshot: priority, limits and backfill decide the real start."
PREDICT_NOTE = "SLURM decides by priority and limits; this asks SLURM itself without submitting anything."
PREDICT_GAP_S = 10.0
_predict_lock = threading.Lock()
_last_predict = [0.0]


def parse_size_mb(text: Optional[str]) -> Optional[float]:
    """``5G`` / ``5120M`` / ``500K`` / ``2T`` / plain MB -> MB."""
    if text in (None, ""):
        return None
    m = re.match(r"^\s*([\d.]+)\s*([KMGT]?)B?\s*$", str(text), re.I)
    if not m:
        raise ValueError(f"not a memory size: {text!r} (use e.g. 4G or 4000M)")
    mult = {"": 1, "K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 ** 2}[m.group(2).upper()]
    return float(m.group(1)) * mult


def parse_gpus(text: Optional[str]) -> Optional[Dict[str, Any]]:
    """``a100:2`` -> {type: a100, n: 2}; ``2`` -> {type: None, n: 2}."""
    if text in (None, ""):
        return None
    parts = str(text).split(":")
    try:
        if len(parts) == 1:
            return {"type": None, "n": int(parts[0])}
        return {"type": parts[-2] if len(parts) >= 2 else None, "n": int(parts[-1])}
    except ValueError:
        raise ValueError(f"not a GPU request: {text!r} (use N or TYPE:N)")


def request_from(cpus: int, nodes: int = 1, mem_per_cpu: Optional[str] = None, mem: Optional[str] = None,
                 gpus: Optional[str] = None, constraint: Optional[str] = None,
                 partition: Optional[str] = None, time_text: Optional[str] = None) -> Dict[str, Any]:
    if cpus is None or int(cpus) < 1:
        raise ValueError("--cpus must be at least 1")
    nodes = max(1, int(nodes or 1))
    per_node = math.ceil(int(cpus) / nodes)
    mpc = parse_size_mb(mem_per_cpu)
    mpn = parse_size_mb(mem)
    t = parse_walltime(time_text) if time_text else None
    if time_text and t is None:
        raise ValueError(f"not a walltime: {time_text!r} (use D-HH:MM:SS)")
    return {"cpus": int(cpus), "nodes": nodes, "cpus_per_node": per_node,
            "mem_per_cpu_mb": mpc, "mem_per_node_mb": mpn if mpn is not None else (mpc * per_node if mpc else 0.0),
            "gpus": parse_gpus(gpus), "constraint": constraint or None, "partition": partition or None,
            "time_s": t, "time_text": time_text or None}


def eval_constraint(expr: Optional[str], features: Sequence[str]) -> Optional[bool]:
    """True/False for ``a&b|c`` (``&`` binds tighter than ``|``, as SLURM);
    None ("not evaluated") for brackets, counts or anything else."""
    if not expr:
        return True
    if re.search(r"[\[\]*()]", expr):
        return None
    feats = set(features)
    for alt in expr.split("|"):
        names = [x.strip() for x in alt.split("&")]
        if any(not re.match(r"^[A-Za-z0-9_.-]+$", n) for n in names):
            return None
        if all(n in feats for n in names):
            return True
    return False


def node_fits(n: Dict[str, Any], req: Dict[str, Any]) -> Optional[str]:
    """None when the node fits, else the reason."""
    if not nodes_mod.is_up(n):
        return "down, drained or reserved"
    if n["cpus_effective"] - n["cpus_alloc"] < req["cpus_per_node"]:
        return "not enough free cores"
    if req.get("mem_per_node_mb") and n["mem_total_mb"] - n["mem_alloc_mb"] < req["mem_per_node_mb"]:
        return "not enough free memory"
    g = req.get("gpus")
    if g and g.get("n"):
        total, used = n.get("gres_total") or {}, n.get("gres_used") or {}
        if g.get("type"):
            free = total.get(g["type"], 0) - used.get(g["type"], 0)
        else:
            free = sum(total.values()) - sum(used.values())
        if free < g["n"]:
            return "not enough free GPUs"
    c = eval_constraint(req.get("constraint"), n.get("features") or [])
    if c is False:
        return "lacks the constraint features"
    return None


def fit(req: Dict[str, Any], nodes: Sequence[Dict[str, Any]], parts: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Per usable partition: how many nodes could take the request now."""
    constraint_note = None
    if req.get("constraint") and eval_constraint(req["constraint"], []) is None:
        constraint_note = f"The constraint {req['constraint']!r} is not evaluated here (brackets or counts); SLURM checks it in Predict Start."
    out_parts = []
    unmet: Dict[str, int] = {}
    wanted = {req["partition"]} if req.get("partition") else None
    for p in parts:
        if wanted is not None and p["name"] not in wanted:
            continue
        if wanted is None and not p.get("usable", True):
            continue
        maxt = parse_walltime(p.get("max_time"))
        if req.get("time_s") and maxt is not None and maxt < req["time_s"]:
            out_parts.append({"name": p["name"], "max_time": p.get("max_time"), "fits_now": 0, "nodes": [],
                              "note": f"MaxTime {p.get('max_time')} is shorter than {req.get('time_text')}",
                              "usable": p.get("usable", True), "why_not": p.get("why_not")})
            continue
        good = []
        for n in nodes:
            if p["name"] not in (n.get("partitions") or []):
                continue
            why = node_fits(n, req)
            if why is None:
                good.append(n["name"])
            else:
                unmet[why] = unmet.get(why, 0) + 1
        enough = len(good) >= req["nodes"]
        out_parts.append({"name": p["name"], "max_time": p.get("max_time"), "fits_now": len(good),
                          "nodes": good[:50], "enough": enough, "usable": p.get("usable", True),
                          "why_not": p.get("why_not"), "note": None})
    out_parts.sort(key=lambda r: (-r["fits_now"], r["name"]))
    return {"request": req, "partitions": out_parts, "unmet_reasons": unmet, "note": SNAPSHOT_NOTE,
            "constraint_note": constraint_note}


def test_only_argv(req: Dict[str, Any]) -> List[str]:
    argv = ["sbatch", "--test-only", "-n", str(req["cpus"]), "-N", str(req["nodes"])]
    if req.get("time_s"):
        argv += ["-t", format_walltime(req["time_s"])]
    if req.get("mem_per_cpu_mb"):
        argv += [f"--mem-per-cpu={int(math.ceil(req['mem_per_cpu_mb']))}M"]
    elif req.get("mem_per_node_mb"):
        argv += [f"--mem={int(math.ceil(req['mem_per_node_mb']))}M"]
    g = req.get("gpus")
    if g and g.get("n"):
        argv += [f"--gres=gpu:{g['type']}:{g['n']}" if g.get("type") else f"--gres=gpu:{g['n']}"]
    if req.get("constraint"):
        argv += ["-C", req["constraint"]]
    if req.get("partition"):
        argv += ["-p", req["partition"]]
    return argv


REQUEST_SCRIPT = "#!/bin/bash\ntrue\n"


def predict(req: Optional[Dict[str, Any]] = None, script_path: Optional[Path] = None,
            refresh: bool = False, server_gap: bool = False, tag: Optional[str] = None) -> Dict[str, Any]:
    """``sbatch --test-only`` for a request (a two-line script on stdin with
    the request's flags) or for an existing job script (its path, run from its
    own folder so site rules see the real directives). Cached two minutes per
    distinct question; ``server_gap`` adds the GUI's one-per-10-s limit."""
    from mace.cluster import query
    if script_path is not None:
        script_path = Path(script_path).resolve()
        if script_path.suffix != ".sh" or not script_path.is_file():
            raise ValueError("Predict Start needs an existing .sh job script")
        argv, stdin, cwd = ["sbatch", "--test-only", str(script_path)], None, str(script_path.parent)
        st = script_path.stat()
        ident = f"script\0{script_path}\0{st.st_size}\0{st.st_mtime_ns}"
    else:
        argv, stdin, cwd = test_only_argv(req), REQUEST_SCRIPT, None
        ident = "req\0" + "\0".join(argv)
    key = "testonly-" + hashlib.sha1(ident.encode()).hexdigest()[:16]
    if server_gap:
        with _predict_lock:
            rec = query.read_cache(key)
            fresh_enough = rec.get("fetched_at") and query.now() - float(rec["fetched_at"]) < 120 and not refresh
            gap = PREDICT_GAP_S - (time.monotonic() - _last_predict[0])
            if not fresh_enough and gap > 0 and _last_predict[0]:
                return {"accepted": None, "throttled": True, "next_allowed_in": round(gap, 1),
                        "reason": f"One prediction per {int(PREDICT_GAP_S)} s; try again in {int(math.ceil(gap))} s",
                        "argv": argv, "note": PREDICT_NOTE}
            if not fresh_enough:
                _last_predict[0] = time.monotonic()
    res = query.run(key, argv, ttl=120, refresh_floor=10, refresh=refresh, tag=tag,
                    parse=lambda p: parse_test_only(p.stdout + "\n" + p.stderr, p.returncode),
                    accept=lambda p: p.returncode in (0, 1) and ("allocation failure" in (p.stdout + p.stderr).lower()
                                                               or "error" in (p.stdout + p.stderr).lower()),
                    input_text=stdin, cwd=cwd)
    out = dict(res.value or {"accepted": None, "reason": res.error or "SLURM did not answer"})
    out.update({"argv": argv, "note": PREDICT_NOTE, "cache": res.as_dict()})
    return out
