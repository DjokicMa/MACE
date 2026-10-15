"""How well a job used the cores and memory it asked for, in plain numbers.

One helper for the GUI's Job document and ``mace cluster job``:

- CPU efficiency = CPU time used / (elapsed x allocated CPUs). The CPU time
  is the job's TotalCPU from ``sacct``, or the sum over its steps (batch,
  extern, ``.0``, ...) when the allocation line has none; a step without
  TotalCPU counts as AveCPU x NTasks. For a running job ``sacct`` has no CPU
  time or MaxRSS yet (SLURM records them when a step ends), so the live
  numbers come from ``sstat`` (AveCPU x NTasks per running step).
- Memory = the peak of MaxRSS over the steps (MaxRSS is one task's peak: a
  step with several tasks counts MaxRSS x NTasks, an upper estimate) over the
  requested memory (AllocTRES ``mem=``, else ReqMem: ``c`` per CPU, ``n`` per
  node, no suffix the job's total).

Each figure is shown only when SLURM measured it; otherwise the note says
why it is missing.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

NOT_MEASURED = "Not measured by SLURM for this job"
_SSTAT_EMPTY = ("sstat returned nothing for the running steps (it may not reach the job's node from this host; "
                "a development node usually can)")

# Shown once under a measured CPU efficiency: the percentage counts busy
# cores, and an MPI rank that waits for messages spins instead of sleeping.
CPU_CAVEAT = ("MPI libraries such as Intel MPI keep waiting ranks busy, so CPU efficiency near 100% does not "
              "show that every core did useful work; compare run times at different core counts to find the "
              "useful size.")

# Advice thresholds: clearly over-requested, after the job ran for a while.
CPU_ADVICE_PCT = 50.0
MEM_ADVICE_PCT = 25.0
MEM_NEAR_LIMIT_PCT = 90.0
ADVICE_MIN_ELAPSED_S = 600
# TotalCPU may exceed cores x elapsed by a little (rounding, clock skew).
PLAUSIBLE_FACTOR = 1.10

_UNITS = {"": None, "K": 1024, "M": 1024 ** 2, "G": 1024 ** 3, "T": 1024 ** 4, "P": 1024 ** 5}
_MEM_RE = re.compile(r"^([\d.]+)\s*([KMGTP]?)([nc]?)$", re.I)


# ------------------------------------------------------------------ parsing
def parse_duration(text: Any) -> Optional[float]:
    """SLURM accounting time to seconds: ``D-HH:MM:SS``, ``HH:MM:SS``,
    ``MM:SS.mmm`` (TotalCPU / AveCPU of short steps), ``D-HH``; None when
    empty or unparseable."""
    s = str(text or "").strip()
    if not s or s.upper() in ("UNLIMITED", "INFINITE", "NONE", "N/A", "INVALID"):
        return None
    days = 0
    if "-" in s:
        d, _, s = s.partition("-")
        try:
            days = int(d)
        except ValueError:
            return None
        parts = s.split(":")
        parts += ["0"] * (3 - len(parts))          # D-HH and D-HH:MM
    else:
        parts = s.split(":")
        if len(parts) == 1:
            parts = ["0", "0"] + parts             # bare seconds
        elif len(parts) == 2:
            parts = ["0"] + parts                  # MM:SS(.mmm)
    try:
        h, m, sec = int(parts[0]), int(parts[1]), float(parts[2])
    except (ValueError, IndexError):
        return None
    return ((days * 24 + h) * 60 + m) * 60 + sec


def parse_memory(text: Any, default_unit: str = "") -> Tuple[Optional[float], str]:
    """(bytes, per) for a SLURM memory value such as ``9813656K``, ``64G``,
    ``4000Mc`` or ``64Gn``; ``per`` is ``"c"`` (per CPU), ``"n"`` (per node)
    or ``""``. A bare number is in ``default_unit`` ("" = bytes, as in
    MaxRSS; "M" for requests, SLURM's default unit)."""
    m = _MEM_RE.match(str(text or "").strip())
    if not m:
        return None, ""
    try:
        val = float(m.group(1))
    except ValueError:
        return None, ""
    unit = (m.group(2) or default_unit).upper()
    return val * (_UNITS.get(unit) or 1), m.group(3).lower()


def tres_value(tres: Any, key: str) -> Optional[str]:
    """``mem`` from ``billing=32,cpu=32,mem=64G,node=1``."""
    for part in str(tres or "").split(","):
        k, _, v = part.partition("=")
        if k.strip() == key:
            return v.strip()
    return None


def _int(v: Any) -> Optional[int]:
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------ formatting
def _trim(v: float) -> str:
    t = f"{v:.1f}"
    return t[:-2] if t.endswith(".0") else t


def fmt_bytes(b: Optional[float]) -> str:
    if b is None:
        return "?"
    for unit, size in (("TiB", 1024 ** 4), ("GiB", 1024 ** 3), ("MiB", 1024 ** 2)):
        if b >= size:
            v = b / size
            return f"{_trim(v)} {unit}" if v < 100 else f"{v:.0f} {unit}"
    return f"{b / 1024:.0f} KiB"


def fmt_hours(s: Optional[float]) -> str:
    if s is None:
        return "?"
    h = s / 3600.0
    if h < 1:
        return f"{s / 60:.0f} min"
    return f"{_trim(h)} h" if h < 100 else f"{h:,.0f} h"


def _pct(v: float) -> str:
    return f"{v:.0f}%" if v >= 10 else f"{v:.1f}%"


def _cores(v: float) -> str:
    return f"{v:.0f}" if v >= 10 else f"{v:.1f}".rstrip("0").rstrip(".")


# ------------------------------------------------------------------ requested memory
def requested_memory(main: Dict[str, Any], ncpus: Optional[int], nnodes: Optional[int]) -> Tuple[Optional[float], str]:
    """(bytes, source) the job asked for in total."""
    for field in ("AllocTRES", "ReqTRES"):
        v = tres_value(main.get(field), "mem")
        if v:
            b, _ = parse_memory(v, "M")
            if b:
                return b, field
    req = str(main.get("ReqMem") or "").strip()
    b, per = parse_memory(req, "M")
    if not b:
        return None, ""
    if per == "c":
        if not ncpus:
            return None, ""
        return b * ncpus, f"ReqMem {req} x {ncpus} CPUs"
    if per == "n":
        n = nnodes or 1
        return b * n, f"ReqMem {req} x {n} node(s)" if n > 1 else f"ReqMem {req}"
    return b, f"ReqMem {req}"


# ------------------------------------------------------------------ the summary
def _step_rows(acct: Dict[str, Dict[str, Any]], job_id: str) -> Dict[str, Dict[str, Any]]:
    return {k: v for k, v in acct.items() if k.startswith(job_id + ".")}


def _step_cpu(r: Dict[str, Any]) -> Optional[float]:
    t = parse_duration(r.get("TotalCPU"))
    if t:
        return t
    ave = parse_duration(r.get("AveCPU"))
    if ave:
        return ave * max(_int(r.get("NTasks")) or 1, 1)
    return None


def _step_mem(r: Dict[str, Any]) -> Tuple[Optional[float], int]:
    b, _ = parse_memory(r.get("MaxRSS"), "")
    n = max(_int(r.get("NTasks")) or 1, 1)
    return (b * n if b else None), n


def summarize(acct: Dict[str, Dict[str, Any]], job_id: str, live: Optional[Iterable[Dict[str, Any]]] = None,
              elapsed_s: Optional[float] = None, ncpus: Optional[int] = None, nnodes: Optional[int] = None,
              state: Optional[str] = None) -> Dict[str, Any]:
    """CPU and memory use of one job from its ``sacct`` rows (allocation +
    steps) and, for a running job, ``sstat`` rows (``live``)."""
    job_id = str(job_id)
    main = acct.get(job_id) or {}
    state = (state or str(main.get("State") or "").split(" ")[0] or "").upper()
    running = state in ("RUNNING", "COMPLETING", "SUSPENDED")
    if elapsed_s is None:
        elapsed_s = parse_duration(main.get("Elapsed"))
    ncpus = _int(ncpus) or _int(main.get("NCPUS")) or _int(main.get("AllocCPUS")) or \
        _int(tres_value(main.get("AllocTRES"), "cpu"))
    nnodes = _int(nnodes) or _int(main.get("NNodes")) or _int(tres_value(main.get("AllocTRES"), "node"))

    steps = _step_rows(acct, job_id)
    live_rows = {str(r.get("JobID") or f"{job_id}.batch"): r for r in (live or []) if isinstance(r, dict)}
    live_rows = {k: v for k, v in live_rows.items() if k == job_id or k.startswith(job_id + ".")}
    merged = dict(steps)
    merged.update(live_rows)                  # a running step: sstat's numbers are the current ones

    # ---- CPU time
    step_cpu = {k: _step_cpu(r) for k, r in merged.items()}
    # One step cannot use more than the whole allocation; a value beyond it is
    # SLURM's garbage (sstat reports e.g. AveCPU 213503982334-14:25:51 for the
    # extern step of a running job) and is dropped, not added to the others.
    bogus: Dict[str, float] = {}
    if ncpus and elapsed_s:
        cap = PLAUSIBLE_FACTOR * ncpus * elapsed_s
        bogus = {k: v for k, v in step_cpu.items() if v and v > cap}
        for k in bogus:
            step_cpu[k] = None
    cpu_s = sum(v for v in step_cpu.values() if v) or None
    cpu_source = "sstat" if any(step_cpu.get(k) for k in live_rows) else "sacct"
    if not live_rows:
        main_cpu = parse_duration(main.get("TotalCPU"))
        if main_cpu and main_cpu >= (cpu_s or 0):
            cpu_s = main_cpu                  # the allocation line is the sum of its steps
    # ---- memory
    peak, peak_step, peak_tasks = None, None, 1
    for k, r in list(merged.items()) + [(job_id, main)]:
        b, n = _step_mem(r)
        if b and (peak is None or b > peak):
            peak, peak_step, peak_tasks = b, k, n
    mem_source = "sstat" if peak_step in live_rows else "sacct"
    req_b, req_src = requested_memory(main, ncpus, nnodes)

    out: Dict[str, Any] = {
        "cpu": None, "memory": None, "cpu_note": None, "memory_note": None,
        "cpu_text": None, "memory_text": None, "cpu_caveat": None, "advice": [], "details": [],
        "cpu_used_s": cpu_s, "cores_used": None, "elapsed_s": elapsed_s, "ncpus": ncpus, "nnodes": nnodes,
        "peak_bytes": peak, "peak_step": peak_step, "peak_tasks": peak_tasks, "req_bytes": req_b,
        "req_source": req_src, "source": "sstat" if live_rows else "sacct", "running": running,
        # the old field names (kept for callers that read them)
        "max_rss_mb": (peak / 1024 ** 2) if peak else None, "req_mem_mb": (req_b / 1024 ** 2) if req_b else None,
        "total_cpu_s": cpu_s,
    }
    so_far = " so far" if running else ""
    oom = state == "OUT_OF_MEMORY"
    not_started = state in ("PENDING", "CONFIGURING", "REQUEUED")

    # ---- CPU sentence
    if not_started:
        out["cpu_note"] = out["memory_note"] = "The job has not started yet"
    elif not elapsed_s or not ncpus:
        out["cpu_note"] = f"{NOT_MEASURED}: SLURM gives no " + ("elapsed time" if not elapsed_s else "core count")
    elif not cpu_s and bogus:
        out["cpu_note"] = (f"{NOT_MEASURED}: SLURM's CPU time ({fmt_hours(max(bogus.values()))}) is more than "
                           f"{ncpus} cores x {fmt_hours(elapsed_s)} allow")
    elif not cpu_s:
        if running and live is not None and not live_rows:
            out["cpu_note"] = _SSTAT_EMPTY
        elif running and not live_rows:
            out["cpu_note"] = ("SLURM reports a running job's CPU time only when its steps end; "
                               "Show Live Use asks sstat for the current use")
        elif running:
            out["cpu_note"] = f"{NOT_MEASURED}: sstat returned no CPU time for its running steps"
        else:
            out["cpu_note"] = (f"{NOT_MEASURED}: SLURM recorded no CPU time for its steps "
                               "(processes started outside SLURM, e.g. by mpirun's ssh launcher, are not counted)")
    elif cpu_s > PLAUSIBLE_FACTOR * ncpus * elapsed_s:
        out["cpu_note"] = (f"{NOT_MEASURED}: SLURM's CPU time ({fmt_hours(cpu_s)}) is more than "
                           f"{ncpus} cores x {fmt_hours(elapsed_s)} allow")
    else:
        pct = min(100.0 * cpu_s / (ncpus * elapsed_s), 100.0)
        used = cpu_s / elapsed_s
        out["cpu"] = round(pct, 1)
        out["cpu_caveat"] = CPU_CAVEAT
        out["cores_used"] = round(min(used, ncpus), 1)
        out["cpu_text"] = (f"CPU efficiency {_pct(pct)}{so_far} (used about {_cores(min(used, ncpus))} "
                           f"of {ncpus} core{'' if ncpus == 1 else 's'} on average)")
        counted = [k.split(".", 1)[1] for k, v in step_cpu.items() if v and "." in k]
        partial = (nnodes or 1) > 1 and bool(counted) and set(counted) <= {"batch"}
        if partial:
            out["cpu_text"] += "; only processes SLURM tracks are counted, ranks on the other nodes may be missing"
        elif pct < CPU_ADVICE_PCT and elapsed_s >= ADVICE_MIN_ELAPSED_S:
            busy = max(1, round(used))
            out["advice"].append(f"You could request fewer cores (about {busy} {'was' if busy == 1 else 'were'} busy).")

    # ---- memory sentence
    if not out["memory_note"]:
        if not peak:
            if running and live is not None and not live_rows:
                out["memory_note"] = _SSTAT_EMPTY
            elif running and not live_rows:
                out["memory_note"] = ("SLURM reports a running job's peak memory only when its steps end; "
                                      "Show Live Use asks sstat for the current use")
            elif running:
                out["memory_note"] = f"{NOT_MEASURED}: sstat returned no memory use for its running steps"
            else:
                out["memory_note"] = f"{NOT_MEASURED}: SLURM recorded no memory use (MaxRSS) for its steps"
        else:
            est = f", up to {fmt_bytes(peak)} across {peak_tasks} tasks" if peak_tasks > 1 else ""
            if req_b:
                pct = 100.0 * peak / req_b
                out["memory"] = round(pct, 1)
                peak_txt = fmt_bytes(peak / peak_tasks) + " per task" + est if peak_tasks > 1 else fmt_bytes(peak)
                out["memory_text"] = f"Memory peak{so_far} {peak_txt} of {fmt_bytes(req_b)} requested ({_pct(pct)})"
                if oom:
                    pass
                elif pct < MEM_ADVICE_PCT and (elapsed_s or 0) >= ADVICE_MIN_ELAPSED_S and not running:
                    out["advice"].append(f"You could request less memory (the peak was {fmt_bytes(peak)}).")
                elif pct >= MEM_NEAR_LIMIT_PCT:
                    out["advice"].append("Memory use is close to the request; ask for more if the job fails "
                                         "with OUT_OF_MEMORY.")
            else:
                out["memory_text"] = f"Memory peak{so_far} {fmt_bytes(peak)} (the requested memory is unknown)"

    if oom:
        out["advice"].append("The job ran out of memory: request more. SLURM samples memory use, so the real "
                             "peak was higher than the one shown.")

    # ---- formatted raw values (for a details toggle)
    if cpu_s:
        out["details"].append(f"CPU time {fmt_hours(cpu_s)} ({cpu_source})")
    if elapsed_s:
        out["details"].append(f"elapsed {fmt_hours(elapsed_s)}")
    if ncpus:
        out["details"].append(f"{ncpus} CPU{'' if ncpus == 1 else 's'}" + (f" on {nnodes} nodes" if (nnodes or 1) > 1 else ""))
    if peak:
        out["details"].append(f"peak memory {fmt_bytes(peak)} in step {peak_step} ({mem_source}"
                              + (f", {peak_tasks} tasks" if peak_tasks > 1 else "") + ")")
    for k, v in sorted(bogus.items()):
        out["details"].append(f"ignored step {k}: CPU time {fmt_hours(v)} is more than the allocation allows")
    if req_b:
        out["details"].append(f"requested {fmt_bytes(req_b)} ({req_src})")
    out["cpu_line"] = out["cpu_text"] or f"CPU efficiency: {out['cpu_note']}"
    out["memory_line"] = out["memory_text"] or f"Memory: {out['memory_note']}"
    return out


def lines(eff: Dict[str, Any]) -> List[str]:
    """The plain lines both the CLI and the GUI show."""
    if not eff:
        return []
    return ([eff.get("cpu_line") or ""] + ([eff["cpu_caveat"]] if eff.get("cpu_caveat") else [])
            + [eff.get("memory_line") or ""] + list(eff.get("advice") or []))
