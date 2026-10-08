"""Progress, ETA and walltime risk of a running CRYSTAL job, from its .out.

Modes (from MACE's own calc-type detection, ``calc_detection``):

* **opt**: points done, the four criteria and DE against their thresholds,
  and an ETA range from CRYSTAL's own stop rules: the predicted end is the
  earliest future point at which any rule holds (``all_criteria``: all four
  met; ``gradient_and_energy``: both gradients met and |DE| < TOLDEE;
  ``energy_stalled``: |DE| < TOLDEE for four points, shown but never used
  for the ETA: no such streak ended a corpus segment, and it gave a quarter
  of all ranges, 3 in 4 of them wrong), capped by MAXCYCLE. CRYSTAL23 runs FINALRUN 4
  unless the deck sets ICODE 0-2, and 48 of 51 multi-point corpus runs
  restarted at least once, so +1 point (low) / +7 points (high) are added for
  the final check and any further restart segment. Per criterion the trend
  is fit on log10(value/threshold) over the last 6 points of the current
  segment (chosen by grouped cross-validation on the dev half of the real
  OPT corpus, family split, and gated on the other half in
  tests/cluster/test_cluster_eta.py): the high end uses the log-linear
  slope weakened by 2 standard errors and is at least 6x the low end, the
  low end a superlinear (BFGS-like) tail where the slope grows per point.
  When the high end would only be bounded by MAXCYCLE, it is 44 points (the
  dev half's 90th percentile of points left) instead. Fewer than 3 points, a
  flat or rising trend of the governing criterion, or a FINALRUN restart
  with fewer than 2 points in the new segment make the ETA "unknown". Time
  per point is the median of the last 5 point durations from CRYSTAL's
  ``TTTT... TELAPSE`` lines; the low end of the time range uses half of it,
  since the points left run fewer SCF cycles than the ones done.
* **scf**: cycles done, |DETOT| against TOLDEE, a log-linear fit on the
  running minimum of log10(|DETOT|/TOLDEE) for the cycles left (see
  scf_progress; once |DETOT| is below TOLDEE one more cycle, since CRYSTAL
  stops after two such cycles in a row), time per cycle from repeated
  timing sections.
* **freq**: stages in CRYSTAL's order (freq_stages): an optional
  preoptimization; for IR/Raman by CPHF a reference SCF, CPKS, CP2 (Raman)
  and the tensors; the central point; the displacements; the frequencies
  and output. A displacement's cost is never extrapolated from the central
  point (displaced cells have lower symmetry; 6x the central point on a
  corpus run). Percent is shown for the displacement stage only, and never
  100 % before CRYSTAL's end marker. FREQINFO.DAT restarts take their rate
  from the rows computed in the restarted run.
* **properties**: which block runs; no ETA.

``progress()["levels"]`` holds the estimate as nested levels (the total,
the stages, the running inner loop), each with a sentence from
``level_text()``; ``progress()["eta"]`` carries the total's numbers.

The .out is read through ``mace.utils.livefile`` (complete lines of a
snapshot) and parsed by ``mace.utils.out_convergence`` when those are
installed; otherwise a small built-in reader of the same lines is used.
"""
from __future__ import annotations

import copy
import math
import os
import re
import sqlite3
import statistics
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

CRITERIA = ("max_grad", "rms_grad", "max_disp", "rms_disp")
# points_to_reach() defaults: a 3-point log-linear fit, the high end weakened by
# 5 standard errors, a x2-per-point tail for the low end. OPT (grouped 5-fold
# cross-validation over the dev families of the real OPT corpus, family split;
# tests/cluster/test_cluster_eta.py holds out the other half as the gate): a
# 6-point fit, 2 standard errors, no 3-point tail gate, the high end at least
# HIGH_FLOOR x the low end.
WINDOW = 3
TAIL_FACTOR = 2.0
SIGMA = 5.0
OPT_WINDOW, OPT_SIGMA = 6, 2.0
HIGH_FLOOR = 6.0
FINALRUN_LOW, FINALRUN_HIGH = 1, 7   # points the FINALRUN 4 check / restarts add
PRIOR_HIGH = 44                      # dev p90 of points left: high end when only MAXCYCLE bounds it
TIME_LOW_FACTOR = 0.5                # later points run fewer SCF cycles than the recent median
TIME_WINDOW = 5
# SCF of a single point (grouped 5-fold CV on the dev half of the corpus SPs,
# family split; tests/cluster/test_cluster_eta.py holds out the other half):
SCF_WINDOW, SCF_SIGMA = 10, 3.0
SCF_HIGH_FLOOR = 8.0                 # high end at least 8x the low end
SCF_PRIOR_HIGH = 30                  # dev p90 of cycles left: high end when the trend gives none
SCF_TIME_LOW_FACTOR = 0.75
OPT_CYCLE_WINDOW = 5                 # next point's SCF cycles: [min-1, max+2] of the last 5 points
FAIR_MAX_WIDTH = 3.0                 # "fair" needs a trend-bounded range at most 3x wide
FREQ_LOW_FACTOR, FREQ_HIGH_FACTOR = 0.75, 1.5   # displacements left x mean so far (kept after the backtest)
FREQ_OUTPUT_MIN_S = 30.0             # after the last displacement: <= 8.5 s in all 30 corpus runs
DEFAULT_TOLDEE_SCF_FREQ = 1e-9       # FREQCALC's SCF default (CRYSTAL23 manual 8.1.3, p.212; the TOLDEE
                                     # table on p.130 says 10; every corpus FREQ deck sets TOLDEE 11)
LEVELS_VERSION = 1
MAX_SCRATCH_READ = 50 * 1024 * 1024
DEFAULT_TOLDEE_OPT = 1e-7
DEFAULT_MAXCYCLE = 100
DEFAULT_TOLDEE_SCF = 1e-6

_TELAPSE = re.compile(r"^\s*T{10,}\s+(\S+)\s+TELAPSE\s+([\d.]+)\s+TCPU\s+([\d.]+)")
_POINT = re.compile(r"OPTIMIZATION - POINT\s+(\d+)\s*$")
_ENERGY_DE = re.compile(r"TOTAL ENERGY\((?:DFT|HF)\)\(AU\)\(\s*\d+\)\s+\S+(?:\s+DE \(AU\)\s+(\S+))?")
_ENERGY_NCYC = re.compile(r"TOTAL ENERGY\((?:DFT|HF)\)\(AU\)\(\s*(\d+)\)")
_STARTING = re.compile(r"EEEEEEEEEE STARTING\s+DATE\s+(\d+)\s+(\d+)\s+(\d+)\s+TIME\s+(\d+):(\d+):([\d.]+)")
_CP_DIR = re.compile(r"ELECTRIC FIELD APPLIED ALONG CARTESIAN DIRECTION\s+([XYZ])\s*$")
_CP2_DIR = re.compile(r"ELECTRIC FIELD APPLIED ALONG CARTESIAN DIRECTIONS\s+([XYZ]{2})\s*$")
_CP_CYCLE = re.compile(r"^\s*CYCLE\s+(\d+)\s+(?:ALPHA|PSEUDO-ENERGY DERIVATIVE)\s.*DELTA\s+(\S+)")
_CP_TOL = re.compile(r"CP2?_DRIVER \*+ CONVERGENCE ON (?:ALPHA|PSEUDO DE):\s*10\*\*\s*(-?\d+)")
_ROW_NCYC = re.compile(r"^\s+\d+\s+\S+\s+D[XYZ]\s+\S+\s+\S+\s+(\d+)\s+\S+\s+\d+\s*$")
_CRIT = re.compile(r"^\s*(MAX|RMS) (GRADIENT|DISPLAC\.)\s+(\S+)\s+THRESHOLD\s+(\S+)\s+CONVERGED\s+(YES|NO)")
_HDR_DE = re.compile(r"THRESHOLD ON ENERGY CHANGE\s+(\S+)")
_HDR_STEPS = re.compile(r"MAXIMUM ALLOWED NUMBER OF STEPS\s+(\d+)")
_CYC = re.compile(r"^\s*CYC\s+(\d+)\s+ETOT\(AU\)\s+(\S+)\s+DETOT\s+(\S+)")
_SCF_TOLDEE = re.compile(r"CONVERGENCE ON ENERGY\s+10\*\*-\s*(\d+)")
_NCALC = re.compile(r"NUMBER OF SCF\+GRADIENT CALCULATIONS\s+(\d+)")
_DISP_ROW = re.compile(r"^\s+\d+\s+\S+\s+D[XYZ]\s+(\S+)")
_CRIT_KEY = {("MAX", "GRADIENT"): "max_grad", ("RMS", "GRADIENT"): "rms_grad",
             ("MAX", "DISPLAC."): "max_disp", ("RMS", "DISPLAC."): "rms_disp"}


def _f(x) -> Optional[float]:
    try:
        v = float(str(x).replace("D", "E"))
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------ reading
def read_text(path: Path) -> Tuple[str, Dict[str, Any]]:
    """Complete lines of the file now (a half-written last line is left out),
    and the snapshot facts. Uses livefile when installed."""
    path = Path(path)
    try:
        from mace.utils import livefile               # PLAN_analysis C18
        snap = livefile.snapshot(path)
        data, _ = livefile.read_complete(path, snap, 0)
        return data.decode("utf-8", "replace"), {"size": snap.size, "token": snap.token(), "via": "livefile",
                                                 "mtime": snap.mtime_ns / 1e9}
    except ImportError:
        pass
    st = os.stat(path)
    with open(path, "rb") as fh:
        data = fh.read(st.st_size)
    k = data.rfind(b"\n")
    data = data[:k + 1] if k >= 0 else b""
    return data.decode("utf-8", "replace"), {"size": st.st_size, "token": f"{st.st_ino}:{st.st_size}",
                                             "via": "read", "mtime": st.st_mtime}


def timings(lines: Sequence[str]) -> List[Tuple[int, str, float]]:
    """(line index, section, TELAPSE seconds) of CRYSTAL's timing lines."""
    out = []
    for i, line in enumerate(lines):
        m = _TELAPSE.match(line)
        if m:
            out.append((i, m.group(1), float(m.group(2))))
    return out


def _mini_parse(lines: Sequence[str]) -> Dict[str, Any]:
    """The subset of out_convergence.parse() progress needs (used only when
    that module is not installed)."""
    points: List[Dict[str, Any]] = []
    cur = None
    toldee = maxcyc = None
    restarts = []
    scf = []
    scf_toldee = None
    for i, line in enumerate(lines):
        m = _POINT.search(line)
        if m:
            cur = {"point": int(m.group(1)), "line": i + 1, "de_printed": None, "energy_line": None,
                   **{k: None for k in CRITERIA}, "thr": {}, "converged": {}}
            points.append(cur)
            continue
        if "FINALRUN" in line and re.search(r"RE-?START", line.upper()) and "INFORMATION" not in line:
            restarts.append(i + 1)
        m = _ENERGY_DE.search(line)
        if m and cur is not None and cur["energy_line"] is None:
            cur["energy_line"] = i + 1
            cur["de_printed"] = _f(m.group(1)) if m.group(1) else None
            mc = _ENERGY_NCYC.search(line)
            cur["scf_cycles"] = int(mc.group(1)) if mc else None
            continue
        m = _CRIT.match(line)
        if m and cur is not None:
            k = _CRIT_KEY[(m.group(1), m.group(2))]
            cur[k] = _f(m.group(3))
            cur["thr"][k] = _f(m.group(4))
            cur["converged"][k] = m.group(5) == "YES"
            continue
        m = _HDR_DE.search(line)
        if m and toldee is None:
            toldee = _f(m.group(1))
        m = _HDR_STEPS.search(line)
        if m and maxcyc is None:
            maxcyc = int(m.group(1))
        m = _CYC.match(line)
        if m:
            scf.append({"cycle": int(m.group(1)), "etot": _f(m.group(2)), "detot": _f(m.group(3)), "line": i + 1})
        m = _SCF_TOLDEE.search(line)
        if m and scf_toldee is None:
            scf_toldee = 10.0 ** -int(m.group(1))
    done = [p for p in points if p["energy_line"] is not None and any(p[k] is not None for k in CRITERIA[:2])]
    segs = 0
    for p in done:
        p["segment"] = sum(1 for r in restarts if r < p["line"])
        segs = max(segs, p["segment"])
    thr = {"toldee_opt": {"value": toldee or DEFAULT_TOLDEE_OPT},
           "maxcycle": {"value": maxcyc or DEFAULT_MAXCYCLE},
           "toldee_scf": {"value": scf_toldee or DEFAULT_TOLDEE_SCF}}
    return {"points": done, "thresholds": thr, "scf": {"detail": scf}, "via": "builtin"}


def parse_out(path: Path, text: str, size: int) -> Dict[str, Any]:
    """C1's parse when installed (same points, thresholds and stop rules as
    the Run Analysis view), else the built-in reader."""
    try:
        from mace.utils import out_convergence           # PLAN_analysis C1
    except ImportError:
        return _mini_parse(text.splitlines())
    res = out_convergence.parse(path, size=size, complete_lines_only=True)
    res["via"] = "out_convergence"
    # C1's per-cycle SCF details for the SCF mode
    if not (res.get("scf") or {}).get("detail"):
        res.setdefault("scf", {})["detail"] = _mini_parse(text.splitlines())["scf"]["detail"]
    return res


# ------------------------------------------------------------------ fits
def _fit(ys: Sequence[float]) -> Tuple[Optional[float], float]:
    """Least-squares slope of ys against 0..n-1 and its standard error."""
    n = len(ys)
    if n < 2:
        return None, 0.0
    xs = list(range(n))
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    if n < 3:
        return slope, abs(slope) * 0.5
    resid = [y - (my + slope * (x - mx)) for x, y in zip(xs, ys)]
    s2 = sum(r * r for r in resid) / (n - 2)
    return slope, math.sqrt(s2 / sxx)


def points_to_reach(ys: Sequence[float], window: int = WINDOW, sigma: float = SIGMA,
                    tail_gate: bool = True) -> Tuple[Optional[float], Optional[float], str]:
    """(low, high, state) points until log10(value/threshold) < 0.
    state: "met" | "trend" | "plateau" | "short"."""
    ys = [y for y in ys if y is not None]
    if not ys:
        return None, None, "short"
    last = ys[-1]
    if last < 0:
        return 0.0, 0.0, "met"
    w = ys[-window:]
    if len(w) < 3:
        return None, None, "short"
    slope, se = _fit(w)
    tail = _fit(ys[-3:])[0] if tail_gate else None
    if slope is None or slope >= 0 or (tail is not None and tail >= 0):
        return None, None, "plateau"
    s_hi = slope + sigma * se           # weaker descent -> more points
    high = math.inf if s_hi >= 0 else math.ceil(last / -s_hi)
    # superlinear tail: the descent per point grows x TAIL_FACTOR
    acc, k, step = 0.0, 0, -slope
    while acc < last and k < 1000:
        k += 1
        step *= TAIL_FACTOR
        acc += step
    low = float(max(1, k))
    high = max(low, high)
    return low, high, "trend"


def _log_ratio(v, thr) -> Optional[float]:
    if v is None or thr in (None, 0) or v <= 0:
        return None if v is None else (-10.0 if v == 0 else None)
    return math.log10(abs(v) / thr)


def stop_rules(points: Sequence[Dict[str, Any]], thresholds: Dict[str, Any]) -> Dict[str, Any]:
    """Points left until each CRYSTAL stop rule holds, as (low, high, state);
    the same rules as out_convergence's end reason."""
    seg = points[-1].get("segment", 0) if points else 0
    cur = [p for p in points if p.get("segment", 0) == seg]
    toldee = ((thresholds or {}).get("toldee_opt") or {}).get("value") or DEFAULT_TOLDEE_OPT
    per: Dict[str, Tuple] = {}
    for k in CRITERIA:
        ys = [_log_ratio(p.get(k), (p.get("thr") or {}).get(k)) for p in cur]
        per[k] = points_to_reach(ys, OPT_WINDOW, OPT_SIGMA, tail_gate=False)
    de = [_log_ratio(abs(p["de_printed"]) if p.get("de_printed") is not None else None, toldee) for p in cur]
    per["energy"] = points_to_reach([y for y in de if y is not None], OPT_WINDOW, OPT_SIGMA, tail_gate=False)

    def combine(keys):
        lows, highs, states = [], [], []
        for k in keys:
            lo, hi, st = per[k]
            states.append(st)
            if lo is None:
                return None, None, st, k
            lows.append(lo)
            highs.append(hi)
        worst = max(range(len(keys)), key=lambda i: highs[i])
        return max(lows), max(highs), "trend", keys[worst]
    rules = {"all_criteria": combine(list(CRITERIA)),
             "gradient_and_energy": combine(["max_grad", "rms_grad", "energy"])}
    lo, hi, st = per["energy"]
    if lo is not None:
        # four points in a row below TOLDEE: the first one, then three more
        recent = [abs(p["de_printed"]) < toldee for p in cur[-4:] if p.get("de_printed") is not None]
        streak = 0
        for ok in reversed(recent):
            if not ok:
                break
            streak += 1
        need = max(0, 4 - streak)
        rules["energy_stalled"] = (lo + max(0, need - 1) if lo else float(need),
                                   hi + max(0, need - 1) if hi else float(need), "trend", "energy")
    else:
        rules["energy_stalled"] = (None, None, st, "energy")
    return {"criteria": per, "rules": rules, "segment": seg, "points_in_segment": len(cur)}


def _fmt_dur(s: Optional[float]) -> str:
    if s is None:
        return "unknown"
    s = int(round(s))
    d, rem = divmod(s, 86400)
    h, rem = divmod(rem, 3600)
    m = rem // 60
    if d:
        return f"{d} d {h} h"
    if h:
        return f"{h} h {m} min"
    if m:
        return f"{m} min"
    return f"{s} s"


RULE_WORDS = {"all_criteria": "all four criteria", "gradient_and_energy": "gradient + energy rule",
              "energy_stalled": "energy-change rule", "maxcycle": "MAXCYCLE"}


def finalrun_from_deck(deck_text: str) -> bool:
    """CRYSTAL23's default is FINALRUN 4; ICODE 0-2 never restarts the
    optimization (no deck: the default)."""
    return not re.search(r"^\s*FINALRUN\s*\n\s*[0-2]\b", deck_text or "", re.M | re.I)


def opt_eta(parsed: Dict[str, Any], lines: Sequence[str], finalrun: bool = True) -> Dict[str, Any]:
    points = parsed.get("points") or []
    thr = parsed.get("thresholds") or {}
    done = len(points)
    maxcyc = int(((thr.get("maxcycle") or {}).get("value")) or DEFAULT_MAXCYCLE)
    eta = {"low_points": None, "high_points": None, "low_s": None, "high_s": None, "rule": None,
           "basis": None, "confidence": "unknown", "per_point_s": None, "capped": False}
    # time per point from the timing lines at each POINT header
    tl = timings(lines)
    headers = [i for i, line in enumerate(lines) if _POINT.search(line)]
    starts = []
    for h in headers:
        before = [t for (i, _, t) in tl if i < h]
        starts.append(before[-1] if before else None)
    durs = [b - a for a, b in zip(starts, starts[1:]) if a is not None and b is not None and b >= a]
    per_point = statistics.median(durs[-TIME_WINDOW:]) if durs else None
    eta["per_point_s"] = per_point
    if done < 3:
        eta["basis"] = f"{done} point{'s' if done != 1 else ''} done; an estimate needs 3"
        return eta
    sr = stop_rules(points, thr)
    if sr["segment"] > 0 and sr["points_in_segment"] < 2:
        eta["basis"] = "a FINALRUN restart began; an estimate needs 2 points of the new segment"
        return eta
    best = None
    for name, (lo, hi, st, gov) in sr["rules"].items():
        if lo is None or name == "energy_stalled":
            continue
        if best is None or hi < best[2] or (hi == best[2] and lo < best[1]):
            best = (name, lo, hi, gov)
    left_cap = max(0, maxcyc - done)
    if best is None:
        crit = sr["rules"]["gradient_and_energy"]
        eta["basis"] = (f"the {crit[3].replace('_', ' ')} trend is flat or rising over the last points"
                        if crit[2] == "plateau" else "the trend is not clear yet")
        return eta
    name, lo, hi, gov = best
    if math.isfinite(hi):
        hi = max(hi, math.ceil(HIGH_FLOOR * max(lo, 1)))
    prior_high = not math.isfinite(hi) or hi >= left_cap
    if prior_high:
        hi = max(lo, PRIOR_HIGH)
    x_lo, x_hi = (FINALRUN_LOW, FINALRUN_HIGH) if finalrun else (0, 0)
    lo, hi = min(lo, left_cap) + x_lo, min(hi, left_cap) + x_hi
    rule = name if hi < left_cap + x_hi or left_cap == 0 else "maxcycle"
    eta.update(low_points=int(lo), high_points=int(hi) if math.isfinite(hi) else None, rule=rule,
               capped=bool(prior_high or rule == "maxcycle"))
    if per_point is not None and math.isfinite(hi):
        eta["low_s"], eta["high_s"] = lo * per_point * TIME_LOW_FACTOR, hi * per_point
    eta["confidence"] = "rough" if prior_high else "fair"
    words = RULE_WORDS.get(rule, rule) + ("; upper end from typical runs" if prior_high else "")
    if lo == 0 and hi == 0:
        eta["basis"] = f"the next point should end it ({words})"
    elif per_point is not None and eta["high_s"] is not None:
        eta["basis"] = (f"about {int(lo)}-{int(hi)} more points ({_fmt_dur(eta['low_s'])} - "
                        f"{_fmt_dur(eta['high_s'])} at {_fmt_dur(per_point)} per point, from the last "
                        f"{min(TIME_WINDOW, len(durs))} points; {words})")
    else:
        eta["basis"] = f"about {int(lo)}-{int(hi)} more points ({words}; time per point unknown)"
    return eta


# ------------------------------------------------------------------ SCF
def scf_cycles(parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The running SCF's cycles as flat dicts (cycle, etot, detot, line).
    out_convergence gives a list of SCF blocks ({"cycles": [{"cyc", ...}]});
    the built-in reader gives the cycles of the whole file flat."""
    detail = (parsed.get("scf") or {}).get("detail") or []
    if detail and isinstance(detail[-1], dict) and "cycles" in detail[-1]:
        flat = [{"cycle": c.get("cyc", c.get("cycle")), "etot": c.get("etot"), "detot": c.get("detot"),
                 "line": c.get("line")} for c in (detail[-1].get("cycles") or [])]
    else:
        flat = [d for d in detail if isinstance(d, dict)]
        zeros = [i for i, d in enumerate(flat) if d.get("cycle") == 0]
        if zeros:
            flat = flat[zeros[-1]:]                   # the last SCF only
    return [d for d in flat if d.get("detot") is not None]


def scf_progress(parsed: Dict[str, Any], lines: Sequence[str]) -> Dict[str, Any]:
    """Cycles done and cycles left of the running SCF. The cycles left come
    from the running minimum of log10(|DETOT|/TOLDEE) (a 10-cycle fit, the
    high end weakened by 3 standard errors and at least SCF_HIGH_FLOOR x the
    low end); an open high end becomes SCF_PRIOR_HIGH (the dev corpus' 90th
    percentile of cycles left) and is flagged as capped; the deck's SCF
    MAXCYCLE caps it too. Once the last |DETOT| is below TOLDEE the SCF ends
    after the next cycle (1-2 cycles left). Chosen by grouped cross-validation on the dev half
    of the corpus SPs; see docs/cli.md for the held-out scores."""
    cyc = scf_cycles(parsed)
    thr = parsed.get("thresholds") or {}
    toldee = (thr.get("toldee_scf") or {}).get("value") or DEFAULT_TOLDEE_SCF
    maxcyc = (thr.get("scf_maxcycle") or {}).get("value")
    out = {"cycles": len(cyc), "last_detot": cyc[-1]["detot"] if cyc else None, "toldee": toldee,
           "low_cycles": None, "high_cycles": None, "per_cycle_s": None, "low_s": None, "high_s": None,
           "basis": None, "confidence": "unknown", "capped": False, "maxcycle": maxcyc, "flags": []}
    tl = timings(lines)
    by = {}
    for _, name, t in tl:
        by.setdefault(name, []).append(t)
    deltas = []
    for name in ("FDIK", "PDIG", "SHELLXN"):
        ts = by.get(name) or []
        if len(ts) >= 2:
            deltas = [b - a for a, b in zip(ts, ts[1:]) if b >= a]
            break
    if deltas:
        out["per_cycle_s"] = statistics.median(deltas[-TIME_WINDOW:])
    ys, best = [], None
    for d in cyc:
        y = _log_ratio(abs(d["detot"]), toldee)
        if y is None:
            continue
        best = y if best is None else min(best, y)
        ys.append(best)
    lo, hi, st = points_to_reach(ys, SCF_WINDOW, SCF_SIGMA)
    if lo is None:
        out["basis"] = ("fewer than 3 cycles" if st == "short" else "the energy change is not falling steadily")
        return out
    capped = False
    if st == "met":
        # CRYSTAL ends the SCF after two cycles in a row with |DE| < TOLDEE: once the
        # last |DE| is below, one more cycle (all 23 such cuts of the dev SPs had
        # exactly 1 left); a bounce back above it adds a few
        below = [abs(d["detot"]) < toldee for d in cyc[-2:]]
        if below and below[-1]:
            lo, hi = (0.0, 1.0) if len(below) == 2 and below[0] else (1.0, 2.0)
        else:
            lo, hi = 1.0, SCF_HIGH_FLOOR
    elif math.isfinite(hi):
        hi = max(hi, math.ceil(SCF_HIGH_FLOOR * max(lo, 1)))
    else:
        hi, capped = max(lo, SCF_PRIOR_HIGH), True
    if maxcyc and len(cyc) + hi > maxcyc:
        hi = max(lo, maxcyc - len(cyc))
        out["flags"].append("maxcycle_cap")
    out["low_cycles"], out["high_cycles"], out["capped"] = int(lo), int(hi), capped
    if out["per_cycle_s"]:
        out["low_s"], out["high_s"] = lo * out["per_cycle_s"] * SCF_TIME_LOW_FACTOR, hi * out["per_cycle_s"]
        if st == "met":
            out["low_s"] = 0.0          # the last cycle's work may be all but done (its CYC line comes last)
    out["confidence"] = "fair" if not capped and hi <= FAIR_MAX_WIDTH * max(lo, 1) else "rough"
    out["basis"] = (f"about {_nspan(int(lo), int(hi))} more cycle{'s' if hi != 1 else ''} (|DE| "
                    f"{_sci(abs(out['last_detot']))} vs TOLDEE {_sci(toldee)}"
                    f"{'; upper end from typical runs' if capped else ''})")
    return out


# ------------------------------------------------------------------ FREQ
def freq_requests_intensities(text: str, deck_text: str = "") -> bool:
    if "NO INTENSITY CALCULATION" in text and not deck_text:
        return False
    recs = {l.strip().upper() for l in (deck_text or "").splitlines()[1:]}
    if recs & {"INTENS", "INTRAMAN", "INTCPHF"}:
        return True
    return bool(re.search(r"^\s*INFORMATION \*+ (INTENS|INTRAMAN|INTCPHF)", text, re.M)) or \
        ("INTENSITIES COMPUTED" in text and "NO INTENSITY CALCULATION" not in text)


def scratch_displacements(scratch: Optional[Path]) -> Optional[int]:
    """Displacement SCFs finished, from SCFOUT.LOG in the job's scratch dir
    (each displaced point's SCF ends with an "SCF ENDED" line there). None
    when it cannot be read. Read-only, at most 50 MB from the end."""
    if scratch is None:
        return None
    f = Path(scratch) / "SCFOUT.LOG"
    try:
        size = f.stat().st_size
        with open(f, "rb") as fh:
            if size > MAX_SCRATCH_READ:
                fh.seek(size - MAX_SCRATCH_READ)
            data = fh.read()
    except OSError:
        return None
    return data.count(b"SCF ENDED")


def freq_stages(lines: Sequence[str], deck_text: str = "") -> Dict[str, Any]:
    """Where each FREQ stage starts, by line index, in the order CRYSTAL23
    prints them (checked on the 9 corpus runs with CPHF): an optional
    preoptimization; for IR/Raman by CPHF a reference SCF, CPKS along X/Y/Z
    (CP_DRIVER), CP2 along XX..ZZ (Raman only, CP2_DRIVER), the IR/Raman
    tensors; then the frequency header (NUMBER OF SCF+GRADIENT
    CALCULATIONS), the central point's own SCF and gradient (to the first
    TOTGRA), the displacement table. FREQINFO.DAT restarts have no SCF and
    no TOTGRA."""
    st: Dict[str, Any] = {"points": [], "cyc0": [], "scf_end": [], "cp_driver": None, "cp_dirs": [],
                          "cp2_driver": None, "cp2_dirs": [], "cp_cycles": [], "cp_tol": None, "tensors": [],
                          "header": None, "totgra": None, "table": None, "restart": False}
    for i, line in enumerate(lines):
        if "OPTIMIZATION - POINT" in line and _POINT.search(line):
            st["points"].append(i)
        elif "CYC   0 " in line and _CYC.match(line):
            st["cyc0"].append(i)
        elif "== SCF ENDED" in line:
            st["scf_end"].append(i)
        elif "CP_DRIVER" in line and "CP2_DRIVER" not in line:
            if st["cp_driver"] is None:
                st["cp_driver"] = i
            m = _CP_TOL.search(line)
            if m:
                st["cp_tol"] = 10.0 ** int(m.group(1))
        elif "CP2_DRIVER" in line:
            if st["cp2_driver"] is None:
                st["cp2_driver"] = i
        elif "ELECTRIC FIELD APPLIED ALONG" in line:
            m = _CP2_DIR.search(line)
            if m:
                st["cp2_dirs"].append((i, m.group(1)))
            else:
                m = _CP_DIR.search(line)
                if m:
                    st["cp_dirs"].append((i, m.group(1)))
        elif line.lstrip().startswith("CYCLE") and _CP_CYCLE.match(line):
            m = _CP_CYCLE.match(line)
            st["cp_cycles"].append((i, int(m.group(1)), _f(m.group(2))))
        elif "COMPUTING" in line and "TENSOR ALONG" in line:
            st["tensors"].append(i)
        elif st["header"] is None and _NCALC.search(line):
            st["header"] = i
        elif st["totgra"] is None and st["header"] is not None and "TOTGRA" in line and _TELAPSE.match(line):
            st["totgra"] = i                      # the central point's gradient (a preoptimization has its own)
        elif st["table"] is None and "MAX ABS(DGRAD)" in line and "ATOM" in line and "TOTAL ENERGY" in line:
            st["table"] = i
        if "FREQUENCY CALCULATION RESTART" in line or "RESTORED FROM THE FREQINFO.DAT" in line:
            st["restart"] = True
    recs = {l.strip().upper() for l in (deck_text or "").splitlines()[1:]}
    # a FREQINFO.DAT restart prints no CPHF stage unless it reruns one
    deck_cphf = bool(recs & {"INTCPHF", "INTRAMAN"}) and not st["restart"]
    st["cphf"] = st["cp_driver"] is not None or deck_cphf
    st["raman"] = st["cp2_driver"] is not None or ("INTRAMAN" in recs and st["cphf"] and not st["restart"])
    hdr = st["header"]
    st["preopt_points"] = [p for p in st["points"] if hdr is None or p < hdr]
    st["central_c0"] = next((c for c in st["cyc0"] if hdr is not None and c > hdr), None)
    first_cphf_line = st["cp_driver"] if st["cp_driver"] is not None else hdr
    st["ref_c0"] = None
    if st["cphf"]:
        cands = [c for c in st["cyc0"] if (first_cphf_line is None or c < first_cphf_line)
                 and (not st["preopt_points"] or c > st["preopt_points"][-1])]
        st["ref_c0"] = cands[0] if cands else None
    if st["table"] is not None or st["totgra"] is not None:
        phase = "central_done"
    elif st["central_c0"] is not None or (hdr is not None and all(i < hdr for i in st["tensors"])
                                           and (st["cp_driver"] is None or st["cp_driver"] < hdr)):
        phase = "central"
    elif st["tensors"]:
        phase = "tensors"
    elif st["cp2_driver"] is not None:
        phase = "cp2"
    elif st["cp_driver"] is not None:
        phase = "cphf"
    elif st["ref_c0"] is not None:
        phase = "ref_scf"
    elif st["preopt_points"] and hdr is None:
        phase = "preopt"
    else:
        phase = "central" if hdr is not None else "setup"
    st["phase"] = phase
    return st


_PHASE_BASIS = {"preopt": "Preoptimization running (levels in a later release)",
                "ref_scf": "Reference SCF for IR/Raman (CPHF/CPKS) running",
                "cphf": "IR/Raman (CPHF/CPKS) running (time unknown)",
                "cp2": "Raman (CPHF/CPKS, second order) running (time unknown)",
                "tensors": "IR/Raman tensors running (time unknown)"}


def freq_progress(text: str, lines: Sequence[str], ended: bool, deck_text: str = "",
                  scratch: Optional[Path] = None, mtime: Optional[float] = None,
                  job_start: Optional[float] = None, stages: Optional[Dict[str, Any]] = None,
                  now: Optional[float] = None) -> Dict[str, Any]:
    """FREQ stages. CRYSTAL prints the displacement table's header and the
    central point's row once the central point ends, then one row per
    displacement as it finishes (measured on a live run): the rows are the
    progress. A time per displacement comes only from displacements finished
    in this run (the file's last change against the end of the central
    point, or, in a FREQINFO.DAT restart, against the end of this run's
    setup), never from the central point's own time."""
    st = stages if stages is not None else freq_stages(lines, deck_text)
    m = _NCALC.search(text)
    total = int(m.group(1)) - 1 if m else None
    table_at = st["table"]
    tl = timings(lines)
    central_end = None
    if st["totgra"] is not None:
        central_end = (st["totgra"], float(_TELAPSE.match(lines[st["totgra"]]).group(2)))
    restored = sum(1 for l in lines if "RESTORED FROM A PREV. HESS. MATRIX" in l)
    computed = 0
    after_table = False
    if table_at is not None:
        for line in lines[table_at:]:
            mm = _DISP_ROW.match(line)
            if mm and _f(mm.group(1)) is not None:
                computed += 1
        after_table = any(i > table_at for i, _, _ in tl)       # the next stage has begun
    intens = freq_requests_intensities(text, deck_text)
    out = {"total": total, "done": None, "percent": None, "stage": "central", "intensities": intens,
           "restored": restored, "computed": computed, "central_s": central_end[1] if central_end else None,
           "low_s": None, "high_s": None, "basis": None, "confidence": "unknown", "restart": st["restart"]}
    done_rows = computed + restored
    if ended:
        out.update(stage="end", basis="finished", done=min(total, done_rows) if total else done_rows or None)
        return out
    if st["phase"] in _PHASE_BASIS:
        out.update(stage=st["phase"], basis=_PHASE_BASIS[st["phase"]])
        return out
    if central_end is None and table_at is None:
        out["basis"] = "Central point (SCF + gradient) running"
        return out
    all_done = total is not None and done_rows >= total
    if table_at is not None and (all_done or after_table):
        out["done"] = min(total, done_rows) if total else done_rows
        # what is left after the table is short: at most 8.5 s after the last row in all 30
        # corpus runs with computed rows; a CPHF run did its IR/Raman stages before the
        # central point, so only the frequencies remain
        if intens and not st["cphf"]:
            out.update(stage="intensities",
                       basis="Displacements done; IR/Raman intensities and frequencies being written")
        else:
            out.update(stage="frequencies", basis="Displacements done; frequencies being written")
        return out
    out["stage"] = "displacements"
    if table_at is not None:
        done = done_rows
    else:
        done = scratch_displacements(scratch)
        if done is not None:
            out["source"] = "scratch SCFOUT.LOG"
    if total:
        done = max(0, min(total, done or 0))
        out["done"] = done
        out["percent"] = round(100.0 * done / total)
    remaining = (total - (out["done"] or 0)) if total is not None else None
    # the rate's anchor: the end of the central point, or this run's setup in a restart
    anchor = central_end[1] if central_end else None
    if anchor is None and st["restart"] and table_at is not None:
        anchor = _t_at(tl, table_at)
    cen_t = central_end[1] if central_end else None
    if central_end and st["central_c0"] is not None:
        # its SCF + gradient, as its level times it: not the setup, not the CPHF stages
        cen_t = central_end[1] - (_t_at(tl, st["central_c0"]) or 0.0)
    cmin = _fmt_dur(cen_t) if cen_t is not None else "unknown"
    if out["done"]:
        out["basis"] = f"Displacement {out['done']} of {total} ({out['percent']} %)"
        # why there is no rate (replaced when there is one)
        out["rate_note"] = ("until the first displacement computed in this run finishes" if not computed else
                            "no job clock to time it (a copied or finished output)" if not job_start else
                            "the job's start does not match the output's timings")
        if computed and mtime and job_start and anchor is not None and remaining is not None:
            per = (mtime - (job_start + anchor)) / computed
            if per > 0:
                out["rate_note"] = None
                out["low_s"], out["high_s"] = per * remaining * FREQ_LOW_FACTOR, per * remaining * FREQ_HIGH_FACTOR
                if now is not None and remaining > 0:
                    # the running displacement writes nothing to the output until its row, so the
                    # time since the output's last change is spent in it: it comes off the low end,
                    # the other displacements left still run in full (as for optimization points)
                    in_disp = max(0.0, now - mtime)
                    out["in_displacement_s"] = in_disp
                    out["low_s"] = max(per * (remaining - 1) * FREQ_LOW_FACTOR, out["low_s"] - in_disp)
                out["confidence"] = "rough"
                out["per_displacement_s"] = per
                out["basis"] += (f"; about {_fmt_dur(out['low_s'])} - {_fmt_dur(out['high_s'])} left at about "
                                 f"{_fmt_dur(per)} per displacement, from the {computed} finished in this run")
                if restored:
                    out["basis"] += f" ({restored} restored from FREQINFO.DAT)"
    else:
        out["done"] = 0 if total is not None else None
        out["percent"] = 0 if total else None
        head = "Central point restored from FREQINFO.DAT" if st["restart"] and not central_end else \
            f"Central point done ({cmin})"
        out["basis"] = (f"{head}; {remaining if remaining is not None else 'some'} "
                        f"displacements remaining; time per displacement unknown until the first finishes "
                        f"(displaced cells have lower symmetry and usually cost several times more than the "
                        f"central point)")
    return out


# ------------------------------------------------------------------ levels
# progress()["levels"]: the estimate as nested levels, outermost first.
# levels[0] is the job's total; depth 1 are the stages of the run (their
# remaining times add up to the total); depth 2 is the inner loop of the
# running stage (shown for orientation, already inside its stage's time).
_CONF_ORDER = {"fair": 0, "rough": 1, "unknown": 2}


def _level(lid: str, depth: int, label: str, status: str = "pending", *, current=None, unit=None,
           low_n=None, high_n=None, exact=None, extra=None, per_unit_s=None, done_s=None,
           low_s=None, high_s=None, confidence="unknown", unknown_reason=None, flags=None,
           thresholds=None, basis="", info=None) -> Dict[str, Any]:
    return {"id": lid, "depth": depth, "label": label, "status": status, "current": current,
            "count": {"low": low_n, "high": high_n, "exact": exact, "unit": unit}, "extra": extra,
            "per_unit_s": per_unit_s, "done_s": done_s, "remaining_s": {"low": low_s, "high": high_s},
            "confidence": confidence, "unknown_reason": unknown_reason, "flags": list(flags or []),
            "thresholds": list(thresholds or []), "basis": basis, "info": dict(info or {}), "text": ""}


def _span(lo, hi) -> str:
    lo_t, hi_t = _fmt_dur(lo), _fmt_dur(hi)
    return hi_t if lo_t == hi_t else f"{lo_t} - {hi_t}"


def _nspan(lo, hi) -> str:
    return f"{hi}" if lo == hi else f"{lo}-{hi}"


def _sci(v: Optional[float]) -> str:
    """1.5e-4, 1e-9 (no exponent padding)."""
    if v is None:
        return "?"
    m, e = f"{v:.1e}".split("e")
    m = m[:-2] if m.endswith(".0") else m
    return f"{m}e{int(e)}"


def _so_far(info: Dict[str, Any], words: str = "so far") -> str:
    """" (12 min so far)" for a running stage without an estimate; "" without the run's clock."""
    s = info.get("so_far_s")
    return f" ({_fmt_dur(s)} {words})" if s is not None else ""


def level_text(lv: Dict[str, Any]) -> str:
    """The one sentence of a level (the CLI and the GUI print this)."""
    i, st, cnt, rem, info = lv["id"], lv["status"], lv["count"], lv["remaining_s"], lv["info"]
    lo_s, hi_s = rem["low"], rem["high"]
    if i == "total":
        if info.get("ended"):
            return f"Total: finished ({info['ended']})"
        if lo_s is None or (hi_s is None and lo_s < 1):
            return f"Total: unknown ({lv['unknown_reason'] or 'no estimate yet'})"
        if hi_s is None:
            return f"Total at least {_fmt_dur(lo_s)} left ({lv['unknown_reason']})"
        return f"Total about {_span(lo_s, hi_s)} left ({lv['confidence']})"
    if i == "opt.points":
        cur = lv["current"]
        if cnt["low"] is None:
            if not cur and st == "running":        # CRYSTAL prints "POINT 1" after the first SCF + gradient
                return f"Optimization: first point running (points left unknown: {lv['unknown_reason']})"
            return f"Optimization point {cur} (points left unknown: {lv['unknown_reason']})"
        if info.get("segment"):
            return f"Final restart point {cur} of about {_nspan(cnt['low'], cnt['high'])}"
        x = lv["extra"]
        tail = f" (+ {x['label']} {_nspan(x['low'], x['high'])})" if x else ""
        return f"Optimization point {cur} of about {_nspan(cnt['low'], cnt['high'])}{tail}"
    if i == "opt.point_scf":
        rng = f"about {_nspan(cnt['low'], cnt['high'])}" if cnt["low"] is not None else "an unknown number of"
        if lv["current"] is not None:
            per = f", ~{_fmt_dur(lv['per_unit_s'])} per cycle" if lv["per_unit_s"] else ""
            of = f" of about {_nspan(cnt['low'], cnt['high'])}" if cnt["low"] is not None else ""
            return f"Next point's SCF: cycle {lv['current']}{of}{per}"
        run = ""
        if info.get("in_point_s") is not None and info.get("point_low_s") is not None:
            if info["in_point_s"] > (info["point_high_s"] or 0):
                run = (f"; running {_fmt_dur(info['in_point_s'])}, longer than the earlier points "
                       f"({_span(info['point_low_s'], info['point_high_s'])} each)")
            else:
                run = (f"; running {_fmt_dur(info['in_point_s'])} of about "
                       f"{_span(info['point_low_s'], info['point_high_s'])} per point")
        elif info.get("point_low_s") is not None:
            run = f"; about {_span(info['point_low_s'], info['point_high_s'])} per point"
        return f"Next point's SCF: {rng} cycles (from earlier points){run}"
    if i == "opt.first_scf":
        de = (f" (|DE| {_sci(info['detot'])}" + (f", threshold {_sci(info['toldee'])})" if info.get("toldee") else ")")
              if info.get("detot") is not None else "")
        return f"First point's SCF: cycle {lv['current']}{de}"
    if i == "sp.setup":
        if st == "done":
            return f"Setup and integrals: done in {_fmt_dur(lv['done_s'])}"
        return f"Setup and integrals: running ({_fmt_dur(info.get('so_far_s'))} so far)"
    if i == "sp.scf":
        if st == "pending":
            return "SCF: pending"
        if st == "done":
            return f"SCF: done in {_fmt_dur(lv['done_s'])} ({lv['current']} cycles)"
        if cnt["low"] is None:
            # the reason already reads "SCF cycles left unknown: ..."
            why = lv["unknown_reason"] or "SCF cycles left unknown"
            txt = f"SCF cycle {lv['current']} ({why})"
        else:
            per = f", ~{_fmt_dur(lv['per_unit_s'])} per cycle" if lv["per_unit_s"] else ""
            de = (f" (|DE| {_sci(info['detot'])}, threshold {_sci(info['toldee'])})"
                  if info.get("detot") is not None else "")
            txt = f"SCF cycle {lv['current']} of about {_nspan(cnt['low'], cnt['high'])}{per}{de}"
            if "maxcycle_cap" in lv["flags"]:
                txt += f" (at most {info['maxcycle']} cycles: MAXCYCLE)"
        if "guessp" in lv["flags"]:
            txt += (" - restart from a previous density (GUESSP): estimate fit on runs without it, "
                    "usually fewer cycles")
        return txt
    if i == "sp.post":
        if st == "done":
            return f"After the SCF: done in {_fmt_dur(lv['done_s'])}"
        if st == "running":
            return f"After the SCF (gradient, analysis, files): running, about {_span(lo_s, hi_s)} left"
        return f"After the SCF (gradient, analysis, files): about {_span(lo_s, hi_s)}"
    if i == "freq.preopt":
        if st == "done":
            return f"Preoptimization: done ({lv['current']} points)"
        return f"Preoptimization: running (optimization point {lv['current']})"
    if i == "freq.ref_scf":
        if st == "done":
            return f"Reference SCF for IR/Raman: done in {_fmt_dur(lv['done_s'])}"
        if st == "running":
            return f"Reference SCF for IR/Raman: running, SCF cycle {lv['current'] or 0}{_so_far(info)}"
        return "Reference SCF for IR/Raman: pending"
    if i in ("freq.cphf", "freq.cp2"):
        name = ("IR/Raman (CPHF/CPKS, first order)" if i == "freq.cphf"
                else "Raman (CPHF/CPKS, second order)")
        if st == "done":
            return f"{name}: done in {_fmt_dur(lv['done_s'])}"
        if st == "pending":
            return f"{name}: pending"
        txt = f"{name}: field direction {info.get('direction', '?')} ({lv['current']} of {cnt['exact']})"
        if info.get("cycle") is not None:
            txt += f", cycle {info['cycle']}"
            if i == "freq.cphf" and info.get("delta") is not None and info.get("tol") is not None:
                txt += f", |DELTA| {_sci(abs(info['delta']))} vs {_sci(info['tol'])}"
        return txt + _so_far(info, "in this stage")
    if i == "freq.tensors":
        return {"done": f"IR/Raman tensors: done in {_fmt_dur(lv['done_s'])}",
                "running": f"IR/Raman tensors: running{_so_far(info)}"}.get(st, "IR/Raman tensors: pending")
    if i == "freq.central":
        if st == "restored":
            return "Central point: restored from FREQINFO.DAT (skipped)"
        if st == "done":
            return f"Central point (SCF + gradient): done in {_fmt_dur(lv['done_s'])}"
        if st == "running":
            if info.get("restart_setup"):
                return "Central point: restarting from FREQINFO.DAT"
            if info.get("scf_done"):
                return "Central point (SCF + gradient): running, gradient"
            if lv["current"] is None:
                return "Central point (SCF + gradient): running, before its first SCF cycle"
            return f"Central point (SCF + gradient): running, SCF cycle {lv['current']}"
        return "Central point (SCF + gradient): pending"
    if i == "freq.displacements":
        total = cnt["exact"]
        if st == "done":
            return f"Displacements: done ({total})"
        if st == "pending":
            return f"Displacements: {total} pending" if total is not None else "Displacements: pending"
        cur = lv["current"] or 0
        if lv["per_unit_s"]:
            pct = f" ({info['percent']} %)" if info.get("percent") is not None else ""
            txt = (f"Displacement {cur} of {total}{pct}, about "
                   f"{_span(lv['per_unit_s'] * FREQ_LOW_FACTOR, lv['per_unit_s'] * FREQ_HIGH_FACTOR)} each")
            if info.get("in_displacement_s") is not None and cur < (total or 0):
                txt += f"; the next running for {_fmt_dur(info['in_displacement_s'])}"
            if info.get("restored"):
                txt += (f" - {info['restored']} restored from FREQINFO.DAT, rate from the "
                        f"{info['computed']} computed in this run")
            return txt
        if cur and info.get("rate_note"):
            pct = f" ({info['percent']} %)" if info.get("percent") is not None else ""
            return f"Displacement {cur} of {total}{pct}: time per displacement unknown ({info['rate_note']})"
        return (f"Displacement {cur} of {total}: time per displacement unknown until the first finishes "
                f"(displaced cells have lower symmetry and usually cost several times more than the central point)")
    if i == "freq.displacement_scf":
        of = f" of about {_nspan(cnt['low'], cnt['high'])}" if cnt["low"] is not None else ""
        return f"Next displacement's SCF: cycle {lv['current']}{of}"
    if i == "freq.output":
        if st == "done":
            return "Frequencies and output: done"
        if st == "running":
            return f"Frequencies and output: running, about {_fmt_dur(hi_s)} or less"
        return f"Frequencies and output: about {_fmt_dur(hi_s)} or less"
    if i == "properties.block":
        return f"Running {lv['current']}" if st == "running" else "Finished"
    return lv["label"]


def _finish_levels(levels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for lv in levels:
        lo, hi = lv["remaining_s"]["low"], lv["remaining_s"]["high"]
        if lo is not None and hi is not None and lo > hi:
            lv["remaining_s"]["low"] = hi
        lv["text"] = level_text(lv)
    return levels


def total_level(stages: Sequence[Dict[str, Any]], ended: Optional[str] = None) -> Dict[str, Any]:
    """Sum of the depth-1 stages still to run: low = sum of the known lows,
    high = sum of the highs only when every remaining stage has one (else
    "at least", naming the first stage without an upper end)."""
    tot = _level("total", 0, "Total", "done" if ended else "running")
    if ended:
        tot["info"]["ended"] = ended
        tot["remaining_s"] = {"low": 0.0, "high": 0.0}
        tot["confidence"] = "fair"
        tot["basis"] = f"finished ({ended})"
        return tot
    rem = [lv for lv in stages if lv["depth"] == 1 and lv["status"] in ("running", "pending")]
    lows = [lv["remaining_s"]["low"] for lv in rem if lv["remaining_s"]["low"] is not None]
    low = sum(lows) if lows else None
    open_ = [lv for lv in rem if lv["remaining_s"]["high"] is None]
    high = None if open_ or not rem else sum(lv["remaining_s"]["high"] for lv in rem)
    if not rem or (high is None and low is not None and low < 1):
        low = None                      # nothing known beyond "a few seconds": unknown, not "at least 0 s"
    reason = None
    if open_:
        o = open_[0]
        reason = o["unknown_reason"] or f"{o['label']} has no time estimate yet"
    elif not rem:
        reason = "no stage to estimate"
    conf = max((lv["confidence"] for lv in rem), key=lambda c: _CONF_ORDER.get(c, 2), default="unknown")
    if high is None:
        conf = "unknown"
    elif conf == "fair" and low and high > FAIR_MAX_WIDTH * low:
        conf = "rough"
    tot.update(confidence=conf, unknown_reason=reason)
    tot["remaining_s"] = {"low": low, "high": high}
    tot["basis"] = _total_basis(tot, rem)
    return tot


def _total_basis(tot: Dict[str, Any], rem: Sequence[Dict[str, Any]]) -> str:
    lo, hi = tot["remaining_s"]["low"], tot["remaining_s"]["high"]
    # a stage whose basis is only the reason already in the head is not said twice
    parts = "; ".join(f"{lv['label']}: {lv['basis']}" for lv in rem
                      if lv.get("basis") and lv["basis"] != tot.get("unknown_reason"))
    if lo is None or (hi is None and lo < 1):
        head = f"time left unknown ({tot['unknown_reason']})"
    elif hi is None:
        head = f"at least {_fmt_dur(lo)} left in total ({tot['unknown_reason']})"
    else:
        head = f"about {_span(lo, hi)} left in total"
    tot["info"]["basis_head"] = head
    return head + (f": {parts}" if parts else "")


def threshold_notes(thresholds: Dict[str, Any], context: str) -> List[Dict[str, Any]]:
    """Each threshold against its CRYSTAL23 default in this context
    ("opt", "sp" or "freq"): display only, the estimates do not use it."""
    try:
        from mace.utils.out_convergence import (CONTEXT_DEFAULTS, DEFAULT_TOLDEE_OPT as OC_TOLDEE_OPT,
                                                DEFAULT_TOLDEE_SCF_OPT, DEFAULT_TOLDEE_SCF_SP)
    except ImportError:                                  # pragma: no cover - module ships with MACE
        CONTEXT_DEFAULTS = {"standard": (0.0003, 0.0012)}
        OC_TOLDEE_OPT, DEFAULT_TOLDEE_SCF_OPT, DEFAULT_TOLDEE_SCF_SP = 1e-7, 1e-7, 1e-6
    thr = thresholds or {}
    rows = []
    if context == "opt":
        rms_g, rms_x = CONTEXT_DEFAULTS.get("standard")
        for key, name, dflt in (("maxgrad", "max gradient", 1.5 * rms_g), ("toldeg", "RMS gradient", rms_g),
                                ("maxdisp", "max displacement", 1.5 * rms_x), ("toldex", "RMS displacement", rms_x),
                                ("toldee_opt", "energy change", OC_TOLDEE_OPT),
                                ("toldee_scf", "SCF energy change", DEFAULT_TOLDEE_SCF_OPT)):
            if key in thr:
                rows.append((key, name, thr[key], dflt, "the default" if key != "toldee_scf"
                             else "the optimization default"))
    else:
        dflt = DEFAULT_TOLDEE_SCF_FREQ if context == "freq" else DEFAULT_TOLDEE_SCF_SP
        where = "the FREQCALC default" if context == "freq" else "the single-point default"
        if "toldee_scf" in thr:
            rows.append(("toldee_scf", "SCF energy change", thr["toldee_scf"], dflt, where))
    out = []
    for key, name, rec, dflt, where in rows:
        v = rec.get("value") if isinstance(rec, dict) else rec
        if v is None:
            continue
        note = None
        if v < dflt * (1 - 1e-6):
            note = f"tighter than {where} {_sci(dflt)}"
        elif v > dflt * (1 + 1e-6):
            note = f"looser than {where} {_sci(dflt)}"
        out.append({"name": name, "key": key, "value": v, "default": dflt,
                    "source": rec.get("source") if isinstance(rec, dict) else None, "note": note})
    return out


def thresholds_line(notes: Sequence[Dict[str, Any]], always: bool = False) -> Optional[str]:
    """"Thresholds: ..." for the thresholds that differ from their default
    (None when none differ, unless ``always``)."""
    diff = [n for n in notes if n.get("note")]
    if not diff:
        return "Thresholds: CRYSTAL23 defaults" if always else None
    return "Thresholds: " + ", ".join(f"{n['name']} {_sci(n['value'])} ({n['note']})" for n in diff)


def run_start(lines: Sequence[str], mtime: Optional[float] = None,
              slurm_start: Optional[float] = None) -> Tuple[Optional[float], Optional[str]]:
    """When CRYSTAL itself started (epoch s) and where that came from. The
    .out's "STARTING DATE ... TIME ..." line (node-local time) is used when it
    lies between 10 min before SLURM's start and the file's last change
    (else a clock or timezone difference is assumed); otherwise SLURM's start
    (now - elapsed); otherwise None. `mace run` (local, no SLURM) runs have only the
    STARTING line."""
    import time as _time
    t = None
    for line in lines[:600]:
        m = _STARTING.search(line)
        if m:
            d, mo, y, h, mi, sec = m.groups()
            try:
                t = _time.mktime((int(y), int(mo), int(d), int(h), int(mi), int(float(sec)), 0, 0, -1))
                t += float(sec) - int(float(sec))
            except (OverflowError, ValueError):
                t = None
            break
    if t is not None:
        lo = slurm_start - 600 if slurm_start is not None else -math.inf
        hi = (mtime if mtime is not None else _time.time()) + 60
        if lo <= t <= hi:
            return t, "crystal"
    if slurm_start is not None:
        return slurm_start, "slurm"
    return None, None


def _read_tail(path: Path, cap: int) -> Optional[bytes]:
    try:
        size = path.stat().st_size
        with open(path, "rb") as fh:
            if size > cap:
                fh.seek(size - cap)
            return fh.read(cap)
    except OSError:
        return None


SCF_LOG_TAIL = 2 * 1024 * 1024


def scf_log_tail(scratch: Optional[Path]) -> Optional[int]:
    """The running SCF's last cycle number from SCFOUT.LOG in the job's
    scratch dir (read-only, the last 2 MB): the last "CYC n" after the last
    "CYC 0". None when it cannot be read or holds no such lines, and when
    that SCF has ended: between "SCF ENDED" and the next "CYC 0" CRYSTAL
    computes the gradient (minutes for a displaced cell), and the finished
    SCF's last cycle is not a running one."""
    if scratch is None:
        return None
    data = _read_tail(Path(scratch) / "SCFOUT.LOG", SCF_LOG_TAIL)
    if not data:
        return None
    try:
        lines = data.decode("utf-8", "replace").splitlines()
        at = [(i, int(m.group(1))) for i, m in ((i, _CYC.match(l)) for i, l in enumerate(lines)) if m]
        cyc = [c for _, c in at]
    except (ValueError, TypeError):
        return None
    if not cyc or 0 not in cyc:
        return None
    if any("SCF ENDED" in l for l in lines[at[-1][0] + 1:]):
        return None
    last0 = len(cyc) - 1 - cyc[::-1].index(0)
    run = cyc[last0:]
    return run[-1] if run == sorted(run) else None


def _cycle_window(counts: Sequence[int]) -> Tuple[Optional[int], Optional[int]]:
    w = [c for c in counts if c][-OPT_CYCLE_WINDOW:]
    if not w:
        return None, None
    return max(1, min(w) - 1), max(w) + 2


def _t_at(tl: Sequence[Tuple[int, str, float]], idx: Optional[int]) -> Optional[float]:
    """TELAPSE of the last timing line at or before line ``idx``."""
    if idx is None:
        return None
    best = None
    for i, _, t in tl:
        if i > idx:
            break
        best = t
    return best


def opt_levels(parsed: Dict[str, Any], eta: Dict[str, Any], lines: Sequence[str], finalrun: bool,
               scratch: Optional[Path], start: Optional[float], now_ref: Optional[float],
               notes: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pts = parsed.get("points") or []
    done = len(pts)
    seg = pts[-1].get("segment", 0) if pts else 0
    tl = timings(lines)
    in_point = None
    if start is not None and now_ref is not None and tl:
        in_point = max(0.0, now_ref - (start + tl[-1][2]))
    lv = _level("opt.points", 1, "Optimization", "running", current=done, unit="points",
                thresholds=notes, basis=eta.get("basis") or "", per_unit_s=eta.get("per_point_s"))
    if eta.get("low_points") is not None:
        x_lo, x_hi = (FINALRUN_LOW, FINALRUN_HIGH) if finalrun else (0, 0)
        lo_n = done + eta["low_points"]
        hi_n = done + (eta["high_points"] if eta["high_points"] is not None else eta["low_points"])
        if seg == 0 and finalrun:
            lv["count"].update(low=max(done, lo_n - x_lo), high=max(done, hi_n - x_hi))
            lv["extra"] = {"label": "final restart", "low": x_lo, "high": x_hi}
        else:
            lv["count"].update(low=lo_n, high=hi_n)
            lv["info"]["segment"] = seg
        if eta.get("low_s") is not None:
            # the running point is part of the points left: time already spent in it comes off the
            # low end, but the other points left still run in full
            rest = max(0, eta["low_points"] - 1) * (eta.get("per_point_s") or 0.0) * TIME_LOW_FACTOR
            low = max(rest, eta["low_s"] - (in_point or 0.0))
            lv["remaining_s"] = {"low": min(low, eta["low_s"]), "high": eta["high_s"]}
            narrow = (eta["high_points"] or 0) <= FAIR_MAX_WIDTH * max(eta["low_points"], 1)
            lv["confidence"] = "fair" if (not eta.get("capped") and narrow) else "rough"
        else:
            lv["confidence"] = "unknown"
            lv["unknown_reason"] = "time per point unknown"
        if eta.get("capped"):
            lv["flags"].append("prior_high")
    else:
        lv["unknown_reason"] = eta.get("basis") or "not enough points yet"
    out = [lv]
    if not done:
        # the first point's SCF runs in the output itself (SCFOUT.LOG takes over from point 1):
        # its cycle and |DE|, no cycles-left estimate (the SP rule was not tested on it)
        cyc = [m for m in (_CYC.match(l) for l in lines) if m]
        zeros = [i for i, m in enumerate(cyc) if int(m.group(1)) == 0]
        if zeros:
            run = cyc[zeros[-1]:]
            try:
                de = abs(float(run[-1].group(3)))
            except ValueError:
                de = None
            tol = ((parsed.get("thresholds") or {}).get("toldee_scf") or {}).get("value")
            out.append(_level("opt.first_scf", 2, "First point's SCF", "running",
                              current=int(run[-1].group(1)), unit="cycles",
                              unknown_reason="no cycles-left estimate inside an optimization's first point",
                              info={"detot": de, "toldee": tol}))
        return out
    counts = [p.get("scf_cycles") for p in pts[1:]]
    lo_c, hi_c = _cycle_window(counts)
    live = scf_log_tail(scratch) if done else None
    if done and (lo_c is not None or live is not None):
        inner = _level("opt.point_scf", 2, "Next point's SCF", "running", current=live, unit="cycles",
                       low_n=lo_c, high_n=hi_c, confidence="rough" if lo_c is not None else "unknown",
                       basis=(f"[min - 1, max + 2] of the SCF cycles of the last "
                              f"{min(OPT_CYCLE_WINDOW, len([c for c in counts if c]))} points"
                              if lo_c is not None else "no finished point after the first"))
        heads = [i for i, line in enumerate(lines) if _POINT.search(line)]
        starts = [_t_at(tl, h) for h in heads]
        durs = [b - a for a, b in zip(starts, starts[1:]) if a is not None and b is not None and b >= a]
        if durs:
            recent = durs[-10:]
            inner["info"]["point_low_s"] = _quantile(recent, 0.1)
            inner["info"]["point_high_s"] = _quantile(recent, 0.9)
            mean_c = statistics.median([c for c in counts if c][-OPT_CYCLE_WINDOW:] or [0]) or None
            if live is not None and mean_c:
                inner["per_unit_s"] = statistics.median(recent[-TIME_WINDOW:]) / mean_c
        inner["info"]["in_point_s"] = in_point
        if live is None:
            inner["info"]["source"] = "earlier points"
        out.append(inner)
    return out


def _quantile(xs: Sequence[float], q: float) -> Optional[float]:
    xs = sorted(xs)
    if not xs:
        return None
    k = q * (len(xs) - 1)
    a = int(math.floor(k))
    b = min(len(xs) - 1, a + 1)
    return xs[a] + (xs[b] - xs[a]) * (k - a)


def _guessp(text: str, deck_text: str) -> bool:
    if re.search(r"INFORMATION \*+ GUESSP \*+", text):
        return True
    return any(l.strip().upper() == "GUESSP" for l in (deck_text or "").splitlines()[1:])


def sp_levels(sp: Dict[str, Any], lines: Sequence[str], text: str, deck_text: str, ended: Optional[str],
              notes: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    tl = timings(lines)
    c0 = next((i for i, l in enumerate(lines) if _CYC.match(l)), None)
    end = None
    if c0 is not None:
        end = next((i for i in range(c0, len(lines)) if "== SCF ENDED" in lines[i]), None)
    t_now = tl[-1][2] if tl else None
    setup = _level("sp.setup", 1, "Setup and integrals", "done" if c0 is not None else "running")
    if c0 is not None:
        setup["done_s"] = _t_at(tl, c0) or 0.0
        setup["confidence"] = "fair"
        setup["remaining_s"] = {"low": 0.0, "high": 0.0}
    else:
        setup["info"]["so_far_s"] = t_now or 0.0
        setup["unknown_reason"] = "setup and integrals are running"
        setup["basis"] = "no estimate for the setup and integrals"
    scf = _level("sp.scf", 1, "SCF", "pending" if c0 is None else ("done" if end is not None else "running"),
                 current=sp["cycles"], unit="cycles", per_unit_s=sp.get("per_cycle_s"), thresholds=notes,
                 flags=list(sp.get("flags") or []), basis=sp.get("basis") or "",
                 info={"detot": abs(sp["last_detot"]) if sp.get("last_detot") is not None else None,
                       "toldee": sp.get("toldee"), "maxcycle": sp.get("maxcycle")})
    if _guessp(text, deck_text):
        scf["flags"].append("guessp")
    if scf["status"] == "done":
        scf["done_s"] = (_t_at(tl, end) or 0.0) - (_t_at(tl, c0) or 0.0)
        scf["remaining_s"] = {"low": 0.0, "high": 0.0}
        scf["confidence"] = "fair"
    elif scf["status"] == "running" and sp.get("low_cycles") is not None:
        scf["count"].update(low=sp["cycles"] + sp["low_cycles"], high=sp["cycles"] + sp["high_cycles"])
        scf["remaining_s"] = {"low": sp.get("low_s"), "high": sp.get("high_s")}
        scf["confidence"] = sp["confidence"] if sp.get("low_s") is not None else "unknown"
        if sp.get("capped"):
            scf["flags"].append("prior_high")
        if sp.get("low_s") is None:
            scf["unknown_reason"] = "time per cycle unknown"
    else:
        scf["unknown_reason"] = ("the SCF has not started" if scf["status"] == "pending"
                                 else f"SCF cycles left unknown: {sp.get('basis') or 'no trend yet'}")
    scf_time = None
    if scf["status"] == "done":
        scf_time = scf["done_s"]
    elif c0 is not None and t_now is not None:
        scf_time = (t_now - (_t_at(tl, c0) or 0.0)) + (scf["remaining_s"]["high"] or 0.0)
    post_hi = max(10.0, 0.01 * (scf_time or 0.0))
    post = _level("sp.post", 1, "After the SCF", "pending", low_s=0.0, high_s=post_hi, confidence="rough",
                  basis="1 % of the SCF time, at least 10 s (corpus: median 0.3 %, 90th percentile 0.9 %)")
    if end is not None:
        post["status"] = "running"
    if ended:
        for lv in (setup, scf, post):
            lv["status"] = "done"
            lv["remaining_s"] = {"low": 0.0, "high": 0.0}
        if end is not None and tl:
            post["done_s"] = tl[-1][2] - (_t_at(tl, end) or 0.0)
    return [setup, scf, post]


def freq_levels(fp: Dict[str, Any], st: Dict[str, Any], lines: Sequence[str], scratch: Optional[Path],
                ended: Optional[str], notes: Sequence[Dict[str, Any]], start: Optional[float] = None,
                now_ref: Optional[float] = None) -> List[Dict[str, Any]]:
    """The FREQ stages in CRYSTAL's order (stages that do not apply are left
    out). IR/Raman CPHF stages carry a status line but no time prediction in
    this release, so a total while they run reads "at least". With the run's
    clock (``start`` and ``now_ref``, only for a job SLURM runs now) a running
    stage without an estimate says how long it has run: the IR tensors can
    print nothing for many minutes."""
    tl = timings(lines)

    def so_far(lv: Dict[str, Any], idx: Optional[int]) -> None:
        t = _t_at(tl, idx) if idx is not None else None
        if lv["status"] == "running" and t is not None and start is not None and now_ref is not None:
            lv["info"]["so_far_s"] = max(0.0, now_ref - (start + t))
    phase = st["phase"]
    order = ["preopt", "ref_scf", "cphf", "cp2", "tensors", "central", "central_done"]
    pos = order.index(phase) if phase in order else (order.index("central") if phase == "setup" else 0)
    out: List[Dict[str, Any]] = []

    def status(my: str) -> str:
        if ended:
            return "done"
        i = order.index(my)
        return "done" if pos > i else ("running" if pos == i else "pending")

    def took(a, b):
        ta, tb = _t_at(tl, a), _t_at(tl, b)
        return (tb - ta) if ta is not None and tb is not None and tb >= ta else None
    open_hi = "IR/Raman (CPHF/CPKS) has no time estimate in this release"
    if st["preopt_points"]:
        lv = _level("freq.preopt", 1, "Preoptimization", status("preopt"), current=len(st["preopt_points"]),
                    unit="points", unknown_reason="preoptimization levels come in a later release",
                    basis="stage name only in this release")
        out.append(lv)
    if st["cphf"]:
        ref_end = st["cp_driver"]
        lv = _level("freq.ref_scf", 1, "Reference SCF for IR/Raman", status("ref_scf"),
                    unknown_reason="the reference SCF has no time estimate", basis="")
        if lv["status"] == "done":
            lv["done_s"] = took(st["ref_c0"], ref_end)
        elif lv["status"] == "running":
            lv["current"] = sum(1 for i in range(st["ref_c0"], len(lines)) if _CYC.match(lines[i])) - 1
            so_far(lv, st["ref_c0"])
        out.append(lv)
        cp_end = st["cp2_driver"] if st["cp2_driver"] is not None else (st["tensors"][0] if st["tensors"] else st["header"])
        lv = _level("freq.cphf", 1, "IR/Raman (CPHF/CPKS)", status("cphf"), unit="directions", exact=3,
                    unknown_reason=open_hi, info={"tol": st["cp_tol"]})
        if lv["status"] == "done":
            lv["done_s"] = took(st["cp_driver"], cp_end)
        elif lv["status"] == "running" and st["cp_dirs"]:
            idx, d = st["cp_dirs"][-1]
            lv["current"] = len(st["cp_dirs"])
            lv["info"]["direction"] = d
            cyc = [c for c in st["cp_cycles"] if c[0] > idx]
            if cyc:
                lv["info"]["cycle"], lv["info"]["delta"] = cyc[-1][1], cyc[-1][2]
        elif lv["status"] == "running":
            lv["current"] = 0
            lv["info"]["direction"] = "-"
        so_far(lv, st["cp_driver"])
        out.append(lv)
        if st["raman"]:
            lv = _level("freq.cp2", 1, "Raman (CPHF/CPKS, second order)", status("cp2"), unit="directions",
                        exact=6, unknown_reason=open_hi)
            if lv["status"] == "done":
                lv["done_s"] = took(st["cp2_driver"], st["tensors"][0] if st["tensors"] else st["header"])
            elif lv["status"] == "running" and st["cp2_dirs"]:
                idx, d = st["cp2_dirs"][-1]
                lv["current"] = len(st["cp2_dirs"])
                lv["info"]["direction"] = d
                cyc = [c for c in st["cp_cycles"] if c[0] > idx]
                if cyc:
                    lv["info"]["cycle"] = cyc[-1][1]
            elif lv["status"] == "running":
                lv["current"], lv["info"]["direction"] = 0, "-"
            so_far(lv, st["cp2_driver"])
            out.append(lv)
        lv = _level("freq.tensors", 1, "IR/Raman tensors", status("tensors"), unknown_reason=open_hi)
        if lv["status"] == "done" and st["tensors"]:
            lv["done_s"] = took(st["tensors"][0], st["header"])
        so_far(lv, st["tensors"][0] if st["tensors"] else None)
        out.append(lv)
    cen = _level("freq.central", 1, "Central point", status("central"),
                 unknown_reason="the central point has no time estimate", thresholds=notes,
                 basis="SCF + gradient of the undisplaced cell")
    if st["restart"] and st["totgra"] is None:
        if st["table"] is not None or ended:
            cen["status"] = "restored"
            cen["remaining_s"] = {"low": 0.0, "high": 0.0}
        else:
            cen["status"] = "running"             # the restart's setup, before the restored table
            cen["info"]["restart_setup"] = True
    elif cen["status"] == "done":
        cen["done_s"] = took(st["central_c0"], st["totgra"])
        cen["remaining_s"] = {"low": 0.0, "high": 0.0}
    elif cen["status"] == "running" and st["central_c0"] is not None:
        cen["current"] = sum(1 for i in range(st["central_c0"], len(lines)) if _CYC.match(lines[i])) - 1
        cen["info"]["scf_done"] = any(e > st["central_c0"] for e in st["scf_end"])
    out.append(cen)
    total = fp.get("total")
    stage = fp.get("stage")
    dsp = _level("freq.displacements", 1, "Displacements", "pending", current=fp.get("done"), unit="displacements",
                 exact=total, per_unit_s=fp.get("per_displacement_s"),
                 info={"percent": fp.get("percent"), "restored": fp.get("restored") if st["restart"] else 0,
                       "computed": fp.get("computed")},
                 unknown_reason=("time per displacement unknown: " + fp["rate_note"]
                                 if fp.get("rate_note") and not fp["rate_note"].startswith("until")
                                 else "time per displacement unknown " + (fp.get("rate_note") or "until the first finishes")),
                 basis=(fp.get("basis") or "") if fp.get("stage") == "displacements" else "")
    dsp["info"]["rate_note"] = fp.get("rate_note")
    dsp["info"]["in_displacement_s"] = fp.get("in_displacement_s")
    if ended or stage in ("frequencies", "intensities"):
        dsp["status"] = "done"
        dsp["remaining_s"] = {"low": 0.0, "high": 0.0}
    elif stage == "displacements":
        dsp["status"] = "running"
        if fp.get("low_s") is not None:
            dsp["remaining_s"] = {"low": fp["low_s"], "high": fp["high_s"]}
            dsp["confidence"] = "rough"
            dsp["unknown_reason"] = None
            dsp["basis"] = (f"rows left x {_fmt_dur(fp['per_displacement_s'])} per displacement x 0.75-1.5, "
                            f"from the {fp.get('computed')} computed in this run (held-out corpus: the cycles of "
                            f"the rows left fall in this band in most runs)")
    out.append(dsp)
    if stage == "displacements" and not ended:
        live = scf_log_tail(scratch)
        if live is not None:
            counts = [int(m.group(1)) for m in (_ROW_NCYC.match(l) for l in lines[st["table"] or 0:]) if m] \
                if st["table"] is not None else []
            lo_c, hi_c = _cycle_window(counts)
            out.append(_level("freq.displacement_scf", 2, "Next displacement's SCF", "running", current=live,
                              unit="cycles", low_n=lo_c, high_n=hi_c,
                              confidence="rough" if lo_c is not None else "unknown",
                              basis="[min - 1, max + 2] of the SCF cycles of the last 5 rows"))
    per = fp.get("per_displacement_s")
    hi_out = max(FREQ_OUTPUT_MIN_S, 0.002 * per * total) if per and total else FREQ_OUTPUT_MIN_S
    outp = _level("freq.output", 1, "Frequencies and output", "pending", low_s=0.0, high_s=hi_out,
                  confidence="rough", basis="after the last displacement: at most 8.5 s in all 30 corpus runs; "
                                            "bound max(30 s, 0.2 % of the displacement time)")
    if ended:
        outp["status"] = "done"
        outp["remaining_s"] = {"low": 0.0, "high": 0.0}
    elif stage in ("frequencies", "intensities"):
        outp["status"] = "running"
    out.append(outp)
    return out


# ------------------------------------------------------------------ walltime risk
def risk_level(elapsed_s: Optional[float], limit_s: Optional[float], low_s: Optional[float],
               high_s: Optional[float]) -> Optional[str]:
    if elapsed_s is None or not limit_s or high_s is None:
        return None
    remaining = limit_s - elapsed_s
    if low_s is not None and low_s > remaining:
        return "likely to exceed"
    if elapsed_s + high_s >= 0.8 * limit_s:
        return "tight"
    return "ok"


TIGHT_FACTOR = 0.8


def risk_from_total(elapsed_s: Optional[float], limit_s: Optional[float],
                    total: Optional[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
    """(level, note) from the total level. W = walltime left. "likely to
    exceed": the low end of the time left does not fit (works for "at
    least" totals); "tight": the central estimate (geometric mean of a
    known range, else the low end) needs 80 % of W or more; otherwise "ok",
    with a soft note when the long end of the range would not fit."""
    if elapsed_s is None or not limit_s or not total:
        return None, None
    lo, hi = total["remaining_s"]["low"], total["remaining_s"]["high"]
    if lo is None or (hi is None and lo < 1):
        return None, None
    w = limit_s - elapsed_s
    if lo > w:
        return "likely to exceed", None
    c = math.sqrt(lo * hi) if (hi is not None and lo > 0) else lo
    if c >= TIGHT_FACTOR * w:
        return "tight", None
    if (hi is not None and hi >= w) or (hi is None and lo >= 0.5 * w):
        long_end = _fmt_dur(hi) if hi is not None else f"at least {_fmt_dur(lo)}"
        return "ok", (f"Could exceed the walltime if it runs to the long end of the range ({long_end} left in "
                      f"total, {_fmt_dur(max(0.0, w))} of walltime remaining).")
    return "ok", None


def ro_uri(db, mode: str = "mode=ro") -> str:
    """An SQLite URI that opens ``db`` read-only; the path is percent-encoded
    (a '?', '#' or '%' in a folder name would otherwise end the path)."""
    return Path(os.path.abspath(str(db))).as_uri() + "?" + mode


def retries_spent(db: Path, job_id: Optional[str], deck_stem: Optional[str]) -> Tuple[Optional[bool], int,
                                                                                    Optional[str]]:
    """(tracked, timeout recoveries spent on the lineage, calc_type) from
    the folder's materials.db, opened read-only."""
    if not db.is_file():
        return False, 0, None
    try:
        conn = sqlite3.connect(ro_uri(db), uri=True, timeout=5)
    except sqlite3.Error:
        return None, 0, None
    try:
        conn.execute("PRAGMA busy_timeout=5000")
        row = None
        if job_id:
            row = conn.execute("SELECT calc_id, material_id, calc_type FROM calculations WHERE slurm_job_id = ? "
                               "ORDER BY rowid DESC LIMIT 1", (str(job_id),)).fetchone()
        if row is None and deck_stem:
            row = conn.execute("SELECT calc_id, material_id, calc_type FROM calculations WHERE input_file LIKE ? "
                               "ORDER BY rowid DESC LIMIT 1", (f"%{deck_stem}.d12",)).fetchone()
        if row is None:
            return False, 0, None
        n = conn.execute("SELECT COUNT(*) FROM calculations WHERE material_id = ? AND calc_type = ? "
                         "AND calc_id != ? AND status = 'resubmitted' AND error_type = 'timeout_error'",
                         (row[1], row[2], row[0])).fetchone()[0]
        return True, int(n or 0), row[2]
    except sqlite3.Error:
        return None, 0, None
    finally:
        conn.close()


def other_tracked_active(db: Path, own_rows: Sequence[Dict[str, Any]], job_id: str) -> bool:
    """Another job tracked in the same DB is still pending or running."""
    if not db.is_file():
        return False
    ids = {r["id"] for r in own_rows if (r.get("state") or "").upper() in ("PENDING", "RUNNING", "CONFIGURING")
           and r["id"] != str(job_id)}
    if not ids:
        return False
    try:
        conn = sqlite3.connect(ro_uri(db), uri=True, timeout=5)
        try:
            q = ",".join("?" * len(ids))
            n = conn.execute(f"SELECT COUNT(*) FROM calculations WHERE slurm_job_id IN ({q})", tuple(ids)).fetchone()[0]
            return bool(n)
        finally:
            conn.close()
    except sqlite3.Error:
        return False


def risk_sentence(level: Optional[str], tracked: Optional[bool], cfg: Dict[str, Any], spent: int,
                  other_active: bool, calc_type: Optional[str], folder: str,
                  queue_limit_text: Optional[str] = None) -> Optional[str]:
    if level in (None, "ok"):
        return None
    t = cfg.get("values") or {}
    note = f" {cfg['note']}" if cfg.get("note") else ""
    if not tracked:
        kind = (calc_type or "").upper()
        if kind == "FREQ":
            how_later = ("To continue it later, resubmit it with RESTART in its FREQCALC block: CRYSTAL "
                         "skips the displacements already in FREQINFO.DAT (keep that file with the job)")
        elif kind in ("OPT", ""):
            how_later = ("To continue an optimization later, resubmit it from its last geometry (OPTGEOM "
                         "RESTART with OPTINFO.DAT; see docs/cli.md#mace-cluster)")
        else:
            how_later = "Resubmit it with a longer walltime"
        return "This job is not tracked, so nothing restarts it automatically. " + how_later + "." + note
    retries = int(t.get("max_retries") or 0)
    if not t.get("enable_restart", True) and (calc_type or "").upper() == "OPT" or retries == 0:
        return "MACE's timeout recovery is switched off for this job (recovery_config.yaml)." + note
    if spent >= retries:
        times = {1: "once", 2: "twice"}.get(spent, f"{spent} times")
        return f"MACE already restarted it {times}; a further timeout is not retried." + note
    factor = t.get("walltime_factor") or 2.0
    fw = "twice the walltime" if float(factor) == 2.0 else f"{factor:g} times the walltime"
    cap = f" (up to the queue limit{' of ' + queue_limit_text if queue_limit_text else ''}; attempt {spent + 1} of {retries})"
    how = "with OPTGEOM RESTART and " if (calc_type or "OPT").upper() == "OPT" else "with "
    if other_active:
        return (f"If it times out, MACE will resubmit it {how}{fw}{cap} when the queue manager next runs: when "
                f"another job of this folder finishes, or when you run `mace manager --callback-mode completion` "
                f"in {folder}." + note)
    return (f"If it times out, nothing will notice it automatically: it is the last MACE job of this folder in "
            f"the queue, and a job killed at its walltime does not run MACE's completion step. After it ends, run "
            f"`mace manager --callback-mode completion` in {folder}; MACE then resubmits it {how}{fw} "
            f"(attempt {spent + 1} of {retries})." + note)


# ------------------------------------------------------------------ entry
def mode_of(text: str) -> str:
    from mace.utils import calc_detection as cd
    if cd.is_frequency_output(text):
        return "freq"
    if cd.is_optimization_output(text):
        return "opt"
    if cd.is_band_output(text) or cd.is_doss_output(text) or cd.is_transport_output(text) or \
            cd.is_charge_potential_output(text):
        return "properties"
    if "CYC " in text or "TOTAL ENERGY" in text or "SCF" in text:
        return "scf"
    return "unknown"


def ended_state(path: Path) -> Optional[str]:
    """CRYSTAL's end marker through completion_checker: None while running."""
    from mace.completion_checker import categorize_output_file
    cat, _ = categorize_output_file(path)
    return None if cat in ("ongoing",) else cat


_MEMO: "OrderedDict[tuple, Dict[str, Any]]" = OrderedDict()
MEMO_SIZE = 32
_MEMO_LOCK = threading.Lock()


def _memo_key(out_path: Path, scratch_dir, job: Optional[Dict[str, Any]]) -> Optional[tuple]:
    try:
        st = os.stat(out_path)
    except OSError:
        return None
    parts = [str(out_path), st.st_ino, st.st_size, st.st_mtime_ns]
    for sib in (out_path.with_suffix(".d12"), Path(scratch_dir) / "SCFOUT.LOG" if scratch_dir else None):
        try:
            s2 = os.stat(sib) if sib is not None else None
            parts.append((s2.st_size, s2.st_mtime_ns) if s2 else None)
        except OSError:
            parts.append(None)
    el = (job or {}).get("elapsed_s")
    parts += [str(scratch_dir), None if el is None else int(float(el) // 30), (job or {}).get("limit_s")]
    return tuple(parts)


def progress(out_path, scratch_dir=None, job: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Progress of one .out (see the module doc). ``job`` may carry
    elapsed_s, limit_s (for the walltime risk). An unchanged snapshot (the
    .out, its deck, SCFOUT.LOG and a 30 s bucket of the elapsed time) returns
    a copy of the previous result."""
    out_path = Path(out_path)
    key = _memo_key(out_path, scratch_dir, job)
    if key is not None:
        with _MEMO_LOCK:                      # the GUI server reads jobs from several threads
            hit = _MEMO.get(key)
            if hit is not None:
                _MEMO.move_to_end(key)
                return copy.deepcopy(hit)
    res = _progress(out_path, scratch_dir, job)
    if key is not None:
        with _MEMO_LOCK:
            _MEMO[key] = copy.deepcopy(res)
            while len(_MEMO) > MEMO_SIZE:
                _MEMO.popitem(last=False)
    return res


def _progress(out_path: Path, scratch_dir=None, job: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    import time as _time
    text, snap = read_text(out_path)
    lines = text.splitlines()
    mode = mode_of(text)
    ended = ended_state(out_path) if text else None
    deck = out_path.with_suffix(".d12")
    deck_text = ""
    if deck.is_file():
        try:
            deck_text = deck.read_text(errors="replace")
        except OSError:
            deck_text = ""
    res: Dict[str, Any] = {"mode": mode, "file": str(out_path), "snapshot": snap, "ended": ended,
                           "stage": None, "done": None, "total": None, "percent": None,
                           "eta": {"low_s": None, "high_s": None, "basis": None, "confidence": "unknown"},
                           "risk": {"level": None, "sentence": None, "note": None}}
    scratch = Path(scratch_dir) if scratch_dir else None
    elapsed = (job or {}).get("elapsed_s")
    slurm_start = _time.time() - float(elapsed) if elapsed is not None else None
    # only for a job SLURM runs now: a copied or finished .out has no clock to anchor to
    start, start_src = run_start(lines, snap.get("mtime"), slurm_start) if slurm_start is not None else (None, None)
    now_ref = _time.time() if elapsed is not None else snap.get("mtime")
    stages: List[Dict[str, Any]] = []
    if mode == "opt":
        parsed = parse_out(out_path, text, snap["size"])
        pts = parsed.get("points") or []
        finalrun = finalrun_from_deck(deck_text)
        eta = opt_eta(parsed, lines, finalrun=finalrun)
        maxcyc = int(((parsed.get("thresholds") or {}).get("maxcycle") or {}).get("value") or DEFAULT_MAXCYCLE)
        res.update(stage="end" if ended else "optimizing", done=len(pts), total=None, eta=eta,
                   criteria=[{k: p.get(k) for k in ("point", "de_printed", "segment", *CRITERIA)} |
                             {"thr": p.get("thr"), "converged": p.get("converged")} for p in pts[-50:]],
                   maxcycle=maxcyc, parser=parsed.get("via"))
        if eta.get("low_points") is not None:
            lo, hi = len(pts) + eta["low_points"], len(pts) + (eta["high_points"] or eta["low_points"])
            res["summary"] = f"Point {len(pts)} of about {lo}-{hi}" if hi != lo else f"Point {len(pts)} of about {lo}"
        else:
            res["summary"] = f"Point {len(pts)}" + (" (finished)" if ended else "")
        notes = threshold_notes(parsed.get("thresholds") or {}, "opt")
        stages = opt_levels(parsed, eta, lines, finalrun, scratch, start, now_ref, notes)
        pl = stages[0]
        if pl["count"]["low"] is not None:
            # the heading uses the level's split (main segment + final restart), so the two agree
            x = pl["extra"]
            res["summary"] = (f"Point {len(pts)} of about {_nspan(pl['count']['low'], pl['count']['high'])}"
                              + (f" (+ {x['label']} {_nspan(x['low'], x['high'])})" if x else ""))
        if ended:
            for lv in stages:
                lv["status"] = "done"
    elif mode == "freq":
        st = freq_stages(lines, deck_text)
        fp = freq_progress(text, lines, bool(ended), deck_text, scratch, mtime=snap.get("mtime"),
                           job_start=start, stages=st, now=now_ref if elapsed is not None else None)
        res.update(stage=fp["stage"], done=fp["done"], total=fp["total"], percent=fp["percent"], freq=fp)
        res["eta"] = {"low_s": fp["low_s"], "high_s": fp["high_s"], "basis": fp["basis"],
                      "confidence": fp["confidence"]}
        if fp["stage"] == "displacements" and fp["total"]:
            res["summary"] = (f"Displacement {fp['done']} of {fp['total']} ({fp['percent']} %)" if fp["done"]
                              else f"Central point done; {fp['total']} displacements to run")
        else:
            res["summary"] = fp["basis"] or fp["stage"]
        parsed_thr = {}
        m = _SCF_TOLDEE.search(text)
        if m:
            parsed_thr["toldee_scf"] = {"value": 10.0 ** -int(m.group(1)), "source": "out"}
        stages = freq_levels(fp, st, lines, scratch, ended, threshold_notes(parsed_thr, "freq"),
                             start=start, now_ref=now_ref if elapsed is not None else None)
    elif mode == "scf":
        parsed = parse_out(out_path, text, snap["size"])
        sp = scf_progress(parsed, lines)
        res.update(stage="end" if ended else "scf", done=sp["cycles"], scf=sp, parser=parsed.get("via"))
        res["eta"] = {"low_s": sp["low_s"], "high_s": sp["high_s"], "basis": sp["basis"],
                      "confidence": sp["confidence"]}
        res["summary"] = f"SCF cycle {sp['cycles']}" + (f", |DE| {_sci(abs(sp['last_detot']))}"
                                                        if sp["last_detot"] is not None else "")
        stages = sp_levels(sp, lines, text, deck_text, ended,
                           threshold_notes(parsed.get("thresholds") or {}, "sp"))
    elif mode == "properties":
        blocks = [name for _, name, _ in timings(lines)]
        res.update(stage="end" if ended else (blocks[-1] if blocks else "starting"))
        res["summary"] = "Finished" if ended else f"Running {res['stage']}"
        stages = [_level("properties.block", 1, "Properties", "done" if ended else "running",
                         current=res["stage"], unknown_reason="properties runs have no time estimate",
                         basis="which block runs; no estimate")]
    if ended:
        res["summary"] = f"Ended ({ended})"
    total = total_level(stages, ended)
    res["levels_version"] = LEVELS_VERSION
    res["levels"] = _finish_levels([total] + stages)
    res["start"] = {"epoch": start, "source": start_src}
    # res["eta"]: the total's numbers (old per-mode sentence kept as stage_basis)
    eta = res["eta"]
    eta["stage_basis"] = eta.get("basis")
    eta["low_s"], eta["high_s"] = total["remaining_s"]["low"], total["remaining_s"]["high"]
    eta["confidence"] = total["confidence"]
    eta["basis"] = total["basis"]
    if job:
        lvl, note = risk_from_total(job.get("elapsed_s"), job.get("limit_s"), total)
        if ended:
            lvl, note = None, None
        res["risk"]["level"], res["risk"]["note"] = lvl, note
    return res
