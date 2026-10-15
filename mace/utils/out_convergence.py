"""Convergence of a CRYSTAL run, read from its .out (PLAN_analysis C1).

One streaming pass (binary, 1 MB reads) gives, per CRYSTAL optimization
POINT (keyed by the number CRYSTAL prints):

* the energy on the optimizer's surface (point 1's dispersion/gCP-corrected
  total, the same rule as ``mace.gui.outindex`` v3), the SCF cycle count from
  ``TOTAL ENERGY(DFT)(AU)( n)``, the printed ``DE (AU)``;
* the four criteria (max/RMS gradient, max/RMS displacement) with their
  thresholds and CONVERGED flags as CRYSTAL prints them;
* the cell of the point's geometry block;

plus the FINALRUN restart segments, how the optimization stopped, the SCF
per-cycle blocks CRYSTAL actually prints (only the first point of an OPT:
later points go to SCFOUT.LOG), and the thresholds with where each came
from (the .out, the sibling .d12, or the CRYSTAL23 default with its manual
page).

Units: every OPT in the corpus is a combined cell/atom optimization, which
prints "THE GRADIENT AND HESSIAN MATRIX IN TERMS OF THIS SYSTEM ARE GIVEN
WITHOUT UNITS"; ``gradient_units`` is then "internal". Without that note
(atom-only runs) the gradient is in Ha/bohr and the displacement in bohr.

How the run stopped (``opt.end.reason``) is decided from the parsed state
(criteria flags, DE, TOLDEE, the counts), never from warning text alone; the
WARNING banner before the end line is kept verbatim. The CRYSTAL23 manual's
alternate stops (section 7.4.2, p.208) are classified from that state; their
exact banner text is not in the test corpus. Read from the CRYSTAL23 binary
(``strings``, read-only, on a cluster that has it), the texts are:

* "CONVERGENCE ON GRADIENT - NO CHANGE IN ENERGY FOR <n> STEPS" (p.208 case
  1: gradients met, energy unchanged; ``gradient_and_energy``);
* "FLAT SURFACE - NO CHANGE IN ENERGY FOR <n> STEPS" (case 2: no criterion
  met, energy unchanged; ``energy_stalled``);
* "CONVERGENCE TESTS NOT SATISFIED AFTER <n> ENERGY AND GRADIENT
  CALCULATIONS" and "... UNSATISFIED AFTER ..." (the point budget ran out;
  ``maxcycle``).

They are extra evidence only: used when the state alone leaves the reason
``unknown``, never to override it.

Pure standard library: the GUI caches ``parse()`` keyed by file stamp and
``VERSION``; the CLI prints it (``mace analyze --convergence FILE.out``).
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

VERSION = 2
CHUNK = 1 << 20

# ------------------------------------------------------------------ patterns
_POINT = re.compile(r"OPTIMIZATION - POINT\s+(\d+)\s*$")
_ENERGY = re.compile(r"TOTAL ENERGY\((?:DFT|HF)\)\(AU\)\(\s*(\d+)\)\s+(\S+)(?:\s+DE \(AU\)\s+(\S+))?")
_CORRECTED = re.compile(r"TOTAL ENERGY \+ (?:DISP|GCP)(?: \+ (?:DISP|GCP))? \(AU\)\s+(\S+)")
_CRIT = re.compile(r"^\s*(MAX|RMS) (GRADIENT|DISPLAC\.)\s+(\S+)\s+THRESHOLD\s+(\S+)\s+CONVERGED\s+(YES|NO)")
_OPT_END = re.compile(r"OPT END - (CONVERGED|FAILED)")
_OPT_END_E = re.compile(r"E\(AU\):\s*(\S+)")
_OPT_END_PTS = re.compile(r"POINTS\s+(\d+)")
_BLOCK_START = re.compile(r"^\s*PRIMITIVE CELL(?:\s+- CENTRING CODE|\s*$)")
_VOLUME = re.compile(r"VOLUME=\s*(\S+)")
_UNIT_CELL_ATOMS = re.compile(r"ATOMS IN THE UNIT CELL:\s*(\d+)")
_CYC = re.compile(r"^\s*CYC\s+(\d+)\s+ETOT\(AU\)\s+(\S+)\s+DETOT\s+(\S+)")
_HDR_MAX = re.compile(r"MAXIMUM GRADIENT COMPONENT\s+(\S+)\s+MAXIMUM DISPLACEMENT COMPONENT\s+(\S+)")
_HDR_RMS = re.compile(r"R\.M\.S\. OF GRADIENT COMPONENT\s+(\S+)\s+R\.M\.S\. OF DISPLACEMENT COMPONENTS\s+(\S+)")
_HDR_DE = re.compile(r"THRESHOLD ON ENERGY CHANGE\s+(\S+)")
_HDR_STEPS = re.compile(r"MAXIMUM ALLOWED NUMBER OF STEPS\s+(\d+)")
_INFO_TOLDEG = re.compile(r"INFORMATION \*+ TOLDEG \*+.*?(\S+)\s*$")
_INFO_TOLDEX = re.compile(r"INFORMATION \*+ TOLDEX \*+.*?(\S+)\s*$")
_INFO_TOLDEE_OPT = re.compile(r"INFORMATION \*+ TOLDEE \*+ CONVERGENCE CRITERIUM FOR ENERGY MODIFIED: 10\*\*-\s*(\d+)")
_INFO_TOLDEE_SCF = re.compile(r"INFORMATION \*+ TOLDEE \*+ SCF TOL ON TOTAL ENERGY SET TO\s+(\d+)")
_INFO_MAXCYC_OPT = re.compile(r"INFORMATION \*+ MAXCYCLE \*+\s+MAXIMUM NUMBER OF OPTIMIZATION CYCLES\s+(\d+)")
_INFO_MAXCYC_SCF = re.compile(r"INFORMATION \*+ MAXCYCLE \*+ MAX NUMBER OF SCF CYCLES SET TO\s+(\d+)")
_SCF_TOLDEE = re.compile(r"CONVERGENCE ON ENERGY\s+10\*\*-\s*(\d+)")
_SCF_MAXCYC = re.compile(r"MAX NUMBER OF SCF CYCLES\s+(\d+)")

FIRST_CYCLE_TEXT = "CONVERGENCE ON GRADIENTS SATISFIED AFTER THE FIRST OPTIMIZATION CYCLE"
UNITS_NOTE = "GIVEN WITHOUT"          # "... HESSIAN MATRIX IN TERMS OF THIS SYSTEM ARE GIVEN WITHOUT UNITS"
UNITS_QUOTE = ("The coordinate system adopted in the optimization is not in general homogeneous units; "
               "the gradient and Hessian matrix in terms of this system are given without units.")
CRITERIA = ("max_grad", "rms_grad", "max_disp", "rms_disp")
_CRIT_KEY = {("MAX", "GRADIENT"): "max_grad", ("RMS", "GRADIENT"): "rms_grad",
             ("MAX", "DISPLAC."): "max_disp", ("RMS", "DISPLAC."): "rms_disp"}

# CRYSTAL23 defaults (user manual; pages from the PDF footers). Section 7.2.2
# (p.188): all four criteria together end the optimization. The default table
# and the context rows (p.189): RMS gradient / RMS displacement; the maximum
# components are 1.5 times the RMS ones.
CONTEXT_DEFAULTS = {
    "standard": (0.0003, 0.0012),
    "freq": (0.00003, 0.00012),        # FREQCALC preoptimization
    "eos": (0.00006, 0.00012),         # EOS preoptimization
    "elastic": (0.00006, 0.00012),     # elastic-constant preoptimization
}
DEFAULT_MAXCYCLE = 100
DEFAULT_TOLDEE_OPT = 1e-7               # p.189 (OPTGEOM TOLDEE keyword: p.193)
DEFAULT_TOLDEE_SCF_OPT = 1e-7           # 7.2.5: the SCF threshold in an optimization (p.189)
DEFAULT_TOLDEE_SCF_SP = 1e-6            # block-3 TOLDEE default (p.130)
PAGES = {"toldeg": 189, "toldex": 189, "maxgrad": 189, "maxdisp": 189, "maxcycle": 189,
         "toldee_opt": 189, "toldee_scf_opt": 189, "toldee_scf_sp": 130}

# How an optimization ended, as one sentence each (the GUI's Problems panel,
# the run card and `mace analyze --convergence` use the same words).
# CRYSTAL23's stop sentences (strings of the binary): extra evidence for _classify
_STOP_TEXTS = (("CONVERGENCE ON GRADIENT - NO CHANGE IN ENERGY FOR", "gradient_and_energy"),
               ("FLAT SURFACE - NO CHANGE IN ENERGY FOR", "energy_stalled"),
               ("CONVERGENCE TESTS NOT SATISFIED AFTER", "maxcycle"),
               ("CONVERGENCE TESTS UNSATISFIED AFTER", "maxcycle"))
REASONS = ("all_criteria", "gradient_first_cycle", "gradient_and_energy", "energy_stalled",
           "maxcycle", "failed", "unfinished", "unknown")


def _f(x: str) -> Optional[float]:
    try:
        v = float(x.replace("D", "E"))
    except (ValueError, AttributeError):
        return None
    return v if math.isfinite(v) else None


def _six(line: str) -> Optional[List[float]]:
    parts = line.split()
    if len(parts) < 6:
        return None
    try:
        return [float(x) for x in parts[:6]]
    except ValueError:
        return None


# ------------------------------------------------------------------ the d12
def _d12_values(d12: Path) -> Dict[str, Any]:
    """TOLDEG/TOLDEX/TOLDEE/MAXCYCLE from a deck: inside OPTGEOM..ENDOPT they
    are the optimization's values, elsewhere TOLDEE/MAXCYCLE are the SCF's
    (the same keyword spellings settings_extractor reads). Also the context
    of the default table (FREQCALC/EOS/elastic preoptimization)."""
    out: Dict[str, Any] = {}
    try:
        lines = d12.read_text(errors="replace").splitlines()[1:]      # line 1 is the free-text title
    except OSError:
        return out
    recs = [ln.strip().upper() for ln in lines]
    in_opt = False
    ctx = "standard"
    keys = set(recs)
    if "FREQCALC" in keys and ("PREOPTGEOM" in keys or "OPTGEOM" in keys):
        ctx = "freq"
    elif "EOS" in keys and "PREOPTGEOM" in keys:
        ctx = "eos"
    elif ({"ELASTCON", "ELAPIEZO"} & keys) and "PREOPTGEOM" in keys:
        ctx = "elastic"
    out["context"] = ctx
    for i, r in enumerate(recs):
        nxt = recs[i + 1].split()[0] if i + 1 < len(recs) and recs[i + 1].split() else ""
        if r in ("OPTGEOM", "PREOPTGEOM"):
            in_opt = True
            continue
        if r == "ENDOPT":
            in_opt = False
            continue
        if r == "TOLDEG" and in_opt:
            out["toldeg"] = _f(nxt)
        elif r == "TOLDEX" and in_opt:
            out["toldex"] = _f(nxt)
        elif r == "TOLDEE":
            v = _f(nxt)
            if v is not None:
                out["toldee_opt" if in_opt else "toldee_scf"] = 10.0 ** (-abs(v))
        elif r == "MAXCYCLE":
            v = _f(nxt)
            if v is not None:
                out["maxcycle" if in_opt else "scf_maxcycle"] = int(v)
    return {k: v for k, v in out.items() if v is not None}


# ------------------------------------------------------------------ the parser
class _State:
    def __init__(self):
        self.points: List[Dict[str, Any]] = []
        self.cur: Optional[Dict[str, Any]] = None      # the point whose lines are being read
        self.blocks: List[Dict[str, Any]] = []         # real geometry blocks: {line, params, volume}
        self.pending_block: Optional[Dict[str, Any]] = None
        self.restarts: List[int] = []
        self.satisfied: List[int] = []
        self.end: Optional[Dict[str, Any]] = None
        self.first_cycle_line: Optional[int] = None
        self.warning_lines: List[Tuple[int, str]] = []
        self.stop_texts: List[Tuple[int, str]] = []      # CRYSTAL23's own stop sentences (extra evidence)
        self.units_note = False
        self.opt_control = False
        self.scf: List[Dict[str, Any]] = []
        self.scf_cur: Optional[Dict[str, Any]] = None
        self.moved = False
        self.out_thr: Dict[str, Tuple[float, int]] = {}
        self.complete = False          # CRYSTAL's normal end (TOTAL CPU TIME)
        self.terminated = False        # CRYSTAL's error end (EEEEEEEEEE TERMINATION)
        self.last_text = ""


def _thr(st: _State, key: str, value: Optional[float], line: int, overwrite: bool = False):
    if value is None:
        return
    if overwrite or key not in st.out_thr:
        st.out_thr[key] = (value, line)


def _line(st: _State, raw: str, n: int):
    """One line (1-based number n) into the state."""
    s = raw
    if "OPTIMIZATION - POINT" in s:
        m = _POINT.search(s)
        if m:
            st.cur = {"point": int(m.group(1)), "line": n, "energy_line": None}
            st.scf_cur = None
            return
    if "PRIMITIVE CELL" in s and _BLOCK_START.match(s):
        m = _VOLUME.search(s)
        st.pending_block = {"line": n, "params": None, "n_atoms": None, "seen": 0,
                            "volume": _f(m.group(1)) if m else None, "header": False}
        return
    pb = st.pending_block
    if pb is not None:
        pb["seen"] += 1
        if pb["params"] is None and pb["seen"] <= 3:
            if "ALPHA" in s and "GAMMA" in s:
                pb["header"] = True
                return
            if pb["header"]:
                pb["params"] = _six(s)
                pb["header"] = False
                return
        if pb["n_atoms"] is None and pb["seen"] <= 7:
            m = _UNIT_CELL_ATOMS.search(s)
            if m:
                pb["n_atoms"] = int(m.group(1))
                if pb["params"] and pb["n_atoms"] > 0:
                    st.blocks.append({"line": pb["line"], "params": pb["params"], "volume": pb["volume"]})
                st.pending_block = None
                return
        if pb["seen"] > 7:
            st.pending_block = None
    if "TOTAL ENERGY" in s:
        m = _ENERGY.search(s)
        if m:
            cyc, e, de = int(m.group(1)), _f(m.group(2)), _f(m.group(3)) if m.group(3) else None
            p = st.cur
            if p is not None and p.get("energy_line") is None and e is not None:
                p.update(energy_line=n, energy_scf_au=e, energy_au=e, scf_cycles=cyc, de_printed=de,
                         criteria={}, corrected=False)
                st.points.append(p)
            return
        m = _CORRECTED.search(s)
        if m:
            p = st.cur
            if p is not None and p.get("energy_line") is not None and not p.get("corrected"):
                v = _f(m.group(1))
                if v is not None:
                    p["energy_au"] = v
                    p["corrected"] = True
            return
    if "CONVERGED" in s and ("GRADIENT" in s or "DISPLAC." in s):
        m = _CRIT.match(s)
        if m and st.cur is not None and st.cur.get("energy_line") is not None:
            key = _CRIT_KEY[(m.group(1), m.group(2))]
            st.cur["criteria"][key] = {"value": _f(m.group(3)), "threshold": _f(m.group(4)),
                                       "converged": m.group(5) == "YES", "line": n}
            thr_key = {"max_grad": "maxgrad", "rms_grad": "toldeg", "max_disp": "maxdisp", "rms_disp": "toldex"}[key]
            _thr(st, thr_key, _f(m.group(4)), n)
            return
    if "CONVERGENCE TESTS SATISFIED" in s:
        st.satisfied.append(n)
        return
    if "FINALRUN OPTION: RE-STARTING" in s:
        st.restarts.append(n)
        return
    if FIRST_CYCLE_TEXT in s:
        st.first_cycle_line = n
        st.warning_lines.append((n, s.strip()))
        return
    for text, why in _STOP_TEXTS:
        if text in s:
            st.stop_texts.append((n, why))
            st.warning_lines.append((n, s.strip()))
            return
    if "GEOMETRY OPTIMIZATION STOPPED" in s or (st.end is None and "WARNING" in s and st.points
                                                    and n > (st.points[-1].get("energy_line") or 0)):
        st.warning_lines.append((n, s.strip()))
    if "OPT END" in s:
        m = _OPT_END.search(s)
        if m:
            me, mp = _OPT_END_E.search(s), _OPT_END_PTS.search(s)
            st.end = {"status": m.group(1).lower(), "line": n, "text": s.strip().strip("*").strip(),
                      "energy": _f(me.group(1)) if me else None, "points": int(mp.group(1)) if mp else None}
        return
    if "ARE GIVEN WITHOUT" in s:
        st.units_note = True
        return
    if "OPTIMIZATION CONTROL" in s:
        st.opt_control = True
        return
    if "CYC" in s and "ETOT(AU)" in s:
        m = _CYC.match(s)
        if m:
            c = int(m.group(1))
            if st.scf_cur is None or c == 0:
                st.scf_cur = {"point": st.cur["point"] if st.cur else None, "start_line": n, "cycles": [], "ended": None}
                st.scf.append(st.scf_cur)
            st.scf_cur["cycles"].append({"cyc": c, "etot": _f(m.group(2)), "detot": _f(m.group(3)), "line": n})
            return
    if "== SCF ENDED" in s:
        if st.scf_cur is not None:
            st.scf_cur["ended"] = "too_many_cycles" if "TOO MANY CYCLES" in s else "energy"
            st.scf_cur["end_line"] = n
            st.scf_cur = None
        return
    if "TOO MANY CYCLES" in s and st.scf_cur is not None:
        st.scf_cur["ended"] = "too_many_cycles"
        st.scf_cur["end_line"] = n
        st.scf_cur = None
        return
    if "INFORMATION ON SCF MOVED TO SCFOUT.LOG" in s:
        st.moved = True
        return
    if "TOTAL CPU TIME =" in s:
        st.complete = True
        return
    if "EEEEEEEEEE TERMINATION" in s:
        st.terminated = True
        return
    if "COMPONENT" in s:
        m = _HDR_MAX.search(s)
        if m:
            _thr(st, "maxgrad", _f(m.group(1)), n, True)
            _thr(st, "maxdisp", _f(m.group(2)), n, True)
            return
        m = _HDR_RMS.search(s)
        if m:
            _thr(st, "toldeg", _f(m.group(1)), n, True)
            _thr(st, "toldex", _f(m.group(2)), n, True)
            return
    if "THRESHOLD ON ENERGY CHANGE" in s:
        m = _HDR_DE.search(s)
        if m:
            _thr(st, "toldee_opt", _f(m.group(1)), n, True)
        return
    if "MAXIMUM ALLOWED NUMBER OF STEPS" in s:
        m = _HDR_STEPS.search(s)
        if m:
            _thr(st, "maxcycle", float(m.group(1)), n, True)
        return
    if "INFORMATION" in s and "****" in s:
        for rx, key, conv in ((_INFO_TOLDEE_OPT, "toldee_opt", lambda v: 10.0 ** -int(v)),
                              (_INFO_TOLDEE_SCF, "toldee_scf", lambda v: 10.0 ** -int(v)),
                              (_INFO_MAXCYC_OPT, "maxcycle", float), (_INFO_MAXCYC_SCF, "scf_maxcycle", float),
                              (_INFO_TOLDEG, "toldeg", _f), (_INFO_TOLDEX, "toldex", _f)):
            m = rx.search(s)
            if m:
                _thr(st, key, conv(m.group(1)), n)
                return
        return
    if "CONVERGENCE ON ENERGY" in s and "10**" in s:
        m = _SCF_TOLDEE.search(s)
        if m:
            _thr(st, "toldee_scf", 10.0 ** -int(m.group(1)), n, True)
        m = _SCF_MAXCYC.search(s)
        if m:
            _thr(st, "scf_maxcycle", float(m.group(1)), n, True)
        return
    if "MAX NUMBER OF SCF CYCLES" in s:
        m = _SCF_MAXCYC.search(s)
        if m:
            _thr(st, "scf_maxcycle", float(m.group(1)), n, True)


def _stream(path: Path, size: Optional[int], st: _State, complete_lines_only: bool) -> int:
    """Feed the file's lines [0, size) into the state; returns the line count.
    With ``complete_lines_only`` a trailing segment without a newline (a line
    the job is still writing) is not parsed."""
    n = 0
    tail = b""
    left = size
    with open(path, "rb") as fh:
        while True:
            want = CHUNK if left is None else min(CHUNK, left)
            if want <= 0:
                break
            chunk = fh.read(want)
            if not chunk:
                break
            if left is not None:
                left -= len(chunk)
            parts = (tail + chunk).split(b"\n")
            tail = parts.pop()
            for raw in parts:
                n += 1
                _line(st, raw.decode("utf-8", "replace").rstrip("\r"), n)
    if tail and not complete_lines_only:
        n += 1
        _line(st, tail.decode("utf-8", "replace").rstrip("\r"), n)
    return n


def frame_of_point(points: List[Dict[str, Any]], frame_lines: List[int]) -> Dict[int, int]:
    """CRYSTAL POINT -> frame index (normative mapping, plan D1a): a point's
    frame is the last geometry block that starts before the point's energy
    line (the block whose energy outindex reports). ``frame_lines`` are the
    1-based start lines of the frames, in order. The JS twin is
    ``runplot.js frameOfPoint``."""
    out = {}
    j = -1
    for p in points:
        el = p.get("energy_line")
        if el is None:
            continue
        while j + 1 < len(frame_lines) and frame_lines[j + 1] < el:
            j += 1
        if j >= 0:
            out[int(p["point"])] = j
    return out


def _thresholds(st: _State, d12: Dict[str, Any], is_opt: bool) -> Dict[str, Any]:
    ctx = d12.get("context", "standard")
    rms_g, rms_x = CONTEXT_DEFAULTS.get(ctx, CONTEXT_DEFAULTS["standard"])
    defaults = {"toldeg": rms_g, "toldex": rms_x, "maxgrad": 1.5 * rms_g, "maxdisp": 1.5 * rms_x,
                "maxcycle": float(DEFAULT_MAXCYCLE), "toldee_opt": DEFAULT_TOLDEE_OPT,
                "toldee_scf": DEFAULT_TOLDEE_SCF_OPT if is_opt else DEFAULT_TOLDEE_SCF_SP}
    keys = ["toldeg", "toldex", "maxgrad", "maxdisp", "maxcycle", "toldee_opt", "toldee_scf"] if is_opt else ["toldee_scf"]
    out: Dict[str, Any] = {}
    for k in keys:
        if k in st.out_thr:
            v, line = st.out_thr[k]
            out[k] = {"value": v, "source": "out", "line": line}
        elif k in d12 or (k == "maxgrad" and "toldeg" in d12) or (k == "maxdisp" and "toldex" in d12):
            v = d12.get(k)
            if v is None:          # CRYSTAL derives the maximum components from the RMS ones
                v = 1.5 * d12["toldeg" if k == "maxgrad" else "toldex"]
            out[k] = {"value": v, "source": "d12"}
        else:
            page = PAGES["toldee_scf_" + ("opt" if is_opt else "sp")] if k == "toldee_scf" else PAGES[k]
            out[k] = {"value": defaults[k], "source": "default", "manual_page": page,
                      "context": ctx}
    if "scf_maxcycle" in st.out_thr:
        v, line = st.out_thr["scf_maxcycle"]
        out["scf_maxcycle"] = {"value": v, "source": "out", "line": line}
    elif "scf_maxcycle" in d12:
        out["scf_maxcycle"] = {"value": float(d12["scf_maxcycle"]), "source": "d12"}
    for k in ("maxcycle", "scf_maxcycle"):
        if k in out and out[k]["value"] is not None:
            out[k]["value"] = int(out[k]["value"])
    return out


def _classify(st: _State, points: List[Dict[str, Any]], thr: Dict[str, Any]) -> Dict[str, Any]:
    end = st.end
    toldee = (thr.get("toldee_opt") or {}).get("value")
    maxcycle = (thr.get("maxcycle") or {}).get("value")
    last = points[-1] if points else None
    conv = (last or {}).get("converged") or {}

    def yes(k):
        return bool(conv.get(k))

    grads = yes("max_grad") and yes("rms_grad")
    disps = yes("max_disp") and yes("rms_disp")

    def small(p):
        de = p.get("de_printed")
        return de is not None and toldee is not None and abs(de) < toldee

    reason = "unknown"
    if end is not None and end["status"] == "failed":
        reason = "failed"
    elif end is not None:
        if st.first_cycle_line is not None and st.first_cycle_line < end["line"]:
            reason = "gradient_first_cycle"
        elif grads and disps:
            reason = "all_criteria"
        elif grads and not disps and last is not None and small(last):
            reason = "gradient_and_energy"
        elif not grads and len(points) >= 4 and all(small(p) for p in points[-4:]):
            reason = "energy_stalled"
    elif points and maxcycle is not None and len(points) >= maxcycle:
        reason = "maxcycle"
    elif not st.complete:
        reason = "unfinished"
    if reason == "unknown" and st.stop_texts:
        # the state did not decide it: CRYSTAL's own stop sentence does (read from the CRYSTAL23 binary)
        reason = st.stop_texts[-1][1]
    warn = [t for (n, t) in st.warning_lines if last is None or n > (last.get("energy_line") or 0)]
    out = {"reason": reason, "line": end["line"] if end else None, "text": end["text"] if end else None,
           "energy": end["energy"] if end else None, "points": end["points"] if end else None,
           "warning_lines": [w for w in warn if w and set(w) != set("WARNING")]}
    return out


def parse(path, size: Optional[int] = None, complete_lines_only: bool = False,
          frame_lines: Optional[List[int]] = None, d12: Optional[Path] = None) -> Dict[str, Any]:
    """Everything above as one JSON-able dict. ``size`` bounds the read
    (a snapshot of a file still being written); ``frame_lines`` are the
    frames' 1-based block start lines (outindex's), else the parser's own
    block detection is used (the same blocks on every corpus output; a test
    compares them)."""
    path = Path(path)
    st = _State()
    n_lines = _stream(path, size, st, complete_lines_only)
    if d12 is None:
        cand = path.with_suffix(".d12")
        d12 = cand if cand.is_file() else None
    d12v = _d12_values(d12) if d12 is not None else {}
    points = [p for p in st.points if p.get("energy_line") is not None]
    is_opt = bool(points) or st.opt_control or st.end is not None
    thr = _thresholds(st, d12v, is_opt)

    flines = list(frame_lines) if frame_lines is not None else [b["line"] for b in st.blocks]
    fmap = frame_of_point(points, flines)
    blocks_by_line = {b["line"]: b for b in st.blocks}
    final_frame = None
    if st.end is not None:
        after = [k for k, ln in enumerate(flines) if ln > st.end["line"]]
        final_frame = after[0] if after else None

    # segments at each FINALRUN restart
    segs: List[Dict[str, Any]] = []
    seg = 0
    ri = 0
    for p in points:
        while ri < len(st.restarts) and st.restarts[ri] < p["line"]:
            if segs and segs[-1]["restart_line"] is None:
                segs[-1]["restart_line"] = st.restarts[ri]
            ri += 1
            if segs and segs[-1]["index"] == seg:
                seg += 1
        if not segs or segs[-1]["index"] != seg:
            segs.append({"index": seg, "first_point": p["point"], "last_point": p["point"], "restart_line": None})
        segs[-1]["last_point"] = p["point"]
        p["segment"] = seg

    out_points = []
    for p in points:
        fr = fmap.get(p["point"])
        blk = None
        if fr is not None and fr < len(flines):
            blk = blocks_by_line.get(flines[fr])
        cell = None
        if blk is not None and blk["params"]:
            a, b, c, al, be, ga = blk["params"]
            cell = {"a": a, "b": b, "c": c, "alpha": al, "beta": be, "gamma": ga, "volume": blk["volume"]}
        crit = p["criteria"]
        out_points.append({
            "point": p["point"], "frame": fr, "line": p["line"], "energy_line": p["energy_line"],
            "segment": p.get("segment", 0), "energy_au": p["energy_au"], "energy_scf_au": p["energy_scf_au"],
            "scf_cycles": p["scf_cycles"], "de_printed": p["de_printed"],
            **{k: (crit[k]["value"] if k in crit else None) for k in CRITERIA},
            "thr": {k: (crit[k]["threshold"] if k in crit else None) for k in CRITERIA},
            "converged": {k: (crit[k]["converged"] if k in crit else None) for k in CRITERIA},
            "cell": cell,
        })

    units = "internal" if st.units_note else "Ha/bohr"
    res = {
        "version": VERSION, "file": path.name, "lines": n_lines, "is_opt": is_opt,
        "points": out_points, "segments": segs, "final_frame": final_frame,
        "n_frames": len(flines),
        "opt": {"end": _classify(st, out_points, thr) if is_opt else None,
                "satisfied": st.satisfied, "finalrun_restarts": st.restarts},
        "scf": {"detail": st.scf, "moved_to_scfout_log": st.moved},
        "thresholds": thr,
        "gradient_units": units if is_opt else None,
        "displacement_units": ("internal" if st.units_note else "bohr") if is_opt else None,
        "complete": st.complete, "terminated": st.terminated,
    }
    return res


# ------------------------------------------------------------------ energy along the run
HARTREE_EV = 27.211386245988
# eV per unit, as the GUI's energy chart converts (app.js E_UNITS, plotkit ENERGY_UNITS)
ENERGY_UNITS = {"eV": 1.0, "meV": 1000.0, "Ha": 1 / HARTREE_EV, "kJ/mol": 96.48533212331, "kcal/mol": 23.060547830619}
SEGMENT_MODES = ("continuous", "separate")
RELATIVE_TO = ("final", "first", "minimum", "absolute")


def energy_along_run(res: Dict[str, Any], relative_to: str = "final",
                     segments: str = "continuous") -> List[Dict[str, Any]]:
    """The energy along the run as the GUI's chart draws it, one row per
    optimization point plus the OPT END energy (the final geometry, at the
    last point, ``final`` True).

    ``segments="continuous"`` (the default) takes one reference for the whole
    run, so the curve runs on across the FINALRUN restarts; ``"separate"``
    takes one per FINALRUN segment (each segment relative to its own first or
    final point). ``relative_to`` is "final", "first", "minimum" or
    "absolute" (no reference). ``de_au`` is E - E_ref in Hartree; ``is_ref``
    marks the reference itself (zero, so no value on a log axis) to the
    printed precision of 1e-10 Ha."""
    if segments not in SEGMENT_MODES:
        raise ValueError(f"segments is one of {', '.join(SEGMENT_MODES)}")
    if relative_to not in RELATIVE_TO:
        raise ValueError(f"relative_to is one of {', '.join(RELATIVE_TO)}")
    rows: List[Dict[str, Any]] = [
        {"point": p["point"], "segment": p.get("segment", 0), "frame": p.get("frame"), "final": False,
         "energy_au": p["energy_au"]}
        for p in res.get("points") or [] if p.get("energy_au") is not None]
    end = (res.get("opt") or {}).get("end") or {}
    if rows and end.get("energy") is not None:
        rows.append({"point": rows[-1]["point"], "segment": rows[-1]["segment"], "frame": res.get("final_frame"),
                     "final": True, "energy_au": end["energy"]})

    def group(r):
        return 0 if segments == "continuous" else r["segment"]

    ref: Dict[int, float] = {}
    for r in rows:
        g, e = group(r), r["energy_au"]
        if relative_to == "first":
            ref.setdefault(g, e)
        elif relative_to == "final":
            ref[g] = e
        elif relative_to == "minimum":
            ref[g] = min(ref.get(g, math.inf), e)
        else:
            ref[g] = 0.0
    for r in rows:
        e0 = ref[group(r)]
        r["ref_au"] = e0
        r["de_au"] = r["energy_au"] - e0
        r["is_ref"] = relative_to != "absolute" and abs(r["de_au"]) <= 1e-10
        r["below"] = relative_to == "final" and r["energy_au"] < e0 - 1e-10
    return rows


def energy_tsv(rows: List[Dict[str, Any]], unit: str = "meV") -> str:
    """The rows as the GUI's Copy TSV writes them: point, FINALRUN segment
    (1-based), frame, E in Hartree, ΔE in ``unit``."""
    if unit not in ENERGY_UNITS:
        raise ValueError(f"unit is one of {', '.join(ENERGY_UNITS)}")
    f = HARTREE_EV * ENERGY_UNITS[unit]
    out = [f"point\tsegment\tframe\tE_Ha\tdE_{unit}"]
    for r in rows:
        fr = "" if r["frame"] is None else r["frame"]
        out.append(f"{r['point']}\t{r['segment'] + 1}\t{fr}\t{r['energy_au']!r}\t{r['de_au'] * f!r}")
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------ words
def stop_sentence(res: Dict[str, Any]) -> str:
    """The one sentence the GUI's Problems row, run card and the CLI print."""
    end = (res.get("opt") or {}).get("end")
    if not res.get("is_opt") or end is None:
        detail = (res.get("scf") or {}).get("detail") or []
        if any(b.get("ended") == "too_many_cycles" for b in detail):
            return "The SCF did not converge (TOO MANY CYCLES)."
        return "Single-point run: no optimization." if res.get("complete") else "Single-point run, not finished."
    r = end["reason"]
    pts = res.get("points") or []
    n = pts[-1]["point"] if pts else end.get("points")
    if r == "all_criteria":
        return f"Converged: all four criteria met at point {n} (CRYSTAL23 manual p.188)."
    if r == "gradient_first_cycle":
        return ("Stopped after the first point: the gradients were already below threshold (CRYSTAL warning). "
                "The geometry is the input geometry.")
    if r == "gradient_and_energy":
        return ("Stopped with a warning: gradients converged and the energy change was below TOLDEE, "
                "but the displacements were not (manual p.208).")
    if r == "energy_stalled":
        return ("Stopped with a warning: the energy did not change (TOLDEE) for four points while the "
                "criteria were unmet (manual p.208).")
    if r == "maxcycle":
        return f"Stopped at MAXCYCLE ({n} points) without converging."
    if r == "failed":
        return "Geometry optimization not converged (OPT END - FAILED)."
    if r == "unfinished":
        if n is None:
            return "Unfinished: the run stopped before its first optimization point finished."
        return f"Unfinished: no OPT END line after point {n}."
    return f"Ended with: {end.get('text') or 'no end line'}"


def _fmt(v, spec=".6f"):
    return "-" if v is None else format(v, spec)


def render_text(res: Dict[str, Any]) -> str:
    """The `mace analyze --convergence` table, stop sentence and footer."""
    rows = []
    thr = res["thresholds"]
    if res["is_opt"]:
        rows.append(f"{'point':>5} {'seg':>3} {'SCF':>4} {'DE (Ha)':>11}   "
                    f"{'max grad':>10} {'rms grad':>10} {'max disp':>10} {'rms disp':>10}")
        for p in res["points"]:
            cells = []
            for k in CRITERIA:
                v, ok = p[k], p["converged"][k]
                cells.append(f"{_fmt(v):>9}{'' if ok is None else ('Y' if ok else 'N')}" if v is not None else f"{'-':>10}")
            de = "" if p["de_printed"] is None else format(p["de_printed"], ".3e")
            rows.append(f"{p['point']:>5} {p['segment']:>3} {p['scf_cycles']:>4} {de:>11}   " + " ".join(f"{c:>10}" for c in cells))
        rows.append("  threshold" + " " * 23 + " ".join(
            f"{_fmt((thr.get(k) or {}).get('value')):>10}" for k in ("maxgrad", "toldeg", "maxdisp", "toldex")))
        rows.append("(Y = CONVERGED YES, N = NO, as CRYSTAL printed them)")
    rows.append("")
    rows.append(stop_sentence(res))
    rows.append("")
    src = []
    for k, d in thr.items():
        where = {"out": f"from the .out (line {d.get('line')})", "d12": "from the .d12",
                 "default": f"CRYSTAL23 default (manual p.{d.get('manual_page')})"}[d["source"]]
        src.append(f"  {k.upper():<13} {d['value']:<10g} {where}")
    rows.append("Thresholds:")
    rows += src
    if res.get("gradient_units") == "internal":
        rows.append("Units: gradients and displacements are in CRYSTAL's internal optimization coordinates "
                    "(the .out says they are given without units).")
    elif res.get("gradient_units"):
        rows.append("Units: gradient in Ha/bohr, displacement in bohr.")
    if res["scf"]["moved_to_scfout_log"]:
        rows.append("Per-cycle SCF lines are printed only for the first point; CRYSTAL moved the rest to SCFOUT.LOG.")
    return "\n".join(rows)


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="mace analyze --convergence",
                                 description="Optimization and SCF convergence of a CRYSTAL output.")
    ap.add_argument("output", help="a CRYSTAL .out file")
    ap.add_argument("--json", action="store_true", help="print the parsed result as JSON")
    ap.add_argument("--energy", action="store_true",
                    help="print the energy along the run instead (tab-separated, as the GUI's Copy TSV; "
                         "with --json, the rows as JSON)")
    ap.add_argument("--segments", choices=SEGMENT_MODES, default="continuous",
                    help="with --energy: one reference for the whole run (continuous, the default) "
                         "or one per FINALRUN segment (separate)")
    ap.add_argument("--relative-to", choices=RELATIVE_TO, default="final",
                    help="with --energy: the reference point (default: final)")
    ap.add_argument("--unit", choices=tuple(ENERGY_UNITS), default="meV", help="with --energy: the ΔE unit (default: meV)")
    a = ap.parse_args(argv)
    p = Path(a.output)
    if not p.is_file():
        print(f"Error: {p} is not a file", file=sys.stderr)
        return 2
    res = parse(p)
    if a.energy:
        rows = energy_along_run(res, relative_to=a.relative_to, segments=a.segments)
        if a.json:
            print(json.dumps(rows, indent=1))
        else:
            sys.stdout.write(energy_tsv(rows, a.unit))
        return 0
    if a.json:
        print(json.dumps(res, indent=1))
    else:
        print(render_text(res))
    return 0


if __name__ == "__main__":       # pragma: no cover
    sys.exit(main())
