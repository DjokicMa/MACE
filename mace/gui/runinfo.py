"""A run's analysis for the GUI (PLAN_analysis C2, C6, C7): convergence,
problems from the .out, the run's metadata card and a measurement tracked
along the run. Read only; nothing is written into the folder.

* ``convergence`` is mace.utils.out_convergence (the module `mace analyze
  --convergence` prints), given outindex's frame lines so its POINT -> frame
  mapping is the run document's.
* ``problems`` turns the completion checker's verdict (the words `mace
  status` and `mace check` use), the convergence stop, the SCF, FINALRUN
  restarts, CRYSTAL WARNING lines, tolerances that differ from the CRYSTAL23
  defaults and imaginary modes into rows {severity, code, message, line, lines?,
  action}.
* ``meta_job`` (a pool job; its prints never reach the server's log) gives
  the card's fields from the property extractor's own parsing helpers, so the
  card shows what the database would store.
* ``track`` follows a distance or angle through every frame, carrying each
  atom's periodic image from frame to frame (the nearest-image rule of
  structures.compare()), in Cartesian coordinates with each frame's cell.
"""
from __future__ import annotations

import contextlib
import io
import math
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

from mace.utils.imaginary_modes import imaginary, is_linear, is_molecule  # noqa: F401 (one rule with the extractor)

from mace.constants import HARTREE_TO_EV as HARTREE_EV  # noqa: E402
DEFAULT_TOLINTEG = (6, 6, 6, 6, 12)


# ------------------------------------------------------------------ convergence
def convergence(path: Path, idx=None, sibling_ok=None) -> Dict[str, Any]:
    from mace.utils import out_convergence as oc
    from mace.utils import livefile
    snap = livefile.snapshot(path)
    live = livefile.is_live(path, snap)
    flines = [b.line + 1 for b in idx.blocks] if idx is not None else None
    res = oc.parse(path, size=snap.size, complete_lines_only=live, frame_lines=flines, sibling_ok=sibling_ok)
    if idx is not None and getattr(idx, "molecule", False):
        # a molecule prints only its initial and final geometry tables: a frame belongs to the
        # first point after it, and the later points have no geometry of their own
        taken = set()
        for q in res.get("points") or []:
            fr = q.get("frame")
            if fr is None:
                continue
            if fr in taken:
                q["frame"], q["cell"] = None, None
            else:
                taken.add(fr)
    if idx is not None and res.get("final_frame") is None and idx.labels and idx.labels[-1] == "final":
        res["final_frame"] = len(idx.labels) - 1
    res["live"] = live
    res["stop"] = oc.stop_sentence(res)
    # the SCF timeline the run view's SCF card and Run Analysis › SCF draw (`--scf` prints the same)
    res["scf_timeline"] = oc.scf_timeline(res)
    res["command"] = f"mace analyze --convergence {path.name}"
    return res


# ------------------------------------------------------------------ problems
def _first_line(path: Path, needles, limit: int = 200_000_000) -> Optional[int]:
    """The 1-based line of the first line holding any of ``needles``."""
    low = [n.lower() for n in needles]
    with open(path, "r", errors="replace") as fh:
        for n, line in enumerate(fh, 1):
            ll = line.lower()
            if any(x in ll for x in low):
                return n
    return None


def _checker(path: Path):
    from mace.completion_checker import ERROR_DESCRIPTIONS, ERROR_PATTERNS, categorize_output_file
    with contextlib.redirect_stdout(io.StringIO()):
        cat, _ = categorize_output_file(path)
    return cat, ERROR_PATTERNS, ERROR_DESCRIPTIONS


def _tolinteg(path: Path) -> Optional[tuple]:
    rx = re.compile(r"(COULOMB OVERLAP TOL|COULOMB PENETRATION TOL|EXCHANGE OVERLAP TOL|EXCHANGE PSEUDO OVP \(F\(G\)\)|"
                    r"EXCHANGE PSEUDO OVP \(P\(G\)\))\s+\(T\d\)\s+10\*\*\s*(-?\d+)")
    vals, line_no = {}, None
    with open(path, "r", errors="replace") as fh:
        for n, line in enumerate(fh, 1):
            m = rx.search(line)
            if m:
                vals.setdefault(m.group(1), abs(int(m.group(2))))
                line_no = line_no or n
            if len(vals) == 5 or n > 3000:
                break
    if len(vals) < 5:
        return None
    order = ["COULOMB OVERLAP TOL", "COULOMB PENETRATION TOL", "EXCHANGE OVERLAP TOL",
             "EXCHANGE PSEUDO OVP (F(G))", "EXCHANGE PSEUDO OVP (P(G))"]
    return tuple(vals[k] for k in order), line_no


def _frequencies(text: str) -> List[float]:
    """Every mode's frequency (cm-1), degenerate rows expanded, read by the
    property extractor's own FREQ parser."""
    from mace.utils.property_extractor import CrystalPropertyExtractor
    x = CrystalPropertyExtractor(db_path=None, enable_tracking=False)
    with contextlib.redirect_stdout(io.StringIO()):
        props = x._extract_frequency_properties(text)
    out = []
    for row in props.get("vibrational_frequencies") or []:
        out += [row["frequency_cm"]] * (row["mode_end"] - row["mode_start"] + 1)
    return out


def _linear_molecule(text: str) -> bool:
    from mace.utils.property_extractor import CrystalPropertyExtractor
    x = CrystalPropertyExtractor(db_path=None, enable_tracking=False)
    with contextlib.redirect_stdout(io.StringIO()):
        pos = x._extract_atomic_positions(text)
    return is_linear([(p["x"], p["y"], p["z"]) for p in pos])


def problems(path: Path, conv: Dict[str, Any], noise_cm: float = 20.0, dim: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    cat, patterns, descriptions = _checker(path)
    end = (conv.get("opt") or {}).get("end") if conv.get("is_opt") else None
    reason = end["reason"] if end else None
    live = conv.get("live")
    # ---- how the run stopped
    if cat in descriptions:
        rows.append({"severity": "error", "code": "stop", "message": descriptions[cat],
                     "line": _first_line(path, patterns.get(cat, [])), "action": "show"})
    elif cat.startswith("complete"):
        if end is not None:
            sev = "warning" if reason in ("gradient_first_cycle", "gradient_and_energy", "energy_stalled") else \
                "info" if reason == "all_criteria" else "error" if reason in ("failed", "maxcycle") else "warning"
            rows.append({"severity": sev, "code": "stop", "message": conv.get("stop") or "", "line": end.get("line"),
                         "reason": reason, "action": "show"})
        else:
            rows.append({"severity": "info", "code": "stop", "message": "Finished normally.", "line": None, "action": None})
    elif cat == "ongoing":
        msg = "Still running." if live else (conv.get("stop") if end is not None else "Unfinished: no end marker.")
        rows.append({"severity": "info" if live else "warning", "code": "stop", "message": msg,
                     "line": (end or {}).get("line"), "action": "show" if (end or {}).get("line") else None})
    else:
        line = (end or {}).get("line")
        rows.append({"severity": "warning", "code": "stop",
                     "message": f"Ended with: {(end or {}).get('text') or cat}", "line": line, "action": "show" if line else None})
    # ---- SCF not converged (only where CRYSTAL printed the cycles)
    bad = [b for b in (conv.get("scf") or {}).get("detail") or [] if b.get("ended") == "too_many_cycles"]
    if bad or cat == "too_many_scf":
        line = bad[0].get("end_line") if bad else _first_line(path, ["TOO MANY CYCLES"])
        rows.append({"severity": "warning", "code": "scf", "message": "The SCF did not converge (TOO MANY CYCLES).",
                     "line": line, "action": "show"})
    # ---- FINALRUN restarts
    restarts = (conv.get("opt") or {}).get("finalrun_restarts") or []
    if restarts:
        converged = reason in ("all_criteria", "gradient_first_cycle", "gradient_and_energy", "energy_stalled")
        n = len(restarts)
        rows.append({"severity": "info" if converged or live else "warning", "code": "finalrun",
                     "message": (f"FINALRUN restarted the optimization {n} time{'s' if n != 1 else ''} "
                                 "(normal: each restart re-references the integrals)") if converged or live else
                     f"FINALRUN restarted the optimization {n} time{'s' if n != 1 else ''} and the run did not converge",
                     "line": restarts[0], "lines": restarts, "action": "show"})
    # ---- CRYSTAL WARNING lines (deduplicated; not the banner of a classified stop). The
    # ones CRYSTAL prints while it reads the input are notes on the input (e.g. "HSEsol-3c
    # has to be combined with sol-def2-mSVP basis set"): info; from the first SCF on: warnings
    stop_text = set((end or {}).get("warning_lines") or [])
    seen: Dict[str, Dict[str, Any]] = {}
    reading_input = True
    with open(path, "r", errors="replace") as fh:
        for n, line in enumerate(fh, 1):
            s = line.strip()
            if reading_input and (s.startswith("CYC ") or "SCF ENDED" in s or "OPTIMIZATION - POINT" in s):
                reading_input = False
            if not s.startswith("WARNING") or set(s) <= set("WARNING"):
                continue
            if s in stop_text:
                continue
            key = re.sub(r"\s+", " ", s)
            if key in seen:
                seen[key]["count"] += 1
            else:
                seen[key] = {"severity": "info" if reading_input else "warning", "code": "crystal_warning",
                             "message": key, "line": n, "count": 1, "action": "show"}
    for r in seen.values():
        if r["count"] > 1:
            r["message"] = f"{r['message']} ({r['count']} times)"
        rows.append(r)
    # ---- tolerances that differ from CRYSTAL23's defaults (info, one per keyword)
    thr = conv.get("thresholds") or {}
    from mace.utils import out_convergence as oc
    is_opt = conv.get("is_opt")
    defaults = {"toldeg": oc.CONTEXT_DEFAULTS["standard"][0], "toldex": oc.CONTEXT_DEFAULTS["standard"][1],
                "toldee_opt": oc.DEFAULT_TOLDEE_OPT, "maxcycle": oc.DEFAULT_MAXCYCLE,
                "toldee_scf": oc.DEFAULT_TOLDEE_SCF_OPT if is_opt else oc.DEFAULT_TOLDEE_SCF_SP}
    names = {"toldeg": "TOLDEG", "toldex": "TOLDEX", "toldee_opt": "TOLDEE (optimization)", "toldee_scf": "TOLDEE (SCF)",
             "maxcycle": "MAXCYCLE (optimization)"}
    for k, label in names.items():
        t = thr.get(k)
        if not t or t.get("source") != "out" or t.get("value") is None:
            continue
        d = defaults[k]
        if abs(float(t["value"]) - float(d)) > 1e-12 * max(1.0, abs(float(d))):
            rows.append({"severity": "info", "code": "tolerance", "message": f"{label} {t['value']:g} (CRYSTAL23 default {d:g})",
                         "line": t.get("line"), "action": "show"})
    ti = _tolinteg(path)
    if ti and ti[0] != DEFAULT_TOLINTEG:
        rows.append({"severity": "info", "code": "tolerance",
                     "message": f"TOLINTEG {' '.join(map(str, ti[0]))} (CRYSTAL23 default {' '.join(map(str, DEFAULT_TOLINTEG))})",
                     "line": ti[1], "action": "show"})
    # ---- imaginary modes (FREQ outputs)
    try:
        text = Path(path).read_text(errors="replace") if Path(path).stat().st_size < 64 * 1024 * 1024 else ""
    except OSError:
        text = ""
    if text and ("FREQUENCIES" in text and ("VIBRATIONAL" in text or "FORCE CONSTANT MATRIX" in text)):
        freqs = _frequencies(text)
        molecule = dim == 0 if dim is not None else is_molecule(text)
        im = imaginary(freqs, molecule, _linear_molecule(text) if molecule else False, noise_cm)
        if im:
            where = " at Γ" if "PHONON FREQUENCIES AT THE GAMMA POINT" in text else ""
            n = im["n"]
            msg = (f"{n} small imaginary mode{'s' if n != 1 else ''}{where} (lowest {im['lowest']:.1f} cm⁻¹): usually "
                   "numerical noise from a numerical-gradient FREQ run") if im["severity"] == "info" else \
                f"{n} imaginary mode{'s' if n != 1 else ''}{where} (lowest {im['lowest']:.1f} cm⁻¹)"
            rows.append({"severity": im["severity"], "code": "imaginary", "message": msg,
                         "line": _first_line(path, ["MODES         EIGV", "EIGENVALUES (HARTREE**2)"]),
                         "lowest_cm": im["lowest"], "action": "modes"})
    return rows


# ------------------------------------------------------------------ the metadata card (a pool job)
_RX = {
    "start": re.compile(r"EEEEEEEEEE STARTING\s+DATE\s+(.+?)\s*$"),
    "end": re.compile(r"EEEEEEEEEE TERMINATION\s+DATE\s+(.+?)\s*$"),
    "telapse": re.compile(r"TELAPSE\s+([\d.]+)"),
    "sg": re.compile(r"SPACE GROUP \((?:CENTROSYMMETRIC|NONCENTROSYMMETRIC)\)\s*:\s*(.+?)\s*$"),
    "dim": re.compile(r"DIMENSIONALITY OF THE SYSTEM\s+(\d)"),
}


def meta_job(ctx=None, path: str = "", **_) -> Dict[str, Any]:
    """In a pool worker: the run card's fields. The extractor's parsing helpers
    (CrystalPropertyExtractor(db_path=None, enable_tracking=False): no
    database is opened or created) give what the database would store; its
    prints are kept off the worker's stdout."""
    p = Path(path)
    text = p.read_text(errors="replace")
    out: Dict[str, Any] = {"file_size": p.stat().st_size}
    with contextlib.redirect_stdout(io.StringIO()):
        from mace.utils.property_extractor import CrystalPropertyExtractor
        from mace.completion_checker import determine_completed_subtype
        # the card's functional, basis, version, processes and energy per atom: the
        # registry's run_info and energy_per_atom (what the database stores)
        from mace.properties.defs import run_info
        from mace.utils.calc_detection import mechanics_reference_text
        x = CrystalPropertyExtractor(db_path=None, enable_tracking=False)
        comp = x._extract_computational_properties(text)
        scf = x._extract_scf_settings(text)
        # an ELASTIC/EOS/QHA run's energy and structure are its reference cell's, as the
        # extractor reads them (mechanics_reference_text), not the last strained cell's
        ref = mechanics_reference_text(text)
        en = x._extract_energy_properties(ref)
        struct = x._extract_structural_properties(ref)
        cryst = x._extract_crystallographic_info(ref)
        lines = text.splitlines(True)
        calc_type = determine_completed_subtype(p, lines, has_opt_end=any("OPT END" in ln for ln in lines))
    if ctx is not None:
        ctx.progress(0.7, "Reading the run's settings")

    def first(key):
        m = _RX[key].search(text) if key not in ("start", "end") else None
        if key in ("start", "end"):
            for ln in lines:
                mm = _RX[key].search(ln)
                if mm:
                    return mm.group(1)
            return None
        return m.group(1).strip() if m else None
    tel = [float(v) for v in _RX["telapse"].findall(text[-200_000:])]
    dims = {"0": "MOLECULE", "1": "POLYMER", "2": "SLAB", "3": "CRYSTAL"}
    d = first("dim")
    n_atoms = struct.get("atoms_in_unit_cell")
    e_au = en.get("total_energy_au")
    out.update({
        "crystal_version": run_info.crystal_version(text),
        "calc_type": calc_type, "dimensionality": dims.get(d, None) if d else None,
        "space_group": first("sg") or cryst.get("space_group_number"),
        "atoms": n_atoms, "functional": _functional(text, p), "basis": _basis(p),
        "shrink": [scf.get(k) for k in ("scf_shrink_k1", "scf_shrink_k2", "scf_shrink_k3") if scf.get(k) is not None] or None,
        "gilat": scf.get("scf_gilat_shrink"), "tolinteg": (lambda t: list(t[0]) if t else None)(_tolinteg(p)),
        "spin": "unrestricted (spin-polarized)" if re.search(r"\bUNRESTRICTED|SPIN POLARIZED|SPINLOCK", text[:400_000]) else "closed shell",
        "mpi_processes": run_info.mpi_processes(text),
        "started": first("start"), "ended": first("end"),
        "wall_s": max(tel) if tel else None, "cpu_s": comp.get("cpu_time_total_s"),
        "energy_au": e_au, "energy_ev": e_au * HARTREE_EV if e_au is not None else None,
        "energy_ev_per_atom": run_info.energy_per_atom_ev(e_au, n_atoms),
    })
    return out


def _functional(text: str, p: Path) -> Optional[str]:
    from mace.properties.defs.run_info import functional
    return functional(text, p)


def _basis(p: Path) -> Optional[str]:
    from mace.properties.defs.run_info import basis
    return basis(p)


def card(meta: Dict[str, Any], conv: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The meta fields plus SCF cycles per point (min/median/max) from the convergence parse."""
    out = dict(meta)
    if conv and conv.get("points"):
        cyc = [p["scf_cycles"] for p in conv["points"] if p.get("scf_cycles") is not None]
        if cyc:
            out["scf_cycles"] = {"min": min(cyc), "median": statistics.median(cyc), "max": max(cyc)}
    return out


# ------------------------------------------------------------------ tracking a measurement
def _frac_of(s):
    import numpy as np
    cell = np.asarray(s["cell"], float)
    if s.get("frac") is not None:
        return np.asarray(s["frac"], float), cell
    return np.linalg.solve(cell.T, np.asarray(s["cart"], float).T).T, cell


def track(frames: List[Dict[str, Any]], measurement: Dict[str, Any]) -> List[Optional[float]]:
    """The measurement's value per frame. ``measurement`` = {type: distance |
    angle | dihedral, atoms: [{base, shift: [i, j, k]}]}. Frame 0 uses the
    given images; each later frame carries every atom's image forward by
    nearest-image continuation (shift_k = shift_{k-1} - round(f_k - f_{k-1}),
    per periodic axis), and the value is computed in Cartesian coordinates
    with that frame's cell, so a wrap from 0.99 to 0.01 never jumps."""
    import numpy as np
    kind = measurement.get("type")
    atoms = measurement.get("atoms") or []
    need = {"distance": 2, "angle": 3, "dihedral": 4}.get(kind)
    if need is None or len(atoms) != need:
        raise ValueError("a distance (2 atoms), angle (3) or dihedral (4)")
    try:
        bases = [int(a["base"]) for a in atoms]
        shifts = [np.asarray(a.get("shift") or [0, 0, 0], float).reshape(3) for a in atoms]
    except (KeyError, TypeError, ValueError, AttributeError):
        raise ValueError("each atom is {base: atom index, shift: [i, j, k]}")
    if any(b < 0 for b in bases):
        # a negative index would silently pick an atom counted from the end
        raise ValueError("an atom of the measurement is not in every frame")
    out: List[Optional[float]] = []
    prev = None
    for s in frames:
        f, cell = _frac_of(s)
        pbc = np.asarray(s.get("pbc") or [True, True, True], bool)
        if any(b >= len(f) for b in bases):
            raise ValueError("an atom of the measurement is not in every frame")
        if prev is not None:
            for k, b in enumerate(bases):
                d = f[b] - prev[b]
                shifts[k] = shifts[k] - np.where(pbc, np.round(d), 0.0)
        prev = f
        pts = []
        for k, b in enumerate(bases):
            if s.get("frac") is None:
                cart = np.asarray(s["cart"], float)[b] + np.where(pbc, shifts[k], 0.0) @ cell
            else:
                cart = (f[b] + shifts[k]) @ cell
            pts.append(cart)
        out.append(_value(kind, pts))
    return out


def _value(kind, pts) -> float:
    import numpy as np
    if kind == "distance":
        return float(np.linalg.norm(pts[1] - pts[0]))
    if kind == "angle":
        a, b = pts[0] - pts[1], pts[2] - pts[1]
        c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        return math.degrees(math.acos(max(-1.0, min(1.0, c))))
    b0, b1, b2 = pts[0] - pts[1], pts[2] - pts[1], pts[3] - pts[2]
    b1n = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    x = np.dot(v, w)
    y = np.dot(np.cross(b1n, v), w)
    return math.degrees(math.atan2(y, x))
