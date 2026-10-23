"""The TAU workflow step: BOLTZTRA's relaxation time from deformation potentials.

CRYSTAL's BOLTZTRA takes one relaxation time, RELAXTIM, in fs (default 10;
manual 14.5, p. 314). A TAU step computes it before a TRANSPORT step, the way
``mace tau strain`` / ``mace tau fit`` do by hand (mace/transport):

* when the SP finishes, the strained single points of its deck are written
  (``strain.write_strain_set``: 4 strains per axis and one unstrained run
  shared by the axes) in ``step_NNN_TAU/<material>/`` and submitted, each as
  an ordinary calculation row of type TAU. The rows of one structure are the
  step's job group: ``tau_group.json`` beside the decks lists its members, and
  each row carries its member name (``tau_member``), which error recovery
  passes on to the run it resubmits;
* the axes come from the deck's symmetry: x for a cubic crystal, x and z for
  a tetragonal, trigonal or hexagonal one, x, y and z below; a slab's in-plane
  axes (x for a square or hexagonal layer), a polymer's x;
* when every member has finished (completed, or failed with no recovery run
  after it), in any order, and the structure's BAND and DOSS runs (the masses)
  are no longer running, the engine fits E1 and C (``strain.fit``), takes the
  masses from those runs (``transport.cli.resolve_masses``) or the step's m*,
  and computes tau (``dpt.estimate``) for both carriers along every axis. It
  writes ``tau_<carrier>_<axis>.json`` (the provenance record ``mace analyze
  --tau-fs`` reads) and ``tau_summary.json``;
* the TRANSPORT step after it waits for that summary and runs with RELAXTIM =
  tau of the chosen carrier along the first axis. It is never blocked: a metal,
  failed strained runs, no m*, or a tau outside 1-1e5 fs give CRYSTAL's 10 fs
  (``fallback_fs``) with the reason recorded; a poor E1 fit (R^2 < 0.9) is used
  and flagged. Only ``strict`` holds TRANSPORT back then.
"""
import json
import math
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

STEP = "TAU"
DEFAULTS: Dict[str, Any] = {
    "axes": "auto",                                  # or a list of x / y / z
    "strains": [-0.01, -0.005, 0.005, 0.01],
    "carrier": "auto",                               # electron | hole | auto
    "m_star": "auto",                                # m_e, or auto (the workflow's BAND/DOSS)
    "strict": False,                                 # hold TRANSPORT when tau fell back or is poor
    "fallback_fs": 10.0,                             # CRYSTAL's default RELAXTIM
    "temperature": 300.0,                            # K
}
OPTION_KEYS = tuple(DEFAULTS)
TAU_RANGE_FS = (1.0, 1e5)
R2_LOW = 0.9
WALLTIME_FACTOR = 1.5
MANIFEST = "tau_group.json"
SUMMARY = "tau_summary.json"
# a member that will not change any more (its successor, after a recovery, is the member then)
FINISHED = frozenset({"completed", "failed", "error", "timeout", "cancelled", "canceled"})  # british-ok: status value
ACTIVE = frozenset({"pending", "held", "submitted", "running", "queued"})
RELAXTIM_OPTIONS = ("default", "auto")


class TauOptionError(ValueError):
    """A TAU step option or a TRANSPORT relaxation_time MACE cannot use."""


def base_of(calc_type: str) -> str:
    return str(calc_type).rstrip("0123456789") or str(calc_type)


def is_tau(calc_type: Optional[str]) -> bool:
    return base_of(calc_type or "") == STEP


# ------------------------------------------------------------------ options
def check_options(opts: Any, where: str = "TAU") -> Dict[str, Any]:
    """The step's options with the defaults filled in; raises TauOptionError."""
    if opts is None:
        opts = {}
    if not isinstance(opts, dict):
        raise TauOptionError(f"{where}: the options are an object")
    bad = sorted(set(opts) - set(OPTION_KEYS))
    if bad:
        raise TauOptionError(f"{where}: unknown option(s) {', '.join(bad)} (options: {', '.join(OPTION_KEYS)})")
    out = json.loads(json.dumps(DEFAULTS))
    if "axes" in opts:
        a = opts["axes"]
        if a != "auto":
            if isinstance(a, str):
                a = [x for x in re.split(r"[\s,]+", a.strip().lower()) if x]
            if not isinstance(a, list) or not a or any(x not in ("x", "y", "z") for x in a) or len(set(a)) != len(a):
                raise TauOptionError(f"{where}.axes: auto, or a list of x, y and z (each once)")
        out["axes"] = a
    if "strains" in opts:
        s = opts["strains"]
        if isinstance(s, str):
            s = [x for x in re.split(r"[\s,]+", s.strip()) if x]
        try:
            s = sorted({float(x) for x in s})
        except (TypeError, ValueError):
            raise TauOptionError(f"{where}.strains: a list of numbers (fractions: 0.01 is 1%)")
        if any(abs(x) < 1e-12 or abs(x) > 0.05 for x in s) or len(s) < 2 or len(s) > 8:
            raise TauOptionError(f"{where}.strains: 2 to 8 non-zero strains within ±0.05 (the unstrained run "
                                 f"is added)")
        out["strains"] = s
    if "carrier" in opts:
        if opts["carrier"] not in ("auto", "electron", "hole"):
            raise TauOptionError(f"{where}.carrier: auto, electron or hole")
        out["carrier"] = opts["carrier"]
    if "m_star" in opts:
        m = opts["m_star"]
        if m in (None, "", "auto"):
            out["m_star"] = "auto"
        else:
            try:
                m = float(m)
            except (TypeError, ValueError):
                raise TauOptionError(f"{where}.m_star: auto or an effective mass in m_e")
            if not (math.isfinite(m) and m > 0):
                raise TauOptionError(f"{where}.m_star: a positive mass in m_e")
            out["m_star"] = m
    if "strict" in opts:
        if not isinstance(opts["strict"], bool):
            raise TauOptionError(f"{where}.strict: true or false")
        out["strict"] = opts["strict"]
    for k, lo, hi in (("fallback_fs", TAU_RANGE_FS[0], TAU_RANGE_FS[1]), ("temperature", 1.0, 5000.0)):
        if k in opts:
            try:
                v = float(opts[k])
            except (TypeError, ValueError):
                raise TauOptionError(f"{where}.{k}: a number")
            if not (lo <= v <= hi):
                raise TauOptionError(f"{where}.{k}: between {lo:g} and {hi:g}")
            out[k] = v
    return out


def check_relaxation_time(v: Any, where: str = "TRANSPORT") -> Any:
    """default | auto | a number of fs (1-1e5)."""
    if isinstance(v, str) and v.strip().lower() in RELAXTIM_OPTIONS:
        return v.strip().lower()
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise TauOptionError(f"{where}.relaxation_time: default, auto or a time in fs")
    if not (TAU_RANGE_FS[0] <= f <= TAU_RANGE_FS[1]):
        raise TauOptionError(f"{where}.relaxation_time: between {TAU_RANGE_FS[0]:g} and {TAU_RANGE_FS[1]:g} fs")
    return f


def tau_before(sequence: Sequence[str], transport_type: str) -> Optional[str]:
    """The TAU step a TRANSPORT step takes its relaxation time from: the
    nearest TAU before it, after the SP it reads (the SP the TAU strains)."""
    seq = list(sequence or [])
    if transport_type not in seq:
        return None
    i = seq.index(transport_type)
    for j in range(i - 1, -1, -1):
        b = base_of(seq[j])
        if b == STEP:
            return seq[j]
        if b in ("SP", "OPT"):
            return None
    return None


def insert_tau(sequence: Sequence[str], transport_index: int) -> List[str]:
    """``sequence`` with a TAU step right before the TRANSPORT at ``transport_index``."""
    seq = list(sequence)
    n = sum(1 for s in seq if base_of(s) == STEP)
    seq.insert(transport_index, STEP if n == 0 else f"{STEP}{n + 1}")
    return seq


# ------------------------------------------------------------------ resources
def _walltime_seconds(wt: str) -> Optional[int]:
    """SLURM's time formats: M, M:S, H:M:S, D-H, D-H:M, D-H:M:S."""
    wt = str(wt or "").strip()
    m = re.match(r"^(?:(\d+)-)?(\d+)(?::(\d+))?(?::(\d+))?$", wt)
    if not m:
        return None
    d, a, b, c = m.groups()
    if d is not None:
        h, mi, s = int(a), int(b or 0), int(c or 0)
        return int(d) * 86400 + h * 3600 + mi * 60 + s
    if c is not None:
        return int(a) * 3600 + int(b) * 60 + int(c)
    if b is not None:
        return int(a) * 60 + int(b)
    return int(a) * 60


def scale_walltime(wt: str, factor: float = WALLTIME_FACTOR) -> str:
    """``wt`` times ``factor`` (a strained cell has less symmetry, so more k
    points than its parent), as D-HH:MM:SS; unchanged when not a SLURM time."""
    s = _walltime_seconds(wt)
    if s is None:
        return wt
    s = int(math.ceil(s * factor / 60.0)) * 60
    d, rest = divmod(s, 86400)
    h, rest = divmod(rest, 3600)
    return f"{d}-{h:02d}:{rest // 60:02d}:00" if d else f"{h}:{rest // 60:02d}:00"


# ------------------------------------------------------------------ symmetry
def symmetry_axes(deck_text: str) -> Tuple[int, List[str], str]:
    """(periodicity, the axes to strain, why) of an SP deck: one axis per set of
    axes the symmetry makes equivalent."""
    from mace.transport import strain
    lines = deck_text.splitlines()
    dim, _ = strain.atoms_end(lines)
    key = lines[1].strip().upper()
    group = None
    if key == "CRYSTAL":
        head = lines[2].split()
        ifhr = int(head[1]) if len(head) >= 2 and head[1].lstrip("-").isdigit() else 0
        ifso = int(head[2]) if len(head) >= 3 and head[2].lstrip("-").isdigit() else 0
        params_line = 4 + (1 if ifso > 1 else 0)
        if ifhr == 0:
            try:
                group = int(lines[3].split()[0])
            except (IndexError, ValueError):
                group = None
    else:
        params_line = 3
    try:
        nparams = len(lines[params_line].split())
    except IndexError:
        nparams = 6
    if dim == 1:
        return 1, ["x"], "a polymer: along the chain (x)"
    if dim == 2:
        if nparams == 1:
            return 2, ["x"], "a square or hexagonal layer: x and y are equivalent"
        return 2, ["x", "y"], "a layer: the in-plane axes x and y"
    if group is not None:
        if 195 <= group <= 230:
            return 3, ["x"], f"cubic (space group {group}): x, y and z are equivalent"
        if 75 <= group <= 194:
            return 3, ["x", "z"], f"tetragonal, trigonal or hexagonal (space group {group}): x and y are equivalent"
        return 3, ["x", "y", "z"], f"space group {group}: x, y and z differ"
    if nparams == 1:
        return 3, ["x"], "cubic (one lattice parameter): x, y and z are equivalent"
    if nparams == 2:
        return 3, ["x", "z"], "two lattice parameters (tetragonal, trigonal or hexagonal): x and y are equivalent"
    return 3, ["x", "y", "z"], "x, y and z differ"


def member_count(axes: Sequence[str], strains: Sequence[float]) -> int:
    return len(axes) * len(strains) + 1


# ------------------------------------------------------------------ the group
def _settings(row: Dict[str, Any]) -> Dict[str, Any]:
    try:
        s = json.loads(row.get("settings_json") or "{}")
    except (json.JSONDecodeError, TypeError):
        s = {}
    return s if isinstance(s, dict) else {}


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, path)


def step_dir(engine, material_id: str, workflow_id: str, tau_type: str) -> Path:
    """workflow_outputs/<id>/step_NNN_TAU/<material>/ (the plan's step number)."""
    base = Path(engine.base_work_dir) / "workflow_outputs" / workflow_id
    num = engine._get_next_step_number(base, tau_type)
    return base / f"step_{num:03d}_{tau_type}" / engine.extract_core_material_name(material_id)


def group_rows(db, material_id: str, tau_type: str) -> List[Dict[str, Any]]:
    rows = [r for r in db.get_calculations_by_status(material_id=material_id) if r.get("calc_type") == tau_type]
    return sorted(rows, key=lambda r: (r.get("created_at") or "", r.get("calc_id") or ""))


def _member_of(row: Dict[str, Any], members: Dict[str, str]) -> Optional[str]:
    """The member a row runs: its tau_member, else the deck it ran (a recovery
    deck is <member deck>_recovery_<time>.d12)."""
    m = _settings(row).get("tau_member")
    if m in members:
        return m
    stem = Path(row.get("input_file") or "").stem
    for name, deck_stem in members.items():
        if stem == deck_stem or stem.startswith(deck_stem + "_recovery"):
            return name
    return None


def group_state(db, material_id: str, tau_type: str, manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Each member's latest row, and whether the group has finished."""
    members = {m["member"]: Path(m["deck"]).stem for m in manifest.get("members") or []}
    latest: Dict[str, Dict[str, Any]] = {}
    for r in group_rows(db, material_id, tau_type):
        m = _member_of(r, members)
        if m is not None:
            latest[m] = r                        # the newest run of the member
    done, failed, active, missing = [], [], [], []
    for m in members:
        r = latest.get(m)
        st = str((r or {}).get("status") or "").lower()
        if r is None:
            missing.append(m)
        elif st == "completed":
            done.append(m)
        elif st in FINISHED or st == "resubmitted":
            failed.append(m)
        else:
            active.append(m)
    return {"total": len(members), "completed": done, "failed": failed, "active": active, "missing": missing,
            "finished": not active and not missing, "latest": latest}


def _manifest_for(engine, material_id: str, tau_type: str, workflow_id: Optional[str]) -> Tuple[Optional[Path],
                                                                                              Optional[Dict]]:
    rows = group_rows(engine.db, material_id, tau_type)
    dirs = [Path(r["work_dir"]) for r in rows if r.get("work_dir")]
    if not dirs and workflow_id:
        dirs = [step_dir(engine, material_id, workflow_id, tau_type)]
    for d in dirs:
        m = read_json(d / MANIFEST)
        if m is not None or (d / SUMMARY).is_file():
            return d, m
    return (dirs[0] if dirs else None), None


def summary_of(engine, material_id: str, tau_type: str, workflow_id: Optional[str]) -> Optional[Dict[str, Any]]:
    d, _ = _manifest_for(engine, material_id, tau_type, workflow_id)
    return read_json(d / SUMMARY) if d else None


def started(engine, material_id: str, tau_type: str, workflow_id: Optional[str]) -> bool:
    """The group was written (or could not be, which its summary says)."""
    if group_rows(engine.db, material_id, tau_type):
        return True
    return summary_of(engine, material_id, tau_type, workflow_id) is not None


# ------------------------------------------------------------------ generate
def generate(engine, sp_calc_id: str, tau_type: str, workflow_id: Optional[str]) -> List[str]:
    """Write and submit the strained single points of the SP ``sp_calc_id``;
    the new calculation ids. A deck that cannot be strained (a molecule, a
    two-component or optimization deck) writes the summary with the fallback
    instead, so TRANSPORT is not held."""
    sp = engine.db.get_calculation(sp_calc_id)
    material_id = sp["material_id"]
    wid = workflow_id or _settings(sp).get("workflow_id") or os.environ.get("MACE_WORKFLOW_ID")
    opts = options(engine, wid, tau_type)
    if wid is None:
        wid = engine.get_workflow_output_base(sp).name
    d = step_dir(engine, material_id, wid, tau_type)
    d.mkdir(parents=True, exist_ok=True)
    parent = Path(sp.get("input_file") or "")
    from mace.transport import strain
    try:
        text = parent.read_text(errors="replace")
        dim, auto_axes, why = symmetry_axes(text)
        axes = auto_axes if opts["axes"] == "auto" else list(opts["axes"])
        allowed = {3: ("x", "y", "z"), 2: ("x", "y"), 1: ("x",)}[dim]
        bad = [a for a in axes if a not in allowed]
        if bad:
            raise strain.StrainError(f"axis {', '.join(bad)} is not periodic in this {dim}D system")
        if opts["axes"] != "auto":
            why = "the step's axes"
        decks: List[Path] = []
        for k, axis in enumerate(axes):
            decks += strain.write_strain_set(parent, axis, opts["strains"], d, force=True, unstrained=(k == 0))
    except (OSError, ValueError) as e:
        print(f"  {tau_type}: no strained decks for {material_id}: {e}")
        _write_summary(d, _fallback_summary(material_id, wid, tau_type, opts,
                                            f"the strained decks could not be written: {e}", {}))
        return []
    members = []
    for deck in decks:
        # a restart's guess (GUESSP) needs the parent's density matrix: the strained cell starts afresh
        lines = deck.read_text().splitlines()
        if any(ln.strip().upper() == "GUESSP" for ln in lines):
            deck.write_text("\n".join(ln for ln in lines if ln.strip().upper() != "GUESSP") + "\n")
        axis, eps = strain.strain_of(deck.read_text())
        tag = re.sub(r"^.*_dp_", "", deck.stem)          # x_m1pct, x_p0pct (the shared unstrained run)
        members.append({"member": tag, "deck": deck.name, "axis": axis, "strain": eps})
    manifest = {"version": 1, "workflow_id": wid, "material_id": material_id, "step": tau_type,
                "sp_calc_id": sp_calc_id, "parent_deck": str(parent), "dimensionality": dim, "axes": axes,
                "axes_reason": why, "options": opts, "members": members,
                "created": datetime.now().isoformat(timespec="seconds")}
    _write_json(d / MANIFEST, manifest)
    step_num = int(re.match(r"step_(\d+)_", d.parent.name).group(1))
    print(f"  {tau_type}: {len(members)} strained single points of {material_id} ({', '.join(axes)}: {why})")
    new_ids = []
    for m in members:
        deck = d / m["deck"]
        script = engine._create_slurm_script_for_calculation(d, deck.stem, tau_type, step_num, wid)
        settings = {"workflow_id": wid, "workflow_step": step_num, "step_number": step_num,
                    "tau_group": tau_type, "tau_member": m["member"], "tau_axis": m["axis"],
                    "tau_strain": m["strain"], "generated_from_sp": sp_calc_id}
        calc_id = engine.db.create_calculation(material_id=material_id, calc_type=tau_type, input_file=str(deck),
                                               work_dir=str(d), prerequisite_calc_id=sp_calc_id, settings=settings)
        job = engine._submit_calculation_to_slurm(script, d, calc_id)
        if job:
            engine.db.update_calculation_status(calc_id, "submitted", slurm_job_id=job)
        elif not engine._was_held():
            from mace.queue import budget
            o = budget.last_outcome()
            engine.db.update_calculation_status(
                calc_id, "failed", error_message=(f"sbatch refused it: {o.message}" if o and o.message
                                                  else "SLURM submission failed"))
        new_ids.append(calc_id)
    return new_ids


def options(engine, workflow_id: Optional[str], tau_type: str) -> Dict[str, Any]:
    cfg = engine._get_plan_step_config(workflow_id, tau_type) if workflow_id else {}
    try:
        return check_options((cfg or {}).get("tau") or {})
    except TauOptionError as e:
        print(f"  Warning: {e}; the step's defaults are used")
        return check_options({})


# ------------------------------------------------------------------ fit
def _fallback_summary(material_id, workflow_id, tau_type, opts, reason, extra) -> Dict[str, Any]:
    return {"version": 1, "material_id": material_id, "workflow_id": workflow_id, "step": tau_type,
            "computed_fs": None, "used_fs": float(opts.get("fallback_fs", 10.0)), "fallback": True,
            "reason": reason, "low_confidence": False, "strict": bool(opts.get("strict")),
            "held": bool(opts.get("strict")), "fitted_at": datetime.now().isoformat(timespec="seconds"),
            **extra}


def _write_summary(d: Path, summary: Dict[str, Any]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    _write_json(d / SUMMARY, summary)
    word = (f"fell back to {summary['used_fs']:g} fs: {summary['reason']}" if summary.get("fallback")
            else f"RELAXTIM {summary['used_fs']:.4g} fs")
    print(f"  {summary.get('step', STEP)} of {summary.get('material_id')}: {word}"
          + (" (held: strict)" if summary.get("held") else ""))


def _property_file(row: Optional[Dict[str, Any]], kind: str) -> Optional[Path]:
    """The BAND.DAT / DOSS.DAT (else the f25) a finished properties run copied back."""
    if not row or str(row.get("status") or "").lower() != "completed" or not row.get("input_file"):
        return None
    deck = Path(row["input_file"])
    for suf in (f".{kind}.DAT", ".f25"):
        p = deck.with_name(deck.stem + suf)
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def _mass_rows(engine, material_id: str, sequence: Sequence[str]) -> Tuple[Dict[str, Optional[Dict]], List[str]]:
    """The workflow's BAND and DOSS rows (the first of each in the plan) and the ones still running."""
    out: Dict[str, Optional[Dict]] = {}
    running: List[str] = []
    rows = engine.db.get_calculations_by_status(material_id=material_id)
    for kind in ("BAND", "DOSS"):
        step = next((s for s in sequence if base_of(s) == kind), None)
        if step is None:
            continue
        mine = sorted((r for r in rows if r.get("calc_type") == step),
                      key=lambda r: (r.get("created_at") or "", r.get("calc_id") or ""))
        out[kind] = mine[-1] if mine else None
        if mine and str(mine[-1].get("status") or "").lower() in ACTIVE:
            running.append(step)
    return out, running


def try_fit(engine, material_id: str, workflow_id: Optional[str], tau_type: str,
            sequence: Sequence[str]) -> Optional[Dict[str, Any]]:
    """The step's summary: read, or fitted now when the group has finished
    and the masses' runs are done; None while it waits."""
    d, manifest = _manifest_for(engine, material_id, tau_type, workflow_id)
    if d is None:
        return None
    done = read_json(d / SUMMARY)
    if done is not None:
        return done
    if manifest is None:
        if not group_rows(engine.db, material_id, tau_type):
            return None
        # rows without their list of members: nothing to fit, and TRANSPORT is not held for it
        summary = _fallback_summary(material_id, workflow_id, tau_type, options(engine, workflow_id, tau_type),
                                    f"{MANIFEST} is missing in {d}", {})
        _write_summary(d, summary)
        return summary
    st = group_state(engine.db, material_id, tau_type, manifest)
    if not st["finished"]:
        return None
    opts = check_options(manifest.get("options") or {})
    mass_rows, running = _mass_rows(engine, material_id, sequence)
    if opts["m_star"] == "auto" and running:
        print(f"  {tau_type}: the strained runs of {material_id} have finished; waiting for "
              f"{' and '.join(running)} (the effective masses)")
        return None
    summary = fit_group(d, manifest, st, opts, mass_rows)
    _write_summary(d, summary)
    return summary


def _carrier_choice(opts: Dict[str, Any]) -> Tuple[str, str]:
    if opts["carrier"] in ("electron", "hole"):
        return opts["carrier"], "the step's carrier"
    # MACE knows no doping (the TRANSPORT run scans the chemical potential):
    # electrons, the n-type side, unless the step names holes
    return "electron", "MACE has no doping for this structure, so electrons (the default; set carrier: hole for p-type)"


def fit_group(d: Path, manifest: Dict[str, Any], st: Dict[str, Any], opts: Dict[str, Any],
              mass_rows: Dict[str, Optional[Dict]]) -> Dict[str, Any]:
    """E1, C, the masses and tau for both carriers along every axis; the
    summary (with the RELAXTIM to use and why)."""
    from mace.transport import dpt, mass, strain
    from mace.transport.cli import resolve_masses
    material_id, wid, tau_type = manifest["material_id"], manifest.get("workflow_id"), manifest["step"]
    carrier, carrier_why = _carrier_choice(opts)
    axes = list(manifest.get("axes") or ["x"])
    axis = axes[0]
    by_member = {m["member"]: m for m in manifest.get("members") or []}
    points, unread = [], []
    for name in st["completed"]:
        row = st["latest"][name]
        out = Path(row.get("output_file") or Path(row["input_file"]).with_suffix(".out"))
        try:
            points.append(strain.read_point(out))
        except (OSError, ValueError) as e:
            unread.append(f"{name}: {e}")
    failed = sorted(st["failed"]) + [u.split(":")[0] for u in unread]
    members = {"total": st["total"], "completed": len(points), "failed": failed}
    extra = {"carrier": carrier, "carrier_reason": carrier_why, "axis": axis, "axes": axes,
             "axes_reason": manifest.get("axes_reason"), "members": members, "results": {}, "warnings": [],
             "temperature_k": opts["temperature"]}
    if unread:
        extra["warnings"] += [f"not used: {u}" for u in unread]
    if failed:
        extra["warnings"].append(f"{len(failed)} of {st['total']} strained runs failed: {', '.join(failed)}")
    zero = [p for p in points if not p.get("axis") and abs(p.get("strain") or 0.0) < 1e-12]
    band = _property_file(mass_rows.get("BAND"), "BAND")
    dos = _property_file(mass_rows.get("DOSS"), "DOSS")
    reasons: Dict[Tuple[str, str], str] = {}
    for ax in axes:
        series = [p for p in points if p.get("axis") == ax] + zero[:1]
        for car in ("electron", "hole"):
            try:
                if not zero or len({round(p["strain"], 8) for p in series}) < 3:
                    raise strain.StrainError(f"too few strained runs finished along {ax} (the fit needs the "
                                             f"unstrained run and two strained ones)")
                ft = strain.fit(series, car)
                m, md, msrc, am = resolve_masses(opts["m_star"], car, ax, ft["dimensionality"], band=band,
                                                 dos=dos)
                srcs = {"E1": f"linear fit of the {'CBM' if car == 'electron' else 'VBM'} over "
                              f"{len(ft['strains'])} strained SP runs ({ax})",
                        "C": "quadratic fit of the total energy over the same runs (clamped ions)", **msrc}
                if opts["m_star"] != "auto":
                    srcs["m_star"] = "the TAU step's m_star"
                res = dpt.estimate(ft["dimensionality"], m, ft["E1_eV"], ft["C"], opts["temperature"], None,
                                   srcs, carrier=car, axis=ax, m_dos=md)
                if am:
                    res["masses"] = am
                res["fit"] = ft
                name = f"tau_{car}_{ax}.json"
                _write_json(d / name, res)
                extra["results"].setdefault(car, {})[ax] = {
                    "tau_fs": res["tau_fs"], "mobility_cm2_Vs": res["mobility_cm2_Vs"], "E1_eV": abs(ft["E1_eV"]),
                    "E1_r2": ft["E1_r2"], "C": ft["C"], "C_unit": ft["C_unit"], "C_r2": ft["C_r2"],
                    "m_c": m, "m_d": md if md is not None else m,
                    "m_c_source": srcs.get("m_star"), "m_d_source": srcs.get("m_dos", srcs.get("m_star")),
                    "warnings": list(ft.get("warnings") or []) + list((am or {}).get("warnings") or []),
                    "file": name}
            except strain.StrainError as e:
                msg = str(e)
                reasons[(car, ax)] = ("a metal or no gap: deformation-potential tau needs a band edge"
                                      if "no band edges" in msg else msg)
            except (mass.MassError, dpt.TauError) as e:
                reasons[(car, ax)] = f"no effective mass: {e}" + (
                    "" if band or dos else " (add a BAND or DOSS step, or set the step's m_star)")
    chosen = (extra["results"].get(carrier) or {}).get(axis)
    if chosen is None:
        why = reasons.get((carrier, axis), "no result")
        return _fallback_summary(material_id, wid, tau_type, opts, why, extra)
    tau = chosen["tau_fs"]
    extra["computed_fs"] = tau
    extra["json"] = chosen["file"]
    if not (TAU_RANGE_FS[0] <= tau <= TAU_RANGE_FS[1]):
        return _fallback_summary(material_id, wid, tau_type, opts, f"tau {tau:.4g} fs is outside "
                                 f"{TAU_RANGE_FS[0]:g}-{TAU_RANGE_FS[1]:g} fs", extra)
    low = chosen["E1_r2"] < R2_LOW
    if low:
        extra["warnings"].append(f"low confidence: the band edge shifts non-linearly with strain (E1 R^2 "
                                 f"{chosen['E1_r2']:.2f} < {R2_LOW})")
    used = float(f"{tau:.4g}")
    return {"version": 1, "material_id": material_id, "workflow_id": wid, "step": tau_type,
            "computed_fs": tau, "used_fs": used, "fallback": False, "reason": None, "low_confidence": low,
            "strict": opts["strict"], "held": bool(opts["strict"] and low),
            "fitted_at": datetime.now().isoformat(timespec="seconds"), **extra}


# ------------------------------------------------------------------ TRANSPORT
def transport_gate(engine, material_id: str, workflow_id: Optional[str], sequence: Sequence[str],
                   transport_type: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """("none" | "go" | "wait" | "held", summary): whether a TRANSPORT step may
    be generated now. "none": it takes no tau from a TAU step."""
    tau_type = tau_before(sequence, transport_type)
    if tau_type is None:
        return "none", None
    summary = try_fit(engine, material_id, workflow_id, tau_type, sequence)
    if summary is None:
        return "wait", None
    if summary.get("held"):
        return "held", summary
    return "go", summary


def apply_to_config(json_config: Dict[str, Any], summary: Dict[str, Any]) -> Dict[str, Any]:
    """The opt2d3 configuration (envelope or bare) with RELAXTIM from the summary."""
    cfg = json.loads(json.dumps(json_config))
    target = cfg["configuration"] if isinstance(cfg.get("configuration"), dict) else cfg
    target["relaxation_time"] = summary["used_fs"]
    return cfg


def provenance(summary: Dict[str, Any], tau_dir: Optional[Path] = None) -> Dict[str, Any]:
    """What a TRANSPORT row records about the RELAXTIM it ran with."""
    p = {"relaxation_time_fs": summary["used_fs"], "source": summary.get("step", STEP),
         "fallback": bool(summary.get("fallback")), "reason": summary.get("reason"),
         "low_confidence": bool(summary.get("low_confidence")), "carrier": summary.get("carrier"),
         "axis": summary.get("axis")}
    if tau_dir is not None:
        p["summary"] = str(Path(tau_dir) / SUMMARY)
        if summary.get("json"):
            p["json"] = str(Path(tau_dir) / summary["json"])
    return p


def line(summary: Dict[str, Any]) -> str:
    """One line: what TRANSPORT runs with, and why."""
    if summary.get("fallback"):
        return f"RELAXTIM {summary['used_fs']:g} fs (fallback: {summary.get('reason')})"
    s = (f"RELAXTIM {summary['used_fs']:.4g} fs (deformation potential, {summary.get('carrier')}s, "
         f"{summary.get('axis')})")
    return s + (", low confidence" if summary.get("low_confidence") else "")


# ------------------------------------------------------------------ sweep
def pending_groups(db) -> List[Tuple[str, str, Optional[str]]]:
    """(material, TAU step, workflow id) of every group in this database without a summary yet."""
    seen = set()
    out = []
    for r in db.get_calculations_by_status():
        if not is_tau(r.get("calc_type")):
            continue
        key = (r["material_id"], r["calc_type"])
        if key in seen:
            continue
        seen.add(key)
        if r.get("work_dir") and (Path(r["work_dir"]) / SUMMARY).is_file():
            continue
        out.append((r["material_id"], r["calc_type"], _settings(r).get("workflow_id")))
    return out
