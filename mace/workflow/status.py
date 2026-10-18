"""What a workflow is doing, derived on every call and never stored.

``mace workflow --status ID [--json]`` and ``mace workflow --list [--json]``
(and the GUI's Workflows view, which imports this module) read:

1. the plan, ``workflow_configs/workflow_plan_<id>.json`` (strict name: the
   ``*_updated.json`` copies are not plans);
2. the workflow's database, read-only: ``.mace_context_<id>/materials.db``,
   else its archive (``archived_workflows/<id>_<time>/``) or its exported
   results (``workflow_results/<id>/<id>_materials.db``), else the folder's
   ``materials.db`` (shared mode), rows whose settings carry the id;
3. your SLURM jobs, through the shared cached squeue reader
   (``mace.cluster.reader.squeue_own``);
4. the control file (``mace workflow --pause/--resume/--cancel``).

Each material x step gets one node state:

planned     not started, and its parent has not finished
waiting     its parent completed; the parent's job callback submits it
stale       waiting for longer than ``stale_minutes``: the callback did not
            run or failed (``mace engine --action process`` continues it)
held        its parent completed, and the plan has auto_progression off
paused / canceled
            its parent completed, and the workflow is paused / canceled
queued / running
            submitted; squeue (when available) says pending / running
ending      the database says queued/running, squeue no longer lists it: the
            job ended and its callback has not recorded it (yet)
completed / failed / canceled
recovering  the latest attempt was resubmitted by recovery
blocked     its parent failed or was canceled, or the parent was processed
            and this step was not created

Works for workflows started from the CLI, the GUI, and plans written by
MACE 1.1.4 and 1.1.5 (ids without the random suffix included).
"""
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from mace.workflow.common.constants import CALC_TYPES
from mace.workflow.control import read_control, state_of

PLAN_RE = re.compile(r"^workflow_plan_(\d{8}_\d{6}(?:_[0-9a-f]{4})?)\.json$")
STALE_MINUTES = 30
STATUS_VERSION = 1

_DB_STATE = {
    "completed": "completed",
    "running": "running",
    "submitted": "queued", "pending": "queued", "queued": "queued",
    "failed": "failed", "error": "failed", "timeout": "failed",
    "cancelled": "canceled", "canceled": "canceled",  # british-ok: stored database value
    "resubmitted": "recovering",
    # waiting for a slot of the user-wide queue budget (mace/queue/budget.py)
    "held": "held",
}
HELD_REASON_WORDS = {"budget": "queue budget full", "workflow_cap": "the workflow's max_jobs reached",
                     "slurm_limit": "SLURM refused it (submit limit)"}
_ACTIVE = ("queued", "running", "ending", "waiting", "recovering")
_PROBLEM = ("failed", "blocked", "stale")


# ------------------------------------------------------------------ plans
def plan_files(folder: Union[str, Path]) -> Dict[str, Path]:
    """{workflow id: plan file} in ``folder/workflow_configs`` (strict names)."""
    out = {}
    configs = Path(folder) / "workflow_configs"
    try:
        entries = sorted(configs.iterdir())
    except OSError:
        return out
    for p in entries:
        m = PLAN_RE.match(p.name)
        if m and p.is_file():
            out[f"workflow_{m.group(1)}"] = p
    return out


def load_plan(folder: Union[str, Path], workflow_id: str) -> Optional[Dict[str, Any]]:
    path = plan_files(folder).get(workflow_id)
    if path is None:
        return None
    try:
        plan = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return plan if isinstance(plan, dict) else None


# ------------------------------------------------------------------ database
def database_of(folder: Union[str, Path], workflow_id: str, plan: Optional[Dict[str, Any]] = None
                ) -> Optional[Path]:
    """The database holding the workflow's calculations, or None."""
    folder = Path(folder)
    own = folder / f".mace_context_{workflow_id}" / "materials.db"
    if own.is_file():
        return own
    try:
        archived = sorted((folder / "archived_workflows").glob(f"{workflow_id}_*/materials.db"))
    except OSError:
        archived = []
    if archived:
        return archived[-1]
    # post-completion action 'export_and_delete' (engine.py): the context
    # was deleted after its database was exported here
    exported = folder / "workflow_results" / workflow_id / f"{workflow_id}_materials.db"
    if exported.is_file():
        return exported
    shared = folder / ((plan or {}).get("database") or "materials.db")
    return shared if shared.is_file() else None


def is_archived(db_path: Optional[Union[str, Path]]) -> bool:
    """True for a finished workflow's archived or exported database (read
    only; nothing is processed there)."""
    if not db_path:
        return False
    return Path(db_path).parent.parent.name in ("archived_workflows", "workflow_results")


def _settings(row: Dict[str, Any]) -> Dict[str, Any]:
    try:
        s = json.loads(row.get("settings_json") or "{}")
    except (TypeError, ValueError):
        return {}
    return s if isinstance(s, dict) else {}


def read_rows(db_path: Optional[Path], workflow_id: str) -> List[Dict[str, Any]]:
    """The workflow's calculation rows, read-only (never creates or locks the
    file for writing). In a workflow's own database rows without a
    workflow_id belong to it; in a shared one only rows stamped with it."""
    if db_path is None or not Path(db_path).is_file():
        return []
    own_db = Path(db_path).parent.name.startswith(".mace_context_") or is_archived(db_path)
    try:
        conn = sqlite3.connect(f"file:{Path(db_path).resolve()}?mode=ro", uri=True, timeout=5)
    except sqlite3.Error:
        return []
    try:
        conn.row_factory = sqlite3.Row
        rows = [dict(r) for r in conn.execute("SELECT * FROM calculations")]
    except sqlite3.Error:
        return []
    finally:
        conn.close()
    out = []
    for r in rows:
        wid = _settings(r).get("workflow_id")
        if wid == workflow_id or (own_db and not wid):
            out.append(r)
    out.sort(key=lambda r: (r.get("created_at") or "", r.get("submitted_at") or ""))
    return out


# ------------------------------------------------------------------ squeue
def squeue_rows() -> Optional[List[Dict[str, Any]]]:
    """Your jobs from the shared, cached squeue reader; None when SLURM is
    not reachable here (the database's states are shown as they are)."""
    try:
        from mace.cluster import reader
        if not reader.slurm_present():
            return None
        res = reader.squeue_own(max_age=60)
    except Exception:
        return None
    if res is None or res.error or not isinstance(res.value, list):
        return None
    return res.value


def _squeue_row(rows: Optional[List[Dict[str, Any]]], job_id: Optional[str]) -> Optional[Dict[str, Any]]:
    if rows is None or not job_id:
        return None
    for r in rows:
        if str(r.get("id", "")).strip() == str(job_id).strip():
            return r
    return {}          # SLURM reachable and the job is not listed


def _squeue_state(rows: Optional[List[Dict[str, Any]]], job_id: Optional[str]) -> Optional[str]:
    r = _squeue_row(rows, job_id)
    if r is None:
        return None
    if not r:
        return ""
    return str(r.get("state") or "").upper() or "PENDING"


# ------------------------------------------------------------------ rules
def base_type(calc_type: str) -> str:
    return calc_type.rstrip("0123456789") or calc_type


def parent_index(sequence: List[str], i: int) -> Optional[int]:
    """The step a step is built from: the nearest earlier step of a type it
    depends on (CALC_TYPES), else the step before it."""
    if i <= 0:
        return None
    deps = CALC_TYPES.get(base_type(sequence[i]), {}).get("depends_on") or []
    for j in range(i - 1, -1, -1):
        if base_type(sequence[j]) in deps:
            return j
    return i - 1


# a job MACE submitted this recently and squeue does not list yet is queued (the
# own-jobs answer is cached for a poll interval), not finishing
JUST_SUBMITTED_MINUTES = 2.0


def _age_minutes(stamp: Optional[str], now: datetime) -> Optional[float]:
    if not stamp:
        return None
    try:
        t = datetime.fromisoformat(str(stamp).replace("Z", ""))
    except ValueError:
        return None
    if t.tzinfo is not None:
        t = t.replace(tzinfo=None)
    return (now - t).total_seconds() / 60.0


def _material_ids(plan: Dict[str, Any]) -> List[str]:
    files = []
    for kind in ("cif", "d12"):
        files += list(((plan.get("input_files") or {}).get(kind)) or [])
    try:
        from mace.database.materials import create_material_id_from_file
    except Exception:            # never fail a status for the id helper
        def create_material_id_from_file(f):
            return Path(f).stem
    ids = []
    for f in files:
        mid = create_material_id_from_file(str(f))
        if mid not in ids:
            ids.append(mid)
    return ids


def _node(row_list: List[Dict[str, Any]], sq: Optional[List[Dict[str, Any]]],
          now: Optional[datetime] = None) -> Dict[str, Any]:
    latest = row_list[-1]
    state = _DB_STATE.get(str(latest.get("status") or "").lower(), "queued")
    job = latest.get("slurm_job_id")
    in_queue = _squeue_state(sq, job)
    sq_row = _squeue_row(sq, job) or {}
    if state in ("queued", "running") and in_queue is not None:
        age = _age_minutes(latest.get("submitted_at") or latest.get("created_at"), now or datetime.now())
        if in_queue == "" and state == "queued" and age is not None and age < JUST_SUBMITTED_MINUTES:
            pass                      # a cached squeue answer may predate the submission
        elif in_queue == "":
            state = "ending"
        elif in_queue in ("RUNNING", "COMPLETING", "CONFIGURING"):
            state = "running"
        else:
            state = "queued"
    extra: Dict[str, Any] = {}
    if str(latest.get("status") or "").lower() == "held":
        st = _settings(latest)
        why = st.get("held_reason") or "budget"
        extra = {"held_reason": why, "held_since": st.get("held_since"),
                 "reason": HELD_REASON_WORDS.get(why, why)}
    return {
        **extra,
        "state": state,
        "calc_id": latest.get("calc_id"),
        "job_id": job,
        "squeue_state": in_queue or None,
        "attempts": len(row_list),
        "attempt_ids": [r.get("calc_id") for r in row_list],
        "db_status": latest.get("status"),
        "error_type": latest.get("error_type"),
        "input_file": latest.get("input_file"),
        "output_file": latest.get("output_file"),
        "submitted_at": latest.get("submitted_at"),
        "completed_at": latest.get("completed_at"),
        "processed": bool(_settings(latest).get("workflow_processed")),
        "elapsed": sq_row.get("elapsed"),
        "time_limit": sq_row.get("time_limit"),
    }


def workflow_status(folder: Union[str, Path], workflow_id: str,
                    squeue: Optional[List[Dict[str, Any]]] = None, use_squeue: bool = True,
                    stale_minutes: float = STALE_MINUTES, now: Optional[datetime] = None,
                    eta: bool = False) -> Optional[Dict[str, Any]]:
    """The derived status of one workflow (a JSON-ready dict), or None when
    ``folder`` has no plan of that id. ``squeue`` injects squeue rows
    (tests, the GUI's own cache); else the shared reader is asked when
    ``use_squeue``. ``eta`` adds each running node's time left
    (``status_eta``)."""
    folder = Path(folder).resolve()
    plan = load_plan(folder, workflow_id)
    if plan is None:
        return None
    now = now or datetime.now()
    sequence = [str(s) for s in plan.get("workflow_sequence") or []]
    db = database_of(folder, workflow_id, plan)
    rows = read_rows(db, workflow_id)
    sq = squeue if squeue is not None else (squeue_rows() if use_squeue else None)
    control = read_control(folder, workflow_id)
    control_state = state_of(control)
    auto = (plan.get("execution_settings") or {}).get("auto_progression", True)
    if isinstance(auto, str):
        auto = auto.strip().lower() not in ("false", "no", "off", "0")
    auto_off = auto is False or auto == 0

    materials = _material_ids(plan)
    for r in rows:
        if r.get("material_id") and r["material_id"] not in materials:
            materials.append(r["material_id"])

    out_materials = []
    counts: Dict[str, int] = {}
    for mid in materials:
        nodes: List[Dict[str, Any]] = []
        for i, step in enumerate(sequence):
            mine = [r for r in rows if r.get("material_id") == mid and r.get("calc_type") == step]
            if mine:
                node = _node(mine, sq, now)
            else:
                node = {"state": "planned", "attempts": 0}
                p = parent_index(sequence, i)
                if p is not None:
                    parent = nodes[p]
                    if parent["state"] == "completed":
                        if control_state in ("paused", "canceled"):
                            node["state"] = control_state
                        elif auto_off:
                            node["state"] = "held"
                        elif parent.get("processed"):
                            node["state"] = "blocked"
                            node["reason"] = "the parent was processed and this step was not created"
                        else:
                            age = _age_minutes(parent.get("completed_at"), now)
                            node["state"] = "stale" if age is not None and age > stale_minutes else "waiting"
                    elif parent["state"] in ("failed", "canceled", "blocked"):
                        node["state"] = "blocked"
                        node["reason"] = f"{sequence[p]} {parent['state']}"
                elif i == 0 and control_state == "canceled":
                    node["state"] = "canceled"
            node["step"] = f"{step}_{i + 1}"
            node["calc_type"] = step
            node["parent"] = None if parent_index(sequence, i) is None else \
                f"{sequence[parent_index(sequence, i)]}_{parent_index(sequence, i) + 1}"
            nodes.append(node)
            counts[node["state"]] = counts.get(node["state"], 0) + 1
        out_materials.append({"material_id": mid, "nodes": nodes})

    states = set(counts)
    if control_state in ("paused", "canceled"):
        overall = control_state
    elif not out_materials or not sequence:
        overall = "planned"
    elif states == {"completed"}:
        overall = "completed"
    elif states & set(_ACTIVE):
        overall = "running"
    elif states & set(_PROBLEM):
        overall = "attention"
    elif states <= {"planned"}:
        overall = "planned"
    else:
        overall = "running"

    held: Dict[str, int] = {}
    for m in out_materials:
        for n in m["nodes"]:
            if n.get("held_reason"):
                held[n["held_reason"]] = held.get(n["held_reason"], 0) + 1

    result = {
        "held": held,
        "version": STATUS_VERSION,
        "workflow_id": workflow_id,
        "label": plan.get("label"),
        "folder": str(folder),
        "plan_file": str(plan_files(folder)[workflow_id]),
        "created": plan.get("created"),
        "mode": plan.get("mode"),
        "input_type": plan.get("input_type"),
        "sequence": sequence,
        "auto_progression": not auto_off,
        "control": {"state": control_state, "updated": (control or {}).get("updated")},
        "database": str(db) if db else None,
        "archived": is_archived(db),
        "squeue": {"available": sq is not None},
        "state": overall,
        "counts": counts,
        "materials": out_materials,
    }
    return status_eta(result) if eta else result


# ------------------------------------------------------------------ ETA
def output_of(node: Dict[str, Any]) -> Optional[Path]:
    """The node's CRYSTAL output: the row's output_file, else the .out next
    to its input deck."""
    for key, swap in (("output_file", False), ("input_file", True)):
        val = node.get(key)
        if not val:
            continue
        path = Path(val).with_suffix(".out") if swap else Path(val)
        if path.is_file():
            return path
    return None


def node_eta(node: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The running node's progress and time left, read from its .out by
    the same estimator as the Jobs view and ``mace cluster job``
    (``mace.cluster.progress``). None when there is no output to read."""
    out = output_of(node)
    if out is None:
        return None
    try:
        from mace.cluster import progress as progress_mod
        from mace.recovery.slurm_limits import parse_walltime
        elapsed = parse_walltime(node.get("elapsed")) if node.get("elapsed") else None
        limit = parse_walltime(node.get("time_limit")) if node.get("time_limit") else None
        prog = progress_mod.progress(out, job={"elapsed_s": elapsed, "limit_s": limit})
    except Exception:                 # never fail a status for an estimate
        return None
    eta = prog.get("eta") or {}
    low, high = eta.get("low_s"), eta.get("high_s")
    if low is None and high is None:
        text = eta.get("basis") or "time left unknown"
    elif high is None:
        text = f"at least {progress_mod._fmt_dur(low)} left"
    elif not low:
        text = f"up to {progress_mod._fmt_dur(high)} left"
    elif progress_mod._fmt_dur(low) == progress_mod._fmt_dur(high):
        text = f"about {progress_mod._fmt_dur(high)} left"
    else:
        text = f"{progress_mod._fmt_dur(low)} to {progress_mod._fmt_dur(high)} left"
    return {"low_s": low, "high_s": high, "confidence": eta.get("confidence"),
            "basis": eta.get("basis"), "stage": prog.get("stage"), "percent": prog.get("percent"),
            "summary": prog.get("summary"), "risk": prog.get("risk"), "text": text,
            "elapsed_s": elapsed, "limit_s": limit, "output": str(out)}


def status_eta(st: Dict[str, Any]) -> Dict[str, Any]:
    """Add ``eta`` to every running node of a ``workflow_status`` result
    (in place; returned for chaining)."""
    for m in st.get("materials") or []:
        for n in m.get("nodes") or []:
            if n.get("state") == "running":
                n["eta"] = node_eta(n)
    st["eta"] = True
    return st


def list_workflows(folder: Union[str, Path], use_squeue: bool = False,
                   squeue: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Every workflow planned in ``folder`` (CLI, GUI, 1.1.4/1.1.5 plans),
    newest first, with its derived state and node counts."""
    folder = Path(folder).resolve()
    sq = squeue if squeue is not None else (squeue_rows() if use_squeue else None)
    out = []
    for wid in sorted(plan_files(folder), reverse=True):
        st = workflow_status(folder, wid, squeue=sq, use_squeue=False)
        if st is None:
            continue
        item = {k: st[k] for k in ("workflow_id", "label", "created", "mode", "input_type",
                                   "sequence", "state", "counts", "plan_file", "database", "archived")}
        item["held"] = st.get("held") or {}
        item["held_total"] = sum(item["held"].values())
        item["materials"] = len(st["materials"])
        item["control"] = st["control"]["state"]
        out.append(item)
    return out


# ------------------------------------------------------------------ text
def format_status(st: Dict[str, Any]) -> str:
    lines = [f"Workflow {st['workflow_id']}" + (f" ({st['label']})" if st.get("label") else ""),
             f"  State:    {st['state']}" + ("" if st["auto_progression"] else "  (auto_progression off)"),
             f"  Sequence: {' -> '.join(st['sequence'])}",
             f"  Plan:     {st['plan_file']}",
             f"  Database: {st['database'] or '(none yet)'}"]
    if not st["squeue"]["available"]:
        lines.append("  SLURM:    not reachable here; states are the database's")
    width = max([len(m["material_id"]) for m in st["materials"]] + [8])
    lines.append("")
    lines.append(f"  {'Material':<{width}}  " + "  ".join(f"{s:<11}" for s in st["sequence"]))
    for m in st["materials"]:
        cells = []
        for n in m["nodes"]:
            cell = n["state"] + (f"x{n['attempts']}" if n.get("attempts", 0) > 1 else "")
            cells.append(f"{cell:<11}")
        lines.append(f"  {m['material_id']:<{width}}  " + "  ".join(cells))
    running = [(m["material_id"], n) for m in st["materials"] for n in m["nodes"] if n.get("eta")]
    if running:
        lines.append("")
        lines.append("  Running:")
        for mid, n in running:
            e = n["eta"]
            extra = f" ({e['summary']})" if e.get("summary") else ""
            lines.append(f"    {mid} {n['step']}: {e['text']}{extra}")
    if st["counts"]:
        lines.append("")
        lines.append("  " + ", ".join(f"{k}: {v}" for k, v in sorted(st["counts"].items())))
    if st.get("held"):
        lines.append("  Held (they start as queue slots free up): " + ", ".join(
            f"{v} {HELD_REASON_WORDS.get(k, k)}" for k, v in sorted(st["held"].items())))
    return "\n".join(lines)


def format_list(items: List[Dict[str, Any]], folder: Union[str, Path]) -> str:
    if not items:
        return f"No workflow plans in {Path(folder).resolve()}/workflow_configs"
    names = [it["workflow_id"] + (f" ({it['label']})" if it.get("label") else "") for it in items]
    w = max(len(n) for n in names + ["Workflow"])
    lines = [f"{'Workflow':<{w}}  {'State':<10} {'Mat':>4}  Sequence"]
    for name, it in zip(names, items):
        lines.append(f"{name:<{w}}  {it['state']:<10} {it['materials']:>4}  {' -> '.join(it['sequence'])}")
    return "\n".join(lines)
