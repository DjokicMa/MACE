"""The Workflows view's actions (GUI_WORKFLOW_DESIGN.md sections 4.7 and 5,
Phase 5). Each one is a CLI command run in the workflow's folder, with the
environment ``mace gui`` was started with; the GUI changes nothing itself:

* Pause / Resume / Cancel: ``mace workflow --pause|--resume|--cancel ID --json``.
  Cancel first runs ``--cancel ID --json`` without ``--yes`` (the CLI lists
  the jobs it would scancel and changes nothing); the confirmation sends that
  list back, and only when the CLI still lists exactly those jobs does it run
  ``--cancel ID --yes --json``.
* Recover (one failed step): ``mace recover --db DB --calc-id ID --dry-run
  --json``, then, with the preview's single-use id, the same with ``--yes``.
* Process now (a step whose completion step never ran):
  ``mace workflow --process ID --json``.
* Clone: the builder's draft of an existing workflow, from the draft the
  builder saved next to its spec (``builder_draft.json``), else the spec as
  launched (``workflow_configs/spec_<id>/spec.json``), else the plan (a
  workflow started with ``mace submit --progress``). Read only.

Pause only writes the control file, so it is allowed while MACE GUI is open
to the network; Resume, Cancel, Recover and Process now change SLURM's
state and are refused there.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mace.workflow import control as wf_control
from mace.workflow import status as wf_status

from . import submit as sub
from . import workflow as wf

ACTIONS = ("pause", "resume", "cancel")
RECOVER_TTL_S = 1800
FAILED = ("failed", "error", "timeout")
DRAFT_NAME = "builder_draft.json"
DRAFT_MAX = 2_000_000
CLONE_RESOURCES = ("ntasks", "walltime", "memory", "memory_per_cpu", "constraint", "exclude")
NETWORK = "is turned off while MACE GUI is open to the network"
_LOCK = threading.Lock()


# ------------------------------------------------------------------ helpers
def target(root: Path, folder_rel: Optional[str], workflow_id: Optional[str]) -> Tuple[Path, str, Dict[str, Any]]:
    """(folder, workflow id, plan) a request names; 400 for a bad id, 403
    outside the served folder, 404 when that folder has no such plan."""
    wid = wf.check_id(workflow_id)
    folder = wf.folder_of(root, folder_rel)
    if wid not in wf_status.plan_files(folder):
        raise FileNotFoundError(f"No plan of {wid} in {folder_rel or 'this folder'}/workflow_configs")
    return folder, wid, wf_status.load_plan(folder, wid) or {}


def held(folder: Path, wid: str) -> Optional[str]:
    state = wf_control.state_of(wf_control.read_control(folder, wid))
    return state if state in wf_control.HOLD_STATES else None


def run_cli(folder: Path, argv: List[str], ctx=None) -> Tuple[int, Dict[str, Any], str]:
    """``mace ARGV`` in ``folder`` (no time limit, not stopped halfway):
    (exit code, the JSON document it printed, its log)."""
    from . import clirun
    from .recover import json_tail
    r = clirun.run(sub.mace_argv(*argv), folder, ctx, timeout=None, env=sub.submit_env(), nice=False,
                   cancelable=False)
    try:
        doc = json_tail(r.stdout)
    except ValueError:
        doc = {"ok": False, "error": ((r.stderr or r.stdout).strip()[-800:] or f"mace {argv[0]} printed nothing")}
    if not isinstance(doc, dict):
        doc = {"ok": False, "error": "unexpected answer"}
    return r.returncode, doc, (r.stdout + r.stderr)[-6000:]


def after_change(st) -> None:
    """Jobs were submitted or canceled: the Jobs panel and every Workflows
    view see it now, not at the next poll."""
    jobs = getattr(st, "jobs", None)
    if jobs is not None:
        try:
            jobs.own.get(fresh="force")
            jobs.kick()
        except Exception:
            pass
    watcher = getattr(st, "workflows", None)
    if watcher is not None:
        try:
            watcher.touch()
            watcher.check(force=True)
        except Exception:
            pass


# ------------------------------------------------------------------ pause / resume / cancel
def control(st, folder: Path, wid: str, action: str, body: Dict[str, Any], network: bool, ctx=None) -> Dict[str, Any]:
    if action in ("pause", "resume"):
        argv = ["workflow", f"--{action}", wid, "--json"]
        _rc, doc, log = run_cli(folder, argv, ctx)
        if action == "resume":
            after_change(st)
        else:
            after_change_light(st)
        return {**doc, "argv": ["mace", *argv]}
    # cancel: the list the CLI would scancel, then exactly that list
    list_argv = ["workflow", "--cancel", wid, "--json"]
    run_argv = ["workflow", "--cancel", wid, "--yes", "--json"]
    _rc, listed, _log = run_cli(folder, list_argv, ctx)
    if not listed.get("confirm"):
        return {**listed, "argv": ["mace", *list_argv]}
    jobs = listed.get("jobs") or []
    if not body.get("confirm"):
        return {"ok": True, "confirm": True, "workflow_id": wid, "jobs": jobs, "squeue": bool(listed.get("squeue")),
                "argv": ["mace", *list_argv], "run_argv": ["mace", *run_argv]}
    want = body.get("jobs")
    if not isinstance(want, list) or not all(isinstance(x, (str, int)) for x in want):
        raise sub.SubmitError("jobs is the list of job ids you confirmed")
    if sorted(str(x) for x in want) != sorted(str(j.get("job_id")) for j in jobs):
        return {"ok": False, "changed": True, "workflow_id": wid, "jobs": jobs, "squeue": bool(listed.get("squeue")),
                "error": "The workflow's jobs changed since you looked: check the list again.",
                "run_argv": ["mace", *run_argv]}
    _rc, doc, _log = run_cli(folder, run_argv, ctx)
    after_change(st)
    return {**doc, "argv": ["mace", *run_argv]}


def after_change_light(st) -> None:
    watcher = getattr(st, "workflows", None)
    if watcher is not None:
        try:
            watcher.touch()
            watcher.check(force=True)
        except Exception:
            pass


def process(st, folder: Path, wid: str, ctx=None) -> Dict[str, Any]:
    argv = ["workflow", "--process", wid, "--json"]
    _rc, doc, _log = run_cli(folder, argv, ctx)
    if doc.get("steps_started"):
        after_change(st)
    return {**doc, "argv": ["mace", *argv]}


# ------------------------------------------------------------------ recover one step
def _store(st) -> Dict[str, Dict[str, Any]]:
    with _LOCK:
        if not hasattr(st, "wf_recover_previews"):
            st.wf_recover_previews = {}
        now = time.time()
        for k in [k for k, v in st.wf_recover_previews.items() if now - v["at"] > RECOVER_TTL_S]:
            st.wf_recover_previews.pop(k, None)
        return st.wf_recover_previews


def recover_argv(folder: Path, wid: str, plan: Dict[str, Any], calc_id: Any) -> List[str]:
    """``recover --db DB --calc-id ID`` for a failed calculation of this
    workflow (400 for anything else)."""
    cid = str(calc_id or "").strip()
    db = wf_status.database_of(folder, wid, plan)
    rows = wf_status.read_rows(db, wid) if db is not None else []
    row = next((r for r in rows if str(r.get("calc_id")) == cid), None)
    if not cid or row is None or str(row.get("status") or "").lower() not in FAILED:
        raise sub.SubmitError(f"{cid[:80] or 'that'} is not a failed step of {wid}")
    try:
        db_arg = db.resolve().relative_to(folder.resolve()).as_posix()
    except ValueError:
        db_arg = str(db)
    return ["recover", "--db", db_arg, "--calc-id", cid]


def recover_preview(st, folder: Path, wid: str, base: List[str], gen: Any, network: bool, ctx=None) -> Dict[str, Any]:
    argv = base + ["--dry-run", "--json"]
    run_argv = base + ["--yes", "--json"]
    _rc, doc, log = run_cli(folder, argv, ctx)
    if "entries" not in doc:
        raise sub.SubmitError("mace recover gave no plan: " + str(doc.get("error") or log[-600:]))
    from .recover import remember_preview          # the Jobs panel's Recover answers the same way
    return remember_preview(_store(st), "wfrec-", gen, doc, argv, run_argv, network, folder=str(folder), wid=wid)


def take_recover(st, pid: str, gen: Any) -> Dict[str, Any]:
    """The preview a run names (single use), refused when it is unknown,
    old, of another folder generation, or its workflow is held now."""
    p = _store(st).pop(str(pid or ""), None)
    if p is None or str(p["gen"]) != str(gen):
        raise sub.SubmitConflict("This preview is no longer current: preview again")
    state = held(Path(p["folder"]), p["wid"])
    if state:
        raise sub.SubmitConflict(f"The workflow is {state}: resume it first (a {state} workflow submits no "
                                 "recovery job)")
    return p


def recover_run(st, p: Dict[str, Any], include_stale: bool, ctx=None) -> Dict[str, Any]:
    from .recover import run_argv_of
    argv = run_argv_of(p, include_stale)
    rc, doc, log = run_cli(Path(p["folder"]), argv, ctx)
    res = doc.get("results", [])
    if any(x.get("status") == "submitted" for x in res):
        after_change(st)
    return {"argv": ["mace", *argv], "text": doc.get("text", "") or ("" if res else log[-3000:]), "results": res,
            "exit": rc}


# ------------------------------------------------------------------ clone
def _inside(root: Path, p: Any) -> Optional[Path]:
    try:
        q = Path(str(p)).resolve()
        q.relative_to(root.resolve())
        return q
    except (ValueError, OSError, TypeError):
        return None


def _read_json(p: Optional[Path]) -> Optional[Dict[str, Any]]:
    if p is None or not p.is_file():
        return None
    try:
        if p.stat().st_size > DRAFT_MAX:
            return None
        v = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    return v if isinstance(v, dict) else None


def _copy_label(label: Any) -> str:
    base = str(label or "").strip() or "workflow"
    return (base[:59] + "-copy") if len(base) > 59 else base + "-copy"


def _step_resources(cfg: Dict[str, Any]) -> Dict[str, Any]:
    res = ((cfg.get("slurm_config") or {}).get("resources")) or cfg.get("resources") or {}
    out = {k: res[k] for k in CLONE_RESOURCES if res.get(k) not in (None, "")}
    if "memory" in out and "memory_per_cpu" in out:
        out.pop("memory_per_cpu" if res.get("mem_form") != "mem_per_cpu" else "memory")
    return out


def clone(root: Path, folder: Path, wid: str, plan: Dict[str, Any], with_draft: bool = True) -> Dict[str, Any]:
    """{available, source: builder|spec|plan, kind, targets, draft, notes}
    or {available: False, reason}: the builder pre-filled from this
    workflow. Only files inside the served folder are read."""
    from mace.workflow.spec import spec_dir
    root = Path(root).resolve()
    seq = list(plan.get("workflow_sequence") or [])
    launched = _read_json(spec_dir(folder, wid) / "spec.json")
    meta = plan.get("spec") if isinstance(plan.get("spec"), dict) else {}
    src = _inside(root, meta.get("source")) if meta.get("source") else None
    saved = _read_json(src.parent / DRAFT_NAME) if src is not None else None
    kind = (launched or {}).get("inputs", {}).get("type") or plan.get("input_type")
    if kind not in ("cif", "d12"):
        return {"available": False, "reason": "The plan does not say whether it starts from CIF or .d12 files."}
    if launched:
        files = list((launched.get("inputs") or {}).get("files") or [])
        seq = list(launched.get("sequence") or seq)
    else:
        files = list(((plan.get("input_files") or {}).get(kind)) or [])
    if not files or not seq:
        return {"available": False, "reason": "The plan names no structures or no steps."}
    targets, missing = [], []
    for f in files:
        p = _inside(root, f if Path(str(f)).is_absolute() else folder / str(f))   # a 1.1.5 plan: relative
        if p is None or not p.is_file():
            missing.append(Path(str(f)).name)
        else:
            targets.append(p.relative_to(root).as_posix())
    if missing:
        return {"available": False, "reason": f"Its structures are not in this folder any more: "
                                              f"{', '.join(missing[:5])}{'…' if len(missing) > 5 else ''}"}
    label = (launched or {}).get("label") or plan.get("label")
    out: Dict[str, Any] = {"available": True, "kind": kind, "targets": targets, "notes": []}
    if saved is not None and isinstance(saved.get("sequence"), list) and list(saved["sequence"]) == seq:
        out["source"] = "builder"
        if not with_draft:
            return out
        draft = {k: saved[k] for k in ("template", "sequence", "steps", "execution") if k in saved}
        out["draft"] = {**draft, "label": _copy_label(saved.get("label") or label), "labelAuto": False}
        return out
    if launched:
        out["source"] = "spec"
        if with_draft:
            out["draft"], out["notes"] = _draft_of_spec(root, launched, seq, folder, wid, label)
        return out
    out["source"] = "plan"
    out["notes"] = ["Started from the command line without a spec: the structures, steps and job sizes are "
                    "copied; each step's calculation settings start from MACE's defaults. Check them in Settings."]
    if with_draft:
        steps: Dict[str, Any] = {}
        for i, t in enumerate(seq):
            key = f"{t}_{i + 1}"
            res = _step_resources((plan.get("step_configurations") or {}).get(key) or {})
            if res:
                steps[key] = {"resources": res}
        mj = (plan.get("execution_settings") or {}).get("max_concurrent_jobs")
        ex = {"max_jobs": str(mj)} if isinstance(mj, int) and mj != 200 else {}
        if (plan.get("execution_settings") or {}).get("auto_progression") is False:
            ex["auto_progression"] = False
        tmpl = _template_of(seq)
        out["draft"] = {"template": tmpl, "sequence": seq, "steps": steps, "execution": ex,
                        "label": _copy_label(label), "labelAuto": False}
    return out


def _template_of(seq: List[str]) -> Optional[str]:
    from mace.workflow.common.constants import WORKFLOW_TEMPLATES
    return next((n for n, s in WORKFLOW_TEMPLATES.items() if list(s) == list(seq)), None)


def _draft_of_spec(root: Path, spec: Dict[str, Any], seq: List[str], folder: Path, wid: str,
                   label: Any) -> Tuple[Dict[str, Any], List[str]]:
    from mace.workflow.spec import spec_dir
    notes: List[str] = []
    steps: Dict[str, Any] = {}
    ex_in = spec.get("execution") or {}
    for i, t in enumerate(seq):
        key = f"{t}_{i + 1}"
        e = (spec.get("steps") or {}).get(key) or {}
        s: Dict[str, Any] = {}
        res = {k: v for k, v in (e.get("resources") or {}).items() if v not in (None, "")}
        if ex_in.get("walltime") and "walltime" not in res:
            res["walltime"] = ex_in["walltime"]
        if res:
            s["resources"] = res
        mm = ((e.get("overrides") or {}).get("method_modifications")) or {}
        base = t.rstrip("0123456789")
        if i > 0 and base in ("OPT", "SP", "FREQ") and mm.get("new_functional"):
            fn = str(mm["new_functional"])
            # no use_dispersion: the step keeps its parent's D3 (the engine and opt2d12 do the same)
            if mm.get("use_dispersion") is None:
                s["method"] = {"functional": fn}
                s["answers"] = {"functional": fn}
            else:
                disp = bool(mm.get("use_dispersion"))
                s["method"] = {"functional": fn, "dispersion": disp}
                s["answers"] = {"functional": fn, "dispersion": "Yes" if disp else "No"}
            if set(e.get("overrides") or {}) - {"method_modifications"} or set(mm) - {"new_functional", "use_dispersion"}:
                notes.append(f"{key}: settings beyond the functional and dispersion are not copied.")
        elif e.get("overrides"):
            notes.append(f"{key}: its changed settings are not copied; set them again in Settings.")
        if e.get("settings_file"):
            text = None
            if i > 0 and base in ("BAND", "DOSS"):
                for cand in (_inside(root, e["settings_file"]),
                             _inside(root, spec_dir(folder, wid) / f"{key}_config.json")):
                    if cand is not None and cand.is_file():
                        try:
                            text = cand.read_text()
                        except OSError:
                            text = None
                        if text:
                            break
            if text:
                s["settings_text"] = text
                s["from_clone"] = True
            else:
                notes.append(f"{key}: its settings file is not copied; set its settings again in Settings.")
        if s:
            steps[key] = s
    ex: Dict[str, Any] = {}
    if isinstance(ex_in.get("max_jobs"), int) and ex_in["max_jobs"] != 200:
        ex["max_jobs"] = str(ex_in["max_jobs"])
    if ex_in.get("job_arrays"):
        ex["array"] = True
        if (ex_in["job_arrays"] or {}).get("max_running"):
            ex["max_running"] = str(ex_in["job_arrays"]["max_running"])
    if ex_in.get("auto_progression") is False:
        ex["auto_progression"] = False
    draft = {"template": spec.get("template") or _template_of(seq), "sequence": seq, "steps": steps,
             "execution": ex, "label": _copy_label(label), "labelAuto": False}
    return draft, notes


def clone_info(root: Path, folder: Path, wid: str) -> Dict[str, Any]:
    """What the Clone button shows: available (and from where) or why not."""
    try:
        plan = wf_status.load_plan(folder, wid) or {}
        info = clone(root, folder, wid, plan, with_draft=False)
    except Exception as exc:              # never breaks the status view
        return {"available": False, "reason": f"The plan could not be read: {exc}"}
    return {k: info[k] for k in ("available", "source", "reason") if k in info}


def ui_draft_file(ui_draft: Any) -> Optional[str]:
    """The builder's own draft, saved next to the spec it wrote (Clone
    reopens it exactly); None when there is none. The CLI never reads it."""
    if ui_draft is None:
        return None
    if not isinstance(ui_draft, dict):
        raise sub.SubmitError("ui_draft is the builder's draft (an object)")
    keep = {k: ui_draft[k] for k in ("label", "template", "sequence", "steps", "execution") if k in ui_draft}
    text = json.dumps(keep, indent=2, default=str) + "\n"
    if len(text) > DRAFT_MAX:
        raise sub.SubmitError("the builder's draft is too large")
    return text
