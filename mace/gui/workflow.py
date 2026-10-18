"""The Workflows view's server side (GUI_WORKFLOW_DESIGN.md section 5): read
only. Everything here is derived on each call from what the CLI writes and
reads (``mace.workflow.status``: the plan, the workflow's database opened
read-only, your SLURM jobs and the control file), so a workflow started from
the CLI shows exactly as one started here, and nothing is stored.

Workflows are looked for in the served folder and in its subfolders down to
two levels (``<root>/workflows/<label>_<time>/`` included), wherever a
``workflow_configs/`` holds a plan.

``WorkflowWatcher`` hangs on the Jobs poll (``jobs.Watcher.hooks``): after
each poll it derives the workflows again and broadcasts ``workflow.changed``
on the event stream when a step's state changed.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from mace.workflow import status as wf_status
from mace.workflow.control import WORKFLOW_ID_RE

SCAN_DEPTH = 2            # the served folder and two levels of subfolders
SCAN_MAX_DIRS = 4000      # directories looked at per scan (a huge home folder stays fast)
MAX_WORKFLOWS = 200
SKIP_DIRS = {"node_modules", "__pycache__", "workflow_outputs", "workflow_staging", "workflow_inputs",
             "workflow_scripts", "archived_workflows", "venv", ".venv"}

# Plain-language names and explanations of the node states (status.py).
STATES: Dict[str, Tuple[str, str]] = {
    "planned": ("Not started", "Waits for the step it is built from."),
    "waiting": ("Starting", "The step it is built from finished; its job's completion step submits this one."),
    "stale": ("Not submitted", "The step it is built from finished a while ago and this step was not "
                               "submitted (the completion step did not run or failed)."),
    "held": ("Held", "Waiting for a free slot of your queue budget (it is submitted when one frees up), "
                     "or, with automatic progression off in the plan, to be started by hand."),
    "paused": ("Paused", "The workflow is paused: nothing new is submitted until it is resumed."),
    "canceled": ("Canceled", "The workflow or this job was canceled."),
    "queued": ("Queued", "Submitted; SLURM has not started it yet."),
    "running": ("Running", "SLURM is running it."),
    "ending": ("Finishing", "The job left the queue; its completion step has not recorded the result yet."),
    "completed": ("Completed", "Finished and recorded."),
    "failed": ("Failed", "The job failed; see its output."),
    "recovering": ("Being recovered", "The failure was fixed and the job resubmitted."),
    "blocked": ("Blocked", "The step it is built from failed or was canceled, so this one cannot run."),
}
OVERALL: Dict[str, Tuple[str, str]] = {
    "planned": ("Not started", "No step has been submitted."),
    "running": ("In progress", "Steps are queued, running or about to be submitted."),
    "completed": ("Completed", "Every step of every material finished."),
    "attention": ("Needs attention", "A step failed, is blocked or was not submitted."),
    "paused": ("Paused", "Nothing new is submitted until it is resumed."),
    "canceled": ("Canceled", "Progression stopped for good."),
}


# ------------------------------------------------------------------ discovery
def workflow_folders(root: Path, depth: int = SCAN_DEPTH) -> List[Path]:
    """Folders under ``root`` (itself included, ``depth`` levels down) whose
    ``workflow_configs/`` holds at least one plan. Hidden folders and MACE's
    own per-workflow folders are not entered; symlinks are not followed.

    ``SCAN_MAX_DIRS`` bounds the walk in directories (files, however many,
    do not count): every folder of a level reached is checked for plans,
    and once that many directories were collected no deeper level is
    read."""
    root = Path(root)
    found: List[Path] = []
    seen = 0
    level = [root]
    for d in range(depth + 1):
        for folder in level:
            if wf_status.plan_files(folder):
                found.append(folder)
        if d == depth or seen >= SCAN_MAX_DIRS:
            break
        nxt: List[Path] = []
        for folder in level:
            if seen >= SCAN_MAX_DIRS and d > 0:
                break
            try:
                with os.scandir(folder) as it:
                    for e in it:
                        if e.name.startswith(".") or e.name in SKIP_DIRS or e.name == "workflow_configs":
                            continue
                        try:
                            if not e.is_dir(follow_symlinks=False):
                                continue
                        except OSError:
                            continue
                        nxt.append(Path(e.path))
                        seen += 1
                        if seen >= SCAN_MAX_DIRS and d > 0:
                            break
            except OSError:
                continue
        level = sorted(nxt)
    return found


def _rel(root: Path, path: Optional[Any]) -> Optional[str]:
    """``path`` relative to the served folder, or None when outside it."""
    if not path:
        return None
    try:
        p = Path(str(path))
        if not p.is_absolute():
            return None
        r = p.resolve().relative_to(root.resolve())
    except (ValueError, OSError):
        return None
    s = r.as_posix()
    return "" if s == "." else s


def folder_of(root: Path, folder_rel: Optional[str]) -> Path:
    """The workflow folder a request names (relative to the served folder),
    refused when it leaves it."""
    root = Path(root).resolve()
    rel = (folder_rel or "").strip().strip("/")
    target = (root / rel).resolve() if rel and rel != "." else root
    if target != root and root not in target.parents:
        raise PermissionError("outside the served directory")
    if not target.is_dir():
        raise FileNotFoundError(f"no folder {rel or '.'}")
    return target


def check_id(workflow_id: Optional[str]) -> str:
    wid = (workflow_id or "").strip()
    if not WORKFLOW_ID_RE.match(wid):
        raise ValueError(f"not a workflow id: {wid[:60]!r}")
    return wid


# ------------------------------------------------------------------ squeue
def squeue_from_jobs(jobs) -> Optional[List[Dict[str, Any]]]:
    """Your jobs as ``mace.workflow.status`` reads them, from the GUI's
    own-jobs cache (``jobs.OwnJobs``, one squeue for the Jobs panel, the
    Cluster view and this view); None when SLURM cannot be asked here."""
    own = getattr(jobs, "own", None)
    if own is None:
        return None
    try:
        if not own.available():
            return None
        rows, _at = own.get(max_age=jobs.interval())
    except Exception:
        return None
    if not own.ok:
        return None
    return [{"id": r.get("id"), "state": r.get("state"), "elapsed": r.get("elapsed"),
             "time_limit": r.get("limit") or r.get("time_limit"), "name": r.get("name")} for r in rows]


# ------------------------------------------------------------------ views
def list_view(root: Path, squeue: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    root = Path(root).resolve()
    items: List[Dict[str, Any]] = []
    for folder in workflow_folders(root):
        for it in wf_status.list_workflows(folder, squeue=squeue, use_squeue=False):
            it = dict(it)
            it["folder_rel"] = _rel(root, folder) or ""
            it["plan_rel"] = _rel(root, it.get("plan_file"))
            it["state_label"], it["state_help"] = OVERALL.get(it["state"], (it["state"], ""))
            items.append(it)
    # the newest MAX_WORKFLOWS (ids sort by their time): sorted before the cut
    items.sort(key=lambda x: x["workflow_id"], reverse=True)
    more = max(0, len(items) - MAX_WORKFLOWS)
    items = items[:MAX_WORKFLOWS]
    return {"workflows": items, "more": more, "slurm": squeue is not None, "states": STATES,
            "overall": OVERALL}


def _eta_inside(root: Path, st: Dict[str, Any]) -> None:
    """Time left of the running steps whose output is inside the served
    folder (nothing outside it is read)."""
    for m in st.get("materials") or []:
        for n in m.get("nodes") or []:
            if n.get("state") != "running":
                continue
            out = wf_status.output_of(n)
            n["eta"] = wf_status.node_eta(n) if out is not None and _rel(root, out) is not None else None


def status_view(root: Path, folder_rel: Optional[str], workflow_id: str,
                squeue: Optional[List[Dict[str, Any]]], eta: bool = True) -> Optional[Dict[str, Any]]:
    """The workflow's derived status (``mace workflow --status ID --json``)
    with the served-folder-relative paths the page links to and the
    plain-language state names."""
    root = Path(root).resolve()
    folder = folder_of(root, folder_rel)
    wid = check_id(workflow_id)
    st = wf_status.workflow_status(folder, wid, squeue=squeue, use_squeue=False)
    if st is None:
        return None
    if squeue is None:
        st["squeue"] = {"available": False}
    if eta:
        _eta_inside(root, st)
    st["folder_rel"] = _rel(root, folder) or ""
    st["plan_rel"] = _rel(root, st.get("plan_file"))
    st["database_rel"] = _rel(root, st.get("database"))
    st["state_label"], st["state_help"] = OVERALL.get(st["state"], (st["state"], ""))
    for m in st["materials"]:
        for n in m["nodes"]:
            n["label"], n["help"] = STATES.get(n["state"], (n["state"], ""))
            if n.get("held_reason"):
                # held under the user-wide queue budget (mace queue status)
                n["help"] = (f"Held: {n.get('reason') or n['held_reason']}. MACE submits it when a slot "
                             f"of your queue budget frees up (mace queue drain).")
            elif n["state"] == "held":
                n["label"], n["help"] = "On hold", "The plan has automatic progression off: start it by hand."
            deck = n.get("input_file")
            n["deck_rel"] = _rel(root, deck)
            out = wf_status.output_of(n)
            n["out_rel"] = _rel(root, out) if out is not None else None
            if deck:
                sh = Path(deck).with_suffix(".sh")
                n["script_rel"] = _rel(root, sh) if sh.is_file() else None
            if n.get("eta") and n["eta"].get("output"):
                n["eta"]["output"] = _rel(root, n["eta"]["output"])
    st["states"] = STATES
    from .workflow_actions import clone_info
    st["clone"] = clone_info(root, folder, wid)
    return st


def signature(st: Dict[str, Any]) -> str:
    """What the view shows of a workflow, minus the time-dependent ETA."""
    cells = [(m["material_id"], n["step"], n["state"], n.get("attempts"), n.get("job_id"))
             for m in st.get("materials") or [] for n in m.get("nodes") or []]
    return json.dumps([st.get("state"), st.get("control", {}).get("state"), cells], default=str)


# ------------------------------------------------------------------ live
class WorkflowWatcher:
    """After each Jobs poll: derive the folder's workflows again and send
    ``workflow.changed`` (the changed workflows' ids and states) when a step
    changed state. Only while a Workflows view was open in the last few
    minutes (``touch``), so an idle server reads no databases."""

    LINGER_S = 600.0

    def __init__(self, root: Callable[[], Path], emit: Callable[[str, Dict[str, Any]], None],
                 squeue: Callable[[], Optional[List[Dict[str, Any]]]]):
        self.root_of = root                 # the served folder now (File > Open Folder changes it)
        self.last_root: Optional[Path] = None
        self.emit = emit
        self.squeue = squeue
        self.lock = threading.Lock()
        self.sigs: Dict[str, str] = {}
        self.until = 0.0
        self.events = 0

    def touch(self) -> None:
        import time
        self.until = time.time() + self.LINGER_S

    def remember(self, key: str, st: Dict[str, Any]) -> None:
        """A baseline for a workflow the view just read, when the watcher has
        none yet. It never replaces one: a page reading a change first (on
        jobs.finished, say) must not hide it from the other pages."""
        with self.lock:
            self.sigs.setdefault(key, signature(st))

    def check(self, squeue: Optional[List[Dict[str, Any]]] = None, force: bool = False) -> List[Dict[str, Any]]:
        import time
        if not force and time.time() > self.until:
            return []
        sq = squeue if squeue is not None else self.squeue()
        root = Path(self.root_of()).resolve()
        with self.lock:
            if self.last_root is not None and root != self.last_root:
                self.sigs = {}              # another folder (Open Folder): no old baselines
            self.last_root = root
        changed = []
        for folder in workflow_folders(root):
            for wid in sorted(wf_status.plan_files(folder)):
                st = wf_status.workflow_status(folder, wid, squeue=sq, use_squeue=False)
                if st is None:
                    continue
                key = f"{_rel(root, folder) or ''}::{wid}"
                sig = signature(st)
                with self.lock:
                    old = self.sigs.get(key)
                    self.sigs[key] = sig
                if old is not None and old != sig:
                    changed.append({"workflow_id": wid, "folder_rel": _rel(root, folder) or "",
                                    "state": st["state"], "counts": st["counts"]})
        if changed:
            self.events += 1
            self.emit("workflow.changed", {"workflows": changed})
        return changed

    # the GUI releases held jobs too (mace queue drain), at most this often
    DRAIN_EVERY_S = 120.0

    def maybe_drain(self, now: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Submit held jobs of every active workflow while the queue budget
        has room (mace/queue/budget.py drain), at most every DRAIN_EVERY_S;
        it never waits long for the budget lock."""
        import time
        now = time.time() if now is None else now
        if now - getattr(self, "last_drain", 0.0) < self.DRAIN_EVERY_S:
            return None
        self.last_drain = now
        try:
            from mace.queue import budget
            if not budget.registered_dbs():
                return None
            rep = budget.drain(out=lambda *_a, **_k: None, quiet=True, timeout=2)
        except Exception:
            return None
        if rep.get("released"):
            self.emit("workflow.changed", {"workflows": [], "released": len(rep["released"])})
        return rep

    def hook(self, _root, _rows) -> None:
        """jobs.Watcher hook: called after each poll of your jobs."""
        try:
            self.check()
        except Exception:
            pass
        self.maybe_drain()
