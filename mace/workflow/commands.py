"""`mace workflow` actions for workflows that already have a plan:
--status ID, --list, --spec ... --write-plan, --pause/--resume/--cancel ID,
--process ID.

--process and --resume start steps under the queue lock the job callbacks
hold (QUEUE_STATE_LOCK), so the two never submit the same step twice.

Each returns an exit code; with ``as_json`` stdout carries exactly one JSON
document (anything else MACE prints goes to stderr), which is what the GUI
reads.
"""
import contextlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from mace.workflow import control, status

# database states of a calculation that may still be in SLURM's queue
_LIVE = ("submitted", "pending", "queued", "running")
# how long --process/--resume wait for the queue lock a job's completion
# step holds (seconds)
LOCK_WAIT_S = 120


class QueueBusy(RuntimeError):
    """Another MACE process held the queue lock for LOCK_WAIT_S seconds."""


def _emit(obj: Any, as_json: bool, text: str = "") -> None:
    if as_json:
        print(json.dumps(obj, indent=2, default=str))
    elif text:
        print(text)


def _err(msg: str, as_json: bool, code: int = 1, **extra) -> int:
    if as_json:
        print(json.dumps(dict({"ok": False, "error": msg}, **extra), indent=2, default=str))
    else:
        print(f"Error: {msg}", file=sys.stderr)
    return code


def _known(folder: Path, workflow_id: str, as_json: bool) -> Optional[int]:
    if workflow_id not in status.plan_files(folder):
        known = ", ".join(sorted(status.plan_files(folder))) or "none"
        return _err(f"no plan of {workflow_id} in {folder}/workflow_configs (workflows here: {known})",
                    as_json)
    return None


# ------------------------------------------------------------------ read
def cmd_status(folder: Path, workflow_id: str, as_json: bool) -> int:
    rc = _known(folder, workflow_id, as_json)
    if rc is not None:
        return rc
    st = status.workflow_status(folder, workflow_id, eta=True)
    _emit(st, as_json, status.format_status(st))
    return 0


def cmd_list(folder: Path, as_json: bool) -> int:
    items = status.list_workflows(folder, use_squeue=True)
    _emit({"folder": str(folder.resolve()), "workflows": items}, as_json,
          status.format_list(items, folder))
    return 0


# ------------------------------------------------------------------ spec
def cmd_spec(spec_path: str, folder: Path, db_path: str, write_plan: bool, as_json: bool,
             work_dir_given: bool = True) -> int:
    from mace.workflow.spec import SpecError, build_plan, load_spec, resolve_defaults
    try:
        spec = load_spec(spec_path)
    except SpecError as e:
        return _err(str(e), as_json, code=2)
    if write_plan and spec["inputs"]["type"] == "d12":
        # decks are submitted from their own folder, where `mace submit
        # --progress-plan` looks for the plan: write it there (the default),
        # never somewhere the launch would refuse
        decks = Path(spec["inputs"]["files"][0]).parent.resolve()
        if decks != Path(folder).resolve():
            if work_dir_given:
                return _err(f"the decks of this spec are submitted from their folder {decks}, so its plan "
                            f"goes there: use --work-dir {decks} (or leave --work-dir out)", as_json, code=2)
            folder = decks
    if not write_plan:
        # check only: what the plan would hold, nothing written
        with contextlib.redirect_stdout(sys.stderr):
            steps = resolve_defaults(spec, folder)
        text = "\n".join(
            f"{k}: settings {v['settings_origin']}; "
            + ", ".join(f"{rk}={rv} ({v['origins'].get(rk)})" for rk, rv in v["resources"].items()
                        if rk in ("ntasks", "walltime", "memory", "memory_per_cpu", "partition", "account",
                                  "constraint"))
            for k, v in steps.items())
        _emit({"ok": True, "written": False, "sequence": spec["sequence"], "steps": steps}, as_json,
              text + "\n\nNothing written; add --write-plan to write the plan.")
        return 0
    try:
        with contextlib.redirect_stdout(sys.stderr if as_json else sys.stdout):
            plan_file, workflow_id, plan = build_plan(spec, folder, db_path)
    except SpecError as e:
        return _err(str(e), as_json, code=2)
    if spec["inputs"]["type"] == "d12":
        files = spec["inputs"]["files"]
        target = files[0] if len(files) == 1 else str(Path(files[0]).parent)
        launch = ["mace", "submit", "--progress-plan", str(plan_file), target]
    else:
        launch = ["mace", "workflow", "--execute", str(plan_file)]
    _emit({"ok": True, "written": True, "workflow_id": workflow_id, "plan_file": str(plan_file),
           "sequence": plan.get("workflow_sequence"), "input_type": spec["inputs"]["type"],
           "launch": launch}, as_json,
          f"Workflow plan written: {plan_file}\n  Workflow ID: {workflow_id}\n"
          f"  Sequence:    {' -> '.join(plan.get('workflow_sequence') or [])}\n"
          f"  Launch it:   {' '.join(launch)}")
    return 0


# ------------------------------------------------------------------ control
def cmd_pause(folder: Path, workflow_id: str, as_json: bool) -> int:
    rc = _known(folder, workflow_id, as_json)
    if rc is not None:
        return rc
    data = control.write_control(folder, workflow_id, "paused")
    _emit({"ok": True, "workflow_id": workflow_id, "state": data["state"]}, as_json,
          f"{workflow_id} is paused: completions are recorded, no next step and no recovery job "
          f"is submitted.\n  Resume with: mace workflow --resume {workflow_id}")
    return 0


def cmd_resume(folder: Path, workflow_id: str, as_json: bool) -> int:
    rc = _known(folder, workflow_id, as_json)
    if rc is not None:
        return rc
    control.write_control(folder, workflow_id, "active")
    # the steps held while paused, as `mace engine --action process` would
    # start them, restricted to this workflow
    try:
        started, db, shared = _process(folder, workflow_id, as_json)
    except QueueBusy as e:
        return _err(f"{workflow_id} is active again, but no held step was started: {e} (mace workflow "
                    f"--process {workflow_id} starts them)", as_json, state="active")
    _emit({"ok": True, "workflow_id": workflow_id, "state": "active", "steps_started": started,
           "database": str(db) if db else None, "shared_database": shared}, as_json,
          f"{workflow_id} is active again; {started} held step(s) started.")
    return 0


def _process(folder: Path, workflow_id: str, as_json: bool):
    """(steps started, database, shared?): the workflow's completed steps
    that nothing processed yet, processed as the job callback would
    (``process_completed_calculations``), restricted to this workflow."""
    db = status.database_of(folder, workflow_id, status.load_plan(folder, workflow_id))
    if db is None or status.is_archived(db):
        return 0, db, False
    # in a shared database (the folder's materials.db) only rows stamped with its id
    shared = not db.parent.name.startswith(".mace_context_")
    from mace.workflow.engine import WorkflowEngine
    from mace.queue.queue_lock_manager import QueueLockManager, QUEUE_STATE_LOCK
    with contextlib.redirect_stdout(sys.stderr if as_json else sys.stdout):
        # the lock the job callbacks hold while they progress this database
        # (manager.py: the context's .queue_locks, which is the database's
        # folder's): a callback generating the same step meanwhile would
        # otherwise submit it a second time
        # (Ctrl+C keeps its usual meaning here: the lock is released by
        # with_lock, and an OS lock goes with the process anyway)
        import signal
        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        locks = QueueLockManager(lock_dir=Path(db).resolve().parent / ".queue_locks", lock_timeout=300)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)

        def locked():
            from mace.queue import budget
            engine = WorkflowEngine(str(db), str(folder))
            # the next steps are held, then submitted by the queue drain in
            # the [queue] order (with every active workflow's held work)
            with budget.deferred():
                n = engine.process_completed_calculations(workflow_id=workflow_id)
            try:
                rep = budget.drain(dbs=[str(db)], out=print)
            except Exception as e:
                print(f"Queue drain skipped: {e}")
                rep = {}
            LAST_DRAIN.clear()
            LAST_DRAIN.update(rep)
            return n
        try:
            started = locks.with_lock(QUEUE_STATE_LOCK, locked, timeout=LOCK_WAIT_S)
        except TimeoutError:
            raise QueueBusy(f"another MACE process (a job's completion step or a sweep) held the queue "
                            f"lock for {LOCK_WAIT_S} s")
    return started, db, shared


# the queue drain report of the last _process (released, held by reason)
LAST_DRAIN: Dict[str, Any] = {}


def _drain_fields() -> Dict[str, Any]:
    rep = LAST_DRAIN or {}
    return {"released": len(rep.get("released") or []), "held": rep.get("held") or {}}


def cmd_process(folder: Path, workflow_id: str, as_json: bool) -> int:
    """Start the steps a missing completion step did not submit (a job that
    finished while its callback failed or never ran). A paused or canceled
    workflow is refused: --resume starts its held steps."""
    rc = _known(folder, workflow_id, as_json)
    if rc is not None:
        return rc
    held = control.state_of(control.read_control(folder, workflow_id))
    if held in control.HOLD_STATES:
        return _err(f"{workflow_id} is {held}: nothing was started (mace workflow --resume {workflow_id} "
                    f"starts its held steps)", as_json, state=held)
    try:
        started, db, shared = _process(folder, workflow_id, as_json)
    except QueueBusy as e:
        return _err(f"{workflow_id}: nothing was started: {e}; run it again", as_json)
    d = _drain_fields()
    held_n = sum(d["held"].values())
    _emit({"ok": True, "workflow_id": workflow_id, "steps_started": started,
           "database": str(db) if db else None, "shared_database": shared, **d}, as_json,
          f"{workflow_id}: {started} step(s) started."
          + (f" {d['released']} held job(s) submitted." if d["released"] else "")
          + (f" {held_n} job(s) held (they start as queue slots free up)." if held_n else ""))
    return 0


def workflow_jobs(folder: Path, workflow_id: str,
                  squeue: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """This workflow's jobs that SLURM lists among your own: its database
    rows that may be queued or running, kept only when a fresh squeue
    shows the same job id. ``squeue`` None asks SLURM; when SLURM cannot be
    asked nothing is confirmed (and nothing would be canceled)."""
    plan = status.load_plan(folder, workflow_id)
    db = status.database_of(folder, workflow_id, plan)
    rows = [r for r in status.read_rows(db, workflow_id)
            if str(r.get("status") or "").lower() in _LIVE and r.get("slurm_job_id")]
    if squeue is None:
        try:
            from mace.cluster import reader
            res = reader.squeue_own(fresh=True, tag="workflow-cancel")
            squeue = res.value if (res is not None and not res.error and isinstance(res.value, list)) else None
        except Exception:
            squeue = None
    if squeue is None:
        return {"database": str(db) if db else None, "confirmed": [], "unconfirmed":
                [r["slurm_job_id"] for r in rows], "squeue": False}
    listed = {str(q.get("id", "")).strip(): q for q in squeue}
    confirmed, unconfirmed = [], []
    for r in rows:
        jid = str(r["slurm_job_id"]).strip()
        if jid in listed:
            confirmed.append({"job_id": jid, "calc_id": r.get("calc_id"), "material_id": r.get("material_id"),
                              "calc_type": r.get("calc_type"), "state": listed[jid].get("state")})
        else:
            unconfirmed.append(jid)
    return {"database": str(db) if db else None, "confirmed": confirmed, "unconfirmed": unconfirmed,
            "squeue": True}


# squeue states of a job that scancel stopped (or that ended by itself)
_GONE = ("COMPLETING", "COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL",
         "PREEMPTED", "BOOT_FAIL", "DEADLINE")


def _classify_scancel(folder: Path, workflow_id: str, ids: List[str], stderr: str):
    """(canceled, refused, ended) after one ``scancel`` of ``ids`` that
    exited non-zero: it still cancels every id it can, so a fresh squeue
    tells them apart. A job squeue still lists is refused (with scancel's
    line about it); one it no longer lists was canceled, or, when scancel
    named it, had ended by itself. Without squeue every id is refused."""
    import re
    lines = [l.strip() for l in stderr.splitlines() if l.strip()]

    def said(jid: str) -> Optional[str]:
        return next((l for l in lines if re.search(rf"(?<![\w.]){re.escape(jid)}(?![\w.])", l)), None)

    after = workflow_jobs(folder, workflow_id)
    if not after["squeue"]:
        why = (stderr or "scancel failed").strip()[:300]
        return [], [{"job_id": j, "reason": why} for j in ids], []
    listed = {j["job_id"] for j in after["confirmed"]
              if str(j.get("state") or "").upper().rstrip("+") not in _GONE}
    canceled, refused, ended = [], [], []
    for j in ids:
        line = said(j)
        if j in listed:
            refused.append({"job_id": j, "reason": (line or stderr.strip() or "scancel failed")[:300]})
        elif line:
            ended.append(j)
        else:
            canceled.append(j)
    return canceled, refused, ended


def cmd_cancel(folder: Path, workflow_id: str, as_json: bool, yes: bool) -> int:
    rc = _known(folder, workflow_id, as_json)
    if rc is not None:
        return rc
    jobs = workflow_jobs(folder, workflow_id)
    listing = "\n".join(f"  {j['job_id']:<14} {j['calc_type']:<8} {j['material_id']} ({j['state']})"
                        for j in jobs["confirmed"]) or "  (none queued or running)"
    if not yes:
        if as_json or not sys.stdin.isatty():
            _emit({"ok": False, "confirm": True, "workflow_id": workflow_id, "jobs": jobs["confirmed"],
                   "squeue": jobs["squeue"],
                   "message": "nothing changed; run again with --yes to cancel"}, as_json,
                  f"{workflow_id}: these jobs would be canceled:\n{listing}\n"
                  f"Nothing changed; run again with --yes to cancel.")
            return 1
        print(f"Canceling {workflow_id} stops its progression and cancels these jobs:\n{listing}")
        if input("Cancel the workflow? [y/N]: ").strip().lower() not in ("y", "yes"):
            print("Nothing changed.")
            return 1
    jobs = workflow_jobs(folder, workflow_id)
    if not jobs["squeue"] and jobs["unconfirmed"]:
        # the database lists queued or running jobs and SLURM cannot say
        # which are still there: canceling the progression alone would
        # leave them running under a workflow marked canceled
        state = control.state_of(control.read_control(folder, workflow_id))
        return _err(f"SLURM could not be asked (squeue failed), so no job was canceled and {workflow_id} "
                    f"was left {state}; its database lists {len(jobs['unconfirmed'])} queued or running "
                    f"job(s): {', '.join(jobs['unconfirmed'])}. Run again where squeue works.",
                    as_json, workflow_id=workflow_id, state=state, canceled=[], refused=[],
                    unverified=jobs["unconfirmed"], squeue=False)
    # progression stops first, so a job ending now submits nothing
    control.write_control(folder, workflow_id, "canceled")
    if jobs.get("database"):
        # jobs held under the queue budget are never submitted now
        from mace.queue import budget
        budget.cancel_held(jobs["database"], workflow_id)
    jobs = workflow_jobs(folder, workflow_id)
    canceled, refused = [], []
    ids = [j["job_id"] for j in jobs["confirmed"]]
    if ids:
        from mace.queue.slurm_ids import safe_scancel_many
        db_obj = None
        if jobs["database"]:
            try:
                from mace.database.materials import MaterialDatabase
                db_obj = MaterialDatabase(jobs["database"])
            except Exception:
                db_obj = None
        proc, refused_pairs = safe_scancel_many(ids, db=db_obj, search_dirs=[str(folder)])
        refused = [{"job_id": j, "reason": why} for j, why in refused_pairs]
        allowed = [j for j in ids if j not in {r["job_id"] for r in refused}]
        if proc is not None and proc.returncode == 0:
            canceled = allowed
        elif proc is not None:
            done, failed, ended = _classify_scancel(folder, workflow_id, allowed, proc.stderr or "")
            canceled = done
            refused += failed
            jobs["unconfirmed"] = list(jobs["unconfirmed"]) + ended
    _emit({"ok": not refused, "workflow_id": workflow_id, "state": "canceled", "canceled": canceled,
           "refused": refused, "not_in_queue": jobs["unconfirmed"], "squeue": jobs["squeue"]}, as_json,
          f"{workflow_id} is canceled; scancel: {', '.join(canceled) or 'nothing'}"
          + "".join(f"\n  not canceled {r['job_id']}: {r['reason']}" for r in refused)
          + ("" if jobs["squeue"] else "\n  SLURM could not be asked: no job was canceled"))
    return 0 if not refused else 1
