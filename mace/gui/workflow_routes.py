"""The Workflows view's and the workflow builder's routes
(GUI_WORKFLOW_DESIGN.md section 5).

Read only (workflow.py):
GET  workflow/list                          every workflow under the served folder
                                            (CLI- and GUI-started), newest first
GET  workflow/status?id=ID&folder=REL&eta=1 one workflow: every material's steps,
                                            as ``mace workflow --status ID --json``,
                                            with the time left of running steps

The builder (workflow_build.py; the GUI writes a spec, the CLI does the rest):
GET  workflow/templates                     the chains to start from, the step types
POST workflow/inputs     {targets, type?}   the structures a workflow starts from
POST workflow/defaults   {draft}            -> task: ``mace workflow --spec SPEC --json``
POST workflow/step-form  {draft, step, answers, patches, json_text}
                                            -> task: the step's Calculation Setup form
POST workflow/preview    {root_gen, draft}  -> task: the CLI's commands in a staged copy
POST workflow/run        {root_gen, preview_id}
                                            -> task: the same commands, for real

The Workflows view's actions (workflow_actions.py; each one a CLI command
in the workflow's folder, given as {root_gen, folder, id}):
POST workflow/control    {action: pause|resume|cancel, confirm?, jobs?}
                                            -> task: mace workflow --pause|--resume|--cancel ID;
                                            cancel without confirm lists the jobs (nothing
                                            changes), with confirm and that exact list cancels
POST workflow/recover/preview {calc_id}     -> task: mace recover --db DB --calc-id ID --dry-run
POST workflow/recover/run {preview_id, include_stale?}
                                            -> task: the same with --yes (single-use id)
POST workflow/process                       -> task: mace workflow --process ID
GET  workflow/clone?id=ID&folder=REL        the builder's draft of that workflow (read only)

Nothing in the read-only routes writes: the plan, the database (opened
read-only), your jobs (the GUI's shared own-jobs cache) and the control file
are read on each call. Opening the view keeps the Jobs poll going for a
while, and each poll that changes a step sends ``workflow.changed``.

The writing routes carry the page's root_gen (body or X-Mace-Root-Gen;
missing is 400, another generation 409 "The folder changed; reload"); the
run, Resume, Cancel, Recover and Process now are refused while the server
is open to the network (they change SLURM's state); Pause is not.
"""
from __future__ import annotations

import json
import secrets
from typing import Any

from . import submit as sub
from . import tasks as gui_tasks
from . import workflow as wf
from . import workflow_actions as wa
from . import workflow_build as wb

GET_ROUTES = {"workflow/list", "workflow/status", "workflow/templates", "workflow/clone"}
POST_ROUTES = {"workflow/inputs", "workflow/defaults", "workflow/step-form", "workflow/preview", "workflow/run",
               "workflow/control", "workflow/recover/preview", "workflow/recover/run", "workflow/process"}


def _q(query, name: str, default: Any = None):
    return (query.get(name) or [default])[0]


def _watching(st) -> None:
    """The view is open: keep the own-jobs poll (and so workflow.changed) going."""
    import time
    jobs = getattr(st, "jobs", None)
    if jobs is not None:
        jobs.visible_until = max(jobs.visible_until, time.time() + 2.5 * jobs.interval())
        try:
            jobs._ensure_thread()
        except Exception:
            pass
    watcher = getattr(st, "workflows", None)
    if watcher is not None:
        watcher.touch()


def get(handler, route: str, query):
    st = handler.state
    if route == "workflow/templates":
        return handler._json(wb.templates_info())
    _watching(st)
    squeue = wf.squeue_from_jobs(getattr(st, "jobs", None))
    if route == "workflow/list":
        out = wf.list_view(st.root, squeue)
        out["poll_seconds"] = st.jobs.interval() if getattr(st, "jobs", None) is not None else 60
        out["network"] = _network(st)
        return handler._json(out)
    if route == "workflow/status":
        wid = wf.check_id(_q(query, "id"))
        folder_rel = _q(query, "folder", "") or ""
        res = wf.status_view(st.root, folder_rel, wid, squeue, eta=_q(query, "eta", "1") != "0")
        if res is None:
            return handler._error(404, f"No plan of {wid} in {folder_rel or 'this folder'}/workflow_configs")
        watcher = getattr(st, "workflows", None)
        if watcher is not None:
            watcher.remember(f"{res['folder_rel']}::{wid}", res)
        res["poll_seconds"] = st.jobs.interval() if getattr(st, "jobs", None) is not None else 60
        res["network"] = _network(st)
        return handler._json(res)
    if route == "workflow/clone":
        folder, wid, plan = wa.target(st.root, _q(query, "folder", "") or "", _q(query, "id"))
        return handler._json(wa.clone(st.root, folder, wid, plan))
    return handler._error(404, f"unknown API route {route}")


# ------------------------------------------------------------------ the builder
def _gen(handler, body) -> Any:
    """The page's folder generation: required on a writing route."""
    want = body.get("root_gen", handler.headers.get("X-Mace-Root-Gen"))
    if want is None:
        raise wb.BuildError("root_gen is required (reload the page)")
    cur = getattr(handler.state, "root_gen", 0)
    if str(want) != str(cur):
        raise sub.SubmitConflict("The folder changed; reload")
    return cur


def _network(st) -> bool:
    from .server import is_loopback
    return not is_loopback(getattr(st, "bind_host", "127.0.0.1"))


def _wait(body) -> float:
    w = body.get("wait")
    try:
        return min(600.0, max(0.0, float(w))) if isinstance(w, (int, float)) else gui_tasks.WAIT_S
    except (TypeError, ValueError):
        return gui_tasks.WAIT_S


def _inputs(st, draft):
    if not isinstance(draft, dict):
        raise wb.BuildError("draft is an object")
    inp = draft.get("inputs") or {}
    if not isinstance(inp, dict):
        raise wb.BuildError("draft.inputs is {type, files}")
    kind = inp.get("type")
    if kind not in ("cif", "d12"):
        raise wb.BuildError("Pick the structures first (CIF or .d12 files)")
    return wb.scan_inputs(st, inp.get("files") or [], kind)


def _cache(st):
    def get_(k):
        return st.parse_cache.get(("setup",) + (json.dumps(k, default=str),))

    def put_(k, v, n):
        st.parse_cache.put(("setup",) + (json.dumps(k, default=str),), v, n)
        st.ledger.enforce()
    return get_, put_


def _job_in_scratch(st, prefix, fn):
    """``fn(work, ctx)`` in its own GUI scratch folder, removed at the end."""
    def job(ctx):
        key = f"{prefix}-{secrets.token_hex(6)}"
        work = st.disk.acquire(key)
        try:
            return fn(work, ctx)
        finally:
            st.disk.release(key, remove=True)
    return job


def post(handler, route: str, body):
    st = handler.state
    from .server import cli_task
    client = handler.headers.get("X-Mace-Client")
    doc = str(body.get("doc") or "wfbuilder")
    if route == "workflow/inputs":
        kind = body.get("type")
        if kind not in (None, "cif", "d12"):
            raise wb.BuildError("type is cif or d12")
        return handler._json(wb.inputs_view(st, wb.scan_inputs(st, body.get("targets"), kind)))
    if route == "workflow/defaults":
        draft = body.get("draft")
        inp = _inputs(st, draft)
        wb.spec_files(inp, draft, None)           # a draft to correct is a 400 now, not a failed task
        get_, put_ = _cache(st)

        def check(work, ctx):
            b = wb.Builder(st, inp, draft, work / "forms", ctx, get_, put_)
            ov = b.step1_overrides()
            return wb.check_spec(st.root, inp, draft, work, ctx, ov, b.later_overrides())
        job = _job_in_scratch(st, "wfcheck", check)
        t = cli_task(st, "workflow-check", "Workflow: resources and settings (mace workflow --spec)", job,
                     doc=doc, client=client, supersede=("workflow-check", doc))
        return gui_tasks.answer(handler, st.tasks, t, _wait(body))
    if route == "workflow/step-form":
        draft = body.get("draft")
        inp = _inputs(st, draft)
        seq, i = wb.step_request(draft, body.get("step"))
        wb.check_form_answers(inp, seq, i, body)
        wb.check_module_settings(draft, seq, i)
        get_, put_ = _cache(st)
        step = f"{seq[i]}_{i + 1}"
        job = _job_in_scratch(st, "wfform", lambda work, ctx: wb.step_form(st, inp, draft, seq, i, body, work, ctx,
                                                                           get_, put_))
        t = cli_task(st, "deck", f"Workflow step {step}: settings", job, doc=f"{doc}:{step}", client=client,
                     supersede=("deck", f"{doc}:{step}"))
        return gui_tasks.answer(handler, st.tasks, t, _wait(body))
    if route == "workflow/preview":
        gen = _gen(handler, body)
        draft = body.get("draft")
        inp = _inputs(st, draft)
        wb.spec_files(inp, draft, inp.folder / wb.SPEC_DIR / "x")
        network = _network(st)
        get_, put_ = _cache(st)
        ui = wa.ui_draft_file(body.get("ui_draft"))

        def preview(work, ctx):
            p, _res = wb.preview(st, st.root, gen, inp, draft, work, ctx, network, get_, put_, ui_draft=ui)
            out = wb.finalize(st.wf_previews, p)
            if st.test_hooks:              # the recorded job scripts, for the preview == run tests
                out["scripts"] = dict(p.scripts)
            out["network"] = network
            return out
        names = ", ".join(inp.materials[:3]) + ("…" if len(inp.materials) > 3 else "")
        t = cli_task(st, "submit-preview", f"Preview workflow: {names}", _job_in_scratch(st, "wfprev", preview),
                     doc=doc, client=client, supersede=("submit-preview", doc))
        return gui_tasks.answer(handler, st.tasks, t, _wait(body))
    if route == "workflow/run":
        gen = _gen(handler, body)
        if _network(st):
            raise sub.SubmitConflict("Submitting is turned off while MACE GUI is open to the network")
        pid = str(body.get("preview_id") or "")
        if not pid.startswith("wf-"):
            raise sub.SubmitConflict("No such preview; preview again")
        p = wb.check_run(st.wf_previews, pid, st.root, gen, {})
        holder = {}

        def run_job(ctx):
            def say(text):
                if "t" in holder:
                    holder["t"].log = text[-6000:]
                    st.tasks._publish(holder["t"])
            out = wb.run(st.submits, p, st.root, ctx, on_output=say)
            if out.get("job_ids"):
                st.jobs.own.get(fresh="force")       # the new jobs show at once
                st.jobs.kick()
            watcher = getattr(st, "workflows", None)
            if watcher is not None:
                watcher.touch()
            return out
        t = cli_task(st, "submit", f"Start workflow: {p.result.get('title') or ''}"[:200], run_job, client=client)
        holder["t"] = t
        return gui_tasks.answer(handler, st.tasks, t, gui_tasks.WAIT_S)
    return actions(handler, route, body, client)


def actions(handler, route: str, body, client):
    """The Workflows view's buttons: one CLI command each, as a task."""
    st = handler.state
    from .server import cli_task
    gen = _gen(handler, body)
    network = _network(st)
    if route == "workflow/recover/run":
        if network:
            raise sub.SubmitConflict(f"Recovering a step {wa.NETWORK}")
        p = wa.take_recover(st, body.get("preview_id"), gen)
        t = cli_task(st, "recover", "Workflow: mace recover", lambda ctx: wa.recover_run(
            st, p, bool(body.get("include_stale")), ctx), client=client)
        return gui_tasks.answer(handler, st.tasks, t, _wait(body))
    folder, wid, plan = wa.target(st.root, body.get("folder"), body.get("id"))
    if route == "workflow/control":
        action = body.get("action")
        if action not in wa.ACTIONS:
            raise wb.BuildError("action is pause, resume or cancel")
        if network and action != "pause":
            raise sub.SubmitConflict(f"{'Resuming' if action == 'resume' else 'Canceling'} a workflow {wa.NETWORK}")
        t = cli_task(st, "workflow-control", f"Workflow {action}: {wid}",
                     lambda ctx: wa.control(st, folder, wid, action, body, network, ctx), client=client)
        return gui_tasks.answer(handler, st.tasks, t, _wait(body))
    if route == "workflow/process":
        if network:
            raise sub.SubmitConflict(f"Starting a step {wa.NETWORK}")
        t = cli_task(st, "workflow-control", f"Workflow process: {wid}", lambda ctx: wa.process(st, folder, wid, ctx),
                     client=client)
        return gui_tasks.answer(handler, st.tasks, t, _wait(body))
    if route == "workflow/recover/preview":
        base = wa.recover_argv(folder, wid, plan, body.get("calc_id"))
        t = cli_task(st, "recover-preview", "Preview: mace recover (workflow step)",
                     lambda ctx: wa.recover_preview(st, folder, wid, base, gen, network, ctx),
                     doc=str(body.get("doc") or "workflows"), client=client,
                     supersede=("recover-preview", "workflows"))
        return gui_tasks.answer(handler, st.tasks, t, _wait(body))
    return handler._error(404, f"unknown API route {route}")


def handle(handler, method: str, route: str, payload):
    """Errors as answers: a draft to correct 400, a stale preview or folder
    409, a bad id 400, a path outside the folder 403, a missing folder 404."""
    try:
        if method == "GET" and route in GET_ROUTES:
            return get(handler, route, payload)
        if method == "POST" and route in POST_ROUTES:
            return post(handler, route, payload or {})
        return handler._error(404, f"unknown API route {route}")
    except sub.SubmitConflict as exc:
        return handler._json({"error": str(exc), "conflict": True, **exc.details}, 409)
    except PermissionError as exc:
        return handler._error(403, str(exc))
    except FileNotFoundError as exc:
        return handler._error(404, str(exc))
    except ValueError as exc:             # BuildError, SubmitError, SetupError, a bad id
        return handler._error(400, str(exc))
