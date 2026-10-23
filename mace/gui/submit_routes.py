"""The Submit, Jobs and Batch Setup routes (PLAN_submit section 2).

GET  submit/capabilities          SLURM tools here, the site settings, the GUI's own job
GET  submit/suggest?path=..&type=  the size-aware job size of each deck (mace site suggest)
GET  jobs?scope=folder|all&refresh=1
GET  jobs/recent                  own jobs that ended in the last two days (sacct)
POST submit/preview               {root_gen, targets, where, options, mixed_choice} -> task
POST submit/run                   {root_gen, preview_id, confirm_*} -> task
POST jobs/cancel                  {root_gen, ids}
POST jobs/release                 {root_gen, ids}
POST jobs/recover/preview         {root_gen, ids | all} -> task (mace recover --dry-run, recover.py)
POST jobs/recover/run             {root_gen, preview_id, include_stale} -> task (mace recover --yes)
POST batch/scan | batch/plan | batch/write, GET batch/deck | batch/diff (batch.py)

Every writing POST carries the page's root_gen (body or X-Mace-Root-Gen);
another generation is 409 "The folder changed; reload".
"""
from __future__ import annotations

import json
from pathlib import Path
import secrets
import time
from typing import Any, Dict

from . import submit as sub
from . import tasks as gui_tasks


def _gen(handler, body: Dict[str, Any]):
    st = handler.state
    want = body.get("root_gen", handler.headers.get("X-Mace-Root-Gen"))
    cur = getattr(st, "root_gen", 0)
    if want is not None and str(want) != str(cur):
        raise sub.SubmitConflict("The folder changed; reload")
    return cur


def _network(st) -> bool:
    from .server import is_loopback
    host = getattr(st, "bind_host", "127.0.0.1")
    return not is_loopback(host)


_SITE_CACHE: Dict[str, Any] = {}


def site_describe() -> Dict[str, Any]:
    """`mace site --json` for the dialog's Site row (cached a minute; the
    module probe itself is cached by envmodules)."""
    now = time.time()
    env = sub.submit_env()
    key = json.dumps(sub.site_inputs(env), sort_keys=True, default=str)
    if _SITE_CACHE.get("t", 0) > now - 60 and _SITE_CACHE.get("key") == key:
        return _SITE_CACHE["v"]
    from mace.submission import site
    try:
        v = site.describe(env, check_account=True)
    except Exception as exc:          # pragma: no cover - the dialog still works
        v = {"errors": [str(exc)]}
    _SITE_CACHE.update(t=now, v=v, key=key)
    return v


def get(handler, route: str, query):
    st = handler.state
    q = lambda k, d=None: (query.get(k) or [d])[0]
    if route == "submit/capabilities":
        caps = sub.capabilities(_network(st))
        caps["site"] = site_describe()
        caps["poll_seconds"] = st.jobs.interval()
        return handler._json(caps)
    if route == "submit/suggest":
        # `mace site suggest --json` through the same function (site.suggest_decks)
        paths = query.get("path") or []
        if not paths or len(paths) > 500:
            raise sub.SubmitError("path names 1 to 500 decks")
        decks = [st.resolve(str(x)) for x in paths]
        for d in decks:
            if not d.is_file() or d.suffix not in sub.DECK_SUFFIXES:
                raise sub.SubmitError(f"{st.rel(d)} is not a deck (.d12 or .d3)")
        return handler._json({"suggestions": sub.suggestions(st.root, decks, calc_type=q("type"))})
    if route == "jobs":
        scope = q("scope", "folder")
        if scope not in ("folder", "all"):
            raise ValueError("scope is folder or all")
        snap = st.jobs.snapshot(scope, refresh=q("refresh") == "1", visible=q("visible", "1") == "1")
        return handler._json(snap)
    if route == "jobs/recent":
        from . import jobs
        return handler._json({"jobs": jobs.recent_with_tracking(st.root)})
    if route in ("batch/deck", "batch/diff"):
        from . import batch
        return batch.get_route(handler, route, query)
    return None


def post(handler, route: str, body: Dict[str, Any]):
    st = handler.state
    if route == "submit/preview":
        gen = _gen(handler, body)
        where = str(body.get("where") or "slurm")
        targets = body.get("targets") or []
        if not isinstance(targets, list) or not targets or len(targets) > 5000:
            raise sub.SubmitError("targets is a list of files or one folder")
        paths = [st.resolve(str(t)) for t in targets]
        sel = sub.selection(st.root, paths)
        if sel.mixed and body.get("mixed_choice") is None:
            return handler._json({"error": "This folder has D12 and D3 decks: choose which to submit",
                                  "mixed": True, "decks": [st.rel(d) for d in sel.decks]}, 400)
        options = body.get("options") or {}
        if not isinstance(options, dict):
            raise sub.SubmitError("options is an object")
        mixed = body.get("mixed_choice")
        # the argv is checked now (a 400 at once, not inside the task)
        sub.build_commands(st.root, where, sel, options, mixed)
        own = None
        if where == "slurm" and st.jobs.own.available():
            # a fresh squeue per preview (one per user action, as the plan's X7 allows); the run checks again
            own, _ = st.jobs.own.get(fresh="force")
        network = _network(st)
        doc = str(body.get("doc") or "submit")

        def job(ctx):
            key = "submit-" + secrets.token_hex(6)
            work = st.disk.acquire(key)
            try:
                p, res = sub.preview(st.root, gen, where, sel, options, mixed, work, ctx, own, network, st.submits)
            finally:
                st.disk.release(key, remove=True)
            acct = (site_describe().get("account_check") if where == "slurm" else None)
            out = sub.finalize(st.submits, p, acct)
            if st.test_hooks:              # the recorded job scripts, for the preview == submit tests
                out["scripts"] = dict(p.scripts)
            out["site"] = site_describe() if where == "slurm" else None
            out["network"] = network
            return out
        from .server import cli_task
        t = cli_task(st, "submit-preview", f"Preview: {'mace run' if where == 'local' else 'mace submit'} "
                     f"{', '.join(str(x) for x in targets[:3])}{'…' if len(targets) > 3 else ''}", job,
                     doc=doc, client=handler.headers.get("X-Mace-Client"), supersede=("submit-preview", doc))
        return gui_tasks.answer(handler, st.tasks, t, float(body.get("wait", gui_tasks.WAIT_S))
                                if isinstance(body.get("wait"), (int, float)) else gui_tasks.WAIT_S)
    if route == "submit/site-check":
        # the Site row's Check button: `mace site check --json`, the CLI itself
        # (GUI route == CLI route), with the environment a submit gets
        kind = str(body.get("kind") or "both")
        if kind not in ("scf", "prop", "both"):
            raise ValueError("kind is scf, prop or both")
        argv = ["site", "check", "--json", "--kind", kind]
        if body.get("no_sbatch"):
            argv.append("--no-sbatch")

        def check_job(ctx):
            from . import clirun
            r = clirun.run(sub.mace_argv(*argv), st.root, ctx, timeout=300, env=sub.submit_env())
            try:
                report = json.loads(r.stdout)
            except ValueError:
                raise sub.SubmitError("mace site check gave no report: " + (r.stderr or r.stdout).strip()[-400:])
            if report.get("kept"):          # never with sbatch; the GUI keeps nothing
                drop_check_folder(report["kept"])
                report["kept"] = None
            _SITE_CACHE.clear()             # the row shows what the check just saw
            return {"report": report, "argv": ["mace", *argv]}
        from .server import cli_task
        t = cli_task(st, "site-check", "Check site settings (mace site check)", check_job,
                     doc=str(body.get("doc") or "submit"), client=handler.headers.get("X-Mace-Client"),
                     supersede=("site-check", "submit"))
        return gui_tasks.answer(handler, st.tasks, t, float(body.get("wait", gui_tasks.WAIT_S))
                                if isinstance(body.get("wait"), (int, float)) else gui_tasks.WAIT_S)
    if route == "submit/run":
        gen = _gen(handler, body)
        if _network(st):
            raise sub.SubmitConflict("Submitting is turned off while MACE GUI is open to the network")
        own = None
        if st.jobs.own.available():
            own, _ = st.jobs.own.get(fresh="force")
        p = sub.check_run(st.submits, str(body.get("preview_id") or ""), st.root, gen,
                          {k: bool(body.get(k)) for k in ("confirm_overwrite", "confirm_duplicate", "confirm_account",
                                                         "confirm_limits")}, own)
        problems = []
        local = p.where == "local"
        title = (f"Run on This Node: {', '.join(d.name for d in p.decks[:3])}" if local
                 else f"Submit: {p.result.get('command', '')}")[:200]
        holder = {}

        def job(ctx):
            def say(text):
                if "t" in holder:
                    holder["t"].log = text[-6000:]
                    st.tasks._publish(holder["t"])
            out = sub.run(st.submits, p, st.root, ctx, problems=problems.append, on_output=say)
            out["problems"] = problems
            out["local"] = local
            out["command"] = p.result.get("command")
            if out.get("job_ids"):
                st.jobs.own.get(fresh="force")       # the new jobs show at once
                st.jobs.kick()
                tgt = p.cmds[0].argv[-1] if p.cmds else ""
                if tgt.endswith("/decks") and (p.cmds[0].cwd / tgt).parent.joinpath("batch.json").is_file():
                    from . import batch
                    batch.record_jobs(st.root, str((p.cmds[0].cwd / tgt).relative_to(st.root)), out["job_ids"])
            return out
        from .server import cli_task
        t = cli_task(st, "local" if local else "submit", title, job, client=handler.headers.get("X-Mace-Client"))
        holder["t"] = t
        return gui_tasks.answer(handler, st.tasks, t, gui_tasks.WAIT_S)
    if route == "jobs/cancel":
        _gen(handler, body)
        if _network(st):              # scancel changes SLURM's state as sbatch does
            raise sub.SubmitConflict("Canceling jobs is turned off while MACE GUI is open to the network")
        ids = body.get("ids") or []
        if not isinstance(ids, list) or not ids or len(ids) > 2000:
            raise sub.SubmitError("ids is a list of job ids")
        from . import jobs
        res = jobs.cancel(st.root, [str(i) for i in ids])
        if res["canceled"]:
            st.jobs.own.get(fresh="force")
        return handler._json(res)
    if route == "jobs/release":
        _gen(handler, body)
        if _network(st):              # scontrol release changes SLURM's state as scancel does
            raise sub.SubmitConflict("Releasing jobs is turned off while MACE GUI is open to the network")
        ids = body.get("ids") or []
        if not isinstance(ids, list) or not ids or len(ids) > 2000:
            raise sub.SubmitError("ids is a list of job ids")
        from . import jobs
        res = jobs.release(st.root, [str(i) for i in ids])
        if res["released"]:
            st.jobs.own.get(fresh="force")
        return handler._json(res)
    if route.startswith("jobs/recover/"):
        from . import recover
        return recover.post(handler, route, body, _gen, _network(st))
    if route.startswith("batch/"):
        from . import batch
        if route not in ("batch/scan", "batch/stutsc"):          # read-only: no folder generation needed
            _gen(handler, body)
        return batch.post_route(handler, route, body)
    return None


def drop_check_folder(kept) -> bool:
    """Remove the folder `mace site check` kept (no SLURM here): only its own
    temporary folder (tempfile.mkdtemp "mace-site-check-..."), never another
    path the report might name. True when it was removed."""
    p = Path(str(kept or ""))
    if not (p.is_absolute() and p.name.startswith("mace-site-check-") and p.is_dir() and not p.is_symlink()):
        return False
    import shutil
    shutil.rmtree(str(p), ignore_errors=True)
    return not p.exists()


GET_ROUTES = {"submit/capabilities", "submit/suggest", "jobs", "jobs/recent", "batch/deck", "batch/diff"}
POST_ROUTES = {"submit/preview", "submit/run", "submit/site-check", "jobs/cancel", "jobs/release", "jobs/recover/preview",
               "jobs/recover/run", "batch/scan", "batch/plan", "batch/write", "batch/stutsc"}
# Settings › Advanced: cluster defaults (site_routes.py: mace site, set/unset, --init)
GET_ROUTES |= {"site/settings"}
POST_ROUTES |= {"site/edit"}


def handle(handler, method: str, route: str, payload):
    """Run a route with this module's errors as answers: SubmitError 400,
    SubmitConflict 409 (with what is missing)."""
    try:
        fn = get if method == "GET" else post
        if route.startswith("site/"):
            from . import site_routes
            fn = site_routes.get if method == "GET" else site_routes.post
        return fn(handler, route, payload)
    except sub.SubmitConflict as exc:
        return handler._json({"error": str(exc), "conflict": True, **exc.details}, 409)
    except sub.SubmitError as exc:
        return handler._error(400, str(exc))
