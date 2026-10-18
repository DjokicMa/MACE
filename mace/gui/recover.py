"""Recover failed jobs from the Jobs panel: `mace recover`, the CLI itself
(GUI route == CLI route).

POST jobs/recover/preview  {root_gen, ids: [job ids]} or {root_gen, all: true}
    -> task: `mace recover ... --dry-run --json` in the served folder; the
       answer holds the dry run's text and plan, the argv it ran and the argv
       a run will use (``run_argv``), and a preview_id.
POST jobs/recover/run      {root_gen, preview_id, include_stale}
    -> task: the preview's ``run_argv`` (``... --yes --json``, plus
       ``--include-stale`` when asked); refused (409) while MACE GUI is open
       to the network, like Submit and Cancel.

A job id is turned into its tracked calculation through the folder's
databases (jobs.tracked_rows); the ids of one preview must share a database.
"Recover Failed…" scopes the preview to the served folder (``--dir .``); its
run names the calculations the preview offered (``--calc-id``), so a job
that failed after the preview is not resubmitted unseen.
"""
from __future__ import annotations

import json
import secrets
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PREVIEW_TTL_S = 1800
_LOCK = threading.Lock()


def _store(st) -> Dict[str, Dict[str, Any]]:
    with _LOCK:
        if not hasattr(st, "recover_previews"):
            st.recover_previews = {}
        now = time.time()
        for k in [k for k, v in st.recover_previews.items() if now - v["at"] > PREVIEW_TTL_S]:
            st.recover_previews.pop(k, None)
        return st.recover_previews


def json_tail(stdout: str) -> Dict[str, Any]:
    """The JSON document `mace recover --json` prints last (the banner and
    any start-up line may come before it)."""
    i = stdout.rfind("\n{\n")
    text = stdout[i + 1:] if i >= 0 else stdout[stdout.find("{"):] if "{" in stdout else stdout
    return json.loads(text)


def remember_preview(store: Dict[str, Dict[str, Any]], prefix: str, gen: Any, doc: Dict[str, Any], argv: List[str],
                     run_argv: List[str], network: bool, **keep: Any) -> Dict[str, Any]:
    """Keep a preview (single use: its run takes it) and answer it: the dry
    run's text and entries, the argv it ran and the one a run uses, how many
    are ready and how many ended too long ago. The Jobs panel's previews and a
    workflow step's (workflow_actions.py) answer the same way."""
    entries = doc.get("entries") or []
    stale = sum(1 for e in entries if e.get("state") == "stale")
    pid = prefix + secrets.token_hex(8)
    store[pid] = {"at": time.time(), "gen": gen, "run_argv": run_argv, "stale": stale, **keep}
    return {"preview_id": pid, "argv": ["mace", *argv], "run_argv": ["mace", *run_argv], "text": doc.get("text", ""),
            "entries": entries, "ready": sum(1 for e in entries if e.get("state") == "ready"), "stale": stale,
            "stale_days": doc.get("stale_days"), "network": network}


def pinned(base: List[str], doc: Dict[str, Any]) -> List[str]:
    """``base`` with a ``--calc-id`` for each calculation the preview offered
    (ready or stale) when it does not name its calculations already; [] when
    it offered none."""
    if "--calc-id" in base:
        return list(base)
    ids = [str(e.get("calc_id")) for e in (doc.get("entries") or [])
           if e.get("state") in ("ready", "stale") and e.get("calc_id")]
    if not ids:
        return []
    out = list(base)
    for cid in dict.fromkeys(ids):
        out += ["--calc-id", cid]
    return out


def run_argv_of(p: Dict[str, Any], include_stale: bool) -> List[str]:
    """A preview's run argv, with --include-stale when asked and it had old jobs."""
    argv = list(p["run_argv"])
    if include_stale and p["stale"]:
        argv.append("--include-stale")
    return argv


def base_argv(root: Path, ids: Optional[List[str]], everything: bool) -> List[str]:
    """`mace recover` arguments (without the mode flags) for these job ids,
    or for every failed calculation in the folder."""
    from . import jobs
    from .submit import SubmitError
    if everything:
        return ["recover", "--dir", "."]
    ids = [str(i) for i in (ids or [])]
    if not ids or len(ids) > 500 or not all(jobs.ID_RE.match(i) for i in ids):
        raise SubmitError("ids is a list of job ids")
    workdirs = [str(root / (r.get("workdir_rel") or "")) for r in jobs.recent_finished(root)
                if r.get("id") in ids and r.get("workdir_rel") is not None]
    tracked = jobs.tracked_rows(root, workdirs)
    missing = [i for i in ids if i not in tracked]
    if missing:
        raise SubmitError(f"Not tracked in this folder's databases: {', '.join(missing)} (only calculations "
                          "MACE tracks can be recovered)")
    dbs = {tracked[i]["db"] for i in ids}
    if len(dbs) != 1:
        raise SubmitError("These jobs are in different databases: recover them one database at a time")
    db = Path(dbs.pop())
    try:
        rel = db.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        rel = str(db)
    argv = ["recover", "--db", rel]
    for i in ids:
        argv += ["--calc-id", tracked[i]["calc_id"]]
    return argv


def post(handler, route: str, body: Dict[str, Any], gen_check, network: bool):
    from . import submit as sub
    from . import tasks as gui_tasks
    from .server import cli_task
    st = handler.state
    if route == "jobs/recover/preview":
        gen = gen_check(handler, body)
        base = base_argv(st.root, body.get("ids"), bool(body.get("all")))
        argv = base + ["--dry-run", "--json"]

        def job(ctx):
            from . import clirun
            r = clirun.run(sub.mace_argv(*argv), st.root, ctx, timeout=900, env=sub.submit_env())
            try:
                doc = json_tail(r.stdout)
            except ValueError:
                raise sub.SubmitError("mace recover gave no plan: " + (r.stderr or r.stdout).strip()[-600:])
            # the run resubmits what this preview offered, nothing found later
            pins = pinned(base, doc)
            run_argv = (pins or base) + ["--yes", "--json"]
            return remember_preview(_store(st), "rec-", gen, doc, argv, run_argv, network, empty=not pins)
        t = cli_task(st, "recover-preview", "Preview: mace recover", job, doc=str(body.get("doc") or "jobs"),
                     client=handler.headers.get("X-Mace-Client"), supersede=("recover-preview", "jobs"))
        wait = body.get("wait")
        return gui_tasks.answer(handler, st.tasks, t, float(wait) if isinstance(wait, (int, float))
                                else gui_tasks.WAIT_S)
    if route == "jobs/recover/run":
        gen = gen_check(handler, body)
        if network:                   # sbatch changes SLURM's state
            raise sub.SubmitConflict("Recovering jobs is turned off while MACE GUI is open to the network")
        p = _store(st).pop(str(body.get("preview_id") or ""), None)
        if p is None or str(p["gen"]) != str(gen):
            raise sub.SubmitConflict("This preview is no longer current: preview again")
        if p.get("empty"):
            raise sub.SubmitConflict("This preview offered nothing to resubmit")
        argv = run_argv_of(p, bool(body.get("include_stale")))

        def job(ctx):
            from . import clirun
            r = clirun.run(sub.mace_argv(*argv), st.root, ctx, timeout=None, env=sub.submit_env(),
                           cancelable=False)
            try:
                doc = json_tail(r.stdout)
            except ValueError:
                doc = {"text": r.stdout + r.stderr, "results": []}
            res = doc.get("results", [])
            if any(x.get("status") == "submitted" for x in res):
                st.jobs.own.get(fresh="force")
                st.jobs.kick()
            return {"argv": ["mace", *argv], "text": doc.get("text", ""), "results": res, "exit": r.returncode}
        t = cli_task(st, "recover", "mace recover", job, client=handler.headers.get("X-Mace-Client"))
        wait = body.get("wait")
        return gui_tasks.answer(handler, st.tasks, t, float(wait) if isinstance(wait, (int, float))
                                else gui_tasks.WAIT_S)
    return None
