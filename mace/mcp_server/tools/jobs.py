"""Job tools: submit decks to SLURM, show and cancel your jobs, workflows,
and the cluster's state.

``job_submit`` and ``job_cancel`` do nothing without ``"confirm": true``;
without it they answer with exactly what they would do. They go through the
same code as the GUI: ``mace.gui.submit.selection`` picks the CLI targets,
the real ``mace submit`` submits (in the login environment, no cancel half
way), ``mace.gui.jobs.cancel`` checks every id (yours, queued, not a whole
array, not the GUI's own job) before ``scancel``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

CLUSTER_VIEWS = ["overview", "nodes", "quota", "limits", "reservations"]


def register(reg) -> None:
    reg.tool("job_submit", (
        "Submit .d12/.d3 decks (files of one folder, or one folder) to SLURM with `mace submit`. Without "
        "confirm it only reports the decks, the command and what would be overwritten. Run deck_preflight "
        "first. track records the jobs in materials.db; progress plans the follow-up steps (a workflow "
        "template, implies track)."),
        {"properties": {"paths": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 500},
                        "track": {"type": "boolean", "default": False},
                        "progress": {"type": "string", "description": "workflow template for the next steps"},
                        "walltime": {"type": "string", "description": "e.g. 2:00:00 or 7-00:00:00"},
                        "array": {"type": "boolean", "default": False,
                                  "description": "submit decks with equal resources as one SLURM job array"},
                        "max_running": {"type": "integer", "minimum": 1, "maximum": 10000},
                        "overwrite_sh": {"type": "boolean", "default": False,
                                         "description": "write the job scripts again from the template"},
                        "exclude_nodes": {"type": "string", "description": 'e.g. "node-042,node-050"'},
                        "use_reserve": {"type": "boolean", "default": False,
                                        "description": "untracked: submit beyond the queue budget from the reserve"}},
         "required": ["paths"]},
        title="Submit decks", group="jobs", writes=True, confirm="always", open_world=True,
        previews=True)(job_submit)

    reg.tool("job_status", (
        "Your SLURM jobs (squeue: id, name, state, elapsed, limit, nodes, cpus, reason, folder), or one job in "
        "detail (state, progress, ETA, walltime risk) with job_id."),
        {"properties": {"job_id": {"type": "string", "description": "N or N_T for one job's detail"}}},
        title="Job status", group="jobs", open_world=True)(job_status)

    reg.tool("job_cancel", (
        "Cancel your queued or running SLURM jobs by id (scancel). Each id is checked first: it must be yours "
        "and queued; a whole job array is never canceled (cancel its tasks). Needs confirm."),
        {"properties": {"job_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 200}},
         "required": ["job_ids"]},
        title="Cancel jobs", group="jobs", writes=True, confirm="always", open_world=True,
        previews=True)(job_cancel)

    reg.tool("workflow_status", (
        "MACE workflows in the served folder (`mace workflow --list`), or one workflow's steps per material "
        "with workflow_id (planned, queued, running, completed, failed, ...)."),
        {"properties": {"workflow_id": {"type": "string"}}},
        title="Workflow status", group="jobs")(workflow_status)

    reg.tool("cluster_info", (
        "The cluster from SLURM (`mace cluster VIEW --json`): overview (partitions, your use vs limits), nodes "
        "(free cores, memory, GPUs), quota (storage and file counts), limits (account/QOS, fairshare), "
        "reservations."),
        {"properties": {"view": {"type": "string", "enum": CLUSTER_VIEWS, "default": "overview"},
                        "partition": {"type": "string", "description": "nodes: only this partition"}}},
        title="Cluster", group="jobs", open_world=True)(cluster_info)


# ---------------------------------------------------------------- submit
def job_submit(ctx, args):
    from mace.gui import submit as gsub
    paths = [ctx.path(x) for x in args["paths"]]
    try:
        sel = gsub.selection(ctx.root if ctx.inside(paths[0]) else paths[0].parent, paths)
    except Exception as exc:
        raise ValueError(str(exc))
    if sel.mixed:
        raise ValueError("this folder has .d12 and .d3 decks: pass the decks to submit as files")
    flags: List[str] = []
    if args.get("track"):
        flags.append("--track")
    if args.get("progress"):
        flags += ["--progress", str(args["progress"])]
    if args.get("walltime"):
        flags += ["--walltime", str(args["walltime"])]
    if args.get("array"):
        if not sel.folder:
            raise ValueError("job arrays are submitted from a folder: pass the folder")
        flags.append("--array")
        if args.get("max_running"):
            flags += ["--max-running", str(int(args["max_running"]))]
    elif args.get("max_running"):
        raise ValueError("max_running goes with array")
    if args.get("overwrite_sh"):
        flags.append("--overwrite-sh")
    if args.get("exclude_nodes"):
        flags += ["--exclude-nodes", str(args["exclude_nodes"])]
    if args.get("use_reserve"):
        flags.append("--yes")
    commands = [["submit", t, *flags] for t in sel.targets]
    outs = [d.with_suffix(".out") for d in sel.decks if d.with_suffix(".out").exists()]
    scripts = [d.with_suffix(".sh") for d in sel.decks if d.with_suffix(".sh").exists()]
    sbatch = gsub.which("sbatch")
    preview = {"folder": ctx.show(sel.cwd), "decks": [ctx.show(d) for d in sel.decks],
               "commands": ["mace " + " ".join(c) for c in commands],
               "existing_outputs_overwritten": [ctx.show(o) for o in outs],
               "existing_job_scripts": [ctx.show(s) for s in scripts] if not args.get("overwrite_sh") else [],
               "slurm_available": bool(sbatch)}
    action = f"submit {len(sel.decks)} deck{'s' if len(sel.decks) != 1 else ''} to SLURM"
    if outs:
        action += f" (overwriting {len(outs)} existing output{'s' if len(outs) != 1 else ''})"
    if not ctx.confirmed:
        ctx.require_confirm(action, preview)
    if ctx.config.read_only:
        raise ValueError("the server runs with --read-only")
    if not sbatch:
        raise ValueError("no SLURM here (sbatch not found): submit from a cluster login or development node")
    from mace.gui import clirun, history
    if outs:
        if history.root() is None:
            history.configure(ctx.root)
        history.snapshot(outs, "Submit (mace mcp) overwrote this result", route="mcp", kind="output")
    env = gsub.submit_env()
    results = []
    for argv in commands:
        r = clirun.run(gsub.mace_argv(*argv), sel.cwd, ctx=None, timeout=1800, env=env, cancelable=False)
        results.append({"command": "mace " + " ".join(argv), "exit_code": r.returncode,
                        "job_ids": _job_ids(r.log), "log": r.log[-6000:]})
    ok = all(x["exit_code"] == 0 for x in results)
    return {"submitted": ok, "results": results, **preview,
            "next": "job_status (and output_summary once a run writes its .out)"}


def _job_ids(log: str) -> List[str]:
    import re
    return sorted(set(re.findall(r"Submitted batch job (\d+)", log or "")), key=int)


# ---------------------------------------------------------------- status / cancel
def job_status(ctx, args):
    if args.get("job_id"):
        r = ctx.run_cli(["cluster", "job", str(args["job_id"]), "--json"], timeout=120)
        if r["code"] == 3:
            return {"slurm": False, "note": (r["stderr"] or r["stdout"]).strip()}
        if r["code"] == 4:
            raise ValueError((r["stderr"] or r["stdout"]).strip() or f"job {args['job_id']} not found or not yours")
        if r["code"] != 0:
            raise ValueError((r["stderr"] or r["stdout"]).strip()[-3000:])
        return json.loads(r["stdout"])
    from mace.gui import jobs as gjobs, submit as gsub
    import subprocess
    env = gsub.submit_env()
    squeue = gsub.which("squeue", env)
    if not squeue:
        return {"slurm": False, "jobs": [], "note": "no SLURM on this machine (squeue not found)"}
    try:
        r = subprocess.run([squeue, "-h", "-r", "-u", gjobs.user(), "-o", gjobs.SQUEUE_FORMAT], env=env,
                           stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(f"squeue failed: {exc}")
    if r.returncode != 0:
        raise ValueError(f"squeue failed: {r.stderr.strip()[:500]}")
    rows = gjobs.parse_squeue(r.stdout)
    for row in rows:
        if row.get("workdir"):
            row["workdir"] = ctx.show(row["workdir"]) if ctx.inside(Path(row["workdir"])) else row["workdir"]
    states: Dict[str, int] = {}
    for row in rows:
        states[row.get("state", "")] = states.get(row.get("state", ""), 0) + 1
    return {"slurm": True, "user": gjobs.user(), "count": len(rows), "states": states, "jobs": rows}


def job_cancel(ctx, args):
    ids = [str(x).strip() for x in args["job_ids"]]
    if not ctx.confirmed:
        ctx.require_confirm(f"cancel job{'s' if len(ids) != 1 else ''} {', '.join(ids)} (scancel)",
                            {"job_ids": ids, "checks": "each id must be yours and queued; arrays by task only"})
    if ctx.config.read_only:
        raise ValueError("the server runs with --read-only")
    from mace.gui import jobs as gjobs
    return gjobs.cancel(ctx.root, ids)


# ---------------------------------------------------------------- workflows / cluster
def workflow_status(ctx, args):
    wid = args.get("workflow_id")
    argv = ["workflow", "--status", str(wid), "--json"] if wid else ["workflow", "--list", "--json"]
    r = ctx.run_cli(argv, cwd=ctx.root, timeout=180)
    try:
        data = json.loads(r["stdout"])
    except json.JSONDecodeError:
        raise ValueError((r["stderr"] or r["stdout"]).strip()[-3000:] or "mace workflow gave no JSON")
    if r["code"] != 0 and isinstance(data, dict) and not data.get("error"):
        data["exit_code"] = r["code"]
    return data


def cluster_info(ctx, args):
    view = args.get("view") or "overview"
    argv = ["cluster", view, "--json"]
    if args.get("partition"):
        if view != "nodes":
            raise ValueError("partition goes with view nodes")
        argv += ["--partition", str(args["partition"])]
    r = ctx.run_cli(argv, timeout=180)
    if r["code"] == 3:
        return {"slurm": False, "note": (r["stderr"] or r["stdout"]).strip()}
    try:
        return json.loads(r["stdout"])
    except json.JSONDecodeError:
        raise ValueError((r["stderr"] or r["stdout"]).strip()[-3000:] or f"mace cluster {view} gave no JSON")
