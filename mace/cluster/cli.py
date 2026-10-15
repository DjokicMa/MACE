"""`mace cluster`: cluster status, limits, quotas and job progress (SLURM).

Standard library only; read-only; every SLURM query cached and rate-limited
(see mace/cluster/query.py). Exit codes: 0 ok (warnings included), 2 bad
arguments, 3 SLURM missing, 4 job not found or not yours, 5 a file scan is
already running.
"""
from __future__ import annotations

import argparse
import shutil
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from mace.cluster import job_efficiency as job_eff

from mace.cluster import render as R

EXIT_OK, EXIT_ARGS, EXIT_NO_SLURM, EXIT_NOT_FOUND, EXIT_BUSY = 0, 2, 3, 4, 5
NEEDS_SLURM = {None, "overview", "nodes", "fit", "reservations", "limits", "job"}


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", help="print JSON instead of tables")
    common.add_argument("--refresh", action="store_true",
                        help="ask the cluster again now (at most once per 15 s per query; the cache is shared)")
    ap = argparse.ArgumentParser(prog="mace cluster", parents=[common],
                                 description="Cluster status, limits, quotas and job progress (SLURM). Read-only.")
    sub = ap.add_subparsers(dest="cmd", metavar="SUBCOMMAND")
    sub.add_parser("overview", parents=[common], help="usable partitions, your use vs limits, quota warnings, "
                                                      "reservations soon (the default)")
    n = sub.add_parser("nodes", parents=[common], help="nodes by CPU type or partition: free cores, memory, GPUs")
    n.add_argument("--partition", metavar="P", help="only nodes of this partition")
    n.add_argument("--feature", metavar="F", help="only nodes with this feature")
    n.add_argument("--free", action="store_true", help="only nodes with free cores")
    n.add_argument("--all-partitions", action="store_true",
                   help="include partitions your accounts cannot use (marked)")
    n.add_argument("--group-by", choices=("feature", "partition"), default="feature",
                   help="group rows by CPU type (feature, the default) or partition")
    f = sub.add_parser("fit", parents=[common], help="which nodes could take a request now; --predict asks SLURM "
                                                     "when it would start")
    f.add_argument("--cpus", type=int, metavar="N", help="cores (tasks) in total")
    f.add_argument("--nodes", type=int, default=1, metavar="N", help="nodes (default 1)")
    g = f.add_mutually_exclusive_group()
    g.add_argument("--mem-per-cpu", metavar="SIZE", help="memory per core, e.g. 4G")
    g.add_argument("--mem", metavar="SIZE", help="memory per node, e.g. 128G")
    f.add_argument("--gpus", metavar="[TYPE:]N", help="GPUs per node, e.g. a100:2")
    f.add_argument("--constraint", metavar="EXPR", help="node features, e.g. 'amd24|intel18'")
    f.add_argument("--partition", metavar="P", help="only this partition")
    f.add_argument("--time", metavar="D-HH:MM:SS", help="walltime")
    f.add_argument("--predict", action="store_true",
                   help="also run sbatch --test-only (nothing is submitted) for the predicted start")
    f.add_argument("--script", metavar="JOB.sh", help="predict an existing job script instead of a request")
    sub.add_parser("reservations", parents=[common], help="reservations and maintenance windows")
    q = sub.add_parser("quota", parents=[common], help="storage and file-count quotas (80/90 %% warnings)")
    q.add_argument("--rediscover", action="store_true", help="look up the site's quota alias again now")
    fl = sub.add_parser("files", parents=[common], help="where are my files: a throttled scan with cleanup "
                                                       "suggestions (never deletes)")
    fl.add_argument("path", nargs="?", default=".", help="folder to scan (default: .)")
    fl.add_argument("--top", type=int, default=20, metavar="N", help="folders to list (default 20)")
    fl.add_argument("--max-seconds", type=float, metavar="S", help="stop after S seconds (default from site.toml, 600)")
    fl.add_argument("--sizes", choices=("top", "all", "off"),
                    help="read sizes in the busiest folders only (top, default), everywhere (all) or not at all")
    li = sub.add_parser("limits", parents=[common], help="account/QOS limits, your use and fairshare")
    li.add_argument("--check-submit", type=int, metavar="N", help="warn about adding N jobs")
    li.add_argument("--cpus", type=int, metavar="C", default=1, help="cores per job for --check-submit")
    j = sub.add_parser("job", parents=[common], help="one of your jobs: detail, progress, ETA, walltime risk")
    j.add_argument("id", help="job id (N or N_T)")
    j.add_argument("--follow", action="store_true", help="refresh every 60 s from the shared cache; Ctrl+C stops")
    j.add_argument("--live", action="store_true", help="also ask sstat for live memory and CPU (one call)")
    e = sub.add_parser("estimate", parents=[common], help="runtime estimate for a deck from similar finished runs")
    e.add_argument("deck", help="the .d12 deck")
    e.add_argument("--cores", type=int, metavar="N", help="cores (default: the deck's .sh --ntasks)")
    e.add_argument("--history", metavar="DIR", help="folder of finished runs (default: the deck's folder)")
    e.add_argument("--db", metavar="materials.db", help="a materials.db to include (read-only)")
    d = sub.add_parser("devnodes", parents=[common], help="the least-loaded configured development node")
    d.add_argument("--pick", action="store_true", help="print only the chosen host name")
    return ap


def _out(args, data, text_fn):
    if args.json:
        print(json.dumps(data, indent=1, default=str))
    else:
        text = text_fn(data)
        if text:
            print(text)


def _meta_lines(data) -> List[str]:
    out = []
    for e in data.get("errors") or []:
        out.append(e)
    for n in data.get("notes") or []:
        out.append(n)
    if data.get("stale"):
        out.append(f"(showing data from {R.age(data.get('age'))}; the cluster did not answer)")
    if data.get("next_allowed_in"):
        out.append(f"Updated {int(data.get('age') or 0)} s ago; the next refresh is allowed in "
                   f"{int(data['next_allowed_in'] + 0.999)} s")
    return out


# ------------------------------------------------------------------ texts
def nodes_text(v) -> str:
    head = (f"Cluster {v['cluster']} · updated {R.age(v.get('age'))} · usable partitions: "
            f"{', '.join(v['usable_partitions']) or '(unknown)'}")
    label = "CPU type" if v["group_by"] == "feature" else "Partition"
    rows = [[r["group"], r["nodes"], r["idle"], r["mixed"], r["allocated"], r["down_drain"],
             f"{r['cpus_free']:,}/{r['cpus_total']:,}", R.gib(r["mem_free_gb"]), R.gpus(r["gpus_free"], r["gpus_total"])]
            for r in v["rows"]]
    tbl = R.table([label, "Nodes", "Idle", "Mixed", "Alloc", "Down/Drain", "Free cores", "Free mem", "Free GPUs"],
                  rows, right=(1, 2, 3, 4, 5, 6, 7))
    extra = []
    if v.get("all_partitions"):
        for p in v["partitions"]:
            if not p["usable"]:
                extra.append(f"  {p['name']}: {p['why_not']}")
        if extra:
            extra.insert(0, "Partitions you cannot use:")
    return "\n".join([head, tbl, f"{v['note']} (mace cluster fit --predict)"] + extra + _meta_lines(v))


def fit_text(v) -> str:
    out = []
    req = v.get("request") or {}
    if req:
        mem = (f"{req['mem_per_cpu_mb'] / 1024:g}G per core" if req.get("mem_per_cpu_mb")
               else f"{req['mem_per_node_mb'] / 1024:g}G per node" if req.get("mem_per_node_mb") else "default memory")
        out.append(f"Request: {req['cpus']} cores on {req['nodes']} node(s), {mem}"
                   + (f", {req['time_text']}" if req.get("time_text") else "")
                   + (f", constraint {req['constraint']}" if req.get("constraint") else ""))
        rows = [[p["name"], p.get("max_time") or "", f"fits now on {p['fits_now']} node{'s' if p['fits_now'] != 1 else ''}",
                 p.get("note") or ""] for p in v["partitions"]]
        out.append(R.table(["Partition", "MaxTime", "Free now", "Note"], rows))
        if v.get("unmet_reasons"):
            out.append("Nodes passed over: " + ", ".join(f"{k} ({n})" for k, n in v["unmet_reasons"].items()))
        if v.get("constraint_note"):
            out.append(v["constraint_note"])
        out.append(v.get("note") or "")
    for r in v.get("reservations") or []:
        kind = "Maintenance" if r.get("maintenance") else "Reservation"
        from mace.cluster.reservations import human_duration
        out.append(f"Warning: {kind} {r['name']} starts in {human_duration(r['starts_in_s'])} on nodes this job "
                   f"could use; a job running past {r['start']} may not start before it.")
    p = v.get("predict")
    if p:
        if p.get("throttled"):
            out.append(p["reason"])
        elif p.get("accepted"):
            out.append(f"SLURM predicts a start at {p['start']} on {p['nodes']} in partition {p['partition']}.")
        elif p.get("accepted") is False:
            out.append(f"SLURM would refuse this request: {p.get('reason')}")
        else:
            out.append(f"SLURM did not answer: {p.get('reason')}")
        out.append(p.get("note") or "")
    return "\n".join([x for x in out if x] + _meta_lines(v))


def reservations_text(v) -> str:
    if not v["reservations"]:
        return "\n".join(["No reservations scheduled."] + _meta_lines(v))
    from mace.cluster.reservations import human_duration
    rows = [[r["name"], "maintenance" if r["maintenance"] else "reservation", r["start"], r["end"],
             human_duration(r["starts_in_s"]) if r.get("starts_in_s") and r["starts_in_s"] > 0 else "now",
             r["nodes"][:40]] for r in v["reservations"]]
    return "\n".join([R.table(["Name", "Kind", "Start", "End", "Starts in", "Nodes"], rows)] + _meta_lines(v))


def quota_text(v) -> str:
    from mace.cluster.quotas import human_bytes
    rows = []
    for r in v["rows"]:
        lvl = {"warning": "WARNING", "critical": "CRITICAL", "info": "(not your quota)"}.get(r.get("level"), "")
        rows.append([r["label"], f"{human_bytes(r.get('space_used_b'))} / {human_bytes(r.get('space_limit_b'))}",
                     R.pct(r.get("space_pct")),
                     f"{R.num(r.get('files_used'))} / {R.num(r.get('files_limit'))}", R.pct(r.get("files_pct")),
                     r["source"], lvl])
    out = [R.table(["Space", "Used / limit", "Space %", "Files used / limit", "Files %", "Source", "Level"], rows,
                   right=(2, 4))] if rows else []
    out.append(f"Levels: warning from {v['warn_percent']} %, critical from {v['critical_percent']} % "
               f"([quota] in site.toml).")
    out += v.get("warnings") or []
    return "\n".join(out + _meta_lines(v))          # _meta_lines prints the notes


def limits_text(v) -> str:
    rows = [[a["account"], a["partition"] or "", a["qos"], R.num(a["max_jobs"]), R.num(a["max_submit"]),
             a["max_wall"] or "", R.num(a["max_cpus"]), f"{a['running']} / {a['submitted']}", a["running_cpus"]]
            for a in v["associations"]]
    out = [R.table(["Account", "Partition", "QOS", "Max running", "Max submitted", "Max wall", "Max cores/user",
                    "Running / submitted", "Cores in use"], rows)]
    if v.get("submit_slots"):
        out.append(f"Submit slots of account {v.get('account') or '?'} (SLURM's own count): "
                   f"{v['submit_slots']['used']} of {v['submit_slots']['limit']} in use.")
    if v.get("slots_note"):
        out.append(v["slots_note"])
    if v["fairshare"]:
        out.append(R.table(["Account", "Fairshare", "Effective usage"],
                           [[f["account"], f["fairshare"], f["effective_usage"]] for f in v["fairshare"]]))
    for w in v.get("headroom") or []:
        out.append("Warning: " + w)
    if v.get("headroom") == []:
        out.append("These jobs fit within your limits.")
    return "\n".join(out + _meta_lines(v))


def _dur(s):
    from mace.cluster.progress import _fmt_dur
    return _fmt_dur(s)


def _level_lines(pr) -> list:
    """The ETA levels, indented by depth: the total, the stages (finished
    stages only for single points and frequencies), the inner loop; a
    Thresholds line only where a threshold differs from its default."""
    from mace.cluster.progress import thresholds_line
    out = []
    mode = pr.get("mode")
    for lv in pr["levels"]:
        if lv["depth"] > 0 and lv["status"] == "done" and mode not in ("scf", "freq"):
            continue
        pad = "  " * (lv["depth"] + 1)
        out.append(pad + lv["text"])
        tl = thresholds_line(lv.get("thresholds") or [])
        if tl:
            out.append(pad + "  " + tl)
    basis = (pr.get("eta") or {}).get("basis")
    tot = next((lv for lv in pr["levels"] if lv["id"] == "total"), None)
    if tot and basis == (tot.get("info") or {}).get("basis_head") and tot["remaining_s"]["low"] is None:
        basis = None                    # "time left unknown (reason)": the Total line already says it
    if pr.get("stopped"):
        basis = None                    # the Total line says the job ended
    if basis:
        out.append(f"  Why: {basis}")
    return out


def job_text(v) -> str:
    out = [f"Job {v['id']} · {v.get('name') or ''} · {v['state']}"
           + (f" ({v['reason']})" if v.get("reason") and v["reason"] != "None" else "")]
    if v.get("state_note"):
        out.append(f"Note: {v['state_note']}")
    if v.get("elapsed_s") is not None or v.get("limit_s"):
        out.append(f"Elapsed {_dur(v.get('elapsed_s'))} of {_dur(v.get('limit_s'))}"
                   + (f" on {v['nodelist']}" if v.get("nodelist") else ""))
    if v["state"] == "PENDING" and v.get("start") not in (None, "", "N/A", "Unknown"):
        out.append(f"Expected start: {v['start']}")
    out.append(f"Resources: {v.get('ncpus') or '?'} cores, {v.get('nnodes') or '?'} node(s), "
               f"partition {v.get('partition') or '?'}, account {v.get('account') or '?'}")
    eff = (v.get("live") or {}).get("efficiency") or v.get("efficiency") or {}
    out += job_eff.lines(eff)
    if eff.get("details"):
        out.append("  Measured: " + ", ".join(eff["details"]))
    pr = v.get("progress")
    if pr:
        out.append(f"Progress ({pr.get('mode')}): {pr.get('summary') or ''}")
        eta = pr.get("eta") or {}
        if pr.get("levels"):
            out += _level_lines(pr)
        elif eta.get("basis"):
            out.append(f"ETA: {eta['basis']}" if eta.get("low_s") is not None or eta.get("low_points") is not None
                       else f"ETA unknown: {eta['basis']}")
        risk = pr.get("risk") or {}
        if risk.get("headline"):
            out.append("Warning: " + risk["headline"])
        if risk.get("sentence"):
            out.append(risk["sentence"])
        if risk.get("note") and not risk.get("headline"):
            out.append("Note: " + risk["note"])
    elif v.get("progress_note"):
        out.append(v["progress_note"])
    files = v.get("files") or {}
    for k, label in (("workdir", "Work dir"), ("out", "Output"), ("stdout", "Job log"), ("scratch", "Scratch")):
        if files.get(k):
            out.append(f"{label}: {files[k]}")
    arr = v.get("array_tasks")
    if arr:
        by = ", ".join(f"{n} {st.lower()}" for st, n in sorted(arr["counts"].items()))
        out.append(f"Array {arr['array_job']}: {arr['total_in_queue']} task(s) in the queue" + (f" ({by})" if by else ""))
    if v.get("live") and not (v["live"].get("rows") or []):
        out.append("Live use (sstat): " + ((v["live"].get("cache") or {}).get("error") or "no running steps"))
    out.append(f"More: {v['scontrol_command']}; in the GUI: Cluster view › Overview › Details.")
    return "\n".join(out)


def estimate_text(v) -> str:
    if not v.get("wall"):
        return "\n".join([v.get("message") or "No estimate."] + v.get("reasons", []))
    w = v["wall"]
    out = [f"Estimate for {v['features']['stem']} ({v['features']['calc_type']}, {v['features']['family']}) "
           f"on {v['cores']} cores: median {_dur(w['median'])}, range {_dur(w['p10'])} - {_dur(w['p90'])} "
           f"(10-90 %, from {v['n']} similar runs)"]
    out.append(R.table(["Run", "Wall", "Cores", "AO", "k points", "Node type"],
                       [[n["stem"][:50], _dur(n["wall_s"]), n["cores"], n["nao"], n.get("nk") or "-",
                         n.get("group") or "unknown"] for n in v["neighbors"]]))
    s = v.get("suggestion") or {}
    if s.get("walltime"):
        out.append(f"Suggested walltime: {s['walltime']} (the 90 % value x 1.3, rounded up to the hour)")
    if s.get("mem_per_cpu_mb"):
        out.append(f"Suggested memory per core: {s['mem_per_cpu_mb']} MB")
    out += v.get("reasons", [])
    return "\n".join(out)


def overview_text(v) -> str:
    u, lim = v["usage"], v["limits"]
    out = [f"Cluster {v['cluster']}" + (f" · MACE runs inside job {v['in_allocation']['job']} on "
                                        f"{v['in_allocation']['node']}" if v.get("in_allocation") else "")]
    out.append(f"Usable partitions: {', '.join(v['usable_partitions']) or '(unknown)'}")
    t = v["totals"]
    out.append(f"Nodes: {t['nodes']} ({t['idle']} idle, {t['mixed']} mixed, {t['allocated']} allocated, "
               f"{t['down_drain']} down or drained); free cores {t['cpus_free']:,} of {t['cpus_total']:,}")
    out.append(f"Your jobs: {u['running']} running · {u['pending']} pending"
               + (f" (of {R.num(lim['max_jobs'])} / {R.num(lim['max_submit'])})" if lim.get("max_jobs") else ""))
    if lim.get("max_cpus"):
        out.append(f"Cores in use: {u['running_cpus']} of {lim['max_cpus']} (QOS)")
    for f in v["fairshare"]:
        out.append(f"Fairshare: {f['fairshare']} ({f['account']})")
    for w in v["quota_warnings"]:
        out.append("Warning: " + w)
    if v["reservations_soon"]:
        from mace.cluster.reservations import human_duration
        for r in v["reservations_soon"]:
            kind = "Maintenance" if r.get("maintenance") else "Reservation"
            out.append(f"{kind} starts in {human_duration(r['starts_in_s'])} ({r['name']}, nodes {r['nodes'][:40]})")
    else:
        out.append("Maintenance: none scheduled")
    return "\n".join(out + _meta_lines(v))


# ------------------------------------------------------------------ main
def _files(args) -> int:
    root = Path(args.path).expanduser()
    if not root.is_dir():
        print(f"mace cluster files: {root} is not a folder", file=sys.stderr)
        return EXIT_ARGS
    import tempfile
    fd, out = tempfile.mkstemp(prefix="mace-scan-", suffix=".json")
    os.close(fd)
    os.unlink(out)
    cmd = ["nice", "-n", "19", sys.executable, "-m", "mace.cluster.filescan", str(root), "--out", out,
           "--top", str(args.top)]
    if not shutil.which("nice"):
        cmd = cmd[3:]
    if args.sizes:
        cmd += ["--sizes", args.sizes]
    if args.max_seconds:
        cmd += ["--max-seconds", str(args.max_seconds)]
    env = dict(os.environ)
    pkg_root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = pkg_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, env=env, cwd=pkg_root)
    rc = 0
    try:
        for line in p.stdout:
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if msg.get("error"):
                print(msg["error"], file=sys.stderr)
            elif "entries" in msg and not msg.get("done") and not args.json:
                print(f"  scanned {msg['entries']:,} entries in {msg['dirs']:,} folders ({msg['elapsed']} s)",
                      file=sys.stderr)
        rc = p.wait()
    except KeyboardInterrupt:
        p.terminate()
        p.wait()
        return 130
    if rc != 0:
        return EXIT_BUSY if rc == 5 else rc
    try:
        res = json.loads(Path(out).read_text())
    finally:
        try:
            os.unlink(out)
        except OSError:
            pass
    if args.json:
        print(json.dumps(res, indent=1))
        return 0
    from mace.cluster.quotas import human_bytes
    print(f"{res['files']:,} files in {res['dirs']:,} folders under {res['root']} ({res['elapsed_s']} s; "
          f"{res['coverage']})")
    print(R.table(["Folder", "Files", "Size"], [[x["path"], f"{x['files']:,}", human_bytes(x["bytes"]) if x["bytes"]
                                                  else "-"] for x in res["top_by_count"]], right=(1,)))
    if res["suggestions"]:
        print("\nCleanup suggestions (check each; nothing was deleted):")
        for s in res["suggestions"]:
            print(f"  [{s['rule']}] {s['reason']}")
            if s.get("command"):
                print(f"      {s['command']}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = build_parser()
    try:
        args = ap.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0) if exc.code in (0, None) else EXIT_ARGS
    cmd = args.cmd or "overview"
    from mace.cluster import reader, views
    if cmd in NEEDS_SLURM and not reader.slurm_present():
        print(views.NO_SLURM, file=sys.stderr)
        return EXIT_NO_SLURM
    try:
        if cmd == "overview":
            _out(args, views.overview(refresh=args.refresh, tag="refresh" if args.refresh else "cli"), overview_text)
        elif cmd == "nodes":
            _out(args, views.nodes_view(group_by=args.group_by, all_partitions=args.all_partitions,
                                        partition=args.partition, feature=args.feature, free=args.free,
                                        refresh=args.refresh), nodes_text)
        elif cmd == "fit":
            from mace.cluster import fit as fit_mod
            if args.script:
                req = None
                if args.cpus:
                    req = fit_mod.request_from(args.cpus, args.nodes, args.mem_per_cpu, args.mem, args.gpus,
                                               args.constraint, args.partition, args.time)
                v = views.fit_view(req or {}, predict=True, script_path=Path(args.script), refresh=args.refresh)
            else:
                if not args.cpus:
                    print("mace cluster fit: give --cpus N (or --script JOB.sh with --predict)", file=sys.stderr)
                    return EXIT_ARGS
                req = fit_mod.request_from(args.cpus, args.nodes, args.mem_per_cpu, args.mem, args.gpus,
                                           args.constraint, args.partition, args.time)
                v = views.fit_view(req, predict=args.predict, refresh=args.refresh)
            _out(args, v, fit_text)
        elif cmd == "reservations":
            _out(args, views.reservations_view(refresh=args.refresh), reservations_text)
        elif cmd == "quota":
            _out(args, views.quotas_view(refresh=args.refresh, rediscover=args.rediscover), quota_text)
        elif cmd == "files":
            return _files(args)
        elif cmd == "limits":
            if args.check_submit is not None and args.check_submit < 0:
                print("mace cluster limits: --check-submit needs a positive number", file=sys.stderr)
                return EXIT_ARGS
            _out(args, views.limits_view(check_submit=args.check_submit, cpus=args.cpus, refresh=args.refresh),
                 limits_text)
        elif cmd == "job":
            from mace.cluster import jobinfo
            caveat_shown = False
            while True:
                try:
                    v = jobinfo.job(args.id, refresh=args.refresh, live=args.live, tag="cli")
                except jobinfo.NotYours as exc:
                    print(f"Job {args.id}: {exc}", file=sys.stderr)
                    return EXIT_NOT_FOUND
                if caveat_shown:         # --follow: the CPU-efficiency caveat once, not every 60 s
                    for e in (v.get("efficiency"), (v.get("live") or {}).get("efficiency")):
                        if isinstance(e, dict):
                            e["cpu_caveat"] = None
                _out(args, v, job_text)
                caveat_shown = True
                if not args.follow:
                    break
                try:
                    time.sleep(60)
                except KeyboardInterrupt:
                    break
                args.live = False
        elif cmd == "estimate":
            from mace.cluster import history
            deck = Path(args.deck)
            if not deck.is_file():
                print(f"mace cluster estimate: {deck} not found", file=sys.stderr)
                return EXIT_ARGS
            v = history.estimate(deck, cores=args.cores, history=Path(args.history) if args.history else None,
                                 db_path=Path(args.db) if args.db else None, use_sacct=reader.slurm_present())
            _out(args, v, estimate_text)
        elif cmd == "devnodes":
            from mace.cluster import devnodes
            v = devnodes.probe_all(refresh=args.refresh)
            if args.pick:
                if not v.get("pick"):
                    print(v.get("message") or "No interactive node answered.", file=sys.stderr)
                    return EXIT_NOT_FOUND
                print(v["pick"])
                return 0
            _out(args, v, lambda d: d["message"] if d.get("message") and not d["rows"] else "\n".join(
                [R.table(["Host", "Load/core", "Cores", "Free memory", "State"],
                         [[r["host"], r.get("load_per_core", "-"), r.get("cores", "-"),
                           f"{r['mem_available_gb']} GiB" if r.get("mem_available_gb") is not None else "-",
                           "chosen" if r["host"] == d.get("pick") else ("unreachable: " + r.get("error", ""))
                           if not r.get("reachable") else ""] for r in d["rows"]])]))
    except ValueError as exc:
        print(f"mace cluster {cmd}: {exc}", file=sys.stderr)
        return EXIT_ARGS
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
