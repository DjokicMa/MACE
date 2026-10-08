"""``mace site``: show (and create) the site settings MACE writes into job scripts.

    mace site [--json] [--refresh] [--init]
    mace site check [--partition P] [--account A] [--kind scf|prop|both] [--no-sbatch] [--json]

Shows the site.toml MACE reads, every value with where it comes from
(``env MACE_...``, ``site.toml`` or ``built-in``), the CRYSTAL module the job
scripts load (with the reason and any rejected candidates), and, where
SLURM is installed, whether the account is one of your SLURM accounts.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from mace.submission import site

SOURCE_WORDS = {"default": "built-in, loads here", "site.toml": "site.toml, loads here",
                "unchecked": "not checked: no module system here", "discovered": "discovered here",
                "not found here": "not found here"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mace site",
        description="Show the site settings MACE writes into the job scripts it generates: the SLURM "
                    "account, partition, QOS, constraint and extra directives, the job size, how CRYSTAL "
                    "is started (mpirun or srun), the CRYSTAL module or directory, the Python modules and "
                    "the scratch directory. They come from the environment (MACE_SLURM_ACCOUNT, "
                    "MACE_SLURM_PARTITION, MACE_CRYSTAL_MODULE, ...), then ~/.config/mace/site.toml "
                    "(MACE_SITE_CONFIG names another file), then group files (/etc/mace/site.toml, or "
                    "MACE_SITE_GROUP_CONFIG), then MACE's built-in values. With no setting of your own, the "
                    "built-in account is left out unless you are a member of it, Python modules that do not "
                    "load are replaced by the Python that runs MACE, and the queue limits follow your "
                    "MaxSubmitJobs. MACE submits jobs with SLURM only; PBS, Torque and LSF are planned for "
                    "MACE 1.2 or later (mace run runs decks on the node you are on). "
                    "mace site check tests the settings against this cluster, with sbatch --test-only.",
        epilog="Example: mace site --init, then edit the [slurm] account line, then mace site check. "
               "See docs/cli.md, section Site settings.")
    p.add_argument("action", nargs="?", choices=("show", "check"), default="show",
                   help="show (default): the settings and where each comes from; check: test them here "
                        "(SLURM, account, partition, job size, CRYSTAL, MPI, Python, scratch, and a dry "
                        "submission with sbatch --test-only; exit 0 ok, 1 warnings, 2 failed)")
    p.add_argument("--json", action="store_true", help="print the same as JSON")
    p.add_argument("--refresh", action="store_true",
                   help="forget the cached CRYSTAL module choice, Python module check and SLURM answers "
                        "(accounts, MaxSubmitJobs, partition limits) and check again")
    p.add_argument("--init", action="store_true",
                   help="create site.toml with every setting commented out (only when it does not exist)")
    p.add_argument("--partition", metavar="P", help="check: test this partition instead of the settings' one")
    p.add_argument("--account", metavar="A", help="check: test this account instead of the settings' one")
    p.add_argument("--kind", choices=("scf", "prop", "both"), default="both",
                   help="check: the job scripts to test (Pcrystal, Pproperties or both; default both)")
    p.add_argument("--no-sbatch", action="store_true",
                   help="check: write the job scripts and read them with bash -n, but do not ask sbatch; "
                        "the scripts are kept for reading")
    return p


def _source_text(res: dict) -> str:
    src = res.get("source") or ""
    if src.startswith("MACE_CRYSTAL_MODULE"):
        return "pinned by MACE_CRYSTAL_MODULE (not checked)" if "pinned" in src else src
    if src == "default":
        return "built-in, loads here"
    return SOURCE_WORDS.get(src, src)


def _src(v: dict) -> str:
    """'built-in', 'site.toml: <path>', 'group: <path>' or 'env MACE_...'."""
    src = v.get("source") or "built-in"
    if v.get("origin") and src in ("site.toml", "group"):
        return f"{src}: {v['origin']}"
    return src


def _res_text(r: dict) -> str:
    v = r["value"]
    flag = "--mem" if v["mem_form"] == "mem" else "--mem-per-cpu"
    return f"--ntasks={v['ntasks']} -N {v['nodes']} --cpus-per-task={v['cpus_per_task']} {flag}={v['mem']}"


def _fit_text(r: dict) -> str:
    f = r.get("fit")
    if not f:
        return ""
    if f["ok"]:
        return f"; fits partition {f['partition']}: {f['node_cores']} cores, {f['node_mem'] // 1024}G per node"
    return (f"; does NOT fit: the largest node of partition {f['partition']} has {f['node_cores']} cores and "
            f"{f['node_mem'] // 1024}G; set [resources] ntasks = {f['suggest']['ntasks']}")


def render(d: dict) -> List[str]:
    cfg = d["config"]
    lines = ["MACE site settings (what MACE writes into the job scripts it generates)"]
    if cfg["path"] is None:
        lines.append("  Config file: none (MACE_SITE_CONFIG=none)")
    else:
        lines.append(f"  Config file: {cfg['path']} " + ("(exists)" if cfg["exists"] else
                                                          "(not created; mace site --init creates it)"))
    for g in cfg.get("group") or []:
        lines.append(f"  Group file:  {g['path']} (owner {g.get('owner') or '?'}; its values apply unless "
                     "yours set them)")
    acct = d["account"]
    lines.append(f"  Account:     {site.account_label(acct['value'])}  ({_src(acct)})")
    part = d["partition"]
    lines.append(f"  Partition:   {part['value'] or '(none)'}  ({_src(part)})")
    extra = d.get("slurm_extra") or {}
    for key, label in (("qos", "QOS:        "), ("constraint", "Constraint: ")):
        if key in extra:
            lines.append(f"  {label} {extra[key]['value'] or '(none)'}  ({_src(extra[key])})")
    if "extra" in extra:
        lines.append(f"  Extra:       {' '.join(extra['extra']['value']) or '(none)'}  ({_src(extra['extra'])})")
    res = d.get("resources") or {}
    if res:
        lines.append(f"  Resources:   {_res_text(res['scf'])}  ({_src(res['scf'])}{_fit_text(res['scf'])})")
        lines.append(f"  Properties:  {_res_text(res['prop'])}  ({_src(res['prop'])}{_fit_text(res['prop'])})")
    mpi = d.get("mpi")
    if mpi:
        how = mpi["launcher"] + (f" --mpi={mpi['srun_mpi']}" if mpi.get("srun_mpi") else "")
        if mpi.get("env"):
            how += " with " + " ".join(mpi["env"])
        lines.append(f"  MPI launch:  {how}  ({_src(mpi)})")
    cr = d["crystal"]
    lines.append(f"  CRYSTAL:     {' then '.join(cr['modules'])}  ({_source_text(cr)})")
    if cr.get("note"):
        lines.append(f"               {cr['note']}")
    for name, why in cr.get("rejected") or []:
        lines.append(f"               not used: {name}: {why}")
    inst = d.get("crystal_install")
    if inst:
        if inst["bin_dir"]:
            lines.append(f"  Executables: {inst['scf']}, {inst['properties']} in {inst['bin_dir']}  ({_src(inst)})")
        else:
            lines.append(f"  Executables: {inst['scf']}, {inst['properties']} from the module  ({_src(inst)})")
    purge = d.get("modules_purge")
    if purge and not purge["value"]:
        lines.append(f"  Modules:     no module purge before loading  ({_src(purge)})")
    sc = d.get("scratch")
    if sc:
        lines.append(f"  Scratch:     {', then '.join(sc['order'])}  ({_src(sc)})")
    py = d["python"]
    if py.get("executable"):
        lines.append(f"  Python:      {py['executable']}  (the Python that runs MACE: "
                     "the built-in Python modules do not load here)")
    else:
        lines.append(f"  Python:      {' '.join(py['value']) or '(none)'}  ({py['source']})")
    q = d.get("queue") or {}
    if q:
        lines.append(f"  Queue:       --max-jobs {q.get('max_jobs')} --reserve {q.get('reserve')}  ({q.get('source')})")
    lr = d.get("local_run")
    if lr:
        if lr.get("max_time"):
            m = lr["max_time"]
            lines.append(f"  mace run:    at most {m // 3600}h{m % 3600 // 60:02d}m per deck  ({_src(lr)})")
        elif lr.get("msu_hpcc"):
            lines.append("  mace run:    at most 1h55m per deck  (built-in: MSU HPCC dev nodes)")
        else:
            lines.append("  mace run:    at most the node's CPU-time limit (ulimit -t) minus 5m  (built-in)")
    if d.get("walltime_cap"):
        w = d["walltime_cap"]
        lines.append(f"  Walltime:    at most {w['value']} (partition {w['partition']})")
    for key, label in (("gateway_message", "Login-node text"), ("module_message", "Module text")):
        m = d["messages"][key]
        if m["source"] != "built-in":
            lines.append(f"  {label}: {m['value']}  ({m['source']})")
    chk = d.get("account_check") or {}
    if chk.get("status") == "member":
        lines.append(f"  Account check: {acct['value']} is one of your SLURM accounts")
    elif chk.get("status") == "default":
        dflt = chk.get("default")
        lines.append("  Account check: no #SBATCH -A line; SLURM charges your default account"
                     + (f" ({dflt})" if dflt else "") + f" (yours: {', '.join(chk.get('accounts') or [])})")
    elif chk.get("status") == "not_member":
        yours = ", ".join(chk.get("accounts") or []) or "none found"
        lines.append(f"  Account {acct['value']} is not one of your SLURM accounts (yours: {yours}). "
                     f"Set [slurm] account in {cfg['path'] or '~/.config/mace/site.toml'} or MACE_SLURM_ACCOUNT.")
    else:
        lines.append("  Account check: account not checked (no SLURM accounting here)")
    if d.get("monitoring_config"):
        lines.append(f"  Monitoring settings: {d['monitoring_config']} (mace cluster)")
    for w in d.get("warnings") or []:
        lines.append(f"  Warning: {w}")
    for e in d.get("errors") or []:
        lines.append(f"  Error: {e}")
    return lines


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.action == "check":
        from mace.submission import site_check
        kinds = ("scf", "prop") if args.kind == "both" else (args.kind,)
        report = site_check.run(partition=args.partition, account=args.account, kinds=kinds,
                                sbatch=not args.no_sbatch)
        if args.json:
            print(json.dumps(report, indent=1, default=str))
        else:
            print("\n".join(site_check.render(report)))
        return report["exit"]
    if args.init:
        try:
            written, path = site.init_file()
        except (site.SiteConfigError, OSError) as exc:
            print(f"mace site: {exc}", file=sys.stderr)
            return 2
        if not written:
            print(f"mace site: {path} already exists; edit it")
            return 1
        print(f"Wrote {path} with every setting commented out; uncomment a line and set your value, "
              "then check with: mace site")
        return 0
    d = site.describe(check_account=True, refresh=args.refresh)
    if args.json:
        print(json.dumps(d, indent=1, default=str))
    else:
        print("\n".join(render(d)))
    return 2 if d.get("errors") else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
