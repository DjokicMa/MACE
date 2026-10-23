"""``mace site``: show (and create) the site settings MACE writes into job scripts.

    mace site [--json] [--refresh] [--init]
    mace site check [--partition P] [--account A] [--kind scf|prop|both] [--no-sbatch] [--json]
    mace site set KEY VALUE        (e.g. mace site set resources.ntasks 16)
    mace site unset KEY
    mace site set --folder [DIR] KEY VALUE    (a project folder's mace-folder.toml)
    mace site unset --folder [DIR] KEY
    mace site suggest DECK... [--type T] [--json]

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
               "Or: mace site set resources.ntasks 16; mace site set resources.properties.ntasks 14; "
               "mace site set resources.FREQ.walltime 3-00:00:00; mace site unset resources.ntasks. "
               "See docs/cli.md, section Site settings.")
    p.add_argument("action", nargs="?", choices=("show", "check", "set", "unset", "suggest"), default="show",
                   help="show (default): the settings and where each comes from; check: test them here "
                        "(SLURM, account, partition, job size, CRYSTAL, MPI, Python, scratch, and a dry "
                        "submission with sbatch --test-only; exit 0 ok, 1 warnings, 2 failed); "
                        "set KEY VALUE / unset KEY: change one setting in your site.toml (created when "
                        "absent), keeping its comments; KEY is table.key, e.g. resources.ntasks, "
                        "resources.properties.mem, resources.OPT.walltime, slurm.account; "
                        "suggest DECK...: the job size MACE suggests for each deck from its size (AOs, "
                        "k-points, spins, type) and why, also when [resources] auto is off")
    p.add_argument("words", nargs="*", metavar="KEY [VALUE]", help="set / unset: the key (and the value)")
    p.add_argument("--folder", nargs="?", const=".", metavar="DIR",
                   help="set / unset: change the project folder's mace-folder.toml instead (DIR, default "
                        "here), as the GUI's Settings > Save to this folder does; KEY is table.key, e.g. "
                        "plots.energy_window -3,5; units.energy meV; submit.walltime 1-00 (see docs/cli.md, "
                        "section Folder settings from the shell)")
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
    p.add_argument("--type", metavar="T", dest="calc_type",
                   help="suggest: size the decks as this calculation type (OPT, SP, FREQ, ELASTIC, BAND, DOSS, ...)")
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
    """'built-in', 'site.toml: <path>', 'group: <path>', 'folder: <path>' or 'env MACE_...'."""
    src = v.get("source") or "built-in"
    if v.get("origin") and src in ("site.toml", "group", "folder"):
        return f"{src}: {v['origin']}"
    return src


def _res_text(r: dict) -> str:
    v = r["value"]
    flag = "--mem" if v["mem_form"] == "mem" else "--mem-per-cpu"
    return f"--ntasks={v['ntasks']} -N {v['nodes']} --cpus-per-task={v['cpus_per_task']} {flag}={v['mem']}"


def _type_src_text(t: dict) -> str:
    """Where each value of a calculation type's job size comes from."""
    groups: dict = {}
    for key, s in (t.get("set") or {}).items():
        where = s["source"] if s["source"].startswith("env") else s["table"]
        groups.setdefault(where, []).append(key)
    if not groups:
        return "built-in"
    text = "; ".join(f"{', '.join(keys)} from {where}" for where, keys in groups.items())
    return text + ("; others built-in" if len(t["set"]) < 5 else "")


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
    walls = d.get("walltime") or {}
    if any(w.get("set") for w in walls.values()):
        scf, prop = walls["scf"], walls["prop"]
        if (scf["value"], _src(scf)) == (prop["value"], _src(prop)):
            lines.append(f"  Walltime:    {scf['value']}, every job  ({_src(scf)})")
        else:
            lines.append(f"  Walltime:    {scf['value']}  ({_src(scf)}); properties {prop['value']}  ({_src(prop)})")
    by_type = d.get("resources_by_type") or {}
    if any(t.get("own") for t in by_type.values()):
        lines.append("  By calculation type ([resources.<TYPE>] wins over [resources.properties] and [resources]):")
        for name, t in by_type.items():
            lines.append(f"    {name:<17}{_res_text(t)} -t {t['walltime']}  ({_type_src_text(t)})")
    auto = d.get("auto") or {}
    if auto:
        on = [t for t, v in auto.items() if v]
        lines.append(f"  Suggestions: {'on for ' + ', '.join(on) if on else 'off'} ([resources] auto: job size from "
                     "each deck's size, below the values you set; mace site suggest DECK shows one)")
    fo = d.get("folder")
    if fo:
        wt = fo.get("walltime") or {}
        lines.append(f"  Folder:      {fo['path']}  (its [submit] table; site.toml [folders] allows "
                     f"{', '.join(fo.get('allow') or []) or 'nothing'})")
        if wt:
            lines.append("  Walltime:    " + ", ".join(f"{'CRYSTAL' if k == 'scf' else 'properties'} {v}"
                                                       for k, v in sorted(wt.items(), key=lambda kv: kv[0] != "scf"))
                         + "  (folder)")
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
    elif chk.get("status") == "unreadable":
        lines.append(f"  Account check: SLURM accounting cannot be read here: {site.NO_ACCOUNT_TEXT} "
                     f"([slurm] account in {cfg['path'] or '~/.config/mace/site.toml'})")
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


def _folder_edit(action: str, folder: str, words: List[str]) -> int:
    """`mace site set|unset --folder DIR KEY [VALUE]`: the folder's
    mace-folder.toml through the GUI's writer (folder_settings.write) and its
    refusals (folder_settings.save_target); a [submit] value the site's
    [folders] rules would ignore is refused too."""
    import os
    import tempfile
    from pathlib import Path
    from mace.utils import folder_settings as F
    want = 2 if action == "set" else 1
    if len(words) != want:
        usage = ("mace site set --folder [DIR] KEY VALUE" if action == "set"
                 else "mace site unset --folder [DIR] KEY")
        print(f"mace site: usage: {usage} (e.g. plots.energy_window -3,5)", file=sys.stderr)
        return 2
    key = words[0]
    d = Path(folder).expanduser()
    if not d.is_dir():
        print(f"mace site: {d} is not a folder", file=sys.stderr)
        return 2
    table, name = key.rsplit(".", 1) if "." in key else ("", key)
    try:
        if key not in F.KEYS:
            raise ValueError(f"{key} is not a folder setting (docs/cli.md lists them)")
        target, why = F.save_target(d)
        if why:
            raise ValueError(why)
        sets, unsets = {}, []
        if action == "set":
            value = F.value_from_text(key, words[1])
            bad = F.check_value(key, value)
            if bad is None and F.KEYS[key]["type"] == "path":
                bad = F.check_path(target.parent, value)
            if bad:
                raise ValueError(f"{key} {bad}")
            sets[key] = value
        else:
            if key not in (F.read(target).values if target.exists() else {}):
                print(f"mace site: {key} is not set in {target}", file=sys.stderr)
                return 1
            unsets.append(key)
        if key.startswith("submit."):
            # the site's [folders] rules, as mace submit applies them: the file as it would be, read by site.load
            old = target.read_text(encoding="utf-8") if target.exists() else F.TEMPLATE
            fd, tmp = tempfile.mkstemp(prefix=".mace-folder-check.", suffix=".toml", dir=str(target.parent))
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(F.edit_text(old, sets, unsets))
                os.chmod(tmp, 0o644)
                env = {**os.environ, F.ENV: tmp}
                tag = f"[submit] {key[len('submit.'):]} is ignored: "
                for w in site.load(env=env).warnings:
                    if tag in w:
                        raise ValueError(f"{key}: {w.split(tag, 1)[1]}")
            finally:
                os.unlink(tmp)
        existed = target.exists()
        F.write(target, sets, unsets)
    except (ValueError, OSError, site.SiteConfigError) as exc:
        print(f"mace site: {exc}", file=sys.stderr)
        return 2
    if not existed:
        print(f"Created {target}")
    if action == "set":
        print(f"Set [{table}] {name} = {F.toml_value(sets[key])} in {target}")
    else:
        print(f"Removed [{table}] {name} from {target}")
    return 0


def _edit(args) -> int:
    if getattr(args, "folder", None) is not None:
        return _folder_edit(args.action, args.folder, args.words)
    from mace.submission import site_edit
    want = 2 if args.action == "set" else 1
    if len(args.words) != want:
        usage = "mace site set KEY VALUE" if args.action == "set" else "mace site unset KEY"
        print(f"mace site: usage: {usage} (e.g. resources.ntasks, resources.OPT.walltime)", file=sys.stderr)
        return 2
    try:
        _, lines = site_edit.edit(args.words[0], args.words[1] if want == 2 else None, unset=args.action == "unset")
    except (site_edit.EditError, site.SiteConfigError, OSError) as exc:
        print(f"mace site: {exc}", file=sys.stderr)
        return 1 if "is not set" in str(exc) or "nothing to unset" in str(exc) else 2
    print("\n".join(lines))
    return 0


def suggest_lines(entries: List[dict]) -> List[str]:
    """The text of `mace site suggest`."""
    lines = []
    for e in entries:
        name = e["deck"]
        lines.append(f"{name}  ({e['calc_type'] or 'type unknown'})")
        s = e.get("suggestion")
        if s:
            mem = f"--mem-per-cpu={s['mem_per_cpu']}" if s["calc_type"] in site.SCF_TYPES else f"--mem={s['mem']}"
            lines.append(f"  Suggested:  --ntasks={s['ntasks']} {mem} -t {s['walltime']}")
            lines.append(f"  Why:        {s['reason']}")
            inp = s.get("inputs") or {}
            src = inp.get("source") or ""
            approx = inp.get("approx") or []
            lines.append(f"  Sized from: {src}" + (f" (estimated: {', '.join(approx)})" if approx else ""))
        if e.get("error"):
            lines.append(f"  {e['error']}")
        a = e.get("applies")
        if a:
            flag = "--mem" if a["mem_form"] == "mem" else "--mem-per-cpu"
            state = ("on: mace submit uses it below the values you set" if e["auto"] else
                     "off: mace site set resources.auto true (or resources.<TYPE>.auto true) turns it on")
            lines.append(f"  Job script: --ntasks={a['ntasks']} {flag}={a['mem']} -t {a['walltime']}  "
                         f"([resources] auto {state})")
    return lines


def _suggest(args) -> int:
    if not args.words:
        print("mace site: usage: mace site suggest DECK... [--type T] [--json]", file=sys.stderr)
        return 2
    try:
        entries = site.suggest_decks(args.words, calc_type=args.calc_type)
    except site.SiteConfigError as exc:
        print(f"mace site: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(entries, indent=1, sort_keys=True, default=str))
    else:
        print("\n".join(suggest_lines(entries)))
    return 0 if all(e.get("suggestion") for e in entries) else 1


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] in ("set", "unset") and not {"-h", "--help"} & set(argv[1:]):
        # the value is taken as written, even one that starts with -- (slurm.extra --exclusive)
        words = argv[1:]
        folder = None
        if words and (words[0] == "--folder" or words[0].startswith("--folder=")):
            # --folder [DIR]: DIR is there when one word more than KEY [VALUE] follows
            if words[0].startswith("--folder="):
                folder, words = words[0].split("=", 1)[1] or ".", words[1:]
            else:
                words = words[1:]
                folder = "."
                if len(words) > (2 if argv[0] == "set" else 1):
                    folder, words = words[0], words[1:]
        if words and words[0] == "--":
            words = words[1:]
        return _edit(argparse.Namespace(action=argv[0], words=words, folder=folder))
    # suggest DECK... --json: options among the deck names
    args = (build_parser().parse_intermixed_args(argv) if argv and argv[0] == "suggest"
            else build_parser().parse_args(argv))
    if args.action in ("set", "unset"):
        return _edit(args)
    if args.action == "suggest":
        return _suggest(args)
    if args.words:
        print(f"mace site: unexpected arguments: {' '.join(args.words)}", file=sys.stderr)
        return 2
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
