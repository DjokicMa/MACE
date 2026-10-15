"""``mace site check``: test the site settings against the cluster MACE runs on.

Each item is ``{"item", "status": ok|warn|fail|skip, "detail", "fix"}``:
settings files, SLURM, account, partition, QOS/constraint/extra, job size,
CRYSTAL, MPI launcher, completion Python, scratch, and a dry submission:
the real generator writes a job script for a tiny synthetic deck in a
temporary folder, ``bash -n`` reads it and ``sbatch --test-only`` asks SLURM
whether it would accept it (no job is created). The temporary folder is
always removed (``--no-sbatch`` keeps the script to read and stops there).

``mace site check --json`` prints the same dictionary; the GUI may call
``run`` for the same answer.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from mace.submission import site

# a deck CRYSTAL would accept (rock-salt MgO, STO-3G, one SCF cycle); sbatch
# --test-only never runs it
CHECK_DECK = """MgO site check (MACE)
CRYSTAL
0 0 0
225
4.21
2
12 0.0 0.0 0.0
8 0.5 0.5 0.5
END
BASISSET
STO-3G
DFT
PBE
END
SHRINK
4 4
MAXCYCLE
1
END
"""
CHECK_D3 = """NEWK
4 4
1 0
BAND
MgO site check (MACE)
2 8 40 1 6 1 0
0 0 0 4 0 4
4 0 4 4 4 4
END
"""
NAMES = {"scf": ("site_check", ".d12", CHECK_DECK), "prop": ("site_check_band", ".d3", CHECK_D3)}
NO_SLURM = ("MACE submits to SLURM only; PBS/Torque and LSF support is planned for MACE 1.2 or later. "
            "mace run runs decks on this node.")
TEST_ONLY_RE = re.compile(r"Job \d+ to start at \S+")
SBATCH_TIMEOUT_S = 30


def _item(name: str, status: str, detail: str, fix: str = "") -> Dict[str, str]:
    return {"item": name, "status": status, "detail": detail, "fix": fix}


def _where(env: Dict[str, str]) -> str:
    return str(site.config_path(env) or "~/.config/mace/site.toml")


def _scontrol_partitions(env: Dict[str, str]) -> Dict[str, Dict[str, str]]:
    exe = shutil.which("scontrol", path=env.get("PATH"))
    if not exe:
        return {}
    r = site._slurm_run([exe, "show", "partition", "-a", "-o"], env)
    out = {}
    if r is None or r.returncode != 0:
        return out
    for line in r.stdout.splitlines():
        fields = dict(re.findall(r"(\w+)=(\S*)", line))
        if "PartitionName" in fields:
            out[fields["PartitionName"]] = fields
    return out


def _user_qos(env: Dict[str, str]) -> Optional[List[str]]:
    exe = shutil.which("sacctmgr", path=env.get("PATH"))
    user = site._user(env)
    if not exe or not user:
        return None
    r = site._slurm_run([exe, "-nP", "show", "assoc", f"user={user}", "format=qos"], env)
    if r is None or r.returncode != 0:
        return None
    qos = set()
    for line in r.stdout.splitlines():
        for q in line.strip().split("|")[0].split(","):
            if q.strip():
                qos.add(q.strip())
    return sorted(qos) or None


def _run(argv: List[str], env: Dict[str, str], cwd=None, timeout: float = 60):
    try:
        return subprocess.run(argv, env=env, cwd=cwd, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                              timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return subprocess.CompletedProcess(argv, 127, "", str(exc))


def _writable_by_others(path: str) -> bool:
    try:
        return bool(os.stat(path).st_mode & 0o022)
    except OSError:
        return False


def _expand(value: str, env: Dict[str, str]) -> str:
    home = env.get("HOME") or os.path.expanduser("~")
    user = site._user(env)
    for tok, val in (("${SLURM_JOB_ID}", "0"), ("$SLURM_JOB_ID", "0"), ("${USER}", user), ("$USER", user),
                     ("${HOME}", home), ("$HOME", home)):
        value = value.replace(tok, val)
    return value


def scratch_suggestion(env: Optional[Dict[str, str]] = None) -> Optional[str]:
    """A shared-looking scratch directory of this user found here
    (/scratch/$USER, /work/$USER, $WORK, $SCRATCHDIR, $PSCRATCH), as a
    [paths] scratch value, or None. Offered only, never applied: on some
    clusters /scratch is a node's own disk."""
    env = dict(os.environ if env is None else env)
    user = site._user(env)
    if not user:
        return None
    root = env.get("MACE_SCRATCH_PROBE_ROOT", "")
    cands = [f"{root}/scratch/{user}", f"{root}/work/{user}"]
    cands += [env[v] for v in ("WORK", "SCRATCHDIR", "PSCRATCH") if env.get(v)]
    for d in cands:
        parts = d.split("/") if d else []
        if d and os.path.isdir(d) and os.access(d, os.W_OK) and user in parts:
            # only a whole path segment is the user name ("/scratch/at" for user at
            # is "/scratch/$USER", never "/scr$USERch/$USER")
            value = "/".join("$USER" if p == user else p for p in parts)
            try:
                return site._valid_scratch(value, "suggestion")
            except site.SiteConfigError:
                continue
    return None


def run(env: Optional[Dict[str, str]] = None, partition: Optional[str] = None, account: Optional[str] = None,
        kinds=("scf", "prop"), sbatch: bool = True, keep: bool = False) -> Dict[str, Any]:
    """Every check, in order: {"items": [...], "summary": {...}, "kept": path or None}."""
    env = dict(os.environ if env is None else env)
    if partition:
        env["MACE_SLURM_PARTITION"] = partition
    if account:
        env["MACE_SLURM_ACCOUNT"] = account
    items: List[Dict[str, str]] = []
    where = _where(env)
    # 1. settings files
    try:
        conf = site.load(env)
        files = [str(p) for p in conf.group_paths] + ([str(conf.path)] if conf.exists else [])
        if conf.warnings:
            items.append(_item("Settings", "warn", "; ".join(conf.warnings),
                               "fix or remove the keys named above"))
        else:
            items.append(_item("Settings", "ok", "read " + (", ".join(files) if files else
                                                             "no settings file (built-in values)")))
        ok_settings = True
    except site.SiteConfigError as exc:
        items.append(_item("Settings", "fail", str(exc), f"fix the value named above in {where}"))
        conf, ok_settings = site.Site(), False
    # 2. SLURM
    tools = {t: shutil.which(t, path=env.get("PATH")) for t in ("sbatch", "squeue", "scontrol")}
    slurm = all(tools.values())
    if slurm:
        items.append(_item("SLURM", "ok", "sbatch, squeue and scontrol found"))
    else:
        other = [t for t in ("qsub", "bsub") if shutil.which(t, path=env.get("PATH"))]
        missing = ", ".join(t for t, p in tools.items() if not p)
        items.append(_item("SLURM", "fail", (f"no {missing} here" + (f" ({' and '.join(other)} found)" if other else ""))
                           + ". " + NO_SLURM, "run MACE on a SLURM cluster's login node, or use mace run"))
    if ok_settings:
        site.resolve(conf, env, out=lambda t: None)
    # 3. account
    if not slurm:
        items.append(_item("Account", "skip", "no SLURM here"))
    else:
        chk = site.check_account_membership(conf.account, env, refresh=True)
        st = chk.get("status")
        if st == "member":
            items.append(_item("Account", "ok", f"{conf.account} is one of your SLURM accounts"))
        elif st == "default":
            items.append(_item("Account", "ok", "no #SBATCH -A line: SLURM charges your default account"
                               + (f" ({chk.get('default')})" if chk.get("default") else "")))
        elif st == "unreadable":
            items.append(_item("Account", "warn", "SLURM accounting does not answer here (sacctmgr): "
                               + site.NO_ACCOUNT_TEXT, f'[slurm] account = "my_account"  in {where}'))
        elif st == "not_member":
            mine = chk.get("accounts") or []
            items.append(_item("Account", "fail", f"{conf.account} is not one of your SLURM accounts "
                               f"(yours: {', '.join(mine) or 'none found'})",
                               f'[slurm] account = "{mine[0] if mine else "my_account"}"  in {where}'))
        else:
            items.append(_item("Account", "skip", "SLURM accounting does not answer here (sacctmgr)"))
    # 4. partition
    parts = _scontrol_partitions(env) if slurm else {}
    target = [p for p in (conf.partition or "").split(",") if p]
    if not slurm:
        items.append(_item("Partition", "skip", "no SLURM here"))
    elif not parts:
        items.append(_item("Partition", "skip", "scontrol show partition gave no answer"))
    else:
        if not target:
            target = [n for n, f in parts.items() if f.get("Default") == "YES"][:1]
        bad, notes = [], []
        for p in target:
            f = parts.get(p)
            if f is None:
                bad.append(f"{p} does not exist (partitions: {', '.join(sorted(parts))})")
                continue
            if f.get("State", "UP") != "UP":
                notes.append(f"{p} is {f.get('State')}")
            allow = f.get("AllowAccounts", "ALL")
            if conf.account and allow not in ("ALL", "") and conf.account not in allow.split(","):
                bad.append(f"{p} does not admit account {conf.account} (AllowAccounts={allow})")
            allow_q = f.get("AllowQos", "ALL")
            if conf.qos and allow_q not in ("ALL", "") and conf.qos not in allow_q.split(","):
                bad.append(f"{p} does not admit QOS {conf.qos} (AllowQos={allow_q})")
        label = ", ".join(target) if target else "(SLURM's default)"
        if bad:
            items.append(_item("Partition", "fail", "; ".join(bad), f'[slurm] partition = "<one of: '
                               f'{", ".join(sorted(parts))}>"  in {where}'))
        elif notes:
            items.append(_item("Partition", "warn", "; ".join(notes), "choose a partition that is UP"))
        else:
            items.append(_item("Partition", "ok", f"{label} exists and admits this account"))
    # 5. qos / constraint / extra
    if not (conf.qos or conf.constraint or conf.extra):
        items.append(_item("QOS/constraint", "ok", "none set"))
    elif not slurm:
        items.append(_item("QOS/constraint", "skip", "no SLURM here"))
    else:
        problems = []
        if conf.qos:
            mine = _user_qos(env)
            if mine is not None and conf.qos not in mine:
                problems.append((f"QOS {conf.qos} is not one of yours ({', '.join(mine)})",
                                 f'[slurm] qos = "{mine[0]}"  in {where}'))
        if conf.constraint:
            part = target[0] if target else None
            feats = set((site.node_shapes(part, env, refresh=True).get(part) or {}).get("features") or []) \
                if part else set()
            wanted = {t for t in re.split(r"[&|,\[\]*!():]+", conf.constraint) if t and not t.isdigit()}
            if feats and not wanted <= feats:
                problems.append((f"no node of {part} has {', '.join(sorted(wanted - feats))} "
                                 f"(features: {', '.join(sorted(feats))})", f"[slurm] constraint in {where}"))
        if problems:
            items.append(_item("QOS/constraint", "fail", "; ".join(p for p, _ in problems), problems[0][1]))
        else:
            shown = " ".join(x for x in ([f"--qos={conf.qos}"] if conf.qos else [])
                             + ([f"--constraint={conf.constraint}"] if conf.constraint else []) + conf.extra)
            items.append(_item("QOS/constraint", "ok", shown))
    # 6. resources
    if not slurm:
        items.append(_item("Resources", "skip", "no SLURM here"))
    else:
        details, fails, warns = [], [], []
        for kind in kinds:
            eff = conf.effective_resources(kind)
            f = site.fit(conf, kind, env, refresh=True)
            flag = "--mem" if eff["mem_form"] == "mem" else "--mem-per-cpu"
            req = f"--ntasks={eff['ntasks']} -N {eff['nodes']} {flag}={eff['mem']}"
            if f is None:
                details.append(f"{kind}: {req} (node shape unknown)")
            elif f["ok"]:
                details.append(f"{kind}: {req} fits {f['partition']} ({f['node_cores']} cores, "
                               f"{f['node_mem'] // 1024}G per node)")
            else:
                fails.append((f"{kind}: {req} does not fit {f['partition']}: its largest node has "
                              f"{f['node_cores']} cores and {f['node_mem'] // 1024}G",
                              f"[resources] ntasks = {f['suggest']['ntasks']} and mem_per_cpu = "
                              f"\"{f['suggest']['mem_per_cpu']}\"  in {where}"))
        if conf.walltime_cap:
            warns.append(f"the partition allows at most {site.walltime_text(conf.walltime_cap)}; MACE asks for that")
        if any(conf.effective_resources(k)["nodes"] > 1 for k in kinds):
            warns.append(site.MULTI_NODE_WARNING)
        if fails:
            items.append(_item("Resources", "fail", "; ".join(d for d, _ in fails), fails[0][1]))
        elif warns:
            # each warning its own fix: a walltime cap is about the partition,
            # only the multi-node one is about scratch
            fixes = ([f'[slurm] partition = "<a partition with a longer time limit>"  in {where}']
                     if conf.walltime_cap else []) \
                + (["[paths] scratch, see docs/cli.md"] if site.MULTI_NODE_WARNING in warns else [])
            items.append(_item("Resources", "warn", "; ".join(details + warns), "; ".join(fixes)))
        else:
            items.append(_item("Resources", "ok", "; ".join(details)))
    # 7. CRYSTAL
    if conf.crystal_bin_dir:
        exes = [os.path.join(conf.crystal_bin_dir, n) for n in conf.binaries()]
        missing = [e for e in exes if not os.access(e, os.X_OK)]
        open_ = [p for p in [conf.crystal_bin_dir] + exes if _writable_by_others(p)]
        if open_:
            items.append(_item("CRYSTAL", "warn", f"writable by group or others: {', '.join(open_)} "
                               "(every job runs what is there)", f"chmod go-w {' '.join(open_)}"))
        elif missing:
            items.append(_item("CRYSTAL", "warn", f"not found here: {', '.join(missing)} (they may exist on "
                               "compute nodes only)", f"[crystal] bin_dir in {where}"))
        else:
            items.append(_item("CRYSTAL", "ok", f"{', '.join(exes)} found"))
    else:
        try:
            res = site.crystal_module(env, conf, refresh=True)
        except site.SiteConfigError as exc:
            res = None
            items.append(_item("CRYSTAL", "fail", str(exc)))
        if res is not None:
            if res.source == "unchecked":
                items.append(_item("CRYSTAL", "skip", f"no module system here; job scripts load "
                                   f"{' '.join(res.modules)}"))
            elif res.source == "not found here":
                items.append(_item("CRYSTAL", "warn", res.note, f'[modules] crystal = "<your CRYSTAL module>" or '
                                   f'[crystal] bin_dir in {where}'))
            else:
                items.append(_item("CRYSTAL", "ok", " then ".join(res.modules) + f" ({res.source})"
                                   + (f": {res.note}" if res.note else "")))
    # 8. MPI launcher
    launcher = conf.launcher
    path = shutil.which(launcher, path=env.get("PATH"))
    if launcher == "srun" and path and conf.srun_mpi:
        r = _run([path, "--mpi=list"], env, timeout=SBATCH_TIMEOUT_S)
        types = r.stdout + r.stderr
        if conf.srun_mpi in re.findall(r"[A-Za-z0-9_]+", types):
            items.append(_item("MPI launch", "ok", f"srun --mpi={conf.srun_mpi} is available"))
        else:
            items.append(_item("MPI launch", "fail", f"srun does not list --mpi={conf.srun_mpi}: "
                               + " ".join(types.split())[:200], f"[mpi] srun_mpi in {where}"))
    elif path:
        items.append(_item("MPI launch", "ok", f"{launcher} found ({path})"))
    else:
        items.append(_item("MPI launch", "warn", f"{launcher} is not on PATH here (it may come from the CRYSTAL "
                           "module on compute nodes)", ""))
    # 9. completion Python
    items.append(_python_item(conf, env))
    # 10. scratch
    items.append(_scratch_item(conf, env))
    # 11. dry submission
    kept = None
    if not ok_settings:
        items.append(_item("Job script", "skip", "settings are invalid: nothing was generated"))
    else:
        it, kept = _dry_submit(env, kinds, sbatch and slurm, keep or not sbatch)
        items.append(it)
    summary = {s: sum(1 for i in items if i["status"] == s) for s in ("ok", "warn", "fail", "skip")}
    return {"items": items, "summary": summary, "kept": kept,
            "exit": 2 if summary["fail"] else 1 if summary["warn"] else 0}


def _python_item(conf: site.Site, env: Dict[str, str]) -> Dict[str, str]:
    mace_home = str(Path(__file__).resolve().parents[2])
    code = "import tomllib, mace.queue.manager"
    penv = dict(env, PYTHONPATH=mace_home)
    if conf.python_executable:
        argv, label = [conf.python_executable, "-c", code], conf.python_executable
    elif conf.python and envmodules_here(env):
        loads = "; ".join(f"module load {m}" for m in conf.python)
        argv = ["bash", "-lc", ("module purge >/dev/null 2>&1; " if conf.module_purge else "")
                + f"{loads} && python -c '{code}'"]
        label = " ".join(conf.python)
    else:
        argv, label = [sys.executable, "-c", code], sys.executable
    r = _run(argv, penv, timeout=60)
    if r.returncode == 0:
        return _item("Completion Python", "ok", f"{label} imports MACE and reads TOML")
    err = " ".join((r.stderr or r.stdout).split())[-300:]
    if argv[0] == "bash":
        # a login node that serves no modules (MSU's gateway) says nothing
        # about the compute nodes the job runs on, as for the CRYSTAL module
        from mace.submission.local_run import gateway_reason
        reason = gateway_reason(env, site.host_names(env)[1])
        if reason:
            return _item("Completion Python", "warn",
                         f"{label} cannot be checked on this login node ({reason}); jobs load them on the "
                         "compute nodes. Run mace site check on a compute or development node to test them.")
    return _item("Completion Python", "fail", f"{label} cannot run the completion step: {err}",
                 f'[modules] python = ["<a Python 3.11 or newer module>"]  in {_where(env)}')


def envmodules_here(env: Dict[str, str]) -> bool:
    return site.envmodules.module_system(env) is not None


def _scratch_item(conf: site.Site, env: Dict[str, str]) -> Dict[str, str]:
    user = site._user(env)
    cands = []
    if env.get("SCRATCH"):
        cands.append(("$SCRATCH", env["SCRATCH"]))
    if conf.scratch:
        cands.append(("[paths] scratch", _expand(conf.scratch, env)))
    if user:
        cands.append(("/mnt/scratch/$USER", f"/mnt/scratch/{user}"))
    found = next(((label, d) for label, d in cands if os.path.isdir(d) and os.access(d, os.W_OK)), None)
    hint = scratch_suggestion(env) if not conf.scratch else None
    fix = (f'# scratch = "{hint}"   # found here; confirm with your admins that it is shared by all nodes'
           if hint else "")
    if found:
        return _item("Scratch", "ok", f"{found[0]}: {found[1]} is writable here (compute nodes may differ)", fix)
    return _item("Scratch", "warn", "no shared scratch directory found here; jobs fall back to the submit "
                 "directory, then $TMPDIR (node-local)", fix or f"[paths] scratch in {_where(env)}")


def _dry_submit(env: Dict[str, str], kinds, sbatch: bool, keep: bool):
    tmp = Path(tempfile.mkdtemp(prefix="mace-site-check-"))
    lines, ok, failed, fix = [], True, [], ""
    try:
        for kind in kinds:
            name, ext, text = NAMES[kind]
            (tmp / f"{name}{ext}").write_text(text)
            gen, cleanup, _ = site.generator_for(site.TEMPLATES[kind], kind, tmp, env, out=lambda t: None)
            try:
                r = _run(["bash", str(gen), name], dict(env, MACE_HOME=str(Path(__file__).resolve().parents[2])),
                         cwd=str(tmp), timeout=120)
            finally:
                cleanup()
            script = tmp / f"{name}.sh"
            if r.returncode != 0 or not script.is_file():
                ok = False
                failed.append(f"{kind}: the generator failed: {(r.stderr or r.stdout).strip()[-300:]}")
                continue
            n = _run(["bash", "-n", str(script)], env)
            if n.returncode != 0:
                ok = False
                failed.append(f"{kind}: bash -n: {n.stderr.strip()[-300:]}")
                continue
            if not sbatch:
                # the path only when the folder is kept (it is removed otherwise)
                lines.append(f"{kind}: " + (f"{script} written" if keep else "job script written")
                             + ", bash -n passes" + ("" if keep else " (no SLURM here: sbatch --test-only "
                                                                    "not asked)"))
                continue
            exe = shutil.which("sbatch", path=env.get("PATH"))
            s = _run([exe, "--test-only", script.name], env, cwd=str(tmp), timeout=SBATCH_TIMEOUT_S)
            out = (s.stdout or "") + (s.stderr or "")
            m = TEST_ONLY_RE.search(out)
            if s.returncode == 0 and m:
                lines.append(f"{kind}: sbatch --test-only: {m.group(0)}")
            else:
                ok = False
                failed.append(f"{kind}: sbatch --test-only refused it: " + " ".join(out.split())[-300:])
                fix = fix or (site.sbatch_hint(out, env) or "")
    except site.SiteConfigError as exc:
        ok = False
        failed.append(str(exc))
    finally:
        if not keep:
            shutil.rmtree(tmp, ignore_errors=True)
    kept = str(tmp) if keep else None
    if ok:
        detail = "; ".join(lines) + (f" (kept in {tmp})" if keep else "")
        return _item("Job script", "ok", detail), kept
    return _item("Job script", "fail", "; ".join(failed + lines), fix or "see mace site"), kept


def render(report: Dict[str, Any]) -> List[str]:
    tags = {"ok": "[ ok ]", "warn": "[warn]", "fail": "[FAIL]", "skip": "[skip]"}
    width = max(len(i["item"]) for i in report["items"])
    out = []
    for i in report["items"]:
        out.append(f"  {tags[i['status']]} {i['item']:<{width}}  {i['detail']}")
        if i["status"] in ("warn", "fail") and i["fix"]:
            out.append(f"         Fix: {i['fix']}")
        elif i["status"] == "ok" and i["fix"]:
            out.append(f"         {i['fix']}")
    s = report["summary"]
    out.append(f"mace site check: {s['ok']} ok, {s['warn']} warning{'s' if s['warn'] != 1 else ''}, "
               f"{s['fail']} failed" + (f", {s['skip']} skipped" if s["skip"] else ""))
    if report.get("kept"):
        out.append(f"The job scripts are in {report['kept']}")
    return out
