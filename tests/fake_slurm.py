"""A fake SLURM for subprocess tests: sbatch, squeue, sacct, scancel, scontrol,
sinfo and sacctmgr, driven by one JSON state file and logging every call.

Tests run the real `mace_cli` (and the real job callback, `manager.py`) as
subprocesses with these tools first on PATH, so nothing reaches a real queue
and every SLURM command MACE runs is recorded: argv, working directory, the
SLURM_* variables in its environment and, for sbatch, the submitted script.

The fake models what was measured on a real cluster (SLURM 26.05):
- a job array's tasks are ``A_T``; squeue without ``-r`` folds the pending
  tasks of an array into one token, ``A_[2-3%2]`` or, once the pending set has
  gaps, ``A_[2,4-5%2]`` (commas inside the brackets);
- without ``SLURM_BITSTR_LEN=0`` a long task expression is cut at 64
  characters and ends in ``...``;
- sacct's fixed-width JobID column is cut and marked with ``+``; ``-P`` never
  cuts;
- ``sbatch --hold`` leaves every task pending (reason JobHeldUser) until
  ``scontrol release A``.

Each tool is a small bash file that runs this module with the tool's name.
State keys (all optional) are documented on ``FakeSlurm``.
"""
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

TOOLS = ("sbatch", "squeue", "sacct", "scancel", "scontrol", "sinfo", "sacctmgr")
HERE = Path(__file__).resolve().parent


class FakeSlurm:
    """Install the fake tools under ``root/fakebin``.

    State keys:
      next_id (5000)            id of the next plain job
      next_array_id (4242)      id of the next job array
      records                   squeue records: {id, state, reason, name,
                                command, start, timeleft, array, task, raw}
      throttle                  {array id: N} from ``--array=..%N``
      held                      [array ids] submitted with --hold, not released
      sacct / sacct_D / sacct_array
                                {queried id: [[jobid, state], ...]} for plain,
                                ``-D`` and ``--array`` queries
      bracket_force             {array id: token} printed for that array's
                                pending tasks whatever the environment says
      fail_sbatch_call          the n-th sbatch call (1-based, --test-only
                                not counted) fails with a submit-limit error
      fail_scontrol_release     scontrol release fails
      assoc_mgr                 text printed by ``scontrol show assoc_mgr``
      max_array_tasks           in ``scontrol show config`` (10000)
      hooks                     [{when: submit|release, cmd: [...],
                                env: {...}, cwd: ...}] run synchronously; the
                                strings may hold {A} (the array id)
      no_array_hold             sbatch ignores --hold (simulates an
                                unheld submit)
    """

    def __init__(self, root, state=None):
        self.root = Path(root)
        self.bindir = self.root / "fakebin"
        self.bindir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.root / "fake_slurm_state.json"
        self.log_file = self.root / "fake_slurm_log.jsonl"
        self.hook_log = self.root / "fake_slurm_hooks.log"
        self.save(dict(state or {}))
        self.log_file.write_text("")
        for tool in TOOLS:
            path = self.bindir / tool
            path.write_text(
                "#!/bin/bash\n"
                f'exec "{sys.executable}" "{HERE / "fake_slurm.py"}" {tool} "$@"\n')
            path.chmod(0o755)
        env = dict(os.environ)
        env["PATH"] = f"{self.bindir}:{env.get('PATH', '')}"
        env["FAKE_SLURM_STATE"] = str(self.state_file)
        env["FAKE_SLURM_LOG"] = str(self.log_file)
        env["FAKE_SLURM_HOOK_LOG"] = str(self.hook_log)
        env["MACE_NO_BANNER"] = "1"
        for var in ("MACE_WORKFLOW_ID", "MACE_PLANLESS_PROGRESSION", "MACE_CONTEXT_DIR",
                    "MACE_DB_PATH", "MACE_ISOLATION_MODE"):
            env.pop(var, None)
        for var in list(env):
            if var.startswith(("SLURM_", "MACE_ARRAY_")):
                env.pop(var)
        self.env = env

    # state -----------------------------------------------------------------
    def load(self):
        return json.loads(self.state_file.read_text() or "{}")

    def save(self, state):
        self.state_file.write_text(json.dumps(state, indent=1))

    def update(self, **kw):
        with _locked(self.state_file):
            st = self.load()
            st.update(kw)
            self.save(st)

    def calls(self, tool=None):
        out = []
        for line in self.log_file.read_text().splitlines():
            if line.strip():
                entry = json.loads(line)
                if tool is None or entry["tool"] == tool:
                    out.append(entry)
        return out

    def records(self):
        return self.load().get("records", [])

    def set_record(self, job_id, **fields):
        with _locked(self.state_file):
            st = self.load()
            for r in st.get("records", []):
                if r["id"] == job_id:
                    r.update(fields)
            self.save(st)

    def add_record(self, **fields):
        with _locked(self.state_file):
            st = self.load()
            rec = {"state": "PENDING", "reason": "None", "name": "job", "command": "",
                   "start": "N/A", "timeleft": "1:00:00"}
            rec.update(fields)
            st.setdefault("records", []).append(rec)
            self.save(st)

    def remove_record(self, job_id):
        with _locked(self.state_file):
            st = self.load()
            st["records"] = [r for r in st.get("records", [])
                             if r["id"] != job_id and str(r.get("array")) != job_id]
            self.save(st)


# --------------------------------------------------------------------------
# tool implementations (run as `python fake_slurm.py <tool> args...`)

@contextmanager
def _locked(state_file):
    lock = Path(str(state_file) + ".lock")
    with open(lock, "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _load(path):
    try:
        return json.loads(Path(path).read_text() or "{}")
    except (OSError, ValueError):
        return {}


def _log(tool, argv, extra=None):
    entry = {"tool": tool, "argv": [tool] + list(argv), "cwd": os.getcwd(),
             "env": {k: v for k, v in sorted(os.environ.items()) if k.startswith("SLURM_")}}
    if extra:
        entry.update(extra)
    with open(os.environ["FAKE_SLURM_LOG"], "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.write(json.dumps(entry) + "\n")
        fcntl.flock(fh, fcntl.LOCK_UN)


def _leading_directives(text):
    out = []
    for i, line in enumerate(text.splitlines()):
        s = line.strip()
        if i == 0 and s.startswith("#!"):
            continue
        if not s:
            continue
        if s.startswith("#SBATCH"):
            out.append(s[len("#SBATCH"):].strip())
            continue
        if s.startswith("#"):
            continue
        break
    return out


def _array_spec(argv_opts, directives):
    spec = None
    for d in directives:
        m = re.match(r"(?:--array[= ]|-a\s*)(\S+)", d)
        if m:
            spec = m.group(1)
    for i, a in enumerate(argv_opts):
        if a.startswith("--array="):
            spec = a.split("=", 1)[1]
        elif a in ("--array", "-a") and i + 1 < len(argv_opts):
            spec = argv_opts[i + 1]
    return spec


def _expand_spec(spec):
    throttle = None
    if "%" in spec:
        spec, throttle = spec.split("%", 1)
    tasks = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-", 1)
            tasks.extend(range(int(lo), int(hi) + 1))
        else:
            tasks.append(int(part))
    return tasks, (int(throttle) if throttle else None)


def _run_hooks(state, when, array_id, hook_env_extra=None):
    for hook in state.get("hooks", []):
        if hook.get("when") != when:
            continue
        if hook.get("array") and str(hook["array"]) != str(array_id):
            continue
        sub = lambda s: str(s).replace("{A}", str(array_id))
        env = dict(os.environ)
        for k, v in (hook.get("env") or {}).items():
            env[k] = sub(v)
        cmd = [sub(c) for c in hook["cmd"]]
        res = subprocess.run(cmd, cwd=sub(hook.get("cwd") or os.getcwd()), env=env,
                             capture_output=True, text=True)
        with open(os.environ.get("FAKE_SLURM_HOOK_LOG", os.devnull), "a") as fh:
            fh.write(f"=== hook {when} {cmd} rc={res.returncode}\n{res.stdout}\n{res.stderr}\n")


def sbatch(argv):
    state_file = os.environ["FAKE_SLURM_STATE"]
    opts, script, rest = [], None, []
    i = 0
    while i < len(argv):
        a = argv[i]
        if script is None and a.startswith("-"):
            opts.append(a)
            if a in ("-t", "-a", "--array", "-J", "-o", "-A", "-p") and i + 1 < len(argv):
                opts.append(argv[i + 1])
                i += 1
        elif script is None:
            script = a
        else:
            rest.append(a)
        i += 1
    text = ""
    if script:
        try:
            text = Path(script).read_text()
        except OSError:
            text = ""
    extra = {"script": script, "script_text": text,
             "script_sha256": hashlib.sha256(text.encode()).hexdigest()}
    _log("sbatch", argv, extra)
    if "--test-only" in opts:
        sys.stderr.write("sbatch: Job 5999 to start at 2030-01-01T00:00:00 using 1 processors "
                         "on nodes n001 in partition general\n")
        return 0
    if not script or not text:
        sys.stderr.write(f"sbatch: error: Unable to open file {script}\n")
        return 1
    with _locked(state_file):
        state = _load(state_file)
        state["sbatch_calls"] = state.get("sbatch_calls", 0) + 1
        if state.get("fail_sbatch_call") == state["sbatch_calls"]:
            Path(state_file).write_text(json.dumps(state, indent=1))
            sys.stderr.write("sbatch: error: QOSMaxSubmitJobPerUserLimit\n"
                             "sbatch: error: Batch job submission failed: Job violates "
                             "accounting/QOS policy (job submit limit, user's size and/or "
                             "time limits)\n")
            return 1
        directives = _leading_directives(text)
        spec = _array_spec(opts, directives)
        hold = "--hold" in opts and not state.get("no_array_hold")
        name = Path(script).stem
        for d in directives:
            m = re.match(r"(?:-J\s*|--job-name[= ])(\S+)", d)
            if m:
                name = m.group(1)
        records = state.setdefault("records", [])
        if spec:
            job_id = state.get("next_array_id", 4242)
            state["next_array_id"] = job_id + 1
            tasks, throttle = _expand_spec(spec)
            if throttle:
                state.setdefault("throttle", {})[str(job_id)] = throttle
            for t in tasks:
                records.append({"id": f"{job_id}_{t}", "array": str(job_id), "task": t,
                                "state": "PENDING",
                                "reason": "JobHeldUser" if hold else "None",
                                "name": name, "command": str(Path(script).resolve()),
                                "start": "N/A", "timeleft": "1:00:00"})
            if hold:
                state.setdefault("held", []).append(str(job_id))
        else:
            job_id = state.get("next_id", 5000)
            state["next_id"] = job_id + 1
            records.append({"id": str(job_id), "state": "PENDING",
                            "reason": "JobHeldUser" if hold else "None",
                            "name": name, "command": str(Path(script).resolve()),
                            "start": "N/A", "timeleft": "1:00:00"})
        Path(state_file).write_text(json.dumps(state, indent=1))
    print(f"Submitted batch job {job_id}")
    sys.stdout.flush()
    if spec and not hold:
        _run_hooks(state, "submit", job_id)
    return 0


def _ranges(tasks):
    tasks = sorted(tasks)
    parts, start, prev = [], None, None
    for t in tasks:
        if start is None:
            start = prev = t
        elif t == prev + 1:
            prev = t
        else:
            parts.append(f"{start}-{prev}" if prev != start else f"{start}")
            start = prev = t
    if start is not None:
        parts.append(f"{start}-{prev}" if prev != start else f"{start}")
    return ",".join(parts)


def _parse_opts(argv, with_value):
    opts, i = {}, 0
    flags = set()
    while i < len(argv):
        a = argv[i]
        if "=" in a and a.startswith("--"):
            k, v = a.split("=", 1)
            opts[k] = v
        elif a in with_value and i + 1 < len(argv):
            opts[a] = argv[i + 1]
            i += 1
        else:
            flags.add(a)
        i += 1
    return opts, flags


def squeue(argv):
    _log("squeue", argv)
    state = _load(os.environ["FAKE_SLURM_STATE"])
    opts, flags = _parse_opts(argv, {"-o", "-u", "-j", "-t", "-O", "--format"})
    fmt = opts.get("-o") or opts.get("--format") or "%i %T"
    records = list(state.get("records", []))
    if "-j" in opts:
        wanted = set(opts["-j"].split(","))
        records = [r for r in records if r["id"] in wanted or str(r.get("array")) in wanted]
    full = os.environ.get("SLURM_BITSTR_LEN") == "0"
    rows = []
    if "-r" in flags:
        rows = [(r["id"], r) for r in records]
    else:
        seen = set()
        for r in records:
            arr = r.get("array")
            if arr and r["state"] == "PENDING":
                if arr in seen:
                    continue
                seen.add(arr)
                pend = [x for x in records if x.get("array") == arr and x["state"] == "PENDING"]
                thr = (state.get("throttle") or {}).get(str(arr))
                token = f"{arr}_[{_ranges([x['task'] for x in pend])}{'%' + str(thr) if thr else ''}]"
                force = (state.get("bracket_force") or {}).get(str(arr))
                if force:
                    token = force
                elif not full and len(token) > 64:
                    token = token[:61] + "..."
                rows.append((token, r))
            else:
                rows.append((r["id"], r))
    if "-h" not in flags:
        print(re.sub(r"%(\w)", lambda m: {"i": "JOBID", "T": "STATE", "S": "START_TIME",
                                           "j": "NAME", "r": "REASON", "o": "COMMAND",
                                           "L": "TIME_LEFT", "A": "JOBID", "F": "ARRAY_JOB_ID",
                                           "K": "ARRAY_TASK_ID"}.get(m.group(1), m.group(1)),
                     fmt))
    for token, r in rows:
        fields = {"i": token, "T": r["state"], "S": r.get("start", "N/A"),
                  "j": r.get("name", ""), "r": r.get("reason", "None"),
                  "o": r.get("command", ""), "L": r.get("timeleft", "1:00:00"),
                  "A": str(r.get("raw") or (r.get("array") if r["state"] == "PENDING" and r.get("array") else r["id"])),
                  "F": str(r.get("array") or r["id"]),
                  "K": str(r.get("task")) if r.get("task") is not None else "N/A"}
        print(re.sub(r"%(\w)", lambda m: fields.get(m.group(1), ""), fmt))
    return 0


def sacct(argv):
    _log("sacct", argv)
    state = _load(os.environ["FAKE_SLURM_STATE"])
    opts, flags = _parse_opts(argv, {"-j", "--format", "-o", "-S", "-E"})
    ids = (opts.get("-j") or "").split(",")
    fmt = (opts.get("--format") or opts.get("-o") or "JobID,State").split(",")
    table = state.get("sacct", {})
    if "-D" in flags and state.get("sacct_D"):
        table = state["sacct_D"]
    if "--array" in flags and state.get("sacct_array"):
        table = state["sacct_array"]
    parsable = "-P" in flags or "--parsable2" in flags
    for jid in ids:
        entries = table.get(jid)
        if entries is None:
            continue
        if isinstance(entries, str):
            entries = [[jid, entries]]
        for jobid, st in entries:
            cols = []
            for f in fmt:
                f = f.strip().split("%")[0].lower()
                cols.append(jobid if f == "jobid" else st if f == "state" else "Unknown")
            if parsable:
                print("|".join(cols))
            else:
                out = []
                for f, c in zip(fmt, cols):
                    if f.strip().lower().startswith("jobid") and len(c) > 12:
                        c = c[:11] + "+"
                    out.append(f"{c:<12}" if f.strip().lower().startswith("jobid") else f"{c:>10}")
                print(" ".join(out))
    return 0


def scancel(argv):
    _log("scancel", argv)
    state_file = os.environ["FAKE_SLURM_STATE"]
    with _locked(state_file):
        state = _load(state_file)
        if state.get("fail_scancel"):
            sys.stderr.write("scancel: error: Kill job error on job id\n")
            return 1
        ids = [a for a in argv if not a.startswith("-")]
        recs = state.get("records", [])
        for jid in ids:
            recs = [r for r in recs if r["id"] != jid and str(r.get("array")) != jid]
        state["records"] = recs
        Path(state_file).write_text(json.dumps(state, indent=1))
    return 0


def scontrol(argv):
    _log("scontrol", argv)
    state_file = os.environ["FAKE_SLURM_STATE"]
    args = [a for a in argv if a not in ("-a", "-o")]
    state = _load(state_file)
    if args[:1] == ["release"] and len(args) > 1:
        if state.get("fail_scontrol_release"):
            sys.stderr.write("scontrol: error: Invalid job id specified\n")
            return 1
        arr = args[1]
        with _locked(state_file):
            state = _load(state_file)
            for r in state.get("records", []):
                if str(r.get("array")) == arr or r["id"] == arr:
                    if r.get("reason") == "JobHeldUser":
                        r["reason"] = "None"
            state["held"] = [h for h in state.get("held", []) if h != arr]
            Path(state_file).write_text(json.dumps(state, indent=1))
        _run_hooks(state, "release", arr)
        return 0
    if args[:2] == ["show", "config"]:
        print(f"MaxArraySize            = {state.get('max_array_size', 4000001)}")
        print("SchedulerParameters     = bf_continue,max_array_tasks="
              f"{state.get('max_array_tasks', 10000)},nohold_on_prolog_fail")
        return 0
    if args[:2] == ["show", "assoc_mgr"]:
        print(state.get("assoc_mgr", ""))
        return 0
    if args[:2] == ["show", "partition"] and len(args) > 2:
        print(f"PartitionName={args[2]} MaxTime=7-00:00:00 State=UP")
        return 0
    return 0


def quiet(tool, argv):
    _log(tool, argv)
    return 0


def tool_main(tool, argv):
    if tool == "sbatch":
        return sbatch(argv)
    if tool == "squeue":
        return squeue(argv)
    if tool == "sacct":
        return sacct(argv)
    if tool == "scancel":
        return scancel(argv)
    if tool == "scontrol":
        return scontrol(argv)
    return quiet(tool, argv)


if __name__ == "__main__":
    sys.exit(tool_main(sys.argv[1], sys.argv[2:]))
