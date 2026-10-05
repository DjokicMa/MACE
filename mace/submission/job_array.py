"""SLURM job arrays: many calculations, one submission (opt-in).

`mace submit --array` and a plan's ``job_arrays`` setting submit the job
scripts of many materials as SLURM job arrays instead of one job each. Every
material keeps its own job script, exactly the one the default path writes;
an array adds one wrapper script that runs the script on line
``$SLURM_ARRAY_TASK_ID`` of a manifest, in that material's directory, the way
sbatch would run it on its own.

Grouping: the tasks of an array share one resource request, so only scripts
whose leading ``#SBATCH`` block asks for exactly the same thing (job name and
output file aside) share an array. Nothing is rounded up or down; ``5G`` and
``5120M`` are different requests. A request only one deck has is submitted as
an ordinary job, as is a script that sets its own working directory or a
dependency.

Submission: every array of a wave is submitted held (``sbatch --hold``), the
database rows are written, and only then are the arrays released
(``scontrol release A``), so no task can end before its row names it. If any
sbatch is refused, the arrays already submitted are removed again: a wave is
all or nothing. These two, release and abort, are the only actions MACE ever
takes on a bare array id; everything else addresses one task, ``A_T``.

This module is the one implementation behind every route: `mace submit`
(untracked and tracked) and the first wave of a workflow.
"""
import json
import os
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

ARRAYS_DIRNAME = ".mace_arrays"
MANIFEST_HEADER = "# MACE job array manifest v1: task\tdir\tjob\tscript\tcalc_id"
SHIM_MESSAGE = "MACE_ARRAY_SBATCH_SHIM"
DEFAULT_MAX_TASKS = 1000          # when the cluster's limit cannot be read
MAX_TASK_INDEX = 99999            # keeps <name>-<A>_<T>.o names unambiguous
SUBMITTED_RE = re.compile(r"Submitted batch job (\d+)")

# sbatch short options and the long option each stands for.
_SHORT = {
    "-A": "--account", "-a": "--array", "-b": "--begin", "-C": "--constraint",
    "-c": "--cpus-per-task", "-D": "--chdir", "-d": "--dependency", "-e": "--error",
    "-F": "--nodefile", "-G": "--gpus", "-H": "--hold", "-i": "--input",
    "-J": "--job-name", "-k": "--no-kill", "-L": "--licenses", "-M": "--clusters",
    "-m": "--distribution", "-N": "--nodes", "-n": "--ntasks", "-O": "--overcommit",
    "-o": "--output", "-p": "--partition", "-q": "--qos", "-Q": "--quiet",
    "-s": "--oversubscribe", "-S": "--core-spec", "-t": "--time", "-v": "--verbose",
    "-W": "--wait", "-w": "--nodelist", "-x": "--exclude",
}
# Options that take no value.
_FLAGS = {"--hold", "--no-kill", "--overcommit", "--quiet", "--oversubscribe",
          "--verbose", "--wait", "--exclusive", "--requeue", "--no-requeue",
          "--contiguous", "--spread-job", "--use-min-nodes", "--parsable",
          "--test-only", "--ignore-pbs", "--no-requeue", "--reboot"}
# Not part of the request: every task gets its own name and log.
_NOT_REQUEST = {"--job-name", "--output", "--error"}


class ArrayRefusal(Exception):
    """The submission cannot be made as job arrays; nothing was submitted."""


class ArraySubmitError(Exception):
    """sbatch refused an array, or the rows could not be written; the arrays
    of this wave that were already submitted have been removed again."""


@dataclass
class ArrayMember:
    dir: Path
    job: str
    script: Path
    calc_id: Optional[str] = None
    kind: str = "d12"
    data: Dict = field(default_factory=dict)   # caller's own bookkeeping


@dataclass
class ArrayGroup:
    key: Tuple
    members: List[ArrayMember]
    directive_lines: List[str]      # the request, verbatim, minus -J/-o/-e
    separate_stderr: bool


@dataclass
class PreparedArray:
    tag: str
    array_dir: Path
    wrapper: Path
    manifest: Path
    members: List[ArrayMember]
    group: ArrayGroup
    job_name: str
    array_spec: str
    array_job_id: Optional[str] = None

    def task_id(self, index: int) -> str:
        return f"{self.array_job_id}_{index}"


# --------------------------------------------------------------------------
# reading job scripts

def _strip_comment(text: str) -> str:
    # sbatch ends a directive at an unquoted '#' that follows whitespace
    out, quote = [], None
    for i, ch in enumerate(text):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch == "#" and (i == 0 or text[i - 1].isspace()):
            break
        out.append(ch)
    return "".join(out).strip()


def sbatch_directives(script_text: str) -> List[str]:
    """The ``#SBATCH`` lines sbatch reads: the leading block only.

    The shebang, blank lines and other comment lines are skipped; the first
    line that is none of these ends the block (a later ``#SBATCH`` is an
    ordinary comment to sbatch, and so it is here).
    """
    lines = []
    for i, line in enumerate(script_text.splitlines()):
        s = line.strip()
        if i == 0 and s.startswith("#!"):
            continue
        if not s:
            continue
        if re.match(r"#SBATCH(\s|$)", s):
            lines.append(line.rstrip("\n"))
            continue
        if s.startswith("#"):
            continue
        break
    return lines


def parse_directive(line: str) -> List[Tuple[str, Optional[str]]]:
    """``#SBATCH -t 7-00:00:00`` -> [('--time', '7-00:00:00')]."""
    body = _strip_comment(re.sub(r"^\s*#SBATCH", "", line, count=1))
    try:
        tokens = shlex.split(body)
    except ValueError:
        tokens = body.split()
    out, i = [], 0
    while i < len(tokens):
        tok = tokens[i]
        i += 1
        if tok.startswith("--"):
            name, eq, value = tok.partition("=")
            if eq:
                out.append((name, value))
            elif name in _FLAGS or i >= len(tokens) or tokens[i].startswith("-"):
                out.append((name, None))
            else:
                out.append((name, tokens[i]))
                i += 1
        elif tok.startswith("-") and len(tok) >= 2:
            short, rest = tok[:2], tok[2:]
            name = _SHORT.get(short, short)
            if rest:
                out.append((name, rest.lstrip("=")))
            elif name in _FLAGS or i >= len(tokens) or tokens[i].startswith("-"):
                out.append((name, None))
            else:
                out.append((name, tokens[i]))
                i += 1
        else:
            out.append((tok, None))
    return out


def request_of(script_text: str) -> Dict[str, Optional[str]]:
    """Every option of the leading #SBATCH block, long names, last one wins."""
    req: Dict[str, Optional[str]] = {}
    for line in sbatch_directives(script_text):
        for name, value in parse_directive(line):
            req[name] = value
    return req


def resource_key(script_text: str) -> Tuple:
    """What two scripts must share to run as tasks of one array: every
    directive except the job name and output file, compared as exact
    strings, plus whether stderr has a file of its own (-e)."""
    req = request_of(script_text)
    items = tuple(sorted((k, v or "") for k, v in req.items() if k not in _NOT_REQUEST))
    return items + ((("<separate-stderr>", "yes" if "--error" in req else "no"),))


def _shell_var(script_text: str, name: str) -> Optional[str]:
    m = None
    for m in re.finditer(rf"^\s*export\s+{name}=(\S*)", script_text, re.M):
        pass
    return m.group(1) if m else None


# --------------------------------------------------------------------------
# grouping

def single_reason(script_text: str) -> Optional[str]:
    """Why this script goes in as an ordinary job, or None."""
    req = request_of(script_text)
    if "--chdir" in req:
        return "it sets its own working directory (-D/--chdir)"
    if "--dependency" in req:
        return "it waits on another job (-d/--dependency)"
    return None


def plan_groups(members: Sequence[ArrayMember]
                ) -> Tuple[List[ArrayGroup], List[Tuple[ArrayMember, str]]]:
    """Split members into arrays (identical requests, two or more members)
    and ordinary jobs (with the reason). Raises ArrayRefusal."""
    seen_scratch: Dict[Tuple[str, str], ArrayMember] = {}
    groups: Dict[Tuple, ArrayGroup] = {}
    order: List[Tuple] = []
    singles: List[Tuple[ArrayMember, str]] = []
    for m in members:
        for p in (str(m.dir), str(m.script), m.job):
            if any(c in p for c in "\t\n'"):
                raise ArrayRefusal(f"{p!r} contains a tab, a newline or a quote; job arrays "
                                   f"need plain paths (submit without --array)")
        text = Path(m.script).read_text(errors="replace")
        req = request_of(text)
        if "--array" in req:
            raise ArrayRefusal(f"{Path(m.script).name} is a job array itself (#SBATCH --array); "
                               f"remove that line or submit without --array")
        scratch = (_shell_var(text, "scratch") or "", _shell_var(text, "JOB") or m.job)
        if scratch in seen_scratch:
            other = seen_scratch[scratch]
            raise ArrayRefusal(f"{other.script} and {m.script} would run in the same scratch "
                               f"directory ({scratch[0]}/{scratch[1]}); rename one deck")
        seen_scratch[scratch] = m
        reason = single_reason(text)
        if reason:
            singles.append((m, reason))
            continue
        key = resource_key(text)
        if key not in groups:
            lines = [l for l in sbatch_directives(text)
                     if not any(n in _NOT_REQUEST for n, _ in parse_directive(l))]
            groups[key] = ArrayGroup(key, [], lines, "--error" in req)
            order.append(key)
        groups[key].members.append(m)
    arrays = []
    for key in order:
        g = groups[key]
        if len(g.members) == 1:
            singles.append((g.members[0], "the only deck with this request: submitted as a "
                                          "normal job"))
        else:
            arrays.append(g)
    return arrays, singles


# --------------------------------------------------------------------------
# cluster limits (best effort)

def _run(cmd, **kw):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60, **kw)
    except (OSError, subprocess.SubprocessError):
        return None


def max_tasks_per_array() -> int:
    """min(max_array_tasks, MaxArraySize - 1, 99999) from `scontrol show
    config`, or DEFAULT_MAX_TASKS when it cannot be read."""
    res = _run(["scontrol", "show", "config"])
    if res is None or res.returncode != 0:
        return DEFAULT_MAX_TASKS
    limits = [MAX_TASK_INDEX + 1]
    m = re.search(r"max_array_tasks=(\d+)", res.stdout)
    if m:
        limits.append(int(m.group(1)))
    m = re.search(r"^\s*MaxArraySize\s*=\s*(\d+)", res.stdout, re.M)
    if m:
        limits.append(int(m.group(1)) - 1)
    if len(limits) == 1:
        return DEFAULT_MAX_TASKS
    return max(min(limits), 1)


def free_submit_slots(account: Optional[str]) -> Optional[Tuple[int, int]]:
    """(limit, used) of MaxSubmitJobs for this user and account, the
    tightest association that sets one, or None when it cannot be read."""
    user = os.environ.get("USER") or os.environ.get("LOGNAME")
    if not user:
        return None
    cmd = ["scontrol", "show", "assoc_mgr", f"users={user}", "flags=assoc"]
    if account:
        cmd.insert(4, f"accounts={account}")
    res = _run(cmd)
    if res is None or res.returncode != 0:
        return None
    best = None
    for limit, used in re.findall(r"MaxSubmitJobs=(\d+)\((\d+)\)", res.stdout):
        limit, used = int(limit), int(used)
        if best is None or limit - used < best[0] - best[1]:
            best = (limit, used)
    return best


def check_submit_limits(prepared: Sequence[PreparedArray]) -> None:
    """Refuse the whole wave up front when its tasks would not fit in the
    free MaxSubmitJobs slots (every task counts against the limit)."""
    per_account: Dict[Optional[str], int] = {}
    for p in prepared:
        acct = dict(p.group.key).get("--account") or None
        per_account[acct] = per_account.get(acct, 0) + len(p.members)
    for acct, count in per_account.items():
        slots = free_submit_slots(acct)
        if slots and count > slots[0] - slots[1]:
            limit, used = slots
            raise ArrayRefusal(
                f"{count} array task(s) for account {acct or '(default)'} do not fit: "
                f"MaxSubmitJobs is {limit} and {used} job(s) are already queued, so "
                f"{max(limit - used, 0)} slot(s) are free. Nothing was submitted; submit "
                f"fewer decks at a time")


# --------------------------------------------------------------------------
# files

def _q(text: str) -> str:
    return "'" + str(text) + "'"


def new_tag(arrays_dir: Path, k: int) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    while (arrays_dir / f"arr_{stamp}_{k}").exists():
        k += 1
    return f"arr_{stamp}_{k}"


def write_manifest(path: Path, members: Sequence[ArrayMember]) -> None:
    lines = [MANIFEST_HEADER]
    for i, m in enumerate(members):
        lines.append("\t".join([str(i), str(Path(m.dir).resolve()), m.job,
                                str(Path(m.script).resolve()), m.calc_id or "-"]))
    path.write_text("\n".join(lines) + "\n")


def read_manifest(path: Path) -> List[Dict[str, str]]:
    out = []
    for line in Path(path).read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        f = line.split("\t")
        if len(f) >= 5:
            out.append({"task": f[0], "dir": f[1], "job": f[2], "script": f[3],
                        "calc_id": f[4]})
    return out


def throttle_spec(n_tasks: int, max_running: Optional[int]) -> str:
    spec = f"0-{n_tasks - 1}"
    if max_running:
        spec += f"%{int(max_running)}"
    return spec


def wrapper_text(tag: str, array_dir: Path, manifest: Path, group: ArrayGroup,
                 job_name: str, array_spec: str) -> str:
    logs = Path(array_dir).resolve() / "logs"
    out = [
        "#!/bin/bash --login",
        f"#SBATCH -J {job_name}",
        f"#SBATCH -o {logs}/{tag}-%A_%a.o",
    ]
    if group.separate_stderr:
        out.append(f"#SBATCH -e {logs}/{tag}-%A_%a.e")
    out.extend(l.strip() for l in group.directive_lines)
    out.append(f"#SBATCH --array={array_spec}")
    ln_e = ""
    if group.separate_stderr:
        ln_e = (f'ln -sfn {_q(str(logs) + "/" + tag)}"-${{SLURM_ARRAY_JOB_ID}}_${{T}}.e" \\\n'
                f'        "$MACE_TASK_DIR/$MACE_TASK_JOB-${{SLURM_ARRAY_JOB_ID}}_${{T}}.e" '
                f'2>/dev/null\n')
    out.append(f"""# MACE job array: task i runs the job script on manifest line i, in its own
# directory, exactly as sbatch would run that script on its own. The
# #SBATCH lines of that script are comments here; this array asks for the
# same request (MACE only groups identical ones).
MANIFEST={_q(Path(manifest).resolve())}
T=${{SLURM_ARRAY_TASK_ID:?not a SLURM array task}}
LINE=$(awk -F'\\t' -v t="$T" '$1==t {{print; exit}}' "$MANIFEST")
[ -n "$LINE" ] || {{ echo "mace array: no manifest line for task $T"; exit 1; }}
IFS=$'\\t' read -r _ MACE_TASK_DIR MACE_TASK_JOB MACE_TASK_SCRIPT MACE_TASK_CALC <<< "$LINE"
# the task's SLURM log, also in the material directory under the name MACE
# reads job ids from (<job>-<array>_<task>.o)
ln -sfn {_q(str(logs) + "/" + tag)}"-${{SLURM_ARRAY_JOB_ID}}_${{T}}.o" \\
        "$MACE_TASK_DIR/$MACE_TASK_JOB-${{SLURM_ARRAY_JOB_ID}}_${{T}}.o" 2>/dev/null
{ln_e}cd "$MACE_TASK_DIR" || exit 1
export SLURM_SUBMIT_DIR="$MACE_TASK_DIR"          # the script's DIR=$SLURM_SUBMIT_DIR
export MACE_ARRAY_TASK="${{SLURM_ARRAY_JOB_ID}}_${{T}}"
export MACE_ARRAY_RAW_ID="$SLURM_JOB_ID"          # a child job has another SLURM_JOB_ID
export MACE_ARRAY_MANIFEST="$MANIFEST"
export MACE_ARRAY_CALC_ID="$MACE_TASK_CALC"       # '-' when not recorded in a database
exec bash --login "$MACE_TASK_SCRIPT"
""")
    return "\n".join(out)


def prepare_arrays(groups: Sequence[ArrayGroup], base_dir: Path, label: str,
                   max_running: Optional[int] = None,
                   max_tasks: Optional[int] = None) -> List[PreparedArray]:
    """Write the manifest and wrapper of every array (groups larger than the
    cluster's task limit are split into several arrays)."""
    arrays_dir = Path(base_dir) / ARRAYS_DIRNAME
    limit = max_tasks or max_tasks_per_array()
    limit = max(1, min(limit, MAX_TASK_INDEX + 1))
    prepared = []
    k = 0
    for g in groups:
        for start in range(0, len(g.members), limit):
            chunk = g.members[start:start + limit]
            assert len(chunk) - 1 <= MAX_TASK_INDEX
            arrays_dir.mkdir(parents=True, exist_ok=True)
            tag = new_tag(arrays_dir, k)
            k = int(tag.rsplit("_", 1)[1]) + 1
            adir = arrays_dir / tag
            (adir / "logs").mkdir(parents=True)
            manifest = adir / "manifest.tsv"
            write_manifest(manifest, chunk)
            job_name = f"mace_arr_{label}_{tag.rsplit('_', 1)[1]}"
            spec = throttle_spec(len(chunk), max_running)
            wrapper = adir / f"{tag}.sh"
            sub = ArrayGroup(g.key, chunk, g.directive_lines, g.separate_stderr)
            wrapper.write_text(wrapper_text(tag, adir, manifest, sub, job_name, spec))
            wrapper.chmod(0o755)
            prepared.append(PreparedArray(tag, adir, wrapper, manifest, chunk, sub,
                                          job_name, spec))
    return prepared


def _write_submitted(p: PreparedArray, argv: List[str], released: bool) -> None:
    data = {
        "version": 1, "tag": p.tag, "array_job_id": p.array_job_id, "argv": argv,
        "array_dir": str(p.array_dir.resolve()), "wrapper": str(p.wrapper.resolve()),
        "array": p.array_spec, "request": [l.strip() for l in p.group.directive_lines],
        "members": [{"task": i, "dir": str(Path(m.dir).resolve()), "job": m.job,
                     "script": str(Path(m.script).resolve()), "calc_id": m.calc_id}
                    for i, m in enumerate(p.members)],
        "submitted_at": datetime.now().isoformat(), "held": True, "released": released,
    }
    (p.array_dir / "submitted.json").write_text(json.dumps(data, indent=1) + "\n")


# --------------------------------------------------------------------------
# submission

def release_array(array_job_id: str) -> bool:
    """`scontrol release A` - one of the two actions MACE takes on a bare
    array id (the other is abort_held_arrays)."""
    res = _run(["scontrol", "release", str(array_job_id)])
    return bool(res is not None and res.returncode == 0)


def abort_held_arrays(array_ids: Sequence[str]) -> List[str]:
    """Remove arrays this invocation submitted held, after checking that
    every task of each is still pending and held. Returns the ids removed."""
    removed = []
    for a in array_ids:
        res = _run(["squeue", "-h", "-r", "-j", str(a), "-o", "%T,%r"],
                   env=dict(os.environ, SLURM_BITSTR_LEN="0"))
        lines = [l.strip() for l in (res.stdout if res else "").splitlines() if l.strip()]
        if res is None or res.returncode != 0 or not lines or \
                any(l != "PENDING,JobHeldUser" for l in lines):
            print(f"  Could not confirm that job array {a} is still held; it was left as it is. "
                  f"Check it with: squeue -r -j {a}   (scancel {a} removes all of it)")
            continue
        res = _run(["scancel", str(a)])
        if res is not None and res.returncode == 0:
            removed.append(str(a))
            print(f"  Removed held job array {a} (nothing of it had started)")
        else:
            print(f"  Could not remove held job array {a}: "
                  f"{(res.stderr if res else '').strip()}  (run: scancel {a})")
    return removed


def submit_prepared(prepared: Sequence[PreparedArray],
                    record_rows: Optional[Callable[[Sequence[PreparedArray]], None]] = None
                    ) -> List[PreparedArray]:
    """Submit held, record, release (see the module docstring).

    ``record_rows(prepared)`` writes every database row of the wave (in one
    transaction); it runs after every sbatch succeeded and before any array
    is released. Raises ArraySubmitError (with the held arrays removed) when
    an sbatch or the recording fails.
    """
    check_submit_limits(prepared)
    submitted: List[PreparedArray] = []
    for p in prepared:
        argv = ["sbatch", "--hold", p.wrapper.name]
        res = _run(argv, cwd=str(p.array_dir))
        m = SUBMITTED_RE.search(res.stdout) if (res is not None and res.returncode == 0) else None
        if not m:
            err = ((res.stderr or res.stdout).strip() if res is not None
                   else "sbatch could not be run")
            abort_held_arrays([q.array_job_id for q in submitted])
            raise ArraySubmitError(f"sbatch refused job array {p.tag} "
                                   f"({len(p.members)} tasks): {err}")
        p.array_job_id = m.group(1)
        submitted.append(p)
        _write_submitted(p, argv, released=False)
    if record_rows is not None:
        try:
            record_rows(submitted)
        except Exception as e:
            abort_held_arrays([q.array_job_id for q in submitted])
            raise ArraySubmitError(f"the submissions could not be recorded ({e}); the held "
                                   f"arrays were removed")
    for p in submitted:
        if release_array(p.array_job_id):
            _write_submitted(p, ["sbatch", "--hold", p.wrapper.name], released=True)
        else:
            print(f"  Could not release job array {p.array_job_id}; its rows are recorded. "
                  f"Release it with: scontrol release {p.array_job_id}")
    return submitted


# --------------------------------------------------------------------------
# generators

def sbatch_shim() -> Tuple[Dict[str, str], Path]:
    """An environment whose ``sbatch`` refuses to submit (prints
    MACE_ARRAY_SBATCH_SHIM, leaves a marker file, exits 1), for running a job
    script generator in array mode: a generator that submits its own script
    must not do so here. Returns (env, marker); remove it with cleanup_shim."""
    import tempfile
    shim_dir = Path(tempfile.mkdtemp(prefix="mace_array_shim_"))
    marker = shim_dir / "hit"
    shim = shim_dir / "sbatch"
    shim.write_text("#!/bin/sh\n"
                    f"echo {SHIM_MESSAGE}\n"
                    f"touch {shlex.quote(str(marker))}\n"
                    "exit 1\n")
    shim.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = f"{shim_dir}:{env.get('PATH', '')}"
    return env, marker


def generator_outcome(result, marker: Path) -> Tuple[str, Optional[str]]:
    """After a generator ran with the shim: ('shim', None) when it tried to
    submit through sbatch on PATH; ('submitted', id) when it submitted
    anyway (sbatch by absolute path); ('generated', None) otherwise."""
    out = (getattr(result, "stdout", "") or "") + (getattr(result, "stderr", "") or "")
    m = SUBMITTED_RE.search(out)
    if m:
        return "submitted", m.group(1)
    if marker.exists() or SHIM_MESSAGE in out:
        return "shim", None
    return "generated", None


def cleanup_shim(marker: Path) -> None:
    import shutil
    shutil.rmtree(Path(marker).parent, ignore_errors=True)


def submit_untracked_arrays(names: Sequence[str], data_folder, generate: Callable,
                            nosubmit: bool, max_running: Optional[int], label: str,
                            sbatch_single: Optional[Callable] = None,
                            say: Callable[[str], None] = print) -> List[str]:
    """`mace submit --array` without --track, for the decks of one folder.

    ``generate(name, env=None, outcome=None)`` is the submit module's own
    generate_or_use_script (it writes or reuses <name>.sh in data_folder and
    puts the generator's result in ``outcome['result']``). Returns the job
    ids submitted (A_T for array tasks)."""
    data_folder = Path(data_folder).resolve()
    original = os.getcwd()
    members: List[ArrayMember] = []
    regenerate: List[Tuple[str, str]] = []
    job_ids: List[str] = []

    def single(script_name: str) -> Optional[str]:
        res = _run(["sbatch", script_name], cwd=str(data_folder))
        if res is None or res.returncode != 0:
            say(f"  Warning: Failed to submit {script_name}")
            if res is not None and res.stderr:
                say(f"    {res.stderr.strip()}")
            return None
        m = SUBMITTED_RE.search(res.stdout)
        if m:
            say(f"  ✓ Submitted {script_name} as a normal job (job {m.group(1)})")
            return m.group(1)
        say(f"  ✓ Submitted {script_name} as a normal job")
        return None

    sbatch_single = sbatch_single or single
    try:
        os.chdir(data_folder)
        for name in names:
            say(f"Preparing: {name}")
            env, marker = sbatch_shim()
            outcome: Dict = {}
            try:
                script, _ = generate(name, env=env, outcome=outcome)
                kind, value = (generator_outcome(outcome["result"], marker)
                               if outcome.get("result") is not None else ("generated", None))
            finally:
                cleanup_shim(marker)
            if kind == "submitted":
                job_ids.append(value)
                say(f"  {name}: its job script generator submitted the job itself "
                    f"(job {value}); not part of an array")
                continue
            if kind == "shim":
                regenerate.append((name, "its job script generator submits the job itself"))
                continue
            if not script:
                say(f"  Error: Failed to generate/find script for {name}")
                continue
            members.append(ArrayMember(dir=data_folder, job=name, script=Path(script)))

        try:
            arrays, singles = plan_groups(members)
            prepared = prepare_arrays(arrays, data_folder, label, max_running) if arrays else []
        except ArrayRefusal as e:
            say(f"Error: {e}")
            raise SystemExit(1)
        if nosubmit:
            for p in prepared:
                say(f"  Job array {p.tag}: {len(p.members)} task(s), each "
                    f"{describe_request(p.group)}")
                say(f"    to submit it: cd {p.array_dir} && sbatch {p.wrapper.name}")
            for m, reason in singles:
                say(f"  {m.job}: {reason} (not submitted: --nosubmit)")
            for name, reason in regenerate:
                say(f"  {name}: {reason} (not submitted: --nosubmit)")
            return []
        try:
            subs = submit_prepared(prepared) if prepared else []
        except (ArrayRefusal, ArraySubmitError) as e:
            say(f"Error: {e}")
            raise SystemExit(1)
        for p in subs:
            job_ids.extend(p.task_id(i) for i in range(len(p.members)))
        for m, _ in singles:
            jid = sbatch_single(Path(m.script).name)
            if jid:
                job_ids.append(jid)
        for name, _ in regenerate:
            script, _ = generate(name)
            if script:
                jid = sbatch_single(Path(script).name)
                if jid:
                    job_ids.append(jid)
        summarize(subs, singles + [(ArrayMember(data_folder, n, data_folder / f"{n}.sh"), r)
                                   for n, r in regenerate], max_running, say)
        return job_ids
    finally:
        os.chdir(original)


# --------------------------------------------------------------------------
# messages

def describe_request(group: ArrayGroup) -> str:
    keep = []
    for line in group.directive_lines:
        for name, value in parse_directive(line):
            if name in ("--ntasks", "--nodes", "--cpus-per-task", "--mem", "--mem-per-cpu",
                        "--time", "--account", "--partition", "--qos", "--constraint",
                        "--exclude", "--gres"):
                keep.append(f"{name}={value}" if value is not None else name)
    return " ".join(keep) or "(no #SBATCH request)"


def summarize(prepared: Sequence[PreparedArray], singles: Sequence[Tuple[ArrayMember, str]],
              max_running: Optional[int], say: Callable[[str], None] = print) -> None:
    for p in prepared:
        ids = f"{p.array_job_id}_[0-{len(p.members) - 1}]" if p.array_job_id else p.tag
        say(f"  Job array {ids}: {len(p.members)} task(s), each {describe_request(p.group)}")
        for i, m in enumerate(p.members):
            tid = p.task_id(i) if p.array_job_id else f"task {i}"
            say(f"    {tid}  {m.job}")
        say(f"    logs: {p.array_dir / 'logs'}")
    for m, reason in singles:
        say(f"  {m.job}: {reason}")
    if prepared and max_running:
        k = len(prepared)
        if k > 1:
            say(f"  At most {max_running} running per array ({k} arrays, up to "
                f"{max_running * k} at once).")
        else:
            say(f"  At most {max_running} task(s) running at once.")
    if prepared:
        say("  Cancel one task with scancel <array>_<task>; scancel <array> cancels every task.")
