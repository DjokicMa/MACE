"""SLURM job ids, plain and job-array, in one place.

A plain job has one id (``12345678``). A job array has one array id ``A`` and
one id per task, ``A_T`` (``12345678_7``). Measured on SLURM 26.05:

- squeue without ``-r`` folds an array's pending tasks into one token,
  ``A_[2-3%2]``, and once the pending set has gaps the brackets hold commas,
  ``A_[2,4-5%2]``. Splitting such a line on every comma cut it in the middle
  of the task list;
- squeue prints at most 64 characters of a task expression unless
  ``SLURM_BITSTR_LEN=0`` is set; a longer one is cut and ends in ``...``;
  sacct's fixed-width JobID column is cut and marked with ``+`` (``-P``
  never cuts);
- a running task has a raw job id of its own (``SLURM_JOB_ID``), and the last
  task to start gets the array id ``A`` itself as its raw id. ``scancel A``
  cancels the whole array, so MACE never stores or cancels a raw id: every
  per-job action addresses ``A_T``;
- a job submitted from inside a task inherits all ``SLURM_ARRAY_*`` and
  ``MACE_ARRAY_*`` variables; only its ``SLURM_JOB_ID`` is its own.
"""
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

ARRAY_TASK_RE = re.compile(r"^(\d+)_(\d+)$")
_BRACKET_RE = re.compile(r"^(\d+)_\[([^\[\]]*)\]$")
# Expanding more ids than this from one token is refused (treated as unknown).
_MAX_EXPAND = 200000


class Truncated:
    """A task expression SLURM printed cut short: nothing can be inferred
    about the tasks of this array from it."""

    __slots__ = ("array_id",)

    def __init__(self, array_id: str):
        self.array_id = array_id

    def __repr__(self):
        return f"Truncated({self.array_id!r})"

    def __eq__(self, other):
        return isinstance(other, Truncated) and other.array_id == self.array_id

    def __hash__(self):
        return hash(("Truncated", self.array_id))


def split_job_id(job_id) -> Tuple[str, Optional[str]]:
    """``"A_T" -> ("A", "T")``; a plain id gives ``(id, None)``."""
    text = str(job_id or "").strip()
    m = ARRAY_TASK_RE.match(text)
    if m:
        return m.group(1), m.group(2)
    return text, None


def is_array_task_id(job_id) -> bool:
    return bool(ARRAY_TASK_RE.match(str(job_id or "").strip()))


def split_squeue_line(line: str, nfields: int) -> List[str]:
    """Split one squeue line on the commas that are outside ``[...]``, at most
    ``nfields - 1`` times (the last field keeps any commas of its own, e.g. a
    job name). For a line without brackets this is ``line.split(',')`` for
    its first ``nfields - 1`` fields."""
    parts, cur, depth = [], [], 0
    for ch in line:
        if ch == "[":
            depth += 1
        elif ch == "]" and depth > 0:
            depth -= 1
        if ch == "," and depth == 0 and len(parts) < nfields - 1:
            parts.append("".join(cur))
            cur = []
            continue
        cur.append(ch)
    parts.append("".join(cur))
    if depth > 0 and nfields > 1:
        # A task list SLURM cut short ("A_[1,3,5,...") never closes its
        # bracket: the token ends at the cut mark.
        cut = line.find("...")
        if cut >= 0 and line[cut + 3:cut + 4] == ",":
            return [line[:cut + 3]] + line[cut + 4:].split(",", nfields - 2)
        return line.rsplit(",", nfields - 1)
    return parts


def expand_queue_id(token: str) -> Union[List[str], Truncated]:
    """The job ids a squeue/sacct id token stands for.

    ``A_[2-3%2]`` -> [A_2, A_3]; ``A_[123]`` -> [A_123]; ``A_[1,4-6%3]`` ->
    [A_1, A_4, A_5, A_6]; plain ids and ``A_T`` pass through. A token SLURM
    cut short (``...`` in it, a trailing ``+``, unbalanced brackets) gives
    ``Truncated(A)``.
    """
    t = str(token or "").strip()
    head = re.match(r"^(\d+)", t)
    array_id = head.group(1) if head else t
    if "..." in t or t.endswith("+"):
        return Truncated(array_id)
    m = _BRACKET_RE.match(t)
    if not m:
        if "[" in t or "]" in t:
            return Truncated(array_id)
        return [t] if t else []
    array_id, body = m.groups()
    body = body.split("%", 1)[0]
    ids: List[str] = []
    try:
        for part in body.split(","):
            part = part.strip()
            if not part:
                continue
            step = 1
            if ":" in part:
                part, step_text = part.split(":", 1)
                step = int(step_text)
            if "-" in part:
                lo, hi = (int(x) for x in part.split("-", 1))
                if hi < lo or step < 1 or (hi - lo) // step + len(ids) > _MAX_EXPAND:
                    return Truncated(array_id)
                ids.extend(f"{array_id}_{i}" for i in range(lo, hi + 1, step))
            else:
                ids.append(f"{array_id}_{int(part)}")
    except ValueError:
        return Truncated(array_id)
    return ids


def slurm_env(base: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """The environment for MACE's own squeue/sacct calls: the caller's, plus
    ``SLURM_BITSTR_LEN=0`` so array task expressions print in full."""
    env = dict(os.environ if base is None else base)
    env["SLURM_BITSTR_LEN"] = "0"
    return env


def verified_array_task(env: Optional[Dict[str, str]] = None) -> Optional[str]:
    """``A_T`` when this process runs inside MACE array task A_T itself.

    The MACE array wrapper exports ``MACE_ARRAY_TASK=A_T`` and
    ``MACE_ARRAY_RAW_ID=$SLURM_JOB_ID``. A job submitted from inside the task
    inherits both (and every ``SLURM_ARRAY_*`` variable, measured), but has a
    ``SLURM_JOB_ID`` of its own, so the task is only this process's own when
    the raw id it recorded is the current ``SLURM_JOB_ID``.
    """
    env = os.environ if env is None else env
    task = (env.get("MACE_ARRAY_TASK") or "").strip()
    raw = (env.get("MACE_ARRAY_RAW_ID") or "").strip()
    job = (env.get("SLURM_JOB_ID") or "").strip()
    if not (task and raw and job and raw == job and is_array_task_id(task)):
        return None
    arr, idx = env.get("SLURM_ARRAY_JOB_ID"), env.get("SLURM_ARRAY_TASK_ID")
    if arr and idx is not None and task != f"{arr.strip()}_{str(idx).strip()}":
        return None
    return task


def own_job_ids(env: Optional[Dict[str, str]] = None) -> Set[str]:
    """The job ids that belong to the process running this code: its
    ``SLURM_JOB_ID`` and, inside a MACE array task, that task's ``A_T``."""
    env = os.environ if env is None else env
    ids = set()
    job = (env.get("SLURM_JOB_ID") or "").strip()
    if job:
        ids.add(job)
    task = verified_array_task(env)
    if task:
        ids.add(task)
    return ids


def manifest_calc_id(manifest: Optional[str], task_index: str) -> Optional[str]:
    """The calc_id on line ``task_index`` of a MACE array manifest."""
    if not manifest:
        return None
    try:
        for line in Path(manifest).read_text().splitlines():
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if fields[0] == str(task_index) and len(fields) >= 5:
                return fields[4] if fields[4] not in ("", "-") else None
    except OSError:
        return None
    return None


def own_rows(db, env: Optional[Dict[str, str]] = None) -> List[Dict]:
    """The calculation rows of this process: by own job id, and inside a
    verified MACE array task also the row the manifest names for the task
    (so a task that ends before its row carries A_T still finds it)."""
    env = os.environ if env is None else env
    rows, seen = [], set()
    for job_id in sorted(own_job_ids(env)):
        try:
            row = db.get_calculation_by_slurm_id(job_id)
        except Exception:
            row = None
        if row and row["calc_id"] not in seen:
            rows.append(row)
            seen.add(row["calc_id"])
    task = verified_array_task(env)
    calc_id = (env.get("MACE_ARRAY_CALC_ID") or "").strip()
    if task and calc_id and calc_id != "-" and calc_id not in seen:
        _, idx = split_job_id(task)
        if manifest_calc_id(env.get("MACE_ARRAY_MANIFEST"), idx) == calc_id:
            try:
                row = db.get_calculation(calc_id)
            except Exception:
                row = None
            if row and str(row.get("slurm_job_id") or "") in ("", task):
                rows.append(row)
    return rows


def _array_ids_in_submitted(search_dirs: Iterable) -> Set[str]:
    ids = set()
    for d in search_dirs or ():
        if not d:
            continue
        d = Path(d)
        for base in {d, d.parent}:
            try:
                files = list(base.glob(".mace_arrays/*/submitted.json"))
            except OSError:
                continue
            for f in files:
                try:
                    data = json.loads(f.read_text())
                except (OSError, ValueError):
                    continue
                for a in data.get("arrays", []) or []:
                    if a.get("array_job_id"):
                        ids.add(str(a["array_job_id"]))
                if data.get("array_job_id"):
                    ids.add(str(data["array_job_id"]))
    return ids


def scancel_refusal(job_id, db=None, search_dirs: Sequence = ()) -> Optional[str]:
    """Why ``scancel <job_id>`` must not run, or None.

    ``scancel A`` cancels every task of array A, so a bare id is refused when
    it is the array id of any task MACE recorded (database rows ``A_T``) or of
    any array MACE submitted (``.mace_arrays/*/submitted.json`` in the given
    directories or their parents, which covers untracked submissions).
    """
    text = str(job_id or "").strip()
    if not text:
        return "no job id"
    if "[" in text or "]" in text:
        return f"{text} is a task range of a job array; MACE cancels single tasks only"
    if is_array_task_id(text):
        return None
    if db is not None:
        try:
            with db._get_connection() as conn:
                hit = conn.execute(
                    "SELECT slurm_job_id FROM calculations WHERE slurm_job_id LIKE ? "
                    "ESCAPE '\\' LIMIT 1", (f"{text}\\_%",)).fetchone()
            if hit:
                return (f"{text} is the id of job array {text} (task {hit[0]} is recorded): "
                        f"scancel {text} would cancel every task; cancel {hit[0]} instead")
        except Exception:
            pass
    if text in _array_ids_in_submitted(search_dirs):
        return (f"{text} is the id of a job array MACE submitted: scancel {text} would "
                f"cancel every task; cancel single tasks ({text}_<task>) instead")
    return None


def safe_scancel(job_id, db=None, search_dirs: Sequence = ()) -> subprocess.CompletedProcess:
    """``scancel <job_id>``, exactly as before for a plain id or ``A_T``;
    refused (returncode 1, reason in stderr, nothing run) for a bare array id
    or a task range."""
    text = str(job_id or "").strip()
    reason = scancel_refusal(text, db=db, search_dirs=search_dirs)
    if reason:
        return subprocess.CompletedProcess(["scancel", text], 1, "", f"refused: {reason}")
    return subprocess.run(["scancel", text], capture_output=True, text=True)


def safe_scancel_many(job_ids: Sequence, db=None, search_dirs: Sequence = ()
                      ) -> Tuple[Optional[subprocess.CompletedProcess], List[Tuple[str, str]]]:
    """``scancel <ids...>`` for the ids that may be canceled (one call, as
    before), and the refused ones with their reasons."""
    allowed, refused = [], []
    for jid in job_ids:
        reason = scancel_refusal(jid, db=db, search_dirs=search_dirs)
        if reason:
            refused.append((str(jid), reason))
        else:
            allowed.append(str(jid))
    if not allowed:
        return None, refused
    return subprocess.run(["scancel"] + allowed, capture_output=True, text=True), refused


# How long a task must stay unseen (no queue presence of its array, no
# accounting record, no output of its own) on two sweeps before it is taken
# as canceled while pending: SLURM keeps no record of such a task anywhere.
ABSENT_WINDOW_SECONDS = 600


def parse_time_left(text: str) -> Optional[int]:
    """squeue %L (``D-HH:MM:SS``, ``HH:MM:SS``, ``MM:SS``) in seconds; None for
    UNLIMITED, NOT_SET or anything unreadable."""
    t = (text or "").strip()
    m = re.match(r"^(?:(\d+)-)?(?:(\d+):)?(\d+):(\d+)$", t)
    if not m:
        return None
    days, hours, minutes, seconds = (int(x) if x else 0 for x in m.groups())
    return ((days * 24 + hours) * 60 + minutes) * 60 + seconds


def remaining_walltime_seconds(env: Optional[Dict[str, str]] = None) -> Optional[int]:
    """Time left of the job this process runs in (squeue %L), or None."""
    env = os.environ if env is None else env
    job = (env.get("SLURM_JOB_ID") or "").strip()
    if not job:
        return None
    try:
        res = subprocess.run(["squeue", "-h", "-j", job, "-o", "%L"], capture_output=True,
                             text=True, timeout=30, env=slurm_env())
    except (OSError, subprocess.SubprocessError):
        return None
    if res.returncode != 0:
        return None
    return parse_time_left(res.stdout.strip().splitlines()[0] if res.stdout.strip() else "")


def queue_states_of_array(array_id: str) -> Union[Dict[str, str], Truncated, None]:
    """{A_T: state} for every task of array A that squeue still lists ({} when
    none), Truncated when a token was cut short, None when squeue failed."""
    try:
        res = subprocess.run(["squeue", "-h", "-r", "-j", str(array_id), "-o", "%i,%T"],
                             capture_output=True, text=True, timeout=30, env=slurm_env())
    except (OSError, subprocess.SubprocessError):
        return None
    if res.returncode != 0:
        # squeue -j on an id the controller has forgotten: "Invalid job id"
        if "invalid job id" in (res.stderr or "").lower():
            return {}
        return None
    states: Dict[str, str] = {}
    for line in res.stdout.splitlines():
        if not line.strip():
            continue
        parts = split_squeue_line(line.strip(), 2)
        ids = expand_queue_id(parts[0])
        if isinstance(ids, Truncated):
            return ids
        for i in ids:
            states[i] = parts[1].strip() if len(parts) > 1 else ""
    return states


def accounting_state_of_task(task_id: str) -> Union[str, Truncated, None]:
    """The state sacct records for task A_T, read from ``sacct -j A --array``
    (one line per task; a bracket record is expanded). '' when sacct has
    records of A but none for T, None when sacct knows nothing about A or
    failed, Truncated when a JobID was cut short."""
    array_id, _ = split_job_id(task_id)
    try:
        res = subprocess.run(["sacct", "-j", array_id, "-X", "-P", "-n", "--array",
                              "--format=JobID,State"], capture_output=True, text=True,
                             timeout=30, env=slurm_env())
    except (OSError, subprocess.SubprocessError):
        return None
    if res.returncode != 0:
        return None
    lines = [l.strip() for l in res.stdout.splitlines() if l.strip()]
    if not lines:
        return None
    state = ""
    for line in lines:
        jid, _, st = line.partition("|")
        ids = expand_queue_id(jid)
        if isinstance(ids, Truncated):
            return ids
        if task_id in ids:
            # the last record is the current one (a requeue keeps the old)
            state = (st.split() or [""])[0].rstrip("+")
    return state
