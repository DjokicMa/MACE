"""Your SLURM jobs in the GUI (PLAN_submit T5, X7): the Jobs panel's list,
finished-job events, and a guarded cancel.

* One own-jobs query (``squeue -h -r -u <user>``, the shared reader's
  format, below), cached server-wide and shared with the Cluster view: at
  most one call per 15 s on Refresh (60 s otherwise), and the panel's poll
  (``cluster.pollSeconds``, default 60, 30..300) asks only while the Jobs
  tab is shown or a job this session submitted is pending or running. Never
  on a machine without squeue.
* A job seen pending or running that leaves the queue gives one ``sacct``
  call and one ``jobs.finished`` server event {id, name, state, workdir,
  out_rel} (the toast and the Problems entry listen to it).
* Cancel runs ``scancel A_T`` / ``scancel J`` only after five checks: the id
  is a job or a task id; a fresh squeue lists it among your own jobs; a bare
  id is not a job array (``squeue -r -j`` shows no ``<id>_<t>`` tasks, also
  for arrays MACE never recorded); it is not the job MACE GUI itself runs in;
  and MACE's own array guard (``slurm_ids.scancel_refusal``) agrees.

The own-jobs query goes through Area F's shared reader
(``mace.cluster.reader.squeue_own``: one cached squeue for the Jobs panel,
the Cluster view and ``mace cluster``); the module's own query below is the
fallback when that package is missing.
"""
from __future__ import annotations

import getpass
import os
import re
import sqlite3
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .childenv import login_env

SQUEUE_FORMAT = "%i|%j|%T|%M|%l|%D|%C|%R|%Z"
FIELDS = ("id", "name", "state", "elapsed", "limit", "nodes", "cpus", "reason", "workdir")
MIN_INTERVAL_S = 10.0
POLL_DEFAULT_S = 60
POLL_RANGE = (30, 300)          # cluster.pollSeconds, as the Cluster view (settings.py)
ID_RE = re.compile(r"^\d+(_\d+)?$")
ACTIVE = ("PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED", "REQUEUED", "RESIZING")


def _env() -> Dict[str, str]:
    from mace.queue.slurm_ids import outside_allocation_env
    from .submit import submit_env
    env = outside_allocation_env(submit_env())
    env["SLURM_BITSTR_LEN"] = "0"
    return env


def _which(tool: str, env: Dict[str, str]) -> Optional[str]:
    import shutil
    return shutil.which(tool, path=env.get("PATH", os.defpath))


def user() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return login_env().get("USER", "")


def _run(argv: List[str], env: Dict[str, str], timeout: float = 10.0) -> Optional[subprocess.CompletedProcess]:
    nice = _which("nice", env)
    cmd = ([nice, "-n", "10"] if nice else []) + argv
    try:
        return subprocess.run(cmd, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                              timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None


def parse_squeue(text: str) -> List[Dict[str, str]]:
    """Rows of SQUEUE_FORMAT output; a '|' inside the job name is kept (the
    name is everything between the id and the last seven fields)."""
    rows = []
    for line in (text or "").splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < len(FIELDS):
            continue
        if len(parts) > len(FIELDS):
            parts = [parts[0], "|".join(parts[1:len(parts) - 7])] + parts[len(parts) - 7:]
        rows.append({k: v.strip() for k, v in zip(FIELDS, parts)})
    return rows


class OwnJobs:
    """The cached own-jobs query (Area F's ``squeue_own`` contract)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.rows: List[Dict[str, str]] = []
        self.at = 0.0
        self.ok = False
        self.error = ""
        self.calls = 0

    def available(self) -> bool:
        return bool(_which("squeue", _env()))

    def get(self, max_age: float = POLL_DEFAULT_S, fresh: bool = False) -> Tuple[List[Dict[str, str]], float]:
        """(rows, polled_at): the cached rows when younger than ``max_age``;
        ``fresh`` asks SLURM now (at most once per 10 s unless the caller
        needs it for a cancel or a preview, ``fresh == 'force'``)."""
        if _shared_reader() is not None:
            return self._get_shared(max_age, fresh)
        with self.lock:
            age = time.time() - self.at
            if fresh != "force" and (age < MIN_INTERVAL_S or (not fresh and age < max_age)):
                return list(self.rows), self.at
            env = _env()
            exe = _which("squeue", env)
            if not exe:
                self.rows, self.at, self.ok, self.error = [], time.time(), False, "no squeue"
                return [], self.at
            self.calls += 1
            r = _run([exe, "-h", "-r", "-u", user(), "-o", SQUEUE_FORMAT], env)
            if r is None or r.returncode != 0:
                self.error = (r.stderr.strip() if r is not None else "squeue did not answer")[:300]
                self.at = time.time()
                return list(self.rows), self.at
            self.rows, self.at, self.ok, self.error = parse_squeue(r.stdout), time.time(), True, ""
            return list(self.rows), self.at


    def _get_shared(self, max_age: float, fresh) -> Tuple[List[Dict[str, str]], float]:
        """``get`` through the Cluster view's shared, cached reader (one squeue for both)."""
        from mace.cluster import reader
        with self.lock:
            if not self.available():
                self.rows, self.at, self.ok, self.error = [], time.time(), False, "no squeue"
                return [], self.at
            if fresh == "force":
                res = reader.squeue_own(fresh=True, tag="jobs")
            else:
                res = reader.squeue_own(max_age=max_age, refresh=bool(fresh), tag="jobs")
            if res.source in ("fetch", "fresh"):
                self.calls += 1
            if res.value is None:
                self.error = (res.error or "squeue did not answer")[:300]
                self.at = time.time()
                return list(self.rows), self.at
            self.rows = [_from_shared(r) for r in res.value]
            self.at, self.ok, self.error = res.fetched_at or time.time(), True, ""
            return list(self.rows), self.at


def _shared_reader():
    """Area F's reader (mace/cluster/reader.py), or None."""
    try:
        from mace.cluster import reader
        return reader
    except ImportError:
        return None


def _from_shared(r: Dict[str, Any]) -> Dict[str, str]:
    """A row of the shared reader's squeue format, in this module's field names
    (``reason`` is squeue's %R: the reason while pending, else the nodes)."""
    state = str(r.get("state") or "")
    return {"id": str(r.get("id") or ""), "name": str(r.get("name") or ""), "state": state,
            "elapsed": str(r.get("elapsed") or ""), "limit": str(r.get("time_limit") or ""),
            "nodes": str(r.get("nodes") or ""), "cpus": str(r.get("cpus") or ""),
            "reason": str((r.get("reason") if state == "PENDING" else r.get("nodelist")) or ""),
            "workdir": str(r.get("workdir") or "")}


def _inside(path: str, root: Path) -> bool:
    try:
        rp = os.path.realpath(path)
        rr = os.path.realpath(str(root))
        return rp == rr or rp.startswith(rr.rstrip(os.sep) + os.sep)
    except (OSError, ValueError):
        return False


def _rel(root: Path, p: Path) -> str:
    try:
        return Path(os.path.realpath(str(p))).relative_to(os.path.realpath(str(root))).as_posix()
    except ValueError:
        return ""


def tracked_rows(root: Path, workdirs: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """{slurm_job_id: {calc_id, status, calc_type}} from the folder's
    materials.db and workflow context databases (read-only)."""
    out: Dict[str, Dict[str, Any]] = {}
    dbs = set()
    for d in {root, *[Path(w) for w in workdirs if w and _inside(w, root)]}:
        for cand in [d / "materials.db", d.parent / "materials.db"]:
            if cand.is_file():
                dbs.add(cand)
        try:
            for ctx in d.glob(".mace_context_*"):
                dbs.update(x for x in ctx.rglob("*.db") if x.is_file())
        except OSError:
            pass
    for db in sorted(dbs):
        try:
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5)
            try:
                for calc_id, job, status, ctype in con.execute(
                        "SELECT calc_id, slurm_job_id, status, calc_type FROM calculations WHERE slurm_job_id IS NOT NULL"):
                    if job:
                        out[str(job)] = {"calc_id": calc_id, "status": status, "calc_type": ctype, "db": str(db)}
            finally:
                con.close()
        except sqlite3.Error:
            continue
    return out


def describe(root: Path, rows: List[Dict[str, str]], scope: str, own_ids: Sequence[str],
             manifests: Optional[Dict[str, Dict[str, str]]] = None) -> List[Dict[str, Any]]:
    """The Jobs panel's rows: scope "folder" keeps jobs whose working folder
    is inside the served folder; deck and output paths; array grouping."""
    out = []
    tracked = tracked_rows(root, [r["workdir"] for r in rows])
    for r in rows:
        inside = _inside(r["workdir"], root) if r["workdir"] else False
        if scope == "folder" and not inside:
            continue
        jid = r["id"]
        arr, _, task = jid.partition("_")
        wd = Path(r["workdir"]) if r["workdir"] else None
        deck = out_rel = None
        name = r["name"]
        log = (wd / f"{name}-{jid}.o") if wd else None        # the job script's "#SBATCH -o <name>-%J.o"
        if manifests and arr in manifests and task:
            m = manifests[arr].get(task)
            if m:
                name = m.get("job", name)
                wd = Path(m.get("dir") or (str(wd) if wd else ""))
                # an array task logs to .mace_arrays/<array>/logs/<tag>-A_T.o (job_array.wrapper_text)
                logs = Path(m.get("_array_dir") or "") / "logs"
                hits = sorted(logs.glob(f"*-{jid}.o")) if m.get("_array_dir") and logs.is_dir() else []
                log = hits[0] if hits else None
        log_rel = _rel(root, log) if log is not None and _inside(str(log), root) and log.is_file() else None
        if wd and inside:
            for ext in (".d12", ".d3"):
                if (wd / f"{name}{ext}").is_file():
                    deck = _rel(root, wd / f"{name}{ext}")
                    out_rel = _rel(root, wd / f"{name}.out")
                    break
        out.append({"id": jid, "array_id": arr if task else None, "task": task or None, "name": name,
                    "state": r["state"], "reason": r["reason"] if r["state"] == "PENDING" else "",
                    "node": r["reason"] if r["state"] != "PENDING" else "", "elapsed": r["elapsed"],
                    "limit": r["limit"], "nodes": r["nodes"], "cpus": r["cpus"],
                    "workdir_rel": _rel(root, wd) if wd and inside else None, "deck_rel": deck, "out_rel": out_rel,
                    "log_rel": log_rel or None,
                    "tracked": tracked.get(jid), "own_gui_job": jid in own_ids or arr in own_ids})
    return out


MANIFEST_TTL_S = 120.0
_MANIFESTS: Dict[str, Tuple[float, Dict[str, Dict[str, Dict[str, str]]]]] = {}
_MANIFESTS_LOCK = threading.Lock()


def forget_manifests() -> None:
    """A submit wrote new arrays: the next read walks the folder again."""
    with _MANIFESTS_LOCK:
        _MANIFESTS.clear()


def manifests_in(root: Path, max_files: int = 400, max_age: float = MANIFEST_TTL_S) -> Dict[str, Dict[str, Dict[str, str]]]:
    """{array id: {task: manifest row}} from the folder's .mace_arrays/*/submitted.json.
    The walk of the folder is kept ``max_age`` seconds (the poll must not walk
    a large tree every time; a submit from the GUI clears it)."""
    key = str(root)
    with _MANIFESTS_LOCK:
        hit = _MANIFESTS.get(key)
        if hit is not None and time.time() - hit[0] < max_age:
            return hit[1]
    out = _manifests_walk(root, max_files)
    with _MANIFESTS_LOCK:
        _MANIFESTS[key] = (time.time(), out)
    return out


def _manifests_walk(root: Path, max_files: int) -> Dict[str, Dict[str, Dict[str, str]]]:
    import json
    out: Dict[str, Dict[str, Dict[str, str]]] = {}
    n = 0
    try:
        files = root.rglob(".mace_arrays/*/submitted.json")
        for f in files:
            n += 1
            if n > max_files:
                break
            try:
                data = json.loads(f.read_text())
            except (OSError, ValueError):
                continue
            a = data.get("array_job_id")
            if a:
                out[str(a)] = {str(m.get("task")): {**m, "_array_dir": str(f.parent)} for m in data.get("members") or []}
    except OSError:
        pass
    return out


def sacct_state(job_id: str) -> Optional[str]:
    env = _env()
    exe = _which("sacct", env)
    if not exe:
        return None
    r = _run([exe, "-n", "-P", "-X", "-j", job_id, "-o", "State"], env, timeout=15)
    if r is None or r.returncode != 0:
        return None
    for line in r.stdout.splitlines():
        s = line.strip().split()[0] if line.strip() else ""
        if s:
            return s.rstrip("+")
    return None


def recent_finished(root: Path) -> List[Dict[str, str]]:
    """Own jobs that ended in the last two days, inside the folder (sacct)."""
    env = _env()
    exe = _which("sacct", env)
    if not exe:
        return []
    r = _run([exe, "-P", "-n", "-X", "-u", user(), "-S", "now-2days", "-o", "JobID,JobName,State,Elapsed,WorkDir"],
             env, timeout=20)
    if r is None or r.returncode != 0:
        return []
    out = []
    for line in r.stdout.splitlines():
        f = line.split("|")
        if len(f) < 5 or not _inside(f[4], root):
            continue
        if f[2].split()[0] in ACTIVE:
            continue
        out.append({"id": f[0], "name": f[1], "state": f[2].split()[0], "elapsed": f[3],
                    "workdir_rel": _rel(root, Path(f[4]))})
    return out[-200:]


def recent_with_tracking(root: Path) -> List[Dict[str, Any]]:
    """recent_finished, each job with its tracked row ({calc_id, status,
    calc_type}) when a database of the folder has it, and whether Recover
    applies (the row is failed)."""
    rows = recent_finished(root)
    tracked = tracked_rows(root, [str(root / r["workdir_rel"]) for r in rows if r.get("workdir_rel")])
    for r in rows:
        t = tracked.get(r["id"])
        r["tracked"] = {k: t[k] for k in ("calc_id", "status", "calc_type")} if t else None
        r["recoverable"] = bool(t and t.get("status") == "failed")
    return rows


class Watcher:
    """Finished transitions: the ids seen pending/running in the previous
    poll that are gone now give one ``jobs.finished`` event each."""

    def __init__(self, emit: Callable[[str, Dict[str, Any]], None]):
        self.emit = emit
        self.lock = threading.Lock()
        self.seen: Dict[str, Dict[str, Any]] = {}
        self.events = 0
        # a job the server announced already (the Cluster view's poller saw it first): no sacct, no event
        self.skip: Callable[[str], bool] = lambda jid: False
        # called with (root, rows) after every poll (the Workflows view's WorkflowWatcher)
        self.hooks: List[Callable[[Path, List[Dict[str, Any]]], None]] = []

    def update(self, root: Path, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        now_active = {r["id"]: r for r in rows if r["state"] in ACTIVE}
        with self.lock:
            gone = [r for jid, r in self.seen.items() if jid not in now_active]
            self.seen = now_active
        events = []
        for r in gone:
            if self.skip(r["id"]):
                continue
            state = sacct_state(r["id"])
            ev = {"id": r["id"], "name": r["name"], "state": state or "left the queue",
                  "known": state is not None, "workdir": r.get("workdir_rel"), "out_rel": r.get("out_rel")}
            self.events += 1
            events.append(ev)
            self.emit("jobs.finished", ev)
        for hook in list(self.hooks):
            try:
                hook(root, rows)
            except Exception:
                pass
        return events


class Jobs:
    """The server's jobs state: the own-jobs cache, the watcher and the poll thread."""

    def __init__(self, root: Path, emit: Callable[[str, Dict[str, Any]], None],
                 session_jobs: Optional[Dict[str, Dict[str, Any]]] = None,
                 poll_seconds: Callable[[], int] = lambda: POLL_DEFAULT_S):
        self.root = Path(root)
        self.own = OwnJobs()
        self.watcher = Watcher(emit)
        self.session_jobs = session_jobs if session_jobs is not None else {}
        self.poll_seconds = poll_seconds
        self.visible_until = 0.0
        self.last_rows: Dict[str, Dict[str, Any]] = {}   # id -> this folder's row, as last described
        self.stop = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

    def interval(self) -> int:
        try:
            v = int(self.poll_seconds())
        except (TypeError, ValueError):
            v = POLL_DEFAULT_S
        return max(POLL_RANGE[0], min(POLL_RANGE[1], v))

    def snapshot(self, scope: str = "folder", refresh: bool = False, visible: bool = True) -> Dict[str, Any]:
        if visible:
            self.visible_until = time.time() + 2.5 * self.interval()
        self._ensure_thread()
        if refresh:
            forget_manifests()
        rows, at = self.own.get(max_age=self.interval(), fresh=refresh)
        from mace.queue.slurm_ids import own_job_ids
        own_ids = sorted(own_job_ids(login_env()))
        man = manifests_in(self.root)
        folder = describe(self.root, rows, "folder", own_ids, man)
        self.last_rows.update({r["id"]: r for r in folder})
        self.watcher.update(self.root, folder) if at and self.own.ok else None
        listed = folder if scope == "folder" else describe(self.root, rows, "all", own_ids, man)
        counts = {"running": sum(1 for r in folder if r["state"] == "RUNNING"),
                  "pending": sum(1 for r in folder if r["state"] == "PENDING")}
        return {"jobs": listed, "polled_at": at, "slurm": self.own.available(), "error": self.own.error,
                "counts": counts, "interval": self.interval(), "own_ids": own_ids, "scope": scope,
                "queue": queue_budget_view(self.root, rows, scope)}

    def session_active(self) -> bool:
        if not self.session_jobs:
            return False
        rows = self.own.rows
        active = {r["id"] for r in rows if r["state"] in ACTIVE} | {r["id"].split("_")[0] for r in rows if r["state"] in ACTIVE}
        if not self.own.at:
            return True
        return any(j in active or j.split("_")[0] in active for j in self.session_jobs)

    def _ensure_thread(self) -> None:
        with self.lock:
            if self.thread is None or not self.thread.is_alive():
                self.thread = threading.Thread(target=self._loop, name="mace-jobs-poll", daemon=True)
                self.thread.start()

    def kick(self) -> None:
        """A submit added jobs: the background poll follows them."""
        forget_manifests()
        self._ensure_thread()

    def _loop(self) -> None:
        while not self.stop.wait(2.0):
            if not self.own.available():
                continue
            due = time.time() - self.own.at >= self.interval()
            if due and (time.time() < self.visible_until or self.session_active()):
                try:
                    rows, at = self.own.get(max_age=self.interval())
                    from mace.queue.slurm_ids import own_job_ids
                    folder = describe(self.root, rows, "folder", sorted(own_job_ids(login_env())),
                                      manifests_in(self.root))
                    self.last_rows.update({r["id"]: r for r in folder})
                    if self.own.ok:
                        self.watcher.update(self.root, folder)
                except Exception:
                    pass

    def _active_rows(self) -> List[Dict[str, Any]]:
        """This folder's queued and running jobs, from the shared own-jobs answer
        (one squeue a minute at most; none on a machine without squeue)."""
        if not self.own.available():
            return []
        rows, _ = self.own.get(max_age=POLL_DEFAULT_S)
        return [r for r in describe(self.root, rows, "folder", [], manifests_in(self.root)) if r["state"] in ACTIVE]

    def deck_reason(self, path: Path) -> Optional[str]:
        """saving.JOB_STATE_PROVIDER (C19-b): why a deck of a queued or running job is not written."""
        try:
            real = os.path.realpath(str(path))
            for r in self._active_rows():
                if r.get("deck_rel") and os.path.realpath(str(self.root / r["deck_rel"])) == real:
                    stem = Path(path).stem
                    if r["state"] == "RUNNING":
                        return f"{stem} is running with this deck (job {r['id']}). Save As a new name, or cancel the job first."
                    return (f"{stem} is queued (job {r['id']}); it reads this deck when it starts. Save As a new name, "
                            "or cancel the job first.")
        except Exception:
            return None
        return None

    def out_state(self, out_path: Path) -> Optional[str]:
        """setup.JOB_STATE_PROVIDER (C18.7): the SLURM state of the job writing ``out_path``, or None."""
        try:
            real = os.path.realpath(str(out_path))
            for r in self._active_rows():
                if r.get("out_rel") and os.path.realpath(str(self.root / r["out_rel"])) == real:
                    return r["state"]
        except Exception:
            return None
        return None

    def queued_decks(self) -> Dict[str, str]:
        """{real path of a queued deck: job id} from the cached rows (C19-b)."""
        out = {}
        for r in describe(self.root, self.own.rows, "folder", [], manifests_in(self.root)):
            if r["state"] in ACTIVE and r.get("deck_rel"):
                out[os.path.realpath(str(self.root / r["deck_rel"]))] = r["id"]
        return out


# ------------------------------------------------------------------ cancel
HELD_WORDS = {"budget": "queue budget full", "workflow_cap": "the workflow's max_jobs reached",
              "slurm_limit": "SLURM refused it (submit limit)"}


def queue_budget_view(root: Path, rows: List[Dict[str, Any]], scope: str = "folder") -> Dict[str, Any]:
    """The user-wide queue budget and the jobs held under it (this folder's,
    or every registered database's for scope 'all'): mace queue status."""
    out: Dict[str, Any] = {"budget": None, "limit": None, "used": None, "held": [], "held_counts": {}}
    try:
        from mace.queue import budget
        cfg = budget.resolve(quiet=True)
        out.update(budget=cfg.budget, limit=cfg.limit, reserve=cfg.reserve, source=cfg.source)
        out["used"] = sum(1 for r in rows if str(r.get("state", "")).upper() in budget.LIVE_SLURM_STATES)
        root = Path(root).resolve()
        for db in budget.registered_dbs():
            if scope == "folder" and not _inside(db, root):
                continue
            for c in budget.held_rows(db):
                cwd = (c.submit or {}).get("cwd") or ""
                out["held"].append({"calc_id": c.calc_id, "material_id": c.material_id, "calc_type": c.calc_type,
                                    "reason": c.reason, "reason_text": HELD_WORDS.get(c.reason, c.reason),
                                    "held_since": c.held_since, "workflow_id": c.workflow_id,
                                    "workdir_rel": _rel(root, Path(cwd)) if cwd and _inside(cwd, root) else cwd})
                out["held_counts"][c.reason] = out["held_counts"].get(c.reason, 0) + 1
        out["held"].sort(key=lambda h: (h["held_since"] or "", h["calc_id"]))
        out["held"] = out["held"][:500]
    except Exception:
        pass
    return out


def cancel(root: Path, ids: Sequence[str]) -> Dict[str, Any]:
    """scancel for the ids that pass every check; the others with the reason."""
    env = _env()
    if not _which("scancel", env) or not _which("squeue", env):
        return {"canceled": [], "refused": [{"id": str(i), "reason": "no SLURM on this machine"} for i in ids]}
    me = user()
    from mace.queue.slurm_ids import own_job_ids, scancel_refusal
    own_gui = own_job_ids(login_env())
    allowed, refused = [], []
    for raw in ids:
        jid = str(raw).strip()
        if not ID_RE.match(jid):
            refused.append({"id": jid, "reason": f"{jid} is not a job id (digits, or <array>_<task>)"})
            continue
        r = _run([_which("squeue", env), "-h", "-r", "-u", me, "-j", jid, "-o", "%i|%Z"], env)
        listed = [l.split("|", 1) for l in (r.stdout.splitlines() if r and r.returncode == 0 else []) if l.strip()]
        if not any(x[0].strip() == jid or x[0].strip().startswith(jid + "_") for x in listed):
            refused.append({"id": jid, "reason": f"{jid} is not one of your queued jobs"})
            continue
        if "_" not in jid:
            r2 = _run([_which("squeue", env), "-h", "-r", "-j", jid, "-o", "%i"], env)
            tasks = [l.strip() for l in (r2.stdout.splitlines() if r2 else []) if l.strip().startswith(jid + "_")]
            if tasks:
                refused.append({"id": jid, "reason": f"{jid} is a job array; cancel its tasks one by one "
                                                     "(MACE never cancels a whole array)"})
                continue
        if jid in own_gui or jid.split("_")[0] in own_gui:
            refused.append({"id": jid, "reason": "This is the job MACE GUI runs in"})
            continue
        workdirs = [x[1].strip() for x in listed if len(x) > 1 and x[1].strip()]
        dbobj = _db_for(root)
        why = scancel_refusal(jid, db=dbobj, search_dirs=[str(root)] + workdirs)
        if why:
            refused.append({"id": jid, "reason": why})
            continue
        allowed.append(jid)
    canceled = []
    if allowed:
        r = _run([_which("scancel", env), *allowed], env, timeout=30)
        if r is not None and r.returncode == 0:
            canceled = allowed
        else:
            msg = (r.stderr.strip() if r is not None else "scancel did not answer")[:300]
            refused += [{"id": j, "reason": msg} for j in allowed]
    return {"canceled": canceled, "refused": refused, "argv": ["scancel", *allowed] if allowed else None}


class _ReadOnlyDb:
    """The one method scancel_refusal uses, over a read-only connection."""

    def __init__(self, path: Path):
        self.path = path

    def _get_connection(self):
        con = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, timeout=5)
        return _Closing(con)


class _Closing:
    def __init__(self, con):
        self.con = con

    def __enter__(self):
        return self.con

    def __exit__(self, *a):
        self.con.close()
        return False


def _db_for(root: Path):
    p = root / "materials.db"
    return _ReadOnlyDb(p) if p.is_file() else None
