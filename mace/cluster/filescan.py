"""Where are my files? A throttled, low-priority scan of one folder tree,
with cleanup suggestions it never carries out.

Run as ``python -m mace.cluster.filescan ROOT --out FILE.json`` (it renices
itself to 19). Load on a parallel file system: file counts come from
``os.scandir``'s file-type field (no per-file stat); sizes need one
``lstat`` per file, so ``--sizes top`` (the default) reads them only inside
the 20 folders holding the most files, ``all`` reads every file (half the
rate) and ``off`` none. Every directory entry and every lstat counts as one
operation; at most ``[scan] entries_per_second`` (2000) per second, for at
most ``max_seconds`` (600) and ``max_entries`` (1,200,000). Symlinks are not
followed and other file systems are not entered. One scan per user at a
time. Nothing is deleted, moved or renamed: the suggestions are commands to
copy, each path quoted on its own.

Cleanup rules (marker or provenance only, never a guess from a name of
user data):

* R1 a CRYSTAL scratch folder of a finished MACE job, found only through the
  ``.<JOB>.scratch`` pointer in its job folder, when ``<JOB>.out`` there has
  CRYSTAL's normal end and no own job of that name is queued (never a
  pointer naming the job folder, the scanned folder, home, a scratch root or
  a folder above one of them);
* R2 the same for the ``.mace_scratch`` fallback folder next to a deck;
* R3 Python's own temp folders (``pymp-XXXXXXXX``, ``tmpXXXXXXXX``) directly
  inside the system temp folders, owned by you and older than 2 days; never
  under your home, the served folder or a scratch root;
* R4 ``core.<pid>`` files (ELF, yours), ``__pycache__`` folders holding only
  ``.pyc`` files (not inside an installed package or virtual environment; at
  most 10 listed), and ``.mace_arrays/<tag>`` folders of arrays no longer in
  the queue;
* R5 folders marked with a ``.mace_test_dir`` file, or listed in
  ``[scan] leftover_paths``.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import stat
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

TOP = 20
DEPTH = 3
PROGRESS_EVERY = 2.0
PY_TEMP = (re.compile(r"^pymp-[a-z0-9_]{8}$"), re.compile(r"^tmp[a-z0-9_]{8}$"))
OLD_S = 2 * 86400
PYCACHE_MAX = 10


class Throttle:
    """At most ``rate`` operations per second, measured on ``clock`` and
    enforced with ``sleep`` (both injectable for tests)."""

    def __init__(self, rate: float, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.rate, self.clock, self.sleep = float(rate), clock, sleep
        self.t0 = clock()
        self.ops = 0

    def tick(self, n: int = 1) -> None:
        self.ops += n
        ahead = self.ops / self.rate - (self.clock() - self.t0)
        if ahead > 0.005:
            self.sleep(ahead)

    def elapsed(self) -> float:
        return self.clock() - self.t0


def suggest_command(path: str) -> str:
    return f"rm -rf -- {shlex.quote(path)}"


def _ends_normally(out: Path) -> bool:
    try:
        size = out.stat().st_size
        with open(out, "rb") as fh:
            fh.seek(max(0, size - 65536))
            return b"EEEEEEEEEE TERMINATION" in fh.read()
    except OSError:
        return False


def _is_elf(p: Path) -> bool:
    try:
        with open(p, "rb") as fh:
            return fh.read(4) == b"\x7fELF"
    except OSError:
        return False


def _inside(p: str, roots: Sequence[str]) -> bool:
    rp = os.path.realpath(p)
    for r in roots:
        if not r:
            continue
        rr = os.path.realpath(r)
        if rp == rr or rp.startswith(rr.rstrip("/") + "/"):
            return True
    return False


def scan(root, sizes: str = "top", rate: float = 2000, max_seconds: float = 600, max_entries: int = 1_200_000,
         clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
         progress: Optional[Callable[[Dict[str, Any]], None]] = None, queued_names: Sequence[str] = (),
         queued_ids: Sequence[str] = (), protected: Sequence[str] = (), leftover_paths: Sequence[str] = (),
         temp_dirs: Optional[Sequence[str]] = None, cancel: Optional[Callable[[], bool]] = None,
         top: int = TOP) -> Dict[str, Any]:
    root = os.path.realpath(str(root))
    thr = Throttle(rate, clock, sleep)
    st_root = os.lstat(root)
    dev = st_root.st_dev
    own_files: Dict[str, int] = {}
    own_bytes: Dict[str, int] = {}
    pointers: List[str] = []
    cores: List[str] = []
    pycache: List[str] = []
    arrays: List[str] = []
    markers: List[str] = []
    stack = [root]
    entries = dirs = 0
    stopped = None
    last_progress = clock()
    while stack:
        if cancel and cancel():
            stopped = "canceled"
            break
        if thr.elapsed() >= max_seconds:
            stopped = f"stopped at {int(max_seconds)} s after {entries:,} entries"
            break
        if entries >= max_entries:
            stopped = f"stopped after {entries:,} entries (the limit)"
            break
        d = stack.pop()
        try:
            if os.lstat(d).st_dev != dev:            # one lstat per directory
                continue
        except OSError:
            continue
        thr.tick()
        dirs += 1
        n = 0
        only_pyc = os.path.basename(d) == "__pycache__"
        try:
            with os.scandir(d) as it:
                for e in it:
                    entries += 1
                    thr.tick()
                    try:
                        if e.is_dir(follow_symlinks=False):
                            stack.append(e.path)
                            only_pyc = False
                            continue
                        n += 1
                        if sizes == "all" and e.is_file(follow_symlinks=False):
                            own_bytes[d] = own_bytes.get(d, 0) + os.lstat(e.path).st_size
                            thr.tick()
                        if only_pyc and not e.name.endswith(".pyc"):
                            only_pyc = False
                        if e.name.startswith(".") and e.name.endswith(".scratch") and len(e.name) > 9:
                            pointers.append(e.path)
                        elif re.match(r"^core\.\d+$", e.name) and e.is_file(follow_symlinks=False):
                            cores.append(e.path)
                        elif e.name == ".mace_test_dir":
                            markers.append(d)
                    except OSError:
                        continue
                    if entries >= max_entries:
                        break
        except OSError:
            continue
        own_files[d] = n
        if only_pyc and n:
            pycache.append(d)
        if os.path.basename(os.path.dirname(d)) == ".mace_arrays":
            arrays.append(d)
        if progress and clock() - last_progress >= PROGRESS_EVERY:
            last_progress = clock()
            progress({"entries": entries, "dirs": dirs, "elapsed": round(thr.elapsed(), 1)})
    # sizes of the top folders by own file count
    if sizes == "top" and not stopped:
        for d, _ in sorted(own_files.items(), key=lambda kv: -kv[1])[:top]:
            try:
                with os.scandir(d) as it:
                    for e in it:
                        thr.tick()
                        if e.is_file(follow_symlinks=False):
                            own_bytes[d] = own_bytes.get(d, 0) + os.lstat(e.path).st_size
                            thr.tick()
            except OSError:
                continue
            if thr.elapsed() >= max_seconds:
                stopped = f"stopped at {int(max_seconds)} s while reading sizes"
                break
    # aggregate: subtree totals, reported at depth <= 3 below the root
    sub_files: Dict[str, int] = {}
    sub_bytes: Dict[str, int] = {}
    for d, n in own_files.items():
        rel = os.path.relpath(d, root)
        parts = [] if rel == "." else rel.split(os.sep)
        b = own_bytes.get(d, 0)
        for k in range(0, min(len(parts), DEPTH) + 1):
            key = os.path.join(root, *parts[:k]) if k else root
            sub_files[key] = sub_files.get(key, 0) + n
            sub_bytes[key] = sub_bytes.get(key, 0) + b
    top_count = sorted(((k, v) for k, v in sub_files.items() if k != root), key=lambda kv: -kv[1])[:top]
    top_size = sorted(((k, v) for k, v in sub_bytes.items() if k != root and v), key=lambda kv: -kv[1])[:top]
    res = {
        "root": root, "entries": entries, "dirs": dirs, "files": sum(own_files.values()),
        "bytes": sum(own_bytes.values()) if sizes == "all" else None,
        "elapsed_s": round(thr.elapsed(), 1), "sizes": sizes, "complete": stopped is None, "stopped": stopped,
        "coverage": (f"{stopped}; the rest is not counted" if stopped else "complete"),
        "top_by_count": [{"path": k, "files": v, "bytes": sub_bytes.get(k) or None} for k, v in top_count],
        "top_by_size": [{"path": k, "bytes": v, "files": sub_files.get(k, 0)} for k, v in top_size],
        "rate": rate, "ops": thr.ops,
    }
    res["suggestions"] = suggestions(root, pointers, cores, pycache, arrays, markers, queued_names, queued_ids,
                                     protected, leftover_paths, temp_dirs)
    return res


def suggestions(root, pointers, cores, pycache, arrays, markers, queued_names, queued_ids, protected,
                leftover_paths, temp_dirs) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    uid = os.getuid() if hasattr(os, "getuid") else None
    queued = set(queued_names or ())

    def add(rule, path, reason):
        out.append({"rule": rule, "path": path, "reason": reason, "command": suggest_command(path)})
    for ptr in pointers:
        job = os.path.basename(ptr)[1:-len(".scratch")]
        jobdir = Path(ptr).parent
        try:
            target = Path(ptr).read_text().strip().splitlines()[0]
        except (OSError, IndexError):
            continue
        if not target or not os.path.isabs(target) or not os.path.isdir(target):
            continue
        if job in queued:
            continue
        # a pointer that names the job folder, the scanned folder, home, a
        # scratch root or any folder holding one of them is never a "scratch
        # folder of a finished job": rm -rf on it would take the outputs too
        keep = [str(jobdir), root, os.path.expanduser("~")] + list(protected or ())
        if any(_inside(k, [target]) for k in keep if k):
            continue
        if not _ends_normally(jobdir / f"{job}.out"):
            continue
        try:
            if uid is not None and os.stat(target).st_uid != uid:
                continue
        except OSError:
            continue
        rule = "R2" if f"{os.sep}.mace_scratch" in target else "R1"
        add(rule, target, f"Scratch folder of the finished job {job} (its output ended normally); safe after "
                          f"checking; removing it means an OPTGEOM RESTART of this job cannot continue")
    for c in cores:
        try:
            st = os.lstat(c)
        except OSError:
            continue
        if stat.S_ISREG(st.st_mode) and (uid is None or st.st_uid == uid) and _is_elf(Path(c)):
            add("R4", c, "A core dump of a crashed program")
    # a package installation's caches (a virtual environment, site-packages) are
    # part of that install: not suggested; the rest is listed up to PYCACHE_MAX
    own = [d for d in pycache if not re.search(r"/(site|dist)-packages/|/lib/python\d", d)]
    for d in own[:PYCACHE_MAX]:
        add("R4", d, "Python's compiled-file cache (rebuilt on demand)")
    if len(own) > PYCACHE_MAX:
        out.append({"rule": "R4", "path": None, "command": None,
                    "reason": f"and {len(own) - PYCACHE_MAX} more __pycache__ folders (not listed)"})
    ids = set(str(i).split("_")[0] for i in queued_ids or ())
    for d in arrays:
        try:
            data = json.loads((Path(d) / "submitted.json").read_text())
        except (OSError, ValueError):
            continue
        aid = str(data.get("array_job_id") or "")
        if aid and aid not in ids:
            add("R4", d, f"Files of MACE job array {aid}, which is no longer in the queue")
    for d in markers:
        add("R5", d, "Marked as a MACE test folder (.mace_test_dir)")
    for p in leftover_paths or ():
        p = os.path.expanduser(p)
        if os.path.isdir(p) and _inside(p, [root]):
            add("R5", p, "Listed in [scan] leftover_paths of site.toml")
    # R3: Python temp folders directly inside the system temp folders
    home = os.path.expanduser("~")
    no_go = [home, root] + list(protected or ())
    tmpdir = os.environ.get("TMPDIR")
    if temp_dirs is None:
        temp_dirs = ["/tmp", "/var/tmp", tempfile.gettempdir()]
    if tmpdir and _inside(tmpdir, no_go):
        out.append({"rule": "R3", "path": None, "command": None,
                    "reason": "TMPDIR is inside your home/scratch; temp folders there are not suggested"})
        temp_dirs = [t for t in temp_dirs if not _inside(t, no_go)]
    now = time.time()
    seen = set()
    for t in temp_dirs:
        if not t or _inside(t, no_go) or os.path.realpath(t) in seen or not os.path.isdir(t):
            continue
        seen.add(os.path.realpath(t))
        try:
            with os.scandir(t) as it:
                for e in it:
                    if not any(rx.match(e.name) for rx in PY_TEMP):
                        continue
                    try:
                        st = e.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    if not stat.S_ISDIR(st.st_mode) or (uid is not None and st.st_uid != uid):
                        continue
                    if now - st.st_mtime < OLD_S:
                        continue
                    add("R3", e.path, "A Python temporary folder older than 2 days")
        except OSError:
            continue
    return out


# ------------------------------------------------------------------ process entry
def lock_path() -> Path:
    from mace.cluster import query
    return query.cache_root() / "scan.lock"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m mace.cluster.filescan")
    ap.add_argument("root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", choices=("top", "all", "off"))
    ap.add_argument("--max-seconds", type=float)
    ap.add_argument("--top", type=int, default=TOP)
    a = ap.parse_args(argv)
    try:
        os.nice(19)
    except (OSError, AttributeError):
        pass
    from mace.cluster import config, query
    cfg = config.load()
    lp = lock_path()
    lp.parent.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl
        fh = open(lp, "a+")
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.seek(0)
            try:
                started = float(fh.read().strip() or 0)
            except ValueError:
                started = 0
            mins = int((time.time() - started) / 60) if started else 0
            print(json.dumps({"error": f"A scan is already running (started {mins} min ago)"}), flush=True)
            return 5
        fh.seek(0)
        fh.truncate()
        fh.write(str(time.time()))
        fh.flush()
    except ImportError:
        fh = None
    queued_names, queued_ids = [], []
    try:
        from mace.cluster import reader
        if reader.slurm_present():
            rows = reader.squeue_own().value or []
            queued_names = [r.get("name") for r in rows]
            queued_ids = [r.get("id") for r in rows]
    except Exception:
        pass
    from mace.cluster.quotas import scratch_roots
    res = scan(a.root, sizes=a.sizes or cfg.get("scan", "sizes"), rate=cfg.get("scan", "entries_per_second"),
               max_seconds=a.max_seconds or cfg.get("scan", "max_seconds"),
               max_entries=cfg.get("scan", "max_entries"),
               progress=lambda p: print(json.dumps(p), flush=True), queued_names=queued_names,
               queued_ids=queued_ids, protected=scratch_roots(), leftover_paths=cfg.get("scan", "leftover_paths"),
               top=max(1, a.top))
    tmp = a.out + ".part"
    with open(tmp, "w") as f:
        json.dump(res, f)
    os.replace(tmp, a.out)
    print(json.dumps({"done": True, "entries": res["entries"]}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
