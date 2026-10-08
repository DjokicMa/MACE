"""File History: the previous version of every file the GUI replaces
(PLAN_analysis C20). One store, one API, one name in the interface.

* **Where:** ``$XDG_STATE_HOME/mace/gui/history/`` (``~/.local/state/...``),
  or ``MACE_GUI_HISTORY_DIR``; outside the served folder, mode 0700, files
  0600. Entries are grouped per served folder by ``root_hash`` (sha1 of the
  folder's real path; the path itself is never stored) and per file by
  ``sha1(relative path)[:16]``; each file's ``index.jsonl`` holds {id, t,
  rel_path, root_hash, size, sha256, reason, route, kind, doc_title}, never
  file contents and never an absolute path.
* **When:** every GUI replace of an existing file goes through
  ``saving.write_atomic``, which calls ``keep()`` first; when the store
  cannot be written the write is refused (nothing is ever replaced without
  a copy). A new file is recorded with ``record_created`` (its "restore" is
  deleting what MACE created, while its bytes still match).
* **Retention:** two pools, ``input`` (decks, CIFs, JSON, scripts) and
  ``output`` (copies of finished results kept before an overwrite), each
  with its own size cap; per file the last ``files.historyKeep`` versions
  within ``files.historyDays``; the newest entry of every file is never
  evicted. A file over 50 MB is not copied (the ``.bak`` beside it stays).

The public API (``keep``, ``snapshot``, ``record_created``, ``entries``,
``entry_path``, ``prune``) is what the editor saves, Setup, Write decks and
the submit routes call; nothing keeps a second copy elsewhere.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

MAX_FILE = 50 * 1024 * 1024
ID_RE = re.compile(r"^[0-9]{8}T[0-9]{6}Z_[0-9]+_[A-Za-z0-9._-]{1,200}$")
DIR_RE = re.compile(r"^[0-9a-f]{16}$")
DEFAULTS = {"files.historyKeep": 20, "files.historyDays": 30, "files.historyMaxMB": 500,
            "files.historyOutputsMaxMB": 1000}
_lock = threading.RLock()
_root: List[Optional[Path]] = [None]


class HistoryUnavailable(OSError):
    """The store cannot be written: the caller refuses its write."""


def configure(root: Optional[Path]) -> None:
    """The served folder whose files are kept (the server calls it)."""
    _root[0] = Path(root).resolve() if root is not None else None


def root() -> Optional[Path]:
    return _root[0]


def store_dir() -> Path:
    env = os.environ.get("MACE_GUI_HISTORY_DIR")
    if env:
        return Path(env)
    base = os.environ.get("XDG_STATE_HOME") or os.path.join(os.path.expanduser("~"), ".local", "state")
    return Path(base) / "mace" / "gui" / "history"


def root_hash(r: Optional[Path] = None) -> str:
    r = Path(r or _root[0]).resolve()
    return hashlib.sha1(str(r).encode()).hexdigest()


def _settings() -> Dict[str, Any]:
    try:
        from . import settings as gs
        s = gs.load()
    except Exception:
        s = {}
    return {k: s.get(k, v) for k, v in DEFAULTS.items()}


def _mkdir(p: Path) -> None:
    """A store folder, 0700 when MACE creates it (an existing one is left as
    it is: a read-only store refuses writes, it is never made writable)."""
    if p.is_dir():
        return
    p.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(p, 0o700)
    except OSError:
        pass


def _rel(path: Path, r: Optional[Path] = None) -> str:
    r = Path(r or _root[0])
    return Path(os.path.abspath(str(path))).relative_to(r).as_posix()


def _file_dir(rel: str, r: Optional[Path] = None) -> Path:
    return store_dir() / root_hash(r) / hashlib.sha1(rel.encode()).hexdigest()[:16]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_name(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", name)[:200]
    return s or "file"


@dataclass
class Entry:
    id: str
    t: float
    rel_path: str
    size: int
    sha256: str
    reason: str
    route: Optional[str]
    kind: str
    doc_title: Optional[str] = None
    skipped: Optional[str] = None
    created: bool = False

    def record(self) -> Dict[str, Any]:
        d = {"id": self.id, "t": self.t, "rel_path": self.rel_path, "size": self.size, "sha256": self.sha256,
             "reason": self.reason, "route": self.route, "kind": self.kind, "doc_title": self.doc_title}
        if self.skipped:
            d["skipped"] = self.skipped
        if self.created:
            d["created"] = True
        return d


def _index(fdir: Path) -> List[Dict[str, Any]]:
    out = []
    try:
        with open(fdir / "index.jsonl") as fh:
            for line in fh:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    except FileNotFoundError:
        pass
    return out


def _write_index(fdir: Path, rows: List[Dict[str, Any]]) -> None:
    tmp = fdir / f".index.{os.getpid()}.tmp"
    with open(tmp, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    os.chmod(tmp, 0o600)
    os.replace(tmp, fdir / "index.jsonl")


def _new_id(fdir: Path, name: str) -> str:
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    n = 1
    while True:
        eid = f"{ts}_{n}_{_safe_name(name)}"
        if not (fdir / eid).exists():
            return eid
        n += 1


def keep(target: Path, reason: str, route: Optional[str] = None, kind: str = "input",
         doc_title: Optional[str] = None) -> Optional[Entry]:
    """Copy the current ``target`` into the store before the caller replaces
    it. None when there is no file to keep or no served folder is
    configured; an entry marked ``skipped`` for a file over 50 MB; raises
    HistoryUnavailable when the store cannot be written."""
    target = Path(target)
    r = _root[0]
    if r is None or not target.is_file():
        return None
    rel = _rel(target, r)
    size = target.stat().st_size
    with _lock:
        fdir = _file_dir(rel, r)
        if size > MAX_FILE:
            return Entry(id="", t=time.time(), rel_path=rel, size=size, sha256="", reason=reason, route=route,
                         kind=kind, doc_title=doc_title,
                         skipped="No history copy (file over 50 MB); the .bak beside it is kept")
        try:
            _mkdir(store_dir())
            _mkdir(fdir.parent)
            _mkdir(fdir)
            eid = _new_id(fdir, target.name)
            tmp = fdir / f".{eid}.tmp"
            shutil.copyfile(target, tmp)
            os.chmod(tmp, 0o600)
            os.replace(tmp, fdir / eid)
            e = Entry(id=eid, t=time.time(), rel_path=rel, size=size, sha256=_sha256(fdir / eid), reason=reason,
                      route=route, kind=kind, doc_title=doc_title)
            rows = _index(fdir) + [e.record()]
            _write_index(fdir, rows)
        except OSError as exc:
            raise HistoryUnavailable(
                f"MACE could not keep the previous version ({exc.strerror or exc}); nothing was written. "
                "Free space or set MACE_GUI_HISTORY_DIR.") from exc
        try:
            prune()
        except OSError:
            pass
        return e


def snapshot(paths: Iterable[Path], reason: str, route: Optional[str] = None, kind: str = "input") -> List[Entry]:
    """``keep`` over several files, all or nothing: a failure removes the
    copies already made and raises."""
    made: List[Entry] = []
    try:
        for p in paths:
            e = keep(Path(p), reason, route, kind)
            if e is not None:
                made.append(e)
    except BaseException:
        for e in made:
            _remove_entry(e.rel_path, e.id)
        raise
    return made


def record_created(path: Path, reason: str, route: Optional[str] = None, sha256: Optional[str] = None) -> Optional[Entry]:
    """A write that created a new file: listed, with no copy (restore = delete it)."""
    path = Path(path)
    r = _root[0]
    if r is None or not path.is_file():
        return None
    rel = _rel(path, r)
    with _lock:
        fdir = _file_dir(rel, r)
        try:
            _mkdir(store_dir())
            _mkdir(fdir.parent)
            _mkdir(fdir)
            e = Entry(id=_new_id(fdir, path.name), t=time.time(), rel_path=rel, size=path.stat().st_size,
                      sha256=sha256 or _sha256(path), reason=reason, route=route, kind="input", created=True)
            _write_index(fdir, _index(fdir) + [e.record()])
        except OSError:
            return None          # listing a creation is a convenience; the write itself is not refused
        return e


def _remove_entry(rel: str, eid: str, r: Optional[Path] = None) -> None:
    with _lock:
        fdir = _file_dir(rel, r)
        try:
            (fdir / eid).unlink()
        except (FileNotFoundError, IsADirectoryError):
            pass
        _write_index(fdir, [x for x in _index(fdir) if x.get("id") != eid])


def entries(rel: str, r: Optional[Path] = None) -> List[Dict[str, Any]]:
    """The versions of one file of the served folder, newest first."""
    fdir = _file_dir(rel, r or _root[0])
    rows = [x for x in _index(fdir) if x.get("created") or (fdir / x.get("id", "")).is_file()]
    return sorted(rows, key=lambda x: (x.get("t") or 0, x.get("id")), reverse=True)


def has_entries(rel: str) -> bool:
    try:
        return bool(entries(rel))
    except Exception:
        return False


def entry_path(eid: str, r: Optional[Path] = None) -> Path:
    """The stored copy of entry ``eid`` of the current folder; anything that
    is not an entry of this folder's store is refused (KeyError)."""
    if not isinstance(eid, str) or "\0" in eid or not ID_RE.match(eid):
        raise KeyError("no such version")
    base = (store_dir() / root_hash(r or _root[0])).resolve()
    if not base.is_dir():
        raise KeyError("no such version")
    for fdir in base.iterdir():
        if not DIR_RE.match(fdir.name):
            continue
        p = fdir / eid
        if p.is_file() and not p.is_symlink():
            real = p.resolve()
            if base in real.parents:
                return real
    raise KeyError("no such version")


def entry_record(eid: str, r: Optional[Path] = None) -> Dict[str, Any]:
    """The index row of entry ``eid`` of the current folder. A "created"
    entry has no stored copy (its restore deletes the file MACE created), so
    it is found by its index row; any other entry needs its copy."""
    if not isinstance(eid, str) or "\0" in eid or not ID_RE.match(eid):
        raise KeyError("no such version")
    base = (store_dir() / root_hash(r or _root[0])).resolve()
    if not base.is_dir():
        raise KeyError("no such version")
    for fdir in base.iterdir():
        if not DIR_RE.match(fdir.name) or fdir.is_symlink() or not fdir.is_dir():
            continue
        for row in _index(fdir):
            if row.get("id") != eid:
                continue
            if row.get("created"):
                return row
            p = entry_path(eid, r)
            if p.parent == fdir.resolve():
                return row
    raise KeyError("no such version")


def usage(kind: Optional[str] = None) -> int:
    total = 0
    base = store_dir()
    if not base.is_dir():
        return 0
    for rdir in base.iterdir():
        if not rdir.is_dir():
            continue
        for fdir in rdir.iterdir():
            if not fdir.is_dir():
                continue
            for row in _index(fdir):
                if row.get("created"):
                    continue
                if kind is None or row.get("kind", "input") == kind:
                    total += int(row.get("size") or 0)
    return total


def prune(settings: Optional[Dict[str, Any]] = None, now: Optional[float] = None) -> int:
    """Apply the retention rules to every folder's store; returns how many
    copies went. The newest entry of a file is never removed."""
    s = settings or _settings()
    keep_n = max(1, int(s["files.historyKeep"]))
    max_age = float(s["files.historyDays"]) * 86400.0
    caps = {"input": float(s["files.historyMaxMB"]) * 1024 * 1024,
            "output": float(s["files.historyOutputsMaxMB"]) * 1024 * 1024}
    now = time.time() if now is None else now
    base = store_dir()
    if not base.is_dir():
        return 0
    removed = 0
    pools: Dict[str, List[Any]] = {"input": [], "output": []}
    with _lock:
        for rdir in base.iterdir():
            if not rdir.is_dir():
                continue
            for fdir in rdir.iterdir():
                if not fdir.is_dir() or not DIR_RE.match(fdir.name):
                    continue
                rows = sorted(_index(fdir), key=lambda x: (x.get("t") or 0, x.get("id")), reverse=True)
                copies = [x for x in rows if not x.get("created")]
                drop = set()
                for k, row in enumerate(copies):
                    if k == 0:
                        continue                         # the newest copy of every file stays
                    if k >= keep_n or now - float(row.get("t") or 0) > max_age:
                        drop.add(row["id"])
                    else:
                        pools[row.get("kind", "input") if row.get("kind") in pools else "input"].append(
                            (float(row.get("t") or 0), fdir, row))
                if drop:
                    for eid in drop:
                        try:
                            (fdir / eid).unlink()
                        except FileNotFoundError:
                            pass
                    _write_index(fdir, [x for x in rows if x.get("id") not in drop])
                    removed += len(drop)
        for kind, items in pools.items():
            total = usage(kind)
            for t, fdir, row in sorted(items, key=lambda x: x[0]):       # oldest first
                if total <= caps[kind]:
                    break
                try:
                    (fdir / row["id"]).unlink()
                except FileNotFoundError:
                    pass
                _write_index(fdir, [x for x in _index(fdir) if x.get("id") != row["id"]])
                total -= int(row.get("size") or 0)
                removed += 1
    return removed


def over_cap_note() -> Optional[str]:
    """The Settings line when the newest copies alone exceed a cap."""
    s = _settings()
    for kind, key in (("input", "files.historyMaxMB"), ("output", "files.historyOutputsMaxMB")):
        used = usage(kind) / (1024 * 1024)
        cap = float(s[key])
        if used > cap:
            return (f"File History uses {used:.0f} MB (over the {cap:.0f} MB limit because it keeps the newest "
                    f"copy of every file)")
    return None
