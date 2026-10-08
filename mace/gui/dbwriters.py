"""One GUI writer per database (PLAN_analysis C10.5, C13.0).

Every write the GUI starts into a materials.db (the "Create or Update
Database from a Folder" build, and the tracked submit of PLAN_submit) holds
this lock for that database, keyed by its real path. A second GUI writer on
the same file is refused with 409 and the first writer's name, and the
Database view shows "Updating..." instead of reading a file a build is
rewriting.

Writers outside the GUI (completion callbacks, a terminal ``mace submit
--track``) are not seen here: SQLite's own locking and MACE's commit retry
(mace/database/dbsafety.py) cover them.
"""
from __future__ import annotations

import os
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple, Union

PathLike = Union[str, Path]

_lock = threading.Lock()
_held: Dict[str, Tuple[str, str]] = {}     # realpath -> (token, owner)


class WriterBusy(Exception):
    """Another GUI task is writing this database (HTTP 409)."""

    def __init__(self, owner: str):
        self.owner = owner
        super().__init__(f"{owner} is writing this database; wait for it to finish")


def key(path: PathLike) -> str:
    return os.path.realpath(str(path))


def acquire(path: PathLike, owner: str) -> str:
    """Take the database for ``owner`` (a short name: "Database build",
    "Submit"); a token for release(). Raises WriterBusy when it is held."""
    k = key(path)
    with _lock:
        if k in _held:
            raise WriterBusy(_held[k][1])
        token = uuid.uuid4().hex
        _held[k] = (token, owner)
        return token


def release(path: PathLike, token: Optional[str] = None) -> bool:
    """Give the database back (only the holder's token releases it)."""
    k = key(path)
    with _lock:
        cur = _held.get(k)
        if cur is None or (token is not None and cur[0] != token):
            return False
        del _held[k]
        return True


def owner(path: PathLike) -> Optional[str]:
    """Who writes this database now, or None."""
    with _lock:
        cur = _held.get(key(path))
        return cur[1] if cur else None


@contextmanager
def holding(path: PathLike, who: str) -> Iterator[str]:
    token = acquire(path, who)
    try:
        yield token
    finally:
        release(path, token)
