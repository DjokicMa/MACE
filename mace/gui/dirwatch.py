"""Notice files added to or removed from the served folder (the Explorer's refresh).

A job writing its .out, a file copied in from a shell: the Explorer lists
them without a reload. Adding, removing or renaming an entry changes its
folder's modification time, so the watch keeps the mtime of each folder the
Explorer lists (the same folders: hidden ones, ``__pycache__`` and the like,
symlinked folders and anything past the listing's depth are left out) and
compares them with one stat() per folder per check, never listing a folder
or touching a file. Only after a change are the folders listed again (to
find new subfolders). At most MAX_DIRS folders are watched, the shallowest
first; on a folder with more, changes deeper down show at the next listing.

Polling, not inotify: a cluster's shared filesystems (NFS, Lustre, GPFS) do  # british-ok: Lustre is a file system
not report changes made on other nodes, where the jobs run.
"""
from __future__ import annotations

import os
from collections import deque
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

MAX_DIRS = 500


class DirWatch:
    def __init__(self, root: Path, max_depth: int, skip: Callable[[str], bool], max_dirs: int = MAX_DIRS):
        self.root, self.max_depth, self.skip, self.max_dirs = Path(root), max_depth, skip, max_dirs
        self.stamps: Dict[str, int] = {}
        self.snapshot()

    def _dirs(self) -> Iterable[str]:
        """The folders the Explorer walks, breadth first, at most max_dirs."""
        todo = deque([(str(self.root), 0)])
        n = 0
        while todo and n < self.max_dirs:
            d, depth = todo.popleft()
            n += 1
            yield d
            if depth >= self.max_depth:
                continue
            try:
                with os.scandir(d) as it:
                    subs = sorted(e.path for e in it if not self.skip(e.name) and e.is_dir(follow_symlinks=False))
            except OSError:
                continue
            todo.extend((s, depth + 1) for s in subs)

    @staticmethod
    def _mtime(d: str) -> Optional[int]:
        try:
            return os.stat(d).st_mtime_ns
        except OSError:
            return None

    def snapshot(self) -> None:
        self.stamps = {d: self._mtime(d) for d in self._dirs()}

    def changed(self) -> bool:
        """True when an entry was added, removed or renamed in a watched folder
        since the last check (the watch then takes the folders anew)."""
        for d, stamp in self.stamps.items():
            if self._mtime(d) != stamp:
                self.snapshot()
                return True
        return False
