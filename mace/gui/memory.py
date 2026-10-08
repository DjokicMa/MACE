"""One memory budget for everything ``mace gui`` keeps (spec section 9.2).

Every bounded store reports its bytes to one ``MemoryLedger`` whose limit is
``cache_mb`` (512 MB by default, ``MACE_GUI_CACHE_MB``). The stores share it:

* the editor's undo history may use at most half of it (and each session at
  most 64 MB or ``undoLimit`` steps, whichever comes first);
* the static-asset cache at most a quarter;
* the live state of every open edit session counts too;
* the parse cache and the held built structures (cache.py) share what is
  left (a quarter), and the held structures' *pinned* bytes are capped at a
  quarter when a build is admitted (HeldStore.check_room), since pinned ones
  are never evicted.

When the total would pass the limit, the ledger frees memory in a fixed
order: the oldest parsed files, then unpinned held structures (both can be
read or built again), then the oldest undo steps of the least recently used session, then whole
idle sessions (least recently used first, never one used in the last 60 s),
then the oldest static files. The current state of a session in use is
never dropped.

The ledger never takes a session's lock while it waits: a session that is
busy is skipped, so an edit in one tab never waits on an eviction in another.
"""
from __future__ import annotations

import threading
import time
from typing import Dict, Optional

IDLE_S = 60.0
SESSION_HISTORY_CAP = 64 * 1024 * 1024
TRIM_EVERY_S = 2.0


def _glibc():
    """glibc's allocator controls, or None (macOS, musl, Windows)."""
    try:
        import ctypes
        import ctypes.util
        name = ctypes.util.find_library("c")
        lib = ctypes.CDLL(name) if name else None
        return lib if lib is not None and hasattr(lib, "malloc_trim") and hasattr(lib, "mallopt") else None
    except (OSError, AttributeError, ImportError):
        return None


_LIBC = _glibc()


def limit_arenas(n: int = 2) -> bool:
    """Fewer malloc arenas for the request threads (spec section 9.2). glibc
    gives each thread that allocates its own arena (up to 8 per core), and
    memory freed in one stays in it: with a few threads building 5000-atom
    payloads, RSS grew some 40 MB past what the ledger counts. Same effect as
    MALLOC_ARENA_MAX; a no-op off glibc."""
    if _LIBC is None:
        return False
    return bool(_LIBC.mallopt(-8, int(n)))         # M_ARENA_MAX


def give_back() -> bool:
    """Return freed heap pages to the system (glibc malloc_trim)."""
    return bool(_LIBC.malloc_trim(0)) if _LIBC is not None else False


class MemoryLedger:
    def __init__(self, limit: int):
        self.limit = int(limit)
        self.lock = threading.RLock()
        self._static = None      # an object with static_used(), drop_oldest_static() -> bool
        self._sessions = None    # editor.SessionStore
        self._parse = None       # cache.ParseCache: used(), drop_oldest()
        self._held = None        # cache.HeldStore: used(), drop_oldest(), pinned_bytes()
        self._disk = None        # cache.DiskCache: the plot renders and Setup runs in the cache dir (on /tmp)
        self.max_total = 0       # the highest total seen after an enforce (bench row)
        self.max_history = 0
        self._freed = False      # something was dropped since the last malloc_trim
        self._trimmed_at = 0.0

    # -- stores --------------------------------------------------------------
    def attach_static(self, store) -> None:
        self._static = store

    def attach_sessions(self, store) -> None:
        self._sessions = store
        store.ledger = self

    def attach_parse(self, store) -> None:
        self._parse = store

    def attach_held(self, store) -> None:
        self._held = store

    def attach_disk(self, store) -> None:
        self._disk = store

    # -- numbers -------------------------------------------------------------
    def history_cap(self) -> int:
        return self.limit // 2

    def static_cap(self) -> int:
        return self.limit // 4

    def files_cap(self) -> int:
        """The parse cache plus the held structures."""
        return self.limit // 4

    def held_pinned_cap(self) -> int:
        return self.limit // 4

    def disk_cap(self) -> int:
        """The plot renders and Setup runs (files in the cache dir, on /tmp)."""
        return self.limit // 4

    def usage(self) -> Dict[str, int]:
        s = self._sessions
        out = {"static": self._static.static_used() if self._static else 0,
               "history": s.history_bytes() if s else 0,
               "sessions": s.live_bytes() if s else 0}
        if self._parse is not None:
            out["parse"] = self._parse.used()
        if self._held is not None:
            out["held"] = self._held.used()
        if self._disk is not None:
            out["disk"] = self._disk.used()
        return out

    def total(self) -> int:
        return sum(self.usage().values())

    def report(self) -> Dict[str, float]:
        u = self.usage()
        mb = lambda b: round(b / (1024 * 1024), 2)  # noqa: E731
        out = {"limit_mb": mb(self.limit), "total_mb": mb(sum(u.values())),
               "history_cap_mb": mb(self.history_cap()), "static_cap_mb": mb(self.static_cap()),
               **{f"{k}_mb": mb(v) for k, v in u.items()},
               "max_total_mb": mb(self.max_total), "max_history_mb": mb(self.max_history)}
        if self._held is not None:
            out["held_pinned_mb"] = mb(self._held.pinned_bytes())
            out["held_pinned_cap_mb"] = mb(self.held_pinned_cap())
            out["max_held_pinned_mb"] = mb(self._held.max_pinned)
            out["files_cap_mb"] = mb(self.files_cap())
        if self._disk is not None:
            out["disk_cap_mb"] = mb(self.disk_cap())
        return out

    # -- rules ---------------------------------------------------------------
    def set_limit(self, limit: int) -> None:
        with self.lock:
            self.limit = int(limit)
        self.enforce()

    def enforce(self, now: Optional[float] = None, record: bool = True) -> None:
        """Bring every store back under its share and the total under the limit.
        ``record=False``: an edit that still trims its own session afterwards
        (EditSession._trim) records the peak itself, once it is done."""
        now = time.monotonic() if now is None else now
        st, ss, pc, hs, dk = self._static, self._sessions, self._parse, self._held, self._disk
        with self.lock:
            while dk is not None and dk.used() > self.disk_cap() and dk.drop_oldest():
                pass
            while pc is not None and pc.used() + (hs.used() if hs else 0) > self.files_cap() and pc.drop_oldest():
                self._freed = True
            while hs is not None and hs.used() + (pc.used() if pc else 0) > self.files_cap() and hs.drop_oldest():
                self._freed = True
            while st and st.static_used() > self.static_cap() and st.drop_oldest_static():
                self._freed = True
            while ss and ss.history_bytes() > self.history_cap() and ss.drop_oldest_history():
                self._freed = True
            while self.total() > self.limit:
                if dk is not None and dk.drop_oldest():
                    continue
                if pc is not None and pc.drop_oldest():
                    self._freed = True
                    continue
                if hs is not None and hs.drop_oldest():
                    self._freed = True
                    continue
                if ss and ss.drop_oldest_history():
                    self._freed = True
                    continue
                if ss and ss.drop_idle_session(now - IDLE_S):
                    self._freed = True
                    continue
                if st and st.drop_oldest_static():
                    self._freed = True
                    continue
                break
            self.release(now)
            if record:
                self.note()

    def freed(self) -> None:
        """A store dropped something on its own (an edit trimming its session)."""
        self._freed = True

    def release(self, now: Optional[float] = None) -> None:
        """Hand freed memory back to the system, at most every 2 s: dropped
        undo steps are freed by Python, but glibc keeps the pages unless asked."""
        now = time.monotonic() if now is None else now
        if self._freed and now - self._trimmed_at >= TRIM_EVERY_S:
            self._freed, self._trimmed_at = False, now
            give_back()

    def note(self) -> None:
        """Record the totals for the peak rows (the bench reads them)."""
        with self.lock:
            u = self.usage()
            self.max_total = max(self.max_total, sum(u.values()))
            self.max_history = max(self.max_history, u["history"])


def ledger_for(limit: int, static=None, sessions=None) -> MemoryLedger:
    led = MemoryLedger(limit)
    if static is not None:
        led.attach_static(static)
    if sessions is not None:
        led.attach_sessions(sessions)
    return led


__all__ = ["MemoryLedger", "ledger_for", "IDLE_S", "SESSION_HISTORY_CAP", "limit_arenas", "give_back"]
