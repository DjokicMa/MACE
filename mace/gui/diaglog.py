"""The server log ring for Copy Diagnostics: ``install()`` tees ``sys.stderr``
into the last 500 lines / 64 KB (writes still reach the real stderr), so
handler tracebacks and the stop lines can go into a bug report (redacted).
Standard library only."""
from __future__ import annotations

import sys
import threading
from collections import deque
from typing import List

MAX_LINES = 500
MAX_BYTES = 64 * 1024

_lock = threading.Lock()
_ring: "deque[str]" = deque()
_bytes = 0
_partial = ""


def add(text: str) -> None:
    global _bytes, _partial
    with _lock:
        _partial += text
        *done, _partial = _partial.split("\n")
        for line in done:
            _ring.append(line)
            _bytes += len(line) + 1
        while _ring and (len(_ring) > MAX_LINES or _bytes > MAX_BYTES):
            _bytes -= len(_ring.popleft()) + 1
        if len(_partial) > 4096:
            _partial = _partial[-4096:]


def tail(n: int = 200) -> List[str]:
    with _lock:
        lines = list(_ring)[-n:]
        if _partial:
            lines.append(_partial)
    return lines[-n:]


class _Tee:
    def __init__(self, real):
        self.real = real

    def write(self, s):
        try:
            add(str(s))
        except Exception:
            pass
        return self.real.write(s)

    def flush(self):
        return self.real.flush()

    def __getattr__(self, name):
        return getattr(self.real, name)


def install() -> None:
    if not isinstance(sys.stderr, _Tee) and sys.stderr is not None:
        sys.stderr = _Tee(sys.stderr)
