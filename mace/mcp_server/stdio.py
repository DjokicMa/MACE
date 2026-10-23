"""The built-in stdio transport (standard library only).

Newline-delimited JSON-RPC 2.0 on stdin/stdout, UTF-8, one message per line;
logs on stderr. While it serves, file descriptor 1 points at stderr and the
protocol goes out through a private duplicate of the original stdout, so a
print() in a parser, a C library's message or a child process can never
corrupt the stream.

``tools/call`` runs in a worker thread (at most ``workers`` at once), so
``ping`` and the protocol's cancel notification are answered while a long call runs;
a canceled call's CLI child is stopped (clirun's process-group stop) and its
response is not sent, as the protocol asks. Progress notifications go out
when the request carries ``_meta.progressToken``. At end of input every
running call is canceled and the server exits.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from typing import Any, BinaryIO, Dict, Optional

from .server import INVALID_REQUEST, PARSE_ERROR, Dispatcher, error_response, throttle

PROGRESS_MIN_INTERVAL_S = 0.25


class StdioServer:
    def __init__(self, dispatcher: Dispatcher, workers: int = 2, stdin: Optional[BinaryIO] = None,
                 stdout: Optional[BinaryIO] = None):
        self.dispatcher = dispatcher
        self.workers = max(1, int(workers))
        self._in = stdin
        self._out = stdout
        self._write_lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(self.workers)
        self._inflight: Dict[Any, threading.Event] = {}
        self._canceled = set()
        self._state = threading.Lock()
        self._threads = []

    # ---------------------------------------------------------- the streams
    def _claim(self) -> None:
        """Protocol output on a private duplicate of fd 1; fd 1 and sys.stdout to stderr."""
        if self._in is None:
            self._in = sys.stdin.buffer
        if self._out is None:
            try:
                sys.stdout.flush()
            except Exception:
                pass
            fd = os.dup(sys.stdout.fileno())
            os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
            self._out = os.fdopen(fd, "wb", buffering=0)
            sys.stdout = sys.stderr

    def send(self, obj: Dict[str, Any]) -> None:
        try:
            text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        except ValueError:                             # a NaN that slipped through: strict JSON anyway
            from .context import jsonable
            text = json.dumps(jsonable(obj), ensure_ascii=False, separators=(",", ":"))
        data = text.encode("utf-8") + b"\n"
        with self._write_lock:
            try:
                self._out.write(data)
                self._out.flush()
            except (BrokenPipeError, OSError, ValueError):
                pass                                   # the client is gone; the read loop ends on EOF

    # ---------------------------------------------------------- the loop
    def serve(self) -> int:
        self._claim()
        for raw in iter(self._in.readline, b""):
            line = raw.strip()
            if not line:
                continue
            try:
                msg = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                self.send(error_response(None, PARSE_ERROR, f"Parse error: {exc}"))
                continue
            if isinstance(msg, list):
                self._batch(msg)
            else:
                self._one(msg)
        self._shutdown()
        return 0

    def _batch(self, msgs) -> None:
        if not msgs:
            self.send(error_response(None, INVALID_REQUEST, "empty batch"))
            return
        out = [r for r in (self.dispatcher.handle(m) for m in msgs) if r is not None]
        if out:
            with self._write_lock:
                try:
                    self._out.write(json.dumps(out, ensure_ascii=False, separators=(",", ":")).encode() + b"\n")
                    self._out.flush()
                except (BrokenPipeError, OSError, ValueError):
                    pass

    def _one(self, msg: Any) -> None:
        if isinstance(msg, dict) and msg.get("method") == "notifications/cancelled" and "id" not in msg:
            rid = (msg.get("params") or {}).get("requestId")
            with self._state:
                ev = self._inflight.get(rid)
                if ev is not None:
                    self._canceled.add(rid)
                    ev.set()
            return
        if isinstance(msg, dict) and msg.get("method") == "tools/call" and "id" in msg:
            rid = msg["id"]
            ev = threading.Event()
            with self._state:
                self._inflight[rid] = ev
            t = threading.Thread(target=self._call, args=(msg, ev), name=f"mcp-call-{rid}", daemon=True)
            self._threads = [x for x in self._threads if x.is_alive()] + [t]
            t.start()
            return
        resp = self.dispatcher.handle(msg)
        if resp is not None:
            self.send(resp)

    def _call(self, msg: Dict[str, Any], ev: threading.Event) -> None:
        rid = msg["id"]
        try:
            with self._slots:
                if ev.is_set():
                    return
                resp = self.dispatcher.handle(msg, cancel_event=ev, on_progress=self._progress_fn(msg))
            with self._state:
                dropped = rid in self._canceled
            if resp is not None and not dropped:
                self.send(resp)
        finally:
            with self._state:
                self._inflight.pop(rid, None)
                self._canceled.discard(rid)

    def _progress_fn(self, msg: Dict[str, Any]):
        token = ((msg.get("params") or {}).get("_meta") or {}).get("progressToken")
        if token is None:
            return None

        def emit(fraction: float, message: str) -> None:
            params = {"progressToken": token, "progress": round(fraction, 4), "total": 1.0}
            if message:
                params["message"] = message
            self.send({"jsonrpc": "2.0", "method": "notifications/progress", "params": params})
        return throttle(emit, PROGRESS_MIN_INTERVAL_S)

    def _shutdown(self, grace: float = 5.0) -> None:
        with self._state:
            for ev in self._inflight.values():
                ev.set()
        end = time.monotonic() + grace
        for t in self._threads:
            t.join(max(0.0, end - time.monotonic()))


def serve(dispatcher: Dispatcher, workers: int = 2) -> int:
    return StdioServer(dispatcher, workers=workers).serve()
