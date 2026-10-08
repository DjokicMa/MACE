"""Big files as text, through windows (spec section 8.4.6, plan B8).

"Open as Text" never sends a whole file: ``window()`` reads at most 256 KB
starting at a byte offset, widened back to the start of a line and cut at
the end of one; ``line_index()`` makes one streaming pass (1 MB reads) and
records where every 4096th line starts, so the viewer can jump to any line
with one request; ``find()`` streams the file for a text, as a cancelable
task, and returns the first matches with their line numbers.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

WINDOW_MAX = 256 * 1024
LINE_STEP = 4096
CHUNK = 1 << 20
FIND_MAX = 500


def is_binary(path: Path) -> bool:
    """NUL bytes in the first 4 KB: not a text file."""
    with open(path, "rb") as fh:
        return b"\0" in fh.read(4096)


def window(path: Path, offset: int = 0, limit: int = WINDOW_MAX) -> Dict[str, Any]:
    """The lines in [offset, offset + limit): ``offset`` moves back to the start
    of its line, the end is cut after the last whole line (one line longer than
    the window is sent cut, flagged)."""
    size = os.path.getsize(path)
    limit = max(1, min(int(limit), WINDOW_MAX))
    offset = max(0, min(int(offset), size))
    with open(path, "rb") as fh:
        start = offset
        if start > 0:
            back = min(start, 4096)
            fh.seek(start - back)
            pre = fh.read(back)
            k = pre.rfind(b"\n")
            while k < 0 and start - back > 0:            # a very long line: walk further back, at most 64 KB
                if back >= 65536:
                    break
                back = min(start, back * 4)
                fh.seek(start - back)
                pre = fh.read(back)
                k = pre.rfind(b"\n")
            start = start - back + k + 1 if k >= 0 else start
        fh.seek(start)
        raw = fh.read(limit)
    end = start + len(raw)
    cut_line = False
    if end < size:
        k = raw.rfind(b"\n")
        if k >= 0:
            raw = raw[:k + 1]
            end = start + len(raw)
        else:
            cut_line = True
    return {"path": None, "offset": start, "end": end, "size": size, "eof": end >= size, "cut_line": cut_line,
            "text": raw.decode("utf-8", "replace")}


def line_index(path: Path, should_stop=None) -> Dict[str, Any]:
    """{lines, size, step, offsets, complete_end, complete_lines}: offsets[k]
    is where line k*step starts (0-based). Only the size of the stat taken
    first is read (a file being written grows meanwhile); ``complete_end``
    is where the last whole line ends, ``complete_lines`` how many end there
    (a live file's Follow continues from there)."""
    offsets: List[int] = [0]
    lines = 0
    pos = 0
    last_nl_end = True
    complete_end = 0
    size = os.path.getsize(path)
    left = size
    with open(path, "rb") as fh:
        while left > 0:
            chunk = fh.read(min(CHUNK, left))
            if should_stop is not None:
                should_stop()
            if not chunk:
                break
            left -= len(chunk)
            start = 0
            while True:
                k = chunk.find(b"\n", start)
                if k < 0:
                    break
                lines += 1
                complete_end = pos + k + 1
                if lines % LINE_STEP == 0:
                    offsets.append(pos + k + 1)
                start = k + 1
            last_nl_end = chunk.endswith(b"\n")
            pos += len(chunk)
    complete_lines = lines
    if size and not last_nl_end:
        lines += 1                       # a last line with no newline
    if offsets and offsets[-1] >= size and len(offsets) > 1:
        offsets.pop()
    return {"lines": lines, "size": size, "step": LINE_STEP, "offsets": offsets,
            "complete_end": complete_end, "complete_lines": complete_lines}


def find(path: Path, query: str, case: bool = False, ctx=None, limit: int = FIND_MAX) -> Dict[str, Any]:
    """The first ``limit`` lines holding ``query``: [{line (0-based), offset, text}], streamed."""
    if not query:
        raise ValueError("find what?")
    if len(query) > 200:
        raise ValueError("search for at most 200 characters")
    needle = query.encode() if case else query.lower().encode()
    size = os.path.getsize(path)
    out: List[Dict[str, Any]] = []
    line_no, pos, tail = 0, 0, b""
    more = False
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(CHUNK)
            if ctx is not None:
                ctx.check()
                ctx.progress(min(0.99, pos / max(1, size)), f"Searching {path.name}")
            if not chunk:
                break
            data = tail + chunk
            base = pos - len(tail)
            parts = data.split(b"\n")
            tail = parts.pop()
            off = base
            for ln in parts:
                hay = ln if case else ln.lower()
                if needle in hay:
                    if len(out) >= limit:
                        more = True
                        break
                    out.append({"line": line_no, "offset": off, "text": ln[:400].decode("utf-8", "replace")})
                line_no += 1
                off += len(ln) + 1
            if more:
                break
            pos += len(chunk)
        if not more and tail:
            hay = tail if case else tail.lower()
            if needle in hay and len(out) < limit:
                out.append({"line": line_no, "offset": pos - len(tail), "text": tail[:400].decode("utf-8", "replace")})
    return {"query": query, "matches": out, "more": more}


def follow(path: Path, since: int, snap_token: str = "", limit: int = WINDOW_MAX) -> Dict[str, Any]:
    """Whole lines written after ``since`` (Follow in the text view): {text,
    end, more, replaced, live, finished, snap}. ``replaced`` when the file is
    not the one ``snap_token`` named (truncated by a resubmission, a new
    inode, a different head): the page offers Reload."""
    from mace.utils import livefile
    snap = livefile.snapshot(path)
    replaced = False
    if snap_token:
        try:
            ino, size, head = snap_token.split(":", 2)
            if int(ino) != snap.ino or snap.size < int(size):
                replaced = True
            else:
                # the token holds the hash of the first min(64 KB, size then) bytes
                import hashlib
                with open(path, "rb") as fh:
                    replaced = hashlib.sha1(fh.read(min(int(size), livefile.HEAD))).hexdigest()[:len(head)] != head
        except (ValueError, OSError):
            replaced = False
    since = max(0, min(int(since), snap.size))
    text, end = "", since
    if not replaced:
        bounded = livefile.Snap(min(snap.size, since + max(1, int(limit))), snap.mtime_ns, snap.ino, snap.dev,
                                snap.head_sha1, snap.head_len)
        data, end = livefile.read_complete(path, bounded, since)
        text = data.decode("utf-8", "replace")
    st = livefile.state(path)
    return {"text": text, "end": end, "more": end < snap.size and not replaced, "replaced": replaced,
            "live": st["live"], "finished": st["finished"], "age_s": st["age_s"], "snap": snap.token(),
            "size": snap.size}
