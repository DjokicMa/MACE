"""Per-document view options: ``$XDG_CONFIG_HOME/mace/gui-docprefs.json``.

A document (a structure, a plot) keeps only the options that differ from
the Settings defaults: its display range, boundary atoms, rendering, a
plot's axes. They are user choices, so they live next to gui.json (not in a
cache folder a cleanup may empty), and on the server, so they come back on a
new local port (an ssh tunnel, VS Code's port forwarding) and in a fresh
browser. Nothing here reaches a deck, a CIF or a settings file.

Shape: ``{"version": 1, "rev": n, "kinds": {kind: {dir_hash: {relpath:
{"t": time, "values": {...}}}}}}``. At most 300 documents per kind (the
least recently changed go first) and 256 kB; written per document under the
same lock and read-back check as gui.json (prefstore.py); mode 0600. A
corrupt file is renamed ``.bad`` (logged once) and the server starts with
none; so is a file that is JSON but not this shape (a hand edit).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from . import prefstore
from . import settings as gui_settings

KINDS = ("display", "plot")
MAX_DOCS = 300
MAX_BYTES = 256_000
MAX_VALUES = 64

_bad_logged = set()


def path() -> Path:
    return gui_settings.path().with_name("gui-docprefs.json")


def _shape_ok(data: Dict[str, Any]) -> bool:
    """Whether a stored object has the shape every reader and writer here
    relies on: {kind: {dir hash: {path: {t, values: {...}}}}}."""
    kinds = data.get("kinds", {})
    if not isinstance(kinds, dict):
        return False
    for dirs in kinds.values():
        if not isinstance(dirs, dict):
            return False
        for docs in dirs.values():
            if not isinstance(docs, dict):
                return False
            for e in docs.values():
                if (not isinstance(e, dict) or not isinstance(e.get("values", {}), dict)
                        or isinstance(e.get("t", 0), bool) or not isinstance(e.get("t", 0), (int, float))):
                    return False
    return True


def _read() -> Dict[str, Any]:
    p = path()
    try:
        data = prefstore.read_json(p)
        if data is not None and not _shape_ok(data):
            raise ValueError("not the per-document options shape")
    except ValueError:
        bad = p.with_name(p.name + ".bad")
        try:
            os.replace(p, bad)
        except OSError:
            pass
        if str(p) not in _bad_logged:
            _bad_logged.add(str(p))
            print(f"mace gui: {p} was not readable; kept as {bad.name}, starting with no per-document options",
                  flush=True)
        return {}
    return data or {}


def check_values(values: Any) -> Dict[str, str]:
    """{key: why} for the values that cannot be stored (the Settings checks, per key)."""
    if not isinstance(values, dict):
        return {"values": "must be an object"}
    if len(values) > MAX_VALUES:
        return {"values": f"has more than {MAX_VALUES} keys"}
    _, rejected, _ = gui_settings._clean(values)
    return rejected


def get(kind: str, dir_hash: str, rel: str) -> Dict[str, Any]:
    entry = (((_read().get("kinds") or {}).get(kind) or {}).get(dir_hash) or {}).get(rel)
    return dict(entry.get("values") or {}) if isinstance(entry, dict) else {}


def all_of(kind: Optional[str] = None) -> Dict[str, Any]:
    """Every stored document (the profile export)."""
    kinds = _read().get("kinds") or {}
    return {k: v for k, v in kinds.items() if kind is None or k == kind}


def _evict(kinds: Dict[str, Any]) -> None:
    """At most MAX_DOCS per kind, the least recently changed going first."""
    for kind, dirs in kinds.items():
        entries = [(e.get("t", 0), h, r) for h, docs in dirs.items() for r, e in docs.items()]
        if len(entries) <= MAX_DOCS:
            continue
        entries.sort()
        for _, h, r in entries[:len(entries) - MAX_DOCS]:
            dirs[h].pop(r, None)
            if not dirs[h]:
                dirs.pop(h)


def put(kind: str, dir_hash: str, rel: str, values: Dict[str, Any], now: Optional[float] = None) -> int:
    """Store one document's overrides ({} deletes the entry); returns the file's revision."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(KINDS)}")
    bad = check_values(values)
    if bad:
        k, why = next(iter(bad.items()))
        raise ValueError(f"{k} {why}")
    t = time.time() if now is None else now
    p = path()
    _read()          # a corrupt file is set aside before the write

    def change(cur):
        new = dict(cur if cur and _shape_ok(cur) else {})
        new["version"] = max(1, new.get("version", 1) if isinstance(new.get("version"), int) else 1)
        new["rev"] = (new.get("rev") if isinstance(new.get("rev"), int) else 0) + 1
        kinds = json.loads(json.dumps(new.get("kinds") or {}))
        docs = kinds.setdefault(kind, {}).setdefault(dir_hash, {})
        if values:
            docs[rel] = {"t": t, "values": values}
        else:
            docs.pop(rel, None)
            if not docs:
                kinds[kind].pop(dir_hash, None)
        _evict(kinds)
        new["kinds"] = kinds
        # over the size cap: the oldest documents of any kind go until it fits
        while len(json.dumps(new)) > MAX_BYTES - 1000:
            entries = sorted((e.get("t", 0), k, h, r) for k, dirs in kinds.items() for h, d in dirs.items()
                             for r, e in d.items())
            if not entries:
                break
            _, k, h, r = entries[0]
            kinds[k][h].pop(r)
            if not kinds[k][h]:
                kinds[k].pop(h)
        return new

    def holds(back):
        got = (((back.get("kinds") or {}).get(kind) or {}).get(dir_hash) or {}).get(rel)
        return (got or {}).get("values") == values if values else got is None

    new = prefstore.update(p, change, holds, MAX_BYTES)
    return new["rev"]


def reset() -> None:
    """Every document's view options gone (a copy of the file is kept first)."""
    p = path()
    gui_settings.keep_reset_copy(p)

    def change(cur):
        new = dict(cur or {})
        new["version"] = 1
        new["rev"] = (new.get("rev") if isinstance(new.get("rev"), int) else 0) + 1
        new["kinds"] = {}
        return new

    prefstore.update(p, change, lambda back: not back.get("kinds"), MAX_BYTES)


def import_all(kinds: Any) -> int:
    """Merge an exported {kind: {dir hash: {path: {t, values}}}} (the profile
    export); each document's values pass the same checks. Returns how many."""
    if not isinstance(kinds, dict):
        raise ValueError("docprefs must be {kind: {folder hash: {path: {t, values}}}}")
    n = 0
    for kind, dirs in kinds.items():
        if kind not in KINDS or not isinstance(dirs, dict):
            continue
        for h, docs in dirs.items():
            if not isinstance(h, str) or len(h) > 64 or not isinstance(docs, dict):
                continue
            for rel, e in docs.items():
                if (not isinstance(rel, str) or rel.startswith("/") or ".." in rel.split("/")
                        or not isinstance(e, dict) or check_values(e.get("values"))):
                    continue
                t = e.get("t") if isinstance(e.get("t"), (int, float)) else None
                put(kind, h, rel, e["values"], now=t)
                n += 1
    return n
