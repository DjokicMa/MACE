"""The GUI's own settings file: ``$XDG_CONFIG_HOME/mace/gui.json``.

It lives outside the served directory, so ``mace gui`` still never writes
into ``--dir`` on its own. The browser keeps the full effective settings in
localStorage and sends only the keys that differ from the defaults; the
file is the cross-device copy (one cluster account, several laptops).

Values are checked one key at a time, for their type and wide fixed bounds
only, never against this host's limits: a file written on a 128-core node
must still load and save on a 24-core workstation (resources.py applies the
host's limits when a value is used). A bad key never blocks the others: it
is reported and not written, and the value stored before is kept. Unknown
keys are kept, because the browser owns the schema and an older server must
not drop a newer client's settings.

File format 2 (the 1.1.5 polish round) makes it the whole GUI profile, so a
new local port or a fresh browser gets everything back:
``{"version": 2, "rev": n, "settings": {...}, "ui": {...}, "memory": {...},
"recent": {"files": {<dir hash>: {root, items}}, "folders": [...]}}``.
``settings`` keeps its meaning and checks; ``ui`` is the layout and panel
state, ``memory`` the remembered choices (last export, New Structure form,
Clean method, ...), ``recent`` the Recent lists. A version-1 file is read
as it is and copied once to ``gui.json.v1.bak`` before the first version-2
write. A newer file's sections this server does not know are kept. Writes
are per-key patches under a lock (prefstore.py), so two servers saving at
once keep both changes.
"""
from __future__ import annotations

import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

VERSION = 2
MAX_BYTES = 256_000
SECTIONS = ("settings", "ui", "memory", "recent")
MAX_SECTION = 32_000
MAX_DEPTH = 3
MAX_RECENT_ITEMS = 100
MAX_RECENT_FOLDERS = 50
MAX_FOLDERS = 20
MAX_KEY = 80
MAX_STRING = 2000
MAX_KEYMAP = 1000
MAX_LIST = 1000

_GPU = re.compile(r"^(none|\d{1,2})$")
# numbers with a fixed range: (low, high). The viewer's (polish A3/A6), then
# plots, run analysis, live files and File History (PLAN_analysis C3, C17, C18, C20)
_RANGES = {"depthCue": (0.0, 1.0), "ambient": (0.0, 1.0), "shininess": (0.0, 1.0), "ambientOcclusion": (0.0, 1.0),
           "lightStrength": (0.3, 1.6), "outlineWidth": (0.02, 0.3), "atomScale": (0.05, 1.5), "bondRadius": (0.01, 1.0),
           "files.liveMinutes": (1, 1440), "files.historyKeep": (1, 1000), "files.historyDays": (1, 3650),
           "files.historyMaxMB": (10, 100000), "files.historyOutputsMaxMB": (10, 100000),
           "analysis.imagNoiseCm": (0, 200), "plots.lineWidth": (0.5, 6), "plots.exportScale": (1, 8)}
# one of a fixed set of values
_CHOICES = {"lightDirection": ("front", "upper-left", "top"), "bondedOutside": ("off", "neighbors", "molecules"),
            "renderPreset": ("default", "flat", "publication", "presentation", "custom"),
            "database.refreshSeconds": (0, 15, 30, 60),
            "plots.relTo": ("first", "final", "minimum", "absolute"),
            "plots.per": ("cell", "atom", "fu"),
            "energyUnit": ("eV", "meV", "Ha", "kJ/mol", "kcal/mol")}
_RANGE_KEYS = ("displayLo", "displayHi")
RANGE_LIMITS = (-5.0, 10.0)
RANGE_MIN_SPAN = 0.05
# plan E keys with a fixed set of values
ENUMS = {
    "files.openFreqAs": ("modes", "structure"),
    "volume.quality": ("low", "medium", "high"),
    "volume.mepColors": ("cli", "chemistry"),
    "spectra.lineShape": ("lorentzian", "gaussian"),
    "export.units": ("in", "cm"),
    "export.vectorShading": ("flat", "shaded"),
}


def path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return Path(base) / "mace" / "gui.json"


def _scalar_problem(v: Any) -> Optional[str]:
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return None if math.isfinite(v) else "must be a finite number"
    if isinstance(v, str):
        return None if len(v) <= MAX_STRING else f"is longer than {MAX_STRING} characters"
    return "must be a number, text, true/false or null"


def _int_in(v: Any, lo: int, hi: int, unit: str = "") -> Optional[str]:
    if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
        return f"must be a whole number from {lo} to {hi}{unit}"
    return None


def keymap_entries(value: Any) -> Tuple[List[Dict[str, Any]], int]:
    """The well-formed {command, keys, remove?} entries, and how many were dropped."""
    if not isinstance(value, list):
        return [], 0
    good, bad = [], 0
    for e in value[:MAX_KEYMAP]:
        ok = (isinstance(e, dict) and set(e) <= {"command", "keys", "remove"}
              and isinstance(e.get("command"), str) and 0 < len(e["command"]) <= 120
              and ("keys" not in e or (isinstance(e["keys"], list) and len(e["keys"]) <= 8
                                       and all(isinstance(k, str) and 0 < len(k) <= 40 for k in e["keys"])))
              and ("remove" not in e or isinstance(e["remove"], bool))
              and ("keys" in e or e.get("remove") is True))
        if ok:
            good.append(e)
        else:
            bad += 1
    return good, bad + max(0, len(value) - MAX_KEYMAP)


def check(key: Any, value: Any) -> Optional[str]:
    """Why this key/value cannot be stored, or None."""
    if not isinstance(key, str) or not key or len(key) > MAX_KEY:
        return f"is not a settings key (1 to {MAX_KEY} characters)"
    if key == "max_workers":
        return _int_in(value, 0, 1024)
    if key == "cache_mb":
        return _int_in(value, 64, 65536, " MB")
    if key == "gpu_device":
        return None if isinstance(value, str) and _GPU.match(value) else 'must be "none" or a GPU number'
    if key == "keymap":
        return None if isinstance(value, list) else "must be a list of {command, keys, remove?}"
    if key in _RANGE_KEYS:
        lo, hi = RANGE_LIMITS
        if (not isinstance(value, list) or len(value) != 3
                or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or not lo <= x <= hi
                       for x in value)):
            return f"must be a list of 3 numbers from {lo:g} to {hi:g}"
        return None
    if key in _CHOICES:
        return None if not isinstance(value, bool) and value in _CHOICES[key] else "must be one of " + ", ".join(map(str, _CHOICES[key]))
    if key == "outlineColor":
        return None if isinstance(value, str) and (value == "" or re.fullmatch(r"#[0-9a-fA-F]{6}", value)) else \
            'must be "" or a color like #1a2b3c'
    if key in _RANGES:
        lo, hi = _RANGES[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lo <= value <= hi:
            return f"must be a number from {lo:g} to {hi:g}"
        return None
    # sessions (PLAN_connect D16): sign out after N hours without use, at most N days, idle stop
    if key == "session.idleHours":
        return None if value is None or (not isinstance(value, bool) and value in (0, 1, 4, 12, 24, 72)) \
            else "must be 0 or null (never) or one of 1, 4, 12, 24, 72 hours"
    if key == "session.maxDays":
        return _int_in(value, 1, 30, " days")
    if key == "server.idleStopHours":
        return None if value is None else _int_in(value, 0, 72, " hours (0 = never)")
    if key == "cluster.pollSeconds":
        return _int_in(value, 30, 300, " s")
    if key == "cluster.showView":
        return None if value in ("auto", "on", "off") else 'must be "auto", "on" or "off"'
    if key == "cluster.notify":
        return None if isinstance(value, bool) else "must be true or false"
    if key in ENUMS:
        return None if value in ENUMS[key] else "must be one of " + ", ".join(map(str, ENUMS[key]))
    if key == "export.dpi":
        return _int_in(value, 30, 2400, " DPI")
    if key == "notifications.infoSeconds":
        return _int_in(value, 2, 60, " s")
    if key == "structure.vacuumGap":
        return None if isinstance(value, (int, float)) and not isinstance(value, bool) and 2 <= value <= 100 \
            else "must be a number from 2 to 100 Å"
    if isinstance(value, list):
        if len(value) > MAX_LIST:
            return f"has more than {MAX_LIST} items"
        for x in value:
            p = _scalar_problem(x)
            if p:
                return f"has an item that {p}"
        return None
    if isinstance(value, dict):
        for k, x in value.items():
            if not isinstance(k, str) or len(k) > MAX_KEY or _scalar_problem(x):
                return "must hold only short keys with plain values"
        return None
    return _scalar_problem(value)


def _clean(values: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str], Dict[str, str]]:
    """(the storable values, {key: why not}, {key: note}) for a settings object."""
    good, rejected, notes = {}, {}, {}
    for k, v in values.items():
        why = check(k, v)
        if why:
            rejected[str(k)[:MAX_KEY]] = why
            continue
        if k == "keymap":
            v, dropped = keymap_entries(v)
            if dropped:
                notes[k] = f"{dropped} entr{'y was' if dropped == 1 else 'ies were'} not {{command, keys, remove?}} and dropped"
        good[k] = v
    # the display range pair: hi above lo by at least 0.05 on every axis
    if "displayLo" in good and "displayHi" in good:
        if any(h - l < RANGE_MIN_SPAN for l, h in zip(good["displayLo"], good["displayHi"])):
            why = f"displayHi must exceed displayLo by at least {RANGE_MIN_SPAN:g} on every axis"
            for k in _RANGE_KEYS:
                good.pop(k)
                rejected[k] = why
    return good, rejected, notes


_logged = set()
_cache: Dict[str, Any] = {"key": None, "data": None}


def _raw() -> Dict[str, Any]:
    """The file as stored (any version), {} when absent or unreadable; cached by mtime and size."""
    p = path()
    try:
        st = p.stat()
    except OSError:
        return {}
    key = (str(p), st.st_mtime_ns, st.st_size)
    if _cache["key"] == key:
        return _cache["data"]
    try:
        data = json.loads(p.read_text())
    except (OSError, ValueError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    _cache.update(key=key, data=data)
    return data


def load() -> Dict[str, Any]:
    """The stored settings; keys that fail the checks (a hand-edited or older
    file) are left out, each logged once, so a bad file never breaks boot."""
    p = path()
    values = _raw().get("settings", {})
    if not isinstance(values, dict):
        return {}
    good, rejected, _ = _clean(values)
    for k, why in rejected.items():
        if (str(p), k) not in _logged:
            _logged.add((str(p), k))
            print(f"mace gui: ignoring {k} in {p}: it {why}", flush=True)
    return good


# ---------------------------------------------------------------- the profile sections
def _depth(v: Any, d: int = 0) -> int:
    if isinstance(v, dict):
        return max([d + 1] + [_depth(x, d + 1) for x in v.values()])
    if isinstance(v, list):
        return max([d + 1] + [_depth(x, d + 1) for x in v])
    return d


def _plain(v: Any) -> Optional[str]:
    """Why a ui/memory value cannot be stored (JSON of plain values, short strings), or None."""
    if isinstance(v, dict):
        for k, x in v.items():
            if not isinstance(k, str) or len(k) > MAX_KEY:
                return "has a key that is not short text"
            p = _plain(x)
            if p:
                return p
        return None
    if isinstance(v, list):
        if len(v) > MAX_LIST:
            return f"has more than {MAX_LIST} items"
        for x in v:
            p = _plain(x)
            if p:
                return p
        return None
    return _scalar_problem(v)


def check_section_value(section: str, key: Any, value: Any) -> Optional[str]:
    """Why ui/memory[key] = value cannot be stored, or None."""
    if not isinstance(key, str) or not key or len(key) > MAX_KEY:
        return f"is not a key (1 to {MAX_KEY} characters)"
    if _depth(value) > MAX_DEPTH:
        return f"is nested more than {MAX_DEPTH} levels"
    return _plain(value)


def _rel_ok(rel: Any) -> bool:
    return (isinstance(rel, str) and 0 < len(rel) <= 1000 and not rel.startswith("/")
            and ".." not in rel.split("/"))


def check_recent_files(entry: Any) -> Optional[str]:
    """{root, items: [{path, t, pinned}]} for one folder."""
    if not isinstance(entry, dict) or not isinstance(entry.get("root"), str) or not entry["root"].startswith("/"):
        return "must be {root: an absolute path, items: [...]}"
    items = entry.get("items", [])
    if not isinstance(items, list) or len(items) > MAX_RECENT_ITEMS:
        return f"items must be a list of at most {MAX_RECENT_ITEMS}"
    for it in items:
        if (not isinstance(it, dict) or not _rel_ok(it.get("path"))
                or isinstance(it.get("t", 0), bool) or not isinstance(it.get("t", 0), (int, float))
                or not isinstance(it.get("pinned", False), bool)):
            return "has an item that is not {path: a relative path inside the folder, t, pinned}"
    return None


def check_folders(v: Any) -> Optional[str]:
    if (not isinstance(v, list) or len(v) > MAX_FOLDERS
            or any(not isinstance(x, str) or not x.startswith("/") or len(x) > 4000 for x in v)):
        return f"must be a list of at most {MAX_FOLDERS} absolute paths"
    return None


def profile_of(data: Dict[str, Any]) -> Dict[str, Any]:
    """The checked profile from a stored object of any version."""
    good, _, _ = _clean(data.get("settings", {}) if isinstance(data.get("settings"), dict) else {})
    out = {"rev": data.get("rev", 0) if isinstance(data.get("rev"), int) else 0, "settings": good}
    for sec in ("ui", "memory"):
        raw = data.get(sec) if isinstance(data.get(sec), dict) else {}
        out[sec] = {k: v for k, v in raw.items() if check_section_value(sec, k, v) is None}
    rec = data.get("recent") if isinstance(data.get("recent"), dict) else {}
    files = rec.get("files") if isinstance(rec.get("files"), dict) else {}
    out["recent"] = {"files": {h: e for h, e in files.items() if isinstance(h, str) and len(h) <= 64
                               and check_recent_files(e) is None},
                     "folders": rec.get("folders", []) if check_folders(rec.get("folders", [])) is None else []}
    return out


def load_profile() -> Dict[str, Any]:
    return profile_of(_raw())


def _new_version(cur: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The stored object to write over `cur`: version 2 with every section;
    a newer file keeps its version and its unknown sections."""
    base = dict(cur or {})
    v = base.get("version") if isinstance(base.get("version"), int) else 1
    base["version"] = max(VERSION, v)
    for sec in ("settings", "ui", "memory"):
        if not isinstance(base.get(sec), dict):
            base[sec] = {}
    rec = dict(base["recent"]) if isinstance(base.get("recent"), dict) else {}
    if not isinstance(rec.get("files"), dict):
        rec["files"] = {}
    if not isinstance(rec.get("folders"), list):
        rec["folders"] = []
    base["recent"] = rec
    base["rev"] = (base.get("rev") if isinstance(base.get("rev"), int) else 0) + 1
    return base


def _keep_v1_copy(p: Path, cur: Optional[Dict[str, Any]]) -> None:
    """Before the first version-2 write over a version-1 file: one copy, never replaced."""
    if not cur or (isinstance(cur.get("version"), int) and cur["version"] >= 2):
        return
    bak = p.with_name(p.name + ".v1.bak")
    try:
        fd = os.open(str(bak), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return
    with os.fdopen(fd, "wb") as fh:
        fh.write(p.read_bytes())


def _trim_recent(files: Dict[str, Any]) -> Dict[str, Any]:
    """At most MAX_RECENT_FOLDERS folders: the most recently used ones."""
    if len(files) <= MAX_RECENT_FOLDERS:
        return files
    last = lambda e: max([it.get("t", 0) for it in e.get("items", [])] + [e.get("t", 0)])  # noqa: E731
    keep = sorted(files.items(), key=lambda kv: last(kv[1]), reverse=True)[:MAX_RECENT_FOLDERS]
    return dict(keep)


def patch(body: Any) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Merge a patch into the stored profile, per key, under the lock.

    body: {"settings": {"set": {...}, "unset": [...]}, "ui": {...}, "memory": {...},
    "recent": {"files": {<dir hash>: entry or null}, "folders": [...]}}.
    Returns (the stored profile, {"section.key": why rejected}). A rejected
    settings key keeps its stored value; a malformed section raises ValueError."""
    from . import prefstore
    if not isinstance(body, dict):
        raise ValueError("the profile patch must be a JSON object")
    rejected: Dict[str, str] = {}
    sets: Dict[str, Dict[str, Any]] = {}
    unsets: Dict[str, List[str]] = {}
    for sec in ("settings", "ui", "memory"):
        part = body.get(sec)
        if part is None:
            continue
        if not isinstance(part, dict) or not set(part) <= {"set", "unset"}:
            raise ValueError(f"{sec} must be {{set: {{...}}, unset: [...]}}")
        st, un = part.get("set") or {}, part.get("unset") or []
        if not isinstance(st, dict) or not isinstance(un, list) or not all(isinstance(k, str) for k in un):
            raise ValueError(f"{sec} must be {{set: {{...}}, unset: [...]}}")
        if sec == "settings":
            good, bad, _ = _clean(st)
            for k, why in bad.items():
                rejected[f"settings.{k}"] = why
            st = good
        else:
            size = len(json.dumps(st))
            if size > MAX_SECTION:
                raise ValueError(f"{sec} is larger than {MAX_SECTION // 1000} kB")
            good = {}
            for k, v in st.items():
                why = check_section_value(sec, k, v)
                if why:
                    raise ValueError(f"{sec}.{k} {why}")
                good[k] = v
            st = good
        sets[sec], unsets[sec] = st, un
    rec = body.get("recent")
    rec_files, rec_folders = None, None
    if rec is not None:
        if not isinstance(rec, dict):
            raise ValueError("recent must be {files: {...}, folders: [...]}")
        if "files" in rec:
            if not isinstance(rec["files"], dict):
                raise ValueError("recent.files must be {<folder hash>: {root, items}}")
            for h, e in rec["files"].items():
                if not isinstance(h, str) or len(h) > 64:
                    raise ValueError("recent.files keys are folder hashes")
                if e is not None:
                    why = check_recent_files(e)
                    if why:
                        raise ValueError(f"recent.files {why}")
            rec_files = rec["files"]
        if "folders" in rec:
            why = check_folders(rec["folders"])
            if why:
                raise ValueError(f"recent.folders {why}")
            rec_folders = rec["folders"]
    p = path()

    def change(cur):
        _keep_v1_copy(p, cur)
        new = _new_version(cur)
        for sec, st in sets.items():
            d = dict(new[sec])
            for k in unsets[sec]:
                d.pop(k, None)
            d.update(st)
            new[sec] = d
        if rec_files is not None:
            files = dict(new["recent"].get("files") or {})
            for h, e in rec_files.items():
                if e is None:
                    files.pop(h, None)
                else:
                    files[h] = e
            new["recent"] = {**new["recent"], "files": _trim_recent(files)}
        if rec_folders is not None:
            new["recent"] = {**new["recent"], "folders": rec_folders}
        return new

    def holds(back):
        for sec, st in sets.items():
            got = back.get(sec) or {}
            if any(got.get(k) != v for k, v in st.items()) or any(k in got for k in unsets[sec]):
                return False
        if rec_files is not None:
            got = (back.get("recent") or {}).get("files") or {}
            if any((got.get(h) != e) for h, e in rec_files.items() if e is not None and h in got):
                return False
        return True

    prefstore.update(p, change, holds, MAX_BYTES)
    return load_profile(), rejected


def save(values: Any) -> Tuple[Path, List[str], Dict[str, str], Dict[str, str]]:
    """Store the non-default keys `values` (the whole settings section), one key at a time.

    Returns (path, saved keys, {rejected key: reason}, {key: note}). A
    rejected key keeps the value stored before (if any). Raises ValueError
    only for a request that is not a settings object at all, or too large.
    The other profile sections are kept (a page from before the profile
    sends this)."""
    from . import prefstore
    if not isinstance(values, dict):
        raise ValueError("settings must be a JSON object")
    good, rejected, notes = _clean(values)
    p = path()

    def change(cur):
        _keep_v1_copy(p, cur)
        new = _new_version(cur)
        before, _, _ = _clean(new["settings"]) if isinstance(new.get("settings"), dict) else ({}, {}, {})
        stored = dict(good)
        for k in rejected:
            if k in before:
                stored[k] = before[k]
        new["settings"] = stored
        return new

    prefstore.update(p, change, lambda back: all((back.get("settings") or {}).get(k) == v for k, v in good.items()),
                     MAX_BYTES)
    saved = sorted(k for k in good if k not in rejected)
    return p, saved, rejected, notes


# ---------------------------------------------------------------- Reset (Settings > Advanced)
KEEP_RESET_COPIES = 3
RESETTABLE = ("settings", "ui", "memory", "recent")


def keep_reset_copy(p: Path) -> Optional[Path]:
    """A copy of the file before a reset: <name>.before-reset-<UTC time>.json,
    0600, never replacing one; only the newest KEEP_RESET_COPIES are kept."""
    import time as _time
    if not p.exists():
        return None
    stamp = _time.strftime("%Y%m%dT%H%M%SZ", _time.gmtime())
    for n in range(100):
        dest = p.with_name(f"{p.name}.before-reset-{stamp}{'' if not n else f'-{n}'}.json")
        try:
            fd = os.open(str(dest), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            continue
        with os.fdopen(fd, "wb") as fh:
            fh.write(p.read_bytes())
        break
    else:
        return None
    copies = sorted(p.parent.glob(f"{p.name}.before-reset-*.json"), key=lambda x: (x.stat().st_mtime_ns, x.name))
    for old in copies[:-KEEP_RESET_COPIES]:
        try:
            old.unlink()
        except OSError:
            pass
    return dest


def reset_sections(sections: Any) -> Dict[str, Any]:
    """Empty the chosen profile sections (a copy of the file is kept first)."""
    from . import prefstore
    if not isinstance(sections, list) or not sections or not set(sections) <= set(RESETTABLE):
        raise ValueError(f"sections must be a list of {', '.join(RESETTABLE)}")
    p = path()
    keep_reset_copy(p)

    def change(cur):
        _keep_v1_copy(p, cur)
        new = _new_version(cur)
        for sec in sections:
            new[sec] = {"files": {}, "folders": []} if sec == "recent" else {}
        return new

    prefstore.update(p, change, lambda back: all(not back.get(sec) or back.get(sec) == {"files": {}, "folders": []}
                                                  for sec in sections), MAX_BYTES)
    return load_profile()
