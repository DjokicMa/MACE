"""Parsed files, held structures, packed payloads and provenance (spec
sections 8.4.2, 8.4.5 and 9.2; plan B2, B5, B7).

* ``ParseCache``: an LRU of parsed structures and response bodies, keyed by
  the file's (realpath, mtime_ns, size, PARSER_VERSION), so a second open of
  an unchanged file neither reads nor parses it; its key gives the ETag.
* ``HeldStore``: the structures that exist only in the server and are
  view-only (built or dropped, over 5000 atoms or not 3D). Each is pinned by
  a lease of the page's event stream that made it: the stream closing (a
  reload, a crash, a closed window) ends it after a grace period, unless the
  reloaded page claims it; an unpinned entry is evicted, oldest first.
* Both report their bytes to the one MemoryLedger (cache_mb).
* ``pack()``: a structure as base64 typed arrays (about 3 B per atom
  gzipped for a supercell).
* Provenance sidecars: a saved file's "built from" record and charge hint,
  under the user's cache folder keyed by the file's sha256, never in --dir.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

PARSER_VERSION = 1
MAX_ENTRIES = 64
ENTRY_OVERHEAD = 2048
LEASE_GRACE_S = 30.0


def file_key(path: Path, extra: Tuple = ()) -> Tuple:
    st = os.stat(path)
    return (os.path.realpath(path), int(st.st_mtime_ns), int(st.st_size), PARSER_VERSION) + tuple(extra)


def etag_of(key: Tuple) -> str:
    return '"' + hashlib.sha1(repr(key).encode()).hexdigest()[:24] + '"'


class ParseCache:
    def __init__(self, max_entries: int = MAX_ENTRIES):
        self.max_entries = max_entries
        self.lock = threading.Lock()
        self.items: "OrderedDict[Tuple, Tuple[Any, int]]" = OrderedDict()
        self.bytes = 0
        self.hits = self.misses = 0

    def get(self, key: Tuple) -> Any:
        with self.lock:
            hit = self.items.get(key)
            if hit is None:
                self.misses += 1
                return None
            self.items.move_to_end(key)
            self.hits += 1
            return hit[0]

    def put(self, key: Tuple, value: Any, nbytes: int) -> None:
        with self.lock:
            old = self.items.pop(key, None)
            if old is not None:
                self.bytes -= old[1]
            self.items[key] = (value, int(nbytes) + ENTRY_OVERHEAD)
            self.bytes += int(nbytes) + ENTRY_OVERHEAD
            while len(self.items) > self.max_entries:
                _, (_, n) = self.items.popitem(last=False)
                self.bytes -= n

    def forget(self, prefix_path: str) -> None:
        with self.lock:
            for k in [k for k in self.items if k[0] == prefix_path]:
                self.bytes -= self.items.pop(k)[1]

    # the ledger's interface
    def used(self) -> int:
        return self.bytes

    def drop_oldest(self) -> bool:
        with self.lock:
            if not self.items:
                return False
            _, (_, n) = self.items.popitem(last=False)
            self.bytes -= n
            return True

    def report(self) -> Dict[str, Any]:
        with self.lock:
            return {"entries": len(self.items), "bytes": self.bytes, "hits": self.hits, "misses": self.misses}


def dir_bytes(path: Path) -> int:
    """What a folder holds on disk (allocated blocks, so a sparse file counts what it uses)."""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for name in filenames:
            try:
                st = os.lstat(os.path.join(dirpath, name))
            except OSError:
                continue
            total += getattr(st, "st_blocks", 0) * 512 or st.st_size
    return total


class DiskCache:
    """Folders the server writes under its own cache directory: the plot
    renders (``mace plotting``) and the Setup runs (plan T1). The cache
    directory is under /tmp, which is often RAM, so their bytes count
    against the one budget (cache_mb) like the parse cache: when it is over,
    the least recently used folder goes, never one a task is using."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.lock = threading.Lock()
        self.items: "OrderedDict[str, int]" = OrderedDict()     # key -> bytes
        self.in_use: Dict[str, int] = {}
        self.dropped = 0

    def path(self, key: str) -> Path:
        return self.root / key

    def acquire(self, key: str) -> Path:
        """The folder for ``key`` (made if missing), held until release()."""
        p = self.path(key)
        p.mkdir(parents=True, exist_ok=True)
        with self.lock:
            self.in_use[key] = self.in_use.get(key, 0) + 1
            self.items.setdefault(key, 0)
            self.items.move_to_end(key)
        return p

    def release(self, key: str, remove: bool = False) -> None:
        """Done with ``key``: its size is counted again, or the folder goes (``remove``)."""
        p = self.path(key)
        if remove:
            import shutil
            shutil.rmtree(p, ignore_errors=True)
        with self.lock:
            n = self.in_use.get(key, 0) - 1
            if n > 0:
                self.in_use[key] = n
            else:
                self.in_use.pop(key, None)
            if remove:
                self.items.pop(key, None)
            elif key in self.items:
                self.items[key] = dir_bytes(p) if p.is_dir() else 0
                self.items.move_to_end(key)

    def touch(self, key: str) -> bool:
        """A reuse of a finished folder: most recently used now; False when it is gone."""
        with self.lock:
            if key not in self.items or not self.path(key).is_dir():
                self.items.pop(key, None)
                return False
            self.items.move_to_end(key)
            return True

    def used(self) -> int:
        with self.lock:
            return sum(self.items.values())

    def drop_oldest(self) -> bool:
        import shutil
        with self.lock:
            key = next((k for k in self.items if k not in self.in_use), None)
            if key is None:
                return False
            self.items.pop(key)
            self.dropped += 1
        shutil.rmtree(self.path(key), ignore_errors=True)
        return True

    def report(self) -> Dict[str, Any]:
        with self.lock:
            return {"entries": len(self.items), "bytes": sum(self.items.values()), "in_use": len(self.in_use),
                    "dropped": self.dropped}


class HeldBudget(ValueError):
    """A held structure would pass the pinned budget (a quarter of cache_mb)."""


class HeldStore:
    def __init__(self, cap_fn):
        self.cap_fn = cap_fn               # the pinned cap in bytes, read at each admission
        self.lock = threading.Lock()
        self.items: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self.grace_s = LEASE_GRACE_S
        self._timers: Dict[str, threading.Timer] = {}
        self.max_pinned = 0

    # -- entries ------------------------------------------------------------
    def pinned_bytes(self) -> int:
        with self.lock:
            return sum(e["nbytes"] for e in self.items.values() if e["pinned"])

    def pinned_count(self) -> int:
        with self.lock:
            return sum(1 for e in self.items.values() if e["pinned"])

    def check_room(self, nbytes: int) -> None:
        """Refuse a build before it runs when its result would pass the cap."""
        cap = int(self.cap_fn())
        with self.lock:
            pinned = [e for e in self.items.values() if e["pinned"]]
        held = sum(e["nbytes"] for e in pinned)
        if held + nbytes > cap:
            raise HeldBudget(f"Close some of the big built structures first ({len(pinned)} open, holding "
                             f"{held / 2**20:.0f} MB; this one needs {nbytes / 2**20:.0f} MB of the "
                             f"{cap / 2**20:.0f} MB they may use)")

    def put(self, structure: Dict[str, Any], nbytes: int, client: Optional[str], meta: Optional[Dict[str, Any]] = None) -> str:
        self.check_room(nbytes)
        hid = "h" + secrets.token_hex(8)
        with self.lock:
            self.items[hid] = {"id": hid, "structure": structure, "nbytes": int(nbytes) + ENTRY_OVERHEAD,
                               "pinned": True, "client": client, "meta": meta or {}, "t": time.time()}
            self.max_pinned = max(self.max_pinned, sum(e["nbytes"] for e in self.items.values() if e["pinned"]))
        return hid

    def get(self, hid: str) -> Dict[str, Any]:
        with self.lock:
            e = self.items.get(hid or "")
            if e is not None:
                self.items.move_to_end(hid)
        if e is None:
            raise KeyError("built structure (the server let go of it; rebuild it)")
        return e

    def release(self, ids: Iterable[str]) -> int:
        n = 0
        with self.lock:
            for hid in ids:
                if self.items.pop(str(hid), None) is not None:
                    n += 1
        return n

    def claim(self, ids: Iterable[str], client: Optional[str]) -> List[str]:
        """A reloaded page takes over its documents' entries (pinned again)."""
        got = []
        with self.lock:
            for hid in ids:
                e = self.items.get(str(hid))
                if e is not None:
                    e["client"], e["pinned"] = client, True
                    got.append(e["id"])
        return got

    # -- leases -------------------------------------------------------------
    def connected(self, client: str) -> None:
        t = self._timers.pop(client, None)
        if t is not None:
            t.cancel()

    def disconnected(self, client: str) -> None:
        """The client's event stream closed: its entries unpin after the grace period."""
        if not client:
            return
        t = threading.Timer(self.grace_s, self._expire, args=(client,))
        t.daemon = True
        old = self._timers.pop(client, None)
        if old is not None:
            old.cancel()
        self._timers[client] = t
        t.start()

    def _expire(self, client: str) -> None:
        self._timers.pop(client, None)
        with self.lock:
            for e in self.items.values():
                if e["client"] == client:
                    e["pinned"] = False

    # -- the ledger's interface ----------------------------------------------
    def used(self) -> int:
        with self.lock:
            return sum(e["nbytes"] for e in self.items.values())

    def drop_oldest(self) -> bool:
        """The oldest unpinned entry (pinned ones are never evicted)."""
        with self.lock:
            for hid, e in self.items.items():
                if not e["pinned"]:
                    del self.items[hid]
                    return True
        return False

    def report(self) -> Dict[str, Any]:
        with self.lock:
            return {"entries": len(self.items), "pinned": sum(1 for e in self.items.values() if e["pinned"]),
                    "pinned_bytes": sum(e["nbytes"] for e in self.items.values() if e["pinned"]),
                    "bytes": sum(e["nbytes"] for e in self.items.values()), "max_pinned_bytes": self.max_pinned}


# ---------------------------------------------------------------- packed payloads
def _b64(arr) -> str:
    return base64.b64encode(arr.tobytes()).decode("ascii")


def pack(payload: Dict[str, Any], frac=None) -> Dict[str, Any]:
    """A structure payload (structures._payload) as packed arrays: cell (9
    numbers), zn (base64 Uint8, or Uint16 as zn16 for CRYSTAL's ECP numbers
    above 255) and frac (base64 little-endian Float64). symbols, radii,
    metal and cart are left out: the page computes them from zn, frac and the
    cell. Everything else (symmetry, lattice, formula, ...) is kept."""
    import numpy as np
    numbers = np.asarray(payload["numbers"], dtype=np.int64)
    f = np.asarray(payload["frac"] if frac is None else frac, dtype="<f8").reshape(-1, 3)
    out = {k: v for k, v in payload.items() if k not in ("numbers", "symbols", "radii", "metal", "cart", "frac", "cell")}
    out["packed"] = 1
    out["cell"] = [float(x) for x in np.asarray(payload["cell"], dtype=float).ravel()]
    out["n"] = int(len(numbers))
    if len(numbers) and (numbers.max() > 255 or numbers.min() < 0):
        out["zn16"] = _b64(numbers.astype("<u2"))
    else:
        out["zn"] = _b64(numbers.astype(np.uint8))
    out["frac"] = _b64(np.ascontiguousarray(f))
    if any(not p for p in payload.get("pbc", [True, True, True])):
        # a slab's or a molecule's Cartesian positions along its non-periodic axes
        out["cart"] = _b64(np.ascontiguousarray(np.asarray(payload["cart"], dtype="<f8").reshape(-1, 3)))
    return out


def unpack_arrays(p: Dict[str, Any]):
    """(cell 3x3, frac Nx3, numbers N) from a packed payload."""
    import numpy as np
    cell = np.asarray(p["cell"], float).reshape(3, 3)
    z = np.frombuffer(base64.b64decode(p["zn16"]), "<u2") if "zn16" in p else np.frombuffer(base64.b64decode(p["zn"]), np.uint8)
    frac = np.frombuffer(base64.b64decode(p["frac"]), "<f8").reshape(-1, 3)
    return cell, frac.copy(), z.astype(int)


# ---------------------------------------------------------------- provenance sidecars
def state_dir() -> Path:
    """MACE_GUI_STATE_DIR, else $XDG_CACHE_HOME/mace/gui, else ~/.cache/mace/gui:
    outside every served folder, kept across restarts."""
    v = os.environ.get("MACE_GUI_STATE_DIR")
    if v:
        return Path(v)
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "mace" / "gui"


MAX_SIDECARS = 2000
SIDECAR_MAX_FILE = 64 * 1024 * 1024


def sha256_file(path: Path) -> Optional[str]:
    try:
        if path.stat().st_size > SIDECAR_MAX_FILE:
            return None
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def write_sidecar(data: bytes, meta: Dict[str, Any]) -> Optional[Path]:
    """Remember a saved file's provenance and hints by its bytes' hash."""
    if not meta or not (meta.get("provenance") or meta.get("hints") or meta.get("preopt")):
        return None
    d = state_dir() / "provenance"
    try:
        d.mkdir(parents=True, exist_ok=True)
        target = d / (hashlib.sha256(data).hexdigest() + ".json")
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps({"v": 1, "t": time.time(),
                                   **{k: meta[k] for k in ("provenance", "hints", "preopt") if k in meta}}))
        os.replace(tmp, target)
        _prune(d)
        return target
    except OSError:
        return None


def read_sidecar(path: Path) -> Optional[Dict[str, Any]]:
    d = state_dir() / "provenance"
    if not d.is_dir():
        return None
    h = sha256_file(path)
    if not h:
        return None
    f = d / (h + ".json")
    try:
        data = json.loads(f.read_text())
    except (OSError, ValueError):
        return None
    return {k: data[k] for k in ("provenance", "hints", "preopt") if k in data} or None


def _prune(d: Path) -> None:
    files = sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime)
    for p in files[:-MAX_SIDECARS]:
        try:
            p.unlink()
        except OSError:
            pass
