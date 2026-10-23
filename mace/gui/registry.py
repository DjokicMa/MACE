"""The registry of your running MACE GUI sessions.

Every server records itself (host, port, folder, process id, start time; never
the token) in ``<cache>/mace/gui/sessions/<host>-<port>.json`` (0600 in a 0700
folder, and the cache is your home, shared by every node). A server touches
its record every minute while it runs and removes it when it stops.

* On the same host a record counts when its process is alive.
* On another node a record counts when it was touched in the last few
  minutes (the process cannot be asked from here).

``mace gui`` reads it at start-up to tell you that a session of yours already
serves the folder (instead of starting a second one), and ``mace gui
--sessions`` lists them with the line that forwards each port. Standard
library only.
"""
from __future__ import annotations

import json
import os
import socket
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

FRESH_S = 240.0                 # another node's record: touched this recently = running
BEAT_S = 60.0


def pid_alive(pid: int) -> bool:
    """Whether a process exists. os.kill(pid, 0) would TERMINATE it on Windows, so ask the OS instead."""
    if not isinstance(pid, int) or pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            h = k32.OpenProcess(0x1000, False, pid)         # PROCESS_QUERY_LIMITED_INFORMATION
            if not h:
                return ctypes.get_last_error() == 5         # ERROR_ACCESS_DENIED: it exists; another user's
            try:
                code = ctypes.c_ulong()
                return bool(k32.GetExitCodeProcess(h, ctypes.byref(code))) and code.value == 259    # STILL_ACTIVE
            finally:
                k32.CloseHandle(h)
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True                                         # it exists; another user's
    except OSError:
        return False


def short_host() -> str:
    return socket.gethostname().split(".")[0] or "host"


def registry_dir() -> Path:
    from .managed import state_dir
    return state_dir() / "sessions"


def _private_dir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(d, 0o700)
    except OSError:
        pass


def record_path(port: int, host: Optional[str] = None) -> Path:
    return registry_dir() / f"{host or short_host()}-{port}.json"


class Registration:
    """This server's record: written now, touched every minute, removed by close()."""

    def __init__(self, port: int, root: str, managed: bool = False, version: str = "", beat_s: float = BEAT_S,
                 directory: Optional[Path] = None, root_fn=None):
        self.root_fn = root_fn          # File > Open Folder changes the folder served: the record follows
        self.path = (directory or registry_dir()) / f"{short_host()}-{port}.json"
        self.beat_s = beat_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        import getpass
        self.data = {"host": short_host(), "port": int(port), "pid": os.getpid(), "root": str(root),
                     "user": getpass.getuser(), "managed": bool(managed), "version": version,
                     "started": time.time()}

    def _write(self) -> bool:
        try:
            _private_dir(self.path.parent)
            tmp = self.path.with_name(self.path.name + f".{os.getpid()}.tmp")
            fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as fh:
                json.dump(self.data, fh)
            os.replace(tmp, self.path)
            return True
        except OSError:
            return False                   # a read-only home: the registry is a convenience

    def start(self) -> "Registration":
        if not self._write():
            return self
        self._thread = threading.Thread(target=self._beat, name="mace-gui-registry", daemon=True)
        self._thread.start()
        return self

    def _beat(self) -> None:
        while not self._stop.wait(self.beat_s):
            try:
                cur = str(self.root_fn()) if self.root_fn else self.data["root"]
            except Exception:
                cur = self.data["root"]
            if cur != self.data["root"]:
                self.data["root"] = cur
                if not self._write():
                    return
                continue
            try:
                os.utime(self.path, None)
            except OSError:
                if not self._write():
                    return

    def close(self) -> None:
        self._stop.set()
        try:
            self.path.unlink()
        except OSError:
            pass


def read_all(directory: Optional[Path] = None) -> List[Dict[str, Any]]:
    out = []
    d = directory or registry_dir()
    try:
        files = sorted(d.glob("*.json"))
    except OSError:
        return out
    for f in files:
        try:
            data = json.loads(f.read_text())
            if not isinstance(data, dict) or not isinstance(data.get("port"), int):
                continue
            data["_file"] = str(f)
            data["_mtime"] = f.stat().st_mtime
        except (OSError, ValueError):
            continue
        out.append(data)
    return out


def live(directory: Optional[Path] = None, host: Optional[str] = None, now: Optional[float] = None,
         alive=pid_alive, port_open=None, prune: bool = True) -> List[Dict[str, Any]]:
    """Your running sessions. Records of dead processes on this host are removed."""
    here = host or short_host()
    now = time.time() if now is None else now
    res = []
    for d in read_all(directory):
        d["here"] = d.get("host") == here
        if d["here"]:
            ok = alive(int(d.get("pid") or 0))
            if ok and port_open is not None:
                ok = port_open(int(d["port"]))
            if not ok:
                if prune:
                    try:
                        os.unlink(d["_file"])
                    except OSError:
                        pass
                continue
        elif now - d["_mtime"] > FRESH_S:
            continue
        res.append(d)
    return res


def serving(root: str, sessions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The session on this host that already serves `root`."""
    want = os.path.realpath(str(root))
    for d in sessions:
        if d.get("here") and os.path.realpath(str(d.get("root", ""))) == want:
            return d
    return None


def age(seconds: float) -> str:
    s = int(max(0, seconds))
    if s < 90:
        return f"{s} s"
    if s < 5400:
        return f"{s // 60} min"
    if s < 172800:
        return f"{s // 3600} h"
    return f"{s // 86400} days"


def describe(d: Dict[str, Any], now: Optional[float] = None) -> str:
    now = time.time() if now is None else now
    kind = "started by mace gui connect" if d.get("managed") else "started from a terminal"
    since = f", running {age(now - d['started'])}" if d.get("started") else ""
    return (f"port {d['port']} on {d.get('host', '?')}: {d.get('root', '?')} ({kind}{since}"
            + (f", process {d['pid']}" if d.get("here") and d.get("pid") else "") + ")")
