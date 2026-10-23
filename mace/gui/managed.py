"""``mace gui --managed``: the server ``mace gui connect`` starts over ssh.

Protocol (stdout, one JSON object per line, whatever the TTY state):

* ``MACE_GUI_READY {...}``   url, port, prefix, token, login, host, fqdn, root,
  version, home (the MACE install folder), reattached, tasks
* ``MACE_GUI_LOGIN {"url"}``  a fresh one-time sign-in link (after ``login``)
* ``MACE_GUI_STOPPED {"reason"}`` the server stopped on purpose
* ``MACE_GUI_ERROR <kind>``  e.g. ``port-busy`` (exit 98)

Control lines on stdin: ``ping`` (keep-alive), ``detach`` (the user left;
keep running work), ``stop`` (stop now), ``login`` (print a new sign-in
link). Unknown lines are ignored.

Watchdog rule (cross-plan contract X1): a *disconnect* is stdin EOF, a
``detach`` line, or no line for MACE_GUI_MANAGED_TIMEOUT seconds (300). With
no task running the server stops at once. With tasks running it *lingers*:
a 0600 state file in the GUI's cache folder lets ``mace gui --managed`` on
the same node and folder reattach (an attach proxy relays a new port to this
server); the server stops when the last task has ended and
MACE_GUI_MANAGED_GRACE seconds (600) passed with nobody attached, or after
24 hours of lingering in any case. A GUI task is never killed because the
browser's connection went away.

Standard library only (the attach proxy runs before the server is imported).
"""
from __future__ import annotations

import http.client
import json
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

DEFAULT_TIMEOUT_S = 300.0
DEFAULT_GRACE_S = 600.0
LINGER_CAP_S = 24 * 3600.0
POLL_S = 0.5


def _env_float(name: str, default: float) -> float:
    try:
        v = float(os.environ.get(name, ""))
        return v if v > 0 else default
    except ValueError:
        return default


def state_dir() -> Path:
    """The GUI's cache folder (same rule as cache.state_dir, without importing it)."""
    v = os.environ.get("MACE_GUI_STATE_DIR")
    if v:
        return Path(v)
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "mace" / "gui"


def short_host() -> str:
    return socket.gethostname().split(".")[0] or "host"


def state_file(port: int) -> Path:
    return state_dir() / f"managed-{short_host()}-{port}.json"


def write_private_json(path: Path, data: Dict[str, Any]) -> None:
    """Atomic 0600 write; the folder made 0700."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        json.dump(data, fh)
    os.replace(tmp, path)


def emit(tag: str, data: Any = None) -> bool:
    """One protocol line on stdout; False when the connection is gone."""
    line = tag if data is None else f"{tag} {json.dumps(data)}"
    try:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()
        return True
    except (OSError, ValueError):
        return False


def log(msg: str) -> None:
    """A line for the node's log (stderr); silent when ssh is gone."""
    try:
        sys.stderr.write(f"mace gui: {msg}\n")
        sys.stderr.flush()
    except (OSError, ValueError):
        pass


def detach_std_streams() -> None:
    """The connection is gone (or going): point fds 1 and 2 at /dev/null, so a
    print, a flush or a new pool worker (multiprocessing flushes both streams
    when it starts one) never fails on the dead ssh pipe."""
    try:
        null = os.open(os.devnull, os.O_WRONLY)
    except OSError:
        return
    for fd, stream in ((1, sys.stdout), (2, sys.stderr)):
        try:
            stream.flush()
        except (OSError, ValueError, AttributeError):
            pass
        try:
            os.dup2(null, fd)
        except OSError:
            pass
    os.close(null)


class Controller:
    """The managed-mode watchdog. ``running`` returns the number of tasks
    running; ``stop(reason)`` stops the server (request_stop)."""

    def __init__(self, running: Callable[[], int], stop: Callable[[str], None],
                 timeout_s: Optional[float] = None, grace_s: Optional[float] = None):
        self.running = running
        self.stop = stop
        self.timeout_s = timeout_s if timeout_s is not None else _env_float("MACE_GUI_MANAGED_TIMEOUT", DEFAULT_TIMEOUT_S)
        self.grace_s = grace_s if grace_s is not None else _env_float("MACE_GUI_MANAGED_GRACE", DEFAULT_GRACE_S)
        self.lock = threading.Lock()
        self.connected = True          # the stdin connection (the ssh session that started us)
        self.last_line = time.monotonic()
        self.watchers = 0              # attach proxies currently attached
        self.watcher_seen: Dict[str, float] = {}
        self.lingering_since: Optional[float] = None
        self.idle_since: Optional[float] = None
        self.info: Dict[str, Any] = {}
        self.on_login: Optional[Callable[[], Optional[str]]] = None
        self.stopped = threading.Event()
        self._file: Optional[Path] = None
        # --http-control (a batch job): no stdin; "connected" is the grace
        # until the first watcher (mace gui --slurm) attaches
        self.http_control = False

    # -- inputs ---------------------------------------------------------------
    def line(self, text: str) -> None:
        cmd = text.strip().lower()
        with self.lock:
            self.last_line = time.monotonic()
        if cmd == "stop":
            self._stop("managed-stop")
        elif cmd == "detach":
            self.disconnect("detach")
        elif cmd == "login" and self.on_login is not None:
            url = self.on_login()
            if url:
                emit("MACE_GUI_LOGIN", {"url": url})

    def disconnect(self, why: str) -> None:
        with self.lock:
            if not self.connected:
                return
            self.connected = False
        n = self.running()
        if n == 0 and self.watchers == 0:
            self._stop("managed-disconnect")
        else:
            self.linger(why, n)

    def attach(self, wid: str) -> Dict[str, Any]:
        with self.lock:
            self.watcher_seen[wid] = time.monotonic()
            self.watchers = len(self.watcher_seen)
            self.idle_since = None
            if self.http_control:
                self.connected = False       # the watchers carry the session from now on
        log(f"reattached ({self.running()} task(s) running)")
        return {"attached": True, "tasks": self.running()}

    def ping(self, wid: str) -> Dict[str, Any]:
        with self.lock:
            known = wid in self.watcher_seen
            if known:
                self.watcher_seen[wid] = time.monotonic()
        if not known and wid:
            # a watcher this process does not know: it attached before an
            # Update Now restart (or was dropped as stale and is alive after
            # all); it pings, so it is attached
            self.attach(wid)
            return {"ok": True, "attached": True}
        return {"ok": True}

    def detach(self, wid: str) -> Dict[str, Any]:
        with self.lock:
            self.watcher_seen.pop(wid, None)
            self.watchers = len(self.watcher_seen)
        if not self.connected and self.watchers == 0:
            n = self.running()
            if n == 0:
                self._stop("managed-disconnect")
            else:
                self.linger("detach", n)
        return {"detached": True}

    def stop_request(self, wid: str = "") -> Dict[str, Any]:
        self._stop("managed-stop")
        return {"stopping": True}

    # -- Update Now and Restart (update_gui.after_stop / cli._serve_managed) ---
    def handoff(self) -> Dict[str, Any]:
        """What the restarted process needs to carry the same session on: the
        attached watchers and whether the first connection is gone."""
        now = time.monotonic()
        with self.lock:
            return {"connected": self.connected, "watchers": sorted(self.watcher_seen),
                    "lingering_for": None if self.lingering_since is None else now - self.lingering_since}

    def restore(self, saved: Any) -> None:
        """The other half of handoff(), before run(). Junk is ignored."""
        if not isinstance(saved, dict):
            return
        now = time.monotonic()
        ws = saved.get("watchers")
        with self.lock:
            if isinstance(ws, list):
                for w in ws:
                    if isinstance(w, str) and w:
                        self.watcher_seen[w] = now
                self.watchers = len(self.watcher_seen)
            self.last_line = now
            if saved.get("connected") is False:
                self.connected = False
            lf = saved.get("lingering_for")
            if isinstance(lf, (int, float)) and lf >= 0:
                self.lingering_since = now - float(lf)
        if self.lingering_since is not None:
            self.write_state_file()          # cli's cleanup removed it before the exec

    # -- lingering -------------------------------------------------------------
    def linger(self, why: str, n: int) -> None:
        with self.lock:
            if self.lingering_since is None:
                self.lingering_since = time.monotonic()
        log(f"connection lost; keeping the server for {n} running task(s)")
        detach_std_streams()
        self.write_state_file()

    def write_state_file(self) -> None:
        """The 0600 file a later mace gui --managed on this node reattaches through."""
        if self.info:
            try:
                self._file = state_file(int(self.info["port"]))
                write_private_json(self._file, {**self.info, "pid": os.getpid(), "lingering": time.time()})
            except (OSError, KeyError, ValueError):
                pass

    def remove_state_file(self) -> None:
        f = self._file or (state_file(int(self.info["port"])) if self.info.get("port") else None)
        if f is not None:
            try:
                f.unlink()
            except OSError:
                pass

    def _stop(self, reason: str) -> None:
        if self.stopped.is_set():
            return
        self.stopped.set()
        self.stop(reason)

    def tick(self) -> None:
        """One watchdog step (every POLL_S seconds)."""
        if self.stopped.is_set():
            return
        now = time.monotonic()
        with self.lock:
            # a watcher that stopped pinging (its ssh died) counts as detached
            stale = [w for w, t in self.watcher_seen.items() if now - t > self.timeout_s]
            for w in stale:
                self.watcher_seen.pop(w, None)
            self.watchers = len(self.watcher_seen)
            silent = self.connected and now - self.last_line > self.timeout_s
        if silent:
            self.disconnect("timeout")
            return
        if stale and self.http_control and not self.connected and self.watchers == 0:
            # the last watcher stopped pinging (its connection died): a disconnect
            n = self.running()
            if n == 0:
                self._stop("managed-disconnect")
            else:
                self.linger("timeout", n)
            return
        if self.connected or self.watchers:
            return
        n = self.running()
        if n:
            self.idle_since = None
        elif self.idle_since is None:
            self.idle_since = now
        if self.lingering_since is not None and now - self.lingering_since > LINGER_CAP_S:
            log("stopping: lingered 24 hours")
            self._stop("managed-cap")
        elif n == 0 and self.idle_since is not None and now - self.idle_since >= self.grace_s:
            log("the last task ended and nobody reattached; stopping")
            self._stop("managed-done")

    def run(self, stdin=None) -> None:
        """Start the stdin reader (not with ``stdin=False``) and the watchdog
        (daemon threads)."""
        src = stdin if stdin is not None else sys.stdin

        def reader():
            try:
                for raw in src:
                    self.line(raw)
                    if self.stopped.is_set():
                        return
            except (OSError, ValueError):
                pass
            self.disconnect("eof")

        def watch():
            while not self.stopped.wait(POLL_S):
                try:
                    self.tick()
                except Exception:          # the watchdog never dies
                    pass
        if stdin is not False:
            threading.Thread(target=reader, name="mace-managed-stdin", daemon=True).start()
        threading.Thread(target=watch, name="mace-managed-watchdog", daemon=True).start()


# ---------------------------------------------------------------- reattach

def _pid_alive(pid: int) -> bool:
    if os.name == "nt":
        from .registry import pid_alive      # os.kill(pid, 0) would end the process on Windows
        return pid_alive(pid)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return False
    except OSError:
        return False


def _api(port: int, prefix: str, token: str, route: str, body: Optional[Dict[str, Any]] = None,
         timeout: float = 5.0) -> Optional[Dict[str, Any]]:
    url = f"http://127.0.0.1:{port}{prefix}api/{route}"
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="GET" if body is None else "POST",
                                 headers={"X-Mace-Token": token, "X-Requested-With": "mace-gui",
                                          "Content-Type": "application/json", "Host": f"127.0.0.1:{port}"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=timeout) as resp:
            return json.loads(resp.read() or b"{}")
    except (OSError, ValueError, urllib.error.URLError, http.client.HTTPException):
        # HTTPException: a reply the server cut off as it stopped (managed/stop
        # shuts it down from another thread; IncompleteRead is no OSError)
        return None


def find_lingering(root: Path) -> Optional[Dict[str, Any]]:
    """A lingering managed server on this node for this folder, whose token answers."""
    d = state_dir()
    try:
        files = sorted(d.glob(f"managed-{short_host()}-*.json"))
    except OSError:
        return None
    want = os.path.realpath(str(root))
    for f in files:
        try:
            info = json.loads(f.read_text())
            if not isinstance(info, dict):
                continue
            pid, port = int(info["pid"]), int(info["port"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if not _pid_alive(pid) or os.path.realpath(str(info.get("root", ""))) != want:
            continue
        if _api(port, info["prefix"], info["token"], "info") is None:
            continue
        return info
    return None


def _pipe(a: socket.socket, b: socket.socket) -> None:
    try:
        while True:
            data = a.recv(65536)
            if not data:
                break
            b.sendall(data)
    except OSError:
        pass
    finally:
        for s in (a, b):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


def attach_proxy(info: Dict[str, Any], port: int, ready_extra: Dict[str, Any], stdin=None) -> int:
    """Relay 127.0.0.1:port -> the lingering server, report READY with its
    prefix and token, forward control lines, detach on EOF."""
    target = int(info["port"])
    lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        lsock.bind(("127.0.0.1", port))
    except OSError:
        emit("MACE_GUI_ERROR port-busy")
        return 98
    lsock.listen(32)
    port = lsock.getsockname()[1]
    wid = f"w{os.getpid()}"
    got = _api(target, info["prefix"], info["token"], "managed/attach", {"watcher": wid}) or {}

    def serve():
        while True:
            try:
                c, _ = lsock.accept()
            except OSError:
                return
            try:
                u = socket.create_connection(("127.0.0.1", target), timeout=10)
                u.settimeout(None)
            except OSError:
                c.close()
                continue
            threading.Thread(target=_pipe, args=(c, u), daemon=True).start()
            threading.Thread(target=_pipe, args=(u, c), daemon=True).start()
    threading.Thread(target=serve, name="mace-attach-proxy", daemon=True).start()
    login = None
    lc = _api(target, info["prefix"], info["token"], "login-code", {})
    if lc and lc.get("code"):
        login = f"http://127.0.0.1:{port}{info['prefix']}?login={lc['code']}"
    emit("MACE_GUI_READY", {**ready_extra, "url": f"http://127.0.0.1:{port}{info['prefix']}", "port": port,
                            "prefix": info["prefix"], "token": info["token"], "login": login,
                            "root": info.get("root"), "reattached": True, "tasks": int(got.get("tasks") or 0)})
    stop_ev = threading.Event()

    def pinger():
        while not stop_ev.wait(30):
            if _api(target, info["prefix"], info["token"], "managed/ping", {"watcher": wid}) is None \
                    and not _pid_alive(int(info["pid"])):
                emit("MACE_GUI_STOPPED", {"reason": "server-gone"})
                os._exit(0)
    threading.Thread(target=pinger, daemon=True).start()
    src = stdin if stdin is not None else sys.stdin
    try:
        for raw in src:
            cmd = raw.strip().lower()
            if cmd in ("ping", "detach", "stop"):
                _api(target, info["prefix"], info["token"], f"managed/{cmd}", {"watcher": wid})
                if cmd in ("detach", "stop"):
                    break
            elif cmd == "login":
                lc = _api(target, info["prefix"], info["token"], "login-code", {})
                if lc and lc.get("code"):
                    emit("MACE_GUI_LOGIN", {"url": f"http://127.0.0.1:{port}{info['prefix']}?login={lc['code']}"})
        else:
            _api(target, info["prefix"], info["token"], "managed/detach", {"watcher": wid})
    except (OSError, ValueError):
        _api(target, info["prefix"], info["token"], "managed/detach", {"watcher": wid})
    stop_ev.set()
    try:
        lsock.close()
    except OSError:
        pass
    return 0
