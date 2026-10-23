"""MACE GUI's update notice (1.1.6): the GUI only drives ``mace update``.

* **Check.** ``mace update --check --json`` for the channel of Settings >
  Server (``update.channel``: release, preview or latest; release by
  default), 20 s after the server starts and then every
  ``update.intervalHours`` (6 by default); ``update.check`` false turns it
  off (Help > Check for Updates still works). git never prompts (no
  terminal, ssh in batch mode) and the check is killed after 90 s; offline
  it fails quietly (the error is kept for the dialog).
* **Notice.** An available update shows a status-bar item; its dialog shows
  the report (current and new version, the CHANGELOG sections in between,
  the refusal ``mace update`` would print, local changes with the git stash
  line, queued jobs, the requirements change with its pip command) and the
  running GUI tasks.
* **Update Now and Restart** (``choice: now``): refused while GUI tasks run
  (``wait: true`` waits for them); runs ``mace update --yes --no-pip`` (or
  ``--pip`` when the user ticked it), checks that the new code starts
  (``SMOKE``: every module compiles and every mace.gui module imports),
  saves the result, and stops the server with the reason
  ``update-restart``: cli.main's cleanup then execs the same command line
  in the same process (``after_stop``): the same port, token
  and prefix, and the signed-in sessions, which pass through a 0600 hand-off
  file. ``--managed`` (mace gui connect, the connect scripts) and ``--slurm``
  (``--http-control``) keep their stdin, stdout, ready file and tunnel; the
  new process sends no second READY line and the old one no STOPPED line.
  The managed controller's watchers and connection state pass through the
  same hand-off, so a --slurm client that only pings and a lingering server
  kept by reattached clients carry on (Controller.handoff/restore).
* **Update When I Sign Out** (``choice: signout``): after Sign Out and Stop
  Server, once the server has stopped, a detached helper waits for this
  process to end and runs the update; the result shows at the next start.
  Inside a SLURM job the job would end the helper, so the update runs there
  before the process exits (the server is stopped already).
* **Not Now** (this session) and **Skip This Version** (kept in
  ``<state dir>/update.json``).
* **Refusals.** A server on a network address (``--insecure-network``): no
  update from the browser, anyone with the token could replace the code. A
  read-only or someone else's install: ask its maintainer. Local changes,
  local commits: ``mace update``'s own text.

Tests point it at another checkout and another update command
(``MACE_GUI_TEST_UPDATE_HOME``, ``MACE_GUI_TEST_UPDATE_ARGV``) only with
the test hooks on; ``MACE_GUI_UPDATE_CHECK=0`` turns the background check
off (the test suite does), ``MACE_GUI_UPDATE_DELAY`` sets the first delay.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import settings as gui_settings

CHANNELS = ("release", "preview", "latest")
CHECK_TIMEOUT_S = 90.0
UPDATE_TIMEOUT_S = 900.0
SMOKE_TIMEOUT_S = 120.0
FIRST_DELAY_S = 20.0
DEFAULT_HOURS = 6
OUTPUT_MAX = 20000
RESTART_ENV = "MACE_GUI_RESTART_FILE"
NETWORK_REFUSAL = ("This server listens on a network address (--insecure-network), where anyone with the token "
                   "could replace MACE's code, so it does not update from the browser. Run mace update in a "
                   "terminal on {node}, then restart mace gui.")
MAINTAINER = ("This MACE install is not yours to change ({why}). Ask its maintainer to update it, or use a clone of "
              "your own.")

# the detached helper of Update When I Sign Out: standard library only, and
# nothing from the checkout it updates (its files change under it)
HELPER = r"""
import json, os, subprocess, sys, time


def alive(pid):
    # os.kill(pid, 0) would TERMINATE the process on Windows: ask the OS there (as registry.pid_alive)
    if os.name == "nt":
        import ctypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        h = k32.OpenProcess(0x1000, False, pid)
        if not h:
            return ctypes.get_last_error() == 5    # ERROR_ACCESS_DENIED: it exists
        try:
            code = ctypes.c_ulong()
            return bool(k32.GetExitCodeProcess(h, ctypes.byref(code))) and code.value == 259
        finally:
            k32.CloseHandle(h)
    os.kill(pid, 0)
    return True


cfg = json.loads(sys.argv[1])
pid, end = cfg["wait_pid"], time.time() + 120
while pid and time.time() < end:
    try:
        st = open("/proc/%d/status" % pid).read() if os.path.isdir("/proc") else ""
        if "State:\tZ" in st:
            break
        if not alive(pid):
            break
    except (OSError, ValueError):
        break
    time.sleep(0.2)
env = dict(os.environ, **cfg["env"])
try:
    r = subprocess.run(cfg["argv"], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       universal_newlines=True, timeout=cfg["timeout"], env=env)
    code, out = r.returncode, r.stdout or ""
except Exception as exc:
    code, out = None, "the update did not run: %s" % exc
res = dict(cfg["result"], ok=code == 0, code=code, output=out[-cfg["max"]:], time=time.time())
path = cfg["state"]
try:
    data = json.load(open(path))
except Exception:
    data = {}
data["result"] = res
os.makedirs(os.path.dirname(path), exist_ok=True)
tmp = path + ".%d.tmp" % os.getpid()
fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
with os.fdopen(fd, "w") as fh:
    json.dump(data, fh)
os.replace(tmp, path)
"""

# the check that the updated code starts, before the in-place exec: every
# .py under mace/ compiles, and every mace.gui module imports (what the
# restarted server loads at start and later). A third-party module that is
# missing counts only when requirements.txt lists it (a new requirement not
# installed); optional ones (seekpath, ...) may be missing as before
SMOKE = r"""
import importlib, os, pkgutil, re, sys
home = sys.argv[1]
bad = []
for root, dirs, files in os.walk(os.path.join(home, "mace")):
    dirs[:] = [d for d in dirs if d != "__pycache__"]
    for f in files:
        if f.endswith(".py"):
            p = os.path.join(root, f)
            try:
                with open(p, "rb") as fh:
                    compile(fh.read(), p, "exec", dont_inherit=True)
            except (SyntaxError, ValueError) as exc:
                bad.append("%s: %s" % (os.path.relpath(p, home), exc))
need = set()
try:
    for ln in open(os.path.join(home, "requirements.txt")):
        m = re.match(r"\s*([A-Za-z0-9_.-]+)", ln.split("#")[0])
        if m:
            need.add(m.group(1).lower().replace("-", "_"))
except OSError:
    pass
alias = {"pyyaml": "yaml", "scikit_learn": "sklearn", "pillow": "PIL"}
need = {alias.get(n, n) for n in need}
if not bad:
    sys.path.insert(0, home)
    import mace.gui
    if os.path.realpath(os.path.dirname(mace.gui.__file__)) != os.path.realpath(os.path.join(home, "mace", "gui")):
        bad.append("imported mace.gui from %s, not from %s" % (mace.gui.__file__, home))
    for m in ["cli", "server"] + sorted(i.name for i in pkgutil.iter_modules(mace.gui.__path__)):
        try:
            importlib.import_module("mace.gui." + m)
        except ModuleNotFoundError as exc:
            top = (exc.name or "").split(".")[0]
            if top == "mace" or top in need:
                bad.append("mace.gui.%s: %s" % (m, exc))
        except BaseException as exc:
            bad.append("mace.gui.%s: %s: %s" % (m, type(exc).__name__, exc))
for b in bad:
    print(b)
sys.exit(1 if bad else 0)
"""


def smoke_home(st) -> Path:
    """The checkout the restart execs (tests may point it elsewhere)."""
    if getattr(st, "test_hooks", False) and os.environ.get("MACE_GUI_TEST_SMOKE_HOME"):
        return Path(os.environ["MACE_GUI_TEST_SMOKE_HOME"])
    return install_home()


def smoke(home: Path) -> Tuple[Optional[int], str]:
    """(0, "") when the code in ``home`` would start the GUI server; else the
    exit code (None after a timeout) and the errors."""
    argv = [sys.executable, "-c", SMOKE, str(home)]
    env = dict(_env(), PYTHONPATH=str(home), PYTHONDONTWRITEBYTECODE="1")
    try:
        r = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           universal_newlines=True, env=env, cwd=str(home), timeout=SMOKE_TIMEOUT_S)
        return r.returncode, r.stdout or ""
    except subprocess.TimeoutExpired:
        return None, f"(the check did not finish in {SMOKE_TIMEOUT_S:g} s)"
    except OSError as exc:
        return None, str(exc)


# ---------------------------------------------------------------- places
def install_home() -> Path:
    """The checkout this server runs from (where mace_cli is)."""
    return Path(__file__).resolve().parents[2]


def _cli() -> str:
    return str(install_home() / "mace_cli")


def _state_path() -> Path:
    from .managed import state_dir
    return state_dir() / "update.json"


def _read_state() -> Dict[str, Any]:
    try:
        d = json.loads(_state_path().read_text())
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_state(**changes: Any) -> None:
    from .managed import write_private_json
    d = _read_state()
    d.update(changes)
    try:
        write_private_json(_state_path(), d)
    except OSError:
        pass


def config() -> Dict[str, Any]:
    s = gui_settings.load()
    ch = s.get("update.channel") if s.get("update.channel") in CHANNELS else "release"
    hours = s.get("update.intervalHours") if isinstance(s.get("update.intervalHours"), int) else DEFAULT_HOURS
    on = s.get("update.check", True) is not False and os.environ.get("MACE_GUI_UPDATE_CHECK", "1") != "0"
    return {"enabled": on, "channel": ch, "hours": hours}


def _env() -> Dict[str, str]:
    """git never waits for a password or a host-key answer (the check is unattended)."""
    e = dict(os.environ, GIT_TERMINAL_PROMPT="0", MACE_PYTHON=os.environ.get("MACE_PYTHON") or sys.executable)
    e.setdefault("GIT_SSH_COMMAND", "ssh -o BatchMode=yes -o ConnectTimeout=15")
    return e


def _run(argv: List[str], timeout: float) -> Tuple[Optional[int], str]:
    """(exit code or None after a timeout, output); its own session, killed whole."""
    try:
        p = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             universal_newlines=True, env=_env(), start_new_session=True)
    except OSError as exc:
        return None, str(exc)
    try:
        out, _ = p.communicate(timeout=timeout)
        return p.returncode, out or ""
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except OSError:
            p.kill()
        out, _ = p.communicate()
        return None, (out or "") + f"\n(stopped after {timeout:g} s)"


def is_network(st) -> bool:
    from .server import is_loopback
    return not is_loopback(getattr(st, "bind_host", "127.0.0.1") or "127.0.0.1")


def restart_mode(st) -> Dict[str, Any]:
    m = getattr(st, "managed", None)
    if getattr(st, "request_stop", None) is None or os.name == "nt":
        # (Windows has no exec: os.execv there starts another process and ends this one)
        return {"mode": "none", "in_place": False,
                "text": "This server cannot restart itself: after the update, stop it and start mace gui again."}
    if m is not None and getattr(m, "http_control", False):
        return {"mode": "slurm", "in_place": True,
                "text": "MACE GUI restarts inside its SLURM job on the same port; the tunnel and this page reconnect."}
    if m is not None:
        return {"mode": "managed", "in_place": True,
                "text": "MACE GUI restarts on the same port; mace gui connect and this page reconnect. If the page "
                        "does not come back, run mace gui connect again."}
    return {"mode": "local", "in_place": True,
            "text": "MACE GUI restarts in the same terminal on the same port; this page reconnects."}


# ---------------------------------------------------------------- the updater
class Updater:
    def __init__(self, st):
        self.st = st
        self.lock = threading.Lock()
        self.started = time.time()
        self.report: Optional[Dict[str, Any]] = None
        self.checked_at: Optional[float] = None
        self.error: Optional[str] = None
        self.checking = False
        self.dismissed = False
        self.deferred: Optional[Dict[str, Any]] = None
        self.phase: Optional[str] = None          # waiting | updating | restarting | failed | done
        self.result: Optional[Dict[str, Any]] = None
        saved = _read_state()
        if saved.get("result"):
            self.result = saved["result"]
            _write_state(result=None)
        self.skipped = saved.get("skipped") if isinstance(saved.get("skipped"), dict) else None
        self.restarted = False
        self._wake = threading.Event()

    # -- where and how
    def home(self) -> Path:
        if self.st.test_hooks and os.environ.get("MACE_GUI_TEST_UPDATE_HOME"):
            return Path(os.environ["MACE_GUI_TEST_UPDATE_HOME"])
        return install_home()

    def update_argv(self, channel: str, pip: bool) -> List[str]:
        tail = ["--channel", channel, "--yes", "--pip" if pip else "--no-pip", "--home", str(self.home())]
        if self.st.test_hooks and os.environ.get("MACE_GUI_TEST_UPDATE_ARGV"):
            return list(json.loads(os.environ["MACE_GUI_TEST_UPDATE_ARGV"])) + tail
        return [sys.executable, _cli(), "--no-banner", "update"] + tail

    # -- the check
    def start(self) -> None:
        threading.Thread(target=self._loop, name="mace-update-check", daemon=True).start()

    def _loop(self) -> None:
        try:
            first = float(os.environ.get("MACE_GUI_UPDATE_DELAY", FIRST_DELAY_S))
        except ValueError:
            first = FIRST_DELAY_S
        wait = first
        while True:
            self._wake.wait(wait)
            self._wake.clear()
            if self.st.stopping.is_set():
                return
            cfg = config()
            if cfg["enabled"]:
                self.check()
            wait = cfg["hours"] * 3600.0

    def check(self, channel: Optional[str] = None) -> Dict[str, Any]:
        ch = channel if channel in CHANNELS else config()["channel"]
        with self.lock:
            if self.checking:
                busy = True
            else:
                busy, self.checking = False, True
        if busy:
            return self.status()
        try:
            code, out = _run([sys.executable, _cli(), "--no-banner", "update", "--check", "--json", "--channel", ch,
                              "--home", str(self.home())], CHECK_TIMEOUT_S)
            line = next((ln for ln in reversed(out.splitlines()) if ln.startswith("{")), "")
            try:
                rep = json.loads(line) if code == 0 else None
            except ValueError:
                rep = None
            with self.lock:
                self.checked_at = time.time()
                if rep is None:
                    self.error = "Could not check for updates" + (" (timed out)" if code is None else "") + \
                        ": " + (out.strip().splitlines() or ["no answer"])[-1][:300]
                elif rep.get("refusal_kind") in ("fetch", "no_origin", "no_target", "not_git"):
                    self.report, self.error = rep, "Could not check for updates: " + str(rep.get("refusal"))[:600]
                else:
                    self.report, self.error = rep, None
        finally:
            with self.lock:
                self.checking = False
        s = self.status()
        self.st.tasks.broadcast("update", {"notice": s["notice"], "new": (s["report"] or {}).get("target", {}),
                                           "current": (s["report"] or {}).get("current", {})})
        return s

    # -- what the dialog shows
    def notice(self) -> bool:
        r = self.report
        if not r or not r.get("available") or self.dismissed or self.phase in ("updating", "restarting"):
            return False
        sk = self.skipped or {}
        t = r.get("target") or {}
        return not (sk.get("ref") == t.get("ref") and sk.get("commit") == t.get("commit"))

    def blocked(self) -> Optional[str]:
        if is_network(self.st):
            return NETWORK_REFUSAL.format(node=getattr(self.st, "node", "") or "the server's computer")
        return None

    def maintainer(self) -> Optional[str]:
        r = self.report or {}
        if r.get("refusal_kind") != "not_writable":
            return None
        why = "it belongs to another user" if r.get("owner_is_you") is False else "its .git folder is read-only"
        return MAINTAINER.format(why=why)

    def status(self) -> Dict[str, Any]:
        cfg = config()
        with self.lock:
            return {"enabled": cfg["enabled"], "channel": cfg["channel"], "interval_hours": cfg["hours"],
                    "checking": self.checking, "checked_at": self.checked_at, "error": self.error,
                    "report": self.report, "notice": self.notice(), "skipped": self.skipped,
                    "deferred": self.deferred, "phase": self.phase, "result": self.result,
                    "blocked": self.blocked(), "maintainer": self.maintainer(),
                    "tasks": [t.get("title") for t in self.st.tasks.running()],
                    "restart": restart_mode(self.st), "started": self.started, "home": str(self.home())}

    # -- the choices
    def choose(self, body: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        c = str(body.get("choice") or "")
        if c == "later":
            self.dismissed = True
            return 200, self.status()
        if c == "skip":
            t = (self.report or {}).get("target") or {}
            if not t.get("ref"):
                return 409, {"error": "nothing to skip: no update was found"}
            self.skipped = {"ref": t.get("ref"), "commit": t.get("commit"), "version": t.get("version")}
            _write_state(skipped=self.skipped)
            return 200, self.status()
        if c == "ack":
            self.result = None
            return 200, self.status()
        if c == "cancel-signout":
            self.deferred = None
            self.st.tasks.broadcast("update", {"notice": self.notice(), "deferred": None})
            return 200, self.status()
        if c not in ("now", "signout"):
            return 400, {"error": "choice is now, signout, later, skip, cancel-signout or ack"}
        why = self.blocked()
        if why:
            return 403, {"error": why}
        r = self.report
        if not r or not r.get("available"):
            return 409, {"error": "no update is available (Help > Check for Updates looks again)"}
        if r.get("refusal"):
            return 409, {"error": r["refusal"], "kind": r.get("refusal_kind"), "maintainer": self.maintainer()}
        if self.phase in ("waiting", "updating", "restarting"):
            return 409, {"error": "an update is in progress"}
        pip = bool(body.get("pip"))
        target = r.get("target") or {}
        if c == "signout":
            self.deferred = {"channel": r.get("channel"), "ref": target.get("ref"), "version": target.get("version"),
                             "commit": target.get("commit"), "pip": pip, "from": (r.get("current") or {}).get("version")}
            self.st.tasks.broadcast("update", {"notice": self.notice(), "deferred": self.deferred})
            return 200, self.status()
        running = self.st.tasks.running()
        if running and not body.get("wait"):
            return 409, {"error": f"{len(running)} task(s) are running; MACE GUI restarts after the update, which "
                                  "would cancel them. Wait for them, or update when they finish.",
                         "tasks": [t.get("title") for t in running]}
        # the test and the claim together: two tabs choosing "now" at once start one update, not two
        with self.lock:
            taken = self.phase in ("waiting", "updating", "restarting")
            if not taken:
                self.phase = "waiting" if running else "updating"
        if taken:
            return 409, {"error": "an update is in progress"}
        threading.Thread(target=self._now, args=(r, pip), name="mace-update-now", daemon=True).start()
        return 202, self.status()

    def _progress(self, phase: str, **extra: Any) -> None:
        self.phase = phase
        self.st.tasks.broadcast("update-progress", {"phase": phase, **extra})

    def _now(self, r: Dict[str, Any], pip: bool) -> None:
        while self.st.tasks.running_count():
            if self.st.stopping.is_set():
                self.phase = None
                return
            time.sleep(1.0)
        self._progress("updating")
        cur = (r.get("current") or {}).get("version")
        new = (r.get("target") or {}).get("version")
        code, out = _run(self.update_argv(str(r.get("channel") or "release"), pip), UPDATE_TIMEOUT_S)
        res = {"ok": code == 0, "code": code, "output": out[-OUTPUT_MAX:], "from": cur, "to": new, "how": "now",
               "time": time.time(), "restart": None}
        if code != 0:
            self.result = res
            self._progress("failed", result=res)
            return
        code, sout = smoke(smoke_home(self.st))
        mode = restart_mode(self.st)
        if code != 0 or not mode["in_place"]:
            res["restart"] = "manual"
            res["restart_why"] = (mode["text"] if not mode["in_place"] else
                                  "The updated MACE GUI would not start (its modules failed to load), so this server "
                                  "keeps running the old code. Fix the error below (often: install the new "
                                  "requirements), then stop and start mace gui:\n" + sout.strip()[-2000:])
            self.result = res
            self._progress("done", result=res)
            return
        res["restart"] = "in-place"
        _write_state(result=res)
        self.st.update_restart = True
        self._progress("restarting", result=res)
        time.sleep(0.5)               # the event reaches the open tabs first
        self.st.request_stop("update-restart")


# ---------------------------------------------------------------- server hooks
def install(st) -> Updater:
    """make_server: the updater, its background check, and a restart's hand-off."""
    u = Updater(st)
    st.updater = u
    if getattr(st, "_update_handoff", None):
        u.restarted = True
    u.start()
    return u


def route(handler, method: str, route: str, body: Optional[Dict[str, Any]] = None):
    st = handler.state
    u = getattr(st, "updater", None) or install(st)
    if method == "GET" and route == "update":
        return handler._json(u.status())
    if method == "POST" and route == "update/check":
        return handler._json(u.check((body or {}).get("channel")))
    if method == "POST" and route == "update/choose":
        code, data = u.choose(body or {})
        return handler._json(data, code)
    return handler._error(404, "no such update route")


# ---------------------------------------------------------------- restart and sign-out
def take_handoff() -> Optional[Dict[str, Any]]:
    """In a restarted server: the token and sessions the old one left (then gone)."""
    p = os.environ.pop(RESTART_ENV, "")
    if not p:
        return None
    try:
        path = Path(p)
        if path.stat().st_uid != os.getuid() or path.stat().st_mode & 0o077:
            return None
        d = json.loads(path.read_text())
        path.unlink()
        return d if isinstance(d, dict) and d.get("token") else None
    except (OSError, ValueError, AttributeError):
        return None


def resume(st, handoff: Optional[Dict[str, Any]]) -> None:
    """The sessions signed in before the restart stay signed in."""
    if not handoff:
        return
    st._update_handoff = True
    st._managed_handoff = handoff.get("managed")      # cli._serve_managed restores it
    if getattr(st, "updater", None) is not None:
        st.updater.restarted = True
    table = handoff.get("sessions") or {}
    with st.web.lock:
        for sid, rec in table.items():
            if isinstance(rec, dict):
                st.web.table[str(sid)] = {"created": float(rec.get("created", time.time())),
                                          "last_input": float(rec.get("last_input", time.time()))}


def restart_argv(argv: List[str], port: int) -> List[str]:
    """The same `mace gui` command line, on the port it got (--port 0 picked
    one), without --open (the browser is open)."""
    out: List[str] = []
    skip = False
    for a in argv:
        if skip:
            skip = False
            continue
        if a == "--port":
            skip = True
            continue
        if a.startswith("--port=") or a == "--open":
            continue
        out.append(a)
    return [sys.executable, _cli(), "--no-banner", "gui"] + out + ["--port", str(port)]


def after_stop(state, argv: List[str], port: int, stops: List[Any]) -> None:
    """cli.main's last step: exec the updated server (Update Now), or start
    the update of Update When I Sign Out after a browser stop."""
    reason = getattr(state, "stop_reason", None)
    u = getattr(state, "updater", None)
    if reason == "update-restart":
        from .managed import write_private_json
        hand = _state_path().with_name(f"restart-{os.getpid()}.json")
        with state.web.lock:
            sessions = {k: dict(v) for k, v in state.web.table.items()}
        m = getattr(state, "managed", None)
        data: Dict[str, Any] = {"token": state.token, "sessions": sessions}
        if m is not None:
            data["managed"] = m.handoff()    # the watchers, and whether the first connection is gone
        try:
            write_private_json(hand, data)
            os.environ[RESTART_ENV] = str(hand)
        except OSError:
            pass
        os.environ["MACE_GUI_RESTARTED"] = "1"
        for sig in stops:                    # what the new process installs its handlers over
            signal.signal(sig, signal.SIG_DFL)
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except (OSError, ValueError):
            pass
        args = restart_argv(argv, port)
        try:
            os.execv(args[0], args)
        except OSError as exc:
            print(f"mace gui: MACE was updated, but the server could not restart ({exc}); start mace gui again.",
                  file=sys.stderr, flush=True)
            return
    if reason == "browser" and u is not None and u.deferred:
        d = u.deferred
        in_job = bool(os.environ.get("SLURM_JOB_ID")) and getattr(getattr(state, "managed", None), "http_control",
                                                                  False)
        cfg = {"wait_pid": 0 if in_job else os.getpid(), "argv": u.update_argv(d["channel"] or "release", d["pip"]),
               "env": {k: v for k, v in _env().items() if k in ("GIT_TERMINAL_PROMPT", "GIT_SSH_COMMAND",
                                                                  "MACE_PYTHON")},
               "timeout": UPDATE_TIMEOUT_S, "max": OUTPUT_MAX, "state": str(_state_path()),
               "result": {"from": d.get("from"), "to": d.get("version"), "how": "signout", "restart": None}}
        argv2 = [sys.executable, "-c", HELPER, json.dumps(cfg)]
        try:
            if in_job:               # the job would end a detached helper with it
                subprocess.run(argv2, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=UPDATE_TIMEOUT_S + 30)
            else:
                subprocess.Popen(argv2, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL, start_new_session=True, close_fds=True)
            print(f"mace gui: updating MACE to {d.get('version')} now that the server has stopped; the result "
                  "shows at the next start.", file=sys.stderr, flush=True)
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
