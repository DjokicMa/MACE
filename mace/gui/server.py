"""``mace gui`` HTTP server: stdlib only, loopback, token-protected.

Security model (single user, possibly on a shared HPCC dev node where any
user can connect to 127.0.0.1):

* binds 127.0.0.1 by default; another address needs ``--insecure-network``
  (cli.py), since the token then crosses the network in clear text;
* the app lives under a per-session path, ``/s/<nonce>/``; anything outside
  it is a 404, so the session cookie (``Path=/s/<nonce>/``) is never sent to
  another server the same browser reaches on 127.0.0.1 or localhost;
* every request needs the session token: ``?token=`` on the first page load
  is swapped for an HttpOnly, SameSite=Strict cookie and the browser is
  redirected to a token-free URL (so the token does not stay in the address
  bar or in history);
* the Host header must name the loopback interface (DNS-rebinding guard),
  unless more host names are allowed explicitly;
* POSTs also need ``X-Requested-With: mace-gui``, which a cross-site page
  cannot send without a CORS preflight this server never answers;
* file access is confined to the served directory (resolved paths only);
  every write goes through saving.py's rules (inside the folder, never
  through a symbolic link, atomic);
* a connection that does not send a request within HANDLER_TIMEOUT_S is
  dropped, and at most MAX_CONNECTIONS are served at once (a connection that
  never authenticated is the first to go when that cap is reached);
* the test hooks (``MACE_GUI_TEST_HOOKS=1``) are refused on a non-loopback
  address and never return the environment.

All URLs the page uses are relative, so the app works under its session path.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import hmac
import json
import mimetypes
import os
import re
import secrets
import shutil
import socket
import sys
import tempfile
import threading
import time
import traceback
from collections import OrderedDict
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, quote as url_quote, urlsplit

from . import db as gui_db
from . import clean as gui_clean
from . import decks, editor, plots, results, structures
from . import setup as gui_setup
from . import preflight as gui_preflight
from . import textwin
from . import spectra as gui_spectra
from . import freq as gui_freq
from .childenv import env_fingerprint, env_report
from . import builders, clirun, compute, dbbuild, dbwriters, memory, outindex, resources, saving
from . import history as gui_history
from . import runinfo
from . import cache as gui_cache
from . import filekind
from . import pool as gui_pool
from . import tasks as gui_tasks
from . import settings as gui_settings
from . import docprefs as gui_docprefs
from . import prefstore
from . import mlip_host as gui_mlip_host
from . import submit as gui_submit
from . import jobs as gui_jobs
from . import submit_routes

STATIC = Path(__file__).resolve().parent / "static"
COOKIE_PREFIX = "mace_gui_"  # + port: browsers share cookies across ports
LOOPBACK_NAMES = {"127.0.0.1", "localhost", "::1", "[::1]"}
MAX_FILES = 4000
MAX_DEPTH = 4
MAX_TEXT = 400_000
MAX_POST = 2_000_000
MAX_PARSE = 50 * 1024 * 1024        # a dropped file or pasted text
_SKIP_DIRS = {"__pycache__", ".git", "node_modules", ".ipynb_checkpoints"}

HEARTBEAT_S = 10.0
STOP_DRAIN_S = 5.0                  # Stop Server waits this long at most for canceled tasks to reach the tabs
HANDLER_TIMEOUT_S = 30.0            # a socket that sends no request in this time is closed
MAX_CONNECTIONS = 64                # connections served at once (one thread each)
IMMUTABLE = "public, max-age=31536000, immutable"


def session_nonce(token: str) -> str:
    """The session path's nonce, derived one way from the token: a fixed
    ``--token`` (or ``MACE_GUI_TOKEN``) keeps the same URL across restarts,
    a random token gives a random path, and the path never reveals the token."""
    digest = hmac.new(token.encode(), b"mace-gui-session-path", hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).decode().rstrip("=")[:22]


def static_version() -> str:
    """A short hash of every static file's name, size and mtime: the
    versioned URL prefix (``static/v-<hash>/``) changes whenever a file
    does, so those URLs can be cached forever."""
    h = hashlib.sha1()
    for p in sorted(STATIC.rglob("*")):
        if p.is_file():
            st = p.stat()
            h.update(f"{p.relative_to(STATIC)}:{st.st_size}:{st.st_mtime_ns}".encode())
    return "v-" + h.hexdigest()[:10]


def cache_limit_bytes(stored: Any = None) -> int:
    """The memory the server may spend on caches (spec section 9):
    MACE_GUI_CACHE_MB, else Settings' cache_mb, else 512 MB; 64 MB to a
    quarter of physical RAM (resources.cache_mb)."""
    if stored is None:
        stored = gui_settings.load().get("cache_mb")
    return resources.cache_limit_bytes(stored)


_MACE_DEFAULTS: Optional[Dict[str, Any]] = None


def mace_defaults() -> Dict[str, Any]:
    """What the CLI does when the user names no functional, basis set or
    example config, read from the CLI's own modules (never a GUI literal);
    None where they cannot be read. Imported on first use, then kept."""
    global _MACE_DEFAULTS
    if _MACE_DEFAULTS is not None:
        return _MACE_DEFAULTS
    out: Dict[str, Any] = {"functional": None, "basis_set": None, "example_config": None,
                           "functionals": [], "basis_sets": [], "source": {}}
    try:
        import d12_constants  # Crystal_d12 is on sys.path through decks.py
        # the CLI's blank-prompt functional (its constant's name is spelled the
        # British way, so it is found by its suffix and reported as it is)
        name = next(k for k in vars(d12_constants) if k.endswith("_FUNCTIONAL_FALLBACK"))
        out["functional"] = getattr(d12_constants, name)
        out["source"]["functional"] = f"Crystal_d12/d12_constants.py: {name}"
        out["basis_sets"] = list(d12_constants.INTERNAL_BASIS_SETS)
        out["functionals"] = [f for cat in d12_constants.FUNCTIONAL_CATEGORIES.values()
                              for f in cat.get("functionals", [])]
    except Exception:  # pragma: no cover - a different module layout: "unknown"
        pass
    try:
        import d12_config
        std = d12_config.get_default_d12_configs()["standard_dft_opt"]
        out["basis_set"] = std.get("basis_set")
        out["example_config"] = std.get("name", "standard_dft_opt")
        out["source"]["basis_set"] = "Crystal_d12/d12_config.py: get_default_d12_configs()['standard_dft_opt']['basis_set']"
        out["source"]["example_config"] = "Crystal_d12/d12_config.py: get_default_d12_configs()['standard_dft_opt']"
    except Exception:  # pragma: no cover
        pass
    _MACE_DEFAULTS = out
    return out


def warm_up():
    """Import the heavy libraries in the background after start, so the
    first structure does not pay for them (about 0.75 s on HPCC)."""
    for mod in ("numpy", "spglib", "ase", "ase.io"):
        try:
            __import__(mod)
        except Exception:
            pass


def plotly_js() -> Optional[Path]:
    """plotly.min.js from the plotly package MACE already requires."""
    try:
        import plotly
    except ImportError:
        return None
    path = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
    return path if path.is_file() else None


def classify(name: str) -> str:
    """The explorer kind of a file name (filekind.kind_by_name: the kinds
    listed before are unchanged, the new ones were hidden)."""
    return filekind.kind_by_name(name)


class Served(tuple):
    """The served folder, its drafts key and generation, swapped in one
    assignment by Open Folder (plan T6); requests read one snapshot."""
    __slots__ = ()

    def __new__(cls, root: Path, gen: int = 0):
        root = Path(root).resolve()
        return tuple.__new__(cls, (root, hashlib.sha256(str(root).encode()).hexdigest()[:16], gen))

    root = property(lambda self: self[0])
    dir_hash = property(lambda self: self[1])
    gen = property(lambda self: self[2])


HIDDEN_REFUSAL = ("Hidden folders cannot be opened here (they hold settings and credentials). Start mace gui "
                  "with --dir to serve one on purpose.")
HOME_REFUSAL = "Open a project folder inside your home, not your home folder itself."
OUTSIDE_REFUSAL = ("This folder is outside the folders this server may open ({names}). Restart mace gui with "
                   "--root to allow it.")
OFF_REFUSAL = "Open Folder is turned off for this server (it was started with --root none)."
MAX_FOLDERS = 2000
# the deny page's own policy: its one script (login.js, the sign-in form) and nothing else
DENY_CSP = ("default-src 'none'; script-src 'self'; connect-src 'self'; style-src 'unsafe-inline'; "
            "form-action 'none'; base-uri 'none'; frame-ancestors 'none'")
# POST routes that do not act on the served folder (a stale page may still use them)
ROOT_FREE_POSTS = {"open-folder", "session/touch", "shutdown", "login-code", "diagnostics", "settings", "profile",
                   "managed/attach", "managed/ping", "managed/detach", "managed/stop"}


class GuiState:
    """What the handlers share: root dir, token, caches."""

    def __init__(self, root: Path, token: str, allowed_hosts: Optional[List[str]] = None,
                 test_hooks: Optional[bool] = None, roots: Optional[List[str]] = None):
        self.root_lock = threading.Lock()
        self.served = Served(root)
        from .sessions import Sessions
        self.web = Sessions(token, gui_settings.load)
        self.start_dir = self.served.root
        self._tls = threading.local()
        from . import roots as gui_roots
        self.allowed_roots = gui_roots.allowed_roots(roots)
        self.token = token
        self.prefix = f"/s/{session_nonce(token)}/"     # the app's path; the cookie's Path
        self.allowed_hosts = set(LOOPBACK_NAMES) | {h.lower() for h in (allowed_hosts or [])}
        self.cache = Path(tempfile.mkdtemp(prefix="mace_gui_cache_"))
        self._static_cache: "OrderedDict[Tuple[str, bool], bytes]" = OrderedDict()
        self._static_bytes = 0
        self.lock = threading.Lock()          # the static cache only
        self.sessions = editor.SessionStore()  # each session has its own lock
        # one memory budget for the static cache, the undo history and the
        # live edit sessions (spec section 9.2)
        self.ledger = memory.MemoryLedger(cache_limit_bytes())
        self.ledger.attach_static(self)
        self.ledger.attach_sessions(self.sessions)
        # parsed files and the view-only built structures (cache.py), on the same budget
        self.parse_cache = gui_cache.ParseCache()
        self.held = gui_cache.HeldStore(lambda: self.ledger.held_pinned_cap())
        self.ledger.attach_parse(self.parse_cache)
        self.ledger.attach_held(self.held)
        # what the CLI children write here (plot renders, Setup runs): on /tmp, so on the same budget
        self.disk = gui_cache.DiskCache(self.cache / "runs")
        self.ledger.attach_disk(self.disk)
        self.version = static_version()
        self.stopping = threading.Event()
        self.started = time.time()
        # test-only hooks (a delay per request, eviction, counters): only when
        # the server process itself was started with MACE_GUI_TEST_HOOKS=1
        self.test_hooks = (os.environ.get("MACE_GUI_TEST_HOOKS") == "1") if test_hooks is None else bool(test_hooks)
        self.test_log: List[Dict[str, Any]] = []
        # test hook: profile, settings and docprefs saves answer 503 (a server that is unreachable for saves)
        self.fail_pref_saves = False
        # background tasks in the process pool (created with the first job, spec section 8.4)
        self.tasks = gui_tasks.TaskManager(lambda m: gui_pool.Pool(
            lambda: resources.effective_workers(gui_settings.load().get("max_workers")), m.on_pool_progress))
        self.clients: Dict[str, Dict[str, Any]] = {}     # live /api/events connections
        # Clean's tiers: the one ML worker (mlip_host.py) and what is installed
        self.mlip_host = gui_mlip_host.MlipHost()
        self.tasks.on_cancel = self.mlip_host.cancel_lease
        self.clean_methods: Dict[str, Any] = {"checking": True, "methods": _clean_methods_skeleton()}
        self.clean_methods_lock = threading.Lock()
        # File History keeps this folder's replaced files (outside the folder, history.py)
        gui_history.configure(self.root)
        # Submit, Jobs and Batch Setup (PLAN_submit): the previews, this session's jobs, the own-jobs poll
        self.bind_host = "127.0.0.1"
        self.submits = gui_submit.Store()
        # jobs.finished: one event per job for the life of the server, whichever of the Jobs
        # panel's refresh (jobs.py) and the Cluster view's poller (cluster_api) sees it first
        self.finished_lock = threading.Lock()
        self.finished_ids: set = set()
        self.jobs = gui_jobs.Jobs(self.root, self._job_event, self.submits.session_jobs,
                                  lambda: gui_settings.load().get("cluster.pollSeconds", gui_jobs.POLL_DEFAULT_S))
        self.jobs.watcher.skip = lambda jid: jid in self.finished_ids
        # the decks and outputs of this folder's queued and running jobs (File History's
        # protected kinds, C19-b; the wavefunction copy's "still running", C18.7)
        saving.JOB_STATE_PROVIDER = self.jobs.deck_reason
        gui_setup.JOB_STATE_PROVIDER = self.jobs.out_state
        self.managed = None             # managed.Controller under `mace gui --managed`
        self.stop_pending: Optional[Dict[str, Any]] = None   # Sign Out ▸ Stop When Tasks Finish
        # file and folder names this session has seen, for Copy Diagnostics' redaction (bounded)
        self.names_seen: set = set()
        self.stopping_browser = False
        self.watchdog_hooks: List[Any] = []    # called every WATCHDOG_S by the server watchdog
        self.request_stop = None        # cli.install_request_stop
        self.stop_reason = None
        self.tunnel = ""
        self.connect = ""
        self.vscode = False
        self.node = socket.gethostname().split(".")[0]

    def _job_event(self, name: str, data: Dict[str, Any]) -> None:
        """Server events of the jobs code; jobs.finished goes out once per job id,
        with what the Jobs panel knows of it (its output) added."""
        if name != "jobs.finished":
            self.tasks.broadcast(name, data)
            return
        jid = str(data.get("id") or "")
        with self.finished_lock:
            if not jid or jid in self.finished_ids:
                return
            self.finished_ids.add(jid)
        ev = dict(data)
        row = self.jobs.last_rows.get(jid) if getattr(self, "jobs", None) is not None else None
        if row and not ev.get("out_rel"):
            ev["out_rel"] = row.get("out_rel")
        if "known" not in ev:
            ev["known"] = bool(ev.get("state")) and not ev.get("left_queue")
        self.tasks.broadcast(name, ev)

    def note_names(self, *names: str) -> None:
        """Remember names to redact from diagnostics (at most 5000)."""
        for n in names:
            if n and len(self.names_seen) < 5000:
                self.names_seen.add(str(n))

    def mint_login_url(self, base: str) -> str:
        """A one-time sign-in link for this server (the printed URL, --open, connect)."""
        return f"{base}?login={self.web.mint_code()}"

    # -- the served folder: one snapshot per request (_dispatch) ----------------
    def snapshot(self) -> None:
        self._tls.served = self.served

    def _current(self) -> Served:
        return getattr(self._tls, "served", None) or self.served

    @property
    def root(self) -> Path:
        return self._current().root

    @property
    def dir_hash(self) -> str:
        """The key browser drafts are stored under: the folder, not the token."""
        return self._current().dir_hash

    @property
    def root_gen(self) -> int:
        return self._current().gen

    def set_root(self, path: Path) -> Served:
        """Serve another folder (File > Open Folder): swap in one assignment,
        drop the editor sessions and held structures of the old folder, tell
        every tab. The parse cache keeps its absolute-path entries."""
        with self.root_lock:
            new = Served(path, self.served.gen + 1)
            self.served = new
        self._tls.served = new
        self.note_names(*new.root.parts[-3:])
        for sid in list(self.sessions.ids()):
            self.sessions.evict(sid)
        try:
            self.held.release(list(getattr(self.held, "items", {}).keys()))
        except Exception:
            pass
        gui_history.configure(new.root)          # File History follows the served folder
        if getattr(self, "jobs", None) is not None:
            self.jobs.root = new.root            # and the Jobs panel's folder scope
            self.jobs.last_rows = {}
            gui_jobs.forget_manifests()
        self.tasks.broadcast("root-changed", {"root": str(new.root), "dir_hash": new.dir_hash, "gen": new.gen})
        print(f"mace gui: now serving {new.root} (Open Folder)", file=sys.stderr, flush=True)
        return new

    @property
    def cache_limit(self) -> int:
        return self.ledger.limit

    @cache_limit.setter
    def cache_limit(self, limit: int):
        self.ledger.limit = int(limit)

    def close(self):
        self.stopping.set()
        self.jobs.stop.set()
        _finish_runs(self)
        self.mlip_host.shutdown()
        p = self.tasks.pool_if_started()
        if p is not None:
            p.shutdown(kill=True)
        shutil.rmtree(self.cache, ignore_errors=True)

    def set_limit(self, limit: int):
        """A new cache cap (Settings' cache_mb changed): evict at once when it shrinks."""
        self.ledger.set_limit(limit)

    def cache_used(self) -> int:
        """Everything the ledger counts: static files, undo history, live sessions."""
        return self.ledger.total()

    # -- the static cache, as the ledger sees it --------------------------------
    def static_used(self) -> int:
        with self.lock:
            return self._static_bytes

    def drop_oldest_static(self) -> bool:
        with self.lock:
            if len(self._static_cache) <= 1:
                return False
            _, old = self._static_cache.popitem(last=False)
            self._static_bytes -= len(old)
            return True

    # -- paths --------------------------------------------------------------
    def resolve(self, rel: Optional[str], must_exist: bool = True) -> Path:
        from . import roots as gui_roots
        if not rel:
            raise PermissionError("no path given")
        root = self.root
        self.note_names(str(rel))
        target = (root / rel).resolve()
        if target != root and root not in target.parents:
            raise PermissionError("outside the served directory")
        # credentials are never read or written, even when the served folder is a home folder
        asked = Path(os.path.normpath(str(rel))).parts
        found = gui_roots.credential_component(asked) or \
            gui_roots.credential_component(target.relative_to(root).parts)
        if found:
            raise PermissionError(gui_roots.secret_refusal(found))
        if must_exist and not target.exists():
            raise FileNotFoundError(rel)
        return target

    def rel(self, path: Path) -> str:
        return path.resolve().relative_to(self.root).as_posix()

    def rel_new(self, path: Path) -> str:
        """A path inside the root (that may not exist yet), relative to it."""
        from . import roots as gui_roots
        root = self.root
        rp = Path(os.path.normpath(os.path.join(str(root), str(path))))
        r = rp.relative_to(root).as_posix()
        hidden = gui_roots.hidden_component(Path(r).parts)
        if hidden:
            raise PermissionError(f"{hidden} is a hidden folder or file; MACE GUI does not write there")
        return "" if r == "." else r

    def check_write(self, target: Path, allow=()) -> None:
        """A file the server is about to write: inside the root, through no
        symlink that leads out, not into a hidden (dot) folder or file, and not
        a protected kind (saving.protected_reason: outputs, wavefunctions, data
        files, databases, MACE job scripts, the deck of a queued job). ``allow``
        names the allowances: "wavefunction" (opt2d3's copy, checked by its
        caller), "database" (the DB build) and "grid" (a new .CUBE that Save
        Result As writes; never one that exists)."""
        from . import roots as gui_roots
        rp = Path(os.path.normpath(str(target)))
        if rp != self.root and self.root not in rp.parents:
            raise PermissionError("outside the served directory")
        hidden = gui_roots.hidden_component(rp.relative_to(self.root).parts)
        if hidden:
            raise PermissionError(f"{hidden} is a hidden folder or file; MACE GUI does not write there")
        parent = rp.parent
        while not parent.exists():
            parent = parent.parent
        real = parent.resolve()
        if real != self.root and self.root not in real.parents:
            raise PermissionError("outside the served directory")
        if rp.is_symlink():
            raise PermissionError(f"{rp.name} is a symbolic link; not overwritten")
        why = saving.protected_reason(rp)
        if why:
            low = rp.name.lower()
            if "wavefunction" in allow and (low.endswith((".f9", ".f98"))):
                return
            if "database" in allow and low.endswith(".db"):
                return
            if "grid" in allow and low.endswith(".cube") and not os.path.lexists(rp):
                return
            raise saving.ProtectedOutput(why)

    def list_files(self) -> Dict[str, Any]:
        from . import roots as gui_roots
        files: List[Dict[str, Any]] = []
        truncated = False

        def walk(d: Path, depth: int):
            nonlocal truncated
            try:
                entries = sorted(d.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
            except OSError:
                return
            for p in entries:
                if len(files) >= MAX_FILES:
                    truncated = True
                    return
                if p.name.startswith(".") or p.name in _SKIP_DIRS:
                    continue
                if gui_roots.credential_component((p.name,)):
                    continue                  # hidden for safety (id_rsa, *_history, ...)
                try:
                    if p.is_dir():
                        if depth < MAX_DEPTH and not p.is_symlink():
                            walk(p, depth + 1)
                        continue
                    kind = classify(p.name)
                    if not kind:
                        continue
                    # a .DAT named like a cube is listed only when it has a cube header
                    if kind == "cube" and p.name.upper().endswith(".DAT") and not filekind.cube_header_ok(p):
                        continue
                    resolved = p.resolve()
                    if resolved != self.root and self.root not in resolved.parents:
                        continue
                    st = p.stat()
                except OSError:
                    continue
                files.append({"path": p.relative_to(self.root).as_posix(),
                              "name": p.name, "kind": kind, "size": st.st_size,
                              "mtime": st.st_mtime, "mtime_ns": str(int(st.st_mtime_ns))})

        walk(self.root, 0)
        self.note_names(*(f["path"] for f in files[:2000]))
        return {"root": str(self.root), "files": files, "truncated": truncated}

    # -- static -------------------------------------------------------------
    def static_bytes(self, name: str, gz: bool) -> Optional[Tuple[bytes, str]]:
        if name == "vendor/plotly.min.js":
            path = plotly_js()
        else:
            path = (STATIC / name).resolve()
            if STATIC.resolve() not in path.parents:
                return None
        if path is None or not path.is_file():
            return None
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript",):
            ctype += "; charset=utf-8"
        key = (name, gz)
        with self.lock:
            hit = self._static_cache.get(key)
            if hit is not None:
                self._static_cache.move_to_end(key)
                return hit, ctype
        # compress outside the lock: plotly takes a moment at level 9, and
        # the other files should not wait for it
        data = path.read_bytes()
        if name == "index.html":
            data = data.replace(b"@V@", self.version.encode())
        body = gzip.compress(data, 9) if gz else data
        with self.lock:
            if key not in self._static_cache:
                self._static_cache[key] = body
                self._static_bytes += len(body)
        # bounded (spec section 9): a quarter of the budget, least recently used first
        self.ledger.enforce()
        with self.lock:
            return self._static_cache.get(key, body), ctype


class GuiHandler(BaseHTTPRequestHandler):
    server_version = "MACE-GUI"
    sys_version = ""
    protocol_version = "HTTP/1.1"
    timeout = HANDLER_TIMEOUT_S   # socketserver: a socket that sends nothing is closed after this
    state: GuiState = None  # set by make_server
    verbose = False

    # -- plumbing -----------------------------------------------------------
    def log_message(self, fmt, *args):  # quiet, and never log the token
        if self.verbose:
            msg = fmt % args
            if self.state and self.state.token:
                msg = msg.replace(self.state.token, "<token>")
            super().log_message("%s", msg)

    def _send(self, status: int, body: bytes, ctype: str,
              headers: Optional[Dict[str, str]] = None):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj: Any, status: int = 200, timing: str = ""):
        body = json.dumps(obj, default=_json_default).encode()
        headers = {"Cache-Control": "no-store"}
        t0 = getattr(self, "_t0", None)
        if t0 is not None:
            headers["Server-Timing"] = f"app;dur={(time.perf_counter() - t0) * 1000:.1f}"
        if timing:
            headers["Server-Timing"] = timing + (", " + headers["Server-Timing"] if "Server-Timing" in headers else "")
        if status == 200 and self.command in ("GET", "HEAD"):
            # same body, same ETag: a revalidating GET costs no body bytes
            etag = '"' + hashlib.sha1(body).hexdigest()[:20] + '"'
            headers["ETag"] = etag
            headers["Cache-Control"] = "no-cache"
            if etag in (self.headers.get("If-None-Match") or ""):
                return self._send(304, b"", "application/json", headers)
        if len(body) > 4096 and "gzip" in self.headers.get("Accept-Encoding", ""):
            body = gzip.compress(body, 5)
            headers["Content-Encoding"] = "gzip"
        self._send(status, body, "application/json", headers)

    def _db_busy(self, exc):
        """A job is writing the database (or a killed writer left a hot
        journal): not an error; the page keeps what it shows and retries."""
        body = json.dumps({"busy": True, "reason": exc.reason, "retry_after": 2, "message": str(exc)}).encode()
        return self._send(503, body, "application/json", {"Retry-After": "2", "Cache-Control": "no-store"})

    def _error(self, status: int, message: str):
        self._json({"error": message}, status)

    def _missing(self, exc: ImportError):
        """A package MACE requires (ase, spglib, plotly) is not installed for
        this Python: say which, and how to install it, instead of failing."""
        name = (getattr(exc, "name", None) or str(exc)).split(".")[0]
        return self._error(503, f"this needs the package {name} (in MACE's requirements.txt), which is not "
                                f"installed for this Python: pip install {name}")

    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").strip().lower()
        if host.startswith("["):
            name = host.split("]")[0] + "]"
        else:
            name = host.rsplit(":", 1)[0] if ":" in host else host
        return name in self.state.allowed_hosts

    def _token_from_cookie(self) -> Optional[str]:
        """The session cookie's value (a session id since the one-time links)."""
        raw = self.headers.get("Cookie")
        if not raw:
            return None
        try:
            c = SimpleCookie()
            c.load(raw)
        except Exception:
            return None
        morsel = c.get(self._cookie_name())
        return morsel.value if morsel else None

    def _cookie_name(self) -> str:
        return f"{COOKIE_PREFIX}{self.server.server_address[1]}"

    def _session_cookie(self, sid: str) -> str:
        return f"{self._cookie_name()}={sid}; HttpOnly; SameSite=Strict; Path={self.state.prefix}"

    def _auth(self) -> str:
        """"ok", "expired", "none" or "paced:<status>" for this request: the
        X-Mace-Token header (scripts, mace gui connect, tests) or a session
        cookie. The token is never accepted in a URL or a cookie. A wrong
        header token is paced (sessions.Sessions.wrong)."""
        ss = self.state.web
        hdr = self.headers.get("X-Mace-Token")
        if hdr is not None:
            if ss.token_ok(hdr):
                return "ok"
            return f"paced:{ss.wrong()}"
        return ss.state_of(self._token_from_cookie())

    def _authorized(self, query: Dict[str, List[str]]) -> bool:
        return self._auth() == "ok"

    def _deny_page(self, status: int = 403, note: str = "", form: bool = True):
        """The page a browser sees without a session (403), with an expired or
        used link, or outside the session path (404, no form: its sign-in
        would go to the wrong path). It never echoes the request."""
        # color-scheme: a browser in dark mode shows it dark, as the app itself
        parts = ["<!doctype html><meta charset=utf-8><meta name=color-scheme content='light dark'><title>MACE</title>",
                 "<body style='font:16px system-ui;margin:3rem;max-width:44rem'><h1>MACE</h1>"]
        if note:
            parts.append(f"<p id=note><b>{note}</b></p>")
        parts.append("<p>Open the sign-in link that <code>mace gui</code> printed in your terminal (it ends in "
                     "<code>?login=...</code> and works once). For a new link run <code>mace gui login-url</code>, "
                     "or press Enter in <code>mace gui connect</code>. If the server was restarted, its new link is "
                     "different.</p>")
        if form:
            parts.append("<form id=signin><label>Or paste the token that <code>mace gui</code> printed: "
                         "<input id=token autocomplete=off size=36 type=password></label> "
                         "<button type=submit>Sign In</button> <span id=err role=alert></span></form>"
                         "<script src='login.js'></script>")
        parts.append("<p>In VS Code: Ctrl+click the URL in the terminal; the Ports tab opens this address "
                     "without the sign-in link.</p><p>See Troubleshooting in docs/gui.md.</p></body>")
        body = "".join(parts).encode()
        self._send(status, body, "text/html; charset=utf-8", {"Cache-Control": "no-store",
                                                              "Content-Security-Policy": DENY_CSP})

    def _same_origin(self) -> bool:
        """A sign-in POST from this page only (Origin / Sec-Fetch-Site when sent)."""
        site = self.headers.get("Sec-Fetch-Site")
        if site and site not in ("same-origin", "none"):
            return False
        origin = self.headers.get("Origin")
        if origin:
            host = urlsplit(origin).hostname or ""
            return host.lower() in self.state.allowed_hosts or f"[{host}]" in self.state.allowed_hosts
        return True

    def _pre_auth(self, method: str, path: str, query) -> bool:
        """The only things served without a session: the ?login= exchange,
        login.js and POST login. True when the request was answered."""
        ss = self.state.web
        if method == "GET" and path in ("/", "/index.html") and "login" in query:
            sid = ss.use_code((query.get("login") or [""])[0])
            if sid is None and ss.state_of(self._token_from_cookie()) == "ok":
                # this browser is signed in already (the link reopened from history or the
                # terminal): show the app; the used code still signs nobody new in
                self._send(303, b"", "text/plain", {"Location": "./", "Cache-Control": "no-store"})
                return True
            if sid is None:
                st = ss.wrong()
                if st == 429:
                    self._error(429, "too many wrong sign-in attempts at once; wait a moment")
                else:
                    self._deny_page(403, "This sign-in link was already used or has expired.")
                return True
            self._send(303, b"", "text/plain", {"Location": "./", "Set-Cookie": self._session_cookie(sid),
                                                "Cache-Control": "no-store"})
            return True
        if method == "GET" and path == "/login.js":
            body = (STATIC / "login.js").read_bytes()
            self._send(200, body, "text/javascript; charset=utf-8",
                       {"Cache-Control": "no-store", "Content-Security-Policy": DENY_CSP})
            return True
        if method == "POST" and path == "/login":
            if self.headers.get("X-Requested-With") != "mace-gui" or not self._same_origin():
                self._error(403, "sign-in refused (not from this page)")
                return True
            try:
                body = self._read_json_body()
            except (ValueError, json.JSONDecodeError):
                self._error(400, "send {\"token\": ...}")
                return True
            if ss.token_ok(str(body.get("token") or "")) and not ss.revoked:
                sid = ss.new_session()
                body = json.dumps({"signed_in": True}).encode()
                self._send(200, body, "application/json", {"Set-Cookie": self._session_cookie(sid),
                                                           "Cache-Control": "no-store"})
                return True
            st = ss.wrong()
            self._error(st, "wrong token" if st == 403 else "too many wrong sign-in attempts at once; wait a moment")
            return True
        return False

    # -- dispatch -----------------------------------------------------------
    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def _dispatch(self, method: str):
        self._t0 = time.perf_counter()
        self.state.snapshot()          # one served folder for the whole request
        try:
            if not self._host_ok():
                return self._send(421, b"Misdirected request: unexpected Host header\n",
                                  "text/plain; charset=utf-8")
            url = urlsplit(self.path)
            query = parse_qs(url.query)
            prefix = self.state.prefix
            if url.path == prefix[:-1]:                      # /s/<nonce> without its slash
                return self._send(301, b"", "text/plain", {"Location": prefix + (f"?{url.query}" if url.query else ""),
                                                           "Cache-Control": "no-store"})
            if not url.path.startswith(prefix):
                if url.path.startswith("/s/"):
                    # another session's path: a page from before a restart. 403,
                    # as for a wrong token, so it says the server restarted
                    if url.path.count("/") == 3 and url.path.endswith("/") or url.path.endswith("/index.html"):
                        return self._deny_page(form=False)
                    return self._error(403, "missing or wrong token")
                # outside any session path: nothing here, with or without the token
                if url.path in ("/", "/index.html"):
                    return self._deny_page(404, form=False)
                return self._error(404, "not found")
            path = "/" + url.path[len(prefix):]
            if self._pre_auth(method, path, query):
                return
            auth = self._auth()
            if auth.startswith("paced:"):
                code = int(auth.split(":")[1])
                return self._error(code, "missing or wrong token" if code == 403 else
                                   "too many wrong attempts at once; wait a moment")
            if auth == "expired":
                # an expired (not revoked) session may still load the page's files, so
                # the Signed Out dialog can show; the API answers 401
                if path.startswith("/static/"):
                    name = path[len("/static/"):]
                    if name.startswith(self.state.version + "/"):
                        return self._static(name[len(self.state.version) + 1:], immutable=True)
                    return self._static(name)
                if path == "/api/session":
                    return self._json(self._session_info(expired=True))
                if path in ("/", "/index.html"):
                    return self._deny_page(403, "You were signed out after a time without use.")
                return self._json({"error": "session expired", "expired": True}, 401)
            if auth != "ok":
                if path in ("/", "/index.html"):
                    return self._deny_page()
                return self._error(403, "missing or wrong token")
            mark = getattr(self.server, "mark_authenticated", None)
            if mark is not None:
                mark(self.connection)
            if path in ("/", "/index.html"):
                if "token" in query:
                    # signed in already: drop the stale ?token= from the address bar (it never signs in)
                    return self._send(303, b"", "text/plain", {"Location": "./", "Cache-Control": "no-store"})
                # ?pane=side: the right editor group (groups.js), framed by this page only
                return self._index(pane=(query.get("pane") or [""])[0] == "side")
            if path.startswith("/static/"):
                name = path[len("/static/"):]
                if name.startswith(self.state.version + "/"):
                    return self._static(name[len(self.state.version) + 1:], immutable=True)
                return self._static(name)
            if path.startswith("/api/"):
                if method == "POST":
                    if self.headers.get("X-Requested-With") != "mace-gui":
                        return self._error(403, "missing X-Requested-With header")
                    gen = self.headers.get("X-Mace-Root-Gen")
                    if gen not in (None, "") and path[5:] not in ROOT_FREE_POSTS and gen != str(self.state.root_gen):
                        return self._error(409, "The folder changed in another tab; reload this page.")
                    if path == "/api/parse":
                        return self._parse(query)          # a raw file body, never JSON
                    return self._api_post(path[5:])
                return self._api_get(path[5:], query)
            return self._error(404, "not found")
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:  # last resort: report, keep serving
            if self.verbose:
                traceback.print_exc()
            try:
                self._error(500, f"{type(exc).__name__}: {exc}")
            except Exception:
                pass

    def _index(self, pane: bool = False):
        """The page, with the profile's theme, density, motion and UI scale on
        its <html> tag, so a new port or a fresh browser paints them at once
        (no flash). Built per request from the cached template, never cached
        itself, no ETag: a returning browser can never revalidate to an older
        page with older preferences (the 304 path is for the static files)."""
        found = self.state.static_bytes("index.html", False)
        if found is None:
            return self._error(404, "no static file index.html")
        template, ctype = found
        body = template.replace(b"@PREFS@", _prefs_attrs().encode())
        headers = {"Cache-Control": "no-store", "Vary": "Accept-Encoding",
                   "Content-Security-Policy": CSP_PANE if pane else CSP}
        if "gzip" in self.headers.get("Accept-Encoding", ""):
            body = gzip.compress(body, 6)
            headers["Content-Encoding"] = "gzip"
        self._send(200, body, ctype, headers)

    def _session_info(self, expired: bool = False) -> Dict[str, Any]:
        p = self.state.web.policy()
        return {"expired": expired, "idle_hours": p.get("idle_hours"), "max_days": p.get("max_days"),
                "idle_stop_hours": p.get("idle_stop_hours")}

    def _static(self, name: str, no_store: bool = False, immutable: bool = False):
        gz = "gzip" in self.headers.get("Accept-Encoding", "")
        found = self.state.static_bytes(name, gz)
        if found is None:
            return self._error(404, f"no static file {name}")
        body, ctype = found
        # The page revalidates (a 304 costs a few hundred bytes); versioned
        # files are immutable, and also answer a revalidation with a 304,
        # since Firefox honors "immutable" only over https.
        headers = {"Cache-Control": "private, no-cache" if no_store else
                   IMMUTABLE if immutable else "private, max-age=3600",
                   "ETag": f'"{self.state.version}-{hashlib.sha1(name.encode()).hexdigest()[:8]}{"-gz" if gz else ""}"',
                   "Vary": "Accept-Encoding"}
        if gz:
            headers["Content-Encoding"] = "gzip"
        if headers["ETag"] in (self.headers.get("If-None-Match") or ""):
            return self._send(304, b"", ctype, {k: v for k, v in headers.items() if k != "Content-Encoding"})
        if name == "index.html":
            headers["Content-Security-Policy"] = CSP
        self._send(200, body, ctype, headers)

    # -- API ----------------------------------------------------------------
    def _q(self, query, name, default=None):
        return (query.get(name) or [default])[0]

    def _api_get(self, route: str, query):
        st = self.state
        try:
            if route.startswith("cluster/"):
                from . import cluster_api
                return cluster_api.get(self, route[len("cluster/"):], query)
            if route == "info":
                return self._json(self._info())
            if route == "boot":
                # everything the page needs to start, in one (gzipped) request
                return self._json({"info": self._info(), "files": st.list_files(),
                                   "templates": decks.templates(summary=True),
                                   "settings": {"settings": gui_settings.load(),
                                                "path": str(gui_settings.path())},
                                   "profile": _profile_view(st),
                                   "mace_defaults": mace_defaults()})
            if route == "files":
                listing = st.list_files()
                if self._q(query, "details") == "1":
                    listing["details"] = _file_details(st, listing["files"], self.headers.get("X-Mace-Client"))
                return self._json(listing)
            if route == "resources":
                # the limits, and what each store of the one memory budget holds
                return self._json(self._resources())
            if route == "tasks":
                return self._json({"tasks": st.tasks.list()})
            if route == "folders":
                return self._json(_folders(st, self._q(query, "path")))
            if route == "session":
                return self._json(self._session_info())
            if route.startswith("tasks/") and route.endswith("/result"):
                t = st.tasks.get(route[len("tasks/"):-len("/result")])
                if t.state != "done":
                    return self._json({"error": f"the task is {t.state}", "task": t.record()}, 409)
                if t.result is None:
                    return self._error(410, "this result was already fetched or let go of; run it again")
                res, t.result = t.result, None
                if isinstance(res, dict):
                    res = {**res, "task": t.record()}
                return self._json(res)
            if route == "settings":
                return self._json({"settings": gui_settings.load(),
                                   "path": str(gui_settings.path())})
            if route == "profile":
                return self._json(_profile_view(st, everything=self._q(query, "all") == "1"))
            if route == "docprefs":
                kind = self._q(query, "kind", "")
                if kind not in gui_docprefs.KINDS:
                    raise ValueError(f"kind must be one of {', '.join(gui_docprefs.KINDS)}")
                rel = _docprefs_rel(st, self._q(query, "path"))
                return self._json({"kind": kind, "path": rel, "values": gui_docprefs.get(kind, st.dir_hash, rel)})
            if route == "clean/methods":
                with st.clean_methods_lock:
                    return self._json(dict(st.clean_methods))
            if route == "events":
                return self._events(query)
            if route == "structure":
                return self._structure(query)
            if route == "filekind":
                p = st.resolve(self._q(query, "path"))
                if not p.is_file():
                    raise FileNotFoundError(st.rel(p))
                out = {"path": st.rel(p), **filekind.kind_payload(st, p)}
                if out.get("role") == "crystal_out":
                    # what opening it needs: modes to show, a geometry at all (the index is cached for the open)
                    idx = _out_index(st, p)
                    out.update(freq=bool(idx.freq), nogeom=not idx.n_steps, kind="freq" if idx.freq else "out")
                    if idx.n_steps:
                        out["dim"] = idx.dim
                return self._json(out)
            if route in ("volume", "volume/grid", "volume/slice"):
                return _volume_get(self, route, query)
            if route == "props":
                p = st.resolve(self._q(query, "path"))
                if not p.is_file():
                    raise FileNotFoundError(st.rel(p))
                return self._json(filekind.props_payload(st, p))
            if route == "fileinfo":
                p = st.resolve(self._q(query, "path"))
                if not p.is_file():
                    raise FileNotFoundError(st.rel(p))
                inf = filekind.info_payload(p, filekind.kind_by_name(p.name))
                inf["sibling"] = filekind.rel_inside(st, inf["sibling"])
                if inf.get("dat"):
                    inf["dat"] = filekind.rel_inside(st, inf["dat"])
                return self._json({"path": st.rel(p), **inf})
            if route == "joblog/link":
                from . import joblink
                p = st.resolve(self._q(query, "path"))
                if not p.is_file():
                    raise FileNotFoundError(st.rel(p))
                if filekind.role_only(st, p) != "joblog":
                    raise filekind.KindConflict("This is not a job log")
                out = joblink.link(st.root, p)
                if p.suffix.lower() in (".e", ".err") or re.search(r"\.e\d+$", p.name):
                    out["errors_first_line"] = joblink.first_line(p)
                return self._json({"path": st.rel(p), **out})
            if route == "held":
                e = st.held.get(self._q(query, "id"))
                return self._json({**e["structure"], "held": e["id"], "doc_meta": e.get("meta") or {}})
            if route == "trajectory":
                p = st.resolve(self._q(query, "path"))
                stamp = saving.file_stamp(p)          # a running OPT: the page notices a change on focus
                data = structures.trajectory(p)
                data["file"] = stamp
                data["path"] = st.rel(p)
                for f in data["frames"]:
                    f["source"]["path"] = data["path"]
                return self._json(data)
            if route == "trajectory/compare":
                p = st.resolve(self._q(query, "path"))
                idx = _out_index(st, p)          # two blocks parsed, not the whole run
                i, j = int(self._q(query, "i", "0")), int(self._q(query, "j", "-1"))
                sa, sb = idx.block_structure(range(idx.n_steps)[i]), idx.block_structure(range(idx.n_steps)[j])
                cmp = structures.compare(sa, sb)
                cmp["match"], cmp["max_mismatch"] = "index", 0.0
                if self._q(query, "sym") == "1":
                    import numpy as np
                    cmp["symmetry_change"] = structures.symmetry_change(
                        {**sa, "frac": np.linalg.solve(np.asarray(sa["cell"]).T, np.asarray(sa["cart"]).T).T.tolist()},
                        {**sb, "frac": np.linalg.solve(np.asarray(sb["cell"]).T, np.asarray(sb["cart"]).T).T.tolist()})
                return self._json(cmp)
            if route == "out/summary":
                return self._out_summary(query)
            if route in ("out/convergence", "out/problems", "out/meta"):
                return _run_route(self, route, query)
            if route == "out/step":
                p = st.resolve(self._q(query, "path"))
                idx = _out_index(st, p)
                k = int(self._q(query, "step", "-1"))
                if not -idx.n_steps <= k < idx.n_steps:
                    raise ValueError(f"steps run from 0 to {idx.n_steps - 1}")
                out = idx.payload(k, {"kind": "out", "path": st.rel(p)})
                out["path"] = st.rel(p)
                out["step"] = range(idx.n_steps)[k]
                return self._json(out)
            if route == "compare":
                a = st.resolve(self._q(query, "a"))
                b = st.resolve(self._q(query, "b"))
                pa, pb = _read_structure(st, a), _read_structure(st, b)
                try:
                    cmp = structures.compare_payloads(pa, pb)
                except structures.MatchError as exc:
                    return self._json({"error": str(exc), "mismatch": True, "a": pa, "b": pb}, 400)
                cmp["symmetry_change"] = structures.symmetry_change(pa, pb)
                if cmp["order"]:
                    pb = {**pb, **{k: [pb[k][i] for i in cmp["order"]] for k in ("frac", "cart", "numbers", "symbols")
                                   if isinstance(pb.get(k), list)}}
                return self._json({"a": pa, "b": pb, "comparison": cmp})
            if route == "compare/input":
                return _compare_input_route(self, query)
            if route == "text":
                # windows of at most 256 KB, on line boundaries (spec 8.4.6): the whole file is never sent
                p = st.resolve(self._q(query, "path"))
                if not p.is_file():
                    raise ValueError("not a file")
                if textwin.is_binary(p):
                    return self._json({"error": "This looks like a binary file (NUL bytes in the first 4 KB)", "binary": True}, 415)
                if self._q(query, "since") not in (None, ""):
                    # Follow: the whole lines written after `since`, and whether the file was replaced
                    out = textwin.follow(p, int(self._q(query, "since")), self._q(query, "snap", "") or "")
                    out["path"] = st.rel(p)
                    return self._json(out)
                w = textwin.window(p, int(self._q(query, "offset", "0") or 0),
                                   int(self._q(query, "limit", str(textwin.WINDOW_MAX)) or textwin.WINDOW_MAX))
                w["path"] = st.rel(p)
                w["truncated"] = not w["eof"] or w["offset"] > 0
                w["file"] = {"mtime_ns": str(p.stat().st_mtime_ns), "size": w["size"]}
                return self._json(w)
            if route == "text/lines":
                p = st.resolve(self._q(query, "path"))
                if not p.is_file():
                    raise ValueError("not a file")
                key = gui_cache.file_key(p, ("lines", textwin.LINE_STEP))
                idx = st.parse_cache.get(key)
                if idx is None:
                    idx = compute.GATE.run(None, lambda: textwin.line_index(p))
                    st.parse_cache.put(key, idx, 64 + 8 * len(idx["offsets"]))
                    st.ledger.enforce()
                from mace.utils import livefile
                lv = livefile.state(p, _live_minutes())
                # Follow continues from the last whole line of this index, for this file (snap)
                return self._json({**idx, "path": st.rel(p), "live": lv["live"], "finished": lv["finished"],
                                   "age_s": lv["age_s"], "snap": lv["snap"]})
            if route == "out/live":
                # stat and the last 64 KB only: a live run's documents poll this, then the summary by ETag
                p = st.resolve(self._q(query, "path"))
                from mace.utils import livefile
                lv = livefile.state(p, _live_minutes())
                fs = saving.file_stamp(p)
                return self._json({**lv, "file": fs, "live_minutes": _live_minutes()})
            if route == "raw":
                p = st.resolve(self._q(query, "path"))
                kind = classify(p.name)
                if kind not in ("image", "html"):
                    return self._error(403, "only images and HTML plots are served raw")
                ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
                headers = {"Cache-Control": "no-store"}
                if kind == "html" or p.suffix.lower() == ".svg":
                    # a plot page runs its own scripts, but in an opaque origin
                    # that cannot reach this API or the session cookie
                    headers["Content-Security-Policy"] = "sandbox allow-scripts"
                return self._send(200, p.read_bytes(), ctype, headers)
            if route == "plot":
                # mace plotting as a cancelable "plot" task; a script's GET waits for it (as before),
                # the page passes detach=1 and follows the task
                p = st.resolve(self._q(query, "path"))
                kind = self._q(query, "kind") or plots.kind_of(p) or classify(p.name)
                if kind not in plots.KINDS:
                    return self._error(400, f"{p.name} is not a BAND, DOSS, IRSPEC or RAMSPEC data file")
                fmt = self._q(query, "format", "") or ""
                if fmt and fmt not in plots.FORMATS:
                    raise ValueError(f"format is one of {', '.join(plots.FORMATS)}")
                t = cli_task(st, "plot", f"Plot {p.name} with mace plotting",
                             lambda ctx: plots.render(p, kind, st.disk, ctx, fmt),
                             client=self.headers.get("X-Mace-Client"))
                return gui_tasks.answer(self, st.tasks, t, 0.3 if self._q(query, "detach") == "1" else 600)
            if route == "spectrum":
                p = st.resolve(self._q(query, "path"))
                if self._q(query, "source") == "modes":
                    return _spectrum_from_modes(self, p, query)
                key = gui_cache.file_key(p, ("spectrum", self._q(query, "average") == "1"))
                data = st.parse_cache.get(key)
                if data is None:
                    avg = self._q(query, "average") == "1"
                    data = _stable_parse(p, lambda: gui_spectra.average(p) if avg else gui_spectra.read(p))
                    st.parse_cache.put(key, data, 16 * len(data["x"]) * (1 + len(data["columns"])))
                    st.ledger.enforce()
                # the FREQ .out of the same run beside it: its modes are the peaks (read now when cold)
                out = _spectrum_out(p)
                return self._json({**data, "path": st.rel(p), "peaks": _spectrum_peaks(st, p, client=self.headers.get("X-Mace-Client")),
                                   "out": st.rel(out) if out is not None else None})
            if route == "freq":
                return _freq_route(self, query)
            if route == "freq/mode":
                p = st.resolve(self._q(query, "path"))
                parsed = st.parse_cache.get(gui_cache.file_key(p, ("freq", outindex.VERSION)))
                if parsed is None:
                    return self._error(409, "read the modes first (api/freq)")
                return self._json(gui_freq.mode_payload(parsed, int(self._q(query, "k", "1"))))
            if route == "convergence":
                # several runs' energies per step, for Compare Convergence (plan E12 fallback)
                paths = query.get("path") or []
                if not 2 <= len(paths) <= 24:
                    raise ValueError("compare 2 to 24 outputs")
                runs = []
                for rel in paths:
                    p = st.resolve(rel)
                    idx = _out_index(st, p)
                    if not idx.n_steps:
                        raise filekind.NoGeometry()
                    n = len(idx.block_structure(-1)["numbers"])
                    runs.append({"path": st.rel(p), "name": p.name, "energies": idx.energies(), "labels": idx.labels,
                                 "n_atoms": n, "status": idx.status, "dim": idx.dim})
                return self._json({"runs": runs})
            if route == "map":
                from . import maps
                p = st.resolve(self._q(query, "path"))
                if not p.is_file():
                    raise FileNotFoundError(st.rel(p))
                return self._json({**maps.map_payload(p), "path": st.rel(p)})
            if route == "transport":
                p = st.resolve(self._q(query, "path"))
                if not p.is_file():
                    raise FileNotFoundError(st.rel(p))
                data = results.transport_data(p)
                data["path"] = st.rel(p)
                data["siblings"] = {k: r for k, v in data["siblings"].items()
                                    if (r := filekind.rel_inside(st, p.with_name(v))) is not None}
                return self._json(data)
            if route in ("bands", "dos"):
                p = st.resolve(self._q(query, "path"))
                kind = "band" if route == "bands" else "dos"
                data = _stable_parse(p, lambda: results.band_data(p) if kind == "band" else results.dos_data(p))
                other = results.partner(p, kind)
                data["path"] = st.rel(p)
                data["partner"] = st.rel(other) if other else None
                return self._json(data)
            if route == "plot-image":
                f = plots.cached_file(st.disk.root, self._q(query, "key", ""),
                                      self._q(query, "name", ""))
                if f is None:
                    return self._error(404, "no such plot")
                ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
                headers = {"Cache-Control": "private, max-age=600"}
                if f.suffix == ".svg":
                    headers["Content-Security-Policy"] = "sandbox"
                return self._send(200, f.read_bytes(), ctype, headers)
            if route == "db":
                p = st.resolve(self._q(query, "path"))
                _not_while_building(p)
                return self._json(gui_db.summary(p))
            if route == "db/material":
                p = st.resolve(self._q(query, "path"))
                _not_while_building(p)
                out = gui_db.material(p, self._q(query, "id", ""))
                for c in out["calculations"]:
                    c["input_rel"] = _db_file(st, p, c.get("input_file"), c.get("work_dir"))
                    c["output_rel"] = _db_file(st, p, c.get("output_file"), c.get("work_dir"))
                return self._json(out)
            if route == "history":
                # File History of one file of this folder (entries of other folders are never listed)
                p = st.resolve(self._q(query, "path"), must_exist=False)
                rel = st.rel_new(p) if p != st.root else ""
                if not rel:
                    raise ValueError("a file of the served folder")
                return self._json({"path": rel, "entries": gui_history.entries(rel), "file": saving.file_stamp(p),
                                   "protected": saving.protected_reason(p), "note": gui_history.over_cap_note()})
            if route == "write/check":
                # Save As asks before it closes: a protected name is refused in the dialog, never offered "overwrite?"
                p = st.resolve(self._q(query, "path"), must_exist=False)
                try:
                    st.check_write(p)
                    why = None
                except saving.ProtectedOutput as exc:
                    why = str(exc)
                return self._json({"path": st.rel_new(p), "exists": p.exists(), "protected": why})
            if route == "history/file":
                rec = gui_history.entry_record(self._q(query, "id", ""))
                data = gui_history.entry_path(rec["id"]).read_bytes()
                return self._send(200, data, "text/plain; charset=utf-8",
                                  {"Cache-Control": "no-store", "X-Mace-Rel-Path": url_quote(rec["rel_path"])})
            if route == "db/stamp":
                # stat only (the DB, -wal, -journal): the Database view's live refresh polls it
                p = st.resolve(self._q(query, "path"))
                return self._json({**gui_db.db_stamp(p), "building": dbwriters.owner(p)})
            if route == "db/build/info":
                return _db_build_info(self, query)
            if route in ("db/page", "db/count", "db/stats", "db/props", "db/export"):
                _not_while_building(st.resolve(self._q(query, "path")))
                return _db_route(self, route, query)
            if route == "edit/state":
                sess = st.sessions.get(self._q(query, "session"))
                with sess.lock:
                    return self._json(sess.payload())
            if route == "templates":
                return self._json({"templates": decks.templates()})
            if route == "setup/schema":
                return self._json(gui_setup.schema())
            if route in submit_routes.GET_ROUTES:
                return submit_routes.handle(self, "GET", route, query)
            if route.startswith("test/") and st.test_hooks:
                if route == "test/gate":
                    return self._json(compute.GATE.stats())
                if route == "test/log":
                    return self._json({"log": st.test_log})
                if route == "test/dblog":
                    return self._json({"log": list(gui_db.QUERY_LOG), "immutable_opens": list(gui_db.IMMUTABLE_OPENS)})
                if route == "test/sessions":
                    return self._json({"ids": st.sessions.ids()})
                if route == "test/parse_tmp":
                    d = st.cache / "parse"
                    return self._json({"files": sorted(x.name for x in d.iterdir()) if d.is_dir() else []})
                if route == "test/held":
                    return self._json({**st.held.report(), "parse": st.parse_cache.report()})
                if route == "test/mlip":
                    return self._json(st.mlip_host.report())
                if route == "test/pool":
                    # never the environment itself: the allowlist, and a one-way
                    # fingerprint for the "same environment as the server" check
                    import sys as _sys
                    from . import pool_worker
                    mace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    p = st.tasks.pool_if_started()
                    return self._json({"pool": p.report() if p else None, "server_pid": os.getpid(),
                                       "budget": compute.GATE.budget(), "env": env_report(),
                                       "env_fingerprint": env_fingerprint(skip=pool_worker.ENV_SET_BY_INIT),
                                       "mace_dir_on_sys_path": any(os.path.abspath(x or os.curdir) == mace_dir
                                                                   for x in _sys.path)})
            return self._error(404, f"unknown API route {route}")
        except gui_db.DbBusy as exc:
            return self._db_busy(exc)
        except StillWriting as exc:
            return self._json({"error": str(exc), "still_writing": True, "age_s": exc.age_s}, 409)
        except ImportError as exc:
            return self._missing(exc)
        except PermissionError as exc:
            return self._error(403, str(exc))
        except FileNotFoundError as exc:
            return self._error(404, f"not found: {exc}")
        except KeyError as exc:
            return self._error(404, f"not found: {exc}")
        except filekind.KindConflict as exc:
            return self._error(409, str(exc))
        except (ValueError, decks.DeckError) as exc:
            return self._error(400, str(exc))

    def _events(self, query):
        """Server-Sent Events: a ``hello`` at once (with this connection's
        client id), then a heartbeat every HEARTBEAT_S seconds (``?interval=``
        overrides it, 0.2 to 60 s, for tests), and ``task`` events as tasks
        change. ``retry: 2000`` makes a dropped stream reconnect in 2 s.
        Nothing polls."""
        try:
            interval = float(self._q(query, "interval", HEARTBEAT_S))
        except ValueError:
            interval = HEARTBEAT_S
        interval = min(max(interval, 0.2), 60.0) if interval == interval else HEARTBEAT_S
        # an authenticated stream that sends its own heartbeat: no idle timeout
        # for this socket (HANDLER_TIMEOUT_S is for sockets that send nothing)
        self.connection.settimeout(None)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        st = self.state
        n = 0
        client = secrets.token_hex(8)
        sid = None if self.headers.get("X-Mace-Token") is not None else self._token_from_cookie()
        q = st.tasks.subscribe()
        st.clients[client] = {"since": time.time()}
        st.held.connected(client)
        drop_gen = getattr(st, "drop_gen", 0)
        try:
            payload = json.dumps({"version": st.version, "interval": interval, "client": client, "gen": st.served.gen})
            self.wfile.write(f"retry: 2000\nevent: hello\ndata: {payload}\n\n".encode())
            self.wfile.flush()
            next_beat = time.monotonic() + interval
            while not st.stopping.is_set():
                if getattr(st, "drop_gen", 0) != drop_gen:
                    break                                  # test hook: the stream drops
                if sid is not None and st.web.state_of(sid) != "ok":
                    # signed out (expired or revoked): say so once, then end the stream
                    self.wfile.write(b"event: expired\ndata: {}\n\n")
                    self.wfile.flush()
                    break
                try:
                    event, data = q.get(timeout=max(0.0, min(0.5, next_beat - time.monotonic())))
                    try:
                        self.wfile.write(f"event: {event}\ndata: {json.dumps(data, default=_json_default)}\n\n"
                                         .encode())
                        self.wfile.flush()
                    finally:
                        q.task_done()                      # sent (a stop waits for it: TaskManager.delivered)
                except gui_tasks._stdlib_queue.Empty:
                    pass
                if time.monotonic() >= next_beat:
                    n += 1
                    self.wfile.write(f"event: heartbeat\ndata: {{\"n\": {n}, \"t\": {time.time():.3f}}}\n\n"
                                     .encode())
                    self.wfile.flush()
                    next_beat += interval
        finally:
            st.tasks.unsubscribe(q)
            st.clients.pop(client, None)
            st.held.disconnected(client)             # its held structures unpin after the grace period

    def _parse(self, query):
        """POST /api/parse: a dropped file or pasted text (application/
        octet-stream, X-File-Name) read without writing into --dir: streamed
        to a temporary file in the server's cache folder (1 MB chunks, at
        most 50 MB, deleted at once), sniffed, read, and opened as a session
        (3D, up to 5000 atoms) or a held view-only structure."""
        from urllib.parse import unquote
        st = self.state
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        if length > MAX_PARSE:
            self.close_connection = True              # the body is not read
            return self._error(413, f"This file is over {MAX_PARSE // 2**20} MB; open it from the folder instead")
        if length <= 0:
            return self._error(400, "an empty file")
        name = Path(unquote(self.headers.get("X-File-Name") or "pasted")).name[:120] or "pasted"
        label = unquote(self.headers.get("X-Mace-Label") or f"Dropped {name}")[:120]
        box = self._q(query, "box", "1") != "0"
        d = st.cache / "parse"
        d.mkdir(exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix="drop_", suffix="_" + "".join(c for c in name if c.isalnum() or c in "._-")[-40:], dir=str(d))
        try:
            left, head = length, b""
            with os.fdopen(fd, "wb") as fh:
                while left > 0:
                    chunk = self.rfile.read(min(1 << 20, left))
                    if not chunk:
                        break
                    if len(head) < 8192:
                        head += chunk[:8192 - len(head)]
                    fh.write(chunk)
                    left -= len(chunk)
            fmt = structures.sniff(name, head)
            if fmt is None:
                return self._json({"error": f"MACE can't tell what format {name} is. Open CIF, POSCAR, XYZ, extXYZ, d12 or out files",
                                   "unknown_format": True}, 400)
            extra = None
            if fmt in ("d12", "cif"):
                # what the file says for Clean (deck records, a pre-optimization marker); never the sidecar
                fm = _file_meta(Path(tmp)) or {}
                extra = {k: fm[k] for k in ("deck", "preopt") if k in fm} or None
                if extra and "deck" in extra:
                    extra["deck"]["name"] = name
            try:
                data = structures.read_any(Path(tmp), fmt, box=box)
            except ImportError as exc:
                return self._missing(exc)
            except Exception as exc:              # the reader's own words, first line
                msg = (str(exc).strip().splitlines() or [""])[0] or "the text is incomplete or malformed"
                return self._json({"error": f"Couldn't read {name} as {structures.FORMAT_NAMES.get(fmt, fmt)}: {msg}",
                                   "format": fmt}, 400)
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        return self._json(_open_parsed(st, data, name, label, fmt, self.headers.get("X-Mace-Client"), extra))

    def _structure(self, query):
        """/api/structure through the parse cache (spec section 8.4.2): the
        ETag is the file's (path, mtime, size) key, so a revalidation costs a
        stat; ``packed=1`` sends base64 typed arrays (section 8.4.5)."""
        st = self.state
        t0 = time.perf_counter()
        p = st.resolve(self._q(query, "path"))
        frame = int(self._q(query, "frame", "-1"))
        packed = self._q(query, "packed") == "1"
        # the parsed payload and each encoded body under their own keys
        base_key = gui_cache.file_key(p, ("structure", frame))
        key = gui_cache.file_key(p, ("body", frame, "packed" if packed else "json"))
        if key[1:3] != base_key[1:3]:
            base_key = key[:4] + ("structure", frame)      # the file changed between the two stats
        etag = gui_cache.etag_of(key)
        if etag in (self.headers.get("If-None-Match") or ""):
            dur = (time.perf_counter() - t0) * 1000
            return self._send(304, b"", "application/json", {"ETag": etag, "Cache-Control": "no-cache",
                                                              "Server-Timing": f"cache;desc=hit, parse;dur={dur:.2f}"})
        body = st.parse_cache.get(key)
        hit = body is not None
        if not hit:
            data = st.parse_cache.get(base_key)
            if data is None:
                # the stamp is the key's stat: a change after it is seen at save
                stamp = {"mtime_ns": str(base_key[1]), "size": base_key[2]}
                data = _read_structure(st, p, frame)
                data["path"] = st.rel(p)
                data["file"] = stamp
                try:
                    # what the structure looks like (plan E10): a molecule, slab or polymer in a box
                    gap = float(gui_settings.load().get("structure.vacuumGap") or structures.DIM_GAP)
                    data["dim_hint"] = structures.dimensionality_hint(data["cell"], data["cart"], data["pbc"], gap)
                except Exception:
                    pass
                meta = _file_meta(p)
                if meta:
                    data["doc_meta"] = meta
                raw = json.dumps(data, default=_json_default).encode()
                st.parse_cache.put(base_key, data, 2 * len(raw))
            body = json.dumps(gui_cache.pack(data) if packed else data, default=_json_default).encode()
            st.parse_cache.put(key, body, len(body))
            st.ledger.enforce()
        dur = (time.perf_counter() - t0) * 1000
        return self._send_body(body, etag, f"cache;desc={'hit' if hit else 'miss'}, parse;dur={dur:.2f}")

    def _out_summary(self, query):
        """GET /api/out/summary: a run's energies, labels and status, its first
        and last frames and their comparison (plan B6); other steps come from
        /api/out/step. Cached with the file's ETag like /api/structure."""
        st = self.state
        t0 = time.perf_counter()
        p = st.resolve(self._q(query, "path"))
        key = gui_cache.file_key(p, ("outsummary", outindex.VERSION))
        etag = gui_cache.etag_of(key)
        if etag in (self.headers.get("If-None-Match") or ""):
            return self._send(304, b"", "application/json", {"ETag": etag, "Cache-Control": "no-cache"})
        body = st.parse_cache.get(key)
        hit = body is not None
        if not hit:
            idx = _out_index(st, p)
            data = outindex.summary(idx, st.rel(p))
            data["file"] = {"mtime_ns": str(key[1]), "size": key[2]}
            body = json.dumps(data, default=_json_default).encode()
            st.parse_cache.put(key, body, len(body))
            st.ledger.enforce()
        dur = (time.perf_counter() - t0) * 1000
        return self._send_body(body, etag, f"cache;desc={'hit' if hit else 'miss'}, parse;dur={dur:.2f}")

    def _send_body(self, body: bytes, etag: Optional[str] = None, timing: str = "", status: int = 200):
        headers = {"Cache-Control": "no-cache" if etag else "no-store"}
        if etag:
            headers["ETag"] = etag
        if timing:
            headers["Server-Timing"] = timing
        if len(body) > 4096 and "gzip" in self.headers.get("Accept-Encoding", ""):
            body = gzip.compress(body, 5)
            headers["Content-Encoding"] = "gzip"
        self._send(status, body, "application/json", headers)

    def _resources(self) -> Dict[str, Any]:
        st = self.state
        p = st.tasks.pool_if_started()
        out = {**resources.report(gui_settings.load(), st.cache_used()),
               "stores": st.ledger.report(), "compute": compute.GATE.stats(), "budget": compute.GATE.budget(),
               "pool": p.report() if p is not None else {"workers": 0, "alive": 0, "busy": 0, "started": False,
                                                          "peak_rss_mb": 0.0, "workers_used": 0},
               "tasks": st.tasks.report()}
        out["pool_started"] = bool(p is not None and p.ex is not None)
        return out

    def _info(self) -> Dict[str, Any]:
        from mace import __version__
        st = self.state
        from .paths import root_aliases
        return {"version": __version__, "root": str(st.root), "home": _home(), "aliases": root_aliases(),
                "hostname": socket.gethostname(),
                "docs": str(Path(__file__).resolve().parents[2] / "docs" / "gui.md"),
                "plotly": plotly_js() is not None, "static": st.version,
                "heartbeat": HEARTBEAT_S, "tunnel": getattr(st, "tunnel", ""), "dir_hash": st.dir_hash,
                "connect": _connect_now(st), "vscode": bool(getattr(st, "vscode", False)),
                "node": getattr(st, "node", ""), "managed": st.managed is not None,
                "root_gen": st.root_gen, "open_folder": st.allowed_roots is not None,
                "stop_pending": st.stop_pending, "session": self._session_info(),
                "cpu_limit_s": _cpu_limit(), "cpu": getattr(st, "cpu_state", None),
                "slurm": getattr(st, "slurm_state", None) if os.environ.get("MACE_GUI_SLURM") else None,
                "roots": _roots_list(st), "start_dir": str(st.start_dir),
                "resources": resources.report(gui_settings.load(), st.cache_used())}

    def _read_json_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_POST:
            raise ValueError("request body missing or too large")
        data = json.loads(self.rfile.read(length))
        if not isinstance(data, dict):
            raise ValueError("request body must be a JSON object")
        return data

    def _edit(self, route, body):
        return _edit_route(self, route, body)

    def _api_post(self, route: str):
        st = self.state
        try:
            body = self._read_json_body()
            if route in ("settings", "profile", "docprefs") and st.test_hooks and st.fail_pref_saves:
                return self._error(503, "profile busy, try again (test hook)")
            if route.startswith("cluster/"):
                from . import cluster_api
                return cluster_api.post(self, route[len("cluster/"):], body)
            if route == "settings":
                # per key: the good ones are stored, a bad one is named and keeps its old value
                with _profile_writes:
                    where, saved, rejected, notes = gui_settings.save(body.get("settings"))
                st.web.forget_policy()
                st.set_limit(cache_limit_bytes())
                compute.GATE.resize(resources.effective_workers(gui_settings.load().get("max_workers")))
                return self._json({"path": str(where), "saved": saved, "rejected": rejected, "notes": notes})
            if route == "profile":
                # per-key patches: two servers (or tabs) saving at once keep both changes
                with _profile_writes:
                    prof, rejected = gui_settings.patch(body)
                st.set_limit(cache_limit_bytes())
                compute.GATE.resize(resources.effective_workers(prof["settings"].get("max_workers")))
                return self._json({**_profile_view(st), "rejected": rejected})
            if route == "profile/reset":
                # Settings > Advanced > Reset: a copy of each file first (the newest 3 are kept)
                secs = body.get("sections") or []
                with _profile_writes:
                    if body.get("docprefs"):
                        gui_docprefs.reset()
                    prof = gui_settings.reset_sections(secs) if secs else gui_settings.load_profile()
                return self._json({**_profile_view(st), "reset": secs, "docprefs_reset": bool(body.get("docprefs"))})
            if route == "profile/docprefs":
                # the per-document part of an imported profile file
                with _profile_writes:
                    n = gui_docprefs.import_all(body.get("kinds"))
                return self._json({"imported": n})
            if route == "docprefs":
                kind = body.get("kind")
                rel = _docprefs_rel(st, body.get("path"), write=True)
                with _profile_writes:
                    rev = gui_docprefs.put(kind, st.dir_hash, rel, body.get("values") or {})
                return self._json({"kind": kind, "path": rel, "rev": rev, "values": gui_docprefs.get(kind, st.dir_hash, rel)})
            if route in ("setup/run", "setup/defaults", "setup/apply", "setup/preview"):
                return _setup_run_route(self, route, body)
            if route == "setup/save":
                return _setup_save_route(self, body)
            if route == "setup/preflight":
                return _preflight_route(self, body)
            if route in ("volume/combine", "volume/mep_range", "volume/save", "volume/check"):
                return _volume_post(self, route, body)
            if route == "text/find":
                p = st.resolve(body.get("path"))
                if not p.is_file():
                    raise ValueError("not a file")
                q, case = str(body.get("query") or ""), bool(body.get("case"))
                if not q or len(q) > 200:
                    raise ValueError("search for 1 to 200 characters")
                t = cli_task(st, "find", f"Find “{q[:40]}” in {p.name}", lambda ctx: {**textwin.find(p, q, case, ctx), "path": st.rel(p)},
                             doc=str(body.get("doc") or "") or None, client=self.headers.get("X-Mace-Client"),
                             supersede=("find", str(body.get("doc") or p)))
                return gui_tasks.answer(self, st.tasks, t, 0.3)
            if route == "deck/preview":
                # the Phase 2 route: a whole JSON object -> the deck, now through the same CLI run as Setup
                data = body.get("json")
                if not isinstance(data, dict):
                    raise ValueError("json must be the settings object")
                mode = body.get("mode", "cif2d12")
                if mode not in ("cif2d12", "opt2d12"):
                    raise ValueError("mode is cif2d12 or opt2d12")
                calc = (data.get("configuration", data) or {}).get("calculation_type") if mode == "cif2d12" else None
                b = {"source": {"path": body.get("structure")}, "mode": mode, "json_text": gui_setup.dumps_like_cli(data),
                     "form": False, "preview": True,
                     "calc_type": calc if calc in gui_setup.CALC_TYPES[mode] else gui_setup.DEFAULT_CALC[mode]}
                if mode == "opt2d12":
                    b["calc_type"] = str((data.get("configuration", data) or {}).get("calculation_type") or "SP").upper()
                    if b["calc_type"] not in gui_setup.CALC_TYPES["opt2d12"]:
                        b["calc_type"] = "SP"
                res = _setup_run_sync(self, b)
                if res.get("deck") is None:
                    raise decks.DeckError("the CLI wrote no deck:\n" + (res.get("refused") or res.get("log") or "")[-3000:])
                return self._json({"mode": mode, "deck_name": res["deck_name"], "deck": res["deck"], "log": res.get("log", ""),
                                   "command": decks.command_line(mode, "<saved>.json", Path(str(body.get("structure"))))})
            if route == "history/restore":
                return _history_restore(self, body)
            if route == "db/build/preview":
                return _db_build_preview(self, body)
            if route == "db/build":
                return _db_build_route(self, body)
            if route == "out/track":
                return _track_route(self, body)
            if route == "deck/save":
                structure = body.get("structure")
                mode = body.get("mode", "cif2d12")
                name = str(body.get("name", "") or "")
                if name and not any(c in name for c in "/\\") and not name.startswith("."):
                    # a plain name: the protected-kind and link rules; anything else is decks.save_json's 400
                    st.check_write(st.root / name)
                target = decks.save_json(st.root, body.get("name", ""), mode,
                                         body.get("json"), bool(body.get("overwrite")),
                                         expected=body.get("expected") if isinstance(body.get("expected"), dict) else None)
                cmd = None
                if structure:
                    cmd = decks.command_line(mode, target.name,
                                             Path(st.rel(st.resolve(structure))))
                return self._json({"saved": st.rel(target), "command": cmd})
            if route.startswith("edit/"):
                return self._edit(route[5:], body)
            if route.startswith("tasks/"):
                tid, _, action = route[len("tasks/"):].partition("/")
                if action == "cancel":
                    return self._json({"task": st.tasks.cancel(tid)})
                if action == "retry":
                    return gui_tasks.answer(self, st.tasks, st.tasks.retry(tid), 0)
                if action == "remove":
                    return self._json({"removed": st.tasks.remove(tid)})
                return self._error(404, f"unknown task action {action}")
            if route == "symmetry":
                return _symmetry_route(self, body)
            if route.startswith("build/"):
                return _build_route(self, route[len("build/"):], body)
            if route == "symmetry/compute":
                return _symmetry_compute_route(self, body)
            if route == "defect/info":
                return _defect_info_route(self, body)
            if route == "symmetrize/sweep":
                return _sweep_route(self, body)
            if route == "clean":
                return _clean_route(self, body)
            if route == "clean/methods/recheck":
                _detect_clean_methods(st, recheck=True)
                with st.clean_methods_lock:
                    return self._json(dict(st.clean_methods))
            if route == "structure/save":
                return _save_route(self, body)
            if route == "structure/export":
                return _export_route(self, body)
            if route == "held/release":
                return self._json({"released": st.held.release([str(x) for x in (body.get("ids") or [])])})
            if route == "held/claim":
                got = st.held.claim([str(x) for x in (body.get("ids") or [])],
                                    self.headers.get("X-Mace-Client") or body.get("client"))
                return self._json({"claimed": got})
            if route == "open-folder":
                return self._json(_open_folder(st, str(body.get("path") or "")))
            if route == "diagnostics":
                from . import diagnostics
                if st.test_hooks and os.environ.get("MACE_GUI_TEST_DIAG_DELAY"):
                    time.sleep(min(10.0, float(os.environ["MACE_GUI_TEST_DIAG_DELAY"])))
                client = body.get("client") if isinstance(body.get("client"), dict) else None
                return self._json(diagnostics.for_state(st, client))
            if route == "session/touch":
                st.web.touch(self._token_from_cookie())
                return self._json({"ok": True})
            if route == "login-code":
                if not st.web.token_ok(self.headers.get("X-Mace-Token")):
                    return self._error(403, "a new sign-in link needs the token (X-Mace-Token)")
                return self._json({"code": st.web.mint_code()})
            if route == "shutdown":
                return _shutdown_route(self, str(body.get("mode") or "now"))
            if route.startswith("managed/") and st.managed is not None:
                wid = str(body.get("watcher") or "")
                op = route[len("managed/"):]
                if op == "attach":
                    return self._json(st.managed.attach(wid))
                if op == "ping":
                    return self._json(st.managed.ping(wid))
                if op == "detach":
                    return self._json(st.managed.detach(wid))
                if op == "stop":
                    return self._json(st.managed.stop_request(wid))
            if route in submit_routes.POST_ROUTES:
                return submit_routes.handle(self, "POST", route, body)
            if route.startswith("test/") and st.test_hooks:
                if route == "test/parse_cache/clear":
                    with st.parse_cache.lock:
                        st.parse_cache.items.clear()
                        st.parse_cache.bytes = 0
                    return self._json({"cleared": True})
                if route == "test/evict":
                    return self._json({"evicted": st.sessions.evict(str(body.get("session", "")))})
                if route == "test/gate/reset":
                    compute.GATE.reset_stats()
                    st.test_log.clear()
                    return self._json(compute.GATE.stats())
                if route == "test/task":
                    # a slow job in the pool (or inline in the dispatcher thread), with progress
                    payload = {k: body[k] for k in ("seconds", "steps", "fail") if k in body}
                    t = st.tasks.submit("test", str(body.get("title") or "Test task"), str(body.get("job") or "sleep"),
                                        payload, client=self.headers.get("X-Mace-Client"))
                    return gui_tasks.answer(self, st.tasks, t, float(body.get("wait", gui_tasks.WAIT_S)))
                if route == "test/profile/fail":
                    st.fail_pref_saves = bool(body.get("on"))
                    return self._json({"on": st.fail_pref_saves})
                if route == "test/pool/config":
                    p = st.tasks.pool
                    if "idle_s" in body:
                        p.idle_s = float(body["idle_s"])
                    return self._json(p.report())
                if route == "test/held/config":
                    if "grace_s" in body:
                        st.held.grace_s = float(body["grace_s"])
                    return self._json(st.held.report())
                if route == "test/held/put":
                    # a view-only structure of n atoms (MgO-like), held for this page's client
                    import numpy as np
                    n = max(8, int(body.get("atoms", 6000)) // 8 * 8)
                    k = int(round((n / 8) ** (1 / 3)))
                    cells = max(1, n // 8)
                    frac = (np.random.default_rng(0).random((cells * 8, 3)))
                    pl = structures._payload(np.eye(3) * 4.212 * max(1, k), frac, frac @ (np.eye(3) * 4.212 * max(1, k)),
                                             [12, 8] * (cells * 4), (True, True, True), {"kind": "test", "path": "held"},
                                             symmetry_data={"available": False, "note": "test"})
                    packed = gui_cache.pack(pl)
                    hid = st.held.put(packed, len(json.dumps(packed)) + frac.nbytes * 3,
                                      self.headers.get("X-Mace-Client") or body.get("client"))
                    st.ledger.enforce()
                    return self._json({"held": hid, "n": cells * 8, "report": st.held.report()})
                if route == "test/cli":
                    # a fake slow CLI child (a sleeping python), for the cancel and supersede tests
                    secs = min(60.0, max(0.0, float(body.get("seconds", 30))))
                    doc = str(body.get("doc") or "") or None
                    argv = [__import__("sys").executable, "-c",
                            "import time, subprocess, sys\n"
                            "kid = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(%s)'])\n"
                            "time.sleep(%s)\n" % (secs, secs)]
                    t = cli_task(st, str(body.get("kind") or "deck"), str(body.get("title") or "Fake CLI"),
                                 lambda ctx: {"code": clirun.run(argv, st.cache, ctx, timeout=120).returncode},
                                 doc=doc, client=self.headers.get("X-Mace-Client"),
                                 supersede=("deck", doc) if doc else None)
                    return gui_tasks.answer(self, st.tasks, t, float(body.get("wait", 0)))
                if route == "test/burn-server":
                    # spend CPU time in the server process itself (the CPU-limit tests)
                    end = time.process_time() + min(30.0, float(body.get("seconds", 1)))
                    while time.process_time() < end:
                        sum(range(20000))
                    return self._json({"cpu": time.process_time()})
                if route == "test/root_gen":
                    # a new folder generation without switching the folder (as Open Folder in
                    # another tab leaves this page): the submit tests' stale-page case
                    with st.root_lock:
                        st.served = Served(st.served.root, st.served.gen + 1)
                    st.snapshot()
                    return self._json({"root_gen": st.root_gen})
                if route == "test/sse/drop":
                    st.drop_gen = getattr(st, "drop_gen", 0) + 1
                    return self._json({"dropped": len(st.clients)})
            return self._error(404, f"unknown API route {route}")
        except gui_db.DbBusy as exc:
            return self._db_busy(exc)
        except dbwriters.WriterBusy as exc:
            return self._json({"error": str(exc), "writer": exc.owner}, 409)
        except ImportError as exc:
            return self._missing(exc)
        except editor.Busy as exc:
            return self._error(409, str(exc))
        except editor.StaleRev as exc:
            return self._json({"conflict": "rev", "rev": exc.rev, "error": str(exc) or "the structure changed since"}, 409)
        except gui_cache.HeldBudget as exc:
            return self._error(400, str(exc))
        except editor.NeedsConfirm as exc:
            return self._json({"needs_confirm": True, "message": str(exc), **exc.details})
        except saving.SaveConflict as exc:
            return self._json({"error": str(exc), **exc.details}, 409)
        except saving.ProtectedOutput as exc:
            return self._json({"error": str(exc), "protected": True}, 409)
        except gui_setup.StillRunning as exc:
            return self._json({"error": str(exc), "still_running": True}, 409)
        except gui_history.HistoryUnavailable as exc:
            return self._json({"error": str(exc), "history_unavailable": True}, 409)
        except filekind.KindConflict as exc:
            return self._error(409, str(exc))
        except gui_clean.NothingToClean as exc:
            return self._json({"nothing_to_clean": True, "message": str(exc)})
        except prefstore.Busy as exc:
            return self._error(503, str(exc))
        except prefstore.Conflict as exc:
            return self._error(409, str(exc))
        except PermissionError as exc:
            return self._error(403, str(exc))
        except (FileNotFoundError, KeyError) as exc:
            return self._error(404, f"not found: {exc}")
        except json.JSONDecodeError as exc:
            return self._error(400, f"bad JSON: {exc}")
        except (ValueError, decks.DeckError, gui_setup.SetupError) as exc:
            return self._error(400, str(exc))


WATCHDOG_S = 2.0


def start_watchdog(st) -> threading.Thread:
    """One server-wide 2 s watchdog: Stop When Tasks Finish, and the hooks
    other features add (session idle stop, CPU-time limits)."""
    def loop():
        while not st.stopping.wait(WATCHDOG_S):
            try:
                if st.stop_pending is not None and st.tasks.running_count() == 0:
                    _stop_from_browser(st)
                for hook in list(st.watchdog_hooks):
                    hook(st)
            except Exception:              # the watchdog never dies
                if os.environ.get("MACE_GUI_TEST_HOOKS") == "1":
                    traceback.print_exc()
    for hook in (_idle_hook, _cpu_hook, _walltime_hook):
        if hook not in st.watchdog_hooks:
            st.watchdog_hooks.append(hook)
    t = threading.Thread(target=loop, name="mace-gui-watchdog", daemon=True)
    t.start()
    return t


def _idle_hook(st) -> None:
    """The idle stop (server.idleStopHours; off by default): N hours without
    input from any session and no task running. A warning 5 minutes before."""
    from .sessions import IDLE_WARN_S
    due, left = st.web.idle_stop_due(st.tasks.running_count())
    warned = getattr(st, "_idle_warned", False)
    if left is None:
        if warned:
            st._idle_warned = False
            st.tasks.broadcast("idle-warning", {"cleared": True})
        return
    if due and st.request_stop is not None:
        hours = st.web.policy().get("idle_stop_hours")
        st.stop_detail = {"hours": f"{hours:g} hour{'s' if hours != 1 else ''}"}
        st.tasks.broadcast("server-stopping", {"reason": "idle"})
        st.request_stop("idle")
    elif left <= IDLE_WARN_S and not warned:
        st._idle_warned = True
        st.tasks.broadcast("idle-warning", {"seconds": round(left)})
    elif left > IDLE_WARN_S and warned:
        st._idle_warned = False
        st.tasks.broadcast("idle-warning", {"cleared": True})


WALLTIME_CHECK_S = 600.0
WALLTIME_WARN_S = (15 * 60, 5 * 60)
WALLTIME_STOP_S = 60.0


def _job_end(job: str) -> Optional[float]:
    """The job's EndTime (epoch) from ``scontrol show job`` through the shared
    cache (key job:<id>); admins can extend a time limit, so it is re-read."""
    from mace.cluster import reader
    from datetime import datetime
    try:
        rec = reader.job_detail(job, refresh=True, tag="gui-walltime").value or {}
        end = rec.get("EndTime") or ""
        return datetime.fromisoformat(end).timestamp() if end and end not in ("Unknown", "None") else None
    except (ValueError, TypeError, OSError):
        return None


def _walltime_hook(st) -> None:
    """mace gui --slurm (MACE_GUI_SLURM = the job id): warnings 15 and 5
    minutes before the job's time limit, a clean stop 60 s before it."""
    job = os.environ.get("MACE_GUI_SLURM", "").strip()
    if not job.isdigit():
        return
    now = time.monotonic()
    every = WALLTIME_CHECK_S
    if st.test_hooks:
        try:
            every = float(os.environ.get("MACE_GUI_TEST_WALLTIME_CHECK_S") or every)
        except ValueError:
            pass
    if now - getattr(st, "_wall_checked", -1e9) >= every:
        st._wall_checked = now
        end = _job_end(job)
        if end is not None:
            st.slurm_end = end
    end = getattr(st, "slurm_end", None)
    st.slurm_state = {"job": job, "node": socket.gethostname().split(".")[0], "ends": end,
                      "left_s": None if end is None else max(0.0, end - time.time())}
    if end is None:
        return
    left = end - time.time()
    warned = getattr(st, "_wall_warned", set())
    st._wall_warned = warned
    if left <= WALLTIME_STOP_S and st.request_stop is not None:
        st.tasks.broadcast("server-stopping", {"reason": "walltime"})
        st.request_stop("walltime")
        return
    for mark in WALLTIME_WARN_S:
        if left <= mark and mark not in warned:
            warned.update(m for m in WALLTIME_WARN_S if m >= mark)
            st.tasks.broadcast("walltime", {"job": job, "left_s": round(left), "ends": end})
            break


CPU_CHECK_S = 60.0


def _cpu_hook(st) -> None:
    """The server's own CPU time against a per-process limit (cpulimit.py):
    a status chip from 50 %, a sticky warning from 80 %, a clean stop at 97 %
    (the kernel's kill would leave the cache folder and the state files)."""
    from . import cpulimit
    lim = cpulimit.limit()
    if lim is None:
        return
    now = time.monotonic()
    fast = st.test_hooks and os.environ.get("MACE_GUI_TEST_CPU_LIMIT_S")
    if not fast and now - getattr(st, "_cpu_checked", 0.0) < cpulimit.check_every(lim, CPU_CHECK_S):
        return
    st._cpu_checked = now
    used = cpulimit.own_used()
    share = used / lim
    level = "stop" if share >= cpulimit.STOP_AT else "warn" if share >= cpulimit.WARN_STICKY else \
        "chip" if share >= cpulimit.WARN_CHIP else ""
    st.cpu_state = {"used_s": round(used, 1), "limit_s": lim, "level": level}
    if level != getattr(st, "_cpu_level", ""):
        st._cpu_level = level
        st.tasks.broadcast("cpu", st.cpu_state)
    if level == "stop" and st.request_stop is not None:
        st.stop_detail = {"minutes": cpulimit.minutes(lim)}
        st.tasks.broadcast("server-stopping", {"reason": "cpu-limit"})
        st.request_stop("cpu-limit")


def _stop_from_browser(st, delay: float = 0.3) -> None:
    """Sign Out and Stop Server: every tab hears it, running tasks are
    canceled, then the server stops through request_stop (after `delay`, so
    the answer to the request that asked for it is sent first)."""
    if st.stopping_browser:
        return
    st.stopping_browser = True
    st.stop_pending = None
    st.web.revoke_all()
    st.tasks.broadcast("server-stopping", {"reason": "browser"})
    canceled = []
    for rec in st.tasks.running():
        try:
            st.tasks.cancel(rec["id"])
            canceled.append(st.tasks.get(rec["id"]))
        except KeyError:
            pass
    if st.request_stop is not None:
        threading.Thread(target=_stop_when_canceled, args=(st, canceled, delay),
                         name="mace-stop-browser", daemon=True).start()


def _stop_when_canceled(st, canceled, delay: float, bound: float = STOP_DRAIN_S) -> None:
    """The stop of Sign Out and Stop Server: after `delay`, and once each
    canceled task has ended and its last event reached every open tab (at
    most `bound` seconds; then the server stops anyway and says so)."""
    time.sleep(delay)
    end = time.monotonic() + bound
    left = [t for t in canceled if not t.done.wait(max(0.0, end - time.monotonic()))]
    sent = st.tasks.delivered(max(0.0, end - time.monotonic()))
    if left or not sent:
        what = (f"{len(left)} canceled task(s) had not stopped" if left
                else "the last events had not reached every open tab")
        try:
            print(f"mace gui: stopping anyway after {bound:g} s: {what}", file=sys.stderr, flush=True)
        except OSError:
            pass
    st.request_stop("browser")


def _shutdown_route(handler, mode: str):
    """POST /api/shutdown {mode: now | after-tasks | cancel} (plan T10)."""
    st = handler.state
    if st.request_stop is None:
        return handler._error(503, "this server cannot be stopped from the browser")
    if mode == "now":
        _stop_from_browser(st)
        cookie = f"{handler._cookie_name()}=; Max-Age=0; HttpOnly; SameSite=Strict; Path={st.prefix}"
        body = json.dumps({"stopping": True}).encode()
        return handler._send(202, body, "application/json", {"Set-Cookie": cookie, "Cache-Control": "no-store"})
    if mode == "after-tasks":
        running = st.tasks.running()
        if not running:
            return _shutdown_route(handler, "now")
        st.stop_pending = {"since": time.time(), "waiting_for": [t["id"] for t in running]}
        st.tasks.broadcast("stop-pending", {"waiting_for": st.stop_pending["waiting_for"],
                                            "titles": [t["title"] for t in running]})
        return handler._json({"pending_stop": True, "waiting_for": st.stop_pending["waiting_for"]}, 202)
    if mode == "cancel":
        if st.stop_pending is None:
            return handler._error(409, "no stop is pending")
        st.stop_pending = None
        st.tasks.broadcast("stop-canceled", {})
        return handler._json({"canceled": True})
    return handler._error(400, "mode is now, after-tasks or cancel")


def _connect_now(st) -> str:
    """The `mace gui connect` line for the folder served now (it follows Open Folder)."""
    base = getattr(st, "connect_base", "")
    if not base:
        return getattr(st, "connect", "")
    import shlex
    return f"{base} --dir {shlex.quote(str(st.root))}"


def _cpu_limit() -> Optional[float]:
    from . import cpulimit
    return cpulimit.limit()


def _roots_list(st) -> List[Dict[str, str]]:
    from . import roots as gui_roots
    return [{"label": gui_roots.root_label(r), "path": str(r)} for r in (st.allowed_roots or [])]


def _check_folder(st, raw: str, for_open: bool) -> Path:
    """An Open Folder target (or a folder to list): realpath-ed, inside an
    allowed root (the start --dir is always allowed), no hidden component
    below the root, not a folder that holds credentials. Raises
    PermissionError / FileNotFoundError with the message to show."""
    from . import roots as gui_roots
    if st.allowed_roots is None:
        raise PermissionError(OFF_REFUSAL)
    if not raw:
        raise FileNotFoundError("no folder given")
    target = Path(os.path.realpath(os.path.expanduser(raw)))
    if target == st.start_dir:
        return target
    root = next((r for r in st.allowed_roots if target == r or r in target.parents), None)
    if root is None:
        names = ", ".join(x["label"] for x in _roots_list(st)) or "none"
        raise PermissionError(OUTSIDE_REFUSAL.format(names=names))
    if gui_roots.hidden_component(target.relative_to(root).parts) or \
            gui_roots.hidden_component(Path(os.path.normpath(os.path.expanduser(raw))).parts[1:]):
        raise PermissionError(HIDDEN_REFUSAL)
    if not target.is_dir():
        raise FileNotFoundError(f"{raw} is not a folder")
    if not os.access(target, os.R_OK | os.X_OK):
        raise PermissionError(f"{target.name} cannot be read")
    if for_open and gui_roots.holds_credentials(target):
        raise PermissionError(HOME_REFUSAL)
    return target


def _folders(st, raw: Optional[str]) -> Dict[str, Any]:
    """GET /api/folders?path=: the subfolders of one folder (no files, no
    hidden folders, at most MAX_FOLDERS), for the Open Folder dialog."""
    roots = _roots_list(st)
    if st.allowed_roots is None:
        raise PermissionError(OFF_REFUSAL)
    path = _check_folder(st, raw or str(st.root), for_open=False)
    dirs, truncated = [], False
    try:
        entries = sorted(os.scandir(path), key=lambda e: e.name.lower())
    except OSError:
        entries = []
    for e in entries:
        if e.name.startswith(".") or e.name in _SKIP_DIRS:
            continue
        try:
            if not e.is_dir():
                continue
        except OSError:
            continue
        if len(dirs) >= MAX_FOLDERS:
            truncated = True
            break
        dirs.append(e.name)
    parent = None
    if not any(str(path) == r["path"] for r in roots) and path != st.start_dir:
        parent = str(path.parent)
    return {"roots": roots, "path": str(path), "parent": parent, "dirs": dirs, "truncated": truncated,
            "current": str(st.root)}


def _open_folder(st, raw: str) -> Dict[str, Any]:
    """POST /api/open-folder {path}: serve another folder (409 while tasks run)."""
    target = _check_folder(st, raw, for_open=True)
    if st.tasks.running_count():
        raise editor.Busy("Cancel or wait for the running tasks first.")
    new = st.set_root(target)
    return {"root": str(new.root), "dir_hash": new.dir_hash, "gen": new.gen}


_SAFE_CIF = __import__("re").compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,120}\.cif$")


def _time_limit(value) -> float:
    """Clean's time limit in seconds, 0.5 to 30 (json.loads accepts NaN)."""
    try:
        t = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"time_limit {value!r} is not a number")
    return min(max(t, 0.5), 30.0) if t == t else 10.0


def _internal(atoms) -> Dict[str, Any]:
    return {"cell": atoms.cell.array, "frac": atoms.get_scaled_positions(wrap=False),
            "cart": atoms.positions, "numbers": [int(n) for n in atoms.numbers],
            "pbc": [True, True, True]}


def _undo_limit() -> int:
    """Settings' undoLimit (200 by default), 1 to 1000 steps."""
    v = gui_settings.load().get("undoLimit", editor.MAX_HISTORY)
    try:
        return min(1000, max(1, int(v)))
    except (TypeError, ValueError, OverflowError):
        return editor.MAX_HISTORY


def _session(atoms, name: str, body: Dict[str, Any], label: str) -> "editor.EditSession":
    """A new edit session with the document's mode: Sym or P1 (``locked``),
    the tolerance and wrapping; none of them is an undo step."""
    return editor.EditSession(atoms, name, float(body.get("symprec", 1e-3)), label=label,
                              locked=bool(body.get("locked", False)), undo_limit=_undo_limit(),
                              wrap=bool(body.get("wrap", False)))


def _test_delay(handler):
    """MACE_GUI_TEST_HOOKS=1 only: X-Mace-Test-Delay-Ms holds this request
    back before it touches its session (to make two edits arrive reversed)."""
    if not handler.state.test_hooks:
        return
    try:
        ms = float(handler.headers.get("X-Mace-Test-Delay-Ms") or 0)
    except ValueError:
        ms = 0
    if ms > 0:
        time.sleep(min(ms, 5000) / 1000)


def _open_session(st, spec: Dict[str, Any]):
    """A session for a document's file (the first edit's ``open`` block, or
    edit/open): the structure as /api/structure gave it, same atom order."""
    p = st.resolve(spec.get("path"))
    stamp = saving.file_stamp(p)
    exp = spec.get("expected")
    if isinstance(exp, dict) and stamp is not None and not saving.same_stamp(stamp, exp):
        raise saving.SaveConflict(f"{p.name} changed on disk since it was opened",
                                  {"conflict": "changed_on_disk", "changed_on_disk": True, "file": stamp})
    data = _read_structure(st, p, int(spec.get("frame", -1)))
    sess = st.sessions.add(_session(editor.atoms_from_payload(data), p.stem, spec, f"Opened {p.name}"))
    meta = _file_meta(p)
    if meta:
        sess.doc_meta = meta
    st.ledger.enforce()
    return sess, p, stamp


def _check_new_name(name: str) -> str:
    if not __import__("re").fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,60}", name):
        raise ValueError("name: letters, digits, _, . and - only")
    return name


def _from_structure(st, body: Dict[str, Any]):
    """A session from a structure the browser holds (a restored draft, or a
    document whose session the server let go): the same checks as edit/new."""
    import numpy as np
    name = _check_new_name(str(body.get("name") or "structure"))
    try:
        cell = np.asarray(body["cell"], float)
        frac = np.asarray(body["frac"], float)
        numbers = [int(n) for n in body["numbers"]]
    except (KeyError, TypeError, ValueError):
        raise ValueError("give cell (3x3), frac (one triple per atom) and numbers")
    if cell.shape != (3, 3) or not np.all(np.isfinite(cell)) or abs(np.linalg.det(cell)) < 1e-6:
        raise ValueError("cell must be three linearly independent vectors")
    if frac.shape != (len(numbers), 3) or not np.all(np.isfinite(frac)):
        raise ValueError("frac must hold one finite triple per atom")
    if len(numbers) > editor.MAX_ATOMS:
        raise ValueError(f"more than {editor.MAX_ATOMS} atoms: too big for the editor")
    if any(not 1 <= n <= 103 for n in numbers):
        raise ValueError("atomic numbers must be 1 to 103")
    from ase import Atoms
    atoms = Atoms(numbers=numbers, cell=cell, scaled_positions=frac, pbc=True)
    label = str(body.get("label") or f"Restored {name}")[:120]
    sess = st.sessions.add(_session(atoms, Path(name).stem if "." in name else name, body, label))
    meta = body.get("meta")
    if isinstance(meta, dict):
        sess.doc_meta = _clean_meta(meta)
    st.ledger.enforce()
    return sess


def _clean_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    """A document's provenance and hints as the page sends them back (a draft): plain values only."""
    out: Dict[str, Any] = {}
    prov = meta.get("provenance")
    if isinstance(prov, dict):
        out["provenance"] = {k: prov[k] for k in ("parent", "parent_sha256", "op", "params", "label")
                             if k in prov and isinstance(prov[k], (str, int, float, list, dict, type(None)))}
    hints = meta.get("hints")
    if isinstance(hints, dict):
        h = {}
        if "charge" in hints:
            try:
                h["charge"] = int(hints["charge"])
            except (TypeError, ValueError):
                pass
        out["hints"] = h
    for k in ("untitled_hint", "name"):
        if isinstance(meta.get(k), str):
            out[k] = meta[k][:200]
    if isinstance(meta.get("deck"), dict):
        from mace.preopt import deckscan
        out["deck"] = deckscan.valid_meta(meta["deck"])
    pre = _valid_preopt(meta.get("preopt"))
    if pre:
        out["preopt"] = pre
    return out


def _valid_preopt(value) -> Optional[Dict[str, Any]]:
    """``doc_meta["preopt"]``: {method_ran, model}, plain short strings only."""
    if not isinstance(value, dict):
        return None
    import re as _re
    m = value.get("method_ran")
    if not isinstance(m, str) or not _re.fullmatch(r"(gfnff|mlip:[A-Za-z0-9._-]{1,60})", m):
        return None
    model = value.get("model")
    return {"method_ran": m, "model": model[:60] if isinstance(model, str) else None}


def _file_meta(p: Path) -> Optional[Dict[str, Any]]:
    """A file's document meta: its sidecar (provenance, hints), and what the
    file itself says for Clean: a deck's geometry-editing and charge records,
    a CIF's FF/ML pre-optimization marker."""
    meta = dict(gui_cache.read_sidecar(p) or {})
    low = p.name.lower()
    try:
        if low.endswith(".d12"):
            from mace.preopt import deckscan
            sc = deckscan.scan(p.read_text(errors="replace"))
            if sc["geometry_edits"] or sc["charge_records"]:
                meta["deck"] = {"geometry_edits": sc["geometry_edits"], "charge_records": sc["charge_records"],
                                "name": p.name}
        elif low.endswith(".cif"):
            from mace.preopt import marker
            info = marker.read(p)
            if marker.is_ff_ml(info):
                pre = _valid_preopt({"method_ran": info.get("method"), "model": info.get("model")})
                if pre:
                    meta["preopt"] = pre
    except OSError:
        pass
    return meta or None


def _edit_route(handler, route: str, body: Dict[str, Any]):
    """The editor API: every edit goes through an EditSession."""
    st = handler.state
    if not st.test_hooks:
        return _edit_route_inner(handler, route, body)
    entry = {"route": route, "op": body.get("op"), "tag": handler.headers.get("X-Mace-Test-Tag"),
             "arrived": time.time()}
    _test_delay(handler)
    try:
        return _edit_route_inner(handler, route, body)
    finally:
        entry["done"] = time.time()
        st.test_log.append(entry)
        del st.test_log[:-500]


def _edit_route_inner(handler, route: str, body: Dict[str, Any]):
    st = handler.state
    if route == "open":
        sess, p, stamp = _open_session(st, body)
        return handler._json(sess.payload(f"editing {p.name}", extra={"file": stamp}))
    if route == "from_structure":
        sess = _from_structure(st, body)
        return handler._json(sess.payload("restored"))
    if route == "close":
        # a closed tab (or a Revert): its undo history is let go of at once
        # rather than when the memory budget needs the room
        return handler._json({"closed": st.sessions.evict(str(body.get("session") or ""))})
    if route == "new":
        atoms = editor.new_cell(body.get("cellpar") or [], body.get("sites") or [],
                                body.get("spacegroup"), int(body.get("setting", 1)))
        name = str(body.get("name") or "new_cell")
        if not __import__("re").fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,60}", name):
            raise ValueError("name: letters, digits, _ and - only")
        sess = st.sessions.add(_session(atoms, name, {k: v for k, v in body.items() if k != "symprec"},
                                        f"New Structure {name}"))
        st.ledger.enforce()
        return handler._json(sess.payload(f"new cell with {len(atoms)} atoms"))
    opened = None
    if route == "op" and not body.get("session") and isinstance(body.get("open"), dict):
        # the document's first edit opens its session in the same request
        sess, p, stamp = _open_session(st, body["open"])
        opened = {"file": stamp, "opened_rev": sess.rev}
    else:
        sess = st.sessions.get(body.get("session"))
    if route == "op":
        args = body.get("args") or {}
        if not isinstance(args, dict):
            raise ValueError("args must be an object")
        label = body.get("label")
        with sess.lock:
            try:
                payload = sess.apply(str(body.get("op", "")), args,
                                     label=str(label)[:120] if label else None,
                                     merge=str(body["merge"])[:200] if body.get("merge") else None,
                                     merge_into=body.get("merge_into"),
                                     record=body.get("record", True) is not False,
                                     base_rev=body.get("base_rev"), known_ops_key=body.get("ops_key"),
                                     merge_count=_count(body.get("merge_count", 1)))
            except editor.StaleRev as exc:
                return _stale(handler, sess, exc)
            except editor.NeedsConfirm as exc:
                if opened:      # the session exists now: the browser must use it for the confirm
                    return handler._json({"needs_confirm": True, "message": str(exc), "session": sess.id,
                                          "rev": sess.rev, **opened, **exc.details})
                raise
        st.ledger.enforce()
        if opened:
            payload.update(opened)
        return handler._json(payload)
    if route == "jump":
        with sess.lock:
            try:
                payload = sess.jump(int(body.get("index", -1)), base_rev=body.get("base_rev"),
                                    known_ops_key=body.get("ops_key"))
            except editor.StaleRev as exc:
                return _stale(handler, sess, exc)
        return handler._json(payload)
    if route == "clean":
        if str(body.get("method") or "springs") != "springs":
            raise ValueError("the synchronous edit/clean runs the spring model only; "
                             "use Clean Geometry (POST /api/clean)")
        with sess.lock:
            if sess.busy:
                raise editor.Busy("Clean is already running on this structure")
            sess.busy = True
            before = sess.atoms.copy()
            symprec = sess.symprec
        try:
            res = gui_clean.clean(before, keep_symmetry=bool(body.get("keep_symmetry", True)),
                                  relax_cell=bool(body.get("relax_cell", False)),
                                  symprec=max(symprec, 1e-5),
                                  time_limit=_time_limit(body.get("time_limit", 10)))
        finally:
            with sess.lock:
                sess.busy = False
        after = res["atoms"]
        comparison = structures.compare(_internal(before), _internal(after))
        status = ("converged" if res["converged"] else
                  "stopped at the time limit" if res["timed_out"] else "stopped at the step limit")
        with sess.lock:
            try:
                payload = sess.apply_clean(after, f"Clean ({status}, {res['steps']} steps, "
                                                  f"max move {comparison['max_displacement']:.3f} Å)",
                                           base_rev=body.get("base_rev"))
            except editor.StaleRev as exc:
                return _stale(handler, sess, exc)
        before_p = structures._payload(before.cell.array, before.get_scaled_positions(wrap=False),
                                       before.positions, before.numbers, (True, True, True),
                                       {"kind": "edit", "path": "before Clean"})
        return handler._json({"structure": payload, "before": before_p,
                              "comparison": comparison,
                              "changed_bonds": [r for r in comparison["bonds"]
                                                if abs(r["delta"]) > 0.01],
                              "report": {k: v for k, v in res.items() if k != "atoms"}})
    if route == "clean_accept":
        with sess.lock:
            prop = sess.proposal
            if not prop or prop["id"] != body.get("proposal_id"):
                return handler._error(404, "no such Clean proposal (run Clean again)")
            if body.get("base_rev") is not None and int(body["base_rev"]) != prop["base_rev"]:
                return _stale(handler, sess, editor.StaleRev(sess.rev))
            try:
                payload = sess.apply_clean(prop["atoms"], prop["label"], base_rev=prop["base_rev"])
            except editor.StaleRev as exc:
                return _stale(handler, sess, exc)
            sess.proposal = None
            _clean_provenance(sess, prop)
            if isinstance(payload, dict):
                payload["doc_meta"] = sess.doc_meta
        st.ledger.enforce()
        return handler._json(payload)
    if route == "clean_discard":
        with sess.lock:
            had = bool(sess.proposal) and sess.proposal["id"] == body.get("proposal_id", sess.proposal["id"])
            if had:
                sess.proposal = None
        return handler._json({"discarded": had, "rev": sess.rev})
    if route == "save_cif":
        name = str(body.get("name") or "")
        if not _SAFE_CIF.match(name):
            raise ValueError("file name must be a plain name ending in .cif")
        # saving.py's rules: inside --dir, never through a symbolic link,
        # <file>.bak kept on the first overwrite, written atomically
        target = st.root / name
        st.check_write(target)
        exists = target.exists()
        if exists and not body.get("overwrite"):
            raise ValueError(f"{name} already exists (tick overwrite to replace it)")
        expected = body.get("expected") if isinstance(body.get("expected"), dict) else None
        if exists and expected is not None and not saving.same_stamp(saving.file_stamp(target), expected):
            raise saving.SaveConflict(f"{name} changed on disk since it was opened", {"changed_on_disk": True})
        with sess.lock:
            text = sess.cif_text(str(body.get("mode", "p1")))
        if exists:
            saving.keep_backup(target)
        # the stamp the browser was shown (or "did not exist") is checked again right before the rename
        hist = saving.write_atomic(target, text.encode(),
                                   expected=expected if expected is not None else (None if not exists else saving._UNSET),
                                   reason=f"Saved from {sess.name or 'the editor'}", route="save_cif")
        return handler._json({"saved": st.rel(target), "history": hist.record() if hist else None})
    if route == "deck_preview":
        name = str(body.get("name") or "")
        if not _SAFE_CIF.match(name):
            raise ValueError("file name must be a plain name ending in .cif")
        with sess.lock:
            text = sess.cif_text(str(body.get("cif_mode", "p1")))
        work = Path(tempfile.mkdtemp(prefix="mace_gui_edit_", dir=st.cache))
        try:
            cif = work / name
            cif.write_text(text)
            result = decks.preview("cif2d12", body.get("json"), cif)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        result["command"] = decks.command_line("cif2d12", "<saved>.json", Path(name))
        result["cif"] = text
        return handler._json(result)
    return handler._error(404, f"unknown editor route {route}")


def _symmetry_route(handler, body: Dict[str, Any]):
    """Stateless space group of a structure the browser holds (live previews
    while dragging, the Edit Cell preview, documents with no session). Runs
    through the compute gate: per document key only the newest waiting call
    runs; an older one answers {superseded: true} without running spglib."""
    import numpy as np
    try:
        cell = np.asarray(body["cell"], float)
        frac = np.asarray(body["frac"], float)
        numbers = [int(n) for n in body["numbers"]]
    except (KeyError, TypeError, ValueError):
        raise ValueError("give cell (3x3), frac (one triple per atom) and numbers")
    if len(numbers) > editor.MAX_ATOMS:
        raise ValueError(f"more than {editor.MAX_ATOMS} atoms: the live space group stops there")
    if cell.shape != (3, 3) or not np.all(np.isfinite(cell)) or abs(np.linalg.det(cell)) < 1e-6:
        raise ValueError("cell must be three linearly independent vectors")
    if frac.shape != (len(numbers), 3) or not np.all(np.isfinite(frac)):
        raise ValueError("frac must hold one finite triple per atom")
    if any(not 1 <= n <= 103 for n in numbers):
        raise ValueError("atomic numbers must be 1 to 103")
    symprec = editor.check_symprec(body.get("symprec", 1e-3))
    key = body.get("key")
    key = str(key)[:200] if key not in (None, "") else None
    hold = 0.0
    if handler.state.test_hooks:
        try:
            hold = min(float(handler.headers.get("X-Mace-Test-Hold-Ms") or 0), 5000) / 1000
        except ValueError:
            hold = 0.0
    arrived = time.perf_counter()
    t = {}

    def work():
        t["start"] = time.perf_counter()
        r = structures.symmetry(cell, frac, numbers, symprec=symprec, with_ops=bool(body.get("ops")))
        t["end"] = time.perf_counter()
        return r

    res = compute.GATE.run(key, work, hold, heavy=len(numbers) > compute.HEAVY_ATOMS)
    if res is compute.SUPERSEDED:
        wait = (time.perf_counter() - arrived) * 1000
        return handler._json({"superseded": True}, timing=f"wait;dur={wait:.1f}")
    wait, dur = (t["start"] - arrived) * 1000, (t["end"] - t["start"]) * 1000
    return handler._json(res, timing=f"symm;dur={dur:.1f}, wait;dur={wait:.1f}")


DETAIL_KINDS = ("cif", "d12", "out", "log")
DETAIL_CHUNK = 64
DETAIL_TASK_OVER = 200


def _file_details(st, files, client=None) -> Dict[str, Any]:
    """GET /api/files?details=1 (plan T11): formula, atom count and space
    group of each structure file, read in the process pool in chunks of 64
    (in parallel up to the pool's cap) and kept in the parse cache under the
    file's (path, mtime, size) key; more than 200 to read show as a task."""
    out: Dict[str, Any] = {}
    todo = []
    for f in files:
        if f.get("kind") not in DETAIL_KINDS:
            continue
        p = st.root / f["path"]
        try:
            key = gui_cache.file_key(p, ("details",))
        except OSError:
            continue
        hit = st.parse_cache.get(key)
        if hit is not None:
            out[f["path"]] = hit
        else:
            todo.append((f["path"], str(p), key))
    if not todo:
        return out
    chunks = [todo[i:i + DETAIL_CHUNK] for i in range(0, len(todo), DETAIL_CHUNK)]
    shown = len(todo) > DETAIL_TASK_OVER
    tasks = []
    for n, ch in enumerate(chunks):
        title = f"Reading file details ({n + 1} of {len(chunks)})" if shown else "Reading file details"
        tasks.append((ch, st.tasks.submit("details", title, "builders.file_details", {"paths": [x[1] for x in ch]},
                                          client=client if shown else None, quiet=not shown)))
    for ch, t in tasks:
        st.tasks.wait(t, 600)
        res = t.result if t.state == "done" and isinstance(t.result, dict) else {}
        t.result = None
        for rel, absp, key in ch:
            d = res.get(absp)
            if d is None:
                continue
            out[rel] = d
            st.parse_cache.put(key, d, 200)
        if not shown:
            st.tasks.remove(t.id)
    st.ledger.enforce()
    return out


# ---------------------------------------------------------------- Calculation Setup (plan B1-B4)
def _setup_source(st, body: Dict[str, Any]):
    """(mode, calc, path, rel, src_key, cif_bytes, cif_name, label) for a Setup request."""
    src = body.get("source") or {}
    if not isinstance(src, dict):
        raise ValueError("source is {path} or {session, rev}")
    calc_in = body.get("calc_type")
    if _structural_source(src, body):
        s = _source(st, src)
        dim = str(((body.get("answers") or {}).get("dimensionality")) or "CRYSTAL").upper()
        if dim != "CRYSTAL":
            # a molecule, slab or polymer: oriented the way the CLI's coordinate writer needs (plan E10),
            # a rigid motion only; the CIF handed to cif2d12 is that cell, in P1
            return _setup_low_dim(st, body, s, dim, calc_in)
        if not all(s.atoms.pbc):
            raise gui_setup.SetupError("cif2d12 sets up 3D cells; this structure is not periodic in 3D: "
                                       "choose its dimensionality (MOLECULE, SLAB or POLYMER)")
        cif_mode = str(body.get("cif_mode") or "p1")
        if cif_mode not in ("p1", "symmetrized"):
            raise ValueError("cif_mode is p1 or symmetrized")
        if s.kind == "session":
            sess = st.sessions.get(str(src["session"]))
            with sess.lock:
                cif = saving.render(sess, "cif", cif_mode)
        else:
            cif = saving.render(_SessLike(s.atoms, s.name), "cif", cif_mode)
        name = __import__("re").sub(r"[^A-Za-z0-9_.-]+", "_", str(body.get("name") or s.name))[:60] or "structure"
        if not name.lower().endswith(".cif"):
            name += ".cif"
        mode = "cif2d12"
        calc = gui_setup.check_calc(mode, calc_in)
        key = ("cif", hashlib.sha256(cif).hexdigest(), name)
        return mode, calc, None, None, key, cif, name, s.label
    p = st.resolve(src.get("path"))
    if not p.is_file():
        raise ValueError("Setup reads a file, not a folder")
    rel = st.rel(p)
    mode = body.get("mode") or gui_setup.mode_for(p.name, calc_in)
    if mode == "opt2d12" and str(calc_in or "").upper() in gui_setup.CALC_TYPES["opt2d3"]:
        mode = "opt2d3"
    if mode not in gui_setup.MODES:
        raise ValueError(f"unknown mode {mode!r}")
    if mode == "cif2d12" and not p.name.lower().endswith(".cif"):
        raise gui_setup.SetupError("cif2d12 reads a CIF")
    if mode != "cif2d12" and not p.name.lower().endswith(gui_setup.OUT_SUFFIXES):
        raise gui_setup.SetupError(f"{mode} reads a CRYSTAL .out")
    calc = gui_setup.check_calc(mode, calc_in)
    return mode, calc, p, rel, gui_setup.source_key(mode, p), None, None, p.name


MOLECULAR_KINDS = ("xyz", "poscar", "mol", "pdb", "gen", "fort34")


def _structural_source(src, body=None) -> bool:
    """A source Setup reads as a structure (session, held, a molecular file, or a CIF set up as a
    SLAB, POLYMER or MOLECULE, which is oriented first), not as a file the CLI reads itself."""
    if src.get("session") or src.get("held"):
        return True
    path = str(src.get("path") or "")
    kind = filekind.kind_by_name(path.split("/")[-1]) if path else ""
    if kind in MOLECULAR_KINDS:
        return True
    dim = str((((body or {}).get("answers") or {}).get("dimensionality")) or "CRYSTAL").upper()
    return kind in ("cif", "d12") and dim in ("SLAB", "POLYMER", "MOLECULE")


def _setup_dimensionality(st, body: Dict[str, Any]) -> None:
    """A source with no 3D cell gets its dimensionality answered for it (CRYSTAL is impossible):
    MOLECULE with no periodic direction, SLAB with two, POLYMER with one. An answer the user gave wins."""
    src = body.get("source") or {}
    if not isinstance(src, dict) or not _structural_source(src):
        return
    answers = body.setdefault("answers", {})
    if not isinstance(answers, dict) or answers.get("dimensionality"):
        return
    try:
        s = _source(st, src)
    except Exception:
        return
    pbc = [bool(x) for x in s.atoms.pbc]
    if not all(pbc):
        answers["dimensionality"] = {0: "MOLECULE", 1: "POLYMER", 2: "SLAB"}[sum(pbc)]


def _setup_low_dim(st, body, s, dim: str, calc_in):
    import numpy as np
    from ase import Atoms
    if dim not in ("SLAB", "POLYMER", "MOLECULE"):
        raise ValueError("dimensionality is CRYSTAL, SLAB, POLYMER or MOLECULE")
    gap = float(gui_settings.load().get("structure.vacuumGap") or structures.DIM_GAP)
    a = s.atoms
    try:
        r = structures.standardize_for_dim(a.numbers, a.positions, np.asarray(a.cell), list(a.pbc), dim, gap)
    except structures.OrientError as exc:
        raise gui_setup.SetupError(str(exc))
    atoms = Atoms(numbers=a.numbers, cell=r["cell"], positions=r["cart"], pbc=True)
    cif = saving.render(_SessLike(atoms, s.name), "cif", "p1")
    name = __import__("re").sub(r"[^A-Za-z0-9_.-]+", "_", str(body.get("name") or s.name))[:60] or "structure"
    if not name.lower().endswith(".cif"):
        name += ".cif"
    mode = "cif2d12"
    calc = gui_setup.check_calc(mode, calc_in)
    key = ("cif", hashlib.sha256(cif).hexdigest(), name)
    label = f"{s.label} · {r['note']}" if r.get("moved") else s.label
    return mode, calc, None, None, key, cif, name, label


_SCHEMA: Dict[str, Any] = {}


def _setup_schema() -> Dict[str, Any]:
    if not _SCHEMA:
        try:
            _SCHEMA.update(gui_setup.schema())
        except Exception:          # pragma: no cover - a CLI module that does not import: the form still works
            pass
    return _SCHEMA


def _setup_job(st, body: Dict[str, Any]):
    """The function a "deck" task runs: one Setup run in its own cache folder (removed at the end)."""
    _setup_dimensionality(st, body)
    req = gui_setup.Request(body)
    mode, calc, p, rel, key, cif, cif_name, label = _setup_source(st, body)
    pre_warn = _setup_preopt_warning(st, body, p, calc)

    def get(k):
        return st.parse_cache.get(("setup",) + (json.dumps(k, default=str),))

    def put(k, v, n):
        st.parse_cache.put(("setup",) + (json.dumps(k, default=str),), v, n)
        st.ledger.enforce()

    def job(ctx):
        folder = "setup-" + secrets.token_hex(6)
        work = st.disk.acquire(folder)
        try:
            out = gui_setup.run_setup(req, mode, calc, work, p, rel, key, get, put, ctx, cif, cif_name, _setup_schema())
        finally:
            st.disk.release(folder, remove=True)        # a finished run's folder goes at once
        out["source_label"] = label
        out["source"] = body.get("source")
        if pre_warn:
            out["preopt_warning"] = pre_warn
        return out
    title = f"Deck preview: {label} ({calc})" if req.preview else f"Calculation Setup: {label} ({calc})"
    return job, title


def _setup_preopt_warning(st, body: Dict[str, Any], p: Optional[Path], calc: str) -> Optional[str]:
    """Setup's non-blocking warning for an FF/ML pre-optimized geometry and a
    calculation type other than OPT (the deck and defaults are unchanged)."""
    if str(calc).upper() == "OPT":
        return None
    src = body.get("source") or {}
    pre = None
    try:
        if src.get("session"):
            pre = (getattr(st.sessions.get(str(src["session"])), "doc_meta", None) or {}).get("preopt")
        elif src.get("held"):
            pre = (st.held.get(src["held"]).get("meta") or {}).get("preopt")
        elif p is not None:
            pre = (_file_meta(p) or {}).get("preopt")
    except Exception:
        pre = None
    if not pre:
        return None
    return (f"This geometry is an FF/ML pre-optimization ({pre.get('method_ran')}). Build an OPT first; "
            f"{str(calc).upper()} on it is not meaningful (FREQ shows spurious imaginary modes).")


def _setup_run_route(handler, route: str, body: Dict[str, Any]):
    """POST /api/setup/run (and its names defaults / apply / preview): the
    CLI's defaults, its questions and these answers, and with preview the
    printed command's deck, as a cancelable "deck" task; a newer run for
    the same document supersedes the one before (plan B2)."""
    st = handler.state
    if route == "setup/defaults":
        body = {**body, "answers": {}, "patches": {}, "json_text": None}
    elif route == "setup/apply":
        body = {"preview": False, **body}
    elif route == "setup/preview":
        body = {**body, "preview": True}
    job, title = _setup_job(st, body)
    doc = str(body.get("doc") or "") or None
    t = cli_task(st, "deck", title, job, doc=doc, client=handler.headers.get("X-Mace-Client"),
                 supersede=("deck", doc) if doc else None)
    try:
        wait = min(600.0, max(0.0, float(body.get("wait", gui_tasks.WAIT_S))))
    except (TypeError, ValueError):
        wait = gui_tasks.WAIT_S
    return gui_tasks.answer(handler, st.tasks, t, wait)


def _setup_run_sync(handler, body: Dict[str, Any]) -> Dict[str, Any]:
    st = handler.state
    job, title = _setup_job(st, body)
    t = cli_task(st, "deck", title, job, client=handler.headers.get("X-Mace-Client"))
    st.tasks.wait(t, 600)
    if t.state != "done":
        raise gui_setup.SetupError(t.error or f"the deck task {t.state}")
    res, t.result = t.result, None
    return res


def _setup_save_route(handler, body: Dict[str, Any]):
    """POST /api/setup/save: the settings file, written only now, inside --dir
    (and for "Own folder" a copy of the CIF beside it); the command that turns
    it into the deck. ``write_decks``: also run that command in --dir (a task)."""
    st = handler.state
    _setup_dimensionality(st, body)
    mode, calc, p, rel, key, cif, cif_name, label = _setup_source(st, body)
    if p is None:
        raise gui_setup.SetupError("Save the structure as a CIF first: the command reads a file in the folder")
    text = body.get("json_text")
    if not isinstance(text, str):
        raise ValueError("json_text is the settings file's text (from the Setup run)")
    bad = gui_setup.validate(mode, text)
    if bad:
        raise gui_setup.SetupError("; ".join(bad))
    layout = str(body.get("layout") or ("own" if mode == "cif2d12" else "beside"))
    overwrite = bool(body.get("overwrite"))
    import re as _re
    name = str(body.get("name") or "")
    if name and not _re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,120}\.json", name):
        raise ValueError("the settings file name: letters, digits, . _ + - and .json at the end")
    writes: List[Tuple[Path, bytes]] = []
    if mode == "cif2d12":
        stem = p.stem
        if layout == "own":
            folder = st.root / "setup" / stem
            jname = name or gui_setup.json_name(mode, calc, stem)
            writes += [(folder / p.name, p.read_bytes()), (folder / jname, text.encode())]
            args = ["convert", "--batch", "--options_file", st.rel_new(folder / jname), "--cif_dir", st.rel_new(folder),
                    "--output_dir", st.rel_new(folder)]
        elif layout == "inplace":
            jname = name or gui_setup.json_name(mode, calc, stem)
            writes.append((p.parent / jname, text.encode()))
            args = ["convert", "--batch", "--options_file", st.rel_new(p.parent / jname), "--cif_dir", st.rel_new(p.parent)]
        else:
            raise ValueError("layout is own or inplace")
    else:
        jname = name or gui_setup.json_name(mode, calc, p.stem)
        writes.append((p.parent / jname, text.encode()))
        args = (["opt2d12", "--out-file", rel, "--config-file", st.rel_new(p.parent / jname), "--yes"] if mode == "opt2d12"
                else ["opt2d3", "-i", rel, "-t", calc, "--config-file", st.rel_new(p.parent / jname)])
    stamps = body.get("expected") if isinstance(body.get("expected"), dict) else {}
    for target, _ in writes:
        st.check_write(target)
        if target.exists() and not overwrite and not (target.suffix == ".cif" and target.read_bytes() == p.read_bytes()):
            raise gui_setup.SetupError(f"{st.rel_new(target)} already exists (tick overwrite to replace it)")
    for target, data in writes:
        target.parent.mkdir(parents=True, exist_ok=True)
        rel_t = st.rel_new(target)
        # the stamp the browser was shown for the file (or none: it did not exist when the dialog opened)
        exp = stamps.get(rel_t, saving._UNSET) if stamps else (None if not target.exists() else saving._UNSET)
        saving.write_atomic(target, data, expected=exp, reason="Setup overwrote " + ("the JSON" if target.suffix == ".json" else "the structure copy"),
                            route="setup")
    others = sorted(x.name for x in p.parent.glob("*.cif") if x != p) if mode == "cif2d12" else []
    out = {"saved": [st.rel_new(t) for t, _ in writes], "json": st.rel_new(writes[-1][0]),
           "json_sha256": hashlib.sha256(text.encode()).hexdigest(), "argv": args,
           "command": gui_setup.command_line(args), "layout": layout,
           "converts": (len(others) + 1) if layout == "inplace" else 1, "others": others[:20]}
    if mode == "opt2d3":
        out["f9"] = gui_setup.find_wavefunction(p)
    if body.get("write_decks"):
        return _write_decks(handler, out, mode, calc, p)
    return handler._json(out)


def _write_decks(handler, saved: Dict[str, Any], mode: str, calc: str, p: Path):
    """"Write decks now": the printed command, run as a task. For opt2d3 the
    CLI also copies the wavefunction: refused when the folder lacks room for
    it plus 1 GB, with a RAM warning on a tmpfs.

    The CLI writes into a fresh private folder inside the output folder (its
    own output-directory option); the new files then move into the output
    folder under the same rules as every other save: never through a
    symbolic link (one already there may lead out of --dir), a replaced deck
    keeps a ``.bak`` (the wavefunction copy does not: it is remade from its
    source). A refused move moves nothing."""
    st = handler.state
    warn = None
    if mode == "opt2d3":
        f9 = saved.get("f9")
        if not f9:
            raise gui_setup.SetupError("opt2d3 writes no deck without the wavefunction beside the .out")
        if f9.get("live"):
            raise gui_setup.StillRunning("The run is still going; its wavefunction (.f9) is copied when the run ends.")
        free = shutil.disk_usage(p.parent).free
        if free < f9["bytes"] + (1 << 30):
            raise gui_setup.SetupError(f"{f9['name']} is {gui_setup._mb(f9['bytes'])}; {st.rel_new(p.parent) or '.'} has "
                                       f"{gui_setup._mb(free)} free (the copy needs its size plus 1 GB)")
        if _on_tmpfs(p.parent):
            warn = f"{st.rel_new(p.parent) or 'This folder'} is on a tmpfs: the {gui_setup._mb(f9['bytes'])} copy of {f9['name']} uses RAM"
    argv = list(saved["argv"])
    out_dir = st.root / (argv[argv.index("--output_dir") + 1] if "--output_dir" in argv else
                         str(Path(argv[argv.index("--cif_dir") + 1])) if "--cif_dir" in argv else
                         str(Path(argv[2]).parent))
    st.check_write(out_dir / "x.d12")
    if out_dir.is_symlink():
        raise PermissionError(f"{st.rel_new(out_dir)} is a symbolic link; nothing written")
    out_flag = "--output_dir" if mode == "cif2d12" else "--output-dir"

    def job(ctx):
        out_dir.mkdir(parents=True, exist_ok=True)
        # mkdtemp: a new name, mode 0700, so nothing can be planted in it
        stage = Path(tempfile.mkdtemp(prefix=".mace-decks-", dir=str(out_dir)))
        try:
            run = argv[:argv.index(out_flag)] + argv[argv.index(out_flag) + 2:] if out_flag in argv else argv
            ctx.progress(0.2, f"Running {saved['command']}")
            r = clirun.run(clirun.mace_argv("--no-banner", *run, out_flag, str(stage)), st.root, ctx, timeout=600)
            made = sorted(x for x in stage.iterdir() if x.is_file() and not x.is_symlink())
            targets = [(x, out_dir / x.name) for x in made]
            for src, target in targets:
                # the opt2d3 wavefunction copy may be written: when the target is absent,
                # or holds the same copy (same size and mtime as the copy just made)
                st.check_write(target, allow=("wavefunction",))
                if target.is_dir():
                    raise PermissionError(f"{st.rel_new(target)} is a folder; not overwritten")
                if target.suffix.lower() in (".f9", ".f98") and target.exists():
                    a, b = target.stat(), src.stat()
                    if a.st_size != b.st_size or int(a.st_mtime) != int(b.st_mtime):
                        raise saving.ProtectedOutput(
                            f"{target.name} holds a finished run's wavefunction; MACE never writes over it. "
                            "Write the decks into another folder, or move it first.")
            written, backups = [], []
            for src, target in targets:
                deck = target.suffix.lower() in (".d12", ".d3")
                if target.exists():
                    if deck:        # the wavefunction copy is remade from its source: no .bak of it
                        bak = saving.keep_backup(target)
                        if bak["created"]:
                            backups.append(st.rel_new(bak["path"]))
                        gui_history.keep(target, "Write decks replaced the deck", route="write_decks")
                    shutil.copymode(target, src)     # as the CLI's own overwrite keeps it
                    os.replace(src, target)      # renames over the name; never follows a link
                else:
                    os.replace(src, target)
                    if deck:
                        gui_history.record_created(target, "Write decks created the deck", route="write_decks")
                if deck:
                    written.append(st.rel_new(target))
        finally:
            shutil.rmtree(stage, ignore_errors=True)
        log = r.log.replace(str(stage) + os.sep, "").replace(str(stage), st.rel_new(out_dir) or ".")
        return {**saved, "written": written, "backups": backups, "code": r.returncode,
                "log": log.replace(str(st.root) + os.sep, "")[-4000:], "warning": warn}
    t = cli_task(st, "deck", f"Write decks: {saved['command']}", job, client=handler.headers.get("X-Mace-Client"))
    return gui_tasks.answer(handler, st.tasks, t, 0.3)


def _convergence(st, p: Path):
    from mace.utils import out_convergence as oc
    live = _out_live(p)
    key = gui_cache.file_key(p, ("convergence", oc.VERSION, outindex.VERSION, live))
    res = st.parse_cache.get(key)
    if res is None:
        try:
            idx = _out_index(st, p)
        except Exception:
            idx = None
        res = runinfo.convergence(p, idx)
        st.parse_cache.put(key, res, 4096 + 600 * len(res.get("points") or []))
        st.ledger.enforce()
    return key, res


def _run_route(handler, route: str, query):
    """GET /api/out/convergence, /api/out/problems, /api/out/meta (PLAN_analysis C2)."""
    st = handler.state
    p = st.resolve(handler._q(query, "path"))
    if route == "out/convergence":
        key, res = _convergence(st, p)
        etag = gui_cache.etag_of(key)
        if etag in (handler.headers.get("If-None-Match") or ""):
            return handler._send(304, b"", "application/json", {"ETag": etag, "Cache-Control": "no-cache"})
        return handler._send_body(json.dumps(res, default=_json_default).encode(), etag)
    if route == "out/problems":
        _, conv = _convergence(st, p)
        try:
            noise = float(gui_settings.load().get("analysis.imagNoiseCm", 20))
        except (TypeError, ValueError):
            noise = 20.0
        try:
            dim = _out_index(st, p).dim
        except Exception:
            dim = None
        return handler._json({"path": st.rel(p), "problems": runinfo.problems(p, conv, noise, dim),
                              "stop": conv.get("stop"), "reason": ((conv.get("opt") or {}).get("end") or {}).get("reason")})
    # out/meta: a pool job (the extractor reads the whole file), the card shows a skeleton meanwhile
    key = gui_cache.file_key(p, ("outmeta", 1))
    hit = st.parse_cache.get(key)
    _, conv = _convergence(st, p)

    def finish(meta):
        st.parse_cache.put(key, meta, 4096)
        st.ledger.enforce()
        return {"path": st.rel(p), "meta": runinfo.card(meta, conv), "stop": conv.get("stop"),
                "reason": ((conv.get("opt") or {}).get("end") or {}).get("reason")}
    if hit is not None:
        return handler._json(finish(hit))
    t = st.tasks.submit("meta", f"Run details of {p.name}", "out.meta", {"path": str(p)},
                        client=handler.headers.get("X-Mace-Client"), finish=finish,
                        supersede=("meta", st.rel(p)))
    return gui_tasks.answer(handler, st.tasks, t, float(handler._q(query, "wait", "0.05")))


TRACK_INLINE = 20


def _track_route(handler, body: Dict[str, Any]):
    """POST /api/out/track {path, measurement}: up to 20 frames answered at
    once, more as a cancelable pool task (not a write)."""
    st = handler.state
    p = st.resolve(str(body.get("path") or ""))
    m = body.get("measurement")
    if not isinstance(m, dict):
        raise ValueError("measurement is {type, atoms: [{base, shift}]}")
    idx = _out_index(st, p)
    if idx.n_steps <= TRACK_INLINE:
        frames = [idx.block_structure(k) for k in range(idx.n_steps)]
        return handler._json({"path": st.rel(p), "values": runinfo.track(frames, m), "n": idx.n_steps})
    t = st.tasks.submit("track", f"Track along {p.name}", "out.track", {"path": str(p), "measurement": m},
                        client=handler.headers.get("X-Mace-Client"),
                        finish=lambda r: {"path": st.rel(p), **r})
    return gui_tasks.answer(handler, st.tasks, t, float(body.get("wait", 0.3)))


def _history_restore(handler, body: Dict[str, Any]):
    """POST /api/history/restore {id, path, expected}: a guarded write. The
    entry must belong to this folder and to ``path``; the target passes
    check_write (root, links, protected kinds: a protected file is never
    restored over); the current file must still be what the browser was
    shown (``expected``, else SaveConflict); the current version goes to
    File History first. A "created" entry restores by deleting the file
    MACE created, while its bytes still match."""
    st = handler.state
    rec = gui_history.entry_record(str(body.get("id") or ""))
    target = st.resolve(str(body.get("path") or ""), must_exist=False)
    if st.rel_new(target) != rec["rel_path"]:
        raise ValueError("that version belongs to another file")
    st.check_write(target)
    expected = body.get("expected") if isinstance(body.get("expected"), dict) else None
    now = saving.file_stamp(target)
    if not saving.same_stamp(now, expected):
        raise saving.SaveConflict(f"{target.name} changed since the list was shown", {"changed_on_disk": True, **(now or {})})
    if rec.get("created"):
        if not target.exists():
            return handler._json({"restored": rec["id"], "deleted": False})
        if gui_history._sha256(target) != rec.get("sha256"):
            raise saving.SaveConflict(f"{target.name} changed after MACE created it; not deleted", {"changed_on_disk": True})
        gui_history.keep(target, "Before deleting the file MACE created", route="history/restore")
        target.unlink()
        return handler._json({"restored": rec["id"], "deleted": True, "path": rec["rel_path"]})
    data = gui_history.entry_path(rec["id"]).read_bytes()
    hist = saving.write_atomic(target, data, expected=expected if expected is not None else None,
                               reason=f"Before restoring the version of {time.strftime('%Y-%m-%d %H:%M', time.localtime(rec['t']))}",
                               route="history/restore")
    return handler._json({"restored": rec["id"], "path": rec["rel_path"], "file": saving.file_stamp(target),
                          "history": hist.record() if hist else None})


def _preflight_route(handler, body: Dict[str, Any]):
    """POST /api/setup/preflight: mace preflight on the previewed deck (a "preflight" task)."""
    st = handler.state
    name, text = str(body.get("deck_name") or ""), body.get("deck")
    if not isinstance(text, str) or len(text) > 5_000_000:
        raise ValueError("deck is the previewed deck's text")
    if not name.endswith(".d12"):
        raise ValueError("preflight checks .d12 decks (CRYSTAL reads them with TESTPDIM)")
    binary = str(body.get("crystal_bin")) if st.test_hooks and body.get("crystal_bin") else None
    doc = str(body.get("doc") or "") or None

    def job(ctx):
        folder = "preflight-" + secrets.token_hex(6)
        work = st.disk.acquire(folder)
        try:
            return gui_preflight.run(name, text, work, ctx, binary)
        finally:
            st.disk.release(folder, remove=True)
    t = cli_task(st, "preflight", f"Preflight {name}", job, doc=doc, client=handler.headers.get("X-Mace-Client"),
                 supersede=("preflight", doc) if doc else None)
    return gui_tasks.answer(handler, st.tasks, t, float(body.get("wait", gui_tasks.WAIT_S)) if isinstance(body.get("wait"), (int, float)) else gui_tasks.WAIT_S)


# ---------------------------------------------------------------- materials.db (plan B6)
def _not_while_building(p: Path) -> None:
    """A database a GUI build is rewriting is not read (no snapshot of a
    file in mid-update): 503 busy "building"; the view says "Updating..."
    and reloads when the build ends (PLAN_analysis C10.4)."""
    who = dbwriters.owner(p)
    if who:
        raise gui_db.DbBusy("building", f"Updating... ({who.lower()} is in progress)")


def _db_build_paths(st, folder_rel, db_rel, open_db_rel=None):
    """(folder, db, facts) for the build dialog: the folder inside the root,
    the database chosen per dbbuild.choose_db (or as typed)."""
    folder = st.root if folder_rel in (None, "", ".") else st.resolve(str(folder_rel))
    if not folder.is_dir():
        raise ValueError("Choose a folder")
    if db_rel:
        raw = Path(str(db_rel))
        db = raw if raw.is_absolute() else Path(os.path.normpath(os.path.join(str(st.root), str(raw))))
        source = "typed"
    else:
        open_db = st.resolve(str(open_db_rel)) if open_db_rel else None
        ch = dbbuild.choose_db(st.root, folder, open_db)
        db, source = ch["path"], ch["source"]
    if db.suffix.lower() != ".db":
        raise ValueError("The database is a .db file (materials.db)")
    inside = dbbuild.inside(st.root, db) and dbbuild.inside(st.root, db.parent)
    return folder, db, {"source": source, "inside": inside}


def _db_build_info(handler, query):
    """What the Create or Update Database dialog states before anything runs."""
    st = handler.state
    folder, db, facts = _db_build_paths(st, handler._q(query, "folder"), handler._q(query, "db"),
                                        handler._q(query, "open"))
    exists = db.is_file()
    keep = int(handler._q(query, "keep") or 3)
    flag = lambda k: handler._q(query, k) not in ("0", "false")    # noqa: E731
    opts = dbbuild.options({"keep": keep, "register": flag("register"), "skip_unchanged": flag("skip"),
                            "backup": exists}, exists)
    folder_arg, db_arg = dbbuild.rel_or_abs(st.root, folder), dbbuild.rel_or_abs(st.root, db)
    args = dbbuild.cli_args(folder_arg, db_arg, opts["register"], opts["skip_unchanged"], opts["backup"], opts["keep"])
    n_out = dbbuild.count_outputs(folder)
    return handler._json({
        "folder": folder_arg, "db": db_arg, "source": facts["source"], "inside": facts["inside"],
        "exists": exists, "property_rows": dbbuild.property_rows(db) if exists and facts["inside"] else None,
        "backups": dbbuild.backups(db, opts["keep"]) if facts["inside"] else {"files": [], "prune": []},
        "outputs": n_out, "tracked": exists and dbbuild.property_rows(db) is not None,
        "writer": dbwriters.owner(db), "command": dbbuild.command_text(args), "guarantee": dbbuild.GUARANTEE,
        "outside_note": None if facts["inside"] else dbbuild.OUTSIDE,
        "root_gen": getattr(st, "root_gen", None)})


def _db_build_body(st, body):
    folder, db, facts = _db_build_paths(st, body.get("folder"), body.get("db"))
    if not facts["inside"]:
        raise PermissionError(dbbuild.OUTSIDE)
    opts = dbbuild.options(body, db.is_file())
    return folder, db, opts, dbbuild.rel_or_abs(st.root, folder), dbbuild.rel_or_abs(st.root, db)


def _db_build_preview(handler, body):
    """--dry-run --json as a task: nothing is written."""
    st = handler.state
    folder, db, opts, folder_arg, db_arg = _db_build_body(st, body)
    t = cli_task(st, "dbpreview", f"Preview database update from {folder.name or 'the folder'}",
                 lambda ctx: dbbuild.preview(st.root, folder_arg, db_arg, opts, ctx),
                 client=handler.headers.get("X-Mace-Client"), supersede=("dbpreview", db_arg))
    return gui_tasks.answer(handler, st.tasks, t, float(body.get("wait", 0.3)))


def _db_build_route(handler, body):
    """The build: confirmed, inside the root, one GUI writer per database,
    the CLI with --progress-json as a cancelable task."""
    st = handler.state
    if body.get("confirm") is not True:
        raise ValueError("confirm the database update")
    # PLAN_connect R6: the folder changed since the page loaded. The page sends the
    # generation it loaded with (and every POST carries X-Mace-Root-Gen, checked in
    # _dispatch); a caller that names none is checked by the header alone.
    gen = getattr(st, "root_gen", None)
    if gen is not None and body.get("root_gen") is not None and body.get("root_gen") != gen:
        return handler._error(409, "The folder changed; reload")
    folder, db, opts, folder_arg, db_arg = _db_build_body(st, body)
    st.check_write(db, allow=("database",))
    if st.test_hooks:
        st.test_log.append({"db_build_argv": dbbuild.cli_args(folder_arg, db_arg, opts["register"], opts["skip_unchanged"],
                                                              opts["backup"], opts["keep"], progress=True)})
    delay = min(60.0, max(0.0, float(body.get("test_delay") or 0))) if st.test_hooks else 0.0
    token = dbwriters.acquire(db, dbbuild.OWNER)
    try:
        def run(ctx):
            end = time.monotonic() + delay           # test hooks: a build that holds the database a while
            while time.monotonic() < end:
                if ctx.canceled:
                    raise dbbuild.BuildCanceled([], None)
                time.sleep(0.1)
            try:
                return dbbuild.build(st.root, folder_arg, db_arg, opts, ctx, ctx.progress)
            except dbbuild.BuildCanceled as exc:
                ctx.task.cancel_note = str(exc)
                raise
        verb = "Update" if db.is_file() else "Create"
        t = cli_task(st, "dbbuild", f"{verb} database {db_arg}", run, client=handler.headers.get("X-Mace-Client"))
    except BaseException:
        dbwriters.release(db, token)
        raise

    def release_when_done():
        t.done.wait()
        dbwriters.release(db, token)
    threading.Thread(target=release_when_done, name=f"mace-dbwriter-{t.id}", daemon=True).start()
    return gui_tasks.answer(handler, st.tasks, t, float(body.get("wait", 0)))


def _db_file(st, db_path: Path, name, work_dir) -> Optional[str]:
    """A calculation's input or output as a path inside --dir, or None ("Not in this folder")."""
    if not name:
        return None
    cands = [Path(name)] if os.path.isabs(name) else [db_path.parent / name] + ([Path(work_dir) / name] if work_dir else [])
    for c in cands:
        try:
            r = c.resolve()
        except (OSError, RuntimeError):
            continue
        if (r == st.root or st.root in r.parents) and r.is_file():
            return st.rel(r)
    return None


def _db_route(handler, route: str, query):
    st = handler.state
    p = st.resolve(handler._q(query, "path"))
    get = lambda k: handler._q(query, k)  # noqa: E731
    filters = gui_db.Filters.from_query(get)
    if route == "db/stats":
        return handler._json(gui_db.stats(p))
    if route == "db/props":
        return handler._json({"names": gui_db.property_names(p)})
    if route == "db/count":
        return handler._json({"count": gui_db.count(p, filters)})
    sort, direction = get("sort") or "material_id", get("dir") or "asc"
    if route == "db/page":
        after = get("after")
        if get("index") not in (None, ""):
            after = gui_db.offset_of(p, sort, direction, int(get("index")), filters)
        t0 = time.perf_counter()
        # the page and its total from one read transaction (one snapshot)
        out = gui_db.page(p, sort, direction, after, int(get("limit") or gui_db.PAGE_MAX), filters, with_total=True)
        out["index"] = int(get("index") or 0) if get("index") not in (None, "") else None
        return handler._json(out, timing=f"db;dur={(time.perf_counter() - t0) * 1000:.1f}")
    # db/export: the filtered, sorted table, streamed to the browser as a download
    fmt = get("format") or "csv"
    gen = gui_db.export(p, fmt, sort, direction, filters)
    first = next(gen, b"")                      # a bad format or sort is a 400 before any header goes out
    handler.send_response(200)
    handler.send_header("Content-Type", "text/csv; charset=utf-8" if fmt == "csv" else "application/json")
    # a file name is the user's, but it goes into a header: no quotes, no line breaks, no non-ASCII
    safe = re.sub(r"[^A-Za-z0-9._+-]+", "_", p.stem)[:80] or "materials"
    handler.send_header("Content-Disposition", f'attachment; filename="{safe}_export.{fmt}"')
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("X-Content-Type-Options", "nosniff")
    handler.send_header("Connection", "close")
    handler.end_headers()
    handler.close_connection = True
    handler.wfile.write(first)
    for chunk in gen:
        handler.wfile.write(chunk)
    return None


# ---------------------------------------------------------------- FREQ modes, IR/Raman (plan B5, T9)
def _spectrum_out(spec: Path) -> Optional[Path]:
    """The FREQ .out of the same run as an IRSPEC/RAMSPEC.DAT, when it is beside it."""
    stem = spec.name
    for suffix in ("_IRSPEC.DAT", "_RAMSPEC.DAT", ".IRSPEC.DAT", ".RAMSPEC.DAT", "IRSPEC.DAT", "RAMSPEC.DAT"):
        if stem.upper().endswith(suffix.upper()):
            stem = stem[: -len(suffix)]
            break
    out = spec.parent / f"{stem}.out"
    return out if out.is_file() else None


def _freq_parsed(st, out: Path, client=None, wait_s: float = 120.0):
    """The parsed modes of a FREQ output, from the parse cache or parsed now in
    the pool (the FREQ section only), or None when it has none."""
    key = gui_cache.file_key(out, ("freq", outindex.VERSION))
    parsed = st.parse_cache.get(key)
    if parsed is not None:
        return parsed
    try:
        idx = _out_index(st, out)
    except (ValueError, OSError):
        return None
    if not idx.freq:
        return None
    start, end = idx.freq
    if end - start > gui_freq.SECTION_MAX:
        return None

    def finish(parsed):
        st.parse_cache.put(key, parsed, gui_freq.nbytes(parsed))
        st.ledger.enforce()
        return parsed
    t = st.tasks.submit("freq", f"Normal modes of {out.name}", "freq.parse",
                        {"path": str(out), "start": start, "end": end, "tmpdir": str(st.cache)},
                        client=client, finish=finish, quiet=True)
    st.tasks.wait(t, wait_s)
    res = t.result if t.state == "done" else None
    st.tasks.remove(t.id)
    return res if isinstance(res, dict) else st.parse_cache.get(key)


def _spectrum_peaks(st, spec: Path, client=None):
    """The IR/Raman-active modes of the FREQ .out of the same run, when one is beside it
    (parsed now when the modes were not read yet): each peak names its mode."""
    out = _spectrum_out(spec)
    if out is None:
        return []
    parsed = _freq_parsed(st, out, client)
    if parsed is None:
        return []
    ir = "IRSPEC" in spec.name.upper()
    return gui_spectra.active_peaks(parsed, "ir" if ir else "raman")


def _spectrum_from_modes(handler, p: Path, query):
    """GET /api/spectrum?source=modes: a FREQ output's IR or Raman curve broadened from
    its mode table (width = the run's DAMPFAC, else CRYSTAL's 8 cm-1 FWHM)."""
    st = handler.state
    q = lambda k, d=None: handler._q(query, k, d)          # noqa: E731
    kind = q("kind", "ir")
    if kind not in ("ir", "raman"):
        raise ValueError("kind is ir or raman")
    parsed = _freq_parsed(st, p, handler.headers.get("X-Mace-Client"))
    if parsed is None:
        raise filekind.KindConflict("No normal modes in this output (FREQ with eigenvectors needed)")
    damp = gui_spectra.dampfac(p, kind)
    fwhm = float(q("fwhm") or damp or gui_spectra.DEFAULT_FWHM)
    shape = q("shape", "lorentzian")
    if shape not in gui_spectra.SHAPES:
        raise ValueError("shape is lorentzian, gaussian or pseudo-voigt")
    eta = float(q("eta", "1"))
    xmax = q("xmax")
    data = gui_spectra.from_modes(parsed, kind, shape, fwhm, eta, 0.0, float(xmax) if xmax else None)
    data.update(path=st.rel(p), out=st.rel(p), name=p.name, dampfac=damp,
                default_fwhm=damp or gui_spectra.DEFAULT_FWHM)
    return handler._json(data)


def _freq_route(handler, query):
    """GET /api/freq: an output's normal modes (a pool task over the FREQ section only)
    and the geometry they move (the output's last frame, through its index)."""
    st = handler.state
    p = st.resolve(handler._q(query, "path"))
    idx = _out_index(st, p)
    if not idx.freq:
        raise ValueError("No normal modes in this output (FREQ with eigenvectors needed)")
    start, end = idx.freq
    if end - start > gui_freq.SECTION_MAX:
        raise ValueError("FREQ section too large to animate here; use mace plotting --freq")
    key = gui_cache.file_key(p, ("freq", outindex.VERSION))
    if not idx.n_steps:
        raise filekind.NoGeometry()
    geo = idx.payload(-1, {"kind": "out", "path": st.rel(p)})

    def finish(parsed):
        if "rotation" not in parsed and idx.dim == 0:
            # a molecule: the printed Cartesian frame is the one drawn
            import numpy as np
            parsed["rotation"] = np.eye(3)
        if "rotation" not in parsed:
            try:
                b = idx.blocks[-1]
                parsed["rotation"] = gui_freq.frame_rotation(idx._lines(b), geo["cell"])
            except Exception:
                parsed["rotation"] = None
        st.parse_cache.put(key, parsed, gui_freq.nbytes(parsed))
        st.ledger.enforce()
        n = len(geo["numbers"])
        nd = {len(v) for v in parsed["displacements"].values()}
        return {"path": st.rel(p), "modes": parsed["modes"], "structure": gui_cache.pack(geo),
                "n_atoms": n, "mode_atoms": max(nd) if nd else 0, "section_bytes": parsed["section_bytes"],
                "n_frames": idx.n_steps, "dim": idx.dim,
                "imaginary": [m for m in parsed["modes"] if m["imaginary"]]}
    hit = st.parse_cache.get(key)
    if hit is not None:
        return handler._json(finish(hit))
    t = st.tasks.submit("freq", f"Normal modes of {p.name}", "freq.parse",
                        {"path": str(p), "start": start, "end": end, "tmpdir": str(st.cache)},
                        client=handler.headers.get("X-Mace-Client"), finish=finish)
    return gui_tasks.answer(handler, st.tasks, t, float(handler._q(query, "wait", "0.3")))


# ---------------------------------------------------------------- Volume document (plan E5)
def _cube_path(st, rel) -> Path:
    p = st.resolve(rel)
    if not p.is_file():
        raise FileNotFoundError(st.rel(p))
    if filekind.kind_by_name(p.name) != "cube" or not filekind.cube_header_ok(p):
        raise ValueError(f"{p.name} is not a Gaussian cube file")
    return p


def _volume_finish(st, key):
    """Keep the grid's bytes under a token the page fetches once (api/volume/grid)."""
    def finish(res):
        token = secrets.token_hex(8)
        st.parse_cache.put(("volgrid", token), res["grid"], len(res["grid"]))
        meta = {**res["meta"], "grid_key": token}
        if key is not None:
            st.parse_cache.put(key, {"meta": meta, "token": token}, 4096)
        st.ledger.enforce()
        return meta
    return finish


def _volume_get(handler, route, query):
    st = handler.state
    q = lambda k, d=None: handler._q(query, k, d)          # noqa: E731
    if route == "volume/grid":
        g = st.parse_cache.get(("volgrid", str(q("key", ""))))
        if g is None:
            return handler._error(410, "this grid was let go of; open the volume again")
        body, headers = g, {"Cache-Control": "no-store"}
        if "gzip" in handler.headers.get("Accept-Encoding", "") and len(body) > 65536:
            body = gzip.compress(body, 1)
            headers["Content-Encoding"] = "gzip"
        return handler._send(200, body, "application/octet-stream", headers)
    if route == "volume/slice":
        axis = q("axis", "c")
        if axis not in ("a", "b", "c"):
            raise ValueError("axis is a, b or c")
        job: Dict[str, Any] = {"axis": axis, "index": int(q("index", "0"))}
        if query.get("coef"):
            # a difference made here: the plane of the combination itself (never of its first operand)
            coefs = []
            for c in query.get("coef") or []:
                try:
                    coefs.append(float(c))
                except ValueError:
                    raise ValueError("coefficients must be finite numbers")
            paths, coefs = _volume_operands(st, {"paths": list(query.get("path") or []), "coefficients": coefs})
            job.update(paths=[str(x) for x in paths], coefficients=coefs)
            title = "Slice of the difference"
        else:
            p = _cube_path(st, q("path"))
            job["path"] = str(p)
            title = f"Slice of {p.name}"
        t = st.tasks.submit("volume", title, "volume.slice", job, client=handler.headers.get("X-Mace-Client"), quiet=True)
        return gui_tasks.answer(handler, st.tasks, t, 120)
    p = _cube_path(st, q("path"))
    quality = q("quality", "medium")
    if quality not in ("low", "medium", "high"):
        raise ValueError("quality is low, medium or high")
    key = gui_cache.file_key(p, ("volume", quality))
    hit = st.parse_cache.get(key)
    if hit is not None and st.parse_cache.get(("volgrid", hit["token"])) is not None:
        return handler._json({**hit["meta"], "path": st.rel(p)})
    t = st.tasks.submit("volume", f"Reading {p.name}", "volume.load", {"path": str(p), "quality": quality},
                        client=handler.headers.get("X-Mace-Client"), finish=_volume_finish(st, key))
    return gui_tasks.answer(handler, st.tasks, t, float(q("wait", "0.3")))


def _volume_operands(st, body):
    paths, coefs = body.get("paths"), body.get("coefficients")
    if not isinstance(paths, list) or not isinstance(coefs, list) or len(paths) != len(coefs):
        raise ValueError("paths and coefficients: two lists of the same length")
    if not 2 <= len(paths) <= gui_volume_max():
        raise ValueError(f"2 to {gui_volume_max()} volumes")
    out = []
    for c in coefs:
        if isinstance(c, bool) or not isinstance(c, (int, float)) or c != c or c in (float("inf"), float("-inf")):
            raise ValueError("coefficients must be finite numbers")
        out.append(float(c))
    return [_cube_path(st, x) for x in paths], out


def gui_volume_max() -> int:
    from . import volume
    return volume.MAX_OPERANDS


def _volume_post(handler, route, body):
    st = handler.state
    client = handler.headers.get("X-Mace-Client")
    if route == "volume/check":
        a, b = _cube_path(st, body.get("a")), _cube_path(st, body.get("b"))
        t = st.tasks.submit("volume", "Checking the grids", "volume.check", {"a_path": str(a), "b_path": str(b)},
                            client=client, quiet=True)
        return gui_tasks.answer(handler, st.tasks, t, 120)
    if route == "volume/mep_range":
        p = _cube_path(st, body.get("path"))
        job: Dict[str, Any] = {"color_path": str(p)}
        if body.get("shape"):
            lev = body.get("level")
            if isinstance(lev, bool) or not isinstance(lev, (int, float)) or lev != lev or not lev:
                raise ValueError("level: a nonzero number")
            job.update(shape_path=str(_cube_path(st, body.get("shape"))), level=float(lev))
        else:
            verts = body.get("vertices")
            if not isinstance(verts, list) or not verts or len(verts) > 3 * 200_000:
                raise ValueError("vertices: a flat list of x, y, z (Angstrom)")
            if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in verts):
                raise ValueError("vertices must be numbers")
            job["vertices"] = verts
        t = st.tasks.submit("volume", f"Potential on the surface ({p.name})", "volume.mep", job, client=client, quiet=True)
        return gui_tasks.answer(handler, st.tasks, t, 120)
    paths, coefs = _volume_operands(st, body)
    names = [x.name for x in paths]
    if route == "volume/combine":
        quality = str(body.get("quality") or "medium")
        if quality not in ("low", "medium", "high"):
            raise ValueError("quality is low, medium or high")
        t = st.tasks.submit("volume", "Volume difference", "volume.combine",
                            {"paths": [str(x) for x in paths], "coefficients": coefs, "quality": quality, "names": names},
                            client=client, finish=_volume_finish(st, None))
        return gui_tasks.answer(handler, st.tasks, t, float(body.get("wait", 0.3)))
    # volume/save: a new .CUBE beside the first operand's folder or where named; never over a file
    name = str(body.get("name") or "")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+-]{0,120}\.(CUBE|cube)", name.split("/")[-1]):
        raise ValueError("name: letters, digits, _ . + - only, ending in .CUBE")
    target = st.root / st.rel_new(Path(name))
    if os.path.lexists(target):
        raise saving.SaveConflict(f"{target.name} exists; Save Result As never writes over a file",
                                  {"conflict": "exists", "name": target.name})
    st.check_write(target, allow=("grid",))      # a new grid file (the protected kinds allow no other)
    t = st.tasks.submit("volume", f"Saving {target.name}", "volume.save",
                        {"paths": [str(x) for x in paths], "coefficients": coefs, "target": str(target),
                         "title": str(body.get("title") or "")[:120]}, client=client)
    if not st.tasks.wait(t, 300) or t.state != "done":
        return handler._json({"error": t.error or "the save did not finish", "task": t.record()}, 500)
    st.tasks.remove(t.id)
    return handler._json({"path": st.rel(target), "bytes": target.stat().st_size})


# ---------------------------------------------------------------- Compare with Input (plan B7)
def _compare_input_route(handler, query):
    """An output against the structure it started from: the .d12 of the same
    name when its atoms can be matched, else the parent CIF its provenance
    names, else the run's own first frame."""
    st = handler.state
    p = st.resolve(handler._q(query, "path"))
    after = _read_structure(st, p, -1)
    tried = []
    cands = []
    d12 = p.with_suffix(".d12")
    if d12.is_file():
        cands.append(("d12", d12))
    meta = gui_cache.read_sidecar(p) or {}
    parent = ((meta.get("provenance") or {}).get("parent")) if isinstance(meta, dict) else None
    if parent:
        try:
            pp = st.resolve(parent)
            if pp.is_file():
                cands.append(("parent", pp))
        except (PermissionError, FileNotFoundError, ValueError):
            pass
    idx = _out_index(st, p)
    for kind, path in cands:
        via = None
        try:
            before = _read_structure(st, path)
            try:
                cmp = structures.compare_payloads(before, after)
            except structures.MatchError:
                # a centered .d12 (R, F, I…) that CRYSTAL ran in its primitive cell: the
                # output's first frame is that deck as CRYSTAL read it, if the two agree
                f0 = idx.payload(0, {"kind": "out", "path": st.rel(p)})
                if kind != "d12" or not structures.same_crystal(before, f0):
                    raise
                via, before = "frame0", f0
                cmp = structures.compare_payloads(before, after)
        except (structures.MatchError, ValueError) as exc:
            tried.append(f"{path.name}: {exc}")
            continue
        cmp["symmetry_change"] = structures.symmetry_change(before, after)
        if cmp["order"]:
            after = {**after, **{k: [after[k][i] for i in cmp["order"]] for k in ("frac", "cart", "numbers", "symbols")
                                 if isinstance(after.get(k), list)}}
        info = {"kind": kind, "path": st.rel(path)}
        if via:
            info["via"] = via
            info["note"] = f"{path.name} is a centered cell ({len(_read_structure(st, path)['numbers'])} atoms); CRYSTAL ran its primitive cell, as its first frame shows"
        return handler._json({"a": before, "b": after, "comparison": cmp, "input": info, "tried": tried})
    before = idx.payload(0, {"kind": "out", "path": st.rel(p)})
    cmp = structures.compare_payloads(before, after, match="index")
    cmp["symmetry_change"] = structures.symmetry_change(before, after)
    return handler._json({"a": before, "b": after, "comparison": cmp, "input": {"kind": "frame0", "path": st.rel(p)},
                          "tried": tried})


def _on_tmpfs(path: Path) -> bool:
    try:
        best, fstype = "", ""
        rp = str(path.resolve())
        with open("/proc/mounts") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) >= 3 and (rp == parts[1] or rp.startswith(parts[1].rstrip("/") + "/")) and len(parts[1]) > len(best):
                    best, fstype = parts[1], parts[2]
        return fstype in ("tmpfs", "ramfs")
    except OSError:
        return False


def cli_task(st, kind: str, title: str, fn, doc: Optional[str] = None, client: Optional[str] = None,
             supersede: Any = None, finish=None) -> "gui_tasks.Task":
    """A task that runs ``fn(ctx)`` (a CLI child through clirun) in a
    dispatcher thread holding one slot of the CPU budget (plan T1)."""
    return st.tasks.submit(kind, title, "", {}, doc=doc, client=client, inline=fn, supersede=supersede,
                           finish=finish)


class StillWriting(Exception):
    """A properties file (BAND.DAT, DOSS.DAT, a cube, a spectrum) a job is
    still writing: not an error, the page offers Retry."""

    def __init__(self, age_s: float):
        self.age_s = age_s
        super().__init__(f"This file is still being written (it changed {age_s:.0f} s ago). It opens when it stops changing.")


def _young_s(path: Path) -> float:
    """How recent a change counts as "still being written": 10 s, 30 s on a
    network file system (metadata reaches other nodes late)."""
    try:
        from .dbsafety_loader import dbsafety
        return 30.0 if dbsafety().is_network(dbsafety().mount_of(path)) else 10.0
    except Exception:
        return 10.0


def _stable_parse(path: Path, fn, wait_s: float = 2.0):
    """``fn()``; a parse error on a file that changed during the parse or
    changed in the last 10 s (30 s on a network file system) is retried once
    after ``wait_s``, and then reported as StillWriting, never as an error.
    A stable file that fails keeps its error."""
    from mace.utils import livefile
    for attempt in range(2):
        before = livefile.snapshot(path)
        try:
            return fn()
        except Exception as exc:             # any parse failure (a cut file can end a reader's iterator)
            after = livefile.snapshot(path)
            age = time.time() - after.mtime_ns / 1e9
            moving = before.size != after.size or before.mtime_ns != after.mtime_ns or age < _young_s(path)
            if not moving:
                raise
            if attempt == 0:
                time.sleep(float(os.environ.get("MACE_GUI_STABLE_WAIT_S", wait_s)) if os.environ.get("MACE_GUI_TEST_HOOKS") == "1" else wait_s)
                continue
            raise StillWriting(age) from exc


def _live_minutes() -> float:
    try:
        return float(gui_settings.load().get("files.liveMinutes", 10))
    except (TypeError, ValueError):
        return 10.0


def _out_live(path: Path) -> bool:
    """A run still being written (no terminal marker; written within the live window)."""
    from mace.utils import livefile
    try:
        snap = livefile.snapshot(path)
        return livefile.is_live(path, snap, livefile.tail_text(path, snap), live_minutes=_live_minutes())
    except OSError:
        return False


_LAST_INDEX: Dict[str, "outindex.OutIndex"] = {}
_LAST_INDEX_LOCK = threading.Lock()


def _out_index(st, path: Path) -> "outindex.OutIndex":
    """A CRYSTAL output's byte-offset index, through the parse cache. The
    key holds the stat taken before the read (the index reads exactly that
    many bytes) and whether the run is live; a run that grew since its last
    index is extended from where that one stopped (outindex.extend)."""
    filekind.assert_not_joblog(st, path)        # a job log is never parsed for geometry (409)
    live = _out_live(path)
    key = gui_cache.file_key(path, ("outindex", outindex.VERSION, live))
    idx = st.parse_cache.get(key)
    if idx is None:
        with _LAST_INDEX_LOCK:
            prev = _LAST_INDEX.get(key[0])
        if prev is not None and prev.size <= key[2]:
            idx = outindex.extend(prev, path, size=key[2], live=live)
        else:
            idx = outindex.build(path, size=key[2], live=live)
        with _LAST_INDEX_LOCK:
            if len(_LAST_INDEX) > 16:
                _LAST_INDEX.pop(next(iter(_LAST_INDEX)))
            _LAST_INDEX[key[0]] = idx
        st.parse_cache.put(key, idx, 256 + 96 * idx.n_steps)
        st.ledger.enforce()
    return idx


def _read_structure(st, path: Path, frame: int = -1) -> Dict[str, Any]:
    """structures.read_structure, with a CRYSTAL output read through its
    index: only the requested frame's block is parsed."""
    if path.suffix.lower() in (".out", ".o", ".log"):
        idx = _out_index(st, path)
        if not idx.n_steps:
            raise filekind.NoGeometry()
        return idx.payload(frame, {"kind": "out", "path": str(path)})
    return structures.read_structure(path, frame)


def _cached_payload(st, path: Path, frame: int = -1) -> Dict[str, Any]:
    """A file's structure payload through the parse cache (as /api/structure gave it)."""
    key = gui_cache.file_key(path, ("structure", frame))
    data = st.parse_cache.get(key)
    if data is None:
        data = _read_structure(st, path, frame)
        data["path"] = st.rel(path)
        data["file"] = {"mtime_ns": str(key[1]), "size": key[2]}
        st.parse_cache.put(key, data, 2 * len(json.dumps(data, default=_json_default)))
        st.ledger.enforce()
    return data


def _atoms_of(cell, frac, numbers, pbc=(True, True, True), cart=None):
    from ase import Atoms
    import numpy as np
    nums = [int(n) % 100 or int(n) for n in numbers]
    if all(pbc):
        return Atoms(numbers=nums, cell=np.asarray(cell, float), scaled_positions=np.asarray(frac, float), pbc=True)
    return Atoms(numbers=nums, cell=np.asarray(cell, float), positions=np.asarray(cart, float), pbc=list(pbc))


class _Source:
    """A structure a builder or an export reads: by reference, never uploaded."""

    def __init__(self, atoms, name: str, kind: str, label: str, meta=None, payload=None, rev=None):
        self.atoms, self.name, self.kind, self.label = atoms, name, kind, label
        self.meta = meta or {}
        self.payload = payload
        self.rev = rev


def _source(st, src: Any) -> _Source:
    """``{path, frame}`` | ``{session, rev}`` | ``{held: id}`` -> the structure (spec plan B1)."""
    if not isinstance(src, dict):
        raise ValueError("source is {path, frame}, {session, rev} or {held}")
    if src.get("held"):
        e = st.held.get(str(src["held"]))
        p = e["structure"]
        cell, frac, z = gui_cache.unpack_arrays(p)
        pbc = tuple(p.get("pbc", [True, True, True]))
        cart = None
        if not all(pbc):
            import numpy as np, base64
            cart = np.frombuffer(base64.b64decode(p["cart"]), "<f8").reshape(-1, 3)
        name = str((e.get("meta") or {}).get("name") or "built")
        return _Source(_atoms_of(cell, frac, z, pbc, cart), name, "held", name, e.get("meta"), payload=p)
    if src.get("session"):
        sess = st.sessions.get(str(src["session"]))
        with sess.lock:
            if src.get("rev") is not None:
                sess.check_rev(src.get("rev"))
            atoms = sess.atoms.copy()
            return _Source(atoms, sess.name, "session", sess.meta.get("label") or sess.name,
                           getattr(sess, "doc_meta", None), rev=sess.rev)
    p = st.resolve(src.get("path"))
    data = _cached_payload(st, p, int(src.get("frame", -1)))
    atoms = _atoms_of(data["cell"], data["frac"], data["numbers"], tuple(data["pbc"]), data.get("cart"))
    return _Source(atoms, p.stem, "path", p.name, data.get("doc_meta"), payload=data)


class _SessLike:
    """What saving.render/save need of a session, for a structure with no session."""

    def __init__(self, atoms, name: str, symprec: float = 1e-3, doc_meta=None):
        self.atoms, self.name, self.symprec, self.rev = atoms, name, symprec, 0
        self.doc_meta = doc_meta or {}

    def cif_text(self, mode: str = "p1") -> str:
        return editor.EditSession.cif_text(self, mode)


EXPORT_FORMATS = {"cif": "chemical/x-cif", "poscar": "text/plain", "xyz": "chemical/x-xyz", "extxyz": "chemical/x-xyz"}


def _export_route(handler, body: Dict[str, Any]):
    """POST /api/structure/export: the file's bytes, for the browser to
    download to this computer (over ssh -L: the laptop). Never writes."""
    st = handler.state
    fmt = str(body.get("format") or "cif").lower()
    if fmt not in EXPORT_FORMATS:
        raise ValueError(f"format must be one of {', '.join(EXPORT_FORMATS)}")
    mode = str(body.get("cif_mode") or "p1")
    if mode not in ("p1", "symmetrized"):
        raise ValueError("CIF mode is p1 or symmetrized")
    src = _source(st, body.get("source"))
    if fmt in ("cif", "poscar") and not all(src.atoms.pbc):
        raise ValueError(f"{fmt.upper()} needs a 3D cell; export this one as extXYZ")
    rfmt = "plainxyz" if fmt == "xyz" else fmt
    if src.kind == "session":
        sess = st.sessions.get(str(body["source"]["session"]))
        with sess.lock:
            data = saving.render(sess, rfmt, mode)
    else:
        data = saving.render(_SessLike(src.atoms, src.name), rfmt, mode)
    return handler._send(200, data, EXPORT_FORMATS[fmt] + "; charset=utf-8",
                         {"Cache-Control": "no-store", "Content-Disposition": "attachment"})


# ---------------------------------------------------------------- builders (plan B1-B4)
INLINE_MAX = 2000          # result atoms built in the request thread; more go to the pool
PREVIEW_MAX = 2000         # a preview above this shows the new cell's outline only
MAX_BUILD_ATOMS = 250000
SYMM_MAX = structures.SYMMETRY_INLINE_MAX    # spglib runs inline up to this many atoms (files and builds)
BYTES_PER_ATOM = 200       # what a held structure costs (packed payload, arrays, symmetry lists)


class BuildKind:
    """One builder: its function in builders.py, what it makes (estimate),
    the keyword arguments it takes, and the title of its task."""

    def __init__(self, fn, estimate, args, title, always_pool=False, test_only=False):
        self.fn, self.estimate, self.args, self.title = fn, estimate, args, title
        self.always_pool, self.test_only = always_pool, test_only


def _identity_args(src, params):
    return {"delay_s": min(30.0, max(0.0, float(params.get("delay_ms", 0)) / 1000))}


def _need_3d(src, what: str):
    if not all(src.atoms.pbc):
        raise ValueError(f"{what} are built from 3D cells")


def _supercell_matrix(params):
    if params.get("matrix") is not None:
        return builders.check_matrix(params["matrix"])
    n = params.get("n") or [2, 2, 2]
    try:
        n = [int(x) for x in n]
    except (TypeError, ValueError):
        raise ValueError("n1 n2 n3 are whole numbers")
    if len(n) != 3 or min(n) < 1:
        raise ValueError("n1 n2 n3 are whole numbers from 1")
    return builders.check_matrix([[n[0], 0, 0], [0, n[1], 0], [0, 0, n[2]]])


def _supercell_estimate(src, params):
    import numpy as np
    _need_3d(src, "Supercells")
    m = _supercell_matrix(params)
    det = int(round(np.linalg.det(m)))
    n = det * len(src.atoms)
    if n > MAX_BUILD_ATOMS:
        raise ValueError(f"This makes {n:,} atoms; builds stop at {MAX_BUILD_ATOMS:,}")
    return {"n": n, "cell": (m @ src.atoms.cell.array).tolist(), "det": det}


def _supercell_title(src, params):
    what = params.get("label")
    if not what:
        m = _supercell_matrix(params)
        diag = all(m[i][j] == 0 for i in range(3) for j in range(3) if i != j)
        what = f"{m[0][0]}×{m[1][1]}×{m[2][2]}" if diag else "det %d" % round(__import__("numpy").linalg.det(m))
    return f"Supercell {str(what)[:40]} of {src.label}"


def _slab_args(src, params):
    return {"hkl": list(builders.check_hkl(params.get("hkl", (0, 0, 1)))), "layers": int(params.get("layers", 5)),
            "vacuum": float(params.get("vacuum", 15.0)), "termination": int(params.get("termination", 0)),
            "symmetric": bool(params.get("symmetric", False)), "center": params.get("center", True) is not False}


def _slab_estimate(src, params):
    import numpy as np
    if not all(src.atoms.pbc):
        raise ValueError("Slabs are cut from 3D bulk cells")
    a = _slab_args(src, params)
    if not 1 <= a["layers"] <= 200:
        raise ValueError("layers: 1 to 200")
    if not 0 <= a["vacuum"] <= 200:
        raise ValueError("vacuum: 0 to 200 Å")
    t = builders.slab_terminations(src.atoms.cell.array, src.atoms.get_scaled_positions(), src.atoms.numbers, a["hkl"])
    per_plane = t["atoms_per_cell"] / max(1, t["period"])
    n = int(round(per_plane * (a["layers"] + (t["period"] if a["symmetric"] else 0))))
    if n > MAX_BUILD_ATOMS:
        raise ValueError(f"This makes about {n:,} atoms; builds stop at {MAX_BUILD_ATOMS:,}")
    return {"n": n, "terminations": t["terminations"], "spacing": t["spacing"]}


def _hkl_text(hkl):
    return " ".join(str(x) for x in hkl) if any(abs(x) > 9 for x in hkl) else "".join(str(x) for x in hkl)


DEFECT_KINDS = ("vacancy", "substitution", "interstitial", "antisite")


def _defect_args(src, params):
    kind = str(params.get("kind") or "vacancy")
    if kind not in DEFECT_KINDS:
        raise ValueError("kind is vacancy, substitution, interstitial or antisite")
    m = builders.check_matrix(params.get("matrix") if params.get("matrix") is not None else [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    out = {"kind": kind, "matrix": m.tolist(), "site": int(params.get("site", 0)), "element": params.get("element")}
    if kind == "interstitial":
        out["position"] = params.get("position")
    elif not 0 <= out["site"] < len(src.atoms):
        raise ValueError(f"no atom {out['site'] + 1}")
    if kind in ("substitution", "antisite", "interstitial") and not out["element"]:
        raise ValueError("Pick an element")
    if kind in ("substitution", "antisite"):
        from ase.data import chemical_symbols
        z = int(src.atoms.numbers[out["site"]])
        if str(out["element"]) == chemical_symbols[z % 100 or z]:
            raise ValueError("Pick a different element")
    return out


def _defect_estimate(src, params):
    import numpy as np
    _need_3d(src, "Defect supercells")
    a = _defect_args(src, params)
    m = np.asarray(a["matrix"])
    det = int(round(np.linalg.det(m)))
    n = det * len(src.atoms) + {"vacancy": -1, "interstitial": 1}.get(a["kind"], 0)
    if n > MAX_BUILD_ATOMS:
        raise ValueError(f"This makes {n:,} atoms; builds stop at {MAX_BUILD_ATOMS:,}")
    return {"n": n, "cell": (m @ src.atoms.cell.array).tolist(), "det": det,
            "nearest_image": builders._shortest(m @ src.atoms.cell.array)}


def _standardize_estimate(src, params):
    import spglib
    _need_3d(src, "Cell transforms")
    kind = str(params.get("kind") or "primitive")
    if kind not in ("primitive", "conventional", "niggli"):
        raise ValueError("kind is primitive, conventional or niggli")
    if kind == "niggli":
        return {"n": len(src.atoms)}
    symprec = editor.check_symprec(params.get("symprec", 1e-3))
    a = src.atoms
    res = compute.GATE.run(None, lambda: spglib.standardize_cell((a.cell.array, a.get_scaled_positions(), a.numbers),
                                                               to_primitive=kind == "primitive", symprec=symprec))
    if res is None:
        raise ValueError(f"Couldn't standardize this cell at {symprec:g} Å")
    return {"n": len(res[2]), "cell": __import__("numpy").asarray(res[0]).tolist()}


def _symmetrize_estimate(src, params):
    import spglib
    _need_3d(src, "Space groups")
    symprec = editor.check_symprec(params.get("symprec", 1e-3))
    a = src.atoms
    res = compute.GATE.run(None, lambda: spglib.standardize_cell((a.cell.array, a.get_scaled_positions(), a.numbers),
                                                               to_primitive=False, symprec=symprec))
    if res is None:
        raise ValueError(f"spglib found no symmetry at {symprec:g} Å; try a larger tolerance")
    return {"n": len(res[2]), "cell": __import__("numpy").asarray(res[0]).tolist()}


BUILD_KINDS: Dict[str, BuildKind] = {
    "standardize": BuildKind("standardize", _standardize_estimate,
                             lambda src, params: {"kind": str(params.get("kind") or "primitive"),
                                                  "symprec": editor.check_symprec(params.get("symprec", 1e-3))},
                             lambda src, params: f"Transform {src.label} to {str(params.get('kind') or 'primitive').title()}"),
    "symmetrize": BuildKind("symmetrize", _symmetrize_estimate,
                            lambda src, params: {"symprec": editor.check_symprec(params.get("symprec", 1e-3))},
                            lambda src, params: f"Symmetrize {src.label}"),
    "defect": BuildKind("make_defect", _defect_estimate, _defect_args,
                        lambda src, params: f"{str(params.get('kind') or 'vacancy').title()} in {src.label}"),
    "slab": BuildKind("slab", _slab_estimate, _slab_args,
                      lambda src, params: f"Slab ({_hkl_text(_slab_args(src, params)['hkl'])}), {int(params.get('layers', 5))} layers of {src.label}"),
    "supercell": BuildKind("supercell", _supercell_estimate, lambda src, params: {"matrix": _supercell_matrix(params).tolist()},
                           _supercell_title),
    # fake_n: the size the build pretends to have (the frame's tests: the pool path on a small cell)
    "identity": BuildKind("identity", lambda src, params: {"n": int(params.get("fake_n") or len(src.atoms)),
                                                           "cell": src.atoms.cell.array.tolist()},
                          _identity_args, lambda src, params: f"Identity of {src.label}", test_only=True),
}


def _source_arrays(src) -> Dict[str, Any]:
    a = src.atoms
    return {"cell": a.cell.array.copy(), "frac": a.get_scaled_positions(wrap=False), "numbers": a.numbers.copy(),
            "pbc": tuple(bool(x) for x in a.pbc)}


def _built_payload(raw, name: str, sym) -> Dict[str, Any]:
    import numpy as np
    cell = np.asarray(raw["cell"], float)
    frac = np.asarray(raw["frac"], float)
    pbc = tuple(raw.get("pbc", (True, True, True)))
    cart = frac @ cell if "cart" not in raw else np.asarray(raw["cart"], float)
    return structures._payload(cell, frac, cart, raw["numbers"], pbc, {"kind": "built", "path": name}, symmetry_data=sym)


def _built_symmetry(raw, src) -> Dict[str, Any]:
    n = len(raw["numbers"])
    if not all(raw.get("pbc", (True, True, True))):
        return {"available": False, "note": "spglib space groups are for 3D cells"}
    if n > SYMM_MAX:
        parent = (src.payload or {}).get("symmetry") if src.payload else None
        group = parent.get("international") if parent and parent.get("available") else None
        return {"available": False, "not_computed": True, "parent": group,
                "note": f"supercell of {group}; not recomputed above {SYMM_MAX:,} atoms" if group
                else f"not recomputed above {SYMM_MAX:,} atoms"}
    return structures._gated_symmetry(raw["cell"], raw["frac"], raw["numbers"], raw.get("pbc", (True, True, True)))


def _provenance(st, src, kind: str, params: Dict[str, Any], label: str) -> Dict[str, Any]:
    prov = {"parent": src.label, "op": kind, "params": params, "label": label}
    if src.kind == "path" and src.payload and src.payload.get("path"):
        prov["parent"] = src.payload["path"]
        try:
            prov["parent_sha256"] = gui_cache.sha256_file(st.resolve(src.payload["path"]))
        except (PermissionError, FileNotFoundError):
            pass
    return prov


def _finish_build(st, raw, src, kind, params, open_as, client) -> Dict[str, Any]:
    """A builder's arrays -> a new document: a session at once when it is
    editable (3D, up to editor.MAX_ATOMS), else a held view-only structure."""
    from ase import Atoms
    open_as = open_as or {}
    name = _check_new_name(str(open_as.get("name") or f"{src.name}_{kind}"))
    label = str(open_as.get("label") or f"{kind.title()} of {src.label}")[:120]
    meta = {"provenance": _provenance(st, src, kind, params, label)}
    hints = open_as.get("hints")
    if isinstance(hints, dict):
        meta["hints"] = _clean_meta({"hints": hints}).get("hints", {})
    n = len(raw["numbers"])
    out: Dict[str, Any] = {"n": n, "summary": raw.get("summary", ""), "name": name, "label": label}
    for k in ("info",):
        if k in raw:
            out[k] = raw[k]
    pbc = tuple(raw.get("pbc", (True, True, True)))
    if n <= editor.MAX_ATOMS and all(pbc):
        atoms = Atoms(numbers=[int(z) for z in raw["numbers"]], cell=raw["cell"], scaled_positions=raw["frac"], pbc=True)
        sess = st.sessions.add(editor.EditSession(atoms, name, float(open_as.get("symprec", 1e-3)), label=label,
                                                  locked=bool(open_as.get("locked", False)), undo_limit=_undo_limit(),
                                                  wrap=bool(open_as.get("wrap", False))))
        sess.doc_meta = meta
        st.ledger.enforce()
        with sess.lock:
            out["session"] = sess.payload(f"{label}: {n} atoms")
        return out
    sym = _built_symmetry(raw, src)
    pl = _built_payload(raw, name, sym)
    packed = gui_cache.pack(pl, raw["frac"])
    hid = st.held.put(packed, BYTES_PER_ATOM * n, client, {**meta, "name": name})
    st.ledger.enforce()
    out.update({"held": hid, "structure": packed, "view_only": True,
                "reason": (f"Editing is limited to {editor.MAX_ATOMS} atoms; this one has {n:,}" if n > editor.MAX_ATOMS
                           else "Editing needs a 3D periodic cell")})
    return out


def _build_route(handler, kind: str, body: Dict[str, Any]):
    """POST /api/build/<kind>: {source, params, preview?, open_as?} -> a new
    structure. Never writes. A preview runs inline and returns the result
    (packed) or, above PREVIEW_MAX atoms, only the new cell. A build up to
    INLINE_MAX atoms runs in this request; a bigger one is a pool task
    (answered within 300 ms when it is done by then, else 202)."""
    st = handler.state
    spec = BUILD_KINDS.get(kind)
    if spec is None or (spec.test_only and not st.test_hooks):
        return handler._error(404, f"unknown builder {kind}")
    t0 = time.perf_counter()
    params = body.get("params") or {}
    if not isinstance(params, dict):
        raise ValueError("params must be an object")
    src = _source(st, body.get("source"))
    est = spec.estimate(src, params)           # a bad parameter is a ValueError: 400, shown in the dialog
    n = int(est["n"])
    args = {**_source_arrays(src), **spec.args(src, params)}
    fn = getattr(builders, spec.fn)
    if body.get("preview"):
        if n > PREVIEW_MAX:
            return handler._json({"outline_only": True, "n": n, "cell": est.get("cell"), "origin": est.get("origin"),
                                  "summary": est.get("summary", ""), "estimate": est},
                                 timing=f"build;dur={(time.perf_counter() - t0) * 1000:.1f}")
        raw = compute.GATE.run(None, lambda: fn(**args))
        sym = {"available": False, "note": "preview"}
        pl = _built_payload(raw, "preview", sym)
        return handler._json({"preview": gui_cache.pack(pl, raw["frac"]), "n": len(raw["numbers"]),
                              "cell": raw["cell"].tolist(), "summary": raw.get("summary", ""), "estimate": est,
                              "info": raw.get("info")},
                             timing=f"build;dur={(time.perf_counter() - t0) * 1000:.1f}")
    if n > MAX_BUILD_ATOMS:
        raise ValueError(f"this makes {n:,} atoms; builds stop at {MAX_BUILD_ATOMS:,}")
    if n > editor.MAX_ATOMS:
        st.held.check_room(BYTES_PER_ATOM * n)     # refused before it runs
    client = handler.headers.get("X-Mace-Client")
    open_as = body.get("open_as") if isinstance(body.get("open_as"), dict) else {}

    def finish(raw):
        return _finish_build(st, raw, src, kind, params, open_as, client)
    if n <= INLINE_MAX and not spec.always_pool:
        with compute.GATE.slot(pool=False):
            raw = fn(**args)
        out = finish(raw)
        return handler._json(out, timing=f"build;dur={(time.perf_counter() - t0) * 1000:.1f}")
    t = st.tasks.submit(kind, spec.title(src, params), "builders." + spec.fn, args,
                        doc=str(body.get("doc") or ""), client=client, finish=finish)
    return gui_tasks.answer(handler, st.tasks, t)


INTERSTITIAL_INLINE = 200      # atoms: a bigger interstitial search (or any matrix search) is a pool task


def _defect_info_route(handler, body: Dict[str, Any]):
    """The Point Defect dialog's facts about its structure: the inequivalent
    sites, the empty sites for an interstitial, and the smallest supercell
    whose defect images are at least min_dist apart."""
    st = handler.state
    src = _source(st, body.get("source"))
    _need_3d(src, "Point defects")
    symprec = editor.check_symprec(body.get("symprec", 1e-3))
    try:
        d = float(body.get("min_dist", 10.0))
    except (TypeError, ValueError):
        raise ValueError("min_dist is a distance in Å")
    d = min(max(d, 1.0), 60.0)
    a = _source_arrays(src)
    sites = compute.GATE.run(None, lambda: builders.defect_sites(cell=a["cell"], frac=a["frac"], numbers=a["numbers"], symprec=symprec))
    out: Dict[str, Any] = {**sites, "n_atoms": len(src.atoms)}
    want_inter = bool(body.get("interstitial"))
    matrices = bool(body.get("search_matrices"))
    n = len(src.atoms)
    if want_inter and n > builders.INTERSTITIAL_MAX:
        out["interstitial_error"] = (f"Interstitial search is limited to {builders.INTERSTITIAL_MAX} atoms; "
                                     "pick a position instead")
        want_inter = False
    if n <= INTERSTITIAL_INLINE and not matrices:
        out["supercell"] = compute.GATE.run(None, lambda: builders.defect_supercell(a["cell"], n, d))
        if want_inter:
            out.update(compute.GATE.run(None, lambda: builders.interstitial_candidates(
                cell=a["cell"], frac=a["frac"], numbers=a["numbers"], symprec=symprec)))
        return handler._json(out)
    args = {"cell": a["cell"], "frac": a["frac"], "numbers": a["numbers"], "symprec": symprec, "min_dist": d,
            "search_matrices": matrices, "interstitial": want_inter}

    def finish(raw):
        return {**out, **raw}
    t = st.tasks.submit("defect", f"Defect sites of {src.label}", "builders.defect_info", args,
                        client=handler.headers.get("X-Mace-Client"), finish=finish)
    return gui_tasks.answer(handler, st.tasks, t)


SWEEP_INLINE = 200          # atoms: a bigger symmetrize sweep is a pool task


def _sweep_route(handler, body: Dict[str, Any]):
    """Symmetrize's tolerance sweep: the group spglib finds at 13 tolerances."""
    st = handler.state
    src = _source(st, body.get("source"))
    _need_3d(src, "Space groups")
    a = _source_arrays(src)
    args = {"cell": a["cell"], "frac": a["frac"], "numbers": a["numbers"]}
    if len(src.atoms) <= SWEEP_INLINE:
        with compute.GATE.slot(pool=False):
            return handler._json(builders.symmetrize_sweep(**args))
    t = st.tasks.submit("symmetrize", f"Symmetry sweep of {src.label}", "builders.symmetrize_sweep", args,
                        client=handler.headers.get("X-Mace-Client"))
    return gui_tasks.answer(handler, st.tasks, t)


CLEAN_MAX_TIME = {"springs": 120.0, "gfnff": 600.0, "mlip": 600.0, "auto": 600.0}


def _clean_time_limit(value, method: str = "springs") -> float:
    """Clean's time limit for a proposal: 0.05 s up to the method's clamp
    (springs 120 s as before; GFN-FF, ML and Auto 600 s). Run Longer doubles it."""
    from mace.preopt import core
    kind = core.method_kind(method)
    try:
        t = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"time_limit {value!r} is not a number")
    default = core.defaults_for(method)["time_limit"]
    return min(max(t, 0.05), CLEAN_MAX_TIME[kind]) if t == t else default


def _clean_methods_skeleton() -> List[Dict[str, Any]]:
    """The method list before detection has answered (all but springs unknown)."""
    from mace.preopt import core, registry
    out = [{"id": "springs", "label": "Springs: rough, always available", "available": True, "status": "ready",
            "defaults": core.defaults_for("springs"), "max_time_limit": CLEAN_MAX_TIME["springs"]},
           {"id": "gfnff", "label": "GFN-FF", "available": False, "status": "checking",
            "defaults": core.defaults_for("gfnff"), "max_time_limit": CLEAN_MAX_TIME["gfnff"]}]
    for m in registry.load():
        out.append({"id": f"mlip:{m['id']}", "label": f"ML potential: {m['label']}", "available": False,
                    "status": "checking", "defaults": core.defaults_for("mlip:" + m["id"]),
                    "max_time_limit": CLEAN_MAX_TIME["mlip"], "unrelated_note": m.get("unrelated_note")})
    out.append({"id": "auto", "label": "Auto (best available)", "available": True, "status": "ready",
                "defaults": core.defaults_for("auto"), "max_time_limit": CLEAN_MAX_TIME["auto"]})
    return out


def _clean_method_ids() -> List[str]:
    return [m["id"] for m in _clean_methods_skeleton()]


def _detect_clean_methods(st, recheck: bool = False) -> None:
    """What Clean can offer here (xtb, the ML environment and weights);
    ``/api/clean/methods`` answers ``checking`` until this has run once."""
    from mace.preopt import core, detect, gfnff, registry
    methods = _clean_methods_skeleton()
    try:
        xt = detect.xtb_status(recheck=recheck)
    except Exception as exc:                # never take the server down
        xt = {"status": "broken", "reason": str(exc), "hint": detect.XTB_HINT, "version": None}
    for m in methods:
        if m["id"] == "gfnff":
            ok = xt["status"] == "ready"
            m.update(available=ok, status=xt["status"], version=xt.get("version"), reason=xt.get("reason", ""),
                     source=xt.get("source"), module=xt.get("module"),
                     hint=xt.get("hint", ""), commands=detect.hint_commands(xt.get("hint", "")),
                     license="LGPL-3.0 (the external xtb program)",
                     label=(f"GFN-FF (xtb {xt['version']}" + (f", module {xt['module']}" if xt.get("module") else "")
                            + ")") if ok else "GFN-FF",
                     citation=gfnff.CITATIONS, training_data=None)
        elif m["id"].startswith("mlip:"):
            model = registry.get(m["id"][5:])
            try:
                ms = detect.mlip_status(model, recheck=recheck, timeout=detect.PROBE_TIMEOUT_GUI)
            except Exception as exc:
                ms = {"status": "import_failed", "reason": str(exc), "hint": detect.setup_hint(model)}
            info = registry.provenance_info(model)
            m.update(available=ms["status"] == "ready", status=ms["status"], reason=ms.get("reason", ""),
                     hint=ms.get("hint", ""), commands=detect.hint_commands(ms.get("hint", "")),
                     version=(ms.get("versions") or {}).get(model["code"]["package"]),
                     license=f"code {info['code_license']}, weights {info['weights_license']}",
                     citation=info["citation"], data_citation=info["data_citation"],
                     training_data=info["training_data"], limits=info["limits"])
    with st.clean_methods_lock:
        st.clean_methods = {"checking": False, "methods": methods, "checked": time.time()}


def _whole(value) -> int:
    """A charge or spin from JSON: a whole number (1, -2, "3", 2.0), never
    truncated (1.5 and true are refused)."""
    if isinstance(value, bool):
        raise ValueError(value)
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(value)
        return int(value)
    return int(str(value).strip())


def _clean_status(st, method: str) -> Dict[str, Any]:
    with st.clean_methods_lock:
        for m in st.clean_methods.get("methods", []):
            if m["id"] == method:
                return dict(m)
    return {}


def _clean_threads() -> int:
    return max(1, min(4, resources.effective_workers(gui_settings.load().get("max_workers"))))


def _clean_route(handler, body: Dict[str, Any]):
    """POST /api/clean: Clean geometry's proposal, from the pool (spec §2.9,
    plan T8). Nothing is committed: the session keeps the proposal until
    edit/clean_accept (one history entry, "Clean Geometry") or
    edit/clean_discard (none).

    ``method`` (default springs: exactly the Clean that was), gfnff,
    mlip:<id> or auto, all through mace.preopt.core like ``mace preopt``;
    the charge comes from the body, else the document's charge hint; a
    document from a charged deck needs it for GFN-FF/ML/Auto, and one whose
    deck edits its geometry (SUPERCEL, ATOMREMO, ...) gets the spring model
    only, with the CLI's words."""
    from mace.preopt import core, deckscan
    st = handler.state
    if not body.get("session") and isinstance(body.get("open"), dict):
        sess, _, _ = _open_session(st, body["open"])
    else:
        sess = st.sessions.get(body.get("session"))
    method = str(body.get("method") or "springs")
    try:
        kind = core.method_kind(method)
    except ValueError as exc:
        return handler._json({"error": str(exc)}, 400)
    with sess.lock:
        sess.check_rev(body.get("base_rev"))
        before = sess.atoms.copy()
        rev, symprec = sess.rev, sess.symprec
        meta = dict(getattr(sess, "doc_meta", None) or {})
    deck = meta.get("deck") or {}
    if kind != "springs":
        if deck.get("geometry_edits"):
            return handler._json({"error": deckscan.refusal_text(deck.get("name") or f"{sess.name}.d12",
                                                                 deck["geometry_edits"]), "refused": "geometry_edits"}, 400)
    charge = body.get("charge")
    if charge is None or charge == "":
        hint = (meta.get("hints") or {}).get("charge")
        if hint is not None:
            charge = hint
        elif deck.get("charge_records") and kind != "springs":
            return handler._json({"error": deckscan.charged_text(deck["charge_records"], gui=True),
                                  "needs_charge": True}, 400)
        else:
            charge = 0
    if kind not in ("springs", "auto"):
        ms = _clean_status(st, method)
        if not ms.get("available"):
            why = ms.get("reason") or ("still checking what is installed" if st.clean_methods.get("checking")
                                       else "not available")
            return handler._json({"error": f"{ms.get('label') or method}: {why}", "hint": ms.get("hint", ""),
                                  "unavailable": True}, 400)
    try:
        charge, uhf = _whole(charge), _whole(body.get("uhf") or 0)
    except (TypeError, ValueError):
        return handler._json({"error": "charge and uhf are whole numbers"}, 400)
    if kind == "springs" and len(before) <= INLINE_MAX:
        if not gui_clean.bonded_pairs(before):
            return handler._json({"nothing_to_clean": True, "session": sess.id, "rev": rev,
                                  "message": "nothing to clean: no bonded pairs (an all-metal cell or a lone atom; "
                                             "the spring model draws no metal-metal bonds)"})
    d = core.defaults_for(method)
    limit = _clean_time_limit(body.get("time_limit", d["time_limit"]), method)
    args = {"cell": before.cell.array, "frac": before.get_scaled_positions(wrap=False), "numbers": before.numbers,
            "keep_symmetry": body.get("keep_symmetry", True) is not False, "relax_cell": bool(body.get("relax_cell")),
            "symprec": symprec, "time_limit": limit, "method": method, "charge": charge, "uhf": uhf}
    if kind != "springs":
        tol = body.get("symmetrize_tol")
        args["symmetrize_tol"] = float(tol) if tol not in (None, "", False) else None
        args["fmax"] = float(body["fmax"]) if body.get("fmax") not in (None, "") else None
        args["max_steps"] = int(body["max_steps"]) if body.get("max_steps") not in (None, "") else None
    elif body.get("max_steps") not in (None, ""):
        args["max_steps"] = int(body["max_steps"])
    if kind == "springs" and body.get("fmax") not in (None, ""):
        args["fmax"] = float(body["fmax"])
    if args.get("symmetrize_tol") is not None and not args["symmetrize_tol"] > 0:
        return handler._json({"error": "the symmetrize tolerance is a positive length in Å"}, 400)
    # the ML worker: which model this job may load (one at a time, server-owned)
    lease_model = None
    if kind == "mlip":
        lease_model = method[5:]
    elif kind == "auto" and charge == 0:
        from mace.preopt import registry
        for mid in (registry.default_id(), registry.fallback_id()):
            if mid and _clean_status(st, f"mlip:{mid}").get("available"):
                lease_model = mid
                break
    weight, acquire = 1, None
    if lease_model:
        threads = _clean_threads()
        weight = threads
        args["threads"] = threads
        acquire = (lambda t, _m=lease_model, _k=threads: st.mlip_host.acquire(t, _m, _k))

    def finish(raw):
        from ase import Atoms
        after = Atoms(numbers=raw["numbers"], cell=raw["cell"], scaled_positions=raw["frac"], pbc=True)
        comparison = structures.compare(_internal(before), _internal(after))
        rep = raw["report"]
        pre = raw.get("preopt") or {}
        g0 = structures._gated_symmetry(before.cell.array, before.get_scaled_positions(), before.numbers, (True, True, True))
        g1 = structures._gated_symmetry(after.cell.array, after.get_scaled_positions(), after.numbers, (True, True, True))
        changed = [r for r in comparison["bonds"] if abs(r["delta"]) > 0.01]
        steps = f"{rep['steps']} step{'' if rep['steps'] == 1 else 's'}"
        n_changed = sum(r["count"] for r in changed)
        name = lambda g: g["international"] if g.get("available") else "no group"  # noqa: E731
        parts = [f"max shift {comparison['max_displacement']:.3f} Å", f"RMS {comparison['rms_displacement']:.3f} Å",
                 f"{n_changed} bond{'' if n_changed == 1 else 's'} changed", f"group {name(g0)} → {name(g1)}"]
        if rep["converged"]:
            parts.append(f"converged in {steps}")
        lead = (f"Stopped at the {limit:g} s time limit after {steps} (not converged)" if rep.get("timed_out")
                else None if rep["converged"] else f"Stopped after {steps} (not converged)")
        ran = pre.get("method_ran", "springs")
        ran_label = ("springs" if ran == "springs" else "GFN-FF" if ran == "gfnff"
                     else (pre.get("label") or ran).replace("ML potential: ", ""))
        pid = secrets.token_hex(6)
        with sess.lock:
            sess.proposal = {"id": pid, "atoms": after, "report": rep, "base_rev": rev, "preopt": pre,
                             "label": (f"Clean Geometry ({steps}, max shift {comparison['max_displacement']:.3f} Å)"
                                       if kind == "springs" else
                                       f"Clean Geometry ({ran_label}, {steps}, max shift "
                                       f"{comparison['max_displacement']:.3f} Å)")}
        before_p = structures._payload(before.cell.array, before.get_scaled_positions(wrap=False), before.positions,
                                       before.numbers, (True, True, True), {"kind": "edit", "path": "before Clean"},
                                       symmetry_data=g0)
        after_p = structures._payload(after.cell.array, after.get_scaled_positions(wrap=False), after.positions,
                                      after.numbers, (True, True, True), {"kind": "edit", "path": "Clean proposal"},
                                      symmetry_data=g1)
        out = {"proposal_id": pid, "session": sess.id, "base_rev": rev, "proposal": gui_cache.pack(after_p),
               "before": gui_cache.pack(before_p), "comparison": comparison, "changed_bonds": changed,
               "report": rep, "summary": " · ".join(parts), "warning": lead, "time_limit": limit,
               "displacements": [d["vector"] for d in comparison["displacements"]],
               "before_cart": comparison["before_cart"]}
        if kind != "springs":
            warns = list(pre.get("warnings") or [])
            if deck.get("geometry_edits"):
                warns.append(deckscan.host_warning(deck.get("name") or sess.name, deck["geometry_edits"]))
            out.update(method=method, method_ran=ran, ran=f"Ran: {pre.get('summary', '')}",
                       attempts=pre.get("attempts"), fell_back=pre.get("fell_back"),
                       sg_change={"before": name(g0), "after": name(g1)}, warnings=warns,
                       provenance=pre.get("provenance"), citation=pre.get("citation"),
                       starting_geometry_note=("Starting geometry only: build a CRYSTAL OPT from it before "
                                               "SP/FREQ/BAND/DOSS") if ran != "springs" else None,
                       max_time_limit=CLEAN_MAX_TIME[kind])
        return out
    t = st.tasks.submit("clean", f"Clean Geometry of {sess.name}", "builders.clean_job", args, doc=sess.id,
                        client=handler.headers.get("X-Mace-Client"), finish=finish, weight=weight, acquire=acquire)
    return gui_tasks.answer(handler, st.tasks, t)


def _clean_provenance(sess, prop: Dict[str, Any]) -> None:
    """On Accept: the method that ran, its model and license, the training
    data, the charge and the citation go into the document's provenance (and
    with it the saved CIF's sidecar), never into a deck title; a GFN-FF/ML
    result also marks the document (doc_meta["preopt"])."""
    pre = prop.get("preopt") or {}
    if not pre or pre.get("method_ran", "springs") == "springs" and pre.get("method_requested", "springs") == "springs":
        return
    prov = pre.get("provenance") or {}
    meta = dict(getattr(sess, "doc_meta", None) or {})
    params = {k: prov.get(k) for k in ("method_ran", "model", "model_label", "code", "code_version", "code_license",
                                         "weights_license", "training_data", "charge", "citation", "fell_back")}
    meta["provenance"] = {**(meta.get("provenance") or {}), "op": "clean", "params": params,
                          "label": prop.get("label", "Clean Geometry")}
    if pre.get("method_ran") not in (None, "springs"):
        meta["preopt"] = {"method_ran": pre["method_ran"], "model": prov.get("model")}
    sess.doc_meta = meta


def _symmetry_compute_route(handler, body: Dict[str, Any]):
    """Inspector › Symmetry › Compute now, for a built structure over SYMM_MAX
    atoms: spglib as a cancellable pool task; the held copy keeps the result."""
    st = handler.state
    src = _source(st, body.get("source"))
    _need_3d(src, "Space groups")
    args = {**_source_arrays(src), "symprec": editor.check_symprec(body.get("symprec", 1e-3))}
    hid = (body.get("source") or {}).get("held")
    spath = (body.get("source") or {}).get("path")

    def finish(raw):
        if spath:                     # a big file: its cached structure keeps the group, so a reopen shows it
            try:
                p = st.resolve(spath)
                data = _cached_payload(st, p, int((body.get("source") or {}).get("frame", -1)))
                data["symmetry"] = raw["symmetry"]
            except Exception:
                pass
        if hid:
            try:
                e = st.held.get(hid)
                e["structure"] = {**e["structure"], "symmetry": raw["symmetry"]}
            except KeyError:
                pass
        return raw
    t = st.tasks.submit("symmetry", f"Space group of {src.label}", "builders.symmetry_of", args,
                        client=handler.headers.get("X-Mace-Client"), finish=finish)
    return gui_tasks.answer(handler, st.tasks, t)


def _open_parsed(st, data: Dict[str, Any], name: str, label: str, fmt: str, client,
                 extra_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """A parsed file as a new untitled document: a session (3D, up to
    editor.MAX_ATOMS atoms) or a held view-only structure."""
    import re as _re
    import numpy as np
    stem = _re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(name).stem).strip("._-")[:60] or "structure"
    if not _re.match(r"[A-Za-z0-9]", stem):
        stem = "s" + stem
    meta = {"provenance": {"parent": name, "op": "clipboard" if label.startswith("Imported") else "drop", "label": label}}
    if extra_meta:
        meta.update(_clean_meta(extra_meta))
    n = len(data["numbers"])
    out = {"format": fmt, "format_name": structures.FORMAT_NAMES.get(fmt, fmt), "n": n, "name": stem}
    if n <= editor.MAX_ATOMS and all(data["pbc"]) and n:
        try:
            atoms = editor.atoms_from_payload(data)
        except editor.EditError:
            atoms = None
        if atoms is not None:
            sess = st.sessions.add(editor.EditSession(atoms, stem, 1e-3, label=label, undo_limit=_undo_limit()))
            sess.doc_meta = meta
            st.ledger.enforce()
            with sess.lock:
                out["session"] = sess.payload(label)
            return out
    if not n:
        raise ValueError(f"{name} holds no atoms")
    nbytes = BYTES_PER_ATOM * n
    st.held.check_room(nbytes)
    pl = dict(data)
    pl["doc_meta"] = meta
    packed = gui_cache.pack(pl)
    hid = st.held.put(packed, nbytes, client, {**meta, "name": stem})
    st.ledger.enforce()
    out.update({"held": hid, "structure": packed, "view_only": True,
                "reason": (data.get("symmetry") or {}).get("note") if data.get("no_cell") else
                (f"Editing is limited to {editor.MAX_ATOMS} atoms; this one has {n:,}" if n > editor.MAX_ATOMS
                 else "Editing needs a 3D periodic cell")})
    return out


def _save_route(handler, body: Dict[str, Any]):
    """POST /api/structure/save: the one save path (see saving.py)."""
    st = handler.state
    if not body.get("session") and isinstance(body.get("source"), dict) and body["source"].get("held"):
        # a view-only built structure (no session): Save As writes it like any other
        src = _source(st, body["source"])
        sl = _SessLike(src.atoms, src.name, doc_meta=src.meta)
        out = saving.save(st, sl, str(body.get("path") or ""), str(body.get("format") or "cif"),
                          str(body.get("cif_mode") or "p1"), overwrite=bool(body.get("overwrite")),
                          backup=body.get("backup", True) is not False, expected=body.get("expected"))
        out["held"] = body["source"]["held"]
        return handler._json(out)
    if not body.get("session") and isinstance(body.get("open"), dict):
        sess, _, _ = _open_session(st, body["open"])     # Save As of a document never edited
    else:
        sess = st.sessions.get(body.get("session"))
    expected = body.get("expected")
    if expected is not None and not isinstance(expected, dict):
        raise ValueError("expected is {mtime_ns, size}")
    with sess.lock:
        try:
            sess.check_rev(body.get("base_rev"))
        except editor.StaleRev as exc:
            return _stale(handler, sess, exc)
        out = saving.save(st, sess, str(body.get("path") or ""), str(body.get("format") or "cif"),
                          str(body.get("cif_mode") or "p1"), overwrite=bool(body.get("overwrite")),
                          backup=body.get("backup", True) is not False, expected=expected)
    out["session"] = sess.id
    return handler._json(out)


def _count(v) -> int:
    try:
        return min(100000, max(1, int(v)))
    except (TypeError, ValueError, OverflowError):
        return 1


def _stale(handler, sess, exc):
    """409: the edit was made against an older revision; nothing was applied.
    The browser adopts the payload and drops its queued edits."""
    return handler._json({"conflict": "rev", "rev": exc.rev, "error": str(exc),
                          "payload": sess.payload("the structure changed elsewhere")}, 409)


CSP = ("default-src 'self'; img-src 'self' data: blob:; "
       "style-src 'self' 'unsafe-inline'; script-src 'self'; "
       "frame-src 'self'; connect-src 'self'; object-src 'none'; "
       "base-uri 'none'; frame-ancestors 'none'")
# the right editor group's page: only this server's own page may frame it
CSP_PANE = CSP.replace("frame-ancestors 'none'", "frame-ancestors 'self'")


def _prefs_attrs() -> str:
    """The <html> attributes for first paint, from the stored settings, fixed
    whitelists only (a hand-edited gui.json is never reflected as markup)."""
    st = gui_settings.load()
    out = ['data-prefs="server"', f'data-prefs-rev="{int(gui_settings.load_profile().get("rev", 0))}"']
    if st.get("theme") in ("light", "dark"):
        out.append(f'data-theme="{st["theme"]}"')
    if st.get("density") == "compact":
        out.append('data-density="compact"')
    motion = {"on": "reduce", "off": "full"}.get(st.get("reduceMotion"))
    if motion:
        out.append(f'data-motion="{motion}"')
    scale = st.get("uiScale")
    scale = scale if isinstance(scale, int) and not isinstance(scale, bool) and 80 <= scale <= 150 else 100
    out.append(f'style="--ui-scale:{scale / 100:g}"')
    return " ".join(out)


class _WriteCounter:
    """Profile, settings and per-document writes in flight: a stopping server
    waits for them (at most 2 s), so a change saved as the page closed lands."""

    def __init__(self):
        self.n, self.cond = 0, threading.Condition()

    def __enter__(self):
        with self.cond:
            self.n += 1
        return self

    def __exit__(self, *exc):
        with self.cond:
            self.n -= 1
            self.cond.notify_all()
        return False

    def drain(self, timeout: float = 2.0) -> bool:
        end = time.monotonic() + timeout
        with self.cond:
            while self.n > 0:
                left = end - time.monotonic()
                if left <= 0:
                    return False
                self.cond.wait(left)
        return True


_profile_writes = _WriteCounter()


def _profile_view(st, everything: bool = False) -> Dict[str, Any]:
    """The profile for a page: this folder's Recent list only, unless the export asks for all."""
    prof = gui_settings.load_profile()
    files = prof["recent"]["files"]
    if not everything:
        files = {k: v for k, v in files.items() if k == st.dir_hash}
    out = {**prof, "recent": {"files": files, "folders": prof["recent"]["folders"]},
           "path": str(gui_settings.path()), "dir_hash": st.dir_hash}
    if everything:
        out["docprefs"] = gui_docprefs.all_of()
    return out


def _docprefs_rel(st, rel: Any, write: bool = False) -> str:
    """A document's path inside the served folder (no other folder's entries).
    A write needs the document to exist, so a stray name never makes an entry."""
    if not isinstance(rel, str) or not rel:
        raise ValueError("path is required")
    try:
        target = st.resolve(rel, must_exist=False)
    except PermissionError:
        raise ValueError("path must be inside the served folder") from None
    if write and not target.is_file():
        raise ValueError("no such document in the served folder")
    return target.relative_to(st.root).as_posix()


def _home() -> str:
    """The server's home folder (the root is a server path): the page shows it as ~.
    Resolved like the root, so a home reached through a symbolic link still matches."""
    try:
        return str(Path.home().resolve())
    except (RuntimeError, KeyError):   # no HOME and no passwd entry
        return ""


def _json_default(obj):
    try:
        import numpy as np
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:  # pragma: no cover
        pass
    return str(obj)


def _finish_runs(st, say=print) -> None:
    """Server shutdown: a submission in flight is waited for (it cannot stop
    halfway; a second Ctrl+C stops waiting, the CLI child finishes in its
    own session), and each Run on This Node is stopped as its Cancel stops
    it (one SIGTERM, mace run's own grace, then SIGKILL)."""
    import signal as _signal
    stop_waiting = []
    old = None
    if threading.current_thread() is threading.main_thread():
        try:
            old = _signal.signal(_signal.SIGINT, lambda *a: stop_waiting.append(1))
        except (ValueError, OSError):
            old = None
    try:
        t0 = time.monotonic()
        next_say = 0.0
        while st.submits.running > 0 and not stop_waiting:
            waited = time.monotonic() - t0
            if waited >= next_say:
                try:
                    say(f"mace gui: waiting for a submission to finish ({int(waited)} s)", flush=True)
                except OSError:
                    pass
                next_say += 10.0
            time.sleep(0.2)
        from mace.submission.local_run import KILL_GRACE
        local = [t for t in list(st.tasks.tasks.values()) if t.kind == "local" and t.state in ("queued", "running")]
        for t in local:
            st.tasks.cancel(t.id)
        for t in local:
            t.done.wait(KILL_GRACE + 15)
    finally:
        if old is not None:
            try:
                _signal.signal(_signal.SIGINT, old)
            except (ValueError, OSError):
                pass


class GuiHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer with a cap on the connections served at once.

    Each connection holds a thread. Past ``max_connections`` the oldest
    connection that has not yet sent an authenticated request is closed to
    make room (an idle socket from another user cannot crowd out the
    browser); when every connection is authenticated, the new one is closed.
    Sockets that send nothing are also dropped after the handler timeout."""

    daemon_threads = True
    max_connections = MAX_CONNECTIONS

    def __init__(self, *args, **kwargs):
        self._conns: "OrderedDict[Any, bool]" = OrderedDict()     # socket -> authenticated
        self._conns_lock = threading.Lock()
        self.refused = 0
        super().__init__(*args, **kwargs)

    def mark_authenticated(self, sock) -> None:
        with self._conns_lock:
            if sock in self._conns:
                self._conns[sock] = True

    def open_connections(self) -> int:
        with self._conns_lock:
            return len(self._conns)

    def process_request(self, request, client_address):
        victim = None
        with self._conns_lock:
            if len(self._conns) >= self.max_connections:
                victim = next((s for s, authed in self._conns.items() if not authed), None)
                if victim is None:
                    self.refused += 1
                    self.shutdown_request(request)
                    return
                del self._conns[victim]
            self._conns[request] = False
        if victim is not None:
            self.refused += 1
            try:
                victim.shutdown(socket.SHUT_RDWR)      # its thread sees EOF and ends
            except OSError:
                pass
        super().process_request(request, client_address)

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            with self._conns_lock:
                self._conns.pop(request, None)


def is_loopback(host: str) -> bool:
    return host.lower() in LOOPBACK_NAMES


def network_names(host: str) -> List[str]:
    """The Host names a browser may use for a non-loopback bind: the address
    itself, or for a wildcard bind (0.0.0.0, ::) this node's names and addresses."""
    if host not in ("0.0.0.0", "::", ""):
        return [host.lower()]
    names = set()
    for n in (socket.gethostname(), socket.getfqdn()):
        if n:
            names.add(n.lower())
            try:
                hn, aliases, addrs = socket.gethostbyname_ex(n)
                names.update(x.lower() for x in [hn, *aliases, *addrs] if x)
            except OSError:
                pass
    return sorted(names - set(LOOPBACK_NAMES))


def make_server(root: Path, host: str = "127.0.0.1", port: int = 8765,
                token: Optional[str] = None, allowed_hosts: Optional[List[str]] = None,
                verbose: bool = False, roots: Optional[List[str]] = None) -> Tuple[ThreadingHTTPServer, GuiState]:
    hooks = os.environ.get("MACE_GUI_TEST_HOOKS") == "1"
    if hooks and not is_loopback(host):
        # never on an address other users or machines can reach (the modules
        # that read the variable themselves, and the pool's workers, see it gone)
        print("mace gui: MACE_GUI_TEST_HOOKS is ignored when --host is not loopback", file=sys.stderr)
        os.environ.pop("MACE_GUI_TEST_HOOKS", None)
        hooks = False
    elif hooks:
        print("mace gui: test hooks enabled (MACE_GUI_TEST_HOOKS=1)", file=sys.stderr)
    state = GuiState(root, token or secrets.token_urlsafe(24), allowed_hosts, test_hooks=hooks, roots=roots)
    state.bind_host = host
    memory.limit_arenas(2)        # section 9.2: what the ledger frees is not kept per thread
    # spglib in the request threads: at most the section 9 worker cap at once
    compute.GATE.resize(resources.effective_workers(gui_settings.load().get("max_workers")))
    if not is_loopback(host):
        state.allowed_hosts.update(network_names(host))
    handler = type("BoundGuiHandler", (GuiHandler,), {"state": state, "verbose": verbose})
    family_server = GuiHTTPServer
    if ":" in host:
        class _V6(GuiHTTPServer):
            address_family = socket.AF_INET6
        family_server = _V6
    httpd = family_server((host, port), handler)
    shutdown = httpd.shutdown

    def stop():  # end the event streams too, or shutdown() waits on them
        state.stopping.set()
        if getattr(state, "cluster", None) is not None:
            state.cluster.stop()
        shutdown()
    httpd.shutdown = stop
    def detect():
        try:
            from mace.preopt import proc
            proc.sweep()
            _detect_clean_methods(state)
        except Exception as exc:          # pragma: no cover - detection never stops the server
            with state.clean_methods_lock:
                state.clean_methods = {"checking": False, "methods": _clean_methods_skeleton(), "error": str(exc)}
    threading.Thread(target=detect, name="mace-clean-detect", daemon=True).start()
    try:                              # the one own-job poller (only where SLURM is on PATH)
        from . import cluster_api
        cluster_api.start(state)
    except Exception as exc:          # pragma: no cover - cluster monitoring never stops the server
        print(f"mace gui: cluster monitoring is off: {exc}", file=sys.stderr)
    if os.environ.get("MACE_GUI_NO_WARMUP") != "1":
        timer = threading.Timer(1.0, warm_up)
        timer.daemon = True
        timer.start()
    return httpd, state
