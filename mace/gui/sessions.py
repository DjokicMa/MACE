"""Browser sessions of MACE GUI (plan PLAN_connect D16): one-time sign-in
links, session cookies, opt-in activity, expiry and the idle stop.

* The printed link carries a **one-time sign-in code** (``?login=``), never
  the token: the first use swaps it for a session cookie; the same URL from
  the browser's history signs nobody in. At most MAX_CODES unused codes are
  kept (the oldest goes first); a code is valid until used or the server stops.
* The master token works only in the ``X-Mace-Token`` header (scripts,
  ``mace gui connect``) and in the sign-in form; never in a URL or a cookie.
* The cookie holds a random session id. A session expires after
  ``session.idleHours`` without **input** (default 12; null = never) or
  ``session.maxDays`` after sign-in (default 7). Only POST
  ``api/session/touch`` (sent from keyboard, pointer and wheel input, at
  most once a minute) counts as input: polls, the event stream and autosave
  never extend a session.
* Wrong credentials (a wrong header token, form token or code) wait 1 s each
  inside one server-wide lock; at most MAX_WAITING wait, beyond that 429 at
  once. Correct credentials are checked first and never wait, so nobody can
  lock the real user out.
* The idle stop (``server.idleStopHours``, off by default): the server stops
  after that long without input from any session and with no task running.

Standard library only.
"""
from __future__ import annotations

import hmac
import os
import secrets
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

MAX_CODES = 8
MAX_SESSIONS = 64
MAX_WAITING = 16
WRONG_DELAY_S = 1.0
TOUCH_EVERY_S = 60.0
IDLE_CHOICES = (1, 4, 12, 24, 72)
DEFAULT_IDLE_HOURS = 12
DEFAULT_MAX_DAYS = 7
POLICY_TTL_S = 5.0
IDLE_WARN_S = 300.0


def _env_hours(name: str) -> Tuple[bool, Optional[float]]:
    """(set, hours) from an environment override; "never"/"off"/"0" = None."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return False, None
    if raw.strip().lower() in ("never", "off", "none", "0"):
        return True, None
    try:
        v = float(raw)
        return True, (v if v > 0 else None)
    except ValueError:
        return False, None


SLURM_IDLE_STOP_HOURS = 1.0

class Sessions:
    def __init__(self, token: str, settings_loader=None):
        self.token = token
        self.lock = threading.Lock()
        self.codes: "OrderedDict[str, float]" = OrderedDict()
        self.table: "OrderedDict[str, Dict[str, float]]" = OrderedDict()
        self.wrong_lock = threading.Lock()
        self.waiting = 0
        self.waiting_lock = threading.Lock()
        self.settings_loader = settings_loader
        self._policy: Optional[Dict[str, Any]] = None
        self._policy_t = 0.0
        self.started = time.time()
        self.last_input = time.time()
        self.revoked = False

    # -- policy (gui.json, environment overrides) --------------------------------
    def policy(self) -> Dict[str, Any]:
        now = time.monotonic()
        if self._policy is not None and now - self._policy_t < POLICY_TTL_S:
            return self._policy
        stored: Dict[str, Any] = {}
        if self.settings_loader is not None:
            try:
                stored = self.settings_loader() or {}
            except Exception:
                stored = {}
        idle = stored.get("session.idleHours", DEFAULT_IDLE_HOURS)
        is_set, env_idle = _env_hours("MACE_GUI_SESSION_IDLE_HOURS")
        if is_set:
            idle = env_idle
        days = stored.get("session.maxDays", DEFAULT_MAX_DAYS)
        try:
            env_days = float(os.environ.get("MACE_GUI_SESSION_MAX_DAYS", ""))
            if env_days > 0:
                days = env_days
        except ValueError:
            pass
        stop = stored.get("server.idleStopHours")
        if stop is None and os.environ.get("MACE_GUI_SLURM"):
            # a batch job (mace gui --slurm): an idle allocation spends your fairshare
            stop = SLURM_IDLE_STOP_HOURS
        is_set, env_stop = _env_hours("MACE_GUI_IDLE_STOP_HOURS")
        if is_set:
            stop = env_stop
        self._policy = {"idle_hours": idle, "max_days": days, "idle_stop_hours": stop}
        self._policy_t = now
        return self._policy

    def forget_policy(self) -> None:
        self._policy = None

    # -- one-time sign-in codes ------------------------------------------------
    def mint_code(self) -> str:
        code = secrets.token_urlsafe(18)
        with self.lock:
            self.codes[code] = time.time()
            while len(self.codes) > MAX_CODES:
                self.codes.popitem(last=False)
        return code

    def use_code(self, code: str) -> Optional[str]:
        """A new session id for a valid unused code, else None (the code is spent)."""
        if not code or self.revoked:
            return None
        with self.lock:
            hit = next((c for c in self.codes if hmac.compare_digest(c.encode(), code.encode())), None)
            if hit is None:
                return None
            del self.codes[hit]
        return self.new_session()

    # -- sessions -----------------------------------------------------------------
    def new_session(self) -> str:
        sid = secrets.token_urlsafe(24)
        now = time.time()
        with self.lock:
            self.table[sid] = {"created": now, "last_input": now}
            while len(self.table) > MAX_SESSIONS:
                self.table.popitem(last=False)
        self.last_input = now
        return sid

    def token_ok(self, candidate: Optional[str]) -> bool:
        return bool(candidate) and hmac.compare_digest(candidate.encode(), self.token.encode())

    def state_of(self, sid: Optional[str]) -> str:
        """"ok", "expired" or "none" for a session cookie value."""
        if not sid or self.revoked:
            return "none"
        with self.lock:
            rec = self.table.get(sid)
        if rec is None:
            return "none"
        p, now = self.policy(), time.time()
        idle = p.get("idle_hours")
        if idle and now - rec["last_input"] > float(idle) * 3600:
            return "expired"
        days = p.get("max_days")
        if days and now - rec["created"] > float(days) * 86400:
            return "expired"
        return "ok"

    def touch(self, sid: Optional[str]) -> bool:
        with self.lock:
            rec = self.table.get(sid or "")
            if rec is None:
                return False
            rec["last_input"] = time.time()
        self.last_input = time.time()
        return True

    def revoke_all(self) -> None:
        with self.lock:
            self.table.clear()
            self.codes.clear()
        self.revoked = True

    def count(self) -> int:
        with self.lock:
            return len(self.table)

    # -- wrong credentials: paced, never blocking a correct one -----------------------
    def wrong(self) -> int:
        """Called after a wrong credential: wait WRONG_DELAY_S inside the one lock,
        then 403; 429 at once when MAX_WAITING already wait."""
        with self.waiting_lock:
            if self.waiting >= MAX_WAITING:
                return 429
            self.waiting += 1
        try:
            with self.wrong_lock:
                time.sleep(WRONG_DELAY_S)
        finally:
            with self.waiting_lock:
                self.waiting -= 1
        return 403

    # -- idle stop -----------------------------------------------------------------
    def idle_stop_due(self, running: int) -> Tuple[bool, Optional[float]]:
        """(stop now?, seconds left) for the idle-stop rule; (False, None) when it is off."""
        hours = self.policy().get("idle_stop_hours")
        if not hours:
            return False, None
        left = float(hours) * 3600 - (time.time() - self.last_input)
        if running:
            return False, max(left, 0.0)
        return left <= 0, left
