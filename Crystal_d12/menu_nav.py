"""
menu_nav.py -- opt-in "press b to go back" navigation for interactive CRYSTAL menus.

Design (validated by the multi-agent review + a passing standalone prototype):

  * MenuBack(BaseException): raised when the user requests "back" at an interceptable
    prompt. It subclasses BaseException (NOT Exception) so an `except Exception:` (and the
    handful of post-input conversion guards in the menus) can never silently swallow the
    signal -- it always reaches run_with_back.

  * NavSession: records exactly ONE entry per real input() call (the raw returned string)
    and replays them on demand. Per-input()-call granularity is what makes the re-prompt /
    validation while-loops inside the shared helpers replay deterministically.

  * run_with_back(flow_fn): runs flow_fn inside a context that temporarily monkeypatches
    builtins.input with a record/replayer. On MenuBack it pops the last recorded answer and
    re-runs flow_fn, replaying the now-shorter prefix so the user lands LIVE one step back.
    builtins.input and sys.stdout are always restored in a finally.

  * INERT BY DEFAULT. Two layers of safety guarantee zero regression outside wrapped flows:
      - the controller only does anything while a flow is wrapped (a contextvar gates it);
        any input() elsewhere in the app calls the genuine builtins.input unchanged.
      - run_with_back is a pass-through no-op when stdin is not a TTY (non-interactive /
        config-file / batch runs), so automated paths behave exactly as before.

  * SENTINEL SCOPING (the critical hardening). 'b'/'back' is intercepted ONLY through
    nav_read(prompt, valid_set=...) and ONLY when the prompt carries a known valid_set that
    excludes 'b'/'back'. Raw input() (free-text: k-path labels like the 'B' high-symmetry
    point, filenames, coordinates) goes through the record/replay patch with NO interception,
    so a legitimate 'b'/'B' answer is never stolen. The shared menu helpers, which already
    know their valid set, call nav_read; free-text prompts keep using plain input().

This module is dependency-free so it can be imported from Crystal_d12, Crystal_d3 and mace.
"""

import builtins
import contextvars
import io
import sys


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------
class MenuBack(BaseException):
    """Raised from an interceptable prompt when the user types a back token.

    Subclasses BaseException on purpose: `except Exception:` will NOT catch it, so the
    back request always propagates out to run_with_back regardless of any broad guards
    around post-input parsing in the menu code.
    """
    pass


# ---------------------------------------------------------------------------
# State (contextvar-gated => inert unless a flow is wrapped)
# ---------------------------------------------------------------------------
_active_session = contextvars.ContextVar("menu_nav_active_session", default=None)

# The genuine builtins.input, captured once. Reads always go through this (referenced as a
# module global so tests can substitute a scripted feeder).
_REAL_INPUT = builtins.input

_BACK_TOKENS = frozenset({"b", "back"})

# Test/escape hatch: force the controller on even when stdin is not a TTY. Production code
# never sets this; the self-tests do, so they can exercise back navigation without a tty.
_FORCE_ENABLE = False


def _enabled():
    """The feature is live only at an interactive terminal (or when forced for tests)."""
    if _FORCE_ENABLE:
        return True
    try:
        return sys.stdin.isatty()
    except Exception:
        return False


class NavSession:
    """Records raw input() return strings and replays them on demand.

    Granularity: exactly one entry per input() invocation (not per logical question).
    """

    def __init__(self):
        self.records = []          # list[str] raw strings returned by real input()
        self.replay_index = 0      # how many records replayed so far this pass
        self.replaying = False     # True while feeding recorded answers back
        self.silencer = None       # the active _ReplaySilencer (set by run_with_back)
        self.pending_default = ""   # value popped on the last "back", shown as a hint once
        self._has_pending_default = False

    # -- replay control -----------------------------------------------------
    def begin_replay(self):
        self.replay_index = 0
        self.replaying = True

    def can_go_back(self):
        return len(self.records) > 0

    def pop_last(self):
        return self.records.pop()

    # -- the patched input() body ------------------------------------------
    def handle_input(self, prompt, intercept):
        """Replacement body for input() while a flow is wrapped.

        Replay phase: return the next recorded string (no terminal read).
        Live phase:   read the real input; if intercept is on and the user typed a back
                      token (and back is available), raise MenuBack WITHOUT recording, so a
                      single pop in run_with_back unwinds exactly one prior answer.
        """
        # ---- Replay: hand back the next recorded answer ----
        if self.replaying and self.replay_index < len(self.records):
            value = self.records[self.replay_index]
            self.replay_index += 1
            if self.silencer is not None:
                self.silencer.drop()  # the just-answered question's output is not re-shown
            return value

        # ---- Transition to live reads ----
        if self.replaying:
            self.replaying = False
            if self.silencer is not None:
                self.silencer.go_live()  # flush the landed question's preamble

        shown = prompt
        if self._has_pending_default:
            shown = f"{prompt} (previously: {self.pending_default})"
            self._has_pending_default = False
            self.pending_default = ""

        value = _REAL_INPUT(shown)

        if intercept and value.strip().lower() in _BACK_TOKENS and self.can_go_back():
            raise MenuBack()

        self.records.append(value)
        return value


class _ReplaySilencer(io.TextIOBase):
    """stdout proxy that buffers writes during replay and reveals only the LANDED question.

    During replay every print() is buffered. On each replayed answer the buffer is dropped
    (that question is already answered, no need to re-show it). When the flow reaches the
    first live read, the buffer -- which now holds only the current question's preamble --
    is flushed, and all subsequent writes pass straight through. This keeps "go back" from
    re-scrolling the whole menu while still showing the header of the question you land on.
    """

    def __init__(self, real, session):
        self._real = real
        self._session = session
        self._buf = []

    def write(self, s):
        if self._session.replaying:
            self._buf.append(s)
            return len(s)
        return self._real.write(s)

    def drop(self):
        self._buf.clear()

    def go_live(self):
        if self._buf:
            self._real.write("".join(self._buf))
            self._buf.clear()

    def flush(self):
        try:
            self._real.flush()
        except Exception:
            pass

    def isatty(self):
        try:
            return self._real.isatty()
        except Exception:
            return False


def _patched_input(prompt=""):
    """Installed onto builtins.input for the duration of a wrapped flow.

    Records/replays every input() call but performs NO back interception (intercept=False),
    so raw free-text input() inside a wrapped flow can never have a 'b'/'back' answer stolen.
    Outside a wrapped flow the contextvar is None and the genuine input() is called => no-op.
    """
    session = _active_session.get()
    if session is None:
        return _REAL_INPUT(prompt)
    return session.handle_input(prompt, intercept=False)


def nav_read(prompt="", valid_set=None):
    """Read one answer, with scoped back interception.

    Use this (instead of input()) for prompts that have a KNOWN set of valid answers.
    'b'/'back' is treated as a back request only when:
        a flow is wrapped, AND valid_set is provided, AND valid_set excludes 'b'/'back'.
    For free-text prompts pass valid_set=None (or just use input()): 'b' is never stolen.

    When no flow is wrapped this is exactly input(prompt) => zero behaviour change.
    """
    session = _active_session.get()
    if session is None:
        return _REAL_INPUT(prompt)

    intercept = False
    if valid_set is not None:
        lowered = {str(v).strip().lower() for v in valid_set}
        if not (_BACK_TOKENS & lowered):
            intercept = True

    shown = prompt
    if intercept and back_available_live():
        shown = f"{prompt} [b=back]"
    return session.handle_input(shown, intercept=intercept)


def run_with_back(flow_fn):
    """Run flow_fn with "press b to go back" navigation enabled.

    Returns flow_fn()'s result once it completes. If stdin is not a TTY the controller is a
    pure pass-through (flow_fn() is called once, unpatched). builtins.input and sys.stdout
    are restored on every exit path.
    """
    if not _enabled():
        return flow_fn()

    session = NavSession()
    previous_input = builtins.input
    builtins.input = _patched_input
    token = _active_session.set(session)
    real_stdout = sys.stdout
    try:
        while True:
            session.begin_replay()
            silencer = _ReplaySilencer(real_stdout, session)
            session.silencer = silencer
            sys.stdout = silencer
            try:
                result = flow_fn()
            except MenuBack:
                sys.stdout = real_stdout
                if session.can_go_back():
                    session.pending_default = session.pop_last()
                    session._has_pending_default = True
                continue
            else:
                return result
            finally:
                if sys.stdout is silencer:
                    sys.stdout = real_stdout
                session.silencer = None
    finally:
        _active_session.reset(token)
        builtins.input = previous_input
        sys.stdout = real_stdout


def back_available_live():
    """True iff we are at a LIVE prompt with at least one earlier answer to go back to.

    This is the correct predicate for deciding whether to show a 'b to go back' affordance:
    it is False during replay and False at the very first question of a flow.
    """
    session = _active_session.get()
    if session is None:
        return False
    if session.replaying and session.replay_index < len(session.records):
        return False
    return session.can_go_back()
