#!/usr/bin/env python3
"""MACE visual layer — the single UI facade.

This module is the **only** place in MACE that imports ``rich``. Every call site
depends on ``ui.*`` (e.g. ``ui.banner()``, ``ui.progress()``, ``ui.ok()``), never
on ``rich`` directly, so ``rich`` can be swapped or removed by editing this one
file. It mirrors the existing ``BANNER_AVAILABLE`` ``try/except ImportError``
pattern in ``mace_cli`` and generalizes it across four presentation surfaces:

    1. startup banner          (``banner``/``credits``)
    2. real progress           (``progress``/``progress_task``/``spinner``/``captured``)
    3. monitor/status dashboard (``status_dashboard``/``badge``/``live_dashboard``)
    4. consistent CLI styling   (``ok``/``info``/``warn``/``err``/``rule``/``table``/``print``)

Every public function has two code paths — a rich path and a stdlib path — and
**none raises on missing rich**. When rich is absent, output is piped/redirected,
``NO_COLOR`` is set, or ``TERM=dumb``, the fallback emits clean plain text with no
ANSI escape sequences and no full-screen clear. The string ``\\033[2J\\033[H`` is
banned, and there is no fake ``time.sleep`` delay.

This module is the design + API contract for the visual layer: the docstrings
on each public function below are the reference for look & feel.
"""

from __future__ import annotations

import io
import os
import random
import re
import sys
import contextlib
from typing import Any, Callable, Iterable, Iterator, NamedTuple, Optional, Sequence

# ---------------------------------------------------------------------------
# rich is optional. Importing it must never break a call site.
# ---------------------------------------------------------------------------
try:  # mirrors mace_cli's BANNER_AVAILABLE pattern
    from rich.console import Console as _RichConsole, Group as _Group
    from rich.live import Live as _Live
    from rich.text import Text as _Text
    from rich.align import Align as _Align
    from rich.panel import Panel as _Panel
    from rich.rule import Rule as _Rule
    from rich.table import Table as _Table
    from rich.progress import (
        Progress as _Progress,
        ProgressColumn as _ProgressColumn,
        SpinnerColumn as _SpinnerColumn,
        BarColumn as _BarColumn,
        TextColumn as _TextColumn,
        MofNCompleteColumn as _MofNCompleteColumn,
        TimeRemainingColumn as _TimeRemainingColumn,
        TimeElapsedColumn as _TimeElapsedColumn,
    )
    from rich.markup import escape as _escape

    _RICH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised via monkeypatched sys.modules
    _RICH_AVAILABLE = False

    def _escape(s):  # rich-absent fallback; the styled paths that call it need rich
        return s


# ===========================================================================
# Palettes (§2.3) — frozen records: crystal, mono, ember, viridis, ocean.
# ===========================================================================
class Palette(NamedTuple):
    """A named color scheme. ``gradient`` = 6 hex stops for the MACE wordmark."""

    name: str
    gradient: tuple  # 6 hex strings
    accent: str
    bar: str
    ok: str
    warn: str
    err: str
    spinner: str = "dots"


CRYSTAL = Palette(
    name="crystal",
    gradient=("#2dd4bf", "#22d3ee", "#38bdf8", "#3b82f6", "#6366f1", "#818cf8"),
    accent="#38bdf8",
    bar="#22d3ee",
    ok="#34d399",
    warn="#fbbf24",
    err="#f87171",
    spinner="dots",
)

MONO = Palette(
    name="mono",
    gradient=("#525252", "#6b6b6b", "#858585", "#a3a3a3", "#c4c4c4", "#ededed"),
    accent="#d4d4d4",
    bar="#a3a3a3",
    ok="#e5e5e5",
    warn="#a3a3a3",
    err="#737373",
    spinner="line",
)

# Optional warm theme — opt in via ``--theme ember`` or ``MACE_THEME=ember``.
EMBER = Palette(
    name="ember",
    gradient=("#7f1d1d", "#b91c1c", "#ea580c", "#f59e0b", "#fbbf24", "#fde68a"),
    accent="#f59e0b",
    bar="#ea580c",
    ok="#a3e635",
    warn="#fbbf24",
    err="#ef4444",
    spinner="dots",
)

# Optional matplotlib-viridis theme — familiar to materials scientists.
VIRIDIS = Palette(
    name="viridis",
    gradient=("#440154", "#414487", "#2a788e", "#22a884", "#7ad151", "#fde725"),
    accent="#22a884",
    bar="#2a788e",
    ok="#7ad151",
    warn="#fde725",
    err="#fd7e14",
    spinner="dots",
)

# Optional deep-blue monochromatic theme — calm, low-contrast.
OCEAN = Palette(
    name="ocean",
    gradient=("#0c4a6e", "#075985", "#0369a1", "#0891b2", "#06b6d4", "#67e8f9"),
    accent="#06b6d4",
    bar="#0891b2",
    ok="#2dd4bf",
    warn="#fcd34d",
    err="#fb7185",
    spinner="dots",
)

_PALETTES = {"crystal": CRYSTAL, "mono": MONO, "ember": EMBER,
             "viridis": VIRIDIS, "ocean": OCEAN}

# MACE wordmark block art (reused from banner.py / the demos).
WORDMARK = [
    "███╗   ███╗ █████╗  ██████╗███████╗",
    "████╗ ████║██╔══██╗██╔════╝██╔════╝",
    "██╔████╔██║███████║██║     █████╗  ",
    "██║╚██╔╝██║██╔══██║██║     ██╔══╝  ",
    "██║ ╚═╝ ██║██║  ██║╚██████╗███████╗",
    "╚═╝     ╚═╝╚═╝  ╚═╝ ╚═════╝╚══════╝",
]

_SUBTITLE = "Mendoza Automated CRYSTAL Engine"


# ===========================================================================
# Capability detection (§2.4 / §3) — computed once, cached, override via configure()
# ===========================================================================
class _Caps:
    """Mutable holder for detected/overridden capabilities + the live console."""

    def __init__(self):
        self.rich_available = _RICH_AVAILABLE
        self.force_tty: Optional[bool] = None
        self.force_color: Optional[bool] = None
        self.palette_name: Optional[str] = None  # explicit override
        self.no_banner: Optional[bool] = None
        self._console = None  # lazily built rich Console

    # -- environment-derived booleans -------------------------------------
    def _detect_tty(self) -> bool:
        if self.force_tty is not None:
            return self.force_tty
        try:
            return bool(sys.stdout.isatty())
        except Exception:
            return False

    def _no_color_env(self) -> bool:
        # no-color.org: ANY non-empty NO_COLOR disables color.
        return bool(os.environ.get("NO_COLOR"))

    def _dumb_term(self) -> bool:
        return os.environ.get("TERM") == "dumb"

    @property
    def is_tty(self) -> bool:
        return self._detect_tty()

    @property
    def color_ok(self) -> bool:
        """Whether colored output is permitted.

        Precedence (highest first):
          1. The environment can *forbid* color and is authoritative: if
             ``NO_COLOR`` is set to ANY value (no-color.org) or ``TERM == "dumb"``,
             color is OFF — even when ``force_color=True``. The NO_COLOR standard
             wins over an explicit force.
          2. ``force_color=False`` always forces plain (color OFF).
          3. ``force_color=True`` enables color only when rich is available and the
             environment does not forbid it (rule 1).
          4. Otherwise (no override): color is on for a rich, color-capable TTY.

        rich must be importable for color in every case; without it color is OFF.
        """
        # Rule 1: environment veto is authoritative, regardless of force_color.
        if self._no_color_env() or self._dumb_term():
            return False
        if self.force_color is not None:
            # Color can only be on if rich is present and not env-forbidden (above).
            return bool(self.force_color) and self.rich_available
        if not self.rich_available:
            return False
        return self.is_tty

    @property
    def interactive(self) -> bool:
        """rich-interactive mode: rich + TTY + color."""
        return self.rich_available and self.is_tty and self.color_ok

    @property
    def palette(self) -> Palette:
        # Precedence: explicit override (configure / --theme) → MACE_THEME env →
        # saved config (~/.config/mace) → auto default (crystal in color, else mono).
        # Exactly one palette per process; the banner captures it ONCE before
        # rich.Live (which redirects stdout and would otherwise flip color_ok->mono).
        name = self.palette_name
        if name not in _PALETTES:
            name = os.environ.get("MACE_THEME", "").strip().lower()
        if name not in _PALETTES:
            name = load_saved_theme() or ""
        if name in _PALETTES:
            return _PALETTES[name]
        return CRYSTAL if self.color_ok else MONO

    def banner_suppressed(self) -> bool:
        if self.no_banner is not None:
            return bool(self.no_banner)
        return os.environ.get("MACE_NO_BANNER", "").lower() in ("1", "true", "yes")

    # -- the shared rich console ------------------------------------------
    def console(self):
        """Return a cached rich Console, or ``None`` when rich is unusable.

        ``force_terminal``/``no_color`` are pinned from our detection so the
        console never emits box-drawing or color escapes into a redirected file.
        """
        if not self.rich_available:
            return None
        # Rebuild if caps changed in a way that affects the console.
        want_color = self.color_ok
        want_tty = self.is_tty or bool(self.force_color)
        signature = (want_color, want_tty)
        if self._console is None or getattr(self._console, "_mace_sig", None) != signature:
            self._console = _RichConsole(
                force_terminal=want_tty if (want_tty or self.force_tty is not None) else None,
                no_color=not want_color,
                highlight=False,
                soft_wrap=False,
                file=sys.stdout,
            )
            # stash our signature so we know when to rebuild
            self._console._mace_sig = signature
        return self._console


_CAPS = _Caps()


# ---------------------------------------------------------------------------
# Setup / capability (§5.1)
# ---------------------------------------------------------------------------
def configure(
    *,
    palette: Optional[str] = None,
    force_tty: Optional[bool] = None,
    force_color: Optional[bool] = None,
    no_banner: Optional[bool] = None,
) -> None:
    """Override auto-detected capabilities (tests / explicit CLI flags).

    ``palette`` ∈ {"crystal", "mono"}. ``None`` args leave auto-detection intact.
    No-op-safe: if rich is absent, color is forced off regardless of args.
    """
    if palette is not None:
        if palette not in _PALETTES:
            raise ValueError(
                f"unknown palette {palette!r} — choose from {', '.join(_PALETTES)}"
            )
        _CAPS.palette_name = palette
    if force_tty is not None:
        _CAPS.force_tty = bool(force_tty)
    if force_color is not None:
        _CAPS.force_color = bool(force_color)
    if no_banner is not None:
        _CAPS.no_banner = bool(no_banner)
    # Invalidate the cached console so the next render reflects new caps.
    _CAPS._console = None


def is_interactive() -> bool:
    """True only in rich + TTY + color mode."""
    return _CAPS.interactive


def active_palette() -> Palette:
    """The current palette record; ``mono`` whenever color is off."""
    return _CAPS.palette


def palette_names() -> list:
    """The names of all selectable palettes (for ``--theme`` choices / help text).

    Single source of truth so the CLI's theme choices never drift from
    ``_PALETTES`` (which is exactly the bug that left viridis/ocean unselectable)."""
    return list(_PALETTES)


# ---------------------------------------------------------------------------
# Persisted theme — survives across runs via a small user config file.
# Precedence at render time (see _Caps.palette): --theme/configure > MACE_THEME
# env > THIS saved theme > auto (crystal if color else mono).
# ---------------------------------------------------------------------------
def _config_path() -> str:
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(
        os.path.expanduser("~"), ".config")
    return os.path.join(base, "mace", "config.json")


_UNSET = object()
_saved_theme = _UNSET  # process cache: None = nothing saved, str = a palette name


def load_saved_theme() -> Optional[str]:
    """Return the theme persisted by :func:`save_theme`, or ``None``.

    Read once and cached for the process. Never raises — a missing/garbled config
    file just yields ``None`` (auto default).
    """
    global _saved_theme
    if _saved_theme is not _UNSET:
        return _saved_theme
    name = None
    try:
        import json
        with open(_config_path(), encoding="utf-8") as f:
            val = json.load(f).get("theme")
        if isinstance(val, str) and val.strip().lower() in _PALETTES:
            name = val.strip().lower()
    except Exception:
        name = None
    _saved_theme = name
    return name


def save_theme(name: str) -> str:
    """Persist ``name`` as the default UI theme; return the config file path.

    Merges into any existing config JSON (so other future settings survive).
    Raises ``ValueError`` on an unknown theme name.
    """
    name = (name or "").strip().lower()
    if name not in _PALETTES:
        raise ValueError(f"unknown theme {name!r} — choose from {', '.join(_PALETTES)}")
    import json
    path = _config_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {}
    try:
        with open(path, encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            data = loaded
    except Exception:
        data = {}
    data["theme"] = name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    global _saved_theme
    _saved_theme = name
    return path


# ===========================================================================
# Surface 4 helpers first — message styling is used by the others.
# ===========================================================================
# Rich-style markup tokens, matched conservatively so plain prose survives:
#   * closing tags: ``[/]``, ``[/bold]``, ``[/bold red]`` (any ``[/...]``)
#   * opening tags: a style-shaped body — space-separated atoms, each a lowercase
#     style word / dotted style (``progress.percentage``) / hex color (``#34d399``),
#     optionally a ``not``/``on`` modifier. Bodies that look like prose are skipped:
#     ``[OK]`` (uppercase), ``[1, 2, 3]`` (comma) and ``[WARN]`` are NOT stripped,
#     mirroring rich's own ``Text.from_markup`` behavior for unrecognized tokens.
_MARKUP_TOKEN = re.compile(
    r"""
    \[
      (?:
        /[^\[\]]*                       # any closing tag: [/], [/bold], ...
        |                               # -- or --
        (?:[a-z][\w.-]*|\#[0-9a-fA-F]{3,8})   # opening: first atom (style word or hex)
        (?:\s+(?:[a-z][\w.-]*|\#[0-9a-fA-F]{3,8}))*   # extra space-separated atoms
      )
    \]
    """,
    re.VERBOSE,
)


def _strip_markup(text: str) -> str:
    """Remove rich-style ``[tag]`` / ``[/tag]`` / ``[/]`` tokens via stdlib regex.

    Used as the rich-absent fallback for :func:`_plain` so literal markup never
    leaks into plain output. Leaves the wrapped text intact and is conservative:
    bracketed prose that isn't tag-shaped (``[1, 2, 3]``, ``[OK]``/``[WARN]``
    markers) is left untouched, matching rich's own parser.
    """
    return _MARKUP_TOKEN.sub("", text)


def _plain(text: str) -> str:
    """Strip rich markup so the plain path never leaks ``[bold]`` etc.

    When rich is present we use its markup parser (the source of truth). When rich
    is ABSENT we fall back to a stdlib regex (:func:`_strip_markup`) so style tokens
    still don't leak literally into plain output.
    """
    if _RICH_AVAILABLE:
        try:
            return _Text.from_markup(str(text)).plain
        except Exception:
            return _strip_markup(str(text))
    return _strip_markup(str(text))


def print(*objects: Any, **kwargs: Any) -> None:  # noqa: A001 (shadow builtin on purpose)
    """Single shared-console pass-through.

    Fallback: builtin print with rich markup stripped (no ANSI, no escapes).
    """
    console = _CAPS.console()
    if console is not None and _CAPS.color_ok:
        # Treat string args as LITERAL data by default — callers pass filenames,
        # formulas, error text etc., not markup. Opt into markup with markup=True.
        kwargs.setdefault("markup", False)
        console.print(*objects, **kwargs)
        return
    # Plain path: route to the requested file (default stdout). Strings pass
    # through RAW — the rich path above prints them literally (markup=False),
    # so stripping bracket tokens here mangled data ('file [with] brackets')
    # and made the same call render differently by environment.
    file = kwargs.pop("file", sys.stdout)
    kwargs.pop("style", None)
    kwargs.pop("markup", None)
    kwargs.pop("highlight", None)
    kwargs.pop("justify", None)
    import builtins

    builtins.print(*objects, file=file, **kwargs)


def _emit(marker_markup: str, marker_plain: str, text: str, *, file=None) -> None:
    """Emit one status line in either the rich or plain style."""
    file = file or sys.stdout
    console = _CAPS.console()
    if console is not None and _CAPS.color_ok:
        # Build a console bound to the right file (stdout/stderr) for routing.
        target = console if file is sys.stdout else _RichConsole(
            file=file, force_terminal=_CAPS.is_tty or bool(_CAPS.force_color),
            no_color=not _CAPS.color_ok, highlight=False,
        )
        # The marker is our own trusted markup; the caller's text is data -> escape
        # it so brackets ([/], [Errno 2], a path with [..]) never raise MarkupError
        # or get silently swallowed by the rich markup parser.
        target.print(f"{marker_markup} {_escape(text)}")
    else:
        import builtins

        # Text is DATA here too: print it raw. Stripping bracket tokens mangled
        # filenames/messages containing [..] in exactly the piped/log output
        # where fidelity matters most (the rich path above preserves them).
        builtins.print(f"{marker_plain} {text}", file=file)


def ok(text: str) -> None:
    """Success line: ``✓`` green (stdout)."""
    p = _CAPS.palette
    _emit(f"[bold {p.ok}]✓[/]", "[OK]", text, file=sys.stdout)


def info(text: str) -> None:
    """Info line: ``i`` accent (stdout)."""
    p = _CAPS.palette
    _emit(f"[bold {p.accent}]i[/]", "[i]", text, file=sys.stdout)


def warn(text: str) -> None:
    """Warning line: ``!`` yellow (stderr)."""
    p = _CAPS.palette
    _emit(f"[bold {p.warn}]![/]", "[WARN]", text, file=sys.stderr)


def err(text: str) -> None:
    """Error line: ``✗`` red (stderr)."""
    p = _CAPS.palette
    _emit(f"[bold {p.err}]✗[/]", "[ERROR]", text, file=sys.stderr)


def rule(title: str = "") -> None:
    """Horizontal section divider, optionally titled."""
    console = _CAPS.console()
    if console is not None and _CAPS.color_ok:
        p = _CAPS.palette
        console.print(_Rule(_escape(title), style=p.accent) if title else _Rule(style=p.accent))
        return
    width = _plain_width()
    title = _plain(title)
    import builtins

    if title:
        bar = "-" * max(3, (width - len(title) - 2))
        builtins.print(f"-- {title} {bar}"[:width])
    else:
        builtins.print("-" * width)


def _plain_width(default: int = 80) -> int:
    try:
        return max(20, min(default, os.get_terminal_size().columns))
    except Exception:
        return default


def table(
    columns: Sequence[str],
    rows: Iterable[Sequence[Any]],
    *,
    title: Optional[str] = None,
) -> None:
    """Render a result table.

    Fallback: aligned plain-text columns (two-space gutter), title as a heading,
    no box characters.
    """
    rows = [list(r) for r in rows]
    console = _CAPS.console()
    if console is not None and _CAPS.color_ok:
        p = _CAPS.palette
        t = _Table(
            title=f"[bold {p.accent}]{_escape(title)}[/]" if title else None,
            header_style=f"bold {p.accent}",
            border_style="grey37",
            padding=(0, 1),
        )
        for col in columns:
            t.add_column(_escape(str(col)))
        for r in rows:
            t.add_row(*[_escape(str(c)) for c in r])
        console.print(t)
        return
    # Plain aligned columns. Match the rich path, which auto-extends the table with
    # a blank header when a row carries MORE cells than declared columns -- render
    # the overflow rather than silently dropping it (data loss).
    cols = [str(c) for c in columns]
    str_rows = [[_plain(str(c)) for c in r] for r in rows]
    ncols = max([len(cols)] + [len(r) for r in str_rows])
    header_cells = list(cols) + [""] * (ncols - len(cols))
    widths = [len(c) for c in header_cells]
    for r in str_rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))
    import builtins

    if title:
        builtins.print(_plain(title))
    header = "  ".join(c.ljust(widths[i]) for i, c in enumerate(header_cells))
    builtins.print(header)
    builtins.print("  ".join("-" * w for w in widths))
    for r in str_rows:
        builtins.print("  ".join(
            (r[i] if i < len(r) else "").ljust(widths[i]) for i in range(ncols)
        ))


# ===========================================================================
# Surface 1 — startup banner (§5.2)
# ===========================================================================
# ---------------------------------------------------------------------------
# Startup animations (§5.2). PALETTE-CORRECT: every color is derived from the
# passed ``grad`` (= ``_CAPS.palette.gradient``) so mono NEVER leaks a crystal
# color. ``banner()`` picks one at random each launch.
#
# Import-safe with rich absent: the bodies reference ``_Text`` only at CALL time,
# and they are only called from the interactive path (which requires rich). The
# module-level names below use only stdlib + ``WORDMARK``.
# ---------------------------------------------------------------------------
_WM_H = len(WORDMARK)
_WM_W = max(len(_l) for _l in WORDMARK)
_DECODE_SYM = "█▓▒░╔╗║═╬◆◇▪▫01∎⬡"


def _center_in_block(content_width: int, block: int):
    """Leading/trailing space counts to center ``content_width`` within ``block``."""
    lead = max(0, (block - content_width) // 2)
    return lead, max(0, block - content_width - lead)


def _wm_compose(grad, style_fn, char_fn=None, block=None):
    """Build the wordmark as a rich Text from per-cell style/char callbacks.

    Every row is padded to a CONSTANT ``block`` width (the wordmark centered within
    it) and the Text uses the DEFAULT justify — NOT ``justify="left"``, which would
    let the console pad each line out to the full terminal width. Constant-width
    lines are what make the banner survive a terminal resize without reflowing
    (the bug: console-width lines wrap when the terminal is shrunk).
    """
    block = block or _WM_W
    lead, trail = _center_in_block(_WM_W, block)
    t = _Text(no_wrap=True, overflow="crop")
    for r, line in enumerate(WORDMARK):
        t.append(" " * lead)
        for c in range(_WM_W):
            ch = line[c] if c < len(line) else " "
            cc = char_fn(r, c, ch) if char_fn else ch
            t.append(cc, style=style_fn(r, c, ch))
        t.append(" " * trail)
        if r < _WM_H - 1:
            t.append("\n")
    return t


def _wm_final(grad, block=None):
    return _wm_compose(grad, lambda r, c, ch: None if ch == " " else grad[r], block=block)


def _gen_shimmer(grad, block=None):
    for f in range(_WM_W + 14):
        pos = f - 7

        def sf(r, c, ch, pos=pos, grad=grad):
            if ch == " ":
                return None
            d = abs(c - pos)
            if d <= 1:
                return "bold white"
            if d <= 3:
                return f"bold {grad[r]}"
            return grad[r]

        yield _wm_compose(grad, sf, block=block)


def _gen_decode(grad, block=None):
    lock = {c: 5 + c * 0.6 for c in range(_WM_W)}
    for f in range(int(max(lock.values())) + 5):

        def cf(r, c, ch, f=f):
            if ch == " ":
                return " "
            return ch if f >= lock[c] else random.choice(_DECODE_SYM)

        def sf(r, c, ch, f=f, grad=grad):
            if ch == " ":
                return None
            # scramble color derives from the ACTIVE palette (never hardcoded) so
            # mono never flashes a crystal color; locked cells use the row gradient.
            return grad[r] if f >= lock[c] else random.choice([grad[1], grad[4], "grey50"])

        yield _wm_compose(grad, sf, cf, block=block)


def _gen_phonon(grad, block=None):
    # Vibrate the lattice horizontally inside a CONSTANT-width canvas. Each row is
    # ``block`` wide: a fixed outer lead centers the _WM_W+2*pad canvas in the block,
    # then the glyphs sit at offset ``pad ± jitter`` within it. At rest (lead == pad)
    # it lines up exactly with the settled _wm_final, so settling causes no jump, and
    # the constant block width means a terminal resize never reflows the frame.
    pad = 3
    block = block or (_WM_W + 2 * pad)
    base_lead, base_trail = _center_in_block(_WM_W + 2 * pad, block)
    for f in range(42):
        amp = max(0.0, 3.0 - f * 0.09)
        t = _Text(no_wrap=True, overflow="crop")
        for r, line in enumerate(WORDMARK):
            off = int(round(random.uniform(-amp, amp)))
            lead = max(0, min(2 * pad, off + pad))
            t.append(" " * (base_lead + lead))
            t.append(line.ljust(_WM_W), style=grad[r])
            t.append(" " * (base_trail + (2 * pad - lead)))
            if r < _WM_H - 1:
                t.append("\n")
        yield t
    yield _wm_final(grad, block=block)


_BANNER_ANIMS = {
    "phonon":  (_gen_phonon, 0.045),
    "decode":  (_gen_decode, 0.05),
    "shimmer": (_gen_shimmer, 0.028),
}


def _meta_text(version: str, meta: Optional[str]) -> str:
    """The banner's bottom credits line (author/affiliation), default or override."""
    return (meta if meta is not None
            else f"v{version}  ·  Michigan State University  ·  Mendoza Group")


def _banner_min_width(version: str, subtitle: str, meta: Optional[str]) -> int:
    """Minimum console width to render the full banner without cropping anything.

    The banner renders fully only when its WIDEST element fits. The credits/meta
    line is wider than the wordmark, so it sets the floor: below it the credits get
    truncated (``overflow="crop"``) while the logo still shows — which is what looks
    "broken" when the terminal is shrunk. Keyed off the actual text so a custom meta
    or a longer version string raises the floor accordingly.
    """
    return max(_WM_W, len(subtitle), len(_meta_text(version, meta)))


def _wm_settled(version: str, subtitle: str, meta: Optional[str], palette: Palette):
    """The settled banner: gradient wordmark + subtitle + credits as ONE constant-
    width ``Text`` block.

    The wordmark and subtitle are centered WITHIN the credits-width block (so the
    logo sits centered over the author/affiliation line — "centered on the text",
    not on the terminal), and the block is printed at its natural width (default
    justify, left-placed). Because every line is the SAME fixed width regardless of
    terminal size, shrinking the terminal down to the block width never reflows or
    breaks the logo — the previous console-centered layout padded each line to the
    full terminal width, which wrapped on resize.

    ``palette`` is passed in (captured BEFORE rich.Live) and never re-read from
    ``_CAPS`` here: Live redirects stdout, flipping the auto palette to mono.
    """
    block = _banner_min_width(version, subtitle, meta)
    t = _wm_final(palette.gradient, block=block)        # block-wide gradient art rows
    t.append("\n\n")
    s_lead, s_trail = _center_in_block(len(subtitle), block)
    t.append(" " * s_lead); t.append(subtitle, style="bold"); t.append(" " * s_trail)
    t.append("\n")
    meta_text = _meta_text(version, meta)
    m_lead, m_trail = _center_in_block(len(meta_text), block)
    t.append(" " * m_lead); t.append(meta_text, style="dim"); t.append(" " * m_trail)
    return t


def banner(
    version: str,
    *,
    subtitle: str = _SUBTITLE,
    meta: Optional[str] = None,
    animate: bool = True,
) -> None:
    """Render the MACE wordmark adaptively.

    ``animate=True`` (default; the startup banner): in rich + TTY + color, play ONE
    random animation from {phonon, decode, shimmer} via ``rich.Live`` **in place**
    (no clear, no forced delay beyond the frames), settling on a SINGLE final frame
    (wordmark + subtitle + meta).
    ``animate=False`` (the static help/credits/version banner): render that SAME
    themed wordmark **statically** — no animation — so the logo still gets the
    crystal/mono gradient instead of plain ASCII art.
    In both modes, when stdout is piped (not a TTY) or rich is missing, only a single
    concise line ``MACE v{version} — {subtitle}`` is printed (never ANSI art into a
    redirect). Honors ``MACE_NO_BANNER`` / ``--no-banner`` (returns when suppressed).
    """
    if _CAPS.banner_suppressed():
        return

    def _concise_line() -> None:
        # Concise, no escapes — for pipes / SLURM / no-color / no-rich.
        line = f"MACE v{version} — {subtitle}"
        if not _CAPS.color_ok:
            import builtins

            builtins.print(_plain(line))
        else:
            _CAPS.console().print(line)

    if animate:
        # Animated startup banner — unchanged behavior.
        if not _CAPS.interactive:
            _concise_line()
            return
        console = _CAPS.console()
        # Width guard: render the full banner only when the WIDEST element — the
        # credits/meta line, which is wider than the wordmark — fits. Below that the
        # credits crop (overflow="crop") while the logo still shows, which is what
        # looks broken on a shrunk terminal; show the concise line instead.
        if console.width < _banner_min_width(version, subtitle, meta) + 2:
            _concise_line()
            return
        p = _CAPS.palette
        import time as _time
        gen, dt = _BANNER_ANIMS[random.choice(list(_BANNER_ANIMS))]
        # Render every frame at a CONSTANT block width (left-placed, NOT console-
        # centered): console-centering padded each line to the terminal width, which
        # reflowed/broke the logo when the terminal was shrunk. The block-width frames
        # share the settle frame's geometry, so settling causes no jump.
        block = _banner_min_width(version, subtitle, meta)
        with _Live(console=console, refresh_per_second=60, transient=False) as live:
            for frame in gen(p.gradient, block):
                live.update(frame)
                _time.sleep(dt)
            # Pass the palette captured BEFORE Live: re-reading _CAPS.palette here
            # would flip to mono (Live redirected stdout -> isatty() False).
            live.update(_wm_settled(version, subtitle, meta, p))
        return

    # Static themed logo: gradient wordmark in a TTY (no animation), concise line
    # when piped / rich-absent. A no-color TTY renders the wordmark as plain ASCII
    # art (the console's no_color drops the gradient styles).
    if not _RICH_AVAILABLE or not _CAPS.is_tty:
        _concise_line()
        return
    console = _CAPS.console()
    if console.width < _banner_min_width(version, subtitle, meta) + 2:  # credits wouldn't fit -> concise
        _concise_line()
        return
    console.print(_wm_settled(version, subtitle, meta, _CAPS.palette))


def credits() -> None:
    """Print the static MACE credits block (reuses ``banner.py::get_credits()``)."""
    try:
        from mace.utils.banner import get_credits

        text = get_credits()
    except Exception:  # pragma: no cover - banner.py always importable in repo
        text = (
            "\n" + "=" * 60 + "\n"
            "Developed at Michigan State University\n"
            "Mendoza Group - Materials Science & Engineering\n"
            + "=" * 60 + "\n"
        )
    console = _CAPS.console()
    if console is not None and _CAPS.color_ok:
        console.print(text, markup=False, highlight=False)
    else:
        import builtins

        builtins.print(text)


# ===========================================================================
# Surface 2 — capture helper (§4 / §5.3)
# ===========================================================================
@contextlib.contextmanager
def captured(stderr: bool = False) -> Iterator[io.StringIO]:
    """Redirect ``sys.stdout`` (and optionally ``sys.stderr``) into a buffer.

    Yields the buffer so the caller can inspect/forward swallowed worker output.
    Pure stdlib — identical in every mode.
    """
    buf = io.StringIO()
    if stderr:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            yield buf
    else:
        with contextlib.redirect_stdout(buf):
            yield buf


# ---------------------------------------------------------------------------
# Progress columns
# ---------------------------------------------------------------------------
if _RICH_AVAILABLE:

    class _RateColumn(_ProgressColumn):
        """Live throughput; rich computes ``task.speed`` for us."""

        def __init__(self, unit: str = "it"):
            self.unit = unit
            super().__init__()

        def render(self, task):
            speed = task.finished_speed or task.speed
            if not speed:
                return _Text(f"—  {self.unit}/s", style="dim")
            return _Text(f"{speed:,.0f} {self.unit}/s", style=f"bold {_CAPS.palette.accent}")

    def _bar_columns(unit: str, eta: bool = False):
        p = _CAPS.palette
        cols = [
            _SpinnerColumn(style=p.accent, spinner_name=p.spinner),
            _TextColumn("[bold]{task.description}"),
            _BarColumn(bar_width=None, complete_style=p.bar, finished_style=p.ok),
            _TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            _MofNCompleteColumn(),
            _TextColumn("·"),
            _RateColumn(unit),
        ]
        if eta:
            cols += [_TextColumn("· eta"), _TimeRemainingColumn()]
        return cols


# ---------------------------------------------------------------------------
# TaskHandle — manual progress control (works in both modes).
# ---------------------------------------------------------------------------
class TaskHandle:
    """Handle for manual progress loops. ``.advance(n)`` / ``.update(...)``.

    In the rich path it drives a real ``Progress`` task; in the fallback it emits
    milestone counter lines (no ``\\r``, no escapes).
    """

    def __init__(self, *, rich_progress=None, task_id=None, description="",
                 total=None, unit="it", milestone_pct=10):
        self._prog = rich_progress
        self._task = task_id
        self._description = description
        self._total = total
        self._unit = unit
        self._completed = 0
        self._milestone_pct = milestone_pct
        self._next_milestone = milestone_pct
        if self._prog is None:
            # plain start line
            tot = f"/{total}" if total else ""
            self._milestone(f"{description}: starting (0{tot} {unit})")

    # -- internal plain emitter -------------------------------------------
    def _milestone(self, text: str) -> None:
        import builtins

        builtins.print(_plain(text), file=sys.stdout)

    def _maybe_plain_milestone(self) -> None:
        if self._total:
            pct = 100 * self._completed / self._total
            crossed = False
            while pct >= self._next_milestone and self._next_milestone <= 100:
                crossed = True
                self._next_milestone += self._milestone_pct
            if crossed:
                self._milestone(
                    f"{self._description}: {self._completed}/{self._total} "
                    f"({pct:.0f}%)"
                )
        else:
            # indeterminate: emit every 100 items
            if self._completed and self._completed % 100 == 0:
                self._milestone(f"{self._description}: {self._completed} {self._unit}")

    def advance(self, n: int = 1) -> None:
        self._completed += n
        if self._prog is not None and self._task is not None:
            self._prog.advance(self._task, n)
        else:
            self._maybe_plain_milestone()

    def update(self, *, completed: Optional[int] = None,
               description: Optional[str] = None) -> None:
        if description is not None:
            self._description = description
        if completed is not None:
            self._completed = completed
        if self._prog is not None and self._task is not None:
            kwargs = {}
            if completed is not None:
                kwargs["completed"] = completed
            if description is not None:
                kwargs["description"] = _escape(description)  # data -> escape (markup column)
            if kwargs:
                self._prog.update(self._task, **kwargs)
        elif completed is not None:
            self._maybe_plain_milestone()


@contextlib.contextmanager
def progress_task(
    description: str,
    *,
    total: Optional[int] = None,
    unit: str = "it",
    eta: bool = False,
) -> Iterator[TaskHandle]:
    """Manual-control progress for loops that can't be a simple ``for``.

    Yields a :class:`TaskHandle`. Fallback: advance/update print milestone lines.
    """
    if _CAPS.interactive:
        prog = _Progress(*_bar_columns(unit, eta=eta and total is not None),
                         console=_CAPS.console())
        with prog:
            task_id = prog.add_task(_escape(description), total=total)  # data -> escape
            yield TaskHandle(rich_progress=prog, task_id=task_id,
                             description=description, total=total, unit=unit)
    else:
        handle = TaskHandle(description=description, total=total, unit=unit)
        yield handle
        # final milestone
        tot = f"/{total}" if total else ""
        import builtins

        builtins.print(_plain(f"{description}: done ({handle._completed}{tot} {unit})"),
                       file=sys.stdout)


def progress(
    iterable: Iterable,
    *,
    total: Optional[int] = None,
    description: str = "",
    unit: str = "it",
    eta: bool = False,
    capture: bool = True,
) -> Iterator:
    """Wrap an iterable, yielding items while driving a live progress bar.

    Drives spinner + description + bar + percent + M/N + live rate + optional ETA.
    Each item is yielded inside :func:`captured` when ``capture=True`` so worker
    ``print()``s don't corrupt the bar. ``total=None`` → indeterminate spinner.

    Fallback (quiet/plain): iterates transparently, emitting milestone counter
    lines (start / every ~10% / done); no ``\\r``, no escapes. Items are unchanged.
    """
    if total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except (TypeError, AttributeError):
            total = None

    if _CAPS.interactive:
        return _progress_rich(iterable, total, description, unit, eta, capture)
    return _progress_plain(iterable, total, description, unit, capture)


def _progress_rich(iterable, total, description, unit, eta, capture):
    prog = _Progress(*_bar_columns(unit, eta=eta and total is not None),
                     console=_CAPS.console())
    with prog:
        # Escape: the description is data, rendered via a markup TextColumn
        # ("[bold]{task.description}"); an unbalanced tag (e.g. "x [/]") would
        # otherwise raise MarkupError, and "[red]x" would be silently swallowed.
        task_id = prog.add_task(_escape(description), total=total)
        for item in iterable:
            if capture:
                with captured():
                    yield item
            else:
                yield item
            prog.advance(task_id)


def _progress_plain(iterable, total, description, unit, capture):
    handle = TaskHandle(description=description, total=total, unit=unit)
    for item in iterable:
        if capture:
            with captured():
                yield item
        else:
            yield item
        handle.advance(1)
    tot = f"/{total}" if total else ""
    import builtins

    builtins.print(_plain(f"{description}: done ({handle._completed}{tot} {unit})"),
                   file=sys.stdout)


@contextlib.contextmanager
def spinner(description: str, *, success: Optional[str] = None) -> Iterator:
    """Indeterminate-duration step with elapsed time.

    On clean exit, prints ``success`` as an :func:`ok` line if given.
    Fallback: prints ``description`` at entry, ``success`` (or "done") at exit.
    """
    if _CAPS.interactive:
        prog = _Progress(
            _SpinnerColumn(style=_CAPS.palette.accent, spinner_name=_CAPS.palette.spinner),
            _TextColumn("[bold]{task.description}"),
            _TimeElapsedColumn(),
            console=_CAPS.console(),
            transient=True,
        )
        with prog:
            prog.add_task(_escape(description), total=None)  # data -> escape
            yield
        if success:
            ok(success)
    else:
        import builtins

        builtins.print(_plain(f"{description} ..."), file=sys.stdout)
        yield
        if success:
            ok(success)
        else:
            builtins.print(_plain(f"{description}: done"), file=sys.stdout)


# ===========================================================================
# Surface 3 — monitor / status dashboard (§5.4)
# ===========================================================================
class StatusRow(NamedTuple):
    subsystem: str
    state: str
    detail: str


_STATES = ("OK", "WARN", "ERROR", "IDLE")


def badge(state: str) -> str:
    """Rendered badge for one state.

    Rich/color: markup like ``[bold #34d399]● OK[/]``. Fallback: bare uppercase word.
    """
    st = str(state).upper()
    if _CAPS.color_ok:
        p = _CAPS.palette
        colors = {"OK": p.ok, "WARN": p.warn, "ERROR": p.err, "IDLE": "grey50"}
        color = colors.get(st, p.accent)
        return f"[bold {color}]● {_escape(st)}[/]"
    return st


def build_status_dashboard(
    title: str,
    rows: Sequence,
    *,
    overall: Optional[str] = None,
    subtitle: Optional[str] = None,
):
    """BUILD (do not print) the status dashboard renderable.

    Rich + color: return a rich ``Panel(Table)`` (the same look
    :func:`status_dashboard` prints today).
    Fallback (no rich / no color): return a plain multi-line ``str`` (no ANSI, no
    clear).

    Consumed by :func:`status_dashboard` (which prints it) and by
    :func:`live_dashboard` (which refreshes it in place for ``monitor --watch``).

    Each row is a ``StatusRow=(subsystem, state, detail)``; ``state`` ∈
    {OK,WARN,ERROR,IDLE} renders as a colored badge. ``overall`` is an optional
    summary line.
    """
    norm = [StatusRow(*r) if not isinstance(r, StatusRow) else r for r in rows]
    console = _CAPS.console()
    if console is not None and _CAPS.color_ok:
        p = _CAPS.palette
        t = _Table(expand=True, border_style="grey37",
                   header_style=f"bold {p.accent}", padding=(0, 1))
        t.add_column("Subsystem", style="bold", no_wrap=True)
        t.add_column("Status", no_wrap=True)
        t.add_column("Detail", style="dim")
        for r in norm:
            t.add_row(_escape(str(r.subsystem)), badge(r.state), _escape(str(r.detail)))
        sub_markup = f"[dim]{_escape(subtitle)}[/dim]" if subtitle else None
        if overall:
            extra = f"overall: {_escape(overall)}"
            sub_markup = f"{sub_markup} · {extra}" if sub_markup else f"[dim]{extra}[/dim]"
        return _Panel(t, title=f"[bold {p.accent}]{_escape(title)}[/]",
                      subtitle=sub_markup, border_style=p.accent)
    # Plain fixed-width table — assembled into a string (no ANSI, no clear).
    lines = [_plain(title)]
    if subtitle:
        lines.append(_plain(subtitle))
    sub_w = max([len("Subsystem")] + [len(r.subsystem) for r in norm], default=9)
    st_w = max([len("Status")] + [len(str(r.state).upper()) for r in norm], default=6)
    lines.append(f"  {'Subsystem'.ljust(sub_w)}  {'Status'.ljust(st_w)}  Detail")
    for r in norm:
        lines.append(
            f"  {r.subsystem.ljust(sub_w)}  {str(r.state).upper().ljust(st_w)}  "
            f"{_plain(r.detail)}"
        )
    if overall:
        lines.append(f"  overall: {_plain(overall)}")
    return "\n".join(lines)


def status_dashboard(
    title: str,
    rows: Sequence,
    *,
    overall: Optional[str] = None,
    subtitle: Optional[str] = None,
) -> None:
    """Render a status table. Never clears the screen.

    Thin wrapper over :func:`build_status_dashboard`: builds the renderable then
    prints it (rich: ``console.print``; fallback: ``builtins.print`` of the plain
    string). Each row is a ``StatusRow=(subsystem, state, detail)``; ``state`` ∈
    {OK,WARN,ERROR,IDLE} renders as a colored badge. ``overall`` is an optional
    summary line. Fallback: fixed-width plain table, uppercase states, no color.
    """
    built = build_status_dashboard(title, rows, overall=overall, subtitle=subtitle)
    console = _CAPS.console()
    if console is not None and _CAPS.color_ok:
        console.print(built)
        return
    import builtins

    builtins.print(built)


class LiveHandle:
    """Handle for a live dashboard. ``.refresh()`` re-renders ``render_fn``.

    Rich: drives a ``rich.Live`` updating in place. Fallback: each refresh appends
    a fresh plain block separated by a rule (scrollback-safe, never clears).
    """

    def __init__(self, render_fn: Callable[[], Any], *, live=None, plain=False):
        self._render_fn = render_fn
        self._live = live
        self._plain = plain
        self._first = True

    def refresh(self) -> None:
        if self._live is not None:
            self._live.update(self._render_fn())
        else:
            if not self._first:
                rule()
            self._first = False
            out = self._render_fn()
            import builtins

            builtins.print(_plain(str(out)) if isinstance(out, str) else out)


@contextlib.contextmanager
def live_dashboard(
    render_fn: Callable[[], Any],
    *,
    refresh_per_second: int = 4,
) -> Iterator[LiveHandle]:
    """Re-render ``render_fn()`` in place via ``rich.Live`` (for ``monitor --watch``).

    Yields a :class:`LiveHandle` with ``.refresh()``. Fallback: append-only plain
    blocks separated by a rule (no clear).
    """
    if _CAPS.interactive:
        # Pin the TTY decision for the Live's lifetime. rich.Live redirects
        # sys.stdout to a non-TTY proxy, and capability detection is live —
        # without the pin, the first refresh() re-ran build_status_dashboard
        # with color_ok=False and the boxed Panel "instantly went to text":
        # Live kept redrawing the plain-string fallback. (Same failure class
        # the banner already guards against by capturing its palette early.)
        prev_force_tty = _CAPS.force_tty
        _CAPS.force_tty = True
        try:
            with _Live(render_fn(), console=_CAPS.console(),
                       refresh_per_second=refresh_per_second, transient=False) as live:
                yield LiveHandle(render_fn, live=live)
        finally:
            _CAPS.force_tty = prev_force_tty
    else:
        # Plain: the caller's refresh() drives each appended block. No eager
        # pre-render here -- the monitor enters this context then refresh()es on the
        # first loop iteration, so an eager render would duplicate the first block.
        # (The rich branch above still renders eagerly on __enter__, which the
        # monitor's last-good seeding already accounts for.)
        yield LiveHandle(render_fn, plain=True)
