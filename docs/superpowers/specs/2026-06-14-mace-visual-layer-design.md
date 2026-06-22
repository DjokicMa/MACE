# MACE Visual Layer — Design Spec

- **Date:** 2026-06-14
- **Status:** Design (implementation-ready)
- **Scope:** Introduce a single UI facade `mace/utils/ui.py`, retire the fake startup animation, and give MACE four consistent presentation surfaces. Built on `rich` (13.9.4, installed) with a graceful no-rich / no-color / non-TTY fallback.
- **Reference (look & feel, not wired in):** `mace_visual_demo.py`, `mace_visual_demo_real.py` (symlinked in this worktree).
- **Stage boundary:** This spec covers the **Foundation phase** (build `ui.py` + tests only). Stage 2 integration call sites are *listed* here but **NOT implemented in this phase**.

---

## 1. Problem — current state

The current startup/UI code is the throwaway-animation era of MACE:

- **Fake loading bar.** `mace/utils/animation.py::loading_bar()` is a pure `time.sleep(duration/width)` loop. It reports nothing real — it just burns ~1–1.5 s on every banner show. `show_banner()` in `mace_cli` (lines 40–54) calls `animate_mace_assembly()` then `loading_bar(1.0, ...)`, so **every** non-suppressed invocation pays a forced delay that is decoupled from any actual work.
- **Full-screen clear / scrollback destruction.** `animate_mace_assembly()` prints `"\033[2J\033[H"` per frame (animation.py:62), and `sparkle_effect()` / `team_credits_animation()` do the same. This wipes the terminal and the user's scrollback. Banned going forward.
- **No real progress anywhere.** Long operations (parsing hundreds of `.out` files, populating `materials.db`, converting CIF→D12) emit a flood of per-item `print()` lines with no bar, no rate, no ETA. Example hot loops: `mace/database/populate_completed_jobs.py::populate_database()` (per-calc prints at lines 129/182/218) and `CrystalPropertyExtractor` in `mace/utils/property_extractor.py` (27 `print()` sites).
- **Scattered raw ANSI.** `mace/queue/monitor.py::print_status_dashboard()` (line 407) does `os.system('clear' if os.name == 'posix' else 'cls')` and hand-writes `\033[91m/\033[93m/\033[92m/\033[0m` color codes inline across ~10 `print()` statements. There is no shared message/status vocabulary; each command styles its own output ad hoc.
- **TTY-blind.** None of the above checks `isatty()` or `NO_COLOR`. Escape codes and the `\r`-driven loading bar leak verbatim into piped logs, redirected files, and SLURM `.out` capture.

The one good pattern already present is `BANNER_AVAILABLE` (mace_cli:32–38): a `try/except ImportError` that degrades to plain text. The new layer generalizes exactly that pattern.

---

## 2. Architecture

### 2.1 The facade

A single module **`mace/utils/ui.py`** is the only thing that ever imports `rich`. All call sites depend on `ui.*`, never on `rich` directly. This makes `rich` swappable or removable by editing one file.

```
call sites (mace_cli, queue/monitor.py, populate_*, extractors, …)
        │  ui.banner() / ui.progress() / ui.status_dashboard() / ui.ok()/warn()/err()/info() …
        ▼
   mace/utils/ui.py   ── single import boundary ──▶  rich (optional)
        │
        └─ when rich missing OR NO_COLOR OR not a TTY ─▶ stdlib plain-text / minimal-ANSI fallback
```

### 2.2 rich-optional + graceful fallback

At import time, `ui.py` attempts `import rich` inside `try/except ImportError`, mirroring `BANNER_AVAILABLE`:

```python
try:
    from rich.console import Console
    ... # other rich imports
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False
```

Every public function has two code paths: the rich path and a stdlib path. The stdlib path is not a stub — it prints the same *information* in plain text (see fallback gallery scene in `mace_visual_demo.py::scene_fallback`). No `ImportError` ever propagates to a call site.

### 2.3 Palettes

Two palettes ship (locked):

- **`crystal`** (primary, color): gradient `#2dd4bf → #22d3ee → #38bdf8 → #3b82f6 → #6366f1 → #818cf8`; `accent #38bdf8`, `bar #22d3ee`, `ok #34d399`, `warn #fbbf24`, `err #f87171`. Spinner glyph `dots`.
- **`mono`** (fallback, grayscale): `accent #d4d4d4` (+ grayscale gradient/bar/ok/warn/err). Spinner glyph `line` (safest for logs / colorblind).

A palette is a small frozen record `(name, gradient[6], accent, bar, ok, warn, err, spinner)`, modeled on `Palette` in `mace_visual_demo_real.py`. The active palette is module-level state, selected once at first use by the capability detection in §3. (The demo's `viridis/ember/ocean` are exploration-only and are **not** shipped.)

### 2.4 Capability detection (single source of truth)

`ui.py` computes three booleans once and caches them:

- `_rich_available` — `import rich` succeeded.
- `_is_tty` — `sys.stdout.isatty()`.
- `_color_ok` — `_is_tty and not os.environ.get("NO_COLOR")` and `os.environ.get("TERM") != "dumb"`.

Derived mode:

| condition | mode | palette | banner | progress |
|---|---|---|---|---|
| rich + TTY + color | **rich-interactive** | crystal | animated in-place | live bar + rate + ETA |
| rich + (no TTY **or** NO_COLOR) | **rich-quiet** | mono | one concise line | plain incremental counter (no Live redraws) |
| no rich | **plain** | n/a (no color) | one concise line | plain incremental counter |

The detection respects `MACE_NO_BANNER` / the existing `--no-banner` plumbing in `mace_cli` (env `MACE_NO_BANNER` already honored in `show_banner`). A `force_*` override hook exists for tests (see API `configure()`).

---

## 3. Adaptive TTY / NO_COLOR rules (exact)

1. **`sys.stdout.isatty()` is the master switch for cinematic vs concise.** Interactive → animated banner via `rich.Live` updating **in place** (never `\033[2J\033[H`, never a forced multi-second sleep). Non-interactive → a single line, e.g. `MACE v{ver} — Mendoza Automated CRYSTAL Engine`.
2. **`NO_COLOR` (any non-empty value) forces `mono` palette and disables all color/markup**, regardless of TTY. Honors the [no-color.org] convention.
3. **`TERM=dumb` → treated as non-color, plain mode.**
4. **No full-screen clear, ever.** The string `\033[2J\033[H` is banned in `ui.py` and in any new code. Banner growth uses `rich.Live(transient=False)` so scrollback survives.
5. **No fake delay.** The retired `loading_bar(duration=…)` time-sleep is gone; the only animation is the ~0.5 s line-by-line wordmark reveal, and it is skipped entirely in non-interactive mode.
6. **Progress redraw is throttled, not the work.** rich caps redraws at ~10–15/s; in quiet/plain mode progress collapses to occasional milestone lines (e.g. every N items or on completion), never `\r` spam into a log.
7. **Width / piping:** in non-TTY mode the rich `Console` is constructed with `force_terminal=False`; tables/panels degrade to plain text rather than emitting box-drawing escapes into a redirected file.

---

## 4. Capturing noisy per-file stdout (keeping bars clean)

The extractor and populate code paths `print()` heavily per item. If those prints interleave with a live progress bar, rich's redraw and the raw prints fight and the bar is corrupted.

**Rule:** while a `ui.progress()` / `ui.live_*` context is active, per-item worker output MUST be redirected away from the real stdout. The mechanism (proven in both demos via `contextlib.redirect_stdout(io.StringIO())`) is wrapped in a reusable helper so call sites do not each reinvent it:

- `ui.captured()` — a context manager that redirects `sys.stdout` (and optionally `sys.stderr`) into an in-memory buffer for the duration of one work unit. Returns the buffer so the caller can inspect/forward it (e.g. surface a captured error line through `ui.err()` after the bar advances).
- `ui.progress(...)` integrates this: the per-iteration callback runs inside `captured()` by default, so the bar stays pristine and any captured text can be attached to the final summary or re-emitted as structured `ui.err()/warn()` lines.
- In **plain/quiet mode**, capture is still applied (so logs are not double-noisy), but the chosen verbosity decides whether the captured buffer is flushed verbatim or summarized.

This is the single behavior that lets us add bars to `populate_database()` and `CrystalPropertyExtractor` loops without rewriting their internals (consistent with the "don't fix what works / layer on top" memory).

---

## 5. Public API of `mace/utils/ui.py` (exact contract)

> Foundation phase implements **exactly** these signatures and behaviors. All functions are import-safe (no side effects at import beyond capability detection). Every function degrades — none raises on missing rich.

### 5.1 Setup / capability

```python
def configure(*, palette: str | None = None, force_tty: bool | None = None,
              force_color: bool | None = None, no_banner: bool | None = None) -> None
```
Override auto-detected capabilities (primarily for tests / explicit CLI flags). `palette` ∈ {`"crystal"`, `"mono"`}. `None` args leave auto-detection intact.
**Fallback:** no-op-safe; if rich absent, color forces off regardless of args.

```python
def is_interactive() -> bool
```
True when mode is rich-interactive (rich + TTY + color). Call sites use this to decide whether an animated/Live presentation is worth it.
**Fallback:** returns `False` when rich absent or non-TTY.

```python
def active_palette() -> Palette
```
Return the current palette record (name, gradient, accent, bar, ok, warn, err, spinner).
**Fallback:** returns the `mono` palette when color is off.

### 5.2 Surface 1 — startup banner

```python
def banner(version: str, *, subtitle: str = "Mendoza Automated CRYSTAL Engine",
           meta: str | None = None) -> None
```
Render the MACE wordmark. Interactive: gradient wordmark revealed line-by-line via `rich.Live` **in place** (no screen clear, no forced delay beyond the ~0.5 s reveal), followed by subtitle + meta. Honors `MACE_NO_BANNER` / `--no-banner` (returns immediately if suppressed).
**Fallback:** single concise line `MACE v{version} — {subtitle}` (plain, no escapes); used in rich-quiet and plain modes, and whenever non-TTY.

```python
def credits() -> None
```
Print the static MACE credits block (reuses `mace/utils/banner.py::get_credits()` text).
**Fallback:** identical plain text in all modes (already plain today).

### 5.3 Surface 2 — real progress

```python
def progress(iterable, *, total: int | None = None, description: str = "",
             unit: str = "it", eta: bool = False, capture: bool = True) -> Iterator
```
Wrap an iterable and yield its items while driving a live bar (spinner + description + bar + percent + M/N + live rate + optional ETA), columns per `mace_visual_demo_real.py::_bar_columns`. When `capture=True`, each item is yielded inside `ui.captured()` so worker `print()`s do not corrupt the bar. `total=None` → indeterminate spinner.
**Fallback:** in quiet/plain mode, iterates transparently and emits milestone counter lines (e.g. start, every ~10%, done) instead of a live bar; no `\r`, no escapes. Never alters the items.

```python
@contextmanager
def progress_task(description: str, *, total: int | None = None, unit: str = "it",
                  eta: bool = False) -> Iterator[TaskHandle]
```
Manual-control form for loops that can't be a simple `for` (e.g. nested scan+populate). Yields a `TaskHandle` with `.advance(n=1)` and `.update(completed=…, description=…)`.
**Fallback:** `TaskHandle.advance/update` become cheap counters that print milestone lines; context still works.

```python
@contextmanager
def spinner(description: str, *, success: str | None = None) -> Iterator
```
Indeterminate-duration step (e.g. "Connecting to materials.db…", "Scanning test/…"). Shows a spinner + elapsed time; on clean exit prints `success` as an `ok()` line if given.
**Fallback:** prints `description` once at entry, `success` (or `done`) at exit. No spinner.

```python
@contextmanager
def captured(stderr: bool = False) -> Iterator[io.StringIO]
```
Redirect `sys.stdout` (and `sys.stderr` if `stderr=True`) into an in-memory buffer for the block; yields the buffer. Used internally by `progress` and directly by call sites that want to swallow/forward noisy worker output (§4).
**Fallback:** identical in all modes (pure stdlib; rich not involved).

### 5.4 Surface 3 — monitor / status dashboard

```python
def status_dashboard(title: str, rows: Sequence[StatusRow],
                     *, overall: str | None = None, subtitle: str | None = None) -> None
```
Render a status table where each `StatusRow = (subsystem: str, state: str, detail: str)` and `state ∈ {"OK","WARN","ERROR","IDLE"}` is rendered as a colored badge (`● OK` etc.). `overall` is an optional summary line/badge. Replaces the hand-rolled ANSI dashboard.
**Fallback:** fixed-width plain table (subsystem / STATE / detail), uppercase state words, no color, no clear — matches `scene_fallback` output.

```python
def badge(state: str) -> str
```
Return the rendered badge markup/text for a single state (used to compose ad-hoc lines).
**Fallback:** returns the bare uppercase word (`"WARN"`).

> Note: `status_dashboard` deliberately does **not** clear the screen. The current `os.system('clear')` + `\033[…m` in `queue/monitor.py::print_status_dashboard` is what this replaces; refresh-in-place for a watch loop is provided by `live_dashboard` below, using `rich.Live`, never a clear.

```python
@contextmanager
def live_dashboard(render_fn: Callable[[], Any], *, refresh_per_second: int = 4) -> Iterator[LiveHandle]
```
For `monitor --watch`-style loops: repeatedly re-render `render_fn()` in place via `rich.Live`. Yields a `LiveHandle` with `.refresh()`.
**Fallback:** each refresh prints a fresh plain dashboard block separated by a rule line (append-only, scrollback-safe); no clear.

### 5.5 Surface 4 — consistent CLI message styling

```python
def ok(text: str) -> None       # "✓" green  + text
def info(text: str) -> None     # "i" accent + text
def warn(text: str) -> None     # "!" yellow + text   (to stderr)
def err(text: str) -> None      # "✗" red    + text   (to stderr)
```
One-line status messages with a consistent icon + color vocabulary across every command. `warn`/`err` write to stderr so they survive stdout redirection.
**Fallback:** plain `"[OK] text"`, `"[i] text"`, `"[WARN] text"`, `"[ERROR] text"` (ASCII markers, no color); routing to stdout/stderr unchanged.

```python
def rule(title: str = "") -> None
```
Print a horizontal section divider (optionally titled). Used to separate phases of a long command.
**Fallback:** a line of `-` (with title inlined) at the console width or a fixed 60 cols.

```python
def table(columns: Sequence[str], rows: Iterable[Sequence[Any]],
          *, title: str | None = None) -> None
```
Render a result table (query results, produced-files lists, etc.).
**Fallback:** aligned plain-text columns (two-space gutter), title as a heading line; no box characters.

```python
def print(*objects, **kwargs) -> None
```
Thin pass-through to the shared console so call sites have **one** print entry point (enables global markup/highlight policy and the color/no-color decision in one place).
**Fallback:** routes to builtin `print` with any markup stripped.

### 5.6 Module-level constants / types (exported)

- `Palette` — the palette record type (§2.3).
- `CRYSTAL`, `MONO` — the two shipped palette instances.
- `WORDMARK` — the 6-line block-art list (reused from `banner.py` / demos).
- `StatusRow`, `TaskHandle`, `LiveHandle` — lightweight typed records/handles named above.

---

## 6. Integration plan — Stage 2 (DO NOT implement in Foundation)

Each surface maps to concrete existing call sites. Foundation builds `ui.py` + tests; Stage 2 rewires these.

### Surface 1 — startup banner
- `mace_cli::show_banner()` (lines 40–54): replace `animate_mace_assembly()` + `loading_bar()` + `get_credits()` with `ui.banner(__version__)` (+ `ui.credits()` when appropriate). Keep `MACE_NO_BANNER` / `--no-banner` checks.
- `mace_cli` banner sites at lines ~1124–1128, 1142–1150, 2944–2946 (`print_banner('banner')` for help/version): route through `ui.banner()` / `ui.credits()`.
- **Retire** `mace/utils/animation.py::loading_bar`, `animate_mace_assembly` (the `\033[2J\033[H` path), `sparkle_effect`, `team_credits_animation`, `mace_startup_animation`. Keep `banner.py::get_credits()` / `WORDMARK` art as data the facade reuses.

### Surface 2 — real progress
- `mace/database/populate_completed_jobs.py`: wrap `scan_for_completed_calculations()` (the `rglob("*.out")` loop) in `ui.spinner()` and the `populate_database()` per-calc loop (line 120) in `ui.progress(..., capture=True)`; the per-calc `print()`s (129/182/218) get captured.
- `mace/utils/property_extractor.py::CrystalPropertyExtractor` batch callers (the `mace analyze --extract-properties` path): wrap the over-files loop in `ui.progress(..., unit="out", eta=True)` with capture (27 internal prints).
- CIF→D12 / opt2cif batch entry points reached via `mace_cli` passthrough (lines 1041–1060 → `NewCifToD12.main`, `CrystalOutToCif.main`): add `ui.progress` at the batch boundary in those mains where a known file count exists.

### Surface 3 — monitor dashboard
- `mace/queue/monitor.py::print_status_dashboard()` (line 407): delete `os.system('clear')` and all inline `\033[…m`; build `StatusRow`s and call `ui.status_dashboard(...)`. Replace `_get_status_color()` (lines 507–515) usage with `ui.badge()`.
- `monitor --watch` loop in `mace/queue/monitor.py::main()` (entry near line 716, wrapper `mace/material_monitor.py`): drive via `ui.live_dashboard()` instead of clear-and-reprint.

### Surface 4 — CLI message styling
- Highest-traffic command handlers in `mace_cli`: `analyze` (≈1574), `convert`/`opt2cif`/`opt2d12`/`opt2d3` (≈1596), `database` (≈1735), `monitor` (≈1478), `credits` (≈1600), `status` (≈1612). Replace ad-hoc success/error `print()`s with `ui.ok/info/warn/err`, section headers with `ui.rule`, and query/result listings with `ui.table`.
- `mace/database/*` result printers used by `database --action query/stats`: route tabular output through `ui.table`.

*(Site list is the integration map, not a Foundation deliverable. Foundation = `ui.py` + unit tests that assert both rich and fallback paths via `configure(force_color=...)`/`force_tty=...` and a captured-stdout check; test the real invocation path per project memory.)*

---

## 7. Non-goals / out of scope (Foundation)

- No edits to `mace_cli`, `queue/monitor.py`, extractors, or populate code in this phase.
- No new palettes beyond `crystal` + `mono`.
- No changes to parsing/detection logic (preserve validated behavior; layer on top).
- No `git` state changes; orchestrator owns commits.
