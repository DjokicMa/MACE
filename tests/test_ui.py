"""Tests for the MACE visual-layer facade ``mace/utils/ui.py``.

Coverage map (per the Foundation spec / API contract):
  (a) rich path works — smoke each public function in forced rich-interactive mode.
  (b) FALLBACK correctness — when output is piped/non-TTY/NO_COLOR, the produced
      text contains NO ANSI escape sequences ('\\x1b[' absent) and no crash.
  (c) rich-absent path — simulate rich missing (block the import + reimport ui)
      and exercise every function with no crash and clean plain text.
  (d) banner is adaptive — a single concise line when not a TTY, the animated
      reveal only when interactive; '\\033[2J\\033[H' never appears.
  (e) REAL-INVOCATION-PATH — a tiny script is run via subprocess with stdout
      PIPED (so not a TTY) and the raw bytes are asserted to contain no escape
      codes, per the project testing principle ("test the real invocation path").

No database is touched (enable_tracking is irrelevant here; ui.py never opens a DB).
"""

import io
import os
import sys
import subprocess
import textwrap
import contextlib
import importlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable  # the anaconda python the suite is run under

ESC = "\x1b["  # CSI prefix — what we assert is absent from plain output
CLEAR = "\033[2J\033[H"  # the banned full-screen clear


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def ui():
    """Import the facade fresh and restore its global caps after the test.

    ``configure()`` mutates module-level state, so we snapshot and restore the
    ``_CAPS`` overrides to keep tests independent.
    """
    import mace.utils.ui as ui_mod

    caps = ui_mod._CAPS
    saved = (caps.force_tty, caps.force_color, caps.palette_name,
             caps.no_banner, caps._console)
    try:
        yield ui_mod
    finally:
        (caps.force_tty, caps.force_color, caps.palette_name,
         caps.no_banner, caps._console) = saved


@pytest.fixture(autouse=True)
def _hermetic_theme_config(tmp_path_factory, monkeypatch):
    """Keep theme persistence out of the user's real ~/.config/mace during tests:
    point XDG_CONFIG_HOME at an empty temp dir and reset the per-process cache, so
    every test sees 'no saved theme' unless it sets one itself."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path_factory.mktemp("xdg")))
    import mace.utils.ui as ui_mod
    ui_mod._saved_theme = ui_mod._UNSET


@pytest.fixture
def force_color_ui(ui, monkeypatch):
    """A ui module GUARANTEED to be on the real rich/color path.

    Anti-masking. The CI/ctx environment can set ``NO_COLOR=1`` (or ``TERM=dumb``),
    which per no-color.org vetoes color *before* ``force_color`` is even consulted
    (see ``_Caps.color_ok`` rule 1). That silently downgrades a "rich-path" test to
    the PLAIN path — exercising none of the rich-only code and giving false
    confidence. That is exactly how the markup-injection crash slipped through the
    whole sub-tool sweep. This fixture strips the env vetoes, forces color, and then
    ASSERTS ``color_ok`` so a still-plain environment fails LOUDLY right here instead
    of masking a rich-path bug downstream.
    """
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")  # also neutralizes TERM=dumb
    ui2 = _rich_ui(ui)
    ui2.configure(force_color=True, force_tty=True, palette="crystal")
    assert ui2._CAPS.color_ok is True, (
        "force_color_ui could not enable the rich path — the environment still "
        "vetoes color (NO_COLOR / TERM=dumb / rich missing). Rich-path coverage "
        "would otherwise run silently on the plain path."
    )
    assert ui2.is_interactive() is True
    return ui2


def _capture_stdout_stderr(fn):
    """Run ``fn`` capturing both stdout and stderr into one string."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        fn()
    return out.getvalue() + err.getvalue()


def _exercise_all(ui):
    """Smoke every public surface once. Returns combined stdout+stderr text."""
    def body():
        ui.banner("9.9.9")
        ui.banner("9.9.9", subtitle="Sub", meta="meta line")
        ui.credits()
        ui.ok("converged")
        ui.info("informational")
        ui.warn("a warning")
        ui.err("an error")
        ui.rule("section")
        ui.rule()
        ui.table(["Material", "E (Ha)"],
                 [["mat_001", "-1234.5"], ["mat_017", "-1890.1"]], title="Query")
        ui.status_dashboard(
            "monitor",
            [ui.StatusRow("DATABASE", "OK", "312 materials"),
             ui.StatusRow("ERRORS", "ERROR", "1 SCF failure"),
             ("FILES", "WARN", "3 missing .out"),
             ui.StatusRow("QUEUE", "IDLE", "nothing pending")],
            overall="degraded", subtitle="real data")
        ui.badge("OK"); ui.badge("WARN"); ui.badge("ERROR"); ui.badge("IDLE")
        for _ in ui.progress([1, 2, 3, 4], description="batch", unit="it"):
            print("worker noise that must be captured")
        for _ in ui.progress(iter([1, 2, 3]), description="nototal", unit="it"):
            pass
        with ui.progress_task("manual", total=4, unit="job") as h:
            for i in range(4):
                h.advance()
            h.update(completed=4, description="manual-done")
        with ui.spinner("connecting", success="connected"):
            print("connect noise")
        with ui.spinner("plain-spin"):
            pass
        with ui.captured() as buf:
            print("swallowed")
        assert "swallowed" in buf.getvalue()
        with ui.live_dashboard(lambda: "DASH") as lh:
            lh.refresh()
            lh.refresh()
        ui.print("[bold]markup[/bold] passthrough")
    return _capture_stdout_stderr(body)


# ===========================================================================
# (a) rich path works — smoke in forced interactive mode
# ===========================================================================
def test_rich_interactive_smoke(ui):
    ui.configure(force_color=True, force_tty=True, palette="crystal")
    assert ui.is_interactive() is True
    assert ui.active_palette().name == "crystal"
    text = _exercise_all(ui)
    # In interactive mode rich *does* emit escapes — that's the whole point.
    assert ESC in text
    # But it must never use the banned full-screen clear.
    assert CLEAR not in text
    assert "2J" not in text


def test_active_palette_and_badge_rich(ui):
    ui.configure(force_color=True, force_tty=True, palette="crystal")
    assert ui.active_palette() is ui.CRYSTAL
    b = ui.badge("WARN")
    assert "WARN" in b and ui.CRYSTAL.warn in b  # color markup present


def test_configure_rejects_unknown_palette(ui):
    with pytest.raises(ValueError):
        ui.configure(palette="rainbow")


# ===========================================================================
# (b) FALLBACK correctness — piped / non-TTY / NO_COLOR => NO escape codes
# ===========================================================================
def test_plain_mode_has_no_ansi(ui):
    ui.configure(force_color=False, force_tty=False, palette="mono")
    assert ui.is_interactive() is False
    text = _exercise_all(ui)
    assert ESC not in text, f"plain output leaked an escape code: {text!r}"
    assert CLEAR not in text
    # Plain markers are present, markup is stripped.
    assert "[OK] converged" in text
    assert "[WARN] a warning" in text
    assert "[ERROR] an error" in text
    assert "[i] informational" in text
    # rich markup must not leak literally into the message text path.
    assert "[bold]" not in text


def test_no_color_env_forces_plain(ui, monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    # Even with force_tty True, NO_COLOR must disable color.
    ui.configure(force_tty=True)
    assert ui._CAPS.color_ok is False
    assert ui.active_palette().name == "mono"
    text = _exercise_all(ui)
    assert ESC not in text


def test_term_dumb_forces_plain(ui, monkeypatch):
    monkeypatch.setenv("TERM", "dumb")
    ui.configure(force_tty=True)
    assert ui._CAPS.color_ok is False
    text = _exercise_all(ui)
    assert ESC not in text


def test_no_color_beats_force_color_true(ui, monkeypatch):
    """NO_COLOR is authoritative: it must win even over ``force_color=True``.

    This is the exact precedence branch that was failing — ``force_color=True``
    previously short-circuited and ignored ``NO_COLOR``, leaking ANSI escapes.
    """
    monkeypatch.setenv("NO_COLOR", "1")
    # Force BOTH color and tty on; NO_COLOR must still disable color.
    ui.configure(force_color=True, force_tty=True)
    assert ui._CAPS.color_ok is False
    assert ui.is_interactive() is False
    assert ui.active_palette().name == "mono"
    text = _exercise_all(ui)
    assert ESC not in text, f"NO_COLOR + force_color leaked escapes: {text!r}"
    assert CLEAR not in text
    # Plain markers present, markup stripped — proves the plain path was taken.
    assert "[OK] converged" in text
    assert "[bold]" not in text


def test_term_dumb_beats_force_color_true(ui, monkeypatch):
    """TERM=dumb is authoritative too: it disables color over ``force_color=True``."""
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "dumb")
    ui.configure(force_color=True, force_tty=True)
    assert ui._CAPS.color_ok is False
    assert ui.is_interactive() is False
    text = _exercise_all(ui)
    assert ESC not in text, f"TERM=dumb + force_color leaked escapes: {text!r}"
    assert CLEAR not in text


def test_no_color_any_value_beats_force_color(ui, monkeypatch):
    """no-color.org: ANY value of NO_COLOR (even '0') disables color, over force."""
    monkeypatch.setenv("NO_COLOR", "0")  # any non-empty value counts
    ui.configure(force_color=True, force_tty=True)
    assert ui._CAPS.color_ok is False
    text = _exercise_all(ui)
    assert ESC not in text


def test_force_color_true_enables_color_when_env_allows(ui, monkeypatch):
    """force_color=True still enables color when the environment does not forbid it."""
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    ui.configure(force_color=True, force_tty=True)
    assert ui._CAPS.color_ok is True
    assert ui.is_interactive() is True


def test_force_color_false_forces_plain(ui, monkeypatch):
    """force_color=False forces plain even on a clean, color-capable environment."""
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    ui.configure(force_color=False, force_tty=True)
    assert ui._CAPS.color_ok is False
    text = _exercise_all(ui)
    assert ESC not in text


def test_warn_and_err_route_to_stderr(ui):
    ui.configure(force_color=False, force_tty=False)
    out, errbuf = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(errbuf):
        ui.ok("ok-line")
        ui.info("info-line")
        ui.warn("warn-line")
        ui.err("err-line")
    o, e = out.getvalue(), errbuf.getvalue()
    assert "ok-line" in o and "info-line" in o
    assert "ok-line" not in e and "info-line" not in e
    assert "warn-line" in e and "err-line" in e
    assert "warn-line" not in o and "err-line" not in o


def test_status_dashboard_plain_table(ui):
    ui.configure(force_color=False, force_tty=False)
    text = _capture_stdout_stderr(lambda: ui.status_dashboard(
        "monitor",
        [ui.StatusRow("DATABASE", "OK", "ok-detail"),
         ui.StatusRow("ERRORS", "ERROR", "err-detail")],
        overall="DEGRADED"))
    assert ESC not in text
    assert "DATABASE" in text and "OK" in text and "ok-detail" in text
    assert "overall: DEGRADED" in text
    # No box-drawing characters in the plain table.
    for boxch in "─│╭╮╰╯┃━":
        assert boxch not in text


def test_badge_plain_is_bare_word(ui):
    ui.configure(force_color=False, force_tty=False)
    assert ui.badge("WARN") == "WARN"
    assert ui.badge("ok") == "OK"  # uppercased
    assert ui.badge("ERROR") == "ERROR"


def test_progress_yields_all_items_unchanged(ui):
    ui.configure(force_color=False, force_tty=False)
    src = [{"a": 1}, "two", 3, None]
    collected = []
    with contextlib.redirect_stdout(io.StringIO()):
        for item in ui.progress(src, description="d"):
            collected.append(item)
    assert collected == src  # identity/content preserved, nothing mutated


def test_progress_captures_worker_prints(ui):
    """capture=True must keep per-item prints out of the visible stream."""
    ui.configure(force_color=False, force_tty=False)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        for _ in ui.progress([1, 2], description="d", capture=True):
            print("INNER_WORKER_NOISE")
    text = out.getvalue()
    assert "INNER_WORKER_NOISE" not in text  # swallowed by captured()


def test_progress_no_capture_lets_prints_through(ui):
    ui.configure(force_color=False, force_tty=False)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        for _ in ui.progress([1], description="d", capture=False):
            print("VISIBLE_NOISE")
    assert "VISIBLE_NOISE" in out.getvalue()


def test_captured_helper_swallows_and_returns_buffer(ui):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        with ui.captured() as buf:
            print("hidden-stdout")
        with ui.captured(stderr=True) as buf2:
            print("hidden-out", file=sys.stdout)
            print("hidden-err", file=sys.stderr)
    assert "hidden-stdout" not in out.getvalue()
    assert "hidden-stdout" in buf.getvalue()
    assert "hidden-out" in buf2.getvalue() and "hidden-err" in buf2.getvalue()


def test_table_plain_alignment(ui):
    ui.configure(force_color=False, force_tty=False)
    text = _capture_stdout_stderr(lambda: ui.table(
        ["Material", "Energy"],
        [["mat_001", "-1.0"], ["a-much-longer-material", "-2.0"]],
        title="Results"))
    assert ESC not in text
    assert "Results" in text
    assert "Material" in text and "Energy" in text
    assert "a-much-longer-material" in text


def test_live_dashboard_plain_append_only(ui):
    ui.configure(force_color=False, force_tty=False)
    calls = {"n": 0}

    def render():
        calls["n"] += 1
        return f"BLOCK-{calls['n']}"

    text = _capture_stdout_stderr(lambda: _drive_live(ui, render))
    assert ESC not in text
    assert "BLOCK-1" in text
    # A rule separates appended blocks (scrollback-safe, never clears).
    assert "-" in text


def _drive_live(ui, render):
    with ui.live_dashboard(render) as lh:
        lh.refresh()
        lh.refresh()


# ===========================================================================
# (c) rich-absent path — simulate rich missing, reimport ui, exercise all
# ===========================================================================
def test_rich_absent_path(monkeypatch):
    """Block ``import rich`` so ui falls back; every function must still work."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "rich" or name.startswith("rich."):
            raise ImportError("rich blocked for test")
        return real_import(name, *args, **kwargs)

    # Drop cached rich + ui so the fresh import takes the ImportError branch.
    for mod in [m for m in list(sys.modules)
                if m == "rich" or m.startswith("rich.") or m == "mace.utils.ui"]:
        monkeypatch.delitem(sys.modules, mod, raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    ui_mod = importlib.import_module("mace.utils.ui")
    try:
        assert ui_mod._RICH_AVAILABLE is False
        # Even when color is force-requested, no rich => color stays off.
        ui_mod.configure(force_color=True, force_tty=True)
        assert ui_mod.is_interactive() is False
        assert ui_mod.active_palette().name == "mono"
        text = _exercise_all(ui_mod)
        assert ESC not in text, f"rich-absent path leaked escapes: {text!r}"
        assert CLEAR not in text
        assert "MACE v9.9.9" in text
    finally:
        # Remove the rich-less ui so the rest of the suite reimports the real one.
        sys.modules.pop("mace.utils.ui", None)
        # monkeypatch.delitem restores the sys.modules entry on teardown but NOT
        # the stale rich-less attribute it leaves on the parent package; drop it
        # so the next `import mace.utils.ui` rebinds to the real (rich-ful) module.
        import mace.utils as _mace_utils
        if getattr(_mace_utils, "ui", None) is ui_mod:
            del _mace_utils.ui


def test_rich_absent_print_strips_markup(monkeypatch):
    """With rich BLOCKED, ui.print() must strip rich markup, not leak the tags.

    Reproduces the markup-leak finding: ``ui.print('[bold]hi[/] there')`` should
    emit ``hi there`` (no ``[bold]`` / ``[/]`` brackets) on the stdlib path.
    """
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "rich" or name.startswith("rich."):
            raise ImportError("rich blocked for test")
        return real_import(name, *args, **kwargs)

    for mod in [m for m in list(sys.modules)
                if m == "rich" or m.startswith("rich.") or m == "mace.utils.ui"]:
        monkeypatch.delitem(sys.modules, mod, raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    ui_mod = importlib.import_module("mace.utils.ui")
    try:
        assert ui_mod._RICH_AVAILABLE is False
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            ui_mod.print("[bold]hi[/] there")
        text = out.getvalue()
        assert text == "hi there\n", f"markup leaked on rich-absent path: {text!r}"
        assert "[bold]" not in text
        assert "[/]" not in text
        assert "[" not in text and "]" not in text
        assert ESC not in text

        # A few more shapes: closing tags and hex-colored markers are stripped,
        # while non-tag bracketed prose (our own [OK] markers, lists) survives.
        out2 = io.StringIO()
        with contextlib.redirect_stdout(out2):
            ui_mod.print("[bold #34d399]done[/bold #34d399]")
            ui_mod.print("values [1, 2, 3] and [OK] marker")
        t2 = out2.getvalue()
        assert "done" in t2 and "[bold" not in t2 and "#34d399" not in t2
        assert "[1, 2, 3]" in t2 and "[OK]" in t2  # prose preserved
    finally:
        sys.modules.pop("mace.utils.ui", None)
        # Also drop the stale rich-less attribute monkeypatch won't restore on the
        # parent package (see test_rich_absent_path), so later tests reimport real.
        import mace.utils as _mace_utils
        if getattr(_mace_utils, "ui", None) is ui_mod:
            del _mace_utils.ui


# ===========================================================================
# (d) banner is adaptive — concise single line when not a TTY
# ===========================================================================
def test_banner_concise_when_not_tty(ui):
    ui.configure(force_color=False, force_tty=False)
    text = _capture_stdout_stderr(lambda: ui.banner("1.2.3"))
    assert ESC not in text
    assert CLEAR not in text
    # exactly the concise single line
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert lines == ["MACE v1.2.3 — Mendoza Automated CRYSTAL Engine"]


def test_banner_suppressed_by_no_banner_config(ui):
    ui.configure(no_banner=True, force_tty=False)
    text = _capture_stdout_stderr(lambda: ui.banner("1.2.3"))
    assert text == ""  # nothing printed when suppressed


def test_banner_suppressed_by_env(ui, monkeypatch):
    monkeypatch.setenv("MACE_NO_BANNER", "1")
    ui.configure(force_tty=False)
    text = _capture_stdout_stderr(lambda: ui.banner("9.9.9"))
    assert text == ""


def test_banner_animated_when_interactive_no_clear(ui):
    ui.configure(force_color=True, force_tty=True)
    text = _capture_stdout_stderr(lambda: ui.banner("3.0.0"))
    # Interactive banner draws via rich.Live (escapes present) but never clears.
    assert CLEAR not in text and "\033[2J" not in text


# ===========================================================================
# (e) REAL-INVOCATION-PATH — subprocess with stdout piped (not a TTY)
# ===========================================================================
REAL_INVOCATION_SCRIPT = textwrap.dedent(
    """
    import sys
    sys.path.insert(0, {repo!r})
    from mace.utils import ui
    # Do NOT force anything: rely on real detection. stdout is a pipe here,
    # so isatty() is False and the facade must choose the plain path.
    assert ui.is_interactive() is False
    ui.banner("7.7.7")
    ui.ok("optimization converged")
    ui.info("using SLURM partition general")
    ui.warn("seekpath not installed")
    ui.err("mat_207 SCF did not converge")
    ui.rule("results")
    ui.table(["Material", "E (Ha)"],
             [["mat_001", "-1234.5"], ["mat_017", "-1890.1"]], title="Query")
    ui.status_dashboard(
        "MACE system monitor",
        [("DATABASE", "OK", "312 materials"),
         ("ERRORS", "ERROR", "1 SCF failure")],
        overall="DEGRADED")
    for _ in ui.progress([1, 2, 3], description="converting", unit="cif"):
        print("per-item worker noise")
    with ui.spinner("connecting to materials.db", success="connected"):
        pass
    ui.credits()
    """
)


def test_real_invocation_no_escapes_when_piped():
    """Run a tiny script in a child process with stdout PIPED (not a TTY) and
    assert the raw bytes contain no ANSI escape sequences. This is the
    real-invocation-path test mandated by project policy: isolated unit calls can
    miss import-order/integration bugs that only surface in a true subprocess.
    """
    script = REAL_INVOCATION_SCRIPT.format(repo=str(REPO_ROOT))
    env = dict(os.environ)
    env.pop("NO_COLOR", None)  # prove the no-TTY path alone is enough
    env.pop("MACE_NO_BANNER", None)
    proc = subprocess.run(
        [PYTHON, "-c", script],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
    )
    assert proc.returncode == 0, (
        f"child failed:\nSTDOUT:\n{proc.stdout.decode(errors='replace')}\n"
        f"STDERR:\n{proc.stderr.decode(errors='replace')}")
    # Raw BYTES on both streams: no CSI escape, no full-screen clear.
    assert b"\x1b[" not in proc.stdout, proc.stdout
    assert b"\x1b[" not in proc.stderr, proc.stderr
    assert b"\x1b[2J" not in proc.stdout
    out = proc.stdout.decode()
    # Adaptive: the banner collapsed to one concise line.
    assert "MACE v7.7.7 — Mendoza Automated CRYSTAL Engine" in out
    assert "[OK] optimization converged" in out
    # warn/err went to stderr, not stdout.
    err = proc.stderr.decode()
    assert "[WARN] seekpath not installed" in err
    assert "[ERROR] mat_207 SCF did not converge" in err
    assert "[WARN]" not in out and "[ERROR]" not in out


def test_banner_plain_no_ansi_and_no_clear(ui):
    """banner() under force_color=False (non-interactive) prints the concise line
    with no '\\x1b' and no '\\033[2J' full-screen clear."""
    ui.configure(force_color=False, force_tty=False)
    text = _capture_stdout_stderr(lambda: ui.banner("5.6.7", subtitle="Concise Sub"))
    assert ESC not in text
    assert CLEAR not in text and "\033[2J" not in text
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert lines == ["MACE v5.6.7 — Concise Sub"]


def test_banner_static_concise_when_piped(ui):
    """animate=False still collapses to the concise line when not a TTY (piped),
    never dumping ANSI art into a redirect."""
    ui2 = _rich_ui(ui)
    ui2.configure(force_color=False, force_tty=False)
    text = _capture_stdout_stderr(lambda: ui2.banner("1.1.0", subtitle="Sub", animate=False))
    assert ESC not in text and "2J" not in text
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert lines == ["MACE v1.1.0 — Sub"]


def test_banner_static_themed_wordmark_in_tty(ui):
    """animate=False in a color TTY renders the themed gradient wordmark STATICALLY:
    the block-art glyph + color escapes are present, with no full-screen clear and
    no fake animation delay. (Regression: the static/help/version banner used to
    bypass the ui theme via banner.py's plain ASCII art.)"""
    ui2 = _rich_ui(ui)
    ui2.configure(force_color=True, force_tty=True, palette="crystal")
    text = _capture_stdout_stderr(lambda: ui2.banner("1.1.0", animate=False))
    assert "█" in text          # the wordmark block art is present
    assert ESC in text          # themed (gradient) — not plain ASCII
    assert CLEAR not in text and "2J" not in text


# ===========================================================================
# Stage 2 — color theme selection (ember palette, MACE_THEME, consistency)
# ===========================================================================
def test_ember_palette_selectable(ui):
    ui2 = _rich_ui(ui)
    ui2.configure(force_color=True, force_tty=True, palette="ember")
    assert ui2.active_palette() is ui2.EMBER
    assert ui2.active_palette().name == "ember"
    assert ui2.EMBER.gradient[0] == "#7f1d1d"


def test_mace_theme_env_selects_palette(ui, monkeypatch):
    """MACE_THEME picks the palette with no explicit configure(palette=...); an
    invalid value falls back to the auto default (crystal when color is available)."""
    ui2 = _rich_ui(ui)
    monkeypatch.delenv("NO_COLOR", raising=False)
    caps = ui2._Caps()            # fresh caps -> no palette_name override
    caps.force_color = True
    caps.force_tty = True
    monkeypatch.setenv("MACE_THEME", "ember")
    assert caps.palette is ui2.EMBER
    monkeypatch.setenv("MACE_THEME", "bogus")
    assert caps.palette is ui2.CRYSTAL


def test_banner_theme_consistency_static_uses_selected_palette(ui, monkeypatch):
    """A selected theme drives the static banner (and, by construction, the same
    _CAPS.palette drives the animation) — so the two never mix crystal/ember."""
    monkeypatch.delenv("NO_COLOR", raising=False)
    ui2 = _rich_ui(ui)
    ui2.configure(force_color=True, force_tty=True, palette="ember")
    text = _capture_stdout_stderr(lambda: ui2.banner("1.1.0", animate=False))
    assert "38;2;127;29;29" in text       # ember gradient[0] = #7f1d1d
    assert "38;2;45;212;191" not in text   # crystal gradient[0] must be absent


def test_wm_settled_uses_passed_palette_not_caps(ui):
    """Regression: rich.Live redirects stdout, so a settle that re-read _CAPS.palette
    saw isatty()=False -> mono mid-animation. The settle MUST use the palette passed
    in. With _CAPS pinned to crystal, rendering the settle with MONO yields mono."""
    from rich.console import Console
    ui2 = _rich_ui(ui)
    ui2.configure(force_color=True, force_tty=True, palette="crystal")

    def render(pal):
        c = Console(force_terminal=True, width=100, record=True, color_system="truecolor")
        c.print(ui2._wm_settled("1.1.0", "Sub", None, pal))
        return c.export_text(styles=True)

    crystal_out, mono_out = render(ui2.CRYSTAL), render(ui2.MONO)
    assert "45;212;191" in crystal_out         # crystal gradient[0]
    assert "45;212;191" not in mono_out         # mono settle has NO crystal, despite _CAPS=crystal
    assert "82;82;82" in mono_out               # mono gradient[0] = #525252


def test_save_and_load_theme_roundtrip(ui, monkeypatch, tmp_path):
    """Persisted theme survives across processes (saved to ~/.config/mace)."""
    import os
    ui2 = _rich_ui(ui)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    ui2._saved_theme = ui2._UNSET
    assert ui2.load_saved_theme() is None
    path = ui2.save_theme("ember")
    assert os.path.exists(path)
    ui2._saved_theme = ui2._UNSET               # force a fresh read from disk
    assert ui2.load_saved_theme() == "ember"
    with pytest.raises(ValueError):
        ui2.save_theme("rainbow")


def test_data_with_markup_does_not_crash_rich_path(force_color_ui):
    """Regression: the sub-tool sweep routes DATA (filenames, formulas, error text)
    through ui.*; on the rich path that data must not be parsed as markup and crash
    with MarkupError. Escape it; literal brackets survive.

    Uses ``force_color_ui`` so NO_COLOR in the CI env can't silently downgrade this
    to the plain path (which would mask the very crash it guards against)."""
    import re
    ui2 = force_color_ui
    danger = ["[/]", "[/red]", "[bad", "bad: [Errno 2]", "Fe2O3_[mp-19770][/].cif", "[red]x[/red]"]
    for d in danger:
        for fn in ("print", "info", "ok", "warn", "err", "rule"):
            _capture_stdout_stderr(lambda fn=fn, d=d: getattr(ui2, fn)(d))   # must not raise
        _capture_stdout_stderr(lambda d=d: ui2.table(["C"], [[d]], title=d))
        _capture_stdout_stderr(
            lambda d=d: ui2.status_dashboard(d, [ui2.StatusRow("db", "ERROR", d)],
                                             overall=d, subtitle=d))
    clean = re.sub(r"\x1b\[[0-9;]*m", "", _capture_stdout_stderr(lambda: ui2.err("x [Errno 2] y")))
    assert "[Errno 2]" in clean   # data preserved, not swallowed


def test_banner_width_guard_narrow_falls_back_to_concise(ui, monkeypatch):
    """Narrow terminal: the banner shows the concise line, not the wrapping art."""
    from rich.console import Console
    import io
    ui2 = _rich_ui(ui)
    ui2.configure(force_color=True, force_tty=True, palette="crystal")

    def render(width):
        buf = io.StringIO()
        c = Console(force_terminal=True, width=width, color_system="truecolor",
                    file=buf, highlight=False)
        monkeypatch.setattr(ui2._CAPS, "console", lambda c=c: c)
        ui2.banner("1.1.0", animate=False)
        return buf.getvalue()

    narrow, wide = render(30), render(80)
    assert "█" not in narrow and "MACE v1.1.0" in narrow   # concise, no art
    assert "█" in wide                                      # full art when wide


# ===========================================================================
# Stage 2 — startup animation color-leak guard (deterministic, no PTY)
# ===========================================================================
def _rich_ui(ui):
    """Return a rich-ENABLED ui module.

    The rich-absent tests reimport ``mace.utils.ui`` with rich blocked and the
    monkeypatch ``delitem`` restore can leave a rich-less (and possibly
    sys.modules-orphaned) module for a later test to inherit. These Stage-2 tests
    need the real rich path, so force a clean fresh import when the inherited
    module came up rich-less.
    """
    if not getattr(ui, "_RICH_AVAILABLE", False):
        sys.modules.pop("mace.utils.ui", None)
        ui = importlib.import_module("mace.utils.ui")
    return ui


def _frame_styles(frame):
    """Collect every style string referenced by a rich Text frame.

    Covers the base ``frame.style`` and each span's style (spans carry the
    per-cell styles produced by the generators)."""
    styles = []
    base = getattr(frame, "style", None)
    if base is not None:
        styles.append(str(base))
    for span in getattr(frame, "spans", []) or []:
        st = getattr(span, "style", None)
        if st is not None:
            styles.append(str(st))
    return styles


def test_banner_anim_generators_no_crystal_leak_in_mono(ui):
    """For each generator, ALL frames built with MONO.gradient must NEVER contain
    a crystal hex anywhere in their span styles (mono must not flash crystal)."""
    ui = _rich_ui(ui)
    ui.configure(force_color=True, force_tty=True, palette="crystal")  # rich available
    crystal_hexes = set(ui.CRYSTAL.gradient)
    mono_grad = ui.MONO.gradient
    for gen in (ui._gen_phonon, ui._gen_decode, ui._gen_shimmer):
        for frame in gen(mono_grad):
            for style in _frame_styles(frame):
                for chex in crystal_hexes:
                    assert chex not in style, (
                        f"{gen.__name__} leaked crystal color {chex!r} "
                        f"in mono frame style {style!r}")


def test_banner_anim_generators_render_with_crystal(ui):
    """All generators must produce frames with CRYSTAL.gradient without raising,
    and render to a string via the rich console."""
    ui = _rich_ui(ui)
    ui.configure(force_color=True, force_tty=True, palette="crystal")
    console = ui._CAPS.console()
    grad = ui.CRYSTAL.gradient
    for gen in (ui._gen_phonon, ui._gen_decode, ui._gen_shimmer):
        frames = list(gen(grad))
        assert frames, f"{gen.__name__} yielded no frames"
        for frame in frames:
            # rendering must not raise (exercises the Text + styles end-to-end)
            with console.capture() as cap:
                console.print(frame)
            assert cap.get() is not None


# ===========================================================================
# Stage 2 — build_status_dashboard() entry point
# ===========================================================================
def test_build_status_dashboard_plain_returns_str(ui):
    """In plain mode build_status_dashboard returns a str with title + each row's
    subsystem and zero ANSI escapes; status_dashboard prints the same content."""
    ui.configure(force_color=False, force_tty=False)
    rows = [ui.StatusRow("DATABASE", "OK", "ok-detail"),
            ("ERRORS", "ERROR", "err-detail"),
            ui.StatusRow("QUEUE", "IDLE", "nothing")]
    built = ui.build_status_dashboard("monitor", rows, overall="DEGRADED",
                                      subtitle="real data")
    assert isinstance(built, str)
    assert ESC not in built
    assert "monitor" in built and "real data" in built
    for sub in ("DATABASE", "ERRORS", "QUEUE"):
        assert sub in built
    assert "overall: DEGRADED" in built
    # status_dashboard prints exactly the built string (+ trailing newline).
    printed = _capture_stdout_stderr(lambda: ui.status_dashboard(
        "monitor", rows, overall="DEGRADED", subtitle="real data"))
    assert printed == built + "\n"
    assert ESC not in printed


def test_build_status_dashboard_rich_returns_renderable(ui):
    """In forced-color mode build_status_dashboard returns a non-str rich
    renderable (a Panel), not a plain string."""
    ui = _rich_ui(ui)
    ui.configure(force_color=True, force_tty=True, palette="crystal")
    built = ui.build_status_dashboard(
        "monitor", [ui.StatusRow("DATABASE", "OK", "312 materials")],
        overall="ok")
    assert not isinstance(built, str)
    # It renders through the rich console without raising.
    console = ui._CAPS.console()
    with console.capture() as cap:
        console.print(built)
    assert "DATABASE" in cap.get()


def test_real_invocation_no_banner_env_piped():
    """With MACE_NO_BANNER set, the subprocess prints no wordmark/banner line."""
    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(REPO_ROOT)!r})
        from mace.utils import ui
        ui.banner("4.4.4")
        ui.ok("after-banner")
        """
    )
    env = dict(os.environ)
    env["MACE_NO_BANNER"] = "1"
    proc = subprocess.run(
        [PYTHON, "-c", script],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
    )
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    out = proc.stdout.decode()
    assert "MACE v4.4.4" not in out  # banner suppressed
    assert "[OK] after-banner" in out
    assert b"\x1b[" not in proc.stdout


# ===========================================================================
# Test hardening — force the REAL rich path so NO_COLOR can't MASK rich-path
# bugs (the env veto in _Caps.color_ok silently downgrades to plain; that hid
# the markup-injection crash through the whole sub-tool sweep).
# ===========================================================================
# Bracketed data that rich's markup parser would choke on or silently swallow
# unless escaped — mirrors what the sub-tools route through ui.* (paths,
# formulas, errno text). Includes a pathological *over-closer* that raises
# MarkupError if it ever reaches rich unescaped.
_MARKUP_DANGER = [
    "[/]", "[/red]", "[bad", "bad: [Errno 2]", "Fe2O3_[mp-19770][/].cif",
    "[red]x[/red]", "[1, 2, 3]", "[OK] done", "done [/] then [/]",
]


def _strip_ansi(s):
    import re
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def test_force_color_fixture_is_genuinely_on_rich_path(force_color_ui):
    """Canary for the anti-masking fixture: it MUST emit real ANSI color.

    If this fails, every ``force_color_ui`` test is silently running on the plain
    path and rich-only regressions (markup crashes, palette flips) go undetected."""
    out = _capture_stdout_stderr(lambda: force_color_ui.ok("rich-path canary"))
    assert ESC in out, "force_color_ui produced no ANSI — the rich path is NOT active"


@pytest.mark.parametrize("data", _MARKUP_DANGER)
def test_rich_path_survives_markup_in_data(force_color_ui, data):
    """On the REAL rich path, every data-bearing surface must (1) not raise on
    bracketed data and (2) genuinely be on the rich path (some surface emits ANSI),
    so this coverage can never be silently downgraded to the plain path."""
    ui2 = force_color_ui
    saw_ansi = []

    def cap(fn):
        out = _capture_stdout_stderr(fn)   # must not raise
        if ESC in out:
            saw_ansi.append(True)

    for name in ("print", "info", "ok", "warn", "err", "rule"):
        cap(lambda name=name: getattr(ui2, name)(data))
    cap(lambda: ui2.table(["C", "X"], [[data, data]], title=data))
    cap(lambda: ui2.status_dashboard(
        data, [ui2.StatusRow("db", "ERROR", data)], overall=data, subtitle=data))
    ui2.badge(data)  # returns a markup string; must not raise building it
    assert saw_ansi, "no surface emitted ANSI — the rich path was not exercised"


def test_rich_path_preserves_literal_bracketed_data(force_color_ui):
    """Escaped, not parsed: literal bracket data survives verbatim on the rich path
    (would be silently swallowed by the markup parser if not escaped)."""
    ui2 = force_color_ui

    def render(fn):
        # The stdout rich console is cached on first build (bound to sys.stdout at
        # that moment), so redirect_stdout must be active *when it is built*.
        # Invalidate the lazy cache before each capture so it rebuilds inside the
        # redirect and the output is captured.
        ui2._CAPS._console = None
        return _strip_ansi(_capture_stdout_stderr(fn))

    for data, needle in [("err [Errno 2] missing", "[Errno 2]"),
                         ("Fe2O3_[mp-19770].cif", "[mp-19770]"),
                         ("values [1, 2, 3] ok", "[1, 2, 3]")]:
        clean = render(lambda d=data: ui2.info(d))
        assert needle in clean, (needle, clean)
