"""Opening MACE GUI in this computer's browser: ``mace gui --open``, ``mace
gui connect`` (and its Enter to open again), ``mace gui login-url --open``.

``app`` (the default): MACE GUI gets its own window from a Chromium-family
browser (Brave, Chrome, Chromium, Edge or Vivaldi, the first one found) in
app mode, ``--app=<url>``: no tabs, no address bar, the page's title (MACE)
and icon. It runs in a profile of its own under MACE GUI's cache folder
(app-profile), apart from your browsing, so it gets its own taskbar entry,
and on Linux the window class MACE. Without such a browser (Firefox has no
app mode), or with ``tab``, the default browser opens a tab. ``--browser
tab|app`` picks one; else a ``$BROWSER`` of yours means tab (VS Code's
terminal sets one that opens the page on your own computer); else ``[gui]
open`` in site.toml; else app.

The browser is started with its own output kept out of the terminal: what a
browser or the desktop's opener prints (KDE's "kf.iconthemes: Icon theme
"elementary" not found.", GPU and sandbox notes) goes to ``browser.log`` in
MACE GUI's cache folder (cache_dir), never between MACE's own lines.
Standard library only: ``mace gui connect`` runs on computers that have
Python and ssh but not MACE's scientific packages.
"""
from __future__ import annotations

import ntpath
import os
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

LOG_NAME = "browser.log"
LOG_MAX_BYTES = 1 << 20          # browser.log starts over past this size
MODES = ("app", "tab")
APP_CLASS = "MACE"               # the app window's class on Linux (WM_CLASS, Wayland app id)
# The Chromium-family browsers, in the order MACE looks for them: (name, commands on PATH,
# macOS app bundle, Windows install path under Program Files or %LOCALAPPDATA%)
APP_BROWSERS: Tuple[Tuple[str, Tuple[str, ...], str, str], ...] = (
    ("Brave", ("brave-browser", "brave-browser-stable", "brave"), "Brave Browser",
     r"BraveSoftware\Brave-Browser\Application\brave.exe"),
    ("Chrome", ("google-chrome", "google-chrome-stable", "chrome"), "Google Chrome",
     r"Google\Chrome\Application\chrome.exe"),
    ("Chromium", ("chromium", "chromium-browser"), "Chromium", r"Chromium\Application\chrome.exe"),
    ("Edge", ("microsoft-edge", "microsoft-edge-stable", "msedge"), "Microsoft Edge",
     r"Microsoft\Edge\Application\msedge.exe"),
    ("Vivaldi", ("vivaldi", "vivaldi-stable"), "Vivaldi", r"Vivaldi\Application\vivaldi.exe"),
)
WINDOWS_ROOTS = ("ProgramFiles", "ProgramFiles(x86)", "ProgramW6432", "LOCALAPPDATA")
# a snap browser (/snap/bin/chromium) may not write a hidden folder of the home folder (~/.cache): its
# app profile goes to the snap's own ~/snap/<name>/common, and a start that fails within this many
# seconds (exits non-zero) opens a tab instead
SNAP_CHECK_S = 2.0


def system_name() -> str:
    return "windows" if os.name == "nt" else "darwin" if sys.platform == "darwin" else "linux"


def cache_dir(env: Optional[Mapping[str, str]] = None) -> Path:
    """MACE GUI's folder in this computer's cache: MACE_GUI_STATE_DIR, else
    %LOCALAPPDATA%\\mace\\gui on Windows, else $XDG_CACHE_HOME/mace/gui
    (~/.cache/mace/gui), the folder cache.state_dir uses."""
    e = os.environ if env is None else env
    if e.get("MACE_GUI_STATE_DIR"):
        return Path(e["MACE_GUI_STATE_DIR"])
    if os.name == "nt" and e.get("LOCALAPPDATA"):
        return Path(e["LOCALAPPDATA"]) / "mace" / "gui"
    base = e.get("XDG_CACHE_HOME") or os.path.join(e.get("HOME") or os.path.expanduser("~"), ".cache")
    return Path(base) / "mace" / "gui"


def log_path(env: Optional[Mapping[str, str]] = None) -> Path:
    return cache_dir(env) / LOG_NAME


def _log_file(env: Optional[Mapping[str, str]] = None) -> Any:
    """browser.log opened for appending (started over past LOG_MAX_BYTES), else DEVNULL."""
    path = log_path(env)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        big = path.is_file() and path.stat().st_size > LOG_MAX_BYTES
        return open(path, "wb" if big else "ab")
    except OSError:
        return subprocess.DEVNULL


def spawn(argv: List[str], env: Optional[Dict[str, str]] = None, check: float = 0.0) -> bool:
    """Start argv on its own (stdin closed, stdout and stderr to browser.log),
    without waiting for it; False when it could not start. ``check``: seconds
    to watch it, False when it exits with an error within them (a browser
    handing the page to one already running exits with 0)."""
    out = _log_file(env)
    kw: Dict[str, Any] = {"stdin": subprocess.DEVNULL, "stdout": out, "stderr": out, "close_fds": True}
    if os.name == "nt":
        # its own process group, no console: Ctrl+C in this window does not reach the browser
        kw["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200) | 0x00000008
    else:
        kw["start_new_session"] = True
    try:
        child = subprocess.Popen(argv, env=env, **kw)
    except (OSError, ValueError):
        return False
    finally:
        if out is not subprocess.DEVNULL:
            out.close()
    if check > 0:
        try:
            return child.wait(timeout=check) == 0
        except subprocess.TimeoutExpired:
            pass
    return True


def open_tab(url: str, env: Optional[Dict[str, str]] = None) -> bool:
    """The URL in the default browser (or $BROWSER), as webbrowser picks it.
    A command webbrowser would start itself (xdg-open, gio, $BROWSER's) is
    started by spawn instead, so its output goes to browser.log; the others
    (Windows' and macOS's own openers, which print nothing) open it as is."""
    try:
        b = webbrowser.get()
    except webbrowser.Error:
        return False
    if isinstance(b, webbrowser.GenericBrowser) and getattr(b, "name", None):
        # the command line GenericBrowser.open (and BackgroundBrowser's) would run
        return spawn([b.name] + [a.replace("%s", url) for a in b.args], env)
    try:
        return bool(b.open(url))
    except Exception:
        return False


def find_app_browser(system: Optional[str] = None, env: Optional[Mapping[str, str]] = None,
                     which: Callable = shutil.which, isfile: Callable = os.path.isfile) -> Optional[Tuple[str, str]]:
    """(name, program) of the first Chromium-family browser here (APP_BROWSERS'
    order): its install folders on Windows (Program Files, %LOCALAPPDATA%) and
    macOS (/Applications, ~/Applications), then its commands on PATH."""
    sysname = system or system_name()
    e = os.environ if env is None else env
    home = e.get("HOME") or e.get("USERPROFILE") or os.path.expanduser("~")
    for name, commands, bundle, winpath in APP_BROWSERS:
        places: List[str] = []
        if sysname == "windows":
            places = [ntpath.join(e[v], winpath) for v in WINDOWS_ROOTS if e.get(v)]
        elif sysname == "darwin":
            places = [f"{root}/{bundle}.app/Contents/MacOS/{bundle}" for root in ("/Applications",
                                                                                  f"{home}/Applications")]
        for p in places:
            if isfile(p):
                return name, p
        for c in commands:
            hit = which(c, path=e.get("PATH"))
            if hit:
                return name, hit
    return None


def profile_dir(env: Optional[Mapping[str, str]] = None) -> Path:
    """The app window's own browser profile (cookies, window size), apart from your browsing."""
    return cache_dir(env) / "app-profile"


def snap_name(program: str, realpath: Callable[[str], str] = os.path.realpath) -> Optional[str]:
    """The snap a browser command belongs to (/snap/bin/chromium: "chromium"), else None."""
    p = str(program)
    if not p.startswith("/snap/"):
        try:
            p = realpath(p)
        except OSError:
            return None
    if p.startswith("/snap/bin/"):
        return Path(p).name.split(".")[0] or None
    parts = Path(p).parts
    return parts[2] if len(parts) > 2 and parts[1] == "snap" and parts[2] != "bin" else None


def snap_profile_dir(name: str, env: Optional[Mapping[str, str]] = None) -> Path:
    """A snap browser's app profile: in the snap's own folder (~/snap/<name>/common), not a hidden one."""
    e = os.environ if env is None else env
    home = e.get("HOME") or os.path.expanduser("~")
    return Path(home) / "snap" / name / "common" / "mace-app-profile"


def app_command(program: str, url: str, profile: str, system: Optional[str] = None) -> List[str]:
    """The command line that opens url in an app window of a Chromium-family browser."""
    argv = [program, f"--app={url}", f"--user-data-dir={profile}", "--no-first-run", "--no-default-browser-check"]
    if (system or system_name()) not in ("windows", "darwin"):
        argv.append(f"--class={APP_CLASS}")
    return argv


def setting_mode() -> Optional[str]:
    """``[gui] open`` in site.toml ("app" or "tab"), else None."""
    try:
        from mace.submission import site
        tables, _, _ = site.load_tables()
        val = (tables.get("gui") or {}).get("open")
    except Exception:
        return None
    val = val.strip().lower() if isinstance(val, str) else None
    return val if val in MODES else None


def resolve_mode(flag: Optional[str] = None, env: Optional[Mapping[str, str]] = None,
                 setting: Callable[[], Optional[str]] = setting_mode) -> str:
    """--browser, else tab when $BROWSER is set, else [gui] open, else app."""
    if flag in MODES:
        return str(flag)
    e = os.environ if env is None else env
    if e.get("BROWSER"):
        return "tab"
    return setting() or "app"


def open_url(url: str, mode: Optional[str] = None, env: Optional[Dict[str, str]] = None,
             say: Optional[Callable[[str], None]] = None) -> bool:
    """MACE GUI's URL in an app window (mode app, when a Chromium-family
    browser is here), else a tab; False when nothing opened it. say: told
    once which window it is."""
    if resolve_mode(mode, env) == "app":
        found = find_app_browser(env=env)
        if found:
            snap = snap_name(found[1]) if system_name() == "linux" else None
            profile = snap_profile_dir(snap, env) if snap else profile_dir(env)
            try:
                profile.mkdir(parents=True, exist_ok=True)
            except OSError:
                pass
            if spawn(app_command(found[1], url, str(profile)), env, check=SNAP_CHECK_S if snap else 0.0):
                if say:
                    say(f"Opened MACE GUI in its own {found[0]} window (--browser tab opens a browser tab "
                        "instead).")
                return True
    return open_tab(url, env)
