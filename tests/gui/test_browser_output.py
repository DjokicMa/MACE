"""The browser MACE starts writes into a log file, not the terminal (1.1.6):
on a KDE desktop, xdg-open (kde-open) printed 'kf.iconthemes: Icon theme
"elementary" not found.' between MACE's own lines. `mace gui --open`, `mace
gui connect`'s open_url and `mace gui login-url --open` start the browser
through appwindow.spawn, whose stdout and stderr go to browser.log in MACE
GUI's cache folder. Stub browsers only; no real browser starts."""
import os
import re
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
MACE_CLI = REPO / "mace_cli"
NOISE = 'kf.iconthemes: Icon theme "elementary" not found.'
pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX stub browsers")


def _env(tmp_path, **extra):
    e = {k: v for k, v in os.environ.items()
         if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH", "MACE_GUI_TOKEN", "MACE_GUI_PORT",
                      "SSH_CONNECTION", "SLURM_JOB_ID", "TERM_PROGRAM", "VSCODE_IPC_HOOK_CLI", "BROWSER",
                      "MACE_GUI_STATE_DIR")}
    (tmp_path / "home").mkdir(exist_ok=True)
    e.update(HOME=str(tmp_path / "home"), XDG_CONFIG_HOME=str(tmp_path / "cfg"), XDG_CACHE_HOME=str(tmp_path / "cache"),
             PYTHONDONTWRITEBYTECODE="1", MACE_GUI_NO_WARMUP="1")
    e.update(extra)
    return e


def _noisy(path: Path, marker: Path):
    """A browser (or opener) that prints like KDE's, then records its arguments."""
    path.write_text(f"#!{sys.executable}\nimport sys\nprint('stdout chatter from the browser')\n"
                    f"print({NOISE!r}, file=sys.stderr)\n"
                    f"open({str(marker)!r}, 'a').write(' '.join(sys.argv[1:]) + '\\n')\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _wait_for(path: Path, s=10.0):
    end = time.monotonic() + s
    while time.monotonic() < end and not (path.exists() and path.read_text()):
        time.sleep(0.05)
    return path.read_text() if path.exists() else ""


def _log(tmp_path) -> str:
    return _wait_for(tmp_path / "cache" / "mace" / "gui" / "browser.log", 5)


def test_mace_gui_open_keeps_the_browsers_output_out_of_the_terminal(tmp_path):
    marker, stub = tmp_path / "opened.txt", tmp_path / "noisybrowser"
    _noisy(stub, marker)
    p = subprocess.Popen([sys.executable, str(MACE_CLI), "--no-banner", "gui", "--dir", str(tmp_path), "--port", "0",
                          "--open"], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, env=_env(tmp_path, BROWSER=str(stub), MACE_GUI_PRINT_TOKEN="1"))
    try:
        opened = _wait_for(marker, 60)
        time.sleep(0.5)
    finally:
        p.terminate()
        out = p.communicate(timeout=15)[0]
    assert re.search(r"http://127\.0\.0\.1:\d+/s/\S+", opened), opened
    assert "Press Ctrl+C to stop." in out                       # MACE's own lines stay
    assert NOISE not in out and "stdout chatter" not in out, out
    log = _log(tmp_path)
    assert NOISE in log and "stdout chatter from the browser" in log


def _open_in_child(tmp_path, env, code):
    return subprocess.run([sys.executable, "-c", code], cwd=str(REPO), env=env, capture_output=True, text=True,
                          timeout=60, stdin=subprocess.DEVNULL)


@pytest.mark.parametrize("call", ["from mace.gui import connect; print(connect.open_url(URL, 'tab'))",
                                  "from mace.gui import appwindow; print(appwindow.open_url(URL, 'tab'))"])
def test_xdg_open_output_goes_to_the_log(tmp_path, call):
    """A browser tab, no $BROWSER, on a desktop: webbrowser picks xdg-open, which MACE starts itself
    (quietly). PATH holds the stub only: no real browser can start."""
    if not sys.platform.startswith("linux"):
        pytest.skip("xdg-open is the Linux desktop opener")
    fakebin, marker = tmp_path / "bin", tmp_path / "opened.txt"
    fakebin.mkdir()
    _noisy(fakebin / "xdg-open", marker)
    env = _env(tmp_path, PATH=str(fakebin), DISPLAY=":0")
    env.pop("WAYLAND_DISPLAY", None)
    url = "http://127.0.0.1:8765/s/abc/login?code=x"
    r = _open_in_child(tmp_path, env, f"URL = {url!r}\n" + call)
    assert r.returncode == 0 and r.stdout.strip() == "True", r.stderr
    assert _wait_for(marker).strip() == url
    assert NOISE not in r.stderr + r.stdout
    assert NOISE in _log(tmp_path)


def test_login_url_open_is_quiet_too():
    src = (REPO / "mace" / "gui" / "loginurl.py").read_text()
    assert "webbrowser" not in src and "from .appwindow import open_url" in src
    for name in ("cli.py", "connect.py"):
        assert "webbrowser" not in (REPO / "mace" / "gui" / name).read_text(), name


def test_the_log_starts_over_when_it_grows(tmp_path):
    from mace.gui import appwindow
    env = _env(tmp_path)
    log = appwindow.log_path(env)
    log.parent.mkdir(parents=True)
    log.write_bytes(b"x" * (appwindow.LOG_MAX_BYTES + 10))
    assert appwindow.spawn([sys.executable, "-c", "print('fresh')"], env)
    assert _wait_for(log).strip() == "fresh"


def test_cache_dir_follows_the_state_dir_rules(tmp_path):
    from mace.gui import appwindow
    assert appwindow.cache_dir({"MACE_GUI_STATE_DIR": "/x/y"}) == Path("/x/y")
    assert appwindow.cache_dir({"XDG_CACHE_HOME": "/c", "HOME": "/h"}) == Path("/c/mace/gui")
    assert appwindow.cache_dir({"HOME": "/h"}) == Path("/h/.cache/mace/gui")


def test_spawn_reports_a_missing_program(tmp_path):
    from mace.gui import appwindow
    assert appwindow.spawn([str(tmp_path / "no-such-browser"), "http://x"], _env(tmp_path)) is False
