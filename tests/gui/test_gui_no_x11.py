"""mace gui in an ssh session with X11 forwarding (MobaXterm turns it on by
default, so DISPLAY is set on the node): the server never draws through X.
matplotlib gets the Agg backend in the server and its workers, and --open
does not start a browser on the node through the forwarded display."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mace.gui import cli

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("env,forwarded", [
    ({"SSH_CONNECTION": "1 2 3 22", "DISPLAY": "localhost:10.0"}, True),
    ({"SSH_CONNECTION": "1 2 3 22", "DISPLAY": "node-a:12"}, True),
    ({"SSH_CONNECTION": "1 2 3 22", "DISPLAY": "localhost:10.0", "BROWSER": "/vscode/browser.sh"}, False),
    ({"DISPLAY": ":0"}, False),
    ({"SSH_CONNECTION": "1 2 3 22", "DISPLAY": ":1"}, False),
])
def test_x11_forwarded_display_is_recognized(env, forwarded):
    assert cli.x11_forwarded(env) is forwarded


def test_server_forces_the_agg_backend(tmp_path):
    """Even with MPLBACKEND=TkAgg and a forwarded DISPLAY, the server process
    (and the workers it starts) use Agg; the jobs it submits keep the login
    environment captured before."""
    code = ("import os; from mace.gui import cli, childenv; childenv.capture_login_env();"
            "cli.server_environment(); print(os.environ['MPLBACKEND']); print(childenv.login_env().get('MPLBACKEND'))")
    env = {**os.environ, "MPLBACKEND": "TkAgg", "DISPLAY": "localhost:10.0", "SSH_CONNECTION": "1 2 3 22"}
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == ["Agg", "TkAgg"]


def test_results_plotters_force_agg(monkeypatch):
    src = (REPO / "mace" / "gui" / "results.py").read_text()
    assert 'os.environ["MPLBACKEND"] = "Agg"' in src and 'setdefault("MPLBACKEND"' not in src


def test_open_with_a_forwarded_display_does_not_start_a_browser_on_the_node(tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr(cli, "open_browser", lambda *a, **k: opened.append(a) or True)
    monkeypatch.setenv("SSH_CONNECTION", "1 2 3 22")
    monkeypatch.setenv("DISPLAY", "localhost:10.0")
    monkeypatch.delenv("BROWSER", raising=False)
    assert cli.browser_blocked_reason() and "X11" in cli.browser_blocked_reason()
    monkeypatch.setenv("BROWSER", "/vscode/browser.sh")
    assert cli.browser_blocked_reason() is None


def test_startup_imports_no_gui_toolkit(tmp_path):
    """Starting the server imports no matplotlib, tkinter or Qt."""
    code = ("import sys; from mace.gui import cli, server, structures, results;"
            "bad = [m for m in ('matplotlib.pyplot', 'tkinter', 'PyQt5', 'PyQt6', 'PySide6') if m in sys.modules];"
            "print(bad)")
    env = {**os.environ, "DISPLAY": "localhost:10.0"}
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "[]"
