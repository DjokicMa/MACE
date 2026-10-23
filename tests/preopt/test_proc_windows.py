"""mace.preopt.proc where os.getuid, os.killpg and SIGKILL do not exist (Windows).

The GUI calls proc.sweep() before it detects the Clean methods; with no
os.getuid that raised AttributeError, so every method showed as failed. The
temp root is then per user by name, the sweeper removes nothing (os.kill(pid,
0) is no liveness check on Windows), and stop() falls back to the child itself.
"""
import os
import signal
import subprocess
import sys
import tempfile

from mace.preopt import proc


def _no_posix(monkeypatch, tmp_path):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    for name in ("getuid", "geteuid", "killpg"):
        monkeypatch.delattr(os, name, raising=False)
    monkeypatch.delattr(signal, "SIGKILL", raising=False)
    for k in ("LOGNAME", "USER", "LNAME", "USERNAME"):          # getpass.getuser() reads these in order
        monkeypatch.setenv(k, "Some User")


def test_temp_dirs_without_getuid(monkeypatch, tmp_path):
    _no_posix(monkeypatch, tmp_path)
    root = proc.temp_root()
    assert root == tmp_path / "mace-preopt-Some_User" and root.is_dir()
    d = proc.private_dir("xtb")
    assert d.parent == root and d.is_dir()
    assert proc._live_cwds() == []
    proc.remove(d)


def test_sweep_and_stop_without_posix_calls(monkeypatch, tmp_path):
    _no_posix(monkeypatch, tmp_path)
    left = proc.temp_root() / "999999999-xtb-abc"          # a dead pid's folder: kept, never guessed at
    left.mkdir()
    with monkeypatch.context() as m:
        m.setattr(os, "name", "nt")
        assert proc.sweep() == 0
    assert left.is_dir()
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    proc.stop(child, grace=5.0)
    assert child.poll() is not None
