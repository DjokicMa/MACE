"""`mace gui connect` (and the other standard-library tools under `mace gui`)
run on a computer with only Python and ssh: numpy, ase, plotly, spglib,
pandas, scipy, yaml, rich, matplotlib and the POSIX-only modules (fcntl,
termios, pwd, grp, resource) are blocked by a sitecustomize import blocker,
through the real mace_cli subprocess (AC-B15)."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
BLOCKED = ["numpy", "ase", "plotly", "spglib", "pandas", "scipy", "yaml", "rich", "matplotlib",
           "fcntl", "termios", "pwd", "grp", "resource"]
BLOCKER = '''
import sys
BLOCKED = %r
class _Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError("blocked " + name)
        return None
sys.meta_path.insert(0, _Blocker())
''' % (BLOCKED,)


@pytest.fixture
def blocked_env(tmp_path):
    d = tmp_path / "blocker"
    d.mkdir()
    (d / "sitecustomize.py").write_text(BLOCKER)
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON") and k not in ("MACE_HOME",)}
    env.update(PYTHONPATH=str(d), XDG_CONFIG_HOME=str(tmp_path / "cfg"), XDG_CACHE_HOME=str(tmp_path / "cache"),
               HOME=str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    return env


def _run(env, *argv):
    return subprocess.run([sys.executable, str(REPO / "mace_cli"), "--no-banner", *argv], env=env,
                          stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120)


def test_the_blocker_blocks(blocked_env):
    r = subprocess.run([sys.executable, "-c", "import numpy"], env=blocked_env, capture_output=True, text=True)
    assert r.returncode != 0 and "blocked numpy" in r.stderr


def test_connect_dry_run_and_help_with_the_standard_library_only(blocked_env):
    r = _run(blocked_env, "gui", "connect", "--dry-run", "--gateway", "g", "--node", "n")
    assert r.returncode == 0, r.stderr
    assert "ssh -J g -o ExitOnForwardFailure=yes" in r.stdout and "--no-banner gui --managed --port 40000" in r.stdout
    r = _run(blocked_env, "gui", "connect", "--help")
    assert r.returncode == 0 and "usage: mace gui connect" in r.stdout and "--hop" in r.stdout
    # --check (connect_check.py) too: a FAIL line depends on this computer, a traceback would not
    r = _run(blocked_env, "gui", "connect", "--check", "--no-net", "--fix-config", "--gateway", "g", "--node", "n")
    assert r.returncode in (0, 1) and "Verdict" in r.stdout and "Host n" in r.stdout, r.stderr
    assert "Traceback" not in r.stderr


def test_importing_the_gui_cli_module_stays_light(blocked_env):
    code = "import mace.gui.cli as c; import sys; print(c.dispatch_tool.__name__, 'numpy' in sys.modules)"
    r = subprocess.run([sys.executable, "-c", code], env={**blocked_env, "PYTHONPATH": blocked_env["PYTHONPATH"]
                                                          + os.pathsep + str(REPO)},
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["dispatch_tool", "False"]


def test_export_script_with_the_standard_library_only(blocked_env):
    r = _run(blocked_env, "gui", "connect", "--export-script", "bash", "--gateway", "g")
    assert r.returncode == 0, r.stderr
    assert 'D_GATEWAY=g' in r.stdout and r.stdout.startswith("#!/usr/bin/env bash")
