"""The workflow planner still imports when its optional helpers are missing.

planner.py falls back to built-in tables when ``d12_constants`` cannot be
imported, and to no ``DummyFileCreator`` when neither of its imports works.
Both fallback branches report through ``ui.warn``; ``ui`` must therefore be
defined before them, or the fallback itself stops the import with NameError.
"""
import subprocess
import sys

from conftest import REPO_ROOT

PROBE = """
import sys

BLOCKED = {"d12_constants", "mace.workflow.dummy_file_creator"}

class _Block:
    def find_spec(self, name, path=None, target=None):
        if name in BLOCKED:
            raise ImportError(f"blocked for test: {name}")
        return None

sys.meta_path.insert(0, _Block())
import mace.workflow.planner as planner
print(planner.D12_CONSTANTS_AVAILABLE, planner.DummyFileCreator)
"""


def test_planner_imports_without_d12_constants_or_dummy_file_creator():
    proc = subprocess.run(
        [sys.executable, "-c", PROBE],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip().splitlines()[-1] == "False None"
    assert "Could not import d12_constants" in proc.stderr
    assert "DummyFileCreator not available" in proc.stderr
