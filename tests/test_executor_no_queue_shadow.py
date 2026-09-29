"""The workflow executor does not import mace/queue as the stdlib ``queue``.

mace_cli does sys.path.insert(0, "<repo>/mace"), so a bare ``import queue``
there finds the mace/queue package, not the standard library. executor.py
had such an import, unused, which registered mace/queue as ``queue`` in
sys.modules for everything imported after it.
"""
import subprocess
import sys

from conftest import REPO_ROOT

PROBE = """
import sys
sys.path.insert(0, {mace_dir!r})
import mace.workflow.executor
q = sys.modules.get("queue")
print(getattr(q, "__file__", None))
"""


def test_importing_the_executor_under_the_cli_path_leaves_queue_alone():
    mace_dir = str(REPO_ROOT / "mace")
    proc = subprocess.run(
        [sys.executable, "-c", PROBE.format(mace_dir=mace_dir)],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]
    queue_file = proc.stdout.strip().splitlines()[-1]
    assert not queue_file.startswith(mace_dir), queue_file
