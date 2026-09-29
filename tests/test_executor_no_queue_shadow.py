"""The workflow executor does not import mace/queue as the stdlib ``queue``.

mace_cli does sys.path.insert(0, "<repo>/mace"), so a bare ``import queue``
there finds the mace/queue package, not the standard library. executor.py
had such an import, unused, which registered mace/queue as ``queue`` in
sys.modules for everything imported after it. Removing it was not enough:
pyarrow.lib (imported eagerly by mace.database) does its own ``import queue``.
mace/__init__.py now pins the stdlib ``queue`` in sys.modules before any of
that runs, and mace_cli names the package ``mace.queue``.
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


BARE_PROBE = """
import sys
sys.path.insert(0, {mace_dir!r})
import mace
import queue
print(queue.__file__)
print(hasattr(queue, "Queue"))
"""


def test_importing_mace_pins_the_stdlib_queue_even_with_mace_dir_first():
    """The cause, not just the pyarrow symptom: with mace/ ahead of the stdlib
    on sys.path, any bare ``import queue`` after ``import mace`` must still get
    the standard library (this does not need pyarrow, so it runs in CI too)."""
    mace_dir = str(REPO_ROOT / "mace")
    proc = subprocess.run(
        [sys.executable, "-c", BARE_PROBE.format(mace_dir=mace_dir)],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]
    queue_file, has_queue_class = proc.stdout.strip().splitlines()[-2:]
    assert not queue_file.startswith(mace_dir), queue_file
    assert has_queue_class == "True"


def test_mace_cli_names_the_queue_package_by_its_full_name():
    """With the stdlib pinned as ``queue``, a bare ``from queue.manager``
    in mace_cli can no longer resolve; it must say ``mace.queue``."""
    import re
    text = (REPO_ROOT / "mace_cli").read_text()
    bare = re.findall(r"^\s*(?:from|import)\s+queue\b.*$", text, re.M)
    assert bare == [], bare
