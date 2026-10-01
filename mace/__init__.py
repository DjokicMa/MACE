"""
Job_Scripts package initialization.

This file maintains backwards compatibility during refactoring.
All modules will be re-exported here to ensure existing imports continue to work.
"""

# During refactoring, imports will be added here to maintain compatibility
# For example:
# from .workflow_core.engine import *
# from .workflow_core.planner import *
# etc.

# Canonical MACE version -- the ONLY place a version literal is written.
# Every display/export site (mace_cli, utils/animation.py, utils/banner.py,
# database/export/formats.py) imports it from here, so a bump needs no other
# edit. test_version_single_source.py enforces that; do not reintroduce copies.
__version__ = "1.1.3"


def _bind_stdlib_queue():
    """Make sure ``sys.modules['queue']`` is the standard library module.

    mace_cli (and a few modules) put this mace/ directory itself on sys.path,
    where the mace/queue package shadows the stdlib ``queue``. The first bare
    ``import queue`` after that -- e.g. inside pyarrow.lib, which
    mace.database imports eagerly -- would register mace/queue as ``queue``
    for the rest of the process. Importing the real module once, with this
    directory hidden from sys.path, pins the stdlib one in sys.modules first.
    Inside MACE the package is always ``mace.queue``.
    """
    import os
    import sys

    if "queue" in sys.modules:
        return
    here = os.path.dirname(os.path.abspath(__file__))
    saved = list(sys.path)
    sys.path[:] = [p for p in saved
                   if os.path.abspath(p or os.curdir) != here]
    try:
        import queue  # noqa: F401  (stdlib, now registered in sys.modules)
    finally:
        sys.path[:] = saved


_bind_stdlib_queue()
del _bind_stdlib_queue
