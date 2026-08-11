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
__version__ = "1.1.1"
