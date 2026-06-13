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

# Canonical MACE version. Other display sites (mace_cli, utils/animation.py,
# database/export/formats.py) mirror this literal -- keep them in sync on bump.
__version__ = "1.0.5"
