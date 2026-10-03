"""The MACE install that is running, exported to what it starts.

Job scripts pin their completion callback to the install that generated
them. A workflow generates its jobs from copies of the generators
(workflow_scripts/, the step directories), which cannot tell from their own
location which install they came from and pin $MACE_HOME instead. A shell
whose ~/.bashrc names another install (a side-by-side release test, a shared
group copy) would then pin that one, so the running MACE sets MACE_HOME, and
PYTHONPATH for the `mace` package, to itself first.
"""
import os
from pathlib import Path
from typing import MutableMapping, Optional

INSTALL_ROOT = Path(__file__).resolve().parent.parent.parent


def pin_install_home(environ: Optional[MutableMapping[str, str]] = None) -> Optional[Path]:
    """Point MACE_HOME and the front of PYTHONPATH at this install.

    Returns the install root, or None (environment untouched) when this is
    not a full install with the queue manager.
    """
    env = os.environ if environ is None else environ
    root = INSTALL_ROOT
    if not (root / "mace" / "queue" / "manager.py").is_file():
        return None
    env["MACE_HOME"] = str(root)
    parts = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    if not parts or parts[0] != str(root):
        env["PYTHONPATH"] = os.pathsep.join([str(root)] + parts)
    return root
