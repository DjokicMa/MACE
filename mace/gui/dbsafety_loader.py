"""mace.database.dbsafety for the GUI server, without importing the
mace.database package (its __init__ imports MaterialDatabase, ASE and
pyarrow; the GUI never opens a database read-write, PLAN_analysis C17.5).
The module is registered under its own name, so a later
``import mace.database`` reuses it."""
import importlib.util
import sys
from pathlib import Path

_NAME = "mace.database.dbsafety"


def dbsafety():
    mod = sys.modules.get(_NAME)
    if mod is not None:
        return mod
    path = Path(__file__).resolve().parents[1] / "database" / "dbsafety.py"
    spec = importlib.util.spec_from_file_location(_NAME, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_NAME] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(_NAME, None)
        raise
    return mod
