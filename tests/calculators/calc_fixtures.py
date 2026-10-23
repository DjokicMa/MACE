"""Shared by the mace calc tests: the fixture files and the preopt sandbox
(a private TMPDIR and XDG dirs; no temp directory may be left behind)."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "tests" / "preopt"))
import preopt_fixtures as F  # noqa: E402,F401
from preopt_fixtures import preopt_sandbox  # noqa: E402,F401


def irmof1():
    from ase import Atoms
    d = json.loads((DATA / "irmof1.json").read_text())
    a = Atoms(d["symbols"], cell=d["cell"], scaled_positions=d["frac"], pbc=True)
    return a, d["eqeq_reference"]


def rdkit_reference():
    return json.loads((DATA / "uff_rdkit_reference.json").read_text())["molecules"]
