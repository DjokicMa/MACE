"""`export` in the interactive database explorer.

It passed the last query's materials as export_materials(material_ids=...),
a keyword export_materials did not take, so every export printed
"Export error: ... unexpected keyword argument 'material_ids'".
"""
import csv
import os
import subprocess
import sys
from pathlib import Path

from mace.database.export import export_materials
from mace.database.materials import MaterialDatabase

MACE_CLI = Path(__file__).resolve().parent.parent / "mace_cli"


def _db(tmp_path):
    db = MaterialDatabase(str(tmp_path / "materials.db"))
    for mid, formula in (("mat_a", "MgO"), ("mat_b", "NaCl"), ("mat_c", "Si")):
        db.create_material(material_id=mid, formula=formula)
    return db


def _ids(path):
    with open(path, newline="") as f:
        return [row["material_id"] for row in csv.DictReader(f)]


def test_export_materials_takes_the_material_ids(tmp_path):
    db = _db(tmp_path)
    out = export_materials(db, format="csv", output_file=str(tmp_path / "x.csv"),
                           material_ids=["mat_c", "mat_a"])
    assert sorted(_ids(out)) == ["mat_a", "mat_c"]
    # None still means every material.
    out = export_materials(db, format="csv", output_file=str(tmp_path / "all.csv"))
    assert sorted(_ids(out)) == ["mat_a", "mat_b", "mat_c"]


def test_interactive_export_writes_the_last_results(tmp_path):
    _db(tmp_path)
    env = {k: v for k, v in os.environ.items()
           if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH")}
    env["HOME"] = str(tmp_path)          # the explorer keeps its history in ~
    r = subprocess.run(
        [sys.executable, str(MACE_CLI), "database", "--action", "interactive",
         "--db-path", str(tmp_path / "materials.db")],
        input="list 2\nexport csv picked.csv\nexit\n", cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=120)
    assert "Export error" not in r.stdout + r.stderr, r.stdout + r.stderr
    assert len(_ids(tmp_path / "picked.csv")) == 2
