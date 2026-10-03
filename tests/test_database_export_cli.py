"""`mace database --action export` through the CLI.

HPCC QA (2026-10-03): a plain CSV/JSON export held only the materials table
(the CLI meant 'all properties' by include_properties=None, which the
exporter read as 'none'), and `--format excel` without openpyxl wrote CSV
text into the .xlsx file and reported success.
"""
import csv
import json
import os
import subprocess
import sys

import pytest

from conftest import REPO_ROOT

MACE_CLI = REPO_ROOT / "mace_cli"


@pytest.fixture
def workdir(tmp_path):
    sys.path.insert(0, str(REPO_ROOT))
    from mace.database.materials import MaterialDatabase
    db = MaterialDatabase(str(tmp_path / "materials.db"))
    db.create_material("MgO", "MgO", space_group=225)
    db.store_material_property("MgO", "band_gap", 7.6)
    db.store_material_property("MgO", "total_energy", -275.1)
    return tmp_path


def _env(extra_path=None):
    env = dict(os.environ)
    env["MACE_NO_BANNER"] = "1"
    for var in ("MACE_CONTEXT_DIR", "MACE_WORKFLOW_ID", "MACE_ISOLATION_MODE",
                "MACE_HOME", "PYTHONPATH"):
        env.pop(var, None)
    if extra_path:
        env["PYTHONPATH"] = str(extra_path)
    return env


def _export(cwd, *args, env=None):
    return subprocess.run(
        [sys.executable, str(MACE_CLI), "--no-banner", "database", "--action", "export", *args],
        cwd=str(cwd), env=env or _env(), capture_output=True, text=True, timeout=300)


def test_csv_export_includes_properties(workdir):
    proc = _export(workdir, "--format", "csv", "--output", "mgo.csv")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    with open(workdir / "mgo.csv") as fh:
        (row,) = list(csv.DictReader(fh))
    assert row["formula"] == "MgO"
    assert float(row["band_gap"]) == pytest.approx(7.6)
    assert float(row["total_energy"]) == pytest.approx(-275.1)


def test_json_export_includes_properties(workdir):
    proc = _export(workdir, "--format", "json", "--output", "mgo.json")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    text = (workdir / "mgo.json").read_text()
    assert "band_gap" in text and "total_energy" in text


def test_excel_without_openpyxl_writes_no_fake_xlsx(workdir, tmp_path):
    blocker = tmp_path / "block"
    (blocker / "openpyxl").mkdir(parents=True)
    (blocker / "openpyxl" / "__init__.py").write_text(
        "raise ImportError('openpyxl blocked for this test')\n")
    proc = _export(workdir, "--format", "excel", "--output", "mgo.xlsx",
                   env=_env(blocker))
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0, out
    assert not (workdir / "mgo.xlsx").exists(), out
    assert "openpyxl" in out
