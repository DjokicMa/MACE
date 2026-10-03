"""`mace analyze --extract-properties` files properties under the
calculation that produced the output, in the database the other commands
read.

HPCC QA (2026-10-03): after the completion callback had stored 65 rows for
an SP, `mace analyze --extract-properties . --fragments` added 65 more with
calc_id NULL: the scan passes relative paths ('LiEC.out') and the database
holds resolved ones. In a workflow directory it wrote to ./materials.db
rather than the context database `mace database` and `mace monitor` open,
and it always printed 'Materials processed: 0'.
"""
import os
import shutil
import sqlite3
import subprocess
import sys

import pytest

from conftest import REPO_ROOT, find_data

MACE_CLI = REPO_ROOT / "mace_cli"
STEM = "dia_sp"


@pytest.fixture
def env():
    e = dict(os.environ)
    e["MACE_NO_BANNER"] = "1"
    for var in ("MACE_CONTEXT_DIR", "MACE_WORKFLOW_ID", "MACE_ISOLATION_MODE",
                "MACE_HOME", "PYTHONPATH"):
        e.pop(var, None)
    return e


def _job(work):
    out = find_data("SP/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized.out")
    work.mkdir(parents=True, exist_ok=True)
    shutil.copy(out, work / f"{STEM}.out")
    shutil.copy(out.with_suffix(".d12"), work / f"{STEM}.d12")


def _record(db_path, work):
    sys.path.insert(0, str(REPO_ROOT))
    from mace.database.materials import MaterialDatabase, create_material_id_from_file
    material_id = create_material_id_from_file(f"{STEM}.out")
    db = MaterialDatabase(str(db_path))
    db.create_material(material_id, "C")
    calc_id = db.create_calculation(material_id, "SP", input_file=str(work / f"{STEM}.d12"),
                                    work_dir=str(work.resolve()))
    db.update_calculation_status(calc_id, "completed",
                                 output_file=str((work / f"{STEM}.out").resolve()))
    return calc_id


def _calc_ids(db_path):
    with sqlite3.connect(db_path) as conn:
        return [r[0] for r in conn.execute("SELECT calc_id FROM properties")]


def _analyze(cwd, env, *extra):
    proc = subprocess.run(
        [sys.executable, str(MACE_CLI), "--no-banner", "analyze",
         "--extract-properties", ".", *extra],
        cwd=str(cwd), env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout


def test_relative_scan_links_rows_to_the_calculation(tmp_path, env):
    work = tmp_path / "run"
    _job(work)
    calc_id = _record(work / "materials.db", work)

    out = _analyze(work, env)

    ids = _calc_ids(work / "materials.db")
    assert ids and set(ids) == {calc_id}, set(ids)
    assert "Materials processed: 1" in out, out


def test_in_place_directory_links_by_output_name(tmp_path, env):
    """Several calculations share an in-place directory: the work_dir alone
    does not say which one an output belongs to."""
    work = tmp_path / "run"
    _job(work)
    sys.path.insert(0, str(REPO_ROOT))
    from mace.database.materials import MaterialDatabase
    db = MaterialDatabase(str(work / "materials.db"))
    db.create_material("other", "C")
    other = db.create_calculation("other", "OPT", input_file=str(work / "other_opt.d12"),
                                  work_dir=str(work.resolve()))
    calc_id = _record(work / "materials.db", work)
    # the stored output path differs (e.g. recorded before the job ran)
    with sqlite3.connect(work / "materials.db") as conn:
        conn.execute("UPDATE calculations SET output_file=NULL")

    _analyze(work, env)

    assert set(_calc_ids(work / "materials.db")) == {calc_id}
    assert other != calc_id


def test_analyze_uses_the_workflow_context_database(tmp_path, env):
    ctx_db = tmp_path / ".mace_context_workflow_20261003_162306" / "materials.db"
    ctx_db.parent.mkdir()
    work = tmp_path / "workflow_outputs" / "workflow_20261003_162306" / "step_002_SP" / "dia"
    _job(work)
    calc_id = _record(ctx_db, work)

    _analyze(tmp_path, env)

    assert set(_calc_ids(ctx_db)) == {calc_id}
    assert not (tmp_path / "materials.db").exists()
