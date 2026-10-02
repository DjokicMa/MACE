"""Re-extracting a calculation's properties removes the rows it no longer produces.

save_properties_to_database only UPDATEd or INSERTed the keys the current
extraction produced, so rows written by the 1.1.1 extractor that 1.1.2+ no
longer produces stayed in materials.db: a DOSS run's has_band_structure and
vbm_energy (a band index stored in eV), an OPTGEOM-titled SP run's
optimization_converged, a doped BAND run's band_gap. Re-extraction is what the
CHANGELOG tells users to do, and it did not clean them.

The extractor now replaces its own rows for that (material_id, calc_id). Rows
written by other tools, rows of other calculations, and untracked rows
(calc_id NULL, where several outputs of one material share the key) are kept.
The database is built the way the 1.1.1 extractor left it (synthetic rows).
"""
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT, TEST_DATA
from mace.database.materials import MaterialDatabase

OUT = Path(__file__).resolve().parent / "data" / "charged_runs" / "li_fsi_ec_neutral.out"
DOSS = TEST_DATA / "DOSS" / "1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized_doss.out"
OLD = "crystal_property_extractor.py"

# What the 1.1.1 extractor wrote and 1.1.5 no longer produces for these runs
STALE_111 = [("has_band_structure", 1.0, "", "electronic"),
             ("vbm_energy", 6.0, "eV", "electronic"),
             ("band_start", 4.0, "", "electronic"),
             ("total_bands", 8.0, "", "electronic"),
             ("optimization_converged", 0.0, "", "computational")]


def _db_from_111(tmp_path, calc_id):
    db_path = tmp_path / "materials.db"
    MaterialDatabase(str(db_path))  # current schema (unchanged since 1.1.1)
    rows = [("mat", calc_id, cat, name, val, str(val), unit, "2026-08-01T00:00:00", OLD)
            for name, val, unit, cat in STALE_111]
    rows += [("mat", calc_id, "electronic", "dat_file_BAND", None, "{}", "N/A",
              "2026-08-01T00:00:00", "dat_file_processor.py"),
             ("mat", "other_calc", "electronic", "vbm_energy", 6.0, "6.0", "eV",
              "2026-08-01T00:00:00", OLD)]
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            "INSERT INTO properties (material_id, calc_id, property_category, property_name,"
            " property_value, property_value_text, property_unit, extracted_at,"
            " extractor_script) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    return db_path


def _rows(db_path):
    with sqlite3.connect(db_path) as conn:
        return set(conn.execute(
            "SELECT calc_id, property_name, extractor_script FROM properties"))


def _extract(tmp_path, out, db_path, *extra):
    shutil.copy(out, tmp_path / out.name)
    env = {k: v for k, v in os.environ.items() if k not in ("COLORTERM", "TERM", "NO_COLOR")}
    run = subprocess.run([sys.executable, str(REPO_ROOT / "mace_cli"), "analyze",
                          "--extract-properties", "--output-file", out.name,
                          "--material-id", "mat", *extra, "--db-path", str(db_path)],
                         cwd=tmp_path, stdin=subprocess.DEVNULL, capture_output=True,
                         text=True, timeout=300, env=env)
    assert run.returncode == 0, run.stdout + run.stderr
    return _rows(db_path)


def test_reextraction_removes_the_1_1_1_rows_it_no_longer_produces(tmp_path):
    db_path = _db_from_111(tmp_path, "calc1")
    rows = _extract(tmp_path, OUT, db_path, "--calc-id", "calc1")
    mine = {name for calc, name, script in rows if calc == "calc1" and script == OLD}
    assert mine, "the extraction stored nothing"
    for name, *_ in STALE_111:
        assert name not in mine, name
    # other tools' rows and other calculations are untouched
    assert ("calc1", "dat_file_BAND", "dat_file_processor.py") in rows
    assert ("other_calc", "vbm_energy", OLD) in rows


def test_rows_the_extraction_still_produces_are_updated_in_place(tmp_path):
    db_path = _db_from_111(tmp_path, "calc1")
    first = _extract(tmp_path, OUT, db_path, "--calc-id", "calc1")
    with sqlite3.connect(db_path) as conn:
        ids = conn.execute("SELECT property_name, property_id FROM properties "
                           "WHERE calc_id = 'calc1'").fetchall()
    assert _extract(tmp_path, OUT, db_path, "--calc-id", "calc1") == first
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT property_name, property_id FROM properties "
                            "WHERE calc_id = 'calc1'").fetchall() == ids


def test_untracked_rows_are_kept(tmp_path):
    """With no calc_id, the BAND, DOSS and SP outputs of one material share
    (material_id, NULL); a later extraction cannot tell whose rows they are."""
    db_path = _db_from_111(tmp_path, None)
    rows = _extract(tmp_path, OUT, db_path)
    for name, *_ in STALE_111:
        assert (None, name, OLD) in rows, name


def test_real_doss_reextraction_drops_the_band_fields(tmp_path):
    if not DOSS.exists():
        pytest.skip("test corpus not present")
    db_path = _db_from_111(tmp_path, "calc1")
    rows = _extract(tmp_path, DOSS, db_path, "--calc-id", "calc1")
    mine = {name for calc, name, script in rows if calc == "calc1" and script == OLD}
    assert "has_band_structure" not in mine and "vbm_energy" not in mine
    assert "band_start" not in mine and "total_bands" not in mine
